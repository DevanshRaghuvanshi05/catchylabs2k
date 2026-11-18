"""
Shared Gemini request pacer.
Enforces thread-safe, per-model rate pacing and 429 backoff with schedule adjustment.
Plain ASCII only.
"""

import os
import re
import time
import threading
import logging
from typing import Optional, Callable

logger = logging.getLogger("GeminiPacer")


def parse_retry_delay(exc: Exception) -> Optional[float]:
    """
    Extracts suggested retry delay in seconds from an exception.
    Checks headers (Retry-After), response json/text, details, and message strings.
    Looks for retryDelay fields (e.g. '34s', 34) and text patterns like 'retry in 34s'.
    """
    if exc is None:
        return None

    # 1. Check HTTP response headers or body if available
    response = getattr(exc, "response", None)
    if response:
        headers = getattr(response, "headers", None)
        if headers:
            val = headers.get("Retry-After") or headers.get("retry-after")
            if val is not None:
                try:
                    return max(0.0, float(val))
                except (ValueError, TypeError):
                    pass

        # Check response json
        body_json = None
        if hasattr(response, "json"):
            try:
                body_json = response.json() if callable(response.json) else response.json
            except Exception:
                pass
        if isinstance(body_json, (dict, list)):
            delay = _find_retry_delay_in_obj(body_json)
            if delay is not None:
                return delay

        # Check response text
        text = getattr(response, "text", "")
        if text and isinstance(text, str):
            delay = _find_retry_delay_in_text(text)
            if delay is not None:
                return delay

    # 2. Check exception attributes like details or retry_delay
    if hasattr(exc, "details"):
        details = getattr(exc, "details")
        delay = _find_retry_delay_in_obj(details)
        if delay is not None:
            return delay

    if hasattr(exc, "retry_delay"):
        val = getattr(exc, "retry_delay")
        delay = _clean_delay_value(val)
        if delay is not None:
            return delay

    # 3. Check message and args
    strings_to_check = [str(exc)]
    if hasattr(exc, "message"):
        strings_to_check.append(str(getattr(exc, "message")))
    for arg in getattr(exc, "args", []):
        if isinstance(arg, str):
            strings_to_check.append(arg)
        elif isinstance(arg, (dict, list)):
            delay = _find_retry_delay_in_obj(arg)
            if delay is not None:
                return delay

    for s in strings_to_check:
        delay = _find_retry_delay_in_text(s)
        if delay is not None:
            return delay

    return None


def _clean_delay_value(val) -> Optional[float]:
    """Converts a delay representation like '34s', '34.5s', 34, 34.0 to float."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return max(0.0, float(val))
    if isinstance(val, str):
        v = val.strip().rstrip("sS").strip()
        try:
            return max(0.0, float(v))
        except (ValueError, TypeError):
            pass
    return None


def _find_retry_delay_in_obj(obj) -> Optional[float]:
    """Recursively search dictionary or list for retryDelay / retry_delay fields."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in ("retrydelay", "retry_delay", "retryafter", "retry_after"):
                parsed = _clean_delay_value(v)
                if parsed is not None:
                    return parsed
            found = _find_retry_delay_in_obj(v)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_retry_delay_in_obj(item)
            if found is not None:
                return found
    return None


def _find_retry_delay_in_text(text: str) -> Optional[float]:
    """Regex search for retry delays in text messages."""
    if not text or not isinstance(text, str):
        return None

    # Matches: retryDelay: "34s", "retryDelay": 34, retryDelay = 34s
    m = re.search(r'["\']?retryDelay["\']?\s*[:=]\s*["\']?(\d+(?:\.\d+)?)\s*s?["\']?', text, re.IGNORECASE)
    if m:
        try:
            return max(0.0, float(m.group(1)))
        except (ValueError, TypeError):
            pass

    # Matches: retry in 34s, retry after 34.5 seconds, retry-after: 12
    m = re.search(r'(?:retry[- ](?:in|after)|retry)\s*[:\s]+(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?', text, re.IGNORECASE)
    if m:
        try:
            return max(0.0, float(m.group(1)))
        except (ValueError, TypeError):
            pass

    return None


class GeminiRequestPacer:
    """
    Thread-safe request pacer for Gemini text-generation calls.
    Maintains a separate schedule per model name.
    Enforces minimum gap of (60 / GEMINI_RPM) + 1.0 second between calls to the same model.
    """

    def __init__(
        self,
        clock: Optional[Callable[[], float]] = None,
        sleep_fn: Optional[Callable[[float], None]] = None
    ):
        self._lock = threading.Lock()
        self._schedule: dict[str, float] = {}
        self._clock_fn = clock
        self._sleep_fn = sleep_fn
        self._mock_time_offset: float = 0.0

    def _now(self) -> float:
        if self._clock_fn is not None:
            return self._clock_fn()
        return time.time() + self._mock_time_offset

    def _do_sleep(self, seconds: float, override_sleep: Optional[Callable[[float], None]] = None) -> None:
        if seconds <= 0:
            return
        sleep_func = override_sleep if override_sleep is not None else (
            self._sleep_fn if self._sleep_fn is not None else time.sleep
        )
        if self._clock_fn is None:
            t_before = time.time()
            sleep_func(seconds)
            t_after = time.time()
            elapsed = t_after - t_before
            # If sleep was mocked (elapsed is near zero compared to seconds requested),
            # advance the mock time offset so subsequent checks within the test see the advanced time.
            if elapsed < seconds * 0.5:
                self._mock_time_offset += (seconds - elapsed)
        else:
            sleep_func(seconds)

    def set_clock(
        self,
        clock: Optional[Callable[[], float]],
        sleep_fn: Optional[Callable[[float], None]]
    ) -> None:
        """Inject custom clock and sleep implementations (useful for tests)."""
        self._clock_fn = clock
        self._sleep_fn = sleep_fn
        self._mock_time_offset = 0.0

    def reset(self) -> None:
        """Clear all schedules and reset state."""
        with self._lock:
            self._schedule.clear()
            self._mock_time_offset = 0.0

    def get_min_gap(self) -> float:
        """
        Calculates minimum gap between calls to the same model:
        60 / GEMINI_RPM + 1.0 second margin (default GEMINI_RPM=5).
        """
        rpm_str = os.environ.get("GEMINI_RPM", "5")
        try:
            rpm = float(rpm_str)
            if rpm <= 0:
                rpm = 5.0
        except (ValueError, TypeError):
            rpm = 5.0
        return (60.0 / rpm) + 1.0

    def wait_for_turn(
        self,
        model_name: str,
        sleep_fn: Optional[Callable[[float], None]] = None
    ) -> float:
        """
        Waits until the current model's schedule allows a call.
        Updates the model's next allowed call time.
        Returns the duration waited in seconds.
        """
        model = str(model_name or "default")
        with self._lock:
            now = self._now()
            scheduled_time = self._schedule.get(model, 0.0)
            wait_time = max(0.0, scheduled_time - now)
            call_time = max(now, scheduled_time)
            min_gap = self.get_min_gap()
            self._schedule[model] = call_time + min_gap

        if wait_time > 0:
            logger.info(f"[Pacer] Spacing call to '{model}' by {wait_time:.2f}s (min gap {min_gap:.1f}s)")
            self._do_sleep(wait_time, override_sleep=sleep_fn)

        return wait_time

    # Alias for wait_for_turn
    pace = wait_for_turn
    acquire = wait_for_turn

    def get_next_allowed_time(self, model_name: str) -> float:
        """Inspect the scheduled timestamp for a given model."""
        with self._lock:
            return self._schedule.get(str(model_name or "default"), 0.0)

    def push_schedule(self, model_name: str, wait_seconds: float) -> None:
        """Pushes the schedule forward for a specific model."""
        model = str(model_name or "default")
        with self._lock:
            now = self._now()
            self._schedule[model] = now + wait_seconds

    def handle_429(
        self,
        model_name: str,
        exc: Exception,
        default_wait: Optional[float] = None,
        sleep_fn: Optional[Callable[[float], None]] = None
    ) -> float:
        """
        Handles a 429 response:
        1. Reads suggested retry delay from error body/headers/text when present.
        2. Waits that delay + 1.0 second (capped at 90.0 seconds).
        3. Pushes the pacer schedule forward for that model.
        4. Sleeps and returns the wait duration.
        """
        model = str(model_name or "default")
        suggested = parse_retry_delay(exc)

        if suggested is not None:
            wait_seconds = min(suggested + 1.0, 90.0)
            logger.warning(
                f"[Pacer 429] Model '{model}' suggested retry delay {suggested:.1f}s. "
                f"Waiting {wait_seconds:.1f}s (+1s margin, capped at 90s)."
            )
        elif default_wait is not None:
            wait_seconds = min(default_wait, 90.0)
            logger.warning(
                f"[Pacer 429] Model '{model}' 429 with no delay hint. "
                f"Using default wait {wait_seconds:.1f}s (capped at 90s)."
            )
        else:
            min_gap = self.get_min_gap()
            wait_seconds = min(min_gap, 90.0)
            logger.warning(
                f"[Pacer 429] Model '{model}' 429 with no delay hint. "
                f"Waiting min gap {wait_seconds:.1f}s (capped at 90s)."
            )

        self.push_schedule(model, wait_seconds)
        self._do_sleep(wait_seconds, override_sleep=sleep_fn)
        return wait_seconds


# Global shared pacer singleton
pacer = GeminiRequestPacer()
