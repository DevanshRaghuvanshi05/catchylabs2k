"""
Unit tests for shared Gemini request pacer and SerpAPI guard.
Uses a fake clock and fake client:
- Spacing is enforced.
- Retry delay is honoured.
- A 429 pushes the schedule.
- Call cap counts retries.
- SerpAPI guard blocks a worker-side call at the cap.
- No real network and no real sleeping.
Plain ASCII only.
"""

import os
from unittest.mock import MagicMock, Mock, patch
import pytest

from gemini_pacer import GeminiRequestPacer, pacer, parse_retry_delay
from search_engine import search_web_serpapi
from content_generator import run_content_generation_pipeline


class FakeClock:
    def __init__(self, start: float = 1000.0):
        self.current_time = start
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.current_time

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.current_time += seconds


class DummyResponseWithJson:
    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


class DummyResponseWithHeaders:
    def __init__(self, headers):
        self.headers = headers


def test_spacing_enforced_between_calls():
    """Verify minimum gap (60/RPM + 1.0 margin) is enforced on successive calls."""
    fake_clock = FakeClock(1000.0)
    test_pacer = GeminiRequestPacer(clock=fake_clock.time, sleep_fn=fake_clock.sleep)

    with patch.dict(os.environ, {"GEMINI_RPM": "5"}):
        assert test_pacer.get_min_gap() == 13.0

        # Call 1: Immediate
        w1 = test_pacer.wait_for_turn("gemini-3.5-flash")
        assert w1 == 0.0
        assert fake_clock.current_time == 1000.0

        # Simulate 3 seconds passing before Call 2
        fake_clock.current_time += 3.0

        # Call 2: Must wait remaining 10.0 seconds
        w2 = test_pacer.wait_for_turn("gemini-3.5-flash")
        assert w2 == 10.0
        assert fake_clock.current_time == 1013.0
        assert fake_clock.sleeps == [10.0]


def test_separate_schedule_per_model_and_lite_fallback():
    """Verify primary model schedule does not delay fallback lite model."""
    fake_clock = FakeClock(1000.0)
    test_pacer = GeminiRequestPacer(clock=fake_clock.time, sleep_fn=fake_clock.sleep)

    with patch.dict(os.environ, {"GEMINI_RPM": "5"}):
        # Call primary model: occupies primary schedule until 1013.0
        test_pacer.wait_for_turn("gemini-3.5-flash")
        assert test_pacer.get_next_allowed_time("gemini-3.5-flash") == 1013.0

        # Immediate fallback to lite model: uses its own separate schedule!
        w_lite = test_pacer.wait_for_turn("gemini-3.5-flash-lite")
        assert w_lite == 0.0
        assert fake_clock.current_time == 1000.0
        assert test_pacer.get_next_allowed_time("gemini-3.5-flash-lite") == 1013.0


def test_retry_delay_honoured_and_capped_at_90s():
    """Verify suggested retry delay + 1s margin is honoured and capped at 90s."""
    fake_clock = FakeClock(1000.0)
    test_pacer = GeminiRequestPacer(clock=fake_clock.time, sleep_fn=fake_clock.sleep)

    # 1. Delay from string: "retry in 34s" -> 34 + 1 = 35s
    exc1 = Exception("429 RESOURCE_EXHAUSTED: Please retry in 34s.")
    w1 = test_pacer.handle_429("gemini-3.5-flash", exc1)
    assert w1 == 35.0
    assert fake_clock.current_time == 1035.0

    # 2. Delay from JSON body with retryDelay field -> 15 + 1 = 16s
    exc2 = Exception("429 Too Many Requests")
    exc2.response = DummyResponseWithJson({"error": {"details": [{"retryDelay": "15s"}]}})
    w2 = test_pacer.handle_429("gemini-3.5-flash", exc2)
    assert w2 == 16.0
    assert fake_clock.current_time == 1051.0

    # 3. Delay from header -> 10 + 1 = 11s
    exc3 = Exception("429 Rate limit")
    exc3.response = DummyResponseWithHeaders({"Retry-After": "10"})
    w3 = test_pacer.handle_429("gemini-3.5-flash", exc3)
    assert w3 == 11.0

    # 4. Cap at 90 seconds when suggested delay is large (e.g. 120s)
    exc4 = Exception("429 RESOURCE_EXHAUSTED: retry in 120s")
    w4 = test_pacer.handle_429("gemini-3.5-flash", exc4)
    assert w4 == 90.0


def test_429_pushes_schedule_forward():
    """Verify a 429 pushes the pacer schedule forward for that model."""
    fake_clock = FakeClock(1000.0)
    test_pacer = GeminiRequestPacer(clock=fake_clock.time, sleep_fn=fake_clock.sleep)

    with patch.dict(os.environ, {"GEMINI_RPM": "5"}):
        test_pacer.wait_for_turn("gemini-3.5-flash")
        # Schedule was initially 1013.0
        assert test_pacer.get_next_allowed_time("gemini-3.5-flash") == 1013.0

        # A 429 arrives with retry in 34s -> wait 35s, schedule pushed to 1000 + 35 = 1035.0
        exc = Exception("429 RESOURCE_EXHAUSTED: retry in 34s")
        w = test_pacer.handle_429("gemini-3.5-flash", exc)
        assert w == 35.0
        assert test_pacer.get_next_allowed_time("gemini-3.5-flash") == 1035.0
        assert fake_clock.current_time == 1035.0

        # Retry proceeds immediately at 1035.0
        w_retry = test_pacer.wait_for_turn("gemini-3.5-flash")
        assert w_retry == 0.0
        # Subsequent call pushed to 1035 + 13 = 1048.0
        assert test_pacer.get_next_allowed_time("gemini-3.5-flash") == 1048.0


def test_call_cap_counts_retries():
    """Verify 429 retries count toward the per-run call cap (max_gemini_calls=4)."""
    fake_clock = FakeClock(1000.0)

    # Use global pacer with fake clock
    pacer.reset()
    pacer.set_clock(fake_clock.time, fake_clock.sleep)

    call_count = 0

    def mock_generate_content(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise Exception("429 RESOURCE_EXHAUSTED: retry in 2s")

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = mock_generate_content

    db_data = {
        "run": {"id": "r1", "brand": "Brand", "market": "India", "audience": "Youth", "objective": "Obj", "evidence": []},
        "opps": [{"id": 1, "creative_gap_id": 1, "rank": 1, "overall_score": 8.0, "is_stale": False}],
        "gaps": [{"id": 1, "gap_title": "Gap", "description": "Desc", "supporting_consumer_insight_ids": [1], "supporting_competitor_names": ["C1"], "confidence": "high"}],
        "cis": [{"id": 1, "insight_type": "segment", "insight_text": "Text", "supporting_evidence_ids": []}],
        "comp": {"competitors": [{"name": "C1", "positioning": "P", "strengths": [], "weaknesses": []}], "competitive_gaps": []},
    }

    tracker = MagicMock()
    tracker.__enter__ = Mock(return_value=tracker)
    tracker.__exit__ = Mock(return_value=False)

    try:
        with patch("content_generator.get_run_with_evidence", return_value=db_data["run"]), \
             patch("content_generator.get_opportunity_scores", return_value=db_data["opps"]), \
             patch("content_generator.get_creative_gaps", return_value=db_data["gaps"]), \
             patch("content_generator.get_consumer_insights", return_value=db_data["cis"]), \
             patch("content_generator.get_competitive_intelligence", return_value=db_data["comp"]), \
             patch("content_generator.save_content_pieces", return_value=[]), \
             patch("content_generator.track_api_call", return_value=tracker), \
             patch.dict(os.environ, {"GEMINI_RPM": "60"}):

            with pytest.raises((RuntimeError, ValueError)):
                run_content_generation_pipeline(
                    "r1",
                    creative_gap_id=1,
                    client=mock_client,
                    max_gemini_calls=4
                )
    finally:
        pacer.reset()
        pacer.set_clock(None, None)

    # Exactly 4 calls were made, counting each retry toward the cap
    assert call_count == 4


def test_serpapi_guard_blocks_worker_call_at_cap():
    """Verify search_web_serpapi blocks worker-side call when daily cap is reached."""
    with patch.dict(os.environ, {"MAX_SERPAPI_PER_DAY": "15"}), \
         patch("search_engine.count_serpapi_calls_today", return_value=15), \
         patch("requests.get") as mock_get:

        with pytest.raises(RuntimeError) as exc_info:
            search_web_serpapi("sustainable running shoes")

        assert "Daily SerpAPI limit reached: 15/15" in str(exc_info.value)
        # Verify call was blocked before any HTTP request was made
        mock_get.assert_not_called()


def test_serpapi_guard_allows_call_under_cap():
    """Verify search_web_serpapi allows call when below daily cap."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "organic_results": [
            {"title": "Test Title", "link": "http://test.com", "snippet": "Test snippet"},
            {"title": "Invalid Scheme", "link": "javascript:alert(1)", "snippet": "Ignored"},
            {"title": "Credential URL", "link": "https://user:pass@example.com/", "snippet": "Ignored"},
        ]
    }

    tracker = MagicMock()
    tracker.__enter__ = Mock(return_value=tracker)
    tracker.__exit__ = Mock(return_value=False)

    with patch.dict(os.environ, {"MAX_SERPAPI_PER_DAY": "15"}), \
         patch("search_engine.count_serpapi_calls_today", return_value=14), \
         patch("search_engine.track_api_call", return_value=tracker), \
         patch("requests.get", return_value=mock_resp) as mock_get:

        results = search_web_serpapi("sustainable running shoes", api_key="dummy_key")
        assert len(results) == 1
        assert results[0]["title"] == "Test Title"
        assert results[0]["url"] == "http://test.com"
        mock_get.assert_called_once()
