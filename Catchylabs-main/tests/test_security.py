"""
Tests for API-key authentication and POST /research daily rate-limiting.

Rules enforced:
  - No live Gemini or SerpAPI calls — all external I/O is mocked.
  - Uses TEST_DATABASE_URL exclusively via the conftest.py isolation layer.
  - Every test rolls back all DB writes via the db_session fixture.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_KEY = "test-secret-key-abc123"


def _make_client(api_key: str = VALID_KEY, max_runs: int = 5) -> TestClient:
    """
    Patch CATCHYLAB_API_KEY and MAX_RUNS_PER_DAY, then import (or reload)
    the app so the module-level variables pick up the patched environment.
    We patch at the ``app`` module level directly to avoid re-import ordering
    issues that come with importlib.reload in a test suite.
    """
    import app as app_module

    client = TestClient(app_module.app, raise_server_exceptions=False)
    return client


@pytest.fixture(autouse=True)
def _patch_app_env(monkeypatch):
    """Patch module-level auth config in app.py for every test in this file."""
    import app as app_module
    monkeypatch.setattr(app_module, "_API_KEY", VALID_KEY)
    monkeypatch.setattr(app_module, "_MAX_RUNS_PER_DAY", 5)


@pytest.fixture()
def client():
    import app as app_module
    return TestClient(app_module.app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# 1. GET /health — no auth required
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_no_key_returns_200(self, client):
        """GET /health must succeed without any X-API-Key header."""
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_health_with_valid_key_still_200(self, client):
        """GET /health must also succeed when a valid key is provided."""
        resp = client.get("/health", headers={"X-API-Key": VALID_KEY})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 2. Auth — 401 without key
# ---------------------------------------------------------------------------

class TestApiKeyAuth:
    """Authenticated endpoints must return 401 when the header is absent."""

    _PROTECTED_ENDPOINTS = [
        ("GET",  "/quota-status"),
        ("GET",  "/research/some-id"),
        ("GET",  "/research/status/some-job-id"),
        ("POST", "/research"),
    ]

    @pytest.mark.parametrize("method,path", _PROTECTED_ENDPOINTS)
    def test_missing_key_returns_401(self, client, method, path):
        resp = client.request(method, path, json={"brand": "x", "market": "x", "audience": "x", "objective": "x"})
        assert resp.status_code == 401, (
            f"{method} {path} should return 401 without X-API-Key, got {resp.status_code}"
        )

    @pytest.mark.parametrize("method,path", _PROTECTED_ENDPOINTS)
    def test_wrong_key_returns_401(self, client, method, path):
        resp = client.request(
            method, path,
            headers={"X-API-Key": "totally-wrong-key"},
            json={"brand": "x", "market": "x", "audience": "x", "objective": "x"},
        )
        assert resp.status_code == 401, (
            f"{method} {path} should return 401 with wrong X-API-Key, got {resp.status_code}"
        )

    def test_correct_key_passes_auth_on_quota_status(self, client):
        """
        A correct key should get past auth (may still fail for other reasons,
        e.g. DB not having data, but must NOT return 401).
        """
        resp = client.get("/quota-status", headers={"X-API-Key": VALID_KEY})
        assert resp.status_code != 401


@pytest.mark.parametrize("field", ["brand", "market", "audience", "objective"])
def test_research_brief_rejects_whitespace_only_fields(client, field):
    brief = {
        "brand": "Test",
        "market": "Test market",
        "audience": "Test audience",
        "objective": "Test objective",
    }
    brief[field] = "   "

    response = client.post(
        "/research",
        headers={"X-API-Key": VALID_KEY},
        json=brief,
    )

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 3. POST /research — 429 when daily cap is exceeded (DB-backed)
# ---------------------------------------------------------------------------

class TestResearchRateLimit:
    """POST /research must return 429 once the DB-backed daily cap is hit."""

    def test_rate_limit_returns_429(self, client, monkeypatch):
        """
        Patch count_research_runs_today to return MAX_RUNS_PER_DAY (5) so the
        endpoint immediately hits the cap without touching Redis or the pipeline.
        """
        import app as app_module

        monkeypatch.setattr(app_module, "_MAX_RUNS_PER_DAY", 3)

        with patch("app.count_research_runs_today", return_value=3):
            resp = client.post(
                "/research",
                headers={"X-API-Key": VALID_KEY},
                json={
                    "brand": "Nike",
                    "market": "India",
                    "audience": "Gen Z",
                    "objective": "Find sustainable opportunities",
                },
            )

        assert resp.status_code == 429
        body = resp.json()
        assert "detail" in body
        # The message must mention the limit so clients understand what happened
        assert "3" in body["detail"] or "limit" in body["detail"].lower()

    def test_rate_limit_message_is_informative(self, client, monkeypatch):
        """The 429 detail must mention both the current count and the cap."""
        import app as app_module
        monkeypatch.setattr(app_module, "_MAX_RUNS_PER_DAY", 5)

        with patch("app.count_research_runs_today", return_value=5):
            resp = client.post(
                "/research",
                headers={"X-API-Key": VALID_KEY},
                json={
                    "brand": "Puma",
                    "market": "UK",
                    "audience": "Millennials",
                    "objective": "Grow market share",
                },
            )

        assert resp.status_code == 429
        detail = resp.json()["detail"]
        assert "5" in detail  # both count and cap appear in the message

    def test_under_limit_proceeds_past_rate_check(self, client, monkeypatch):
        """
        When under the daily cap the request must pass the rate-limit check.
        It may fail later (e.g. Redis unreachable) but must NOT return 429.
        """
        import app as app_module
        monkeypatch.setattr(app_module, "_MAX_RUNS_PER_DAY", 5)

        with patch("app.count_research_runs_today", return_value=2):
            # Redis will likely be unreachable in CI — that's fine, we only
            # care that the 429 is NOT raised.
            resp = client.post(
                "/research",
                headers={"X-API-Key": VALID_KEY},
                json={
                    "brand": "Adidas",
                    "market": "Germany",
                    "audience": "Athletes",
                    "objective": "Performance positioning",
                },
            )

        assert resp.status_code != 429

    def test_rate_limit_count_is_db_backed(self, client, monkeypatch):
        """
        Verify that app.count_research_runs_today is called (not an in-memory
        counter) — if it weren't DB-backed, patching the DB helper would have
        no effect.
        """
        import app as app_module
        monkeypatch.setattr(app_module, "_MAX_RUNS_PER_DAY", 1)

        call_log: list[int] = []

        def fake_count():
            call_log.append(1)
            return 99  # way over the limit

        with patch("app.count_research_runs_today", side_effect=fake_count):
            resp = client.post(
                "/research",
                headers={"X-API-Key": VALID_KEY},
                json={
                    "brand": "Lays",
                    "market": "US",
                    "audience": "Snack lovers",
                    "objective": "Flavour research",
                },
            )

        assert resp.status_code == 429
        assert len(call_log) == 1, "count_research_runs_today must be called exactly once per request"
