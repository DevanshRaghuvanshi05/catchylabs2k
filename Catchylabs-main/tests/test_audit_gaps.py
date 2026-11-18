"""
tests/test_audit_gaps.py
========================
Tests for the four gaps found in the final verification audit:

  REQ 3  -- MAX_STAGE_RUNS_PER_DAY and MAX_SERPAPI_PER_DAY DB-counted limits.
  REQ 1g -- DISABLE_DOCS env flag blocks /docs, /openapi.json, /redoc.
  REQ 6c -- Unwritable / missing report directory handled gracefully.

Rules:
  - No live Gemini or SerpAPI calls (all external I/O is mocked).
  - Uses TEST_DATABASE_URL exclusively via the conftest.py isolation layer.
  - All DB writes roll back at the end of each test via the db_session fixture.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_KEY = "test-secret-key-abc123"

_STAGE_ENDPOINTS = [
    "/research/test-run/consumer-intelligence",
    "/research/test-run/competitor-intelligence",
    "/research/test-run/creative-gaps",
    "/research/test-run/opportunities",
    "/research/test-run/content",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_app_env(monkeypatch):
    """Patch auth and quota config for every test in this module."""
    import app as app_module
    monkeypatch.setattr(app_module, "_API_KEY", VALID_KEY)
    monkeypatch.setattr(app_module, "_MAX_RUNS_PER_DAY", 5)
    monkeypatch.setattr(app_module, "_MAX_STAGE_RUNS_PER_DAY", 3)
    monkeypatch.setattr(app_module, "_MAX_SERPAPI_PER_DAY", 2)
    monkeypatch.setattr(app_module, "_DISABLE_DOCS", False)


@pytest.fixture()
def client():
    import app as app_module
    return TestClient(app_module.app, raise_server_exceptions=False)


# ===========================================================================
# REQ 3 -- MAX_STAGE_RUNS_PER_DAY
# ===========================================================================

class TestStageRunQuota:
    """All five POST stage endpoints must return 429 once the daily cap is hit."""

    @pytest.mark.parametrize("path", _STAGE_ENDPOINTS)
    def test_stage_limit_returns_429_at_cap(self, client, monkeypatch, path):
        """count_stage_runs_today == MAX_STAGE_RUNS_PER_DAY => HTTP 429."""
        import app as app_module
        monkeypatch.setattr(app_module, "_MAX_STAGE_RUNS_PER_DAY", 3)
        with patch("app.count_stage_runs_today", return_value=3):
            resp = client.post(path, headers={"X-API-Key": VALID_KEY})
        assert resp.status_code == 429, (
            f"Expected 429 for {path} at cap, got {resp.status_code}: {resp.text}"
        )
        detail = resp.json()["detail"]
        assert "3" in detail, f"Detail must mention count: {detail}"

    @pytest.mark.parametrize("path", _STAGE_ENDPOINTS)
    def test_stage_under_limit_does_not_get_429(self, client, monkeypatch, path):
        """count_stage_runs_today < MAX_STAGE_RUNS_PER_DAY => must NOT return 429."""
        import app as app_module
        monkeypatch.setattr(app_module, "_MAX_STAGE_RUNS_PER_DAY", 5)
        with patch("app.count_stage_runs_today", return_value=0), \
             patch("app.count_serpapi_calls_today", return_value=0):
            resp = client.post(path, headers={"X-API-Key": VALID_KEY})
        assert resp.status_code != 429, (
            f"Got unexpected 429 for {path}: {resp.text}"
        )

    def test_stage_limit_uses_db_not_memory(self, client, monkeypatch):
        """Patching count_stage_runs_today is effective -- proves DB-backed counting."""
        import app as app_module
        monkeypatch.setattr(app_module, "_MAX_STAGE_RUNS_PER_DAY", 1)
        calls = []

        def _fake_count():
            calls.append(1)
            return 99

        with patch("app.count_stage_runs_today", side_effect=_fake_count):
            resp = client.post(
                "/research/test-run/consumer-intelligence",
                headers={"X-API-Key": VALID_KEY},
            )
        assert resp.status_code == 429
        assert len(calls) == 1, "count_stage_runs_today must be called exactly once per request"

    def test_stage_limit_detail_is_informative(self, client, monkeypatch):
        """HTTP 429 body must identify the stage-run limit so callers understand the error."""
        import app as app_module
        monkeypatch.setattr(app_module, "_MAX_STAGE_RUNS_PER_DAY", 3)
        with patch("app.count_stage_runs_today", return_value=3):
            resp = client.post(
                "/research/test-run/consumer-intelligence",
                headers={"X-API-Key": VALID_KEY},
            )
        assert resp.status_code == 429
        detail = resp.json()["detail"].lower()
        assert "stage" in detail or "limit" in detail or "3" in detail

    def test_stage_quota_database_failure_fails_closed(self, client):
        with patch("app.count_stage_runs_today", side_effect=RuntimeError("database unavailable")):
            response = client.post(
                "/research/test-run/consumer-intelligence",
                headers={"X-API-Key": VALID_KEY},
            )

        assert response.status_code == 503
        assert "could not verify" in response.json()["detail"].lower()


# ===========================================================================
# REQ 3 -- MAX_SERPAPI_PER_DAY (competitor-intelligence only)
# ===========================================================================

class TestSerpApiQuota:
    """POST /competitor-intelligence must enforce its own SerpAPI daily cap."""

    def test_serpapi_limit_returns_429_at_cap(self, client, monkeypatch):
        """count_serpapi_calls_today == MAX_SERPAPI_PER_DAY => HTTP 429."""
        import app as app_module
        monkeypatch.setattr(app_module, "_MAX_STAGE_RUNS_PER_DAY", 99)
        monkeypatch.setattr(app_module, "_MAX_SERPAPI_PER_DAY", 2)
        with patch("app.count_stage_runs_today", return_value=0), \
             patch("app.count_serpapi_calls_today", return_value=2):
            resp = client.post(
                "/research/test-run/competitor-intelligence",
                headers={"X-API-Key": VALID_KEY},
            )
        assert resp.status_code == 429, resp.text
        detail = resp.json()["detail"].lower()
        assert "serpapi" in detail or "competitor" in detail or "2" in detail

    def test_serpapi_limit_uses_db_not_memory(self, client, monkeypatch):
        """Patching count_serpapi_calls_today is effective -- proves DB-backed counting."""
        import app as app_module
        monkeypatch.setattr(app_module, "_MAX_STAGE_RUNS_PER_DAY", 99)
        monkeypatch.setattr(app_module, "_MAX_SERPAPI_PER_DAY", 1)
        calls = []

        def _fake_count():
            calls.append(1)
            return 99

        with patch("app.count_stage_runs_today", return_value=0), \
             patch("app.count_serpapi_calls_today", side_effect=_fake_count):
            resp = client.post(
                "/research/test-run/competitor-intelligence",
                headers={"X-API-Key": VALID_KEY},
            )
        assert resp.status_code == 429
        assert len(calls) == 1

    def test_serpapi_cap_not_applied_to_other_stages(self, client, monkeypatch):
        """SerpAPI cap must NOT affect non-competitor-intelligence endpoints."""
        import app as app_module
        monkeypatch.setattr(app_module, "_MAX_STAGE_RUNS_PER_DAY", 99)
        monkeypatch.setattr(app_module, "_MAX_SERPAPI_PER_DAY", 0)
        with patch("app.count_stage_runs_today", return_value=0), \
             patch("app.count_serpapi_calls_today", return_value=99):
            resp = client.post(
                "/research/test-run/consumer-intelligence",
                headers={"X-API-Key": VALID_KEY},
            )
        assert resp.status_code != 429, (
            f"consumer-intelligence incorrectly got 429 due to SerpAPI cap: {resp.text}"
        )

    def test_serpapi_under_limit_does_not_get_429(self, client, monkeypatch):
        """When SerpAPI count is below cap, competitor-intelligence must not get 429."""
        import app as app_module
        monkeypatch.setattr(app_module, "_MAX_STAGE_RUNS_PER_DAY", 99)
        monkeypatch.setattr(app_module, "_MAX_SERPAPI_PER_DAY", 5)
        with patch("app.count_stage_runs_today", return_value=0), \
             patch("app.count_serpapi_calls_today", return_value=0):
            resp = client.post(
                "/research/test-run/competitor-intelligence",
                headers={"X-API-Key": VALID_KEY},
            )
        assert resp.status_code != 429, resp.text


def test_research_quota_database_failure_fails_closed(client):
    brief = {
        "brand": "Test",
        "market": "Test market",
        "audience": "Test audience",
        "objective": "Test objective",
    }

    for path in ("/research", "/research/pipeline"):
        with patch("app.count_research_runs_today", side_effect=RuntimeError("database unavailable")):
            response = client.post(path, headers={"X-API-Key": VALID_KEY}, json=brief)

        assert response.status_code == 503
        assert "could not verify" in response.json()["detail"].lower()


def test_serpapi_quota_database_failure_fails_closed(client, monkeypatch):
    monkeypatch.setattr("app._MAX_STAGE_RUNS_PER_DAY", 99)
    with patch("app.count_stage_runs_today", return_value=0), \
         patch("app.count_serpapi_calls_today", side_effect=RuntimeError("database unavailable")):
        response = client.post(
            "/research/test-run/competitor-intelligence",
            headers={"X-API-Key": VALID_KEY},
        )

    assert response.status_code == 503
    assert "could not verify" in response.json()["detail"].lower()


def test_app_startup_fails_if_database_initialization_fails():
    import app as app_module

    async def enter_lifespan():
        async with app_module.lifespan(app_module.app):
            pass

    with patch("app.init_db", side_effect=RuntimeError("database unavailable")):
        with pytest.raises(RuntimeError, match="database unavailable"):
            asyncio.run(enter_lifespan())


# ===========================================================================
# REQ 3 -- DB-level daily reset and isolation
# ===========================================================================

class TestDailyResetAndIsolation:
    """Quota counters respect UTC midnight boundaries and use the test DB exclusively."""

    def test_count_stage_runs_today_excludes_yesterday(self, db_session):
        """StageRun rows before UTC midnight must not be counted."""
        from database import StageRun, count_stage_runs_today

        yesterday = datetime.now(timezone.utc) - timedelta(days=1, hours=1)
        db_session.add(StageRun(
            stage_name="consumer-intelligence",
            research_run_id="old-run",
            created_at=yesterday,
        ))
        db_session.flush()

        count = count_stage_runs_today()
        assert count == 0, f"Expected 0 (yesterday excluded), got {count}"

    def test_count_stage_runs_today_includes_today(self, db_session):
        """StageRun rows created today (UTC) must be counted."""
        from database import StageRun, count_stage_runs_today

        db_session.add(StageRun(
            stage_name="creative-gaps",
            research_run_id="today-run",
            created_at=datetime.now(timezone.utc),
        ))
        db_session.flush()

        count = count_stage_runs_today()
        assert count >= 1, f"Expected >= 1, got {count}"

    def test_record_stage_run_increments_count(self, db_session):
        """record_stage_run() must persist a row visible to count_stage_runs_today()."""
        from database import record_stage_run, count_stage_runs_today

        before = count_stage_runs_today()
        record_stage_run("opportunities", "test-run-xyz")
        after = count_stage_runs_today()
        assert after == before + 1, (
            f"Expected count to go from {before} to {before+1}, got {after}"
        )

    def test_count_serpapi_excludes_yesterday(self, db_session):
        """APICall rows with provider='serpapi' before UTC midnight must not be counted."""
        from database import APICall, count_serpapi_calls_today

        before = count_serpapi_calls_today()
        yesterday = datetime.now(timezone.utc) - timedelta(days=1, hours=1)
        db_session.add(APICall(
            provider="serpapi",
            call_type="search",
            stage="competitor-intelligence",
            status="success",
            duration_ms=100,
            created_at=yesterday,
        ))
        db_session.flush()

        count = count_serpapi_calls_today()
        assert count == before, f"Expected {before} (yesterday excluded), got {count}"

    def test_count_serpapi_includes_today(self, db_session):
        """APICall rows with provider='serpapi' created today must be counted."""
        from database import APICall, count_serpapi_calls_today

        db_session.add(APICall(
            provider="serpapi",
            call_type="search",
            stage="competitor-intelligence",
            status="success",
            duration_ms=200,
            created_at=datetime.now(timezone.utc),
        ))
        db_session.flush()

        count = count_serpapi_calls_today()
        assert count >= 1, f"Expected >= 1, got {count}"

    def test_count_serpapi_ignores_gemini_calls(self, db_session):
        """APICall rows with provider='gemini' must NOT be counted as SerpAPI calls."""
        from database import APICall, count_serpapi_calls_today

        before = count_serpapi_calls_today()
        db_session.add(APICall(
            provider="gemini",
            model="gemini-3.5-flash",
            call_type="generate_content",
            stage="researcher",
            status="success",
            duration_ms=300,
            created_at=datetime.now(timezone.utc),
        ))
        db_session.flush()

        after = count_serpapi_calls_today()
        assert after == before, (
            f"gemini call incorrectly counted as SerpAPI: before={before}, after={after}"
        )


# ===========================================================================
# REQ 1 gap -- DISABLE_DOCS env flag
# ===========================================================================

class TestDocsProtection:
    """DISABLE_DOCS=1 must block /docs, /openapi.json, /redoc with HTTP 404."""

    def test_docs_accessible_when_flag_off(self, client):
        """/docs returns 200 when _DISABLE_DOCS is False (default)."""
        import app as app_module
        assert app_module._DISABLE_DOCS is False
        assert client.get("/docs").status_code == 200

    def test_openapi_accessible_when_flag_off(self, client):
        """/openapi.json returns 200 when _DISABLE_DOCS is False."""
        import app as app_module
        assert app_module._DISABLE_DOCS is False
        assert client.get("/openapi.json").status_code == 200

    def test_redoc_accessible_when_flag_off(self, client):
        """/redoc returns 200 when _DISABLE_DOCS is False."""
        import app as app_module
        assert app_module._DISABLE_DOCS is False
        assert client.get("/redoc").status_code == 200

    def test_docs_returns_404_when_disabled(self, monkeypatch):
        """With _DISABLE_DOCS=True, /docs returns 404."""
        import app as app_module
        monkeypatch.setattr(app_module, "_DISABLE_DOCS", True)
        c = TestClient(app_module.app, raise_server_exceptions=False)
        resp = c.get("/docs")
        assert resp.status_code == 404, (
            f"Expected 404 when DISABLE_DOCS=True, got {resp.status_code}"
        )
        assert "disabled" in resp.json().get("detail", "").lower()

    def test_openapi_returns_404_when_disabled(self, monkeypatch):
        """With _DISABLE_DOCS=True, /openapi.json returns 404."""
        import app as app_module
        monkeypatch.setattr(app_module, "_DISABLE_DOCS", True)
        c = TestClient(app_module.app, raise_server_exceptions=False)
        assert c.get("/openapi.json").status_code == 404

    def test_redoc_returns_404_when_disabled(self, monkeypatch):
        """With _DISABLE_DOCS=True, /redoc returns 404."""
        import app as app_module
        monkeypatch.setattr(app_module, "_DISABLE_DOCS", True)
        c = TestClient(app_module.app, raise_server_exceptions=False)
        assert c.get("/redoc").status_code == 404

    def test_health_unaffected_by_disable_docs(self, monkeypatch):
        """/health must remain 200 regardless of DISABLE_DOCS."""
        import app as app_module
        monkeypatch.setattr(app_module, "_DISABLE_DOCS", True)
        c = TestClient(app_module.app, raise_server_exceptions=False)
        assert c.get("/health").status_code == 200

    def test_authenticated_route_returns_401_not_404_when_docs_disabled(self, monkeypatch):
        """Authenticated routes must still get 401 (not 404) when DISABLE_DOCS=True."""
        import app as app_module
        monkeypatch.setattr(app_module, "_DISABLE_DOCS", True)
        c = TestClient(app_module.app, raise_server_exceptions=False)
        assert c.get("/quota-status").status_code == 401


# ===========================================================================
# REQ 6c -- Unwritable / missing report directory
# ===========================================================================

class TestUnwritableReportDirectory:
    """run_research_job falls back gracefully when the report file cannot be read."""

    def test_missing_report_file_falls_back_to_in_memory(self):
        """
        When the pipeline returns a report_file path that does not exist on disk
        (simulating an unwritable directory), run_research_job must complete
        without raising and must persist the in-memory report_data instead.
        """
        import jobs

        fake_brief = {
            "brand": "AuditBrand",
            "market": "US",
            "audience": "All",
            "objective": "Audit test",
        }
        fake_pipeline_result = {
            "report_file": "/nonexistent/locked-dir/report.json",
            "report_data": {"brand": "AuditBrand", "summary": "in-memory-data"},
            "research_run_id": "audit-run-001",
            "queries": ["q1"],
        }

        with patch("jobs.run_research_pipeline", return_value=fake_pipeline_result), \
             patch("jobs.save_research_run", return_value="audit-run-001") as mock_save:
            result = jobs.run_research_job(fake_brief)

        assert result["research_run_id"] == "audit-run-001"
        saved_report = mock_save.call_args[0][1]
        assert saved_report.get("summary") == "in-memory-data", (
            f"Expected in-memory fallback data to be saved, got: {saved_report}"
        )

    def test_ioerror_on_open_falls_back_to_in_memory(self, tmp_path):
        """
        When report_file exists but open() raises OSError (permission denied),
        run_research_job must fall back to in-memory data without raising.
        """
        import jobs

        report_path = tmp_path / "report.json"
        report_path.write_text("{}")

        fake_brief = {
            "brand": "AuditBrand",
            "market": "US",
            "audience": "All",
            "objective": "Audit test",
        }
        fake_pipeline_result = {
            "report_file": str(report_path),
            "report_data": {"brand": "AuditBrand", "summary": "in-memory-fallback"},
            "research_run_id": "audit-run-002",
            "queries": [],
        }

        with patch("jobs.run_research_pipeline", return_value=fake_pipeline_result), \
             patch("jobs.save_research_run", return_value="audit-run-002") as mock_save, \
             patch("builtins.open", side_effect=OSError("Permission denied")):
            result = jobs.run_research_job(fake_brief)

        assert result["research_run_id"] == "audit-run-002"
        saved_report = mock_save.call_args[0][1]
        assert saved_report.get("summary") == "in-memory-fallback", (
            f"Expected in-memory fallback after OSError, got: {saved_report}"
        )

    def test_warning_logged_for_missing_report_file(self, caplog):
        """
        A WARNING-level log message must be emitted when the report file is absent,
        so the unwritable-directory condition is visible in logs.
        """
        import jobs

        fake_brief = {
            "brand": "AuditBrand",
            "market": "US",
            "audience": "All",
            "objective": "Audit test",
        }
        fake_pipeline_result = {
            "report_file": "/does/not/exist/report.json",
            "report_data": {"ok": True},
            "research_run_id": "audit-run-003",
            "queries": [],
        }

        with patch("jobs.run_research_pipeline", return_value=fake_pipeline_result), \
             patch("jobs.save_research_run", return_value="audit-run-003"), \
             caplog.at_level(logging.WARNING, logger="CatchyLabResearcher"):
            jobs.run_research_job(fake_brief)

        warning_texts = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(
            "not written" in msg or "read-only" in msg or "report" in msg.lower()
            for msg in warning_texts
        ), f"Expected a warning about unwritable/missing report file. Got: {warning_texts}"
