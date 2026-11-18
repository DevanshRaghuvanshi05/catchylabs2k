"""
Tests for End-to-End Research Pipeline Orchestrator.
---------------------------------------------------
Verifies:
1. run_full_pipeline_job executes all 6 stages sequentially when each succeeds.
2. run_full_pipeline_job halts immediately when a stage fails and records the failure.
3. POST /research/pipeline endpoint enqueues the orchestrator job and returns job_id.
4. DEMO_MODE=1 blocks POST /research/pipeline with 403 Forbidden.
5. GET /research/status/{job_id} exposes stage-level progress metadata.

Rules enforced:
- Zero live Gemini or SerpAPI calls.
- Purely mocked unit/contract tests.
"""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from jobs import run_full_pipeline_job
import app as app_module

VALID_KEY = "test-secret-key-abc123"


@pytest.fixture(autouse=True)
def _patch_app_env(monkeypatch):
    """Ensure clean test environment with valid API key and demo mode off for endpoint tests."""
    monkeypatch.setattr(app_module, "_API_KEY", VALID_KEY)
    monkeypatch.setattr(app_module, "_MAX_RUNS_PER_DAY", 100)
    monkeypatch.setattr(app_module, "_DEMO_MODE", False)
    monkeypatch.delenv("DEMO_MODE", raising=False)


@pytest.fixture()
def client():
    return TestClient(app_module.app, raise_server_exceptions=False)


def test_orchestrator_success_runs_all_six_stages():
    """Verify run_full_pipeline_job executes all 6 stages and passes run_id downstream."""
    brief = {
        "brand": "Nike",
        "market": "India",
        "audience": "Gen Z",
        "objective": "Sustainable running shoes",
    }
    mock_run_id = "test-run-12345"

    with patch("jobs.run_research_job") as m1, \
         patch("jobs.run_consumer_intelligence_job") as m2, \
         patch("jobs.run_competitive_intelligence_job") as m3, \
         patch("jobs.run_creative_gaps_job") as m4, \
         patch("jobs.run_opportunity_scoring_job") as m5, \
         patch("jobs.run_content_generation_job") as m6:

        m1.return_value = {"job_type": "research", "research_run_id": mock_run_id, "evidence_count": 8}
        m2.return_value = {"job_type": "consumer_intelligence", "research_run_id": mock_run_id, "total_insights": 4}
        m3.return_value = {"job_type": "competitive_intelligence", "research_run_id": mock_run_id, "competitors": ["Puma"]}
        m4.return_value = {"job_type": "creative_gaps", "research_run_id": mock_run_id, "total_gaps": 3}
        m5.return_value = {"job_type": "opportunity_scoring", "research_run_id": mock_run_id, "total_opportunities": 3}
        m6.return_value = {"job_type": "content_generation", "research_run_id": mock_run_id, "total_pieces": 4}

        result = run_full_pipeline_job(brief)

        assert result["job_type"] == "full_pipeline"
        assert result["research_run_id"] == mock_run_id
        for stage in ("research", "consumer_intelligence", "competitor_intelligence", "creative_gaps", "opportunity_scoring", "content_generation"):
            assert result["stages"][stage] == "completed"

        # Verify downstream stages received the generated run_id
        m1.assert_called_once_with(brief)
        m2.assert_called_once_with(mock_run_id)
        m3.assert_called_once_with(mock_run_id)
        m4.assert_called_once_with(mock_run_id)
        m5.assert_called_once_with(mock_run_id)
        m6.assert_called_once_with(mock_run_id, platform="instagram")


def test_orchestrator_halts_on_stage_failure():
    """Verify run_full_pipeline_job halts immediately if an intermediate stage fails."""
    brief = {"brand": "Puma", "market": "US", "audience": "Athletes", "objective": "Trail shoes"}
    mock_run_id = "test-run-halt"

    with patch("jobs.run_research_job") as m1, \
         patch("jobs.run_consumer_intelligence_job") as m2, \
         patch("jobs.run_competitive_intelligence_job") as m3, \
         patch("jobs.run_creative_gaps_job") as m4, \
         patch("jobs.run_opportunity_scoring_job") as m5, \
         patch("jobs.run_content_generation_job") as m6:

        m1.return_value = {"job_type": "research", "research_run_id": mock_run_id}
        # Simulate failure in Stage 2 (Consumer Intelligence)
        m2.side_effect = RuntimeError("Gemini quota exhausted")

        with pytest.raises(RuntimeError, match="Gemini quota exhausted"):
            run_full_pipeline_job(brief)

        # Stage 1 succeeded, Stage 2 was attempted
        m1.assert_called_once_with(brief)
        m2.assert_called_once_with(mock_run_id)
        # Stages 3, 4, 5, 6 MUST NOT be called
        m3.assert_not_called()
        m4.assert_not_called()
        m5.assert_not_called()
        m6.assert_not_called()


def test_post_research_pipeline_endpoint(client):
    """Verify POST /research/pipeline enqueues job and returns job_id."""
    payload = {
        "brand": "Adidas",
        "market": "Germany",
        "audience": "Marathoners",
        "objective": "Energy return soles",
    }
    headers = {"X-API-Key": VALID_KEY}

    mock_job = MagicMock()
    mock_job.id = "mock-pipeline-job-456"

    with patch("app.get_redis_connection") as mock_get_redis, \
         patch("app.Queue") as mock_queue_cls:
        mock_redis = MagicMock()
        mock_get_redis.return_value = mock_redis
        mock_queue = MagicMock()
        mock_queue_cls.return_value = mock_queue
        mock_queue.enqueue.return_value = mock_job

        resp = client.post("/research/pipeline", json=payload, headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == "mock-pipeline-job-456"
        assert data["status"] == "queued"
        assert data["job_type"] == "full_pipeline"


def test_post_research_pipeline_requires_auth(client):
    """Verify POST /research/pipeline returns 401 without X-API-Key."""
    payload = {"brand": "Adidas", "market": "Germany", "audience": "All", "objective": "Test"}
    resp = client.post("/research/pipeline", json=payload)
    assert resp.status_code == 401


def test_post_research_pipeline_blocked_by_demo_mode(client, monkeypatch):
    """Verify DEMO_MODE=1 returns 403 Forbidden for POST /research/pipeline."""
    monkeypatch.setattr(app_module, "_DEMO_MODE", True)
    monkeypatch.setenv("DEMO_MODE", "1")

    payload = {"brand": "Adidas", "market": "Germany", "audience": "All", "objective": "Test"}
    headers = {"X-API-Key": VALID_KEY}

    resp = client.post("/research/pipeline", json=payload, headers=headers)
    assert resp.status_code == 403
    assert resp.json()["detail"] == "Disabled in demo mode"


def test_get_job_status_exposes_stages(client):
    """Verify GET /research/status/{job_id} returns stage progress for pipeline jobs."""
    headers = {"X-API-Key": VALID_KEY}
    mock_job = MagicMock()
    mock_job.get_status.return_value = "started"
    mock_job.meta = {
        "job_type": "full_pipeline",
        "current_stage": "consumer_intelligence",
        "research_run_id": "run-xyz-789",
        "stages": {
            "research": "completed",
            "consumer_intelligence": "running",
            "competitor_intelligence": "pending",
            "creative_gaps": "pending",
            "opportunity_scoring": "pending",
            "content_generation": "pending",
        },
    }

    with patch("app.get_redis_connection") as mock_get_redis, \
         patch("app.Job.fetch") as mock_job_fetch:
        mock_get_redis.return_value = MagicMock()
        mock_job_fetch.return_value = mock_job

        resp = client.get("/research/status/pipeline-job-123", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "started"
        assert data["job_type"] == "full_pipeline"
        assert data["current_stage"] == "consumer_intelligence"
        assert data["research_run_id"] == "run-xyz-789"
        assert data["stages"]["research"] == "completed"
        assert data["stages"]["consumer_intelligence"] == "running"
