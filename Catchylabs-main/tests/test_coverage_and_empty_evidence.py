"""
test_coverage_and_empty_evidence.py
Tests for coverage check failure handling, empty evidence validation,
and pipeline halt before downstream stages.
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from google.genai import types

from evidence_extractor import check_evidence_coverage
import pipeline
import jobs
import agent2


# ---------------------------------------------------------------------------
# 1. Coverage Check Failure Handling Tests
# ---------------------------------------------------------------------------

def test_coverage_check_429_returns_failed_and_not_sufficient():
    """Verify coverage check on 429 returns sufficient=False and status=failed."""
    client = MagicMock()
    # Hard quota exhaustion error message
    client.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED: You exceeded your current quota")

    brief = {"brand": "Pringles", "market": "India", "audience": "Gen Z", "objective": "Market gap"}
    evidence_items = [{"claim": "Sample claim", "topic": "trends"}]

    with patch("time.sleep"):
        res = check_evidence_coverage(client, brief, evidence_items, max_retries=2)

    assert res["sufficient"] is False
    assert res["status"] == "failed"
    assert res["evaluated"] is False
    assert res["gaps"] == []
    assert res["additional_queries"] == []
    assert "quota" in res["error"].lower() or "rate limit" in res["error"].lower()
    # Hard quota exhaustion stops immediately without cascading
    assert client.models.generate_content.call_count == 1


def test_coverage_check_exhausted_retries_transient_rate_limit():
    """Verify transient rate limit retries boundedly and returns status=failed and sufficient=False."""
    client = MagicMock()
    client.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED: Rate limit exceeded")

    brief = {"brand": "Pringles", "market": "India"}
    evidence_items = [{"claim": "Sample claim", "topic": "trends"}]

    with patch("time.sleep"):
        res = check_evidence_coverage(client, brief, evidence_items, max_retries=2)

    assert res["sufficient"] is False
    assert res["status"] == "failed"
    assert res["evaluated"] is False
    assert res["gaps"] == []
    assert res["additional_queries"] == []
    # 2 attempts on model, then stops without hitting fallback models
    assert client.models.generate_content.call_count == 2


def test_coverage_check_exhausted_retries_transient_error():
    """Verify exhausted retries on 503 transient error across fallback models returns status=failed."""
    client = MagicMock()
    client.models.generate_content.side_effect = Exception("503 UNAVAILABLE: Model is overloaded")

    brief = {"brand": "Nike", "market": "US"}
    evidence_items = [{"claim": "Running trend", "topic": "fitness"}]

    with patch("time.sleep"):
        res = check_evidence_coverage(client, brief, evidence_items, max_retries=2)

    assert res["sufficient"] is False
    assert res["status"] == "failed"
    assert res["evaluated"] is False
    assert res["gaps"] == []
    assert res["additional_queries"] == []
    # Retries 2 attempts on primary + 2 attempts on fallback model = 4
    assert client.models.generate_content.call_count == 4


def test_coverage_check_non_rate_limit_api_failure():
    """Verify non-rate-limit unexpected exception returns status=failed and sufficient=False."""
    client = MagicMock()
    client.models.generate_content.side_effect = RuntimeError("Fatal connection error")

    brief = {"brand": "Puma", "market": "UK"}
    evidence_items = []

    with patch("time.sleep"):
        res = check_evidence_coverage(client, brief, evidence_items, max_retries=2)

    assert res["sufficient"] is False
    assert res["status"] == "failed"
    assert res["evaluated"] is False
    assert res["gaps"] == []
    assert res["additional_queries"] == []
    assert "Fatal connection error" in res["error"]


def test_coverage_check_successful_evaluation():
    """Verify successful coverage check returns status=success, evaluated=True, and valid fields."""
    client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = json.dumps({
        "sufficient": False,
        "gaps": ["Missing pricing data"],
        "additional_queries": ["Pringles India price comparison"]
    })
    mock_resp.usage_metadata = None
    client.models.generate_content.return_value = mock_resp

    brief = {"brand": "Pringles", "market": "India"}
    evidence_items = [{"claim": "Existing claim", "topic": "flavor"}]

    res = check_evidence_coverage(client, brief, evidence_items, max_retries=2)

    assert res["status"] == "success"
    assert res["evaluated"] is True
    assert res["sufficient"] is False
    assert res["gaps"] == ["Missing pricing data"]
    assert res["additional_queries"] == ["Pringles India price comparison"]
    assert res["error"] is None


# ---------------------------------------------------------------------------
# 2. Empty-Evidence Validation Tests
# ---------------------------------------------------------------------------

def test_run_research_pipeline_raises_on_zero_evidence_and_preserves_artifact(tmp_path):
    """Verify run_research_pipeline raises ValueError and writes failed report artifact when evidence is empty."""
    brief = {
        "brand": "Pringles",
        "market": "India",
        "audience": "Gen Z",
        "objective": "Market positioning gaps"
    }

    mock_client = MagicMock()
    # Stage 2 query generation returns 2 queries
    mock_queries = ["query 1", "query 2"]
    # Stage 3 search returns 2 docs
    mock_docs = [
        {"title": "Doc 1", "url": "https://example.com/1", "snippet": "Snippet 1", "query": "query 1"},
        {"title": "Doc 2", "url": "https://example.com/2", "snippet": "Snippet 2", "query": "query 2"},
    ]

    with patch("pipeline.genai.Client", return_value=mock_client), \
         patch("pipeline.init_research_run"), \
         patch("pipeline.perform_semantic_recall", return_value=[]), \
         patch("pipeline.generate_queries", return_value=mock_queries), \
         patch("pipeline.search_web_serpapi", return_value=mock_docs), \
         patch("pipeline.extract_evidence_from_doc", return_value={"status": "no_claim", "data": None, "reason": "No claim"}), \
         patch("pipeline.check_evidence_coverage", return_value={"status": "failed", "evaluated": False, "sufficient": False, "gaps": [], "additional_queries": [], "error": "429 quota"}), \
         patch("pipeline.save_report") as mock_save_report:

        fake_report_file = str(tmp_path / "failed_report.json")
        mock_save_report.return_value = fake_report_file

        with pytest.raises(ValueError, match="zero evidence items"):
            pipeline.run_research_pipeline(brief)

        # Confirm save_report was called to preserve diagnostic artifact
        mock_save_report.assert_called_once()
        saved_evidence = mock_save_report.call_args[0][3]
        assert saved_evidence == []
        stats = mock_save_report.call_args[1].get("extraction_stats") or mock_save_report.call_args[0][4]
        assert stats.get("documents_skipped_no_claim") == 2


def test_run_research_job_raises_on_zero_evidence():
    """Verify run_research_job raises ValueError when research produces zero evidence."""
    brief = {"brand": "Pringles", "market": "India"}
    fake_pipeline_result = {
        "research_run_id": "test-zero-ev-run",
        "report_file": "/path/to/report.json",
        "evidence_count": 0,
        "report_data": {
            "metadata": {"total_evidence_count": 0, "status": "failed"},
            "evidence": []
        }
    }

    with patch("jobs.run_research_pipeline", return_value=fake_pipeline_result):
        with pytest.raises(ValueError, match="zero evidence items"):
            jobs.run_research_job(brief)


def test_run_research_job_succeeds_with_valid_evidence():
    """Verify run_research_job persists and succeeds when valid evidence items exist."""
    brief = {"brand": "Nike", "market": "India"}
    valid_evidence = [
        {"claim": "Nike dominates premium market", "topic": "market_share", "source": {"url": "https://test.com"}}
    ]
    fake_pipeline_result = {
        "research_run_id": "test-valid-run",
        "report_file": "/path/to/valid_report.json",
        "evidence_count": 1,
        "report_data": {
            "metadata": {"total_evidence_count": 1, "status": "success"},
            "evidence": valid_evidence
        },
        "queries": ["q1"]
    }

    with patch("jobs.run_research_pipeline", return_value=fake_pipeline_result), \
         patch("jobs.save_research_run", return_value="test-valid-run") as mock_save:

        res = jobs.run_research_job(brief)

        assert res["job_type"] == "research"
        assert res["research_run_id"] == "test-valid-run"
        assert res["total_evidence_count"] == 1
        mock_save.assert_called_once()


# ---------------------------------------------------------------------------
# 3. Downstream Validation & Orchestrator Halting Tests
# ---------------------------------------------------------------------------

def test_consumer_intelligence_rejects_empty_evidence():
    """Verify Agent 2 Consumer Intelligence raises actionable ValueError when run has 0 evidence."""
    with patch("agent2.get_run_with_evidence", return_value={"id": "run-empty", "evidence": []}):
        with pytest.raises(ValueError, match="at least one verified evidence item"):
            agent2.run_consumer_intelligence_pipeline("run-empty")


def test_orchestrator_halts_and_skips_downstream_when_research_fails():
    """Verify run_full_pipeline_job stops before Stage 2 and marks research as failed when Stage 1 fails."""
    brief = {"brand": "Pringles", "market": "India"}

    mock_job = MagicMock()
    mock_job.meta = {}

    with patch("rq.get_current_job", return_value=mock_job), \
         patch("jobs.run_research_job", side_effect=ValueError("Zero evidence items")), \
         patch("jobs.run_consumer_intelligence_job") as mock_agent2, \
         patch("jobs.run_competitive_intelligence_job") as mock_agent3:

        with pytest.raises(ValueError, match="Zero evidence items"):
            jobs.run_full_pipeline_job(brief)

        # Stage 2 and downstream stages must NOT have been called
        mock_agent2.assert_not_called()
        mock_agent3.assert_not_called()

        # Job meta must record research as failed
        assert mock_job.meta["stages"]["research"] == "failed"
        assert mock_job.meta["stages"]["consumer_intelligence"] == "pending"
        assert mock_job.meta["error_stage"] == "research"
        assert "Zero evidence items" in mock_job.meta["error_message"]


# ---------------------------------------------------------------------------
# 4. Saved Coverage Status Tests
# ---------------------------------------------------------------------------

def test_save_report_records_successful_coverage_status(tmp_path):
    """Verify save_report writes coverage_evaluation_status='success' when coverage passed."""
    brief = {"brand": "Nike", "market": "US", "audience": "Athletes", "objective": "Shoes"}
    cov_info = {
        "coverage_evaluation_status": "success",
        "coverage_rounds": 1,
        "coverage_gaps_identified": [],
        "additional_queries_used": [],
    }
    evidence = [{"claim": "Test claim", "topic": "fitness"}]

    fpath = pipeline.save_report(
        brief=brief,
        queries=["q1"],
        total_raw_docs=5,
        evidence_items=evidence,
        coverage_info=cov_info,
        output_dir=str(tmp_path),
        research_run_id="run-succ-cov"
    )

    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data["metadata"]
    assert meta["coverage_evaluation_status"] == "success"
    assert meta["status"] == "success"
    assert meta["research_run_id"] == "run-succ-cov"


def test_save_report_records_failed_coverage_status(tmp_path):
    """Verify save_report writes coverage_evaluation_status='failed' when coverage evaluation failed."""
    brief = {"brand": "Pringles", "market": "India", "audience": "Gen Z", "objective": "Snacks"}
    cov_info = {
        "coverage_evaluation_status": "failed",
        "coverage_check_result": {"status": "failed", "evaluated": False, "error": "429 Rate limit"},
        "coverage_rounds": 1,
        "coverage_gaps_identified": [],
        "additional_queries_used": [],
    }
    evidence = [{"claim": "Test claim", "topic": "flavor"}]

    fpath = pipeline.save_report(
        brief=brief,
        queries=["q1"],
        total_raw_docs=5,
        evidence_items=evidence,
        coverage_info=cov_info,
        output_dir=str(tmp_path),
        research_run_id="run-fail-cov"
    )

    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data["metadata"]
    assert meta["coverage_evaluation_status"] == "failed"
    assert meta["coverage_gaps_identified"] == []


def test_save_report_records_not_evaluated_coverage_status(tmp_path):
    """Verify save_report writes coverage_evaluation_status='not_evaluated' when coverage check was skipped."""
    brief = {"brand": "Puma", "market": "UK", "audience": "Runners", "objective": "Speed"}
    evidence = [{"claim": "Fast shoes", "topic": "performance"}]

    fpath = pipeline.save_report(
        brief=brief,
        queries=["q1"],
        total_raw_docs=3,
        evidence_items=evidence,
        coverage_info=None,
        output_dir=str(tmp_path)
    )

    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data["metadata"]
    assert meta["coverage_evaluation_status"] == "not_evaluated"


# ---------------------------------------------------------------------------
# 5. Failed-Run API Visibility Tests
# ---------------------------------------------------------------------------

def test_api_visibility_failed_run_and_successful_run(monkeypatch):
    """Verify GET /runs and GET /research/{run_id}/full expose status and diagnostics correctly."""
    import database as db
    import app as app_module
    from fastapi.testclient import TestClient

    monkeypatch.setattr(app_module, "_API_KEY", "test-key-xyz")
    client = TestClient(app_module.app, raise_server_exceptions=False)

    fake_diag = {
        "status": "failed",
        "error": "Zero evidence extracted from 12 documents (8 failed, 4 skipped).",
        "total_evidence_count": 0,
        "documents_processed": 12,
        "documents_failed_extraction": 8,
    }

    def mock_get_status(run_id, brand=None, evidence_count=0):
        if run_id == "run-failed":
            return "failed", fake_diag["error"], fake_diag
        elif run_id == "run-success":
            return "completed", None, None
        else:
            # Historical empty run without evidence and without failure proof must return unknown
            return "unknown", "No evidence recorded and no diagnostic report found.", None

    monkeypatch.setattr(db, "get_run_status_and_diagnostics", mock_get_status)

    fake_runs = [
        {
            "id": "run-failed",
            "brand": "Pringles",
            "market": "India",
            "audience": "Gen Z",
            "objective": "Gaps",
            "created_at": "2026-10-26T18:03:56",
            "status": "failed",
            "failure_reason": fake_diag["error"],
            "counts": {"evidence": 0, "consumer_insights": 0, "creative_gaps": 0, "opportunity_scores": 0, "content_pieces": 0},
        },
        {
            "id": "run-success",
            "brand": "Adidas",
            "market": "Germany",
            "audience": "All",
            "objective": "Market",
            "created_at": "2026-10-25T17:34:42",
            "status": "completed",
            "failure_reason": None,
            "counts": {"evidence": 11, "consumer_insights": 4, "creative_gaps": 2, "opportunity_scores": 2, "content_pieces": 3},
        },
        {
            "id": "run-historical-empty",
            "brand": "DraftBrand",
            "market": "US",
            "audience": "All",
            "objective": "Draft",
            "created_at": "2026-10-24T12:00:00",
            "status": "unknown",
            "failure_reason": "No evidence recorded and no diagnostic report found.",
            "counts": {"evidence": 0, "consumer_insights": 0, "creative_gaps": 0, "opportunity_scores": 0, "content_pieces": 0},
        }
    ]

    with patch("app.list_research_runs", return_value=fake_runs):
        resp = client.get("/runs", headers={"X-API-Key": "test-key-xyz"})
        assert resp.status_code == 200
        runs_list = resp.json()["runs"]
        assert len(runs_list) == 3

        failed_entry = next(r for r in runs_list if r["id"] == "run-failed")
        assert failed_entry["status"] == "failed"
        assert "Zero evidence extracted" in failed_entry["failure_reason"]

        success_entry = next(r for r in runs_list if r["id"] == "run-success")
        assert success_entry["status"] == "completed"
        assert success_entry["failure_reason"] is None

        historical_entry = next(r for r in runs_list if r["id"] == "run-historical-empty")
        # Historical run with 0 evidence and no failure proof must be marked unknown
        assert historical_entry["status"] == "unknown"
        assert historical_entry["failure_reason"] == "No evidence recorded and no diagnostic report found."

    # Test full endpoint returns diagnostic_report for failed run
    fake_full_failed = {
        "brief": {"id": "run-failed", "brand": "Pringles"},
        "status": "failed",
        "failure_reason": fake_diag["error"],
        "diagnostic_report": fake_diag,
        "evidence": [],
        "consumer_insights": [],
        "competitors": [],
        "competitive_gaps": [],
        "creative_gaps": [],
        "ranked_opportunities": [],
        "content_pieces": [],
        "cost_summary": {"call_count": 20},
    }

    with patch("app.get_full_research_run", return_value=fake_full_failed):
        resp = client.get("/research/run-failed/full", headers={"X-API-Key": "test-key-xyz"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "failed"
        assert body["diagnostic_report"]["documents_processed"] == 12
        assert body["diagnostic_report"]["documents_failed_extraction"] == 8


# ---------------------------------------------------------------------------
# 6. Report-Writing Failure Handling Test
# ---------------------------------------------------------------------------

def test_report_writing_oserror_propagates_and_orchestrator_halts():
    """Verify that if report writing raises OSError, exception propagates and research is marked failed."""
    brief = {"brand": "Puma", "market": "UK", "audience": "Athletes", "objective": "Test"}

    mock_client = MagicMock()
    mock_queries = ["query 1"]
    mock_docs = [{"title": "Doc 1", "url": "https://example.com/1", "snippet": "Snippet 1", "query": "query 1"}]

    with patch("pipeline.genai.Client", return_value=mock_client), \
         patch("pipeline.init_research_run"), \
         patch("pipeline.perform_semantic_recall", return_value=[]), \
         patch("pipeline.generate_queries", return_value=mock_queries), \
         patch("pipeline.search_web_serpapi", return_value=mock_docs), \
         patch("pipeline.extract_evidence_from_doc", return_value={"status": "success", "data": {"claim": "Valid claim", "source": {"url": "https://example.com/1"}}}), \
         patch("pipeline.check_evidence_coverage", return_value={"status": "success", "evaluated": True, "sufficient": True, "gaps": [], "additional_queries": []}), \
         patch("pipeline.save_report", side_effect=OSError("Read-only file system simulation")):

        with pytest.raises(OSError, match="Read-only file system simulation"):
            pipeline.run_research_pipeline(brief)

    # Now verify orchestrator marks research failed and downstream stages pending on OSError
    mock_job = MagicMock()
    mock_job.meta = {}

    with patch("rq.get_current_job", return_value=mock_job), \
         patch("jobs.run_research_job", side_effect=OSError("Simulated unwritable disk")), \
         patch("jobs.run_consumer_intelligence_job") as mock_agent2:

        with pytest.raises(OSError, match="Simulated unwritable disk"):
            jobs.run_full_pipeline_job(brief)

        mock_agent2.assert_not_called()
        assert mock_job.meta["stages"]["research"] == "failed"
        assert mock_job.meta["stages"]["consumer_intelligence"] == "pending"
        assert mock_job.meta["error_stage"] == "research"
        assert "Simulated unwritable disk" in mock_job.meta["error_message"]


# ---------------------------------------------------------------------------
# 7. Status Classification & Exact Diagnostic Matching Tests
# ---------------------------------------------------------------------------

def test_zero_evidence_run_with_no_report_returns_unknown(monkeypatch):
    """Verify get_run_status_and_diagnostics returns status='unknown' when evidence is 0 and no report exists."""
    import database as db

    monkeypatch.setattr(db, "find_diagnostic_report_for_run", lambda run_id, brand=None: None)

    status, reason, diag = db.get_run_status_and_diagnostics(
        run_id="run-zero-no-report",
        brand="SomeBrand",
        evidence_count=0
    )

    assert status == "unknown"
    assert reason == "No evidence recorded and no diagnostic report found."
    assert diag is None


def test_failed_run_with_exact_matching_diagnostic_report_returns_failed(tmp_path, monkeypatch):
    """Verify get_run_status_and_diagnostics returns status='failed' when diagnostic report matches exact run_id."""
    import database as db

    report_data = {
        "metadata": {
            "research_run_id": "run-fail-exact-123",
            "brand": "Pringles",
            "status": "failed",
            "error": "Quota exceeded during evidence extraction",
            "total_evidence_count": 0
        }
    }
    report_file = tmp_path / "research_report_Pringles_fail.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f)

    monkeypatch.chdir(tmp_path)

    status, reason, diag = db.get_run_status_and_diagnostics(
        run_id="run-fail-exact-123",
        brand="Pringles",
        evidence_count=0
    )

    assert status == "failed"
    assert reason == "Quota exceeded during evidence extraction"
    assert diag is not None
    assert diag["research_run_id"] == "run-fail-exact-123"


def test_same_brand_different_runs_cannot_inherit_failure(tmp_path, monkeypatch):
    """Verify that a run does NOT inherit another run's failure report just because they share the same brand."""
    import database as db

    report_data = {
        "metadata": {
            "research_run_id": "run-pringles-1",
            "brand": "Pringles",
            "status": "failed",
            "error": "Run 1 extraction failure",
            "total_evidence_count": 0
        }
    }
    report_file = tmp_path / "research_report_Pringles_1.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f)

    monkeypatch.chdir(tmp_path)

    # Query for run-2 (same brand "Pringles", but different run_id "run-pringles-2")
    status, reason, diag = db.get_run_status_and_diagnostics(
        run_id="run-pringles-2",
        brand="Pringles",
        evidence_count=0
    )

    # Must NOT be marked failed with run-1's failure
    assert status == "unknown"
    assert reason == "No evidence recorded and no diagnostic report found."
    assert diag is None


def test_successful_run_with_evidence_returns_completed_without_reading_disk(monkeypatch):
    """Verify runs with evidence_count > 0 immediately return status='completed' without reading disk."""
    import database as db

    def blow_up(*args, **kwargs):
        raise RuntimeError("Disk should not be inspected for runs with evidence")

    monkeypatch.setattr(db, "find_diagnostic_report_for_run", blow_up)

    status, reason, diag = db.get_run_status_and_diagnostics(
        run_id="run-with-evidence",
        brand="Nike",
        evidence_count=5
    )

    assert status == "completed"
    assert reason is None
    assert diag is None


def test_find_diagnostic_report_handles_malformed_and_unreadable_files(tmp_path, monkeypatch):
    """Verify find_diagnostic_report_for_run safely ignores corrupt or malformed JSON files."""
    import database as db

    # 1. Non-dict JSON (e.g., list)
    with open(tmp_path / "research_report_list.json", "w", encoding="utf-8") as f:
        f.write("[1, 2, 3]")

    # 2. Corrupt JSON
    with open(tmp_path / "research_report_corrupt.json", "w", encoding="utf-8") as f:
        f.write("{invalid json content")

    # 3. Valid JSON with different run_id
    valid_data = {
        "metadata": {
            "research_run_id": "run-other",
            "brand": "Adidas"
        }
    }
    with open(tmp_path / "research_report_valid_other.json", "w", encoding="utf-8") as f:
        json.dump(valid_data, f)

    monkeypatch.chdir(tmp_path)

    # Looking for a nonexistent run should return None without error
    result = db.find_diagnostic_report_for_run("run-nonexistent")
    assert result is None

    # Looking for run-other should succeed despite adjacent corrupt files
    result = db.find_diagnostic_report_for_run("run-other")
    assert result is not None
    assert result["research_run_id"] == "run-other"


def test_database_persistence_failure_propagates_as_failure():
    """Verify that a DB persistence error after extraction propagates as a failure and halts orchestrator."""
    brief = {"brand": "Nike", "market": "US"}
    valid_evidence = [{"claim": "Claim 1", "topic": "topic1", "source": {"url": "https://test.com"}}]
    fake_pipeline_result = {
        "research_run_id": "run-db-fail",
        "report_file": "/tmp/report.json",
        "evidence_count": 1,
        "report_data": {
            "metadata": {"total_evidence_count": 1, "status": "success"},
            "evidence": valid_evidence
        },
        "queries": ["q1"]
    }

    # 1. run_research_job raises DB error from save_research_run
    with patch("jobs.run_research_pipeline", return_value=fake_pipeline_result), \
         patch("jobs.save_research_run", side_effect=RuntimeError("Database connection lost")):

        with pytest.raises(RuntimeError, match="Database connection lost"):
            jobs.run_research_job(brief)

    # 2. In orchestrator, DB error in Stage 1 marks research failed and downstream stays pending
    mock_job = MagicMock()
    mock_job.meta = {}

    with patch("rq.get_current_job", return_value=mock_job), \
         patch("jobs.run_research_pipeline", return_value=fake_pipeline_result), \
         patch("jobs.save_research_run", side_effect=RuntimeError("Database connection lost")), \
         patch("jobs.run_consumer_intelligence_job") as mock_agent2:

        with pytest.raises(RuntimeError, match="Database connection lost"):
            jobs.run_full_pipeline_job(brief)

        mock_agent2.assert_not_called()
        assert mock_job.meta["stages"]["research"] == "failed"
        assert mock_job.meta["stages"]["consumer_intelligence"] == "pending"
        assert mock_job.meta["error_stage"] == "research"
        assert "Database connection lost" in mock_job.meta["error_message"]


