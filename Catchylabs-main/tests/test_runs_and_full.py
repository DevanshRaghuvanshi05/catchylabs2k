"""
Tests for GET /runs and GET /research/{run_id}/full.

Rules:
  - No live Gemini or SerpAPI calls — no external I/O.
  - Uses TEST_DATABASE_URL exclusively via conftest.py isolation layer.
  - Every test rolls back all DB writes via the db_session / _patch_session_local fixtures.
  - Secrets (API keys, DB URLs) are never printed or logged.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

VALID_KEY = "test-secret-key-xyz789"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_api_key(monkeypatch):
    """Patch the module-level API key for every test in this file."""
    import app as app_module
    monkeypatch.setattr(app_module, "_API_KEY", VALID_KEY)


@pytest.fixture()
def client():
    import app as app_module
    return TestClient(app_module.app, raise_server_exceptions=False)


@pytest.fixture()
def seeded_run(db_session):
    """
    Insert a minimal ResearchRun row (plus one Evidence, Source, ConsumerInsight,
    CreativeGap, OpportunityScore, ContentPiece) into the test DB.
    The db_session fixture rolls everything back after each test.
    Returns the run_id string.
    """
    import database as db

    run_id = str(uuid.uuid4())

    # ResearchRun
    run = db.ResearchRun(
        id=run_id,
        brand="TestBrand",
        market="TestMarket",
        audience="TestAudience",
        objective="TestObjective",
        created_at=datetime.now(timezone.utc),
    )
    db_session.add(run)
    db_session.flush()

    # Source
    from hashlib import sha256
    url = "https://example.com/article"
    title = "Test Article"
    c_hash = sha256(f"{url}:{title}".encode()).hexdigest()
    source = db.Source(
        url=url,
        title=title,
        publisher="Example Publisher",
        content_hash=c_hash,
    )
    db_session.add(source)
    db_session.flush()

    # Evidence
    evidence = db.Evidence(
        research_run_id=run_id,
        source_id=source.id,
        claim="Test claim about the market.",
        excerpt="Full excerpt here.",
        topic="market",
        confidence=0.9,
    )
    db_session.add(evidence)
    db_session.flush()

    # ConsumerInsight
    insight = db.ConsumerInsight(
        research_run_id=run_id,
        insight_type="segment",
        insight_text="Young urban consumers aged 18-24.",
        supporting_evidence_ids=[evidence.id],
    )
    db_session.add(insight)
    db_session.flush()

    # CreativeGap
    gap = db.CreativeGap(
        research_run_id=run_id,
        gap_title="Sustainability messaging gap",
        description="Competitors lack eco-friendly narrative.",
        supporting_consumer_insight_ids=[insight.id],
        supporting_competitor_names=["CompetitorA"],
        supporting_competitive_gap_ids=[],
        confidence="high",
    )
    db_session.add(gap)
    db_session.flush()

    # OpportunityScore
    opp = db.OpportunityScore(
        research_run_id=run_id,
        creative_gap_id=gap.id,
        rank=1,
        overall_score=8.5,
        audience_demand_score=9,
        audience_demand_reasoning="Strong demand for eco content.",
        competitive_whitespace_score=8,
        competitive_whitespace_reasoning="Few competitors cover this.",
        brand_fit_score=8,
        brand_fit_reasoning="Brand is eco-aligned.",
        feasibility_score=9,
        feasibility_reasoning="Low production cost.",
    )
    db_session.add(opp)
    db_session.flush()

    # ContentPiece
    piece = db.ContentPiece(
        research_run_id=run_id,
        creative_gap_id=gap.id,
        content_type="reel_script",
        platform="instagram",
        title="Eco Reel",
        hook="Did you know?",
        body={"scenes": []},
        caption="Go green.",
        hashtags=["#eco", "#green"],
        call_to_action="Shop now",
        supporting_consumer_insight_ids=[insight.id],
        supporting_competitor_names=["CompetitorA"],
        claims_to_verify=["[VERIFIED CLAIM NEEDED: recycled materials]"],
        status="draft",
    )
    db_session.add(piece)
    db_session.flush()

    db_session.commit()
    return run_id


# ---------------------------------------------------------------------------
# GET /runs
# ---------------------------------------------------------------------------

class TestListRuns:
    """Tests for GET /runs."""

    def test_missing_key_returns_401(self, client):
        """GET /runs must return 401 without X-API-Key."""
        resp = client.get("/runs")
        assert resp.status_code == 401, f"Expected 401, got {resp.status_code}"

    def test_wrong_key_returns_401(self, client):
        """GET /runs must return 401 with a wrong key."""
        resp = client.get("/runs", headers={"X-API-Key": "definitely-wrong"})
        assert resp.status_code == 401, f"Expected 401, got {resp.status_code}"

    def test_correct_key_returns_200(self, client):
        """GET /runs must return 200 with valid key (even with no data)."""
        resp = client.get("/runs", headers={"X-API-Key": VALID_KEY})
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

    def test_response_shape_empty(self, client):
        """GET /runs returns total_runs and runs list even when DB is empty."""
        with patch("app.list_research_runs", return_value=[]):
            resp = client.get("/runs", headers={"X-API-Key": VALID_KEY})
        assert resp.status_code == 200
        body = resp.json()
        assert "total_runs" in body
        assert "runs" in body
        assert isinstance(body["runs"], list)
        assert body["total_runs"] == 0

    def test_seeded_run_appears_in_list(self, client, seeded_run):
        """GET /runs must include the seeded run with correct fields and counts."""
        resp = client.get("/runs", headers={"X-API-Key": VALID_KEY})
        assert resp.status_code == 200
        body = resp.json()
        run_ids = [r["id"] for r in body["runs"]]
        assert seeded_run in run_ids, f"Seeded run {seeded_run} not in response: {run_ids}"

        matched = next(r for r in body["runs"] if r["id"] == seeded_run)
        # Required fields
        assert matched["brand"] == "TestBrand"
        assert matched["market"] == "TestMarket"
        assert matched["audience"] == "TestAudience"
        assert matched["objective"] == "TestObjective"
        assert matched["created_at"] is not None

        # Counts sub-object
        counts = matched["counts"]
        assert counts["evidence"] >= 1
        assert counts["consumer_insights"] >= 1
        assert counts["creative_gaps"] >= 1
        assert counts["opportunity_scores"] >= 1
        assert counts["content_pieces"] >= 1

    def test_count_fields_are_integers(self, client, seeded_run):
        """All count fields in GET /runs must be integers."""
        resp = client.get("/runs", headers={"X-API-Key": VALID_KEY})
        assert resp.status_code == 200
        body = resp.json()
        matched = next(r for r in body["runs"] if r["id"] == seeded_run)
        for key, val in matched["counts"].items():
            assert isinstance(val, int), f"counts.{key} is not int: {val!r}"

    def test_total_runs_matches_list_length(self, client, seeded_run):
        """total_runs must equal len(runs)."""
        resp = client.get("/runs", headers={"X-API-Key": VALID_KEY})
        body = resp.json()
        assert body["total_runs"] == len(body["runs"])


# ---------------------------------------------------------------------------
# GET /research/{run_id}/full
# ---------------------------------------------------------------------------

class TestFullResearchRun:
    """Tests for GET /research/{run_id}/full."""

    def test_missing_key_returns_401(self, client):
        """GET /research/{run_id}/full must return 401 without X-API-Key."""
        resp = client.get("/research/any-id/full")
        assert resp.status_code == 401

    def test_wrong_key_returns_401(self, client):
        """GET /research/{run_id}/full must return 401 with wrong key."""
        resp = client.get(
            "/research/any-id/full",
            headers={"X-API-Key": "bad-key"},
        )
        assert resp.status_code == 401

    def test_unknown_run_returns_404(self, client):
        """GET /research/{run_id}/full must return 404 for a non-existent run_id."""
        fake_id = str(uuid.uuid4())
        resp = client.get(
            f"/research/{fake_id}/full",
            headers={"X-API-Key": VALID_KEY},
        )
        assert resp.status_code == 404, f"Expected 404, got {resp.status_code}: {resp.text}"
        assert "not found" in resp.json()["detail"].lower()

    def test_seeded_run_returns_200(self, client, seeded_run):
        """GET /research/{run_id}/full returns 200 for a known run."""
        resp = client.get(
            f"/research/{seeded_run}/full",
            headers={"X-API-Key": VALID_KEY},
        )
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

    def test_response_top_level_keys(self, client, seeded_run):
        """Response must contain all required top-level keys."""
        resp = client.get(
            f"/research/{seeded_run}/full",
            headers={"X-API-Key": VALID_KEY},
        )
        body = resp.json()
        required_keys = {
            "brief", "evidence", "consumer_insights",
            "competitors", "competitive_gaps", "creative_gaps",
            "ranked_opportunities", "content_pieces", "cost_summary",
        }
        missing = required_keys - set(body.keys())
        assert not missing, f"Missing keys in /full response: {missing}"

    def test_brief_fields(self, client, seeded_run):
        """brief must contain id, brand, market, audience, objective, created_at."""
        resp = client.get(
            f"/research/{seeded_run}/full",
            headers={"X-API-Key": VALID_KEY},
        )
        brief = resp.json()["brief"]
        assert brief["id"] == seeded_run
        assert brief["brand"] == "TestBrand"
        assert brief["market"] == "TestMarket"
        assert brief["audience"] == "TestAudience"
        assert brief["objective"] == "TestObjective"
        assert brief["created_at"] is not None

    def test_evidence_has_source_title_and_url(self, client, seeded_run):
        """Each evidence item must have source.title and source.url."""
        resp = client.get(
            f"/research/{seeded_run}/full",
            headers={"X-API-Key": VALID_KEY},
        )
        evidence = resp.json()["evidence"]
        assert len(evidence) >= 1, "Expected at least one evidence item"
        for item in evidence:
            assert "source" in item, "Evidence missing 'source' key"
            assert "title" in item["source"], "Evidence source missing 'title'"
            assert "url" in item["source"], "Evidence source missing 'url'"

    def test_consumer_insights_present(self, client, seeded_run):
        """consumer_insights must be a non-empty list."""
        resp = client.get(
            f"/research/{seeded_run}/full",
            headers={"X-API-Key": VALID_KEY},
        )
        insights = resp.json()["consumer_insights"]
        assert isinstance(insights, list)
        assert len(insights) >= 1

    def test_creative_gaps_present(self, client, seeded_run):
        """creative_gaps must be a non-empty list with required fields."""
        resp = client.get(
            f"/research/{seeded_run}/full",
            headers={"X-API-Key": VALID_KEY},
        )
        gaps = resp.json()["creative_gaps"]
        assert len(gaps) >= 1
        gap = gaps[0]
        assert "gap_title" in gap
        assert "description" in gap
        assert "confidence" in gap

    def test_ranked_opportunities_score_recomputed(self, client, seeded_run):
        """overall_score must be recomputed from component scores using opportunity_engine weights."""
        from opportunity_engine import (
            WEIGHT_AUDIENCE_DEMAND,
            WEIGHT_COMPETITIVE_WHITESPACE,
            WEIGHT_BRAND_FIT,
            WEIGHT_FEASIBILITY,
        )
        resp = client.get(
            f"/research/{seeded_run}/full",
            headers={"X-API-Key": VALID_KEY},
        )
        opps = resp.json()["ranked_opportunities"]
        assert len(opps) >= 1
        opp = opps[0]
        scores = opp["scores"]
        expected = round(
            scores["audience_demand"] * WEIGHT_AUDIENCE_DEMAND
            + scores["competitive_whitespace"] * WEIGHT_COMPETITIVE_WHITESPACE
            + scores["brand_fit"] * WEIGHT_BRAND_FIT
            + scores["feasibility"] * WEIGHT_FEASIBILITY,
            2,
        )
        assert abs(opp["overall_score"] - expected) < 0.001, (
            f"Recomputed score {opp['overall_score']} != expected {expected}"
        )

    def test_full_overall_score_equals_stored_and_report_and_rank_matches(
        self, client, db_session, seeded_run
    ):
        """
        overall_score from /full must equal stored_overall_score,
        equal the value in generate_report.py output, and rank order must match.
        """
        import database as db
        from generate_report import fetch_run_report_data, generate_markdown_report
        from opportunity_engine import compute_overall_score

        # Add a second creative gap and opportunity to verify multi-item rank ordering
        insight = db_session.query(db.ConsumerInsight).filter_by(research_run_id=seeded_run).first()
        gap2 = db.CreativeGap(
            research_run_id=seeded_run,
            gap_title="Secondary Gap Title",
            description="Second opportunity gap description.",
            supporting_consumer_insight_ids=[insight.id] if insight else [],
            supporting_competitor_names=["CompetitorB"],
            supporting_competitive_gap_ids=[],
            confidence="medium",
        )
        db_session.add(gap2)
        db_session.flush()

        opp2_score = compute_overall_score(7, 6, 8, 7)
        opp2 = db.OpportunityScore(
            research_run_id=seeded_run,
            creative_gap_id=gap2.id,
            rank=2,
            overall_score=opp2_score,
            audience_demand_score=7,
            audience_demand_reasoning="Reasoning for audience demand.",
            competitive_whitespace_score=6,
            competitive_whitespace_reasoning="Reasoning for competitive whitespace.",
            brand_fit_score=8,
            brand_fit_reasoning="Reasoning for brand fit.",
            feasibility_score=7,
            feasibility_reasoning="Reasoning for feasibility.",
        )
        db_session.add(opp2)
        db_session.flush()

        # Query GET /research/{run_id}/full
        resp = client.get(
            f"/research/{seeded_run}/full",
            headers={"X-API-Key": VALID_KEY},
        )
        assert resp.status_code == 200
        full_opps = resp.json()["ranked_opportunities"]
        assert len(full_opps) >= 2

        # Query report data from generate_report.py
        report_data = fetch_run_report_data(seeded_run, session=db_session)
        report_sections = report_data["scored_sections"]
        assert len(report_sections) == len(full_opps)

        # 1. Rank order matches between /full, generate_report.py, and is strictly ascending
        full_ranks = [o["rank"] for o in full_opps]
        report_ranks = [s["opportunity_score"].rank for s in report_sections]
        assert full_ranks == report_ranks == sorted(full_ranks)

        # 2. overall_score from /full equals stored_overall_score and equals value in generate_report.py
        for full_opp, report_sec in zip(full_opps, report_sections):
            stored_score = full_opp["stored_overall_score"]
            computed_full_score = full_opp["overall_score"]
            report_score = report_sec["recomputed_overall_score"]

            assert computed_full_score == stored_score, (
                f"/full overall_score ({computed_full_score}) != stored_overall_score ({stored_score})"
            )
            assert computed_full_score == report_score, (
                f"/full overall_score ({computed_full_score}) != report score ({report_score})"
            )

        # 3. Check formatted markdown report output
        report_md = generate_markdown_report(seeded_run, session=db_session)
        for full_opp in full_opps:
            expected_table_entry = f"| Recomputed Overall Score | {full_opp['overall_score']:.2f} |"
            assert expected_table_entry in report_md

    def test_content_pieces_have_claims_to_verify(self, client, seeded_run):
        """Content pieces must include claims_to_verify list."""
        resp = client.get(
            f"/research/{seeded_run}/full",
            headers={"X-API-Key": VALID_KEY},
        )
        pieces = resp.json()["content_pieces"]
        assert len(pieces) >= 1
        piece = pieces[0]
        assert "claims_to_verify" in piece
        assert isinstance(piece["claims_to_verify"], list)
        # The seeded piece has one claim
        assert len(piece["claims_to_verify"]) == 1

    def test_cost_summary_keys(self, client, seeded_run):
        """cost_summary must contain the expected keys."""
        resp = client.get(
            f"/research/{seeded_run}/full",
            headers={"X-API-Key": VALID_KEY},
        )
        cost = resp.json()["cost_summary"]
        assert "hypothetical_paid_tier_cost_usd" in cost
        assert "total_duration_ms" in cost
        assert "call_count" in cost
        assert "serpapi_calls_this_run" in cost

    def test_ranked_opportunities_sorted_by_rank(self, client, seeded_run):
        """ranked_opportunities must be ordered by rank ascending."""
        resp = client.get(
            f"/research/{seeded_run}/full",
            headers={"X-API-Key": VALID_KEY},
        )
        opps = resp.json()["ranked_opportunities"]
        ranks = [o["rank"] for o in opps]
        assert ranks == sorted(ranks), f"Opportunities not sorted by rank: {ranks}"
