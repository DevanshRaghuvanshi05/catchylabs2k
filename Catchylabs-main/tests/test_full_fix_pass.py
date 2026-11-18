"""
Comprehensive test suite for CatchyLab backend fix pass.
Covers Steps 2-6: schema, stale citations, content validators,
claim resolution, and smoke tests for all agents.

Requires TEST_DATABASE_URL to be set (conftest.py enforces this).
All Gemini and SerpAPI calls are mocked.
"""

import json
import os
import uuid
import pytest
from unittest.mock import patch, MagicMock, Mock
from datetime import datetime, timezone, timedelta


# ---------------------------------------------------------------------------
# Helper: seed a research run + evidence into the test DB
# ---------------------------------------------------------------------------

def seed_run_and_evidence(db_session, run_id=None, brand="TestBrand", market="India",
                          audience="Gen Z", objective="Test objective", n_evidence=3):
    """Insert a research_run + N evidence rows. Returns (run_id, evidence_ids)."""
    from database import ResearchRun, Source, Evidence

    run_id = run_id or str(uuid.uuid4())
    run = ResearchRun(id=run_id, brand=brand, audience=audience, market=market, objective=objective)
    db_session.add(run)
    db_session.flush()

    ev_ids = []
    for i in range(n_evidence):
        src = Source(url=f"https://example.com/{i}", title=f"Source {i}",
                     publisher="TestPub", content_hash=f"{run_id}-hash-{i}")
        db_session.add(src)
        db_session.flush()
        ev = Evidence(research_run_id=run_id, source_id=src.id,
                      claim=f"Test claim {i} about market trends",
                      excerpt=f"Excerpt {i}", topic="test", confidence=0.9)
        db_session.add(ev)
        db_session.flush()
        ev_ids.append(ev.id)

    db_session.commit()
    return run_id, ev_ids


# ===========================================================================
# STEP 2 — SCHEMA TESTS: RESTRICT behavior on research run deletion
# ===========================================================================

class TestTestDatabaseInitialization:
    """Assert init_db-created guardrails exist in the test database."""

    def test_init_db_installs_partial_unique_index_triggers_and_rls(self, db_session):
        from sqlalchemy import text

        indexdef = db_session.execute(text(
            "SELECT indexdef FROM pg_indexes "
            "WHERE schemaname='public' AND indexname='ix_ci_research_run_active'"
        )).scalar_one()
        assert "UNIQUE INDEX" in indexdef.upper()
        assert "where (superseded_at is null)" in indexdef.lower() or "where superseded_at is null" in indexdef.lower()

        triggers = set(db_session.execute(text(
            "SELECT tgname FROM pg_trigger "
            "WHERE NOT tgisinternal AND tgname IN "
            "('trg_check_supporting_consumer_insight_ids', 'trg_check_content_piece_insight_ids')"
        )).scalars())
        assert triggers == {
            "trg_check_supporting_consumer_insight_ids",
            "trg_check_content_piece_insight_ids",
        }

        rls = db_session.execute(text(
            "SELECT relrowsecurity FROM pg_class "
            "WHERE oid='public.content_feedback'::regclass"
        )).scalar_one()
        assert rls is True

    def test_creative_gap_trigger_rejects_insight_id_from_another_run(self, db_session):
        from database import CreativeGap

        run_id, _ = seed_run_and_evidence(db_session)
        invalid_gap = CreativeGap(
            research_run_id=run_id,
            gap_title="Trigger validation",
            description="This cites an insight ID that does not exist in this run.",
            supporting_consumer_insight_ids=[2147483000],
            supporting_competitor_names=[],
            supporting_competitive_gap_ids=[],
            confidence="high",
        )
        with pytest.raises(Exception, match="Foreign key violation|supporting_consumer_insight_ids"):
            with db_session.begin_nested():
                db_session.add(invalid_gap)
                db_session.flush()

class TestRestrictDeleteBehavior:
    """Test that RESTRICT FKs prevent deleting runs with downstream data."""

    def test_delete_run_blocked_by_content_pieces(self, db_session):
        """Deleting a research run with content_pieces should raise (RESTRICT)."""
        from database import (ResearchRun, ConsumerInsight, CreativeGap,
                               ContentPiece)
        from sqlalchemy import text

        run_id, ev_ids = seed_run_and_evidence(db_session)

        # Add consumer insight
        ci = ConsumerInsight(research_run_id=run_id, insight_type="segment",
                             insight_text="Test insight for deletion test",
                             supporting_evidence_ids=ev_ids[:1])
        db_session.add(ci)
        db_session.flush()

        # Add creative gap referencing that insight
        gap = CreativeGap(research_run_id=run_id, gap_title="Test Gap",
                          description="Test desc",
                          supporting_consumer_insight_ids=[ci.id],
                          supporting_competitor_names=["CompA"],
                          supporting_competitive_gap_ids=[],
                          confidence="high")
        db_session.add(gap)
        db_session.flush()

        # Add content piece (has RESTRICT FK to research_runs)
        cp = ContentPiece(research_run_id=run_id, creative_gap_id=gap.id,
                          content_type="reel_script", platform="instagram",
                          title="Test", hook="Hook", body=[],
                          caption="Cap", hashtags=[], call_to_action="CTA",
                          supporting_consumer_insight_ids=[ci.id],
                          supporting_competitor_names=["CompA"])
        db_session.add(cp)
        db_session.commit()

        # Try to delete the research run — should fail with RESTRICT
        with pytest.raises(Exception) as exc_info:
            db_session.execute(
                text("DELETE FROM research_runs WHERE id = :rid"),
                {"rid": run_id}
            )
            db_session.commit()
        db_session.rollback()
        # The error should mention a FK violation
        assert "violates foreign key" in str(exc_info.value).lower() or \
               "restrict" in str(exc_info.value).lower() or \
               "still referenced" in str(exc_info.value).lower()

    def test_delete_run_blocked_by_opportunity_scores(self, db_session):
        """Deleting a research run with opportunity_scores should raise (RESTRICT)."""
        from database import (ResearchRun, ConsumerInsight, CreativeGap,
                               OpportunityScore)
        from sqlalchemy import text

        run_id, ev_ids = seed_run_and_evidence(db_session)

        ci = ConsumerInsight(research_run_id=run_id, insight_type="motivation",
                             insight_text="Test insight for opp test",
                             supporting_evidence_ids=ev_ids[:1])
        db_session.add(ci)
        db_session.flush()

        gap = CreativeGap(research_run_id=run_id, gap_title="Test Gap 2",
                          description="Test desc 2",
                          supporting_consumer_insight_ids=[ci.id],
                          supporting_competitor_names=["CompB"],
                          supporting_competitive_gap_ids=[],
                          confidence="medium")
        db_session.add(gap)
        db_session.flush()

        opp = OpportunityScore(
            research_run_id=run_id, creative_gap_id=gap.id,
            rank=1, overall_score=7.5,
            audience_demand_score=4, audience_demand_reasoning="Strong demand",
            competitive_whitespace_score=3, competitive_whitespace_reasoning="Some gaps",
            brand_fit_score=4, brand_fit_reasoning="Good fit",
            feasibility_score=3, feasibility_reasoning="Feasible")
        db_session.add(opp)
        db_session.commit()

        with pytest.raises(Exception) as exc_info:
            db_session.execute(
                text("DELETE FROM research_runs WHERE id = :rid"),
                {"rid": run_id}
            )
            db_session.commit()
        db_session.rollback()
        assert "violates foreign key" in str(exc_info.value).lower() or \
               "restrict" in str(exc_info.value).lower() or \
               "still referenced" in str(exc_info.value).lower()

    def test_delete_run_succeeds_without_restrict_children(self, db_session):
        """Deleting a run that only has CASCADE children (evidence, insights) should succeed."""
        from sqlalchemy import text

        run_id, _ = seed_run_and_evidence(db_session)

        # This should succeed since evidence uses CASCADE
        db_session.execute(
            text("DELETE FROM research_runs WHERE id = :rid"),
            {"rid": run_id}
        )
        db_session.commit()


# ===========================================================================
# STEP 2 — Competitive intelligence re-run (partial unique index)
# ===========================================================================

class TestCompetitiveIntelligenceRerun:
    """Test that CI can be saved twice for the same run without violating unique constraint."""

    def test_save_ci_twice_no_unique_violation(self, db_session):
        """Second save_competitive_intelligence should supersede old record, not raise."""
        from database import save_competitive_intelligence, get_competitive_intelligence

        run_id, _ = seed_run_and_evidence(db_session)

        data1 = {
            "competitors": [{"name": "CompA", "positioning": "Budget leader"}],
            "market_structure": {"type": "oligopoly"},
            "competitive_gaps": [{"id": 1, "description": "Gap 1"}]
        }
        result1 = save_competitive_intelligence(run_id, data1)
        assert result1["competitors"][0]["name"] == "CompA"

        data2 = {
            "competitors": [{"name": "CompB", "positioning": "Premium player"}],
            "market_structure": {"type": "monopolistic"},
            "competitive_gaps": [{"id": 1, "description": "Gap 2"}]
        }
        result2 = save_competitive_intelligence(run_id, data2)
        assert result2["competitors"][0]["name"] == "CompB"

        # GET should return only the latest (non-superseded)
        current = get_competitive_intelligence(run_id)
        assert current["competitors"][0]["name"] == "CompB"


# ===========================================================================
# STEP 3 — STALE CITATION CHECKS
# ===========================================================================

class TestStaleCitations:
    """Test stale citation detection when consumer insights are superseded."""

    def test_stale_flag_on_creative_gaps(self, db_session):
        """Creative gaps citing superseded consumer insights should be flagged stale."""
        from database import (save_consumer_insights, save_creative_gaps,
                               get_creative_gaps, ConsumerInsight)

        run_id, ev_ids = seed_run_and_evidence(db_session)

        # First Agent 2 run
        insights_v1 = [
            {"insight_type": "segment", "insight_text": "Young buyers prefer eco options",
             "supporting_evidence_ids": ev_ids[:1]},
            {"insight_type": "motivation", "insight_text": "Price sensitivity drives choices",
             "supporting_evidence_ids": ev_ids[1:2]},
        ]
        saved_v1 = save_consumer_insights(run_id, insights_v1)
        v1_ids = [s["id"] for s in saved_v1]

        # Save creative gaps referencing v1 insights
        gaps = [
            {"gap_title": "Eco Budget Gap", "description": "Budget eco opportunity",
             "supporting_consumer_insight_ids": v1_ids,
             "supporting_competitor_names": ["CompA"],
             "supporting_competitive_gap_ids": [],
             "confidence": "high"}
        ]
        save_creative_gaps(run_id, gaps)

        # Re-run Agent 2 — v1 insights become superseded
        insights_v2 = [
            {"insight_type": "segment", "insight_text": "Updated: eco preference stronger",
             "supporting_evidence_ids": ev_ids[:2]},
        ]
        save_consumer_insights(run_id, insights_v2)

        # Now creative gaps cite superseded insight IDs
        # The get_creative_gaps function should flag them as stale
        result = get_creative_gaps(run_id)
        assert result is not None
        assert len(result) > 0

        # Check staleness by verifying cited insight IDs are superseded
        gap = result[0]
        cited_ids = gap["supporting_consumer_insight_ids"]
        # Verify cited IDs are from v1 (now superseded)
        for cid in cited_ids:
            ci = db_session.query(ConsumerInsight).filter_by(id=cid).first()
            assert ci is not None
            assert ci.superseded_at is not None, f"Insight {cid} should be superseded"

        # Verify the GET return value decorates is_stale and stale_insight_ids
        assert gap["is_stale"] is True, "Gap should be flagged stale when citing superseded insights"
        assert set(gap["stale_insight_ids"]) == set(v1_ids), (
            f"stale_insight_ids should match v1 IDs {v1_ids}, got {gap['stale_insight_ids']}"
        )

    def test_opportunity_engine_refuses_stale_gaps(self, db_session):
        """Opportunity Engine should raise StaleInputError when creative gaps cite superseded insights."""
        from database import (save_consumer_insights, save_creative_gaps,
                               StaleInputError)

        run_id, ev_ids = seed_run_and_evidence(db_session)

        # Agent 2 v1
        v1 = save_consumer_insights(run_id, [
            {"insight_type": "segment", "insight_text": "V1 insight",
             "supporting_evidence_ids": ev_ids[:1]},
        ])
        v1_ids = [s["id"] for s in v1]

        # Agent 4 gaps citing v1
        save_creative_gaps(run_id, [{
            "gap_title": "Gap on v1", "description": "Cites v1",
            "supporting_consumer_insight_ids": v1_ids,
            "supporting_competitor_names": ["CompX"],
            "supporting_competitive_gap_ids": [], "confidence": "high"
        }])

        # Agent 2 re-run — v1 superseded
        save_consumer_insights(run_id, [
            {"insight_type": "motivation", "insight_text": "V2 insight",
             "supporting_evidence_ids": ev_ids[:2]},
        ])

        # Opportunity Engine should refuse
        with patch("opportunity_engine.genai"), \
             pytest.raises(StaleInputError, match="superseded"):
            from opportunity_engine import run_opportunity_engine_pipeline
            run_opportunity_engine_pipeline(run_id)

    def test_content_generator_refuses_stale_opportunities(self, db_session):
        """Content Generator should raise StaleInputError when opportunities cite superseded insights."""
        from database import (save_consumer_insights, save_creative_gaps,
                               save_opportunity_scores, StaleInputError)

        run_id, ev_ids = seed_run_and_evidence(db_session)

        # Agent 2 v1
        v1 = save_consumer_insights(run_id, [
            {"insight_type": "segment", "insight_text": "V1 insight",
             "supporting_evidence_ids": ev_ids[:1]},
        ])
        v1_ids = [s["id"] for s in v1]

        # Agent 4 gaps citing v1
        gaps = save_creative_gaps(run_id, [{
            "gap_title": "Gap on v1", "description": "Cites v1",
            "supporting_consumer_insight_ids": v1_ids,
            "supporting_competitor_names": ["CompX"],
            "supporting_competitive_gap_ids": [], "confidence": "high"
        }])
        gap_id = gaps[0]["id"]

        # Opportunity scores referencing the gap
        save_opportunity_scores(run_id, [{
            "creative_gap_id": gap_id, "rank": 1, "overall_score": 8.0,
            "audience_demand_score": 4, "audience_demand_reasoning": "Strong",
            "competitive_whitespace_score": 3, "competitive_whitespace_reasoning": "Wide",
            "brand_fit_score": 4, "brand_fit_reasoning": "Good",
            "feasibility_score": 3, "feasibility_reasoning": "Feasible"
        }])

        # Agent 2 re-run — v1 superseded
        save_consumer_insights(run_id, [
            {"insight_type": "motivation", "insight_text": "V2 insight",
             "supporting_evidence_ids": ev_ids[:2]},
        ])

        # Content Generator should refuse
        with patch("content_generator.genai"), \
             pytest.raises(StaleInputError, match="superseded"):
            from content_generator import run_content_generation_pipeline
            run_content_generation_pipeline(run_id, creative_gap_id=gap_id)


# ===========================================================================
# STEP 4 — CONTENT VALIDATORS (new test sentences)
# ===========================================================================

class TestContentValidatorsNewSentences:
    """New test sentences not used to tune the rules."""

    def _check(self, text, brand="Puma", cited_evidence=""):
        """Helper to run brand claim validation."""
        # Import locally to avoid module-level issues
        import sys
        # Make sure we have access to content_generator
        from content_generator import _brand_claim_errors
        return _brand_claim_errors(text, 1, brand, cited_evidence)

    # --- 5 brand claims that MUST be flagged ---

    def test_brand_claim_1_recycled_materials(self):
        """Brand claim: Puma uses recycled materials — should be flagged."""
        errors, claims = self._check("Puma uses 100% recycled materials in every shoe.")
        assert len(claims) > 0, "Should flag unverified material claim"

    def test_brand_claim_2_lowest_price(self):
        """Brand claim: our shoes are the most affordable — should be flagged."""
        errors, claims = self._check("Our shoes offer the most affordable option on the market.")
        assert len(claims) > 0, "Should flag unverified affordability claim"

    def test_brand_claim_3_durability_promise(self):
        """Brand claim: Puma sneakers last 3x longer — should be flagged."""
        errors, claims = self._check("Puma sneakers are built for durability and last longer than competitors.")
        assert len(claims) > 0, "Should flag unverified durability claim"

    def test_brand_claim_4_eco_certification(self):
        """Brand claim: we are eco-friendly certified — should be flagged."""
        errors, claims = self._check("We are eco-friendly and committed to sustainable practices.")
        assert len(claims) > 0, "Should flag unverified eco claim"

    def test_brand_claim_5_premium_comfort(self):
        """Brand claim: these shoes offer premium comfort — should be flagged."""
        errors, claims = self._check("These shoes deliver premium comfort with every step.")
        assert len(claims) > 0, "Should flag unverified premium/comfort claim"

    # --- 5 audience-side lines that MUST pass ---

    def test_audience_line_1_budget_worry(self):
        """Audience language: people worry about prices — should pass."""
        errors, claims = self._check("Many young consumers struggle with rising footwear prices.")
        assert len(claims) == 0, f"Should not flag audience price concern, got: {claims}"

    def test_audience_line_2_sustainability_desire(self):
        """Audience language: Gen Z wants sustainable fashion — should pass."""
        errors, claims = self._check("Gen Z shoppers are increasingly drawn to sustainable fashion choices.")
        assert len(claims) == 0, f"Should not flag audience sustainability desire, got: {claims}"

    def test_audience_line_3_comfort_seeking(self):
        """Audience language: runners look for comfortable shoes — should pass."""
        errors, claims = self._check("Runners are always searching for comfortable and supportive footwear.")
        assert len(claims) == 0, f"Should not flag audience comfort seeking, got: {claims}"

    def test_audience_line_4_quality_expectation(self):
        """Audience language: shoppers expect good quality — should pass."""
        errors, claims = self._check("Today's shoppers expect quality craftsmanship at every price point.")
        assert len(claims) == 0, f"Should not flag audience quality expectation, got: {claims}"

    def test_audience_line_5_eco_awareness(self):
        """Audience language: environmental awareness growing — should pass."""
        errors, claims = self._check("Environmental awareness among consumers continues to grow rapidly.")
        assert len(claims) == 0, f"Should not flag audience eco awareness, got: {claims}"


class TestVoiceoverRule:
    """Test the 3 words/second voiceover rule."""

    def test_voiceover_over_budget_rejected(self):
        """Scene with 20 words in 3 seconds (max 9) should be rejected."""
        from content_generator import validate_content_pieces
        data = {
            "content_pieces": [{
                "content_type": "reel_script", "title": "Test", "hook": "Hook here",
                "body": [
                    {"visual": "V", "voiceover": "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty",
                     "on_screen_text": "OST", "duration_seconds": 3},
                    {"visual": "V", "voiceover": "Short line", "on_screen_text": "OST", "duration_seconds": 12}
                ],
                "caption": "Cap text", "call_to_action": "CTA text",
                "supporting_consumer_insight_ids": [1],
                "supporting_competitor_names": ["CompA"]
            }]
        }
        validated, errors = validate_content_pieces(
            data, valid_ci_ids={1}, valid_comp_names={"CompA"},
            insight_texts={1: "test"}, competitor_texts={"compa": "test"},
            brand_name="TestBrand"
        )
        assert any("Voiceover too long" in e for e in errors)


class TestGeminiCallCap:
    """Confirm the 4-call cap across API retries and validation retries."""

    def test_max_4_gemini_calls(self):
        """Content generator should make at most 4 Gemini calls total."""
        from content_generator import run_content_generation_pipeline

        call_count = 0

        def mock_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise Exception("503 UNAVAILABLE")

        mock_client = MagicMock()
        mock_client.models.generate_content = mock_generate

        # Patch all DB reads
        with patch("content_generator.get_run_with_evidence") as m_run, \
             patch("content_generator.get_opportunity_scores") as m_opp, \
             patch("content_generator.get_creative_gaps") as m_gaps, \
             patch("content_generator.get_consumer_insights") as m_ci, \
             patch("content_generator.get_competitive_intelligence") as m_comp, \
             patch("content_generator.save_content_pieces") as m_save, \
             patch("content_generator.track_api_call") as m_track:

            m_run.return_value = {"id": "r1", "brand": "B", "market": "M",
                                  "audience": "A", "objective": "O", "evidence": []}
            m_opp.return_value = [{"id": 1, "creative_gap_id": 1, "rank": 1,
                                   "overall_score": 8.0}]
            m_gaps.return_value = [{"id": 1, "gap_title": "G1", "description": "D1",
                                    "supporting_consumer_insight_ids": [1],
                                    "supporting_competitor_names": ["C1"],
                                    "confidence": "high"}]
            m_ci.return_value = [{"id": 1, "insight_type": "segment",
                                  "insight_text": "Test", "supporting_evidence_ids": []}]
            m_comp.return_value = {"competitors": [{"name": "C1", "positioning": "P",
                                                     "strengths": [], "weaknesses": []}],
                                   "competitive_gaps": []}
            tracker = MagicMock()
            tracker.__enter__ = Mock(return_value=tracker)
            tracker.__exit__ = Mock(return_value=False)
            m_track.return_value = tracker

            with pytest.raises((RuntimeError, ValueError)):
                run_content_generation_pipeline("r1", creative_gap_id=1, client=mock_client, max_gemini_calls=4)

        assert call_count == 4, f"Expected exactly 4 Gemini calls, got {call_count}"


# ===========================================================================
# STEP 5 — CLAIM RESOLUTION AND APPROVAL
# ===========================================================================

class TestClaimResolutionAndApproval:
    """Test claim resolution workflow and approval blocking."""

    def test_approval_blocked_with_unresolved_claims(self, db_session):
        """Cannot approve a content piece that has unresolved claims_to_verify."""
        from database import (update_content_piece_status, ConsumerInsight,
                               CreativeGap, ContentPiece)

        run_id, ev_ids = seed_run_and_evidence(db_session)

        ci = ConsumerInsight(research_run_id=run_id, insight_type="segment",
                             insight_text="Test", supporting_evidence_ids=ev_ids[:1])
        db_session.add(ci)
        db_session.flush()

        gap = CreativeGap(research_run_id=run_id, gap_title="G1", description="D1",
                          supporting_consumer_insight_ids=[ci.id],
                          supporting_competitor_names=["C1"],
                          supporting_competitive_gap_ids=[], confidence="high")
        db_session.add(gap)
        db_session.flush()

        cp = ContentPiece(
            research_run_id=run_id, creative_gap_id=gap.id,
            content_type="reel_script", platform="instagram",
            title="Claim Test", hook="Hook",
            body=[{"visual": "V", "voiceover": "VO", "on_screen_text": "OST", "duration_seconds": 5}],
            caption="Caption with [VERIFIED CLAIM NEEDED: eco materials]",
            hashtags=[], call_to_action="CTA",
            supporting_consumer_insight_ids=[ci.id],
            supporting_competitor_names=["C1"],
            claims_to_verify=["[VERIFIED CLAIM NEEDED: eco materials]"],
            status="draft"
        )
        db_session.add(cp)
        db_session.commit()

        with pytest.raises(ValueError, match="unresolved claims"):
            update_content_piece_status(cp.id, "approved")

    def test_claim_resolution_then_approval(self, db_session):
        """Resolve claim, then approve should succeed."""
        from database import (resolve_content_claim, update_content_piece_status,
                               ConsumerInsight, CreativeGap, ContentPiece,
                               ContentFeedback)

        run_id, ev_ids = seed_run_and_evidence(db_session)

        ci = ConsumerInsight(research_run_id=run_id, insight_type="segment",
                             insight_text="Test", supporting_evidence_ids=ev_ids[:1])
        db_session.add(ci)
        db_session.flush()

        gap = CreativeGap(research_run_id=run_id, gap_title="G2", description="D2",
                          supporting_consumer_insight_ids=[ci.id],
                          supporting_competitor_names=["C1"],
                          supporting_competitive_gap_ids=[], confidence="high")
        db_session.add(gap)
        db_session.flush()

        cp = ContentPiece(
            research_run_id=run_id, creative_gap_id=gap.id,
            content_type="static_post", platform="instagram",
            title="Resolve Test", hook="Hook",
            body={"visual_direction": "Test [VERIFIED CLAIM NEEDED: eco materials] here"},
            caption="Cap", hashtags=[], call_to_action="CTA",
            supporting_consumer_insight_ids=[ci.id],
            supporting_competitor_names=["C1"],
            claims_to_verify=["[VERIFIED CLAIM NEEDED: eco materials]"],
            status="draft"
        )
        db_session.add(cp)
        db_session.commit()
        piece_id = cp.id

        # Resolve the claim
        result = resolve_content_claim(piece_id, "eco materials", "made with 30% recycled polyester")
        assert result is not None

        # Now approval should work
        approved = update_content_piece_status(piece_id, "approved", reviewer_note="Looks good")
        assert approved["status"] == "approved"

        # Check feedback snapshot exists
        fb = db_session.query(ContentFeedback).filter_by(content_piece_id=piece_id).first()
        assert fb is not None
        assert fb.status == "approved"
        assert fb.snapshot_title == "Resolve Test"

    def test_feedback_snapshot_survives_regeneration(self, db_session):
        """After regeneration (supersede + new piece), old feedback snapshot persists."""
        from database import (ConsumerInsight, CreativeGap, ContentPiece,
                               ContentFeedback, update_content_piece_status)

        run_id, ev_ids = seed_run_and_evidence(db_session)

        ci = ConsumerInsight(research_run_id=run_id, insight_type="segment",
                             insight_text="Test", supporting_evidence_ids=ev_ids[:1])
        db_session.add(ci)
        db_session.flush()

        gap = CreativeGap(research_run_id=run_id, gap_title="G3", description="D3",
                          supporting_consumer_insight_ids=[ci.id],
                          supporting_competitor_names=["C1"],
                          supporting_competitive_gap_ids=[], confidence="high")
        db_session.add(gap)
        db_session.flush()

        # Original content piece — no claims, approve immediately
        cp1 = ContentPiece(
            research_run_id=run_id, creative_gap_id=gap.id,
            content_type="carousel", platform="instagram",
            title="V1 Title", hook="V1 Hook",
            body=[{"slide_number": 1, "headline": "H", "body_text": "B", "visual_direction": "V"}],
            caption="Cap", hashtags=[], call_to_action="CTA",
            supporting_consumer_insight_ids=[ci.id],
            supporting_competitor_names=["C1"],
            claims_to_verify=[],
            status="draft"
        )
        db_session.add(cp1)
        db_session.commit()

        # Reject with a note
        update_content_piece_status(cp1.id, "rejected", reviewer_note="Needs more punch")

        # Simulate regeneration: supersede old, create new
        cp1_obj = db_session.query(ContentPiece).filter_by(id=cp1.id).first()
        cp1_obj.superseded_at = datetime.now(timezone.utc)
        db_session.flush()

        cp2 = ContentPiece(
            research_run_id=run_id, creative_gap_id=gap.id,
            content_type="carousel", platform="instagram",
            title="V2 Title", hook="V2 Hook",
            body=[{"slide_number": 1, "headline": "H2", "body_text": "B2", "visual_direction": "V2"}],
            caption="Cap2", hashtags=[], call_to_action="CTA2",
            supporting_consumer_insight_ids=[ci.id],
            supporting_competitor_names=["C1"],
            claims_to_verify=[],
            status="draft"
        )
        db_session.add(cp2)
        db_session.commit()

        # Original feedback should still exist with snapshot
        fb = db_session.query(ContentFeedback).filter_by(content_piece_id=cp1.id).first()
        assert fb is not None
        assert fb.snapshot_title == "V1 Title"
        assert fb.note == "Needs more punch"


# ===========================================================================
# STEP 6 — SMOKE TESTS: save/get double-run + agent pipeline mocks
# ===========================================================================

class TestDoubleRunSaveGet:
    """Each save_* function run twice for the same run — second must not raise,
    old rows superseded, only new rows returned by default."""

    def test_consumer_insights_double_run(self, db_session):
        from database import save_consumer_insights, get_consumer_insights

        run_id, ev_ids = seed_run_and_evidence(db_session)

        v1 = save_consumer_insights(run_id, [
            {"insight_type": "segment", "insight_text": "V1 insight",
             "supporting_evidence_ids": ev_ids[:1]}
        ])
        assert len(v1) == 1
        v1_id = v1[0]["id"]

        v2 = save_consumer_insights(run_id, [
            {"insight_type": "motivation", "insight_text": "V2 insight updated",
             "supporting_evidence_ids": ev_ids[:2]}
        ])
        assert len(v2) == 1
        v2_id = v2[0]["id"]
        assert v2_id != v1_id

        # Default GET returns only non-superseded
        current = get_consumer_insights(run_id)
        assert len(current) == 1
        assert current[0]["id"] == v2_id

        # Include superseded shows both
        all_insights = get_consumer_insights(run_id, include_superseded=True)
        assert len(all_insights) == 2
        from database import ConsumerInsight
        assert db_session.query(ConsumerInsight).filter_by(id=v1_id).one().superseded_at is not None
        assert db_session.query(ConsumerInsight).filter_by(id=v2_id).one().superseded_at is None

    def test_competitive_intelligence_double_run(self, db_session):
        from database import save_competitive_intelligence, get_competitive_intelligence

        run_id, _ = seed_run_and_evidence(db_session)

        save_competitive_intelligence(run_id, {
            "competitors": [{"name": "A"}], "market_structure": {}, "competitive_gaps": []
        })
        save_competitive_intelligence(run_id, {
            "competitors": [{"name": "B"}], "market_structure": {}, "competitive_gaps": []
        })
        current = get_competitive_intelligence(run_id)
        assert current["competitors"][0]["name"] == "B"
        from database import CompetitiveIntelligence
        rows = db_session.query(CompetitiveIntelligence).filter_by(research_run_id=run_id).order_by(CompetitiveIntelligence.id).all()
        assert len(rows) == 2
        assert rows[0].superseded_at is not None
        assert rows[1].superseded_at is None

    def test_creative_gaps_double_run(self, db_session):
        from database import (save_consumer_insights, save_creative_gaps,
                               get_creative_gaps)

        run_id, ev_ids = seed_run_and_evidence(db_session)
        ci = save_consumer_insights(run_id, [
            {"insight_type": "segment", "insight_text": "Test CI for gaps",
             "supporting_evidence_ids": ev_ids[:1]}
        ])
        ci_id = ci[0]["id"]

        v1 = save_creative_gaps(run_id, [{
            "gap_title": "V1 Gap", "description": "D1",
            "supporting_consumer_insight_ids": [ci_id],
            "supporting_competitor_names": ["C1"],
            "supporting_competitive_gap_ids": [], "confidence": "high"
        }])
        v1_id = v1[0]["id"]

        v2 = save_creative_gaps(run_id, [{
            "gap_title": "V2 Gap", "description": "D2",
            "supporting_consumer_insight_ids": [ci_id],
            "supporting_competitor_names": ["C1"],
            "supporting_competitive_gap_ids": [], "confidence": "medium"
        }])
        v2_id = v2[0]["id"]
        assert v2_id != v1_id

        current = get_creative_gaps(run_id)
        assert len(current) == 1
        assert current[0]["id"] == v2_id
        from database import CreativeGap
        assert db_session.query(CreativeGap).filter_by(id=v1_id).one().superseded_at is not None
        assert db_session.query(CreativeGap).filter_by(id=v2_id).one().superseded_at is None

    def test_opportunity_scores_double_run(self, db_session):
        from database import (save_consumer_insights, save_creative_gaps,
                               save_opportunity_scores, get_opportunity_scores)

        run_id, ev_ids = seed_run_and_evidence(db_session)
        ci = save_consumer_insights(run_id, [
            {"insight_type": "segment", "insight_text": "CI for opp test",
             "supporting_evidence_ids": ev_ids[:1]}
        ])
        ci_id = ci[0]["id"]
        gaps = save_creative_gaps(run_id, [{
            "gap_title": "Gap for opp", "description": "D",
            "supporting_consumer_insight_ids": [ci_id],
            "supporting_competitor_names": ["C1"],
            "supporting_competitive_gap_ids": [], "confidence": "high"
        }])
        gap_id = gaps[0]["id"]

        score_data = {
            "creative_gap_id": gap_id, "rank": 1, "overall_score": 7.0,
            "audience_demand_score": 4, "audience_demand_reasoning": "Strong",
            "competitive_whitespace_score": 3, "competitive_whitespace_reasoning": "Moderate",
            "brand_fit_score": 4, "brand_fit_reasoning": "Good",
            "feasibility_score": 3, "feasibility_reasoning": "Feasible"
        }

        v1 = save_opportunity_scores(run_id, [score_data])
        v1_id = v1[0]["id"]

        v2 = save_opportunity_scores(run_id, [score_data])
        assert len(v2) == 1
        v2_id = v2[0]["id"]
        assert v2_id != v1_id

        current = get_opportunity_scores(run_id)
        assert len(current) == 1
        assert current[0]["id"] == v2_id
        all_scores = get_opportunity_scores(run_id, include_superseded=True)
        assert {score["id"] for score in all_scores} == {v1_id, v2_id}
        from database import OpportunityScore
        assert db_session.query(OpportunityScore).filter_by(id=v1_id).one().superseded_at is not None
        assert db_session.query(OpportunityScore).filter_by(id=v2_id).one().superseded_at is None

    def test_content_pieces_double_run(self, db_session):
        from database import (save_consumer_insights, save_creative_gaps,
                               save_content_pieces, get_content_pieces)

        run_id, ev_ids = seed_run_and_evidence(db_session)
        ci = save_consumer_insights(run_id, [
            {"insight_type": "segment", "insight_text": "CI for content test",
             "supporting_evidence_ids": ev_ids[:1]}
        ])
        ci_id = ci[0]["id"]
        gaps = save_creative_gaps(run_id, [{
            "gap_title": "Gap for content", "description": "D",
            "supporting_consumer_insight_ids": [ci_id],
            "supporting_competitor_names": ["C1"],
            "supporting_competitive_gap_ids": [], "confidence": "high"
        }])
        gap_id = gaps[0]["id"]

        piece_data = {
            "creative_gap_id": gap_id, "content_type": "static_post",
            "platform": "instagram", "title": "T1", "hook": "H1",
            "body": {"visual_direction": "V1"},
            "caption": "Cap1", "hashtags": [], "call_to_action": "CTA1",
            "supporting_consumer_insight_ids": [ci_id],
            "supporting_competitor_names": ["C1"],
            "claims_to_verify": []
        }

        v1 = save_content_pieces(run_id, [piece_data])
        v1_id = v1[0]["id"]

        piece_data["title"] = "T2"
        v2 = save_content_pieces(run_id, [piece_data])
        v2_id = v2[0]["id"]
        assert v2_id != v1_id

        current = get_content_pieces(run_id)
        assert len(current) == 1
        assert current[0]["id"] == v2_id
        all_pieces = get_content_pieces(run_id, include_superseded=True)
        assert {piece["id"] for piece in all_pieces} == {v1_id, v2_id}
        from database import ContentPiece
        assert db_session.query(ContentPiece).filter_by(id=v1_id).one().superseded_at is not None
        assert db_session.query(ContentPiece).filter_by(id=v2_id).one().superseded_at is None


# ===========================================================================
# STEP 6 — Agent pipeline smoke tests (mocked Gemini, real test DB)
# ===========================================================================

class TestAgent2Pipeline:
    """Agent 2 pipeline with mocked Gemini."""

    def test_agent2_runs_with_mock(self, db_session):
        from agent2 import run_consumer_intelligence_pipeline

        run_id, ev_ids = seed_run_and_evidence(db_session, n_evidence=5)

        mock_response_data = {
            "insights": [
                {"insight_type": "segment",
                 "insight_text": "Young consumers in India show strong preference for eco-conscious brands aligned with social media trends",
                 "supporting_evidence_ids": ev_ids[:2]},
                {"insight_type": "motivation",
                 "insight_text": "Price sensitivity combined with aspiration for quality drives purchase decisions among Gen Z buyers",
                 "supporting_evidence_ids": ev_ids[1:3]},
            ]
        }

        mock_response = MagicMock()
        mock_response.text = json.dumps(mock_response_data)
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=100, candidates_token_count=50, total_token_count=150
        )

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch("agent2.track_api_call") as m_track:
            tracker = MagicMock()
            tracker.__enter__ = Mock(return_value=tracker)
            tracker.__exit__ = Mock(return_value=False)
            m_track.return_value = tracker

            result = run_consumer_intelligence_pipeline(run_id, client=mock_client)

        assert len(result) >= 1
        assert result[0]["insight_type"] in ("segment", "motivation", "pain_point", "behavioral_pattern")


class TestAgent3Pipeline:
    """Agent 3 pipeline with mocked Gemini and SerpAPI."""

    def test_agent3_runs_with_mock(self, db_session):
        from agent3 import run_competitive_intelligence_pipeline

        run_id, ev_ids = seed_run_and_evidence(db_session, n_evidence=5)

        # Mock SerpAPI search results
        mock_search_results = [
            {"title": "Nike market report", "link": "https://example.com/nike",
             "snippet": "Nike leads with 30% market share in athletic footwear"}
        ]

        # Mock synthesis response
        synthesis_response = MagicMock()
        synthesis_response.text = json.dumps({
            "competitors": [
                {"name": "Nike", "positioning": "Premium athletic", "strengths": ["brand"], "weaknesses": ["price"],
                 "supporting_evidence_ids": ev_ids[:1]}
            ],
            "market_structure": {"concentration": "Oligopoly", "supporting_evidence_ids": ev_ids[:1]},
            "competitive_gaps": [{"id": 1, "gap": "Sustainability gap", "supporting_evidence_ids": ev_ids[:1]}]
        })
        synthesis_response.usage_metadata = MagicMock(
            prompt_token_count=100, candidates_token_count=50, total_token_count=150
        )

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = synthesis_response

        with patch("agent3.track_api_call") as m_track, \
             patch("agent3.identify_competitors_from_evidence", return_value=["Nike"]), \
             patch("agent3.generate_competitor_queries", return_value=["Nike pricing shoes"]), \
             patch("agent3.search_web_serpapi", return_value=mock_search_results), \
             patch("agent3.extract_evidence_from_doc", return_value={
                 "status": "success",
                 "data": {
                     "claim": "Nike leads in market share",
                     "evidence": "Nike 30% market share in athletic footwear",
                     "source": {"title": "T", "url": "https://example.com", "publisher": "P"},
                     "topic": "competition",
                     "confidence": 0.9
                 }
             }):
            tracker = MagicMock()
            tracker.__enter__ = Mock(return_value=tracker)
            tracker.__exit__ = Mock(return_value=False)
            m_track.return_value = tracker

            result = run_competitive_intelligence_pipeline(run_id, client=mock_client)

        assert "competitors" in result
        assert len(result["competitors"]) >= 1


class TestAgent4Pipeline:
    """Agent 4 pipeline with mocked Gemini."""

    def test_agent4_runs_with_mock(self, db_session):
        from agent4 import run_creative_gaps_pipeline
        from database import save_consumer_insights, save_competitive_intelligence

        run_id, ev_ids = seed_run_and_evidence(db_session, n_evidence=5)

        # Set up Agent 2 output
        ci_saved = save_consumer_insights(run_id, [
            {"insight_type": "segment",
             "insight_text": "Young buyers prefer eco brands in India",
             "supporting_evidence_ids": ev_ids[:2]},
            {"insight_type": "pain_point",
             "insight_text": "Price sensitivity is the top barrier for Gen Z",
             "supporting_evidence_ids": ev_ids[2:4]},
        ])

        # Set up Agent 3 output
        save_competitive_intelligence(run_id, {
            "competitors": [
                {"name": "Nike", "positioning": "Premium", "strengths": ["brand"], "weaknesses": ["price"]}
            ],
            "market_structure": {"type": "oligopoly"},
            "competitive_gaps": [{"id": 1, "description": "Sustainability gap"}]
        })

        ci_ids = [c["id"] for c in ci_saved]

        mock_response_data = {
            "creative_gaps": [
                {"gap_title": "Eco Budget Gap",
                 "description": "Opportunity to serve eco-conscious budget buyers",
                 "supporting_consumer_insight_ids": ci_ids,
                 "supporting_competitor_names": ["Nike"],
                 "supporting_competitive_gap_ids": [1],
                 "confidence": "high"}
            ]
        }

        mock_response = MagicMock()
        mock_response.text = json.dumps(mock_response_data)
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=100, candidates_token_count=50, total_token_count=150
        )

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch("agent4.track_api_call") as m_track:
            tracker = MagicMock()
            tracker.__enter__ = Mock(return_value=tracker)
            tracker.__exit__ = Mock(return_value=False)
            m_track.return_value = tracker

            result = run_creative_gaps_pipeline(run_id, client=mock_client)

        assert len(result) >= 1
        assert result[0]["gap_title"] == "Eco Budget Gap"


class TestOpportunityEnginePipeline:
    """Opportunity Engine pipeline with mocked Gemini."""

    def test_opportunity_engine_runs_with_mock(self, db_session):
        from opportunity_engine import run_opportunity_engine_pipeline
        from database import (save_consumer_insights, save_competitive_intelligence,
                               save_creative_gaps)

        run_id, ev_ids = seed_run_and_evidence(db_session, n_evidence=5)

        ci_saved = save_consumer_insights(run_id, [
            {"insight_type": "segment",
             "insight_text": "Young eco-conscious buyers",
             "supporting_evidence_ids": ev_ids[:2]}
        ])
        ci_ids = [c["id"] for c in ci_saved]

        save_competitive_intelligence(run_id, {
            "competitors": [{"name": "Nike", "positioning": "Premium",
                              "strengths": ["brand"], "weaknesses": ["price"]}],
            "market_structure": {"type": "oligopoly"},
            "competitive_gaps": [{"id": 1, "description": "Sustainability gap"}]
        })

        gaps_saved = save_creative_gaps(run_id, [{
            "gap_title": "Eco Budget Gap", "description": "Budget eco opportunity",
            "supporting_consumer_insight_ids": ci_ids,
            "supporting_competitor_names": ["Nike"],
            "supporting_competitive_gap_ids": [1], "confidence": "high"
        }])
        gap_id = gaps_saved[0]["id"]

        mock_response_data = {
            "opportunity_scores": [{
                "creative_gap_id": gap_id,
                "audience_demand_score": 4, "audience_demand_reasoning": "Strong demand from eco-conscious youth",
                "competitive_whitespace_score": 3, "competitive_whitespace_reasoning": "Moderate competitor presence",
                "brand_fit_score": 4, "brand_fit_reasoning": "Aligns with brand heritage",
                "feasibility_score": 3, "feasibility_reasoning": "Achievable with current supply chain"
            }]
        }

        mock_response = MagicMock()
        mock_response.text = json.dumps(mock_response_data)
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=100, candidates_token_count=50, total_token_count=150
        )

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch("opportunity_engine.track_api_call") as m_track:
            tracker = MagicMock()
            tracker.__enter__ = Mock(return_value=tracker)
            tracker.__exit__ = Mock(return_value=False)
            m_track.return_value = tracker

            result = run_opportunity_engine_pipeline(run_id, client=mock_client)

        assert len(result) >= 1
        assert result[0]["creative_gap_id"] == gap_id


class TestContentGeneratorPipeline:
    """Content Generator pipeline with mocked Gemini."""

    def test_content_generator_runs_with_mock(self, db_session):
        from content_generator import run_content_generation_pipeline
        from database import (save_consumer_insights, save_competitive_intelligence,
                               save_creative_gaps, save_opportunity_scores)

        run_id, ev_ids = seed_run_and_evidence(db_session, n_evidence=5)

        ci_saved = save_consumer_insights(run_id, [
            {"insight_type": "segment",
             "insight_text": "Young eco-conscious buyers prefer value brands",
             "supporting_evidence_ids": ev_ids[:2]}
        ])
        ci_id = ci_saved[0]["id"]

        save_competitive_intelligence(run_id, {
            "competitors": [{"name": "Nike", "positioning": "Premium",
                              "strengths": ["brand equity"], "weaknesses": ["high price"]}],
            "market_structure": {"type": "oligopoly"},
            "competitive_gaps": []
        })

        gaps = save_creative_gaps(run_id, [{
            "gap_title": "Value Eco", "description": "Value eco opportunity",
            "supporting_consumer_insight_ids": [ci_id],
            "supporting_competitor_names": ["Nike"],
            "supporting_competitive_gap_ids": [], "confidence": "high"
        }])
        gap_id = gaps[0]["id"]

        save_opportunity_scores(run_id, [{
            "creative_gap_id": gap_id, "rank": 1, "overall_score": 7.5,
            "audience_demand_score": 4, "audience_demand_reasoning": "Strong",
            "competitive_whitespace_score": 3, "competitive_whitespace_reasoning": "Moderate",
            "brand_fit_score": 4, "brand_fit_reasoning": "Good",
            "feasibility_score": 3, "feasibility_reasoning": "Feasible"
        }])

        # Build a valid content response
        content_response = {
            "content_pieces": [
                {
                    "content_type": "reel_script", "platform": "instagram",
                    "title": "Eco Choice Reel 1", "hook": "Your style your planet",
                    "body": [
                        {"visual": "Close-up shoes", "voiceover": "Your style your planet",
                         "on_screen_text": "Your style", "duration_seconds": 3},
                        {"visual": "Walking shot", "voiceover": "Gen Z wants better choices for tomorrow",
                         "on_screen_text": "Better choices", "duration_seconds": 7},
                        {"visual": "Logo reveal", "voiceover": "Step into what matters",
                         "on_screen_text": "Step forward", "duration_seconds": 5}
                    ],
                    "caption": "Make every step count",
                    "hashtags": ["ecostyle"], "call_to_action": "Explore the collection",
                    "supporting_consumer_insight_ids": [ci_id],
                    "supporting_competitor_names": ["Nike"]
                },
                {
                    "content_type": "reel_script", "platform": "instagram",
                    "title": "Eco Choice Reel 2", "hook": "What if your kicks could change things",
                    "body": [
                        {"visual": "Street scene", "voiceover": "What if your kicks mattered",
                         "on_screen_text": "What if", "duration_seconds": 3},
                        {"visual": "People walking", "voiceover": "Every step tells a story about what you value",
                         "on_screen_text": "Your values", "duration_seconds": 7},
                        {"visual": "Product shot", "voiceover": "Walk the talk",
                         "on_screen_text": "Walk the talk", "duration_seconds": 5}
                    ],
                    "caption": "Walk the talk with every step",
                    "hashtags": ["walkforward"], "call_to_action": "Shop now",
                    "supporting_consumer_insight_ids": [ci_id],
                    "supporting_competitor_names": ["Nike"]
                },
                {
                    "content_type": "carousel", "platform": "instagram",
                    "title": "Eco Carousel", "hook": "The future walks different",
                    "body": [
                        {"slide_number": 1, "headline": "The future walks different", "body_text": "Gen Z is rewriting the rules", "visual_direction": "Bold typography"},
                        {"slide_number": 2, "headline": "Values matter", "body_text": "What you wear says who you are", "visual_direction": "Lifestyle shot"},
                        {"slide_number": 3, "headline": "Step forward", "body_text": "Join the movement", "visual_direction": "Community shot"},
                        {"slide_number": 4, "headline": "Real impact", "body_text": "Every choice counts", "visual_direction": "Nature overlay"},
                        {"slide_number": 5, "headline": "Your move", "body_text": "Make it count", "visual_direction": "CTA slide"}
                    ],
                    "caption": "Swipe to see the future", "hashtags": ["futurewalks"],
                    "call_to_action": "Link in bio",
                    "supporting_consumer_insight_ids": [ci_id],
                    "supporting_competitor_names": ["Nike"]
                },
                {
                    "content_type": "static_post", "platform": "instagram",
                    "title": "Eco Static", "hook": "One step at a time",
                    "body": {"visual_direction": "Minimalist product shot with nature backdrop"},
                    "caption": "One step at a time", "hashtags": ["onestep"],
                    "call_to_action": "Discover more",
                    "supporting_consumer_insight_ids": [ci_id],
                    "supporting_competitor_names": ["Nike"]
                }
            ]
        }

        mock_response = MagicMock()
        mock_response.text = json.dumps(content_response)
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=500, candidates_token_count=800, total_token_count=1300
        )

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch("content_generator.track_api_call") as m_track:
            tracker = MagicMock()
            tracker.__enter__ = Mock(return_value=tracker)
            tracker.__exit__ = Mock(return_value=False)
            m_track.return_value = tracker

            result = run_content_generation_pipeline(
                run_id, creative_gap_id=gap_id, client=mock_client, max_gemini_calls=4
            )

        assert len(result) == 4
        types = [p["content_type"] for p in result]
        assert types.count("reel_script") == 2
        assert types.count("carousel") == 1
        assert types.count("static_post") == 1
