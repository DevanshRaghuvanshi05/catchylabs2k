"""
Step 5 — Claim Resolution & Approval
=====================================
Tests run against TEST_DATABASE_URL (enforced by conftest.py).
All Gemini / SerpAPI calls are mocked.

Scenarios covered:
  1. Approval is blocked when claims_to_verify is non-empty.
  2. PATCH /content/{id}/claims replaces the placeholder in body/caption/hook
     with the verified text and removes the entry from claims_to_verify.
  3. After all claims are resolved, approval is allowed.
  4. content_feedback retains its snapshot of the original piece even after
     the content piece is regenerated (superseded + new piece created).
  5. Resolving a claim that does not exist raises ValueError.
  6. Resolving the last of multiple claims clears the list and unblocks approval.
"""

import uuid
from datetime import datetime, timezone

import pytest

# API key injected by the autouse fixture below so the auth middleware
# does not reject or 500-error these functional tests.
_TEST_KEY = "test-secret-key-abc123"


@pytest.fixture(autouse=True)
def _patch_api_key(monkeypatch):
    """Patch module-level _API_KEY in app.py for every test in this module."""
    import app as app_module  # noqa: PLC0415
    monkeypatch.setattr(app_module, "_API_KEY", _TEST_KEY)

from database import (
    ConsumerInsight,
    ContentFeedback,
    ContentPiece,
    CreativeGap,
    Evidence,
    ResearchRun,
    Source,
    resolve_content_claim,
    save_content_pieces,
    update_content_piece_status,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_run(db_session, brand="Puma"):
    run_id = str(uuid.uuid4())
    run = ResearchRun(id=run_id, brand=brand, market="India",
                      audience="Gen Z", objective="Test")
    db_session.add(run)
    db_session.flush()
    return run_id


def _make_evidence(db_session, run_id, n=1):
    ev_ids = []
    for i in range(n):
        src = Source(url=f"https://example.com/{uuid.uuid4()}",
                     title=f"Source {i}",
                     publisher="TestPub",
                     content_hash=str(uuid.uuid4()))
        db_session.add(src)
        db_session.flush()
        ev = Evidence(research_run_id=run_id,
                      source_id=src.id,
                      claim=f"Claim {i}",
                      excerpt=f"Excerpt {i}",
                      topic="test",
                      confidence=0.9)
        db_session.add(ev)
        db_session.flush()
        ev_ids.append(ev.id)
    return ev_ids


def _make_ci(db_session, run_id, ev_ids):
    ci = ConsumerInsight(research_run_id=run_id,
                         insight_type="segment",
                         insight_text="Test insight",
                         supporting_evidence_ids=ev_ids[:1])
    db_session.add(ci)
    db_session.flush()
    return ci


def _make_gap(db_session, run_id, ci_id):
    gap = CreativeGap(research_run_id=run_id,
                      gap_title="Test Gap",
                      description="Test description",
                      supporting_consumer_insight_ids=[ci_id],
                      supporting_competitor_names=["CompA"],
                      supporting_competitive_gap_ids=[],
                      confidence="high")
    db_session.add(gap)
    db_session.flush()
    return gap


def _make_content_piece(db_session, run_id, gap_id, ci_id,
                        claims=None, status="draft"):
    """
    Insert a ContentPiece directly (bypassing save_content_pieces pipeline
    to stay focused on the claim-resolution behaviour).
    """
    cp = ContentPiece(
        research_run_id=run_id,
        creative_gap_id=gap_id,
        content_type="static_post",
        platform="instagram",
        title="Test Piece",
        hook="Test hook line",
        body={"visual_direction": "Full-bleed product shot"},
        caption="Caption mentioning [VERIFIED CLAIM NEEDED: eco materials].",
        hashtags=[],
        call_to_action="Click the link",
        supporting_consumer_insight_ids=[ci_id],
        supporting_competitor_names=["CompA"],
        claims_to_verify=claims if claims is not None else [],
        status=status,
    )
    db_session.add(cp)
    db_session.commit()
    return cp


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

class TestClaimResolutionAndApproval:
    """Step 5: full lifecycle — block → resolve → approve."""

    # ------------------------------------------------------------------
    # 1. Approval blocked when claims_to_verify is non-empty
    # ------------------------------------------------------------------

    def test_approval_blocked_with_pending_claims(self, db_session):
        """update_content_piece_status('approved') raises ValueError when
        claims_to_verify is non-empty."""
        run_id = _make_run(db_session)
        ev_ids = _make_evidence(db_session, run_id)
        ci = _make_ci(db_session, run_id, ev_ids)
        gap = _make_gap(db_session, run_id, ci.id)

        cp = _make_content_piece(
            db_session, run_id, gap.id, ci.id,
            claims=["[VERIFIED CLAIM NEEDED: eco materials]"],
        )
        piece_id = cp.id
        db_session.expire_all()

        with pytest.raises(ValueError, match="unresolved claims"):
            update_content_piece_status(piece_id, "approved")

    def test_rejection_allowed_with_pending_claims(self, db_session):
        """Rejection should be allowed even when claims_to_verify is non-empty."""
        run_id = _make_run(db_session)
        ev_ids = _make_evidence(db_session, run_id)
        ci = _make_ci(db_session, run_id, ev_ids)
        gap = _make_gap(db_session, run_id, ci.id)

        cp = _make_content_piece(
            db_session, run_id, gap.id, ci.id,
            claims=["[VERIFIED CLAIM NEEDED: durability claim]"],
        )
        piece_id = cp.id
        db_session.expire_all()

        result = update_content_piece_status(piece_id, "rejected",
                                             reviewer_note="Not enough evidence")
        assert result is not None
        assert result["status"] == "rejected"

    # ------------------------------------------------------------------
    # 2. PATCH /content/{id}/claims (resolve_content_claim)
    # ------------------------------------------------------------------

    def test_resolve_claim_removes_entry_from_list(self, db_session):
        """Resolving a claim removes it from claims_to_verify."""
        run_id = _make_run(db_session)
        ev_ids = _make_evidence(db_session, run_id)
        ci = _make_ci(db_session, run_id, ev_ids)
        gap = _make_gap(db_session, run_id, ci.id)

        cp = _make_content_piece(
            db_session, run_id, gap.id, ci.id,
            claims=["[VERIFIED CLAIM NEEDED: eco materials]"],
        )
        piece_id = cp.id
        db_session.expire_all()

        result = resolve_content_claim(
            piece_id,
            original_claim="eco materials",
            verified_replacement="made with 30% recycled polyester",
        )

        assert result is not None
        assert result["claims_to_verify"] == [], (
            "claims_to_verify must be empty after resolving the only claim"
        )

    def test_resolve_claim_replaces_text_in_caption(self, db_session):
        """The placeholder in the caption is replaced with the verified text."""
        run_id = _make_run(db_session)
        ev_ids = _make_evidence(db_session, run_id)
        ci = _make_ci(db_session, run_id, ev_ids)
        gap = _make_gap(db_session, run_id, ci.id)

        cp = _make_content_piece(
            db_session, run_id, gap.id, ci.id,
            claims=["[VERIFIED CLAIM NEEDED: eco materials]"],
        )
        piece_id = cp.id
        db_session.expire_all()

        result = resolve_content_claim(
            piece_id,
            original_claim="eco materials",
            verified_replacement="30% recycled polyester",
        )

        assert "30% recycled polyester" in result["caption"], (
            "Verified replacement must appear in the caption"
        )
        assert "VERIFIED CLAIM NEEDED" not in result["caption"], (
            "Placeholder must be removed from the caption"
        )

    def test_resolve_claim_replaces_text_in_body(self, db_session):
        """The placeholder inside the body JSON is replaced with verified text."""
        run_id = _make_run(db_session)
        ev_ids = _make_evidence(db_session, run_id)
        ci = _make_ci(db_session, run_id, ev_ids)
        gap = _make_gap(db_session, run_id, ci.id)

        # Place placeholder in the body
        cp = ContentPiece(
            research_run_id=run_id,
            creative_gap_id=gap.id,
            content_type="static_post",
            platform="instagram",
            title="Body Test",
            hook="Hook",
            body={"visual_direction": "Shoe shot. [VERIFIED CLAIM NEEDED: material source]"},
            caption="Caption text",
            hashtags=[],
            call_to_action="CTA",
            supporting_consumer_insight_ids=[ci.id],
            supporting_competitor_names=["CompA"],
            claims_to_verify=["[VERIFIED CLAIM NEEDED: material source]"],
            status="draft",
        )
        db_session.add(cp)
        db_session.commit()
        piece_id = cp.id
        db_session.expire_all()

        result = resolve_content_claim(
            piece_id,
            original_claim="material source",
            verified_replacement="sourced from 100% ocean plastic",
        )

        body_text = str(result["body"])
        assert "ocean plastic" in body_text, "Verified text must appear in body"
        assert "VERIFIED CLAIM NEEDED" not in body_text, "Placeholder must be removed from body"

    def test_resolve_nonexistent_claim_raises(self, db_session):
        """Resolving a claim key not present in claims_to_verify raises ValueError."""
        run_id = _make_run(db_session)
        ev_ids = _make_evidence(db_session, run_id)
        ci = _make_ci(db_session, run_id, ev_ids)
        gap = _make_gap(db_session, run_id, ci.id)

        cp = _make_content_piece(
            db_session, run_id, gap.id, ci.id,
            claims=["[VERIFIED CLAIM NEEDED: eco materials]"],
        )
        piece_id = cp.id
        db_session.expire_all()

        with pytest.raises(ValueError):
            resolve_content_claim(
                piece_id,
                original_claim="nonexistent claim key",
                verified_replacement="some replacement",
            )

    # ------------------------------------------------------------------
    # 3. Approval allowed after all claims resolved
    # ------------------------------------------------------------------

    def test_approval_allowed_after_resolution(self, db_session):
        """update_content_piece_status('approved') succeeds once claims_to_verify
        is empty after calling resolve_content_claim."""
        run_id = _make_run(db_session)
        ev_ids = _make_evidence(db_session, run_id)
        ci = _make_ci(db_session, run_id, ev_ids)
        gap = _make_gap(db_session, run_id, ci.id)

        cp = _make_content_piece(
            db_session, run_id, gap.id, ci.id,
            claims=["[VERIFIED CLAIM NEEDED: eco materials]"],
        )
        piece_id = cp.id
        db_session.expire_all()

        # Resolve the claim
        resolve_content_claim(
            piece_id,
            original_claim="eco materials",
            verified_replacement="made with 30% recycled polyester",
        )

        # Now approval must succeed
        result = update_content_piece_status(piece_id, "approved",
                                             reviewer_note="Verified externally")
        assert result["status"] == "approved"

    def test_resolve_multiple_claims_one_by_one(self, db_session):
        """Resolving all claims one-by-one fully clears claims_to_verify."""
        run_id = _make_run(db_session)
        ev_ids = _make_evidence(db_session, run_id)
        ci = _make_ci(db_session, run_id, ev_ids)
        gap = _make_gap(db_session, run_id, ci.id)

        cp = ContentPiece(
            research_run_id=run_id,
            creative_gap_id=gap.id,
            content_type="carousel",
            platform="instagram",
            title="Multi-Claim Piece",
            hook="Hook",
            body=[{"slide_number": 1, "headline": "H",
                   "body_text": "[VERIFIED CLAIM NEEDED: recycled %]",
                   "visual_direction": "V"}],
            caption="Also [VERIFIED CLAIM NEEDED: certification body].",
            hashtags=[],
            call_to_action="CTA",
            supporting_consumer_insight_ids=[ci.id],
            supporting_competitor_names=["CompA"],
            claims_to_verify=[
                "[VERIFIED CLAIM NEEDED: recycled %]",
                "[VERIFIED CLAIM NEEDED: certification body]",
            ],
            status="draft",
        )
        db_session.add(cp)
        db_session.commit()
        piece_id = cp.id
        db_session.expire_all()

        # Resolve first claim
        r1 = resolve_content_claim(piece_id, "recycled %", "uses 40% recycled materials")
        assert len(r1["claims_to_verify"]) == 1, "One claim must remain"

        # Still blocked
        with pytest.raises(ValueError, match="unresolved claims"):
            update_content_piece_status(piece_id, "approved")

        # Resolve second claim
        r2 = resolve_content_claim(piece_id, "certification body", "certified by GreenLabel")
        assert r2["claims_to_verify"] == [], "All claims must be resolved"

        # Approval unblocked
        approved = update_content_piece_status(piece_id, "approved",
                                               reviewer_note="All verified")
        assert approved["status"] == "approved"

    # ------------------------------------------------------------------
    # 4. content_feedback snapshot survives regeneration
    # ------------------------------------------------------------------

    def test_feedback_snapshot_survives_regeneration(self, db_session):
        """ContentFeedback record retains the snapshot of the original piece
        after the piece is superseded and a new one is created."""
        run_id = _make_run(db_session)
        ev_ids = _make_evidence(db_session, run_id)
        ci = _make_ci(db_session, run_id, ev_ids)
        gap = _make_gap(db_session, run_id, ci.id)

        # Original piece — no claims, so we can approve immediately
        cp1 = _make_content_piece(
            db_session, run_id, gap.id, ci.id,
            claims=[],
        )
        piece_id_v1 = cp1.id
        db_session.expire_all()

        # Reject with a reviewer note (creates feedback snapshot)
        update_content_piece_status(
            piece_id_v1, "rejected",
            reviewer_note="Needs stronger hook"
        )

        # Simulate regeneration: supersede the old piece
        db_session.query(ContentPiece).filter_by(id=piece_id_v1).update(
            {"superseded_at": datetime.now(timezone.utc)}
        )
        db_session.flush()

        # Create new piece (v2)
        cp2 = ContentPiece(
            research_run_id=run_id,
            creative_gap_id=gap.id,
            content_type="static_post",
            platform="instagram",
            title="Regenerated Piece v2",
            hook="New powerful hook",
            body={"visual_direction": "New visual"},
            caption="New caption",
            hashtags=[],
            call_to_action="New CTA",
            supporting_consumer_insight_ids=[ci.id],
            supporting_competitor_names=["CompA"],
            claims_to_verify=[],
            status="draft",
        )
        db_session.add(cp2)
        db_session.commit()

        # The feedback for v1 must still exist with the correct snapshot
        fb = (
            db_session.query(ContentFeedback)
            .filter_by(content_piece_id=piece_id_v1)
            .first()
        )
        assert fb is not None, "ContentFeedback row must exist for the original piece"
        assert fb.snapshot_title == "Test Piece", (
            "snapshot_title must reflect the original piece title"
        )
        assert fb.note == "Needs stronger hook", (
            "Reviewer note must be preserved in the feedback snapshot"
        )
        assert fb.status == "rejected"

        # The FK in ContentFeedback is SET NULL on delete, not CASCADE, so
        # even if cp1 were deleted the fb row would survive.  Here we verify
        # the snapshot data is intact with the piece still present.
        assert fb.snapshot_body is not None

    def test_feedback_fk_set_null_on_piece_supersede(self, db_session):
        """ContentFeedback survives even when content_piece_id is NULL
        (simulates a hard-delete scenario for the FK SET NULL path)."""
        run_id = _make_run(db_session)
        ev_ids = _make_evidence(db_session, run_id)
        ci = _make_ci(db_session, run_id, ev_ids)
        gap = _make_gap(db_session, run_id, ci.id)

        cp = _make_content_piece(db_session, run_id, gap.id, ci.id, claims=[])
        piece_id = cp.id
        db_session.expire_all()

        update_content_piece_status(piece_id, "rejected",
                                    reviewer_note="Poor quality")

        # Verify feedback row exists
        fb_before = (
            db_session.query(ContentFeedback)
            .filter_by(content_piece_id=piece_id)
            .first()
        )
        assert fb_before is not None
        assert fb_before.snapshot_title == "Test Piece"

        # Null out the FK manually (simulating SET NULL behaviour)
        db_session.query(ContentFeedback).filter_by(
            content_piece_id=piece_id
        ).update({"content_piece_id": None})
        db_session.commit()

        # Feedback row must still exist with its snapshot
        fb_after = (
            db_session.query(ContentFeedback)
            .filter_by(id=fb_before.id)
            .first()
        )
        assert fb_after is not None
        assert fb_after.content_piece_id is None
        assert fb_after.snapshot_title == "Test Piece"
        assert fb_after.note == "Poor quality"

    # ------------------------------------------------------------------
    # 5. API-layer contract via FastAPI TestClient
    # ------------------------------------------------------------------

    def test_patch_claims_endpoint_resolves_and_clears(self, db_session):
        """
        PATCH /content/{id}/claims → placeholder replaced, entry removed,
        approval unblocked.

        Uses FastAPI TestClient so we also exercise the HTTP layer end-to-end.
        """
        from fastapi.testclient import TestClient
        from unittest.mock import patch as mock_patch
        from app import app

        run_id = _make_run(db_session)
        ev_ids = _make_evidence(db_session, run_id)
        ci = _make_ci(db_session, run_id, ev_ids)
        gap = _make_gap(db_session, run_id, ci.id)

        cp = _make_content_piece(
            db_session, run_id, gap.id, ci.id,
            claims=["[VERIFIED CLAIM NEEDED: eco materials]"],
        )
        piece_id = cp.id
        db_session.expire_all()

        _HEADERS = {"X-API-Key": _TEST_KEY}
        client = TestClient(app, raise_server_exceptions=True)

        # 1. PATCH /content/{id}/claims
        resp = client.patch(
            f"/content/{piece_id}/claims",
            json={
                "original_claim": "eco materials",
                "verified_replacement": "made with 30% recycled polyester",
            },
            headers=_HEADERS,
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["claims_to_verify"] == [], (
            "claims_to_verify must be empty after PATCH /claims"
        )
        assert "30% recycled polyester" in data["caption"]

        # 2. PATCH /content/{id}/status → approved (must succeed now)
        resp2 = client.patch(
            f"/content/{piece_id}/status",
            json={"status": "approved", "reviewer_note": "Verified"},
            headers=_HEADERS,
        )
        assert resp2.status_code == 200, resp2.text
        assert resp2.json()["status"] == "approved"

    def test_patch_status_blocked_returns_400(self, db_session):
        """PATCH /content/{id}/status with unresolved claims returns HTTP 400."""
        from fastapi.testclient import TestClient
        from app import app

        run_id = _make_run(db_session)
        ev_ids = _make_evidence(db_session, run_id)
        ci = _make_ci(db_session, run_id, ev_ids)
        gap = _make_gap(db_session, run_id, ci.id)

        cp = _make_content_piece(
            db_session, run_id, gap.id, ci.id,
            claims=["[VERIFIED CLAIM NEEDED: material source]"],
        )
        piece_id = cp.id
        db_session.expire_all()

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.patch(
            f"/content/{piece_id}/status",
            json={"status": "approved"},
            headers={"X-API-Key": _TEST_KEY},
        )
        assert resp.status_code == 400, resp.text
        assert "unresolved claims" in resp.json()["detail"].lower()
