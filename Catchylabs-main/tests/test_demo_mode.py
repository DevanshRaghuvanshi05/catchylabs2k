"""
Tests for demo mode functionality, GET /demo/status, and snapshot export safety.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

import app as app_module
import database as db
from export_demo_snapshot import export_snapshot

VALID_KEY = "test-secret-key-xyz789"


@pytest.fixture(autouse=True)
def _patch_api_key(monkeypatch):
    """Patch the module-level API key for all tests."""
    monkeypatch.setattr(app_module, "_API_KEY", VALID_KEY)


@pytest.fixture()
def client():
    return TestClient(app_module.app, raise_server_exceptions=False)


@pytest.fixture()
def seeded_run_id(db_session):
    """Insert a minimal ResearchRun row and roll back after test."""
    run_id = str(uuid.uuid4())
    run = db.ResearchRun(
        id=run_id,
        brand="DemoBrand",
        market="India",
        audience="Youth",
        objective="Demo Objective",
        created_at=datetime.now(timezone.utc),
    )
    db_session.add(run)
    db_session.flush()

    source = db.Source(
        url="https://example.com/demo-source",
        title="Demo Source Title",
        publisher="Demo Publisher",
        content_hash="demohash12345",
    )
    db_session.add(source)
    db_session.flush()

    evidence = db.Evidence(
        research_run_id=run_id,
        source_id=source.id,
        claim="Market grew by 15% in 2025.",
        excerpt="Excerpt text.",
        topic="growth",
        confidence=0.85,
    )
    db_session.add(evidence)
    db_session.flush()

    return run_id


class TestDemoModeDisabled:
    """Tests when DEMO_MODE is off (default)."""

    def test_post_not_blocked_by_demo_mode(self, client, monkeypatch):
        monkeypatch.setattr(app_module, "_DEMO_MODE", False)
        monkeypatch.delenv("DEMO_MODE", raising=False)

        # POST without body returns 422 Unprocessable Entity (not 403 Forbidden)
        resp = client.post(
            "/research",
            headers={"X-API-Key": VALID_KEY},
            json={},
        )
        assert resp.status_code != 403
        assert "Disabled in demo mode" not in resp.text

    def test_patch_not_blocked_by_demo_mode(self, client, monkeypatch):
        monkeypatch.setattr(app_module, "_DEMO_MODE", False)
        monkeypatch.delenv("DEMO_MODE", raising=False)

        resp = client.patch(
            "/content/999/status",
            headers={"X-API-Key": VALID_KEY},
            json={"status": "approved"},
        )
        assert resp.status_code != 403
        assert "Disabled in demo mode" not in resp.text

    def test_demo_status_reports_demo_mode_false(self, client, monkeypatch, seeded_run_id):
        monkeypatch.setattr(app_module, "_DEMO_MODE", False)
        monkeypatch.delenv("DEMO_MODE", raising=False)

        resp = client.get("/demo/status", headers={"X-API-Key": VALID_KEY})
        assert resp.status_code == 200
        body = resp.json()
        assert body["demo_mode"] is False
        assert body["number_of_runs"] >= 1
        assert body["id"] == seeded_run_id
        assert body["brand"] == "DemoBrand"


class TestDemoModeEnabled:
    """Tests when DEMO_MODE is on (1)."""

    def test_every_post_endpoint_returns_403(self, client, monkeypatch):
        monkeypatch.setattr(app_module, "_DEMO_MODE", True)
        monkeypatch.setenv("DEMO_MODE", "1")

        post_endpoints = [
            "/research",
            "/research/any-id/consumer-intelligence",
            "/research/any-id/competitor-intelligence",
            "/research/any-id/creative-gaps",
            "/research/any-id/opportunities",
            "/research/any-id/content",
        ]
        for endpoint in post_endpoints:
            resp = client.post(
                endpoint,
                headers={"X-API-Key": VALID_KEY},
                json={"brand": "Test"},
            )
            assert resp.status_code == 403, f"Expected 403 for POST {endpoint}, got {resp.status_code}"
            assert resp.json()["detail"] == "Disabled in demo mode"

    def test_every_patch_endpoint_returns_403(self, client, monkeypatch):
        monkeypatch.setattr(app_module, "_DEMO_MODE", True)
        monkeypatch.setenv("DEMO_MODE", "1")

        patch_endpoints = [
            "/content/1/status",
            "/content/1/claims",
        ]
        for endpoint in patch_endpoints:
            resp = client.patch(
                endpoint,
                headers={"X-API-Key": VALID_KEY},
                json={"status": "approved"},
            )
            assert resp.status_code == 403, f"Expected 403 for PATCH {endpoint}, got {resp.status_code}"
            assert resp.json()["detail"] == "Disabled in demo mode"

    def test_get_endpoints_keep_working(self, client, monkeypatch, seeded_run_id):
        monkeypatch.setattr(app_module, "_DEMO_MODE", True)
        monkeypatch.setenv("DEMO_MODE", "1")

        # GET /health
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

        # GET /demo/status
        resp = client.get("/demo/status", headers={"X-API-Key": VALID_KEY})
        assert resp.status_code == 200
        assert resp.json()["demo_mode"] is True

        # GET /runs
        resp = client.get("/runs", headers={"X-API-Key": VALID_KEY})
        assert resp.status_code == 200

        # GET /research/{run_id}/full
        resp = client.get(f"/research/{seeded_run_id}/full", headers={"X-API-Key": VALID_KEY})
        assert resp.status_code == 200


class TestDemoStatusEndpoint:
    """Tests for GET /demo/status authorization and data shape."""

    def test_missing_key_returns_401(self, client):
        resp = client.get("/demo/status")
        assert resp.status_code == 401

    def test_wrong_key_returns_401(self, client):
        resp = client.get("/demo/status", headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 401

    def test_correct_key_returns_recent_run_info(self, client, seeded_run_id):
        resp = client.get("/demo/status", headers={"X-API-Key": VALID_KEY})
        assert resp.status_code == 200
        data = resp.json()
        assert "demo_mode" in data
        assert "number_of_runs" in data
        assert data["number_of_runs"] >= 1
        assert data["id"] == seeded_run_id
        assert data["brand"] == "DemoBrand"


class TestExportDemoSnapshot:
    """Tests for export_demo_snapshot.py."""

    def test_refuses_to_run_without_run_id(self):
        with pytest.raises(ValueError, match="Missing required --run-id"):
            export_snapshot("")

        with pytest.raises(ValueError, match="Missing required --run-id"):
            export_snapshot("   ")

    def test_snapshot_exported_and_contains_no_secrets(self, seeded_run_id, tmp_path):
        target_file = str(tmp_path / "test_snapshot.json")
        out_file = export_snapshot(seeded_run_id, target_file)
        assert os.path.exists(out_file)

        with open(out_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Secret patterns check
        secret_patterns = [
            r"postgres(?:ql)?://",
            r"redis[s]?://",
            r"AIza",
            r"\bsb[_-]",
            r"ghp_[a-zA-Z0-9]+",
        ]
        for pat in secret_patterns:
            assert not re.search(pat, content, re.IGNORECASE), f"Found potential secret matching {pat}"

        # Confirm data shape
        data = json.loads(content)
        assert data["brief"]["id"] == seeded_run_id
        assert data["brief"]["brand"] == "DemoBrand"
        assert len(data["evidence"]) >= 1
