#!/usr/bin/env python3
import sys
from pathlib import Path

if '--live' not in sys.argv[1:]:
    raise SystemExit('This legacy script can access live services or databases. Re-run with --live to confirm.')
sys.argv.remove('--live')
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

"""
Test FastAPI endpoints for Agent 2 Consumer Intelligence:
- GET /research/{run_id}/consumer-intelligence
- GET /research/{invalid_id}/consumer-intelligence
- POST /research/{invalid_id}/consumer-intelligence
"""

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

TEST_RUN_ID = "eb892e4a-4c06-4627-93fc-1b4caa7fc7ce"
INVALID_RUN_ID = "00000000-0000-0000-0000-000000000000"

def test_endpoints():
    print("Testing GET /research/{run_id}/consumer-intelligence with valid run...")
    resp = client.get(f"/research/{TEST_RUN_ID}/consumer-intelligence")
    print(f"Status Code: {resp.status_code}")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()
    print(f"Total Insights: {data.get('total_insights')}")
    assert data.get("total_insights") > 0, "Expected at least 1 insight"
    insights = data.get("insights", [])
    print(f"Sample Insight: {insights[0]}")
    assert "insight_type" in insights[0]
    assert "supporting_evidence_ids" in insights[0]
    assert len(insights[0]["supporting_evidence_ids"]) > 0

    print("\nTesting GET /research/{invalid_id}/consumer-intelligence...")
    resp_invalid = client.get(f"/research/{INVALID_RUN_ID}/consumer-intelligence")
    print(f"Status Code: {resp_invalid.status_code}")
    assert resp_invalid.status_code == 404, f"Expected 404, got {resp_invalid.status_code}"

    print("\nTesting POST /research/{invalid_id}/consumer-intelligence...")
    resp_post_invalid = client.post(f"/research/{INVALID_RUN_ID}/consumer-intelligence")
    print(f"Status Code: {resp_post_invalid.status_code}")
    assert resp_post_invalid.status_code == 404, f"Expected 404, got {resp_post_invalid.status_code}"

    print("\nALL ENDPOINT TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_endpoints()
