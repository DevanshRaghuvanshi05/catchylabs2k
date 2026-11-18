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
Test script for Async Agent 3 (Competitor Intelligence):
1. Pre-validation: POST with invalid ID returns 404
2. Enqueueing: POST /research/{run_id}/competitor-intelligence returns job_id & queued
3. Initial Status: GET /research/status/{job_id} confirms job_type="competitive_intelligence"
4. Worker Execution: SimpleWorker processes targeted queries, SerpAPI searches, extraction, and synthesis
5. Final Status: GET /research/status/{job_id} confirms status="finished"
6. Database Retrieval: GET /research/{run_id}/competitor-intelligence
7. Numerical Grounding Audit: Audits all statistics in the output against cited evidence
"""

import json
import logging
from fastapi.testclient import TestClient
from rq.worker import SimpleWorker

from app import app
from config import get_redis_connection
from database import get_run_with_evidence, get_quota_status
from agent3 import extract_statistical_figures, verify_text_numerical_grounding

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TestAsyncAgent3")

TEST_RUN_ID = "eb892e4a-4c06-4627-93fc-1b4caa7fc7ce"
INVALID_RUN_ID = "00000000-0000-0000-0000-000000000000"

def main():
    client = TestClient(app)
    redis_conn = get_redis_connection()

    quota_before = get_quota_status().get("serpapi", {})
    serpapi_before = quota_before.get("calls_all_time", 0)

    print("=" * 80)
    print("STEP 1: Testing Pre-Validation (404 on Invalid Run ID)")
    print("=" * 80)
    resp_404 = client.post(f"/research/{INVALID_RUN_ID}/competitor-intelligence")
    print(f"POST /research/{INVALID_RUN_ID}/competitor-intelligence -> Status: {resp_404.status_code}")
    assert resp_404.status_code == 404, f"Expected 404, got {resp_404.status_code}"
    print(f"Detail: {resp_404.json().get('detail')}")
    print("[PASS] 404 correctly returned without enqueuing.\n")

    print("=" * 80)
    print(f"STEP 2: Enqueuing Competitor Intelligence Job for Run: {TEST_RUN_ID}")
    print("=" * 80)
    resp_post = client.post(f"/research/{TEST_RUN_ID}/competitor-intelligence")
    print(f"POST /research/{TEST_RUN_ID}/competitor-intelligence -> Status: {resp_post.status_code}")
    assert resp_post.status_code == 200, f"Expected 200, got {resp_post.status_code}"
    post_data = resp_post.json()
    job_id = post_data.get("job_id")
    job_status = post_data.get("status")
    print(f"Response: {json.dumps(post_data, indent=2)}")
    assert job_id, "No job_id returned"
    assert job_status == "queued"
    print(f"[PASS] Successfully enqueued with Job ID: {job_id}\n")

    print("=" * 80)
    print(f"STEP 3: Checking Initial Job Status for: {job_id}")
    print("=" * 80)
    resp_status = client.get(f"/research/status/{job_id}")
    print(f"GET /research/status/{job_id} -> Status: {resp_status.status_code}")
    assert resp_status.status_code == 200
    status_data = resp_status.json()
    print(f"Status Data: {json.dumps(status_data, indent=2)}")
    assert status_data.get("job_type") == "competitive_intelligence"
    assert status_data.get("status") in ["queued", "started"]
    print(f"[PASS] Job type correctly recognized as '{status_data.get('job_type')}'\n")

    print("=" * 80)
    print("STEP 4: Executing Worker on 'research_jobs' queue (burst=True)")
    print("=" * 80)
    print("Worker running Agent 3 targeted research, SerpAPI calls, and synthesis...")
    worker = SimpleWorker(["research_jobs"], connection=redis_conn)
    worker.work(burst=True)
    print("Worker finished processing job.\n")

    print("=" * 80)
    print(f"STEP 5: Polling Final Job Status for: {job_id}")
    print("=" * 80)
    resp_final = client.get(f"/research/status/{job_id}")
    print(f"GET /research/status/{job_id} -> Status: {resp_final.status_code}")
    assert resp_final.status_code == 200
    final_data = resp_final.json()
    print(f"Final Status: status='{final_data.get('status')}', job_type='{final_data.get('job_type')}'")
    assert final_data.get("status") == "finished", f"Expected finished, got {final_data.get('status')}: {final_data.get('error')}"
    assert final_data.get("job_type") == "competitive_intelligence"
    print("[PASS] Background job transitioned to 'finished'.\n")

    print("=" * 80)
    print(f"STEP 6: Fetching Saved Competitor Intelligence from Database")
    print("=" * 80)
    resp_get = client.get(f"/research/{TEST_RUN_ID}/competitor-intelligence")
    print(f"GET /research/{TEST_RUN_ID}/competitor-intelligence -> Status: {resp_get.status_code}")
    assert resp_get.status_code == 200
    comp_report = resp_get.json()
    print(json.dumps(comp_report, indent=2))

    quota_after = get_quota_status().get("serpapi", {})
    serpapi_after = quota_after.get("calls_all_time", 0)
    serpapi_used = serpapi_after - serpapi_before
    print(f"\n[SerpAPI Quota Tracking] Calls before: {serpapi_before} | Calls after: {serpapi_after} | Additional calls used: {serpapi_used}")

    print("\n" + "=" * 80)
    print("STEP 7: Automated Evidence-Citation & Numerical Grounding Audit")
    print("=" * 80)
    run_info = get_run_with_evidence(TEST_RUN_ID)
    ev_map = {e["id"]: e for e in run_info["evidence"]}
    all_valid_ids = set(ev_map.keys())

    # Audit Competitors
    print("--- Auditing Competitors ---")
    for comp in comp_report.get("competitors", []):
        cname = comp["name"]
        cids = comp["supporting_evidence_ids"]
        print(f"\nCompetitor: '{cname}' | Citing IDs: {cids}")
        assert all(cid in all_valid_ids for cid in cids), f"Hallucinated evidence ID in competitor '{cname}'!"
        full_text = f"{comp.get('positioning', '')} {' '.join(comp.get('strengths', []))} {' '.join(comp.get('weaknesses', []))}"
        stats = extract_statistical_figures(full_text)
        print(f"  Extracted Stats/Figures: {[s['raw'] for s in stats]}")
        num_errs = verify_text_numerical_grounding(full_text, cids, ev_map, f"Competitor {cname}")
        assert not num_errs, f"Numerical grounding failed for '{cname}': {num_errs}"
        print("  [PASS] 100% grounded in cited evidence.")

    # Audit Market Structure
    print("\n--- Auditing Market Structure ---")
    m_struct = comp_report.get("market_structure", {})
    m_conc = m_struct.get("concentration", "")
    m_ids = m_struct.get("supporting_evidence_ids", [])
    print(f"Concentration: \"{m_conc}\" | Citing IDs: {m_ids}")
    assert all(cid in all_valid_ids for cid in m_ids), f"Hallucinated ID in market_structure!"
    m_stats = extract_statistical_figures(m_conc)
    print(f"  Extracted Stats/Figures: {[s['raw'] for s in m_stats]}")
    m_errs = verify_text_numerical_grounding(m_conc, m_ids, ev_map, "market_structure")
    assert not m_errs, f"Numerical grounding failed for market_structure: {m_errs}"
    print("  [PASS] 100% grounded in cited evidence.")

    # Audit Competitive Gaps
    print("\n--- Auditing Competitive Gaps ---")
    for idx, gap_item in enumerate(comp_report.get("competitive_gaps", []), 1):
        gtext = gap_item.get("gap", "")
        gids = gap_item.get("supporting_evidence_ids", [])
        print(f"\nGap #{idx}: \"{gtext}\" | Citing IDs: {gids}")
        assert all(cid in all_valid_ids for cid in gids), f"Hallucinated ID in gap #{idx}!"
        g_stats = extract_statistical_figures(gtext)
        print(f"  Extracted Stats/Figures: {[s['raw'] for s in g_stats]}")
        g_errs = verify_text_numerical_grounding(gtext, gids, ev_map, f"Gap #{idx}")
        assert not g_errs, f"Numerical grounding failed for gap #{idx}: {g_errs}"
        print("  [PASS] 100% grounded in cited evidence.")

    print("\n" + "=" * 80)
    print("ALL AGENT 3 ASYNC & EVIDENCE-GROUNDING TESTS PASSED PERFECTLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
