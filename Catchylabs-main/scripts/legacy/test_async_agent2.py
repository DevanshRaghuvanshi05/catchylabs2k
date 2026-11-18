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
Test script for Async Agent 2 (Consumer Intelligence):
1. Enqueues job via POST /research/{run_id}/consumer-intelligence
2. Verifies immediate response: {job_id, status: 'queued'}
3. Polls GET /research/status/{job_id} showing job_type and queued state
4. Executes worker on 'research_jobs' queue (burst mode)
5. Verifies GET /research/status/{job_id} transitions to finished
6. Verifies GET /research/{run_id}/consumer-intelligence returns saved insights
7. Verifies 404 for non-existent run_id
"""

import time
import json
import logging
from fastapi.testclient import TestClient
from rq.worker import SimpleWorker

from app import app
from config import get_redis_connection
from test_grounding import extract_statistical_figures, verify_numerical_grounding
from database import get_run_with_evidence

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TestAsyncAgent2")

TEST_RUN_ID = "eb892e4a-4c06-4627-93fc-1b4caa7fc7ce"
INVALID_RUN_ID = "00000000-0000-0000-0000-000000000000"

def main():
    client = TestClient(app)
    redis_conn = get_redis_connection()

    print("=" * 80)
    print("STEP 1: Testing Pre-Validation (404 on Invalid Run ID)")
    print("=" * 80)
    resp_404 = client.post(f"/research/{INVALID_RUN_ID}/consumer-intelligence")
    print(f"POST /research/{INVALID_RUN_ID}/consumer-intelligence -> Status: {resp_404.status_code}")
    assert resp_404.status_code == 404, f"Expected 404, got {resp_404.status_code}"
    print(f"Detail: {resp_404.json().get('detail')}")
    print("[PASS] 404 correctly returned without enqueuing job.\n")

    print("=" * 80)
    print(f"STEP 2: Enqueuing Async Consumer Intelligence Job for Run: {TEST_RUN_ID}")
    print("=" * 80)
    resp_post = client.post(f"/research/{TEST_RUN_ID}/consumer-intelligence")
    print(f"POST /research/{TEST_RUN_ID}/consumer-intelligence -> Status: {resp_post.status_code}")
    assert resp_post.status_code == 200, f"Expected 200, got {resp_post.status_code}"
    post_data = resp_post.json()
    job_id = post_data.get("job_id")
    job_status = post_data.get("status")
    print(f"Response: {json.dumps(post_data, indent=2)}")
    assert job_id, "No job_id returned"
    assert job_status == "queued", f"Expected 'queued', got {job_status}"
    print(f"[PASS] Job successfully enqueued with Job ID: {job_id}\n")

    print("=" * 80)
    print(f"STEP 3: Checking Initial Job Status for: {job_id}")
    print("=" * 80)
    resp_status = client.get(f"/research/status/{job_id}")
    print(f"GET /research/status/{job_id} -> Status: {resp_status.status_code}")
    assert resp_status.status_code == 200
    status_data = resp_status.json()
    print(f"Status Data: {json.dumps(status_data, indent=2)}")
    assert status_data.get("job_type") == "consumer_intelligence", f"Expected job_type='consumer_intelligence', got {status_data.get('job_type')}"
    assert status_data.get("status") in ["queued", "started"]
    print(f"[PASS] Job type correctly recognized as '{status_data.get('job_type')}' with status '{status_data.get('status')}'.\n")

    print("=" * 80)
    print(f"STEP 4: Executing Worker on 'research_jobs' queue (burst=True)")
    print("=" * 80)
    print("Worker picking up job from queue and executing Agent 2 synthesis...")
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
    print(f"Final Status Data: {json.dumps(final_data, indent=2)}")
    assert final_data.get("status") == "finished", f"Expected 'finished', got {final_data.get('status')}"
    assert final_data.get("job_type") == "consumer_intelligence"
    assert final_data.get("research_run_id") == TEST_RUN_ID
    result = final_data.get("result", {})
    assert result.get("total_insights", 0) > 0
    print(f"[PASS] Job status transitioned to 'finished' with {result.get('total_insights')} insights synthesized.\n")

    print("=" * 80)
    print(f"STEP 6: Fetching Saved Insights via GET /research/{TEST_RUN_ID}/consumer-intelligence")
    print("=" * 80)
    resp_get = client.get(f"/research/{TEST_RUN_ID}/consumer-intelligence")
    print(f"GET /research/{TEST_RUN_ID}/consumer-intelligence -> Status: {resp_get.status_code}")
    assert resp_get.status_code == 200
    get_data = resp_get.json()
    insights = get_data.get("insights", [])
    print(f"Total Insights in DB: {len(insights)}")
    assert len(insights) > 0

    # Grounding check on all saved insights
    run_info = get_run_with_evidence(TEST_RUN_ID)
    ev_map = {e["id"]: e for e in run_info["evidence"]}

    print("\nAuditing numerical and citation grounding of all fetched insights:")
    for idx, ins in enumerate(insights, 1):
        itype = ins["insight_type"].upper()
        c_ids = ins["supporting_evidence_ids"]
        text = ins["insight_text"]
        stats = extract_statistical_figures(text)
        stat_strs = [s["raw"] for s in stats]

        cited_evs = [ev_map[cid] for cid in c_ids if cid in ev_map]
        num_errs = verify_numerical_grounding(text, cited_evs)

        print(f"\n[Insight #{idx}] [{itype}] (Citing IDs: {c_ids})")
        print(f"  Text: {text}")
        print(f"  Numbers/Stats: {stat_strs}")
        if num_errs:
            print(f"  [FAIL] Errors: {num_errs}")
            raise AssertionError(f"Insight #{idx} failed numerical grounding: {num_errs}")
        else:
            print(f"  [PASS] All citations and figures 100% verified against cited evidence.")

    print("\n" + "=" * 80)
    print("ALL ASYNC AGENT 2 TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
