import sys
from pathlib import Path

if '--live' not in sys.argv[1:]:
    raise SystemExit('This legacy script can access live services or databases. Re-run with --live to confirm.')
sys.argv.remove('--live')
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

"""
Phase 3 Verification Script - Raw Payload & Status Log Output
--------------------------------------------------------------
Executes POST /research, polls GET /research/status/{job_id}, and fetches GET /research/{run_id}.
Prints RAW JSON responses for every step.
"""

import sys
import time
import json
import logging
from fastapi.testclient import TestClient
from app import app
from database import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("Phase3Verifier")

client = TestClient(app)


def test_async_workflow():
    print("=" * 70)
    print("  PHASE 3 VERIFICATION: RAW PAYLOAD & STATUS LOG EXECUTION")
    print("=" * 70)

    # Brief payload
    brief = {
        "brand": "Puma",
        "market": "India",
        "audience": "Gen Z",
        "objective": "Find opportunities for sustainable running shoes",
        "max_documents": 4
    }

    # 1. POST /research (Enqueue Job)
    print("\n--- [1] POST /research Request & Response ---")
    start_time = time.time()
    r_post = client.post("/research", json=brief)
    elapsed_ms = (time.time() - start_time) * 1000
    
    print(f"Response Code: {r_post.status_code}")
    print(f"Elapsed Time : {elapsed_ms:.2f} ms (Target: < 1000 ms)")
    print(f"Raw JSON Response:\n{json.dumps(r_post.json(), indent=2)}")

    assert r_post.status_code == 200, f"Expected HTTP 200, got {r_post.status_code}"
    post_data = r_post.json()
    assert "job_id" in post_data, "Response missing job_id"
    assert post_data["status"] == "queued", f"Expected status 'queued', got {post_data['status']}"
    assert elapsed_ms < 1000, f"POST /research took too long ({elapsed_ms:.2f} ms)"

    job_id = post_data["job_id"]

    # 2. Poll GET /research/status/{job_id}
    print(f"\n--- [2] GET /research/status/{job_id} Status Transitions ---")
    max_polls = 40
    poll_interval = 2
    finished = False
    run_id = None
    poll_history = []

    for attempt in range(1, max_polls + 1):
        r_status = client.get(f"/research/status/{job_id}")
        assert r_status.status_code == 200, f"Status poll returned HTTP {r_status.status_code}"
        status_data = r_status.json()
        current_state = status_data.get("status")
        
        poll_history.append((attempt, current_state, status_data))
        print(f"Poll #{attempt} ({attempt * poll_interval}s): Status = '{current_state}'")
        print(f"  Raw JSON: {json.dumps(status_data)}")

        if current_state == "finished":
            finished = True
            run_id = status_data.get("research_run_id")
            print(f"\n[+] RQ Job Finished! Resulting research_run_id: {run_id}")
            break
        elif current_state == "failed":
            error_msg = status_data.get("error")
            print(f"\n[!] RQ Job Failed: {error_msg}")
            sys.exit(1)

        time.sleep(poll_interval)

    assert finished, "Job did not finish within timeout period"
    assert run_id, "Job finished but returned no research_run_id"

    # 3. GET /research/{run_id}
    print(f"\n--- [3] GET /research/{run_id} Full Report Payload ---")
    r_get = client.get(f"/research/{run_id}")
    print(f"Response Code: {r_get.status_code}")
    report_data = r_get.json()
    print(f"Raw JSON Report:\n{json.dumps(report_data, indent=2)}")

    evidence = report_data.get("evidence", [])
    print(f"\nEvidence Count in Report: {len(evidence)}")
    assert len(evidence) > 0, "Expected evidence array to be populated with real claims"
    for idx, ev in enumerate(evidence, 1):
        assert ev.get("claim") is not None, f"Evidence #{idx} claim is null!"
        print(f"  Item #{idx} Claim: {ev.get('claim')}")

    print("\n" + "=" * 70)
    print("  VERIFICATION COMPLETE - ALL CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    init_db()
    test_async_workflow()
