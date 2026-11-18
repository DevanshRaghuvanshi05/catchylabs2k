import sys
from pathlib import Path

if '--live' not in sys.argv[1:]:
    raise SystemExit('This legacy script can access live services or databases. Re-run with --live to confirm.')
sys.argv.remove('--live')
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

"""
Async RQ Worker End-to-End Test for Agent 4
-------------------------------------------
1. Starts an in-process worker listening on creative_gaps_jobs
2. Enqueues job via POST /research/{run_id}/creative-gaps
3. Polls GET /research/status/{job_id} until finished
4. Fetches GET /research/{run_id}/creative-gaps and verifies output
"""

import time
import threading
from fastapi.testclient import TestClient
import redis
from rq import SimpleWorker

from app import app
from config import REDIS_URL, get_redis_connection
from worker import LISTEN_QUEUES

PUMA_RUN_ID = "eb892e4a-4c06-4627-93fc-1b4caa7fc7ce"

def run_worker_thread(stop_event):
    redis_conn = get_redis_connection()
    worker = SimpleWorker(LISTEN_QUEUES, connection=redis_conn)
    # Process one job with a short timeout
    worker.work(burst=True)

def main():
    print("=== Testing Async RQ Job Execution for Agent 4 ===")
    client = TestClient(app)

    # 1. POST /research/{run_id}/creative-gaps
    resp = client.post(f"/research/{PUMA_RUN_ID}/creative-gaps")
    assert resp.status_code == 200, f"Expected 200, got: {resp.status_code}: {resp.text}"
    job_data = resp.json()
    job_id = job_data["job_id"]
    print(f"  [1] Successfully enqueued job: {job_id}")

    # 2. Run RQ worker in burst mode to process the job
    print(f"  [2] Processing job with worker listening on {LISTEN_QUEUES}...")
    redis_conn = get_redis_connection()
    worker = SimpleWorker(LISTEN_QUEUES, connection=redis_conn)
    worker.work(burst=True)

    # 3. Poll status
    status_resp = client.get(f"/research/status/{job_id}")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    print(f"  [3] Job status: {status_data['status']}, job_type: {status_data.get('job_type')}")
    assert status_data["status"] == "finished", f"Job status was {status_data['status']}"
    assert status_data.get("job_type") == "creative_gaps"

    # 4. Fetch results via GET endpoint
    get_resp = client.get(f"/research/{PUMA_RUN_ID}/creative-gaps")
    assert get_resp.status_code == 200
    res = get_resp.json()
    print(f"  [4] GET endpoint returned {res['total_gaps']} creative gaps.")
    assert res["total_gaps"] > 0
    print("\nASYNC RQ WORKER END-TO-END TEST PASSED!")

if __name__ == "__main__":
    main()
