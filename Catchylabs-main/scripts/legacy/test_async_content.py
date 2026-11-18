import sys
from pathlib import Path

if '--live' not in sys.argv[1:]:
    raise SystemExit('This legacy script can access live services or databases. Re-run with --live to confirm.')
sys.argv.remove('--live')
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

"""
Async RQ Worker End-to-End Test for Content Generation
------------------------------------------------------
1. Enqueues job via POST /research/{run_id}/content
2. Runs SimpleWorker burst on queues including content_jobs
3. Polls GET /research/status/{job_id} until finished
4. Verifies GET /research/{run_id}/content returns draft pieces
"""

from fastapi.testclient import TestClient
from rq import SimpleWorker

from app import app
from config import get_redis_connection
from worker import LISTEN_QUEUES

PUMA_RUN_ID = "eb892e4a-4c06-4627-93fc-1b4caa7fc7ce"


def main():
    print("=== Testing Async RQ Job Execution for Content Generation ===")
    client = TestClient(app)

    resp = client.post(f"/research/{PUMA_RUN_ID}/content", json={"platform": "instagram"})
    assert resp.status_code == 200, f"Expected 200, got: {resp.status_code}: {resp.text}"
    job_id = resp.json()["job_id"]
    print(f"  [1] Successfully enqueued job: {job_id}")
    assert "content_jobs" in LISTEN_QUEUES

    print(f"  [2] Processing job with worker listening on {LISTEN_QUEUES}...")
    redis_conn = get_redis_connection()
    worker = SimpleWorker(LISTEN_QUEUES, connection=redis_conn)
    worker.work(burst=True)

    status_resp = client.get(f"/research/status/{job_id}")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    print(f"  [3] Job status: {status_data['status']}, job_type: {status_data.get('job_type')}")
    assert status_data["status"] == "finished", f"Job status was {status_data['status']}"
    assert status_data.get("job_type") == "content_generation"

    get_resp = client.get(f"/research/{PUMA_RUN_ID}/content")
    assert get_resp.status_code == 200
    res = get_resp.json()
    print(f"  [4] GET endpoint returned {res['total_pieces']} content pieces.")
    assert res["total_pieces"] >= 4
    print("\nASYNC CONTENT GENERATION RQ TEST PASSED!")


if __name__ == "__main__":
    main()
