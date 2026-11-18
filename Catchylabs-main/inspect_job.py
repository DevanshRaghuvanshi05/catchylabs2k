"""
inspect_job.py
Takes a job id and prints only its RQ status, queue name, position, enqueued time,
and whether a worker is registered.
Uses strictly read-only calls and does not start or modify anything.
Plain ASCII only.
"""

import sys
import redis
from rq import Queue, Worker
from rq.job import Job
from rq.exceptions import NoSuchJobError
from config import REDIS_URL


def inspect_job(job_id: str) -> None:
    if not job_id:
        print("Error: No job ID provided.")
        sys.exit(1)

    if not REDIS_URL:
        print("Error: REDIS_URL is not configured.")
        sys.exit(1)

    conn = redis.Redis.from_url(REDIS_URL)

    try:
        job = Job.fetch(job_id, connection=conn)
    except NoSuchJobError:
        print(f"Error: Job '{job_id}' not found.")
        sys.exit(1)
    except Exception as exc:
        print(f"Error fetching job: {exc}")
        sys.exit(1)

    # 1. RQ status
    status = job.get_status()

    # 2. Queue name
    queue_name = job.origin or "None"

    # 3. Position in queue
    position = "None"
    if status == "queued" and job.origin:
        try:
            q = Queue(job.origin, connection=conn)
            pos = q.get_job_position(job_id)
            position = str(pos) if pos is not None else "None"
        except Exception:
            position = "None"

    # 4. Enqueued time
    enqueued_time = job.enqueued_at.isoformat() if job.enqueued_at else "None"

    # 5. Whether a worker is registered
    try:
        workers = Worker.all(connection=conn)
        worker_registered = bool(workers and len(workers) > 0)
    except Exception:
        worker_registered = False

    print(f"RQ status: {status}")
    print(f"Queue name: {queue_name}")
    print(f"Position: {position}")
    print(f"Enqueued time: {enqueued_time}")
    print(f"Worker registered: {worker_registered}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_job.py <job_id>")
        sys.exit(1)

    inspect_job(sys.argv[1].strip())
