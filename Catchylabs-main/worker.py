#!/usr/bin/env python3
"""
CatchyLab Agent 1 - RQ Worker Process
-------------------------------------
Listens on research, intelligence, opportunity, and content job queues.

Usage:
  Direct Python Execution:  python worker.py
  CLI RQ Execution:         rq worker research_jobs consumer_intelligence_jobs competitive_intelligence_jobs creative_gaps_jobs opportunity_jobs content_jobs
"""

import os
import sys
import logging
import redis
from rq import Worker, Queue
from rq.worker import SimpleWorker
from config import REDIS_URL

# Configure worker logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [RQ-Worker] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("CatchyLabWorker")

LISTEN_QUEUES = ["research_jobs", "consumer_intelligence_jobs", "competitive_intelligence_jobs", "creative_gaps_jobs", "opportunity_jobs", "content_jobs"]


def start_worker():
    """Initializes Redis connection and runs RQ worker loop."""
    logger.info("============================================================")
    logger.info("  CATCHYLAB AGENT 1 - RQ WORKER PROCESS STARTING            ")
    logger.info("============================================================")
    
    clean_url_display = REDIS_URL.split("@")[-1] if "@" in REDIS_URL else REDIS_URL
    logger.info(f"Connecting to Upstash Redis at: {clean_url_display}")
    
    try:
        redis_conn = redis.Redis.from_url(REDIS_URL)
        redis_conn.ping()
        logger.info("Successfully connected and authenticated with Upstash Redis!")
    except Exception as err:
        logger.error(f"CRITICAL: Failed to connect to Redis ({err}). Worker shutting down.", exc_info=True)
        sys.exit(1)

    logger.info(f"Listening for background jobs on queue: {LISTEN_QUEUES}")
    
    if os.name == "nt":
        logger.info("Windows detected: Using SimpleWorker for in-process job execution.")
        worker = SimpleWorker(LISTEN_QUEUES, connection=redis_conn)
    else:
        worker = Worker(LISTEN_QUEUES, connection=redis_conn)
        
    worker.work()



if __name__ == "__main__":
    start_worker()

