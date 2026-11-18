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
Test Observability & Cost Tracking Verification Script
-----------------------------------------------------
Executes a real market research pipeline run, verifies api_calls logged in database,
and retrieves cost breakdown via get_research_run_cost() and logs via get_research_run_logs().
"""

import sys
import json
import logging
from fastapi.testclient import TestClient

from app import app
from database import init_db, get_research_run_cost, get_research_run_logs
from pipeline import run_research_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CatchyLabResearcher")

def main():
    print("=" * 80)
    print("  RUNNING REAL TEST PIPELINE WITH OBSERVABILITY & COST TRACKING")
    print("=" * 80)

    # 1. Initialize DB
    init_db()

    # 2. Brief definition
    brief = {
        "brand": "Puma",
        "market": "India",
        "audience": "Gen Z",
        "objective": "Find opportunities for sustainable running shoes and eco-friendly sportswear"
    }

    # 3. Run Pipeline with max_documents=5
    logger.info("Executing 7-stage research pipeline...")
    result = run_research_pipeline(brief, max_documents=5)
    run_id = result.get("research_run_id")
    print(f"\n[PIPELINE SUCCESS] Research Run ID: {run_id}")
    print(f"Elapsed Time: {result.get('elapsed_seconds')}s | Evidence Extracted: {result.get('evidence_count')}")

    # 4. Test GET /research/{run_id}/cost endpoint via FastAPI TestClient
    client = TestClient(app)

    print("\n" + "=" * 80)
    print(f"  GET /research/{run_id}/cost RESPONSE")
    print("=" * 80)
    resp_cost = client.get(f"/research/{run_id}/cost")
    print(f"HTTP Status: {resp_cost.status_code}")
    cost_json = resp_cost.json()
    print(json.dumps(cost_json, indent=2))

    # 5. Test GET /research/{run_id}/logs endpoint via FastAPI TestClient
    print("\n" + "=" * 80)
    print(f"  GET /research/{run_id}/logs RESPONSE")
    print("=" * 80)
    resp_logs = client.get(f"/research/{run_id}/logs")
    print(f"HTTP Status: {resp_logs.status_code}")
    logs_json = resp_logs.json()
    print(f"Total API Calls Logged: {logs_json.get('total_calls')}")
    print("\nSample Log Records (First 5):")
    for log in logs_json.get("logs", [])[:5]:
        print(f"  ID {log['id']} | Provider: {log['provider']} | Model: {log['model']} | Stage: {log['stage']} | CallType: {log['call_type']} | Duration: {log['duration_ms']}ms | Tokens: {log['estimated_tokens']} | Cost USD: {log['estimated_cost_usd']}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
