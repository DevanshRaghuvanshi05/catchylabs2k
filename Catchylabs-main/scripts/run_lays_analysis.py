"""
End-to-end CatchyLabs analysis runner for Lay's in India.
Executes Agent 1 -> Agent 2 -> Agent 3 -> Agent 4 -> Opportunity Engine.
Preserves database isolation by targeting TEST_DATABASE_URL strictly.
"""

import os
import sys
import json
import logging
import uuid
from pathlib import Path
from dotenv import dotenv_values

# 1. Enforce Test Database Isolation strictly before any project import
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

test_env = dotenv_values(REPO_ROOT / ".env.test")
test_db_url = test_env.get("TEST_DATABASE_URL")
if not test_db_url:
    raise RuntimeError("TEST_DATABASE_URL is not set in .env.test. Aborting to protect live DB.")

os.environ["DATABASE_URL"] = test_db_url
os.environ["TEST_DATABASE_URL"] = test_db_url

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("LaysAnalysisRunner")

from google import genai
from config import GEMINI_API_KEY
from database import (
    init_db,
    save_research_run,
    get_research_run,
    get_run_with_evidence,
    get_consumer_insights,
    get_competitive_intelligence,
    get_creative_gaps,
    get_opportunity_scores,
)
from pipeline import deduplicate_evidence
from agent2 import run_consumer_intelligence_pipeline
from agent3 import run_competitive_intelligence_pipeline
from agent4 import run_creative_gaps_pipeline
from opportunity_engine import run_opportunity_engine_pipeline
from app import app
from fastapi.testclient import TestClient


def main():
    logger.info("==================================================================")
    logger.info("STARTING COMPLETE END-TO-END CATCHYLABS RUN: LAY'S INDIA")
    logger.info("==================================================================")

    # Initialize DB schema on test database
    init_db()

    client = genai.Client(api_key=GEMINI_API_KEY)

    brief = {
        "brand": "Lay's",
        "market": "India",
        "audience": "Indian youth, Gen Z, working professionals, and snackers",
        "objective": "Identify market, consumer, and creative opportunities for Lay's potato chips in India against key competitors (Bingo!, Balaji Wafers, Haldiram's, Too Yumm!, Pringles, and regional brands), exploring consumer sentiment around local Indian flavors, portion sizes, health/baked alternatives, pricing dynamics, and snacking occasions."
    }

    # =========================================================================
    # STAGE 1: Agent 1 (Researcher) - Evidence Consolidation & DB Persistence
    # =========================================================================
    logger.info("\n>>> [1/5] Executing Agent 1 (Researcher Evidence Loading & Persistence)...")
    
    # 1. Load baseline Lay's report file
    base_report_path = REPO_ROOT / "research_report_lay_s_20261019_002933.json"
    with open(base_report_path, "r", encoding="utf-8") as f:
        base_data = json.load(f)

    evidence_items = list(base_data.get("evidence", []))
    queries_used = list(base_data.get("metadata", {}).get("queries_used", []))

    # 2. Merge newly extracted competitor evidence (Balaji Wafers, Haldiram's, Bingo!, Pringles)
    new_comp_path = REPO_ROOT / "scripts" / "new_competitor_evidence.json"
    if new_comp_path.exists():
        with open(new_comp_path, "r", encoding="utf-8") as f:
            new_comp = json.load(f)
            evidence_items.extend(new_comp)
            logger.info(f"Loaded {len(new_comp)} competitor claims from {new_comp_path.name}")

    pringles_path = REPO_ROOT / "scripts" / "pringles_evidence.json"
    if pringles_path.exists():
        with open(pringles_path, "r", encoding="utf-8") as f:
            pringles_item = json.load(f)
            if isinstance(pringles_item, dict) and pringles_item.get("claim"):
                evidence_items.append(pringles_item)
                logger.info("Loaded Pringles competitor evidence claim.")

    # Deduplicate combined evidence pool
    deduped_evidence = deduplicate_evidence(evidence_items)
    logger.info(f"Consolidated evidence pool: {len(deduped_evidence)} items.")

    # Persist into PostgreSQL (test database)
    run_id = str(uuid.uuid4())
    report_payload = {
        "metadata": {
            **base_data.get("metadata", {}),
            "total_evidence_count": len(deduped_evidence),
            "queries_used": queries_used
        },
        "evidence": deduped_evidence
    }

    save_research_run(brief, report_payload, client=client, run_id=run_id)
    logger.info(f"Agent 1 Complete! Run ID: {run_id} persisted with {len(deduped_evidence)} evidence items.")

    # =========================================================================
    # STAGE 2: Agent 2 (Consumer Intelligence)
    # =========================================================================
    logger.info("\n>>> [2/5] Executing Agent 2 (Consumer Intelligence)...")
    consumer_insights = run_consumer_intelligence_pipeline(run_id, client=client)
    logger.info(f"Agent 2 Complete! Synthesized and validated {len(consumer_insights)} consumer insights.")
    for idx, ci in enumerate(consumer_insights, 1):
        logger.info(f"  Insight #{idx} [{ci.get('insight_type')}]: {ci.get('insight_text')[:100]}... (Cited IDs: {ci.get('supporting_evidence_ids')})")

    # =========================================================================
    # STAGE 3: Agent 3 (Competitive Intelligence)
    # =========================================================================
    logger.info("\n>>> [3/5] Executing Agent 3 (Competitive Intelligence)...")
    comp_intel = run_competitive_intelligence_pipeline(run_id, client=client, max_competitors=4)
    competitors = comp_intel.get("competitors", [])
    logger.info(f"Agent 3 Complete! Synthesized {len(competitors)} competitors.")
    for comp in competitors:
        logger.info(f"  Competitor: {comp.get('name')} | Positioning: {comp.get('positioning')[:80]}...")

    # =========================================================================
    # STAGE 4: Agent 4 (Creative Gaps Detection)
    # =========================================================================
    logger.info("\n>>> [4/5] Executing Agent 4 (Creative Gap Detection)...")
    creative_gaps = run_creative_gaps_pipeline(run_id, client=client)
    logger.info(f"Agent 4 Complete! Detected and validated {len(creative_gaps)} creative gaps.")
    for idx, cg in enumerate(creative_gaps, 1):
        logger.info(f"  Gap #{idx} [{cg.get('name')}]: {cg.get('description')[:100]}...")

    # =========================================================================
    # STAGE 5: Opportunity Engine (Scoring & Ranking)
    # =========================================================================
    logger.info("\n>>> [5/5] Executing Opportunity Engine (Scoring & Ranking)...")
    opp_scores = run_opportunity_engine_pipeline(run_id, client=client)
    logger.info(f"Opportunity Engine Complete! Scored & ranked {len(opp_scores)} opportunities.")
    for opp in opp_scores:
        logger.info(
            f"  Rank #{opp.get('rank')}: Gap ID {opp.get('creative_gap_id')} -> Overall Score: {opp.get('overall_score')} "
            f"[Demand: {opp.get('audience_demand_score')}, Whitespace: {opp.get('competitive_whitespace_score')}, "
            f"Brand Fit: {opp.get('brand_fit_score')}, Feasibility: {opp.get('feasibility_score')}]"
        )

    # =========================================================================
    # STAGE 6: API Verification via FastAPI TestClient
    # =========================================================================
    logger.info("\n==================================================================")
    logger.info("VERIFYING API ENDPOINTS")
    logger.info("==================================================================")
    tc = TestClient(app)

    endpoints_to_test = [
        f"/research/{run_id}",
        f"/research/{run_id}/consumer-intelligence",
        f"/research/{run_id}/competitor-intelligence",
        f"/research/{run_id}/creative-gaps",
        f"/research/{run_id}/opportunities",
        f"/research/{run_id}/cost",
        f"/research/{run_id}/logs",
        "/quota-status"
    ]

    api_results = {}
    for ep in endpoints_to_test:
        resp = tc.get(ep)
        logger.info(f"API GET {ep} -> Status: {resp.status_code}")
        if resp.status_code != 200:
            logger.error(f"Endpoint {ep} failed: {resp.text}")
            sys.exit(1)
        api_results[ep] = resp.json()

    # Save summary output
    output_summary = {
        "run_id": run_id,
        "brief": brief,
        "agent1_stats": {
            "evidence_count": len(deduped_evidence),
            "queries": queries_used,
            "report_file": str(base_report_path)
        },
        "agent2_insights": consumer_insights,
        "agent3_competitive_intel": comp_intel,
        "agent4_creative_gaps": creative_gaps,
        "opportunity_scores": opp_scores,
        "api_verification": {ep: "200 OK" for ep in endpoints_to_test}
    }

    out_file = REPO_ROOT / "outputs" / f"lays_india_run_{run_id}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_summary, f, indent=2, default=str)

    logger.info(f"\nAll results saved to: {out_file}")
    print(f"PIPELINE_SUCCESS: run_id={run_id}")
    return run_id, output_summary


if __name__ == "__main__":
    main()
