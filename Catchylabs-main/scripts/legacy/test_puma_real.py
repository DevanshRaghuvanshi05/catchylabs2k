import sys
from pathlib import Path

if '--live' not in sys.argv[1:]:
    raise SystemExit('This legacy script can access live services or databases. Re-run with --live to confirm.')
sys.argv.remove('--live')
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

"""
Real Pipeline Runner & DB Persistence Verification Script for Puma
-------------------------------------------------------------------
Runs the full 7-stage research pipeline for Puma/India/Gen Z,
persists the resulting run, sources, and evidence into PostgreSQL DB,
and prints the exact database entries.
"""

import sys
import json
import logging
from database import init_db, save_research_run, get_research_run, SessionLocal, ResearchRun, Evidence, Source
from researcher import run_research_pipeline

# Configure logging to show pipeline stages clearly
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("CatchyLabResearcher")

def main():
    print("=" * 70)
    print("  RUNNING REAL MARKET RESEARCH PIPELINE FOR PUMA / INDIA / GEN Z")
    print("=" * 70)

    # Define research brief for Puma
    brief = {
        "brand": "Puma",
        "market": "India",
        "audience": "Gen Z",
        "objective": "Find opportunities for sustainable running shoes and eco-friendly sportswear"
    }

    # Step 1: Initialize Database Tables
    init_db()

    # Step 2: Run 7-stage pipeline
    logger.info("Executing real search queries & Gemini evidence extraction...")
    pipeline_result = run_research_pipeline(brief)
    report_filepath = pipeline_result.get("report_file")

    with open(report_filepath, "r", encoding="utf-8") as f:
        report_data = json.load(f)

    evidence_count = report_data.get("metadata", {}).get("total_evidence_count", 0)
    print(f"\n[PIPELINE OUTPUT] Report file generated: {report_filepath}")
    print(f"[PIPELINE OUTPUT] Total evidence items extracted: {evidence_count}")

    # Step 3: Persist into PostgreSQL Database
    run_id = save_research_run(brief, report_data)
    print(f"[DB PERSISTENCE] Successfully saved to PostgreSQL! Assigned run_id: {run_id}")

    # Step 4: Verify Database Contents
    session = SessionLocal()
    try:
        db_run = session.query(ResearchRun).filter_by(id=run_id).first()
        db_evidence = session.query(Evidence).filter_by(research_run_id=run_id).all()

        print("\n" + "=" * 70)
        print(f"  VERIFYING POSTGRESQL DB RECORD FOR RUN ID: {run_id}")
        print("=" * 70)
        print(f"Brand      : {db_run.brand}")
        print(f"Market     : {db_run.market}")
        print(f"Audience   : {db_run.audience}")
        print(f"Objective  : {db_run.objective}")
        print(f"Created At : {db_run.created_at}")
        print(f"Evidence Count in DB: {len(db_evidence)}")
        print("-" * 70)

        for idx, ev in enumerate(db_evidence, 1):
            src = session.query(Source).filter_by(id=ev.source_id).first()
            print(f"\nEvidence #{idx}:")
            print(f"  Claim      : {ev.claim}")
            print(f"  Excerpt    : {ev.excerpt}")
            print(f"  Topic      : {ev.topic}")
            print(f"  Confidence : {ev.confidence}")
            if src:
                print(f"  Source Title: {src.title}")
                print(f"  Source URL  : {src.url}")
                print(f"  Publisher   : {src.publisher}")
            print("-" * 40)

        # Step 5: Test GET /research/{run_id} retrieval via get_research_run()
        retrieved_data = get_research_run(run_id)
        print("\n[DB RETRIEVAL CHECK] get_research_run() verified:")
        print(f"  Metadata ID: {retrieved_data['metadata']['id']}")
        print(f"  Report Total Evidence: {retrieved_data['metadata']['total_evidence_count']}")

    finally:
        session.close()

if __name__ == "__main__":
    main()
