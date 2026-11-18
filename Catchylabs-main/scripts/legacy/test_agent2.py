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
Test script for Agent 2 (Consumer Intelligence).
Runs the pipeline against existing Puma/India/Gen Z run eb892e4a-4c06-4627-93fc-1b4caa7fc7ce
and displays synthesized insights with cited evidence.
"""

import json
import logging
from agent2 import run_consumer_intelligence_pipeline
from database import get_consumer_insights, get_run_with_evidence

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

TEST_RUN_ID = "eb892e4a-4c06-4627-93fc-1b4caa7fc7ce"

def main():
    print("=" * 80)
    print(f"Testing Agent 2 with Research Run ID: {TEST_RUN_ID}")
    print("=" * 80)

    # 1. Inspect run evidence
    run_data = get_run_with_evidence(TEST_RUN_ID)
    if not run_data:
        print(f"ERROR: Run {TEST_RUN_ID} not found!")
        return

    print(f"Brand: {run_data['brand']} | Market: {run_data['market']} | Audience: {run_data['audience']}")
    print(f"Objective: {run_data['objective']}")
    print(f"Total Evidence Items: {len(run_data['evidence'])}")
    evidence_lookup = {ev["id"]: ev for ev in run_data["evidence"]}

    # 2. Run Agent 2 pipeline
    print("\nExecuting Agent 2 Consumer Intelligence synthesis...")
    insights = run_consumer_intelligence_pipeline(TEST_RUN_ID)
    print(f"Synthesized and saved {len(insights)} consumer insights.\n")

    # 3. Fetch from DB to confirm persistence
    saved_insights = get_consumer_insights(TEST_RUN_ID)
    print(f"Verified from database: {len(saved_insights)} insights retrieved.\n")

    # 4. Display each insight with its cited evidence
    for idx, ins in enumerate(saved_insights, 1):
        itype = ins["insight_type"].upper()
        itext = ins["insight_text"]
        c_ids = ins["supporting_evidence_ids"]

        print(f"\n--- [INSIGHT #{idx}] [{itype}] (DB ID: {ins['id']}) ---")
        print(f"Insight: {itext}")
        print(f"Supporting Evidence IDs: {c_ids}")
        for cid in c_ids:
            ev = evidence_lookup.get(cid)
            if ev:
                print(f"  -> Evidence [{cid}] ({ev.get('topic')}): {ev.get('claim')}")
            else:
                print(f"  -> ERROR: Cited evidence ID {cid} NOT found in run!")

    print("\n" + "=" * 80)
    print("Agent 2 synthesis test complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
