#!/usr/bin/env python3
import sys
from pathlib import Path

if '--live' not in sys.argv[1:]:
    raise SystemExit('This legacy script can access live services or databases. Re-run with --live to confirm.')
sys.argv.remove('--live')
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

"""Display saved consumer insights with evidence grounding for review."""

from database import get_consumer_insights, get_run_with_evidence

RUN_ID = "eb892e4a-4c06-4627-93fc-1b4caa7fc7ce"

def main():
    run_data = get_run_with_evidence(RUN_ID)
    ev_map = {e["id"]: e for e in run_data["evidence"]}
    insights = get_consumer_insights(RUN_ID)

    print(f"================================================================================")
    print(f"CONSUMER INTELLIGENCE REPORT FOR RUN: {RUN_ID}")
    print(f"Brand: {run_data['brand']} | Market: {run_data['market']} | Audience: {run_data['audience']}")
    print(f"Objective: {run_data['objective']}")
    print(f"Total Insights: {len(insights)} | Grounded Evidence Pool: {len(ev_map)} items")
    print(f"================================================================================")

    for i, ins in enumerate(insights, 1):
        itype = ins["insight_type"].upper()
        ids = ins["supporting_evidence_ids"]
        print(f"\n[{i}] TYPE: {itype}")
        print(f"INSIGHT: {ins['insight_text']}")
        print(f"CITED EVIDENCE IDs: {ids}")
        for eid in ids:
            ev = ev_map.get(eid)
            if ev:
                print(f"   * [{eid}] ({ev['topic']}): \"{ev['claim']}\"")
            else:
                print(f"   * [{eid}]: NOT FOUND IN DATABASE")
    print(f"\n================================================================================")

if __name__ == "__main__":
    main()
