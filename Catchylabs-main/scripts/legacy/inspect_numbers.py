#!/usr/bin/env python3
import sys
from pathlib import Path

if '--live' not in sys.argv[1:]:
    raise SystemExit('This legacy script can access live services or databases. Re-run with --live to confirm.')
sys.argv.remove('--live')
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

"""Inspect all numbers across all saved insights and verify against cited evidence."""
import re
from database import get_consumer_insights, get_run_with_evidence

RUN_ID = "eb892e4a-4c06-4627-93fc-1b4caa7fc7ce"

def main():
    run_data = get_run_with_evidence(RUN_ID)
    ev_map = {e["id"]: e for e in run_data["evidence"]}
    insights = get_consumer_insights(RUN_ID)

    print("=" * 80)
    print(f"NUMERICAL AUDIT FOR RUN: {RUN_ID}")
    print("=" * 80)

    pattern = re.compile(
        r"(?:"
        r"\b\d+(?:\.\d+)?\s*(?:%|percent(?:age)?)\b"           # 77%, 14.52%, 65 percent
        r"|(?:\$|USD|INR|EUR|€|£)\s*\d+(?:\.\d+)?(?:\s*(?:billion|million|trillion|crore|lakh))?\b" # $108.3B, USD 0.71B
        r"|\b\d+(?:\.\d+)?\s*(?:billion|million|trillion|crore|lakh)\b" # 10.72 Billion
        r"|CAGR\s*(?:of)?\s*\d+(?:\.\d+)?%?"                  # CAGR of 14.52%
        r"|\b\d+(?:\.\d+)?\b"                                 # standalone numbers / years
        r")",
        re.IGNORECASE
    )

    for i, ins in enumerate(insights, 1):
        text = ins["insight_text"]
        c_ids = ins["supporting_evidence_ids"]
        print(f"\n--- INSIGHT #{i} [{ins['insight_type'].upper()}] ---")
        print(f"Text: {text}")
        print(f"Cited Evidence IDs: {c_ids}")

        # Combine text of cited evidence
        combined_ev_text = " ".join([
            f"{ev_map[cid]['claim']} {ev_map[cid]['excerpt']}"
            for cid in c_ids if cid in ev_map
        ])

        # Find all numbers/statistics
        matches = pattern.findall(text)
        print(f"Extracted Numbers/Stats: {matches}")

        for m in matches:
            # Clean number for search
            # Extract just digits and decimal
            clean_digits = re.search(r"\d+(?:\.\d+)?", m)
            if not clean_digits:
                continue
            digit_val = clean_digits.group(0)

            # Check if digit_val or pattern is in combined evidence
            found = (digit_val in combined_ev_text)
            status = "FOUND (VALID)" if found else "NOT FOUND -> HALLUCINATED!"
            print(f"   [{m}] (digit: {digit_val}) -> {status}")

if __name__ == "__main__":
    main()
