import sys
from pathlib import Path

if '--live' not in sys.argv[1:]:
    raise SystemExit('This legacy script can access live services or databases. Re-run with --live to confirm.')
sys.argv.remove('--live')
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

"""
Verification script for evidence extraction bug fix.
Runs the research pipeline for 3-5 documents to verify logging, retry logic, and metadata tracking.
"""

import json
import logging
from pipeline import run_research_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("CatchyLabResearcher")

def main():
    logger.info("=" * 70)
    logger.info("  RUNNING PIPELINE TEST (Max 4 Documents)")
    logger.info("=" * 70)

    brief = {
        "brand": "Puma",
        "market": "India",
        "audience": "Gen Z",
        "objective": "Find opportunities for sustainable running shoes and eco-friendly sportswear"
    }

    result = run_research_pipeline(brief, max_documents=4)
    report_file = result.get("report_file")
    
    with open(report_file, "r", encoding="utf-8") as f:
        report = json.load(f)
        
    print("\n" + "=" * 70)
    print("  REPORT METADATA & EVIDENCE SUMMARY")
    print("=" * 70)
    print("Metadata:")
    print(json.dumps(report["metadata"], indent=2))
    print(f"\nTotal Evidence Items Extracted: {len(report['evidence'])}")
    for idx, ev in enumerate(report["evidence"], 1):
        print(f"\nItem #{idx}:")
        print(f"  Claim     : {ev.get('claim')}")
        print(f"  Topic     : {ev.get('topic')}")
        print(f"  Confidence: {ev.get('confidence')}")
        print(f"  Source URL: {ev.get('source', {}).get('url')}")
    print("=" * 70)

if __name__ == "__main__":
    main()
