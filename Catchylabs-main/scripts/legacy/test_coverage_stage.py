import sys
from pathlib import Path

if '--live' not in sys.argv[1:]:
    raise SystemExit('This legacy script can access live services or databases. Re-run with --live to confirm.')
sys.argv.remove('--live')
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

"""
Test script for verifying Stage 5.5 Coverage Check & Optional Round 2 Execution.
Tests against the Puma / India / Gen Z brief.
"""

import json
import logging
from pipeline import run_research_pipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("CatchyLabResearcher")

def main():
    brief = {
        "brand": "Puma",
        "market": "India",
        "audience": "Gen Z",
        "objective": "Find opportunities for sustainable running shoes and eco-friendly sportswear"
    }

    logger.info("Executing research pipeline with coverage check...")
    res = run_research_pipeline(brief)

    report_filepath = res.get("report_file")
    with open(report_filepath, "r", encoding="utf-8") as f:
        report_data = json.load(f)

    metadata = report_data.get("metadata", {})
    coverage_info = res.get("coverage_info", {})

    print("\n" + "=" * 70)
    print("  VERIFICATION RESULTS: COVERAGE CHECK STAGE")
    print("=" * 70)
    print("\n1. Coverage Check Response JSON:")
    print(json.dumps(coverage_info.get("coverage_check_result"), indent=2))

    print("\n2. Round 2 Execution Status:")
    print(f"  Coverage Rounds Executed : {metadata.get('coverage_rounds')}")
    print(f"  Did Round 2 Trigger?     : {metadata.get('coverage_rounds') == 2}")

    print("\n3. Final Report Metadata Fields:")
    print(f"  coverage_rounds          : {metadata.get('coverage_rounds')}")
    print(f"  coverage_gaps_identified : {json.dumps(metadata.get('coverage_gaps_identified'))}")
    print(f"  additional_queries_used  : {json.dumps(metadata.get('additional_queries_used'))}")
    print(f"  queries_used (total)     : {json.dumps(metadata.get('queries_used'))}")
    print(f"  total_evidence_count     : {metadata.get('total_evidence_count')}")
    print("=" * 70)

if __name__ == "__main__":
    main()
