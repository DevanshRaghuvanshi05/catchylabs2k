import sys
from pathlib import Path

if '--live' not in sys.argv[1:]:
    raise SystemExit('This legacy script can access live services or databases. Re-run with --live to confirm.')
sys.argv.remove('--live')
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

"""
Parallel Pipeline Performance Benchmark Script
------------------------------------------------
Executes the full 7-stage research pipeline with parallelized Stage 3 & Stage 5.
Measures and prints total execution time.
"""

import time
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
    print("=" * 70)
    print("  BENCHMARK: PARALLEL RESEARCH PIPELINE RUNTIME TEST")
    print("=" * 70)

    brief = {
        "brand": "Puma",
        "market": "India",
        "audience": "Gen Z",
        "objective": "Find opportunities for sustainable running shoes and eco-friendly sportswear"
    }

    start_time = time.time()
    result = run_research_pipeline(brief)
    elapsed_seconds = time.time() - start_time
    
    report_file = result.get("report_file")
    with open(report_file, "r", encoding="utf-8") as f:
        report = json.load(f)

    print("\n" + "=" * 70)
    print("  BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total Pipeline Runtime     : {elapsed_seconds:.2f} seconds ({elapsed_seconds / 60:.2f} minutes)")
    print(f"Total Raw Docs Retrieved   : {report['metadata']['total_raw_docs_retrieved']}")
    print(f"Total Evidence Extracted   : {len(report['evidence'])}")
    print(f"Extraction Stats Summary   :")
    print(f"  Processed : {report['metadata']['documents_processed']}")
    print(f"  Success   : {report['metadata']['documents_successful_extraction']}")
    print(f"  Skipped   : {report['metadata']['documents_skipped_no_claim']}")
    print(f"  Failed    : {report['metadata']['documents_failed_extraction']}")
    print("=" * 70)

if __name__ == "__main__":
    main()
