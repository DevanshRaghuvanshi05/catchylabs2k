import sys
from pathlib import Path

if '--live' not in sys.argv[1:]:
    raise SystemExit('This legacy script can access live services or databases. Re-run with --live to confirm.')
sys.argv.remove('--live')
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

"""
Verification script for Stage 1.5 Semantic Recall using pgvector.
Runs the research pipeline for Puma / India / Gen Z brief, checks past evidence recall,
similarity scores, and metadata fields.
"""

import json
import logging
from google import genai
from config import GEMINI_API_KEY
from pipeline import run_research_pipeline
from database import save_research_run

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

    logger.info("Running Research Pipeline with Stage 1.5 Semantic Recall...")
    res = run_research_pipeline(brief)

    report_filepath = res.get("report_file")
    with open(report_filepath, "r", encoding="utf-8") as f:
        report_data = json.load(f)

    # Save research run to DB (including generating embeddings for new evidence items)
    client = genai.Client(api_key=GEMINI_API_KEY)
    run_id = save_research_run(brief, report_data, client=client)
    logger.info(f"Saved run to database with run_id: {run_id}")

    metadata = report_data.get("metadata", {})
    evidence_items = report_data.get("evidence", [])
    reused_items = [ev for ev in evidence_items if ev.get("reused_from_run_id")]

    print("\n" + "=" * 70)
    print("  VERIFICATION RESULTS: STAGE 1.5 SEMANTIC RECALL (PGVECTOR)")
    print("=" * 70)
    print(f"\n1. Semantic Recall Summary:")
    print(f"  Total Past Evidence Items Recalled : {len(reused_items)}")
    print(f"  Reused Evidence Count (Metadata)   : {metadata.get('reused_evidence_count')}")
    print(f"  Fresh Evidence Count (Metadata)    : {metadata.get('fresh_evidence_count')}")
    print(f"  Total Evidence Count               : {metadata.get('total_evidence_count')}")

    print("\n2. Recalled Evidence & Similarity Scores:")
    for idx, ev in enumerate(reused_items, 1):
        print(f"\n  Recalled Item #{idx}:")
        print(f"    Claim            : {ev.get('claim')[:80]}...")
        print(f"    Similarity Score : {ev.get('similarity_score')}")
        print(f"    Reused From Run  : {ev.get('reused_from_run_id')}")
        print(f"    Topic            : {ev.get('topic')}")

    print("\n3. Report Metadata Fields:")
    print(json.dumps({
        "reused_evidence_count": metadata.get("reused_evidence_count"),
        "fresh_evidence_count": metadata.get("fresh_evidence_count"),
        "total_evidence_count": metadata.get("total_evidence_count"),
        "coverage_rounds": metadata.get("coverage_rounds"),
        "coverage_gaps_identified": metadata.get("coverage_gaps_identified"),
        "additional_queries_used": metadata.get("additional_queries_used")
    }, indent=2))
    print("=" * 70)

if __name__ == "__main__":
    main()
