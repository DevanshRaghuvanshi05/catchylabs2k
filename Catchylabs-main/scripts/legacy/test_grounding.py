#!/usr/bin/env python3
import sys
from pathlib import Path

if '--live' not in sys.argv[1:]:
    raise SystemExit('This legacy script can access live services or databases. Re-run with --live to confirm.')
sys.argv.remove('--live')
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

"""Test statistical verification function against existing insights."""
import re
from database import get_consumer_insights, get_run_with_evidence

RUN_ID = "eb892e4a-4c06-4627-93fc-1b4caa7fc7ce"

STATS_PATTERN = re.compile(
    r"(?:"
    r"\b\d+(?:\.\d+)?\s*%"                                                      # e.g. 65%, 77%, 14.52%
    r"|\b\d+(?:\.\d+)?\s*percent\b"                                             # e.g. 65 percent
    r"|CAGR\s*(?:of\s*)?\d+(?:\.\d+)?%?"                                        # e.g. CAGR of 14.52%
    r"|\b\d+(?:\.\d+)?%\s*CAGR\b"                                               # e.g. 14.52% CAGR
    r"|(?:[\$€£]|(?:USD|INR|EUR|GBP)\s*)\d+(?:\.\d+)?(?:\s*(?:billion|million|trillion|crore|lakh))?\b" # e.g. $1.59 Billion
    r"|\b\d+(?:\.\d+)?\s*(?:billion|million|trillion|crore|lakh)\b"             # e.g. 10.72 Billion
    r"|\b\d+(?:\.\d+)?\s*x\b"                                                   # e.g. 2x, 3x
    r"|\b\d+\s*(?:out of|\/)\s*\d+\b"                                           # e.g. 7 out of 10
    r"|\b\d+\.\d+\b"                                                            # Any decimal number, e.g. 14.52, 0.71, 108.3
    r")",
    re.IGNORECASE
)

def extract_statistical_figures(text: str) -> list[dict]:
    results = []
    for match in STATS_PATTERN.finditer(text):
        raw = match.group(0).strip()
        num_m = re.search(r"\d+(?:\.\d+)?", raw)
        core_num = num_m.group(0) if num_m else raw
        results.append({
            "raw": raw,
            "core_num": core_num
        })
    return results

def verify_numerical_grounding(insight_text: str, cited_evidences: list[dict]) -> list[str]:
    """
    Verifies that all statistical figures in insight_text appear in at least one
    of the cited evidence items' claim or excerpt.
    Returns list of ungrounded figure error messages.
    """
    figures = extract_statistical_figures(insight_text)
    if not figures:
        return []

    # Combined evidence text from all cited items
    combined_ev_text = " ".join([
        f"{ev.get('claim', '')} {ev.get('excerpt', '')}"
        for ev in cited_evidences
    ])

    errors = []
    for fig in figures:
        core = fig["core_num"]
        raw = fig["raw"]
        
        # Check if the core number appears in combined evidence text as a number
        # Word boundary or preceded/followed by non-digit
        pattern = r"(?<!\d)" + re.escape(core) + r"(?!\d)"
        if not re.search(pattern, combined_ev_text):
            errors.append(f"Ungrounded statistic '{raw}' (number: {core}) not found in cited evidence.")

    return errors

if __name__ == "__main__":
    run_data = get_run_with_evidence(RUN_ID)
    ev_map = {e["id"]: e for e in run_data["evidence"]}
    insights = get_consumer_insights(RUN_ID)

    print("Auditing existing database insights:")
    for i, ins in enumerate(insights, 1):
        cited = [ev_map[cid] for cid in ins["supporting_evidence_ids"] if cid in ev_map]
        errs = verify_numerical_grounding(ins["insight_text"], cited)
        stats = extract_statistical_figures(ins["insight_text"])
        stat_strs = [s["raw"] for s in stats]
        print(f"Insight #{i} [{ins['insight_type']}]: Stats found: {stat_strs}")
        if errs:
            for e in errs:
                print(f"  [ERROR] {e}")
        else:
            print("  [OK] All statistics grounded in cited evidence.")
