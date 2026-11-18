#!/usr/bin/env python3
import sys
from pathlib import Path

if '--live' not in sys.argv[1:]:
    raise SystemExit('This legacy script can access live services or databases. Re-run with --live to confirm.')
sys.argv.remove('--live')
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import re

STATS_PATTERN = re.compile(
    r"(?:"
    r"\b\d+(?:\.\d+)?\s*%"                                                      # e.g. 65%, 77%, 14.52%
    r"|\b\d+(?:\.\d+)?\s*percent\b"                                             # e.g. 65 percent
    r"|CAGR\s*(?:of\s*)?\d+(?:\.\d+)?%?"                                        # e.g. CAGR of 14.52%
    r"|\b\d+(?:\.\d+)?%\s*CAGR\b"                                               # e.g. 14.52% CAGR
    r"|(?:[\$€£]|(?:USD|INR|EUR|GBP)\s*)\d+(?:\.\d+)?(?:\s*(?:billion|million|trillion|crore|lakh))?\b" # e.g. $1.59 Billion
    r"|\b\d+(?:\.\d+)?\s*(?:billion|million|trillion|crore|lakh)\b"             # e.g. 10.72 Billion
    r"|\b\d+(?:\.\d+)?\s*x\b"                                                   # e.g. 2x, 3x
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

if __name__ == "__main__":
    sample = (
        "Indian Gen Z consumers show 65% preference for recycled materials, "
        "while 77% cite price. Market is growing at a CAGR of 14.52% to reach $1.59 Billion. "
        "Market share is 6% in 2024. Puma leads with 100% vegan styles."
    )
    res = extract_statistical_figures(sample)
    for r in res:
        print(f"Match: '{r['raw']}' -> Core: '{r['core_num']}'")
