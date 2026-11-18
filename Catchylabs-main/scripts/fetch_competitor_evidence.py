"""
Fetch real research evidence for key Lay's competitors in India:
Balaji Wafers, Haldiram's, and Pringles.
Uses SerpAPI and Gemini 2.5 Flash via extract_evidence_from_doc.
"""

import os
import sys
import json
from pathlib import Path
from dotenv import dotenv_values

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import GEMINI_API_KEY, SERPAPI_API_KEY
from search_engine import search_web_serpapi
from evidence_extractor import extract_evidence_from_doc
from google import genai

client = genai.Client(api_key=GEMINI_API_KEY)

brief = {
    "brand": "Lay's",
    "market": "India",
    "audience": "Indian youth, Gen Z, working professionals, and snackers",
    "objective": "Identify market, consumer, and creative opportunities for Lay's potato chips in India against key competitors (Bingo!, Balaji Wafers, Haldiram's, Too Yumm!, Pringles, and regional brands), exploring consumer sentiment around local Indian flavors, portion sizes, health/baked alternatives, pricing dynamics, and snacking occasions."
}

competitor_queries = [
    "Balaji Wafers market share India potato chips volume",
    "Balaji Wafers price quantity value for money snacks India",
    "Haldiram potato chips snacks market share India",
    "Pringles potato chips market share India premium price positioning",
    "ITC Bingo Mad Angles potato chips market share India youth"
]

extracted = []
for q in competitor_queries:
    print(f"Searching: {q}")
    docs = search_web_serpapi(q, api_key=SERPAPI_API_KEY, max_results=3)
    for doc in docs:
        res = extract_evidence_from_doc(client, doc, brief)
        if res.get("status") == "success" and res.get("data"):
            d = res["data"]
            if d.get("claim"):
                print(f"  [+] EXTRACTED: {d.get('claim')}")
                extracted.append(d)
                break

out_path = REPO_ROOT / "scripts" / "new_competitor_evidence.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(extracted, f, indent=2)

print(f"\nSuccessfully extracted {len(extracted)} competitor claims saved to {out_path}")
