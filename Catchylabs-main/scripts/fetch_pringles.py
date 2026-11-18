"""
Fetch Pringles evidence in India using SerpAPI and Gemini.
"""

import sys
import json
from pathlib import Path

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
    "objective": "Identify market, consumer, and creative opportunities for Lay's potato chips in India against key competitors (Bingo!, Balaji Wafers, Haldiram's, Too Yumm!, Pringles, and regional brands)."
}

docs = search_web_serpapi("Pringles potato chips market share positioning India premium packaging", api_key=SERPAPI_API_KEY, max_results=4)
for doc in docs:
    res = extract_evidence_from_doc(client, doc, brief)
    if res.get("status") == "success" and res.get("data") and res["data"].get("claim"):
        d = res["data"]
        print("PRINGLES_CLAIM:", d.get("claim"))
        with open(REPO_ROOT / "scripts" / "pringles_evidence.json", "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)
        break
