#!/usr/bin/env python3
"""
CatchyLab Agent 1 (Researcher) - Market Research Agent MVP
----------------------------------------------------------
A CLI script that takes a research brief, generates search queries via Gemini API,
retrieves market results via SerpAPI, filters relevance, extracts structured evidence
via Gemini API, dedupes evidence, and outputs a timestamped JSON report.
"""

import os
import re
import json
import time
import logging
import hashlib
from datetime import datetime
import requests
from google import genai
from google.genai import types
from config import GEMINI_MODEL_NAME
from gemini_pacer import pacer
from evidence_extractor import (
    parse_retry_after,
    is_transient_error,
    is_hard_quota_exhausted,
    calculate_backoff,
)

# Setup clean console logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("CatchyLabResearcher")

# API keys must be provided via environment variables (never hardcoded)
DEFAULT_GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
DEFAULT_SERPAPI_KEY = os.environ.get("SERPAPI_API_KEY", "")


def clean_json_response(raw_text: str) -> str:
    """Helper to remove markdown code fences and sanitize raw LLM response into clean JSON text."""
    if not raw_text:
        return ""
    text = raw_text.strip()
    # Strip markdown ```json ... ``` or ``` ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def normalize_text(text: str) -> str:
    """Normalize text for deduplication comparison."""
    if not text:
        return ""
    # Lowercase and keep only alphanumeric characters
    return re.sub(r"[^a-z0-9]", "", text.lower())


# ============================================================================
# STAGE 2: Query Generation
# ============================================================================
def generate_queries(client: genai.Client, brief: dict) -> list[str]:
    """
    Prompt Gemini to generate 4-6 targeted search queries based on the research brief.
    Includes a 1-retry fallback for JSON parsing errors.
    """
    logger.info("=== STAGE 2: Generating Search Queries via Gemini API ===")
    
    prompt = (
        "You are a strategic market research strategist.\n"
        "Given the following research brief, generate 4-5 concise, keyword-focused search queries (3 to 6 words maximum per query) "
        "to gather consumer insights, market statistics, industry trends, and opportunities.\n\n"
        f"Research Brief:\n"
        f"- Brand: {brief.get('brand')}\n"
        f"- Market: {brief.get('market')}\n"
        f"- Audience: {brief.get('audience')}\n"
        f"- Objective: {brief.get('objective')}\n\n"
        "Keep each query short and punchy (e.g. 'Nike India Gen Z sustainable shoes', 'India eco running footwear market').\n"
        "Return ONLY a JSON array of 4-5 search query strings. Do NOT wrap in markdown fences or add extra explanations.\n"
        'Example output: ["query 1", "query 2", "query 3", "query 4"]'
    )

    queries = []
    
    for attempt in range(2):
        try:
            current_prompt = prompt if attempt == 0 else (
                prompt + "\n\nCRITICAL: Your previous response was invalid JSON. "
                "Return ONLY a valid raw JSON array of short query strings without markdown or pre-ambles."
            )
            
            pacer.wait_for_turn(GEMINI_MODEL_NAME)
            response = client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=current_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    response_mime_type="application/json"
                )
            )
            
            cleaned_text = clean_json_response(response.text)
            parsed = json.loads(cleaned_text)
            
            if isinstance(parsed, list) and len(parsed) >= 2:
                queries = [str(q).strip() for q in parsed if str(q).strip()]
                logger.info(f"Successfully generated {len(queries)} search queries (Attempt {attempt + 1}):")
                for idx, q in enumerate(queries, 1):
                    logger.info(f"  {idx}. {q}")
                return queries
            else:
                logger.warning(f"Gemini output was not a list of queries (Attempt {attempt + 1}). Content: {cleaned_text[:100]}")
        except Exception as e:
            logger.error(f"Error generating queries (Attempt {attempt + 1}): {e}")

    # Fallback default queries if LLM generation fails twice
    logger.warning("Using fallback default queries due to generation failure.")
    brand = brief.get('brand', '')
    market = brief.get('market', '')
    audience = brief.get('audience', '')
    return [
        f"{brand} {market} {audience} sustainable shoes",
        f"{brand} {market} market research trends",
        f"{audience} sustainable shoe preference {market}",
        f"{brand} market opportunities {market}"
    ]


# ============================================================================
# STAGE 3: Search & Retrieval (SerpAPI Web Search)
# ============================================================================
def search_web(query: str, serpapi_key: str, max_results: int = 5) -> list[dict]:
    """
    Search source implementation using SerpAPI.
    Returns a list of raw document dicts with title, url, snippet, source info.
    """
    logger.info(f"Executing web search for: '{query}'")
    url = "https://serpapi.com/search.json"
    params = {
        "q": query,
        "api_key": serpapi_key,
        "engine": "google",
        "num": max_results,
        "hl": "en",
        "gl": "in"
    }
    
    results = []
    for attempt in range(2):
        try:
            resp = requests.get(url, params=params, timeout=35)
            if resp.status_code == 200:
                data = resp.json()
                organic_results = data.get("organic_results", [])
                for item in organic_results:
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                        "publisher": item.get("displayed_link", "") or item.get("source", ""),
                        "query": query
                    })
                logger.info(f"  -> Retrieved {len(results)} search results.")
                return results
            else:
                logger.error(f"SerpAPI returned HTTP status {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            if attempt == 0:
                logger.warning(f"SerpAPI request attempt 1 failed ({e}). Retrying...")
            else:
                logger.error(f"Exception during SerpAPI request for query '{query}': {e}")
        
    return results


# ============================================================================
# STAGE 4: Relevance Filtering
# ============================================================================
def filter_documents(all_retrieved: list[dict], top_n_per_query: int = 3) -> list[dict]:
    """
    Filters retrieved documents by keeping top N results per query and removing duplicate URLs.
    """
    logger.info(f"=== STAGE 4: Filtering Documents (Top {top_n_per_query} per query) ===")
    seen_urls = set()
    filtered_docs = []
    
    query_counts = {}
    for doc in all_retrieved:
        url = doc.get("url")
        q = doc.get("query")
        if not url or url in seen_urls:
            continue
        
        current_count = query_counts.get(q, 0)
        if current_count < top_n_per_query:
            seen_urls.add(url)
            query_counts[q] = current_count + 1
            filtered_docs.append(doc)
            
    logger.info(f"Retained {len(filtered_docs)} documents out of {len(all_retrieved)} total raw results.")
    return filtered_docs


# ============================================================================
# STAGE 5: Evidence Extraction per Document
# ============================================================================
def extract_evidence_from_doc(client: genai.Client, doc: dict, brief: dict) -> dict | None:
    """
    Prompts Gemini to extract a structured evidence claim from a retained document.
    Returns None if no relevant evidence/claim is present or if extraction fails.
    """
    prompt = (
        "You are an expert market analyst extracting actionable intelligence for a market research report.\n"
        f"Research Brief:\n"
        f"- Brand: {brief.get('brand')}\n"
        f"- Market: {brief.get('market')}\n"
        f"- Audience: {brief.get('audience')}\n"
        f"- Objective: {brief.get('objective')}\n\n"
        f"Document Information:\n"
        f"- Title: {doc.get('title')}\n"
        f"- URL: {doc.get('url')}\n"
        f"- Publisher: {doc.get('publisher')}\n"
        f"- Snippet/Content: {doc.get('snippet')}\n\n"
        "Extract ONE key market research claim/evidence item from this document relevant to the brief.\n"
        "Return ONLY a JSON object matching this exact schema:\n"
        "{\n"
        '  "claim": "string (the core insight or statistic found)",\n'
        '  "source": {\n'
        '    "title": "string",\n'
        '    "url": "string",\n'
        '    "publisher": "string",\n'
        '    "published_date": "string or null"\n'
        '  },\n'
        '  "evidence": "string (supporting snippet or text quotation)",\n'
        '  "market": "string",\n'
        '  "audience": "string",\n'
        '  "topic": "string (e.g., consumer behavior, eco-friendly preference, market size, competition)",\n'
        '  "confidence": float (0.0 to 1.0)\n'
        "}\n\n"
        "NOTE: If the document snippet contains NO relevant claim or insight for the research brief, "
        'return JSON: {"claim": null}'
    )
    
    max_retries = 2
    for attempt in range(1, max_retries + 1):
        try:
            pacer.wait_for_turn(GEMINI_MODEL_NAME)
            response = client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type="application/json"
                )
            )
            
            cleaned_text = clean_json_response(response.text)
            data = json.loads(cleaned_text)
            
            if isinstance(data, dict) and data.get("claim"):
                # Ensure URL and Title in source match document if LLM missed it
                if "source" not in data or not isinstance(data["source"], dict):
                    data["source"] = {}
                data["source"]["title"] = data["source"].get("title") or doc.get("title", "")
                data["source"]["url"] = data["source"].get("url") or doc.get("url", "")
                data["source"]["publisher"] = data["source"].get("publisher") or doc.get("publisher", "")
                if "published_date" not in data["source"]:
                    data["source"]["published_date"] = None
                    
                return data
            else:
                logger.debug(f"No valid claim extracted for doc: {doc.get('url')}")
                return None
        except Exception as e:
            err_str = str(e)
            if is_hard_quota_exhausted(e):
                logger.warning(f"Quota exhausted on doc {doc.get('url')}. Failing gracefully without long retries.")
                return None
            if is_transient_error(e):
                if attempt < max_retries:
                    retry_after = parse_retry_after(e)
                    wait_seconds = calculate_backoff(attempt, base_delay=1.0, max_delay=4.0, retry_after=retry_after)
                    logger.warning(f"Rate limited (429) or transient error on doc {doc.get('url')}. Retrying in {wait_seconds:.2f}s (Attempt {attempt}/{max_retries})...")
                    pacer.handle_429(GEMINI_MODEL_NAME, e, default_wait=wait_seconds)
                else:
                    logger.warning(f"Retries exhausted on doc {doc.get('url')} due to rate limiting.")
                    return None
            else:
                logger.error(f"Error extracting evidence from doc ({doc.get('url')}): {e}")
                return None
                
    return None


def extract_all_evidence(client: genai.Client, documents: list[dict], brief: dict) -> list[dict]:
    """Iterates through all retained documents and extracts structured evidence with slight pacing."""
    logger.info(f"=== STAGE 5: Extracting Evidence from {len(documents)} Documents ===")
    extracted_evidence = []
    
    for idx, doc in enumerate(documents, 1):
        logger.info(f" Processing doc {idx}/{len(documents)}: {doc.get('title', '')[:50]}...")
        evidence = extract_evidence_from_doc(client, doc, brief)
        if evidence:
            logger.info(f"  [+] Claim Extracted: {evidence.get('claim')[:80]}...")
            extracted_evidence.append(evidence)
        else:
            logger.info(f"  [-] No relevant claim found or skipped.")
        # Brief sleep between calls to respect rate limits
        time.sleep(1.5)
            
    logger.info(f"Total evidence items extracted: {len(extracted_evidence)}")
    return extracted_evidence


# ============================================================================
# STAGE 6: Deduplication
# ============================================================================
def deduplicate_evidence(evidence_list: list[dict]) -> list[dict]:
    """
    Deduplicates evidence items based on normalized claim text hashes and source URL uniqueness.
    Keep higher confidence item when duplicates are found.
    """
    logger.info(f"=== STAGE 6: Deduplicating Evidence ({len(evidence_list)} initial items) ===")
    unique_claims = {}
    
    for item in evidence_list:
        claim_text = item.get("claim", "")
        norm_text = normalize_text(claim_text)
        url = item.get("source", {}).get("url", "") if isinstance(item.get("source"), dict) else ""
        
        # Create hash signature combining URL and normalized claim
        claim_hash = hashlib.md5(f"{url}:{norm_text[:100]}".encode("utf-8")).hexdigest()
        
        if claim_hash in unique_claims:
            existing_conf = unique_claims[claim_hash].get("confidence", 0.0)
            new_conf = item.get("confidence", 0.0)
            if new_conf > existing_conf:
                unique_claims[claim_hash] = item
        else:
            unique_claims[claim_hash] = item
            
    deduped = list(unique_claims.values())
    removed_count = len(evidence_list) - len(deduped)
    logger.info(f"Deduplication complete. Removed {removed_count} duplicate items. Final count: {len(deduped)}")
    return deduped


# ============================================================================
# STAGE 7: Output Generation & File Writing
# ============================================================================
def save_report(brief: dict, queries: list[str], total_raw_docs: int, evidence_items: list[dict], output_dir: str = ".") -> str:
    """
    Formats metadata and evidence list into a final report dict, and writes to a local JSON file.
    """
    logger.info("=== STAGE 7: Writing Final Research Report ===")
    
    brand_slug = re.sub(r"[^a-zA-Z0-9]", "_", brief.get("brand", "brand").lower())
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"research_report_{brand_slug}_{timestamp_str}.json"
    filepath = os.path.join(output_dir, filename)
    
    report_data = {
        "metadata": {
            "brand": brief.get("brand"),
            "market": brief.get("market"),
            "audience": brief.get("audience"),
            "objective": brief.get("objective"),
            "generated_at": datetime.now().isoformat(),
            "queries_used": queries,
            "total_raw_docs_retrieved": total_raw_docs,
            "total_evidence_count": len(evidence_items)
        },
        "evidence": evidence_items
    }
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Research Report successfully written to: {os.path.abspath(filepath)}")
    return filepath


# ============================================================================
# MAIN PIPELINE EXECUTION
# ============================================================================
# Import unified pipeline and extraction functions
from pipeline import run_research_pipeline, filter_documents, extract_all_evidence, deduplicate_evidence, save_report
from evidence_extractor import generate_queries, extract_evidence_from_doc, clean_json_response

if __name__ == "__main__":
    # Hardcoded example brief as specified in requirements
    example_brief = {
        "brand": "Nike",
        "market": "India",
        "audience": "Gen Z",
        "objective": "Find opportunities for sustainable running shoes"
    }
    
    result = run_research_pipeline(example_brief, max_documents=5)
    print("\n" + "=" * 60)
    print(f"PIPELINE COMPLETE! Final evidence count: {result['evidence_count']}")
    print(f"Report JSON written to: {result['report_file']}")
    print("=" * 60)

