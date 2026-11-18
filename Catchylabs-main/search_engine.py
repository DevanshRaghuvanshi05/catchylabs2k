"""
Search engine module for CatchyLab Agent 1 (Researcher).
Abstracts search source execution so alternative search providers can be swapped easily.
"""

import logging
import os
import requests
from urllib.parse import urlsplit
from config import SERPAPI_ENDPOINT, SERPAPI_API_KEY
from database import track_api_call, count_serpapi_calls_today, redact_secrets

logger = logging.getLogger("CatchyLabResearcher")


def search_web_serpapi(query: str, api_key: str = None, max_results: int = 5) -> list[dict]:
    """
    Retrieves search results using SerpAPI (Google Search Engine).
    Enforces daily cap from MAX_SERPAPI_PER_DAY (default 15).
    Raises RuntimeError before the call is made when the cap is reached.
    Raises RuntimeError on search provider failures after bounded retries.
    Returns list of dicts: [{'title', 'url', 'snippet', 'publisher', 'query'}]
    """
    max_cap = int(os.environ.get("MAX_SERPAPI_PER_DAY", "15"))
    calls_today = count_serpapi_calls_today()
    if calls_today >= max_cap:
        raise RuntimeError(
            f"Daily SerpAPI limit reached: {calls_today}/{max_cap} calls used today (UTC)."
        )

    key = api_key or SERPAPI_API_KEY
    logger.info(f"Executing web search for: '{query}'")
    
    params = {
        "q": query,
        "api_key": key,
        "engine": "google",
        "num": max_results,
        "hl": "en",
        "gl": "in"
    }
    
    results = []
    last_error = None
    for attempt in range(2):
        try:
            with track_api_call(provider="serpapi", call_type="web_search") as tracker:
                resp = requests.get(SERPAPI_ENDPOINT, params=params, timeout=35)
                if resp.status_code == 200:
                    tracker.status = "success"
                    data = resp.json()
                    organic_results = data.get("organic_results", [])
                    for item in organic_results:
                        if not isinstance(item, dict):
                            continue
                        url = item.get("link")
                        if not isinstance(url, str):
                            continue
                        try:
                            parsed_url = urlsplit(url.strip())
                        except ValueError:
                            continue
                        if (
                            parsed_url.scheme.lower() not in ("http", "https")
                            or not parsed_url.hostname
                            or parsed_url.username is not None
                            or parsed_url.password is not None
                        ):
                            continue
                        results.append({
                            "title": item.get("title", "") if isinstance(item.get("title"), str) else "",
                            "url": url.strip(),
                            "snippet": item.get("snippet", "") if isinstance(item.get("snippet"), str) else "",
                            "publisher": (
                                item.get("displayed_link", "") or item.get("source", "")
                            ) if isinstance(item.get("displayed_link", "") or item.get("source", ""), str) else "",
                            "query": query
                        })
                    logger.info(f"  -> Retrieved {len(results)} search results.")
                    return results
                else:
                    raw_err = f"SerpAPI HTTP status {resp.status_code}: {resp.text[:200]}"
                    err_msg = redact_secrets(raw_err)
                    if resp.status_code in (429, 403):
                        tracker.status = "rate_limited"
                    else:
                        tracker.status = "failed"
                    tracker.error_message = err_msg
                    last_error = err_msg
                    logger.error(err_msg)
        except Exception as e:
            sanitized_err = redact_secrets(str(e))
            last_error = sanitized_err
            if attempt == 0:
                logger.warning(f"SerpAPI attempt 1 failed ({sanitized_err}). Retrying...")
            else:
                logger.error(f"SerpAPI request error for '{query}': {sanitized_err}")
                
    if last_error:
        raise RuntimeError(f"SerpAPI search failed for query '{query}': {last_error}")
    return results
