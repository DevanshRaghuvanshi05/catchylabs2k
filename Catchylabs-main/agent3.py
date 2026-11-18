"""
Agent 3: Competitor Intelligence Synthesis Pipeline.
----------------------------------------------------
Identifies named competitors from existing evidence, executes targeted
competitor research (SerpAPI + Gemini extraction), saves competitor evidence,
and synthesizes structured competitive intelligence with strict citation and
numerical grounding from day one.
"""

import json
import logging
import re
import time
from typing import Dict, Any, List, Optional, Set

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_MODEL_NAME, GEMINI_FALLBACK_MODELS, SERPAPI_API_KEY
from database import (
    get_run_with_evidence,
    save_competitor_evidence,
    save_competitive_intelligence,
    track_api_call,
)
from evidence_extractor import clean_json_response, extract_evidence_from_doc
from search_engine import search_web_serpapi
from gemini_pacer import pacer
from intelligence.knowledge import append_knowledge_guidance

logger = logging.getLogger("CatchyLabAgent3")

# Statistical and quantified figure extraction regex (same as Agent 2)
STATS_PATTERN = re.compile(
    r"(?:"
    r"\b\d+(?:\.\d+)?\s*%"                                                      # e.g. 65%, 77%, 14.52%
    r"|\b\d+(?:\.\d+)?\s*percent\b"                                             # e.g. 65 percent
    r"|CAGR\s*(?:of\s*)?\d+(?:\.\d+)?%?"                                        # e.g. CAGR of 14.52%
    r"|\b\d+(?:\.\d+)?%\s*CAGR\b"                                               # e.g. 14.52% CAGR
    r"|(?:[\$€£]|(?:USD|INR|EUR|GBP)\s*)\d+(?:\.\d+)?(?:\s*(?:billion|million|trillion|crore|lakh|b|m|k))?\b" # e.g. $1.59 Billion
    r"|\b\d+(?:\.\d+)?\s*(?:billion|million|trillion|crore|lakh)\b"             # e.g. 10.72 Billion
    r"|\b\d+(?:\.\d+)?\s*x\b"                                                   # e.g. 2x, 3x
    r"|\b\d+\s*(?:out of|\/)\s*\d+\b"                                           # e.g. 7 out of 10
    r"|\b\d+\.\d+\b"                                                            # Any decimal number, e.g. 14.52, 0.71, 108.3
    r")",
    re.IGNORECASE
)


def extract_statistical_figures(text: str) -> List[Dict[str, str]]:
    """Extracts all statistical figures, percentages, and metrics from text."""
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


def verify_text_numerical_grounding(
    text: str,
    cited_ids: List[int],
    evidence_lookup: Dict[int, dict],
    context_label: str
) -> List[str]:
    """Verifies that all numbers/percentages in text appear in cited evidence claim/excerpt."""
    figures = extract_statistical_figures(text)
    if not figures:
        return []

    combined_text = " ".join([
        f"{evidence_lookup[cid].get('claim', '')} {evidence_lookup[cid].get('excerpt', '')}"
        for cid in cited_ids if cid in evidence_lookup
    ])

    errors = []
    for fig in figures:
        core = fig["core_num"]
        raw = fig["raw"]
        num_pattern = r"(?<!\d)" + re.escape(core) + r"(?!\d)"
        if not re.search(num_pattern, combined_text):
            errors.append(
                f"[{context_label}] contains ungrounded/hallucinated statistic '{raw}' "
                f"(number: {core}) not found in cited evidence items {cited_ids}."
            )
    return errors


def identify_competitors_from_evidence(
    client: genai.Client,
    run_data: dict,
    evidence_items: List[dict],
    max_competitors: int = 3
) -> List[str]:
    """
    Scans existing evidence claims via Gemini to identify named competitors/brands
    active in the market (excluding the target brand itself). Capped at max_competitors.
    """
    brand = run_data.get("brand", "")
    market = run_data.get("market", "")
    audience = run_data.get("audience", "")
    objective = run_data.get("objective", "")

    claims_text = "\n".join([
        f"- [ID: {ev['id']}] {ev.get('claim', '')}"
        for ev in evidence_items
    ])

    prompt = (
        "You are an expert market intelligence analyst.\n"
        f"Target Brand: {brand}\n"
        f"Target Market: {market}\n"
        f"Audience: {audience}\n"
        f"Objective: {objective}\n\n"
        "Here are the verified research claims gathered so far:\n"
        f"{claims_text}\n\n"
        f"TASK: Identify specific NAMED competitor brands, rival companies, or local market labels "
        f"mentioned in these claims. Do NOT include '{brand}' itself as a competitor.\n"
        f"Return ONLY a JSON array of up to {max_competitors} distinct competitor brand names.\n"
        "Example output: [\"Brand A\", \"Brand B\", \"Brand C\"]\n"
        "If fewer than 3 competitors are mentioned, return only the ones mentioned. If none are mentioned, return prominent direct competitors in this category."
    )

    models_to_try = GEMINI_FALLBACK_MODELS or [GEMINI_MODEL_NAME]
    for model_name in models_to_try:
        try:
            with track_api_call(
                provider="gemini",
                call_type="competitor_identification",
                stage="Agent3_Competitor_ID",
                model=model_name,
                research_run_id=run_data.get("id"),
                input_text=prompt[:300]
            ) as tracker:
                pacer.wait_for_turn(model_name)
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        response_mime_type="application/json"
                    )
                )
                tracker.set_response(response)

            raw = response.text if hasattr(response, "text") and response.text else ""
            cleaned = clean_json_response(raw)
            parsed = json.loads(cleaned)

            if isinstance(parsed, list):
                # Filter out the target brand itself
                clean_names = [
                    str(c).strip() for c in parsed
                    if str(c).strip() and str(c).strip().lower() != brand.lower()
                ][:max_competitors]
                if clean_names:
                    logger.info(f"Identified {len(clean_names)} competitors from evidence: {clean_names}")
                    return clean_names
        except Exception as exc:
            logger.warning(f"Error identifying competitors with {model_name}: {exc}")
            pacer.handle_429(model_name, exc, default_wait=1.0)

    # Fallback default if LLM extraction fails
    return ["Thaely", "Satva"]


def generate_competitor_queries(
    client: genai.Client,
    run_data: dict,
    competitor_name: str,
    queries_per_competitor: int = 2
) -> List[str]:
    """Generates 2 targeted search queries for a specific competitor brand."""
    brand = run_data.get("brand", "")
    market = run_data.get("market", "")
    audience = run_data.get("audience", "")

    prompt = (
        f"Generate {queries_per_competitor} concise Google search queries (3 to 6 words each) "
        f"to investigate competitor '{competitor_name}' in the {market} market, focused on "
        f"pricing, market strategy, products, or positioning relevant to {audience}.\n"
        "Return ONLY a JSON array of query strings.\n"
        f"Example: [\"{competitor_name} pricing strategy {market}\", \"{competitor_name} market share {market}\"]"
    )

    models_to_try = GEMINI_FALLBACK_MODELS or [GEMINI_MODEL_NAME]
    for model_name in models_to_try:
        try:
            with track_api_call(
                provider="gemini",
                call_type="competitor_query_generation",
                stage="Agent3_QueryGen",
                model=model_name,
                research_run_id=run_data.get("id")
            ) as tracker:
                pacer.wait_for_turn(model_name)
                resp = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        response_mime_type="application/json"
                    )
                )
                tracker.set_response(resp)

            raw = resp.text if hasattr(resp, "text") and resp.text else ""
            cleaned = clean_json_response(raw)
            parsed = json.loads(cleaned)
            if isinstance(parsed, list) and len(parsed) >= 1:
                return [str(q).strip() for q in parsed[:queries_per_competitor]]
        except Exception as exc:
            logger.warning(f"Error generating queries for {competitor_name} with {model_name}: {exc}")
            pacer.handle_429(model_name, exc, default_wait=1.0)

    # Fallback queries
    return [
        f"{competitor_name} pricing strategy {market}",
        f"{competitor_name} market share {market}"
    ]


def build_competitive_synthesis_prompt(
    run_data: dict,
    all_evidence: List[dict],
    feedback: Optional[str] = None
) -> str:
    """Builds prompt for Gemini to synthesize structured competitive intelligence."""
    brand = run_data.get("brand", "Unknown")
    market = run_data.get("market", "Unknown")
    audience = run_data.get("audience", "Unknown")
    objective = run_data.get("objective", "Unknown")

    ev_lines = []
    for ev in all_evidence:
        ev_id = ev["id"]
        agent_tag = f"[{ev.get('source_agent', 'agent1')}]"
        topic = ev.get("topic") or "competition"
        claim = ev.get("claim") or ""
        excerpt = ev.get("excerpt") or ""
        line = f"- [ID: {ev_id}] {agent_tag} ({topic}): Claim: \"{claim}\""
        if excerpt and excerpt.strip() != claim.strip():
            line += f" | Excerpt: \"{excerpt[:180]}\""
        ev_lines.append(line)

    ev_block = "\n".join(ev_lines)
    valid_ids_str = ", ".join(str(ev["id"]) for ev in all_evidence)

    prompt = (
        "You are an elite Competitive Intelligence Director.\n"
        "Synthesize all research evidence (both market-level and competitor-targeted) into structured competitive intelligence.\n\n"
        "### RESEARCH BRIEF:\n"
        f"- Target Brand: {brand}\n"
        f"- Market: {market}\n"
        f"- Audience: {audience}\n"
        f"- Objective: {objective}\n\n"
        "### ALL EVIDENCE ITEMS (CITE THESE IDS):\n"
        f"{ev_block}\n\n"
        "### CRITICAL CONSTRAINT 1 - PERMITTED EVIDENCE IDS:\n"
        f"You may ONLY cite evidence IDs from this exact list: [{valid_ids_str}].\n"
        "Every competitor entry, market_structure, and competitive_gap MUST cite valid supporting_evidence_ids.\n\n"
        "### CRITICAL CONSTRAINT 2 - STRICT NUMERICAL GROUNDING (ZERO FABRICATION):\n"
        "- Do NOT state any specific percentage, market share (e.g. '6%'), dollar value, CAGR, or statistic "
        "UNLESS that exact figure explicitly appears in the claim or excerpt of the cited evidence items.\n"
        "- If no cited evidence contains a relevant number, keep statements qualitative (describe relative positioning, scale, strengths).\n"
        "- Automated regex checks will verify every number. Any ungrounded figure will be REJECTED.\n\n"
        "### REQUIRED JSON OUTPUT SCHEMA:\n"
        "Return ONLY a valid JSON object matching this exact schema:\n"
        "{\n"
        '  "competitors": [\n'
        "    {\n"
        '      "name": "Competitor Name",\n'
        '      "positioning": "Summary of market positioning and target segment appeal",\n'
        '      "strengths": ["Strength 1", "Strength 2"],\n'
        '      "weaknesses": ["Weakness 1", "Weakness 2"],\n'
        '      "supporting_evidence_ids": [id1, id2]\n'
        "    }\n"
        "  ],\n"
        '  "market_structure": {\n'
        '    "concentration": "Description of market concentration, market shares, and dynamics",\n'
        '    "supporting_evidence_ids": [id1]\n'
        "  },\n"
        '  "competitive_gaps": [\n'
        "    {\n"
        '      "id": 1,\n'
        '      "gap": "Strategic unmet market gap, whitespace, or competitive vulnerability",\n'
        '      "supporting_evidence_ids": [id1, id2]\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )

    if feedback:
        prompt += f"\n\n### RETRY INSTRUCTIONS (FIX PREVIOUS ERRORS):\n{feedback}\n"

    return prompt


def validate_competitive_intelligence(
    data: dict,
    valid_ids: Set[int],
    evidence_lookup: Dict[int, dict]
) -> tuple[dict, List[str]]:
    """Validates competitive intelligence schema, evidence ID grounding, and numerical grounding."""
    errors = []

    if not isinstance(data, dict):
        return {}, ["Response is not a JSON object."]

    # 1. Validate competitors list
    competitors = data.get("competitors")
    if not isinstance(competitors, list) or len(competitors) == 0:
        errors.append("Output must contain at least one competitor in 'competitors' list.")
        valid_competitors = []
    else:
        valid_competitors = []
        for idx, comp in enumerate(competitors, 1):
            if not isinstance(comp, dict):
                errors.append(f"Competitor #{idx} is not a JSON object.")
                continue

            name = str(comp.get("name", "")).strip()
            if not name:
                errors.append(f"Competitor #{idx} missing name.")
                continue

            positioning = str(comp.get("positioning", "")).strip()
            strengths = [str(s).strip() for s in comp.get("strengths", []) if str(s).strip()]
            weaknesses = [str(w).strip() for w in comp.get("weaknesses", []) if str(w).strip()]
            raw_ids = comp.get("supporting_evidence_ids", [])

            if not isinstance(raw_ids, list) or len(raw_ids) == 0:
                errors.append(f"Competitor '{name}' must cite at least one supporting_evidence_id.")
                continue

            try:
                c_ids = sorted(list(set(int(cid) for cid in raw_ids)))
            except (ValueError, TypeError):
                errors.append(f"Competitor '{name}' has non-integer evidence IDs: {raw_ids}.")
                continue

            hallucinated = [cid for cid in c_ids if cid not in valid_ids]
            if hallucinated:
                errors.append(f"Competitor '{name}' cited hallucinated evidence IDs: {hallucinated}.")
                continue

            # Numerical grounding check
            full_comp_text = f"{positioning} {' '.join(strengths)} {' '.join(weaknesses)}"
            num_errs = verify_text_numerical_grounding(
                full_comp_text, c_ids, evidence_lookup, f"Competitor: {name}"
            )
            if num_errs:
                errors.extend(num_errs)
                continue

            valid_competitors.append({
                "name": name,
                "positioning": positioning,
                "strengths": strengths,
                "weaknesses": weaknesses,
                "supporting_evidence_ids": c_ids
            })

    # 2. Validate market_structure
    market_struct = data.get("market_structure")
    valid_market_struct = {}
    if not isinstance(market_struct, dict):
        errors.append("'market_structure' must be a JSON object.")
    else:
        concentration = str(market_struct.get("concentration", "")).strip()
        raw_m_ids = market_struct.get("supporting_evidence_ids", [])
        if not concentration:
            errors.append("'market_structure.concentration' text is missing.")
        if not isinstance(raw_m_ids, list) or len(raw_m_ids) == 0:
            errors.append("'market_structure' must cite at least one supporting_evidence_id.")
        else:
            try:
                m_ids = sorted(list(set(int(cid) for cid in raw_m_ids)))
                hallucinated_m = [cid for cid in m_ids if cid not in valid_ids]
                if hallucinated_m:
                    errors.append(f"'market_structure' cited hallucinated evidence IDs: {hallucinated_m}.")
                else:
                    num_errs_m = verify_text_numerical_grounding(
                        concentration, m_ids, evidence_lookup, "market_structure"
                    )
                    if num_errs_m:
                        errors.extend(num_errs_m)
                    else:
                        valid_market_struct = {
                            "concentration": concentration,
                            "supporting_evidence_ids": m_ids
                        }
            except (ValueError, TypeError):
                errors.append(f"'market_structure' has non-integer IDs: {raw_m_ids}.")

    # 3. Validate competitive_gaps
    gaps = data.get("competitive_gaps")
    valid_gaps = []
    if not isinstance(gaps, list) or len(gaps) == 0:
        errors.append("Output must contain at least one gap in 'competitive_gaps'.")
    else:
        for idx, g in enumerate(gaps, 1):
            if not isinstance(g, dict):
                errors.append(f"Competitive gap #{idx} is not a JSON object.")
                continue

            gap_text = str(g.get("gap", "")).strip()
            if not gap_text:
                errors.append(f"Competitive gap #{idx} text is empty.")
                continue

            raw_g_ids = g.get("supporting_evidence_ids", [])
            if not isinstance(raw_g_ids, list) or len(raw_g_ids) == 0:
                errors.append(f"Competitive gap #{idx} must cite at least one supporting_evidence_id.")
                continue

            try:
                g_ids = sorted(list(set(int(cid) for cid in raw_g_ids)))
                hallucinated_g = [cid for cid in g_ids if cid not in valid_ids]
                if hallucinated_g:
                    errors.append(f"Competitive gap #{idx} cited hallucinated evidence IDs: {hallucinated_g}.")
                    continue

                num_errs_g = verify_text_numerical_grounding(
                    gap_text, g_ids, evidence_lookup, f"Competitive Gap #{idx}"
                )
                if num_errs_g:
                    errors.extend(num_errs_g)
                    continue

                gap_id = g.get("id") if isinstance(g.get("id"), int) else idx
                valid_gaps.append({
                    "id": gap_id,
                    "gap": gap_text,
                    "supporting_evidence_ids": g_ids
                })
            except (ValueError, TypeError):
                errors.append(f"Competitive gap #{idx} has non-integer evidence IDs: {raw_g_ids}.")

    cleaned_result = {
        "competitors": valid_competitors,
        "market_structure": valid_market_struct,
        "competitive_gaps": valid_gaps
    }
    return cleaned_result, errors


def run_competitive_intelligence_pipeline(
    research_run_id: str,
    client: Optional[genai.Client] = None,
    serpapi_key: Optional[str] = None,
    max_competitors: int = 2,
    queries_per_competitor: int = 2,
    max_retries: int = 2
) -> Dict[str, Any]:
    """
    Agent 3 Pipeline Execution:
    1. Reads run brief and existing evidence from PostgreSQL.
    2. Identifies named competitors from evidence claims via Gemini.
    3. Generates 2 targeted queries per competitor.
    4. Searches via SerpAPI and extracts competitor evidence via Gemini.
    5. Saves competitor evidence to evidence table (source_agent='agent3').
    6. Synthesizes competitive intelligence report with strict ID and numerical grounding.
    7. Saves to competitive_intelligence table and returns.
    """
    logger.info(f"=== Starting Agent 3 Competitive Intelligence Pipeline for Run: {research_run_id} ===")

    # 1. Fetch run and existing evidence
    run_data = get_run_with_evidence(research_run_id)
    if not run_data:
        logger.error(f"Research run '{research_run_id}' not found in database.")
        raise ValueError(f"Research run '{research_run_id}' not found.")

    existing_evidence = run_data.get("evidence", [])
    if not existing_evidence:
        logger.error(f"No evidence found for research run '{research_run_id}'.")
        raise ValueError(f"No evidence found for research run '{research_run_id}'. Cannot run competitive intelligence.")

    if client is None:
        from services.credential_manager import get_gemini_client
        from database import get_research_run_owner
        _owner_id = get_research_run_owner(research_run_id)
        client = get_gemini_client(user_id=_owner_id)

    # Resolve SerpAPI key from user credentials (never from raw env var)
    if serpapi_key is None:
        from services.credential_manager import get_serpapi_key as _get_serpapi
        from database import get_research_run_owner as _get_owner
        serpapi_key = _get_serpapi(user_id=_get_owner(research_run_id))

    # 2. Identify named competitors mentioned in existing evidence claims
    logger.info("Scanning existing evidence to identify named competitors...")
    competitor_names = identify_competitors_from_evidence(
        client, run_data, existing_evidence, max_competitors=max_competitors
    )
    logger.info(f"Identified competitors for targeted research: {competitor_names}")

    # 3 & 4. Generate targeted queries and run SerpAPI + Extraction
    total_serpapi_calls = 0
    new_competitor_evidence = []

    brief_dict = {
        "brand": run_data["brand"],
        "market": run_data["market"],
        "audience": run_data["audience"],
        "objective": run_data["objective"]
    }

    for comp_name in competitor_names:
        queries = generate_competitor_queries(
            client, run_data, comp_name, queries_per_competitor=queries_per_competitor
        )
        logger.info(f"Generated {len(queries)} targeted queries for competitor '{comp_name}': {queries}")

        for q in queries:
            try:
                total_serpapi_calls += 1
                logger.info(f"[SerpAPI Call #{total_serpapi_calls}] Searching: '{q}'")
                docs = search_web_serpapi(q, api_key=serpapi_key, max_results=3)

                # Extract evidence from top 2 docs per query to conserve Gemini calls
                for doc in docs[:2]:
                    ext_res = extract_evidence_from_doc(client, doc, brief_dict)
                    if ext_res.get("status") == "success" and ext_res.get("data"):
                        d = ext_res["data"]
                        if d.get("claim"):
                            new_competitor_evidence.append(d)
                            logger.info(f"  [+] Extracted competitor claim: \"{d['claim'][:70]}...\"")
            except Exception as search_err:
                logger.warning(f"Error during search/extraction for query '{q}': {search_err}")

    logger.info(
        f"Agent 3 Targeted Research Stage Complete: Used {total_serpapi_calls} SerpAPI search calls. "
        f"Extracted {len(new_competitor_evidence)} new competitor evidence items."
    )

    # 5. Persist competitor evidence with source_agent='agent3'
    if new_competitor_evidence:
        save_competitor_evidence(research_run_id, new_competitor_evidence, client=client)

    # Re-fetch all evidence (original Agent 1 + newly added Agent 3)
    refreshed_run_data = get_run_with_evidence(research_run_id)
    all_evidence = refreshed_run_data.get("evidence", [])
    valid_ids: Set[int] = {ev["id"] for ev in all_evidence}
    evidence_lookup: Dict[int, dict] = {ev["id"]: ev for ev in all_evidence}
    logger.info(f"Synthesis Evidence Pool: {len(all_evidence)} total items available (IDs: {sorted(list(valid_ids))})")

    # 6. Synthesize Competitive Intelligence with strict grounding
    models_to_try = GEMINI_FALLBACK_MODELS or [GEMINI_MODEL_NAME]
    feedback: Optional[str] = None
    final_validated_data: Optional[dict] = None

    for model_name in models_to_try:
        for attempt in range(1, max_retries + 2):
            prompt = append_knowledge_guidance(
                build_competitive_synthesis_prompt(run_data, all_evidence, feedback),
                "competitor analysis positioning differentiation and research source limitations",
            )
            logger.info(f"Calling Gemini model '{model_name}' for competitive synthesis (Attempt {attempt})...")

            try:
                with track_api_call(
                    provider="gemini",
                    call_type="competitive_intelligence_synthesis",
                    stage="Agent3_Synthesis",
                    model=model_name,
                    research_run_id=research_run_id,
                    input_text=prompt[:400]
                ) as tracker:
                    pacer.wait_for_turn(model_name)
                    resp = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.3,
                            response_mime_type="application/json"
                        )
                    )
                    tracker.set_response(resp)

                raw_text = resp.text if hasattr(resp, "text") and resp.text else ""
                cleaned = clean_json_response(raw_text)
                parsed = json.loads(cleaned)

                validated_data, errors = validate_competitive_intelligence(parsed, valid_ids, evidence_lookup)

                if errors:
                    logger.warning(f"Competitive validation warnings on attempt {attempt}: {errors}")
                    if attempt <= max_retries:
                        feedback = (
                            "Your previous output contained validation and evidence-grounding errors:\n"
                            + "\n".join(f"- {e}" for e in errors)
                            + f"\nCRITICAL: You may ONLY cite valid IDs from: {sorted(list(valid_ids))}."
                            + "\nCRITICAL: Do NOT invent ANY numbers or percentages. Every number MUST appear verbatim in the cited evidence."
                        )
                        continue
                    else:
                        if validated_data.get("competitors"):
                            logger.info("Proceeding with validated competitive intelligence elements after retries.")
                            final_validated_data = validated_data
                            break
                        raise ValueError(f"Competitive intelligence validation failed after retries: {errors}")
                else:
                    logger.info("Successfully synthesized and validated competitive intelligence.")
                    final_validated_data = validated_data
                    break

            except Exception as exc:
                err_str = str(exc)
                logger.error(f"Error during competitive synthesis attempt {attempt} with {model_name}: {err_str}")
                is_transient = any(k in err_str.lower() for k in [
                    "429", "resource_exhausted", "quota", "rate limit",
                    "503", "unavailable", "high demand", "overloaded", "server error", "500"
                ])
                if is_transient:
                    logger.warning(f"Transient error on '{model_name}'. Trying fallback model...")
                    pacer.handle_429(model_name, exc, default_wait=2.0)
                    break
                if attempt > max_retries:
                    raise

        if final_validated_data:
            break

    if not final_validated_data:
        raise RuntimeError("Failed to synthesize competitive intelligence: no valid output generated.")

    # 7. Persist to competitive_intelligence table
    saved = save_competitive_intelligence(research_run_id, final_validated_data)
    logger.info(f"Successfully saved competitive intelligence for run '{research_run_id}'.")
    return saved
