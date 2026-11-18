"""
Agent 2: Consumer Intelligence Synthesis Pipeline.
----------------------------------------------------
Synthesizes structured consumer intelligence (segments, motivations,
pain points, behavioral patterns) grounded in evidence from Agent 1 research runs.
Strictly validates that:
1. Every cited evidence_id actually exists in the database.
2. Every number, percentage, or statistic mentioned in the insight text is
   strictly grounded in the cited evidence items' claim or excerpt (zero numerical fabrication).
"""

import json
import logging
import re
import time
from typing import Dict, Any, List, Optional, Set

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_MODEL_NAME, GEMINI_FALLBACK_MODELS
from database import (
    get_run_with_evidence,
    save_consumer_insights,
    track_api_call,
)
from evidence_extractor import clean_json_response
from gemini_pacer import pacer
from intelligence.knowledge import append_knowledge_guidance

logger = logging.getLogger("CatchyLabAgent2")

VALID_INSIGHT_TYPES = {
    "segment",
    "motivation",
    "pain_point",
    "behavioral_pattern"
}

# Type normalization mapping for slight LLM variations (plural, dashes, spaces)
TYPE_NORMALIZATION = {
    "segment": "segment",
    "segments": "segment",
    "consumer_segment": "segment",
    "consumer_segments": "segment",
    "motivation": "motivation",
    "motivations": "motivation",
    "pain_point": "pain_point",
    "pain_points": "pain_point",
    "painpoint": "pain_point",
    "painpoints": "pain_point",
    "behavioral_pattern": "behavioral_pattern",
    "behavioral_patterns": "behavioral_pattern",
    "behavior_pattern": "behavioral_pattern",
    "behavior_patterns": "behavioral_pattern",
    "behavioral": "behavioral_pattern",
}

# Regex pattern capturing statistical & quantified figures:
# 1. Percentages: '65%', '77.5%', '14.52%', '65 percent'
# 2. Compound growth: 'CAGR of 14.52%', '4.90% CAGR'
# 3. Currency / Market sizing: '$1.59 Billion', 'USD 10.72 Billion', '$108.3B', 'â‚¬50M'
# 4. Magnitudes: '10.72 Billion', '785.63 Million'
# 5. Multipliers & Proportions: '2x', '7 out of 10'
# 6. Decimals: '14.52', '10.72', '0.71'
STATS_PATTERN = re.compile(
    r"(?:"
    r"\b\d+(?:\.\d+)?\s*%"                                                      # e.g. 65%, 77%, 14.52%
    r"|\b\d+(?:\.\d+)?\s*percent\b"                                             # e.g. 65 percent
    r"|CAGR\s*(?:of\s*)?\d+(?:\.\d+)?%?"                                        # e.g. CAGR of 14.52%
    r"|\b\d+(?:\.\d+)?%\s*CAGR\b"                                               # e.g. 14.52% CAGR
    r"|(?:[\$â‚¬Â£]|(?:USD|INR|EUR|GBP)\s*)\d+(?:\.\d+)?(?:\s*(?:billion|million|trillion|crore|lakh|b|m|k))?\b" # e.g. $1.59 Billion
    r"|\b\d+(?:\.\d+)?\s*(?:billion|million|trillion|crore|lakh)\b"             # e.g. 10.72 Billion
    r"|\b\d+(?:\.\d+)?\s*x\b"                                                   # e.g. 2x, 3x
    r"|\b\d+\s*(?:out of|\/)\s*\d+\b"                                           # e.g. 7 out of 10
    r"|\b\d+\.\d+\b"                                                            # Any decimal number, e.g. 14.52, 0.71, 108.3
    r")",
    re.IGNORECASE
)


def extract_statistical_figures(text: str) -> List[Dict[str, str]]:
    """
    Extracts all statistical figures, percentages, and metrics from text.
    Returns list of dicts with 'raw' match and 'core_num' numerical token.
    """
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


def build_synthesis_prompt(run_data: dict, evidence_items: List[dict], feedback: Optional[str] = None) -> str:
    """Constructs prompt for Gemini synthesis with strict grounding, zero numerical fabrication, and JSON schema."""
    brand = run_data.get("brand", "Unknown")
    market = run_data.get("market", "Unknown")
    audience = run_data.get("audience", "Unknown")
    objective = run_data.get("objective", "Unknown")

    evidence_text_lines = []
    for ev in evidence_items:
        ev_id = ev["id"]
        topic = ev.get("topic") or "General"
        claim = ev.get("claim") or ""
        excerpt = ev.get("excerpt") or ""
        line = f"- [ID: {ev_id}] ({topic}): Claim: \"{claim}\""
        if excerpt and excerpt.strip() != claim.strip():
            line += f" | Excerpt: \"{excerpt[:200]}\""
        evidence_text_lines.append(line)

    evidence_block = "\n".join(evidence_text_lines)
    valid_ids_str = ", ".join(str(ev["id"]) for ev in evidence_items)

    prompt = (
        "You are an elite Consumer Intelligence & Market Strategy Director.\n"
        "Your task is to synthesize verified market evidence into deep, actionable consumer intelligence.\n\n"
        "### RESEARCH CONTEXT:\n"
        f"- Brand: {brand}\n"
        f"- Target Market: {market}\n"
        f"- Target Audience: {audience}\n"
        f"- Core Objective: {objective}\n\n"
        "### VERIFIED EVIDENCE FROM RESEARCH RUN:\n"
        f"{evidence_block}\n\n"
        "### CRITICAL CONSTRAINT 1 - PERMITTED EVIDENCE IDS:\n"
        f"You may ONLY cite evidence IDs from this exact list: [{valid_ids_str}].\n"
        "NEVER invent, hallucinate, or reference any ID not in this list.\n\n"
        "### CRITICAL CONSTRAINT 2 - STRICT NUMERICAL GROUNDING (ZERO FABRICATION):\n"
        "- You must NOT state any specific number, percentage, dollar/currency amount, market size, growth rate, or statistic "
        "in an insight's text UNLESS that exact figure explicitly appears in the claim or excerpt of the evidence items cited for that specific insight.\n"
        "- If no cited evidence contains a relevant number or percentage, the insight MUST remain purely qualitative (describe sentiments, preferences, lifestyle tensions, brand perceptions â€” do NOT invent or estimate numbers like '65%', '40%', or '3x').\n"
        "- Do NOT attribute a percentage to a different specific material or category than what is stated in the cited source "
        "(e.g., if a source states '65% prefer recycled materials', do NOT state '65% prefer recycled polyester').\n"
        "- Every single number and percentage in your output will be automatically extracted via regex and verified against the text of the cited evidence items. Any insight containing an unverified statistic will be REJECTED.\n\n"
        "### SYNTHESIS OBJECTIVES:\n"
        "Synthesize the evidence into four structured intelligence categories:\n"
        "1. 'segment': Distinct behavioral or psychographic consumer clusters within the target audience "
        "(their identity, economic posture, priorities, and tensions).\n"
        "2. 'motivation': Underlying emotional, social, practical, or aspirational drivers compelling consumers "
        "toward or away from products/solutions.\n"
        "3. 'pain_point': Critical frictions, barriers, price sensitivities, greenwashing skepticism, or unmet market needs.\n"
        "4. 'behavioral_pattern': Observable routines, shopping habits, channel preferences, and verification behaviors.\n\n"
        "### QUALITY STANDARD - GENUINE SYNTHESIS VS REPACKAGING:\n"
        "- DO NOT merely rephrase or repeat single isolated claims (e.g. do not just write 'Gen Z likes vegan shoes').\n"
        "- Connect multiple pieces of evidence together (synthesize 2 to 4 evidence IDs per insight where applicable) "
        "to reveal non-obvious tensions, consumer trade-offs, and strategic implications.\n"
        "- Provide 2 to 3 deep, rich insights for EACH of the 4 categories (total 8 to 12 insights).\n\n"
        "### REQUIRED JSON FORMAT:\n"
        "Return ONLY a valid JSON object matching this exact schema:\n"
        "{\n"
        '  "insights": [\n'
        "    {\n"
        '      "insight_type": "segment" | "motivation" | "pain_point" | "behavioral_pattern",\n'
        '      "insight_text": "Detailed synthesized insight articulating the consumer dynamic...",\n'
        '      "supporting_evidence_ids": [id1, id2]\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )

    if feedback:
        prompt += f"\n\n### RETRY INSTRUCTIONS (FIX PREVIOUS ERRORS):\n{feedback}\n"

    return prompt


def validate_insights(
    raw_insights: List[dict],
    valid_ids: Set[int],
    evidence_lookup: Dict[int, dict]
) -> tuple[List[dict], List[str]]:
    """
    Validates synthesized insights against schema, valid evidence IDs,
    AND verifies that every number/percentage mentioned in insight_text is grounded
    in the text of at least one of the cited evidence items.
    Returns (validated_insights, error_messages).
    """
    validated = []
    errors = []

    for idx, item in enumerate(raw_insights, 1):
        if not isinstance(item, dict):
            errors.append(f"Insight #{idx} is not a JSON object.")
            continue

        raw_type = str(item.get("insight_type", "")).strip().lower()
        norm_type = TYPE_NORMALIZATION.get(raw_type)
        if not norm_type or norm_type not in VALID_INSIGHT_TYPES:
            errors.append(f"Insight #{idx} has invalid insight_type '{raw_type}'. Must be one of: {list(VALID_INSIGHT_TYPES)}.")
            continue

        text = str(item.get("insight_text", "")).strip()
        if len(text) < 20:
            errors.append(f"Insight #{idx} text is too short or empty (min 20 chars).")
            continue

        cited_ids = item.get("supporting_evidence_ids")
        if not isinstance(cited_ids, list) or len(cited_ids) == 0:
            errors.append(f"Insight #{idx} must cite at least one supporting_evidence_id.")
            continue

        # Check for integer conversion
        parsed_ids = []
        has_id_type_err = False
        for cid in cited_ids:
            try:
                parsed_ids.append(int(cid))
            except (ValueError, TypeError):
                has_id_type_err = True
                break

        if has_id_type_err:
            errors.append(f"Insight #{idx} cited non-integer evidence IDs: {cited_ids}.")
            continue

        # Check for hallucinated IDs
        hallucinated = [cid for cid in parsed_ids if cid not in valid_ids]
        if hallucinated:
            errors.append(f"Insight #{idx} cited hallucinated evidence IDs not in DB: {hallucinated}.")
            continue

        clean_ids = sorted(list(set(parsed_ids)))

        # Numerical & Statistical Grounding Check:
        # Extract any numbers, percentages, currencies, or growth rates mentioned in insight_text
        figures = extract_statistical_figures(text)
        if figures:
            # Gather combined text from all cited evidence items
            combined_evidence_text = " ".join([
                f"{evidence_lookup[cid].get('claim', '')} {evidence_lookup[cid].get('excerpt', '')}"
                for cid in clean_ids if cid in evidence_lookup
            ])

            insight_has_num_err = False
            for fig in figures:
                core = fig["core_num"]
                raw_fig = fig["raw"]

                # Match core number bounded by non-digits in combined evidence text
                num_pattern = r"(?<!\d)" + re.escape(core) + r"(?!\d)"
                if not re.search(num_pattern, combined_evidence_text):
                    errors.append(
                        f"Insight #{idx} ({norm_type}) contains ungrounded/hallucinated statistic '{raw_fig}' "
                        f"(number: {core}) not found in cited evidence items {clean_ids}."
                    )
                    insight_has_num_err = True

            if insight_has_num_err:
                continue

        validated.append({
            "insight_type": norm_type,
            "insight_text": text,
            "supporting_evidence_ids": clean_ids
        })

    return validated, errors


def run_consumer_intelligence_pipeline(
    research_run_id: str,
    client: Optional[genai.Client] = None,
    max_retries: int = 2
) -> List[Dict[str, Any]]:
    """
    Single pipeline execution function for Agent 2:
    1. Fetches run metadata and all evidence items from database.
    2. Calls Gemini to synthesize segments, motivations, pain points, behavioral patterns.
    3. Validates that every cited evidence_id actually exists in the run's evidence.
    4. Validates that every number/percentage mentioned in insight_text traces back to cited evidence.
    5. Retries if Gemini hallucinates IDs or fabricates numbers.
    6. Persists validated insights to 'consumer_insights' table.
    7. Returns saved insights.
    """
    logger.info(f"=== Starting Agent 2 Consumer Intelligence Pipeline for Run: {research_run_id} ===")

    # 1. Fetch run and evidence from database
    run_data = get_run_with_evidence(research_run_id)
    if not run_data:
        logger.error(f"Research run '{research_run_id}' not found in database.")
        raise ValueError(f"Research run '{research_run_id}' not found.")

    evidence_items = run_data.get("evidence", [])
    if not evidence_items:
        logger.error(f"No evidence found for research run '{research_run_id}'.")
        raise ValueError(
            f"No evidence found for research run '{research_run_id}'. "
            "Cannot synthesize consumer intelligence: research run must contain at least one verified evidence item."
        )

    valid_ids: Set[int] = {ev["id"] for ev in evidence_items}
    evidence_lookup: Dict[int, dict] = {ev["id"]: ev for ev in evidence_items}
    logger.info(f"Loaded {len(evidence_items)} evidence records with IDs: {sorted(list(valid_ids))}")

    # Initialize Gemini client if not provided
    if client is None:
        from services.credential_manager import get_gemini_client
        from database import get_research_run_owner
        _owner_id = get_research_run_owner(research_run_id)
        client = get_gemini_client(user_id=_owner_id)

    models_to_try = GEMINI_FALLBACK_MODELS or [GEMINI_MODEL_NAME]
    feedback: Optional[str] = None
    final_validated_insights: List[dict] = []

    for model_name in models_to_try:
        for attempt in range(1, max_retries + 2):
            prompt = append_knowledge_guidance(
                build_synthesis_prompt(run_data, evidence_items, feedback),
                "consumer segmentation motivations jobs to be done purchase barriers behavior",
            )
            logger.info(f"Calling Gemini model '{model_name}' (Attempt {attempt})...")

            try:
                with track_api_call(
                    provider="gemini",
                    call_type="consumer_intelligence_synthesis",
                    stage="Agent2_Synthesis",
                    model=model_name,
                    research_run_id=research_run_id,
                    input_text=prompt[:400]
                ) as tracker:
                    pacer.wait_for_turn(model_name)
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.3,
                            response_mime_type="application/json"
                        )
                    )
                    tracker.set_response(response)

                raw_text = response.text if hasattr(response, "text") and response.text else ""
                cleaned = clean_json_response(raw_text)
                parsed = json.loads(cleaned)

                # Extract list of insights
                if isinstance(parsed, dict) and "insights" in parsed:
                    insights_list = parsed["insights"]
                elif isinstance(parsed, list):
                    insights_list = parsed
                else:
                    raise ValueError(f"Unexpected JSON format: expected dict with 'insights' or list, got {type(parsed)}")

                # Validate insights against evidence IDs and numerical grounding
                validated, errors = validate_insights(insights_list, valid_ids, evidence_lookup)

                if errors:
                    logger.warning(f"Validation warnings on attempt {attempt}: {errors}")
                    if attempt <= max_retries:
                        feedback = (
                            "Your previous output contained evidence-grounding errors:\n"
                            + "\n".join(f"- {e}" for e in errors)
                            + f"\nCRITICAL: You may ONLY cite valid IDs from: {sorted(list(valid_ids))}."
                            + "\nCRITICAL: Do NOT invent ANY numbers or percentages. Every number MUST appear verbatim in the cited evidence items, or keep the insight qualitative without numbers."
                        )
                        continue
                    else:
                        # On final retry, filter only valid insights if some passed
                        if validated:
                            logger.info(f"Proceeding with {len(validated)} validated insights after retries.")
                            final_validated_insights = validated
                            break
                        raise ValueError(f"Consumer intelligence validation failed after retries: {errors}")
                else:
                    logger.info(f"Successfully synthesized and validated {len(validated)} consumer insights.")
                    final_validated_insights = validated
                    break

            except Exception as exc:
                err_str = str(exc)
                logger.error(f"Error during synthesis attempt {attempt} with {model_name}: {err_str}")
                is_transient = any(k in err_str.lower() for k in [
                    "429", "resource_exhausted", "quota", "rate limit",
                    "503", "unavailable", "high demand", "overloaded", "server error", "500"
                ])
                if is_transient:
                    logger.warning(f"Transient error on '{model_name}' ({err_str[:100]}). Trying fallback model...")
                    pacer.handle_429(model_name, exc, default_wait=2.0)
                    break
                if attempt > max_retries:
                    raise

        if final_validated_insights:
            break

    if not final_validated_insights:
        raise RuntimeError("Failed to synthesize consumer intelligence: no valid insights generated.")

    # 5. Persist to database
    saved = save_consumer_insights(research_run_id, final_validated_insights)
    logger.info(f"Successfully saved {len(saved)} consumer insights to database for run '{research_run_id}'.")
    return saved

