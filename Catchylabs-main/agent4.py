"""
Agent 4: Creative Gap Detection Pipeline.
-----------------------------------------
Synthesizes actionable creative gaps by cross-referencing already-synthesized
Agent 2 (Consumer Insights) and Agent 3 (Competitive Intelligence) outputs.

Identifies strategic white spaces where audience desire/motivation/tension
meets competitive absence, vulnerability, or failure to deliver.

Strict Grounding & Cross-Reference Constraints:
1. Every creative gap MUST reference AT LEAST ONE real consumer_insight_id (Agent 2)
   AND AT LEAST ONE real competitor name (Agent 3). Single-sided entries are rejected.
2. Every cited consumer_insight_id must exist in the run's saved consumer insights.
3. Every cited competitor name must exist in the run's saved competitor roster.
4. Numerical Grounding: every statistic, percentage, or number in the gap text must
   appear verbatim in the combined text of the cited consumer insights and competitor entries
   (the synthesized layer is the source of truth).
5. Must produce novel creative strategy synthesis, not merely repeat Agent 3 competitive gaps.
"""

import json
import logging
import re
import time
from typing import Dict, Any, List, Optional, Set, Tuple

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_MODEL_NAME, GEMINI_FALLBACK_MODELS
from database import (
    get_run_with_evidence,
    get_consumer_insights,
    get_competitive_intelligence,
    save_creative_gaps,
    track_api_call,
)
from evidence_extractor import clean_json_response
from gemini_pacer import pacer
from intelligence.knowledge import append_knowledge_guidance

logger = logging.getLogger("CatchyLabAgent4")

# Regex pattern capturing statistical & quantified figures:
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


def build_creative_gaps_prompt(
    run_data: dict,
    consumer_insights: List[dict],
    competitive_intel: dict,
    feedback: Optional[str] = None
) -> str:
    """
    Constructs the prompt for Gemini to synthesize creative gaps at the intersection
    of consumer desires (Agent 2) and competitive absence (Agent 3).
    """
    brand = run_data.get("brand", "Unknown")
    market = run_data.get("market", "Unknown")
    audience = run_data.get("audience", "Unknown")
    objective = run_data.get("objective", "Unknown")

    # Format Consumer Insights (Agent 2)
    ci_lines = []
    valid_ci_ids = []
    for ci in consumer_insights:
        cid = ci["id"]
        valid_ci_ids.append(cid)
        itype = ci.get("insight_type", "insight")
        text_val = ci.get("insight_text", "")
        ci_lines.append(f"- [Insight ID: {cid}] ({itype}): \"{text_val}\"")
    ci_block = "\n".join(ci_lines)
    valid_ci_ids_str = ", ".join(str(cid) for cid in valid_ci_ids)

    # Format Competitors (Agent 3)
    competitors = competitive_intel.get("competitors", [])
    comp_lines = []
    valid_comp_names = []
    for comp in competitors:
        name = comp.get("name", "Unknown")
        valid_comp_names.append(name)
        pos = comp.get("positioning", "")
        strengths = ", ".join(comp.get("strengths", []))
        weaknesses = ", ".join(comp.get("weaknesses", []))
        comp_lines.append(
            f"- Competitor: \"{name}\"\n"
            f"  Positioning: \"{pos}\"\n"
            f"  Strengths: {strengths or 'None listed'}\n"
            f"  Weaknesses: {weaknesses or 'None listed'}"
        )
    comp_block = "\n".join(comp_lines)
    valid_comp_names_str = ", ".join(f'"{name}"' for name in valid_comp_names)

    # Format Market Structure (Agent 3)
    market_struct = competitive_intel.get("market_structure", {})
    market_concentration = market_struct.get("concentration", "Not specified")

    # Format Existing Competitive Gaps (Agent 3)
    existing_comp_gaps = competitive_intel.get("competitive_gaps", [])
    cg_lines = []
    valid_cg_ids = []
    for cg in existing_comp_gaps:
        cg_id = cg.get("id")
        if cg_id is not None:
            valid_cg_ids.append(cg_id)
        cg_text = cg.get("gap", "")
        cg_lines.append(f"- [Competitive Gap ID: {cg_id}] \"{cg_text}\"")
    cg_block = "\n".join(cg_lines) if cg_lines else "None listed"
    valid_cg_ids_str = ", ".join(str(cg_id) for cg_id in valid_cg_ids)

    prompt = (
        "You are an elite Chief Brand & Creative Strategist.\n"
        "Your mission is CREATIVE GAP DETECTION: uncover actionable creative, positioning, and storytelling territories "
        "that emerge exclusively at the intersection of consumer psychology/desire (Agent 2) and competitive absence/failure (Agent 3).\n\n"
        "### RESEARCH CONTEXT:\n"
        f"- Target Brand: {brand}\n"
        f"- Market: {market}\n"
        f"- Target Audience: {audience}\n"
        f"- Core Objective: {objective}\n\n"
        "### AGENT 2: VERIFIED CONSUMER INSIGHTS (CITE THESE IDS):\n"
        f"{ci_block}\n\n"
        "### AGENT 3: COMPETITIVE INTELLIGENCE:\n"
        f"Market Structure Concentration: {market_concentration}\n\n"
        f"Competitors Roster:\n{comp_block}\n\n"
        f"Agent 3 Existing Competitive Gaps:\n{cg_block}\n\n"
        "### CRITICAL CONSTRAINT 1 - MANDATORY DUAL-SIDED CROSS-REFERENCING:\n"
        "- Every single creative gap MUST cite:\n"
        f"  1. AT LEAST ONE valid Consumer Insight ID from this exact list: [{valid_ci_ids_str}].\n"
        f"  2. AT LEAST ONE valid Competitor Name from this exact roster: [{valid_comp_names_str}].\n"
        f"  3. (Optional) Supporting Competitive Gap IDs from Agent 3 if directly applicable: [{valid_cg_ids_str}].\n"
        "- REJECT any entry that only cites consumer insights without naming competitors, or only cites competitors without consumer insights.\n"
        "- A genuine creative gap ONLY exists where consumer desire directly collides with competitor deficiency.\n\n"
        "### CRITICAL CONSTRAINT 2 - NOVEL CREATIVE SYNTHESIS (DO NOT REHASH AGENT 3):\n"
        "- Agent 3 already identified broad market gaps above. DO NOT merely restate Agent 3's existing competitive gaps!\n"
        "- Agent 4 must find NOVEL creative opportunities that neither agent surfaced alone:\n"
        "  * What specific consumer emotional tension, aesthetic desire, or skepticism (from Agent 2) is being neglected or bungled by the named competitors (from Agent 3)?\n"
        "  * What creative brand narrative, proposition, or campaign territory could the target brand claim to win the audience?\n"
        "  * Provide a clear, evocative 'gap_title' and a comprehensive, strategic 'description' detailing the audience tension, competitor shortcomings, and the creative whitespace.\n"
        "  * Synthesize 3 to 5 distinct, high-impact creative gaps (each exploring a different strategic whitespace).\n\n"
        "### CRITICAL CONSTRAINT 3 - STRICT NUMERICAL GROUNDING (ZERO FABRICATION):\n"
        "- The provided synthesized outputs from Agent 2 and Agent 3 are the SOLE source of truth for facts and figures.\n"
        "- Do NOT invent, hallucinate, or estimate any numbers, percentages (e.g. '70%', '40%'), market shares, or metrics.\n"
        "- Any number or percentage mentioned in 'gap_title' or 'description' MUST appear verbatim in the text of the cited consumer insights, competitor profiles, or competitive gaps.\n"
        "- If the cited entries do not contain numbers, formulate the gap in qualitative, strategic terms.\n"
        "- Automated regex checks will verify every figure. Any ungrounded figure will be REJECTED.\n\n"
        "### REQUIRED JSON OUTPUT SCHEMA:\n"
        "Return ONLY a valid JSON object matching this exact schema:\n"
        "{\n"
        '  "creative_gaps": [\n'
        "    {\n"
        '      "gap_title": "Evocative, strategic title of the creative gap",\n'
        '      "description": "In-depth explanation articulating the intersection of consumer desire and competitor absence, explaining why competitors fail and how the brand can creatively seize this whitespace.",\n'
        '      "supporting_consumer_insight_ids": [id1, id2],\n'
        '      "supporting_competitor_names": ["Competitor1", "Competitor2"],\n'
        '      "supporting_competitive_gap_ids": [1],\n'
        '      "confidence": "high" | "medium" | "low"\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )

    if feedback:
        prompt += f"\n\n### RETRY INSTRUCTIONS (FIX PREVIOUS ERRORS):\n{feedback}\n"

    return prompt


def validate_creative_gaps(
    data: dict,
    valid_ci_ids: Set[int],
    ci_lookup: Dict[int, dict],
    valid_comp_names_map: Dict[str, str],
    comp_lookup: Dict[str, dict],
    valid_cg_ids: Set[int],
    comp_gap_lookup: Dict[int, dict]
) -> Tuple[List[dict], List[str]]:
    """
    Validates creative gaps for:
    1. Schema compliance (non-empty fields, correct types)
    2. Strict dual-sided cross-referencing (must cite at least one consumer insight AND at least one competitor)
    3. Valid IDs & competitor names (no hallucinations)
    4. Numerical grounding against combined cited text (zero fabrication)
    """
    errors: List[str] = []
    validated: List[dict] = []

    if not isinstance(data, dict):
        return [], ["Response is not a JSON object."]

    raw_gaps = data.get("creative_gaps")
    if not isinstance(raw_gaps, list) or len(raw_gaps) == 0:
        return [], ["Output must contain at least one creative gap in 'creative_gaps' list."]

    for idx, item in enumerate(raw_gaps, 1):
        if not isinstance(item, dict):
            errors.append(f"Creative gap #{idx} is not a JSON object.")
            continue

        title = str(item.get("gap_title", "")).strip()
        if len(title) < 5:
            errors.append(f"Creative gap #{idx} 'gap_title' is too short or missing.")
            continue

        desc = str(item.get("description", "")).strip()
        if len(desc) < 30:
            errors.append(f"Creative gap #{idx} 'description' is too short or missing (min 30 characters).")
            continue

        # 1. Validate supporting_consumer_insight_ids
        raw_ci_ids = item.get("supporting_consumer_insight_ids")
        if not isinstance(raw_ci_ids, list) or len(raw_ci_ids) == 0:
            errors.append(
                f"Creative gap #{idx} ('{title}') only cites competitor data without any consumer insights. "
                "Must cite AT LEAST ONE valid consumer insight."
            )
            continue

        parsed_ci_ids = []
        has_ci_type_err = False
        for cid in raw_ci_ids:
            try:
                parsed_ci_ids.append(int(cid))
            except (ValueError, TypeError):
                has_ci_type_err = True
                break

        if has_ci_type_err:
            errors.append(f"Creative gap #{idx} ('{title}') cited non-integer consumer insight IDs: {raw_ci_ids}.")
            continue

        clean_ci_ids = sorted(list(set(parsed_ci_ids)))
        hallucinated_ci = [cid for cid in clean_ci_ids if cid not in valid_ci_ids]
        if hallucinated_ci:
            errors.append(
                f"Creative gap #{idx} ('{title}') cited hallucinated consumer insight IDs: {hallucinated_ci}. "
                f"Valid IDs: {sorted(list(valid_ci_ids))}."
            )
            continue

        # 2. Validate supporting_competitor_names+
        raw_comp_names = item.get("supporting_competitor_names")
        if not isinstance(raw_comp_names, list) or len(raw_comp_names) == 0:
            errors.append(
                f"Creative gap #{idx} ('{title}') only cites consumer insights without any competitor reference. "
                "Must cite AT LEAST ONE valid competitor name."
            )
            continue

        clean_comp_names = []
        invalid_comp_names = []
        for cname in raw_comp_names:
            cname_str = str(cname).strip()
            canonical = valid_comp_names_map.get(cname_str.lower())
            if canonical:
                if canonical not in clean_comp_names:
                    clean_comp_names.append(canonical)
            else:
                invalid_comp_names.append(cname_str)

        if invalid_comp_names:
            errors.append(
                f"Creative gap #{idx} ('{title}') cited competitor names not in Agent 3 competitor list: {invalid_comp_names}. "
                f"Valid competitors: {list(valid_comp_names_map.values())}."
            )
            continue

        # 3. Validate supporting_competitive_gap_ids (optional, but must be valid if provided)
        raw_cg_ids = item.get("supporting_competitive_gap_ids", [])
        clean_cg_ids = []
        if isinstance(raw_cg_ids, list) and raw_cg_ids:
            for cgid in raw_cg_ids:
                try:
                    int_cgid = int(cgid)
                    if int_cgid in valid_cg_ids:
                        if int_cgid not in clean_cg_ids:
                            clean_cg_ids.append(int_cgid)
                    else:
                        errors.append(
                            f"Creative gap #{idx} ('{title}') cited unknown competitive gap ID: {int_cgid}. "
                            f"Valid IDs: {sorted(list(valid_cg_ids))}."
                        )
                except (ValueError, TypeError):
                    errors.append(f"Creative gap #{idx} ('{title}') cited non-integer competitive gap ID: {cgid}.")

        clean_cg_ids.sort()

        # 4. Confidence
        confidence = str(item.get("confidence", "high")).strip().lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = "high"

        # 5. Numerical Grounding Check against combined cited text
        # Combine text of cited consumer insights, competitors, and competitive gaps
        cited_text_snippets = []
        for cid in clean_ci_ids:
            if cid in ci_lookup:
                cited_text_snippets.append(ci_lookup[cid].get("insight_text", ""))

        for cname in clean_comp_names:
            if cname in comp_lookup:
                c_data = comp_lookup[cname]
                cited_text_snippets.append(c_data.get("name", ""))
                cited_text_snippets.append(c_data.get("positioning", ""))
                cited_text_snippets.extend(c_data.get("strengths", []))
                cited_text_snippets.extend(c_data.get("weaknesses", []))

        for cgid in clean_cg_ids:
            if cgid in comp_gap_lookup:
                cited_text_snippets.append(comp_gap_lookup[cgid].get("gap", ""))

        combined_cited_text = " ".join(cited_text_snippets)

        # Extract all statistical figures from gap_title and description
        gap_full_text = f"{title} {desc}"
        figures = extract_statistical_figures(gap_full_text)
        gap_has_num_err = False

        for fig in figures:
            core = fig["core_num"]
            raw_fig = fig["raw"]
            num_pattern = r"(?<!\d)" + re.escape(core) + r"(?!\d)"
            if not re.search(num_pattern, combined_cited_text):
                errors.append(
                    f"Creative gap #{idx} ('{title}') contains ungrounded/hallucinated statistic '{raw_fig}' "
                    f"(number: {core}) not found in the combined text of cited consumer insights or competitor data."
                )
                gap_has_num_err = True

        if gap_has_num_err:
            continue

        validated.append({
            "gap_title": title,
            "description": desc,
            "supporting_consumer_insight_ids": clean_ci_ids,
            "supporting_competitor_names": clean_comp_names,
            "supporting_competitive_gap_ids": clean_cg_ids,
            "confidence": confidence
        })

    return validated, errors


def run_creative_gaps_pipeline(
    research_run_id: str,
    client: Optional[genai.Client] = None,
    max_retries: int = 2
) -> List[Dict[str, Any]]:
    """
    Executes the Agent 4 Creative Gap Detection pipeline:
    1. Fetches run data and verifies both Agent 2 (Consumer Insights) and
       Agent 3 (Competitive Intelligence) have completed for this run.
       FAILS CLEARLY with an informative error if either is missing.
    2. Makes ONE Gemini call passing both full synthesized outputs,
       instructing it to discover intersections between audience desire and competitive absence.
    3. Strictly validates grounding: FK checks against consumer_insights.id,
       competitor name verification against Agent 3 competitors, and numerical
       grounding check against combined cited text.
    4. Rejects any gap citing only one side.
    5. Persists validated creative gaps to 'creative_gaps' table.
    6. Uses ZERO SerpAPI calls.
    Returns list of saved creative gaps.
    """
    logger.info(f"=== Starting Agent 4 Creative Gap Detection Pipeline for Run: {research_run_id} ===")

    # 1. Fetch run metadata
    run_data = get_run_with_evidence(research_run_id)
    if not run_data:
        logger.error(f"Research run '{research_run_id}' not found in database.")
        raise ValueError(f"Research run '{research_run_id}' not found.")

    # 2. Fetch Agent 2 (Consumer Insights) & Agent 3 (Competitive Intelligence)
    consumer_insights = get_consumer_insights(research_run_id)
    competitive_intel = get_competitive_intelligence(research_run_id)

    has_ci = consumer_insights is not None and len(consumer_insights) > 0
    competitors = competitive_intel.get("competitors", []) if competitive_intel else []
    has_comp = bool(competitors)

    if not has_ci and not has_comp:
        raise ValueError(
            f"Both Agent 2 (Consumer Intelligence) and Agent 3 (Competitive Intelligence) outputs are missing "
            f"for run '{research_run_id}'. Both agents must be run before Agent 4 can execute."
        )
    if not has_ci:
        raise ValueError(
            f"Agent 2 output missing: No consumer insights found for run '{research_run_id}'. "
            f"Please run Agent 2 before running Agent 4."
        )
    if not has_comp:
        raise ValueError(
            f"Agent 3 output missing: No competitive intelligence found for run '{research_run_id}'. "
            f"Please run Agent 3 before running Agent 4."
        )

    logger.info(
        f"Inputs verified for run '{research_run_id}': "
        f"{len(consumer_insights)} consumer insights, {len(competitors)} competitors, "
        f"{len(competitive_intel.get('competitive_gaps', []))} competitive gaps."
    )

    # 3. Prepare Lookups for Grounding Validation
    ci_lookup = {ci["id"]: ci for ci in consumer_insights}
    valid_ci_ids = set(ci_lookup.keys())

    comp_lookup = {c["name"]: c for c in competitors}
    valid_comp_names_map = {c["name"].lower().strip(): c["name"] for c in competitors}

    comp_gaps = competitive_intel.get("competitive_gaps", [])
    comp_gap_lookup = {g["id"]: g for g in comp_gaps if isinstance(g, dict) and "id" in g}
    valid_cg_ids = set(comp_gap_lookup.keys())

    # 4. Initialize Gemini Client
    if client is None:
        from services.credential_manager import get_gemini_client
        from database import get_research_run_owner
        _owner_id = get_research_run_owner(research_run_id)
        client = get_gemini_client(user_id=_owner_id)

    models_to_try = GEMINI_FALLBACK_MODELS or [GEMINI_MODEL_NAME]
    feedback: Optional[str] = None
    final_validated_gaps: List[dict] = []

    # 5. ONE Gemini call (with retry if validation fails)
    for model_name in models_to_try:
        for attempt in range(1, max_retries + 2):
            prompt = build_creative_gaps_prompt(
                run_data=run_data,
                consumer_insights=consumer_insights,
                competitive_intel=competitive_intel,
                feedback=feedback
            )
            prompt = append_knowledge_guidance(
                prompt,
                "creative strategy brand positioning audience motivation and differentiation",
            )
            logger.info(f"Calling Gemini model '{model_name}' for Creative Gap Detection (Attempt {attempt})...")

            try:
                with track_api_call(
                    provider="gemini",
                    call_type="creative_gap_synthesis",
                    stage="Agent4_CreativeGaps",
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

                validated, errors = validate_creative_gaps(
                    data=parsed,
                    valid_ci_ids=valid_ci_ids,
                    ci_lookup=ci_lookup,
                    valid_comp_names_map=valid_comp_names_map,
                    comp_lookup=comp_lookup,
                    valid_cg_ids=valid_cg_ids,
                    comp_gap_lookup=comp_gap_lookup
                )

                if errors:
                    logger.warning(f"Agent 4 validation warnings on attempt {attempt}: {errors}")
                    if attempt <= max_retries:
                        feedback = (
                            "Your previous creative gaps contained grounding or cross-referencing errors:\n"
                            + "\n".join(f"- {e}" for e in errors)
                            + f"\nCRITICAL: Every gap must cite at least ONE valid consumer insight ID from: {sorted(list(valid_ci_ids))}."
                            + f"\nCRITICAL: Every gap must cite at least ONE valid competitor name from: {list(valid_comp_names_map.values())}."
                            + "\nCRITICAL: Do NOT fabricate ANY numbers or percentages. Every number must appear in cited text, or omit numbers entirely."
                        )
                        continue
                    else:
                        if validated:
                            logger.info(f"Proceeding with {len(validated)} validated creative gaps after retries.")
                            final_validated_gaps = validated
                            break
                        raise ValueError(f"Creative gap validation failed after retries: {errors}")
                else:
                    logger.info(f"Successfully synthesized and validated {len(validated)} creative gaps.")
                    final_validated_gaps = validated
                    break

            except Exception as exc:
                err_str = str(exc)
                logger.error(f"Error during creative gap synthesis attempt {attempt} with {model_name}: {err_str}")
                is_transient = any(k in err_str.lower() for k in [
                    "429", "resource_exhausted", "quota", "rate limit",
                    "503", "unavailable", "high demand", "overloaded", "server error", "500"
                ])
                if is_transient:
                    logger.warning(f"Transient error on '{model_name}' ({err_str[:100]}). Trying fallback...")
                    pacer.handle_429(model_name, exc, default_wait=2.0)
                    break
                if attempt > max_retries:
                    raise

        if final_validated_gaps:
            break

    if not final_validated_gaps:
        raise RuntimeError("Failed to synthesize creative gaps: no valid gaps generated.")

    # 6. Persist to PostgreSQL database (creative_gaps table)
    saved_records = save_creative_gaps(research_run_id, final_validated_gaps)
    logger.info(f"Successfully saved {len(saved_records)} creative gaps to database for run '{research_run_id}'.")
    return saved_records
