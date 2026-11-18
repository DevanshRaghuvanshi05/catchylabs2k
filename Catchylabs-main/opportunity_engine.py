"""
Opportunity Engine: Opportunity Scoring & Ranking Pipeline.
-----------------------------------------------------------
Scores Agent 4's creative_gaps using an explicit, objective 4-criteria rubric:
1. audience_demand (30%): Depth and urgency of consumer desire per Agent 2 insights.
2. competitive_whitespace (25%): Absence of effective competitor solutions per Agent 3 data.
3. brand_fit (25%): Credibility and asset alignment for the target brand from brief.
4. feasibility (20%): Practical achievability without unrealistic technological/economic hurdles.

Evaluates all gaps comparatively in ONE Gemini call per run for scoring consistency,
applies strict numerical grounding validation on reasonings, deterministically computes
weighted overall scores, ranks from highest to lowest, and persists to 'opportunity_scores'.
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
    get_creative_gaps,
    save_opportunity_scores,
    StaleInputError,
    track_api_call,
)
from evidence_extractor import clean_json_response
from gemini_pacer import pacer
from intelligence.knowledge import append_knowledge_guidance

logger = logging.getLogger("CatchyLabOpportunityEngine")

# Named Scoring Weight Constants (adjustable)
WEIGHT_AUDIENCE_DEMAND = 0.30
WEIGHT_COMPETITIVE_WHITESPACE = 0.25
WEIGHT_BRAND_FIT = 0.25
WEIGHT_FEASIBILITY = 0.20

# Regex pattern capturing statistical & quantified figures (same as Agents 2-4):
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


def compute_overall_score(
    audience_demand: int,
    competitive_whitespace: int,
    brand_fit: int,
    feasibility: int
) -> float:
    """
    Computes overall score as a weighted average using named constants:
    30% Audience Demand + 25% Competitive Whitespace + 25% Brand Fit + 20% Feasibility.
    """
    total = (
        audience_demand * WEIGHT_AUDIENCE_DEMAND
        + competitive_whitespace * WEIGHT_COMPETITIVE_WHITESPACE
        + brand_fit * WEIGHT_BRAND_FIT
        + feasibility * WEIGHT_FEASIBILITY
    )
    return round(total, 2)


def build_scoring_prompt(
    run_data: dict,
    creative_gaps: List[dict],
    consumer_insights: List[dict],
    competitive_intel: dict,
    feedback: Optional[str] = None
) -> str:
    """
    Constructs the prompt for Gemini to score all creative gaps comparatively
    across the 4 fixed criteria (1-5 each) with grounded reasoning.
    """
    brand = run_data.get("brand", "Unknown")
    market = run_data.get("market", "Unknown")
    audience = run_data.get("audience", "Unknown")
    objective = run_data.get("objective", "Unknown")

    # Format Consumer Insights reference lookup
    ci_map = {ci["id"]: ci for ci in consumer_insights}

    # Format Creative Gaps block
    gaps_lines = []
    valid_gap_ids = []
    for gap in creative_gaps:
        gid = gap["id"]
        valid_gap_ids.append(gid)
        title = gap.get("gap_title", "")
        desc = gap.get("description", "")
        ci_ids = gap.get("supporting_consumer_insight_ids", [])
        comps = gap.get("supporting_competitor_names", [])
        cg_ids = gap.get("supporting_competitive_gap_ids", [])
        conf = gap.get("confidence", "high")

        # Include summary of supporting insights for direct evaluator context
        ci_snippets = []
        for cid in ci_ids:
            if cid in ci_map:
                ci_snippets.append(f"  * Insight #{cid} ({ci_map[cid].get('insight_type')}): \"{ci_map[cid].get('insight_text')}\"")
        ci_text = "\n".join(ci_snippets) if ci_snippets else "  None"

        gaps_lines.append(
            f"=== CREATIVE GAP [ID: {gid}] ===\n"
            f"Title: {title}\n"
            f"Confidence: {conf}\n"
            f"Description: {desc}\n"
            f"Supporting Competitors: {', '.join(comps)}\n"
            f"Supporting Competitive Gap IDs: {cg_ids}\n"
            f"Supporting Consumer Insights:\n{ci_text}\n"
        )

    gaps_block = "\n".join(gaps_lines)
    valid_gap_ids_str = ", ".join(str(gid) for gid in valid_gap_ids)

    # Competitor context
    comp_list = competitive_intel.get("competitors", [])
    comp_lines = [
        f"- {c.get('name')}: {c.get('positioning')} (Strengths: {', '.join(c.get('strengths', []))}; Weaknesses: {', '.join(c.get('weaknesses', []))})"
        for c in comp_list
    ]
    comp_block = "\n".join(comp_lines) if comp_lines else "None"

    prompt = (
        "You are an elite Brand Strategy & Opportunity Evaluation Director.\n"
        "Your task is to judge and score each Creative Gap identified for the target brand using a rigorous, "
        "fixed 4-criteria rubric. You must evaluate ALL gaps together so that your scores are comparative, "
        "consistent, and calibrated across the entire opportunity set.\n\n"
        "### RESEARCH BRIEF CONTEXT:\n"
        f"- Target Brand: {brand}\n"
        f"- Target Market: {market}\n"
        f"- Target Audience: {audience}\n"
        f"- Core Strategic Objective: {objective}\n\n"
        "### COMPETITIVE LANDSCAPE SUMMARY:\n"
        f"{comp_block}\n\n"
        "### CREATIVE GAPS TO EVALUATE:\n"
        f"{gaps_block}\n\n"
        "### FIXED SCORING RUBRIC (Integer 1-5 for EACH criterion):\n"
        "1. 'audience_demand' (Weight 30%):\n"
        "   - How deeply and urgently does the target audience desire or experience this tension?\n"
        "   - 1 = Niche or weak consumer interest; 3 = Moderate interest; 5 = Urgent, visceral, widely shared tension with strong emotional/lifestyle pull.\n"
        "2. 'competitive_whitespace' (Weight 25%):\n"
        "   - How genuinely unaddressed or mishandled is this space by competitors in the market?\n"
        "   - 1 = Competitors already serve this well; 3 = Partially addressed or contested; 5 = Total competitive vacuum; current players either fail, ignore, or cannot credibly fulfill this.\n"
        "3. 'brand_fit' (Weight 25%):\n"
        "   - Does the target brand ({brand}) have the authentic brand equity, market permission, and assets to credibly lead this opportunity?\n"
        "   - 1 = Off-brand, contradicts brand heritage; 3 = Plausible stretch; 5 = Natural, powerful fit that amplifies the brand's core identity and market authority.\n"
        "4. 'feasibility' (Weight 20%):\n"
        "   - Is this opportunity realistically achievable in terms of execution, economics, channels, and operational reality?\n"
        "   - 1 = Prohibitive cost/complexity or severe operational barriers; 3 = Achievable with significant investment; 5 = Readily executable with existing or accessible capabilities.\n\n"
        "### CRITICAL CONSTRAINTS:\n"
        f"1. You MUST evaluate EVERY gap listed: IDs [{valid_gap_ids_str}]. Do NOT omit any gap or hallucinate new IDs.\n"
        "2. Every score MUST be an integer from 1 to 5 (inclusive). Decimals or ratings outside 1-5 will be rejected.\n"
        "3. Provide concise, grounded reasoning (2 to 4 sentences) for each of the 4 criteria per gap.\n"
        "4. STRICT NUMERICAL GROUNDING: Do NOT invent or estimate any metrics, percentages, or numbers in your reasonings. "
        "Every number mentioned in your reasoning must appear verbatim in the gap's description or cited evidence text. "
        "If no exact number appears in the source, keep your reasoning purely qualitative.\n\n"
        "### REQUIRED JSON OUTPUT SCHEMA:\n"
        "Return ONLY a valid JSON object matching this exact schema:\n"
        "{\n"
        '  "opportunity_scores": [\n'
        "    {\n"
        f'      "creative_gap_id": <int from {valid_gap_ids}>,\n'
        '      "audience_demand_score": <1-5 int>,\n'
        '      "audience_demand_reasoning": "Reasoning text grounded in consumer insights...",\n'
        '      "competitive_whitespace_score": <1-5 int>,\n'
        '      "competitive_whitespace_reasoning": "Reasoning text grounded in competitor data...",\n'
        '      "brand_fit_score": <1-5 int>,\n'
        '      "brand_fit_reasoning": "Reasoning text assessing brand credibility and alignment...",\n'
        '      "feasibility_score": <1-5 int>,\n'
        '      "feasibility_reasoning": "Reasoning text assessing operational and market execution..."\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )

    if feedback:
        prompt += f"\n\n### RETRY INSTRUCTIONS (FIX PREVIOUS ERRORS):\n{feedback}\n"

    return prompt


def validate_opportunity_scores(
    data: dict,
    valid_gap_ids: Set[int],
    gap_lookup: Dict[int, dict],
    ci_lookup: Dict[int, dict],
    comp_lookup: Dict[str, dict]
) -> Tuple[List[dict], List[str]]:
    """
    Validates LLM-generated opportunity scores:
    1. Schema compliance (all 4 scores present and are integers 1-5)
    2. Valid creative_gap_ids (must match this run's gaps, no hallucinated IDs)
    3. Completeness (all gaps scored)
    4. Numerical grounding check across all reasoning texts against the gap's source data
    """
    errors: List[str] = []
    validated: List[dict] = []

    if not isinstance(data, dict):
        return [], ["Response is not a JSON object."]

    raw_scores = data.get("opportunity_scores")
    if not isinstance(raw_scores, list) or len(raw_scores) == 0:
        return [], ["Output must contain a non-empty list of 'opportunity_scores'."]

    seen_gap_ids = set()

    for idx, item in enumerate(raw_scores, 1):
        if not isinstance(item, dict):
            errors.append(f"Opportunity score #{idx} is not a JSON object.")
            continue

        raw_gid = item.get("creative_gap_id")
        try:
            gid = int(raw_gid)
        except (ValueError, TypeError):
            errors.append(f"Opportunity score #{idx} has invalid creative_gap_id '{raw_gid}'. Must be an integer.")
            continue

        if gid not in valid_gap_ids:
            errors.append(
                f"Opportunity score #{idx} references hallucinated creative_gap_id {gid}. "
                f"Valid gap IDs for this run: {sorted(list(valid_gap_ids))}."
            )
            continue

        if gid in seen_gap_ids:
            errors.append(f"Duplicate scoring entry for creative_gap_id {gid}.")
            continue
        seen_gap_ids.add(gid)

        gap_record = gap_lookup[gid]

        # Validate the 4 score criteria
        score_fields = [
            ("audience_demand_score", "audience_demand_reasoning"),
            ("competitive_whitespace_score", "competitive_whitespace_reasoning"),
            ("brand_fit_score", "brand_fit_reasoning"),
            ("feasibility_score", "feasibility_reasoning"),
        ]

        scores_dict = {}
        reasonings_dict = {}
        gap_has_err = False

        for s_field, r_field in score_fields:
            s_val = item.get(s_field)
            r_val = str(item.get(r_field, "")).strip()

            try:
                int_val = int(s_val)
                if int_val < 1 or int_val > 5:
                    errors.append(f"Gap #{gid} '{s_field}' value {s_val} is out of bounds (must be integer 1-5).")
                    gap_has_err = True
                else:
                    scores_dict[s_field] = int_val
            except (ValueError, TypeError):
                errors.append(f"Gap #{gid} '{s_field}' is not an integer: '{s_val}'.")
                gap_has_err = True

            if len(r_val) < 15:
                errors.append(f"Gap #{gid} '{r_field}' is too short or missing (min 15 characters).")
                gap_has_err = True
            else:
                reasonings_dict[r_field] = r_val

        if gap_has_err:
            continue

        # Numerical Grounding Check:
        # Build reference text: gap title + description + supporting CI text + supporting comp text
        ref_text_parts = [
            gap_record.get("gap_title", ""),
            gap_record.get("description", "")
        ]
        for cid in gap_record.get("supporting_consumer_insight_ids", []):
            if cid in ci_lookup:
                ref_text_parts.append(ci_lookup[cid].get("insight_text", ""))

        for cname in gap_record.get("supporting_competitor_names", []):
            c_data = comp_lookup.get(cname.lower())
            if c_data:
                ref_text_parts.append(c_data.get("name", ""))
                ref_text_parts.append(c_data.get("positioning", ""))
                ref_text_parts.extend(c_data.get("strengths", []))
                ref_text_parts.extend(c_data.get("weaknesses", []))

        combined_ref_text = " ".join(ref_text_parts)

        # Extract and verify statistics from all 4 reasonings
        all_reasoning_text = " ".join(reasonings_dict.values())
        figures = extract_statistical_figures(all_reasoning_text)
        has_num_err = False

        for fig in figures:
            core = fig["core_num"]
            raw_fig = fig["raw"]
            num_pattern = r"(?<!\d)" + re.escape(core) + r"(?!\d)"
            if not re.search(num_pattern, combined_ref_text):
                errors.append(
                    f"Gap #{gid} reasoning contains ungrounded/hallucinated statistic '{raw_fig}' "
                    f"(number: {core}) not found in the gap's description or underlying cited evidence."
                )
                has_num_err = True

        if has_num_err:
            continue

        # Compute deterministic weighted overall score in Python
        overall = compute_overall_score(
            audience_demand=scores_dict["audience_demand_score"],
            competitive_whitespace=scores_dict["competitive_whitespace_score"],
            brand_fit=scores_dict["brand_fit_score"],
            feasibility=scores_dict["feasibility_score"]
        )

        validated.append({
            "creative_gap_id": gid,
            "overall_score": overall,
            "audience_demand_score": scores_dict["audience_demand_score"],
            "audience_demand_reasoning": reasonings_dict["audience_demand_reasoning"],
            "competitive_whitespace_score": scores_dict["competitive_whitespace_score"],
            "competitive_whitespace_reasoning": reasonings_dict["competitive_whitespace_reasoning"],
            "brand_fit_score": scores_dict["brand_fit_score"],
            "brand_fit_reasoning": reasonings_dict["brand_fit_reasoning"],
            "feasibility_score": scores_dict["feasibility_score"],
            "feasibility_reasoning": reasonings_dict["feasibility_reasoning"]
        })

    # Ensure all gaps were scored
    missing_gaps = valid_gap_ids - seen_gap_ids
    if missing_gaps:
        errors.append(f"Gemini output missed scoring creative_gap_ids: {sorted(list(missing_gaps))}.")

    return validated, errors


def run_opportunity_engine_pipeline(
    research_run_id: str,
    client: Optional[genai.Client] = None,
    max_retries: int = 2
) -> List[Dict[str, Any]]:
    """
    Executes the Opportunity Engine scoring and ranking pipeline:
    1. Fetches creative gaps (Agent 4) and brief.
       FAILS CLEARLY with an informative error if Agent 4 hasn't run yet.
    2. Makes ONE Gemini call per run passing all gaps together for comparative scoring.
    3. Validates scores (1-5 integer check), gap ID integrity, and numerical grounding on reasonings.
    4. Computes deterministic overall_score via weighted average.
    5. Sorts by overall_score descending and assigns ranks (1, 2, ...).
    6. Persists to 'opportunity_scores' table in PostgreSQL.
    7. Uses ZERO SerpAPI calls.
    Returns ranked opportunity list.
    """
    logger.info(f"=== Starting Opportunity Engine Pipeline for Run: {research_run_id} ===")

    # 1. Fetch Run Metadata
    run_data = get_run_with_evidence(research_run_id)
    if not run_data:
        logger.error(f"Research run '{research_run_id}' not found in database.")
        raise ValueError(f"Research run '{research_run_id}' not found.")

    # 2. Fetch Agent 4 Output (Creative Gaps) - only non-superseded
    creative_gaps = get_creative_gaps(research_run_id, include_superseded=False)
    if not creative_gaps:
        logger.error(f"No creative gaps found for run '{research_run_id}'.")
        raise ValueError(
            f"Agent 4 output missing: No creative gaps found for run '{research_run_id}'. "
            "Please run Agent 4 before executing the Opportunity Engine."
        )

    # 2b. Refuse to build on stale inputs — any gap citing superseded insights
    stale_gaps = [g for g in creative_gaps if g.get("is_stale")]
    if stale_gaps:
        stale_ids = []
        for g in stale_gaps:
            stale_ids.extend(g.get("stale_insight_ids", []))
        raise StaleInputError(
            entity_type="CreativeGap",
            entity_id=[g["id"] for g in stale_gaps],
            stale_insight_ids=sorted(set(stale_ids))
        )

    # 3. Fetch Supporting Synthesized Context (Agent 2 & Agent 3)
    consumer_insights = get_consumer_insights(research_run_id) or []
    competitive_intel = get_competitive_intelligence(research_run_id) or {}

    logger.info(
        f"Loaded {len(creative_gaps)} creative gaps to score for run '{research_run_id}' "
        f"({len(consumer_insights)} consumer insights, {len(competitive_intel.get('competitors', []))} competitors)."
    )

    # Build reference lookup maps
    valid_gap_ids = {g["id"] for g in creative_gaps}
    gap_lookup = {g["id"]: g for g in creative_gaps}
    ci_lookup = {ci["id"]: ci for ci in consumer_insights}
    comp_lookup = {
        c["name"].lower().strip(): c
        for c in competitive_intel.get("competitors", [])
    }

    # 4. Initialize Gemini Client
    if client is None:
        from services.credential_manager import get_gemini_client
        from database import get_research_run_owner
        _owner_id = get_research_run_owner(research_run_id)
        client = get_gemini_client(user_id=_owner_id)

    models_to_try = GEMINI_FALLBACK_MODELS or [GEMINI_MODEL_NAME]
    feedback: Optional[str] = None
    final_validated_scores: List[dict] = []

    # 5. ONE Gemini Call per run for comparative scoring
    for model_name in models_to_try:
        for attempt in range(1, max_retries + 2):
            prompt = build_scoring_prompt(
                run_data=run_data,
                creative_gaps=creative_gaps,
                consumer_insights=consumer_insights,
                competitive_intel=competitive_intel,
                feedback=feedback
            )
            prompt = append_knowledge_guidance(
                prompt,
                "marketing analytics opportunity evaluation demand brand fit feasibility experiments",
            )
            logger.info(f"Calling Gemini model '{model_name}' for Opportunity Scoring (Attempt {attempt})...")

            try:
                with track_api_call(
                    provider="gemini",
                    call_type="opportunity_scoring",
                    stage="OpportunityEngine_Scoring",
                    model=model_name,
                    research_run_id=research_run_id,
                    input_text=prompt[:400]
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

                raw_text = response.text if hasattr(response, "text") and response.text else ""
                cleaned = clean_json_response(raw_text)
                parsed = json.loads(cleaned)

                validated, errors = validate_opportunity_scores(
                    data=parsed,
                    valid_gap_ids=valid_gap_ids,
                    gap_lookup=gap_lookup,
                    ci_lookup=ci_lookup,
                    comp_lookup=comp_lookup
                )

                if errors:
                    logger.warning(f"Opportunity scoring validation warnings on attempt {attempt}: {errors}")
                    if attempt <= max_retries:
                        feedback = (
                            "Your previous opportunity scoring output contained errors:\n"
                            + "\n".join(f"- {e}" for e in errors)
                            + f"\nCRITICAL: You must score every creative_gap_id from: {sorted(list(valid_gap_ids))}."
                            + "\nCRITICAL: Every score must be an INTEGER 1 to 5."
                            + "\nCRITICAL: Do NOT invent any numbers in the reasonings. Keep reasonings qualitative or grounded in cited text."
                        )
                        continue
                    else:
                        if validated and len(validated) == len(creative_gaps):
                            final_validated_scores = validated
                            break
                        raise ValueError(f"Opportunity scoring validation failed after retries: {errors}")
                else:
                    logger.info(f"Successfully scored and validated {len(validated)} creative gaps.")
                    final_validated_scores = validated
                    break

            except Exception as exc:
                err_str = str(exc)
                logger.error(f"Error during opportunity scoring attempt {attempt} with {model_name}: {err_str}")
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

        if final_validated_scores:
            break

    if not final_validated_scores:
        raise RuntimeError("Failed to score opportunities: no valid scores generated.")

    # 6. Sort by overall_score descending & assign ranks (1, 2, 3...)
    # Tie-breaking: audience_demand_score desc, then brand_fit_score desc
    final_validated_scores.sort(
        key=lambda x: (
            x["overall_score"],
            x["audience_demand_score"],
            x["brand_fit_score"]
        ),
        reverse=True
    )

    for rank_idx, item in enumerate(final_validated_scores, 1):
        item["rank"] = rank_idx

    # 7. Persist to PostgreSQL database (opportunity_scores table)
    saved_records = save_opportunity_scores(research_run_id, final_validated_scores)
    logger.info(
        f"Successfully persisted {len(saved_records)} ranked opportunity scores "
        f"for run '{research_run_id}'."
    )
    return saved_records
