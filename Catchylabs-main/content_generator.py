"""
Content Generator: Draft Social Content Pipeline.
---------------------------------------------------
Takes the top-ranked opportunity (or a specific creative_gap_id) from a completed
Opportunity Engine run and generates draft social content grounded in the research.

Generates in ONE Gemini call:
- 2 Reel Scripts (15-30 seconds, hook in first 3 seconds)
- 1 Carousel (5-7 slides)
- 1 Static Post

GROUNDING RULES:
- Statistics must appear verbatim in cited evidence or insight text.
- Never assert specific product facts (materials, recycled %, certifications,
  prices, performance) unless they appear in cited evidence. Otherwise insert
  placeholders like [VERIFIED CLAIM NEEDED: material source].
- No negative statements or direct comparisons about named competitors.

ZERO SerpAPI calls.
"""

import json
import logging
import random
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
    get_opportunity_scores,
    save_content_pieces,
    track_api_call,
    StaleInputError,
)
from evidence_extractor import clean_json_response
from gemini_pacer import pacer
from intelligence.knowledge import append_knowledge_guidance
from intelligence.language import output_language_instruction, resolve_language

logger = logging.getLogger("CatchyLabContentGenerator")

# Reuse the same stats regex as opportunity_engine.py for grounding validation
STATS_PATTERN = re.compile(
    r"(?:"
    r"\b\d+(?:\.\d+)?\s*%"
    r"|\b\d+(?:\.\d+)?\s*percent\b"
    r"|CAGR\s*(?:of\s*)?\d+(?:\.\d+)?%?"
    r"|\b\d+(?:\.\d+)?%\s*CAGR\b"
    r"|(?:[\$€£]|(?:USD|INR|EUR|GBP)\s*)\d+(?:\.\d+)?(?:\s*(?:billion|million|trillion|crore|lakh|b|m|k))?\b"
    r"|\b\d+(?:\.\d+)?\s*(?:billion|million|trillion|crore|lakh)\b"
    r"|\b\d+(?:\.\d+)?\s*x\b"
    r"|\b\d+\s*(?:out of|\/)\s*\d+\b"
    r"|\b\d+\.\d+\b"
    r")",
    re.IGNORECASE
)

PLACEHOLDER_PATTERN = re.compile(r"\[VERIFIED CLAIM NEEDED:[^\]]*\]", re.IGNORECASE)


def extract_statistical_figures(text: str) -> List[Dict[str, str]]:
    """Extracts all statistical figures from text using the shared regex pattern."""
    results = []
    for match in STATS_PATTERN.finditer(text):
        raw = match.group(0).strip()
        num_m = re.search(r"\d+(?:\.\d+)?", raw)
        core_num = num_m.group(0) if num_m else raw
        results.append({"raw": raw, "core_num": core_num})
    return results


def parse_llm_json(raw_text: str) -> dict:
    """Parse Gemini JSON, repairing common trailing-comma issues."""
    cleaned = clean_json_response(raw_text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        repaired = re.sub(r",\s*([}\]])", r"\1", cleaned)
        parsed = json.loads(repaired)
    if not isinstance(parsed, dict):
        raise ValueError("LLM JSON root must be an object.")
    return parsed


def _normalize_body(body: Any, content_type: str) -> Any:
    """Coerce common LLM body shapes into the expected list/object structure."""
    if isinstance(body, str):
        stripped = body.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                body = json.loads(stripped)
            except json.JSONDecodeError:
                if content_type == "static_post":
                    return {"visual_direction": body}
                return body
        elif content_type == "static_post":
            return {"visual_direction": body}

    if content_type == "reel_script" and isinstance(body, dict):
        for key in ("scenes", "scene_list", "items"):
            if isinstance(body.get(key), list):
                return body[key]
    if content_type == "carousel" and isinstance(body, dict):
        for key in ("slides", "slide_list", "items"):
            if isinstance(body.get(key), list):
                return body[key]
    if content_type == "static_post" and isinstance(body, dict) and "visual_direction" not in body:
        if body.get("description"):
            return {"visual_direction": body["description"]}
    return body


def _coerce_int_ids(raw) -> List:
    """Coerces LLM-provided ID lists to ints where possible."""
    if not isinstance(raw, list):
        return raw
    coerced = []
    for item in raw:
        try:
            coerced.append(int(item))
        except (TypeError, ValueError):
            coerced.append(item)
    return coerced


def _piece_text_blob(piece: dict) -> str:
    """Concatenates all human-readable text fields on a content piece."""
    parts = [
        str(piece.get("hook", "")),
        str(piece.get("caption", "")),
        str(piece.get("title", "")),
        str(piece.get("call_to_action", "")),
        " ".join(str(t) for t in (piece.get("hashtags") or [])),
    ]
    body = piece.get("body")
    if isinstance(body, list):
        for scene in body:
            if isinstance(scene, dict):
                for k in ("visual", "voiceover", "on_screen_text", "body_text", "headline", "visual_direction"):
                    parts.append(str(scene.get(k, "")))
    elif isinstance(body, dict):
        parts.append(str(body.get("visual_direction", "")))
    return " ".join(parts)


def _competitor_attack_errors(text: str, competitor_names: Set[str], piece_idx: int) -> List[str]:
    """Rejects negative statements or direct comparisons about named competitors."""
    errors = []
    for name in competitor_names:
        if not name:
            continue
        escaped = re.escape(name)
        patterns = [
            rf"\bunlike\s+{escaped}\b",
            rf"\bcompared\s+to\s+{escaped}\b",
            rf"\bvs\.?\s+{escaped}\b",
            rf"\bversus\s+{escaped}\b",
            rf"\bbetter than\s+{escaped}\b",
            rf"\bworse than\s+{escaped}\b",
            rf"\bbeats?\s+{escaped}\b",
            rf"{escaped}\s+(?:fails|failed|sucks|can't|cannot|doesn't|does not|is worse|is bad)",
        ]
        for pat in patterns:
            if re.search(pat, text, re.IGNORECASE):
                errors.append(
                    f"Piece #{piece_idx}: Negative statement or direct comparison about named competitor '{name}'."
                )
                break
    return errors


def _brand_claim_errors(text: str, piece_idx: int, brand_name: str, cited_evidence: str = "") -> Tuple[List[str], List[str]]:
    """
    Detects unverified brand product claims about price, affordability, quality, durability,
    comfort, materials, or environmental benefits. Returns (errors, claims_to_verify).
    
    ONLY flags claim terms in sentences that refer to the brand or its product:
    - Brand name (e.g., "Puma")
    - "we", "our", "this", "these"
    - "the shoe", "the sneaker", "this pair", "these shoes", "our shoes"
    
    Audience-problem language (price sensitivity, budget worries, wanting sustainable options)
    is allowed to pass. Claims inside [VERIFIED CLAIM NEEDED: ...] placeholders are allowed.
    Multi-word phrases from cited evidence are allowed (whole phrase match only, not single words).
    
    Brand-claim hits no longer cause rejection - they are saved to claims_to_verify for manual review.
    """
    # Remove placeholders before checking for claim terms
    text_without_placeholders = PLACEHOLDER_PATTERN.sub(" ", text)
    
    # Claim terms that require verification (lowercase for matching)
    # Order matters: longer terms first to avoid partial matches
    claim_terms = [
        "premium-free", "real prices", "leading performance", "premium performance",
        "high quality", "environmental benefit", "planet-friendly",
        "recycled materials", "recycled material", "recycled",
        "affordable", "cheap", "budget", "fair", "premium",
        "durable", "durability", "comfort", "comfortable", "quality",
        "low impact", "eco-friendly", "sustainable", "performance",
        "wallet", "price", "prices", "cost", "costs", "afford", "markup"
    ]
    
    # Brand/product reference patterns
    brand_lower = brand_name.lower()
    brand_patterns = [
        rf"\b{re.escape(brand_lower)}\b",
        r"\bwe\b",
        r"\bour\b",
        r"\bthis\b",
        r"\bthese\b",
        r"\bthe shoe\b",
        r"\bthe sneaker\b",
        r"\bthis pair\b",
        r"\bthese shoes\b",
        r"\bour shoes\b",
        r"\bour sneaker\b"
    ]
    
    errors = []
    claims_to_verify = []
    
    # Extract multi-word phrases from cited evidence (2+ words)
    cited_phrases = set()
    if cited_evidence:
        # Extract phrases of 2+ words from cited evidence
        words = cited_evidence.split()
        for i in range(len(words) - 1):
            for j in range(i + 2, min(i + 5, len(words) + 1)):  # 2-4 word phrases
                phrase = " ".join(words[i:j])
                cited_phrases.add(phrase.lower())
                # Also extract exact matches for brand name + term patterns
                if brand_lower in phrase.lower():
                    cited_phrases.add(phrase.lower())
    
    # Split text into sentences
    sentences = re.split(r'[.!?]', text)

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Check if sentence refers to brand/product
        refers_to_brand = any(re.search(pattern, sentence.lower()) for pattern in brand_patterns)

        # Additional check: treat imperative CTA-style lines with claim terms as brand-voice
        # (e.g., "Upgrade your run, protect your wallet" or "High Quality. Eco-conscious. Real Prices.")
        # Exception: Questions about audience's problem should pass
        has_claim_term = any(re.search(rf"\b{re.escape(term)}\b", sentence.lower()) for term in claim_terms)
        is_question = "?" in sentence
        # Detect imperative/imperative-style short phrases (no subject, starts with verb or adjective)
        words = sentence.split()
        is_imperative = len(words) <= 8 and (words[0].lower() in ["check", "upgrade", "buy", "shop", "get", "high", "real", "affordable", "premium", "low"])

        if not refers_to_brand and has_claim_term and not is_question and is_imperative:
            # Treat as brand-voice claim (likely hook/on_screen_text/CTA)
            refers_to_brand = True

        if not refers_to_brand:
            # Audience-side language - skip claim validation
            continue
        
        # Sentence refers to brand - check for claim terms
        for term in claim_terms:
            pattern = rf"\b{re.escape(term)}\b"
            if re.search(pattern, sentence.lower()):
                # Check if term is inside a placeholder
                full_text_with_sentence = text
                placeholder_pattern = rf"\[VERIFIED CLAIM NEEDED:[^\]]*{re.escape(term)}[^\]]*\]"
                if re.search(placeholder_pattern, full_text_with_sentence, re.IGNORECASE):
                    # Extract claims that ARE in placeholders
                    matches = re.findall(placeholder_pattern, full_text_with_sentence, re.IGNORECASE)
                    claims_to_verify.extend(matches)
                elif cited_phrases and any(phrase in sentence.lower() for phrase in cited_phrases):
                    # Multi-word phrase from cited evidence - allow it
                    continue
                else:
                    # Brand claim hit - add to claims_to_verify with the offending sentence
                    claims_to_verify.append(f"[VERIFIED CLAIM NEEDED: {term} in '{sentence[:100]}...']")
    
    return errors, claims_to_verify


def build_content_prompt(
    run_data: dict,
    opportunity: dict,
    creative_gap: dict,
    consumer_insights: List[dict],
    competitor_entries: List[dict],
    platform: str = "instagram",
    feedback: Optional[str] = None
) -> str:
    """
    Constructs the prompt for Gemini to generate 4 content pieces
    (2 reels, 1 carousel, 1 static post) for the selected opportunity.
    """
    brand = run_data.get("brand", "Unknown")
    market = run_data.get("market", "Unknown")
    audience = run_data.get("audience", "Unknown")
    objective = run_data.get("objective", "Unknown")
    language = resolve_language(
        f"{objective} {audience}",
        str(run_data.get("language_preference") or "auto"),
    )

    # Format supporting consumer insights
    ci_lines = []
    ci_ids = creative_gap.get("supporting_consumer_insight_ids", [])
    ci_map = {ci["id"]: ci for ci in consumer_insights}
    for cid in ci_ids:
        if cid in ci_map:
            ci = ci_map[cid]
            ci_lines.append(
                f"  * Insight #{cid} ({ci.get('insight_type')}): \"{ci.get('insight_text')}\""
            )
    ci_block = "\n".join(ci_lines) if ci_lines else "  None"

    # Format competitor context
    comp_names = creative_gap.get("supporting_competitor_names", [])
    comp_lines = []
    for comp in competitor_entries:
        if comp.get("name") in comp_names:
            comp_lines.append(
                f"  * {comp.get('name')}: {comp.get('positioning', '')} "
                f"(Strengths: {', '.join(comp.get('strengths', []))}; "
                f"Weaknesses: {', '.join(comp.get('weaknesses', []))})"
            )
    comp_block = "\n".join(comp_lines) if comp_lines else "  None"

    # Valid IDs for reference
    valid_ci_ids = [cid for cid in ci_ids if cid in ci_map]
    valid_comp_names = [n for n in comp_names]

    prompt = (
        "You are an elite Social Media Creative Director specializing in research-backed content.\n\n"
        "### YOUR TASK:\n"
        f"Generate draft social content for {platform.upper()} based on the top-ranked creative opportunity below.\n\n"
        "### RESEARCH BRIEF CONTEXT:\n"
        f"- Target Brand: {brand}\n"
        f"- Target Market: {market}\n"
        f"- Target Audience: {audience}\n"
        f"- Core Strategic Objective: {objective}\n\n"
        f"### OUTPUT LANGUAGE ({language['language']}, {language['script']} script):\n"
        f"{output_language_instruction(language['language'])}\n"
        "- Preserve source titles, cited excerpts, brand names, product names, and technical terms in their original form.\n\n"
        "### SELECTED OPPORTUNITY:\n"
        f"- Creative Gap Title: {creative_gap.get('gap_title', '')}\n"
        f"- Description: {creative_gap.get('description', '')}\n"
        f"- Opportunity Rank: #{opportunity.get('rank', 1)}\n"
        f"- Overall Score: {opportunity.get('overall_score', 'N/A')}\n\n"
        "### SUPPORTING CONSUMER INSIGHTS:\n"
        f"{ci_block}\n\n"
        "### SUPPORTING COMPETITOR CONTEXT:\n"
        f"{comp_block}\n\n"
        "### CONTENT REQUIREMENTS:\n"
        "Generate EXACTLY 4 content pieces:\n\n"
        "1. **Reel Script 1** (content_type: 'reel_script'):\n"
        "   - Total duration: 15-30 seconds\n"
        "   - The HOOK must appear in the FIRST scene and last no more than 3 seconds\n"
        "   - Body: ordered list of scenes, each with: visual, voiceover, on_screen_text, duration_seconds\n"
        "   - Scene durations must add up to between 15 and 30 seconds total\n"
        "   - Voiceover word budget: maximum 3 words per second (e.g., a 3-second scene can have ~9 words)\n\n"
        "2. **Reel Script 2** (content_type: 'reel_script'):\n"
        "   - Same format as Reel 1 but with a DIFFERENT creative angle/hook\n"
        "   - Total duration: 15-30 seconds, hook in first 3 seconds\n"
        "   - Voiceover word budget: maximum 3 words per second (e.g., a 3-second scene can have ~9 words)\n\n"
        "3. **Carousel** (content_type: 'carousel'):\n"
        "   - 5 to 7 slides\n"
        "   - Body: ordered list of slides, each with: slide_number, headline, body_text, visual_direction\n"
        "   - First slide must contain the hook\n\n"
        "4. **Static Post** (content_type: 'static_post'):\n"
        "   - Body: object with visual_direction (detailed description for a designer)\n"
        "   - Single image concept\n\n"
        "### TONE & STYLE:\n"
        f"- Tone should authentically resonate with {audience} in {market}\n"
        "- Be culturally relevant and platform-native\n"
        "- Avoid corporate jargon; be conversational and real\n\n"
        "### CRITICAL GROUNDING RULES:\n"
        "1. **Statistical Grounding**: Any statistic, percentage, or number you use in the content MUST appear "
        "verbatim in the consumer insights or competitor context quoted above. Do NOT invent numbers.\n"
        "2. **Product Claim Safety**: NEVER assert specific product facts about the brand (materials, recycled "
        "percentages, certifications, prices, performance specs) unless they appear in the cited evidence above. "
        "Instead, insert a placeholder like:\n"
        "   [VERIFIED CLAIM NEEDED: <what needs verification>]\n"
        "   For example: [VERIFIED CLAIM NEEDED: percentage of recycled materials used]\n"
        "3. **No Competitor Attacks**: Do NOT make negative statements about or direct comparisons with named competitors.\n"
        "4. **Brand Claims Restriction**: Any statement about the brand's own products (price, affordability, quality, "
        "durability, comfort, materials, eco or environmental benefit) MUST use [VERIFIED CLAIM NEEDED: ...] "
        "placeholder unless it appears verbatim in the cited evidence. Content may describe the audience's problem, "
        "but it must NOT claim the brand already solves it without evidence.\n\n"
        "### ATTRIBUTION:\n"
        f"- Each piece MUST declare 'supporting_consumer_insight_ids' from: {valid_ci_ids}\n"
        f"- Each piece MUST declare 'supporting_competitor_names' from: {valid_comp_names}\n"
        "- Only cite IDs and names that actually informed that specific piece of content.\n\n"
        "### REQUIRED JSON OUTPUT:\n"
        "Return ONLY a valid JSON object. Do not include trailing commas or markdown.\n"
        "body MUST be a JSON array for reel_script (scenes) and carousel (slides), "
        "and a JSON object for static_post.\n"
        "Example:\n"
        "{\n"
        '  "content_pieces": [\n'
        "    {\n"
        '      "content_type": "reel_script",\n'
        f'      "platform": "{platform}",\n'
        '      "title": "Example reel title",\n'
        '      "hook": "Hook line here",\n'
        '      "body": [\n'
        '        {"visual": "...", "voiceover": "Hook line here", "on_screen_text": "Hook line here", "duration_seconds": 2},\n'
        '        {"visual": "...", "voiceover": "...", "on_screen_text": "...", "duration_seconds": 8},\n'
        '        {"visual": "...", "voiceover": "...", "on_screen_text": "...", "duration_seconds": 8}\n'
        "      ],\n"
        '      "caption": "...",\n'
        '      "hashtags": ["example"],\n'
        '      "call_to_action": "...",\n'
        '      "supporting_consumer_insight_ids": ' + json.dumps(valid_ci_ids[:1] or [0]) + ",\n"
        '      "supporting_competitor_names": ' + json.dumps(valid_comp_names[:1]) + "\n"
        "    },\n"
        "    {\n"
        '      "content_type": "carousel",\n'
        f'      "platform": "{platform}",\n'
        '      "title": "Example carousel",\n'
        '      "hook": "Hook on slide 1",\n'
        '      "body": [\n'
        '        {"slide_number": 1, "headline": "Hook on slide 1", "body_text": "...", "visual_direction": "..."},\n'
        '        {"slide_number": 2, "headline": "...", "body_text": "...", "visual_direction": "..."},\n'
        '        {"slide_number": 3, "headline": "...", "body_text": "...", "visual_direction": "..."},\n'
        '        {"slide_number": 4, "headline": "...", "body_text": "...", "visual_direction": "..."},\n'
        '        {"slide_number": 5, "headline": "...", "body_text": "...", "visual_direction": "..."}\n'
        "      ],\n"
        '      "caption": "...",\n'
        '      "hashtags": ["example"],\n'
        '      "call_to_action": "...",\n'
        '      "supporting_consumer_insight_ids": ' + json.dumps(valid_ci_ids[:1] or [0]) + ",\n"
        '      "supporting_competitor_names": ' + json.dumps(valid_comp_names[:1]) + "\n"
        "    },\n"
        "    {\n"
        '      "content_type": "static_post",\n'
        f'      "platform": "{platform}",\n'
        '      "title": "Example static",\n'
        '      "hook": "Hook line",\n'
        '      "body": {"visual_direction": "Detailed art direction..."},\n'
        '      "caption": "...",\n'
        '      "hashtags": ["example"],\n'
        '      "call_to_action": "...",\n'
        '      "supporting_consumer_insight_ids": ' + json.dumps(valid_ci_ids[:1] or [0]) + ",\n"
        '      "supporting_competitor_names": ' + json.dumps(valid_comp_names[:1]) + "\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "Generate 4 pieces: reel_script, reel_script, carousel, static_post.\n"
    )

    if feedback:
        prompt += f"\n\n### RETRY INSTRUCTIONS (FIX PREVIOUS ERRORS):\n{feedback}\n"

    return prompt


def validate_content_pieces(
    data: dict,
    valid_ci_ids: Set[int],
    valid_comp_names: Set[str],
    insight_texts: Dict[int, str],
    competitor_texts: Dict[str, str],
    platform: str = "instagram",
    brand_name: str = "Brand"
) -> Tuple[List[dict], List[str]]:
    """
    Validates LLM-generated content pieces:
    1. Schema compliance (required fields present)
    2. Correct count (2 reels, 1 carousel, 1 static post)
    3. Hook presence on every piece
    4. Reel scene durations sum to 15-30 seconds
    5. Hook in first scene within 3 seconds (reels)
    6. Carousel has 5-7 slides
    7. Statistics grounding check against the piece's cited insights/competitors
    8. Supporting IDs/names exist
    9. No negative competitor statements
    10. Brand claim validation (product-scoped)
    11. Voiceover length validation
    """
    errors: List[str] = []
    validated: List[dict] = []

    if not isinstance(data, dict):
        return [], ["Response is not a JSON object."]

    raw_pieces = data.get("content_pieces")
    if not isinstance(raw_pieces, list) or len(raw_pieces) == 0:
        return [], ["Output must contain a non-empty list of 'content_pieces'."]

    type_counts = {"reel_script": 0, "carousel": 0, "static_post": 0}
    comp_by_casefold = {n.casefold(): n for n in valid_comp_names}

    for idx, piece in enumerate(raw_pieces, 1):
        if not isinstance(piece, dict):
            errors.append(f"Content piece #{idx} is not a JSON object.")
            continue

        piece_errors: List[str] = []
        content_type = str(piece.get("content_type", "")).strip().lower()
        piece["content_type"] = content_type
        body = _normalize_body(piece.get("body"), content_type)
        piece["body"] = body

        for field in ["content_type", "title", "hook", "body", "caption", "call_to_action"]:
            val = piece.get(field)
            if not val or (isinstance(val, str) and len(val.strip()) < 3):
                piece_errors.append(f"Piece #{idx}: Missing or too short field '{field}'.")

        if content_type not in type_counts:
            piece_errors.append(
                f"Piece #{idx}: Invalid content_type '{content_type}'. Must be reel_script, carousel, or static_post."
            )
        else:
            type_counts[content_type] += 1

        hook = str(piece.get("hook", "")).strip()
        piece["hook"] = hook
        if len(hook) < 5:
            piece_errors.append(f"Piece #{idx}: Hook is missing or too short (< 5 chars).")

        if content_type == "reel_script":
            if not isinstance(body, list) or len(body) < 2:
                piece_errors.append(f"Piece #{idx} (reel_script): Body must be a list of at least 2 scenes.")
            else:
                total_duration = 0.0
                first_scene_duration = None
                for s_idx, scene in enumerate(body, 1):
                    if not isinstance(scene, dict):
                        piece_errors.append(f"Piece #{idx} scene #{s_idx}: Not a JSON object.")
                        continue
                    dur = scene.get("duration_seconds")
                    try:
                        dur_num = float(dur)
                        if dur_num <= 0:
                            piece_errors.append(f"Piece #{idx} scene #{s_idx}: duration_seconds must be positive.")
                        total_duration += dur_num
                        if s_idx == 1:
                            first_scene_duration = dur_num
                        for required in ("visual", "voiceover", "on_screen_text"):
                            if not isinstance(scene.get(required), str) or not scene[required].strip():
                                piece_errors.append(f"Piece #{idx} scene #{s_idx}: Missing '{required}'.")
                    except (ValueError, TypeError):
                        piece_errors.append(f"Piece #{idx} scene #{s_idx}: Invalid duration_seconds '{dur}'.")

                if total_duration < 15 or total_duration > 30:
                    piece_errors.append(
                        f"Piece #{idx} (reel_script): Total scene duration is {total_duration}s, "
                        "must be between 15-30 seconds."
                    )
                if first_scene_duration is not None and first_scene_duration > 3:
                    piece_errors.append(
                        f"Piece #{idx} (reel_script): First scene (hook) is {first_scene_duration}s, "
                        "must be ≤ 3 seconds so the hook lands in the first 3 seconds."
                    )

        elif content_type == "carousel":
            if not isinstance(body, list) or len(body) < 5 or len(body) > 7:
                piece_errors.append(
                    f"Piece #{idx} (carousel): Body must be a list of 5-7 slides, got {len(body) if isinstance(body, list) else 'non-list'}."
                )
            elif not isinstance(body[0], dict) or not (
                str(body[0].get("headline", "")).strip() or str(body[0].get("body_text", "")).strip()
            ):
                piece_errors.append(f"Piece #{idx} (carousel): First slide must carry the hook (headline or body_text).")

        elif content_type == "static_post":
            if not isinstance(body, dict):
                piece_errors.append(f"Piece #{idx} (static_post): Body must be a JSON object with visual_direction.")
            elif not body.get("visual_direction"):
                piece_errors.append(f"Piece #{idx} (static_post): Body missing 'visual_direction'.")

        ci_ids = _coerce_int_ids(piece.get("supporting_consumer_insight_ids", []))
        piece["supporting_consumer_insight_ids"] = ci_ids
        if not isinstance(ci_ids, list) or len(ci_ids) == 0:
            piece_errors.append(f"Piece #{idx}: Must have at least one supporting_consumer_insight_id.")
            ci_ids = []
        else:
            invalid_ids = [cid for cid in ci_ids if cid not in valid_ci_ids]
            if invalid_ids:
                piece_errors.append(
                    f"Piece #{idx}: Invalid consumer insight IDs {invalid_ids}. Valid: {sorted(list(valid_ci_ids))}."
                )

        raw_comp_names = piece.get("supporting_competitor_names", [])
        if not isinstance(raw_comp_names, list):
            piece_errors.append(f"Piece #{idx}: supporting_competitor_names must be a list.")
            canonical_comps: List[str] = []
        else:
            invalid_comps = [n for n in raw_comp_names if str(n).casefold() not in comp_by_casefold]
            if invalid_comps:
                piece_errors.append(
                    f"Piece #{idx}: Invalid competitor names {invalid_comps}. Valid: {sorted(list(valid_comp_names))}."
                )
            canonical_comps = [
                comp_by_casefold[str(n).casefold()]
                for n in raw_comp_names
                if str(n).casefold() in comp_by_casefold
            ]
            piece["supporting_competitor_names"] = canonical_comps
            if valid_comp_names and not canonical_comps and not invalid_comps:
                piece_errors.append(
                    f"Piece #{idx}: Must declare at least one supporting_competitor_name from {sorted(list(valid_comp_names))}."
                )

        all_text = _piece_text_blob(piece)
        piece_errors.extend(_competitor_attack_errors(all_text, valid_comp_names, idx))

        # Brand claim validation (brand_name and cited evidence passed as parameters)
        cited_ref_parts = [insight_texts.get(cid, "") for cid in ci_ids if cid in insight_texts]
        for name in canonical_comps if isinstance(raw_comp_names, list) else []:
            cited_ref_parts.append(competitor_texts.get(name.casefold(), ""))
        cited_evidence = " ".join(cited_ref_parts)
        
        claim_errors, claims_to_verify = _brand_claim_errors(all_text, idx, brand_name, cited_evidence)
        # Brand claim errors no longer cause rejection - they are saved to claims_to_verify
        piece["claims_to_verify"] = claims_to_verify

        # Statistics grounding validation
        stats_text = PLACEHOLDER_PATTERN.sub(" ", all_text)
        figures = extract_statistical_figures(stats_text)
        for fig in figures:
            core = fig["core_num"]
            raw_fig = fig["raw"]
            num_pat = r"(?<!\d)" + re.escape(core) + r"(?!\d)"
            if not re.search(num_pat, cited_evidence):
                piece_errors.append(
                    f"Piece #{idx}: Ungrounded statistic '{raw_fig}' (number: {core}) "
                    "not found in the cited consumer insight or competitor text."
                )

        # Voiceover length validation for reel scripts
        if content_type == "reel_script" and isinstance(body, list):
            for s_idx, scene in enumerate(body, 1):
                voiceover = str(scene.get("voiceover", ""))
                duration = scene.get("duration_seconds")
                try:
                    dur_num = float(duration) if duration else 0
                    # Estimate words: split by spaces, roughly 3 words per second
                    word_count = len(voiceover.split())
                    max_words = int(dur_num * 3)
                    if word_count > max_words:
                        piece_errors.append(
                            f"Piece #{idx} scene #{s_idx}: Voiceover too long for duration. "
                            f"Voiceover has {word_count} words but duration is {dur_num}s "
                            f"(max ~{max_words} words at 3 words/sec). Shorten voiceover or increase duration."
                        )
                except (ValueError, TypeError):
                    pass  # Duration validation already handled above

        if piece_errors:
            errors.extend(piece_errors)
            logger.warning(f"Piece #{idx} validation failed: {piece_errors}")
        else:
            validated.append(piece)

    if type_counts["reel_script"] != 2:
        errors.append(f"Expected 2 reel_script pieces but got {type_counts['reel_script']}.")
    if type_counts["carousel"] != 1:
        errors.append(f"Expected 1 carousel piece but got {type_counts['carousel']}.")
    if type_counts["static_post"] != 1:
        errors.append(f"Expected 1 static_post piece but got {type_counts['static_post']}.")

    # Emit detailed rejection log for every failed validation attempt
    if errors:
        logger.warning(
            f"Content validation REJECTED: {len(errors)} error(s) found across "
            f"{len(raw_pieces)} piece(s). {len(validated)} piece(s) passed."
        )
        for err_idx, err_msg in enumerate(errors, 1):
            logger.warning(f"  REJECTION REASON [{err_idx}/{len(errors)}]: {err_msg}")

    return validated, errors


def run_content_generation_pipeline(
    research_run_id: str,
    creative_gap_id: Optional[int] = None,
    platform: str = "instagram",
    client: Optional[genai.Client] = None,
    max_gemini_calls: int = 4
) -> List[Dict[str, Any]]:
    """
    Executes the Content Generation pipeline:
    1. Fetches brief, opportunity scores (requires Opportunity Engine completion).
    2. Selects top-ranked opportunity (rank 1) or specific creative_gap_id.
    3. Loads the creative gap's supporting insights and competitor data.
    4. Makes ONE Gemini call to generate 4 draft content pieces.
    5. Validates grounding, hooks, durations, IDs.
    6. Saves with status=draft.
    7. Uses ZERO SerpAPI calls.
    Returns list of saved content piece dicts.
    """
    logger.info(f"=== Starting Content Generation Pipeline for Run: {research_run_id} ===")

    # 1. Fetch Run Metadata
    run_data = get_run_with_evidence(research_run_id)
    if not run_data:
        raise ValueError(f"Research run '{research_run_id}' not found.")

    # 2. Fetch Opportunity Engine Output (REQUIRED)
    opportunity_scores = get_opportunity_scores(research_run_id)
    if not opportunity_scores:
        raise ValueError(
            f"Opportunity Engine output missing for run '{research_run_id}'. "
            "Please run the Opportunity Engine before generating content."
        )

    # 2b. Refuse to build on stale inputs — any opportunity citing superseded insights
    stale_opps = [o for o in opportunity_scores if o.get("is_stale")]
    if stale_opps:
        stale_ids = []
        for o in stale_opps:
            stale_ids.extend(o.get("stale_insight_ids", []))
        raise StaleInputError(
            entity_type="OpportunityScore",
            entity_id=[o["id"] for o in stale_opps],
            stale_insight_ids=sorted(set(stale_ids))
        )

    # 3. Select target opportunity
    selected_opportunity = None
    if creative_gap_id:
        # Find the opportunity for this specific creative_gap_id
        for opp in opportunity_scores:
            if opp.get("creative_gap_id") == creative_gap_id:
                selected_opportunity = opp
                break
        if not selected_opportunity:
            raise ValueError(
                f"No opportunity found for creative_gap_id {creative_gap_id} in run '{research_run_id}'. "
                f"Available gap IDs: {[o.get('creative_gap_id') for o in opportunity_scores]}."
            )
    else:
        selected_opportunity = next(
            (opp for opp in opportunity_scores if opp.get("rank") == 1),
            opportunity_scores[0],
        )

    target_gap_id = selected_opportunity["creative_gap_id"]
    logger.info(
        f"Selected opportunity: Rank #{selected_opportunity.get('rank')}, "
        f"Gap ID {target_gap_id}, Score {selected_opportunity.get('overall_score')}"
    )

    # 4. Fetch the creative gap detail - only non-superseded
    all_gaps = get_creative_gaps(research_run_id, include_superseded=False) or []
    target_gap = None
    for g in all_gaps:
        if g["id"] == target_gap_id:
            target_gap = g
            break
    if not target_gap:
        raise ValueError(f"Creative gap ID {target_gap_id} not found for run '{research_run_id}'.")

    # 5. Fetch consumer insights and competitor entries
    all_insights = get_consumer_insights(research_run_id) or []
    competitive_intel = get_competitive_intelligence(research_run_id) or {}
    competitor_entries = competitive_intel.get("competitors", [])

    ci_map = {ci["id"]: ci for ci in all_insights}
    gap_ci_ids = set(target_gap.get("supporting_consumer_insight_ids", []))
    missing_ci_ids = gap_ci_ids - set(ci_map)
    if missing_ci_ids:
        raise ValueError(
            f"Creative gap {target_gap_id} cites consumer insight IDs that are missing from run "
            f"'{research_run_id}': {sorted(missing_ci_ids)}."
        )
    valid_ci_ids = gap_ci_ids
    competitor_by_name = {
        str(comp.get("name", "")).casefold(): comp
        for comp in competitor_entries if comp.get("name")
    }
    gap_comp_names = set(target_gap.get("supporting_competitor_names", []))
    missing_competitors = [name for name in gap_comp_names if name.casefold() not in competitor_by_name]
    if missing_competitors:
        raise ValueError(
            f"Creative gap {target_gap_id} cites competitors missing from run '{research_run_id}': "
            f"{sorted(missing_competitors)}."
        )
    valid_comp_names = {competitor_by_name[name.casefold()]["name"] for name in gap_comp_names}

    insight_texts = {cid: ci_map[cid].get("insight_text", "") for cid in valid_ci_ids}
    competitor_texts: Dict[str, str] = {}
    for name in valid_comp_names:
        comp = competitor_by_name[name.casefold()]
        competitor_texts[name.casefold()] = " ".join([
            str(comp.get("name", "")),
            str(comp.get("positioning", "")),
            " ".join(str(s) for s in (comp.get("strengths") or [])),
            " ".join(str(w) for w in (comp.get("weaknesses") or [])),
        ])

    logger.info(
        f"Context loaded: {len(valid_ci_ids)} supporting insights, "
        f"{len(valid_comp_names)} supporting competitors."
    )

    # 6. Initialize Gemini Client
    if client is None:
        from services.credential_manager import get_gemini_client
        from database import get_research_run_owner
        _owner_id = get_research_run_owner(research_run_id)
        client = get_gemini_client(user_id=_owner_id)

    models_to_try = GEMINI_FALLBACK_MODELS or [GEMINI_MODEL_NAME]
    feedback: Optional[str] = None
    final_validated_pieces: List[dict] = []

    # 7. ONE Gemini Call for content generation with global attempt counter (max 4 calls total)
    global_attempt = 0
    max_global_attempts = max_gemini_calls

    for model_idx, model_name in enumerate(models_to_try):
        if global_attempt >= max_global_attempts:
            logger.warning(f"Global attempt limit ({max_global_attempts}) reached. Stopping.")
            break

        while global_attempt < max_global_attempts:
            global_attempt += 1
            prompt = build_content_prompt(
                run_data=run_data,
                opportunity=selected_opportunity,
                creative_gap=target_gap,
                consumer_insights=all_insights,
                competitor_entries=competitor_entries,
                platform=platform,
                feedback=feedback
            )
            prompt = append_knowledge_guidance(
                prompt,
                "creative strategy brand voice audience motivation platform content measurement "
                f"{run_data.get('objective', '')} {run_data.get('audience', '')}",
            )
            logger.info(f"Calling Gemini model '{model_name}' for Content Generation (Global attempt {global_attempt}/{max_global_attempts})...")

            try:
                with track_api_call(
                    provider="gemini",
                    call_type="content_generation",
                    stage="ContentGeneration_Draft",
                    model=model_name,
                    research_run_id=research_run_id,
                    input_text=prompt[:400]
                ) as tracker:
                    pacer.wait_for_turn(model_name, sleep_fn=time.sleep)
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.4,
                            response_mime_type="application/json"
                        )
                    )
                    tracker.set_response(response)

                raw_text = response.text if hasattr(response, "text") and response.text else ""
                parsed = parse_llm_json(raw_text)

                validated, errors = validate_content_pieces(
                    data=parsed,
                    valid_ci_ids=valid_ci_ids,
                    valid_comp_names=valid_comp_names,
                    insight_texts=insight_texts,
                    competitor_texts=competitor_texts,
                    platform=platform,
                    brand_name=run_data.get("brand", "Brand")
                )

                if errors:
                    logger.warning(f"Content validation errors on global attempt {global_attempt}: {errors}")
                    if global_attempt < max_global_attempts:
                        feedback = (
                            "Your previous content generation output contained errors:\n"
                            + "\n".join(f"- {e}" for e in errors)
                            + "\nCRITICAL: You must generate EXACTLY 2 reel_script, 1 carousel, 1 static_post."
                            + f"\nCRITICAL: Valid consumer insight IDs: {sorted(list(valid_ci_ids))}."
                            + f"\nCRITICAL: Valid competitor names: {sorted(list(valid_comp_names))}."
                            + "\nCRITICAL: Do NOT invent statistics. Every number must appear in the cited insight or competitor text."
                            + "\nCRITICAL: Use [VERIFIED CLAIM NEEDED: ...] for any specific product fact not in the cited evidence."
                            + "\nCRITICAL: Reel first-scene duration must be ≤ 3 seconds so the hook lands in the first 3 seconds."
                            + "\nCRITICAL: Total reel duration must be 15-30 seconds."
                            + "\nCRITICAL: Voiceover length must fit scene duration (~2.5 words per second)."
                            + "\nCRITICAL: Do not make negative statements or direct comparisons about named competitors."
                            + "\nCRITICAL: Brand product claims (price, affordability, quality, durability, comfort, materials, eco benefits) must use [VERIFIED CLAIM NEEDED: ...] placeholder unless in cited evidence."
                        )
                        continue
                    raise ValueError(f"Content generation validation failed after {global_attempt} global attempts: {errors}")
                else:
                    logger.info(f"Successfully generated and validated {len(validated)} content pieces on global attempt {global_attempt}.")
                    final_validated_pieces = validated
                    break

            except json.JSONDecodeError as jde:
                logger.error(f"Invalid JSON from {model_name} on global attempt {global_attempt}: {jde}")
                if global_attempt < max_global_attempts:
                    feedback = "Your previous output was not valid JSON. Return ONLY the required JSON object."
                    continue
                raise ValueError(f"Content generation failed after {global_attempt} global attempts due to invalid JSON: {jde}")
            except Exception as exc:
                err_str = str(exc)
                logger.error(f"Error during content generation global attempt {global_attempt} with {model_name}: {err_str}")
                if "validation failed after" in err_str:
                    raise

                is_transient = any(k in err_str.lower() for k in [
                    "429", "resource_exhausted", "quota", "rate limit",
                    "503", "unavailable", "high demand", "overloaded", "server error", "500"
                ])

                if is_transient:
                    if global_attempt < max_global_attempts:
                        if any(k in err_str.lower() for k in ["429", "resource_exhausted", "quota", "rate limit", "too many requests"]):
                            pacer.handle_429(model_name, exc, sleep_fn=time.sleep)
                        else:
                            # Exponential backoff with jitter: 2s, 5s, 10s
                            base_wait = [2, 5, 10][min(global_attempt - 1, 2)]
                            jitter = random.uniform(0.5, 1.5)
                            wait_s = base_wait * jitter
                            logger.warning(
                                f"Transient error on '{model_name}' ({err_str[:100]}). "
                                f"Retrying in {wait_s:.1f}s (global attempt {global_attempt}/{max_global_attempts})."
                            )
                            time.sleep(wait_s)
                        continue
                    else:
                        # Fall back to next model if available
                        if model_idx < len(models_to_try) - 1:
                            logger.warning(f"Global attempt limit ({max_global_attempts}) reached on '{model_name}'. Trying fallback model...")
                            break
                        else:
                            raise RuntimeError(
                                f"Content generation failed after {max_global_attempts} global attempts across all models. "
                                f"Last error: {err_str}"
                            )
                else:
                    # Non-transient error - raise immediately
                    raise

        if final_validated_pieces:
            break

    if not final_validated_pieces:
        raise RuntimeError(
            f"Failed to generate content after {global_attempt} global attempts across all models ({models_to_try}). "
            "No valid pieces were produced. Nothing was saved to the database."
        )

    # 8. Attach creative_gap_id and platform to each piece for DB persistence
    for piece in final_validated_pieces:
        piece["creative_gap_id"] = target_gap_id
        piece["platform"] = platform

    # 9. Save to database with status = draft
    saved_records = save_content_pieces(research_run_id, final_validated_pieces)
    logger.info(
        f"Successfully saved {len(saved_records)} draft content pieces "
        f"for run '{research_run_id}', gap #{target_gap_id}."
    )
    return saved_records
