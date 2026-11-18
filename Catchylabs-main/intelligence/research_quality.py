"""Provider-independent research planning and evidence provenance checks."""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .knowledge import KNOWLEDGE_VERSION, retrieve_knowledge
from .language import detect_language, resolve_language


RESEARCH_CONTEXT_VERSION = "1.0.0"


def _canonical_http_url(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = urlsplit(value.strip())
        if (
            parsed.scheme.lower() not in ("http", "https")
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
        ):
            return None
        host = parsed.hostname.lower()
        if parsed.port:
            host = f"{host}:{parsed.port}"
        if ":" in parsed.hostname and not parsed.hostname.startswith("["):
            host = f"[{parsed.hostname.lower()}]" + (f":{parsed.port}" if parsed.port else "")
        return urlunsplit((parsed.scheme.lower(), host, parsed.path or "/", parsed.query, ""))
    except ValueError:
        return None


def build_research_plan(brief: dict[str, Any]) -> dict[str, Any]:
    """Describe the method actually supported by the configured public-search workflow."""
    objective = str(brief.get("objective") or "").strip()
    return {
        "version": RESEARCH_CONTEXT_VERSION,
        "management_decision": None,
        "management_decision_status": "not_specified_in_brief",
        "research_question": objective or None,
        "research_objectives": [objective] if objective else [],
        "hypotheses": [],
        "hypotheses_status": "not_specified",
        "design": "secondary_desk_research",
        "design_rationale": (
            "The configured workflow retrieves public web search results and synthesizes "
            "selected source excerpts. It does not collect primary data or identify causal effects."
        ),
        "evidence_requirements": [
            "Traceable source URL and source title where returned by search",
            "Source excerpt supporting each reported observation",
            "Multiple independent sources for consequential conclusions",
            "Counterevidence and source limitations before strategic recommendations",
        ],
        "workflow": [
            "Define the management decision and researchable question",
            "Set objectives and hypotheses only where appropriate",
            "Retrieve and record public secondary sources",
            "Extract source-grounded observations",
            "Assess source provenance, diversity, contradictions, and limitations",
            "Separate observation, interpretation, hypothesis, recommendation, and limitation",
        ],
        "open_questions": [
            "Which management decision will these findings inform?",
            "Is primary customer data available to validate secondary-source patterns?",
        ],
        "limitations": [
            "No primary interviews, surveys, or representative sampling are performed.",
            "Search-result excerpts are selected observations and do not represent the target population.",
            "The workflow does not support causal identification or statistical significance testing.",
            "Source reliability, recency, and independence require human review.",
            "Language detection is heuristic; language alone does not establish region, identity, or audience behavior.",
        ],
    }


def build_evidence_registry(
    evidence_items: list[dict[str, Any]],
    retrieved_at: str | None = None,
) -> list[dict[str, Any]]:
    """Create provenance records without inventing missing source fields."""
    timestamp = retrieved_at or datetime.now(timezone.utc).isoformat()
    registry = []
    for index, item in enumerate(evidence_items, start=1):
        if not isinstance(item, dict):
            continue
        source = item.get("source") if isinstance(item.get("source"), dict) else {}
        excerpt = item.get("evidence") or item.get("excerpt") or None
        claim = item.get("claim") or None
        canonical_url = _canonical_http_url(source.get("url"))
        raw_confidence = item.get("confidence")
        confidence = (
            float(raw_confidence)
            if isinstance(raw_confidence, (int, float))
            and not isinstance(raw_confidence, bool)
            and math.isfinite(raw_confidence)
            and 0 <= raw_confidence <= 1
            else None
        )
        has_provenance = bool(claim and excerpt and canonical_url)
        registry.append(
            {
                "evidence_id": item.get("id") or f"evidence-{index}",
                "evidence_type": "source_reported_finding" if has_provenance else "unverified_claim",
                "verification_status": (
                    "requires_human_source_review" if has_provenance else "incomplete_provenance"
                ),
                "claim_support_status": "not_semantically_verified",
                "claim_support_rationale": (
                    "The stored excerpt is provenance, not proof that it entails the extracted claim. "
                    "Review the original source and context."
                ),
                "claim": claim,
                "excerpt": excerpt,
                "source_language": detect_language(str(excerpt or "")),
                "claim_language": detect_language(str(claim or "")),
                "translation_status": "not_asserted",
                "translation_note": (
                    "The stored evidence excerpt remains the retrieved source text. The extracted claim is a model summary, "
                    "not a verified translation."
                ),
                "topic": item.get("topic") or None,
                "confidence": confidence,
                "confidence_rationale": (
                    "Model-provided extraction score; it is not calibrated and has not been independently verified."
                    if confidence is not None
                    else "No valid model confidence score was available."
                ),
                "contradictory_evidence": {
                    "status": "not_assessed",
                    "evidence_ids": [],
                },
                "freshness_limitation": (
                    "Publication date was not available; freshness has not been assessed."
                    if not (source.get("published_date") or source.get("published_at"))
                    else "Publication date is recorded; recency and continuing validity have not been assessed."
                ),
                "source": {
                    "title": source.get("title") or None,
                    "publisher_or_author": source.get("publisher") or None,
                    "canonical_url": canonical_url,
                    "publication_date": source.get("published_date") or source.get("published_at"),
                    "retrieval_date": timestamp,
                    "source_type": "search_result_excerpt",
                    "usage_rights_status": "not_assessed",
                    "retention_status": "not_assessed",
                    "original_excerpt_preserved": bool(excerpt),
                },
                "reliability_considerations": [
                    "Only a search-result excerpt was retained; the full page was not necessarily reviewed.",
                    "Search-result metadata and extracted excerpts are not an independent source-quality assessment.",
                    "The selection is not a representative sample of the target audience.",
                ],
            }
        )
    return registry


def evaluate_evidence_quality(registry: list[dict[str, Any]]) -> dict[str, Any]:
    """Run transparent descriptive checks; do not imply statistical validation."""
    domains = set()
    missing_urls = 0
    for entry in registry:
        source = entry.get("source", {})
        url = source.get("canonical_url")
        if not url:
            missing_urls += 1
            continue
        domains.add(urlsplit(url).hostname or "")

    evidence_count = len(registry)
    if evidence_count == 0:
        diversity_status = "not_assessable"
    elif len(domains) < 3:
        diversity_status = "caution"
    else:
        diversity_status = "review_required"

    return {
        "version": RESEARCH_CONTEXT_VERSION,
        "evidence_count": evidence_count,
        "unique_source_domain_count": len(domains),
        "missing_source_url_count": missing_urls,
        "checks": [
            {
                "name": "source_diversity",
                "status": diversity_status,
                "finding": (
                    "Fewer than three distinct source domains are represented; treat patterns as preliminary."
                    if diversity_status == "caution"
                    else "Domain count is descriptive only; source independence and quality remain unverified."
                ),
            },
            {
                "name": "primary_data_and_representativeness",
                "status": "not_available",
                "finding": "Public search excerpts are not primary research or a representative audience sample.",
            },
            {
                "name": "causal_and_statistical_claims",
                "status": "not_tested",
                "finding": "No causal identification or statistical significance test was performed.",
            },
            {
                "name": "source_quality_recency_and_independence",
                "status": "human_review_required",
                "finding": "Publisher incentives, publication recency, and independent corroboration require review.",
            },
        ],
        "interpretation_boundary": (
            "These checks describe available provenance. They do not establish truth, representativeness, "
            "causality, statistical significance, or the validity of downstream recommendations."
        ),
    }


def build_research_context(
    brief: dict[str, Any],
    evidence_items: list[dict[str, Any]],
    retrieved_at: str | None = None,
    search_queries: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble stable planning, knowledge, and evidence context for persistence."""
    registry = build_evidence_registry(evidence_items, retrieved_at=retrieved_at)
    knowledge_query = " ".join(
        str(brief.get(field) or "")
        for field in ("brand", "market", "audience", "objective")
    )
    language = resolve_language(
        " ".join(str(brief.get(field) or "") for field in ("objective", "audience")),
        str(brief.get("language_preference") or "auto"),
    )
    return {
        "version": RESEARCH_CONTEXT_VERSION,
        "language": language,
        "research_plan": build_research_plan(brief),
        "search_language_coverage": {
            "status": "query_language_hints_only",
            "queries": [
                {"query": query, **detect_language(query)}
                for query in (search_queries or [])
                if isinstance(query, str)
            ],
            "limitation": (
                "Detected query language is heuristic and does not establish source availability, "
                "audience representation, or translation fidelity."
            ),
        },
        "knowledge_base": {
            "version": KNOWLEDGE_VERSION,
            "retrieval_method": "lexical_token_overlap_with_multilingual_keywords",
            "documents": retrieve_knowledge(knowledge_query, limit=4),
        },
        "evidence_registry": registry,
        "quality_review": evaluate_evidence_quality(registry),
        "output_labels": {
            "source_reported_finding": "A model-extracted claim paired with a retrieved excerpt; claim support requires source review.",
            "unverified_claim": "Claim provenance is incomplete and must not be presented as a verified fact.",
            "interpretation": "A reasoned explanation of evidence, not a directly observed fact.",
            "hypothesis": "A plausible explanation or opportunity that remains untested.",
            "recommendation": "A proposed action that should be evaluated against the decision context.",
            "limitation": "An uncertainty or constraint on what the available evidence establishes.",
        },
    }
