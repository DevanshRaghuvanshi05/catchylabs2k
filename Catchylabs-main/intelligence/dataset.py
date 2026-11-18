"""Validation helpers for provenance-aware supervised-learning candidates."""

from __future__ import annotations

import hashlib
import re
from typing import Any


DATASET_VERSION = "0.1.0"
SUPPORTED_TASKS = {
    "creative_format",
    "humor_style",
    "hook_type",
    "audience_tension",
    "brand_voice",
    "strategy_quality",
    "originality_risk",
    "cta_funnel_stage",
    "evidence_quality",
    "creative_preference",
    "brief_completeness",
    "objective_kpi_match",
    "language_script",
}
SUPPORTED_SPLITS = {"train", "validation", "test"}
SUPPORTED_RIGHTS = {"internal_original", "licensed", "user_authorized"}
SUPPORTED_REVIEW = {"unreviewed", "human_reviewed", "rejected"}
SUPPORTED_LANGUAGES = {"english", "hindi", "hinglish"}
SUPPORTED_SCRIPTS = {"latin", "devanagari", "mixed"}


def validate_example(example: dict[str, Any]) -> list[str]:
    """Return schema issues without upgrading synthetic or unreviewed labels."""
    issues: list[str] = []
    if example.get("dataset_version") != DATASET_VERSION:
        issues.append("unsupported_dataset_version")
    if example.get("task") not in SUPPORTED_TASKS:
        issues.append("unsupported_task")
    if example.get("split") not in SUPPORTED_SPLITS:
        issues.append("invalid_split")
    if not isinstance(example.get("input"), str) or not example["input"].strip():
        issues.append("missing_input")
    if not isinstance(example.get("expected_output"), (str, dict, list)):
        issues.append("missing_expected_output")
    language = example.get("language")
    if not isinstance(language, dict) or language.get("language") not in SUPPORTED_LANGUAGES:
        issues.append("language_annotation_required")
    elif language.get("script") not in SUPPORTED_SCRIPTS:
        issues.append("script_annotation_required")
    if example.get("annotation_language") not in SUPPORTED_LANGUAGES:
        issues.append("annotation_language_required")
    if not isinstance(example.get("synthetic"), bool):
        issues.append("synthetic_status_required")
    provenance = example.get("provenance")
    if not isinstance(provenance, dict):
        issues.append("provenance_required")
    else:
        if provenance.get("usage_rights") not in SUPPORTED_RIGHTS:
            issues.append("usage_rights_unverified")
        if provenance.get("source_type") not in {"original_synthetic", "user_authorized", "licensed"}:
            issues.append("unsupported_source_type")
        if provenance.get("source_type") == "licensed" and not provenance.get("license_reference"):
            issues.append("license_reference_required")
    review = example.get("review")
    if not isinstance(review, dict) or review.get("status") not in SUPPORTED_REVIEW:
        issues.append("review_status_required")
    elif review.get("status") == "human_reviewed" and not review.get("reviewer_reference"):
        issues.append("reviewer_reference_required")
    if isinstance(review, dict) and review.get("disagreement") not in (None, True, False):
        issues.append("invalid_review_disagreement")
    confidence = review.get("confidence") if isinstance(review, dict) else None
    if confidence is not None and (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not 0 <= confidence <= 1
    ):
        issues.append("invalid_review_confidence")
    return issues


def find_split_leakage(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Detect exact normalized input duplicates assigned to different splits."""
    seen: dict[tuple[str, str], str] = {}
    leaked: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        text = example.get("input")
        task = example.get("task")
        split = example.get("split")
        if not isinstance(text, str) or not isinstance(task, str) or split not in SUPPORTED_SPLITS:
            continue
        normalized = re.sub(r"\W+", " ", text.casefold(), flags=re.UNICODE).strip()
        fingerprint = hashlib.sha256(f"{task}\0{normalized}".encode("utf-8")).hexdigest()
        prior_split = seen.get((task, fingerprint))
        if prior_split is not None and prior_split != split:
            leaked.append(
                {"index": index, "task": task, "prior_split": prior_split, "split": split}
            )
        else:
            seen[(task, fingerprint)] = split
    return leaked


def eligible_for_training(example: dict[str, Any]) -> bool:
    """Only vetted, rights-cleared examples can enter a future training run."""
    return (
        not validate_example(example)
        and example["review"]["status"] == "human_reviewed"
        and example["provenance"]["usage_rights"]
        in {"internal_original", "licensed", "user_authorized"}
        and example["split"] == "train"
    )
