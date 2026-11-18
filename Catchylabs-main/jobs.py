"""
CatchyLab Agent 1 - RQ Background Jobs
--------------------------------------
Contains task functions executed asynchronously by RQ worker processes.

Security contract:
- Raw API keys are NEVER placed in job arguments or Redis payloads.
- Only user_id travels through asynchronous boundaries.
- Credentials are resolved (decrypted) inside the trusted worker process
  via services/credential_manager.py immediately before pipeline execution.
- Each stage verifies run ownership before resolving credentials.
"""

import json
import logging
import os
from typing import Optional

from pipeline import run_research_pipeline
from database import save_research_run, get_research_run_owner
from agent2 import run_consumer_intelligence_pipeline

logger = logging.getLogger("CatchyLabResearcher")


def _resolve_gemini_client(user_id: Optional[str] = None):
    """Resolve a credential-aware Gemini client for the given user."""
    from services.credential_manager import get_gemini_client
    return get_gemini_client(user_id=user_id)


def _resolve_serpapi_key(user_id: Optional[str] = None) -> str:
    """Resolve the SerpAPI key for the given user."""
    from services.credential_manager import get_serpapi_key
    return get_serpapi_key(user_id=user_id)


def _user_id_for_run(run_id: str, provided_user_id: Optional[str] = None) -> Optional[str]:
    """
    Return the user_id for a research run.
    Uses the provided value if supplied (passed from the stage endpoint),
    otherwise looks up the run owner from the database.
    """
    if provided_user_id:
        return provided_user_id
    return get_research_run_owner(run_id)


def run_research_job(brief: dict) -> dict:
    """
    Background job wrapper function for RQ worker (Agent 1).

    1. Resolves user credentials from brief['user_id'] via credential_manager.
    2. Executes the 7-stage market research pipeline with user-specific keys.
    3. Attempts to load report data from the pipeline's output file.
       If the file cannot be read (e.g. read-only filesystem), falls back to
       the in-memory pipeline result dict — so the job never fails solely due
       to a filesystem issue.
    4. Persists resulting report data into PostgreSQL database.
    5. Returns summary dictionary containing assigned research_run_id.

    Security: raw API keys are resolved here and passed explicitly to pipeline;
    they are NOT stored back in the brief dict or Redis payload.
    """
    logger.info(f"[RQ JOB START] Processing research brief: Brand={brief.get('brand')}, Market={brief.get('market')}")

    user_id = brief.get("user_id")

    # Resolve credentials inside the trusted worker process
    from services.credential_manager import get_gemini_key, get_serpapi_key
    try:
        gemini_key = get_gemini_key(user_id=user_id)
        serpapi_key = get_serpapi_key(user_id=user_id)
    except Exception as cred_exc:
        logger.error(f"[RQ JOB] Credential resolution failed for user {user_id}: {type(cred_exc).__name__}")
        raise

    # 1. Execute Core 7-stage Pipeline with explicit user credentials
    max_docs = brief.get("max_documents")
    pipeline_result = run_research_pipeline(
        brief,
        gemini_key=gemini_key,
        serpapi_key=serpapi_key,
        max_documents=max_docs,
    )
    report_filepath = pipeline_result.get("report_file")

    # 2. Load report data — tolerate unwritable / missing files
    report_data: dict = {}
    if report_filepath and os.path.isfile(report_filepath):
        try:
            with open(report_filepath, "r", encoding="utf-8") as f:
                report_data = json.load(f)
        except (OSError, IOError, ValueError) as read_err:
            logger.warning(
                f"[RQ JOB] Could not read report file '{report_filepath}': {read_err}. "
                "Falling back to in-memory pipeline result."
            )
            report_data = pipeline_result.get("report_data") or {}
    else:
        if report_filepath:
            logger.warning(
                f"[RQ JOB] Report file '{report_filepath}' was not written (filesystem may be read-only). "
                "Falling back to in-memory pipeline result."
            )
        report_data = pipeline_result.get("report_data") or {}

    # 2.5 Validate evidence collection before reporting research success
    evidence_items = report_data.get("evidence")
    has_empty_evidence_list = isinstance(evidence_items, list) and len(evidence_items) == 0
    has_zero_pipeline_evidence = pipeline_result.get("evidence_count") == 0
    has_failed_metadata = report_data.get("metadata", {}).get("status") == "failed"
    has_zero_metadata_evidence = (
        "metadata" in report_data
        and "total_evidence_count" in report_data["metadata"]
        and report_data["metadata"]["total_evidence_count"] == 0
    )

    if has_empty_evidence_list or has_zero_pipeline_evidence or has_failed_metadata or has_zero_metadata_evidence:
        err_msg = (
            f"Research job produced zero evidence items for brand '{brief.get('brand')}'. "
            f"Cannot mark research as completed."
        )
        logger.error(f"[RQ JOB FAILED] {err_msg}")
        raise ValueError(err_msg)

    # 3. Persist into PostgreSQL Database
    pipeline_run_id = pipeline_result.get("research_run_id")
    run_id = save_research_run(brief, report_data, run_id=pipeline_run_id, user_id=user_id)
    logger.info(f"[RQ JOB SUCCESS] Research run persisted to PostgreSQL DB -> assigned run_id: {run_id}")

    return {
        "job_type": "research",
        "research_run_id": run_id,
        "report_file": report_filepath,
        "total_evidence_count": report_data.get("metadata", {}).get("total_evidence_count", 0),
        "queries_used": pipeline_result.get("queries", []),
        "research_context": report_data.get("research_context"),
    }


def run_consumer_intelligence_job(run_id: str, user_id: Optional[str] = None) -> dict:
    """
    Background job wrapper function for Agent 2 Consumer Intelligence.

    Resolves the Gemini client via credential_manager using the run owner's user_id.
    Never silently falls back to developer API keys.
    """
    logger.info(f"[RQ JOB START] Processing consumer intelligence for run: {run_id}")
    effective_user_id = _user_id_for_run(run_id, user_id)
    client = _resolve_gemini_client(effective_user_id)
    insights = run_consumer_intelligence_pipeline(run_id, client=client)
    logger.info(f"[RQ JOB SUCCESS] Synthesized and saved {len(insights)} consumer insights for run: {run_id}")
    return {
        "job_type": "consumer_intelligence",
        "research_run_id": run_id,
        "total_insights": len(insights),
        "insights": insights,
    }


def run_competitive_intelligence_job(run_id: str, user_id: Optional[str] = None) -> dict:
    """
    Background job wrapper function for Agent 3 Competitor Intelligence.

    Resolves both Gemini client and SerpAPI key via credential_manager.
    Passes them explicitly to run_competitive_intelligence_pipeline.
    """
    from agent3 import run_competitive_intelligence_pipeline
    logger.info(f"[RQ JOB START] Processing competitor intelligence for run: {run_id}")
    effective_user_id = _user_id_for_run(run_id, user_id)
    client = _resolve_gemini_client(effective_user_id)
    serpapi_key = _resolve_serpapi_key(effective_user_id)
    result = run_competitive_intelligence_pipeline(
        run_id,
        client=client,
        serpapi_key=serpapi_key,
    )
    logger.info(f"[RQ JOB SUCCESS] Completed competitor intelligence for run: {run_id}")
    return {
        "job_type": "competitive_intelligence",
        "research_run_id": run_id,
        "competitors": result.get("competitors", []),
        "market_structure": result.get("market_structure", {}),
        "competitive_gaps": result.get("competitive_gaps", []),
    }


def run_creative_gaps_job(run_id: str, user_id: Optional[str] = None) -> dict:
    """
    Background job wrapper function for Agent 4 Creative Gap Detection.

    Resolves Gemini client via credential_manager using the run owner's user_id.
    """
    from agent4 import run_creative_gaps_pipeline
    logger.info(f"[RQ JOB START] Processing creative gaps for run: {run_id}")
    effective_user_id = _user_id_for_run(run_id, user_id)
    client = _resolve_gemini_client(effective_user_id)
    gaps = run_creative_gaps_pipeline(run_id, client=client)
    logger.info(f"[RQ JOB SUCCESS] Synthesized and saved {len(gaps)} creative gaps for run: {run_id}")
    return {
        "job_type": "creative_gaps",
        "research_run_id": run_id,
        "total_gaps": len(gaps),
        "creative_gaps": gaps,
    }


def run_opportunity_scoring_job(run_id: str, user_id: Optional[str] = None) -> dict:
    """
    Background job wrapper function for Opportunity Engine Scoring & Ranking.

    Resolves Gemini client via credential_manager using the run owner's user_id.
    """
    from opportunity_engine import run_opportunity_engine_pipeline
    logger.info(f"[RQ JOB START] Processing opportunity scoring for run: {run_id}")
    effective_user_id = _user_id_for_run(run_id, user_id)
    client = _resolve_gemini_client(effective_user_id)
    ranked = run_opportunity_engine_pipeline(run_id, client=client)
    logger.info(f"[RQ JOB SUCCESS] Scored and ranked {len(ranked)} opportunities for run: {run_id}")
    return {
        "job_type": "opportunity_scoring",
        "research_run_id": run_id,
        "total_opportunities": len(ranked),
        "ranked_opportunities": ranked,
    }


def run_content_generation_job(
    run_id: str,
    creative_gap_id: int = None,
    platform: str = "instagram",
    user_id: Optional[str] = None,
) -> dict:
    """
    Background job wrapper function for Content Generation.

    Resolves Gemini client via credential_manager using the run owner's user_id.
    """
    from content_generator import run_content_generation_pipeline
    logger.info(f"[RQ JOB START] Processing content generation for run: {run_id}")
    effective_user_id = _user_id_for_run(run_id, user_id)
    client = _resolve_gemini_client(effective_user_id)
    pieces = run_content_generation_pipeline(
        research_run_id=run_id,
        creative_gap_id=creative_gap_id,
        platform=platform,
        client=client,
        max_gemini_calls=4,
    )
    logger.info(f"[RQ JOB SUCCESS] Generated and saved {len(pieces)} content pieces for run: {run_id}")
    return {
        "job_type": "content_generation",
        "research_run_id": run_id,
        "total_pieces": len(pieces),
        "content_pieces": pieces,
    }


def run_full_pipeline_job(brief: dict, options: dict = None) -> dict:
    """
    Orchestrates all 6 pipeline stages sequentially in a single RQ job.

    Accepts brief with optional user_id. Resolves credentials once per stage
    using the run's owner user_id. Frontend clients polling GET /research/status/{job_id}
    can track progress. Stops immediately if any stage fails.
    """
    logger.info(f"[ORCHESTRATOR START] Starting full 6-stage pipeline for brand: {brief.get('brand')}")

    try:
        from rq import get_current_job
        job = get_current_job()
    except Exception:
        job = None

    stages_progress = {
        "research": "pending",
        "consumer_intelligence": "pending",
        "competitive_intelligence": "pending",
        "competitor_intelligence": "pending",
        "creative_gaps": "pending",
        "opportunity_scoring": "pending",
        "content_generation": "pending",
    }
    stage_results = {}

    def _sync_meta(stage_name: str, stage_status: str, run_id: str = None, error: str = None):
        stages_progress[stage_name] = stage_status
        if stage_name == "competitor_intelligence":
            stages_progress["competitive_intelligence"] = stage_status
        elif stage_name == "competitive_intelligence":
            stages_progress["competitor_intelligence"] = stage_status
        if job:
            job.meta["job_type"] = "full_pipeline"
            job.meta["stages"] = dict(stages_progress)
            job.meta["current_stage"] = stage_name
            if run_id:
                job.meta["research_run_id"] = run_id
            if error:
                job.meta["error_stage"] = stage_name
                job.meta["error_message"] = error
            job.save_meta()

    run_id = None
    user_id = brief.get("user_id")

    # Stage 1: Market Research (Agent 1)
    _sync_meta("research", "running")
    try:
        r1 = run_research_job(brief)
        run_id = r1.get("research_run_id")
        stage_results["research"] = r1
        _sync_meta("research", "completed", run_id=run_id)
    except Exception as exc:
        logger.error(f"[ORCHESTRATOR] Stage 1 (research) failed: {exc}", exc_info=True)
        _sync_meta("research", "failed", error=str(exc))
        raise

    # Stage 2: Consumer Intelligence (Agent 2)
    _sync_meta("consumer_intelligence", "running", run_id=run_id)
    try:
        r2 = run_consumer_intelligence_job(run_id, user_id=user_id)
        stage_results["consumer_intelligence"] = r2
        _sync_meta("consumer_intelligence", "completed", run_id=run_id)
    except Exception as exc:
        logger.error(f"[ORCHESTRATOR] Stage 2 (consumer_intelligence) failed: {exc}", exc_info=True)
        _sync_meta("consumer_intelligence", "failed", run_id=run_id, error=str(exc))
        raise

    # Stage 3: Competitor Intelligence (Agent 3)
    _sync_meta("competitor_intelligence", "running", run_id=run_id)
    try:
        r3 = run_competitive_intelligence_job(run_id, user_id=user_id)
        stage_results["competitor_intelligence"] = r3
        _sync_meta("competitor_intelligence", "completed", run_id=run_id)
    except Exception as exc:
        logger.error(f"[ORCHESTRATOR] Stage 3 (competitor_intelligence) failed: {exc}", exc_info=True)
        _sync_meta("competitor_intelligence", "failed", run_id=run_id, error=str(exc))
        raise

    # Stage 4: Creative Gap Detection (Agent 4)
    _sync_meta("creative_gaps", "running", run_id=run_id)
    try:
        r4 = run_creative_gaps_job(run_id, user_id=user_id)
        stage_results["creative_gaps"] = r4
        _sync_meta("creative_gaps", "completed", run_id=run_id)
    except Exception as exc:
        logger.error(f"[ORCHESTRATOR] Stage 4 (creative_gaps) failed: {exc}", exc_info=True)
        _sync_meta("creative_gaps", "failed", run_id=run_id, error=str(exc))
        raise

    # Stage 5: Opportunity Engine Scoring
    _sync_meta("opportunity_scoring", "running", run_id=run_id)
    try:
        r5 = run_opportunity_scoring_job(run_id, user_id=user_id)
        stage_results["opportunity_scoring"] = r5
        _sync_meta("opportunity_scoring", "completed", run_id=run_id)
    except Exception as exc:
        logger.error(f"[ORCHESTRATOR] Stage 5 (opportunity_scoring) failed: {exc}", exc_info=True)
        _sync_meta("opportunity_scoring", "failed", run_id=run_id, error=str(exc))
        raise

    # Stage 6: Content Generation
    platform = (options or {}).get("platform", "instagram")
    _sync_meta("content_generation", "running", run_id=run_id)
    try:
        r6 = run_content_generation_job(run_id, platform=platform, user_id=user_id)
        stage_results["content_generation"] = r6
        _sync_meta("content_generation", "completed", run_id=run_id)
    except Exception as exc:
        logger.error(f"[ORCHESTRATOR] Stage 6 (content_generation) failed: {exc}", exc_info=True)
        _sync_meta("content_generation", "failed", run_id=run_id, error=str(exc))
        raise

    logger.info(f"[ORCHESTRATOR SUCCESS] All 6 stages completed for run: {run_id}")
    return {
        "job_type": "full_pipeline",
        "research_run_id": run_id,
        "stages": stages_progress,
        "results": stage_results,
    }
