"""
CatchyLab Agent 1 (Researcher) - FastAPI Application
----------------------------------------------------
Exposes async market research job execution, status polling, and database retrieval REST endpoints.

Security:
  - Legacy research endpoints except GET /health require the shared X-API-Key header.
    Provider-connection endpoints validate a per-user Supabase bearer token instead.
  - CORS origins are loaded from the ALLOWED_ORIGINS environment variable
    (comma-separated list).
  - POST /research is capped at MAX_RUNS_PER_DAY (default 5) per UTC day.
    The count is stored in the database so a process restart does not reset it.
"""

import json
import logging
import os
import secrets
from typing import Dict, Any, Optional, Literal
from contextlib import asynccontextmanager
from fastapi import Body, Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import redis
from rq import Queue
from rq.job import Job
from rq.exceptions import NoSuchJobError

from config import get_redis_connection
from database import (
    init_db,
    get_research_run,
    get_research_run_cost,
    get_research_run_logs,
    get_quota_status,
    get_consumer_insights,
    get_run_with_evidence,
    get_competitive_intelligence,
    get_creative_gaps,
    get_opportunity_scores,
    get_content_pieces,
    update_content_piece_status,
    resolve_content_claim,
    count_research_runs_today,
    count_stage_runs_today,
    count_serpapi_calls_today,
    record_stage_run,
    list_research_runs,
    get_full_research_run,
    get_demo_status_data,
    redact_secrets,
)
from jobs import (
    run_research_job,
    run_consumer_intelligence_job,
    run_competitive_intelligence_job,
    run_creative_gaps_job,
    run_opportunity_scoring_job,
    run_content_generation_job,
    run_full_pipeline_job,
)
from provider_connections import router as provider_connections_router
from local_voice import router as local_voice_router

logger = logging.getLogger("CatchyLabResearcher")

# ---------------------------------------------------------------------------
# Environment-driven configuration
# ---------------------------------------------------------------------------
_API_KEY: str = os.environ.get("CATCHYLAB_API_KEY", "")
_MAX_RUNS_PER_DAY: int = int(os.environ.get("MAX_RUNS_PER_DAY", "5"))
_MAX_STAGE_RUNS_PER_DAY: int = int(os.environ.get("MAX_STAGE_RUNS_PER_DAY", "20"))
_MAX_SERPAPI_PER_DAY: int = int(os.environ.get("MAX_SERPAPI_PER_DAY", "15"))
_DISABLE_DOCS: bool = os.environ.get("DISABLE_DOCS", "").lower() in ("1", "true", "yes")
_DEMO_MODE: bool = os.environ.get("DEMO_MODE", "0") in ("1", "true", "True", "yes", "YES")


def is_demo_mode() -> bool:
    """Returns True if DEMO_MODE is enabled via module flag or environment."""
    return bool(_DEMO_MODE or os.environ.get("DEMO_MODE", "0") in ("1", "true", "True", "yes", "YES"))

_DEFAULT_LOCAL_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]
_ALLOWED_ORIGINS_RAW: str = os.environ.get("ALLOWED_ORIGINS", "")
_CONFIGURED_ORIGINS: list[str] = (
    [o.strip() for o in _ALLOWED_ORIGINS_RAW.split(",") if o.strip()]
    if _ALLOWED_ORIGINS_RAW.strip()
    else []
)
_ALLOWED_ORIGINS: list[str] = (
    _CONFIGURED_ORIGINS if _CONFIGURED_ORIGINS else _DEFAULT_LOCAL_ORIGINS
)


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------
async def require_api_key(request: Request) -> None:
    """
    FastAPI dependency for legacy shared-key research routes.

    Uses secrets.compare_digest for constant-time comparison to guard against
    timing attacks.  Raises HTTP 401 when the key is absent or wrong.

    Applied to every route except GET /health.
    """
    if not _API_KEY:
        # If CATCHYLAB_API_KEY is not configured, reject all requests with a
        # clear server-configuration error so the problem is obvious.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="CATCHYLAB_API_KEY is not configured on this server.",
        )

    incoming_key = request.headers.get("X-API-Key", "")
    if not secrets.compare_digest(incoming_key, _API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )


# ---------------------------------------------------------------------------
# Stage quota helper
# ---------------------------------------------------------------------------
def _check_stage_quota(stage_name: str) -> None:
    """
    Raises HTTP 429 if the daily stage-run cap is exceeded.  For
    competitor-intelligence, also enforces MAX_SERPAPI_PER_DAY.

    Counts are read from the database so the check is restart-safe.
    Concurrent requests at the exact limit boundary may transiently
    exceed the cap by at most the number of simultaneous in-flight
    requests (best-effort, consistent with the existing research cap).
    """
    try:
        stage_today = count_stage_runs_today()
    except Exception as exc:
        logger.exception("Failed to query stage run count")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not verify the daily stage quota because the database is unavailable. Try again shortly.",
        ) from exc

    if stage_today >= _MAX_STAGE_RUNS_PER_DAY:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Daily stage-run limit reached: {stage_today}/{_MAX_STAGE_RUNS_PER_DAY} "
                "stage runs used today (UTC). Try again after UTC midnight or raise "
                "MAX_STAGE_RUNS_PER_DAY."
            ),
        )

    if stage_name == "competitor-intelligence":
        try:
            serp_today = count_serpapi_calls_today()
        except Exception as exc:
            logger.exception("Failed to query SerpAPI call count")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Could not verify the daily SerpAPI quota because the database is unavailable. Try again shortly.",
            ) from exc

        if serp_today >= _MAX_SERPAPI_PER_DAY:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Daily SerpAPI limit reached: {serp_today}/{_MAX_SERPAPI_PER_DAY} "
                    "competitor-intelligence calls used today (UTC). Try again after "
                    "UTC midnight or raise MAX_SERPAPI_PER_DAY."
                ),
            )


def _check_research_run_quota() -> None:
    """Fail closed when the database cannot verify the daily research limit."""
    try:
        runs_today = count_research_runs_today()
    except Exception as exc:
        logger.exception("Failed to query daily research run count")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not verify the daily research quota because the database is unavailable. Try again shortly.",
        ) from exc

    if runs_today >= _MAX_RUNS_PER_DAY:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Daily research limit reached: {runs_today}/{_MAX_RUNS_PER_DAY} runs used today (UTC). "
                "Please try again after UTC midnight or contact the administrator to raise MAX_RUNS_PER_DAY."
            ),
        )


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler initializing DB tables on startup."""
    try:
        init_db()
    except Exception as exc:
        logger.error(
            f"Database initialization failed on startup: {exc}. "
            "Backend is running, but database-dependent operations will fail until connection issues are resolved."
        )
    yield


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
app = FastAPI(
    title="CatchyLab Agent 1 - Market Research API",
    description=(
        "HTTP API for automated market research powered by Gemini API, SerpAPI, "
        "RQ + Upstash Redis, and PostgreSQL."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware — must be added before any route is evaluated
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*", "X-API-Key"],
)
app.include_router(provider_connections_router)
app.include_router(local_voice_router, dependencies=[Depends(require_api_key)])


@app.middleware("http")
async def _docs_gate(request: Request, call_next):
    """
    When DISABLE_DOCS=1 (or 'true'/'yes') is set, return 404 for /docs,
    /redoc, and /openapi.json so the API schema is not publicly accessible.
    The check runs at request time so it can be patched in tests via
    monkeypatch.setattr(app_module, '_DISABLE_DOCS', True).
    """
    if _DISABLE_DOCS and request.url.path in ("/docs", "/redoc", "/openapi.json"):
        return JSONResponse(
            {"detail": "API documentation is disabled on this server."},
            status_code=404,
        )
    return await call_next(request)


@app.middleware("http")
async def _redact_oauth_callback_query(request: Request, call_next):
    """Keep Google authorization codes and state out of application access logs."""
    if request.url.path == "/providers/gemini/oauth/callback":
        request.state.provider_oauth_query = request.scope.get("query_string", b"")
        request.scope["query_string"] = b""
    return await call_next(request)


@app.middleware("http")
async def _demo_mode_gate(request: Request, call_next):
    """
    When DEMO_MODE=1, reject every POST and PATCH request with 403 Forbidden.
    GET requests and /health continue working normally.
    """
    if is_demo_mode() and request.method.upper() in ("POST", "PATCH", "PUT", "DELETE"):
        return JSONResponse(
            {"detail": "Disabled in demo mode"},
            status_code=status.HTTP_403_FORBIDDEN,
        )
    return await call_next(request)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class ResearchBrief(BaseModel):
    """Pydantic model validating incoming research brief payload."""
    brand: str = Field(..., min_length=1, description="Brand name", json_schema_extra={"example": "Nike"})
    market: str = Field(..., min_length=1, description="Target country or market region", json_schema_extra={"example": "India"})
    audience: str = Field(..., min_length=1, description="Target demographic or audience segment", json_schema_extra={"example": "Gen Z"})
    objective: str = Field(..., min_length=1, description="Core market research objective", json_schema_extra={"example": "Find opportunities for sustainable running shoes"})
    language_preference: Literal["auto", "english", "hindi", "hinglish"] = Field(
        "auto",
        description="Preferred language for research synthesis and generated campaign content",
    )

    @field_validator("brand", "market", "audience", "objective")
    @classmethod
    def require_non_whitespace(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Field must contain non-whitespace characters.")
        return value


class ContentGenerationRequest(BaseModel):
    """Optional body for POST /research/{run_id}/content."""
    creative_gap_id: Optional[int] = Field(None, description="Specific creative gap ID to generate content for. Defaults to rank 1 opportunity.")
    platform: str = Field("instagram", description="Target social media platform", json_schema_extra={"example": "instagram"})


class ContentStatusUpdate(BaseModel):
    """Body for PATCH /content/{id}/status."""
    status: str = Field(..., description="New status: approved or rejected")
    reviewer_note: Optional[str] = Field(None, description="Optional reviewer note")


class ClaimResolution(BaseModel):
    """Body for PATCH /content/{id}/claims."""
    original_claim: str = Field(..., description="Original claim text (the part after 'VERIFIED CLAIM NEEDED:')")
    verified_replacement: str = Field(..., description="Verified replacement text")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> Dict[str, str]:
    """Health check endpoint — no authentication required."""
    return {"status": "ok"}


@app.get(
    "/demo/status",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Get Demo Status",
    description="Returns demo_mode status, total run count, and most recent run id and brand.",
    dependencies=[Depends(require_api_key)],
)
def get_demo_status() -> Dict[str, Any]:
    """
    GET /demo/status endpoint handler.
    Requires API key. Reads DB only.
    """
    logger.info("Received HTTP GET /demo/status request")
    try:
        data = get_demo_status_data()
        data["demo_mode"] = is_demo_mode()
        return data
    except Exception as exc:
        logger.error(f"Error fetching demo status: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch demo status: {redact_secrets(str(exc))}",
        )


@app.get(
    "/runs",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="List All Research Runs",
    description=(
        "Returns every research run ordered newest-first, each with its id, brand, "
        "market, audience, objective, created_at, and counts of evidence, active "
        "consumer_insights, creative_gaps, opportunity_scores, and content_pieces. "
        "Requires X-API-Key. Read-only — no DB writes."
    ),
    dependencies=[Depends(require_api_key)],
)
def list_runs_endpoint() -> Dict[str, Any]:
    """
    GET /runs endpoint handler.
    Returns all research runs with summary counts from stored data.
    """
    logger.info("Received HTTP GET /runs request")
    try:
        runs = list_research_runs()
        return {
            "total_runs": len(runs),
            "runs": runs,
        }
    except Exception as exc:
        logger.error(f"Error listing research runs: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list research runs: {redact_secrets(str(exc))}",
        )


@app.get(
    "/research/{run_id}/full",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Fetch Full Research Run",
    description=(
        "Returns a comprehensive view of one research run from stored data only: "
        "brief, evidence (with source title, URL, publication date, and source-record timestamp), consumer insights, competitors, "
        "competitive gaps, creative gaps, ranked opportunities (overall_score recomputed "
        "from stored component scores), content pieces (with claims_to_verify), persisted "
        "research methodology/evidence-quality context, and cost summary. Requires X-API-Key. "
        "Returns 404 for an unknown run_id."
    ),
    dependencies=[Depends(require_api_key)],
)
def fetch_full_research_run(run_id: str) -> Dict[str, Any]:
    """
    GET /research/{run_id}/full endpoint handler.
    Assembles all pipeline outputs for a run from stored data. No Gemini or SerpAPI calls.
    """
    logger.info(f"Received HTTP GET /research/{run_id}/full request")
    try:
        data = get_full_research_run(run_id)
        if data is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Research run '{run_id}' not found.",
            )
        return data
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error fetching full run '{run_id}': {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch full research run: {redact_secrets(str(exc))}",
        )


@app.post(
    "/research",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Enqueue Market Research Job",
    description=(
        "Accepts a research brief, enqueues an asynchronous RQ background job, and "
        "immediately returns a job_id.  Capped at MAX_RUNS_PER_DAY per UTC day."
    ),
    dependencies=[Depends(require_api_key)],
)
def create_research_report(
    brief: ResearchBrief,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    """
    POST /research endpoint handler.
    Enforces the daily run cap (DB-backed), then enqueues the research job.
    """
    logger.info(f"Received HTTP POST /research request for Brand={brief.brand}, Market={brief.market}")
    brief_dict = brief.model_dump()
    # Inject user_id from Supabase JWT (optional)
    if authorization and authorization.startswith("Bearer "):
        import asyncio
        from provider_auth import verify_supabase_access_token
        try:
            loop = asyncio.new_event_loop()
            user = loop.run_until_complete(verify_supabase_access_token(authorization[7:]))
            loop.close()
            brief_dict["user_id"] = user.get("id")
        except Exception:
            pass  # Non-authenticated runs get no user_id

    # 1. Daily rate-limit check (DB-backed — restart-safe)
    _check_research_run_quota()

    # 2. Connect to Redis with graceful error handling
    try:
        redis_conn = get_redis_connection()
        redis_conn.ping()
    except (redis.exceptions.ConnectionError, redis.exceptions.RedisError, Exception) as red_err:
        logger.error(f"Redis connection failed on POST /research: {red_err}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Redis service unreachable. Please ensure Redis worker/connection is available. Error: {redact_secrets(str(red_err))}",
        )

    # 3. Enqueue job on 'research_jobs' queue
    try:
        q = Queue("research_jobs", connection=redis_conn)
        job = q.enqueue(
            run_research_job,
            brief_dict,
            job_timeout="15m",
            meta={"job_type": "research"},
        )
        logger.info(f"Successfully enqueued RQ job_id: {job.id}")
        return {
            "job_id": job.id,
            "status": "queued",
        }
    except Exception as exc:
        logger.error(f"Error enqueuing research job: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enqueue research job: {redact_secrets(str(exc))}",
        )


@app.post(
    "/research/pipeline",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Enqueue Full 6-Stage Research Pipeline Job",
    description=(
        "Accepts a research brief, enqueues an asynchronous end-to-end RQ background job "
        "that sequentially executes all 6 stages (Research, Consumer Intelligence, "
        "Competitor Intelligence, Creative Gaps, Opportunity Engine, and Content Generation), "
        "and immediately returns a job_id."
    ),
    dependencies=[Depends(require_api_key)],
)
def create_full_pipeline_run(brief: ResearchBrief) -> Dict[str, Any]:
    """
    POST /research/pipeline endpoint handler.
    Enforces the daily run cap (DB-backed), then enqueues the end-to-end pipeline job.
    """
    logger.info(f"Received HTTP POST /research/pipeline request for Brand={brief.brand}, Market={brief.market}")
    brief_dict = brief.model_dump()

    # 1. Daily rate-limit check (DB-backed — restart-safe)
    _check_research_run_quota()

    # 2. Connect to Redis with graceful error handling
    try:
        redis_conn = get_redis_connection()
        redis_conn.ping()
    except (redis.exceptions.ConnectionError, redis.exceptions.RedisError, Exception) as red_err:
        logger.error(f"Redis connection failed on POST /research/pipeline: {red_err}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Redis service unreachable. Please ensure Redis worker/connection is available. Error: {redact_secrets(str(red_err))}",
        )

    # 3. Enqueue orchestrator job on 'research_jobs' queue
    try:
        q = Queue("research_jobs", connection=redis_conn)
        job = q.enqueue(
            run_full_pipeline_job,
            brief_dict,
            job_timeout="30m",
            meta={
                "job_type": "full_pipeline",
                "stages": {
                    "research": "pending",
                    "consumer_intelligence": "pending",
                    "competitive_intelligence": "pending",
                    "creative_gaps": "pending",
                    "opportunity_scoring": "pending",
                    "content_generation": "pending",
                },
                "current_stage": "research",
            },
        )
        logger.info(f"Successfully enqueued full pipeline RQ job_id: {job.id}")
        return {
            "job_id": job.id,
            "status": "queued",
            "job_type": "full_pipeline",
        }
    except Exception as exc:
        logger.error(f"Error enqueuing full pipeline job: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enqueue full pipeline job: {redact_secrets(str(exc))}",
        )


@app.get(
    "/research/status/{job_id}",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Check RQ Job Status",
    description="Checks the execution status of an enqueued research job by job_id.",
    dependencies=[Depends(require_api_key)],
)
def get_job_status(job_id: str) -> Dict[str, Any]:
    """
    GET /research/status/{job_id} endpoint handler.
    Checks RQ job status (queued, started, finished, failed). Returns research_run_id when finished.
    """
    logger.info(f"Received HTTP GET /research/status/{job_id} request")

    # 1. Connect to Redis
    try:
        redis_conn = get_redis_connection()
        redis_conn.ping()
    except (redis.exceptions.ConnectionError, redis.exceptions.RedisError, Exception) as red_err:
        logger.error(f"Redis connection failed on status check for job '{job_id}': {red_err}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Redis service unreachable. Please ensure Redis worker/connection is available. Error: {redact_secrets(str(red_err))}",
        )

    # 2. Fetch Job by ID
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except NoSuchJobError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )
    except Exception as exc:
        logger.error(f"Error fetching job '{job_id}': {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching job status: {redact_secrets(str(exc))}",
        )

    # 3. Determine job_type
    job_meta = getattr(job, "meta", {}) or {}
    job_type = job_meta.get("job_type")
    if not job_type:
        func_name = getattr(job, "func_name", "") or ""
        if "full_pipeline" in func_name:
            job_type = "full_pipeline"
        elif "content_generation" in func_name:
            job_type = "content_generation"
        elif "opportunity_scoring" in func_name:
            job_type = "opportunity_scoring"
        elif "creative_gaps" in func_name:
            job_type = "creative_gaps"
        elif "competitive_intelligence" in func_name:
            job_type = "competitive_intelligence"
        elif "consumer_intelligence" in func_name:
            job_type = "consumer_intelligence"
        elif "research" in func_name:
            job_type = "research"
        else:
            job_type = "unknown"

    # 4. Evaluate job status
    job_status = job.get_status()
    stages = job_meta.get("stages")
    current_stage = job_meta.get("current_stage")
    meta_run_id = job_meta.get("research_run_id")

    if job_status == "finished":
        result = job.result if isinstance(job.result, dict) else {}
        effective_job_type = result.get("job_type") or job_type
        effective_run_id = result.get("research_run_id") or meta_run_id
        effective_stages = result.get("stages") or stages
        return {
            "job_id": job_id,
            "job_type": effective_job_type,
            "status": "finished",
            "research_run_id": effective_run_id,
            "stages": effective_stages,
            "current_stage": current_stage,
            "result": result,
        }
    elif job_status == "failed":
        error_msg = redact_secrets(str(job.exc_info)) if job.exc_info else "Job failed during execution"
        return {
            "job_id": job_id,
            "job_type": job_type,
            "status": "failed",
            "error": error_msg,
            "error_stage": job_meta.get("error_stage"),
            "stages": stages,
            "current_stage": current_stage,
            "research_run_id": meta_run_id,
        }
    else:
        return {
            "job_id": job_id,
            "job_type": job_type,
            "status": job_status,
            "stages": stages,
            "current_stage": current_stage,
            "research_run_id": meta_run_id,
        }


@app.get(
    "/research/{run_id}",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Fetch Research Run by ID",
    description="Retrieves a persisted research run from PostgreSQL by ID along with its associated evidence and sources.",
    dependencies=[Depends(require_api_key)],
)
def fetch_research_report(run_id: str) -> Dict[str, Any]:
    """
    GET /research/{run_id} endpoint handler.
    Fetches research_runs row joined with associated evidence and sources. Returns HTTP 404 if not found.
    """
    logger.info(f"Received HTTP GET /research/{run_id} request")

    try:
        report_data = get_research_run(run_id)
        if not report_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Research run '{run_id}' not found.",
            )
        return report_data
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error fetching research run '{run_id}' from database: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch research report: {redact_secrets(str(exc))}",
        )


@app.get(
    "/research/{run_id}/cost",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Get Research Run Cost Breakdown (Hypothetical Paid-Tier)",
    description="Returns hypothetical paid-tier cost estimates for Gemini calls, call counts, and duration breakdown. All services are on free tiers — see GET /quota-status for real usage limits.",
    dependencies=[Depends(require_api_key)],
)
def fetch_research_cost(run_id: str) -> Dict[str, Any]:
    """
    GET /research/{run_id}/cost endpoint handler.
    Sums total estimated_cost_usd and duration_ms across all api_calls tied to research_run_id.
    """
    logger.info(f"Received HTTP GET /research/{run_id}/cost request")
    try:
        cost_data = get_research_run_cost(run_id)
        if not cost_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Research run '{run_id}' not found.",
            )
        return cost_data
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error fetching cost breakdown for research run '{run_id}': {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch cost breakdown: {redact_secrets(str(exc))}",
        )


@app.get(
    "/research/{run_id}/logs",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Get Research Run Execution Logs",
    description="Returns chronological raw list of api_calls rows logged during a research run.",
    dependencies=[Depends(require_api_key)],
)
def fetch_research_logs(run_id: str) -> Dict[str, Any]:
    """
    GET /research/{run_id}/logs endpoint handler.
    Returns chronological raw list of api_calls rows for that run_id.
    """
    logger.info(f"Received HTTP GET /research/{run_id}/logs request")
    try:
        logs_data = get_research_run_logs(run_id)
        if not logs_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Research run '{run_id}' not found.",
            )
        return logs_data
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error fetching execution logs for research run '{run_id}': {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch execution logs: {redact_secrets(str(exc))}",
        )


@app.get(
    "/quota-status",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Account-Wide Free-Tier Quota Status",
    description="Shows usage vs. quota limits across all free-tier services: Gemini API calls today, SerpAPI searches this month, Supabase DB row counts, and Upstash Redis command estimates.",
    dependencies=[Depends(require_api_key)],
)
def get_quota_status_endpoint() -> Dict[str, Any]:
    """
    GET /quota-status endpoint handler.
    Returns account-wide usage data across all free-tier services to answer
    'how close am I to breaking something?'
    """
    logger.info("Received HTTP GET /quota-status request")
    try:
        return get_quota_status()
    except Exception as exc:
        logger.error(f"Error fetching quota status: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch quota status: {redact_secrets(str(exc))}",
        )


@app.post(
    "/research/{run_id}/consumer-intelligence",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Enqueue Consumer Intelligence Synthesis Job",
    description="Asynchronously enqueues an RQ job for Agent 2 to synthesize evidence-grounded consumer segments, motivations, pain points, and behavioral patterns for a research run. Immediately returns job_id.",
    dependencies=[Depends(require_api_key)],
)
def synthesize_consumer_intelligence_endpoint(run_id: str) -> Dict[str, Any]:
    """
    POST /research/{run_id}/consumer-intelligence endpoint handler.
    Validates that the research run exists and contains evidence, enqueues an RQ job
    on 'research_jobs' queue, and returns job_id immediately.
    """
    logger.info(f"Received HTTP POST /research/{run_id}/consumer-intelligence request for run '{run_id}'")

    # 0. Daily stage-run quota check (before any DB lookup)
    _check_stage_quota("consumer-intelligence")

    # 1. Pre-validate that run exists and contains evidence
    run_data = get_run_with_evidence(run_id)
    if not run_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Research run '{run_id}' not found.",
        )
    evidence_items = run_data.get("evidence", [])
    if not evidence_items:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No evidence found for research run '{run_id}'. Cannot synthesize consumer intelligence.",
        )

    # 2. Connect to Redis
    try:
        redis_conn = get_redis_connection()
        redis_conn.ping()
    except (redis.exceptions.ConnectionError, redis.exceptions.RedisError, Exception) as red_err:
        logger.error(f"Redis connection failed on POST /research/{run_id}/consumer-intelligence: {red_err}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Redis service unreachable. Please ensure Redis worker/connection is available. Error: {redact_secrets(str(red_err))}",
        )

    # 3. Enqueue job on 'research_jobs' queue
    try:
        q = Queue("research_jobs", connection=redis_conn)
        job = q.enqueue(
            run_consumer_intelligence_job,
            run_id,
            job_timeout="10m",
            meta={"job_type": "consumer_intelligence", "research_run_id": run_id},
        )
        logger.info(f"Successfully enqueued consumer intelligence job_id: {job.id} for run '{run_id}'")
        record_stage_run("consumer-intelligence", run_id)
        return {
            "job_id": job.id,
            "status": "queued",
        }
    except Exception as exc:
        logger.error(f"Error enqueuing consumer intelligence job for run '{run_id}': {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enqueue consumer intelligence job: {redact_secrets(str(exc))}",
        )


@app.get(
    "/research/{run_id}/consumer-intelligence",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Fetch Saved Consumer Intelligence",
    description="Retrieves persisted consumer intelligence insights for a research run.",
    dependencies=[Depends(require_api_key)],
)
def get_consumer_intelligence_endpoint(run_id: str) -> Dict[str, Any]:
    """
    GET /research/{run_id}/consumer-intelligence endpoint handler.
    Fetches persisted consumer insights from the consumer_insights table.
    """
    logger.info(f"Received HTTP GET /research/{run_id}/consumer-intelligence request for run '{run_id}'")
    try:
        insights = get_consumer_insights(run_id)
        if insights is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Research run '{run_id}' not found.",
            )
        return {
            "research_run_id": run_id,
            "total_insights": len(insights),
            "insights": insights,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error fetching consumer intelligence for run '{run_id}': {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch consumer intelligence: {redact_secrets(str(exc))}",
        )


@app.post(
    "/research/{run_id}/competitor-intelligence",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Enqueue Competitor Intelligence Research Job",
    description="Asynchronously enqueues an RQ job for Agent 3 to execute targeted competitor research, evidence extraction, and synthesis. Immediately returns job_id.",
    dependencies=[Depends(require_api_key)],
)
def create_competitor_intelligence_job(run_id: str) -> Dict[str, Any]:
    """
    POST /research/{run_id}/competitor-intelligence endpoint handler.
    Pre-validates that research run exists and contains evidence, then enqueues an RQ job
    on 'research_jobs' queue.
    """
    logger.info(f"Received HTTP POST /research/{run_id}/competitor-intelligence request for run '{run_id}'")

    # 0. Daily stage-run and SerpAPI quota checks (before any DB lookup)
    _check_stage_quota("competitor-intelligence")

    # 1. Pre-validate run exists and has evidence
    run_data = get_run_with_evidence(run_id)
    if not run_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Research run '{run_id}' not found.",
        )
    evidence_items = run_data.get("evidence", [])
    if not evidence_items:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No evidence found for research run '{run_id}'. Cannot run competitor intelligence.",
        )

    # 2. Connect to Redis
    try:
        redis_conn = get_redis_connection()
        redis_conn.ping()
    except (redis.exceptions.ConnectionError, redis.exceptions.RedisError, Exception) as red_err:
        logger.error(f"Redis connection failed on POST /research/{run_id}/competitor-intelligence: {red_err}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Redis service unreachable. Please ensure Redis worker/connection is available. Error: {redact_secrets(str(red_err))}",
        )

    # 3. Enqueue job on 'research_jobs' queue
    try:
        q = Queue("research_jobs", connection=redis_conn)
        job = q.enqueue(
            run_competitive_intelligence_job,
            run_id,
            job_timeout="12m",
            meta={"job_type": "competitive_intelligence", "research_run_id": run_id},
        )
        logger.info(f"Successfully enqueued competitor intelligence job_id: {job.id} for run '{run_id}'")
        record_stage_run("competitor-intelligence", run_id)
        return {
            "job_id": job.id,
            "status": "queued",
        }
    except Exception as exc:
        logger.error(f"Error enqueuing competitor intelligence job for run '{run_id}': {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enqueue competitor intelligence job: {redact_secrets(str(exc))}",
        )


@app.get(
    "/research/{run_id}/competitor-intelligence",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Fetch Saved Competitor Intelligence",
    description="Retrieves persisted competitor intelligence (competitors, market structure, competitive gaps) for a research run.",
    dependencies=[Depends(require_api_key)],
)
def get_competitor_intelligence_endpoint(run_id: str) -> Dict[str, Any]:
    """
    GET /research/{run_id}/competitor-intelligence endpoint handler.
    Fetches persisted competitive intelligence from the competitive_intelligence table.
    """
    logger.info(f"Received HTTP GET /research/{run_id}/competitor-intelligence request for run '{run_id}'")
    try:
        comp_data = get_competitive_intelligence(run_id)
        if comp_data is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Research run '{run_id}' not found.",
            )
        return comp_data
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error fetching competitor intelligence for run '{run_id}': {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch competitor intelligence: {redact_secrets(str(exc))}",
        )


@app.post(
    "/research/{run_id}/creative-gaps",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Enqueue Creative Gap Detection Job",
    description="Asynchronously enqueues an RQ job for Agent 4 to cross-reference Agent 2 consumer intelligence and Agent 3 competitive intelligence to identify creative gaps. Returns job_id immediately.",
    dependencies=[Depends(require_api_key)],
)
def create_creative_gaps_job(run_id: str) -> Dict[str, Any]:
    """
    POST /research/{run_id}/creative-gaps endpoint handler.
    Validates that research run exists, and that BOTH Agent 2 (consumer_insights)
    and Agent 3 (competitive_intelligence) have completed for this run.
    Fails with 400 Bad Request naming any missing agent output before enqueueing.
    """
    logger.info(f"Received HTTP POST /research/{run_id}/creative-gaps request for run '{run_id}'")

    # 0. Daily stage-run quota check (before any DB lookup)
    _check_stage_quota("creative-gaps")

    # 1. Pre-validate run exists
    run_data = get_run_with_evidence(run_id)
    if not run_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Research run '{run_id}' not found.",
        )

    # 2. Pre-validate Agent 2 and Agent 3 outputs exist
    consumer_insights = get_consumer_insights(run_id)
    has_agent2 = consumer_insights is not None and len(consumer_insights) > 0

    comp_data = get_competitive_intelligence(run_id)
    competitors = comp_data.get("competitors", []) if comp_data else []
    has_agent3 = bool(competitors)

    if not has_agent2 and not has_agent3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Both Agent 2 (Consumer Intelligence) and Agent 3 (Competitive Intelligence) outputs are missing for run '{run_id}'. Both must be run before Agent 4 can execute.",
        )
    elif not has_agent2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Agent 2 output missing: Consumer intelligence has not been run for run '{run_id}'. Please run Agent 2 first.",
        )
    elif not has_agent3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Agent 3 output missing: Competitive intelligence has not been run for run '{run_id}'. Please run Agent 3 first.",
        )

    # 3. Connect to Redis
    try:
        redis_conn = get_redis_connection()
        redis_conn.ping()
    except (redis.exceptions.ConnectionError, redis.exceptions.RedisError, Exception) as red_err:
        logger.error(f"Redis connection failed on POST /research/{run_id}/creative-gaps: {red_err}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Redis service unreachable. Please ensure Redis worker/connection is available. Error: {redact_secrets(str(red_err))}",
        )

    # 4. Enqueue job on 'creative_gaps_jobs' queue
    try:
        q = Queue("creative_gaps_jobs", connection=redis_conn)
        job = q.enqueue(
            run_creative_gaps_job,
            run_id,
            job_timeout="10m",
            meta={"job_type": "creative_gaps", "research_run_id": run_id},
        )
        logger.info(f"Successfully enqueued creative gaps job_id: {job.id} for run '{run_id}'")
        record_stage_run("creative-gaps", run_id)
        return {
            "job_id": job.id,
            "status": "queued",
        }
    except Exception as exc:
        logger.error(f"Error enqueuing creative gaps job for run '{run_id}': {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enqueue creative gaps job: {redact_secrets(str(exc))}",
        )


@app.get(
    "/research/{run_id}/creative-gaps",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Fetch Saved Creative Gaps",
    description="Retrieves persisted creative gaps cross-referencing consumer desires and competitive absence for a research run.",
    dependencies=[Depends(require_api_key)],
)
def get_creative_gaps_endpoint(run_id: str) -> Dict[str, Any]:
    """
    GET /research/{run_id}/creative-gaps endpoint handler.
    Fetches persisted creative gaps from the creative_gaps table.
    """
    logger.info(f"Received HTTP GET /research/{run_id}/creative-gaps request for run '{run_id}'")
    try:
        gaps = get_creative_gaps(run_id)
        if gaps is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Research run '{run_id}' not found.",
            )
        return {
            "research_run_id": run_id,
            "total_gaps": len(gaps),
            "creative_gaps": gaps,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error fetching creative gaps for run '{run_id}': {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch creative gaps: {redact_secrets(str(exc))}",
        )


@app.post(
    "/research/{run_id}/opportunities",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Enqueue Opportunity Scoring & Ranking Job",
    description="Asynchronously enqueues an RQ job for the Opportunity Engine to score Agent 4's creative gaps across the 4 fixed criteria and rank them. Returns job_id immediately.",
    dependencies=[Depends(require_api_key)],
)
def create_opportunity_scoring_job(run_id: str) -> Dict[str, Any]:
    """
    POST /research/{run_id}/opportunities endpoint handler.
    Validates that research run exists and that Agent 4 (creative_gaps) has completed.
    Fails with 400 Bad Request naming missing Agent 4 output if not completed.
    """
    logger.info(f"Received HTTP POST /research/{run_id}/opportunities request for run '{run_id}'")

    # 0. Daily stage-run quota check (before any DB lookup)
    _check_stage_quota("opportunities")

    # 1. Pre-validate run exists
    run_data = get_run_with_evidence(run_id)
    if not run_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Research run '{run_id}' not found.",
        )

    # 2. Pre-validate Agent 4 creative gaps exist
    creative_gaps = get_creative_gaps(run_id)
    if not creative_gaps:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Agent 4 output missing: Creative gaps have not been run for run '{run_id}'. Please run Agent 4 first.",
        )

    # 3. Connect to Redis
    try:
        redis_conn = get_redis_connection()
        redis_conn.ping()
    except (redis.exceptions.ConnectionError, redis.exceptions.RedisError, Exception) as red_err:
        logger.error(f"Redis connection failed on POST /research/{run_id}/opportunities: {red_err}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Redis service unreachable. Please ensure Redis worker/connection is available. Error: {redact_secrets(str(red_err))}",
        )

    # 4. Enqueue job on 'opportunity_jobs' queue
    try:
        q = Queue("opportunity_jobs", connection=redis_conn)
        job = q.enqueue(
            run_opportunity_scoring_job,
            run_id,
            job_timeout="10m",
            meta={"job_type": "opportunity_scoring", "research_run_id": run_id},
        )
        logger.info(f"Successfully enqueued opportunity scoring job_id: {job.id} for run '{run_id}'")
        record_stage_run("opportunities", run_id)
        return {
            "job_id": job.id,
            "status": "queued",
        }
    except Exception as exc:
        logger.error(f"Error enqueuing opportunity scoring job for run '{run_id}': {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enqueue opportunity scoring job: {redact_secrets(str(exc))}",
        )


@app.get(
    "/research/{run_id}/opportunities",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Fetch Ranked Opportunities",
    description="Retrieves persisted ranked opportunities scored against the 4 fixed criteria for a research run.",
    dependencies=[Depends(require_api_key)],
)
def get_opportunities_endpoint(run_id: str) -> Dict[str, Any]:
    """
    GET /research/{run_id}/opportunities endpoint handler.
    Fetches persisted opportunity scores sorted by rank ascending.
    """
    logger.info(f"Received HTTP GET /research/{run_id}/opportunities request for run '{run_id}'")
    try:
        opps = get_opportunity_scores(run_id)
        if opps is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Research run '{run_id}' not found.",
            )
        return {
            "research_run_id": run_id,
            "total_opportunities": len(opps),
            "ranked_opportunities": opps,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error fetching opportunity scores for run '{run_id}': {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch opportunity scores: {redact_secrets(str(exc))}",
        )


@app.post(
    "/research/{run_id}/content",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Enqueue Content Generation Job",
    description="Asynchronously enqueues an RQ job to generate draft social content from the top-ranked opportunity. Requires completed Opportunity Engine output. Returns job_id immediately.",
    dependencies=[Depends(require_api_key)],
)
def create_content_generation_job(
    run_id: str,
    body: Optional[ContentGenerationRequest] = Body(default=None),
) -> Dict[str, Any]:
    """
    POST /research/{run_id}/content endpoint handler.
    Validates that run exists and Opportunity Engine has completed.
    Enqueues content generation job on 'content_jobs' queue.
    """
    logger.info(f"Received HTTP POST /research/{run_id}/content request")

    # 0. Daily stage-run quota check (before any DB lookup)
    _check_stage_quota("content")

    # 1. Pre-validate run exists
    run_data = get_run_with_evidence(run_id)
    if not run_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Research run '{run_id}' not found.",
        )

    # 2. Pre-validate Opportunity Engine has run
    opps = get_opportunity_scores(run_id)
    if not opps:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Opportunity Engine output missing for run '{run_id}'. Please run the Opportunity Engine first.",
        )

    creative_gap_id = body.creative_gap_id if body else None
    platform = body.platform if body and body.platform else "instagram"

    # 3. Connect to Redis
    try:
        redis_conn = get_redis_connection()
        redis_conn.ping()
    except (redis.exceptions.ConnectionError, redis.exceptions.RedisError, Exception) as red_err:
        logger.error(f"Redis connection failed on POST /research/{run_id}/content: {red_err}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Redis service unreachable. Error: {redact_secrets(str(red_err))}",
        )

    # 4. Enqueue job on 'content_jobs' queue
    try:
        q = Queue("content_jobs", connection=redis_conn)
        job = q.enqueue(
            run_content_generation_job,
            run_id,
            creative_gap_id,
            platform,
            job_timeout="10m",
            meta={"job_type": "content_generation", "research_run_id": run_id},
        )
        logger.info(f"Successfully enqueued content generation job_id: {job.id} for run '{run_id}'")
        record_stage_run("content", run_id)
        return {
            "job_id": job.id,
            "status": "queued",
        }
    except Exception as exc:
        logger.error(f"Error enqueuing content generation job for run '{run_id}': {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enqueue content generation job: {redact_secrets(str(exc))}",
        )


@app.get(
    "/research/{run_id}/content",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Fetch Generated Content Pieces",
    description="Retrieves persisted draft content pieces for a research run.",
    dependencies=[Depends(require_api_key)],
)
def get_content_pieces_endpoint(run_id: str) -> Dict[str, Any]:
    """
    GET /research/{run_id}/content endpoint handler.
    Fetches persisted content pieces from the content_pieces table.
    """
    logger.info(f"Received HTTP GET /research/{run_id}/content request")
    try:
        pieces = get_content_pieces(run_id)
        if pieces is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Research run '{run_id}' not found.",
            )
        return {
            "research_run_id": run_id,
            "total_pieces": len(pieces),
            "content_pieces": pieces,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error fetching content pieces for run '{run_id}': {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch content pieces: {redact_secrets(str(exc))}",
        )


@app.patch(
    "/content/{piece_id}/status",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Update Content Piece Status",
    description="Sets content piece status to approved or rejected. Groundwork for the feedback loop.",
    dependencies=[Depends(require_api_key)],
)
def update_content_status_endpoint(piece_id: int, body: ContentStatusUpdate) -> Dict[str, Any]:
    """
    PATCH /content/{piece_id}/status endpoint handler.
    Updates the status of a specific content piece.
    """
    logger.info(f"Received HTTP PATCH /content/{piece_id}/status with status='{body.status}'")

    if body.status not in ("approved", "rejected"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid status '{body.status}'. Must be 'approved' or 'rejected'.",
        )

    try:
        result = update_content_piece_status(piece_id, body.status, body.reviewer_note)
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Content piece #{piece_id} not found.",
            )
        return result
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error updating content piece #{piece_id} status: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update content piece status: {redact_secrets(str(exc))}",
        )


@app.patch(
    "/content/{piece_id}/claims",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Resolve Content Claim",
    description="Resolves a claim by replacing the placeholder with verified text. Removes from claims_to_verify.",
    dependencies=[Depends(require_api_key)],
)
def resolve_claim_endpoint(piece_id: int, body: ClaimResolution) -> Dict[str, Any]:
    """
    PATCH /content/{piece_id}/claims endpoint handler.
    Resolves a single claim by replacing the placeholder with verified text.
    """
    logger.info(f"Received HTTP PATCH /content/{piece_id}/claims for claim: '{body.original_claim}'")

    try:
        result = resolve_content_claim(piece_id, body.original_claim, body.verified_replacement)
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Content piece #{piece_id} not found.",
            )
        return result
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error resolving claim for content piece #{piece_id}: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to resolve claim: {redact_secrets(str(exc))}",
        )


# =============================================================================
# BYOK (Bring Your Own API Key) Credential Endpoints
# =============================================================================
# These endpoints let authenticated users submit, view, and delete their own
# Gemini and SerpAPI keys.  Keys are encrypted at rest (AES-256-GCM) and
# stored per-user in the user_api_credentials table.
# Auth: Supabase bearer token (Authorization: Bearer <access_token>)
# =============================================================================


class CredentialSubmission(BaseModel):
    """Request body for submitting an API key."""
    api_key: str = Field(..., min_length=10, description="Raw API key (never stored in plaintext).")


async def _require_supabase_user(authorization: Optional[str] = Header(default=None)) -> dict:
    """
    Dependency: validates Supabase JWT and returns the authenticated user dict.
    Raises HTTP 401 if the token is missing or invalid.
    """
    from provider_auth import verify_supabase_access_token
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header. Expected: Bearer <supabase_access_token>",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return await verify_supabase_access_token(authorization[7:])


@app.post(
    "/api/credentials/gemini",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Submit Gemini API Key",
    description=(
        "Validates and securely stores the user's Gemini API key (encrypted at rest). "
        "Requires a valid Supabase session token in the Authorization header. "
        "Returns only the masked key — the plaintext key is never returned."
    ),
)
async def submit_gemini_credential(
    body: CredentialSubmission,
    user: dict = Depends(_require_supabase_user),
) -> Dict[str, Any]:
    from services.credential_manager import store_credential, validate_gemini_key
    user_id = user["id"]
    logger.info(f"[BYOK] POST /api/credentials/gemini for user {user_id[:8]}...")
    # Validate before storing (attempt a live API call)
    is_valid = validate_gemini_key(body.api_key)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Gemini API key validation failed. Please check the key and try again.",
        )
    result = store_credential(user_id, "gemini", body.api_key)
    return {"success": True, "credential": result}


@app.post(
    "/api/credentials/serpapi",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Submit SerpAPI Key",
    description=(
        "Validates and securely stores the user's SerpAPI key (encrypted at rest). "
        "Requires a valid Supabase session token. Returns only the masked key."
    ),
)
async def submit_serpapi_credential(
    body: CredentialSubmission,
    user: dict = Depends(_require_supabase_user),
) -> Dict[str, Any]:
    from services.credential_manager import store_credential, validate_serpapi_key
    user_id = user["id"]
    logger.info(f"[BYOK] POST /api/credentials/serpapi for user {user_id[:8]}...")
    is_valid = validate_serpapi_key(body.api_key)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="SerpAPI key validation failed. Please check the key and try again.",
        )
    result = store_credential(user_id, "serpapi", body.api_key)
    return {"success": True, "credential": result}


@app.get(
    "/api/credentials/status",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Credential Status",
    description=(
        "Returns the masked configuration status for the authenticated user's Gemini and SerpAPI keys. "
        "Never returns plaintext keys."
    ),
)
async def get_credentials_status(
    user: dict = Depends(_require_supabase_user),
) -> Dict[str, Any]:
    from services.credential_manager import get_credential_status
    user_id = user["id"]
    return {"user_id": user_id, "credentials": get_credential_status(user_id)}


@app.delete(
    "/api/credentials/gemini",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Delete Gemini API Key",
    description="Deletes the authenticated user's stored Gemini API key.",
)
async def delete_gemini_credential(
    user: dict = Depends(_require_supabase_user),
) -> Dict[str, Any]:
    from services.credential_manager import delete_credential
    user_id = user["id"]
    deleted = delete_credential(user_id, "gemini")
    logger.info(f"[BYOK] DELETE /api/credentials/gemini for user {user_id[:8]}... deleted={deleted}")
    return {"success": True, "deleted": deleted, "provider": "gemini"}


@app.delete(
    "/api/credentials/serpapi",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Delete SerpAPI Key",
    description="Deletes the authenticated user's stored SerpAPI key.",
)
async def delete_serpapi_credential(
    user: dict = Depends(_require_supabase_user),
) -> Dict[str, Any]:
    from services.credential_manager import delete_credential
    user_id = user["id"]
    deleted = delete_credential(user_id, "serpapi")
    logger.info(f"[BYOK] DELETE /api/credentials/serpapi for user {user_id[:8]}... deleted={deleted}")
    return {"success": True, "deleted": deleted, "provider": "serpapi"}
