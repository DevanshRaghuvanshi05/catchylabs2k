"""
PostgreSQL Persistence Layer for CatchyLab Agent 1 (Researcher).
Uses SQLAlchemy models for research_runs, sources, and evidence tables, with pgvector support.
"""

import os
import re
import uuid
import requests
import time
import hashlib
import logging
import contextvars
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from sqlalchemy import (
    create_engine, Column, Integer, String, Text, Float, DateTime, ForeignKey, Index, func, text, JSON, Boolean
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, joinedload
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.compiler import compiles
from pgvector.sqlalchemy import Vector
from google import genai

# Allow SQLite compilation of PostgreSQL-specific ARRAY and Vector types
# so local development without PostgreSQL and in-memory test databases work smoothly.
@compiles(ARRAY, "sqlite")
def _compile_array_sqlite(element, compiler, **kw):
    return "TEXT"

@compiles(Vector, "sqlite")
def _compile_vector_sqlite(element, compiler, **kw):
    return "TEXT"

from config import DATABASE_URL, GEMINI_API_KEY, SERPAPI_API_KEY

logger = logging.getLogger("CatchyLabResearcher")


def redact_secrets(text: Optional[str]) -> Optional[str]:
    """
    Sanitizes secrets, credentials, API keys, and connection strings from strings.
    Replaces matched secrets with [REDACTED].
    """
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)

    s = text

    # Redact known configured secret values from environment/config if non-empty and len >= 6
    for env_var in (
        "GEMINI_API_KEY",
        "SERPAPI_API_KEY",
        "CATCHYLAB_API_KEY",
        "UPSTASH_REDIS_REST_TOKEN",
    ):
        val = os.environ.get(env_var, "").strip()
        if val and len(val) >= 6:
            s = s.replace(val, "[REDACTED]")

    # Redact credentials from database URLs / connection strings (e.g. postgresql://user:password@host:port/db)
    s = re.sub(r"://([^:@\s]+):([^@\s]+)@", r"://\1:[REDACTED]@", s)

    # Redact query param secrets (api_key=..., key=..., password=..., token=..., secret=...)
    s = re.sub(r"([?&](?:api_key|key|token|secret|password)=)[^&\s\"']+", r"\1[REDACTED]", s, flags=re.IGNORECASE)

    # Redact Bearer tokens
    s = re.sub(r"(Bearer\s+)[A-Za-z0-9_\-\.]+", r"\1[REDACTED]", s, flags=re.IGNORECASE)

    # Redact Gemini API key pattern (AIzaSy...)
    s = re.sub(r"AIza[0-9A-Za-z-_]{35}", "[REDACTED]", s)

    # Redact SerpAPI key pattern (64-char hex string)
    s = re.sub(r"\b[0-9a-fA-F]{64}\b", "[REDACTED]", s)

    return s


class StaleInputError(ValueError):
    """Raised when a downstream agent is given inputs that cite superseded consumer insights."""
    def __init__(self, entity_type: str, entity_id, stale_insight_ids: list):
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.stale_insight_ids = stale_insight_ids
        super().__init__(
            f"{entity_type} #{entity_id} cites superseded consumer insight IDs {stale_insight_ids}. "
            f"Re-run the upstream agent to refresh before building on stale inputs."
        )

# Create SQLAlchemy Engine and Session factory
# Guard against empty DATABASE_URL so the module can still be imported for
# the FastAPI app to start (health-check, Swagger docs) even when no DB
# connection string is configured yet.  Any route that actually touches the
# database will get a clear error at call time instead of a crash at import.
if DATABASE_URL:
    db_engine = create_engine(DATABASE_URL, pool_pre_ping=True)
else:
    # Use an in-memory SQLite as a placeholder so ORM model definitions still
    # bind and the module-level import chain succeeds.  Real DB calls will
    # fail clearly at session time when the app hits a missing connection.
    logger.warning(
        "DATABASE_URL is not set.  Using a transient in-memory SQLite placeholder.  "
        "Set DATABASE_URL to a valid PostgreSQL connection string for production."
    )
    db_engine = create_engine("sqlite:///:memory:", pool_pre_ping=False)
engine = db_engine  # keep backward-compat alias used elsewhere in this module
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)

Base = declarative_base()

# Thread-safe context tracking for research_run_id and pipeline stage
current_research_run_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("current_research_run_id", default=None)
current_stage: contextvars.ContextVar[str] = contextvars.ContextVar("current_stage", default="Unknown")

# PLACEHOLDER PRICING (USD per 1,000,000 tokens)
# NOTE: Verify actual current pricing before production deployment as rates may change over time.
GEMINI_PRICING = {
    "gemini-3.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-3.5-flash-lite": {"input": 0.0375, "output": 0.15},
    "gemini-embedding-001": {"input": 0.02, "output": 0.0},
}

# Free-tier quota limits (for reference in /quota-status)
GEMINI_FREE_TIER_RPD = 1500   # Requests per day (per model) — Gemini API free tier
SERPAPI_FREE_TIER_MONTHLY = 250  # Searches per month — SerpAPI free tier (verified via /account endpoint)
UPSTASH_FREE_TIER_DAILY_COMMANDS = 10000  # Commands per day — Upstash Redis free tier


class ResearchRun(Base):
    __tablename__ = "research_runs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), nullable=True, index=True)
    brand = Column(Text, nullable=False)
    audience = Column(Text, nullable=False)
    market = Column(Text, nullable=False)
    objective = Column(Text, nullable=False)
    language_preference = Column(String(16), nullable=False, default="auto")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    evidence_items = relationship("Evidence", back_populates="research_run", cascade="all, delete-orphan")
    api_calls = relationship("APICall", back_populates="research_run", cascade="all, delete-orphan")
    consumer_insights = relationship("ConsumerInsight", back_populates="research_run", cascade="all, delete-orphan")
    competitive_intelligence = relationship("CompetitiveIntelligence", back_populates="research_run", cascade="all, delete-orphan")
    creative_gaps = relationship("CreativeGap", back_populates="research_run", cascade="all, delete-orphan")
    opportunity_scores = relationship("OpportunityScore", back_populates="research_run")
    content_pieces = relationship("ContentPiece", back_populates="research_run")


class ResearchContextRecord(Base):
    __tablename__ = "research_contexts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    research_run_id = Column(
        String(36),
        ForeignKey("research_runs.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    context = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class Source(Base):
    __tablename__ = "sources"

    id = Column(Integer, primary_key=True, autoincrement=True)
    url = Column(Text, nullable=False)
    title = Column(Text, nullable=True)
    publisher = Column(Text, nullable=True)
    published_at = Column(DateTime(timezone=True), nullable=True)
    retrieved_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    content_hash = Column(Text, unique=True, index=True, nullable=False)

    evidence_items = relationship("Evidence", back_populates="source")


class Evidence(Base):
    __tablename__ = "evidence"

    id = Column(Integer, primary_key=True, autoincrement=True)
    research_run_id = Column(String(36), ForeignKey("research_runs.id", ondelete="CASCADE"), nullable=False)
    source_id = Column(Integer, ForeignKey("sources.id", ondelete="CASCADE"), nullable=False)
    claim = Column(Text, nullable=True)
    excerpt = Column(Text, nullable=True)
    topic = Column(Text, nullable=True)
    confidence = Column(Float, nullable=True)
    embedding = Column(Vector(3072), nullable=True)
    source_agent = Column(String(50), nullable=False, default="agent1")

    research_run = relationship("ResearchRun", back_populates="evidence_items")
    source = relationship("Source", back_populates="evidence_items")


class APICall(Base):
    __tablename__ = "api_calls"

    id = Column(Integer, primary_key=True, autoincrement=True)
    research_run_id = Column(String(36), ForeignKey("research_runs.id", ondelete="CASCADE"), nullable=True, index=True)
    provider = Column(Text, nullable=False)
    model = Column(Text, nullable=True)
    call_type = Column(Text, nullable=False)
    stage = Column(Text, nullable=False)
    status = Column(Text, nullable=False)
    duration_ms = Column(Integer, nullable=False)
    estimated_tokens = Column(Integer, nullable=True)
    estimated_cost_usd = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    error_message = Column(Text, nullable=True)

    research_run = relationship("ResearchRun", back_populates="api_calls")


class ConsumerInsight(Base):
    __tablename__ = "consumer_insights"

    id = Column(Integer, primary_key=True, autoincrement=True)
    research_run_id = Column(String(36), ForeignKey("research_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    insight_type = Column(Text, nullable=False)  # segment, motivation, pain_point, behavioral_pattern
    insight_text = Column(Text, nullable=False)
    supporting_evidence_ids = Column(ARRAY(Integer), nullable=False)
    superseded_at = Column(DateTime(timezone=True), nullable=True)  # Marked when superseded by newer insights
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    research_run = relationship("ResearchRun", back_populates="consumer_insights")


class CompetitiveIntelligence(Base):
    __tablename__ = "competitive_intelligence"

    id = Column(Integer, primary_key=True, autoincrement=True)
    research_run_id = Column(String(36), ForeignKey("research_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    competitors = Column(JSON, nullable=False)
    market_structure = Column(JSON, nullable=False)
    competitive_gaps = Column(JSON, nullable=False)
    superseded_at = Column(DateTime(timezone=True), nullable=True)  # Marked when superseded by newer intelligence
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    research_run = relationship("ResearchRun", back_populates="competitive_intelligence")


class CreativeGap(Base):
    __tablename__ = "creative_gaps"

    id = Column(Integer, primary_key=True, autoincrement=True)
    research_run_id = Column(String(36), ForeignKey("research_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    gap_title = Column(Text, nullable=False)
    description = Column(Text, nullable=False)
    supporting_consumer_insight_ids = Column(ARRAY(Integer), nullable=False)
    supporting_competitor_names = Column(ARRAY(Text), nullable=False)
    supporting_competitive_gap_ids = Column(ARRAY(Integer), nullable=False, default=list)
    confidence = Column(String(20), nullable=False)  # high, medium, low
    superseded_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    research_run = relationship("ResearchRun", back_populates="creative_gaps")
    opportunity_scores = relationship("OpportunityScore", back_populates="creative_gap")
    content_pieces = relationship("ContentPiece", back_populates="creative_gap")


class OpportunityScore(Base):
    __tablename__ = "opportunity_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    research_run_id = Column(String(36), ForeignKey("research_runs.id", ondelete="RESTRICT"), nullable=False, index=True)
    creative_gap_id = Column(Integer, ForeignKey("creative_gaps.id", ondelete="RESTRICT"), nullable=False, index=True)
    rank = Column(Integer, nullable=False)
    overall_score = Column(Float, nullable=False)
    audience_demand_score = Column(Integer, nullable=False)
    audience_demand_reasoning = Column(Text, nullable=False)
    competitive_whitespace_score = Column(Integer, nullable=False)
    competitive_whitespace_reasoning = Column(Text, nullable=False)
    brand_fit_score = Column(Integer, nullable=False)
    brand_fit_reasoning = Column(Text, nullable=False)
    feasibility_score = Column(Integer, nullable=False)
    feasibility_reasoning = Column(Text, nullable=False)
    superseded_at = Column(DateTime(timezone=True), nullable=True)  # Marked when superseded by newer scores
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    research_run = relationship("ResearchRun", back_populates="opportunity_scores")
    creative_gap = relationship("CreativeGap", back_populates="opportunity_scores")


class ContentPiece(Base):
    __tablename__ = "content_pieces"

    id = Column(Integer, primary_key=True, autoincrement=True)
    research_run_id = Column(String(36), ForeignKey("research_runs.id", ondelete="RESTRICT"), nullable=False, index=True)
    creative_gap_id = Column(Integer, ForeignKey("creative_gaps.id", ondelete="RESTRICT"), nullable=False, index=True)
    content_type = Column(String(30), nullable=False)  # reel_script / carousel / static_post
    platform = Column(String(30), nullable=False, default="instagram")
    title = Column(Text, nullable=False)
    hook = Column(Text, nullable=False)
    body = Column(JSON, nullable=False)  # scenes list / slides list / visual direction
    caption = Column(Text, nullable=False)
    hashtags = Column(ARRAY(Text), nullable=False, default=list)
    call_to_action = Column(Text, nullable=False)
    supporting_consumer_insight_ids = Column(ARRAY(Integer), nullable=False, default=list)
    supporting_competitor_names = Column(ARRAY(Text), nullable=False, default=list)
    claims_to_verify = Column(ARRAY(Text), nullable=True, default=list)  # List of [VERIFIED CLAIM NEEDED: ...] placeholders
    status = Column(String(20), nullable=False, default="draft")  # draft / approved / rejected
    reviewer_note = Column(Text, nullable=True)
    superseded_at = Column(DateTime(timezone=True), nullable=True)  # Marked when superseded by newer content
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    research_run = relationship("ResearchRun", back_populates="content_pieces")
    creative_gap = relationship("CreativeGap", back_populates="content_pieces")
    feedback = relationship("ContentFeedback", back_populates="content_piece")


class ContentFeedback(Base):
    __tablename__ = "content_feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    content_piece_id = Column(Integer, ForeignKey("content_pieces.id", ondelete="SET NULL"), nullable=True, index=True)
    # Snapshot columns to preserve feedback across regeneration
    snapshot_title = Column(Text, nullable=True)
    snapshot_hook = Column(Text, nullable=True)
    snapshot_body = Column(JSON, nullable=True)
    snapshot_claims_to_verify = Column(ARRAY(String), nullable=True)
    snapshot_creative_gap_id = Column(Integer, nullable=True)
    status = Column(String(20), nullable=False)  # approved / rejected
    note = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    content_piece = relationship("ContentPiece", back_populates="feedback")


class StageRun(Base):
    """
    Lightweight audit/quota table that records every time a POST stage endpoint
    (consumer-intelligence, competitor-intelligence, creative-gaps, opportunities,
    content) is successfully enqueued.  Used by count_stage_runs_today() to
    enforce MAX_STAGE_RUNS_PER_DAY without in-memory counters that reset on
    process restart.
    """
    __tablename__ = "stage_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    stage_name = Column(String(60), nullable=False, index=True)
    research_run_id = Column(String(36), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class ProviderConnection(Base):
    """Encrypted provider credential and model selection owned by one Supabase user."""
    __tablename__ = "provider_connections"
    __table_args__ = (
        Index("uq_provider_connections_owner_provider", "user_id", "provider", unique=True),
        Index(
            "uq_provider_connections_active_user",
            "user_id",
            unique=True,
            postgresql_where=text("is_active = TRUE"),
            sqlite_where=text("is_active = 1"),
        ),
    )

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), nullable=False, index=True)
    provider = Column(String(20), nullable=False)
    auth_method = Column(String(20), nullable=False)
    credential_ciphertext = Column(Text, nullable=False)
    status = Column(String(20), nullable=False, default="authorized")
    available_models = Column(JSON, nullable=False, default=list)
    selected_model = Column(Text, nullable=True)
    is_active = Column(Boolean, nullable=False, default=False)
    granted_scopes = Column(JSON, nullable=False, default=list)
    token_expires_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class ProviderOAuthState(Base):
    """Short-lived, single-use OAuth state and encrypted PKCE verifier."""
    __tablename__ = "provider_oauth_states"

    state_hash = Column(String(64), primary_key=True)
    user_id = Column(String(36), nullable=False, index=True)
    quota_project_id = Column(String(30), nullable=False)
    verifier_ciphertext = Column(Text, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class ProviderGenerationUsage(Base):
    """Rolling-window generation counters used to cap per-user provider spend."""
    __tablename__ = "provider_generation_usage"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(36), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)


class UserAPICredential(Base):
    """
    Secure encrypted storage for user-owned AI provider credentials (BYOK: Gemini, SerpAPI).
    Each user can configure at most one active credential per provider.
    Credentials are encrypted at rest using AES-256-GCM authenticated encryption.
    """
    __tablename__ = "user_api_credentials"
    __table_args__ = (
        Index("uq_user_api_credentials_user_provider", "user_id", "provider", unique=True),
    )

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), nullable=False, index=True)
    provider = Column(String(50), nullable=False)  # 'gemini', 'serpapi'
    encrypted_key = Column(Text, nullable=False)
    key_version = Column(String(20), nullable=False, default="v1")
    is_valid = Column(Boolean, nullable=False, default=True)
    masked_key = Column(String(100), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


RLS_TABLES = (
    "research_runs", "sources", "evidence", "api_calls",
    "consumer_insights", "competitive_intelligence", "creative_gaps",
    "opportunity_scores", "content_pieces", "content_feedback", "stage_runs",
    "research_contexts", "provider_connections", "provider_oauth_states",
    "provider_generation_usage", "user_api_credentials"
)


def init_db(db_engine=None):
    """Creates database tables if they do not exist, enables vector extension, and enforces RLS."""
    import json
    db_engine = db_engine or engine
    logger.info("Initializing database tables...")
    # 1. Enable pgvector extension on PostgreSQL
    if db_engine.dialect.name == "postgresql":
        with db_engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.commit()

    # 2. Create all tables via SQLAlchemy models
    Base.metadata.create_all(bind=db_engine)
    if db_engine.dialect.name == "postgresql":
        with db_engine.begin() as conn:
            conn.execute(text(
                "ALTER TABLE public.research_runs "
                "ADD COLUMN IF NOT EXISTS user_id VARCHAR(36)"
            ))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_research_runs_user_id ON public.research_runs (user_id)"
            ))
            conn.execute(text(
                "ALTER TABLE public.research_runs "
                "ADD COLUMN IF NOT EXISTS language_preference VARCHAR(16) NOT NULL DEFAULT 'auto'"
            ))
            conn.execute(text(
                "ALTER TABLE public.provider_connections "
                "ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT FALSE"
            ))
            conn.execute(text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_provider_connections_active_user "
                "ON public.provider_connections (user_id) WHERE is_active = TRUE"
            ))
            conn.execute(text(
                "ALTER TABLE public.provider_oauth_states "
                "ADD COLUMN IF NOT EXISTS quota_project_id VARCHAR(30)"
            ))

    # Keep one active competitive-intelligence report per research run while
    # retaining all superseded reports for history. This is the only table
    # whose contract permits a single active row per research_run_id.
    if db_engine.dialect.name == "postgresql":
        with db_engine.begin() as conn:
            conn.execute(text("""
                DO $$ DECLARE item RECORD; BEGIN
                    FOR item IN
                        SELECT conname FROM pg_constraint
                        WHERE conrelid = 'public.competitive_intelligence'::regclass
                          AND contype = 'u'
                          AND pg_get_constraintdef(oid) ILIKE 'UNIQUE (research_run_id)%'
                    LOOP
                        EXECUTE format(
                            'ALTER TABLE public.competitive_intelligence DROP CONSTRAINT %I',
                            item.conname
                        );
                    END LOOP;
                END $$;
            """))
            conn.execute(text(
                "DROP INDEX IF EXISTS public.ix_competitive_intelligence_research_run_id"
            ))
            conn.execute(text("DROP INDEX IF EXISTS public.ix_ci_research_run_active"))
            conn.execute(text(
                "CREATE UNIQUE INDEX ix_ci_research_run_active "
                "ON public.competitive_intelligence (research_run_id) "
                "WHERE superseded_at IS NULL"
            ))

        # 3. Ensure embedding and source_agent columns exist on evidence
        with db_engine.connect() as conn:
            conn.execute(text("ALTER TABLE evidence ADD COLUMN IF NOT EXISTS embedding vector(3072);"))
            conn.execute(text("ALTER TABLE evidence ADD COLUMN IF NOT EXISTS source_agent VARCHAR(50) DEFAULT 'agent1';"))
            conn.commit()

        # 4. Enable Row Level Security on all tables.
        # With zero permissive policies, Supabase's public REST API (anon/authenticated roles)
        # is fully locked out. The direct DATABASE_URL connection uses the postgres superuser
        # role which bypasses RLS entirely, so backend operations are unaffected.
        with db_engine.connect() as conn:
            for table_name in RLS_TABLES:
                conn.execute(text(f"ALTER TABLE public.{table_name} ENABLE ROW LEVEL SECURITY;"))
                logger.info(f"Row Level Security enabled on '{table_name}'.")

        # 5. Create trigger to validate supporting_consumer_insight_ids against consumer_insights(id)
        trigger_sql = """
        CREATE OR REPLACE FUNCTION check_supporting_consumer_insight_ids()
        RETURNS TRIGGER AS $$
        DECLARE
            invalid_ids integer[];
        BEGIN
            IF NEW.supporting_consumer_insight_ids IS NOT NULL AND array_length(NEW.supporting_consumer_insight_ids, 1) > 0 THEN
                SELECT array_agg(elem) INTO invalid_ids
                FROM unnest(NEW.supporting_consumer_insight_ids) AS elem
                WHERE elem NOT IN (SELECT id FROM consumer_insights WHERE research_run_id = NEW.research_run_id);
                
                IF invalid_ids IS NOT NULL AND array_length(invalid_ids, 1) > 0 THEN
                    RAISE EXCEPTION 'Foreign key violation: supporting_consumer_insight_ids % do not exist in consumer_insights for run %', invalid_ids, NEW.research_run_id;
                END IF;
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;

        DROP TRIGGER IF EXISTS trg_check_supporting_consumer_insight_ids ON creative_gaps;
        CREATE TRIGGER trg_check_supporting_consumer_insight_ids
        BEFORE INSERT OR UPDATE ON creative_gaps
        FOR EACH ROW EXECUTE FUNCTION check_supporting_consumer_insight_ids();
        """
        conn.execute(text(trigger_sql))

        # 6. Ensure existing competitive_gaps in competitive_intelligence have stable IDs
        try:
            ci_rows = conn.execute(text("SELECT id, competitive_gaps FROM competitive_intelligence")).fetchall()
            for ci_id, gaps_json in ci_rows:
                if gaps_json and isinstance(gaps_json, list):
                    updated = False
                    for idx, g in enumerate(gaps_json, 1):
                        if isinstance(g, dict) and "id" not in g:
                            g["id"] = idx
                            updated = True
                    if updated:
                        conn.execute(
                            text("UPDATE competitive_intelligence SET competitive_gaps = :gaps WHERE id = :id"),
                            {"gaps": json.dumps(gaps_json), "id": ci_id}
                        )
        except Exception as mig_err:
            logger.warning(f"Note on competitive_gaps ID migration: {mig_err}")

        # 7. Create trigger to validate content_pieces supporting_consumer_insight_ids
        content_trigger_sql = """
        CREATE OR REPLACE FUNCTION check_content_piece_insight_ids()
        RETURNS TRIGGER AS $$
        DECLARE
            invalid_ids integer[];
        BEGIN
            IF NEW.supporting_consumer_insight_ids IS NOT NULL AND array_length(NEW.supporting_consumer_insight_ids, 1) > 0 THEN
                SELECT array_agg(elem) INTO invalid_ids
                FROM unnest(NEW.supporting_consumer_insight_ids) AS elem
                WHERE elem NOT IN (SELECT id FROM consumer_insights WHERE research_run_id = NEW.research_run_id);
                
                IF invalid_ids IS NOT NULL AND array_length(invalid_ids, 1) > 0 THEN
                    RAISE EXCEPTION 'Foreign key violation: content_pieces.supporting_consumer_insight_ids % do not exist in consumer_insights for run %', invalid_ids, NEW.research_run_id;
                END IF;
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;

        DROP TRIGGER IF EXISTS trg_check_content_piece_insight_ids ON content_pieces;
        CREATE TRIGGER trg_check_content_piece_insight_ids
        BEFORE INSERT OR UPDATE ON content_pieces
        FOR EACH ROW EXECUTE FUNCTION check_content_piece_insight_ids();
        """
        conn.execute(text(content_trigger_sql))

        conn.commit()
    logger.info("Database tables, pgvector, triggers, and RLS initialized successfully.")


def calculate_gemini_cost(model: Optional[str], input_tokens: int, output_tokens: int) -> Optional[float]:
    """Calculates estimated USD cost for Gemini calls based on GEMINI_PRICING placeholder rates."""
    if not model:
        return None
    rates = GEMINI_PRICING.get(model)
    if not rates:
        rates = GEMINI_PRICING.get("gemini-3.5-flash", {"input": 0.075, "output": 0.30})
    
    input_cost = (input_tokens / 1_000_000.0) * rates.get("input", 0.0)
    output_cost = (output_tokens / 1_000_000.0) * rates.get("output", 0.0)
    return round(input_cost + output_cost, 8)


def extract_gemini_usage_and_cost(
    response: Any,
    model: Optional[str],
    input_text: Optional[str] = None
) -> tuple[Optional[int], Optional[float]]:
    """
    Extracts total tokens and calculates estimated cost from Gemini SDK response or input text fallback.
    Returns (estimated_tokens, estimated_cost_usd).
    """
    if response is not None:
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
            candidates_tokens = getattr(usage, "candidates_token_count", 0) or 0
            total_tokens = getattr(usage, "total_token_count", None)
            
            if total_tokens is not None and total_tokens > 0:
                est_tokens = total_tokens
                output_tokens = max(0, total_tokens - prompt_tokens)
            else:
                est_tokens = prompt_tokens + candidates_tokens
                output_tokens = candidates_tokens
                
            cost = calculate_gemini_cost(model, prompt_tokens, output_tokens)
            return est_tokens, cost

    # Fallback for embedding models or when usage_metadata is not returned by SDK
    if model == "gemini-embedding-001" or (model and "embedding" in model):
        if input_text and isinstance(input_text, str):
            est_tokens = max(1, len(input_text) // 4)
            cost = calculate_gemini_cost(model, est_tokens, 0)
            return est_tokens, cost

    return None, None


class APICallTracker:
    def __init__(
        self,
        provider: str,
        call_type: str,
        stage: Optional[str] = None,
        model: Optional[str] = None,
        research_run_id: Optional[str] = None,
        input_text: Optional[str] = None
    ):
        self.provider = provider
        self.call_type = call_type
        self.stage = stage
        self.model = model
        self.research_run_id = research_run_id
        self.input_text = input_text
        self.response = None
        self.status = "success"
        self.error_message = None
        self.start_time = None
        self.duration_ms = 0

    def set_response(self, response: Any):
        self.response = response

    def set_model(self, model: str):
        self.model = model

    def set_stage(self, stage: str):
        self.stage = stage

    def set_research_run_id(self, run_id: str):
        self.research_run_id = run_id


@contextmanager
def track_api_call(
    provider: str,
    call_type: str,
    stage: Optional[str] = None,
    model: Optional[str] = None,
    research_run_id: Optional[str] = None,
    input_text: Optional[str] = None
):
    """
    Lightweight context manager wrapping API calls to track timing, status,
    extract tokens & estimated costs, and insert a log row into api_calls.
    """
    tracker = APICallTracker(
        provider=provider,
        call_type=call_type,
        stage=stage,
        model=model,
        research_run_id=research_run_id,
        input_text=input_text
    )
    tracker.start_time = time.time()
    
    try:
        yield tracker
    except Exception as exc:
        err_str = str(exc)
        is_rate_limit = any(k in err_str.lower() for k in ["429", "resource_exhausted", "quota", "rate limit", "too many requests"])
        tracker.status = "rate_limited" if is_rate_limit else "failed"
        tracker.error_message = redact_secrets(err_str[:1000])
        raise
    finally:
        end_time = time.time()
        tracker.duration_ms = max(1, int((end_time - tracker.start_time) * 1000))
        
        eff_run_id = tracker.research_run_id or current_research_run_id.get(None)
        eff_stage = tracker.stage or current_stage.get("Unknown")
        
        est_tokens = None
        est_cost = None
        if tracker.provider == "gemini":
            est_tokens, est_cost = extract_gemini_usage_and_cost(tracker.response, tracker.model, tracker.input_text)
        # SerpAPI is on the free tier — no dollar cost, only quota tracking
            
        save_api_call_log(
            research_run_id=eff_run_id,
            provider=tracker.provider,
            model=tracker.model,
            call_type=tracker.call_type,
            stage=eff_stage,
            status=tracker.status,
            duration_ms=tracker.duration_ms,
            estimated_tokens=est_tokens,
            estimated_cost_usd=est_cost,
            error_message=tracker.error_message
        )


def save_api_call_log(
    research_run_id: Optional[str],
    provider: str,
    model: Optional[str],
    call_type: str,
    stage: str,
    status: str,
    duration_ms: int,
    estimated_tokens: Optional[int] = None,
    estimated_cost_usd: Optional[float] = None,
    error_message: Optional[str] = None
) -> None:
    """Inserts a record into api_calls table safely without crashing main thread."""
    session = SessionLocal()
    try:
        if research_run_id:
            exists = session.query(ResearchRun.id).filter_by(id=research_run_id).first()
            if not exists:
                research_run_id = None
                
        sanitized_error = redact_secrets(error_message)
        record = APICall(
            research_run_id=research_run_id,
            provider=provider,
            model=model,
            call_type=call_type,
            stage=stage,
            status=status,
            duration_ms=duration_ms,
            estimated_tokens=estimated_tokens,
            estimated_cost_usd=estimated_cost_usd,
            error_message=sanitized_error
        )
        session.add(record)
        session.commit()
    except Exception as exc:
        session.rollback()
        logger.error(f"Error persisting API call log: {redact_secrets(str(exc))}")
    finally:
        session.close()


def init_research_run(run_id: str, brief: dict) -> None:
    """Ensures research_runs record exists in DB at pipeline start before API calls log."""
    session = SessionLocal()
    try:
        existing = session.query(ResearchRun).filter_by(id=run_id).first()
        user_id = brief.get("user_id")
        if not existing:
            run_record = ResearchRun(
                id=run_id,
                user_id=user_id,
                brand=brief.get("brand", ""),
                audience=brief.get("audience", ""),
                market=brief.get("market", ""),
                objective=brief.get("objective", ""),
                language_preference=brief.get("language_preference", "auto"),
            )
            session.add(run_record)
            session.commit()
        elif user_id and not existing.user_id:
            existing.user_id = user_id
            session.commit()
    except Exception as exc:
        session.rollback()
        logger.error(f"Error initializing research run {run_id}: {exc}")
    finally:
        session.close()


def generate_embedding(client: genai.Client, text_content: str) -> Optional[list[float]]:
    """Generates text embedding vector using Gemini API (gemini-embedding-001)."""
    if not text_content or not text_content.strip():
        return None
    model_name = "gemini-embedding-001"
    try:
        with track_api_call(
            provider="gemini",
            call_type="embedding",
            model=model_name,
            input_text=text_content
        ) as tracker:
            resp = client.models.embed_content(
                model=model_name,
                contents=text_content.strip()
            )
            tracker.set_response(resp)
            if resp.embeddings and len(resp.embeddings) > 0:
                return resp.embeddings[0].values
    except Exception as e:
        logger.error(f"Error generating embedding via Gemini API: {e}")
    return None


def perform_semantic_recall(
    client: genai.Client,
    brief: dict,
    top_k: int = 5,
    max_distance: float = 0.25
) -> list[dict]:
    """
    Performs semantic search across past evidence in PostgreSQL using pgvector cosine distance.
    Returns list of matched evidence items tagged with 'reused_from_run_id'.
    """
    logger.info("=== STAGE 1.5: Semantic Recall ===")
    brief_text = (
        f"Brand: {brief.get('brand')} | Market: {brief.get('market')} | "
        f"Audience: {brief.get('audience')} | Objective: {brief.get('objective')}"
    )
    
    query_vec = generate_embedding(client, brief_text)
    if not query_vec:
        logger.warning("Failed to generate embedding for brief. Skipping semantic recall.")
        return []

    session = SessionLocal()
    reused_evidence = []
    try:
        # Distance calculation via pgvector cosine distance
        results = (
            session.query(
                Evidence,
                ResearchRun,
                Source,
                Evidence.embedding.cosine_distance(query_vec).label("distance")
            )
            .join(ResearchRun, Evidence.research_run_id == ResearchRun.id)
            .join(Source, Evidence.source_id == Source.id)
            .filter(Evidence.embedding.isnot(None))
            .filter(
                func.lower(func.trim(ResearchRun.brand))
                == str(brief.get("brand") or "").strip().lower(),
                func.lower(func.trim(ResearchRun.market))
                == str(brief.get("market") or "").strip().lower(),
            )
            .order_by("distance")
            .limit(top_k)
            .all()
        )

        for ev, run_rec, src, dist in results:
            if dist is None:
                continue
            similarity_score = round(1.0 - float(dist), 4)
            if dist <= max_distance:
                logger.info(
                    f"  [+] Match Found: Claim='{ev.claim[:70]}...' | "
                    f"Similarity Score: {similarity_score:.4f} (Distance: {dist:.4f}) | Run ID: {run_rec.id}"
                )
                reused_evidence.append({
                    "claim": ev.claim,
                    "source": {
                        "title": src.title if src else "",
                        "url": src.url if src else "",
                        "publisher": src.publisher if src else "",
                        "published_date": src.published_at.isoformat() if src and src.published_at else None
                    },
                    "evidence": ev.excerpt,
                    "market": run_rec.market,
                    "audience": run_rec.audience,
                    "topic": ev.topic,
                    "confidence": ev.confidence,
                    "reused_from_run_id": run_rec.id,
                    "similarity_score": similarity_score
                })
            else:
                logger.info(
                    f"  [-] Below Threshold Match Skipped: Claim='{ev.claim[:60]}...' | "
                    f"Similarity Score: {similarity_score:.4f} (Distance: {dist:.4f})"
                )

        logger.info(f"Semantic Recall Stage Complete -> Recalled {len(reused_evidence)} past evidence items.")
        return reused_evidence
    except Exception as e:
        logger.error(f"Error during semantic recall query: {e}", exc_info=True)
        return []
    finally:
        session.close()


def backfill_evidence_embeddings(client: genai.Client) -> int:
    """Utility function to backfill missing embeddings for existing evidence rows."""
    session = SessionLocal()
    count = 0
    try:
        rows = session.query(Evidence).filter(Evidence.embedding.is_(None)).all()
        if not rows:
            logger.info("No evidence rows require embedding backfill.")
            return 0
        logger.info(f"Backfilling embeddings for {len(rows)} evidence rows...")
        for row in rows:
            if row.claim:
                vec = generate_embedding(client, row.claim)
                if vec:
                    row.embedding = vec
                    count += 1
        session.commit()
        logger.info(f"Successfully backfilled embeddings for {count} evidence rows.")
        return count
    except Exception as e:
        session.rollback()
        logger.error(f"Error backfilling evidence embeddings: {e}")
        return count
    finally:
        session.close()


def compute_content_hash(url: str, title: str) -> str:
    """Generates unique content hash signature for source deduplication."""
    raw = f"{url.strip() if url else ''}:{title.strip() if title else ''}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def parse_published_date(date_val: Optional[str]) -> Optional[datetime]:
    """Parses published date string to datetime if valid."""
    if not date_val:
        return None
    try:
        return datetime.fromisoformat(date_val)
    except Exception:
        return None


def save_research_run(brief: dict, report_data: dict, client: Optional[genai.Client] = None, run_id: Optional[str] = None, user_id: Optional[str] = None) -> str:
    """
    Persists research brief, evidence sources, and evidence claims (with embeddings) to PostgreSQL.
    Returns assigned research_run_id.
    """
    session = SessionLocal()
    try:
        run_id = run_id or current_research_run_id.get(None) or str(uuid.uuid4())
        effective_user_id = user_id or brief.get("user_id")
        
        # 1. Create or update Research Run row
        run_record = session.query(ResearchRun).filter_by(id=run_id).first()
        if not run_record:
            run_record = ResearchRun(
                id=run_id,
                user_id=effective_user_id,
                brand=brief.get("brand", ""),
                audience=brief.get("audience", ""),
                market=brief.get("market", ""),
                objective=brief.get("objective", ""),
                language_preference=brief.get("language_preference", "auto"),
            )
            session.add(run_record)
            session.flush()
        else:
            if effective_user_id and not run_record.user_id:
                run_record.user_id = effective_user_id
                session.flush()
        
        evidence_items = report_data.get("evidence", [])
        research_context = report_data.get("research_context")
        context_record = None
        if isinstance(research_context, dict):
            context_record = (
                session.query(ResearchContextRecord)
                .filter_by(research_run_id=run_id)
                .first()
            )
        
        # Ensure client is available for embeddings if not passed
        if client is None:
            from services.credential_manager import get_gemini_client
            g_key = brief.get("gemini_key")
            if g_key:
                client = genai.Client(api_key=g_key)
            else:
                client = get_gemini_client(user_id=effective_user_id)

        pending_evidence = []
        for item in evidence_items:
            source_info = item.get("source", {}) if isinstance(item.get("source"), dict) else {}
            url = source_info.get("url", "")
            title = source_info.get("title", "")
            publisher = source_info.get("publisher", "")
            published_at = parse_published_date(source_info.get("published_date"))
            
            c_hash = compute_content_hash(url, title)
            
            # Deduplicate source by content_hash
            source_record = session.query(Source).filter_by(content_hash=c_hash).first()
            if not source_record:
                source_record = Source(
                    url=url,
                    title=title,
                    publisher=publisher,
                    published_at=published_at,
                    content_hash=c_hash
                )
                session.add(source_record)
                session.flush()
                
            claim_text = item.get("claim", "")
            emb_vector = None
            if client and claim_text and not item.get("reused_from_run_id"):
                emb_vector = generate_embedding(client, claim_text)

            # Create Evidence row
            evidence_record = Evidence(
                research_run_id=run_id,
                source_id=source_record.id,
                claim=claim_text,
                excerpt=item.get("evidence", ""),
                topic=item.get("topic", ""),
                confidence=item.get("confidence", 0.0),
                embedding=emb_vector
            )
            session.add(evidence_record)
            pending_evidence.append((evidence_record, source_record.id))

        session.flush()
        persisted_evidence = [
            {
                "evidence_id": evidence_record.id,
                "source_id": source_id,
            }
            for evidence_record, source_id in pending_evidence
        ]

        if isinstance(research_context, dict):
            registry = research_context.get("evidence_registry")
            if isinstance(registry, list):
                for entry, persisted in zip(registry, persisted_evidence):
                    if isinstance(entry, dict):
                        entry["evidence_id"] = persisted["evidence_id"]
                        entry["supporting_source_ids"] = [persisted["source_id"]]

            if context_record:
                context_record.context = research_context
            else:
                session.add(
                    ResearchContextRecord(
                        research_run_id=run_id,
                        context=research_context,
                    )
                )
            
        session.commit()
        logger.info(f"Successfully saved research run '{run_id}' to PostgreSQL ({len(evidence_items)} evidence items).")
        return run_id
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to save research run to PostgreSQL: {e}", exc_info=True)
        raise
    finally:
        session.close()


def get_research_run(run_id: str) -> Optional[dict]:
    """
    Fetches research run by id from PostgreSQL, joining evidence and sources.
    Returns dictionary matching original research report JSON structure or None.
    """
    session = SessionLocal()
    try:
        run_record = session.query(ResearchRun).filter(ResearchRun.id == run_id).first()
        if not run_record:
            return None
            
        evidence_records = (
            session.query(Evidence)
            .options(joinedload(Evidence.source))
            .filter(Evidence.research_run_id == run_id)
            .all()
        )
        
        evidence_list = []
        for ev in evidence_records:
            src = ev.source
            evidence_list.append({
                "claim": ev.claim,
                "source": {
                    "title": src.title if src else "",
                    "url": src.url if src else "",
                    "publisher": src.publisher if src else "",
                    "published_date": src.published_at.isoformat() if src and src.published_at else None
                },
                "evidence": ev.excerpt,
                "market": run_record.market,
                "audience": run_record.audience,
                "topic": ev.topic,
                "confidence": ev.confidence
            })
            
        return {
            "metadata": {
                "id": run_record.id,
                "user_id": run_record.user_id,
                "brand": run_record.brand,
                "market": run_record.market,
                "audience": run_record.audience,
                "objective": run_record.objective,
                "generated_at": run_record.created_at.isoformat() if run_record.created_at else None,
                "total_evidence_count": len(evidence_list)
            },
            "evidence": evidence_list
        }
    finally:
        session.close()


def get_research_run_cost(run_id: str) -> Optional[dict]:
    """
    Computes aggregated cost breakdown and duration metrics across all api_calls tied to research_run_id.
    Gemini costs are hypothetical paid-tier estimates for future planning — all services are currently on free tiers.
    Returns None if research_run_id does not exist in database.
    """
    session = SessionLocal()
    try:
        run_record = session.query(ResearchRun).filter_by(id=run_id).first()
        if not run_record:
            return None
            
        calls = session.query(APICall).filter_by(research_run_id=run_id).order_by(APICall.id.asc()).all()
        
        total_cost = 0.0
        total_duration = 0
        call_count = len(calls)
        serpapi_calls = 0
        
        breakdown_by_stage: Dict[str, Dict[str, Any]] = {}
        breakdown_by_call_type: Dict[str, Dict[str, Any]] = {}
        breakdown_by_model: Dict[str, Dict[str, Any]] = {}
        
        for c in calls:
            cost = c.estimated_cost_usd or 0.0
            dur = c.duration_ms or 0
            stg = c.stage or "Unknown"
            ctype = c.call_type or "Unknown"
            mdl = c.model or c.provider or "Unknown"
            
            if c.provider == "serpapi":
                serpapi_calls += 1
            
            total_cost += cost
            total_duration += dur
            
            # Stage breakdown
            if stg not in breakdown_by_stage:
                breakdown_by_stage[stg] = {"count": 0, "hypothetical_cost_usd": 0.0, "duration_ms": 0}
            breakdown_by_stage[stg]["count"] += 1
            breakdown_by_stage[stg]["hypothetical_cost_usd"] += cost
            breakdown_by_stage[stg]["duration_ms"] += dur
            
            # Call Type breakdown
            if ctype not in breakdown_by_call_type:
                breakdown_by_call_type[ctype] = {"count": 0, "hypothetical_cost_usd": 0.0, "duration_ms": 0}
            breakdown_by_call_type[ctype]["count"] += 1
            breakdown_by_call_type[ctype]["hypothetical_cost_usd"] += cost
            breakdown_by_call_type[ctype]["duration_ms"] += dur
            
            # Model breakdown
            if mdl not in breakdown_by_model:
                breakdown_by_model[mdl] = {"count": 0, "hypothetical_cost_usd": 0.0, "duration_ms": 0}
            breakdown_by_model[mdl]["count"] += 1
            breakdown_by_model[mdl]["hypothetical_cost_usd"] += cost
            breakdown_by_model[mdl]["duration_ms"] += dur
            
        # Round cost numbers for display
        for sub in (breakdown_by_stage, breakdown_by_call_type, breakdown_by_model):
            for key in sub:
                sub[key]["hypothetical_cost_usd"] = round(sub[key]["hypothetical_cost_usd"], 6)
                
        return {
            "note": (
                "All services are on FREE tiers — no actual charges. "
                "Gemini costs shown are hypothetical paid-tier estimates for future planning. "
                "SerpAPI, Supabase, and Redis have no dollar cost. "
                "See GET /quota-status for real usage vs. quota limits."
            ),
            "hypothetical_paid_tier_cost_usd": round(total_cost, 6),
            "total_duration_ms": total_duration,
            "call_count": call_count,
            "serpapi_calls_this_run": serpapi_calls,
            "breakdown_by_stage": breakdown_by_stage,
            "breakdown_by_call_type": breakdown_by_call_type,
            "breakdown_by_model": breakdown_by_model
        }
    finally:
        session.close()


def get_research_run_logs(run_id: str) -> Optional[dict]:
    """
    Returns raw list of api_calls rows for research_run_id in chronological order.
    Returns None if research_run_id does not exist in database.
    """
    session = SessionLocal()
    try:
        run_record = session.query(ResearchRun).filter_by(id=run_id).first()
        if not run_record:
            return None
            
        calls = session.query(APICall).filter_by(research_run_id=run_id).order_by(APICall.id.asc()).all()
        
        logs_list = []
        for c in calls:
            logs_list.append({
                "id": c.id,
                "research_run_id": c.research_run_id,
                "provider": c.provider,
                "model": c.model,
                "call_type": c.call_type,
                "stage": c.stage,
                "status": c.status,
                "duration_ms": c.duration_ms,
                "estimated_tokens": c.estimated_tokens,
                "estimated_cost_usd": c.estimated_cost_usd,
                "created_at": c.created_at.isoformat() if c.created_at else None,
                "error_message": redact_secrets(c.error_message)
            })
            
        return {
            "research_run_id": run_id,
            "total_calls": len(logs_list),
            "logs": logs_list
        }
    finally:
        session.close()


def count_research_runs_today() -> int:
    """
    Returns the number of research runs (rows in research_runs) created since
    UTC midnight today.  Used by POST /research to enforce MAX_RUNS_PER_DAY.
    Counting is done in the database so a process restart does not reset the
    counter.
    """
    now = datetime.now(timezone.utc)
    today_start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    session = SessionLocal()
    try:
        return (
            session.query(func.count(ResearchRun.id))
            .filter(ResearchRun.created_at >= today_start)
            .scalar()
            or 0
        )
    finally:
        session.close()


def record_stage_run(stage_name: str, research_run_id: Optional[str] = None) -> None:
    """
    Persist a StageRun row so the daily invocation count survives process
    restarts.  Called by each POST stage endpoint after a job is successfully
    enqueued.
    """
    session = SessionLocal()
    try:
        session.add(StageRun(stage_name=stage_name, research_run_id=research_run_id))
        session.commit()
    finally:
        session.close()


def count_stage_runs_today() -> int:
    """
    Returns the total number of expensive stage jobs enqueued since UTC midnight
    today (all stage types combined).  Used by POST stage endpoints to enforce
    MAX_STAGE_RUNS_PER_DAY.  DB-backed — restart-safe.
    """
    now = datetime.now(timezone.utc)
    today_start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    session = SessionLocal()
    try:
        return (
            session.query(func.count(StageRun.id))
            .filter(StageRun.created_at >= today_start)
            .scalar()
            or 0
        )
    finally:
        session.close()


def count_serpapi_calls_today() -> int:
    """
    Returns the number of api_calls rows with provider='serpapi' logged since
    UTC midnight today.  Used by POST /competitor-intelligence to enforce
    MAX_SERPAPI_PER_DAY.  DB-backed — restart-safe.
    """
    now = datetime.now(timezone.utc)
    today_start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    session = SessionLocal()
    try:
        return (
            session.query(func.count(APICall.id))
            .filter(
                APICall.provider == "serpapi",
                APICall.created_at >= today_start,
            )
            .scalar()
            or 0
        )
    finally:
        session.close()


def get_quota_status() -> dict:
    """
    Returns account-wide quota usage across all free-tier services:
    - Gemini: API calls today vs. free-tier daily limit (per model)
    - SerpAPI: Search calls this month vs. 100/month free limit
    - Supabase: Row counts across core tables as storage growth proxy
    - Upstash Redis: Estimated command usage vs. 10,000/day free limit
    """
    now = datetime.now(timezone.utc)
    today_start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    month_start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)

    session = SessionLocal()
    try:
        # --- Gemini: calls today by model ---
        gemini_today = (
            session.query(APICall.model, func.count(APICall.id))
            .filter(
                APICall.provider == "gemini",
                APICall.created_at >= today_start
            )
            .group_by(APICall.model)
            .all()
        )
        gemini_all_time = (
            session.query(func.count(APICall.id))
            .filter(APICall.provider == "gemini")
            .scalar() or 0
        )
        gemini_today_total = sum(count for _, count in gemini_today)
        gemini_by_model = {}
        for model_name, count in gemini_today:
            mdl = model_name or "unknown"
            gemini_by_model[mdl] = {
                "calls_today": count,
                "daily_limit": GEMINI_FREE_TIER_RPD
            }

        # --- SerpAPI: authoritative numbers from SerpAPI account API + local DB cross-reference ---
        # Local DB counts (for cross-reference)
        serpapi_all_time_total = (
            session.query(func.count(APICall.id))
            .filter(APICall.provider == "serpapi")
            .scalar() or 0
        )
        serpapi_all_time_success = (
            session.query(func.count(APICall.id))
            .filter(
                APICall.provider == "serpapi",
                APICall.status == "success"
            )
            .scalar() or 0
        )
        serpapi_this_calendar_month_total = (
            session.query(func.count(APICall.id))
            .filter(
                APICall.provider == "serpapi",
                APICall.created_at >= month_start
            )
            .scalar() or 0
        )
        serpapi_this_calendar_month_success = (
            session.query(func.count(APICall.id))
            .filter(
                APICall.provider == "serpapi",
                APICall.status == "success",
                APICall.created_at >= month_start
            )
            .scalar() or 0
        )

        # Live SerpAPI account API (authoritative billing source)
        serpapi_account = _get_serpapi_account_info()

        # --- Supabase DB: row counts across core tables ---
        row_counts = {}
        for table_cls in (ResearchRun, Source, Evidence, APICall, ConsumerInsight, CompetitiveIntelligence, CreativeGap, OpportunityScore, ContentPiece):
            row_counts[table_cls.__tablename__] = (
                session.query(func.count(table_cls.id)).scalar() or 0
            )
        total_rows = sum(row_counts.values())

        # --- Upstash Redis: try INFO stats, fall back to estimate ---
        redis_info = _get_redis_usage_estimate()

        serpapi_section = {
            "billing_source": serpapi_account,
            "local_db": {
                "note": "Local tracking from api_calls table. 'billable' = status='success' only; 'total' includes failed/timed-out calls.",
                "this_calendar_month_billable": serpapi_this_calendar_month_success,
                "this_calendar_month_total": serpapi_this_calendar_month_total,
                "all_time_billable": serpapi_all_time_success,
                "all_time_total": serpapi_all_time_total
            }
        }

        return {
            "generated_at": now.isoformat(),
            "note": "All services are on FREE tiers. This shows usage vs. quota limits.",
            "gemini": {
                "calls_today": gemini_today_total,
                "daily_limit_per_model": GEMINI_FREE_TIER_RPD,
                "by_model": gemini_by_model,
                "total_calls_all_time": gemini_all_time
            },
            "serpapi": serpapi_section,
            "supabase_db": {
                "note": "Row counts are a rough proxy for storage growth. Free tier: 500 MB database, 50,000 rows.",
                "total_rows": total_rows,
                "by_table": row_counts
            },
            "upstash_redis": redis_info
        }
    finally:
        session.close()


def _get_serpapi_account_info() -> dict:
    """
    Queries SerpAPI's /account endpoint for authoritative billing data.
    Returns the ground-truth usage numbers that SerpAPI uses for billing,
    including this_month_usage, searches_left, and plan limits.
    
    NOTE: SerpAPI's billing month does NOT align with calendar months —
    it resets on plan_renewal_date (e.g., the 11th), so their 'this_month_usage'
    may differ from a naive calendar-month count of our api_calls table.
    """
    try:
        resp = requests.get(
            "https://serpapi.com/account",
            params={"api_key": SERPAPI_API_KEY},
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            return {
                "source": "serpapi_account_api",
                "this_month_usage": data.get("this_month_usage"),
                "searches_per_month": data.get("searches_per_month"),
                "total_searches_left": data.get("total_searches_left"),
                "plan_renewal_date": data.get("plan_renewal_date"),
                "plan_name": data.get("plan_name"),
            }
        else:
            logger.warning(f"SerpAPI /account returned HTTP {resp.status_code}")
            return {
                "source": "unavailable",
                "error": f"HTTP {resp.status_code}",
                "fallback_monthly_limit": SERPAPI_FREE_TIER_MONTHLY
            }
    except Exception as e:
        logger.warning(f"Failed to reach SerpAPI /account endpoint: {e}")
        return {
            "source": "unavailable",
            "error": str(e),
            "fallback_monthly_limit": SERPAPI_FREE_TIER_MONTHLY
        }


def _get_redis_usage_estimate() -> dict:
    """
    Attempts to get Redis command usage from Upstash INFO stats.
    Falls back to DBSIZE if INFO is unavailable.
    """
    result = {
        "note": f"Free tier: {UPSTASH_FREE_TIER_DAILY_COMMANDS:,} commands/day.",
        "daily_limit": UPSTASH_FREE_TIER_DAILY_COMMANDS,
        "source": "unavailable"
    }
    try:
        from config import get_redis_connection
        redis_conn = get_redis_connection()

        # Attempt Redis INFO to get total_commands_processed
        try:
            info = redis_conn.info("stats")
            total_cmds = info.get("total_commands_processed")
            if total_cmds is not None:
                result["total_commands_processed"] = total_cmds
                result["source"] = "redis_info"
                return result
        except Exception:
            pass  # INFO may not be fully supported by Upstash REST

        # Attempt DBSIZE as a simpler metric
        try:
            db_size = redis_conn.dbsize()
            result["active_keys"] = db_size
            result["source"] = "redis_dbsize"
        except Exception:
            result["source"] = "unavailable"

    except Exception as exc:
        result["error"] = f"Could not connect to Redis: {str(exc)[:200]}"
        result["source"] = "error"

    return result


def get_run_with_evidence(run_id: str) -> Optional[dict]:
    """
    Fetches research run along with all its evidence items and sources.
    Returns structured dict with run metadata and list of evidence items (including evidence ID).
    """
    session = SessionLocal()
    try:
        run = session.query(ResearchRun).filter_by(id=run_id).first()
        if not run:
            return None
        
        evidence_records = (
            session.query(Evidence)
            .options(joinedload(Evidence.source))
            .filter_by(research_run_id=run_id)
            .order_by(Evidence.id)
            .all()
        )
        
        evidence_list = []
        for ev in evidence_records:
            evidence_list.append({
                "id": ev.id,
                "claim": ev.claim,
                "excerpt": ev.excerpt,
                "topic": ev.topic,
                "confidence": ev.confidence,
                "source_agent": getattr(ev, "source_agent", "agent1") or "agent1",
                "source": {
                    "id": ev.source_id,
                    "title": ev.source.title if ev.source else "",
                    "url": ev.source.url if ev.source else "",
                    "publisher": ev.source.publisher if ev.source else ""
                }
            })
            
        return {
            "id": run.id,
            "user_id": run.user_id,
            "brand": run.brand,
            "market": run.market,
            "audience": run.audience,
            "objective": run.objective,
            "language_preference": run.language_preference or "auto",
            "created_at": run.created_at.isoformat() if run.created_at else None,
            "evidence": evidence_list
        }
    finally:
        session.close()


def save_consumer_insights(run_id: str, insights: list[dict]) -> list[dict]:
    """
    Persists consumer intelligence insights for a research run.
    Marks previous insights as superseded instead of deleting.
    Returns list of saved insight dictionaries.
    """
    session = SessionLocal()
    try:
        run = session.query(ResearchRun).filter_by(id=run_id).first()
        if not run:
            raise ValueError(f"Research run '{run_id}' not found.")

        # Mark previous insights as superseded
        session.query(ConsumerInsight).filter_by(research_run_id=run_id).filter(ConsumerInsight.superseded_at.is_(None)).update({"superseded_at": datetime.now(timezone.utc)})
        
        saved_records = []
        for item in insights:
            record = ConsumerInsight(
                research_run_id=run_id,
                insight_type=item.get("insight_type", "").lower().strip(),
                insight_text=item.get("insight_text", "").strip(),
                supporting_evidence_ids=item.get("supporting_evidence_ids", [])
            )
            session.add(record)
            saved_records.append(record)
            
        session.commit()
        
        result = []
        for rec in saved_records:
            result.append({
                "id": rec.id,
                "research_run_id": rec.research_run_id,
                "insight_type": rec.insight_type,
                "insight_text": rec.insight_text,
                "supporting_evidence_ids": rec.supporting_evidence_ids,
                "created_at": rec.created_at.isoformat() if rec.created_at else None
            })
        logger.info(f"Successfully saved {len(result)} consumer insights for run '{run_id}'.")
        return result
    except Exception as exc:
        session.rollback()
        logger.error(f"Failed to save consumer insights for run '{run_id}': {exc}", exc_info=True)
        raise
    finally:
        session.close()


def get_consumer_insights(run_id: str, include_superseded: bool = False) -> Optional[list[dict]]:
    """
    Retrieves saved consumer insights for a given research_run_id.
    By default returns only non-superseded insights.
    Returns list of insight dicts or None if research_run does not exist.
    """
    session = SessionLocal()
    try:
        run_exists = session.query(ResearchRun.id).filter_by(id=run_id).first()
        if not run_exists:
            return None

        query = session.query(ConsumerInsight).filter_by(research_run_id=run_id)
        if not include_superseded:
            query = query.filter(ConsumerInsight.superseded_at.is_(None))

        records = query.order_by(ConsumerInsight.id).all()
        
        return [
            {
                "id": r.id,
                "research_run_id": r.research_run_id,
                "insight_type": r.insight_type,
                "insight_text": r.insight_text,
                "supporting_evidence_ids": r.supporting_evidence_ids,
                "created_at": r.created_at.isoformat() if r.created_at else None
            }
            for r in records
        ]
    finally:
        session.close()


def save_competitor_evidence(
    run_id: str,
    evidence_items: list[dict],
    client: Optional[genai.Client] = None
) -> list[int]:
    """
    Persists competitor-specific evidence items (extracted by Agent 3) into the evidence table
    with source_agent='agent3'. Deduplicates sources and generates vector embeddings.
    Returns list of assigned new evidence row IDs.
    """
    if not evidence_items:
        return []

    session = SessionLocal()
    new_ids = []
    try:
        run_record = session.query(ResearchRun).filter_by(id=run_id).first()
        if not run_record:
            raise ValueError(f"Research run '{run_id}' not found.")

        for item in evidence_items:
            source_info = item.get("source", {}) if isinstance(item.get("source"), dict) else {}
            url = source_info.get("url", "")
            title = source_info.get("title", "")
            publisher = source_info.get("publisher", "")
            published_at = parse_published_date(source_info.get("published_date"))

            c_hash = compute_content_hash(url, title)

            # Deduplicate source by content_hash
            source_record = session.query(Source).filter_by(content_hash=c_hash).first()
            if not source_record:
                source_record = Source(
                    url=url,
                    title=title,
                    publisher=publisher,
                    published_at=published_at,
                    content_hash=c_hash
                )
                session.add(source_record)
                session.flush()

            claim_text = item.get("claim", "")
            emb_vector = None
            if client and claim_text:
                emb_vector = generate_embedding(client, claim_text)

            evidence_record = Evidence(
                research_run_id=run_id,
                source_id=source_record.id,
                claim=claim_text,
                excerpt=item.get("evidence", ""),
                topic=item.get("topic", "competition"),
                confidence=item.get("confidence", 0.8),
                embedding=emb_vector,
                source_agent="agent3"
            )
            session.add(evidence_record)
            session.flush()
            new_ids.append(evidence_record.id)

        session.commit()
        logger.info(f"Successfully saved {len(new_ids)} competitor evidence items (source_agent='agent3') for run '{run_id}': IDs={new_ids}")
        return new_ids
    except Exception as exc:
        session.rollback()
        logger.error(f"Failed to save competitor evidence for run '{run_id}': {exc}", exc_info=True)
        raise
    finally:
        session.close()


def save_competitive_intelligence(run_id: str, data: dict) -> dict:
    """
    Persists structured competitive intelligence report for a research run.
    Marks previous record as superseded instead of deleting.
    Ensures every competitive gap has a stable integer ID.
    Returns saved competitive intelligence dictionary.
    """
    session = SessionLocal()
    try:
        run = session.query(ResearchRun).filter_by(id=run_id).first()
        if not run:
            raise ValueError(f"Research run '{run_id}' not found.")

        # Mark previous record as superseded
        session.query(CompetitiveIntelligence).filter_by(research_run_id=run_id).filter(CompetitiveIntelligence.superseded_at.is_(None)).update({"superseded_at": datetime.now(timezone.utc)})

        raw_gaps = data.get("competitive_gaps", [])
        stable_gaps = []
        for idx, g in enumerate(raw_gaps, 1):
            if isinstance(g, dict):
                gap_copy = dict(g)
                if "id" not in gap_copy or not isinstance(gap_copy["id"], int):
                    gap_copy["id"] = idx
                stable_gaps.append(gap_copy)

        record = CompetitiveIntelligence(
            research_run_id=run_id,
            competitors=data.get("competitors", []),
            market_structure=data.get("market_structure", {}),
            competitive_gaps=stable_gaps
        )
        session.add(record)
        session.commit()

        result = {
            "research_run_id": record.research_run_id,
            "competitors": record.competitors,
            "market_structure": record.market_structure,
            "competitive_gaps": record.competitive_gaps,
            "created_at": record.created_at.isoformat() if record.created_at else None
        }
        logger.info(f"Successfully saved competitive intelligence for run '{run_id}'.")
        return result
    except Exception as exc:
        session.rollback()
        logger.error(f"Failed to save competitive intelligence for run '{run_id}': {exc}", exc_info=True)
        raise
    finally:
        session.close()


def get_competitive_intelligence(run_id: str, include_superseded: bool = False) -> Optional[dict]:
    """
    Retrieves saved competitive intelligence report for a given research_run_id.
    By default returns only non-superseded records.
    Returns dict or None if research_run does not exist.
    Ensures every competitive gap has a stable integer ID.
    """
    session = SessionLocal()
    try:
        run_exists = session.query(ResearchRun.id).filter_by(id=run_id).first()
        if not run_exists:
            return None

        query = session.query(CompetitiveIntelligence).filter_by(research_run_id=run_id)
        if not include_superseded:
            query = query.filter(CompetitiveIntelligence.superseded_at.is_(None))

        # A run can have several historical reports. Select the current (most
        # recently saved) report deterministically, including when callers
        # request a lookup that can see superseded rows.
        record = query.order_by(CompetitiveIntelligence.id.desc()).first()
        if not record:
            return {
                "research_run_id": run_id,
                "competitors": [],
                "market_structure": {},
                "competitive_gaps": []
            }

        raw_gaps = record.competitive_gaps or []
        stable_gaps = []
        for idx, g in enumerate(raw_gaps, 1):
            if isinstance(g, dict):
                g_dict = dict(g)
                if "id" not in g_dict:
                    g_dict["id"] = idx
                stable_gaps.append(g_dict)
            else:
                stable_gaps.append(g)

        return {
            "research_run_id": record.research_run_id,
            "competitors": record.competitors,
            "market_structure": record.market_structure,
            "competitive_gaps": stable_gaps,
            "created_at": record.created_at.isoformat() if record.created_at else None
        }
    finally:
        session.close()


def save_creative_gaps(run_id: str, gaps: list[dict]) -> list[dict]:
    """
    Persists validated creative gaps for a research run.
    Validates that supporting_consumer_insight_ids actually exist in consumer_insights for run_id.
    Deletes prior records for run_id to ensure idempotency.
    Returns list of saved creative gap dictionaries.
    """
    session = SessionLocal()
    try:
        run = session.query(ResearchRun).filter_by(id=run_id).first()
        if not run:
            raise ValueError(f"Research run '{run_id}' not found.")

        # Query all valid consumer_insight IDs for this run
        existing_ci_ids = {
            r[0] for r in session.query(ConsumerInsight.id).filter_by(research_run_id=run_id).all()
        }

        # Validate that each gap's supporting_consumer_insight_ids exist
        for idx, gap in enumerate(gaps, 1):
            ci_ids = gap.get("supporting_consumer_insight_ids", [])
            if not ci_ids:
                raise ValueError(f"Creative gap #{idx} ('{gap.get('gap_title', '')}') has no supporting consumer insight IDs.")
            invalid = [cid for cid in ci_ids if cid not in existing_ci_ids]
            if invalid:
                raise ValueError(
                    f"Creative gap #{idx} cited invalid consumer insight IDs not belonging to run '{run_id}': {invalid}. "
                    f"Valid IDs: {sorted(list(existing_ci_ids))}"
                )

        # Mark existing creative gaps as superseded instead of deleting
        session.query(CreativeGap).filter_by(research_run_id=run_id).filter(
            CreativeGap.superseded_at.is_(None)
        ).update({"superseded_at": datetime.now(timezone.utc)})

        saved_records = []
        for item in gaps:
            record = CreativeGap(
                research_run_id=run_id,
                gap_title=item.get("gap_title", "").strip(),
                description=item.get("description", "").strip(),
                supporting_consumer_insight_ids=item.get("supporting_consumer_insight_ids", []),
                supporting_competitor_names=item.get("supporting_competitor_names", []),
                supporting_competitive_gap_ids=item.get("supporting_competitive_gap_ids", []),
                confidence=item.get("confidence", "high").strip().lower()
            )
            session.add(record)
            saved_records.append(record)

        session.commit()

        result = [
            {
                "id": rec.id,
                "research_run_id": rec.research_run_id,
                "gap_title": rec.gap_title,
                "description": rec.description,
                "supporting_consumer_insight_ids": rec.supporting_consumer_insight_ids,
                "supporting_competitor_names": rec.supporting_competitor_names,
                "supporting_competitive_gap_ids": rec.supporting_competitive_gap_ids,
                "confidence": rec.confidence,
                "created_at": rec.created_at.isoformat() if rec.created_at else None
            }
            for rec in saved_records
        ]
        logger.info(f"Successfully saved {len(result)} creative gaps for run '{run_id}'.")
        return result
    except Exception as exc:
        session.rollback()
        logger.error(f"Failed to save creative gaps for run '{run_id}': {exc}", exc_info=True)
        raise
    finally:
        session.close()


def _check_stale_insight_citations(session, run_id: str, cited_ids: list) -> list:
    """
    Checks whether any of the cited consumer insight IDs are superseded.
    Returns a list of superseded insight IDs (empty = not stale).
    """
    if not cited_ids:
        return []
    stale_ids = []
    for cid in cited_ids:
        ci = session.query(ConsumerInsight).filter_by(id=cid, research_run_id=run_id).first()
        if ci is not None and ci.superseded_at is not None:
            stale_ids.append(cid)
    return stale_ids


def get_creative_gaps(run_id: str, include_superseded: bool = False) -> Optional[list[dict]]:
    """
    Retrieves saved creative gaps for a given research_run_id.
    Returns list of gap dicts or None if research_run does not exist.
    By default, only returns non-superseded gaps (superseded_at is null).
    Each gap includes an 'is_stale' flag and 'stale_insight_ids' list
    when it cites consumer insight IDs that have been superseded.
    """
    session = SessionLocal()
    try:
        run_exists = session.query(ResearchRun.id).filter_by(id=run_id).first()
        if not run_exists:
            return None

        query = session.query(CreativeGap).filter_by(research_run_id=run_id)
        if not include_superseded:
            query = query.filter(CreativeGap.superseded_at.is_(None))
        records = query.order_by(CreativeGap.id).all()

        results = []
        for r in records:
            stale_ids = _check_stale_insight_citations(
                session, run_id, r.supporting_consumer_insight_ids or []
            )
            results.append({
                "id": r.id,
                "research_run_id": r.research_run_id,
                "gap_title": r.gap_title,
                "description": r.description,
                "supporting_consumer_insight_ids": r.supporting_consumer_insight_ids,
                "supporting_competitor_names": r.supporting_competitor_names,
                "supporting_competitive_gap_ids": r.supporting_competitive_gap_ids,
                "confidence": r.confidence,
                "is_stale": len(stale_ids) > 0,
                "stale_insight_ids": stale_ids,
                "superseded_at": r.superseded_at.isoformat() if r.superseded_at else None,
                "created_at": r.created_at.isoformat() if r.created_at else None
            })
        return results
    finally:
        session.close()


def save_opportunity_scores(run_id: str, scores: list[dict]) -> list[dict]:
    """
    Persists ranked opportunity scores for a research run.
    Validates that each creative_gap_id exists for this run.
    Marks previous scores as superseded instead of deleting.
    Returns list of saved opportunity score dicts.
    """
    session = SessionLocal()
    try:
        run = session.query(ResearchRun).filter_by(id=run_id).first()
        if not run:
            raise ValueError(f"Research run '{run_id}' not found.")

        # Only consider non-superseded gaps
        valid_gap_ids = {
            r[0] for r in session.query(CreativeGap.id).filter_by(research_run_id=run_id).filter(CreativeGap.superseded_at.is_(None)).all()
        }

        for idx, item in enumerate(scores, 1):
            gid = item.get("creative_gap_id")
            if gid not in valid_gap_ids:
                raise ValueError(
                    f"Opportunity score #{idx} references invalid creative_gap_id {gid} "
                    f"not belonging to run '{run_id}'. Valid IDs: {sorted(list(valid_gap_ids))}"
                )

        # Mark previous scores as superseded
        session.query(OpportunityScore).filter_by(research_run_id=run_id).filter(OpportunityScore.superseded_at.is_(None)).update({"superseded_at": datetime.now(timezone.utc)})

        saved_records = []
        for item in scores:
            record = OpportunityScore(
                research_run_id=run_id,
                creative_gap_id=item["creative_gap_id"],
                rank=item["rank"],
                overall_score=item["overall_score"],
                audience_demand_score=item["audience_demand_score"],
                audience_demand_reasoning=item["audience_demand_reasoning"].strip(),
                competitive_whitespace_score=item["competitive_whitespace_score"],
                competitive_whitespace_reasoning=item["competitive_whitespace_reasoning"].strip(),
                brand_fit_score=item["brand_fit_score"],
                brand_fit_reasoning=item["brand_fit_reasoning"].strip(),
                feasibility_score=item["feasibility_score"],
                feasibility_reasoning=item["feasibility_reasoning"].strip()
            )
            session.add(record)
            saved_records.append(record)

        session.commit()

        # Return only rows written by this call. Older score versions remain
        # stored with superseded_at set, but are available through the GET
        # helper only when include_superseded=True.
        records = sorted(saved_records, key=lambda rec: rec.rank)

        result = [
            {
                "id": rec.id,
                "research_run_id": rec.research_run_id,
                "creative_gap_id": rec.creative_gap_id,
                "rank": rec.rank,
                "overall_score": rec.overall_score,
                "scores": {
                    "audience_demand": rec.audience_demand_score,
                    "competitive_whitespace": rec.competitive_whitespace_score,
                    "brand_fit": rec.brand_fit_score,
                    "feasibility": rec.feasibility_score
                },
                "reasoning": {
                    "audience_demand": rec.audience_demand_reasoning,
                    "competitive_whitespace": rec.competitive_whitespace_reasoning,
                    "brand_fit": rec.brand_fit_reasoning,
                    "feasibility": rec.feasibility_reasoning
                },
                "created_at": rec.created_at.isoformat() if rec.created_at else None
            }
            for rec in records
        ]
        logger.info(f"Successfully saved {len(result)} opportunity scores for run '{run_id}'.")
        return result
    except Exception as exc:
        session.rollback()
        logger.error(f"Failed to save opportunity scores for run '{run_id}': {exc}", exc_info=True)
        raise
    finally:
        session.close()


def get_opportunity_scores(run_id: str, include_superseded: bool = False) -> Optional[list[dict]]:
    """
    Retrieves ranked opportunity scores for a given research_run_id, ordered by rank ascending.
    By default returns only non-superseded scores and non-superseded creative gaps.
    Joins creative gap details for rich context. Returns list of opportunity dicts or None if run does not exist.
    Each entry includes an 'is_stale' flag and 'stale_insight_ids' list
    when the underlying creative gap cites superseded consumer insight IDs.
    """
    session = SessionLocal()
    try:
        run_exists = session.query(ResearchRun.id).filter_by(id=run_id).first()
        if not run_exists:
            return None

        query = (
            session.query(OpportunityScore, CreativeGap)
            .join(CreativeGap, OpportunityScore.creative_gap_id == CreativeGap.id)
            .filter(OpportunityScore.research_run_id == run_id)
        )

        if not include_superseded:
            query = query.filter(OpportunityScore.superseded_at.is_(None)).filter(CreativeGap.superseded_at.is_(None))

        records = query.order_by(OpportunityScore.rank.asc()).all()

        results = []
        for opp, gap in records:
            stale_ids = _check_stale_insight_citations(
                session, run_id, gap.supporting_consumer_insight_ids or []
            )
            results.append({
                "id": opp.id,
                "research_run_id": opp.research_run_id,
                "creative_gap_id": opp.creative_gap_id,
                "rank": opp.rank,
                "overall_score": opp.overall_score,
                "scores": {
                    "audience_demand": opp.audience_demand_score,
                    "competitive_whitespace": opp.competitive_whitespace_score,
                    "brand_fit": opp.brand_fit_score,
                    "feasibility": opp.feasibility_score
                },
                "reasoning": {
                    "audience_demand": opp.audience_demand_reasoning,
                    "competitive_whitespace": opp.competitive_whitespace_reasoning,
                    "brand_fit": opp.brand_fit_reasoning,
                    "feasibility": opp.feasibility_reasoning
                },
                "creative_gap": {
                    "id": gap.id,
                    "gap_title": gap.gap_title,
                    "description": gap.description,
                    "supporting_consumer_insight_ids": gap.supporting_consumer_insight_ids,
                    "supporting_competitor_names": gap.supporting_competitor_names,
                    "supporting_competitive_gap_ids": gap.supporting_competitive_gap_ids,
                    "confidence": gap.confidence
                },
                "is_stale": len(stale_ids) > 0,
                "stale_insight_ids": stale_ids,
                "created_at": opp.created_at.isoformat() if opp.created_at else None
            })
        return results
    finally:
        session.close()


def _content_piece_to_dict(rec: ContentPiece) -> dict:
    """Converts a ContentPiece ORM record to a serializable dictionary."""
    return {
        "id": rec.id,
        "research_run_id": rec.research_run_id,
        "creative_gap_id": rec.creative_gap_id,
        "content_type": rec.content_type,
        "platform": rec.platform,
        "title": rec.title,
        "hook": rec.hook,
        "body": rec.body,
        "caption": rec.caption,
        "hashtags": rec.hashtags,
        "call_to_action": rec.call_to_action,
        "supporting_consumer_insight_ids": rec.supporting_consumer_insight_ids,
        "supporting_competitor_names": rec.supporting_competitor_names,
        "claims_to_verify": rec.claims_to_verify,
        "status": rec.status,
        "reviewer_note": rec.reviewer_note,
        "created_at": rec.created_at.isoformat() if rec.created_at else None
    }


def save_content_pieces(run_id: str, pieces: list[dict]) -> list[dict]:
    """
    Persists content pieces for a research run.
    Validates that supporting_consumer_insight_ids actually exist (only non-superseded).
    Marks previous pieces as superseded instead of deleting.
    Returns list of saved content piece dicts.
    """
    session = SessionLocal()
    try:
        run = session.query(ResearchRun).filter_by(id=run_id).first()
        if not run:
            raise ValueError(f"Research run '{run_id}' not found.")

        # Only consider non-superseded consumer insights to prevent stale citations
        existing_ci_ids = {
            r[0] for r in session.query(ConsumerInsight.id).filter_by(research_run_id=run_id).filter(ConsumerInsight.superseded_at.is_(None)).all()
        }

        # Determine which creative_gap_ids are being written and mark old pieces as superseded
        gap_ids_writing = {p.get("creative_gap_id") for p in pieces if p.get("creative_gap_id")}
        for gid in gap_ids_writing:
            # Check if gap is superseded
            gap = session.query(CreativeGap).filter_by(id=gid).first()
            if gap and gap.superseded_at is None:
                session.query(ContentPiece).filter_by(research_run_id=run_id, creative_gap_id=gid).filter(ContentPiece.superseded_at.is_(None)).update({"superseded_at": datetime.now(timezone.utc)})

        saved_records = []
        for idx, item in enumerate(pieces, 1):
            ci_ids = item.get("supporting_consumer_insight_ids", [])
            invalid = [cid for cid in ci_ids if cid not in existing_ci_ids]
            if invalid:
                raise ValueError(
                    f"Content piece #{idx} cited invalid consumer insight IDs not belonging to run '{run_id}': {invalid}. "
                    f"Valid IDs: {sorted(list(existing_ci_ids))}"
                )

            record = ContentPiece(
                research_run_id=run_id,
                creative_gap_id=item["creative_gap_id"],
                content_type=item.get("content_type", "").strip().lower(),
                platform=item.get("platform", "instagram").strip().lower(),
                title=item.get("title", "").strip(),
                hook=item.get("hook", "").strip(),
                body=item.get("body", {}),
                caption=item.get("caption", "").strip(),
                hashtags=item.get("hashtags", []),
                call_to_action=item.get("call_to_action", "").strip(),
                supporting_consumer_insight_ids=ci_ids,
                supporting_competitor_names=item.get("supporting_competitor_names", []),
                claims_to_verify=item.get("claims_to_verify", []),
                status="draft"
            )
            session.add(record)
            saved_records.append(record)

        session.commit()

        result = [_content_piece_to_dict(rec) for rec in saved_records]
        logger.info(f"Successfully saved {len(result)} content pieces for run '{run_id}'.")
        return result
    except Exception as exc:
        session.rollback()
        logger.error(f"Failed to save content pieces for run '{run_id}': {exc}", exc_info=True)
        raise
    finally:
        session.close()


def get_content_pieces(run_id: str, include_superseded: bool = False) -> Optional[list[dict]]:
    """
    Retrieves saved content pieces for a given research_run_id.
    By default returns only non-superseded pieces.
    Returns list of content piece dicts or None if research_run does not exist.
    """
    session = SessionLocal()
    try:
        run_exists = session.query(ResearchRun.id).filter_by(id=run_id).first()
        if not run_exists:
            return None

        query = session.query(ContentPiece).filter_by(research_run_id=run_id)
        if not include_superseded:
            query = query.filter(ContentPiece.superseded_at.is_(None))

        records = query.order_by(ContentPiece.id).all()

        results = []
        for r in records:
            d = _content_piece_to_dict(r)
            stale_ids = _check_stale_insight_citations(
                session, run_id, r.supporting_consumer_insight_ids or []
            )
            d["is_stale"] = len(stale_ids) > 0
            d["stale_insight_ids"] = stale_ids
            results.append(d)
        return results
    finally:
        session.close()


def update_content_piece_status(piece_id: int, new_status: str, reviewer_note: Optional[str] = None) -> Optional[dict]:
    """
    Updates the status of a content piece (approved / rejected).
    Creates a ContentFeedback record to persist feedback across re-runs.
    Returns the updated content piece dict or None if not found.
    """
    if new_status not in ("approved", "rejected"):
        raise ValueError(f"Invalid status '{new_status}'. Must be 'approved' or 'rejected'.")

    session = SessionLocal()
    try:
        record = session.query(ContentPiece).filter_by(id=piece_id).first()
        if not record:
            return None

        # Check if there are unresolved claims to verify
        if new_status == "approved" and record.claims_to_verify:
            raise ValueError(
                f"Cannot approve content piece #{piece_id} with {len(record.claims_to_verify)} unresolved claims. "
                f"Claims to verify: {record.claims_to_verify}. Resolve all claims before approval."
            )

        record.status = new_status
        if reviewer_note is not None:
            record.reviewer_note = reviewer_note

        # Create feedback record with snapshot data
        feedback = ContentFeedback(
            content_piece_id=piece_id,
            snapshot_title=record.title,
            snapshot_hook=record.hook,
            snapshot_body=record.body,
            snapshot_claims_to_verify=record.claims_to_verify,
            snapshot_creative_gap_id=record.creative_gap_id,
            status=new_status,
            note=reviewer_note
        )
        session.add(feedback)

        session.commit()

        result = _content_piece_to_dict(record)
        logger.info(f"Content piece #{piece_id} status updated to '{new_status}' with feedback record.")
        return result
    except Exception as exc:
        session.rollback()
        logger.error(f"Failed to update content piece #{piece_id} status: {exc}", exc_info=True)
        raise
    finally:
        session.close()


def resolve_content_claim(piece_id: int, original_claim: str, verified_replacement: str) -> Optional[dict]:
    """
    Resolves a single claim by replacing the placeholder with verified text.
    Removes the claim from claims_to_verify array.

    Strategy:
    1. Build the full placeholder: ``[VERIFIED CLAIM NEEDED: <original_claim>]``
    2. Replace every occurrence in body (JSON), caption, and hook with
       ``verified_replacement``.
    3. Remove the matching entry from claims_to_verify (exact full-placeholder
       match first; substring match as fallback — either way the entry is
       **deleted**, never converted to a [RESOLVED:...] residue that would
       keep the list non-empty and block approval).
    """
    session = SessionLocal()
    try:
        record = session.query(ContentPiece).filter_by(id=piece_id).first()
        if not record:
            return None

        if not record.claims_to_verify:
            raise ValueError(f"Content piece #{piece_id} has no claims to verify.")

        # Full canonical placeholder (with closing bracket)
        full_placeholder = f"[VERIFIED CLAIM NEEDED: {original_claim}]"
        # Prefix-only form used by older generator code (no closing bracket stored)
        prefix_placeholder = f"[VERIFIED CLAIM NEEDED: {original_claim}"

        def _replace(text: str) -> str:
            """Replace both full and prefix placeholder forms in a string."""
            text = text.replace(full_placeholder, verified_replacement)
            # Handle prefix form only if a closing bracket follows
            import re as _re
            text = _re.sub(
                r"\[VERIFIED CLAIM NEEDED:\s*" + _re.escape(original_claim) + r"[^\]]*\]",
                verified_replacement,
                text,
            )
            return text

        # Replace in body (JSON blob)
        body_str = json.dumps(record.body)
        new_body_str = _replace(body_str)
        if new_body_str != body_str:
            record.body = json.loads(new_body_str)

        # Replace in caption
        record.caption = _replace(record.caption)

        # Replace in hook
        if record.hook:
            record.hook = _replace(record.hook)

        # ----------------------------------------------------------------
        # Remove the claim from claims_to_verify
        # ----------------------------------------------------------------
        old_claims = list(record.claims_to_verify or [])
        new_claims = []
        removed = False
        for claim in old_claims:
            # Exact full match
            if not removed and (
                claim == full_placeholder
                or claim.startswith(prefix_placeholder)
                or original_claim in claim
            ):
                removed = True  # drop this entry — don't append
            else:
                new_claims.append(claim)

        if not removed:
            raise ValueError(
                f"Claim '{original_claim}' not found in claims_to_verify for "
                f"content piece #{piece_id}. Existing claims: {old_claims}"
            )

        # SQLAlchemy needs a new list object to detect the mutation on ARRAY
        from sqlalchemy.orm.attributes import flag_modified
        record.claims_to_verify = new_claims
        flag_modified(record, "claims_to_verify")

        session.commit()

        result = _content_piece_to_dict(record)
        logger.info(
            f"Content piece #{piece_id} claim resolved: "
            f"'{original_claim}' -> '{verified_replacement}'. "
            f"Remaining claims: {len(new_claims)}"
        )
        return result
    except Exception as exc:
        session.rollback()
        logger.error(
            f"Failed to resolve claim for content piece #{piece_id}: {exc}",
            exc_info=True,
        )
        raise
    finally:
        session.close()


def find_diagnostic_report_for_run(run_id: str, brand: Optional[str] = None) -> Optional[dict]:
    """
    Looks for an existing research report JSON file matching the exact run_id.
    Returns the report metadata dict if found, else None.
    Safe, read-only inspection.
    """
    if not run_id:
        return None

    import glob
    try:
        report_files = glob.glob("research_report_*.json")
        for fpath in report_files:
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    meta = data.get("metadata", {})
                    if isinstance(meta, dict) and meta.get("research_run_id") == run_id:
                        return meta
            except Exception:
                continue
    except Exception:
        pass
    return None


def get_run_status_and_diagnostics(
    run_id: str,
    brand: Optional[str] = None,
    evidence_count: int = 0
) -> tuple[str, Optional[str], Optional[dict]]:
    """
    Determines status ("completed", "failed", or "unknown"), failure_reason, and diagnostic_report
    without database schema changes.
    """
    if evidence_count > 0:
        return "completed", None, None

    diag = find_diagnostic_report_for_run(run_id, brand)
    if diag and (diag.get("status") == "failed" or diag.get("error") or diag.get("total_evidence_count") == 0):
        raw_reason = diag.get("error") or "Research extraction produced zero evidence items."
        return "failed", redact_secrets(raw_reason), diag

    # A run with zero evidence and no trustworthy artifact establishing success or failure must return status="unknown"
    return "unknown", "No evidence recorded and no diagnostic report found.", None


def list_research_runs(user_id: Optional[str] = None) -> list[dict]:
    """
    Returns a summary list of all research runs ordered newest-first.
    Each entry includes: id, user_id, brand, market, audience, objective, created_at,
    and counts of evidence, consumer_insights (active), creative_gaps (active),
    opportunity_scores (active), and content_pieces (active).
    If user_id is provided, filters runs owned by that user.
    Uses SELECT-only queries; no DML.
    """
    session = SessionLocal()
    try:
        query = session.query(ResearchRun)
        if user_id:
            query = query.filter_by(user_id=user_id)
        runs = (
            query
            .order_by(ResearchRun.created_at.desc())
            .all()
        )

        results = []
        for run in runs:
            evidence_count = (
                session.query(func.count(Evidence.id))
                .filter(Evidence.research_run_id == run.id)
                .scalar() or 0
            )
            insight_count = (
                session.query(func.count(ConsumerInsight.id))
                .filter(
                    ConsumerInsight.research_run_id == run.id,
                    ConsumerInsight.superseded_at.is_(None),
                )
                .scalar() or 0
            )
            gap_count = (
                session.query(func.count(CreativeGap.id))
                .filter(
                    CreativeGap.research_run_id == run.id,
                    CreativeGap.superseded_at.is_(None),
                )
                .scalar() or 0
            )
            opp_count = (
                session.query(func.count(OpportunityScore.id))
                .filter(
                    OpportunityScore.research_run_id == run.id,
                    OpportunityScore.superseded_at.is_(None),
                )
                .scalar() or 0
            )
            content_count = (
                session.query(func.count(ContentPiece.id))
                .filter(
                    ContentPiece.research_run_id == run.id,
                    ContentPiece.superseded_at.is_(None),
                )
                .scalar() or 0
            )

            status, failure_reason, _ = get_run_status_and_diagnostics(
                run_id=run.id,
                brand=run.brand,
                evidence_count=evidence_count
            )

            results.append({
                "id": run.id,
                "user_id": run.user_id,
                "brand": run.brand,
                "market": run.market,
                "audience": run.audience,
                "objective": run.objective,
                "language_preference": run.language_preference or "auto",
                "created_at": run.created_at.isoformat() if run.created_at else None,
                "status": status,
                "failure_reason": failure_reason,
                "counts": {
                    "evidence": evidence_count,
                    "consumer_insights": insight_count,
                    "creative_gaps": gap_count,
                    "opportunity_scores": opp_count,
                    "content_pieces": content_count,
                },
            })
        return results
    finally:
        session.close()


def get_research_run_owner(run_id: str) -> Optional[str]:
    """Returns the user_id owner of research_run_id, or None if unowned/not found."""
    session = SessionLocal()
    try:
        run = session.query(ResearchRun.user_id).filter_by(id=run_id).first()
        return run[0] if run else None
    finally:
        session.close()


def get_demo_status_data() -> dict:
    """
    Reads the database to return summary statistics for demo status:
    - number_of_runs: total count of research runs
    - id: run id of the most recent research run (or None)
    - brand: brand of the most recent research run (or None)
    Uses SELECT-only queries.
    """
    session = SessionLocal()
    try:
        total_runs = session.query(func.count(ResearchRun.id)).scalar() or 0
        recent = (
            session.query(ResearchRun)
            .order_by(ResearchRun.created_at.desc(), ResearchRun.id.desc())
            .first()
        )
        return {
            "number_of_runs": total_runs,
            "id": recent.id if recent else None,
            "brand": recent.brand if recent else None,
        }
    finally:
        session.close()


def get_full_research_run(run_id: str) -> Optional[dict]:
    """
    Returns a comprehensive view of a single research run from stored data only.

    Includes:
    - brief (run metadata)
    - evidence (with source title and URL)
    - consumer_insights (active only)
    - competitors + competitive_gaps (from active CI report)
    - creative_gaps (active only)
    - ranked_opportunities (with scores recomputed from stored component scores)
    - content_pieces (active only, with claims_to_verify)
    - cost_summary (aggregated from api_calls)

    Returns None if run_id does not exist.  Uses SELECT-only queries.
    """
    session = SessionLocal()
    try:
        run = session.query(ResearchRun).filter_by(id=run_id).first()
        if not run:
            return None

        # --- Brief ---
        brief = {
            "id": run.id,
            "user_id": run.user_id,
            "brand": run.brand,
            "market": run.market,
            "audience": run.audience,
            "objective": run.objective,
            "language_preference": run.language_preference or "auto",
            "created_at": run.created_at.isoformat() if run.created_at else None,
        }

        # --- Evidence (with source title + URL) ---
        ev_records = (
            session.query(Evidence)
            .options(joinedload(Evidence.source))
            .filter(Evidence.research_run_id == run_id)
            .order_by(Evidence.id)
            .all()
        )
        evidence = [
            {
                "id": ev.id,
                "claim": ev.claim,
                "excerpt": ev.excerpt,
                "topic": ev.topic,
                "confidence": ev.confidence,
                "source_agent": getattr(ev, "source_agent", "agent1") or "agent1",
                "source": {
                    "id": ev.source_id,
                    "title": ev.source.title if ev.source else "",
                    "url": ev.source.url if ev.source else "",
                    "publisher": ev.source.publisher if ev.source else "",
                    "published_at": (
                        ev.source.published_at.isoformat()
                        if ev.source and ev.source.published_at
                        else None
                    ),
                    "retrieved_at": (
                        ev.source.retrieved_at.isoformat()
                        if ev.source and ev.source.retrieved_at
                        else None
                    ),
                },
            }
            for ev in ev_records
        ]

        # --- Consumer Insights (active only) ---
        insight_records = (
            session.query(ConsumerInsight)
            .filter(
                ConsumerInsight.research_run_id == run_id,
                ConsumerInsight.superseded_at.is_(None),
            )
            .order_by(ConsumerInsight.id)
            .all()
        )
        consumer_insights = [
            {
                "id": r.id,
                "insight_type": r.insight_type,
                "insight_text": r.insight_text,
                "supporting_evidence_ids": r.supporting_evidence_ids,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in insight_records
        ]

        # --- Competitive Intelligence (active report) ---
        ci_record = (
            session.query(CompetitiveIntelligence)
            .filter(
                CompetitiveIntelligence.research_run_id == run_id,
                CompetitiveIntelligence.superseded_at.is_(None),
            )
            .order_by(CompetitiveIntelligence.id.desc())
            .first()
        )
        competitors = ci_record.competitors if ci_record else []
        competitive_gaps = list(ci_record.competitive_gaps if ci_record else [])
        # Ensure stable integer IDs on each competitive gap
        for idx, g in enumerate(competitive_gaps, 1):
            if isinstance(g, dict) and "id" not in g:
                g["id"] = idx

        # --- Creative Gaps (active only) ---
        gap_records = (
            session.query(CreativeGap)
            .filter(
                CreativeGap.research_run_id == run_id,
                CreativeGap.superseded_at.is_(None),
            )
            .order_by(CreativeGap.id)
            .all()
        )
        creative_gaps = [
            {
                "id": r.id,
                "gap_title": r.gap_title,
                "description": r.description,
                "confidence": r.confidence,
                "supporting_consumer_insight_ids": r.supporting_consumer_insight_ids,
                "supporting_competitor_names": r.supporting_competitor_names,
                "supporting_competitive_gap_ids": r.supporting_competitive_gap_ids,
            }
            for r in gap_records
        ]

        # --- Ranked Opportunities (active, overall_score recomputed from stored components) ---
        opp_records = (
            session.query(OpportunityScore, CreativeGap)
            .join(CreativeGap, OpportunityScore.creative_gap_id == CreativeGap.id)
            .filter(
                OpportunityScore.research_run_id == run_id,
                OpportunityScore.superseded_at.is_(None),
                CreativeGap.superseded_at.is_(None),
            )
            .order_by(OpportunityScore.rank.asc())
            .all()
        )

        from opportunity_engine import (
            WEIGHT_AUDIENCE_DEMAND,
            WEIGHT_COMPETITIVE_WHITESPACE,
            WEIGHT_BRAND_FIT,
            WEIGHT_FEASIBILITY,
        )

        def _recompute_score(opp: OpportunityScore) -> float:
            """Weighted overall score from stored integer component scores."""
            return round(
                opp.audience_demand_score * WEIGHT_AUDIENCE_DEMAND
                + opp.competitive_whitespace_score * WEIGHT_COMPETITIVE_WHITESPACE
                + opp.brand_fit_score * WEIGHT_BRAND_FIT
                + opp.feasibility_score * WEIGHT_FEASIBILITY,
                2,
            )

        ranked_opportunities = [
            {
                "id": opp.id,
                "rank": opp.rank,
                "creative_gap_id": opp.creative_gap_id,
                "gap_title": gap.gap_title,
                "overall_score": _recompute_score(opp),
                "stored_overall_score": opp.overall_score,
                "scores": {
                    "audience_demand": opp.audience_demand_score,
                    "competitive_whitespace": opp.competitive_whitespace_score,
                    "brand_fit": opp.brand_fit_score,
                    "feasibility": opp.feasibility_score,
                },
                "reasoning": {
                    "audience_demand": opp.audience_demand_reasoning,
                    "competitive_whitespace": opp.competitive_whitespace_reasoning,
                    "brand_fit": opp.brand_fit_reasoning,
                    "feasibility": opp.feasibility_reasoning,
                },
            }
            for opp, gap in opp_records
        ]

        # --- Content Pieces (active only, with claims_to_verify) ---
        piece_records = (
            session.query(ContentPiece)
            .filter(
                ContentPiece.research_run_id == run_id,
                ContentPiece.superseded_at.is_(None),
            )
            .order_by(ContentPiece.id)
            .all()
        )
        content_pieces = [
            {
                "id": r.id,
                "creative_gap_id": r.creative_gap_id,
                "content_type": r.content_type,
                "platform": r.platform,
                "title": r.title,
                "hook": r.hook,
                "body": r.body,
                "caption": r.caption,
                "hashtags": r.hashtags,
                "call_to_action": r.call_to_action,
                "status": r.status,
                "claims_to_verify": r.claims_to_verify or [],
                "supporting_consumer_insight_ids": r.supporting_consumer_insight_ids,
            }
            for r in piece_records
        ]

        # --- Cost Summary (aggregated from api_calls, SELECT only) ---
        calls = (
            session.query(APICall)
            .filter(APICall.research_run_id == run_id)
            .all()
        )
        context_record = (
            session.query(ResearchContextRecord)
            .filter_by(research_run_id=run_id)
            .first()
        )
        total_cost = sum(c.estimated_cost_usd or 0.0 for c in calls)
        total_ms = sum(c.duration_ms or 0 for c in calls)
        serpapi_calls = sum(1 for c in calls if c.provider == "serpapi")
        cost_summary = {
            "note": (
                "All services are on FREE tiers. "
                "Gemini costs shown are hypothetical paid-tier estimates."
            ),
            "hypothetical_paid_tier_cost_usd": round(total_cost, 6),
            "total_duration_ms": total_ms,
            "call_count": len(calls),
            "serpapi_calls_this_run": serpapi_calls,
        }

        status, failure_reason, diagnostic_report = get_run_status_and_diagnostics(
            run_id=run.id,
            brand=run.brand,
            evidence_count=len(evidence)
        )

        return {
            "brief": brief,
            "status": status,
            "failure_reason": failure_reason,
            "diagnostic_report": diagnostic_report,
            "evidence": evidence,
            "consumer_insights": consumer_insights,
            "competitors": competitors,
            "competitive_gaps": competitive_gaps,
            "creative_gaps": creative_gaps,
            "ranked_opportunities": ranked_opportunities,
            "content_pieces": content_pieces,
            "research_context": context_record.context if context_record else None,
            "cost_summary": cost_summary,
        }
    finally:
        session.close()
