"""Idempotent live schema migration; dry-run by default.

Dry run:  python migrate_live.py
Apply:    python migrate_live.py --apply
Inspect:  python migrate_live.py --inspect

Apply mode reads DATABASE_URL directly from the process environment (it never
loads .env) and requires:
  1. A .dump backup file in the backups/ folder that is less than 24 hours old.
  2. The user to type the live Supabase project ID before the connection is made.

Inspect mode connects read-only and prints schema metadata and row counts using
SELECT statements only — no DDL or DML is executed.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from sqlalchemy import create_engine, text


MIGRATION_SQL = [
    "CREATE EXTENSION IF NOT EXISTS vector",
    "ALTER TABLE public.consumer_insights ADD COLUMN IF NOT EXISTS superseded_at TIMESTAMPTZ NULL",
    "ALTER TABLE public.competitive_intelligence ADD COLUMN IF NOT EXISTS superseded_at TIMESTAMPTZ NULL",
    "ALTER TABLE public.creative_gaps ADD COLUMN IF NOT EXISTS superseded_at TIMESTAMPTZ NULL",
    "ALTER TABLE public.opportunity_scores ADD COLUMN IF NOT EXISTS superseded_at TIMESTAMPTZ NULL",
    "ALTER TABLE public.content_pieces ADD COLUMN IF NOT EXISTS superseded_at TIMESTAMPTZ NULL",
    "ALTER TABLE public.content_pieces ADD COLUMN IF NOT EXISTS claims_to_verify TEXT[] DEFAULT '{}'::text[]",
    """CREATE TABLE IF NOT EXISTS public.content_feedback (
        id SERIAL PRIMARY KEY,
        content_piece_id INTEGER NULL REFERENCES public.content_pieces(id) ON DELETE SET NULL,
        snapshot_title TEXT NULL,
        snapshot_hook TEXT NULL,
        snapshot_body JSON NULL,
        snapshot_claims_to_verify VARCHAR[] NULL,
        snapshot_creative_gap_id INTEGER NULL,
        status VARCHAR(20) NOT NULL,
        note TEXT NULL,
        created_at TIMESTAMPTZ NULL DEFAULT now()
    )""",
    "CREATE INDEX IF NOT EXISTS ix_content_feedback_content_piece_id ON public.content_feedback (content_piece_id)",
    "ALTER TABLE public.content_feedback ADD COLUMN IF NOT EXISTS snapshot_title TEXT NULL",
    "ALTER TABLE public.content_feedback ADD COLUMN IF NOT EXISTS snapshot_hook TEXT NULL",
    "ALTER TABLE public.content_feedback ADD COLUMN IF NOT EXISTS snapshot_body JSON NULL",
    "ALTER TABLE public.content_feedback ADD COLUMN IF NOT EXISTS snapshot_claims_to_verify VARCHAR[] NULL",
    "ALTER TABLE public.content_feedback ADD COLUMN IF NOT EXISTS snapshot_creative_gap_id INTEGER NULL",
    "ALTER TABLE public.content_feedback ADD COLUMN IF NOT EXISTS status VARCHAR(20)",
    "ALTER TABLE public.content_feedback ADD COLUMN IF NOT EXISTS note TEXT NULL",
    "ALTER TABLE public.content_feedback ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT now()",
    # The model contract permits only one active CI report per run. Other
    # agent tables intentionally allow multiple active rows per run.
    """DO $$ DECLARE item RECORD; BEGIN
        FOR item IN
            SELECT conname FROM pg_constraint
            WHERE conrelid = 'public.competitive_intelligence'::regclass
              AND contype = 'u'
              AND pg_get_constraintdef(oid) ILIKE 'UNIQUE (research_run_id)%'
        LOOP
            EXECUTE format('ALTER TABLE public.competitive_intelligence DROP CONSTRAINT %I', item.conname);
        END LOOP;
    END $$""",
    """DO $$ DECLARE item RECORD; BEGIN
        FOR item IN
            SELECT conname FROM pg_constraint
            WHERE contype = 'f'
              AND conrelid = 'public.content_feedback'::regclass
              AND pg_get_constraintdef(oid) ILIKE 'FOREIGN KEY (content_piece_id)%'
        LOOP
            EXECUTE format('ALTER TABLE public.content_feedback DROP CONSTRAINT %I', item.conname);
        END LOOP;
        ALTER TABLE public.content_feedback ADD CONSTRAINT fk_content_feedback_piece_set_null
          FOREIGN KEY (content_piece_id) REFERENCES public.content_pieces(id) ON DELETE SET NULL;
    END $$""",
    "DROP INDEX IF EXISTS public.ix_competitive_intelligence_research_run_id",
    "DROP INDEX IF EXISTS public.ix_ci_research_run_active",
    "CREATE UNIQUE INDEX ix_ci_research_run_active ON public.competitive_intelligence (research_run_id) WHERE superseded_at IS NULL",
    """DO $$ DECLARE item RECORD; BEGIN
        FOR item IN
            SELECT conrelid::regclass AS table_name, conname
            FROM pg_constraint
            WHERE contype = 'f'
              AND conrelid IN ('public.opportunity_scores'::regclass, 'public.content_pieces'::regclass)
              AND pg_get_constraintdef(oid) ILIKE ANY (ARRAY[
                  'FOREIGN KEY (research_run_id)%', 'FOREIGN KEY (creative_gap_id)%'
              ])
        LOOP
            EXECUTE format('ALTER TABLE %s DROP CONSTRAINT %I', item.table_name, item.conname);
        END LOOP;
        ALTER TABLE public.opportunity_scores ADD CONSTRAINT fk_opportunity_scores_research_run_restrict
          FOREIGN KEY (research_run_id) REFERENCES public.research_runs(id) ON DELETE RESTRICT;
        ALTER TABLE public.opportunity_scores ADD CONSTRAINT fk_opportunity_scores_creative_gap_restrict
          FOREIGN KEY (creative_gap_id) REFERENCES public.creative_gaps(id) ON DELETE RESTRICT;
        ALTER TABLE public.content_pieces ADD CONSTRAINT fk_content_pieces_research_run_restrict
          FOREIGN KEY (research_run_id) REFERENCES public.research_runs(id) ON DELETE RESTRICT;
        ALTER TABLE public.content_pieces ADD CONSTRAINT fk_content_pieces_creative_gap_restrict
          FOREIGN KEY (creative_gap_id) REFERENCES public.creative_gaps(id) ON DELETE RESTRICT;
    END $$""",
    "ALTER TABLE public.research_runs ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE public.sources ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE public.evidence ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE public.api_calls ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE public.consumer_insights ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE public.competitive_intelligence ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE public.creative_gaps ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE public.opportunity_scores ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE public.content_pieces ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE public.content_feedback ENABLE ROW LEVEL SECURITY",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_project_ref(url: str) -> str | None:
    """Extract Supabase project ref from host (db.<ref>.supabase.co) or username."""
    parsed = urlparse(url)
    if parsed.hostname:
        m = re.search(r"\bdb\.([a-zA-Z0-9_-]+)\.supabase\.co\b", parsed.hostname, re.IGNORECASE)
        if m:
            return m.group(1)
    if parsed.username:
        m = re.search(r"\bpostgres\.([a-zA-Z0-9_-]+)\b", parsed.username, re.IGNORECASE)
        if m:
            return m.group(1)
        if "." in parsed.username:
            parts = parsed.username.split(".", 1)
            if parts[0] in ("postgres", "default"):
                return parts[1]
    return None


def _find_recent_backup(backups_dir: Path, max_age_seconds: int = 86_400) -> Path | None:
    """Return the most recent .dump file if it is within max_age_seconds, else None."""
    candidates = glob.glob(str(backups_dir / "*.dump"))
    now = time.time()
    fresh = [p for p in candidates if (now - os.path.getmtime(p)) <= max_age_seconds]
    if not fresh:
        return None
    return Path(max(fresh, key=os.path.getmtime))


def _preflight_check(database_url: str) -> None:
    """Run pre-flight checks before applying migrations.

    Checks (in order):
      1. A .dump backup in backups/ must be less than 24 hours old.
      2. Prints live host and project ID, then requires the user to type the
         project ID to confirm intent.

    Raises SystemExit on any failure.
    """
    project_root = Path(__file__).resolve().parent
    backups_dir = project_root / "backups"

    if not backups_dir.exists():
        print(
            f"[PREFLIGHT FAIL] Backups directory '{backups_dir}' does not exist.\n"
            "Create it and run a pg_dump before applying migrations:\n"
            "  mkdir backups\n"
            '  pg_dump "$DATABASE_URL" -Fc -f backups/$(date +%Y%m%d_%H%M%S).dump',
            file=sys.stderr,
        )
        raise SystemExit(3)

    recent = _find_recent_backup(backups_dir)
    if recent is None:
        print(
            "[PREFLIGHT FAIL] No .dump file in backups/ is less than 24 hours old.\n"
            "Take a fresh backup before applying:\n"
            '  pg_dump "$DATABASE_URL" -Fc -f backups/$(date +%Y%m%d_%H%M%S).dump',
            file=sys.stderr,
        )
        raise SystemExit(3)

    age_h = (time.time() - os.path.getmtime(recent)) / 3600
    print(f"[PREFLIGHT OK] Recent backup found: {recent.name}  (age: {age_h:.1f}h)")

    # Print target identity and require typed project-ID confirmation
    parsed = urlparse(database_url)
    live_host = parsed.hostname or "unknown-host"
    live_db = (parsed.path or "").lstrip("/") or "unknown-db"
    project_ref = _extract_project_ref(database_url) or live_db

    print()
    print("=" * 60)
    print("  LIVE TARGET DETAILS")
    print(f"  Host       : {live_host}")
    print(f"  Database   : {live_db}")
    print(f"  Project ID : {project_ref}")
    print("=" * 60)
    print()
    print("Type the Project ID exactly to confirm (Ctrl-C to abort):")
    try:
        entered = input("  Project ID: ").strip()
    except KeyboardInterrupt:
        print("\n[ABORTED] Ctrl-C received; no changes applied.", file=sys.stderr)
        raise SystemExit(2)

    if entered != project_ref:
        print(
            f"\n[ABORTED] You entered '{entered}' but expected '{project_ref}'.\n"
            "No connection made; no changes applied.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    print(f"\n[CONFIRMED] Proceeding with migration on project '{project_ref}'.\n")


# ---------------------------------------------------------------------------
# Read-only inspection queries (SELECT only — no DML/DDL)
# ---------------------------------------------------------------------------

_INSPECT_SECTIONS: list[tuple[str, str]] = [
    ("TABLES", """
        SELECT t.table_name,
               obj_description(
                   (quote_ident(t.table_schema) || '.' || quote_ident(t.table_name))::regclass,
                   'pg_class'
               ) AS comment
        FROM information_schema.tables t
        WHERE t.table_schema = 'public' AND t.table_type = 'BASE TABLE'
        ORDER BY t.table_name
    """),
    ("COLUMNS", """
        SELECT c.table_name, c.column_name, c.data_type, c.is_nullable, c.column_default
        FROM information_schema.columns c
        WHERE c.table_schema = 'public'
        ORDER BY c.table_name, c.ordinal_position
    """),
    ("INDEXES", """
        SELECT tablename, indexname, indexdef
        FROM pg_indexes
        WHERE schemaname = 'public'
        ORDER BY tablename, indexname
    """),
    ("CONSTRAINTS", """
        SELECT tc.table_name, tc.constraint_name, tc.constraint_type, kcu.column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name
         AND tc.table_schema   = kcu.table_schema
        WHERE tc.table_schema = 'public'
        ORDER BY tc.table_name, tc.constraint_name
    """),
    ("ROW COUNTS (approximate)", """
        SELECT relname AS table_name, reltuples::BIGINT AS approx_row_count
        FROM pg_class
        WHERE relnamespace = 'public'::regnamespace AND relkind = 'r'
        ORDER BY relname
    """),
]


def _print_table(rows: list, headers: list[str]) -> None:
    col_widths = [
        max(len(h), max((len(str(r._mapping.get(h) or "")) for r in rows), default=0))
        for h in headers
    ]
    print("  " + "  ".join(h.ljust(w) for h, w in zip(headers, col_widths)))
    print("  " + "  ".join("-" * w for w in col_widths))
    for row in rows:
        print("  " + "  ".join(str(row._mapping.get(h) or "").ljust(w) for h, w in zip(headers, col_widths)))


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_dry_run() -> int:
    """Print every SQL statement with numbering; no database connection."""
    print("DRY RUN ONLY — no database connection made. SQL to apply:\n")
    for i, statement in enumerate(MIGRATION_SQL, start=1):
        print(f"-- Statement {i}")
        print(statement.strip().rstrip(";") + ";")
        print()
    return 0


def run_apply(database_url: str) -> int:
    """Run pre-flight checks, then apply all statements in one transaction."""
    _preflight_check(database_url)

    engine = create_engine(database_url, pool_pre_ping=True)
    try:
        with engine.begin() as connection:
            for statement in MIGRATION_SQL:
                connection.execute(text(statement))
    except Exception:
        # engine.begin() rolls back the transaction automatically on any exception
        raise
    finally:
        engine.dispose()

    parsed = urlparse(database_url)
    database_name = (parsed.path or "").lstrip("/") or "unknown"
    print(f"Migration applied successfully to database '{database_name}'.")
    return 0


def run_inspect(database_url: str) -> int:
    """Connect and print live schema + row counts using SELECT statements only."""
    engine = create_engine(database_url, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            for title, sql in _INSPECT_SECTIONS:
                print(f"\n{'=' * 60}")
                print(f"  {title}")
                print(f"{'=' * 60}")
                rows = conn.execute(text(sql.strip())).fetchall()
                if not rows:
                    print("  (no rows)")
                    continue
                headers = list(rows[0]._mapping.keys())
                _print_table(rows, headers)
    finally:
        engine.dispose()
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--apply", action="store_true",
        help="Apply the migration to DATABASE_URL after pre-flight checks and typed confirmation.",
    )
    mode.add_argument(
        "--inspect", action="store_true",
        help=(
            "Read-only: print live schema (tables, columns, indexes, constraints) "
            "and approximate row counts via SELECT only. No DDL or DML is executed."
        ),
    )
    args = parser.parse_args()

    if args.inspect:
        database_url = os.environ.get("DATABASE_URL")
        if not database_url:
            parser.error("--inspect requires DATABASE_URL in the process environment")
        return run_inspect(database_url)

    if args.apply:
        database_url = os.environ.get("DATABASE_URL")
        if not database_url:
            parser.error("--apply requires DATABASE_URL in the process environment")
        return run_apply(database_url)

    return run_dry_run()


if __name__ == "__main__":
    raise SystemExit(main())
