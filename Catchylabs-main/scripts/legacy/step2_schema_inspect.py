#!/usr/bin/env python3
import sys
from pathlib import Path

if '--live' not in sys.argv[1:]:
    raise SystemExit('This legacy script can access live services or databases. Re-run with --live to confirm.')
sys.argv.remove('--live')
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

"""
Step 2 - Schema Inspection (READ ONLY)
Checks:
  1. superseded_at column exists on the 5 tables
"""
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
  2. All unique constraints and indexes on research_run_id
  3. FK list with delete rules
  4. No destructive SQL is run.
"""
import os
import sys
from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine, text

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL not set")
    sys.exit(1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

TABLES_NEED_SUPERSEDED = [
    "consumer_insights",
    "competitive_intelligence",
    "opportunity_scores",
    "content_pieces",
    "creative_gaps",
]

with engine.connect() as conn:
    # 1. Check superseded_at exists
    print("=" * 70)
    print("1. SUPERSEDED_AT COLUMN CHECK")
    print("=" * 70)
    for tbl in TABLES_NEED_SUPERSEDED:
        row = conn.execute(text("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = :tbl
              AND column_name = 'superseded_at'
        """), {"tbl": tbl}).fetchone()
        if row:
            print(f"  ✓ {tbl}.superseded_at  type={row[1]}  nullable={row[2]}")
        else:
            print(f"  ✗ {tbl}.superseded_at  MISSING!")

    # 2. All unique constraints and indexes on research_run_id
    print()
    print("=" * 70)
    print("2. UNIQUE CONSTRAINTS & INDEXES involving research_run_id")
    print("=" * 70)
    # Check unique constraints
    rows = conn.execute(text("""
        SELECT tc.table_name, tc.constraint_name, tc.constraint_type,
               string_agg(kcu.column_name, ', ' ORDER BY kcu.ordinal_position) AS columns
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
            ON kcu.constraint_name = tc.constraint_name
            AND kcu.table_schema = tc.table_schema
        WHERE tc.table_schema = 'public'
          AND tc.constraint_type IN ('UNIQUE', 'PRIMARY KEY')
        GROUP BY tc.table_name, tc.constraint_name, tc.constraint_type
        ORDER BY tc.table_name, tc.constraint_name
    """)).fetchall()
    for r in rows:
        if 'research_run_id' in r[3]:
            print(f"  CONSTRAINT: {r[0]}.{r[1]}  type={r[2]}  columns=({r[3]})")

    # Check all indexes (including partial)
    rows = conn.execute(text("""
        SELECT tablename, indexname, indexdef
        FROM pg_indexes
        WHERE schemaname = 'public'
        ORDER BY tablename, indexname
    """)).fetchall()
    print()
    print("  ALL INDEXES on tables with research_run_id:")
    for r in rows:
        if 'research_run_id' in r[2]:
            print(f"  INDEX: {r[0]}.{r[1]}")
            print(f"    DEF: {r[2]}")

    # 3. FK list with delete rules
    print()
    print("=" * 70)
    print("3. FOREIGN KEY LIST WITH DELETE RULES")
    print("=" * 70)
    fk_rows = conn.execute(text("""
        SELECT
            tc.table_name AS child_table,
            kcu.column_name AS child_column,
            ccu.table_name AS parent_table,
            ccu.column_name AS parent_column,
            rc.delete_rule,
            tc.constraint_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
            ON kcu.constraint_name = tc.constraint_name
            AND kcu.table_schema = tc.table_schema
        JOIN information_schema.constraint_column_usage ccu
            ON ccu.constraint_name = tc.constraint_name
            AND ccu.table_schema = tc.table_schema
        JOIN information_schema.referential_constraints rc
            ON rc.constraint_name = tc.constraint_name
            AND rc.constraint_schema = tc.table_schema
        WHERE tc.constraint_type = 'FOREIGN KEY'
          AND tc.table_schema = 'public'
        ORDER BY tc.table_name, kcu.column_name
    """)).fetchall()
    for r in fk_rows:
        print(f"  {r[0]}.{r[1]} -> {r[2]}.{r[3]}  ON DELETE {r[4]}  ({r[5]})")

    # 4. Check for unique constraints on competitive_intelligence.research_run_id specifically
    print()
    print("=" * 70)
    print("4. UNIQUE CONSTRAINTS ON competitive_intelligence SPECIFICALLY")
    print("=" * 70)
    ci_unique = conn.execute(text("""
        SELECT tc.constraint_name, tc.constraint_type,
               string_agg(kcu.column_name, ', ' ORDER BY kcu.ordinal_position) AS columns
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
            ON kcu.constraint_name = tc.constraint_name
            AND kcu.table_schema = tc.table_schema
        WHERE tc.table_schema = 'public'
          AND tc.table_name = 'competitive_intelligence'
          AND tc.constraint_type IN ('UNIQUE', 'PRIMARY KEY')
        GROUP BY tc.constraint_name, tc.constraint_type
        ORDER BY tc.constraint_name
    """)).fetchall()
    if ci_unique:
        for r in ci_unique:
            print(f"  {r[1]}: {r[0]}  columns=({r[2]})")
    else:
        print("  No UNIQUE constraints found on competitive_intelligence")

    # Also check for unique indexes (not via constraints)
    ci_idx = conn.execute(text("""
        SELECT indexname, indexdef
        FROM pg_indexes
        WHERE schemaname = 'public'
          AND tablename = 'competitive_intelligence'
        ORDER BY indexname
    """)).fetchall()
    print("  All indexes on competitive_intelligence:")
    for r in ci_idx:
        print(f"    {r[0]}: {r[1]}")

    # 5. Check whether there are any unique indexes on research_run_id for ANY table
    print()
    print("=" * 70)
    print("5. ALL UNIQUE INDEXES across all tables")
    print("=" * 70)
    unique_idx = conn.execute(text("""
        SELECT
            i.relname AS index_name,
            t.relname AS table_name,
            pg_get_indexdef(i.oid) AS index_def
        FROM pg_index ix
        JOIN pg_class i ON i.oid = ix.indexrelid
        JOIN pg_class t ON t.oid = ix.indrelid
        JOIN pg_namespace n ON n.oid = t.relnamespace
        WHERE n.nspname = 'public'
          AND ix.indisunique = true
        ORDER BY t.relname, i.relname
    """)).fetchall()
    for r in unique_idx:
        print(f"  {r[1]}.{r[0]}: {r[2]}")

print()
print("Schema inspection complete (read-only).")
