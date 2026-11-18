import sys
from pathlib import Path

if '--live' not in sys.argv[1:]:
    raise SystemExit('This legacy script can access live services or databases. Re-run with --live to confirm.')
sys.argv.remove('--live')
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

"""
STEP 2 — Schema inspection: Check superseded_at columns, unique constraints, FK rules.
Read-only queries only.
"""
import os
from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine, text

DATABASE_URL = os.environ.get("DATABASE_URL")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

with engine.connect() as conn:
    # 1. Check superseded_at columns exist
    print("=" * 70)
    print("SUPERSEDED_AT COLUMNS")
    print("=" * 70)
    tables_to_check = [
        "consumer_insights", "competitive_intelligence",
        "opportunity_scores", "content_pieces", "creative_gaps"
    ]
    for t in tables_to_check:
        r = conn.execute(text("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = :tbl AND column_name = 'superseded_at'
        """), {"tbl": t}).fetchall()
        if r:
            for row in r:
                print(f"  {t:30s}  column={row[0]}  type={row[1]}  nullable={row[2]}")
        else:
            print(f"  {t:30s}  *** MISSING ***")

    # 2. Show ALL unique constraints and indexes on research_run_id
    print()
    print("=" * 70)
    print("UNIQUE CONSTRAINTS & INDEXES ON research_run_id")
    print("=" * 70)
    r = conn.execute(text("""
        SELECT
            tc.table_name,
            tc.constraint_name,
            tc.constraint_type,
            kcu.column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
            ON tc.constraint_name = kcu.constraint_name
            AND tc.table_name = kcu.table_name
        WHERE kcu.column_name = 'research_run_id'
            AND tc.constraint_type IN ('UNIQUE', 'PRIMARY KEY')
        ORDER BY tc.table_name, tc.constraint_name
    """)).fetchall()
    for row in r:
        print(f"  table={row[0]:30s}  constraint={row[1]:50s}  type={row[2]}  column={row[3]}")

    # Also check pg_indexes for any unique indexes on research_run_id
    print()
    print("PG INDEXES with 'research_run_id' and UNIQUE:")
    r = conn.execute(text("""
        SELECT tablename, indexname, indexdef
        FROM pg_indexes
        WHERE indexdef LIKE '%research_run_id%'
            AND indexdef LIKE '%UNIQUE%'
        ORDER BY tablename
    """)).fetchall()
    for row in r:
        print(f"  table={row[0]}  index={row[1]}")
        print(f"    def: {row[2]}")

    # 3. Show all FK constraints with delete rules
    print()
    print("=" * 70)
    print("FOREIGN KEY CONSTRAINTS WITH DELETE RULES")
    print("=" * 70)
    r = conn.execute(text("""
        SELECT
            tc.table_name AS child_table,
            kcu.column_name AS child_column,
            ccu.table_name AS parent_table,
            ccu.column_name AS parent_column,
            rc.delete_rule
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
            ON tc.constraint_name = kcu.constraint_name
            AND tc.table_name = kcu.table_name
        JOIN information_schema.constraint_column_usage ccu
            ON tc.constraint_name = ccu.constraint_name
        JOIN information_schema.referential_constraints rc
            ON tc.constraint_name = rc.constraint_name
        WHERE tc.constraint_type = 'FOREIGN KEY'
        ORDER BY tc.table_name, kcu.column_name
    """)).fetchall()
    for row in r:
        print(f"  {row[0]:30s}.{row[1]:25s} -> {row[2]:25s}.{row[3]:15s}  ON DELETE {row[4]}")
