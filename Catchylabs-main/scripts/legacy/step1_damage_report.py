#!/usr/bin/env python3
import sys
from pathlib import Path

if '--live' not in sys.argv[1:]:
    raise SystemExit('This legacy script can access live services or databases. Re-run with --live to confirm.')
sys.argv.remove('--live')
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

"""Step 1 - Damage Report: row counts for ALL tables + evidence per run."""
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine, text

DATABASE_URL = os.environ.get("DATABASE_URL")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

with engine.connect() as conn:
    print("=" * 70)
    print("ROW COUNTS FOR ALL TABLES")
    print("=" * 70)
    tables = conn.execute(text("""
        SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename
    """)).fetchall()
    for (t,) in tables:
        count = conn.execute(text(f"SELECT count(*) FROM public.\"{t}\"")).scalar()
        print(f"  {t}: {count}")

    print()
    print("=" * 70)
    print("EVIDENCE COUNT PER RESEARCH RUN")
    print("=" * 70)
    rows = conn.execute(text("""
        SELECT r.id, r.brand, r.market, count(e.id) AS evidence_count
        FROM research_runs r
        LEFT JOIN evidence e ON e.research_run_id = r.id
        GROUP BY r.id, r.brand, r.market
        ORDER BY evidence_count DESC
    """)).fetchall()
    for r in rows:
        print(f"  run={r[0][:12]}...  brand={r[1]}  market={r[2]}  evidence={r[3]}")

    if rows:
        best = rows[0]
        print()
        print(f"  >>> Run with most evidence: {best[0]} ({best[1]}/{best[2]}) = {best[3]} items")
