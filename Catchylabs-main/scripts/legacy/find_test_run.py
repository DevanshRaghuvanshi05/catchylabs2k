#!/usr/bin/env python3
import sys
from pathlib import Path

if '--live' not in sys.argv[1:]:
    raise SystemExit('This legacy script can access live services or databases. Re-run with --live to confirm.')
sys.argv.remove('--live')
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

"""Find the best test run for Agent 2."""
from database import engine
from sqlalchemy import text

with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT r.id, r.brand, r.market, r.audience, r.objective, r.created_at,
               COUNT(e.id) AS evidence_count
        FROM research_runs r
        LEFT JOIN evidence e ON e.research_run_id = r.id
        GROUP BY r.id, r.brand, r.market, r.audience, r.objective, r.created_at
        HAVING COUNT(e.id) > 0
        ORDER BY COUNT(e.id) DESC
        LIMIT 5;
    """))
    print(f"{'RUN ID':<40} {'BRAND':<8} {'MARKET':<8} {'AUDIENCE':<10} {'EVIDENCE':<10}")
    print("=" * 80)
    for row in result:
        print(f"{row[0]:<40} {row[1]:<8} {row[2]:<8} {row[3]:<10} {row[6]:<10}")
        print(f"  Objective: {row[4][:70]}")
