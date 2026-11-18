import sys
from pathlib import Path

if '--live' not in sys.argv[1:]:
    raise SystemExit('This legacy script can access live services or databases. Re-run with --live to confirm.')
sys.argv.remove('--live')
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

"""
STEP 1 — Damage Report: Read-only row counts for all tables.
No destructive SQL. No writes.
"""
import os

# Prevent importing config.py which might trigger side effects
# Instead, read DATABASE_URL from env directly
from dotenv import load_dotenv
load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL not set")
    sys.exit(1)

from sqlalchemy import create_engine, text

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

TABLES = [
    "research_runs",
    "sources",
    "evidence",
    "api_calls",
    "consumer_insights",
    "competitive_intelligence",
    "creative_gaps",
    "opportunity_scores",
    "content_pieces",
    "content_feedback",
]

print("=" * 60)
print("TABLE ROW COUNTS (live DB)")
print("=" * 60)

with engine.connect() as conn:
    for table in TABLES:
        try:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
            count = result.scalar()
            print(f"  {table:30s}  {count}")
        except Exception as e:
            print(f"  {table:30s}  ERROR: {e}")

    print()
    print("=" * 60)
    print("EVIDENCE COUNT PER RESEARCH RUN")
    print("=" * 60)

    rows = conn.execute(text("""
        SELECT r.id, r.brand, r.market, COUNT(e.id) AS evidence_count
        FROM research_runs r
        LEFT JOIN evidence e ON e.research_run_id = r.id
        GROUP BY r.id, r.brand, r.market
        ORDER BY evidence_count DESC
    """)).fetchall()

    max_ev = 0
    max_run = None
    for row in rows:
        run_id, brand, market, ev_count = row
        print(f"  {run_id}  brand={brand:10s} market={market:10s}  evidence={ev_count}")
        if ev_count > max_ev:
            max_ev = ev_count
            max_run = run_id

    print()
    if max_run:
        print(f">>> Run with MOST evidence: {max_run} ({max_ev} items)")
        print(f"    Rebuild agents 2-4 on this run after quota resets.")
