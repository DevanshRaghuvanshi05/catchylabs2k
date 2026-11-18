import sys
from pathlib import Path

if '--live' not in sys.argv[1:]:
    raise SystemExit('This legacy script can access live services or databases. Re-run with --live to confirm.')
sys.argv.remove('--live')
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

"""Apply partial unique index on competitive_intelligence."""
import os
from dotenv import load_dotenv
load_dotenv()
from sqlalchemy import create_engine, text

engine = create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True)
with engine.connect() as conn:
    conn.execute(text("DROP INDEX IF EXISTS ix_competitive_intelligence_research_run_id;"))
    conn.execute(text(
        "CREATE UNIQUE INDEX ix_ci_research_run_active "
        "ON competitive_intelligence (research_run_id) "
        "WHERE superseded_at IS NULL;"
    ))
    conn.commit()
    print("SUCCESS: Unique index replaced with partial unique index")

    # Verify
    r = conn.execute(text(
        "SELECT indexname, indexdef FROM pg_indexes "
        "WHERE tablename = 'competitive_intelligence'"
    )).fetchall()
    for row in r:
        print(f"  {row[0]}: {row[1]}")
