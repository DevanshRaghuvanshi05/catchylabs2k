#!/usr/bin/env bash
# start.sh — CatchyLab unified startup script
# Launches the RQ worker in the background, then starts uvicorn on 0.0.0.0:$PORT.
#
# Environment variables:
#   PORT              (default: 8000)   — uvicorn listen port
#   CATCHYLAB_API_KEY (required)        — API key for all authenticated endpoints
#   DATABASE_URL      (required)        — PostgreSQL connection string
#   REDIS_URL         (required)        — Redis connection string
#   ALLOWED_ORIGINS   (optional)        — comma-separated CORS origins
#   MAX_RUNS_PER_DAY  (optional, def 5) — daily POST /research cap
#
# Worker selection:
#   Linux / macOS  → rq Worker (supports fork-based concurrency)
#   Windows        → rq SimpleWorker (in-process, no fork required)

set -euo pipefail

PORT="${PORT:-8000}"

echo "=== CatchyLab — Starting RQ Worker ==="

# Detect OS and pick the right worker class
case "$(uname -s 2>/dev/null || echo Windows)" in
    Linux|Darwin)
        echo "[start.sh] Unix detected — using standard rq Worker"
        python -m rq worker \
            research_jobs \
            consumer_intelligence_jobs \
            competitive_intelligence_jobs \
            creative_gaps_jobs \
            opportunity_jobs \
            content_jobs \
            &
        RQ_PID=$!
        echo "[start.sh] RQ worker started (PID ${RQ_PID})"
        ;;
    *)
        # Windows (Git Bash / MSYS / Cygwin) — SimpleWorker runs in-process
        echo "[start.sh] Windows detected — using SimpleWorker (python worker.py)"
        python worker.py &
        RQ_PID=$!
        echo "[start.sh] RQ SimpleWorker started (PID ${RQ_PID})"
        ;;
esac

echo "=== CatchyLab — Starting uvicorn on 0.0.0.0:${PORT} ==="
exec uvicorn app:app --host 0.0.0.0 --port "${PORT}"
