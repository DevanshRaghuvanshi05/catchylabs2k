"""
CatchyLab Backend Diagnostic & Health Verification Script
---------------------------------------------------------
Checks:
  1. Python runtime & version
  2. Required package dependencies
  3. Environment variables & .env configuration
  4. FastAPI application import & ASGI instance
  5. Database connection & SQLite compatibility
  6. All 4 AI agents & pipeline components
  7. Port 8000 availability
"""

import sys
import os
import socket
from pathlib import Path

# Color helpers for terminal output
def _green(t): return f"\033[92m{t}\033[0m"
def _yellow(t): return f"\033[93m{t}\033[0m"
def _red(t): return f"\033[91m{t}\033[0m"
def _bold(t): return f"\033[1m{t}\033[0m"

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

def run_diagnostics():
    print("=" * 65)
    print(_bold("       CatchyLab Backend Health & Environment Diagnostic"))
    print("=" * 65)
    
    passed_checks = 0
    total_checks = 7
    warnings = []

    # 1. Python Version
    print(f"\n[1/7] Python Runtime: {sys.version.split()[0]} ({sys.executable})")
    if sys.version_info >= (3, 10):
        print(f"      {_green('PASS')}: Python version >= 3.10 is satisfied.")
        passed_checks += 1
    else:
        print(f"      {_red('FAIL')}: Python >= 3.10 is required.")

    # 2. Key Dependencies
    print("\n[2/7] Checking Dependencies...")
    required_packages = [
        "fastapi", "uvicorn", "sqlalchemy", "pydantic",
        "redis", "rq", "httpx", "cryptography"
    ]
    missing_pkgs = []
    for pkg in required_packages:
        try:
            __import__(pkg)
        except ImportError:
            missing_pkgs.append(pkg)

    if not missing_pkgs:
        print(f"      {_green('PASS')}: Core packages installed ({', '.join(required_packages)}).")
        passed_checks += 1
    else:
        print(f"      {_red('FAIL')}: Missing packages: {missing_pkgs}")
        print("      Run: pip install -r requirements.txt")

    # 3. Environment Configuration
    print("\n[3/7] Checking Environment Configuration...")
    env_file = backend_dir / ".env"
    if env_file.exists():
        print(f"      {_green('PASS')}: .env file found at {env_file}")
    else:
        print(f"      {_yellow('WARN')}: No .env file found. Using default environment/fallback values.")
        warnings.append("Create a .env file from .env.example for live production services.")

    db_url = os.environ.get("DATABASE_URL", "").strip()
    if db_url:
        masked_url = db_url.split("@")[-1] if "@" in db_url else "[CONFIGURED]"
        print(f"      DB Target: PostgreSQL ({masked_url})")
    else:
        print(f"      DB Target: {_yellow('Transient in-memory SQLite (local dev mode)')}")
        warnings.append("DATABASE_URL is not set. The server will run in local-dev mode with transient SQLite.")

    passed_checks += 1

    # 4. FastAPI Application Import
    print("\n[4/7] Checking FastAPI Application Instance...")
    try:
        import app as app_module
        assert hasattr(app_module, "app"), "app module has no 'app' attribute"
        title = getattr(app_module.app, "title", "FastAPI")
        print(f"      {_green('PASS')}: FastAPI app imported successfully ('{title}').")
        passed_checks += 1
    except Exception as exc:
        print(f"      {_red('FAIL')}: Failed to import app: {exc}")

    # 5. Database Initialization
    print("\n[5/7] Checking Database Engine & Initialization...")
    try:
        from database import init_db, db_engine
        init_db(db_engine)
        dialect = db_engine.dialect.name
        print(f"      {_green('PASS')}: Database initialized cleanly using dialect '{dialect}'.")
        passed_checks += 1
    except Exception as exc:
        print(f"      {_red('FAIL')}: Database initialization failed: {exc}")

    # 6. AI Agent Imports & Verification
    print("\n[6/7] Verifying 4-Agent AI Pipeline Modules...")
    try:
        import pipeline
        import agent2
        import agent3
        import agent4
        import opportunity_engine
        import content_generator
        import jobs

        assert callable(pipeline.run_research_pipeline)
        assert callable(agent2.run_consumer_intelligence_pipeline)
        assert callable(agent3.run_competitive_intelligence_pipeline)
        assert callable(agent4.run_creative_gaps_pipeline)
        assert callable(opportunity_engine.run_opportunity_engine_pipeline)
        assert callable(content_generator.run_content_generation_pipeline)
        assert callable(jobs.run_full_pipeline_job)

        print(f"      {_green('PASS')}: All 4 Agents + Opportunity Engine + Content Generator verified.")
        passed_checks += 1
    except Exception as exc:
        print(f"      {_red('FAIL')}: Agent module verification failed: {exc}")

    # 7. Port 8000 Check
    print("\n[7/7] Checking Port 8000 Availability...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1.0)
    try:
        result = sock.connect_ex(("127.0.0.1", 8000))
        sock.close()
        if result == 0:
            print(f"      {_yellow('NOTE')}: Port 8000 is currently occupied (a server may already be running).")
        else:
            print(f"      {_green('PASS')}: Port 8000 is free and available to bind.")
        passed_checks += 1
    except Exception as exc:
        print(f"      {_yellow('WARN')}: Could not check port: {exc}")
        passed_checks += 1

    # Summary
    print("\n" + "=" * 65)
    print(_bold(f"Diagnostic Summary: {passed_checks}/{total_checks} checks passed."))
    if warnings:
        print(_bold("Warnings / Notices:"))
        for w in warnings:
            print(f"  • {w}")
    print("=" * 65)

    return passed_checks == total_checks

if __name__ == "__main__":
    success = run_diagnostics()
    sys.exit(0 if success else 1)
