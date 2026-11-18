#!/usr/bin/env python3
"""
CatchyLab Agent 1 (Researcher) - Main Entry Point
-------------------------------------------------
Supports both CLI execution and FastAPI Web Server hosting.

Usage:
  CLI Mode:         python main.py
  Web Server Mode:  python main.py --serve
"""

import sys
import uvicorn
from researcher import run_research_pipeline
from app import app

def run_cli():
    """Runs hardcoded example research brief via CLI."""
    example_brief = {
        "brand": "Nike",
        "market": "India",
        "audience": "Gen Z",
        "objective": "Find opportunities for sustainable running shoes"
    }
    
    print("=" * 60)
    print("  CatchyLab Agent 1 (Researcher) - Market Research Agent  ")
    print("=" * 60)
    
    try:
        result = run_research_pipeline(example_brief)
        print("\n" + "=" * 60)
        print(f"SUCCESS! Pipeline completed.")
        print(f"Extracted Evidence Count: {result['evidence_count']}")
        print(f"Generated Search Queries: {len(result['queries'])}")
        print(f"Report written to local JSON: {result['report_file']}")
        print("=" * 60)
    except Exception as exc:
        print(f"Pipeline Execution Error: {exc}", file=sys.stderr)
        sys.exit(1)


def serve_api():
    """Launches FastAPI uvicorn web server."""
    print("Starting CatchyLab Agent 1 (Researcher) FastAPI Server...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--serve", "-s", "serve"):
        serve_api()
    else:
        run_cli()
