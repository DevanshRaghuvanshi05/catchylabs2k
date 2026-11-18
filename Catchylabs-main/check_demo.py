#!/usr/bin/env python3
"""
check_demo.py — Validates CatchyLabs read-only demo API endpoints.
Takes a base URL as an argument, queries endpoints, and prints status codes and counts.
Do not print secrets, payloads, or credentials.
"""

import os
import sys
import urllib.parse
import urllib.request
import json


def make_request(url: str, api_key: str = None) -> tuple[int, dict]:
    """Helper to perform HTTP GET and return (status_code, response_json)."""
    headers = {"User-Agent": "CatchyLabDemoChecker/1.0"}
    if api_key:
        headers["X-API-Key"] = api_key
    
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req) as resp:
            status_code = resp.status
            body = json.loads(resp.read().decode("utf-8"))
            return status_code, body
    except urllib.error.HTTPError as err:
        try:
            body = json.loads(err.read().decode("utf-8"))
        except Exception:
            body = {}
        return err.code, body
    except Exception as exc:
        print(f"Request failed for {url}: {exc}", file=sys.stderr)
        return 0, {}


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_demo.py <base_url>")
        sys.exit(1)

    base_url = sys.argv[1].rstrip("/")
    api_key = os.environ.get("CATCHYLAB_API_KEY", "")
    if not api_key:
        print("Warning: CATCHYLAB_API_KEY environment variable is not set.", file=sys.stderr)

    # 1. GET /health
    health_url = f"{base_url}/health"
    status_health, body_health = make_request(health_url)
    print(f"GET /health -> status: {status_health}")

    # 2. GET /demo/status
    status_url = f"{base_url}/demo/status"
    status_demo, body_demo = make_request(status_url, api_key=api_key)
    demo_mode_val = body_demo.get("demo_mode")
    runs_val = body_demo.get("number_of_runs")
    print(f"GET /demo/status -> status: {status_demo}, demo_mode: {demo_mode_val}, number_of_runs: {runs_val}")

    recent_run_id = body_demo.get("id")

    # 3. GET /runs
    runs_url = f"{base_url}/runs"
    status_runs, body_runs = make_request(runs_url, api_key=api_key)
    total_runs = body_runs.get("total_runs", len(body_runs.get("runs", [])))
    print(f"GET /runs -> status: {status_runs}, total_runs: {total_runs}")

    if not recent_run_id and body_runs.get("runs"):
        recent_run_id = body_runs["runs"][0].get("id")

    # 4. GET /research/<run_id>/full
    if recent_run_id:
        full_url = f"{base_url}/research/{recent_run_id}/full"
        status_full, body_full = make_request(full_url, api_key=api_key)
        
        evidence_count = len(body_full.get("evidence", []))
        insights_count = len(body_full.get("consumer_insights", []))
        gaps_count = len(body_full.get("creative_gaps", []))
        opportunities_count = len(body_full.get("ranked_opportunities", []))

        print(
            f"GET /research/{recent_run_id}/full -> status: {status_full}, "
            f"evidence_count: {evidence_count}, "
            f"consumer_insights_count: {insights_count}, "
            f"creative_gaps_count: {gaps_count}, "
            f"opportunities_count: {opportunities_count}"
        )
    else:
        print("GET /research/<run_id>/full -> skipped (no runs found)")


if __name__ == "__main__":
    main()
