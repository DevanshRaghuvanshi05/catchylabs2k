#!/usr/bin/env python3
"""
export_demo_snapshot.py — Exports a single research run to demo_data/lays_full.json.
Uses the existing get_full_research_run function from database.py.
Refuses to run unless --run-id is passed.
"""

import argparse
import json
import os
import sys

from database import get_full_research_run


def export_snapshot(run_id: str, output_path: str = "demo_data/lays_full.json") -> str:
    """
    Exports a research run snapshot to output_path using get_full_research_run.
    Refuses to execute if run_id is empty or missing.
    """
    if not run_id or not str(run_id).strip():
        raise ValueError("Missing required --run-id argument.")

    run_id_clean = str(run_id).strip()
    data = get_full_research_run(run_id_clean)
    if data is None:
        raise ValueError(f"Research run '{run_id_clean}' not found.")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Export research run to demo_data/lays_full.json using get_full_research_run."
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Research run UUID to export (required).",
    )
    parser.add_argument(
        "--output",
        default="demo_data/lays_full.json",
        help="Destination JSON path (default: demo_data/lays_full.json).",
    )

    args = parser.parse_args()
    try:
        out_path = export_snapshot(args.run_id, args.output)
        print(f"Exported run {args.run_id} to {out_path}")
    except Exception as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
