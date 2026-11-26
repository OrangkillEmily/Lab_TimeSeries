"""Repository-style Lab 03 launcher; delegates all work to ``src.lab03.pipeline``.

Usage (from the repository root):
    python script/lab03/run_lab03.py                 # full run (all data, 5-fold CV)
    python script/lab03/run_lab03.py --fast           # quick smoke run (reduced data/trees/folds)
    python script/lab03/run_lab03.py --no-cv          # skip the 5-fold walk-forward CV
    python script/lab03/run_lab03.py --run-id my_run  # reuse a specific run id
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lab03.pipeline import run_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Purpose: parse the Lab 03 CLI flags documented in docs/lab03/phases/Training_Phase.md.

    Input: none (reads sys.argv). Output: an argparse.Namespace with fast/no_cv/run_id.
    """
    parser = argparse.ArgumentParser(description="Run the full Lab 03 time-series pipeline.")
    parser.add_argument("--fast", action="store_true", help="Apply config.yaml's lab03.fast_overrides for a quick smoke run.")
    parser.add_argument("--no-cv", action="store_true", help="Skip the 5-fold expanding-window cross-validation phase.")
    parser.add_argument("--run-id", default=None, help="Explicit run id (default: a timestamp).")
    return parser.parse_args()


def main() -> int:
    """Purpose: run the pipeline and print a short human-readable summary.

    Input: none. Output: process exit code (0 on success, 1 if the pipeline raised).
    """
    args = parse_args()
    try:
        result = run_pipeline(run_id=args.run_id, fast=args.fast, run_cv=not args.no_cv)
    except Exception as exc:  # noqa: BLE001 - surface a clean CLI failure
        print(f"Lab 03 pipeline FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("=" * 70)
    print(f"Run ID    : {result['run_id']}")
    print(f"Status    : {result['status']}")
    print(f"Champion  : {result['champion']}")
    print(f"Elapsed   : {result['elapsed_sec']}s")
    print(f"Run dir   : {result['run_dir']}")
    print(f"Report    : {result['report']['md']}\n            {result['report']['html']}\n            {result['report']['pdf']}")
    print(f"Model sum.: {result['model_summary_dir']}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
