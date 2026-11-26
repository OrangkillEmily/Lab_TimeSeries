"""Root one-command launcher: ``python scripts/run_lab.py --lab 03 [--fast] [--no-cv]``.

Dispatches to the repository-style ``script/lab0<N>/run_lab0<N>.py`` entry point for
the requested lab (default: 03), matching the launcher documented in
``docs/lab03/FILE_GUIDE.md``. Keeping one thin root script means a reviewer can run
any lab with a single, repository-root-relative command without knowing the
per-lab module layout.
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    """Purpose: parse ``--lab`` plus any flags to forward to that lab's own runner.

    Input: none (reads sys.argv). Output: namespace with ``lab`` and ``forward`` (the
    remaining, unparsed CLI arguments to hand to the target lab's runner unchanged).
    """
    parser = argparse.ArgumentParser(description="Run any lab in this repository with one command.",
                                     epilog="Any extra flags (e.g. --fast, --no-cv, --run-id X) are forwarded as-is "
                                            "to the target lab's own script/lab0N/run_lab0N.py.")
    parser.add_argument("--lab", default="03", help="Lab number to run, e.g. 01, 02, 03, 04 (default: 03).")
    args, forward = parser.parse_known_args()
    args.forward = forward
    return args


def main() -> int:
    """Purpose: locate and execute ``script/lab0<N>/run_lab0<N>.py`` as ``__main__``.

    Input: none. Output: the target script's own exit code.
    """
    args = parse_args()
    lab = args.lab.zfill(2)
    target = ROOT / "script" / f"lab{lab}" / f"run_lab{lab}.py"
    if not target.is_file():
        print(f"No runner found for lab '{lab}' (expected {target}).", file=sys.stderr)
        return 1
    sys.argv = [str(target), *args.forward]
    try:
        runpy.run_path(str(target), run_name="__main__")
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
