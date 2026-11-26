"""Timestamped run logging and progress-status files for Lab 03.

Fills the "Remaining gaps" item from docs/lab03/DELIVERABLE_CHECKLIST.md:
the notebook alone did not write timestamped START/SUCCESS logs or
progress-status files. This module is the single place that does both for
the pipeline (and can be reused from the notebook).
"""
from __future__ import annotations

import json
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


def _now() -> str:
    """Purpose: timestamp string shared by the text log and the JSON lines log.

    Input: none. Output: ``YYYY-MM-DD HH:MM:SS`` string (local time).
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class RunLogger:
    """Purpose: write one timestamped text log plus one JSON-lines log per pipeline run.

    Input: ``logs_dir`` (``logs/lab03``) and ``run_id``.
    Output: ``log(stage, status, message, extra=None)`` appends a line to
    ``pipeline_<run_id>.log`` (``[timestamp] [STAGE] [STATUS] message``) and to
    ``pipeline_structured.jsonl`` (one JSON object per line).
    """

    def __init__(self, logs_dir: str | Path, run_id: str) -> None:
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.text_path = self.logs_dir / f"pipeline_{run_id}.log"
        self.jsonl_path = self.logs_dir / "pipeline_structured.jsonl"

    def log(self, stage: str, status: str, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Purpose: append one timestamped event to both log files.

        Input: stage name (e.g. ``TRAINING``), status (``START``/``SUCCESS``/``ERROR``/``INFO``),
        a human-readable message and optional structured ``extra`` payload.
        Output: none (appends to disk); never raises on logging failure.
        """
        ts = _now()
        with open(self.text_path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] [{stage}] [{status}] {message}\n")
        record = {"timestamp": ts, "run_id": self.run_id, "stage": stage, "status": status, "message": message}
        if extra is not None:
            record["extra"] = extra
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    @contextmanager
    def stage(self, name: str, start_message: str) -> Iterator[Dict[str, Any]]:
        """Purpose: wrap one pipeline phase with guaranteed START/SUCCESS or START/ERROR logging.

        Input: stage name and the message logged at START.
        Output: a mutable ``ctx`` dict; set ``ctx["summary"]`` inside the ``with`` block to
        customize the SUCCESS message. Re-raises any exception after logging ERROR with traceback.
        """
        ctx: Dict[str, Any] = {"summary": ""}
        t0 = time.perf_counter()
        self.log(name, "START", start_message)
        try:
            yield ctx
        except Exception as exc:  # noqa: BLE001 - we log then re-raise
            self.log(name, "ERROR", f"{type(exc).__name__}: {exc}", {"traceback": traceback.format_exc()})
            raise
        dt = time.perf_counter() - t0
        self.log(name, "SUCCESS", f"Completed in {dt:.2f}s | {ctx.get('summary', '')}")


class ProgressTracker:
    """Purpose: maintain ``progress_results/lab03/run_summary_<run_id>.json`` and ``latest_run.json``.

    Input: ``progress_dir`` and ``run_id``. Output: ``update(stage, data)`` records one
    phase's result; ``finish(status)`` marks the run COMPLETED or FAILED.
    """

    def __init__(self, progress_dir: str | Path, run_id: str) -> None:
        self.progress_dir = Path(progress_dir)
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.summary_path = self.progress_dir / f"run_summary_{run_id}.json"
        self.latest_path = self.progress_dir / "latest_run.json"
        now = _now()
        self._payload: Dict[str, Any] = {"run_id": run_id, "started_at": now, "updated_at": now,
                                          "status": "IN_PROGRESS", "stages": {}}
        self._write()

    def _write(self) -> None:
        for path in (self.summary_path, self.latest_path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._payload, f, ensure_ascii=False, indent=2, default=str)

    def update(self, stage: str, data: Dict[str, Any]) -> None:
        """Purpose: record one phase's outcome. Input: stage name and JSON-serializable data.

        Output: none; writes both the per-run and latest progress files immediately.
        """
        self._payload["stages"][stage] = {"updated_at": _now(), "data": data}
        self._payload["updated_at"] = _now()
        self._write()

    def finish(self, status: str) -> None:
        """Purpose: mark the run COMPLETED or FAILED. Input: status string. Output: none."""
        self._payload["status"] = status
        self._payload["updated_at"] = _now()
        self._write()
