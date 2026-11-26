"""Shared utilities for Lab 03: config loading, path resolution and reproducibility.

Follows the same ``load_config``/``get_path``/``ensure_directories`` convention
already used by ``src/lab01/utils.py``, ``src/lab02/utils.py`` and ``src/lab04/utils.py``.
"""
from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml


def get_project_root() -> Path:
    """Purpose: locate the repository root (the directory containing ``config.yaml``).

    Input: none. Output: a ``Path``; searched from this file upward.
    """
    here = Path(__file__).resolve()
    for parent in (here.parent, here.parent.parent, here.parent.parent.parent):
        if (parent / "config.yaml").exists():
            return parent
    return here.parent.parent.parent


PROJECT_ROOT: Path = get_project_root()


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Purpose: recursively merge ``override`` on top of ``base`` (used for ``--fast``).

    Input: two dictionaries. Output: a new merged dictionary; ``base`` is not mutated.
    """
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(config_path: Optional[Path] = None, fast: bool = False) -> Dict[str, Any]:
    """Purpose: load the ``lab03`` section of the repository ``config.yaml``.

    Input: optional explicit path (defaults to ``PROJECT_ROOT/config.yaml``) and a
    ``fast`` flag. Output: the ``lab03`` dict, with ``fast_overrides`` merged in when
    ``fast=True`` (used by ``--fast`` smoke runs; never affects a normal run).
    """
    target = Path(config_path) if config_path is not None else PROJECT_ROOT / "config.yaml"
    if not target.exists():
        raise FileNotFoundError(f"Missing configuration file: {target}")
    with open(target, "r", encoding="utf-8") as f:
        full = yaml.safe_load(f)
    if "lab03" not in full:
        raise KeyError("config.yaml is missing the 'lab03' section.")
    cfg = copy.deepcopy(full["lab03"])
    if fast and "fast_overrides" in cfg:
        cfg = _deep_merge(cfg, cfg["fast_overrides"])
    return cfg


def get_path(key: str, cfg: Dict[str, Any]) -> Path:
    """Purpose: resolve a ``paths.<key>`` entry to an absolute path under the repo root.

    Input: a key present in ``cfg["paths"]``. Output: an absolute ``Path``.
    """
    rel = cfg["paths"][key]
    return PROJECT_ROOT / rel


def ensure_directories(cfg: Dict[str, Any]) -> Dict[str, Path]:
    """Purpose: create every Lab 03 output directory declared in ``config.yaml``.

    Input: the ``lab03`` config dict. Output: a mapping ``{key: Path}`` of the
    directories that were created (idempotent; safe to call every run).
    """
    keys = ["data_processed_dir", "models_dir", "reports_dir", "figures_dir", "model_summary_dir",
            "logs_dir", "progress_dir", "docs_dir", "notebooks_dir", "scripts_dir"]
    out: Dict[str, Path] = {}
    for key in keys:
        path = get_path(key, cfg)
        path.mkdir(parents=True, exist_ok=True)
        out[key] = path
    return out


def set_global_seed(seed: int = 42) -> None:
    """Purpose: fix ``random``/NumPy seeds for reproducible bootstraps and subsampling.

    Input: integer seed. Output: none (process-global side effect).
    """
    random.seed(seed)
    np.random.seed(seed)
