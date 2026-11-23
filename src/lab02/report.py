"""
src/lab02/report.py
Wrapper chuẩn hóa giống src/lab04/report.py — gọi duy nhất script/lab02/render_report.py
để kết xuất đồng bộ 3 định dạng (.html, .pdf, .md) cho Dataset 1 và Dataset 2.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_render_module() -> Any:
    target = PROJECT_ROOT / "script" / "lab02" / "render_report.py"
    spec = importlib.util.spec_from_file_location("lab02_render_report", target)
    if spec is None or spec.loader is None:
        raise ImportError(f"Không thể nạp module từ: {target}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lab02_render_report"] = mod
    spec.loader.exec_module(mod)
    return mod

def main(
    dataset: str = "all",
    run_id: str | None = None,
    datasets: list[str] | None = None,
) -> Dict[str, Dict[str, str]]:
    mod = _load_render_module()
    if datasets:
        target = "all" if len(datasets) > 1 else datasets[0]
    else:
        target = dataset
    return mod.render_reports(dataset=target)


if __name__ == "__main__":
    main()