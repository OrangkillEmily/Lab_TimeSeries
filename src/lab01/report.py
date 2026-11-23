r"""
Mô-đun Kết xuất Báo cáo Tự động (Automated Report Generator) cho Lab 01.
Gọi script/lab01/render_report.py để xuất đồng bộ LAB_01_REPORT.{html, pdf, md}.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from src.lab01.tracking import get_active_run_id, record_stage_result, write_log
from src.lab01.utils import PROJECT_ROOT, ensure_directories, get_path, load_config


def _load_render_report_module() -> Any:
    r"""
    Purpose: Nạp động mô-đun script/lab01/render_report.py để dùng chung 1 luồng kết xuất báo cáo.
    Input-Output:
        - Input: Không có.
        - Output: Mô-đun Python lab01_render_report đã được nạp.
    How: Sử dụng importlib.util.spec_from_file_location trỏ tới script/lab01/render_report.py.
    """
    target_script = PROJECT_ROOT / "script" / "lab01" / "render_report.py"
    spec = importlib.util.spec_from_file_location("lab01_render_report", target_script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Không thể nạp mô-đun render_report từ: {target_script}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lab01_render_report"] = mod
    spec.loader.exec_module(mod)
    return mod


def main(run_id: Optional[str] = None) -> Dict[str, Any]:
    r"""
    Purpose: Thực thi Giai đoạn 6 (Bài tập 6: Kết xuất báo cáo tự động HTML, PDF, Markdown) cho Lab 01.
    Input-Output:
        - Input: run_id (Optional[str]).
        - Output: Từ điển chứa đường dẫn và kích thước của 3 tệp báo cáo đã tạo.
    How: Nạp script/lab01/render_report.py, gọi render_html(run_id) và đánh dấu trạng thái COMPLETED.
    """
    t0 = time.perf_counter()
    cfg = load_config()
    ensure_directories(cfg)
    active_run_id = get_active_run_id(run_id)

    write_log(active_run_id, "REPORTING", "START", "Bắt đầu kết xuất bộ 3 báo cáo tự động Lab 01 (HTML, PDF, Markdown).")
    render_mod = _load_render_report_module()
    html_path: Path = render_mod.render_html(run_id=active_run_id)

    reports_dir = get_path("reports_dir", cfg)
    pdf_path = reports_dir / "LAB_01_REPORT.pdf"
    md_path = reports_dir / "LAB_01_REPORT.md"

    duration = time.perf_counter() - t0
    rep_summary: Dict[str, Any] = {
        "html_report": str(html_path.relative_to(PROJECT_ROOT)) if html_path.exists() else "",
        "pdf_report": str(pdf_path.relative_to(PROJECT_ROOT)) if pdf_path.exists() else "",
        "md_report": str(md_path.relative_to(PROJECT_ROOT)) if md_path.exists() else "",
        "html_size_bytes": int(html_path.stat().st_size) if html_path.exists() else 0,
        "pdf_size_bytes": int(pdf_path.stat().st_size) if pdf_path.exists() else 0,
        "md_size_bytes": int(md_path.stat().st_size) if md_path.exists() else 0,
        "duration_sec": round(duration, 4),
    }

    record_stage_result(active_run_id, "reporting", rep_summary, mark_completed=True)
    write_log(
        active_run_id,
        "REPORTING",
        "SUCCESS",
        f"Hoàn tất xuất báo cáo Lab 01 trong {duration:.2f}s | HTML={rep_summary['html_size_bytes']:,}B | PDF={rep_summary['pdf_size_bytes']:,}B | MD={rep_summary['md_size_bytes']:,}B.",
    )
    return rep_summary