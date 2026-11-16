r"""
Mô-đun Kết xuất Báo cáo Tự động (Automated Report Generator) trong gói src/lab04/report.py.
Gọi trực tiếp bộ kết xuất học thuật chuẩn UTH (script/lab04/render_report.py) để đảm bảo
cả 3 định dạng báo cáo:
- reports/lab04/LAB_04_REPORT.html (Interactive HTML5 + 23 Base64 Figures + SVG Flow Map)
- reports/lab04/LAB_04_REPORT.pdf  (A4 Academic PDF rendered via Headless Edge/Chrome hoặc ReportLab)
- reports/lab04/LAB_04_REPORT.md   (Synchronized Markdown Tables & Discussion Answers)
luôn đồng bộ tuyệt đối 100% ở mọi nơi được gọi (từ pipeline.py, run_lab04.py hay lab04.ipynb).
"""

import importlib.util
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from src.lab04.tracking import get_active_run_id, record_stage_result, write_log
from src.lab04.utils import PROJECT_ROOT, ensure_directories, get_path, load_config


def _load_render_report_module() -> Any:
    r"""
    Purpose: Nạp động mô-đun script/lab04/render_report.py để dùng chung duy nhất một pipeline kết xuất báo cáo.
    Input-Output:
        - Input: Không có.
        - Output: Mô-đun render_report đã nạp.
    How: Sử dụng importlib.util.spec_from_file_location trỏ tới script/lab04/render_report.py.
    """
    candidates = [
        PROJECT_ROOT / "script" / "lab04" / "render_report.py",
        PROJECT_ROOT / "render_report.py",
    ]
    target_script = next((p for p in candidates if p.exists()), candidates[0])
    spec = importlib.util.spec_from_file_location("lab04_render_report", target_script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Không thể nạp mô-đun render_report từ: {target_script}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lab04_render_report"] = mod
    spec.loader.exec_module(mod)
    return mod


def main(run_id: Optional[str] = None) -> Dict[str, Any]:
    r"""
    Purpose: Điểm vào chính của bước tạo báo cáo tự động (Stage 6 - Reporting).
    Input-Output:
        - Input: run_id (Optional[str]).
        - Output: Từ điển chứa đường dẫn và dung lượng của 3 tệp báo cáo (.html, .pdf, .md).
    How: Gọi render_report.render_html(run_id=active_run_id) và ghi nhận trạng thái COMPLETED vào run_summary.
    """
    t0 = time.perf_counter()
    cfg = load_config()
    ensure_directories(cfg)
    active_run_id = get_active_run_id(run_id)

    write_log(active_run_id, "REPORTING", "START", "Bắt đầu kết xuất bộ 3 báo cáo tự động (HTML, PDF, Markdown) chuẩn giao diện Lab04_ML.")

    render_mod = _load_render_report_module()
    html_path: Path = render_mod.render_html(run_id=active_run_id)

    reports_dir = get_path("reports_dir", cfg)
    pdf_path = reports_dir / "LAB_04_REPORT.pdf"
    md_path = reports_dir / "LAB_04_REPORT.md"

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
        f"Hoàn tất xuất báo cáo trong {duration:.2f}s | HTML={rep_summary['html_size_bytes']:,}B | PDF={rep_summary['pdf_size_bytes']:,}B | MD={rep_summary['md_size_bytes']:,}B.",
    )
    return rep_summary


if __name__ == "__main__":
    main()