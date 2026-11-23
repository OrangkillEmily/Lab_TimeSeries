r"""
Trình Kết xuất Báo cáo Học thuật Toàn diện (HTML5 + 15 Base64 Figures + Headless Edge/Chrome PDF + Markdown)
cho Bài thực hành Lab 01 (Chương 2): EDA, ACF/PACF, White Noise, Decomposition & Data Quality trên AirPassengers.csv.
"""

from __future__ import annotations

import base64
import html
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional


def get_project_root() -> Path:
    r"""
    Purpose: Xác định thư mục gốc của dự án chứa tệp config.yaml.
    Input-Output:
    - Input: Không có.
    - Output: Đối tượng Path trỏ tới thư mục gốc dự án.
    How: Duyệt ngược từ script/lab01/render_report.py lên các thư mục cha cho tới khi gặp config.yaml.
    """
    curr = Path(__file__).resolve()
    for p in [curr.parent, curr.parent.parent, curr.parent.parent.parent]:
        if (p / "config.yaml").exists():
            return p
    return curr.parent.parent.parent


PROJECT_ROOT = get_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lab01.tracking import get_active_run_id  # noqa: E402
from src.lab01.utils import ensure_directories, get_path, load_config, load_json  # noqa: E402


def load_b64(path: Path) -> str:
    r"""
    Purpose: Mã hóa ảnh PNG sang chuỗi Data URI Base64 để nhúng độc lập 100% vào tệp HTML/PDF.
    Input-Output:
    - Input: path (Path tới tệp .png).
    - Output: Chuỗi 'data:image/png;base64,...' hoặc SVG giữ chỗ nếu tệp chưa tồn tại.
    How: Đọc bytes nhị phân và gọi base64.b64encode().
    """
    if not path.exists():
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="800" height="220">'
            f'<rect width="100%" height="100%" fill="#f3f5f0"/>'
            f'<text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" '
            f'fill="#446b52" font-family="sans-serif" font-size="14">'
            f'Chưa tìm thấy biểu đồ: {html.escape(path.name)}</text></svg>'
        )
        return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _find_chromium_binary() -> Optional[str]:
    r"""
    Purpose: Tìm trình duyệt Microsoft Edge, Google Chrome hoặc Chromium để kết xuất HTML sang PDF chuẩn A4.
    Input-Output:
    - Input: Không có.
    - Output: Đường dẫn tệp thực thi trình duyệt hoặc None nếu không tìm thấy.
    How: Kiểm tra shutil.which và các thư mục cài đặt mặc định trên Windows.
    """
    candidates: List[str] = []
    for cmd in ("msedge", "chrome", "google-chrome", "chromium", "chromium-browser"):
        found = shutil.which(cmd)
        if found:
            candidates.append(found)

    pf = os.environ.get("PROGRAMFILES", r"C:\Program Files")
    pf86 = os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")
    local = os.environ.get("LOCALAPPDATA", "")
    win_paths = [
        Path(pf86) / "Microsoft/Edge/Application/msedge.exe",
        Path(pf) / "Microsoft/Edge/Application/msedge.exe",
        Path(pf) / "Google/Chrome/Application/chrome.exe",
        Path(pf86) / "Google/Chrome/Application/chrome.exe",
    ]
    if local:
        win_paths.append(Path(local) / "Google/Chrome/Application/chrome.exe")
        win_paths.append(Path(local) / "Microsoft/Edge/Application/msedge.exe")

    for p in win_paths:
        if p.exists():
            candidates.append(str(p))
    return candidates[0] if candidates else None


def _export_pdf_reportlab_fallback(pdf_path: Path, facts: Dict[str, Any], fig_dir: Path) -> bool:
    r"""
    Purpose: Tạo tệp PDF dự phòng bằng ReportLab nếu môi trường không có Edge/Chrome Headless.
    Input-Output:
    - Input: pdf_path (Path), facts (Dict), fig_dir (Path).
    - Output: True nếu xuất PDF thành công, ngược lại False.
    How: Dùng SimpleDocTemplate, Paragraph và RLImage nhúng toàn bộ biểu đồ trong fig_dir.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import Image as RLImage
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

        font_reg, font_bold = "Helvetica", "Helvetica-Bold"
        for reg_p, bold_p in [
            ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
            ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ]:
            if Path(reg_p).exists() and Path(bold_p).exists():
                pdfmetrics.registerFont(TTFont("CustomSans", reg_p))
                pdfmetrics.registerFont(TTFont("CustomSansBold", bold_p))
                font_reg, font_bold = "CustomSans", "CustomSansBold"
                break

        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=A4,
            leftMargin=14 * mm,
            rightMargin=14 * mm,
            topMargin=14 * mm,
            bottomMargin=14 * mm,
        )
        styles = getSampleStyleSheet()
        h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName=font_bold, fontSize=15, textColor=colors.HexColor("#183f35"))
        h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName=font_bold, fontSize=12, textColor=colors.HexColor("#1d5843"))
        body = ParagraphStyle("Body", parent=styles["Normal"], fontName=font_reg, fontSize=9.5, leading=13.5)

        story: List[Any] = [
            Paragraph("BÁO CÁO KỸ THUẬT LAB 01: PHÂN TÍCH CHUỖI THỜI GIAN (CHƯƠNG 2)", h1),
            Paragraph(f"Run ID: {facts['run_id']} | Dataset: AirPassengers.csv ({facts['data_info'].get('n_obs', 144)} quan sát)", body),
            Spacer(1, 4 * mm),
        ]
        for png in sorted(fig_dir.glob("*.png")):
            story.append(Paragraph(f"Biểu đồ: {png.name}", h2))
            story.append(RLImage(str(png), width=175 * mm, height=82 * mm, kind="proportional"))
            story.append(Spacer(1, 3 * mm))
        doc.build(story)
        return pdf_path.exists() and pdf_path.stat().st_size > 1000
    except Exception:
        return False


def export_html_to_pdf(
    html_path: Path,
    pdf_path: Path,
    facts: Optional[Dict[str, Any]] = None,
    fig_dir: Optional[Path] = None,
) -> bool:
    r"""
    Purpose: Chuyển đổi LAB_01_REPORT.html sang LAB_01_REPORT.pdf qua Headless Edge/Chrome (kèm dự phòng ReportLab).
    Input-Output:
    - Input: html_path (Path), pdf_path (Path), facts (Optional[Dict]), fig_dir (Optional[Path]).
    - Output: True nếu tạo thành công tệp PDF.
    How: Gọi msedge/chrome với cờ --headless=new --print-to-pdf; nếu thất bại thì gọi _export_pdf_reportlab_fallback().
    """
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    html_abs = html_path.resolve()
    pdf_abs = pdf_path.resolve()

    browser_bin = _find_chromium_binary()
    if browser_bin:
        user_data_dir = tempfile.mkdtemp(prefix="lab01_ts_edge_pdf_")
        try:
            for headless_flag in ("--headless=new", "--headless"):
                cmd = [
                    browser_bin,
                    headless_flag,
                    "--disable-gpu",
                    "--no-sandbox",
                    "--no-pdf-header-footer",
                    "--print-to-pdf-no-header",
                    f"--user-data-dir={user_data_dir}",
                    f"--print-to-pdf={pdf_abs}",
                    html_abs.as_uri(),
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60, check=False)
                if pdf_abs.exists() and pdf_abs.stat().st_size > 5000:
                    return True
        except Exception:
            pass
        finally:
            shutil.rmtree(user_data_dir, ignore_errors=True)

    if facts is not None and fig_dir is not None:
        return _export_pdf_reportlab_fallback(pdf_abs, facts, fig_dir)
    return False


def collect_report_facts(run_id: Optional[str] = None) -> Dict[str, Any]:
    r"""
    Purpose: Thu thập động 100% số liệu thực nghiệm của Lab 01 từ JSON metrics, processed data và nhật ký.
    Input-Output:
    - Input: run_id (Optional[str]).
    - Output: Từ điển facts chứa toàn bộ dữ liệu cho mẫu HTML và Markdown.
    How: Đọc data_inspection_summary.json, eda_acf_whitenoise_summary.json, decomposition_summary.json,
         data_quality_summary.json, visualization_summary.json và tệp pipeline_<run_id>.log.
    """
    cfg = load_config()
    ensure_directories(cfg)
    active_run_id = get_active_run_id(run_id)

    proc_dir = get_path("data_processed_dir", cfg)
    metrics_dir = get_path("metrics_dir", cfg)
    logs_dir = get_path("logs_dir", cfg)
    fig_dir = get_path("figures_dir", cfg)

    data_info = load_json(proc_dir / "data_inspection_summary.json", default={})
    eda_info = load_json(metrics_dir / "eda_acf_whitenoise_summary.json", default={})
    decomp_info = load_json(metrics_dir / "decomposition_summary.json", default={})
    qual_info = load_json(metrics_dir / "data_quality_summary.json", default={})
    viz_info = load_json(metrics_dir / "visualization_summary.json", default={})

    log_file = logs_dir / f"pipeline_{active_run_id}.log"
    if not log_file.exists():
        cands = sorted(logs_dir.glob("pipeline_*.log"))
        log_file = cands[-1] if cands else log_file
    log_text = log_file.read_text(encoding="utf-8", errors="ignore") if log_file.exists() else "Nhật ký chưa được tạo."

    return {
        "run_id": active_run_id,
        "cfg": cfg,
        "fig_dir": fig_dir,
        "data_info": data_info,
        "eda_info": eda_info,
        "decomp_info": decomp_info,
        "qual_info": qual_info,
        "viz_info": viz_info,
        "log_text": log_text,
    }


def _build_eda_table_rows(facts: Dict[str, Any]) -> tuple[List[List[str]], str]:
    r"""Chuẩn bị bảng hồ sơ EDA đầy đủ và ghi chú định hướng Train/Test dùng chung cho HTML, PDF và Markdown."""
    d = facts["data_info"]
    eda = facts["eda_info"].get("eda_profile") or d.get("eda_profile") or {}
    split_info = eda.get("recommended_split", {})

    n_obs = eda.get("n_obs", d.get("n_obs", 144))
    start_d = eda.get("start_date", d.get("start_date", "1949-01-01"))
    end_d = eda.get("end_date", d.get("end_date", "1960-12-01"))
    freq = eda.get("frequency", d.get("frequency", "MS (Monthly Start)"))
    dtype = eda.get("dtype", "int64")
    miss = eda.get("missing_count", d.get("missing_values", 0))
    dups = eda.get("duplicate_dates", 0)
    mean_v = eda.get("mean", round(float(d.get("mean", 0)), 2))
    std_v = eda.get("std", round(float(d.get("std", 0)), 2))
    cv_v = eda.get("cv_pct", round(std_v / mean_v * 100, 2) if mean_v else 0)
    skew_v = eda.get("skewness", "-")
    min_v, min_d = eda.get("min", d.get("min", 0)), eda.get("min_date", "11/1949")
    q25, med, q75 = eda.get("q25", "-"), eda.get("median", d.get("median", "-")), eda.get("q75", "-")
    max_v, max_d = eda.get("max", d.get("max", 0)), eda.get("max_date", "07/1960")
    growth = eda.get("growth_ratio", "-")
    fy, ly = eda.get("first_year", "1949"), eda.get("last_year", "1960")
    f_amp, l_amp = eda.get("first_year_amp", "-"), eda.get("last_year_amp", "-")

    ratio = split_info.get("ratio", "80% Train / 20% Test (shuffle=False)")
    tr_n = split_info.get("train_n", int(n_obs * 0.8))
    te_n = split_info.get("test_n", n_obs - tr_n)
    tr_rng = split_info.get("train_range", "")
    te_rng = split_info.get("test_range", "")
    split_txt = (
        f"{ratio} | Train: {tr_n} tháng{f' ({tr_rng})' if tr_rng else ''} | "
        f"Test: {te_n} tháng{f' ({te_rng})' if te_rng else ''}"
    )
    note = split_info.get(
        "note",
        "Lab 01 tập trung phân tích cấu trúc, phân rã và làm sạch; bước chia Train/Test tuần tự được áp dụng ở Lab 02 để chống Data Leakage.",
    )

    rows = [
        ["Tên bộ dữ liệu & Nguồn", "AirPassengers (data/raw/AirPassengers.csv)"],
        ["Số quan sát / Tần suất / Kiểu dữ liệu", f"{n_obs} tháng | {freq} | DatetimeIndex & {dtype}"],
        ["Khoảng thời gian", f"{start_d} → {end_d}"],
        ["Chất lượng dữ liệu gốc", f"Missing = {miss} | Ngày trùng lặp = {dups} (Sạch 100%)"],
        ["Trung bình (Mean) / Độ lệch chuẩn (Std)", f"{mean_v} / {std_v} nghìn khách (CV = {cv_v}%, Skew = {skew_v})"],
        ["Min / Q25 / Median / Q75 / Max", f"{min_v} ({min_d}) / {q25} / {med} / {q75} / {max_v} ({max_d})"],
        ["Đà tăng trưởng & Biên độ mùa vụ", f"Tăng gấp {growth} lần; biên độ năm {fy}: {f_amp} → năm {ly}: {l_amp}"],
        ["Định hướng Train/Test (cho Lab 02)", split_txt],
    ]
    return rows, note


def render_markdown_report(facts: Dict[str, Any], md_path: Path) -> Path:
    r"""
    Purpose: Kết xuất báo cáo Markdown (LAB_01_REPORT.md) đồng bộ 100% số liệu với bản HTML/PDF, phủ đủ 7 mục Bài tập 6.
    Input-Output:
    - Input: facts (Dict[str, Any]), md_path (Path).
    - Output: Đường dẫn Path tới tệp LAB_01_REPORT.md đã ghi.
    How: Trích xuất các khóa từ facts và định dạng thành 7 mục Markdown chuẩn.
    """
    d = facts["data_info"]
    acf_r = facts["eda_info"].get("acf_pacf", {})
    wn_r = facts["eda_info"].get("white_noise", {})
    dec_r = facts["decomp_info"]
    stl_r = dec_r.get("stl", {})
    q_imp = facts["qual_info"].get("imputation", {})
    q_clean = facts["qual_info"].get("final_cleaning_metrics_on_damaged_points", {})
    viz_r = facts["viz_info"]

    eda_rows, eda_note = _build_eda_table_rows(facts)

    lines: List[str] = [
        "# BÁO CÁO KỸ THUẬT LAB 01: PHÂN TÍCH CHUỖI THỜI GIAN (CHƯƠNG 2)",
        "",
        f"- **Mã phiên chạy (Run ID)**: `{facts['run_id']}`",
        f"- **Dataset**: `data/raw/AirPassengers.csv` ({d.get('n_obs', 144)} quan sát, {d.get('start_date', '1949-01-01')} → {d.get('end_date', '1960-12-01')}, tần suất `{d.get('frequency', 'MS')}`)",
        "",
        "## 1. Mô tả & Khám phá Dữ liệu (EDA) — AirPassengers",
        "| Chỉ tiêu EDA | Giá trị trích xuất |",
        "| --- | --- |",
    ]
    lines.extend(f"| **{k}** | {v} |" for k, v in eda_rows)
    lines.extend([
        "",
        f"> {eda_note}",
        "",
        "## 2. Kiểm tra ACF / PACF (Bài tập 1 — 40 Lags)",
        f"- ACF(1) = **{acf_r.get('acf_lag1')}**, ACF(12) = **{acf_r.get('acf_lag12')}**, ACF(24) = **{acf_r.get('acf_lag24')}**, ACF(36) = **{acf_r.get('acf_lag36')}**.",
        f"- Số lag ACF vượt ngưỡng 95% (±{acf_r.get('conf_bound_95')}): **{acf_r.get('significant_acf_count')}/{acf_r.get('nlags')}** → chuỗi có tự tương quan rất mạnh, xu hướng tăng và chu kỳ mùa vụ 12 tháng rõ rệt.",
        "",
        "## 3. Kiểm tra White Noise (Bài tập 2 — 1000 điểm)",
        f"- White Noise mô phỏng: **{wn_r.get('inside_95_pct')}%** lags nằm trong dải tin cậy 95%, Ljung-Box p-value (lag 20) = **{wn_r.get('ljung_box_wn_p20')}** (> 0.05).",
        "",
    ])
    for item in wn_r.get("comparison_5_lines", []):
        lines.append(f"- {item}")

    lines.extend([
        "",
        "## 4. Phân rã Chuỗi Thời gian (Bài tập 3 — Classical Multiplicative & STL)",
        f"- Phương sai phần dư Classical: Additive = **{dec_r.get('classical_comparison', {}).get('additive_resid_variance')}** vs Multiplicative = **{dec_r.get('classical_comparison', {}).get('multiplicative_resid_variance')}** → Mô hình **Multiplicative** phù hợp hơn.",
        f"- Độ mạnh Xu hướng $F_T$ = **{stl_r.get('trend_strength')}** | Độ mạnh Mùa vụ $F_S$ = **{stl_r.get('seasonal_strength')}**.",
        f"- Kiểm định phần dư STL: Mean = **{stl_r.get('mean')}**, Std = **{stl_r.get('std')}**, Ljung-Box p(10) = **{stl_r.get('ljung_box_p10')}**, p(20) = **{stl_r.get('ljung_box_p20')}**.",
        "",
        "## 5. Vấn đề Chất lượng Dữ liệu (Bài tập 4 — 11 NaN & 3 Outlier giả lập)",
        f"- Điền khuyết: Forward-fill (MAE={q_imp.get('forward_fill', {}).get('MAE')}, RMSE={q_imp.get('forward_fill', {}).get('RMSE')}) vs Time Interpolation (MAE={q_imp.get('time_interpolation', {}).get('MAE')}, RMSE={q_imp.get('time_interpolation', {}).get('RMSE')}) → Tốt hơn: **{q_imp.get('best_method')}**.",
        f"- Làm sạch cuối cùng trên 14 điểm bị làm hỏng (11 NaN + 3 Outlier): MAE = **{q_clean.get('MAE')}**, RMSE = **{q_clean.get('RMSE')}**, MAPE = **{q_clean.get('MAPE')}%**.",
        "",
        "## 6. Ý nghĩa Trực quan hóa (Bài tập 5)",
        f"- Tương quan giữa Rolling Mean và Rolling Std (window=12): **r = {viz_r.get('rolling_mean_std_corr')}** (phương sai tăng theo xu hướng).",
        f"- Cao điểm: **Tháng {viz_r.get('peak_month')}** (TB {viz_r.get('peak_month_avg')}, Seasonal Index = {viz_r.get('peak_seasonal_index')}) | Thấp điểm: **Tháng {viz_r.get('trough_month')}** (TB {viz_r.get('trough_month_avg')}, Seasonal Index = {viz_r.get('trough_seasonal_index')}).",
        f"- Tăng trưởng kép hàng năm (CAGR): **{viz_r.get('annual_cagr_pct')}%/năm**.",
        "",
        "## 7. Kết luận: Dữ liệu có phù hợp để dự báo không?",
        "- **RẤT PHÙ HỢP**: Tập dữ liệu AirPassengers không phải nhiễu trắng, sạch (0 missing gốc), có xu hướng tăng dài hạn và cấu trúc mùa vụ 12 tháng cực kỳ ổn định. Vì phương sai tăng theo xu hướng (Multiplicative), nên áp dụng biến đổi log và mô hình **SARIMA(p,d,q)(P,D,Q,12)** hoặc **Holt-Winters Multiplicative** ở Lab 02.",
        "",
    ])

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def render_html(run_id: Optional[str] = None, out_file: Optional[Path] = None) -> Path:
    r"""
    Purpose: Kết xuất tệp báo cáo LAB_01_REPORT.html chuẩn giao diện học thuật UTH, nhúng đủ 15 biểu đồ gốc Base64 và tự động xuất PDF + MD.
    Input-Output:
    - Input: run_id (Optional[str]), out_file (Optional[Path]).
    - Output: Đường dẫn Path tới tệp LAB_01_REPORT.html đã tạo.
    How: Thu thập facts, lắp vào template HTML5 6 Stages, ghi ra reports/lab01/, đồng bộ docs/lab01/ và xuất PDF.
    """
    facts = collect_report_facts(run_id)
    cfg = facts["cfg"]
    active_run_id = facts["run_id"]
    fig_dir: Path = facts["fig_dir"]

    def _b64(name: str) -> str:
        return load_b64(fig_dir / name)

    d = facts["data_info"]
    acf_r = facts["eda_info"].get("acf_pacf", {})
    wn_r = facts["eda_info"].get("white_noise", {})
    dec_r = facts["decomp_info"]
    stl_r = dec_r.get("stl", {})
    q_imp = facts["qual_info"].get("imputation", {})
    q_out_m = facts["qual_info"].get("outliers", {}).get("methods_comparison", {})
    q_clean = facts["qual_info"].get("final_cleaning_metrics_on_damaged_points", {})
    viz_r = facts["viz_info"]
    n_figs = len(list(fig_dir.glob("*.png")))

    eda_rows, eda_note = _build_eda_table_rows(facts)
    eda_table_html = "".join(
        f"<tr><td><b>{html.escape(k)}</b></td><td>{html.escape(str(v))}</td></tr>\n"
        for k, v in eda_rows
    )

    wn_lines_html = "".join(f"<li>{html.escape(line)}</li>" for line in wn_r.get("comparison_5_lines", []))

    outlier_rows_html = ""
    for m_name, m_val in q_out_m.items():
        row_cls = ' class="row-featured"' if m_val.get("f1", 0) == 1.0 else ""
        outlier_rows_html += (
            f"<tr{row_cls}><td><b>{html.escape(m_name)}</b></td>"
            f"<td>{m_val.get('detected_count', 0)}</td>"
            f"<td><b>{m_val.get('tp', 0)}/3</b></td>"
            f"<td>{m_val.get('fp', 0)}</td>"
            f"<td>{m_val.get('fn', 0)}</td>"
            f"<td>{m_val.get('precision', 0):.2f}</td>"
            f"<td>{m_val.get('recall', 0):.2f}</td>"
            f"<td><b>{m_val.get('f1', 0):.2f}</b></td></tr>\n"
        )

    html_content = f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Lab 01 — Phân Tích Chuỗi Thời Gian (Chương 2 · AirPassengers)</title>
  <style>
    @page {{ size: A4; margin: 11mm 12mm; }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; scroll-padding-top: 86px; }}
    body {{ margin: 0; background: #f3f5f0; color: #243d32; font: 15.5px/1.68 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }}
    main {{ max-width: 1180px; margin: auto; padding: 26px; }}
    header {{ background: linear-gradient(135deg, #183f35 0%, #102a23 100%); border-radius: 20px; padding: 40px 44px; color: white; margin-bottom: 18px; }}
    header h1 {{ font-size: clamp(24px, 3.6vw, 38px); line-height: 1.2; margin: 12px 0; }}
    header p {{ color: #d6e5d9; max-width: 920px; }}
    .eyebrow {{ font-size: 11px; letter-spacing: .14em; text-transform: uppercase; color: #678273; font-weight: 600; }}
    header .eyebrow {{ color: #bcd8b5; }}
    .kpi-strip {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; margin-top: 20px; }}
    .kpi-box {{ background: rgba(255,255,255,.08); border: 1px solid rgba(188,216,181,.28); border-radius: 12px; padding: 14px 16px; }}
    .kpi-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: .08em; color: #bcd8b5; }}
    .kpi-val {{ font-size: 20px; font-weight: 700; color: #fff; margin: 4px 0 2px; }}
    .kpi-sub {{ font-size: 12.5px; color: #cde2d2; }}
    nav {{ position: sticky; top: 0; z-index: 100; background: #f3f5f0f5; backdrop-filter: blur(8px); display: flex; gap: 8px; flex-wrap: wrap; padding: 12px 0; border-bottom: 1px solid #d7e2d7; margin-bottom: 24px; }}
    nav a {{ font-size: 13px; text-decoration: none; padding: 6px 11px; border-radius: 6px; color: #244d37; background: #e7efe5; }}
    nav a:hover {{ background: #1d5843; color: #fff; }}
    nav button {{ margin-left: auto; cursor: pointer; background: white; border: 1px solid #c8d8ca; border-radius: 7px; padding: 7px 14px; color: #224c38; font-weight: 600; }}
    .stage {{ border-left: 2.5px solid #c2d5c3; margin: 0 0 38px 22px; padding: 0 0 20px 28px; }}
    .stage-head {{ display: flex; align-items: center; gap: 16px; margin-bottom: 14px; }}
    .number {{ display: grid; place-items: center; flex-shrink: 0; width: 46px; height: 46px; margin-left: -53px; background: #1d5843; color: white; border: 5px solid #f3f5f0; border-radius: 50%; font-weight: 700; }}
    .stage h2 {{ font-size: 23px; margin: 4px 0; }}
    .card-grid {{ display: grid; grid-template-columns: 1fr; gap: 18px; margin: 18px 0; }}
    .image-card {{ background: white; border: 1px solid #dce5d8; border-radius: 12px; overflow: hidden; margin: 0; }}
    .image-card img {{ display: block; width: 100%; max-height: 760px; object-fit: contain; background: #fff; padding: 8px; }}
    .image-card figcaption {{ padding: 12px 16px; font-size: 13.8px; color: #476650; border-top: 1px solid #edf1e9; background: #fafbfa; }}
    .image-card figcaption b {{ display: block; color: #244e36; font-size: 14.5px; margin-bottom: 3px; }}
    .takeaway {{ padding: 15px 20px; background: #e7f0df; border-left: 4px solid #2a7950; border-radius: 6px; font-size: 14.5px; margin: 16px 0; }}
    table {{ width: 100%; border-collapse: collapse; margin: 14px 0; font-size: 13.8px; background: white; }}
    th, td {{ padding: 9px 12px; border: 1px solid #d7e3d6; text-align: left; }}
    th {{ background: #eaf1e8; color: #214c37; }}
    tr.row-featured {{ background: #eef6ec; }}
    details {{ background: white; border: 1px solid #d9e3d8; border-radius: 10px; margin: 14px 0; }}
    summary {{ padding: 13px 16px; cursor: pointer; font-weight: 600; color: #315740; }}
    pre {{ padding: 14px; background: #192f26; color: #e0eee2; border-radius: 8px; overflow: auto; font-size: 12.5px; margin: 0 16px 16px; }}
    footer {{ font-size: 13px; color: #667b6a; padding: 20px 0; border-top: 1px solid #dbe4db; margin-top: 36px; text-align: center; }}
    @media print {{
      body {{ background: #fff !important; font-size: 10pt !important; -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }}
      nav, details {{ display: none !important; }}
      .image-card, .takeaway, tr {{ break-inside: avoid; page-break-inside: avoid; }}
      .image-card img {{ max-height: 280px !important; }}
    }}
  </style>
</head>
<body>
<main>
<header>
  <div class="eyebrow">TRƯỜNG ĐẠI HỌC GIAO THÔNG VẬN TẢI TP.HCM (UTH) · TIME SERIES ANALYSIS · LAB 01 (CHƯƠNG 2)</div>
  <h1>Khám Phá Chuỗi Thời Gian, White Noise, Phân Rã Chuỗi &amp; Chất Lượng Dữ Liệu</h1>
  <p>Báo cáo tự động đồng bộ 100% số liệu thực nghiệm và trọn bộ {n_figs} biểu đồ gốc trên tập dữ liệu <code>AirPassengers.csv</code> (01/1949 – 12/1960, Run ID: <code>{html.escape(active_run_id)}</code>).</p>
  <div class="kpi-strip">
    <div class="kpi-box"><div class="kpi-label">Quy mô Dataset</div><div class="kpi-val">{d.get('n_obs', 144)} Tháng</div><div class="kpi-sub">Mean {d.get('mean', 0):.1f} · Min {d.get('min', 0):.0f} · Max {d.get('max', 0):.0f}</div></div>
    <div class="kpi-box"><div class="kpi-label">Tự tương quan &amp; Mùa vụ</div><div class="kpi-val">ACF(1) = {acf_r.get('acf_lag1')}</div><div class="kpi-sub">ACF(12) = {acf_r.get('acf_lag12')} · Chu kỳ s = 12</div></div>
    <div class="kpi-box"><div class="kpi-label">Độ mạnh Phân rã STL</div><div class="kpi-val">F<sub>T</sub> = {stl_r.get('trend_strength')}</div><div class="kpi-sub">F<sub>S</sub> = {stl_r.get('seasonal_strength')} · Multiplicative</div></div>
    <div class="kpi-box"><div class="kpi-label">Chất lượng &amp; Khôi phục</div><div class="kpi-val">RMSE = {q_clean.get('RMSE')}</div><div class="kpi-sub">MAPE = {q_clean.get('MAPE')}% · {n_figs} Biểu đồ</div></div>
  </div>
</header>

<nav>
  <a href="#s1">01 · Dataset &amp; ACF/PACF</a>
  <a href="#s2">02 · White Noise</a>
  <a href="#s3">03 · Phân rã Chuỗi</a>
  <a href="#s4">04 · Chất lượng Dữ liệu</a>
  <a href="#s5">05 · Trực quan hóa</a>
  <a href="#s6">06 · Kết luận &amp; Log</a>
  <button type="button" onclick="window.print()">In / Xuất PDF</button>
</nav>

<section class="stage" id="s1">
  <div class="stage-head"><span class="number">01</span><div><div class="eyebrow">BÀI TẬP 1 · KHÁM PHÁ &amp; TỰ TƯƠNG QUAN</div><h2>Mô tả Dataset AirPassengers &amp; Phân tích ACF / PACF (40 Lags)</h2></div></div>
  <table>
    <thead><tr><th>Chỉ tiêu Khám phá Dữ liệu (EDA)</th><th>Giá trị trích xuất tự động</th></tr></thead>
    <tbody>{eda_table_html}</tbody>
  </table>
  <div class="takeaway"><b>Nhận định EDA &amp; Định hướng chia Train/Test:</b> {html.escape(eda_note)}</div>
  <div class="card-grid">
    <figure class="image-card"><img src="{_b64('01_full_series_overview.png')}" alt="Full Series"><figcaption><b>Hình 1.1: Tổng quan chuỗi AirPassengers (01/1949 – 12/1960)</b>Đánh dấu trung bình ({d.get('mean', 0):.1f}), điểm thấp nhất Min ({d.get('min', 0):.0f}) và cao nhất Max ({d.get('max', 0):.0f}).</figcaption></figure>
    <figure class="image-card"><img src="{_b64('02_acf_pacf_airpassengers.png')}" alt="ACF PACF"><figcaption><b>Hình 1.2: Hàm tự tương quan ACF và PACF đến 40 lags</b>ACF(1) = {acf_r.get('acf_lag1')}, ACF(12) = {acf_r.get('acf_lag12')}, ACF(24) = {acf_r.get('acf_lag24')} suy giảm chậm kèm đỉnh chu kỳ 12 tháng.</figcaption></figure>
  </div>
</section>

<section class="stage" id="s2">
  <div class="stage-head"><span class="number">02</span><div><div class="eyebrow">BÀI TẬP 2 · KIỂM ĐỊNH NHIỄU TRẮNG</div><h2>Mô phỏng White Noise (1000 điểm) &amp; So sánh ACF với AirPassengers</h2></div></div>
  <div class="card-grid">
    <figure class="image-card"><img src="{_b64('03_white_noise_series.png')}" alt="White Noise"><figcaption><b>Hình 2.1: Chuỗi White Noise giả lập (1000 quan sát)</b>Dao động ngẫu nhiên quanh 0 với Mean = {wn_r.get('mean')}, Std = {wn_r.get('std')}.</figcaption></figure>
    <figure class="image-card"><img src="{_b64('04_white_noise_vs_airpassengers_acf.png')}" alt="WN vs AP ACF"><figcaption><b>Hình 2.2: Đối chiếu ACF — White Noise ({wn_r.get('inside_95_pct')}% trong dải 95%) vs. AirPassengers</b>Kiểm định Ljung-Box (lag 20): White Noise p = {wn_r.get('ljung_box_wn_p20')} (&gt; 0.05) vs. AirPassengers p = {wn_r.get('ljung_box_ap_p20'):.2e}.</figcaption></figure>
  </div>
  <div class="takeaway"><b>5 điểm khác biệt cốt lõi giữa White Noise và AirPassengers:</b><ul>{wn_lines_html}</ul></div>
</section>

<section class="stage" id="s3">
  <div class="stage-head"><span class="number">03</span><div><div class="eyebrow">BÀI TẬP 3 · PHÂN RÃ CHUỖI THỜI GIAN</div><h2>Classical Multiplicative Decomposition, STL Loess &amp; Kiểm định Phần dư</h2></div></div>
  <div class="card-grid">
    <figure class="image-card"><img src="{_b64('05_classical_decomposition_multiplicative.png')}" alt="Classical"><figcaption><b>Hình 3.1: Classical Decomposition (Multiplicative Model)</b>Phương sai phần dư Additive = {dec_r.get('classical_comparison', {}).get('additive_resid_variance')} so với Multiplicative = {dec_r.get('classical_comparison', {}).get('multiplicative_resid_variance')}.</figcaption></figure>
    <figure class="image-card"><img src="{_b64('06_stl_decomposition.png')}" alt="STL"><figcaption><b>Hình 3.2: STL Decomposition (Loess Regression, robust=True)</b>Độ mạnh xu hướng F<sub>T</sub> = {stl_r.get('trend_strength')} và độ mạnh mùa vụ F<sub>S</sub> = {stl_r.get('seasonal_strength')}.</figcaption></figure>
    <figure class="image-card"><img src="{_b64('07_stl_residual_diagnostics.png')}" alt="STL Residual"><figcaption><b>Hình 3.3: Phân phối Residual (kèm đường chuẩn N({stl_r.get('mean')}, {stl_r.get('std')}²)) &amp; ACF 30 lags</b>Ljung-Box p(10) = {stl_r.get('ljung_box_p10')}, p(20) = {stl_r.get('ljung_box_p20')}.</figcaption></figure>
  </div>
</section>

<section class="stage" id="s4">
  <div class="stage-head"><span class="number">04</span><div><div class="eyebrow">BÀI TẬP 4 · KIỂM TRA &amp; XỬ LÝ CHẤT LƯỢNG DỮ LIỆU</div><h2>Điền khuyết 11 điểm NaN &amp; Phát hiện, Thay thế 3 Outlier giả lập</h2></div></div>
  <table>
    <thead><tr><th>Phương pháp phát hiện Outlier</th><th>Số điểm gắn cờ</th><th>Bắt đúng (TP)</th><th>Báo nhầm (FP)</th><th>Bỏ sót (FN)</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>
    <tbody>{outlier_rows_html}</tbody>
  </table>
  <div class="card-grid">
    <figure class="image-card"><img src="{_b64('08_missing_imputation_eval.png')}" alt="Imputation"><figcaption><b>Hình 4.1: Toàn chuỗi &amp; Phóng to khoảng thiếu (03/1955 – 12/1955)</b>Forward-fill RMSE = {q_imp.get('forward_fill', {}).get('RMSE')} vs. Time Interpolation RMSE = {q_imp.get('time_interpolation', {}).get('RMSE')}.</figcaption></figure>
    <figure class="image-card"><img src="{_b64('09_rolling_outliers_detection.png')}" alt="Rolling Outliers"><figcaption><b>Hình 4.2: Phát hiện Outlier bằng Rolling mean ± 3 Rolling std &amp; Rolling variance</b></figcaption></figure>
    <figure class="image-card"><img src="{_b64('10_stl_log_outliers_detection.png')}" alt="STL Outliers"><figcaption><b>Hình 4.3: Z-score / IQR trên chuỗi gốc (A) vs. trên phần dư STL thang log (B)</b></figcaption></figure>
    <figure class="image-card"><img src="{_b64('11_before_after_cleaning.png')}" alt="Before After"><figcaption><b>Hình 4.4: Đối chiếu 3 giai đoạn — Trước xử lý → Điền missing → Làm sạch đầy đủ</b>Sai số khôi phục trên 14 điểm bị làm hỏng: MAE = {q_clean.get('MAE')}, RMSE = {q_clean.get('RMSE')}, MAPE = {q_clean.get('MAPE')}%.</figcaption></figure>
    <figure class="image-card"><img src="{_b64('12_residuals_comparison.png')}" alt="Residuals Cmp"><figcaption><b>Hình 4.5: Phần dư STL (thang log) &amp; Boxplot Trước và Sau xử lý Outlier</b></figcaption></figure>
  </div>
</section>

<section class="stage" id="s5">
  <div class="stage-head"><span class="number">05</span><div><div class="eyebrow">BÀI TẬP 5 · TRỰC QUAN HÓA CHUYÊN SÂU</div><h2>Rolling Statistics, Seasonal Trends &amp; Dual Seasonality Heatmaps</h2></div></div>
  <div class="card-grid">
    <figure class="image-card"><img src="{_b64('13_rolling_statistics.png')}" alt="Rolling"><figcaption><b>Hình 5.1: Chuỗi gốc, Rolling Mean &amp; Rolling Standard Deviation (window=12)</b>Hệ số tương quan r(Mean, Std) = {viz_r.get('rolling_mean_std_corr')} xác nhận phương sai giãn nở theo xu hướng.</figcaption></figure>
    <figure class="image-card"><img src="{_b64('14_seasonal_trends.png')}" alt="Seasonal"><figcaption><b>Hình 5.2: Seasonal Plot (1949–1960) &amp; Chỉ số mùa vụ trung bình</b>Cao điểm Tháng {viz_r.get('peak_month')} (SI = {viz_r.get('peak_seasonal_index')}), thấp điểm Tháng {viz_r.get('trough_month')} (SI = {viz_r.get('trough_seasonal_index')}).</figcaption></figure>
    <figure class="image-card"><img src="{_b64('15_heatmaps_seasonality.png')}" alt="Heatmaps"><figcaption><b>Hình 5.3: Heatmap số hành khách (tháng × năm) &amp; Heatmap tỷ lệ mùa vụ</b>Tốc độ tăng trưởng kép hàng năm CAGR = {viz_r.get('annual_cagr_pct')}%/năm.</figcaption></figure>
  </div>
</section>

<section class="stage" id="s6">
  <div class="stage-head"><span class="number">06</span><div><div class="eyebrow">BÀI TẬP 6 · KẾT LUẬN TỔNG HỢP &amp; NHẬT KÝ THỰC THI</div><h2>Đánh giá Mức độ Phù hợp cho Bài toán Dự báo</h2></div></div>
  <div class="takeaway">
    <b>Kết luận:</b> Tập dữ liệu <code>AirPassengers</code> <b>RẤT PHÙ HỢP</b> để xây dựng mô hình dự báo chuỗi thời gian vì: (1) hoàn toàn không phải nhiễu trắng (tự tương quan rất mạnh, Ljung-Box p ≪ 0.05); (2) dữ liệu gốc đầy đủ 144 tháng liên tục, không thiếu mốc thời gian; (3) xu hướng tăng ổn định (CAGR = {viz_r.get('annual_cagr_pct')}%/năm) và chu kỳ mùa vụ 12 tháng lặp lại cực kỳ nhất quán. Do biên độ mùa vụ giãn nở theo xu hướng (Multiplicative), bước tiếp theo ở Lab 02 nên dùng biến đổi log kết hợp mô hình <b>SARIMA(p,d,q)(P,D,Q,12)</b>.
  </div>
  <details>
    <summary>Xem Nhật ký Thực thi Tự động (logs/lab01/pipeline_{html.escape(active_run_id)}.log)</summary>
    <pre><code>{html.escape(facts['log_text'])}</code></pre>
  </details>
</section>

<footer>Trường Đại học Giao thông Vận tải TP.HCM (UTH) &copy; 2026 · Báo cáo Kỹ thuật Lab 01 · Run ID: <code>{html.escape(active_run_id)}</code></footer>
</main>
</body>
</html>
"""

    reports_dir = get_path("reports_dir", cfg)
    docs_dir = get_path("docs_dir", cfg)
    reports_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    target_html = out_file if out_file is not None else (reports_dir / "LAB_01_REPORT.html")
    target_html.write_text(html_content, encoding="utf-8")
    for extra in (docs_dir / "LAB_01_REPORT.html", reports_dir / f"report_{active_run_id}.html"):
        shutil.copy2(target_html, extra)

    target_md = reports_dir / "LAB_01_REPORT.md"
    render_markdown_report(facts, target_md)
    for extra in (docs_dir / "LAB_01_REPORT.md", reports_dir / f"report_{active_run_id}.md"):
        shutil.copy2(target_md, extra)

    target_pdf = reports_dir / "LAB_01_REPORT.pdf"
    if export_html_to_pdf(target_html, target_pdf, facts=facts, fig_dir=fig_dir):
        for extra in (docs_dir / "LAB_01_REPORT.pdf", reports_dir / f"report_{active_run_id}.pdf"):
            shutil.copy2(target_pdf, extra)
        print(f"[RENDER_REPORT] Đã xuất PDF : {target_pdf} ({target_pdf.stat().st_size:,} bytes)")

    print(f"[RENDER_REPORT] Đã xuất HTML: {target_html} ({target_html.stat().st_size:,} bytes)")
    print(f"[RENDER_REPORT] Đã xuất MD  : {target_md} ({target_md.stat().st_size:,} bytes)")
    return target_html


def main(run_id: Optional[str] = None) -> Dict[str, Any]:
    r"""
    Purpose: Điểm vào chính để kết xuất đồng bộ 3 báo cáo HTML, PDF, Markdown cho Lab 01.
    Input-Output:
    - Input: run_id (Optional[str]).
    - Output: Từ điển chứa đường dẫn 3 tệp báo cáo.
    How: Gọi render_html(run_id=run_id) và trả về đường dẫn 3 tệp đầu ra.
    """
    html_path = render_html(run_id=run_id)
    reports_dir = get_path("reports_dir", load_config())
    return {
        "html_report": str(html_path),
        "pdf_report": str(reports_dir / "LAB_01_REPORT.pdf"),
        "md_report": str(reports_dir / "LAB_01_REPORT.md"),
    }


if __name__ == "__main__":
    main()