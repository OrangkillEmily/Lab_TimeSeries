"""
Module tổng hợp báo cáo thực nghiệm tự động (`src/lab04/report.py`) cho dự án Lab 04 Time Series.

Trách nhiệm của module:
1. Đọc kết quả thật từ `reports/lab04/metrics/evaluation_latest.json`, `eda_summary.json`,
   `preprocessing_summary.json`, `models/lab04/training_metadata.json` và `logs/lab04/pipeline_<run_id>.log`.
2. Tự động biên soạn báo cáo Markdown chuẩn học thuật UTH tại:
   - `reports/lab04/LAB_04_REPORT.md`
   - `reports/lab04/experiment_summary.md`
   - `reports/lab04/report_<run_id>.md`
   - `docs/lab04/LAB_04_REPORT.md`
3. Ủy quyền cho `script/lab04/render_report.py` sinh báo cáo HTML tương tác (`LAB_04_REPORT.html`)
   và báo cáo PDF (`LAB_04_REPORT.pdf`) đồng bộ ở cả `reports/lab04/` và `docs/lab04/`.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.lab04.tracking import PROGRESS_DIR, update_progress, write_log
from src.lab04.utils import ensure_dir, get_project_root, load_config, load_json

ROOT_DIR = get_project_root()


def _load_render_report_module() -> Any:
    """
    What: Nạp động module `script/lab04/render_report.py`.
    Why: Hỗ trợ gọi `collect_report_facts()` và `render_html()` dù `script/lab04/` nằm ngoài `src/lab04/`.
    Input: Không có.
    Output:
        - Module `render_report`.
    How: Dùng `importlib.util.spec_from_file_location`.
    """
    script_path = ROOT_DIR / "script" / "lab04" / "render_report.py"
    spec = importlib.util.spec_from_file_location("lab04_render_report", script_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def get_latest_run_id() -> str:
    """
    What: Lấy mã `run_id` mới nhất từ `progress_results/lab04/`.
    Why: Cho phép chạy `python -m src.lab04.report` hoặc `python script/lab04/render_report.py` độc lập.
    Input: Không có.
    Output:
        - str: Mã `run_id` gần nhất.
    How: Sắp xếp các tệp `progress_*.json` theo thời gian sửa đổi (`st_mtime`).
    """
    files = sorted(PROGRESS_DIR.glob("progress_*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        return "latest"
    return files[-1].stem.replace("progress_", "", 1)


def generate_markdown_report(facts: Dict[str, Any], fig_prefix: str = "figures") -> str:
    """
    What: Xây dựng nội dung báo cáo kỹ thuật định dạng Markdown (`LAB_04_REPORT.md`) từ số liệu thực tế.
    Why: Đảm bảo 100% số liệu trong bảng `Hold-out Test`, `5-Fold TimeSeriesSplit`, `Walk-Forward`
         và `PEMS-BAY DCRNN` khớp tuyệt đối với kết quả chạy code.
    Input:
        - facts (Dict[str, Any]): Toàn bộ dữ liệu thu thập từ `collect_report_facts()`.
        - fig_prefix (str): Tiền tố đường dẫn thư mục ảnh (`"figures"` hoặc `"../reports/lab04/figures"`).
    Output:
        - str: Nội dung Markdown hoàn chỉnh.
    How: Định dạng các bảng Markdown cho 5 chỉ số (`MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²`) và
         6 câu trả lời phân tích ở Mục 11.
    """
    eval_data = facts.get("evaluation", {})
    prep_data = facts.get("preprocessing", {})
    train_data = facts.get("training", {})
    eda_data = facts.get("eda", {})

    run_id = eval_data.get("run_id", "latest")
    hw = eval_data.get("hardware", train_data.get("hardware", {}))
    split_counts = prep_data.get("chronological_split_counts", {})
    holdout = eval_data.get("holdout_test_metrics", {})
    wf_metrics = eval_data.get("walk_forward_backtesting", {}).get("metrics", {})
    kfold_sum = eval_data.get("kfold_cv", {}).get("summary_by_model", {})
    pems_eval = eval_data.get("pems_bay_dcrnn", {})
    q_answers = eval_data.get("section11_analytical_answers", {})

    lines: List[str] = [
        "# BÁO CÁO THỰC NGHIỆM LAB 04: DỰ BÁO CHUỖI THỜI GIAN ĐA BIẾN (MULTIVARIATE TIME SERIES FORECASTING)",
        "",
        f"- **Học phần:** Phân tích Chuỗi Thời gian (Time Series Analysis — UTH)",
        f"- **Phiên chạy (`Run_ID`):** `{run_id}`",
        f"- **Thiết bị tính toán (`Auto-GPU`):** `{hw.get('device_name', 'CPU')}` (`device={hw.get('torch_device', 'cpu')}`, `AMP_FP16={hw.get('amp_enabled', False)}`)",
        f"- **Phân chia dữ liệu thời gian (`shuffle=False`):** Train 70% (`{split_counts.get('train_hours_70pct', 0)}`h) | Validation 15% (`{split_counts.get('val_hours_15pct', 0)}`h) | Test 15% (`{split_counts.get('test_hours_15pct', 0)}`h) kết hợp `5-Fold TimeSeriesSplit`",
        "",
        "## 1. Kết quả Khám phá Dữ liệu (EDA — Mục 3)",
        "",
        f"- **Biến thời tiết ảnh hưởng mạnh nhất tới phụ tải điện (`load`):** `{eda_data.get('dominant_weather_variable', 'temperature')}`",
        f"- **Tổng số giá trị khuyết (`Missing values`):** `{eda_data.get('missing_values_total', 0)}`",
        "",
        f"![Load Overview]({fig_prefix}/load_timeseries_overview.png)",
        f"![Load vs Temperature]({fig_prefix}/load_vs_temperature_scatter.png)",
        f"![Correlation Heatmap]({fig_prefix}/correlation_heatmap.png)",
        f"![Seasonality]({fig_prefix}/seasonality_daily_weekly.png)",
        "",
        "## 2. Tiền xử lý, Chia Dữ liệu Thời gian (70/15/15) & 5-Fold TimeSeriesSplit (Mục 4 & Yêu cầu 4)",
        "",
        f"![Train Val Test Split]({fig_prefix}/train_val_test_split.png)",
        f"![5-Fold Timeline]({fig_prefix}/kfold_splits_timeline.png)",
        "",
        "## 3. Kết quả Đánh giá 5 Chỉ số (`MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²`)",
        "",
        "### 3.1. Bảng Đối sánh trên Tập Kiểm tra Độc lập (`Hold-out Test Set 15%` — Mục 10 & Yêu cầu 6)",
        "",
        "| Mô hình (Model) | Test MSE (↓) | Test RMSE (↓) | Test MAE (↓) | Test MAPE (↓) | Test R² Score (↑) | Train Time (s) | Device |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    for m_name in ["VAR", "XGBoost", "LSTM"]:
        m = holdout.get(m_name, {})
        if m:
            lines.append(
                f"| **{m_name}** | {m.get('MSE', 0.0):.4f} | {m.get('RMSE', 0.0):.4f} | "
                f"{m.get('MAE', 0.0):.4f} | {m.get('MAPE', 0.0):.2f}% | **{m.get('R2', 0.0):.4f}** | "
                f"{m.get('Training_Time_Sec', 0.0):.2f}s | `{m.get('Device', 'cpu')}` |"
            )

    lines.extend([
        "",
        "### 3.2. Bảng Đối sánh Kiểm định Chéo 5-Fold Theo Thời gian (`5-Fold TimeSeriesSplit Mean ± Std` — Yêu cầu 4)",
        "",
        "| Mô hình (Model) | 5-Fold MSE (Mean ± Std) | 5-Fold RMSE (Mean ± Std) | 5-Fold MAE (Mean ± Std) | 5-Fold MAPE (Mean ± Std) | 5-Fold R² Score (Mean ± Std) |",
        "|---|---:|---:|---:|---:|---:|",
    ])

    for m_name in ["VAR", "XGBoost", "LSTM"]:
        kf = kfold_sum.get(m_name, {})
        if kf:
            lines.append(
                f"| **{m_name}** | {kf.get('MSE', {}).get('formatted', 'N/A')} | "
                f"{kf.get('RMSE', {}).get('formatted', 'N/A')} | {kf.get('MAE', {}).get('formatted', 'N/A')} | "
                f"{kf.get('MAPE', {}).get('formatted', 'N/A')}% | **{kf.get('R2', {}).get('formatted', 'N/A')}** |"
            )

    lines.extend([
        "",
        "### 3.3. Bảng Kết quả Walk-Forward Backtesting (Mục 9: `Train -> predict t+1 -> append -> predict t+2`)",
        "",
        "| Mô hình (Model) | Walk-Forward MSE (↓) | Walk-Forward RMSE (↓) | Walk-Forward MAE (↓) | Walk-Forward MAPE (↓) | Walk-Forward R² (↑) |",
        "|---|---:|---:|---:|---:|---:|",
    ])

    for m_name in ["VAR", "XGBoost", "LSTM"]:
        wm = wf_metrics.get(m_name, {})
        if wm:
            lines.append(
                f"| **{m_name}** | {wm.get('MSE', 0.0):.4f} | {wm.get('RMSE', 0.0):.4f} | "
                f"{wm.get('MAE', 0.0):.4f} | {wm.get('MAPE', 0.0):.2f}% | **{wm.get('R2', 0.0):.4f}** |"
            )

    lines.extend([
        "",
        f"![Actual vs Predicted]({fig_prefix}/actual_vs_predicted_all.png)",
        f"![Walk-Forward Backtesting]({fig_prefix}/walk_forward_backtest.png)",
        f"![K-Fold Comparison]({fig_prefix}/kfold_cv_comparison.png)",
        f"![LSTM Convergence]({fig_prefix}/lstm_train_vs_val_loss.png)",
        "",
        "## 4. Giải đáp 6 Câu hỏi Phân tích Học thuật (Mục 11 Đề bài)",
        "",
        f"1. **Biến thời tiết nào ảnh hưởng mạnh nhất đến điện năng?**  \n   {q_answers.get('Q1_Dominant_Weather_Variable', '')}",
        f"2. **Khi nào VAR thất bại?**  \n   {q_answers.get('Q2_When_VAR_Fails', '')}",
        f"3. **XGBoost học được thông tin gì từ thời tiết?**  \n   {q_answers.get('Q3_What_XGBoost_Learns_From_Weather', '')}",
        f"4. **LSTM học được chu kỳ nào (ngày, tuần)?**  \n   {q_answers.get('Q4_What_Cycles_LSTM_Learns', '')}",
        f"5. **LSTM có bị overfitting không?**  \n   {q_answers.get('Q5_Does_LSTM_Overfit', '')}",
        f"6. **Sự khác biệt giữa Machine Learning và Deep Learning trong bài toán này?**  \n   {q_answers.get('Q6_Difference_Between_ML_And_DL', '')}",
        "",
        "## 5. Kết quả Bài tập 12 Nâng cao: Dự báo Giao thông Đa bước với DCRNN trên PEMS-BAY (`pems-bay.h5` & `sensor_graph/adj_mx_bay.pkl`)",
        "",
        "| Chân trời Dự báo (Horizon) | MSE (mph² ↓) | RMSE (mph ↓) | MAE (mph ↓) | MAPE (% ↓) | R² Score (↑) |",
        "|---|---:|---:|---:|---:|---:|",
    ])

    for h_row in pems_eval.get("horizon_rows", []):
        lines.append(
            f"| **{h_row.get('Horizon', '')}** | {h_row.get('MSE', 0.0):.4f} | "
            f"{h_row.get('RMSE', 0.0):.4f} | {h_row.get('MAE', 0.0):.4f} | "
            f"{h_row.get('MAPE', 0.0):.2f}% | **{h_row.get('R2', 0.0):.4f}** |"
        )

    lines.extend([
        "",
        f"![PEMS-BAY Horizons]({fig_prefix}/pems_bay_horizon_comparison.png)",
        "",
    ])

    return "\n".join(lines)


def main(run_id: Optional[str] = None) -> Dict[str, Any]:
    """
    What: Kết xuất đồng bộ toàn bộ báo cáo `.md`, `.html` và `.pdf` vào `reports/lab04/` và `docs/lab04/`.
    Why: Hoàn tất bước cuối cùng của Pipeline tự động (`REPORTING`).
    Input:
        - run_id (Optional[str]): Mã định danh phiên chạy.
    Output:
        - Dict[str, Any]: Đường dẫn các tệp báo cáo đã sinh.
    How: Gọi `generate_markdown_report()` và `render_report.render_html()`.
    """
    t0 = time.perf_counter()
    cfg = load_config()
    active_run_id = run_id or get_latest_run_id()
    write_log(active_run_id, "REPORTING", "START", "Bắt đầu tổng hợp và kết xuất LAB_04_REPORT.md, .html và .pdf.")

    render_mod = _load_render_report_module()
    facts = render_mod.collect_report_facts()

    reports_dir = ROOT_DIR / cfg["paths"]["reports_dir"]
    docs_dir = ROOT_DIR / cfg["paths"]["docs_dir"]
    ensure_dir(reports_dir)
    ensure_dir(docs_dir)

    md_reports = generate_markdown_report(facts, fig_prefix="figures")
    md_docs = generate_markdown_report(facts, fig_prefix="../../reports/lab04/figures")

    rep_md_path = reports_dir / "LAB_04_REPORT.md"
    exp_sum_path = reports_dir / "experiment_summary.md"
    run_md_path = reports_dir / f"report_{active_run_id}.md"
    docs_md_path = docs_dir / "LAB_04_REPORT.md"

    rep_md_path.write_text(md_reports, encoding="utf-8")
    exp_sum_path.write_text(md_reports, encoding="utf-8")
    run_md_path.write_text(md_reports, encoding="utf-8")
    docs_md_path.write_text(md_docs, encoding="utf-8")

    html_path = render_mod.render_html(active_run_id)
    pdf_path = reports_dir / "LAB_04_REPORT.pdf"

    duration = time.perf_counter() - t0
    out_info = {
        "report_md": str(rep_md_path.relative_to(ROOT_DIR)).replace("\\", "/"),
        "report_html": str(html_path.relative_to(ROOT_DIR)).replace("\\", "/"),
        "report_pdf": str(pdf_path.relative_to(ROOT_DIR)).replace("\\", "/") if pdf_path.exists() else None,
        "docs_report_md": str(docs_md_path.relative_to(ROOT_DIR)).replace("\\", "/"),
        "duration_sec": round(duration, 4),
    }

    update_progress(
        active_run_id,
        "REPORTING",
        "COMPLETED",
        input_data={"run_id": active_run_id},
        output_data=out_info,
        duration_sec=duration,
    )
    write_log(active_run_id, "REPORTING", "SUCCESS", f"Hoàn tất xuất báo cáo MD/HTML/PDF ({duration:.2f}s).")
    return out_info


if __name__ == "__main__":
    main()