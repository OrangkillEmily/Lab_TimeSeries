"""
Bộ kết xuất báo cáo HTML tương tác (Base64 Embedded Figures) và xuất PDF tự động (`script/lab04/render_report.py`).

Trách nhiệm của module:
1. Trích xuất TỰ ĐỘNG 100% số liệu thực nghiệm từ kết quả đã chạy tại:
   - `reports/lab04/metrics/evaluation_latest.json`
   - `reports/lab04/metrics/eda_summary.json`
   - `reports/lab04/metrics/preprocessing_summary.json`
   - `models/lab04/training_metadata.json`
   - (Và hỗ trợ đồng bộ tự động từ `notebooks/lab04/lab04.ipynb` nếu người dùng chạy trên Notebook).
   Tuyệt đối KHÔNG gắn cứng bất kỳ con số giả định nào (Rule 5 & Rule 8).
2. Nhúng toàn bộ bộ biểu đồ PNG 300 DPI từ `reports/lab04/figures/` dưới dạng chuỗi `data:image/png;base64,...`
   để tệp `reports/lab04/LAB_04_REPORT.html` và `docs/lab04/LAB_04_REPORT.html` hiển thị độc lập trên mọi máy.
3. Tự động chuyển đổi báo cáo HTML sang `reports/lab04/LAB_04_REPORT.pdf` và `docs/lab04/LAB_04_REPORT.pdf`
   bằng trình duyệt Edge / Chrome / Chromium Headless hoặc Selenium CDP.
"""

from __future__ import annotations

import base64
import html
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve()
ROOT_DIR = HERE.parent.parent.parent
if not (ROOT_DIR / "src").exists() and (HERE.parent / "src").exists():
    ROOT_DIR = HERE.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from src.lab04.utils import ensure_dir, get_project_root, load_config, load_json


def _require_json(path: Path) -> Dict[str, Any]:
    """
    What: Đọc tệp JSON bắt buộc từ kết quả thực nghiệm; nếu chưa tồn tại thì thông báo rõ ràng.
    Why: Đảm bảo 100% số liệu trong `LAB_04_REPORT.md`, `LAB_04_REPORT.html` và `LAB_04_REPORT.pdf`
         được trích xuất tự động từ kết quả chạy thật trên máy của người dùng, tuyệt đối không tự gắn số giả.
    Input:
        - path (Path): Đường dẫn tệp JSON cần đọc.
    Output:
        - Dict[str, Any]: Dữ liệu thực nghiệm đã giải mã.
    How: Kiểm tra `path.exists()` và đọc bằng `json.loads()`.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy tệp kết quả thực nghiệm '{path}'. "
            "Hãy chạy Pipeline ('python script/lab04/run_lab04.py' hoặc chạy 'notebooks/lab04/lab04.ipynb') "
            "trước khi gọi 'render_report.py' để toàn bộ số liệu được trích xuất tự động từ kết quả huấn luyện thực tế."
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Tệp JSON '{path}' bị hỏng: {exc}") from exc


def load_b64(path: Path, max_width: int = 1050) -> str:
    """
    What: Đọc và mã hóa tệp hình ảnh PNG sang chuỗi Data URI Base64.
    Why: Giúp tệp `LAB_04_REPORT.html` tự chứa toàn bộ hình ảnh mà không phụ thuộc đường dẫn ngoài.
    Input:
        - path (Path): Đường dẫn tới tệp ảnh `.png`.
        - max_width (int): Chiều rộng tối đa khi tối ưu hiển thị (`1050` px).
    Output:
        - str: Chuỗi `data:image/png;base64,...` hoặc chuỗi rỗng nếu tệp chưa tồn tại.
    How: Dùng `PIL.Image` thu phóng giữ nguyên tỷ lệ và mã hóa `base64`.
    """
    if not path.exists():
        return ""
    if HAS_PIL:
        try:
            with Image.open(path) as img:
                if img.width > max_width:
                    ratio = max_width / float(img.width)
                    new_h = int(img.height * ratio)
                    img = img.resize((max_width, new_h), Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="PNG", optimize=True)
                return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
        except Exception:
            pass
    with path.open("rb") as f:
        return f"data:image/png;base64,{base64.b64encode(f.read()).decode('ascii')}"


def _find_chromium_binary() -> Optional[str]:
    """
    What: Tìm đường dẫn trình duyệt Microsoft Edge, Google Chrome hoặc Chromium trên Windows/macOS/Linux.
    Why: Phục vụ việc xuất tự động `LAB_04_REPORT.pdf` từ `LAB_04_REPORT.html`.
    Input: Không có.
    Output:
        - Optional[str]: Đường dẫn thực thi của trình duyệt nếu tìm thấy.
    How: Quét `shutil.which` và các đường dẫn cài đặt mặc định trên Windows 11.
    """
    for cmd in ("msedge", "chrome", "google-chrome", "chromium", "chromium-browser"):
        found = shutil.which(cmd)
        if found:
            return found

    pf = os.environ.get("PROGRAMFILES", r"C:\Program Files")
    pf86 = os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")
    local_app = os.environ.get("LOCALAPPDATA", "")
    win_candidates = [
        Path(pf86) / "Microsoft/Edge/Application/msedge.exe",
        Path(pf) / "Microsoft/Edge/Application/msedge.exe",
        Path(pf) / "Google/Chrome/Application/chrome.exe",
        Path(pf86) / "Google/Chrome/Application/chrome.exe",
    ]
    if local_app:
        win_candidates.append(Path(local_app) / "Google/Chrome/Application/chrome.exe")
        win_candidates.append(Path(local_app) / "Microsoft/Edge/Application/msedge.exe")

    for p in win_candidates:
        if p.exists():
            return str(p)
    return None


def export_html_to_pdf(html_path: Path, pdf_path: Path) -> bool:
    """
    What: Tự động chuyển đổi tệp `LAB_04_REPORT.html` sang `LAB_04_REPORT.pdf`.
    Why: Đáp ứng yêu cầu tự động xuất `report.pdf` theo đúng chuẩn style báo cáo `Lab04_ML-main`.
    Input:
        - html_path (Path): Đường dẫn tệp HTML nguồn.
        - pdf_path (Path): Đường dẫn tệp PDF đích.
    Output:
        - bool: `True` nếu xuất PDF thành công, `False` nếu không tìm thấy trình kết xuất.
    How: Thử gọi trực tiếp Chromium/Edge/Chrome Headless (`--print-to-pdf`), sau đó dự phòng
         qua Selenium CDP (`Page.printToPDF`).
    """
    ensure_dir(pdf_path.parent)
    html_abs = html_path.resolve()
    pdf_abs = pdf_path.resolve()

    browser_bin = _find_chromium_binary()
    if browser_bin:
        user_data_dir = tempfile.mkdtemp(prefix="lab04_ts_pdf_")
        try:
            for flag in ("--headless=new", "--headless"):
                cmd = [
                    browser_bin,
                    flag,
                    "--disable-gpu",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--no-pdf-header-footer",
                    "--print-to-pdf-no-header",
                    f"--user-data-dir={user_data_dir}",
                    f"--print-to-pdf={pdf_abs}",
                    html_abs.as_uri(),
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=45, check=False)
                if pdf_abs.exists() and pdf_abs.stat().st_size > 1000:
                    return True
        except Exception:
            pass
        finally:
            shutil.rmtree(user_data_dir, ignore_errors=True)

    try:
        import time
        from selenium import webdriver
        from selenium.webdriver.chrome import options as chrome_options

        user_data_dir = tempfile.mkdtemp(prefix="lab04_sel_pdf_")
        try:
            opts = chrome_options.Options()
            opts.add_argument("--headless=new")
            opts.add_argument("--no-sandbox")
            opts.add_argument("--disable-dev-shm-usage")
            opts.add_argument(f"--user-data-dir={user_data_dir}")
            driver = webdriver.Chrome(options=opts)
            try:
                driver.get(html_abs.as_uri())
                time.sleep(0.8)
                res = driver.execute_cdp_cmd("Page.printToPDF", {"printBackground": True, "preferCSSPageSize": True})
                if res and "data" in res:
                    pdf_abs.write_bytes(base64.b64decode(res["data"]))
                    if pdf_abs.exists() and pdf_abs.stat().st_size > 1000:
                        return True
            finally:
                driver.quit()
        finally:
            shutil.rmtree(user_data_dir, ignore_errors=True)
    except Exception:
        pass
    return False


def collect_report_facts() -> Dict[str, Any]:
    """
    What: Trích xuất tự động toàn bộ số liệu thực nghiệm từ `reports/lab04/metrics/` và `models/lab04/`.
    Why: Đảm bảo báo cáo Markdown, HTML và PDF lấy 100% số liệu thật từ quá trình huấn luyện trên máy.
    Input: Không có.
    Output:
        - Dict[str, Any]: Từ điển tổng hợp đầy đủ kết quả EDA, Preprocessing, Training, Evaluation.
    How: Gọi `_require_json` trên `evaluation_latest.json` và nạp các tệp JSON/CSV thực nghiệm liên quan.
    """
    root = get_project_root()
    cfg = load_config()
    metrics_dir = root / cfg["paths"]["metrics_dir"]
    models_dir = root / cfg["paths"]["models_dir"]

    eval_data = _require_json(metrics_dir / "evaluation_latest.json")
    eda_data = load_json(metrics_dir / "eda_summary.json", default={})
    prep_data = load_json(metrics_dir / "preprocessing_summary.json", default={})
    train_data = load_json(models_dir / "training_metadata.json", default={})
    data_load = load_json(metrics_dir / "data_loading_summary.json", default={})

    return {
        "config": cfg,
        "data_loading": data_load,
        "eda": eda_data,
        "preprocessing": prep_data,
        "training": train_data,
        "evaluation": eval_data,
    }


def render_html(run_id: Optional[str] = None) -> Path:
    """
    What: Tạo tệp báo cáo giao diện học thuật `reports/lab04/LAB_04_REPORT.html` (nhúng ảnh Base64)
          và tự động kết xuất sang `reports/lab04/LAB_04_REPORT.pdf` + đồng bộ sang `docs/lab04/`.
    Why: Cung cấp báo cáo chuyên nghiệp theo chuẩn `Lab04_ML-main` với số liệu trích xuất tự động 100%.
    Input:
        - run_id (Optional[str]): Mã định danh phiên chạy.
    Output:
        - Path: Đường dẫn tới tệp `reports/lab04/LAB_04_REPORT.html`.
    How: Tổng hợp dữ liệu từ `collect_report_facts()`, nhúng toàn bộ biểu đồ `.png` dạng Base64,
         ghi tệp HTML và gọi `export_html_to_pdf()`.
    """
    root = get_project_root()
    facts = collect_report_facts()
    cfg = facts["config"]
    eval_data = facts["evaluation"]
    prep_data = facts["preprocessing"]
    train_data = facts["training"]

    active_run_id = run_id or eval_data.get("run_id", "latest")
    fig_dir = root / cfg["paths"]["figures_dir"]
    reports_dir = root / cfg["paths"]["reports_dir"]
    docs_dir = root / cfg["paths"]["docs_dir"]
    ensure_dir(reports_dir)
    ensure_dir(docs_dir)

    holdout = eval_data.get("holdout_test_metrics", {})
    wf_metrics = eval_data.get("walk_forward_backtesting", {}).get("metrics", {})
    kfold_sum = eval_data.get("kfold_cv", {}).get("summary_by_model", {})
    pems_eval = eval_data.get("pems_bay_dcrnn", {})
    q_answers = eval_data.get("section11_analytical_answers", {})
    hw = eval_data.get("hardware", train_data.get("hardware", {}))

    def _img_tag(fname: str, caption: str) -> str:
        b64 = load_b64(fig_dir / fname)
        if not b64:
            return ""
        return (
            f'<figure class="fig-box">'
            f'<img src="{b64}" alt="{html.escape(caption)}" />'
            f'<figcaption>{html.escape(caption)}</figcaption>'
            f'</figure>'
        )

    holdout_rows_html = ""
    for m_name in ["VAR", "XGBoost", "LSTM"]:
        m = holdout.get(m_name, {})
        if not m:
            continue
        holdout_rows_html += (
            f"<tr>"
            f"<td><strong>{m_name}</strong></td>"
            f"<td class='num'>{m.get('MSE', 0.0):.4f}</td>"
            f"<td class='num'>{m.get('RMSE', 0.0):.4f}</td>"
            f"<td class='num'>{m.get('MAE', 0.0):.4f}</td>"
            f"<td class='num'>{m.get('MAPE', 0.0):.2f}%</td>"
            f"<td class='num'><strong>{m.get('R2', 0.0):.4f}</strong></td>"
            f"<td class='num'>{m.get('Training_Time_Sec', 0.0):.2f}s</td>"
            f"<td><code>{m.get('Device', 'cpu')}</code></td>"
            f"</tr>"
        )

    kfold_rows_html = ""
    for m_name in ["VAR", "XGBoost", "LSTM"]:
        kf = kfold_sum.get(m_name, {})
        if not kf:
            continue
        kfold_rows_html += (
            f"<tr>"
            f"<td><strong>{m_name}</strong></td>"
            f"<td class='num'>{kf.get('MSE', {}).get('formatted', 'N/A')}</td>"
            f"<td class='num'>{kf.get('RMSE', {}).get('formatted', 'N/A')}</td>"
            f"<td class='num'>{kf.get('MAE', {}).get('formatted', 'N/A')}</td>"
            f"<td class='num'>{kf.get('MAPE', {}).get('formatted', 'N/A')}%</td>"
            f"<td class='num'><strong>{kf.get('R2', {}).get('formatted', 'N/A')}</strong></td>"
            f"</tr>"
        )

    wf_rows_html = ""
    for m_name in ["VAR", "XGBoost", "LSTM"]:
        wm = wf_metrics.get(m_name, {})
        if not wm:
            continue
        wf_rows_html += (
            f"<tr>"
            f"<td><strong>{m_name}</strong></td>"
            f"<td class='num'>{wm.get('MSE', 0.0):.4f}</td>"
            f"<td class='num'>{wm.get('RMSE', 0.0):.4f}</td>"
            f"<td class='num'>{wm.get('MAE', 0.0):.4f}</td>"
            f"<td class='num'>{wm.get('MAPE', 0.0):.2f}%</td>"
            f"<td class='num'><strong>{wm.get('R2', 0.0):.4f}</strong></td>"
            f"</tr>"
        )

    pems_rows_html = ""
    for h_row in pems_eval.get("horizon_rows", []):
        pems_rows_html += (
            f"<tr>"
            f"<td><strong>{html.escape(str(h_row.get('Horizon', '')))}</strong></td>"
            f"<td class='num'>{h_row.get('MSE', 0.0):.4f}</td>"
            f"<td class='num'>{h_row.get('RMSE', 0.0):.4f}</td>"
            f"<td class='num'>{h_row.get('MAE', 0.0):.4f}</td>"
            f"<td class='num'>{h_row.get('MAPE', 0.0):.2f}%</td>"
            f"<td class='num'><strong>{h_row.get('R2', 0.0):.4f}</strong></td>"
            f"</tr>"
        )

    q_items_html = ""
    q_titles = [
        ("Q1_Dominant_Weather_Variable", "Câu 1: Biến thời tiết nào ảnh hưởng mạnh nhất đến điện năng?"),
        ("Q2_When_VAR_Fails", "Câu 2: Khi nào mô hình VAR thất bại?"),
        ("Q3_What_XGBoost_Learns_From_Weather", "Câu 3: XGBoost học được thông tin gì từ thời tiết?"),
        ("Q4_What_Cycles_LSTM_Learns", "Câu 4: LSTM học được chu kỳ nào (ngày, tuần)?"),
        ("Q5_Does_LSTM_Overfit", "Câu 5: LSTM có bị overfitting không?"),
        ("Q6_Difference_Between_ML_And_DL", "Câu 6: Sự khác biệt giữa Machine Learning và Deep Learning trong bài toán này?"),
    ]
    for q_key, q_title in q_titles:
        ans = q_answers.get(q_key, "")
        q_items_html += (
            f"<div class='qa-card'>"
            f"<h4>{html.escape(q_title)}</h4>"
            f"<p>{html.escape(ans)}</p>"
            f"</div>"
        )

    split_counts = prep_data.get("chronological_split_counts", {})
    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <title>Báo cáo Thực nghiệm Lab 04 — Dự báo Chuỗi Thời gian Đa biến (UTH)</title>
  <style>
    :root {{
      --primary: #1e3a8a;
      --accent: #2563eb;
      --bg: #f8fafc;
      --card: #ffffff;
      --text: #0f172a;
      --muted: #475569;
      --border: #e2e8f0;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 24px;
      line-height: 1.6;
    }}
    .container {{
      max-width: 1120px;
      margin: 0 auto;
      background: var(--card);
      padding: 36px 44px;
      border-radius: 14px;
      box-shadow: 0 4px 20px rgba(15, 23, 42, 0.06);
      border: 1px solid var(--border);
    }}
    .header-banner {{
      border-bottom: 3px solid var(--primary);
      padding-bottom: 18px;
      margin-bottom: 26px;
    }}
    .header-banner h1 {{
      margin: 0 0 8px 0;
      color: var(--primary);
      font-size: 24px;
    }}
    .meta-strip {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      font-size: 13.5px;
      color: var(--muted);
    }}
    .badge {{
      background: #eff6ff;
      color: #1d4ed8;
      border: 1px solid #bfdbfe;
      padding: 4px 10px;
      border-radius: 6px;
      font-weight: 600;
    }}
    h2 {{
      color: var(--primary);
      border-left: 5px solid var(--accent);
      padding-left: 12px;
      margin-top: 32px;
      font-size: 19px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 16px 0 24px 0;
      font-size: 14px;
    }}
    th, td {{
      border: 1px solid var(--border);
      padding: 10px 12px;
      text-align: left;
    }}
    th {{
      background: #f1f5f9;
      color: var(--primary);
      font-weight: 700;
    }}
    td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .fig-grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 20px;
      margin: 18px 0;
    }}
    .fig-grid-2 {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
      margin: 18px 0;
    }}
    figure.fig-box {{
      margin: 0;
      padding: 12px;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: #ffffff;
      text-align: center;
    }}
    figure.fig-box img {{
      max-width: 100%;
      height: auto;
      border-radius: 6px;
    }}
    figure.fig-box figcaption {{
      margin-top: 8px;
      font-size: 13px;
      color: var(--muted);
      font-style: italic;
    }}
    .qa-card {{
      background: #f8fafc;
      border: 1px solid var(--border);
      border-left: 4px solid var(--accent);
      padding: 14px 18px;
      border-radius: 8px;
      margin-bottom: 14px;
    }}
    .qa-card h4 {{ margin: 0 0 6px 0; color: var(--primary); }}
    .qa-card p {{ margin: 0; font-size: 14px; }}
    @media print {{
      body {{ background: #ffffff; padding: 0; }}
      .container {{ box-shadow: none; border: none; padding: 12px; }}
      figure.fig-box, .qa-card, table {{ page-break-inside: avoid; }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header-banner">
      <h1>BÁO CÁO THỰC NGHIỆM LAB 04: DỰ BÁO CHUỖI THỜI GIAN ĐA BIẾN (MULTIVARIATE TIME SERIES FORECASTING)</h1>
      <div class="meta-strip">
        <span class="badge">Học phần: Phân tích Chuỗi Thời gian (UTH)</span>
        <span class="badge">Run ID: {html.escape(str(active_run_id))}</span>
        <span class="badge">Thiết bị Auto-GPU: {html.escape(str(hw.get('device_name', 'CPU')))} ({html.escape(str(hw.get('torch_device', 'cpu')))})</span>
        <span class="badge">Chia dữ liệu: Train 70% ({split_counts.get('train_hours_70pct', 0)}h) | Val 15% ({split_counts.get('val_hours_15pct', 0)}h) | Test 15% ({split_counts.get('test_hours_15pct', 0)}h) + 5-Fold TimeSeriesSplit</span>
      </div>
    </div>

    <h2>1. Khám phá Dữ liệu (EDA — Mục 3: Missing, Outliers, Seasonality & Tương quan Thời tiết)</h2>
    <div class="fig-grid">
      {_img_tag("load_timeseries_overview.png", "Hình 1: Diễn biến phụ tải điện (load) theo thời gian và lát cắt chu kỳ 14 ngày.")}
      {_img_tag("load_vs_temperature_scatter.png", "Hình 2: Quan hệ phi tuyến Load vs Temperature và tương quan chéo theo độ trễ (Lag Heterogeneity).")}
    </div>
    <div class="fig-grid-2">
      {_img_tag("correlation_heatmap.png", "Hình 3: Ma trận tương quan Pearson 4 biến.")}
      {_img_tag("seasonality_daily_weekly.png", "Hình 4: Chu kỳ ngày (24h) và chu kỳ tuần (7 ngày).")}
    </div>

    <h2>2. Tiền xử lý, Chia Dữ liệu Thời gian (70/15/15) & Kiểm định Chéo 5-Fold TimeSeriesSplit</h2>
    <div class="fig-grid-2">
      {_img_tag("train_val_test_split.png", "Hình 5: Chia tập dữ liệu theo thứ tự thời gian 70% / 15% / 15% (No Shuffle).")}
      {_img_tag("kfold_splits_timeline.png", "Hình 6: Cấu trúc 5-Fold Expanding Window TimeSeriesSplit (In-Fold StandardScaler).")}
    </div>

    <h2>3. Quá trình Huấn luyện Đa Mô hình (VAR, XGBoost, LSTM) & Kiểm soát Overfitting</h2>
    <div class="fig-grid">
      {_img_tag("lstm_train_vs_val_loss.png", "Hình 7: Đường cong hội tụ Train vs Validation MSE & MAE của mạng LSTM(64)->LSTM(32)->Dense(1).")}
    </div>
    <div class="fig-grid-2">
      {_img_tag("xgboost_feature_importance.png", "Hình 8: Top 15 đặc trưng quan trọng nhất của XGBoost.")}
      {_img_tag("training_time_comparison.png", "Hình 9: So sánh thời gian huấn luyện giữa các mô hình.")}
    </div>

    <h2>4. Bảng Tổng hợp 5 Chỉ số Đánh giá (`MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²`)</h2>
    <h3>4.1. Kết quả trên Tập Kiểm tra Độc lập (`Hold-out Test Set 15%` — Mục 10 & Yêu cầu 6)</h3>
    <table>
      <thead>
        <tr>
          <th>Mô hình (Model)</th>
          <th>Test MSE (↓)</th>
          <th>Test RMSE (↓)</th>
          <th>Test MAE (↓)</th>
          <th>Test MAPE (↓)</th>
          <th>Test R² Score (↑)</th>
          <th>Train Time</th>
          <th>Device</th>
        </tr>
      </thead>
      <tbody>
        {holdout_rows_html}
      </tbody>
    </table>

    <h3>4.2. Kết quả Kiểm định Chéo 5-Fold Theo Thời gian (`5-Fold TimeSeriesSplit Mean ± Std` — Yêu cầu 4)</h3>
    <table>
      <thead>
        <tr>
          <th>Mô hình (Model)</th>
          <th>5-Fold MSE (Mean ± Std)</th>
          <th>5-Fold RMSE (Mean ± Std)</th>
          <th>5-Fold MAE (Mean ± Std)</th>
          <th>5-Fold MAPE (Mean ± Std)</th>
          <th>5-Fold R² Score (Mean ± Std)</th>
        </tr>
      </thead>
      <tbody>
        {kfold_rows_html}
      </tbody>
    </table>

    <h3>4.3. Kết quả Walk-Forward Validation / Backtesting (Mục 9: `Train -> predict t+1 -> append -> predict t+2`)</h3>
    <table>
      <thead>
        <tr>
          <th>Mô hình (Model)</th>
          <th>Walk-Forward MSE (↓)</th>
          <th>Walk-Forward RMSE (↓)</th>
          <th>Walk-Forward MAE (↓)</th>
          <th>Walk-Forward MAPE (↓)</th>
          <th>Walk-Forward R² (↑)</th>
        </tr>
      </thead>
      <tbody>
        {wf_rows_html}
      </tbody>
    </table>

    <div class="fig-grid">
      {_img_tag("actual_vs_predicted_all.png", "Hình 10: Đối chiếu phụ tải thực tế và dự báo của VAR, XGBoost, LSTM trên tập Test.")}
      {_img_tag("walk_forward_backtest.png", "Hình 11: Mô phỏng dự báo cuốn chiếu Walk-Forward Backtesting.")}
      {_img_tag("kfold_cv_comparison.png", "Hình 12: Độ ổn định RMSE và R² qua 5 Folds TimeSeriesSplit.")}
    </div>
    <div class="fig-grid-2">
      {_img_tag("model_comparison_mse.png", "Hình 13a: So sánh MSE.")}
      {_img_tag("model_comparison_rmse.png", "Hình 13b: So sánh RMSE.")}
      {_img_tag("model_comparison_mae.png", "Hình 13c: So sánh MAE.")}
      {_img_tag("model_comparison_mape.png", "Hình 13d: So sánh MAPE (%).")}
    </div>
    <div class="fig-grid-2">
      {_img_tag("model_comparison_r2.png", "Hình 13e: So sánh R² Score.")}
      {_img_tag("residual_distribution_all.png", "Hình 14: Phân phối xác suất phần dư.")}
    </div>

    <h2>5. Giải đáp 6 Câu hỏi Phân tích Học thuật (Mục 11 Đề bài)</h2>
    {q_items_html}

    <h2>6. Bài tập 12 Nâng cao: Dự báo Giao thông Đa bước với DCRNN trên PEMS-BAY (`pems-bay.h5` & `sensor_graph/adj_mx_bay.pkl`)</h2>
    <table>
      <thead>
        <tr>
          <th>Chân trời Dự báo (Horizon)</th>
          <th>MSE (mph² ↓)</th>
          <th>RMSE (mph ↓)</th>
          <th>MAE (mph ↓)</th>
          <th>MAPE (% ↓)</th>
          <th>R² Score (↑)</th>
        </tr>
      </thead>
      <tbody>
        {pems_rows_html}
      </tbody>
    </table>
    <div class="fig-grid">
      {_img_tag("pems_bay_horizon_comparison.png", "Hình 15: Kết quả dự báo tốc độ giao thông đa bước (15, 30, 60 phút) của DCRNN trên PEMS-BAY.")}
    </div>
  </div>
</body>
</html>
"""

    target_html = reports_dir / "LAB_04_REPORT.html"
    docs_html = docs_dir / "LAB_04_REPORT.html"
    run_html = reports_dir / f"report_{active_run_id}.html"

    for p in (target_html, docs_html, run_html):
        ensure_dir(p.parent)
        p.write_text(html_content, encoding="utf-8")

    pdf_out = reports_dir / "LAB_04_REPORT.pdf"
    if export_html_to_pdf(target_html, pdf_out):
        for extra_pdf in (docs_dir / "LAB_04_REPORT.pdf", reports_dir / f"report_{active_run_id}.pdf"):
            try:
                ensure_dir(extra_pdf.parent)
                shutil.copy2(pdf_out, extra_pdf)
            except Exception:
                pass
        print(f"[RENDER_REPORT] Đã xuất thành công HTML & PDF: {pdf_out} ({pdf_out.stat().st_size:,} bytes)")
    else:
        print(f"[RENDER_REPORT] Đã xuất thành công HTML: {target_html}")

    return target_html


if __name__ == "__main__":
    render_html()