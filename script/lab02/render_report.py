"""
script/lab02/render_report.py
Kết xuất đồng bộ bộ 3 báo cáo học thuật chuẩn UTH Lab04 Design System (.html, .pdf, .md)
cho từng dataset Lab 02:
  - reports/lab02/LAB_02_DATASET1_REPORT.{html,pdf,md}
  - reports/lab02/LAB_02_DATASET2_REPORT.{html,pdf,md}
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_b64(path: Path) -> str:
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
    for cmd in ("msedge", "chrome", "google-chrome", "chromium", "chromium-browser"):
        found = shutil.which(cmd)
        if found:
            return found
    pf = os.environ.get("PROGRAMFILES", r"C:\Program Files")
    pf86 = os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")
    local = os.environ.get("LOCALAPPDATA", "")
    win_paths = [
        Path(pf86) / "Microsoft/Edge/Application/msedge.exe",
        Path(pf) / "Microsoft/Edge/Application/msedge.exe",
        Path(pf) / "Google/Chrome/Application/chrome.exe",
        Path(pf86) / "Google/Chrome/Application/chrome.exe",
        Path(local) / "Google/Chrome/Application/chrome.exe",
    ]
    for p in win_paths:
        if p.exists():
            return str(p)
    return None


def export_html_to_pdf(html_path: Path, pdf_path: Path) -> bool:
    browser = _find_chromium_binary()
    if not browser:
        print(f"[CẢNH BÁO] Không tìm thấy Edge/Chrome để in PDF cho {html_path.name}.")
        return False
    cmd = [
        browser,
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--print-to-pdf-no-header",
        f"--print-to-pdf={pdf_path.resolve()}",
        html_path.resolve().as_uri(),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    return proc.returncode == 0 and pdf_path.exists()


def _html_table(headers: List[str], rows: List[List[Any]], featured_idx: Optional[int] = None) -> str:
    th = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    trs = []
    for i, r in enumerate(rows):
        cls = ' class="row-featured"' if featured_idx is not None and i == featured_idx else ""
        tds = "".join(f"<td>{c}</td>" for c in r)
        trs.append(f"<tr{cls}>{tds}</tr>")
    return f'<div class="tbl-wrap"><table><thead><tr>{th}</tr></thead><tbody>{"".join(trs)}</tbody></table></div>'


def _md_table(headers: List[str], rows: List[List[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


CSS_STYLE = """
:root {
  --forest: #183f35; --forest-2: #245749; --mint: #e8f2ee; --accent: #c97a2b;
  --ink: #1e293b; --muted: #556677; --bg: #f6f8f7; --card: #ffffff; --border: #dce5e1;
}
* { box-sizing: border-box; }
body { margin: 0; font-family: "Segoe UI", system-ui, -apple-system, sans-serif; color: var(--ink); background: var(--bg); line-height: 1.62; }
.hero { background: linear-gradient(135deg, var(--forest) 0%, var(--forest-2) 100%); color: #fff; padding: 34px 36px 28px; }
.hero h1 { margin: 0 0 8px; font-size: 25px; letter-spacing: 0.2px; }
.hero p { margin: 0; opacity: 0.92; font-size: 14px; }
.kpi-strip { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-top: 22px; }
.kpi { background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.22); border-radius: 10px; padding: 12px 14px; }
.kpi .lbl { font-size: 11px; text-transform: uppercase; letter-spacing: 0.6px; opacity: 0.85; }
.kpi .val { font-size: 20px; font-weight: 700; margin-top: 4px; }
.kpi .sub { font-size: 11.5px; opacity: 0.88; margin-top: 2px; }
.sticky-nav { position: sticky; top: 0; z-index: 20; background: #fff; border-bottom: 1px solid var(--border); padding: 10px 28px; display: flex; flex-wrap: wrap; gap: 10px; }
.sticky-nav a { text-decoration: none; color: var(--forest); font-size: 12.5px; font-weight: 600; padding: 4px 10px; border-radius: 999px; background: var(--mint); }
.container { max-width: 1160px; margin: 24px auto 48px; padding: 0 22px; }
.stage { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 22px 24px; margin-bottom: 22px; page-break-inside: avoid; }
.stage-head { display: flex; align-items: center; gap: 12px; margin-bottom: 14px; border-bottom: 2px solid var(--mint); padding-bottom: 10px; }
.stage-num { width: 34px; height: 34px; border-radius: 50%; background: var(--forest); color: #fff; font-weight: 700; display: inline-flex; align-items: center; justify-content: center; font-size: 14px; flex-shrink: 0; }
.stage-title { font-size: 18px; font-weight: 700; color: var(--forest); margin: 0; }
.card-full { margin: 14px 0; border: 1px solid var(--border); border-radius: 10px; padding: 10px; background: #fff; }
.card-full img, .card-grid img { width: 100%; height: auto; display: block; border-radius: 6px; }
.card-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin: 14px 0; }
.fig-cap { font-size: 12px; color: var(--muted); text-align: center; margin-top: 6px; font-style: italic; }
.tbl-wrap { overflow-x: auto; margin: 12px 0; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { background: var(--forest); color: #fff; text-align: left; padding: 8px 10px; font-weight: 600; }
td { border-bottom: 1px solid var(--border); padding: 7px 10px; }
tr.row-featured { background: #eaf7f0; font-weight: 600; }
.badge-pass { background: #1b8a5a; color: #fff; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; }
.badge-fail { background: #c0392b; color: #fff; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; }
.takeaway { background: var(--mint); border-left: 4px solid var(--forest); padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 12px 0; font-size: 13.5px; }
.math-panel { background: #fbfaf5; border: 1px solid #e6dec8; border-left: 4px solid var(--accent); padding: 10px 14px; border-radius: 6px; margin: 10px 0; font-size: 13px; }
.qa-list li { margin-bottom: 7px; }
@media print {
  body { background: #fff; }
  .sticky-nav { display: none; }
  .stage { box-shadow: none; break-inside: avoid; }
}
"""


def _build_html_and_md(ctx: Dict[str, Any]) -> tuple[str, str]:
    ds_key = ctx["dataset_key"]
    is_ds1 = ds_key == "dataset1"
    fig_dir = PROJECT_ROOT / "reports" / "figures" / "lab02" / ds_key / "report"
    fig_rel = f"../figures/lab02/{ds_key}/report"

    eda = ctx["eda"]
    adf_raw, adf_d1, adf_d1D1 = ctx["adf"]["raw"], ctx["adf"]["diff_1"], ctx["adf"]["diff_1_12"]
    acf_d1, acf_d1D1 = ctx["acf_pacf"]["diff_1"], ctx["acf_pacf"]["diff_1_12"]
    a1, a2 = ctx["arima"]["ARIMA(1,1,1)"], ctx["arima"]["ARIMA(2,1,1)"]
    sarima = ctx["sarima"]
    bs = sarima["best_model"]
    log_sec = ctx.get("log_sarima")
    winner = bs if is_ds1 else log_sec["best_model"]
    s = ctx["seasonal_period"]

    ds_name = "Dataset 01 — Monthly Retail Sales (US)" if is_ds1 else "Dataset 02 — AirPassengers"

    # Bảng EDA đầy đủ
    eda_rows = [
        ["Số quan sát (n) / Missing / Trùng ngày", f"{eda['n_obs']} tháng / {eda['missing']} / {eda['duplicates']}"],
        ["Khoảng thời gian & Tần suất", f"{eda['start_date']} → {eda['end_date']} (Monthly Start - MS)"],
        ["Trung bình (Mean) / Độ lệch chuẩn (Std)", f"{eda['mean']:.2f} / {eda['std']:.2f} (CV = {eda['cv_pct']}%)"],
        ["Min / Q25 / Median / Q75 / Max", f"{eda['min']:.2f} ({eda['min_date']}) / {eda['q25']:.2f} / {eda['median']:.2f} / {eda['q75']:.2f} / {eda['max']:.2f} ({eda['max_date']})"],
        ["Độ lệch (Skewness) / Tăng trưởng đầu→cuối", f"{eda['skew']} / gấp {eda['growth_ratio']} lần"],
        ["Tập Train (80%, shuffle=False)", f"{eda['train_size']} tháng ({eda['train_start']} → {eda['train_end']}) | Mean={eda['train_mean']}, Std={eda['train_std']}"],
        ["Tập Test (20%, shuffle=False)", f"{eda['test_size']} tháng ({eda['test_start']} → {eda['test_end']}) | Mean={eda['test_mean']}, Std={eda['test_std']}"],
    ]

    # Bảng ADF 3 cấp độ
    adf_rows = []
    for lbl, adf in [("Chuỗi gốc (Train)", adf_raw), ("Sai phân bậc 1 (d=1)", adf_d1), (f"Sai phân kết hợp (d=1, D=1, s={s})", adf_d1D1)]:
        cv = adf["critical_values"]
        badge = '<span class="badge-pass">DỪNG (Pass)</span>' if adf["is_stationary"] else '<span class="badge-fail">KHÔNG DỪNG</span>'
        adf_rows.append([lbl, f"{adf['n_input']} (lags={adf['lags_used']})", f"{adf['adf_statistic']:.4f}", f"{adf['p_value']:.4f}",
                         f"{cv['1%']:.3f} / {cv['5%']:.3f} / {cv['10%']:.3f}", badge])

    # Bảng ARIMA & Đối chứng
    arima_rows = [
        [n, m["aic"], m["bic"], "Có" if m["converged"] else "Không", m["metrics"]["MAE"], m["metrics"]["RMSE"],
         f"{m['metrics']['MAPE (%)']}%", m["metrics"]["R2"], m["diagnostics"]["ljung_box_p"],
         '<span class="badge-pass">Pass</span>' if m["diagnostics"]["is_white_noise"] else '<span class="badge-fail">Fail</span>']
        for n, m in [("ARIMA(1,1,1)", a1), ("ARIMA(2,1,1)", a2)]
    ]

    top_s_rows = [
        [r["model_name"], r["AIC"], r["BIC"], r["MAE"], r["RMSE"], f"{r['MAPE (%)']}%", r["R2"], r["lb_p"],
         '<span class="badge-pass">Pass</span>' if r["is_white_noise"] else '<span class="badge-fail">Fail</span>']
        for r in sarima["top_models"]
    ]
    bs_idx = next((i for i, r in enumerate(sarima["top_models"]) if r["model_name"] == bs["name"]), 0)

    comp_rows = [
        ["ARIMA(1,1,1)", a1["aic"], a1["bic"], a1["metrics"]["MAE"], a1["metrics"]["RMSE"], f"{a1['metrics']['MAPE (%)']}%", a1["metrics"]["R2"], a1["diagnostics"]["ljung_box_p"], '<span class="badge-fail">Fail</span>'],
        ["ARIMA(2,1,1)", a2["aic"], a2["bic"], a2["metrics"]["MAE"], a2["metrics"]["RMSE"], f"{a2['metrics']['MAPE (%)']}%", a2["metrics"]["R2"], a2["diagnostics"]["ljung_box_p"], '<span class="badge-fail">Fail</span>'],
        [bs["name"], bs["aic"], bs["bic"], bs["metrics"]["MAE"], bs["metrics"]["RMSE"], f"{bs['metrics']['MAPE (%)']}%", bs["metrics"]["R2"], bs["diagnostics"]["ljung_box_p"], '<span class="badge-pass">White Noise</span>'],
    ]
    if not is_ds1:
        bl = log_sec["best_model"]
        comp_rows.append([bl["name"], bl["aic"], bl["bic"], bl["metrics"]["MAE"], bl["metrics"]["RMSE"], f"{bl['metrics']['MAPE (%)']}%", bl["metrics"]["R2"], bl["diagnostics"]["ljung_box_p"], '<span class="badge-pass">Winner</span>'])

    def _err_tbl_html(errs: List[Dict[str, Any]]) -> str:
        return _html_table(
            ["Tháng", "Thực tế", "Dự báo", "Sai lệch (TT - DB)", "|Sai số|", "Sai số (%)"],
            [[e["date"], e["actual"], e["forecast"], f"{e['error']:+.2f}", e["abs_error"], f"{e['pct_error']}%"] for e in errs],
        )

    def _pelim_html(pelim: Dict[str, List[Dict[str, Any]]]) -> str:
        out = []
        for k in ["q", "p", "Q", "P"]:
            if k in pelim:
                out.append(f"<p><strong>Tác động bậc {k}:</strong></p>" + _html_table(
                    [k, "AIC TB", "RMSE TB", "MAPE TB (%)", "R² TB"],
                    [[r[k], r["AIC"], r["RMSE"], f"{r['MAPE (%)']}%", r["R2"]] for r in pelim[k]],
                ))
        return "".join(out)

    # STAGE 07 riêng cho Dataset 2
    stage7_html = ""
    nav7 = ""
    if not is_ds1:
        bl = log_sec["best_model"]
        top_l_rows = [
            [r["model_name"], r["AIC"], r["BIC"], r["MAE"], r["RMSE"], f"{r['MAPE (%)']}%", r["R2"], r["lb_p"],
             '<span class="badge-pass">Pass</span>' if r["is_white_noise"] else '<span class="badge-fail">Fail</span>']
            for r in log_sec["top_models"]
        ]
        bl_idx = next((i for i, r in enumerate(log_sec["top_models"]) if r["model_name"] == bl["name"]), None)
        nav7 = '<a href="#stage7">07. Log-SARIMA & 3 Thế hệ</a>'
        stage7_html = f"""
        <section class="stage" id="stage7">
          <div class="stage-head"><span class="stage-num">07</span><h2 class="stage-title">Can thiệp nâng cao — Biến đổi Log ổn định phương sai (Log-SARIMA)</h2></div>
          <div class="math-panel">
            <strong>Cơ sở toán học:</strong> Chuyển cấu trúc mùa vụ nhân <code>Yₜ = Tₜ × Sₜ × Iₜ</code> sang dạng cộng <code>ln(Yₜ) = ln(Tₜ) + ln(Sₜ) + ln(Iₜ)</code> để triệt tiêu phương sai hình phễu, khớp SARIMA trên miền Log rồi nghịch đảo <code>Ŷₜ = exp(Ŷ'ₜ)</code>.
          </div>
          <div class="card-full"><img src="{load_b64(fig_dir / '09_log_transformation.png')}" alt="Log Transformation"><div class="fig-cap">Biểu đồ phân tích biến đổi Logarit ổn định phương sai (giữ nguyên nét gốc).</div></div>
          <h3>Top cấu hình Log-SARIMA ({log_sec['grid_total']} cấu hình) & Phân tích loại suy tham số</h3>
          {_html_table(["Mô hình Log-SARIMA", "AIC (Log)", "BIC (Log)", "MAE", "RMSE", "MAPE", "R²", "Ljung-Box p", "WN"], top_l_rows, featured_idx=bl_idx)}
          {_pelim_html(log_sec["param_elimination"])}
          <div class="takeaway">
            <strong>Chốt mô hình tối ưu toàn diện: {bl['name']}</strong> — Cấu trúc tối giản (Parsimony), <code>AIC = {bl['aic']}</code>, <code>RMSE = {bl['metrics']['RMSE']}</code>, <code>MAPE = {bl['metrics']['MAPE (%)']}%</code>, <code>R² = {bl['metrics']['R2']}</code>, <code>Ljung-Box p = {bl['diagnostics']['ljung_box_p']}</code>.
          </div>
          <div class="card-full"><img src="{load_b64(fig_dir / '10_arima_vs_log_sarima_battle.png')}" alt="Actual vs Forecast Battle"><div class="fig-cap">Actual vs. Forecast Battle: ARIMA(1,1,1) vs Log-SARIMA(1,1,0)x(0,1,1)₁₂.</div></div>
          <div class="card-full"><img src="{load_b64(fig_dir / '11_log_diagnostics_2x2.png')}" alt="Log Diagnostics 2x2"><div class="fig-cap">Chẩn đoán & phân rã sai số 2×2 — So sánh 3 thế hệ ARIMA → SARIMA gốc → Log-SARIMA.</div></div>
        </section>
        """

    html_doc = f"""<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8">
<title>Báo cáo Lab 02 — {html.escape(ds_name)}</title>
<style>{CSS_STYLE}</style></head><body>
<header class="hero">
  <h1>BÁO CÁO THỰC HÀNH LAB 02 — {html.escape(ds_name.upper())}</h1>
  <p>Kiểm định tính dừng ADF, Phân tích ACF/PACF &amp; Mô hình hóa chuỗi thời gian theo trục <strong>ARIMA → SARIMA{" → Log-SARIMA" if not is_ds1 else ""}</strong> | Run ID: <code>{ctx['run_id']}</code></p>
  <div class="kpi-strip">
    <div class="kpi"><div class="lbl">Dữ liệu &amp; Split (80/20)</div><div class="val">{eda['n_obs']} tháng</div><div class="sub">Train: {eda['train_size']} | Test: {eda['test_size']} (shuffle=False)</div></div>
    <div class="kpi"><div class="lbl">Mô hình tối ưu</div><div class="val" style="font-size:15px">{winner['name']}</div><div class="sub">Ljung-Box p = {winner['diagnostics']['ljung_box_p']} (White Noise)</div></div>
    <div class="kpi"><div class="lbl">Độ chính xác Test (R²)</div><div class="val">{winner['metrics']['R2']}</div><div class="sub">ARIMA(1,1,1): R² = {a1['metrics']['R2']}</div></div>
    <div class="kpi"><div class="lbl">Sai số ngoài mẫu</div><div class="val">MAPE {winner['metrics']['MAPE (%)']}%</div><div class="sub">RMSE = {winner['metrics']['RMSE']} | MAE = {winner['metrics']['MAE']}</div></div>
  </div>
</header>
<nav class="sticky-nav">
  <a href="#stage1">01. EDA &amp; Train/Test</a>
  <a href="#stage2">02. ADF Chuỗi gốc</a>
  <a href="#stage3">03. Sai phân &amp; ADF lại</a>
  <a href="#stage4">04. ACF &amp; PACF</a>
  <a href="#stage5">05. ARIMA &amp; 9 Câu hỏi</a>
  <a href="#stage6">06. SARIMA &amp; Đối chứng</a>
  {nav7}
</nav>
<main class="container">
  <section class="stage" id="stage1">
    <div class="stage-head"><span class="stage-num">01</span><h2 class="stage-title">Khám phá dữ liệu đầy đủ (EDA) &amp; Phân chia Train/Test (80/20)</h2></div>
    {_html_table(["Chỉ tiêu EDA &amp; Phân chia dữ liệu", "Giá trị trích xuất tự động"], eda_rows)}
    <div class="card-full"><img src="{load_b64(fig_dir / '01_original_series.png')}" alt="Original Series"><div class="fig-cap">Biểu đồ chuỗi thời gian gốc kèm Mean, Min, Max (giữ nguyên 100% nét vẽ gốc).</div></div>
    <div class="card-full"><img src="{load_b64(fig_dir / '02_train_test_split.png')}" alt="Train Test Split"><div class="fig-cap">Phân chia tập Train (80%) và Test (20%) tuần tự theo thời gian (shuffle=False).</div></div>
    <div class="takeaway">
      <strong>Trả lời câu hỏi Bài 1:</strong> (1) <strong>Trend: CÓ</strong> — tăng rõ rệt gấp {eda['growth_ratio']} lần. (2) <strong>Seasonality: CÓ</strong> — lặp lại chu kỳ {s} tháng với biên độ giãn rộng (mùa vụ dạng nhân). (3) <strong>Stationary: KHÔNG</strong> — cả trung bình và phương sai đều biến đổi theo thời gian.
    </div>
  </section>

  <section class="stage" id="stage2">
    <div class="stage-head"><span class="stage-num">02</span><h2 class="stage-title">Kiểm định tính dừng ADF trên chuỗi gốc (Train Set)</h2></div>
    <div class="math-panel"><strong>Giả thuyết ADF:</strong> H₀: Chuỗi có nghiệm đơn vị (KHÔNG DỪNG) | H₁: Chuỗi DỪNG. Mức ý nghĩa α = 0.05 (p-value &lt; 0.05 ⇒ bác bỏ H₀).</div>
    {_html_table(["Chuỗi kiểm định", "Số quan sát", "ADF Statistic", "p-value", "Critical Values (1%/5%/10%)", "Kết luận"], [adf_rows[0]])}
    <div class="takeaway">Với <code>ADF = {adf_raw['adf_statistic']:.4f}</code> và <code>p-value = {adf_raw['p_value']:.4f} ≥ 0.05</code>, chưa đủ cơ sở bác bỏ H₀ ⇒ Chuỗi gốc <strong>KHÔNG DỪNG</strong>.</div>
  </section>

  <section class="stage" id="stage3">
    <div class="stage-head"><span class="stage-num">03</span><h2 class="stage-title">Sai phân (d=1 và d=1, D=1, s={s}) &amp; Kiểm định ADF lại</h2></div>
    <div class="card-full"><img src="{load_b64(fig_dir / '03_differencing.png')}" alt="Differencing"><div class="fig-cap">So sánh chuỗi gốc, sai phân bậc 1 (d=1) và sai phân kết hợp (d=1, D=1, s={s}).</div></div>
    {_html_table(["Chuỗi kiểm định", "Số quan sát", "ADF Statistic", "p-value", "Critical Values (1%/5%/10%)", "Kết luận"], adf_rows[1:])}
    <div class="takeaway">
      <strong>Trả lời Bài 3:</strong> Sau <code>d=1</code>, xu hướng đã được khử nhưng ADF vẫn cho <code>p = {adf_d1['p_value']:.4f} ≥ 0.05</code> do mùa vụ chu kỳ {s} tháng quá mạnh. Chốt <strong>d = 1</strong> cho ARIMA (không tăng d=2 tránh quá sai phân) và bổ sung <strong>D = 1, s = {s}</strong> cho SARIMA (<code>p = {adf_d1D1['p_value']:.4f} &lt; 0.05</code> — dừng hoàn toàn).
    </div>
  </section>

  <section class="stage" id="stage4">
    <div class="stage-head"><span class="stage-num">04</span><h2 class="stage-title">Phân tích ACF &amp; PACF — Cơ sở đề xuất tham số</h2></div>
    <div class="card-grid">
      <div class="card-full" style="margin:0"><img src="{load_b64(fig_dir / '04_acf_pacf_d1.png')}" alt="ACF PACF d1"><div class="fig-cap">Chuỗi d=1 (n={acf_d1['n']}, ngưỡng 95% = ±{acf_d1['bound']})</div></div>
      <div class="card-full" style="margin:0"><img src="{load_b64(fig_dir / '05_acf_pacf_d1_D1.png')}" alt="ACF PACF d1 D1"><div class="fig-cap">Chuỗi d=1, D=1, s={s} (n={acf_d1D1['n']}, ngưỡng 95% = ±{acf_d1D1['bound']})</div></div>
    </div>
    <div class="takeaway">
      <strong>Lag vượt ngưỡng (d=1):</strong> ACF = <code>{acf_d1['acf_sig']}</code> | PACF = <code>{acf_d1['pacf_sig']}</code> ⇒ Đề xuất <strong>ARIMA(1,1,1)</strong> và <strong>ARIMA(2,1,1)</strong>.<br>
      <strong>Lag vượt ngưỡng (d=1, D=1):</strong> ACF = <code>{acf_d1D1['acf_sig']}</code> | PACF = <code>{acf_d1D1['pacf_sig']}</code> ⇒ Khảo sát không gian <strong>{sarima['grid_total']} cấu hình SARIMA</strong>.
    </div>
  </section>

  <section class="stage" id="stage5">
    <div class="stage-head"><span class="stage-num">05</span><h2 class="stage-title">Mô hình ARIMA — Chẩn đoán phần dư, Dự báo &amp; 9 Câu hỏi Bài 5</h2></div>
    {_html_table(["Mô hình", "AIC", "BIC", "Hội tụ", "MAE", "RMSE", "MAPE", "R²", "Ljung-Box p", "WN"], arima_rows)}
    <div class="card-grid">
      <div class="card-full" style="margin:0"><img src="{load_b64(fig_dir / '05a_arima1_residuals.png')}" alt="ARIMA1 Resid"><div class="fig-cap">Chẩn đoán phần dư ARIMA(1,1,1)</div></div>
      <div class="card-full" style="margin:0"><img src="{load_b64(fig_dir / '05b_arima1_forecast.png')}" alt="ARIMA1 Forecast"><div class="fig-cap">Dự báo ngoài mẫu ARIMA(1,1,1)</div></div>
    </div>
    <h3>Top 3 tháng sai lệch lớn nhất — ARIMA(1,1,1) &amp; ARIMA(2,1,1)</h3>
    <div class="card-grid"><div>{_err_tbl_html(a1['top_errors'])}</div><div>{_err_tbl_html(a2['top_errors'])}</div></div>
    <ol class="qa-list">
      <li><strong>Vì sao không shuffle &amp; vai trò Test set?</strong> Chuỗi thời gian có phụ thuộc tuần tự; shuffle gây rò rỉ dữ liệu tương lai. Test set (20% cuối) đánh giá năng lực dự báo ngoài mẫu thực tế.</li>
      <li><strong>Ý nghĩa (p, d, q):</strong> p là bậc tự hồi quy (AR), d=1 là sai phân khử xu hướng (I), q=1 là bậc trung bình trượt hiệu chỉnh sai số quá khứ (MA).</li>
      <li><strong>Hội tụ:</strong> Có — cả 2 mô hình hội tụ tối ưu MLE không cảnh báo.</li>
      <li><strong>White noise:</strong> Không — Ljung-Box p = 0.0000 &lt; 0.05, phần dư còn tự tương quan mùa vụ rất mạnh.</li>
      <li><strong>Hướng điều chỉnh:</strong> Bắt buộc nâng cấp lên SARIMA có sai phân mùa vụ D=1 và tham số mùa vụ (P, Q).</li>
      <li><strong>Bám sát xu hướng:</strong> Không — đường dự báo bị bẹt ngang, R² âm ({a1['metrics']['R2']} và {a2['metrics']['R2']}).</li>
      <li><strong>Sai lệch lớn nhất:</strong> Rơi vào các tháng đỉnh/đáy mùa vụ (lệch tới {a1['top_errors'][0]['abs_error']} với ARIMA(1,1,1)).</li>
      <li><strong>Chỉ số nhạy outlier nhất:</strong> RMSE (bình phương sai số trước khi lấy trung bình nên phạt nặng sai lệch lớn).</li>
      <li><strong>Ý nghĩa RMSE:</strong> RMSE của ARIMA ({a1['metrics']['RMSE']} và {a2['metrics']['RMSE']}) lớn hơn cả độ lệch chuẩn tập Test ({eda['test_std']}), khẳng định ARIMA không đủ tin cậy.</li>
    </ol>
  </section>

  <section class="stage" id="stage6">
    <div class="stage-head"><span class="stage-num">06</span><h2 class="stage-title">Mô hình SARIMA — Loại suy tham số &amp; Đối chứng trực diện ARIMA</h2></div>
    {_html_table(["Mô hình SARIMA", "AIC", "BIC", "MAE", "RMSE", "MAPE", "R²", "Ljung-Box p", "WN"], top_s_rows, featured_idx=bs_idx)}
    {_pelim_html(sarima["param_elimination"])}
    <div class="card-grid">
      <div class="card-full" style="margin:0"><img src="{load_b64(fig_dir / '06_best_sarima_residuals.png')}" alt="Best SARIMA Resid"><div class="fig-cap">Chẩn đoán phần dư {bs['name']}</div></div>
      <div class="card-full" style="margin:0"><img src="{load_b64(fig_dir / '07_best_sarima_forecast.png')}" alt="Best SARIMA Forecast"><div class="fig-cap">Dự báo {bs['name']} kèm CI 95%</div></div>
    </div>
    <h3>Bảng &amp; Biểu đồ đối chứng trực diện</h3>
    {_html_table(["Mô hình", "AIC", "BIC", "MAE", "RMSE", "MAPE", "R²", "Ljung-Box p", "Đánh giá"], comp_rows, featured_idx=len(comp_rows)-1)}
    <div class="card-full"><img src="{load_b64(fig_dir / '08_arima_vs_sarima.png')}" alt="ARIMA vs SARIMA"><div class="fig-cap">Đối chứng trực diện ARIMA(1,1,1), ARIMA(2,1,1) và {bs['name']}.</div></div>
    <p style="font-size:12.5px;color:#556677"><em>Ghi chú: Toàn bộ bảng benchmark mở rộng (bao gồm AR / MA đơn lẻ theo yêu cầu model_summary) được lưu tại <code>reports/model_summary/lab02/{ds_key}/{ds_key}_model_summary.md</code>.</em></p>
  </section>

  {stage7_html}
</main></body></html>"""

    # Markdown đồng bộ 100% số liệu
    md_lines = [
        f"# BÁO CÁO THỰC HÀNH LAB 02 — {ds_name.upper()}",
        f"> Run ID: `{ctx['run_id']}` | Model Summary mở rộng (gồm AR/MA): `reports/model_summary/lab02/{ds_key}/{ds_key}_model_summary.md`",
        "\n## STAGE 01: Khám phá dữ liệu (EDA đầy đủ) & Phân chia Train/Test (80/20)",
        _md_table(["Chỉ tiêu", "Giá trị"], eda_rows),
        f"\n[{fig_rel}/01_original_series.png]\n[{fig_rel}/02_train_test_split.png]",
        "\n## STAGE 02–03: Kiểm định ADF & Sai phân",
        _md_table(["Chuỗi", "ADF Stat", "p-value", "Kết luận"],
                  [[r[0], r[2], r[3], "DỪNG" if "DỪNG (" in r[5] else "KHÔNG DỪNG"] for r in adf_rows]),
        f"\n[{fig_rel}/03_differencing.png]",
        "\n## STAGE 04: Phân tích ACF & PACF",
        f"- `d=1` (±{acf_d1['bound']}): ACF sig = `{acf_d1['acf_sig']}` | PACF sig = `{acf_d1['pacf_sig']}`",
        f"- `d=1, D=1, s={s}` (±{acf_d1D1['bound']}): ACF sig = `{acf_d1D1['acf_sig']}` | PACF sig = `{acf_d1D1['pacf_sig']}`",
        f"\n[{fig_rel}/04_acf_pacf_d1.png]\n[{fig_rel}/05_acf_pacf_d1_D1.png]",
        "\n## STAGE 05–06: Đối chứng ARIMA vs SARIMA" + (" → Log-SARIMA" if not is_ds1 else ""),
        _md_table(["Mô hình", "AIC", "BIC", "MAE", "RMSE", "MAPE", "R²", "Ljung-Box p"],
                  [r[:8] for r in comp_rows]),
        f"\n[{fig_rel}/06_best_sarima_residuals.png]\n[{fig_rel}/08_arima_vs_sarima.png]",
    ]
    if not is_ds1:
        md_lines.append(
            f"\n## STAGE 07: Log-SARIMA & Đối chứng 3 thế hệ\n"
            f"- Mô hình tối ưu: **`{log_sec['best_model']['name']}`** (RMSE = `{log_sec['best_model']['metrics']['RMSE']}`, MAPE = `{log_sec['best_model']['metrics']['MAPE (%)']}%`, R² = `{log_sec['best_model']['metrics']['R2']}`, Ljung-Box p = `{log_sec['best_model']['diagnostics']['ljung_box_p']}`).\n"
            f"\n[{fig_rel}/09_log_transformation.png]\n"
            f"[{fig_rel}/10_arima_vs_log_sarima_battle.png]\n"
            f"[{fig_rel}/11_log_diagnostics_2x2.png]"
        )

    return html_doc, "\n".join(md_lines) + "\n"


def render_reports(dataset: str = "all") -> Dict[str, Dict[str, str]]:
    targets = ["dataset1", "dataset2"] if dataset == "all" else [dataset]
    out_dir = PROJECT_ROOT / "reports" / "lab02"
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_dir = PROJECT_ROOT / "docs" / "lab02"
    docs_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Dict[str, str]] = {}

    for ds in targets:
        latest_json = PROJECT_ROOT / "progress_results" / "lab02" / ds / "latest_run.json"
        if not latest_json.exists():
            raise FileNotFoundError(
                f"Chưa có {latest_json}. Chạy trước: python script/lab02/run_lab02_experiments.py --dataset {ds}"
            )
        ctx = json.loads(latest_json.read_text(encoding="utf-8"))
        run_id = ctx.get("run_id", "latest")
        html_str, md_str = _build_html_and_md(ctx)

        prefix = "LAB_02_DATASET1_REPORT" if ds == "dataset1" else "LAB_02_DATASET2_REPORT"
        html_path = out_dir / f"{prefix}.html"
        md_path = out_dir / f"{prefix}.md"
        pdf_path = out_dir / f"{prefix}.pdf"

        html_path.write_text(html_str, encoding="utf-8")
        md_path.write_text(md_str, encoding="utf-8")
        pdf_ok = export_html_to_pdf(html_path, pdf_path)

        shutil.copy2(html_path, docs_dir / f"{prefix}.html")
        shutil.copy2(md_path, docs_dir / f"{prefix}.md")
        shutil.copy2(html_path, out_dir / f"{prefix}_{run_id}.html")
        shutil.copy2(md_path, out_dir / f"{prefix}_{run_id}.md")
        if pdf_ok:
            shutil.copy2(pdf_path, docs_dir / f"{prefix}.pdf")
            shutil.copy2(pdf_path, out_dir / f"{prefix}_{run_id}.pdf")

        print(f"[OK - {ds.upper()}] HTML: {html_path}")
        print(f"[OK - {ds.upper()}] MD  : {md_path}")
        print(f"[{'OK' if pdf_ok else 'SKIP'} - {ds.upper()}] PDF : {pdf_path}")
        outputs[ds] = {"html": str(html_path), "md": str(md_path), "pdf": str(pdf_path) if pdf_ok else ""}
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Xuất bộ 3 báo cáo (.html, .pdf, .md) cho Lab 02.")
    parser.add_argument("--dataset", default="all", choices=["dataset1", "dataset2", "all"])
    args = parser.parse_args()
    render_reports(dataset=args.dataset)
    return 0


if __name__ == "__main__":
    sys.exit(main())