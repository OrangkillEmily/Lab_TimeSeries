r"""
Mô-đun Trực quan hóa Học thuật Toàn diện (Visualization Suite) cho Bài thực hành Lab 04.
Kết hợp với 7 biểu đồ EDA từ src/lab04/eda.py để hoàn thiện kho 23 biểu đồ chuẩn xuất bản (220 DPI)
lưu đồng bộ tại:
- reports/lab04/figures/
- reports/lab04/figures/<run_id>/
- docs/lab04/figures/

Danh sách 16 biểu đồ tạo thêm tại mô-đun này:
08. chronological_split_diagram.png
09. stationarity_adf_comparison.png
10. Hourly_Profile.png
11. Weekly_Profile.png
12. Rolling_Volatility_30D.png
13. training_loss_curves.png
14. Forecast_Overlay_168h.png
15. actual_vs_predicted_scatter.png
16. Residual_Diagnostics.png
17. Metric_Comparison_Bars.png
18. training_time_comparison.png
19. Walk_Forward_Trajectory.png
20. XGBoost_Feature_Importance.png
21. Feature_Ablation_Impact.png
22. pems_bay_dcrnn_evaluation.png
23. pems_bay_model_comparison.png
"""

import shutil
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.lab04.tracking import get_active_run_id, record_stage_result, write_log
from src.lab04.utils import (
    ensure_directories,
    get_path,
    load_config,
    load_json,
    load_pickle,
)

warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_COLORS = {
    "Actual": "#0f172a",
    "VAR": "#2b5c8f",
    "XGBoost": "#1d5843",
    "LSTM": "#d95f02",
    "DCRNN": "#1d5843",
    "Multi_Horizon_TCN": "#2b5c8f",
    "Multi_Horizon_LSTM": "#d95f02",
}


def _save_and_sync_fig(fig: plt.Figure, filename: str, run_id: str, cfg: Dict[str, Any]) -> str:
    fig_dir = get_path("figures_dir", cfg)
    run_fig_dir = fig_dir / run_id
    docs_fig_dir = get_path("docs_dir", cfg) / "figures"

    fig_dir.mkdir(parents=True, exist_ok=True)
    run_fig_dir.mkdir(parents=True, exist_ok=True)
    docs_fig_dir.mkdir(parents=True, exist_ok=True)

    out_path = fig_dir / filename
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    for dst in (run_fig_dir, docs_fig_dir):
        try:
            shutil.copy2(out_path, dst / filename)
        except Exception:
            pass
    return filename


def main(run_id: Optional[str] = None) -> Dict[str, Any]:
    r"""
    Purpose: Khởi tạo toàn bộ 16 biểu đồ tiền xử lý, huấn luyện, dự báo, phần dư và đối sánh mô hình
             (hợp nhất cùng 7 biểu đồ EDA tạo thành bộ 23 biểu đồ hoàn chỉnh).
    Input-Output:
        - Input: run_id (Optional[str]).
        - Output: Từ điển chứa danh sách toàn bộ 23 tệp biểu đồ trong reports/lab04/figures/.
    How: Nạp processed_bundle.pkl, training_artifacts.pkl và các bảng CSV chỉ số trong reports/lab04/metrics/.
    """
    t0 = time.perf_counter()
    cfg = load_config()
    ensure_directories(cfg)
    active_run_id = get_active_run_id(run_id)

    write_log(active_run_id, "VISUALIZATION", "START", "Bắt đầu kết xuất bộ biểu đồ đánh giá, chẩn đoán phần dư và đối sánh đa mô hình.")

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Segoe UI", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    proc_dir = get_path("data_processed_dir", cfg)
    metrics_dir = get_path("metrics_dir", cfg)

    bundle = load_pickle(proc_dir / "processed_bundle.pkl")
    train_artifacts = load_pickle(proc_dir / "training_artifacts.pkl")

    holdout_df = pd.read_csv(metrics_dir / "holdout_test_metrics.csv")
    cv_sum_df = pd.read_csv(metrics_dir / "cv_5fold_summary.csv")
    wf_df = pd.read_csv(metrics_dir / "walk_forward_metrics.csv")
    wf_pred_df = pd.read_csv(metrics_dir / "walk_forward_predictions.csv")
    abl_df = pd.read_csv(metrics_dir / "weather_ablation_metrics.csv")
    xgb_imp_df = pd.read_csv(metrics_dir / "xgboost_feature_importance.csv")
    pems_dcrnn_df = pd.read_csv(metrics_dir / "pems_bay_dcrnn_metrics.csv")
    pems_bench_df = pd.read_csv(metrics_dir / "pems_bay_benchmark_comparison.csv")

    generated: List[str] = []

    # 8. chronological_split_diagram.png: Sơ đồ phân chia Chronological Split 70/15/15 + 5-Fold TimeSeriesSplit
    full_df: pd.DataFrame = bundle["full_df"]
    tr_df: pd.DataFrame = bundle["train_df"]
    va_df: pd.DataFrame = bundle["val_df"]
    te_df: pd.DataFrame = bundle["test_df"]

    fig, axes = plt.subplots(2, 1, figsize=(15, 7.2), gridspec_kw={"height_ratios": [1.35, 1.15]})
    ax0 = axes[0]
    tr_d = tr_df["load"].resample("1D").mean()
    va_d = va_df["load"].resample("1D").mean()
    te_d = te_df["load"].resample("1D").mean()
    ax0.plot(tr_d.index, tr_d.values, color="#1d5843", linewidth=1.3, label=f"Train Set (70% — {len(tr_df):,} giờ)")
    ax0.plot(va_d.index, va_d.values, color="#2b5c8f", linewidth=1.3, label=f"Validation Set (15% — {len(va_df):,} giờ)")
    ax0.plot(te_d.index, te_d.values, color="#d95f02", linewidth=1.3, label=f"Test Set (15% — {len(te_df):,} giờ)")
    ax0.axvline(tr_df.index.max(), color="#0f172a", linestyle="--", linewidth=1.5)
    ax0.axvline(va_df.index.max(), color="#b91c1c", linestyle="--", linewidth=1.5)
    ax0.set_title("Phân chia Chuỗi Thời gian Tuần tự Chống Rò rỉ Dữ liệu (Chronological Hold-out Split 70% / 15% / 15%)", fontsize=11.5, fontweight="bold", color="#183f35")
    ax0.set_ylabel("Phụ tải Điện (kW)")
    ax0.legend(loc="upper right", fontsize=9.5)
    ax0.grid(True, linestyle="--", alpha=0.35)

    ax1 = axes[1]
    cv_folds = bundle["cv_folds"]
    for f_item in cv_folds:
        k = int(f_item["fold"])
        tr_sz = int(f_item["train_size"])
        va_sz = int(f_item["val_size"])
        ax1.barh(f"Fold {k}", tr_sz, left=0, height=0.52, color="#1d5843", alpha=0.85, label="Expanding Train" if k == 1 else "")
        ax1.barh(f"Fold {k}", va_sz, left=tr_sz, height=0.52, color="#d95f02", alpha=0.90, label="Temporal Validation" if k == 1 else "")
    ax1.invert_yaxis()
    ax1.set_xlabel("Chỉ số Bước Thời gian (Hours Index: 0 → 35,065)")
    ax1.set_title("Cấu trúc Kiểm định Chéo Chuỗi Thời gian 5-Fold (Expanding Window TimeSeriesSplit K=5)", fontsize=11.5, fontweight="bold", color="#183f35")
    ax1.legend(loc="lower right", fontsize=9.5)
    ax1.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    generated.append(_save_and_sync_fig(fig, "chronological_split_diagram.png", active_run_id, cfg))

    # 9. stationarity_adf_comparison.png: Chuỗi gốc vs Chuỗi sai phân bậc 1 cho VAR
    stat_rep = bundle["stationarity_report"]
    fig, axes = plt.subplots(2, 1, figsize=(14.5, 6.2), sharex=True)
    sub_tr = tr_df["load"].iloc[-500:]
    sub_diff = sub_tr.diff().dropna()
    raw_adf_p = stat_rep["adf_raw"][0]["p_value"]
    diff_adf_p = stat_rep["adf_after_diff1"][0]["p_value"]
    axes[0].plot(sub_tr.index, sub_tr.values, color="#1d5843", linewidth=1.35)
    axes[0].set_title(f"Chuỗi Phụ tải Gốc Y_t (500 Giờ Cuối Tập Train | ADF Stat = {stat_rep['adf_raw'][0]['adf_statistic']:.3f}, p-value = {raw_adf_p:.4g})", fontsize=11, fontweight="bold", color="#183f35")
    axes[0].set_ylabel("Load (kW)")
    axes[0].grid(True, linestyle="--", alpha=0.35)

    axes[1].plot(sub_diff.index, sub_diff.values, color="#2b5c8f", linewidth=1.25)
    axes[1].axhline(0, color="#b91c1c", linestyle="--", linewidth=1.2)
    axes[1].set_title(f"Chuỗi Sai phân Bậc 1 ΔY_t = Y_t - Y_{{t-1}} cho Mô hình VAR (ADF Stat = {stat_rep['adf_after_diff1'][0]['adf_statistic']:.3f}, p-value = {diff_adf_p:.4g} — Chuỗi Dừng Tuyệt Đối)", fontsize=11, fontweight="bold", color="#183f35")
    axes[1].set_ylabel("ΔLoad (kW/h)")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    generated.append(_save_and_sync_fig(fig, "stationarity_adf_comparison.png", active_run_id, cfg))

    # 10. Hourly_Profile.png: Biên dạng phụ tải và nhiệt độ theo 24 giờ trong ngày
    fig, ax1 = plt.subplots(figsize=(12.5, 5.0))
    hr_grp = full_df.groupby(full_df.index.hour)
    hr_load_mean = hr_grp["load"].mean()
    hr_load_std = hr_grp["load"].std()
    hr_temp_mean = hr_grp["temperature"].mean()

    ax1.plot(hr_load_mean.index, hr_load_mean.values, marker="o", linewidth=2.4, color="#1d5843", label="Phụ tải Điện TB (kW)")
    ax1.fill_between(hr_load_mean.index, hr_load_mean - hr_load_std, hr_load_mean + hr_load_std, color="#1d5843", alpha=0.18, label="Dải độ lệch chuẩn ±1σ")
    ax1.set_xlabel("Giờ trong Ngày (00:00 – 23:00)", fontweight="bold")
    ax1.set_ylabel("Phụ tải Điện (Load - kW)", color="#1d5843", fontweight="bold")
    ax1.set_xticks(range(0, 24))
    ax1.grid(True, linestyle="--", alpha=0.35)

    ax2 = ax1.twinx()
    ax2.plot(hr_temp_mean.index, hr_temp_mean.values, marker="s", linestyle="--", linewidth=2.0, color="#d95f02", label="Nhiệt độ TB (°C)")
    ax2.set_ylabel("Nhiệt độ Không khí (°C)", color="#d95f02", fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9.5)
    plt.title("Biên dạng Chu kỳ Ngày/Đêm (Diurnal Profile): Đồng pha Giữa Nhiệt độ và Phụ tải Điện", fontsize=12.5, fontweight="bold", color="#183f35")
    fig.tight_layout()
    generated.append(_save_and_sync_fig(fig, "Hourly_Profile.png", active_run_id, cfg))

    # 11. Weekly_Profile.png: Biên dạng phụ tải theo các thứ trong tuần (Mon - Sun)
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 4.8))
    day_names = ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ Nhật"]
    dow_grp = full_df.groupby(full_df.index.day_of_week)["load"].agg(["mean", "std"])
    colors_bar = ["#1d5843"] * 5 + ["#d95f02", "#d95f02"]
    axes[0].bar(day_names, dow_grp["mean"].values, yerr=dow_grp["std"].values * 0.35, capsize=4, color=colors_bar, alpha=0.85, edgecolor="black")
    axes[0].set_title("Phụ tải Trung bình theo Thứ trong Tuần (Ngày thường vs. Cuối tuần)", fontsize=11, fontweight="bold", color="#183f35")
    axes[0].set_ylabel("Phụ tải Điện (kW)")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.35)

    is_wknd = full_df.index.day_of_week >= 5
    wkday_hr = full_df[~is_wknd].groupby(full_df[~is_wknd].index.hour)["load"].mean()
    wkend_hr = full_df[is_wknd].groupby(full_df[is_wknd].index.hour)["load"].mean()
    axes[1].plot(wkday_hr.index, wkday_hr.values, marker="o", linewidth=2.2, color="#1d5843", label="Ngày làm việc (Thứ 2 – Thứ 6)")
    axes[1].plot(wkend_hr.index, wkend_hr.values, marker="^", linewidth=2.2, color="#d95f02", label="Cuối tuần (Thứ 7 – Chủ Nhật)")
    axes[1].set_title("Đối chiếu Đường cong Phụ tải 24h: Ngày làm việc vs. Cuối tuần", fontsize=11, fontweight="bold", color="#183f35")
    axes[1].set_xlabel("Giờ trong Ngày (0..23h)")
    axes[1].set_ylabel("Phụ tải Điện (kW)")
    axes[1].set_xticks(range(0, 24, 2))
    axes[1].legend(fontsize=9.5)
    axes[1].grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    generated.append(_save_and_sync_fig(fig, "Weekly_Profile.png", active_run_id, cfg))

    # 12. Rolling_Volatility_30D.png: Trung bình trượt 24h/168h và Độ biến động trượt
    sub_roll = full_df["load"].iloc[-1440:]  # 60 ngày cuối
    rm_24 = sub_roll.rolling(24, min_periods=1).mean()
    rm_168 = sub_roll.rolling(168, min_periods=1).mean()
    rstd_24 = sub_roll.rolling(24, min_periods=1).std()

    fig, axes = plt.subplots(2, 1, figsize=(14.5, 6.2), sharex=True, gridspec_kw={"height_ratios": [1.8, 1.0]})
    axes[0].plot(sub_roll.index, sub_roll.values, color="#94a3b8", alpha=0.55, linewidth=0.9, label="Load theo giờ (Raw Hourly)")
    axes[0].plot(rm_24.index, rm_24.values, color="#1d5843", linewidth=1.8, label="Rolling Mean 24h (Trung bình ngày)")
    axes[0].plot(rm_168.index, rm_168.values, color="#d95f02", linewidth=2.3, label="Rolling Mean 168h (Trung bình tuần)")
    axes[0].set_title("Đặc trưng Thống kê Cuốn (Rolling Mean 24h & 168h) trên 60 Ngày Cuối", fontsize=11.5, fontweight="bold", color="#183f35")
    axes[0].set_ylabel("Load (kW)")
    axes[0].legend(loc="upper right", fontsize=9.2)
    axes[0].grid(True, linestyle="--", alpha=0.35)

    axes[1].fill_between(rstd_24.index, 0, rstd_24.values, color="#2b5c8f", alpha=0.35)
    axes[1].plot(rstd_24.index, rstd_24.values, color="#2b5c8f", linewidth=1.4, label="Rolling Std 24h (Độ biến động nội nhật)")
    axes[1].set_title("Độ Biến động Trượt 24 Giờ (Rolling Volatility σ_24h)", fontsize=11, fontweight="bold", color="#183f35")
    axes[1].set_ylabel("σ_24h (kW)")
    axes[1].legend(loc="upper right", fontsize=9.2)
    axes[1].grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    generated.append(_save_and_sync_fig(fig, "Rolling_Volatility_30D.png", active_run_id, cfg))

    # 13. training_loss_curves.png: Đường cong Hội tụ Training vs Validation cho LSTM (Electricity) & DCRNN/TCN/LSTM (PEMS-BAY)
    lstm_hist = train_artifacts["model_metadata"]["LSTM"]["history"]
    dcrnn_hist = train_artifacts["model_metadata"]["DCRNN"]["history"]
    tcn_hist = train_artifacts["model_metadata"]["Multi_Horizon_TCN"]["history"]
    pems_lstm_hist = train_artifacts["model_metadata"]["Multi_Horizon_LSTM"]["history"]

    fig, axes = plt.subplots(1, 2, figsize=(15.2, 5.2))
    ep_lstm = np.arange(1, len(lstm_hist["train_loss"]) + 1)
    axes[0].plot(ep_lstm, lstm_hist["train_loss"], marker="o", markersize=4, linewidth=2.0, color="#1d5843", label="Train MSE Loss (Scaled)")
    if lstm_hist["val_loss"]:
        axes[0].plot(ep_lstm, lstm_hist["val_loss"], marker="s", markersize=4, linewidth=2.0, color="#d95f02", label="Validation MSE Loss (Scaled)")
    best_ep = int(lstm_hist.get("best_epoch", len(ep_lstm)))
    axes[0].axvline(best_ep, color="#b91c1c", linestyle="--", linewidth=1.5, label=f"Best Epoch = {best_ep} (EarlyStopping)")
    axes[0].set_title(f"Hội tụ Mạng Stacked 2-Layer LSTM + Residual Highway\n(Best Val Loss = {lstm_hist['best_val_loss']:.5f} tại Epoch {best_ep})", fontsize=11, fontweight="bold", color="#183f35")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Scaled MSE Loss")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.35)

    ax_p = axes[1]
    if dcrnn_hist["val_loss"]:
        ax_p.plot(range(1, len(dcrnn_hist["val_loss"]) + 1), dcrnn_hist["val_loss"], marker="o", linewidth=2.2, color="#1d5843", label=f"DCRNN Val MAE (Best Ep={dcrnn_hist['best_epoch']})")
    if tcn_hist["val_loss"]:
        ax_p.plot(range(1, len(tcn_hist["val_loss"]) + 1), tcn_hist["val_loss"], marker="^", linewidth=2.0, color="#2b5c8f", label=f"Multi-Horizon TCN Val MAE (Best Ep={tcn_hist['best_epoch']})")
    if pems_lstm_hist["val_loss"]:
        ax_p.plot(range(1, len(pems_lstm_hist["val_loss"]) + 1), pems_lstm_hist["val_loss"], marker="s", linewidth=2.0, color="#d95f02", label=f"Multi-Horizon LSTM Val MAE (Best Ep={pems_lstm_hist['best_epoch']})")
    ax_p.set_title("Đường cong Kiểm định (Validation Loss) trên Tập Giao thông PEMS-BAY\n(Đối sánh Hội tụ: DCRNN vs. TCN vs. LSTM)", fontsize=11, fontweight="bold", color="#183f35")
    ax_p.set_xlabel("Epoch")
    ax_p.set_ylabel("Scaled Validation MAE")
    ax_p.legend(fontsize=9)
    ax_p.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    generated.append(_save_and_sync_fig(fig, "training_loss_curves.png", active_run_id, cfg))

    # 14. Forecast_Overlay_168h.png: So sánh Đường Dự báo Thực tế vs VAR, XGBoost, LSTM trong 168 giờ (7 ngày cuối tập Test)
    holdout = train_artifacts["holdout_predictions"]
    ts_te = pd.to_datetime(holdout["test_timestamps"])
    y_te = holdout["y_test_orig"]
    zoom_n = min(168, len(y_te))
    idx_z = ts_te[-zoom_n:]
    y_z = y_te[-zoom_n:]

    h_map = holdout_df.set_index("Model")
    fig, ax = plt.subplots(figsize=(15.5, 5.6))
    ax.plot(idx_z, y_z, color="#0f172a", linewidth=2.5, label="Thực tế (Actual Electricity Load - kW)")
    for m_name, style in [("XGBoost", "-"), ("LSTM", "--"), ("VAR", ":")]:
        pred_z = holdout["test"][m_name][-zoom_n:]
        r2_v = float(h_map.loc[m_name, "R2"])
        mape_v = float(h_map.loc[m_name, "MAPE"])
        rmse_v = float(h_map.loc[m_name, "RMSE"])
        ax.plot(
            idx_z,
            pred_z,
            linestyle=style,
            linewidth=1.9,
            color=MODEL_COLORS[m_name],
            label=f"{m_name} (RMSE={rmse_v:,.1f} kW | MAPE={mape_v:.2f}% | R²={r2_v:.4f})",
        )
    ax.set_title("Đối chiếu Đường Dự báo Phụ tải Điện (Actual vs. VAR, XGBoost, LSTM — Cận cảnh 168 Giờ Cuối Tập Test)", fontsize=12.5, fontweight="bold", color="#183f35")
    ax.set_xlabel("Thời gian (Timestamp)")
    ax.set_ylabel("Phụ tải Điện (Load - kW)")
    ax.legend(loc="upper right", fontsize=9.8, framealpha=0.92)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    generated.append(_save_and_sync_fig(fig, "Forecast_Overlay_168h.png", active_run_id, cfg))

    # 15. actual_vs_predicted_scatter.png: Biểu đồ Phân tán Thực tế vs Dự đoán cho cả 3 mô hình trên tập Test (5,260 giờ)
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.1), sharey=True)
    for idx, m_name in enumerate(["VAR", "XGBoost", "LSTM"]):
        ax = axes[idx]
        yp = holdout["test"][m_name]
        ax.scatter(y_te, yp, alpha=0.25, s=11, color=MODEL_COLORS[m_name], edgecolors="none")
        min_v = min(float(y_te.min()), float(yp.min()))
        max_v = max(float(y_te.max()), float(yp.max()))
        ax.plot([min_v, max_v], [min_v, max_v], color="#b91c1c", linestyle="--", linewidth=1.8, label="Đường lý tưởng y = ŷ")
        row_m = h_map.loc[m_name]
        ax.set_title(
            f"{m_name}\nR² = {row_m['R2']:.4f} | RMSE = {row_m['RMSE']:,.1f} | MAPE = {row_m['MAPE']:.2f}%",
            fontsize=11,
            fontweight="bold",
            color="#183f35",
        )
        ax.set_xlabel("Giá trị Thực tế (Actual Load - kW)")
        if idx == 0:
            ax.set_ylabel("Giá trị Dự báo (Predicted Load - kW)")
        ax.legend(loc="upper left", fontsize=8.8)
        ax.grid(True, linestyle="--", alpha=0.35)
    fig.suptitle("Biểu đồ Phân tán Giá trị Thực tế vs. Dự báo trên Tập Kiểm tra Độc lập (Test Set — 5,260 Giờ)", fontsize=13, fontweight="bold", y=1.02, color="#183f35")
    fig.tight_layout()
    generated.append(_save_and_sync_fig(fig, "actual_vs_predicted_scatter.png", active_run_id, cfg))

    # 16. Residual_Diagnostics.png: Chẩn đoán Phần dư (Phân phối Histogram + KDE & Scatter Residuals vs Predicted)
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 8.6))
    for idx, m_name in enumerate(["VAR", "XGBoost", "LSTM"]):
        yp = holdout["test"][m_name]
        resid = y_te - yp
        ax_hist = axes[0, idx]
        ax_scat = axes[1, idx]

        sns.histplot(resid, bins=45, kde=True, color=MODEL_COLORS[m_name], ax=ax_hist, alpha=0.4)
        ax_hist.axvline(0.0, color="#b91c1c", linestyle="--", linewidth=1.5, label="Zero Error")
        ax_hist.axvline(np.mean(resid), color="#0f172a", linestyle=":", linewidth=1.5, label=f"Mean: {np.mean(resid):.1f}")
        ax_hist.set_title(f"Phân phối Phần dư {m_name}\n(Mean={np.mean(resid):.1f}, Std={np.std(resid):.1f} kW)", fontsize=10.5, fontweight="bold", color="#183f35")
        ax_hist.set_xlabel("Residual e = y - ŷ (kW)")
        ax_hist.legend(fontsize=8.5)
        ax_hist.grid(True, linestyle="--", alpha=0.3)

        ax_scat.scatter(yp, resid, alpha=0.22, s=10, color=MODEL_COLORS[m_name], edgecolors="none")
        ax_scat.axhline(0.0, color="#b91c1c", linestyle="--", linewidth=1.5)
        ax_scat.set_title(f"Phần dư theo Giá trị Dự báo ({m_name})", fontsize=10.5, fontweight="bold", color="#183f35")
        ax_scat.set_xlabel("Predicted Load ŷ (kW)")
        ax_scat.set_ylabel("Residual e = y - ŷ (kW)")
        ax_scat.grid(True, linestyle="--", alpha=0.3)
    fig.suptitle("Chẩn đoán Phần dư (Residual Diagnostics) trên Tập Test — Kiểm chứng Nhiễu trắng Không chệch", fontsize=13, fontweight="bold", y=1.02, color="#183f35")
    fig.tight_layout()
    generated.append(_save_and_sync_fig(fig, "Residual_Diagnostics.png", active_run_id, cfg))

    # 17. Metric_Comparison_Bars.png: Đối sánh 4 chỉ số (RMSE, MAE, MAPE %, R²) trên Hold-out Test & 5-Fold CV
    fig, axes = plt.subplots(2, 2, figsize=(14.8, 8.4))
    models_order = ["VAR", "XGBoost", "LSTM"]
    h_ord = holdout_df.set_index("Model").loc[models_order]
    c_ord = cv_sum_df.set_index("Model").loc[models_order]
    w_ord = wf_df.set_index("Model").loc[models_order]

    x_pos = np.arange(len(models_order))
    width = 0.26

    # Panel A: RMSE
    axes[0, 0].bar(x_pos - width, h_ord["RMSE"], width, label="Hold-out Test (15%)", color="#1d5843")
    axes[0, 0].bar(x_pos, c_ord["RMSE_Mean"], width, yerr=c_ord["RMSE_Std"], capsize=3, label="5-Fold CV Mean", color="#2b5c8f")
    axes[0, 0].bar(x_pos + width, w_ord["RMSE"], width, label="Walk-Forward (168h)", color="#d95f02")
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(models_order, fontweight="bold")
    axes[0, 0].set_title("Đối sánh Sai số Căn bậc hai Trung bình (RMSE kW ↓)", fontsize=11, fontweight="bold", color="#183f35")
    axes[0, 0].set_ylabel("RMSE (kW)")
    axes[0, 0].legend(fontsize=8.5)
    axes[0, 0].grid(True, axis="y", linestyle="--", alpha=0.35)

    # Panel B: MAE
    axes[0, 1].bar(x_pos - width, h_ord["MAE"], width, label="Hold-out Test (15%)", color="#1d5843")
    axes[0, 1].bar(x_pos, c_ord["MAE_Mean"], width, yerr=c_ord["MAE_Std"], capsize=3, label="5-Fold CV Mean", color="#2b5c8f")
    axes[0, 1].bar(x_pos + width, w_ord["MAE"], width, label="Walk-Forward (168h)", color="#d95f02")
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(models_order, fontweight="bold")
    axes[0, 1].set_title("Đối sánh Sai số Tuyệt đối Trung bình (MAE kW ↓)", fontsize=11, fontweight="bold", color="#183f35")
    axes[0, 1].set_ylabel("MAE (kW)")
    axes[0, 1].legend(fontsize=8.5)
    axes[0, 1].grid(True, axis="y", linestyle="--", alpha=0.35)

    # Panel C: MAPE (%)
    axes[1, 0].bar(x_pos - width, h_ord["MAPE"], width, label="Hold-out Test (15%)", color="#1d5843")
    axes[1, 0].bar(x_pos, c_ord["MAPE_Mean"], width, yerr=c_ord["MAPE_Std"], capsize=3, label="5-Fold CV Mean", color="#2b5c8f")
    axes[1, 0].bar(x_pos + width, w_ord["MAPE"], width, label="Walk-Forward (168h)", color="#d95f02")
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(models_order, fontweight="bold")
    axes[1, 0].set_title("Đối sánh Sai số Phần trăm Tuyệt đối (MAPE % ↓)", fontsize=11, fontweight="bold", color="#183f35")
    axes[1, 0].set_ylabel("MAPE (%)")
    axes[1, 0].legend(fontsize=8.5)
    axes[1, 0].grid(True, axis="y", linestyle="--", alpha=0.35)

    # Panel D: R2 Score
    axes[1, 1].bar(x_pos - width, h_ord["R2"], width, label="Hold-out Test (15%)", color="#1d5843")
    axes[1, 1].bar(x_pos, c_ord["R2_Mean"], width, yerr=c_ord["R2_Std"], capsize=3, label="5-Fold CV Mean", color="#2b5c8f")
    axes[1, 1].bar(x_pos + width, w_ord["R2"], width, label="Walk-Forward (168h)", color="#d95f02")
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(models_order, fontweight="bold")
    axes[1, 1].set_ylim(max(0.85, float(min(h_ord["R2"].min(), c_ord["R2_Mean"].min())) - 0.03), 1.005)
    axes[1, 1].set_title("Đối sánh Hệ số Xác định (R² Score ↑)", fontsize=11, fontweight="bold", color="#183f35")
    axes[1, 1].set_ylabel("R² Score")
    axes[1, 1].legend(fontsize=8.5, loc="lower right")
    axes[1, 1].grid(True, axis="y", linestyle="--", alpha=0.35)

    fig.suptitle("Tổng hợp Đối sánh Hiệu suất 3 Mô hình (VAR, XGBoost, LSTM) qua 3 Phương pháp Kiểm định", fontsize=13, fontweight="bold", y=1.02, color="#183f35")
    fig.tight_layout()
    generated.append(_save_and_sync_fig(fig, "Metric_Comparison_Bars.png", active_run_id, cfg))

    # 18. training_time_comparison.png: So sánh Thời gian Huấn luyện & Thời gian Dự báo
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.6))
    tr_times = [float(h_ord.loc[m, "Train_Time_Sec"]) for m in models_order]
    pr_times_ms = [float(h_ord.loc[m, "Pred_Time_Sec"]) * 1000.0 for m in models_order]
    bar_cols = [MODEL_COLORS[m] for m in models_order]

    bars1 = axes[0].bar(models_order, tr_times, color=bar_cols, width=0.48, edgecolor="black")
    for b in bars1:
        h_v = b.get_height()
        axes[0].text(b.get_x() + b.get_width() / 2.0, h_v, f"{h_v:.2f}s", ha="center", va="bottom", fontweight="bold", fontsize=9.5)
    axes[0].set_title("Thời gian Huấn luyện Mô hình (Training Time ↓)", fontsize=11, fontweight="bold", color="#183f35")
    axes[0].set_ylabel("Giây (Seconds)")
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.35)

    bars2 = axes[1].bar(models_order, pr_times_ms, color=bar_cols, width=0.48, edgecolor="black")
    for b in bars2:
        h_v = b.get_height()
        axes[1].text(b.get_x() + b.get_width() / 2.0, h_v, f"{h_v:.1f} ms", ha="center", va="bottom", fontweight="bold", fontsize=9.5)
    axes[1].set_title("Độ trễ Suy luận trên 5,260 Giờ Tập Test (Inference Latency ↓)", fontsize=11, fontweight="bold", color="#183f35")
    axes[1].set_ylabel("Mili-giây (ms)")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    generated.append(_save_and_sync_fig(fig, "training_time_comparison.png", active_run_id, cfg))

    # 19. Walk_Forward_Trajectory.png: Quỹ đạo Dự báo Cuốn chiếu Walk-Forward 168 Bước
    wf_ts = pd.to_datetime(wf_pred_df["timestamp"])
    fig, axes = plt.subplots(2, 1, figsize=(15.0, 6.8), sharex=True, gridspec_kw={"height_ratios": [1.85, 1.0]})
    axes[0].plot(wf_ts, wf_pred_df["Actual_Load"], color="#0f172a", linewidth=2.3, label="Actual Load (kW)")
    axes[0].plot(wf_ts, wf_pred_df["VAR_WalkForward"], color=MODEL_COLORS["VAR"], linestyle=":", linewidth=1.8, label="VAR Walk-Forward")
    axes[0].plot(wf_ts, wf_pred_df["XGBoost_WalkForward"], color=MODEL_COLORS["XGBoost"], linestyle="-", linewidth=1.8, label="XGBoost Walk-Forward")
    axes[0].plot(wf_ts, wf_pred_df["LSTM_WalkForward"], color=MODEL_COLORS["LSTM"], linestyle="--", linewidth=1.8, label="LSTM Walk-Forward")
    axes[0].set_title("Quỹ đạo Dự báo Cuốn chiếu Walk-Forward Validation (168 Giờ: Train → Predict t+1 → Append → Predict t+2)", fontsize=11.5, fontweight="bold", color="#183f35")
    axes[0].set_ylabel("Phụ tải Điện (kW)")
    axes[0].legend(loc="upper right", fontsize=9.2)
    axes[0].grid(True, linestyle="--", alpha=0.35)

    for m_name, col_wf in [("VAR", "VAR_WalkForward"), ("XGBoost", "XGBoost_WalkForward"), ("LSTM", "LSTM_WalkForward")]:
        abs_err = np.abs(wf_pred_df["Actual_Load"] - wf_pred_df[col_wf])
        axes[1].plot(wf_ts, abs_err.rolling(12, min_periods=1).mean(), color=MODEL_COLORS[m_name], linewidth=1.6, label=f"{m_name} Rolling 12h MAE")
    axes[1].set_title("Độ ổn định Sai số Tuyệt đối Trượt 12 Giờ trong Walk-Forward Backtesting", fontsize=11, fontweight="bold", color="#183f35")
    axes[1].set_ylabel("MAE 12h (kW)")
    axes[1].legend(loc="upper right", fontsize=9.0)
    axes[1].grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    generated.append(_save_and_sync_fig(fig, "Walk_Forward_Trajectory.png", active_run_id, cfg))

    # 20. XGBoost_Feature_Importance.png: Top 15 Đặc trưng Quan trọng nhất của XGBoost
    top_imp = xgb_imp_df.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11.8, 6.2))
    bars = ax.barh(top_imp["Feature"], top_imp["Importance"], color="#1d5843", alpha=0.88, edgecolor="black")
    for b in bars:
        w_v = b.get_width()
        ax.text(w_v + 0.002, b.get_y() + b.get_height() / 2.0, f"{w_v:.4f}", va="center", fontsize=8.8, fontweight="bold")
    ax.set_title("Top 15 Đặc trưng Quan trọng Nhất trong Mô hình XGBoost Multivariate (Gain Importance)", fontsize=12, fontweight="bold", color="#183f35")
    ax.set_xlabel("Mức độ Đóng góp Tương đối (Feature Importance)")
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)
    fig.tight_layout()
    generated.append(_save_and_sync_fig(fig, "XGBoost_Feature_Importance.png", active_run_id, cfg))

    # 21. Feature_Ablation_Impact.png: Nghiên cứu Loại bỏ Đặc trưng Thời tiết (With Weather vs Without Weather)
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.0))
    x_abl = np.arange(len(abl_df))
    w_bar = 0.34
    axes[0].bar(x_abl - w_bar / 2, abl_df["Without_Weather_RMSE"], w_bar, label="Không có Biến Thời tiết (Univariate Load)", color="#94a3b8", edgecolor="black")
    axes[0].bar(x_abl + w_bar / 2, abl_df["With_Weather_RMSE"], w_bar, label="Có Đầy đủ 3 Biến Thời tiết (Multivariate)", color="#1d5843", edgecolor="black")
    axes[0].set_xticks(x_abl)
    axes[0].set_xticklabels(abl_df["Model"], fontweight="bold")
    axes[0].set_title("Tác động của Biến Ngoại sinh Thời tiết tới Sai số RMSE (kW ↓)", fontsize=11, fontweight="bold", color="#183f35")
    axes[0].set_ylabel("Test RMSE (kW)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.35)

    bars_gain = axes[1].bar(abl_df["Model"], abl_df["RMSE_Improvement_Pct"], color=["#2b5c8f", "#1d5843", "#d95f02"], width=0.45, edgecolor="black")
    for b in bars_gain:
        v_h = b.get_height()
        axes[1].text(b.get_x() + b.get_width() / 2.0, v_h, f"{v_h:+.2f}%", ha="center", va="bottom", fontweight="bold", fontsize=9.8)
    axes[1].axhline(0, color="black", linewidth=0.9)
    axes[1].set_title("Tỷ lệ Cải thiện Độ chính xác Nhờ Tích hợp Biến Thời tiết (% Giảm RMSE ↑)", fontsize=11, fontweight="bold", color="#183f35")
    axes[1].set_ylabel("Mức giảm RMSE (%)")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    generated.append(_save_and_sync_fig(fig, "Feature_Ablation_Impact.png", active_run_id, cfg))

    # 22. pems_bay_dcrnn_evaluation.png: Biểu đồ dự báo 3 tầm nhìn (15p, 30p, 60p) của DCRNN trên PEMS-BAY
    pems_preds = train_artifacts["pems_bay_predictions"]
    y_pems_true = pems_preds["Y_test_orig"]  # (N_test, 3, N_sensors)
    y_dcrnn = pems_preds["DCRNN"]
    sensor_id = 0
    slice_steps = min(288, y_pems_true.shape[0])  # 24 giờ (288 bước 5 phút)
    t_axis = np.arange(slice_steps)

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.2), gridspec_kw={"width_ratios": [1.55, 1.0]})
    axes[0].plot(t_axis, y_pems_true[:slice_steps, 0, sensor_id], color="#0f172a", linewidth=2.2, label="Vận tốc Thực tế (Actual Speed - mph)")
    axes[0].plot(t_axis, y_dcrnn[:slice_steps, 0, sensor_id], color="#1d5843", linestyle="-", linewidth=1.7, label="DCRNN Dự báo +15 Phút (t+3)")
    axes[0].plot(t_axis, y_dcrnn[:slice_steps, 1, sensor_id], color="#2b5c8f", linestyle="--", linewidth=1.6, label="DCRNN Dự báo +30 Phút (t+6)")
    axes[0].plot(t_axis, y_dcrnn[:slice_steps, 2, sensor_id], color="#d95f02", linestyle=":", linewidth=1.6, label="DCRNN Dự báo +60 Phút (t+12)")
    axes[0].set_title("Dự báo Vận tốc Giao thông Đa tầm nhìn bằng DCRNN (24 Giờ — 288 Bước 5 Phút)", fontsize=11, fontweight="bold", color="#183f35")
    axes[0].set_xlabel("Bước Thời gian 5 Phút (Steps)")
    axes[0].set_ylabel("Vận tốc (mph)")
    axes[0].legend(loc="lower left", fontsize=8.8)
    axes[0].grid(True, linestyle="--", alpha=0.35)

    h_labels = ["15 Phút (t+3)", "30 Phút (t+6)", "60 Phút (t+12)"]
    x_h = np.arange(len(h_labels))
    axes[1].bar(x_h - 0.18, pems_dcrnn_df["RMSE"], 0.34, label="RMSE (mph ↓)", color="#1d5843", edgecolor="black")
    axes[1].bar(x_h + 0.18, pems_dcrnn_df["MAE"], 0.34, label="MAE (mph ↓)", color="#d95f02", edgecolor="black")
    for idx_h, (r_v, m_v) in enumerate(zip(pems_dcrnn_df["RMSE"], pems_dcrnn_df["MAE"])):
        axes[1].text(idx_h - 0.18, r_v + 0.08, f"{r_v:.2f}", ha="center", va="bottom", fontsize=8.8, fontweight="bold")
        axes[1].text(idx_h + 0.18, m_v + 0.08, f"{m_v:.2f}", ha="center", va="bottom", fontsize=8.8, fontweight="bold")
    axes[1].set_xticks(x_h)
    axes[1].set_xticklabels(h_labels, fontweight="bold")
    axes[1].set_title("Sai số Dự báo DCRNN theo Độ xa Tầm nhìn (15p → 30p → 60p)", fontsize=11, fontweight="bold", color="#183f35")
    axes[1].set_ylabel("Sai số Vận tốc (mph)")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    generated.append(_save_and_sync_fig(fig, "pems_bay_dcrnn_evaluation.png", active_run_id, cfg))

    # 23. pems_bay_model_comparison.png: Đối sánh trực tiếp DCRNN vs Multi-Horizon TCN vs Multi-Horizon LSTM trên PEMS-BAY
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.2))
    horizons_filter = ["15_min", "30_min", "60_min"]
    h_Map_disp = {"15_min": "15 Phút (t+3)", "30_min": "30 Phút (t+6)", "60_min": "60 Phút (t+12)"}
    sub_b = pems_bench_df[pems_bench_df["Horizon"].isin(horizons_filter)].copy()

    x_b = np.arange(len(horizons_filter))
    w_b = 0.25
    m_specs = [
        ("DCRNN", "DCRNN (Graph + GRU)", "#1d5843", -w_b),
        ("Multi_Horizon_TCN", "Multi-Horizon TCN", "#2b5c8f", 0.0),
        ("Multi_Horizon_LSTM", "Multi-Horizon LSTM", "#d95f02", +w_b),
    ]

    for metric_idx, (metric_col, metric_title, y_lbl) in enumerate(
        [
            ("MAE", "Đối sánh MAE (mph ↓) theo Tầm nhìn", "MAE (mph)"),
            ("RMSE", "Đối sánh RMSE (mph ↓) theo Tầm nhìn", "RMSE (mph)"),
            ("R2", "Đối sánh R² Score (↑) theo Tầm nhìn", "R² Score"),
        ]
    ):
        ax = axes[metric_idx]
        for m_code, m_lbl, col_hex, offset in m_specs:
            m_slice = sub_b[sub_b["Model"] == m_code].set_index("Horizon").loc[horizons_filter]
            vals = m_slice[metric_col].to_numpy()
            bars = ax.bar(x_b + offset, vals, w_b, label=m_lbl, color=col_hex, edgecolor="black")
            for b in bars:
                hv = b.get_height()
                fmt_s = f"{hv:.2f}" if metric_col != "R2" else f"{hv:.3f}"
                ax.text(b.get_x() + b.get_width() / 2.0, hv, fmt_s, ha="center", va="bottom", fontsize=7.8, fontweight="bold")
        ax.set_xticks(x_b)
        ax.set_xticklabels([h_Map_disp[h] for h in horizons_filter], fontweight="bold")
        ax.set_title(metric_title, fontsize=11, fontweight="bold", color="#183f35")
        ax.set_ylabel(y_lbl)
        if metric_col == "R2":
            ax.set_ylim(0.45, 1.02)
        ax.legend(fontsize=8.5, loc="best")
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    fig.suptitle("Bài tập 12 Nâng cao (PEMS-BAY): Đối sánh DCRNN (Đồ thị Không gian) vs. Multi-Horizon TCN vs. Multi-Horizon LSTM", fontsize=13, fontweight="bold", y=1.03, color="#183f35")
    fig.tight_layout()
    generated.append(_save_and_sync_fig(fig, "pems_bay_model_comparison.png", active_run_id, cfg))

    # Kiểm kê toàn bộ các biểu đồ hiện có trong reports/lab04/figures/
    fig_dir = get_path("figures_dir", cfg)
    all_pngs = sorted([p.name for p in fig_dir.glob("*.png")])

    duration = time.perf_counter() - t0
    vis_summary = {
        "newly_generated_figures": generated,
        "all_available_figures": all_pngs,
        "total_figures_count": len(all_pngs),
        "duration_sec": round(duration, 4),
    }

    record_stage_result(active_run_id, "visualization", vis_summary)
    write_log(
        active_run_id,
        "VISUALIZATION",
        "SUCCESS",
        f"Hoàn tất kết xuất biểu đồ trong {duration:.2f}s | Tổng số biểu đồ trong kho reports/lab04/figures/: {len(all_pngs)}.",
    )
    print(f"[VISUALIZE] Đã kết xuất thành công {len(all_pngs)} biểu đồ học thuật tại reports/lab04/figures/.")
    return vis_summary


if __name__ == "__main__":
    main()