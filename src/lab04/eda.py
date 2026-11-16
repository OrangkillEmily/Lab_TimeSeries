r"""
Mô-đun Phân tích Khám phá Dữ liệu (Exploratory Data Analysis - EDA) cho Bài thực hành Lab 04.
Thực hiện đầy đủ Mục 1 và Mục 2 trong đề bài (practice 4.pdf):
1. Kiểm tra cấu trúc dữ liệu, thống kê mô tả 12 chỉ tiêu (Mean, Std, Min, Q25, Median, Q75, Max,
   Skewness, Kurtosis, IQR Outliers, Z-score Outliers) cho cả 4 biến: load, temperature, humidity, wind_speed.
2. Phân tích ma trận tương quan tuyến tính (Pearson r) và phi tham số (Spearman rho).
3. Khởi tạo trực tiếp 7 biểu đồ EDA cốt lõi lưu vào reports/lab04/figures/:
   - multivariate_overview.png
   - feature_distributions_and_outliers.png
   - correlation_heatmap.png
   - weather_vs_load_joint.png
   - seasonal_decomposition.png
   - acf_pacf_load.png
   - pems_bay_eda_overview.png
"""

import shutil
import time
import matplotlib.dates as mdates
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.stattools import acf, pacf

from src.lab04.data_loader import load_electricity_dataset, load_pems_bay_dataset
from src.lab04.tracking import get_active_run_id, record_stage_result, write_log
from src.lab04.utils import ensure_directories, get_path, load_config, save_json

warnings.filterwarnings("ignore", category=FutureWarning)

PALETTE = {
    "load": "#1d5843",
    "temperature": "#d95f02",
    "humidity": "#2b5c8f",
    "wind_speed": "#7570b3",
}


def compute_descriptive_statistics(
    df: pd.DataFrame,
    iqr_mult: float = 1.5,
    z_thresh: float = 3.0,
) -> pd.DataFrame:
    r"""
    Purpose: Tính bảng thống kê mô tả toàn diện và phát hiện điểm ngoại lai (Outliers) theo IQR và Z-score.
    Input-Output:
        - Input: df (pd.DataFrame), iqr_mult (float = 1.5), z_thresh (float = 3.0).
        - Output: DataFrame chứa 13 cột thống kê cho từng biến số.
    How: Với mỗi cột, tính Mean, Std, Min, Q25, Median, Q75, Max, Skewness, Kurtosis,
         khoảng [Q25 - 1.5*IQR, Q75 + 1.5*IQR] và ngưỡng |Z| > 3.0.
    """
    rows: List[Dict[str, Any]] = []
    n_total = max(len(df), 1)
    for col in df.columns:
        s = df[col].dropna().astype(float)
        mean_val = float(s.mean())
        std_val = float(s.std())
        q25 = float(s.quantile(0.25))
        med = float(s.quantile(0.50))
        q75 = float(s.quantile(0.75))
        iqr = q75 - q25
        low_fence = q25 - iqr_mult * iqr
        high_fence = q75 + iqr_mult * iqr
        iqr_outliers = int(((s < low_fence) | (s > high_fence)).sum())
        z_scores = np.abs((s - mean_val) / (std_val + 1e-9))
        z_outliers = int((z_scores > z_thresh).sum())

        rows.append(
            {
                "Variable": col,
                "Count": int(len(s)),
                "Missing": int(df[col].isna().sum()),
                "Mean": round(mean_val, 4),
                "Std": round(std_val, 4),
                "Min": round(float(s.min()), 4),
                "Q25": round(q25, 4),
                "Median": round(med, 4),
                "Q75": round(q75, 4),
                "Max": round(float(s.max()), 4),
                "Skewness": round(float(s.skew()), 4),
                "Kurtosis": round(float(s.kurtosis()), 4),
                "IQR_Outliers": iqr_outliers,
                "IQR_Outlier_Pct": round(100.0 * iqr_outliers / n_total, 2),
                "ZScore_Outliers": z_outliers,
            }
        )
    return pd.DataFrame(rows)


def compute_correlation_analysis(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    r"""
    Purpose: Tính ma trận tương quan tuyến tính Pearson (r) và tương quan hạng Spearman (rho).
    Input-Output:
        - Input: df (pd.DataFrame chứa load, temperature, humidity, wind_speed).
        - Output: Tuple (pearson_df, spearman_df, summary_dict).
    How: Gọi df.corr(method='pearson') và df.corr(method='spearman'), xác định biến thời tiết
         có mức độ ảnh hưởng mạnh nhất tới 'load' (theo |r|).
    """
    pearson_df = df.corr(method="pearson").round(4)
    spearman_df = df.corr(method="spearman").round(4)

    exog_cols = [c for c in df.columns if c != "load"]
    load_corrs = {c: float(pearson_df.loc["load", c]) for c in exog_cols}
    dominant_var = max(exog_cols, key=lambda c: abs(load_corrs[c]))

    summary = {
        "pearson_with_load": load_corrs,
        "spearman_with_load": {c: float(spearman_df.loc["load", c]) for c in exog_cols},
        "dominant_weather_variable": dominant_var,
        "dominant_pearson_r": round(load_corrs[dominant_var], 4),
    }
    return pearson_df, spearman_df, summary


def _sync_figure(fig: plt.Figure, filename: str, run_id: str, cfg: Dict[str, Any]) -> Path:
    r"""
    Purpose: Lưu biểu đồ vào kho trung tâm reports/lab04/figures/, bản sao theo run_id và docs/lab04/figures/.
    Input-Output:
        - Input: fig (plt.Figure), filename (str), run_id (str), cfg (Dict).
        - Output: Đường dẫn Path chính tại reports/lab04/figures/<filename>.
    How: Gọi fig.savefig(dpi=220) và shutil.copy2 sang các thư mục đồng bộ.
    """
    fig_dir = get_path("figures_dir", cfg)
    run_fig_dir = fig_dir / run_id
    docs_fig_dir = get_path("docs_dir", cfg) / "figures"

    fig_dir.mkdir(parents=True, exist_ok=True)
    run_fig_dir.mkdir(parents=True, exist_ok=True)
    docs_fig_dir.mkdir(parents=True, exist_ok=True)

    main_path = fig_dir / filename
    fig.savefig(main_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    for dst_dir in (run_fig_dir, docs_fig_dir):
        try:
            shutil.copy2(main_path, dst_dir / filename)
        except Exception:
            pass
    return main_path


def generate_eda_figures(
    df_elec: pd.DataFrame,
    df_pems: pd.DataFrame,
    adj_mx: np.ndarray,
    pearson_df: pd.DataFrame,
    spearman_df: pd.DataFrame,
    run_id: str,
    cfg: Dict[str, Any],
) -> List[str]:
    r"""
    Purpose: Vẽ đầy đủ 7 biểu đồ EDA chuyên sâu phục vụ hiển thị trực tiếp trong lab04.ipynb và báo cáo HTML/PDF.
    Input-Output:
        - Input: df_elec, df_pems, adj_mx, pearson_df, spearman_df, run_id, cfg.
        - Output: Danh sách tên 7 tệp ảnh .png đã khởi tạo.
    How: Sử dụng Matplotlib & Seaborn với bảng màu học thuật đồng nhất.
    """
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Segoe UI", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    saved_files: List[str] = []
    
    # 1. Biểu đồ 1: Tổng quan 4 biến toàn thời gian + Cận cảnh 14 ngày (336 giờ)
    fig, axes = plt.subplots(
        4, 2,
        figsize=(16.5, 11.5),
        sharex="col",
        gridspec_kw={"width_ratios": [1.65, 1.05]},
    )
    var_specs = [
        ("load", "Phụ tải điện (Load - kW)", PALETTE["load"]),
        ("temperature", "Nhiệt độ (Temperature - °C)", PALETTE["temperature"]),
        ("humidity", "Độ ẩm (Humidity - %)", PALETTE["humidity"]),
        ("wind_speed", "Tốc độ gió (Wind Speed - m/s)", PALETTE["wind_speed"]),
    ]
    zoom_df = df_elec.iloc[-336:]
    for idx, (col, label, color) in enumerate(var_specs):
        ax_full = axes[idx, 0]
        ax_zoom = axes[idx, 1]
        daily_s = df_elec[col].resample("1D").mean()
        ax_full.plot(daily_s.index, daily_s.values, color=color, linewidth=1.1, alpha=0.9)
        ax_full.set_ylabel(label, fontsize=9.5, fontweight="bold")
        ax_full.grid(True, linestyle="--", alpha=0.35)
        if idx == 0:
            ax_full.set_title("Toàn cảnh Chuỗi Thời gian Đa biến (Trung bình Ngày 2011–2014)", fontsize=11.5, fontweight="bold", color="#183f35")

        ax_zoom.plot(zoom_df.index, zoom_df[col].values, color=color, linewidth=1.35)
        ax_zoom.grid(True, linestyle="--", alpha=0.35)
        if idx == 0:
            ax_zoom.set_title("Cận cảnh Chu kỳ Ngày/Đêm (14 Ngày Cuối — 336 Giờ)", fontsize=11.5, fontweight="bold", color="#183f35")
        if idx == 3:
            ax_full.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            ax_full.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
            ax_full.tick_params(axis="x", rotation=15)

            ax_zoom.xaxis.set_major_locator(mdates.DayLocator(interval=3))
            ax_zoom.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))
            ax_zoom.tick_params(axis="x", rotation=18)
    fig.suptitle("EDA 01: Biến động Chuỗi Thời gian Đa biến Điện năng & Thời tiết (Mục 1 - Practice 4)", fontsize=13.5, fontweight="bold", y=1.01, color="#183f35")
    fig.tight_layout()
    _sync_figure(fig, "multivariate_overview.png", run_id, cfg)
    saved_files.append("multivariate_overview.png")

    # 2. Biểu đồ 2: Phân phối Histogram/KDE + Boxplot Phát hiện Ngoại lai cho 4 biến
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.2), gridspec_kw={"height_ratios": [2.2, 1.0]})
    for idx, (col, label, color) in enumerate(var_specs):
        s = df_elec[col].dropna()
        ax_hist = axes[0, idx]
        ax_box = axes[1, idx]
        sns.histplot(s, bins=45, kde=True, color=color, ax=ax_hist, alpha=0.35, edgecolor="white")
        ax_hist.axvline(s.mean(), color="#b91c1c", linestyle="--", linewidth=1.5, label=f"Mean: {s.mean():.1f}")
        ax_hist.axvline(s.median(), color="#0f172a", linestyle=":", linewidth=1.5, label=f"Median: {s.median():.1f}")
        ax_hist.set_title(label, fontsize=10.5, fontweight="bold", color="#183f35")
        ax_hist.set_xlabel("")
        ax_hist.legend(fontsize=8, loc="upper right")
        ax_hist.grid(True, linestyle="--", alpha=0.3)

        sns.boxplot(x=s, color=color, ax=ax_box, width=0.45, fliersize=2.5, boxprops=dict(alpha=0.65))
        q25, q75 = s.quantile(0.25), s.quantile(0.75)
        iqr = q75 - q25
        n_out = int(((s < q25 - 1.5 * iqr) | (s > q75 + 1.5 * iqr)).sum())
        ax_box.set_xlabel(f"IQR Outliers: {n_out} ({100.0*n_out/len(s):.2f}%)", fontsize=8.8, fontweight="bold")
        ax_box.grid(True, linestyle="--", alpha=0.3)
    fig.suptitle("EDA 02: Phân phối Xác suất (Histogram + KDE) & Kiểm định Ngoại lai Boxplot (IQR)", fontsize=13, fontweight="bold", y=1.02, color="#183f35")
    fig.tight_layout()
    _sync_figure(fig, "feature_distributions_and_outliers.png", run_id, cfg)
    saved_files.append("feature_distributions_and_outliers.png")

    # 3. Biểu đồ 3: Ma trận Tương quan Pearson (r) & Spearman (rho)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    sns.heatmap(
        pearson_df,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=-1.0,
        vmax=1.0,
        cbar=True,
        square=True,
        linewidths=1.0,
        ax=axes[0],
        annot_kws={"size": 10.5, "weight": "bold"},
    )
    axes[0].set_title("Ma trận Tương quan Tuyến tính Pearson (r)", fontsize=11.5, fontweight="bold", color="#183f35")

    sns.heatmap(
        spearman_df,
        annot=True,
        fmt=".3f",
        cmap="PuBuGn",
        vmin=-1.0,
        vmax=1.0,
        cbar=True,
        square=True,
        linewidths=1.0,
        ax=axes[1],
        annot_kws={"size": 10.5, "weight": "bold"},
    )
    axes[1].set_title("Ma trận Tương quan Hạng Spearman (ρ)", fontsize=11.5, fontweight="bold", color="#183f35")
    fig.suptitle("EDA 03: Phân tích Tương quan Đa biến Giữa Phụ tải Điện (Load) & Thời tiết (Mục 2)", fontsize=13, fontweight="bold", y=1.02, color="#183f35")
    fig.tight_layout()
    _sync_figure(fig, "correlation_heatmap.png", run_id, cfg)
    saved_files.append("correlation_heatmap.png")

    # 4. Biểu đồ 4: Quan hệ Phi tuyến Scatter + Đường xu hướng bậc 2 giữa 3 biến thời tiết và Load
    sample_df = df_elec.sample(n=min(4000, len(df_elec)), random_state=42)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    exog_specs = [
        ("temperature", "Nhiệt độ (°C)", PALETTE["temperature"]),
        ("humidity", "Độ ẩm (%)", PALETTE["humidity"]),
        ("wind_speed", "Tốc độ gió (m/s)", PALETTE["wind_speed"]),
    ]
    for idx, (col, x_label, color) in enumerate(exog_specs):
        ax = axes[idx]
        x_val = sample_df[col].to_numpy()
        y_val = sample_df["load"].to_numpy()
        ax.scatter(x_val, y_val, alpha=0.22, s=12, color=color, edgecolors="none", label="Mẫu quan sát (Hourly)")
        poly_coef = np.polyfit(x_val, y_val, deg=2)
        x_grid = np.linspace(x_val.min(), x_val.max(), 200)
        y_trend = np.polyval(poly_coef, x_grid)
        ax.plot(x_grid, y_trend, color="#0f172a", linewidth=2.2, label="Xu hướng Bậc 2 (Quadratic Fit)")
        r_val = float(pearson_df.loc["load", col])
        rho_val = float(spearman_df.loc["load", col])
        ax.set_title(f"Load vs. {x_label}\n(Pearson r = {r_val:+.3f} | Spearman ρ = {rho_val:+.3f})", fontsize=10.5, fontweight="bold", color="#183f35")
        ax.set_xlabel(x_label, fontsize=9.5, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("Phụ tải Điện (Load - kW)", fontsize=9.5, fontweight="bold")
        ax.legend(fontsize=8.2, loc="best")
        ax.grid(True, linestyle="--", alpha=0.35)
    fig.suptitle("EDA 04: Mối quan hệ Phi tuyến giữa Biến Ngoại sinh Thời tiết và Phụ tải Điện", fontsize=13, fontweight="bold", y=1.03, color="#183f35")
    fig.tight_layout()
    _sync_figure(fig, "weather_vs_load_joint.png", run_id, cfg)
    saved_files.append("weather_vs_load_joint.png")

    # 5. Biểu đồ 5: Phân rã Chuỗi Thời gian 4 Tầng (Observed - Trend - Daily Seasonal - Residual)
    sub_s = df_elec["load"].iloc[-720:]  # 30 ngày cuối (720 giờ)
    trend_s = sub_s.rolling(window=24, center=True, min_periods=1).mean()
    detrended = sub_s - trend_s
    hourly_pattern = detrended.groupby(detrended.index.hour).transform("mean")
    resid_s = sub_s - trend_s - hourly_pattern

    fig, axes = plt.subplots(4, 1, figsize=(14.5, 8.8), sharex=True)
    decomp_items = [
        (sub_s, "1. Chuỗi Thực tế Quan sát (Observed Load - kW)", "#1d5843"),
        (trend_s, "2. Thành phần Xu hướng Dài hạn (Trend - 24h Moving Average)", "#d95f02"),
        (hourly_pattern, "3. Thành phần Mùa vụ Ngày/Đêm (Diurnal Seasonal Period = 24h)", "#2b5c8f"),
        (resid_s, "4. Thành phần Nhiễu Phần dư (Irregular Residual)", "#64748b"),
    ]
    for ax, (series_item, title_str, col_hex) in zip(axes, decomp_items):
        ax.plot(series_item.index, series_item.values, color=col_hex, linewidth=1.25)
        ax.set_title(title_str, fontsize=10, fontweight="bold", loc="left", color="#183f35")
        ax.grid(True, linestyle="--", alpha=0.35)
    fig.suptitle("EDA 05: Phân rã Chuỗi Thời gian Phụ tải Điện (Additive Seasonal Decomposition — 720 Giờ)", fontsize=13, fontweight="bold", y=1.01, color="#183f35")
    fig.tight_layout()
    _sync_figure(fig, "seasonal_decomposition.png", run_id, cfg)
    saved_files.append("seasonal_decomposition.png")

    # 6. Biểu đồ 6: Hàm Tự tương quan ACF (72 giờ) & Tự tương quan Riêng phần PACF (48 giờ)
    load_vals = df_elec["load"].dropna().to_numpy()
    acf_vals = acf(load_vals, nlags=72, fft=True)
    pacf_vals = pacf(load_vals, nlags=48, method="ywm")
    conf_bound = 1.96 / np.sqrt(len(load_vals))

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 4.6))
    lags_acf = np.arange(len(acf_vals))
    axes[0].vlines(lags_acf, 0, acf_vals, color="#1d5843", linewidth=1.6)
    axes[0].scatter(lags_acf, acf_vals, color="#1d5843", s=18)
    for peak_lag in (24, 48, 72):
        axes[0].axvline(peak_lag, color="#d95f02", linestyle="--", alpha=0.75, linewidth=1.2)
    axes[0].axhline(conf_bound, color="#b91c1c", linestyle=":", linewidth=1.0)
    axes[0].axhline(-conf_bound, color="#b91c1c", linestyle=":", linewidth=1.0)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Hàm Tự tương quan ACF (Đỉnh chu kỳ rõ rệt tại Lag 24h, 48h, 72h)", fontsize=11, fontweight="bold", color="#183f35")
    axes[0].set_xlabel("Độ trễ (Lag Hours)")
    axes[0].set_ylabel("Hệ số ACF")
    axes[0].grid(True, linestyle="--", alpha=0.3)

    lags_pacf = np.arange(len(pacf_vals))
    axes[1].vlines(lags_pacf, 0, pacf_vals, color="#2b5c8f", linewidth=1.6)
    axes[1].scatter(lags_pacf, pacf_vals, color="#2b5c8f", s=18)
    axes[1].axvline(24, color="#d95f02", linestyle="--", alpha=0.8, linewidth=1.2, label="Window Lookback = 24h")
    axes[1].axhline(conf_bound, color="#b91c1c", linestyle=":", linewidth=1.0)
    axes[1].axhline(-conf_bound, color="#b91c1c", linestyle=":", linewidth=1.0)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Hàm Tự tương quan Riêng phần PACF (Cơ sở chọn Lookback = 24h)", fontsize=11, fontweight="bold", color="#183f35")
    axes[1].set_xlabel("Độ trễ (Lag Hours)")
    axes[1].set_ylabel("Hệ số PACF")
    axes[1].legend(fontsize=8.5, loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.3)

    fig.suptitle("EDA 06: Phân tích Tự tương quan ACF & PACF Xác định Độ trễ Tối ưu (Lag Selection)", fontsize=13, fontweight="bold", y=1.02, color="#183f35")
    fig.tight_layout()
    _sync_figure(fig, "acf_pacf_load.png", run_id, cfg)
    saved_files.append("acf_pacf_load.png")

    # 7. Biểu đồ 7: Khám phá dữ liệu Không gian - Thời gian PEMS-BAY (Mục 12 Nâng cao)
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.7))
    speed_flat = df_pems.values.flatten()
    speed_sample = speed_flat[:: max(1, len(speed_flat) // 25000)]
    sns.histplot(speed_sample, bins=45, kde=True, color="#1d5843", ax=axes[0], alpha=0.4)
    axes[0].axvline(np.mean(speed_flat), color="#b91c1c", linestyle="--", linewidth=1.6, label=f"Mean: {np.mean(speed_flat):.1f} mph")
    axes[0].set_title(f"Phân phối Vận tốc {df_pems.shape[1]} Trạm Cảm biến PEMS-BAY", fontsize=10.5, fontweight="bold", color="#183f35")
    axes[0].set_xlabel("Vận tốc Giao thông (mph)")
    axes[0].legend(fontsize=8.5)
    axes[0].grid(True, linestyle="--", alpha=0.3)

    im = axes[1].imshow(adj_mx[:80, :80], cmap="YlGnBu", aspect="auto")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_title(f"Ma trận Kề Đồ thị Cảm biến adj_mx_bay.pkl\n(Cận cảnh 80×80 / Tổng {adj_mx.shape[0]}×{adj_mx.shape[1]} Nút)", fontsize=10.5, fontweight="bold", color="#183f35")
    axes[1].set_xlabel("Chỉ số Trạm đích (Sensor j)")
    axes[1].set_ylabel("Chỉ số Trạm nguồn (Sensor i)")

    hourly_speed = df_pems.groupby(df_pems.index.hour).mean()
    mean_across_sensors = hourly_speed.mean(axis=1)
    q25_sensors = hourly_speed.quantile(0.25, axis=1)
    q75_sensors = hourly_speed.quantile(0.75, axis=1)
    axes[2].plot(mean_across_sensors.index, mean_across_sensors.values, marker="o", color="#d95f02", linewidth=2.0, label="Vận tốc TB toàn mạng")
    axes[2].fill_between(mean_across_sensors.index, q25_sensors.values, q75_sensors.values, color="#d95f02", alpha=0.22, label="Dải phân vị IQR (25%–75%)")
    axes[2].set_title("Biến thiên Vận tốc theo Giờ trong Ngày\n(Suy giảm rõ rệt giờ Cao điểm Sáng 8h & Chiều 17h)", fontsize=10.5, fontweight="bold", color="#183f35")
    axes[2].set_xlabel("Giờ trong ngày (00:00 – 23:00)")
    axes[2].set_ylabel("Vận tốc (mph)")
    axes[2].set_xticks(range(0, 24, 3))
    axes[2].legend(fontsize=8.5, loc="lower left")
    axes[2].grid(True, linestyle="--", alpha=0.35)

    fig.suptitle("EDA 07: Khám phá Cấu trúc Đồ thị Không gian & Chu kỳ Giao thông PEMS-BAY (Mục 12)", fontsize=13, fontweight="bold", y=1.02, color="#183f35")
    fig.tight_layout()
    _sync_figure(fig, "pems_bay_eda_overview.png", run_id, cfg)
    saved_files.append("pems_bay_eda_overview.png")

    return saved_files


def main(run_id: Optional[str] = None) -> Dict[str, Any]:
    r"""
    Purpose: Thực thi toàn bộ quy trình Phân tích Khám phá Dữ liệu (EDA - Mục 1 & 2) và xuất 7 biểu đồ EDA.
    Input-Output:
        - Input: run_id (Optional[str]).
        - Output: Từ điển chứa kết quả thống kê mô tả, tương quan Pearson/Spearman và danh sách 7 biểu đồ EDA.
    How: Nạp dữ liệu từ data_loader, tính thống kê, lưu các bảng CSV vào reports/lab04/metrics/
         và vẽ 7 biểu đồ EDA vào reports/lab04/figures/.
    """
    t0 = time.perf_counter()
    cfg = load_config()
    ensure_directories(cfg)
    active_run_id = get_active_run_id(run_id)

    write_log(active_run_id, "EDA", "START", "Bắt đầu phân tích khám phá dữ liệu (EDA) và khởi tạo 7 biểu đồ chuyên sâu.")

    df_elec, elec_meta = load_electricity_dataset(cfg)
    df_pems, adj_mx, pems_meta = load_pems_bay_dataset(cfg)

    elec_cfg = cfg.get("data", {}).get("electricity", {})
    stats_df = compute_descriptive_statistics(
        df_elec,
        iqr_mult=float(elec_cfg.get("outlier_iqr_multiplier", 1.5)),
        z_thresh=float(elec_cfg.get("outlier_zscore_threshold", 3.0)),
    )
    pearson_df, spearman_df, corr_summary = compute_correlation_analysis(df_elec)

    metrics_dir = get_path("metrics_dir", cfg)
    stats_df.to_csv(metrics_dir / "eda_descriptive_stats.csv", index=False)
    pearson_df.to_csv(metrics_dir / "eda_pearson_correlation.csv")
    spearman_df.to_csv(metrics_dir / "eda_spearman_correlation.csv")

    eda_figures = generate_eda_figures(df_elec, df_pems, adj_mx, pearson_df, spearman_df, active_run_id, cfg)

    hourly_profile = df_elec.groupby(df_elec.index.hour)["load"].agg(["mean", "std"]).round(2)
    peak_hour = int(hourly_profile["mean"].idxmax())
    trough_hour = int(hourly_profile["mean"].idxmin())

    duration = time.perf_counter() - t0
    eda_summary: Dict[str, Any] = {
        "descriptive_stats": stats_df.to_dict(orient="records"),
        "correlation_summary": corr_summary,
        "diurnal_peaks": {
            "peak_hour": peak_hour,
            "peak_load_kw": float(hourly_profile.loc[peak_hour, "mean"]),
            "trough_hour": trough_hour,
            "trough_load_kw": float(hourly_profile.loc[trough_hour, "mean"]),
        },
        "pems_bay_summary": pems_meta,
        "generated_figures": eda_figures,
        "duration_sec": round(duration, 4),
    }

    save_json(eda_summary, metrics_dir / "eda_summary.json")
    record_stage_result(active_run_id, "eda", eda_summary)

    write_log(
        active_run_id,
        "EDA",
        "SUCCESS",
        f"Hoàn tất EDA trong {duration:.2f}s | Đã xuất {len(eda_figures)} biểu đồ EDA | Biến thời tiết mạnh nhất: {corr_summary['dominant_weather_variable']} (r={corr_summary['dominant_pearson_r']:+.4f}).",
    )
    print(
        f"[EDA] Hoàn tất thống kê 4 biến & xuất {len(eda_figures)} biểu đồ EDA | "
        f"Dominant Weather Var: {corr_summary['dominant_weather_variable']} (r = {corr_summary['dominant_pearson_r']:+.4f})"
    )
    return eda_summary


if __name__ == "__main__":
    main()