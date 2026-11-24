r"""
Mô-đun Phân tích Khám phá Dữ liệu (Exploratory Data Analysis - EDA) cho Bài thực hành Lab 04.
Thực hiện đầy đủ Mục 1 và Mục 2 trong đề bài (practice 4.pdf):
1. Kiểm tra cấu trúc dữ liệu, thống kê mô tả 13 chỉ tiêu (Mean, Std, Min, Q25, Median, Q75, Max,
   Skewness, Kurtosis, IQR Outliers, Z-score Outliers) cho cả 4 biến: load, temperature, humidity, wind_speed.
2. Phân tích ma trận tương quan tuyến tính (Pearson r) và phi tham số (Spearman rho).
3. Khởi tạo trực tiếp 8 biểu đồ EDA cốt lõi lưu vào reports/lab04/figures/:
   - multivariate_overview.png (đồng bộ 100% Min/Mean/Max ngày kèm ngày/tháng/năm dd/mm/yyyy)
   - feature_distributions_and_outliers.png
   - missing_values_audit.png
   - correlation_heatmap.png
   - weather_vs_load_joint.png
   - seasonal_decomposition.png
   - acf_pacf_load.png (đã loại bỏ hoàn toàn Lag 0, legend không che Lag 72h)
   - pems_bay_eda_overview.png
"""

import shutil
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
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
    - Output: DataFrame chứa 15 cột thống kê cho từng biến số.
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
    Purpose: Vẽ đầy đủ 8 biểu đồ EDA chuyên sâu (đồng bộ 100% Min/Mean/Max kèm ngày/tháng/năm dd/mm/yyyy,
             loại bỏ hoàn toàn Lag 0 ở ACF/PACF, không chồng lấn chú thích và có biểu đồ kiểm toán Missing Values).
    """
    plt.rcParams.update({
        "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial"],
        "axes.unicode_minus": False,
        "font.size": 10.5,
        "axes.titlesize": 12.0,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
    })
    saved_files: List[str] = []

    var_specs = [
        ("load", "Phụ tải điện (Load - kW)", PALETTE["load"], ",.0f", "kW"),
        ("temperature", "Nhiệt độ (Temperature - °C)", PALETTE["temperature"], ".1f", "°C"),
        ("humidity", "Độ ẩm (Humidity - %)", PALETTE["humidity"], ".1f", "%"),
        ("wind_speed", "Tốc độ gió (Wind Speed - m/s)", PALETTE["wind_speed"], ".2f", "m/s"),
    ]

    # =========================================================================
    # 1. EDA 01: Toàn cảnh Đa biến — Đồng bộ 100% Min/Mean/Max ngày kèm dd/mm/yyyy
    # =========================================================================
    fig, axes = plt.subplots(
        4, 2,
        figsize=(19, 16.2),
        sharex="col",
        gridspec_kw={"width_ratios": [1.7, 1.0], "hspace": 0.40, "wspace": 0.16},
    )
    zoom_df = df_elec.iloc[-336:]  # 14 ngày cuối (336 giờ)

    for idx, (col, label, color, fmt, unit) in enumerate(var_specs):
        ax_full = axes[idx, 0]
        ax_zoom = axes[idx, 1]

        s_hourly = df_elec[col].dropna()
        # Lọc bỏ ngày/tháng cuối không đủ dữ liệu (vd: 2015-01-01 chỉ có 1 giờ ban đêm)
        daily_grp = s_hourly.resample("1D")
        daily_s = daily_grp.mean()[daily_grp.count() >= 12]

        monthly_grp = s_hourly.resample("ME")
        monthly_s = monthly_grp.mean()[monthly_grp.count() >= 24 * 15]

        # Tính Min / Mean / Max đồng bộ 100% từ chính chuỗi daily_s kèm ngày/tháng/năm (dd/mm/yyyy)
        d_max_idx, d_min_idx = daily_s.idxmax(), daily_s.idxmin()
        d_max_val = float(daily_s.loc[d_max_idx])
        d_min_val = float(daily_s.loc[d_min_idx])
        d_mean_val = float(daily_s.mean())
        d_max_date = d_max_idx.strftime("%d/%m/%Y")
        d_min_date = d_min_idx.strftime("%d/%m/%Y")

        # Đường trung bình ngày (nền) + trung bình tháng (nổi bật) + Mean đồng bộ
        ax_full.plot(daily_s.index, daily_s.values, color=color, linewidth=0.9, alpha=0.35, label="Trung bình ngày")
        ax_full.plot(monthly_s.index, monthly_s.values, color="#0f172a", linewidth=2.1, marker="o", markersize=3.2, label="Trung bình tháng")
        ax_full.axhline(d_mean_val, color="#b91c1c", linestyle="--", linewidth=1.2, alpha=0.85, label="Trung bình toàn kỳ (Mean)")

        y_span = max(d_max_val - d_min_val, 1e-6)
        ax_full.set_ylim(d_min_val - 0.25 * y_span, d_max_val + 0.28 * y_span)

        ax_full.scatter([d_max_idx], [d_max_val], color="#dc2626", s=45, zorder=5)
        ax_full.scatter([d_min_idx], [d_min_val], color="#2563eb", s=45, zorder=5)

        ax_full.annotate(
            f"Max: {format(d_max_val, fmt)} {unit} ({d_max_date})",
            xy=(d_max_idx, d_max_val),
            xytext=(0, 10), textcoords="offset points", ha="center", va="bottom",
            fontsize=8.6, fontweight="bold", color="#991b1b",
            bbox=dict(boxstyle="round,pad=0.22", fc="#fef2f2", ec="#fca5a5", alpha=0.95),
        )
        ax_full.annotate(
            f"Min: {format(d_min_val, fmt)} {unit} ({d_min_date})",
            xy=(d_min_idx, d_min_val),
            xytext=(0, -12), textcoords="offset points", ha="center", va="top",
            fontsize=8.6, fontweight="bold", color="#1e40af",
            bbox=dict(boxstyle="round,pad=0.22", fc="#eff6ff", ec="#93c5fd", alpha=0.95),
        )

        ax_full.set_ylabel(label, fontweight="bold")
        ax_full.set_title(
            f"{label}  |  Min ngày: {format(d_min_val, fmt)} ({d_min_date})  •  Mean: {format(d_mean_val, fmt)}  •  Max ngày: {format(d_max_val, fmt)} ({d_max_date})",
            fontsize=9.8, fontweight="bold", color="#183f35", loc="left",
        )
        ax_full.grid(True, linestyle="--", alpha=0.35)

        # Cột phải: Cận cảnh 14 ngày cuối (hourly)
        ax_zoom.plot(zoom_df.index, zoom_df[col].values, color=color, linewidth=1.45)
        z_min, z_max = float(zoom_df[col].min()), float(zoom_df[col].max())
        z_span = max(z_max - z_min, 1e-6)
        ax_zoom.set_ylim(z_min - 0.12 * z_span, z_max + 0.15 * z_span)
        ax_zoom.set_title(
            f"14 Ngày Cuối (336h) — Mean: {format(float(zoom_df[col].mean()), fmt)} {unit}",
            fontsize=10.2, fontweight="bold", color="#183f35", loc="left",
        )
        ax_zoom.grid(True, linestyle="--", alpha=0.35)

        if idx == 3:
            ax_full.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            ax_full.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
            ax_full.tick_params(axis="x", rotation=15)

            ax_zoom.xaxis.set_major_locator(mdates.DayLocator(interval=3))
            ax_zoom.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
            ax_zoom.tick_params(axis="x", rotation=20)

    # Thanh Legend chung ở phía trên cùng để không che nhãn Max/Min trong các ô
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.35, 0.962),
        ncol=3, frameon=True, facecolor="white", framealpha=0.95, fontsize=9.5,
    )

    fig.suptitle(
        "EDA 01: Biến động Chuỗi Thời gian Đa biến (Trung bình Ngày/Tháng & Cận cảnh 14 Ngày Cuối)",
        fontsize=15, fontweight="bold", y=0.992, color="#183f35",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _sync_figure(fig, "multivariate_overview.png", run_id, cfg)
    saved_files.append("multivariate_overview.png")

    # =========================================================================
    # 2. EDA 02: Phân phối Xác suất & Boxplot (4x2 rộng rãi, không che KDE)
    # =========================================================================
    fig, axes = plt.subplots(
        4, 2,
        figsize=(17.5, 15.0),
        gridspec_kw={"width_ratios": [1.45, 1.0], "hspace": 0.42, "wspace": 0.18},
    )
    for idx, (col, label, color, fmt, unit) in enumerate(var_specs):
        s = df_elec[col].dropna()
        ax_hist = axes[idx, 0]
        ax_box = axes[idx, 1]

        sns.histplot(s, bins=45, kde=True, color=color, ax=ax_hist, alpha=0.35, edgecolor="white")
        mean_v, med_v = float(s.mean()), float(s.median())
        ax_hist.axvline(mean_v, color="#b91c1c", linestyle="--", linewidth=1.7, label=f"Mean: {format(mean_v, fmt)} {unit}")
        ax_hist.axvline(med_v, color="#0f172a", linestyle=":", linewidth=1.7, label=f"Median: {format(med_v, fmt)} {unit}")

        _, y_top = ax_hist.get_ylim()
        ax_hist.set_ylim(0, y_top * 1.24)
        ax_hist.set_title(
            f"{label}  |  Skew = {s.skew():+.3f}  •  Kurtosis = {s.kurtosis():+.3f}",
            fontsize=11, fontweight="bold", color="#183f35", loc="left",
        )
        ax_hist.set_xlabel("")
        ax_hist.set_ylabel("Tần suất")
        ax_hist.legend(loc="upper right", ncol=2, frameon=True, facecolor="white", framealpha=0.95)
        ax_hist.grid(True, linestyle="--", alpha=0.3)

        sns.boxplot(x=s, color=color, ax=ax_box, width=0.42, fliersize=3.0, boxprops=dict(alpha=0.65))
        q25, q75 = float(s.quantile(0.25)), float(s.quantile(0.75))
        iqr = q75 - q25
        n_out = int(((s < q25 - 1.5 * iqr) | (s > q75 + 1.5 * iqr)).sum())
        ax_box.set_title(
            f"Boxplot IQR — Outliers: {n_out:,} ({100.0 * n_out / len(s):.2f}%)  |  IQR = [{format(q25, fmt)}, {format(q75, fmt)}]",
            fontsize=10.5, fontweight="bold", color="#183f35", loc="left",
        )
        ax_box.set_xlabel(unit)
        ax_box.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle(
        "EDA 02: Phân phối Xác suất (Histogram + KDE) & Kiểm định Ngoại lai Boxplot (IQR)",
        fontsize=15, fontweight="bold", y=0.995, color="#183f35",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    _sync_figure(fig, "feature_distributions_and_outliers.png", run_id, cfg)
    saved_files.append("feature_distributions_and_outliers.png")

    # =========================================================================
    # 3. EDA 03: Trực quan hóa Kiểm tra Missing Values & Tính liên tục Thời gian
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 5.2), gridspec_kw={"wspace": 0.25})

    audit_names = [
        "load (Elec)", "temperature (Elec)", "humidity (Elec)", "wind_speed (Elec)",
        f"PEMS-BAY ({df_pems.shape[1]} sensors)",
    ]
    missing_counts = [int(df_elec[c].isna().sum()) for c in ["load", "temperature", "humidity", "wind_speed"]]
    missing_counts.append(int(df_pems.isna().sum().sum()))
    total_cells = [len(df_elec)] * 4 + [int(df_pems.size)]
    missing_pcts = [100.0 * m / max(t, 1) for m, t in zip(missing_counts, total_cells)]

    y_pos = np.arange(len(audit_names))
    bar_colors = ["#10b981" if m == 0 else "#ef4444" for m in missing_counts]
    axes[0].barh(y_pos, missing_pcts, color=bar_colors, height=0.52, alpha=0.85, edgecolor="#0f172a")
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(audit_names, fontweight="bold")
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, max(max(missing_pcts) * 1.3, 5.0))
    axes[0].set_xlabel("Tỷ lệ dữ liệu thiếu (%)", fontweight="bold")
    axes[0].set_title("Kiểm tra Giá trị Thiếu (Missing Values Audit)", fontsize=11.5, fontweight="bold", color="#183f35")
    for idx_b, (cnt, pct, tot) in enumerate(zip(missing_counts, missing_pcts, total_cells)):
        axes[0].text(
            pct + 0.15, idx_b,
            f"Missing: {cnt:,} / {tot:,} ({pct:.2f}%) — ĐẠT" if cnt == 0 else f"Missing: {cnt:,} ({pct:.2f}%)",
            va="center", fontsize=9.5, fontweight="bold",
            color="#065f46" if cnt == 0 else "#991b1b",
        )
    axes[0].grid(True, axis="x", linestyle="--", alpha=0.35)

    elec_diffs = df_elec.index.to_series().diff().dropna().dt.total_seconds() / 3600.0
    pems_diffs = df_pems.index.to_series().diff().dropna().dt.total_seconds() / 60.0
    elec_gaps = int((elec_diffs != 1.0).sum())
    pems_gaps = int((pems_diffs != 5.0).sum())

    gap_labels = ["Electricity Load\n(Chuẩn Δt = 1 giờ)", "PEMS-BAY Traffic\n(Chuẩn Δt = 5 phút)"]
    valid_counts = [int((elec_diffs == 1.0).sum()), int((pems_diffs == 5.0).sum())]
    gap_counts = [elec_gaps, pems_gaps]

    x_g = np.arange(len(gap_labels))
    axes[1].bar(x_g - 0.18, valid_counts, width=0.34, color="#1d5843", label="Bước thời gian hợp lệ (Đúng chu kỳ)")
    axes[1].bar(x_g + 0.18, gap_counts, width=0.34, color="#dc2626", label="Bước bị đứt quãng (Time Gap)")
    axes[1].set_xticks(x_g)
    axes[1].set_xticklabels(gap_labels, fontweight="bold")
    axes[1].set_ylabel("Số lượng khoảng thời gian (Intervals)", fontweight="bold")
    axes[1].set_ylim(0, max(valid_counts) * 1.22)
    axes[1].set_title("Kiểm tra Tính Liên tục Mốc Thời gian (Timestamp Continuity)", fontsize=11.5, fontweight="bold", color="#183f35")
    for i_g, (v_cnt, g_cnt) in enumerate(zip(valid_counts, gap_counts)):
        axes[1].text(i_g - 0.18, v_cnt + max(valid_counts) * 0.02, f"{v_cnt:,}", ha="center", fontweight="bold", fontsize=9.5, color="#183f35")
        axes[1].text(i_g + 0.18, g_cnt + max(valid_counts) * 0.02, f"{g_cnt:,}", ha="center", fontweight="bold", fontsize=9.5, color="#991b1b" if g_cnt > 0 else "#065f46")
    axes[1].legend(loc="upper right", frameon=True, facecolor="white")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.35)

    fig.suptitle("EDA 03: Kiểm định Toàn vẹn Dữ liệu — Missing Values & Tính Liên tục Thời gian", fontsize=14, fontweight="bold", y=1.02, color="#183f35")
    fig.tight_layout()
    _sync_figure(fig, "missing_values_audit.png", run_id, cfg)
    saved_files.append("missing_values_audit.png")

    # =========================================================================
    # 4. EDA 04: Ma trận Tương quan Pearson (r) & Spearman (rho)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2), gridspec_kw={"wspace": 0.22})
    sns.heatmap(
        pearson_df, annot=True, fmt=".3f", cmap="RdYlGn", vmin=-1.0, vmax=1.0,
        cbar=True, square=True, linewidths=1.2, ax=axes[0],
        annot_kws={"size": 11.5, "weight": "bold"},
    )
    axes[0].set_title("Ma trận Tương quan Tuyến tính Pearson (r)", fontsize=12, fontweight="bold", color="#183f35")

    sns.heatmap(
        spearman_df, annot=True, fmt=".3f", cmap="PuBuGn", vmin=-1.0, vmax=1.0,
        cbar=True, square=True, linewidths=1.2, ax=axes[1],
        annot_kws={"size": 11.5, "weight": "bold"},
    )
    axes[1].set_title("Ma trận Tương quan Hạng Spearman (ρ)", fontsize=12, fontweight="bold", color="#183f35")
    fig.suptitle("EDA 04: Phân tích Tương quan Đa biến Giữa Phụ tải Điện (Load) & Thời tiết (Mục 2)", fontsize=14.5, fontweight="bold", y=1.02, color="#183f35")
    fig.tight_layout()
    _sync_figure(fig, "correlation_heatmap.png", run_id, cfg)
    saved_files.append("correlation_heatmap.png")

    # =========================================================================
    # 5. EDA 05: Quan hệ Phi tuyến Scatter + Đường xu hướng Bậc 2
    # =========================================================================
    sample_df = df_elec.sample(n=min(4000, len(df_elec)), random_state=42)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), gridspec_kw={"wspace": 0.2})
    exog_specs = [
        ("temperature", "Nhiệt độ (°C)", PALETTE["temperature"]),
        ("humidity", "Độ ẩm (%)", PALETTE["humidity"]),
        ("wind_speed", "Tốc độ gió (m/s)", PALETTE["wind_speed"]),
    ]
    for idx, (col, x_label, color) in enumerate(exog_specs):
        ax = axes[idx]
        x_val = sample_df[col].to_numpy()
        y_val = sample_df["load"].to_numpy()
        ax.scatter(x_val, y_val, alpha=0.22, s=14, color=color, edgecolors="none", label="Mẫu quan sát (Hourly)")
        poly_coef = np.polyfit(x_val, y_val, deg=2)
        x_grid = np.linspace(x_val.min(), x_val.max(), 200)
        y_trend = np.polyval(poly_coef, x_grid)
        ax.plot(x_grid, y_trend, color="#0f172a", linewidth=2.4, label="Xu hướng Bậc 2 (Quadratic Fit)")

        y_lo, y_hi = ax.get_ylim()
        ax.set_ylim(y_lo, y_hi + 0.16 * (y_hi - y_lo))
        r_val = float(pearson_df.loc["load", col])
        rho_val = float(spearman_df.loc["load", col])
        ax.set_title(f"Load vs. {x_label}\nPearson r = {r_val:+.3f}  |  Spearman ρ = {rho_val:+.3f}", fontsize=11.5, fontweight="bold", color="#183f35")
        ax.set_xlabel(x_label, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("Phụ tải Điện (Load - kW)", fontweight="bold")
        ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92)
        ax.grid(True, linestyle="--", alpha=0.35)

    fig.suptitle("EDA 05: Mối quan hệ Phi tuyến giữa Biến Ngoại sinh Thời tiết và Phụ tải Điện", fontsize=14.5, fontweight="bold", y=1.02, color="#183f35")
    fig.tight_layout()
    _sync_figure(fig, "weather_vs_load_joint.png", run_id, cfg)
    saved_files.append("weather_vs_load_joint.png")

    # =========================================================================
    # 6. EDA 06: Phân rã Chuỗi Thời gian 4 Tầng (30 ngày cuối - 720 giờ)
    # =========================================================================
    sub_s = df_elec["load"].iloc[-720:]
    trend_s = sub_s.rolling(window=24, center=True, min_periods=1).mean()
    detrended = sub_s - trend_s
    hourly_pattern = detrended.groupby(detrended.index.hour).transform("mean")
    resid_s = sub_s - trend_s - hourly_pattern

    fig, axes = plt.subplots(4, 1, figsize=(17, 11.2), sharex=True, gridspec_kw={"hspace": 0.34})
    decomp_items = [
        (sub_s, "1. Chuỗi Thực tế Quan sát (Observed Load - kW)", "#1d5843", False),
        (trend_s, "2. Thành phần Xu hướng Dài hạn (Trend - 24h Moving Average)", "#d95f02", False),
        (hourly_pattern, "3. Thành phần Mùa vụ Ngày/Đêm (Diurnal Period = 24h)", "#2b5c8f", True),
        (resid_s, f"4. Thành phần Nhiễu Phần dư (Residual | Mean={resid_s.mean():+.1f}, Std={resid_s.std():.1f} kW)", "#64748b", True),
    ]
    for ax, (series_item, title_str, col_hex, zero_line) in zip(axes, decomp_items):
        ax.plot(series_item.index, series_item.values, color=col_hex, linewidth=1.35)
        if zero_line:
            ax.axhline(0, color="#0f172a", linestyle="--", linewidth=0.9, alpha=0.7)
        ax.set_title(title_str, fontsize=11, fontweight="bold", loc="left", color="#183f35")
        ax.set_ylabel("kW", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.35)
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=3))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))
    axes[-1].tick_params(axis="x", rotation=15)

    fig.suptitle("EDA 06: Phân rã Chuỗi Thời gian Phụ tải Điện (Additive Seasonal Decomposition — 720 Giờ Cuối)", fontsize=14.5, fontweight="bold", y=0.995, color="#183f35")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    _sync_figure(fig, "seasonal_decomposition.png", run_id, cfg)
    saved_files.append("seasonal_decomposition.png")

    # =========================================================================
    # 7. EDA 07: ACF (1..72h) & PACF (1..48h) — LOẠI BỎ HOÀN TOÀN LAG 0
    # =========================================================================
    load_vals = df_elec["load"].dropna().to_numpy()
    acf_vals = acf(load_vals, nlags=72, fft=True)[1:]       # Cắt bỏ Lag 0
    pacf_vals = pacf(load_vals, nlags=48, method="ywm")[1:]  # Cắt bỏ Lag 0
    conf_bound = 1.96 / np.sqrt(len(load_vals))

    fig, axes = plt.subplots(1, 2, figsize=(17.5, 5.8), gridspec_kw={"wspace": 0.18})

    lags_acf = np.arange(1, len(acf_vals) + 1)
    axes[0].vlines(lags_acf, 0, acf_vals, color="#1d5843", linewidth=1.8)
    axes[0].scatter(lags_acf, acf_vals, color="#1d5843", s=24, zorder=4)
    for peak_lag in (24, 48, 72):
        axes[0].axvline(peak_lag, color="#d95f02", linestyle="--", alpha=0.8, linewidth=1.3)
        axes[0].text(
            peak_lag, acf_vals[peak_lag - 1] + 0.04,
            f"Lag {peak_lag}h\n({acf_vals[peak_lag - 1]:.2f})",
            ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#b45309",
        )
    axes[0].fill_between(lags_acf, -conf_bound, conf_bound, color="#ef4444", alpha=0.14, label="95% Confidence Band")
    axes[0].axhline(0, color="black", linewidth=0.9)
    axes[0].set_xlim(0.5, 74.5)
    axes[0].set_ylim(min(float(acf_vals.min()), -0.1) - 0.16, max(float(acf_vals.max()), 0.5) + 0.22)
    axes[0].set_xticks([1, 12, 24, 36, 48, 60, 72])
    axes[0].set_title("Hàm Tự tương quan ACF (Lag 1 → 72h, Đã loại Lag 0)", fontsize=12, fontweight="bold", color="#183f35")
    axes[0].set_xlabel("Độ trễ (Lag Hours)", fontweight="bold")
    axes[0].set_ylabel("Hệ số ACF", fontweight="bold")
    # Đặt legend ở góc dưới phải để không bao giờ che nhãn Lag 72h
    axes[0].legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.94)
    axes[0].grid(True, linestyle="--", alpha=0.35)

    lags_pacf = np.arange(1, len(pacf_vals) + 1)
    axes[1].vlines(lags_pacf, 0, pacf_vals, color="#2b5c8f", linewidth=1.8)
    axes[1].scatter(lags_pacf, pacf_vals, color="#2b5c8f", s=24, zorder=4)
    axes[1].axvline(24, color="#d95f02", linestyle="--", alpha=0.85, linewidth=1.4, label="Lookback Window = 24h")
    axes[1].fill_between(lags_pacf, -conf_bound, conf_bound, color="#ef4444", alpha=0.14, label="95% Confidence Band")
    axes[1].axhline(0, color="black", linewidth=0.9)
    axes[1].set_xlim(0.5, 49.0)
    p_min, p_max = float(pacf_vals.min()), float(pacf_vals.max())
    p_span = max(p_max - p_min, 0.2)
    axes[1].set_ylim(p_min - 0.16 * p_span, p_max + 0.22 * p_span)
    axes[1].set_xticks([1, 6, 12, 18, 24, 30, 36, 42, 48])
    axes[1].set_title("Hàm Tự tương quan Riêng phần PACF (Lag 1 → 48h, Đã loại Lag 0)", fontsize=12, fontweight="bold", color="#183f35")
    axes[1].set_xlabel("Độ trễ (Lag Hours)", fontweight="bold")
    axes[1].set_ylabel("Hệ số PACF", fontweight="bold")
    axes[1].legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.94)
    axes[1].grid(True, linestyle="--", alpha=0.35)

    fig.suptitle("EDA 07: Phân tích Tự tương quan ACF & PACF Xác định Độ trễ Tối ưu (Không lấy Lag 0)", fontsize=14.5, fontweight="bold", y=1.02, color="#183f35")
    fig.tight_layout()
    _sync_figure(fig, "acf_pacf_load.png", run_id, cfg)
    saved_files.append("acf_pacf_load.png")

    # =========================================================================
    # 8. EDA 08: Khám phá Không gian - Thời gian PEMS-BAY (Mục 12 Nâng cao)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.6), gridspec_kw={"wspace": 0.24})
    speed_flat = df_pems.values.flatten()
    speed_sample = speed_flat[:: max(1, len(speed_flat) // 25000)]
    sns.histplot(speed_sample, bins=45, kde=True, color="#1d5843", ax=axes[0], alpha=0.4)
    p_mean, p_med = float(np.mean(speed_flat)), float(np.median(speed_flat))
    axes[0].axvline(p_mean, color="#b91c1c", linestyle="--", linewidth=1.7, label=f"Mean: {p_mean:.1f} mph")
    axes[0].axvline(p_med, color="#0f172a", linestyle=":", linewidth=1.7, label=f"Median: {p_med:.1f} mph")
    _, py_top = axes[0].get_ylim()
    axes[0].set_ylim(0, py_top * 1.2)
    axes[0].set_title(
        f"Phân phối Vận tốc {df_pems.shape[1]} Trạm Cảm biến\n(Min={np.min(speed_flat):.1f} | Max={np.max(speed_flat):.1f} mph)",
        fontsize=11, fontweight="bold", color="#183f35",
    )
    axes[0].set_xlabel("Vận tốc Giao thông (mph)", fontweight="bold")
    axes[0].legend(loc="upper left", frameon=True, facecolor="white")
    axes[0].grid(True, linestyle="--", alpha=0.3)

    im = axes[1].imshow(adj_mx[:80, :80], cmap="YlGnBu", aspect="auto")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_title(
        f"Ma trận Kề Đồ thị adj_mx_bay.pkl\n(Cận cảnh 80×80 / Tổng {adj_mx.shape[0]}×{adj_mx.shape[1]} Nút)",
        fontsize=11, fontweight="bold", color="#183f35",
    )
    axes[1].set_xlabel("Chỉ số Trạm đích (Sensor j)", fontweight="bold")
    axes[1].set_ylabel("Chỉ số Trạm nguồn (Sensor i)", fontweight="bold")

    hourly_speed = df_pems.groupby(df_pems.index.hour).mean()
    mean_across_sensors = hourly_speed.mean(axis=1)
    q25_sensors = hourly_speed.quantile(0.25, axis=1)
    q75_sensors = hourly_speed.quantile(0.75, axis=1)
    axes[2].plot(mean_across_sensors.index, mean_across_sensors.values, marker="o", color="#d95f02", linewidth=2.2, label="Vận tốc TB toàn mạng")
    axes[2].fill_between(mean_across_sensors.index, q25_sensors.values, q75_sensors.values, color="#d95f02", alpha=0.22, label="Dải phân vị IQR (25%–75%)")
    axes[2].set_ylim(float(q25_sensors.min()) - 6.0, float(q75_sensors.max()) + 4.5)
    axes[2].set_title("Biến thiên Vận tốc theo Giờ trong Ngày\n(Giảm mạnh giờ Cao điểm 8h & 17h)", fontsize=11, fontweight="bold", color="#183f35")
    axes[2].set_xlabel("Giờ trong ngày (00:00 – 23:00)", fontweight="bold")
    axes[2].set_ylabel("Vận tốc (mph)", fontweight="bold")
    axes[2].set_xticks(range(0, 24, 3))
    axes[2].legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.94)
    axes[2].grid(True, linestyle="--", alpha=0.35)

    fig.suptitle("EDA 08: Khám phá Cấu trúc Đồ thị Không gian & Chu kỳ Giao thông PEMS-BAY (Mục 12)", fontsize=14.5, fontweight="bold", y=1.02, color="#183f35")
    fig.tight_layout()
    _sync_figure(fig, "pems_bay_eda_overview.png", run_id, cfg)
    saved_files.append("pems_bay_eda_overview.png")

    return saved_files


def main(run_id: Optional[str] = None) -> Dict[str, Any]:
    r"""
    Purpose: Thực thi toàn bộ quy trình Phân tích Khám phá Dữ liệu (EDA - Mục 1 & 2) và xuất 8 biểu đồ EDA.
    """
    t0 = time.perf_counter()
    cfg = load_config()
    ensure_directories(cfg)
    active_run_id = get_active_run_id(run_id)

    write_log(active_run_id, "EDA", "START", "Bắt đầu phân tích khám phá dữ liệu (EDA) và khởi tạo 8 biểu đồ chuyên sâu.")

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