"""
Module khám phá và phân tích dữ liệu (`src/lab04/eda.py`) cho dự án Lab 04 Time Series.

Trách nhiệm của module (Thực hiện Mục 3 — Data Understanding của `practice 4.pdf`):
1. Kiểm tra giá trị khuyết (`Missing values`), phát hiện giá trị ngoại lai (`Outliers`) bằng
   quy tắc `IQR (1.5 x IQR)` và `Z-Score (|z| > 3.0)` trên cả 4 biến `[load, temperature, humidity, wind_speed]`.
2. Phân tích tính mùa vụ (`Seasonality`) theo chu kỳ ngày (`24 giờ`) và chu kỳ tuần (`168 giờ`),
   kèm định lượng tương quan chéo theo độ trễ (`Lag Cross-Correlation`) để làm rõ hiện tượng
   `Lag Heterogeneity` trong Slide 4 Chương 4–5.
3. Gọi `src/lab04/visualize.py` sinh đầy đủ các biểu đồ bắt buộc:
   - Biểu đồ `load` theo thời gian (`load_timeseries_overview.png`)
   - Biểu đồ `load` vs `temperature` (`load_vs_temperature_scatter.png`)
   - Ma trận tương quan (`correlation_heatmap.png`)
   - Biểu đồ mùa vụ chu kỳ ngày & chu kỳ tuần (`seasonality_daily_weekly.png`)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.lab04.data_loader import load_electricity_dataset, load_pems_bay_dataset
from src.lab04.tracking import create_run, update_progress, write_log
from src.lab04.utils import ensure_dir, get_project_root, load_config, save_json
from src.lab04.visualize import (
    create_stage_figure_dir,
    plot_correlation_heatmap,
    plot_load_timeseries_overview,
    plot_load_vs_temperature,
    plot_seasonality_profiles,
)

ROOT_DIR = get_project_root()


def run_eda_analysis(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    What: Thực hiện thống kê mô tả, kiểm tra Missing values, Outliers, Seasonality (ngày/tuần)
          và ma trận tương quan giữa điện năng và thời tiết.
    Why: Đáp ứng toàn bộ yêu cầu của Mục 3 (Data Understanding) và cung cấp số liệu định lượng
         để trả lời Câu hỏi 1 ở Mục 11 (*"Biến thời tiết nào ảnh hưởng mạnh nhất đến điện năng?"*).
    Input:
        - df (pd.DataFrame): Bảng dữ liệu 4 biến `[load, temperature, humidity, wind_speed]`.
        - config (Optional[Dict[str, Any]]): Cấu hình từ `config.yaml`.
    Output:
        - Dict[str, Any]: Từ điển chứa bảng thống kê, số lượng outliers, ma trận tương quan Pearson,
          độ trễ tương quan mạnh nhất (`best_lag_hours`) và biên độ mùa vụ ngày/tuần.
    How: Tính các phân vị `Q1, Median, Q3`, ngưỡng `IQR`, `Z-score`, ma trận tương quan `df.corr()`
         và tương quan trễ $r(\tau)$ với $\tau \in \{0..24\}$ giờ.
    """
    cfg = config or load_config()
    iqr_mult = float(cfg.get("preprocessing", {}).get("outlier_detection", {}).get("iqr_multiplier", 1.5))
    z_thresh = float(cfg.get("preprocessing", {}).get("outlier_detection", {}).get("zscore_threshold", 3.0))

    stats_rows = []
    outlier_summary: Dict[str, Any] = {}

    for col in df.columns:
        s = df[col].dropna()
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - iqr_mult * iqr
        upper = q3 + iqr_mult * iqr
        iqr_outliers = int(((s < lower) | (s > upper)).sum())

        mean_val = float(s.mean())
        std_val = float(s.std()) if float(s.std()) > 1e-9 else 1.0
        z_outliers = int((np.abs((s - mean_val) / std_val) > z_thresh).sum())
        missing_cnt = int(df[col].isna().sum())

        stats_rows.append({
            "Variable": col,
            "Count": int(s.count()),
            "Missing_Count": missing_cnt,
            "Missing_Pct": round(100.0 * missing_cnt / max(1, len(df)), 4),
            "Mean": round(mean_val, 4),
            "Std": round(std_val, 4),
            "Min": round(float(s.min()), 4),
            "Q25": round(q1, 4),
            "Median": round(float(s.median()), 4),
            "Q75": round(q3, 4),
            "Max": round(float(s.max()), 4),
            "Skewness": round(float(s.skew()), 4),
            "Kurtosis": round(float(s.kurtosis()), 4),
            "IQR_Outliers": iqr_outliers,
            "ZScore_Outliers": z_outliers,
        })
        outlier_summary[col] = {
            "iqr_outliers": iqr_outliers,
            "zscore_outliers": z_outliers,
            "iqr_bounds": [round(lower, 3), round(upper, 3)],
        }

    df_clean = df.ffill().bfill()
    corr_df = df_clean.corr(method="pearson")

    # Phân tích độ trễ tác động (Lag heterogeneity: tau = 0..24 giờ)
    lag_correlations: Dict[str, Dict[str, float]] = {}
    load_series = df_clean["load"]
    for w_col in ["temperature", "humidity", "wind_speed"]:
        best_lag = 0
        best_corr = 0.0
        lag_curve = {}
        for lag in range(0, 25):
            c_val = float(load_series.corr(df_clean[w_col].shift(lag)))
            if np.isnan(c_val):
                c_val = 0.0
            lag_curve[f"lag_{lag}h"] = round(c_val, 4)
            if abs(c_val) > abs(best_corr):
                best_corr = c_val
                best_lag = lag
        lag_correlations[w_col] = {
            "instant_corr_lag0": round(float(corr_df.loc["load", w_col]), 4),
            "max_abs_corr": round(float(best_corr), 4),
            "optimal_lag_hours": int(best_lag),
        }

    # Xác định biến thời tiết ảnh hưởng mạnh nhất đến điện năng
    dominant_weather_var = max(
        ["temperature", "humidity", "wind_speed"],
        key=lambda c: abs(lag_correlations[c]["max_abs_corr"]),
    )

    # Phân tích Seasonality chu kỳ ngày (24h) và chu kỳ tuần (7 ngày)
    hourly_profile = df_clean.groupby(df_clean.index.hour)["load"].mean()
    dow_profile = df_clean.groupby(df_clean.index.dayofweek)["load"].mean()

    seasonality_summary = {
        "daily_peak_hour": int(hourly_profile.idxmax()),
        "daily_Peak_load_mean": round(float(hourly_profile.max()), 3),
        "daily_trough_hour": int(hourly_profile.idxmin()),
        "daily_trough_load_mean": round(float(hourly_profile.min()), 3),
        "daily_amplitude": round(float(hourly_profile.max() - hourly_profile.min()), 3),
        "weekday_mean_load": round(float(df_clean[df_clean.index.dayofweek < 5]["load"].mean()), 3),
        "weekend_mean_load": round(float(df_clean[df_clean.index.dayofweek >= 5]["load"].mean()), 3),
        "weekly_drop_on_weekend_pct": round(
            100.0
            * (float(df_clean[df_clean.index.dayofweek < 5]["load"].mean()) - float(df_clean[df_clean.index.dayofweek >= 5]["load"].mean()))
            / max(1e-6, float(df_clean[df_clean.index.dayofweek < 5]["load"].mean())),
            2,
        ),
    }

    return {
        "stats_df": pd.DataFrame(stats_rows),
        "corr_df": corr_df,
        "outlier_summary": outlier_summary,
        "lag_correlations": lag_correlations,
        "dominant_weather_variable": dominant_weather_var,
        "seasonality_summary": seasonality_summary,
    }


def main(run_id: Optional[str] = None) -> Dict[str, Any]:
    """
    What: Thực thi toàn bộ giai đoạn `EDA` (Mục 3 đề bài) và lưu bảng thống kê + biểu đồ vào `reports/lab04/`.
    Why: Đảm bảo toàn bộ kết quả khám phá dữ liệu được lưu tự động cho báo cáo `.pdf`, `.html`, `.md`.
    Input:
        - run_id (Optional[str]): Mã định danh phiên chạy.
    Output:
        - Dict[str, Any]: Tóm tắt kết quả phân tích EDA và danh sách các biểu đồ đã xuất.
    How: Gọi `run_eda_analysis()`, lưu CSV/JSON vào `reports/lab04/metrics/` và vẽ 4 biểu đồ EDA ở 300 DPI.
    """
    t0 = time.perf_counter()
    cfg = load_config()
    active_run_id = run_id or create_run()
    write_log(active_run_id, "EDA", "START", "Bắt đầu phân tích khám phá dữ liệu (EDA) và kiểm tra Seasonality.")

    df_elec, elec_meta = load_electricity_dataset(cfg)
    df_pems, adj_mx, pems_meta = load_pems_bay_dataset(cfg)

    eda_res = run_eda_analysis(df_elec, cfg)
    metrics_dir = ROOT_DIR / cfg["paths"]["metrics_dir"]
    ensure_dir(metrics_dir)

    stats_csv = metrics_dir / "eda_descriptive_statistics.csv"
    corr_csv = metrics_dir / "eda_correlation_matrix.csv"
    eda_res["stats_df"].to_csv(stats_csv, index=False, encoding="utf-8")
    eda_res["corr_df"].round(4).to_csv(corr_csv, encoding="utf-8")

    stage_fig_dir = create_stage_figure_dir(active_run_id, "eda")
    saved_figs = []
    saved_figs.extend(plot_load_timeseries_overview(df_elec, stage_dir=stage_fig_dir))
    saved_figs.extend(plot_load_vs_temperature(df_elec, eda_res["lag_correlations"], stage_dir=stage_fig_dir))
    saved_figs.extend(plot_correlation_heatmap(eda_res["corr_df"], stage_dir=stage_fig_dir))
    saved_figs.extend(plot_seasonality_profiles(df_elec, stage_dir=stage_fig_dir))

    duration = time.perf_counter() - t0
    eda_summary = {
        "electricity_shape": list(df_elec.shape),
        "pems_bay_shape": list(df_pems.shape),
        "pems_adj_shape": list(adj_mx.shape),
        "missing_values_total": int(df_elec.isna().sum().sum()),
        "outlier_summary": eda_res["outlier_summary"],
        "pearson_correlation_with_load": {
            col: round(float(eda_res["corr_df"].loc["load", col]), 4)
            for col in ["load", "temperature", "humidity", "wind_speed"]
        },
        "lag_correlations": eda_res["lag_correlations"],
        "dominant_weather_variable": eda_res["dominant_weather_variable"],
        "seasonality_summary": eda_res["seasonality_summary"],
        "figures": saved_figs,
        "duration_sec": round(duration, 4),
    }
    save_json(eda_summary, metrics_dir / "eda_summary.json")

    update_progress(
        active_run_id,
        "EDA",
        "COMPLETED",
        input_data={"num_rows": len(df_elec), "variables": list(df_elec.columns)},
        output_data=eda_summary,
        duration_sec=duration,
    )
    write_log(
        active_run_id,
        "EDA",
        "SUCCESS",
        f"Hoàn tất EDA: Biến thời tiết ảnh hưởng mạnh nhất = {eda_res['dominant_weather_variable']} ({duration:.2f}s).",
    )
    print(f"[EDA] Biến thời tiết ảnh hưởng mạnh nhất tới Load: {eda_res['dominant_weather_variable']} | Đã xuất {len(saved_figs)} biểu đồ.")
    return eda_summary


if __name__ == "__main__":
    main()