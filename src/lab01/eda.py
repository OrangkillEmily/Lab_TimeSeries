r"""
Mô-đun Phân tích Khám phá (Bài 1: Tổng quan + ACF/PACF 40 lags & Bài 2: Kiểm tra White Noise 1000 điểm) cho Lab 01.
"""

from __future__ import annotations

import time
from pathlib import Path

from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, pacf

from src.lab01.data_loader import load_airpassengers_dataset
from src.lab01.tracking import get_active_run_id, record_stage_result, write_log
from src.lab01.utils import ensure_directories, get_path, load_config, save_json, set_global_seed
from src.lab01.visualize import plot_full_series
    
def build_eda_profile(df: pd.DataFrame, value_col: str = "Passengers") -> Dict[str, Any]:
    r"""Tạo hồ sơ mô tả EDA đầy đủ cho tập dữ liệu chuỗi thời gian."""
    ser = df[value_col]
    first_year = str(df.index.min().year)
    last_year = str(df.index.max().year)
    train_n = int(len(df) * 0.8)
    test_n = int(len(df) - train_n)

    return {
        "n_obs": int(len(df)),
        "start_date": str(df.index.min().date()),
        "end_date": str(df.index.max().date()),
        "frequency": "MS (Monthly Start)",
        "dtype": str(ser.dtype),
        "missing_count": int(ser.isna().sum()),
        "duplicate_dates": int(df.index.duplicated().sum()),
        "mean": round(float(ser.mean()), 2),
        "std": round(float(ser.std()), 2),
        "cv_pct": round(float(ser.std() / ser.mean() * 100.0), 2),
        "skewness": round(float(ser.skew()), 3),
        "min": round(float(ser.min()), 2),
        "min_date": ser.idxmin().strftime("%m/%Y"),
        "q25": round(float(ser.quantile(0.25)), 1),
        "median": round(float(ser.median()), 1),
        "q75": round(float(ser.quantile(0.75)), 1),
        "max": round(float(ser.max()), 2),
        "max_date": ser.idxmax().strftime("%m/%Y"),
        "growth_ratio": round(float(ser.iloc[-1] / ser.iloc[0]), 2),
        "first_year": first_year,
        "last_year": last_year,
        "first_year_amp": round(float(ser.loc[first_year].max() - ser.loc[first_year].min()), 2),
        "last_year_amp": round(float(ser.loc[last_year].max() - ser.loc[last_year].min()), 2),
        "recommended_split": {
            "ratio": "80% Train / 20% Test (shuffle=False)",
            "train_n": train_n,
            "train_range": f"{df.index[0].strftime('%m/%Y')} → {df.index[train_n - 1].strftime('%m/%Y')}",
            "test_n": test_n,
            "test_range": f"{df.index[train_n].strftime('%m/%Y')} → {df.index[-1].strftime('%m/%Y')}",
            "note": "Lab 01 tập trung phân tích cấu trúc, phân rã và làm sạch; bước chia Train/Test tuần tự được áp dụng ở Lab 02 để chống Data Leakage.",
        },
    }

def analyze_acf_pacf(
    series: pd.Series,
    nlags: int = 40,
    fig_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    r"""
    Purpose: Tính và trực quan hóa chuỗi tổng quan + hàm tự tương quan ACF/PACF đến nlags=40 (Bài tập 1).
    Input-Output:
        - Input: series (pd.Series), nlags (int, mặc định 40), fig_dir (Optional[Path]).
        - Output: Từ điển chứa số lag có ý nghĩa, ACF tại lag 1/12/24/36 và top 5 lag PACF quan trọng nhất.
    How: Vẽ 01_full_series_overview.png qua plot_full_series(), tính acf/pacf, vẽ 02_acf_pacf_airpassengers.png với zero=False.
    """
    n = len(series)
    effective_lags = min(nlags, n // 2 - 1)
    conf_bound = float(1.96 / np.sqrt(n))

    acf_vals = acf(series, nlags=effective_lags, fft=True)
    pacf_vals = pacf(series, nlags=effective_lags, method="ywm")

    sig_acf = [int(i) for i in range(1, effective_lags + 1) if abs(acf_vals[i]) > conf_bound]
    sig_pacf = [int(i) for i in range(1, effective_lags + 1) if abs(pacf_vals[i]) > conf_bound]
    top_pacf = sorted(
        [(int(i), round(float(pacf_vals[i]), 4)) for i in range(1, effective_lags + 1)],
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:5]

    if fig_dir is not None:
        plot_full_series(series, save_path=fig_dir / "01_full_series_overview.png")

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        plot_acf(series, lags=effective_lags, alpha=0.05, zero=False, ax=axes[0], color="#ff7f0e")
        axes[0].set_title(f"Autocorrelation Function (ACF) — Lags {effective_lags}", fontweight="bold")
        axes[0].set_xlabel("Lag (Tháng)")
        axes[0].set_ylabel("Hệ số tự tương quan")
        axes[0].grid(True, alpha=0.25)

        plot_pacf(series, lags=effective_lags, alpha=0.05, method="ywm", zero=False, ax=axes[1], color="#2ca02c")
        axes[1].set_title(f"Partial Autocorrelation (PACF) — Lags {effective_lags}", fontweight="bold")
        axes[1].set_xlabel("Lag (Tháng)")
        axes[1].set_ylabel("Hệ số tự tương quan riêng phần")
        axes[1].grid(True, alpha=0.25)

        plt.tight_layout()
        fig.savefig(fig_dir / "02_acf_pacf_airpassengers.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    return {
        "nlags": int(effective_lags),
        "conf_bound_95": round(conf_bound, 4),
        "acf_lag1": round(float(acf_vals[1]), 4),
        "acf_lag12": round(float(acf_vals[12]), 4) if effective_lags >= 12 else None,
        "acf_lag24": round(float(acf_vals[24]), 4) if effective_lags >= 24 else None,
        "acf_lag36": round(float(acf_vals[36]), 4) if effective_lags >= 36 else None,
        "significant_acf_count": len(sig_acf),
        "significant_pacf_count": len(sig_pacf),
        "significant_pacf_lags": sig_pacf[:10],
        "top_pacf_lags": [{"lag": lag, "pacf": val} for lag, val in top_pacf],
        "has_strong_autocorrelation": bool(abs(acf_vals[1]) > 0.7),
        "has_seasonal_peaks_12": bool(effective_lags >= 24 and acf_vals[12] > acf_vals[11] and acf_vals[24] > acf_vals[23]),
    }


def analyze_white_noise(
    series: pd.Series,
    n_points: int = 1000,
    seed: int = 42,
    nlags: int = 40,
    fig_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    r"""
    Purpose: Sinh 1000 điểm White Noise, vẽ biểu đồ chuỗi và đối chiếu trực tiếp ACF với AirPassengers (Bài tập 2).
    Input-Output:
        - Input: series (pd.Series), n_points (int), seed (int), nlags (int), fig_dir (Optional[Path]).
        - Output: Từ điển chứa thống kê White Noise, tỷ lệ lag trong dải 95%, p-value Ljung-Box và 5 dòng mô tả khác biệt.
    How: Sinh N(0,1) bằng default_rng(seed), chạy acorr_ljungbox và lưu 2 ảnh: 03_white_noise_series.png & 04_white_noise_vs_airpassengers_acf.png.
    """
    rng = np.random.default_rng(seed)
    wn = pd.Series(rng.normal(loc=0.0, scale=1.0, size=n_points), name="WhiteNoise")

    wn_conf = float(1.96 / np.sqrt(n_points))
    wn_acf = acf(wn, nlags=nlags, fft=True)
    outside_lags = [int(i) for i in range(1, nlags + 1) if abs(wn_acf[i]) > wn_conf]
    inside_pct = round(100.0 * (nlags - len(outside_lags)) / nlags, 2)

    lb_wn = acorr_ljungbox(wn, lags=[10, 20], return_df=True)
    lb_ap = acorr_ljungbox(series, lags=[10, 20], return_df=True)
    wn_p20 = float(lb_wn.loc[20, "lb_pvalue"])
    ap_p20 = float(lb_ap.loc[20, "lb_pvalue"])

    if fig_dir is not None:
        fig, ax = plt.subplots(figsize=(12, 4.2))
        ax.plot(wn.values, linewidth=0.8, color="#1f77b4")
        ax.axhline(0, linestyle="--", color="red", linewidth=1.0)
        ax.set_title(f"Chuỗi White Noise mô phỏng ({n_points} quan sát)", fontweight="bold")
        ax.set_xlabel("Thời gian (t)")
        ax.set_ylabel("Giá trị")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(fig_dir / "03_white_noise_series.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
        plot_acf(wn, lags=nlags, alpha=0.05, zero=False, ax=axes[0], color="#1f77b4")
        axes[0].set_title(f"ACF — White Noise ({n_points} điểm)\n({inside_pct:.1f}% lags nằm trong dải 95%)", fontweight="bold")
        axes[0].set_xlabel("Lag")
        axes[0].set_ylabel("Hệ số tự tương quan (ACF)")
        axes[0].grid(True, linestyle="--", alpha=0.4)

        plot_acf(series, lags=nlags, alpha=0.05, zero=False, ax=axes[1], color="#d62728")
        axes[1].set_title("ACF — AirPassengers (144 điểm)\n(Có xu hướng và tính mùa vụ mạnh)", fontweight="bold")
        axes[1].set_xlabel("Lag")
        axes[1].set_ylabel("Hệ số tự tương quan (ACF)")
        axes[1].grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        fig.savefig(fig_dir / "04_white_noise_vs_airpassengers_acf.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    comparison_5_lines: List[str] = [
        f"1. Tính ngẫu nhiên & cấu trúc: White Noise dao động ngẫu nhiên quanh trung bình 0 (mean={wn.mean():.3f}, std={wn.std():.3f}), còn AirPassengers có xu hướng tăng trưởng dài hạn và mùa vụ rõ rệt.",
        f"2. Ngưỡng tin cậy 95%: White Noise có {inside_pct:.1f}% số lag nằm trong dải tin cậy ±{wn_conf:.4f} ({len(outside_lags)}/{nlags} lag vượt nhẹ do ngẫu nhiên), trong khi AirPassengers vượt xa ngưỡng ở hầu hết các lag đầu.",
        f"3. Tốc độ suy giảm ACF: ACF của White Noise xấp xỉ 0 ngay từ lag 1 (ACF(1)={wn_acf[1]:.4f}), còn AirPassengers suy giảm rất chậm (ACF(1)={acf(series, nlags=1)[1]:.4f}) chứng tỏ chuỗi không dừng.",
        f"4. Kiểm định Ljung-Box (lag 20): White Noise đạt p-value={wn_p20:.4f} (> 0.05, không bác bỏ H0 — không có tự tương quan), còn AirPassengers có p-value={ap_p20:.2e} (<< 0.05, tự tương quan cực mạnh).",
        "5. Khả năng dự báo: White Noise không chứa quy luật quá khứ để dự báo, ngược lại AirPassengers có cấu trúc tự tương quan và chu kỳ 12 tháng rất giàu thông tin cho mô hình SARIMA/ETS.",
    ]

    return {
        "n_points": int(n_points),
        "mean": round(float(wn.mean()), 4),
        "std": round(float(wn.std()), 4),
        "conf_bound_95": round(wn_conf, 4),
        "outside_lags": outside_lags,
        "inside_95_pct": inside_pct,
        "ljung_box_wn_p10": round(float(lb_wn.loc[10, "lb_pvalue"]), 4),
        "ljung_box_wn_p20": round(wn_p20, 4),
        "ljung_box_ap_p20": float(ap_p20),
        "is_white_noise_confirmed": bool(wn_p20 > 0.05),
        "comparison_5_lines": comparison_5_lines,
    }


def main(run_id: Optional[str] = None) -> Dict[str, Any]:
    r"""
    Purpose: Thực thi Giai đoạn 2 (Bài tập 1: ACF/PACF & Bài tập 2: White Noise) cho Lab 01.
    Input-Output:
        - Input: run_id (Optional[str]).
        - Output: Từ điển tổng hợp kết quả ACF/PACF, White Noise và danh sách 4 ảnh đã sinh.
    How: Nạp AirPassengers, gọi analyze_acf_pacf() và analyze_white_noise(), lưu JSON vào metrics_dir và ghi log.
    """
    t0 = time.perf_counter()
    cfg = load_config()
    ensure_directories(cfg)
    set_global_seed(int(cfg["project"]["random_seed"]))
    active_run_id = get_active_run_id(run_id)

    write_log(active_run_id, "EDA_ACF_WHITENOISE", "START", "Bắt đầu Bài 1 (Tổng quan + ACF/PACF 40 lags) & Bài 2 (White Noise 1000 điểm).")
    df, _ = load_airpassengers_dataset(cfg)
    series = df[cfg["data"]["value_col"]].astype(float)

    fig_dir = get_path("figures_dir", cfg)
    acf_res = analyze_acf_pacf(series, nlags=int(cfg["analysis"]["acf_pacf_nlags"]), fig_dir=fig_dir)
    wn_res = analyze_white_noise(
        series,
        n_points=int(cfg["analysis"]["white_noise_n"]),
        seed=int(cfg["project"]["random_seed"]),
        nlags=int(cfg["analysis"]["acf_pacf_nlags"]),
        fig_dir=fig_dir,
    )

    payload: Dict[str, Any] = {
        "eda_profile": build_eda_profile(df, value_col="Passengers"),
        "acf_pacf": acf_res,
        "white_noise": wn_res,
        "generated_figures": [
            "01_full_series_overview.png",
            "02_acf_pacf_airpassengers.png",
            "03_white_noise_series.png",
            "04_white_noise_vs_airpassengers_acf.png",
        ],
        "duration_sec": round(time.perf_counter() - t0, 4),
    }

    save_json(payload, get_path("metrics_dir", cfg) / "eda_acf_whitenoise_summary.json")
    record_stage_result(active_run_id, "eda", payload)
    write_log(
        active_run_id,
        "EDA_ACF_WHITENOISE",
        "SUCCESS",
        f"Hoàn tất 4 hình EDA | ACF1={acf_res['acf_lag1']}, ACF12={acf_res['acf_lag12']}, LB_WN_p20={wn_res['ljung_box_wn_p20']}.",
    )
    return payload