"""
src/lab02/visualize.py
Giữ nguyên 100% code vẽ biểu đồ gốc từ 2 notebook Lab 02 (Dataset 1 & Dataset 2),
chỉ chuẩn hóa thêm tham số save_path và show_plot để xuất PNG chất lượng cao cho báo cáo.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from statsmodels.tsa.stattools import acf


def _finalize_fig(save_path: Optional[Path | str], dpi: int = 120, show_plot: bool = False) -> None:
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=dpi, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close("all")


def plot_original_series_original_style(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    ylabel: str,
    month_interval: int = 3,
    save_path: Optional[Path | str] = None,
    show_plot: bool = False,
) -> None:
    """Giữ nguyên 100% nét vẽ Cell 4 của cả 2 notebook (figsize 20x8, Mean/Min/Max, trục MM/YYYY xoay 90°)."""
    min_val = df[value_col].min()
    max_val = df[value_col].max()
    mean_val = df[value_col].mean()
    min_date = df[value_col].idxmin()
    max_date = df[value_col].idxmax()

    plt.figure(figsize=(20, 8))
    plt.plot(df.index, df[value_col], color="royalblue", label="Original Series", marker=".", alpha=0.8)

    plt.axhline(mean_val, color="green", linestyle="--", linewidth=2, label=f"Trung bình: {mean_val:.1f}")
    plt.axhline(min_val, color="red", linestyle=":", alpha=0.5)
    plt.axhline(max_val, color="darkorange", linestyle=":", alpha=0.5)

    min_fmt = f"{min_val:.2f}" if isinstance(min_val, (float, np.floating)) and not float(min_val).is_integer() else f"{int(min_val)}"
    max_fmt = f"{max_val:.2f}" if isinstance(max_val, (float, np.floating)) and not float(max_val).is_integer() else f"{int(max_val)}"
    plt.scatter(min_date, min_val, color="red", s=100, zorder=5, label=f"Min: {min_fmt} ({min_date.strftime('%m/%Y')})")
    plt.scatter(max_date, max_val, color="darkorange", s=100, zorder=5, label=f"Max: {max_fmt} ({max_date.strftime('%m/%Y')})")

    plt.xlim(df.index.min(), df.index.max())
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=month_interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    plt.xticks(rotation=90, fontsize=8)
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)

    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel("Thời gian (Tháng/Năm)", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(loc="upper left")
    plt.tight_layout()
    _finalize_fig(save_path, dpi=140, show_plot=show_plot)


def plot_train_test_split_original_style(
    train: pd.Series,
    test: pd.Series,
    title: str,
    ylabel: str,
    save_path: Optional[Path | str] = None,
    show_plot: bool = False,
) -> None:
    """Giữ nguyên 100% nét vẽ Cell 7 của cả 2 notebook."""
    plt.figure(figsize=(13, 5))
    plt.plot(train.index, train.values, color="tab:blue", linewidth=2, label=f"Train (80%: {len(train)} tháng)")
    plt.plot(test.index, test.values, color="tab:orange", linewidth=2.5, label=f"Test thực tế (20%: {len(test)} tháng)")
    plt.axvline(test.index[0], color="red", linestyle="--", linewidth=1.5, alpha=0.8, label="Ranh giới phân tách Train/Test")

    plt.title(title, fontweight="bold")
    plt.xlabel("Thời gian (Năm)")
    plt.ylabel(ylabel)
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    _finalize_fig(save_path, dpi=130, show_plot=show_plot)


def plot_differencing_original_style(
    train: pd.Series,
    diff_1: pd.Series,
    diff_1_12: pd.Series,
    value_label: str,
    seasonal_period: int = 12,
    save_path: Optional[Path | str] = None,
    show_plot: bool = False,
) -> None:
    """Giữ nguyên 100% nét vẽ 3 tầng (Chuỗi gốc, Sai phân d=1, Sai phân d=1 & D=1)."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.2), sharex=True)

    axes[0].plot(train.index, train.values, color="blue")
    axes[0].set_title("Chuỗi gốc (Original Train Series)")
    axes[0].set_ylabel(value_label)

    axes[1].plot(diff_1.index, diff_1.values, color="green")
    axes[1].axhline(0, color="red", linestyle="--", alpha=0.5)
    axes[1].set_title("Chuỗi sau khi lấy Sai phân bậc 1 (d=1)")
    axes[1].set_ylabel("Diff (d=1)")

    axes[2].plot(diff_1_12.index, diff_1_12.values, color="purple")
    axes[2].axhline(0, color="red", linestyle="--", alpha=0.5)
    axes[2].set_title(f"Chuỗi sau khi lấy Sai phân bậc 1 và Sai phân mùa vụ (d=1, D=1, s={seasonal_period})")
    axes[2].set_ylabel("Diff (d=1, D=1)")

    plt.tight_layout()
    _finalize_fig(save_path, dpi=130, show_plot=show_plot)


def plot_actual_vs_forecast_battle_log(
    train: pd.Series,
    test: pd.Series,
    fc_arima111: pd.Series,
    sarima_log_pred_mean: pd.Series,
    sarima_log_conf_orig: pd.DataFrame,
    arima_m: Dict[str, float],
    sarima_log_m: Dict[str, float],
    save_path: Optional[Path | str] = None,
    show_plot: bool = False,
) -> None:
    """Giữ nguyên 100% nét vẽ Cell 48 của Dataset 2 ('ACTUAL VS. FORECAST BATTLE')."""
    plt.figure(figsize=(16, 8), dpi=120)

    plt.plot(train.index, train.values, color="#2b5c8f", linewidth=1.9, alpha=0.9,
             label=f'Train (80%: {train.index.min().strftime("%m/%Y")} – {train.index.max().strftime("%m/%Y")})')

    plt.plot(test.index, test.values, color="#e6550d", linewidth=2.8, marker="o", markersize=5.5,
             markerfacecolor="white", markeredgecolor="#e6550d", markeredgewidth=1.8, zorder=5,
             label=f'Test thực tế (20%: {test.index.min().strftime("%m/%Y")} – {test.index.max().strftime("%m/%Y")})')

    plt.plot(test.index, fc_arima111.values, color="#8c564b", linestyle="--", linewidth=2.3, zorder=4,
             label=f'ARIMA(1,1,1) Forecast [RMSE={arima_m["RMSE"]:.1f}, MAPE={arima_m["MAPE (%)"]:.1f}%]')

    plt.plot(test.index, sarima_log_pred_mean.values, color="#1b9e77", linestyle="-", linewidth=2.8,
             marker="s", markersize=4.5, markerfacecolor="#1b9e77", markeredgecolor="white", markeredgewidth=1.2, zorder=6,
             label=f'Log-SARIMA(1,1,0)x(0,1,1)₁₂ [RMSE={sarima_log_m["RMSE"]:.1f}, MAPE={sarima_log_m["MAPE (%)"]:.1f}%]')

    plt.fill_between(test.index, sarima_log_conf_orig.iloc[:, 0], sarima_log_conf_orig.iloc[:, 1],
                     color="#1b9e77", alpha=0.18, zorder=3,
                     label="Log-SARIMA Dải tin cậy 95% (Confidence Interval)")
    plt.plot(test.index, sarima_log_conf_orig.iloc[:, 0], color="#1b9e77", linestyle=":", linewidth=0.9, alpha=0.6)
    plt.plot(test.index, sarima_log_conf_orig.iloc[:, 1], color="#1b9e77", linestyle=":", linewidth=0.9, alpha=0.6)

    split_date = test.index[0]
    plt.axvline(split_date, color="#d62728", linestyle="--", linewidth=1.8, alpha=0.85, zorder=2)
    plt.text(split_date, 670, "Ranh giới Split (80% Train | 20% Test)  ",
             color="#d62728", fontsize=10.5, fontweight="bold", va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff5f5", edgecolor="#d62728", alpha=0.92))

    metrics_text = (
        "BẢNG ĐỐI ĐẦU CHỈ SỐ SAI SỐ (TEST SET)\n"
        "--------------------------------------------------\n"
        f"ARIMA(1,1,1) [Thang đo gốc]:\n"
        f"  • RMSE = {arima_m['RMSE']:>6.2f}    MAE = {arima_m['MAE']:>6.2f}\n"
        f"  • MAPE = {arima_m['MAPE (%)']:>5.2f}%    R²  = {arima_m['R2']:>6.2f} (Không đạt)\n"
        "--------------------------------------------------\n"
        f"SARIMA(1,1,0)x(0,1,1)₁₂ [Biến đổi Log-Exp]:\n"
        f"  • RMSE = {sarima_log_m['RMSE']:>6.2f}    MAE = {sarima_log_m['MAE']:>6.2f}\n"
        f"  • MAPE = {sarima_log_m['MAPE (%)']:>5.2f}%    R²  = {sarima_log_m['R2']:>6.2f} (Đỉnh cao)\n"
        "--------------------------------------------------\n"
        "Đánh giá đột phá: Nhờ triệt tiêu phương sai hình phễu,\n"
        "Log-SARIMA giảm vọt sai số và bám trọn đỉnh sóng!"
    )
    plt.gca().text(0.02, 0.96, metrics_text, transform=plt.gca().transAxes,
                   fontsize=10, va="top", ha="left", linespacing=1.35,
                   bbox=dict(boxstyle="round,pad=0.6", facecolor="#ffffff", edgecolor="#b0bec5", linewidth=1.2, alpha=0.95))

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    plt.xticks(fontsize=10)
    plt.title(
        "ACTUAL VS. FORECAST BATTLE: ĐỐI ĐẦU DỰ BÁO NGOÀI MẪU (ARIMA VS. SARIMA LOG-TRANSFORMED)\n"
        "Dataset: AirPassengers (144 tháng, 1949–1960) | Tập Test: 29 tháng (08/1958 – 12/1960)",
        fontsize=14, fontweight="bold", pad=15, color="#1a237e",
    )
    plt.xlabel("Thời gian (Năm)", fontsize=12, fontweight="bold", labelpad=8)
    plt.ylabel("Số lượng hành khách (nghìn người)", fontsize=12, fontweight="bold", labelpad=8)
    plt.ylim(80, 720)
    plt.grid(True, which="major", linestyle="--", alpha=0.6)
    plt.grid(True, which="minor", linestyle=":", alpha=0.3)
    plt.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.95, edgecolor="#cccccc", fontsize=10.5)
    plt.tight_layout()
    _finalize_fig(save_path, dpi=150, show_plot=show_plot)


def plot_log_diagnostics_2x2(
    test: pd.Series,
    arima_fc: pd.Series,
    sarima_log_pred_mean: pd.Series,
    sarima_log_resid: pd.Series,
    arima_m: Dict[str, float],
    sarima_raw_m: Dict[str, float],
    sarima_log_m: Dict[str, float],
    save_path: Optional[Path | str] = None,
    show_plot: bool = False,
) -> None:
    """Giữ nguyên 100% nét vẽ lưới 2x2 Cell 49 của Dataset 2 ('MODEL DIAGNOSTICS & ERROR DECOMPOSITION')."""
    ci_bound = 1.96 / np.sqrt(len(sarima_log_resid))
    fig, axes = plt.subplots(2, 2, figsize=(17, 12.5), dpi=120)
    fig.suptitle(
        "MODEL DIAGNOSTICS & ERROR DECOMPOSITION: CHẨN ĐOÁN & PHÂN RÃ SAI SỐ\n"
        "Đánh giá bản chất thống kê & kiểm chứng tính ngẫu nhiên (White Noise) của mô hình Log-SARIMA",
        fontsize=15, fontweight="bold", y=0.99, color="#1a237e",
    )

    # Ô 1: Sai số theo thời gian
    ax_tl = axes[0, 0]
    ax_tl.axhline(0, color="black", linestyle="--", linewidth=1.5, alpha=0.8, label="Đường sai số mục tiêu (e = 0)")
    ax_tl.fill_between(test.index, -20, 20, color="#2ca02c", alpha=0.08, label="Vùng sai số mong đợi (±20)")
    ax_tl.plot(test.index, (test - arima_fc).values, color="#8c564b", linestyle="--", linewidth=2.0,
               marker="o", markersize=4.5, label="Sai số ARIMA(1,1,1): Dao động lớn (Bỏ sót mùa vụ)")
    ax_tl.plot(test.index, (test - sarima_log_pred_mean).values, color="#1b9e77", linestyle="-", linewidth=2.3,
               marker="s", markersize=4.5, label="Sai số Log-SARIMA(1,1,0)x(0,1,1)₁₂: Co cụm quanh trục 0")
    ax_tl.set_title("(1) Sai số dự báo theo thời gian trên tập Test (eₜ = yₜ - ŷₜ)", fontsize=12, fontweight="bold", color="#0d47a1")
    ax_tl.set_xlabel("Thời gian (Tháng)", fontsize=10.5, fontweight="bold")
    ax_tl.set_ylabel("Sai số dự báo (Thực tế – Dự báo)", fontsize=10.5, fontweight="bold")
    ax_tl.grid(True, linestyle="--", alpha=0.55)
    ax_tl.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.95, fontsize=9.2)

    # Ô 2: Bar chart 3 thế hệ
    ax_tr = axes[0, 1]
    metric_labels = ["RMSE\n(Hành khách)", "MAE\n(Hành khách)", "MAPE\n(%)"]
    vals_arima = [arima_m["RMSE"], arima_m["MAE"], arima_m["MAPE (%)"]]
    vals_sarima_raw = [sarima_raw_m["RMSE"], sarima_raw_m["MAE"], sarima_raw_m["MAPE (%)"]]
    vals_sarima_log = [sarima_log_m["RMSE"], sarima_log_m["MAE"], sarima_log_m["MAPE (%)"]]
    x = np.arange(len(metric_labels))
    w = 0.26
    ax_tr.bar(x - w, vals_arima, w, label="ARIMA(1,1,1) (Gốc)", color="#8c564b", edgecolor="#4a2810", linewidth=1.1, alpha=0.9)
    ax_tr.bar(x, vals_sarima_raw, w, label="SARIMA(1,1,3)x(0,1,1)₁₂ (Gốc)", color="#ff7f0e", edgecolor="#8c4100", linewidth=1.1, alpha=0.9)
    ax_tr.bar(x + w, vals_sarima_log, w, label="Log-SARIMA(1,1,0)x(0,1,1)₁₂", color="#1b9e77", edgecolor="#0f5a43", linewidth=1.1, alpha=0.9)
    ax_tr.set_title("(2) So sánh sai số định lượng 3 thế hệ mô hình (Sụt giảm ngoạn mục)", fontsize=12, fontweight="bold", color="#0d47a1")
    ax_tr.set_xticks(x)
    ax_tr.set_xticklabels(metric_labels, fontsize=10.5, fontweight="bold")
    ax_tr.set_ylabel("Giá trị sai số", fontsize=10.5, fontweight="bold")
    ax_tr.set_ylim(0, 120)
    ax_tr.grid(True, axis="y", linestyle="--", alpha=0.55)
    ax_tr.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95, fontsize=9.2)

    # Ô 3: ACF phần dư
    ax_bl = axes[1, 0]
    lags_count = 24
    acf_vals = acf(sarima_log_resid, nlags=lags_count, fft=True)
    lags = np.arange(1, lags_count + 1)
    ax_bl.axhspan(-ci_bound, ci_bound, color="#1f77b4", alpha=0.15, label="Dải tin cậy 95% (±1.96/√N)")
    ax_bl.axhline(ci_bound, color="#1f77b4", linestyle="--", linewidth=1.2, alpha=0.85)
    ax_bl.axhline(-ci_bound, color="#1f77b4", linestyle="--", linewidth=1.2, alpha=0.85)
    ax_bl.axhline(0, color="black", linestyle="-", linewidth=0.9, alpha=0.6)
    ax_bl.stem(lags, acf_vals[1:lags_count + 1], linefmt="#1b9e77", markerfmt="o", basefmt=" ")
    ax_bl.set_title("(3) Đồ thị ACF của phần dư Log-SARIMA(1,1,0) (Kiểm chứng Nhiễu trắng)", fontsize=12, fontweight="bold", color="#0d47a1")
    ax_bl.set_xlabel("Độ trễ (Lag)", fontsize=10.5, fontweight="bold")
    ax_bl.set_ylabel("Hệ số tự tương quan (ACF)", fontsize=10.5, fontweight="bold")
    ax_bl.set_xticks(range(1, lags_count + 1, 2))
    ax_bl.set_ylim(-0.4, 0.4)
    ax_bl.grid(True, linestyle="--", alpha=0.55)
    ax_bl.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.95, fontsize=9.5)

    # Ô 4: Histogram + KDE vs Normal
    ax_br = axes[1, 1]
    ax_br.hist(sarima_log_resid, bins=15, density=True, color="#2ca02c", alpha=0.45, edgecolor="white", linewidth=1.2, label="Histogram phần dư")
    kde = stats.gaussian_kde(sarima_log_resid)
    x_eval = np.linspace(sarima_log_resid.min() - 0.03, sarima_log_resid.max() + 0.03, 200)
    ax_br.plot(x_eval, kde(x_eval), color="#1b9e77", linewidth=2.4, label="KDE (Mật độ thực tế)")
    norm_c = stats.norm.pdf(x_eval, loc=sarima_log_resid.mean(), scale=sarima_log_resid.std())
    ax_br.plot(x_eval, norm_c, color="#d62728", linestyle="--", linewidth=2.2, label="Đường chuẩn Gauss N(μ, σ)")
    ax_br.axvline(0, color="black", linestyle=":", linewidth=1.4, alpha=0.7, label="Trục tham chiếu x = 0")
    ax_br.set_title("(4) Phân phối phần dư Log-SARIMA (Histogram & KDE vs. Normal Curve)", fontsize=12, fontweight="bold", color="#0d47a1")
    ax_br.set_xlabel("Giá trị phần dư trên không gian Log", fontsize=10.5, fontweight="bold")
    ax_br.set_ylabel("Mật độ xác suất (Density)", fontsize=10.5, fontweight="bold")
    ax_br.grid(True, linestyle="--", alpha=0.55)
    ax_br.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95, fontsize=9.2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _finalize_fig(save_path, dpi=130, show_plot=show_plot)