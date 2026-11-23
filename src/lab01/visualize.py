r"""
Mô-đun Trực quan hóa Chuẩn (Visualization Suite) cho Bài thực hành Lab 01.
Bảo toàn 100% phong cách và cấu trúc biểu đồ gốc từ src/lab01_utils/plotting.py và decomposition.py,
bổ sung chế độ lưu PNG tự động (save_path) phục vụ pipeline và báo cáo HTML/PDF.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf

from src.lab01.data_loader import load_airpassengers_dataset
from src.lab01.tracking import get_active_run_id, record_stage_result, write_log
from src.lab01.utils import ensure_directories, get_path, load_config, save_json

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


def _finalize_figure(fig: plt.Figure, save_path: Optional[Path] = None, show: bool = False) -> None:
    r"""
    Purpose: Lưu đối tượng Figure ra tệp PNG (nếu có save_path) và hiển thị hoặc giải phóng bộ nhớ.
    Input-Output:
        - Input: fig (plt.Figure), save_path (Optional[Path]), show (bool).
        - Output: Không trả về giá trị.
    How: Tạo thư mục cha, gọi fig.savefig(..., dpi=150, bbox_inches='tight'), sau đó plt.show() hoặc plt.close(fig).
    """
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_full_series(ts: pd.Series, save_path: Optional[Path] = None, show: bool = False) -> None:
    r"""
    Purpose: Vẽ line chart toàn bộ chuỗi AirPassengers theo từng tháng (MM/YYYY), đánh dấu Min/Max và đường Trung bình.
    Input-Output:
        - Input: ts (pd.Series), save_path (Optional[Path]), show (bool).
        - Output: Không trả về giá trị.
    How: Giữ nguyên cấu trúc gốc (figsize=(20, 8), royalblue, MonthLocator, DateFormatter '%m/%Y', rotation=90).
    """
    min_val = ts.min()
    max_val = ts.max()
    mean_val = ts.mean()
    min_date = ts.idxmin()
    max_date = ts.idxmax()

    fig, ax = plt.subplots(figsize=(20, 8))
    ax.plot(ts.index, ts.values, color="royalblue", label="Original Series", marker=".", alpha=0.8)
    ax.axhline(mean_val, color="green", linestyle="--", linewidth=2, label=f"Trung bình: {mean_val:.1f}")
    ax.axhline(min_val, color="red", linestyle=":", alpha=0.5)
    ax.axhline(max_val, color="darkorange", linestyle=":", alpha=0.5)
    ax.scatter(min_date, min_val, color="red", s=100, zorder=5, label=f"Min: {int(min_val)} ({min_date.strftime('%m/%Y')})")
    ax.scatter(max_date, max_val, color="darkorange", s=100, zorder=5, label=f"Max: {int(max_val)} ({max_date.strftime('%m/%Y')})")
    ax.set_xlim(ts.index.min(), ts.index.max())

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=8)
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)

    ax.set_title("Air Passengers Over Time (01/1949 - 12/1960)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Thời gian (Tháng/Năm)", fontsize=12)
    ax.set_ylabel("Số lượng hành khách (Nghìn người)", fontsize=12)
    ax.legend(loc="upper left")
    plt.tight_layout()
    _finalize_figure(fig, save_path, show)


def plot_decomposition(
    result: Any,
    title: str,
    is_multiplicative: bool = False,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    r"""
    Purpose: Vẽ 4 thành phần của phân rã chuỗi thời gian (Classical hoặc STL): Observed, Trend, Seasonal, Residual.
    Input-Output:
        - Input: result (DecomposeResult/STL), title (str), is_multiplicative (bool), save_path (Optional[Path]), show (bool).
        - Output: Không trả về giá trị.
    How: Giữ nguyên bố cục 4x1 (figsize=(12, 9)), scatter quanh y=1 cho mô hình Nhân và line quanh y=0 cho mô hình Cộng/STL.
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(result.observed, color="#1f77b4", linewidth=1.2)
    axes[0].set_ylabel("Observed", fontsize=10, fontweight="bold")
    axes[0].set_title(title, fontsize=12, fontweight="bold")

    axes[1].plot(result.trend, color="#ff7f0e", linewidth=1.5)
    axes[1].set_ylabel("Trend", fontsize=10, fontweight="bold")

    axes[2].plot(result.seasonal, color="#2ca02c", linewidth=1.2)
    axes[2].set_ylabel("Seasonal", fontsize=10, fontweight="bold")

    if is_multiplicative:
        axes[3].scatter(result.resid.index, result.resid, color="#d62728", s=15, alpha=0.7)
        axes[3].axhline(1, color="black", linestyle="--", linewidth=1)
    else:
        axes[3].plot(result.resid, color="#d62728", linewidth=1.0)
        axes[3].axhline(0, color="black", linestyle="--", linewidth=1)

    axes[3].set_ylabel("Residual", fontsize=10, fontweight="bold")
    axes[3].set_xlabel("Thời gian", fontsize=11)

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    _finalize_figure(fig, save_path, show)


def analyze_residuals(
    residual: pd.Series,
    lags: int = 30,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> Dict[str, Any]:
    r"""
    Purpose: Vẽ Histogram kèm đường chuẩn N(mu, std^2), ACF của phần dư (30 lags) và thực hiện kiểm định Ljung-Box.
    Input-Output:
        - Input: residual (pd.Series), lags (int, mặc định 30), save_path (Optional[Path]), show (bool).
        - Output: Từ điển chứa mean, std, ci_limit, số lag trong ngưỡng 95% và p-value Ljung-Box (lag 10, 20).
    How: Giữ nguyên đồ thị gốc 1x2 (figsize=(14, 4.5)) từ src/lab01_utils/decomposition.py và trả về thống kê định lượng.
    """
    residual = residual.dropna()
    mu, std = float(residual.mean()), float(residual.std())

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    axes[0].hist(residual, bins=20, density=True, alpha=0.6, color="#4c72b0", edgecolor="black")
    x_norm = np.linspace(mu - 3 * std, mu + 3 * std, 100)
    p_norm = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_norm - mu) / std) ** 2)
    axes[0].plot(x_norm, p_norm, "r--", linewidth=1.5, label=f"N({mu:.2f}, {std:.2f}^2)")
    axes[0].set_title("Phân phối của Residual", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Giá trị Residual")
    axes[0].set_ylabel("Mật độ")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.3)

    plot_acf(residual, lags=lags, alpha=0.05, zero=False, ax=axes[1])
    axes[1].set_title(f"ACF của Residual ({lags} lags)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Lag")
    axes[1].set_ylabel("Tự tương quan")
    axes[1].grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    _finalize_figure(fig, save_path, show)

    ci_limit = float(1.96 / np.sqrt(len(residual)))
    acf_vals = acf(residual, nlags=lags)[1:]
    num_inside = int(np.sum(np.abs(acf_vals) <= ci_limit))
    lb_test = acorr_ljungbox(residual, lags=[10, 20], return_df=True)

    return {
        "mean": round(mu, 4),
        "std": round(std, 4),
        "ci_limit_95": round(ci_limit, 4),
        "lags_inside_ci": num_inside,
        "total_lags": int(lags),
        "inside_ci_pct": round(num_inside / lags * 100.0, 2),
        "ljung_box_p10": round(float(lb_test.loc[10, "lb_pvalue"]), 4),
        "ljung_box_p20": round(float(lb_test.loc[20, "lb_pvalue"]), 4),
    }


def evaluate_and_plot_imputation(
    ts_dirty: pd.Series,
    ts_true: pd.Series,
    missing_dates: pd.DatetimeIndex,
    zoom_range: Tuple[str, str] = ("1955-03-01", "1955-12-01"),
    save_path: Optional[Path] = None,
    show: bool = False,
) -> Tuple[pd.Series, pd.Series, Dict[str, Any]]:
    r"""
    Purpose: Điền khuyết bằng Forward-fill và Time Interpolation, đánh giá MAE/RMSE/MAPE và vẽ 2 biểu đồ toàn chuỗi + phóng to.
    Input-Output:
        - Input: ts_dirty, ts_true, missing_dates, zoom_range (Tuple[str, str]), save_path, show.
        - Output: Tuple gồm (ts_ffill, ts_interp, metrics_dict).
    How: Giữ nguyên đồ thị gốc 1x2 (figsize=(15, 4.8)) với axes[1].step(..., where='post') cho Forward-fill và format %m/%Y.
    """
    ts_ffill = ts_dirty.ffill()
    ts_interp = ts_dirty.interpolate(method="time")

    def _metrics(est: pd.Series) -> Dict[str, float]:
        e = est[missing_dates] - ts_true[missing_dates]
        return {
            "MAE": round(float(e.abs().mean()), 4),
            "RMSE": round(float(np.sqrt((e ** 2).mean())), 4),
            "MAPE": round(float((e.abs() / ts_true[missing_dates]).mean() * 100.0), 4),
        }

    ff_m = _metrics(ts_ffill)
    ip_m = _metrics(ts_interp)

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))

    axes[0].plot(ts_dirty.index, ts_dirty.values, color="#1f77b4", linewidth=1.3, label="Dữ liệu có NaN")
    axes[0].scatter(missing_dates, ts_ffill[missing_dates], color="#ff7f0e", marker="s", s=28, zorder=5, label="Forward-fill")
    axes[0].scatter(missing_dates, ts_interp[missing_dates], color="#2ca02c", marker="^", s=28, zorder=5, label="Interpolation")
    axes[0].set_title("Toàn chuỗi: các điểm được điền", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Thời gian (Năm)")
    axes[0].set_ylabel("Số hành khách (nghìn người)")
    axes[0].legend(loc="upper left")

    zoom = slice(zoom_range[0], zoom_range[1])
    axes[1].plot(ts_true[zoom].index, ts_true[zoom].values, color="gray", linestyle="--", marker="o", markersize=4, label="Giá trị thật")
    axes[1].step(ts_ffill[zoom].index, ts_ffill[zoom].values, where="post", color="#ff7f0e", linewidth=1.8, label="Forward-fill")
    axes[1].plot(ts_interp[zoom].index, ts_interp[zoom].values, color="#2ca02c", linewidth=1.8, marker="^", markersize=5, label="Interpolation")

    title_start = pd.to_datetime(zoom_range[0]).strftime("%m/%Y")
    title_end = pd.to_datetime(zoom_range[1]).strftime("%m/%Y")
    axes[1].set_title(f"Phóng to khoảng thiếu ({title_start} – {title_end})", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Thời gian")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    axes[1].legend(loc="upper left")

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    _finalize_figure(fig, save_path, show)

    return ts_ffill, ts_interp, {"forward_fill": ff_m, "time_interpolation": ip_m}


def detect_and_plot_rolling_outliers(
    ts_filled: pd.Series,
    outlier_dates: pd.DatetimeIndex,
    window: int = 12,
    k: float = 3.0,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> pd.Series:
    r"""
    Purpose: Phát hiện outlier bằng Rolling mean ± k*Rolling std (dùng shift(1) chống rò rỉ cửa sổ) và vẽ biểu đồ Rolling variance.
    Input-Output:
        - Input: ts_filled (pd.Series), outlier_dates (pd.DatetimeIndex), window (int), k (float), save_path, show.
        - Output: Boolean Series đánh dấu các điểm có |roll_z| > k.
    How: Giữ nguyên logic shift(1) và bố cục 2x1 (figsize=(12, 8)) gồm dải ±3 rolling std và Rolling variance màu đỏ.
    """
    roll_mean = ts_filled.rolling(window).mean().shift(1)
    roll_std = ts_filled.rolling(window).std().shift(1)
    roll_var = roll_std ** 2

    roll_z = (ts_filled - roll_mean) / roll_std
    out_roll = (roll_z.abs() > k).fillna(False)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(ts_filled.index, ts_filled.values, color="#1f77b4", linewidth=1.2, label="Chuỗi (đã điền missing)")
    axes[0].plot(roll_mean.index, roll_mean.values, color="#ff7f0e", linewidth=2, label=f"Rolling mean (window={window})")
    axes[0].fill_between(
        roll_mean.index,
        roll_mean - k * roll_std,
        roll_mean + k * roll_std,
        color="#ff7f0e",
        alpha=0.15,
        label=f"Dải ±{int(k) if float(k).is_integer() else k} rolling std",
    )
    axes[0].scatter(ts_filled.index[out_roll], ts_filled[out_roll], color="#d62728", s=60, zorder=5, label="Bị đánh dấu outlier")
    axes[0].scatter(outlier_dates, ts_filled[outlier_dates], facecolors="none", edgecolors="black", s=140, zorder=6, label="Outlier thật (đã chèn)")
    axes[0].set_title(f"Phát hiện outlier bằng Rolling mean ± {int(k) if float(k).is_integer() else k} Rolling std", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Số hành khách (nghìn người)")
    axes[0].legend(loc="upper left", fontsize=8)

    axes[1].plot(roll_var.index, roll_var.values, color="#d62728", linewidth=1.8, label="Rolling variance")
    axes[1].fill_between(roll_var.index, roll_var.values, alpha=0.15, color="#d62728")
    axes[1].set_title("Rolling variance - phương sai tăng dần theo thời gian", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Thời gian (Năm)")
    axes[1].set_ylabel("Phương sai")
    axes[1].legend(loc="upper left")
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    _finalize_figure(fig, save_path, show)
    return out_roll


def detect_and_plot_stl_outliers(
    ts_filled: pd.Series,
    outlier_dates: pd.DatetimeIndex,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> Tuple[pd.Series, pd.Series]:
    r"""
    Purpose: So sánh phát hiện outlier bằng Z-score / IQR trên chuỗi gốc (A) và trên phần dư STL thang log (B).
    Input-Output:
        - Input: ts_filled (pd.Series), outlier_dates (pd.DatetimeIndex), save_path, show.
        - Output: Tuple gồm (out_z_res Boolean Series, log_resid pd.Series).
    How: Giữ nguyên STL(np.log(ts_filled), period=12, robust=False) và bố cục 2x1 (figsize=(12, 8)) từ bản gốc.
    """
    q1, q3 = ts_filled.quantile([0.25, 0.75])
    iqr = q3 - q1

    log_resid = STL(np.log(ts_filled), period=12, robust=False).fit().resid
    z_res = (log_resid - log_resid.mean()) / log_resid.std()
    out_z_res = (z_res.abs() > 3).fillna(False)

    q1r, q3r = log_resid.quantile([0.25, 0.75])
    iqr_r = q3r - q1r

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(ts_filled.index, ts_filled.values, color="#1f77b4", linewidth=1.2)
    axes[0].axhline(q3 + 1.5 * iqr, color="#9467bd", linestyle="--", label="Hàng rào IQR trên (toàn cục)")
    axes[0].axhline(ts_filled.mean() + 3 * ts_filled.std(), color="#8c564b", linestyle=":", label="Mean + 3 std (toàn cục)")
    axes[0].scatter(outlier_dates, ts_filled[outlier_dates], facecolors="none", edgecolors="black", s=140, zorder=6, label="Outlier thật (đã chèn)")
    axes[0].set_title("(A) Z-score / IQR trực tiếp trên chuỗi gốc: ngưỡng toàn cục không bắt được outlier", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Số hành khách")
    axes[0].legend(loc="upper left", fontsize=8)

    axes[1].plot(log_resid.index, log_resid.values, color="#2ca02c", linewidth=1.2, label="Phần dư STL (thang log)")
    axes[1].axhline(log_resid.mean() + 3 * log_resid.std(), color="#d62728", linestyle="--", label="±3 std (Z-score)")
    axes[1].axhline(log_resid.mean() - 3 * log_resid.std(), color="#d62728", linestyle="--")
    axes[1].axhline(q3r + 1.5 * iqr_r, color="#9467bd", linestyle=":", label="Hàng rào IQR 1.5x")
    axes[1].axhline(q1r - 1.5 * iqr_r, color="#9467bd", linestyle=":")
    axes[1].scatter(log_resid.index[out_z_res], log_resid[out_z_res], color="#d62728", s=60, zorder=5, label="Được đánh dấu (Z-score)")
    axes[1].scatter(outlier_dates, log_resid[outlier_dates], facecolors="none", edgecolors="black", s=140, zorder=6, label="Outlier thật (đã chèn)")
    axes[1].set_title("(B) Z-score / IQR trên phần dư STL (Outlier tách biệt rất rõ ràng)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Thời gian (Năm)")
    axes[1].set_ylabel("Phần dư (log)")
    axes[1].legend(loc="lower left", fontsize=8, ncol=2)
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    _finalize_figure(fig, save_path, show)
    return out_z_res, log_resid


def plot_before_after_cleaning(
    ts_dirty: pd.Series,
    ts_ffill: pd.Series,
    ts_interp: pd.Series,
    ts_true: pd.Series,
    ts_clean: pd.Series,
    missing_dates: pd.DatetimeIndex,
    outlier_dates: pd.DatetimeIndex,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    r"""
    Purpose: Vẽ 3 biểu đồ đối chiếu các giai đoạn: Trước xử lý -> Sau điền missing -> Sau làm sạch toàn bộ (Missing + Outlier).
    Input-Output:
        - Input: ts_dirty, ts_ffill, ts_interp, ts_true, ts_clean, missing_dates, outlier_dates, save_path, show.
        - Output: Không trả về giá trị.
    How: Giữ nguyên bố cục 3x1 (figsize=(13, 12)) gồm các dải cam axvspan cho 11 NaN, tam giác xanh và hình thoi tím.
    """
    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True, sharey=True)

    axes[0].plot(ts_dirty.index, ts_dirty.values, color="#1f77b4", linewidth=1.3, marker="o", markersize=3, label="Dữ liệu bẩn")
    for d in missing_dates:
        axes[0].axvspan(d - pd.Timedelta(days=15), d + pd.Timedelta(days=15), color="#ff7f0e", alpha=0.35, linewidth=0)
    axes[0].scatter(outlier_dates, ts_dirty[outlier_dates], color="#d62728", s=70, zorder=5, label="Outlier")
    axes[0].plot([], [], color="#ff7f0e", alpha=0.5, linewidth=8, label=f"Vị trí NaN ({len(missing_dates)} điểm)")
    axes[0].set_title(f"TRƯỚC khi xử lý: có NaN (đứt đoạn) và {len(outlier_dates)} outlier", fontsize=12, fontweight="bold")
    axes[0].legend(loc="upper left")

    axes[1].plot(ts_ffill.index, ts_ffill.values, color="#ff7f0e", linewidth=1.3, label="Forward-fill")
    axes[1].plot(ts_interp.index, ts_interp.values, color="#2ca02c", linewidth=1.3, label="Interpolation")
    axes[1].scatter(outlier_dates, ts_interp[outlier_dates], color="#d62728", s=70, zorder=5, label="Outlier vẫn còn")
    axes[1].set_title("SAU khi xử lý missing: Forward-fill vs Interpolation (outlier chưa xử lý)", fontsize=12, fontweight="bold")
    axes[1].legend(loc="upper left")

    axes[2].plot(ts_true.index, ts_true.values, color="gray", linestyle="--", linewidth=1.2, label="Dữ liệu thật (đáp án)")
    axes[2].plot(ts_clean.index, ts_clean.values, color="#1f77b4", linewidth=1.5, label="Dữ liệu sạch (ts_clean)")
    axes[2].scatter(missing_dates, ts_clean[missing_dates], color="#2ca02c", marker="^", s=40, zorder=5, label="Điểm đã nội suy (missing)")
    axes[2].scatter(outlier_dates, ts_clean[outlier_dates], color="#9467bd", marker="D", s=55, zorder=5, label="Điểm đã thay thế (outlier)")
    axes[2].set_title("SAU khi xử lý đầy đủ: điền missing + thay outlier", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Thời gian (Năm)")
    axes[2].legend(loc="upper left")
    axes[2].xaxis.set_major_locator(mdates.YearLocator())
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    for ax in axes:
        ax.set_ylabel("Số hành khách (nghìn)")
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    _finalize_figure(fig, save_path, show)


def plot_residuals_comparison(
    ts_clean: pd.Series,
    log_resid: pd.Series,
    out_z_res: pd.Series,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> Dict[str, Any]:
    r"""
    Purpose: Vẽ đối chiếu phần dư STL (thang log) trước và sau khi làm sạch kèm Boxplot Trước/Sau.
    Input-Output:
        - Input: ts_clean (pd.Series), log_resid (pd.Series), out_z_res (Boolean Series), save_path, show.
        - Output: Từ điển chứa độ lệch chuẩn và số điểm |z|>3 trước và sau làm sạch.
    How: Giữ nguyên bố cục 1x2 (figsize=(14, 4.5)) gồm line chart chồng nhau và boxplot(showmeans=True).
    """
    resid_before = log_resid
    resid_after = STL(np.log(ts_clean), period=12, robust=False).fit().resid

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    axes[0].plot(resid_before.index, resid_before.values, color="#d62728", linewidth=1.2, label="Trước (còn outlier)")
    axes[0].plot(resid_after.index, resid_after.values, color="#2ca02c", linewidth=1.2, label="Sau (ts_clean)")
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Phần dư STL (thang log) trước và sau xử lý", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Thời gian (Năm)")
    axes[0].set_ylabel("Phần dư (log)")
    axes[0].legend()

    try:
        axes[1].boxplot([resid_before.values, resid_after.values], tick_labels=["Trước", "Sau"], showmeans=True)
    except TypeError:
        axes[1].boxplot([resid_before.values, resid_after.values], labels=["Trước", "Sau"], showmeans=True)
    axes[1].set_title("Boxplot phần dư STL", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Phần dư (log)")

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    _finalize_figure(fig, save_path, show)

    z_after = (resid_after - resid_after.mean()) / resid_after.std()
    outliers_after = int((z_after.abs() > 3).sum())
    return {
        "resid_std_before": round(float(resid_before.std()), 4),
        "resid_std_after": round(float(resid_after.std()), 4),
        "outliers_z3_before": int(out_z_res.fillna(False).sum()),
        "outliers_z3_after": outliers_after,
    }


def plot_rolling_statistics(
    ts: pd.Series,
    roll_mean: pd.Series,
    roll_std: pd.Series,
    window: int = 12,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    r"""
    Purpose: Vẽ Chuỗi gốc + Rolling Mean và Rolling Standard Deviation (window=12).
    Input-Output:
        - Input: ts, roll_mean, roll_std, window (int), save_path, show.
        - Output: Không trả về giá trị.
    How: Giữ nguyên bố cục 2x1 (figsize=(12, 8)) màu xanh nhạt/cam đậm ở trên và đỏ tô vùng hồng ở dưới.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(ts.index, ts.values, color="#1f77b4", linewidth=1.0, alpha=0.6, label="Chuỗi gốc")
    axes[0].plot(roll_mean.index, roll_mean.values, color="#ff7f0e", linewidth=2.2, label=f"Rolling mean (window={window})")
    axes[0].set_title("Chuỗi gốc và Rolling Mean", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Số hành khách (nghìn người)")
    axes[0].legend(loc="upper left")

    axes[1].plot(roll_std.index, roll_std.values, color="#d62728", linewidth=2.0, label=f"Rolling std (window={window})")
    axes[1].fill_between(roll_std.index, roll_std.values, alpha=0.15, color="#d62728")
    axes[1].set_title("Rolling Standard Deviation", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Thời gian (Năm)")
    axes[1].set_ylabel("Độ lệch chuẩn (nghìn người)")
    axes[1].legend(loc="upper left")
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    _finalize_figure(fig, save_path, show)


def plot_seasonal_trends(
    table: pd.DataFrame,
    seasonal_ratio: pd.DataFrame,
    si_mean: pd.Series,
    years: List[int],
    month_labels: List[str],
    save_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    r"""
    Purpose: Vẽ Seasonal plot (mỗi năm 1 đường với colorbar viridis) và Chỉ số mùa vụ trung bình kèm khoảng min–max.
    Input-Output:
        - Input: table (Month x Year), seasonal_ratio, si_mean, years, month_labels, save_path, show.
        - Output: Không trả về giá trị.
    How: Giữ nguyên bố cục 1x2 (figsize=(14, 5)) với nhãn năm bên phải tháng 12, colorbar và dải min-max màu cam.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cmap = cm.viridis
    norm = Normalize(vmin=min(years), vmax=max(years))
    for y in years:
        axes[0].plot(table.index, table[y], marker="o", markersize=3, linewidth=1.5, color=cmap(norm(y)))
        axes[0].annotate(str(y), xy=(12, table.loc[12, y]), xytext=(4, 0), textcoords="offset points", fontsize=7, va="center")
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=axes[0], label="Năm", pad=0.02)
    axes[0].set_xticks(range(1, 13))
    axes[0].set_xticklabels(month_labels)
    axes[0].set_title("Seasonal plot - mỗi đường là một năm", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Tháng")
    axes[0].set_ylabel("Số hành khách (nghìn người)")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].fill_between(
        table.index,
        seasonal_ratio.min(axis=1),
        seasonal_ratio.max(axis=1),
        alpha=0.2,
        color="#ff7f0e",
        label="Khoảng min – max qua các năm",
    )
    axes[1].plot(table.index, si_mean, color="#ff7f0e", marker="o", linewidth=2.2, label="Chỉ số mùa vụ trung bình")
    axes[1].axhline(1, color="black", linestyle="--", linewidth=1)
    axes[1].set_xticks(range(1, 13))
    axes[1].set_xticklabels(month_labels)
    axes[1].set_title("Chỉ số mùa vụ (giá trị tháng / trung bình năm)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Tháng")
    axes[1].set_ylabel("Chỉ số mùa vụ")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    _finalize_figure(fig, save_path, show)


def _draw_heatmap(
    ax: plt.Axes,
    data: pd.DataFrame,
    title: str,
    cmap_name: str,
    fmt: str,
    cbar_label: str,
    month_labels: List[str],
    diverging: bool = False,
) -> None:
    r"""
    Purpose: Vẽ một bảng Heatmap bằng matplotlib.imshow kèm nhãn số tự động đổi màu trắng/đen theo độ tối nền.
    Input-Output:
        - Input: ax, data (DataFrame), title, cmap_name, fmt, cbar_label, month_labels, diverging (bool).
        - Output: Không trả về giá trị.
    How: Giữ nguyên hàm nội bộ gốc từ src/lab01_utils/plotting.py.
    """
    im = ax.imshow(data.values, aspect="auto", cmap=cmap_name)
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=45)
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(month_labels)
    ax.set_xlabel("Năm")
    ax.set_ylabel("Tháng")
    ax.set_title(title, fontsize=12, fontweight="bold")

    vmin, vmax = np.nanmin(data.values), np.nanmax(data.values)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data.values[i, j]
            if diverging:
                dark = abs(val - 1) / max(abs(vmin - 1), abs(vmax - 1)) > 0.5
            else:
                dark = (val - vmin) / (vmax - vmin) > 0.6
            ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=7, color="white" if dark else "black")
    plt.colorbar(im, ax=ax, label=cbar_label, pad=0.02)


def plot_heatmaps(
    table: pd.DataFrame,
    seasonal_ratio: pd.DataFrame,
    month_labels: List[str],
    save_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    r"""
    Purpose: Vẽ 2 Heatmap cạnh nhau: Số hành khách (YlOrRd) và Mùa vụ (RdBu_r).
    Input-Output:
        - Input: table (Month x Year), seasonal_ratio, month_labels, save_path, show.
        - Output: Không trả về giá trị.
    How: Gọi _draw_heatmap() hai lần trên lưới 1x2 (figsize=(16, 6)) giữ nguyên phong cách gốc.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    _draw_heatmap(axes[0], table, "Heatmap số hành khách (tháng × năm)", "YlOrRd", ".0f", "Nghìn người", month_labels)
    _draw_heatmap(axes[1], seasonal_ratio, "Heatmap mùa vụ (giá trị tháng / trung bình năm)", "RdBu_r", ".2f", "Chỉ số", month_labels, diverging=True)
    plt.tight_layout()
    _finalize_figure(fig, save_path, show)


def generate_visualization_suite(
    series: pd.Series,
    rolling_window: int = 12,
    fig_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    r"""
    Purpose: Thực thi 3 biểu đồ cuối của Bài tập 5 (13_rolling_statistics, 14_seasonal_trends, 15_heatmaps_seasonality) và tính thống kê mùa vụ.
    Input-Output:
        - Input: series (pd.Series), rolling_window (int), fig_dir (Optional[Path]).
        - Output: Từ điển chứa tháng đỉnh/đáy, tương quan Rolling Mean-Std, CAGR và danh sách 15 biểu đồ Lab 01.
    How: Tạo pivot Month x Year, tính seasonal_ratio và gọi 3 hàm vẽ nguyên bản.
    """
    roll_mean = series.rolling(window=rolling_window).mean()
    roll_std = series.rolling(window=rolling_window).std()
    valid_roll = pd.DataFrame({"mean": roll_mean, "std": roll_std}).dropna()
    corr_mean_std = float(valid_roll["mean"].corr(valid_roll["std"]))

    df_tmp = pd.DataFrame({"Passengers": series.values}, index=series.index)
    df_tmp["Year"] = df_tmp.index.year
    df_tmp["Month"] = df_tmp.index.month
    table = df_tmp.pivot(index="Month", columns="Year", values="Passengers")
    seasonal_ratio = table / table.mean(axis=0)
    si_mean = seasonal_ratio.mean(axis=1)

    years = table.columns.tolist()
    month_labels_en = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    month_labels_vi = [f"T{m}" for m in range(1, 13)]

    if fig_dir is not None:
        plot_rolling_statistics(series, roll_mean, roll_std, window=rolling_window, save_path=fig_dir / "13_rolling_statistics.png")
        plot_seasonal_trends(table, seasonal_ratio, si_mean, years, month_labels_en, save_path=fig_dir / "14_seasonal_trends.png")
        plot_heatmaps(table, seasonal_ratio, month_labels_vi, save_path=fig_dir / "15_heatmaps_seasonality.png")

    month_means = table.mean(axis=1)
    peak_month = int(month_means.idxmax())
    trough_month = int(month_means.idxmin())
    yearly_means = table.mean(axis=0)
    cagr_pct = float(((yearly_means.iloc[-1] / yearly_means.iloc[0]) ** (1.0 / (len(yearly_means) - 1)) - 1.0) * 100.0)

    all_figs = sorted(p.name for p in fig_dir.glob("*.png")) if fig_dir is not None else []
    return {
        "peak_month": peak_month,
        "peak_month_avg": round(float(month_means.loc[peak_month]), 2),
        "peak_seasonal_index": round(float(si_mean.loc[peak_month]), 4),
        "trough_month": trough_month,
        "trough_month_avg": round(float(month_means.loc[trough_month]), 2),
        "trough_seasonal_index": round(float(si_mean.loc[trough_month]), 4),
        "rolling_mean_std_corr": round(corr_mean_std, 4),
        "annual_cagr_pct": round(cagr_pct, 2),
        "generated_figures": [
            "13_rolling_statistics.png",
            "14_seasonal_trends.png",
            "15_heatmaps_seasonality.png",
        ],
        "all_available_figures": all_figs,
        "total_figures_count": len(all_figs),
    }


def main(run_id: Optional[str] = None) -> Dict[str, Any]:
    r"""
    Purpose: Thực thi Giai đoạn 5 (Bài tập 5: Trực quan hóa dữ liệu chuỗi thời gian) cho Lab 01.
    Input-Output:
        - Input: run_id (Optional[str]).
        - Output: Từ điển kết quả trực quan hóa và danh sách toàn bộ 15 biểu đồ Lab 01.
    How: Nạp AirPassengers, gọi generate_visualization_suite(), lưu JSON vào metrics_dir và cập nhật tracking.
    """
    t0 = time.perf_counter()
    cfg = load_config()
    ensure_directories(cfg)
    active_run_id = get_active_run_id(run_id)

    write_log(active_run_id, "VISUALIZATION", "START", "Bắt đầu Bài 5: Rolling Statistics, Seasonal Trends & Heatmaps.")
    df, _ = load_airpassengers_dataset(cfg)
    series = df[cfg["data"]["value_col"]].astype(float)

    res = generate_visualization_suite(
        series,
        rolling_window=int(cfg["analysis"]["rolling_window"]),
        fig_dir=get_path("figures_dir", cfg),
    )
    res["duration_sec"] = round(time.perf_counter() - t0, 4)

    save_json(res, get_path("metrics_dir", cfg) / "visualization_summary.json")
    record_stage_result(active_run_id, "visualization", res)
    write_log(
        active_run_id,
        "VISUALIZATION",
        "SUCCESS",
        f"Hoàn tất trực quan hóa ({res['total_figures_count']} biểu đồ) | Corr(Mean,Std)={res['rolling_mean_std_corr']} | CAGR={res['annual_cagr_pct']}%/năm.",
    )
    return res