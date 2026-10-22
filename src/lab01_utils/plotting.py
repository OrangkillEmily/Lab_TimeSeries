import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import Normalize
import matplotlib.cm as cm

def plot_missing_filling(ts_dirty, ts_ffill, ts_interp, ts_true, missing_dates, zoom_range, zoom_only=False):
    """Vẽ đối chiếu điền dữ liệu khuyết thiếu (toàn chuỗi và phóng to, hoặc chỉ phóng to để tránh trùng lặp)."""
    if zoom_only:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        zoom = slice(zoom_range[0], zoom_range[1])
        ax.plot(ts_true[zoom].index, ts_true[zoom].values, color="gray", linestyle="--", marker="o", markersize=4, label="Giá trị thật (đáp án)")
        ax.step(ts_ffill[zoom].index, ts_ffill[zoom].values, where="post", color="#ff7f0e", linewidth=1.8, label="Forward-fill (bậc thang)")
        ax.plot(ts_interp[zoom].index, ts_interp[zoom].values, color="#2ca02c", linewidth=1.8, marker="^", markersize=5, label="Interpolation (nội suy)")
        ax.set_title(f"Phóng to khoảng thiếu ({zoom_range[0][:7]} đến {zoom_range[1][:7]}): Forward-fill vs Interpolation", fontsize=12, fontweight="bold")
        ax.set_xlabel("Thời gian")
        ax.set_ylabel("Số hành khách (nghìn người)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
        ax.legend(loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.show()
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))

    # Biểu đồ 1: Toàn chuỗi
    axes[0].plot(ts_dirty.index, ts_dirty.values, color="#1f77b4", linewidth=1.3, label="Dữ liệu có NaN")
    axes[0].scatter(missing_dates, ts_ffill[missing_dates], color="#ff7f0e", marker="s", s=28, zorder=5, label="Forward-fill")
    axes[0].scatter(missing_dates, ts_interp[missing_dates], color="#2ca02c", marker="^", s=28, zorder=5, label="Interpolation")
    axes[0].set_title("Toàn chuỗi: Các điểm được điền", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Thời gian (Năm)")
    axes[0].set_ylabel("Số hành khách")
    axes[0].legend(loc="upper left")

    # Biểu đồ 2: Phóng to khu vực thiếu liên tiếp
    zoom = slice(zoom_range[0], zoom_range[1])
    axes[1].plot(ts_true[zoom].index, ts_true[zoom].values, color="gray", linestyle="--", marker="o", markersize=4, label="Giá trị thật")
    axes[1].step(ts_ffill[zoom].index, ts_ffill[zoom].values, where="post", color="#ff7f0e", linewidth=1.8, label="Forward-fill")
    axes[1].plot(ts_interp[zoom].index, ts_interp[zoom].values, color="#2ca02c", linewidth=1.8, marker="^", markersize=5, label="Interpolation")
    axes[1].set_title(f"Phóng to khoảng thiếu ({zoom_range[0][:7]} đến {zoom_range[1][:7]})", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Thời gian")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    axes[1].legend(loc="upper left")

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

def plot_outlier_detection_stl(ts_filled, log_resid, outlier_dates, out_z_res):
    """Vẽ phương pháp phát hiện outlier trên phần dư STL (bài toán thực tế)."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Tính ngưỡng IQR
    q1r, q3r = log_resid.quantile([0.25, 0.75])
    iqr_r = q3r - q1r

    # Biểu đồ 1: Chuỗi đã nội suy missing (nhưng vẫn còn outlier)
    axes[0].plot(ts_filled.index, ts_filled.values, color="#1f77b4", linewidth=1.2)
    axes[0].scatter(outlier_dates, ts_filled[outlier_dates], facecolors="none", edgecolors="black", s=140, zorder=6, label="Outlier thật (đã chèn)")
    axes[0].set_title("Chuỗi quan sát (đã nội suy NaN)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Số hành khách")
    axes[0].legend(loc="upper left")

    # Biểu đồ 2: Áp dụng Z-score và IQR trên phần dư STL
    axes[1].plot(log_resid.index, log_resid.values, color="#2ca02c", linewidth=1.2, label="Phần dư STL (thang log)")
    axes[1].axhline(log_resid.mean() + 3 * log_resid.std(), color="#d62728", linestyle="--", label="±3 std (Z-score)")
    axes[1].axhline(log_resid.mean() - 3 * log_resid.std(), color="#d62728", linestyle="--")
    axes[1].axhline(q3r + 1.5 * iqr_r, color="#9467bd", linestyle=":", label="Hàng rào IQR 1.5x")
    axes[1].axhline(q1r - 1.5 * iqr_r, color="#9467bd", linestyle=":")
    axes[1].scatter(log_resid.index[out_z_res], log_resid[out_z_res], color="#d62728", s=60, zorder=5, label="Được đánh dấu (Z-score)")
    axes[1].scatter(outlier_dates, log_resid[outlier_dates], facecolors="none", edgecolors="black", s=140, zorder=6, label="Outlier thật")
    
    axes[1].set_title("Z-score / IQR trên phần dư STL (Outlier tách biệt rất rõ ràng)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Thời gian (Năm)")
    axes[1].set_ylabel("Phần dư (log)")
    axes[1].legend(loc="lower left", fontsize=8, ncol=2)
    
    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

def plot_before_after_cleaning(ts_dirty, ts_clean, ts_true, missing_dates, outlier_dates):
    """Vẽ chuỗi trước và sau toàn bộ quá trình Data Quality."""
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    # TRƯỚC khi xử lý
    axes[0].plot(ts_dirty.index, ts_dirty.values, color="#1f77b4", linewidth=1.3, marker="o", markersize=3, label="Dữ liệu bẩn")
    for d in missing_dates:
        axes[0].axvspan(d - pd.Timedelta(days=15), d + pd.Timedelta(days=15), color="#ff7f0e", alpha=0.35, linewidth=0)
    axes[0].scatter(outlier_dates, ts_dirty[outlier_dates], color="#d62728", s=70, zorder=5, label="Outlier")
    axes[0].plot([], [], color="#ff7f0e", alpha=0.5, linewidth=8, label="Vị trí NaN (11 điểm)")
    axes[0].set_title("TRƯỚC KHI XỬ LÝ: Dữ liệu chứa Outlier và bị thủng (NaN)", fontsize=12, fontweight="bold")
    axes[0].legend(loc="upper left")

    # SAU khi xử lý (Clean)
    axes[1].plot(ts_true.index, ts_true.values, color="gray", linestyle="--", linewidth=1.2, label="Dữ liệu thật (đáp án)")
    axes[1].plot(ts_clean.index, ts_clean.values, color="#1f77b4", linewidth=1.5, label="Dữ liệu sạch (ts_clean)")
    axes[1].scatter(missing_dates, ts_clean[missing_dates], color="#2ca02c", marker="^", s=40, zorder=5, label="Điểm đã nội suy (missing)")
    axes[1].scatter(outlier_dates, ts_clean[outlier_dates], color="#9467bd", marker="D", s=55, zorder=5, label="Điểm đã thay thế (outlier)")
    axes[1].set_title("SAU KHI XỬ LÝ (CLEAN): Điền Missing bằng Interpolation + Lọc Outlier bằng Z-score trên phần dư", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Thời gian (Năm)")
    axes[1].legend(loc="upper left")

    for ax in axes:
        ax.set_ylabel("Số hành khách")
        ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()
    
 
 
    

def plot_full_series(ts):
    """Vẽ line chart toàn bộ chuỗi thời gian, đánh dấu điểm Min/Max."""
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(ts.index, ts.values, color="#1f77b4", linewidth=1.5, marker="o", markersize=3)

    # Đánh dấu giá trị lớn nhất và nhỏ nhất
    t_max, v_max = ts.idxmax(), ts.max()
    t_min, v_min = ts.idxmin(), ts.min()
    
    ax.scatter([t_max], [v_max], color="#d62728", zorder=5)
    ax.annotate(f"Cao nhất: {v_max} ({t_max:%m/%Y})", xy=(t_max, v_max),
                xytext=(-150, -5), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="#d62728"), color="#d62728")
                
    ax.scatter([t_min], [v_min], color="#2ca02c", zorder=5)
    ax.annotate(f"Thấp nhất: {v_min} ({t_min:%m/%Y})", xy=(t_min, v_min),
                xytext=(15, 25), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="#2ca02c"), color="#2ca02c")

    ax.set_title("AirPassengers - Số hành khách theo tháng (1949–1960)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Thời gian (Năm)")
    ax.set_ylabel("Số hành khách (nghìn người)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()

def plot_rolling_statistics(ts, roll_mean, roll_std, window):
    """Vẽ Rolling Mean và Rolling Standard Deviation."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Biểu đồ 1: chuỗi gốc và rolling mean
    axes[0].plot(ts.index, ts.values, color="#1f77b4", linewidth=1.0, alpha=0.6, label="Chuỗi gốc")
    axes[0].plot(roll_mean.index, roll_mean.values, color="#ff7f0e", linewidth=2.2,
                 label=f"Rolling mean (window={window})")
    axes[0].set_title("Chuỗi gốc và Rolling Mean", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Số hành khách (nghìn người)")
    axes[0].legend(loc="upper left")

    # Biểu đồ 2: rolling std
    axes[1].plot(roll_std.index, roll_std.values, color="#d62728", linewidth=2.0,
                 label=f"Rolling std (window={window})")
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
    plt.show()

def plot_seasonal_trends(table, seasonal_ratio, si_mean, years, month_labels):
    """Vẽ biểu đồ mùa vụ (mỗi năm 1 đường) và chỉ số mùa vụ trung bình."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Biểu đồ 1: mỗi năm một đường
    cmap = cm.viridis
    norm = Normalize(vmin=min(years), vmax=max(years))
    for y in years:
        axes[0].plot(table.index, table[y], marker="o", markersize=3, linewidth=1.5, color=cmap(norm(y)))
        axes[0].annotate(str(y), xy=(12, table.loc[12, y]), xytext=(4, 0),
                         textcoords="offset points", fontsize=7, va="center")
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=axes[0], label="Năm", pad=0.02)
    axes[0].set_xticks(range(1, 13))
    axes[0].set_xticklabels(month_labels)
    axes[0].set_title("Seasonal plot - mỗi đường là một năm", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Tháng")
    axes[0].set_ylabel("Số hành khách (nghìn người)")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # Biểu đồ 2: chỉ số mùa vụ 
    axes[1].fill_between(table.index, seasonal_ratio.min(axis=1), seasonal_ratio.max(axis=1),
                         alpha=0.2, color="#ff7f0e", label="Khoảng min – max qua các năm")
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
    plt.show()

def _draw_heatmap(ax, data, title, cmap_name, fmt, cbar_label, month_labels, diverging=False):
    """Hàm nội bộ (helper) để vẽ 1 heatmap."""
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
            ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=7,
                    color="white" if dark else "black")
    plt.colorbar(im, ax=ax, label=cbar_label, pad=0.02)

def plot_heatmaps(table, seasonal_ratio, month_labels):
    """Vẽ 2 heatmap cạnh nhau (Số tuyệt đối và Tỷ lệ mùa vụ)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    _draw_heatmap(axes[0], table, "Heatmap số hành khách (tháng × năm)", "YlOrRd", ".0f", "Nghìn người", month_labels)
    _draw_heatmap(axes[1], seasonal_ratio, "Heatmap mùa vụ (giá trị tháng / trung bình năm)", "RdBu_r", ".2f", "Chỉ số", month_labels, diverging=True)
    plt.tight_layout()
    plt.show()