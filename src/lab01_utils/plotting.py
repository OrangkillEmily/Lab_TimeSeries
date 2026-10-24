import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import STL
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from IPython.display import display 

def evaluate_and_plot_imputation(ts_dirty, ts_true, missing_dates, zoom_range=("1955-03-01", "1955-12-01")):
    """
    Thực hiện điền khuyết (Forward-fill, Interpolation), đánh giá sai số 
    và vẽ biểu đồ đối chiếu với giá trị thật.
    """
    # 1. Điền missing
    ts_ffill = ts_dirty.ffill()
    ts_interp = ts_dirty.interpolate(method="time")

    print("Số NaN còn lại sau Forward-fill :", ts_ffill.isna().sum())
    print("Số NaN còn lại sau Interpolation:", ts_interp.isna().sum())

    # 2. Đối chiếu với giá trị thật
    cmp = pd.DataFrame({
        "Thật": ts_true[missing_dates],
        "Forward-fill": ts_ffill[missing_dates],
        "Interpolation": ts_interp[missing_dates],
    })
    cmp["Sai số FF"] = cmp["Forward-fill"] - cmp["Thật"]
    cmp["Sai số IP"] = cmp["Interpolation"] - cmp["Thật"]
    cmp.index = cmp.index.strftime("%Y-%m")
    
    print("\nSo sánh với giá trị thật tại các điểm bị thiếu:")
    display(cmp.round(1))

    # 3. Tính toán Metrics
    def metrics(est):
        e = est[missing_dates] - ts_true[missing_dates]
        return {
            "MAE": e.abs().mean(),
            "RMSE": np.sqrt((e ** 2).mean()),
            "MAPE (%)": (e.abs() / ts_true[missing_dates]).mean() * 100
        }

    score = pd.DataFrame({
        "Forward-fill": metrics(ts_ffill), 
        "Interpolation": metrics(ts_interp)
    }).T
    
    print("Sai số tổng hợp trên các điểm bị thiếu:")
    display(score.round(2))

    # 4. Vẽ biểu đồ
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))

    # Biểu đồ 1: Toàn chuỗi
    axes[0].plot(ts_dirty.index, ts_dirty.values, color="#1f77b4", linewidth=1.3, label="Dữ liệu có NaN")
    axes[0].scatter(missing_dates, ts_ffill[missing_dates], color="#ff7f0e", marker="s", s=28, zorder=5, label="Forward-fill")
    axes[0].scatter(missing_dates, ts_interp[missing_dates], color="#2ca02c", marker="^", s=28, zorder=5, label="Interpolation")
    axes[0].set_title("Toàn chuỗi: các điểm được điền", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Thời gian (Năm)")
    axes[0].set_ylabel("Số hành khách (nghìn người)")
    axes[0].legend(loc="upper left")

    # Biểu đồ 2: Phóng to
    zoom = slice(zoom_range[0], zoom_range[1])
    axes[1].plot(ts_true[zoom].index, ts_true[zoom].values, color="gray", linestyle="--", marker="o", markersize=4, label="Giá trị thật")
    axes[1].step(ts_ffill[zoom].index, ts_ffill[zoom].values, where="post", color="#ff7f0e", linewidth=1.8, label="Forward-fill")
    axes[1].plot(ts_interp[zoom].index, ts_interp[zoom].values, color="#2ca02c", linewidth=1.8, marker="^", markersize=5, label="Interpolation")
    
    # Lấy năm/tháng để làm title động
    title_start = pd.to_datetime(zoom_range[0]).strftime('%m/%Y')
    title_end = pd.to_datetime(zoom_range[1]).strftime('%m/%Y')
    axes[1].set_title(f"Phóng to khoảng thiếu ({title_start} – {title_end})", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Thời gian")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    axes[1].legend(loc="upper left")

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()
    
    # Trả về các chuỗi đã xử lý để có thể dùng tiếp ở các cell khác nếu cần
    return ts_ffill, ts_interp

def detect_and_plot_rolling_outliers(ts_filled, outlier_dates, window=12, k=3):
    """
    Phát hiện outlier bằng Rolling mean / Rolling variance và vẽ biểu đồ minh họa.
    """
    # 1. Tính toán Rolling Statistics
    # shift(1): cửa sổ chỉ gồm window tháng TRƯỚC đó, để điểm đang xét không làm nhiễu chính cửa sổ của nó
    roll_mean = ts_filled.rolling(window).mean().shift(1)
    roll_std = ts_filled.rolling(window).std().shift(1)
    roll_var = roll_std ** 2
    
    roll_z = (ts_filled - roll_mean) / roll_std
    out_roll = roll_z.abs() > k

    # 2. In kết quả phát hiện
    print(f"Rolling (window={window}, ngưỡng |z| > {k}) phát hiện {out_roll.sum()} điểm:")
    display(pd.DataFrame({
        "Giá trị": ts_filled[out_roll], 
        "Rolling mean": roll_mean[out_roll].round(1),
        "Rolling std": roll_std[out_roll].round(1), 
        "z": roll_z[out_roll].round(2)
    }))
    print("z của các outlier đã chèn:", roll_z[outlier_dates].round(2).to_dict())

    # 3. Vẽ biểu đồ
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Biểu đồ 1: Phát hiện Outlier
    axes[0].plot(ts_filled.index, ts_filled.values, color="#1f77b4", linewidth=1.2, label="Chuỗi (đã điền missing)")
    axes[0].plot(roll_mean.index, roll_mean.values, color="#ff7f0e", linewidth=2, label=f"Rolling mean (window={window})")
    axes[0].fill_between(roll_mean.index, roll_mean - k * roll_std, roll_mean + k * roll_std,
                         color="#ff7f0e", alpha=0.15, label=f"Dải ±{k} rolling std")
    axes[0].scatter(ts_filled.index[out_roll], ts_filled[out_roll], color="#d62728", s=60, zorder=5, label="Bị đánh dấu outlier")
    axes[0].scatter(outlier_dates, ts_filled[outlier_dates], facecolors="none", edgecolors="black", s=140, zorder=6, label="Outlier thật (đã chèn)")
    
    axes[0].set_title(f"Phát hiện outlier bằng Rolling mean ± {k} Rolling std", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Số hành khách (nghìn người)")
    axes[0].legend(loc="upper left", fontsize=8)

    # Biểu đồ 2: Phương sai (Variance)
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
    plt.show()
    
    return out_roll

def detect_and_plot_stl_outliers(ts_filled, outlier_dates):
    """
    Thực hiện tính toán Z-score và IQR trên chuỗi gốc và phần dư STL, 
    in kết quả phát hiện outlier và vẽ biểu đồ đối chiếu.
    """
    # --- 1. PHẦN TÍNH TOÁN ---
    # (A) Áp dụng trực tiếp lên chuỗi gốc
    z_raw = (ts_filled - ts_filled.mean()) / ts_filled.std()
    out_z_raw = z_raw.abs() > 3

    q1, q3 = ts_filled.quantile([0.25, 0.75])
    iqr = q3 - q1
    out_iqr_raw = (ts_filled < q1 - 1.5 * iqr) | (ts_filled > q3 + 1.5 * iqr)

    # (B) Áp dụng lên phần dư STL (thang log)
    log_resid = STL(np.log(ts_filled), period=12, robust=False).fit().resid
    z_res = (log_resid - log_resid.mean()) / log_resid.std()
    out_z_res = z_res.abs() > 3

    q1r, q3r = log_resid.quantile([0.25, 0.75])
    iqr_r = q3r - q1r
    out_iqr_res = (log_resid < q1r - 1.5 * iqr_r) | (log_resid > q3r + 1.5 * iqr_r)       
    out_iqr3_res = (log_resid < q1r - 3 * iqr_r) | (log_resid > q3r + 3 * iqr_r)          

    # --- 2. PHẦN IN KẾT QUẢ ---
    print(f"(A) Chuỗi gốc  - Z-score |z|>3 : {out_z_raw.sum()} điểm | IQR 1.5x: {out_iqr_raw.sum()} điểm")
    print(f"(B) Phần dư STL - Z-score |z|>3 : {out_z_res.sum()} điểm | IQR 1.5x: {out_iqr_res.sum()} điểm | IQR 3x: {out_iqr3_res.sum()} điểm\n")
    print("z-score (phần dư STL) tại 3 outlier đã chèn:", z_res[outlier_dates].round(2).to_dict())

    # --- 3. PHẦN VẼ BIỂU ĐỒ ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Biểu đồ 1: Chuỗi gốc
    axes[0].plot(ts_filled.index, ts_filled.values, color="#1f77b4", linewidth=1.2)
    axes[0].axhline(q3 + 1.5 * iqr, color="#9467bd", linestyle="--", label="Hàng rào IQR trên (toàn cục)")
    axes[0].axhline(ts_filled.mean() + 3 * ts_filled.std(), color="#8c564b", linestyle=":", label="Mean + 3 std (toàn cục)")
    axes[0].scatter(outlier_dates, ts_filled[outlier_dates], facecolors="none", edgecolors="black", s=140, zorder=6, label="Outlier thật (đã chèn)")
    axes[0].set_title("(A) Z-score / IQR trực tiếp trên chuỗi gốc: ngưỡng toàn cục không bắt được outlier", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Số hành khách")
    axes[0].legend(loc="upper left", fontsize=8)

    # Biểu đồ 2: Phần dư STL (log)
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
    plt.show()
    
    return out_z_res, log_resid

def evaluate_outlier_methods(ts_filled, log_resid, out_roll, outlier_dates):
    """
    So sánh độ hiệu quả (Bắt đúng, Báo nhầm, Bỏ sót) của các phương pháp phát hiện outlier.
    """
    # 1. Tính toán lại nhanh các mask 
    # --- Chuỗi gốc ---
    z_raw = (ts_filled - ts_filled.mean()) / ts_filled.std()
    out_z_raw = z_raw.abs() > 3
    q1, q3 = ts_filled.quantile([0.25, 0.75])
    iqr = q3 - q1
    out_iqr_raw = (ts_filled < q1 - 1.5 * iqr) | (ts_filled > q3 + 1.5 * iqr)
    
    # --- Phần dư STL ---
    z_res = (log_resid - log_resid.mean()) / log_resid.std()
    out_z_res = z_res.abs() > 3
    q1r, q3r = log_resid.quantile([0.25, 0.75])
    iqr_r = q3r - q1r
    out_iqr_res = (log_resid < q1r - 1.5 * iqr_r) | (log_resid > q3r + 1.5 * iqr_r)
    out_iqr3_res = (log_resid < q1r - 3 * iqr_r) | (log_resid > q3r + 3 * iqr_r)

    # 2. Gom vào dictionary
    methods = {
        "Rolling mean/std (|z|>3)": out_roll,
        "Z-score chuỗi gốc (|z|>3)": out_z_raw,
        "IQR 1.5x chuỗi gốc": out_iqr_raw,
        "Z-score phần dư STL (|z|>3)": out_z_res,
        "IQR 1.5x phần dư STL": out_iqr_res,
        "IQR 3x phần dư STL": out_iqr3_res,
    }
    
    # 3. Tạo Ground Truth một cách an toàn
    true_out = pd.Series(False, index=ts_filled.index)
    outlier_dates_dt = pd.to_datetime(outlier_dates) # Ép kiểu để tránh lỗi Index
    true_out.loc[outlier_dates_dt] = True

    # 4. Đánh giá (Tính TP, FP, FN)
    rows = []
    for name, flag in methods.items():
        # Xử lý an toàn: điền False cho các ô NaN và ép chuẩn về kiểu Boolean
        flag_safe = flag.fillna(False).astype(bool)
        
        tp = int((flag_safe & true_out).sum())
        fp = int((flag_safe & ~true_out).sum())
        fn = int((~flag_safe & true_out).sum())
        
        rows.append({
            "Phương pháp": name, 
            "Số điểm đánh dấu": int(flag_safe.sum()),
            "Bắt đúng (/3)": tp, 
            "Báo nhầm": fp, 
            "Bỏ sót": fn
        })
        
    summary = pd.DataFrame(rows).set_index("Phương pháp")
    display(summary)

    # 5. Khắc phục lỗi NaN ở lệnh Print bằng flag_safe
    out_z_res_safe = out_z_res.fillna(False).astype(bool)
    out_iqr_res_safe = out_iqr_res.fillna(False).astype(bool)

    print("\nCác điểm bị Z-score (phần dư STL) đánh dấu:", 
          [d.strftime("%Y-%m") for d in ts_filled.index[out_z_res_safe]])
    print("Các điểm bị IQR 1.5x (phần dư STL) đánh dấu:", 
          [d.strftime("%Y-%m") for d in ts_filled.index[out_iqr_res_safe]])
          
    return summary


def plot_before_after_cleaning(ts_dirty, ts_ffill, ts_interp, ts_true, ts_clean, missing_dates, outlier_dates):
    """
    Vẽ 3 biểu đồ đối chiếu các giai đoạn: 
    Trước xử lý -> Sau khi điền Missing -> Sau khi làm sạch toàn bộ (Missing + Outlier)
    """
    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True, sharey=True)

    # --- TRƯỚC khi xử lý ---
    axes[0].plot(ts_dirty.index, ts_dirty.values, color="#1f77b4", linewidth=1.3, marker="o", markersize=3, label="Dữ liệu bẩn")
    for d in missing_dates:
        axes[0].axvspan(d - pd.Timedelta(days=15), d + pd.Timedelta(days=15), color="#ff7f0e", alpha=0.35, linewidth=0)
    axes[0].scatter(outlier_dates, ts_dirty[outlier_dates], color="#d62728", s=70, zorder=5, label="Outlier")
    axes[0].plot([], [], color="#ff7f0e", alpha=0.5, linewidth=8, label="Vị trí NaN (11 điểm)")
    axes[0].set_title("TRƯỚC khi xử lý: có NaN (đứt đoạn) và 3 outlier", fontsize=12, fontweight="bold")
    axes[0].legend(loc="upper left")

    # --- SAU khi chỉ điền missing ---
    axes[1].plot(ts_ffill.index, ts_ffill.values, color="#ff7f0e", linewidth=1.3, label="Forward-fill")
    axes[1].plot(ts_interp.index, ts_interp.values, color="#2ca02c", linewidth=1.3, label="Interpolation")
    axes[1].scatter(outlier_dates, ts_interp[outlier_dates], color="#d62728", s=70, zorder=5, label="Outlier vẫn còn")
    axes[1].set_title("SAU khi xử lý missing: Forward-fill vs Interpolation (outlier chưa xử lý)", fontsize=12, fontweight="bold")
    axes[1].legend(loc="upper left")

    # --- SAU khi xử lý cả missing và outlier ---
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
    plt.show()


def plot_residuals_comparison(ts_clean, log_resid, out_z_res):
    """
    Tính toán và vẽ đối chiếu phần dư STL trước và sau khi làm sạch dữ liệu.
    """
    # 1. Tính toán phần dư
    resid_before = log_resid
    resid_after = STL(np.log(ts_clean), period=12, robust=False).fit().resid

    # 2. Vẽ 2 biểu đồ
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    
    # Biểu đồ 1: Line chart phần dư
    axes[0].plot(resid_before.index, resid_before.values, color="#d62728", linewidth=1.2, label="Trước (còn outlier)")
    axes[0].plot(resid_after.index, resid_after.values, color="#2ca02c", linewidth=1.2, label="Sau (ts_clean)")
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Phần dư STL (thang log) trước và sau xử lý", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Thời gian (Năm)")
    axes[0].set_ylabel("Phần dư (log)")
    axes[0].legend()

    # Biểu đồ 2: Boxplot
    # (Lưu ý: Nếu Matplotlib của bạn bản cũ báo lỗi tham số tick_labels, hãy đổi thành labels=["Trước", "Sau"])
    axes[1].boxplot([resid_before.values, resid_after.values], tick_labels=["Trước", "Sau"], showmeans=True)
    axes[1].set_title("Boxplot phần dư STL", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Phần dư (log)")
    
    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.4)
        
    plt.tight_layout()
    plt.show()

    # 3. Tính và in kết quả thống kê (Thêm xử lý an toàn fillna để không bị lỗi NaN)
    z_after = (resid_after - resid_after.mean()) / resid_after.std()
    outliers_after = (z_after.abs() > 3).sum()
    
    print(f"Độ lệch chuẩn phần dư: trước = {resid_before.std():.4f} | sau = {resid_after.std():.4f}")
    print(f"Số điểm |z|>3 trên phần dư: trước = {int(out_z_res.fillna(False).sum())} | sau = {int(outliers_after)}")



def plot_full_series(ts):
    """Vẽ line chart toàn bộ chuỗi thời gian, đánh dấu điểm Min/Max."""
    min_val = ts.min()
    max_val = ts.max()
    mean_val = ts.mean()
    min_date = ts.idxmin()
    max_date = ts.idxmax()
    plt.figure(figsize=(20, 8)) 
    plt.plot(ts.index, ts.values, color='royalblue', label='Original Series', marker='.', alpha=0.8)
    plt.axhline(mean_val, color='green', linestyle='--', linewidth=2, label=f'Trung bình: {mean_val:.1f}')
    plt.axhline(min_val, color='red', linestyle=':', alpha=0.5)
    plt.axhline(max_val, color='darkorange', linestyle=':', alpha=0.5)
    plt.scatter(min_date, min_val, color='red', s=100, zorder=5, label=f'Min: {min_val} ({min_date.strftime("%m/%Y")})')
    plt.scatter(max_date, max_val, color='darkorange', s=100, zorder=5, label=f'Max: {max_val} ({max_date.strftime("%m/%Y")})')
    plt.xlim(ts.index.min(), ts.index.max())

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
    plt.xticks(rotation=90, fontsize=8)
# Thêm lưới
    ax.xaxis.grid(True, linestyle='--', alpha=0.5)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)

    plt.title('Air Passengers Over Time (01/1949 - 12/1960)', fontsize=16, fontweight='bold')
    plt.xlabel('Thời gian (Tháng/Năm)', fontsize=12)
    plt.ylabel('Số lượng hành khách (Nghìn người)', fontsize=12)

    plt.legend(loc='upper left')
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