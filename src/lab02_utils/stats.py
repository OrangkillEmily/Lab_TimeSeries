# src/lab02_utils/stats.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, adfuller, pacf

def perform_adf_test(series: pd.Series, title: str = "Original Series") -> float:
    """
    1. Mục đích:
       Thực hiện kiểm định Augmented Dickey-Fuller (ADF) để kiểm tra tính dừng (stationarity).
       In ra các chỉ số thống kê (ADF Statistic, p-value, Lags, Obs) và kết luận.
       
    2. Tham số đầu vào:
       - series (pd.Series): Chuỗi thời gian 1 chiều cần kiểm định.
       - title (str): Tên của chuỗi dữ liệu (dùng để in tiêu đề).
       
    3. Kết quả đầu ra:
       - float: Giá trị p-value của phép kiểm định.
    """
    print(f"========== KIỂM ĐỊNH ADF CHO: {title.upper()} ==========")
    
    # Chạy hàm adfuller, regression="c" (có hằng số) là mặc định
    result = adfuller(series.dropna(), autolag='AIC', result_object=False)
    
    adf_statistic = result[0]
    p_value = result[1]
    lags_used = result[2]
    obs_used = result[3]
    critical_values = result[4]
    
    print(f"Số quan sát đầu vào : {len(series.dropna())}")
    print(f"Số độ trễ (Lags)    : {lags_used}")
    print(f"Số quan sát sử dụng : {obs_used}")
    print(f"ADF Statistic       : {adf_statistic:.4f}")
    print(f"p-value             : {p_value:.4f}")
    
    print("\n[Giá trị tới hạn - Critical Values]:")
    for key, value in critical_values.items():
        print(f"   > {key}: {value:.4f}")
        
    print("\n[KẾT LUẬN]:")
    print("H0: Chuỗi có nghiệm đơn vị (Chuỗi KHÔNG DỪNG)")
    print("H1: Chuỗi không có nghiệm đơn vị (Chuỗi DỪNG)")
    
    if p_value < 0.05:
        print(f"=> p-value = {p_value:.4f} < 0.05: Đủ cơ sở bác bỏ H0.")
        print("=> CHUỖI ĐÃ DỪNG (Stationary).")
    else:
        print(f"=> p-value = {p_value:.4f} >= 0.05: Chưa đủ cơ sở bác bỏ H0.")
        print("=> CHUỖI KHÔNG DỪNG (Non-stationary). Cần lấy sai phân!")
        
    print("=" * 60 + "\n")
    return p_value


def significant_lags(series: pd.Series, nlags: int = 36, pacf_method: str = "ywm") -> dict:
    """
    1. Mục đích:
       Tính ACF và PACF của một chuỗi đã sai phân rồi liệt kê các lag có giá trị vượt
       ngưỡng tin cậy 95% (±1.96/√n). Dùng để xác định lag bị "cắt cụt" một cách khách quan.

    2. Tham số đầu vào:
       - series (pd.Series): Chuỗi đã lấy sai phân (NaN sẽ bị loại).
       - nlags (int): Số lag tối đa cần xét (mặc định 36 = 3 chu kỳ năm).
       - pacf_method (str): Phương pháp tính PACF của statsmodels (mặc định 'ywm').

    3. Kết quả đầu ra:
       - dict: {'n', 'bound', 'acf_values', 'pacf_values', 'acf_sig', 'pacf_sig'};
         *_values là mảng giá trị theo lag (chỉ số 0 = lag 0), *_sig là danh sách lag vượt ngưỡng.
    """
    x = series.dropna()
    bound = 1.96 / np.sqrt(len(x))
    acf_values = acf(x, nlags=nlags)
    pacf_values = pacf(x, nlags=nlags, method=pacf_method)
    return {
        "n": len(x),
        "bound": bound,
        "acf_values": acf_values,
        "pacf_values": pacf_values,
        "acf_sig": [k for k in range(1, nlags + 1) if abs(acf_values[k]) > bound],
        "pacf_sig": [k for k in range(1, nlags + 1) if abs(pacf_values[k]) > bound],
    }


def plot_acf_pacf(series: pd.Series, lags: int = 36, title: str = "",
                  seasonal_period: int = 12, save_path: str = None) -> None:
    """
    1. Mục đích:
       Vẽ cạnh nhau đồ thị ACF và PACF của chuỗi đã sai phân, kèm đường dọc đánh dấu các
       lag mùa vụ (bội số của chu kỳ s) để dễ phân biệt lag ngắn (p, q) và lag mùa vụ (P, Q).

    2. Tham số đầu vào:
       - series (pd.Series): Chuỗi đã lấy sai phân (KHÔNG dùng chuỗi gốc).
       - lags (int): Số lag hiển thị (mặc định 36).
       - title (str): Tiêu đề chung của hình.
       - seasonal_period (int): Chu kỳ mùa vụ s (mặc định 12 với dữ liệu tháng).
       - save_path (str | None): Đường dẫn lưu ảnh .png (None = không lưu).

    3. Kết quả đầu ra:
       - None: Hiển thị (và lưu nếu có save_path) đồ thị.
    """
    x = series.dropna()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    plot_acf(x, lags=lags, zero=False, ax=axes[0])
    plot_pacf(x, lags=lags, zero=False, method="ywm", ax=axes[1])
    axes[0].set_title("ACF (gợi ý bậc q)")
    axes[1].set_title("PACF (gợi ý bậc p)")
    for ax in axes:
        for k in range(seasonal_period, lags + 1, seasonal_period):
            ax.axvline(k, color="tab:red", linestyle=":", alpha=0.6,
                       label=f"Lag mùa vụ (bội của {seasonal_period})" if k == seasonal_period else None)
        ax.set_xlabel("Lag (tháng)")
        ax.set_ylabel("Hệ số tương quan")
        ax.set_xticks(sorted(set([1, 2, 3, 4] + list(range(seasonal_period, lags + 1, seasonal_period)))))
        ax.legend(loc="upper right")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()

