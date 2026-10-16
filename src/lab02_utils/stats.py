# src/lab02_utils/stats.py
import pandas as pd
from statsmodels.tsa.stattools import adfuller

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