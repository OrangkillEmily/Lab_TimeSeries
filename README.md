# Phân tích Chuỗi thời gian (Time Series Analysis) - Lab 01 & Lab 02

Repository này chứa toàn bộ mã nguồn, dữ liệu và báo cáo cho hai bài thực hành (Lab 01 và Lab 02) về Phân tích Chuỗi thời gian.

Dự án tập trung vào việc khám phá dữ liệu (EDA), xử lý nhiễu, kiểm định tính dừng và xây dựng các mô hình dự báo thống kê (ARIMA/SARIMA).

## 1. Tổng quan Dự án

**Dataset:** airline-passengers.csv (144 quan sát, dữ liệu hàng tháng về số lượng hành khách hàng không, chu kỳ $s=12$).

**Mục tiêu Lab 01:** Khám phá dữ liệu (EDA), phân tích nhiễu trắng (White Noise), phân rã chuỗi (Decomposition), xử lý dữ liệu khuyết thiếu/ngoại lệ và trực quan hóa chuyên sâu.

**Mục tiêu Lab 02:** Kiểm định tính dừng (ADF Test), lấy sai phân, phân tích ACF/PACF, huấn luyện và đánh giá mô hình dự báo ARIMA và SARIMA.

**Nguyên tắc cốt lõi:** Code tường minh, tận dụng tối đa pandas và statsmodels. Dữ liệu gốc (raw/) là Read-only; mọi dữ liệu sinh ra hoặc đã xử lý đều được lưu tại processed/. Giữ nguyên cấu trúc thời gian khi chia tập huấn luyện (shuffle=False).


## 2. Cấu trúc Thư mục (Repository Structure)

```text
├── agents/                 # Chứa các file yêu cầu gốc của Lab 01 và Lab 02
├── data/
│   ├── raw/                # Dữ liệu gốc (airline-passengers.csv)
│   └── processed/          # Dữ liệu đã qua xử lý (missing/outliers) và tập Train/Test
├── docs/                   # Báo cáo đánh giá (Lab 01) và methodology/câu hỏi lý thuyết (Lab 02)
├── notebooks/
│   ├── lab01/              # Các notebook phân tích EDA, White Noise, Decomposition...
│   │   └── Lab01_TimeSeries.ipynb  # File tổng hợp Lab 01
│   └── lab02/              # Các notebook kiểm định ADF, sai phân, ACF/PACF, ARIMA, SARIMA
│       └── Lab02_TimeSeries.ipynb  # File tổng hợp Lab 02
├── reports/                # Nơi lưu trữ biểu đồ xuất ra (ACF/PACF, STL Plot, Heatmap, Forecast Plot)
└── src/                    # Mã nguồn dùng chung
    ├── data_loader.py      # Hàm load file CSV và thiết lập DatetimeIndex
    ├── lab01_utils/        # Các module hỗ trợ Lab 01: decomposition.py, plotting.py
    └── lab02_utils/        # Các module hỗ trợ Lab 02: stats.py (ADF/ACF), modeling.py (ARIMA/Metrics)
```

## 3. Luồng Xử lý Tiêu chuẩn (Workflow)

### Lab 01: Nhận diện Đặc trưng & Đánh giá Dữ liệu

**Khám phá ban đầu:** Tải dữ liệu, tính và vẽ đồ thị ACF/PACF đến 40 lags.

**Phân tích Nhiễu trắng:** Giả lập 1000 điểm White Noise bằng numpy.random.normal, đối chiếu đặc tính tự tương quan với chuỗi dữ liệu thực tế.

**Phân rã chuỗi (Decomposition):** Tách chuỗi thành 3 thành phần: Trend (Xu hướng), Seasonal (Mùa vụ, chu kỳ 12), và Residual (Phần dư).

**Xử lý chất lượng dữ liệu:** Giả lập lỗi, lấp đầy dữ liệu thiếu bằng Forward-fill/Interpolation, quét ngoại lệ bằng Z-score/IQR.

**Trực quan hóa:** Vẽ Rolling mean/std, Seasonal plot và Heatmap để tìm ra các quy luật ẩn.


### Lab 02: Kiểm định & Xây dựng Mô hình Dự báo

**Tiền xử lý:** Tách tập Train (80%) và Test (20%) theo đúng trình tự thời gian.

**Kiểm định tính dừng:** Sử dụng Augmented Dickey-Fuller (ADF) trên chuỗi gốc và chuỗi sau khi lấy sai phân (thường $d=1$ và mùa vụ $D=1$).

**Chốt tham số:** Đọc đồ thị ACF/PACF của chuỗi đã sai phân để thiết lập cấu trúc $(p,d,q)$ và $(P,D,Q,s)$.

**Huấn luyện & Chẩn đoán:** Fit các mô hình ARIMA và SARIMA. Kiểm tra phần dư (Residual Diagnostics) để đảm bảo mô hình đã trích xuất hết thông tin (hướng tới White Noise).

**Dự báo & Đánh giá:** Chạy dự báo trên tập Test, vẽ biểu đồ đối chứng và tính toán các thang đo sai số: MAE, RMSE, MAPE, $R^2$. So sánh hiệu năng giữa ARIMA và SARIMA.

## 4. Công cụ & Thư viện sử dụng

**Xử lý dữ liệu:** pandas, numpy

**Mô hình & Thống kê:** statsmodels (ADF, Ljung-Box, ARIMA, SARIMAX), scikit-learn (Metrics)

**Trực quan hóa:** matplotlib, seaborn