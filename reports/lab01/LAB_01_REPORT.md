# BÁO CÁO KỸ THUẬT LAB 01: PHÂN TÍCH CHUỖI THỜI GIAN (CHƯƠNG 2)

- **Mã phiên chạy (Run ID)**: `20261123_154119_4f8c41`
- **Dataset**: `data/raw/AirPassengers.csv` (144 quan sát, 1949-01-01 → 1960-12-01, tần suất `MS`)

## 1. Mô tả & Khám phá Dữ liệu (EDA) — AirPassengers
| Chỉ tiêu EDA | Giá trị trích xuất |
| --- | --- |
| **Tên bộ dữ liệu & Nguồn** | AirPassengers (data/raw/AirPassengers.csv) |
| **Số quan sát / Tần suất / Kiểu dữ liệu** | 144 tháng | MS (Monthly Start) | DatetimeIndex & int64 |
| **Khoảng thời gian** | 1949-01-01 → 1960-12-01 |
| **Chất lượng dữ liệu gốc** | Missing = 0 | Ngày trùng lặp = 0 (Sạch 100%) |
| **Trung bình (Mean) / Độ lệch chuẩn (Std)** | 280.3 / 119.97 nghìn khách (CV = 42.8%, Skew = 0.583) |
| **Min / Q25 / Median / Q75 / Max** | 104.0 (11/1949) / 180.0 / 265.5 / 360.5 / 622.0 (07/1960) |
| **Đà tăng trưởng & Biên độ mùa vụ** | Tăng gấp 3.86 lần; biên độ năm 1949: 44.0 → năm 1960: 232.0 |
| **Định hướng Train/Test (cho Lab 02)** | 80% Train / 20% Test (shuffle=False) | Train: 115 tháng (01/1949 → 07/1958) | Test: 29 tháng (08/1958 → 12/1960) |

> Lab 01 tập trung phân tích cấu trúc, phân rã và làm sạch; bước chia Train/Test tuần tự được áp dụng ở Lab 02 để chống Data Leakage.

## 2. Kiểm tra ACF / PACF (Bài tập 1 — 40 Lags)
- ACF(1) = **0.948**, ACF(12) = **0.7604**, ACF(24) = **0.5322**, ACF(36) = **0.337**.
- Số lag ACF vượt ngưỡng 95% (±0.1633): **40/40** → chuỗi có tự tương quan rất mạnh, xu hướng tăng và chu kỳ mùa vụ 12 tháng rõ rệt.

## 3. Kiểm tra White Noise (Bài tập 2 — 1000 điểm)
- White Noise mô phỏng: **97.5%** lags nằm trong dải tin cậy 95%, Ljung-Box p-value (lag 20) = **0.1443** (> 0.05).

- 1. Tính ngẫu nhiên & cấu trúc: White Noise dao động ngẫu nhiên quanh trung bình 0 (mean=-0.029, std=0.989), còn AirPassengers có xu hướng tăng trưởng dài hạn và mùa vụ rõ rệt.
- 2. Ngưỡng tin cậy 95%: White Noise có 97.5% số lag nằm trong dải tin cậy ±0.0620 (1/40 lag vượt nhẹ do ngẫu nhiên), trong khi AirPassengers vượt xa ngưỡng ở hầu hết các lag đầu.
- 3. Tốc độ suy giảm ACF: ACF của White Noise xấp xỉ 0 ngay từ lag 1 (ACF(1)=0.0871), còn AirPassengers suy giảm rất chậm (ACF(1)=0.9480) chứng tỏ chuỗi không dừng.
- 4. Kiểm định Ljung-Box (lag 20): White Noise đạt p-value=0.1443 (> 0.05, không bác bỏ H0 — không có tự tương quan), còn AirPassengers có p-value=5.30e-292 (<< 0.05, tự tương quan cực mạnh).
- 5. Khả năng dự báo: White Noise không chứa quy luật quá khứ để dự báo, ngược lại AirPassengers có cấu trúc tự tương quan và chu kỳ 12 tháng rất giàu thông tin cho mô hình SARIMA/ETS.

## 4. Phân rã Chuỗi Thời gian (Bài tập 3 — Classical Multiplicative & STL)
- Phương sai phần dư Classical: Additive = **371.2225** vs Multiplicative = **0.001106** → Mô hình **Multiplicative** phù hợp hơn.
- Độ mạnh Xu hướng $F_T$ = **0.9957** | Độ mạnh Mùa vụ $F_S$ = **0.972**.
- Kiểm định phần dư STL: Mean = **-0.6682**, Std = **7.3585**, Ljung-Box p(10) = **0.0356**, p(20) = **0.2567**.

## 5. Vấn đề Chất lượng Dữ liệu (Bài tập 4 — 11 NaN & 3 Outlier giả lập)
- Điền khuyết: Forward-fill (MAE=38.0909, RMSE=46.3357) vs Time Interpolation (MAE=19.4069, RMSE=23.4042) → Tốt hơn: **Time Interpolation**.
- Làm sạch cuối cùng trên 14 điểm bị làm hỏng (11 NaN + 3 Outlier): MAE = **21.1379**, RMSE = **24.5704**, MAPE = **7.1345%**.

## 6. Ý nghĩa Trực quan hóa (Bài tập 5)
- Tương quan giữa Rolling Mean và Rolling Std (window=12): **r = 0.9591** (phương sai tăng theo xu hướng).
- Cao điểm: **Tháng 7** (TB 351.33, Seasonal Index = 1.2364) | Thấp điểm: **Tháng 11** (TB 232.83, Seasonal Index = 0.832).
- Tăng trưởng kép hàng năm (CAGR): **12.79%/năm**.

## 7. Kết luận: Dữ liệu có phù hợp để dự báo không?
- **RẤT PHÙ HỢP**: Tập dữ liệu AirPassengers không phải nhiễu trắng, sạch (0 missing gốc), có xu hướng tăng dài hạn và cấu trúc mùa vụ 12 tháng cực kỳ ổn định. Vì phương sai tăng theo xu hướng (Multiplicative), nên áp dụng biến đổi log và mô hình **SARIMA(p,d,q)(P,D,Q,12)** hoặc **Holt-Winters Multiplicative** ở Lab 02.
