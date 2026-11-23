# 01 — Giai đoạn Khám phá Dữ liệu, ACF/PACF & Kiểm định White Noise (`EDA_Phase`)

## 1. Purpose
Giai đoạn này thiết lập bức tranh thống kê nền tảng cho chuỗi thời gian `AirPassengers` trước khi phân rã và làm sạch dữ liệu:
- Nạp và kiểm toán tính toàn vẹn dữ liệu gốc qua lõi dùng chung `src/common/data_loader.py` (tần suất tháng `MS`, `DatetimeIndex` tăng dần, không có giá trị thiếu hay mốc thời gian trùng lặp).
- Định lượng xu hướng tăng trưởng dài hạn và biên độ mùa vụ giãn nở qua hồ sơ thống kê mô tả chi tiết.
- Đánh giá mức độ tự tương quan tuyến tính qua **ACF / PACF (40 lags)** và chứng minh bằng kiểm định **Ljung-Box** rằng chuỗi **không phải Nhiễu trắng (White Noise)**, có cấu trúc dự báo rõ rệt.

## 2. Input-Output
| Thành phần | Đặc tả chi tiết |
|---|---|
| **Input** | `data/raw/AirPassengers.csv` (`144 × 2`: `Month`, `Passengers`), nhánh cấu hình `lab01` trong `config.yaml` (`random_seed = 42`, `acf_pacf_nlags = 40`, `white_noise_n = 1000`). |
| **Module xử lý** | `src/common/data_loader.py` (`load_time_series_csv`), `src/lab01/data_loader.py` (`load_airpassengers_dataset`), `src/lab01/eda.py` (`build_eda_profile`, `analyze_acf_pacf`, `analyze_white_noise`), `src/lab01/visualize.py` (`plot_full_series`). |
| **Output (Data/JSON)** | `data/processed/lab01/airpassengers_clean.csv`, `data/processed/lab01/data_inspection_summary.json`, `reports/lab01/metrics/eda_acf_whitenoise_summary.json`, `progress_results/lab01/latest_run.json`. |
| **Output (Figures)** | `reports/lab01/figures/01_full_series_overview.png`, `02_acf_pacf_airpassengers.png`, `03_white_noise_series.png`, `04_white_noise_vs_airpassengers_acf.png`. |

## 3. How to do
1. **Nạp & Kiểm toán Schema (`src/lab01/data_loader.py`):** Đọc `data/raw/AirPassengers.csv` qua `load_time_series_csv(filepath, date_col="Month", freq="MS")`, xác nhận `144` quan sát liên tục (`1949-01-01` → `1960-12-01`), `missing_values = 0`, `duplicate_timestamps = 0`, `is_monotonic = True`.
2. **Xây dựng hồ sơ EDA (`build_eda_profile`):** Tính `mean`, `std`, `CV (%)`, `skewness`, `kurtosis`, `min/max` kèm tháng (`MM/YYYY`), phân vị `Q25 / Median / Q75`, hệ số tăng trưởng đầu–cuối kỳ (`growth_ratio`) và biên độ đỉnh–đáy năm đầu (`1949`) so với năm cuối (`1960`).
3. **Phân tích ACF & PACF (`analyze_acf_pacf`, `nlags = 40`):**
   - Hệ số tự tương quan tại độ trễ $k$:
     $$\rho_k = \frac{\sum_{t=k+1}^{n}(y_t - \bar{y})(y_{t-k} - \bar{y})}{\sum_{t=1}^{n}(y_t - \bar{y})^2}$$
   - Ngưỡng tin cậy 95% của nhiễu trắng: $\pm \frac{1.96}{\sqrt{n}} = \pm 0.1633$ (với $n = 144$).
4. **Mô phỏng & Kiểm định White Noise (`analyze_white_noise`):** Sinh chuỗi $\varepsilon_t \sim \mathcal{N}(0, 1)$ gồm `1000` điểm (`seed = 42`, ngưỡng 95% $\pm 0.0620$), đối chiếu trực quan ACF và chạy kiểm định **Ljung-Box** tại lag 10 & lag 20 cho cả White Noise và `AirPassengers`.

## 4. Evaluation Metrics
| Chỉ tiêu / Kiểm định | Công thức / Ngưỡng quyết định | Kết quả thực nghiệm (`run_id: 20261123_154119_4f8c41`) |
|---|---|---|
| **Thống kê mô tả gốc** | Mean, Std, Min, Median, Max | `Mean = 280.30`, `Std = 119.97`, `Min = 104 (11/1949)`, `Median = 265.50`, `Max = 622 (07/1960)` |
| **Biến thiên & Tăng trưởng** | $\text{CV} = \frac{\sigma}{\mu}\times 100\%$, tỷ lệ tăng trưởng | `CV = 42.80%`, `Skewness = 0.583`, `Growth Ratio = 3.86x`, biên độ tăng từ `44.0 (1949)` lên `232.0 (1960)` |
| **ACF tại các lag chính** | $\rho_1, \rho_{12}, \rho_{24}, \rho_{36}$ (ngưỡng $\pm 0.1633$) | `ACF(1) = 0.9480`, `ACF(12) = 0.7604`, `ACF(24) = 0.5322`, `ACF(36) = 0.3370` (`40/40` lag vượt ngưỡng) |
| **PACF có ý nghĩa** | Các lag vượt $\pm 0.1633$ | `6` lag (`1, 2, 9, 10, 11, 13`), mạnh nhất tại `lag 1 (0.9480)` và `lag 13 (-0.5397)` |
| **White Noise (1000 điểm)** | Tỷ lệ lag trong dải 95% ($\pm 0.0620$), Ljung-Box | `Mean = -0.0289`, `Std = 0.9892`, `97.5%` lag trong ngưỡng (chỉ `lag 1` vượt nhẹ), `LB p(10) = 0.3282`, `p(20) = 0.1443` |
| **Ljung-Box trên AirPassengers** | $H_0$: Chuỗi là nhiễu trắng ($\alpha = 0.05$) | `p(20) = 5.30e-292` ($\ll 0.05$) → bác bỏ hoàn toàn $H_0$, khẳng định tự tương quan cực mạnh |

## 5. Used data
- **Nguồn dữ liệu:** `data/raw/AirPassengers.csv` — lượng hành khách hàng không quốc tế hàng tháng (đơn vị: nghìn người), giai đoạn `01/1949 – 12/1960` (`144` quan sát, `0` giá trị thiếu, `0` mốc trùng lặp, kiểu `int64`).
- **Định hướng chia tập cho bước mô hình hóa (Lab 02):** Bảo toàn thứ tự thời gian (`shuffle=False`), tỷ lệ `80% Train (115 tháng: 01/1949 → 07/1958)` / `20% Test (29 tháng: 08/1958 → 12/1960)` nhằm ngăn chặn Data Leakage.
