# 03 — Giai đoạn Kiểm tra & Xử lý Chất lượng Dữ liệu (`Data_Quality_Phase`)

## 1. Purpose
Tập `AirPassengers` gốc hoàn toàn sạch (`0` NaN, không có outlier cực đoan). Giai đoạn này (Bài tập 4) chủ động tạo một kịch bản lỗi có kiểm soát (biết trước đáp án gốc `ts_true`) nhằm đánh giá khách quan:
- So sánh hiệu quả điền khuyết giữa **Forward-fill** và **Time Interpolation** trên các khoảng thiếu ngắn và khoảng thiếu liên tiếp 3 tháng.
- So sánh định lượng **4 phương pháp phát hiện Outlier** (Rolling 3σ, Z-score toàn cục, IQR toàn cục và Z-score trên phần dư STL thang Log).
- Đánh giá sai số khôi phục toàn diện sau khi thay thế outlier bằng NaN và nội suy lại thành chuỗi sạch `ts_clean`.

## 2. Input-Output
| Thành phần | Đặc tả chi tiết |
|---|---|
| **Input** | Chuỗi sạch `ts_true` (`AirPassengers`, `144` quan sát) và các tham số trong `config.yaml`: `random_seed = 42`, `missing_simulation_count = 8`, `rolling_window = 12`, `zscore_threshold = 3.0`, `iqr_multiplier = 1.5`. |
| **Module xử lý** | `src/lab01/quality.py` (`simulate_dirty_series`, `_score_outlier_method`, `run_quality_pipeline`), `src/lab01/visualize.py` (`evaluate_and_plot_imputation`, `detect_and_plot_rolling_outliers`, `detect_and_plot_stl_outliers`, `plot_before_after_cleaning`, `plot_residuals_comparison`). |
| **Output (Data/JSON)** | `data/processed/lab01/airpassengers_quality_cleaned.csv`, `reports/lab01/metrics/data_quality_summary.json`, nhánh `stages.data_quality` trong `progress_results/lab01/latest_run.json`. |
| **Output (Figures)** | `08_missing_imputation_eval.png`, `09_rolling_outliers_detection.png`, `10_stl_log_outliers_detection.png`, `11_before_after_cleaning.png`, `12_residuals_comparison.png` trong `reports/lab01/figures/`. |

## 3. How to do
1. **Giả lập chuỗi bẩn (`simulate_dirty_series`, `seed = 42`):**
   - **Chèn 3 Outlier có chủ đích:** `1952-08-01` ($\times 1.6$), `1955-02-01` ($\times 0.5$), `1958-11-01` ($\times 1.7$).
   - **Chèn 11 điểm thiếu NaN (`7.64%`):** Gồm 1 khoảng thiếu liên tiếp 3 tháng (`1955-05` → `1955-07`) và `8` điểm ngẫu nhiên không sát biên/outlier (`1949-12`, `1953-11`, `1956-09`, `1957-06`, `1957-11`, `1959-04`, `1960-08`, `1960-10`).
2. **Xử lý & Đánh giá Missing (`evaluate_and_plot_imputation`):** So sánh `ffill()` và `interpolate(method="time")` trên đúng `11` mốc thiếu bằng `MAE`, `RMSE`, `MAPE (%)`, đồng thời phóng to cửa sổ `03/1955 – 12/1955`.
3. **Phát hiện Outlier theo 4 phương pháp:**
   - **Rolling 3σ (`window = 12`, `shift(1)` chống rò rỉ):** $|y_t - \mu_{t-1}^{(12)}| > 3\,\sigma_{t-1}^{(12)}$.
   - **Z-score toàn cục:** $|(y_t - \bar{y})/s| > 3.0$.
   - **IQR toàn cục:** $y_t < Q_1 - 1.5\,\text{IQR}$ hoặc $y_t > Q_3 + 1.5\,\text{IQR}$.
   - **STL trên thang Log (`STL(np.log(ts_filled), period=12)`):** Tính Z-score trên phần dư $r_t^{\log}$ với ngưỡng $|z_t| > 3.0$.
4. **Làm sạch toàn diện:** Gán `NaN` tại các mốc outlier do STL phát hiện (`3` điểm) rồi nội suy lại bằng `interpolate(method="time")` để tạo `ts_clean`, lưu bảng 5 cột (`Original_True`, `Dirty_Input`, `Forward_Fill`, `Time_Interpolation`, `Cleaned_Final`) ra `airpassengers_quality_cleaned.csv`.

## 4. Evaluation Metrics

### 4.1. So sánh phương pháp điền khuyết (trên 11 điểm NaN)
| Phương pháp Imputation | MAE | RMSE | MAPE (%) | NaN còn lại | Nhận xét |
|---|---:|---:|---:|---:|---|
| **Forward-fill (`ffill`)** | `38.0909` | `46.3357` | `11.7501%` | `0` | Kéo ngang giá trị cũ, phá vỡ xu hướng & mùa vụ tại khoảng thiếu 3 tháng |
| **Time Interpolation** | **`19.4069`** | **`23.4042`** | **`6.2623%`** | `0` | Giảm ~49% RMSE so với Forward-fill; được chọn làm `ts_filled` |

### 4.2. So sánh 4 phương pháp phát hiện Outlier (đáp án: 3 outlier)
| Phương pháp | Phát hiện | TP | FP | FN | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Z-score toàn cục (`|z| > 3`)** | `0` | `0` | `0` | `3` | `0.0000` | `0.0000` | `0.0000` |
| **IQR toàn cục (`1.5×IQR`)** | `0` | `0` | `0` | `3` | `0.0000` | `0.0000` | `0.0000` |
| **Rolling mean ± 3σ (`w = 12`)** | `2` | `2` | `0` | `1` | `1.0000` | `0.6667` | `0.8000` |
| **STL Log Residual Z-score** | **`3`** | **`3`** | **`0`** | **`0`** | **`1.0000`** | **`1.0000`** | **`1.0000`** |

### 4.3. Sai số khôi phục cuối cùng (trên toàn bộ 14 điểm bị làm hỏng: 11 NaN + 3 Outlier)
- **Chuỗi sạch `ts_clean`:** `MAE = 21.1379`, `RMSE = 24.5704`, `MAPE = 7.1345%`, `missing_remaining_after_clean = 0`.

## 5. Used data
- **Chuỗi gốc làm chuẩn đối chiếu:** `AirPassengers` (`144` quan sát).
- **Dữ liệu sau làm sạch:** `data/processed/lab01/airpassengers_quality_cleaned.csv` chứa đồng thời cả 5 chuỗi từ đầu vào bẩn tới kết quả sạch cuối cùng để phục vụ kiểm toán tái lập.
