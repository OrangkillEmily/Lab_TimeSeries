# 04 — Giai đoạn Huấn luyện, Chẩn đoán Phần dư & Đánh giá Mô hình (`Modeling_and_Evaluation_Phase`)

## 1. Purpose
Giai đoạn trọng tâm của Lab 02 thực thi huấn luyện có hệ thống toàn bộ 5 họ mô hình (**AR**, **MA**, **ARIMA**, **SARIMA**, **Log-SARIMA**), đánh giá khách quan trên **cùng tập Test và cùng thang đo gốc**, đồng thời kiểm định chặt chẽ giả định phần dư (White Noise — Ljung-Box) và xuất bộ hồ sơ tổng hợp (`model_summary` + `progress_results` + `experiment_logs.csv` + báo cáo HTML/PDF/MD).

## 2. Input-Output
| Thành phần | Đặc tả chi tiết |
|---|---|
| **Input** | Tập Train/Test (80/20) từ `src/lab02/data_loader.py` và danh sách mô hình ứng viên từ `src/lab02/lab02_models/candidate_registry.py` (`mode = standard | full | auto`). |
| **Module xử lý** | `src/lab02/lab02_models/{ar_model, ma_model, arima_model, sarima_model}.py`, `src/lab02/modeling.py`, `src/lab02/model_summary.py` (`ModelSummaryRegistry`), `src/lab02/pipeline.py`, `script/lab02/run_lab02_experiments.py`, `script/lab02/render_report.py`. |
| **Output (Model Summary)** | `reports/model_summary/lab02/{dataset1,dataset2}/{dataset}_model_summary.{csv,json,md}` và bản lưu trữ từng phiên trong `runs/`. |
| **Output (Pipeline & Reports)** | `progress_results/lab02/{dataset1,dataset2}/latest_run.json`, `docs/lab02/experiment_logs.csv`, `reports/lab02/LAB_02_DATASET{1,2}_REPORT.{html,pdf,md}`. |

## 3. How to do
1. **Huấn luyện đa họ mô hình:**
   - **Không mùa vụ (`original_d1`):** `AR(p)`, `MA(q)`, `ARIMA(p,1,q)` làm đường cơ sở chứng minh tác hại khi bỏ qua chu kỳ $s = 12$.
   - **Có mùa vụ thang gốc (`original_d1_D1_s12`):** Lưới `SARIMA(p,1,q)x(P,1,Q)_12` (18 mô hình DS1, 24 mô hình DS2).
   - **Có mùa vụ thang Log (`log_d1_D1_s12`):** Huấn luyện trên $\ln(Y_t)$ rồi biến đổi ngược $\hat{Y}_t = \exp(\widehat{\ln Y_t})$ để ổn định phương sai giãn nở.
2. **Đánh giá ngoài mẫu trên cùng thang đo gốc (`calculate_metrics`):**
   $$\text{MAE} = \frac{1}{n}\sum |y_t - \hat{y}_t|,\quad \text{RMSE} = \sqrt{\frac{1}{n}\sum (y_t - \hat{y}_t)^2},\quad \text{MAPE} = \frac{100}{n}\sum \left|\frac{y_t - \hat{y}_t}{y_t}\right|,\quad R^2 = 1 - \frac{\sum (y_t - \hat{y}_t)^2}{\sum (y_t - \bar{y})^2}$$
3. **Chẩn đoán phần dư (`residual_diagnostics`):** Kiểm định **Ljung-Box** tại lag 24 (điều chỉnh bậc tự do `model_df = k_params`, đạt chuẩn White Noise khi $p \ge 0.05$), **Jarque-Bera** (chuẩn tắc) và **Heteroskedasticity** (phương sai thay đổi).
4. **Nguyên tắc so sánh AIC/BIC hợp lệ:** Chỉ so sánh AIC/BIC trong **cùng một `Scale_Group`** (`original_d1`, `original_d1_D1_s12`, hoặc `log_d1_D1_s12`), tuyệt đối không so sánh chéo AIC giữa thang gốc và thang Log.

## 4. Evaluation Metrics (Kết quả chế độ `full` thực tế)

### 4.1. Dataset 1 — Monthly Retail Sales (`27/27` mô hình SUCCESS, `run_id: dataset1_20261123_154923_d2775a`)
| Mô hình đại diện | Họ / Scale_Group | AIC | BIC | MAE | RMSE | MAPE (%) | R² | Ljung-Box p | White Noise / Hội tụ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `AR(1) [ARIMA(1,1,0)]` | AR (`original_d1`) | `597.93` | `604.10` | `3.56` | `4.21` | `19.99%` | `-0.0358` | `0.0000` | Fail / True |
| `MA(1) [ARIMA(0,1,1)]` | MA (`original_d1`) | `587.92` | `594.09` | `3.46` | `4.15` | `18.39%` | `-0.0068` | `0.0000` | Fail / True |
| `ARIMA(1,1,1)` | ARIMA (`original_d1`) | `582.47` | `591.73` | `3.67` | `4.62` | `18.04%` | `-0.2520` | `0.0000` | Fail / True |
| **`SARIMA(0,1,1)x(0,1,1)_12`** | **SARIMA (`original_d1_D1_s12`)** | `233.85` | `242.59` | `1.45` | **`1.84`** | `8.21%` | **`0.8012`** | **`0.1053`** | **Pass / True (Tốt nhất đạt chuẩn WN)** |
| `SARIMA(2,1,3)x(0,1,1)_12` | SARIMA (`original_d1_D1_s12`) | **`223.07`** | `243.35` | `1.48` | `1.87` | `8.38%` | `0.7956` | `0.9712` | Pass / True (AIC thấp nhất nhóm gốc) |
| `Log-SARIMA(1,1,1)x(0,1,1)_12` | Log-SARIMA (`log_d1_D1_s12`) | `-396.57` | `-384.92` | **`1.18`** | **`1.49`** | `6.40%` | `0.8697` | `0.0170` | Fail / True (MAE/RMSE thấp nhất) |
| `Log-SARIMA(0,1,1)x(0,1,1)_12` | Log-SARIMA (`log_d1_D1_s12`) | `-395.54` | `-386.80` | `1.19` | **`1.49`** | **`6.38%`** | **`0.8698`** | `0.0039` | Fail / True (MAPE thấp nhất) |
| `Log-SARIMA(2,1,3)x(0,1,1)_12` | Log-SARIMA (`log_d1_D1_s12`) | `-402.78` | `-382.49` | `1.20` | `1.50` | `6.53%` | `0.8686` | `0.8259` | Pass / False (cảnh báo hội tụ) |

### 4.2. Dataset 2 — AirPassengers (`54/54` mô hình SUCCESS, `run_id: dataset2_20261123_154951_786eea`)
| Mô hình đại diện | Họ / Scale_Group | AIC | BIC | MAE | RMSE | MAPE (%) | R² | Ljung-Box p | White Noise / Hội tụ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `AR(1) [ARIMA(1,1,0)]` | AR (`original_d1`) | `1068.54` | `1074.01` | `94.00` | `107.22` | `23.69%` | `-0.8832` | `0.0000` | Fail / True |
| `MA(1) [ARIMA(0,1,1)]` | MA (`original_d1`) | `1063.65` | `1069.12` | `88.39` | `101.07` | `22.17%` | `-0.6734` | `0.0000` | Fail / True |
| `ARIMA(2,1,1)` | ARIMA (`original_d1`) | `1058.65` | `1069.59` | `60.53` | `82.62` | `12.90%` | `-0.1182` | `0.0000` | Fail / True |
| `SARIMA(1,1,3)x(0,1,1)_12` | SARIMA (`original_d1_D1_s12`) | `639.62` | `654.35` | `19.43` | `24.85` | `4.31%` | `0.8988` | `0.4105` | Pass / True (Tốt nhất thang gốc) |
| **`Log-SARIMA(1,1,1)x(0,1,1)_12`** | **Log-SARIMA (`log_d1_D1_s12`)** | `-322.17` | `-312.26` | `12.27` | **`15.02`** | `2.80%` | **`0.9630`** | **`0.9959`** | **Pass / True (Tốt nhất đạt chuẩn WN + Hội tụ)** |
| **`Log-SARIMA(1,1,0)x(0,1,1)_12`** | **Log-SARIMA (`log_d1_D1_s12`)** | **`-327.47`** | **`-320.00`** | `12.35` | `15.12` | `2.81%` | `0.9626` | **`0.9970`** | **Pass / True (Tối giản tốt nhất)** |
| `Log-SARIMA(1,1,3)x(0,1,1)_12` | Log-SARIMA (`log_d1_D1_s12`) | `-322.90` | `-308.17` | **`12.02`** | `14.67` | **`2.79%`** | **`0.9648`** | `0.9952` | Pass / False (MAE/MAPE thấp nhất) |
| `Log-SARIMA(0,1,0)x(1,1,1)_12` | Log-SARIMA (`log_d1_D1_s12`) | `-310.59` | `-303.13` | `12.06` | **`14.66`** | `2.80%` | **`0.9648`** | `0.0353` | Fail / True (RMSE thấp nhất) |

## 5. Used data
- **Dataset 1:** `Train = 163`, `Test = 41` (`Monthly Retail Sales.csv`).
- **Dataset 2:** `Train = 115`, `Test = 29` (`AirPassengers.csv`).
- **Tệp kết quả kiểm toán:** `reports/model_summary/lab02/{dataset1,dataset2}/` và `docs/lab02/experiment_logs.csv` (gộp toàn bộ 81 dòng kết quả của cả 2 dataset).
