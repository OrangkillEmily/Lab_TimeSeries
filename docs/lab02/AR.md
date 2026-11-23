# Thuật toán Tự hồi quy — AutoRegressive (`AR(p)`)

## 1. Purpose
Mô hình tự hồi quy **AR($p$)** biểu diễn giá trị hiện tại của chuỗi (sau sai phân $d$ lần để đạt tính dừng) như tổ hợp tuyến tính của $p$ giá trị quá khứ liền kề. Trong Lab 02, họ AR đóng vai trò **đường cơ sở (baseline) ngắn hạn** để kiểm chứng giới hạn của mô hình không có thành phần mùa vụ trên cả 2 bộ dữ liệu.

## 2. Input-Output
| Thành phần | Đặc tả chi tiết |
|---|---|
| **Input** | Chuỗi Train `y_train`, bậc $p \ge 1$, bậc sai phân $d$ (`d = 1`), cờ `check_stationarity=True` (báo `ValueError` nếu $d=0$ trên chuỗi không dừng). |
| **Module xử lý** | `src/lab02/lab02_models/ar_model.py` (`fit_ar_model`, `run_ar_experiment`). |
| **Output** | Từ điển kết quả gồm `model_name`, `model_family="AR"`, `order=(p,d,0)`, `scale_group="original_d1"`, `aic`, `bic`, `metrics` (`MAE`, `RMSE`, `MAPE (%)`, `R2`), `diagnostics` (Ljung-Box, Jarque-Bera, Het). |

## 3. How to do
1. **Phương trình toán học của `AR(p)` trên chuỗi sai phân $z_t = \nabla^d y_t$:**
   $$z_t = c + \sum_{i=1}^{p} \phi_i z_{t-i} + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{WN}(0, \sigma^2)$$
2. **Triển khai trong `ar_model.py`:** Dùng `ARIMA(train, order=(p, d, 0))` (qua `statsmodels` hoặc lõi dự phòng `scipy_tsa_compat.py`), kiểm tra ràng buộc `p >= 1` và kiểm định ADF nếu `d == 0` với `check_stationarity=True`.
3. **Dự báo & Đánh giá:** Dự báo đa bước trên tập Test (`len(test)`) và tính `MAE`, `RMSE`, `MAPE (%)`, `R²` + kiểm định phần dư Ljung-Box (lag 24).

## 4. Evaluation Metrics (Kết quả thực nghiệm thực tế)
| Dataset | Mô hình | AIC | BIC | MAE | RMSE | MAPE (%) | R² | Ljung-Box p | Kết luận |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Dataset 1** | `AR(1) [ARIMA(1,1,0)]` | `597.93` | `604.10` | `3.56` | `4.21` | `19.99%` | `-0.0358` | `0.0000` | Không đạt White Noise |
| **Dataset 1** | `AR(2) [ARIMA(2,1,0)]` | `593.58` | `602.84` | `3.50` | `4.15` | `19.29%` | `-0.0063` | `0.0000` | Không đạt White Noise |
| **Dataset 2** | `AR(1) [ARIMA(1,1,0)]` | `1068.54` | `1074.01` | `94.00` | `107.22` | `23.69%` | `-0.8832` | `0.0000` | Không đạt White Noise |
| **Dataset 2** | `AR(2) [ARIMA(2,1,0)]` | `1065.64` | `1073.85` | `76.24` | `88.31` | `18.76%` | `-0.2776` | `0.0000` | Không đạt White Noise |
| **Dataset 2** | `AR(4) [ARIMA(4,1,0)]` | `1059.67` | `1073.35` | `65.33` | `78.19` | `15.36%` | `-0.0016` | `0.0000` | Không đạt White Noise |

## 5. Used data
- Huấn luyện trên tập Train (`163` tháng DS1, `115` tháng DS2) và kiểm thử trên tập Test (`41` tháng DS1, `29` tháng DS2). Vì đường dự báo AR hội tụ nhanh về hằng số khi $h$ lớn, $R^2 < 0$ trên cả hai tập dữ liệu có mùa vụ mạnh.
