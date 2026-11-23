# Thuật toán ARIMA (`ARIMA(p, d, q)`)

## 1. Purpose
**ARIMA($p, d, q$)** kết hợp tự hồi quy AR($p$), sai phân bậc $d$ và trung bình trượt MA($q$) nhằm xử lý các chuỗi không dừng có xu hướng nhưng **không có thành phần mùa vụ tường minh**. Trong Lab 02, `ARIMA(1,1,1)` và `ARIMA(2,1,1)` là mốc đối chứng trung gian chứng minh rằng chỉ dùng sai phân thường ($d=1$) là **chưa đủ** với dữ liệu có chu kỳ 12 tháng.

## 2. Input-Output
| Thành phần | Đặc tả chi tiết |
|---|---|
| **Input** | Chuỗi Train `y_train`, bộ tham số `order=(p, d, q)` (`(1,1,1)` và `(2,1,1)`), chuỗi Test `y_test`. |
| **Module xử lý** | `src/lab02/lab02_models/arima_model.py` (`fit_arima_model`, `run_arima_experiment`), `src/lab02/modeling.py` (`fit_arima`, `residual_diagnostics`, `plot_arima_vs_sarima`). |
| **Output** | Tham số ước lượng, `AIC/BIC`, `MAE/RMSE/MAPE (%)/R2`, kiểm định Ljung-Box/JB/Het và top 3 thời điểm sai số lớn nhất trong `latest_run.json`. |

## 3. How to do
1. **Dạng toán tử trễ (Backshift Operator $B$):**
   $$\phi_p(B)(1 - B)^d y_t = \theta_q(B)\varepsilon_t$$
   trong đó $\phi_p(B) = 1 - \sum_{i=1}^{p}\phi_i B^i$ và $\theta_q(B) = 1 + \sum_{j=1}^{q}\theta_j B^j$.
2. **Nhược điểm trên chuỗi mùa vụ mạnh:** Khi dự báo đa bước ($h = 29..41$ tháng), đường dự báo `ARIMA(p,1,q)` nhanh chóng phẳng hóa thành đường thẳng ngang, bỏ lỡ toàn bộ đỉnh–đáy hàng năm (tháng 12/1 ở Retail Sales và tháng 7–8 ở AirPassengers).

## 4. Evaluation Metrics (Kết quả thực nghiệm thực tế)
| Dataset | Mô hình | AIC | BIC | MAE | RMSE | MAPE (%) | R² | Ljung-Box Q (p) | Điểm sai số lớn nhất |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Dataset 1** | `ARIMA(1,1,1)` | `582.47` | `591.73` | `3.67` | `4.62` | `18.04%` | `-0.2520` | `239.86 (0.0000)` | `01/2008`: thực `29.67`, dự báo `17.54` (`40.88%`) |
| **Dataset 1** | `ARIMA(2,1,1)` | `584.46` | `596.81` | `3.67` | `4.63` | `18.06%` | `-0.2543` | `239.98 (0.0000)` | `01/2008`: thực `29.67`, dự báo `17.53` (`40.90%`) |
| **Dataset 2** | `ARIMA(1,1,1)` | `1058.25` | `1066.45` | `85.25` | `97.50` | `21.29%` | `-0.5572` | `162.71 (0.0000)` | `11/1958`: thực `310`, dự báo `498.18` (`60.70%`) |
| **Dataset 2** | `ARIMA(2,1,1)` | `1058.65` | `1069.59` | `60.53` | `82.62` | `12.90%` | `-0.1182` | `186.90 (0.0000)` | `07/1960`: thực `622`, dự báo `408.26` (`34.36%`) |

## 5. Used data
- **Dataset 1** (`Train=163`, `Test=41`) và **Dataset 2** (`Train=115`, `Test=29`). Kết quả `Ljung-Box p = 0.0000` và $R^2 < 0$ khẳng định sự cần thiết phải nâng cấp lên **SARIMA** và **Log-SARIMA**.
