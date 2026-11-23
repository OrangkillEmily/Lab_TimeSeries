# Thuật toán SARIMA & Log-SARIMA (`SARIMA(p,d,q)(P,D,Q)_s`)

## 1. Purpose
**SARIMA** mở rộng ARIMA bằng cách bổ sung thành phần tự hồi quy mùa vụ ($P$), sai phân mùa vụ ($D$) và trung bình trượt mùa vụ ($Q$) theo chu kỳ $s = 12$. Khi biên độ mùa vụ giãn nở theo xu hướng (cấu trúc nhân — Multiplicative), biến thể **Log-SARIMA** áp dụng biến đổi $\ln(Y_t)$ trước khi huấn luyện nhằm ổn định phương sai và chuyển cấu trúc nhân về cộng.

## 2. Input-Output
| Thành phần | Đặc tả chi tiết |
|---|---|
| **Input** | Chuỗi Train `y_train`, `order=(p,1,q)`, `seasonal_order=(P,1,Q,12)`, cờ `use_log_transform` (`False` cho SARIMA, `True` cho Log-SARIMA). |
| **Module xử lý** | `src/lab02/lab02_models/sarima_model.py` (`fit_sarima_model`, `run_sarima_experiment`), `src/lab02/modeling.py` (`fit_sarima`, `plot_log_transformation`, `plot_arima_vs_sarima`). |
| **Output** | Dự báo ngoài mẫu trên **thang đo gốc** (đã $\exp(\cdot)$ khi `use_log_transform=True`), `AIC/BIC` theo từng `Scale_Group`, kiểm định Ljung-Box/JB/Het và bảng phân tích loại trừ tham số (`param_elimination`). |

## 3. How to do
1. **Phương trình toán tử trễ SARIMA($p,d,q$)$\times$($P,D,Q$)$_s$:**
   $$\phi_p(B)\,\Phi_P(B^s)\,(1 - B)^d(1 - B^s)^D y_t = \theta_q(B)\,\Theta_Q(B^s)\,\varepsilon_t$$
2. **Biến đổi Log-SARIMA (`use_log_transform=True`):**
   - Biến đổi đầu vào: $z_t = \ln(y_t)$ (chỉ dùng trên tập Train).
   - Huấn luyện SARIMA trên $z_t$, dự báo $\hat{z}_{T+h}$ và chuyển ngược về thang đo gốc $\hat{y}_{T+h} = \exp(\hat{z}_{T+h})$ trước khi tính `MAE`, `RMSE`, `MAPE (%)`, `R²`.
3. **Phân tích loại trừ tham số (`param_elimination`):**
   - Trên cả hai dataset, thành phần **$Q = 1$ (Seasonal MA)** tạo ra bước nhảy chất lượng lớn nhất (DS1: giảm RMSE TB từ `2.54 → 1.95`; DS2 Log-SARIMA: giảm RMSE TB từ `34.15 → 18.40`, tăng $R^2$ từ `0.80 → 0.94`).

## 4. Evaluation Metrics (Các mô hình dẫn đầu thực tế)
| Dataset | Mô hình | Scale_Group | AIC | BIC | MAE | RMSE | MAPE (%) | R² | Ljung-Box p | Vai trò |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **DS1** | **`SARIMA(0,1,1)x(0,1,1)_12`** | `original_d1_D1_s12` | `233.85` | `242.59` | `1.45` | `1.84` | `8.21%` | `0.8012` | `0.1053` | **Tốt nhất đạt chuẩn WN + Hội tụ** |
| **DS1** | `SARIMA(2,1,3)x(0,1,1)_12` | `original_d1_D1_s12` | `223.07` | `243.35` | `1.48` | `1.87` | `8.38%` | `0.7956` | `0.9712` | AIC thấp nhất thang gốc |
| **DS1** | `Log-SARIMA(1,1,1)x(0,1,1)_12` | `log_d1_D1_s12` | `-396.57` | `-384.92` | `1.18` | `1.49` | `6.40%` | `0.8697` | `0.0170` | RMSE & MAE thấp nhất Test |
| **DS2** | `SARIMA(1,1,3)x(0,1,1)_12` | `original_d1_D1_s12` | `639.62` | `654.35` | `19.43` | `24.85` | `4.31%` | `0.8988` | `0.4105` | Tốt nhất thang gốc (Pass WN) |
| **DS2** | **`Log-SARIMA(1,1,1)x(0,1,1)_12`** | `log_d1_D1_s12` | `-322.17` | `-312.26` | `12.27` | **`15.02`** | `2.80%` | **`0.9630`** | **`0.9959`** | **Tốt nhất đạt chuẩn WN + Hội tụ** |
| **DS2** | **`Log-SARIMA(1,1,0)x(0,1,1)_12`** | `log_d1_D1_s12` | `-327.47` | `-320.00` | `12.35` | `15.12` | `2.81%` | `0.9626` | **`0.9970`** | **Tối giản tốt nhất (Best Model)** |

## 5. Used data
- **Dataset 1:** `Monthly Retail Sales` (`18` cấu hình SARIMA + `3` cấu hình Log-SARIMA trong lưới full).
- **Dataset 2:** `AirPassengers` (`24` cấu hình SARIMA + `24` cấu hình Log-SARIMA trong lưới full). Log-SARIMA giúp giảm RMSE trên AirPassengers từ `24.85` xuống `15.02–15.12` (giảm ~39% sai số) và nâng $R^2$ lên `> 0.96`.
