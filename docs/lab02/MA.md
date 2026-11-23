# Thuật toán Trung bình Trượt — Moving Average (`MA(q)`)

## 1. Purpose
Mô hình **MA($q$)** mô tả chuỗi dừng như tổ hợp tuyến tính của nhiễu trắng hiện tại và $q$ cú sốc sai số quá khứ. Trong Lab 02, `MA(q)` được huấn luyện song song với `AR(p)` để đánh giá khả năng nắm bắt cú sốc ngắn hạn ở bậc sai phân $d = 1$ trước khi kết hợp thành ARIMA/SARIMA.

## 2. Input-Output
| Thành phần | Đặc tả chi tiết |
|---|---|
| **Input** | Chuỗi Train `y_train`, bậc $q \ge 1$ (báo `ValueError` nếu $q \le 0$), bậc sai phân $d = 1$. |
| **Module xử lý** | `src/lab02/lab02_models/ma_model.py` (`fit_ma_model`, `run_ma_experiment`). |
| **Output** | Hồ sơ thực nghiệm chứa `order=(0,d,q)`, `scale_group="original_d1"`, `aic`, `bic`, `metrics` trên tập Test và `diagnostics` phần dư. |

## 3. How to do
1. **Phương trình toán học của `MA(q)` trên chuỗi sai phân $z_t = \nabla^d y_t$:**
   $$z_t = \mu + \varepsilon_t + \sum_{j=1}^{q} \theta_j \varepsilon_{t-j}, \qquad \varepsilon_t \sim \mathcal{WN}(0, \sigma^2)$$
2. **Triển khai trong `ma_model.py`:** Khởi tạo `ARIMA(train, order=(0, d, q))`, xác thực đầu vào `q >= 1`, ước lượng tham số bằng Maximum Likelihood (MLE) và xuất thẻ tóm tắt `format_terminal_model_card`.

## 4. Evaluation Metrics (Kết quả thực nghiệm thực tế)
| Dataset | Mô hình | AIC | BIC | MAE | RMSE | MAPE (%) | R² | Ljung-Box p | Kết luận |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **Dataset 1** | `MA(1) [ARIMA(0,1,1)]` | `587.92` | `594.09` | `3.46` | `4.15` | `18.39%` | `-0.0068` | `0.0000` | Phần dư còn tự tương quan mạnh |
| **Dataset 1** | `MA(3) [ARIMA(0,1,3)]` | `584.61` | `596.96` | `3.64` | `4.59` | `17.96%` | `-0.2352` | `0.0000` | Phần dư còn tự tương quan mạnh |
| **Dataset 2** | `MA(1) [ARIMA(0,1,1)]` | `1063.65` | `1069.12` | `88.39` | `101.07` | `22.17%` | `-0.6734` | `0.0000` | Phần dư còn tự tương quan mạnh |

## 5. Used data
- Đánh giá trên tập Test của `Monthly Retail Sales` (`41` tháng) và `AirPassengers` (`29` tháng). Tương tự AR, MA thuần túy không thể mô phỏng chu kỳ 12 tháng nên `Ljung-Box p = 0.0000` và $R^2 < 0$.
