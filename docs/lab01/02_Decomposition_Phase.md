# 02 — Giai đoạn Phân rã Chuỗi Thời gian (`Decomposition_Phase`)

## 1. Purpose
Giai đoạn này tách chuỗi `AirPassengers` thành các thành phần cấu trúc độc lập (**Trend**, **Seasonal**, **Residual**) để trả lời 3 câu hỏi cốt lõi của Bài tập 3:
- So sánh định lượng giữa **Classical Additive** và **Classical Multiplicative** nhằm chứng minh cơ chế mùa vụ giãn nở theo xu hướng.
- Áp dụng phương pháp hiện đại **STL (Seasonal and Trend decomposition using Loess)** bền vững với nhiễu (`robust=True`) để tách 4 thành phần trực quan.
- Đo lường **Trend Strength ($F_T$)**, **Seasonal Strength ($F_S$)** và kiểm định tính nhiễu trắng của phần dư STL.

## 2. Input-Output
| Thành phần | Đặc tả chi tiết |
|---|---|
| **Input** | Chuỗi `Passengers` từ `load_airpassengers_dataset()` (`144` quan sát, `freq = "MS"`), cấu hình `lab01.analysis.decomposition_period = 12` trong `config.yaml`. |
| **Module xử lý** | `src/lab01/decomposition.py` (`compute_strength_metrics`, `run_decomposition_analysis`), `src/lab01/visualize.py` (`plot_decomposition`, `analyze_residuals`). |
| **Output (Data/JSON)** | `reports/lab01/metrics/decomposition_summary.json`, cập nhật nhánh `stages.decomposition` trong `progress_results/lab01/latest_run.json`. |
| **Output (Figures)** | `reports/lab01/figures/05_classical_decomposition_multiplicative.png`, `06_stl_decomposition.png`, `07_stl_residual_diagnostics.png`. |

## 3. How to do
1. **So sánh Phân rã Cổ điển (`seasonal_decompose`, `period = 12`):**
   - Mô hình Cộng (Additive): $Y_t = T_t + S_t + R_t$
   - Mô hình Nhân (Multiplicative): $Y_t = T_t \times S_t \times R_t$
   - Đối chiếu phương sai phần dư $\operatorname{Var}(R_t)$ và hình thái phân tán để xác định mô hình phù hợp.
2. **Phân rã STL (`STL(series, period=12, seasonal=13, robust=True)`):**
   - Dùng hồi quy cục bộ Loess cho phép biên độ mùa vụ biến thiên linh hoạt theo thời gian và ít bị kéo lệch bởi các điểm ngoại lai.
3. **Định lượng Độ mạnh Xu hướng & Mùa vụ (`compute_strength_metrics`):**
   $$F_T = \max\left(0,\, 1 - \frac{\operatorname{Var}(R_t)}{\operatorname{Var}(T_t + R_t)}\right), \qquad F_S = \max\left(0,\, 1 - \frac{\operatorname{Var}(R_t)}{\operatorname{Var}(S_t + R_t)}\right)$$
4. **Kiểm định Phần dư STL (`analyze_residuals`, `lags = 30`):** Vẽ Histogram kèm đường mật độ chuẩn $\mathcal{N}(\hat{\mu}, \hat{\sigma}^2)$, biểu đồ ACF 30 lags với ngưỡng 95% và chạy kiểm định Ljung-Box tại lag 10 và lag 20.

## 4. Evaluation Metrics
| Chỉ tiêu đánh giá | Ngưỡng / Diễn giải | Kết quả thực nghiệm (`run_id: 20261123_154119_4f8c41`) |
|---|---|---|
| **Var(Resid) — Classical Additive** | Càng nhỏ càng tốt | `371.2225` (biên độ phần dư phình lớn ở cuối chuỗi) |
| **Var(Resid) — Classical Multiplicative** | So sánh mô hình phù hợp | `0.001106` (phần dư dao động ổn định quanh `1.0` → khuyến nghị **Multiplicative**) |
| **Trend Strength ($F_T$ — STL)** | $[0, 1]$, càng gần 1 càng mạnh | `0.9957` (xu hướng tăng trưởng cực mạnh và ổn định) |
| **Seasonal Strength ($F_S$ — STL)** | $[0, 1]$, càng gần 1 càng mạnh | `0.9720` (mùa vụ chu kỳ 12 tháng rất sắc nét) |
| **Thống kê phần dư STL** | Kỳ vọng Mean $\approx 0$ | `Mean = -0.6682`, `Std = 7.3585`, `27/30` lags (`90.0%`) nằm trong dải tin cậy 95% ($\pm 0.1633$) |
| **Ljung-Box phần dư STL** | $p \ge 0.05$ đạt chuẩn nhiễu trắng | `p(10) = 0.0356`, `p(20) = 0.2567` (ở tầm nhìn 20 lag đạt chuẩn nhiễu trắng, phần dư chỉ còn dao động nhẹ ở lag ngắn do biên độ nhân) |

## 5. Used data
- **Đầu vào trực tiếp:** Chuỗi `AirPassengers` (`144` tháng, `01/1949 – 12/1960`, `period = 12`).
- **Ý nghĩa cho các giai đoạn sau:** Kết quả `Multiplicative` ($F_T = 0.9957$, $F_S = 0.9720$) là cơ sở trực tiếp cho việc áp dụng biến đổi $\ln(Y_t)$ trước khi phát hiện outlier ở **Bài tập 4** và mô hình hóa **Log-SARIMA** ở **Lab 02**.
