# 02 — Giai đoạn Kiểm định Tính dừng & Sai phân (`Stationarity_ADF_Phase`)

## 1. Purpose
Các mô hình họ AR / MA / ARIMA / SARIMA yêu cầu thành phần nội tại phải đạt **tính dừng yếu (Weak Stationarity)** — kỳ vọng và phương sai không đổi theo thời gian, hiệp phương sai chỉ phụ thuộc độ trễ. Giai đoạn này sử dụng **kiểm định Augmented Dickey-Fuller (ADF)** trên **riêng tập Train** (tránh Data Leakage) để xác định chính xác bậc sai phân thường ($d$) và bậc sai phân mùa vụ ($D$, chu kỳ $s = 12$).

## 2. Input-Output
| Thành phần | Đặc tả chi tiết |
|---|---|
| **Input** | Tập Train của Dataset 1 (`163` tháng) và Dataset 2 (`115` tháng); chu kỳ mùa vụ `seasonal_period = 12` từ `config.yaml`. |
| **Module xử lý** | `src/lab02/stats.py` (`perform_adf_test`), hỗ trợ tương thích `src/lab02/scipy_tsa_compat.py` khi thiếu `statsmodels`. |
| **Output (Data/JSON)** | Nhánh `adf` (`raw`, `diff_1`, `diff_1_12`) trong `progress_results/lab02/{dataset1,dataset2}/latest_run.json` và `logs/lab02/`. |
| **Output (Figures)** | Biểu đồ đối chiếu chuỗi gốc vs `Diff(d=1)` vs `Diff(d=1, D=1, s=12)` cho từng dataset. |

## 3. How to do
1. **Giả thuyết kiểm định ADF (mức ý nghĩa $\alpha = 0.05$):**
   - $H_0$: Chuỗi có nghiệm đơn vị (Non-stationary — **không dừng**).
   - $H_1$: Chuỗi không có nghiệm đơn vị (Stationary — **đã dừng**).
   - Phương trình hồi quy ADF:
     $$\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta y_{t-i} + \varepsilon_t$$
   - Bác bỏ $H_0$ khi $p\text{-value} < 0.05$ (tương đương $\text{ADF Statistic} < \text{Critical Value 5\%}$).
2. **Quy trình kiểm định 3 bậc tuần tự trên tập Train:**
   - **Bước 1 — Chuỗi gốc (`raw`):** Xác nhận chuỗi chưa dừng.
   - **Bước 2 — Sai phân bậc 1 (`diff_1`: $\nabla y_t = y_t - y_{t-1}$):** Khử xu hướng tuyến tính; kiểm tra xem còn nghiệm đơn vị mùa vụ không.
   - **Bước 3 — Sai phân kết hợp (`diff_1_12`: $\nabla \nabla_{12} y_t = (1-B)(1-B^{12})y_t$):** Khử đồng thời xu hướng và tính mùa vụ 12 tháng.

## 4. Evaluation Metrics
| Chuỗi kiểm định (Tập Train) | Dataset 1 — ADF Stat / p-value / CV(5%) | Dataset 1 — Kết luận | Dataset 2 — ADF Stat / p-value / CV(5%) | Dataset 2 — Kết luận |
|---|---|---|---|---|
| **Train Raw (`d=0, D=0`)** | `2.2309` / `0.9989` / `-2.8813` (`lags=14`) | Không dừng | `-0.3569` / `0.9171` / `-2.8906` (`lags=13`) | Không dừng |
| **Train `Diff(d=1)`** | `-1.7061` / `0.4280` / `-2.8813` (`lags=13`) | Vẫn không dừng | `-2.5396` / `0.1061` / `-2.8909` (`lags=13`) | Vẫn không dừng |
| **Train `Diff(d=1, D=1, s=12)`** | **`-8.7879` / `2.28e-14`** / `-2.8817` (`lags=4`) | **ĐÃ DỪNG ($p \ll 0.05$)** | **`-12.1763` / `1.38e-22`** / `-2.8906` (`lags=0`) | **ĐÃ DỪNG ($p \ll 0.05$)** |

## 5. Used data
- **Dataset 1 Train:** `163` quan sát → sau `Diff(d=1)` còn `162` → sau `Diff(d=1, D=1, s=12)` còn `150` quan sát.
- **Dataset 2 Train:** `115` quan sát → sau `Diff(d=1)` còn `114` → sau `Diff(d=1, D=1, s=12)` còn `102` quan sát.
- **Kết luận cho pha nhận diện:** Cả hai tập dữ liệu đều bắt buộc dùng **$d = 1, D = 1, s = 12$** trong họ SARIMA / Log-SARIMA (và $d = 1$ để so sánh cơ sở ở họ AR / MA / ARIMA không mùa vụ).
