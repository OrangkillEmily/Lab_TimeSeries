# ĐỐI CHIẾU VÀ PHÂN TÍCH CHUYÊN SÂU CÁC MÔ HÌNH DỰ BÁO CHUỖI THỜI GIAN ĐA BIẾN (`Model_Comparison.md`)

> **Học phần:** Phân tích Chuỗi Thời gian (Time Series Analysis — UTH)  
> **Cơ sở lý thuyết:** `chapter4-5.pdf` & `practice 4.pdf`

## 1. Bảng Đối sánh Lý thuyết & Kiến trúc 4 Mô hình Triển khai trong Lab 04

Bảng dưới đây tổng hợp sự khác biệt cốt lõi về cấu trúc dữ liệu, cơ chế học phụ thuộc thời gian – không gian, tài nguyên tính toán và phạm vi áp dụng của 4 thuật toán trong dự án ([Slide 5–10 Chương 4–5](https://www.statsmodels.org/stable/vector_ar.html)):

| Tiêu chí So sánh | `VAR` (Statistical Baseline — Mục 6) | `XGBoost` (Machine Learning — Mục 7) | `LSTM` (Deep Learning — Mục 8) | `DCRNN` (Spatio-Temporal DL — Mục 12) |
|---|---|---|---|---|
| **Trường phái Mô hình** | Thống kê cổ điển tuyến tính ([Statsmodels VAR](https://www.statsmodels.org/stable/vector_ar.html)) | Gradient Boosting Decision Trees ([Kaggle XGBoost TS](https://www.kaggle.com/code/davidandressanchez/xgboost-for-multivariate-ts)) | Recurrent Neural Network có cổng nhớ ([MLM LSTM Guide](https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/)) | Graph Diffusion + GRU Seq2Seq ([Li et al., ICLR 2018](https://arxiv.org/abs/1707.01926)) |
| **Biểu diễn Dữ liệu Đầu vào** | Ma trận chuỗi dừng 2D $\tilde{\mathbf{X}}_t \in \mathbb{R}^{T \times 4}$ | Bảng phẳng 2D (`96` lags + Rolling 6h/24h + Trend) | Tensor 3D `(batch, 24, 4)` | Tensor 4D `(batch, 12, N_sensors, C)` + Đồ thị `adj_mx_bay.pkl` |
| **Giả định Tính dừng (Stationarity)** | **Bắt buộc** (phải kiểm định ADF & lấy sai phân) | Không yêu cầu (nhưng `load_trend` giúp nắm bắt xu hướng) | Không yêu cầu giả định dừng | Không yêu cầu giả định dừng |
| **Feature Engineering Thủ công** | Chọn bậc trễ $p^*$ theo AIC/BIC | **Bắt buộc, quyết định hiệu năng** (Lag 24h, Rolling mean 6h, std 24h, trend) | Gần như không cần (tự học biểu diễn từ Tensor 3D) | Tự học biểu diễn không gian – thời gian trên đồ thị |
| **Cơ chế Học Phụ thuộc Chéo** | Ma trận hệ số tuyến tính $\mathbf{A}_i \in \mathbb{R}^{4 \times 4}$ | Chia nhánh cây quyết định trên các biến trễ thời tiết & điện năng | Tổ hợp phi tuyến qua các tầng ẩn `LSTM(64) -> LSTM(32)` | Tích chập khuếch tán bước ngẫu nhiên hai chiều $(\mathbf{P}_f, \mathbf{P}_b)$ |
| **Khả năng Dự báo Đa bước (`Multi-step`)** | Tự hồi quy lặp vector $\hat{\mathbf{X}}_{t+h}$ (sai số tích lũy nhanh) | Khó, phải lặp cuốn chiếu hoặc huấn luyện nhiều mô hình rời | Tự nhiên qua trạng thái ẩn tuần tự | Thiết kế sẵn **Encoder–Decoder + Scheduled Sampling** (`15m, 30m, 60m`) |
| **Mở rộng khi Số biến $K$ lớn** | Kém (số tham số tăng bậc hai $p K^2$) | Khó khăn khi $K > 50$ ($K \times L$ quá lớn) | Trung bình – Tốt | **Rất tốt** (chia sẻ trọng số bộ lọc đồ thị trên `207–325` trạm) |
| **Khả năng Giải thích Mô hình** | Cao (hệ số hồi quy, kiểm định Granger) | Cao (`Gain Feature Importance`, SHAP) | Thấp (Hộp đen không gian ẩn) | Trung bình (trực quan hóa trọng số lan truyền đồ thị) |
| **Tài nguyên & Tăng tốc `Auto-GPU`** | CPU chạy cực nhanh (vài mili-giây) | Hỗ trợ cả CPU đa luồng và GPU (`tree_method="hist", device="cuda"`) | Cần GPU (`CUDA AMP FP16`) khi chạy nhiều epochs & 5-Fold CV | **Rất cần GPU (`CUDA AMP FP16`)** do tích chập đồ thị qua chuỗi |

## 2. Khung Bảng Tổng hợp 5 Chỉ số Đánh giá Thực nghiệm (`reports/metrics/lab04/`)

Khi thực thi lệnh `python run_lab04.py`, module `src/lab04/evaluate.py` và `src/lab04/report.py` tự động điền toàn bộ số liệu thực tế từ `reports/lab04/metrics/evaluation_latest.json` và `reports/lab04/metrics/kfold_cv_results.csv` vào 3 bảng chuẩn dưới đây trong `LAB_04_REPORT.md`, `LAB_04_REPORT.html` và `LAB_04_REPORT.pdf`:

### 2.1. Bảng 1 — Đối sánh trên Tập Kiểm tra Độc lập (`Hold-out Test Set 15%` — Yêu cầu 1–10)
- **Các cột chỉ số bắt buộc:** `Test MSE (↓)`, `Test RMSE (↓)`, `Test MAE (↓)`, `Test MAPE (%) (↓)`, `Test R² Score (↑)`, `Training Time (s)`, `Inference Latency (ms)`, `Hardware Device`.
- **Các mô hình đối sánh:** `VAR (Statistical Baseline)`, `XGBoost (Lag 24h + Rolling 6h/24h + Trend)`, `LSTM (64 -> 32 -> 1)`.

### 2.2. Bảng 2 — Đối sánh Kiểm định Chéo 5-Fold Thời gian (`5-Fold TimeSeriesSplit` — Yêu cầu bổ sung 4)
- Báo cáo kết quả từng Fold (`Fold 1`, `Fold 2`, `Fold 3`, `Fold 4`, `Fold 5`) và dòng tổng hợp **`Mean ± Std`** của cả 5 chỉ số (`MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²`) nhằm chứng minh tính ổn định qua các giai đoạn mùa vụ khác nhau.

### 2.3. Bảng 3 — Kết quả Walk-Forward Backtesting (Yêu cầu 9) & Dự báo Giao thông Đa bước `DCRNN` trên `PEMS-BAY` (Yêu cầu 12)
- Đối sánh sai số cuốn chiếu `Walk-Forward` của 3 mô hình phụ tải điện.
- Báo cáo chi tiết 5 chỉ số `MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²` của `DCRNN` tại 3 chân trời dự báo: **`15 phút (t+3)`**, **`30 phút (t+6)`** và **`60 phút (t+12)`**.
