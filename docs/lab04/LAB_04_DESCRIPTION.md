# TỔNG QUAN ĐỀ BÀI, KIẾN TRÚC HỆ THỐNG VÀ TIÊU CHÍ NGHIỆM THU (`LAB_04_DESCRIPTION.md`)

> **Học phần:** Phân tích Chuỗi Thời gian (Time Series Analysis) – Trường Đại học Giao thông Vận tải TP.HCM (UTH)  
> **Lab:** 04 · **Chương:** 4–5 (Mô hình Dự đoán Chuỗi Thời gian Đa biến)  
> **Chủ đề:** Multivariate Time Series Forecasting — Electricity Load + Weather & PEMS-BAY Highway Traffic Forecasting  
> **Tài liệu gốc:** `practice 4.pdf` & `chapter4-5.pdf`

## 1. Giới thiệu chung

Bài thực hành số 04 xây dựng một hệ thống thực nghiệm hoàn chỉnh cho bài toán **Dự báo Chuỗi Thời gian Đa biến (Multivariate Time Series Forecasting)** theo kiến trúc phần mềm mô-đun hóa chuẩn công nghiệp:
- **Bài toán Chính (Yêu cầu 1–11):** Sử dụng bộ dữ liệu [Electricity Load Diagrams Dataset (UCI / Kaggle)](https://www.kaggle.com/datasets/jacopoferretti/electricity-load-diagrams-2011-2014-feature-engin) (`data/raw/electricity_load.csv`) gồm 4 biến đồng thời $\mathbf{X}_t = [\text{load}_t, \text{temp}_t, \text{humidity}_t, \text{wind}_t]$ với cửa sổ lịch sử $L = 24$ giờ để dự báo phụ tải điện năng ở bước tiếp theo ($\text{load}_{t+1}$). Hệ thống triển khai và so sánh 3 trường phái mô hình: **VAR** (thống kê cổ điển), **XGBoost** (Machine Learning dạng bảng có Rolling features) và **LSTM** (Deep Learning trên Tensor 3D).
- **Bài toán Nâng cao (Yêu cầu 12):** Sử dụng bộ dữ liệu chuẩn [PEMS-BAY (California Highway Sensors)](https://drive.google.com/file/d/1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq/view?usp=drive_link) (`data/raw/pems-bay.h5`) kết hợp ma trận kề khoảng cách mạng lưới đường bộ `data/raw/sensor_graph/adj_mx_bay.pkl` từ công trình [DCRNN (Li et al., ICLR 2018 — arXiv:1707.01926)](https://arxiv.org/abs/1707.01926) để dự báo đa bước tốc độ xe tại các mốc **15 phút ($t+3$)**, **30 phút ($t+6$)** và **60 phút ($t+12$)**.
- **Nâng cấp Kỹ thuật Toàn diện:**
  1. **Kiểm định Thời gian 3 Tầng:** Kết hợp phép chia theo thứ tự thời gian **Train 70% – Validation 15% – Test 15%** (`shuffle=False`), kiểm định chéo **5-Fold `TimeSeriesSplit`** (khớp `StandardScaler` độc lập trong từng Fold) và **Walk-Forward Backtesting**.
  2. **Tự động Chuyển đổi GPU (`Auto-GPU Acceleration`):** Tự động phát hiện GPU (`CUDA AMP FP16` / `MPS`) để tăng tốc huấn luyện `LSTM`, `DCRNN` và `XGBoost`, tự động lùi về CPU đa luồng nếu máy không có GPU.
  3. **Hệ thống 5 Chỉ số & Tự động Xuất Báo cáo:** Tính toán đồng bộ **`MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²`** trên đơn vị gốc và tự động kết xuất `LAB_04_REPORT.pdf`, `LAB_04_REPORT.html`, `LAB_04_REPORT.md` kèm lưu nhật ký `Run_ID`.

## 2. Danh mục Tài liệu Chuẩn 5 Mục trong `docs/lab04/`

Toàn bộ các file tài liệu Giai đoạn (`*_Phase.md`) và Thuật toán (`VAR.md`, `XGBoost.md`, `LSTM.md`, `DCRNN.md`) trong thư mục `docs/` đều tuân thủ nghiêm ngặt bố cục 5 phần (`1. Purpose`, `2. Input-Output`, `3. How to do`, `4. Evaluation metrics`, `5. Used data`):

| Nhóm Tài liệu | Tên Tệp trong `docs/lab04/` | Nội dung Trọng tâm |
|---|---|---|
| **Giai đoạn 1 (Mục 3)** | `docs/lab04/EDA_Phase.md` | Kiểm tra Missing values, Outliers, Seasonality (ngày 24h, tuần 168h), biểu đồ Load, Load vs Temp, Correlation Matrix |
| **Giai đoạn 2 (Mục 4 & K-Fold)** | `docs/lab04/Preprocessing_Phase.md` | Chuẩn hóa `StandardScaler`, chia `70/15/15` (`shuffle=False`), chia `5-Fold TimeSeriesSplit` và kiểm định ADF |
| **Giai đoạn 3 (Mục 5)** | `docs/lab04/Feature_Engineering_Phase.md` | Cửa sổ trễ $L=24\text{h}$, bộ đặc trưng Rolling (`mean_6h`, `std_24h`, `load_trend_24h`) cho XGBoost và đồ thị PEMS-BAY |
| **Giai đoạn 4 (Mục 6, 7, 8, 12)** | `docs/lab04/Training_Phase.md` | Huấn luyện đa mô hình với `Auto-GPU Dispatcher`, chọn bậc trễ AIC/BIC và giám sát `EarlyStopping` |
| **Giai đoạn 5 (Mục 9 & 12)** | `docs/lab04/Inference_Phase.md` | Dự báo bước $t+1$, quy trình Walk-Forward Backtesting cuốn chiếu và suy luận đa bước 15m, 30m, 60m |
| **Giai đoạn 6 (Mục 10 & 11)** | `docs/lab04/Evaluation_Phase.md` | Đánh giá 5 chỉ số (`MSE`, `RMSE`, `MAE`, `MAPE`, `R²`) và giải đáp chi tiết 6 câu hỏi phân tích ở Mục 11 |
| **Thuật toán 1 (Mục 6)** | `docs/lab04/VAR_Baseline.md` | Mô hình thống kê Vector Autoregression (`ADF test`, `Differencing`, `AIC/BIC`, dự báo vector 4 biến) |
| **Thuật toán 2 (Mục 7)** | `docs/lab04/Multivariate_XGBoost.md` | Mô hình Machine Learning Gradient Boosting trên vector phẳng Lag + Rolling + Feature Importance |
| **Thuật toán 3 (Mục 8)** | `docs/lab04/LSTM_Architecture.md` | Mô hình Deep Learning `LSTM(64) -> LSTM(32) -> Dense(1)` trên Tensor 3D `(batch, 24, 4)` |
| **Thuật toán 4 (Mục 12)** | `docs/lab04/DCRNN.md` | Mô hình Diffusion Convolutional RNN (`DCGRU Seq2Seq` + `Scheduled Sampling`) trên `pems-bay.h5` & `adj_mx_bay.pkl` |

## 3. Hướng dẫn Thực thi Nhanh (Quickstart)

```bash
# 1. Chạy toàn bộ Pipeline thực nghiệm tự động (Yêu cầu 1 -> Yêu cầu 12 + K-Fold CV + Auto Report PDF/HTML/MD)
python script/lab04/run_lab04.py

# 2. Chạy riêng phần kết xuất báo cáo HTML tương tác và PDF từ kết quả mới nhất
python script/lab04/render_report.py

# 3. Chạy kèm đồng bộ biểu đồ sang docs/images/ khi cần tách lẻ thư mục docs/
python script/lab04/run_lab04.py --bundle-docs
```
