# HƯỚNG DẪN QUY TRÌNH TỪNG BƯỚC & BÀN GIAO DỰ ÁN (LAB04 TIME SERIES — STEPBYSTEP.MD)

> **Áp dụng cho:** Dự án `Lab04_Multivariate_TimeSeries` (`agents/lab04/StepbyStep.md`)  
> **Điểm khởi chạy chính:** `python run_lab04.py` & `python render_report.py`

## 1. Bảng Ánh xạ Chuyển đổi Kiến trúc từ `Lab04_ML-main` sang `Lab04_TimeSeries`

| Thành phần | `Lab04_ML-main` (MLP Regression — Giá nhà Boston) | `Lab04_TimeSeries` (Multivariate Time Series — Điện năng & Giao thông) | Lý do Chuyển đổi & Cải tiến Kỹ thuật |
|---|---|---|---|
| **Bài toán cốt lõi** | Hồi quy tĩnh trên dữ liệu bảng (`MEDV` từ 13 đặc trưng đô thị) | Dự báo chuỗi thời gian đa biến $\mathbf{X}_{t-24:t} \Rightarrow \text{load}_{t+1}$ (Yêu cầu 1–11) & Dự báo đa bước giao thông $15, 30, 60$ phút (Yêu cầu 12) | Mô hình hóa động lực hệ thống đa biến có phụ thuộc chéo và phụ thuộc thời gian ([Slide 3–4 Chương 4–5](https://www.statsmodels.org/stable/vector_ar.html)) |
| **Nguồn dữ liệu gốc** | `data/raw/boston_housing.csv` (`506 × 14`) | `data/raw/electricity_load.csv` ([UCI / Kaggle](https://www.kaggle.com/datasets/jacopoferretti/electricity-load-diagrams-2011-2014-feature-engin)) & `data/raw/pems-bay.h5` ([PEMS-BAY](https://drive.google.com/file/d/1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq/view?usp=drive_link)) + `data/raw/sensor_graph/adj_mx_bay.pkl` ([DCRNN Graph](https://github.com/liyaguang/dcrnn)) | Đáp ứng đầy đủ bài toán phụ tải điện + thời tiết (Mục 2) và bài toán nâng cao PEMS-BAY (Mục 12) |
| **Chia dữ liệu & Kiểm định** | `train_test_split` ngẫu nhiên (`80/20`, `random_state=42`) | **Chronological Split (`70% Train - 15% Val - 15% Test`, `shuffle=False`)** + **Temporal 5-Fold CV (`TimeSeriesSplit`)** + **Walk-Forward Backtesting** | Ngăn chặn rò rỉ tương lai, kiểm định độ ổn định qua 5 Folds và mô phỏng dự báo cuốn chiếu thực tế |
| **Kỹ thuật Đặc trưng** | Chỉ chuẩn hóa `StandardScaler` trên 13 cột gốc | Cửa sổ trễ $L=24\text{h}$ (`96` lag features cho ML, Tensor 3D `(N, 24, 4)` cho DL) + Rolling (`mean_6h`, `std_24h`, `load_trend`) + Đồ thị khuếch tán hai chiều | Giúp `XGBoost` nắm bắt xu hướng/biến động, `LSTM` học chu kỳ ngày/tuần và `DCRNN` học lan truyền không gian |
| **Hệ thống Thuật toán** | `MLP`, `Linear Regression`, `Decision Tree`, `Random Forest` | **Bài 1–11:** `VAR`, `XGBoost` ([Kaggle Ref](https://www.kaggle.com/code/davidandressanchez/xgboost-for-multivariate-ts)), `LSTM` ([MLM Ref](https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/)) — **Bài 12:** `DCRNN` ([arXiv:1707.01926](https://arxiv.org/abs/1707.01926)) | Đại diện đầy đủ 3 trường phái Thống kê cổ điển, Machine Learning, Deep Learning và Spatio-Temporal GNN |
| **Tăng tốc Phần cứng** | Chạy mặc định trên CPU | **Auto-GPU Hardware Dispatcher (`CUDA` / `MPS` / `CPU`)** + **Mixed Precision (`AMP FP16`)** + `XGBoost device="cuda"` | Tự động chuyển sang GPU khi khả dụng để tăng tốc huấn luyện `LSTM`, `DCRNN` và `5-Fold CV` |
| **Chỉ số Đánh giá** | `MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²` | **`MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²`** tính trên cả `5-Fold CV (Mean ± Std)`, `Hold-out Test (15%)`, `Walk-Forward` và 3 chân trời `15m, 30m, 60m` | Giữ nguyên bộ 5 chỉ số chuẩn của nhóm đồng thời mở rộng cho đánh giá đa tầng thời gian |
| **Tài liệu `docs/lab04/`** | Các file mô tả mô hình và báo cáo tổng hợp | Toàn bộ file **Phases** (`EDA`, `Preprocessing`, `Feature_Engineering`, `Training`, `Inference`, `Evaluation`) & **Algorithms** (`VAR`, `XGBoost`, `LSTM`, `DCRNN`) theo đúng **5 mục bắt buộc** | Chuẩn hóa 100% theo bố cục `1. Purpose`, `2. Input-Output`, `3. How to do`, `4. Evaluation metrics`, `5. Used data` |

## 2. Luồng Thực thi 7 Giai đoạn trong Pipeline (`run_lab04.py`)

```text
[0. HARDWARE & RUN INIT]   src/utils.py + src/tracking.py
       │                   Khởi tạo Run_ID, cố định Seed=42, tự động dò tìm GPU (CUDA AMP / MPS / CPU)
       ▼
[1. DATA LOADING]          src/data_loader.py
       │                   Nạp data/raw/electricity_load.csv (4 biến: load, temp, humidity, wind)
       │                   Nạp data/raw/pems-bay.h5 + data/raw/sensor_graph/adj_mx_bay.pkl (bài toán nâng cao Mục 12)
       ▼
[2. EDA & SEASONALITY]     src/eda.py
       │                   Kiểm tra Missing values, Outliers (IQR/Z-score), Seasonality (ngày 24h, tuần 168h)
       │                   Vẽ biểu đồ Load theo thời gian, Load vs Temperature, Correlation Matrix
       ▼
[3. PREPROCESS & K-FOLD]   src/preprocess.py
       │                   Chia Chronological Split (Train 70% - Val 15% - Test 15%, shuffle=False)
       │                   Khởi tạo 5-Fold TimeSeriesSplit (fit StandardScaler độc lập trong từng Fold)
       │                   Kiểm định tính dừng ADF cho VAR + Tạo cửa sổ trễ L=24 & Rolling (6h, 24h, trend)
       ▼
[4. MODEL TRAINING]        src/model.py + src/train.py
       │                   Huấn luyện VAR (chọn bậc trễ p theo AIC/BIC trên chuỗi dừng)
       │                   Huấn luyện XGBoost (max_depth, n_estimators, early_stopping trên Val, GPU hist)
       │                   Huấn luyện LSTM: LSTM(64) -> LSTM(32) -> Dense(1), MSE, Adam trên GPU/CPU
       │                   Huấn luyện DCRNN (DCGRU Encoder-Decoder) trên PEMS-BAY cho 3 mốc 15m, 30m, 60m
       ▼
[5. BACKTEST & EVALUATE]   src/evaluate.py + src/visualize.py
       │                   Chạy 5-Fold CV (xuất Mean ± Std) + Đánh giá Hold-out Test (15%)
       │                   Chạy Walk-Forward Backtesting: Train -> predict t+1 -> append -> predict t+2
       │                   Tính đủ 5 chỉ số MSE, RMSE, MAE, MAPE (%), R² & xuất bộ biểu đồ 300 DPI
       ▼
[6. AUTO REPORTING]        src/report.py + render_report.py
                           Tự động kết xuất LAB_04_REPORT.md, LAB_04_REPORT.html (Base64) & LAB_04_REPORT.pdf
                           Đồng bộ sang thư mục docs/, cập nhật docs/experiment_logs.csv & trả lời 6 câu hỏi Mục 11
```

## 3. Cấu trúc Cây Thư mục Chuẩn của Dự án

```text
Lab04_TimeSeries/
├── config.yaml                          # Cấu hình trung tâm (Auto-GPU, Data, 70/15/15, 5-Fold CV, VAR, XGB, LSTM, DCRNN)
├── run_lab04.py                         # Điểm khởi chạy CLI toàn trình (hỗ trợ --skip-eda, --skip-kfold, --skip-pems, --bundle-docs)
├── render_report.py                     # Bộ kết xuất báo cáo HTML tương tác (Base64 figures) và xuất PDF tự động
├── requirements.txt                     # Danh mục thư viện chuẩn (numpy, pandas, statsmodels, xgboost, torch, tensorflow, tables, h5py)
│
├── agents/
│   └── lab04/
│       ├── LAB04.md                     # Đặc tả bài toán, công thức toán học và yêu cầu thực hành Mục 1–12
│       ├── RULE.md                      # 8 quy tắc kiến trúc, chống Data Leakage, K-Fold, Auto-GPU, 5 metrics & chuẩn 5 mục docs/
│       └── StepbyStep.md                # Lộ trình thực thi từng bước, sơ đồ pipeline và ma trận nghiệm thu AC01–AC15
│
├── data/
│   ├── raw/
│   │   ├── electricity_load.csv         # Bộ dữ liệu phụ tải điện & thời tiết (UCI / Kaggle — Yêu cầu 1–11)
│   │   ├── pems-bay.h5                  # Bộ dữ liệu tốc độ giao thông California Highway Sensors (Yêu cầu 12)
├   ├   ├── sensor_graph/
│   │       └── adj_mx_bay.pkl               # Ma trận kề khoảng cách mạng lưới trạm cảm biến PEMS-BAY cho DCRNN
│   └── processed/                       # Lưu trữ các tensor cửa sổ trễ, dữ liệu chuẩn hóa và cấu trúc đồ thị đã xử lý
│
├── docs/
|    └── lab04/
│      ├── EDA_Phase.md                     # Tài liệu Giai đoạn Khám phá Dữ liệu (theo đúng 5 mục chuẩn)
│      ├── Preprocessing_Phase.md           # Tài liệu Giai đoạn Tiền xử lý, Chia 70/15/15 & 5-Fold CV (5 mục chuẩn)
│      ├── Feature_Engineering_Phase.md     # Tài liệu Giai đoạn Tạo cửa sổ trễ L=24 & Rolling Features (5 mục chuẩn)
│      ├── Training_Phase.md                # Tài liệu Giai đoạn Huấn luyện Mô hình & Auto-GPU (5 mục chuẩn)
│      ├── Inference_Phase.md               # Tài liệu Giai đoạn Dự báo & Walk-Forward Backtesting (5 mục chuẩn)
│      ├── Evaluation_Phase.md              # Tài liệu Giai đoạn Đánh giá 5 chỉ số & Trả lời 6 câu hỏi Mục 11 (5 mục chuẩn)
│      ├── VAR.md                           # Tài liệu Thuật toán Vector Autoregression (5 mục chuẩn)
│      ├── XGBoost.md                       # Tài liệu Thuật toán Extreme Gradient Boosting (5 mục chuẩn)
│      ├── LSTM.md                          # Tài liệu Thuật toán Long Short-Term Memory (5 mục chuẩn)
│      ├── DCRNN.md                         # Tài liệu Thuật toán Diffusion Convolutional RNN cho PEMS-BAY (5 mục chuẩn)
│      ├── Model_Comparison.md              # Đối sánh chuyên sâu giữa VAR, XGBoost, LSTM và DCRNN
│      ├── LAB_04_DESCRIPTION.md            # Tổng quan đề bài, kiến trúc hệ thống và tiêu chí nghiệm thu
│      ├── FILE_GUIDE.md                    # Giải thích 5W1H và Code Walkthrough cho từng file trong dự án
│      ├── EXECUTION_GUIDE.md               # Hướng dẫn cài đặt môi trường, kích hoạt GPU và chạy lệnh từng bước
│      ├── experiment_logs.csv              # Bảng nhật ký lịch sử các phiên chạy thực nghiệm (Run_ID)
│      ├── LAB_04_REPORT.md                 # Bản sao đồng bộ báo cáo kỹ thuật Markdown
│      ├── LAB_04_REPORT.html               # Bản sao đồng bộ báo cáo HTML tương tác
│      └── LAB_04_REPORT.pdf                # Bản sao đồng bộ báo cáo PDF chuẩn in ấn
│
├── src/
|    └── lab04/
│      ├── __init__.py                      # Khởi tạo gói mã nguồn Python
│      ├── utils.py                         # Nạp config.yaml, cố định Seed=42, Auto-GPU HardwareAccelerator
│      ├── tracking.py                      # Quản lý Run_ID, ghi logs/pipeline_<run_id>.log và progress_results/
│      ├── data_loader.py                   # Đọc electricity_load.csv, pems-bay.h5 và adj_mx_bay.pkl (Python 2/3 pickle safe)
│      ├── eda.py                           # Kiểm tra Missing, Outliers, phân rã Seasonality ngày/tuần, Correlation Matrix
│      ├── preprocess.py                    # Chia 70/15/15, 5-Fold TimeSeriesSplit, StandardScaler, ADF test, Lag & Rolling FE
│      ├── model.py                         # Định nghĩa kiến trúc VAR, XGBoost, LSTM(64->32->1) và DCRNN (DCGRU Seq2Seq)
│      ├── train.py                         # Điều phối huấn luyện trên GPU/CPU, EarlyStopping và lưu trọng số vào models/
│      ├── evaluate.py                      # Chạy 5-Fold CV, Walk-Forward Backtesting và tính MSE, RMSE, MAE, MAPE, R²
│      ├── visualize.py                     # Xuất toàn bộ biểu đồ 300 DPI vào reports/figures/ và reports/figures/<run_id>/
│      ├── report.py                        # Tổng hợp số liệu thực tế và gọi render_report.py xuất .md, .html, .pdf
│      └── pipeline.py                      # Master pipeline kết nối liền mạch toàn bộ các giai đoạn
│
├── models/
|    └── lab04/                              # Lưu trữ trọng số đã huấn luyện (*.pkl, *.pt / *.keras, training_metadata.json)
├── logs/ 
|    └── lab04/                               # Lưu trữ tệp nhật ký chi tiết theo từng phiên chạy (pipeline_<run_id>.log)
├── progress_results/
|    └── lab04/                    # Lưu trữ trạng thái JSON từng bước (progress_<run_id>.json)
├── notebooks/
|    └── lab04/
│      └── lab04.ipynb                      # Jupyter Notebook trình bày trực quan toàn bộ Yêu cầu 1–12
└── reports/
    ├── figures/  
    |    ├── lab04/  # Kho lưu trữ biểu đồ tập trung (kèm thư mục con theo từng Run_ID)
    |
    ├──lab04/  
       ├── LAB_04_REPORT.md                 # Báo cáo Markdown chính thức
       ├── LAB_04_REPORT.html               # Báo cáo HTML tương tác nhúng ảnh Base64
       ├── LAB_04_REPORT.pdf                # Báo cáo PDF xuất tự động
       ├── metrics/                         # Lưu trữ evaluation_latest.json, kfold_cv_results.csv, test_metrics_comparison.csv
       └── predictions/                     # Lưu trữ chuỗi dự đoán thực tế vs dự báo trên tập Test và Walk-Forward
```

## 4. Ma trận Tiêu chí Nghiệm thu Dự án (`Acceptance Criteria AC01–AC15`)

| Mã AC | Hạng mục Kiểm định | Điều kiện Nghiệm thu Kỹ thuật |
|---|---|---|
| **AC01** | Quản trị `agents/lab04/` | Có đủ 3 tệp `LAB04.md`, `RULE.md`, `StepbyStep.md` đặt đúng trong `agents/lab04/` |
| **AC02** | Nạp dữ liệu Điện năng & Thời tiết | Đọc thành công `data/raw/electricity_load.csv` với 4 biến `[load, temperature, humidity, wind_speed]` |
| **AC03** | Nạp dữ liệu Giao thông Nâng cao | Đọc thành công `data/raw/pems-bay.h5` và ma trận kề `data/raw/sensor_graph/adj_mx_bay.pkl` cho bài toán Mục 12 |
| **AC04** | Khám phá Dữ liệu (EDA — Mục 3) | Kiểm tra Missing, Outliers, Seasonality (ngày/tuần) và xuất 3 nhóm biểu đồ bắt buộc (Load theo thời gian, Load vs Temp, Correlation Matrix) |
| **AC05** | Chống Rò rỉ & Chia 70/15/15 (Mục 4) | Chia theo đúng thứ tự thời gian `Train 70% - Val 15% - Test 15%`, không shuffle, `StandardScaler` chỉ fit trên Train |
| **AC06** | Temporal K-Fold Cross-Validation | Triển khai `TimeSeriesSplit(n_splits=5)` với `In-Fold StandardScaler`, xuất bảng `Mean ± Std` qua 5 Folds |
| **AC07** | Cửa sổ Trễ & Rolling Features (Mục 5) | Tạo đúng cửa sổ trễ $L=24\text{h}$ và 3 nhóm đặc trưng `rolling_mean_6h`, `rolling_std_24h`, `load_trend` cho XGBoost |
| **AC08** | Mô hình 1 — VAR (Mục 6) | Thực hiện kiểm định ADF, sai phân nếu chuỗi không dừng, chọn bậc trễ theo AIC/BIC, dự báo vector 4 biến và đánh giá trên cột `load` |
| **AC09** | Mô hình 2 — XGBoost (Mục 7) | Đầu vào vector phẳng từ cửa sổ lag + rolling, tinh chỉnh `max_depth`, `n_estimators` và `early_stopping` trên tập Validation |
| **AC10** | Mô hình 3 — LSTM (Mục 8) | Đầu vào Tensor 3D `(batch, 24, 4)`, kiến trúc `LSTM(64) -> LSTM(32) -> Dense(1)`, Loss `MSE`, Optimizer `Adam` |
| **AC11** | Walk-Forward Backtesting (Mục 9) | Thực thi cuốn chiếu `Train -> predict t+1 -> thêm vào chuỗi -> predict t+2` cho cả 3 mô hình |
| **AC12** | Bộ 5 Chỉ số Đánh giá (Mục 10 & Yêu cầu 6) | Tính toán đầy đủ `MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²` trên đơn vị gốc và trả lời trọn vẹn 6 câu hỏi phân tích ở Mục 11 |
| **AC13** | Bài tập 12 — DCRNN trên PEMS-BAY | Dự báo đa bước $t+3, t+6, t+12$ (15, 30, 60 phút) bằng kiến trúc `DCRNN (DCGRU Seq2Seq)` với `adj_mx_bay.pkl` |
| **AC14** | Tự động Chuyển sang GPU (`Auto-GPU`) | Tự động nhận diện và huấn luyện trên GPU (`CUDA AMP FP16` / `MPS`) nếu máy có GPU, tự động fallback về CPU nếu không có |
| **AC15** | Chuẩn `docs/lab04/` 5 Mục & Auto Reports | Toàn bộ file Phase và Algorithm trong `docs/lab04/` tuân thủ đúng 5 mục; tự động xuất `report.pdf`, `report.html`, `report.md` và lưu logs |
