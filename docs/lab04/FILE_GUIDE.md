# HƯỚNG DẪN KIẾN TRÚC TỆP TIN TOÀN DIỆN (`FILE_GUIDE.md` — 5W1H & CODE WALKTHROUGH)

> **Dự án:** `LAB04_Multivariate_TimeSeries` (Phân tích Chuỗi Thời gian — UTH)  
> **Phương pháp mô tả:** Khung phân tích **7 phần (`WHAT — WHY — WHERE — WHICH — WHEN — HOW — CODE WALKTHROUGH`)** đồng bộ từ chuẩn `Lab04_ML-main`

## MỤC LỤC HỆ THỐNG TỆP TIN
1. **Nhóm Tệp Gốc (Root Files):** `config.yaml`, `requirements.txt`, `run_lab04.py`, `render_report.py`.
2. **Nhóm Quản trị Quy tắc (`agents/lab04/`):** `LAB04.md`, `RULE.md`, `StepbyStep.md`.
3. **Nhóm Tài liệu Học thuật (`docs/`):** 6 tệp `*_Phase.md`, 4 tệp thuật toán (`VAR.md`, `XGBoost.md`, `LSTM.md`, `DCRNN.md`), `Model_Comparison.md`, `LAB_04_DESCRIPTION.md`, `FILE_GUIDE.md`, `EXECUTION_GUIDE.md`.
4. **Nhóm Mã nguồn Cốt lõi (`src/`):** `__init__.py`, `utils.py`, `tracking.py`, `data_loader.py`, `eda.py`, `preprocess.py`, `model.py`, `train.py`, `evaluate.py`, `visualize.py`, `report.py`, `pipeline.py`.
5. **Nhóm Dữ liệu (`data/raw/` & `data/processed/`):** `electricity_load.csv` ([UCI / Kaggle](https://www.kaggle.com/datasets/jacopoferretti/electricity-load-diagrams-2011-2014-feature-engin)), `pems-bay.h5` ([PEMS-BAY](https://drive.google.com/file/d/1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq/view?usp=drive_link)), `adj_mx_bay.pkl` ([DCRNN GitHub](https://github.com/liyaguang/dcrnn)).

## PHẦN I — NHÓM TỆP GỐC & QUẢN TRỊ (`ROOT` & `agents/lab04/`)

### 1. `config.yaml`
1. **WHAT — File này là gì?** Tệp cấu hình trung tâm định dạng YAML lưu trữ toàn bộ siêu tham số phần cứng (`Auto-GPU`), đường dẫn dữ liệu, tỷ lệ chia `70/15/15`, cấu hình `5-Fold TimeSeriesSplit`, cửa sổ trễ $L=24$, đặc trưng trượt và thông số của 4 mô hình (`VAR`, `XGBoost`, `LSTM`, `DCRNN`).
2. **WHY — Tại sao cần file này?** Tách biệt hoàn toàn siêu tham số khỏi mã nguồn Python, đảm bảo tính tái lập (`Reproducibility`) và giúp điều chỉnh thực nghiệm tại một nơi duy nhất.
3. **WHERE — File nằm ở đâu?** Tại thư mục gốc `Lab04_TimeSeries/config.yaml`.
4. **WHICH — File này sử dụng những thành phần nào?** Được đọc bởi `yaml.safe_load` trong `src/utils.py -> load_config()`.
5. **WHEN — Khi nào file được đọc?** Ngay khi khởi chạy `run_lab04.py`, `render_report.py` hoặc bất kỳ module nào trong `src/`.
6. **HOW — File hoạt động như thế nào?** Tổ chức thành 8 khối khóa phân cấp: `project`, `hardware`, `electricity_data`, `preprocessing`, `splitting_and_validation`, `feature_engineering`, `var_model` / `xgboost_model` / `lstm_model`, `pems_bay_advanced`, `paths`.
7. **CODE WALKTHROUGH:** Khối `hardware` cho phép đặt `device: "auto"` và `use_mixed_precision: true` để toàn bộ hệ thống tự động nhận diện GPU mà không cần sửa code.

### 2. `agents/lab04/LAB04.md`, `agents/lab04/RULE.md`, `agents/lab04/StepbyStep.md`
1. **WHAT:** Bộ 3 tài liệu đặc tả đề bài (`LAB04.md`), quy tắc kiểm soát chất lượng & chống rò rỉ dữ liệu (`RULE.md`), và lộ trình thực thi từng bước (`StepbyStep.md`).
2. **WHY:** Đóng vai trò là "bản thiết kế gốc" định hướng cho toàn bộ thành viên trong nhóm và đảm bảo mọi tệp code/báo cáo tuân thủ đúng yêu cầu của giảng viên.
3. **WHERE:** Trong thư mục `agents/lab04/`.
4. **WHICH:** Kết nối trực tiếp yêu cầu của `practice 4.pdf`, lý thuyết `chapter4-5.pdf` và kiến trúc mẫu `Lab04_ML-main`.
5. **WHEN:** Được thiết lập và kiểm duyệt ở pha **Define – Design** trước khi viết code thực thi.
6. **HOW:** Quy định rõ 8 quy tắc kỹ thuật và 15 tiêu chí nghiệm thu (`AC01–AC15`).
7. **CODE WALKTHROUGH:** Mọi kiểm thử tự động trong `src/lab04/preprocess.py` và `src/lab04/report.py` đều đối chiếu trực tiếp với `AC01–AC15` trong `StepbyStep.md`.

## PHẦN II — NHÓM MÃ NGUỒN CỐT LÕI (`src/lab04`)

### 3. `src/lab04/utils.py` (Tiện ích Hệ thống & Điều phối `Auto-GPU`)
1. **WHAT:** Module tiện ích nền tảng cung cấp hàm đọc YAML/JSON, cố định hạt giống ngẫu nhiên `set_global_seeds(42)` và lớp điều phối phần cứng `HardwareAccelerator`.
2. **WHY:** Giải quyết bài toán mô hình Deep Learning (`LSTM`, `DCRNN`) quá nặng trên CPU bằng cách tự động phát hiện và chuyển sang GPU (`CUDA` / `MPS`), đồng thời hỗ trợ tự động lùi về CPU nếu máy không có GPU.
3. **WHERE:** `src/lab04/utils.py`.
4. **WHICH:** Sử dụng `torch`, `os`, `random`, `numpy`, `yaml`, `json`.
5. **WHEN:** Được gọi đầu tiên trong `src/lab04/pipeline.py` và `src/lab04/train.py`.
6. **HOW:** Kiểm tra `torch.cuda.is_available()`, cấu hình `cudnn.benchmark = True`, trả về đối tượng `device`, cờ `amp_enabled` và tham số `xgb_device = "cuda"` hoặc `"cpu"`.
7. **CODE WALKTHROUGH:** Hàm `clear_hardware_memory()` gọi `gc.collect()` và `torch.cuda.empty_cache()` sau mỗi Fold của `5-Fold CV` để giữ VRAM luôn thông thoáng.

### 4. `src/lab04/data_loader.py` (Nạp Dữ liệu Kép `electricity_load.csv`, `pems-bay.h5` & `adj_mx_bay.pkl`)
1. **WHAT:** Module nạp và kiểm định tính toàn vẹn của cả 2 bộ dữ liệu thực nghiệm.
2. **WHY:** Đảm bảo đọc đúng 4 biến `[load, temperature, humidity, wind_speed]` cho Yêu cầu 1–11 và xử lý an toàn định dạng HDF5 (`.h5`) cùng tệp đồ thị Python 2/3 Pickle (`adj_mx_bay.pkl`) cho Yêu cầu 12.
3. **WHERE:** `src/lab04/data_loader.py`.
4. **WHICH:** Sử dụng `pandas`, `numpy`, `h5py` / `tables`, `pickle`.
5. **WHEN:** Thực thi tại Bước 1 (`DATA_LOADING`) của Pipeline.
6. **HOW:** Khi đọc `data/raw/sensor_graph/adj_mx_bay.pkl`, hàm dùng `pickle.load(f, encoding="latin1")` để tránh lỗi `UnpicklingError` giữa Python 2 và Python 3 ([DCRNN Issue #36](https://github.com/liyaguang/DCRNN/issues/36)), sau đó đồng bộ thứ tự cột `sensor_ids` với `pems-bay.h5`.
7. **CODE WALKTHROUGH:** Nếu tên cột trong `electricity_load.csv` viết hoa hoặc dùng tên viết tắt (`temp`, `hum`, `wnd_spd`), bộ ánh xạ `column_aliases` từ `config.yaml` sẽ tự động chuẩn hóa về đúng 4 tên cột chuẩn.

### 5. `src/lab04/eda.py` (Khám phá Dữ liệu, Mùa vụ & Tương quan — Mục 3)
1. **WHAT:** Module phân tích thống kê mô tả, kiểm tra Missing values, Outliers, phân rã chu kỳ ngày/tuần và tính ma trận tương quan.
2. **WHY:** Xác định các biến thời tiết ảnh hưởng như thế nào đến phụ tải điện trước khi xây dựng mô hình.
3. **WHERE:** `src/lab04/eda.py`.
4. **WHICH:** Sử dụng `pandas`, `numpy`, `scipy.stats`, `statsmodels.tsa.seasonal`.
5. **WHEN:** Thực thi tại Bước 2 (`EDA`) của Pipeline.
6. **HOW:** Xuất bảng thống kê ra `reports/lab04/metrics/eda_descriptive_statistics.csv`, `eda_correlation_matrix.csv` và gọi `src/lab04/visualize.py` sinh 4 biểu đồ EDA.
7. **CODE WALKTHROUGH:** Tính thêm tương quan chéo theo độ trễ $r(\tau)$ với $\tau \in \{0..24\}$ giờ để chứng minh độ trễ tác động khác nhau (*Lag heterogeneity*) giữa nhiệt độ và độ ẩm đối với phụ tải điện.

### 6. `src/lab04/preprocess.py` (Chia `70/15/15`, `5-Fold TimeSeriesSplit`, `StandardScaler` & Feature Engineering — Mục 4 & 5)
1. **WHAT:** Module tiền xử lý chống rò rỉ dữ liệu, chuẩn hóa `StandardScaler`, kiểm định ADF và tạo cửa sổ trễ $L=24$ cùng Rolling features.
2. **WHY:** Cung cấp dữ liệu sạch, không rò rỉ tương lai cho cả 3 mô hình chính và 5-Fold CV.
3. **WHERE:** `src/lab04/preprocess.py`.
4. **WHICH:** Sử dụng `sklearn.preprocessing.StandardScaler`, `sklearn.model_selection.TimeSeriesSplit`, `statsmodels.tsa.stattools.adfuller`.
5. **WHEN:** Thực thi tại Bước 3 (`PREPROCESSING`) của Pipeline.
6. **HOW:** Chia `Train 70% - Val 15% - Test 15%` (`shuffle=False`), khớp `StandardScaler` chỉ trên Train, tạo 5 Folds mở rộng theo thời gian với `In-Fold StandardScaler`, và xuất song song dữ liệu 2D cho `XGBoost` cùng Tensor 3D `(N, 24, 4)` cho `LSTM`.
7. **CODE WALKTHROUGH:** Các cửa sổ trượt ở ranh giới giữa các tập được xử lý cẩn trọng để nhãn mục tiêu $y_{t+1}$ không bao giờ nằm lẫn sang tập khác.

### 7. `src/lab04/model.py` & `src/lab04/train.py` (Kiến trúc & Huấn luyện `VAR`, `XGBoost`, `LSTM`, `DCRNN`)
1. **WHAT:** Định nghĩa kiến trúc toán học và điều phối huấn luyện cho cả 4 thuật toán.
2. **WHY:** Gom toàn bộ logic khởi tạo mô hình vào `src/lab04/model.py` và vòng lặp huấn luyện vào `src/lab04/train.py` để code gọn, dễ bảo trì và tái sử dụng cho cả Hold-out lẫn 5-Fold CV.
3. **WHERE:** `src/lab04/model.py` và `src/lab04/train.py`.
4. **WHICH:** Sử dụng `statsmodels.tsa.api.VAR`, `xgboost.XGBRegressor`, `torch.nn` (`LSTM`, `DCGRUCell`, `DCRNNModel`).
5. **WHEN:** Thực thi tại Bước 4 (`TRAINING`) của Pipeline.
6. **HOW:** Tự động đẩy `LSTM` và `DCRNN` lên GPU với `GradScaler` (AMP FP16), giám sát `EarlyStopping` trên tập `Validation (15%)` và lưu trọng số tốt nhất vào `models/lab04/`.
7. **CODE WALKTHROUGH:** Lớp `DCGRUCell` trong `src/lab04/model.py` cài đặt chính xác công thức tích chập khuếch tán hai chiều trên hai ma trận chuyển tiếp稀疏/đầy đủ $\mathbf{P}_f = \mathbf{D}_O^{-1}\mathbf{W}$ và $\mathbf{P}_b = \mathbf{D}_I^{-1}\mathbf{W}^T$ từ `adj_mx_bay.pkl`.

### 8. `src/lab04/evaluate.py`, `src/lab04/visualize.py`, `src/lab04/report.py` & `render_report.py`
1. **WHAT:** Nhóm module thực hiện Walk-Forward Backtesting, tính 5 chỉ số (`MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²`), vẽ biểu đồ 300 DPI và tự động xuất `LAB_04_REPORT.pdf`, `LAB_04_REPORT.html`, `LAB_04_REPORT.md`.
2. **WHY:** Tự động hóa 100% khâu chấm điểm, vẽ hình và biên soạn báo cáo chuẩn phong cách `Lab04_ML-main`.
3. **WHERE:** `src/lab04/evaluate.py`, `src/lab04/visualize.py`, `src/lab04/report.py` và `render_report.py`.
4. **WHICH:** Sử dụng `numpy`, `pandas`, `matplotlib`, `seaborn`, `base64` và trình kết xuất PDF tự động.
5. **WHEN:** Thực thi tại Bước 5 (`EVALUATION`) và Bước 6 (`REPORTING`) của Pipeline.
6. **HOW:** Đọc trực tiếp số liệu từ `reports/lab04/metrics/evaluation_latest.json` và `reports/lab04/metrics/kfold_cv_results.csv`, nhúng ảnh PNG dạng Base64 vào HTML và xuất ra PDF chuẩn in ấn.
7. **CODE WALKTHROUGH:** Hàm `_require_json()` trong `render_report.py` đảm bảo báo cáo luôn lấy 100% số liệu thật từ kết quả chạy code, tuyệt đối không dùng số giả định.
