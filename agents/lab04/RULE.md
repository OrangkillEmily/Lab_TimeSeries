# QUY TẮC LẬP TRÌNH, KIẾN TRÚC VÀ KIỂM ĐỊNH CHẤT LƯỢNG (LAB04 TIME SERIES — RULE.MD)

> **Áp dụng cho:** Dự án `Lab04_Multivariate_TimeSeries` (`agents/lab04/RULE.md`)  

## Quy tắc 1: Kế thừa Kiến trúc Chuẩn có Chọn lọc (Reference Architecture)
- **Được phép kế thừa:**
  - Tư duy phân tách module sạch sẽ trong `src/` (`data_loader.py`, `eda.py`, `preprocess.py`, `model.py`, `train.py`, `evaluate.py`, `visualize.py`, `tracking.py`, `report.py`, `pipeline.py`, `utils.py`).
  - Quản lý toàn bộ siêu tham số tập trung qua `config.yaml` và điểm khởi chạy duy nhất `run_lab04.py` + `render_report.py`.
  - Hệ thống theo dõi phiên thực nghiệm theo mã định danh `Run_ID` đồng bộ giữa `logs/pipeline_<run_id>.log`, `progress_results/progress_<run_id>.json`, `docs/lab04/experiment_logs.csv`, `reports/figures/<run_id>/` và bộ 3 báo cáo tự động (`LAB_04_REPORT.pdf`, `LAB_04_REPORT.html`, `LAB_04_REPORT.md`).
- **Tuyệt đối không sao chép nhầm bài toán cũ:**
  - Loại bỏ hoàn toàn dữ liệu giá nhà Boston Housing (`MEDV`, `RM`, `LSTAT`) và các mô hình không thuộc đề bài.
  - Toàn bộ dự án phục vụ duy nhất 2 bài toán chuỗi thời gian đa biến: **Electricity Load + Weather Forecasting (Yêu cầu 1–11)** trên [Electricity Load Diagrams Dataset (UCI / Kaggle)](https://www.kaggle.com/datasets/jacopoferretti/electricity-load-diagrams-2011-2014-feature-engin) và **Traffic Forecasting trên Đồ thị Cảm biến (Yêu cầu 12)** với [PEMS-BAY Dataset](https://drive.google.com/file/d/1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq/view?usp=drive_link) + [DCRNN Sensor Graph `adj_mx_bay.pkl`](https://github.com/liyaguang/dcrnn).

## Quy tắc 2: Chống Rò rỉ Dữ liệu Thời gian Tuyệt đối (Strict Temporal Anti-Leakage)
1. **Cấm tuyệt đối `shuffle=True` khi chia tập dữ liệu:**
   - Theo Mục 4.2 của `practice 4.pdf`, chuỗi thời gian phải được chia theo **đúng thứ tự thời gian (Chronological Order)** với tỷ lệ chuẩn: **Train 70% – Validation 15% – Test 15%**:
     $$t_{\max}(\text{Train}) < t_{\min}(\text{Validation}) \le t_{\max}(\text{Validation}) < t_{\min}(\text{Test})$$
2. **Chuẩn hóa `StandardScaler` Không Rò rỉ (`In-Split & In-Fold Scaling`):**
   - Trong phép chia `70% / 15% / 15%`: Chỉ được gọi `scaler.fit()` trên tập `Train (70%)`, sau đó dùng `scaler.transform()` cho `Validation (15%)` và `Test (15%)`.
   - Trong phép kiểm định chéo **Temporal K-Fold (`TimeSeriesSplit`, $K=5$ Folds)**: Ở mỗi Fold $k \in \{1..5\}$, bắt buộc khởi tạo một đối tượng `StandardScaler` mới, chỉ `fit()` trên $Train^{(k)}$ và `transform()` trên $Val^{(k)}$ ([Scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)).
3. **Tính Nhân quả của Đặc trưng Trễ và Thống kê Trượt (Causal Feature Engineering):**
   - Tại thời điểm dự báo $t \to t+1$, toàn bộ cửa sổ trễ $L = 24$ giờ và các đặc trưng trượt (`rolling_mean_6h`, `rolling_std_24h`, `load_trend = load_t - load_{t-24}`) chỉ được phép sử dụng các quan sát ở thời điểm $\le t$. Tuyệt đối không dùng cửa sổ trượt trung tâm (`center=True`) hoặc dùng giá trị tại $t+1$.
4. **Cô lập Tập Test (`Test Set Isolation`):**
   - Cấm truyền dữ liệu `Test` vào `eval_set` của `XGBoost` hoặc `validation_data` của `LSTM` / `DCRNN`. Mọi quá trình chọn bậc trễ, tinh chỉnh siêu tham số và `EarlyStopping` chỉ được phép giám sát trên tập `Validation (15%)`.

## Quy tắc 3: Bắt buộc Kết hợp Chia `Train-Val-Test (70/15/15)` và `Temporal K-Fold CV`
- Mọi mô hình dự báo phụ tải điện (`VAR`, `XGBoost`, `LSTM`) phải được đánh giá trên cả **3 tầng kiểm định**:
  1. **Tầng 1 — Chronological Hold-out (`70% Train - 15% Val - 15% Test`)**: Đánh giá hiệu năng độc lập trên 15% dữ liệu cuối chuỗi.
  2. **Tầng 2 — Temporal K-Fold Cross-Validation (`TimeSeriesSplit`, $K=5$ Folds)**: Đánh giá độ ổn định của mô hình qua 5 lát cắt thời gian liên tiếp, báo cáo chi tiết từng Fold (`Fold 1..5`) kèm **`Mean ± Std`** vào `reports/metrics/kfold_cv_results.csv`.
  3. **Tầng 3 — Walk-Forward Validation / Backtesting (Yêu cầu 9)**: Thực thi cuốn chiếu `Train -> predict t+1 -> cập nhật vào chuỗi -> predict t+2 -> ...` mô phỏng môi trường vận hành thực tế.

## Quy tắc 4: Tự động Chuyển sang GPU (`Auto-GPU Hardware Acceleration`)
- Các mô hình học sâu (`LSTM` ở Yêu cầu 8 và `DCRNN` ở Yêu cầu 12) cũng như `XGBoost` trên chuỗi thời gian dài có khối lượng tính toán lớn; do đó module `src/utils.py` và `src/model.py` bắt buộc triển khai bộ điều phối phần cứng tự động **`HardwareAccelerator`**:
  1. **Thứ tự ưu tiên thiết bị:** Tự động kiểm tra `NVIDIA CUDA GPU (cuda:0)` $\rightarrow$ `Apple Metal (mps)` / `DirectML` $\rightarrow$ `Multi-core CPU (fallback)`.
  2. **Tối ưu Deep Learning trên GPU:** Khi phát hiện GPU CUDA, tự động kích hoạt **Automatic Mixed Precision (`AMP FP16`)**, bật `torch.backends.cudnn.benchmark = True`, sử dụng `pin_memory=True` cho `DataLoader` và giải phóng bộ nhớ đệm VRAM (`torch.cuda.empty_cache()` / `tf.keras.backend.clear_session()`) ngay sau mỗi Fold để ngăn ngừa lỗi tràn bộ nhớ (`Out-Of-Memory`).
  3. **Tối ưu XGBoost trên GPU:** Tự động cấu hình `tree_method="hist"` và `device="cuda"` khi có GPU NVIDIA, và tự động lùi về `device="cpu", n_jobs=-1` nếu chạy trên máy chỉ có CPU.
  4. **Minh bạch phần cứng:** Ghi nhận tên GPU/CPU, dung lượng VRAM và trạng thái Mixed Precision vào `logs/pipeline_<run_id>.log`, `models/training_metadata.json` và hiển thị trên trang đầu báo cáo `LAB_04_REPORT.pdf` / `.html` / `.md`.

## Quy tắc 5: Đánh giá Đầy đủ 5 Chỉ số (`MSE`, `RMSE`, `MAE`, `MAPE`, `R²`) trên Đơn vị Gốc
- Ngoài 3 chỉ số trong đề bài (`RMSE`, `MAE`, `MAPE`), hệ thống bắt buộc tính toán thêm **`MSE`** và **`R² Score`** (tổng cộng **5 chỉ số**) cho mọi mô hình, mọi Fold trong K-Fold CV, tập Test và Walk-Forward Backtesting.
- **Quy tắc Đơn vị Vật lý (`Inverse Transform before Evaluation`):** Vì mô hình được huấn luyện trên dữ liệu đã chuẩn hóa `StandardScaler` ($\mu=0, \sigma=1$), trước khi tính 5 chỉ số `MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²`, bắt buộc thực hiện phép biến đổi ngược $\hat{y}_{\text{orig}} = \hat{y}_{\text{scaled}} \cdot \sigma_{\text{load, train}} + \mu_{\text{load, train}}$ (và tương tự cho tốc độ xe mph trên PEMS-BAY) để các con số báo cáo mang đúng ý nghĩa vật lý và tránh lỗi chia cho $0$ khi tính `MAPE`.

## Quy tắc 6: Chuẩn Hóa Tài liệu `docs/lab04` theo Đúng Bố cục 5 Phần Bắt buộc
- Tất cả các file tài liệu theo giai đoạn (`docs/lab04EDA_Phase.md`, `docs/lab04/Preprocessing_Phase.md`, `docs/lab04/Feature_Engineering_Phase.md`, `docs/lab04/Training_Phase.md`, `docs/lab04/Inference_Phase.md`, `docs/lab04/Evaluation_Phase.md`) và các file riêng cho từng thuật toán (`docs/lab04/VAR_Baseline.md`, `docs/lab04/Multivariate_XGBoost.md`, `docs/lab04/LSTM_Architecture.md`, `docs/lab04/DCRNN.md`) **bắt buộc** phải tuân thủ đúng cấu trúc 5 mục:
  1. **`## 1. Purpose`** — Mục đích học thuật và vai trò trong hệ thống.
  2. **`## 2. Input-Output`** — Bảng đặc tả chi tiết đầu vào (kích thước Tensor/DataFrame, kiểu dữ liệu, cấu hình `config.yaml`) và đầu ra (tệp model, bảng metrics, biểu đồ).
  3. **`## 3. How to do`** — Sơ đồ luồng xử lý, công thức toán học tường minh và giải thích từng bước thực hiện.
  4. **`## 4. Evaluation metrics`** — Công thức toán học, cách tính và bảng đối chiếu đủ 5 chỉ số (`MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²`).
  5. **`## 5. Used data`** — Chi tiết tập dữ liệu sử dụng (`electricity_load.csv` hoặc `pems-bay.h5` + `adj_mx_bay.pkl`), tỷ lệ chia `70/15/15`, cấu hình `5-Fold CV` và ánh xạ tới module code trong `src/`.

## Quy tắc 7: Bắt buộc có Docstring `5W1H` cho Mọi Hàm trong `src/`
- Mọi hàm trong `src/` và `script/lab04/render_report.py` phải có docstring chuẩn giải thích rõ 5 mục:
  - **What:** Hàm thực hiện nhiệm vụ gì?
  - **Why:** Tại sao cần hàm này trong pipeline dự báo chuỗi thời gian đa biến?
  - **Input:** Các tham số đầu vào, kiểu dữ liệu và kích thước mảng/tensor là gì?
  - **Output:** Giá trị trả về và các tệp tạo ra là gì?
  - **How:** Thuật toán hoặc cơ chế xử lý bên trong hoạt động như thế nào?

## Quy tắc 8: Trung thực Thực nghiệm, Khả năng Tái lập (`Reproducibility`) & Tự động Xuất Báo cáo
- Cố định hạt giống ngẫu nhiên `random_state = 42` trong `config.yaml` và đồng bộ qua `random`, `numpy.random`, `torch.manual_seed`, `torch.cuda.manual_seed_all` và `tensorflow.random.set_seed`.
- Toàn bộ bảng số liệu trong `LAB_04_REPORT.md`, `LAB_04_REPORT.html`, `LAB_04_REPORT.pdf` đều phải được tổng hợp tự động từ `reports/lab04/metrics/evaluation_latest.json`, `reports/lab04/metrics/kfold_cv_results.csv` và `progress_results/lab04/progress_<run_id>.json` thông qua `src/lab04/report.py` và `script/lab04/render_report.py`.
