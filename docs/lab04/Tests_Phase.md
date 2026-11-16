# Giai đoạn Kiểm thử Tự động & Đảm bảo Chất lượng Phần mềm (`Tests_Phase.md` — Automated Unit & Integration Testing)

## 1. Purpose — Mục đích

**Giai đoạn Kiểm thử Tự động (`Tests_Phase`)** thực hiện kiểm định chất lượng mã nguồn, tính toàn vẹn toán học và khả năng tái lập (`Reproducibility`) trên toàn bộ kiến trúc `src/lab04/` thông qua bộ **19 ca kiểm thử tự động (`19 pytest test cases`)** trong thư mục `tests/lab04/`.

Mục đích cốt lõi của giai đoạn này là:
1. **Kiểm thử Chống Rò rỉ Thời gian (`Anti-Leakage & Temporal Split Verification`)**: Đảm bảo phép chia `70% Train – 15% Validation – 15% Test` (`shuffle=False`) và `5-Fold Expanding Window TimeSeriesSplit` tuân thủ tuyệt đối trật tự nhân quả thời gian ($t_{\max}(\text{Train}) < t_{\min}(\text{Val}) < t_{\min}(\text{Test})$) và bộ `StandardScaler` chỉ được `fit` trên tập `Train`.
2. **Kiểm thử Cấu trúc Tensor & Kiến trúc 6 Mô hình (`Model & Tensor Shape Verification`)**: Xác nhận kích thước ma trận `105` đặc trưng phẳng của `XGBoost`, Tensor 3D `(B, 24, 4)` của `Residual LSTM`, chuỗi sai phân bậc 1 của `VAR(24)` và Tensor không gian – thời gian `(B, 12, 207)` cùng ma trận khuếch tán hai chiều $(\mathbf{P}_f, \mathbf{P}_b)$ của bộ 3 `DCRNN`, `Multi-Horizon TCN`, `Multi-Horizon LSTM`.
3. **Kiểm thử Vòng lặp Huấn luyện GPU/AMP & Quy tắc PyTorch Step Order (`Training Loop & AMP Safety`)**: Đảm bảo thứ tự gọi `scaler.step(optimizer)` trước `scheduler.step()`, khởi tạo tường minh biến `optimizer_stepped` trong mọi nhánh thực thi (ngăn ngừa `UnboundLocalError`), và kiểm tra cơ chế tự động chuyển đổi `CUDA / CPU`.
4. **Kiểm thử 5 Chỉ số Đánh giá & Pipeline Kết xuất Báo cáo (`Metrics, Visualization & Report Rendering Audit`)**: Xác nhận công thức toán học của 5 chỉ số (`MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²`), kiểm tra không có cảnh báo `SyntaxWarning` (chuỗi docstring raw string `r"""..."""`) hay `FutureWarning` (`adfuller(..., result_object=False)`), và nghiệm thu đầy đủ `23` biểu đồ cùng bộ 3 báo cáo `LAB_04_REPORT.{html, pdf, md}` theo bộ tiêu chí `AC01–AC15`.

## 2. Input-Output

### Input

| Đầu vào (Input) | Định dạng & Nguồn | Mô tả Chi tiết |
|---|---|---|
| Các module mã nguồn `src/lab04/` | Python Package (`8` modules) | Gồm `utils.py`, `tracking.py`, `data_loader.py`, `eda.py`, `preprocess.py`, `model.py`, `train.py`, `evaluate.py`, `visualize.py`, `report.py` |
| `config.yaml` & Synthetic / Sample Fixtures | `YAML` & `pytest.fixture` | Cấu hình trung tâm của dự án và các tensor/dataframe giả lập (`synthetic fixtures`) giúp chạy nhanh unit tests độc lập |
| Dữ liệu & Artifacts Thực nghiệm | `data/raw/`, `models/lab04/`, `reports/lab04/` | Các bảng chỉ số `.csv` trong `reports/lab04/metrics/`, `23` biểu đồ `.png` và bộ 3 báo cáo `LAB_04_REPORT.{html,pdf,md}` dùng cho kiểm thử tích hợp (*Integration Tests*) |

### Output

| Đầu ra (Output) | Đường dẫn / Kết quả | Mô tả Chi tiết |
|---|---|---|
| Kết quả thực thi `pytest tests/lab04` | Terminal / CI Log (`19 passed, 0 warnings`) | Toàn bộ `19/19` test cases vượt qua kiểm định (`100% pass rate`) mà không phát sinh `SyntaxWarning`, `FutureWarning` hay `UserWarning` |
| Kiểm chứng Nghiệm thu `AC01–AC15` | `reports/lab04/LAB_04_REPORT.{html,pdf,md}` | Bảng trạng thái `PASS` toàn bộ 15 tiêu chí nghiệm thu kỹ thuật từ khâu nạp dữ liệu đến xuất báo cáo PDF/HTML/MD |

## 3. How to do — Quy trình thực hiện & Cấu trúc 19 Test Cases (`tests/lab04/`)

### Sơ đồ luồng thực thi `Tests_Phase`

```text
                         Khởi chạy: pytest tests/lab04 -v
                                        |
        +-------------------------------+-------------------------------+
        |                               |                               |
        v                               v                               v
 [Nhóm 1: Data, EDA & Preprocess] [Nhóm 2: Models & Training Loop] [Nhóm 3: Eval, Figures & Report]
 - Kiểm tra config & GPU device   - Forward pass Residual LSTM     - Kiểm định công thức 5 Metrics
 - Kiểm tra nạp CSV & HDF5/PKL    - Forward pass DCRNN, TCN, LSTM  - Kiểm tra bảng Hold-out, CV, WF
 - Kiểm định 70/15/15 & 5-Fold CV - Kiểm tra AMP optimizer_stepped - Kiểm tra đủ 23 biểu đồ PNG
 - Kiểm tra StandardScaler & ADF  - Kiểm tra thứ tự Optimizer->LR  - Nghiệm thu báo cáo HTML/PDF/MD
        |                               |                               |
        +-------------------------------+-------------------------------+
                                        |
                                        v
                 Kết quả: 19 passed in tests/lab04 (Zero Warnings)
```

### Danh mục & Nội dung Kiểm định của 19 Test Cases trong `tests/lab04/`

1. **Nhóm Kiểm thử Cấu hình, Nạp Dữ liệu & Tiền xử lý (`TC01 – TC07`):**
   - `TC01 (test_config_and_hardware_detection)`: Kiểm tra `load_config()` nạp đầy đủ khóa cấu hình, `set_global_seed(42)` cố định hạt giống ngẫu nhiên và `get_compute_device()` trả về đúng cấu trúc thiết bị (`torch_device`, `xgb_device`, `amp_enabled`).
   - `TC02 (test_electricity_dataset_integrity)`: Kiểm tra `load_electricity_dataset()` trả về đúng `4` cột `[load, temperature, humidity, wind_speed]`, chỉ số `DatetimeIndex` tăng đơn điệu theo từng giờ và không có giá trị `NaN`.
   - `TC03 (test_pems_bay_and_graph_loading)`: Kiểm tra đọc file HDF5 `pems-bay.h5` (thông qua `tables` / `pytables`) và ma trận kề `adj_mx_bay.pkl` kích thước vuông `(207, 207)` không âm, xử lý an toàn đường dẫn dự phòng (`fallback path resolution`).
   - `TC04 (test_chronological_split_no_leakage)`: Xác minh tỷ lệ chia `70% Train – 15% Val – 15% Test` với `shuffle=False`, đảm bảo mốc thời gian cuối của tập trước luôn nhỏ hơn nghiêm ngặt mốc đầu của tập sau:
     $$\max(t_{\text{train}}) < \min(t_{\text{val}}) \quad \wedge \quad \max(t_{\text{val}}) < \min(t_{\text{test}})$$
   - `TC05 (test_standard_scaler_train_only_fit)`: Kiểm증 bộ `StandardScaler` chỉ khớp trên tập `Train` ($\|\boldsymbol{\mu}(\mathbf{X}_{\text{train}}^{\text{scaled}})\|_2 < 10^{-6}$, $\|\boldsymbol{\sigma}(\mathbf{X}_{\text{train}}^{\text{scaled}}) - \mathbf{1}\|_2 < 10^{-6}$) và biến đổi ngược `inverse_transform` khôi phục chính xác chuỗi gốc ($\|\mathbf{X} - \text{Scaler}^{-1}(\text{Scaler}(\mathbf{X}))\|_{\infty} < 10^{-5}$).
   - `TC06 (test_expanding_window_5fold_cv)`: Kiểm tra `TimeSeriesSplit(n_splits=5)` tạo đúng 5 Fold mở rộng dần mà không rò rỉ dữ liệu tương lai giữa các Fold.
   - `TC07 (test_adf_stationarity_and_clean_warnings)`: Kiểm tra hàm kiểm định dừng `ADF` gọi `adfuller(..., result_object=False)` không sinh `FutureWarning`, chuỗi sai phân bậc 1 đạt $p\text{-value} < 0.05$, và toàn bộ docstring trong `preprocess.py` dùng raw strings `r"""..."""` không sinh `SyntaxWarning`.

2. **Nhóm Kiểm thử Kỹ thuật Đặc trưng, Kiến trúc Mô hình & Vòng lặp Huấn luyện (`TC08 – TC14`):**
   - `TC08 (test_supervised_windows_shapes)`: Kiểm tra hàm tạo cửa sổ trượt sinh đúng ma trận `105` đặc trưng cho `XGBoost` `(N, 105)` và Tensor 3D `(N, 24, 4)` cho `Residual LSTM`.
   - `TC09 (test_residual_lstm_forward_shape)`: Kiểm tra mạng `Stacked 2-Layer LSTM + Autoregressive Residual Highway` nhận đầu vào `(batch_size, 24, 4)` và xuất dự báo `(batch_size, 1)` hữu hạn (không chứa `NaN/Inf`).
   - `TC10 (test_dcrnn_graph_diffusion_forward)`: Kiểm tra toán tử khuếch tán đồ thị hai chiều $(\mathbf{P}_f, \mathbf{P}_b)$ và kiến trúc `DCRNN` nhận tensor `(batch_size, 12, 207)` cùng `adj_mx (207, 207)` xuất đúng tensor đa bước `(batch_size, 12, 207)`.
   - `TC11 (test_multi_horizon_tcn_causal_conv)`: Kiểm tra mạng `Multi-Horizon TCN` (Dilated Causal Conv1D) đảm bảo tính nhân quả (đầu ra tại bước $t$ không phụ thuộc vào bước $>t$) và xuất đúng chiều `(batch_size, 12, 207)`.
   - `TC12 (test_multi_horizon_lstm_baseline)`: Kiểm tra mạng `Multi-Horizon LSTM` thuần thời gian trên `PEMS-BAY` xuất đúng tensor dự báo ở cả 3 mốc `t+3 (15m)`, `t+6 (30m)`, `t+12 (60m)`.
   - `TC13 (test_optimizer_scheduler_step_order_and_scope)`: Kiểm tra vòng lặp huấn luyện PyTorch (cả khi bật và tắt `AMP FP16`) luôn khởi tạo `optimizer_stepped = False` trước mỗi batch và chỉ gọi `scheduler.step()` khi `optimizer_stepped == True`, ngăn chặn hoàn toàn lỗi `UnboundLocalError` và cảnh báo thứ tự bước của PyTorch.
   - `TC14 (test_var_differencing_and_inverse_reconstruction)`: Kiểm tra mô hình `VAR` dự báo trên miền sai phân bậc 1 và cộng hoàn nguyên chính xác về mức phụ tải gốc $\hat{\mathbf{X}}_{t+1} = \mathbf{X}_t + \widehat{\Delta\mathbf{X}}_{t+1}$.

3. **Nhóm Kiểm thử 5 Chỉ số Đánh giá, Kho Biểu đồ & Báo cáo Tự động (`TC15 – TC19`):**
   - `TC15 (test_five_regression_metrics_math)`: Kiểm tra hàm tính bộ 5 chỉ số (`MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²`) khớp tuyệt đối với công thức toán học chuẩn ($\text{RMSE} = \sqrt{\text{MSE}}$, $R^2 = 1.0$ khi $\hat{\mathbf{y}} = \mathbf{y}$, và xử lý an toàn mẫu số $\max(|y_t|, \epsilon)$ trong `MAPE`).
   - `TC16 (test_walk_forward_and_ablation_schema)`: Kiểm tra cấu trúc bảng `walk_forward_metrics.csv` (`168` bước cuốn chiếu) và `weather_ablation_metrics.csv` (`With_Weather` vs. `Without_Weather`).
   - `TC17 (test_pems_bay_3model_benchmark_schema)`: Kiểm tra bảng `pems_bay_benchmark_comparison.csv` chứa đầy đủ kết quả của cả 3 mô hình (`DCRNN`, `Multi_Horizon_TCN`, `Multi_Horizon_LSTM`) trên 3 tầm nhìn (`15_min`, `30_min`, `60_min`) và `Average_All_Horizons`.
   - `TC18 (test_all_23_figures_generated)`: Kiểm tra `visualize.py` và `eda.py` kết xuất đầy đủ `23` file biểu đồ `.png` có dung lượng $> 0\text{ bytes}$ trong `reports/lab04/figures/`.
   - `TC19 (test_synchronized_report_rendering)`: Kiểm tra `report.py` / `render_report.py` kết xuất đồng bộ cả 3 định dạng `LAB_04_REPORT.html`, `LAB_04_REPORT.pdf` và `LAB_04_REPORT.md` với số liệu khớp chính xác 100% với các file CSV trong `reports/lab04/metrics/`.

## 4. Evaluation metrics — Các chỉ số kiểm định chất lượng bộ Test Suite (`QA Metrics`)

Bộ kiểm thử `Tests_Phase` đóng vai trò cổng kiểm soát chất lượng (*Quality Gate*) để bảo vệ độ chính xác của cả 5 chỉ số hồi quy (`MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²`) trên toàn dự án:

| Chỉ số Đảm bảo Chất lượng (`QA Metric`) | Ngưỡng Yêu cầu (`Target`) | Kết quả Thực tế Đạt được | Ý nghĩa Kỹ thuật đối với Dự án Lab 04 |
|---|---|---|---|
| **Tỷ lệ Vượt qua Kiểm thử (`Test Pass Rate`)** | `19 / 19 (100%)` | **`19 passed (100%)`** | Xác nhận toàn bộ 8 module trong `src/lab04/` hoạt động ổn định, không có lỗi logic |
| **Cảnh báo Biên dịch & Runtime (`Warnings Count`)** | `0 warnings` | **`0 SyntaxWarning, 0 FutureWarning`** | Đảm bảo mã nguồn sạch tuyệt đối trên Python 3.11–3.14, PyTorch CUDA và Statsmodels mới |
| **Sai số Khôi phục Chuẩn hóa (`Inverse Scaling Error`)** | $\|\mathbf{y} - \hat{\mathbf{y}}_{\text{reconstructed}}\|_{\infty} < 10^{-5}$ | **`< 1e-6 kW / mph`** | Đảm bảo `MSE`, `RMSE`, `MAE`, `MAPE`, `R²` không bị sai lệch do lỗi làm tròn số học khi `inverse_transform` |
| **Độ Bao phủ Nghiệm thu (`Acceptance Criteria AC01–AC15`)** | `15 / 15 PASS` | **`15 / 15 PASS`** | Xác nhận báo cáo cuối cùng đáp ứng trọn vẹn mọi yêu cầu cơ bản và nâng cao của đề bài |

## 5. Used data — Dữ liệu sử dụng trong Kiểm thử

| Nguồn Dữ liệu Kiểm thử | Đường dẫn / Phạm vi | Vai trò trong `Tests_Phase` | Module Thực thi |
|---|---|---|---|
| **Synthetic Unit Test Fixtures** | Khởi tạo trong bộ nhớ qua `pytest.fixture` | Kiểm thử nhanh (dưới vài giây) cho từng hàm toán học, tầng mạng thần kinh (`LSTM`, `DCRNN`, `TCN`) và thứ tự bước `optimizer / scheduler` | `tests/lab04/test_*.py` |
| **Raw Datasets & Sensor Graph** | `data/raw/electricity_load.csv`, `data/raw/pems-bay.h5`, `data/raw/sensor_graph/adj_mx_bay.pkl` | Kiểm thử tích hợp bộ nạp dữ liệu (`data_loader.py`) và tiền xử lý (`preprocess.py`) | `tests/lab04/`, `src/lab04/data_loader.py` |
| **Production Metrics, Figures & Reports** | `reports/lab04/metrics/*.csv`, `reports/lab04/figures/*.png`, `reports/lab04/LAB_04_REPORT.*` | Kiểm thử nghiệm thu đầu-cuối (*End-to-End Acceptance Testing*) cho toàn bộ pipeline | `tests/lab04/`, `src/lab04/evaluate.py`, `src/lab04/report.py` |
