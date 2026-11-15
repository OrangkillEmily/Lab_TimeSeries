# Giai đoạn Huấn luyện Đa Mô hình & Tự động Tăng tốc GPU (`Training_Phase.md`)

## 1. Purpose — Mục đích

**Giai đoạn Huấn luyện (`Training_Phase`)** thực hiện **Mục 6 (Huấn luyện VAR)**, **Mục 7 (Huấn luyện XGBoost)**, **Mục 8 (Huấn luyện LSTM)** và **Mục 12 (Huấn luyện DCRNN cho Traffic Forecasting)** trong `practice 4.pdf`, kết hợp **Yêu cầu bổ sung số 4 (Huấn luyện trên cả `Train-Val-Test` và `5-Fold Cross-Validation`)** cùng **Yêu cầu bổ sung số 5 (Tự động chuyển sang chạy bằng GPU khi máy có GPU)**.

Mục đích của giai đoạn này là ước lượng bộ tham số tối ưu cho cả 4 thuật toán (`VAR`, `XGBoost`, `LSTM`, `DCRNN`), giám sát chống quá khớp bằng cơ chế `EarlyStopping` trên tập `Validation (15%)`, đồng thời tận dụng tối đa năng lực tính toán của GPU (`CUDA AMP FP16` / `MPS`) để rút ngắn thời gian huấn luyện.

## 2. Input-Output

### Input

| Đầu vào (Input) | Định dạng & Kích thước | Mô tả Chi tiết |
|---|---|---|
| `train_stationary` (`VAR`) | `pd.DataFrame` `(T_train, 4)` | Chuỗi đa biến 4 cột đã chuẩn hóa và lấy sai phân đảm bảo tính dừng ($p\text{-value ADF} < 0.05$) |
| `X_xgb_train`, `y_train`, `X_xgb_val`, `y_val` | `np.ndarray` `(N, D_xgb)` | Ma trận đặc trưng phẳng (Lag 24h + Rolling 6h/24h + Trend) từ [Electricity Load Diagrams Dataset](https://www.kaggle.com/datasets/jacopoferretti/electricity-load-diagrams-2011-2014-feature-engin) |
| `X_lstm_train`, `y_train`, `X_lstm_val`, `y_val` | `Tensor` `(N, 24, 4)` | Tensor 3D chuỗi 24 giờ $\times$ 4 biến dùng để huấn luyện mạng `LSTM(64) -> LSTM(32) -> Dense(1)` |
| `pems_train_loader`, `pems_val_loader`, `adj_mx` | `DataLoader` & `np.ndarray` | Dữ liệu cửa sổ 12 bước tốc độ xe từ [PEMS-BAY Dataset](https://drive.google.com/file/d/1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq/view?usp=drive_link) và đồ thị cảm biến từ [DCRNN GitHub](https://github.com/liyaguang/dcrnn) |
| `config.yaml` (`hardware`, `var_model`, `xgboost_model`, `lstm_model`, `pems_bay_advanced`) | `Dict[str, Any]` | Cấu hình siêu tham số huấn luyện và cờ điều phối phần cứng `device = "auto"`, `use_mixed_precision = true` |

### Output

| Đầu ra (Output) | Đường dẫn Lưu trữ | Mô tả Chi tiết |
|---|---|---|
| `var_model.pkl` | `models/var_model.pkl` | Mô hình thống kê `VAR(p*)` đã khớp với bậc trễ tối ưu $p^*$ theo tiêu chuẩn AIC/BIC |
| `xgboost_model.json` / `.pkl` | `models/xgboost_model.pkl` | Mô hình `XGBRegressor` tại vòng lặp tối ưu (`best_iteration`) cùng bảng `Feature Importance` |
| `lstm_model.pt` / `.keras` | `models/lstm_model.pt` | Trọng số mạng `LSTM(64) -> LSTM(32) -> Dense(1)` tại epoch có `val_loss` thấp nhất |
| `dcrnn_pems_bay.pt` | `models/dcrnn_pems_bay.pt` | Trọng số mô hình `DCRNN (DCGRU Encoder-Decoder)` đã huấn luyện cho bài toán dự báo giao thông 15, 30, 60 phút |
| `training_history.csv` & `training_metadata.json` | `models/training_history.csv`, `models/training_metadata.json` | Lịch sử hội tụ `train_loss`, `val_loss` qua các epochs, thông tin thiết bị thực thi (`GPU CUDA` / `CPU`) và thời gian huấn luyện |
| `lstm_train_vs_val_loss.png` & `training_time_comparison.png` | `reports/figures/` | Biểu đồ đường cong hội tụ kiểm tra Overfitting (Câu hỏi 5 Mục 11) và biểu đồ so sánh thời gian huấn luyện |

## 3. How to do — Quy trình thực hiện & Cơ sở toán học

### Sơ đồ luồng điều phối `Auto-GPU` và huấn luyện 4 mô hình

```text
                        Khởi động src/lab04/train.py (Run_ID)
                                      |
                                      v
      [HardwareAccelerator] Kiểm tra torch.cuda.is_available() / TF GPU / MPS
      +--> Nếu có GPU NVIDIA: Bật device="cuda:0", AMP FP16, XGBoost tree_method="hist"
      +--> Nếu không có GPU : Fallback tự động về Multi-core CPU (n_jobs=-1)
                                      |
        +-----------------------------+-----------------------------+
        |                             |                             |
        v                             v                             v
 [1. VAR Baseline]             [2. XGBoost ML]               [3. LSTM & 4. DCRNN DL]
 - Quét p = 1..24              - Input phẳng (Lag+Rolling)   - LSTM: (B,24,4)->64->32->1
 - Chọn p* cực tiểu AIC/BIC    - Tối ưu Squared Error        - DCRNN: DCGRU Seq2Seq (12->12)
 - Ước lượng OLS đa biến       - EarlyStopping trên Val      - Adam + MSE/MAE + EarlyStopping
        |                             |                             |
        +-----------------------------+-----------------------------+
                                      |
                                      v
      Lặp lại quy trình huấn luyện độc lập trên 5 Folds của TimeSeriesSplit(n_splits=5)
```

### Các bước thực hiện chi tiết

**Step 1 — Tự động Nhận diện và Chuyển đổi Phần cứng (`Auto-GPU Dispatcher`)**

Trước khi khởi tạo mô hình, hàm `detect_compute_device()` trong `src/lab04/utils.py` kiểm tra phần cứng thực tế:
- Với **LSTM** và **DCRNN**: Tự động chuyển toàn bộ mô hình và mini-batch tensor sang `cuda:0` (nếu có GPU NVIDIA) hoặc `mps` (Apple Silicon), kết hợp `torch.cuda.amp.autocast(dtype=torch.float16)` để tăng tốc phép nhân ma trận tuần tự và giảm một nửa dung lượng VRAM tiêu thụ.
- Với **XGBoost**: Tự động truyền `tree_method="hist"` và `device="cuda"` khi phát hiện GPU, giúp tốc độ dựng cây quyết định trên hàng chục ngàn mẫu nhanh gấp nhiều lần so với CPU.

**Step 2 — Huấn luyện Mô hình 1: VAR (`Vector Autoregression` — Mục 6)**

Trên chuỗi dừng $\tilde{\mathbf{X}}_t \in \mathbb{R}^{4}$, hệ thống khớp mô hình $\text{VAR}(p)$ cho các bậc trễ $p \in \{1, \dots, 24\}$ và chọn $p^*$ tối ưu hóa tiêu chuẩn thông tin Akaike (**AIC**):

$$\text{AIC}(p) = \ln\left|\hat{\boldsymbol{\Sigma}}_{\varepsilon}(p)\right| + \frac{2 p K^2}{T_{\text{train}}}$$

trong đó $\hat{\boldsymbol{\Sigma}}_{\varepsilon}(p)$ là ma trận hiệp phương sai phần dư của hệ phương trình tự hồi quy vector ([Statsmodels VAR](https://www.statsmodels.org/stable/vector_ar.html)).

**Step 3 — Huấn luyện Mô hình 2: XGBoost (`Extreme Gradient Boosting` — Mục 7)**

Tối ưu hóa hàm mục tiêu có điều chuẩn bậc hai theo thuật toán Gradient Boosting ([XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/tutorials/model.html)):

$$\mathcal{L}^{(m)} = \sum_{i=1}^{N_{\text{train}}} \left(y_i - \left(\hat{y}_i^{(m-1)} + f_m(\mathbf{x}_i)\right)\right)^2 + \gamma T_m + \frac{1}{2}\lambda \sum_{j=1}^{T_m} w_j^2$$

Theo dõi sai số trên tập `Validation (15%)` ở mỗi cây bổ sung và dừng sớm khi `val_rmse` không giảm sau `early_stopping_rounds = 35` vòng.

**Step 4 — Huấn luyện Mô hình 3: LSTM (`LSTM(64) -> LSTM(32) -> Dense(1)` — Mục 8)**

Đưa Tensor 3D $\mathcal{X} \in \mathbb{R}^{B \times 24 \times 4}$ qua 2 tầng LSTM xếp chồng ([Machine Learning Mastery Multivariate LSTM](https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/)):
- Tầng 1 (`LSTM 64`): Xuất toàn bộ chuỗi trạng thái ẩn $\mathbf{H}^{(1)} \in \mathbb{R}^{B \times 24 \times 64}$ (`return_sequences=True`).
- Tầng 2 (`LSTM 32`): Cô đọng thông tin lịch sử 24 giờ thành vector ngữ cảnh cuối $\mathbf{h}_{24}^{(2)} \in \mathbb{R}^{B \times 32}$.
- Tầng đầu ra (`Dense 1`): Ánh xạ tuyến tính $\hat{y}_{t+1} = \mathbf{W}_d \mathbf{h}_{24}^{(2)} + b_d$, tối ưu hàm mất mát **MSE** bằng **Adam** ($\eta = 0.001$) kèm `EarlyStopping(patience=10)` trên tập `Validation`.

**Step 5 — Huấn luyện Mô hình 4: DCRNN trên PEMS-BAY (Mục 12 Nâng cao)**

Huấn luyện kiến trúc **Encoder–Decoder DCGRU** ([Li et al., ICLR 2018](https://arxiv.org/abs/1707.01926)) trên `data/raw/pems-bay.h5` và `data/raw/sensor_graph/adj_mx_bay.pkl` với kỹ thuật **Scheduled Sampling** (xác suất dùng nhãn thật $\epsilon_i$ giảm dần theo số vòng lặp huấn luyện) để mô hình học cách tự sửa lỗi khi dự báo đa bước tới 15, 30 và 60 phút.

## 4. Evaluation metrics — Các chỉ số giám sát quá trình huấn luyện & hội tụ

Trong suốt giai đoạn huấn luyện (cả trên nhánh `70/15/15` và `5-Fold CV`), hệ thống giám sát trực tiếp sự hội tụ thông qua 5 chỉ số chuẩn (`MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²`) trên tập `Train` và tập `Validation`:

| Chỉ số Giám sát Huấn luyện | Công thức / Thước đo | Ý nghĩa Kiểm định Chất lượng Huấn luyện |
|---|---|---|
| **Train vs. Validation MSE (`Loss`)** | $\mathcal{L}_{\text{MSE}} = \frac{1}{N}\sum(y_t - \hat{y}_t)^2$ | Hàm mục tiêu tối ưu trực tiếp của `LSTM` và `XGBoost`; khoảng cách $|\text{Train MSE} - \text{Val MSE}|$ dùng để kiểm tra hiện tượng Overfitting (Câu hỏi 5 Mục 11) |
| **Validation RMSE & MAE** | $\sqrt{\text{MSE}}$ & $\frac{1}{N}\sum|y_t - \hat{y}_t|$ | Tiêu chí chọn `best_epoch` (cho `LSTM` / `DCRNN`) và `best_iteration` (cho `XGBoost`) |
| **Validation MAPE (%) & $R^2$ Score** | Sai số phần trăm & hệ số xác định trên tập `Validation` | Đảm bảo mô hình đạt độ khớp cao trước khi chuyển sang giai đoạn suy luận trên tập `Test` |
| **5-Fold CV Training Stability** | `Mean ± Std` của 5 chỉ số qua 5 Folds | Xác nhận mô hình không bị phụ thuộc vào một giai đoạn thời gian duy nhất |

## 5. Used data — Dữ liệu sử dụng

| Tập dữ liệu | Vai trò trong `Training_Phase` | Cơ chế Chống Overfitting & Rò rỉ | Module Thực thi |
|---|---|---|---|
| **Electricity Train (`70%`) & Val (`15%`)** | Huấn luyện trọng số chính và giám sát dừng sớm cho `VAR`, `XGBoost`, `LSTM` | Tập `Validation (15%)` tách biệt hoàn toàn với tập `Test (15%)` | `src/lab04/model.py`, `src/lab04/train.py` |
| **5-Fold `TimeSeriesSplit` (`Fold 1..5`)** | Huấn luyện lặp lại 5 lần trên các cửa sổ thời gian mở rộng | Giải phóng VRAM (`empty_cache`) và khởi tạo lại trọng số mới ở đầu mỗi Fold | `src/lab04/train.py`, `src/lab04/evaluate.py` |
| **PEMS-BAY Train (`70%`) & Val (`15%`)** | Huấn luyện mạng `DCRNN` cùng ma trận kề `adj_mx_bay.pkl` | Áp dụng `Scheduled Sampling` và `Gradient Clipping (max_grad_norm = 5.0)` | `src/lab04/model.py`, `src/lab04/train.py` |
