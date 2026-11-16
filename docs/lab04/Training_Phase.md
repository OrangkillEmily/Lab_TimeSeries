# Giai đoạn Huấn luyện Đa Mô hình & Tự động Tăng tốc GPU (`Training_Phase.md`)

## 1. Purpose — Mục đích

**Giai đoạn Huấn luyện (`Training_Phase`)** thực hiện **Mục 6 (Huấn luyện VAR)**, **Mục 7 (Huấn luyện XGBoost)**, **Mục 8 (Huấn luyện LSTM)** và **Mục 12 (Huấn luyện mở rộng trên PEMS-BAY)** trong `practice 4.pdf`, kết hợp **Yêu cầu bổ sung số 4 (Huấn luyện trên cả `Train-Val-Test` và `5-Fold Cross-Validation`)** cùng **Yêu cầu bổ sung số 5 (Tự động chuyển sang chạy bằng GPU khi máy có GPU)**.

Mục đích của giai đoạn này là ước lượng bộ tham số tối ưu cho toàn bộ **6 mô hình** trong pipeline:
1. **Nhóm Phụ tải Điện & Thời tiết (Mục 6–11)**: `VAR(24)` (trên miền sai phân bậc 1), `XGBoost` (`105` đặc trưng, GPU `cuda`), và `Stacked 2-Layer LSTM + Autoregressive Residual Highway` (trên GPU `cuda` với `AMP FP16`).
2. **Nhóm Giao thông Đồ thị PEMS-BAY (Mục 12 Nâng cao)**: Huấn luyện đồng bộ cả 3 kiến trúc **`DCRNN` (Graph Diffusion + GRU)**, **`Multi-Horizon TCN` (Causal Dilated Conv1D — Slide 10 Chương 4–5)** và **`Multi-Horizon LSTM` (Temporal Only)** để đối sánh công bằng ở cả 3 tầm nhìn `15 phút`, `30 phút`, `60 phút`.

## 2. Input-Output

### Input

| Đầu vào (Input) | Định dạng & Kích thước | Mô tả Chi tiết |
|---|---|---|
| `train_stationary` (`VAR`) | `pd.DataFrame` `(24544, 4)` | Chuỗi đa biến 4 cột đã chuẩn hóa và lấy sai phân bậc 1 đảm bảo tính dừng ($p\text{-value ADF} = 0.0000 < 0.05$) |
| `X_xgb_train`, `y_train`, `X_xgb_val`, `y_val` | `np.ndarray` `(24521, 105)` & `(5260, 105)` | Ma trận `105` đặc trưng phẳng (`96` Lags + Rolling + Trend + Cyclical) từ [Electricity Load Diagrams Dataset](https://www.kaggle.com/datasets/jacopoferretti/electricity-load-diagrams-2011-2014-feature-engin) |
| `X_lstm_train`, `y_train`, `X_lstm_val`, `y_val` | `Tensor` `(24521, 24, 4)` & `(5260, 24, 4)` | Tensor 3D chuỗi 24 giờ $\times$ 4 biến dùng để huấn luyện mạng `Stacked 2-Layer LSTM + Residual Highway` |
| `pems_X_train`, `pems_X_val`, `adj_mx` | `(6025, 12, 207)`, `(1285, 12, 207)` & `(207, 207)` | Dữ liệu cửa sổ 12 bước tốc độ xe từ [PEMS-BAY Dataset](https://drive.google.com/file/d/1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq/view?usp=drive_link) và đồ thị cảm biến từ [DCRNN GitHub](https://github.com/liyaguang/dcrnn) |
| `config.yaml` (`hardware`, `models`) | `Dict[str, Any]` | Cấu hình siêu tham số huấn luyện và cờ điều phối phần cứng `device = "auto"`, `amp_enabled = true` |

### Output

| Đầu ra (Output) | Đường dẫn Lưu trữ | Mô tả Chi tiết |
|---|---|---|
| `var_model.pkl` | `models/lab04/var_model.pkl` | Mô hình thống kê `VAR(p*=24)` đã khớp trên miền sai phân bậc 1 (`train_time = 3.07s` trên `cpu`) |
| `xgboost_model.pkl` / `.json` | `models/lab04/xgboost_model.pkl` | Mô hình `XGBRegressor` trên `cuda` tại vòng lặp tối ưu (`best_iteration = 449`, `train_time = 5.54s`) |
| `lstm_model.pt` | `models/lab04/lstm_model.pt` | Trọng số mạng `Residual LSTM` trên `cuda` tại epoch tối ưu (`best_epoch = 27`, `train_time = 92.20s`) |
| `dcrnn_model.pt`, `tcn_model.pt`, `pems_lstm_model.pt` | `models/lab04/` | Trọng số bộ 3 mô hình giao thông trên `cuda`: `DCRNN` (`best_epoch = 11`, `69.03s`), `Multi_Horizon_TCN` (`best_epoch = 7`, `15.15s`), `Multi_Horizon_LSTM` (`best_epoch = 4`, `8.24s`) |
| `train_summary.json` | `reports/lab04/metrics/train_summary.json` | Hồ sơ lưu lịch sử hội tụ `train_loss`, `val_loss`, thông tin thiết bị (`NVIDIA GeForce RTX 3050 6GB Laptop GPU`) và tổng thời gian huấn luyện (`432.39s`) |
| `training_loss_curves.png` & `training_time_comparison.png` | `reports/lab04/figures/` | Biểu đồ đường cong hội tụ hàm mất mát kiểm tra Overfitting và biểu đồ đối sánh thời gian huấn luyện / suy luận |

## 3. How to do — Quy trình thực hiện & Cơ sở toán học

### Sơ đồ luồng điều phối `Auto-GPU` và huấn luyện 6 mô hình

```text
                        Khởi động src/lab04/train.py (Run_ID)
                                      |
                                      v
      [get_compute_device] Phát hiện NVIDIA GeForce RTX 3050 6GB Laptop GPU
      +--> Bật torch_device="cuda", xgb_device="cuda", AMP_FP16=True
                                      |
        +-----------------------------+-----------------------------+
        |                             |                             |
        v                             v                             v
 [1. VAR(24) Baseline]         [2. XGBoost (105 feats)]      [3. Residual LSTM (DL)]
 - Miền sai phân bậc 1         - tree_method="hist", cuda    - Stacked 2-Layer LSTM (64->32)
 - Chọn bậc trễ p* = 24        - best_iteration = 449        - + Autoregressive Residual Highway
 - Thời gian: 3.07s (CPU)      - Thời gian: 5.54s (GPU)      - CosineAnnealingLR, best_epoch = 27
        |                             |                             |
        +-----------------------------+-----------------------------+
                                      |
                                      v
      [4. Benchmark Đa Tầm nhìn PEMS-BAY (15m, 30m, 60m trên 207 trạm)]:
      - DCRNN (Graph Diffusion + GRU) : best_epoch = 11 (69.03s trên GPU)
      - Multi-Horizon TCN (Dilated 1D): best_epoch = 7  (15.15s trên GPU)
      - Multi-Horizon LSTM (Temporal) : best_epoch = 4  ( 8.24s trên GPU)
```

### Các bước thực hiện chi tiết

**Step 1 — Tự động Nhận diện Phần cứng & Quy tắc Thứ tự Bước Tối ưu PyTorch (`Auto-GPU & AMP Safety`)**

Hàm `get_compute_device(cfg)` trong `src/lab04/utils.py` tự động nhận diện `NVIDIA GeForce RTX 3050 6GB Laptop GPU` và kích hoạt `torch.amp.autocast("cuda", dtype=torch.float16)` cùng `GradScaler`.
- **Quy tắc an toàn PyTorch Step Order:** Trong vòng lặp huấn luyện `AMP FP16`, cờ `optimizer_stepped` được khởi tạo tường minh ở mọi batch để kiểm tra xem `scaler.step(optimizer)` có thực sự cập nhật trọng số hay không (do `GradScaler` có thể bỏ qua bước cập nhật nếu phát hiện `Inf/NaN` gradients), đảm bảo `scheduler.step()` **chỉ được gọi sau khi `optimizer.step()` đã thực thi**, triệt tiêu hoàn toàn cảnh báo của PyTorch và lỗi `UnboundLocalError`.

**Step 2 — Huấn luyện Mô hình 1: VAR (`Vector Autoregression` — Mục 6)**

Trên chuỗi sai phân bậc 1 dừng $\Delta\mathbf{X}_t \in \mathbb{R}^{4}$, hệ thống khớp mô hình $\text{VAR}(p)$ với bậc trễ tối ưu $p^* = 24$ (bao phủ trọn vẹn chu kỳ ngày 24 giờ) theo tiêu chuẩn thông tin Akaike (**AIC**, [Statsmodels VAR](https://www.statsmodels.org/stable/vector_ar.html)):

$$\text{AIC}(p) = \ln\left|\hat{\boldsymbol{\Sigma}}_{\varepsilon}(p)\right| + \frac{2 p K^2}{T_{\text{train}}}$$

**Step 3 — Huấn luyện Mô hình 2: XGBoost (`Extreme Gradient Boosting` — Mục 7)**

Tối ưu hóa hàm mục tiêu bình phương sai số có điều chuẩn trên GPU (`device="cuda"`, `tree_method="hist"`) với `105` đặc trưng ([XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/tutorials/model.html)):

$$\mathcal{L}^{(m)} = \sum_{i=1}^{N_{\text{train}}} \left(y_i - \left(\hat{y}_i^{(m-1)} + f_m(\mathbf{x}_i)\right)\right)^2 + \gamma T_m + \frac{1}{2}\lambda \sum_{j=1}^{T_m} w_j^2$$

Mô hình hội tụ tại `best_iteration = 449` chỉ sau `5.54 giây`.

**Step 4 — Huấn luyện Mô hình 3: Stacked 2-Layer LSTM + Autoregressive Residual Highway (Mục 8)**

Đưa Tensor 3D $\mathcal{X} \in \mathbb{R}^{B \times 24 \times 4}$ qua 2 tầng LSTM xếp chồng (`64 -> 32`) kết hợp nhánh **Autoregressive Residual Highway** (ánh xạ tuyến tính trực tiếp chuỗi `load` trong cửa sổ 24h cộng vào đầu ra phi tuyến của khối LSTM) và lịch trình giảm tốc độ học `CosineAnnealingLR`. Mô hình đạt điểm dừng tối ưu tại `best_epoch = 27` (`92.20 giây` trên GPU).

**Step 5 — Huấn luyện Bộ 3 Mô hình Giao thông trên PEMS-BAY (`DCRNN`, `TCN`, `LSTM` — Mục 12 Nâng cao)**

Để chứng minh thuyết phục vai trò của cấu trúc đồ thị không gian và đối sánh với kiến trúc TCN trong **Slide 10 Chương 4–5**, pipeline huấn luyện đồng thời 3 mô hình dự báo đa tầm nhìn (`12` bước quá khứ $\to$ `12` bước tương lai trên `207` trạm cảm biến):
1. **`DCRNN` (Graph Diffusion + GRU)** ([Li et al., ICLR 2018](https://arxiv.org/abs/1707.01926)): Tích hợp phép tích chập khuếch tán hai chiều trên ma trận kề `adj_mx_bay.pkl` (`best_epoch = 11`, `69.03s`).
2. **`Multi-Horizon TCN` (Causal Dilated Conv1D)**: Sử dụng các tầng tích chập nhân quả giãn nở (*dilated causal convolutions*) có kết nối tắt residual (`best_epoch = 7`, `15.15s`).
3. **`Multi-Horizon LSTM` (Temporal Only)**: Mô hình tuần tự thuần thời gian không sử dụng ma trận kề đồ thị (`best_epoch = 4`, `8.24s`).

## 4. Evaluation metrics — Các chỉ số giám sát quá trình huấn luyện & hội tụ

| Mô hình Huấn luyện | Thiết bị (`Device`) | Thời gian Huấn luyện (`Sec`) | Mốc Hội tụ Tối ưu (`Best Epoch / Lag`) | Chỉ số Kiểm định `Validation` / `Test` Đạt được |
|---|---|---:|---:|---|
| **`VAR`** | `cpu` | `3.07s` | `selected_lag = 24` | `Val RMSE = 6,501.03 kW` $\mid$ `Test RMSE = 7,122.36 kW` ($R^2 = 0.9935$) |
| **`XGBoost`** | `cuda` | `5.54s` | `best_iteration = 449` | `Val RMSE = 5,604.27 kW` $\mid$ `Test RMSE = 6,482.45 kW` ($R^2 = 0.9946$) |
| **`Residual LSTM`** | `cuda` | `92.20s` | `best_epoch = 27` | `Val RMSE = 5,968.28 kW` $\mid$ `Test RMSE = 6,570.41 kW` ($R^2 = 0.9944$) |
| **`DCRNN` (`PEMS-BAY`)** | `cuda` | `69.03s` | `best_epoch = 11` | Trung bình 3 tầm nhìn: `MAE = 1.7824 mph`, `RMSE = 3.9916 mph`, $R^2 = 0.8210$ |
| **`Multi-Horizon TCN`** | `cuda` | `15.15s` | `best_epoch = 7` | Trung bình 3 tầm nhìn: `MAE = 1.9446 mph`, `RMSE = 4.4687 mph`, $R^2 = 0.7722$ |
| **`Multi-Horizon LSTM`** | `cuda` | `8.24s` | `best_epoch = 4` | Trung bình 3 tầm nhìn: `MAE = 1.9351 mph`, `RMSE = 4.0948 mph`, $R^2 = 0.8129$ |

## 5. Used data — Dữ liệu sử dụng

| Tập dữ liệu | Vai trò trong `Training_Phase` | Cơ chế Chống Overfitting & Rò rỉ | Module Thực thi |
|---|---|---|---|
| **Electricity Train (`24,521` mẫu) & Val (`5,260` mẫu)** | Huấn luyện trọng số chính và giám sát dừng sớm cho `VAR`, `XGBoost`, `Residual LSTM` | Tập `Validation (15%)` tách biệt hoàn toàn với tập `Test (15%)` | `src/lab04/model.py`, `src/lab04/train.py` |
| **5-Fold `TimeSeriesSplit` (`Fold 1..5`)** | Huấn luyện lặp lại 5 lần trên các cửa sổ mở rộng (`5,845 -> 29,221` giờ) + huấn luyện `Weather Ablation` | Khởi tạo lại trọng số và bộ chuẩn hóa mới ở đầu mỗi Fold | `src/lab04/train.py`, `src/lab04/evaluate.py` |
| **PEMS-BAY Train (`6,025` mẫu) & Val (`1,285` mẫu)** | Huấn luyện bộ 3 `DCRNN`, `Multi-Horizon TCN` và `Multi-Horizon LSTM` | Áp dụng `EarlyStopping`, `CosineAnnealingLR` và `Gradient Clipping` | `src/lab04/model.py`, `src/lab04/train.py` |
