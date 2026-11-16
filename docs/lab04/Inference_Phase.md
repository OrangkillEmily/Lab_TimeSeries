# Giai đoạn Suy luận Dự báo & Kiểm thử Cuốn chiếu (`Inference_Phase.md` — Inference & Walk-Forward Backtesting)

## 1. Purpose — Mục đích

**Giai đoạn Suy luận và Kiểm thử Cuốn chiếu (`Inference_Phase`)** thực hiện **Mục 9 (Backtesting / Walk-forward Validation)** và **Mục 12 (Multi-horizon Traffic Inference 15, 30, 60 phút)** trong đề bài `practice 4.pdf` kết hợp lý thuyết **Slide 4, Slide 9 & Slide 10 Chương 4–5** (`chapter4-5.pdf`).

Mục đích cốt lõi của giai đoạn này là:
1. Thực hiện dự báo bước tiếp theo ($\hat{y}_{t+1} = \widehat{\text{load}}_{t+1}$) trên tập `Hold-out Test (15% — 5,260 giờ)` độc lập cho cả 3 mô hình **VAR**, **XGBoost**, **Residual LSTM**.
2. Triển khai quy trình **Walk-Forward Backtesting cuốn chiếu 168 giờ (1 tuần trọn vẹn)** đúng theo sơ đồ đề bài:
   $$\text{Train} \longrightarrow \text{Predict } t+1 \longrightarrow \text{Thêm quan sát vào chuỗi} \longrightarrow \text{Predict } t+2 \longrightarrow \dots$$
   nhằm mô phỏng chính xác cách hệ thống dự báo phụ tải điện vận hành trực tuyến theo từng giờ.
3. Thực hiện suy luận đa tầm nhìn (**15 phút $t+3$, 30 phút $t+6$, 60 phút $t+12$**) cho cả 3 mô hình **DCRNN** ([Li et al., ICLR 2018](https://arxiv.org/abs/1707.01926)), **Multi-Horizon TCN** và **Multi-Horizon LSTM** trên tập kiểm tra `1,285` cửa sổ $\times$ `207` trạm cảm biến của [PEMS-BAY Dataset](https://drive.google.com/file/d/1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq/view?usp=drive_link).
4. Biến đổi ngược (`inverse_transform`) toàn bộ giá trị dự báo từ không gian Z-Score về đơn vị vật lý gốc (`kW` và `mph`) trước khi tính 5 chỉ số đánh giá.

## 2. Input-Output

### Input

| Đầu vào (Input) | Định dạng & Kích thước | Mô tả Chi tiết |
|---|---|---|
| Trọng số 6 mô hình đã huấn luyện | Trained Model Artifacts | Gồm `VAR(24)`, `XGBoost(449 trees)`, `Residual LSTM`, `DCRNN`, `Multi-Horizon TCN`, `Multi-Horizon LSTM` từ `Training_Phase` |
| Tham số `StandardScaler` | $\boldsymbol{\mu}_{\text{train}}, \boldsymbol{\sigma}_{\text{train}}$ | Bộ tham số chuẩn hóa khớp trên `70%` tập Train ($\mu_{\text{load}} = 244,810.58\text{ kW}$, $\sigma_{\text{load}} = 84,767.53\text{ kW}$) để khôi phục đơn vị gốc |
| `X_xgb_test`, `X_lstm_test` | `(5260, 105)` & `(5260, 24, 4)` | Tập kiểm tra độc lập (`15%` cuối chuỗi: `2014-05-26 21:00` đến `2015-01-01 00:00`) từ [Electricity Load Diagrams Dataset](https://www.kaggle.com/datasets/jacopoferretti/electricity-load-diagrams-2011-2014-feature-engin) |
| `pems_X_test` & `adj_mx_bay.pkl` | `(1285, 12, 207)` & `(207, 207)` | Tập kiểm tra tốc độ giao thông và ma trận kề từ [DCRNN Repository](https://github.com/liyaguang/dcrnn) |

### Output

| Đầu ra (Output) | Đường dẫn Lưu trữ | Mô tả Chi tiết |
|---|---|---|
| `holdout_test_metrics.csv` | `reports/lab04/metrics/holdout_test_metrics.csv` | Bảng kết quả suy luận trên `5,260` giờ tập Test kèm thời gian suy luận (`Pred_Time_Sec`) và thống kê phần dư (`Mean_Residual`, `Std_Residual`, `Max_Error`) |
| `walk_forward_metrics.csv` | `reports/lab04/metrics/walk_forward_metrics.csv` | Kết quả dự báo cuốn chiếu `168` giờ theo quy trình Walk-Forward Backtesting (Mục 9) |
| `pems_bay_dcrnn_metrics.csv` & `pems_bay_benchmark_comparison.csv` | `reports/lab04/metrics/` | Bảng kết quả dự báo tốc độ xe (`mph`) của `DCRNN`, `Multi-Horizon TCN` và `Multi-Horizon LSTM` tại `15_min`, `30_min`, `60_min` và `Average_All_Horizons` |
| `Forecast_Overlay_168h.png`, `actual_vs_predicted_scatter.png`, `Walk_Forward_Trajectory.png` | `reports/lab04/figures/` | Bộ biểu đồ đối chiếu đường thực tế vs. dự báo 168 giờ, biểu đồ phân tán `Actual vs. Predicted` và quỹ đạo Walk-Forward Backtesting |

## 3. How to do — Quy trình thực hiện & Cơ sở toán học

### Sơ đồ luồng suy luận & Walk-Forward Backtesting (Mục 9)

```text
  Lịch sử Chuỗi Thời gian tính đến thời điểm t: H_t = [X_1, X_2, ..., X_t]
                                    |
                                    v
  [Bước 1] Trích xuất cửa sổ 24 giờ gần nhất: W_t = X_{t-23 : t}
           - Nhánh VAR    : Lấy 24 bước sai phân gần nhất -> dự báo Delta X_{t+1} + X_t
           - Nhánh XGBoost: Tạo vector phẳng 105 đặc trưng tại mốc t -> suy luận trên GPU (0.118s)
           - Nhánh LSTM   : Đưa Tensor 3D (1, 24, 4) qua Residual LSTM -> suy luận trên GPU (0.775s)
                                    |
                                    v
  [Bước 2] Biến đổi ngược về đơn vị vật lý gốc (Inverse StandardScaler):
           load_hat_{t+1} (kW) = y_hat_{t+1}^{(scaled)} * 84,767.5263 + 244,810.5838
                                    |
                                    v
  [Bước 3] Walk-Forward Update (Trong cửa sổ cuốn chiếu 168 giờ):
           Cập nhật quan sát thực tế X_{t+1} vào lịch sử: H_{t+1} = H_t U {X_{t+1}}
           Dịch chuyển cửa sổ sang t+1 -> Dự báo bước tiếp theo t+2 (tới đủ 168 bước)
```

### Các bước thực hiện chi tiết

**Step 1 — Suy luận Bước Tiếp theo ($\hat{y}_{t+1}$) trên `5,260` Giờ Tập Hold-out Test**

Tại mỗi mốc thời gian $t$ trong `5,260` giờ kiểm thử:
1. **Mô hình VAR (`Pred_Time = 0.2445s` trên `cpu`):** Dự báo bậc sai phân $\widehat{\Delta\mathbf{X}}_{t+1}^{\text{scaled}} = \sum_{i=1}^{24}\hat{\mathbf{A}}_i \Delta\mathbf{X}_{t+1-i}^{\text{scaled}}$, sau đó cộng hoàn nguyên vào mức gốc $\mathbf{X}_t^{\text{scaled}}$:
   $$\hat{y}_{t+1}^{\text{VAR, scaled}} = x_{\text{load}, t}^{\text{scaled}} + \left[\widehat{\Delta\mathbf{X}}_{t+1}^{\text{scaled}}\right]_{\text{load}}$$
2. **Mô hình XGBoost (`Pred_Time = 0.1180s` trên `cuda`):** Đưa vector phẳng $\mathbf{z}_t \in \mathbb{R}^{105}$ qua `449` cây quyết định tăng cường để thu được $\hat{y}_{t+1}^{\text{XGB, scaled}}$ với tốc độ suy luận nhanh nhất trong cả 3 mô hình.
3. **Mô hình Residual LSTM (`Pred_Time = 0.7753s` trên `cuda`):** Đưa tensor $\mathcal{X}_{t-23:t} \in \mathbb{R}^{1 \times 24 \times 4}$ qua mạng `Stacked 2-Layer LSTM + Autoregressive Residual Highway` ở chế độ `model.eval()` (`torch.no_grad()`).

**Step 2 — Quy trình Walk-Forward Backtesting Cuốn chiếu 168 Giờ (Mục 9)**

Trên cửa sổ kiểm thử cuốn chiếu `Eval_Steps = 168 giờ` (1 tuần liên tục):
- Ở mỗi bước $t$, mô hình dự báo $\hat{y}_{t+1}$, sau đó quan sát thực tế $\mathbf{X}_{t+1}$ được nối trực tiếp vào bộ đệm lịch sử $\mathcal{H}_{t+1} = \mathcal{H}_t \cup \{\mathbf{X}_{t+1}\}$ để cập nhật tức thời các biến trễ và đặc trưng trượt cho bước dự báo $t+2$.
- Thực nghiệm cho thấy cả 3 mô hình đều bám sát đường cong phụ tải thực tế trong suốt 168 giờ: `XGBoost` dẫn đầu với `RMSE = 4,870.64 kW` (`MAPE = 1.64%`, $R^2 = 0.9950$), theo sau là `VAR` (`RMSE = 5,282.87 kW`, `MAPE = 1.82%`, $R^2 = 0.9941$) và `Residual LSTM` (`RMSE = 5,925.55 kW`, `MAPE = 1.99%`, $R^2 = 0.9926$).

**Step 3 — Suy luận Đa tầm nhìn (`15m, 30m, 60m`) trên `207` Trạm Cảm biến PEMS-BAY (Mục 12)**

Trên `1,285` cửa sổ kiểm tra của `PEMS-BAY`, cả 3 mô hình (`DCRNN`, `Multi-Horizon TCN`, `Multi-Horizon LSTM`) nhận `12` bước quá khứ (`60 phút`) và xuất dự báo đồng thời cho `12` bước tương lai, trích xuất kết quả tại:
- **Tầm nhìn 15 phút (`Steps_Ahead = 3`)**: `DCRNN` đạt `MAE = 1.3094 mph` ($R^2 = 0.9196$), vượt qua `TCN` (`1.3543 mph`) và `LSTM` (`1.4182 mph`).
- **Tầm nhìn 30 phút (`Steps_Ahead = 6`)**: `DCRNN` đạt `MAE = 1.7338 mph` ($R^2 = 0.8372$), vượt qua `TCN` (`1.8553 mph`) và `LSTM` (`1.8885 mph`).
- **Tầm nhìn 60 phút (`Steps_Ahead = 12`)**: `DCRNN` đạt `MAE = 2.3040 mph` ($R^2 = 0.7063$), bỏ xa `LSTM` (`2.4986 mph`) và `TCN` (`2.6243 mph`).

## 4. Evaluation metrics — Các chỉ số đánh giá hiệu năng Suy luận & Backtesting

| Chế độ Suy luận / Kiểm thử | Mô hình Dẫn đầu & Chỉ số Thực nghiệm | Ý nghĩa trong Vận hành Thực tế & Backtesting |
|---|---|---|
| **Hold-out Test (`5,260 giờ`)** | `XGBoost`: `RMSE = 6,482.45 kW`, `MAE = 4,337.53 kW`, `MAPE = 1.89%`, $R^2 = 0.9946$ | Đánh giá độ chính xác tổng thể trên 7 tháng dữ liệu kiểm tra độc lập |
| **Walk-Forward Backtesting (`168 giờ`)** | `XGBoost`: `RMSE = 4,870.64 kW`, `MAPE = 1.64%`, $R^2 = 0.9950$ (`VAR`: $R^2 = 0.9941$) | Chứng minh khả năng thích nghi cuốn chiếu từng giờ với sai số cực đại thấp (`Max_Error = 17,249.55 kW`) |
| **Multi-Horizon Degradation (`15m -> 30m -> 60m`)** | `DCRNN`: `MAE` tăng từ `1.3094` $\to$ `1.7338` $\to$ `2.3040 mph` | Minh chứng thực nghiệm cho **Slide 4 Chương 4–5**: sai số tích lũy tăng dần theo tầm nhìn dự báo nhưng được kìm hãm hiệu quả nhờ đồ thị không gian |
| **Tổng thời gian Suy luận (`Pred_Time_Sec`)** | `XGBoost`: `0.1180s` $\mid$ `VAR`: `0.2445s` $\mid$ `LSTM`: `0.7753s` (trên toàn bộ `5,260` mẫu) | Tốc độ suy luận $< 0.15\text{ ms/mẫu}$, đáp ứng hoàn toàn yêu cầu dự báo thời gian thực |

## 5. Used data — Dữ liệu sử dụng

| Tập dữ liệu | Phạm vi Sử dụng trong `Inference_Phase` | Vai trò Kiểm thử | Module Thực thi |
|---|---|---|---|
| **Electricity Test Set (`5,260 giờ`)** | `15%` cuối chuỗi của `data/raw/electricity_load.csv` | Suy luận Hold-out và chạy Walk-Forward Backtesting `168` giờ cho `VAR`, `XGBoost`, `Residual LSTM` | `src/lab04/evaluate.py`, `src/lab04/visualize.py` |
| **5-Fold Validation Slices (`Val_1..Val_5`)** | 5 tập kiểm định (`5,844 giờ / Fold`) của `TimeSeriesSplit(n_splits=5)` | Suy luận trên từng Fold để tổng hợp bảng `cv_5fold_summary.csv` (`Mean ± Std`) | `src/lab04/evaluate.py` |
| **PEMS-BAY Test Set (`1,285 cửa sổ × 207 trạm`)** | `15%` cuối chuỗi từ `data/raw/pems-bay.h5` + `data/raw/sensor_graph/adj_mx_bay.pkl` | Suy luận đa chân trời 15, 30, 60 phút cho bộ 3 `DCRNN`, `Multi-Horizon TCN` và `Multi-Horizon LSTM` | `src/lab04/evaluate.py`, `src/lab04/visualize.py` |
