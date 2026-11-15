# Giai đoạn Suy luận Dự báo & Kiểm thử Cuốn chiếu (`Inference_Phase.md` — Inference & Walk-Forward Backtesting)

## 1. Purpose — Mục đích

**Giai đoạn Suy luận và Kiểm thử Cuốn chiếu (`Inference_Phase`)** thực hiện **Mục 9 (Backtesting / Walk-forward Validation)** và **Mục 12 (Multi-horizon Traffic Inference 15, 30, 60 phút)** trong đề bài `practice 4.pdf` kết hợp lý thuyết **Slide 4 & Slide 9 Chương 4–5** (`chapter4-5.pdf`).

Mục đích cốt lõi của giai đoạn này là:
1. Thực hiện dự báo bước tiếp theo ($\hat{y}_{t+1} = \widehat{\text{load}}_{t+1}$) trên tập `Test (15%)` độc lập cho cả 3 mô hình **VAR**, **XGBoost**, **LSTM**.
2. Triển khai quy trình **Backtesting (Walk-Forward Validation)** đúng theo sơ đồ đề bài:
   $$\text{Train} \longrightarrow \text{Predict } t+1 \longrightarrow \text{Thêm quan sát vào chuỗi} \longrightarrow \text{Predict } t+2 \longrightarrow \dots$$
   nhằm mô phỏng chính xác cách hệ thống dự báo phụ tải điện vận hành liên tục theo từng giờ trong thực tế.
3. Thực hiện suy luận đa bước (**Multi-step Autoregressive Seq2Seq Inference**) với mô hình **DCRNN** ([Li et al., ICLR 2018](https://arxiv.org/abs/1707.01926)) trên tập kiểm tra của [PEMS-BAY Dataset](https://drive.google.com/file/d/1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq/view?usp=drive_link) để xuất dự báo tốc độ xe ở 3 chân trời tương lai: **15 phút ($t+3$)**, **30 phút ($t+6$)** và **60 phút ($t+12$)**.
4. Biến đổi ngược (`inverse_transform`) toàn bộ giá trị dự báo từ không gian chuẩn hóa Z-score về đơn vị vật lý gốc trước khi chuyển sang giai đoạn tính 5 chỉ số đánh giá.

## 2. Input-Output

### Input

| Đầu vào (Input) | Định dạng & Kích thước | Mô tả Chi tiết |
|---|---|---|
| `models/lab04/var_model.pkl`, `xgboost_model.pkl`, `lstm_model.pt`, `dcrnn_pems_bay.pt` | Trained Model Artifacts | Các mô hình đã huấn luyện và khóa trọng số từ `Training_Phase` |
| `models/lab04/standard_scaler.pkl` | `StandardScaler` | Bộ chuẩn hóa chứa $\boldsymbol{\mu}_{\text{train}}$ và $\boldsymbol{\sigma}_{\text{train}}$ để khôi phục đơn vị gốc |
| `test_scaled` & lịch sử liền kề trước Test | `np.ndarray` `(T_test, 4)` | Tập kiểm tra độc lập (15% cuối chuỗi) từ [Electricity Load Diagrams Dataset](https://www.kaggle.com/datasets/jacopoferretti/electricity-load-diagrams-2011-2014-feature-engin) |
| `pems_test_loader` & `adj_mx_bay.pkl` | `DataLoader` & `np.ndarray` | Tập kiểm tra tốc độ giao thông và ma trận kề từ [DCRNN Repository](https://github.com/liyaguang/dcrnn) |

### Output

| Đầu ra (Output) | Đường dẫn Lưu trữ | Mô tả Chi tiết |
|---|---|---|
| `test_predictions_latest.csv` | `reports/lab04/predictions/test_predictions_latest.csv` | Bảng đối chiếu chi tiết theo từng giờ trên tập Test: `datetime`, `actual_load`, `var_pred`, `xgboost_pred`, `lstm_pred` và phần dư (`residual`) của từng mô hình |
| `walk_forward_predictions.csv` | `reports/lab04/predictions/walk_forward_predictions.csv` | Chuỗi kết quả dự báo cuốn chiếu từng bước theo quy trình Walk-Forward Backtesting (Mục 9) |
| `pems_bay_horizon_predictions.npz` | `reports/lab04/predictions/pems_bay_horizon_predictions.npz` | Mảng dự đoán tốc độ xe (mph) của `DCRNN` tại 3 chân trời `15_min_ahead`, `30_min_ahead`, `60_min_ahead` |
| `actual_vs_predicted_all.png` & `walk_forward_backtest.png` | `reports/figures/lab04/` | Biểu đồ so sánh đường thực tế và đường dự báo của `VAR`, `XGBoost`, `LSTM` trên tập Test và cửa sổ Backtesting |

## 3. How to do — Quy trình thực hiện & Cơ sở toán học

### Sơ đồ luồng suy luận & Walk-Forward Backtesting (Mục 9)

```text
  Lịch sử Chuỗi Thời gian tính đến thời điểm t: H_t = [X_1, X_2, ..., X_t]
                                    |
                                    v
  [Bước 1] Trích xuất cửa sổ 24 giờ gần nhất: W_t = X_{t-23 : t}
           - Nhánh VAR    : Lấy p* bước sai phân gần nhất -> forecast 1 bước vector 4 biến
           - Nhánh XGBoost: Tạo vector phẳng 96 lags + Rolling(6h, 24h) + Trend tại mốc t
           - Nhánh LSTM   : Tạo Tensor 3D (1, 24, 4) đưa lên GPU/CPU -> suy luận y_hat_{t+1}
                                    |
                                    v
  [Bước 2] Biến đổi ngược về đơn vị gốc (Inverse StandardScaler):
           load_hat_{t+1} = y_hat_{t+1}^{(scaled)} * sigma_{load,train} + mu_{load,train}
                                    |
                                    v
  [Bước 3] Walk-Forward Update (Khi bước thời gian t+1 kết thúc trong thực tế):
           Cập nhật quan sát mới X_{t+1} vào lịch sử: H_{t+1} = H_t U {X_{t+1}}
           Dịch chuyển cửa sổ sang t+1 -> Dự báo bước tiếp theo t+2
```

### Các bước thực hiện chi tiết

**Step 1 — Suy luận Bước Tiếp theo ($\hat{y}_{t+1}$) cho 3 Mô hình Chính**

Tại mỗi mốc thời gian $t$ trong tập kiểm thử:
1. **Mô hình VAR:** Dự báo toàn bộ vector 4 biến ở bước $t+1$ (nếu huấn luyện trên chuỗi sai phân bậc 1 thì cộng hoàn nguyên $\hat{\mathbf{X}}_{t+1}^{\text{scaled}} = \mathbf{X}_t^{\text{scaled}} + \sum_{i=1}^{p^*}\hat{\mathbf{A}}_i \Delta\mathbf{X}_{t+1-i}^{\text{scaled}}$), sau đó chỉ trích xuất phần tử đầu tiên ứng với cột `load`:
   $$\hat{y}_{t+1}^{\text{VAR, scaled}} = \left[\hat{\mathbf{X}}_{t+1}^{\text{scaled}}\right]_{\text{load}}$$
2. **Mô hình XGBoost:** Tính vector đặc trưng phẳng $\mathbf{z}_t = \text{Concat}(\text{Lags}_{24}, \text{RM}_6, \text{RSTD}_{24}, \text{Trend}_{24})$ từ cửa sổ lịch sử tính đến $t$ và tổng hợp dự đoán từ $M^*$ cây quyết định:
   $$\hat{y}_{t+1}^{\text{XGB, scaled}} = \sum_{m=1}^{M^*} f_m(\mathbf{z}_t)$$
3. **Mô hình LSTM:** Đưa tensor $\mathcal{X}_{t-23:t} \in \mathbb{R}^{1 \times 24 \times 4}$ qua mạng `LSTM(64) -> LSTM(32) -> Dense(1)` ở chế độ `model.eval()` (tắt Dropout, tắt tính toán gradient `torch.no_grad()`) để thu được $\hat{y}_{t+1}^{\text{LSTM, scaled}}$.

**Step 2 — Quy trình Walk-Forward Backtesting Cuốn chiếu (Mục 9)**

Để mô phỏng sát nhất môi trường vận hành dự báo phụ tải điện thực tế:
- Hệ thống bắt đầu từ ranh giới kết thúc tập Validation, dự báo bước $t+1$.
- Ngay sau khi dự báo $\hat{y}_{t+1}$, quan sát thực tế tại $t+1$ được nối (`append`) vào bộ đệm lịch sử $\mathcal{H}_{t+1} = \mathcal{H}_t \cup \{\mathbf{X}_{t+1}\}$ để cập nhật cửa sổ trễ và các chỉ báo trượt (`rolling_mean_6h`, `rolling_std_24h`, `load_trend_24h`) cho bước dự báo $t+2$.
- Định kỳ sau mỗi chu kỳ `refit_every_hours = 24` giờ, bộ dự báo thống kê `VAR` cập nhật lại ma trận hệ số trên lịch sử mở rộng để thích nghi với sự dịch chuyển xu hướng ngắn hạn.

**Step 3 — Suy luận Đa chân trời (`15m, 30m, 60m`) với DCRNN trên PEMS-BAY (Mục 12)**

Trong giai đoạn suy luận của **DCRNN**, nhánh **Encoder** đọc 12 bước tốc độ quá khứ $[\mathbf{X}_{t-11}, \dots, \mathbf{X}_t]$ trên đồ thị $\mathcal{G}$ để nén thành trạng thái ẩn $\mathbf{H}_t$. Nhánh **Decoder** tự hồi quy sinh tuần tự $\hat{\mathbf{X}}_{t+1}, \hat{\mathbf{X}}_{t+2}, \dots, \hat{\mathbf{X}}_{t+12}$ (trong lúc inference hoàn toàn dùng dự đoán của bước trước làm đầu vào cho bước sau, không dùng nhãn thật), và trích xuất kết quả tại $h \in \{3, 6, 12\}$ tương ứng **15, 30, 60 phút**.

**Step 4 — Biến đổi Ngược về Thang đo Vật lý Gốc (`Inverse Transformation`)**

Mọi dự đoán $\hat{y}_{t+1}^{\text{scaled}}$ đều được khôi phục về thang đo gốc trước khi lưu xuống CSV và tính sai số:

$$\hat{y}_{t+1} = \hat{y}_{t+1}^{\text{scaled}} \cdot \sigma_{\text{load, train}} + \mu_{\text{load, train}}$$

## 4. Evaluation metrics — Các chỉ số đánh giá hiệu năng Suy luận & Backtesting

Kết quả đầu ra của `Inference_Phase` được chấm điểm trực tiếp bằng bộ **5 chỉ số hồi quy chuẩn** và **độ trễ suy luận (Inference Latency)**:

| Chỉ số Đánh giá Suy luận | Công thức / Thước đo | Ý nghĩa trong Vận hành Thực tế & Backtesting |
|---|---|---|
| **Walk-Forward MSE & RMSE (↓)** | $\text{MSE} = \frac{1}{N}\sum(y_t - \hat{y}_t)^2$, $\text{RMSE} = \sqrt{\text{MSE}}$ | Đo mức độ sai lệch năng lượng tại các giờ cao điểm khi chạy cuốn chiếu từng bước |
| **Walk-Forward MAE & MAPE (%) (↓)** | $\text{MAE} = \frac{1}{N}\sum|y_t - \hat{y}_t|$, $\text{MAPE} = \frac{100\%}{N}\sum\left|\frac{y_t - \hat{y}_t}{y_t}\right|$ | Thước đo sai số phần trăm vận hành trực quan cho điều độ lưới điện và điều tiết giao thông |
| **Walk-Forward $R^2$ Score (↑)** | $R^2 = 1 - \frac{\sum(y_t - \hat{y}_t)^2}{\sum(y_t - \bar{y})^2}$ | Đánh giá tỷ lệ phương sai thực tế được mô hình nắm bắt trong suốt chuỗi kiểm thử cuốn chiếu |
| **Multi-Horizon Degradation (`15m -> 30m -> 60m`)** | Tốc độ tăng của `MAE`, `RMSE`, `MAPE` khi $h$ tăng từ $3 \to 6 \to 12$ | Minh chứng thực nghiệm cho **Slide 4 Chương 4–5**: sai số tích lũy khi dự báo xa nhiều bước |
| **Inference Latency (ms/sample)** | Thời gian chạy trung bình trên 1 bước dự báo | Đảm bảo khả năng đáp ứng thời gian thực (Real-time Forecasting) trên cả GPU và CPU |

## 5. Used data — Dữ liệu sử dụng

| Tập dữ liệu | Phạm vi Sử dụng trong `Inference_Phase` | Vai trò Kiểm thử | Module Thực thi |
|---|---|---|---|
| **Electricity Test Set (`15%` cuối chuỗi)** | Toàn bộ các mẫu kiểm tra của `data/raw/electricity_load.csv` | Suy luận độc lập và chạy Walk-Forward Backtesting cho `VAR`, `XGBoost`, `LSTM` | `src/lab04/evaluate.py`, `src/lab04/visualize.py` |
| **5-Fold Validation Slices (`Val_1..Val_5`)** | 5 tập kiểm định của `TimeSeriesSplit(n_splits=5)` | Suy luận trên từng Fold để tổng hợp bảng `Mean ± Std` của 5 chỉ số đánh giá | `src/lab04/evaluate.py` |
| **PEMS-BAY Test Set (`15%` cuối chuỗi)** | Cửa sổ kiểm tra từ `data/raw/pems-bay.h5` + `data/raw/sensor_graph/adj_mx_bay.pkl` | Suy luận đa chân trời 15, 30, 60 phút bằng mô hình `DCRNN` | `src/lab04/evaluate.py`, `src/lab04/visualize.py` |
