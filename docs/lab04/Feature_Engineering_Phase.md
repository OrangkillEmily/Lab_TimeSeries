# Giai đoạn Kỹ thuật Đặc trưng Chuỗi Thời gian Đa biến (`Feature_Engineering_Phase.md`)

## 1. Purpose — Mục đích

**Giai đoạn Kỹ thuật Đặc trưng (`Feature_Engineering_Phase`)** thực hiện **Mục 5 (Feature Engineering cho ML và DL: 5.1 Cửa sổ trễ Lag Window $L=24$ & 5.2 Rolling Features cho XGBoost)** cùng **Mục 12 (Cửa sổ không gian – thời gian cho PEMS-BAY)** trong `practice 4.pdf` và lý thuyết **Slide 7–10 Chương 4–5** (`chapter4-5.pdf`).

Mục đích cốt lõi của giai đoạn này là chuyển đổi chuỗi thời gian đa biến liên tục $\mathbf{X}_t \in \mathbb{R}^{K}$ thành 3 cấu trúc biểu diễn chuyên biệt phù hợp với bản chất toán học của từng trường phái mô hình:
1. **Vector phẳng `105` đặc trưng (`96` Lags + Rolling + Trend + Cyclical Calendar)** dành cho **XGBoost** ([Kaggle XGBoost Multivariate TS](https://www.kaggle.com/code/davidandressanchez/xgboost-for-multivariate-ts)).
2. **Tensor 3D Chuỗi tuần tự `(N_samples, 24, 4)`** dành cho mạng **Stacked 2-Layer LSTM + Autoregressive Residual Highway** ([Machine Learning Mastery Multivariate LSTM](https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/)).
3. **Tensor Đa tầm nhìn `(N_bay, 12, 207)`** kết hợp ma trận chuyển tiếp bước ngẫu nhiên hai chiều từ `adj_mx_bay.pkl` (`207 × 207`) dành cho bộ 3 mô hình **DCRNN** ([Li et al., ICLR 2018](https://arxiv.org/abs/1707.01926)), **Multi-Horizon TCN** (Slide 10 Chương 4–5) và **Multi-Horizon LSTM**.

## 2. Input-Output

### Input

| Đầu vào (Input) | Định dạng & Kích thước | Mô tả Chi tiết |
|---|---|---|
| `train_scaled`, `val_scaled`, `test_scaled` | `np.ndarray` `(24545, 4)`, `(5260, 4)`, `(5260, 4)` | Chuỗi đa biến 4 cột `[load, temperature, humidity, wind_speed]` đã chuẩn hóa Z-Score từ [Electricity Load Diagrams Dataset](https://www.kaggle.com/datasets/jacopoferretti/electricity-load-diagrams-2011-2014-feature-engin) |
| `pems_scaled` & `adj_mx_bay.pkl` | `np.ndarray` `(8640, 207)` & `np.ndarray` `(207, 207)` | Chuỗi tốc độ xe đã chuẩn hóa từ [PEMS-BAY Dataset](https://drive.google.com/file/d/1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq/view?usp=drive_link) và ma trận kề từ [DCRNN Repository](https://github.com/liyaguang/dcrnn) |
| `config.yaml -> feature_engineering` | `Dict[str, Any]` | Cấu hình `lag_window = 24`, `num_tabular_features = 105`, `pems_lookback = 12`, `pems_horizons = [3, 6, 12]` |

### Output

| Đầu ra (Output) | Định dạng & Kích thước Thực tế | Mô tả Chi tiết |
|---|---|---|
| `X_xgb_train`, `X_xgb_val`, `X_xgb_test` | `(24521, 105)`, `(5260, 105)`, `(5260, 105)` | Ma trận phẳng `105` đặc trưng cho `XGBoost` gồm `96` biến trễ ($24\text{h} \times 4\text{ biến}$) + `9` đặc trưng động lượng trượt, xu hướng & lịch chu kỳ (`sin_hour`, `cos_hour`, `is_weekend`, v.v.) |
| `X_lstm_train`, `X_lstm_val`, `X_lstm_test` | `(24521, 24, 4)`, `(5260, 24, 4)`, `(5260, 24, 4)` | Tensor 3D chuẩn cho mạng `Residual LSTM` theo Mục 8.1 đề bài |
| `y_train`, `y_val`, `y_test` | `(24521,)`, `(5260,)`, `(5260,)` | Vector mục tiêu $\text{load}_{t+1}$ tương ứng với từng cửa sổ thời gian |
| `pems_X_train`, `pems_X_val`, `pems_X_test` | `(6025, 12, 207)`, `(1285, 12, 207)`, `(1285, 12, 207)` | Cửa sổ 12 bước quá khứ (60 phút) ánh xạ tới nhãn mục tiêu ở 15 phút ($t+3$), 30 phút ($t+6$) và 60 phút ($t+12$) cho `DCRNN`, `TCN` và `PEMS-LSTM` |
| `xgboost_feature_importance.csv` & `XGBoost_Feature_Importance.png` | `reports/lab04/metrics/` & `reports/lab04/figures/` | Bảng xếp hạng và biểu đồ Top 15 đặc trưng quan trọng nhất trong mô hình `XGBoost` |

## 3. How to do — Quy trình thực hiện & Cơ sở toán học

### Sơ đồ luồng kiến tạo đặc trưng (`Feature Engineering Workflow`)

```text
         Chuỗi Đa biến Đã Chuẩn hóa X_t = [load_t, temp_t, humidity_t, wind_t]
                                          |
             +----------------------------+----------------------------+
             |                                                         |
             v                                                         v
 [Nhánh A: Cửa sổ Trượt Tensor 3D cho LSTM]         [Nhánh B: Tabular 105 Features cho XGBoost]
 Trích xuất 24 bước liên tiếp [t-23 .. t]:          1. Làm phẳng 24 bước x 4 biến = 96 Lag Features
 Tensor X_lstm in R^(N_samples x 24 x 4)            2. Rolling Mean & Std (6h, 24h) + Load Trend 24h
 Nhãn mục tiêu: y = load_{t+1}                      3. Đặc trưng chu kỳ: sin_hour, cos_hour, is_weekend
                                                    -> Tổng cộng D_tabular = 105 đặc trưng
```

### Các bước thực hiện chi tiết

**Step 1 — Kiến tạo Cửa sổ Trễ Đa biến $L = 24$ giờ (Mục 5.1)**

Với mỗi thời điểm neo $t \ge 24$, mẫu đầu vào sử dụng trọn vẹn lịch sử 24 giờ gần nhất của cả 4 biến để dự báo $\text{load}_{t+1}$:
- **Dạng Tensor 3D cho Deep Learning (`Residual LSTM`):**
  $$\mathcal{X}_t^{(\text{LSTM})} = \begin{bmatrix} \text{load}_{t-23} & \text{temp}_{t-23} & \text{humidity}_{t-23} & \text{wind}_{t-23} \\ \vdots & \vdots & \vdots & \vdots \\ \text{load}_{t} & \text{temp}_{t} & \text{humidity}_{t} & \text{wind}_{t} \end{bmatrix} \in \mathbb{R}^{24 \times 4}$$
- **Dạng Vector Phẳng cho Machine Learning (`XGBoost`):**
  $$\mathbf{x}_t^{(\text{lag})} = \left[\text{load}_{\text{lag\_1h}}, \dots, \text{load}_{\text{lag\_24h}},\ \text{temperature}_{\text{lag\_1h}}, \dots, \text{wind\_speed}_{\text{lag\_24h}}\right] \in \mathbb{R}^{96}$$

**Step 2 — Kiến tạo `9` Đặc trưng Động lượng Trượt & Lịch Chu kỳ cho XGBoost (`96 + 9 = 105` Đặc trưng)**

Để bổ sung tri thức động lực học thời gian và nhịp sinh hoạt ngày/tuần cho mô hình cây quyết định (`XGBoost`), hệ thống kết hợp thêm `9` đặc trưng nâng cao:
1. **Rolling Mean & Rolling Std (`rolling_mean_6h`, `rolling_std_24h`):** Nắm bắt mức nền phụ tải ngắn hạn 6 giờ và biên độ dao động 24 giờ:
   $$\text{RM}_{6,t} = \frac{1}{6}\sum_{i=0}^{5} \text{load}_{t-i}, \qquad \text{RSTD}_{24,t} = \sqrt{\frac{1}{24}\sum_{i=0}^{23}\left(\text{load}_{t-i} - \overline{\text{load}}_{t-23:t}\right)^2}$$
2. **Load Trend 24 giờ (`load_trend_24h`):** Độ lệch xu hướng phụ tải hiện tại so với cùng giờ ngày hôm trước ($\text{load}_t - \text{load}_{t-24}$).
3. **Cyclical Calendar & Interaction Features:** Mã hóa lượng giác giờ trong ngày ($\sin\left(\frac{2\pi h}{24}\right), \cos\left(\frac{2\pi h}{24}\right)$), cờ cuối tuần (`is_weekend`), và tương tác chéo thời tiết – phụ tải, nâng tổng số chiều đầu vào của `XGBoost` lên đúng **`105` đặc trưng**.

**Step 3 — Kiến tạo Cửa sổ Đa bước & Ma trận Khuếch tán Đồ thị cho PEMS-BAY (Mục 12)**

Đối với bài toán nâng cao trên `data/raw/pems-bay.h5` (`8,640 × 207`) và `data/raw/sensor_graph/adj_mx_bay.pkl` (`207 × 207`):
- **Cửa sổ không gian – thời gian:** Lấy $L = 12$ bước quá khứ ($60\text{ phút}$) trên $N = 207$ trạm cảm biến, tạo ra `6,025` mẫu Train, `1,285` mẫu Val và `1,285` mẫu Test dùng chung cho cả 3 mô hình `DCRNN`, `Multi-Horizon TCN` và `Multi-Horizon LSTM`.
- **Toán tử chuyển tiếp đồ thị hai chiều cho DCRNN:** Từ ma trận kề có hướng $\mathbf{W} \in \mathbb{R}^{207 \times 207}$, tính ma trận bậc ra $\mathbf{D}_O = \text{diag}(\mathbf{W}\mathbf{1})$ và bậc vào $\mathbf{D}_I = \text{diag}(\mathbf{W}^T\mathbf{1})$ để tạo cặp ma trận khuếch tán xuôi dòng $\mathbf{P}_f = \mathbf{D}_O^{-1}\mathbf{W}$ và ngược dòng $\mathbf{P}_b = \mathbf{D}_I^{-1}\mathbf{W}^T$.

## 4. Evaluation metrics — Đánh giá hiệu quả của Feature Engineering qua 5 Chỉ số & Ablation Study

| Tiêu chí Đánh giá Đặc trưng | Kết quả Định lượng Thực nghiệm từ Pipeline | Tác động tới `MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²` |
|---|---|---|
| **Top Gain Feature Importance của `XGBoost`** | `load_lag_24h` (`0.6281`), `load_lag_23h` (`0.1898`), `load_lag_1h` (`0.0935`), `cos_hour` (`0.0064`), `humidity_lag_12h` (`0.0055`), `sin_hour` (`0.0045`), `is_weekend` (`0.0040`), `temperature_lag_10h` (`0.0022`) | Chứng minh chu kỳ tự hồi quy 24h kết hợp pha giờ trong ngày (`cos_hour`, `sin_hour`) và độ trễ thời tiết `10h–12h` đóng vai trò quyết định giúp `XGBoost` đạt $R^2 = 0.9946$ |
| **Nghiên cứu Loại bỏ Đặc trưng Thời tiết (`Weather Ablation`)** | `Residual LSTM`: giảm `RMSE` từ `6,946.54 kW` xuống `6,570.41 kW` (**cải thiện `+5.41%`**, `MAPE` giảm `+7.16%`) khi có thời tiết | Khẳng định mạng tuần tự `LSTM` khai thác hiệu quả tín hiệu ngoại sinh thời tiết trên cửa sổ 3D `(24, 4)`, trong khi `XGBoost` đã bão hòa thông tin qua 24 lags tự hồi quy và biến lịch |
| **Hiệu quả Đồ thị Không gian `adj_mx_bay.pkl`** | Ở tầm nhìn xa `60 phút`, `DCRNN` đạt `MAE = 2.3040 mph` so với `TCN` (`2.6243 mph`) và `LSTM` (`2.4986 mph`) | Chứng minh việc nhúng cấu trúc đồ thị `207` trạm giúp giảm mạnh sai số tích lũy khi dự báo xa |

## 5. Used data — Dữ liệu sử dụng

| Tập dữ liệu | Cấu trúc Đầu ra sau Feature Engineering | Mục đích Sử dụng | Module Thực thi |
|---|---|---|---|
| **Electricity Hold-out (`70/15/15`)** | Tabular 2D `(24521, 105)` & Tensor 3D `(24521, 24, 4)` | Huấn luyện chính, `EarlyStopping` trên Validation (`5,260` mẫu) và đánh giá trên Test (`5,260` mẫu) | `src/lab04/preprocess.py` |
| **Electricity 5-Fold CV (`Fold 1..5`)** | 5 cặp `(X_train_k, y_train_k)` & `(X_val_k, y_val_k)` | Đánh giá chéo 5-Fold (`Mean ± Std`) cho `VAR`, `XGBoost` và `Residual LSTM` không rò rỉ biên thời gian | `src/lab04/preprocess.py`, `src/lab04/evaluate.py` |
| **PEMS-BAY + Sensor Graph** | Cửa sổ `(6025 / 1285 / 1285, 12, 207)` & $(\mathbf{P}_f, \mathbf{P}_b)$ | Đối sánh công bằng bộ 3 `DCRNN`, `Multi-Horizon TCN` và `Multi-Horizon LSTM` ở 15, 30, 60 phút | `src/lab04/data_loader.py`, `src/lab04/preprocess.py` |
