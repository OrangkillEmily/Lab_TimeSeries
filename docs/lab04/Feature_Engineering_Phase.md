# Giai đoạn Kỹ thuật Đặc trưng Chuỗi Thời gian Đa biến (`Feature_Engineering_Phase.md`)

## 1. Purpose — Mục đích

**Giai đoạn Kỹ thuật Đặc trưng (`Feature_Engineering_Phase`)** thực hiện **Mục 5 (Feature Engineering cho ML và DL: 5.1 Cửa sổ trễ Lag Window $L=24$ & 5.2 Rolling Features cho XGBoost)** cùng **Mục 12 (Cửa sổ không gian – thời gian cho PEMS-BAY)** trong `practice 4.pdf` và lý thuyết **Slide 7–8 Chương 4–5** (`chapter4-5.pdf`).

Mục đích cốt lõi của giai đoạn này là chuyển đổi chuỗi thời gian đa biến liên tục $\mathbf{X}_t \in \mathbb{R}^{K}$ thành 3 cấu trúc biểu diễn chuyên biệt phù hợp với bản chất toán học của từng trường phái mô hình:
1. **Vector phẳng Lag + Rolling + Trend (Tabular 2D Matrix)** dành cho **XGBoost** ([Kaggle XGBoost Multivariate TS](https://www.kaggle.com/code/davidandressanchez/xgboost-for-multivariate-ts)).
2. **Tensor 3D Chuỗi tuần tự `(batch_samples, 24, 4)`** dành cho mạng **LSTM** ([Machine Learning Mastery Multivariate LSTM](https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/)).
3. **Tensor 4D Đồ thị Không gian – Thời gian `(batch_samples, 12, N_sensors, C)`** kết hợp ma trận chuyển tiếp bước ngẫu nhiên hai chiều từ `adj_mx_bay.pkl` dành cho **DCRNN** ([Li et al., ICLR 2018](https://arxiv.org/abs/1707.01926)).

## 2. Input-Output

### Input

| Đầu vào (Input) | Định dạng & Kích thước | Mô tả Chi tiết |
|---|---|---|
| `train_scaled`, `val_scaled`, `test_scaled` | `np.ndarray` `(T_split, 4)` | Chuỗi đa biến 4 cột `[load, temperature, humidity, wind_speed]` đã chuẩn hóa từ [Electricity Load Diagrams Dataset](https://www.kaggle.com/datasets/jacopoferretti/electricity-load-diagrams-2011-2014-feature-engin) |
| `pems_scaled` & `adj_mx_bay.pkl` | `np.ndarray` `(T_bay, N)` & `adj_mx` `(N, N)` | Chuỗi tốc độ xe đã chuẩn hóa từ [PEMS-BAY Dataset](https://drive.google.com/file/d/1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq/view?usp=drive_link) và ma trận kề từ [DCRNN Repository](https://github.com/liyaguang/dcrnn) |
| `config.yaml -> feature_engineering` | `Dict[str, Any]` | Cấu hình `lag_window = 24`, `rolling_mean_windows = [6]`, `rolling_std_windows = [24]`, `include_load_trend_24h = true` |

### Output

| Đầu ra (Output) | Định dạng & Kích thước | Mô tả Chi tiết |
|---|---|---|
| `X_xgb_train`, `X_xgb_val`, `X_xgb_test` | `np.ndarray` `(N_samples, D_tabular)` | Ma trận phẳng cho `XGBoost` gồm $24 \times 4 = 96$ biến trễ + Rolling mean 6h + Rolling std 24h + Load trend 24h + đặc trưng chu kỳ |
| `X_lstm_train`, `X_lstm_val`, `X_lstm_test` | `np.ndarray` `(N_samples, 24, 4)` | Tensor 3D chuẩn cho mạng `LSTM(64) -> LSTM(32) -> Dense(1)` theo Mục 8.1 đề bài |
| `y_train`, `y_val`, `y_test` | `np.ndarray` `(N_samples,)` | Vector mục tiêu $\text{load}_{t+1}$ tương ứng với từng cửa sổ thời gian |
| `X_pems`, `Y_pems` (`t+3, t+6, t+12`) | `Tensor` `(N_bay, 12, N_sensors, C)` | Cửa sổ 12 bước quá khứ (60 phút) và nhãn mục tiêu ở 15, 30, 60 phút phía trước cho `DCRNN` |

## 3. How to do — Quy trình thực hiện & Cơ sở toán học

### Sơ đồ luồng kiến tạo đặc trưng (`Feature Engineering Workflow`)

```text
         Chuỗi Đa biến Đã Chuẩn hóa X_t = [load_t, temp_t, humidity_t, wind_t]
                                          |
             +----------------------------+----------------------------+
             |                                                         |
             v                                                         v
 [Nhánh A: Cửa sổ Trượt Tensor 3D cho LSTM]         [Nhánh B: Tabular Lag + Rolling cho XGBoost]
 Trích xuất 24 bước liên tiếp [t-23 .. t]:          1. Làm phẳng 24 bước x 4 biến = 96 Lag Features
 Tensor X_lstm in R^(N_samples x 24 x 4)            2. Rolling Mean 6h  : mean(load_{t-5 : t})
 Nhãn mục tiêu: y = load_{t+1}                      3. Rolling Std 24h  : std(load_{t-23 : t})
                                                    4. Load Trend 24h   : load_t - load_{t-24}
                                                    5. Cross & Cyclical : load/temp, sin/cos(hour)
```

### Các bước thực hiện chi tiết

**Step 1 — Kiến tạo Cửa sổ Trễ Đa biến $L = 24$ giờ (Mục 5.1)**

Với mỗi thời điểm neo $t \ge 24$, mẫu đầu vào sử dụng trọn vẹn lịch sử 24 giờ gần nhất của cả 4 biến để dự báo $\text{load}_{t+1}$:
- **Dạng Tensor 3D cho Deep Learning (`LSTM`):**
  $$\mathcal{X}_t^{(\text{LSTM})} = \begin{bmatrix} \text{load}_{t-23} & \text{temp}_{t-23} & \text{humidity}_{t-23} & \text{wind}_{t-23} \\ \vdots & \vdots & \vdots & \vdots \\ \text{load}_{t} & \text{temp}_{t} & \text{humidity}_{t} & \text{wind}_{t} \end{bmatrix} \in \mathbb{R}^{24 \times 4}$$
- **Dạng Vector Phẳng cho Machine Learning (`XGBoost`):**
  $$\mathbf{x}_t^{(\text{lag})} = \left[\text{load}_{t-1}, \dots, \text{load}_{t-24},\ \text{temp}_{t-1}, \dots, \text{temp}_{t-24},\ \text{humidity}_{t-1}, \dots, \text{wind}_{t-24}\right] \in \mathbb{R}^{96}$$

**Step 2 — Kiến tạo Bộ Đặc trưng Trượt Chuyên biệt cho XGBoost (Mục 5.2 & Slide 7 Chương 4–5)**

Vì mô hình cây quyết định (`XGBoost`) không có trạng thái nhớ tuần tự như `LSTM`, hệ thống bổ sung 3 nhóm đặc trưng động lực học thời gian theo đúng yêu cầu Mục 5.2:
1. **Rolling Mean 6 giờ (`rolling_mean_6h`):** Nắm bắt mức nền phụ tải ngắn hạn trong buổi (sáng/trưa/chiều/tối):
   $$\text{RM}_{6,t} = \frac{1}{6}\sum_{i=0}^{5} \text{load}_{t-i}$$
2. **Rolling Standard Deviation 24 giờ (`rolling_std_24h`):** Đo lường biên độ biến động phụ tải trong một chu kỳ ngày trọn vẹn:
   $$\text{RSTD}_{24,t} = \sqrt{\frac{1}{24}\sum_{i=0}^{23}\left(\text{load}_{t-i} - \overline{\text{load}}_{t-23:t}\right)^2}$$
3. **Load Trend 24 giờ (`load_trend_24h`):** Đo độ lệch xu hướng phụ tải hiện tại so với đúng giờ đó ngày hôm trước:
   $$\text{Trend}_{24,t} = \text{load}_t - \text{load}_{t-24}$$
4. **Cross-variable & Cyclical Features (Slide 7 Chương 4–5):** Bổ sung tương tác chéo giữa điện năng và nhiệt độ cùng mã hóa lượng giác $\sin/\cos$ của giờ trong ngày ($24\text{h}$) và ngày trong tuần ($7\text{d}$).

**Step 3 — Kiến tạo Cửa sổ Đa bước & Ma trận Chuyển tiếp Đồ thị cho PEMS-BAY (Mục 12)**

Đối với bài toán nâng cao trên `data/raw/pems-bay.h5` và `data/raw/sensor_graph/adj_mx_bay.pkl`:
- **Cửa sổ thời gian:** Lấy $L = 12$ bước quá khứ ($12 \times 5\text{ phút} = 60\text{ phút}$) ánh xạ tới chuỗi 12 bước tương lai, trong đó trích xuất 3 mốc đánh giá chính là $t+3$ (**15 phút**), $t+6$ (**30 phút**) và $t+12$ (**60 phút**).
- **Toán tử chuyển tiếp đồ thị hai chiều:** Từ ma trận kề có hướng $\mathbf{W} \in \mathbb{R}^{N \times N}$, tính ma trận bậc ra $\mathbf{D}_O = \text{diag}(\mathbf{W}\mathbf{1})$ và bậc vào $\mathbf{D}_I = \text{diag}(\mathbf{W}^T\mathbf{1})$ để tạo cặp ma trận khuếch tán xuôi dòng $\mathbf{P}_f = \mathbf{D}_O^{-1}\mathbf{W}$ và ngược dòng $\mathbf{P}_b = \mathbf{D}_I^{-1}\mathbf{W}^T$.

## 4. Evaluation metrics — Đánh giá hiệu quả của Feature Engineering qua 5 Chỉ số

Chất lượng của các đặc trưng trễ và đặc trưng trượt được kiểm chứng định lượng thông qua mức cải thiện trên cả 5 chỉ số đánh giá (`MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²`) và độ quan trọng đặc trưng (`Feature Importance`):

| Tiêu chí Đánh giá Đặc trưng | Phương pháp Định lượng | Tác động tới `MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²` |
|---|---|---|
| **Gain Feature Importance của `XGBoost`** | Trích xuất tỷ trọng đóng góp giảm hàm mục tiêu `MSE` của từng biến trễ và biến trượt | Xác nhận vai trò của `load_lag_1`, `load_lag_24`, `rolling_mean_6h`, `load_trend_24h` và các biến trễ nhiệt độ `temp_lag_*` |
| **Độ giảm `RMSE` & `MAE` khi có Rolling Features** | Đối chiếu `XGBoost` (chỉ có Lag) vs `XGBoost` (Lag + Rolling 6h/24h + Trend) | Chứng minh thực nghiệm nhận định tại Mục 5.2: Rolling features giúp `XGBoost` học xu hướng và độ biến động tốt hơn |
| **Tương thích Không gian Đặc trưng $K \times L$** | Kiểm soát số chiều $4 \times 24 = 96$ (Slide 4 Chương 4–5) | Giữ cho hệ số xác định $R^2$ trên tập Test cao và không bị suy giảm do quá khớp (*Overfitting*) |

## 5. Used data — Dữ liệu sử dụng

| Tập dữ liệu | Cấu trúc Đầu ra sau Feature Engineering | Mục đích Sử dụng | Module Thực thi |
|---|---|---|---|
| **Electricity Hold-out (`70/15/15`)** | Tabular 2D `(N, D_xgb)` & Tensor 3D `(N, 24, 4)` | Huấn luyện chính, `EarlyStopping` trên Validation và đánh giá độc lập trên Test + Walk-Forward | `src/lab04/preprocess.py` |
| **Electricity 5-Fold CV (`Fold 1..5`)** | 5 cặp `(X_train_k, y_train_k)` & `(X_val_k, y_val_k)` | Đánh giá chéo 5-Fold (`Mean ± Std`) cho cả `XGBoost` và `LSTM` mà không rò rỉ biên thời gian giữa các Fold | `src/lab04/preprocess.py`, `src/lab04/evaluate.py` |
| **PEMS-BAY + Sensor Graph** | Cửa sổ `(N_bay, 12, N_sensors)` & $(\mathbf{P}_f, \mathbf{P}_b)$ | Huấn luyện và đánh giá mô hình `DCRNN` ở các chân trời dự báo 15 phút, 30 phút và 60 phút | `src/lab04/data_loader.py`, `src/lab04/preprocess.py` |
