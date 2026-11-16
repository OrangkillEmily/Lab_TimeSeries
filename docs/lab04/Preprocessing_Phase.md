# Giai đoạn Tiền xử lý, Chia Dữ liệu Thời gian & K-Fold CV (`Preprocessing_Phase.md`)

## 1. Purpose — Mục đích

**Giai đoạn Tiền xử lý (`Preprocessing_Phase`)** thực hiện **Mục 4 (Preprocessing: 4.1 Chuẩn hóa dữ liệu & 4.2 Chia tập dữ liệu theo thời gian)** kết hợp **Mục 6.1 (Kiểm tra tính dừng ADF & Differencing cho VAR)** trong `practice 4.pdf` và **Yêu cầu bổ sung số 4 (Kết hợp chia `Train-Val-Test` với `Temporal K-Fold Cross-Validation`)**.

Mục đích cốt lõi của giai đoạn này là:
1. Đảm bảo tính toàn vẹn nhân quả thời gian (chỉ dùng thông tin quá khứ, không rò rỉ dữ liệu tương lai `temporal_leakage_check = True`).
2. Chia tập dữ liệu `35,065 giờ` theo **đúng thứ tự thời gian (`shuffle=False`)** theo tỷ lệ chuẩn **Train 70% (`24,545 giờ`) – Validation 15% (`5,260 giờ`) – Test 15% (`5,260 giờ`)**, đồng thời thiết lập **5-Fold Expanding Window Cross-Validation (`TimeSeriesSplit`, $K=5$)** ([Scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)).
3. Chuẩn hóa Z-Score toàn bộ các biến bằng **`StandardScaler`** ([Scikit-learn StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)) chỉ được khớp (`fit`) độc lập trên `24,545 giờ` tập `Train` (và riêng biệt trong từng Fold của K-Fold CV) để giúp mạng **Residual LSTM**, **DCRNN**, **Multi-Horizon TCN** và **Multi-Horizon LSTM** hội tụ ổn định.
4. Thực hiện kiểm định nghiệm đơn vị **ADF Test** ([Statsmodels adfuller](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html) với `result_object=False` tương thích `statsmodels` mới) trước và sau khi lấy sai phân bậc 1 ($\Delta Y_t$) cho mô hình **VAR**.

## 2. Input-Output

### Input

| Đầu vào (Input) | Định dạng & Kích thước | Mô tả Chi tiết |
|---|---|---|
| `df_elec` | `pd.DataFrame` `(35065, 4)` | Chuỗi thời gian 4 biến `[load, temperature, humidity, wind_speed]` từ `data/raw/electricity_load.csv` ([UCI / Kaggle](https://www.kaggle.com/datasets/jacopoferretti/electricity-load-diagrams-2011-2014-feature-engin)) |
| `df_pems` & `adj_mx` | `pd.DataFrame` `(8640, 207)` & `np.ndarray` `(207, 207)` | Ma trận tốc độ xe tại 207 trạm cảm biến từ `data/raw/pems-bay.h5` ([PEMS-BAY Google Drive](https://drive.google.com/file/d/1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq/view?usp=drive_link)) và ma trận kề `data/raw/sensor_graph/adj_mx_bay.pkl` |
| `config.yaml -> splitting` | `Dict[str, Any]` | Tỷ lệ `train_ratio = 0.70`, `val_ratio = 0.15`, `test_ratio = 0.15`, `shuffle = false`, `n_splits = 5` |

### Output

| Đầu ra (Output) | Đường dẫn / Cấu trúc Lưu trữ | Mô tả Chi tiết |
|---|---|---|
| `X_xgb_train`, `X_xgb_val`, `X_xgb_test` | `(24521, 105)`, `(5260, 105)`, `(5260, 105)` | Ma trận đặc trưng bảng 105 chiều đã chuẩn hóa theo đúng trật tự thời gian `70% / 15% / 15%` |
| `X_lstm_train`, `X_lstm_val`, `X_lstm_test` | `(24521, 24, 4)`, `(5260, 24, 4)`, `(5260, 24, 4)` | Tensor 3D chuỗi 24 giờ $\times$ 4 biến đã chuẩn hóa Z-Score không rò rỉ |
| `pems_X_train`, `pems_X_val`, `pems_X_test` | `(6025, 12, 207)`, `(1285, 12, 207)`, `(1285, 12, 207)` | Cửa sổ không gian – thời gian 12 bước quá khứ trên 207 trạm cảm biến `PEMS-BAY` |
| `scaler_statistics` & `cv_folds_meta` | `prep_summary` (`reports/lab04/metrics/`) | Lưu tham số $\boldsymbol{\mu}_{\text{train}}$, $\boldsymbol{\sigma}_{\text{train}}$ của `StandardScaler` và thông tin 5 Fold (`train_size = 5845..29221`, `val_size = 5844`) |
| `stationarity_report` (`adf_raw`, `adf_after_diff1`) | `prep_summary` (`reports/lab04/metrics/`) | Kết quả kiểm định dừng ADF (`adf_statistic`, `p_value`, `used_lag = 24`, `is_stationary`) trước và sau sai phân bậc 1 |
| `chronological_split_diagram.png` & `stationarity_adf_comparison.png` | `reports/lab04/figures/` | Biểu đồ trực quan hóa phân vùng thời gian `70% - 15% - 15%` và đối sánh thống kê kiểm định dừng ADF |

## 3. How to do — Quy trình thực hiện & Cơ sở toán học

### Sơ đồ luồng thực thi `Preprocessing_Phase`

```text
                Raw Multivariate Series X_t (35,065 x 4)
                                |
          +---------------------+---------------------+
          |                                           |
          v                                           v
 [Nhánh 1: Chronological Split 70/15/15]     [Nhánh 2: Temporal 5-Fold CV]
 Chia theo thứ tự thời gian (shuffle=False): TimeSeriesSplit(n_splits = 5):
 - Train : 24,545h (2011-01-01 -> 2013-10-19) Với mỗi Fold k = 1..5 (val_size = 5,844h):
 - Val   :  5,260h (2013-10-19 -> 2014-05-26) - Train^(k) mở rộng từ 5,845 -> 29,221 giờ
 - Test  :  5,260h (2014-05-26 -> 2015-01-01) - Fit scaler_k CHỈ trên Train^(k)
          |                                   - Transform Train^(k) & Val^(k)
          v                                           |
 Fit StandardScaler CHỈ trên 24,545h Train            v
 Transform Train, Val, Test                  Tạo dữ liệu kiểm định chéo 5 Folds độc lập
          |
          v
 Kiểm định ADF Test (used_lag=24) -> Lấy sai phân bậc 1 (Delta Y_t) cho nhánh VAR
```

### Các bước thực hiện chi tiết

**Step 1 — Chia Tập Dữ liệu theo Thứ tự Thời gian (`Chronological Hold-out 70/15/15`)**

Với chuỗi thời gian điện năng có tổng độ dài $T = 35,065$ giờ, các điểm cắt chỉ số thời gian được xác định nghiêm ngặt mà không xáo trộn (`shuffle = False`):
- **Tập Train (`70%` — `24,545 giờ`)**: Từ `2011-01-01 00:00:00` đến `2013-10-19 16:00:00` (sau khi trượt cửa sổ `lag = 24` còn `24,521` mẫu huấn luyện có nhãn).
- **Tập Validation (`15%` — `5,260 giờ`)**: Từ `2013-10-19 17:00:00` đến `2014-05-26 20:00:00` (dùng ngữ cảnh 24 giờ cuối của Train để tạo đủ `5,260` mẫu kiểm định).
- **Tập Test (`15%` — `5,260 giờ`)**: Từ `2014-05-26 21:00:00` đến `2015-01-01 00:00:00` (dùng ngữ cảnh 24 giờ cuối của Validation để tạo đủ `5,260` mẫu kiểm tra).

**Step 2 — Thiết lập Kiểm định Chéo Chuỗi Thời gian (`5-Fold Expanding Window TimeSeriesSplit`)**

Sử dụng cơ chế cửa sổ mở rộng với $K = 5$ Folds, trong đó mỗi tập kiểm định $\mathcal{D}_{\text{val}}^{(k)}$ cố định `5,844` giờ và tập huấn luyện $\mathcal{D}_{\text{train}}^{(k)}$ mở rộng tuyến tính từ `5,845` giờ (Fold 1) lên `11,689` (Fold 2), `17,533` (Fold 3), `23,377` (Fold 4) và `29,221` giờ (Fold 5). Mỗi Fold $k$ khởi tạo một bộ `StandardScaler` độc lập chỉ khớp trên $\mathcal{D}_{\text{train}}^{(k)}$.

**Step 3 — Chuẩn hóa Z-Score Độc lập bằng `StandardScaler` (Mục 4.1)**

Mỗi biến thứ $j \in \{1, \dots, 4\}$ được chuẩn hóa theo thống kê tính riêng trên `24,545` giờ tập `Train`:
- `load`: $\mu_{\text{train}} = 244,810.5838\text{ kW}$, $\sigma_{\text{train}} = 84,767.5263\text{ kW}$
- `temperature`: $\mu_{\text{train}} = 19.1475\ ^\circ\text{C}$, $\sigma_{\text{train}} = 8.2999\ ^\circ\text{C}$
- `humidity`: $\mu_{\text{train}} = 66.0641\%$, $\sigma_{\text{train}} = 14.7141\%$
- `wind_speed`: $\mu_{\text{train}} = 4.9278\text{ m/s}$, $\sigma_{\text{train}} = 1.1829\text{ m/s}$

**Step 4 — Kiểm định Tính Dừng ADF & Sai phân Bậc 1 cho Mô hình VAR (Mục 6.1)**

Kiểm định giả thuyết $H_0$: *"Chuỗi có nghiệm đơn vị (không dừng)"* bằng kiểm định **Augmented Dickey-Fuller (`used_lag = 24`)**:
- **Trên chuỗi gốc (Level Series):** Ba biến chính chưa đạt tính dừng ở mức ý nghĩa $\alpha = 0.05$ gồm `load` ($\text{ADF} = -2.8161, p = 0.0560$), `temperature` ($\text{ADF} = -2.0663, p = 0.2583$) và `humidity` ($\text{ADF} = -2.5623, p = 0.1010$); chỉ có `wind_speed` dừng ($\text{ADF} = -6.5418, p = 0.0000$).
- **Sau khi lấy sai phân bậc 1 ($\Delta \mathbf{X}_t = \mathbf{X}_t - \mathbf{X}_{t-1}$):** Cả 4 biến đều đạt tính dừng tuyệt đối với $p\text{-value} = 0.0000$ (`load`: $-29.9928$, `temperature`: $-34.5548$, `humidity`: $-45.1783$, `wind_speed`: $-41.6275$), đảm bảo điều kiện toán học nghiêm ngặt cho mô hình **VAR(24)**.

## 4. Evaluation metrics — Các chỉ số kiểm định chất lượng tiền xử lý & tác động tới 5 Metrics

| Tiêu chí Kiểm định Tiền xử lý | Điều kiện & Kết quả Thực nghiệm | Tác động Trực tiếp tới `MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²` |
|---|---|---|
| **Kiểm tra Chuẩn hóa Tập Train** | `leakage_free_verified = True`; $\|\boldsymbol{\mu}_{\text{scaled}}\|_2 < 10^{-6}$, $\|\boldsymbol{\sigma}_{\text{scaled}} - \mathbf{1}\|_2 < 10^{-6}$ | Ngăn chặn bão hòa đạo hàm trong `Residual LSTM`, `DCRNN` và `TCN`, giúp giảm mạnh `MSE` và `RMSE` |
| **Kiểm tra Chống Rò rỉ (`Anti-Leakage Audit`)** | `temporal_leakage_check = True` trên cả Hold-out `70/15/15` và `5-Fold CV` | Đảm bảo các chỉ số `MAE`, `MAPE (%)` và `R²` phản ánh trung thực năng lực dự báo tương lai |
| **Kiểm định Tính Dừng ADF sau Sai phân** | $p\text{-value} = 0.0000 < 0.05$ trên cả 4 biến $\Delta \mathbf{X}_t$ | Loại bỏ hiện tượng hồi quy giả mạo (*Spurious Regression*), giúp `VAR` đạt $R^2 = 0.9935$ trên tập Test |
| **Khôi phục Thang đo Gốc (`Inverse Scaling`)** | $\hat{y}_t = \hat{y}_t^{\text{scaled}} \cdot 84,767.53 + 244,810.58$ (`kW`) | Đảm bảo toàn bộ 5 chỉ số đánh giá được tính toán chuẩn xác trên đơn vị vật lý gốc (`kW` và `mph`) |

## 5. Used data — Dữ liệu sử dụng

| Tập dữ liệu | Tỷ lệ / Kích thước Thực tế | Vai trò trong `Preprocessing_Phase` | Module Thực thi |
|---|---|---|---|
| **Electricity Train Set** | `70%` đầu chuỗi (`24,545` giờ $\to$ `24,521` cửa sổ) | Khớp (`fit`) `StandardScaler` chính, kiểm định dừng ADF và làm gốc cho `5-Fold TimeSeriesSplit` | `src/lab04/preprocess.py` |
| **Electricity Validation Set** | `15%` kế tiếp (`5,260` giờ $\to$ `5,260` cửa sổ) | Được biến đổi bằng `scaler.transform()`, dùng để giám sát `EarlyStopping` và lịch trình học | `src/lab04/preprocess.py` |
| **Electricity Test Set** | `15%` cuối chuỗi (`5,260` giờ $\to$ `5,260` cửa sổ) | Được biến đổi bằng `scaler.transform()`, khóa kín cho đánh giá Hold-out và Walk-Forward 168h | `src/lab04/preprocess.py` |
| **PEMS-BAY Splits (`70/15/15`)** | `Train = 6,025`, `Val = 1,285`, `Test = 1,285` cửa sổ | Chuẩn hóa tốc độ xe (`mph`) trên 207 trạm cảm biến cho bộ 3 mô hình `DCRNN`, `TCN`, `LSTM` | `src/lab04/preprocess.py`, `src/lab04/data_loader.py` |
