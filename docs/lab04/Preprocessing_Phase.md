# Giai đoạn Tiền xử lý, Chia Dữ liệu Thời gian & K-Fold CV (`Preprocessing_Phase.md`)

## 1. Purpose — Mục đích

**Giai đoạn Tiền xử lý (`Preprocessing_Phase`)** thực hiện **Mục 4 (Preprocessing: 4.1 Chuẩn hóa dữ liệu & 4.2 Chia tập dữ liệu theo thời gian)** kết hợp **Mục 6.1 (Kiểm tra tính dừng ADF & Differencing cho VAR)** trong `practice 4.pdf` và **Yêu cầu bổ sung số 4 (Kết hợp chia `Train-Val-Test` với `Temporal K-Fold Cross-Validation`)**.

Mục đích cốt lõi của giai đoạn này là:
1. Làm sạch các điểm khuyết theo nguyên tắc nhân quả thời gian (chỉ dùng thông tin quá khứ).
2. Chia tập dữ liệu theo **đúng thứ tự thời gian (`shuffle=False`)** theo tỷ lệ chuẩn **Train 70% – Validation 15% – Test 15%**, đồng thời thiết lập **5-Fold Expanding Window Cross-Validation (`TimeSeriesSplit`, $K=5$)** ([Scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)).
3. Chuẩn hóa toàn bộ các biến bằng **`StandardScaler`** ([Scikit-learn StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)) được khớp (`fit`) độc lập trên tập `Train` (và riêng biệt trong từng Fold của K-Fold CV) để giúp mạng **LSTM** và **DCRNN** hội tụ ổn định, loại bỏ độ lệch do khác đơn vị đo (`kW/MW`, `°C`, `%`, `m/s`, `mph`) và cải thiện hiệu quả của **VAR** cùng **XGBoost**.
4. Kiểm định tính dừng bằng **ADF Test** ([Statsmodels adfuller](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html)) và thực hiện sai phân (**Differencing**) cho mô hình **VAR** nếu chuỗi chưa dừng.

## 2. Input-Output

### Input

| Đầu vào (Input) | Định dạng & Kích thước | Mô tả Chi tiết |
|---|---|---|
| `df_electricity` | `pd.DataFrame` `(T, 4)` | Chuỗi thời gian 4 biến `[load, temperature, humidity, wind_speed]` đã qua kiểm tra từ `data/raw/electricity_load.csv` ([UCI / Kaggle](https://www.kaggle.com/datasets/jacopoferretti/electricity-load-diagrams-2011-2014-feature-engin)) |
| `df_pems_bay` | `pd.DataFrame` `(T_bay, N_sensors)` | Ma trận tốc độ xe tại các trạm cảm biến từ `data/raw/pems-bay.h5` ([PEMS-BAY Google Drive](https://drive.google.com/file/d/1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq/view?usp=drive_link)) |
| `config.yaml -> splitting_and_validation` | `Dict[str, Any]` | Tỷ lệ `train_ratio = 0.70`, `val_ratio = 0.15`, `test_ratio = 0.15`, `shuffle = false`, `n_splits = 5` |

### Output

| Đầu ra (Output) | Đường dẫn Lưu trữ | Mô tả Chi tiết |
|---|---|---|
| `train_scaled.npy`, `val_scaled.npy`, `test_scaled.npy` | `data/processed/electricity/` | Các ma trận chuỗi đa biến đã chuẩn hóa Z-score theo đúng trật tự thời gian `70% / 15% / 15%` |
| `kfold_splits_metadata.json` | `data/processed/electricity/kfold_splits_metadata.json` | Chỉ số mốc thời gian (`train_start`, `train_end`, `val_start`, `val_end`) và thống kê `StandardScaler` riêng của từng Fold $k \in \{1..5\}$ |
| `standard_scaler.pkl` | `models/standard_scaler.pkl` | Đối tượng `StandardScaler` đã khớp trên tập `Train (70%)`, lưu $\boldsymbol{\mu}_{\text{train}}$ và $\boldsymbol{\sigma}_{\text{train}}$ để `inverse_transform` khi đánh giá |
| `adf_stationarity_report.json` | `reports/metrics/adf_stationarity_report.json` | Kết quả kiểm định ADF (`ADF Statistic`, `p-value`, `Critical Values 1%, 5%, 10%`, bậc sai phân $d$ áp dụng cho VAR) |
| `train_val_test_split.png` & `kfold_splits_timeline.png` | `reports/figures/` | Biểu đồ trực quan hóa phân vùng thời gian `70% - 15% - 15%` và 5 lát cắt `TimeSeriesSplit` |

## 3. How to do — Quy trình thực hiện & Cơ sở toán học

### Sơ đồ luồng thực thi `Preprocessing_Phase`

```text
                Raw Multivariate Series X_t (T x K)
                                |
          +---------------------+---------------------+
          |                                           |
          v                                           v
 [Nhánh 1: Chronological Split]             [Nhánh 2: Temporal 5-Fold CV]
 Chia theo thứ tự thời gian (No Shuffle):   TimeSeriesSplit(n_splits = 5):
 - Train : 0%   -> 70%                      Với mỗi Fold k = 1..5:
 - Val   : 70%  -> 85%                      - Train^(k) luôn đứng trước Val^(k)
 - Test  : 85%  -> 100%                     - Fit scaler_k CHỈ trên Train^(k)
          |                                 - Transform Train^(k) & Val^(k)
          v                                           |
 Fit StandardScaler CHỈ trên Train (70%)              v
 Transform Train, Val, Test                 Tạo dữ liệu kiểm định chéo 5 Folds
          |
          v
 Kiểm định ADF Test trên Train -> Sai phân (Differencing) cho nhánh VAR nếu p >= 0.05
```

### Các bước thực hiện chi tiết

**Step 1 — Chia Tập Dữ liệu theo Thứ tự Thời gian (`Chronological Hold-out 70/15/15`)**

Với chuỗi thời gian có tổng độ dài $T$, các điểm cắt chỉ số thời gian được xác định nghiêm ngặt mà không xáo trộn:

$$T_{\text{train}} = \lfloor 0.70 \times T \rfloor, \qquad T_{\text{val}} = \lfloor 0.85 \times T \rfloor$$

$$\mathcal{D}_{\text{train}} = \mathbf{X}_{1 : T_{\text{train}}}, \qquad \mathcal{D}_{\text{val}} = \mathbf{X}_{T_{\text{train}}+1 : T_{\text{val}}}, \qquad \mathcal{D}_{\text{test}} = \mathbf{X}_{T_{\text{val}}+1 : T}$$

**Step 2 — Thiết lập Kiểm định Chéo Chuỗi Thời gian (`Temporal K-Fold Cross-Validation`)**

Sử dụng cơ chế cửa sổ mở rộng (**Expanding Window `TimeSeriesSplit`** với $K = 5$ Folds). Với mỗi Fold $k \in \{1, 2, 3, 4, 5\}$:
- Tập huấn luyện $\mathcal{D}_{\text{train}}^{(k)}$ bao gồm các quan sát từ đầu chuỗi đến mốc cắt thứ $k$.
- Tập kiểm định $\mathcal{D}_{\text{val}}^{(k)}$ là khối thời gian liền kề ngay sau $\mathcal{D}_{\text{train}}^{(k)}$.
- **Nguyên tắc chống rò rỉ trong K-Fold:** Mỗi Fold $k$ khởi tạo một bộ chuẩn hóa $\text{Scaler}^{(k)}$ độc lập, tính $\boldsymbol{\mu}_{\text{train}}^{(k)}$ và $\boldsymbol{\sigma}_{\text{train}}^{(k)}$ chỉ trên $\mathcal{D}_{\text{train}}^{(k)}$.

**Step 3 — Chuẩn hóa Z-Score bằng `StandardScaler` (Mục 4.1)**

Mỗi biến thứ $j \in \{1, \dots, K\}$ được chuẩn hóa về phân phối có trung bình bằng $0$ và độ lệch chuẩn bằng $1$ dựa trên thống kê của tập huấn luyện:

$$\mu_{j,\text{train}} = \frac{1}{T_{\text{train}}}\sum_{t=1}^{T_{\text{train}}} x_{j,t}, \qquad \sigma_{j,\text{train}} = \sqrt{\frac{1}{T_{\text{train}}}\sum_{t=1}^{T_{\text{train}}}\left(x_{j,t} - \mu_{j,\text{train}}\right)^2}$$

$$x_{j,t}^{\text{scaled}} = \frac{x_{j,t} - \mu_{j,\text{train}}}{\sigma_{j,\text{train}}}$$

**Step 4 — Kiểm định Tính Dừng ADF & Sai phân cho Mô hình VAR (Mục 6.1)**

Kiểm định giả thuyết $H_0$: *"Chuỗi có nghiệm đơn vị (không dừng)"* bằng kiểm định **Augmented Dickey-Fuller (ADF)**:

$$\Delta x_{j,t} = \alpha + \beta t + \gamma x_{j,t-1} + \sum_{i=1}^{p} \delta_i \Delta x_{j,t-i} + \varepsilon_t$$

Nếu $p\text{-value} \ge 0.05$, chuỗi được lấy sai phân bậc 1 ($\Delta \mathbf{X}_t = \mathbf{X}_t - \mathbf{X}_{t-1}$) trước khi đưa vào huấn luyện mô hình **VAR**, và tự động khôi phục mức gốc ($\hat{\mathbf{X}}_{t+1} = \mathbf{X}_t + \widehat{\Delta \mathbf{X}}_{t+1}$) khi dự báo.

## 4. Evaluation metrics — Các chỉ số kiểm định chất lượng tiền xử lý & tác động tới 5 Metrics

Giai đoạn Tiền xử lý đảm bảo tính toàn vẹn toán học cho cả 5 chỉ số đánh giá mô hình (`MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²`):

| Tiêu chí Kiểm định Tiền xử lý | Điều kiện Đạt chuẩn | Tác động Trực tiếp tới `MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²` |
|---|---|---|
| **Kiểm tra Chuẩn hóa Tập Train** | $\|\boldsymbol{\mu}(\mathbf{X}_{\text{train}}^{\text{scaled}})\|_2 < 10^{-6}$ và $\|\boldsymbol{\sigma}(\mathbf{X}_{\text{train}}^{\text{scaled}}) - \mathbf{1}\|_2 < 10^{-6}$ | Ngăn chặn bão hòa gradient trong `LSTM` / `DCRNN`, giúp giảm mạnh `MSE` và `RMSE` |
| **Kiểm tra Chống Rò rỉ (`Anti-Leakage Audit`)** | $t_{\max}(\text{Train}) < t_{\min}(\text{Val}) < t_{\min}(\text{Test})$ trong cả Hold-out và 5 Folds | Đảm bảo các chỉ số `MAE`, `MAPE (%)` và `R²` phản ánh trung thực năng lực dự báo tương lai |
| **Kiểm định Tính Dừng ADF sau Sai phân** | $p\text{-value} < 0.05$ trên cả 4 biến đầu vào của `VAR` | Loại bỏ hiện tượng hồi quy giả mạo (*Spurious Regression*), giữ cho `R²` và `RMSE` của `VAR` hợp lệ |
| **Khôi phục Thang đo Gốc (`Inverse Scaling`)** | $\hat{y}_t = \hat{y}_t^{\text{scaled}} \cdot \sigma_{\text{load, train}} + \mu_{\text{load, train}}$ | Giúp `MSE`, `RMSE`, `MAE` mang đúng đơn vị điện năng/tốc độ gốc và tránh chia cho $0$ khi tính `MAPE (%)` |

## 5. Used data — Dữ liệu sử dụng

| Tập dữ liệu | Tỷ lệ / Cấu hình | Vai trò trong `Preprocessing_Phase` | Module Thực thi |
|---|---|---|---|
| **Electricity Train Set** | `70%` đầu chuỗi (`0% -> 70%`) | Khớp (`fit`) `StandardScaler` chính, kiểm định tính dừng ADF và làm gốc mở rộng cho `5-Fold TimeSeriesSplit` | `src/lab04/preprocess.py` |
| **Electricity Validation Set** | `15%` kế tiếp (`70% -> 85%`) | Được biến đổi bằng `scaler.transform()`, dùng để giám sát `EarlyStopping` và chọn siêu tham số | `src/lab04/preprocess.py` |
| **Electricity Test Set** | `15%` cuối chuỗi (`85% -> 100%`) | Được biến đổi bằng `scaler.transform()`, khóa kín cho đánh giá cuối cùng và Walk-Forward Backtesting | `src/lab04/preprocess.py` |
| **PEMS-BAY Splits (`70/15/15`)** | `data/raw/pems-bay.h5` | Chuẩn hóa tốc độ xe (mph) theo thống kê tập Train (bỏ qua các điểm cảm biến bị khuyết `0` khi tính $\mu, \sigma$) | `src/lab04/preprocess.py`, `src/lab04/data_loader.py` |
