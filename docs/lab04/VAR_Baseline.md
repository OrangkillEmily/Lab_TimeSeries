# Vector Autoregression (`VAR.md` — Mô hình Thống kê Cổ điển, Baseline 1)

## 1. Purpose — Mục đích

**Vector Autoregression (`VAR`)** là mô hình thống kê cổ điển đóng vai trò **Mô hình Đối chiếu Cơ sở Số 1 (`Statistical Baseline — Mô hình 1, Mục 6`)** trong dự án Lab 04, đại diện cho trường phái thống kê tuyến tính đa biến trong **Slide 5 Chương 4–5** (`chapter4-5.pdf`).

Mục đích đưa `VAR` vào thực nghiệm là để mô hình hóa mối quan hệ tương tác tuyến tính hai chiều (*Cross-dependence*) giữa 4 chuỗi thời gian đồng thời $\mathbf{X}_t = \left[\text{load}_t, \text{temp}_t, \text{humidity}_t, \text{wind}_t\right]^T$, đồng thời làm mốc cơ sở toán học để định lượng giá trị gia tăng của các mô hình phi tuyến (`XGBoost` và `LSTM`) khi dự báo phụ tải điện năng ([Statsmodels VAR Documentation](https://www.statsmodels.org/stable/vector_ar.html)).

## 2. Input-Output

### Input

| Đầu vào (Input) | Định dạng & Kích thước | Mô tả Chi tiết |
|---|---|---|
| `X_train_scaled` | `pd.DataFrame` `(T_train, 4)` | Chuỗi đa biến 4 cột `[load, temperature, humidity, wind_speed]` đã chuẩn hóa Z-score từ [Electricity Load Diagrams Dataset](https://www.kaggle.com/datasets/jacopoferretti/electricity-load-diagrams-2011-2014-feature-engin) |
| `X_val_scaled`, `X_test_scaled` | `pd.DataFrame` `(T_val, 4)`, `(T_test, 4)` | Chuỗi kiểm định và kiểm tra độc lập theo đúng thứ tự thời gian (`70% / 15% / 15%`) |
| `config.yaml -> var_model` | `Dict[str, Any]` | Cấu hình `adf_significance = 0.05`, `max_diff_order = 2`, `maxlags = 24`, `information_criterion = "aic"` |

### Output

| Đầu ra (Output) | Đường dẫn Lưu trữ | Mô tả Chi tiết |
|---|---|---|
| `Predicted Vector` $\hat{\mathbf{X}}_{t+1} \in \mathbb{R}^{4}$ | Trích xuất cột `load` ($\hat{y}_{t+1}$) | Dự báo đồng thời cả 4 biến ở bước tiếp theo, chỉ lấy cột `load` khi chấm điểm theo đúng Mục 6.3 đề bài |
| `models/lab04/var_model.pkl` | `models/lab04/var_model.pkl` | Đối tượng `VARResultsWrapper` lưu vector hằng số $\hat{\mathbf{c}} \in \mathbb{R}^{4}$ và $p^*$ ma trận hệ số $\hat{\mathbf{A}}_1, \dots, \hat{\mathbf{A}}_{p^*} \in \mathbb{R}^{4 \times 4}$ |
| `adf_stationarity_report.json` | `reports/lab04/metrics/adf_stationarity_report.json` | Bảng kết quả kiểm định ADF trước và sau khi lấy sai phân, cùng bảng chọn bậc trễ `AIC` / `BIC` cho $p \in \{1..24\}$ |

## 3. How to do — Quy trình thực hiện & Cơ sở toán học

### Sơ đồ luồng thực thi `VAR (Vector Autoregression)`

```text
      Chuỗi Đã Chuẩn hóa X_train_scaled = [load, temp, humidity, wind] (T_train x 4)
                                           |
                                           v
      [Bước 1] Kiểm định Tính Dừng ADF (Augmented Dickey-Fuller Test, alpha = 0.05)
               Nếu chuỗi không dừng (p-value >= 0.05) -> Sai phân: dX_t = X_t - X_{t-1}
                                           |
                                           v
      [Bước 2] Chọn Bậc Trễ Tối ưu p* trong khoảng [1 .. 24] theo tiêu chuẩn AIC / BIC:
               p* = argmin_{p} AIC(p)
                                           |
                                           v
      [Bước 3] Ước lượng Hệ Phương trình VAR(p*) bằng Bình phương Tối thiểu (OLS):
               X_t = c + A_1 X_{t-1} + A_2 X_{t-2} + ... + A_{p*} X_{t-p*} + eps_t
                                           |
                                           v
      [Bước 4] Dự báo Vector 4 biến [load, temp, humidity, wind] tại t+1
               (Cộng hoàn nguyên nếu đã sai phân -> Inverse StandardScaler -> Lấy cột load)
```

### Các bước thực hiện chi tiết

**Step 1 — Kiểm tra Tính Dừng bằng ADF Test và Sai phân (Mục 6.1)**

Mô hình VAR đòi hỏi mọi chuỗi thành phần phải có tính dừng (*Stationary*). Với từng chuỗi $x_{k,t}$ ($k \in \{1..4\}$), kiểm định giả thuyết $H_0$ (chuỗi không dừng) qua phương trình **ADF** ([Statsmodels adfuller](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html)):

$$\Delta x_{k,t} = \alpha_k + \beta_k t + \gamma_k x_{k,t-1} + \sum_{i=1}^{m} \delta_{k,i} \Delta x_{k,t-i} + u_{k,t}$$

Nếu bất kỳ chuỗi nào có $p\text{-value} \ge 0.05$, hệ thống thực hiện sai phân bậc $d = 1$:

$$\Delta \mathbf{X}_t = \mathbf{X}_t - \mathbf{X}_{t-1}$$

và kiểm định lại ADF cho đến khi cả 4 biến đều dừng ($p\text{-value} < 0.05$).

**Step 2 — Chọn Bậc Trễ Tối ưu $p^*$ theo AIC hoặc BIC (Mục 6.2)**

Xét các bậc trễ ứng viên $p \in \{1, 2, \dots, 24\}$ (bao phủ trọn vẹn chu kỳ ngày 24 giờ), tính tiêu chuẩn **AIC** và **BIC**:

$$\text{AIC}(p) = \ln\left|\hat{\boldsymbol{\Sigma}}_{\varepsilon}(p)\right| + \frac{2 p K^2}{T_{\text{train}}}, \qquad \text{BIC}(p) = \ln\left|\hat{\boldsymbol{\Sigma}}_{\varepsilon}(p)\right| + \frac{\ln(T_{\text{train}}) p K^2}{T_{\text{train}}}$$

với $K = 4$ biến và $\hat{\boldsymbol{\Sigma}}_{\varepsilon}(p) = \frac{1}{T_{\text{train}} - p}\sum_{t=p+1}^{T_{\text{train}}} \hat{\boldsymbol{\varepsilon}}_t \hat{\boldsymbol{\varepsilon}}_t^T$.

**Step 3 — Huấn luyện Hệ Phương trình $\text{VAR}(p^*)$ (Mục 6.3 & Slide 5 Chương 4–5)**

Biểu diễn tường minh hệ phương trình $\text{VAR}(p^*)$ cho 4 biến điện năng và thời tiết:

$$\begin{bmatrix} \text{load}_t \\ \text{temp}_t \\ \text{humidity}_t \\ \text{wind}_t \end{bmatrix} = \begin{bmatrix} c_1 \\ c_2 \\ c_3 \\ c_4 \end{bmatrix} + \sum_{i=1}^{p^*} \begin{bmatrix} a_{11}^{(i)} & a_{12}^{(i)} & a_{13}^{(i)} & a_{14}^{(i)} \\ a_{21}^{(i)} & a_{22}^{(i)} & a_{23}^{(i)} & a_{24}^{(i)} \\ a_{31}^{(i)} & a_{32}^{(i)} & a_{33}^{(i)} & a_{34}^{(i)} \\ a_{41}^{(i)} & a_{42}^{(i)} & a_{43}^{(i)} & a_{44}^{(i)} \end{bmatrix} \begin{bmatrix} \text{load}_{t-i} \\ \text{temp}_{t-i} \\ \text{humidity}_{t-i} \\ \text{wind}_{t-i} \end{bmatrix} + \begin{bmatrix} \varepsilon_{1,t} \\ \varepsilon_{2,t} \\ \varepsilon_{3,t} \\ \varepsilon_{4,t} \end{bmatrix}$$

Số lượng tham số ước lượng cho phương trình phụ tải điện là $1 + 4 p^*$ (và cho toàn hệ 4 phương trình là $4 + 16 p^*$).

**Step 4 — Dự báo Cuốn chiếu & Trích xuất Cột `load`**

Tại mỗi bước dự báo $t \to t+1$, mô hình sử dụng $p^*$ quan sát gần nhất để dự báo $\hat{\mathbf{X}}_{t+1}$, khôi phục sai phân (nếu $d > 0$), thực hiện `inverse_transform` về đơn vị gốc và lấy phần tử đầu tiên $\widehat{\text{load}}_{t+1}$ để đánh giá.

## 4. Evaluation metrics — Các chỉ số đánh giá thực nghiệm của VAR

Mô hình `VAR` được đánh giá đồng bộ trên cả **5 chỉ số hồi quy chuẩn** (`MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²`) ở cả 3 tầng kiểm định (**Hold-out Test `15%`**, **5-Fold `TimeSeriesSplit`** và **Walk-Forward Backtesting**):

| Chỉ số Đánh giá | Công thức Toán học trên Cột `load` | Đặc điểm Hiệu năng của Mô hình `VAR` (Theo Slide 4–5 Chương 4–5) |
|---|---|---|
| **MSE (↓)** | $\frac{1}{N}\sum_{t=1}^{N}\left(\text{load}_t - \widehat{\text{load}}_t^{\text{VAR}}\right)^2$ | Nhạy cảm với các đỉnh phụ tải phi tuyến do điều hòa nhiệt độ vào giờ cao điểm trưa/tối |
| **RMSE (↓)** | $\sqrt{\text{MSE}}$ | Phản ánh độ lệch chuẩn sai số dự báo của mô hình tuyến tính cơ sở trên đơn vị `kW/MW` |
| **MAE (↓)** | $\frac{1}{N}\sum_{t=1}^{N}\left|\text{load}_t - \widehat{\text{load}}_t^{\text{VAR}}\right|$ | Đo sai số tuyệt đối trung bình; ổn định khi dự báo ngắn hạn 1 bước ($t+1$) trong Walk-Forward |
| **MAPE (%) (↓)** | $\frac{100\%}{N}\sum_{t=1}^{N}\left|\frac{\text{load}_t - \widehat{\text{load}}_t^{\text{VAR}}}{\text{load}_t}\right|$ | Cho phép đối chiếu trực tiếp tỷ lệ phần trăm cải thiện của `XGBoost` và `LSTM` so với `VAR` |
| **$R^2$ Score (↑)** | $1 - \frac{\sum(\text{load}_t - \widehat{\text{load}}_t^{\text{VAR}})^2}{\sum(\text{load}_t - \overline{\text{load}})^2}$ | Đo phần phương sai tuyến tính giải thích được bởi các bậc trễ tự hồi quy và phụ thuộc chéo |

## 5. Used data — Dữ liệu sử dụng

| Tập dữ liệu | Cấu hình Chia & Xử lý | Vai trò đối với `VAR` | Module Thực thi |
|---|---|---|---|
| **Electricity Train (`70%`)** | Chuỗi thời gian 4 biến đã chuẩn hóa & sai phân dừng | Kiểm định ADF, chọn bậc trễ $p^*$ theo AIC/BIC và ước lượng ma trận hệ số $\hat{\mathbf{A}}_i$ | `src/lab04/preprocess.py`, `src/lab04/model.py`, `src/lab04/train.py` |
| **Electricity Val (`15%`) & Test (`15%`)** | Giữ nguyên thứ tự thời gian (`shuffle=False`) | Kiểm tra bậc trễ và thực hiện dự báo cuốn chiếu Walk-Forward trên cột `load` | `src/lab04/evaluate.py`, `src/lab04/visualize.py` |
| **5-Fold `TimeSeriesSplit`** | $K = 5$ Folds mở rộng theo thời gian | Ước lượng lại `VAR` độc lập trên từng Fold để xuất bảng `Mean ± Std` của 5 chỉ số | `src/lab04/evaluate.py` |
