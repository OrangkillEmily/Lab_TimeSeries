# Giai đoạn Khám phá và Thấu hiểu Dữ liệu (`Eda_Phase.md` — Data Understanding)

## 1. Purpose — Mục đích

**Giai đoạn Khám phá Dữ liệu (`Eda_Phase`)** thực hiện **Mục 3 (Data Understanding)** trong đề bài `practice 4.pdf` kết hợp nền tảng lý thuyết **Chương 4–5: Mô hình dự đoán chuỗi thời gian đa biến** (`chapter4-5.pdf`).

Mục đích cốt lõi của giai đoạn này là:
1. Kiểm toán chất lượng dữ liệu thô (`35,065 giờ × 4 biến` điện năng & thời tiết và `8,640 bước × 207 trạm cảm biến` giao thông), lập biểu đồ kiểm toán giá trị khuyết (`missing values`) và xác nhận chuỗi liên tục không đứt gãy.
2. Tính bảng thống kê mô tả **13 chỉ tiêu** (`Count`, `Missing`, `Mean`, `Std`, `Min`, `Q25`, `Median`, `Q75`, `Max`, `Skewness`, `Kurtosis`, `IQR_Outliers`, `ZScore_Outliers`) kèm thống kê `Min / Mean / Max` gắn với các tháng đặc trưng trên biểu đồ không bị cắt chữ.
3. Khám phá quy luật mùa vụ kép (**chu kỳ ngày 24 giờ** và **chu kỳ tuần 168 giờ**), phân tích tự tương quan **ACF / PACF** (đã loại bỏ `Lag 0 = 1.0` để làm nổi bật độ trễ $1\text{h} \dots 48\text{h}$), và định lượng **sự phụ thuộc chéo (Cross-dependence)** qua cả tương quan tuyến tính **Pearson ($r$)** lẫn phi tham số **Spearman ($\rho$)** trên bộ dữ liệu [Electricity Load Diagrams Dataset (UCI / Kaggle)](https://www.kaggle.com/datasets/jacopoferretti/electricity-load-diagrams-2011-2014-feature-engin) và mạng lưới cảm biến giao thông [PEMS-BAY Dataset (Google Drive)](https://drive.google.com/file/d/1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq/view?usp=drive_link).

## 2. Input-Output

### Input

| Đầu vào (Input) | Định dạng & Kích thước | Mô tả Chi tiết |
|---|---|---|
| `data/raw/electricity_load.csv` | `pd.DataFrame` `(35065, 4)` | Chuỗi thời gian đa biến theo giờ (`2011-01-01 00:00` đến `2015-01-01 00:00`) gồm 4 cột: `load` (`kW`), `temperature` (`°C`), `humidity` (`%`), `wind_speed` (`m/s`) từ [Kaggle UCI Electricity Load Diagrams](https://www.kaggle.com/datasets/jacopoferretti/electricity-load-diagrams-2011-2014-feature-engin) |
| `data/raw/pems-bay.h5` | `pd.DataFrame` `(8640, 207)` | Dữ liệu tốc độ dòng phương tiện (`mph`, `2017-01-01` đến `2017-01-30`, bước `5 phút`) từ [PEMS-BAY Google Drive](https://drive.google.com/file/d/1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq/view?usp=drive_link) |
| `data/raw/sensor_graph/adj_mx_bay.pkl` | `tuple` / `np.ndarray` `(207, 207)` | Ma trận kề khoảng cách mạng lưới đường bộ (`1,238` cạnh có hướng khác không, độ thưa `97.11%`) từ [DCRNN GitHub Repository](https://github.com/liyaguang/dcrnn) |
| `config.yaml -> preprocessing` | `Dict[str, Any]` | Cấu hình ngưỡng phát hiện ngoại lai (`iqr_multiplier = 1.5`, `zscore_threshold = 3.0`) và chu kỳ mùa vụ (`daily = 24`, `weekly = 168`) |

### Output

| Đầu ra (Output) | Đường dẫn Lưu trữ | Mô tả Chi tiết |
|---|---|---|
| `eda_descriptive_stats.csv` | `reports/lab04/metrics/eda_descriptive_stats.csv` | Bảng thống kê mô tả 13 chỉ tiêu cho 4 biến (`Mean`, `Std`, `Min`, `Q25`, `Median`, `Q75`, `Max`, `Skewness`, `Kurtosis`, `IQR_Outliers`, `ZScore_Outliers`) |
| `eda_pearson_correlation.csv` & `eda_spearman_correlation.csv` | `reports/lab04/metrics/` | Hai ma trận tương quan tuyến tính Pearson ($r$) và tương quan hạng Spearman ($\rho$) giữa 4 biến `[load, temperature, humidity, wind_speed]` |
| `multivariate_overview.png` | `reports/lab04/figures/multivariate_overview.png` | Biểu đồ toàn cảnh 4 chuỗi thời gian đa biến với chú thích `Min / Mean / Max` và trục thời gian rõ ràng không chồng lấp |
| `feature_distributions_and_outliers.png` | `reports/lab04/figures/feature_distributions_and_outliers.png` | Biểu đồ phân phối Histogram/KDE kết hợp Boxplot kiểm định ngoại lai và kiểm toán giá trị khuyết (`Missing = 0`) |
| `correlation_heatmap.png` & `weather_vs_load_joint.png` | `reports/lab04/figures/` | Biểu đồ nhiệt tương quan kép (`Pearson r` & `Spearman rho`) và biểu đồ phân tán kèm đường khớp bậc hai (*Quadratic Fit*) giữa thời tiết và phụ tải |
| `seasonal_decomposition.png` & `acf_pacf_load.png` | `reports/lab04/figures/` | Biểu đồ phân rã 4 tầng (`Observed – Trend – Daily Seasonal 24h – Residual`) và lược đồ `ACF / PACF` (đã bỏ `Lag 0`) |
| `pems_bay_eda_overview.png` | `reports/lab04/figures/pems_bay_eda_overview.png` | Biểu đồ khám phá ma trận kề `adj_mx_bay.pkl`, phân phối tốc độ xe (`Mean = 62.46 mph`) và chu kỳ tắc nghẽn giờ cao điểm trên `PEMS-BAY` |

## 3. How to do — Quy trình thực hiện & Cơ sở toán học

### Sơ đồ luồng thực thi `Eda_Phase`

```text
  data/raw/electricity_load.csv (35,065 x 4) + data/raw/pems-bay.h5 (8,640 x 207) & adj_mx_bay.pkl
                                              |
                                              v
  [Bước 1] Kiểm toán Missing Values (Missing = 0) & Tính liên tục trục thời gian (DatetimeIndex)
                                              |
                                              v
  [Bước 2] Thống kê 13 chỉ tiêu (Min/Mean/Max theo tháng) & Ngoại lai IQR (1.5xIQR) / Z-Score (|z|>3)
                                              |
                                              v
  [Bước 3] Phân tích Tương quan Pearson (r), Spearman (rho) & Hồi quy bậc hai (Quadratic Fit)
                                              |
                                              v
  [Bước 4] Phân rã Mùa vụ 4 tầng (24h/168h), ACF/PACF (bỏ Lag 0) & Cấu trúc đồ thị PEMS-BAY
```

### Các bước thực hiện chi tiết

**Step 1 — Kiểm tra Missing Values và Tính toàn vẹn trục thời gian**

Kiểm tra tỷ lệ khuyết dữ liệu trên từng biến $k \in \{\text{load}, \text{temperature}, \text{humidity}, \text{wind\_speed}\}$ và bước nhảy thời gian $\Delta t = 1\text{ giờ}$:

$$\text{Missing Ratio}_k = \frac{1}{T}\sum_{t=1}^{T} \mathbb{I}\left(x_{k,t} = \text{NaN}\right) \times 100\% = 0.00\% \quad (T = 35,065)$$

**Step 2 — Thống kê 13 Chỉ tiêu & Kiểm định Giá trị Ngoại lai (Outliers Detection)**

Áp dụng đồng thời hai tiêu chuẩn thống kê cho từng biến để nhận diện các giờ phụ tải đỉnh đột biến hoặc gió giật mạnh:
- **Phương pháp khoảng tứ phân vị (IQR Rule):**
  $$\text{IQR}_k = Q_{3,k} - Q_{1,k}, \qquad x_{k,t} \notin \left[Q_{1,k} - 1.5\cdot\text{IQR}_k,\ Q_{3,k} + 1.5\cdot\text{IQR}_k\right]$$
  Thực nghiệm ghi nhận `load`, `temperature`, `humidity` có `0` điểm ngoại lai IQR (`0.00%`), trong khi `wind_speed` có `124` điểm (`0.35%`).
- **Phương pháp điểm chuẩn hóa (Z-Score Rule):**
  $$z_{k,t} = \frac{|x_{k,t} - \mu_k|}{\sigma_k} > 3.0$$
  Ghi nhận `10` điểm trên `load` và `65` điểm trên `wind_speed`. Các biểu đồ được tự động nới rộng giới hạn trục (`ylim` / `xlim`) để hiển thị trọn vẹn nhãn `Min / Mean / Max` và chú giải mà không bị cắt chữ.

**Step 3 — Phân tích Tính Mùa vụ & Lược đồ Tự tương quan ACF/PACF (Loại bỏ Lag 0)**

Thực hiện phân rã chuỗi thời gian cộng tính ([Statsmodels Seasonal Decompose](https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.seasonal_decompose.html)) và tính hàm tự tương quan `ACF` / tự tương quan riêng phần `PACF` từ độ trễ $k = 1$ đến $k = 48$ (loại bỏ `Lag 0 = 1.0` để tránh ép dẹp thang đo của các độ trễ thực sự):

$$\text{load}_t = T_t + S_t^{(\text{daily, } 24)} + S_t^{(\text{weekly, } 168)} + R_t$$

Đỉnh cực đại rõ nét tại $\text{Lag } 1\text{h}, 23\text{h}, 24\text{h}, 48\text{h}$ xác lập cơ sở toán học trực tiếp để chọn cửa sổ nhìn lại `lookback = 24 giờ`.

**Step 4 — Định lượng Tương quan Tuyến tính Pearson ($r$) & Phi tham số Spearman ($\rho$)**

Hệ thống tính đồng thời cả hệ số tương quan tuyến tính **Pearson ($r$)** và tương quan hạng **Spearman ($\rho$)** để đánh giá quan hệ đơn điệu lẫn phi tuyến bậc hai:
- **`humidity` vs. `load`**: $r = -0.9224$, $\rho = -0.9208$ (biến thời tiết có độ lớn tương quan mạnh nhất $|r| = 0.9224$).
- **`temperature` vs. `load`**: $r = +0.8853$, $\rho = +0.8717$ (quan hệ đồng biến mạnh, thúc đẩy nhu cầu làm mát).
- **`wind_speed` vs. `load`**: $r = -0.3096$, $\rho = -0.2899$ (tác động làm mát tự nhiên mức trung bình yếu).

## 4. Evaluation metrics — Các chỉ số kiểm định trong giai đoạn EDA

Giai đoạn EDA cung cấp các chỉ số thống kê nền tảng định hướng trực tiếp cho thiết kế tiền xử lý và đánh giá mô hình theo 5 chỉ số hồi quy (`MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²`):

| Chỉ số Thống kê / Kiểm định | Giá trị Thực nghiệm từ Pipeline | Vai trò Định hướng cho 5 Chỉ số Dự báo (`MSE`, `RMSE`, `MAE`, `MAPE`, `R²`) |
|---|---|---|
| **Tương quan Pearson ($r$) & Spearman ($\rho$)** | `humidity`: $r = -0.9224$; `temperature`: $r = +0.8853$ | Xác định biến thời tiết chủ lực (Câu hỏi 1 Mục 11) và giải thích sự cộng tuyến cao giữa `temperature` và `humidity` ($r = -0.9657$) |
| **Trung bình & Độ lệch chuẩn Phụ tải ($\mu_{\text{load}}, \sigma_{\text{load}}$)** | $\mu = 238,118.60\text{ kW}$, $\sigma = 83,328.33\text{ kW}$ | Làm mẫu số chuẩn hóa để tính hệ số xác định $R^2 = 1 - \frac{\text{MSE}}{\sigma_{\text{load}}^2}$ |
| **Giá trị Cực tiểu ($\min \text{load}_t$)** | $\min = 459.43\text{ kW} > 0$ | Đảm bảo chỉ số phần trăm `MAPE (%)` hoàn toàn xác định, không xảy ra lỗi chia cho $0$ |
| **Cấu trúc Đồ thị `PEMS-BAY`** | $207$ nút, $1,238$ cạnh có hướng, $\bar{v} = 62.46\text{ mph}$ | Định hình ma trận khuếch tán đồ thị hai chiều cho `DCRNN` để tối ưu `MAE`, `RMSE`, `MAPE` ở mốc 15, 30, 60 phút |

## 5. Used data — Dữ liệu sử dụng

| Tập dữ liệu | Đường dẫn | Phạm vi Sử dụng trong `Eda_Phase` | Module Thực thi |
|---|---|---|---|
| **Electricity Load + Weather** | `data/raw/electricity_load.csv` | Toàn bộ `35,065` giờ $\times$ `4` biến `[load, temperature, humidity, wind_speed]` để thống kê 13 chỉ tiêu, phân rã mùa vụ và vẽ biểu đồ EDA | `src/lab04/data_loader.py`, `src/lab04/eda.py`, `src/lab04/visualize.py` |
| **PEMS-BAY Speed Matrix** | `data/raw/pems-bay.h5` | Toàn bộ `8,640` bước (5 phút/bước) $\times$ `207` trạm cảm biến tốc độ xe (`3.0` đến `83.5 mph`) | `src/lab04/data_loader.py`, `src/lab04/eda.py` |
| **PEMS-BAY Sensor Graph** | `data/raw/sensor_graph/adj_mx_bay.pkl` | Ma trận kề khoảng cách `[207, 207]` với `1,238` cạnh kết nối có hướng (`sparsity = 97.11%`) | `src/lab04/data_loader.py`, `src/lab04/eda.py` |
