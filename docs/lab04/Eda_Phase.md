# Giai đoạn Khám phá và Thấu hiểu Dữ liệu (`EDA_Phase.md` — Data Understanding)

## 1. Purpose — Mục đích

**Giai đoạn Khám phá Dữ liệu (`EDA_Phase`)** thực hiện **Mục 3 (Data Understanding)** trong đề bài `practice 4.pdf` kết hợp nền tảng lý thuyết **Chương 4–5: Mô hình dự đoán chuỗi thời gian đa biến** (`chapter4-5.pdf`).

Mục đích cốt lõi của giai đoạn này là kiểm định chất lượng dữ liệu thô, khám phá quy luật mùa vụ kép (**chu kỳ ngày 24 giờ** và **chu kỳ tuần 168 giờ**) cũng như định lượng **sự phụ thuộc chéo (Cross-dependence)** và **độ trễ tác động khác nhau (Lag heterogeneity)** giữa các biến thời tiết (`temperature`, `humidity`, `wind_speed`) đối với phụ tải điện năng (`load`) trên bộ dữ liệu [Electricity Load Diagrams Dataset (UCI / Kaggle)](https://www.kaggle.com/datasets/jacopoferretti/electricity-load-diagrams-2011-2014-feature-engin) và mạng lưới cảm biến giao thông [PEMS-BAY Dataset (Google Drive)](https://drive.google.com/file/d/1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq/view?usp=drive_link).

## 2. Input-Output

### Input

| Đầu vào (Input) | Định dạng & Kích thước | Mô tả Chi tiết |
|---|---|---|
| `data/raw/electricity_load.csv` | `pd.DataFrame` `(T, 4)` | Chuỗi thời gian đa biến theo giờ gồm 4 cột: `load`, `temperature`, `humidity`, `wind_speed` từ [Kaggle UCI Electricity Load Diagrams](https://www.kaggle.com/datasets/jacopoferretti/electricity-load-diagrams-2011-2014-feature-engin) |
| `data/raw/pems-bay.h5` | `pd.DataFrame` `(T_bay, N_sensors)` | Dữ liệu tốc độ dòng phương tiện (mph) thu thập mỗi 5 phút tại các trạm cảm biến cao tốc từ [PEMS-BAY Google Drive](https://drive.google.com/file/d/1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq/view?usp=drive_link) |
| `data/raw/sensor_graph/adj_mx_bay.pkl` | `tuple` `(sensor_ids, id_to_ind, adj_mx)` | Ma trận kề khoảng cách mạng lưới đường bộ từ [DCRNN GitHub Repository](https://github.com/liyaguang/dcrnn) |
| `config.yaml -> preprocessing` | `Dict[str, Any]` | Cấu hình ngưỡng phát hiện ngoại lai (`iqr_multiplier = 1.5`, `zscore_threshold = 3.0`) và chu kỳ mùa vụ (`daily = 24`, `weekly = 168`) |

### Output

| Đầu ra (Output) | Đường dẫn Lưu trữ | Mô tả Chi tiết |
|---|---|---|
| `eda_descriptive_statistics.csv` | `reports/lab04/metrics/eda_descriptive_statistics.csv` | Bảng thống kê mô tả (`Count`, `Mean`, `Std`, `Min`, `25%`, `Median`, `75%`, `Max`, `Skewness`, `Kurtosis`, `Missing_Count`, `Outlier_Count`) |
| `eda_correlation_matrix.csv` | `reports/lab04/metrics/eda_correlation_matrix.csv` | Ma trận tương quan tuyến tính Pearson và tương quan hạng Spearman giữa 4 biến `[load, temp, humidity, wind]` |
| `eda_summary.json` | `reports/lab04/metrics/eda_summary.json` | Hồ sơ JSON tổng hợp kết quả kiểm tra Missing values, Outliers, Seasonality và cấu trúc đồ thị `PEMS-BAY` |
| `load_timeseries_overview.png` | `reports/figures/lab04/load_timeseries_overview.png` | Biểu đồ diễn biến phụ tải điện `load` theo thời gian (toàn chuỗi và lát cắt zoom-in 14 ngày) |
| `load_vs_temperature_scatter.png` | `reports/figures/lab04/load_vs_temperature_scatter.png` | Biểu đồ phân tán & đường xu hướng phi tuyến giữa `load` và `temperature` kèm phân tích tương quan trễ (Cross-correlation) |
| `correlation_heatmap.png` | `reports/figures/lab04/correlation_heatmap.png` | Biểu đồ nhiệt ma trận tương quan (Correlation Matrix) giữa 4 biến điện năng và thời tiết |
| `seasonality_daily_weekly.png` | `reports/figures/lab04/seasonality_daily_weekly.png` | Biểu đồ rã chuỗi (Seasonal Decomposition) và hồ sơ phụ tải trung bình theo giờ trong ngày (24h) & các thứ trong tuần (168h) |

## 3. How to do — Quy trình thực hiện & Cơ sở toán học

### Sơ đồ luồng thực thi `EDA_Phase`

```text
  data/raw/electricity_load.csv (4 biến) + data/raw/pems-bay.h5 & data/raw/sensor_graph/adj_mx_bay.pkl
                                      |
                                      v
  [Bước 1] Kiểm tra Missing Values & Tính liên tục của trục thời gian (DatetimeIndex)
                                      |
                                      v
  [Bước 2] Phát hiện Outliers bằng quy tắc IQR (1.5 x IQR) và Z-Score (|z| > 3.0)
                                      |
                                      v
  [Bước 3] Phân tích Seasonality: Chu kỳ ngày (T_d = 24h) & Chu kỳ tuần (T_w = 168h)
                                      |
                                      v
  [Bước 4] Trực quan hóa 3 biểu đồ bắt buộc (Load theo thời gian, Load vs Temp, Heatmap)
           + Phân tích tương quan chéo có độ trễ (Lag Cross-Correlation tau = 0..24h)
```

### Các bước thực hiện chi tiết

**Step 1 — Kiểm tra Missing Values và Tính toàn vẹn trục thời gian**

Kiểm tra tỷ lệ khuyết dữ liệu trên từng biến $k \in \{\text{load}, \text{temp}, \text{humidity}, \text{wind}\}$ và khoảng cách bước thời gian $\Delta t = t_i - t_{i-1}$:

$$\text{Missing Ratio}_k = \frac{1}{T}\sum_{t=1}^{T} \mathbb{I}\left(x_{k,t} = \text{NaN}\right) \times 100\%$$

**Step 2 — Kiểm tra Giá trị Ngoại lai (Outliers Detection)**

Áp dụng đồng thời hai tiêu chuẩn thống kê cho từng biến để nhận diện các giờ phụ tải đỉnh đột biến hoặc thời tiết cực đoan:
- **Phương pháp khoảng tứ phân vị (IQR Rule):**
  $$\text{IQR}_k = Q_{3,k} - Q_{1,k}, \qquad x_{k,t} \notin \left[Q_{1,k} - 1.5\cdot\text{IQR}_k,\ Q_{3,k} + 1.5\cdot\text{IQR}_k\right]$$
- **Phương pháp điểm chuẩn hóa (Z-Score Rule):**
  $$z_{k,t} = \frac{|x_{k,t} - \mu_k|}{\sigma_k} > 3.0$$

**Step 3 — Phân tích Tính Mùa vụ (Seasonality: Chu kỳ Ngày & Chu kỳ Tuần)**

Thực hiện phân rã chuỗi thời gian cộng tính ([Statsmodels Seasonal Decompose](https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.seasonal_decompose.html)) và tính hồ sơ trung bình theo khung giờ trong ngày $h \in \{0, \dots, 23\}$ cùng ngày trong tuần $d \in \{0, \dots, 6\}$:

$$\text{load}_t = T_t + S_t^{(\text{daily, } 24)} + S_t^{(\text{weekly, } 168)} + R_t$$

**Step 4 — Định lượng Mối quan hệ Điện năng – Thời tiết & Độ trễ Tác động (Lag Heterogeneity)**

Theo **Slide 4 Chương 4–5**, nhiệt độ ảnh hưởng tới điện năng sau $1\text{–}3$ giờ và độ ẩm ảnh hưởng sau $6\text{–}12$ giờ. Do đó, bên cạnh hệ số tương quan Pearson tức thời ($\tau = 0$), hệ thống tính thêm hệ số tương quan chéo theo độ trễ $\tau \in \{0, 1, \dots, 24\}$:

$$r_{\text{load}, x_j}(\tau) = \frac{\sum_{t=\tau+1}^{T}\left(\text{load}_t - \overline{\text{load}}\right)\left(x_{j, t-\tau} - \bar{x}_j\right)}{\sqrt{\sum_{t=\tau+1}^{T}\left(\text{load}_t - \overline{\text{load}}\right)^2}\sqrt{\sum_{t=\tau+1}^{T}\left(x_{j, t-\tau} - \bar{x}_j\right)^2}}$$

## 4. Evaluation metrics — Các chỉ số kiểm định trong giai đoạn EDA

Giai đoạn EDA không trực tiếp huấn luyện mô hình dự báo nhưng cung cấp các chỉ số thống kê nền tảng quyết định cách thiết kế tiền xử lý và lựa chọn mô hình chấm điểm theo 5 chỉ số hồi quy (`MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²`):

| Chỉ số Thống kê / Kiểm định | Công thức / Thước đo | Vai trò Định hướng cho 5 Chỉ số Dự báo (`MSE`, `RMSE`, `MAE`, `MAPE`, `R²`) |
|---|---|---|
| **Hệ số Tương quan Pearson ($r_{ij}$) & Cross-Correlation ($r(\tau)$)** | $r_{ij} \in [-1, 1]$ | Xác định biến thời tiết chủ lực (trả lời Câu hỏi 1 Mục 11) giúp giảm `MSE` và `RMSE` khi đưa vào `VAR`, `XGBoost`, `LSTM` |
| **Biên độ Dao động & Phương sai Phụ tải ($\sigma_{\text{load}}^2$)** | $\sigma_{\text{load}}^2 = \frac{1}{T}\sum(\text{load}_t - \overline{\text{load}})^2$ | Là mẫu số chuẩn hóa trực tiếp để tính hệ số xác định $R^2 = 1 - \frac{\text{MSE}}{\sigma_{\text{load}}^2}$ |
| **Tỷ lệ Giá trị Gần Không ($\min |\text{load}_t|$)** | Kiểm tra $\text{load}_t > 0$ | Đảm bảo chỉ số phần trăm `MAPE (%)` ổn định, không bị bùng nổ số học khi chia cho giá trị quá nhỏ |
| **Tỷ lệ Khuyết & Mật độ Đồ thị `PEMS-BAY`** | Số cạnh khác không trong `adj_mx_bay.pkl` | Định hình ma trận chuyển tiếp bước ngẫu nhiên hai chiều cho `DCRNN` để tối ưu `MAE`, `RMSE`, `MAPE` ở các mốc 15, 30, 60 phút |

## 5. Used data — Dữ liệu sử dụng

| Tập dữ liệu | Đường dẫn | Phạm vi Sử dụng trong `EDA_Phase` | Module Thực thi |
|---|---|---|---|
| **Electricity Load + Weather** | `data/raw/electricity_load.csv` | Toàn bộ chuỗi 4 biến `[load, temperature, humidity, wind_speed]` để kiểm tra chất lượng dữ liệu, mùa vụ ngày/tuần và tương quan thời tiết | `src/lab04/data_loader.py`, `src/lab04/eda.py`, `src/lab04/visualize.py` |
| **PEMS-BAY Speed Matrix** | `data/raw/pems-bay.h5` | Kiểm tra phân phối tốc độ xe (mph), tỷ lệ điểm `0` (missing sensor readings) và chu kỳ giờ cao điểm sáng/chiều | `src/lab04/data_loader.py`, `src/lab04/eda.py` |
| **PEMS-BAY Sensor Graph** | `data/raw/sensor_graph/adj_mx_bay.pkl` | Thống kê số nút trạm cảm biến, số cạnh kết nối có hướng và phân phối bậc vào/ra (`in-degree` / `out-degree`) | `src/lab04/data_loader.py`, `src/lab04/eda.py` |
