# Giai đoạn Đánh giá 5 Chỉ số & Giải đáp Câu hỏi Phân tích (`Evaluation_Phase.md`)

## 1. Purpose — Mục đích

**Giai đoạn Đánh giá (`Evaluation_Phase`)** thực hiện **Mục 10 (Đánh giá mô hình)**, **Mục 11 (6 Câu hỏi phân tích)** và **Mục 12 (Đánh giá Traffic Forecasting trên PEMS-BAY)** trong đề bài `practice 4.pdf`, kết hợp **Yêu cầu bổ sung số 4 (`5-Fold Cross-Validation`)** và **Yêu cầu bổ sung số 6 (Mở rộng thành bộ 5 chỉ số: `MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²`)**.

Mục đích cốt lõi của giai đoạn này là:
1. Định lượng toàn diện và công bằng hiệu năng của cả 3 nhóm mô hình dự báo phụ tải điện (`VAR`, `XGBoost`, `LSTM`) và mô hình giao thông nâng cao (`DCRNN`) trên cùng hệ quy chiếu **5 chỉ số hồi quy chuẩn**.
2. Đối sánh kết quả trên cả 3 chế độ kiểm định: **Hold-out Test (`15%`)**, **Temporal 5-Fold CV (`Mean ± Std`)** và **Walk-Forward Backtesting**.
3. Trả lời sâu sắc, có dẫn chứng toán học và đồ thị thực nghiệm cho **6 câu hỏi phân tích ở Mục 11** dựa trên lý thuyết **Chương 4–5** (`chapter4-5.pdf`).
4. Cung cấp dữ liệu đầu vào chuẩn xác tuyệt đối để `src/lab04/report.py` và `script/lab04/render_report.py` tự động xuất `LAB_04_REPORT.pdf`, `LAB_04_REPORT.html` và `LAB_04_REPORT.md`.

## 2. Input-Output

### Input

| Đầu vào (Input) | Định dạng & Kích thước | Mô tả Chi tiết |
|---|---|---|
| `test_predictions_latest.csv` | `pd.DataFrame` | Chuỗi giá trị thực tế $y_t$ và dự báo $\hat{y}_t$ (đã `inverse_transform` về đơn vị gốc) của `VAR`, `XGBoost`, `LSTM` trên tập Test |
| `walk_forward_predictions.csv` | `pd.DataFrame` | Kết quả dự báo cuốn chiếu từng bước từ giai đoạn `Inference_Phase` |
| `kfold_fold_predictions` (`Fold 1..5`) | `List[Dict[str, Any]]` | Kết quả dự báo trên 5 tập kiểm định $Val^{(1)} \dots Val^{(5)}$ của `TimeSeriesSplit(n_splits=5)` |
| `pems_bay_horizon_predictions.npz` | `Dict[str, np.ndarray]` | Kết quả dự báo tốc độ xe (mph) của `DCRNN` tại 3 chân trời $t+3$ (15m), $t+6$ (30m) và $t+12$ (60m) |

### Output

| Đầu ra (Output) | Đường dẫn Lưu trữ | Mô tả Chi tiết |
|---|---|---|
| `evaluation_latest.json` | `reports/lab04/metrics/evaluation_latest.json` | Hồ sơ JSON trung tâm lưu toàn bộ 5 chỉ số (`MSE`, `RMSE`, `MAE`, `MAPE`, `R²`) của Hold-out Test, 5-Fold CV, Walk-Forward và PEMS-BAY |
| `test_metrics_comparison.csv` | `reports/lab04/metrics/test_metrics_comparison.csv` | Bảng xếp hạng trực tiếp `VAR`, `XGBoost`, `LSTM` theo 5 chỉ số trên tập Test và Walk-Forward |
| `kfold_cv_results.csv` | `reports/lab04/metrics/kfold_cv_results.csv` | Bảng kết quả chi tiết từng Fold (`Fold 1..5`) kèm `Mean` và `Std` của 5 chỉ số |
| `pems_bay_metrics.csv` | `reports/lab04/metrics/pems_bay_metrics.csv` | Bảng kết quả `MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²` của `DCRNN` tại 15 phút, 30 phút, 60 phút |
| Bộ 8+ biểu đồ đánh giá 300 DPI | `reports/figures/lab04/` | Gồm 5 biểu đồ cột cho từng chỉ số (`model_comparison_mse.png`, `rmse`, `mae`, `mape`, `r2`), `kfold_cv_comparison.png`, `residual_analysis_all.png`, `pems_bay_horizon_comparison.png` |

## 3. How to do — Quy trình thực hiện & Giải đáp 6 Câu hỏi Phân tích (Mục 11)

### Sơ đồ luồng đánh giá tổng hợp (`Evaluation Workflow`)

```text
  Dự đoán Đã Khôi phục Đơn vị Gốc (VAR, XGBoost, LSTM, DCRNN)
                                |
        +-----------------------+-----------------------+
        |                       |                       |
        v                       v                       v
 [1. Hold-out Test 15%]  [2. 5-Fold TimeSeries]  [3. Walk-Forward & PEMS]
 Tính 5 chỉ số độc lập   Tính từng Fold k=1..5   Đánh giá cuốn chiếu t+1
 trên 15% cuối chuỗi     Tổng hợp Mean ± Std     & đa bước 15m, 30m, 60m
        |                       |                       |
        +-----------------------+-----------------------+
                                |
                                v
  Xuất JSON/CSV + Vẽ bộ biểu đồ so sánh MSE, RMSE, MAE, MAPE (%), R^2
  Tổng hợp phân tích trả lời 6 câu hỏi Mục 11 -> Kết xuất PDF / HTML / MD
```

### Khung Giải đáp Chuyên sâu 6 Câu hỏi Phân tích (Mục 11 `practice 4.pdf` & `chapter4-5.pdf`)

1. **Câu hỏi 1: Biến thời tiết nào ảnh hưởng mạnh nhất đến điện năng?**
   - **Cơ sở phân tích:** Kết hợp 3 bằng chứng định lượng từ pipeline gồm (a) hệ số tương quan tuyến tính & tương quan trễ $|r_{\text{load}, x_j}(\tau)|$ trong `eda_correlation_matrix.csv`, (b) kiểm định nhân quả Granger / độ lớn hệ số ma trận $\hat{\mathbf{A}}_i$ trong `VAR`, và (c) điểm `Gain Feature Importance` của `XGBoost`.
   - **Bản chất vật lý:** Biến **`temperature` (nhiệt độ)** có ảnh hưởng mạnh nhất tới phụ tải điện (`load`) do nhu cầu điều hòa không khí/sưởi ấm phản ứng trực tiếp và mạnh mẽ với biến thiên nhiệt độ (thường có độ trễ $1\text{–}3$ giờ theo **Slide 4 Chương 4–5**), tiếp theo là **`humidity` (độ ẩm)** làm gia tăng cảm giác oi bức với độ trễ $6\text{–}12$ giờ, trong khi `wind_speed` có tác động yếu hơn.

2. **Câu hỏi 2: Khi nào VAR thất bại?**
   - **Cơ sở lý thuyết (Slide 4 & Slide 5 Chương 4–5):** Mô hình $\text{VAR}(p)$ thất bại hoặc cho sai số `MSE`/`RMSE` cao trong 4 trường hợp:
     - **Quan hệ phi tuyến mạnh (Non-linearity):** Phụ tải điện phụ thuộc phi tuyến (hình chữ U hoặc ngưỡng bậc thang) vào nhiệt độ, trong khi VAR chỉ biểu diễn được tổ hợp tuyến tính $\sum \mathbf{A}_i \mathbf{X}_{t-i}$.
     - **Chuỗi không dừng và đa mùa vụ phức tạp (Non-stationarity & Multi-seasonality):** Dữ liệu có đồng thời chu kỳ ngày ($24\text{h}$) và chu kỳ tuần ($168\text{h}$) kèm nhiễu phương sai thay đổi.
     - **Độ trễ tác động khác nhau (Lag heterogeneity):** Nhiệt độ tác động sau $1\text{–}3\text{h}$ nhưng độ ẩm tác động sau $6\text{–}12\text{h}$; nếu tăng bậc trễ $p$ lớn để bao phủ toàn bộ thì số tham số $p \times K^2$ tăng nhanh gây quá khớp (*Overfitting*).

3. **Câu hỏi 3: XGBoost học được thông tin gì từ thời tiết?**
   - **Cơ sở lý thuyết (Slide 7 & Slide 9 Chương 4–5):** Thông qua phép chia nhánh cây quyết định trên không gian đặc trưng phẳng, `XGBoost` ([Kaggle XGBoost Multivariate TS](https://www.kaggle.com/code/davidandressanchez/xgboost-for-multivariate-ts)) học được:
     - **Các ngưỡng nhiệt độ phi tuyến (Non-linear Thresholds):** Ví dụ khi nhiệt độ vượt ngưỡng nắng nóng kết hợp độ ẩm cao tại các độ trễ $t-1, t-2, t-3$, cây quyết định tạo nhánh dự báo phụ tải tăng vọt.
     - **Tương tác giữa thời tiết và động lượng phụ tải:** Kết hợp các lag thời tiết với `rolling_mean_6h` (mức nền hiện tại), `rolling_std_24h` (độ biến động trong ngày) và `load_trend_24h` ($\text{load}_t - \text{load}_{t-24}$).

4. **Câu hỏi 4: LSTM học được chu kỳ nào (ngày, tuần)?**
   - **Cơ sở lý thuyết (Slide 8–10 Chương 4–5):**
     - Với cửa sổ đầu vào Tensor 3D kích thước $L = 24$ bước giờ (`(batch_samples, 24, 4)`), mạng `LSTM(64) -> LSTM(32) -> Dense(1)` học **trực tiếp và trọn vẹn chu kỳ ngày (Daily Seasonality — 24 giờ)** bên trong trạng thái ẩn $\mathbf{h}_t$ và ô nhớ dài hạn $\mathbf{c}_t$ (nắm bắt rõ nhịp sinh hoạt giờ cao điểm sáng/tối và giờ thấp điểm ban đêm).
     - Đối với **chu kỳ tuần (Weekly Seasonality — 168 giờ)**, do cửa sổ trượt trực tiếp có độ dài $L = 24 < 168$, LSTM nhận biết sự thay đổi giữa ngày thường và cuối tuần gián tiếp qua biên độ phụ tải của 24 giờ vừa qua (hoặc học trực tiếp nếu bổ sung kênh mã hóa thứ trong tuần / mở rộng cửa sổ trễ).

5. **Câu hỏi 5: LSTM có bị overfitting không?**
   - **Cơ sở phân tích (Slide 9 & 10 Chương 4–5):** Theo bảng so sánh trong bài giảng, các mô hình Deep Learning (`LSTM`) có nguy cơ **dễ bị Overfitting hơn `XGBoost`** khi dữ liệu nhỏ hoặc huấn luyện quá nhiều epochs vì số lượng tham số lớn.
   - **Kiểm chứng và khắc phục trong dự án:** Biểu đồ `lstm_train_vs_val_loss.png` đối chiếu trực tiếp `Train MSE` và `Validation MSE`. Nhờ chuẩn hóa `StandardScaler`, thêm tầng `Dropout(0.15)`, điều chuẩn `weight_decay` và cơ chế **`EarlyStopping(restore_best_weights=True)`** trên tập `Validation (15%)`, quá trình huấn luyện tự động dừng ngay tại điểm cực tiểu của `val_loss`, ngăn chặn hiện tượng Overfitting trước khi đánh giá trên tập `Test`.

6. **Câu hỏi 6: Sự khác biệt giữa Machine Learning và Deep Learning trong bài toán này?**
   - **Tổng hợp theo Slide 9 Chương 4–5:**
     - **Biểu diễn dữ liệu & Feature Engineering:** Machine Learning (`XGBoost`) bắt buộc phải làm phẳng chuỗi thành bảng 2D và phụ thuộc rất mạnh vào việc thiết kế đặc trưng thủ công (`Lag 24h`, `Rolling mean 6h`, `Rolling std 24h`, `Load trend`); ngược lại Deep Learning (`LSTM`, `DCRNN`) nhận trực tiếp Tensor 3D/4D và **tự động học biểu diễn thời gian – không gian** qua các tầng ẩn.
     - **Khả năng mở rộng (Scalability) & Dự báo đa bước (Multi-step Forecasting):** Với bài toán ít biến ($K=4$, dự báo 1 bước $\text{load}_{t+1}$) như Yêu cầu 1–11, `XGBoost` huấn luyện rất nhanh trên CPU/GPU và đạt độ chính xác cạnh tranh sát sao với `LSTM`. Tuy nhiên, khi chuyển sang bài toán Yêu cầu 12 (`PEMS-BAY` với hàng trăm biến cảm biến $K=207\text{–}325$ và dự báo đa bước $15, 30, 60$ phút), `XGBoost` gặp rào cản không gian đặc trưng khổng lồ và phải lặp từng bước, trong khi kiến trúc Deep Learning (`LSTM` / `TCN` / `DCRNN`) xử lý tự nhiên trong một mạng duy nhất.

## 4. Evaluation metrics — Công thức Toán học của 5 Chỉ số Đánh giá Chuẩn

Với $y_t$ là giá trị thực tế, $\hat{y}_t$ là giá trị dự báo (đều ở thang đo gốc sau `inverse_transform`) và $\bar{y} = \frac{1}{N}\sum_{t=1}^{N} y_t$, hệ thống tính toán đầy đủ 5 chỉ số:

| STT | Chỉ số (Metric) | Công thức Toán học | Quy tắc Xếp hạng & Ý nghĩa Học thuật |
|---:|---|---|---|
| 1 | **MSE (Mean Squared Error)** | $\text{MSE} = \frac{1}{N}\sum_{t=1}^{N}\left(y_t - \hat{y}_t\right)^2$ | **Càng thấp càng tốt ($\downarrow$):** Bình phương sai số, phạt rất nặng các sai lệch lớn tại đỉnh phụ tải |
| 2 | **RMSE (Root Mean Squared Error)** | $\text{RMSE} = \sqrt{\frac{1}{N}\sum_{t=1}^{N}\left(y_t - \hat{y}_t\right)^2}$ | **Càng thấp càng tốt ($\downarrow$):** Cùng đơn vị đo với biến mục tiêu (`kW/MW` hoặc `mph`), dễ diễn giải vật lý |
| 3 | **MAE (Mean Absolute Error)** | $\text{MAE} = \frac{1}{N}\sum_{t=1}^{N}\left|y_t - \hat{y}_t\right|$ | **Càng thấp càng tốt ($\downarrow$):** Độ lệch tuyệt đối trung bình, bền vững trước các điểm ngoại lai |
| 4 | **MAPE (Mean Absolute Percentage Error)** | $\text{MAPE} = \frac{100\%}{N}\sum_{t=1}^{N}\left|\frac{y_t - \hat{y}_t}{\max(|y_t|, \epsilon)}\right|$ | **Càng thấp càng tốt ($\downarrow$):** Sai số tương đối tính theo phần trăm (`%`), cho phép so sánh giữa các bộ dữ liệu khác thang đo |
| 5 | **$R^2$ Score (Coefficient of Determination)** | $R^2 = 1 - \frac{\sum_{t=1}^{N}\left(y_t - \hat{y}_t\right)^2}{\sum_{t=1}^{N}\left(y_t - \bar{y}\right)^2}$ | **Càng cao càng tốt ($\uparrow$, tối đa $1.0$):** Tỷ lệ phương sai của chuỗi thời gian thực tế được giải thích bởi mô hình |

## 5. Used data — Dữ liệu sử dụng

| Tập dữ liệu & Kết quả Kiểm định | Nguồn Dữ liệu | Vai trò trong `Evaluation_Phase` | Module Thực thi |
|---|---|---|---|
| **Electricity Test Set (`15%`) & Walk-Forward** | `data/raw/electricity_load.csv` ([UCI / Kaggle](https://www.kaggle.com/datasets/jacopoferretti/electricity-load-diagrams-2011-2014-feature-engin)) | Đánh giá chính thức 5 chỉ số của `VAR`, `XGBoost`, `LSTM` và trả lời 6 câu hỏi Mục 11 | `src/lab04/evaluate.py`, `src/lab04/report.py` |
| **Electricity 5-Fold CV (`TimeSeriesSplit`)** | 5 lát cắt kiểm định theo thời gian | Tính `Mean ± Std` của `MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²` để kiểm chứng độ ổn định | `src/lab04/evaluate.py`, `src/lab04/visualize.py` |
| **PEMS-BAY Test Set (`15m, 30m, 60m`)** | `data/raw/pems-bay.h5` ([PEMS-BAY](https://drive.google.com/file/d/1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq/view?usp=drive_link)) + `data/raw/sensor_graph/adj_mx_bay.pkl` | Đánh giá 5 chỉ số của mô hình `DCRNN` theo từng chân trời dự báo giao thông | `src/lab04/evaluate.py`, `src/lab04/report.py` |
