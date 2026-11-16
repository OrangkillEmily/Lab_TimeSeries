# Extreme Gradient Boosting (`XGBoost.md` — Mô hình Machine Learning Dạng Bảng)

## 1. Purpose — Mục đích

**Extreme Gradient Boosting (`XGBoost`)** đại diện cho nhóm **Mô hình Machine Learning (`Mô hình 2 — Mục 7`)** trong đề bài `practice 4.pdf` và **Slide 7 & Slide 9 Chương 4–5** (`chapter4-5.pdf`).

Mục đích của `XGBoost` trong dự án là chuyển đổi bài toán dự báo chuỗi thời gian đa biến thành bài toán **Hồi quy có giám sát dạng bảng (Tabular Supervised Regression)** thông qua việc kết hợp cửa sổ trễ phẳng ($L = 24$ giờ $\times$ 4 biến) với bộ đặc trưng trượt chuyên biệt (`rolling_mean_6h`, `rolling_std_24h`, `load_trend_24h`). Nhờ cơ chế chia nhánh cây quyết định tăng cường gradient có điều chuẩn, `XGBoost` khắc phục hoàn toàn giả định tuyến tính và giả định dừng của `VAR`, đồng thời cung cấp bảng xếp hạng **Feature Importance** để giải thích trực tiếp biến thời tiết nào ảnh hưởng mạnh nhất đến điện năng ([Kaggle XGBoost for Multivariate Time Series](https://www.kaggle.com/code/davidandressanchez/xgboost-for-multivariate-ts)).

## 2. Input-Output

### Input

| Đầu vào (Input) | Định dạng & Kích thước | Mô tả Chi tiết |
|---|---|---|
| `X_xgb_train`, `y_train` | `np.ndarray` `(N_train, D_xgb)`, `(N_train,)` | Ma trận đặc trưng phẳng gồm 96 biến trễ ($\text{load}_{t-24:t}, \text{temp}_{t-24:t}, \text{hum}_{t-24:t}, \text{wind}_{t-24:t}$) + Rolling mean 6h + Rolling std 24h + Load trend 24h từ [Electricity Load Diagrams Dataset](https://www.kaggle.com/datasets/jacopoferretti/electricity-load-diagrams-2011-2014-feature-engin) |
| `X_xgb_val`, `y_val` | `np.ndarray` `(N_val, D_xgb)`, `(N_val,)` | Tập kiểm định độc lập (15%) dùng để theo dõi `val_rmse` và kích hoạt `early_stopping` |
| `X_xgb_test`, `y_test` | `np.ndarray` `(N_test, D_xgb)`, `(N_test,)` | Tập kiểm tra độc lập (15%) dùng để đánh giá 5 chỉ số cuối cùng và chạy Walk-Forward Backtesting |
| `config.yaml -> xgboost_model` | `Dict[str, Any]` | Cấu hình `n_estimators = 600`, `max_depth = 6`, `learning_rate = 0.03`, `tree_method = "hist"`, `early_stopping_rounds = 35` |

### Output

| Đầu ra (Output) | Đường dẫn Lưu trữ | Mô tả Chi tiết |
|---|---|---|
| `Predicted Load` $\hat{y}_{t+1} = \widehat{\text{load}}_{t+1}$ | `reports/lab04/predictions/test_predictions_latest.csv` | Giá trị phụ tải điện dự báo ở bước tiếp theo (đã `inverse_transform` về đơn vị gốc) |
| `models/lab04/xgboost_model.pkl` | `models/lab04/xgboost_model.pkl` | Đối tượng `XGBRegressor` đã huấn luyện tại vòng lặp tối ưu `best_iteration` |
| `xgboost_feature_importance.csv` & `.png` | `reports/lab04/metrics/` & `reports/figures/lab04/` | Bảng điểm và biểu đồ Top 20 đặc trưng quan trọng nhất (`Gain`, `Weight`, `Cover`) để trả lời Câu hỏi 1 & Câu hỏi 3 Mục 11 |

## 3. How to do — Quy trình thực hiện & Cơ sở toán học

### Sơ đồ luồng thực thi `XGBoost (Tabular Time Series Regression)`

```text
  Cửa sổ Lịch sử 24 giờ X_{t-24:t} = [load, temp, humidity, wind]
                                |
                                v
  [Bước 1] Kiến tạo Vector Đặc trưng Phẳng z_t (Mục 5.1, 5.2 & 7.1):
           - 96 Lag Features: load_{t-1..t-24}, temp_{t-1..t-24}, hum_{t-1..t-24}, wind_{t-1..t-24}
           - Rolling Features: rolling_mean_6h, rolling_std_24h, load_trend = load_t - load_{t-24}
           - Cross & Cyclical: load/temp ratio, sin/cos(hour_of_day), sin/cos(day_of_week)
                                |
                                v
  [Bước 2] Điều phối Auto-GPU cho XGBoost:
           Nếu có GPU NVIDIA -> tree_method="hist", device="cuda"
           Nếu chỉ có CPU    -> tree_method="hist", device="cpu", n_jobs=-1
                                |
                                v
  [Bước 3] Huấn luyện Cộng tính M Cây Quyết định CART (max_depth=6, lr=0.03):
           Giám sát eval_set=[(X_val, y_val)] với early_stopping_rounds=35
                                |
                                v
  [Bước 4] Trích xuất Feature Importance & Dự báo y_hat_{t+1} trên Test + Walk-Forward
```

### Các bước thực hiện chi tiết

**Step 1 — Định hình Bài toán Hồi quy Có Giám sát trên Cửa sổ Trễ & Rolling (Mục 7.1)**

Tại mỗi thời điểm $t$, đầu vào $\mathbf{z}_t \in \mathbb{R}^{D}$ là sự ghép nối giữa vector phẳng từ cửa sổ trễ $L=24$ và vector đặc trưng động lượng trượt:

$$\mathbf{z}_t = \left[\text{load}_{t-24:t},\ \text{temp}_{t-24:t},\ \text{hum}_{t-24:t},\ \text{wind}_{t-24:t},\ \text{RM}_{6,t},\ \text{RSTD}_{24,t},\ \left(\text{load}_t - \text{load}_{t-24}\right)\right]$$

với đầu ra mục tiêu là giá trị vô hướng $y_t = \text{load}_{t+1}$.

**Step 2 — Tối ưu hóa Hàm Mục tiêu Gradient Boosting Bậc Hai ([XGBoost Model Tutorial](https://xgboost.readthedocs.io/en/stable/tutorials/model.html))**

Dự đoán của mô hình sau $M$ cây quyết định cộng tính có dạng:

$$\hat{y}_i^{(M)} = \sum_{m=1}^{M} \eta f_m(\mathbf{z}_i), \qquad f_m \in \mathcal{F}$$

Tại bước thứ $m$, cây mới $f_m$ được xây dựng để cực tiểu hóa khai triển Taylor bậc hai của hàm mất mát bình phương kèm điều chuẩn $L_1 (\alpha)$ và $L_2 (\lambda)$:

$$\tilde{\mathcal{L}}^{(m)} = \sum_{i=1}^{N_{\text{train}}} \left[g_i f_m(\mathbf{z}_i) + \frac{1}{2} h_i f_m^2(\mathbf{z}_i)\right] + \gamma T_m + \frac{1}{2}\lambda \sum_{j=1}^{T_m} w_j^2 + \alpha \sum_{j=1}^{T_m} |w_j|$$

với đạo hàm bậc nhất $g_i = 2\left(\hat{y}_i^{(m-1)} - y_i\right)$ và đạo hàm bậc hai $h_i = 2$. Độ lợi thông tin (`Gain`) khi tách một nút lá thành hai nút con $I_L, I_R$ là:

$$\mathcal{G}_{\text{split}} = \frac{1}{2}\left[\frac{\left(\sum_{i \in I_L} g_i\right)^2}{\sum_{i \in I_L} h_i + \lambda} + \frac{\left(\sum_{i \in I_R} g_i\right)^2}{\sum_{i \in I_R} h_i + \lambda} - \frac{\left(\sum_{i \in I} g_i\right)^2}{\sum_{i \in I} h_i + \lambda}\right] - \gamma$$

**Step 3 — Kiểm soát Độ Phức tạp & Early Stopping trên Tập Validation (Mục 7.2)**

Điều chỉnh độ sâu tối đa `max_depth = 6` và số lượng cây `n_estimators = 600` kết hợp cơ chế **Early Stopping** trên tập `Validation (15%)`: nếu sau 35 vòng lặp liên tiếp sai số trên tập `Validation` không giảm thêm, mô hình tự động dừng và giữ lại số cây tối ưu $M^* = \text{best\_iteration}$.

## 4. Evaluation metrics — Các chỉ số đánh giá thực nghiệm của XGBoost

Hiệu năng của `XGBoost` được định lượng qua trọn bộ **5 chỉ số hồi quy chuẩn** (`MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²`) trên cả **Hold-out Test (`15%`)**, **5-Fold `TimeSeriesSplit`** và **Walk-Forward Backtesting**:

| Chỉ số Đánh giá | Công thức Toán học | Đặc điểm Hiệu năng của `XGBoost` (Theo Slide 7 & Slide 9 Chương 4–5) |
|---|---|---|
| **MSE (↓)** | $\frac{1}{N}\sum_{t=1}^{N}\left(y_t - \hat{y}_t^{\text{XGB}}\right)^2$ | Giảm mạnh so với `VAR` nhờ nắm bắt được quan hệ phi tuyến giữa nhiệt độ, độ ẩm và phụ tải điện |
| **RMSE (↓)** | $\sqrt{\text{MSE}}$ | Phản ánh độ chính xác cao trên dữ liệu chuỗi thời gian dạng bảng khi có đầy đủ Lag + Rolling features |
| **MAE (↓)** | $\frac{1}{N}\sum_{t=1}^{N}\left|y_t - \hat{y}_t^{\text{XGB}}\right|$ | Sai số tuyệt đối thấp nhờ khả năng phân hoạch chính xác các khung giờ cao điểm và thấp điểm |
| **MAPE (%) (↓)** | $\frac{100\%}{N}\sum_{t=1}^{N}\left|\frac{y_t - \hat{y}_t^{\text{XGB}}}{y_t}\right|$ | Thể hiện độ lệch phần trăm nhỏ và ít bị Overfitting hơn mạng học sâu khi kích thước mẫu vừa phải |
| **$R^2$ Score (↑)** | $1 - \frac{\sum(y_t - \hat{y}_t^{\text{XGB}})^2}{\sum(y_t - \bar{y})^2}$ | Đạt mức giải thích phương sai rất cao trên bài toán dự báo 1 bước ($\text{load}_{t+1}$) |

## 5. Used data — Dữ liệu sử dụng

| Tập dữ liệu | Cấu trúc Biểu diễn cho `XGBoost` | Vai trò Thực nghiệm | Module Thực thi |
|---|---|---|---|
| **Electricity Train (`70%`)** | Ma trận 2D `(N_train, D_xgb)` gồm 96 lags + rolling features | Huấn luyện cây quyết định tăng cường gradient trên GPU (`hist` + `cuda`) hoặc CPU | `src/lab04/preprocess.py`, `src/lab04/model.py`, `src/lab04/train.py` |
| **Electricity Validation (`15%`)** | Ma trận 2D `(N_val, D_xgb)` | Giám sát `early_stopping_rounds = 35` và chọn siêu tham số `max_depth`, `n_estimators` | `src/lab04/train.py` |
| **Electricity Test (`15%`) & 5-Fold CV** | Ma trận 2D `(N_test, D_xgb)` & 5 Folds `TimeSeriesSplit` | Đánh giá độc lập 5 chỉ số, chạy Walk-Forward Backtesting và trích xuất `Feature Importance` | `src/lab04/evaluate.py`, `src/lab04/visualize.py` |
