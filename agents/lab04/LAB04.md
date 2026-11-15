# THỰC HÀNH 4 (LAB04): DỰ BÁO CHUỖI THỜI GIAN ĐA BIẾN — ĐIỆN NĂNG, THỜI TIẾT & GIAO THÔNG ĐÔ THỊ
**(Multivariate Time Series Forecasting: Electricity Load + Weather & Advanced Spatio-Temporal Traffic Forecasting — Practice 4)**

> **Học phần:** Phân tích Chuỗi Thời gian (Time Series Analysis) — Trường Đại học Giao thông Vận tải TP.HCM (UTH)  
> **Giảng viên lý thuyết:** TS. Nguyễn Thị Khánh Tiên (`Chương 4–5: Mô hình Dự đoán Chuỗi Thời gian Đa biến`)  
> **Tài liệu gốc:** `practice 4.pdf` & `chapter4-5.pdf`  
> **Kho lưu trữ bộ dữ liệu chính (Yêu cầu 1–11):** [Electricity Load Diagrams Dataset 2011–2014 (UCI / Kaggle)](https://www.kaggle.com/datasets/jacopoferretti/electricity-load-diagrams-2011-2014-feature-engin) (`data/raw/electricity_load.csv`)  
> **Kho lưu trữ bộ dữ liệu nâng cao (Yêu cầu 12):** [PEMS-BAY California Highway Traffic Sensors (Google Drive)](https://drive.google.com/file/d/1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq/view?usp=drive_link) (`data/raw/pems-bay.h5`) & [DCRNN Sensor Graph Repository (GitHub)](https://github.com/liyaguang/dcrnn) (`data/raw/sensor_graph/adj_mx_bay.pkl`)

## 1. Phát biểu Bài toán (Problem Statement)

Bài thực hành **Lab 04** giải quyết bài toán **Dự báo Chuỗi Thời gian Đa biến (Multivariate Time Series Forecasting)**, nơi nhiều chuỗi thời gian được quan sát đồng thời trên cùng một trục thời gian và có sự **phụ thuộc chéo (Cross-dependence)** cũng như **độ trễ tác động khác nhau (Lag heterogeneity)** giữa các biến. Hệ thống được thiết kế thành 2 bài toán thực nghiệm hoàn chỉnh:

### 1.1. Bài toán Chính (Yêu cầu 1 đến Yêu cầu 11): Dự báo Phụ tải Điện năng kết hợp Thời tiết
Tại mỗi thời điểm $t$ (theo tần suất giờ), vector quan sát gồm $K = 4$ biến đồng thời từ bộ dữ liệu [Electricity Load Diagrams Dataset (UCI / Kaggle)](https://www.kaggle.com/datasets/jacopoferretti/electricity-load-diagrams-2011-2014-feature-engin):

$$\mathbf{X}_t = \left[\text{load}_t,\ \text{temp}_t,\ \text{humidity}_t,\ \text{wind}_t\right] \in \mathbb{R}^{4}$$

- **Mục tiêu dự báo (Single-step Target Forecasting):** Sử dụng lịch sử cửa sổ trễ $L = 24$ giờ gần nhất để dự đoán điện năng tiêu thụ ở bước thời gian tiếp theo ($t+1$):
  $$\mathbf{X}_{t-24:t} \implies \hat{y}_{t+1} = \widehat{\text{load}}_{t+1}$$
- **Ba nhóm mô hình đối sánh trọng tâm:**
  1. **VAR (Vector Autoregression)** — Mô hình thống kê cổ điển ([Statsmodels VAR Documentation](https://www.statsmodels.org/stable/vector_ar.html)).
  2. **XGBoost (Extreme Gradient Boosting)** — Mô hình Machine Learning dạng bảng ([Kaggle XGBoost for Multivariate Time Series](https://www.kaggle.com/code/davidandressanchez/xgboost-for-multivariate-ts)).
  3. **LSTM (Long Short-Term Memory)** — Mô hình Deep Learning chuỗi tuần tự trên Tensor 3D ([Machine Learning Mastery Multivariate LSTM Guide](https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/)).

### 1.2. Bài toán Nâng cao (Yêu cầu 12): Dự báo Giao thông Đa bước trên Đồ thị Cảm biến (Traffic Forecasting)
Sử dụng bộ dữ liệu chuẩn [PEMS-BAY (California Highway Sensors)](https://drive.google.com/file/d/1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq/view?usp=drive_link) lưu tại `data/raw/pems-bay.h5` kết hợp ma trận kề khoảng cách mạng lưới đường bộ `data/raw/sensor_graph/adj_mx_bay.pkl` từ công trình gốc [Diffusion Convolutional Recurrent Neural Network (Li et al., ICLR 2018 — arXiv:1707.01926)](https://arxiv.org/abs/1707.01926):

- Tại mỗi mốc thời gian $t$ (cách nhau mỗi **5 phút**), vector quan sát biểu diễn tốc độ xe (mph) tại $N$ trạm cảm biến cao tốc:
  $$\mathbf{X}_t = \left[\text{speed}_{1,t},\ \text{speed}_{2,t},\ \dots,\ \text{speed}_{N,t}\right]$$
- **Mục tiêu dự báo đa chân trời (Multi-step Spatio-Temporal Forecasting):** Từ 12 bước thời gian lịch sử (60 phút qua), dự báo đồng thời tốc độ xe tại toàn bộ các trạm ở 3 chân trời tương lai $t+3$ (**15 phút**), $t+6$ (**30 phút**) và $t+12$ (**60 phút**):
  $$\left[\mathbf{X}_{t-11},\ \dots,\ \mathbf{X}_t;\ \mathcal{G}\right] \implies \left[\hat{\mathbf{X}}_{t+3},\ \hat{\mathbf{X}}_{t+6},\ \hat{\mathbf{X}}_{t+12}\right]$$

## 2. Đặc tả Bộ Dữ liệu (Datasets Specification)

| Thuộc tính | Bài toán Chính (Yêu cầu 1–11) | Bài toán Nâng cao (Yêu cầu 12) |
|---|---|---|
| **Tên bộ dữ liệu** | [Electricity Load Diagrams 2011–2014 (UCI / Kaggle)](https://www.kaggle.com/datasets/jacopoferretti/electricity-load-diagrams-2011-2014-feature-engin) | [PEMS-BAY California Highway Sensors](https://drive.google.com/file/d/1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq/view?usp=drive_link) + [Sensor Graph `adj_mx_bay.pkl`](https://github.com/liyaguang/dcrnn) |
| **Đường dẫn lưu trữ** | `data/raw/electricity_load.csv` | `data/raw/pems-bay.h5` & `data/raw/sensor_graph/adj_mx_bay.pkl` |
| **Biến quan sát** | `load` (điện năng), `temperature` (nhiệt độ), `humidity` (độ ẩm), `wind_speed` (tốc độ gió) | Tốc độ dòng phương tiện (mph) tại các trạm cảm biến cao tốc (`325` trạm gốc hoặc subset `207` trạm theo cấu hình) |
| **Độ phân giải thời gian** | Mỗi **1 giờ** (`1H`), chu kỳ ngày $24\text{h}$, chu kỳ tuần $168\text{h}$ | Mỗi **5 phút** (`5min`), `288` bước/ngày |
| **Kích thước cửa sổ trễ ($L$)** | $L = 24$ bước (24 giờ quá khứ) | $L = 12$ bước (60 phút quá khứ) |
| **Đầu ra dự báo ($H$)** | $\text{load}_{t+1}$ (1 giờ tiếp theo) | $\mathbf{X}_{t+3}, \mathbf{X}_{t+6}, \mathbf{X}_{t+12}$ (15, 30, 60 phút phía trước) |

## 3. Các Mô hình Triển khai & Cơ sở Lý thuyết (Chương 4–5)

### 3.1. Mô hình 1 — VAR (`Vector Autoregression` — Statistical Baseline, Yêu cầu 6)
- **Nguyên lý ([Slide 5 Chương 4–5 & Statsmodels VAR](https://www.statsmodels.org/stable/vector_ar.html)):** Mở rộng mô hình tự hồi quy AR sang chuỗi đa biến, giả định mỗi biến là tổ hợp tuyến tính của quá khứ chính nó và quá khứ các biến khác:
  $$\mathbf{X}_t = \mathbf{c} + \sum_{i=1}^{p} \mathbf{A}_i \mathbf{X}_{t-i} + \boldsymbol{\varepsilon}_t, \qquad \mathbf{A}_i \in \mathbb{R}^{4 \times 4}$$
- **Quy trình bắt buộc:**
  1. Kiểm tra tính dừng của từng chuỗi bằng kiểm định **ADF (Augmented Dickey-Fuller test)** với mức ý nghĩa $\alpha = 0.05$.
  2. Nếu chuỗi không dừng ($p\text{-value} \ge 0.05$) $\rightarrow$ thực hiện sai phân (**Differencing** $\Delta \mathbf{X}_t = \mathbf{X}_t - \mathbf{X}_{t-1}$) trước khi huấn luyện và cộng tích lũy ngược (**Inverse Differencing**) khi dự báo.
  3. Chọn bậc trễ tối ưu $p^*$ dựa trên tiêu chuẩn thông tin **AIC** hoặc **BIC**.
  4. Dự báo toàn bộ vector $[\text{load}, \text{temp}, \text{humidity}, \text{wind}]$, sau đó chỉ trích xuất cột `load` để tính 5 chỉ số đánh giá.

### 3.2. Mô hình 2 — XGBoost (`Extreme Gradient Boosting` — Machine Learning, Yêu cầu 7)
- **Nguyên lý ([Slide 7 & 9 Chương 4–5](https://www.kaggle.com/code/davidandressanchez/xgboost-for-multivariate-ts)):** Chuyển đổi bài toán dự báo chuỗi đa biến thành bài toán **Hồi quy có giám sát dạng bảng (Tabular Supervised Regression)**.
- **Biểu diễn đặc trưng đầu vào (Yêu cầu 5.1 & 5.2):**
  - **Cửa sổ trễ phẳng (Flattened Lag Window $L = 24$):** $\left[\text{load}_{t-1}, \dots, \text{load}_{t-24}, \text{temp}_{t-1}, \dots, \text{temp}_{t-24}, \text{humidity}_{t-1}, \dots, \text{wind}_{t-24}\right] \in \mathbb{R}^{96}$.
  - **Rolling & Trend Features chuyên biệt cho XGBoost:**
    - `rolling_mean_6h`: Trung bình trượt 6 giờ gần nhất của phụ tải điện.
    - `rolling_std_24h`: Độ lệch chuẩn trượt 24 giờ gần nhất phản ánh độ biến động trong ngày.
    - `load_trend_24h`: $\text{load}_t - \text{load}_{t-24}$ nắm bắt xu hướng tăng/giảm so với cùng giờ ngày hôm trước.
- **Huấn luyện & Tối ưu:** Điều chỉnh `max_depth`, `n_estimators`, `learning_rate`, bật tăng tốc phần cứng tự động (`tree_method="hist"`, `device="cuda"` khi có GPU) và sử dụng **Early Stopping** trên tập `Validation`.

### 3.3. Mô hình 3 — LSTM (`Long Short-Term Memory` — Deep Learning, Yêu cầu 8)
- **Nguyên lý ([Slide 8, 9 & 10 Chương 4–5](https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/)):** Giữ nguyên cấu trúc không gian – thời gian của chuỗi đa biến dưới dạng **Tensor 3D** $\mathcal{X} \in \mathbb{R}^{\text{batch\_samples} \times 24 \times 4}$, học trực tiếp phụ thuộc dài hạn qua cơ chế cổng nhớ (*Forget Gate, Input Gate, Output Gate*).
- **Kiến trúc mạng chuẩn đề bài:**
  $$\text{Input }(24, 4) \longrightarrow \text{LSTM}(64, \text{return\_sequences=True}) \longrightarrow \text{LSTM}(32) \longrightarrow \text{Dense}(1)$$
- **Cấu hình huấn luyện:** Hàm mất mát `Loss = MSE`, bộ tối ưu `Optimizer = Adam`, giám sát `EarlyStopping` trên tập `Validation` và tự động chạy trên **GPU (CUDA AMP FP16)** khi máy có GPU.

### 3.4. Mô hình 4 — DCRNN (`Diffusion Convolutional Recurrent Neural Network` — Yêu cầu 12 Nâng cao)
- **Nguyên lý ([Li et al., ICLR 2018 — arXiv:1707.01926](https://arxiv.org/abs/1707.01926) & [PyTorch DCRNN Implementation](https://github.com/chnsh/DCRNN_PyTorch)):**
  - Mô hình hóa sự phụ thuộc không gian giữa các trạm cảm biến giao thông bằng quá trình khuếch tán bước ngẫu nhiên hai chiều (**Bidirectional Random Walk Diffusion Convolution**) trên ma trận kề có hướng $\mathbf{W}$ nạp từ `data/raw/sensor_graph/adj_mx_bay.pkl`:
    $$\mathbf{X}_{:, p} \star_{\mathcal{G}} f_{\boldsymbol{\theta}} = \sum_{k=0}^{K-1} \left(\theta_{k,1}\left(\mathbf{D}_O^{-1}\mathbf{W}\right)^k + \theta_{k,2}\left(\mathbf{D}_I^{-1}\mathbf{W}^T\right)^k\right)\mathbf{X}_{:, p}$$
  - Thay thế phép nhân ma trận tuyến tính trong cổng Reset ($r_t$) và cổng Update ($u_t$) của mạng **GRU** bằng phép **Diffusion Convolution** để tạo thành tế bào **DCGRU (Diffusion Convolutional Gated Recurrent Unit)**.
  - Kiến trúc **Sequence-to-Sequence (Encoder–Decoder)** kèm kỹ thuật **Scheduled Sampling** giúp giảm thiểu sai số tích lũy khi dự báo xa ở các mốc **15 phút ($t+3$)**, **30 phút ($t+6$)** và **60 phút ($t+12$)**.

## 4. Cơ chế Chia Dữ liệu, K-Fold Cross-Validation & Walk-Forward Backtesting

Hệ thống thực thi đồng thời **3 cơ chế kiểm định không rò rỉ tương lai**:
1. **Chronological Hold-out Split (Yêu cầu 4.2):** Chia dữ liệu theo đúng thứ tự thời gian, tuyệt đối không xáo trộn (`shuffle=False`), theo tỷ lệ **Train 70% – Validation 15% – Test 15%**.
2. **Temporal K-Fold Cross-Validation (Yêu cầu bổ sung 4):** Áp dụng **`TimeSeriesSplit(n_splits=5)`** ([Scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)) theo cơ chế cửa sổ mở rộng (*Expanding Window*). Tại mỗi Fold $k \in \{1..5\}$, bộ chuẩn hóa `StandardScaler` được `fit()` độc lập chỉ trên $Train^{(k)}$ trước khi `transform()` sang $Val^{(k)}$, xuất bảng kết quả từng Fold và **`Mean ± Std`** cho toàn bộ 5 chỉ số.
3. **Backtesting / Walk-Forward Validation (Yêu cầu 9):** Mô phỏng quá trình vận hành dự báo thực tế trên tập `Test`:
   $$\text{Train} \longrightarrow \text{Predict } \hat{y}_{t+1} \longrightarrow \text{Cập nhật quan sát thực } \mathbf{X}_{t+1} \text{ vào chuỗi} \longrightarrow \text{Predict } \hat{y}_{t+2} \longrightarrow \dots$$

## 5. Hệ thống 5 Chỉ số Đánh giá & 6 Câu hỏi Phân tích Học thuật

### 5.1. Bộ 5 Chỉ số Đánh giá Bắt buộc (Yêu cầu 10 & Yêu cầu bổ sung 6)
Toàn bộ dự đoán $\hat{y}_t$ đều được biến đổi ngược (`inverse_transform`) về đơn vị gốc trước khi tính 5 chỉ số trên cùng tập dữ liệu:
1. **MSE (Mean Squared Error $\downarrow$):** $\text{MSE} = \frac{1}{N}\sum_{t=1}^{N}(y_t - \hat{y}_t)^2$
2. **RMSE (Root Mean Squared Error $\downarrow$):** $\text{RMSE} = \sqrt{\frac{1}{N}\sum_{t=1}^{N}(y_t - \hat{y}_t)^2}$
3. **MAE (Mean Absolute Error $\downarrow$):** $\text{MAE} = \frac{1}{N}\sum_{t=1}^{N}|y_t - \hat{y}_t|$
4. **MAPE (Mean Absolute Percentage Error $\downarrow$):** $\text{MAPE} = \frac{100\%}{N}\sum_{t=1}^{N}\left|\frac{y_t - \hat{y}_t}{\max(|y_t|, \epsilon)}\right|$
5. **$R^2$ Score (Coefficient of Determination $\uparrow$):** $R^2 = 1 - \frac{\sum_{t=1}^{N}(y_t - \hat{y}_t)^2}{\sum_{t=1}^{N}(y_t - \bar{y})^2}$

### 5.2. Sáu Câu hỏi Phân tích Chuyên sâu (Yêu cầu 11)
Báo cáo tự động (`report.md`, `report.html`, `report.pdf`) và `docs/lab04/Evaluation_Phase.md` bắt buộc giải đáp định lượng 6 câu hỏi:
1. Biến thời tiết nào ảnh hưởng mạnh nhất đến điện năng? *(Phân tích qua ma trận tương quan Pearson, Granger Causality / hệ số VAR và Feature Importance của XGBoost)*.
2. Khi nào VAR thất bại? *(Khi quan hệ giữa thời tiết và phụ tải điện phi tuyến mạnh, có độ trễ khác nhau giữa các biến - Lag heterogeneity, hoặc chuỗi có đa mùa vụ phức tạp và nhiễu không đồng nhất)*.
3. XGBoost học được thông tin gì từ thời tiết? *(Học được ngưỡng nhiệt độ phi tuyến làm tăng nhu cầu làm mát/sưởi ấm thông qua chia nhánh cây trên các lag của `temperature`, `humidity`, kết hợp xu hướng ngắn hạn `rolling_mean_6h` và `load_trend_24h`)*.
4. LSTM học được chu kỳ nào (ngày, tuần)? *(Cửa sổ 24 bước giúp trạng thái ẩn $h_t$ và ô nhớ $c_t$ nắm bắt trực tiếp chu kỳ ngày 24h; chu kỳ tuần 168h được phản ánh qua chuỗi trạng thái liên tiếp và độ lệch phụ tải giữa ngày làm việc và cuối tuần)*.
5. LSTM có bị overfitting không? *(Đánh giá trực tiếp qua khoảng cách giữa đường cong `Train MSE Loss` và `Validation MSE Loss` cùng cơ chế kiểm soát bằng `EarlyStopping` và `Dropout`)*.
6. Sự khác biệt giữa Machine Learning và Deep Learning trong bài toán này? *(Đối chiếu toàn diện theo Slide 9 Chương 4–5: Biểu diễn dữ liệu Tabular vs Tensor 3D, sự phụ thuộc vào Feature Engineering thủ công vs tự học biểu diễn, khả năng mở rộng sang bài toán đa biến lớn và dự báo đa bước trên PEMS-BAY)*.
