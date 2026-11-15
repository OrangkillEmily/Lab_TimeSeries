# Long Short-Term Memory (`LSTM.md` — Mô hình Deep Learning Chuỗi Tuần tự)

## 1. Purpose — Mục đích

**Long Short-Term Memory (`LSTM`)** đại diện cho nhóm **Mô hình Deep Learning (`Mô hình 3 — Mục 8`)** trong đề bài `practice 4.pdf` và **Slide 8, 9, 10 Chương 4–5** (`chapter4-5.pdf`).

Mục đích của `LSTM` trong dự án là học trực tiếp động lực học thời gian và sự phụ thuộc chéo giữa 4 biến `[load, temperature, humidity, wind_speed]` ngay trên cấu trúc **Tensor 3D `(batch_samples, 24 bước thời gian, 4 biến)`** mà không cần làm phẳng chuỗi hay phụ thuộc vào các đặc trưng trượt thủ công. Thông qua cơ chế ô nhớ dài hạn ($\mathbf{c}_t$) và 3 cổng điều khiển (*Forget, Input, Output Gates*), kiến trúc xếp chồng `LSTM(64) -> LSTM(32) -> Dense(1)` khắc phục hiện tượng triệt tiêu đạo hàm (*Vanishing Gradient*) của RNN truyền thống và nắm bắt trọn vẹn chu kỳ phụ tải điện theo ngày/tuần ([Machine Learning Mastery Multivariate LSTM](https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/) & [GeeksforGeeks PyTorch Time Series Forecasting](https://www.geeksforgeeks.org/data-analysis/time-series-forecasting-using-pytorch/)).

## 2. Input-Output

### Input

| Đầu vào (Input) | Định dạng & Kích thước | Mô tả Chi tiết |
|---|---|---|
| `X_lstm_train`, `y_train` | `Tensor` `(N_train, 24, 4)`, `(N_train, 1)` | Tensor 3D gồm 24 bước thời gian quá khứ $\times$ 4 biến đã chuẩn hóa Z-score từ [Electricity Load Diagrams Dataset](https://www.kaggle.com/datasets/jacopoferretti/electricity-load-diagrams-2011-2014-feature-engin) |
| `X_lstm_val`, `y_val` | `Tensor` `(N_val, 24, 4)`, `(N_val, 1)` | Tập kiểm định độc lập (15%) dùng để theo dõi `val_loss` (`MSE`) và kích hoạt `EarlyStopping` |
| `X_lstm_test`, `y_test` | `Tensor` `(N_test, 24, 4)`, `(N_test, 1)` | Tập kiểm tra độc lập (15%) dùng cho đánh giá cuối cùng và Walk-Forward Backtesting |
| `config.yaml -> lstm_model` | `Dict[str, Any]` | Cấu hình `lstm_units = [64, 32]`, `dense_units = 1`, `loss = "mse"`, `optimizer = "adam"`, `batch_size = 64`, `epochs = 60`, `patience = 10` |

### Output

| Đầu ra (Output) | Đường dẫn Lưu trữ | Mô tả Chi tiết |
|---|---|---|
| `Predicted Load` $\hat{y}_{t+1} \in \mathbb{R}^{N \times 1}$ | `reports/lab04/predictions/test_predictions_latest.csv` | Dự báo phụ tải điện ở bước thời gian $t+1$ (đã `inverse_transform` về đơn vị gốc) |
| `models/lab04/lstm_model.pt` / `.keras` | `models/lab04/lstm_model.pt` | Trọng số mạng `LSTM(64) -> LSTM(32) -> Dense(1)` tại `best_epoch` |
| `lstm_training_history.csv` | `models/lab04/lstm_training_history.csv` | Lịch sử hội tụ `train_mse`, `val_mse`, `train_mae`, `val_mae` qua từng epoch |
| `lstm_train_vs_val_loss.png` | `reports/lab04/figures/lstm_train_vs_val_loss.png` | Biểu đồ đối sánh đường cong huấn luyện và kiểm định để phân tích Overfitting (Câu hỏi 5 Mục 11) |

### Bảng Kiến trúc & Số lượng Tham số của Mạng `LSTM(64) -> LSTM(32) -> Dense(1)`

Với công thức đếm tham số của một tầng LSTM có kích thước đầu vào $d_{\text{in}}$ và số đơn vị ẩn $d_{\text{hid}}$ là $4 \times \left[(d_{\text{in}} + d_{\text{hid}})\times d_{\text{hid}} + d_{\text{hid}}\right]$:

| Tầng Mạng (Layer) | Đầu vào (Input Shape) | Đầu ra (Output Shape) | Công thức Tham số | Tổng Tham số |
|---|---|---|---|---:|
| **Layer 1: `LSTM(64, return_sequences=True)`** | `(batch, 24, 4)` | `(batch, 24, 64)` | $4 \times [(4 + 64)\times 64 + 64]$ | **`17,664`** |
| **Layer 2: `LSTM(32, return_sequences=False)`** | `(batch, 24, 64)` | `(batch, 32)` | $4 \times [(64 + 32)\times 32 + 32]$ | **`12,416`** |
| **Layer 3: `Dense(1, activation='linear')`** | `(batch, 32)` | `(batch, 1)` | $32 \times 1 + 1$ | **`33`** |
| **Tổng cộng (`Total Trainable Parameters`)** | — | — | — | **`30,113`** |

## 3. How to do — Quy trình thực hiện & Cơ sở toán học

### Sơ đồ luồng thực thi `Stacked LSTM (64 -> 32 -> 1)`

```text
          Input Tensor 3D: X_t in R^(Batch x 24 x 4)  [load, temp, humidity, wind]
                                        |
                                        v
          [Auto-GPU Dispatcher] Chuyển Tensor sang CUDA GPU (AMP FP16) / MPS / CPU
                                        |
                                        v
          [Layer 1] LSTM(64, return_sequences=True)
                    Cập nhật (h_tau^(1), c_tau^(1)) qua 24 bước thời gian tau = t-23 .. t
                    Xuất chuỗi trạng thái ẩn H^(1) in R^(Batch x 24 x 64)
                                        |
                                        v
          [Layer 2] LSTM(32, return_sequences=False)
                    Nén toàn bộ chuỗi 24 bước thành vector ngữ cảnh cuối h_24^(2) in R^(Batch x 32)
                                        |
                                        v
          [Layer 3] Dense(1, Linear): y_hat_{t+1} = W_d * h_24^(2) + b_d
                                        |
                                        v
          Tính MSE Loss -> Backpropagation Through Time (BPTT) -> Cập nhật Adam Optimizer
          Giám sát val_loss trên tập Validation (15%) -> EarlyStopping khôi phục Best Epoch
```

### Các bước thực hiện chi tiết

**Step 1 — Cơ chế Cổng Nhớ bên trong Tế bào LSTM tại mỗi bước thời gian $\tau \in \{t-23, \dots, t\}$**

Tại mỗi bước thời gian $\tau$, tế bào LSTM nhận vector đầu vào $\mathbf{x}_\tau$, trạng thái ẩn bước trước $\mathbf{h}_{\tau-1}$ và trạng thái ô nhớ $\mathbf{c}_{\tau-1}$ để tính 3 cổng điều khiển:
1. **Cổng Quên (Forget Gate $\mathbf{f}_\tau$):** Quyết định giữ lại bao nhiêu thông tin lịch sử từ $\mathbf{c}_{\tau-1}$:
   $$\mathbf{f}_\tau = \sigma\left(\mathbf{W}_f \mathbf{x}_\tau + \mathbf{U}_f \mathbf{h}_{\tau-1} + \mathbf{b}_f\right)$$
2. **Cổng Nhập (Input Gate $\mathbf{i}_\tau$) và Ứng viên Ô nhớ ($\tilde{\mathbf{c}}_\tau$):** Chọn lọc thông tin mới từ biến động điện năng và thời tiết tại bước $\tau$:
   $$\mathbf{i}_\tau = \sigma\left(\mathbf{W}_i \mathbf{x}_\tau + \mathbf{U}_i \mathbf{h}_{\tau-1} + \mathbf{b}_i\right), \qquad \tilde{\mathbf{c}}_\tau = \tanh\left(\mathbf{W}_c \mathbf{x}_\tau + \mathbf{U}_c \mathbf{h}_{\tau-1} + \mathbf{b}_c\right)$$
3. **Cập nhật Ô nhớ Dài hạn ($\mathbf{c}_\tau$):** Kết hợp tuyến tính giúp gradient truyền ngược qua 24 bước thời gian mà không bị suy giảm:
   $$\mathbf{c}_\tau = \mathbf{f}_\tau \odot \mathbf{c}_{\tau-1} + \mathbf{i}_\tau \odot \tilde{\mathbf{c}}_\tau$$
4. **Cổng Xuất (Output Gate $\mathbf{o}_\tau$) và Trạng thái Ẩn ($\mathbf{h}_\tau$):**
   $$\mathbf{o}_\tau = \sigma\left(\mathbf{W}_o \mathbf{x}_\tau + \mathbf{U}_o \mathbf{h}_{\tau-1} + \mathbf{b}_o\right), \qquad \mathbf{h}_\tau = \mathbf{o}_\tau \odot \tanh(\mathbf{c}_\tau)$$

**Step 2 — Tầng Đầu ra Tuyến tính & Tối ưu hóa Hàm mất mát MSE bằng Adam (Mục 8.2)**

Trạng thái ẩn tại bước cuối cùng của tầng LSTM thứ hai ($\mathbf{h}_t^{(2)} \in \mathbb{R}^{32}$) đi qua tầng `Dense(1)`:

$$\hat{y}_{t+1} = \mathbf{w}_d^T \mathbf{h}_t^{(2)} + b_d$$

Mạng được huấn luyện để cực tiểu hóa hàm mất mát **Mean Squared Error (MSE)** bằng thuật toán **Adam** ($\eta = 0.001$) thông qua lan truyền ngược theo thời gian (**Backpropagation Through Time — BPTT**):

$$\mathcal{L}_{\text{MSE}}(\Theta) = \frac{1}{B}\sum_{b=1}^{B}\left(y_{b, t+1} - \hat{y}_{b, t+1}\right)^2$$

**Step 3 — Tăng tốc Tự động trên GPU (`Auto-GPU`) & Kiểm soát Overfitting**

Khi chạy trên máy có GPU, mô hình tự động bật `torch.cuda.amp.autocast` (hoặc Keras GPU), giảm thời gian chạy mỗi epoch từ hàng chục giây trên CPU xuống chỉ còn vài phần giây trên GPU, đồng thời dùng `EarlyStopping(patience=10, restore_best_weights=True)` trên tập `Validation (15%)` để chống Overfitting.

## 4. Evaluation metrics — Các chỉ số đánh giá thực nghiệm của LSTM

Mô hình `LSTM` được đánh giá trên đầy đủ **5 chỉ số hồi quy chuẩn** (`MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²`) ở cả **Hold-out Test (`15%`)**, **5-Fold `TimeSeriesSplit`** và **Walk-Forward Backtesting**:

| Chỉ số Đánh giá | Công thức Toán học | Đặc điểm Hiệu năng của `LSTM` (Theo Slide 8–10 Chương 4–5) |
|---|---|---|
| **MSE (↓)** | $\frac{1}{N}\sum_{t=1}^{N}\left(y_t - \hat{y}_t^{\text{LSTM}}\right)^2$ | Là hàm mất mát trực tiếp khi huấn luyện mạng (`Loss: MSE`), giúp tối ưu trực diện sai số bình phương |
| **RMSE (↓)** | $\sqrt{\text{MSE}}$ | Nắm bắt mượt mà các đỉnh và đáy chu kỳ phụ tải 24 giờ nhờ trạng thái ô nhớ $\mathbf{c}_t$ |
| **MAE (↓)** | $\frac{1}{N}\sum_{t=1}^{N}\left|y_t - \hat{y}_t^{\text{LSTM}}\right|$ | Phản ánh độ lệch tuyệt đối thấp khi mạng học tốt tương tác ẩn giữa nhiệt độ, độ ẩm và điện năng |
| **MAPE (%) (↓)** | $\frac{100\%}{N}\sum_{t=1}^{N}\left|\frac{y_t - \hat{y}_t^{\text{LSTM}}}{y_t}\right|$ | Sai số phần trăm ổn định trên toàn bộ các khung giờ trong ngày |
| **$R^2$ Score (↑)** | $1 - \frac{\sum(y_t - \hat{y}_t^{\text{LSTM}})^2}{\sum(y_t - \bar{y})^2}$ | Đạt hệ số xác định cao nhờ khả năng tự học biểu diễn phi tuyến đa tầng |

## 5. Used data — Dữ liệu sử dụng

| Tập dữ liệu | Kích thước Tensor 3D | Vai trò đối với `LSTM` | Module Thực thi |
|---|---|---|---|
| **Electricity Train (`70%`)** | `(N_train, 24, 4)` | Cập nhật `30,113` tham số của mạng `LSTM(64) -> LSTM(32) -> Dense(1)` qua BPTT | `src/lab04/preprocess.py`, `src/lab04/model.py`, `src/lab04/train.py` |
| **Electricity Validation (`15%`)** | `(N_val, 24, 4)` | Giám sát `val_loss` sau mỗi epoch để chọn `best_epoch` và đánh giá Overfitting (Câu hỏi 5 Mục 11) | `src/lab04/train.py`, `src/lab04/visualize.py` |
| **Electricity Test (`15%`) & 5-Fold CV** | `(N_test, 24, 4)` & 5 Folds | Đánh giá độc lập 5 chỉ số, chạy Walk-Forward Backtesting và tổng hợp `Mean ± Std` qua 5 Folds | `src/lab04/evaluate.py`, `src/lab04/report.py` |
