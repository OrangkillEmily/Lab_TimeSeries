# Diffusion Convolutional Recurrent Neural Network (`DCRNN.md` — Bài tập 12 Nâng cao: Traffic Forecasting)

## 1. Purpose — Mục đích

**Diffusion Convolutional Recurrent Neural Network (`DCRNN`)** là thuật toán học sâu Không gian – Thời gian (*Spatio-Temporal Graph Deep Learning*) giải quyết **Mục 12 (Bài tập nâng cao – Traffic Forecasting)** trong đề bài `practice 4.pdf`, dựa trực tiếp trên công trình gốc [Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting (Yaguang Li et al., ICLR 2018 — arXiv:1707.01926)](https://arxiv.org/abs/1707.01926) và kho mã nguồn chính thức [liyaguang/DCRNN](https://github.com/liyaguang/dcrnn) cùng [chnsh/DCRNN_PyTorch](https://github.com/chnsh/DCRNN_PyTorch).

Mục đích của `DCRNN` là giải quyết bài toán dự báo đa biến quy mô lớn và đa bước (*Multi-step Vector Forecasting* — **Slide 4 & Slide 8–9 Chương 4–5**): từ lịch sử 12 bước thời gian (60 phút, mỗi bước 5 phút) của mạng lưới cảm biến giao thông [PEMS-BAY Dataset](https://drive.google.com/file/d/1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq/view?usp=drive_link), dự báo đồng thời tốc độ xe tại toàn bộ các trạm ở 3 chân trời tương lai: **15 phút ($t+3$)**, **30 phút ($t+6$)** và **60 phút ($t+12$)**.

## 2. Input-Output

### Input

| Đầu vào (Input) | Định dạng & Kích thước | Mô tả Chi tiết |
|---|---|---|
| `data/raw/pems-bay.h5` | `pd.DataFrame` `(T_bay, N_sensors)` | Chuỗi thời gian tốc độ xe (mph) thu thập mỗi 5 phút tại các trạm cảm biến cao tốc Bay Area (`325` trạm gốc hoặc subset `207` trạm theo cấu hình đề bài) từ [PEMS-BAY Google Drive](https://drive.google.com/file/d/1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq/view?usp=drive_link) |
| `data/raw/sensor_graph/adj_mx_bay.pkl` | `tuple` `(sensor_ids, sensor_id_to_ind, adj_mx)` | Ma trận kề có hướng $\mathbf{W} \in \mathbb{R}^{N \times N}$ tính từ khoảng cách mạng lưới đường bộ theo hàm Thresholded Gaussian Kernel từ [liyaguang/DCRNN](https://github.com/liyaguang/dcrnn) |
| Cửa sổ đầu vào $\mathcal{X}_{t-11:t}$ | `Tensor` `(Batch, 12, N_sensors, C)` | 12 bước thời gian quá khứ (60 phút) gồm tốc độ xe đã chuẩn hóa kết hợp đặc trưng thời điểm trong ngày (`time_in_day`) |
| `config.yaml -> pems_bay_advanced` | `Dict[str, Any]` | Cấu hình `filter_type = "dual_random_walk"`, `max_diffusion_step = 2`, `num_rnn_layers = 2`, `rnn_units = 64`, `forecast_horizons = [3, 6, 12]` |

### Output

| Đầu ra (Output) | Đường dẫn Lưu trữ | Mô tả Chi tiết |
|---|---|---|
| Dự báo đa bước $\left[\hat{\mathbf{X}}_{t+3}, \hat{\mathbf{X}}_{t+6}, \hat{\mathbf{X}}_{t+12}\right]$ | `reports/lab04/predictions/pems_bay_horizon_predictions.npz` | Tốc độ xe dự báo (mph) tại toàn bộ các trạm ở 3 mốc **15 phút**, **30 phút** và **60 phút** phía trước |
| `models/lab04/dcrnn_pems_bay.pt` | `models/lab04/dcrnn_pems_bay.pt` | Trọng số mạng `DCRNN (DCGRU Encoder-Decoder)` đã huấn luyện trên GPU/CPU |
| `pems_bay_metrics.csv` | `reports/lab04/metrics/pems_bay_metrics.csv` | Bảng kết quả đầy đủ 5 chỉ số (`MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²`) tại 3 chân trời 15m, 30m, 60m |
| `pems_bay_horizon_comparison.png` | `reports/figures/lab04/pems_bay_horizon_comparison.png` | Biểu đồ trực quan hóa hiệu năng dự báo giao thông và độ suy giảm sai số theo chân trời dự báo |

## 3. How to do — Quy trình thực hiện & Cơ sở toán học

### Sơ đồ kiến trúc `DCRNN (Diffusion Convolutional Recurrent Neural Network)`

```text
  Ma trận kề đường bộ adj_mx_bay.pkl (W)          Chuỗi 12 bước quá khứ [X_{t-11} .. X_t]
                  |                                                   |
                  v                                                   v
  Tính 2 Ma trận Chuyển tiếp Khuếch tán:          [Encoder: 2 tầng DCGRU xếp chồng]
  - Xuôi dòng (Downstream): P_f = D_O^{-1} W      Đọc tuần tự 12 bước với phép tích chập
  - Ngược dòng (Upstream) : P_b = D_I^{-1} W^T    khuếch tán đồ thị bậc K = 2
                  |                                                   |
                  +-------------------------+-------------------------+
                                            |
                                            v
                  Trạng thái ẩn cuối của Encoder H_t khởi tạo cho Decoder
                                            |
                                            v
                  [Decoder: 2 tầng DCGRU + Scheduled Sampling]
                  Sinh tuần tự 12 bước tương lai: X_hat_{t+1} ... X_hat_{t+12}
                  Trích xuất 3 mốc mục tiêu: t+3 (15m), t+6 (30m), t+12 (60m)
```

### Các bước thực hiện chi tiết

**Step 1 — Xây dựng Ma trận Kề từ Khoảng cách Đường bộ (`adj_mx_bay.pkl`)**

Trong bài báo [DCRNN (Li et al., ICLR 2018)](https://arxiv.org/abs/1707.01926), mạng lưới cảm biến giao thông được biểu diễn dưới dạng đồ thị có hướng có trọng số $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathbf{W})$ với $|\mathcal{V}| = N$ trạm. Trọng số cạnh $W_{ij}$ trong `adj_mx_bay.pkl` được tính từ khoảng cách lộ trình thực tế $\text{dist}(v_i, v_j)$ qua nhân Gaussian có ngưỡng cắt $\kappa = 0.1$:

$$W_{ij} = \begin{cases} \exp\left(-\dfrac{\text{dist}(v_i, v_j)^2}{\sigma^2}\right), & \text{nếu } \exp\left(-\dfrac{\text{dist}(v_i, v_j)^2}{\sigma^2}\right) \ge 0.1 \\ 0, & \text{ngược lại} \end{cases}$$

**Step 2 — Phép Tích chập Khuếch tán Hai chiều trên Đồ thị (`Bidirectional Diffusion Convolution`)**

Dòng giao thông không chỉ lan truyền xuôi theo chiều xe chạy mà sóng xung kích ùn tắc (*traffic shockwaves*) còn lan truyền ngược dòng về phía các trạm phía trước. Với ma trận bậc ra $\mathbf{D}_O = \text{diag}(\mathbf{W}\mathbf{1})$ và ma trận bậc vào $\mathbf{D}_I = \text{diag}(\mathbf{W}^T\mathbf{1})$, phép tích chập khuếch tán bậc $K$ trên tín hiệu đồ thị $\mathbf{X}_{:, p} \in \mathbb{R}^{N}$ với bộ lọc $f_{\boldsymbol{\theta}}$ được định nghĩa là:

$$\mathbf{X}_{:, p} \star_{\mathcal{G}} f_{\boldsymbol{\theta}} = \sum_{k=0}^{K-1} \left(\theta_{k,1}\left(\mathbf{D}_O^{-1}\mathbf{W}\right)^k + \theta_{k,2}\left(\mathbf{D}_I^{-1}\mathbf{W}^T\right)^k\right)\mathbf{X}_{:, p}$$

trong đó $\boldsymbol{\theta} \in \mathbb{R}^{K \times 2}$ là các tham số học được của bộ lọc không gian.

**Step 3 — Tế bào DCGRU (`Diffusion Convolutional Gated Recurrent Unit`)**

Thay thế phép nhân ma trận thông thường trong mạng **GRU** bằng phép tích chập khuếch tán $\star_{\mathcal{G}}$ để đồng thời nắm bắt phụ thuộc không gian và thời gian tại mỗi bước $t$:

$$\mathbf{r}_t = \sigma\left(\boldsymbol{\Theta}_r \star_{\mathcal{G}} [\mathbf{X}_t, \mathbf{H}_{t-1}] + \mathbf{b}_r\right)$$
$$\mathbf{u}_t = \sigma\left(\boldsymbol{\Theta}_u \star_{\mathcal{G}} [\mathbf{X}_t, \mathbf{H}_{t-1}] + \mathbf{b}_u\right)$$
$$\mathbf{C}_t = \tanh\left(\boldsymbol{\Theta}_c \star_{\mathcal{G}} [\mathbf{X}_t, (\mathbf{r}_t \odot \mathbf{H}_{t-1})] + \mathbf{b}_c\right)$$
$$\mathbf{H}_t = \mathbf{u}_t \odot \mathbf{H}_{t-1} + (1 - \mathbf{u}_t) \odot \mathbf{C}_t$$

với $\mathbf{r}_t$ là cổng đặt lại (*Reset Gate*), $\mathbf{u}_t$ là cổng cập nhật (*Update Gate*) và $\mathbf{H}_t \in \mathbb{R}^{N \times d_{\text{rnn}}}$ là trạng thái ẩn của toàn bộ $N$ trạm tại thời điểm $t$.

**Step 4 — Kiến trúc Sequence-to-Sequence (Encoder–Decoder) & Scheduled Sampling**

Để dự báo chuỗi 12 bước tương lai ($\hat{\mathbf{X}}_{t+1} \dots \hat{\mathbf{X}}_{t+12}$), `DCRNN` sử dụng kiến trúc **Encoder–Decoder** gồm các tầng `DCGRU`. Trong quá trình huấn luyện trên GPU (`CUDA AMP FP16`), kỹ thuật **Scheduled Sampling** nạp quan sát thực tế $\mathbf{X}_{t+h-1}$ vào Decoder với xác suất $\epsilon_i = \frac{\tau}{\tau + \exp(i / \tau)}$ và nạp dự đoán của chính mô hình $\hat{\mathbf{X}}_{t+h-1}$ với xác suất $1 - \epsilon_i$, giúp xóa bỏ sự lệch phân phối giữa giai đoạn huấn luyện và suy luận thực tế.

## 4. Evaluation metrics — Các chỉ số đánh giá đa chân trời (15m, 30m, 60m)

Mô hình `DCRNN` được đánh giá đầy đủ bằng bộ **5 chỉ số hồi quy chuẩn** (`MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²`) tính riêng cho từng chân trời dự báo $h \in \{3, 6, 12\}$ (tương ứng **15 phút**, **30 phút**, **60 phút**) và trung bình chung, có loại trừ các điểm cảm biến bị khuyết (`masked evaluation` trên các quan sát $y_{n,t} > 0$):

| Chân trời Dự báo (Horizon) | Ký hiệu Toán học | Bộ 5 Chỉ số Đánh giá (`MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²`) | Benchmark Tham chiếu Bài báo Gốc [DCRNN ICLR 2018](https://github.com/liyaguang/dcrnn) trên PEMS-BAY |
|---|---|---|---|
| **15 Phút Phía trước ($h = 3$)** | $\hat{\mathbf{X}}_{t+3}$ | Tính `MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²` ở bước thứ 3 (`3 x 5m = 15m`) | `MAE = 1.31 mph`, `RMSE = 2.74 mph`, `MAPE = 2.73%` |
| **30 Phút Phía trước ($h = 6$)** | $\hat{\mathbf{X}}_{t+6}$ | Tính `MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²` ở bước thứ 6 (`6 x 5m = 30m`) | `MAE = 1.65 mph`, `RMSE = 3.77 mph`, `MAPE = 3.68%` |
| **60 Phút Phía trước ($h = 12$)** | $\hat{\mathbf{X}}_{t+12}$ | Tính `MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²` ở bước thứ 12 (`12 x 5m = 60m`) | `MAE = 1.97 mph`, `RMSE = 4.60 mph`, `MAPE = 4.64%` |

## 5. Used data — Dữ liệu sử dụng

| Tập dữ liệu & Thành phần | Đường dẫn Lưu trữ | Cấu hình Chia & Vai trò đối với `DCRNN` | Module Thực thi |
|---|---|---|---|
| **PEMS-BAY Speed Time Series** | `data/raw/pems-bay.h5` | Chia theo đúng thứ tự thời gian `Train 70% - Val 15% - Test 15%`, tạo cửa sổ $12 \to 12$ bước | `src/lab04/data_loader.py`, `src/lab04/preprocess.py` |
| **PEMS-BAY Sensor Road Graph** | `data/raw/sensor_graph/adj_mx_bay.pkl` | Nạp bằng `pickle.load(encoding='latin1')`, căn chỉnh đúng thứ tự `sensor_ids` với các cột của `.h5` và tính $(\mathbf{P}_f, \mathbf{P}_b)$ | `src/lab04/data_loader.py`, `src/lab04/model.py` |
| **PEMS-BAY Test Windows** | `data/processed/pems_bay/` | Suy luận trên GPU/CPU và xuất bảng so sánh 5 chỉ số tại 15m, 30m, 60m vào báo cáo tổng hợp | `src/lab04/train.py`, `src/lab04/evaluate.py`, `src/lab04/report.py` |
