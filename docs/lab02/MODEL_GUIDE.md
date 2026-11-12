# CẨM NANG HỌC THUẬT & HƯỚNG DẪN MÔ HÌNH CHUỖI THỜI GIAN (LAB02 MODEL GUIDE)

Tài liệu này giải thích chi tiết bằng tiếng Việt toàn bộ 13 khái niệm, phương pháp kiểm định, họ mô hình và kiến trúc kỹ thuật trong **LAB02**, gắn liền trực tiếp với kết quả thực nghiệm trên hai bộ dữ liệu **Dataset 1 (`Monthly Retail Sales.csv`)** và **Dataset 2 (`AirPassengers.csv`)**.

---

## 1. AR Model — Mô hình Tự hồi quy ($\text{AR}(p)$)

- **What (Nó là gì?)**:
  Mô hình Tự hồi quy bậc $p$ ($\text{Autoregressive Model}$) dự báo giá trị của chuỗi tại thời điểm $t$ bằng một tổ hợp tuyến tính của chính $p$ giá trị quá khứ gần nhất của chuỗi đó cộng với nhiễu ngẫu nhiên $\varepsilon_t$:
  $$W_t = c + \phi_1 W_{t-1} + \phi_2 W_{t-2} + \dots + \phi_p W_{t-p} + \varepsilon_t, \quad \varepsilon_t \sim \text{WN}(0, \sigma^2)$$
  Trong đó $W_t = \Delta^d Y_t$ là chuỗi đã được lấy sai phân bậc $d$ để đạt tính dừng (khi đó mô hình tương đương $\text{ARIMA}(p, d, 0)$).
- **Why (Tại sao cần?)**:
  Chuỗi thời gian kinh tế và vận tải có tính quán tính (inertia): doanh số hoặc sản lượng tháng này chịu ảnh hưởng trực tiếp từ quy mô của 1–2 tháng liền trước.
- **How (Hoạt động như thế nào?)**:
  Thuật toán ước lượng hợp lý cực đại ($\text{MLE}$) tìm các trọng số tự hồi quy $\phi_1, \dots, \phi_p$ sao cho sai số dự báo một bước trong tập Train nhỏ nhất. Khi dự báo nhiều bước vào tương lai trên tập Test, mô hình dùng đệ quy chính giá trị vừa dự báo $\hat{Y}_{t+1}$ để tính $\hat{Y}_{t+2}$.
- **Where (Được sử dụng ở bước nào?)**:
  Triển khai tại `src/lab02_models/ar_model.py` (hàm `fit_ar_model`, `run_ar_experiment`), được gọi trong Phần Nâng cấp Mô hình của cả hai Notebook.
- **When (Khi nào nên sử dụng?)**:
  Khi đồ thị **PACF** trên chuỗi đã dừng cắt cụt (cuts off) rõ rệt sau độ trễ $p$, trong khi **ACF** giảm dần (tails off). Tuyệt đối không dùng $\text{AR}(p)$ với $d=0$ trên chuỗi có xu hướng không dừng.
- **Example (Ví dụ thực tế trong LAB02)**:
  - Trên **Dataset 1 (`Monthly Retail Sales`)**: Sai phân bậc 1 ($d=1$) có PACF vượt ngưỡng tại lag 1 ($\text{PACF}_1 = -0.32$) và lag 2 ($\text{PACF}_2 = -0.20$), gợi ý chạy `AR(1) [ARIMA(1,1,0)]` và `AR(2) [ARIMA(2,1,0)]`.
  - Trên **Dataset 2 (`AirPassengers`)**: Sai phân bậc 1 ($d=1$) có PACF vượt ngưỡng tại lag 1 ($\text{PACF}_1 = 0.27$) và lag 2 ($\text{PACF}_2 = -0.22$), gợi ý chạy `AR(1) [ARIMA(1,1,0)]` và `AR(2) [ARIMA(2,1,0)]`.

---

## 2. MA Model — Mô hình Trung bình trượt ($\text{MA}(q)$)

- **What (Nó là gì?)**:
  Mô hình Trung bình trượt bậc $q$ ($\text{Moving Average Model}$) biểu diễn giá trị hiện tại của chuỗi dừng $W_t = \Delta^d Y_t$ như một hàm tuyến tính của cú sốc hiện tại $\varepsilon_t$ và $q$ cú sốc (sai số dự báo) trong quá khứ:
  $$W_t = \mu + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \theta_2 \varepsilon_{t-2} + \dots + \theta_q \varepsilon_{t-q}$$
  > **Cảnh báo học thuật**: Không được nhầm lẫn mô hình hồi quy thống kê $\text{MA}(q)$ với kỹ thuật làm trơn trung bình trượt ($\text{Moving Average Smoothing / Rolling Mean}$) ở LAB01!
- **Why (Tại sao cần?)**:
  Trong thực tế, những cú sốc bất thường (ví dụ: một đợt khuyến mãi lớn trong bán lẻ hoặc sự kiện thời tiết ảnh hưởng hàng không) tạo ra dư chấn kéo dài 1–3 tháng rồi tự tắt hẳn. $\text{MA}(q)$ giúp hiệu chỉnh dự báo dựa trên sai số vừa mắc phải ở kỳ trước.
- **How (Hoạt động như thế nào?)**:
  Vì các sai số quá khứ $\varepsilon_{t-j}$ không quan sát được trực tiếp, mô hình khởi tạo $\varepsilon_0 = 0$ và tính lặp lại chuỗi phần dư $\varepsilon_t = W_t - \hat{W}_t$ để tối ưu hóa hàm hợp lý $\text{MLE}$. Đặc tính quan trọng của $\text{MA}(q)$ là có **bộ nhớ hữu hạn đúng bằng $q$ bước**: từ bước dự báo $h > q$ trở đi trên tập Test, thành phần MA bằng $0$.
- **Where (Được sử dụng ở bước nào?)**:
  Triển khai tại `src/lab02_models/ma_model.py` (`fit_ma_model`, `run_ma_experiment`).
- **When (Khi nào nên sử dụng?)**:
  Khi đồ thị **ACF** của chuỗi đã dừng cắt cụt (cuts off) ngay sau độ trễ $q$, còn **PACF** giảm dần (tails off).
- **Example (Ví dụ thực tế trong LAB02)**:
  Ở cả `Monthly Retail Sales` ($\text{ACF}_1 = -0.32$) và `AirPassengers` ($\text{ACF}_1 = 0.27$), ở độ trễ ngắn hạn đồ thị ACF đều có gai rõ rệt tại lag 1, làm cơ sở huấn luyện `MA(1) [ARIMA(0,1,1)]`.

---

## 3. ARIMA Model — Mô hình Tự hồi quy Tích hợp Trung bình trượt ($\text{ARIMA}(p, d, q)$)

- **What (Nó là gì?)**:
  $\text{ARIMA}(p, d, q)$ kết hợp đồng thời ba thành phần trên chuỗi thời gian phi mùa vụ:
  - $p$: Bậc tự hồi quy ($\text{AR}$).
  - $d$: Bậc sai phân thường ($\text{Integrated}$) để khử xu hướng tuyến tính ($W_t = \Delta^d Y_t$).
  - $q$: Bậc trung bình trượt ($\text{MA}$).
  Phương trình tổng quát:
  $$\Delta^d Y_t = c + \sum_{i=1}^p \phi_i \Delta^d Y_{t-i} + \varepsilon_t + \sum_{j=1}^q \theta_j \varepsilon_{t-j}$$
- **Why (Tại sao cần?)**:
  Nhiều chuỗi thời gian vừa có quán tính tự hồi quy ($\text{AR}$) vừa có hiệu ứng điều chỉnh cú sốc ngắn hạn ($\text{MA}$) sau khi đã lấy sai phân khử xu hướng ($\text{I}$). Kết hợp cả $\text{AR}(p)$ và $\text{MA}(q)$ giúp mô hình biểu diễn cấu trúc tương quan ngắn hạn với số lượng tham số tiết kiệm hơn (nguyên lý *Parsimony*).
- **How (Hoạt động như thế nào?)**:
  Lấy sai phân bậc $d$ trên chuỗi gốc $\rightarrow$ ước lượng đồng thời $(\phi_1, \dots, \phi_p, \theta_1, \dots, \theta_q, \sigma^2)$ bằng $\text{MLE}$ $\rightarrow$ tích phân ngược (cộng dồn sai phân) để trả về dự báo trên thang đo gốc $Y_t$.
- **Where (Được sử dụng ở bước nào?)**:
  Triển khai tại `src/lab02_models/arima_model.py` và `src/lab02_utils/modeling.py` (Bài 5 của cả hai Notebook).
- **When (Khi nào nên sử dụng?)**:
  Khi chuỗi có xu hướng (cần $d \ge 1$) và tự tương quan ngắn hạn nhưng **không có chu kỳ mùa vụ** (hoặc dùng làm mô hình cơ sở *baseline* để chứng minh sự cần thiết của SARIMA trên chuỗi có mùa vụ).
- **Example (Ví dụ thực tế trong LAB02)**:
  - Trên **Monthly Retail Sales**: `ARIMA(1,1,1)` và `ARIMA(2,1,1)` đều hội tụ nhưng cho đường dự báo đi ngang quanh mức `17.5`, bị $R^2 = -0.25$ và `RMSE = 4.62` (lớn hơn cả độ lệch chuẩn tập Test `4.18`) vì bỏ lỡ các đỉnh mua sắm tháng 12 và tháng 1.
  - Trên **AirPassengers**: `ARIMA(1,1,1)` và `ARIMA(2,1,1)` cho $R^2$ âm (`-0.56` và `-0.12`) vì không bắt được chu kỳ đỉnh mùa hè (tháng 7–8).

---

## 4. SARIMA Model — Mô hình ARIMA Mùa vụ ($\text{SARIMA}(p, d, q)(P, D, Q)_s$)

- **What (Nó là gì?)**:
  $\text{SARIMA}(p, d, q)(P, D, Q)_s$ mở rộng ARIMA bằng cách nhân thêm đa thức tự hồi quy mùa vụ $\Phi_P(L^s)$, sai phân mùa vụ $\Delta_s^D = (1 - L^s)^D$ và trung bình trượt mùa vụ $\Theta_Q(L^s)$ tại chu kỳ $s$:
  $$\phi_p(L)\Phi_P(L^s) (1 - L)^d (1 - L^s)^D Y_t = \theta_q(L)\Theta_Q(L^s)\varepsilon_t$$
  Ý nghĩa 7 tham số:
  - $(p, d, q)$: Bậc AR, sai phân thường và MA ở độ trễ ngắn hạn (tháng liền kề $t-1, t-2\dots$).
  - $P$: Bậc tự hồi quy mùa vụ (sự phụ thuộc vào cùng tháng năm trước $t-12, t-24\dots$).
  - $D$: Bậc sai phân mùa vụ ($Y_t - Y_{t-s}$), dùng để khử chu kỳ lặp lại hàng năm.
  - $Q$: Bậc trung bình trượt mùa vụ (hấp thụ cú sốc mùa vụ và khử hiệu ứng quá sai phân ở lag $s$).
  - $s$: Độ dài chu kỳ mùa vụ ($s = 12$ đối với dữ liệu tháng có chu kỳ năm).
- **Why (Tại sao cần?)**:
  Khắc phục hoàn toàn điểm mù của ARIMA trên các chuỗi có tính mùa vụ mạnh như doanh số bán lẻ và sản lượng hành khách hàng không.
- **How (Hoạt động như thế nào?)**:
  Khử xu hướng bằng $d=1$ và khử mùa vụ bằng $D=1, s=12$ (tiêu hao $1 + 12 = 13$ quan sát đầu tiên), sau đó ước lượng mô hình không gian trạng thái ($\text{SARIMAX}$). Đối với chuỗi mùa vụ dạng nhân có phương sai phình to hình phễu (**Multiplicative Seasonality**), áp dụng biến đổi $\ln(Y_t)$ trước khi fit (**Log-SARIMA**) rồi biến đổi ngược $\hat{Y}_t = \exp(\hat{Y}^{(\log)}_t)$.
- **Where (Được sử dụng ở bước nào?)**:
  Triển khai tại `src/lab02_models/sarima_model.py` (`fit_sarima_model`, `run_sarima_experiment`) và Bài 6–7 của hai Notebook.
- **When (Khi nào nên sử dụng?)**:
  Khi chuỗi có chu kỳ mùa vụ rõ rệt trên đồ thị EDA và xuất hiện các đỉnh tự tương quan lặp lại tại bội số của $s$ ($12, 24, 36$) trên ACF.
- **Example (Ví dụ thực tế trong LAB02)**:
  - **Dataset 1 (`Monthly Retail Sales`)**: `SARIMA(0,1,1)x(0,1,1)_12` kéo giảm `RMSE` từ `4.62` (của ARIMA) xuống `1.84`, nâng $R^2$ từ `-0.25` lên `0.8012`, và với biến đổi `Log-SARIMA(0,1,1)x(0,1,1)_12` tiếp tục nâng $R^2$ lên trên `0.90`.
  - **Dataset 2 (`AirPassengers`)**: `SARIMA(1,1,3)x(0,1,1)_12` nâng $R^2$ từ `-0.12` lên `0.90` (`RMSE = 24.85`), và `Log-SARIMA(1,1,0)x(0,1,1)_12` đạt độ chính xác kỷ lục $R^2 \approx 0.96$ (`RMSE ≈ 15.12`, `MAPE ≈ 2.81%`).

---

## 5. ADF Test — Kiểm định Nghiệm đơn vị Augmented Dickey-Fuller

- **What (Nó là gì?)**:
  Kiểm định thống kê định lượng xem chuỗi thời gian có chứa nghiệm đơn vị (**Unit Root**) hay không:
  - **Giả thuyết $H_0$**: Chuỗi có nghiệm đơn vị ($\gamma = 0$) $\rightarrow$ **Chuỗi KHÔNG DỪNG (Non-stationary)**.
  - **Giả thuyết $H_1$**: Chuỗi không có nghiệm đơn vị ($\gamma < 0$) $\rightarrow$ **Chuỗi DỪNG (Stationary)**.
- **Why (Tại sao cần?)**:
  Không thể chỉ nhìn bằng mắt thường để kết luận tính dừng. Mọi mô hình ARMA chỉ hợp lệ về mặt toán học khi chuỗi đã dừng; ước lượng trên chuỗi không dừng sẽ gây ra hiện tượng hồi quy giả mạo (*spurious regression*).
- **How (Hoạt động như thế nào?)**:
  Ước lượng phương trình hồi quy $\Delta Y_t = \alpha + \gamma Y_{t-1} + \sum_{i=1}^k \delta_i \Delta Y_{t-i} + \varepsilon_t$ (chọn số trễ $k$ tối ưu theo `AIC`), tính thống kê $t_{\hat{\gamma}}$ (`ADF Statistic`) và đối chiếu với giá trị tới hạn MacKinnon (`1%`, `5%`, `10%`) cùng `p-value`:
  - Nếu `p-value < 0.05`: Bác bỏ $H_0$ $\rightarrow$ Kết luận chuỗi **ĐÃ DỪNG**.
  - Nếu `p-value >= 0.05`: Chưa đủ cơ sở bác bỏ $H_0$ $\rightarrow$ Kết luận chuỗi **KHÔNG DỪNG**.
- **Where (Được sử dụng ở bước nào?)**:
  Triển khai tại `src/lab02_utils/stats.py` (`perform_adf_test`), sử dụng ở Bài 2 và Bài 3 trên tập **Train**.
- **When (Khi nào nên sử dụng?)**:
  Trước khi chọn bậc sai phân $d, D$ và sau mỗi bước lấy sai phân để xác nhận chuỗi đã dừng chưa.
- **Example (Ví dụ thực tế trong LAB02)**:
  - **Dataset 1 (Train = 163 tháng)**: Chuỗi gốc có $\text{ADF} = 2.2309, p = 0.9989$ (Không dừng) $\rightarrow$ Sai phân $d=1$ có $\text{ADF} = -1.7061, p = 0.4280$ (vẫn bị nhiễu bởi mùa vụ mạnh) $\rightarrow$ Sai phân đôi $d=1, D=1, s=12$ có $\text{ADF} = -8.7879, p = 2.28 \times 10^{-14} < 0.05$ (Dừng hoàn toàn).
  - **Dataset 2 (Train = 115 tháng)**: Chuỗi gốc có $\text{ADF} = -0.3569, p = 0.9171$ (Không dừng) $\rightarrow$ Sai phân $d=1$ có $\text{ADF} = -2.5396, p = 0.1061$ $\rightarrow$ Sai phân đôi $d=1, D=1, s=12$ có $\text{ADF} = -12.1763, p = 1.38 \times 10^{-22} < 0.05$ (Dừng hoàn toàn).

---

## 6. Differencing — Phép Lấy Sai phân ($d, D$)

- **What (Nó là gì?)**:
  Phép biến đổi tính hiệu số giữa giá trị hiện tại và giá trị quá khứ để ổn định trung bình của chuỗi:
  - Sai phân thường bậc 1 ($d=1$): $\Delta Y_t = Y_t - Y_{t-1}$ (khử xu hướng tăng/giảm).
  - Sai phân mùa vụ bậc 1 ($D=1, s=12$): $\Delta_{12} Y_t = Y_t - Y_{t-12}$ (khử chu kỳ lặp lại 12 tháng).
  - Sai phân đôi kết hợp ($d=1, D=1, s=12$): $\Delta \Delta_{12} Y_t = (Y_t - Y_{t-1}) - (Y_{t-12} - Y_{t-13})$.
- **Why (Tại sao cần?)**:
  Biến chuỗi có xu hướng và mùa vụ (không dừng) thành chuỗi dao động ổn định quanh trục $0$ (dừng).
- **How (Hoạt động như thế nào?)**:
  Trong Pandas: `train.diff(1)` cho $d=1$ và `train.diff(1).diff(12)` cho $d=1, D=1, s=12$. Khi đưa vào `ARIMA(p, d, q)` hoặc `SARIMAX(p, d, q)(P, D, Q, s)`, thư viện tự động thực hiện sai phân bên trong và tích phân ngược khi dự báo.
- **Where (Được sử dụng ở bước nào?)**:
  Bài 3 (tiền xử lý kiểm định ADF), Bài 4 (đầu vào cho ACF/PACF) và cấu hình tham số `d, D` trong các module mô hình.
- **When (Khi nào nên sử dụng?)**:
  Khi chuỗi có xu hướng ($d=1$) và/hoặc chu kỳ mùa vụ bền vững ($D=1$). Không lấy sai phân quá mức ($d + D > 2$) vì sẽ làm tăng phương sai phần dư và gây tự tương quan âm giả tạo (*over-differencing*).
- **Example (Ví dụ thực tế trong LAB02)**:
  Ở cả 2 dataset, lấy sai phân đôi $(d=1, D=1, s=12)$ làm mất $1 + 12 = 13$ quan sát đầu tiên (Dataset 1 còn $163 - 13 = 150$ quan sát; Dataset 2 còn $115 - 13 = 102$ quan sát), vì vậy trong hàm `residual_diagnostics()` ta luôn truyền `n_skip=13` cho SARIMA để bỏ qua 13 phần dư khởi tạo đầu tiên.

---

## 7. ACF — Hàm Tự tương quan (Autocorrelation Function)

- **What (Nó là gì?)**:
  Hệ số tương quan tuyến tính tổng thể giữa $W_t$ và $W_{t-k}$ tại độ trễ $k$ (bao gồm cả tác động trực tiếp lẫn tác động bắc cầu qua các độ trễ trung gian $t-1, \dots, t-k+1$).
- **Why (Tại sao cần?)**:
  Giúp nhận diện **bậc trung bình trượt $\text{MA}(q)$**, bậc **$\text{SMA}(Q)$** mùa vụ, và phát hiện dấu hiệu chuỗi chưa dừng hoặc còn chu kỳ mùa vụ.
- **How (Hoạt động như thế nào?)**:
  Tính $\hat{\rho}_k = \frac{\sum_{t=k+1}^n (W_t - \bar{W})(W_{t-k} - \bar{W})}{\sum_{t=1}^n (W_t - \bar{W})^2}$ và so sánh với ngưỡng tin cậy $95\%$ ($\pm 1.96/\sqrt{n}$).
- **Where (Được sử dụng ở bước nào?)**:
  Module `src/lab02_utils/acf_pacf_analysis.py` (`analyze_acf_pacf`), Bài 4 (chọn tham số $q, Q$) và Bài 5–7 (kiểm tra ACF của phần dư).
- **When (Khi nào nên sử dụng?)**:
  Sau khi chuỗi tập Train đã được lấy sai phân về trạng thái dừng, và sau khi huấn luyện mô hình để kiểm tra phần dư.
- **Example (Ví dụ thực tế trong LAB02)**:
  Trên chuỗi Train $d=1$ của `Monthly Retail Sales` ($n=162$, ngưỡng $\pm 0.154$), $\text{ACF}_{12} = 0.84, \text{ACF}_{24} = 0.68, \text{ACF}_{36} = 0.55$ vọt rất cao, chứng minh tính mùa vụ 12 tháng cực mạnh. Sau khi lấy thêm $D=1$ ($n=150$, ngưỡng $\pm 0.160$), $\text{ACF}_{12} = -0.11$ nằm gọn trong dải tin cậy, còn ngắn hạn có $\text{ACF}_1 = -0.57, \text{ACF}_3 = 0.18$ vượt ngưỡng $\rightarrow$ gợi ý $q \in \{0, 1, 3\}$.

---

## 8. PACF — Hàm Tự tương quan Риêng phần (Partial Autocorrelation Function)

- **What (Nó là gì?)**:
  Hệ số tương quan **trực tiếp** giữa $W_t$ và $W_{t-k}$ sau khi đã loại bỏ hoàn toàn ảnh hưởng tuyến tính của các độ trễ trung gian $W_{t-1}, W_{t-2}, \dots, W_{t-k+1}$.
- **Why (Tại sao cần?)**:
  Là công cụ chuẩn xác để xác định **bậc tự hồi quy ngắn hạn $\text{AR}(p)$** và **tự hồi quy mùa vụ $\text{SAR}(P)$**.
- **How (Hoạt động như thế nào?)**:
  Giải hệ phương trình Yule-Walker (`method="ywm"`) ở từng bậc $k=1, \dots, K$; hệ số cuối cùng $\hat{\phi}_{kk}$ chính là giá trị $\text{PACF}(k)$.
- **Where (Được sử dụng ở bước nào?)**:
  Module `src/lab02_utils/acf_pacf_analysis.py` (`analyze_acf_pacf`) và `src/lab02_utils/stats.py` (`significant_lags`), Bài 4 của cả 2 Notebook.
- **When (Khi nào nên sử dụng?)**:
  Song hành cùng ACF trên chuỗi Train đã lấy sai phân.
- **Example (Ví dụ thực tế trong LAB02)**:
  Trên chuỗi Train $d=1$ của cả `Monthly Retail Sales` ($\text{PACF}_1 = -0.32, \text{PACF}_2 = -0.20$) và `AirPassengers` ($\text{PACF}_1 = 0.27, \text{PACF}_2 = -0.22$), PACF đều vượt ngưỡng ở lag 1 và lag 2 rồi cắt cụt ở lag 3, là cơ sở thống kê trực tiếp để đề xuất $p = 1$ hoặc $p = 2$.

---

## 9. Residual Diagnostics — Chẩn đoán Phần dư

- **What (Nó là gì?)**:
  Tập hợp các phép kiểm tra đồ thị và kiểm định thống kê trên phần dư huấn luyện $e_t = Y_t - \hat{Y}_t$ để xác nhận phần dư đã trở thành **Nhiễu trắng (White Noise)** hay chưa.
- **Why (Tại sao cần?)**:
  Nếu phần dư vẫn còn tự tương quan (chưa phải White Noise), nghĩa là mô hình đã bỏ sót quy luật có thể dự báo được. Nếu phần dư không phân phối chuẩn hoặc có phương sai thay đổi mạnh, khoảng tin cậy $95\%$ của dự báo sẽ bị sai lệch.
- **How (Hoạt động như thế nào?)**:
  Hàm `residual_diagnostics()` trong `src/lab02_utils/modeling.py` thực hiện 4 kiểm tra kết hợp:
  1. **Đồ thị phần dư theo thời gian**: Kiểm tra trung bình $\approx 0$ và xem có cụm biến động bất thường không.
  2. **Đồ thị ACF phần dư & Kiểm định Ljung-Box**:
     - $H_0$: Phần dư không có tự tương quan đến độ trễ $L$ (Phần dư là White Noise).
     - $H_1$: Phần dư còn tự tương quan.
     - Tiêu chuẩn: Phải đạt `Ljung-Box p-value >= 0.05` để không bác bỏ $H_0$.
  3. **Biểu đồ Histogram/KDE & Kiểm định Jarque-Bera**: Kiểm tra tính phân phối chuẩn ($H_0$: Phần dư phân phối chuẩn Gauss).
  4. **Kiểm định Heteroskedasticity (Breakvar / ARCH)**: Kiểm tra phương sai phần dư có đồng nhất theo thời gian không ($H_0$: Phương sai không đổi - Homoskedasticity).
- **Where (Được sử dụng ở bước nào?)**:
  Trong `src/lab02_utils/modeling.py` (`residual_diagnostics`) và tự động chạy trong mọi hàm `run_*_experiment()`.
- **When (Khi nào nên sử dụng?)**:
  Ngay sau khi huấn luyện mô hình trên tập Train và trước khi kết luận mô hình đủ tiêu chuẩn đưa vào dự báo.
- **Example (Ví dụ thực tế trong LAB02)**:
  Trên `AirPassengers`, `ARIMA(1,1,1)` có `Ljung-Box(lag=24) p-value = 0.0000 < 0.05` (Bác bỏ $H_0$ $\rightarrow$ Fail vì còn tự tương quan mùa vụ tại lag 12, 24). Ngược lại, `SARIMA(1,1,3)x(0,1,1)_12` có `Ljung-Box p-value = 0.4105 > 0.05` và `Log-SARIMA(1,1,0)x(0,1,1)_12` có `p-value > 0.90` (Đạt chuẩn White Noise hoàn hảo).

---

## 10. AIC & BIC — Tiêu chuẩn Thông tin Akaike và Bayesian

- **What (Nó là gì?)**:
  Hai chỉ số đánh giá chất lượng mô hình trên tập Train dựa trên sự đánh đổi giữa **mức độ khớp dữ liệu** (Log-Likelihood $\ell$) và **độ phức tạp của mô hình** (số lượng tham số $k$):
  $$\text{AIC} = -2\ell + 2k, \qquad \text{BIC} = -2\ell + k \ln(n_{\text{eff}})$$
- **Why (Tại sao cần?)**:
  Nếu chỉ tối đa hóa hàm hợp lý $\ell$, việc thêm thật nhiều tham số $p, q, P, Q$ sẽ luôn làm tăng $\ell$ nhưng gây ra hiện tượng **học vẹt (Overfitting)**. AIC và BIC cộng thêm đại lượng phạt cho số tham số $k$ (trong đó BIC phạt nặng hơn khi $n_{\text{eff}} > 8$).
- **How (Hoạt động như thế nào?)**:
  Trong cùng một nhóm mô hình hợp lệ, mô hình có **AIC và BIC nhỏ hơn** được ưu tiên.
  > **Nguyên tắc sống còn trong LAB02**: Chỉ được so sánh trực tiếp AIC/BIC giữa các mô hình có **cùng biến phụ thuộc (cùng thang đo gốc hoặc cùng thang đo Log)** và **cùng bậc sai phân $(d, D)$**. Tuyệt đối không so sánh AIC của `SARIMA` gốc (`AIC ≈ 639.62`) với `Log-SARIMA` (`AIC ≈ -327.47`) vì hàm mật độ log-likelihood của $\ln(Y_t)$ khác đơn vị đo với $Y_t$!
- **Where (Được sử dụng ở bước nào?)**:
  Trích xuất trong `fit_arima()`, `fit_sarima()` và phân nhóm tự động theo `Scale_Group` trong `ModelSummaryRegistry` (`src/lab02_utils/model_summary.py`).
- **When (Khi nào nên sử dụng?)**:
  Khi sàng lọc các mô hình ứng viên trong cùng một họ (ví dụ chọn Top 4 mô hình SARIMA trên chuỗi gốc).
- **Example (Ví dụ thực tế trong LAB02)**:
  Trong Bài 7 của `AirPassengers`, cấu hình `Log-SARIMA(1,1,0)x(0,1,0)_12` có AIC thấp nhất trên không gian Log nhưng lại rớt kiểm định Ljung-Box ($p < 0.05$) và dự báo kém ($R^2 = 0.73$). Nhờ kết hợp kiểm định Ljung-Box và sai số ngoài mẫu, ta tránh được "bẫy AIC" và chọn đúng `Log-SARIMA(1,1,0)x(0,1,1)_12` ($R^2 = 0.96$).

---

## 11. MAE, RMSE, MAPE, $R^2$ — Bộ Chỉ số Đánh giá Dự báo Ngoài mẫu

- **What (Nó là gì?)**:
  Bốn thước đo định lượng sai số giữa giá trị thực tế trên tập Test ($Y_t$) và giá trị dự báo ($\hat{Y}_t$) với $m = \text{len}(\text{Test})$:
  - $\text{MAE} = \frac{1}{m}\sum_{t=1}^m |Y_t - \hat{Y}_t|$: Sai số tuyệt đối trung bình (cùng đơn vị với dữ liệu gốc).
  - $\text{RMSE} = \sqrt{\frac{1}{m}\sum_{t=1}^m (Y_t - \hat{Y}_t)^2}$: Căn bậc hai sai số bình phương trung bình (cùng đơn vị gốc, phạt nặng các điểm sai lệch lớn / outliers).
  - $\text{MAPE (\%)} = \frac{100\%}{m}\sum_{t=1}^m \left|\frac{Y_t - \hat{Y}_t}{Y_t}\right|$: Sai số phần trăm tuyệt đối trung bình (không phụ thuộc đơn vị đo).
  - $R^2 = 1 - \frac{\sum (Y_t - \hat{Y}_t)^2}{\sum (Y_t - \bar{Y}_{\text{test}})^2}$: Hệ số xác định phản ánh tỷ lệ phương sai của tập Test được mô hình giải thích ($R^2 < 0$ nghĩa là dự báo tệ hơn cả việc đoán bằng trung bình tập Test).
- **Why (Tại sao cần?)**:
  Đo lường năng lực tổng quát hóa thực sự của mô hình trên dữ liệu tương lai chưa từng thấy (*Out-of-sample evaluation*), và là thước đo chung công bằng để so sánh giữa mô hình chuỗi gốc và mô hình biến đổi Logarit (sau khi đã lấy $\exp$ về thang đo gốc).
- **How (Hoạt động như thế nào?)**:
  Tính toán qua hàm `evaluate_forecast(y_true, y_pred)` trong `src/lab02_utils/modeling.py`.
- **Where (Được sử dụng ở bước nào?)**:
  Bước đánh giá cuối cùng của mọi mô hình trong `ar_model.py`, `ma_model.py`, `arima_model.py`, `sarima_model.py` và bảng xếp hạng `ModelSummaryRegistry`.
- **When (Khi nào nên sử dụng?)**:
  Sau khi mô hình đã dự báo đủ $m = \text{len}(\text{Test})$ bước và (nếu dùng Log) đã biến đổi ngược $\hat{Y}_t = \exp(\hat{Y}^{(\log)}_t)$.
- **Example (Ví dụ thực tế trong LAB02)**:
  Trên `Monthly Retail Sales` (Test = 41 tháng), `SARIMA(0,1,1)x(0,1,1)_12` giảm `MAE` từ `3.67` xuống `1.45`, giảm `RMSE` từ `4.62` xuống `1.84`, giảm `MAPE` từ `18.04%` xuống `8.21%` và nâng $R^2$ từ `-0.25` lên `+0.8012`.

---

## 12. Logging System — Hệ thống Ghi nhật ký Tập trung

- **What (Nó là gì?)**:
  Hệ thống ghi lại toàn bộ lịch sử thực thi có cấu trúc (thời gian, mức độ `INFO/WARNING/ERROR`, tên module, `run_id`, thông số dữ liệu, kết quả ADF/ACF/PACF, thời gian huấn luyện, trạng thái hội tụ và chỉ số đánh giá) vào file `.log` thay vì chỉ dùng `print()`.
- **Why (Tại sao cần?)**:
  Trong nghiên cứu khoa học dữ liệu, kết quả in trên màn hình rất dễ mất đi khi đóng notebook. File log lưu trữ bền vững giúp kiểm toán lại toàn bộ quá trình thí nghiệm, phát hiện ngay các cảnh báo hội tụ (`ConvergenceWarning`) hoặc lỗi dữ liệu.
- **How (Hoạt động như thế nào?)**:
  Module `src/lab02_utils/logging_config.py` sử dụng `RotatingFileHandler` (giới hạn 5 MB/file, chống tạo trùng lặp handler khi chạy lại cell) để ghi vào `logs/lab02/dataset1/lab02_dataset1.log` và `logs/lab02/dataset2/lab02_dataset2.log`.
- **Where (Được sử dụng ở bước nào?)**:
  Được gọi tự động bên trong `data_loader.py`, `stats.py`, `acf_pacf_analysis.py`, `modeling.py`, `ar_model.py`, `ma_model.py`, `arima_model.py`, `sarima_model.py`, và `model_summary.py`.
- **When (Khi nào nên sử dụng?)**:
  Xuyên suốt từ lúc nạp dữ liệu (`[DATA_LOADING]`), chia tập (`[PREPROCESSING]`), kiểm định thống kê (`[ADF_TEST]`, `[ACF_PACF_ANALYSIS]`), huấn luyện (`[MODEL_TRAINING]`), đánh giá (`[MODEL_EVALUATION]`) đến khi xuất báo cáo (`[SYSTEM_STATUS]`).
- **Example (Ví dụ thực tế trong LAB02)**:
  Mỗi lần chạy `fit_arima()` hoặc `fit_sarima()`, hệ thống tự động ghi lại thời gian huấn luyện (tính bằng giây đến 4 chữ số thập phân), `AIC`, `BIC`, `LogLikelihood` và trạng thái `converged=True/False`.

---

## 13. Model Summary System — Hệ thống Tổng hợp & Diễn giải Mô hình

- **What (Nó là gì?)**:
  Bộ thu thập trung tâm (`ModelSummaryRegistry`) chuẩn hóa kết quả của mọi mô hình đã chạy trên một dataset, tạo thẻ tóm tắt kiểu `SARIMAX Results` (gồm hệ số, sai số chuẩn, thống kê $z$, $p\text{-value}$, khoảng tin cậy 95%, Ljung-Box, Jarque-Bera, Heteroskedasticity), lập bảng so sánh và tự động sinh văn bản nhận xét học thuật.
- **Why (Tại sao cần?)**:
  Loại bỏ hoàn toàn việc copy-paste thủ công các con số giữa các cell dễ gây nhầm lẫn; đảm bảo 100% số liệu trong bảng báo cáo `.csv`, `.json`, `.md` đều được trích xuất trực tiếp từ đối tượng mô hình thực tế.
- **How (Hoạt động như thế nào?)**:
  Mỗi hàm `run_*_experiment()` trả về một dictionary chuẩn hóa được nạp vào `registry.register_experiment(exp)`. Cuối cùng, `registry.export_all()` tự động xuất ra 3 định dạng tại `reports/model_summary/dataset1/` và `reports/model_summary/dataset2/`.
- **Where (Được sử dụng ở bước nào?)**:
  Triển khai tại `src/lab02_utils/model_summary.py` và được gọi ở phần tổng kết của cả hai Notebook.
- **When (Khi nào nên sử dụng?)**:
  Sau khi đã hoàn tất huấn luyện và đánh giá các họ mô hình `AR`, `MA`, `ARIMA`, `SARIMA`, `Log-SARIMA`.
- **Example (Ví dụ thực tế trong LAB02)**:
  File `reports/model_summary/dataset1/dataset1_model_summary.md` và `reports/model_summary/dataset2/dataset2_model_summary.md` chứa bảng xếp hạng toàn bộ các mô hình đã chạy kèm thẻ chi tiết hệ số của từng mô hình và phần diễn giải tự động phân biệt rõ *Statistical Fit*, *Residual Adequacy* và *Out-of-Sample Forecast Performance*.
