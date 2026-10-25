# HƯỚNG DẪN TỪNG BƯỚC & BÀN GIAO DỰ ÁN (STEP-BY-STEP & HANDOFF) - LAB 02 - DATASET 1

## 1. Quy tắc từ Đề bài (Rules From Project)
- **Phạm vi yêu cầu:** Hoàn thành từ Bài 1 đến Bài 5 (Yêu cầu cơ bản). Sử dụng mô hình ARIMA.
- **Quy tắc chia dữ liệu:** Tỷ lệ 80% Train - 20% Test, tuyệt đối KHÔNG ĐƯỢC XÁO TRỘN (`shuffle=False`) để giữ nguyên cấu trúc tuần tự của chuỗi thời gian doanh số.
- **Quy tắc xây dựng mô hình:** Tận dụng thư viện `statsmodels` (class `ARIMA`), không code tay thuật toán từ đầu. Đánh giá bằng 3 chỉ số bắt buộc: MAE, RMSE, MAPE.

## 2. Định hướng & Kiến trúc (Chosen Direction)
- **Nguồn dữ liệu:** Dùng file `dataset1 Monthly Retail Sales.csv` (Cột thời gian là `date`, cột dữ liệu là `value`).
- **Đặc trưng dữ liệu:** Doanh số bán lẻ hàng tháng của Mỹ (US Monthly Retail Sales).
- **Quy trình Phân tích (ARIMA Workflow):** Khám phá (EDA) -> Kiểm định ADF gốc -> Lấy sai phân (d=1) -> Kiểm định ADF lần 2 -> Vẽ ACF/PACF để tìm (p,q) -> Chia tập Train/Test -> Huấn luyện ARIMA -> Chẩn đoán phần dư -> Dự báo & Đánh giá sai số.

## 3. Cấu trúc File triển khai (Current Files And Roles)
- `docs/methodology_dataset1.md`: Nơi lưu trữ tài liệu giải thích phương pháp chọn tham số (p, d, q) và trả lời các câu hỏi lý thuyết từ Bài 1 đến Bài 5 cho Dataset 1.
- `notebooks/lab02/lab02_dataset1.ipynb`: File Jupyter Notebook thực thi toàn bộ code từ Bài 1 đến Bài 5.
- `reports/figures/lab02_dataset1/`: Nơi xuất và lưu các hình ảnh biểu đồ (EDA, Sai phân, ACF/PACF, Forecast).
- `src/`: Thư mục chứa các code logic tái sử dụng (`lab02_utils/stats.py` và `lab02_utils/modeling.py`).

## 4. Chi tiết Tiến độ Thực thi (Workflow Timeline - 5 Bài)

**Bài 1: Khám phá dữ liệu (EDA)**
*Mục tiêu: Đọc chuẩn xác file Retail Sales và nhận diện hình thái dữ liệu.*
- **Nhiệm vụ 1:** Load file `dataset1 Monthly Retail Sales.csv`, set cột `date` làm index thời gian.
- **Nhiệm vụ 2:** Vẽ đồ thị chuỗi thời gian gốc.
- **Nhiệm vụ 3:** Quan sát đồ thị và trả lời 3 câu hỏi của đề: Chuỗi có xu hướng không? Có dấu hiệu mùa vụ không? Có khả năng dừng không?

**Bài 2: Kiểm định ADF trên chuỗi gốc**
*Mục tiêu: Dùng toán học thống kê để chứng minh chuỗi chưa dừng.*
- **Nhiệm vụ 1:** Chạy kiểm định ADF trên chuỗi `value` gốc. In ra ADF statistic và p-value.
- **Nhiệm vụ 2:** Trả lời các câu hỏi: $H_0$ của ADF là gì? p-value cho biết điều gì? Kết luận chuỗi có dừng không.

**Bài 3: Sai phân & Kiểm định lại**
*Mục tiêu: Đưa chuỗi doanh số về trạng thái dừng (Stationary).*
- **Nhiệm vụ 1:** Lấy sai phân bậc 1 ($d=1$) cho chuỗi gốc.
- **Nhiệm vụ 2:** Vẽ lại đồ thị chuỗi sau khi lấy sai phân để so sánh.
- **Nhiệm vụ 3:** Chạy kiểm định ADF lần 2 trên chuỗi đã sai phân. Trả lời câu hỏi: Chuỗi đã dừng chưa? Giá trị $d$ trong ARIMA là bao nhiêu?

**Bài 4: Phân tích ACF & PACF**
*Mục tiêu: Xác định bậc tự hồi quy (p) và trung bình trượt (q).*
- **Nhiệm vụ 1:** Vẽ đồ thị ACF và PACF trên chuỗi **đã lấy sai phân** (chắc chắn không vẽ trên chuỗi gốc).
- **Nhiệm vụ 2:** Quan sát điểm cắt cụt (vượt dải tin cậy) của ACF để đề xuất bậc $q$, và điểm cắt cụt của PACF để đề xuất bậc $p$.
- **Nhiệm vụ 3:** Ghi nhận và chốt lại 1-2 ứng viên mô hình ARIMA(p,d,q) tiềm năng.

**Bài 5: Thiết lập ARIMA - Dự đoán - Đánh giá**
*Mục tiêu: Tìm ra mô hình dự báo tốt nhất và chấm điểm sai số.*
- **Nhiệm vụ 1 (Chia tập):** Chia dữ liệu `train` (80%) và `test` (20%) không shuffle.
- **Nhiệm vụ 2 (Huấn luyện):** Fit các cấu hình ARIMA(p,d,q) đã đề xuất trên tập Train. Ghi nhận chỉ số AIC, BIC.
- **Nhiệm vụ 3 (Chẩn đoán):** Vẽ phần dư (Residuals) theo thời gian và ACF của phần dư. Trả lời: Residuals có mean xấp xỉ 0 không? Có tự tương quan không (White noise)?
- **Nhiệm vụ 4 (Dự báo & Đánh giá):** Chạy Forecast trên tập Test. Vẽ đồ thị đè 3 đường: Train, Test thực tế và Forecast. Tính các chỉ số MAE, RMSE, MAPE.
- **Nhiệm vụ 5 (Tổng kết lý thuyết):** Ghi chép câu trả lời cho 9 câu hỏi lý thuyết của Bài 5 (ý nghĩa p,d,q; mô hình hội tụ; nhận xét xu hướng dự báo; sai lệch lớn nhất; tính nhạy cảm của các chỉ số) vào file docs.