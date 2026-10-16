# HƯỚNG DẪN TỪNG BƯỚC & BÀN GIAO DỰ ÁN (STEP-BY-STEP & HANDOFF) - LAB 01

## 1. Rules From Project (Quy tắc làm việc & AI)
- **AI Role:** AI đóng vai trò trợ lý chuyên môn phân tích dữ liệu, con người đưa ra quyết định cuối cùng trong việc nhận định đồ thị (đọc ACF/PACF, đánh giá Trend/Seasonality).
- **Triết lý Code (Vibe Coding):** Code đơn giản, tường minh, tận dụng tối đa sức mạnh của `pandas` và `statsmodels`. Không "over-engineer" các biểu đồ, ưu tiên tính rõ ràng, dễ đọc.
- **Yêu cầu cốt lõi:** Bắt buộc hoàn thành 6 bài tập. Phải sinh chính xác 1000 điểm dữ liệu White Noise bằng NumPy và tính ACF/PACF tối đa đến 40 lags.
- **Nguyên tắc Dữ liệu:** Dữ liệu gốc là "bất khả xâm phạm" (Read-only). Bất kỳ thao tác tạo thêm NaN hay nội suy (Interpolation) đều phải lưu ra tệp mới tại thư mục `processed/`.

## 2. Chosen Direction (Định hướng & Kiến trúc)
- **Nguồn dữ liệu:** Đọc từ `data/raw/airline-passengers.csv` (144 quan sát, dữ liệu hàng tháng).
- **Quy trình Phân tích (EDA Workflow):** Load data & ACF/PACF -> Phân tích White Noise -> Phân rã chuỗi (Decomposition) -> Kiểm tra chất lượng (Missing/Outliers) -> Trực quan hóa chuyên sâu -> Báo cáo.
- **Tiền xử lý & Khắc phục nhiễu:** Dùng Forward-fill / Interpolation cho dữ liệu khuyết thiếu. Phát hiện ngoại lệ bằng Rolling mean/std, Z-score hoặc IQR.
- **Thư viện cốt lõi:** `pandas`, `numpy`, `matplotlib`, `seaborn` (vẽ heatmap), và `statsmodels` (chạy Decomposition, tính ACF/PACF).

## 3. Current Files And Roles (Cấu trúc file hiện tại của Lab 01)
- `agents/lab01/`: chứa yêu cầu bài lab01
- `docs/`: Nơi chứa file báo cáo kết luận Bài 6 (Đánh giá tính phù hợp của dữ liệu).
- `notebooks/lab01/`: Chứa các file `.ipynb` thực thi được chia nhỏ rành mạch theo từng bài:
  - `bai1_acf_pacf.ipynb`: Tải dữ liệu, tính và vẽ đồ thị ACF/PACF đến 40 lags.
  - `bai2_white_noise.ipynb`: Sinh dữ liệu giả lập nhiễu trắng và so sánh.
  - `bai3_decomposition.ipynb`: Phân rã chuỗi thời gian thành các thành phần.
  - `bai4_data_quality.ipynb`: Giả lập dữ liệu thiếu, xử lý missing và outliers.
  - `bai5_visualization.ipynb`: Trực quan hóa chuyên sâu (Rolling, Seasonal plot, Heatmap).
  - `Lab01_TimeSeries.ipynb`: File notebook tổng hợp luồng chạy hoàn chỉnh để nộp bài.
- `reports/`: Nơi lưu các biểu đồ xuất ra từ code (ACF/PACF, STL Plot, Heatmap) dạng ảnh `.png`.
- `src/`: Thư mục chứa các code logic dùng chung:
  - `data_loader.py`: Hàm gọi dữ liệu CSV và ép kiểu cột Month về Datetime Index.
  - `lab01_utils/decomposition.py`: Hàm xử lý phân rã chuỗi bằng STL hoặc Classical Decomposition.
  - `lab01_utils/plotting.py`: Hàm đóng gói chức năng vẽ đồ thị phức tạp (Rolling mean w=12, Seasonal plot, Heatmap).

## 4. Workflow Timeline (Tiến độ thực thi chi tiết 5 Giai đoạn)
**Giai đoạn 1: Khởi tạo không gian làm việc** (✅ Đã xong)
- Đã dựng xong khung thư mục và file rỗng tĩnh (`src`, `data`, `reports`, `docs`, `notebooks`).
- Đã cấu hình `.gitignore` chặn các file CSV trong thư mục `processed/` và ảnh trong `figures/` để kho lưu trữ Git nhẹ gọn.
- Đã chốt yêu cầu tại thư mục `agents/lab01/`.

**Giai đoạn 2: Nhận diện Đặc trưng & Phân tích Nhiễu trắng (Bài 1 & Bài 2)**
*Mục tiêu: Đánh giá nhanh sự tự tương quan của dữ liệu và hiểu bản chất của White Noise.*
- **Nơi thực thi:** Lần lượt tại `notebooks/lab01/bai1_acf_pacf.ipynb` và `bai2_white_noise.ipynb` (gọi hàm load từ `src/data_loader.py`).
- **Nhiệm vụ 1:** Tính và vẽ ACF, PACF đến 40 lags. Trả lời: Có xu hướng không? Mùa vụ không? Lag nào quan trọng nhất?
- **Nhiệm vụ 2:** Dùng `numpy.random.normal` sinh 1000 điểm dữ liệu giả lập. Vẽ đồ thị ACF của tập này để kiểm chứng xem các thanh lag có lọt thỏm vào vùng màu xanh (ngưỡng 95%) hay không. Viết 5 dòng nhận xét sự khác biệt so với AirPassengers.

**Giai đoạn 3: Phân rã chuỗi thời gian - Decomposition (Bài 3)**
*Mục tiêu: "Mổ xẻ" chuỗi thời gian thành các thành phần cấu tạo nên nó.*
- **Nơi thực thi:** File `notebooks/lab01/bai3_decomposition.ipynb` (kết hợp module `src/lab01_utils/decomposition.py`).
- **Nhiệm vụ:** Chọn tham số chu kỳ theo tháng (`period=12`). Vẽ biểu đồ 4 ô (Observed, Trend, Seasonal, Residual). 
- **Kết luận:** Mô tả xu hướng (tăng/giảm), sức mạnh mùa vụ và quan sát thành phần Residual xem có giống nhiễu trắng ở Bài 2 không.

**Giai đoạn 4: Đánh giá Chất lượng & Trực quan hóa chuyên sâu (Bài 4 & Bài 5)**
*Mục tiêu: Đưa dữ liệu về trạng thái hoàn hảo nhất và khai thác insight (góc nhìn) qua biểu đồ.*
- **Nơi thực thi:** Lần lượt tại `notebooks/lab01/bai4_data_quality.ipynb` và `bai5_visualization.ipynb` (kết hợp `src/lab01_utils/plotting.py`).
- **Nhiệm vụ 1 (Chất lượng):** Cố tình đục lỗ (tạo NaN) vào dữ liệu. Thử nghiệm lấp đầy bằng Forward-fill và Interpolation. Quét Outlier bằng Z-score/IQR và lưu dữ liệu đã vá lỗi hoàn chỉnh vào thư mục `processed`.
- **Nhiệm vụ 2 (Trực quan):** Vẽ line chart toàn chuỗi, Rolling mean & std (`window=12`), Seasonal plot theo tháng, và đặc biệt là Heatmap (month x year). Đưa ra nhận xét về những tháng bất thường.

**Giai đoạn 5: Viết báo cáo & Tổng kết (Bài 6 - Nghiệm thu)**
*Mục tiêu: Đóng gói toàn bộ quá trình Khám phá dữ liệu thành kết luận nghiên cứu.*
- **Nhiệm vụ 1 (Chốt hạ):** Mở file báo cáo trong `docs/`, trình bày lại 6 phần đã phân tích. Trả lời trực diện câu hỏi: "Dữ liệu có phù hợp để dự báo (chạy mô hình ARIMA/SARIMA) không? Tại sao?"
- **Nhiệm vụ 2 (Báo cáo Notebook):** Mở `notebooks/lab01/Lab01_TimeSeries.ipynb`, gộp các đoạn code chuẩn xác nhất từ các bước trên thành một luồng chạy liền mạch (kèm text giải thích Markdown) để nộp bài cho giảng viên.

