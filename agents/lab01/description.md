# Mô tả Lab 01: Tiền xử lý và Khám phá dữ liệu Chuỗi thời gian

Lab 01 bao gồm các bước nền tảng để làm quen, phân tích đặc trưng và đánh giá chất lượng của một chuỗi thời gian trước khi tiến hành xây dựng mô hình dự báo.

**Dataset sử dụng:** 
Bài thực hành này sử dụng bộ dữ liệu gợi ý là `AirPassengers` (dữ liệu hành khách hàng không theo tháng).

**Mục tiêu chính:**
* Làm quen với đồ thị ACF/PACF và hiểu rõ về khái niệm nhiễu trắng (white noise).
* Biết cách phân rã chuỗi thời gian và trực quan hóa dữ liệu hiệu quả.
* Kiểm tra và xử lý các vấn đề về chất lượng dữ liệu thực tế.
* Viết báo cáo ngắn để đánh giá tổng quan dữ liệu.

**Nội dung thực hành cụ thể (6 bài tập):**
* **Bài 1 - Tính và phân tích ACF/PACF:** Tính toán ACF/PACF đến 40 lags để kết luận về tính tự tương quan, xu hướng, mùa vụ và tìm ra lag quan trọng nhất.
* **Bài 2 - Kiểm tra White Noise:** Sinh 1000 điểm dữ liệu nhiễu trắng bằng NumPy, vẽ ACF để xem các lag có nằm trong ngưỡng tin cậy 95% không, từ đó so sánh và chỉ ra sự khác biệt với chuỗi `AirPassengers`.
* **Bài 3 - Phân rã chuỗi thời gian:** Dùng phương pháp STL hoặc Classical Decomposition để tách chuỗi thành 4 thành phần (Observed, Trend, Seasonal, Residual) và đánh giá độ mạnh/yếu của xu hướng, mùa vụ.
* **Bài 4 - Xử lý chất lượng dữ liệu:** Cố tình tạo dữ liệu thiếu (NaN) để thực hành xử lý bằng phương pháp Forward-fill và Interpolation; phát hiện ngoại lệ (outliers) bằng Rolling, Z-score hoặc IQR.
* **Bài 5 - Trực quan hóa dữ liệu:** Vẽ line chart, biểu đồ rolling mean/std (window=12), seasonal plot theo tháng và heatmap (month x year) để nhận diện bức tranh tổng thể.
* **Bài 6 - Viết báo cáo:** Tổng hợp lại các bước phân tích trên để đưa ra kết luận cuối cùng: Dữ liệu này có đủ tốt và phù hợp để đưa vào dự báo hay không.