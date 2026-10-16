# Mô tả Lab 02: Kiểm định ADF & Chuẩn bị mô hình ARIMA/SARIMA

Lab 02 đi sâu vào quy trình phân tích tính dừng của dữ liệu và các bước kỹ thuật để xây dựng, chẩn đoán, đánh giá mô hình dự báo chuỗi thời gian. 

**Mục tiêu chính:**
* Nhận diện được chuỗi dừng và chuỗi không dừng.
* Thực hiện và diễn giải chính xác ý nghĩa của kiểm định ADF.
* Đọc đồ thị ACF/PACF kết hợp với ADF để xác định bộ tham số cho mô hình ARIMA.
* Chuẩn bị dữ liệu đúng chuẩn và thực hiện đánh giá mô hình.

**Nội dung thực hành cụ thể:**
* **Bài 1 - Khám phá dữ liệu (EDA):** Chuyển đổi cột ngày tháng thành index, vẽ đồ thị để nhận định ban đầu về xu hướng, tính mùa vụ và khả năng dừng của chuỗi.
* **Bài 2 - Kiểm định ADF trên chuỗi gốc:** Chạy kiểm định ADF, phân tích giá trị ADF statistic và p-value để đưa ra kết luận chuỗi có dừng hay không.
* **Bài 3 - Sai phân & Kiểm định lại:** Áp dụng sai phân bậc 1, vẽ lại đồ thị và chạy ADF lần 2 để xác nhận chuỗi đã dừng chưa, từ đó chốt tham số `d`.
* **Bài 4 - Phân tích ACF & PACF:** Vẽ đồ thị ACF/PACF trên chuỗi đã lấy sai phân, tìm vị trí lag bị cắt cụt để đề xuất bộ tham số `(p, d, q)` cho ARIMA.
* **Bài 5 - Thiết lập ARIMA, Dự đoán & Đánh giá:** 
  * Chia dữ liệu Train (80%) và Test (20%) tuyệt đối không được shuffle (giữ nguyên thứ tự thời gian).
  * Đề xuất 1-2 mô hình ARIMA, huấn luyện (fit) trên tập train và ghi lại chỉ số AIC, BIC.
  * Chẩn đoán phần dư (Residuals) để kiểm tra xem có phải là white noise (nhiễu trắng) hay không.
  * Chạy dự đoán (forecast), vẽ đồ thị so sánh thực tế và đánh giá bắt buộc bằng các chỉ số: MAE, RMSE, MAPE.
* **Bài tập nâng cao (Mô hình SARIMA):** Chuyển sang sử dụng Dataset 2 (AirPassengers). Thực hiện sai phân thường (`d=1`) kết hợp sai phân mùa vụ (`D=1`) với chu kỳ `s=12`. Xác định tham số `(p,q,P,Q)` từ ACF/PACF, huấn luyện SARIMA, sau đó đánh giá và so sánh trực tiếp với kết quả của mô hình ARIMA.