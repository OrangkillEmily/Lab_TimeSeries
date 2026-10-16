# Bài tập thực hành 2: Kiểm định ADF & Chuẩn bị mô hình ARIMA/SARIMA

## MỤC TIÊU:
1. Nhận diện chuỗi dừng / không dừng.
2. Thực hiện và diễn giải kiểm định ADF.
3. Xác định tham số trong ARIMA.
4. Kết hợp ADF với ACF/PACF.
5. Chuẩn bị dữ liệu đúng cách cho ARIMA / SARIMA.
6. Đánh giá mô hình.

---

## BÀI TẬP THỰC HÀNH:
Sử dụng dataset1 (trong phần dataset bên dưới).

### Bài 1: Khám phá dữ liệu (EDA)
* Load dữ liệu.
* Chuyển date thành index thời gian.
* Vẽ đồ thị chuỗi thời gian.

**Câu hỏi cần trả lời:**
* Chuỗi có xu hướng không?
* Có dấu hiệu mùa vụ không?
* Chuỗi có khả năng dừng không?

### Bài 2: Kiểm định ADF trên chuỗi gốc
* Thực hiện kiểm định ADF cho chuỗi gốc.
* Giá trị ADF statistic.
* p-value.
* Kết luận: Dừng / Không dừng.

**Câu hỏi cần trả lời:**
* Ho của ADF là gì?
* p-value cho biết điều gì?
* Chuỗi có dừng không?

### Bài 3: Sai phân & kiểm định lại
* Sai phân bậc 1.
* Vẽ lại đồ thị chuỗi sau sai phân.
* Kiểm định ADF lần 2.

**Câu hỏi cần trả lời:**
* Sau sai phân, chuỗi đã dừng chưa?
* Giá trị d trong ARIMA là bao nhiêu?

### Bài 4: Phân tích ACF & PACF
**Nhiệm vụ:**
* Vẽ ACF và PACF của chuỗi đã dừng.
* Đề xuất (p, d, q).

**Câu hỏi cần trả lời:**
* ACF cắt cụt ở lag nào?
* PACF cắt cụt ở lag nào?
* Gợi ý mô hình ARIMA(p,d,q)?

### Bài 5: Thiết lập ARIMA / SARIMA – Dự đoán – Đánh giá
1. Chia dữ liêu thành: train(80%), test(20%). Giữ nguyên thứ tự thời gian (không shuffle).
2. Thiết lập mô hình ARIMA: Dựa trên ACF/PACF, đề xuất 1–2 mô hình ARIMA. Ví dụ: ARIMA(1,1,1), ARIMA(2,1,1).
3. Fit mô hình trên tập train. Ghi lại AIC và BIC.
   * Chẩn đoán phần dư (Residual Diagnostics): Vẽ Residuals theo thời gian, ACF của residuals. 
   * Kiểm tra: Residuals có mean ≈ 0? Có tự tương quan không?
4. Dự đoán (Forecasting):
   * Dự đoán trên tập test.
   * Vẽ đồ thị: Train, Test (thực tế), Forecast.
5. Đánh giá ARIMA. Chỉ số đánh giá bắt buộc: MAE, RMSE, MAPE, R^2.

**Câu hỏi cần trả lời:**
1. Vì sao không được shuffle dữ liệu chuỗi thời gian? Test set dùng để làm gì?
2. Ý nghĩa của p, d, q đã chọn?
3. Mô hình có hội tụ không?
4. Residuals có phải white noise không?
5. Nếu residuals còn tự tương quan điều chỉnh gì?
6. Dự đoán có bám sát xu hướng không?
7. Sai lệch lớn nhất xảy ra ở đâu?
8. Chỉ số nào nhạy với outlier nhất?
9. RMSE lớn hay nhỏ nói lên điều gì?

---

## BÀI TẬP NÂNG CAO với mô hình SARIMA, sử dụng dataset2:
1. Vẽ chuỗi thời gian -> nhận diện mùa vụ.
2. ADF chuỗi gốc -> không dừng.
3. Sai phân thường + mùa vụ.
4. ACF/PACF -> xác định (p,q,P,Q).
5. Fit SARIMA.
6. Forecast & đánh giá (MAE, RMSE).
7. So sánh với ARIMA.

**Gợi ý: Thiết lập tham số**
* s = 12.
* Sai phân: d=1 (loại trend), D=1 (loại mùa vụ).

---

## DATASET:

**Dataset 1. Monthly Retail Sales (bán lẻ – ARIMA/SARIMA)**
* Doanh số bán lẻ theo tháng (US).
* Đặc điểm: Trend + seasonality.
* link: https://raw.githubusercontent.com/selva86/datasets/master/a10.csv

**Dataset 2. AirPassengers - Monthly Airline Passengers**
* Số hành khách hàng không theo tháng từ 1949 – 1960 (144 quan sát).
* Tần suất: Monthly.
* Chu kỳ mùa vụ: s = 12[cite: 20].
* link: https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv