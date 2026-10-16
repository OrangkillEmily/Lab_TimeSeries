# Bài tập thực hành chương 2

## Mục tiêu chung của bài tập
* Làm quen với ACF/PACF
* Hiểu white noise.
* Phân rã chuỗi thời gian.
* Kiểm tra chất lượng dữ liệu.
* Trực quan hóa dữ liệu.
* Viết báo cáo ngắn đánh giá dữ liệu.

**Dataset gợi ý:** AirPassengers, Daily-Min-Temperatures.csv, hoặc bất kì chuỗi theo ngày.

---

## Bài tập 1: Tính và phân tích ACF/PACF
1. Tải và xem trước dữ liệu (head, info, plot).
2. Tính ACF, PACF đến 40 lags.
3. Kết luận:
   * Chuỗi có tự tương quan mạnh không?
   * Có xu hướng/mùa vụ thể hiện qua ACF/PACF không?
   * Lag nào quan trọng nhất?
## Bài tập 2: Kiểm tra White Noise
1. Sinh dữ liệu white noise bằng NumPy (1000 điểm).
2. Vẽ ACF → kiểm tra xem các lag có nằm trong ngưỡng tin cậy 95% không.
3. So sánh ACF của white noise với ACF của chuỗi AirPassengers.
4. Viết 5 dòng mô tả sự khác nhau.

## Bài tập 3: Phân rã chuỗi thời gian (Decomposition)
1. Chọn chuỗi dữ liệu theo tháng hoặc theo quý.
2. Dùng phương pháp STL hoặc classical decomposition.
3. Vẽ 4 thành phần: Observed, Trend, Seasonal, Residual.
4. Mô tả:
   * Xu hướng tăng/giảm?
   * Mùa vụ mạnh hay yếu?
   * Residual có giống white noise không?

## Bài tập 4: Kiểm tra và xử lý chất lượng dữ liệu
1. Tạo dữ liệu thiếu (NaN) giả lập hoặc dùng dataset có missing
2. Kiểm tra missing: isna().sum().
3. Xử lý missing bằng 2 cách: Forward-fill và Interpolation.
4. Phát hiện outliers bằng: Rolling mean/variance; Z-score hoặc IQR.
5. Vẽ trước và sau khi xử lý.

## Bài tập 5: Trực quan hóa dữ liệu chuỗi thời gian
1. Vẽ line chart toàn bộ chuỗi.
2. Vẽ rolling mean (window=12) và rolling std.
3. Vẽ seasonal plot theo tháng (đối với dữ liệu theo tháng).
4. Vẽ heatmap seasonality (month × year).
5. Kết luận: Xu hướng chung, các mùa vụ nổi bật, bất thường (nếu có).

## Bài tập 6: Viết báo cáo gồm:
1. Mô tả dataset.
2. Kiểm tra ACF/PACF.
3. Kiểm tra white noise.
4. Phân rã chuỗi.
5. Vấn đề chất lượng dữ liệu.
6. Ý nghĩa trực quan hóa.
7. Kết luận: dữ liệu có phù hợp để dự báo không?