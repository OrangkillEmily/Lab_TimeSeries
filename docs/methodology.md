### Trả lời câu hỏi lý thuyết: Kiểm định ADF (Augmented Dickey-Fuller)

1. **Giả thuyết Ho (Null Hypothesis) của kiểm định ADF là gì?**
   - **Ho:** Chuỗi thời gian có nghiệm đơn vị (unit root), nghĩa là **chuỗi KHÔNG DỪNG** (phụ thuộc vào thời gian, có xu hướng hoặc mùa vụ).
   - **H1:** Chuỗi không có nghiệm đơn vị, nghĩa là **chuỗi DỪNG**.

2. **Dựa vào chỉ số nào để kết luận?**
   - Ta dựa vào giá trị **p-value** (thường so sánh với mức ý nghĩa alpha = 0.05) hoặc so sánh trực tiếp **ADF Statistic** với **Critical Values**.
   - Nếu `p-value < 0.05` (hoặc `ADF Statistic` nhỏ hơn các ngưỡng `Critical Value`): Bác bỏ Ho $\rightarrow$ Chuỗi DỪNG.
   - Nếu `p-value >= 0.05`: Không thể bác bỏ Ho $\rightarrow$ Chuỗi KHÔNG DỪNG.

3. **Kết luận cho chuỗi `AirPassengers` hiện tại:**
   - Kết quả chạy code cho thấy `p-value` lớn hơn rất nhiều so với mức 0.05.
   - Do đó, ta **không thể bác bỏ Ho**. Khẳng định lại một lần nữa bằng toán học: **Chuỗi dữ liệu gốc KHÔNG DỪNG**. Chúng ta bắt buộc phải sử dụng phương pháp Lấy sai phân (Differencing) ở Bài 3.

   ### Trả lời câu hỏi lý thuyết: Ý nghĩa của tham số d, D và Kết quả kiểm định

1. **Lấy sai phân (Differencing) là gì và tại sao phải làm?**
   - Sai phân là phép tính trừ giá trị quan sát hiện tại cho giá trị quan sát trong quá khứ. 
   - Mục đích là để triệt tiêu các yếu tố xu hướng (trend) và mùa vụ (seasonality), giúp ổn định trung bình của chuỗi, từ đó đưa chuỗi về dạng **Dừng (Stationary)** để đáp ứng điều kiện tiên quyết của mô hình ARIMA/SARIMA.

2. **Tham số d, D, s có ý nghĩa gì?**
   - **$d=1$**: Sai phân bậc 1 (lấy tháng này trừ tháng trước: $Y_t - Y_{t-1}$). Dùng để triệt tiêu xu hướng tăng/giảm.
   - **$s=12$**: Chu kỳ mùa vụ của dữ liệu là 12 tháng (do đây là dữ liệu hàng tháng của từng năm).
   - **$D=1$**: Sai phân mùa vụ bậc 1 (lấy giá trị tháng này trừ đi đúng tháng này của năm ngoái: $Y_t - Y_{t-12}$). Dùng để triệt tiêu chu kỳ lặp lại.

3. **Kết luận từ ADF Test:**
   - **Với chuỗi `Diff_1`:** Giá trị p-value có thể đã giảm nhưng vẫn phải cẩn thận vì yếu tố mùa vụ làm đồ thị dao động mạnh.
   - **Với chuỗi `Diff_1_12`:** Sau khi kết hợp cả sai phân thường và sai phân mùa vụ, đồ thị dao động ổn định quanh trục 0 (mean $\approx$ 0). p-value từ kiểm định ADF rơi xuống $< 0.05$. 
   => **Kết luận:** Chốt tham số $d=1$ cho mô hình ARIMA. Chốt $(d=1, D=1, s=12)$ cho mô hình SARIMA.