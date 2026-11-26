# Phương pháp Lab 03 - Dự báo mức tiêu thụ điện theo giờ

## 1. Xác định bài toán

Với chuỗi phụ tải theo giờ một biến `y_t`, ước lượng giá trị ở giờ kế tiếp dựa trên những thông tin có trước thời điểm `t`. Biến mục tiêu trong dữ liệu nguồn đã làm sạch là `AEP_MW`; cột thời gian là `Datetime`.

## 2. Chuẩn bị dữ liệu

- Sắp xếp timestamp và loại các bản ghi trùng.
- Chuẩn hóa chỉ mục thành tần suất một giờ.
- Chỉ nội suy các khoảng trống bên trong chuỗi có độ dài tối đa ba giờ liên tiếp.
- Dừng nếu còn khoảng trống dài hơn hoặc giá trị thiếu ở hai đầu chuỗi.
- Giữ nguyên CSV gốc; lưu riêng chuỗi đã làm sạch và nhật ký kiểm toán.

## 3. Đặc trưng có giám sát

Với mỗi timestamp mục tiêu `t`, tạo đặc trưng trễ `y_(t-k)` tại `k` bằng 1, 2, 3, 6, 12, 24, 48 và 168. Tính trung bình/độ lệch chuẩn trượt sau khi dịch chuỗi một bước (`y.shift(1)`), trên các cửa sổ 6, 12, 24 và 168 giờ. Mã hóa giờ/ngày trong tuần/tháng bằng sin/cos. Loại các dòng đầu chuỗi chưa đủ lịch sử.

## 4. Chia theo thời gian và quy trình xác thực

Chia các dòng có giám sát theo thứ tự thời gian: 70% Huấn luyện, 15% Xác thực và 15% Kiểm tra. Không xáo trộn. Tạo năm fold cửa sổ mở rộng chỉ bên trong tập Huấn luyện. Phần huấn luyện của mỗi fold luôn đứng trước khối dùng để xác thực, tránh khớp mô hình trên dữ liệu xảy ra sau giai đoạn xác thực.

## 5. Các thuật toán ứng viên

- Mốc đối chứng giá trị gần nhất: dự báo bằng `lag_1`.
- Hồi quy tuyến tính: bình phương tối thiểu thông thường với hệ số Ridge nhỏ để ổn định.
- CART: tham lam tối đa hóa mức giảm tổng bình phương sai số với ngưỡng ứng viên theo phân vị.
- Rừng ngẫu nhiên: kết hợp các cây CART tự cài đặt độc lập, lấy mẫu bootstrap và chọn ngẫu nhiên đặc trưng.
- Mô hình kiểu XGBoost: cây tăng cường gradient dạng histogram, dùng đạo hàm bậc nhất/bậc hai cho sai số bình phương và điều chuẩn L1/L2. Đây là bản cài đặt phục vụ học tập, không phải thư viện XGBoost chính thức.

Bốn mô hình học tham số được cài đặt trực tiếp bằng NumPy. Các chỉ số hồi quy cũng được tự cài đặt; không dùng estimator scikit-learn/XGBoost.

## 6. Chọn mô hình và kiểm tra cuối

Khớp từng ứng viên trên tập Huấn luyện và chấm trên cùng các timestamp của tập Xác thực. Chọn mô hình có RMSE xác thực thấp nhất. Sau khi khóa lựa chọn, khớp lại các mô hình cuối trên Huấn luyện + Xác thực rồi đánh giá trên tập Kiểm tra chưa sử dụng. Điểm kiểm tra không làm thay đổi lựa chọn mô hình.

## 7. Hai quy trình dự báo

- **Dự báo cuốn chiếu trước một bước trên tập Kiểm tra:** giá trị thực trước đó trong tập Kiểm tra có thể được dùng để tạo độ trễ ở bước sau; mô phỏng vận hành liên tục, mỗi giờ có thêm quan sát mới.
- **Dự báo tương lai đệ quy từ một mốc cố định:** chỉ dùng lịch sử có tại mốc dự báo; nối mỗi giá trị dự báo vào lịch sử để làm đầu vào cho bước tiếp theo. Mặc định dự báo 24 giờ.

Không trình bày hai quy trình này như cùng một bài toán.

## 8. Chỉ số đánh giá

MAE, MSE và RMSE cùng đơn vị với mục tiêu (MW); MAPE và sMAPE là phần trăm; R² so sánh tổng bình phương phần dư với độ biến thiên của mục tiêu. MAPE bỏ qua giá trị thực bằng 0 và có thể bất ổn khi giá trị gần 0. R² có thể âm trên dữ liệu ngoài mẫu.

## 9. Khả năng tái lập và nguồn gốc

Seed ngẫu nhiên được đặt trong cấu hình mô hình. Mỗi lần chạy ghi nhật ký có dấu thời gian, JSON kết quả, bảng chỉ số, tệp mô hình, biểu đồ và báo cáo Markdown/HTML/PDF. Kết quả notebook cũ được lưu riêng vì cách cài đặt mô hình và giao thức chia dữ liệu khác với pipeline hiện tại.
