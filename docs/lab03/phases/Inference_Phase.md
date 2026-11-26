# Giai đoạn suy luận

## Mục đích

Tạo dự báo cho các điểm Test theo giao thức rolling one-step và sinh dự báo tương lai 24 giờ theo giao thức recursive fixed-origin. Hai bài toán được phân biệt vì cách sử dụng thông tin quan sát khác nhau.

## Đầu vào - Đầu ra

**Đầu vào:** mô hình ứng viên đã khớp lại trên Huấn luyện + Xác thực, cấu trúc đặc trưng, lịch sử biến mục tiêu, `recursive_horizon=24`.

**Đầu ra:** `test_predictions.csv` căn chỉnh với giá trị thực; `recursive_forecast.csv` gồm 24 timestamp theo giờ sau thời điểm cuối dữ liệu; biểu đồ so sánh dự báo.

## Cách thực hiện

1. Với dự báo cuốn chiếu trước một bước trên tập Kiểm tra, mỗi dòng được tạo đặc trưng từ lịch sử mục tiêu trước timestamp đó. Trong vận hành liên tục, giá trị thực ở timestamp Kiểm tra trước đó có thể dùng cho bước tiếp theo; đây là dự báo trước một bước, không phải dự báo nhiều bước từ một mốc cố định.
2. Với forecast tương lai, lấy toàn bộ lịch sử quan sát tới origin cuối cùng.
3. Xây một hàng feature bằng lag/rolling/calendar schema đã dùng khi fit.
4. Gọi `predict` để nhận dự báo tại timestamp kế tiếp.
5. Nối giá trị dự báo vào lịch sử (không có giá trị thực tương lai) rồi lặp cho đủ 24 bước.
6. Ghi dự báo và timestamp, giữ nguyên tên mô hình tốt nhất và metadata lần chạy.

## Chỉ số đánh giá

Dự báo cuốn chiếu một bước trên tập Kiểm tra: MAE, MSE, RMSE, MAPE, sMAPE, R² khi có giá trị thực. Dự báo tương lai đệ quy không có giá trị chuẩn trong cùng lần chạy nên không tạo chỉ số giả; chỉ xác nhận đúng 24 mốc, timestamp cách nhau một giờ và dự báo hữu hạn. Khi có giá trị thực tương lai, có thể đánh giá bằng một phép kiểm thử hồi cứu riêng.

## Dữ liệu sử dụng

Test feature tại t chỉ dùng lịch sử tới t-1; future recursive dùng quan sát lịch sử đến origin và các prediction vừa sinh ra. Không dùng actual tương lai trong recursive loop.

## Tham chiếu cài đặt

- `src/lab03/features.py:make_next_step_features`
- `src/lab03/features.py:recursive_forecast`
- `pipeline.py`
