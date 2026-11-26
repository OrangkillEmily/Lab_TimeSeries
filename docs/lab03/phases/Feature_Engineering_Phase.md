# Giai đoạn tạo đặc trưng

## Mục đích

Chuyển chuỗi univariate thành supervised learning table cho dự báo một bước. Lag features nắm bắt sự phụ thuộc tức thời/ngày/tuần; rolling features mô tả mức trung bình và biến động gần đây; biến lịch mã hóa tính chu kỳ giờ/ngày/tháng mà không tạo bước nhảy nhân tạo ở ranh giới chu kỳ.

## Đầu vào - Đầu ra

**Đầu vào:** Chuỗi theo giờ đã làm sạch; độ trễ `[1,2,3,6,12,24,48,168]`; cửa sổ trượt `[6,12,24,168]`; tùy chọn đặc trưng lịch.

**Đầu ra:** `X ∈ R^(n×p)` và `y ∈ R^n` cùng `DatetimeIndex` của timestamp mục tiêu. Loại các dòng đầu chưa đủ lịch sử; không còn NA/Inf.

## Cách thực hiện

1. Với mỗi lag `k`, tạo `lag_k(t)=y(t-k)` bằng `shift(k)`.
2. Định nghĩa `past=y.shift(1)` trước khi tính cửa sổ trượt; tạo `roll_mean_w` và `roll_std_w` trên w quan sát gần nhất của `past`. Dịch chuỗi trước khi tính là điều kiện ngăn rò rỉ dữ liệu.
3. Mã hóa `hour`, `dayofweek`, `month` bằng sin/cos: `sin(2πc/P)`, `cos(2πc/P)` để chu kỳ liền kề nhau ở biên.
4. Chỉ ghép giá trị mục tiêu hiện tại vào cột y; thay Inf bằng NA và loại các dòng khởi động chưa đủ độ trễ/cửa sổ trượt.
5. Giữ index tăng dần, đảm bảo cột `lag_1` đứng trước để Naive baseline đọc được đúng cột.
6. Lưu các bảng đặc trưng đã chia để kiểm toán.
7. Khi suy luận đệ quy, dùng `make_next_step_features` với tên đặc trưng gốc và lịch sử hiện có; nối dự báo vào lịch sử cho bước tiếp theo.

## Chỉ số đánh giá

Đếm số feature, số hàng ban đầu, số hàng warm-up bị loại và số hàng cuối. Unit test kiểm tra `lag_1(t)=y(t-1)`, rolling kết thúc tại `t-1`, index đồng bộ và không NaN. Phase không có forecast metric riêng.

## Dữ liệu sử dụng

Tại timestamp target `t`, chỉ sử dụng observations có timestamp nhỏ hơn `t`; không dùng `y(t)` hay giá trị tương lai trong predictor. Calendar features là xác định từ timestamp, không sử dụng target.

## Tham chiếu cài đặt

- `src/lab03/features.py:build_supervised_features`
- `src/lab03/features.py:make_next_step_features`
- `src/lab03/features.py:recursive_forecast`
