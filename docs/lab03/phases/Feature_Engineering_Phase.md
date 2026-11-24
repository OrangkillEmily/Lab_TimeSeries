# Feature Engineering Phase

## Purpose

Chuyển chuỗi univariate thành supervised learning table cho dự báo một bước. Lag features nắm bắt sự phụ thuộc tức thời/ngày/tuần; rolling features mô tả mức trung bình và biến động gần đây; biến lịch mã hóa tính chu kỳ giờ/ngày/tháng mà không tạo bước nhảy nhân tạo ở ranh giới chu kỳ.

## Input - Output

**Input:** Series hourly đã sạch; lags `[1,2,3,6,12,24,48,168]`; rolling windows `[6,12,24,168]`; tùy chọn calendar features.

**Output:** `X ∈ R^(n×p)` và `y ∈ R^n` cùng DatetimeIndex target timestamp. Các hàng đầu tiên chưa đủ lịch sử bị loại; không còn NA/Inf.

## How to do

1. Với mỗi lag `k`, tạo `lag_k(t)=y(t-k)` bằng `shift(k)`.
2. Định nghĩa `past=y.shift(1)` trước khi tính rolling; tạo `roll_mean_w` và `roll_std_w` trên w quan sát gần nhất của `past`. Shift trước rolling là điều kiện chống leakage.
3. Mã hóa `hour`, `dayofweek`, `month` bằng sin/cos: `sin(2πc/P)`, `cos(2πc/P)` để chu kỳ liền kề nhau ở biên.
4. Join target hiện tại chỉ ở cột y; thay Inf bằng NA và drop các hàng warm-up không đủ lag/rolling.
5. Giữ index tăng dần, đảm bảo cột `lag_1` đứng trước để Naive baseline đọc được đúng cột.
6. Lưu các split feature table để kiểm toán.
7. Khi inference recursive, dùng `make_next_step_features` với feature names gốc và lịch sử hiện có; prediction được nối vào history cho bước tiếp theo.

## Evaluation Metrics

Đếm số feature, số hàng ban đầu, số hàng warm-up bị loại và số hàng cuối. Unit test kiểm tra `lag_1(t)=y(t-1)`, rolling kết thúc tại `t-1`, index đồng bộ và không NaN. Phase không có forecast metric riêng.

## Used Data

Tại timestamp target `t`, chỉ sử dụng observations có timestamp nhỏ hơn `t`; không dùng `y(t)` hay giá trị tương lai trong predictor. Calendar features là xác định từ timestamp, không sử dụng target.

## Implementation Reference

- `src/lab03/features.py:build_supervised_features`
- `src/lab03/features.py:make_next_step_features`
- `src/lab03/features.py:recursive_forecast`
