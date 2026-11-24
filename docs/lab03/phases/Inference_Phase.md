# Inference Phase

## Purpose

Tạo dự báo cho các điểm Test theo giao thức rolling one-step và sinh dự báo tương lai 24 giờ theo giao thức recursive fixed-origin. Hai bài toán được phân biệt vì cách sử dụng thông tin quan sát khác nhau.

## Input - Output

**Input:** candidate model đã refit Train+Validation, feature schema, lịch sử target, `recursive_horizon=24`.

**Output:** `test_predictions.csv` căn chỉnh với actual; `recursive_forecast.csv` gồm 24 timestamp hourly sau thời điểm cuối dữ liệu; figure so sánh dự báo.

## How to do

1. Với Test rolling one-step, mỗi hàng Test được tạo feature từ target history đứng trước timestamp đó. Trong vận hành liên tục, actual của các timestamp Test trước có thể được dùng cho bước tiếp theo; đây là one-step-ahead, không phải dự báo nhiều bước từ một origin cố định.
2. Với forecast tương lai, lấy toàn bộ lịch sử quan sát tới origin cuối cùng.
3. Xây một hàng feature bằng lag/rolling/calendar schema đã dùng khi fit.
4. Gọi `predict` để nhận dự báo tại timestamp kế tiếp.
5. Nối giá trị dự báo vào history (không có actual tương lai) rồi lặp cho tới đủ 24 bước.
6. Ghi cả prediction lẫn timestamp, giữ nguyên tên champion và run metadata.

## Evaluation Metrics

Test rolling one-step: MAE, MSE, RMSE, MAPE, sMAPE, R² khi có actual. Recursive future không có ground truth trong cùng run, do đó không bịa metric; chỉ nghiệm thu horizon đúng 24, timestamp cách nhau một giờ và dự báo hữu hạn. Khi có actual tương lai mới, có thể đánh giá backtest riêng.

## Used Data

Test feature tại t chỉ dùng lịch sử tới t-1; future recursive dùng quan sát lịch sử đến origin và các prediction vừa sinh ra. Không dùng actual tương lai trong recursive loop.

## Implementation Reference

- `src/lab03/features.py:make_next_step_features`
- `src/lab03/features.py:recursive_forecast`
- `pipeline.py`
