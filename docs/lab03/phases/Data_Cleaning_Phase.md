# Data Cleaning Phase

## Purpose

Chuẩn hóa chuỗi theo lưới thời gian hourly liên tục, loại timestamp/target không hợp lệ và khôi phục các khoảng thiếu nhỏ theo thời gian. Phase này ngăn các lag bị lệch giờ và không tự lấp các khoảng mất dữ liệu dài không đáng tin cậy.

## Input - Output

**Input:** Series đọc từ CSV, timestamp có thể chưa đều, có thể có duplicate hoặc missing hour; `max_interpolation_gap=3`.

**Output:** Series unique timestamp, index tăng dần, frequency hourly, không còn missing values; `data_audit.json` và `cleaned_hourly_series.csv` lưu trong `data/processed/lab03/`.

## How to do

1. Parse/loại hàng có timestamp hoặc target invalid và ghi số dòng bị loại.
2. Sort theo timestamp tăng dần.
3. Đếm duplicate timestamp rồi giữ bản ghi đầu tiên theo thứ tự đã sắp xếp.
4. Đặt DatetimeIndex và áp dụng `asfreq('h')` để thêm vị trí giờ thiếu dưới dạng NA.
5. Nội suy theo thời gian (`method='time'`) tối đa ba bước liên tiếp và chỉ vùng nằm bên trong chuỗi (`limit_area='inside'`).
6. Đếm missing còn lại. Nếu còn NA (ví dụ gap dài hơn cấu hình hoặc khoảng thiếu ở biên), fail-fast và yêu cầu kiểm tra nguồn.
7. Lưu audit JSON cùng dữ liệu đã regularize; không ghi đè file raw.

## Evaluation Metrics

Số hàng raw, invalid rows, số timestamp duplicate bị loại, missing periods được chèn, số NA sau nội suy, số quan sát cuối cùng, start/end timestamp và frequency. Điều kiện để pipeline tiếp tục: không còn NA; không còn timestamp trùng; index tăng dần.

## Used Data

Chỉ dữ liệu raw AEP hoặc synthetic được gắn nhãn. Nội suy chỉ xử lý lỗ hổng nhỏ bên trong chuỗi; không ngoại suy ở hai đầu và không dùng target tương lai ngoài khoảng cho mục đích feature trực tiếp.

## Implementation Reference

- `src/lab03/data.py:load_hourly_series`
- `pipeline.py`
