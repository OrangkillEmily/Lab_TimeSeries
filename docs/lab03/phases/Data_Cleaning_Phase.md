# Giai đoạn làm sạch dữ liệu

## Mục đích

Chuẩn hóa chuỗi theo lưới thời gian hourly liên tục, loại timestamp/target không hợp lệ và khôi phục các khoảng thiếu nhỏ theo thời gian. Phase này ngăn các lag bị lệch giờ và không tự lấp các khoảng mất dữ liệu dài không đáng tin cậy.

## Đầu vào - Đầu ra

**Đầu vào:** Chuỗi đọc từ CSV, timestamp có thể chưa đều, có thể trùng hoặc thiếu giờ; `max_interpolation_gap=3`.

**Đầu ra:** Chuỗi không có timestamp trùng, chỉ mục tăng dần, tần suất theo giờ và không còn giá trị thiếu; `data_audit.json` và `cleaned_hourly_series.csv` được lưu trong `data/processed/lab03/`.

## Cách thực hiện

1. Chuyển đổi và loại các dòng có timestamp hoặc giá trị mục tiêu không hợp lệ; ghi lại số dòng bị loại.
2. Sort theo timestamp tăng dần.
3. Đếm timestamp trùng rồi giữ bản ghi đầu tiên theo thứ tự đã sắp xếp.
4. Đặt DatetimeIndex và áp dụng `asfreq('h')` để thêm vị trí giờ thiếu dưới dạng NA.
5. Nội suy theo thời gian (`method='time'`) tối đa ba bước liên tiếp và chỉ vùng nằm bên trong chuỗi (`limit_area='inside'`).
6. Đếm giá trị thiếu còn lại. Nếu vẫn có NA (ví dụ khoảng trống dài hơn cấu hình hoặc vị trí thiếu ở biên), dừng ngay và yêu cầu kiểm tra dữ liệu nguồn.
7. Lưu audit JSON cùng dữ liệu đã regularize; không ghi đè file raw.

## Chỉ số đánh giá

Số dòng thô, dòng không hợp lệ, timestamp trùng bị loại, khoảng thời gian thiếu được bổ sung, giá trị NA sau nội suy, số quan sát cuối cùng, thời điểm bắt đầu/kết thúc và tần suất. Điều kiện để pipeline tiếp tục: không còn NA hoặc timestamp trùng; chỉ mục tăng dần.

## Dữ liệu sử dụng

Chỉ dùng dữ liệu AEP thô hoặc dữ liệu tổng hợp đã được gắn nhãn. Nội suy chỉ xử lý khoảng trống nhỏ bên trong chuỗi; không ngoại suy ở hai đầu và không dùng giá trị mục tiêu tương lai làm đặc trưng trực tiếp.

## Tham chiếu cài đặt

- `src/lab03/data.py:load_hourly_series`
- `pipeline.py`
