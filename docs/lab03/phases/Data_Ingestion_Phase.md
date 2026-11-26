# Giai đoạn nạp dữ liệu

## Mục đích

Nạp dữ liệu tải điện hourly từ nguồn CSV, kiểm tra schema, chuyển kiểu dữ liệu và ghi metadata nguồn để bảo đảm mỗi lần chạy có thể truy vết. Đây là bước đầu tiên; không được thay đổi file raw hoặc gán nhãn synthetic thành AEP.

## Đầu vào - Đầu ra

**Đầu vào:** `data/raw/AEP_hourly.csv`, tối thiểu có hai cột `Datetime` (mốc thời gian) và `AEP_MW` (giá trị số); tần suất `h`. Có thể truyền đường dẫn khác bằng `--data`. Khi cần kiểm tra mã ngoại tuyến, phải bật rõ `--synthetic`.

**Đầu ra:** chuỗi pandas có `DatetimeIndex` tăng dần, giữ nguyên tên biến mục tiêu; metadata gồm tên tệp, loại dữ liệu, đường dẫn, số dòng ban đầu, khoảng thời gian, tần suất và số quan sát sau làm sạch.

## Cách thực hiện

1. Xác định thư mục gốc dự án và đường dẫn dữ liệu từ `config.yaml` hoặc tham số dòng lệnh.
2. Kiểm tra file tồn tại; đọc CSV bằng pandas.
3. Xác nhận tên cột thời gian/biến mục tiêu. Chuyển thời gian bằng `to_datetime`, biến mục tiêu bằng `to_numeric`; loại và thống kê các dòng không thể chuyển đổi.
4. Sắp xếp thời gian tăng dần; xử lý timestamp trùng ở giai đoạn làm sạch.
5. Khi thiếu tệp dữ liệu thật, pipeline mặc định dừng với `FileNotFoundError`. Chỉ `--synthetic` mới tạo chuỗi minh họa nhân tạo.
6. Ghi tên bộ dữ liệu và `dataset_type` vào nhật ký kiểm toán, báo cáo và nhật ký chạy; không gọi dữ liệu minh họa là AEP.

## Chỉ số đánh giá

Ingestion không có metric dự báo. Tiêu chí nghiệm thu là schema đúng, timestamp parse được, target dạng số, frequency nhất quán, số dòng hợp lệ được ghi nhận và dataset type chính xác (`real_source` hoặc `synthetic_demo`).

## Dữ liệu sử dụng

Nguồn dữ liệu mục tiêu là phụ tải điện AEP theo giờ. Tệp thô do người dùng tải từ kho mã gốc hoặc cung cấp; gói Lab 03 chỉ gồm mã/tài liệu, không nhúng CSV dung lượng lớn. Bộ dữ liệu chỉ được đọc, không bị sửa đổi.

## Tham chiếu cài đặt

- `src/lab03/data.py:load_hourly_series`
- `pipeline.py`
- `script/lab03/run_lab03.py`, `scripts/run_lab.py`
