# Data Ingestion Phase

## Purpose

Nạp dữ liệu tải điện hourly từ nguồn CSV, kiểm tra schema, chuyển kiểu dữ liệu và ghi metadata nguồn để bảo đảm mỗi lần chạy có thể truy vết. Đây là bước đầu tiên; không được thay đổi file raw hoặc gán nhãn synthetic thành AEP.

## Input - Output

**Input:** `data/raw/AEP_hourly.csv`, tối thiểu có hai cột `Datetime` (timestamp) và `AEP_MW` (giá trị số); frequency `h`. Có thể truyền đường dẫn khác bằng `--data`. Khi cần kiểm tra code ngoại tuyến, phải bật rõ `--synthetic`.

**Output:** pandas Series có `DatetimeIndex` tăng dần, tên target được giữ lại; metadata gồm tên file, loại dữ liệu, đường dẫn, số dòng ban đầu, khoảng thời gian, frequency và số quan sát sau làm sạch.

## How to do

1. Resolve project root và đường dẫn dữ liệu từ `config.yaml` hoặc tham số CLI.
2. Kiểm tra file tồn tại; đọc CSV bằng pandas.
3. Xác nhận tên cột timestamp/target. Parse timestamp với `to_datetime`, target bằng `to_numeric`; hàng không parse được bị loại và thống kê.
4. Sort timestamp tăng dần, xử lý timestamp trùng ở Cleaning Phase.
5. Khi thiếu file thật, pipeline mặc định dừng với `FileNotFoundError`. Chỉ `--synthetic` mới tạo chuỗi demo nhân tạo.
6. Ghi tên dataset và `dataset_type` vào audit/report/log; không gọi demo là AEP.

## Evaluation Metrics

Ingestion không có metric dự báo. Tiêu chí nghiệm thu là schema đúng, timestamp parse được, target dạng số, frequency nhất quán, số dòng hợp lệ được ghi nhận và dataset type chính xác (`real_source` hoặc `synthetic_demo`).

## Used Data

Nguồn mục tiêu: AEP hourly electric load. File raw do người dùng tải từ repository gốc hoặc cung cấp; gói Lab 3 chỉ gồm code/tài liệu, không nhúng bản CSV nhiều MB. Dataset được dùng ở chế độ read-only.

## Implementation Reference

- `src/lab03/data.py:load_hourly_series`
- `pipeline.py`
- `script/lab03/run_lab03.py`, `scripts/run_lab.py`
