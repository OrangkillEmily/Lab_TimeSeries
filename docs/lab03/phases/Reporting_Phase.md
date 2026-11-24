# Reporting Phase

## Purpose

Tạo bộ kết quả có thể kiểm toán và nộp: log, report ba định dạng, model summary, bảng metric, dữ liệu dự báo và mọi biểu đồ được lưu thành file ảnh. Báo cáo phải phân biệt lần chạy mới với thí nghiệm legacy trong notebook.

## Input - Output

**Input:** dataset audit, split metadata, feature config, CV details, validation/test metrics, model parameters, predictions, figures và run log.

**Output:** `reports/lab03/Lab03_Report.md`, `.html`, `.pdf`; `reports/model_summary/lab03/` gồm CSV/JSON/MD và CV fold metrics; `images/lab03/*.png`; `logs/lab03/<run_id>.log`; `data/processed/lab03/run_result.json`.

## How to do

1. Ghi log START ngay khi pipeline bắt đầu; mọi phase ghi INFO/WARNING và các score chính.
2. Tổng hợp metric, champion, model parameters và data audit.
3. Ghi Markdown report gốc.
4. Render Markdown thành HTML có CSS, metadata và đường dẫn hình.
5. Dùng ReportLab render PDF và nhúng các figure thực tế vào tài liệu.
6. Ghi model_summary dạng CSV/JSON/Markdown để đọc bằng code hoặc người dùng.
7. Lưu mọi figure do plotting sinh ra tại images folder; không để ảnh chỉ tồn tại inline trong notebook.
8. Ghi `run_result.json`, progress result và log SUCCESS. Nếu lỗi, log traceback và FAILED rồi raise exception.
9. Legacy notebook figures/metrics được giữ riêng dưới legacy; không nhập chung với kết quả scratch.

## Evaluation Metrics

Acceptance checklist: ba report files tồn tại và khác rỗng; PDF mở/render được; log có START/SUCCESS hoặc FAILED + traceback; model_summary đủ model/metrics; CV fold file có 5 fold khi CV bật; tất cả ảnh nằm trong `images/lab03/`; JSON có dataset type, champion và paths.

## Used Data

Chỉ artifact của run hiện tại được đưa vào report kết quả chính. Kết quả notebook cũ nằm trong section/dir Legacy và được mô tả là thí nghiệm dùng estimator thư viện với protocol split cũ; không xem là benchmark tương đương.

## Implementation Reference

- `src/lab03/lab03_utils/report.py`
- `src/lab03/plotting.py`
- `pipeline.py`
