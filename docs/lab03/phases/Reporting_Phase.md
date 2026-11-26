# Giai đoạn lập báo cáo

## Mục đích

Tạo bộ kết quả có thể kiểm toán và nộp: log, report ba định dạng, model summary, bảng metric, dữ liệu dự báo và mọi biểu đồ được lưu thành file ảnh. Báo cáo phải phân biệt lần chạy mới với thí nghiệm legacy trong notebook.

## Đầu vào - Đầu ra

**Đầu vào:** kiểm toán dữ liệu, metadata chia tập, cấu hình đặc trưng, chi tiết CV, chỉ số Xác thực/Kiểm tra, tham số mô hình, dự báo, biểu đồ và nhật ký lần chạy.

**Đầu ra:** `reports/lab03/Lab03_Report.md`, `.html`, `.pdf`; `reports/model_summary/lab03/` gồm CSV/JSON/Markdown và chỉ số từng fold CV; `images/lab03/*.png`; `logs/lab03/<run_id>.log`; `data/processed/lab03/run_result.json`.

## Cách thực hiện

1. Ghi log START ngay khi pipeline bắt đầu; mọi phase ghi INFO/WARNING và các score chính.
2. Tổng hợp metric, champion, model parameters và data audit.
3. Ghi báo cáo Markdown gốc.
4. Kết xuất Markdown thành HTML có CSS, metadata và đường dẫn hình.
5. Dùng ReportLab tạo PDF và nhúng biểu đồ thực tế vào tài liệu.
6. Ghi tóm tắt mô hình dạng CSV/JSON/Markdown để chương trình và người dùng đọc được.
7. Lưu mọi biểu đồ do mô-đun trực quan hóa tạo ra; không để hình chỉ tồn tại trong notebook.
8. Ghi `run_result.json`, trạng thái tiến trình và nhật ký thành công. Nếu có lỗi, ghi traceback/trạng thái thất bại rồi phát sinh lại ngoại lệ.
9. Giữ riêng biểu đồ/chỉ số notebook cũ trong thư mục lưu trữ; không gộp vào kết quả mô hình tự cài đặt hiện tại.

## Chỉ số đánh giá

Tiêu chí nghiệm thu: ba tệp báo cáo tồn tại và không rỗng; PDF mở/kết xuất được; nhật ký có BẮT ĐẦU/THÀNH CÔNG hoặc THẤT BẠI kèm traceback; tóm tắt mô hình có đủ mô hình/chỉ số; tệp fold CV có năm fold khi bật CV; đường dẫn hình trong báo cáo hợp lệ; JSON có loại dữ liệu, mô hình được chọn và đường dẫn sản phẩm.

## Dữ liệu sử dụng

Chỉ đưa sản phẩm của lần chạy hiện tại vào báo cáo kết quả chính. Kết quả notebook cũ được lưu riêng và mô tả là thí nghiệm dùng mô hình từ thư viện với cách chia dữ liệu cũ; không xem là phép so sánh tương đương.

## Tham chiếu cài đặt

- `src/lab03/lab03_utils/report.py`
- `src/lab03/plotting.py`
- `pipeline.py`
