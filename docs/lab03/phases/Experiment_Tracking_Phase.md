# Giai đoạn theo dõi thí nghiệm / ghi nhật ký

## Mục đích

Lưu lại cấu hình và diễn tiến của mỗi lần thực thi để có thể kiểm tra nguồn dữ liệu, xác định phase lỗi, so sánh các run và tái lập kết quả. Log là artifact bắt buộc, không chỉ là nội dung in trên terminal.

## Đầu vào - Đầu ra

**Đầu vào:** mã lần chạy theo múi giờ Asia/Ho_Chi_Minh, cấu hình, metadata dữ liệu, chỉ số fold/mô hình, đường dẫn sản phẩm và ngoại lệ nếu xảy ra.

**Đầu ra:** `logs/lab03/<run_id>.log`, `progress_results/lab03/latest_run.json` và `data/processed/lab03/run_result.json`.

## Cách thực hiện

1. Tạo mã lần chạy duy nhất và cấu hình bộ ghi tệp cùng bộ xử lý đầu ra dòng lệnh.
2. Ghi đường dẫn dự án/cấu hình, nguồn/loại dữ liệu, kích thước dữ liệu, kiểm toán làm sạch và kích thước các tập.
3. Ghi từng fold, mô hình, RMSE/MAE, thời gian khớp/dự báo; nêu rõ mô hình được chọn theo RMSE xác thực.
4. Ghi đường dẫn mô hình, hình, báo cáo và tóm tắt khi tạo sản phẩm.
5. Chỉ ghi `PIPELINE SUCCESS` sau khi ghi xong bộ báo cáo và mọi đầu ra chính.
6. Nếu có ngoại lệ, dùng `logger.exception` để lưu traceback/trạng thái thất bại rồi phát sinh lại lỗi.

## Chỉ số đánh giá

Kiểm tra nhật ký tồn tại, đọc được bằng UTF-8, mã lần chạy khớp tên tệp và có START/SUCCESS khi chạy bình thường; nhật ký lỗi phải có FAILED/traceback. Metadata JSON phải nhất quán với nhật ký và báo cáo.

## Dữ liệu sử dụng

Thông tin cấu hình/dữ liệu của run hiện tại; không ghi dữ liệu nhạy cảm. Synthetic run phải được đánh dấu `synthetic_demo`.

## Tham chiếu cài đặt

- `pipeline.py:_setup_logger`
- `logs/lab03/`
- `progress_results/lab03/`
