# Giai đoạn chọn mô hình

## Mục đích

Chọn một mô hình triển khai dựa trên dữ liệu Validation mà không nhìn vào Test. Khóa tiêu chí trước khi mở Test giúp tránh lựa chọn mô hình có vẻ tốt chỉ vì vô tình tối ưu trên tập đánh giá cuối.

## Đầu vào - Đầu ra

**Đầu vào:** chỉ số của các mô hình ứng viên sau khi khớp trên tập Huấn luyện và dự báo tại cùng timestamp của tập Xác thực.

**Đầu ra:** tên mô hình tốt nhất, chỉ số xác thực, tiêu chí lựa chọn và mã lần chạy được ghi trong nhật ký, báo cáo và tóm tắt mô hình.

## Cách thực hiện

1. Xác nhận mọi ứng viên được đánh giá trên cùng các dòng Xác thực và cùng cấu trúc đặc trưng.
2. Xếp hạng theo RMSE xác thực tăng dần; nếu hòa, dùng MAE xác thực để phân định.
3. Ghi ứng viên được chọn cùng RMSE/MAE và quy tắc lựa chọn.
4. Khóa mô hình tốt nhất trước khi đánh giá trên tập Kiểm tra.
5. Sau khi khóa, khớp lại các ứng viên trên Huấn luyện + Xác thực theo quy trình pipeline; chỉ dùng tập Kiểm tra cho đánh giá cuối và không đổi mô hình đã chọn.

## Chỉ số đánh giá

Tiêu chí chính: **Validation RMSE** (càng thấp càng tốt). MAE, MAPE, sMAPE và R² dùng để diễn giải thêm. Trung bình/độ lệch chuẩn CV là thông tin về độ ổn định, không thay thế tiêu chí Xác thực. Chỉ số trên tập Kiểm tra không thuộc điểm lựa chọn.

## Dữ liệu sử dụng

Chỉ Validation partition 15% để lựa chọn. Train là dữ liệu fit trước đó. Không dùng Test labels hoặc kết quả Test để quyết định model/hyperparameters.

## Tham chiếu cài đặt

- `pipeline.py` (validation loop và chọn `min(validation_RMSE)`)
- `reports/model_summary/lab03/model_summary.*`
