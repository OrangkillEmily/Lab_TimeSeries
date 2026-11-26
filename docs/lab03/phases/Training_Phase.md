# Giai đoạn huấn luyện

## Mục đích

Huấn luyện các candidate models trên tập dữ liệu hợp lệ, ghi lại tham số/timing và tạo ra mô hình đủ điều kiện đánh giá. Tất cả estimator dự báo trong Lab 3 được tự cài đặt; không gọi estimator `sklearn` hoặc `xgboost`.

## Đầu vào - Đầu ra

**Đầu vào:** `X_train/y_train` và cấu hình thuật toán từ `config.yaml`; seed để tái lập; hàm tạo các mô hình Giá trị gần nhất, Hồi quy tuyến tính, CART, Rừng ngẫu nhiên và kiểu XGBoost.

**Đầu ra:** các mô hình ứng viên được khớp trên tập Huấn luyện để dự báo tập Xác thực; mô hình cuối được khớp lại trên Huấn luyện + Xác thực; tệp pickle trong `models/lab03/`; tham số và thời gian khớp/dự báo trong `model_summary`.

## Cách thực hiện

1. Tạo mô hình mới bằng hàm khởi tạo để các lần khớp không dùng chung trạng thái.
2. Khớp mô hình ứng viên trên 70% tập Huấn luyện; dự báo 15% tập Xác thực và ghi chỉ số. Kết quả Xác thực dùng để chọn mô hình.
3. Sau khi xác định mô hình tốt nhất, khớp lại các mô hình cuối trên Huấn luyện + Xác thực (85%).
4. Lưu mô hình đã khớp thành tệp `.pkl` trong `models/lab03/`; tên tệp thể hiện thuật toán.
5. Ghi tham số, thời gian khớp/dự báo, số đặc trưng và số dòng thực dùng (có thể bị giới hạn bởi `max_train_rows`).
6. Không khớp mô hình trên tập Kiểm tra và không dùng tập này để chọn siêu tham số.

`--fast` chỉ giảm số cây/độ sâu/số dòng để kiểm tra nhanh; chế độ mặc định dùng cấu hình đầy đủ trong YAML. `--no-cv` là tùy chọn kiểm tra nhanh, không dùng cho lần chạy nộp cuối nếu yêu cầu đủ 5 fold.

## Chỉ số đánh giá

Huấn luyện tự thân không đo chất lượng tổng quát hóa. Ghi thời gian khớp, số dòng, tham số và tệp mô hình; chất lượng được báo cáo bằng chỉ số Xác thực/CV/Kiểm tra ở các giai đoạn tương ứng.

## Dữ liệu sử dụng

Huấn luyện để đánh giá ứng viên: chỉ tập Huấn luyện. Khớp lại lần cuối: Huấn luyện + Xác thực. Tập Kiểm tra không tham gia khớp. Trong kiểm định chéo, mỗi fold tạo mô hình mới và chỉ dùng phần huấn luyện của fold đó.

## Tham chiếu cài đặt

- `src/lab03/lab03_models/*.py`
- `pipeline.py:_model_factories`, `_fit_predict`
- `models/lab03/`
