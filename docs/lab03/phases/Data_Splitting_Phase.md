# Giai đoạn chia dữ liệu

## Mục đích

Tạo ba partition theo đúng thứ tự thời gian: Train để fit; Validation để chọn mô hình; Test giữ kín để đánh giá cuối. Cách chia này mô phỏng việc dự báo tương lai từ quá khứ và ngăn rò rỉ do shuffle.

## Đầu vào - Đầu ra

**Đầu vào:** ma trận `X`, biến mục tiêu `y` đã được sắp tăng dần; cấu hình `train_ratio=0.70`, `val_ratio=0.15`, `test_ratio=0.15`, `shuffle=false`.

**Đầu ra:** `X_train/y_train`, `X_val/y_val`, `X_test/y_test` là các đoạn liên tiếp, không chồng lấp; metadata gồm số dòng và timestamp bắt đầu/kết thúc của từng phần.

## Cách thực hiện

1. Kiểm tra ba tỷ lệ có tổng bằng 1 trong sai số số học cho phép.
2. Tính `n_train=floor(0.70n)`, `n_val=floor(0.15n)`, phần còn lại dành cho Test.
3. Cắt theo vị trí thời gian bằng `iloc`; tuyệt đối không gọi shuffle/random split.
4. Kiểm tra chỉ mục từng phần tăng dần và điều kiện `max(train) < min(validation)`, `max(validation) < min(test)`.
5. Lưu ba file `train_features.csv`, `validation_features.csv`, `test_features.csv`.
6. Chỉ đọc tập Xác thực khi chọn mô hình; chỉ dùng tập Kiểm tra sau khi đã khóa mô hình tốt nhất.

## Chỉ số đánh giá

Báo cáo số dòng, tỷ lệ thực tế, timestamp bắt đầu/kết thúc, số đặc trưng và các điều kiện kiểm tra về thứ tự/không chồng lấp. Đây là tiêu chí nghiệm thu cách chia, không phải chỉ số chất lượng dự báo.

## Dữ liệu sử dụng

Dùng toàn bộ supervised rows từ chuỗi đã làm sạch. Phần warm-up do lag/rolling đã bị loại trước khi chia. Không shuffle và không chia ngẫu nhiên theo từng quan sát.

## Tham chiếu cài đặt

- `src/lab03/data.py:chronological_split`
- `pipeline.py`
