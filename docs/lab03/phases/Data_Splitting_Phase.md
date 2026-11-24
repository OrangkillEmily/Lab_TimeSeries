# Data Splitting Phase

## Purpose

Tạo ba partition theo đúng thứ tự thời gian: Train để fit; Validation để chọn mô hình; Test giữ kín để đánh giá cuối. Cách chia này mô phỏng việc dự báo tương lai từ quá khứ và ngăn rò rỉ do shuffle.

## Input - Output

**Input:** ma trận `X`, target `y` đã được sắp tăng dần; cấu hình `train_ratio=0.70`, `val_ratio=0.15`, `test_ratio=0.15`, `shuffle=false`.

**Output:** `X_train/y_train`, `X_val/y_val`, `X_test/y_test` liên tiếp, không overlap; metadata gồm số hàng và timestamp bắt đầu/kết thúc của từng phần.

## How to do

1. Kiểm tra ba tỷ lệ có tổng bằng 1 trong sai số số học cho phép.
2. Tính `n_train=floor(0.70n)`, `n_val=floor(0.15n)`, phần còn lại dành cho Test.
3. Cắt theo vị trí thời gian bằng `iloc`; tuyệt đối không gọi shuffle/random split.
4. Kiểm tra index từng phần tăng dần và điều kiện `max(train) < min(validation)`, `max(validation) < min(test)`.
5. Lưu ba file `train_features.csv`, `validation_features.csv`, `test_features.csv`.
6. Model selection chỉ được đọc Validation; Test chỉ dùng sau khi champion đã được khóa.

## Evaluation Metrics

Báo cáo số dòng, tỷ lệ thực tế, start/end timestamp, feature count và các assertion về ordering/non-overlap. Đây là acceptance criteria của split, không phải metric chất lượng forecast.

## Used Data

Dùng toàn bộ supervised rows từ chuỗi đã làm sạch. Phần warm-up do lag/rolling đã bị loại trước khi chia. Không shuffle và không chia ngẫu nhiên theo từng quan sát.

## Implementation Reference

- `src/lab03/data.py:chronological_split`
- `pipeline.py`
