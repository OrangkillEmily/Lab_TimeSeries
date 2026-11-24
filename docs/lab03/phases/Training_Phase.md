# Training Phase

## Purpose

Huấn luyện các candidate models trên tập dữ liệu hợp lệ, ghi lại tham số/timing và tạo ra mô hình đủ điều kiện đánh giá. Tất cả estimator dự báo trong Lab 3 được tự cài đặt; không gọi estimator `sklearn` hoặc `xgboost`.

## Input - Output

**Input:** `X_train/y_train` và cấu hình thuật toán từ `config.yaml`; seed tái lập; factories tạo Naive, Linear Regression, CART, Random Forest và XGBoost-style.

**Output:** candidate fitted trên Train để dự báo Validation; final estimators refit trên Train+Validation; pickle files trong `models/lab03/`; thông số và thời gian fit/predict trong `model_summary`.

## How to do

1. Tạo một model mới từ factory để đảm bảo các lần fit không chia sẻ trạng thái.
2. Fit candidate trên Train 70%; dự đoán Validation 15% và ghi metric. Kết quả Validation được dùng cho Model Selection.
3. Khi champion đã được xác định, refit các candidate cuối cùng trên Train+Validation (85%).
4. Serialize model đã refit ra file `.pkl` trong `models/lab03/`; tên file thể hiện thuật toán.
5. Ghi parameters, fit seconds, predict seconds, feature count và số hàng thực dùng (có thể bị giới hạn bởi max_train_rows).
6. Không fit model trên Test và không dùng Test để chọn hyperparameter.

`--fast` chỉ giảm số cây/độ sâu/số hàng để smoke test; chế độ mặc định lấy cấu hình đầy đủ trong YAML. `--no-cv` là tùy chọn kiểm tra nhanh, không dùng cho lần chạy nộp cuối nếu yêu cầu đủ 5 folds.

## Evaluation Metrics

Training tự thân không đo chất lượng tổng quát hóa. Ghi thời gian fit, số hàng, parameters và model artifact; chất lượng được báo cáo bằng Validation/CV/Test metrics ở các phase tương ứng.

## Used Data

Training đánh giá candidate: Train only. Final refit: Train+Validation. Test bị giữ ngoài quá trình fit. Trong CV, mỗi fold tạo model mới và chỉ dùng fold training prefix.

## Implementation Reference

- `src/lab03/lab03_models/*.py`
- `pipeline.py:_model_factories`, `_fit_predict`
- `models/lab03/`
