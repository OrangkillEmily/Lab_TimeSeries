# Model Selection Phase

## Purpose

Chọn một mô hình triển khai dựa trên dữ liệu Validation mà không nhìn vào Test. Khóa tiêu chí trước khi mở Test giúp tránh lựa chọn mô hình có vẻ tốt chỉ vì vô tình tối ưu trên tập đánh giá cuối.

## Input - Output

**Input:** metric của các candidate sau khi fit trên Train và dự đoán trên cùng Validation timestamps.

**Output:** tên champion, validation metric, selection criterion và run ID được ghi trong log, report và model summary.

## How to do

1. Xác nhận tất cả candidate được đánh giá trên cùng các dòng Validation và cùng feature schema.
2. Xếp hạng theo Validation RMSE tăng dần; khi cần tie-break dùng Validation MAE.
3. Ghi candidate được chọn cùng RMSE/MAE và rule lựa chọn.
4. Khóa champion trước khi chạy phần Test evaluation.
5. Sau khi khóa, refit các candidate trên Train+Validation theo protocol của pipeline; Test chỉ được dùng để đánh giá cuối, không thay đổi champion đã chọn.

## Evaluation Metrics

Tiêu chí chính: **Validation RMSE** (càng thấp càng tốt). MAE, MAPE, sMAPE và R² dùng để diễn giải thêm. CV mean/std là bằng chứng ổn định nhưng không thay thế tiêu chí Validation. Test metric không thuộc selection score.

## Used Data

Chỉ Validation partition 15% để lựa chọn. Train là dữ liệu fit trước đó. Không dùng Test labels hoặc kết quả Test để quyết định model/hyperparameters.

## Implementation Reference

- `pipeline.py` (validation loop và chọn `min(validation_RMSE)`)
- `reports/model_summary/lab03/model_summary.*`
