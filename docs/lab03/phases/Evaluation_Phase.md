# Evaluation Phase

## Purpose

Đo độ chính xác, độ ổn định và cấu trúc sai số sau khi model đã được chọn. Metric tables cung cấp bằng chứng định lượng; residual plots giúp nhận diện bias, outlier và pattern còn chưa được mô hình hóa.

## Input - Output

**Input:** y_true/y_pred của Validation, Test, các CV folds; champion residuals; model metadata.

**Output:** validation/test JSON, CV fold CSV, model_summary, test prediction CSV, residual/comparison figures và kết luận trong report.

## How to do

1. Tính `error = y_pred - y_true`.
2. MAE là trung bình trị tuyệt đối error; MSE là trung bình bình phương error; RMSE là căn bậc hai MSE.
3. MAPE là mean `|error/y| × 100` trên target khác 0; sMAPE là mean `2|error|/(|y|+|prediction|) × 100` trên mẫu có denominator khác 0.
4. R² so SSE với tổng bình phương độ lệch quanh mean target; nếu target constant, R² không xác định.
5. Tổng hợp metric theo model trên Validation, Test và folds; Test chỉ được mở sau model selection.
6. Vẽ actual-prediction, residual theo thời gian, residual distribution, so sánh model RMSE, CV-by-fold và feature importance.
7. Nêu rõ Test là rolling one-step; recursive future fixed-origin được báo riêng.

## Evaluation Metrics

**MAE/MSE/RMSE/MAPE/sMAPE/R²**. RMSE nhạy với lỗi lớn; MAE dễ diễn giải theo đơn vị MW; MAPE không ổn định khi target gần 0; sMAPE bounded hơn nhưng cũng cần cẩn trọng với denominator gần 0; R² có thể âm nếu dự báo kém hơn mean baseline. Residual mean gần 0 là tốt nhưng không đủ; cần xem residual theo thời gian.

## Used Data

Validation dùng cho lựa chọn; CV chỉ nằm trong Train; Test là held-out cuối cùng. Tất cả candidate dự báo trên cùng timestamp để so sánh công bằng. Demo synthetic được gắn nhãn trên mọi bảng.

## Implementation Reference

- `src/lab03/metrics.py:regression_metrics`
- `src/lab03/plotting.py:save_all_figures`
- `pipeline.py`
