# Giai đoạn đánh giá

## Mục đích

Đo độ chính xác, độ ổn định và cấu trúc sai số sau khi model đã được chọn. Metric tables cung cấp bằng chứng định lượng; residual plots giúp nhận diện bias, outlier và pattern còn chưa được mô hình hóa.

## Đầu vào - Đầu ra

**Đầu vào:** `y_true/y_pred` của tập Xác thực, tập Kiểm tra và các fold CV; phần dư của mô hình tốt nhất; metadata mô hình.

**Đầu ra:** JSON Xác thực/Kiểm tra, CSV các fold CV, tóm tắt mô hình, CSV dự báo tập Kiểm tra, biểu đồ phần dư/so sánh và kết luận trong báo cáo.

## Cách thực hiện

1. Tính sai số `error = y_pred - y_true`.
2. MAE là trung bình trị tuyệt đối error; MSE là trung bình bình phương error; RMSE là căn bậc hai MSE.
3. MAPE là mean `|error/y| × 100` trên target khác 0; sMAPE là mean `2|error|/(|y|+|prediction|) × 100` trên mẫu có denominator khác 0.
4. R² so SSE với tổng bình phương độ lệch quanh mean target; nếu target constant, R² không xác định.
5. Tổng hợp chỉ số theo mô hình trên Xác thực, Kiểm tra và các fold; chỉ mở tập Kiểm tra sau khi chọn mô hình.
6. Vẽ giá trị thực/dự báo, phần dư theo thời gian, phân bố phần dư, so sánh RMSE mô hình, CV theo fold và mức quan trọng đặc trưng.
7. Nêu rõ tập Kiểm tra dùng dự báo cuốn chiếu trước một bước; dự báo tương lai đệ quy từ mốc cố định được báo cáo riêng.

## Chỉ số đánh giá

**MAE/MSE/RMSE/MAPE/sMAPE/R²**. RMSE nhạy với lỗi lớn; MAE dễ diễn giải theo đơn vị MW; MAPE không ổn định khi target gần 0; sMAPE bounded hơn nhưng cũng cần cẩn trọng với denominator gần 0; R² có thể âm nếu dự báo kém hơn mean baseline. Residual mean gần 0 là tốt nhưng không đủ; cần xem residual theo thời gian.

## Dữ liệu sử dụng

Tập Xác thực dùng để lựa chọn; CV chỉ nằm trong tập Huấn luyện; tập Kiểm tra được giữ riêng đến bước cuối. Mọi ứng viên dự báo trên cùng timestamp để so sánh công bằng. Dữ liệu tổng hợp phải được gắn nhãn trên mọi bảng.

## Tham chiếu cài đặt

- `src/lab03/metrics.py:regression_metrics`
- `src/lab03/plotting.py:save_all_figures`
- `pipeline.py`
