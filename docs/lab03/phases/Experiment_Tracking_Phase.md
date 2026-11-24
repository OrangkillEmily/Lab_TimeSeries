# Experiment Tracking / Logging Phase

## Purpose

Lưu lại cấu hình và diễn tiến của mỗi lần thực thi để có thể kiểm tra nguồn dữ liệu, xác định phase lỗi, so sánh các run và tái lập kết quả. Log là artifact bắt buộc, không chỉ là nội dung in trên terminal.

## Input - Output

**Input:** run ID theo thời gian Asia/Ho_Chi_Minh, config, dataset metadata, fold/model metrics, artifact paths và exception nếu xảy ra.

**Output:** `logs/lab03/<run_id>.log`, `progress_results/lab03/latest_run.json` và `data/processed/lab03/run_result.json`.

## How to do

1. Tạo run ID duy nhất cho lần chạy và cấu hình file logger cùng console handler.
2. Ghi project/config path, dataset source/type, data shape, cleaning audit và split sizes.
3. Ghi từng fold, model, RMSE/MAE, thời gian fit/predict; ghi rõ model được chọn theo Validation RMSE.
4. Ghi đường dẫn model, ảnh, report và summary khi artifact được tạo.
5. Ghi `PIPELINE SUCCESS` chỉ sau khi report bundle và toàn bộ output chính đã ghi xong.
6. Nếu có exception, dùng logger.exception để lưu traceback và trạng thái FAILED rồi truyền lỗi ra ngoài.

## Evaluation Metrics

Kiểm tra log tồn tại, UTF-8 đọc được, run ID khớp file name, có START và SUCCESS trong run bình thường; log lỗi phải có FAILED/traceback. Metadata JSON phải nhất quán với log và report.

## Used Data

Thông tin cấu hình/dữ liệu của run hiện tại; không ghi dữ liệu nhạy cảm. Synthetic run phải được đánh dấu `synthetic_demo`.

## Implementation Reference

- `pipeline.py:_setup_logger`
- `logs/lab03/`
- `progress_results/lab03/`
