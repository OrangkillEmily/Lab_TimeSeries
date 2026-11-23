# 05 — Giai đoạn Tổng hợp & Kết xuất Báo cáo Học thuật (`Reporting_Phase`)

## 1. Purpose
Giai đoạn cuối của đường ống Lab 01 tự động tổng hợp toàn bộ số liệu định lượng từ `progress_results/lab01/latest_run.json` (nguyên tắc **Zero Hardcoding**) cùng **15 biểu đồ PNG** mã hóa Base64 để xuất đồng bộ bộ 3 báo cáo học thuật chuẩn nộp bài (`.html`, `.pdf`, `.md`).

## 2. Input-Output
| Thành phần | Đặc tả chi tiết |
|---|---|
| **Input** | `progress_results/lab01/latest_run.json`, các tệp JSON trong `reports/lab01/metrics/` và 15 tệp PNG trong `reports/lab01/figures/`. |
| **Module xử lý** | `src/lab01/report.py` (wrapper gọi `script/lab01/render_report.py`), `src/lab01/tracking.py` (`record_stage_result`, `finalize_run_session`). |
| **Output (Reports)** | `reports/lab01/LAB_01_REPORT.html` (HTML5 độc lập nhúng Base64), `reports/lab01/LAB_01_REPORT.pdf` (kết xuất qua Headless Edge/Chrome), `reports/lab01/LAB_01_REPORT.md` (tóm tắt Markdown). |
| **Output (Tracking)** | Cập nhật `stages.reporting` và trạng thái `COMPLETED` trong `progress_results/lab01/latest_run.json` và `logs/lab01/pipeline_<run_id>.log`. |

## 3. How to do
1. **Đọc trạng thái thực nghiệm (`load_json`):** Lấy `run_id`, thống kê EDA, ACF/PACF, White Noise, Classical/STL Decomposition, bảng đánh giá Imputation & 4 phương pháp Outlier, cùng thống kê trực quan hóa từ `latest_run.json`.
2. **Nhúng ảnh Base64 độc lập (`load_b64`):** Mã hóa toàn bộ 15 tệp PNG thành chuỗi `data:image/png;base64,...` giúp tệp HTML hiển thị trọn vẹn ngay cả khi gửi đi độc lập mà không cần thư mục ảnh đi kèm.
3. **Kết xuất PDF không cần phụ thuộc ngoài phức tạp:** Tự động dò tìm trình duyệt Chromium/Edge có sẵn trên máy (`msedge`, `chrome`, `chromium`) ở chế độ `--headless --print-to-pdf` để bảo toàn 100% bố cục CSS, công thức và độ nét biểu đồ.
4. **Xuất bản tóm tắt Markdown (`LAB_01_REPORT.md`):** Ghi nhanh các bảng số liệu chính và kết luận đánh giá khả năng dự báo của chuỗi `AirPassengers`.

## 4. Evaluation Metrics
| Tiêu chí nghiệm thu báo cáo | Yêu cầu | Kết quả phiên chạy `20261123_154119_4f8c41` |
|---|---|---|
| **Tính đầy đủ 3 định dạng** | `.html`, `.pdf`, `.md` đều tồn tại và dung lượng > 0 | `LAB_01_REPORT.html` (`3,982,187 bytes`), `LAB_01_REPORT.pdf` (`3,016,905 bytes`), `LAB_01_REPORT.md` (`4,684 bytes`) |
| **Tỷ lệ nhúng biểu đồ** | Đủ 15/15 biểu đồ không lỗi SVG giữ chỗ | `15/15` biểu đồ nhúng thành công |
| **Thời gian kết xuất** | Nhanh, tự động hoàn toàn | `4.20 giây` (toàn pipeline Lab 01 hoàn tất trong ~17 giây) |
| **Kết luận khả năng dự báo** | Tổng hợp từ Bài 1 → Bài 5 | **RẤT PHÙ HỢP** cho dự báo chuỗi thời gian; khuyến nghị biến đổi Log + sai phân $(d=1, D=1, s=12)$ và mô hình **Log-SARIMA** (triển khai ở Lab 02) |

## 5. Used data
- **Dữ liệu tổng hợp:** `progress_results/lab01/latest_run.json` (và bản lưu trữ `progress_results/lab01/run_summary_20261123_154119_4f8c41.json`).
- **Nhật ký kiểm toán:** `logs/lab01/pipeline_<run_id>.log` và `logs/lab01/pipeline_structured.jsonl`.
