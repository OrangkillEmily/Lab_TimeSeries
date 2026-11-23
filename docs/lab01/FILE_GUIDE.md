# BẢN ĐỒ CẤU TRÚC FILE LAB 01 (`FILE_GUIDE.md`)

Toàn bộ mã nguồn Lab 01 đã được chuẩn hóa sang kiến trúc mô-đun `src/lab01/` kết hợp lõi nạp dữ liệu dùng chung `src/common/data_loader.py` (đã loại bỏ hoàn toàn các file cũ `src/data_loader.py` và `src/lab01_utils/`).

| Đường dẫn | Vai trò & Chức năng chính |
|---|---|
| `config.yaml` (nhánh `lab01`) | Cấu hình trung tâm: đường dẫn dữ liệu/báo cáo, `random_seed=42`, `acf_pacf_nlags=40`, `white_noise_n=1000`, `rolling_window=12`, `zscore_threshold=3.0`, `iqr_multiplier=1.5`. |
| `data/raw/AirPassengers.csv` | Dữ liệu gốc (144 tháng, `01/1949 – 12/1960`, cột `Month`, `Passengers`). |
| `src/common/data_loader.py` | Lõi nạp CSV (`load_time_series_csv`) và chia tập tuần tự (`split_chronological`) dùng chung cho các lab. |
| `src/lab01/utils.py` | Tiện ích đường dẫn (`PROJECT_ROOT`, `get_path`), nạp cấu hình (`load_config`), seed toàn cục và đọc/ghi JSON. |
| `src/lab01/tracking.py` | Quản lý phiên chạy (`run_id`), ghi log song song (`.log` + `.jsonl`) và cập nhật `progress_results/lab01/latest_run.json`. |
| `src/lab01/data_loader.py` | Giai đoạn 1: `load_airpassengers_dataset()` nạp dữ liệu, kiểm toán schema và lưu `data/processed/lab01/airpassengers_clean.csv`. |
| `src/lab01/eda.py` | Giai đoạn 2 (Bài 1 & 2): `build_eda_profile()`, `analyze_acf_pacf()`, `analyze_white_noise()` và xuất 4 biểu đồ `01_` → `04_`. |
| `src/lab01/decomposition.py` | Giai đoạn 3 (Bài 3): `compute_strength_metrics()`, `run_decomposition_analysis()` (Classical Additive/Multiplicative & STL) và xuất 3 biểu đồ `05_` → `07_`. |
| `src/lab01/quality.py` | Giai đoạn 4 (Bài 4): `simulate_dirty_series()`, `_score_outlier_method()`, `run_quality_pipeline()` và xuất 5 biểu đồ `08_` → `12_` + `airpassengers_quality_cleaned.csv`. |
| `src/lab01/visualize.py` | Kho hàm vẽ chuẩn cho toàn bộ Lab 01 và Giai đoạn 5 (Bài 5): `generate_visualization_suite()` xuất 3 biểu đồ `13_` → `15_` (tổng 15 PNG). |
| `src/lab01/report.py` | Wrapper chuẩn hóa gọi `script/lab01/render_report.py`. |
| `src/lab01/pipeline.py` | Bộ điều phối toàn trình chạy tuần tự 6 giai đoạn từ nạp dữ liệu đến kết xuất báo cáo. |
| `script/lab01/run_lab01.py` | Điểm khởi chạy CLI toàn trình cho Lab 01 (hỗ trợ cờ `--bundle-docs`). |
| `script/lab01/render_report.py` | Engine kết xuất bộ 3 báo cáo `reports/lab01/LAB_01_REPORT.{html,pdf,md}` nhúng 15 ảnh Base64. |
| `script/lab01/Lab01_TimeSeries.ipynb` | Notebook thực hành tương tác đã cập nhật hoàn toàn sang `src.lab01.*`. |
| `data/processed/lab01/` | Chứa `airpassengers_clean.csv`, `data_inspection_summary.json` và `airpassengers_quality_cleaned.csv`. |
| `reports/lab01/figures/` | Thư mục chứa 15 biểu đồ PNG (`01_full_series_overview.png` → `15_heatmaps_seasonality.png`). |
| `reports/lab01/metrics/` | Chứa các tệp JSON tóm tắt chỉ số từng giai đoạn (`eda_acf_whitenoise_summary.json`, `decomposition_summary.json`, `data_quality_summary.json`, `visualization_summary.json`). |
| `reports/lab01/` | Chứa bộ 3 báo cáo cuối: `LAB_01_REPORT.html`, `LAB_01_REPORT.pdf`, `LAB_01_REPORT.md`. |
| `progress_results/lab01/` | Lưu `latest_run.json` và lịch sử `run_summary_<run_id>.json`. |
| `logs/lab01/` | Lưu nhật ký chạy `pipeline_<run_id>.log` và `pipeline_structured.jsonl`. |
