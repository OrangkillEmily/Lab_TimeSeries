# BẢN ĐỒ CẤU TRÚC FILE LAB 02 (`FILE_GUIDE.md`)

Toàn bộ mã nguồn Lab 02 đã được chuẩn hóa trực tiếp trong `src/lab02/` (đã loại bỏ `src/data_loader.py` và làm phẳng `src/lab02/lab02_utils/`).

| Đường dẫn | Vai trò & Chức năng chính |
|---|---|
| `config.yaml` (nhánh `lab02`) | Cấu hình trung tâm: đường dẫn đầu ra, `test_ratio=0.20`, `shuffle=false`, thông số cột và `report_prefix` cho `dataset1` & `dataset2`. |
| `data/raw/Monthly Retail Sales.csv` | Dataset 1 (`204` tháng, `07/1991 – 06/2008`, cột `date`, `value`). |
| `data/raw/AirPassengers.csv` | Dataset 2 (`144` tháng, `01/1949 – 12/1960`, cột `Month`, `Passengers`). |
| `src/common/data_loader.py` | Lõi nạp CSV (`load_time_series_csv`) và chia tuần tự (`split_chronological`) dùng chung. |
| `src/lab02/utils.py` | `PROJECT_ROOT`, `load_config()`, `get_path()`, `ensure_directories()`, `set_global_seed()`, đọc/ghi JSON. |
| `src/lab02/data_loader.py` | `load_dataset(dataset_key, cfg)` và `split_dataset(df, dataset_key, cfg)` lưu ra `data/processed/lab02/`. |
| `src/lab02/tracking.py` | Quản lý `run_id`, ghi `logs/lab02/pipeline_<run_id>.log`, `pipeline_structured.jsonl` và `progress_results/lab02/latest_run.json`. |
| `src/lab02/logging_config.py` | Logger theo dataset (`setup_lab02_logger`, `get_log_file_path`) và theo từng phiên thực nghiệm (`setup_experiment_logger`). |
| `src/lab02/stats.py` | Kiểm định ADF (`perform_adf_test`), trích lag có ý nghĩa (`significant_lags`) và vẽ ACF/PACF (`plot_acf_pacf`). |
| `src/lab02/acf_pacf_analysis.py` | Hàm `analyze_acf_pacf()` phân tích bảng lag chi tiết (36 lags) và ngưỡng 95%. |
| `src/lab02/modeling.py` | Hàm huấn luyện/dự báo/chẩn đoán lõi (`fit_arima`, `fit_sarima`, `residual_diagnostics`, `forecast_test`, `calculate_metrics`, `evaluate_forecast`, `plot_arima_vs_sarima`, `plot_log_transformation`). |
| `src/lab02/model_summary.py` | `ModelSummaryRegistry` & `format_terminal_model_card()` xuất bảng so sánh và diễn giải đa tiêu chí tự động (`.csv`, `.json`, `.md`). |
| `src/lab02/visualize.py` | Kho hàm vẽ biểu đồ bảo toàn 100% phong cách gốc phục vụ báo cáo HTML/PDF. |
| `src/lab02/scipy_tsa_compat.py` | Động cơ tương thích NumPy/SciPy dự phòng cho các hàm `statsmodels` và `sklearn`. |
| `src/lab02/lab02_models/` | 5 module mô hình chuyên biệt: `ar_model.py`, `ma_model.py`, `arima_model.py`, `sarima_model.py`, `candidate_registry.py`. |
| `src/lab02/pipeline.py` | Điều phối toàn trình `run_full_pipeline()` và gộp `docs/lab02/experiment_logs.csv`. |
| `src/lab02/report.py` | Wrapper gọi `script/lab02/render_report.py`. |
| `script/lab02/run_lab02.py` | CLI chạy toàn trình Lab 02 (`--dataset`, `--mode`, `--no-figures`, `--verbose`, `--bundle-docs`). |
| `script/lab02/run_lab02_experiments.py` | CLI chạy bộ thực nghiệm, xuất `model_summary`, `progress_results` và biểu đồ báo cáo. |
| `script/lab02/render_report.py` | Engine xuất 2 bộ báo cáo `.html`, `.pdf`, `.md` (6 tệp) tại `reports/lab02/`. |
| `tests/lab02/` | Bộ kiểm thử tự động: `run_all_tests.py`, `test_lab02_architecture.py` (7 test), `test_lab02_pipeline_and_reports.py` (2 test). |
| `reports/model_summary/lab02/` | Bảng tổng hợp toàn bộ mô hình ứng viên (`.csv`, `.json`, `.md`) cho `dataset1` và `dataset2`. |
| `reports/lab02/` | 2 bộ báo cáo hoàn chỉnh: `LAB_02_DATASET1_REPORT.{html,pdf,md}` và `LAB_02_DATASET2_REPORT.{html,pdf,md}`. |
| `progress_results/lab02/` | `latest_run.json` tổng và `dataset1/latest_run.json`, `dataset2/latest_run.json`. |
