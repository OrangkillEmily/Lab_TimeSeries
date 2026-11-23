# KIẾN TRÚC HỆ THỐNG LAB 02 (`LAB02_ARCHITECTURE.md`)

## 1. Tổng quan Kiến trúc Phiên bản Hiện tại (`v2.0.0`)
Hệ thống Lab 02 được chuẩn hóa theo kiến trúc **modular pipeline** đồng bộ với `config.yaml` trung tâm và lõi nạp dữ liệu dùng chung `src/common/data_loader.py` (đã loại bỏ hoàn toàn `src/data_loader.py` cũ và làm phẳng thư mục `src/lab02/lab02_utils/` trực tiếp vào `src/lab02/`).

```text
config.yaml (nhánh lab02)
   │
   ├── src/common/data_loader.py          (load_time_series_csv, split_chronological)
   │        │
   ├── src/lab02/
   │   ├── utils.py                       (PROJECT_ROOT, load_config, get_path, JSON)
   │   ├── data_loader.py                 (load_dataset, split_dataset -> data/processed/lab02/)
   │   ├── tracking.py                    (pipeline_<run_id>.log, latest_run.json)
   │   ├── logging_config.py              (setup_lab02_logger, setup_experiment_logger)
   │   ├── stats.py                       (perform_adf_test, significant_lags, plot_acf_pacf)
   │   ├── acf_pacf_analysis.py           (analyze_acf_pacf)
   │   ├── modeling.py                    (fit_arima/sarima, residual_diagnostics, metrics, plots)
   │   ├── model_summary.py               (ModelSummaryRegistry, multi-criterion interpretation)
   │   ├── visualize.py                   (hàm vẽ biểu đồ nét gốc cho báo cáo)
   │   ├── scipy_tsa_compat.py            (lõi dự phòng SciPy/NumPy khi thiếu statsmodels)
   │   ├── report.py                      (wrapper gọi script/lab02/render_report.py)
   │   ├── pipeline.py                    (run_full_pipeline, export_combined_experiment_logs_csv)
   │   └── lab02_models/                  (ar_model, ma_model, arima_model, sarima_model, candidate_registry)
   │
   ├── script/lab02/
   │   ├── run_lab02.py                   (CLI toàn trình: pipeline + experiment_logs + 2 bộ báo cáo)
   │   ├── run_lab02_experiments.py       (CLI thực nghiệm chi tiết: model_summary + JSON + PNG)
   │   └── render_report.py               (Engine kết xuất 6 file HTML/PDF/MD cho DS1 & DS2)
   │
   └── tests/lab02/
       ├── run_all_tests.py               (Test runner discover toàn bộ test_*.py)
       ├── test_lab02_architecture.py     (7 unit tests kiểm tra logging, split, ADF, ACF/PACF, models, runner, auto-id)
       └── test_lab02_pipeline_and_reports.py (2 integration tests kiểm tra data_loader, pipeline & 6 file báo cáo)
```

## 2. Luồng Dữ liệu (End-to-End Data Flow)
1. **Nạp & Chia tuần tự:** `load_dataset()` → `split_dataset()` ghi `train_data.csv`, `test_data.csv`, `split_summary.json` vào `data/processed/lab02/{dataset1,dataset2}/`.
2. **Thực nghiệm & Kiểm toán:** `run_lab02_experiments.py` chạy ADF, ACF/PACF, huấn luyện toàn bộ ứng viên theo chế độ (`standard` / `full` / `auto`), đăng ký vào `ModelSummaryRegistry` và ghi:
   - `reports/model_summary/lab02/<dataset>/<dataset>_model_summary.{csv,json,md}` (kèm bản lưu trữ `runs/experiment_<run_id>_summary.*`).
   - `progress_results/lab02/<dataset>/latest_run.json`.
   - `logs/lab02/experiments/<dataset>/experiment_<run_id>.log`.
3. **Tổng hợp & Báo cáo:** `src/lab02/pipeline.py` tổng hợp `docs/lab02/experiment_logs.csv`, cập nhật `progress_results/lab02/latest_run.json` và gọi `script/lab02/render_report.py` xuất 2 bộ báo cáo tách riêng (`LAB_02_DATASET1_REPORT.{html,pdf,md}` và `LAB_02_DATASET2_REPORT.{html,pdf,md}`).

## 3. Nguyên tắc Thiết kế Chính
- **Zero Hardcoding:** Mọi số liệu trong báo cáo HTML/PDF/MD được đọc động từ `progress_results/lab02/<dataset>/latest_run.json`.
- **Chống Data Leakage:** Kiểm định ADF, nhận diện ACF/PACF và biến đổi Log chỉ ước lượng trên tập Train; tập Test chỉ dùng để đánh giá cuối cùng.
- **Đánh giá Đa tiêu chí Công bằng:** Phân nhóm `Scale_Group` (`original_d1`, `original_d1_D1_s12`, `log_d1_D1_s12`) khi đối chiếu AIC/BIC, đồng thời quy đổi toàn bộ dự báo Log-SARIMA qua $\exp(\cdot)$ về cùng thang đo gốc để so sánh `MAE`, `RMSE`, `MAPE (%)`, `R²`.
