# HƯỚNG DẪN VẬN HÀNH LAB 02 (`EXECUTION_GUIDE.md`)

Tất cả lệnh bên dưới được chạy từ **thư mục gốc của dự án** (thư mục chứa `config.yaml`, ví dụ `D:\Lab_TimeSeries`) sau khi kích hoạt môi trường ảo `.venv`.

## 1. Chạy toàn trình Pipeline Lab 02 (Khuyến nghị)
Chạy đồng thời cả 2 dataset ở chế độ đầy đủ (`full`), xuất `model_summary`, `progress_results`, `docs/lab02/experiment_logs.csv` và cả 6 file báo cáo HTML/PDF/MD:

```powershell
python script/lab02/run_lab02.py
```

Tùy chọn nâng cao:
```powershell
# Chạy nhanh chế độ standard (ít ứng viên hơn để kiểm tra nhanh)
python script/lab02/run_lab02.py --dataset all --mode standard

# Chạy riêng 1 dataset kèm đồng bộ biểu đồ sang docs/lab02/images/
python script/lab02/run_lab02.py --dataset dataset1 --mode full --bundle-docs
python script/lab02/run_lab02.py --dataset dataset2 --mode full --bundle-docs

# Chế độ tự động nhận diện ứng viên 2 không gian (Original + Log)
python script/lab02/run_lab02.py --dataset all --mode auto
```

## 2. Chạy riêng thực nghiệm hoặc kết xuất lại Báo cáo
```powershell
# Chỉ chạy thực nghiệm & xuất model_summary + progress_results
python script/lab02/run_lab02_experiments.py --dataset all --mode full

# Chỉ kết xuất lại 2 bộ báo cáo từ latest_run.json hiện có
python script/lab02/render_report.py --dataset all
```

## 3. Chạy bộ Kiểm thử Tự động (9 Unit & Integration Tests)
```powershell
# Chạy toàn bộ test Lab 02 (7 test kiến trúc + 2 test pipeline/báo cáo)
python tests/lab02/run_all_tests.py

# Hoặc chạy riêng từng tệp test
python tests/lab02/test_lab02_architecture.py
python tests/lab02/test_lab02_pipeline_and_reports.py
```

## 4. Danh sách đầu ra cần kiểm tra sau khi chạy
1. **Model Summary nộp giảng viên:**
   - `reports/model_summary/lab02/dataset1/dataset1_model_summary.{csv,json,md}` (`27` mô hình ở mode `full`).
   - `reports/model_summary/lab02/dataset2/dataset2_model_summary.{csv,json,md}` (`54` mô hình ở mode `full`).
2. **Dữ liệu tiến độ JSON:** `progress_results/lab02/latest_run.json`, `progress_results/lab02/dataset1/latest_run.json`, `progress_results/lab02/dataset2/latest_run.json`.
3. **Bảng gộp thực nghiệm:** `docs/lab02/experiment_logs.csv`.
4. **Bộ 6 tệp báo cáo học thuật:**
   - `reports/lab02/LAB_02_DATASET1_REPORT.{html,pdf,md}`
   - `reports/lab02/LAB_02_DATASET2_REPORT.{html,pdf,md}`
