# HƯỚNG DẪN VẬN HÀNH LAB 01 (`EXECUTION_GUIDE.md`)

Tất cả lệnh dưới đây được thực thi từ **thư mục gốc của dự án** (thư mục chứa `config.yaml`, ví dụ `D:\Lab_TimeSeries`) sau khi đã kích hoạt môi trường ảo `.venv`.

## 1. Chạy toàn trình Pipeline Lab 01 (Khuyến nghị)
Chạy trọn vẹn 6 giai đoạn (Data Loading → EDA/ACF/PACF/White Noise → Decomposition → Data Quality → Visualization → Report HTML/PDF/MD):

```powershell
python script/lab01/run_lab01.py
```

Nếu muốn tự động sao chép đồng bộ 15 biểu đồ từ `reports/lab01/figures/` sang `docs/lab01/images/`:

```powershell
python script/lab01/run_lab01.py --bundle-docs
```

## 2. Kết xuất lại riêng Báo cáo (không cần chạy lại tính toán)
Sau khi đã có `progress_results/lab01/latest_run.json` và 15 biểu đồ trong `reports/lab01/figures/`:

```powershell
python script/lab01/render_report.py
```

## 3. Chạy từng mô-đun độc lập (khi cần debug theo giai đoạn)
```powershell
python -m src.lab01.data_loader
python -m src.lab01.eda
python -m src.lab01.decomposition
python -m src.lab01.quality
python -m src.lab01.visualize
python -m src.lab01.report
```

## 4. Chạy Notebook thực hành tương tác
Mở `script/lab01/Lab01_TimeSeries.ipynb` bằng VS Code hoặc Jupyter và chọn **Run All**. Notebook đã tích hợp tự động định vị `PROJECT_ROOT`, nạp hàm từ `src.lab01.*` và khôi phục backend `%matplotlib inline` để hiển thị trực tiếp mọi biểu đồ.

## 5. Danh sách đầu ra cần kiểm tra sau khi chạy
1. **15 biểu đồ PNG:** `reports/lab01/figures/01_*.png` → `15_*.png`.
2. **Dữ liệu đã xử lý:** `data/processed/lab01/airpassengers_clean.csv` và `airpassengers_quality_cleaned.csv`.
3. **JSON tiến độ & chỉ số:** `progress_results/lab01/latest_run.json` (`status: "COMPLETED"`) và `reports/lab01/metrics/*.json`.
4. **Bộ 3 báo cáo nộp bài:** `reports/lab01/LAB_01_REPORT.html`, `LAB_01_REPORT.pdf`, `LAB_01_REPORT.md`.
