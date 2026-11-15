# HƯỚNG DẪN CÀI ĐẶT, KÍCH HOẠT GPU VÀ THỰC THI CHI TIẾT TỪNG BƯỚC (`EXECUTION_GUIDE.md`)

> **Dự án:** `LAB04_Multivariate_TimeSeries` (Phân tích Chuỗi Thời gian — UTH)  
> **Hỗ trợ Hệ điều hành:** Windows 11 (PowerShell / CMD / Git Bash — tối ưu cho laptop có GPU rời NVIDIA như HP Victus) & macOS / Linux

## Step 1: Chuẩn bị Cấu trúc Thư mục và Đặt 3 Tệp Dữ liệu vào `data/raw/`

Đảm bảo thư mục `data/raw/` trong dự án của bạn có mặt đầy đủ các tệp dữ liệu theo đúng tên cấu hình trong `config.yaml`:
1. **Cho Yêu cầu 1 đến Yêu cầu 11 (Electricity Load + Weather):**
   - Tải bộ dữ liệu từ [Electricity Load Diagrams 2011–2014 (UCI / Kaggle)](https://www.kaggle.com/datasets/jacopoferretti/electricity-load-diagrams-2011-2014-feature-engin) và đặt tại:
     ```text
     data/raw/electricity_load.csv
     ```
2. **Cho Yêu cầu 12 Nâng cao (Traffic Forecasting với DCRNN):**
   - Tải tệp tốc độ giao thông từ [PEMS-BAY Google Drive](https://drive.google.com/file/d/1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq/view?usp=drive_link) và ma trận kề khoảng cách cảm biến từ [DCRNN GitHub Repository](https://github.com/liyaguang/dcrnn), đặt tại:
     ```text
     data/raw/pems-bay.h5
     data/raw/sensor_graph/adj_mx_bay.pkl
     ```

## Step 2: Tạo và Kích hoạt Môi trường Ảo (`.venv`)

Sử dụng **Python 3.11** (phiên bản ổn định nhất cho cả `PyTorch CUDA`, `XGBoost`, `statsmodels` và `tables`/`h5py` trên Windows 11):

### Trên Windows PowerShell:
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(Nếu gặp thông báo `ExecutionPolicy`, chạy lệnh `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` rồi kích hoạt lại)*.

### Trên Windows Git Bash / macOS / Linux:
```bash
python3 -m venv .venv
source .venv/Scripts/activate   # Hoặc source .venv/bin/activate trên macOS/Linux
```

## Step 3: Cài đặt Thư viện và Kích hoạt GPU NVIDIA (`CUDA`)

Để tính năng **Tự động chuyển sang chạy bằng GPU (`Auto-GPU`)** hoạt động tối đa công suất trên card đồ họa rời NVIDIA (tránh chạy bằng CPU bị chậm đối với `LSTM` và `DCRNN`):

```powershell
python -m pip install --upgrade pip
# Cài đặt PyTorch hỗ trợ CUDA 12.1 (nếu máy có GPU NVIDIA; nếu không có GPU thì bản này vẫn chạy tốt trên CPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# Cài đặt toàn bộ các thư viện còn lại của dự án
pip install -r requirements.txt
```

Kiểm tra nhanh trạng thái nhận diện GPU:
```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU Fallback')"
```

## Step 4: Kiểm tra Nạp Dữ liệu (`src/lab04/data_loader.py`)

Chạy kiểm định nhanh tệp `electricity_load.csv`, `pems-bay.h5` và `adj_mx_bay.pkl`:
```powershell
python -m src.data_loader
```

## Step 5: Thực thi Toàn bộ Pipeline Thực nghiệm Tự động (`run_lab04.py`)

Chạy một lệnh duy nhất tại thư mục gốc để thực thi tuần tự từ **Data Loading $\rightarrow$ EDA $\rightarrow$ Preprocessing (70/15/15 + 5-Fold CV) $\rightarrow$ Training (VAR, XGBoost, LSTM, DCRNN trên GPU) $\rightarrow$ Walk-Forward & 5-Metric Evaluation $\rightarrow$ Auto Report (`.pdf`, `.html`, `.md`)**:

```powershell
python run_lab04.py
```

Các cờ dòng lệnh hỗ trợ khi bạn chỉ muốn chạy lại từng phần để tiết kiệm thời gian:
```powershell
python script/lab04/run_lab04.py --skip-eda                    # Bỏ qua bước vẽ lại biểu đồ EDA
python script/lab04/run_lab04.py --skip-kfold                  # Chỉ chạy nhánh chính 70/15/15 + Walk-Forward (bỏ qua 5-Fold CV)
python script/lab04/run_lab04.py --skip-pems                   # Chỉ chạy Yêu cầu 1-11 trên electricity_load.csv
python script/lab04/run_lab04.py --skip-train                  # Chỉ chạy lại bước Đánh giá và Xuất báo cáo từ model đã lưu
python script/lab04/run_lab04.py --bundle-docs                 # Đồng bộ toàn bộ ảnh biểu đồ sang docs/images/
```

## Step 6: Kết xuất Lại Báo cáo (`report.pdf`, `report.html`, `report.md`)

Nếu bạn chỉnh sửa giao diện hoặc muốn kết xuất lại bộ 3 báo cáo từ số liệu JSON/CSV đã chạy mà không cần huấn luyện lại mô hình:
```powershell
python script/lab04/render_report.py
```

## Step 7: Kiểm tra Hệ thống Đầu ra & Logs

Sau khi Pipeline hoàn tất, kiểm tra các tệp kết quả được sinh tự động tại:
- **Báo cáo hoàn chỉnh (3 định dạng):**
  - `reports/lab04/LAB_04_REPORT.pdf` & `docs/lab04/LAB_04_REPORT.pdf`
  - `reports/lab04/LAB_04_REPORT.html` & `docs/lab04/LAB_04_REPORT.html` (nhúng sẵn ảnh Base64, mở trực tiếp trên trình duyệt)
  - `reports/lab04/LAB_04_REPORT.md` & `docs/lab04/LAB_04_REPORT.md`
- **Số liệu 5 chỉ số (`MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²`) & K-Fold CV:**
  - `reports/lab04/metrics/evaluation_latest.json`
  - `reports/lab04/metrics/test_metrics_comparison.csv`
  - `reports/lab04/metrics/kfold_cv_results.csv`
  - `reports/lab04/metrics/pems_bay_metrics.csv`
- **Nhật ký & Tiến trình theo `Run_ID`:**
  - `logs/lab04/pipeline_<run_id>.log`
  - `progress_results/lab04/progress_<run_id>.json`
  - `docs/lab04/experiment_logs.csv`
