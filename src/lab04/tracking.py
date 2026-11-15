"""
Module quản lý theo dõi thực nghiệm và nhật ký hệ thống (`src/lab04/tracking.py`).

Trách nhiệm của module:
1. Khởi tạo phiên thực nghiệm mới với mã định danh `Run_ID` duy nhất.
2. Ghi nhận chi tiết trạng thái, dữ liệu đầu vào, đầu ra và thời gian thực thi của
   từng bước trong Pipeline vào `progress_results/lab04/progress_<run_id>.json`.
3. Ghi nhật ký hoạt động tuần tự theo dòng thời gian vào `logs/lab04/pipeline_<run_id>.log`.
4. Lưu trữ bảng lịch sử chạy thực nghiệm và 5 chỉ số đánh giá (`MSE`, `RMSE`, `MAE`, `MAPE`, `R²`)
   vào `docs/lab04/experiment_logs.csv` và `reports/lab04/metrics/experiment_history.csv`.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.lab04.utils import (
    ensure_dir,
    generate_run_id,
    get_current_timestamp,
    get_project_root,
    load_config,
    load_json,
    save_json,
)

ROOT_DIR = get_project_root()
_CFG = load_config()
_PATHS = _CFG.get("paths", {})

LOG_DIR = ROOT_DIR / _PATHS.get("logs_dir", "logs/lab04")
PROGRESS_DIR = ROOT_DIR / _PATHS.get("progress_dir", "progress_results/lab04")
METRICS_DIR = ROOT_DIR / _PATHS.get("metrics_dir", "reports/lab04/metrics")
EXPERIMENT_LOG_CSV = ROOT_DIR / _PATHS.get("experiment_log_csv", "docs/lab04/experiment_logs.csv")

ensure_dir(LOG_DIR)
ensure_dir(PROGRESS_DIR)
ensure_dir(METRICS_DIR)
ensure_dir(EXPERIMENT_LOG_CSV.parent)


def create_run(run_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    What: Khởi tạo một phiên chạy thực nghiệm mới và tạo tệp JSON theo dõi tiến độ tại `progress_results/lab04/`.
    Why: Đảm bảo mọi lần thực thi Pipeline hoặc từng module độc lập đều có hồ sơ truy vết riêng.
    Input:
        - run_id (Optional[str]): Mã `Run_ID` tùy chọn (nếu `None` sẽ tự động sinh mới).
        - metadata (Optional[Dict[str, Any]]): Thông tin phần cứng GPU/CPU, Seed và đường dẫn cấu hình.
    Output:
        - str: Chuỗi `run_id` vừa khởi tạo.
    How: Tạo cấu trúc từ điển chứa `run_id`, `start_time`, `status="RUNNING"`, `metadata` và `steps=[]`,
         sau đó ghi vào `progress_results/lab04/progress_<run_id>.json`.
    """
    active_run_id = run_id or generate_run_id()
    run_data = {
        "run_id": active_run_id,
        "project": "LAB04_Multivariate_TimeSeries",
        "lab_id": "lab04",
        "start_time": get_current_timestamp("iso"),
        "end_time": None,
        "status": "RUNNING",
        "metadata": metadata or {},
        "steps": [],
    }
    progress_path = PROGRESS_DIR / f"progress_{active_run_id}.json"
    save_json(run_data, progress_path)
    return active_run_id


def write_log(run_id: str, step_name: str, status: str, message: str) -> Path:
    """
    What: Ghi một dòng nhật ký sự kiện vào tệp `logs/lab04/pipeline_<run_id>.log`.
    Why: Cung cấp lịch sử hoạt động chi tiết theo thời gian thực để kiểm tra tiến độ và thiết bị chạy.
    Input:
        - run_id (str): Mã định danh của lần chạy hiện tại.
        - step_name (str): Tên giai đoạn (`DATA_LOADING`, `EDA`, `PREPROCESSING`, `TRAINING`, `EVALUATION`, `REPORTING`).
        - status (str): Trạng thái sự kiện (`START`, `INFO`, `SUCCESS`, `WARNING`, `FAILED`).
        - message (str): Nội dung thông điệp chi tiết.
    Output:
        - Path: Đường dẫn tới tệp log đã ghi.
    How: Mở tệp `logs/lab04/pipeline_<run_id>.log` ở chế độ nối thêm (`append`) với mã hóa UTF-8.
    """
    ensure_dir(LOG_DIR)
    log_path = LOG_DIR / f"pipeline_{run_id}.log"
    timestamp = get_current_timestamp("iso")
    with open(log_path, "a", encoding="utf-8") as file:
        file.write(f"[{timestamp}] [{status}] [{step_name}] {message}\n")
    return log_path


def update_progress(
    run_id: str,
    step_name: str,
    status: str,
    input_data: Optional[Dict[str, Any]] = None,
    output_data: Optional[Dict[str, Any]] = None,
    duration_sec: Optional[float] = None,
) -> Path:
    """
    What: Cập nhật trạng thái và thông tin đầu vào/đầu ra của một giai đoạn trong `progress_<run_id>.json`.
    Why: Giúp module `report.py` và `render_report.py` tổng hợp chính xác thời gian và kết quả từng pha.
    Input:
        - run_id (str): Mã định danh phiên chạy.
        - step_name (str): Tên giai đoạn.
        - status (str): Trạng thái (`RUNNING`, `COMPLETED`, `SKIPPED`, `FAILED`).
        - input_data (Optional[Dict[str, Any]]): Tóm tắt đầu vào của giai đoạn.
        - output_data (Optional[Dict[str, Any]]): Tóm tắt kết quả đầu ra của giai đoạn.
        - duration_sec (Optional[float]): Thời gian thực thi giai đoạn (giây).
    Output:
        - Path: Đường dẫn tệp JSON tiến độ đã cập nhật.
    How: Đọc `progress_<run_id>.json`, cập nhật mục có `step == step_name` (hoặc thêm mới) rồi ghi lại.
    """
    progress_path = PROGRESS_DIR / f"progress_{run_id}.json"
    data = load_json(progress_path, default={
        "run_id": run_id,
        "project": "LAB04_Multivariate_TimeSeries",
        "start_time": get_current_timestamp("iso"),
        "status": "RUNNING",
        "metadata": {},
        "steps": [],
    })

    steps: List[Dict[str, Any]] = data.setdefault("steps", [])
    existing = next((s for s in steps if s.get("step") == step_name), None)
    entry = {
        "step": step_name,
        "status": status,
        "timestamp": get_current_timestamp("iso"),
        "duration_sec": round(float(duration_sec), 4) if duration_sec is not None else None,
        "input": input_data or {},
        "output": output_data or {},
    }
    if existing is not None:
        existing.update(entry)
    else:
        steps.append(entry)

    save_json(data, progress_path)
    return progress_path


def finish_run(run_id: str, status: str = "COMPLETED", summary: Optional[Dict[str, Any]] = None) -> Path:
    """
    What: Đóng phiên chạy thực nghiệm và ghi nhận trạng thái hoàn tất vào `progress_<run_id>.json`.
    Why: Đánh dấu kết thúc toàn bộ Pipeline và lưu tổng thời gian thực thi.
    Input:
        - run_id (str): Mã định danh phiên chạy.
        - status (str): Trạng thái kết thúc (`COMPLETED` hoặc `FAILED`).
        - summary (Optional[Dict[str, Any]]): Tóm tắt kết quả toàn trình.
    Output:
        - Path: Đường dẫn tệp JSON tiến độ đã đóng.
    How: Cập nhật trường `end_time`, `status` và `summary` trong `progress_<run_id>.json`.
    """
    progress_path = PROGRESS_DIR / f"progress_{run_id}.json"
    data = load_json(progress_path, default={"run_id": run_id, "steps": []})
    data["end_time"] = get_current_timestamp("iso")
    data["status"] = status
    if summary is not None:
        data["summary"] = summary
    save_json(data, progress_path)
    return progress_path


def append_experiment_log(run_id: str, model_rows: List[Dict[str, Any]]) -> List[Path]:
    """
    What: Ghi nối thêm kết quả đánh giá của các mô hình vào `docs/lab04/experiment_logs.csv`
          và `reports/lab04/metrics/experiment_history.csv`.
    Why: Lưu vết lịch sử toàn bộ các lần chạy thực nghiệm để đối sánh giữa các phiên bản siêu tham số.
    Input:
        - run_id (str): Mã định danh phiên chạy.
        - model_rows (List[Dict[str, Any]]): Danh sách kết quả 5 chỉ số của từng mô hình.
    Output:
        - List[Path]: Danh sách các tệp CSV lịch sử đã được cập nhật.
    How: Ghi dòng mới với đầy đủ các cột `Run_ID`, `Timestamp`, `Task`, `Evaluation_Mode`,
         `Model`, `MSE`, `RMSE`, `MAE`, `MAPE_Pct`, `R2`, `Device`, `Training_Time_Sec`.
    """
    fieldnames = [
        "Run_ID",
        "Timestamp",
        "Task",
        "Evaluation_Mode",
        "Model",
        "MSE",
        "RMSE",
        "MAE",
        "MAPE_Pct",
        "R2",
        "Device",
        "Training_Time_Sec",
    ]
    targets = [
        EXPERIMENT_LOG_CSV,
        METRICS_DIR / "experiment_history.csv",
    ]
    now_str = get_current_timestamp("iso")
    saved: List[Path] = []

    for csv_path in targets:
        ensure_dir(csv_path.parent)
        file_exists = csv_path.exists() and csv_path.stat().st_size > 0
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for row in model_rows:
                writer.writerow({
                    "Run_ID": run_id,
                    "Timestamp": now_str,
                    "Task": row.get("Task", "Electricity_Load"),
                    "Evaluation_Mode": row.get("Evaluation_Mode", "Holdout_Test"),
                    "Model": row.get("Model", ""),
                    "MSE": round(float(row.get("MSE", 0.0)), 4),
                    "RMSE": round(float(row.get("RMSE", 0.0)), 4),
                    "MAE": round(float(row.get("MAE", 0.0)), 4),
                    "MAPE_Pct": round(float(row.get("MAPE", 0.0)), 4),
                    "R2": round(float(row.get("R2", 0.0)), 4),
                    "Device": row.get("Device", "cpu"),
                    "Training_Time_Sec": round(float(row.get("Training_Time_Sec", 0.0)), 4),
                })
        saved.append(csv_path)
    return saved