r"""
Mô-đun theo dõi tiến độ thực nghiệm và ghi nhật ký (Tracking & Logging) cho Bài thực hành Lab 04.
Quản lý tập trung các tệp:
- logs/lab04/pipeline_<run_id>.log
- logs/lab04/training_structured.jsonl
- progress_results/lab04/run_summary_<run_id>.json
- progress_results/lab04/latest_run.json
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.lab04.utils import (
    ensure_directories,
    generate_run_id,
    get_compute_device,
    get_path,
    load_config,
    load_json,
    save_json,
)


def init_run_session(run_id: Optional[str] = None) -> str:
    r"""
    Purpose: Khởi tạo phiên thực thi mới (hoặc tiếp tục phiên hiện tại) và ghi nhận cấu hình phần cứng vào latest_run.json.
    Input-Output:
        - Input: run_id (Chuỗi định danh tùy chọn; nếu None sẽ tự sinh mã mới).
        - Output: Chuỗi active_run_id.
    How: Tạo các thư mục lưu trữ, phát hiện GPU/CPU, ghi tệp progress_results/lab04/latest_run.json và run_summary_<run_id>.json.
    """
    cfg = load_config()
    ensure_directories(cfg)
    active_run_id = run_id if run_id is not None else generate_run_id()
    hw_info = get_compute_device(cfg)

    progress_dir = get_path("progress_dir", cfg)
    latest_path = progress_dir / "latest_run.json"
    summary_path = progress_dir / f"run_summary_{active_run_id}.json"

    existing = load_json(summary_path, default={})
    now_iso = datetime.now().isoformat(timespec="seconds")

    payload: Dict[str, Any] = {
        "run_id": active_run_id,
        "project": cfg.get("project", {}),
        "hardware": hw_info,
        "started_at": existing.get("started_at", now_iso),
        "updated_at": now_iso,
        "status": existing.get("status", "IN_PROGRESS"),
        "stages": existing.get("stages", {}),
    }

    save_json(payload, summary_path)
    save_json(payload, latest_path)
    return active_run_id


def get_active_run_id(run_id: Optional[str] = None) -> str:
    r"""
    Purpose: Lấy mã phiên chạy hiện hành từ tham số truyền vào hoặc từ tệp latest_run.json.
    Input-Output:
        - Input: run_id (Optional[str]).
        - Output: Chuỗi active_run_id hợp lệ.
    How: Nếu run_id khác None thì trả về ngay; ngược lại đọc progress_results/lab04/latest_run.json,
         nếu chưa tồn tại thì gọi init_run_session().
    """
    if run_id is not None and str(run_id).strip():
        return str(run_id).strip()
    cfg = load_config()
    latest_path = get_path("progress_dir", cfg) / "latest_run.json"
    if latest_path.exists():
        data = load_json(latest_path, default={})
        if data.get("run_id"):
            return str(data["run_id"])
    return init_run_session()


def write_log(
    run_id: str,
    stage: str,
    status: str,
    message: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    r"""
    Purpose: Ghi nhật ký đồng thời ra tệp văn bản pipeline_<run_id>.log và tệp cấu trúc training_structured.jsonl.
    Input-Output:
        - Input: run_id (str), stage (str), status (str), message (str), extra (Dict tùy chọn).
        - Output: Không trả về giá trị.
    How: Định dạng dòng log chuẩn '[YYYY-MM-DD HH:MM:SS] [STAGE] [STATUS] Message' và append vào cả 2 tệp log.
    """
    cfg = load_config()
    logs_dir = get_path("logs_dir", cfg)
    logs_dir.mkdir(parents=True, exist_ok=True)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now_str}] [{stage}] [{status}] {message}\n"

    log_file = logs_dir / f"pipeline_{run_id}.log"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line)

    jsonl_file = logs_dir / "training_structured.jsonl"
    record: Dict[str, Any] = {
        "timestamp": now_str,
        "run_id": run_id,
        "stage": stage,
        "status": status,
        "message": message,
    }
    if extra:
        record["extra"] = extra
    with open(jsonl_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def record_stage_result(
    run_id: str,
    stage_name: str,
    payload: Dict[str, Any],
    mark_completed: bool = False,
) -> Path:
    r"""
    Purpose: Cập nhật kết quả đầu ra của từng giai đoạn (EDA, Preprocessing, Training, Evaluation, Visualization, Report)
             vào tệp run_summary_<run_id>.json và latest_run.json.
    Input-Output:
        - Input: run_id (str), stage_name (str), payload (Dict[str, Any]), mark_completed (bool).
        - Output: Đường dẫn Path tới tệp run_summary_<run_id>.json.
    How: Đọc tệp JSON hiện tại, cập nhật khóa stages[stage_name] = payload, ghi lại xuống đĩa.
    """
    cfg = load_config()
    progress_dir = get_path("progress_dir", cfg)
    progress_dir.mkdir(parents=True, exist_ok=True)

    summary_path = progress_dir / f"run_summary_{run_id}.json"
    latest_path = progress_dir / "latest_run.json"

    current = load_json(summary_path, default={})
    if not current:
        init_run_session(run_id)
        current = load_json(summary_path, default={})

    stages = current.get("stages", {})
    stages[stage_name] = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "data": payload,
    }
    current["stages"] = stages
    current["updated_at"] = datetime.now().isoformat(timespec="seconds")
    if mark_completed:
        current["status"] = "COMPLETED"

    save_json(current, summary_path)
    save_json(current, latest_path)
    return summary_path
