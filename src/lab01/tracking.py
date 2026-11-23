r"""
Mô-đun theo dõi tiến độ và ghi nhật ký thực thi (Tracking & Logging) cho Lab 01.
Quản lý đồng bộ:
- logs/lab01/pipeline_<run_id>.log
- logs/lab01/pipeline_structured.jsonl
- progress_results/lab01/run_summary_<run_id>.json
- progress_results/lab01/latest_run.json
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.lab01.utils import (
    ensure_directories,
    generate_run_id,
    get_path,
    load_config,
    load_json,
    save_json,
)


def init_run_session(run_id: Optional[str] = None) -> str:
    r"""
    Purpose: Khởi tạo phiên chạy mới cho Lab 01 và ghi nhận thông tin dự án vào latest_run.json.
    Input-Output:
        - Input: run_id (Optional[str]).
        - Output: Chuỗi active_run_id.
    How: Tạo các thư mục cần thiết và ghi payload khởi tạo vào run_summary_<run_id>.json và latest_run.json.
    """
    cfg = load_config()
    ensure_directories(cfg)
    active_run_id = run_id if run_id is not None else generate_run_id()

    progress_dir = get_path("progress_dir", cfg)
    latest_path = progress_dir / "latest_run.json"
    summary_path = progress_dir / f"run_summary_{active_run_id}.json"

    existing = load_json(summary_path, default={})
    now_iso = datetime.now().isoformat(timespec="seconds")

    payload: Dict[str, Any] = {
        "run_id": active_run_id,
        "project": cfg.get("project", {}),
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
    Purpose: Lấy mã phiên chạy hiện hành từ tham số truyền vào hoặc từ progress_results/lab01/latest_run.json.
    Input-Output:
        - Input: run_id (Optional[str]).
        - Output: Chuỗi active_run_id hợp lệ.
    How: Ưu tiên run_id truyền vào; nếu None thì đọc latest_run.json, chưa có thì khởi tạo mới.
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
    Purpose: Ghi nhật ký đồng thời ra tệp văn bản pipeline_<run_id>.log và pipeline_structured.jsonl.
    Input-Output:
        - Input: run_id (str), stage (str), status (str), message (str), extra (Optional[Dict]).
        - Output: Không trả về giá trị.
    How: Định dạng dòng log '[YYYY-MM-DD HH:MM:SS] [STAGE] [STATUS] Message' và ghi nối tiếp vào 2 tệp.
    """
    cfg = load_config()
    logs_dir = get_path("logs_dir", cfg)
    logs_dir.mkdir(parents=True, exist_ok=True)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{now_str}] [{stage}] [{status}] {message}\n"

    with open(logs_dir / f"pipeline_{run_id}.log", "a", encoding="utf-8") as f:
        f.write(line)

    record: Dict[str, Any] = {
        "timestamp": now_str,
        "run_id": run_id,
        "stage": stage,
        "status": status,
        "message": message,
    }
    if extra:
        record["extra"] = extra
    with open(logs_dir / "pipeline_structured.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def record_stage_result(
    run_id: str,
    stage_name: str,
    payload: Dict[str, Any],
    mark_completed: bool = False,
) -> Path:
    r"""
    Purpose: Lưu kết quả của từng giai đoạn Lab 01 vào run_summary_<run_id>.json và latest_run.json.
    Input-Output:
        - Input: run_id (str), stage_name (str), payload (Dict[str, Any]), mark_completed (bool).
        - Output: Đường dẫn Path tới run_summary_<run_id>.json.
    How: Đọc JSON hiện tại, cập nhật stages[stage_name] kèm dấu thời gian và ghi đè lại cả 2 tệp.
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