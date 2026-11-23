r"""
Mô-đun theo dõi tiến độ thực nghiệm và ghi nhật ký tổng hợp (Tracking & Logging) cho Lab 02.
Quản lý đồng bộ với hệ thống log chi tiết của lab02_utils:
- logs/lab02/pipeline_<run_id>.log
- logs/lab02/pipeline_structured.jsonl
- progress_results/lab02/run_summary_<run_id>.json
- progress_results/lab02/latest_run.json
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.lab02.utils import (
    ensure_directories,
    generate_run_id,
    get_path,
    load_config,
    load_json,
    save_json,
)


def init_run_session(run_id: Optional[str] = None) -> str:
    r"""
    Purpose: Khởi tạo phiên thực thi mới cho Lab 02 và lưu thông tin ban đầu vào latest_run.json.
    Input-Output:
        - Input: run_id (Optional[str]).
        - Output: Chuỗi active_run_id.
    How: Tạo thư mục cần thiết và ghi payload khởi tạo vào run_summary_<run_id>.json và latest_run.json.
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
    Purpose: Lấy mã phiên chạy hiện hành từ tham số truyền vào hoặc từ progress_results/lab02/latest_run.json.
    Input-Output:
        - Input: run_id (Optional[str]).
        - Output: Chuỗi active_run_id hợp lệ.
    How: Ưu tiên run_id truyền vào; nếu None thì đọc latest_run.json, chưa có thì gọi init_run_session().
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
    Purpose: Ghi nhật ký đồng thời ra logs/lab02/pipeline_<run_id>.log và pipeline_structured.jsonl.
    Input-Output:
        - Input: run_id (str), stage (str), status (str), message (str), extra (Optional[Dict]).
        - Output: Không trả về giá trị.
    How: Định dạng dòng log '[YYYY-MM-DD HH:MM:SS] [STAGE] [STATUS] Message' và append vào cả 2 tệp log.
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
    Purpose: Ghi nhận kết quả từng giai đoạn hoặc từng dataset của Lab 02 vào run_summary_<run_id>.json và latest_run.json.
    Input-Output:
        - Input: run_id (str), stage_name (str), payload (Dict[str, Any]), mark_completed (bool).
        - Output: Đường dẫn Path tới tệp run_summary_<run_id>.json.
    How: Đọc JSON hiện tại, cập nhật khóa stages[stage_name] kèm timestamp và lưu đè lại cả 2 tệp.
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