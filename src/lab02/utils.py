r"""
Mô-đun tiện ích hệ thống (Utilities) cho Bài thực hành Lab 02:
Kiểm định ADF, Phân tích ACF/PACF và Dự báo ARIMA / SARIMA trên Dataset 1 & Dataset 2.
"""

from __future__ import annotations

import json
import random
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml


def get_project_root() -> Path:
    r"""
    Purpose: Xác định thư mục gốc của dự án chứa tệp config.yaml.
    Input-Output:
        - Input: Không có.
        - Output: Đối tượng Path trỏ tới thư mục gốc dự án.
    How: Duyệt ngược từ vị trí tệp src/lab02/utils.py lên các thư mục cha cho tới khi gặp config.yaml.
    """
    current_path = Path(__file__).resolve()
    for parent in [current_path.parent, current_path.parent.parent, current_path.parent.parent.parent]:
        if (parent / "config.yaml").exists():
            return parent
    return current_path.parent.parent.parent


PROJECT_ROOT: Path = get_project_root()


def _default_lab02_config() -> Dict[str, Any]:
    r"""
    Purpose: Cung cấp cấu hình mặc định dự phòng cho Lab 02 nếu config.yaml chưa có nhánh lab02.
    Input-Output:
        - Input: Không có.
        - Output: Từ điển cấu hình mặc định của Lab 02.
    How: Trả về dict chứa project, paths, split và datasets (dataset1 & dataset2).
    """
    return {
        "project": {
            "name": "Lab02_Stationarity_ADF_ARIMA_SARIMA",
            "version": "2.0.0",
            "lab_id": "lab02",
            "course": "Time Series Analysis (Phân tích Chuỗi Thời gian)",
            "institution": "University of Transport in Ho Chi Minh City (UTH)",
            "random_seed": 42,
        },
        "paths": {
            "data_processed_dir": "data/processed/lab02",
            "models_dir": "models/lab02",
            "reports_dir": "reports/lab02",
            "figures_dir": "reports/lab02/figures",
            "metrics_dir": "reports/lab02/metrics",
            "model_summary_dir": "reports/model_summary/lab02",
            "logs_dir": "logs/lab02",
            "progress_dir": "progress_results/lab02",
            "docs_dir": "docs/lab02",
            "notebooks_dir": "notebooks/lab02",
            "scripts_dir": "script/lab02",
        },
        "split": {
            "test_ratio": 0.20,
            "shuffle": False,
        },
        "datasets": {
            "dataset1": {
                "dataset_key": "dataset1",
                "dataset_title": "Monthly Retail Sales (US, 07/1991 - 06/2008)",
                "raw_path": "data/raw/Monthly Retail Sales.csv",
                "date_col": "date",
                "value_col": "value",
                "freq": "MS",
                "seasonal_period": 12,
                "report_prefix": "LAB_02_DATASET1_REPORT",
            },
            "dataset2": {
                "dataset_key": "dataset2",
                "dataset_title": "AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)",
                "raw_path": "data/raw/AirPassengers.csv",
                "date_col": "Month",
                "value_col": "Passengers",
                "freq": "MS",
                "seasonal_period": 12,
                "report_prefix": "LAB_02_DATASET2_REPORT",
            },
        },
    }


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    r"""
    Purpose: Đọc nhánh cấu hình lab02 từ tệp config.yaml trung tâm (có dự phòng mặc định).
    Input-Output:
        - Input: config_path (Optional[Path]).
        - Output: Từ điển cấu hình hoàn chỉnh của Lab 02.
    How: Nạp config.yaml bằng yaml.safe_load, trích nhánh 'lab02' và gộp vào _default_lab02_config().
    """
    default_cfg = _default_lab02_config()
    target_path = Path(config_path) if config_path is not None else (PROJECT_ROOT / "config.yaml")
    if not target_path.exists():
        return default_cfg
    with open(target_path, "r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f) or {}
    lab_cfg = full_cfg.get("lab02", {})
    for section in ("project", "paths", "split", "datasets"):
        if section in lab_cfg and isinstance(lab_cfg[section], dict):
            default_cfg[section].update(lab_cfg[section])
    return default_cfg


def get_path(key_or_relpath: str, config: Optional[Dict[str, Any]] = None) -> Path:
    r"""
    Purpose: Chuyển khóa đường dẫn trong config['paths'] thành đường dẫn tuyệt đối Path.
    Input-Output:
        - Input: key_or_relpath (str), config (Optional[Dict[str, Any]]).
        - Output: Đối tượng Path tuyệt đối.
    How: Tra cứu trong config['paths']; nếu là đường dẫn tương đối thì nối với PROJECT_ROOT.
    """
    cfg = config if config is not None else load_config()
    rel = cfg.get("paths", {}).get(key_or_relpath, key_or_relpath)
    p = Path(rel)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def ensure_directories(config: Optional[Dict[str, Any]] = None) -> Dict[str, Path]:
    r"""
    Purpose: Tạo đầy đủ các thư mục lưu trữ đầu ra của Lab 02 (gồm figures/dataset1, figures/dataset2, metrics, logs, progress, docs).
    Input-Output:
        - Input: config (Optional[Dict[str, Any]]).
        - Output: Từ điển ánh xạ tên khóa thư mục tới đối tượng Path đã tạo.
    How: Duyệt danh sách dir_keys và các thư mục con theo từng dataset rồi gọi mkdir(parents=True, exist_ok=True).
    """
    cfg = config if config is not None else load_config()
    dir_keys = [
        "data_processed_dir",
        "models_dir",
        "reports_dir",
        "figures_dir",
        "metrics_dir",
        "model_summary_dir",
        "logs_dir",
        "progress_dir",
        "docs_dir",
    ]
    created: Dict[str, Path] = {}
    for k in dir_keys:
        p = get_path(k, cfg)
        p.mkdir(parents=True, exist_ok=True)
        created[k] = p
    for ds_key in ("dataset1", "dataset2"):
        (created["figures_dir"] / ds_key).mkdir(parents=True, exist_ok=True)
        (created["data_processed_dir"] / ds_key).mkdir(parents=True, exist_ok=True)
    return created


def set_global_seed(seed: int = 42) -> None:
    r"""
    Purpose: Cố định hạt giống ngẫu nhiên cho Python và NumPy để đảm bảo khả năng tái lập.
    Input-Output:
        - Input: seed (int, mặc định 42).
        - Output: Không trả về giá trị.
    How: Gọi random.seed(seed) và np.random.seed(seed).
    """
    random.seed(seed)
    np.random.seed(seed)


def generate_run_id() -> str:
    r"""
    Purpose: Sinh mã phiên chạy duy nhất (Run ID) cho pipeline Lab 02.
    Input-Output:
        - Input: Không có.
        - Output: Chuỗi dạng 'YYYYMMDD_HHMMSS_<6_hex>'.
    How: Ghép thời gian hiện tại với 6 ký tự đầu của uuid4().hex.
    """
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def save_json(data: Any, path: Path) -> Path:
    r"""
    Purpose: Lưu dữ liệu Python xuống tệp JSON chuẩn UTF-8.
    Input-Output:
        - Input: data (Any), path (Path).
        - Output: Đường dẫn Path đã lưu.
    How: Tạo thư mục cha và ghi bằng json.dump(..., ensure_ascii=False, indent=2, default=str).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    return path


def load_json(path: Path, default: Any = None) -> Any:
    r"""
    Purpose: Đọc tệp JSON an toàn, trả về giá trị mặc định nếu tệp chưa tồn tại hoặc lỗi.
    Input-Output:
        - Input: path (Path), default (Any).
        - Output: Đối tượng Python đã giải mã hoặc giá trị default.
    How: Kiểm tra path.exists(), mở tệp đọc JSON trong khối try/except.
    """
    if not path.exists():
        return {} if default is None else default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {} if default is None else default