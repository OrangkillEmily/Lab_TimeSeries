r"""
Mô-đun tiện ích hệ thống (Utilities) cho Bài thực hành Lab 01:
Khám phá Chuỗi Thời gian, White Noise, Phân rã Chuỗi và Chất lượng Dữ liệu.
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
        - Output: Đối tượng Path trỏ tới thư mục gốc của dự án.
    How: Duyệt ngược từ vị trí tệp src/lab01/utils.py lên các thư mục cha cho tới khi gặp config.yaml.
    """
    current_path = Path(__file__).resolve()
    for parent in [current_path.parent, current_path.parent.parent, current_path.parent.parent.parent]:
        if (parent / "config.yaml").exists():
            return parent
    return current_path.parent.parent.parent


PROJECT_ROOT: Path = get_project_root()


def _default_lab01_config() -> Dict[str, Any]:
    r"""
    Purpose: Cung cấp cấu hình mặc định dự phòng cho Lab 01 nếu chưa khai báo mục lab01 trong config.yaml.
    Input-Output:
        - Input: Không có.
        - Output: Từ điển cấu hình mặc định cho Lab 01.
    How: Trả về dict chứa project, paths, data và analysis mặc định theo chuẩn đa-lab.
    """
    return {
        "project": {
            "name": "Lab01_TimeSeries_EDA_Decomposition_Quality",
            "version": "2.0.0",
            "lab_id": "lab01",
            "course": "Time Series Analysis (Phân tích Chuỗi Thời gian)",
            "institution": "University of Transport in Ho Chi Minh City (UTH)",
            "random_seed": 42,
        },
        "paths": {
            "data_raw": "data/raw/AirPassengers.csv",
            "data_processed_dir": "data/processed/lab01",
            "models_dir": "models/lab01",
            "reports_dir": "reports/lab01",
            "figures_dir": "reports/lab01/figures",
            "metrics_dir": "reports/lab01/metrics",
            "logs_dir": "logs/lab01",
            "progress_dir": "progress_results/lab01",
            "docs_dir": "docs/lab01",
            "notebooks_dir": "notebooks/lab01",
            "scripts_dir": "script/lab01",
        },
        "data": {
            "date_col": "Month",
            "value_col": "Passengers",
            "freq": "MS",
            "seasonal_period": 12,
        },
        "analysis": {
            "acf_pacf_nlags": 40,
            "white_noise_n": 1000,
            "white_noise_mean": 0.0,
            "white_noise_std": 1.0,
            "decomposition_period": 12,
            "rolling_window": 12,
            "missing_simulation_count": 8,
            "zscore_threshold": 3.0,
            "iqr_multiplier": 1.5,
        },
    }


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    r"""
    Purpose: Đọc cấu hình mục lab01 từ tệp config.yaml trung tâm (có dự phòng mặc định).
    Input-Output:
        - Input: config_path (Optional[Path], mặc định là PROJECT_ROOT / 'config.yaml').
        - Output: Từ điển cấu hình của riêng Lab 01.
    How: Đọc config.yaml bằng yaml.safe_load, lấy nhánh 'lab01' và gộp với _default_lab01_config().
    """
    default_cfg = _default_lab01_config()
    target_path = Path(config_path) if config_path is not None else (PROJECT_ROOT / "config.yaml")
    if not target_path.exists():
        return default_cfg
    with open(target_path, "r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f) or {}
    lab_cfg = full_cfg.get("lab01", {})
    for section in ("project", "paths", "data", "analysis"):
        if section in lab_cfg and isinstance(lab_cfg[section], dict):
            default_cfg[section].update(lab_cfg[section])
    return default_cfg


def get_path(key_or_relpath: str, config: Optional[Dict[str, Any]] = None) -> Path:
    r"""
    Purpose: Chuyển đổi khóa đường dẫn trong config['paths'] thành đường dẫn tuyệt đối Path.
    Input-Output:
        - Input: key_or_relpath (str), config (Optional[Dict[str, Any]]).
        - Output: Đường dẫn tuyệt đối Path.
    How: Tra cứu key trong config['paths'], nếu không có thì xem như đường dẫn tương đối từ PROJECT_ROOT.
    """
    cfg = config if config is not None else load_config()
    rel = cfg.get("paths", {}).get(key_or_relpath, key_or_relpath)
    p = Path(rel)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def ensure_directories(config: Optional[Dict[str, Any]] = None) -> Dict[str, Path]:
    r"""
    Purpose: Đảm bảo toàn bộ các thư mục lưu trữ đầu ra của Lab 01 đều đã tồn tại.
    Input-Output:
        - Input: config (Optional[Dict[str, Any]]).
        - Output: Từ điển ánh xạ tên khóa thư mục tới đối tượng Path tương ứng.
    How: Duyệt danh sách các khóa thư mục trong config['paths'] và gọi mkdir(parents=True, exist_ok=True).
    """
    cfg = config if config is not None else load_config()
    dir_keys = [
        "data_processed_dir",
        "models_dir",
        "reports_dir",
        "figures_dir",
        "metrics_dir",
        "logs_dir",
        "progress_dir",
        "docs_dir",
    ]
    created: Dict[str, Path] = {}
    for k in dir_keys:
        p = get_path(k, cfg)
        p.mkdir(parents=True, exist_ok=True)
        created[k] = p
    return created


def set_global_seed(seed: int = 42) -> None:
    r"""
    Purpose: Cố định hạt giống ngẫu nhiên cho Python và NumPy để đảm bảo khả năng tái lập 100%.
    Input-Output:
        - Input: seed (int, mặc định 42).
        - Output: Không trả về giá trị.
    How: Gọi random.seed(seed) và np.random.seed(seed).
    """
    random.seed(seed)
    np.random.seed(seed)


def generate_run_id() -> str:
    r"""
    Purpose: Sinh mã định danh duy nhất (Run ID) cho mỗi phiên chạy pipeline Lab 01.
    Input-Output:
        - Input: Không có.
        - Output: Chuỗi dạng 'YYYYMMDD_HHMMSS_<6_hex>'.
    How: Ghép dấu thời gian hiện tại với 6 ký tự đầu của uuid4().hex.
    """
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def save_json(data: Any, path: Path) -> Path:
    r"""
    Purpose: Lưu từ điển hoặc danh sách kết quả xuống tệp JSON chuẩn UTF-8.
    Input-Output:
        - Input: data (Any), path (Path).
        - Output: Đường dẫn Path đã lưu.
    How: Tạo thư mục cha nếu chưa có và gọi json.dump(..., ensure_ascii=False, indent=2, default=str).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    return path


def load_json(path: Path, default: Any = None) -> Any:
    r"""
    Purpose: Đọc dữ liệu từ tệp JSON một cách an toàn.
    Input-Output:
        - Input: path (Path), default (Any).
        - Output: Dữ liệu Python đã giải mã hoặc giá trị default nếu tệp chưa tồn tại/lỗi.
    How: Kiểm tra path.exists(), đọc bằng json.load và bắt ngoại lệ để trả về default.
    """
    if not path.exists():
        return {} if default is None else default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {} if default is None else default