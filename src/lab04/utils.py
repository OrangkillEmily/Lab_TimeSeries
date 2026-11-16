r"""
Mô-đun tiện ích hệ thống (Utilities) cho Bài thực hành Lab 04: Dự báo Chuỗi Thời gian Đa biến.
Cung cấp các hàm quản lý đường dẫn đa Lab (reports/lab04, models/lab04, logs/lab04,
progress_results/lab04, docs/lab04), nạp cấu hình YAML, cố định hạt giống ngẫu nhiên (Random Seed),
tự động phát hiện và kích hoạt GPU (CUDA / Apple MPS / CPU Fallback), và lưu trữ tuần tự hóa (.pkl, .json).
"""

import json
import os
import pickle
import platform
import random
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")


def get_project_root() -> Path:
    r"""
    Purpose: Xác định thư mục gốc của dự án (Project Root) chứa tệp config.yaml.
    Input-Output:
        - Input: Không có.
        - Output: Đối tượng Path trỏ tới thư mục gốc của dự án (ví dụ: D:\Lab_TimeSeries).
    How: Duyệt ngược từ vị trí tệp src/lab04/utils.py lên 2 cấp thư mục cha.
    """
    current_path = Path(__file__).resolve()
    for parent in [current_path.parent, current_path.parent.parent, current_path.parent.parent.parent]:
        if (parent / "config.yaml").exists():
            return parent
    return current_path.parent.parent.parent


PROJECT_ROOT: Path = get_project_root()


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    r"""
    Purpose: Đọc và phân tích cú pháp tệp cấu hình trung tâm config.yaml.
    Input-Output:
        - Input: config_path (Path tùy chọn, mặc định là PROJECT_ROOT / 'config.yaml').
        - Output: Từ điển Python (Dict[str, Any]) chứa toàn bộ thông số cấu hình.
    How: Sử dụng yaml.safe_load với bảng mã UTF-8.
    """
    target_path = Path(config_path) if config_path is not None else (PROJECT_ROOT / "config.yaml")
    if not target_path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp cấu hình tại: {target_path}")
    with open(target_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_path(relative_key_or_path: str, config: Optional[Dict[str, Any]] = None) -> Path:
    r"""
    Purpose: Chuyển đổi khóa cấu hình trong mục 'paths' hoặc đường dẫn tương đối thành Path tuyệt đối.
    Input-Output:
        - Input: relative_key_or_path (str), config (Dict tùy chọn).
        - Output: Đường dẫn tuyệt đối (Path) tính từ PROJECT_ROOT.
    How: Kiểm tra xem khóa có nằm trong config['paths'] không; nếu có thì lấy giá trị tương ứng,
         sau đó ghép nối với PROJECT_ROOT.
    """
    cfg = config if config is not None else load_config()
    paths_dict = cfg.get("paths", {})
    rel_str = paths_dict.get(relative_key_or_path, relative_key_or_path)
    full_path = PROJECT_ROOT / rel_str
    return full_path


def ensure_directories(config: Optional[Dict[str, Any]] = None) -> Dict[str, Path]:
    r"""
    Purpose: Tự động khởi tạo đầy đủ các thư mục đầu ra của Lab 04 nếu chưa tồn tại.
    Input-Output:
        - Input: config (Dict tùy chọn).
        - Output: Từ điển ánh xạ tên thư mục sang đối tượng Path đã được tạo.
    How: Tạo các thư mục data_processed_dir, models_dir, reports_dir, figures_dir,
         metrics_dir, logs_dir, progress_dir, docs_dir, notebooks_dir, scripts_dir.
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
        "notebooks_dir",
        "scripts_dir",
    ]
    resolved: Dict[str, Path] = {}
    for k in dir_keys:
        p = get_path(k, cfg)
        p.mkdir(parents=True, exist_ok=True)
        resolved[k] = p
    (resolved["docs_dir"] / "figures").mkdir(parents=True, exist_ok=True)
    return resolved


def generate_run_id() -> str:
    r"""
    Purpose: Sinh mã định danh phiên chạy duy nhất (Run ID) theo định dạng YYYYMMDD_HHMMSS_<6hex>.
    Input-Output:
        - Input: Không có.
        - Output: Chuỗi run_id (ví dụ: '20261116_154500_a1b2c3').
    How: Kết hợp thời gian hệ thống hiện tại và 6 ký tự đầu của UUID4.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uid = uuid.uuid4().hex[:6]
    return f"{ts}_{short_uid}"


def set_global_seed(seed: int = 42) -> None:
    r"""
    Purpose: Cố định hạt giống ngẫu nhiên trên toàn bộ thư viện Python, NumPy và PyTorch.
    Input-Output:
        - Input: seed (int, mặc định = 42).
        - Output: Không trả về giá trị (cố định trạng thái ngẫu nhiên toàn cục).
    How: Gán seed cho random, os.environ['PYTHONHASHSEED'], np.random, torch.manual_seed và torch.cuda.manual_seed_all.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
    except ImportError:
        pass


def get_compute_device(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    r"""
    Purpose: Tự động phát hiện phần cứng GPU (NVIDIA CUDA / Apple MPS) và cấu hình tăng tốc phần cứng.
    Input-Output:
        - Input: config (Dict tùy chọn).
        - Output: Từ điển chứa thông tin thiết bị:
            + torch_device: Chuỗi 'cuda', 'mps' hoặc 'cpu'
            + xgb_device: Chuỗi 'cuda' hoặc 'cpu'
            + device_name: Tên chi tiết phần cứng (ví dụ: 'NVIDIA GeForce RTX 3050 6GB Laptop GPU')
            + amp_enabled: Bool bật/tắt Mixed Precision FP16
            + vram_gb: Dung lượng VRAM khả dụng (GB)
    How: Kiểm tra torch.cuda.is_available() để lấy tên card đồ họa NVIDIA và dung lượng VRAM,
         cấu hình tree_method='hist' + device='cuda' cho XGBoost nếu có GPU.
    """
    cfg = config if config is not None else load_config()
    hw_cfg = cfg.get("hardware", {})
    auto_gpu = hw_cfg.get("auto_gpu", True)
    enable_amp = hw_cfg.get("enable_amp_fp16", True)

    info: Dict[str, Any] = {
        "torch_device": "cpu",
        "xgb_device": "cpu",
        "xgb_tree_method": "hist",
        "device_name": f"CPU ({platform.processor() or platform.machine()})",
        "amp_enabled": False,
        "vram_gb": 0.0,
        "cuda_available": False,
    }

    if not auto_gpu:
        return info

    try:
        import torch

        if torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram_gb = round(props.total_memory / (1024**3), 2)
            info.update(
                {
                    "torch_device": "cuda",
                    "xgb_device": "cuda",
                    "xgb_tree_method": "hist",
                    "device_name": dev_name,
                    "amp_enabled": bool(enable_amp),
                    "vram_gb": vram_gb,
                    "cuda_available": True,
                }
            )
            if hw_cfg.get("cudnn_benchmark", True):
                torch.backends.cudnn.benchmark = True
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info.update(
                {
                    "torch_device": "mps",
                    "xgb_device": "cpu",
                    "xgb_tree_method": "hist",
                    "device_name": "Apple Silicon MPS GPU",
                    "amp_enabled": False,
                    "vram_gb": 0.0,
                    "cuda_available": False,
                }
            )
    except Exception:
        pass

    return info


def save_pickle(obj: Any, filepath: Path) -> Path:
    r"""
    Purpose: Lưu tuần tự hóa đối tượng Python (DataFrame, ndarray, Scaler, Model) ra tệp nhị phân .pkl.
    Input-Output:
        - Input: obj (Any), filepath (Path).
        - Output: Đường dẫn Path đã lưu.
    How: Sử dụng pickle.dump với giao thức HIGHEST_PROTOCOL.
    """
    target = Path(filepath)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return target


def load_pickle(filepath: Path) -> Any:
    r"""
    Purpose: Nạp đối tượng Python từ tệp nhị phân .pkl.
    Input-Output:
        - Input: filepath (Path).
        - Output: Đối tượng Python đã giải tuần tự hóa.
    How: Sử dụng pickle.load; hỗ trợ tương thích ngược encoding='latin1' cho tệp .pkl từ Python 2 (như adj_mx_bay.pkl).
    """
    target = Path(filepath)
    if not target.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp pickle tại: {target}")
    try:
        with open(target, "rb") as f:
            return pickle.load(f)
    except UnicodeDecodeError:
        with open(target, "rb") as f:
            return pickle.load(f, encoding="latin1")


def save_json(data: Any, filepath: Path, indent: int = 2) -> Path:
    r"""
    Purpose: Lưu cấu trúc dữ liệu từ điển hoặc danh sách ra tệp JSON chuẩn UTF-8.
    Input-Output:
        - Input: data (Dict/List), filepath (Path), indent (int).
        - Output: Đường dẫn Path của tệp JSON đã lưu.
    How: Chuyển đổi tự động các kiểu dữ liệu NumPy (int64, float32, ndarray) sang kiểu Python nguyên thủy trước khi ghi.
    """
    def _default_serializer(o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            val = float(o)
            return None if (np.isnan(val) or np.isinf(val)) else val
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (Path,)):
            return str(o)
        return str(o)

    target = Path(filepath)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent, default=_default_serializer)
    return target


def load_json(filepath: Path, default: Optional[Any] = None) -> Any:
    r"""
    Purpose: Đọc dữ liệu từ tệp JSON.
    Input-Output:
        - Input: filepath (Path), default (giá trị trả về nếu tệp chưa tồn tại).
        - Output: Từ điển hoặc danh sách đọc từ JSON.
    How: Sử dụng json.load với bảng mã UTF-8.
    """
    target = Path(filepath)
    if not target.exists():
        if default is not None:
            return default
        raise FileNotFoundError(f"Không tìm thấy tệp JSON tại: {target}")
    with open(target, "r", encoding="utf-8") as f:
        return json.load(f)
