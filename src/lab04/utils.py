"""
Module tiện ích hệ thống và điều phối phần cứng Auto-GPU (`src/lab04/utils.py`).

Trách nhiệm của module:
1. Xác định thư mục gốc của kho lưu trữ đa Lab (Multi-Lab Repository Root) và chuẩn hóa
   các đường dẫn con của Lab 04 (`reports/lab04`, `models/lab04`, `logs/lab04`,
   `progress_results/lab04`, `docs/lab04`, `data/raw/sensor_graph/adj_mx_bay.pkl`).
2. Tự động nhận diện và chuyển đổi thiết bị tính toán sang GPU (`NVIDIA CUDA` / `Apple MPS`)
   kèm `Automatic Mixed Precision (AMP FP16)` cho các mô hình nặng (`LSTM`, `DCRNN`, `XGBoost`),
   và tự động lùi về CPU đa luồng nếu máy không có GPU.
3. Thiết lập hạt giống ngẫu nhiên toàn cục (`random_state = 42`) đảm bảo khả năng tái lập.
4. Cung cấp các hàm đọc/ghi cấu hình YAML, JSON, Pickle/Joblib an toàn.
"""

from __future__ import annotations

import gc
import json
import os
import pickle
import random
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import yaml

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False


def get_project_root() -> Path:
    """
    What: Xác định đường dẫn tuyệt đối tới thư mục gốc của dự án đa Lab.
    Why: Đảm bảo dù người dùng chạy lệnh từ thư mục gốc (`python script/lab04/run_lab04.py`),
         từ bên trong `script/lab04/` hay từ `notebooks/lab04/`, mọi đường dẫn dữ liệu và
         báo cáo đều trỏ về đúng gốc dự án.
    Input: Không có.
    Output:
        - Path: Đường dẫn thư mục gốc chứa `src/`, `data/`, `reports/`, `config.yaml`.
    How: Đi ngược từ vị trí tệp `src/lab04/utils.py` lên 2 cấp (`.parent.parent.parent`)
         và kiểm tra sự tồn tại của `config.yaml` hoặc thư mục `src/`.
    """
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent.parent,
        here.parent.parent,
        Path.cwd(),
    ]
    for cand in candidates:
        if (cand / "config.yaml").exists() or (cand / "src" / "lab04").exists():
            return cand
    return here.parent.parent.parent


ROOT_DIR = get_project_root()


def ensure_dir(directory_path: Union[str, Path]) -> Path:
    """
    What: Kiểm tra và tự động khởi tạo thư mục (cùng các thư mục cha) nếu chưa tồn tại.
    Why: Tránh lỗi `FileNotFoundError` khi ghi mô hình, biểu đồ, báo cáo hoặc nhật ký.
    Input:
        - directory_path (Union[str, Path]): Đường dẫn thư mục cần tạo.
    Output:
        - Path: Đối tượng `Path` của thư mục đã tồn tại.
    How: Chuyển đầu vào thành `Path` và gọi `mkdir(parents=True, exist_ok=True)`.
    """
    path_obj = Path(directory_path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def configure_console_utf8() -> None:
    """
    What: Cấu hình luồng xuất chuẩn `stdout` và `stderr` sang bảng mã UTF-8.
    Why: Ngăn ngừa lỗi `UnicodeEncodeError` khi in tiếng Việt hoặc ký hiệu toán học trên
         Windows PowerShell / Command Prompt.
    Input: Không có.
    Output: Không có.
    How: Gọi `reconfigure(encoding='utf-8')` trên `sys.stdout` và `sys.stderr` nếu hỗ trợ.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


def get_current_timestamp(fmt: str = "iso") -> str:
    """
    What: Sinh chuỗi thời gian hiện tại theo định dạng chuẩn.
    Why: Đóng dấu thời gian cho tệp nhật ký (`logs/lab04/`), tiến độ (`progress_results/lab04/`)
         và mã định danh phiên chạy `Run_ID`.
    Input:
        - fmt (str): `"iso"` (ISO-8601), `"compact"` (`YYYYMMDD_HHMMSS`), hoặc `"readable"`.
    Output:
        - str: Chuỗi thời gian đã định dạng.
    How: Lấy `datetime.now().astimezone()` và định dạng theo tham số `fmt`.
    """
    now = datetime.now().astimezone()
    if fmt == "compact":
        return now.strftime("%Y%m%d_%H%M%S")
    if fmt == "readable":
        return now.strftime("%Y-%m-%d %H:%M:%S")
    return now.isoformat(timespec="seconds")


def generate_run_id() -> str:
    """
    What: Sinh mã định danh `Run_ID` duy nhất cho mỗi lần thực thi Pipeline.
    Why: Giúp lưu trữ lịch sử biểu đồ, nhật ký và báo cáo theo từng phiên chạy độc lập
         mà không ghi đè mất kết quả của các lần thử nghiệm trước.
    Input: Không có.
    Output:
        - str: Chuỗi định dạng `YYYYMMDD_HHMMSS_<6_ky_tu_hex>`.
    How: Kết hợp dấu thời gian nén (`compact`) với 6 ký tự đầu của `uuid.uuid4().hex`.
    """
    return f"{get_current_timestamp('compact')}_{uuid.uuid4().hex[:6]}"


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    What: Đọc tệp cấu hình YAML trung tâm của dự án và chuẩn hóa các đường dẫn của Lab 04.
    Why: Cung cấp nguồn cấu hình thống nhất cho toàn bộ các module trong `src/lab04/`.
    Input:
        - config_path (Optional[Union[str, Path]]): Đường dẫn tùy chọn tới `config.yaml`.
    Output:
        - Dict[str, Any]: Từ điển cấu hình đã nạp.
    How: Tìm `config.yaml` tại thư mục gốc (hoặc `agents/lab04/config.yaml`), đọc bằng
         `yaml.safe_load` và bổ sung các giá trị mặc định an toàn cho cấu trúc đa Lab.
    """
    root = get_project_root()
    search_paths = []
    if config_path is not None:
        search_paths.append(Path(config_path))
    search_paths.extend([
        root / "config.yaml",
        root / "agents" / "lab04" / "config.yaml",
        root / "src" / "lab04" / "config.yaml",
    ])
    for path in search_paths:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
                return _normalize_multilab_config(cfg)
    return _normalize_multilab_config({})


def _normalize_multilab_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    What: Chuẩn hóa các đường dẫn trong từ điển cấu hình về đúng cấu trúc đa Lab (`.../lab04/`).
    Why: Ngay cả khi tệp `config.yaml` của người dùng còn giữ đường dẫn cũ (như `reports/`
         hoặc `data/raw/adj_mx_bay.pkl`), hàm này vẫn đảm bảo hệ thống ưu tiên đúng cấu trúc
         `reports/lab04/`, `models/lab04/`, `logs/lab04/`, `docs/lab04/` và `data/raw/sensor_graph/adj_mx_bay.pkl`.
    Input:
        - cfg (Dict[str, Any]): Từ điển cấu hình thô.
    Output:
        - Dict[str, Any]: Từ điển cấu hình đã chuẩn hóa đường dẫn Lab 04.
    How: Kiểm tra và điền các khóa mặc định trong `paths`, `electricity_data` và `pems_bay_advanced`.
    """
    paths = cfg.setdefault("paths", {})
    default_paths = {
        "agents_dir": "agents/lab04",
        "docs_dir": "docs/lab04",
        "src_dir": "src/lab04",
        "script_dir": "script/lab04",
        "notebooks_dir": "notebooks/lab04",
        "models_dir": "models/lab04",
        "logs_dir": "logs/lab04",
        "progress_dir": "progress_results/lab04",
        "reports_dir": "reports/lab04",
        "figures_dir": "reports/lab04/figures",
        "metrics_dir": "reports/lab04/metrics",
        "predictions_dir": "reports/lab04/predictions",
        "experiment_log_csv": "docs/lab04/experiment_logs.csv",
        "report_md": "reports/lab04/LAB_04_REPORT.md",
        "report_html": "reports/lab04/LAB_04_REPORT.html",
        "report_pdf": "reports/lab04/LAB_04_REPORT.pdf",
        "docs_report_md": "docs/lab04/LAB_04_REPORT.md",
        "docs_report_html": "docs/lab04/LAB_04_REPORT.html",
        "docs_report_pdf": "docs/lab04/LAB_04_REPORT.pdf",
    }
    for key, val in default_paths.items():
        if key not in paths or not str(paths[key]).strip():
            paths[key] = val
        elif "lab04" not in str(paths[key]) and key in default_paths:
            # Tự động ánh xạ sang thư mục con lab04 theo yêu cầu đa Lab
            paths[key] = val

    elec = cfg.setdefault("electricity_data", {})
    elec.setdefault("raw_path", "data/raw/electricity_load.csv")
    if "lab04" not in str(elec.get("processed_dir", "")):
        elec["processed_dir"] = "data/processed/lab04/electricity"

    pems = cfg.setdefault("pems_bay_advanced", {})
    pems.setdefault("h5_path", "data/raw/pems-bay.h5")
    pems["adj_mx_path"] = "data/raw/sensor_graph/adj_mx_bay.pkl"
    if "lab04" not in str(pems.get("processed_dir", "")):
        pems["processed_dir"] = "data/processed/lab04/pems_bay"

    return cfg


def set_global_seeds(seed: int = 42) -> Dict[str, Any]:
    """
    What: Cố định hạt giống ngẫu nhiên cho Python, NumPy, PyTorch (CPU & CUDA) và TensorFlow.
    Why: Đảm bảo kết quả chia dữ liệu, khởi tạo trọng số mạng LSTM / DCRNN và XGBoost
         hoàn toàn có thể tái lập giữa các lần chạy (Quy tắc 8).
    Input:
        - seed (int): Giá trị hạt giống ngẫu nhiên (mặc định `42`).
    Output:
        - Dict[str, Any]: Thông tin trạng thái các thư viện đã được cố định seed.
    How: Đặt `PYTHONHASHSEED`, `random.seed`, `np.random.seed`, `torch.manual_seed`,
         `torch.cuda.manual_seed_all` và `tf.random.set_seed`.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    info: Dict[str, Any] = {"seed": int(seed), "python": True, "numpy": True, "torch": False, "tensorflow": False}
    if HAS_TORCH:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            info["torch"] = True
        except Exception:
            pass
    if HAS_TENSORFLOW:
        try:
            tf.random.set_seed(seed)
            info["tensorflow"] = True
        except Exception:
            pass
    return info


def detect_hardware_accelerator(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    What: Tự động dò tìm và kích hoạt phần cứng tăng tốc GPU (NVIDIA CUDA / Apple MPS) hoặc CPU.
    Why: Đáp ứng Yêu cầu bổ sung số 5 — tự động chuyển sang chạy bằng GPU nếu máy có GPU
         (vì mô hình Deep Learning `LSTM` và `DCRNN` rất nặng khi chạy trên CPU), đồng thời
         tự động fallback an toàn về CPU đa luồng nếu máy không có GPU.
    Input:
        - config (Optional[Dict[str, Any]]): Cấu hình từ `config.yaml`.
    Output:
        - Dict[str, Any]: Thông tin chi tiết thiết bị gồm `device_type` (`"cuda"`, `"mps"`, `"cpu"`),
          `torch_device`, `device_name`, `vram_gb`, `amp_enabled`, `xgb_device`, `xgb_tree_method`.
    How: Kiểm tra `torch.cuda.is_available()`, `torch.backends.mps.is_available()` và
         `tf.config.list_physical_devices('GPU')`, đồng thời bật `cudnn.benchmark = True` khi có CUDA.
    """
    cfg = config or load_config()
    hw_cfg = cfg.get("hardware", {})
    requested = str(hw_cfg.get("device", "auto")).lower()
    use_amp = bool(hw_cfg.get("use_mixed_precision", True))
    cudnn_bench = bool(hw_cfg.get("cudnn_benchmark", True))

    device_type = "cpu"
    device_name = "Multi-Core CPU"
    vram_gb = 0.0
    amp_enabled = False
    xgb_device = "cpu"

    if HAS_TORCH and requested in ("auto", "cuda", "gpu"):
        if torch.cuda.is_available():
            device_type = "cuda"
            device_name = torch.cuda.get_device_name(0)
            try:
                props = torch.cuda.get_device_properties(0)
                vram_gb = round(props.total_memory / (1024 ** 3), 2)
            except Exception:
                vram_gb = 0.0
            if cudnn_bench:
                torch.backends.cudnn.benchmark = True
            amp_enabled = use_amp
            xgb_device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_type = "mps"
            device_name = "Apple Silicon Metal (MPS)"
            amp_enabled = False
            xgb_device = "cpu"

    if device_type == "cpu" and HAS_TENSORFLOW and requested in ("auto", "cuda", "gpu"):
        try:
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                device_type = "tf_gpu"
                device_name = f"TensorFlow GPU ({gpus[0].name})"
        except Exception:
            pass

    return {
        "device_type": device_type,
        "torch_device": "cuda" if device_type == "cuda" else ("mps" if device_type == "mps" else "cpu"),
        "device_name": device_name,
        "vram_gb": vram_gb,
        "amp_enabled": amp_enabled,
        "xgb_device": xgb_device,
        "xgb_tree_method": "hist",
        "has_torch": HAS_TORCH,
        "has_tensorflow": HAS_TENSORFLOW,
    }


def clear_hardware_memory() -> None:
    """
    What: Giải phóng bộ nhớ RAM và bộ nhớ đệm GPU VRAM.
    Why: Ngăn chặn lỗi tràn bộ nhớ (`CUDA Out-Of-Memory`) khi huấn luyện lặp qua 5 Folds
         của `TimeSeriesSplit` hoặc khi chuyển từ `LSTM` sang `DCRNN`.
    Input: Không có.
    Output: Không có.
    How: Gọi `gc.collect()`, `torch.cuda.empty_cache()` và `tf.keras.backend.clear_session()`.
    """
    gc.collect()
    if HAS_TORCH and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    if HAS_TENSORFLOW:
        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass


def save_json(data: Any, file_path: Union[str, Path]) -> Path:
    """
    What: Lưu cấu trúc dữ liệu Python sang tệp JSON với mã hóa UTF-8.
    Why: Phục vụ lưu trữ kết quả đánh giá `evaluation_latest.json`, tiến trình `progress_<run_id>.json`
         và metadata huấn luyện.
    Input:
        - data (Any): Dữ liệu kiểu Dict/List cần lưu.
        - file_path (Union[str, Path]): Đường dẫn tệp JSON đích.
    Output:
        - Path: Đường dẫn tệp đã lưu.
    How: Tự động chuyển đổi các kiểu số NumPy (`np.integer`, `np.floating`, `np.ndarray`)
         sang kiểu Python chuẩn trước khi ghi bằng `json.dump(..., indent=2, ensure_ascii=False)`.
    """
    path_obj = Path(file_path)
    ensure_dir(path_obj.parent)

    def _default_serializer(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            val = float(obj)
            return None if (np.isnan(val) or np.isinf(val)) else val
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return str(obj)

    with open(path_obj, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_default_serializer)
    return path_obj


def load_json(file_path: Union[str, Path], default: Optional[Any] = None) -> Any:
    """
    What: Đọc dữ liệu từ tệp JSON.
    Why: Tải thông tin đánh giá, tiến độ và cấu hình thực nghiệm giữa các module.
    Input:
        - file_path (Union[str, Path]): Đường dẫn tệp JSON cần đọc.
        - default (Optional[Any]): Giá trị trả về mặc định nếu tệp không tồn tại.
    Output:
        - Any: Dữ liệu đã giải mã từ JSON.
    How: Mở tệp với `encoding="utf-8"` và gọi `json.load()`.
    """
    path_obj = Path(file_path)
    if not path_obj.exists():
        if default is not None:
            return default
        raise FileNotFoundError(f"Không tìm thấy tệp JSON: {path_obj}")
    with open(path_obj, "r", encoding="utf-8") as f:
        return json.load(f)


def save_object(obj: Any, file_path: Union[str, Path]) -> Path:
    """
    What: Tuần tự hóa (serialize) đối tượng Python/Model ra tệp `.pkl` hoặc `.joblib`.
    Why: Lưu trữ bộ chuẩn hóa `StandardScaler`, mô hình `VAR` và `XGBoost` trong `models/lab04/`.
    Input:
        - obj (Any): Đối tượng cần lưu.
        - file_path (Union[str, Path]): Đường dẫn tệp đích.
    Output:
        - Path: Đường dẫn tệp đã lưu.
    How: Ưu tiên dùng `joblib.dump` nếu có, hoặc dùng `pickle.dump` với `protocol=pickle.HIGHEST_PROTOCOL`.
    """
    path_obj = Path(file_path)
    ensure_dir(path_obj.parent)
    if HAS_JOBLIB:
        joblib.dump(obj, path_obj)
    else:
        with open(path_obj, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path_obj


def load_object(file_path: Union[str, Path]) -> Any:
    """
    What: Đọc đối tượng Python/Model từ tệp `.pkl` hoặc `.joblib` (hỗ trợ cả pickle Python 2 `latin1`).
    Why: Nạp lại các mô hình đã lưu cũng như ma trận kề `data/raw/sensor_graph/adj_mx_bay.pkl`
         của bài báo DCRNN gốc mà không bị lỗi `UnpicklingError`.
    Input:
        - file_path (Union[str, Path]): Đường dẫn tệp cần đọc.
    Output:
        - Any: Đối tượng Python đã khôi phục.
    How: Thử đọc bằng `pickle.load` chuẩn trước; nếu gặp `UnicodeDecodeError` (do tệp tạo từ
         Python 2 như `adj_mx_bay.pkl`) thì tự động đọc lại với `encoding="latin1"`.
    """
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp đối tượng: {path_obj}")
    try:
        with open(path_obj, "rb") as f:
            return pickle.load(f)
    except UnicodeDecodeError:
        with open(path_obj, "rb") as f:
            return pickle.load(f, encoding="latin1")
    except Exception:
        if HAS_JOBLIB:
            return joblib.load(path_obj)
        with open(path_obj, "rb") as f:
            return pickle.load(f, encoding="latin1")