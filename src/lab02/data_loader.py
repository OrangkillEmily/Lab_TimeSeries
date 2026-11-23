r"""
Mô-đun nạp và chia Train/Test tuần tự (80/20, shuffle=False) cho Dataset 1 & Dataset 2 của Lab 02.
Sử dụng lại lõi dùng chung tại src/common/data_loader.py.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import pandas as pd

from src.common.data_loader import load_time_series_csv, split_chronological
from src.lab02.utils import PROJECT_ROOT, get_path, load_config, save_json


def load_dataset(
    dataset_key: str,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    r"""
    Purpose: Nạp dữ liệu thô cho 'dataset1' (Monthly Retail Sales) hoặc 'dataset2' (AirPassengers) kèm metadata kiểm toán.
    Input-Output:
        - Input: dataset_key (str: 'dataset1'|'dataset2'), cfg (Optional[Dict[str, Any]]).
        - Output: Tuple (DataFrame có DatetimeIndex tần suất 'MS', Dict metadata kiểm toán).
    How: Lấy thông số cột/đường dẫn từ config['datasets'][dataset_key] và gọi load_time_series_csv().
    """
    config = cfg if cfg is not None else load_config()
    if dataset_key not in config["datasets"]:
        raise ValueError(f"Unsupported dataset_key='{dataset_key}'. Choose 'dataset1' or 'dataset2'.")

    spec = config["datasets"][dataset_key]
    raw_path = PROJECT_ROOT / spec["raw_path"]
    df, meta = load_time_series_csv(filepath=raw_path, date_col=spec["date_col"], freq=spec["freq"])
    series = df[spec["value_col"]].astype(float)

    meta.update(
        {
            "dataset_key": dataset_key,
            "dataset_title": spec["dataset_title"],
            "value_col": spec["value_col"],
            "seasonal_period": int(spec["seasonal_period"]),
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std()), 4),
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
        }
    )
    return df, meta


def split_dataset(
    df: pd.DataFrame,
    dataset_key: str,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    r"""
    Purpose: Chia dữ liệu Lab 02 thành 80% Train / 20% Test theo đúng thứ tự thời gian (shuffle=False) và lưu ra data/processed/lab02/.
    Input-Output:
        - Input: df (pd.DataFrame), dataset_key (str), cfg (Optional[Dict[str, Any]]).
        - Output: Tuple (train_df, test_df, split_meta).
    How: Gọi split_chronological() từ src/common/data_loader.py và lưu train_data.csv, test_data.csv, split_summary.json.
    """
    config = cfg if cfg is not None else load_config()
    test_ratio = float(config.get("split", {}).get("test_ratio", 0.20))
    save_dir = get_path("data_processed_dir", config) / dataset_key
    train_df, test_df, meta = split_chronological(df, test_ratio=test_ratio, save_dir=save_dir)
    save_json(meta, save_dir / "split_summary.json")
    return train_df, test_df, meta