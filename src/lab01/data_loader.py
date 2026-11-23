r"""
Mô-đun nạp và kiểm toán dữ liệu gốc AirPassengers.csv cho Bài thực hành Lab 01.
Sử dụng lại lõi dùng chung tại src/common/data_loader.py.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.common.data_loader import load_time_series_csv
from src.lab01.tracking import get_active_run_id, record_stage_result, write_log
from src.lab01.utils import ensure_directories, get_path, load_config, save_json


def load_airpassengers_dataset(
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    r"""
    Purpose: Nạp tập dữ liệu data/raw/AirPassengers.csv, kiểm tra tính liên tục thời gian và tính thống kê mô tả.
    Input-Output:
        - Input: cfg (Optional[Dict[str, Any]]).
        - Output: Tuple gồm (DataFrame có DatetimeIndex tần suất 'MS', Dict metadata kiểm toán).
    How: Gọi load_time_series_csv từ src/common/data_loader.py và bổ sung các chỉ số mean, std, min, max, skewness.
    """
    config = cfg if cfg is not None else load_config()
    raw_path = get_path("data_raw", config)
    date_col = config["data"]["date_col"]
    value_col = config["data"]["value_col"]
    freq = config["data"]["freq"]

    df, meta = load_time_series_csv(filepath=raw_path, date_col=date_col, freq=freq)
    series = df[value_col].astype(float)

    meta.update(
        {
            "dataset_name": "AirPassengers (Monthly Airline Passengers 1949-1960)",
            "value_col": value_col,
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std()), 4),
            "min": round(float(series.min()), 4),
            "q25": round(float(series.quantile(0.25)), 4),
            "median": round(float(series.median()), 4),
            "q75": round(float(series.quantile(0.75)), 4),
            "max": round(float(series.max()), 4),
            "skewness": round(float(series.skew()), 4),
            "kurtosis": round(float(series.kurtosis()), 4),
            "growth_ratio": round(float(series.iloc[-1] / series.iloc[0]), 4),
        }
    )
    return df, meta


def main(run_id: Optional[str] = None) -> Dict[str, Any]:
    r"""
    Purpose: Thực thi Giai đoạn 1 (Data Loading & Schema Audit) cho Lab 01.
    Input-Output:
        - Input: run_id (Optional[str]).
        - Output: Từ điển kết quả kiểm toán dữ liệu gốc.
    How: Nạp AirPassengers.csv, lưu bản sạch vào data/processed/lab01/, lưu JSON kiểm toán và ghi log.
    """
    t0 = time.perf_counter()
    cfg = load_config()
    ensure_directories(cfg)
    active_run_id = get_active_run_id(run_id)

    write_log(active_run_id, "DATA_LOADING", "START", "Bắt đầu nạp và kiểm toán dữ liệu AirPassengers.csv.")
    df, meta = load_airpassengers_dataset(cfg)

    proc_dir = get_path("data_processed_dir", cfg)
    df.to_csv(proc_dir / "airpassengers_clean.csv")
    meta["duration_sec"] = round(time.perf_counter() - t0, 4)
    save_json(meta, proc_dir / "data_inspection_summary.json")

    record_stage_result(active_run_id, "data_loading", meta)
    write_log(
        active_run_id,
        "DATA_LOADING",
        "SUCCESS",
        f"Nạp thành công {meta['n_obs']} quan sát ({meta['start_date']} -> {meta['end_date']}), missing={meta['missing_values']}.",
    )
    return meta