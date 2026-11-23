"""
Mô-đun nạp và chia dữ liệu chuỗi thời gian dùng chung (src/common/data_loader.py).
Hoàn toàn độc lập, không phụ thuộc ngược vào bất kỳ gói lab riêng lẻ nào.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import pandas as pd


def load_time_series_csv(
    filepath: str | Path,
    date_col: str = "Month",
    freq: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Đọc file CSV chuỗi thời gian, chuyển date_col thành DatetimeIndex,
    sắp xếp tăng dần theo thời gian và trả về kèm metadata kiểm toán.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at: {path}")

    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(
            f"Column '{date_col}' not found in {path.name}. "
            f"Available columns: {list(df.columns)}"
        )

    df[date_col] = pd.to_datetime(df[date_col])
    duplicate_dates = int(df[date_col].duplicated().sum())
    is_monotonic = bool(df[date_col].is_monotonic_increasing)

    df = df.set_index(date_col)
    if not is_monotonic:
        df = df.sort_index()

    inferred_freq = pd.infer_freq(df.index) if len(df.index) >= 3 else None
    if freq is not None:
        df = df.asfreq(freq)
        effective_freq = freq
    elif inferred_freq is not None:
        effective_freq = str(inferred_freq)
    else:
        effective_freq = "Unknown"

    meta: Dict[str, Any] = {
        "file_path": str(path),
        "n_obs": int(len(df)),
        "start_date": str(df.index.min().date()) if len(df) > 0 else "N/A",
        "end_date": str(df.index.max().date()) if len(df) > 0 else "N/A",
        "frequency": effective_freq,
        "columns": [date_col] + list(df.columns),
        "value_columns": list(df.columns),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_timestamps": duplicate_dates,
        "is_monotonic": is_monotonic,
    }
    return df, meta


def split_chronological(
    df_or_series: pd.DataFrame | pd.Series,
    test_ratio: float = 0.2,
    save_dir: Optional[str | Path] = None,
) -> Tuple[pd.DataFrame | pd.Series, pd.DataFrame | pd.Series, Dict[str, Any]]:
    """
    Chia chuỗi thời gian thành Train/Test tuần tự (shuffle=False) bằng cắt chỉ số thuần pandas.
    """
    if df_or_series is None or len(df_or_series) < 10:
        raise ValueError("Time series must contain at least 10 observations to split Train/Test.")
    if not (0.05 <= test_ratio <= 0.5):
        raise ValueError(f"Invalid test_ratio={test_ratio}. Expected value between 0.05 and 0.50.")

    if not df_or_series.index.is_monotonic_increasing:
        df_or_series = df_or_series.sort_index()

    n_total = len(df_or_series)
    test_size = int(round(n_total * test_ratio))
    # Khớp hành vi sklearn train_test_split(test_size=float): dùng ceil khi chia lẻ
    import math
    test_size = math.ceil(n_total * test_ratio)
    train_size = n_total - test_size

    train_data = df_or_series.iloc[:train_size].copy()
    test_data = df_or_series.iloc[train_size:].copy()

    if save_dir is not None:
        out_path = Path(save_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        train_data.to_csv(out_path / "train_data.csv")
        test_data.to_csv(out_path / "test_data.csv")

    train_ratio_pct = round((1.0 - test_ratio) * 100)
    test_ratio_pct = round(test_ratio * 100)
    split_label = f"{train_ratio_pct}/{test_ratio_pct} (shuffle=False)"

    meta: Dict[str, Any] = {
        "n_obs": int(n_total),
        "train_size": int(len(train_data)),
        "test_size": int(len(test_data)),
        "train_start": str(train_data.index.min().date()),
        "train_end": str(train_data.index.max().date()),
        "test_start": str(test_data.index.min().date()),
        "test_end": str(test_data.index.max().date()),
        "split_ratio": split_label,
        "frequency": str(df_or_series.index.freqstr or pd.infer_freq(df_or_series.index) or "MS"),
    }
    return train_data, test_data, meta