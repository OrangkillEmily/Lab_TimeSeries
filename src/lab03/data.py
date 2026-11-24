"""Data ingestion and chronological splitting for Lab 03."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def make_synthetic_hourly_series(periods: int = 3500, seed: int = 42) -> pd.Series:
    """Purpose: create an explicitly synthetic hourly series for offline smoke tests.

    Input: number of hourly observations and a reproducibility seed.
    Output: hourly pandas Series; this is never labelled as AEP source data.
    """
    if periods < 500:
        raise ValueError("Synthetic demo cần tối thiểu 500 quan sát để đủ lag 168 và CV.")
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01 00:00:00", periods=periods, freq="h")
    t = np.arange(periods, dtype=float)
    daily = 1150 * np.sin(2 * np.pi * (t - 8) / 24)
    weekly = 520 * np.sin(2 * np.pi * t / 168 + 0.6)
    annual = 650 * np.sin(2 * np.pi * t / (24 * 365.25) - 0.4)
    trend = 0.045 * t
    noise = rng.normal(0, 175, periods)
    y = 15000 + daily + weekly + annual + trend + noise
    return pd.Series(y, index=idx, name="AEP_MW_SYNTHETIC")


def load_hourly_series(path: str | Path, timestamp_col="Datetime", value_col="AEP_MW",
                       frequency="h", max_interpolation_gap=3) -> tuple[pd.Series, dict]:
    """Purpose: read, sort and regularize the source hourly time series.

    Input: CSV path, timestamp and value column names, frequency and max gap.
    Output: regularized Series plus auditable ingestion metadata.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu: {path}")
    raw = pd.read_csv(path)
    missing_cols = [c for c in (timestamp_col, value_col) if c not in raw.columns]
    if missing_cols:
        raise ValueError(f"CSV thiếu cột bắt buộc: {missing_cols}. Cột hiện có: {list(raw.columns)}")
    raw[timestamp_col] = pd.to_datetime(raw[timestamp_col], errors="coerce")
    raw[value_col] = pd.to_numeric(raw[value_col], errors="coerce")
    before = len(raw)
    raw = raw.dropna(subset=[timestamp_col, value_col]).sort_values(timestamp_col, kind="mergesort")
    duplicate_count = int(raw[timestamp_col].duplicated(keep="first").sum())
    raw = raw.drop_duplicates(subset=[timestamp_col], keep="first")
    series = raw.set_index(timestamp_col)[value_col].astype(float).sort_index()
    series = series.asfreq(frequency)
    missing_before = int(series.isna().sum())
    series = series.interpolate(method="time", limit=int(max_interpolation_gap), limit_area="inside")
    remaining = int(series.isna().sum())
    if remaining:
        raise ValueError(
            f"Sau khi nội suy tối đa {max_interpolation_gap} kỳ vẫn còn {remaining} giá trị thiếu. "
            "Hãy kiểm tra khoảng trống dài trong dữ liệu gốc; pipeline không tự điền bừa."
        )
    series.name = value_col
    meta = {
        "dataset_name": path.name,
        "dataset_type": "real_source",
        "source_path": str(path),
        "raw_rows": before,
        "rows_after_drop_invalid": int(len(raw)),
        "duplicate_timestamps_removed": duplicate_count,
        "missing_periods_inserted": missing_before,
        "observations_after_cleaning": int(len(series)),
        "start": str(series.index.min()),
        "end": str(series.index.max()),
        "frequency": frequency,
        "target_column": value_col,
    }
    return series, meta


def chronological_split(X, y, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """Purpose: split supervised rows in order into non-overlapping train/val/test.

    Input: time-indexed feature matrix and target plus three ratios summing to 1.
    Output: (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    total = float(train_ratio + val_ratio + test_ratio)
    if not np.isclose(total, 1.0, atol=1e-8):
        raise ValueError("train_ratio + val_ratio + test_ratio phải bằng 1.")
    if not X.index.is_monotonic_increasing or not y.index.is_monotonic_increasing:
        raise ValueError("Dữ liệu phải được sắp xếp tăng dần theo thời gian.")
    n = len(X)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    if min(n_train, n_val, n_test) < 1:
        raise ValueError("Không đủ dữ liệu để tạo ba tập Train/Validation/Test.")
    X_train, X_val, X_test = X.iloc[:n_train], X.iloc[n_train:n_train+n_val], X.iloc[n_train+n_val:]
    y_train, y_val, y_test = y.iloc[:n_train], y.iloc[n_train:n_train+n_val], y.iloc[n_train+n_val:]
    if not (X_train.index.max() < X_val.index.min() < X_test.index.min()):
        raise AssertionError("Temporal ordering violation in split.")
    return X_train, X_val, X_test, y_train, y_val, y_test
