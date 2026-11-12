from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import pandas as pd


def load_and_prep_data(
    filepath: str | Path,
    date_col: str = "Month",
    freq: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Purpose:
        Read a time series CSV file, parse the date column as DatetimeIndex,
        sort chronologically without shuffling, validate missing/duplicate timestamps,
        optionally assign frequency, and log metadata if dataset_name is provided.
        Fully backward-compatible with LAB01 and LAB02 calls.

    Parameters:
        filepath (str | Path): Relative or absolute path to the CSV file.
        date_col (str): Name of the date/time column (default 'Month').
        freq (Optional[str]): Optional pandas frequency string (e.g., 'MS') to assign.
        dataset_name (Optional[str]): Optional LAB02 dataset identifier to trigger structured logging.

    Returns:
        pd.DataFrame: Prepared DataFrame indexed by a chronological DatetimeIndex.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at: {path}")

    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in {path.name}. Available columns: {list(df.columns)}")

    df[date_col] = pd.to_datetime(df[date_col])
    duplicate_dates = int(df[date_col].duplicated().sum())
    is_monotonic = bool(df[date_col].is_monotonic_increasing)

    df.set_index(date_col, inplace=True)
    if not is_monotonic:
        df.sort_index(inplace=True)

    inferred_freq = pd.infer_freq(df.index) if len(df.index) >= 3 else None
    if freq is not None:
        df = df.asfreq(freq)
        effective_freq = freq
    elif inferred_freq is not None:
        effective_freq = str(inferred_freq)
    else:
        effective_freq = "Unknown"

    if dataset_name is not None:
        from src.lab02.lab02_utils.logging_config import log_data_loading

        missing_count = int(df.isna().sum().sum())
        log_data_loading(
            dataset_name=dataset_name,
            file_path=path,
            n_obs=len(df),
            start_date=str(df.index.min().date()) if len(df) > 0 else "N/A",
            end_date=str(df.index.max().date()) if len(df) > 0 else "N/A",
            frequency=effective_freq,
            columns=[date_col] + list(df.columns),
            missing_values=missing_count,
            duplicate_timestamps=duplicate_dates,
            is_monotonic=is_monotonic,
        )

    return df


def split_train_test_chronological(
    df_or_series: pd.DataFrame | pd.Series,
    test_ratio: float = 0.2,
    dataset_name: Optional[str] = None,
    save_dir: Optional[str | Path] = None,
    differencing_info: str = "d=1 (ARIMA) / d=1, D=1, s=12 (SARIMA)",
    transformations: str = "None (Original Scale)",
) -> Tuple[pd.DataFrame | pd.Series, pd.DataFrame | pd.Series, Dict[str, Any]]:
    """
    Purpose:
        Split a time series DataFrame or Series into chronological Train and Test sets
        (default 80% Train / 20% Test) strictly without shuffling (shuffle=False),
        optionally saving train_data.csv and test_data.csv and logging preprocessing details.

    Parameters:
        df_or_series (pd.DataFrame | pd.Series): Input time series indexed by DatetimeIndex.
        test_ratio (float): Fraction of observations reserved for out-of-sample testing (default 0.2).
        dataset_name (Optional[str]): Dataset identifier for structured logging.
        save_dir (Optional[str | Path]): Directory to save train_data.csv and test_data.csv.
        differencing_info (str): Summary of planned differencing orders for logging.
        transformations (str): Summary of variance transformations for logging.

    Returns:
        Tuple[train, test, metadata]: Chronological Train set, Test set, and split metadata dict.
    """
    if df_or_series is None or len(df_or_series) < 10:
        raise ValueError("Time series must contain at least 10 observations to split Train/Test.")
    if not (0.05 <= test_ratio <= 0.5):
        raise ValueError(f"Invalid test_ratio={test_ratio}. Expected value between 0.05 and 0.50.")

    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        from src.lab02.lab02_utils.scipy_tsa_compat import train_test_split

    if not df_or_series.index.is_monotonic_increasing:
        df_or_series = df_or_series.sort_index()

    train_data, test_data = train_test_split(df_or_series, test_size=test_ratio, shuffle=False)

    if save_dir is not None:
        out_path = Path(save_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        train_data.to_csv(out_path / "train_data.csv")
        test_data.to_csv(out_path / "test_data.csv")

    train_ratio_pct = round((1.0 - test_ratio) * 100)
    test_ratio_pct = round(test_ratio * 100)
    split_label = f"{train_ratio_pct}/{test_ratio_pct} (shuffle=False)"

    meta = {
        "n_obs": len(df_or_series),
        "train_size": len(train_data),
        "test_size": len(test_data),
        "train_start": str(train_data.index.min().date()),
        "train_end": str(train_data.index.max().date()),
        "test_start": str(test_data.index.min().date()),
        "test_end": str(test_data.index.max().date()),
        "split_ratio": split_label,
        "frequency": str(df_or_series.index.freqstr or pd.infer_freq(df_or_series.index) or "MS"),
    }

    if dataset_name is not None:
        from src.lab02.lab02_utils.logging_config import log_preprocessing

        log_preprocessing(
            dataset_name=dataset_name,
            train_size=meta["train_size"],
            test_size=meta["test_size"],
            train_start=meta["train_start"],
            train_end=meta["train_end"],
            test_start=meta["test_start"],
            test_end=meta["test_end"],
            split_ratio=split_label,
            differencing_info=differencing_info,
            transformations=transformations,
            output_dir=save_dir,
        )

    return train_data, test_data, meta