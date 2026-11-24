"""Leakage-aware lag, rolling and calendar feature construction."""
from __future__ import annotations
import numpy as np
import pandas as pd


def build_supervised_features(series: pd.Series, lags=(1, 2, 3, 6, 12, 24, 48, 168),
                              rolling_windows=(6, 12, 24, 168),
                              include_calendar=True) -> tuple[pd.DataFrame, pd.Series]:
    """Purpose: transform a univariate series into a one-step supervised table.

    Input: sorted time series, lag offsets, rolling windows and calendar-feature flag.
    Output: X/y indexed by target timestamp; rolling statistics use shift(1) to prevent leakage.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("series.index phải là DatetimeIndex.")
    series = series.sort_index().astype(float)
    frame = pd.DataFrame(index=series.index)
    for lag in sorted(set(int(x) for x in lags)):
        if lag < 1:
            raise ValueError("Mọi lag phải >= 1.")
        frame[f"lag_{lag}"] = series.shift(lag)
    past = series.shift(1)
    for window in sorted(set(int(x) for x in rolling_windows)):
        if window < 2:
            raise ValueError("Rolling window phải >= 2.")
        frame[f"roll_mean_{window}"] = past.rolling(window, min_periods=window).mean()
        frame[f"roll_std_{window}"] = past.rolling(window, min_periods=window).std(ddof=1)
    if include_calendar:
        idx = frame.index
        hour = idx.hour.to_numpy()
        dow = idx.dayofweek.to_numpy()
        month = idx.month.to_numpy() - 1
        frame["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        frame["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        frame["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        frame["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        frame["month_sin"] = np.sin(2 * np.pi * month / 12)
        frame["month_cos"] = np.cos(2 * np.pi * month / 12)
    y = series.rename("target")
    joined = frame.join(y).replace([np.inf, -np.inf], np.nan).dropna()
    return joined.drop(columns=["target"]), joined["target"]


def make_next_step_features(history: pd.Series, timestamp: pd.Timestamp, feature_names,
                            lags=(1, 2, 3, 6, 12, 24, 48, 168),
                            rolling_windows=(6, 12, 24, 168), include_calendar=True) -> pd.DataFrame:
    """Purpose: create a single inference row using observations available before timestamp.

    Input: observed/predicted history, next timestamp and exact feature-column order.
    Output: one-row DataFrame compatible with a model trained by build_supervised_features.
    """
    values = {}
    for lag in lags:
        if len(history) < lag:
            raise ValueError(f"Cần tối thiểu {lag} điểm lịch sử để tạo lag_{lag}.")
        values[f"lag_{lag}"] = float(history.iloc[-lag])
    past = history.astype(float)
    for window in rolling_windows:
        if len(past) < window:
            raise ValueError(f"Cần tối thiểu {window} điểm lịch sử để tạo rolling_{window}.")
        values[f"roll_mean_{window}"] = float(past.iloc[-window:].mean())
        values[f"roll_std_{window}"] = float(past.iloc[-window:].std(ddof=1))
    if include_calendar:
        hour, dow, month = timestamp.hour, timestamp.dayofweek, timestamp.month - 1
        values.update({
            "hour_sin": float(np.sin(2*np.pi*hour/24)), "hour_cos": float(np.cos(2*np.pi*hour/24)),
            "dow_sin": float(np.sin(2*np.pi*dow/7)), "dow_cos": float(np.cos(2*np.pi*dow/7)),
            "month_sin": float(np.sin(2*np.pi*month/12)), "month_cos": float(np.cos(2*np.pi*month/12)),
        })
    row = pd.DataFrame([values], index=[timestamp])
    return row.reindex(columns=list(feature_names))


def recursive_forecast(model, history: pd.Series, horizon: int, feature_names,
                       lags, rolling_windows, include_calendar=True) -> pd.Series:
    """Purpose: roll a one-step model forward without using future ground truth.

    Input: fitted model, last observed history, forecast horizon and feature schema.
    Output: future Series where each prediction is fed back into subsequent lag features.
    """
    hist = history.copy().astype(float)
    predictions = []
    for _ in range(int(horizon)):
        timestamp = hist.index[-1] + pd.tseries.frequencies.to_offset(pd.infer_freq(hist.index) or "h")
        row = make_next_step_features(hist, timestamp, feature_names, lags, rolling_windows, include_calendar)
        pred = float(np.asarray(model.predict(row.to_numpy(dtype=float))).reshape(-1)[0])
        predictions.append((timestamp, pred))
        hist.loc[timestamp] = pred
    return pd.Series({ts: value for ts, value in predictions}, name="forecast", dtype=float)
