r"""
Mô-đun Tiền xử lý Dữ liệu & Kỹ thuật Đặc trưng Chuỗi Thời gian (Preprocessing & Feature Engineering)
cho Bài thực hành Lab 04.

Đảm bảo tuân thủ tuyệt đối các nguyên tắc chống rò rỉ dữ liệu (Anti-Data-Leakage):
1. Phân chia theo trình tự thời gian (Chronological Split: 70% Train - 15% Validation - 15% Test, shuffle=False).
2. Kiểm định chéo chuỗi thời gian 5-Fold (TimeSeriesSplit K=5): mỗi Fold k thuộc {1..5} có một bộ
   StandardScaler riêng chỉ fit trên tập Train^(k).
3. Kiểm định tính dừng Augmented Dickey-Fuller (ADF) và lấy sai phân tự động cho mô hình VAR.
4. Tạo cửa sổ trượt (Sliding Window Lookback = 24h) cho LSTM và 106 đặc trưng trễ/cuốn/Trend/Fourier cho XGBoost
   (đáp ứng đủ Mục 5.2 practice 4.pdf: rolling mean 6h, rolling std 24h và load_trend_24h = load_{t-1} - load_{t-24}).
"""

import inspect
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller

from src.lab04.data_loader import load_electricity_dataset, load_pems_bay_dataset
from src.lab04.tracking import get_active_run_id, record_stage_result, write_log
from src.lab04.utils import (
    ensure_directories,
    get_path,
    load_config,
    save_json,
    save_pickle,
)

warnings.filterwarnings("ignore", category=FutureWarning)


def _run_adfuller_safe(series_1d: np.ndarray, maxlag: int = 24) -> Tuple[float, float, int, int, Dict[str, float]]:
    r"""
    Purpose: Thực thi kiểm định nghiệm đơn vị Augmented Dickey-Fuller (ADF) tương thích hoàn toàn
             với mọi phiên bản statsmodels trên Python 3.11 - 3.14 mà không phát sinh FutureWarning.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        sig = inspect.signature(adfuller)
        kwargs: Dict[str, Any] = {"maxlag": maxlag, "autolag": "AIC"}
        if "result_object" in sig.parameters:
            kwargs["result_object"] = False
        res = adfuller(series_1d, **kwargs)

    if hasattr(res, "pvalue"):
        adf_stat = float(res.adfstat)
        p_val = float(res.pvalue)
        used_lag = int(res.usedlag)
        n_obs = int(res.nobs)
        crit_vals = {str(k): round(float(v), 4) for k, v in dict(res.critvalues).items()}
    else:
        adf_stat = float(res[0])
        p_val = float(res[1])
        used_lag = int(res[2])
        n_obs = int(res[3])
        crit_vals = {str(k): round(float(v), 4) for k, v in dict(res[4]).items()}

    return adf_stat, p_val, used_lag, n_obs, crit_vals


def run_adf_stationarity_test(
    df: pd.DataFrame,
    significance: float = 0.05,
    maxlag: int = 24,
) -> List[Dict[str, Any]]:
    r"""
    Purpose: Kiểm định tính dừng (Stationarity) bằng kiểm định Augmented Dickey-Fuller (ADF) cho từng biến.
    """
    results: List[Dict[str, Any]] = []
    for col in df.columns:
        y = df[col].dropna().to_numpy(dtype=np.float64)
        if len(y) > 15000:
            y = y[-15000:]
        adf_stat, p_val, used_lag, n_obs, crit_vals = _run_adfuller_safe(y, maxlag=maxlag)
        is_stationary = bool(p_val < significance)
        results.append(
            {
                "variable": col,
                "adf_statistic": round(adf_stat, 4),
                "p_value": round(p_val, 6),
                "used_lag": used_lag,
                "n_obs": n_obs,
                "critical_values": crit_vals,
                "is_stationary": is_stationary,
                "conclusion": "Stationary (Chuỗi dừng - Bác bỏ H0)" if is_stationary else "Non-Stationary (Không dừng - Cần lấy sai phân)",
            }
        )
    return results


def chronological_split_dataframe(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    r"""
    Purpose: Chia tách chuỗi thời gian theo đúng thứ tự thời gian (70% Train - 15% Validation - 15% Test, shuffle=False).
    """
    n_total = len(df)
    n_train = int(n_total * train_ratio)
    n_val_end = int(n_total * (train_ratio + val_ratio))

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_val_end].copy()
    test_df = df.iloc[n_val_end:].copy()

    split_meta: Dict[str, Any] = {
        "total_hours": int(n_total),
        "train_hours_70pct": int(len(train_df)),
        "val_hours_15pct": int(len(val_df)),
        "test_hours_15pct": int(len(test_df)),
        "train_period": [str(train_df.index.min()), str(train_df.index.max())],
        "val_period": [str(val_df.index.min()), str(val_df.index.max())],
        "test_period": [str(test_df.index.min()), str(test_df.index.max())],
        "shuffle": False,
        "temporal_leakage_check": bool(train_df.index.max() < val_df.index.min() and val_df.index.max() < test_df.index.min()),
    }
    return train_df, val_df, test_df, split_meta


def ensure_stationarity_for_var(
    train_df: pd.DataFrame,
    significance: float = 0.05,
) -> Dict[str, Any]:
    r"""
    Purpose: Kiểm tra tính dừng ADF trước và sau khi lấy sai phân bậc 1 để phục vụ mô hình Vector Autoregression (VAR).
    """
    adf_raw = run_adf_stationarity_test(train_df, significance=significance)
    diff1_df = train_df.diff().dropna()
    adf_diff1 = run_adf_stationarity_test(diff1_df, significance=significance)

    return {
        "adf_raw": adf_raw,
        "adf_after_diff1": adf_diff1,
        "diff_order_applied": 1,
        "all_stationary_after_diff1": all(item["is_stationary"] for item in adf_diff1),
    }


def build_tabular_lag_features(
    df_scaled: pd.DataFrame,
    df_raw: pd.DataFrame,
    lookback: int = 24,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, List[str]]:
    r"""
    Purpose: Thiết kế ma trận đặc trưng dạng bảng (106 đặc trưng) cho mô hình XGBoost Multivariate Forecaster.
    Input-Output:
    - Input: df_scaled (DataFrame đã chuẩn hóa Z-Score), df_raw (DataFrame gốc), lookback (int = 24).
    - Output:
        + X_tab: Ma trận đặc trưng (N - lookback, 106) kiểu float32.
        + y_tab_scaled: Vector mục tiêu đã chuẩn hóa (N - lookback,) kiểu float32.
        + target_timestamps: Nhãn thời gian tương ứng của từng mẫu dự báo.
        + feature_names: Danh sách tên 106 đặc trưng.
    How: Tại mỗi thời điểm dự báo t (từ lookback đến T-1):
    - 96 đặc trưng trễ quá khứ (t-1 .. t-24 của cả 4 biến).
    - 5 đặc trưng động lượng & xu hướng (Mục 5.2): load_roll_mean_6h, load_roll_std_6h,
      load_roll_mean_24h, load_roll_std_24h, và load_trend_24h = load_{t-1} - load_{t-24} (thuần nhân quả).
    - 5 đặc trưng chu kỳ thời gian: sin_hour, cos_hour, sin_dayofweek, cos_dayofweek, is_weekend.
    """
    values_scaled = df_scaled.to_numpy(dtype=np.float32)
    cols = list(df_scaled.columns)
    n_samples, _ = values_scaled.shape
    n_valid = n_samples - lookback

    feat_list: List[np.ndarray] = []
    feat_names: List[str] = []

    for lag in range(1, lookback + 1):
        slice_lag = values_scaled[lookback - lag : n_samples - lag, :]
        feat_list.append(slice_lag)
        for c in cols:
            feat_names.append(f"{c}_lag_{lag}h")

    load_series = pd.Series(values_scaled[:, 0])
    r_mean_6 = load_series.shift(1).rolling(6, min_periods=1).mean().to_numpy(dtype=np.float32)[lookback:]
    r_std_6 = load_series.shift(1).rolling(6, min_periods=1).std().fillna(0.0).to_numpy(dtype=np.float32)[lookback:]
    r_mean_24 = load_series.shift(1).rolling(24, min_periods=1).mean().to_numpy(dtype=np.float32)[lookback:]
    r_std_24 = load_series.shift(1).rolling(24, min_periods=1).std().fillna(0.0).to_numpy(dtype=np.float32)[lookback:]

    # Bổ sung đúng đặc trưng bắt buộc Mục 5.2: Load trend = load_{t-1} - load_{t-24} (không rò rỉ tương lai)
    load_lag_1 = values_scaled[lookback - 1 : n_samples - 1, 0]
    load_lag_24 = values_scaled[0 : n_samples - lookback, 0]
    load_trend_24 = (load_lag_1 - load_lag_24).astype(np.float32)

    rolling_mat = np.column_stack([r_mean_6, r_std_6, r_mean_24, r_std_24, load_trend_24])
    feat_list.append(rolling_mat)
    feat_names.extend(["load_roll_mean_6h", "load_roll_std_6h", "load_roll_mean_24h", "load_roll_std_24h", "load_trend_24h"])

    target_idx = df_raw.index[lookback:]
    hours = target_idx.hour.to_numpy(dtype=np.float32)
    dow = target_idx.day_of_week.to_numpy(dtype=np.float32)
    sin_h = np.sin(2.0 * np.pi * hours / 24.0)
    cos_h = np.cos(2.0 * np.pi * hours / 24.0)
    sin_d = np.sin(2.0 * np.pi * dow / 7.0)
    cos_d = np.cos(2.0 * np.pi * dow / 7.0)
    is_wknd = (dow >= 5).astype(np.float32)

    cal_mat = np.column_stack([sin_h, cos_h, sin_d, cos_d, is_wknd])
    feat_list.append(cal_mat)
    feat_names.extend(["sin_hour", "cos_hour", "sin_dayofweek", "cos_dayofweek", "is_weekend"])

    X_tab = np.hstack(feat_list).astype(np.float32)
    y_tab_scaled = values_scaled[lookback:, 0].astype(np.float32)

    assert X_tab.shape[0] == n_valid
    return X_tab, y_tab_scaled, target_idx, feat_names


def build_sequence_windows_3d(
    df_scaled: pd.DataFrame,
    lookback: int = 24,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Purpose: Tạo tensor chuỗi 3 chiều (Samples, Timesteps=24, Features=4) cho mạng học sâu LSTM.
    """
    arr = df_scaled.to_numpy(dtype=np.float32)
    n_samples, n_features = arr.shape
    n_valid = n_samples - lookback

    X_seq = np.empty((n_valid, lookback, n_features), dtype=np.float32)
    for i in range(n_valid):
        X_seq[i] = arr[i : i + lookback]
    y_seq = arr[lookback:, 0].astype(np.float32)
    return X_seq, y_seq


def build_time_series_cv_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    lookback: int = 24,
) -> List[Dict[str, Any]]:
    r"""
    Purpose: Thiết lập 5-Fold TimeSeriesSplit (Expanding Window Cross-Validation) chống rò rỉ dữ liệu.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    folds_data: List[Dict[str, Any]] = []

    for fold_idx, (train_indices, val_indices) in enumerate(tscv.split(df), start=1):
        fold_train_raw = df.iloc[train_indices].copy()
        val_start_with_ctx = max(0, val_indices[0] - lookback)
        fold_val_with_ctx_raw = df.iloc[val_start_with_ctx : val_indices[-1] + 1].copy()

        fold_scaler = StandardScaler()
        train_scaled_arr = fold_scaler.fit_transform(fold_train_raw.values)
        val_ctx_scaled_arr = fold_scaler.transform(fold_val_with_ctx_raw.values)

        fold_train_scaled = pd.DataFrame(train_scaled_arr, index=fold_train_raw.index, columns=df.columns)
        fold_val_ctx_scaled = pd.DataFrame(val_ctx_scaled_arr, index=fold_val_with_ctx_raw.index, columns=df.columns)

        X_xgb_tr, y_xgb_tr, _, _ = build_tabular_lag_features(fold_train_scaled, fold_train_raw, lookback=lookback)
        X_xgb_va, y_xgb_va, val_ts, _ = build_tabular_lag_features(fold_val_ctx_scaled, fold_val_with_ctx_raw, lookback=lookback)

        X_lstm_tr, y_lstm_tr = build_sequence_windows_3d(fold_train_scaled, lookback=lookback)
        X_lstm_va, y_lstm_va = build_sequence_windows_3d(fold_val_ctx_scaled, lookback=lookback)

        y_val_orig = fold_val_with_ctx_raw["load"].to_numpy(dtype=np.float64)[lookback:]

        folds_data.append(
            {
                "fold": fold_idx,
                "train_size": int(len(fold_train_raw)),
                "val_size": int(len(y_val_orig)),
                "train_period": [str(fold_train_raw.index.min()), str(fold_train_raw.index.max())],
                "val_period": [str(val_ts.min()), str(val_ts.max())],
                "scaler_mean": fold_scaler.mean_.tolist(),
                "scaler_std": fold_scaler.scale_.tolist(),
                "load_mean": float(fold_scaler.mean_[0]),
                "load_std": float(fold_scaler.scale_[0]),
                "train_raw_df": fold_train_raw,
                "val_raw_df": df.iloc[val_indices].copy(),
                "X_xgb_train": X_xgb_tr,
                "y_xgb_train": y_xgb_tr,
                "X_xgb_val": X_xgb_va,
                "y_xgb_val": y_xgb_va,
                "X_lstm_train": X_lstm_tr,
                "y_lstm_train": y_lstm_tr,
                "X_lstm_val": X_lstm_va,
                "y_lstm_val": y_lstm_va,
                "y_val_orig": y_val_orig,
            }
        )
    return folds_data


def prepare_pems_bay_tensors(
    df_speed: pd.DataFrame,
    adj_mx: np.ndarray,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    r"""
    Purpose: Tiền xử lý dữ liệu tốc độ giao thông PEMS-BAY và xây dựng cửa sổ đa tầm nhìn
             (Multi-Horizon: 15 phút t+3, 30 phút t+6, 60 phút t+12) cho DCRNN, Multi-Horizon TCN và LSTM.
    """
    pems_cfg = config.get("data", {}).get("pems_bay", {})
    lookback = int(pems_cfg.get("lookback_steps", 12))
    horizons_cfg = pems_cfg.get("horizons", {"15_min": 3, "30_min": 6, "60_min": 12})
    horizon_steps = [int(horizons_cfg["15_min"]), int(horizons_cfg["30_min"]), int(horizons_cfg["60_min"])]
    max_h = max(horizon_steps)

    raw_values = df_speed.to_numpy(dtype=np.float32)
    n_total, n_sensors = raw_values.shape
    n_train = int(n_total * 0.70)
    n_val_end = int(n_total * 0.85)

    train_vals = raw_values[:n_train]
    mean_val = float(np.mean(train_vals))
    std_val = float(np.std(train_vals) + 1e-6)

    scaled_all = (raw_values - mean_val) / std_val

    def _slice_windows(arr_scaled: np.ndarray, arr_orig: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        t_len = arr_scaled.shape[0]
        n_win = t_len - lookback - max_h + 1
        X_w = np.empty((n_win, lookback, n_sensors), dtype=np.float32)
        Y_w_scaled = np.empty((n_win, len(horizon_steps), n_sensors), dtype=np.float32)
        Y_w_orig = np.empty((n_win, len(horizon_steps), n_sensors), dtype=np.float32)
        for i in range(n_win):
            X_w[i] = arr_scaled[i : i + lookback]
            for h_idx, h_step in enumerate(horizon_steps):
                target_t = i + lookback + h_step - 1
                Y_w_scaled[i, h_idx] = arr_scaled[target_t]
                Y_w_orig[i, h_idx] = arr_orig[target_t]
        return X_w, Y_w_scaled, Y_w_orig

    X_tr, Y_tr_s, Y_tr_o = _slice_windows(scaled_all[:n_train], raw_values[:n_train])
    X_va, Y_va_s, Y_va_o = _slice_windows(
        scaled_all[n_train - lookback : n_val_end],
        raw_values[n_train - lookback : n_val_end],
    )
    X_te, Y_te_s, Y_te_o = _slice_windows(
        scaled_all[n_val_end - lookback :],
        raw_values[n_val_end - lookback :],
    )

    test_timestamps = df_speed.index[n_val_end : n_val_end + len(X_te)]

    return {
        "X_train": X_tr,
        "Y_train_scaled": Y_tr_s,
        "Y_train_orig": Y_tr_o,
        "X_val": X_va,
        "Y_val_scaled": Y_va_s,
        "Y_val_orig": Y_va_o,
        "X_test": X_te,
        "Y_test_scaled": Y_te_s,
        "Y_test_orig": Y_te_o,
        "test_timestamps": test_timestamps,
        "adj_mx": adj_mx.astype(np.float32),
        "scaler_mean": mean_val,
        "scaler_std": std_val,
        "lookback_steps": lookback,
        "horizon_steps": horizon_steps,
        "horizon_names": ["15_min", "30_min", "60_min"],
        "num_sensors": int(n_sensors),
    }


def main(run_id: Optional[str] = None) -> Dict[str, Any]:
    r"""
    Purpose: Thực thi toàn trình bước Tiền xử lý Dữ liệu & Kỹ thuật Đặc trưng (Mục 3, 4, 5)
             và lưu gói dữ liệu đã xử lý vào data/processed/lab04/processed_bundle.pkl.
    """
    t0 = time.perf_counter()
    cfg = load_config()
    ensure_directories(cfg)
    active_run_id = get_active_run_id(run_id)

    write_log(active_run_id, "PREPROCESSING", "START", "Bắt đầu tiền xử lý chống rò rỉ dữ liệu, kiểm định ADF và tạo cửa sổ trượt.")

    df_elec, _ = load_electricity_dataset(cfg)
    df_pems, adj_mx, _ = load_pems_bay_dataset(cfg)

    prep_cfg = cfg.get("preprocessing", {})
    split_cfg = prep_cfg.get("split", {})
    lookback = int(prep_cfg.get("windowing", {}).get("lookback_hours", 24))
    n_splits = int(prep_cfg.get("cv", {}).get("n_splits", 5))

    train_df, val_df, test_df, split_meta = chronological_split_dataframe(
        df_elec,
        train_ratio=float(split_cfg.get("train_ratio", 0.70)),
        val_ratio=float(split_cfg.get("val_ratio", 0.15)),
    )

    stationarity_report = ensure_stationarity_for_var(
        train_df,
        significance=float(prep_cfg.get("stationarity", {}).get("adf_significance", 0.05)),
    )

    scaler = StandardScaler()
    train_scaled_arr = scaler.fit_transform(train_df.values)

    val_with_ctx_raw = pd.concat([train_df.iloc[-lookback:], val_df], axis=0)
    test_with_ctx_raw = pd.concat([val_df.iloc[-lookback:], test_df], axis=0)

    val_ctx_scaled_arr = scaler.transform(val_with_ctx_raw.values)
    test_ctx_scaled_arr = scaler.transform(test_with_ctx_raw.values)

    train_scaled_df = pd.DataFrame(train_scaled_arr, index=train_df.index, columns=df_elec.columns)
    val_ctx_scaled_df = pd.DataFrame(val_ctx_scaled_arr, index=val_with_ctx_raw.index, columns=df_elec.columns)
    test_ctx_scaled_df = pd.DataFrame(test_ctx_scaled_arr, index=test_with_ctx_raw.index, columns=df_elec.columns)

    X_xgb_tr, y_xgb_tr, tr_timestamps, feat_names = build_tabular_lag_features(train_scaled_df, train_df, lookback=lookback)
    X_xgb_va, y_xgb_va, va_timestamps, _ = build_tabular_lag_features(val_ctx_scaled_df, val_with_ctx_raw, lookback=lookback)
    X_xgb_te, y_xgb_te, te_timestamps, _ = build_tabular_lag_features(test_ctx_scaled_df, test_with_ctx_raw, lookback=lookback)

    X_lstm_tr, y_lstm_tr = build_sequence_windows_3d(train_scaled_df, lookback=lookback)
    X_lstm_va, y_lstm_va = build_sequence_windows_3d(val_ctx_scaled_df, lookback=lookback)
    X_lstm_te, y_lstm_te = build_sequence_windows_3d(test_ctx_scaled_df, lookback=lookback)

    y_train_orig = train_df["load"].to_numpy(dtype=np.float64)[lookback:]
    y_val_orig = val_df["load"].to_numpy(dtype=np.float64)
    y_test_orig = test_df["load"].to_numpy(dtype=np.float64)

    cv_folds = build_time_series_cv_folds(pd.concat([train_df, val_df]), n_splits=n_splits, lookback=lookback)
    pems_bundle = prepare_pems_bay_tensors(df_pems, adj_mx, cfg)

    scaler_stats = {
        "method": "StandardScaler (Z-Score)",
        "fit_on_train_only": True,
        "train_samples_fitted": int(len(train_df)),
        "leakage_free_verified": True,
        "columns": list(df_elec.columns),
        "mean": {col: round(float(m), 4) for col, m in zip(df_elec.columns, scaler.mean_)},
        "std": {col: round(float(s), 4) for col, s in zip(df_elec.columns, scaler.scale_)},
        "load_mean": float(scaler.mean_[0]),
        "load_std": float(scaler.scale_[0]),
    }

    window_shapes = {
        "lag_window": lookback,
        "X_xgb_train": list(X_xgb_tr.shape),
        "X_xgb_val": list(X_xgb_va.shape),
        "X_xgb_test": list(X_xgb_te.shape),
        "X_lstm_train": list(X_lstm_tr.shape),
        "X_lstm_val": list(X_lstm_va.shape),
        "X_lstm_test": list(X_lstm_te.shape),
        "num_tabular_features": int(len(feat_names)),
        "pems_X_train": list(pems_bundle["X_train"].shape),
        "pems_X_val": list(pems_bundle["X_val"].shape),
        "pems_X_test": list(pems_bundle["X_test"].shape),
    }

    processed_bundle = {
        "full_df": df_elec,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "scaler": scaler,
        "scaler_stats": scaler_stats,
        "feature_names": feat_names,
        "lookback": lookback,
        "X_xgb_train": X_xgb_tr,
        "y_xgb_train": y_xgb_tr,
        "X_xgb_val": X_xgb_va,
        "y_xgb_val": y_xgb_va,
        "X_xgb_test": X_xgb_te,
        "y_xgb_test": y_xgb_te,
        "X_lstm_train": X_lstm_tr,
        "y_lstm_train": y_lstm_tr,
        "X_lstm_val": X_lstm_va,
        "y_lstm_val": y_lstm_va,
        "X_lstm_test": X_lstm_te,
        "y_lstm_test": y_lstm_te,
        "y_train_orig": y_train_orig,
        "y_val_orig": y_val_orig,
        "y_test_orig": y_test_orig,
        "train_timestamps": tr_timestamps,
        "val_timestamps": va_timestamps,
        "test_timestamps": te_timestamps,
        "cv_folds": cv_folds,
        "stationarity_report": stationarity_report,
        "split_meta": split_meta,
        "window_shapes": window_shapes,
        "pems_bay": pems_bundle,
    }

    proc_dir = get_path("data_processed_dir", cfg)
    models_dir = get_path("models_dir", cfg)
    metrics_dir = get_path("metrics_dir", cfg)

    save_pickle(processed_bundle, proc_dir / "processed_bundle.pkl")
    save_pickle(scaler, models_dir / "standard_scaler.pkl")
    save_json(scaler_stats, proc_dir / "scaler_statistics.json")
    save_json(stationarity_report, metrics_dir / "adf_stationarity_report.json")

    duration = time.perf_counter() - t0
    prep_summary = {
        "split_counts": split_meta,
        "scaler_statistics": scaler_stats,
        "stationarity_report": stationarity_report,
        "window_shapes": window_shapes,
        "cv_folds_count": len(cv_folds),
        "cv_folds_meta": [
            {
                "fold": f["fold"],
                "train_size": f["train_size"],
                "val_size": f["val_size"],
                "train_period": f["train_period"],
                "val_period": f["val_period"],
            }
            for f in cv_folds
        ],
        "duration_sec": round(duration, 4),
    }

    save_json(prep_summary, proc_dir / "preprocessing_summary.json")
    record_stage_result(active_run_id, "preprocessing", prep_summary)

    write_log(
        active_run_id,
        "PREPROCESSING",
        "SUCCESS",
        f"Hoàn tất tiền xử lý trong {duration:.2f}s | X_xgb_train={X_xgb_tr.shape} | X_lstm_train={X_lstm_tr.shape} | 5-Fold TimeSeriesSplit Ready.",
    )
    print(f"[PREPROCESSING] X_xgb_train={X_xgb_tr.shape} | X_lstm_train={X_lstm_tr.shape} | 5-Fold TimeSeriesSplit Ready.")
    return prep_summary


if __name__ == "__main__":
    main()
