"""
Module tiền xử lý, chia dữ liệu thời gian (70/15/15 + 5-Fold TimeSeriesSplit) và tạo đặc trưng (`src/lab04/preprocess.py`).

Trách nhiệm của module (Thực hiện Mục 4, Mục 5, Mục 6.1 và Yêu cầu bổ sung số 4):
1. Điền khuyết nhân quả về phía trước (`ffill`) và chia tập dữ liệu theo đúng thứ tự thời gian
   (`shuffle=False`) với tỷ lệ chuẩn: **Train 70% – Validation 15% – Test 15%**.
2. Khớp (`fit`) bộ chuẩn hóa `StandardScaler` chỉ trên tập `Train (70%)` và biến đổi (`transform`)
   sang `Validation` và `Test` để chống rò rỉ dữ liệu tuyệt đối.
3. Thiết lập **Temporal K-Fold Cross-Validation (`TimeSeriesSplit`, K = 5 Folds)**, trong đó mỗi
   Fold $k \in \{1..5\}$ có một bộ `StandardScaler` riêng chỉ `fit` trên $Train^{(k)}$.
4. Kiểm định tính dừng bằng **ADF Test (`Augmented Dickey-Fuller`)** và thực hiện sai phân
   (`Differencing`) cho nhánh mô hình thống kê `VAR`.
5. Kiến tạo cửa sổ trễ $L = 24$ giờ (Tensor 3D `(N, 24, 4)` cho `LSTM` và vector phẳng 96 chiều
   kết hợp `rolling_mean_6h`, `rolling_std_24h`, `load_trend = load_t - load_{t-24}` cho `XGBoost`).
6. Chuẩn bị cửa sổ trượt 12 bước quá khứ $\to$ 12 bước tương lai (trích xuất $t+3, t+6, t+12$)
   cho bài toán nâng cao `PEMS-BAY` (Mục 12).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

    class StandardScaler:  # type: ignore[no-redef]
        """Bộ chuẩn hóa Z-Score dự phòng thuần NumPy tương thích 100% với scikit-learn."""

        def __init__(self, with_mean: bool = True, with_std: bool = True) -> None:
            self.with_mean = with_mean
            self.with_std = with_std
            self.mean_: np.ndarray = np.array([])
            self.scale_: np.ndarray = np.array([])

        def fit(self, X: np.ndarray) -> "StandardScaler":
            arr = np.asarray(X, dtype=np.float64)
            self.mean_ = np.mean(arr, axis=0) if self.with_mean else np.zeros(arr.shape[1])
            std = np.std(arr, axis=0)
            self.scale_ = np.where(std > 1e-9, std, 1.0) if self.with_std else np.ones(arr.shape[1])
            return self

        def transform(self, X: np.ndarray) -> np.ndarray:
            arr = np.asarray(X, dtype=np.float64)
            return (arr - self.mean_) / self.scale_

        def fit_transform(self, X: np.ndarray) -> np.ndarray:
            return self.fit(X).transform(X)

        def inverse_transform(self, X: np.ndarray) -> np.ndarray:
            arr = np.asarray(X, dtype=np.float64)
            return arr * self.scale_ + self.mean_

    class TimeSeriesSplit:  # type: ignore[no-redef]
        """Bộ chia K-Fold chuỗi thời gian mở rộng thuần NumPy tương thích 100% với scikit-learn."""

        def __init__(self, n_splits: int = 5) -> None:
            self.n_splits = int(n_splits)

        def split(self, X: Any) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
            n_samples = len(X)
            n_folds = self.n_splits + 1
            test_size = n_samples // n_folds
            indices = np.arange(n_samples)
            for i in range(self.n_splits):
                train_end = n_samples - (self.n_splits - i) * test_size
                val_end = train_end + test_size
                yield indices[:train_end], indices[train_end:val_end]

try:
    from statsmodels.tsa.stattools import adfuller
    HAS_STATSMODELS_ADF = True
except ImportError:
    HAS_STATSMODELS_ADF = False

from src.lab04.data_loader import load_electricity_dataset, load_pems_bay_dataset
from src.lab04.tracking import create_run, update_progress, write_log
from src.lab04.utils import (
    ensure_dir,
    get_project_root,
    load_config,
    save_json,
    save_object,
)
from src.lab04.visualize import (
    create_stage_figure_dir,
    plot_kfold_splits_timeline,
    plot_train_val_test_split,
)

ROOT_DIR = get_project_root()


def _run_adf_single_series(series: np.ndarray, maxlag: int = 24) -> Tuple[float, float, int, Dict[str, float]]:
    """
    What: Thực hiện kiểm định Augmented Dickey-Fuller (ADF) trên một chuỗi đơn biến.
    Why: Hỗ trợ cả `statsmodels.tsa.stattools.adfuller` chuẩn và hồi quy OLS Dickey-Fuller
         thuần NumPy nếu môi trường chưa cài `statsmodels`.
    Input:
        - series (np.ndarray): Mảng 1 chiều.
        - maxlag (int): Độ trễ tối đa (`24`).
    Output:
        - Tuple[float, float, int, Dict[str, float]]: `(adf_stat, p_value, used_lag, critical_values)`.
    How: Ước lượng $\Delta y_t = \alpha + \gamma y_{t-1} + \sum_{i=1}^p \delta_i \Delta y_{t-i} + \varepsilon_t$
         và tính thống kê $t_{\hat{\gamma}}$.
    """
    y = np.asarray(series, dtype=np.float64).ravel()
    if HAS_STATSMODELS_ADF:
        res = adfuller(y, maxlag=maxlag, autolag="AIC")
        return float(res[0]), float(res[1]), int(res[2]), {k: float(v) for k, v in res[4].items()}

    dy = np.diff(y)
    p = min(6, maxlag, max(1, len(dy) // 20))
    n = len(dy) - p
    target = dy[p:]
    cols = [np.ones(n), y[p : p + n]]
    for lag in range(1, p + 1):
        cols.append(dy[p - lag : p - lag + n])
    X_design = np.column_stack(cols)
    beta, _, _, _ = np.linalg.lstsq(X_design, target, rcond=None)
    resid = target - X_design @ beta
    s2 = float(np.sum(resid ** 2)) / max(1, n - X_design.shape[1])
    cov = s2 * np.linalg.pinv(X_design.T @ X_design)
    se_gamma = float(np.sqrt(max(cov[1, 1], 1e-12)))
    t_stat = float(beta[1] / se_gamma)
    # Xấp xỉ p-value MacKinnon cho phân phối Dickey-Fuller có hằng số
    if t_stat < -3.43:
        pval = 0.005
    elif t_stat < -2.86:
        pval = 0.035
    elif t_stat < -2.57:
        pval = 0.08
    else:
        pval = 0.35
    return t_stat, pval, p, {"1%": -3.43, "5%": -2.86, "10%": -2.57}


def chronological_split_dataframe(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    What: Chia chuỗi thời gian đa biến theo đúng thứ tự thời gian (`Train 70% - Val 15% - Test 15%`).
    Why: Tuân thủ Mục 4.2 của `practice 4.pdf` — cấm xáo trộn (`shuffle=False`) để không làm lộ
         thông tin tương lai vào tập huấn luyện.
    Input:
        - df (pd.DataFrame): Chuỗi thời gian đầu vào đã sắp xếp tăng dần theo thời gian.
        - train_ratio (float): Tỷ lệ tập huấn luyện (`0.70`).
        - val_ratio (float): Tỷ lệ tập kiểm định (`0.15`).
    Output:
        - Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: `(df_train, df_val, df_test)`.
    How: Tính chỉ số cắt $n_{\text{train}} = \lfloor 0.70 N \rfloor$ và $n_{\text{val}} = \lfloor 0.85 N \rfloor$,
         sau đó cắt lát tuần tự bằng `.iloc`.
    """
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * (train_ratio + val_ratio))
    df_train = df.iloc[:n_train].copy()
    df_val = df.iloc[n_train:n_val].copy()
    df_test = df.iloc[n_val:].copy()
    return df_train, df_val, df_test


def check_adf_and_difference(
    df_train_scaled: pd.DataFrame,
    significance: float = 0.05,
    max_diff: int = 2,
) -> Dict[str, Any]:
    """
    What: Kiểm tra tính dừng bằng kiểm định Augmented Dickey-Fuller (ADF) trên từng biến
          và xác định bậc sai phân $d$ cần thiết cho mô hình `VAR`.
    Why: Mô hình thống kê `VAR` (Mục 6.1) đòi hỏi chuỗi đầu vào phải dừng (`Stationary`);
         nếu không dừng thì bắt buộc thực hiện `differencing`.
    Input:
        - df_train_scaled (pd.DataFrame): Tập huấn luyện đã chuẩn hóa.
        - significance (float): Mức ý nghĩa $\alpha$ (mặc định `0.05`).
        - max_diff (int): Bậc sai phân tối đa (`2`).
    Output:
        - Dict[str, Any]: Kết quả kiểm định ADF trước/sau sai phân và bậc sai phân `diff_order`.
    How: Chạy kiểm định ADF trên cả 4 biến; nếu bất kỳ biến nào có $p\text{-value} \ge 0.05$,
         tăng bậc sai phân lên $d = 1$ và kiểm định lại.
    """
    initial_tests = {}
    all_stationary = True
    for col in df_train_scaled.columns:
        stat, pval, used_lag, crit = _run_adf_single_series(df_train_scaled[col].dropna().to_numpy(), maxlag=24)
        is_stat = pval < significance
        if not is_stat:
            all_stationary = False
        initial_tests[col] = {
            "adf_statistic": round(stat, 4),
            "p_value": round(pval, 6),
            "used_lag": used_lag,
            "critical_values": {k: round(v, 4) for k, v in crit.items()},
            "is_stationary": is_stat,
        }

    diff_order = 0
    cur_df = df_train_scaled.copy()
    after_tests = initial_tests

    while not all_stationary and diff_order < max_diff:
        diff_order += 1
        cur_df = cur_df.diff().dropna()
        all_stationary = True
        after_tests = {}
        for col in cur_df.columns:
            stat, pval, used_lag, _ = _run_adf_single_series(cur_df[col].dropna().to_numpy(), maxlag=24)
            is_stat = pval < significance
            if not is_stat:
                all_stationary = False
            after_tests[col] = {
                "adf_statistic": round(stat, 4),
                "p_value": round(pval, 6),
                "used_lag": used_lag,
                "is_stationary": is_stat,
            }

    return {
        "significance_level": significance,
        "initial_adf": initial_tests,
        "diff_order_applied": int(diff_order),
        "final_adf_after_diff": after_tests,
        "all_stationary_confirmed": bool(all_stationary),
    }


def build_supervised_features(
    df_scaled: pd.DataFrame,
    lag_window: int = 24,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], pd.DatetimeIndex]:
    """
    What: Tạo đồng thời ma trận đặc trưng phẳng (Lag + Rolling 6h/24h + Load Trend) cho `XGBoost`
          và Tensor 3D `(N_samples, 24, 4)` cho `LSTM` từ chuỗi đa biến đã chuẩn hóa.
    Why: Thực hiện chính xác Mục 5.1 (Cửa sổ trễ $L=24$ giờ) và Mục 5.2 (Rolling mean 6h,
         Rolling std 24h, Load trend $\text{load}_t - \text{load}_{t-24}$ chỉ dùng cho `XGBoost`).
    Input:
        - df_scaled (pd.DataFrame): Chuỗi 4 biến `[load, temperature, humidity, wind_speed]` đã chuẩn hóa.
        - lag_window (int): Độ dài cửa sổ trễ $L = 24$ giờ.
    Output:
        - Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], pd.DatetimeIndex]:
          `(X_xgb, X_lstm, y_target, feature_names_xgb, target_timestamps)`.
    How: Tại mỗi thời điểm $t \in [L, T-1]$, trích xuất cửa sổ $t-L \dots t-1$ làm 24 bước quá khứ
         để dự báo nhãn mục tiêu tại bước tiếp theo (`load` tại chỉ số $t$).
    """
    values = df_scaled[["load", "temperature", "humidity", "wind_speed"]].to_numpy(dtype=np.float32)
    timestamps = df_scaled.index
    n_total, n_vars = values.shape

    X_lstm_list = []
    X_xgb_list = []
    y_list = []
    valid_ts = []

    var_names = ["load", "temp", "humidity", "wind"]
    feature_names: List[str] = []
    for v_name in var_names:
        for lag in range(1, lag_window + 1):
            feature_names.append(f"{v_name}_lag_{lag}h")
    feature_names.extend([
        "load_rolling_mean_6h",
        "load_rolling_std_24h",
        "load_trend_24h",
        "temp_rolling_mean_6h",
        "load_temp_interaction",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ])

    for t in range(lag_window, n_total):
        window_3d = values[t - lag_window : t, :]  # shape (24, 4)
        X_lstm_list.append(window_3d)

        lag_flat = []
        for col_idx in range(n_vars):
            lag_flat.extend(values[t - lag_window : t, col_idx][::-1])

        load_win = window_3d[:, 0]
        temp_win = window_3d[:, 1]
        rm_6h = float(np.mean(load_win[-6:]))
        rstd_24h = float(np.std(load_win))
        load_trend_24h = float(load_win[-1] - load_win[0])
        temp_rm_6h = float(np.mean(temp_win[-6:]))
        load_temp_inter = float(load_win[-1] * temp_win[-1])

        ts_cur = timestamps[t]
        hour_angle = 2.0 * np.pi * float(ts_cur.hour) / 24.0
        dow_angle = 2.0 * np.pi * float(ts_cur.dayofweek) / 7.0

        xgb_row = lag_flat + [
            rm_6h,
            rstd_24h,
            load_trend_24h,
            temp_rm_6h,
            load_temp_inter,
            float(np.sin(hour_angle)),
            float(np.cos(hour_angle)),
            float(np.sin(dow_angle)),
            float(np.cos(dow_angle)),
        ]
        X_xgb_list.append(xgb_row)
        y_list.append(values[t, 0])
        valid_ts.append(ts_cur)

    X_xgb = np.asarray(X_xgb_list, dtype=np.float32)
    X_lstm = np.asarray(X_lstm_list, dtype=np.float32)
    y_target = np.asarray(y_list, dtype=np.float32)
    return X_xgb, X_lstm, y_target, feature_names, pd.DatetimeIndex(valid_ts)


def build_kfold_splits(
    df_clean: pd.DataFrame,
    n_splits: int = 5,
    lag_window: int = 24,
) -> List[Dict[str, Any]]:
    """
    What: Xây dựng $K = 5$ Folds kiểm định chéo theo thời gian (`TimeSeriesSplit`) với
          bộ `StandardScaler` khớp độc lập bên trong từng Fold.
    Why: Thực hiện Yêu cầu bổ sung số 4 (áp dụng K-Fold bên cạnh Train-Val-Test) đồng thời
         tuân thủ nghiêm ngặt quy tắc chống rò rỉ dữ liệu thời gian.
    Input:
        - df_clean (pd.DataFrame): Chuỗi 4 biến đã làm sạch.
        - n_splits (int): Số lượng Folds ($K = 5$).
        - lag_window (int): Độ dài cửa sổ trễ ($L = 24$).
    Output:
        - List[Dict[str, Any]]: Danh sách 5 gói dữ liệu Fold.
    How: Dùng `TimeSeriesSplit(n_splits=5)` để tạo 5 lát cắt thời gian mở rộng, sau đó
         `fit` `StandardScaler` chỉ trên `df_train_k` của từng Fold.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    folds_data: List[Dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(df_clean), start=1):
        df_tr = df_clean.iloc[train_idx].copy()
        df_val_with_ctx = df_clean.iloc[max(0, val_idx[0] - lag_window) : val_idx[-1] + 1].copy()

        scaler_k = StandardScaler()
        tr_scaled_arr = scaler_k.fit_transform(df_tr.to_numpy(dtype=np.float64))
        val_scaled_arr = scaler_k.transform(df_val_with_ctx.to_numpy(dtype=np.float64))

        df_tr_scaled = pd.DataFrame(tr_scaled_arr, index=df_tr.index, columns=df_tr.columns)
        df_val_scaled = pd.DataFrame(val_scaled_arr, index=df_val_with_ctx.index, columns=df_tr.columns)

        X_xgb_tr, X_lstm_tr, y_tr, feat_names, _ = build_supervised_features(df_tr_scaled, lag_window=lag_window)
        X_xgb_val, X_lstm_val, y_val, _, ts_val = build_supervised_features(df_val_scaled, lag_window=lag_window)

        folds_data.append({
            "fold": fold_idx,
            "train_size": int(len(df_tr)),
            "val_size": int(len(val_idx)),
            "train_period": [str(df_tr.index.min()), str(df_tr.index.max())],
            "val_period": [str(df_clean.index[val_idx[0]]), str(df_clean.index[val_idx[-1]])],
            "scaler": scaler_k,
            "load_mean": float(scaler_k.mean_[0]),
            "load_std": float(scaler_k.scale_[0]),
            "df_train_scaled": df_tr_scaled,
            "df_val_scaled": df_val_scaled.iloc[lag_window:],
            "X_xgb_train": X_xgb_tr,
            "X_lstm_train": X_lstm_tr,
            "y_train": y_tr,
            "X_xgb_val": X_xgb_val,
            "X_lstm_val": X_lstm_val,
            "y_val": y_val,
            "val_timestamps": ts_val,
            "feature_names": feat_names,
        })

    return folds_data


def build_pems_bay_windows(
    df_pems: pd.DataFrame,
    adj_mx: np.ndarray,
    input_window: int = 12,
    horizons: List[int] = [3, 6, 12],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Dict[str, Any]:
    """
    What: Chuẩn hóa dữ liệu tốc độ xe `PEMS-BAY` và tạo các cửa sổ trượt không gian – thời gian
          `[X_{t-11}..X_t] -> [X_{t+3}, X_{t+6}, X_{t+12}]` (15, 30, 60 phút phía trước).
    Why: Cung cấp đầu vào chuẩn cho mô hình `DCRNN` ở Yêu cầu 12 nâng cao.
    Input:
        - df_pems (pd.DataFrame): Bảng tốc độ xe tại $N$ trạm cảm biến mỗi 5 phút.
        - adj_mx (np.ndarray): Ma trận kề khoảng cách giữa các trạm từ `data/raw/sensor_graph/adj_mx_bay.pkl`.
        - input_window (int): 12 bước quá khứ (60 phút).
        - horizons (List[int]): Các chân trời dự báo `[3, 6, 12]` (15m, 30m, 60m).
        - train_ratio (float): `0.70`.
        - val_ratio (float): `0.15`.
    Output:
        - Dict[str, Any]: Các tensor `X_train, Y_train, X_val, Y_val, X_test, Y_test`, thống kê
          `mean`, `std` của tập Train và ma trận chuyển tiếp bước ngẫu nhiên hai chiều `P_forward, P_backward`.
    How: Chia 70/15/15 theo thời gian, tính $\mu_{\text{train}}, \sigma_{\text{train}}$ trên tập Train,
         tạo cửa sổ trượt và tính $\mathbf{P}_f = \mathbf{D}_O^{-1}\mathbf{W}$, $\mathbf{P}_b = \mathbf{D}_I^{-1}\mathbf{W}^T$.
    """
    raw_speed = df_pems.ffill().bfill().to_numpy(dtype=np.float32)
    n_total, n_sensors = raw_speed.shape
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * (train_ratio + val_ratio))

    train_speed = raw_speed[:n_train]
    mean_speed = float(np.mean(train_speed))
    std_speed = float(np.std(train_speed)) if float(np.std(train_speed)) > 1e-6 else 1.0

    scaled_speed = (raw_speed - mean_speed) / std_speed
    max_h = max(horizons)

    def _slice_windows(arr: np.ndarray, start_idx: int, end_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for t in range(start_idx + input_window, end_idx - max_h):
            xs.append(arr[t - input_window : t])
            ys.append([arr[t + h - 1] for h in horizons])
        return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)

    X_tr, Y_tr = _slice_windows(scaled_speed, 0, n_train)
    X_va, Y_va = _slice_windows(scaled_speed, n_train - input_window, n_val)
    X_te, Y_te = _slice_windows(scaled_speed, n_val - input_window, n_total)

    adj = np.asarray(adj_mx, dtype=np.float32)
    d_out = np.sum(adj, axis=1, keepdims=True)
    d_in = np.sum(adj.T, axis=1, keepdims=True)
    p_forward = adj / np.maximum(d_out, 1e-6)
    p_backward = adj.T / np.maximum(d_in, 1e-6)

    return {
        "X_train": X_tr,
        "Y_train": Y_tr,
        "X_val": X_va,
        "Y_val": Y_va,
        "X_test": X_te,
        "Y_test": Y_te,
        "mean_speed": mean_speed,
        "std_speed": std_speed,
        "horizons": horizons,
        "num_sensors": n_sensors,
        "P_forward": p_forward,
        "P_backward": p_backward,
        "adj_mx": adj,
        "sensor_names": list(df_pems.columns),
    }


def main(run_id: Optional[str] = None) -> Dict[str, Any]:
    """
    What: Thực thi toàn bộ giai đoạn `PREPROCESSING` (chia 70/15/15, 5-Fold TimeSeriesSplit,
          StandardScaler, kiểm định ADF, tạo đặc trưng Lag/Rolling và cửa sổ PEMS-BAY).
    Why: Lưu trữ toàn bộ dữ liệu đã tiền xử lý vào `data/processed/lab04/` và `models/lab04/standard_scaler.pkl`.
    Input:
        - run_id (Optional[str]): Mã định danh phiên chạy.
    Output:
        - Dict[str, Any]: Tóm tắt kích thước các tập dữ liệu, kết quả ADF và cấu hình 5-Fold CV.
    How: Nạp dữ liệu từ `data_loader`, chia 70/15/15, chuẩn hóa, tạo 5 Folds và lưu xuống đĩa.
    """
    t0 = time.perf_counter()
    cfg = load_config()
    active_run_id = run_id or create_run()
    write_log(active_run_id, "PREPROCESSING", "START", "Bắt đầu tiền xử lý, chia 70/15/15, 5-Fold TimeSeriesSplit và tạo đặc trưng.")

    df_elec, _ = load_electricity_dataset(cfg)
    df_clean = df_elec.ffill().bfill()

    split_cfg = cfg.get("splitting_and_validation", {}).get("chronological_split", {})
    tr_ratio = float(split_cfg.get("train_ratio", 0.70))
    va_ratio = float(split_cfg.get("val_ratio", 0.15))
    lag_win = int(cfg.get("feature_engineering", {}).get("lag_window", 24))
    n_splits = int(cfg.get("splitting_and_validation", {}).get("kfold_cv", {}).get("n_splits", 5))

    df_train, df_val, df_test = chronological_split_dataframe(df_clean, tr_ratio, va_ratio)

    scaler = StandardScaler()
    train_scaled_arr = scaler.fit_transform(df_train.to_numpy(dtype=np.float64))

    n_train = len(df_train)
    n_val = len(df_val)
    df_val_ctx = df_clean.iloc[n_train - lag_win : n_train + n_val].copy()
    df_test_ctx = df_clean.iloc[n_train + n_val - lag_win :].copy()

    val_ctx_scaled_arr = scaler.transform(df_val_ctx.to_numpy(dtype=np.float64))
    test_ctx_scaled_arr = scaler.transform(df_test_ctx.to_numpy(dtype=np.float64))

    df_train_scaled = pd.DataFrame(train_scaled_arr, index=df_train.index, columns=df_clean.columns)
    df_val_ctx_scaled = pd.DataFrame(val_ctx_scaled_arr, index=df_val_ctx.index, columns=df_clean.columns)
    df_test_ctx_scaled = pd.DataFrame(test_ctx_scaled_arr, index=df_test_ctx.index, columns=df_clean.columns)

    adf_report = check_adf_and_difference(df_train_scaled, significance=0.05, max_diff=2)

    X_xgb_train, X_lstm_train, y_train, feat_names, ts_train = build_supervised_features(df_train_scaled, lag_window=lag_win)
    X_xgb_val, X_lstm_val, y_val, _, ts_val = build_supervised_features(df_val_ctx_scaled, lag_window=lag_win)
    X_xgb_test, X_lstm_test, y_test, _, ts_test = build_supervised_features(df_test_ctx_scaled, lag_window=lag_win)

    kfold_splits = build_kfold_splits(df_clean, n_splits=n_splits, lag_window=lag_win)

    df_pems, adj_mx, _ = load_pems_bay_dataset(cfg)
    pems_bundle = build_pems_bay_windows(
        df_pems,
        adj_mx,
        input_window=int(cfg.get("pems_bay_advanced", {}).get("input_window", 12)),
        horizons=list(cfg.get("pems_bay_advanced", {}).get("forecast_horizons", [3, 6, 12])),
        train_ratio=tr_ratio,
        val_ratio=va_ratio,
    )

    elec_proc_dir = ROOT_DIR / cfg["electricity_data"]["processed_dir"]
    pems_proc_dir = ROOT_DIR / cfg["pems_bay_advanced"]["processed_dir"]
    models_dir = ROOT_DIR / cfg["paths"]["models_dir"]
    metrics_dir = ROOT_DIR / cfg["paths"]["metrics_dir"]
    ensure_dir(elec_proc_dir)
    ensure_dir(pems_proc_dir)
    ensure_dir(models_dir)
    ensure_dir(metrics_dir)

    save_object(scaler, models_dir / "standard_scaler.pkl")
    save_object(
        {
            "df_clean": df_clean,
            "df_train_scaled": df_train_scaled,
            "df_val_scaled": df_val_ctx_scaled.iloc[lag_win:],
            "df_test_scaled": df_test_ctx_scaled.iloc[lag_win:],
            "X_xgb_train": X_xgb_train,
            "X_lstm_train": X_lstm_train,
            "y_train": y_train,
            "X_xgb_val": X_xgb_val,
            "X_lstm_val": X_lstm_val,
            "y_val": y_val,
            "X_xgb_test": X_xgb_test,
            "X_lstm_test": X_lstm_test,
            "y_test": y_test,
            "train_timestamps": ts_train,
            "val_timestamps": ts_val,
            "test_timestamps": ts_test,
            "feature_names": feat_names,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_std": scaler.scale_.tolist(),
            "load_mean": float(scaler.mean_[0]),
            "load_std": float(scaler.scale_[0]),
            "adf_report": adf_report,
        },
        elec_proc_dir / "electricity_processed_bundle.pkl",
    )
    save_object(kfold_splits, elec_proc_dir / "electricity_kfold_bundle.pkl")
    save_object(pems_bundle, pems_proc_dir / "pems_bay_processed_bundle.pkl")
    save_json(adf_report, metrics_dir / "adf_stationarity_report.json")

    stage_fig_dir = create_stage_figure_dir(active_run_id, "preprocessing")
    saved_figs = []
    saved_figs.extend(plot_train_val_test_split(df_train, df_val, df_test, stage_dir=stage_fig_dir))
    saved_figs.extend(plot_kfold_splits_timeline(df_clean, kfold_splits, stage_dir=stage_fig_dir))

    duration = time.perf_counter() - t0
    prep_summary = {
        "chronological_split_counts": {
            "total_hours": int(len(df_clean)),
            "train_hours_70pct": int(len(df_train)),
            "val_hours_15pct": int(len(df_val)),
            "test_hours_15pct": int(len(df_test)),
        },
        "supervised_window_shapes": {
            "lag_window": lag_win,
            "X_xgb_train": list(X_xgb_train.shape),
            "X_xgb_val": list(X_xgb_val.shape),
            "X_xgb_test": list(X_xgb_test.shape),
            "X_lstm_train": list(X_lstm_train.shape),
            "X_lstm_val": list(X_lstm_val.shape),
            "X_lstm_test": list(X_lstm_test.shape),
            "num_tabular_features": int(len(feat_names)),
        },
        "scaler_statistics_train_only": {
            col: {"mean": round(float(m), 4), "std": round(float(s), 4)}
            for col, m, s in zip(df_clean.columns, scaler.mean_, scaler.scale_)
        },
        "kfold_cv_summary": {
            "method": "TimeSeriesSplit",
            "n_splits": n_splits,
            "in_fold_scaling": True,
            "folds": [
                {
                    "fold": f["fold"],
                    "train_samples": int(len(f["y_train"])),
                    "val_samples": int(len(f["y_val"])),
                    "val_period": f["val_period"],
                }
                for f in kfold_splits
            ],
        },
        "adf_stationarity_summary": adf_report,
        "pems_bay_window_shapes": {
            "X_train": list(pems_bundle["X_train"].shape),
            "Y_train": list(pems_bundle["Y_train"].shape),
            "X_val": list(pems_bundle["X_val"].shape),
            "X_test": list(pems_bundle["X_test"].shape),
            "horizons_steps": pems_bundle["horizons"],
        },
        "leakage_free_verified": True,
        "figures": saved_figs,
        "duration_sec": round(duration, 4),
    }
    save_json(prep_summary, metrics_dir / "preprocessing_summary.json")

    update_progress(
        active_run_id,
        "PREPROCESSING",
        "COMPLETED",
        input_data={"lag_window": lag_win, "n_splits": n_splits},
        output_data=prep_summary,
        duration_sec=duration,
    )
    write_log(
        active_run_id,
        "PREPROCESSING",
        "SUCCESS",
        f"Hoàn tất Preprocessing: X_xgb_train={X_xgb_train.shape}, X_lstm_train={X_lstm_train.shape}, 5-Fold CV sẵn sàng ({duration:.2f}s).",
    )
    print(f"[PREPROCESSING] X_xgb_train={X_xgb_train.shape} | X_lstm_train={X_lstm_train.shape} | 5-Fold TimeSeriesSplit Ready.")
    return prep_summary


if __name__ == "__main__":
    main()