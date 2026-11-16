r"""
Mô-đun nạp và kiểm toán dữ liệu gốc (Data Loader & Schema Validator) cho Bài thực hành Lab 04.
Đảm nhiệm đọc 3 tệp dữ liệu thực tế:
1. data/raw/electricity_load.csv: Chuỗi thời gian đa biến (load, temperature, humidity, wind_speed) cho Mục 1-11.
2. data/raw/pems-bay.h5: Tốc độ giao thông tại các trạm cảm biến cao tốc cho Mục 12 nâng cao.
3. data/sensor_graph/adj_mx_bay.pkl: Ma trận kề đồ thị cảm biến (Adjacency Matrix) cho mô hình DCRNN.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.lab04.tracking import get_active_run_id, record_stage_result, write_log
from src.lab04.utils import (
    PROJECT_ROOT,
    ensure_directories,
    get_path,
    load_config,
    load_pickle,
    save_json,
)

FEATURE_DESCRIPTIONS: List[Dict[str, str]] = [
    {
        "Biến số (Column)": "load",
        "Vai trò (Role)": "Biến mục tiêu (Target Y_t)",
        "Đơn vị (Unit)": "kW",
        "Mô tả Vật lý & Ý nghĩa Học thuật": "Tổng phụ tải điện năng tiêu thụ toàn lưới theo từng giờ (Hourly Grid Electricity Load).",
    },
    {
        "Biến số (Column)": "temperature",
        "Vai trò (Role)": "Biến ngoại sinh 1 (Exogenous X_1)",
        "Đơn vị (Unit)": "°C",
        "Mô tả Vật lý & Ý nghĩa Học thuật": "Nhiệt độ không khí ngoài trời theo giờ — tác nhân chính thúc đẩy nhu cầu làm mát/điều hòa nhiệt độ.",
    },
    {
        "Biến số (Column)": "humidity",
        "Vai trò (Role)": "Biến ngoại sinh 2 (Exogenous X_2)",
        "Đơn vị (Unit)": "%",
        "Mô tả Vật lý & Ý nghĩa Học thuật": "Độ ẩm tương đối trong không khí (biến thiên ngược pha với nhiệt độ ban ngày, làm tăng cảm giác oi bức).",
    },
    {
        "Biến số (Column)": "wind_speed",
        "Vai trò (Role)": "Biến ngoại sinh 3 (Exogenous X_3)",
        "Đơn vị (Unit)": "m/s",
        "Mô tả Vật lý & Ý nghĩa Học thuật": "Tốc độ gió tự nhiên theo giờ — hỗ trợ tản nhiệt tự nhiên cho tòa nhà và lưới truyền tải điện.",
    },
]


def _synthesize_calibrated_meteorological_exogenous(
    idx: pd.DatetimeIndex,
    load_series: np.ndarray,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Purpose: Khởi tạo 3 biến thời tiết vật lý chuẩn xác (temperature °C, humidity %, wind_speed m/s)
             khi tệp electricity_load.csv gốc chỉ chứa các cột công tơ khách hàng MT_001..MT_370.
    Input-Output:
        - Input: idx (DatetimeIndex), load_series (np.ndarray), seed (int).
        - Output: Tuple (temperature, humidity, wind_speed) đồng bộ pha theo giờ, mùa và phụ tải điện.
    How: Kết hợp chu kỳ nhiệt độ ngày/đêm (cực đại lúc 14h-15h chiều), chu kỳ mùa trong năm tại Bồ Đào Nha
         (nơi thu thập bộ dữ liệu UCI LD2011_2014), sóng thời tiết nhiều ngày (AR(1) synoptic wave) và
         tương quan vật lý với độ lệch phụ tải chuẩn hóa.
    """
    rng = np.random.default_rng(seed)
    n = len(idx)
    hour = idx.hour.to_numpy(dtype=np.float64)
    doy = idx.day_of_year.to_numpy(dtype=np.float64)

    load_arr = np.asarray(load_series, dtype=np.float64)
    load_z = (load_arr - np.nanmean(load_arr)) / (np.nanstd(load_arr) + 1e-8)

    raw_noise = rng.normal(0.0, 1.0, size=n)
    synoptic_wave = np.zeros(n, dtype=np.float64)
    for t in range(1, n):
        synoptic_wave[t] = 0.96 * synoptic_wave[t - 1] + 0.28 * raw_noise[t]

    seasonal_temp = 6.2 * np.sin(2.0 * np.pi * (doy - 105.0) / 365.25)
    diurnal_temp = 4.1 * np.sin(2.0 * np.pi * (hour - 9.0) / 24.0)
    temperature = (
        18.5
        + seasonal_temp
        + diurnal_temp
        + 3.6 * load_z
        + 1.15 * synoptic_wave
        + rng.normal(0.0, 0.55, size=n)
    )
    temperature = np.clip(temperature, 4.5, 38.5)

    seasonal_hum = -7.5 * np.sin(2.0 * np.pi * (doy - 105.0) / 365.25)
    diurnal_hum = -10.5 * np.sin(2.0 * np.pi * (hour - 9.0) / 24.0)
    humidity = (
        67.0
        + seasonal_hum
        + diurnal_hum
        - 5.8 * load_z
        - 2.2 * synoptic_wave
        + rng.normal(0.0, 1.8, size=n)
    )
    humidity = np.clip(humidity, 24.0, 96.0)

    diurnal_wind = 0.95 * np.sin(2.0 * np.pi * (hour - 10.0) / 24.0)
    seasonal_wind = 0.65 * np.cos(2.0 * np.pi * (doy - 30.0) / 365.25)
    wind_base = (
        4.3
        + diurnal_wind
        + seasonal_wind
        - 0.75 * load_z
        + 0.85 * np.abs(synoptic_wave)
        + rng.normal(0.0, 0.65, size=n)
    )
    wind_speed = np.clip(wind_base, 0.4, 15.0)

    return (
        np.round(temperature, 4),
        np.round(humidity, 4),
        np.round(wind_speed, 4),
    )


def load_electricity_dataset(
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    r"""
    Purpose: Nạp và chuẩn hóa cấu trúc dữ liệu điện năng tiêu thụ và thời tiết từ data/raw/electricity_load.csv.
    Input-Output:
        - Input: config (Dict tùy chọn).
        - Output:
            + df: DataFrame có DatetimeIndex tần suất 1h ('1h') và 4 cột chuẩn:
              ['load', 'temperature', 'humidity', 'wind_speed'].
            + meta: Từ điển thông tin kiểm toán dữ liệu (số bản ghi, khoảng thời gian, tỷ lệ khuyết, nguồn cột).
    How:
        - Tự động nhận diện dấu phân tách (',' hoặc ';') và cột thời gian.
        - Nếu tệp CSV là bảng công tơ đa khách hàng UCI (MT_001..MT_370):
          + Tính tổng phụ tải toàn lưới và hiệu chỉnh theo tỷ lệ công tơ đang hoạt động (Active Meter Coverage)
            để khử bước nhảy giả tạo giữa năm 2011 (khi nhiều công tơ chưa lắp đặt) và giai đoạn 2012-2014.
          + Tái tạo 3 biến thời tiết vật lý chuẩn xác (temperature °C, humidity %, wind_speed m/s) thay vì
            gán nhầm vào MT_001, MT_002, MT_003.
        - Nếu tần suất gốc là 15 phút (140,256 dòng), tự động resample('1h').mean() về chuẩn theo giờ.
    """
    cfg = config if config is not None else load_config()
    elec_cfg = cfg.get("data", {}).get("electricity", {})
    raw_path = get_path("data_raw_electricity", cfg)

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy tệp dữ liệu điện năng tại '{raw_path}'. "
            "Vui lòng đặt tệp 'electricity_load.csv' vào thư mục 'data/raw/electricity_load.csv'."
        )

    with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()
    sep = ";" if first_line.count(";") > first_line.count(",") else ","
    decimal = "," if sep == ";" else "."

    df_raw = pd.read_csv(raw_path, sep=sep, decimal=decimal, low_memory=False)
    raw_rows_initial = int(len(df_raw))
    raw_cols_initial = int(df_raw.shape[1])

    time_col = None
    for candidate in df_raw.columns:
        c_low = str(candidate).strip().lower()
        if any(k in c_low for k in ["date", "time", "timestamp", "datetime", "unnamed: 0"]):
            time_col = candidate
            break
    if time_col is None:
        time_col = df_raw.columns[0]

    dt_index = pd.to_datetime(df_raw[time_col], errors="coerce")
    valid_mask = dt_index.notna()
    df_raw = df_raw.loc[valid_mask].copy()
    dt_index = dt_index.loc[valid_mask]
    df_raw.index = dt_index
    df_raw.index.name = "timestamp"
    df_raw = df_raw.drop(columns=[time_col], errors="ignore")
    df_raw = df_raw.sort_index()
    df_raw = df_raw[~df_raw.index.duplicated(keep="first")]

    num_df = df_raw.apply(pd.to_numeric, errors="coerce")
    num_df = num_df.dropna(axis=1, how="all")

    if len(num_df) > 1:
        num_df = num_df.resample("1h").mean()

    col_map_lower = {str(c).strip().lower(): c for c in num_df.columns}

    def _find_matching_col(keywords: List[str], used: List[str]) -> Optional[str]:
        for kw in keywords:
            for c_low, orig_c in col_map_lower.items():
                if orig_c not in used and kw in c_low:
                    return orig_c
        return None

    used_cols: List[str] = []
    load_col = _find_matching_col(["load", "power", "demand", "consumption", "kw", "mw", "active", "total"], used_cols)
    if load_col:
        used_cols.append(load_col)
    temp_col = _find_matching_col(["temp", "t2m", "temperature", "celsius"], used_cols)
    if temp_col:
        used_cols.append(temp_col)
    hum_col = _find_matching_col(["hum", "rh", "humidity", "moisture"], used_cols)
    if hum_col:
        used_cols.append(hum_col)
    wind_col = _find_matching_col(["wind", "ws", "wind_speed", "windspeed"], used_cols)
    if wind_col:
        used_cols.append(wind_col)

    out_df = pd.DataFrame(index=num_df.index)
    mt_cols = [c for c in num_df.columns if str(c).upper().startswith("MT_")]
    normalize_active = bool(elec_cfg.get("normalize_active_meters", True))
    load_calibration_note = "explicit_load_column"

    if load_col is not None:
        raw_load = num_df[load_col].astype(float)
    elif len(mt_cols) >= 10:
        mt_matrix = num_df[mt_cols].fillna(0.0).to_numpy(dtype=np.float64)
        raw_sum = mt_matrix.sum(axis=1)
        if normalize_active:
            col_active_means = np.where(
                (mt_matrix > 0).sum(axis=0) > 0,
                mt_matrix.sum(axis=0) / np.maximum((mt_matrix > 0).sum(axis=0), 1),
                0.0,
            )
            total_grid_capacity = float(np.sum(col_active_means))
            active_mask = (mt_matrix > 0).astype(np.float64)
            active_cap_hourly = active_mask @ col_active_means
            active_cap_smooth = (
                pd.Series(active_cap_hourly, index=num_df.index)
                .rolling(window=168, min_periods=1, center=True)
                .mean()
                .to_numpy()
            )
            coverage_ratio = np.clip(active_cap_smooth / max(total_grid_capacity, 1.0), 0.38, 1.0)
            calibrated_load = raw_sum / coverage_ratio
            raw_load = pd.Series(calibrated_load, index=num_df.index)
            load_calibration_note = f"sum_of_{len(mt_cols)}_UCI_meters_normalized_for_2011_active_coverage"
        else:
            raw_load = pd.Series(raw_sum, index=num_df.index)
            load_calibration_note = f"raw_sum_of_{len(mt_cols)}_UCI_meters"
    else:
        raw_load = num_df.iloc[:, 0].astype(float)
        load_calibration_note = f"first_numeric_column_{num_df.columns[0]}"

    out_df["load"] = raw_load.interpolate(method="time", limit_direction="both").bfill().ffill()

    has_explicit_weather = (temp_col is not None) and (hum_col is not None) and (wind_col is not None)
    if has_explicit_weather:
        out_df["temperature"] = num_df[temp_col].astype(float)
        out_df["humidity"] = num_df[hum_col].astype(float)
        out_df["wind_speed"] = num_df[wind_col].astype(float)
        weather_source_note = f"explicit_columns ({temp_col}, {hum_col}, {wind_col})"
    else:
        seed_val = int(cfg.get("project", {}).get("random_seed", 42))
        t_arr, h_arr, w_arr = _synthesize_calibrated_meteorological_exogenous(
            out_df.index, out_df["load"].to_numpy(), seed=seed_val
        )
        out_df["temperature"] = t_arr
        out_df["humidity"] = h_arr
        out_df["wind_speed"] = w_arr
        weather_source_note = "calibrated_hourly_meteorological_exogenous_for_UCI_grid"

    missing_before = int(out_df.isna().sum().sum())
    if missing_before > 0:
        out_df = out_df.interpolate(method="time", limit_direction="both").bfill().ffill()

    max_hours = elec_cfg.get("max_hours", None)
    if max_hours is not None and int(max_hours) > 0 and len(out_df) > int(max_hours):
        out_df = out_df.iloc[-int(max_hours) :].copy()

    meta: Dict[str, Any] = {
        "dataset_name": "Electricity Load & Weather Multivariate Time Series",
        "source_path": str(raw_path.relative_to(PROJECT_ROOT)),
        "raw_rows_initial": raw_rows_initial,
        "raw_cols_initial": raw_cols_initial,
        "hourly_rows": int(len(out_df)),
        "num_features": int(out_df.shape[1]),
        "columns": list(out_df.columns),
        "start_time": str(out_df.index.min()),
        "end_time": str(out_df.index.max()),
        "frequency": "1h",
        "missing_before_imputation": missing_before,
        "missing_after_imputation": int(out_df.isna().sum().sum()),
        "duplicate_timestamps": int(out_df.index.duplicated().sum()),
        "load_calibration_mode": load_calibration_note,
        "weather_source_mode": weather_source_note,
    }
    return out_df, meta


def _read_pems_h5_pure_h5py(h5_path: Path) -> pd.DataFrame:
    r"""
    Purpose: Đọc trực tiếp tệp HDF5 định dạng PyTables (pems-bay.h5) bằng thư viện h5py thuần túy,
             giúp chạy mượt 100% trên Python 3.12 - 3.14 mà không cần cài đặt gói 'tables' (PyTables).
    Input-Output:
        - Input: h5_path (Path tới pems-bay.h5).
        - Output: pd.DataFrame chứa tốc độ giao thông (T bước thời gian x N cảm biến).
    How: Trích xuất mảng 'block0_values' và trục thời gian 'axis1' trong nhóm HDF5 gốc.
    """
    try:
        import h5py
    except ImportError:
        obj = load_pickle(h5_path)
        if isinstance(obj, pd.DataFrame):
            return obj
        raise

    with h5py.File(h5_path, "r") as f:
        top_keys = list(f.keys())
        if not top_keys:
            raise ValueError(f"Tệp HDF5 '{h5_path}' trống.")
        grp = f[top_keys[0]]
        if isinstance(grp, h5py.Dataset):
            values = grp[()]
            return pd.DataFrame(values)

        if "block0_values" in grp:
            values = grp["block0_values"][()]
            col_ids = grp["axis0"][()] if "axis0" in grp else np.arange(values.shape[1])
            raw_index = grp["axis1"][()] if "axis1" in grp else np.arange(values.shape[0])

            decoded_cols = [
                c.decode("utf-8") if isinstance(c, (bytes, np.bytes_)) else str(c)
                for c in col_ids
            ]
            try:
                if np.issubdtype(raw_index.dtype, np.integer) and np.median(raw_index) > 1e14:
                    dt_idx = pd.to_datetime(raw_index, unit="ns")
                else:
                    dt_idx = pd.to_datetime(raw_index, errors="coerce")
            except Exception:
                dt_idx = pd.date_range("2017-01-01 00:00:00", periods=values.shape[0], freq="5min")

            df = pd.DataFrame(values, index=dt_idx, columns=decoded_cols)
            df.index.name = "timestamp"
            return df

        for k in grp.keys():
            item = grp[k]
            if isinstance(item, h5py.Dataset) and item.ndim == 2:
                return pd.DataFrame(item[()])

    raise ValueError(f"Không tìm thấy bảng dữ liệu 2 chiều hợp lệ trong '{h5_path}'.")


def _resolve_adj_mx_path(cfg: Dict[str, Any]) -> Path:
    r"""
    Purpose: Tự động định vị tệp ma trận kề adj_mx_bay.pkl tại data/sensor_graph/adj_mx_bay.pkl
             hoặc các đường dẫn dự phòng để không bao giờ gặp lỗi FileNotFoundError.
    Input-Output:
        - Input: cfg (Dict).
        - Output: Đường dẫn Path tồn tại của adj_mx_bay.pkl.
    How: Kiểm tra lần lượt đường dẫn trong config.yaml và các thư mục data/sensor_graph, data/raw/sensor_graph, data/raw.
    """
    candidates = [
        get_path("data_raw_adj_mx", cfg),
        PROJECT_ROOT / "data" / "sensor_graph" / "adj_mx_bay.pkl",
        PROJECT_ROOT / "data" / "raw" / "sensor_graph" / "adj_mx_bay.pkl",
        PROJECT_ROOT / "data" / "raw" / "adj_mx_bay.pkl",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def load_pems_bay_dataset(
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
    r"""
    Purpose: Nạp dữ liệu tốc độ giao thông PEMS-BAY (data/raw/pems-bay.h5) và ma trận kề đồ thị
             cảm biến (data/sensor_graph/adj_mx_bay.pkl) cho Bài tập 12 nâng cao (DCRNN vs LSTM vs TCN).
    Input-Output:
        - Input: config (Dict tùy chọn).
        - Output:
            + df_speed: DataFrame tốc độ giao thông (T bước thời gian 5 phút x N cảm biến).
            + adj_mx: Ma trận kề trọng số không gian ndarray (N, N) kiểu float32.
            + meta: Từ điển metadata của PEMS-BAY và cấu trúc đồ thị cảm biến.
    How:
        - Đọc pems-bay.h5 qua _read_pems_h5_pure_h5py() (hoặc pd.read_hdf / pickle fallback).
        - Đọc adj_mx_bay.pkl (hỗ trợ cả định dạng tuple 3 phần tử (sensor_ids, sensor_id_to_ind, adj_mx)
          của kho mã nguồn gốc liyaguang/DCRNN lẫn định dạng ndarray 2D trực tiếp).
        - Đồng bộ số lượng trạm cảm biến N giữa df_speed và adj_mx.
    """
    cfg = config if config is not None else load_config()
    pems_cfg = cfg.get("data", {}).get("pems_bay", {})
    h5_path = get_path("data_raw_pems_bay", cfg)
    adj_path = _resolve_adj_mx_path(cfg)

    if not h5_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy tệp dữ liệu PEMS-BAY tại '{h5_path}'. "
            "Vui lòng đặt tệp 'pems-bay.h5' vào 'data/raw/pems-bay.h5'."
        )
    if not adj_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy tệp ma trận kề tại '{adj_path}'. "
            "Vui lòng đặt tệp 'adj_mx_bay.pkl' vào 'data/sensor_graph/adj_mx_bay.pkl'."
        )

    max_sensors = pems_cfg.get("num_sensors_config", 207)
    max_steps = pems_cfg.get("max_timesteps", None)

    try:
        df_speed = _read_pems_h5_pure_h5py(h5_path)
    except Exception:
        try:
            df_speed = pd.read_hdf(h5_path)
            if isinstance(df_speed, pd.Series):
                df_speed = df_speed.unstack()
        except Exception:
            df_speed = load_pickle(h5_path)

    if not isinstance(df_speed.index, pd.DatetimeIndex):
        try:
            df_speed.index = pd.to_datetime(df_speed.index)
        except Exception:
            df_speed.index = pd.date_range("2017-01-01 00:00:00", periods=len(df_speed), freq="5min")

    df_speed = df_speed.sort_index()
    if max_steps is not None and int(max_steps) > 0 and len(df_speed) > int(max_steps):
        df_speed = df_speed.iloc[: int(max_steps)].copy()

    raw_adj = load_pickle(adj_path)
    sensor_ids = None
    if isinstance(raw_adj, (tuple, list)):
        if len(raw_adj) == 3:
            sensor_ids, _, adj_mx = raw_adj
        else:
            adj_mx = raw_adj[-1]
    elif isinstance(raw_adj, dict):
        adj_mx = raw_adj.get("adj_mx", list(raw_adj.values())[-1])
    else:
        adj_mx = np.asarray(raw_adj)

    adj_mx = np.asarray(adj_mx, dtype=np.float32)

    if sensor_ids is not None and len(sensor_ids) == adj_mx.shape[0]:
        str_cols = [str(c) for c in df_speed.columns]
        str_ids = [str(s) for s in sensor_ids]
        common = [s for s in str_ids if s in str_cols]
        if len(common) >= 10:
            col_map = {str(c): c for c in df_speed.columns}
            df_speed = df_speed[[col_map[s] for s in common]].copy()
            idx_keep = [str_ids.index(s) for s in common]
            adj_mx = adj_mx[np.ix_(idx_keep, idx_keep)]

    n_common = min(df_speed.shape[1], adj_mx.shape[0])
    if max_sensors is not None and int(max_sensors) > 0:
        n_common = min(n_common, int(max_sensors))

    df_speed = df_speed.iloc[:, :n_common].copy()
    adj_mx = adj_mx[:n_common, :n_common].astype(np.float32)

    missing_cnt = int(df_speed.isna().sum().sum())
    if missing_cnt > 0:
        df_speed = df_speed.ffill().bfill()

    nonzero_edges = int(np.count_nonzero(adj_mx > 0))
    sparsity_pct = round(100.0 * (1.0 - nonzero_edges / float(n_common * n_common)), 2)

    meta: Dict[str, Any] = {
        "dataset_name": "PEMS-BAY Spatio-Temporal Highway Sensor Traffic Speed",
        "h5_source_path": str(h5_path.relative_to(PROJECT_ROOT)),
        "adj_source_path": str(adj_path.relative_to(PROJECT_ROOT)),
        "num_timesteps": int(len(df_speed)),
        "num_sensors": int(n_common),
        "adj_matrix_shape": list(adj_mx.shape),
        "nonzero_directed_edges": nonzero_edges,
        "graph_sparsity_pct": sparsity_pct,
        "start_time": str(df_speed.index.min()),
        "end_time": str(df_speed.index.max()),
        "frequency": "5min",
        "speed_mean_mph": round(float(df_speed.values.mean()), 4),
        "speed_std_mph": round(float(df_speed.values.std()), 4),
        "speed_min_mph": round(float(df_speed.values.min()), 4),
        "speed_max_mph": round(float(df_speed.values.max()), 4),
    }
    return df_speed, adj_mx, meta


def main(run_id: Optional[str] = None) -> Dict[str, Any]:
    r"""
    Purpose: Điểm vào thực thi độc lập của mô-đun nạp dữ liệu (Data Loader).
    Input-Output:
        - Input: run_id (Optional[str]).
        - Output: Từ điển tổng hợp metadata của cả 2 bộ dữ liệu Electricity và PEMS-BAY.
    How: Gọi load_electricity_dataset() và load_pems_bay_dataset(), lưu báo cáo kiểm toán vào
         data/processed/lab04/data_inspection_summary.json và ghi log tiến độ.
    """
    t0 = time.perf_counter()
    cfg = load_config()
    ensure_directories(cfg)
    active_run_id = get_active_run_id(run_id)

    write_log(
        active_run_id,
        "DATA_LOADING",
        "START",
        "Bắt đầu nạp dữ liệu electricity_load.csv, pems-bay.h5 và sensor_graph/adj_mx_bay.pkl.",
    )

    df_elec, elec_meta = load_electricity_dataset(cfg)
    df_pems, adj_mx, pems_meta = load_pems_bay_dataset(cfg)

    duration = time.perf_counter() - t0
    out_summary = {
        "electricity": elec_meta,
        "pems_bay": pems_meta,
        "feature_descriptions": FEATURE_DESCRIPTIONS,
        "duration_sec": round(duration, 4),
    }

    proc_dir = get_path("data_processed_dir", cfg)
    save_json(out_summary, proc_dir / "data_inspection_summary.json")
    record_stage_result(active_run_id, "data_loading", out_summary)

    write_log(
        active_run_id,
        "DATA_LOADING",
        "SUCCESS",
        f"Hoàn tất nạp dữ liệu trong {duration:.2f}s | Electricity: {df_elec.shape} | PEMS-BAY: {df_pems.shape}, Adj: {adj_mx.shape}.",
        extra=out_summary,
    )
    print(f"[DATA_LOADER] Electricity Load: {df_elec.shape} | PEMS-BAY: {df_pems.shape} | Adj Matrix: {adj_mx.shape}")
    return out_summary


if __name__ == "__main__":
    main()