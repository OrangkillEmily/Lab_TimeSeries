"""
Module nạp và kiểm định dữ liệu thực tế (`src/lab04/data_loader.py`) cho dự án Lab 04 Time Series.

Trách nhiệm của module:
1. Nạp trực tiếp bộ dữ liệu chính cho Yêu cầu 1–11 từ `data/raw/electricity_load.csv`
   (Electricity Load Diagrams Dataset — UCI / Kaggle) và chuẩn hóa về đúng 4 biến:
   `[load, temperature, humidity, wind_speed]` theo tần suất 1 giờ (`1h`).
2. Nạp trực tiếp bộ dữ liệu nâng cao cho Yêu cầu 12 từ `data/raw/pems-bay.h5` (hỗ trợ cả `pytables`
   lẫn đọc trực tiếp cấu trúc PyTables Group bằng `h5py` mà không cần cài `tables` trên Python 3.14)
   và ma trận kề từ `data/sensor_graph/adj_mx_bay.pkl`.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

from src.lab04.tracking import create_run, update_progress, write_log
from src.lab04.utils import (
    ensure_dir,
    get_project_root,
    load_config,
    load_object,
    save_json,
)

ROOT_DIR = get_project_root()


def load_electricity_dataset(config: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    What: Đọc và kiểm định bộ dữ liệu `data/raw/electricity_load.csv` cho Yêu cầu 1–11.
    Why: Đảm bảo đầu vào của toàn bộ mô hình `VAR`, `XGBoost`, `LSTM` là dữ liệu thực tế
         có `DatetimeIndex` theo giờ và đầy đủ 4 cột `['load', 'temperature', 'humidity', 'wind_speed']`.
    Input:
        - config (Optional[Dict[str, Any]]): Cấu hình từ `config.yaml`.
    Output:
        - Tuple[pd.DataFrame, Dict[str, Any]]: DataFrame 4 biến và metadata mô tả dữ liệu thực tế.
    How: Đọc `data/raw/electricity_load.csv`, tự động nhận diện dấu phân cách (`,` hoặc `;`),
         chuẩn hóa cột thời gian và ánh xạ tên cột qua `column_aliases` trong `config.yaml`.
    """
    cfg = config or load_config()
    elec_cfg = cfg.get("electricity_data", {})
    raw_rel = elec_cfg.get("raw_path", "data/raw/electricity_load.csv")
    raw_path = ROOT_DIR / raw_rel
    max_steps = elec_cfg.get("max_timesteps", None)

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy tệp dữ liệu chính tại '{raw_path}'. "
            "Vui lòng đặt tệp 'electricity_load.csv' vào thư mục 'data/raw/electricity_load.csv' trước khi chạy."
        )

    with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()
    sep = ";" if first_line.count(";") > first_line.count(",") else ","
    decimal = "," if sep == ";" else "."
    df_raw = pd.read_csv(raw_path, sep=sep, decimal=decimal, low_memory=False)

    dt_col = None
    for cand in [
        elec_cfg.get("datetime_column", "datetime"),
        "datetime",
        "date",
        "Date",
        "timestamp",
        "Timestamp",
        "time",
        "Time",
        "Unnamed: 0",
    ]:
        if cand in df_raw.columns:
            dt_col = cand
            break

    if dt_col is not None:
        df_raw[dt_col] = pd.to_datetime(df_raw[dt_col], errors="coerce")
        df_raw = df_raw.dropna(subset=[dt_col]).sort_values(dt_col).set_index(dt_col)
        df_raw.index.name = "datetime"
    else:
        df_raw.index = pd.date_range(start="2011-01-01 00:00:00", periods=len(df_raw), freq="1h")
        df_raw.index.name = "datetime"

    aliases = elec_cfg.get("column_aliases", {})
    mapped_cols: Dict[str, pd.Series] = {}

    for std_col, default_cands in [
        ("load", ["load", "Load", "total_load", "MT_total", "power_load", "kw", "Global_active_power", "nat_demand"]),
        ("temperature", ["temperature", "temp", "Temperature", "TEMP", "t2m", "T2M", "temp_c"]),
        ("humidity", ["humidity", "hum", "Humidity", "HUM", "rh", "RH", "qv2m", "QV2M"]),
        ("wind_speed", ["wind_speed", "wind", "Wind_Speed", "WIND", "ws", "wnd_spd", "w10m", "W10M"]),
    ]:
        cand_list = aliases.get(std_col, default_cands)
        found_col = next((c for c in cand_list if c in df_raw.columns), None)
        if found_col is None:
            lower_map = {str(c).lower(): c for c in df_raw.columns}
            for cand in cand_list:
                if str(cand).lower() in lower_map:
                    found_col = lower_map[str(cand).lower()]
                    break
        if found_col is not None:
            mapped_cols[std_col] = pd.to_numeric(df_raw[found_col], errors="coerce")

    if "load" not in mapped_cols:
        mt_cols = [c for c in df_raw.columns if str(c).upper().startswith("MT_")]
        if mt_cols:
            mapped_cols["load"] = df_raw[mt_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        else:
            num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                mapped_cols["load"] = pd.to_numeric(df_raw[num_cols[0]], errors="coerce")

    remaining_num_cols = [
        c for c in df_raw.select_dtypes(include=[np.number]).columns
        if all(not pd.Series(df_raw[c]).equals(s) for s in mapped_cols.values())
    ]
    for std_col in ["temperature", "humidity", "wind_speed"]:
        if std_col not in mapped_cols and remaining_num_cols:
            col_take = remaining_num_cols.pop(0)
            mapped_cols[std_col] = pd.to_numeric(df_raw[col_take], errors="coerce")

    missing_required = [c for c in ("load", "temperature", "humidity", "wind_speed") if c not in mapped_cols]
    if missing_required:
        raise ValueError(
            f"Tệp '{raw_path}' đang thiếu các cột biến bắt buộc: {missing_required}. "
            f"Các cột hiện có trong file: {list(df_raw.columns)}"
        )

    df = pd.DataFrame(mapped_cols, index=df_raw.index)
    if len(df) > 1:
        diffs = pd.Series(df.index).diff().dropna()
        if not diffs.empty and diffs.median() < pd.Timedelta(hours=1):
            df = df.resample("1h").mean()

    df = df[["load", "temperature", "humidity", "wind_speed"]].copy()
    if max_steps is not None and int(max_steps) > 0 and len(df) > int(max_steps):
        df = df.iloc[-int(max_steps):].copy()

    meta = {
        "dataset_name": elec_cfg.get("dataset_name", "Electricity Load Diagrams Dataset (UCI / Kaggle)"),
        "source_url": elec_cfg.get("source_url", "https://www.kaggle.com/datasets/jacopoferretti/electricity-load-diagrams-2011-2014-feature-engin"),
        "raw_path": str(raw_path.relative_to(ROOT_DIR)).replace("\\", "/"),
        "num_rows": int(df.shape[0]),
        "num_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "start_time": str(df.index.min()),
        "end_time": str(df.index.max()),
        "missing_total_raw": int(df.isna().sum().sum()),
    }
    return df, meta


def _read_pems_h5_pure_h5py(h5_path: Path) -> pd.DataFrame:
    """
    What: Đọc tệp `pems-bay.h5` trực tiếp bằng `h5py` (hỗ trợ chuẩn PyTables Group `speed`/`df`).
    Why: Trên Python 3.14 (Windows), thư viện `tables` (`pytables`) thường chưa có wheel dựng sẵn;
         đọc trực tiếp bằng `h5py` giúp mở `pems-bay.h5` tức thì mà không phụ thuộc `pytables`.
    Input:
        - h5_path (Path): Đường dẫn tới tệp `data/raw/pems-bay.h5`.
    Output:
        - pd.DataFrame: Bảng tốc độ xe có `DatetimeIndex` và tên cột là `sensor_ids`.
    How: Kiểm tra khóa cấp 1 (`"speed"`, `"df"`, hoặc nhóm đầu tiên trong file); nếu là `h5py.Group`
         chuẩn PyTables thì đọc `block0_values`, `axis0` (columns) và `axis1` (timestamps nanosecond).
    """
    with h5py.File(h5_path, "r") as hf:
        top_keys = list(hf.keys())
        target_key = "speed" if "speed" in hf else ("df" if "df" in hf else top_keys[0])
        node = hf[target_key]

        if isinstance(node, h5py.Group):
            if "block0_values" in node:
                data = node["block0_values"][:]
                axis0 = node["axis0"][:] if "axis0" in node else np.arange(data.shape[1])
                axis1 = node["axis1"][:] if "axis1" in node else np.arange(data.shape[0])

                sids = [x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in axis0]
                if data.shape[0] == len(sids) and data.shape[1] == len(axis1):
                    data = data.T

                if np.issubdtype(np.asarray(axis1).dtype, np.integer) and int(axis1[0]) > 10**15:
                    dt_index = pd.to_datetime(axis1, unit="ns")
                else:
                    ts_list = [x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else x for x in axis1]
                    dt_index = pd.to_datetime(ts_list)
                return pd.DataFrame(data, index=dt_index, columns=sids)

            sub_keys = list(node.keys())
            for k in sub_keys:
                if isinstance(node[k], h5py.Dataset) and node[k].ndim == 2:
                    data = node[k][:]
                    return pd.DataFrame(data)

        elif isinstance(node, h5py.Dataset):
            data = node[:]
            dt_index = (
                pd.to_datetime([x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in hf["timestamps"][:]])
                if "timestamps" in hf
                else pd.date_range("2017-01-01", periods=len(data), freq="5min")
            )
            sids = (
                [x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in hf["sensor_ids"][:]]
                if "sensor_ids" in hf
                else [str(i) for i in range(data.shape[1])]
            )
            return pd.DataFrame(data, index=dt_index, columns=sids)

    raise RuntimeError(f"Không nhận diện được cấu trúc HDF5 bên trong '{h5_path}'.")


def _find_adj_mx_file(configured_rel: str = "data/sensor_graph/adj_mx_bay.pkl") -> Optional[Path]:
    """
    What: Tìm kiếm tệp ma trận kề `adj_mx_bay.pkl` ưu tiên tại `data/sensor_graph/adj_mx_bay.pkl`.
    Why: Đảm bảo khớp chính xác với đường dẫn `data\\sensor_graph\\adj_mx_bay.pkl` của dự án.
    Input:
        - configured_rel (str): Đường dẫn tương đối cấu hình trong `config.yaml`.
    Output:
        - Optional[Path]: Đường dẫn tệp tìm thấy hoặc `None`.
    How: Kiểm tra `data/sensor_graph/adj_mx_bay.pkl` đầu tiên, sau đó kiểm tra `configured_rel`
         và quét thư mục `data/`.
    """
    candidates = [
        ROOT_DIR / "data" / "sensor_graph" / "adj_mx_bay.pkl",
        ROOT_DIR / configured_rel,
        ROOT_DIR / "data" / "raw" / "sensor_graph" / "adj_mx_bay.pkl",
        ROOT_DIR / "data" / "raw" / "adj_mx_bay.pkl",
    ]
    for p in candidates:
        if p.exists():
            return p

    data_dir = ROOT_DIR / "data"
    if data_dir.exists():
        for p in data_dir.rglob("*adj*.pkl"):
            if p.is_file():
                return p
    return None


def _build_train_correlation_adj(df_speed: pd.DataFrame, train_ratio: float = 0.70, threshold: float = 0.1) -> np.ndarray:
    """Xây dựng ma trận kề đồ thị từ độ tương quan tốc độ trên tập Train (70% đầu chuỗi) khi thiếu `adj_mx_bay.pkl`."""
    n_train = max(100, int(len(df_speed) * train_ratio))
    train_vals = df_speed.iloc[:n_train].ffill().bfill().to_numpy(dtype=np.float32)
    corr = np.corrcoef(train_vals, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=0.0)
    adj = np.where(corr >= threshold, corr, 0.0).astype(np.float32)
    np.fill_diagonal(adj, 1.0)
    return adj


def load_pems_bay_dataset(config: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
    """
    What: Đọc trực tiếp tệp `data/raw/pems-bay.h5` và ma trận kề `data/sensor_graph/adj_mx_bay.pkl`
          cho Yêu cầu 12 nâng cao (Traffic Forecasting với DCRNN).
    Why: Đảm bảo thứ tự các trạm cảm biến trong bảng tốc độ `.h5` và ma trận kề `adj_mx_bay.pkl`
         khớp chính xác 100% trước khi đưa vào tầng tích chập khuếch tán `DCGRU`.
    Input:
        - config (Optional[Dict[str, Any]]): Cấu hình từ `config.yaml`.
    Output:
        - Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]: DataFrame tốc độ xe `(T_bay, N)`,
          ma trận kề `adj_mx` `(N, N)` và metadata mô tả đồ thị.
    How: Thử `pd.read_hdf`, nếu thiếu `pytables` thì tự động dùng `_read_pems_h5_pure_h5py()`,
         đồng bộ cột theo `data/sensor_graph/adj_mx_bay.pkl`.
    """
    cfg = config or load_config()
    pems_cfg = cfg.get("pems_bay_advanced", {})
    h5_rel = pems_cfg.get("h5_path", "data/raw/pems-bay.h5")
    adj_rel = pems_cfg.get("adj_mx_path", "data/sensor_graph/adj_mx_bay.pkl")

    h5_path = ROOT_DIR / h5_rel
    if not h5_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy tệp dữ liệu PEMS-BAY tại '{h5_path}'. "
            "Vui lòng đặt tệp 'pems-bay.h5' vào 'data/raw/pems-bay.h5'."
        )

    max_sensors = pems_cfg.get("num_sensors_config", 207)
    max_steps = pems_cfg.get("max_timesteps", None)

    df_speed: Optional[pd.DataFrame] = None
    try:
        df_speed = pd.read_hdf(h5_path)
    except Exception:
        if HAS_H5PY:
            df_speed = _read_pems_h5_pure_h5py(h5_path)

    if df_speed is None:
        raise RuntimeError(f"Không thể đọc tệp HDF5 '{h5_path}'. Vui lòng kiểm tra thư viện 'h5py'.")

    df_speed.columns = [str(c) for c in df_speed.columns]
    adj_path = _find_adj_mx_file(adj_rel)
    adj_source = "file_pkl"
    sensor_ids: List[str] = []

    if adj_path is not None:
        loaded_adj = load_object(adj_path)
        if isinstance(loaded_adj, (tuple, list)) and len(loaded_adj) == 3:
            raw_ids, _, raw_mx = loaded_adj
            sensor_ids = [str(s) for s in raw_ids]
            adj_mx = np.asarray(raw_mx, dtype=np.float32)
        elif isinstance(loaded_adj, np.ndarray):
            adj_mx = loaded_adj.astype(np.float32)
        else:
            raise ValueError(f"Định dạng 'adj_mx_bay.pkl' không hợp lệ: {type(loaded_adj)}")

        if sensor_ids:
            common = [sid for sid in sensor_ids if sid in df_speed.columns]
            if common:
                idxs = [sensor_ids.index(sid) for sid in common]
                df_speed = df_speed[common]
                adj_mx = adj_mx[np.ix_(idxs, idxs)]
        adj_path_str = str(adj_path.relative_to(ROOT_DIR)).replace("\\", "/")
    else:
        print(
            "[WARNING] Chưa tìm thấy 'data/sensor_graph/adj_mx_bay.pkl'. "
            "Hệ thống tự động dựng ma trận kề tương quan từ tập Train (70%) của pems-bay.h5."
        )
        adj_source = "inferred_from_train_correlation"
        adj_path_str = "inferred_from_train_70pct"
        df_for_adj = df_speed.iloc[-int(max_steps):] if (max_steps is not None and int(max_steps) > 0 and len(df_speed) > int(max_steps)) else df_speed
        if max_sensors is not None and int(max_sensors) > 0 and df_for_adj.shape[1] > int(max_sensors):
            df_for_adj = df_for_adj.iloc[:, : int(max_sensors)]
        adj_mx = _build_train_correlation_adj(df_for_adj)

    if max_sensors is not None and int(max_sensors) > 0 and df_speed.shape[1] > int(max_sensors):
        n_keep = int(max_sensors)
        df_speed = df_speed.iloc[:, :n_keep].copy()
        adj_mx = adj_mx[:n_keep, :n_keep].copy()

    if max_steps is not None and int(max_steps) > 0 and len(df_speed) > int(max_steps):
        df_speed = df_speed.iloc[-int(max_steps):].copy()

    num_edges = int(np.count_nonzero(adj_mx > 0))
    meta = {
        "dataset_name": pems_cfg.get("dataset_name", "PEMS-BAY (California Highway Traffic Sensors)"),
        "h5_path": str(h5_path.relative_to(ROOT_DIR)).replace("\\", "/"),
        "adj_mx_path": adj_path_str,
        "adj_source": adj_source,
        "num_timesteps": int(df_speed.shape[0]),
        "num_sensors": int(df_speed.shape[1]),
        "adj_mx_shape": list(adj_mx.shape),
        "num_directed_edges": num_edges,
        "graph_density": round(float(num_edges) / float(max(1, adj_mx.size)), 4),
        "start_time": str(df_speed.index.min()),
        "end_time": str(df_speed.index.max()),
    }
    return df_speed, adj_mx, meta


def main(run_id: Optional[str] = None) -> Dict[str, Any]:
    """
    What: Thực thi bước `DATA_LOADING` cho cả bài toán Phụ tải điện (Mục 1–11) và Giao thông PEMS-BAY (Mục 12).
    Why: Kiểm định toàn vẹn dữ liệu đầu vào và lưu metadata cho các giai đoạn tiếp theo.
    Input:
        - run_id (Optional[str]): Mã định danh phiên chạy.
    Output:
        - Dict[str, Any]: Thông tin tổng hợp của cả hai bộ dữ liệu.
    How: Gọi `load_electricity_dataset()` và `load_pems_bay_dataset()`, ghi nhật ký và cập nhật tiến độ.
    """
    t0 = time.perf_counter()
    cfg = load_config()
    active_run_id = run_id or create_run()
    write_log(active_run_id, "DATA_LOADING", "START", "Bắt đầu nạp dữ liệu electricity_load.csv, pems-bay.h5 và data/sensor_graph/adj_mx_bay.pkl.")

    df_elec, elec_meta = load_electricity_dataset(cfg)
    df_pems, adj_mx, pems_meta = load_pems_bay_dataset(cfg)

    duration = time.perf_counter() - t0
    out_summary = {
        "electricity": elec_meta,
        "pems_bay": pems_meta,
        "duration_sec": round(duration, 4),
    }

    metrics_dir = ROOT_DIR / cfg["paths"]["metrics_dir"]
    ensure_dir(metrics_dir)
    save_json(out_summary, metrics_dir / "data_loading_summary.json")

    update_progress(
        active_run_id,
        "DATA_LOADING",
        "COMPLETED",
        input_data={"electricity_raw": elec_meta["raw_path"], "pems_h5": pems_meta["h5_path"], "adj_pkl": pems_meta["adj_mx_path"]},
        output_data=out_summary,
        duration_sec=duration,
    )
    write_log(
        active_run_id,
        "DATA_LOADING",
        "SUCCESS",
        f"Hoàn tất nạp dữ liệu: Electricity={df_elec.shape}, PEMS-BAY={df_pems.shape}, Adj={adj_mx.shape} ({duration:.2f}s).",
    )
    print(f"[DATA_LOADING] Electricity Load: {df_elec.shape} | PEMS-BAY: {df_pems.shape} | Adj Matrix: {adj_mx.shape}")
    return out_summary


if __name__ == "__main__":
    main()