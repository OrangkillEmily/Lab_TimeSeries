r"""
Mô-đun Kiểm tra & Xử lý Chất lượng Dữ liệu (Bài tập 4: Missing Values & Outliers) cho Lab 01.
Tái lập trọn vẹn kịch bản giả lập từ Lab01_TimeSeries.ipynb (3 outlier có chủ đích + 11 điểm NaN gồm 1 khoảng 3 tháng liên tiếp và 8 điểm ngẫu nhiên).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.lab01.data_loader import load_airpassengers_dataset
from src.lab01.tracking import get_active_run_id, record_stage_result, write_log
from src.lab01.utils import ensure_directories, get_path, load_config, save_json, set_global_seed
from src.lab01.visualize import (
    detect_and_plot_rolling_outliers,
    detect_and_plot_stl_outliers,
    evaluate_and_plot_imputation,
    plot_before_after_cleaning,
    plot_residuals_comparison,
)


def simulate_dirty_series(
    ts_true: pd.Series,
    n_random_missing: int = 8,
    seed: int = 42,
) -> Tuple[pd.Series, pd.DatetimeIndex, pd.DatetimeIndex]:
    r"""
    Purpose: Tạo bản chuỗi bẩn có chủ đích (3 outlier + 1 khoảng thiếu 3 tháng liên tiếp + 8 điểm NaN ngẫu nhiên) để đối chiếu với đáp án thật.
    Input-Output:
        - Input: ts_true (pd.Series), n_random_missing (int, mặc định 8), seed (int, mặc định 42).
        - Output: Tuple gồm (ts_dirty, missing_dates, outlier_dates).
    How: Nhân 3 mốc 1952-08 (1.6x), 1955-02 (0.5x), 1958-11 (1.7x); chèn NaN vào 1955-05..07 và 8 vị trí ngẫu nhiên không trùng outlier.
    """
    ts_dirty = ts_true.astype(float).copy()
    outlier_factors = {"1952-08-01": 1.6, "1955-02-01": 0.5, "1958-11-01": 1.7}
    outlier_dates = pd.DatetimeIndex(list(outlier_factors.keys()))
    for d_str, factor in outlier_factors.items():
        ts_dirty.loc[pd.Timestamp(d_str)] = round(float(ts_true.loc[pd.Timestamp(d_str)]) * factor)

    rng = np.random.default_rng(seed)
    gap_start = ts_dirty.index.get_loc(pd.Timestamp("1955-05-01"))
    gap_pos = [gap_start, gap_start + 1, gap_start + 2]

    blocked = set(gap_pos) | {0, len(ts_dirty) - 1, gap_start - 1, gap_start + 3}
    for d in outlier_dates:
        idx_o = ts_dirty.index.get_loc(d)
        blocked.update({idx_o - 1, idx_o, idx_o + 1})

    candidates = [i for i in range(len(ts_dirty)) if i not in blocked]
    rand_pos = rng.choice(candidates, size=min(n_random_missing, len(candidates)), replace=False).tolist()
    all_miss_pos = sorted(gap_pos + rand_pos)

    ts_dirty.iloc[all_miss_pos] = np.nan
    missing_dates = ts_dirty.index[all_miss_pos]
    return ts_dirty, missing_dates, outlier_dates


def _score_outlier_method(detected_mask: pd.Series, true_dates: pd.DatetimeIndex) -> Dict[str, Any]:
    r"""
    Purpose: Tính Precision, Recall, F1 cho một phương pháp phát hiện outlier so với 3 outlier thật đã chèn.
    Input-Output:
        - Input: detected_mask (Boolean Series), true_dates (pd.DatetimeIndex).
        - Output: Từ điển gồm detected_count, tp, fp, fn, precision, recall, f1.
    How: Đối chiếu tập ngày phát hiện được với tập true_dates.
    """
    pred_set = set(detected_mask.index[detected_mask])
    true_set = set(true_dates)
    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "detected_count": int(len(pred_set)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def run_quality_pipeline(
    ts_true: pd.Series,
    n_random_missing: int = 8,
    window: int = 12,
    z_thresh: float = 3.0,
    iqr_mult: float = 1.5,
    seed: int = 42,
    proc_dir: Optional[Path] = None,
    fig_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    r"""
    Purpose: Chạy trọn vẹn Bài tập 4: giả lập dữ liệu bẩn, so sánh điền khuyết, so sánh 4 phương pháp phát hiện outlier và làm sạch.
    Input-Output:
        - Input: ts_true, n_random_missing, window, z_thresh, iqr_mult, seed, proc_dir, fig_dir.
        - Output: Từ điển chứa kết quả imputation, bảng đánh giá 4 phương pháp outlier và sai số khôi phục cuối cùng.
    How: Gọi simulate_dirty_series(), chạy 5 hàm vẽ 08_ -> 12_ từ visualize.py và lưu chuỗi sạch vào data/processed/lab01/.
    """
    ts_dirty, missing_dates, outlier_dates = simulate_dirty_series(ts_true, n_random_missing=n_random_missing, seed=seed)

    ts_ffill, ts_interp, imp_metrics = evaluate_and_plot_imputation(
        ts_dirty=ts_dirty,
        ts_true=ts_true,
        missing_dates=missing_dates,
        zoom_range=("1955-03-01", "1955-12-01"),
        save_path=(fig_dir / "08_missing_imputation_eval.png") if fig_dir else None,
    )
    ts_filled = ts_interp.copy()

    out_roll = detect_and_plot_rolling_outliers(
        ts_filled=ts_filled,
        outlier_dates=outlier_dates,
        window=window,
        k=z_thresh,
        save_path=(fig_dir / "09_rolling_outliers_detection.png") if fig_dir else None,
    )
    out_stl, log_resid = detect_and_plot_stl_outliers(
        ts_filled=ts_filled,
        outlier_dates=outlier_dates,
        save_path=(fig_dir / "10_stl_log_outliers_detection.png") if fig_dir else None,
    )

    z_global = ((ts_filled - ts_filled.mean()) / ts_filled.std()).abs() > z_thresh
    q1, q3 = float(ts_filled.quantile(0.25)), float(ts_filled.quantile(0.75))
    iqr = q3 - q1
    iqr_global = (ts_filled < q1 - iqr_mult * iqr) | (ts_filled > q3 + iqr_mult * iqr)

    # Làm sạch chuỗi: thay outlier phát hiện từ STL bằng NaN rồi nội suy theo thời gian
    ts_clean = ts_filled.copy()
    ts_clean[out_stl] = np.nan
    ts_clean = ts_clean.interpolate(method="time")

    plot_before_after_cleaning(
        ts_dirty=ts_dirty,
        ts_ffill=ts_ffill, 
        ts_interp=ts_interp,
        ts_true=ts_true,
        ts_clean=ts_clean,
        missing_dates=missing_dates,
        outlier_dates=outlier_dates,
        save_path=(fig_dir / "11_before_after_cleaning.png") if fig_dir else None,
    )
    plot_residuals_comparison(
        ts_clean=ts_clean,
        log_resid=log_resid,
        out_z_res=out_stl,
        save_path=(fig_dir / "12_residuals_comparison.png") if fig_dir else None,
    )

    damaged = missing_dates.union(outlier_dates)
    clean_err = ts_clean[damaged] - ts_true[damaged]
    clean_metrics = {
        "MAE": round(float(clean_err.abs().mean()), 4),
        "RMSE": round(float(np.sqrt((clean_err ** 2).mean())), 4),
        "MAPE": round(float((clean_err.abs() / ts_true[damaged]).mean() * 100.0), 4),
    }

    if proc_dir is not None:
        pd.DataFrame(
            {
                "Original_True": ts_true,
                "Dirty_Input": ts_dirty,
                "Forward_Fill": ts_ffill,
                "Time_Interpolation": ts_interp,
                "Cleaned_Final": ts_clean,
            }
        ).to_csv(proc_dir / "airpassengers_quality_cleaned.csv")

    best_imp = (
        "Time Interpolation"
        if imp_metrics["time_interpolation"]["RMSE"] <= imp_metrics["forward_fill"]["RMSE"]
        else "Forward-fill"
    )
    return {
        "imputation": {
            "missing_simulated_count": int(len(missing_dates)),
            "missing_pct": round(float(len(missing_dates) / len(ts_true) * 100.0), 2),
            "missing_dates": [str(d.date()) for d in missing_dates],
            "missing_after_ffill": int(ts_ffill.isna().sum()),
            "missing_after_interp": int(ts_interp.isna().sum()),
            "forward_fill": imp_metrics["forward_fill"],
            "time_interpolation": imp_metrics["time_interpolation"],
            "best_method": best_imp,
        },
        "outliers": {
            "injected_outlier_dates": [str(d.date()) for d in outlier_dates],
            "methods_comparison": {
                "Rolling_3sigma": _score_outlier_method(out_roll, outlier_dates),
                "ZScore_Global": _score_outlier_method(z_global, outlier_dates),
                "IQR_Global": _score_outlier_method(iqr_global, outlier_dates),
                "STL_Log_Robust_MAD": _score_outlier_method(out_stl, outlier_dates),
            },
            "outliers_replaced_count": int(out_stl.sum()),
            "missing_remaining_after_clean": int(ts_clean.isna().sum()),
        },
        "final_cleaning_metrics_on_damaged_points": clean_metrics,
    }


def main(run_id: Optional[str] = None) -> Dict[str, Any]:
    r"""
    Purpose: Thực thi Giai đoạn 4 (Bài tập 4: Kiểm tra và xử lý chất lượng dữ liệu) cho Lab 01.
    Input-Output:
        - Input: run_id (Optional[str]).
        - Output: Từ điển tổng hợp kết quả điền khuyết, phát hiện ngoại lai và sai số khôi phục cuối cùng.
    How: Nạp AirPassengers, gọi run_quality_pipeline(), lưu JSON vào metrics_dir và cập nhật tracking.
    """
    t0 = time.perf_counter()
    cfg = load_config()
    ensure_directories(cfg)
    set_global_seed(int(cfg["project"]["random_seed"]))
    active_run_id = get_active_run_id(run_id)

    write_log(active_run_id, "DATA_QUALITY", "START", "Bắt đầu Bài 4: Giả lập 11 NaN + 3 Outlier, Imputation & Outlier Cleaning.")
    df, _ = load_airpassengers_dataset(cfg)
    series = df[cfg["data"]["value_col"]].astype(float)

    payload = run_quality_pipeline(
        ts_true=series,
        n_random_missing=int(cfg["analysis"]["missing_simulation_count"]),
        window=int(cfg["analysis"]["rolling_window"]),
        z_thresh=float(cfg["analysis"]["zscore_threshold"]),
        iqr_mult=float(cfg["analysis"]["iqr_multiplier"]),
        seed=int(cfg["project"]["random_seed"]),
        proc_dir=get_path("data_processed_dir", cfg),
        fig_dir=get_path("figures_dir", cfg),
    )
    payload["generated_figures"] = [
        "08_missing_imputation_eval.png",
        "09_rolling_outliers_detection.png",
        "10_stl_log_outliers_detection.png",
        "11_before_after_cleaning.png",
        "12_residuals_comparison.png",
    ]
    payload["duration_sec"] = round(time.perf_counter() - t0, 4)

    save_json(payload, get_path("metrics_dir", cfg) / "data_quality_summary.json")
    record_stage_result(active_run_id, "data_quality", payload)
    write_log(
        active_run_id,
        "DATA_QUALITY",
        "SUCCESS",
        f"Hoàn tất 5 hình Bài 4 | STL Outlier F1={payload['outliers']['methods_comparison']['STL_Log_Robust_MAD']['f1']} | Clean RMSE={payload['final_cleaning_metrics_on_damaged_points']['RMSE']}.",
    )
    return payload