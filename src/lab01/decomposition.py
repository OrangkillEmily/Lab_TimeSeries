r"""
Mô-đun Phân rã Chuỗi Thời gian (Bài tập 3: Classical Additive/Multiplicative & STL Loess Decomposition) cho Lab 01.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL, seasonal_decompose

from src.lab01.data_loader import load_airpassengers_dataset
from src.lab01.tracking import get_active_run_id, record_stage_result, write_log
from src.lab01.utils import ensure_directories, get_path, load_config, save_json
from src.lab01.visualize import analyze_residuals, plot_decomposition


def compute_strength_metrics(trend: pd.Series, seasonal: pd.Series, resid: pd.Series) -> Dict[str, float]:
    r"""
    Purpose: Tính độ mạnh Xu hướng (F_T) và độ mạnh Mùa vụ (F_S) theo chuẩn Hyndman & Athanasopoulos.
    Input-Output:
        - Input: trend (pd.Series), seasonal (pd.Series), resid (pd.Series).
        - Output: Từ điển chứa trend_strength và seasonal_strength trong [0, 1].
    How: Tính F_T = max(0, 1 - Var(R)/Var(T+R)) và F_S = max(0, 1 - Var(R)/Var(S+R)) sau khi loại bỏ NaN.
    """
    valid = pd.concat([trend, seasonal, resid], axis=1).dropna()
    t_v, s_v, r_v = valid.iloc[:, 0], valid.iloc[:, 1], valid.iloc[:, 2]

    var_r = float(np.var(r_v))
    var_tr = float(np.var(t_v + r_v))
    var_sr = float(np.var(s_v + r_v))

    f_t = max(0.0, 1.0 - var_r / var_tr) if var_tr > 0 else 0.0
    f_s = max(0.0, 1.0 - var_r / var_sr) if var_sr > 0 else 0.0
    return {
        "trend_strength": round(f_t, 4),
        "seasonal_strength": round(f_s, 4),
    }


def run_decomposition_analysis(
    series: pd.Series,
    period: int = 12,
    fig_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    r"""
    Purpose: So sánh Classical Additive vs. Multiplicative, chạy STL Loess và kiểm định phần dư bằng analyze_residuals bản gốc.
    Input-Output:
        - Input: series (pd.Series), period (int, mặc định 12), fig_dir (Optional[Path]).
        - Output: Từ điển chứa phương sai phần dư Additive/Multiplicative, độ mạnh F_T/F_S và thống kê Ljung-Box phần dư STL.
    How: Gọi plot_decomposition() cho Classical Multiplicative & STL và analyze_residuals() cho stl_raw.resid.
    """
    decomp_add = seasonal_decompose(series, model="additive", period=period)
    decomp_mul = seasonal_decompose(series, model="multiplicative", period=period)
    stl_raw = STL(series, period=period, seasonal=13, robust=True).fit()

    var_resid_add = float(np.nanvar(decomp_add.resid))
    var_resid_mul = float(np.nanvar(decomp_mul.resid))
    stl_strength = compute_strength_metrics(stl_raw.trend, stl_raw.seasonal, stl_raw.resid)

    if fig_dir is not None:
        plot_decomposition(
            result=decomp_mul,
            title="Classical Decomposition (Multiplicative Model) - AirPassengers",
            is_multiplicative=True,
            save_path=fig_dir / "05_classical_decomposition_multiplicative.png",
        )
        plot_decomposition(
            result=stl_raw,
            title="STL Decomposition (Loess Regression) - AirPassengers",
            is_multiplicative=False,
            save_path=fig_dir / "06_stl_decomposition.png",
        )

    resid_stats = analyze_residuals(
        stl_raw.resid,
        lags=30,
        save_path=(fig_dir / "07_stl_residual_diagnostics.png") if fig_dir else None,
    )

    return {
        "period": int(period),
        "classical_comparison": {
            "additive_resid_variance": round(var_resid_add, 4),
            "multiplicative_resid_variance": round(var_resid_mul, 6),
            "recommended_model": "Multiplicative",
        },
        "stl": {
            "trend_strength": stl_strength["trend_strength"],
            "seasonal_strength": stl_strength["seasonal_strength"],
            **resid_stats,
        },
        "trend_direction": "Tăng trưởng mạnh và ổn định suốt giai đoạn 1949–1960",
        "seasonality_level": "Rất mạnh (biên độ giãn nở theo xu hướng — dạng Multiplicative)",
    }


def main(run_id: Optional[str] = None) -> Dict[str, Any]:
    r"""
    Purpose: Thực thi Giai đoạn 3 (Bài tập 3: Phân rã chuỗi thời gian) cho Lab 01.
    Input-Output:
        - Input: run_id (Optional[str]).
        - Output: Từ điển tổng hợp kết quả phân rã và danh sách 3 biểu đồ đã sinh.
    How: Nạp AirPassengers, gọi run_decomposition_analysis(), lưu JSON vào metrics_dir và cập nhật tracking.
    """
    t0 = time.perf_counter()
    cfg = load_config()
    ensure_directories(cfg)
    active_run_id = get_active_run_id(run_id)

    write_log(active_run_id, "DECOMPOSITION", "START", "Bắt đầu Bài 3: Phân rã Classical Multiplicative & STL Loess.")
    df, _ = load_airpassengers_dataset(cfg)
    series = df[cfg["data"]["value_col"]].astype(float)

    res = run_decomposition_analysis(
        series,
        period=int(cfg["analysis"]["decomposition_period"]),
        fig_dir=get_path("figures_dir", cfg),
    )
    res["generated_figures"] = [
        "05_classical_decomposition_multiplicative.png",
        "06_stl_decomposition.png",
        "07_stl_residual_diagnostics.png",
    ]
    res["duration_sec"] = round(time.perf_counter() - t0, 4)

    save_json(res, get_path("metrics_dir", cfg) / "decomposition_summary.json")
    record_stage_result(active_run_id, "decomposition", res)
    write_log(
        active_run_id,
        "DECOMPOSITION",
        "SUCCESS",
        f"Hoàn tất 3 hình phân rã | F_T={res['stl']['trend_strength']}, F_S={res['stl']['seasonal_strength']}, LB_p20={res['stl']['ljung_box_p20']}.",
    )
    return res