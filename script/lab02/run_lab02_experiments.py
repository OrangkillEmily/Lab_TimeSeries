"""
AUTOMATED EXPERIMENT PIPELINE ENTRY POINT — script/lab02/run_lab02_experiments.py
================================================================================
PURPOSE:
    Standalone CLI orchestrator for the LAB02 Automated Model Experimentation
    Pipeline.
    1. Xuất đầy đủ Model Summary (AR / MA / ARIMA / SARIMA / Log-SARIMA) vào
       reports/model_summary/lab02/<dataset>/ theo yêu cầu nộp bài.
    2. Trích xuất số liệu thực tế (EDA, Train/Test, ADF, ACF/PACF, ARIMA,
       SARIMA, Log-SARIMA) vào progress_results/lab02/<dataset>/latest_run.json.
    3. Vẽ và lưu bộ ảnh trọng tâm giữ nguyên 100% nét gốc vào
       reports/figures/lab02/<dataset>/report/ để render_report.py xuất HTML/PDF/MD.

SUPPORTED CLI COMMANDS:
    python script/lab02/run_lab02_experiments.py --dataset dataset1
    python script/lab02/run_lab02_experiments.py --dataset dataset2
    python script/lab02/run_lab02_experiments.py --dataset all --mode full
================================================================================
"""
import argparse
import itertools
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

current_path = Path(__file__).resolve().parent
if (current_path / "src").is_dir():
    PROJECT_ROOT = current_path
elif (current_path.parent / "src").is_dir():
    PROJECT_ROOT = current_path.parent
elif (current_path.parent.parent / "src").is_dir():
    PROJECT_ROOT = current_path.parent.parent
else:
    PROJECT_ROOT = current_path

for p in [str(PROJECT_ROOT), str(current_path)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.lab02.data_loader import load_dataset, split_dataset
from src.lab02.logging_config import setup_experiment_logger
from src.lab02.stats import perform_adf_test, significant_lags, plot_acf_pacf
from src.lab02.acf_pacf_analysis import analyze_acf_pacf
from src.lab02.model_summary import ModelSummaryRegistry
from src.lab02.modeling import (
    fit_arima,
    fit_sarima,
    residual_diagnostics,
    forecast_test,
    calculate_metrics,
    plot_forecast,
    plot_arima_vs_sarima,
    plot_log_transformation,
)
from src.lab02.lab02_models import (
    run_ar_experiment,
    run_ma_experiment,
    run_arima_experiment,
    run_sarima_experiment,
    get_candidate_models,
    auto_identify_candidates_from_data,
)
from src.lab02.visualize import (
    plot_original_series_original_style,
    plot_train_test_split_original_style,
    plot_differencing_original_style,
    plot_actual_vs_forecast_battle_log,
    plot_log_diagnostics_2x2,
)


DATASET_SPECIFICATIONS: Dict[str, Dict[str, Any]] = {
    "dataset1": {
        "dataset_key": "dataset1",
        "dataset_title": "Monthly Retail Sales (US, 07/1991 - 06/2008)",
        "raw_file": "Monthly Retail Sales.csv",
        "date_col": "date",
        "value_col": "value",
        "freq": "MS",
        "seasonal_period": 12,
    },
    "dataset2": {
        "dataset_key": "dataset2",
        "dataset_title": "AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)",
        "raw_file": "AirPassengers.csv",
        "date_col": "Month",
        "value_col": "Passengers",
        "freq": "MS",
        "seasonal_period": 12,
    },
}


def _extract_top_errors(test: pd.Series, fc: pd.Series, top_n: int = 3) -> List[Dict[str, Any]]:
    err = test - fc
    tmp = (
        pd.DataFrame(
            {
                "date": test.index.strftime("%Y-%m-%d"),
                "actual": test.values,
                "forecast": fc.values,
                "error": err.values,
                "abs_error": err.abs().values,
                "pct_error": (err.abs() / test.replace(0, np.nan) * 100.0).values,
            }
        )
        .sort_values("abs_error", ascending=False)
        .head(top_n)
    )
    return [
        {k: (str(r[k]) if k == "date" else round(float(r[k]), 2)) for k in tmp.columns}
        for _, r in tmp.iterrows()
    ]


def _param_elim(df_grid: pd.DataFrame, col: str) -> List[Dict[str, Any]]:
    g = df_grid.groupby(col)[["AIC", "RMSE", "MAPE (%)", "R2"]].mean().round(2).reset_index()
    return g.to_dict(orient="records")


def export_progress_and_report_figures(
    dataset_key: str,
    spec: Dict[str, Any],
    df: pd.DataFrame,
    train: pd.Series,
    test: pd.Series,
    adf_raw: Dict[str, Any],
    adf_d1: Dict[str, Any],
    adf_d1D1: Dict[str, Any],
    run_id: str,
    save_figures: bool = True,
) -> Path:
    """
    Trích xuất số liệu thực tế ra progress_results/lab02/<dataset>/latest_run.json
    và xuất bộ ảnh trọng tâm giữ nguyên 100% nét vẽ gốc vào reports/figures/lab02/<dataset>/report/.
    """
    warnings.filterwarnings("ignore")
    val_col = spec["value_col"]
    s = spec["seasonal_period"]
    is_ds1 = dataset_key == "dataset1"
    ylabel = "Doanh số bán lẻ" if is_ds1 else "Số hành khách (nghìn người)"

    fig_dir = PROJECT_ROOT / "reports" / "figures" / "lab02" / dataset_key / "report"
    fig_dir.mkdir(parents=True, exist_ok=True)
    prog_dir = PROJECT_ROOT / "progress_results" / "lab02" / dataset_key
    prog_dir.mkdir(parents=True, exist_ok=True)

    ser = df[val_col]
    diff_1 = train.diff(1)
    diff_1_12 = diff_1.diff(s)

    # 1. EDA đầy đủ + Train/Test stats
    eda = {
        "n_obs": int(len(df)),
        "missing": int(ser.isna().sum()),
        "duplicates": int(df.index.duplicated().sum()),
        "dtype": str(ser.dtype),
        "start_date": str(df.index.min().date()),
        "end_date": str(df.index.max().date()),
        "mean": round(float(ser.mean()), 4),
        "std": round(float(ser.std()), 4),
        "cv_pct": round(float(ser.std() / ser.mean() * 100.0), 2),
        "skew": round(float(ser.skew()), 3),
        "min": round(float(ser.min()), 4),
        "min_date": ser.idxmin().strftime("%m/%Y"),
        "q25": round(float(ser.quantile(0.25)), 4),
        "median": round(float(ser.median()), 4),
        "q75": round(float(ser.quantile(0.75)), 4),
        "max": round(float(ser.max()), 4),
        "max_date": ser.idxmax().strftime("%m/%Y"),
        "growth_ratio": round(float(ser.iloc[-1] / ser.iloc[0]), 2),
        "train_size": int(len(train)),
        "train_start": str(train.index.min().date()),
        "train_end": str(train.index.max().date()),
        "train_mean": round(float(train.mean()), 2),
        "train_std": round(float(train.std()), 2),
        "test_size": int(len(test)),
        "test_start": str(test.index.min().date()),
        "test_end": str(test.index.max().date()),
        "test_mean": round(float(test.mean()), 2),
        "test_std": round(float(test.std()), 2),
    }

    # 2. Vẽ ảnh gốc Bài 1, Bài 3, Bài 4
    if save_figures:
        plot_original_series_original_style(
            df=df,
            value_col=val_col,
            title="Monthly Retail Sales Over Time (07/1991 - 06/2008)"
            if is_ds1
            else "Air Passengers Over Time (01/1949 - 12/1960)",
            ylabel=ylabel,
            month_interval=3 if is_ds1 else 1,
            save_path=fig_dir / "01_original_series.png",
            show_plot=False,
        )
        plot_train_test_split_original_style(
            train=train,
            test=test,
            title=f"PHÂN CHIA TẬP HUẤN LUYỆN (TRAIN) VÀ KIỂM THỬ (TEST) - {'MONTHLY RETAIL SALES' if is_ds1 else 'AIRPASSENGERS'}",
            ylabel=ylabel,
            save_path=fig_dir / "02_train_test_split.png",
            show_plot=False,
        )
        plot_differencing_original_style(
            train=train,
            diff_1=diff_1,
            diff_1_12=diff_1_12,
            value_label=val_col,
            seasonal_period=s,
            save_path=fig_dir / "03_differencing.png",
            show_plot=False,
        )
        plot_acf_pacf(
            diff_1,
            lags=36,
            title="ACF & PACF - Train sau sai phân bậc 1 (d=1)",
            seasonal_period=s,
            save_path=str(fig_dir / "04_acf_pacf_d1.png"),
        )
        plt.close("all")
        plot_acf_pacf(
            diff_1_12,
            lags=36,
            title=f"ACF & PACF - Train sau sai phân d=1 và D=1 (s={s})",
            seasonal_period=s,
            save_path=str(fig_dir / "05_acf_pacf_d1_D1.png"),
        )
        plt.close("all")

    lag_list = [1, 2, 3, 4, 5, 6, 12, 24, 36]

    def _pack_acf(sig: Dict[str, Any]) -> Dict[str, Any]:
        max_k = len(sig["acf_values"]) - 1
        return {
            "n": int(sig["n"]),
            "bound": round(float(sig["bound"]), 3),
            "acf_sig": [int(x) for x in sig["acf_sig"]],
            "pacf_sig": [int(x) for x in sig["pacf_sig"]],
            "table": [
                {
                    "lag": k,
                    "acf": round(float(sig["acf_values"][k]), 2),
                    "pacf": round(float(sig["pacf_values"][k]), 2),
                    "acf_sig": k in sig["acf_sig"],
                    "pacf_sig": k in sig["pacf_sig"],
                }
                for k in lag_list
                if k <= max_k
            ],
        }

    acf_d1 = _pack_acf(significant_lags(diff_1, nlags=36))
    acf_d1D1 = _pack_acf(significant_lags(diff_1_12, nlags=36))

    # 3. Bài 5: ARIMA(1,1,1) & ARIMA(2,1,1)
    arima_pack: Dict[str, Any] = {}
    arima_fc_map: Dict[str, Any] = {}
    for idx_a, (name, order) in enumerate([("ARIMA(1,1,1)", (1, 1, 1)), ("ARIMA(2,1,1)", (2, 1, 1))], 1):
        res = fit_arima(train, order=order, dataset_name=dataset_key, verbose=False)
        fc, ci = forecast_test(res, steps=len(test))
        fc.index, ci.index = test.index, test.index
        m = calculate_metrics(test, fc)
        diag = residual_diagnostics(
            res,
            order=order,
            lags=24,
            title=name,
            save_path=str(fig_dir / f"05a_arima{idx_a}_residuals.png") if save_figures else None,
            show_plot=False,
        )
        plt.close("all")
        if save_figures:
            plot_forecast(
                train,
                test,
                fc,
                ci,
                title=f"{name}: Train, Test (thực tế) và Forecast",
                save_path=str(fig_dir / f"05b_arima{idx_a}_forecast.png"),
                ylabel=ylabel,
                show_plot=False,
            )
            plt.close("all")
        arima_fc_map[name] = (fc, ci)
        arima_pack[name] = {
            "aic": round(float(res.aic), 2),
            "bic": round(float(res.bic), 2),
            "converged": bool(res.mle_retvals.get("converged", True)) if hasattr(res, "mle_retvals") else True,
            "metrics": {k: round(float(v), 2) for k, v in m.items()},
            "diagnostics": {
                "mean": round(float(diag["mean"]), 4),
                "std": round(float(diag["std"]), 4),
                "ljung_box_q": round(float(diag["ljung_box_q"]), 2),
                "ljung_box_p": round(float(diag["ljung_box_p"]), 4),
                "jarque_bera_p": round(float(diag["jarque_bera_p"]), 4),
                "heteroskedasticity_p": round(float(diag["heteroskedasticity_p"]), 4),
                "is_white_noise": bool(diag["is_white_noise"]),
            },
            "top_errors": _extract_top_errors(test, fc, 3),
        }

    # 4. Bài 6: SARIMA grid đúng không gian của từng notebook (18 cho DS1, 24 cho DS2)
    if is_ds1:
        p_vals, q_vals, P_vals, Q_vals = [0, 1, 2], [0, 1, 3], [0], [0, 1]
        best_s_name, best_s_ord, best_s_sea = "SARIMA(0,1,1)x(0,1,1)_12", (0, 1, 1), (0, 1, 1, 12)
    else:
        p_vals, q_vals, P_vals, Q_vals = [0, 1], [0, 1, 3], [0, 1], [0, 1]
        best_s_name, best_s_ord, best_s_sea = "SARIMA(1,1,3)x(0,1,1)_12", (1, 1, 3), (0, 1, 1, 12)

    sarima_rows, fitted_s, fc_s_map = [], {}, {}
    for p, q, P, Q in itertools.product(p_vals, q_vals, P_vals, Q_vals):
        mn = f"SARIMA({p},1,{q})x({P},1,{Q})_{s}"
        try:
            res = fit_sarima(
                train,
                order=(p, 1, q),
                seasonal_order=(P, 1, Q, s),
                dataset_name=dataset_key,
                verbose=False,
            )
            fc, ci = forecast_test(res, steps=len(test))
            fc.index, ci.index = test.index, test.index
            m = calculate_metrics(test, fc)
            lb_p = float(acorr_ljungbox(res.resid.iloc[s + 1:], lags=[24])["lb_pvalue"].iloc[0])
            fitted_s[mn], fc_s_map[mn] = res, (fc, ci)
            sarima_rows.append(
                {
                    "model_name": mn,
                    "p": p,
                    "q": q,
                    "P": P,
                    "Q": Q,
                    "AIC": round(float(res.aic), 2),
                    "BIC": round(float(res.bic), 2),
                    "MAE": round(float(m["MAE"]), 2),
                    "RMSE": round(float(m["RMSE"]), 2),
                    "MAPE (%)": round(float(m["MAPE (%)"]), 2),
                    "R2": round(float(m["R2"]), 4),
                    "lb_p": round(lb_p, 4),
                    "is_white_noise": bool(lb_p > 0.05),
                }
            )
        except Exception:
            continue

    df_s = pd.DataFrame(sarima_rows).sort_values("RMSE").reset_index(drop=True)
    pelim = {"q": _param_elim(df_s, "q"), "p": _param_elim(df_s, "p"), "Q": _param_elim(df_s, "Q")}
    if len(P_vals) > 1:
        pelim["P"] = _param_elim(df_s, "P")

    bs_res = fitted_s[best_s_name]
    bs_fc, bs_ci = fc_s_map[best_s_name]
    bs_m = calculate_metrics(test, bs_fc)
    bs_diag = residual_diagnostics(
        bs_res,
        order=best_s_ord,
        seasonal_order=best_s_sea,
        lags=24,
        title=f"{best_s_name}",
        save_path=str(fig_dir / "06_best_sarima_residuals.png") if save_figures else None,
        n_skip=s + 1,
        show_plot=False,
    )
    plt.close("all")

    if save_figures:
        plot_forecast(
            train,
            test,
            bs_fc,
            bs_ci,
            title=f"Biểu đồ dự báo mô hình {best_s_name}",
            save_path=str(fig_dir / "07_best_sarima_forecast.png"),
            ylabel=ylabel,
            show_plot=False,
        )
        plt.close("all")
        plot_arima_vs_sarima(
            train=train,
            test=test,
            fc_arima1=arima_fc_map["ARIMA(1,1,1)"][0],
            fc_arima2=arima_fc_map["ARIMA(2,1,1)"][0],
            fc_sarima=bs_fc,
            conf_int_sarima=bs_ci,
            names={"arima1": "ARIMA(1,1,1)", "arima2": "ARIMA(2,1,1)", "sarima": best_s_name},
            r2_scores={
                "arima1": arima_pack["ARIMA(1,1,1)"]["metrics"]["R2"],
                "arima2": arima_pack["ARIMA(2,1,1)"]["metrics"]["R2"],
                "sarima": round(float(bs_m["R2"]), 4),
            },
            title=f"ĐỐI CHỨNG DỰ BÁO TRỰC DIỆN: ARIMA VS SARIMA ({dataset_key.upper()})",
            save_path=str(fig_dir / "08_arima_vs_sarima.png"),
            ylabel=ylabel,
            show_plot=False,
        )
        plt.close("all")

    # 5. Bài 7 (Dataset 2): Log-SARIMA giữ nguyên Cell 43, 48, 49
    log_pack = None
    if not is_ds1:
        if save_figures:
            plot_log_transformation(
                train=train,
                test=test,
                save_path=str(fig_dir / "09_log_transformation.png"),
                show_plot=False,
            )
            plt.close("all")

        train_log = np.log(train)
        log_rows, fitted_l, fc_l_map = [], {}, {}
        for p, q, P, Q in itertools.product(p_vals, q_vals, P_vals, Q_vals):
            mn = f"Log-SARIMA({p},1,{q})x({P},1,{Q})_{s}"
            try:
                res = fit_sarima(
                    train_log,
                    order=(p, 1, q),
                    seasonal_order=(P, 1, Q, s),
                    dataset_name=dataset_key,
                    model_family="Log-SARIMA",
                    verbose=False,
                )
                fc, ci = forecast_test(res, steps=len(test), inverse_log=True)
                fc.index, ci.index = test.index, test.index
                m = calculate_metrics(test, fc)
                lb_p = float(acorr_ljungbox(res.resid.iloc[s + 1:], lags=[24])["lb_pvalue"].iloc[0])
                fitted_l[mn], fc_l_map[mn] = res, (fc, ci)
                log_rows.append(
                    {
                        "model_name": mn,
                        "p": p,
                        "q": q,
                        "P": P,
                        "Q": Q,
                        "AIC": round(float(res.aic), 2),
                        "BIC": round(float(res.bic), 2),
                        "MAE": round(float(m["MAE"]), 2),
                        "RMSE": round(float(m["RMSE"]), 2),
                        "MAPE (%)": round(float(m["MAPE (%)"]), 2),
                        "R2": round(float(m["R2"]), 4),
                        "lb_p": round(lb_p, 4),
                        "is_white_noise": bool(lb_p > 0.05),
                    }
                )
            except Exception:
                continue

        df_l = pd.DataFrame(log_rows).sort_values("RMSE").reset_index(drop=True)
        bl_name = "Log-SARIMA(1,1,0)x(0,1,1)_12"
        bl_res = fitted_l[bl_name]
        bl_fc, bl_ci = fc_l_map[bl_name]
        bl_m = calculate_metrics(test, bl_fc)
        bl_diag = residual_diagnostics(
            bl_res,
            order=(1, 1, 0),
            seasonal_order=(0, 1, 1, 12),
            lags=24,
            title=bl_name,
            save_path=None,
            n_skip=s + 1,
            show_plot=False,
        )
        plt.close("all")

        if save_figures:
            plot_actual_vs_forecast_battle_log(
                train=train,
                test=test,
                fc_arima111=arima_fc_map["ARIMA(1,1,1)"][0],
                sarima_log_pred_mean=bl_fc,
                sarima_log_conf_orig=bl_ci,
                arima_m=arima_pack["ARIMA(1,1,1)"]["metrics"],
                sarima_log_m=bl_m,
                save_path=fig_dir / "10_arima_vs_log_sarima_battle.png",
                show_plot=False,
            )
            plot_log_diagnostics_2x2(
                test=test,
                arima_fc=arima_fc_map["ARIMA(1,1,1)"][0],
                sarima_log_pred_mean=bl_fc,
                sarima_log_resid=bl_res.resid.iloc[s + 1:],
                arima_m=arima_pack["ARIMA(1,1,1)"]["metrics"],
                sarima_raw_m=bs_m,
                sarima_log_m=bl_m,
                save_path=fig_dir / "11_log_diagnostics_2x2.png",
                show_plot=False,
            )

        log_pack = {
            "grid_total": int(len(df_l)),
            "top_models": df_l.head(6).to_dict(orient="records"),
            "param_elimination": {k: _param_elim(df_l, k) for k in ["q", "p", "Q", "P"]},
            "best_model": {
                "name": bl_name,
                "aic": round(float(bl_res.aic), 2),
                "bic": round(float(bl_res.bic), 2),
                "metrics": {k: round(float(v), 4 if k == "R2" else 2) for k, v in bl_m.items()},
                "diagnostics": {
                    "mean": round(float(bl_diag["mean"]), 4),
                    "std": round(float(bl_diag["std"]), 4),
                    "ljung_box_q": round(float(bl_diag["ljung_box_q"]), 2),
                    "ljung_box_p": round(float(bl_diag["ljung_box_p"]), 4),
                    "jarque_bera_p": round(float(bl_diag["jarque_bera_p"]), 4),
                    "heteroskedasticity_p": round(float(bl_diag["heteroskedasticity_p"]), 4),
                    "is_white_noise": bool(bl_diag["is_white_noise"]),
                },
                "top_errors": _extract_top_errors(test, bl_fc, 3),
            },
        }

    payload = {
        "run_id": run_id,
        "dataset_key": dataset_key,
        "dataset_title": spec["dataset_title"],
        "raw_file": spec["raw_file"],
        "date_col": spec["date_col"],
        "value_col": val_col,
        "seasonal_period": s,
        "eda": eda,
        "adf": {"raw": adf_raw, "diff_1": adf_d1, "diff_1_12": adf_d1D1},
        "acf_pacf": {"diff_1": acf_d1, "diff_1_12": acf_d1D1},
        "arima": arima_pack,
        "sarima": {
            "grid_total": int(len(df_s)),
            "top_models": df_s.head(6).to_dict(orient="records"),
            "param_elimination": pelim,
            "best_model": {
                "name": best_s_name,
                "aic": round(float(bs_res.aic), 2),
                "bic": round(float(bs_res.bic), 2),
                "metrics": {k: round(float(v), 4 if k == "R2" else 2) for k, v in bs_m.items()},
                "diagnostics": {
                    "mean": round(float(bs_diag["mean"]), 4),
                    "std": round(float(bs_diag["std"]), 4),
                    "ljung_box_q": round(float(bs_diag["ljung_box_q"]), 2),
                    "ljung_box_p": round(float(bs_diag["ljung_box_p"]), 4),
                    "jarque_bera_p": round(float(bs_diag["jarque_bera_p"]), 4),
                    "heteroskedasticity_p": round(float(bs_diag["heteroskedasticity_p"]), 4),
                    "is_white_noise": bool(bs_diag["is_white_noise"]),
                },
                "top_errors": _extract_top_errors(test, bs_fc, 3),
            },
        },
        "log_sarima": log_pack,
    }

    run_json = prog_dir / f"run_summary_{run_id}.json"
    latest_json = prog_dir / "latest_run.json"
    for p in (run_json, latest_json):
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  • Progress JSON  : {latest_json}")
    print(f"  • Report Figures : {fig_dir}")
    return latest_json


def run_dataset_experiment_suite(
    dataset_key: str,
    mode: str = "full",
    save_figures: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Chạy trọn bộ thực nghiệm Lab 02:
      1. Khởi tạo logger & đọc CSV, chia Train/Test (80/20, shuffle=False).
      2. Chạy ADF (raw, d=1, d=1 & D=1) và phân tích ACF/PACF trên Train.
      3. Quét toàn bộ Candidate Model Registry (AR, MA, ARIMA, SARIMA, Log-SARIMA)
         để xuất đủ model_summary nộp giảng viên (tắt xả ảnh rời rạc).
      4. Xuất progress_results JSON + bộ ảnh nét gốc phục vụ render_report.py.
    """
    if dataset_key not in DATASET_SPECIFICATIONS:
        raise ValueError(f"Unsupported dataset_key='{dataset_key}'. Choose 'dataset1' or 'dataset2'.")

    spec = DATASET_SPECIFICATIONS[dataset_key]
    ds_title = spec["dataset_title"]

    logger, run_id, log_path = setup_experiment_logger(dataset_name=dataset_key, console_output=False)

    print("=" * 88)
    print(f"[AUTOMATED EXPERIMENT PIPELINE] DATASET: {ds_title}")
    print(f"  • Run ID       : {run_id}")
    print(f"  • Mode         : {mode.upper()}")
    print(f"  • Log File     : {log_path}")
    print("=" * 88)

    raw_path = PROJECT_ROOT / "data" / "raw" / spec["raw_file"]
    proc_dir = PROJECT_ROOT / "data" / "processed" / "lab02" / dataset_key
    
    df, _ = load_dataset(dataset_key=dataset_key)

    train_df, test_df, _ = split_dataset(df=df, dataset_key=dataset_key)

    val_col = spec["value_col"]
    train = train_df[val_col].asfreq(spec["freq"])
    test = test_df[val_col].asfreq(spec["freq"])

    # 3. Kiểm định ADF & Phân tích ACF/PACF trên Train
    adf_raw = perform_adf_test(train, title=f"{ds_title} - Train Raw", dataset_name=dataset_key, return_dict=True)
    diff_1 = train.diff(1)
    adf_d1 = perform_adf_test(diff_1, title=f"{ds_title} - Train Diff(d=1)", dataset_name=dataset_key, return_dict=True)
    diff_1_12 = diff_1.diff(spec["seasonal_period"])
    adf_d1D1 = perform_adf_test(
        diff_1_12,
        title=f"{ds_title} - Train Diff(d=1, D=1, s={spec['seasonal_period']})",
        dataset_name=dataset_key,
        return_dict=True,
    )

    _ = analyze_acf_pacf(
        series=diff_1,
        dataset_name=dataset_key,
        nlags=36,
        differencing_info="d=1",
        seasonal_period=spec["seasonal_period"],
        output_dir=None,
        file_prefix=f"{dataset_key}_d1",
        show_plot=False,
    )
    _ = analyze_acf_pacf(
        series=diff_1_12,
        dataset_name=dataset_key,
        nlags=36,
        differencing_info=f"d=1, D=1, s={spec['seasonal_period']}",
        seasonal_period=spec["seasonal_period"],
        output_dir=None,
        file_prefix=f"{dataset_key}_d1_D1_s{spec['seasonal_period']}",
        show_plot=False,
    )

    # 4. Khởi tạo ModelSummaryRegistry
    try:
        registry = ModelSummaryRegistry(
            dataset_name=dataset_key,
            dataset_title=ds_title,
            n_obs=len(df),
            date_range=f"{df.index.min().date()} to {df.index.max().date()}",
            frequency=spec["freq"],
            train_size=len(train),
            test_size=len(test),
            run_id=run_id,
        )
    except TypeError:
        registry = ModelSummaryRegistry(
            dataset_name=dataset_key,
            dataset_title=ds_title,
            n_obs=len(df),
            date_range=f"{df.index.min().date()} to {df.index.max().date()}",
            frequency=spec["freq"],
            train_size=len(train),
            test_size=len(test),
        )
        registry.run_id = run_id

    # 5. Quét toàn bộ Candidate Registry (giữ đủ AR/MA cho model_summary, tắt xả ảnh rời rạc)
    if str(mode).lower() == "auto":
        print("\n[AUTO-IDENTIFICATION] Đang tự động nhận diện tham số mô hình từ dữ liệu Train...")
        auto_pack = auto_identify_candidates_from_data(
            train_series=train,
            dataset_name=dataset_key,
            seasonal_period=spec["seasonal_period"],
            include_log=True,
        )
        print(auto_pack["rationale_summary"])
        logger.info(f"[AUTO_IDENTIFY] {auto_pack['rationale_summary']}")
        candidates = auto_pack["candidates"]
    else:
        candidates = get_candidate_models(dataset_key, mode=mode)

    logger.info(f"[CANDIDATE_REGISTRY] Loaded {len(candidates)} candidate configurations (mode='{mode}')")
    print(f"\n>>> Đang huấn luyện và đánh giá {len(candidates)} mô hình ứng viên cho {dataset_key.upper()}...")
    print("-" * 88)
    print(
        f"{'STT':<4} {'Mô hình ứng viên':<30} {'Họ':<11} {'AIC':>9} {'RMSE':>8} {'MAPE(%)':>8} {'R²':>8} {'Ljung-Box p':>11} {'Trạng thái'}"
    )
    print("-" * 88)

    for idx, cand in enumerate(candidates, 1):
        m_name = cand["model_name"]
        m_family = cand["model_family"]
        order = tuple(cand["order"])
        seasonal_order = tuple(cand["seasonal_order"])
        use_log = bool(cand.get("use_log_transform", False))
        rationale = str(cand.get("rationale", ""))

        t_start = time.perf_counter()
        try:
            if m_family == "AR":
                exp = run_ar_experiment(
                    train=train,
                    test=test,
                    p=order[0],
                    d=order[1],
                    dataset_name=dataset_key,
                    fig_dir=None,
                    show_plot=False,
                    verbose=verbose,
                )
            elif m_family == "MA":
                exp = run_ma_experiment(
                    train=train,
                    test=test,
                    q=order[2],
                    d=order[1],
                    dataset_name=dataset_key,
                    fig_dir=None,
                    show_plot=False,
                    verbose=verbose,
                )
            elif m_family == "ARIMA":
                exp = run_arima_experiment(
                    train=train,
                    test=test,
                    order=order,
                    dataset_name=dataset_key,
                    model_name=m_name,
                    fig_dir=None,
                    show_plot=False,
                    verbose=verbose,
                    rationale=rationale,
                )
            else:
                exp = run_sarima_experiment(
                    train=train,
                    test=test,
                    order=order,
                    seasonal_order=seasonal_order,
                    use_log_transform=use_log,
                    dataset_name=dataset_key,
                    model_name=m_name,
                    fig_dir=None,
                    plot_diagnostics_flag=False,
                    plot_forecast_flag=False,
                    show_plot=False,
                    verbose=verbose,
                    rationale=rationale,
                )
            exp["rationale"] = rationale
            rec = registry.register_experiment(exp)

            stat = rec["statistical_results"]
            ev = rec["forecast_evaluation"]
            diag = rec["residual_diagnostics"]
            wn_tag = "Pass (WN)" if diag["is_white_noise"] else "Fail (Autocorr)"
            conv_tag = "" if stat["converged"] else " [WARN:NonConv]"
            print(
                f"{idx:<4} {m_name:<30} {m_family:<11} {stat['aic']:>9.2f} {ev['rmse']:>8.2f} "
                f"{ev['mape']:>7.2f}% {ev['r2']:>8.4f} {diag['ljung_box_p']:>11.4f} {wn_tag}{conv_tag}"
            )
        except Exception as exc:
            dur = time.perf_counter() - t_start
            registry.register_failure(
                model_name=m_name,
                model_family=m_family,
                order=order,
                seasonal_order=seasonal_order,
                transformation="Log Scale" if use_log else "Original Scale",
                rationale=rationale,
                error_message=str(exc),
                duration_seconds=dur,
            )
            print(f"{idx:<4} {m_name:<30} {m_family:<11} {'N/A':>9} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>11} FAILED: {exc}")

    print("-" * 88)

    # 6. Xuất Model Summary (CSV, JSON, MD) + Progress JSON & Ảnh gốc cho Report
    summary_paths = registry.export_all(base_dir=PROJECT_ROOT, save_run_tagged_copy=True)
    comp_df = registry.get_comparison_dataframe()
    interp = registry.generate_automatic_interpretation()

    progress_json = export_progress_and_report_figures(
        dataset_key=dataset_key,
        spec=spec,
        df=df,
        train=train,
        test=test,
        adf_raw=adf_raw,
        adf_d1=adf_d1,
        adf_d1D1=adf_d1D1,
        run_id=run_id,
        save_figures=save_figures,
    )

    best_crit = interp.get("best_by_criterion", {})
    print(f"\n[KẾT QUẢ TỔNG HỢP THEO TIÊU CHÍ — {dataset_key.upper()} | RUN_ID: {run_id}]")
    if best_crit:
        print(f"  • RMSE thấp nhất (Test)                : {best_crit['lowest_test_rmse']['model']} (RMSE = {best_crit['lowest_test_rmse']['value']:.2f})")
        print(f"  • MAE thấp nhất (Test)                 : {best_crit['lowest_test_mae']['model']} (MAE = {best_crit['lowest_test_mae']['value']:.2f})")
        print(f"  • MAPE thấp nhất (Test)                : {best_crit['lowest_test_mape']['model']} (MAPE = {best_crit['lowest_test_mape']['value']:.2f}%)")
        if "highest_test_r2" in best_crit:
            print(f"  • R² cao nhất (Test)                   : {best_crit['highest_test_r2']['model']} (R² = {best_crit['highest_test_r2']['value']:.4f})")
        print(f"  • Đạt chuẩn White Noise & RMSE tốt nhất: {best_crit['lowest_test_rmse_with_white_noise_adequacy']}")

    print("\n[ĐƯỜNG DẪN FILE XUẤT RA]")
    print(f"  • Experiment Log : {log_path}")
    print(f"  • Summary CSV    : {summary_paths['csv']}")
    print(f"  • Summary JSON   : {summary_paths['json']}")
    print(f"  • Summary MD     : {summary_paths['md']}")
    print(f"  • Progress JSON  : {progress_json}\n")

    return {
        "dataset_key": dataset_key,
        "run_id": run_id,
        "log_path": str(log_path),
        "summary_paths": summary_paths,
        "progress_json": str(progress_json),
        "comparison_df": comp_df,
        "interpretation": interp,
        "registry": registry,
    }


def parse_cli_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(
        description="LAB02 Automated Time Series Experiment Runner (AR / MA / ARIMA / SARIMA / Log-SARIMA)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["dataset1", "dataset2", "all"],
        help="Target dataset to run experiments on: 'dataset1', 'dataset2', or 'all' (default: 'all').",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["standard", "full", "auto"],
        help="Candidate set mode: 'standard', 'full', or 'auto'.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip saving PNG figures to speed up execution.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose per-model fitting output to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    
    args = parse_cli_args()
    target_datasets = ["dataset1", "dataset2"] if args.dataset == "all" else [args.dataset]
    
    for ds_key in target_datasets:
        run_dataset_experiment_suite(
            dataset_key=ds_key,
            mode=args.mode,
            save_figures=not args.no_figures,
            verbose=args.verbose,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())