import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.lab02.lab02_utils.logging_config import (
    get_project_root,
    normalize_dataset_key,
    get_run_id,
    get_lab02_logger,
    log_system_status,
)
from src.lab02.lab02_utils.modeling import extract_model_coefficients


def format_terminal_model_card(record: Dict[str, Any]) -> str:
    """
    Purpose:
        Format a single model experiment record into an academic console summary card
        inspired by statsmodels SARIMAX Results (displaying Run ID, Dataset Info, Orders,
        Log-Likelihood, AIC, BIC, HQIC, Parameter Coefficient Table, Ljung-Box,
        Jarque-Bera, Heteroskedasticity, Out-of-Sample Test Metrics, and Warnings).

    Parameters:
        record (Dict[str, Any]): Standardized model summary record dictionary.

    Returns:
        str: Formatted multi-line text card suitable for console or Markdown code blocks.
    """
    ds_info = record.get("dataset_info", {})
    cfg = record.get("model_config", {})
    stat = record.get("statistical_results", {})
    diag = record.get("residual_diagnostics", {})
    eval_m = record.get("forecast_evaluation", {})
    exec_info = record.get("execution_info", {})
    status = exec_info.get("status", "SUCCESS")

    p, d, q = cfg.get("order", (0, 0, 0))
    P, D, Q, s = cfg.get("seasonal_order", (0, 0, 0, 0))
    lines = [
        f"{cfg.get('model_family', 'MODEL')} Model Summary - {ds_info.get('dataset_title', ds_info.get('dataset_name', ''))}",
        "=" * 78,
        f"Run ID: {exec_info.get('run_id', 'N/A')} | Status: {status}",
        f"Data: {ds_info.get('dataset_title', ds_info.get('dataset_name', 'N/A'))}",
        f"Date range: {ds_info.get('date_range', 'N/A')} (Freq: {ds_info.get('frequency', 'MS')})",
        f"Number of observations: {ds_info.get('n_obs', 'N/A')} (Train: {ds_info.get('train_size', 'N/A')}, Test: {ds_info.get('test_size', 'N/A')})",
        f"AR order (p): {p} | Differencing order (d): {d} | MA order (q): {q}",
    ]
    if s and s > 0:
        lines.append(f"Seasonal order (P, D, Q, s): ({P}, {D}, {Q}, {s})")
    lines.extend(
        [
            f"Transformation: {cfg.get('transformation', 'Original Scale')}",
            f"Model: {cfg.get('model_name', 'N/A')}",
            "=" * 78,
        ]
    )

    if status != "SUCCESS":
        lines.append(f"EXECUTION FAILED: {exec_info.get('error_message', 'Unknown error')}")
        lines.append("=" * 78)
        return "\n".join(lines)

    lines.extend(
        [
            f"{'SARIMAX / ARIMA Statistical Fit & Diagnostics':^78}",
            "=" * 78,
            f"Log Likelihood: {stat.get('log_likelihood', float('nan')):>14.3f}   AIC: {stat.get('aic', float('nan')):>14.3f}",
            f"BIC:            {stat.get('bic', float('nan')):>14.3f}   HQIC:{stat.get('hqic', float('nan')):>14.3f}",
            f"Converged:      {str(stat.get('converged', True)):>14}   Run Time: {exec_info.get('duration_seconds', 0.0):>8.4f}s",
            "-" * 78,
            f"{'param':<16} {'coef':>10} {'std err':>10} {'z':>9} {'P>|z|':>9} {'[0.025':>10} {'0.975]':>10}",
            "-" * 78,
        ]
    )

    for c in stat.get("coefficients", []):
        lines.append(
            f"{str(c.get('param', '')):<16} "
            f"{c.get('coef', float('nan')):>10.4f} "
            f"{c.get('std_err', float('nan')):>10.4f} "
            f"{c.get('z_stat', float('nan')):>9.3f} "
            f"{c.get('p_value', float('nan')):>9.3f} "
            f"{c.get('ci_lower', float('nan')):>10.3f} "
            f"{c.get('ci_upper', float('nan')):>10.3f}"
        )

    lines.extend(
        [
            "=" * 78,
            f"Ljung-Box (L={diag.get('ljung_box_lag', 24)}) (Q): {diag.get('ljung_box_q', float('nan')):>10.2f}   "
            f"Jarque-Bera (JB): {diag.get('jarque_bera_stat', float('nan')):>10.2f}",
            f"Prob(Q):               {diag.get('ljung_box_p', float('nan')):>10.4f}   "
            f"Prob(JB):         {diag.get('jarque_bera_p', float('nan')):>10.4f}",
            f"Heteroskedasticity(H): {diag.get('heteroskedasticity_stat', float('nan')):>10.2f}   "
            f"Skew:             {diag.get('skew', float('nan')):>10.2f}",
            f"Prob(H) (two-sided):   {diag.get('heteroskedasticity_p', float('nan')):>10.4f}   "
            f"Kurtosis:         {diag.get('kurtosis', float('nan')):>10.2f}",
            f"Residual Mean:         {diag.get('residual_mean', float('nan')):>10.4f}   "
            f"Residual Std:     {diag.get('residual_std', float('nan')):>10.4f}",
            "-" * 78,
            f"OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test={ds_info.get('test_size', 'N/A')}):",
            f"  MAE = {eval_m.get('mae', float('nan')):.4f} | RMSE = {eval_m.get('rmse', float('nan')):.4f} | "
            f"MAPE = {eval_m.get('mape', float('nan')):.2f}% | R² = {eval_m.get('r2', float('nan')):.4f}",
            "=" * 78,
        ]
    )

    warn_list = stat.get("warnings", [])
    if warn_list:
        lines.append("Warnings:")
        for idx, w in enumerate(warn_list, 1):
            lines.append(f"  [{idx}] {w}")
        lines.append("=" * 78)

    return "\n".join(lines)


class ModelSummaryRegistry:
    """
    Purpose:
        Centralized registry to collect, validate, compare, interpret, and export
        all time series model experiments (both succeeded and failed candidates across
        AR, MA, ARIMA, SARIMA, Log-SARIMA) with full RUN_ID provenance into CSV,
        JSON, and Markdown formats.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_title: str,
        n_obs: int,
        date_range: str,
        frequency: str,
        train_size: int,
        test_size: int,
        run_id: Optional[str] = None,
    ) -> None:
        self.dataset_key = normalize_dataset_key(dataset_name)
        self.dataset_title = dataset_title
        self.dataset_info = {
            "dataset_name": self.dataset_key,
            "dataset_title": dataset_title,
            "n_obs": int(n_obs),
            "date_range": str(date_range),
            "frequency": str(frequency),
            "train_size": int(train_size),
            "test_size": int(test_size),
        }
        self.run_id = run_id or get_run_id(self.dataset_key)
        self.records: Dict[str, Dict[str, Any]] = {}
        self.logger = get_lab02_logger(self.dataset_key, "model_summary")

    def register_experiment(self, exp_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Purpose:
            Register a successful model experiment record into the registry.
        """
        result = exp_output["result"]
        meta = getattr(result, "_lab02_meta", {})
        model_name = str(exp_output.get("model_name") or meta.get("model_name") or "Unnamed_Model")
        model_family = str(exp_output.get("model_family") or meta.get("model_family") or "ARIMA")

        order = tuple(exp_output.get("order") or meta.get("order") or (0, 0, 0))
        seasonal_order = tuple(exp_output.get("seasonal_order") or meta.get("seasonal_order") or (0, 0, 0, 0))

        coef_df = exp_output.get("coefficients")
        if coef_df is None:
            coef_df = extract_model_coefficients(result)
        coef_list = coef_df.to_dict(orient="records") if isinstance(coef_df, pd.DataFrame) else list(coef_df)

        diag = exp_output.get("diagnostics", {})
        metrics = exp_output.get("metrics", {})
        fig_paths = exp_output.get("figure_paths", {})

        p, d, q = int(order[0]), int(order[1]), int(order[2])
        P, D, Q, s = (
            int(seasonal_order[0]),
            int(seasonal_order[1]),
            int(seasonal_order[2]),
            int(seasonal_order[3]),
        )

        record = {
            "dataset_info": dict(self.dataset_info),
            "model_config": {
                "model_name": model_name,
                "model_family": model_family,
                "order": [p, d, q],
                "seasonal_order": [P, D, Q, s],
                "p": p,
                "d": d,
                "q": q,
                "P": P,
                "D": D,
                "Q": Q,
                "s": s,
                "transformation": str(exp_output.get("transformation", "Original Scale")),
                "scale_for_ic": str(exp_output.get("scale_for_ic", f"original_d{d}_D{D}_s{s}")),
                "training_config": str(exp_output.get("rationale", "MLE via statsmodels")),
            },
            "statistical_results": {
                "aic": float(result.aic),
                "bic": float(result.bic),
                "hqic": float(getattr(result, "hqic", np.nan)),
                "log_likelihood": float(result.llf),
                "converged": bool(meta.get("converged", True)),
                "warnings": list(meta.get("warnings", [])),
                "coefficients": coef_list,
            },
            "residual_diagnostics": {
                "residual_mean": float(diag.get("mean", np.nan)),
                "residual_std": float(diag.get("std", np.nan)),
                "ljung_box_lag": int(diag.get("ljung_box_lag", 24)),
                "ljung_box_q": float(diag.get("ljung_box_q", np.nan)),
                "ljung_box_p": float(diag.get("ljung_box_p", np.nan)),
                "jarque_bera_stat": float(diag.get("jarque_bera_stat", np.nan)),
                "jarque_bera_p": float(diag.get("jarque_bera_p", np.nan)),
                "skew": float(diag.get("skew", np.nan)),
                "kurtosis": float(diag.get("kurtosis", np.nan)),
                "heteroskedasticity_stat": float(diag.get("heteroskedasticity_stat", np.nan)),
                "heteroskedasticity_p": float(diag.get("heteroskedasticity_p", np.nan)),
                "is_white_noise": bool(diag.get("is_white_noise", float(diag.get("ljung_box_p", 0.0)) >= 0.05)),
            },
            "forecast_evaluation": {
                "mae": float(metrics.get("MAE", np.nan)),
                "rmse": float(metrics.get("RMSE", np.nan)),
                "mape": float(metrics.get("MAPE (%)", metrics.get("MAPE", np.nan))),
                "r2": float(metrics.get("R2", np.nan)),
            },
            "execution_info": {
                "run_id": self.run_id,
                "status": "SUCCESS",
                "error_message": None,
                "timestamp": str(meta.get("end_time") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                "duration_seconds": float(meta.get("duration_sec", 0.0)),
                "output_paths": fig_paths,
            },
        }

        self.records[model_name] = record
        self.logger.info(
            f"[SUMMARY_REGISTERED] run_id='{self.run_id}' | model='{model_name}' | status='SUCCESS'"
        )
        return record

    def register_failure(
        self,
        model_name: str,
        model_family: str,
        order: tuple,
        seasonal_order: tuple = (0, 0, 0, 0),
        transformation: str = "Original Scale",
        rationale: str = "",
        error_message: str = "",
        duration_seconds: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Purpose:
            Record a failed candidate model experiment so that failures are
            transparently tracked in the Model Summary rather than silently dropped.
        """
        p, d, q = int(order[0]), int(order[1]), int(order[2])
        P, D, Q, s = (
            int(seasonal_order[0]),
            int(seasonal_order[1]),
            int(seasonal_order[2]),
            int(seasonal_order[3]),
        )
        record = {
            "dataset_info": dict(self.dataset_info),
            "model_config": {
                "model_name": model_name,
                "model_family": model_family,
                "order": [p, d, q],
                "seasonal_order": [P, D, Q, s],
                "p": p,
                "d": d,
                "q": q,
                "P": P,
                "D": D,
                "Q": Q,
                "s": s,
                "transformation": transformation,
                "scale_for_ic": "failed",
                "training_config": rationale,
            },
            "statistical_results": {
                "aic": float("nan"),
                "bic": float("nan"),
                "hqic": float("nan"),
                "log_likelihood": float("nan"),
                "converged": False,
                "warnings": [error_message] if error_message else [],
                "coefficients": [],
            },
            "residual_diagnostics": {
                "residual_mean": float("nan"),
                "residual_std": float("nan"),
                "ljung_box_lag": 24,
                "ljung_box_q": float("nan"),
                "ljung_box_p": float("nan"),
                "jarque_bera_stat": float("nan"),
                "jarque_bera_p": float("nan"),
                "skew": float("nan"),
                "kurtosis": float("nan"),
                "heteroskedasticity_stat": float("nan"),
                "heteroskedasticity_p": float("nan"),
                "is_white_noise": False,
            },
            "forecast_evaluation": {
                "mae": float("nan"),
                "rmse": float("nan"),
                "mape": float("nan"),
                "r2": float("nan"),
            },
            "execution_info": {
                "run_id": self.run_id,
                "status": "FAILED",
                "error_message": str(error_message),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": float(duration_seconds),
                "output_paths": {},
            },
        }
        self.records[model_name] = record
        self.logger.error(
            f"[SUMMARY_REGISTERED_FAILURE] run_id='{self.run_id}' | model='{model_name}' | error='{error_message}'"
        )
        return record

    def get_comparison_dataframe(self) -> pd.DataFrame:
        """
        Purpose:
            Construct a comprehensive comparison DataFrame across all registered
            candidate models (including Run ID and execution Status), sorted by
            out-of-sample RMSE ascending (with failed models placed at the bottom).
        """
        rows = []
        for name, rec in self.records.items():
            cfg = rec.get("model_config", {})
            stat = rec.get("statistical_results", {})
            diag = rec.get("residual_diagnostics", {})
            ev = rec.get("forecast_evaluation", {})
            ex = rec.get("execution_info", {})
            status = ex.get("status", "SUCCESS")
            rows.append(
                {
                    "Run_ID": ex.get("run_id", self.run_id),
                    "Model": name,
                    "Family": cfg.get("model_family", "N/A"),
                    "Status": status,
                    "Order (p,d,q)": f"({cfg.get('p',0)},{cfg.get('d',0)},{cfg.get('q',0)})",
                    "Seasonal (P,D,Q,s)": f"({cfg.get('P',0)},{cfg.get('D',0)},{cfg.get('Q',0)},{cfg.get('s',0)})",
                    "Transformation": cfg.get("transformation", "Original Scale"),
                    "Scale_Group": cfg.get("scale_for_ic", "original"),
                    "AIC": round(stat.get("aic", float("nan")), 2) if not np.isnan(stat.get("aic", float("nan"))) else np.nan,
                    "BIC": round(stat.get("bic", float("nan")), 2) if not np.isnan(stat.get("bic", float("nan"))) else np.nan,
                    "LogLik": round(stat.get("log_likelihood", float("nan")), 2) if not np.isnan(stat.get("log_likelihood", float("nan"))) else np.nan,
                    "MAE": round(ev.get("mae", float("nan")), 2) if not np.isnan(ev.get("mae", float("nan"))) else np.nan,
                    "RMSE": round(ev.get("rmse", float("nan")), 2) if not np.isnan(ev.get("rmse", float("nan"))) else np.nan,
                    "MAPE (%)": round(ev.get("mape", float("nan")), 2) if not np.isnan(ev.get("mape", float("nan"))) else np.nan,
                    "R²": round(ev.get("r2", float("nan")), 4) if not np.isnan(ev.get("r2", float("nan"))) else np.nan,
                    "Resid Mean": round(diag.get("residual_mean", float("nan")), 4) if not np.isnan(diag.get("residual_mean", float("nan"))) else np.nan,
                    "Resid Std": round(diag.get("residual_std", float("nan")), 4) if not np.isnan(diag.get("residual_std", float("nan"))) else np.nan,
                    "Ljung-Box Q": round(diag.get("ljung_box_q", float("nan")), 2) if not np.isnan(diag.get("ljung_box_q", float("nan"))) else np.nan,
                    "Ljung-Box p": round(diag.get("ljung_box_p", float("nan")), 4) if not np.isnan(diag.get("ljung_box_p", float("nan"))) else np.nan,
                    "Jarque-Bera p": round(diag.get("jarque_bera_p", float("nan")), 4) if not np.isnan(diag.get("jarque_bera_p", float("nan"))) else np.nan,
                    "Het p": round(diag.get("heteroskedasticity_p", float("nan")), 4) if not np.isnan(diag.get("heteroskedasticity_p", float("nan"))) else np.nan,
                    "White Noise": (
                        "Pass (Đạt)"
                        if (status == "SUCCESS" and diag.get("is_white_noise", False))
                        else ("Fail (Không đạt)" if status == "SUCCESS" else "Failed Run")
                    ),
                    "Converged": bool(stat.get("converged", False)),
                    "Duration (s)": round(ex.get("duration_seconds", 0.0), 4),
                }
            )
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(by=["RMSE", "MAE"], ascending=[True, True], na_position="last").reset_index(drop=True)
        return df

    def generate_automatic_interpretation(self) -> Dict[str, Any]:
        """
        Purpose:
            Evaluate all registered models across each distinct statistical and
            forecasting criterion without declaring an arbitrary single 'overall winner'
            when criteria conflict.
        """
        df = self.get_comparison_dataframe()
        valid_df = df[df["Status"] == "SUCCESS"].dropna(subset=["RMSE", "MAE"]) if not df.empty else pd.DataFrame()
        if valid_df.empty:
            return {"summary_markdown": "Chưa có mô hình thành công nào được đăng ký.", "best_by_criterion": {}}

        best_rmse_row = valid_df.loc[valid_df["RMSE"].idxmin()]
        best_mae_row = valid_df.loc[valid_df["MAE"].idxmin()]
        best_mape_row = valid_df.loc[valid_df["MAPE (%)"].idxmin()]
        best_r2_row = valid_df.loc[valid_df["R²"].idxmax()]

        aic_by_group = {}
        for grp, grp_df in valid_df.groupby("Scale_Group"):
            if grp_df["AIC"].notna().any():
                b_aic = grp_df.loc[grp_df["AIC"].idxmin()]
                b_bic = grp_df.loc[grp_df["BIC"].idxmin()]
                aic_by_group[str(grp)] = {
                    "lowest_aic_model": str(b_aic["Model"]),
                    "aic": float(b_aic["AIC"]),
                    "lowest_bic_model": str(b_bic["Model"]),
                    "bic": float(b_bic["BIC"]),
                    "test_rmse_of_lowest_aic": float(b_aic["RMSE"]),
                }

        passed_lb_df = valid_df[valid_df["Ljung-Box p"] >= 0.05]
        passed_lb_models = passed_lb_df["Model"].tolist()
        failed_lb_models = valid_df[valid_df["Ljung-Box p"] < 0.05]["Model"].tolist()
        non_converged_models = df[~df["Converged"]]["Model"].tolist()
        failed_runs = df[df["Status"] != "SUCCESS"]["Model"].tolist()

        # Best RMSE among models that pass Ljung-Box white noise & converge
        adequate_df = valid_df[(valid_df["Ljung-Box p"] >= 0.05) & (valid_df["Converged"])]
        best_adequate_rmse_model = str(adequate_df.loc[adequate_df["RMSE"].idxmin()]["Model"]) if not adequate_df.empty else None

        best_by_criterion = {
            "lowest_test_rmse": {"model": str(best_rmse_row["Model"]), "value": float(best_rmse_row["RMSE"])},
            "lowest_test_mae": {"model": str(best_mae_row["Model"]), "value": float(best_mae_row["MAE"])},
            "lowest_test_mape": {"model": str(best_mape_row["Model"]), "value": float(best_mape_row["MAPE (%)"])},
            "highest_test_r2": {"model": str(best_r2_row["Model"]), "value": float(best_r2_row["R²"])},
            "lowest_test_rmse_with_white_noise_adequacy": best_adequate_rmse_model,
            "best_information_criteria_by_scale_group": aic_by_group,
        }

        md_lines = [
            f"### Diễn giải Đa Tiêu chí Tự động (Run ID: `{self.run_id}` — {self.dataset_title})",
            "",
            "#### A. Mô hình Tốt nhất theo Từng Tiêu chí Độc lập (Criterion-Specific Leaders)",
            f"- **Tiêu chí 1 — Sai số RMSE trên tập Test thấp nhất (Thang đo gốc)**: **{best_rmse_row['Model']}** (`RMSE = {best_rmse_row['RMSE']:.2f}`, `R² = {best_rmse_row['R²']:.4f}`).",
            f"- **Tiêu chí 2 — Sai số tuyệt đối MAE trên tập Test thấp nhất**: **{best_mae_row['Model']}** (`MAE = {best_mae_row['MAE']:.2f}`).",
            f"- **Tiêu chí 3 — Sai số phần trăm MAPE (%) trên tập Test thấp nhất**: **{best_mape_row['Model']}** (`MAPE = {best_mape_row['MAPE (%)']:.2f}%`).",
            f"- **Tiêu chí 4 — Mô hình thỏa mãn đồng thời Chuẩn Nhiễu trắng (Ljung-Box $p \\ge 0.05$), Hội tụ và có RMSE thấp nhất**: **{best_adequate_rmse_model or 'Không có'}**.",
            "- **Tiêu chí 5 — Tiêu chuẩn Thông tin (AIC / BIC) trên Tập Train theo từng nhóm thang đo hợp lệ (`Scale_Group`)**:",
        ]
        for grp, info in aic_by_group.items():
            md_lines.append(
                f"  - Nhóm `{grp}`: AIC thấp nhất là **{info['lowest_aic_model']}** (`AIC = {info['aic']:.2f}`, `Test RMSE = {info['test_rmse_of_lowest_aic']:.2f}`); "
                f"BIC thấp nhất là **{info['lowest_bic_model']}** (`BIC = {info['bic']:.2f}`)."
            )

        md_lines.extend(
            [
                "",
                "#### B. Kiểm định Phần dư & Trạng thái Hội tụ",
                f"- **Các mô hình vượt qua kiểm định Ljung-Box ($p \\ge 0.05$, phần dư đạt chuẩn White Noise)**: {', '.join(f'`{m}`' for m in passed_lb_models) if passed_lb_models else 'Không có'}.",
                f"- **Các mô hình vi phạm kiểm định Ljung-Box ($p < 0.05$, phần dư còn tự tương quan)**: {', '.join(f'`{m}`' for m in failed_lb_models) if failed_lb_models else 'Không có'}.",
                f"- **Các mô hình có cảnh báo không hội tụ (`Converged = False`)**: {', '.join(f'`{m}`' for m in non_converged_models) if non_converged_models else 'Không có (100% mô hình hội tụ)'}.",
                f"- **Các mô hình gặp lỗi thực thi (`Status = FAILED`)**: {', '.join(f'`{m}`' for m in failed_runs) if failed_runs else 'Không có (0 lỗi)'}.",
                "",
                "#### C. Lưu ý Phương pháp luận (Tại sao không khẳng định một mô hình tốt nhất tuyệt đối duy nhất?)",
                "1. **Không so sánh chéo AIC/BIC giữa các nhóm khác bậc sai phân hoặc khác phép biến đổi**: Giá trị Log-Likelihood của nhóm `log_d1_D1_s12` được tính trên biến $\\ln(Y_t)$ nên có AIC âm, không thể đem so sánh trực tiếp với AIC dương của nhóm `original_d1_D1_s12` hoặc `original_d1`.",
                "2. **Sự đánh đổi giữa Statistical Fit (AIC/BIC), Residual Adequacy (Ljung-Box) và Out-of-Sample Error (RMSE/MAE)**: Một mô hình có thể đạt RMSE thấp nhất trên tập Test nhờ biến đổi Logarit nhưng phần dư trên không gian Log vẫn còn tự tương quan nhẹ (như trên Dataset 1), hoặc một mô hình bậc cao có AIC thấp nhất nhưng tốn nhiều tham số hơn cấu hình tối giản (như `SARIMA(2,1,3)` so với `SARIMA(0,1,1)`). Do đó, việc lựa chọn mô hình cuối cùng phụ thuộc vào mục tiêu ưu tiên độ chính xác dự báo ngoài mẫu hay tính chặt chẽ của giả định nhiễu trắng.",
            ]
        )

        return {
            "run_id": self.run_id,
            "best_rmse_model": str(best_rmse_row["Model"]),
            "best_mae_model": str(best_mae_row["Model"]),
            "best_by_criterion": best_by_criterion,
            "best_aic_by_scale_group": aic_by_group,
            "passed_ljung_box_models": passed_lb_models,
            "failed_ljung_box_models": failed_lb_models,
            "non_converged_models": non_converged_models,
            "failed_runs": failed_runs,
            "recommended_model": best_adequate_rmse_model or str(best_rmse_row["Model"]),
            "summary_markdown": "\n".join(md_lines),
        }

    def export_all(
        self,
        base_dir: Optional[str | Path] = None,
        save_run_tagged_copy: bool = True,
    ) -> Dict[str, str]:
        """
        Purpose:
            Export all collected model summary records, comparison tables, and
            multi-criterion interpretations to:
            - Canonical latest files:
                reports/model_summary/<dataset_key>/<dataset_key>_model_summary.{csv,json,md}
            - Run-tagged archive files (when save_run_tagged_copy=True):
                reports/model_summary/<dataset_key>/runs/experiment_<RUN_ID>_summary.{csv,json,md}
        """
        root = Path(base_dir).resolve() if base_dir is not None else get_project_root()
        out_dir = root / "reports" / "model_summary" /"lab02" / self.dataset_key
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_path = out_dir / f"{self.dataset_key}_model_summary.csv"
        json_path = out_dir / f"{self.dataset_key}_model_summary.json"
        md_path = out_dir / f"{self.dataset_key}_model_summary.md"

        comp_df = self.get_comparison_dataframe()
        interp = self.generate_automatic_interpretation()

        # 1. Xuất CSV
        comp_df.to_csv(csv_path, index=False, encoding="utf-8")

        # 2. Xuất JSON
        json_payload = {
            "run_id": self.run_id,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_info": self.dataset_info,
            "total_candidates": len(self.records),
            "succeeded_count": int((comp_df["Status"] == "SUCCESS").sum()) if not comp_df.empty else 0,
            "failed_count": int((comp_df["Status"] != "SUCCESS").sum()) if not comp_df.empty else 0,
            "automatic_interpretation": interp,
            "comparison_table": comp_df.to_dict(orient="records"),
            "models": list(self.records.values()),
        }
        with open(json_path, "w", encoding="utf-8") as f_json:
            json.dump(json_payload, f_json, indent=2, ensure_ascii=False)

        # 3. Xuất Markdown
        core_cols = [
            "Model",
            "Family",
            "Status",
            "Scale_Group",
            "AIC",
            "BIC",
            "MAE",
            "RMSE",
            "MAPE (%)",
            "R²",
            "Ljung-Box p",
            "White Noise",
            "Converged",
        ]
        valid_cols = [c for c in core_cols if c in comp_df.columns]
        try:
            md_table = comp_df[valid_cols].to_markdown(index=False) if not comp_df.empty else "No models registered."
        except Exception:
            md_table = comp_df[valid_cols].to_string(index=False) if not comp_df.empty else "No models registered."

        md_content = [
            f"# BÁO CÁO TỔNG HỢP MÔ HÌNH (MODEL SUMMARY) — {self.dataset_title.upper()}",
            "",
            "## 1. Thông tin Phiên Thực nghiệm & Bộ dữ liệu (Provenance & Dataset Info)",
            f"- **Run ID**: `{self.run_id}`",
            f"- **Thời điểm xuất báo cáo**: `{json_payload['generated_at']}`",
            f"- **Dataset Key**: `{self.dataset_info['dataset_name']}`",
            f"- **Tên bộ dữ liệu**: {self.dataset_info['dataset_title']}",
            f"- **Tổng số quan sát**: {self.dataset_info['n_obs']} tháng ({self.dataset_info['date_range']})",
            f"- **Tần suất**: `{self.dataset_info['frequency']}` (Monthly Start)",
            f"- **Tập Huấn luyện (Train - 80%)**: {self.dataset_info['train_size']} quan sát",
            f"- **Tập Kiểm thử (Test - 20%)**: {self.dataset_info['test_size']} quan sát",
            f"- **Tổng số mô hình ứng viên**: {json_payload['total_candidates']} (Thành công: {json_payload['succeeded_count']}, Thất bại: {json_payload['failed_count']})",
            "",
            "## 2. Bảng So sánh Tổng hợp các Mô hình Ứng viên (Model Comparison Table)",
            "",
            "> **Nguyên tắc đối chiếu**: Tất cả chỉ số dự báo ngoài mẫu (`MAE`, `RMSE`, `MAPE (%)`, `R²`) đều được đánh giá trên **cùng tập Test và cùng thang đo gốc** (đã áp dụng biến đổi ngược $\\exp(\\cdot)$ cho các mô hình `Log-SARIMA`). Chỉ số `AIC` và `BIC` chỉ được so sánh trực tiếp trong cùng một nhóm `Scale_Group`.",
            "",
            md_table,
            "",
            interp["summary_markdown"],
            "",
            "## 3. Hồ sơ Thống kê Chi tiết Từng Mô hình (Individual Model Cards)",
            "",
        ]

        for idx, (name, rec) in enumerate(self.records.items(), 1):
            md_content.append(f"### 3.{idx}. {name}")
            md_content.append("```text")
            md_content.append(format_terminal_model_card(rec))
            md_content.append("```")
            md_content.append("")

        full_md_str = "\n".join(md_content)
        with open(md_path, "w", encoding="utf-8") as f_md:
            f_md.write(full_md_str)

        paths = {"csv": str(csv_path), "json": str(json_path), "md": str(md_path)}

        if save_run_tagged_copy:
            runs_dir = out_dir / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            run_csv = runs_dir / f"experiment_{self.run_id}_summary.csv"
            run_json = runs_dir / f"experiment_{self.run_id}_summary.json"
            run_md = runs_dir / f"experiment_{self.run_id}_summary.md"
            comp_df.to_csv(run_csv, index=False, encoding="utf-8")
            with open(run_json, "w", encoding="utf-8") as f_rj:
                json.dump(json_payload, f_rj, indent=2, ensure_ascii=False)
            with open(run_md, "w", encoding="utf-8") as f_rm:
                f_rm.write(full_md_str)
            paths["run_csv"] = str(run_csv)
            paths["run_json"] = str(run_json)
            paths["run_md"] = str(run_md)

        log_system_status(
            dataset_name=self.dataset_key,
            status="SUCCESS",
            output_paths=paths,
            message=f"Exported Model Summary ({len(self.records)} models) for {self.dataset_title} [run_id={self.run_id}]",
        )
        return paths
