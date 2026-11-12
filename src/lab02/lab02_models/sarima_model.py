# src/lab02_models/sarima_model.py
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd

try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.stats.stattools import jarque_bera
except ImportError:
    from src.lab02.lab02_utils.scipy_tsa_compat import acorr_ljungbox, jarque_bera

from src.lab02.lab02_utils.logging_config import get_lab02_logger, log_model_evaluation
from src.lab02.lab02_utils.modeling import (
    fit_sarima,
    residual_diagnostics,
    forecast_test,
    evaluate_forecast,
    plot_forecast,
    extract_model_coefficients,
)


def fit_sarima_model(
    train: pd.Series,
    order: Tuple[int, int, int] = (0, 1, 1),
    seasonal_order: Tuple[int, int, int, int] = (0, 1, 1, 12),
    use_log_transform: bool = False,
    dataset_name: str = "dataset1",
    verbose: bool = True,
    **fit_kwargs,
):
    """
    Purpose:
        Validate inputs and fit a SARIMA(p, d, q)(P, D, Q, s) model (on original
        or natural log scale) using statsmodels SARIMAX, capturing convergence
        warnings and logging full training metadata.

    Parameters:
        train (pd.Series): Chronological training series (on original scale if use_log_transform=True).
        order (Tuple[int, int, int]): Non-seasonal order (p, d, q).
        seasonal_order (Tuple[int, int, int, int]): Seasonal order (P, D, Q, s).
        use_log_transform (bool): If True, apply np.log(train) prior to fitting.
        dataset_name (str): Dataset identifier ('dataset1' or 'dataset2').
        verbose (bool): Whether to print summary output to console.
        **fit_kwargs: Optional keyword arguments passed to SARIMAX.fit().

    Returns:
        SARIMAXResults: Fitted statsmodels result object.
    """
    logger = get_lab02_logger(dataset_name, "sarima_model")
    if train is None or len(pd.Series(train).dropna()) < 24:
        raise ValueError("Training series for SARIMA must contain at least 24 observations.")

    clean_train = pd.Series(train).astype(float).dropna()
    if use_log_transform:
        if (clean_train <= 0).any():
            raise ValueError("Log transformation requires all training observations to be strictly positive (> 0).")
        fit_series = np.log(clean_train)
        family = "Log-SARIMA"
    else:
        fit_series = clean_train
        family = "SARIMA"

    try:
        return fit_sarima(
            train=fit_series,
            order=order,
            seasonal_order=seasonal_order,
            dataset_name=dataset_name,
            model_family=family,
            verbose=verbose,
            **fit_kwargs,
        )
    except Exception as exc:
        logger.exception(f"Failed to fit {family}{order}x{seasonal_order} on {dataset_name}: {exc}")
        raise


def run_sarima_experiment(
    train: pd.Series,
    test: pd.Series,
    order: Tuple[int, int, int] = (0, 1, 1),
    seasonal_order: Tuple[int, int, int, int] = (0, 1, 1, 12),
    use_log_transform: bool = False,
    dataset_name: str = "dataset1",
    model_name: Optional[str] = None,
    diagnostic_lags: int = 24,
    fig_dir: Optional[str | Path] = None,
    plot_diagnostics_flag: bool = True,
    plot_forecast_flag: bool = True,
    show_plot: bool = True,
    verbose: bool = True,
    rationale: str = "Seasonal candidate model with d=1, D=1, s=12",
) -> Dict[str, Any]:
    """
    Purpose:
        Execute a complete end-to-end SARIMA or Log-SARIMA experiment: fit model
        on Train (applying np.log if use_log_transform=True), run residual diagnostics
        (skipping d + D*s initial observations), forecast Test horizon (applying np.exp
        inverse transformation before computing metrics when use_log_transform=True),
        log evaluation results, and return a standardized record for ModelSummaryRegistry.

    Parameters:
        train (pd.Series): Original-scale training time series.
        test (pd.Series): Original-scale testing time series.
        order (Tuple[int, int, int]): Non-seasonal order (p, d, q).
        seasonal_order (Tuple[int, int, int, int]): Seasonal order (P, D, Q, s).
        use_log_transform (bool): Whether to train on ln(Y_t) and inverse-transform via exp().
        dataset_name (str): Dataset identifier ('dataset1' or 'dataset2').
        model_name (Optional[str]): Custom model name if provided.
        diagnostic_lags (int): Lag order for Ljung-Box test (default 24).
        fig_dir (Optional[str | Path]): Directory to save diagnostic and forecast plots.
        plot_diagnostics_flag (bool): Whether to generate residual diagnostic plot.
        plot_forecast_flag (bool): Whether to generate forecast comparison plot.
        show_plot (bool): Whether to display plots inline via plt.show().
        verbose (bool): Whether to print summary text to stdout.
        rationale (str): Statistical rationale for this candidate configuration.

    Returns:
        Dict[str, Any]: Standardized experiment record for ModelSummaryRegistry.
    """
    order = (int(order[0]), int(order[1]), int(order[2]))
    seasonal_order = (
        int(seasonal_order[0]),
        int(seasonal_order[1]),
        int(seasonal_order[2]),
        int(seasonal_order[3]),
    )
    p, d, q = order
    P, D, Q, s = seasonal_order
    prefix = "Log-SARIMA" if use_log_transform else "SARIMA"
    name = model_name or f"{prefix}({p},{d},{q})x({P},{D},{Q})_{s}"
    safe_tag = name.replace("(", "_").replace(")", "_").replace(",", "_").replace(" ", "")

    result = fit_sarima_model(
        train=train,
        order=order,
        seasonal_order=seasonal_order,
        use_log_transform=use_log_transform,
        dataset_name=dataset_name,
        verbose=verbose,
    )

    out_dir = Path(fig_dir) if fig_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    resid_path = str(out_dir / f"{dataset_name}_{safe_tag}_residuals.png") if (out_dir and plot_diagnostics_flag) else None
    fc_path = str(out_dir / f"{dataset_name}_{safe_tag}_forecast.png") if (out_dir and plot_forecast_flag) else None

    n_skip = d + D * s
    if plot_diagnostics_flag:
        diag = residual_diagnostics(
            result=result,
            order=order,
            lags=diagnostic_lags,
            title=f"{name} ({dataset_name.upper()})",
            save_path=resid_path,
            n_skip=n_skip,
            seasonal_order=seasonal_order,
            show_plot=show_plot,
        )
    else:
        resid = pd.Series(result.resid).iloc[n_skip:].dropna()
        eff_lags = min(int(diagnostic_lags), max(1, (len(resid) // 2) - 1))
        lb_df = acorr_ljungbox(resid, lags=[eff_lags], return_df=True)
        lb_q = float(lb_df["lb_stat"].iloc[0])
        lb_p = float(lb_df["lb_pvalue"].iloc[0])
        jb_stat, jb_p, skew_val, kurt_val = jarque_bera(resid)
        het_stat, het_p = float("nan"), float("nan")
        try:
            het_res = result.test_heteroskedasticity(method="breakvar")
            het_stat, het_p = float(het_res[0, 0]), float(het_res[0, 1])
        except Exception:
            pass
        diag = {
            "mean": float(resid.mean()),
            "std": float(resid.std()),
            "ljung_box_q": lb_q,
            "ljung_box_p": lb_p,
            "ljung_box_lag": eff_lags,
            "n_estimated_params": p + q + P + Q,
            "jarque_bera_stat": float(jb_stat),
            "jarque_bera_p": float(jb_p),
            "skew": float(skew_val),
            "kurtosis": float(kurt_val),
            "heteroskedasticity_stat": het_stat,
            "heteroskedasticity_p": het_p,
            "is_white_noise": bool(lb_p >= 0.05),
        }

    fc_mean, conf_int = forecast_test(result, steps=len(test), inverse_log=use_log_transform)
    fc_mean.index = test.index
    conf_int.index = test.index

    if plot_forecast_flag:
        plot_forecast(
            train=train,
            test=test,
            forecast=fc_mean,
            conf_int=conf_int,
            title=f"Biểu đồ dự báo mô hình {name} ({dataset_name.upper()})",
            save_path=fc_path,
            show_plot=show_plot,
        )

    metrics = evaluate_forecast(test, fc_mean)
    coef_df = extract_model_coefficients(result)

    log_model_evaluation(
        dataset_name=dataset_name,
        model_name=name,
        mae=metrics["MAE"],
        rmse=metrics["RMSE"],
        mape=metrics["MAPE (%)"],
        r2=metrics["R2"],
        resid_mean=diag["mean"],
        resid_std=diag["std"],
        lb_pvalue=diag["ljung_box_p"],
        jb_pvalue=diag.get("jarque_bera_p"),
        het_pvalue=diag.get("heteroskedasticity_p"),
        extra_notes=rationale,
    )

    transform_desc = (
        f"Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12"
        if use_log_transform
        else f"Original Scale | d={d}, D={D}, s={s}"
    )
    scale_ic = f"log_d{d}_D{D}_s{s}" if use_log_transform else f"original_d{d}_D{D}_s{s}"

    return {
        "model_name": name,
        "model_family": prefix,
        "order": order,
        "seasonal_order": seasonal_order,
        "transformation": transform_desc,
        "scale_for_ic": scale_ic,
        "rationale": rationale,
        "result": result,
        "coefficients": coef_df,
        "diagnostics": diag,
        "forecast": fc_mean,
        "conf_int": conf_int,
        "metrics": metrics,
        "figure_paths": {"residuals": resid_path, "forecast": fc_path},
    }