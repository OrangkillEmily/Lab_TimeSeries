# src/lab02_models/ma_model.py
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

try:
    from statsmodels.tsa.stattools import adfuller
except ImportError:
    from src.lab02.lab02_utils.scipy_tsa_compat import adfuller

from src.lab02.lab02_utils.logging_config import get_lab02_logger, log_model_evaluation
from src.lab02.lab02_utils.modeling import (
    fit_arima,
    residual_diagnostics,
    forecast_test,
    evaluate_forecast,
    plot_forecast,
    extract_model_coefficients,
)


def fit_ma_model(
    train: pd.Series,
    q: int = 1,
    d: int = 1,
    dataset_name: str = "dataset1",
    check_stationarity: bool = True,
    verbose: bool = True,
    **fit_kwargs,
):
    """
    Purpose:
        Validate inputs and fit a Moving Average MA(q) model formulated as
        ARIMA(0, d, q) using statsmodels.tsa.arima.model.ARIMA.

    Parameters:
        train (pd.Series): Chronological training time series.
        q (int): Moving average lag order (must be >= 1).
        d (int): Differencing order required to achieve trend stationarity (default 1).
        dataset_name (str): Dataset identifier ('dataset1' or 'dataset2') for logging.
        check_stationarity (bool): If True and d == 0, verify train stationarity via ADF.
        verbose (bool): Whether to print summary output to console.
        **fit_kwargs: Optional keyword arguments passed to ARIMA.fit().

    Returns:
        ARIMAResults: Fitted statsmodels result object for ARIMA(0, d, q).
    """
    logger = get_lab02_logger(dataset_name, "ma_model")
    if train is None or len(pd.Series(train).dropna()) < 10:
        raise ValueError("Training series for MA model must have at least 10 valid observations.")
    if not isinstance(q, int) or q < 1:
        raise ValueError(f"MA order q must be a positive integer (>= 1), got q={q}.")
    if not isinstance(d, int) or d < 0 or d > 2:
        raise ValueError(f"Differencing order d must be 0, 1, or 2, got d={d}.")

    clean_train = pd.Series(train).astype(float).dropna()
    if d == 0 and check_stationarity:
        adf_p = float(adfuller(clean_train, autolag="AIC")[1])
        if adf_p >= 0.05:
            msg = (
                f"Statistical Guardrail Triggered: Attempted to fit MA({q}) with d=0 on a "
                f"non-stationary series (ADF p-value = {adf_p:.4f} >= 0.05). Please use d >= 1."
            )
            logger.error(msg)
            raise ValueError(msg)

    order = (0, int(d), int(q))
    try:
        return fit_arima(
            train=clean_train,
            order=order,
            dataset_name=dataset_name,
            model_family="MA",
            verbose=verbose,
            **fit_kwargs,
        )
    except Exception as exc:
        logger.exception(f"Failed to fit MA({q}) [ARIMA{order}] on {dataset_name}: {exc}")
        raise


def run_ma_experiment(
    train: pd.Series,
    test: pd.Series,
    q: int = 1,
    d: int = 1,
    dataset_name: str = "dataset1",
    diagnostic_lags: int = 24,
    fig_dir: Optional[str | Path] = None,
    show_plot: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Purpose:
        Execute a complete end-to-end MA(q) experiment: fit ARIMA(0, d, q) on Train,
        extract parameter table, run residual diagnostics, forecast Test set,
        compute evaluation metrics (MAE, RMSE, MAPE, R²), log results, and return
        a structured record ready for ModelSummaryRegistry.

    Parameters:
        train (pd.Series): Training time series.
        test (pd.Series): Testing time series.
        q (int): Moving Average order q (>= 1).
        d (int): Differencing order d (default 1).
        dataset_name (str): Dataset identifier ('dataset1' or 'dataset2').
        diagnostic_lags (int): Number of lags for Ljung-Box and residual ACF (default 24).
        fig_dir (Optional[str | Path]): Directory to save diagnostic and forecast PNGs.
        show_plot (bool): Whether to render plots inline.
        verbose (bool): Whether to print metrics to stdout.

    Returns:
        Dict[str, Any]: Standardized experiment record for ModelSummaryRegistry.
    """
    order = (0, int(d), int(q))
    model_name = f"MA({q}) [ARIMA(0,{d},{q})]"
    result = fit_ma_model(train=train, q=q, d=d, dataset_name=dataset_name, verbose=verbose)

    out_dir = Path(fig_dir) if fig_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    resid_path = str(out_dir / f"{dataset_name}_ma_{q}_d{d}_residuals.png") if out_dir else None
    fc_path = str(out_dir / f"{dataset_name}_ma_{q}_d{d}_forecast.png") if out_dir else None

    diag = residual_diagnostics(
        result=result,
        order=order,
        lags=diagnostic_lags,
        title=f"{model_name} ({dataset_name.upper()})",
        save_path=resid_path,
        show_plot=show_plot,
    )

    fc_mean, conf_int = forecast_test(result, steps=len(test))
    fc_mean.index = test.index
    conf_int.index = test.index

    plot_forecast(
        train=train,
        test=test,
        forecast=fc_mean,
        conf_int=conf_int,
        title=f"Dự báo {model_name} — {dataset_name.upper()}",
        save_path=fc_path,
        show_plot=show_plot,
    )

    metrics = evaluate_forecast(test, fc_mean)
    coef_df = extract_model_coefficients(result)

    log_model_evaluation(
        dataset_name=dataset_name,
        model_name=model_name,
        mae=metrics["MAE"],
        rmse=metrics["RMSE"],
        mape=metrics["MAPE (%)"],
        r2=metrics["R2"],
        resid_mean=diag["mean"],
        resid_std=diag["std"],
        lb_pvalue=diag["ljung_box_p"],
        jb_pvalue=diag.get("jarque_bera_p"),
        het_pvalue=diag.get("heteroskedasticity_p"),
        extra_notes="Pure Moving Average baseline model",
    )

    return {
        "model_name": model_name,
        "model_family": "MA",
        "order": order,
        "seasonal_order": (0, 0, 0, 0),
        "transformation": "Original Scale (Sai phân bậc d=" + str(d) + ")",
        "scale_for_ic": "original_d" + str(d),
        "result": result,
        "coefficients": coef_df,
        "diagnostics": diag,
        "forecast": fc_mean,
        "conf_int": conf_int,
        "metrics": metrics,
        "figure_paths": {"residuals": resid_path, "forecast": fc_path},
    }