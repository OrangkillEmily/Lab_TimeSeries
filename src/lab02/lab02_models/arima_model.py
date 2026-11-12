# src/lab02_models/arima_model.py
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import pandas as pd

from src.lab02.lab02_utils.logging_config import get_lab02_logger, log_model_evaluation
from src.lab02.lab02_utils.modeling import (
    fit_arima,
    residual_diagnostics,
    forecast_test,
    evaluate_forecast,
    plot_forecast,
    extract_model_coefficients,
)


def fit_arima_model(
    train: pd.Series,
    order: Tuple[int, int, int] = (1, 1, 1),
    dataset_name: str = "dataset1",
    verbose: bool = True,
    **fit_kwargs,
):
    """
    Purpose:
        Validate inputs and train an ARIMA(p, d, q) model on the chronological
        training series, logging parameters, AIC/BIC, Log-Likelihood, and convergence.

    Parameters:
        train (pd.Series): Chronological training series.
        order (Tuple[int, int, int]): Non-seasonal order (p, d, q).
        dataset_name (str): Dataset identifier ('dataset1' or 'dataset2').
        verbose (bool): Whether to print summary output to console.
        **fit_kwargs: Additional arguments passed to statsmodels ARIMA.fit().

    Returns:
        ARIMAResults: Fitted statsmodels result object.
    """
    logger = get_lab02_logger(dataset_name, "arima_model")
    if train is None or len(pd.Series(train).dropna()) < 10:
        raise ValueError("Training series for ARIMA model must contain at least 10 observations.")
    if not isinstance(order, (tuple, list)) or len(order) != 3 or any(int(x) < 0 for x in order):
        raise ValueError(f"Invalid ARIMA order={order}. Expected 3 non-negative integers (p, d, q).")

    try:
        return fit_arima(
            train=train,
            order=tuple(int(x) for x in order),
            dataset_name=dataset_name,
            model_family="ARIMA",
            verbose=verbose,
            **fit_kwargs,
        )
    except Exception as exc:
        logger.exception(f"Failed to fit ARIMA{order} on {dataset_name}: {exc}")
        raise


def run_arima_experiment(
    train: pd.Series,
    test: pd.Series,
    order: Tuple[int, int, int] = (1, 1, 1),
    dataset_name: str = "dataset1",
    model_name: Optional[str] = None,
    diagnostic_lags: int = 24,
    fig_dir: Optional[str | Path] = None,
    show_plot: bool = True,
    verbose: bool = True,
    rationale: str = "Candidate model identified from short-run ACF/PACF cut-offs",
) -> Dict[str, Any]:
    """
    Purpose:
        Execute a complete end-to-end ARIMA(p, d, q) experiment: fit model on Train,
        extract coefficients/std_err/p-values, perform residual diagnostics, forecast
        on Test set with 95% confidence intervals, evaluate MAE/RMSE/MAPE/R², log all
        metrics, and return a structured record for ModelSummaryRegistry.

    Parameters:
        train (pd.Series): Training time series.
        test (pd.Series): Testing time series.
        order (Tuple[int, int, int]): (p, d, q) order tuple.
        dataset_name (str): Dataset identifier ('dataset1' or 'dataset2').
        model_name (Optional[str]): Custom label (default 'ARIMA(p,d,q)').
        diagnostic_lags (int): Lag order for Ljung-Box and residual ACF (default 24).
        fig_dir (Optional[str | Path]): Directory to save residual and forecast figures.
        show_plot (bool): Whether to display figures inline.
        verbose (bool): Whether to print summary text to stdout.
        rationale (str): Statistical rationale for selecting this candidate order.

    Returns:
        Dict[str, Any]: Standardized experiment record for ModelSummaryRegistry.
    """
    order = (int(order[0]), int(order[1]), int(order[2]))
    name = model_name or f"ARIMA({order[0]},{order[1]},{order[2]})"
    tag = f"arima{order[0]}{order[1]}{order[2]}"

    result = fit_arima_model(train=train, order=order, dataset_name=dataset_name, verbose=verbose)

    out_dir = Path(fig_dir) if fig_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    resid_path = str(out_dir / f"ARIMA_residuals_{dataset_name}_{tag}.png") if out_dir else None
    fc_path = str(out_dir / f"ARIMA_Train_Test_Forecast_{dataset_name}_{tag}.png") if out_dir else None

    diag = residual_diagnostics(
        result=result,
        order=order,
        lags=diagnostic_lags,
        title=f"{name} ({dataset_name.upper()})",
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
        title=f"{name}: Train, Test (thực tế) và Forecast — {dataset_name.upper()}",
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

    return {
        "model_name": name,
        "model_family": "ARIMA",
        "order": order,
        "seasonal_order": (0, 0, 0, 0),
        "transformation": f"Original Scale (Sai phân bậc d={order[1]})",
        "scale_for_ic": f"original_d{order[1]}",
        "rationale": rationale,
        "result": result,
        "coefficients": coef_df,
        "diagnostics": diag,
        "forecast": fc_mean,
        "conf_int": conf_int,
        "metrics": metrics,
        "figure_paths": {"residuals": resid_path, "forecast": fc_path},
    }