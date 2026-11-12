import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats as scipy_stats

try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except ImportError:
    from src.lab02.lab02_utils.scipy_tsa_compat import mean_absolute_error, mean_squared_error, r2_score

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.graphics.tsaplots import plot_acf
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
    from statsmodels.stats.stattools import jarque_bera
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError:
    from src.lab02.lab02_utils.scipy_tsa_compat import (
        ARIMA,
        SARIMAX,
        plot_acf,
        acorr_ljungbox,
        het_arch,
        jarque_bera,
    )

from src.lab02.lab02_utils.logging_config import (
    log_model_training,
    log_model_evaluation,
)


def extract_model_coefficients(result) -> pd.DataFrame:
    """
    Purpose:
        Extract parameter estimates, standard errors, z-statistics, p-values,
        and 95% confidence intervals from a fitted statsmodels ARIMA or SARIMAX result.

    Parameters:
        result: Fitted ARIMAResults or SARIMAXResults object.

    Returns:
        pd.DataFrame: Table of parameters with columns
            ['param', 'coef', 'std_err', 'z_stat', 'p_value', 'ci_lower', 'ci_upper'].
    """
    params = result.params
    bse = getattr(result, "bse", pd.Series(np.nan, index=params.index))
    zvalues = getattr(result, "zvalues", pd.Series(np.nan, index=params.index))
    pvalues = getattr(result, "pvalues", pd.Series(np.nan, index=params.index))
    conf = result.conf_int(alpha=0.05)

    rows = []
    for name in params.index:
        rows.append(
            {
                "param": str(name),
                "coef": float(params[name]),
                "std_err": float(bse[name]) if name in bse else float("nan"),
                "z_stat": float(zvalues[name]) if name in zvalues else float("nan"),
                "p_value": float(pvalues[name]) if name in pvalues else float("nan"),
                "ci_lower": float(conf.loc[name].iloc[0]) if name in conf.index else float("nan"),
                "ci_upper": float(conf.loc[name].iloc[1]) if name in conf.index else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def fit_arima(
    train: pd.Series,
    order: tuple = (1, 1, 1),
    dataset_name: Optional[str] = None,
    model_family: str = "ARIMA",
    verbose: bool = True,
    **fit_kwargs,
):
    """
    Purpose:
        Fit an ARIMA(p, d, q) model on the training series using statsmodels ARIMA,
        capture convergence status and warnings, log training telemetry, and return
        the fitted ARIMAResults object (with attached _lab02_meta dictionary).

    Parameters:
        train (pd.Series): Chronological training series.
        order (tuple): (p, d, q) order tuple.
        dataset_name (Optional[str]): Dataset identifier ('dataset1' or 'dataset2') for logging.
        model_family (str): Family label ('AR', 'MA', or 'ARIMA').
        verbose (bool): Whether to print summary lines to stdout.
        **fit_kwargs: Additional keyword arguments passed to ARIMA.fit().

    Returns:
        ARIMAResults: Fitted statsmodels result object with `_lab02_meta` attribute.
    """
    if train is None:
        raise ValueError("Training series cannot be None.")
    values = pd.Series(train).astype(float).dropna()
    if values.empty:
        raise ValueError("Chuỗi train rỗng sau khi loại bỏ giá trị thiếu.")
    if not isinstance(order, (tuple, list)) or len(order) != 3 or any(int(x) < 0 for x in order):
        raise ValueError(f"Invalid ARIMA order={order}. Expected 3 non-negative integers (p, d, q).")

    order = (int(order[0]), int(order[1]), int(order[2]))
    start_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.perf_counter()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = ARIMA(values, order=order)
        result = model.fit(**fit_kwargs)

    duration = time.perf_counter() - t0
    end_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    mle_retvals = getattr(result, "mle_retvals", None) or {}
    converged_flag = bool(mle_retvals.get("converged", True))
    warning_msgs = [str(w.message) for w in caught]
    has_conv_warning = any("onverge" in msg.lower() for msg in warning_msgs)
    is_converged = bool(converged_flag and not has_conv_warning)

    if model_family.upper() == "AR":
        model_label = f"AR({order[0]}) [ARIMA{order}]"
    elif model_family.upper() == "MA":
        model_label = f"MA({order[2]}) [ARIMA{order}]"
    else:
        model_label = f"ARIMA{order}"

    result._lab02_meta = {
        "model_name": model_label,
        "model_family": model_family,
        "order": order,
        "seasonal_order": None,
        "start_time": start_dt,
        "end_time": end_dt,
        "duration_sec": float(duration),
        "converged": is_converged,
        "warnings": warning_msgs,
        "aic": float(result.aic),
        "bic": float(result.bic),
        "hqic": float(getattr(result, "hqic", np.nan)),
        "llf": float(result.llf),
    }

    if verbose:
        print(f"===== {model_label} =====")
        print(f"AIC : {result.aic:.2f}")
        print(f"BIC : {result.bic:.2f}")
        print(f"Hoi tu: {'CO' if is_converged else 'KHONG / co canh bao'}")

    if dataset_name is not None:
        log_model_training(
            dataset_name=dataset_name,
            model_name=model_label,
            model_family=model_family,
            order=order,
            seasonal_order=None,
            start_time=start_dt,
            end_time=end_dt,
            duration_sec=duration,
            aic=float(result.aic),
            bic=float(result.bic),
            llf=float(result.llf),
            converged=is_converged,
            warnings_list=warning_msgs,
        )

    return result


def fit_sarima(
    train: pd.Series,
    order: tuple = (1, 1, 1),
    seasonal_order: tuple = (1, 1, 1, 12),
    dataset_name: Optional[str] = None,
    model_family: str = "SARIMA",
    enforce_stationarity: bool = False,
    enforce_invertibility: bool = False,
    verbose: bool = True,
    **fit_kwargs,
):
    """
    Purpose:
        Fit a SARIMA(p, d, q)(P, D, Q, s) model on the training series using
        statsmodels SARIMAX, capture convergence status and warnings, log training
        telemetry, and return the fitted SARIMAXResults object.

    Parameters:
        train (pd.Series): Chronological training series (original or log-transformed).
        order (tuple): Non-seasonal order (p, d, q).
        seasonal_order (tuple): Seasonal order (P, D, Q, s).
        dataset_name (Optional[str]): Dataset identifier ('dataset1' or 'dataset2') for logging.
        model_family (str): Family label ('SARIMA' or 'Log-SARIMA').
        enforce_stationarity (bool): Whether to enforce AR stationarity in SARIMAX.
        enforce_invertibility (bool): Whether to enforce MA invertibility in SARIMAX.
        verbose (bool): Whether to print summary lines to stdout.
        **fit_kwargs: Additional keyword arguments passed to SARIMAX.fit().

    Returns:
        SARIMAXResults: Fitted statsmodels result object with `_lab02_meta` attribute.
    """
    if train is None:
        raise ValueError("Training series cannot be None.")
    values = pd.Series(train).astype(float).dropna()
    if values.empty:
        raise ValueError("Chuỗi train rỗng sau khi loại bỏ giá trị thiếu.")
    if not isinstance(order, (tuple, list)) or len(order) != 3 or any(int(x) < 0 for x in order):
        raise ValueError(f"Invalid SARIMA non-seasonal order={order}.")
    if not isinstance(seasonal_order, (tuple, list)) or len(seasonal_order) != 4 or any(int(x) < 0 for x in seasonal_order):
        raise ValueError(f"Invalid SARIMA seasonal_order={seasonal_order}. Expected (P, D, Q, s).")

    order = (int(order[0]), int(order[1]), int(order[2]))
    seasonal_order = (int(seasonal_order[0]), int(seasonal_order[1]), int(seasonal_order[2]), int(seasonal_order[3]))
    start_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.perf_counter()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = SARIMAX(
            values,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
        )
        result = model.fit(disp=False, **fit_kwargs)

    duration = time.perf_counter() - t0
    end_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    mle_retvals = getattr(result, "mle_retvals", None) or {}
    converged_flag = bool(mle_retvals.get("converged", True))
    warning_msgs = [str(w.message) for w in caught]
    has_conv_warning = any("onverge" in msg.lower() for msg in warning_msgs)
    is_converged = bool(converged_flag and not has_conv_warning)

    prefix = "Log-SARIMA" if "log" in model_family.lower() else "SARIMA"
    model_label = f"{prefix}{order}x{seasonal_order}"

    result._lab02_meta = {
        "model_name": model_label,
        "model_family": model_family,
        "order": order,
        "seasonal_order": seasonal_order,
        "start_time": start_dt,
        "end_time": end_dt,
        "duration_sec": float(duration),
        "converged": is_converged,
        "warnings": warning_msgs,
        "aic": float(result.aic),
        "bic": float(result.bic),
        "hqic": float(getattr(result, "hqic", np.nan)),
        "llf": float(result.llf),
    }

    if verbose:
        print(f"===== {model_label} =====")
        print(f"AIC : {result.aic:.2f}")
        print(f"BIC : {result.bic:.2f}")
        print(f"Hoi tu: {'CO' if is_converged else 'KHONG / co canh bao'}")

    if dataset_name is not None:
        log_model_training(
            dataset_name=dataset_name,
            model_name=model_label,
            model_family=model_family,
            order=order,
            seasonal_order=seasonal_order,
            start_time=start_dt,
            end_time=end_dt,
            duration_sec=duration,
            aic=float(result.aic),
            bic=float(result.bic),
            llf=float(result.llf),
            converged=is_converged,
            warnings_list=warning_msgs,
        )

    return result


def residual_diagnostics(
    result,
    order: tuple,
    lags: int = 24,
    title: str = "",
    save_path: Optional[str] = None,
    n_skip: Optional[int] = None,
    seasonal_order: Optional[tuple] = None,
    show_plot: bool = True,
) -> dict:
    """
    Purpose:
        Perform comprehensive residual diagnostics on a fitted time series model:
        1. Residual time-series plot.
        2. Residual histogram with KDE and theoretical Gaussian N(mean, std) curve.
        3. Residual ACF plot with 95% confidence bounds.
        4. Compute residual mean, standard deviation, Ljung-Box test, Jarque-Bera
           normality test, and Heteroskedasticity test.

    Parameters:
        result: Fitted ARIMAResults or SARIMAXResults object.
        order (tuple): Non-seasonal order (p, d, q).
        lags (int): Lag order for ACF plot and Ljung-Box test (default 24).
        title (str): Descriptive model name for plot titles.
        save_path (Optional[str]): File path to save the diagnostic figure as PNG.
        n_skip (Optional[int]): Number of initial diffuse residuals to skip.
        seasonal_order (Optional[tuple]): Seasonal order (P, D, Q, s) if applicable.
        show_plot (bool): Whether to display the figure inline via plt.show().

    Returns:
        dict: Dictionary containing {'mean', 'std', 'ljung_box_q', 'ljung_box_p',
            'ljung_box_lag', 'jarque_bera_stat', 'jarque_bera_p', 'skew', 'kurtosis',
            'heteroskedasticity_stat', 'heteroskedasticity_p', 'is_white_noise'}.
    """
    if seasonal_order is None and hasattr(result, "_lab02_meta"):
        seasonal_order = result._lab02_meta.get("seasonal_order")

    if n_skip is None:
        d_val = int(order[1]) if len(order) > 1 else 0
        if seasonal_order is not None and len(seasonal_order) == 4:
            d_val += int(seasonal_order[1]) * int(seasonal_order[3])
        skip = d_val
    else:
        skip = int(n_skip)

    resid = pd.Series(result.resid).iloc[skip:].dropna()
    effective_lags = min(int(lags), max(1, (len(resid) // 2) - 1))

    lb_df = acorr_ljungbox(resid, lags=[effective_lags], return_df=True)
    lb_q = float(lb_df["lb_stat"].iloc[0])
    lb_p = float(lb_df["lb_pvalue"].iloc[0])

    n_arma_params = int(order[0]) + int(order[2])
    if seasonal_order is not None and len(seasonal_order) == 4:
        n_arma_params += int(seasonal_order[0]) + int(seasonal_order[2])

    jb_stat, jb_p, skew_val, kurt_val = jarque_bera(resid)
    jb_stat, jb_p, skew_val, kurt_val = float(jb_stat), float(jb_p), float(skew_val), float(kurt_val)

    het_stat, het_p = float("nan"), float("nan")
    try:
        het_res = result.test_heteroskedasticity(method="breakvar")
        if het_res is not None and len(het_res) > 0:
            het_stat = float(het_res[0, 0])
            het_p = float(het_res[0, 1])
    except Exception:
        try:
            arch_stat, arch_p, _, _ = het_arch(resid, nlags=min(4, max(1, len(resid) // 10)))
            het_stat, het_p = float(arch_stat), float(arch_p)
        except Exception:
            pass

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05])
    ax_ts = fig.add_subplot(gs[0, :])
    ax_acf = fig.add_subplot(gs[1, 0])
    ax_hist = fig.add_subplot(gs[1, 1])

    ax_ts.plot(resid.index, resid.values, color="tab:blue", linewidth=1.4, label="Residuals")
    ax_ts.axhline(0, color="red", linestyle="--", alpha=0.7, label="Mean = 0")
    ax_ts.set_title(f"(1) Residuals theo thời gian — {title}", fontsize=11.5, fontweight="bold")
    ax_ts.set_xlabel("Thời gian")
    ax_ts.set_ylabel("Residual")
    ax_ts.grid(True, linestyle="--", alpha=0.45)
    ax_ts.legend(loc="upper left")

    plot_acf(resid, lags=effective_lags, zero=False, ax=ax_acf)
    ax_acf.set_title(f"(2) ACF của Residuals (lag={effective_lags}) — {title}", fontsize=11.5, fontweight="bold")
    ax_acf.set_xlabel("Độ trễ (Lag)")
    ax_acf.set_ylabel("Tự tương quan")
    ax_acf.grid(True, linestyle="--", alpha=0.45)

    ax_hist.hist(
        resid.values,
        bins=min(18, max(8, len(resid) // 6)),
        density=True,
        color="tab:green",
        alpha=0.45,
        edgecolor="white",
        label="Histogram",
    )
    x_grid = np.linspace(float(resid.min()), float(resid.max()), 150)
    if resid.std() > 0:
        norm_pdf = scipy_stats.norm.pdf(x_grid, loc=float(resid.mean()), scale=float(resid.std()))
        ax_hist.plot(x_grid, norm_pdf, color="tab:red", linestyle="--", linewidth=2.0, label="Phân phối chuẩn N(μ, σ²)")
        try:
            kde = scipy_stats.gaussian_kde(resid.values)
            ax_hist.plot(x_grid, kde(x_grid), color="darkgreen", linewidth=1.8, label="KDE thực nghiệm")
        except Exception:
            pass
    ax_hist.axvline(0, color="black", linestyle=":", alpha=0.7)
    ax_hist.set_title(f"(3) Phân phối Phần dư (Histogram & KDE) — {title}", fontsize=11.5, fontweight="bold")
    ax_hist.set_xlabel("Giá trị Residual")
    ax_hist.set_ylabel("Mật độ")
    ax_hist.grid(True, linestyle="--", alpha=0.45)
    ax_hist.legend(loc="upper right", fontsize=8.5)

    fig.suptitle(
        f"Chẩn đoán Phần dư (Residual Diagnostics): {title} | "
        f"Mean={resid.mean():.3f}, Std={resid.std():.3f}, LB(lag={effective_lags}) p={lb_p:.4f}, JB p={jb_p:.4f}",
        fontsize=12.5,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    is_wn = bool(lb_p >= 0.05)
    summary = {
        "mean": float(resid.mean()),
        "std": float(resid.std()),
        "ljung_box_q": lb_q,
        "ljung_box_p": lb_p,
        "ljung_box_lag": effective_lags,
        "n_estimated_params": n_arma_params,
        "jarque_bera_stat": jb_stat,
        "jarque_bera_p": jb_p,
        "skew": skew_val,
        "kurtosis": kurt_val,
        "heteroskedasticity_stat": het_stat,
        "heteroskedasticity_p": het_p,
        "is_white_noise": is_wn,
    }
    print(f"Mean residual : {summary['mean']:.3f}  (Std = {summary['std']:.3f})")
    print(
        f"Ljung-Box (lag={effective_lags}, k_params={n_arma_params}) Q = {lb_q:.2f}, p-value : {lb_p:.4f} | "
        f"Jarque-Bera p-value : {jb_p:.4f} | Het p-value : {het_p:.4f}"
    )
    if lb_p < 0.05:
        print("=> Con tu tuong quan (khong phai white noise).")
    else:
        print("=> Chua thay tu tuong quan o lag kiem tra (Dat chuan White Noise).")
    return summary


def forecast_test(result, steps: int, inverse_log: bool = False) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Purpose:
        Generate out-of-sample forecasts and 95% confidence intervals for `steps`
        periods ahead, optionally applying exponential inverse transformation exp(.)
        when the model was trained on natural log scale.

    Parameters:
        result: Fitted ARIMAResults or SARIMAXResults object.
        steps (int): Forecast horizon (= len(test)).
        inverse_log (bool): If True, transform predictions and CI bounds via np.exp().

    Returns:
        Tuple[pd.Series, pd.DataFrame]: (predicted_mean, conf_int_95).
    """
    if steps <= 0:
        raise ValueError(f"Forecast steps must be positive, got {steps}.")
    fc = result.get_forecast(steps=int(steps))
    pred_mean = fc.predicted_mean
    conf_int = fc.conf_int(alpha=0.05)
    if inverse_log:
        pred_mean = np.exp(pred_mean)
        conf_int = np.exp(conf_int)
    return pred_mean, conf_int


def evaluate_forecast(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """
    Purpose:
        Calculate mandatory out-of-sample forecasting accuracy metrics between
        actual Test values and predicted values on the original scale:
        MAE, RMSE, MAPE (%), and R².

    Parameters:
        y_true (pd.Series | array-like): Ground-truth Test observations on original scale.
        y_pred (pd.Series | array-like): Forecast values on original scale.

    Returns:
        dict: Dictionary containing {'MAE', 'RMSE', 'MAPE (%)', 'R2'}.
    """
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    if actual.size != predicted.size or actual.size == 0:
        raise ValueError(f"Do dai khong hop le: thuc te = {actual.size}, du bao = {predicted.size}.")

    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    nonzero = actual != 0
    mape = (
        float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100)
        if nonzero.any()
        else float("nan")
    )
    r2 = float(r2_score(actual, predicted))
    return {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape, "R2": r2}


def calculate_metrics(y_true, y_pred) -> dict:
    """
    Purpose:
        Compute MAE, RMSE, MAPE (both 'MAPE (%)' and 'MAPE' keys), and R².
        Acts as a backward-compatible alias for evaluate_forecast.

    Parameters:
        y_true: Ground-truth Test values.
        y_pred: Predicted Test values.

    Returns:
        dict: {'MAE', 'RMSE', 'MAPE (%)', 'MAPE', 'R2'}.
    """
    metrics = evaluate_forecast(y_true, y_pred)
    metrics["MAPE"] = metrics["MAPE (%)"]
    return metrics


def _infer_ylabel(title: str, ylabel: Optional[str] = None) -> str:
    """
    Purpose:
        Determine the appropriate Y-axis label based on the dataset context in title
        when ylabel is not explicitly passed.

    Parameters:
        title (str): Plot title.
        ylabel (Optional[str]): Explicit ylabel if provided.

    Returns:
        str: Appropriate Y-axis label in Vietnamese.
    """
    if ylabel:
        return ylabel
    lower = (title or "").lower()
    if any(k in lower for k in ["retail", "bán lẻ", "doanh số", "dataset1", "dataset 1"]):
        return "Doanh số bán lẻ (value)"
    return "Số hành khách (nghìn người)"


def plot_forecast(
    train,
    test,
    forecast,
    conf_int=None,
    title: str = "",
    save_path: Optional[str] = None,
    ylabel: Optional[str] = None,
    show_plot: bool = True,
) -> None:
    """
    Purpose:
        Plot Train, actual Test, and out-of-sample Forecast series with 95% confidence
        intervals and a clear vertical boundary separating Train and Test periods.

    Parameters:
        train (pd.Series): Training time series.
        test (pd.Series): Testing time series.
        forecast (pd.Series): Out-of-sample forecasted series aligned with test.
        conf_int (Optional[pd.DataFrame]): 95% confidence interval DataFrame.
        title (str): Chart title.
        save_path (Optional[str]): Path to save the PNG image.
        ylabel (Optional[str]): Y-axis label (auto-detected from title if None).
        show_plot (bool): Whether to call plt.show().

    Returns:
        None.
    """
    fc_series = pd.Series(forecast.values if hasattr(forecast, "values") else forecast, index=test.index)
    fig, ax = plt.subplots(figsize=(15, 7))

    ax.plot(
        train.index,
        train.values,
        color="tab:blue",
        linewidth=1.5,
        alpha=0.7,
        marker=".",
        markersize=4,
        label="Train (80%)",
    )
    ax.plot(
        test.index,
        test.values,
        color="tab:orange",
        linewidth=2.2,
        marker="o",
        markersize=5,
        label="Test thực tế",
    )
    ax.plot(
        fc_series.index,
        fc_series.values,
        color="forestgreen",
        linestyle="--",
        linewidth=2.2,
        marker="D",
        markersize=4,
        label="Dự báo (Forecast)",
    )

    if conf_int is not None:
        ci_low = np.asarray(conf_int.iloc[:, 0], dtype=float)
        ci_high = np.asarray(conf_int.iloc[:, 1], dtype=float)
        ax.fill_between(
            test.index,
            ci_low,
            ci_high,
            color="forestgreen",
            alpha=0.15,
            label="Khoảng tin cậy 95%",
        )

    split_date = test.index[0]
    ax.axvline(x=split_date, color="black", linestyle="--", linewidth=1.5, alpha=0.8)

    y_top = ax.get_ylim()[1]
    ax.text(
        split_date - pd.DateOffset(months=3),
        y_top * 0.95,
        "Huấn luyện (Train)",
        ha="right",
        color="tab:blue",
        fontweight="bold",
    )
    ax.text(
        split_date + pd.DateOffset(months=2),
        y_top * 0.95,
        "Kiểm thử (Test)",
        ha="left",
        color="tab:orange",
        fontweight="bold",
    )

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")

    ax.grid(True, which="major", linestyle="-", alpha=0.6)
    ax.grid(True, which="minor", linestyle=":", alpha=0.4)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Thời gian (Tháng/Năm)", fontsize=11, fontweight="bold")
    ax.set_ylabel(_infer_ylabel(title, ylabel), fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", frameon=True, shadow=True, fancybox=True, facecolor="white", framealpha=0.95)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_arima_vs_sarima(
    train,
    test,
    fc_arima1,
    fc_arima2,
    fc_sarima,
    conf_int_sarima,
    names: dict,
    r2_scores: dict,
    title: str = "",
    save_path: Optional[str] = None,
    ylabel: Optional[str] = None,
    show_plot: bool = True,
) -> None:
    """
    Purpose:
        Plot a head-to-head comparison of two non-seasonal ARIMA forecasts against
        the best seasonal SARIMA forecast on the Test set.

    Parameters:
        train (pd.Series): Training time series.
        test (pd.Series): Testing time series.
        fc_arima1 (pd.Series): Forecast from ARIMA candidate 1.
        fc_arima2 (pd.Series): Forecast from ARIMA candidate 2.
        fc_sarima (pd.Series): Forecast from the best SARIMA model.
        conf_int_sarima (Optional[pd.DataFrame]): 95% confidence interval for SARIMA.
        names (dict): Mapping {'arima1', 'arima2', 'sarima'} to display names.
        r2_scores (dict): Mapping {'arima1', 'arima2', 'sarima'} to R² scores.
        title (str): Chart title.
        save_path (Optional[str]): Path to save the PNG image.
        ylabel (Optional[str]): Y-axis label (auto-detected from title if None).
        show_plot (bool): Whether to call plt.show().

    Returns:
        None.
    """
    fig, ax = plt.subplots(figsize=(15, 7.5))

    train_range = f"{train.index.min().strftime('%m/%Y')} - {train.index.max().strftime('%m/%Y')}"
    test_range = f"{test.index.min().strftime('%m/%Y')} - {test.index.max().strftime('%m/%Y')}"

    ax.plot(
        train.index,
        train.values,
        color="tab:blue",
        linewidth=1.5,
        alpha=0.5,
        label=f"Tập Huấn luyện (Train: {train_range})",
    )
    ax.plot(
        test.index,
        test.values,
        color="black",
        linewidth=2.8,
        marker="o",
        markersize=4,
        label=f"Thực tế (Test: {test_range})",
    )
    ax.plot(
        test.index,
        fc_arima1.values,
        color="tab:red",
        linestyle=":",
        linewidth=2.2,
        label=f"{names['arima1']} [Tuyến tính, R² = {r2_scores['arima1']:.2f}]",
    )
    ax.plot(
        test.index,
        fc_arima2.values,
        color="tab:purple",
        linestyle="-.",
        linewidth=2.2,
        label=f"{names['arima2']} [Tuyến tính, R² = {r2_scores['arima2']:.2f}]",
    )
    ax.plot(
        test.index,
        fc_sarima.values,
        color="forestgreen",
        linestyle="-",
        linewidth=2.8,
        label=f"{names['sarima']} [Khớp chu kỳ mùa vụ, R² = {r2_scores['sarima']:.2f}]",
    )

    if conf_int_sarima is not None:
        ax.fill_between(
            test.index,
            conf_int_sarima.iloc[:, 0],
            conf_int_sarima.iloc[:, 1],
            color="forestgreen",
            alpha=0.15,
            label="SARIMA 95% Confidence Interval",
        )

    split_date = test.index[0]
    ax.axvline(x=split_date, color="dimgrey", linestyle="--", linewidth=1.5, alpha=0.8)

    y_top = ax.get_ylim()[1]
    ax.text(
        split_date - pd.DateOffset(months=3),
        y_top * 0.95,
        "Huấn luyện",
        ha="right",
        color="dimgrey",
        fontstyle="italic",
        fontsize=11,
    )
    ax.text(
        split_date + pd.DateOffset(months=2),
        y_top * 0.95,
        "Dự báo (Ngoài mẫu)",
        ha="left",
        color="black",
        fontweight="bold",
        fontsize=11,
    )

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.grid(True, which="major", linestyle="-", alpha=0.6)
    ax.grid(True, which="minor", linestyle=":", alpha=0.3)

    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Năm", fontsize=12)
    ax.set_ylabel(_infer_ylabel(title, ylabel), fontsize=12)
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.95, shadow=True, fontsize=10)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_log_transformation(
    train,
    test=None,
    save_path: Optional[str | Path] = None,
    show_plot: bool = True,
):
    """
    Purpose:
        Apply natural logarithm transformation ln(Y_t) to stabilize multiplicative
        variance and plot the log-transformed training series alongside its doubly
        differenced (d=1, D=1, s=12) series.

    Parameters:
        train (pd.Series | pd.DataFrame): Chronological training series ( strictly > 0).
        test (Optional[pd.Series | pd.DataFrame]): Optional testing series.
        save_path (Optional[str | Path]): File path to save the figure as PNG.
        show_plot (bool): Whether to call plt.show().

    Returns:
        Tuple[train_log, test_log, diff_log]: Log-transformed train, test, and differenced series.
    """
    train_log = np.log(train)
    test_log = np.log(test) if test is not None else None
    diff_log = train_log.diff(1).diff(12).dropna()

    fig, axes = plt.subplots(2, 1, figsize=(18, 10))
    axes[0].plot(train_log.index, train_log.values, color="#2ca02c", linewidth=1.5, marker=".", markersize=4)
    axes[0].set_title(
        "1. Chuỗi sau khi lấy ln(Train): Biên độ sóng đều đặn (Homoscedasticity)",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].set_ylabel("ln(Value)", fontsize=10, fontweight="bold")

    axes[1].plot(diff_log.index, diff_log.values, color="#9467bd", linewidth=1.5, marker=".", markersize=4)
    axes[1].axhline(0, color="black", linestyle="--", alpha=0.7)
    axes[1].set_title(
        "2. Sai phân chuỗi Log (d=1, D=1): Dao động dừng chuẩn & phương sai đồng nhất",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].set_ylabel("Δ Δ₁₂ ln(Yₜ)", fontsize=10, fontweight="bold")

    for ax in axes:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
        ax.tick_params(axis="x", rotation=90, labelsize=8)
        ax.grid(True, which="major", linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    return train_log, test_log, diff_log