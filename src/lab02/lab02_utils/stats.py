from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.stattools import acf, adfuller, pacf
except ImportError:
    from src.lab02.lab02_utils.scipy_tsa_compat import acf, adfuller, pacf, plot_acf, plot_pacf

from src.lab02.lab02_utils.logging_config import log_statistical_analysis


def perform_adf_test(
    series: pd.Series,
    title: str = "Original Series",
    dataset_name: Optional[str] = None,
    return_dict: bool = False,
) -> float | Dict[str, Any]:
    """
    Purpose:
        Execute the Augmented Dickey-Fuller (ADF) unit root test to check stationarity,
        display formatted test statistics and critical values, log results to the
        centralized LAB02 logger, and return the p-value (or full dictionary if requested).

    Parameters:
        series (pd.Series): 1D time series to test for unit root.
        title (str): Descriptive label of the series for display and logging.
        dataset_name (Optional[str]): Dataset identifier ('dataset1' or 'dataset2') for logging.
        return_dict (bool): If True, return a detailed dictionary instead of only float p_value.

    Returns:
        float | Dict[str, Any]: The p-value of the ADF test (default for backward compatibility)
            or a dictionary of full ADF statistics when return_dict=True.
    """
    clean_series = pd.Series(series).astype(float).dropna()
    if clean_series.empty:
        raise ValueError(f"Cannot run ADF test on empty series: {title}")

    print(f"========== KIỂM ĐỊNH ADF CHO: {title.upper()} ==========")

    result = adfuller(clean_series, autolag="AIC", result_object=False)

    adf_statistic = float(result[0])
    p_value = float(result[1])
    lags_used = int(result[2])
    obs_used = int(result[3])
    critical_values = {k: float(v) for k, v in result[4].items()}
    is_stationary = bool(p_value < 0.05)

    print(f"Số quan sát đầu vào : {len(clean_series)}")
    print(f"Số độ trễ (Lags)    : {lags_used}")
    print(f"Số quan sát sử dụng : {obs_used}")
    print(f"ADF Statistic       : {adf_statistic:.4f}")
    print(f"p-value             : {p_value:.4f}")

    print("\n[Giá trị tới hạn - Critical Values]:")
    for key, value in critical_values.items():
        print(f"   > {key}: {value:.4f}")

    print("\n[KẾT LUẬN]:")
    print("H0: Chuỗi có nghiệm đơn vị (Chuỗi KHÔNG DỪNG)")
    print("H1: Chuỗi không có nghiệm đơn vị (Chuỗi DỪNG)")

    if is_stationary:
        conclusion = "STATIONARY (Bác bỏ H0: Chuỗi ĐÃ DỪNG ở mức ý nghĩa 5%)"
        print(f"=> p-value = {p_value:.4f} < 0.05: Đủ cơ sở bác bỏ H0.")
        print("=> CHUỖI ĐÃ DỪNG (Stationary).")
    else:
        conclusion = "NON-STATIONARY (Chưa bác bỏ H0: Chuỗi KHÔNG DỪNG ở mức ý nghĩa 5%)"
        print(f"=> p-value = {p_value:.4f} >= 0.05: Chưa đủ cơ sở bác bỏ H0.")
        print("=> CHUỖI KHÔNG DỪNG (Non-stationary). Cần lấy sai phân!")

    print("=" * 60 + "\n")

    inferred_ds = dataset_name
    if inferred_ds is None:
        lower_t = title.lower()
        if any(w in lower_t for w in ["retail", "bán lẻ", "dataset1", "dataset 1"]):
            inferred_ds = "dataset1"
        elif any(w in lower_t for w in ["hành khách", "passenger", "dataset2", "dataset 2"]):
            inferred_ds = "dataset2"

    if inferred_ds is not None:
        log_statistical_analysis(
            dataset_name=inferred_ds,
            analysis_type="ADF_TEST",
            series_name=title,
            details={
                "n_input": len(clean_series),
                "lags_used": lags_used,
                "obs_used": obs_used,
                "adf_stat": round(adf_statistic, 4),
                "p_value": round(p_value, 6),
                "crit_5pct": round(critical_values.get("5%", 0.0), 4),
                "conclusion": conclusion,
            },
        )

    if return_dict:
        return {
            "title": title,
            "n_input": len(clean_series),
            "lags_used": lags_used,
            "obs_used": obs_used,
            "adf_statistic": adf_statistic,
            "p_value": p_value,
            "critical_values": critical_values,
            "is_stationary": is_stationary,
            "conclusion": conclusion,
        }
    return p_value


def significant_lags(series: pd.Series, nlags: int = 36, pacf_method: str = "ywm") -> dict:
    """
    Purpose:
        Compute ACF and PACF values up to nlags for a differenced series and identify
        lags exceeding the 95% confidence threshold (±1.96/sqrt(n)).

    Parameters:
        series (pd.Series): Differenced/stationary time series (NaN values are dropped).
        nlags (int): Maximum lag order to evaluate (default 36).
        pacf_method (str): PACF calculation method in statsmodels (default 'ywm').

    Returns:
        dict: Dictionary with keys {'n', 'bound', 'acf_values', 'pacf_values', 'acf_sig', 'pacf_sig'}.
    """
    x = pd.Series(series).astype(float).dropna()
    if x.empty:
        raise ValueError("Input series for significant_lags is empty after dropping NaNs.")
    effective_lags = min(int(nlags), max(1, (len(x) // 2) - 1))
    bound = float(1.96 / np.sqrt(len(x)))
    acf_values = acf(x, nlags=effective_lags)
    pacf_values = pacf(x, nlags=effective_lags, method=pacf_method)
    return {
        "n": len(x),
        "bound": bound,
        "acf_values": acf_values,
        "pacf_values": pacf_values,
        "acf_sig": [k for k in range(1, effective_lags + 1) if abs(acf_values[k]) > bound],
        "pacf_sig": [k for k in range(1, effective_lags + 1) if abs(pacf_values[k]) > bound],
    }


def plot_acf_pacf(
    series: pd.Series,
    lags: int = 36,
    title: str = "",
    seasonal_period: int = 12,
    save_path: Optional[str] = None,
) -> None:
    """
    Purpose:
        Plot side-by-side ACF and PACF charts for a differenced series with vertical
        markers at multiples of the seasonal period s. Preserved for backward compatibility.

    Parameters:
        series (pd.Series): Differenced time series (never use raw non-stationary series).
        lags (int): Number of lags to display (default 36).
        title (str): Overall figure suptitle.
        seasonal_period (int): Seasonal cycle length s (default 12 for monthly data).
        save_path (Optional[str]): Optional file path to save the figure as PNG.

    Returns:
        None.
    """
    x = pd.Series(series).astype(float).dropna()
    effective_lags = min(int(lags), max(1, (len(x) // 2) - 1))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    plot_acf(x, lags=effective_lags, zero=False, ax=axes[0])
    plot_pacf(x, lags=effective_lags, zero=False, method="ywm", ax=axes[1])
    axes[0].set_title("ACF (gợi ý bậc q)")
    axes[1].set_title("PACF (gợi ý bậc p)")
    for ax in axes:
        if seasonal_period and seasonal_period > 1:
            for k in range(seasonal_period, effective_lags + 1, seasonal_period):
                ax.axvline(
                    k,
                    color="tab:red",
                    linestyle=":",
                    alpha=0.6,
                    label=f"Lag mùa vụ (bội của {seasonal_period})" if k == seasonal_period else None,
                )
            ax.set_xticks(sorted(set([1, 2, 3, 4] + list(range(seasonal_period, effective_lags + 1, seasonal_period)))))
            ax.legend(loc="upper right")
        ax.set_xlabel("Lag (tháng)")
        ax.set_ylabel("Hệ số tương quan")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()