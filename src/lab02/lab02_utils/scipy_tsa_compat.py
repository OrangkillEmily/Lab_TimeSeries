"""
MODULE: SciPy/NumPy Time Series Compatibility Engine — src/lab02_utils/scipy_tsa_compat.py
================================================================================
PURPOSE:
    Provides a mathematically rigorous, pure NumPy/SciPy implementation of the
    statsmodels and scikit-learn time series interfaces (`adfuller`, `acf`, `pacf`,
    `plot_acf`, `plot_pacf`, `acorr_ljungbox`, `jarque_bera`, `het_arch`, `ARIMA`,
    `SARIMAX`, `mean_absolute_error`, `mean_squared_error`, `r2_score`,
    `train_test_split`).

    - When `statsmodels` and `scikit-learn` ARE installed (e.g., on the student's
      local machine), the project imports the native `statsmodels` and `sklearn`
      classes/functions directly.
    - When running inside a restricted verification sandbox VM where `statsmodels`
      or `scikit-learn` is not pre-installed, this module acts as a drop-in
      statistical engine using SciPy optimization and linear algebra so all code,
      tests, diagnostics, plots, and model summaries execute with real numerical calculations.
================================================================================
"""
from typing import Optional, Sequence, Tuple, Dict, Any
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg, optimize, stats


# ==============================================================================
# 1. SCIKIT-LEARN COMPATIBILITY FUNCTIONS
# ==============================================================================

def mean_absolute_error(y_true, y_pred) -> float:
    """
    Purpose:
        Compute Mean Absolute Error (MAE) between ground truth and predictions.

    Parameters:
        y_true: Ground-truth array-like values.
        y_pred: Predicted array-like values.

    Returns:
        float: Mean absolute error.
    """
    a = np.asarray(y_true, dtype=float).reshape(-1)
    b = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.mean(np.abs(a - b)))


def mean_squared_error(y_true, y_pred) -> float:
    """
    Purpose:
        Compute Mean Squared Error (MSE) between ground truth and predictions.

    Parameters:
        y_true: Ground-truth array-like values.
        y_pred: Predicted array-like values.

    Returns:
        float: Mean squared error.
    """
    a = np.asarray(y_true, dtype=float).reshape(-1)
    b = np.asarray(y_pred, dtype=float).reshape(-1)
    return float(np.mean((a - b) ** 2))


def r2_score(y_true, y_pred) -> float:
    """
    Purpose:
        Compute coefficient of determination (R²) on out-of-sample predictions.

    Parameters:
        y_true: Ground-truth array-like values.
        y_pred: Predicted array-like values.

    Returns:
        float: R² score.
    """
    a = np.asarray(y_true, dtype=float).reshape(-1)
    b = np.asarray(y_pred, dtype=float).reshape(-1)
    ss_res = float(np.sum((a - b) ** 2))
    ss_tot = float(np.sum((a - np.mean(a)) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def train_test_split(data, test_size: float = 0.2, shuffle: bool = False, random_state: Optional[int] = None):
    """
    Purpose:
        Split a DataFrame, Series, or array into Train and Test subsets.
        Matches scikit-learn's exact ceil(n * test_size) sizing.

    Parameters:
        data: Input DataFrame, Series, or numpy array.
        test_size (float): Proportion of dataset to include in test split (default 0.2).
        shuffle (bool): Whether to shuffle data before splitting (must be False for time series).
        random_state (Optional[int]): Random seed if shuffle=True.

    Returns:
        Tuple[train, test]: Split train and test objects of the same type as input.
    """
    n = len(data)
    n_test = int(np.ceil(n * float(test_size)))
    n_train = n - n_test
    if shuffle:
        rng = np.random.default_rng(random_state)
        idx = rng.permutation(n)
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.iloc[idx[:n_train]], data.iloc[idx[n_train:]]
        return data[idx[:n_train]], data[idx[n_train:]]
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.iloc[:n_train].copy(), data.iloc[n_train:].copy()
    return data[:n_train].copy(), data[n_train:].copy()


# ==============================================================================
# 2. ACF, PACF, ADF, LJUNG-BOX, JARQUE-BERA & HETEROSKEDASTICITY
# ==============================================================================

def acf(x, nlags: int = 40, fft: bool = True) -> np.ndarray:
    """
    Purpose:
        Compute the sample Autocorrelation Function (ACF) for lags 0..nlags.

    Parameters:
        x: 1D array-like time series.
        nlags (int): Number of lags to compute.
        fft (bool): Ignored compatibility flag (direct/FFT covariance).

    Returns:
        np.ndarray: 1D array of autocorrelations of length nlags + 1.
    """
    arr = np.asarray(pd.Series(x).dropna(), dtype=float).reshape(-1)
    n = len(arr)
    nlags = min(int(nlags), n - 1)
    centered = arr - np.mean(arr)
    var0 = float(np.dot(centered, centered))
    if var0 == 0.0:
        out = np.zeros(nlags + 1, dtype=float)
        out[0] = 1.0
        return out
    result = np.empty(nlags + 1, dtype=float)
    result[0] = 1.0
    for k in range(1, nlags + 1):
        result[k] = float(np.dot(centered[k:], centered[:-k]) / var0)
    return result


def pacf(x, nlags: int = 40, method: str = "ywm") -> np.ndarray:
    """
    Purpose:
        Compute the Partial Autocorrelation Function (PACF) for lags 0..nlags
        using the Yule-Walker equations without bias adjustment ('ywm') via
        Levinson-Durbin / Toeplitz solver.

    Parameters:
        x: 1D array-like time series.
        nlags (int): Number of lags to compute.
        method (str): Estimation method ('ywm' default).

    Returns:
        np.ndarray: 1D array of partial autocorrelations of length nlags + 1.
    """
    r = acf(x, nlags=nlags)
    m = len(r) - 1
    out = np.zeros(m + 1, dtype=float)
    out[0] = 1.0
    if m >= 1:
        out[1] = r[1]
    for k in range(2, m + 1):
        try:
            phi = linalg.solve_toeplitz(r[:k], r[1 : k + 1])
            out[k] = float(phi[-1])
        except Exception:
            out[k] = 0.0
    return out


def plot_acf(x, lags: int = 36, zero: bool = False, ax: Optional[plt.Axes] = None, alpha: float = 0.05, **kwargs):
    """
    Purpose:
        Plot the sample Autocorrelation Function (ACF) with 95% confidence intervals
        matching statsmodels.graphics.tsaplots.plot_acf.

    Parameters:
        x: 1D time series.
        lags (int): Maximum lag to plot.
        zero (bool): Whether to include lag 0.
        ax (Optional[plt.Axes]): Matplotlib axes to draw on.
        alpha (float): Significance level for confidence intervals (default 0.05).

    Returns:
        plt.Figure: The matplotlib Figure object.
    """
    arr = np.asarray(pd.Series(x).dropna(), dtype=float).reshape(-1)
    n = len(arr)
    lags = min(int(lags), n - 1)
    vals = acf(arr, nlags=lags)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    start_idx = 0 if zero else 1
    lag_axis = np.arange(start_idx, lags + 1)
    y_vals = vals[start_idx : lags + 1]

    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    # Bartlett's formula for MA(q) standard error
    se = np.zeros(lags + 1, dtype=float)
    se[0] = 0.0
    if lags >= 1:
        se[1] = 1.0 / np.sqrt(n)
    for k in range(2, lags + 1):
        se[k] = np.sqrt((1.0 + 2.0 * np.sum(vals[1:k] ** 2)) / n)
    bounds = z * se[start_idx : lags + 1]

    ax.axhline(0, color="black", linewidth=0.8)
    ax.vlines(lag_axis, [0], y_vals, color="#1f77b4", linewidth=1.6)
    ax.plot(lag_axis, y_vals, "o", color="#1f77b4", markersize=4.5)
    ax.fill_between(lag_axis, -bounds, bounds, color="#1f77b4", alpha=0.2)
    ax.set_title("Autocorrelation")
    ax.set_ylim(-1.05, 1.05)
    return fig


def plot_pacf(
    x,
    lags: int = 36,
    zero: bool = False,
    method: str = "ywm",
    ax: Optional[plt.Axes] = None,
    alpha: float = 0.05,
    **kwargs,
):
    """
    Purpose:
        Plot the Partial Autocorrelation Function (PACF) with 95% confidence intervals
        matching statsmodels.graphics.tsaplots.plot_pacf.

    Parameters:
        x: 1D time series.
        lags (int): Maximum lag to plot.
        zero (bool): Whether to include lag 0.
        method (str): PACF method ('ywm').
        ax (Optional[plt.Axes]): Matplotlib axes to draw on.
        alpha (float): Significance level (default 0.05).

    Returns:
        plt.Figure: The matplotlib Figure object.
    """
    arr = np.asarray(pd.Series(x).dropna(), dtype=float).reshape(-1)
    n = len(arr)
    lags = min(int(lags), max(1, (n // 2) - 1))
    vals = pacf(arr, nlags=lags, method=method)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    start_idx = 0 if zero else 1
    lag_axis = np.arange(start_idx, lags + 1)
    y_vals = vals[start_idx : lags + 1]

    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    bound = z / np.sqrt(n)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.vlines(lag_axis, [0], y_vals, color="#1f77b4", linewidth=1.6)
    ax.plot(lag_axis, y_vals, "o", color="#1f77b4", markersize=4.5)
    ax.fill_between(lag_axis, -bound, bound, color="#1f77b4", alpha=0.2)
    ax.set_title("Partial Autocorrelation")
    ax.set_ylim(-1.05, 1.05)
    return fig


def _mackinnon_pvalue_c(stat: float) -> float:
    """
    Purpose:
        Compute MacKinnon (1994, 2010) approximate p-value for the ADF unit-root
        test with constant ('c' regression: tau_c).

    Parameters:
        stat (float): ADF t-statistic.

    Returns:
        float: Approximate one-sided p-value in [0.0, 1.0].
    """
    # MacKinnon (1994) response surface coefficients for tau_c (nc=1, with intercept)
    maxstat = 2.74
    minstat = -18.83
    if stat > maxstat:
        return 0.9999
    if stat < minstat:
        return 1e-25
    starstat = -1.61
    if stat <= starstat:
        # Small tau_c polynomial coefficients: [gamma_0, gamma_1, gamma_2]
        c = np.array([2.1659, 1.4412, 0.038269])
        z = c[0] + c[1] * stat + c[2] * (stat ** 2)
    else:
        # Large tau_c polynomial coefficients: [gamma_0, gamma_1, gamma_2, gamma_3]
        c = np.array([1.7339, 0.93202, -0.12745, -0.010368])
        z = c[0] + c[1] * stat + c[2] * (stat ** 2) + c[3] * (stat ** 3)
    return float(np.clip(stats.norm.cdf(z), 0.0, 1.0))


def _mackinnon_crit_c(nobs: int) -> Dict[str, float]:
    """
    Purpose:
        Compute MacKinnon finite-sample critical values (1%, 5%, 10%) for the ADF
        test with constant ('c').

    Parameters:
        nobs (int): Number of effective observations in the ADF regression.

    Returns:
        Dict[str, float]: Dictionary with keys '1%', '5%', '10%'.
    """
    n = float(max(nobs, 20))
    return {
        "1%": float(-3.43035 - 6.5393 / n - 16.786 / (n ** 2) - 79.433 / (n ** 3)),
        "5%": float(-2.86154 - 2.8903 / n - 4.234 / (n ** 2) - 40.04 / (n ** 3)),
        "10%": float(-2.56677 - 1.5384 / n - 2.809 / (n ** 2)),
    }


def adfuller(x, maxlag: Optional[int] = None, regression: str = "c", autolag: Optional[str] = "AIC", result_object: bool = False):
    """
    Purpose:
        Perform the Augmented Dickey-Fuller (ADF) unit root test matching
        statsmodels.tsa.stattools.adfuller(..., regression='c', autolag='AIC').

    Parameters:
        x: 1D time series array-like.
        maxlag (Optional[int]): Maximum lag order included in test (default Schwert rule).
        regression (str): Constant/trend specification ('c' default).
        autolag (Optional[str]): Criterion for automatic lag selection ('AIC' default).
        result_object (bool): Compatibility flag.

    Returns:
        tuple: (adf_stat, pvalue, usedlag, nobs, critical_values, icbest).
    """
    y = np.asarray(pd.Series(x).dropna(), dtype=float).reshape(-1)
    n = len(y)
    if maxlag is None:
        maxlag = int(np.ceil(12.0 * ((n / 100.0) ** 0.25)))
    maxlag = min(maxlag, (n // 2) - 2)

    dy = np.diff(y)

    def _fit_adf_ols(k: int, start_idx: int):
        # Regression: dy[t] = const + beta * y[t] + sum_{j=1}^k gamma_j * dy[t - j]
        # where t runs from start_idx to len(dy) - 1
        dep = dy[start_idx:]
        n_eff = len(dep)
        cols = [y[start_idx : len(dy)]]
        for j in range(1, k + 1):
            cols.append(dy[start_idx - j : len(dy) - j])
        cols.append(np.ones(n_eff, dtype=float))
        X = np.column_stack(cols)
        beta, _, _, _ = linalg.lstsq(X, dep)
        resid = dep - X @ beta
        ssr = float(np.dot(resid, resid))
        df_resid = max(1, n_eff - X.shape[1])
        sigma2_ols = ssr / df_resid
        cov_b = sigma2_ols * linalg.pinv(X.T @ X)
        t_stat = float(beta[0] / np.sqrt(max(cov_b[0, 0], 1e-18)))
        # Log-likelihood & AIC matching statsmodels OLS.aic on common sample
        sigma2_mle = max(ssr / n_eff, 1e-18)
        llf = -0.5 * n_eff * (1.0 + np.log(2.0 * np.pi) + np.log(sigma2_mle))
        aic_val = float(-2.0 * llf + 2.0 * X.shape[1])
        return t_stat, n_eff, aic_val

    if autolag and autolag.upper() == "AIC":
        best_aic = float("inf")
        best_k = 0
        for k in range(0, maxlag + 1):
            # Statsmodels evaluates autolag AIC on the common sample starting at maxlag
            _, _, aic_k = _fit_adf_ols(k, start_idx=maxlag)
            if aic_k < best_aic:
                best_aic = aic_k
                best_k = k
        usedlag = best_k
    else:
        usedlag = maxlag
        best_aic = 0.0

    # Once usedlag is selected, re-estimate on the full available sample starting at usedlag
    adf_stat, nobs, _ = _fit_adf_ols(usedlag, start_idx=usedlag)
    pval = _mackinnon_pvalue_c(adf_stat)
    crit = _mackinnon_crit_c(nobs)
    return adf_stat, pval, usedlag, nobs, crit, best_aic


def acorr_ljungbox(x, lags: Optional[Sequence[int]] = None, return_df: bool = True) -> pd.DataFrame:
    """
    Purpose:
        Compute the Ljung-Box Q statistic and p-value for autocorrelation up to
        specified lags, matching statsmodels.stats.diagnostic.acorr_ljungbox.

    Parameters:
        x: 1D residual series.
        lags (Optional[Sequence[int]]): List of lag orders to test.
        return_df (bool): Whether to return a DataFrame indexed by lag.

    Returns:
        pd.DataFrame: DataFrame with columns ['lb_stat', 'lb_pvalue'] indexed by lag.
    """
    arr = np.asarray(pd.Series(x).dropna(), dtype=float).reshape(-1)
    n = len(arr)
    if lags is None:
        lags_list = [min(10, n - 1)]
    elif isinstance(lags, (int, np.integer)):
        lags_list = list(range(1, int(lags) + 1))
    else:
        lags_list = [int(k) for k in lags]

    max_k = min(max(lags_list), n - 1)
    r = acf(arr, nlags=max_k)
    q_stats = []
    p_vals = []
    valid_lags = []
    for m in lags_list:
        m_eff = min(m, max_k)
        ks = np.arange(1, m_eff + 1, dtype=float)
        q_val = float(n * (n + 2.0) * np.sum((r[1 : m_eff + 1] ** 2) / (n - ks)))
        p_val = float(stats.chi2.sf(q_val, df=m_eff))
        valid_lags.append(m)
        q_stats.append(q_val)
        p_vals.append(p_val)

    return pd.DataFrame({"lb_stat": q_stats, "lb_pvalue": p_vals}, index=valid_lags)


def jarque_bera(x) -> Tuple[float, float, float, float]:
    """
    Purpose:
        Compute the Jarque-Bera normality test statistic, p-value, skewness, and
        kurtosis matching statsmodels.stats.stattools.jarque_bera.

    Parameters:
        x: 1D array-like residual series.

    Returns:
        Tuple[float, float, float, float]: (jb_stat, jb_pvalue, skew, kurtosis).
    """
    arr = np.asarray(pd.Series(x).dropna(), dtype=float).reshape(-1)
    n = len(arr)
    if n < 3 or np.std(arr) == 0.0:
        return 0.0, 1.0, 0.0, 3.0
    s = float(stats.skew(arr, bias=True))
    k = float(stats.kurtosis(arr, fisher=False, bias=True))
    jb = float((n / 6.0) * (s ** 2 + ((k - 3.0) ** 2) / 4.0))
    pval = float(stats.chi2.sf(jb, df=2))
    return jb, pval, s, k


def het_arch(resid, nlags: int = 4) -> Tuple[float, float, float, float]:
    """
    Purpose:
        Compute Engle's ARCH LM test for heteroskedasticity on a residual series.

    Parameters:
        resid: 1D residual array-like.
        nlags (int): Number of squared-residual lags to include.

    Returns:
        Tuple[float, float, float, float]: (lm_stat, lm_pvalue, f_stat, f_pvalue).
    """
    arr = np.asarray(pd.Series(resid).dropna(), dtype=float).reshape(-1)
    r2 = (arr - np.mean(arr)) ** 2
    n = len(r2)
    k = min(int(nlags), max(1, (n // 4) - 1))
    dep = r2[k:]
    n_eff = len(dep)
    cols = [r2[k - j : n - j] for j in range(1, k + 1)]
    cols.append(np.ones(n_eff))
    X = np.column_stack(cols)
    beta, _, _, _ = linalg.lstsq(X, dep)
    hat = X @ beta
    ss_tot = float(np.sum((dep - np.mean(dep)) ** 2))
    ss_res = float(np.sum((dep - hat) ** 2))
    r2_det = max(0.0, 1.0 - ss_res / max(ss_tot, 1e-18))
    lm_stat = float(n_eff * r2_det)
    lm_pval = float(stats.chi2.sf(lm_stat, df=k))
    f_stat = float((r2_det / k) / max((1.0 - r2_det) / max(1, n_eff - k - 1), 1e-18))
    f_pval = float(stats.f.sf(f_stat, k, max(1, n_eff - k - 1)))
    return lm_stat, lm_pval, f_stat, f_pval


# ==============================================================================
# 3. ARIMA & SARIMAX ESTIMATION ENGINE (EXACT RECURSION & FORECASTING)
# ==============================================================================

class _ForecastResult:
    """
    Purpose:
        Container mimicking statsmodels PredictionResults (`predicted_mean` and `conf_int`).
    """

    def __init__(self, mean_series: pd.Series, se_series: pd.Series) -> None:
        self.predicted_mean = mean_series
        self._se = se_series

    def conf_int(self, alpha: float = 0.05) -> pd.DataFrame:
        z = float(stats.norm.ppf(1.0 - alpha / 2.0))
        lower = self.predicted_mean - z * self._se
        upper = self.predicted_mean + z * self._se
        col_name = str(self.predicted_mean.name or "y")
        return pd.DataFrame(
            {f"lower {col_name}": lower.values, f"upper {col_name}": upper.values},
            index=self.predicted_mean.index,
        )


def _expand_seasonal_poly(non_seasonal: np.ndarray, seasonal: np.ndarray, s: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Purpose:
        Multiply the non-seasonal lag polynomial (1 +/- sum a_i L^i) with the
        seasonal lag polynomial (1 +/- sum A_j L^{j*s}) to obtain the combined
        lag indices and coefficients.
    """
    # Build dict of lag -> coefficient for product
    p_non = len(non_seasonal)
    p_sea = len(seasonal)
    non_terms = {0: 1.0}
    for i in range(1, p_non + 1):
        non_terms[i] = float(non_seasonal[i - 1])
    sea_terms = {0: 1.0}
    if s > 0:
        for j in range(1, p_sea + 1):
            sea_terms[j * s] = float(seasonal[j - 1])

    combined: Dict[int, float] = {}
    for l1, c1 in non_terms.items():
        for l2, c2 in sea_terms.items():
            if l1 == 0 and l2 == 0:
                continue
            lag = l1 + l2
            combined[lag] = combined.get(lag, 0.0) + c1 * c2

    if not combined:
        return np.array([], dtype=int), np.array([], dtype=float)
    lags = np.array(sorted(combined.keys()), dtype=int)
    coefs = np.array([combined[k] for k in lags], dtype=float)
    return lags, coefs


def _difference_series(y: np.ndarray, d: int, D: int, s: int) -> np.ndarray:
    """
    Purpose:
        Apply non-seasonal differencing of order d and seasonal differencing of
        order D (period s) to a 1D array y.
    """
    w = y.copy()
    for _ in range(d):
        w = np.diff(w, n=1)
    if D > 0 and s > 0:
        for _ in range(D):
            w = w[s:] - w[:-s]
    return w


def _undifference_step(history_y: list, w_next: float, d: int, D: int, s: int) -> float:
    """
    Purpose:
        Given history_y (y_0, ..., y_{t-1}) and the predicted stationary difference
        w_t = (1 - L)^d (1 - L^s)^D y_t, compute y_t exactly via polynomial expansion
        of (1 - L)^d (1 - L^s)^D = 1 - sum_{k>=1} c_k L^k.
    """
    # Construct polynomial (1 - L)^d * (1 - L^s)^D
    poly = np.array([1.0], dtype=float)
    for _ in range(d):
        poly = np.convolve(poly, np.array([1.0, -1.0]))
    if D > 0 and s > 0:
        sea_poly = np.zeros(s + 1, dtype=float)
        sea_poly[0] = 1.0
        sea_poly[s] = -1.0
        for _ in range(D):
            poly = np.convolve(poly, sea_poly)
    # poly[0] * y_t + sum_{k=1}^M poly[k] * y_{t-k} = w_next
    val = float(w_next)
    max_k = min(len(poly) - 1, len(history_y))
    for k in range(1, max_k + 1):
        val -= float(poly[k]) * float(history_y[-k])
    return val


class _FittedSARIMAXResult:
    """
    Purpose:
        Result object returned by ARIMA.fit() and SARIMAX.fit(), providing `.aic`,
        `.bic`, `.hqic`, `.llf`, `.params`, `.bse`, `.zvalues`, `.pvalues`, `.resid`,
        `.mle_retvals`, `.conf_int()`, `.get_forecast()`, and `.test_heteroskedasticity()`.
    """

    def __init__(
        self,
        endog: pd.Series,
        order: Tuple[int, int, int],
        seasonal_order: Tuple[int, int, int, int],
        param_names: list,
        params: np.ndarray,
        bse: np.ndarray,
        llf: float,
        n_eff: int,
        resid_full: pd.Series,
        converged: bool,
    ) -> None:
        self.endog = endog
        self.order = order
        self.seasonal_order = seasonal_order
        self.params = pd.Series(params, index=param_names, dtype=float)
        self.bse = pd.Series(bse, index=param_names, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            z_arr = np.where(bse > 0, params / bse, 0.0)
        self.zvalues = pd.Series(z_arr, index=param_names, dtype=float)
        p_arr = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_arr)))
        self.pvalues = pd.Series(p_arr, index=param_names, dtype=float)

        k_params = len(params)
        self.llf = float(llf)
        self.aic = float(-2.0 * self.llf + 2.0 * k_params)
        self.bic = float(-2.0 * self.llf + k_params * np.log(max(n_eff, 1)))
        self.hqic = float(-2.0 * self.llf + 2.0 * k_params * np.log(max(np.log(max(n_eff, 2)), 1.0)))
        self.resid = resid_full
        self.mle_retvals = {"converged": bool(converged)}

    def conf_int(self, alpha: float = 0.05) -> pd.DataFrame:
        z = float(stats.norm.ppf(1.0 - alpha / 2.0))
        lower = self.params - z * self.bse
        upper = self.params + z * self.bse
        return pd.DataFrame({0: lower, 1: upper}, index=self.params.index)

    def test_heteroskedasticity(self, method: str = "breakvar") -> np.ndarray:
        """
        Purpose:
            Compute the two-sided breakvar F-test for heteroskedasticity on the
            effective residuals (comparing variance of the last third vs. first third),
            matching statsmodels SARIMAXResults.test_heteroskedasticity('breakvar').
        """
        p, d, q = self.order
        P, D, Q, s = self.seasonal_order
        skip = d + (D * s if s > 0 else 0)
        r = self.resid.iloc[skip:].dropna().values
        n = len(r)
        h = max(1, int(np.round(n / 3.0)))
        r1 = r[:h]
        r2 = r[-h:]
        denom = float(np.sum(r1 ** 2))
        if denom <= 0:
            return np.array([[1.0, 1.0]])
        h_stat = float(np.sum(r2 ** 2) / denom)
        p_one = min(stats.f.cdf(h_stat, h, h), stats.f.sf(h_stat, h, h))
        p_two = float(min(1.0, 2.0 * p_one))
        return np.array([[h_stat, p_two]])

    def get_forecast(self, steps: int = 1) -> _ForecastResult:
        p, d, q = self.order
        P, D, Q, s = self.seasonal_order
        ar_params = np.array([self.params[f"ar.L{i}"] for i in range(1, p + 1)], dtype=float)
        ma_params = np.array([self.params[f"ma.L{j}"] for j in range(1, q + 1)], dtype=float)
        sar_params = np.array([self.params[f"ar.S.L{i * s}"] for i in range(1, P + 1)], dtype=float)
        sma_params = np.array([self.params[f"ma.S.L{j * s}"] for j in range(1, Q + 1)], dtype=float)
        sigma2 = float(max(self.params.get("sigma2", 1.0), 1e-12))

        # Multiplicative AR: (1 - sum phi_i L^i)(1 - sum Phi_j L^{j*s}) = 1 - sum alpha_k L^k
        ar_lags, ar_comb = _expand_seasonal_poly(-ar_params, -sar_params, s)
        ar_coefs = -ar_comb  # since w_t = sum (-ar_comb_k) w_{t-k} + ...
        ma_lags, ma_coefs = _expand_seasonal_poly(ma_params, sma_params, s)

        y_hist = list(self.endog.values.astype(float))
        w_hist = list(_difference_series(np.asarray(y_hist), d, D, s))
        skip = d + (D * s if s > 0 else 0)
        eps_hist = list(self.resid.iloc[skip:].values.astype(float))

        # Multi-step forecast recursion
        preds_y = []
        for _ in range(int(steps)):
            w_hat = 0.0
            for lag, c in zip(ar_lags, ar_coefs):
                if lag <= len(w_hist):
                    w_hat += float(c) * float(w_hist[-lag])
            for lag, c in zip(ma_lags, ma_coefs):
                if lag <= len(eps_hist):
                    w_hat += float(c) * float(eps_hist[-lag])
            y_hat = _undifference_step(y_hist, w_hat, d, D, s)
            preds_y.append(y_hat)
            y_hist.append(y_hat)
            w_hist.append(w_hat)
            eps_hist.append(0.0)

        # Compute infinite-MA psi weights on Y_t for exact forecast error variance
        # (1 - L)^d (1 - L^s)^D Phi(L) Psi(L) = Theta(L)
        diff_poly = np.array([1.0], dtype=float)
        for _ in range(d):
            diff_poly = np.convolve(diff_poly, np.array([1.0, -1.0]))
        if D > 0 and s > 0:
            sea_d = np.zeros(s + 1, dtype=float)
            sea_d[0] = 1.0
            sea_d[s] = -1.0
            for _ in range(D):
                diff_poly = np.convolve(diff_poly, sea_d)

        max_ar_lag = int(ar_lags.max()) if len(ar_lags) > 0 else 0
        ar_poly = np.zeros(max_ar_lag + 1, dtype=float)
        ar_poly[0] = 1.0
        for lag, c in zip(ar_lags, ar_comb):
            ar_poly[lag] = float(c)
        full_ar_poly = np.convolve(diff_poly, ar_poly)

        max_ma_lag = int(ma_lags.max()) if len(ma_lags) > 0 else 0
        ma_poly = np.zeros(max_ma_lag + 1, dtype=float)
        ma_poly[0] = 1.0
        for lag, c in zip(ma_lags, ma_coefs):
            ma_poly[lag] = float(c)

        psi = np.zeros(int(steps), dtype=float)
        psi[0] = 1.0
        for h in range(1, int(steps)):
            val = float(ma_poly[h]) if h < len(ma_poly) else 0.0
            for j in range(1, min(h + 1, len(full_ar_poly))):
                val -= float(full_ar_poly[j]) * psi[h - j]
            psi[h] = val

        fc_vars = sigma2 * np.cumsum(psi ** 2)
        fc_se = np.sqrt(np.maximum(fc_vars, 1e-12))

        # Construct forecast index
        if isinstance(self.endog.index, pd.DatetimeIndex) and len(self.endog.index) >= 2:
            freq = self.endog.index.freq or pd.infer_freq(self.endog.index) or "MS"
            fc_index = pd.date_range(
                start=self.endog.index[-1] + pd.tseries.frequencies.to_offset(freq),
                periods=int(steps),
                freq=freq,
            )
        else:
            fc_index = pd.RangeIndex(start=len(self.endog), stop=len(self.endog) + int(steps))

        mean_s = pd.Series(preds_y, index=fc_index, name="predicted_mean")
        se_s = pd.Series(fc_se, index=fc_index, name="se")
        return _ForecastResult(mean_s, se_s)


def _fit_multiplicative_sarimax(
    endog: pd.Series,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
    enforce_stationarity: bool = True,
    enforce_invertibility: bool = True,
) -> _FittedSARIMAXResult:
    """
    Purpose:
        Estimate a general ARIMA(p, d, q) or SARIMAX(p, d, q)(P, D, Q, s) model via
        Gaussian Conditional/Exact-Variance Likelihood Optimization using L-BFGS-B.
    """
    s_series = pd.Series(endog).astype(float).dropna()
    y = s_series.values
    p, d, q = int(order[0]), int(order[1]), int(order[2])
    P, D, Q, s = (
        int(seasonal_order[0]),
        int(seasonal_order[1]),
        int(seasonal_order[2]),
        int(seasonal_order[3]),
    )

    w = _difference_series(y, d, D, s)
    n_w = len(w)
    # Match statsmodels SARIMAX effective sample reduction when enforce_stationarity=False
    eff_skip = 0
    if not enforce_stationarity or not enforce_invertibility:
        eff_skip = max(p + P * s, q + Q * s)
    n_eff = max(5, n_w - eff_skip)

    param_names = (
        [f"ar.L{i}" for i in range(1, p + 1)]
        + [f"ma.L{j}" for j in range(1, q + 1)]
        + [f"ar.S.L{i * s}" for i in range(1, P + 1)]
        + [f"ma.S.L{j * s}" for j in range(1, Q + 1)]
    )
    n_theta = len(param_names)

    def _compute_residuals(theta: np.ndarray) -> np.ndarray:
        ar_p = theta[:p]
        ma_q = theta[p : p + q]
        sar_P = theta[p + q : p + q + P]
        sma_Q = theta[p + q + P : p + q + P + Q]

        ar_lags, ar_comb = _expand_seasonal_poly(-ar_p, -sar_P, s)
        ar_coefs = -ar_comb
        ma_lags, ma_coefs = _expand_seasonal_poly(ma_q, sma_Q, s)

        eps = np.zeros(n_w, dtype=float)
        has_ar = len(ar_lags) > 0
        has_ma = len(ma_lags) > 0
        for t in range(n_w):
            pred = 0.0
            if has_ar:
                for lag, c in zip(ar_lags, ar_coefs):
                    if t >= lag:
                        pred += c * w[t - lag]
            if has_ma:
                for lag, c in zip(ma_lags, ma_coefs):
                    if t >= lag:
                        pred += c * eps[t - lag]
            eps[t] = w[t] - pred
            if abs(eps[t]) > 1e8:
                eps[t:] = 1e8
                break
        return eps

    def _neg_loglike(theta: np.ndarray) -> float:
        if np.any(np.abs(theta) >= 0.995):
            return 1e12
        eps = _compute_residuals(theta)
        eps_eval = eps[eff_skip:]
        sigma2 = float(np.mean(eps_eval ** 2))
        if sigma2 <= 1e-15 or not np.isfinite(sigma2):
            return 1e12
        llf = -0.5 * len(eps_eval) * (1.0 + np.log(2.0 * np.pi) + np.log(sigma2))
        return float(-llf)

    if n_theta > 0:
        # Initial guess from sample autocorrelations of w
        r_w = acf(w, nlags=max(1, p + q + (P + Q) * max(s, 1)))
        x0 = np.zeros(n_theta, dtype=float)
        for i in range(p):
            x0[i] = np.clip(r_w[i + 1] * 0.5, -0.7, 0.7)
        for j in range(q):
            x0[p + j] = np.clip(r_w[j + 1] * 0.4, -0.7, 0.7)
        for i in range(P):
            idx_lag = min((i + 1) * s, len(r_w) - 1)
            x0[p + q + i] = np.clip(r_w[idx_lag] * 0.5, -0.7, 0.7)
        for j in range(Q):
            idx_lag = min((j + 1) * s, len(r_w) - 1)
            x0[p + q + P + j] = np.clip(r_w[idx_lag] * 0.4, -0.7, 0.7)

        bounds = [(-0.985, 0.985)] * n_theta
        opt = optimize.minimize(_neg_loglike, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 400})
        theta_hat = opt.x
        converged = bool(opt.success)
    else:
        theta_hat = np.array([], dtype=float)
        converged = True

    eps_hat = _compute_residuals(theta_hat)
    eps_eval = eps_hat[eff_skip:]
    sigma2_hat = float(max(np.mean(eps_eval ** 2), 1e-12))
    llf_val = float(-0.5 * len(eps_eval) * (1.0 + np.log(2.0 * np.pi) + np.log(sigma2_hat)))

    # Approximate standard errors via finite-difference Hessian of log-likelihood
    all_names = param_names + ["sigma2"]
    all_params = np.append(theta_hat, sigma2_hat)
    all_bse = np.zeros(len(all_params), dtype=float)
    if n_theta > 0:
        step = 1e-4
        hess = np.zeros((n_theta, n_theta), dtype=float)
        f0 = _neg_loglike(theta_hat)
        for i in range(n_theta):
            ei = np.zeros(n_theta, dtype=float)
            ei[i] = step
            f_plus = _neg_loglike(theta_hat + ei)
            f_minus = _neg_loglike(theta_hat - ei)
            hess[i, i] = max((f_plus - 2.0 * f0 + f_minus) / (step ** 2), 1.0)
        cov_diag = 1.0 / np.maximum(np.diag(hess), 1.0)
        all_bse[:n_theta] = np.sqrt(cov_diag)
    all_bse[-1] = float(sigma2_hat * np.sqrt(2.0 / max(len(eps_eval), 1)))

    # Reconstruct full residual series aligned with endog.index
    skip_init = d + (D * s if s > 0 else 0)
    full_resid = np.zeros(len(y), dtype=float)
    if skip_init > 0:
        full_resid[:skip_init] = y[:skip_init]
    full_resid[skip_init:] = eps_hat
    resid_series = pd.Series(full_resid, index=s_series.index, name="resid")

    return _FittedSARIMAXResult(
        endog=s_series,
        order=(p, d, q),
        seasonal_order=(P, D, Q, s),
        param_names=all_names,
        params=all_params,
        bse=all_bse,
        llf=llf_val,
        n_eff=len(eps_eval),
        resid_full=resid_series,
        converged=converged,
    )


class ARIMA:
    """
    Purpose:
        Drop-in ARIMA(p, d, q) compatibility class when statsmodels is not installed.
    """

    def __init__(self, endog, order: Tuple[int, int, int] = (0, 0, 0), **kwargs) -> None:
        self.endog = endog
        self.order = (int(order[0]), int(order[1]), int(order[2]))

    def fit(self, **fit_kwargs) -> _FittedSARIMAXResult:
        return _fit_multiplicative_sarimax(
            endog=self.endog,
            order=self.order,
            seasonal_order=(0, 0, 0, 0),
            enforce_stationarity=True,
            enforce_invertibility=True,
        )


class SARIMAX:
    """
    Purpose:
        Drop-in SARIMAX(p, d, q)(P, D, Q, s) compatibility class when statsmodels is not installed.
    """

    def __init__(
        self,
        endog,
        order: Tuple[int, int, int] = (1, 0, 0),
        seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
        enforce_stationarity: bool = True,
        enforce_invertibility: bool = True,
        **kwargs,
    ) -> None:
        self.endog = endog
        self.order = (int(order[0]), int(order[1]), int(order[2]))
        self.seasonal_order = (
            int(seasonal_order[0]),
            int(seasonal_order[1]),
            int(seasonal_order[2]),
            int(seasonal_order[3]),
        )
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility

    def fit(self, disp: bool = False, **fit_kwargs) -> _FittedSARIMAXResult:
        return _fit_multiplicative_sarimax(
            endog=self.endog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
        )
