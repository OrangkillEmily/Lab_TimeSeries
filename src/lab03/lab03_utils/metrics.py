"""Regression metrics implemented with NumPy (no sklearn metrics)."""
import numpy as np


def regression_metrics(y_true, y_pred):
    """Purpose: calculate error and goodness-of-fit metrics.

    Input: actual and predicted 1-D arrays with equal non-zero length.
    Output: dict containing MAE, MSE, RMSE, MAPE (%), sMAPE (%) and R2.
    """
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(y_pred, dtype=float).reshape(-1)
    if y.size == 0 or y.size != p.size:
        raise ValueError("y_true/y_pred phải cùng độ dài và không rỗng.")
    error = p - y
    abs_error = np.abs(error)
    mse = float(np.mean(error ** 2))
    denom = np.abs(y)
    nz = denom > 1e-12
    mape = float(np.mean(abs_error[nz] / denom[nz]) * 100) if np.any(nz) else float("nan")
    smape_denom = np.abs(y) + np.abs(p)
    snz = smape_denom > 1e-12
    smape = float(np.mean(2 * abs_error[snz] / smape_denom[snz]) * 100) if np.any(snz) else float("nan")
    ss_res = float(np.sum(error ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
    return {"MAE": float(np.mean(abs_error)), "MSE": mse, "RMSE": float(np.sqrt(mse)),
            "MAPE (%)": mape, "sMAPE (%)": smape, "R2": r2}
