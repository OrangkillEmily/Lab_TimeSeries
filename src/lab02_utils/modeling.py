"""Các hàm huấn luyện mô hình ARIMA/SARIMA và đánh giá dự báo cho Lab 02."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


def fit_arima(series, order=(1, 1, 1), **fit_kwargs):
    """Fit mô hình ARIMA trên chuỗi train và trả về kết quả statsmodels.

    Parameters
    ----------
    series : array-like hoặc pandas.Series
        Chuỗi thời gian tập huấn luyện theo đúng thứ tự thời gian.
    order : tuple, mặc định (1, 1, 1)
        Bộ tham số (p, d, q) của ARIMA.
    **fit_kwargs : dict
        Các tuỳ chọn truyền tiếp vào ARIMA.fit().
    """
    values = pd.Series(series).astype(float).dropna()
    if values.empty:
        raise ValueError("Chuỗi train rỗng sau khi loại giá trị thiếu.")
    return ARIMA(values, order=order).fit(**fit_kwargs)


def fit_sarima(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), **fit_kwargs):
    """Fit mô hình SARIMA bằng SARIMAX và trả về kết quả statsmodels."""
    values = pd.Series(series).astype(float).dropna()
    if values.empty:
        raise ValueError("Chuỗi train rỗng sau khi loại giá trị thiếu.")
    model = SARIMAX(
        values,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False, **fit_kwargs)


def calculate_metrics(y_true, y_pred):
    """Tính MAE, RMSE, MAPE (%) và R² giữa giá trị thực tế và dự báo."""
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    if actual.size != predicted.size:
        raise ValueError(
            f"Số giá trị thực tế ({actual.size}) khác số dự báo ({predicted.size})."
        )
    if actual.size == 0:
        raise ValueError("Không thể đánh giá trên tập dữ liệu rỗng.")
    if not (np.isfinite(actual).all() and np.isfinite(predicted).all()):
        raise ValueError("Dữ liệu thực tế và dự báo phải là các số hữu hạn.")

    errors = actual - predicted
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    nonzero = actual != 0
    mape = float(np.mean(np.abs(errors[nonzero] / actual[nonzero])) * 100) if nonzero.any() else float("nan")
    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}
