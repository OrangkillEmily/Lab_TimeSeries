# src/lab02_utils/modeling.py
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX


def fit_arima(train: pd.Series, order: tuple = (1, 1, 1), **fit_kwargs):
    """
    1. Muc dich:
       Huan luyen (fit) mo hinh ARIMA(p,d,q) tren tap Train bang class `ARIMA`
       cua statsmodels, in ra AIC, BIC va kiem tra mo hinh co hoi tu hay khong.

    2. Tham so dau vao:
       - train (pd.Series hoac array-like): Chuoi Train, giu dung thu tu thoi gian.
       - order (tuple): Bo tham so (p, d, q). Mac dinh (1, 1, 1).
       - **fit_kwargs: Cac tuy chon truyen tiep vao ARIMA.fit().

    3. Ket qua dau ra:
       - ARIMAResults: Doi tuong ket qua da fit (dung cho residual diagnostics, forecast).
    """
    values = pd.Series(train).astype(float).dropna()
    if values.empty:
        raise ValueError("Chuoi train rong sau khi loai bo gia tri thieu.")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = ARIMA(values, order=order)
        result = model.fit(**fit_kwargs)

    converged = result.mle_retvals.get("converged", True) if result.mle_retvals else True
    has_warning = any("onverge" in str(w.message) for w in caught)

    print(f"===== ARIMA{order} =====")
    print(f"AIC : {result.aic:.2f}")
    print(f"BIC : {result.bic:.2f}")
    print(f"Hoi tu: {'CO' if converged and not has_warning else 'KHONG / co canh bao'}")
    return result


def fit_sarima(train: pd.Series, order: tuple = (1, 1, 1),
               seasonal_order: tuple = (1, 1, 1, 12), **fit_kwargs):
    """
    1. Muc dich:
       Huan luyen (fit) mo hinh SARIMA(p,d,q)(P,D,Q)s tren tap Train bang `SARIMAX`
       cua statsmodels (phuc vu Bai 6), in ra AIC va BIC.

    2. Tham so dau vao:
       - train (pd.Series hoac array-like): Chuoi Train theo thu tu thoi gian.
       - order (tuple): Bo (p, d, q) phan khong mua vu. Mac dinh (1, 1, 1).
       - seasonal_order (tuple): Bo (P, D, Q, s) phan mua vu (s = 12 voi du lieu thang).
       - **fit_kwargs: Cac tuy chon truyen tiep vao fit().

    3. Ket qua dau ra:
       - SARIMAXResults: Doi tuong ket qua da fit.
    """
    values = pd.Series(train).astype(float).dropna()
    if values.empty:
        raise ValueError("Chuoi train rong sau khi loai bo gia tri thieu.")

    model = SARIMAX(
        values,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False, **fit_kwargs)
    print(f"===== SARIMA{order}x{seasonal_order} =====")
    print(f"AIC : {result.aic:.2f}")
    print(f"BIC : {result.bic:.2f}")
    return result


def residual_diagnostics(result, order: tuple, lags: int = 24, title: str = "",
                         save_path: str = None, n_skip: int = None) -> dict:
    """
    1. Muc dich:
       Chan doan phan du: ve Residuals theo thoi gian + ACF cua Residuals, tinh mean
       va thuc hien kiem dinh Ljung-Box (H0: phan du khong tu tuong quan = white noise).
       Bo d phan du dau tien vi voi d >= 1 statsmodels khoi tao mo hinh bang chinh
       gia tri quan sat dau tien, lam phan du dau bi lon bat thuong.

    2. Tham so dau vao:
       - result (ARIMAResults): Mo hinh da fit tren Train.
       - order (tuple): Bo (p, d, q) cua mo hinh (lay d de bo phan du dau).
       - lags (int): So lag ve ACF va dung cho Ljung-Box. Mac dinh 24 (= 2 chu ky nam).
       - title (str): Ten mo hinh hien thi tren do thi.
       - save_path (str | None): Duong dan luu anh .png (None = khong luu).
       - n_skip (int | None): So phan du dau bi bo (mac dinh = d; voi SARIMA nen dat d + D*s).

    3. Ket qua dau ra:
       - dict: {'mean', 'std', 'ljung_box_p'}; dong thoi hien thi do thi.
    """
    skip = order[1] if n_skip is None else n_skip
    resid = result.resid.iloc[skip:]
    lb_df = acorr_ljungbox(resid, lags=[lags], return_df=True)
    lb_p = lb_df["lb_pvalue"].iloc[0]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].plot(resid.index, resid.values, color="tab:blue", label="Residuals")
    axes[0].axhline(0, color="red", linestyle="--", alpha=0.6, label="y = 0")
    axes[0].set_title(f"Residuals theo thoi gian - {title}")
    axes[0].set_xlabel("Nam")
    axes[0].set_ylabel("Residual")
    axes[0].legend()

    plot_acf(resid, lags=lags, zero=False, ax=axes[1])
    axes[1].set_title(f"ACF cua Residuals - {title}")
    axes[1].set_xlabel("Lag")
    axes[1].set_ylabel("Tu tuong quan")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()

    summary = {"mean": float(resid.mean()), "std": float(resid.std()), "ljung_box_p": float(lb_p)}
    print(f"Mean residual : {summary['mean']:.3f}  (Std = {summary['std']:.3f})")
    print(f"Ljung-Box (lag={lags}) p-value : {lb_p:.4f}")
    if lb_p < 0.05:
        print("=> Con tu tuong quan (khong phai white noise).")
    else:
        print("=> Chua thay tu tuong quan o lag kiem tra.")
    return summary


def forecast_test(result, steps: int):
    """
    1. Muc dich:
       Du bao `steps` buoc tiep theo sau tap Train (tuong ung dung so thang cua Test).

    2. Tham so dau vao:
       - result (ARIMAResults): Mo hinh da fit tren Train.
       - steps (int): So buoc can du bao (= len(test)).

    3. Ket qua dau ra:
       - tuple (pd.Series, pd.DataFrame): Gia tri du bao va khoang tin cay 95%.
    """
    fc = result.get_forecast(steps=steps)
    return fc.predicted_mean, fc.conf_int(alpha=0.05)


def evaluate_forecast(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """
    1. Muc dich:
       Tinh cac chi so sai so giua gia tri thuc te (Test) va du bao: 3 chi so bat buoc
       MAE, RMSE, MAPE (%) va them R2 (R2 < 0 nghia la kem hon viec doan bang trung binh Test).

    2. Tham so dau vao:
       - y_true (pd.Series hoac array-like): Gia tri thuc te cua tap Test.
       - y_pred (pd.Series hoac array-like): Gia tri du bao cung do dai voi y_true.

    3. Ket qua dau ra:
       - dict: {'MAE', 'RMSE', 'MAPE (%)', 'R2'}.
    """
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    if actual.size != predicted.size or actual.size == 0:
        raise ValueError(f"Do dai khong hop le: thuc te = {actual.size}, du bao = {predicted.size}.")

    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    nonzero = actual != 0
    mape = float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100) if nonzero.any() else float("nan")
    r2 = float(r2_score(actual, predicted))
    return {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape, "R2": r2}


def calculate_metrics(y_true, y_pred) -> dict:
    """
    Ham tinh cac chi so MAE, RMSE, MAPE (ho tro ca khoa 'MAPE' va 'MAPE (%)') va R2.
    Dong thoi dong vai tro alias tuong thich cho evaluate_forecast.
    """
    metrics = evaluate_forecast(y_true, y_pred)
    # Cung cap them khoa 'MAPE' de tuong thich voi cac notebook cu
    metrics["MAPE"] = metrics["MAPE (%)"]
    return metrics


def plot_forecast(train, test, forecast, conf_int=None, title: str = "",
                  save_path: str = None) -> None:
    """
    1. Muc dich:
       Ve do thi so sanh Train, Test (thuc te) va Forecast (kem khoang tin cay 95% neu co).

    2. Tham so dau vao:
       - train, test (pd.Series): Chuoi Train va Test thuc te.
       - forecast (pd.Series): Gia tri du bao tren tap Test.
       - conf_int (pd.DataFrame | None): Khoang tin cay 95% (2 cot: lower, upper).
       - title (str): Tieu de do thi.
       - save_path (str | None): Duong dan luu anh .png (None = khong luu).

    3. Ket qua dau ra:
       - None: Hien thi (va luu neu co save_path) do thi.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train.values, color="tab:blue", label="Train (80%)")
    plt.plot(test.index, test.values, color="tab:orange", label="Test (thuc te)")
    plt.plot(forecast.index, forecast.values, color="tab:green", linestyle="--",
             linewidth=2, label="Forecast")
    if conf_int is not None:
        plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                         color="tab:green", alpha=0.15, label="Khoang tin cay 95%")
    plt.title(title)
    plt.xlabel("Nam")
    plt.ylabel("So hanh khach (nghin nguoi)")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
