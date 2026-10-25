# src/lab02_utils/modeling.py
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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


def plot_forecast(train, test, forecast, conf_int=None, title: str = "", save_path: str = None) -> None:
    """
    1. Muc dich:
       Ve do thi so sanh Train, Test (thuc te) va Forecast voi do phan giai cao, 
       chi tiet den tung diem du lieu (co marker) va phan tach ro rang cac giai doan.
    """
    fig, ax = plt.subplots(figsize=(15, 7)) # Kéo dài form ngang thêm một chút
    
    # 1. Vẽ chuỗi Train (Thêm marker='.' để thấy rõ điểm dữ liệu của từng tháng)
    ax.plot(train.index, train.values, color="tab:blue", linewidth=1.5, alpha=0.7, 
            marker='.', markersize=4, label="Train (80%)")
    
    # 2. Vẽ chuỗi Test (Marker hình tròn nhỏ)
    ax.plot(test.index, test.values, color="tab:orange", linewidth=2.2, 
            marker='o', markersize=5, label="Test thực tế")
    
    # 3. Vẽ chuỗi Forecast (Marker hình thoi)
    ax.plot(forecast.index, forecast.values, color="forestgreen", linestyle="--",
            linewidth=2.2, marker='D', markersize=4, label="Dự báo (Forecast)")
    
    # 4. Vẽ khoảng tin cậy 95%
    if conf_int is not None:
        ax.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                        color="forestgreen", alpha=0.15, label="Khoảng tin cậy 95%")
        
    # 5. Vạch kẻ dọc phân chia ranh giới Train & Test
    split_date = test.index[0]
    ax.axvline(x=split_date, color='black', linestyle='--', linewidth=1.5, alpha=0.8)
    
    # Thêm text chú thích cho 2 giai đoạn 
    ax.text(split_date - pd.DateOffset(months=3), ax.get_ylim()[1]*0.95, 
            'Huấn luyện (Train)', ha='right', color='tab:blue', fontweight='bold')
    ax.text(split_date + pd.DateOffset(months=2), ax.get_ylim()[1]*0.95, 
            'Kiểm thử (Test)', ha='left', color='tab:orange', fontweight='bold')

    # 6. TINH CHỈNH TRỤC X: Hiển thị nhãn mỗi 6 tháng, định dạng Tháng/Năm
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
    
    # Giữ vạch chia nhỏ (minor ticks) cho từng tháng đơn lẻ
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
    
    # Xoay nhãn 45 độ và căn lề phải để chữ không đâm vào trục
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('right')
    
    # 7. Tinh chỉnh Lưới, Tiêu đề và Legend
    ax.grid(True, which='major', linestyle='-', alpha=0.6) 
    ax.grid(True, which='minor', linestyle=':', alpha=0.4) 
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Thời gian (Tháng/Năm)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Số hành khách (nghìn người)", fontsize=11, fontweight='bold')
    
    # Viền legend có nền trắng tinh, đổ bóng nhẹ để không bị đường biểu đồ đâm xuyên
    ax.legend(loc="upper left", frameon=True, shadow=True, fancybox=True, facecolor='white', framealpha=0.95)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight") 
    plt.show()


    
def plot_arima_vs_sarima(train, test, fc_arima1, fc_arima2, fc_sarima, conf_int_sarima, 
                         names: dict, r2_scores: dict, title: str = "", save_path: str = None) -> None:
    """
    Hàm chuyên dụng để vẽ biểu đồ đối chiếu trực tiếp hiệu năng dự báo 
    giữa 2 mô hình ARIMA và 1 mô hình SARIMA tốt nhất.
    """
    fig, ax = plt.subplots(figsize=(15, 7.5))
    
    # Vẽ tập Train (Làm mờ để nhường sự chú ý cho tập Test)
    ax.plot(train.index, train.values, color='tab:blue', linewidth=1.5, alpha=0.5, 
            label='Tập Huấn luyện (Train: 01/1949 - 07/1958)')
    
    # Vẽ tập Test (Đường màu đen đậm, có marker để thấy rõ thực tế)
    ax.plot(test.index, test.values, color='black', linewidth=2.8, marker='o', markersize=4,
            label='Thực tế (Test: 08/1958 - 12/1960)')
    
    # Vẽ 2 đường ARIMA (Nét đứt/chấm, màu cảnh báo)
    ax.plot(test.index, fc_arima1.values, color='tab:red', linestyle=':', linewidth=2.2,
            label=f"{names['arima1']} [Tuyến tính, R² = {r2_scores['arima1']:.2f}]")
    
    ax.plot(test.index, fc_arima2.values, color='tab:purple', linestyle='-.', linewidth=2.2,
            label=f"{names['arima2']} [Tuyến tính, R² = {r2_scores['arima2']:.2f}]")
    
    # Vẽ đường SARIMA tốt nhất (Màu xanh lá đậm, nét liền nổi bật)
    ax.plot(test.index, fc_sarima.values, color='forestgreen', linestyle='-', linewidth=2.8,
            label=f"{names['sarima']} [Khớp chu kỳ mùa vụ, R² = {r2_scores['sarima']:.2f}]")
    
    # Vẽ dải tin cậy cho SARIMA
    if conf_int_sarima is not None:
        ax.fill_between(conf_int_sarima.index, conf_int_sarima.iloc[:, 0], conf_int_sarima.iloc[:, 1],
                        color='forestgreen', alpha=0.15, label='SARIMA 95% Confidence Interval')

    # Đường vạch dọc phân chia Train / Test
    split_date = test.index[0]
    ax.axvline(x=split_date, color='dimgrey', linestyle='--', linewidth=1.5, alpha=0.8)
    
    ax.text(split_date - pd.DateOffset(months=3), ax.get_ylim()[1]*0.95, 
            'Huấn luyện', ha='right', color='dimgrey', fontstyle='italic', fontsize=11)
    ax.text(split_date + pd.DateOffset(months=2), ax.get_ylim()[1]*0.95, 
            'Dự báo (Ngoài mẫu)', ha='left', color='black', fontweight='bold', fontsize=11)

    # Tinh chỉnh hiển thị (Lưới, Tiêu đề, Legend)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.grid(True, which='major', linestyle='-', alpha=0.6)
    ax.grid(True, which='minor', linestyle=':', alpha=0.3)
    
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel('Năm', fontsize=12)
    ax.set_ylabel('Số hành khách (nghìn người)', fontsize=12)
    
    # Khung Legend có viền bo tròn, đổ bóng để không bị chìm vào nền
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.95, shadow=True, fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight') # dpi=150 cho ảnh sắc nét
    plt.show()
    
    

from pathlib import Path

def plot_log_transformation(train, test=None, save_path: str | Path = None):
    """"
    Biến đổi Log tự nhiên để ổn định phương sai và vẽ biểu đồ chi tiết theo từng tháng.
    Args:
        train (pd.Series hoặc pd.DataFrame): Dữ liệu huấn luyện (index phải là Datetime).
        test (pd.Series hoặc pd.DataFrame, optional): Dữ liệu kiểm thử.
        fig_dir (str hoặc Path, optional): Đường dẫn thư mục để lưu ảnh.
        
    Returns:
        train_log: Chuỗi train sau khi lấy log.
        test_log: Chuỗi test sau khi lấy log (None nếu không truyền test).
        diff_log: Chuỗi train sau khi lấy log và sai phân (d=1, D=1)
    """
    # 1. Biến đổi Log tự nhiên
    train_log = np.log(train)
    test_log = np.log(test) if test is not None else None

    # 2. Trực quan hóa đặc tính chuỗi sau khi lấy Log
    fig, axes = plt.subplots(2, 1, figsize=(18, 10))

    # Ô 1: Chuỗi sau khi lấy ln(Train)
    axes[0].plot(train_log.index, train_log.values, color='#2ca02c', linewidth=1.5, marker='.', markersize=4)
    axes[0].set_title('1. Chuỗi sau khi lấy ln(Train): Biên độ sóng đều đặn (Homoscedasticity)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('ln(Passengers)', fontsize=10, fontweight='bold')
    
    # Ô 2: Sai phân chuỗi Log (d=1, D=1)
    diff_log = train_log.diff(1).diff(12).dropna()
    axes[1].plot(diff_log.index, diff_log.values, color='#9467bd', linewidth=1.5, marker='.', markersize=4)
    axes[1].axhline(0, color='black', linestyle='--', alpha=0.7)
    axes[1].set_title('2. Sai phân chuỗi Log (d=1, D=1): Dao động dừng chuẩn & phương sai đồng nhất', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Δ Δ₁₂ ln(Yₜ)', fontsize=10, fontweight='bold')
    
    for ax in axes:
        # Cài đặt hiển thị từng tháng một
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        # Định dạng nhãn theo kiểu Tháng/Năm (VD: 01/1949)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
        # Xoay nhãn 90 độ và thu nhỏ font chữ để không bị chồng chéo
        ax.tick_params(axis='x', rotation=90, labelsize=8)
        # Thêm lưới chi tiết cho trục x
        ax.grid(True, which='major', linestyle='--', alpha=0.5)

    plt.tight_layout()
    
    if save_path:
         plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return train_log, test_log, diff_log