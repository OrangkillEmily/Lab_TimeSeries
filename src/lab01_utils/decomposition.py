import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox

def plot_decomposition(result, title, is_multiplicative=False):
    """
    Vẽ 4 thành phần của phân rã chuỗi thời gian (Classical hoặc STL).
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(result.observed, color="#1f77b4", linewidth=1.2)
    axes[0].set_ylabel("Observed", fontsize=10, fontweight="bold")
    axes[0].set_title(title, fontsize=12, fontweight="bold")

    axes[1].plot(result.trend, color="#ff7f0e", linewidth=1.5)
    axes[1].set_ylabel("Trend", fontsize=10, fontweight="bold")

    axes[2].plot(result.seasonal, color="#2ca02c", linewidth=1.2)
    axes[2].set_ylabel("Seasonal", fontsize=10, fontweight="bold")

    # Xử lý riêng phần Residual tùy theo loại mô hình
    if is_multiplicative:
        axes[3].scatter(result.resid.index, result.resid, color="#d62728", s=15, alpha=0.7)
        axes[3].axhline(1, color="black", linestyle="--", linewidth=1)
    else:
        axes[3].plot(result.resid, color="#d62728", linewidth=1.0)
        axes[3].axhline(0, color="black", linestyle="--", linewidth=1)
        
    axes[3].set_ylabel("Residual", fontsize=10, fontweight="bold")
    axes[3].set_xlabel("Thời gian", fontsize=11)

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()

def analyze_residuals(residual, lags=30):
    """
    Vẽ histogram, ACF và thực hiện kiểm định Ljung-Box cho phần dư.
    """
    residual = residual.dropna()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    # Đồ thị 1: Phân phối của Residual (Histogram)
    axes[0].hist(residual, bins=20, density=True, alpha=0.6, color="#4c72b0", edgecolor="black")
    
    # Đường phân phối chuẩn tham khảo
    mu, std = residual.mean(), residual.std()
    x_norm = np.linspace(mu - 3*std, mu + 3*std, 100)
    p_norm = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_norm - mu) / std) ** 2)
    axes[0].plot(x_norm, p_norm, 'r--', linewidth=1.5, label=f"N({mu:.2f}, {std:.2f}^2)")
    axes[0].set_title("Phân phối của Residual", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Giá trị Residual")
    axes[0].set_ylabel("Mật độ")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.3)

    # Đồ thị 2: ACF của Residual
    plot_acf(residual, lags=lags, alpha=0.05, zero=False, ax=axes[1])
    axes[1].set_title(f"ACF của Residual ({lags} lags)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Lag")
    axes[1].set_ylabel("Tự tương quan")
    axes[1].grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Thống kê định lượng
    ci_limit = 1.96 / np.sqrt(len(residual))
    acf_vals = acf(residual, nlags=lags)[1:]
    inside_ci = np.abs(acf_vals) <= ci_limit
    num_inside = np.sum(inside_ci)

    # Kiểm định Ljung-Box
    lb_test = acorr_ljungbox(residual, lags=[10, 20], return_df=True)

    print("--- KẾT QUẢ THỐNG KÊ CỦA THÀNH PHẦN RESIDUAL ---")
    print(f"1. Trung bình mẫu (Mean): {mu:.4f} (Kỳ vọng ~ 0)")
    print(f"2. Độ lệch chuẩn (Std): {std:.4f}")
    print(f"3. Giới hạn khoảng tin cậy 95%: ±{ci_limit:.4f}")
    print(f"4. Số lag ACF nằm trong khoảng tin cậy 95%: {num_inside}/{lags} ({num_inside/lags*100:.1f}%)")
    print("\n5. Kiểm định Ljung-Box Test (H0: Dữ liệu là White Noise / không tự tương quan):")
    print(lb_test)