from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.stattools import acf, pacf
except ImportError:
    from src.lab02.lab02_utils.scipy_tsa_compat import acf, pacf, plot_acf, plot_pacf

from src.lab02.lab02_utils.logging_config import get_lab02_logger, log_statistical_analysis


def _decorate_correlation_axis(
    ax: plt.Axes,
    title: str,
    lags: int,
    bound: float,
    seasonal_period: Optional[int] = 12,
) -> None:
    """
    Purpose:
        Apply consistent styling, 95% confidence bound lines, seasonal lag markers,
        axis labels, grid, and legend to an ACF or PACF matplotlib Axes object.

    Parameters:
        ax (plt.Axes): Target matplotlib Axes to style.
        title (str): Plot title.
        lags (int): Maximum lag displayed on the x-axis.
        bound (float): 95% confidence interval bound (±1.96/sqrt(n)).
        seasonal_period (Optional[int]): Seasonal period s for vertical marker lines.

    Returns:
        None.
    """
    ax.axhline(bound, color="tab:blue", linestyle="--", linewidth=1.0, alpha=0.75, label=f"Ngưỡng 95% (±{bound:.3f})")
    ax.axhline(-bound, color="tab:blue", linestyle="--", linewidth=1.0, alpha=0.75)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)

    if seasonal_period and seasonal_period > 1:
        for k in range(seasonal_period, lags + 1, seasonal_period):
            ax.axvline(
                k,
                color="tab:red",
                linestyle=":",
                linewidth=1.3,
                alpha=0.75,
                label=f"Lag mùa vụ (bội số s={seasonal_period})" if k == seasonal_period else None,
            )
        xticks = sorted(set([1, 2, 3, 4, 6] + list(range(seasonal_period, lags + 1, seasonal_period))))
    else:
        xticks = list(range(1, min(lags + 1, 7))) + list(range(10, lags + 1, 5))

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Độ trễ (Lag)", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Hệ số tương quan", fontsize=10.5, fontweight="bold")
    ax.set_xticks([k for k in xticks if k <= lags])
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9, fontsize=9)


def interpret_acf_pacf_structure(
    acf_values: np.ndarray,
    pacf_values: np.ndarray,
    bound: float,
    nlags: int,
    seasonal_period: Optional[int] = 12,
    differencing_info: str = "d=1",
) -> Dict[str, Any]:
    """
    Purpose:
        Analyze numerical ACF and PACF arrays to distinguish between short-run
        cut-off vs. tail-off behavior and seasonal spikes, proposing statistically
        grounded candidate orders for (p, q, P, Q) alongside diagnostic warnings.

    Parameters:
        acf_values (np.ndarray): Array of ACF values from lag 0 to nlags.
        pacf_values (np.ndarray): Array of PACF values from lag 0 to nlags.
        bound (float): 95% confidence threshold (±1.96/sqrt(n)).
        nlags (int): Total number of lags evaluated.
        seasonal_period (Optional[int]): Seasonal cycle length s (e.g., 12 for monthly data).
        differencing_info (str): Description of differencing applied to the input series.

    Returns:
        Dict[str, Any]: Dictionary containing p_candidates, q_candidates,
            P_candidates, Q_candidates, behavior_summary, explanations, and warnings.
    """
    s = seasonal_period if (seasonal_period and seasonal_period > 1) else 12
    short_max = min(5, s - 2, nlags)

    short_acf_sig = [k for k in range(1, short_max + 1) if abs(acf_values[k]) > bound]
    short_pacf_sig = [k for k in range(1, short_max + 1) if abs(pacf_values[k]) > bound]

    seasonal_lags = [k for k in range(s, nlags + 1, s)]
    seasonal_acf_sig = [k for k in seasonal_lags if abs(acf_values[k]) > bound]
    seasonal_pacf_sig = [k for k in seasonal_lags if abs(pacf_values[k]) > bound]

    if not short_acf_sig:
        acf_short_behavior = "Không có gai vượt ngưỡng ở độ trễ ngắn hạn (gợi ý q = 0)."
    elif len(short_acf_sig) == 1 and short_acf_sig[0] == 1:
        acf_short_behavior = f"ACF cắt cụt (cut-off) rõ nét sau lag 1 (ACF[1]={acf_values[1]:.3f}), gợi ý mạnh bậc MA ngắn hạn q = 1."
    elif len(short_acf_sig) <= 2:
        acf_short_behavior = f"ACF có gai vượt ngưỡng tại các lag ngắn hạn {short_acf_sig} rồi tắt dần/cắt cụt, gợi ý q thuộc { [0] + short_acf_sig }."
    else:
        acf_short_behavior = f"ACF vượt ngưỡng tại nhiều lag ngắn hạn {short_acf_sig} (dấu hiệu giảm dần/dao động tail-off hoặc nhiễu bậc cao)."

    if not short_pacf_sig:
        pacf_short_behavior = "Không có gai vượt ngưỡng ở độ trễ ngắn hạn (gợi ý p = 0)."
    elif short_pacf_sig == [1]:
        pacf_short_behavior = f"PACF cắt cụt (cut-off) ngay sau lag 1 (PACF[1]={pacf_values[1]:.3f}), gợi ý mạnh bậc AR ngắn hạn p = 1."
    elif short_pacf_sig[:2] == [1, 2] and len(short_pacf_sig) == 2:
        pacf_short_behavior = (
            f"PACF vượt ngưỡng tại lag 1 ({pacf_values[1]:.3f}) và lag 2 ({pacf_values[2]:.3f}) rồi cắt cụt ở lag 3, "
            "gợi ý bậc AR ngắn hạn p = 1 hoặc p = 2."
        )
    else:
        pacf_short_behavior = (
            f"PACF vượt ngưỡng tại nhiều lag ngắn hạn {short_pacf_sig} (dấu hiệu giảm dần tail-off do hiệu ứng MA ngắn hạn "
            "hoặc tương tác tự hồi quy bậc cao)."
        )

    p_candidates = sorted(set([0, 1] + ([2] if 2 in short_pacf_sig else [])))
    q_candidates = sorted(set([0, 1] + ([k for k in short_acf_sig if k <= 3])))

    if seasonal_acf_sig or seasonal_pacf_sig:
        seasonal_behavior = (
            f"Phát hiện gai mùa vụ rất rõ tại bội số chu kỳ s={s}: "
            f"ACF vượt ngưỡng ở lag {seasonal_acf_sig}, PACF vượt ngưỡng ở lag {seasonal_pacf_sig}. "
            "Nếu chuỗi chưa lấy sai phân mùa vụ (D=0), các đỉnh dương lớn tại lag 12, 24, 36 phản ánh chu kỳ mùa vụ chưa được khử "
            "(cần lấy sai phân mùa vụ D=1 trước khi chốt P, Q). Nếu chuỗi đã lấy sai phân mùa vụ (D=1), cần khảo sát đầy đủ P ∈ {0, 1} và Q ∈ {0, 1}"
        )
        P_candidates = [0, 1] 
        Q_candidates = [0, 1]
   
    else:
        acf_s_val = acf_values[s] if s <= nlags else 0.0
        pacf_s_val = pacf_values[s] if s <= nlags else 0.0
        seasonal_behavior = (
            f"Tại các lag mùa vụ {seasonal_lags}, không còn gai nào vượt ngưỡng tin cậy ±{bound:.3f} "
            f"(ACF[{s}]={acf_s_val:.3f}, PACF[{s}]={pacf_s_val:.3f}). "
            "Điều này chứng tỏ phép sai phân mùa vụ (D=1) đã triệt tiêu thành công chu kỳ lặp lại 12 tháng. "
            "Tuy nhiên theo nguyên lý Box-Jenkins, việc lấy D=1 dễ gây hiện tượng quá sai phân (over-differencing) sinh nhiễu MA mùa vụ cần Q=1, "
            "đồng thời cần dự phòng thành phần tự hồi quy mùa vụ bậc 1 (P=1) — do đó không gian khảo sát đầy đủ là P ∈ {0, 1} và Q ∈ {0, 1}."
        )
        P_candidates = [0, 1]
        Q_candidates = [0, 1]

    caution_note = (
        "CẢNH BÁO HỌC THUẬT: Đồ thị ACF/PACF chỉ là công cụ nhận diện sơ bộ (heuristic identification) để thu hẹp "
        "không gian tham số. Việc nhiều độ trễ vượt ngưỡng 95% không đồng nghĩa với điểm cắt cụt chính xác ở bậc cao "
        "(do sai số mẫu ngẫu nhiên khoảng 5% số lag có thể vượt ngưỡng, hoặc do hiệu ứng vọng từ chu kỳ mùa vụ). "
        "Mọi bộ tham số ứng viên (p, d, q)(P, D, Q)_s bắt buộc phải được kiểm chứng qua: "
        "(1) Tiêu chuẩn thông tin AIC/BIC trên cùng thang đo, (2) Kiểm định phần dư nhiễu trắng Ljung-Box, và "
        "(3) Sai số dự báo ngoài mẫu (MAE, RMSE, MAPE, R²) trên tập Test."
    )

    return {
        "differencing_info": differencing_info,
        "p_candidates": p_candidates,
        "q_candidates": q_candidates,
        "P_candidates": P_candidates,
        "Q_candidates": Q_candidates,
        "short_acf_sig": short_acf_sig,
        "short_pacf_sig": short_pacf_sig,
        "seasonal_acf_sig": seasonal_acf_sig,
        "seasonal_pacf_sig": seasonal_pacf_sig,
        "acf_short_behavior": acf_short_behavior,
        "pacf_short_behavior": pacf_short_behavior,
        "seasonal_behavior": seasonal_behavior,
        "caution_note": caution_note,
    }


def analyze_acf_pacf(
    series: pd.Series,
    dataset_name: str = "dataset1",
    nlags: int = 36,
    differencing_info: str = "d=1",
    seasonal_period: Optional[int] = 12,
    output_dir: Optional[str | Path] = None,
    file_prefix: str = "series",
    pacf_method: str = "ywm",
    dpi: int = 140,
    show_plot: bool = True,
) -> Dict[str, Any]:
    """
    Purpose:
        Perform comprehensive numerical and visual ACF/PACF analysis on a
        stationary (differenced) training time series, generating standalone ACF,
        standalone PACF, and combined ACF/PACF plots, a complete lag significance
        table, and automated Box-Jenkins parameter recommendations.

    Parameters:
        series (pd.Series): Stationary/differenced training series (never include Test data).
        dataset_name (str): Dataset identifier for logging and path resolution.
        nlags (int): Number of lags to compute and plot (default 36).
        differencing_info (str): Description of differencing applied (e.g., 'd=1' or 'd=1, D=1, s=12').
        seasonal_period (Optional[int]): Seasonal cycle length s (default 12).
        output_dir (Optional[str | Path]): Directory to save PNG plots.
        file_prefix (str): Filename prefix for saved PNG images.
        pacf_method (str): Method for PACF estimation in statsmodels (default 'ywm').
        dpi (int): Image resolution in dots per inch (default 140).
        show_plot (bool): Whether to call plt.show() for interactive notebook display.

    Returns:
        Dict[str, Any]: Dictionary containing 'n', 'bound', 'acf_values',
            'pacf_values', 'acf_sig', 'pacf_sig', 'full_table', 'summary_table',
            'interpretation', and 'figure_paths'.
    """
    logger = get_lab02_logger(dataset_name, "acf_pacf_analysis")
    if series is None or len(series.dropna()) == 0:
        raise ValueError("Input series for ACF/PACF analysis cannot be empty.")

    x = pd.Series(series).astype(float).dropna()
    n = len(x)
    max_allowed_lags = max(1, (n // 2) - 1)
    effective_lags = min(int(nlags), max_allowed_lags)
    if effective_lags < nlags:
        logger.warning(
            f"Requested nlags={nlags} exceeds safe sample limit for n={n}; adjusted to effective_lags={effective_lags}."
        )

    bound = float(1.96 / np.sqrt(n))
    acf_values = acf(x, nlags=effective_lags, fft=True)
    pacf_values = pacf(x, nlags=effective_lags, method=pacf_method)

    acf_sig = [int(k) for k in range(1, effective_lags + 1) if abs(acf_values[k]) > bound]
    pacf_sig = [int(k) for k in range(1, effective_lags + 1) if abs(pacf_values[k]) > bound]

    lags_index = list(range(1, effective_lags + 1))
    full_table = pd.DataFrame(
        {
            "Lag": lags_index,
            "ACF": [float(acf_values[k]) for k in lags_index],
            "PACF": [float(pacf_values[k]) for k in lags_index],
            "Significant ACF": [bool(k in acf_sig) for k in lags_index],
            "Significant PACF": [bool(k in pacf_sig) for k in lags_index],
        }
    )

    key_lags = [k for k in [1, 2, 3, 4, 5, 6, 12, 24, 36] if k <= effective_lags]
    summary_table = full_table[full_table["Lag"].isin(key_lags)].copy().set_index("Lag")

    interpretation = interpret_acf_pacf_structure(
        acf_values=acf_values,
        pacf_values=pacf_values,
        bound=bound,
        nlags=effective_lags,
        seasonal_period=seasonal_period,
        differencing_info=differencing_info,
    )

    figure_paths: Dict[str, Optional[str]] = {"acf": None, "pacf": None, "combined": None}
    out_dir = Path(output_dir) if output_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Biểu đồ ACF độc lập
    fig_acf, ax_acf = plt.subplots(figsize=(10, 4.5))
    plot_acf(x, lags=effective_lags, zero=False, ax=ax_acf)
    _decorate_correlation_axis(
        ax_acf,
        title=f"Autocorrelation Function (ACF) — {dataset_name.upper()} ({differencing_info})",
        lags=effective_lags,
        bound=bound,
        seasonal_period=seasonal_period,
    )
    fig_acf.tight_layout()
    if out_dir is not None:
        acf_path = out_dir / f"{file_prefix}_acf.png"
        fig_acf.savefig(acf_path, dpi=dpi, bbox_inches="tight")
        figure_paths["acf"] = str(acf_path)
    plt.close(fig_acf)

    # 2. Biểu đồ PACF độc lập
    fig_pacf, ax_pacf = plt.subplots(figsize=(10, 4.5))
    plot_pacf(x, lags=effective_lags, zero=False, method=pacf_method, ax=ax_pacf)
    _decorate_correlation_axis(
        ax_pacf,
        title=f"Partial Autocorrelation Function (PACF) — {dataset_name.upper()} ({differencing_info})",
        lags=effective_lags,
        bound=bound,
        seasonal_period=seasonal_period,
    )
    fig_pacf.tight_layout()
    if out_dir is not None:
        pacf_path = out_dir / f"{file_prefix}_pacf.png"
        fig_pacf.savefig(pacf_path, dpi=dpi, bbox_inches="tight")
        figure_paths["pacf"] = str(pacf_path)
    plt.close(fig_pacf)

    # 3. Biểu đồ kết hợp Combined ACF & PACF
    fig_comb, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    plot_acf(x, lags=effective_lags, zero=False, ax=axes[0])
    plot_pacf(x, lags=effective_lags, zero=False, method=pacf_method, ax=axes[1])
    _decorate_correlation_axis(
        axes[0],
        title="ACF (Nhận diện bậc MA: q, Q)",
        lags=effective_lags,
        bound=bound,
        seasonal_period=seasonal_period,
    )
    _decorate_correlation_axis(
        axes[1],
        title="PACF (Nhận diện bậc AR: p, P)",
        lags=effective_lags,
        bound=bound,
        seasonal_period=seasonal_period,
    )
    fig_comb.suptitle(
        f"Phân tích Tương quan ACF & PACF — {dataset_name.upper()} | Chuỗi Train ({differencing_info}, n={n})",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig_comb.tight_layout()
    if out_dir is not None:
        comb_path = out_dir / f"{file_prefix}_combined_acf_pacf.png"
        fig_comb.savefig(comb_path, dpi=dpi, bbox_inches="tight")
        figure_paths["combined"] = str(comb_path)

    if show_plot:
        plt.show()
    else:
        plt.close(fig_comb)

    log_statistical_analysis(
        dataset_name=dataset_name,
        analysis_type="ACF_PACF_ANALYSIS",
        series_name=f"Train ({differencing_info})",
        details={
            "n_obs": n,
            "nlags": effective_lags,
            "conf_bound_95": round(bound, 4),
            "sig_acf_lags": acf_sig,
            "sig_pacf_lags": pacf_sig,
            "p_candidates": interpretation["p_candidates"],
            "q_candidates": interpretation["q_candidates"],
            "P_candidates": interpretation["P_candidates"],
            "Q_candidates": interpretation["Q_candidates"],
            "combined_plot": figure_paths["combined"],
        },
    )

    return {
        "n": n,
        "bound": bound,
        "acf_values": acf_values,
        "pacf_values": pacf_values,
        "acf_sig": acf_sig,
        "pacf_sig": pacf_sig,
        "full_table": full_table,
        "summary_table": summary_table,
        "interpretation": interpretation,
        "figure_paths": figure_paths,
    }