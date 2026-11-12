# src/lab02_models/candidate_registry.py
"""
MODULE: Statistically Grounded Candidate Model Registry — src/lab02_models/candidate_registry.py
================================================================================
PURPOSE:
    Defines the statistically justified candidate model configurations for each
    LAB02 dataset (AR, MA, ARIMA, SARIMA, Log-SARIMA) based on empirical ADF
    stationarity tests, short-run and seasonal ACF/PACF cut-off/tail-off patterns,
    and Box-Jenkins parsimony principles.

    - Does NOT use blind brute-force grid search.
    - Tailors candidate sets specifically to Dataset 1 (Monthly Retail Sales)
      and Dataset 2 (AirPassengers).
    - Supports 'standard' (core representative candidates) and 'full' (all
      ACF/PACF-identified candidate configurations) execution modes.
================================================================================
"""
from typing import List, Dict, Any
import pandas as pd
from src.lab02.lab02_utils.logging_config import normalize_dataset_key


DATASET1_CANDIDATES: List[Dict[str, Any]] = [
    # --- 1. AR Family: ARIMA(p, d, 0) với d=1 khử xu hướng tuyến tính ---
    {
        "model_name": "AR(1) [ARIMA(1,1,0)]",
        "model_family": "AR",
        "order": (1, 1, 0),
        "seasonal_order": (0, 0, 0, 0),
        "use_log_transform": False,
        "tier": "standard",
        "rationale": "PACF trên chuỗi d=1 cắt cụt mạnh nhất tại lag 1 (PACF[1] = -0.32), kiểm tra quán tính tự hồi quy bậc 1.",
    },
    {
        "model_name": "AR(2) [ARIMA(2,1,0)]",
        "model_family": "AR",
        "order": (2, 1, 0),
        "seasonal_order": (0, 0, 0, 0),
        "use_log_transform": False,
        "tier": "standard",
        "rationale": "PACF trên chuỗi d=1 vượt ngưỡng tin cậy 95% (±0.154) tại cả lag 1 (-0.32) và lag 2 (-0.20) trước khi cắt cụt ở lag 3.",
    },
    # --- 2. MA Family: ARIMA(0, d, q) với d=1 ---
    {
        "model_name": "MA(1) [ARIMA(0,1,1)]",
        "model_family": "MA",
        "order": (0, 1, 1),
        "seasonal_order": (0, 0, 0, 0),
        "use_log_transform": False,
        "tier": "standard",
        "rationale": "Ở độ trễ ngắn hạn trên chuỗi d=1, ACF chỉ có duy nhất 1 gai vượt ngưỡng tại lag 1 (ACF[1] = -0.32) rồi cắt cụt.",
    },
    {
        "model_name": "MA(3) [ARIMA(0,1,3)]",
        "model_family": "MA",
        "order": (0, 1, 3),
        "seasonal_order": (0, 0, 0, 0),
        "use_log_transform": False,
        "tier": "full",
        "rationale": "Khảo sát khả năng hấp thụ dư chấn ngắn hạn theo quý (lag 3) của mô hình trung bình trượt phi mùa vụ.",
    },
    # --- 3. ARIMA Family: ARIMA(p, d, q) kết hợp AR và MA ngắn hạn ---
    {
        "model_name": "ARIMA(1,1,1)",
        "model_family": "ARIMA",
        "order": (1, 1, 1),
        "seasonal_order": (0, 0, 0, 0),
        "use_log_transform": False,
        "tier": "standard",
        "rationale": "Mô hình kết hợp AR(1) và MA(1) cơ bản từ điểm cắt cụt ngắn hạn lag 1 trên cả PACF và ACF (Bài 5 đề LAB02).",
    },
    {
        "model_name": "ARIMA(2,1,1)",
        "model_family": "ARIMA",
        "order": (2, 1, 1),
        "seasonal_order": (0, 0, 0, 0),
        "use_log_transform": False,
        "tier": "standard",
        "rationale": "Mô hình kết hợp AR(2) (PACF vượt ngưỡng ở lag 1, 2) và MA(1) (ACF cắt cụt ở lag 1) theo yêu cầu Bài 5.",
    },
    # --- 4. SARIMA Family: SARIMA(p, d, q)(P, D, Q, 12) sau sai phân đôi d=1, D=1, s=12 (ADF p = 2.28e-14) ---
    {
        "model_name": "SARIMA(0,1,1)x(0,1,0)_12",
        "model_family": "SARIMA",
        "order": (0, 1, 1),
        "seasonal_order": (0, 1, 0, 12),
        "use_log_transform": False,
        "tier": "full",
        "rationale": "Ứng viên SARIMA nhóm Q=0 (vì sau sai phân D=1, lag 12 trên ACF/PACF không còn vượt ngưỡng ±0.160).",
    },
    {
        "model_name": "SARIMA(1,1,1)x(0,1,0)_12",
        "model_family": "SARIMA",
        "order": (1, 1, 1),
        "seasonal_order": (0, 1, 0, 12),
        "use_log_transform": False,
        "tier": "full",
        "rationale": "Ứng viên SARIMA nhóm Q=0 kết hợp ARMA(1,1) ngắn hạn với sai phân mùa vụ D=1.",
    },
    {
        "model_name": "SARIMA(0,1,1)x(0,1,1)_12",
        "model_family": "SARIMA",
        "order": (0, 1, 1),
        "seasonal_order": (0, 1, 1, 12),
        "use_log_transform": False,
        "tier": "standard",
        "rationale": "Mô hình Airline Box-Jenkins kinh điển: kết hợp q=1 (ACF lag 1 = -0.57) và Q=1 để khử nhiễu quá sai phân mùa vụ.",
    },
    {
        "model_name": "SARIMA(1,1,0)x(0,1,1)_12",
        "model_family": "SARIMA",
        "order": (1, 1, 0),
        "seasonal_order": (0, 1, 1, 12),
        "use_log_transform": False,
        "tier": "full",
        "rationale": "Ứng viên SARIMA kết hợp p=1 (PACF lag 1 = -0.57) và Q=1 mùa vụ.",
    },
    {
        "model_name": "SARIMA(1,1,1)x(0,1,1)_12",
        "model_family": "SARIMA",
        "order": (1, 1, 1),
        "seasonal_order": (0, 1, 1, 12),
        "use_log_transform": False,
        "tier": "standard",
        "rationale": "Ứng viên SARIMA kết hợp đồng thời AR(1), MA(1) ngắn hạn và SMA(1) mùa vụ.",
    },
    {
        "model_name": "SARIMA(0,1,3)x(0,1,1)_12",
        "model_family": "SARIMA",
        "order": (0, 1, 3),
        "seasonal_order": (0, 1, 1, 12),
        "use_log_transform": False,
        "tier": "full",
        "rationale": "Ứng viên SARIMA bắt thêm gai vượt ngưỡng tại lag 3 trên ACF của chuỗi sai phân đôi (ACF[3] = 0.18 > 0.160).",
    },
    {
        "model_name": "SARIMA(2,1,1)x(0,1,1)_12",
        "model_family": "SARIMA",
        "order": (2, 1, 1),
        "seasonal_order": (0, 1, 1, 12),
        "use_log_transform": False,
        "tier": "full",
        "rationale": "Ứng viên SARIMA bắt cả 2 độ trễ PACF ngắn hạn (PACF[1] = -0.57, PACF[2] = -0.42) cùng Q=1.",
    },
    {
        "model_name": "SARIMA(2,1,3)x(0,1,1)_12",
        "model_family": "SARIMA",
        "order": (2, 1, 3),
        "seasonal_order": (0, 1, 1, 12),
        "use_log_transform": False,
        "tier": "standard",
        "rationale": "Ứng viên SARIMA bậc cao nhất từ tập ứng viên ACF/PACF (p=2, q=3, Q=1), dùng để kiểm tra đánh đổi giữa AIC và độ phức tạp.",
    },
    # --- 5. Log-SARIMA Family: Ổn định phương sai mùa vụ nhân ---
    {
        "model_name": "Log-SARIMA(0,1,1)x(0,1,1)_12",
        "model_family": "Log-SARIMA",
        "order": (0, 1, 1),
        "seasonal_order": (0, 1, 1, 12),
        "use_log_transform": True,
        "tier": "standard",
        "rationale": "Mô hình Box-Jenkins Airline cổ điển trên chuỗi logarit kèm biến đổi ngược exp() về thang đo gốc.",
    },
    {
        "model_name": "Log-SARIMA(1,1,1)x(0,1,1)_12",
        "model_family": "Log-SARIMA",
        "order": (1, 1, 1),
        "seasonal_order": (0, 1, 1, 12),
        "use_log_transform": True,
        "tier": "full",
        "rationale": "Mô hình Log-SARIMA kết hợp AR(1), MA(1) ngắn hạn và SMA(1) mùa vụ trên chuỗi ln(Y_t).",
    },
    {
        "model_name": "Log-SARIMA(2,1,3)x(0,1,1)_12",
        "model_family": "Log-SARIMA",
        "order": (2, 1, 3),
        "seasonal_order": (0, 1, 1, 12),
        "use_log_transform": True,
        "tier": "standard",
        "rationale": "Cấu hình bậc cao tối ưu trên không gian logarit: đạt White Noise (Ljung-Box p=0.6296) và kỷ lục RMSE thấp nhất.",
    },
]


DATASET2_CANDIDATES: List[Dict[str, Any]] = [
    # --- 1. AR Family: ARIMA(p, d, 0) với d=1 ---
    {
        "model_name": "AR(1) [ARIMA(1,1,0)]",
        "model_family": "AR",
        "order": (1, 1, 0),
        "seasonal_order": (0, 0, 0, 0),
        "use_log_transform": False,
        "tier": "standard",
        "rationale": "PACF trên chuỗi d=1 của AirPassengers vượt ngưỡng (±0.184) tại lag 1 (PACF[1] = 0.27).",
    },
    {
        "model_name": "AR(2) [ARIMA(2,1,0)]",
        "model_family": "AR",
        "order": (2, 1, 0),
        "seasonal_order": (0, 0, 0, 0),
        "use_log_transform": False,
        "tier": "standard",
        "rationale": "PACF trên chuỗi d=1 vượt ngưỡng tại lag 1 (0.27) và lag 2 (-0.22) rồi cắt cụt tại lag 3 (-0.02).",
    },
    {
        "model_name": "AR(4) [ARIMA(4,1,0)]",
        "model_family": "AR",
        "order": (4, 1, 0),
        "seasonal_order": (0, 0, 0, 0),
        "use_log_transform": False,
        "tier": "full",
        "rationale": "PACF trên chuỗi d=1 có gai âm vượt ngưỡng tại lag 4 (PACF[4] = -0.29).",
    },
    # --- 2. MA Family: ARIMA(0, d, q) với d=1 ---
    {
        "model_name": "MA(1) [ARIMA(0,1,1)]",
        "model_family": "MA",
        "order": (0, 1, 1),
        "seasonal_order": (0, 0, 0, 0),
        "use_log_transform": False,
        "tier": "standard",
        "rationale": "ACF trên chuỗi d=1 có gai dương vượt ngưỡng tại lag 1 (ACF[1] = 0.27) trước khi giảm về trong ngưỡng ở lag 2, 3.",
    },
    # --- 3. ARIMA Family: ARIMA(p, d, q) ---
    {
        "model_name": "ARIMA(1,1,1)",
        "model_family": "ARIMA",
        "order": (1, 1, 1),
        "seasonal_order": (0, 0, 0, 0),
        "use_log_transform": False,
        "tier": "standard",
        "rationale": "Ứng viên ARIMA(1,1,1) cơ bản kết hợp p=1 và q=1 trên chuỗi d=1 (Bài 5 đề LAB02).",
    },
    {
        "model_name": "ARIMA(2,1,1)",
        "model_family": "ARIMA",
        "order": (2, 1, 1),
        "seasonal_order": (0, 0, 0, 0),
        "use_log_transform": False,
        "tier": "standard",
        "rationale": "Ứng viên ARIMA(2,1,1) kết hợp p=2 (PACF lag 1, 2) và q=1 (ACF lag 1) trên chuỗi d=1.",
    },
    # --- 4. SARIMA Family trên thang đo gốc: SARIMA(p, 1, q)(0, 1, Q, 12) (ADF d=1,D=1 p = 1.38e-22) ---
    {
        "model_name": "SARIMA(1,1,1)x(0,1,0)_12",
        "model_family": "SARIMA",
        "order": (1, 1, 1),
        "seasonal_order": (0, 1, 0, 12),
        "use_log_transform": False,
        "tier": "full",
        "rationale": "Nhóm ưu tiên Q=0 từ quan sát ACF/PACF chuỗi d=1, D=1 không còn gai vượt ngưỡng ±0.194 tại lag 12.",
    },
    {
        "model_name": "SARIMA(0,1,3)x(0,1,0)_12",
        "model_family": "SARIMA",
        "order": (0, 1, 3),
        "seasonal_order": (0, 1, 0, 12),
        "use_log_transform": False,
        "tier": "full",
        "rationale": "Nhóm Q=0 mở rộng q=3 do ACF chuỗi d=1, D=1 vượt ngưỡng tại lag 1 (-0.20) và lag 3 (-0.21).",
    },
    {
        "model_name": "SARIMA(0,1,1)x(0,1,1)_12",
        "model_family": "SARIMA",
        "order": (0, 1, 1),
        "seasonal_order": (0, 1, 1, 12),
        "use_log_transform": False,
        "tier": "standard",
        "rationale": "Ứng viên SARIMA Airline Model cổ điển với Q=1 dự phòng quá sai phân mùa vụ trên thang đo gốc.",
    },
    {
        "model_name": "SARIMA(1,1,0)x(0,1,1)_12",
        "model_family": "SARIMA",
        "order": (1, 1, 0),
        "seasonal_order": (0, 1, 1, 12),
        "use_log_transform": False,
        "tier": "standard",
        "rationale": "Ứng viên SARIMA kết hợp PACF cắt cụt ở lag 1 (p=1) và MA mùa vụ bậc 1 (Q=1).",
    },
    {
        "model_name": "SARIMA(1,1,1)x(0,1,1)_12",
        "model_family": "SARIMA",
        "order": (1, 1, 1),
        "seasonal_order": (0, 1, 1, 12),
        "use_log_transform": False,
        "tier": "standard",
        "rationale": "Ứng viên SARIMA kết hợp đồng thời p=1, q=1 và Q=1 trên thang đo gốc.",
    },
    {
        "model_name": "SARIMA(1,1,3)x(0,1,1)_12",
        "model_family": "SARIMA",
        "order": (1, 1, 3),
        "seasonal_order": (0, 1, 1, 12),
        "use_log_transform": False,
        "tier": "standard",
        "rationale": "Ứng viên SARIMA bắt đầy đủ PACF lag 1 (p=1), ACF lag 3 (q=3) và MA mùa vụ (Q=1) trên thang đo gốc.",
    },
    # --- 5. Log-SARIMA Family: Ổn định phương sai hình phễu (Multiplicative Seasonality) ---
    {
        "model_name": "Log-SARIMA(0,1,1)x(0,1,1)_12",
        "model_family": "Log-SARIMA",
        "order": (0, 1, 1),
        "seasonal_order": (0, 1, 1, 12),
        "use_log_transform": True,
        "tier": "full",
        "rationale": "Mô hình Airline Box-Jenkins trên chuỗi biến đổi ln(Y_t) kèm biến đổi ngược exp() về thang đo hành khách gốc.",
    },
    {
        "model_name": "Log-SARIMA(1,1,0)x(0,1,1)_12",
        "model_family": "Log-SARIMA",
        "order": (1, 1, 0),
        "seasonal_order": (0, 1, 1, 12),
        "use_log_transform": True,
        "tier": "standard",
        "rationale": "Mô hình Log-SARIMA tối giản tham số (p=1, Q=1) khử hoàn toàn phương sai hình phễu của AirPassengers.",
    },
    {
        "model_name": "Log-SARIMA(1,1,3)x(0,1,1)_12",
        "model_family": "Log-SARIMA",
        "order": (1, 1, 3),
        "seasonal_order": (0, 1, 1, 12),
        "use_log_transform": True,
        "tier": "standard",
        "rationale": "Mô hình Log-SARIMA mở rộng bậc q=3 trên không gian ln(Y_t) để đối chiếu với cấu hình tối giản (1,1,0)x(0,1,1)_12.",
    },
]


def get_candidate_models(
    dataset_name: str,
    mode: str = "full",
    train_series: Any = None,
    seasonal_period: int = 12,
) -> List[Dict[str, Any]]:
    if str(mode).lower() == "auto" and train_series is not None:
        auto_pack = auto_identify_candidates_from_data(
            train_series=train_series,
            dataset_name=dataset_name,
            seasonal_period=seasonal_period,
        )
        return auto_pack["candidates"]
    """
    Purpose:
        Retrieve the statistically grounded list of candidate model configurations
        for the specified dataset ('dataset1' or 'dataset2') and mode ('standard' or 'full').

    Parameters:
        dataset_name (str): Target dataset identifier.
        mode (str): 'standard' (core subset) or 'full' (complete candidate registry).

    Returns:
        List[Dict[str, Any]]: List of candidate configuration dictionaries.
    """
    ds_key = normalize_dataset_key(dataset_name)
    base_list = DATASET1_CANDIDATES if ds_key == "dataset1" else DATASET2_CANDIDATES
    if str(mode).lower() == "standard":
        return [dict(c) for c in base_list if c.get("tier") == "standard"]
    return [dict(c) for c in base_list]


def _detect_differencing_orders(series: pd.Series, label: str, s: int, dataset_name: str) -> Dict[str, Any]:
    from src.lab02.lab02_utils.stats import perform_adf_test
    from src.lab02.lab02_utils.acf_pacf_analysis import analyze_acf_pacf

    adf_raw = perform_adf_test(series, f"{dataset_name} [{label}] (Train Raw)", return_dict=True)
    d = 1 if float(adf_raw.get("p_value", 1.0)) >= 0.05 else 0

    diff_1 = series.diff(1).dropna() if d == 1 else series
    adf_d1 = perform_adf_test(diff_1, f"{dataset_name} [{label}] (Diff d={d})", return_dict=True)

    D = 0
    acf_s_val = 0.0
    if s > 1 and len(diff_1) > 2 * s:
        acf_d1_res = analyze_acf_pacf(
            series=diff_1,
            dataset_name=dataset_name,
            nlags=min(2 * s, len(diff_1) // 2 - 1),
            differencing_info=f"{label} d={d}",
            seasonal_period=s,
            show_plot=False,
        )
        acf_vals = acf_d1_res["acf_values"]
        acf_s_val = float(acf_vals[s]) if len(acf_vals) > s else 0.0
        bound = float(acf_d1_res["bound"])
        # ADF không phát hiện được mùa vụ xác định (deterministic seasonality),
        # nên bắt buộc kết hợp kiểm tra độ lớn ACF tại lag s (ACF[s] vượt ngưỡng mạnh).
        if float(adf_d1.get("p_value", 1.0)) >= 0.05 or abs(acf_s_val) > max(bound, 0.30):
            D = 1

    return {
        "d": d,
        "D": D,
        "adf_raw_p": float(adf_raw.get("p_value", 1.0)),
        "adf_d1_p": float(adf_d1.get("p_value", 1.0)),
        "acf_s_on_diff1": acf_s_val,
        "diff_1": diff_1,
    }


def auto_identify_candidates_from_data(
    train_series: Any,
    dataset_name: str = "dataset1",
    seasonal_period: int = 12,
    include_log: bool = True,
) -> Dict[str, Any]:
    """
    Purpose:
        Fully automated, data-driven candidate model identification pipeline following
        the Box-Jenkins methodology independently on BOTH Original Space (Y_t) and
        Log Space (ln(Y_t)):
        1. ADF Unit Root Test -> determines regular differencing order d and seasonal D.
        2. Statistical ACF/PACF cutoff analysis -> derives p, q, P, Q candidate orders
           independently for Original Space and Log Space.
        3. Exhaustively generates all valid combinations within the ACF/PACF-bounded
           candidate space for AR, MA, ARIMA, SARIMA, and Log-SARIMA.

    Parameters:
        train_series: Chronological Training series (never touches Test data).
        dataset_name (str): Identifier for logging and reports.
        seasonal_period (int): Seasonal cycle length s (default 12 for monthly data).
        include_log (bool): Whether to independently analyze ln(Y_t) and generate Log-SARIMA space.

    Returns:
        Dict[str, Any]: {
            'orders_detected': {...},
            'candidates': List[Dict[str, Any]],
            'rationale_summary': str
        }
    """
    import numpy as np
    import pandas as pd
    from src.lab02.lab02_utils.stats import perform_adf_test
    from src.lab02.lab02_utils.acf_pacf_analysis import analyze_acf_pacf

    s = int(seasonal_period)
    y = pd.Series(train_series).astype(float).dropna()

    # =========================================================================
    # LUỒNG 1: PHÂN TÍCH ĐỘC LẬP TRÊN THANG ĐO GỐC (ORIGINAL SPACE: Y_t)
    # =========================================================================
    orig_diff_info = _detect_differencing_orders(y, "Original", s, dataset_name)
    d = orig_diff_info["d"]
    D = orig_diff_info["D"]
    diff_1 = orig_diff_info["diff_1"]
    adf_raw = {"p_value": orig_diff_info["adf_raw_p"]}
    adf_d1 = {"p_value": orig_diff_info["adf_d1_p"]}

    diff_final = diff_1.diff(s).dropna() if D == 1 else diff_1
    acf_pacf_res = analyze_acf_pacf(
        series=diff_final,
        dataset_name=dataset_name,
        nlags=min(36, len(diff_final) // 2 - 1),
        differencing_info=f"Original d={d}, D={D}, s={s}",
        seasonal_period=s,
        show_plot=False,
    )
    interp = acf_pacf_res["interpretation"]
    p_cands = interp.get("p_candidates", [0, 1])
    q_cands = interp.get("q_candidates", [0, 1])
    P_cands = interp.get("P_candidates", [0])
    Q_cands = interp.get("Q_candidates", [0, 1])

    candidates: List[Dict[str, Any]] = []

    # 1. Họ AR (phi mùa vụ)
    for p_val in [p for p in p_cands if p > 0]:
        candidates.append({
            "model_name": f"AR({p_val}) [ARIMA({p_val},{d},0)]",
            "model_family": "AR",
            "order": (p_val, d, 0),
            "seasonal_order": (0, 0, 0, 0),
            "use_log_transform": False,
            "tier": "standard",
            "rationale": f"Tự động phát hiện từ PACF thang đo gốc vượt ngưỡng tại lag {p_val} (d={d}).",
        })

    # 2. Họ MA (phi mùa vụ)
    for q_val in [q for q in q_cands if q > 0]:
        candidates.append({
            "model_name": f"MA({q_val}) [ARIMA(0,{d},{q_val})]",
            "model_family": "MA",
            "order": (0, d, q_val),
            "seasonal_order": (0, 0, 0, 0),
            "use_log_transform": False,
            "tier": "standard",
            "rationale": f"Tự động phát hiện từ ACF thang đo gốc vượt ngưỡng tại lag {q_val} (d={d}).",
        })

    # 3. Họ ARIMA kết hợp đầy đủ trong không gian (p > 0, q > 0)
    for p_val in [p for p in p_cands if p > 0]:
        for q_val in [q for q in q_cands if q > 0]:
            tier = "standard" if (p_val <= 2 and q_val == 1) else "full"
            candidates.append({
                "model_name": f"ARIMA({p_val},{d},{q_val})",
                "model_family": "ARIMA",
                "order": (p_val, d, q_val),
                "seasonal_order": (0, 0, 0, 0),
                "use_log_transform": False,
                "tier": tier,
                "rationale": f"Kết hợp tự động AR({p_val}) và MA({q_val}) trên thang đo gốc (d={d}).",
            })

    # 4. Họ SARIMA: Quét toàn bộ tổ hợp hợp lệ trong không gian (p x q x P x Q) thang đo gốc
    sarima_count = 0
    for p_val in p_cands:
        for q_val in q_cands:
            for P_val in P_cands:
                for Q_val in Q_cands:
                    if p_val == 0 and q_val == 0 and P_val == 0 and Q_val == 0:
                        continue
                    m_name = f"SARIMA({p_val},{d},{q_val})x({P_val},{D},{Q_val})_{s}"
                    tier = "standard" if (p_val <= 1 and q_val <= 1 and P_val <= 1 and Q_val <= 1) else "full"
                    candidates.append({
                        "model_name": m_name,
                        "model_family": "SARIMA",
                        "order": (p_val, d, q_val),
                        "seasonal_order": (P_val, D, Q_val, s),
                        "use_log_transform": False,
                        "tier": tier,
                        "rationale": f"Quét tự động trên không gian gốc (p={p_val}, q={q_val}, P={P_val}, Q={Q_val}) sau sai phân d={d}, D={D}.",
                    })
                    sarima_count += 1

    # =========================================================================
    # LUỒNG 2: PHÂN TÍCH ĐỘC LẬP TRÊN KHÔNG GIAN LOGARIT (LOG SPACE: ln(Y_t))
    # =========================================================================
    log_orders_info = None
    log_sarima_count = 0
    if include_log and (y > 0).all():
        y_log = np.log(y)
        log_diff_info = _detect_differencing_orders(y_log, "Log Space", s, dataset_name)
        d_log = log_diff_info["d"]
        D_log = log_diff_info["D"]
        diff_log_1 = log_diff_info["diff_1"]

        diff_log_final = diff_log_1.diff(s).dropna() if D_log == 1 else diff_log_1
        acf_pacf_log = analyze_acf_pacf(
            series=diff_log_final,
            dataset_name=dataset_name,
            nlags=min(36, len(diff_log_final) // 2 - 1),
            differencing_info=f"Log d={d_log}, D={D_log}, s={s}",
            seasonal_period=s,
            show_plot=False,
        )
        interp_log = acf_pacf_log["interpretation"]
        p_log = interp_log.get("p_candidates", [0, 1])
        q_log = interp_log.get("q_candidates", [0, 1])
        P_log = interp_log.get("P_candidates", [0])
        Q_log = interp_log.get("Q_candidates", [0, 1])

        log_orders_info = {
            "d": d_log,
            "D": D_log,
            "p_candidates": p_log,
            "q_candidates": q_log,
            "P_candidates": P_log,
            "Q_candidates": Q_log,
        }

        for p_val in p_log:
            for q_val in q_log:
                for P_val in P_log:
                    for Q_val in Q_log:
                        if p_val == 0 and q_val == 0 and P_val == 0 and Q_val == 0:
                            continue
                        m_name = f"Log-SARIMA({p_val},{d_log},{q_val})x({P_val},{D_log},{Q_val})_{s}"
                        tier = "standard" if (p_val <= 1 and q_val <= 1 and P_val <= 1 and Q_val <= 1) else "full"
                        candidates.append({
                            "model_name": m_name,
                            "model_family": "Log-SARIMA",
                            "order": (p_val, d_log, q_val),
                            "seasonal_order": (P_val, D_log, Q_val, s),
                            "use_log_transform": True,
                            "tier": tier,
                            "rationale": f"Quét tự động độc lập trên không gian ln(Y_t) với (p={p_val}, q={q_val}, P={P_val}, Q={Q_val}) sau sai phân d={d_log}, D={D_log}.",
                        })
                        log_sarima_count += 1

    summary_text = (
        f"=== KẾT QUẢ ĐỊNH DANH MÔ HÌNH TỰ ĐỘNG ĐA KHÔNG GIAN ({dataset_name.upper()}) ===\n"
        f"[1] Không gian Gốc (Original Space Y_t):\n"
        f"  • Bậc sai phân : d={d} (ADF raw p={adf_raw.get('p_value', 1.0):.4f}), D={D} (ADF diff1 p={adf_d1.get('p_value', 1.0):.4f}), s={s}\n"
        f"  • Không gian ACF/PACF : p={p_cands}, q={q_cands}, P={P_cands}, Q={Q_cands} -> {sarima_count} mô hình SARIMA\n"
    )
    if log_orders_info is not None:
        summary_text += (
            f"[2] Không gian Logarit (Log Space ln(Y_t)):\n"
            f"  • Bậc sai phân : d={log_orders_info['d']}, D={log_orders_info['D']}, s={s}\n"
            f"  • Không gian ACF/PACF : p={log_orders_info['p_candidates']}, q={log_orders_info['q_candidates']}, "
            f"P={log_orders_info['P_candidates']}, Q={log_orders_info['Q_candidates']} -> {log_sarima_count} mô hình Log-SARIMA\n"
        )
    summary_text += f"  • Tổng cộng toàn pipeline tự động sinh: {len(candidates)} mô hình ứng viên.\n"

    return {
        "orders_detected": {
            "original": {
                "d": d,
                "D": D,
                "seasonal_period": s,
                "p_candidates": p_cands,
                "q_candidates": q_cands,
                "P_candidates": P_cands,
                "Q_candidates": Q_cands,
            },
            "log": log_orders_info,
        },
        "candidates": candidates,
        "rationale_summary": summary_text,
    }
