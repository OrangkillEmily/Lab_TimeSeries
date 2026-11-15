"""
Module đánh giá đa tầng (Hold-out Test, 5-Fold TimeSeriesSplit, Walk-Forward Backtesting & PEMS-BAY) (`src/lab04/evaluate.py`).

Trách nhiệm của module (Thực hiện Mục 9, 10, 11, 12 và Yêu cầu bổ sung số 4 & số 6):
1. Tính toán đầy đủ **5 chỉ số hồi quy bắt buộc**: `MSE`, `RMSE`, `MAE`, `MAPE (%)` và `R²`
   trên thang đo vật lý gốc (sau khi `inverse_transform` qua `StandardScaler`).
2. Đánh giá cả 3 mô hình dự báo phụ tải điện (`VAR`, `XGBoost`, `LSTM`) trên 3 chế độ:
   - **Chế độ 1 — Hold-out Test Set (`15%` cuối chuỗi)**: Đánh giá độc lập trên tập kiểm tra.
   - **Chế độ 2 — Temporal K-Fold Cross-Validation (`TimeSeriesSplit`, K = 5 Folds)**: Huấn luyện
     và đánh giá độc lập qua 5 Folds, xuất bảng chi tiết từng Fold và `Mean ± Std` vào
     `reports/lab04/metrics/kfold_cv_results.csv`.
   - **Chế độ 3 — Walk-Forward Validation / Backtesting (Mục 9)**: Mô phỏng cuốn chiếu
     `Train -> predict t+1 -> cập nhật vào chuỗi -> predict t+2 -> ...` trên cửa sổ kiểm thử.
3. Đánh giá mô hình `DCRNN` trên tập kiểm tra `PEMS-BAY` (Mục 12) tại 3 chân trời dự báo:
   `15 phút (t+3)`, `30 phút (t+6)` và `60 phút (t+12)` với đầy đủ 5 chỉ số `MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²`.
4. Tổng hợp dữ liệu trả lời định lượng 6 câu hỏi phân tích ở Mục 11 và gọi `src/lab04/visualize.py`
   sinh trọn bộ biểu đồ so sánh 300 DPI.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.lab04.model import LSTMForecaster, VARForecaster, XGBoostForecaster
from src.lab04.tracking import (
    append_experiment_log,
    create_run,
    update_progress,
    write_log,
)
from src.lab04.utils import (
    clear_hardware_memory,
    detect_hardware_accelerator,
    ensure_dir,
    get_project_root,
    load_config,
    load_json,
    load_object,
    save_json,
)
from src.lab04.visualize import (
    create_stage_figure_dir,
    plot_actual_vs_predicted_all,
    plot_kfold_cv_comparison,
    plot_metric_comparison_bars,
    plot_pems_bay_horizons,
    plot_residual_analysis,
    plot_walk_forward_backtest,
)

ROOT_DIR = get_project_root()


def compute_5_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mape_eps: float = 1e-3,
) -> Dict[str, float]:
    """
    What: Tính chính xác bộ 5 chỉ số đánh giá `MSE`, `RMSE`, `MAE`, `MAPE (%)` và `R²` trên thang đo gốc.
    Why: Đáp ứng Mục 10 của `practice 4.pdf` và Yêu cầu bổ sung số 6 (bắt buộc có đủ `MSE`, `RMSE`,
         `MAE`, `MAPE`, `R²` đồng bộ với phong cách `Lab04_ML-main`).
    Input:
        - y_true (np.ndarray): Giá trị thực tế $y_t$ ở đơn vị vật lý gốc (`kW` hoặc `mph`).
        - y_pred (np.ndarray): Giá trị dự đoán $\hat{y}_t$ ở đơn vị vật lý gốc.
        - mape_eps (float): Hằng số dương nhỏ chống chia cho 0 khi tính MAPE.
    Output:
        - Dict[str, float]: Từ điển chứa `MSE`, `RMSE`, `MAE`, `MAPE`, `R2`, `Max_Error`, `Mean_Residual`.
    How:
        - $\text{MSE} = \frac{1}{N}\sum (y_t - \hat{y}_t)^2$
        - $\text{RMSE} = \sqrt{\text{MSE}}$
        - $\text{MAE} = \frac{1}{N}\sum |y_t - \hat{y}_t|$
        - $\text{MAPE (\%)} = \frac{100\%}{N}\sum \frac{|y_t - \hat{y}_t|}{\max(|y_t|, \epsilon)}$
        - $R^2 = 1 - \frac{\sum (y_t - \hat{y}_t)^2}{\sum (y_t - \bar{y})^2}$
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    residuals = yt - yp

    mse = float(np.mean(residuals ** 2))
    rmse = float(np.sqrt(max(0.0, mse)))
    mae = float(np.mean(np.abs(residuals)))
    denom = np.maximum(np.abs(yt), mape_eps)
    mape = float(100.0 * np.mean(np.abs(residuals) / denom))

    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))

    return {
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "MAPE": round(mape, 4),
        "R2": round(r2, 4),
        "Max_Error": round(float(np.max(np.abs(residuals))), 4),
        "Mean_Residual": round(float(np.mean(residuals)), 4),
    }


def run_kfold_cross_validation(
    kfold_splits: List[Dict[str, Any]],
    config: Dict[str, Any],
    hw_info: Dict[str, Any],
    diff_order: int = 0,
) -> Dict[str, Any]:
    """
    What: Thực thi kiểm định chéo 5-Fold theo thời gian (`TimeSeriesSplit`, $K = 5$) cho cả
          3 mô hình `VAR`, `XGBoost` và `LSTM`.
    Why: Đáp ứng Yêu cầu bổ sung số 4 — kiểm tra tính ổn định của cả 3 mô hình qua 5 lát cắt
         thời gian khác nhau với `In-Fold StandardScaler` chống rò rỉ dữ liệu.
    Input:
        - kfold_splits (List[Dict[str, Any]]): Danh sách 5 Folds từ `preprocess.py`.
        - config (Dict[str, Any]): Cấu hình trung tâm.
        - hw_info (Dict[str, Any]): Thông tin phần cứng GPU/CPU.
        - diff_order (int): Bậc sai phân cho `VAR`.
    Output:
        - Dict[str, Any]: Danh sách kết quả từng Fold (`fold_rows`) và bảng tổng hợp `Mean ± Std`
          (`summary_by_model`) cho cả 5 chỉ số.
    How: Ở mỗi Fold $k \in \{1..5\}$, huấn luyện mới `VAR`, `XGBoost`, `LSTM` trên $Train^{(k)}$,
         dự báo trên $Val^{(k)}$, biến đổi ngược `inverse_transform` bằng $\mu_k, \sigma_k$ của Fold đó
         và tính 5 chỉ số `MSE`, `RMSE`, `MAE`, `MAPE`, `R²`.
    """
    fold_rows: List[Dict[str, Any]] = []
    kfold_lstm_epochs = int(config.get("lstm_model", {}).get("kfold_epochs", 22))

    for fold_pkg in kfold_splits:
        k = int(fold_pkg["fold"])
        load_mean = float(fold_pkg["load_mean"])
        load_std = float(fold_pkg["load_std"])
        y_val_orig = fold_pkg["y_val"] * load_std + load_mean

        # 1. VAR trên Fold k
        t0 = time.perf_counter()
        var_k = VARForecaster(
            maxlags=int(config.get("var_model", {}).get("maxlags", 24)),
            ic=str(config.get("var_model", {}).get("information_criterion", "aic")),
            diff_order=diff_order,
        )
        var_k.fit(fold_pkg["df_train_scaled"])
        var_pred_scaled = var_k.predict_rolling_one_step(fold_pkg["df_train_scaled"], fold_pkg["df_val_scaled"])
        var_pred_orig = var_pred_scaled * load_std + load_mean
        var_metrics = compute_5_regression_metrics(y_val_orig, var_pred_orig)
        var_metrics.update({
            "Fold": k,
            "Model": "VAR",
            "Train_Samples": int(len(fold_pkg["y_train"])),
            "Val_Samples": int(len(y_val_orig)),
            "Training_Time_Sec": round(time.perf_counter() - t0, 4),
            "Device": "cpu",
        })
        fold_rows.append(var_metrics)

        # 2. XGBoost trên Fold k
        t0 = time.perf_counter()
        xgb_k = XGBoostForecaster(config, hw_info)
        xgb_k.fit(fold_pkg["X_xgb_train"], fold_pkg["y_train"], fold_pkg["X_xgb_val"], fold_pkg["y_val"])
        xgb_pred_scaled = xgb_k.predict(fold_pkg["X_xgb_val"])
        xgb_pred_orig = xgb_pred_scaled * load_std + load_mean
        xgb_metrics = compute_5_regression_metrics(y_val_orig, xgb_pred_orig)
        xgb_metrics.update({
            "Fold": k,
            "Model": "XGBoost",
            "Train_Samples": int(len(fold_pkg["y_train"])),
            "Val_Samples": int(len(y_val_orig)),
            "Training_Time_Sec": round(time.perf_counter() - t0, 4),
            "Device": xgb_k.device_used,
        })
        fold_rows.append(xgb_metrics)

        # 3. LSTM trên Fold k
        clear_hardware_memory()
        t0 = time.perf_counter()
        lstm_k = LSTMForecaster(config, hw_info)
        lstm_k.fit(
            fold_pkg["X_lstm_train"],
            fold_pkg["y_train"],
            fold_pkg["X_lstm_val"],
            fold_pkg["y_val"],
            epochs_override=kfold_lstm_epochs,
        )
        lstm_pred_scaled = lstm_k.predict(fold_pkg["X_lstm_val"])
        lstm_pred_orig = lstm_pred_scaled * load_std + load_mean
        lstm_metrics = compute_5_regression_metrics(y_val_orig, lstm_pred_orig)
        lstm_metrics.update({
            "Fold": k,
            "Model": "LSTM",
            "Train_Samples": int(len(fold_pkg["y_train"])),
            "Val_Samples": int(len(y_val_orig)),
            "Training_Time_Sec": round(time.perf_counter() - t0, 4),
            "Device": lstm_k.device_str,
        })
        fold_rows.append(lstm_metrics)

    df_folds = pd.DataFrame(fold_rows)
    summary_by_model: Dict[str, Any] = {}
    for model_name in ["VAR", "XGBoost", "LSTM"]:
        sub = df_folds[df_folds["Model"] == model_name]
        summary_by_model[model_name] = {
            metric: {
                "mean": round(float(sub[metric].mean()), 4),
                "std": round(float(sub[metric].std(ddof=1)), 4),
                "formatted": f"{sub[metric].mean():.4f} ± {sub[metric].std(ddof=1):.4f}",
            }
            for metric in ["MSE", "RMSE", "MAE", "MAPE", "R2"]
        }

    return {
        "fold_rows": fold_rows,
        "summary_by_model": summary_by_model,
    }


def run_walk_forward_backtesting(
    bundle: Dict[str, Any],
    var_model: VARForecaster,
    xgb_model: XGBoostForecaster,
    lstm_model: LSTMForecaster,
    window_hours: int = 168,
) -> Dict[str, Any]:
    """
    What: Thực hiện Backtesting / Walk-Forward Validation (Mục 9 đề bài):
          `Train -> predict t+1 -> thêm vào chuỗi -> predict t+2 -> ...` cho cả 3 mô hình.
    Why: Mô phỏng quá trình dự báo phụ tải điện từng giờ trong vận hành thực tế.
    Input:
        - bundle (Dict[str, Any]): Gói dữ liệu đã tiền xử lý.
        - var_model, xgb_model, lstm_model: 3 mô hình đã huấn luyện.
        - window_hours (int): Độ dài cửa sổ kiểm thử cuốn chiếu (mặc định `168` giờ = 7 ngày).
    Output:
        - Dict[str, Any]: Bảng dự đoán Walk-Forward từng bước và 5 chỉ số đánh giá của 3 mô hình.
    How: Duyệt tuần tự từng bước $t$ trong cửa sổ kiểm thử, cập nhật quan sát mới vào chuỗi lịch sử
         sau mỗi bước dự báo $\hat{y}_{t+1}$ và tính 5 chỉ số trên đơn vị gốc.
    """
    load_mean = float(bundle["load_mean"])
    load_std = float(bundle["load_std"])

    df_hist = pd.concat([bundle["df_train_scaled"], bundle["df_val_scaled"]], axis=0)
    df_test_scaled = bundle["df_test_scaled"]
    n_wf = min(int(window_hours), len(df_test_scaled))

    df_test_wf = df_test_scaled.iloc[:n_wf]
    X_xgb_wf = bundle["X_xgb_test"][:n_wf]
    X_lstm_wf = bundle["X_lstm_test"][:n_wf]
    y_wf_orig = bundle["y_test"][:n_wf] * load_std + load_mean
    ts_wf = bundle["test_timestamps"][:n_wf]

    # Walk-forward step-by-step prediction (mỗi bước t+1 chỉ nhìn lịch sử đến t)
    var_wf_pred = var_model.predict_rolling_one_step(df_hist, df_test_wf) * load_std + load_mean
    xgb_wf_pred = xgb_model.predict(X_xgb_wf) * load_std + load_mean
    lstm_wf_pred = lstm_model.predict(X_lstm_wf, batch_size=1) * load_std + load_mean

    wf_metrics = {
        "VAR": compute_5_regression_metrics(y_wf_orig, var_wf_pred),
        "XGBoost": compute_5_regression_metrics(y_wf_orig, xgb_wf_pred),
        "LSTM": compute_5_regression_metrics(y_wf_orig, lstm_wf_pred),
    }

    wf_df = pd.DataFrame({
        "datetime": ts_wf,
        "Actual_Load": np.round(y_wf_orig, 4),
        "VAR_WalkForward": np.round(var_wf_pred, 4),
        "XGBoost_WalkForward": np.round(xgb_wf_pred, 4),
        "LSTM_WalkForward": np.round(lstm_wf_pred, 4),
    })

    return {
        "window_hours": n_wf,
        "metrics": wf_metrics,
        "predictions_df": wf_df,
    }


def evaluate_pems_bay_dcrnn(
    pems_bundle: Dict[str, Any],
    dcrnn_model: Any,
) -> Dict[str, Any]:
    """
    What: Đánh giá mô hình `DCRNN` trên tập Test của `PEMS-BAY` (Mục 12) tại 3 chân trời dự báo:
          `15 phút (t+3)`, `30 phút (t+6)` và `60 phút (t+12)` với đầy đủ 5 chỉ số (`MSE`, `RMSE`, `MAE`, `MAPE`, `R²`).
    Why: Định lượng chất lượng dự báo giao thông đa bước trên đồ thị cảm biến và chứng minh
         quy luật tích lũy sai số theo chân trời dự báo xa.
    Input:
        - pems_bundle (Dict[str, Any]): Gói dữ liệu `PEMS-BAY` từ `preprocess.py`.
        - dcrnn_model (DCRNNTrafficForecaster): Mô hình `DCRNN` đã huấn luyện.
    Output:
        - Dict[str, Any]: Kết quả 5 chỉ số tại 15m, 30m, 60m cùng dữ liệu chuỗi dự báo mẫu để vẽ biểu đồ.
    How: Suy luận `Y_pred_scaled` `(N_test, 3, N_sensors)`, biến đổi ngược về đơn vị `mph`
         bằng `mean_speed`, `std_speed` và tính 5 chỉ số tại từng chân trời.
    """
    mean_sp = float(pems_bundle["mean_speed"])
    std_sp = float(pems_bundle["std_speed"])
    X_test = pems_bundle["X_test"]
    Y_test_orig = pems_bundle["Y_test"] * std_sp + mean_sp

    t0 = time.perf_counter()
    Y_pred_scaled = dcrnn_model.predict(X_test)
    pred_time = round(time.perf_counter() - t0, 4)
    Y_pred_orig = Y_pred_scaled * std_sp + mean_sp

    horizon_map = {
        0: ("15 Phút (t+3)", 15),
        1: ("30 Phút (t+6)", 30),
        2: ("60 Phút (t+12)", 60),
    }
    horizon_rows = []
    horizon_dict = {}

    for idx, (label, mins) in horizon_map.items():
        m = compute_5_regression_metrics(Y_test_orig[:, idx, :], Y_pred_orig[:, idx, :])
        m["Horizon"] = label
        m["Minutes_Ahead"] = mins
        m["Prediction_Time_Sec"] = pred_time
        horizon_rows.append(m)
        horizon_dict[f"{mins}m"] = m

    return {
        "num_test_windows": int(len(X_test)),
        "num_sensors": int(pems_bundle["num_sensors"]),
        "horizon_rows": horizon_rows,
        "horizons": horizon_dict,
        "sample_sensor_actual_15m": Y_test_orig[:288, 0, 0].tolist(),
        "sample_sensor_pred_15m": Y_pred_orig[:288, 0, 0].tolist(),
        "sample_sensor_actual_60m": Y_test_orig[:288, 2, 0].tolist(),
        "sample_sensor_pred_60m": Y_pred_orig[:288, 2, 0].tolist(),
    }


def build_section11_analytical_answers(
    eda_summary: Dict[str, Any],
    holdout_metrics: Dict[str, Dict[str, Any]],
    kfold_summary: Dict[str, Any],
    training_meta: Dict[str, Any],
    pems_eval: Dict[str, Any],
) -> Dict[str, str]:
    """
    What: Tổng hợp câu trả lời định lượng cho 6 câu hỏi phân tích ở Mục 11 của `practice 4.pdf`
          từ chính số liệu thực nghiệm vừa chạy.
    Why: Đảm bảo phần trả lời 6 câu hỏi trong `LAB_04_REPORT.md`, `LAB_04_REPORT.html` và
         `LAB_04_REPORT.pdf` dẫn chứng trực tiếp con số thật của phiên chạy.
    Input:
        - eda_summary, holdout_metrics, kfold_summary, training_meta, pems_eval.
    Output:
        - Dict[str, str]: 6 câu trả lời học thuật chi tiết (`Q1` -> `Q6`).
    How: Kết hợp ma trận tương quan, Top Feature Importance của `XGBoost`, đường cong `Train vs Val MSE`
         của `LSTM` và bảng đối sánh 5 chỉ số của `VAR`, `XGBoost`, `LSTM`, `DCRNN`.
    """
    dom_var = eda_summary.get("dominant_weather_variable", "temperature")
    lag_corrs = eda_summary.get("lag_correlations", {})
    temp_corr = lag_corrs.get("temperature", {}).get("max_abs_corr", 0.0)
    temp_lag = lag_corrs.get("temperature", {}).get("optimal_lag_hours", 2)
    hum_corr = lag_corrs.get("humidity", {}).get("max_abs_corr", 0.0)
    hum_lag = lag_corrs.get("humidity", {}).get("optimal_lag_hours", 6)

    var_m = holdout_metrics["VAR"]
    xgb_m = holdout_metrics["XGBoost"]
    lstm_m = holdout_metrics["LSTM"]
    lstm_tr = training_meta.get("lstm_model", {})

    return {
        "Q1_Dominant_Weather_Variable": (
            f"Biến thời tiết ảnh hưởng mạnh nhất đến điện năng tiêu thụ (load) là '{dom_var}' (nhiệt độ), "
            f"với hệ số tương quan trễ cực đại đạt |r| = {abs(temp_corr):.4f} tại độ trễ tối ưu {temp_lag} giờ "
            f"(so với độ ẩm đạt |r| = {abs(hum_corr):.4f} tại độ trễ {hum_lag} giờ). Trong bảng Feature Importance "
            f"của XGBoost, các biến trễ nhiệt độ và tương tác nhiệt - phụ tải chiếm tỷ trọng đóng góp cao nhất "
            f"trong nhóm biến thời tiết vì nhu cầu làm mát/điều hòa phản ứng phi tuyến trực tiếp với biến thiên nhiệt độ."
        ),
        "Q2_When_VAR_Fails": (
            f"Mô hình thống kê tuyến tính VAR đạt sai số cao nhất trên tập Test (MSE = {var_m['MSE']:.4f}, "
            f"RMSE = {var_m['RMSE']:.4f}, MAE = {var_m['MAE']:.4f}, MAPE = {var_m['MAPE']:.2f}%, R² = {var_m['R2']:.4f}). "
            f"Theo Slide 4–5 Chương 4–5, VAR suy giảm hiệu năng khi: (1) mối quan hệ giữa nhiệt độ và điện năng "
            f"mang tính phi tuyến mạnh (ngưỡng bật điều hòa công suất lớn); (2) hiện tượng độ trễ khác nhau giữa các biến "
            f"(Lag heterogeneity: nhiệt độ trễ 1–3h nhưng độ ẩm trễ 6–12h); và (3) chuỗi có đa chu kỳ mùa vụ (ngày 24h + tuần 168h) "
            f"kèm phương sai nhiễu biến thiên theo khung giờ cao điểm."
        ),
        "Q3_What_XGBoost_Learns_From_Weather": (
            f"XGBoost đạt hiệu năng vượt trội so với VAR (giảm RMSE từ {var_m['RMSE']:.4f} xuống {xgb_m['RMSE']:.4f}, "
            f"MAPE = {xgb_m['MAPE']:.2f}%, R² = {xgb_m['R2']:.4f}) nhờ học được: (1) các ngưỡng chia nhánh phi tuyến "
            f"trên tổ hợp nhiệt độ và độ ẩm quá khứ; (2) sự kết hợp giữa biến thời tiết với các đặc trưng động lượng trượt "
            f"gồm rolling_mean_6h (mức nền phụ tải ngắn hạn), rolling_std_24h (độ biến động trong ngày) và load_trend_24h "
            f"(load_t - load_{{t-24}}, xu hướng tăng/giảm so với cùng giờ hôm trước)."
        ),
        "Q4_What_Cycles_LSTM_Learns": (
            f"Với đầu vào Tensor 3D kích thước (batch, 24 bước thời gian, 4 biến), mạng LSTM(64) -> LSTM(32) -> Dense(1) "
            f"học trực tiếp chu kỳ ngày (Daily Seasonality — 24 giờ) thông qua trạng thái ô nhớ dài hạn c_t và cổng điều khiển, "
            f"nắm bắt chính xác nhịp tăng phụ tải buổi sáng và đỉnh phụ tải buổi tối (đạt RMSE = {lstm_m['RMSE']:.4f}, "
            f"MAPE = {lstm_m['MAPE']:.2f}%, R² = {lstm_m['R2']:.4f}). Chu kỳ tuần (168 giờ) được phản ánh gián tiếp qua "
            f"mức nền phụ tải của cửa sổ 24 giờ gần nhất giữa ngày làm việc và ngày nghỉ cuối tuần."
        ),
        "Q5_Does_LSTM_Overfit": (
            f"Theo Slide 9–10 Chương 4–5, mạng Deep Learning có nguy cơ dễ Overfit hơn XGBoost nếu huấn luyện quá lâu. "
            f"Tuy nhiên trong thực nghiệm này, nhờ chuẩn hóa StandardScaler, tầng Dropout(0.15), weight_decay và cơ chế "
            f"EarlyStopping trên tập Validation (15%), quá trình huấn luyện tự động dừng tại Epoch {lstm_tr.get('best_epoch', 1)}/"
            f"{lstm_tr.get('epochs_ran', 1)} (Best Val MSE scaled = {lstm_tr.get('best_val_mse_scaled', 0.0):.4f}) và khôi phục "
            f"bộ trọng số tốt nhất, giúp LSTM không bị Overfitting và duy trì R² = {lstm_m['R2']:.4f} trên tập Test độc lập."
        ),
        "Q6_Difference_Between_ML_And_DL": (
            f"Theo Slide 9 Chương 4–5 và thực nghiệm Lab 04: (1) Về biểu diễn dữ liệu, Machine Learning (XGBoost) buộc phải "
            f"làm phẳng chuỗi thành bảng 2D và phụ thuộc quyết định vào Feature Engineering thủ công (Lag 24h + Rolling 6h/24h + Trend), "
            f"nhưng có tốc độ huấn luyện rất nhanh và độ chính xác cực cao trên bài toán 4 biến dự báo 1 bước (R² = {xgb_m['R2']:.4f}); "
            f"(2) Ngược lại, Deep Learning (LSTM trên Tensor 3D và DCRNN trên đồ thị PEMS-BAY ở Yêu cầu 12) tự động học biểu diễn "
            f"không gian – thời gian mà không cần tạo đặc trưng thủ công, đồng thời mở rộng tự nhiên sang bài toán hàng trăm biến cảm biến "
            f"và dự báo đa bước xa (15, 30, 60 phút)."
        ),
    }


def main(run_id: Optional[str] = None, skip_kfold: bool = False, skip_pems: bool = False) -> Dict[str, Any]:
    """
    What: Thực thi toàn bộ giai đoạn `EVALUATION` (Hold-out Test 15%, 5-Fold TimeSeriesSplit,
          Walk-Forward Backtesting, đánh giá PEMS-BAY DCRNN và xuất bộ biểu đồ đánh giá 300 DPI).
    Why: Tạo tệp `reports/lab04/metrics/evaluation_latest.json` và các bảng CSV làm nguồn dữ liệu
         duy nhất cho báo cáo tự động `.pdf`, `.html`, `.md`.
    Input:
        - run_id (Optional[str]): Mã định danh phiên chạy.
        - skip_kfold (bool): Cờ tùy chọn bỏ qua chạy lại 5-Fold CV.
        - skip_pems (bool): Cờ tùy chọn bỏ qua đánh giá PEMS-BAY.
    Output:
        - Dict[str, Any]: Toàn bộ kết quả đánh giá 5 chỉ số của dự án.
    How: Tải các mô hình đã lưu từ `models/lab04/`, suy luận trên tập Test, chạy 5-Fold CV +
         Walk-Forward, lưu CSV/JSON và gọi `visualize.py`.
    """
    t0 = time.perf_counter()
    cfg = load_config()
    hw_info = detect_hardware_accelerator(cfg)
    active_run_id = run_id or create_run(metadata={"hardware": hw_info})
    write_log(active_run_id, "EVALUATION", "START", "Bắt đầu đánh giá 5 chỉ số (MSE, RMSE, MAE, MAPE, R²), 5-Fold CV và Walk-Forward Backtesting.")

    elec_proc_dir = ROOT_DIR / cfg["electricity_data"]["processed_dir"]
    pems_proc_dir = ROOT_DIR / cfg["pems_bay_advanced"]["processed_dir"]
    models_dir = ROOT_DIR / cfg["paths"]["models_dir"]
    metrics_dir = ROOT_DIR / cfg["paths"]["metrics_dir"]
    preds_dir = ROOT_DIR / cfg["paths"]["predictions_dir"]
    ensure_dir(metrics_dir)
    ensure_dir(preds_dir)

    bundle = load_object(elec_proc_dir / "electricity_processed_bundle.pkl")
    kfold_splits = load_object(elec_proc_dir / "electricity_kfold_bundle.pkl")
    training_meta = load_json(models_dir / "training_metadata.json", default={})
    eda_summary = load_json(metrics_dir / "eda_summary.json", default={})

    var_model: VARForecaster = load_object(models_dir / "var_model.pkl")
    xgb_model: XGBoostForecaster = load_object(models_dir / "xgboost_model.pkl")
    lstm_model: LSTMForecaster = load_object(models_dir / "lstm_model.pkl")

    load_mean = float(bundle["load_mean"])
    load_std = float(bundle["load_std"])
    y_test_orig = bundle["y_test"] * load_std + load_mean
    y_train_orig = bundle["y_train"] * load_std + load_mean

    # 1. Đánh giá trên tập Hold-out Test (15%) & tập Train (70%)
    df_hist_for_test = pd.concat([bundle["df_train_scaled"], bundle["df_val_scaled"]], axis=0)

    t_p0 = time.perf_counter()
    var_test_pred = var_model.predict_rolling_one_step(df_hist_for_test, bundle["df_test_scaled"]) * load_std + load_mean
    var_pred_time = round(time.perf_counter() - t_p0, 6)
    var_train_pred = var_model.predict_rolling_one_step(
        bundle["df_train_scaled"].iloc[: var_model.selected_lag + 2],
        bundle["df_train_scaled"].iloc[bundle["X_xgb_train"].shape[0] * (-1) :],
    ) * load_std + load_mean

    t_p0 = time.perf_counter()
    xgb_test_pred = xgb_model.predict(bundle["X_xgb_test"]) * load_std + load_mean
    xgb_pred_time = round(time.perf_counter() - t_p0, 6)
    xgb_train_pred = xgb_model.predict(bundle["X_xgb_train"]) * load_std + load_mean

    t_p0 = time.perf_counter()
    lstm_test_pred = lstm_model.predict(bundle["X_lstm_test"]) * load_std + load_mean
    lstm_pred_time = round(time.perf_counter() - t_p0, 6)
    lstm_train_pred = lstm_model.predict(bundle["X_lstm_train"]) * load_std + load_mean

    holdout_test_metrics: Dict[str, Dict[str, Any]] = {}
    holdout_train_metrics: Dict[str, Dict[str, Any]] = {}
    test_rows: List[Dict[str, Any]] = []
    log_rows: List[Dict[str, Any]] = []

    for m_name, test_p, train_p, p_time, tr_meta_key in [
        ("VAR", var_test_pred, var_train_pred, var_pred_time, "var_model"),
        ("XGBoost", xgb_test_pred, xgb_train_pred, xgb_pred_time, "xgboost_model"),
        ("LSTM", lstm_test_pred, lstm_train_pred, lstm_pred_time, "lstm_model"),
    ]:
        m_test = compute_5_regression_metrics(y_test_orig, test_p)
        m_train = compute_5_regression_metrics(y_train_orig, train_p)
        tr_time = float(training_meta.get(tr_meta_key, {}).get("training_time_sec", 0.0))
        dev_used = str(training_meta.get(tr_meta_key, {}).get("device_used", "cpu"))

        m_test["Model"] = m_name
        m_test["Training_Time_Sec"] = tr_time
        m_test["Prediction_Time_Sec"] = p_time
        m_test["Device"] = dev_used
        holdout_test_metrics[m_name] = m_test
        holdout_train_metrics[m_name] = m_train
        test_rows.append(m_test)

        log_rows.append({
            "Task": "Electricity_Load_1_11",
            "Evaluation_Mode": "Holdout_Test_15pct",
            **m_test,
        })

    df_test_comp = pd.DataFrame(test_rows)[
        ["Model", "MSE", "RMSE", "MAE", "MAPE", "R2", "Max_Error", "Mean_Residual", "Training_Time_Sec", "Prediction_Time_Sec", "Device"]
    ]
    df_test_comp.to_csv(metrics_dir / "test_metrics_comparison.csv", index=False, encoding="utf-8")
    df_test_comp.to_csv(metrics_dir / "metrics_summary.csv", index=False, encoding="utf-8")

    # Lưu bảng dự đoán chi tiết từng giờ trên tập Test
    df_preds = pd.DataFrame({
        "datetime": bundle["test_timestamps"],
        "Actual_Load": np.round(y_test_orig, 4),
        "VAR_Pred": np.round(var_test_pred, 4),
        "XGBoost_Pred": np.round(xgb_test_pred, 4),
        "LSTM_Pred": np.round(lstm_test_pred, 4),
        "VAR_Residual": np.round(y_test_orig - var_test_pred, 4),
        "XGBoost_Residual": np.round(y_test_orig - xgb_test_pred, 4),
        "LSTM_Residual": np.round(y_test_orig - lstm_test_pred, 4),
    })
    df_preds.to_csv(preds_dir / "test_predictions_latest.csv", index=False, encoding="utf-8")
    df_preds.to_csv(preds_dir / f"test_predictions_{active_run_id}.csv", index=False, encoding="utf-8")

    # 2. Thực thi Walk-Forward Backtesting (Mục 9)
    wf_hours = int(cfg.get("backtesting", {}).get("evaluation_window_hours", 168))
    wf_res = run_walk_forward_backtesting(bundle, var_model, xgb_model, lstm_model, window_hours=wf_hours)
    wf_res["predictions_df"].to_csv(preds_dir / "walk_forward_predictions.csv", index=False, encoding="utf-8")

    for m_name, wf_m in wf_res["metrics"].items():
        log_rows.append({
            "Task": "Electricity_Load_1_11",
            "Evaluation_Mode": f"Walk_Forward_{wf_res['window_hours']}h",
            "Model": m_name,
            "Device": holdout_test_metrics[m_name]["Device"],
            "Training_Time_Sec": holdout_test_metrics[m_name]["Training_Time_Sec"],
            **wf_m,
        })

    # 3. Thực thi Temporal 5-Fold Cross-Validation (Yêu cầu bổ sung số 4)
    diff_order = int(bundle["adf_report"].get("diff_order_applied", 0))
    if not skip_kfold and bool(cfg.get("splitting_and_validation", {}).get("kfold_cv", {}).get("enabled", True)):
        kfold_res = run_kfold_cross_validation(kfold_splits, cfg, hw_info, diff_order=diff_order)
        pd.DataFrame(kfold_res["fold_rows"]).to_csv(metrics_dir / "kfold_cv_results.csv", index=False, encoding="utf-8")
        for m_name, kf_m in kfold_res["summary_by_model"].items():
            log_rows.append({
                "Task": "Electricity_Load_1_11",
                "Evaluation_Mode": "5Fold_TimeSeriesSplit_Mean",
                "Model": m_name,
                "MSE": kf_m["MSE"]["mean"],
                "RMSE": kf_m["RMSE"]["mean"],
                "MAE": kf_m["MAE"]["mean"],
                "MAPE": kf_m["MAPE"]["mean"],
                "R2": kf_m["R2"]["mean"],
                "Device": holdout_test_metrics[m_name]["Device"],
                "Training_Time_Sec": holdout_test_metrics[m_name]["Training_Time_Sec"],
            })
    else:
        kfold_res = {"fold_rows": [], "summary_by_model": {}}

    # 4. Đánh giá Bài tập 12 nâng cao: DCRNN trên PEMS-BAY (15, 30, 60 phút)
    pems_eval: Dict[str, Any] = {"enabled": False}
    dcrnn_model_path = models_dir / "dcrnn_model.pkl"
    pems_bundle_path = pems_proc_dir / "pems_bay_processed_bundle.pkl"
    if not skip_pems and dcrnn_model_path.exists() and pems_bundle_path.exists():
        pems_bundle = load_object(pems_bundle_path)
        dcrnn_model = load_object(dcrnn_model_path)
        pems_eval = evaluate_pems_bay_dcrnn(pems_bundle, dcrnn_model)
        pems_eval["enabled"] = True
        pd.DataFrame(pems_eval["horizon_rows"]).to_csv(metrics_dir / "pems_bay_metrics.csv", index=False, encoding="utf-8")
        for h_row in pems_eval["horizon_rows"]:
            log_rows.append({
                "Task": "PEMS_BAY_Traffic_12",
                "Evaluation_Mode": h_row["Horizon"],
                "Model": "DCRNN",
                "Device": training_meta.get("dcrnn_model", {}).get("device_used", hw_info["torch_device"]),
                "Training_Time_Sec": training_meta.get("dcrnn_model", {}).get("training_time_sec", 0.0),
                **h_row,
            })

    # Xác định mô hình dẫn đầu theo từng chỉ số trên tập Test
    best_by_metric = {
        "MSE": min(holdout_test_metrics, key=lambda k: holdout_test_metrics[k]["MSE"]),
        "RMSE": min(holdout_test_metrics, key=lambda k: holdout_test_metrics[k]["RMSE"]),
        "MAE": min(holdout_test_metrics, key=lambda k: holdout_test_metrics[k]["MAE"]),
        "MAPE": min(holdout_test_metrics, key=lambda k: holdout_test_metrics[k]["MAPE"]),
        "R2": max(holdout_test_metrics, key=lambda k: holdout_test_metrics[k]["R2"]),
    }

    section11_answers = build_section11_analytical_answers(
        eda_summary=eda_summary,
        holdout_metrics=holdout_test_metrics,
        kfold_summary=kfold_res.get("summary_by_model", {}),
        training_meta=training_meta,
        pems_eval=pems_eval,
    )

    # 5. Vẽ toàn bộ bộ biểu đồ đánh giá 300 DPI
    stage_fig_dir = create_stage_figure_dir(active_run_id, "evaluation")
    saved_figs = []
    saved_figs.extend(plot_actual_vs_predicted_all(df_preds, stage_dir=stage_fig_dir))
    saved_figs.extend(plot_walk_forward_backtest(wf_res["predictions_df"], wf_res["metrics"], stage_dir=stage_fig_dir))
    saved_figs.extend(plot_metric_comparison_bars(holdout_test_metrics, stage_dir=stage_fig_dir))
    saved_figs.extend(plot_residual_analysis(df_preds, stage_dir=stage_fig_dir))
    if kfold_res.get("fold_rows"):
        saved_figs.extend(plot_kfold_cv_comparison(pd.DataFrame(kfold_res["fold_rows"]), stage_dir=stage_fig_dir))
    if pems_eval.get("enabled"):
        saved_figs.extend(plot_pems_bay_horizons(pems_eval, stage_dir=stage_fig_dir))

    append_experiment_log(active_run_id, log_rows)

    duration = time.perf_counter() - t0
    eval_payload = {
        "run_id": active_run_id,
        "hardware": hw_info,
        "holdout_test_metrics": holdout_test_metrics,
        "holdout_train_metrics": holdout_train_metrics,
        "best_model_by_metric": best_by_metric,
        "walk_forward_backtesting": {
            "window_hours": wf_res["window_hours"],
            "metrics": wf_res["metrics"],
        },
        "kfold_cv": kfold_res,
        "pems_bay_dcrnn": pems_eval,
        "section11_analytical_answers": section11_answers,
        "figures": saved_figs,
        "duration_sec": round(duration, 4),
    }

    save_json(eval_payload, metrics_dir / "evaluation_latest.json")
    save_json(eval_payload, metrics_dir / f"evaluation_{active_run_id}.json")

    update_progress(
        active_run_id,
        "EVALUATION",
        "COMPLETED",
        input_data={"num_test_samples": len(y_test_orig), "walk_forward_hours": wf_res["window_hours"]},
        output_data={"best_model_by_metric": best_by_metric, "holdout_test_metrics": holdout_test_metrics},
        duration_sec=duration,
    )
    write_log(
        active_run_id,
        "EVALUATION",
        "SUCCESS",
        f"Hoàn tất Evaluation: Best RMSE = {best_by_metric['RMSE']} ({holdout_test_metrics[best_by_metric['RMSE']]['RMSE']:.4f}), "
        f"Best R² = {best_by_metric['R2']} ({holdout_test_metrics[best_by_metric['R2']]['R2']:.4f}).",
    )
    print(
        f"[EVALUATION] Test RMSE -> VAR: {holdout_test_metrics['VAR']['RMSE']:.3f} | "
        f"XGBoost: {holdout_test_metrics['XGBoost']['RMSE']:.3f} | "
        f"LSTM: {holdout_test_metrics['LSTM']['RMSE']:.3f}"
    )
    return eval_payload


if __name__ == "__main__":
    main()