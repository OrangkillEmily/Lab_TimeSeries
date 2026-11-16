r"""
Mô-đun Huấn luyện Đa Mô hình & Kiểm định Chéo Chuỗi Thời gian (Training & Cross-Validation Orchestrator)
cho Bài thực hành Lab 04.

Thực hiện huấn luyện đồng bộ trên GPU/CPU:
1. Huấn luyện trên tập Hold-out (Train 70% - Validation 15%) cho 3 mô hình phụ tải điện:
   - VAR (Vector Autoregression trên miền sai phân dừng bậc 1)
   - XGBoost Multivariate Regressor (105 đặc trưng)
   - Stacked 2-Layer LSTM + Residual Highway
2. Thực thi 5-Fold TimeSeriesSplit Cross-Validation để kiểm tra độ ổn định qua các giai đoạn.
3. Thực thi Nghiên cứu Loại bỏ Đặc trưng (Feature Ablation: Có Thời tiết vs. Không Thời tiết).
4. Huấn luyện Đối sánh 3 mô hình Đa tầm nhìn trên PEMS-BAY (Mục 12 Nâng cao):
   - DCRNN (Diffusion Convolutional Recurrent Neural Network - có ma trận đồ thị adj_mx_bay.pkl)
   - Multi-Horizon TCN (Temporal Convolutional Network - Slide 10 Chương 4-5)
   - Multi-Horizon LSTM (Temporal LSTM đối chứng không dùng đồ thị)
"""

import time
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from src.lab04.model import (
    DCRNNNet,
    LSTMForecaster,
    MultiHorizonPemsLSTMNet,
    MultiHorizonTCNNet,
    SpatioTemporalTrainer,
    VARForecaster,
    XGBoostForecaster,
    compute_dual_random_walk_supports,
)
from src.lab04.tracking import get_active_run_id, record_stage_result, write_log
from src.lab04.utils import (
    ensure_directories,
    get_compute_device,
    get_path,
    load_config,
    load_pickle,
    save_json,
    save_pickle,
    set_global_seed,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def run_5fold_cross_validation(
    cv_folds: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    r"""
    Purpose: Thực thi 5-Fold TimeSeriesSplit Cross-Validation cho cả 3 mô hình VAR, XGBoost và LSTM.
    Input-Output:
        - Input: cv_folds (Danh sách 5 gói dữ liệu Fold từ preprocess.py), cfg (Dict).
        - Output: Danh sách kết quả dự báo và thời gian huấn luyện trên từng Fold k = 1..5.
    How: Với mỗi Fold, khởi tạo mới hoàn toàn mô hình, chỉ fit trên tập Train^(k) và dự báo trên Val^(k).
    """
    var_cfg = cfg.get("models", {}).get("var", {})
    cv_records: List[Dict[str, Any]] = []

    for fold_bundle in cv_folds:
        k = int(fold_bundle["fold"])
        load_mean = float(fold_bundle["load_mean"])
        load_std = float(fold_bundle["load_std"])
        y_true_orig = fold_bundle["y_val_orig"]

        # 1. VAR trên Fold k
        var_m = VARForecaster(
            maxlags=int(var_cfg.get("maxlags", 24)),
            ic=str(var_cfg.get("ic", "aic")),
        )
        var_m.fit(fold_bundle["train_raw_df"])
        var_pred_orig = var_m.predict_rolling_1step(fold_bundle["train_raw_df"], fold_bundle["val_raw_df"])

        # 2. XGBoost trên Fold k
        xgb_m = XGBoostForecaster(cfg, use_gpu=True)
        xgb_m.fit(
            fold_bundle["X_xgb_train"],
            fold_bundle["y_xgb_train"],
            fold_bundle["X_xgb_val"],
            fold_bundle["y_xgb_val"],
        )
        xgb_pred_orig = xgb_m.predict_original(fold_bundle["X_xgb_val"], load_mean, load_std)

        # 3. LSTM trên Fold k
        n_features = fold_bundle["X_lstm_train"].shape[2]
        lstm_m = LSTMForecaster(input_dim=n_features, config=cfg)
        lstm_m.fit(
            fold_bundle["X_lstm_train"],
            fold_bundle["y_lstm_train"],
            fold_bundle["X_lstm_val"],
            fold_bundle["y_lstm_val"],
            epochs_override=14,
        )
        lstm_pred_orig = lstm_m.predict_original(fold_bundle["X_lstm_val"], load_mean, load_std)

        cv_records.append(
            {
                "fold": k,
                "train_size": fold_bundle["train_size"],
                "val_size": fold_bundle["val_size"],
                "train_period": fold_bundle["train_period"],
                "val_period": fold_bundle["val_period"],
                "y_true_orig": y_true_orig,
                "preds_orig": {
                    "VAR": var_pred_orig,
                    "XGBoost": xgb_pred_orig,
                    "LSTM": lstm_pred_orig,
                },
                "train_times": {
                    "VAR": round(var_m.train_time_sec, 4),
                    "XGBoost": round(xgb_m.train_time_sec, 4),
                    "LSTM": round(lstm_m.train_time_sec, 4),
                },
            }
        )
    return cv_records


def run_weather_ablation_training(
    processed_bundle: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    r"""
    Purpose: Huấn luyện đối chứng Khi KHÔNG có biến thời tiết (Univariate Load Only) để chứng minh
             đóng góp thực nghiệm của 3 biến ngoại sinh (temperature, humidity, wind_speed) ở Mục 11.
    Input-Output:
        - Input: processed_bundle (Dict), cfg (Dict).
        - Output: Dự báo trên tập Test của các mô hình khi chỉ dùng duy nhất chuỗi 'load'.
    How: Lọc bỏ toàn bộ cột thời tiết khỏi ma trận đặc trưng của XGBoost và tensor 3D của LSTM, huấn luyện lại
         trên cùng cấu hình siêu tham số.
    """
    feat_names: List[str] = processed_bundle["feature_names"]
    keep_idx = [
        i
        for i, name in enumerate(feat_names)
        if not any(w in name for w in ("temperature", "humidity", "wind_speed"))
    ]

    load_mean = float(processed_bundle["scaler_stats"]["load_mean"])
    load_std = float(processed_bundle["scaler_stats"]["load_std"])

    X_xgb_tr_uni = processed_bundle["X_xgb_train"][:, keep_idx]
    X_xgb_va_uni = processed_bundle["X_xgb_val"][:, keep_idx]
    X_xgb_te_uni = processed_bundle["X_xgb_test"][:, keep_idx]

    xgb_uni = XGBoostForecaster(cfg, use_gpu=True)
    xgb_uni.fit(X_xgb_tr_uni, processed_bundle["y_xgb_train"], X_xgb_va_uni, processed_bundle["y_xgb_val"])
    xgb_uni_test_pred = xgb_uni.predict_original(X_xgb_te_uni, load_mean, load_std)

    X_lstm_tr_uni = processed_bundle["X_lstm_train"][:, :, :1]
    X_lstm_va_uni = processed_bundle["X_lstm_val"][:, :, :1]
    X_lstm_te_uni = processed_bundle["X_lstm_test"][:, :, :1]

    lstm_uni = LSTMForecaster(input_dim=1, config=cfg)
    lstm_uni.fit(
        X_lstm_tr_uni,
        processed_bundle["y_lstm_train"],
        X_lstm_va_uni,
        processed_bundle["y_lstm_val"],
        epochs_override=16,
    )
    lstm_uni_test_pred = lstm_uni.predict_original(X_lstm_te_uni, load_mean, load_std)

    # Đối chứng AR(p) đơn biến so với VAR(p) đa biến
    train_load_only = processed_bundle["train_df"][["load"]].copy()
    val_test_load_only = pd.concat([processed_bundle["val_df"][["load"]], processed_bundle["test_df"][["load"]]], axis=0)
    raw_load = pd.concat([train_load_only.iloc[-25:], val_test_load_only], axis=0)["load"].to_numpy(dtype=np.float64)
    diff_load = np.diff(raw_load)
    # Ước lượng AR(24) bằng OLS trên sai phân bậc 1 của tập Train
    tr_diff = np.diff(train_load_only["load"].to_numpy(dtype=np.float64))
    p = 24
    X_ar = np.column_stack([np.ones(len(tr_diff) - p)] + [tr_diff[p - k : len(tr_diff) - k] for k in range(1, p + 1)])
    y_ar = tr_diff[p:]
    beta_ar, _, _, _ = np.linalg.lstsq(X_ar, y_ar, rcond=None)

    n_val = len(processed_bundle["val_df"])
    n_test = len(processed_bundle["test_df"])
    ar_test_preds = np.empty(n_test, dtype=np.float64)
    for i in range(n_test):
        offset = n_val + i
        lags_vec = diff_load[offset : offset + p][::-1]
        delta_hat = float(beta_ar[0] + np.dot(beta_ar[1:], lags_vec))
        ar_test_preds[i] = raw_load[offset + p] + delta_hat

    return {
        "VAR_Univariate_AR24": ar_test_preds,
        "XGBoost_NoWeather": xgb_uni_test_pred,
        "LSTM_NoWeather": lstm_uni_test_pred,
    }


def main(run_id: Optional[str] = None) -> Dict[str, Any]:
    r"""
    Purpose: Thực thi toàn trình giai đoạn Huấn luyện (Training Phase) cho cả bộ dữ liệu Điện năng (Mục 5-11)
             và bộ dữ liệu Giao thông Đồ thị PEMS-BAY (Mục 12 Nâng cao: DCRNN vs. TCN vs. LSTM).
    Input-Output:
        - Input: run_id (Optional[str]).
        - Output: Từ điển tổng hợp thời gian huấn luyện, siêu tham số, lịch sử hội tụ và đường dẫn lưu mô hình.
    How: Nạp processed_bundle.pkl, huấn luyện tuần tự từng mô hình trên GPU/CPU, lưu trọng số vào models/lab04/
         và gói dự báo vào data/processed/lab04/training_artifacts.pkl.
    """
    t0_all = time.perf_counter()
    cfg = load_config()
    ensure_directories(cfg)
    set_global_seed(int(cfg.get("project", {}).get("random_seed", 42)))
    active_run_id = get_active_run_id(run_id)
    hw_info = get_compute_device(cfg)

    write_log(
        active_run_id,
        "TRAINING",
        "START",
        f"Bắt đầu huấn luyện đa mô hình trên thiết bị: {hw_info['device_name']} (torch={hw_info['torch_device']}, xgb={hw_info['xgb_device']}).",
    )

    proc_dir = get_path("data_processed_dir", cfg)
    models_dir = get_path("models_dir", cfg)
    bundle = load_pickle(proc_dir / "processed_bundle.pkl")

    load_mean = float(bundle["scaler_stats"]["load_mean"])
    load_std = float(bundle["scaler_stats"]["load_std"])
    lookback = int(bundle["lookback"])

    # -------------------------------------------------------------------------
    # A. HUẤN LUYỆN 3 MÔ HÌNH CHÍNH TRÊN TẬP ELECTRICITY LOAD (MỤC 5, 6, 7)
    # -------------------------------------------------------------------------
    var_cfg = cfg.get("models", {}).get("var", {})
    var_model = VARForecaster(
        maxlags=int(var_cfg.get("maxlags", 24)),
        ic=str(var_cfg.get("ic", "aic")),
        trend=str(var_cfg.get("trend", "c")),
    )
    var_model.fit(bundle["train_df"])
    t_p0 = time.perf_counter()
    var_train_pred = var_model.predict_in_sample(bundle["train_df"], lookback=lookback)
    var_val_pred = var_model.predict_rolling_1step(bundle["train_df"], bundle["val_df"])
    train_plus_val_df = pd.concat([bundle["train_df"], bundle["val_df"]], axis=0)
    var_test_pred = var_model.predict_rolling_1step(train_plus_val_df, bundle["test_df"])
    var_pred_sec = time.perf_counter() - t_p0

    xgb_model = XGBoostForecaster(cfg, use_gpu=True)
    xgb_model.fit(bundle["X_xgb_train"], bundle["y_xgb_train"], bundle["X_xgb_val"], bundle["y_xgb_val"])
    t_p0 = time.perf_counter()
    xgb_train_pred = xgb_model.predict_original(bundle["X_xgb_train"], load_mean, load_std)
    xgb_val_pred = xgb_model.predict_original(bundle["X_xgb_val"], load_mean, load_std)
    xgb_test_pred = xgb_model.predict_original(bundle["X_xgb_test"], load_mean, load_std)
    xgb_pred_sec = time.perf_counter() - t_p0
    xgb_importance_df = xgb_model.get_feature_importances(bundle["feature_names"])

    n_features = bundle["X_lstm_train"].shape[2]
    lstm_model = LSTMForecaster(input_dim=n_features, config=cfg)
    lstm_model.fit(bundle["X_lstm_train"], bundle["y_lstm_train"], bundle["X_lstm_val"], bundle["y_lstm_val"])
    t_p0 = time.perf_counter()
    lstm_train_pred = lstm_model.predict_original(bundle["X_lstm_train"], load_mean, load_std)
    lstm_val_pred = lstm_model.predict_original(bundle["X_lstm_val"], load_mean, load_std)
    lstm_test_pred = lstm_model.predict_original(bundle["X_lstm_test"], load_mean, load_std)
    lstm_pred_sec = time.perf_counter() - t_p0

    # -------------------------------------------------------------------------
    # B. KIỂM ĐỊNH CHÉO 5-FOLD TIMESERIESSPLIT & FEATURE ABLATION
    # -------------------------------------------------------------------------
    cv_records = run_5fold_cross_validation(bundle["cv_folds"], cfg)
    ablation_preds = run_weather_ablation_training(bundle, cfg)

    # -------------------------------------------------------------------------
    # C. BÀI TẬP 12 NÂNG CAO (PEMS-BAY): DCRNN vs. MULTI-HORIZON TCN vs. LSTM
    # -------------------------------------------------------------------------
    pems = bundle["pems_bay"]
    dcrnn_cfg = cfg.get("models", {}).get("dcrnn", {})
    supports = compute_dual_random_walk_supports(
        pems["adj_mx"],
        max_diffusion_step=int(dcrnn_cfg.get("diffusion_k", 2)),
    )
    n_sensors = int(pems["num_sensors"])
    pems_lookback = int(pems["lookback_steps"])
    n_horizons = len(pems["horizon_steps"])
    pems_mean = float(pems["scaler_mean"])
    pems_std = float(pems["scaler_std"])

    # 1. Mô hình chính: DCRNN (Spatio-Temporal Graph Diffusion + GRU)
    dcrnn_net = DCRNNNet(
        supports=supports,
        num_sensors=n_sensors,
        lookback=pems_lookback,
        hidden_dim=int(dcrnn_cfg.get("hidden_dim", 32)),
        num_horizons=n_horizons,
    )
    dcrnn_trainer = SpatioTemporalTrainer(
        model_name="DCRNN",
        net=dcrnn_net,
        config=cfg,
        epochs=int(dcrnn_cfg.get("epochs", 12)),
        batch_size=int(dcrnn_cfg.get("batch_size", 64)),
        lr=float(dcrnn_cfg.get("learning_rate", 0.003)),
    )
    dcrnn_trainer.fit(pems["X_train"], pems["Y_train_scaled"], pems["X_val"], pems["Y_val_scaled"])
    dcrnn_test_pred_orig = dcrnn_trainer.predict_original(pems["X_test"], pems_mean, pems_std)

    # 2. Mô hình đối chứng 1 trên PEMS-BAY: Multi-Horizon TCN (Slide 10 Chương 4-5)
    base_cfg = dcrnn_cfg.get("compare_baselines", {})
    base_epochs = int(base_cfg.get("baseline_epochs", 10))
    tcn_net = MultiHorizonTCNNet(
        lookback=pems_lookback,
        channels=int(base_cfg.get("tcn_channels", 32)),
        num_horizons=n_horizons,
    )
    tcn_trainer = SpatioTemporalTrainer(
        model_name="Multi-Horizon TCN",
        net=tcn_net,
        config=cfg,
        epochs=base_epochs,
        batch_size=int(dcrnn_cfg.get("batch_size", 64)),
        lr=0.003,
    )
    tcn_trainer.fit(pems["X_train"], pems["Y_train_scaled"], pems["X_val"], pems["Y_val_scaled"])
    tcn_test_pred_orig = tcn_trainer.predict_original(pems["X_test"], pems_mean, pems_std)

    # 3. Mô hình đối chứng 2 trên PEMS-BAY: Multi-Horizon LSTM (Không dùng ma trận đồ thị A)
    pems_lstm_net = MultiHorizonPemsLSTMNet(
        num_sensors=n_sensors,
        lookback=pems_lookback,
        hidden_dim=64,
        num_horizons=n_horizons,
    )
    pems_lstm_trainer = SpatioTemporalTrainer(
        model_name="Multi-Horizon LSTM",
        net=pems_lstm_net,
        config=cfg,
        epochs=base_epochs,
        batch_size=int(dcrnn_cfg.get("batch_size", 64)),
        lr=0.003,
    )
    pems_lstm_trainer.fit(pems["X_train"], pems["Y_train_scaled"], pems["X_val"], pems["Y_val_scaled"])
    pems_lstm_test_pred_orig = pems_lstm_trainer.predict_original(pems["X_test"], pems_mean, pems_std)

    # Lưu trọng số các mô hình vào models/lab04/
    save_pickle(
        {"selected_lag": var_model.selected_lag, "coefs": var_model.var_result.coefs, "intercept": var_model.var_result.intercept},
        models_dir / "var_model.pkl",
    )
    save_pickle(xgb_model.model, models_dir / "xgboost_model.pkl")
    torch.save(lstm_model.net.state_dict(), models_dir / "lstm_model.pt")
    torch.save(dcrnn_trainer.net.state_dict(), models_dir / "dcrnn_pems_bay.pt")
    torch.save(tcn_trainer.net.state_dict(), models_dir / "tcn_pems_bay.pt")
    torch.save(pems_lstm_trainer.net.state_dict(), models_dir / "lstm_pems_bay.pt")

    training_artifacts: Dict[str, Any] = {
        "hardware": hw_info,
        "holdout_predictions": {
            "y_train_orig": bundle["y_train_orig"],
            "y_val_orig": bundle["y_val_orig"],
            "y_test_orig": bundle["y_test_orig"],
            "train_timestamps": bundle["train_timestamps"],
            "val_timestamps": bundle["val_timestamps"],
            "test_timestamps": bundle["test_timestamps"],
            "train": {
                "VAR": var_train_pred,
                "XGBoost": xgb_train_pred,
                "LSTM": lstm_train_pred,
            },
            "val": {
                "VAR": var_val_pred,
                "XGBoost": xgb_val_pred,
                "LSTM": lstm_val_pred,
            },
            "test": {
                "VAR": var_test_pred,
                "XGBoost": xgb_test_pred,
                "LSTM": lstm_test_pred,
            },
        },
        "model_metadata": {
            "VAR": {
                "selected_lag": var_model.selected_lag,
                "train_time_sec": round(var_model.train_time_sec, 4),
                "pred_time_sec": round(var_pred_sec, 6),
                "device": "cpu",
            },
            "XGBoost": {
                "best_iteration": xgb_model.best_iteration,
                "train_time_sec": round(xgb_model.train_time_sec, 4),
                "pred_time_sec": round(xgb_pred_sec, 6),
                "device": xgb_model.device_used,
                "evals_result": xgb_model.evals_result,
            },
            "LSTM": {
                "history": lstm_model.history,
                "train_time_sec": round(lstm_model.train_time_sec, 4),
                "pred_time_sec": round(lstm_pred_sec, 6),
                "device": lstm_model.device_str,
                "hidden_dims": list(lstm_model.hidden_dims),
                "use_residual_highway": lstm_model.use_residual,
            },
            "DCRNN": {
                "history": dcrnn_trainer.history,
                "train_time_sec": round(dcrnn_trainer.train_time_sec, 4),
                "device": dcrnn_trainer.device_str,
            },
            "Multi_Horizon_TCN": {
                "history": tcn_trainer.history,
                "train_time_sec": round(tcn_trainer.train_time_sec, 4),
                "device": tcn_trainer.device_str,
            },
            "Multi_Horizon_LSTM": {
                "history": pems_lstm_trainer.history,
                "train_time_sec": round(pems_lstm_trainer.train_time_sec, 4),
                "device": pems_lstm_trainer.device_str,
            },
        },
        "xgb_feature_importance": xgb_importance_df,
        "cv_records": cv_records,
        "ablation_predictions": ablation_preds,
        "pems_bay_predictions": {
            "Y_test_orig": pems["Y_test_orig"],
            "DCRNN": dcrnn_test_pred_orig,
            "Multi_Horizon_TCN": tcn_test_pred_orig,
            "Multi_Horizon_LSTM": pems_lstm_test_pred_orig,
            "test_timestamps": pems["test_timestamps"],
            "horizon_names": pems["horizon_names"],
            "horizon_steps": pems["horizon_steps"],
        },
    }

    save_pickle(training_artifacts, proc_dir / "training_artifacts.pkl")

    total_sec = time.perf_counter() - t0_all
    train_summary = {
        "hardware": hw_info,
        "models": {
            "VAR": training_artifacts["model_metadata"]["VAR"],
            "XGBoost": {
                k: v for k, v in training_artifacts["model_metadata"]["XGBoost"].items() if k != "evals_result"
            },
            "LSTM": training_artifacts["model_metadata"]["LSTM"],
            "DCRNN": training_artifacts["model_metadata"]["DCRNN"],
            "Multi_Horizon_TCN": training_artifacts["model_metadata"]["Multi_Horizon_TCN"],
            "Multi_Horizon_LSTM": training_artifacts["model_metadata"]["Multi_Horizon_LSTM"],
        },
        "total_duration_sec": round(total_sec, 4),
    }

    save_json(train_summary, get_path("metrics_dir", cfg) / "training_summary.json")
    record_stage_result(active_run_id, "training", train_summary)

    write_log(
        active_run_id,
        "TRAINING",
        "SUCCESS",
        f"Hoàn tất huấn luyện toàn bộ mô hình trong {total_sec:.2f}s | VAR lag={var_model.selected_lag} | LSTM best_epoch={lstm_model.history['best_epoch']} | DCRNN, TCN, LSTM trên PEMS-BAY hoàn tất.",
    )
    print(
        f"[TRAIN] Hoàn tất trong {total_sec:.2f}s | Thiết bị: {hw_info['device_name']} | "
        f"VAR={var_model.train_time_sec:.2f}s, XGBoost={xgb_model.train_time_sec:.2f}s, "
        f"LSTM={lstm_model.train_time_sec:.2f}s, DCRNN={dcrnn_trainer.train_time_sec:.2f}s, "
        f"TCN={tcn_trainer.train_time_sec:.2f}s, PEMS-LSTM={pems_lstm_trainer.train_time_sec:.2f}s"
    )
    return train_summary


if __name__ == "__main__":
    main()