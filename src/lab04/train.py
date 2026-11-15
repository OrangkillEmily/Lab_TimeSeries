"""
Module điều phối huấn luyện đa mô hình trên GPU/CPU (`src/lab04/train.py`).

Trách nhiệm của module (Thực hiện Mục 6, 7, 8, 12 và Yêu cầu bổ sung số 5 Auto-GPU):
1. Tự động dò tìm thiết bị tính toán qua `detect_hardware_accelerator()` (ưu tiên GPU NVIDIA CUDA
   kèm Mixed Precision FP16 -> Apple MPS -> Multi-core CPU).
2. Huấn luyện cả 3 mô hình dự báo phụ tải điện năng trên tập `Train (70%)` và giám sát trên `Validation (15%)`:
   - `VAR (Statistical Baseline)`
   - `XGBoost (Machine Learning Tabular)`
   - `LSTM (Deep Learning Tensor 3D: 64 -> 32 -> 1)`
3. Huấn luyện mô hình nâng cao `DCRNN (Diffusion Convolutional Recurrent Neural Network)`
   trên dữ liệu giao thông `PEMS-BAY` (`data/raw/pems-bay.h5` + `data/raw/sensor_graph/adj_mx_bay.pkl`).
4. Lưu toàn bộ trọng số mô hình, bảng `Feature Importance`, lịch sử hội tụ `lstm_training_history.csv`
   và `training_metadata.json` vào `models/lab04/`, đồng thời vẽ biểu đồ đường cong hội tụ.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.lab04.model import (
    DCRNNTrafficForecaster,
    LSTMForecaster,
    VARForecaster,
    XGBoostForecaster,
)
from src.lab04.tracking import create_run, update_progress, write_log
from src.lab04.utils import (
    clear_hardware_memory,
    detect_hardware_accelerator,
    ensure_dir,
    get_project_root,
    load_config,
    load_object,
    save_json,
    save_object,
    set_global_seeds,
)
from src.lab04.visualize import (
    create_stage_figure_dir,
    plot_lstm_training_curves,
    plot_training_time_comparison,
    plot_xgboost_feature_importance,
)

ROOT_DIR = get_project_root()


def main(run_id: Optional[str] = None, skip_pems: bool = False) -> Dict[str, Any]:
    """
    What: Thực thi toàn bộ giai đoạn `TRAINING` cho `VAR`, `XGBoost`, `LSTM` và `DCRNN` (PEMS-BAY).
    Why: Huấn luyện và lưu trữ các mô hình tối ưu vào `models/lab04/` để phục vụ giai đoạn đánh giá
         và Walk-Forward Backtesting.
    Input:
        - run_id (Optional[str]): Mã định danh phiên chạy.
        - skip_pems (bool): Cờ tùy chọn bỏ qua huấn luyện bài 12 nếu người dùng chỉ chạy bài 1–11.
    Output:
        - Dict[str, Any]: Metadata huấn luyện chi tiết của cả 4 mô hình và thời gian thực thi.
    How: Nạp gói dữ liệu đã tiền xử lý từ `data/processed/lab04/`, khởi tạo bộ tăng tốc phần cứng
         `Auto-GPU`, huấn luyện tuần tự từng mô hình và lưu kết quả.
    """
    t0 = time.perf_counter()
    cfg = load_config()
    seed = int(cfg.get("project", {}).get("random_state", 42))
    set_global_seeds(seed)
    hw_info = detect_hardware_accelerator(cfg)

    active_run_id = run_id or create_run(metadata={"hardware": hw_info})
    write_log(
        active_run_id,
        "TRAINING",
        "START",
        f"Bắt đầu huấn luyện mô hình trên thiết bị: {hw_info['device_name']} (device={hw_info['torch_device']}, AMP={hw_info['amp_enabled']}).",
    )

    elec_proc_path = ROOT_DIR / cfg["electricity_data"]["processed_dir"] / "electricity_processed_bundle.pkl"
    pems_proc_path = ROOT_DIR / cfg["pems_bay_advanced"]["processed_dir"] / "pems_bay_processed_bundle.pkl"
    models_dir = ROOT_DIR / cfg["paths"]["models_dir"]
    metrics_dir = ROOT_DIR / cfg["paths"]["metrics_dir"]
    ensure_dir(models_dir)
    ensure_dir(metrics_dir)

    bundle = load_object(elec_proc_path)
    df_train_scaled = bundle["df_train_scaled"]
    X_xgb_train, y_train = bundle["X_xgb_train"], bundle["y_train"]
    X_xgb_val, y_val = bundle["X_xgb_val"], bundle["y_val"]
    X_lstm_train, X_lstm_val = bundle["X_lstm_train"], bundle["X_lstm_val"]
    feat_names = bundle["feature_names"]
    diff_order = int(bundle["adf_report"].get("diff_order_applied", 0))

    # 1. Huấn luyện Mô hình 1: VAR (Statistical Baseline)
    t_var0 = time.perf_counter()
    var_forecaster = VARForecaster(
        maxlags=int(cfg.get("var_model", {}).get("maxlags", 24)),
        ic=str(cfg.get("var_model", {}).get("information_criterion", "aic")),
        diff_order=diff_order,
    )
    var_info = var_forecaster.fit(df_train_scaled)
    var_time = round(time.perf_counter() - t_var0, 4)
    var_info["training_time_sec"] = var_time
    var_info["device_used"] = "cpu"
    save_object(var_forecaster, models_dir / "var_model.pkl")

    # 2. Huấn luyện Mô hình 2: XGBoost (Machine Learning Tabular)
    t_xgb0 = time.perf_counter()
    xgb_forecaster = XGBoostForecaster(cfg, hw_info)
    xgb_info = xgb_forecaster.fit(X_xgb_train, y_train, X_xgb_val, y_val)
    xgb_time = round(time.perf_counter() - t_xgb0, 4)
    xgb_info["training_time_sec"] = xgb_time
    save_object(xgb_forecaster, models_dir / "xgboost_model.pkl")

    df_imp = xgb_forecaster.get_feature_importances(feat_names)
    df_imp.to_csv(metrics_dir / "xgboost_feature_importance.csv", index=False, encoding="utf-8")

    # 3. Huấn luyện Mô hình 3: LSTM (Deep Learning: 64 -> 32 -> 1)
    clear_hardware_memory()
    t_lstm0 = time.perf_counter()
    lstm_forecaster = LSTMForecaster(cfg, hw_info)
    lstm_info = lstm_forecaster.fit(X_lstm_train, y_train, X_lstm_val, y_val)
    lstm_time = round(time.perf_counter() - t_lstm0, 4)
    lstm_info["training_time_sec"] = lstm_time
    if HAS_TORCH and hasattr(lstm_forecaster.net, "state_dict"):
        try:
            torch.save(lstm_forecaster.net.state_dict(), models_dir / "lstm_weights.pt")
        except Exception:
            pass
    save_object(lstm_forecaster, models_dir / "lstm_model.pkl")

    df_lstm_hist = pd.DataFrame(lstm_forecaster.history)
    df_lstm_hist.to_csv(models_dir / "lstm_training_history.csv", index=False, encoding="utf-8")
    save_json({"history": lstm_forecaster.history, "info": lstm_info}, models_dir / "lstm_training_history.json")

    # 4. Huấn luyện Mô hình 4: DCRNN trên PEMS-BAY (Mục 12 Nâng cao)
    dcrnn_info: Dict[str, Any] = {"enabled": False}
    if not skip_pems and bool(cfg.get("pems_bay_advanced", {}).get("enabled", True)) and pems_proc_path.exists():
        clear_hardware_memory()
        pems_bundle = load_object(pems_proc_path)
        t_dcrnn0 = time.perf_counter()
        dcrnn_forecaster = DCRNNTrafficForecaster(cfg, hw_info)
        dcrnn_info = dcrnn_forecaster.fit(pems_bundle)
        dcrnn_time = round(time.perf_counter() - t_dcrnn0, 4)
        dcrnn_info["training_time_sec"] = dcrnn_time
        dcrnn_info["enabled"] = True
        if HAS_TORCH and dcrnn_forecaster.net is not None:
            try:
                torch.save(dcrnn_forecaster.net.state_dict(), models_dir / "dcrnn_pems_bay.pt")
            except Exception:
                pass
        save_object(dcrnn_forecaster, models_dir / "dcrnn_model.pkl")
        pd.DataFrame(dcrnn_forecaster.history).to_csv(models_dir / "dcrnn_training_history.csv", index=False, encoding="utf-8")

    stage_fig_dir = create_stage_figure_dir(active_run_id, "training")
    saved_figs = []
    saved_figs.extend(plot_lstm_training_curves(df_lstm_hist, lstm_info["best_epoch"], stage_dir=stage_fig_dir))
    saved_figs.extend(plot_xgboost_feature_importance(df_imp, stage_dir=stage_fig_dir))
    time_map = {
        "VAR": var_time,
        "XGBoost": xgb_time,
        "LSTM": lstm_time,
    }
    if dcrnn_info.get("enabled"):
        time_map["DCRNN (PEMS-BAY)"] = float(dcrnn_info["training_time_sec"])
    saved_figs.extend(plot_training_time_comparison(time_map, stage_dir=stage_fig_dir))

    total_duration = time.perf_counter() - t0
    training_metadata = {
        "run_id": active_run_id,
        "hardware": hw_info,
        "var_model": var_info,
        "xgboost_model": xgb_info,
        "lstm_model": lstm_info,
        "dcrnn_model": dcrnn_info,
        "top_10_xgboost_features": df_imp.head(10).to_dict(orient="records"),
        "figures": saved_figs,
        "duration_sec": round(total_duration, 4),
    }
    save_json(training_metadata, models_dir / "training_metadata.json")
    save_json(training_metadata, metrics_dir / "training_summary.json")

    update_progress(
        active_run_id,
        "TRAINING",
        "COMPLETED",
        input_data={"device": hw_info["device_name"], "torch_device": hw_info["torch_device"]},
        output_data=training_metadata,
        duration_sec=total_duration,
    )
    write_log(
        active_run_id,
        "TRAINING",
        "SUCCESS",
        f"Hoàn tất huấn luyện: VAR={var_time}s, XGBoost={xgb_time}s, LSTM={lstm_time}s ({total_duration:.2f}s).",
    )
    print(
        f"[TRAINING] Device: {hw_info['device_name']} | VAR (p={var_info['selected_lag']}): {var_time:.2f}s | "
        f"XGBoost (iter={xgb_info['best_iteration']}): {xgb_time:.2f}s | LSTM (best_ep={lstm_info['best_epoch']}): {lstm_time:.2f}s"
    )
    return training_metadata


if __name__ == "__main__":
    main()