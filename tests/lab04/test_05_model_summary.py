r"""
Kiểm thử Tự động Mô-đun Trích xuất Hồ sơ Thống kê Mô hình (`src/lab04/model_summary.py`):
Đảm bảo thống kê đầy đủ 36/36 thực thể mô hình (24 Dataset 1 + 12 Dataset 2), trích xuất đúng
tham số trọng số thực tế (.pkl, .pt), kiểm định phần dư (Ljung-Box, Jarque-Bera, Heteroskedasticity,
Durbin-Watson) và tính toàn vẹn của bộ tệp .md, .csv, .json trong `reports/model_summary/lab04/`
(tests/lab04/test_05_model_summary.py).

Hỗ trợ tự động 2 chế độ:
- Chế độ Thực nghiệm (Production Mode): Khi đã chạy `run_lab04.py` (có sẵn `models/lab04/` và `reports/lab04/metrics/`),
  kiểm định trực tiếp trên dữ liệu và trọng số thực tế.
- Chế độ Giả lập Độc lập (Clean-Repo Fixture Mode): Khi chưa chạy `run_lab04.py`, tự động tạo bộ fixture giả lập
  tạm thời trong `tempfile.TemporaryDirectory()` để vẫn kiểm thử trọn vẹn 5/5 test cases mà không gây lỗi FileNotFoundError.
"""

import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.lab04.model_summary as ms_mod  # noqa: E402
from src.lab04.model_summary import (  # noqa: E402
    _extract_pytorch_state_dict_stats,
    _extract_var_statistical_table,
    _extract_xgboost_details,
    compute_detailed_residual_diagnostics,
    generate_all_model_summaries,
)
from src.lab04.utils import get_path, load_config  # noqa: E402


def _create_synthetic_lab04_workspace(tmp_root: Path) -> dict:
    r"""
    Tạo một workspace Lab 04 giả lập đầy đủ cấu trúc (`models/lab04`, `reports/lab04/metrics`,
    `progress_results/lab04`) trong thư mục tạm để test độc lập khi chưa chạy `run_lab04.py`.
    """
    models_dir = tmp_root / "models" / "lab04"
    metrics_dir = tmp_root / "reports" / "lab04" / "metrics"
    proc_dir = tmp_root / "data" / "processed" / "lab04"
    progress_dir = tmp_root / "progress_results" / "lab04"
    logs_dir = tmp_root / "logs" / "lab04"
    for d in (models_dir, metrics_dir, proc_dir, progress_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    run_id = "20261117_000000_test01"

    # 1. Mock var_model.pkl
    with open(models_dir / "var_model.pkl", "wb") as f:
        pickle.dump(
            {
                "selected_lag": 24,
                "coefs": np.random.default_rng(42).normal(0, 0.1, size=(24, 4, 4)),
                "intercept": np.array([1.5, 0.01, -0.01, 0.0]),
            },
            f,
        )

    # 2. Mock xgboost_model.pkl (lưu dict/stub đơn giản có __dict__ tương thích _safe_load_pickle)
    class _DummyXGB:
        def __init__(self) -> None:
            self.n_estimators = 450
            self.max_depth = 6
            self.learning_rate = 0.045
            self.subsample = 0.85
            self.colsample_bytree = 0.85
            self.reg_alpha = 0.1
            self.reg_lambda = 1.0
            self.tree_method = "hist"
            self.objective = "reg:squarederror"
            self.early_stopping_rounds = 30
            self.random_state = 42
            self.evals_result_ = {
                "validation_0": {"rmse": list(np.linspace(0.95, 0.04, 450))},
                "validation_1": {"rmse": list(np.linspace(0.82, 0.06, 450))},
            }

    # Lưu bằng dict thuộc tính qua _GenericPickleStub-compatible hoặc torch/pickle
    with open(models_dir / "xgboost_model.pkl", "wb") as f:
        pickle.dump(ms_mod._GenericPickleStub(), f)

    # 3. Mock 4 file .pt bằng torch (nếu có) hoặc bỏ qua nhẹ nhàng nếu không có torch
    try:
        import torch

        torch.save(
            {
                "lstm1.weight_ih_l0": torch.randn(256, 4),
                "lstm1.weight_hh_l0": torch.randn(256, 64),
                "highway.weight": torch.randn(1, 27),
                "highway.bias": torch.zeros(1),
            },
            models_dir / "lstm_model.pt",
        )
        torch.save(
            {
                "supports_stack": torch.randn(5, 207, 207),
                "cell1.gate_linear.weight": torch.randn(64, 165),
                "ar_highway.weight": torch.randn(3, 12),
            },
            models_dir / "dcrnn_pems_bay.pt",
        )
        torch.save({"conv1.weight": torch.randn(32, 1, 3), "ar_linear.weight": torch.randn(3, 12)}, models_dir / "tcn_pems_bay.pt")
        torch.save({"lstm.weight_ih_l0": torch.randn(256, 207), "ar_linear.weight": torch.randn(3, 12)}, models_dir / "lstm_pems_bay.pt")
    except Exception:
        pass

    # 4. Mock CSVs & JSONs trong reports/lab04/metrics/
    pd.DataFrame(
        [
            {
                "Model": "XGBoost",
                "MSE": 4.2e7,
                "RMSE": 6480.0,
                "MAE": 4330.0,
                "MAPE": 1.89,
                "R2": 0.9946,
                "Max_Error": 120000.0,
                "Mean_Residual": -150.0,
                "Std_Residual": 6478.0,
                "Train_MSE": 1.3e7,
                "Train_RMSE": 3670.0,
                "Train_MAE": 2690.0,
                "Train_MAPE": 1.24,
                "Train_R2": 0.9981,
                "Val_RMSE": 5600.0,
                "Val_MAPE": 7.07,
                "Val_R2": 0.9912,
                "Train_Time_Sec": 5.5,
                "Pred_Time_Sec": 0.12,
                "Device": "cuda",
            },
            {
                "Model": "LSTM",
                "MSE": 4.3e7,
                "RMSE": 6570.0,
                "MAE": 4520.0,
                "MAPE": 2.01,
                "R2": 0.9944,
                "Max_Error": 125000.0,
                "Mean_Residual": -360.0,
                "Std_Residual": 6560.0,
                "Train_MSE": 3.7e7,
                "Train_RMSE": 6100.0,
                "Train_MAE": 4070.0,
                "Train_MAPE": 1.96,
                "Train_R2": 0.9948,
                "Val_RMSE": 5968.0,
                "Val_MAPE": 7.17,
                "Val_R2": 0.9901,
                "Train_Time_Sec": 86.7,
                "Pred_Time_Sec": 0.77,
                "Device": "cuda",
            },
            {
                "Model": "VAR",
                "MSE": 5.0e7,
                "RMSE": 7122.0,
                "MAE": 4576.0,
                "MAPE": 2.00,
                "R2": 0.9935,
                "Max_Error": 130000.0,
                "Mean_Residual": -25.0,
                "Std_Residual": 7122.0,
                "Train_MSE": 4.5e7,
                "Train_RMSE": 6739.0,
                "Train_MAE": 4197.0,
                "Train_MAPE": 2.03,
                "Train_R2": 0.9937,
                "Val_RMSE": 6501.0,
                "Val_MAPE": 7.27,
                "Val_R2": 0.9882,
                "Train_Time_Sec": 3.5,
                "Pred_Time_Sec": 0.30,
                "Device": "cpu",
            },
        ]
    ).to_csv(metrics_dir / "holdout_test_metrics.csv", index=False)

    split_rows = []
    for m in ("VAR", "XGBoost", "LSTM"):
        for sp in ("Train (70%)", "Validation (15%)", "Test (15%)"):
            split_rows.append(
                {
                    "Model": m,
                    "Split": sp,
                    "MSE": 4.2e7,
                    "RMSE": 6500.0,
                    "MAE": 4300.0,
                    "MAPE": 2.0,
                    "R2": 0.994,
                    "Max_Error": 120000.0,
                    "Mean_Residual": -50.0,
                    "Std_Residual": 6500.0,
                }
            )
    pd.DataFrame(split_rows).to_csv(metrics_dir / "train_val_test_metrics.csv", index=False)

    cv_det_rows = []
    for fk in range(1, 6):
        for m in ("VAR", "XGBoost", "LSTM"):
            cv_det_rows.append(
                {
                    "Fold": fk,
                    "Model": m,
                    "Train_Size": 5845 * fk,
                    "Val_Size": 5844,
                    "MSE": 4.5e7,
                    "RMSE": 6700.0,
                    "MAE": 4400.0,
                    "MAPE": 2.5,
                    "R2": 0.991,
                    "Max_Error": 125000.0,
                    "Mean_Residual": -30.0,
                    "Std_Residual": 6700.0,
                    "Train_Time_Sec": 2.5,
                }
            )
    pd.DataFrame(cv_det_rows).to_csv(metrics_dir / "cv_5fold_detailed.csv", index=False)

    pd.DataFrame(
        [
            {
                "Model": m,
                "Folds": 5,
                "MSE_Mean": 4.5e7,
                "MSE_Std": 1.0e7,
                "RMSE_Mean": 6700.0,
                "RMSE_Std": 900.0,
                "MAE_Mean": 4400.0,
                "MAE_Std": 700.0,
                "MAPE_Mean": 2.5,
                "MAPE_Std": 0.5,
                "R2_Mean": 0.991,
                "R2_Std": 0.002,
            }
            for m in ("VAR", "LSTM", "XGBoost")
        ]
    ).to_csv(metrics_dir / "cv_5fold_summary.csv", index=False)

    pd.DataFrame(
        [
            {
                "Model": m,
                "Eval_Steps": 168,
                "MSE": 2.5e7,
                "RMSE": 5000.0,
                "MAE": 3600.0,
                "MAPE": 1.7,
                "R2": 0.994,
                "Max_Error": 18000.0,
                "Mean_Residual": 300.0,
                "Std_Residual": 4990.0,
            }
            for m in ("XGBoost", "VAR", "LSTM")
        ]
    ).to_csv(metrics_dir / "walk_forward_metrics.csv", index=False)

    pd.DataFrame(
        [
            {
                "Model": m,
                "With_Weather_RMSE": 6500.0,
                "Without_Weather_RMSE": 6900.0,
                "RMSE_Improvement_Pct": 5.8,
                "With_Weather_MAE": 4400.0,
                "Without_Weather_MAE": 4800.0,
                "With_Weather_MAPE": 2.0,
                "Without_Weather_MAPE": 2.2,
                "MAPE_Improvement_Pct": 9.0,
                "With_Weather_R2": 0.994,
                "Without_Weather_R2": 0.993,
            }
            for m in ("VAR", "XGBoost", "LSTM")
        ]
    ).to_csv(metrics_dir / "weather_ablation_metrics.csv", index=False)

    pems_rows = []
    for m_k, m_d in [
        ("DCRNN", "DCRNN (Graph Diffusion + GRU)"),
        ("Multi_Horizon_TCN", "Multi-Horizon TCN (Causal Conv1D)"),
        ("Multi_Horizon_LSTM", "Multi-Horizon LSTM (Temporal Only)"),
    ]:
        for h_n, st, mn in [("15_min", 3, 15), ("30_min", 6, 30), ("60_min", 12, 60), ("Average_All_Horizons", 7, 35)]:
            pems_rows.append(
                {
                    "Model": m_k,
                    "Model_Display": m_d,
                    "Horizon": h_n,
                    "Steps_Ahead": st,
                    "Minutes_Ahead": mn,
                    "MSE": 10.0,
                    "RMSE": 3.16,
                    "MAE": 1.5,
                    "MAPE": 3.0,
                    "R2": 0.88,
                    "Max_Error": 45.0,
                    "Mean_Residual": -0.05,
                    "Std_Residual": 3.16,
                    "Train_Time_Sec": 20.0,
                }
            )
    pd.DataFrame(pems_rows).to_csv(metrics_dir / "pems_bay_benchmark_comparison.csv", index=False)

    ts_168 = pd.date_range("2014-05-26 21:00:00", periods=200, freq="1h")
    rng = np.random.default_rng(42)
    actual_arr = rng.normal(240000.0, 30000.0, size=200)
    pd.DataFrame(
        {
            "timestamp": ts_168,
            "Actual_Load": actual_arr,
            "VAR_Residual": rng.normal(0, 7000, size=200),
            "XGBoost_Residual": rng.normal(0, 6400, size=200),
            "LSTM_Residual": rng.normal(0, 6500, size=200),
        }
    ).to_csv(metrics_dir / "test_predictions_detailed.csv", index=False)

    pd.DataFrame(
        {
            "timestamp": ts_168[:168],
            "Actual_Load": actual_arr[:168],
            "VAR_WalkForward": actual_arr[:168] - rng.normal(0, 5000, size=168),
            "XGBoost_WalkForward": actual_arr[:168] - rng.normal(0, 4800, size=168),
            "LSTM_WalkForward": actual_arr[:168] - rng.normal(0, 5500, size=168),
        }
    ).to_csv(metrics_dir / "walk_forward_predictions.csv", index=False)

    feat_names = [f"load_lag_{i}h" for i in range(1, 25)] + [f"temperature_lag_{i}h" for i in range(1, 25)] + [
        f"humidity_lag_{i}h" for i in range(1, 25)
    ] + [f"wind_speed_lag_{i}h" for i in range(1, 25)] + [
        "load_roll_mean_6h",
        "load_roll_std_6h",
        "load_roll_mean_24h",
        "load_roll_std_24h",
        "sin_hour",
        "cos_hour",
        "sin_dayofweek",
        "cos_dayofweek",
        "is_weekend",
    ]
    imps = np.linspace(0.5, 0.001, len(feat_names))
    imps = imps / imps.sum()
    pd.DataFrame({"Feature": feat_names, "Importance": imps}).to_csv(
        metrics_dir / "xgboost_feature_importance.csv", index=False
    )

    corr_df = pd.DataFrame(
        np.eye(4),
        index=["load", "temperature", "humidity", "wind_speed"],
        columns=["load", "temperature", "humidity", "wind_speed"],
    )
    corr_df.to_csv(metrics_dir / "eda_pearson_correlation.csv")
    pd.DataFrame(
        [
            {"Variable": c, "Std": s}
            for c, s in [("load", 83328.0), ("temperature", 8.25), ("humidity", 14.62), ("wind_speed", 1.16)]
        ]
    ).to_csv(metrics_dir / "eda_descriptive_stats.csv", index=False)

    (metrics_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "hardware": {"device_name": "Mock CPU/GPU"},
                "models": {
                    "VAR": {"selected_lag": 24, "train_time_sec": 3.5, "device": "cpu"},
                    "XGBoost": {"best_iteration": 449, "train_time_sec": 5.5, "device": "cuda"},
                    "LSTM": {
                        "history": {
                            "train_loss": [0.02, 0.01],
                            "val_loss": [0.015, 0.008],
                            "train_mae": [0.09, 0.05],
                            "val_mae": [0.07, 0.04],
                            "best_epoch": 2,
                            "best_val_loss": 0.008,
                            "total_params": 30589,
                            "epochs_executed": 2,
                        },
                        "train_time_sec": 86.7,
                        "device": "cuda",
                    },
                    "DCRNN": {"history": {"train_loss": [0.1, 0.09], "val_loss": [0.25, 0.23], "best_epoch": 2, "best_val_loss": 0.23, "total_params": 21290}, "device": "cuda"},
                    "Multi_Horizon_TCN": {"history": {"train_loss": [0.1, 0.09], "val_loss": [0.26, 0.25], "best_epoch": 2, "best_val_loss": 0.25, "total_params": 3370}, "device": "cuda"},
                    "Multi_Horizon_LSTM": {"history": {"train_loss": [0.1, 0.08], "val_loss": [0.24, 0.22], "best_epoch": 2, "best_val_loss": 0.22, "total_params": 110292}, "device": "cuda"},
                },
            }
        ),
        encoding="utf-8",
    )
    (metrics_dir / "adf_stationarity_report.json").write_text("{}", encoding="utf-8")
    (progress_dir / "latest_run.json").write_text(json.dumps({"run_id": run_id, "stages": {}}), encoding="utf-8")

    return {
        "models_dir": models_dir,
        "metrics_dir": metrics_dir,
        "data_processed_dir": proc_dir,
        "progress_dir": progress_dir,
        "logs_dir": logs_dir,
    }


class Test05ModelSummaryRegistry(unittest.TestCase):
    r"""
    Purpose: Kiểm tra tính chính xác toán học và tính đầy đủ của toàn bộ hệ thống xuất tự động
             Model Summary (`reports/model_summary/lab04/`) cho cả 2 bộ dữ liệu của Lab 04.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_config()
        real_metrics = get_path("metrics_dir", cls.cfg)
        real_models = get_path("models_dir", cls.cfg)
        cls.use_real_artifacts = (real_metrics / "holdout_test_metrics.csv").exists() and (
            real_models / "var_model.pkl"
        ).exists()
        cls._tmp_ctx = None

        if cls.use_real_artifacts:
            cls.active_root = PROJECT_ROOT
            cls.models_dir = real_models
            cls.metrics_dir = real_metrics
            cls.proc_dir = get_path("data_processed_dir", cls.cfg)
            cls.summary_root = PROJECT_ROOT / "reports" / "model_summary" / "lab04"
            cls.summary_info = generate_all_model_summaries()
        else:
            cls._tmp_ctx = tempfile.TemporaryDirectory(prefix="lab04_ms_test_")
            tmp_root = Path(cls._tmp_ctx.name)
            path_map = _create_synthetic_lab04_workspace(tmp_root)
            cls.active_root = tmp_root
            cls.models_dir = path_map["models_dir"]
            cls.metrics_dir = path_map["metrics_dir"]
            cls.proc_dir = path_map["data_processed_dir"]
            cls.summary_root = tmp_root / "reports" / "model_summary" / "lab04"

            def _mock_get_path(key: str, config: dict = None) -> Path:
                return path_map.get(key, tmp_root / key)

            with (
                patch.object(ms_mod, "PROJECT_ROOT", tmp_root),
                patch.object(ms_mod, "get_path", side_effect=_mock_get_path),
                patch.object(ms_mod, "ensure_directories", return_value=path_map),
                patch.object(ms_mod, "get_active_run_id", return_value="20261117_000000_test01"),
                patch.object(ms_mod, "record_stage_result", return_value=tmp_root / "run.json"),
                patch.object(ms_mod, "write_log", return_value=None),
            ):
                cls.summary_info = generate_all_model_summaries()

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._tmp_ctx is not None:
            cls._tmp_ctx.cleanup()

    def test_01_residual_diagnostics_math_and_white_noise(self) -> None:
        r"""Kiểm tra hàm chẩn đoán phần dư tính đúng Ljung-Box Q, JB, H, Durbin-Watson và phân biệt nhiễu trắng."""
        rng = np.random.default_rng(42)
        white_noise = rng.normal(0.0, 1.0, size=500)
        diag_wn = compute_detailed_residual_diagnostics(white_noise, lag=24)

        self.assertEqual(diag_wn["n_obs"], 500)
        self.assertEqual(diag_wn["ljung_box_lag"], 24)
        self.assertGreater(diag_wn["ljung_box_p"], 0.05)
        self.assertTrue(diag_wn["is_white_noise"])
        self.assertEqual(diag_wn["white_noise_label"], "Pass (Đạt)")
        self.assertAlmostEqual(diag_wn["durbin_watson"], 2.0, delta=0.25)

        ar1_resid = np.zeros(500, dtype=np.float64)
        for t in range(1, 500):
            ar1_resid[t] = 0.85 * ar1_resid[t - 1] + white_noise[t]
        diag_ar = compute_detailed_residual_diagnostics(ar1_resid, lag=24)
        self.assertLess(diag_ar["ljung_box_p"], 0.001)
        self.assertFalse(diag_ar["is_white_noise"])
        self.assertEqual(diag_ar["white_noise_label"], "Fail (Không đạt)")
        self.assertLess(diag_ar["durbin_watson"], 0.6)

    def test_02_real_model_weights_and_parameters_extraction(self) -> None:
        r"""Kiểm tra trích xuất trực tiếp tham số từ các tệp .pkl và .pt trong models/lab04/."""
        var_details = _extract_var_statistical_table(
            var_pkl_path=self.models_dir / "var_model.pkl",
            train_mse_load=45418594.3215,
            n_train_obs=24545,
            eda_corr_path=self.metrics_dir / "eda_pearson_correlation.csv",
            eda_stats_path=self.metrics_dir / "eda_descriptive_stats.csv",
            processed_bundle_path=self.proc_dir / "processed_bundle.pkl",
        )
        self.assertEqual(var_details["selected_lag"], 24)
        self.assertEqual(var_details["num_equations"], 4)
        self.assertEqual(var_details["params_per_equation"], 97)
        self.assertEqual(var_details["total_system_params"], 388)
        self.assertEqual(len(var_details["all_coefficients"]), 97)
        self.assertEqual(len(var_details["cross_variable_summary"]), 4)

        xgb_details = _extract_xgboost_details(
            xgb_pkl_path=self.models_dir / "xgboost_model.pkl",
            imp_csv_path=self.metrics_dir / "xgboost_feature_importance.csv",
        )
        self.assertEqual(xgb_details["total_features"], 106)
        self.assertEqual(len(xgb_details["all_features"]), 106)
        self.assertGreater(len(xgb_details["feature_group_summary"]), 0)

        if self.use_real_artifacts:
            self.assertGreater(xgb_details["total_trees_evaluated"], 0)
            self.assertLessEqual(xgb_details["total_trees_evaluated"], 450)
            lstm_st = _extract_pytorch_state_dict_stats(self.models_dir / "lstm_model.pt")
            self.assertEqual(lstm_st["trainable_params"], 30589)
            self.assertEqual(len(lstm_st["layers"]), 14)
            self.assertIn("highway.weight", lstm_st["special_weights"])
            self.assertEqual(len(lstm_st["special_weights"]["highway.weight"][0]), 27)

            dcrnn_st = _extract_pytorch_state_dict_stats(self.models_dir / "dcrnn_pems_bay.pt")
            self.assertEqual(dcrnn_st["trainable_params"], 21290)
            self.assertEqual(dcrnn_st["buffer_params"], 214245)

            tcn_st = _extract_pytorch_state_dict_stats(self.models_dir / "tcn_pems_bay.pt")
            self.assertEqual(tcn_st["trainable_params"], 3370)

            pems_lstm_st = _extract_pytorch_state_dict_stats(self.models_dir / "lstm_pems_bay.pt")
            self.assertEqual(pems_lstm_st["trainable_params"], 110292)

    def test_03_all_36_models_cataloged_without_omission(self) -> None:
        r"""Đảm bảo đủ 36 thực thể mô hình (24 Dataset 1 + 12 Dataset 2) không sót bất kỳ cấu hình nào."""
        self.assertEqual(self.summary_info["total_models_cataloged"], 36)
        self.assertEqual(self.summary_info["dataset1_models_count"], 24)
        self.assertEqual(self.summary_info["dataset2_models_count"], 12)

        d1_df = pd.read_csv(self.summary_root / "dataset1" / "dataset1_model_summary.csv")
        d2_df = pd.read_csv(self.summary_root / "dataset2" / "dataset2_model_summary.csv")
        master_df = pd.read_csv(self.summary_root / "lab04_model_summary.csv")

        self.assertEqual(len(d1_df), 24)
        self.assertEqual(len(d2_df), 12)
        self.assertEqual(len(master_df), 36)

        d1_regimes = d1_df["Regime"].value_counts().to_dict()
        self.assertEqual(d1_regimes.get("Hold-out_Test_15pct"), 3)
        self.assertEqual(d1_regimes.get("Ablation_No_Weather"), 3)
        self.assertEqual(d1_regimes.get("Walk_Forward_168h"), 3)
        for k in range(1, 6):
            self.assertEqual(d1_regimes.get(f"5Fold_CV_Fold_{k}"), 3)

        d2_families = d2_df["Family"].value_counts().to_dict()
        self.assertEqual(d2_families.get("DCRNN"), 4)
        self.assertEqual(d2_families.get("Multi_Horizon_TCN"), 4)
        self.assertEqual(d2_families.get("Multi_Horizon_LSTM"), 4)

    def test_04_csv_and_json_files_validity_and_no_nan(self) -> None:
        r"""Kiểm tra toàn bộ các tệp .csv và .json hợp lệ, không chứa NaN/Inf ở 5 chỉ số đánh giá."""
        run_id = self.summary_info["run_id"]
        expected_files = [
            self.summary_root / "lab04_model_summary.csv",
            self.summary_root / "lab04_model_summary.json",
            self.summary_root / "lab04_model_summary.md",
            self.summary_root / "runs" / f"experiment_lab04_{run_id}_summary.csv",
            self.summary_root / "runs" / f"experiment_lab04_{run_id}_summary.json",
            self.summary_root / "runs" / f"experiment_lab04_{run_id}_summary.md",
            self.summary_root / "dataset1" / "dataset1_model_summary.csv",
            self.summary_root / "dataset1" / "dataset1_model_summary.json",
            self.summary_root / "dataset1" / "dataset1_model_summary.md",
            self.summary_root / "dataset1" / "runs" / f"experiment_dataset1_{run_id}_summary.csv",
            self.summary_root / "dataset1" / "runs" / f"experiment_dataset1_{run_id}_summary.json",
            self.summary_root / "dataset1" / "runs" / f"experiment_dataset1_{run_id}_summary.md",
            self.summary_root / "dataset2" / "dataset2_model_summary.csv",
            self.summary_root / "dataset2" / "dataset2_model_summary.json",
            self.summary_root / "dataset2" / "dataset2_model_summary.md",
            self.summary_root / "dataset2" / "runs" / f"experiment_dataset2_{run_id}_summary.csv",
            self.summary_root / "dataset2" / "runs" / f"experiment_dataset2_{run_id}_summary.json",
            self.summary_root / "dataset2" / "runs" / f"experiment_dataset2_{run_id}_summary.md",
        ]
        for fp in expected_files:
            self.assertTrue(fp.exists(), f"Thiếu tệp model_summary: {fp}")
            self.assertGreater(fp.stat().st_size, 500, f"Tệp rỗng hoặc quá nhỏ: {fp}")

        master_df = pd.read_csv(self.summary_root / "lab04_model_summary.csv")
        required_cols = ["MSE", "RMSE", "MAE", "MAPE (%)", "R²", "Max_Error", "Resid_Mean", "Resid_Std", "Durbin_Watson"]
        for col in required_cols:
            self.assertIn(col, master_df.columns)
            self.assertFalse(master_df[col].isna().any(), f"Cột {col} chứa giá trị NaN")
            self.assertTrue(np.all(np.isfinite(master_df[col].to_numpy(dtype=np.float64))), f"Cột {col} chứa Inf")

        d1_json = json.loads((self.summary_root / "dataset1" / "dataset1_model_summary.json").read_text(encoding="utf-8"))
        d2_json = json.loads((self.summary_root / "dataset2" / "dataset2_model_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(d1_json["total_candidates"], 24)
        self.assertEqual(len(d1_json["models"]), 24)
        self.assertEqual(d2_json["total_candidates"], 12)
        self.assertEqual(len(d2_json["models"]), 12)

    def test_05_markdown_cards_and_metric_synchronization(self) -> None:
        r"""Kiểm tra các tệp Markdown có đủ 24 thẻ Dataset 1 + 12 thẻ Dataset 2 và khớp số liệu với metrics gốc."""
        d1_md = (self.summary_root / "dataset1" / "dataset1_model_summary.md").read_text(encoding="utf-8")
        d2_md = (self.summary_root / "dataset2" / "dataset2_model_summary.md").read_text(encoding="utf-8")

        for i in range(1, 25):
            self.assertIn(f"### 3.{i}. ", d1_md, f"Thiếu thẻ mô hình 3.{i} trong dataset1_model_summary.md")
        for i in range(1, 13):
            self.assertIn(f"### 3.{i}. ", d2_md, f"Thiếu thẻ mô hình 3.{i} trong dataset2_model_summary.md")

        for kw in (
            "Ljung-Box (L=24) (Q):",
            "Jarque-Bera (JB):",
            "Heteroskedasticity(H):",
            "Durbin-Watson (DW):",
            "FULL PARAMETER ESTIMATES FOR TARGET EQUATION (Delta_load_t — All 97 Parameters):",
            "COMPLETE FEATURE IMPORTANCE RANKING (All 106/106 Features Extracted):",
        ):
            self.assertIn(kw, d1_md)

        if self.use_real_artifacts:
            self.assertIn("LAYER-BY-LAYER TENSOR PARAMETER ANATOMY (Extracted from lstm_model.pt):", d1_md)
            for kw in (
                "LAYER-BY-LAYER TENSOR PARAMETER ANATOMY (Extracted from dcrnn_pems_bay.pt):",
                "LAYER-BY-LAYER TENSOR PARAMETER ANATOMY (Extracted from tcn_pems_bay.pt):",
                "LAYER-BY-LAYER TENSOR PARAMETER ANATOMY (Extracted from lstm_pems_bay.pt):",
                "MULTI-HORIZON SPATIO-TEMPORAL EVALUATION BREAKDOWN",
            ):
                self.assertIn(kw, d2_md)


if __name__ == "__main__":
    unittest.main(verbosity=2)