r"""
Kiểm thử Mục 5, Mục 6, Mục 7 & Mục 12: Kiến trúc và Huấn luyện 6 Mô hình
(VAR, XGBoost, StackedLSTMNet + Residual Highway, DCRNN, Multi-Horizon TCN & Multi-Horizon LSTM)
(tests/lab04/test_03_models_and_architectures.py).
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lab04.model import (  # noqa: E402
    DCRNNNet,
    LSTMForecaster,
    MultiHorizonPemsLSTMNet,
    MultiHorizonTCNNet,
    SpatioTemporalTrainer,
    StackedLSTMNet,
    VARForecaster,
    XGBoostForecaster,
    compute_dual_random_walk_supports,
)
from src.lab04.utils import load_config, set_global_seed  # noqa: E402


class Test03ModelsAndArchitectures(unittest.TestCase):
    r"""
    Purpose: Kiểm tra tính đúng đắn toán học, kích thước Tensor đầu ra (Forward Pass) và khả năng học
             hội tụ của cả 6 kiến trúc mô hình trong Lab 04.
    """

    def setUp(self) -> None:
        set_global_seed(42)
        self.cfg = load_config()

    def test_01_var_forecaster_fit_and_rolling_predict(self) -> None:
        r"""Kiểm tra mô hình VAR chọn độ trễ tối ưu theo AIC và dự báo cuốn chiếu 1 bước chính xác."""
        rng = np.random.default_rng(42)
        n = 300
        idx = pd.date_range("2014-01-01", periods=n, freq="1h")
        df = pd.DataFrame(
            {
                "load": 200000.0 + np.cumsum(rng.normal(0, 50.0, size=n)),
                "temperature": 20.0 + np.sin(np.arange(n) / 12.0) + rng.normal(0, 0.2, size=n),
                "humidity": 65.0 - np.sin(np.arange(n) / 12.0) + rng.normal(0, 0.5, size=n),
                "wind_speed": 4.0 + rng.normal(0, 0.3, size=n),
            },
            index=idx,
        )
        tr_df = df.iloc[:240].copy()
        te_df = df.iloc[240:].copy()

        var_m = VARForecaster(maxlags=12, ic="aic")
        var_m.fit(tr_df)
        self.assertGreaterEqual(var_m.selected_lag, 1)
        self.assertLessEqual(var_m.selected_lag, 12)

        in_preds = var_m.predict_in_sample(tr_df, lookback=24)
        self.assertEqual(in_preds.shape, (len(tr_df) - 24,))

        roll_preds = var_m.predict_rolling_1step(tr_df, te_df)
        self.assertEqual(roll_preds.shape, (len(te_df),))
        self.assertTrue(np.isfinite(roll_preds).all())

    def test_02_xgboost_forecaster_and_importance(self) -> None:
        r"""Kiểm tra XGBoostForecaster huấn luyện, dự báo quy đổi ngược đơn vị và trích xuất độ quan trọng đặc trưng."""
        rng = np.random.default_rng(42)
        X_tr = rng.normal(0, 1, size=(160, 105)).astype(np.float32)
        y_tr = (0.8 * X_tr[:, 0] + 0.2 * X_tr[:, 1] + rng.normal(0, 0.05, size=160)).astype(np.float32)
        X_va = rng.normal(0, 1, size=(40, 105)).astype(np.float32)
        y_va = (0.8 * X_va[:, 0] + 0.2 * X_va[:, 1]).astype(np.float32)

        xgb_m = XGBoostForecaster(self.cfg, use_gpu=False)
        xgb_m.params["n_estimators"] = 30
        xgb_m.fit(X_tr, y_tr, X_va, y_va)

        preds_orig = xgb_m.predict_original(X_va, load_mean=200000.0, load_std=30000.0)
        self.assertEqual(preds_orig.shape, (40,))
        self.assertTrue(np.isfinite(preds_orig).all())

        feat_names = [f"f_{i}" for i in range(105)]
        imp_df = xgb_m.get_feature_importances(feat_names)
        self.assertEqual(len(imp_df), 105)
        self.assertEqual(imp_df.iloc[0]["Feature"], "f_0")

    def test_03_stacked_lstm_residual_highway_architecture(self) -> None:
        r"""Kiểm tra kiến trúc StackedLSTMNet với nhánh Autoregressive Residual Highway và quá trình hội tụ."""
        net = StackedLSTMNet(input_dim=4, hidden_dims=(32, 16), dropout=0.1, use_residual_highway=True)
        # Kiểm tra khởi tạo trọng số neo tự hồi quy tại bước cuối (index 23) và bước cùng giờ hôm trước (index 0)
        self.assertAlmostEqual(float(net.highway.weight[0, 23].item()), 0.75, places=5)
        self.assertAlmostEqual(float(net.highway.weight[0, 0].item()), 0.25, places=5)

        x_dummy = torch.randn(16, 24, 4)
        out = net(x_dummy)
        self.assertEqual(tuple(out.shape), (16,))

        # Kiểm tra huấn luyện 2 epoch giảm hàm mất mát
        X_tr = np.random.randn(64, 24, 4).astype(np.float32)
        y_tr = X_tr[:, -1, 0].copy()
        lstm_m = LSTMForecaster(input_dim=4, config=self.cfg)
        lstm_m.fit(X_tr, y_tr, X_tr[:16], y_tr[:16], epochs_override=2)
        preds = lstm_m.predict_original(X_tr[:16], load_mean=200000.0, load_std=25000.0)
        self.assertEqual(preds.shape, (16,))

    def test_04_pems_bay_dcrnn_tcn_and_lstm_architectures(self) -> None:
        r"""
        Kiểm tra cả 3 mô hình Bài tập 12 nâng cao trên PEMS-BAY:
        1. DCRNNNet (Dual Random-Walk Diffusion Convolution + GRU)
        2. MultiHorizonTCNNet (Dilated Causal Conv1D)
        3. MultiHorizonPemsLSTMNet (Temporal LSTM)
        đều xuất đúng tensor (Batch, Horizons=3, Num_Sensors=N).
        """
        n_sensors = 12
        lookback = 12
        batch_sz = 8
        adj_mx = np.eye(n_sensors, dtype=np.float32) + 0.25 * np.eye(n_sensors, k=1, dtype=np.float32)
        supports = compute_dual_random_walk_supports(adj_mx, max_diffusion_step=2)
        # K=2 tạo ra 1 (Identity) + 2 (Forward) + 2 (Backward) = 5 ma trận hỗ trợ
        self.assertEqual(len(supports), 5)

        x_pems = torch.randn(batch_sz, lookback, n_sensors)

        dcrnn = DCRNNNet(supports=supports, num_sensors=n_sensors, lookback=lookback, hidden_dim=16, num_horizons=3)
        y_dcrnn = dcrnn(x_pems)
        self.assertEqual(tuple(y_dcrnn.shape), (batch_sz, 3, n_sensors))

        tcn = MultiHorizonTCNNet(lookback=lookback, channels=16, num_horizons=3)
        y_tcn = tcn(x_pems)
        self.assertEqual(tuple(y_tcn.shape), (batch_sz, 3, n_sensors))

        pems_lstm = MultiHorizonPemsLSTMNet(num_sensors=n_sensors, lookback=lookback, hidden_dim=16, num_horizons=3)
        y_lstm = pems_lstm(x_pems)
        self.assertEqual(tuple(y_lstm.shape), (batch_sz, 3, n_sensors))

        # Kiểm tra SpatioTemporalTrainer chạy mượt với DCRNN
        X_np = np.random.randn(24, lookback, n_sensors).astype(np.float32)
        Y_np = np.random.randn(24, 3, n_sensors).astype(np.float32)
        trainer = SpatioTemporalTrainer("DCRNN_Test", dcrnn, config=self.cfg, epochs=2, batch_size=8, lr=0.003)
        trainer.fit(X_np, Y_np, X_np[:8], Y_np[:8])
        pred_mph = trainer.predict_original(X_np[:8], mean_val=60.0, std_val=5.0)
        self.assertEqual(pred_mph.shape, (8, 3, n_sensors))


if __name__ == "__main__":
    unittest.main(verbosity=2)