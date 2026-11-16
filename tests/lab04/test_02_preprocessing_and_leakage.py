r"""
Kiểm thử Mục 3, Mục 4 & Mục 5: Phân chia Tuần tự Chống Rò rỉ Dữ liệu (Anti-Data-Leakage),
Kiểm định Dừng ADF, Cửa sổ Trượt 24h (105 Đặc trưng XGBoost & Tensor 3D LSTM) và 5-Fold TimeSeriesSplit
(tests/lab04/test_02_preprocessing_and_leakage.py).
"""

import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lab04.preprocess import (  # noqa: E402
    build_sequence_windows_3d,
    build_tabular_lag_features,
    build_time_series_cv_folds,
    chronological_split_dataframe,
    ensure_stationarity_for_var,
    prepare_pems_bay_tensors,
)
from src.lab04.utils import load_config, set_global_seed  # noqa: E402


class Test02PreprocessingAndLeakage(unittest.TestCase):
    r"""
    Purpose: Kiểm chứng nghiêm ngặt quy tắc chống rò rỉ dữ liệu tương lai (Rule 7),
             kiểm định nghiệm đơn vị ADF không sinh FutureWarning, và kích thước các tensor cửa sổ trượt.
    """

    def setUp(self) -> None:
        set_global_seed(42)
        self.cfg = load_config()
        rng = np.random.default_rng(42)
        n = 600
        idx = pd.date_range("2013-01-01 00:00:00", periods=n, freq="1h")
        hours = idx.hour.to_numpy(dtype=np.float64)
        load = 210000.0 + 35000.0 * np.sin(2.0 * np.pi * hours / 24.0) + np.cumsum(rng.normal(0, 120.0, size=n))
        temp = 19.0 + 5.0 * np.sin(2.0 * np.pi * (hours - 9.0) / 24.0) + rng.normal(0, 0.8, size=n)
        hum = 68.0 - 8.0 * np.sin(2.0 * np.pi * (hours - 9.0) / 24.0) + rng.normal(0, 1.5, size=n)
        wind = 4.2 + 0.8 * np.cos(2.0 * np.pi * hours / 24.0) + rng.normal(0, 0.4, size=n)
        self.df = pd.DataFrame(
            {"load": load, "temperature": temp, "humidity": hum, "wind_speed": wind},
            index=idx,
        )

    def test_01_chronological_split_no_leakage(self) -> None:
        r"""Kiểm tra chia 70% Train - 15% Val - 15% Test giữ nguyên thứ tự thời gian (train_end < val_start < test_start)."""
        tr_df, va_df, te_df, meta = chronological_split_dataframe(self.df, train_ratio=0.70, val_ratio=0.15)
        self.assertEqual(len(tr_df), 420)
        self.assertEqual(len(va_df), 90)
        self.assertEqual(len(te_df), 90)
        self.assertFalse(meta["shuffle"])
        self.assertTrue(meta["temporal_leakage_check"])
        self.assertLess(tr_df.index.max(), va_df.index.min())
        self.assertLess(va_df.index.max(), te_df.index.min())

    def test_02_standard_scaler_fit_on_train_only(self) -> None:
        r"""Kiểm tra StandardScaler chỉ fit trên tập Train (trung bình = 0, độ lệch chuẩn = 1 trên Train, khác 0 trên Test)."""
        tr_df, va_df, te_df, _ = chronological_split_dataframe(self.df, 0.70, 0.15)
        scaler = StandardScaler()
        tr_s = scaler.fit_transform(tr_df.values)
        te_s = scaler.transform(te_df.values)

        np.testing.assert_allclose(tr_s.mean(axis=0), np.zeros(4), atol=1e-6)
        np.testing.assert_allclose(tr_s.std(axis=0), np.ones(4), atol=1e-6)
        self.assertFalse(np.allclose(te_s.mean(axis=0), np.zeros(4), atol=1e-3))

    def test_03_adf_stationarity_zero_warnings(self) -> None:
        r"""Kiểm tra kiểm định ADF không phát sinh FutureWarning và chuỗi sau sai phân bậc 1 dừng tuyệt đối (p < 0.05)."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", category=FutureWarning)
            rep = ensure_stationarity_for_var(self.df, significance=0.05)
            future_warns = [w for w in caught if issubclass(w.category, FutureWarning)]
            self.assertEqual(len(future_warns), 0, f"Phát hiện FutureWarning: {future_warns}")

        self.assertEqual(rep["diff_order_applied"], 1)
        self.assertTrue(rep["all_stationary_after_diff1"])
        for item in rep["adf_after_diff1"]:
            self.assertLess(item["p_value"], 0.05)

    def test_04_sliding_window_105_features_and_3d_sequence(self) -> None:
        r"""Kiểm tra tạo cửa sổ trượt lookback=24h cho đúng 105 đặc trưng XGBoost và tensor (N-24, 24, 4) cho LSTM."""
        scaler = StandardScaler()
        df_s = pd.DataFrame(scaler.fit_transform(self.df.values), index=self.df.index, columns=self.df.columns)

        X_tab, y_tab, ts_idx, feat_names = build_tabular_lag_features(df_s, self.df, lookback=24)
        X_seq, y_seq = build_sequence_windows_3d(df_s, lookback=24)

        expected_n = len(self.df) - 24
        self.assertEqual(X_tab.shape, (expected_n, 105))
        self.assertEqual(len(feat_names), 105)
        self.assertEqual(X_seq.shape, (expected_n, 24, 4))
        self.assertEqual(len(ts_idx), expected_n)
        np.testing.assert_allclose(y_tab, y_seq, atol=1e-6)

    def test_05_5fold_time_series_split_isolation(self) -> None:
        r"""Kiểm tra 5-Fold TimeSeriesSplit có tập Train mở rộng dần và mỗi Fold có bộ chuẩn hóa độc lập."""
        folds = build_time_series_cv_folds(self.df, n_splits=5, lookback=24)
        self.assertEqual(len(folds), 5)
        prev_train_size = 0
        for f in folds:
            self.assertGreater(f["train_size"], prev_train_size)
            prev_train_size = f["train_size"]
            self.assertEqual(f["X_xgb_val"].shape[0], len(f["y_val_orig"]))
            self.assertEqual(f["X_lstm_val"].shape, (len(f["y_val_orig"]), 24, 4))

    def test_06_pems_bay_multi_horizon_tensors(self) -> None:
        r"""Kiểm tra cắt cửa sổ đa tầm nhìn [t+3, t+6, t+12] trên dữ liệu giao thông PEMS-BAY."""
        rng = np.random.default_rng(42)
        df_speed = pd.DataFrame(
            rng.uniform(45.0, 68.0, size=(200, 10)).astype(np.float32),
            index=pd.date_range("2017-01-01", periods=200, freq="5min"),
        )
        adj_mx = np.eye(10, dtype=np.float32)
        pems_bundle = prepare_pems_bay_tensors(df_speed, adj_mx, self.cfg)

        self.assertEqual(pems_bundle["X_train"].shape[1:], (12, 10))
        self.assertEqual(pems_bundle["Y_train_scaled"].shape[1:], (3, 10))
        self.assertEqual(pems_bundle["horizon_steps"], [3, 6, 12])


if __name__ == "__main__":
    unittest.main(verbosity=2)