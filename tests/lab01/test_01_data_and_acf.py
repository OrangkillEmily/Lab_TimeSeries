r"""Kiểm thử Giai đoạn 1 & 2 (Lab 01): Nạp dữ liệu AirPassengers và phân tích ACF/PACF 40 lags."""

import unittest
from src.lab01.data_loader import load_airpassengers_dataset
from src.lab01.eda import analyze_acf_pacf
from src.lab01.utils import load_config


class TestLab01DataAndACF(unittest.TestCase):
    def test_load_airpassengers(self) -> None:
        r"""
        Purpose: Xác minh AirPassengers.csv nạp đủ 144 quan sát, DatetimeIndex tăng dần và 0 giá trị thiếu.
        Input-Output: Input: không; Output: Assertion pass.
        How: Gọi load_airpassengers_dataset() và kiểm tra n_obs, missing_values, is_monotonic.
        """
        cfg = load_config()
        df, meta = load_airpassengers_dataset(cfg)
        self.assertEqual(len(df), 144)
        self.assertEqual(meta["missing_values"], 0)
        self.assertTrue(meta["is_monotonic"])

    def test_acf_pacf_40_lags(self) -> None:
        r"""
        Purpose: Xác minh ACF/PACF tính đủ 40 lags và nhận diện đúng tự tương quan mạnh + mùa vụ chu kỳ 12.
        Input-Output: Input: không; Output: Assertion pass.
        How: Chạy analyze_acf_pacf(series, nlags=40, fig_dir=None) và kiểm tra acf_lag1, has_seasonal_peaks_12.
        """
        df, _ = load_airpassengers_dataset()
        res = analyze_acf_pacf(df["Passengers"].astype(float), nlags=40, fig_dir=None)
        self.assertEqual(res["nlags"], 40)
        self.assertTrue(res["has_strong_autocorrelation"])
        self.assertTrue(res["has_seasonal_peaks_12"])