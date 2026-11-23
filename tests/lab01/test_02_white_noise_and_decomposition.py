r"""Kiểm thử Bài 2 (White Noise) & Bài 3 (Classical/STL Decomposition) cho Lab 01."""

import unittest
from src.lab01.data_loader import load_airpassengers_dataset
from src.lab01.decomposition import run_decomposition_analysis
from src.lab01.eda import analyze_white_noise


class TestLab01WhiteNoiseAndDecomposition(unittest.TestCase):
    def test_white_noise_properties(self) -> None:
        r"""
        Purpose: Xác minh chuỗi White Noise 1000 điểm có p-value Ljung-Box > 0.05 và sinh đủ 5 dòng so sánh.
        Input-Output: Input: không; Output: Assertion pass.
        How: Gọi analyze_white_noise() với seed=42, fig_dir=None.
        """
        df, _ = load_airpassengers_dataset()
        res = analyze_white_noise(df["Passengers"].astype(float), n_points=1000, seed=42, fig_dir=None)
        self.assertTrue(res["is_white_noise_confirmed"])
        self.assertEqual(len(res["comparison_5_lines"]), 5)

    def test_decomposition_strength(self) -> None:
        r"""
        Purpose: Xác minh phân rã STL nhận diện đúng xu hướng mạnh (F_T > 0.9) và mùa vụ mạnh (F_S > 0.8).
        Input-Output: Input: không; Output: Assertion pass.
        How: Gọi run_decomposition_analysis(series, period=12, fig_dir=None).
        """
        df, _ = load_airpassengers_dataset()
        res = run_decomposition_analysis(df["Passengers"].astype(float), period=12, fig_dir=None)
        self.assertGreater(res["stl"]["trend_strength"], 0.9)
        self.assertGreater(res["stl"]["seasonal_strength"], 0.8)