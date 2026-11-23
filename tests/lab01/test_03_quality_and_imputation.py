r"""Kiểm thử Bài 4 (Data Quality: 11 NaN + 3 Outlier giả lập & khôi phục) cho Lab 01."""

import unittest
from src.lab01.data_loader import load_airpassengers_dataset
from src.lab01.quality import run_quality_pipeline


class TestLab01QualityAndImputation(unittest.TestCase):
    def test_imputation_and_outlier_cleaning(self) -> None:
        r"""
        Purpose: Xác minh điền khuyết sạch 0 NaN và phương pháp STL phát hiện đúng 3/3 outlier giả lập.
        Input-Output: Input: không; Output: Assertion pass.
        How: Gọi run_quality_pipeline() với fig_dir=None và kiểm tra missing_remaining_after_clean, tp=3.
        """
        df, _ = load_airpassengers_dataset()
        res = run_quality_pipeline(df["Passengers"].astype(float), fig_dir=None, proc_dir=None)
        self.assertEqual(res["imputation"]["missing_simulated_count"], 11)
        self.assertEqual(res["imputation"]["missing_after_interp"], 0)
        self.assertEqual(res["outliers"]["missing_remaining_after_clean"], 0)
        self.assertEqual(res["outliers"]["methods_comparison"]["STL_Log_Robust_MAD"]["tp"], 3)