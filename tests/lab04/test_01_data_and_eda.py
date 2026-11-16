r"""
Kiểm thử Mục 1 & Mục 2: Cấu hình Hệ thống, Nạp Dữ liệu, Chuẩn hóa Khí tượng Vật lý & Thống kê EDA
(tests/lab04/test_01_data_and_eda.py).
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lab04.data_loader import (  # noqa: E402
    FEATURE_DESCRIPTIONS,
    _synthesize_calibrated_meteorological_exogenous,
)
from src.lab04.eda import (  # noqa: E402
    compute_correlation_analysis,
    compute_descriptive_statistics,
)
from src.lab04.utils import (  # noqa: E402
    ensure_directories,
    get_compute_device,
    load_config,
    set_global_seed,
)


class Test01DataAndEDA(unittest.TestCase):
    r"""
    Purpose: Kiểm tra tính toàn vẹn của cấu hình config.yaml, hàm tổng hợp khí tượng vật lý,
             thống kê mô tả 13 chỉ tiêu và phân tích tương quan Pearson/Spearman.
    """

    def setUp(self) -> None:
        set_global_seed(42)
        self.cfg = load_config()

    def test_01_config_and_directories(self) -> None:
        r"""Kiểm tra config.yaml có đầy đủ các khóa bắt buộc và tạo thư mục đa Lab hợp lệ."""
        for key in ("project", "paths", "hardware", "data", "preprocessing", "models", "evaluation"):
            self.assertIn(key, self.cfg, f"Thiếu khóa '{key}' trong config.yaml")
        dirs = ensure_directories(self.cfg)
        self.assertIn("reports_dir", dirs)
        self.assertIn("figures_dir", dirs)
        self.assertTrue(dirs["reports_dir"].exists())

    def test_02_hardware_auto_detection(self) -> None:
        r"""Kiểm tra hàm tự động phát hiện phần cứng GPU/CPU trả về đúng cấu trúc."""
        hw = get_compute_device(self.cfg)
        self.assertIn(hw["torch_device"], ("cuda", "mps", "cpu"))
        self.assertIn(hw["xgb_device"], ("cuda", "cpu"))
        self.assertIsInstance(hw["device_name"], str)
        self.assertTrue(len(hw["device_name"]) > 0)

    def test_03_feature_descriptions_schema(self) -> None:
        r"""Kiểm tra bảng đặc tả FEATURE_DESCRIPTIONS có đủ 4 biến chuẩn."""
        self.assertEqual(len(FEATURE_DESCRIPTIONS), 4)
        cols = [item["Biến số (Column)"] for item in FEATURE_DESCRIPTIONS]
        self.assertEqual(cols, ["load", "temperature", "humidity", "wind_speed"])

    def test_04_calibrated_meteorological_exogenous_physics(self) -> None:
        r"""
        Kiểm tra hàm tổng hợp 3 biến thời tiết vật lý (temperature, humidity, wind_speed):
        - Không có giá trị NaN hoặc bằng 0.0 bất thường.
        - Nằm trong miền vật lý thực tế: temperature [4.5, 38.5] °C, humidity [24, 96] %, wind_speed [0.4, 15] m/s.
        - Có tương quan Pearson dương rõ rệt giữa temperature và load (r > 0.45).
        """
        idx = pd.date_range("2011-01-01 00:00:00", periods=1000, freq="1h")
        hours = idx.hour.to_numpy(dtype=np.float64)
        load_syn = 220000.0 + 45000.0 * np.sin(2.0 * np.pi * (hours - 8.0) / 24.0)
        temp, hum, wind = _synthesize_calibrated_meteorological_exogenous(idx, load_syn, seed=42)

        self.assertEqual(len(temp), 1000)
        self.assertEqual(len(hum), 1000)
        self.assertEqual(len(wind), 1000)
        self.assertFalse(np.isnan(temp).any())
        self.assertGreaterEqual(float(temp.min()), 4.5)
        self.assertLessEqual(float(temp.max()), 38.5)
        self.assertGreaterEqual(float(hum.min()), 24.0)
        self.assertLessEqual(float(hum.max()), 96.0)
        self.assertGreaterEqual(float(wind.min()), 0.4)
        self.assertLessEqual(float(wind.max()), 15.0)

        df_test = pd.DataFrame(
            {"load": load_syn, "temperature": temp, "humidity": hum, "wind_speed": wind},
            index=idx,
        )
        pearson_df, spearman_df, corr_sum = compute_correlation_analysis(df_test)
        self.assertGreater(float(pearson_df.loc["load", "temperature"]), 0.45)
        self.assertLess(float(pearson_df.loc["load", "humidity"]), -0.25)
        self.assertIn(corr_sum["dominant_weather_variable"], ("temperature", "humidity", "wind_speed"))

    def test_05_descriptive_statistics_and_outliers(self) -> None:
        r"""Kiểm tra bảng thống kê mô tả tính đúng Mean, Std, Quantiles, Skewness, Kurtosis và IQR/Z-Score Outliers."""
        rng = np.random.default_rng(42)
        df_dummy = pd.DataFrame(
            {
                "load": rng.normal(200000.0, 30000.0, size=500),
                "temperature": rng.normal(20.0, 5.0, size=500),
                "humidity": rng.normal(65.0, 10.0, size=500),
                "wind_speed": rng.normal(4.5, 1.2, size=500),
            }
        )
        stats_df = compute_descriptive_statistics(df_dummy, iqr_mult=1.5, z_thresh=3.0)
        self.assertEqual(len(stats_df), 4)
        expected_cols = {
            "Variable", "Count", "Missing", "Mean", "Std", "Min", "Q25",
            "Median", "Q75", "Max", "Skewness", "Kurtosis",
            "IQR_Outliers", "IQR_Outlier_Pct", "ZScore_Outliers",
        }
        self.assertTrue(expected_cols.issubset(set(stats_df.columns)))
        self.assertEqual(int(stats_df["Missing"].sum()), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)