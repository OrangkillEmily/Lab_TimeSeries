r"""Kiểm thử Pipeline mới, progress_results/lab02/ và 2 bộ báo cáo riêng của Lab 02."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.lab02 import data_loader
from src.lab02.pipeline import run_full_pipeline
from src.lab02.utils import get_path, load_config


class TestLab02PipelineAndReports(unittest.TestCase):
    def test_01_config_and_data_loaders(self) -> None:
        r"""
        Purpose: Xác minh nạp đúng số quan sát cho cả dataset1 (204) và dataset2 (144) qua src/lab02/data_loader.py.
        Input-Output: Input: không; Output: Assertion pass.
        How: Gọi data_loader.load_dataset() cho cả 2 dataset.
        """
        cfg = load_config()
        df1, m1 = data_loader.load_dataset("dataset1", cfg)
        df2, m2 = data_loader.load_dataset("dataset2", cfg)
        self.assertEqual(len(df1), 204)
        self.assertEqual(len(df2), 144)
        self.assertEqual(m1["missing_values"], 0)
        self.assertEqual(m2["missing_values"], 0)

    def test_02_pipeline_standard_and_two_reports(self) -> None:
        r"""
        Purpose: Xác minh pipeline chế độ standard sinh progress_results/lab02/latest_run.json và đủ 2 bộ báo cáo tách riêng.
        Input-Output: Input: không; Output: Assertion pass.
        How: Chạy run_full_pipeline(mode='standard', save_figures=False) và kiểm tra sự tồn tại các tệp đầu ra.
        """
        cfg = load_config()
        _ = run_full_pipeline(datasets=["dataset1", "dataset2"], mode="standard", save_figures=False)
        self.assertTrue((get_path("progress_dir", cfg) / "latest_run.json").exists())
        rep_dir = get_path("reports_dir", cfg)
        for prefix in ("LAB_02_DATASET1_REPORT", "LAB_02_DATASET2_REPORT"):
            self.assertTrue((rep_dir / f"{prefix}.html").exists())
            self.assertTrue((rep_dir / f"{prefix}.md").exists())
            self.assertTrue((rep_dir / f"{prefix}.pdf").exists())


if __name__ == "__main__":
    unittest.main()