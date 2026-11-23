r"""Kiểm thử toàn trình Pipeline Lab 01: 15 biểu đồ gốc, log, progress_results và 3 báo cáo HTML/PDF/MD."""

import unittest
from src.lab01.pipeline import run_full_pipeline
from src.lab01.utils import get_path, load_config


class TestLab01PipelineAndReport(unittest.TestCase):
    def test_end_to_end_pipeline(self) -> None:
        r"""
        Purpose: Xác minh pipeline Lab 01 sinh đủ 15 biểu đồ, log, latest_run.json và 3 báo cáo HTML/PDF/MD.
        Input-Output: Input: không; Output: Assertion pass.
        How: Chạy run_full_pipeline() và kiểm tra sự tồn tại các tệp đầu ra.
        """
        cfg = load_config()
        res = run_full_pipeline()
        self.assertEqual(res["visualization"]["total_figures_count"], 15)
        self.assertTrue((get_path("progress_dir", cfg) / "latest_run.json").exists())
        self.assertTrue((get_path("reports_dir", cfg) / "LAB_01_REPORT.html").exists())
        self.assertTrue((get_path("reports_dir", cfg) / "LAB_01_REPORT.md").exists())
        self.assertTrue((get_path("reports_dir", cfg) / "LAB_01_REPORT.pdf").exists())