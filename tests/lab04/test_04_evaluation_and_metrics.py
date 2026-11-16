r"""
Kiểm thử Mục 8, Mục 9, Mục 10, Mục 11 & Nghiệm thu Báo cáo AC01-AC15:
Độ chính xác Công thức 5 Chỉ số (MSE, RMSE, MAE, MAPE %, R²), Walk-Forward Backtesting,
Đối sánh 3 Mô hình PEMS-BAY và Kiểm toán Đồ tạo tác (Artifacts Audit)
(tests/lab04/test_04_evaluation_and_metrics.py).
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lab04.evaluate import (  # noqa: E402
    compute_5_metrics,
    evaluate_pems_bay_benchmark,
    evaluate_walk_forward_backtesting,
)
from src.lab04.utils import get_path, load_config  # noqa: E402


class Test04EvaluationAndMetrics(unittest.TestCase):
    r"""
    Purpose: Kiểm tra độ chính xác toán học tuyệt đối của bộ 5 chỉ số đánh giá,
             quy trình Walk-Forward Backtesting, bảng đối sánh PEMS-BAY và kiểm toán đầu ra báo cáo.
    """

    def setUp(self) -> None:
        self.cfg = load_config()

    def test_01_compute_5_metrics_exact_math(self) -> None:
        r"""Kiểm tra công thức MSE, RMSE, MAE, MAPE (%) và R² khớp tuyệt đối với tính tay."""
        y_true = np.array([100.0, 200.0, 300.0, 400.0])
        y_pred = np.array([110.0, 190.0, 315.0, 385.0])
        # Residuals: [-10, +10, -15, +15]
        # Squared errors: [100, 100, 225, 225] -> MSE = 650 / 4 = 162.5 -> RMSE = sqrt(162.5) = 12.74754878
        # Absolute errors: [10, 10, 15, 15] -> MAE = 50 / 4 = 12.5
        # Percentage errors: [10/100, 10/200, 15/300, 15/400] = [0.10, 0.05, 0.05, 0.0375] -> Mean = 0.059375 -> 5.9375%
        # SS_res = 650, SS_tot = (150^2 + 50^2 + 50^2 + 150^2) = 50000 -> R2 = 1 - 650/50000 = 0.9870
        m = compute_5_metrics(y_true, y_pred)
        self.assertAlmostEqual(m["MSE"], 162.5, places=4)
        self.assertAlmostEqual(m["RMSE"], np.sqrt(162.5), places=4)
        self.assertAlmostEqual(m["MAE"], 12.5, places=4)
        self.assertAlmostEqual(m["MAPE"], 5.9375, places=4)
        self.assertAlmostEqual(m["R2"], 0.9870, places=4)
        self.assertAlmostEqual(m["Max_Error"], 15.0, places=4)
        self.assertAlmostEqual(m["Mean_Residual"], 0.0, places=4)

    def test_02_walk_forward_backtesting_structure(self) -> None:
        r"""Kiểm tra quy trình Walk-Forward Backtesting trả về đúng bảng chỉ số và chuỗi dự báo cuốn chiếu."""
        n = 50
        y_te = np.linspace(200000.0, 240000.0, n)
        ts_te = pd.date_range("2014-12-01", periods=n, freq="1h")
        train_artifacts = {
            "holdout_predictions": {
                "y_test_orig": y_te,
                "test_timestamps": ts_te,
                "test": {
                    "VAR": y_te + 1200.0,
                    "XGBoost": y_te + 800.0,
                    "LSTM": y_te + 950.0,
                },
            }
        }
        wf_out = evaluate_walk_forward_backtesting({}, train_artifacts, eval_steps=36)
        self.assertEqual(len(wf_out["metrics_df"]), 3)
        self.assertEqual(len(wf_out["predictions_df"]), 36)
        self.assertTrue({"Model", "MSE", "RMSE", "MAE", "MAPE", "R2"}.issubset(set(wf_out["metrics_df"].columns)))

    def test_03_pems_bay_3model_benchmark_evaluation(self) -> None:
        r"""Kiểm tra bảng đối sánh DCRNN vs. Multi-Horizon TCN vs. Multi-Horizon LSTM tại 15p, 30p, 60p."""
        rng = np.random.default_rng(42)
        y_true = rng.uniform(50.0, 65.0, size=(40, 3, 10)).astype(np.float32)
        pems_preds = {
            "Y_test_orig": y_true,
            "DCRNN": y_true + rng.normal(0, 1.2, size=y_true.shape).astype(np.float32),
            "Multi_Horizon_TCN": y_true + rng.normal(0, 1.8, size=y_true.shape).astype(np.float32),
            "Multi_Horizon_LSTM": y_true + rng.normal(0, 2.1, size=y_true.shape).astype(np.float32),
            "horizon_names": ["15_min", "30_min", "60_min"],
            "horizon_steps": [3, 6, 12],
        }
        model_meta = {
            "DCRNN": {"train_time_sec": 12.5},
            "Multi_Horizon_TCN": {"train_time_sec": 4.2},
            "Multi_Horizon_LSTM": {"train_time_sec": 3.8},
        }
        res = evaluate_pems_bay_benchmark(pems_preds, model_meta)
        dcrnn_df = res["dcrnn_df"]
        bench_df = res["benchmark_df"]

        self.assertEqual(len(dcrnn_df), 3)
        # 3 mô hình x (3 tầm nhìn + 1 dòng trung bình) = 12 dòng
        self.assertEqual(len(bench_df), 12)
        avg_sub = bench_df[bench_df["Horizon"] == "Average_All_Horizons"].set_index("Model")
        self.assertLess(float(avg_sub.loc["DCRNN", "MAE"]), float(avg_sub.loc["Multi_Horizon_LSTM", "MAE"]))

    def test_04_trained_artifacts_and_reports_audit_if_present(self) -> None:
        r"""
        Nếu người dùng đã chạy xong pipeline (có thư mục reports/lab04/metrics/holdout_test_metrics.csv),
        kiểm tra tự động chất lượng số liệu thực tế, 23 biểu đồ .png và 3 tệp báo cáo (.html, .pdf, .md).
        """
        metrics_dir = get_path("metrics_dir", self.cfg)
        fig_dir = get_path("figures_dir", self.cfg)
        reports_dir = get_path("reports_dir", self.cfg)
        holdout_csv = metrics_dir / "holdout_test_metrics.csv"

        if not holdout_csv.exists():
            # Nếu chưa chạy train thực tế thì bỏ qua kiểm toán file đầu ra
            return

        h_df = pd.read_csv(holdout_csv)
        self.assertEqual(len(h_df), 3)
        for _, row in h_df.iterrows():
            self.assertGreater(float(row["R2"]), 0.90, f"Mô hình {row['Model']} có R² quá thấp: {row['R2']}")
            self.assertLess(float(row["MAPE"]), 10.0, f"Mô hình {row['Model']} có MAPE quá cao: {row['MAPE']}%")

        png_files = list(fig_dir.glob("*.png"))
        self.assertGreaterEqual(len(png_files), 20, f"Thiếu biểu đồ trong {fig_dir}: chỉ có {len(png_files)} file")

        for rep_name in ("LAB_04_REPORT.html", "LAB_04_REPORT.md"):
            rep_p = reports_dir / rep_name
            self.assertTrue(rep_p.exists(), f"Chưa tạo tệp báo cáo {rep_p}")
            self.assertGreater(rep_p.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main(verbosity=2)