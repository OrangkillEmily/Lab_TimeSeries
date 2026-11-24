"""Focused no-leakage and scratch-model tests; run with python -m unittest."""
import unittest
import tempfile
from pathlib import Path
import yaml
import numpy as np
import pandas as pd

from src.lab03.data import make_synthetic_hourly_series, chronological_split
from src.lab03.features import build_supervised_features
from src.lab03.cv import expanding_window_splits
from src.lab03.metrics import regression_metrics
from src.lab03.lab03_models import (
    NaiveLastValue, LinearRegressionScratch, DecisionTreeRegressorScratch,
    RandomForestRegressorScratch, XGBoostScratch,
)


class Lab03Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        series = make_synthetic_hourly_series(900, seed=7)
        cls.X, cls.y = build_supervised_features(series, lags=(1, 2, 24),
                                                 rolling_windows=(6, 24),
                                                 include_calendar=True)

    def test_feature_engineering_uses_only_past_values(self):
        series = make_synthetic_hourly_series(500, seed=1)
        X, y = build_supervised_features(series, lags=(1, 2), rolling_windows=(6,),
                                         include_calendar=False)
        ts = X.index[20]
        pos = series.index.get_loc(ts)
        self.assertAlmostEqual(X.loc[ts, "lag_1"], series.iloc[pos - 1])
        self.assertAlmostEqual(X.loc[ts, "roll_mean_6"], series.iloc[pos - 6:pos].mean())
        self.assertAlmostEqual(y.loc[ts], series.loc[ts])

    def test_chronological_split_is_ordered_and_non_overlapping(self):
        parts = chronological_split(self.X, self.y, 0.70, 0.15, 0.15)
        Xtr, Xv, Xte, ytr, yv, yte = parts
        self.assertLess(Xtr.index.max(), Xv.index.min())
        self.assertLess(Xv.index.max(), Xte.index.min())
        self.assertEqual(len(Xtr), len(ytr))
        self.assertEqual(len(Xv), len(yv))
        self.assertEqual(len(Xte), len(yte))

    def test_walk_forward_folds_are_causal(self):
        folds = list(expanding_window_splits(120, n_splits=5))
        self.assertEqual(len(folds), 5)
        previous_train = 0
        for train_idx, val_idx in folds:
            self.assertGreater(len(train_idx), previous_train)
            self.assertLess(train_idx.max(), val_idx.min())
            previous_train = len(train_idx)

    def test_metrics_known_values(self):
        m = regression_metrics([1, 2, 3], [1, 3, 2])
        self.assertAlmostEqual(m["MAE"], 2/3)
        self.assertAlmostEqual(m["MSE"], 2/3)
        self.assertAlmostEqual(m["RMSE"], np.sqrt(2/3))

    def test_all_scratch_models_fit_and_predict(self):
        X = self.X.iloc[:350].to_numpy(dtype=float)
        y = self.y.iloc[:350].to_numpy(dtype=float)
        Q = self.X.iloc[350:370].to_numpy(dtype=float)
        models = [
            NaiveLastValue(),
            LinearRegressionScratch(ridge=1e-6),
            DecisionTreeRegressorScratch(max_depth=3, min_samples_split=10,
                                          min_samples_leaf=4, max_thresholds=5,
                                          max_train_rows=0),
            RandomForestRegressorScratch(n_estimators=3, max_depth=3,
                                         min_samples_split=10, min_samples_leaf=4,
                                         max_thresholds=5, max_train_rows=0),
            XGBoostScratch(n_estimators=3, max_depth=2, min_child_weight=2,
                           max_bins=8, max_train_rows=0),
        ]
        for model in models:
            with self.subTest(model=type(model).__name__):
                model.fit(X, y)
                pred = np.asarray(model.predict(Q))
                self.assertEqual(pred.shape, (len(Q),))
                self.assertTrue(np.isfinite(pred).all())

    def test_root_config_is_lab03_only_and_requires_real_data_by_default(self):
        root = Path(__file__).resolve().parents[2]
        config = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
        self.assertIn("lab03", config)
        self.assertIn("lab01", config)
        self.assertIn("lab02", config)
        self.assertFalse(config["lab03"]["data"]["fallback_synthetic"])
        self.assertEqual(config["lab03"]["preprocessing"]["cross_validation"]["n_splits"], 5)

    def test_model_files_are_independent_and_do_not_use_library_estimators(self):
        root = Path(__file__).resolve().parents[2]
        model_dir = root / "src" / "lab03" / "lab03_models"
        expected = {"naive_baseline.py", "linear_regression.py", "decision_tree.py",
                    "random_forest.py", "xgboost_scratch.py"}
        self.assertTrue(expected.issubset({p.name for p in model_dir.glob("*.py")}))
        for path in model_dir.glob("*.py"):
            source = path.read_text(encoding="utf-8").lower()
            self.assertNotIn("from sklearn", source)
            self.assertNotIn("import sklearn", source)
            self.assertNotRegex(source, r"(?m)^\s*(?:from|import)\s+xgboost(?:\.|\s|$)")

    def test_all_model_and_phase_docs_have_required_sections(self):
        root = Path(__file__).resolve().parents[2]
        required = ["## Purpose", "## Input - Output", "## How to do",
                    "## Evaluation Metrics", "## Used Data"]
        for folder in [root / "docs/lab03/models", root / "docs/lab03/phases"]:
            docs = list(folder.glob("*.md"))
            self.assertGreater(len(docs), 0)
            for path in docs:
                content = path.read_text(encoding="utf-8")
                for heading in required:
                    self.assertIn(heading, content, msg=f"{path} missing {heading}")

    def test_report_bundle_writes_markdown_html_and_pdf(self):
        from src.lab03.lab03_utils.report import write_report_bundle
        report = {
            "run_id": "unit-test", "status": "SUCCESS", "champion": "Naive Baseline",
            "summary_text": "A small report-generation test.",
            "dataset": {"dataset_name": "unit_test.csv", "dataset_type": "synthetic_demo",
                        "observations_after_cleaning": 100, "start": "2020-01-01",
                        "end": "2020-01-05", "frequency": "h",
                        "duplicate_timestamps_removed": 0, "missing_periods_inserted": 0,
                        "missing_values_remaining": 0},
            "split": {"train": {"rows": 70, "start": "2020-01-01", "end": "2020-01-03", "ratio": .70},
                      "validation": {"rows": 15, "start": "2020-01-04", "end": "2020-01-04", "ratio": .15},
                      "test": {"rows": 15, "start": "2020-01-05", "end": "2020-01-05", "ratio": .15}},
            "feature_config": {"lags": [1], "rolling_windows": [6], "include_calendar_features": True},
            "feature_count": 3, "cv_summary": [], "n_splits": 5,
            "model_rows": [{"model": "Naive Baseline", "cv_RMSE_mean": None,
                            "validation_RMSE": 1.0, "test_MAE": 1.0, "test_RMSE": 1.2,
                            "test_MAPE (%)": 2.0, "test_R2": 0.9, "fit_seconds": 0.01}],
            "forecast_horizon": 24, "forecast_training_rows": 99,
            "forecast_training_end": "2020-01-05 00:00:00",
            "figures": {}, "figure_relative_paths": {},
            "legacy_snapshot": [],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = write_report_bundle(report, temp_dir)
            for key in ("md", "html", "pdf"):
                self.assertTrue(Path(paths[key]).exists())
                self.assertGreater(Path(paths[key]).stat().st_size, 0)
            report_text = Path(paths["md"]).read_text(encoding="utf-8")
            self.assertIn("99 labeled observations through 2020-01-05 00:00:00", report_text)
            self.assertIn("separate from the Train + Validation fit used for Test scoring", report_text)


if __name__ == "__main__":
    unittest.main()
