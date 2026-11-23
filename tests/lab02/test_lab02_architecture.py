r"""Kiểm thử kiến trúc mô-đun Lab 02 (src/lab02/ & script/lab02/) tương thích phiên bản hiện tại."""

from __future__ import annotations

import inspect
import json
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.lab02.acf_pacf_analysis import analyze_acf_pacf
from src.lab02.data_loader import load_dataset, split_dataset
from src.lab02.lab02_models import (
    auto_identify_candidates_from_data,
    fit_ar_model,
    fit_arima_model,
    fit_ma_model,
    fit_sarima_model,
    get_candidate_models,
    run_ar_experiment,
    run_arima_experiment,
    run_ma_experiment,
    run_sarima_experiment,
)
from src.lab02.logging_config import (
    get_log_file_path,
    setup_experiment_logger,
    setup_lab02_logger,
)
from src.lab02.model_summary import ModelSummaryRegistry, format_terminal_model_card
from src.lab02.modeling import calculate_metrics, evaluate_forecast
from src.lab02.stats import perform_adf_test, significant_lags
from src.lab02.utils import PROJECT_ROOT, load_config


def _split_dataset_compat(
    df: pd.DataFrame,
    value_col: str,
    dataset_key: str,
    cfg: Dict[str, Any],
    test_ratio: float = 0.2,
) -> Tuple[pd.Series, pd.Series]:
    """Gọi split_dataset khớp chữ ký thực tế của src/lab02/data_loader.py."""
    sig = inspect.signature(split_dataset)
    params = list(sig.parameters.keys())
    first_arg = df if (params and params[0] in {"df", "data", "dataframe"}) else df[value_col]

    kwargs: Dict[str, Any] = {}
    if "dataset_key" in sig.parameters:
        kwargs["dataset_key"] = dataset_key
    elif "dataset_name" in sig.parameters:
        kwargs["dataset_name"] = dataset_key
    if "cfg" in sig.parameters:
        kwargs["cfg"] = cfg
    elif "config" in sig.parameters:
        kwargs["config"] = cfg
    if "test_ratio" in sig.parameters:
        kwargs["test_ratio"] = test_ratio

    result = split_dataset(first_arg, **kwargs)
    if not (isinstance(result, tuple) and len(result) >= 2):
        raise AssertionError(f"split_dataset trả về cấu trúc không hợp lệ: {type(result)}")

    train_obj, test_obj = result[0], result[1]
    if isinstance(train_obj, pd.DataFrame):
        train_obj = train_obj[value_col]
    if isinstance(test_obj, pd.DataFrame):
        test_obj = test_obj[value_col]
    return train_obj, test_obj


class TestLab02Architecture(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = PROJECT_ROOT
        cls.cfg = load_config()
        cls.df1, cls.meta1 = load_dataset("dataset1", cls.cfg)
        cls.df2, cls.meta2 = load_dataset("dataset2", cls.cfg)

    def test_01_logging_no_duplicate_handlers(self) -> None:
        logger1 = setup_lab02_logger("dataset1", "unit_test")
        parent_logger = setup_lab02_logger("dataset1")
        initial_handlers = len(parent_logger.handlers)
        _ = setup_lab02_logger("dataset1", "unit_test")
        _ = setup_lab02_logger("dataset1")
        self.assertEqual(len(parent_logger.handlers), initial_handlers)

        logger1.info("Unit test log message for dataset1.")
        log_path = get_log_file_path("dataset1")
        self.assertTrue(log_path.exists())
        self.assertGreater(log_path.stat().st_size, 0)

    def test_02_data_loading_and_chronological_split(self) -> None:
        self.assertEqual(len(self.df1), 204)
        self.assertTrue(self.df1.index.is_monotonic_increasing)
        train1, test1 = _split_dataset_compat(
            self.df1, value_col="value", dataset_key="dataset1", cfg=self.cfg, test_ratio=0.2
        )
        self.assertEqual(len(train1), 163)
        self.assertEqual(len(test1), 41)
        self.assertLess(train1.index.max(), test1.index.min())

        self.assertEqual(len(self.df2), 144)
        self.assertTrue(self.df2.index.is_monotonic_increasing)
        train2, test2 = _split_dataset_compat(
            self.df2, value_col="Passengers", dataset_key="dataset2", cfg=self.cfg, test_ratio=0.2
        )
        self.assertEqual(len(train2), 115)
        self.assertEqual(len(test2), 29)
        self.assertLess(train2.index.max(), test2.index.min())

    def test_03_adf_and_acf_pacf_analysis(self) -> None:
        train1 = self.df1["value"].iloc[:163]
        p_raw = perform_adf_test(train1, title="Test Raw Retail Sales", dataset_name="dataset1")
        self.assertGreater(p_raw, 0.05)

        diff_1_12 = train1.diff(1).diff(12).dropna()
        p_diff = perform_adf_test(diff_1_12, title="Test Diff(1,12) Retail Sales", dataset_name="dataset1")
        self.assertLess(p_diff, 0.05)

        acf_res = analyze_acf_pacf(
            series=diff_1_12,
            dataset_name="dataset1",
            nlags=36,
            differencing_info="d=1, D=1, s=12",
            seasonal_period=12,
            show_plot=False,
        )
        self.assertEqual(acf_res["n"], 150)
        self.assertIn("Lag", acf_res["full_table"].columns)

    def test_04_model_wrappers_and_invalid_inputs(self) -> None:
        train2 = self.df2["Passengers"].iloc[:115]
        test2 = self.df2["Passengers"].iloc[115:]

        with self.assertRaises(ValueError):
            fit_ar_model(train2, p=1, d=0, dataset_name="dataset2", check_stationarity=True, verbose=False)

        with self.assertRaises(ValueError):
            fit_ma_model(train2, q=0, d=1, dataset_name="dataset2", verbose=False)

        ar_exp = run_ar_experiment(train2, test2, p=1, d=1, dataset_name="dataset2", show_plot=False, verbose=False)
        ma_exp = run_ma_experiment(train2, test2, q=1, d=1, dataset_name="dataset2", show_plot=False, verbose=False)
        arima_exp = run_arima_experiment(
            train2, test2, order=(1, 1, 1), dataset_name="dataset2", show_plot=False, verbose=False
        )
        sarima_exp = run_sarima_experiment(
            train2,
            test2,
            order=(1, 1, 0),
            seasonal_order=(0, 1, 1, 12),
            use_log_transform=True,
            dataset_name="dataset2",
            plot_diagnostics_flag=False,
            plot_forecast_flag=False,
            show_plot=False,
            verbose=False,
        )

        self.assertIn("MAE", ar_exp["metrics"])
        self.assertIn("RMSE", ma_exp["metrics"])
        self.assertIn("MAPE (%)", arima_exp["metrics"])
        self.assertGreater(sarima_exp["metrics"]["R2"], 0.90)

    def test_05_experiment_logger_and_registry_summary(self) -> None:
        exp_logger, run_id_1, exp_path_1 = setup_experiment_logger("dataset1")
        exp_logger.info("First experiment run log entry.")
        _, run_id_2, exp_path_2 = setup_experiment_logger("dataset1")
        self.assertNotEqual(run_id_1, run_id_2)
        self.assertNotEqual(exp_path_1, exp_path_2)
        self.assertTrue(exp_path_1.exists())

        c1_std = get_candidate_models("dataset1", mode="standard")
        c1_full = get_candidate_models("dataset1", mode="full")
        self.assertGreater(len(c1_full), len(c1_std))

        train2, test2 = self.df2["Passengers"].iloc[:115], self.df2["Passengers"].iloc[115:]
        ar_exp = run_ar_experiment(train2, test2, p=1, d=1, dataset_name="dataset2", show_plot=False, verbose=False)

        reg = ModelSummaryRegistry(
            dataset_name="dataset2",
            dataset_title="AirPassengers (Unit Test)",
            n_obs=144,
            date_range="1949-01-01 to 1960-12-01",
            frequency="MS",
            train_size=115,
            test_size=29,
            run_id="unit_test_run_001",
        )
        reg.register_experiment(ar_exp)
        reg.register_failure(
            model_name="AR(0) [Invalid]",
            model_family="AR",
            order=(0, 1, 0),
            error_message="Invalid AR order p=0",
        )
        comp_df = reg.get_comparison_dataframe()
        self.assertIn("FAILED", comp_df["Status"].values)

        best_crit = reg.generate_automatic_interpretation().get("best_by_criterion", {})
        self.assertIn("lowest_test_rmse", best_crit)
        self.assertIn("highest_test_r2", best_crit)

    def test_06_automated_experiment_runner_and_all_exports(self) -> None:
        from script.lab02.run_lab02_experiments import run_dataset_experiment_suite

        res = run_dataset_experiment_suite("dataset1", mode="standard", save_figures=False, verbose=False)
        self.assertTrue(Path(res["log_path"]).exists())
        for key in ["csv", "json", "md", "run_md"]:
            self.assertIn(key, res["summary_paths"])
            self.assertTrue(Path(res["summary_paths"][key]).exists(), f"Missing file: {key}")

        with open(res["summary_paths"]["json"], "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.assertIn("highest_test_r2", payload["automatic_interpretation"]["best_by_criterion"])

    def test_07_dual_space_auto_identification_pipeline(self) -> None:
        train1 = self.df1["value"].iloc[:163]
        auto1 = auto_identify_candidates_from_data(
            train1, dataset_name="dataset1", seasonal_period=12, include_log=True
        )

        self.assertEqual(auto1["orders_detected"]["original"]["d"], 1)
        self.assertEqual(auto1["orders_detected"]["original"]["D"], 1)
        self.assertEqual(auto1["orders_detected"]["log"]["d"], 1)
        self.assertEqual(auto1["orders_detected"]["log"]["D"], 1)

        families1 = {c["model_family"] for c in auto1["candidates"]}
        self.assertEqual(families1, {"AR", "MA", "ARIMA", "SARIMA", "Log-SARIMA"})
        self.assertGreater(len([c for c in auto1["candidates"] if c["model_family"] == "Log-SARIMA"]), 5)


if __name__ == "__main__":
    unittest.main()