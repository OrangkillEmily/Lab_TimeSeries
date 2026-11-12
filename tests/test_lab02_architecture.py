# $env:PYTHONPATH = ".."
# python test_lab02_architecture.py
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.data_loader import load_and_prep_data, split_train_test_chronological
from src.lab02.lab02_utils.logging_config import (
    get_project_root,
    setup_lab02_logger,
    get_log_file_path,
)
from src.lab02.lab02_utils.stats import perform_adf_test, significant_lags
from src.lab02.lab02_utils.acf_pacf_analysis import analyze_acf_pacf
from src.lab02.lab02_utils.modeling import evaluate_forecast, calculate_metrics
from src.lab02.lab02_utils.model_summary import ModelSummaryRegistry, format_terminal_model_card
from src.lab02.lab02_models import (
    fit_ar_model,
    run_ar_experiment,
    fit_ma_model,
    run_ma_experiment,
    fit_arima_model,
    run_arima_experiment,
    fit_sarima_model,
    run_sarima_experiment,
)


class TestLab02Architecture(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = get_project_root()
        cls.ds1_path = cls.root / "data" / "raw" / "Monthly Retail Sales.csv"
        cls.ds2_path = cls.root / "data" / "raw" / "AirPassengers.csv"

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
        df1 = load_and_prep_data(self.ds1_path, date_col="date", freq="MS", dataset_name="dataset1")
        self.assertEqual(len(df1), 204)
        self.assertTrue(df1.index.is_monotonic_increasing)
        train1, test1, meta1 = split_train_test_chronological(df1["value"], test_ratio=0.2, dataset_name="dataset1")
        self.assertEqual(len(train1), 163)
        self.assertEqual(len(test1), 41)
        self.assertLess(train1.index.max(), test1.index.min())

        df2 = load_and_prep_data(self.ds2_path, date_col="Month", freq="MS", dataset_name="dataset2")
        self.assertEqual(len(df2), 144)
        train2, test2, _ = split_train_test_chronological(df2["Passengers"], test_ratio=0.2, dataset_name="dataset2")
        self.assertEqual(len(train2), 115)
        self.assertEqual(len(test2), 29)

    def test_03_adf_and_acf_pacf_analysis(self) -> None:
        df1 = load_and_prep_data(self.ds1_path, date_col="date", freq="MS")
        train1 = df1["value"].iloc[:163]
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
        df2 = load_and_prep_data(self.ds2_path, date_col="Month", freq="MS")
        train2 = df2["Passengers"].iloc[:115]
        test2 = df2["Passengers"].iloc[115:]

        with self.assertRaises(ValueError):
            fit_ar_model(train2, p=1, d=0, dataset_name="dataset2", check_stationarity=True, verbose=False)

        with self.assertRaises(ValueError):
            fit_ma_model(train2, q=0, d=1, dataset_name="dataset2", verbose=False)

        ar_exp = run_ar_experiment(train2, test2, p=1, d=1, dataset_name="dataset2", show_plot=False, verbose=False)
        ma_exp = run_ma_experiment(train2, test2, q=1, d=1, dataset_name="dataset2", show_plot=False, verbose=False)
        arima_exp = run_arima_experiment(train2, test2, order=(1, 1, 1), dataset_name="dataset2", show_plot=False, verbose=False)
        sarima_exp = run_sarima_experiment(
            train2, test2, order=(1, 1, 0), seasonal_order=(0, 1, 1, 12),
            use_log_transform=True, dataset_name="dataset2",
            plot_diagnostics_flag=False, plot_forecast_flag=False, show_plot=False, verbose=False
        )

        self.assertIn("MAE", ar_exp["metrics"])
        self.assertGreater(sarima_exp["metrics"]["R2"], 0.90)
    
    def test_05_experiment_logger_and_registry_summary(self) -> None:
        from src.lab02.lab02_utils.logging_config import setup_experiment_logger
        from src.lab02.lab02_models import get_candidate_models

        # Kiểm tra experiment logger tạo Run ID riêng biệt (không ghi đè)
        exp_logger, run_id_1, exp_path_1 = setup_experiment_logger("dataset1")
        exp_logger.info("First experiment run log entry.")
        _, run_id_2, exp_path_2 = setup_experiment_logger("dataset1")
        self.assertNotEqual(run_id_1, run_id_2)
        self.assertNotEqual(exp_path_1, exp_path_2)
        self.assertTrue(exp_path_1.exists())

        # Kiểm tra Candidate Registry và theo dõi mô hình lỗi trong Summary
        c1_std = get_candidate_models("dataset1", mode="standard")
        c1_full = get_candidate_models("dataset1", mode="full")
        self.assertGreater(len(c1_full), len(c1_std))

        df2 = load_and_prep_data(self.ds2_path, date_col="Month", freq="MS")
        train2, test2 = df2["Passengers"].iloc[:115], df2["Passengers"].iloc[115:]
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
        import json
        from script.run_lab02_experiments import run_dataset_experiment_suite

        res = run_dataset_experiment_suite("dataset1", mode="standard", save_figures=False, verbose=False)
        self.assertTrue(Path(res["log_path"]).exists())
        for key in ["csv", "json", "md", "run_md"]:
            self.assertIn(key, res["summary_paths"])
            self.assertTrue(Path(res["summary_paths"][key]).exists(), f"Missing file: {key}")

        with open(res["summary_paths"]["json"], "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.assertIn("highest_test_r2", payload["automatic_interpretation"]["best_by_criterion"])

    def test_07_dual_space_auto_identification_pipeline(self) -> None:
        from src.lab02.lab02_models import auto_identify_candidates_from_data

        df1 = load_and_prep_data(self.ds1_path, date_col="date", freq="MS")
        train1 = df1["value"].iloc[:163]
        auto1 = auto_identify_candidates_from_data(train1, dataset_name="dataset1", seasonal_period=12, include_log=True)

       
        self.assertEqual(auto1["orders_detected"]["original"]["d"], 1)
        self.assertEqual(auto1["orders_detected"]["original"]["D"], 1)
        self.assertEqual(auto1["orders_detected"]["log"]["d"], 1)
        self.assertEqual(auto1["orders_detected"]["log"]["D"], 1)

        
        families1 = {c["model_family"] for c in auto1["candidates"]}
        self.assertEqual(families1, {"AR", "MA", "ARIMA", "SARIMA", "Log-SARIMA"})
        self.assertGreater(len([c for c in auto1["candidates"] if c["model_family"] == "Log-SARIMA"]), 5)

if __name__ == "__main__":
    unittest.main()