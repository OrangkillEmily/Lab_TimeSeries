"""End-to-end Lab 03 orchestrator.

Runs, in order, every phase documented in ``docs/lab03/phases/``: Data Ingestion,
Data Cleaning, Feature Engineering, Data Splitting, Cross Validation, Training,
Model Selection, Inference, Evaluation, Reporting and Experiment Tracking/Logging.
This is the single function that ``script/lab03/run_lab03.py`` and the root
``scripts/run_lab.py`` call to run the whole lab with one command, and it is the
piece explicitly listed as missing in ``docs/lab03/DELIVERABLE_CHECKLIST.md``.

Every model is one of the from-scratch NumPy estimators in ``src/lab03/lab03_models/``;
no scikit-learn or xgboost estimator is fit here. Train/Validation/Test are a
chronological 70/15/15 split (no shuffle); cross-validation is five expanding-window
folds on Train only; the champion is chosen by Validation RMSE before Test is touched.
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.lab03.cv import evaluate_walk_forward
from src.lab03.data import chronological_split, load_hourly_series
from src.lab03.features import build_supervised_features, recursive_forecast
from src.lab03.lab03_models import (
    DecisionTreeRegressorScratch, LinearRegressionScratch, NaiveLastValue,
    RandomForestRegressorScratch, XGBoostScratch,
)
from src.lab03.lab03_utils.report import display_model_name, write_report_bundle
from src.lab03.logging_utils import ProgressTracker, RunLogger
from src.lab03.metrics import regression_metrics
from src.lab03.plotting import save_all_figures
from src.lab03.utils import PROJECT_ROOT, ensure_directories, get_path, load_config, set_global_seed

MODEL_SLUGS = {
    "Naive Last-Value": "naive_last_value", "Linear Regression": "linear_regression",
    "CART": "cart", "Random Forest": "random_forest", "XGBoost-style": "xgboost_style",
}
METRIC_ORDER = ["MAE", "MSE", "RMSE", "MAPE (%)", "sMAPE (%)", "R2"]


def build_model_factories(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Purpose: build the five zero-argument model factories from ``config.yaml`` hyperparameters.

    Input: the (optionally ``--fast``-overridden) ``lab03`` config dict.
    Output: ``{display_name: factory}``; each factory returns a freshly initialized,
    unfitted estimator so CV folds and the final refit never share mutable state.
    """
    m = cfg["models"]
    return {
        "Naive Last-Value": lambda: NaiveLastValue(),
        "Linear Regression": lambda: LinearRegressionScratch(**m["linear_regression"]),
        "CART": lambda: DecisionTreeRegressorScratch(**m["decision_tree"]),
        "Random Forest": lambda: RandomForestRegressorScratch(**m["random_forest"]),
        "XGBoost-style": lambda: XGBoostScratch(**m["xgboost_style"]),
    }


def _frame_metadata(name: str, target: pd.Series, total_rows: int) -> Dict[str, Any]:
    """Purpose: one partition's row-count/time-span/ratio summary for the report.

    Input: partition label, its target Series, and the total row count across all partitions.
    Output: a dict with ``rows``, ``start``, ``end`` and ``ratio`` (share of total_rows).
    """
    return {"rows": int(len(target)), "start": str(target.index.min()), "end": str(target.index.max()),
            "ratio": len(target) / total_rows}


def run_pipeline(run_id: Optional[str] = None, fast: bool = False, run_cv: bool = True) -> Dict[str, Any]:
    """Purpose: run the complete Lab 03 pipeline exactly once, end to end.

    Input: optional explicit ``run_id`` (default: a timestamp), ``fast`` (apply
    ``config.yaml``'s ``lab03.fast_overrides`` for a quick smoke run) and ``run_cv``
    (set False to skip the 5-fold walk-forward CV, e.g. for a very fast dry run).
    Output: a dict with ``run_id``, ``status`` (``COMPLETED``/``FAILED``), ``champion``,
    ``run_dir`` and the paths of the generated report bundle and model summary.
    How: loads data, builds features, splits chronologically, optionally cross-validates
    on Train, selects the champion by Validation RMSE, refits on Train+Validation,
    evaluates once on held-out Test, saves all models/figures/predictions/forecast,
    writes the model summary (3 formats + side files) and the report bundle (md/html/pdf),
    logging START/SUCCESS/ERROR with timestamps and updating the progress-status files
    at every phase.
    """
    cfg = load_config(fast=fast)
    set_global_seed(cfg["project"].get("random_seed", 42))
    dirs = ensure_directories(cfg)
    run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
    logger = RunLogger(dirs["logs_dir"], run_id)
    progress = ProgressTracker(dirs["progress_dir"], run_id)
    run_dir = get_path("figures_dir", cfg) / run_id  # reports/lab03/notebook_runs/<run_id>
    figure_dir = run_dir / "figures"
    model_dir = run_dir / "models"
    for d in (run_dir, figure_dir, model_dir):
        d.mkdir(parents=True, exist_ok=True)
    logger.log("PIPELINE", "START", f"Lab 03 pipeline starting | run_id={run_id} | fast={fast} | run_cv={run_cv}")
    t_start = time.perf_counter()
    try:
        # ---- 1-2. Data Ingestion & Data Cleaning ----------------------------------------------
        with logger.stage("DATA_INGESTION", "Loading and regularizing AEP_hourly.csv") as ctx:
            data_cfg = cfg["data"]
            data_path = get_path("data_raw", cfg) if "data_raw" in cfg["paths"] else PROJECT_ROOT / "data" / "raw" / "AEP_hourly.csv"
            series, ingestion_meta = load_hourly_series(data_path, timestamp_col=data_cfg["date_col"],
                                                        value_col=data_cfg["target_col"], frequency=data_cfg["freq"])
            max_rows = cfg.get("fast_max_rows")
            if max_rows:
                series = series.iloc[-int(max_rows):]
                ingestion_meta["observations_after_cleaning"] = int(len(series))
                ingestion_meta["start"], ingestion_meta["end"] = str(series.index.min()), str(series.index.max())
            assert series.index.is_monotonic_increasing and series.index.is_unique and series.notna().all()
            series.rename(data_cfg["target_col"]).to_csv(run_dir / "cleaned_hourly_series.csv", header=True)
            ctx["summary"] = f"{len(series):,} hourly rows | {ingestion_meta['start']} -> {ingestion_meta['end']}"
            progress.update("data_ingestion", ingestion_meta)

        # ---- 3. Feature Engineering ------------------------------------------------------------
        with logger.stage("FEATURE_ENGINEERING", "Building causal lag/rolling/calendar features") as ctx:
            f_cfg = cfg["features"]
            X, y = build_supervised_features(series, lags=tuple(f_cfg["lags"]),
                                             rolling_windows=tuple(f_cfg["rolling_windows"]),
                                             include_calendar=f_cfg["include_calendar"])
            timestamp = X.index[200]
            position = series.index.get_loc(timestamp)
            assert X.loc[timestamp, "lag_1"] == series.iloc[position - 1]
            assert y.loc[timestamp] == series.loc[timestamp]
            ctx["summary"] = f"X={X.shape}, y={y.shape}, leakage check passed"
            progress.update("feature_engineering", {"feature_count": int(X.shape[1]), "rows": int(len(X)),
                                                    "lags": list(f_cfg["lags"]), "rolling_windows": list(f_cfg["rolling_windows"])})

        # ---- 4. Data Splitting -------------------------------------------------------------------
        with logger.stage("DATA_SPLITTING", "Chronological 70/15/15 split") as ctx:
            sp = cfg["preprocessing"]["split"]
            X_train, X_val, X_test, y_train, y_val, y_test = chronological_split(
                X, y, train_ratio=sp["train_ratio"], val_ratio=sp["val_ratio"], test_ratio=sp["test_ratio"])
            total = len(X)
            split_meta = {"train": _frame_metadata("train", y_train, total), "validation": _frame_metadata("validation", y_val, total),
                          "test": _frame_metadata("test", y_test, total)}
            ctx["summary"] = f"train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
            progress.update("data_splitting", split_meta)

        # ---- 5. Cross Validation (Train only) -----------------------------------------------------
        factories = build_model_factories(cfg)
        cv_detail = pd.DataFrame(columns=["fold", "model", "train_rows", "validation_rows"] + METRIC_ORDER)
        if run_cv and cfg["preprocessing"]["cross_validation"].get("enabled", True):
            with logger.stage("CROSS_VALIDATION", "5-fold expanding-window CV on Train") as ctx:
                n_splits = cfg["preprocessing"]["cross_validation"]["n_splits"]
                cv_detail, cv_aggregate = evaluate_walk_forward(X_train, y_train, factories, n_splits=n_splits)
                cv_detail.to_csv(run_dir / "walk_forward_cv.csv", index=False)
                ctx["summary"] = f"{n_splits} folds x {len(factories)} models = {len(cv_detail)} rows"
                progress.update("cross_validation", {"n_splits": n_splits, "rows": int(len(cv_detail))})
        else:
            logger.log("CROSS_VALIDATION", "INFO", "Skipped (run_cv=False or disabled in config).")

        # ---- 6. Training + Model Selection (Validation) ---------------------------------------------
        with logger.stage("TRAINING", "Fitting candidates on Train, scoring on Validation") as ctx:
            validation_rows = []
            for name, factory in factories.items():
                model = factory()
                t0 = time.perf_counter()
                model.fit(X_train.to_numpy(dtype=float), y_train.to_numpy(dtype=float))
                fit_seconds = time.perf_counter() - t0
                t0 = time.perf_counter()
                prediction = model.predict(X_val.to_numpy(dtype=float))
                predict_seconds = time.perf_counter() - t0
                validation_rows.append({"model": name, **regression_metrics(y_val, prediction),
                                        "fit_seconds": fit_seconds, "predict_seconds": predict_seconds})
                logger.log("TRAINING", "INFO", f"{name}: validation RMSE={validation_rows[-1]['RMSE']:.3f} "
                           f"(fit={fit_seconds:.2f}s)")
            validation_summary = pd.DataFrame(validation_rows).set_index("model").sort_values("RMSE")
            champion_name = str(validation_summary.index[0])
            ctx["summary"] = f"champion={champion_name} (validation RMSE={validation_summary.loc[champion_name, 'RMSE']:.2f})"
            progress.update("model_selection", {"champion": champion_name,
                                                "validation_rmse": {k: float(v) for k, v in validation_summary["RMSE"].items()}})

        # ---- 7. Final refit (Train+Validation) + held-out Test evaluation ---------------------------
        with logger.stage("EVALUATION", "Refitting on Train+Validation, scoring held-out Test") as ctx:
            X_tv, y_tv = pd.concat([X_train, X_val]), pd.concat([y_train, y_val])
            final_models, test_predictions, test_rows = {}, {}, []
            fit_seconds_map, predict_seconds_map, rows_used_map = {}, {}, {}
            for name, factory in factories.items():
                model = factory()
                t0 = time.perf_counter()
                model.fit(X_tv.to_numpy(dtype=float), y_tv.to_numpy(dtype=float))
                fit_seconds_map[name] = time.perf_counter() - t0
                t0 = time.perf_counter()
                prediction = np.asarray(model.predict(X_test.to_numpy(dtype=float)), dtype=float)
                predict_seconds_map[name] = time.perf_counter() - t0
                final_models[name] = model
                test_predictions[name] = prediction
                rows_used_map[name] = int(getattr(model, "n_training_rows_", len(y_tv)))
                test_rows.append({"model": name, **regression_metrics(y_test, prediction)})
                slug = MODEL_SLUGS[name]
                with (model_dir / f"{slug}.pkl").open("wb") as fh:
                    pickle.dump(model, fh)
            test_summary = pd.DataFrame(test_rows).set_index("model")
            test_summary.insert(0, "validation_RMSE", validation_summary["RMSE"])
            pd.DataFrame({"actual": y_test, **test_predictions}).to_csv(run_dir / "test_predictions.csv")
            ctx["summary"] = f"champion Test RMSE={test_summary.loc[champion_name, 'RMSE']:.2f}"
            progress.update("evaluation", {"champion_test_rmse": float(test_summary.loc[champion_name, "RMSE"]),
                                           "test_rmse": {k: float(v) for k, v in test_summary["RMSE"].items()}})

        # ---- 8. Inference: 24h recursive forecast ------------------------------------------------
        with logger.stage("INFERENCE", "Recursive multi-step forecast beyond the last observation") as ctx:
            horizon = cfg["forecast"]["horizon_hours"]
            forecast_model = factories[champion_name]()
            forecast_model.fit(X.to_numpy(dtype=float), y.to_numpy(dtype=float))
            forecast = recursive_forecast(forecast_model, series, horizon=horizon, feature_names=X.columns,
                                          lags=tuple(f_cfg["lags"]), rolling_windows=tuple(f_cfg["rolling_windows"]),
                                          include_calendar=f_cfg["include_calendar"])
            assert len(forecast) == horizon and forecast.index.min() > series.index.max()
            forecast.to_csv(run_dir / "future_24h_forecast.csv", header=True)
            with (model_dir / "champion_future_forecast.pkl").open("wb") as fh:
                pickle.dump(forecast_model, fh)
            with (model_dir / "champion.pkl").open("wb") as fh:
                pickle.dump(final_models[champion_name], fh)
            ctx["summary"] = f"{horizon}h forecast {forecast.index.min()} -> {forecast.index.max()}"
            progress.update("inference", {"forecast_start": str(forecast.index.min()), "forecast_end": str(forecast.index.max())})

        # ---- 9. Figures (10 PNGs) ------------------------------------------------------------------
        with logger.stage("VISUALIZATION", "Saving all evaluation figures") as ctx:
            figure_paths = save_all_figures(series, X_train, X_val, X_test, y_test, test_predictions,
                                            test_summary.reset_index().rename(columns={"RMSE": "test_RMSE"}),
                                            cv_detail, champion_name, final_models[champion_name], forecast, figure_dir)
            ctx["summary"] = f"{len(figure_paths)} figures written to {figure_dir}"
            progress.update("visualization", {"n_figures": len(figure_paths)})

        # ---- 10. Model Summary (csv/json/md + side files) -------------------------------------------
        with logger.stage("MODEL_SUMMARY", "Writing model_summary.{csv,json,md} and side metric files") as ctx:
            cv_cols = ["MAE", "MSE", "RMSE", "MAPE (%)", "sMAPE (%)", "R2"]
            cv_stats = cv_detail.groupby("model")[cv_cols].agg(["mean", "std"]) if len(cv_detail) else pd.DataFrame()
            summary_rows = []
            for name in factories:
                row: Dict[str, Any] = {
                    "run_id": run_id, "model": name, "selected_champion": name == champion_name,
                    "feature_count": int(X.shape[1]), "total_train_validation_rows": int(len(X_tv)),
                    "training_rows_used": rows_used_map[name], "fit_seconds": fit_seconds_map[name],
                    "predict_seconds_test": predict_seconds_map[name],
                    "model_file": str((model_dir / f"{MODEL_SLUGS[name]}.pkl").relative_to(PROJECT_ROOT)),
                    "model_parameters": json.dumps(final_models[name].get_params(), sort_keys=True),
                }
                for metric in cv_cols:
                    row[f"cv_{metric}_mean"] = float(cv_stats.loc[name, (metric, "mean")]) if name in getattr(cv_stats, "index", []) else float("nan")
                    row[f"cv_{metric}_std"] = float(cv_stats.loc[name, (metric, "std")]) if name in getattr(cv_stats, "index", []) else float("nan")
                for metric in cv_cols:
                    row[f"validation_{metric}"] = float(validation_summary.loc[name, metric])
                for metric in cv_cols:
                    row[f"test_{metric}"] = float(test_summary.loc[name, metric])
                summary_rows.append(row)
            model_summary_df = pd.DataFrame(summary_rows)
            summ_dir = dirs["model_summary_dir"]
            model_summary_df.to_csv(summ_dir / "model_summary.csv", index=False)
            model_summary_df.to_csv(run_dir / "model_summary.csv", index=False)
            payload = {"run_id": run_id, "champion_selected_by": "minimum validation_RMSE", "champion": champion_name,
                       "models": model_summary_df.to_dict(orient="records")}
            for target in (summ_dir / "model_summary.json", run_dir / "model_summary.json"):
                with open(target, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
            markdown_summary = model_summary_df[["model", "selected_champion", "cv_RMSE_mean", "validation_RMSE",
                                                 "test_RMSE", "fit_seconds", "training_rows_used"]].rename(columns={
                "model": "Mô hình",
                "selected_champion": "Được chọn",
                "cv_RMSE_mean": "RMSE CV trung bình",
                "validation_RMSE": "RMSE xác thực",
                "test_RMSE": "RMSE kiểm tra",
                "fit_seconds": "Thời gian khớp (giây)",
                "training_rows_used": "Số dòng huấn luyện",
            })
            markdown_summary["Mô hình"] = markdown_summary["Mô hình"].map(display_model_name)
            markdown_summary["Được chọn"] = markdown_summary["Được chọn"].map({True: "Có", False: "Không"})
            md_lines = [f"# Tóm tắt mô hình Lab 03 (lần chạy {run_id})", "",
                        f"Mô hình được chọn theo RMSE xác thực thấp nhất: **{display_model_name(champion_name)}**", "",
                        markdown_summary.round(3).to_markdown(index=False)]
            for target in (summ_dir / "model_summary.md", run_dir / "model_summary.md"):
                Path(target).write_text("\n".join(md_lines) + "\n", encoding="utf-8")
            cv_detail.to_csv(summ_dir / "cv_fold_metrics.csv", index=False)
            val_payload = {"run_id": run_id, "selection_metric": "RMSE",
                           "models": [{"model": n, **{k: float(v) for k, v in validation_summary.loc[n].items()}}
                                      for n in validation_summary.index]}
            with open(summ_dir / "validation_metrics.json", "w", encoding="utf-8") as fh:
                json.dump(val_payload, fh, ensure_ascii=False, indent=2, default=str)
            test_payload = {"run_id": run_id, "champion_selected_by_validation": champion_name,
                            "models": [{"model": n, "validation_RMSE": float(validation_summary.loc[n, "RMSE"]),
                                       **{k: float(v) for k, v in test_summary.loc[n].drop(labels="validation_RMSE").items()}}
                                       for n in test_summary.index]}
            with open(summ_dir / "test_metrics.json", "w", encoding="utf-8") as fh:
                json.dump(test_payload, fh, ensure_ascii=False, indent=2, default=str)
            validation_summary.to_csv(run_dir / "validation_metrics.csv")
            test_summary.to_csv(run_dir / "test_metrics.csv")
            ctx["summary"] = f"{len(model_summary_df)} models written to {summ_dir}"
            progress.update("model_summary", {"champion": champion_name})

        # ---- 11. Reporting (md/html/pdf) -----------------------------------------------------------
        with logger.stage("REPORTING", "Writing Lab03_Report.{md,html,pdf}") as ctx:
            legacy_path = get_path("reports_dir", cfg).parent / "lab03" / "legacy" / "Legacy_Experiment_Results.md"
            legacy_rows = []
            report = {
                "run_id": run_id, "status": "COMPLETED", "champion": champion_name,
                "summary_text": (f"Lab 03 huấn luyện {len(factories)} mô hình tự cài đặt trên dữ liệu AEP_hourly thật. "
                                 f"Mô hình được chọn theo RMSE xác thực thấp nhất là **{display_model_name(champion_name)}** "
                                 f"(RMSE xác thực={validation_summary.loc[champion_name, 'RMSE']:.2f}, "
                                 f"RMSE kiểm tra={test_summary.loc[champion_name, 'RMSE']:.2f})."),
                "dataset": ingestion_meta, "split": split_meta,
                "feature_config": {"lags": list(f_cfg["lags"]), "rolling_windows": list(f_cfg["rolling_windows"]),
                                   "include_calendar_features": f_cfg["include_calendar"]},
                "feature_count": int(X.shape[1]), "n_splits": cfg["preprocessing"]["cross_validation"]["n_splits"],
                "cv_summary": ([{"model": n, "RMSE_mean": float(cv_stats.loc[n, ("RMSE", "mean")]),
                                "RMSE_std": float(cv_stats.loc[n, ("RMSE", "std")]),
                                "MAE_mean": float(cv_stats.loc[n, ("MAE", "mean")]),
                                "R2_mean": float(cv_stats.loc[n, ("R2", "mean")])} for n in cv_stats.index]
                               if len(cv_detail) else []),
                "model_rows": [{"model": n, "cv_RMSE_mean": (float(cv_stats.loc[n, ("RMSE", "mean")]) if len(cv_detail) else None),
                               "validation_RMSE": float(validation_summary.loc[n, "RMSE"]),
                               "test_MAE": float(test_summary.loc[n, "MAE"]), "test_RMSE": float(test_summary.loc[n, "RMSE"]),
                               "test_MAPE (%)": float(test_summary.loc[n, "MAPE (%)"]), "test_R2": float(test_summary.loc[n, "R2"]),
                               "fit_seconds": fit_seconds_map[n]} for n in factories],
                "forecast_training_rows": int(len(X)), "forecast_training_end": str(X.index.max()),
                "forecast_horizon": horizon, "legacy_snapshot": legacy_rows,
                "figures": figure_paths,
                "figure_relative_paths": {k: f"figures/{Path(v).name}" for k, v in figure_paths.items()},
            }
            paths = write_report_bundle(report, run_dir)
            latest_report = {
                **report,
                "figure_relative_paths": {
                    key: f"notebook_runs/{run_id}/figures/{Path(value).name}"
                    for key, value in figure_paths.items()
                },
            }
            latest_paths = write_report_bundle(latest_report, get_path("reports_dir", cfg))
            ctx["summary"] = f"report -> {latest_paths['pdf']}"
            progress.update("reporting", latest_paths)

        progress.finish("COMPLETED")
        elapsed = time.perf_counter() - t_start
        logger.log("PIPELINE", "SUCCESS", f"Lab 03 pipeline COMPLETED in {elapsed:.1f}s | champion={champion_name}")
        return {"run_id": run_id, "status": "COMPLETED", "champion": champion_name, "run_dir": str(run_dir),
                "report": latest_paths, "model_summary_dir": str(summ_dir), "elapsed_sec": round(elapsed, 2)}
    except Exception as exc:
        progress.finish("FAILED")
        logger.log("PIPELINE", "ERROR", f"Lab 03 pipeline FAILED: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    print(run_pipeline())
