"""
AUTOMATED EXPERIMENT PIPELINE ENTRY POINT — run_lab02_experiments.py
================================================================================
PURPOSE:
    Standalone CLI orchestrator for the LAB02 Automated Model Experimentation
    Pipeline. Operates independently of Jupyter Notebooks while sharing 100% of
    the core Python modules (`src/data_loader.py`, `src/lab02_utils/*`, and
    `src/lab02_models/*`).

SUPPORTED CLI COMMANDS:
    python run_lab02_experiments.py --dataset dataset1
    python run_lab02_experiments.py --dataset dataset2
    python run_lab02_experiments.py --dataset all
    python run_lab02_experiments.py --dataset all --mode full
    python run_lab02_experiments.py --dataset dataset1 --mode standard
    $env:PYTHONPATH = ".."
    python run_lab02_experiments.py --dataset dataset2 --mode auto
    python run_lab02_experiments.py --dataset dataset1 --mode auto
================================================================================
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any, List
import matplotlib

# Sử dụng backend không giao diện khi chạy tự động qua CLI để không chặn terminal
matplotlib.use("Agg")

current_path = Path(__file__).resolve().parent
if (current_path / "src").is_dir():
    PROJECT_ROOT = current_path
elif (current_path.parent / "src").is_dir():
    PROJECT_ROOT = current_path.parent
elif (current_path.parent.parent / "src").is_dir():
    PROJECT_ROOT = current_path.parent.parent
else:
    PROJECT_ROOT = current_path

for p in [str(PROJECT_ROOT), str(current_path)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.data_loader import load_and_prep_data, split_train_test_chronological
from src.lab02.lab02_utils.logging_config import (
    setup_experiment_logger,
    log_system_status,
)
from src.lab02.lab02_utils.stats import perform_adf_test
from src.lab02.lab02_utils.acf_pacf_analysis import analyze_acf_pacf
from src.lab02.lab02_utils.model_summary import ModelSummaryRegistry, format_terminal_model_card
from src.lab02.lab02_models import (
    run_ar_experiment,
    run_ma_experiment,
    run_arima_experiment,
    run_sarima_experiment,
    get_candidate_models,
    auto_identify_candidates_from_data,
)


DATASET_SPECIFICATIONS: Dict[str, Dict[str, Any]] = {
    "dataset1": {
        "dataset_key": "dataset1",
        "dataset_title": "Monthly Retail Sales (US, 07/1991 - 06/2008)",
        "raw_file": "Monthly Retail Sales.csv",
        "date_col": "date",
        "value_col": "value",
        "freq": "MS",
        "seasonal_period": 12,
    },
    "dataset2": {
        "dataset_key": "dataset2",
        "dataset_title": "AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)",
        "raw_file": "AirPassengers.csv",
        "date_col": "Month",
        "value_col": "Passengers",
        "freq": "MS",
        "seasonal_period": 12,
    },
}


def run_dataset_experiment_suite(
    dataset_key: str,
    mode: str = "full",
    save_figures: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Purpose:
        Execute the complete automated experiment pipeline for a single dataset
        ('dataset1' or 'dataset2'):
        1. Initialize non-overwriting experiment log:
           logs/lab02/experiments/<dataset_key>/experiment_<RUN_ID>.log
        2. Load raw CSV and split Train (80%) / Test (20%) chronologically (shuffle=False).
        3. Run ADF stationarity tests and ACF/PACF analysis strictly on Train.
        4. Iterate through the Candidate Model Registry (AR, MA, ARIMA, SARIMA, Log-SARIMA),
           recording both succeeded and failed candidates with full telemetry.
        5. Export CSV, JSON, and Markdown Model Summaries (both canonical and RUN_ID-tagged).

    Parameters:
        dataset_key (str): 'dataset1' or 'dataset2'.
        mode (str): 'standard' or 'full' candidate set.
        save_figures (bool): Whether to save diagnostic and forecast PNG figures.
        verbose (bool): Whether to print per-model statsmodels output to console.

    Returns:
        Dict[str, Any]: Execution summary including run_id, log_path, summary_paths,
            comparison_df, and interpretation.
    """
    if dataset_key not in DATASET_SPECIFICATIONS:
        raise ValueError(f"Unsupported dataset_key='{dataset_key}'. Choose 'dataset1' or 'dataset2'.")

    spec = DATASET_SPECIFICATIONS[dataset_key]
    ds_title = spec["dataset_title"]

    # 1. Khởi tạo logger độc lập cho từng lần chạy thực nghiệm (không ghi đè log cũ)
    logger, run_id, log_path = setup_experiment_logger(dataset_name=dataset_key, console_output=False)

    print("=" * 88)
    print(f"[AUTOMATED EXPERIMENT PIPELINE] DATASET: {ds_title}")
    print(f"  • Run ID       : {run_id}")
    print(f"  • Mode         : {mode.upper()}")
    print(f"  • Log File     : {log_path}")
    print("=" * 88)

    raw_path = PROJECT_ROOT / "data" / "raw" / spec["raw_file"]
    proc_dir = PROJECT_ROOT / "data" / "processed" / "lab02" / dataset_key
    fig_dir = (PROJECT_ROOT / "reports" / "figures" / "lab02" / dataset_key) if save_figures else None

    # 2. Load dữ liệu & Chia Train/Test (80/20, shuffle=False)
    df = load_and_prep_data(
        filepath=raw_path,
        date_col=spec["date_col"],
        freq=spec["freq"],
        dataset_name=dataset_key,
    )
    train_df, test_df, split_meta = split_train_test_chronological(
        df_or_series=df,
        test_ratio=0.2,
        dataset_name=dataset_key,
        save_dir=proc_dir,
        differencing_info="d=1 (AR/MA/ARIMA) & d=1, D=1, s=12 (SARIMA/Log-SARIMA)",
        transformations="Original Scale & Natural Log ln(Y_t) with Inverse exp()",
    )

    val_col = spec["value_col"]
    train = train_df[val_col].asfreq(spec["freq"])
    test = test_df[val_col].asfreq(spec["freq"])

    # 3. Kiểm định ADF & Phân tích ACF/PACF trên tập Train
    _ = perform_adf_test(train, title=f"{ds_title} - Train Raw", dataset_name=dataset_key, return_dict=True)
    diff_1 = train.diff(1)
    _ = perform_adf_test(diff_1, title=f"{ds_title} - Train Diff(d=1)", dataset_name=dataset_key, return_dict=True)
    diff_1_12 = diff_1.diff(spec["seasonal_period"])
    _ = perform_adf_test(
        diff_1_12,
        title=f"{ds_title} - Train Diff(d=1, D=1, s={spec['seasonal_period']})",
        dataset_name=dataset_key,
        return_dict=True,
    )

    _ = analyze_acf_pacf(
        series=diff_1,
        dataset_name=dataset_key,
        nlags=36,
        differencing_info="d=1",
        seasonal_period=spec["seasonal_period"],
        output_dir=fig_dir,
        file_prefix=f"{dataset_key}_d1",
        show_plot=False,
    )
    _ = analyze_acf_pacf(
        series=diff_1_12,
        dataset_name=dataset_key,
        nlags=36,
        differencing_info=f"d=1, D=1, s={spec['seasonal_period']}",
        seasonal_period=spec["seasonal_period"],
        output_dir=fig_dir,
        file_prefix=f"{dataset_key}_d1_D1_s{spec['seasonal_period']}",
        show_plot=False,
    )

    # 4. Khởi tạo ModelSummaryRegistry gắn với RUN_ID hiện tại (hỗ trợ tương thích ngược)
    try:
        registry = ModelSummaryRegistry(
            dataset_name=dataset_key,
            dataset_title=ds_title,
            n_obs=len(df),
            date_range=f"{df.index.min().date()} to {df.index.max().date()}",
            frequency=spec["freq"],
            train_size=len(train),
            test_size=len(test),
            run_id=run_id,
        )
    except TypeError:
        registry = ModelSummaryRegistry(
            dataset_name=dataset_key,
            dataset_title=ds_title,
            n_obs=len(df),
            date_range=f"{df.index.min().date()} to {df.index.max().date()}",
            frequency=spec["freq"],
            train_size=len(train),
            test_size=len(test),
        )
        registry.run_id = run_id

    # 5. Xác định danh sách ứng viên trong Candidate Model Registry
    if str(mode).lower() == "auto":
        print("\n[AUTO-IDENTIFICATION] Đang tự động nhận diện tham số mô hình từ dữ liệu Train...")
        auto_pack = auto_identify_candidates_from_data(
            train_series=train,
            dataset_name=dataset_key,
            seasonal_period=spec["seasonal_period"],
            include_log=True,
        )
        print(auto_pack["rationale_summary"])
        logger.info(f"[AUTO_IDENTIFY] {auto_pack['rationale_summary']}")
        candidates = auto_pack["candidates"]
    else:
        candidates = get_candidate_models(dataset_key, mode=mode)
    logger.info(f"[CANDIDATE_REGISTRY] Loaded {len(candidates)} candidate configurations (mode='{mode}')")
    print(f"\n>>> Đang huấn luyện và đánh giá {len(candidates)} mô hình ứng viên cho {dataset_key.upper()}...")
    print("-" * 88)
    print(
        f"{'STT':<4} {'Mô hình ứng viên':<30} {'Họ':<11} {'AIC':>9} {'RMSE':>8} {'MAPE(%)':>8} {'R²':>8} {'Ljung-Box p':>11} {'Trạng thái'}"
    )
    print("-" * 88)

    for idx, cand in enumerate(candidates, 1):
        m_name = cand["model_name"]
        m_family = cand["model_family"]
        order = tuple(cand["order"])
        seasonal_order = tuple(cand["seasonal_order"])
        use_log = bool(cand.get("use_log_transform", False))
        rationale = str(cand.get("rationale", ""))

        t_start = time.perf_counter()
        try:
            if m_family == "AR":
                exp = run_ar_experiment(
                    train=train,
                    test=test,
                    p=order[0],
                    d=order[1],
                    dataset_name=dataset_key,
                    fig_dir=fig_dir,
                    show_plot=False,
                    verbose=verbose,
                )
            elif m_family == "MA":
                exp = run_ma_experiment(
                    train=train,
                    test=test,
                    q=order[2],
                    d=order[1],
                    dataset_name=dataset_key,
                    fig_dir=fig_dir,
                    show_plot=False,
                    verbose=verbose,
                )
            elif m_family == "ARIMA":
                exp = run_arima_experiment(
                    train=train,
                    test=test,
                    order=order,
                    dataset_name=dataset_key,
                    model_name=m_name,
                    fig_dir=fig_dir,
                    show_plot=False,
                    verbose=verbose,
                    rationale=rationale,
                )
            else:
                exp = run_sarima_experiment(
                    train=train,
                    test=test,
                    order=order,
                    seasonal_order=seasonal_order,
                    use_log_transform=use_log,
                    dataset_name=dataset_key,
                    model_name=m_name,
                    fig_dir=fig_dir,
                    plot_diagnostics_flag=save_figures,
                    plot_forecast_flag=save_figures,
                    show_plot=False,
                    verbose=verbose,
                    rationale=rationale,
                )
            exp["rationale"] = rationale
            rec = registry.register_experiment(exp)

            stat = rec["statistical_results"]
            ev = rec["forecast_evaluation"]
            diag = rec["residual_diagnostics"]
            wn_tag = "Pass (WN)" if diag["is_white_noise"] else "Fail (Autocorr)"
            conv_tag = "" if stat["converged"] else " [WARN:NonConv]"
            print(
                f"{idx:<4} {m_name:<30} {m_family:<11} {stat['aic']:>9.2f} {ev['rmse']:>8.2f} "
                f"{ev['mape']:>7.2f}% {ev['r2']:>8.4f} {diag['ljung_box_p']:>11.4f} {wn_tag}{conv_tag}"
            )
        except Exception as exc:
            dur = time.perf_counter() - t_start
            registry.register_failure(
                model_name=m_name,
                model_family=m_family,
                order=order,
                seasonal_order=seasonal_order,
                transformation="Log Scale" if use_log else "Original Scale",
                rationale=rationale,
                error_message=str(exc),
                duration_seconds=dur,
            )
            print(f"{idx:<4} {m_name:<30} {m_family:<11} {'N/A':>9} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>11} FAILED: {exc}")

    print("-" * 88)

    # 6. Xuất báo cáo Model Summary (CSV, JSON, Markdown) kèm bản lưu gắn nhãn RUN_ID
    summary_paths = registry.export_all(base_dir=PROJECT_ROOT, save_run_tagged_copy=True)
    comp_df = registry.get_comparison_dataframe()
    interp = registry.generate_automatic_interpretation()

    best_crit = interp.get("best_by_criterion", {})
    print(f"\n[KẾT QUẢ TỔNG HỢP THEO TIÊU CHÍ — {dataset_key.upper()} | RUN_ID: {run_id}]")
    if best_crit:
        print(
            f"  • RMSE thấp nhất (Test)                : {best_crit['lowest_test_rmse']['model']} "
            f"(RMSE = {best_crit['lowest_test_rmse']['value']:.2f})"
        )
        print(
            f"  • MAE thấp nhất (Test)                 : {best_crit['lowest_test_mae']['model']} "
            f"(MAE = {best_crit['lowest_test_mae']['value']:.2f})"
        )
        print(
            f"  • MAPE thấp nhất (Test)                : {best_crit['lowest_test_mape']['model']} "
            f"(MAPE = {best_crit['lowest_test_mape']['value']:.2f}%)"
        )
        if 'highest_test_r2' in best_crit:
            print(
                f"  • R² cao nhất (Test)                   : {best_crit['highest_test_r2']['model']} "
                f"(R² = {best_crit['highest_test_r2']['value']:.4f})"
            )
        print(
            f"  • Đạt chuẩn White Noise & RMSE tốt nhất: {best_crit['lowest_test_rmse_with_white_noise_adequacy']}"
        )
        for grp, ginfo in best_crit.get("best_information_criteria_by_scale_group", {}).items():
            print(
                f"  • AIC thấp nhất nhóm [{grp:<18}]: {ginfo['lowest_aic_model']} (AIC = {ginfo['aic']:.2f})"
            )

    print("\n[ĐƯỜNG DẪN FILE XUẤT RA]")
    print(f"  • Experiment Log : {log_path}")
    print(f"  • Summary CSV    : {summary_paths['csv']}")
    print(f"  • Summary JSON   : {summary_paths['json']}")
    print(f"  • Summary MD     : {summary_paths['md']}")
    print(f"  • Run Archive MD : {summary_paths.get('run_md', 'N/A')}\n")

    return {
        "dataset_key": dataset_key,
        "run_id": run_id,
        "log_path": str(log_path),
        "summary_paths": summary_paths,
        "comparison_df": comp_df,
        "interpretation": interp,
        "registry": registry,
    }


def parse_cli_args() -> argparse.Namespace:
    """
    Purpose:
        Parse command-line arguments for run_lab02_experiments.py.
    """
    parser = argparse.ArgumentParser(
        description="LAB02 Automated Time Series Experiment Runner (AR / MA / ARIMA / SARIMA / Log-SARIMA)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["dataset1", "dataset2", "all"],
        help="Target dataset to run experiments on: 'dataset1', 'dataset2', or 'all' (default: 'all').",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["standard", "full", "auto"],
        help="Candidate set mode: 'standard' (core subset) or 'full' (all statistically grounded candidates).",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip saving diagnostic and forecast PNG figures to speed up execution.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose per-model fitting output to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    """
    Purpose:
        Main CLI entry point orchestrating requested dataset experiment runs.
    """
    args = parse_cli_args()
    target_datasets = ["dataset1", "dataset2"] if args.dataset == "all" else [args.dataset]

    for ds_key in target_datasets:
        run_dataset_experiment_suite(
            dataset_key=ds_key,
            mode=args.mode,
            save_figures=not args.no_figures,
            verbose=args.verbose,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
