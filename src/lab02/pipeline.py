r"""
Mô-đun Điều phối Đường ống Thực thi Toàn trình (End-to-End Pipeline Orchestrator) cho Lab 02.
Nối trực tiếp runner thực nghiệm gốc (script/lab02/run_lab02_experiments.py) với:
- Hệ thống tracking & progress_results/lab02/latest_run.json
- Xuất docs/lab02/experiment_logs.csv (chuẩn Lab04-ML)
- Kết xuất 2 bộ báo cáo tách riêng LAB_02_DATASET1_REPORT & LAB_02_DATASET2_REPORT (HTML, PDF, MD).
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.lab02 import data_loader, report
from src.lab02.stats import perform_adf_test
from src.lab02.tracking import init_run_session, record_stage_result, write_log
from src.lab02.utils import PROJECT_ROOT, ensure_directories, get_path, load_config, save_json, set_global_seed


def _load_experiment_runner() -> Any:
    r"""
    Purpose: Nạp động script/lab02/run_lab02_experiments.py để tái sử dụng 100% pipeline mô hình gốc Lab 02.
    Input-Output:
        - Input: Không có.
        - Output: Mô-đun run_lab02_experiments đã nạp.
    How: Sử dụng importlib.util.spec_from_file_location trỏ tới script/lab02/run_lab02_experiments.py.
    """
    target = PROJECT_ROOT / "script" / "lab02" / "run_lab02_experiments.py"
    spec = importlib.util.spec_from_file_location("lab02_exp_runner", target)
    if spec is None or spec.loader is None:
        raise ImportError(f"Không thể nạp run_lab02_experiments từ: {target}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lab02_exp_runner"] = mod
    spec.loader.exec_module(mod)
    return mod


def export_combined_experiment_logs_csv(cfg: Optional[Dict[str, Any]] = None) -> Path:
    r"""
    Purpose: Tổng hợp kết quả mô hình của cả Dataset 1 và Dataset 2 vào docs/lab02/experiment_logs.csv (giống Lab04-ML).
    Input-Output:
        - Input: cfg (Optional[Dict[str, Any]]).
        - Output: Đường dẫn Path tới tệp docs/lab02/experiment_logs.csv.
    How: Đọc dataset1_model_summary.csv và dataset2_model_summary.csv, gắn cột dataset_key rồi gộp bằng pd.concat.
    """
    config = cfg if cfg is not None else load_config()
    ms_dir = get_path("model_summary_dir", config)
    docs_dir = get_path("docs_dir", config)
    docs_dir.mkdir(parents=True, exist_ok=True)

    frames: List[pd.DataFrame] = []
    for ds_key in ("dataset1", "dataset2"):
        csv_p = ms_dir / ds_key / f"{ds_key}_model_summary.csv"
        if csv_p.exists():
            df_part = pd.read_csv(csv_p)
            df_part.insert(0, "dataset_key", ds_key)
            frames.append(df_part)

    out_csv = docs_dir / "experiment_logs.csv"
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(out_csv, index=False, encoding="utf-8-sig")
    return out_csv


def run_full_pipeline(
    datasets: Optional[List[str]] = None,
    mode: str = "full",
    save_figures: bool = True,
    verbose: bool = False,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    r"""
    Purpose: Thực thi toàn trình Lab 02 cho 1 hoặc cả 2 dataset, ghi log, lưu progress_results và xuất 2 bộ báo cáo riêng.
    Input-Output:
        - Input: datasets (List[str]), mode ('standard'|'full'|'auto'), save_figures (bool), verbose (bool), run_id (Optional[str]).
        - Output: Từ điển tổng hợp kết quả từng dataset, đường dẫn experiment_logs.csv và 2 bộ báo cáo.
    How: Khởi tạo run session, kiểm toán dữ liệu + ADF từng dataset, gọi run_dataset_experiment_suite(), ghi JSON và render report.
    """
    t0 = time.perf_counter()
    cfg = load_config()
    ensure_directories(cfg)
    set_global_seed(int(cfg["project"]["random_seed"]))

    active_run_id = init_run_session(run_id)
    target_datasets = datasets if datasets is not None else ["dataset1", "dataset2"]
    write_log(active_run_id, "PIPELINE", "START", f"Khởi chạy Pipeline Lab 02 | datasets={target_datasets} | mode={mode}.")

    runner = _load_experiment_runner()
    metrics_dir = get_path("metrics_dir", cfg)
    dataset_results: Dict[str, Any] = {}

    for ds_key in target_datasets:
        ds_t0 = time.perf_counter()
        df, data_meta = data_loader.load_dataset(ds_key, cfg)
        train_df, test_df, split_meta = data_loader.split_dataset(df, ds_key, cfg)

        val_col = data_meta["value_col"]
        s_period = int(data_meta["seasonal_period"])
        train_s = train_df[val_col].asfreq(data_meta["frequency"])

        adf_raw = perform_adf_test(train_s, title=f"{ds_key} Train Raw", dataset_name=ds_key, return_dict=True)
        adf_d1 = perform_adf_test(train_s.diff(1), title=f"{ds_key} Train Diff(d=1)", dataset_name=ds_key, return_dict=True)
        adf_d1_D1 = perform_adf_test(
            train_s.diff(1).diff(s_period),
            title=f"{ds_key} Train Diff(d=1, D=1, s={s_period})",
            dataset_name=ds_key,
            return_dict=True,
        )

        suite_res = runner.run_dataset_experiment_suite(
            dataset_key=ds_key,
            mode=mode,
            save_figures=save_figures,
            verbose=verbose,
        )

        fig_dir = get_path("figures_dir", cfg) / ds_key
        figs = sorted(p.name for p in fig_dir.glob("*.png")) if fig_dir.exists() else []

        ds_payload: Dict[str, Any] = {
            "dataset_meta": data_meta,
            "split_meta": split_meta,
            "adf_tests": {
                "raw": adf_raw,
                "diff_1": adf_d1,
                "diff_1_seasonal_12": adf_d1_D1,
            },
            "experiment_run_id": suite_res["run_id"],
            "experiment_log_path": suite_res["log_path"],
            "summary_paths": suite_res["summary_paths"],
            "interpretation": suite_res["interpretation"],
            "figures": figs,
            "figures_count": len(figs),
            "duration_sec": round(time.perf_counter() - ds_t0, 4),
        }
        save_json(ds_payload, metrics_dir / f"{ds_key}_pipeline_summary.json")
        record_stage_result(active_run_id, ds_key, ds_payload)
        write_log(
            active_run_id,
            ds_key.upper(),
            "SUCCESS",
            f"Hoàn tất {ds_key} ({figs.__len__()} biểu đồ, exp_run_id={suite_res['run_id']}).",
        )
        dataset_results[ds_key] = ds_payload

    exp_csv = export_combined_experiment_logs_csv(cfg)
    rep_res = report.main(run_id=active_run_id, datasets=target_datasets)

    total_sec = time.perf_counter() - t0
    write_log(active_run_id, "PIPELINE", "COMPLETED", f"Hoàn tất toàn trình Lab 02 trong {total_sec:.2f}s.")

    return {
        "run_id": active_run_id,
        "datasets": dataset_results,
        "experiment_logs_csv": str(exp_csv),
        "reporting": rep_res,
        "total_duration_sec": round(total_sec, 4),
    }


def main(
    datasets: Optional[List[str]] = None,
    mode: str = "full",
    save_figures: bool = True,
    verbose: bool = False,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    r"""
    Purpose: Điểm vào chuẩn gọi run_full_pipeline() cho Lab 02.
    Input-Output:
        - Input: datasets, mode, save_figures, verbose, run_id.
        - Output: Từ điển kết quả toàn trình Lab 02.
    How: Ủy quyền trực tiếp sang run_full_pipeline().
    """
    return run_full_pipeline(
        datasets=datasets,
        mode=mode,
        save_figures=save_figures,
        verbose=verbose,
        run_id=run_id,
    )