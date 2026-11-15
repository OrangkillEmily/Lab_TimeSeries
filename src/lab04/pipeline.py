"""
Module điều phối quy trình thực thi tổng thể (`src/lab04/pipeline.py`) cho dự án Lab 04 Time Series.

Trách nhiệm của module:
1. Kết nối tuần tự toàn bộ 6 giai đoạn của bài thực hành Lab 04 trong một luồng thống nhất:
   `DATA_LOADING` -> `EDA` -> `PREPROCESSING (70/15/15 + 5-Fold CV)` -> `TRAINING (Auto-GPU)`
   -> `EVALUATION (5 Metrics + Walk-Forward + PEMS-BAY)` -> `REPORTING (.md, .html, .pdf)`.
2. Tạo và truyền một mã định danh `Run_ID` duy nhất xuyên suốt tất cả các bước để nhật ký
   (`logs/lab04/pipeline_<run_id>.log`), tiến độ (`progress_results/lab04/progress_<run_id>.json`)
   và báo cáo (`reports/lab04/LAB_04_REPORT.*`, `docs/lab04/LAB_04_REPORT.*`) đồng bộ 100%.
3. Hỗ trợ các cờ dòng lệnh (`--skip-eda`, `--skip-preprocess`, `--skip-train`, `--skip-kfold`, `--skip-pems`).
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve()
ROOT_DIR = HERE.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.lab04 import data_loader, eda, evaluate, preprocess, report, train
from src.lab04.tracking import create_run, finish_run, update_progress, write_log
from src.lab04.utils import (
    configure_console_utf8,
    detect_hardware_accelerator,
    load_config,
    set_global_seeds,
)


def run_pipeline(
    skip_eda: bool = False,
    skip_preprocess: bool = False,
    skip_train: bool = False,
    skip_kfold: bool = False,
    skip_pems: bool = False,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    What: Thực thi toàn bộ quy trình dự báo chuỗi thời gian đa biến của Lab 04 từ đầu đến cuối.
    Why: Cho phép chạy trọn gói cả Yêu cầu 1–11, Yêu cầu 12 (PEMS-BAY DCRNN), 5-Fold CV,
         Walk-Forward Backtesting và xuất báo cáo `.pdf`, `.html`, `.md` chỉ bằng 1 lệnh duy nhất.
    Input:
        - skip_eda (bool): Bỏ qua bước EDA nếu `True`.
        - skip_preprocess (bool): Bỏ qua bước tiền xử lý nếu `True`.
        - skip_train (bool): Bỏ qua bước huấn luyện nếu `True`.
        - skip_kfold (bool): Bỏ qua 5-Fold CV nếu `True`.
        - skip_pems (bool): Bỏ qua bài tập 12 nâng cao nếu `True`.
        - run_id (Optional[str]): Mã định danh phiên chạy tùy chọn.
    Output:
        - Dict[str, Any]: Tóm tắt kết quả thực thi toàn trình.
    How: Khởi tạo `run_id`, dò tìm phần cứng `Auto-GPU`, chạy tuần tự từng giai đoạn và đóng phiên.
    """
    configure_console_utf8()
    cfg = load_config()
    seed = int(cfg.get("project", {}).get("random_state", 42))
    seed_info = set_global_seeds(seed)
    hw_info = detect_hardware_accelerator(cfg)

    active_run_id = create_run(
        run_id=run_id,
        metadata={
            "seed_info": seed_info,
            "hardware": hw_info,
            "config_path": "config.yaml",
        },
    )

    pipeline_start = time.perf_counter()
    print("\n" + "#" * 74)
    print("   KHỞI ĐỘNG PIPELINE LAB04: MULTIVARIATE TIME SERIES FORECASTING")
    print("   Bài 1-11: Electricity Load + Weather (VAR, XGBoost, LSTM)")
    print("   Bài 12  : Highway Traffic PEMS-BAY với DCRNN (15m, 30m, 60m)")
    print("#" * 74)
    print(f"[INFO] Run ID        : {active_run_id}")
    print(f"[INFO] Random Seed   : {seed}")
    print(f"[INFO] Auto-GPU Dev  : {hw_info['device_name']} (torch={hw_info['torch_device']}, xgb={hw_info['xgb_device']})")

    write_log(active_run_id, "PIPELINE", "START", f"Bắt đầu Pipeline Lab 04 (Run_ID={active_run_id}).")

    completed_steps: List[str] = []
    skipped_steps: List[str] = []
    step_outputs: Dict[str, Any] = {}

    steps_plan = [
        ("DATA_LOADING", lambda: data_loader.main(active_run_id), False),
        ("EDA", lambda: eda.main(active_run_id), skip_eda),
        ("PREPROCESSING", lambda: preprocess.main(active_run_id), skip_preprocess),
        ("TRAINING", lambda: train.main(active_run_id, skip_pems=skip_pems), skip_train),
        ("EVALUATION", lambda: evaluate.main(active_run_id, skip_kfold=skip_kfold, skip_pems=skip_pems), False),
        ("REPORTING", lambda: report.main(active_run_id), False),
    ]

    try:
        for step_name, fn, should_skip in steps_plan:
            if should_skip:
                print(f"[SKIP] Bỏ qua bước {step_name} theo cờ dòng lệnh.")
                skipped_steps.append(step_name)
                update_progress(active_run_id, step_name, "SKIPPED", output_data={"message": "Skipped by CLI flag"})
                write_log(active_run_id, step_name, "SKIPPED", "Bỏ qua theo cờ dòng lệnh.")
                continue

            step_outputs[step_name] = fn()
            completed_steps.append(step_name)

        total_duration = round(time.perf_counter() - pipeline_start, 4)
        summary = {
            "run_id": active_run_id,
            "status": "COMPLETED",
            "total_duration_sec": total_duration,
            "completed_steps": completed_steps,
            "skipped_steps": skipped_steps,
            "hardware": hw_info,
            "reports": step_outputs.get("REPORTING", {}),
        }
        finish_run(active_run_id, status="COMPLETED", summary=summary)
        write_log(active_run_id, "PIPELINE", "SUCCESS", f"Hoàn tất toàn bộ Pipeline Lab 04 trong {total_duration:.2f}s.")
        print("\n" + "=" * 74)
        print(f"   HOÀN TẤT PIPELINE LAB04 ({total_duration:.2f}s) — RUN_ID: {active_run_id}")
        print(f"   Báo cáo MD  : {summary['reports'].get('report_md')}")
        print(f"   Báo cáo HTML: {summary['reports'].get('report_html')}")
        print(f"   Báo cáo PDF : {summary['reports'].get('report_pdf')}")
        print("=" * 74 + "\n")
        return summary

    except Exception as exc:
        err_msg = f"Lỗi tại Pipeline: {exc}\n{traceback.format_exc()}"
        write_log(active_run_id, "PIPELINE", "FAILED", err_msg)
        finish_run(active_run_id, status="FAILED", summary={"error": str(exc)})
        raise


def main() -> None:
    """
    What: Phân tích tham số dòng lệnh và kích hoạt `run_pipeline()`.
    Why: Hỗ trợ chạy linh hoạt từ `script/lab04/run_lab04.py` hoặc `python -m src.lab04.pipeline`.
    Input: Các cờ dòng lệnh từ `sys.argv`.
    Output: Không có.
    How: Dùng `argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(description="Chạy Pipeline Thực nghiệm Lab 04 Time Series.")
    parser.add_argument("--skip-eda", action="store_true", help="Bỏ qua bước EDA.")
    parser.add_argument("--skip-preprocess", action="store_true", help="Bỏ qua bước Preprocessing.")
    parser.add_argument("--skip-train", action="store_true", help="Bỏ qua bước Training.")
    parser.add_argument("--skip-kfold", action="store_true", help="Bỏ qua bước 5-Fold Cross-Validation.")
    parser.add_argument("--skip-pems", action="store_true", help="Bỏ qua bài tập 12 nâng cao PEMS-BAY.")
    parser.add_argument("--bundle-docs", action="store_true", help="Đồng bộ ảnh sang docs/lab04/images/.")
    parser.add_argument("--run-id", type=str, default=None, help="Mã định danh Run_ID tùy chọn.")
    args, _ = parser.parse_known_args()

    run_pipeline(
        skip_eda=args.skip_eda,
        skip_preprocess=args.skip_preprocess,
        skip_train=args.skip_train,
        skip_kfold=args.skip_kfold,
        skip_pems=args.skip_pems,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()