r"""
Mô-đun Điều phối Đường ống Thực thi Toàn trình (End-to-End Pipeline Orchestrator) cho Bài thực hành Lab 04.
Chạy tuần tự và đồng bộ 7 giai đoạn:
1. Data Loading & Schema Audit (src/lab04/data_loader.py)
2. Exploratory Data Analysis & 7 EDA Figures (src/lab04/eda.py)
3. Anti-Leakage Preprocessing, ADF Stationarity & 5-Fold TimeSeriesSplit (src/lab04/preprocess.py)
4. Multi-Model GPU Training: VAR, XGBoost, Residual LSTM & PEMS-BAY Benchmark (src/lab04/train.py)
5. Comprehensive 5-Metric Evaluation & Walk-Forward Backtesting (src/lab04/evaluate.py)
6. Academic Visualization Suite - 23 Figures (src/lab04/visualize.py)
7. Automated Synchronized Reporting: HTML + PDF + Markdown (src/lab04/report.py & script/lab04/render_report.py)
"""

import time
from typing import Any, Dict, Optional

from src.lab04 import data_loader, eda, evaluate, preprocess, report, train, visualize
from src.lab04.tracking import init_run_session, write_log
from src.lab04.utils import ensure_directories, get_compute_device, load_config, set_global_seed


def run_full_pipeline(run_id: Optional[str] = None) -> Dict[str, Any]:
    r"""
    Purpose: Thực thi tự động từ đầu đến cuối toàn bộ 7 giai đoạn của bài thực hành Lab 04.
    Input-Output:
        - Input: run_id (Optional[str]).
        - Output: Từ điển tổng hợp kết quả của cả 7 giai đoạn và đường dẫn báo cáo cuối cùng.
    How: Khởi tạo phiên chạy, cố định seed = 42, gọi tuần tự 7 mô-đun và ghi nhật ký thời gian thực.
    """
    t0 = time.perf_counter()
    cfg = load_config()
    ensure_directories(cfg)
    set_global_seed(int(cfg.get("project", {}).get("random_seed", 42)))

    active_run_id = init_run_session(run_id)
    hw_info = get_compute_device(cfg)

    print("=" * 88)
    print(f"BẮT ĐẦU PIPELINE LAB 04 (V2.0) | RUN ID: {active_run_id}")
    print(f"Thiết bị Tính toán: {hw_info['device_name']} (torch={hw_info['torch_device']}, xgb={hw_info['xgb_device']}, AMP={hw_info['amp_enabled']})")
    print("=" * 88)

    write_log(active_run_id, "PIPELINE", "START", f"Khởi chạy toàn trình Lab 04 trên {hw_info['device_name']}.")

    s1_data = data_loader.main(active_run_id)
    s2_eda = eda.main(active_run_id)
    s3_prep = preprocess.main(active_run_id)
    s4_train = train.main(active_run_id)
    s5_eval = evaluate.main(active_run_id)
    s6_vis = visualize.main(active_run_id)
    s7_rep = report.main(active_run_id)

    total_sec = time.perf_counter() - t0
    write_log(
        active_run_id,
        "PIPELINE",
        "COMPLETED",
        f"Hoàn tất toàn bộ 7 giai đoạn Lab 04 trong {total_sec:.2f}s.",
    )

    print("=" * 88)
    print(f"HOÀN TẤT PIPELINE LAB 04 TRONG {total_sec:.2f} GIÂY!")
    print(f"  - Báo cáo HTML : {s7_rep.get('html_report')}")
    print(f"  - Báo cáo PDF  : {s7_rep.get('pdf_report')}")
    print(f"  - Báo cáo MD   : {s7_rep.get('md_report')}")
    print("=" * 88)

    return {
        "run_id": active_run_id,
        "hardware": hw_info,
        "data_loading": s1_data,
        "eda": s2_eda,
        "preprocessing": s3_prep,
        "training": s4_train,
        "evaluation": s5_eval,
        "visualization": s6_vis,
        "reporting": s7_rep,
        "total_duration_sec": round(total_sec, 4),
    }

def main(run_id: Optional[str] = None) -> Dict[str, Any]:
    return run_full_pipeline(run_id=run_id)

if __name__ == "__main__":
    main()
