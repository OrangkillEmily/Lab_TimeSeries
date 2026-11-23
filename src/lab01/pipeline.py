r"""
Mô-đun Điều phối Đường ống Thực thi Toàn trình (End-to-End Pipeline) cho Bài thực hành Lab 01.
Chạy tuần tự 6 giai đoạn khớp 100% 6 bài tập thực hành chương 2.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from src.lab01 import data_loader, decomposition, eda, quality, report, visualize
from src.lab01.tracking import init_run_session, write_log
from src.lab01.utils import ensure_directories, load_config, set_global_seed 


def run_full_pipeline(run_id: Optional[str] = None) -> Dict[str, Any]:
    r"""
    Purpose: Thực thi toàn trình 6 giai đoạn của Lab 01 (Data -> ACF/White Noise -> Decomposition -> Quality -> Viz -> Report).
    Input-Output:
        - Input: run_id (Optional[str]).
        - Output: Từ điển tổng hợp kết quả toàn bộ 6 giai đoạn và tổng thời gian chạy.
    How: Cố định seed, khởi tạo phiên chạy, gọi lần lượt main(active_run_id) của từng mô-đun và ghi nhật ký.
    """
    t0 = time.perf_counter()
    cfg = load_config()
    ensure_directories(cfg)
    set_global_seed(int(cfg["project"]["random_seed"]))

    active_run_id = init_run_session(run_id)
    write_log(active_run_id, "PIPELINE", "START", "Khởi chạy toàn trình Pipeline Lab 01 (AirPassengers.csv).")

    s1_data = data_loader.main(active_run_id)
    s2_eda = eda.main(active_run_id)
    s3_dec = decomposition.main(active_run_id)
    s4_qual = quality.main(active_run_id)
    s5_vis = visualize.main(active_run_id)
    s6_rep = report.main(active_run_id)

    total_sec = time.perf_counter() - t0
    write_log(active_run_id, "PIPELINE", "COMPLETED", f"Hoàn tất toàn bộ 6 giai đoạn Lab 01 trong {total_sec:.2f}s.")

    return {
        "run_id": active_run_id,
        "data_loading": s1_data,
        "eda": s2_eda,
        "decomposition": s3_dec,
        "data_quality": s4_qual,
        "visualization": s5_vis,
        "reporting": s6_rep,
        "total_duration_sec": round(total_sec, 4),
    }


def main(run_id: Optional[str] = None) -> Dict[str, Any]:
    r"""
    Purpose: Điểm vào chuẩn gọi run_full_pipeline() cho Lab 01.
    Input-Output:
        - Input: run_id (Optional[str]).
        - Output: Từ điển kết quả pipeline Lab 01.
    How: Ủy quyền trực tiếp sang run_full_pipeline(run_id=run_id).
    """
    return run_full_pipeline(run_id=run_id)