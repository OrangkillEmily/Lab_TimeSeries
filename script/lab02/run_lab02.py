r"""
Điểm khởi chạy chính (`script/lab02/run_lab02.py`) cho Bài thực hành Lab 02:
Kiểm định ADF, Phân tích ACF/PACF và Dự báo AR / MA / ARIMA / SARIMA / Log-SARIMA.

Cách sử dụng (từ thư mục gốc repo):
    1. Chạy toàn trình cả 2 dataset (Dataset 1: Monthly Retail Sales & Dataset 2: AirPassengers):
       python script/lab02/run_lab02.py
    2. Chạy riêng 1 dataset hoặc đổi chế độ ứng viên (standard / full / auto):
       python script/lab02/run_lab02.py --dataset dataset1 --mode full
       python script/lab02/run_lab02.py --dataset dataset2 --mode standard
    3. Chạy kèm đồng bộ toàn bộ biểu đồ sang `docs/lab02/images/`:
       python script/lab02/run_lab02.py --bundle-docs
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT_DIR = HERE.parent.parent.parent
if not (ROOT_DIR / "src").exists() and (HERE.parent / "src").exists():
    ROOT_DIR = HERE.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.lab02.pipeline import main as run_pipeline
from src.lab02.utils import get_path, load_config


def bundle_standalone_docs() -> None:
    r"""
    Purpose: Sao chép toàn bộ biểu đồ `.png` từ `reports/lab02/figures/{dataset1,dataset2}/` sang `docs/lab02/images/`.
    Input-Output:
        - Input: Không có.
        - Output: Không trả về giá trị (in tổng số biểu đồ đã đồng bộ).
    How: Duyệt 2 thư mục con dataset1 và dataset2 trong figures_dir và gọi shutil.copy2().
    """
    cfg = load_config()
    fig_root = get_path("figures_dir", cfg)
    docs_img_root = get_path("docs_dir", cfg) / "images"
    count = 0
    for ds_key in ("dataset1", "dataset2"):
        src_dir = fig_root / ds_key
        dst_dir = docs_img_root / ds_key
        if not src_dir.exists():
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        for png in sorted(src_dir.rglob("*.png")):
            shutil.copy2(png, dst_dir / png.name)
            count += 1
    print(f"[BUNDLE-DOCS] Đã đồng bộ {count} biểu đồ Lab 02 vào {docs_img_root.relative_to(ROOT_DIR)}.")


def parse_args() -> argparse.Namespace:
    r"""
    Purpose: Phân tích các tham số dòng lệnh cho script/lab02/run_lab02.py.
    Input-Output:
        - Input: sys.argv.
        - Output: argparse.Namespace chứa dataset, mode, no_figures, verbose, bundle_docs.
    How: Khai báo các cờ tùy chọn bằng argparse.ArgumentParser.
    """
    parser = argparse.ArgumentParser(description="Lab 02 End-to-End Pipeline Runner (Dataset 1 & Dataset 2)")
    parser.add_argument("--dataset", type=str, default="all", choices=["dataset1", "dataset2", "all"])
    parser.add_argument("--mode", type=str, default="full", choices=["standard", "full", "auto"])
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--bundle-docs", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ds_list = ["dataset1", "dataset2"] if args.dataset == "all" else [args.dataset]
    run_pipeline(
        datasets=ds_list,
        mode=args.mode,
        save_figures=not args.no_figures,
        verbose=args.verbose,
    )
    if args.bundle_docs:
        bundle_standalone_docs()