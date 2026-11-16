"""
Điểm khởi chạy chính (`script/lab04/run_lab04.py`) cho Bài thực hành Lab 04: Dự báo Chuỗi Thời gian Đa biến.

Cách sử dụng (từ thư mục gốc của kho lưu trữ đa Lab):
    1. Chạy toàn trình (Yêu cầu 1 -> 12 + 5-Fold TimeSeriesSplit + Walk-Forward + Auto Report PDF/HTML/MD):
       python script/lab04/run_lab04.py
    2. Chạy kèm đồng bộ bộ biểu đồ sang `docs/lab04/images/` khi cần tách riêng thư mục `docs/lab04/`:
       python script/lab04/run_lab04.py --bundle-docs
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT_DIR = HERE.parent.parent.parent
if not (ROOT_DIR / "src").exists() and (HERE.parent / "src").exists():
    ROOT_DIR = HERE.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.lab04.pipeline import main
from src.lab04.utils import load_config


def bundle_standalone_docs() -> None:
    """
    What: Sao chép toàn bộ biểu đồ `.png` từ `reports/lab04/figures/` sang `docs/lab04/images/`.
    Why: Giúp thư mục `docs/lab04/` có thể đóng gói độc lập khi nộp lẻ tài liệu.
    Input: Không có.
    Output: Không có.
    How: Duyệt `*.png` trong `reports/lab04/figures/` và gọi `shutil.copy2`.
    """
    cfg = load_config()
    fig_dir = ROOT_DIR / cfg["paths"]["figures_dir"]
    docs_img_dir = ROOT_DIR / cfg["paths"]["docs_dir"] / "images"
    docs_img_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for png in fig_dir.glob("*.png"):
        shutil.copy2(png, docs_img_dir / png.name)
        count += 1
    print(f"[BUNDLE-DOCS] Đã đồng bộ {count} biểu đồ vào {docs_img_dir.relative_to(ROOT_DIR)}.")


if __name__ == "__main__":
    main()
    if "--bundle-docs" in sys.argv:
        bundle_standalone_docs()