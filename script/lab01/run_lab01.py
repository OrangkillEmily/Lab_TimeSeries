r"""
Điểm khởi chạy chính (`script/lab01/run_lab01.py`) cho Bài thực hành Lab 01:
Khám phá Chuỗi Thời gian, White Noise, Phân rã Chuỗi và Chất lượng Dữ liệu (AirPassengers.csv).

Cách sử dụng (từ thư mục gốc repo):
    1. Chạy toàn trình (Bài 1 -> 6 + Logging + progress_results + Báo cáo HTML/PDF/MD):
       python script/lab01/run_lab01.py
    2. Chạy kèm đồng bộ 15 biểu đồ sang `docs/lab01/images/`:
       python script/lab01/run_lab01.py --bundle-docs
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

from src.lab01.pipeline import main
from src.lab01.utils import get_path, load_config


def bundle_standalone_docs() -> None:
    r"""
    Purpose: Sao chép toàn bộ 15 biểu đồ `.png` từ `reports/lab01/figures/` sang `docs/lab01/images/`.
    Input-Output:
        - Input: Không có.
        - Output: Không trả về giá trị (in số lượng tệp đã đồng bộ ra terminal).
    How: Duyệt toàn bộ `*.png` trong `figures_dir` và gọi `shutil.copy2` sang `docs/lab01/images/`.
    """
    cfg = load_config()
    fig_dir = get_path("figures_dir", cfg)
    docs_img_dir = get_path("docs_dir", cfg) / "images"
    docs_img_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for png in sorted(fig_dir.glob("*.png")):
        shutil.copy2(png, docs_img_dir / png.name)
        count += 1
    print(f"[BUNDLE-DOCS] Đã đồng bộ {count} biểu đồ vào {docs_img_dir.relative_to(ROOT_DIR)}.")


if __name__ == "__main__":
    main()
    if "--bundle-docs" in sys.argv:
        bundle_standalone_docs()