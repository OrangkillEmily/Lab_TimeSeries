r"""Trình chạy toàn bộ bộ kiểm thử đơn vị cho Bài thực hành Lab 01."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def run_all() -> int:
    r"""
    Purpose: Khám phá và chạy toàn bộ tệp test_*.py trong tests/lab01/.
    Input-Output:
        - Input: Không có.
        - Output: Mã thoát 0 nếu pass 100%, ngược lại 1.
    How: Dùng unittest.defaultTestLoader.discover() trên thư mục tests/lab01/.
    """
    loader = unittest.defaultTestLoader
    suite = loader.discover(start_dir=str(Path(__file__).resolve().parent), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_all())