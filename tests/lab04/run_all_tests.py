r"""
Trình Chạy Kiểm thử Tự động Tổng hợp (Master Test Runner) cho Bài thực hành Lab 04
(tests/lab04/run_all_tests.py).

Cách sử dụng từ thư mục gốc dự án (D:\Lab_TimeSeries):
    python tests/lab04/run_all_tests.py
hoặc dùng pytest:
    pytest tests/lab04 -v
"""

import sys
import time
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_suite() -> int:
    r"""
    Purpose: Tự động khám phá và thực thi toàn bộ các kịch bản kiểm thử trong thư mục tests/lab04/.
    Input-Output:
        - Input: Không có.
        - Output: Mã thoát hệ thống (0 nếu 100% PASS, 1 nếu có bất kỳ lỗi nào).
    How: Sử dụng unittest.TestLoader().discover() và in bảng tổng kết trực quan.
    """
    t0 = time.perf_counter()
    test_dir = Path(__file__).resolve().parent

    print("=" * 88)
    print("BẮT ĐẦU KIỂM THỬ TỰ ĐỘNG LAB 04 (UNIT & INTEGRATION TEST SUITE)")
    print(f"Thư mục kiểm thử: {test_dir}")
    print("=" * 88)

    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(test_dir), pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    elapsed = time.perf_counter() - t0
    total_run = result.testsRun
    n_fail = len(result.failures)
    n_err = len(result.errors)
    n_pass = total_run - n_fail - n_err

    print("\n" + "=" * 88)
    print(
        f"TỔNG KẾT KIỂM THỬ LAB 04 | Thời gian: {elapsed:.2f}s | "
        f"Tổng số Test: {total_run} | PASS: {n_pass} | FAIL: {n_fail} | ERROR: {n_err}"
    )
    print("=" * 88)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_suite())