# 01 — Giai đoạn Khám phá Dữ liệu & Chia Tập Tuần tự (`EDA_and_Split_Phase`)

## 1. Purpose
Giai đoạn này nạp, kiểm toán chất lượng và phân tách dữ liệu theo đúng trật tự thời gian (`shuffle=False`) cho đồng thời **2 tập dữ liệu chuẩn của Lab 02**:
- **Dataset 1:** `Monthly Retail Sales` (Doanh số bán lẻ hàng tháng tại Mỹ, `07/1991 – 06/2008`, `204` tháng).
- **Dataset 2:** `AirPassengers` (Lượng hành khách hàng không quốc tế, `01/1949 – 12/1960`, `144` tháng).

Mục tiêu cốt lõi là đảm bảo **tính toàn vẹn chỉ mục thời gian (`MS`)**, đo lường độ biến thiên/xu hướng trước khi mô hình hóa và **ngăn chặn tuyệt đối Data Leakage** bằng phép chia 80% Train / 20% Test tuần tự.

## 2. Input-Output
| Thành phần | Đặc tả chi tiết |
|---|---|
| **Input** | `data/raw/Monthly Retail Sales.csv` (`date`, `value`), `data/raw/AirPassengers.csv` (`Month`, `Passengers`), nhánh cấu hình `lab02` trong `config.yaml` (`test_ratio = 0.20`, `shuffle = false`). |
| **Module xử lý** | `src/common/data_loader.py` (`load_time_series_csv`, `split_chronological`), `src/lab02/data_loader.py` (`load_dataset`, `split_dataset`), `src/lab02/visualize.py`. |
| **Output (Data/JSON)** | `data/processed/lab02/{dataset1,dataset2}/{train_data.csv, test_data.csv, split_summary.json}`, nhánh `eda` trong `progress_results/lab02/{dataset1,dataset2}/latest_run.json`. |
| **Output (Figures)** | Biểu đồ tổng quan chuỗi gốc và chia Train/Test (80/20) lưu tại `reports/figures/lab02/{dataset1,dataset2}/report/` và `reports/lab02/figures/{dataset1,dataset2}/`. |

## 3. How to do
1. **Nạp & Kiểm toán Schema (`load_dataset(dataset_key, cfg)`):** Gọi lõi dùng chung `load_time_series_csv()` với `freq = "MS"`, kiểm tra `missing_values == 0`, `duplicate_timestamps == 0`, `is_monotonic == True`.
2. **Thống kê mô tả toàn chuỗi:** Tính `mean`, `std`, `CV (%)`, `skew`, `min/max` kèm mốc `MM/YYYY`, phân vị `Q25 / Median / Q75` và hệ số tăng trưởng `growth_ratio`.
3. **Chia tập Train/Test tuần tự (`split_dataset(df, dataset_key, cfg)`):**
   - Áp dụng `test_size = math.ceil(n_total * 0.20)`, `train_size = n_total - test_size`, giữ nguyên thứ tự thời gian (`train.index.max() < test.index.min()`).
   - Lưu vật lý `train_data.csv`, `test_data.csv` và `split_summary.json` vào `data/processed/lab02/<dataset>/`.

## 4. Evaluation Metrics
| Tiêu chí thống kê & Chia tập | Dataset 1 (`Monthly Retail Sales`) | Dataset 2 (`AirPassengers`) |
|---|---|---|
| **Giai đoạn & Số quan sát ($N$)** | `1991-07-01` → `2008-06-01` (`204` tháng) | `1949-01-01` → `1960-12-01` (`144` tháng) |
| **Missing / Duplicate** | `0` / `0` (`float64`) | `0` / `0` (`int64`) |
| **Mean ± Std (CV %)** | `10.6944 ± 5.9570` (`CV = 55.7%`) | `280.2986 ± 119.9663` (`CV = 42.8%`) |
| **Min / Median / Max** | `2.8145 (02/1992)` / `9.3193` / `29.6654 (01/2008)` | `104.0 (11/1949)` / `265.5` / `622.0 (07/1960)` |
| **Tỷ lệ tăng trưởng (`growth_ratio`)** | `5.51x` | `3.86x` |
| **Tập Train (80%)** | `163` tháng (`1991-07` → `2005-01`, `Mean = 8.48 ± 3.94`) | `115` tháng (`1949-01` → `1958-07`, `Mean = 239.95 ± 91.35`) |
| **Tập Test (20%)** | `41` tháng (`2005-02` → `2008-06`, `Mean = 19.51 ± 4.18`) | `29` tháng (`1958-08` → `1960-12`, `Mean = 440.31 ± 79.51`) |

## 5. Used data
- **Dataset 1:** `data/raw/Monthly Retail Sales.csv` (`204` quan sát, tần suất `MS`).
- **Dataset 2:** `data/raw/AirPassengers.csv` (`144` quan sát, tần suất `MS`).
- **Lưu ý phương pháp luận:** Cả hai tập đều có `Test Mean` cao hơn rõ rệt `Train Mean` (xu hướng tăng trưởng mạnh) và biên độ dao động giãn nở theo thời gian — dấu hiệu trực tiếp cần kiểm định tính dừng (ADF) và cân nhắc biến đổi Logarit.
