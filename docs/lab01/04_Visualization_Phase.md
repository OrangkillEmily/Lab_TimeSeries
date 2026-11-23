# 04 — Giai đoạn Trực quan hóa Chuỗi Thời gian (`Visualization_Phase`)

## 1. Purpose
Giai đoạn này (Bài tập 5) chuyển hóa toàn bộ đặc trưng thống kê của `AirPassengers` thành hệ thống biểu đồ phân tích đa góc nhìn, giúp nhận diện trực quan:
- Tính không dừng cả về **trung bình** lẫn **phương sai** qua Rolling Mean & Rolling Std (`window = 12`).
- Cấu trúc lặp lại ổn định của chu kỳ **12 tháng** qua Seasonal Plot và Chỉ số Mùa vụ (Seasonal Index).
- Sự giãn nở biên độ theo quy mô qua **cặp Heatmap** (thang gốc vs thang chuẩn hóa theo trung bình năm).

## 2. Input-Output
| Thành phần | Đặc tả chi tiết |
|---|---|
| **Input** | Chuỗi `AirPassengers` (`144` tháng) từ `load_airpassengers_dataset()`, tham số `lab01.analysis.rolling_window = 12` trong `config.yaml`. |
| **Module xử lý** | `src/lab01/visualize.py` (`plot_rolling_statistics`, `plot_seasonal_trends`, `plot_heatmaps`, `generate_visualization_suite`). |
| **Output (Data/JSON)** | `reports/lab01/metrics/visualization_summary.json`, nhánh `stages.visualization` trong `progress_results/lab01/latest_run.json`. |
| **Output (Figures)** | `13_rolling_statistics.png`, `14_seasonal_trends.png`, `15_heatmaps_seasonality.png` (hoàn thiện trọn bộ **15 biểu đồ PNG** tại `reports/lab01/figures/`). |

## 3. How to do
1. **Rolling Statistics (`plot_rolling_statistics`, `window = 12`):** Tính trung bình trượt $\bar{y}_t^{(12)}$ và độ lệch chuẩn trượt $s_t^{(12)}$, vẽ biểu đồ 2 tầng kèm hệ số tương quan Pearson giữa Rolling Mean và Rolling Std.
2. **Seasonal Plot & Chỉ số Mùa vụ (`plot_seasonal_trends`):**
   - Tạo bảng xoay `table` kích thước `12 tháng × 12 năm` (`1949–1960`).
   - Tính tỷ số mùa vụ chuẩn hóa theo năm: $SI_{m,y} = \frac{Y_{m,y}}{\bar{Y}_{\cdot,y}}$ và vẽ dải biến thiên `min–max` qua 12 năm kèm đường trung bình.
3. **Cặp Heatmap Song song (`plot_heatmaps`):**
   - Heatmap trái (`YlOrRd`): Giá trị tuyệt đối theo tháng × năm (thể hiện rõ cả trend tăng dọc theo cột năm lẫn đỉnh hè).
   - Heatmap phải (`RdBu_r`, phân kỳ quanh `1.0`): Chỉ số mùa vụ $SI_{m,y}$ (làm nổi bật cấu trúc mùa vụ bất biến qua mọi năm).
4. **Tính tốc độ tăng trưởng kép hàng năm (CAGR):**
   $$\text{CAGR} = \left(\frac{\bar{Y}_{1960}}{\bar{Y}_{1949}}\right)^{\frac{1}{11}} - 1$$

## 4. Evaluation Metrics
| Chỉ tiêu trực quan hóa | Ý nghĩa thống kê | Kết quả thực nghiệm (`run_id: 20261123_154119_4f8c41`) |
|---|---|---|
| **Tương quan Rolling Mean – Std** | Kiểm chứng phương sai tăng theo quy mô | `r = 0.9591` (khẳng định chuỗi không dừng cả trung bình lẫn phương sai → cần biến đổi Log) |
| **Tốc độ tăng trưởng kép (CAGR)** | Tăng trưởng bình quân năm (1949 → 1960) | `12.79% / năm` |
| **Tháng cao điểm (Peak)** | Đỉnh mùa vụ du lịch hè | **Tháng 7** (`Trung bình = 351.33 nghìn`, `Seasonal Index = 1.2364`, cao hơn TB năm ~23.6%) |
| **Tháng thấp điểm (Trough)** | Đáy mùa vụ cuối thu – đầu đông | **Tháng 11** (`Trung bình = 232.83 nghìn`, `Seasonal Index = 0.8320`, thấp hơn TB năm ~16.8%) |
| **Tổng số biểu đồ Lab 01** | Đầy đủ từ `01_` đến `15_` | `15 / 15` tệp PNG tại `reports/lab01/figures/` |

## 5. Used data
- **Dữ liệu nguồn:** `data/raw/AirPassengers.csv` (`144` quan sát, `12` năm × `12` tháng).
- **Đồng bộ tài liệu độc lập:** Khi chạy `python script/lab01/run_lab01.py --bundle-docs`, toàn bộ 15 biểu đồ từ `reports/lab01/figures/` được tự động sao chép sang `docs/lab01/images/`.
