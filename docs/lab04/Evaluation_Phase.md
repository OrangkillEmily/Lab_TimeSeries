# Giai đoạn Đánh giá 5 Chỉ số & Giải đáp Câu hỏi Phân tích (`Evaluation_Phase.md`)

## 1. Purpose — Mục đích

**Giai đoạn Đánh giá (`Evaluation_Phase`)** thực hiện **Mục 10 (Đánh giá mô hình)**, **Mục 11 (Các câu hỏi phân tích chuyên sâu)** và **Mục 12 (Đánh giá Traffic Forecasting trên PEMS-BAY)** trong đề bài `practice 4.pdf`, kết hợp **Yêu cầu bổ sung số 4 (`5-Fold Cross-Validation`)** và **Yêu cầu bổ sung số 6 (Mở rộng thành bộ 5 chỉ số: `MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²`)**.

Mục đích cốt lõi của giai đoạn này là:
1. Định lượng toàn diện và công bằng hiệu năng của cả 3 mô hình dự báo phụ tải điện (`VAR`, `XGBoost`, `Residual LSTM`) và bộ 3 mô hình giao thông nâng cao (`DCRNN`, `Multi-Horizon TCN`, `Multi-Horizon LSTM`) trên cùng hệ quy chiếu **5 chỉ số hồi quy chuẩn**.
2. Đối sánh kết quả trên cả 4 chế độ thực nghiệm: **Hold-out Test (`15% — 5,260 giờ`)**, **Temporal 5-Fold CV (`Mean ± Std`)**, **Walk-Forward Backtesting (`168 giờ`)** và **Weather Feature Ablation (`With vs. Without Weather`)**.
3. Kết xuất tự động kho **23 biểu đồ học thuật** vào `reports/lab04/figures/` và trả lời định lượng cho các câu hỏi thảo luận ở Mục 11 & Mục 12.
4. Cung cấp dữ liệu đầu vào chuẩn xác tuyệt đối để `src/lab04/report.py` và `script/lab04/render_report.py` tự động xuất đồng bộ bộ 3 báo cáo `LAB_04_REPORT.pdf`, `LAB_04_REPORT.html` và `LAB_04_REPORT.md`.

## 2. Input-Output

### Input

| Đầu vào (Input) | Định dạng & Kích thước | Mô tả Chi tiết |
|---|---|---|
| Dự báo trên tập `Hold-out Test` & `Train` | `5,260 giờ` (Test) & `24,521 giờ` (Train) | Chuỗi giá trị thực tế $y_t$ và dự báo $\hat{y}_t$ (đã `inverse_transform` về đơn vị `kW`) của `VAR`, `XGBoost`, `Residual LSTM` |
| Dự báo `5-Fold CV` & `Walk-Forward 168h` | `5 Folds × 5,844 giờ` & `168 giờ` | Kết quả dự báo kiểm định chéo 5 Fold mở rộng và kiểm thử cuốn chiếu 168 bước |
| Dự báo `Weather Ablation` | `With Weather` vs. `Without Weather` | Kết quả đối sánh mô hình Đa biến (có 3 biến thời tiết) và Đơn biến (chỉ dùng chuỗi `load`) |
| Dự báo `PEMS-BAY` (`15m, 30m, 60m`) | `1,285 cửa sổ × 207 trạm` | Kết quả dự báo tốc độ xe (`mph`) của `DCRNN`, `Multi-Horizon TCN` và `Multi-Horizon LSTM` |

### Output

| Đầu ra (Output) | Đường dẫn Lưu trữ | Mô tả Chi tiết |
|---|---|---|
| `holdout_test_metrics.csv` | `reports/lab04/metrics/holdout_test_metrics.csv` | Bảng đối sánh 5 chỉ số (`MSE`, `RMSE`, `MAE`, `MAPE`, `R2`) trên cả tập `Test (15%)`, `Validation (15%)` và `Train (70%)` |
| `cv_5fold_summary.csv` & `cv_5fold_detailed.csv` | `reports/lab04/metrics/` | Bảng tổng hợp `Mean ± Std` và bảng chi tiết từng Fold (`Fold 1..5`) của 5 chỉ số đánh giá |
| `walk_forward_metrics.csv` & `weather_ablation_metrics.csv` | `reports/lab04/metrics/` | Bảng kết quả Walk-Forward Backtesting `168h` và nghiên cứu loại bỏ đặc trưng thời tiết (`Feature Ablation`) |
| `pems_bay_dcrnn_metrics.csv` & `pems_bay_benchmark_comparison.csv` | `reports/lab04/metrics/` | Bảng chi tiết `DCRNN` và bảng đối sánh toàn diện `DCRNN vs. Multi-Horizon TCN vs. Multi-Horizon LSTM` |
| Kho **23 biểu đồ học thuật** | `reports/lab04/figures/` | Bao gồm `Forecast_Overlay_168h.png`, `actual_vs_predicted_scatter.png`, `Residual_Diagnostics.png`, `Metric_Comparison_Bars.png`, `Walk_Forward_Trajectory.png`, `XGBoost_Feature_Importance.png`, `Feature_Ablation_Impact.png`, `pems_bay_model_comparison.png`, `pems_bay_dcrnn_evaluation.png`, v.v. |
| Bộ 3 Báo cáo Tự động (`HTML`, `PDF`, `MD`) | `reports/lab04/LAB_04_REPORT.{html,pdf,md}` | Báo cáo giao diện chuẩn `Lab04_ML` nhúng trực tiếp 23 biểu đồ Base64 và bảng nghiệm thu `AC01–AC15` |

## 3. How to do — Quy trình thực hiện & Giải đáp Câu hỏi Phân tích (Mục 11 & Mục 12)

### Sơ đồ luồng đánh giá tổng hợp (`Evaluation & Reporting Workflow`)

```text
  Dự đoán Đã Khôi phục Đơn vị Gốc (VAR, XGBoost, Residual LSTM, DCRNN, TCN, PEMS-LSTM)
                                            |
        +-----------------------------------+-----------------------------------+
        |                                   |                                   |
        v                                   v                                   v
 [1. Hold-out Test (5,260h)]         [2. 5-Fold CV & Walk-Forward]       [3. Ablation & PEMS-BAY]
 - XGBoost: RMSE=6,482.45 (R2=.9946) - 5-Fold CV: VAR ổn định nhất       - Ablation: LSTM giảm 5.41% RMSE
 - LSTM   : RMSE=6,570.41 (R2=.9944)   (RMSE Mean = 7,053.54 kW)         - PEMS-BAY (60m): DCRNN MAE=2.304
 - VAR    : RMSE=7,122.36 (R2=.9935) - Walk-Forward 168h: XGB RMSE=4,871   < LSTM (2.499) < TCN (2.624 mph)
        |                                   |                                   |
        +-----------------------------------+-----------------------------------+
                                            |
                                            v
  Xuất 8 file CSV Metrics + Kết xuất 23 biểu đồ tại reports/lab04/figures/
  Trích xuất tự động section_11_answers -> Render LAB_04_REPORT.{html, pdf, md}
```

### Khung Giải đáp Chuyên sâu Các Câu hỏi Phân tích (Đồng bộ với `eval_summary["section_11_answers"]`)

1. **Câu hỏi 1 (`Q1_Dominant_Weather_Variable` — Biến thời tiết ảnh hưởng mạnh nhất & Tác động Ablation):**
   - **Phân tích tương quan:** Trên bộ dữ liệu thực nghiệm `electricity_load.csv`, biến thời tiết có hệ số tương quan tuyến tính và tương quan hạng mạnh nhất với phụ tải điện (`load`) là **`humidity` (độ ẩm tương đối `%`)** với Pearson $r = -0.9224$ ($|r| = 0.9224$, Spearman $\rho = -0.9208$), theo sát là **`temperature` (nhiệt độ `°C`)** với $r = +0.8853$ ($\rho = +0.8717$). Hai biến này có quan hệ nghịch biến chặt chẽ ($r = -0.9657$), phản ánh đặc trưng những giờ nắng nóng đỉnh điểm có độ ẩm giảm thấp khiến nhu cầu làm mát tăng vọt.
   - **Phân tích Feature Ablation (`weather_ablation_metrics.csv`):** Khi đưa bộ 3 biến thời tiết vào mô hình, mạng tuần tự **`Residual LSTM` cải thiện mạnh mẽ nhất**, giảm `RMSE` từ `6,946.54 kW` xuống `6,570.41 kW` (**giảm `5.41%` RMSE** và **giảm `7.16%` MAPE**). Trong khi đó, `VAR` (`-0.02%`) và `XGBoost` (`-1.62%`) gần như giữ nguyên hiệu năng do chuỗi `load` tự hồi quy 24 giờ cùng các biến lịch chu kỳ (`cos_hour`, `sin_hour`) đã giải thích tới $>99.3\%$ phương sai. Tính trung bình trên cả 3 kiến trúc, việc tích hợp thời tiết giúp giảm trung bình `1.26%` sai số `RMSE`.

2. **Câu hỏi 2 (`Q2_Best_Model_And_Why` — Mô hình tốt nhất & Đối sánh Hold-out vs. 5-Fold CV):**
   - **Trên tập Hold-out Test (`15% — 5,260 giờ`) và Walk-Forward (`168 giờ`):** Mô hình **`XGBoost`** đạt độ chính xác cao nhất với `RMSE = 6,482.45 kW`, `MAE = 4,337.53 kW`, `MAPE = 1.89%` và $R^2 = 0.9946$ (Walk-Forward `RMSE = 4,870.64 kW`, $R^2 = 0.9950$). Nhờ `105` đặc trưng kết hợp biến trễ, thống kê trượt và pha lượng giác giờ, `XGBoost` nắm bắt chính xác các ngưỡng chuyển pha phi tuyến. Mạng **`Residual LSTM`** bám rất sát ở vị trí thứ hai (`RMSE = 6,570.41 kW`, $R^2 = 0.9944$).
   - **Trên kiểm định chéo 5-Fold (`cv_5fold_summary.csv`):** Khi kích thước tập huấn luyện ở các Fold đầu còn nhỏ (`Fold 1 = 5,845 giờ`, `Fold 2 = 11,689 giờ`), mô hình thống kê **`VAR(24)`** trên miền sai phân bậc 1 lại đạt độ ổn định cao nhất (`RMSE Mean = 7,053.54 ± 970.44 kW`, $R^2 = 0.9925$), trong khi `XGBoost` và `LSTM` chỉ vượt lên dẫn đầu từ `Fold 4` và `Fold 5` khi dữ liệu huấn luyện đạt trên `23,000 giờ`.

3. **Câu hỏi 3 (`Q3_Model_Limitations_And_Challenges` — Hạn chế của VAR, XGBoost, LSTM & Kiểm tra Overfitting):**
   - **Hạn chế của `VAR`:** Chỉ biểu diễn được quan hệ tuyến tính giữa các biến sai phân và số lượng tham số tăng bậc hai $\mathcal{O}(p \cdot K^2)$ theo số chuỗi $K$, không bắt được các điểm uốn phi tuyến bậc hai của thời tiết.
   - **Hạn chế của `XGBoost`:** Phụ thuộc chặt vào việc thiết kế thủ công `105` đặc trưng trễ/cuốn và cây quyết định không thể ngoại suy vượt ngoài miền giá trị cực đại/cực tiểu của tập huấn luyện (dẫn tới sai số cao ở Fold 1–2 khi mẫu huấn luyện ít).
   - **Hạn chế & Kiểm soát Overfitting của `LSTM`:** Đòi hỏi chuẩn hóa Z-Score nghiêm ngặt, nhánh kết nối tắt (`Autoregressive Residual Highway`) và lịch trình `CosineAnnealingLR`. Đối chiếu trên `holdout_test_metrics.csv`, `Residual LSTM` có `Train RMSE = 6,104.39 kW` ($R^2 = 0.9948$) và `Test RMSE = 6,570.41 kW` ($R^2 = 0.9944$), độ chênh lệch cực nhỏ chứng minh cơ chế `EarlyStopping(best_epoch = 27)` đã kiểm soát hoàn toàn hiện tượng Overfitting.

4. **Câu hỏi 4 (`Q4_PEMS_BAY_SpatioTemporal_Benchmark` — Đối sánh DCRNN vs. Multi-Horizon TCN vs. Multi-Horizon LSTM ở Mục 12):**
   - Trên bộ dữ liệu giao thông `PEMS-BAY` (`207` trạm cảm biến $\times$ `8,640` bước 5 phút), mô hình **`DCRNN`** (tích hợp ma trận đồ thị không gian `adj_mx_bay.pkl` qua phép tích chập khuếch tán hai chiều) vượt trội hoàn toàn so với cả **`Multi-Horizon TCN`** (Causal Conv1D — Slide 10 Chương 4–5) và **`Multi-Horizon LSTM`** (thuần thời gian) ở tất cả các tầm nhìn:
     - **15 phút ($t+3$):** `DCRNN` đạt `MAE = 1.3094 mph` (`RMSE = 2.7621 mph`, $R^2 = 0.9196$) $<$ `TCN` (`MAE = 1.3543 mph`) $<$ `LSTM` (`MAE = 1.4182 mph`).
     - **30 phút ($t+6$):** `DCRNN` đạt `MAE = 1.7338 mph` (`RMSE = 3.9311 mph`, $R^2 = 0.8372$) $<$ `TCN` (`MAE = 1.8553 mph`) $<$ `LSTM` (`MAE = 1.8885 mph`).
     - **60 phút ($t+12$):** `DCRNN` đạt `MAE = 2.3040 mph` (`RMSE = 5.2817 mph`, $R^2 = 0.7063$), thấp hơn rõ rệt so với `Multi-Horizon LSTM` (`MAE = 2.4986 mph`, $R^2 = 0.6987$) và `Multi-Horizon TCN` (`MAE = 2.6243 mph`, $R^2 = 0.6087$), chứng minh giá trị cốt lõi của cấu trúc mạng lưới không gian khi dự báo sóng lan truyền tắc nghẽn giao thông ở tầm nhìn xa.

## 4. Evaluation metrics

Với $y_t$ là giá trị thực tế và $\hat{y}_t$ là giá trị dự báo (trên thang đo gốc `kW` hoặc `mph`), 5 chỉ số được tính nhất quán trên mọi bảng kết quả:

| STT | Chỉ số (Metric) |
|---:|---|
| 1 | **MSE (Mean Squared Error $\downarrow$)** |
| 2 | **RMSE (Root Mean Squared Error $\downarrow$)** |
| 3 | **MAE (Mean Absolute Error $\downarrow$)**| 
| 4 | **MAPE (Mean Absolute Percentage Error $\downarrow$)** |
| 5 | **$R^2$ Score (Coefficient of Determination $\uparrow$)** |

## 5. Used data — Dữ liệu sử dụng

| Tập dữ liệu & Bảng Chỉ số | Đường dẫn File Thực tế | Vai trò trong `Evaluation_Phase` | Module Thực thi |
|---|---|---|---|
| **Hold-out Test & Ablation** | `reports/lab04/metrics/holdout_test_metrics.csv`, `weather_ablation_metrics.csv` | Đánh giá chính thức 5 chỉ số của `VAR`, `XGBoost`, `Residual LSTM` và tác động biến thời tiết | `src/lab04/evaluate.py`, `src/lab04/visualize.py`, `src/lab04/report.py` |
| **5-Fold CV & Walk-Forward** | `reports/lab04/metrics/cv_5fold_summary.csv`, `cv_5fold_detailed.csv`, `walk_forward_metrics.csv` | Đánh giá độ ổn định qua 5 Fold thời gian và kiểm thử cuốn chiếu 168 giờ | `src/lab04/evaluate.py`, `src/lab04/visualize.py` |
| **PEMS-BAY Benchmark** | `reports/lab04/metrics/pems_bay_dcrnn_metrics.csv`, `pems_bay_benchmark_comparison.csv` | Đối sánh 5 chỉ số của `DCRNN`, `Multi-Horizon TCN` và `Multi-Horizon LSTM` ở 15m, 30m, 60m | `src/lab04/evaluate.py`, `src/lab04/report.py` |
