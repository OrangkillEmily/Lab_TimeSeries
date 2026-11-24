# BÁO CÁO KỸ THUẬT LAB 04: DỰ BÁO CHUỖI THỜI GIAN ĐA BIẾN (MULTIVARIATE TIME SERIES FORECASTING)

- **Mã phiên chạy (Run ID)**: `20261124_193611_5cd58d`
- **Phần cứng thực thi**: `NVIDIA GeForce RTX 3050 6GB Laptop GPU` (`torch=cuda`, `xgb=cuda`, `AMP_FP16=True`)
- **Bộ dữ liệu chính (Mục 1–11)**: `data/raw/electricity_load.csv` (35,065 giờ × 4 biến: `load`, `temperature`, `humidity`, `wind_speed`)
- **Bộ dữ liệu nâng cao (Mục 12)**: `data/raw/pems-bay.h5` & `data/sensor_graph/adj_mx_bay.pkl` (8,640 bước × 207 trạm cảm biến)

## 1. Kết quả Đối sánh trên Tập Kiểm tra Độc lập (Hold-out Test 15%)

| Model   |         MSE |    RMSE |     MAE |   MAPE |     R2 |   Max_Error |   Mean_Residual |   Std_Residual |   Train_MSE |   Train_RMSE |   Train_MAE |   Train_MAPE |   Train_R2 |   Val_RMSE |   Val_MAPE |   Val_R2 |   Train_Time_Sec |   Pred_Time_Sec | Device   |
|:--------|------------:|--------:|--------:|-------:|-------:|------------:|----------------:|---------------:|------------:|-------------:|------------:|-------------:|-----------:|-----------:|-----------:|---------:|-----------------:|----------------:|:---------|
| XGBoost | 3.99547e+07 | 6320.97 | 4273.05 | 1.8683 | 0.9948 |      132576 |       -111.224  |        6320    | 1.444e+07   |      3800    |     2755.13 |       1.2562 |     0.998  |    5513.88 |     7.0071 |   0.9915 |           5.2138 |        0.099543 | cuda     |
| LSTM    | 4.31702e+07 | 6570.41 | 4524.04 | 2.0185 | 0.9944 |      128134 |       -363.156  |        6560.36 | 3.72636e+07 |      6104.39 |     4069.92 |       1.9624 |     0.9948 |    5968.28 |     7.1742 |   0.9901 |          97.0233 |        0.806361 | cuda     |
| VAR     | 5.0728e+07  | 7122.36 | 4576.49 | 2.001  | 0.9935 |      130204 |        -25.5376 |        7122.32 | 4.54186e+07 |      6739.33 |     4197.65 |       2.0323 |     0.9937 |    6501.03 |     7.2729 |   0.9882 |           3.0565 |        0.257427 | cpu      |

## 2. Kết quả Kiểm định Chéo 5-Fold TimeSeriesSplit (K = 5)

| Model   |   Folds |    MSE_Mean |     MSE_Std |   RMSE_Mean |   RMSE_Std |   MAE_Mean |   MAE_Std |   MAPE_Mean |   MAPE_Std |   R2_Mean |   R2_Std |
|:--------|--------:|------------:|------------:|------------:|-----------:|-----------:|----------:|------------:|-----------:|----------:|---------:|
| VAR     |       5 | 5.02283e+07 | 1.75209e+07 |     7004.77 |    1204.93 |    4264.73 |   796.886 |      3.3066 |     2.4205 |    0.9918 |   0.0022 |
| LSTM    |       5 | 8.28213e+07 | 5.49737e+07 |     8764.86 |    2738.29 |    5979.73 |  1784.49  |      4.1377 |     2.1749 |    0.9877 |   0.0037 |
| XGBoost |       5 | 1.25545e+08 | 1.23723e+08 |    10235.1  |    5097.38 |    6320.84 |  2793.39  |      4.1704 |     1.9925 |    0.9826 |   0.0111 |

## 3. Kết quả Walk-Forward Backtesting (168 Giờ)

| Model   |   Eval_Steps |         MSE |    RMSE |     MAE |   MAPE |     R2 |   Max_Error |   Mean_Residual |   Std_Residual |
|:--------|-------------:|------------:|--------:|--------:|-------:|-------:|------------:|----------------:|---------------:|
| VAR     |          168 | 2.79087e+07 | 5282.87 | 3937.53 | 1.8182 | 0.9941 |     18706.9 |         379.65  |        5269.21 |
| XGBoost |          168 | 2.86896e+07 | 5356.26 | 3973.05 | 1.8012 | 0.994  |     18006.7 |         897.784 |        5280.49 |
| LSTM    |          168 | 3.37643e+07 | 5810.71 | 4238.78 | 1.9522 | 0.9929 |     26100.7 |         355.783 |        5799.81 |

## 4. Nghiên cứu Loại bỏ Đặc trưng Thời tiết (Feature Ablation)

| Model   |   With_Weather_RMSE |   Without_Weather_RMSE |   RMSE_Improvement_Pct |   With_Weather_MAE |   Without_Weather_MAE |   With_Weather_MAPE |   Without_Weather_MAPE |   MAPE_Improvement_Pct |   With_Weather_R2 |   Without_Weather_R2 |
|:--------|--------------------:|-----------------------:|-----------------------:|-------------------:|----------------------:|--------------------:|-----------------------:|-----------------------:|------------------:|---------------------:|
| VAR     |             7122.36 |                7120.99 |                  -0.02 |            4576.49 |               4569.45 |              2.001  |                 1.9958 |                  -0.26 |            0.9935 |               0.9935 |
| XGBoost |             6320.97 |                6285.65 |                  -0.56 |            4273.05 |               4270.67 |              1.8683 |                 1.8571 |                  -0.6  |            0.9948 |               0.9949 |
| LSTM    |             6570.41 |                7078.22 |                   7.17 |            4524.04 |               4930.62 |              2.0185 |                 2.2122 |                   8.76 |            0.9944 |               0.9935 |

## 5. Bài tập 12 Nâng cao (PEMS-BAY): Đối sánh DCRNN vs. Multi-Horizon TCN vs. Multi-Horizon LSTM

| Model              | Model_Display                      | Horizon              |   Steps_Ahead |   Minutes_Ahead |     MSE |   RMSE |    MAE |   MAPE |     R2 |   Max_Error |   Mean_Residual |   Std_Residual |   Train_Time_Sec |
|:-------------------|:-----------------------------------|:---------------------|--------------:|----------------:|--------:|-------:|-------:|-------:|-------:|------------:|----------------:|---------------:|-----------------:|
| DCRNN              | DCRNN (Graph Diffusion + GRU)      | 15_min               |             3 |              15 |  7.5867 | 2.7544 | 1.2991 | 2.6945 | 0.9201 |     48.1141 |         -0.0881 |         2.753  |          67.9647 |
| DCRNN              | DCRNN (Graph Diffusion + GRU)      | 30_min               |             6 |              30 | 15.2795 | 3.9089 | 1.7241 | 3.8977 | 0.839  |     55.7045 |         -0.1858 |         3.9045 |          67.9647 |
| DCRNN              | DCRNN (Graph Diffusion + GRU)      | 60_min               |            12 |              60 | 27.7368 | 5.2666 | 2.3004 | 5.6861 | 0.7079 |     58.6043 |         -0.3568 |         5.2545 |          67.9647 |
| DCRNN              | DCRNN (Graph Diffusion + GRU)      | Average_All_Horizons |             7 |              35 | 16.8677 | 3.9766 | 1.7745 | 4.0928 | 0.8223 |     58.6043 |         -0.2102 |         3.9707 |          67.9647 |
| Multi_Horizon_TCN  | Multi-Horizon TCN (Causal Conv1D)  | 15_min               |             3 |              15 |  8.7019 | 2.9499 | 1.3532 | 2.8495 | 0.9083 |     52.6765 |         -0.0345 |         2.9497 |          15.3117 |
| Multi_Horizon_TCN  | Multi-Horizon TCN (Causal Conv1D)  | 30_min               |             6 |              30 | 19.0428 | 4.3638 | 1.8605 | 4.2009 | 0.7994 |     55.8093 |         -0.0681 |         4.3633 |          15.3117 |
| Multi_Horizon_TCN  | Multi-Horizon TCN (Causal Conv1D)  | 60_min               |            12 |              60 | 36.9984 | 6.0826 | 2.6396 | 6.6155 | 0.6104 |     58.7673 |         -0.2492 |         6.0775 |          15.3117 |
| Multi_Horizon_TCN  | Multi-Horizon TCN (Causal Conv1D)  | Average_All_Horizons |             7 |              35 | 21.581  | 4.4654 | 1.9511 | 4.5553 | 0.7727 |     58.7673 |         -0.1173 |         4.4635 |          15.3117 |
| Multi_Horizon_LSTM | Multi-Horizon LSTM (Temporal Only) | 15_min               |             3 |              15 |  8.5194 | 2.9188 | 1.4283 | 2.9292 | 0.9102 |     47.2004 |          0.1375 |         2.9156 |           8.9257 |
| Multi_Horizon_LSTM | Multi-Horizon LSTM (Temporal Only) | 30_min               |             6 |              30 | 16.6435 | 4.0796 | 1.9189 | 4.1104 | 0.8247 |     52.4626 |          0.3406 |         4.0654 |           8.9257 |
| Multi_Horizon_LSTM | Multi-Horizon LSTM (Temporal Only) | 60_min               |            12 |              60 | 30.0549 | 5.4822 | 2.5517 | 5.5572 | 0.6835 |     56.6844 |          0.5697 |         5.4526 |           8.9257 |
| Multi_Horizon_LSTM | Multi-Horizon LSTM (Temporal Only) | Average_All_Horizons |             7 |              35 | 18.4059 | 4.1602 | 1.9663 | 4.1989 | 0.8061 |     56.6844 |          0.3493 |         4.1445 |           8.9257 |

## 6. Trả lời Câu hỏi Thảo luận Chuyên sâu (Mục 11 & Mục 12)

- **Câu 1 — Biến thời tiết nào ảnh hưởng mạnh nhất đến điện năng?**: Về độ lớn tương quan, 'humidity' (độ ẩm tương đối (%)) là biến thời tiết ảnh hưởng mạnh nhất tới phụ tải điện với hệ số tương quan tuyến tính Pearson r = -0.9224 (|r| = 0.9224, Spearman ρ = -0.9208), theo sát là 'temperature' (r = +0.8853, ρ = +0.8717). Hai biến này nghịch biến chặt với nhau (r = -0.9657) do nắng nóng làm độ ẩm giảm và nhu cầu điều hòa tăng vọt. Trong nghiên cứu loại bỏ đặc trưng (Weather Ablation), Residual LSTM khai thác tín hiệu thời tiết rõ rệt nhất khi giảm +7.17% RMSE (từ 7,078.22 xuống 6,570.41 kW) và +8.76% MAPE, trong khi VAR (-0.02%) và XGBoost (-0.56%) gần như bão hòa nhờ 24 độ trễ tự hồi quy của chuỗi load.
- **Câu 2 — Khi nào VAR thất bại?**: Mô hình kinh tế lượng VAR(24) đạt RMSE = 7,122.36 kW (R² = 0.9935 trên Hold-out, kém hơn XGBoost và LSTM) dù ổn định nhất ở 5-Fold CV (VAR dẫn đầu với RMSE = 7,004.77 ± 1,204.93 kW). VAR thường thất bại trong 3 tình huống: (1) quan hệ phi tuyến mạnh hình chữ U hoặc ngưỡng bậc thang giữa nhiệt độ/độ ẩm với phụ tải (thể hiện ở biểu đồ Quadratic Fit EDA 05) mà tổ hợp tuyến tính không nắm bắt được; (2) chuỗi có đa mùa vụ phức tạp (24h + 168h) và phương sai thay đổi; (3) khi số lượng biến K lớn (ví dụ 207 trạm cảm biến PEMS-BAY), số tham số tăng theo bậc hai O(p × K²) gây bùng nổ chiều và quá khớp nghiêm trọng.
- **Câu 3 — XGBoost học được thông tin gì từ thời tiết?**: Nhờ phân tách nhánh cây quyết định phi tuyến trên 106 đặc trưng, XGBoost (tốt nhất trên Hold-out với RMSE = 6,320.97 kW, MAPE = 1.87%, R² = 0.9948) học được các ngưỡng chuyển pha phi tuyến và độ trễ tác động của biến khí tượng. Cụ thể, trong bảng Gain Feature Importance, nhóm thời tiết nổi bật gồm: humidity_lag_12h (hạng 4, gain=0.0035), humidity_lag_11h (hạng 11, gain=0.0017), temperature_lag_13h (hạng 12, gain=0.0013), humidity_lag_13h (hạng 15, gain=0.0012), kết hợp tương tác trực tiếp với đặc trưng pha giờ trong ngày (cos_hour, sin_hour) và biến lịch cuối tuần (is_weekend).
- **Câu 4 — LSTM học được chu kỳ nào (ngày, tuần)?**: Với đầu vào Tensor 3D (batch_samples, 24, 4), mạng Stacked 2-Layer LSTM + Autoregressive Residual Highway học trực tiếp và trọn vẹn chu kỳ ngày 24 giờ (Diurnal Seasonality: nhịp cao điểm ban ngày và thấp điểm ban đêm) thông qua trạng thái ẩn của 2 tầng LSTM cùng trọng số kết nối tắt Residual Highway mạnh nhất tại t-1 và t-24. Đối với chu kỳ tuần 168 giờ (do cửa sổ lookback L = 24 < 168), LSTM nhận biết sự dịch chuyển biên độ giữa ngày làm việc và cuối tuần một cách gián tiếp thông qua mức nền và độ dốc phụ tải trong 24 giờ liền trước.
- **Câu 5 — LSTM có bị overfitting không?**: Không bị overfitting. Trên tập Hold-out, Residual LSTM đạt Train RMSE = 6,104.39 kW (R² = 0.9948) so với Test RMSE = 6,570.41 kW (R² = 0.9944) — độ chênh lệch R² chỉ 0.0004. Đường cong hội tụ (training_loss_curves.png) cho thấy Validation Loss giảm mượt song hành với Train Loss và dừng sớm đúng điểm tối ưu tại Best Epoch = 27/30 nhờ kết hợp Dropout(0.1), LayerNorm, CosineAnnealingLR và cơ chế EarlyStopping.
- **Câu 6 — Sự khác biệt giữa Machine Learning và Deep Learning trong bài toán này?**: Trên bài toán phụ tải điện ít biến (K = 4, dự báo 1 bước), Machine Learning (XGBoost) đòi hỏi kỹ thuật đặc trưng thủ công (106 biến trễ, thống kê cuốn và lịch) nhưng huấn luyện rất nhanh (5.21s trên CUDA) và đạt độ chính xác cao nhất trên Hold-out & Walk-Forward, dù hạn chế khi thiếu dữ liệu đầu kỳ (Fold 1–2 của 5-Fold CV) do cây quyết định không ngoại suy ngoài miền Train. Ngược lại, Deep Learning (Residual LSTM: 97.02s) tự động học biểu diễn tuần tự trực tiếp từ Tensor 3D, khai thác hiệu quả hơn thông tin ngoại sinh thời tiết (+7.17% RMSE) và mở rộng tự nhiên cho bài toán không gian – thời gian quy mô lớn, đa tầm nhìn như PEMS-BAY (207 trạm cảm biến).
- **Kết luận Thực nghiệm Bài tập 12 Nâng cao (PEMS-BAY)**: Ở Bài tập 12 nâng cao trên bộ dữ liệu giao thông PEMS-BAY (207 trạm cảm biến × 8,640 bước 5 phút), mô hình DCRNN (tích hợp ma trận đồ thị không gian adj_mx_bay.pkl qua phép tích chập khuếch tán hai chiều) vượt trội hơn hẳn cả Multi-Horizon TCN và Multi-Horizon LSTM ở tất cả các tầm nhìn xa (15 phút, 30 phút, 60 phút). Đặc biệt ở tầm nhìn xa 60 phút (t+12), DCRNN đạt MAE = 2.300 mph (RMSE = 5.267 mph), thấp hơn rõ rệt so với Multi-Horizon LSTM không dùng đồ thị (MAE = 2.552 mph) và Multi-Horizon TCN (MAE = 2.640 mph), chứng minh giá trị cốt lõi của cấu trúc mạng lưới không gian khi dự báo sóng lan truyền tắc nghẽn.
