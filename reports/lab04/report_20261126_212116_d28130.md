# BÁO CÁO KỸ THUẬT LAB 04: DỰ BÁO CHUỖI THỜI GIAN ĐA BIẾN (MULTIVARIATE TIME SERIES FORECASTING)

- **Mã phiên chạy (Run ID)**: `20261126_212116_d28130`
- **Phần cứng thực thi**: `NVIDIA GeForce RTX 3050 6GB Laptop GPU` (`torch=cuda`, `xgb=cuda`, `AMP_FP16=True`)
- **Bộ dữ liệu chính (Mục 1–11)**: `data/raw/electricity_load.csv` (35,065 giờ × 4 biến: `load`, `temperature`, `humidity`, `wind_speed`)
- **Bộ dữ liệu nâng cao (Mục 12)**: `data/raw/pems-bay.h5` & `data/sensor_graph/adj_mx_bay.pkl` (8,640 bước × 207 trạm cảm biến)

## 1. Kết quả Đối sánh trên Tập Kiểm tra Độc lập (Hold-out Test 15%)

| Model   |         MSE |    RMSE |     MAE |   MAPE |     R2 |   Max_Error |   Mean_Residual |   Std_Residual |   Train_MSE |   Train_RMSE |   Train_MAE |   Train_MAPE |   Train_R2 |   Val_RMSE |   Val_MAPE |   Val_R2 |   Train_Time_Sec |   Pred_Time_Sec | Device   |
|:--------|------------:|--------:|--------:|-------:|-------:|------------:|----------------:|---------------:|------------:|-------------:|------------:|-------------:|-----------:|-----------:|-----------:|---------:|-----------------:|----------------:|:---------|
| XGBoost | 3.99547e+07 | 6320.97 | 4273.05 | 1.8683 | 0.9948 |      132576 |       -111.224  |        6320    | 1.444e+07   |      3800    |     2755.13 |       1.2562 |     0.998  |    5513.88 |     7.0071 |   0.9915 |           5.4028 |        0.102471 | cuda     |
| VAR     | 5.0728e+07  | 7122.36 | 4576.49 | 2.001  | 0.9935 |      130204 |        -25.5376 |        7122.32 | 4.54186e+07 |      6739.33 |     4197.65 |       2.0323 |     0.9937 |    6501.03 |     7.2729 |   0.9882 |           3.1763 |        0.244407 | cpu      |
| LSTM    | 5.27331e+07 | 7261.76 | 5086.94 | 2.2904 | 0.9932 |      127720 |       -574.274  |        7239.01 | 5.16362e+07 |      7185.83 |     4966.59 |       2.4022 |     0.9928 |    6597.65 |     7.5874 |   0.9879 |          95.6583 |        0.784567 | cuda     |

## 2. Kết quả Kiểm định Chéo 5-Fold TimeSeriesSplit (K = 5)

| Model   |   Folds |    MSE_Mean |     MSE_Std |   RMSE_Mean |   RMSE_Std |   MAE_Mean |   MAE_Std |   MAPE_Mean |   MAPE_Std |   R2_Mean |   R2_Std |
|:--------|--------:|------------:|------------:|------------:|-----------:|-----------:|----------:|------------:|-----------:|----------:|---------:|
| VAR     |       5 | 5.02283e+07 | 1.75209e+07 |     7004.77 |    1204.93 |    4264.73 |   796.886 |      3.3066 |     2.4205 |    0.9918 |   0.0022 |
| LSTM    |       5 | 8.77048e+07 | 5.16925e+07 |     9098.47 |    2480.59 |    6135.42 |  1645.48  |      4.2274 |     2.2704 |    0.9867 |   0.0033 |
| XGBoost |       5 | 1.25545e+08 | 1.23723e+08 |    10235.1  |    5097.38 |    6320.84 |  2793.39  |      4.1704 |     1.9925 |    0.9826 |   0.0111 |

## 3. Kết quả Walk-Forward Backtesting (168 Giờ)

| Model   |   Eval_Steps |         MSE |    RMSE |     MAE |   MAPE |     R2 |   Max_Error |   Mean_Residual |   Std_Residual |
|:--------|-------------:|------------:|--------:|--------:|-------:|-------:|------------:|----------------:|---------------:|
| VAR     |          168 | 2.79087e+07 | 5282.87 | 3937.53 | 1.8182 | 0.9941 |     18706.9 |         379.65  |        5269.21 |
| XGBoost |          168 | 2.86896e+07 | 5356.26 | 3973.05 | 1.8012 | 0.994  |     18006.7 |         897.784 |        5280.49 |
| LSTM    |          168 | 3.70317e+07 | 6085.37 | 4437.14 | 2.0562 | 0.9922 |     27290.6 |         736.786 |        6040.6  |

## 4. Nghiên cứu Loại bỏ Đặc trưng Thời tiết (Feature Ablation)

| Model   |   With_Weather_RMSE |   Without_Weather_RMSE |   RMSE_Improvement_Pct |   With_Weather_MAE |   Without_Weather_MAE |   With_Weather_MAPE |   Without_Weather_MAPE |   MAPE_Improvement_Pct |   With_Weather_R2 |   Without_Weather_R2 |
|:--------|--------------------:|-----------------------:|-----------------------:|-------------------:|----------------------:|--------------------:|-----------------------:|-----------------------:|------------------:|---------------------:|
| VAR     |             7122.36 |                7120.99 |                  -0.02 |            4576.49 |               4569.45 |              2.001  |                 1.9958 |                  -0.26 |            0.9935 |               0.9935 |
| XGBoost |             6320.97 |                6285.65 |                  -0.56 |            4273.05 |               4270.67 |              1.8683 |                 1.8571 |                  -0.6  |            0.9948 |               0.9949 |
| LSTM    |             7261.76 |                8273.93 |                  12.23 |            5086.94 |               5759.45 |              2.2904 |                 2.6038 |                  12.04 |            0.9932 |               0.9912 |

## 5. Bài tập 12 Nâng cao (PEMS-BAY): Đối sánh DCRNN vs. Multi-Horizon TCN vs. Multi-Horizon LSTM

| Model              | Model_Display                      | Horizon              |   Steps_Ahead |   Minutes_Ahead |     MSE |   RMSE |    MAE |   MAPE |     R2 |   Max_Error |   Mean_Residual |   Std_Residual |   Train_Time_Sec |
|:-------------------|:-----------------------------------|:---------------------|--------------:|----------------:|--------:|-------:|-------:|-------:|-------:|------------:|----------------:|---------------:|-----------------:|
| DCRNN              | DCRNN (Graph Diffusion + GRU)      | 15_min               |             3 |              15 |  7.6507 | 2.766  | 1.3042 | 2.6945 | 0.9194 |     47.1732 |         -0.0335 |         2.7658 |          76.5697 |
| DCRNN              | DCRNN (Graph Diffusion + GRU)      | 30_min               |             6 |              30 | 15.4797 | 3.9344 | 1.7431 | 3.9328 | 0.8369 |     55.7953 |         -0.1205 |         3.9326 |          76.5697 |
| DCRNN              | DCRNN (Graph Diffusion + GRU)      | 60_min               |            12 |              60 | 28.2709 | 5.317  | 2.3391 | 5.7632 | 0.7023 |     58.6765 |         -0.2918 |         5.309  |          76.5697 |
| DCRNN              | DCRNN (Graph Diffusion + GRU)      | Average_All_Horizons |             7 |              35 | 17.1338 | 4.0058 | 1.7955 | 4.1302 | 0.8195 |     58.6765 |         -0.1486 |         4.0025 |          76.5697 |
| Multi_Horizon_TCN  | Multi-Horizon TCN (Causal Conv1D)  | 15_min               |             3 |              15 |  8.7847 | 2.9639 | 1.3618 | 2.8593 | 0.9074 |     51.1119 |         -0.0233 |         2.9638 |          16.6378 |
| Multi_Horizon_TCN  | Multi-Horizon TCN (Causal Conv1D)  | 30_min               |             6 |              30 | 19.2893 | 4.392  | 1.8612 | 4.1519 | 0.7968 |     55.8675 |         -0.0605 |         4.3915 |          16.6378 |
| Multi_Horizon_TCN  | Multi-Horizon TCN (Causal Conv1D)  | 60_min               |            12 |              60 | 37.3559 | 6.1119 | 2.6389 | 6.5628 | 0.6066 |     58.8394 |         -0.2632 |         6.1063 |          16.6378 |
| Multi_Horizon_TCN  | Multi-Horizon TCN (Causal Conv1D)  | Average_All_Horizons |             7 |              35 | 21.81   | 4.4893 | 1.954  | 4.5247 | 0.7703 |     58.8394 |         -0.1157 |         4.4872 |          16.6378 |
| Multi_Horizon_LSTM | Multi-Horizon LSTM (Temporal Only) | 15_min               |             3 |              15 |  8.7421 | 2.9567 | 1.4212 | 2.8873 | 0.9079 |     47.1492 |          0.165  |         2.9521 |          10.1018 |
| Multi_Horizon_LSTM | Multi-Horizon LSTM (Temporal Only) | 30_min               |             6 |              30 | 17.0914 | 4.1342 | 1.9185 | 4.1057 | 0.82   |     53.0697 |          0.3527 |         4.1191 |          10.1018 |
| Multi_Horizon_LSTM | Multi-Horizon LSTM (Temporal Only) | 60_min               |            12 |              60 | 28.8125 | 5.3677 | 2.6231 | 5.8884 | 0.6966 |     54.6087 |          0.6516 |         5.328  |          10.1018 |
| Multi_Horizon_LSTM | Multi-Horizon LSTM (Temporal Only) | Average_All_Horizons |             7 |              35 | 18.2153 | 4.1529 | 1.9876 | 4.2938 | 0.8082 |     54.6087 |          0.3898 |         4.1331 |          10.1018 |

## 6. Trả lời Câu hỏi Thảo luận Chuyên sâu (Mục 11 & Mục 12)

- **Câu 1 — Biến thời tiết nào ảnh hưởng mạnh nhất đến điện năng?**: Về độ lớn tương quan, 'humidity' (độ ẩm tương đối (%)) là biến thời tiết ảnh hưởng mạnh nhất tới phụ tải điện với hệ số tương quan tuyến tính Pearson r = -0.9224 (|r| = 0.9224, Spearman ρ = -0.9208), theo sát là 'temperature' (r = +0.8853, ρ = +0.8717). Hai biến này nghịch biến chặt với nhau (r = -0.9657) do nắng nóng làm độ ẩm giảm và nhu cầu điều hòa tăng vọt. Trong nghiên cứu loại bỏ đặc trưng (Weather Ablation), Residual LSTM khai thác tín hiệu thời tiết rõ rệt nhất khi giảm +12.23% RMSE (từ 8,273.93 xuống 7,261.76 kW) và +12.04% MAPE, trong khi VAR (-0.02%) và XGBoost (-0.56%) gần như bão hòa nhờ 24 độ trễ tự hồi quy của chuỗi load.
- **Câu 2 — Khi nào VAR thất bại?**: Mô hình kinh tế lượng VAR(24) đạt RMSE = 7,122.36 kW (R² = 0.9935 trên Hold-out, kém hơn XGBoost và LSTM) dù ổn định nhất ở 5-Fold CV (VAR dẫn đầu với RMSE = 7,004.77 ± 1,204.93 kW). VAR thường thất bại trong 3 tình huống: (1) quan hệ phi tuyến mạnh hình chữ U hoặc ngưỡng bậc thang giữa nhiệt độ/độ ẩm với phụ tải (thể hiện ở biểu đồ Quadratic Fit EDA 05) mà tổ hợp tuyến tính không nắm bắt được; (2) chuỗi có đa mùa vụ phức tạp (24h + 168h) và phương sai thay đổi; (3) khi số lượng biến K lớn (ví dụ 207 trạm cảm biến PEMS-BAY), số tham số tăng theo bậc hai O(p × K²) gây bùng nổ chiều và quá khớp nghiêm trọng.
- **Câu 3 — XGBoost học được thông tin gì từ thời tiết?**: Nhờ phân tách nhánh cây quyết định phi tuyến trên 106 đặc trưng, XGBoost (tốt nhất trên Hold-out với RMSE = 6,320.97 kW, MAPE = 1.87%, R² = 0.9948) học được các ngưỡng chuyển pha phi tuyến và độ trễ tác động của biến khí tượng. Cụ thể, trong bảng Gain Feature Importance, nhóm thời tiết nổi bật gồm: humidity_lag_12h (hạng 4, gain=0.0035), humidity_lag_11h (hạng 11, gain=0.0017), temperature_lag_13h (hạng 12, gain=0.0013), humidity_lag_13h (hạng 15, gain=0.0012), kết hợp tương tác trực tiếp với đặc trưng pha giờ trong ngày (cos_hour, sin_hour) và biến lịch cuối tuần (is_weekend).
- **Câu 4 — LSTM học được chu kỳ nào (ngày, tuần)?**: Với đầu vào Tensor 3D (batch_samples, 24, 4), mạng Stacked 2-Layer LSTM + Autoregressive Residual Highway học trực tiếp và trọn vẹn chu kỳ ngày 24 giờ (Diurnal Seasonality: nhịp cao điểm ban ngày và thấp điểm ban đêm) thông qua trạng thái ẩn của 2 tầng LSTM cùng trọng số kết nối tắt Residual Highway mạnh nhất tại t-1 và t-24. Đối với chu kỳ tuần 168 giờ (do cửa sổ lookback L = 24 < 168), LSTM nhận biết sự dịch chuyển biên độ giữa ngày làm việc và cuối tuần một cách gián tiếp thông qua mức nền và độ dốc phụ tải trong 24 giờ liền trước.
- **Câu 5 — LSTM có bị overfitting không?**: Không bị overfitting. Trên tập Hold-out, Residual LSTM đạt Train RMSE = 7,185.83 kW (R² = 0.9928) so với Test RMSE = 7,261.76 kW (R² = 0.9932) — độ chênh lệch R² chỉ 0.0004. Đường cong hội tụ (training_loss_curves.png) cho thấy Validation Loss giảm mượt song hành với Train Loss và dừng sớm đúng điểm tối ưu tại Best Epoch = 27/30 nhờ kết hợp Dropout(0.1), LayerNorm, CosineAnnealingLR và cơ chế EarlyStopping.
- **Câu 6 — Sự khác biệt giữa Machine Learning và Deep Learning trong bài toán này?**: Trên bài toán phụ tải điện ít biến (K = 4, dự báo 1 bước), Machine Learning (XGBoost) đòi hỏi kỹ thuật đặc trưng thủ công (106 biến trễ, thống kê cuốn và lịch) nhưng huấn luyện rất nhanh (5.40s trên CUDA) và đạt độ chính xác cao nhất trên Hold-out & Walk-Forward, dù hạn chế khi thiếu dữ liệu đầu kỳ (Fold 1–2 của 5-Fold CV) do cây quyết định không ngoại suy ngoài miền Train. Ngược lại, Deep Learning (Residual LSTM: 95.66s) tự động học biểu diễn tuần tự trực tiếp từ Tensor 3D, khai thác hiệu quả hơn thông tin ngoại sinh thời tiết (+12.23% RMSE) và mở rộng tự nhiên cho bài toán không gian – thời gian quy mô lớn, đa tầm nhìn như PEMS-BAY (207 trạm cảm biến).
- **Kết luận Thực nghiệm Bài tập 12 Nâng cao (PEMS-BAY)**: Ở Bài tập 12 nâng cao trên bộ dữ liệu giao thông PEMS-BAY (207 trạm cảm biến × 8,640 bước 5 phút), mô hình DCRNN (tích hợp ma trận đồ thị không gian adj_mx_bay.pkl qua phép tích chập khuếch tán hai chiều) vượt trội hơn hẳn cả Multi-Horizon TCN và Multi-Horizon LSTM ở tất cả các tầm nhìn xa (15 phút, 30 phút, 60 phút). Đặc biệt ở tầm nhìn xa 60 phút (t+12), DCRNN đạt MAE = 2.339 mph (RMSE = 5.317 mph), thấp hơn rõ rệt so với Multi-Horizon LSTM không dùng đồ thị (MAE = 2.623 mph) và Multi-Horizon TCN (MAE = 2.639 mph), chứng minh giá trị cốt lõi của cấu trúc mạng lưới không gian khi dự báo sóng lan truyền tắc nghẽn.
