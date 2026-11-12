# BÁO CÁO TỔNG HỢP MÔ HÌNH (MODEL SUMMARY) — AIRPASSENGERS (MONTHLY AIRLINE PASSENGERS, 01/1949 - 12/1960)

## 1. Thông tin Phiên Thực nghiệm & Bộ dữ liệu (Provenance & Dataset Info)
- **Run ID**: `dataset2_20261112_082858_8425b2`
- **Thời điểm xuất báo cáo**: `2026-11-12 08:30:57`
- **Dataset Key**: `dataset2`
- **Tên bộ dữ liệu**: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
- **Tổng số quan sát**: 144 tháng (1949-01-01 to 1960-12-01)
- **Tần suất**: `MS` (Monthly Start)
- **Tập Huấn luyện (Train - 80%)**: 115 quan sát
- **Tập Kiểm thử (Test - 20%)**: 29 quan sát
- **Tổng số mô hình ứng viên**: 51 (Thành công: 51, Thất bại: 0)

## 2. Bảng So sánh Tổng hợp các Mô hình Ứng viên (Model Comparison Table)

> **Nguyên tắc đối chiếu**: Tất cả chỉ số dự báo ngoài mẫu (`MAE`, `RMSE`, `MAPE (%)`, `R²`) đều được đánh giá trên **cùng tập Test và cùng thang đo gốc** (đã áp dụng biến đổi ngược $\exp(\cdot)$ cho các mô hình `Log-SARIMA`). Chỉ số `AIC` và `BIC` chỉ được so sánh trực tiếp trong cùng một nhóm `Scale_Group`.

| Model                        | Family     | Status   | Scale_Group        |     AIC |     BIC |   MAE |   RMSE |   MAPE (%) |      R² |   Ljung-Box p | White Noise      | Converged   |
|:-----------------------------|:-----------|:---------|:-------------------|--------:|--------:|------:|-------:|-----------:|--------:|--------------:|:-----------------|:------------|
| Log-SARIMA(0,1,0)x(1,1,1)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -310.59 | -303.13 | 12.06 |  14.66 |       2.8  |  0.9648 |        0.0353 | Fail (Không đạt) | True        |
| Log-SARIMA(1,1,3)x(0,1,1)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -322.9  | -308.17 | 12.02 |  14.67 |       2.79 |  0.9647 |        0.9954 | Pass (Đạt)       | False       |
| Log-SARIMA(1,1,1)x(0,1,1)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -322.51 | -312.6  | 12.13 |  14.85 |       2.79 |  0.9639 |        1      | Pass (Đạt)       | True        |
| Log-SARIMA(1,1,0)x(0,1,1)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -327.47 | -320    | 12.35 |  15.12 |       2.81 |  0.9626 |        0.997  | Pass (Đạt)       | True        |
| Log-SARIMA(0,1,0)x(0,1,1)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -318.57 | -313.59 | 12.67 |  15.45 |       3    |  0.9609 |        0.1826 | Pass (Đạt)       | True        |
| Log-SARIMA(0,1,1)x(0,1,1)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -323.75 | -316.32 | 12.95 |  16.16 |       2.89 |  0.9572 |        0.7605 | Pass (Đạt)       | True        |
| Log-SARIMA(0,1,3)x(0,1,1)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -324.05 | -311.78 | 13.07 |  16.26 |       2.92 |  0.9567 |        0.921  | Pass (Đạt)       | True        |
| Log-SARIMA(1,1,0)x(1,1,1)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -320.06 | -310.11 | 17.7  |  21.51 |       3.88 |  0.9242 |        0.2187 | Pass (Đạt)       | False       |
| Log-SARIMA(1,1,3)x(1,1,1)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -313.73 | -296.54 | 18.55 |  22.53 |       4.05 |  0.9169 |        0.849  | Pass (Đạt)       | False       |
| Log-SARIMA(1,1,1)x(1,1,1)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -315.09 | -302.7  | 18.66 |  22.66 |       4.06 |  0.9159 |        0.7975 | Pass (Đạt)       | True        |
| Log-SARIMA(0,1,0)x(1,1,0)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -312.5  | -307.51 | 18.73 |  22.94 |       4.12 |  0.9138 |        0.0062 | Fail (Không đạt) | True        |
| Log-SARIMA(0,1,1)x(1,1,1)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -317    | -307.09 | 18.88 |  22.95 |       4.11 |  0.9137 |        0.9777 | Pass (Đạt)       | True        |
| Log-SARIMA(0,1,3)x(1,1,1)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -315.7  | -300.97 | 19.46 |  23.66 |       4.24 |  0.9083 |        0.935  | Pass (Đạt)       | True        |
| SARIMA(1,1,3)x(0,1,1)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  639.62 |  654.35 | 19.43 |  24.85 |       4.31 |  0.8988 |        0.4105 | Pass (Đạt)       | True        |
| SARIMA(1,1,3)x(1,1,1)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  641.62 |  658.8  | 19.43 |  24.85 |       4.31 |  0.8988 |        0.4107 | Pass (Đạt)       | True        |
| SARIMA(1,1,3)x(1,1,0)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  663.59 |  678.52 | 19.52 |  25.09 |       4.32 |  0.8969 |        0.4773 | Pass (Đạt)       | True        |
| Log-SARIMA(1,1,0)x(1,1,0)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -321.65 | -314.18 | 21.34 |  26.25 |       4.64 |  0.8871 |        0.0647 | Pass (Đạt)       | True        |
| Log-SARIMA(1,1,1)x(1,1,0)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -319.65 | -309.7  | 21.5  |  26.46 |       4.67 |  0.8853 |        0.0538 | Pass (Đạt)       | True        |
| SARIMA(0,1,0)x(0,1,1)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  668.04 |  673.02 | 20.53 |  26.62 |       4.45 |  0.8839 |        0.025  | Fail (Không đạt) | True        |
| SARIMA(0,1,0)x(1,1,1)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  670.04 |  677.51 | 20.53 |  26.62 |       4.45 |  0.8839 |        0.025  | Fail (Không đạt) | True        |
| SARIMA(0,1,0)x(1,1,0)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  674.72 |  679.72 | 20.67 |  26.83 |       4.48 |  0.882  |        0.0286 | Fail (Không đạt) | True        |
| Log-SARIMA(0,1,3)x(1,1,0)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -325.26 | -312.77 | 22.67 |  28.06 |       4.91 |  0.8711 |        0.0568 | Pass (Đạt)       | True        |
| Log-SARIMA(1,1,3)x(1,1,0)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -319.7  | -304.77 | 23.1  |  28.54 |       5    |  0.8665 |        0.9951 | Pass (Đạt)       | True        |
| SARIMA(1,1,1)x(1,1,0)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  666.9  |  676.86 | 22.11 |  28.61 |       4.75 |  0.8659 |        0.5469 | Pass (Đạt)       | True        |
| Log-SARIMA(0,1,1)x(1,1,0)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -325.31 | -317.81 | 23.54 |  29.13 |       5.08 |  0.861  |        0.3629 | Pass (Đạt)       | True        |
| SARIMA(1,1,1)x(0,1,1)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  659.33 |  669.24 | 23.37 |  29.92 |       5.01 |  0.8533 |        0.3909 | Pass (Đạt)       | True        |
| SARIMA(1,1,0)x(1,1,0)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  666.51 |  673.98 | 23.9  |  30.43 |       5.12 |  0.8484 |        0.0676 | Pass (Đạt)       | True        |
| SARIMA(0,1,1)x(0,1,1)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  659.34 |  666.77 | 24.06 |  30.58 |       5.15 |  0.8468 |        0.0468 | Fail (Không đạt) | True        |
| SARIMA(0,1,1)x(1,1,0)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  673.77 |  681.26 | 24.14 |  30.65 |       5.17 |  0.8462 |        0.0501 | Pass (Đạt)       | True        |
| SARIMA(1,1,0)x(0,1,1)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  666.75 |  674.21 | 24.25 |  30.77 |       5.19 |  0.8449 |        0.071  | Pass (Đạt)       | True        |
| SARIMA(1,1,1)x(1,1,1)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  657.6  |  669.99 | 25.27 |  31.79 |       5.43 |  0.8345 |        0.4519 | Pass (Đạt)       | True        |
| SARIMA(1,1,1)x(0,1,0)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  742.94 |  750.75 | 27.54 |  34.06 |       5.91 |  0.8099 |        0.1167 | Pass (Đạt)       | True        |
| SARIMA(1,1,0)x(0,1,0)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  748.83 |  754.06 | 28.57 |  35.1  |       6.13 |  0.7981 |        0.0652 | Pass (Đạt)       | True        |
| SARIMA(0,1,1)x(0,1,0)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  742.92 |  748.13 | 29.21 |  35.74 |       6.27 |  0.7907 |        0.0576 | Pass (Đạt)       | True        |
| SARIMA(1,1,3)x(0,1,0)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  728.13 |  741.05 | 29.49 |  36.05 |       6.34 |  0.7871 |        0.4953 | Pass (Đạt)       | True        |
| SARIMA(1,1,0)x(1,1,1)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  665.93 |  675.89 | 29.58 |  36.1  |       6.35 |  0.7865 |        0.1351 | Pass (Đạt)       | True        |
| SARIMA(0,1,3)x(1,1,0)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  671.03 |  683.53 | 29.68 |  36.33 |       6.35 |  0.7838 |        0.3149 | Pass (Đạt)       | True        |
| SARIMA(0,1,3)x(0,1,1)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  641.08 |  653.35 | 29.75 |  36.39 |       6.36 |  0.783  |        0.3362 | Pass (Đạt)       | True        |
| SARIMA(0,1,1)x(1,1,1)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  658.39 |  668.3  | 30.42 |  36.96 |       6.53 |  0.7763 |        0.1314 | Pass (Đạt)       | True        |
| SARIMA(0,1,3)x(0,1,0)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  726.91 |  737.25 | 33.2  |  39.99 |       7.13 |  0.7381 |        0.4398 | Pass (Đạt)       | True        |
| SARIMA(0,1,3)x(1,1,1)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  640.22 |  654.95 | 33.47 |  40.27 |       7.19 |  0.7343 |        0.1787 | Pass (Đạt)       | True        |
| Log-SARIMA(1,1,0)x(0,1,0)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -341.24 | -336.01 | 34.71 |  40.68 |       7.56 |  0.729  |        0.0017 | Fail (Không đạt) | False       |
| Log-SARIMA(1,1,1)x(0,1,0)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -334.91 | -327.09 | 34.73 |  40.7  |       7.57 |  0.7287 |        0.0017 | Fail (Không đạt) | True        |
| Log-SARIMA(0,1,3)x(0,1,0)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -329.22 | -318.88 | 36.5  |  42.74 |       7.96 |  0.7007 |        0.0651 | Pass (Đạt)       | True        |
| Log-SARIMA(0,1,1)x(0,1,0)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -337.49 | -332.28 | 37.39 |  43.74 |       8.14 |  0.6866 |        0.0054 | Fail (Không đạt) | True        |
| Log-SARIMA(1,1,3)x(0,1,0)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -327.62 | -314.69 | 37.78 |  44.2  |       8.22 |  0.6799 |        0.0924 | Pass (Đạt)       | True        |
| ARIMA(1,1,3)                 | ARIMA      | SUCCESS  | original_d1        | 1058.95 | 1072.63 | 62.28 |  82.54 |      13.68 | -0.116  |        0      | Fail (Không đạt) | True        |
| MA(3) [ARIMA(0,1,3)]         | MA         | SUCCESS  | original_d1        | 1062.47 | 1073.41 | 85.17 |  97.4  |      21.27 | -0.5541 |        0      | Fail (Không đạt) | True        |
| ARIMA(1,1,1)                 | ARIMA      | SUCCESS  | original_d1        | 1058.25 | 1066.45 | 85.25 |  97.5  |      21.29 | -0.5572 |        0      | Fail (Không đạt) | True        |
| MA(1) [ARIMA(0,1,1)]         | MA         | SUCCESS  | original_d1        | 1063.65 | 1069.12 | 88.39 | 101.07 |      22.17 | -0.6734 |        0      | Fail (Không đạt) | True        |
| AR(1) [ARIMA(1,1,0)]         | AR         | SUCCESS  | original_d1        | 1068.54 | 1074.01 | 94    | 107.22 |      23.69 | -0.8832 |        0      | Fail (Không đạt) | True        |

### Diễn giải Đa Tiêu chí Tự động (Run ID: `dataset2_20261112_082858_8425b2` — AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960))

#### A. Mô hình Tốt nhất theo Từng Tiêu chí Độc lập (Criterion-Specific Leaders)
- **Tiêu chí 1 — Sai số RMSE trên tập Test thấp nhất (Thang đo gốc)**: **Log-SARIMA(0,1,0)x(1,1,1)_12** (`RMSE = 14.66`, `R² = 0.9648`).
- **Tiêu chí 2 — Sai số tuyệt đối MAE trên tập Test thấp nhất**: **Log-SARIMA(1,1,3)x(0,1,1)_12** (`MAE = 12.02`).
- **Tiêu chí 3 — Sai số phần trăm MAPE (%) trên tập Test thấp nhất**: **Log-SARIMA(1,1,3)x(0,1,1)_12** (`MAPE = 2.79%`).
- **Tiêu chí 4 — Mô hình thỏa mãn đồng thời Chuẩn Nhiễu trắng (Ljung-Box $p \ge 0.05$), Hội tụ và có RMSE thấp nhất**: **Log-SARIMA(1,1,1)x(0,1,1)_12**.
- **Tiêu chí 5 — Tiêu chuẩn Thông tin (AIC / BIC) trên Tập Train theo từng nhóm thang đo hợp lệ (`Scale_Group`)**:
  - Nhóm `log_d1_D1_s12`: AIC thấp nhất là **Log-SARIMA(1,1,0)x(0,1,0)_12** (`AIC = -341.24`, `Test RMSE = 40.68`); BIC thấp nhất là **Log-SARIMA(1,1,0)x(0,1,0)_12** (`BIC = -336.01`).
  - Nhóm `original_d1`: AIC thấp nhất là **ARIMA(1,1,1)** (`AIC = 1058.25`, `Test RMSE = 97.50`); BIC thấp nhất là **ARIMA(1,1,1)** (`BIC = 1066.45`).
  - Nhóm `original_d1_D1_s12`: AIC thấp nhất là **SARIMA(1,1,3)x(0,1,1)_12** (`AIC = 639.62`, `Test RMSE = 24.85`); BIC thấp nhất là **SARIMA(0,1,3)x(0,1,1)_12** (`BIC = 653.35`).

#### B. Kiểm định Phần dư & Trạng thái Hội tụ
- **Các mô hình vượt qua kiểm định Ljung-Box ($p \ge 0.05$, phần dư đạt chuẩn White Noise)**: `Log-SARIMA(1,1,3)x(0,1,1)_12`, `Log-SARIMA(1,1,1)x(0,1,1)_12`, `Log-SARIMA(1,1,0)x(0,1,1)_12`, `Log-SARIMA(0,1,0)x(0,1,1)_12`, `Log-SARIMA(0,1,1)x(0,1,1)_12`, `Log-SARIMA(0,1,3)x(0,1,1)_12`, `Log-SARIMA(1,1,0)x(1,1,1)_12`, `Log-SARIMA(1,1,3)x(1,1,1)_12`, `Log-SARIMA(1,1,1)x(1,1,1)_12`, `Log-SARIMA(0,1,1)x(1,1,1)_12`, `Log-SARIMA(0,1,3)x(1,1,1)_12`, `SARIMA(1,1,3)x(0,1,1)_12`, `SARIMA(1,1,3)x(1,1,1)_12`, `SARIMA(1,1,3)x(1,1,0)_12`, `Log-SARIMA(1,1,0)x(1,1,0)_12`, `Log-SARIMA(1,1,1)x(1,1,0)_12`, `Log-SARIMA(0,1,3)x(1,1,0)_12`, `Log-SARIMA(1,1,3)x(1,1,0)_12`, `SARIMA(1,1,1)x(1,1,0)_12`, `Log-SARIMA(0,1,1)x(1,1,0)_12`, `SARIMA(1,1,1)x(0,1,1)_12`, `SARIMA(1,1,0)x(1,1,0)_12`, `SARIMA(0,1,1)x(1,1,0)_12`, `SARIMA(1,1,0)x(0,1,1)_12`, `SARIMA(1,1,1)x(1,1,1)_12`, `SARIMA(1,1,1)x(0,1,0)_12`, `SARIMA(1,1,0)x(0,1,0)_12`, `SARIMA(0,1,1)x(0,1,0)_12`, `SARIMA(1,1,3)x(0,1,0)_12`, `SARIMA(1,1,0)x(1,1,1)_12`, `SARIMA(0,1,3)x(1,1,0)_12`, `SARIMA(0,1,3)x(0,1,1)_12`, `SARIMA(0,1,1)x(1,1,1)_12`, `SARIMA(0,1,3)x(0,1,0)_12`, `SARIMA(0,1,3)x(1,1,1)_12`, `Log-SARIMA(0,1,3)x(0,1,0)_12`, `Log-SARIMA(1,1,3)x(0,1,0)_12`.
- **Các mô hình vi phạm kiểm định Ljung-Box ($p < 0.05$, phần dư còn tự tương quan)**: `Log-SARIMA(0,1,0)x(1,1,1)_12`, `Log-SARIMA(0,1,0)x(1,1,0)_12`, `SARIMA(0,1,0)x(0,1,1)_12`, `SARIMA(0,1,0)x(1,1,1)_12`, `SARIMA(0,1,0)x(1,1,0)_12`, `SARIMA(0,1,1)x(0,1,1)_12`, `Log-SARIMA(1,1,0)x(0,1,0)_12`, `Log-SARIMA(1,1,1)x(0,1,0)_12`, `Log-SARIMA(0,1,1)x(0,1,0)_12`, `ARIMA(1,1,3)`, `MA(3) [ARIMA(0,1,3)]`, `ARIMA(1,1,1)`, `MA(1) [ARIMA(0,1,1)]`, `AR(1) [ARIMA(1,1,0)]`.
- **Các mô hình có cảnh báo không hội tụ (`Converged = False`)**: `Log-SARIMA(1,1,3)x(0,1,1)_12`, `Log-SARIMA(1,1,0)x(1,1,1)_12`, `Log-SARIMA(1,1,3)x(1,1,1)_12`, `Log-SARIMA(1,1,0)x(0,1,0)_12`.
- **Các mô hình gặp lỗi thực thi (`Status = FAILED`)**: Không có (0 lỗi).

#### C. Lưu ý Phương pháp luận (Tại sao không khẳng định một mô hình tốt nhất tuyệt đối duy nhất?)
1. **Không so sánh chéo AIC/BIC giữa các nhóm khác bậc sai phân hoặc khác phép biến đổi**: Giá trị Log-Likelihood của nhóm `log_d1_D1_s12` được tính trên biến $\ln(Y_t)$ nên có AIC âm, không thể đem so sánh trực tiếp với AIC dương của nhóm `original_d1_D1_s12` hoặc `original_d1`.
2. **Sự đánh đổi giữa Statistical Fit (AIC/BIC), Residual Adequacy (Ljung-Box) và Out-of-Sample Error (RMSE/MAE)**: Một mô hình có thể đạt RMSE thấp nhất trên tập Test nhờ biến đổi Logarit nhưng phần dư trên không gian Log vẫn còn tự tương quan nhẹ (như trên Dataset 1), hoặc một mô hình bậc cao có AIC thấp nhất nhưng tốn nhiều tham số hơn cấu hình tối giản (như `SARIMA(2,1,3)` so với `SARIMA(0,1,1)`). Do đó, việc lựa chọn mô hình cuối cùng phụ thuộc vào mục tiêu ưu tiên độ chính xác dự báo ngoài mẫu hay tính chặt chẽ của giả định nhiễu trắng.

## 3. Hồ sơ Thống kê Chi tiết Từng Mô hình (Individual Model Cards)

### 3.1. AR(1) [ARIMA(1,1,0)]
```text
AR Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 0
Transformation: Original Scale (Sai phân bậc d=1)
Model: AR(1) [ARIMA(1,1,0)]
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -532.268   AIC:       1068.536
BIC:                  1074.008   HQIC:      1070.757
Converged:                True   Run Time:   0.0374s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1                0.2904     0.0886     3.278     0.001      0.117      0.464
sigma2             664.7320    89.2526     7.448     0.000    489.800    839.664
==============================================================================
Ljung-Box (L=24) (Q):     198.80   Jarque-Bera (JB):       3.25
Prob(Q):                   0.0000   Prob(JB):             0.1969
Heteroskedasticity(H):       6.18   Skew:                   0.40
Prob(H) (two-sided):       0.0000   Kurtosis:               2.79
Residual Mean:             2.5016   Residual Std:        25.7746
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 94.0007 | RMSE = 107.2198 | MAPE = 23.69% | R² = -0.8832
==============================================================================
```

### 3.2. MA(1) [ARIMA(0,1,1)]
```text
MA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 1
Transformation: Original Scale (Sai phân bậc d=1)
Model: MA(1) [ARIMA(0,1,1)]
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -529.823   AIC:       1063.646
BIC:                  1069.118   HQIC:      1065.867
Converged:                True   Run Time:   0.0374s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ma.L1                0.4454     0.0847     5.259     0.000      0.279      0.611
sigma2             636.0663    97.0054     6.557     0.000    445.939    826.193
==============================================================================
Ljung-Box (L=24) (Q):     178.83   Jarque-Bera (JB):       3.13
Prob(Q):                   0.0000   Prob(JB):             0.2094
Heteroskedasticity(H):       5.71   Skew:                   0.32
Prob(H) (two-sided):       0.0000   Kurtosis:               2.51
Residual Mean:             2.3849   Residual Std:        25.2203
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 88.3921 | RMSE = 101.0700 | MAPE = 22.17% | R² = -0.6734
==============================================================================
```

### 3.3. MA(3) [ARIMA(0,1,3)]
```text
MA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 3
Transformation: Original Scale (Sai phân bậc d=1)
Model: MA(3) [ARIMA(0,1,3)]
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -527.233   AIC:       1062.466
BIC:                  1073.411   HQIC:      1066.908
Converged:                True   Run Time:   0.0639s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ma.L1                0.4354     0.0934     4.664     0.000      0.252      0.618
ma.L2               -0.2130     0.0889    -2.395     0.017     -0.387     -0.039
ma.L3                0.0622     0.0958     0.650     0.516     -0.126      0.250
sigma2             605.9598   112.0738     5.407     0.000    386.299    825.620
==============================================================================
Ljung-Box (L=24) (Q):     160.07   Jarque-Bera (JB):       3.13
Prob(Q):                   0.0000   Prob(JB):             0.2089
Heteroskedasticity(H):       5.10   Skew:                   0.03
Prob(H) (two-sided):       0.0000   Kurtosis:               2.19
Residual Mean:             2.6502   Residual Std:        24.5892
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 85.1729 | RMSE = 97.4015 | MAPE = 21.27% | R² = -0.5541
==============================================================================
```

### 3.4. ARIMA(1,1,1)
```text
ARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 1
Transformation: Original Scale (Sai phân bậc d=1)
Model: ARIMA(1,1,1)
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -526.123   AIC:       1058.246
BIC:                  1066.454   HQIC:      1061.577
Converged:                True   Run Time:   0.0620s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.5111     0.1139    -4.488     0.000     -0.734     -0.288
ma.L1                0.9144     0.0563    16.251     0.000      0.804      1.025
sigma2             592.7851   101.2077     5.857     0.000    394.422    791.148
==============================================================================
Ljung-Box (L=24) (Q):     162.71   Jarque-Bera (JB):       2.32
Prob(Q):                   0.0000   Prob(JB):             0.3138
Heteroskedasticity(H):       5.22   Skew:                   0.03
Prob(H) (two-sided):       0.0000   Kurtosis:               2.30
Residual Mean:             2.6881   Residual Std:        24.3208
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 85.2542 | RMSE = 97.4996 | MAPE = 21.29% | R² = -0.5572
==============================================================================
```

### 3.5. ARIMA(1,1,3)
```text
ARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 3
Transformation: Original Scale (Sai phân bậc d=1)
Model: ARIMA(1,1,3)
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -524.475   AIC:       1058.949
BIC:                  1072.630   HQIC:      1064.502
Converged:                True   Run Time:   0.0917s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1                0.6029     0.3156     1.910     0.056     -0.016      1.221
ma.L1               -0.2651     0.3377    -0.785     0.432     -0.927      0.397
ma.L2               -0.6045     0.1047    -5.771     0.000     -0.810     -0.399
ma.L3                0.0759     0.2026     0.375     0.708     -0.321      0.473
sigma2             576.4252   107.0187     5.386     0.000    366.672    786.178
==============================================================================
Ljung-Box (L=24) (Q):     162.14   Jarque-Bera (JB):       4.67
Prob(Q):                   0.0000   Prob(JB):             0.0966
Heteroskedasticity(H):       5.39   Skew:                   0.15
Prob(H) (two-sided):       0.0000   Kurtosis:               2.05
Residual Mean:             5.0732   Residual Std:        23.5807
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 62.2777 | RMSE = 82.5369 | MAPE = 13.68% | R² = -0.1160
==============================================================================
```

### 3.6. SARIMA(0,1,0)x(0,1,1)_12
```text
SARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 0
Seasonal order (P, D, Q, s): (0, 1, 1, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(0,1,0)x(0,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -332.020   AIC:        668.040
BIC:                   673.017   HQIC:       670.046
Converged:                True   Run Time:   0.0691s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ma.S.L12            -0.1728     0.0990    -1.746     0.081     -0.367      0.021
sigma2             101.8146    17.4633     5.830     0.000     67.587    136.042
==============================================================================
Ljung-Box (L=24) (Q):      39.36   Jarque-Bera (JB):       2.07
Prob(Q):                   0.0250   Prob(JB):             0.3558
Heteroskedasticity(H):       0.65   Skew:                   0.30
Prob(H) (two-sided):       0.2400   Kurtosis:               2.64
Residual Mean:             0.3237   Residual Std:         9.7337
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 20.5311 | RMSE = 26.6201 | MAPE = 4.45% | R² = 0.8839
==============================================================================
```

### 3.7. SARIMA(0,1,0)x(1,1,0)_12
```text
SARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 0
Seasonal order (P, D, Q, s): (1, 1, 0, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(0,1,0)x(1,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -335.358   AIC:        674.716
BIC:                   679.716   HQIC:       676.732
Converged:                True   Run Time:   0.0249s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.S.L12            -0.1809     0.0894    -2.023     0.043     -0.356     -0.006
sigma2             100.9400    17.3437     5.820     0.000     66.947    134.933
==============================================================================
Ljung-Box (L=24) (Q):      38.81   Jarque-Bera (JB):       2.26
Prob(Q):                   0.0286   Prob(JB):             0.3231
Heteroskedasticity(H):       0.66   Skew:                   0.29
Prob(H) (two-sided):       0.2572   Kurtosis:               2.56
Residual Mean:             0.3142   Residual Std:         9.7974
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 20.6695 | RMSE = 26.8348 | MAPE = 4.48% | R² = 0.8820
==============================================================================
```

### 3.8. SARIMA(0,1,0)x(1,1,1)_12
```text
SARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 0
Seasonal order (P, D, Q, s): (1, 1, 1, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(0,1,0)x(1,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -332.020   AIC:        670.040
BIC:                   677.506   HQIC:       673.049
Converged:                True   Run Time:   0.1304s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.S.L12            -0.0003     0.6217    -0.000     1.000     -1.219      1.218
ma.S.L12            -0.1725     0.6428    -0.268     0.788     -1.432      1.087
sigma2             101.8132    17.8808     5.694     0.000     66.768    136.859
==============================================================================
Ljung-Box (L=24) (Q):      39.36   Jarque-Bera (JB):       2.07
Prob(Q):                   0.0250   Prob(JB):             0.3557
Heteroskedasticity(H):       0.65   Skew:                   0.30
Prob(H) (two-sided):       0.2401   Kurtosis:               2.64
Residual Mean:             0.3235   Residual Std:         9.7336
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 20.5308 | RMSE = 26.6198 | MAPE = 4.45% | R² = 0.8839
==============================================================================
```

### 3.9. SARIMA(0,1,1)x(0,1,0)_12
```text
SARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 1
Seasonal order (P, D, Q, s): (0, 1, 0, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(0,1,1)x(0,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -369.459   AIC:        742.918
BIC:                   748.128   HQIC:       745.026
Converged:                True   Run Time:   0.0375s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ma.L1               -0.1908     0.0923    -2.067     0.039     -0.372     -0.010
sigma2              94.7553    14.0540     6.742     0.000     67.210    122.301
==============================================================================
Ljung-Box (L=24) (Q):      35.78   Jarque-Bera (JB):       2.60
Prob(Q):                   0.0576   Prob(JB):             0.2721
Heteroskedasticity(H):       0.93   Skew:                   0.38
Prob(H) (two-sided):       0.8389   Kurtosis:               2.82
Residual Mean:             0.2454   Residual Std:         9.6974
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 29.2051 | RMSE = 35.7429 | MAPE = 6.27% | R² = 0.7907
==============================================================================
```

### 3.10. SARIMA(0,1,1)x(0,1,1)_12
```text
SARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 1
Seasonal order (P, D, Q, s): (0, 1, 1, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(0,1,1)x(0,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -326.671   AIC:        659.342
BIC:                   666.774   HQIC:       662.336
Converged:                True   Run Time:   0.0928s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ma.L1               -0.1672     0.1008    -1.659     0.097     -0.365      0.030
ma.S.L12            -0.1283     0.0983    -1.305     0.192     -0.321      0.064
sigma2              98.1421    16.7042     5.875     0.000     65.403    130.882
==============================================================================
Ljung-Box (L=24) (Q):      36.70   Jarque-Bera (JB):       2.94
Prob(Q):                   0.0468   Prob(JB):             0.2297
Heteroskedasticity(H):       0.75   Skew:                   0.37
Prob(H) (two-sided):       0.4478   Kurtosis:               2.63
Residual Mean:             0.3073   Residual Std:         9.6160
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 24.0592 | RMSE = 30.5800 | MAPE = 5.15% | R² = 0.8468
==============================================================================
```

### 3.11. SARIMA(0,1,1)x(1,1,0)_12
```text
SARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 1
Seasonal order (P, D, Q, s): (1, 1, 0, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(0,1,1)x(1,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -333.883   AIC:        673.765
BIC:                   681.265   HQIC:       676.789
Converged:                True   Run Time:   0.0648s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ma.L1               -0.1760     0.0995    -1.769     0.077     -0.371      0.019
ar.S.L12            -0.1429     0.0908    -1.573     0.116     -0.321      0.035
sigma2              97.6831    16.7209     5.842     0.000     64.911    130.455
==============================================================================
Ljung-Box (L=24) (Q):      36.40   Jarque-Bera (JB):       3.21
Prob(Q):                   0.0501   Prob(JB):             0.2009
Heteroskedasticity(H):       0.74   Skew:                   0.38
Prob(H) (two-sided):       0.4181   Kurtosis:               2.58
Residual Mean:             0.2770   Residual Std:         9.6347
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 24.1388 | RMSE = 30.6450 | MAPE = 5.17% | R² = 0.8462
==============================================================================
```

### 3.12. SARIMA(0,1,1)x(1,1,1)_12
```text
SARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 1
Seasonal order (P, D, Q, s): (1, 1, 1, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(0,1,1)x(1,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -325.193   AIC:        658.386
BIC:                   668.296   HQIC:       662.379
Converged:                True   Run Time:   0.1926s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ma.L1               -0.1997     0.0963    -2.073     0.038     -0.388     -0.011
ar.S.L12            -0.6588     0.1160    -5.681     0.000     -0.886     -0.432
ma.S.L12             0.6944     0.2447     2.838     0.005      0.215      1.174
sigma2              87.5648    22.0887     3.964     0.000     44.272    130.858
==============================================================================
Ljung-Box (L=24) (Q):      31.83   Jarque-Bera (JB):       0.02
Prob(Q):                   0.1314   Prob(JB):             0.9916
Heteroskedasticity(H):       0.88   Skew:                   0.00
Prob(H) (two-sided):       0.7307   Kurtosis:               2.94
Residual Mean:            -0.0658   Residual Std:        10.1842
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 30.4202 | RMSE = 36.9564 | MAPE = 6.53% | R² = 0.7763
==============================================================================
```

### 3.13. SARIMA(0,1,3)x(0,1,0)_12
```text
SARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 3
Seasonal order (P, D, Q, s): (0, 1, 0, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(0,1,3)x(0,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -359.454   AIC:        726.908
BIC:                   737.248   HQIC:       731.090
Converged:                True   Run Time:   0.0592s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ma.L1               -0.2262     0.0983    -2.302     0.021     -0.419     -0.034
ma.L2                0.1303     0.1116     1.168     0.243     -0.088      0.349
ma.L3               -0.3026     0.1073    -2.821     0.005     -0.513     -0.092
sigma2              89.8162    14.4001     6.237     0.000     61.593    118.040
==============================================================================
Ljung-Box (L=24) (Q):      24.39   Jarque-Bera (JB):       2.77
Prob(Q):                   0.4398   Prob(JB):             0.2497
Heteroskedasticity(H):       1.09   Skew:                   0.37
Prob(H) (two-sided):       0.8120   Kurtosis:               2.68
Residual Mean:             0.2773   Residual Std:         9.3551
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 33.1954 | RMSE = 39.9856 | MAPE = 7.13% | R² = 0.7381
==============================================================================
```

### 3.14. SARIMA(0,1,3)x(0,1,1)_12
```text
SARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 3
Seasonal order (P, D, Q, s): (0, 1, 1, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(0,1,3)x(0,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -315.540   AIC:        641.080
BIC:                   653.352   HQIC:       646.019
Converged:                True   Run Time:   0.1606s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ma.L1               -0.2178     0.1081    -2.014     0.044     -0.430     -0.006
ma.L2                0.0941     0.1286     0.732     0.464     -0.158      0.346
ma.L3               -0.2974     0.1180    -2.521     0.012     -0.529     -0.066
ma.S.L12            -0.1007     0.1024    -0.984     0.325     -0.301      0.100
sigma2              90.0382    15.3015     5.884     0.000     60.048    120.029
==============================================================================
Ljung-Box (L=24) (Q):      26.34   Jarque-Bera (JB):       3.00
Prob(Q):                   0.3362   Prob(JB):             0.2234
Heteroskedasticity(H):       0.88   Skew:                   0.37
Prob(H) (two-sided):       0.7332   Kurtosis:               2.62
Residual Mean:             0.3301   Residual Std:         9.2931
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 29.7464 | RMSE = 36.3925 | MAPE = 6.36% | R² = 0.7830
==============================================================================
```

### 3.15. SARIMA(0,1,3)x(1,1,0)_12
```text
SARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 3
Seasonal order (P, D, Q, s): (1, 1, 0, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(0,1,3)x(1,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -330.514   AIC:        671.027
BIC:                   683.526   HQIC:       676.068
Converged:                True   Run Time:   0.1102s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ma.L1               -0.2409     0.1048    -2.299     0.021     -0.446     -0.036
ma.L2                0.1458     0.1213     1.202     0.229     -0.092      0.384
ma.L3               -0.3068     0.1191    -2.575     0.010     -0.540     -0.073
ar.S.L12            -0.1047     0.0963    -1.088     0.277     -0.293      0.084
sigma2              90.4458    15.2026     5.949     0.000     60.649    120.242
==============================================================================
Ljung-Box (L=24) (Q):      26.78   Jarque-Bera (JB):       3.14
Prob(Q):                   0.3149   Prob(JB):             0.2081
Heteroskedasticity(H):       0.79   Skew:                   0.39
Prob(H) (two-sided):       0.5122   Kurtosis:               2.65
Residual Mean:             0.2101   Residual Std:         9.2875
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 29.6795 | RMSE = 36.3250 | MAPE = 6.35% | R² = 0.7838
==============================================================================
```

### 3.16. SARIMA(0,1,3)x(1,1,1)_12
```text
SARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 3
Seasonal order (P, D, Q, s): (1, 1, 1, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(0,1,3)x(1,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -314.111   AIC:        640.222
BIC:                   654.948   HQIC:       646.149
Converged:                True   Run Time:   0.3060s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ma.L1               -0.2370     0.1095    -2.165     0.030     -0.452     -0.022
ma.L2                0.1452     0.1237     1.174     0.240     -0.097      0.388
ma.L3               -0.3338     0.1239    -2.694     0.007     -0.577     -0.091
ar.S.L12            -0.6225     0.1551    -4.014     0.000     -0.926     -0.319
ma.S.L12             0.6362     0.2572     2.474     0.013      0.132      1.140
sigma2              82.2972    18.2918     4.499     0.000     46.446    118.148
==============================================================================
Ljung-Box (L=24) (Q):      30.18   Jarque-Bera (JB):       0.07
Prob(Q):                   0.1787   Prob(JB):             0.9655
Heteroskedasticity(H):       0.97   Skew:                   0.02
Prob(H) (two-sided):       0.9291   Kurtosis:               2.88
Residual Mean:            -0.0577   Residual Std:         9.7551
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 33.4712 | RMSE = 40.2743 | MAPE = 7.19% | R² = 0.7343
==============================================================================
```

### 3.17. SARIMA(1,1,0)x(0,1,0)_12
```text
SARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 0
Seasonal order (P, D, Q, s): (0, 1, 0, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(1,1,0)x(0,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -372.414   AIC:        748.828
BIC:                   754.058   HQIC:       750.946
Converged:                True   Run Time:   0.0161s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.2063     0.0908    -2.271     0.023     -0.384     -0.028
sigma2              93.3784    13.5423     6.895     0.000     66.836    119.921
==============================================================================
Ljung-Box (L=24) (Q):      35.22   Jarque-Bera (JB):       2.44
Prob(Q):                   0.0652   Prob(JB):             0.2955
Heteroskedasticity(H):       0.97   Skew:                   0.37
Prob(H) (two-sided):       0.9381   Kurtosis:               2.86
Residual Mean:             0.2457   Residual Std:         9.6729
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 28.5671 | RMSE = 35.1029 | MAPE = 6.13% | R² = 0.7981
==============================================================================
```

### 3.18. SARIMA(1,1,0)x(0,1,1)_12
```text
SARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 0
Seasonal order (P, D, Q, s): (0, 1, 1, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(1,1,0)x(0,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -330.374   AIC:        666.748
BIC:                   674.214   HQIC:       669.758
Converged:                True   Run Time:   0.0945s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.2004     0.1008    -1.988     0.047     -0.398     -0.003
ma.S.L12            -0.1168     0.0968    -1.207     0.227     -0.307      0.073
sigma2              98.1245    16.6805     5.883     0.000     65.431    130.818
==============================================================================
Ljung-Box (L=24) (Q):      34.83   Jarque-Bera (JB):       3.10
Prob(Q):                   0.0710   Prob(JB):             0.2119
Heteroskedasticity(H):       0.72   Skew:                   0.39
Prob(H) (two-sided):       0.3771   Kurtosis:               2.67
Residual Mean:             0.2203   Residual Std:         9.6011
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 24.2494 | RMSE = 30.7687 | MAPE = 5.19% | R² = 0.8449
==============================================================================
```

### 3.19. SARIMA(1,1,0)x(1,1,0)_12
```text
SARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 0
Seasonal order (P, D, Q, s): (1, 1, 0, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(1,1,0)x(1,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -330.255   AIC:        666.510
BIC:                   673.976   HQIC:       669.519
Converged:                True   Run Time:   0.0400s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.2017     0.1004    -2.009     0.045     -0.399     -0.005
ar.S.L12            -0.1381     0.0902    -1.531     0.126     -0.315      0.039
sigma2              97.8690    16.8139     5.821     0.000     64.914    130.824
==============================================================================
Ljung-Box (L=24) (Q):      35.05   Jarque-Bera (JB):       3.39
Prob(Q):                   0.0676   Prob(JB):             0.1838
Heteroskedasticity(H):       0.75   Skew:                   0.41
Prob(H) (two-sided):       0.4371   Kurtosis:               2.65
Residual Mean:             0.1526   Residual Std:         9.5935
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 23.8951 | RMSE = 30.4258 | MAPE = 5.12% | R² = 0.8484
==============================================================================
```

### 3.20. SARIMA(1,1,0)x(1,1,1)_12
```text
SARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 0
Seasonal order (P, D, Q, s): (1, 1, 1, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(1,1,0)x(1,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -328.967   AIC:        665.934
BIC:                   675.889   HQIC:       669.947
Converged:                True   Run Time:   0.1574s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.2182     0.1001    -2.179     0.029     -0.415     -0.022
ar.S.L12            -0.6563     0.1210    -5.424     0.000     -0.893     -0.419
ma.S.L12             0.6850     0.2468     2.775     0.006      0.201      1.169
sigma2              87.6912    21.9826     3.989     0.000     44.606    130.776
==============================================================================
Ljung-Box (L=24) (Q):      31.68   Jarque-Bera (JB):       0.36
Prob(Q):                   0.1351   Prob(JB):             0.8362
Heteroskedasticity(H):       0.83   Skew:                   0.09
Prob(H) (two-sided):       0.6207   Kurtosis:               2.77
Residual Mean:            -0.1083   Residual Std:        10.1137
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 29.5782 | RMSE = 36.1025 | MAPE = 6.35% | R² = 0.7865
==============================================================================
```

### 3.21. SARIMA(1,1,1)x(0,1,0)_12
```text
SARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 1
Seasonal order (P, D, Q, s): (0, 1, 0, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(1,1,1)x(0,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -368.468   AIC:        742.936
BIC:                   750.752   HQIC:       746.099
Converged:                True   Run Time:   0.0541s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.6432     0.3101    -2.074     0.038     -1.251     -0.035
ma.L1                0.4553     0.3478     1.309     0.190     -0.226      1.137
sigma2              92.8941    13.8471     6.709     0.000     65.754    120.034
==============================================================================
Ljung-Box (L=24) (Q):      32.43   Jarque-Bera (JB):       2.25
Prob(Q):                   0.1167   Prob(JB):             0.3244
Heteroskedasticity(H):       1.01   Skew:                   0.35
Prob(H) (two-sided):       0.9677   Kurtosis:               2.84
Residual Mean:             0.2410   Residual Std:         9.6026
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 27.5363 | RMSE = 34.0632 | MAPE = 5.91% | R² = 0.8099
==============================================================================
```

### 3.22. SARIMA(1,1,1)x(0,1,1)_12
```text
SARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 1
Seasonal order (P, D, Q, s): (0, 1, 1, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(1,1,1)x(0,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -325.666   AIC:        659.333
BIC:                   669.242   HQIC:       663.325
Converged:                True   Run Time:   0.1505s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.6512     0.3392    -1.920     0.055     -1.316      0.014
ma.L1                0.4706     0.3810     1.235     0.217     -0.276      1.217
ma.S.L12            -0.1139     0.0945    -1.206     0.228     -0.299      0.071
sigma2              95.9296    16.2783     5.893     0.000     64.025    127.834
==============================================================================
Ljung-Box (L=24) (Q):      25.28   Jarque-Bera (JB):       0.45
Prob(Q):                   0.3909   Prob(JB):             0.7991
Heteroskedasticity(H):       0.76   Skew:                   0.16
Prob(H) (two-sided):       0.4643   Kurtosis:               2.95
Residual Mean:            -0.0064   Residual Std:         9.9496
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 23.3679 | RMSE = 29.9231 | MAPE = 5.01% | R² = 0.8533
==============================================================================
```

### 3.23. SARIMA(1,1,1)x(1,1,0)_12
```text
SARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 1
Seasonal order (P, D, Q, s): (1, 1, 0, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(1,1,1)x(1,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -329.452   AIC:        666.904
BIC:                   676.858   HQIC:       670.916
Converged:                True   Run Time:   0.0999s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.8349     0.1117    -7.471     0.000     -1.054     -0.616
ma.L1                0.6979     0.1503     4.645     0.000      0.403      0.992
ar.S.L12            -0.1561     0.0862    -1.811     0.070     -0.325      0.013
sigma2              95.2745    16.3556     5.825     0.000     63.218    127.331
==============================================================================
Ljung-Box (L=24) (Q):      22.54   Jarque-Bera (JB):       4.89
Prob(Q):                   0.5469   Prob(JB):             0.0869
Heteroskedasticity(H):       0.81   Skew:                  -0.19
Prob(H) (two-sided):       0.5639   Kurtosis:               4.00
Residual Mean:            -0.1005   Residual Std:        10.3399
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 22.1123 | RMSE = 28.6114 | MAPE = 4.75% | R² = 0.8659
==============================================================================
```

### 3.24. SARIMA(1,1,1)x(1,1,1)_12
```text
SARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 1
Seasonal order (P, D, Q, s): (1, 1, 1, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(1,1,1)x(1,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -323.801   AIC:        657.601
BIC:                   669.988   HQIC:       662.592
Converged:                True   Run Time:   0.4350s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.8224     0.1843    -4.461     0.000     -1.184     -0.461
ma.L1                0.7176     0.2240     3.204     0.001      0.279      1.157
ar.S.L12            -0.5040     0.1692    -2.979     0.003     -0.836     -0.172
ma.S.L12             0.4616     0.2465     1.873     0.061     -0.021      0.945
sigma2              88.7360    18.8656     4.704     0.000     51.760    125.712
==============================================================================
Ljung-Box (L=24) (Q):      24.17   Jarque-Bera (JB):       1.47
Prob(Q):                   0.4519   Prob(JB):             0.4797
Heteroskedasticity(H):       0.89   Skew:                  -0.12
Prob(H) (two-sided):       0.7603   Kurtosis:               3.53
Residual Mean:            -0.2254   Residual Std:        10.2847
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 25.2680 | RMSE = 31.7873 | MAPE = 5.43% | R² = 0.8345
==============================================================================
```

### 3.25. SARIMA(1,1,3)x(0,1,0)_12
```text
SARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 3
Seasonal order (P, D, Q, s): (0, 1, 0, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(1,1,3)x(0,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -359.064   AIC:        728.127
BIC:                   741.052   HQIC:       733.355
Converged:                True   Run Time:   0.0986s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1                0.3259     0.2533     1.287     0.198     -0.170      0.822
ma.L1               -0.5246     0.2551    -2.057     0.040     -1.024     -0.025
ma.L2                0.1711     0.1319     1.297     0.195     -0.087      0.430
ma.L3               -0.3398     0.1124    -3.022     0.003     -0.560     -0.119
sigma2              89.0647    14.3242     6.218     0.000     60.990    117.140
==============================================================================
Ljung-Box (L=24) (Q):      23.42   Jarque-Bera (JB):       2.17
Prob(Q):                   0.4953   Prob(JB):             0.3373
Heteroskedasticity(H):       0.97   Skew:                   0.33
Prob(H) (two-sided):       0.9384   Kurtosis:               2.71
Residual Mean:             0.4052   Residual Std:         9.3119
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 29.4898 | RMSE = 36.0508 | MAPE = 6.34% | R² = 0.7871
==============================================================================
```

### 3.26. SARIMA(1,1,3)x(0,1,1)_12
```text
SARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 3
Seasonal order (P, D, Q, s): (0, 1, 1, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(1,1,3)x(0,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -313.810   AIC:        639.621
BIC:                   654.347   HQIC:       645.547
Converged:                True   Run Time:   0.3547s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1                0.6353     0.1623     3.914     0.000      0.317      0.953
ma.L1               -0.8157     0.1856    -4.396     0.000     -1.179     -0.452
ma.L2                0.1650     0.1705     0.968     0.333     -0.169      0.499
ma.L3               -0.2988     0.1424    -2.099     0.036     -0.578     -0.020
ma.S.L12            -0.1467     0.1068    -1.374     0.170     -0.356      0.063
sigma2              85.9030    13.3072     6.455     0.000     59.821    111.985
==============================================================================
Ljung-Box (L=24) (Q):      24.91   Jarque-Bera (JB):       8.97
Prob(Q):                   0.4105   Prob(JB):             0.0113
Heteroskedasticity(H):       0.75   Skew:                   0.61
Prob(H) (two-sided):       0.4447   Kurtosis:               3.77
Residual Mean:             1.3471   Residual Std:         9.7504
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 19.4272 | RMSE = 24.8549 | MAPE = 4.31% | R² = 0.8988
==============================================================================
```

### 3.27. SARIMA(1,1,3)x(1,1,0)_12
```text
SARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 3
Seasonal order (P, D, Q, s): (1, 1, 0, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(1,1,3)x(1,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -325.796   AIC:        663.591
BIC:                   678.523   HQIC:       669.610
Converged:                True   Run Time:   0.2314s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1                0.6479     0.1591     4.072     0.000      0.336      0.960
ma.L1               -0.8572     0.1741    -4.923     0.000     -1.198     -0.516
ma.L2                0.2174     0.1635     1.329     0.184     -0.103      0.538
ma.L3               -0.3297     0.1397    -2.361     0.018     -0.603     -0.056
ar.S.L12            -0.1506     0.1048    -1.437     0.151     -0.356      0.055
sigma2              87.1588    13.4820     6.465     0.000     60.735    113.583
==============================================================================
Ljung-Box (L=24) (Q):      23.73   Jarque-Bera (JB):      11.58
Prob(Q):                   0.4773   Prob(JB):             0.0031
Heteroskedasticity(H):       0.68   Skew:                   0.65
Prob(H) (two-sided):       0.2874   Kurtosis:               4.01
Residual Mean:             1.1466   Residual Std:         9.8260
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 19.5157 | RMSE = 25.0879 | MAPE = 4.32% | R² = 0.8969
==============================================================================
```

### 3.28. SARIMA(1,1,3)x(1,1,1)_12
```text
SARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 3
Seasonal order (P, D, Q, s): (1, 1, 1, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(1,1,3)x(1,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -313.810   AIC:        641.620
BIC:                   658.801   HQIC:       648.534
Converged:                True   Run Time:   0.5736s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1                0.6353     0.1642     3.869     0.000      0.313      0.957
ma.L1               -0.8156     0.1856    -4.395     0.000     -1.179     -0.452
ma.L2                0.1650     0.1708     0.966     0.334     -0.170      0.500
ma.L3               -0.2988     0.1434    -2.084     0.037     -0.580     -0.018
ar.S.L12            -0.0004     0.1429    -0.003     0.998     -0.280      0.280
ma.S.L12            -0.1464     0.1936    -0.756     0.450     -0.526      0.233
sigma2              85.9025    14.0681     6.106     0.000     58.330    113.475
==============================================================================
Ljung-Box (L=24) (Q):      24.91   Jarque-Bera (JB):       8.98
Prob(Q):                   0.4107   Prob(JB):             0.0112
Heteroskedasticity(H):       0.75   Skew:                   0.61
Prob(H) (two-sided):       0.4450   Kurtosis:               3.77
Residual Mean:             1.3459   Residual Std:         9.7503
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 19.4265 | RMSE = 24.8540 | MAPE = 4.31% | R² = 0.8988
==============================================================================
```

### 3.29. Log-SARIMA(0,1,0)x(0,1,1)_12
```text
Log-SARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 0
Seasonal order (P, D, Q, s): (0, 1, 1, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(0,1,0)x(0,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        161.284   AIC:       -318.569
BIC:                  -313.592   HQIC:      -316.563
Converged:                True   Run Time:   0.1239s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ma.S.L12            -0.6185     0.1055    -5.864     0.000     -0.825     -0.412
sigma2               0.0015     0.0002     7.777     0.000      0.001      0.002
==============================================================================
Ljung-Box (L=24) (Q):      30.06   Jarque-Bera (JB):       2.23
Prob(Q):                   0.1826   Prob(JB):             0.3277
Heteroskedasticity(H):       0.17   Skew:                   0.03
Prob(H) (two-sided):       0.0000   Kurtosis:               3.72
Residual Mean:             0.0011   Residual Std:         0.0402
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 12.6699 | RMSE = 15.4465 | MAPE = 3.00% | R² = 0.9609
==============================================================================
```

### 3.30. Log-SARIMA(0,1,0)x(1,1,0)_12
```text
Log-SARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 0
Seasonal order (P, D, Q, s): (1, 1, 0, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(0,1,0)x(1,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        158.252   AIC:       -312.505
BIC:                  -307.505   HQIC:      -310.489
Converged:                True   Run Time:   0.1164s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.S.L12            -0.4403     0.0738    -5.967     0.000     -0.585     -0.296
sigma2               0.0017     0.0002     7.381     0.000      0.001      0.002
==============================================================================
Ljung-Box (L=24) (Q):      44.78   Jarque-Bera (JB):      31.25
Prob(Q):                   0.0062   Prob(JB):             0.0000
Heteroskedasticity(H):       0.18   Skew:                  -0.52
Prob(H) (two-sided):       0.0000   Kurtosis:               5.50
Residual Mean:            -0.0018   Residual Std:         0.0481
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 18.7312 | RMSE = 22.9400 | MAPE = 4.12% | R² = 0.9138
==============================================================================
```

### 3.31. Log-SARIMA(0,1,0)x(1,1,1)_12
```text
Log-SARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 0
Seasonal order (P, D, Q, s): (1, 1, 1, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(0,1,0)x(1,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        158.296   AIC:       -310.593
BIC:                  -303.127   HQIC:      -307.584
Converged:                True   Run Time:   0.1693s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.S.L12            -0.1604     0.1538    -1.043     0.297     -0.462      0.141
ma.S.L12            -0.4834     0.1918    -2.520     0.012     -0.859     -0.107
sigma2               0.0016     0.0002     7.701     0.000      0.001      0.002
==============================================================================
Ljung-Box (L=24) (Q):      37.92   Jarque-Bera (JB):     230.02
Prob(Q):                   0.0353   Prob(JB):             0.0000
Heteroskedasticity(H):       0.16   Skew:                  -1.35
Prob(H) (two-sided):       0.0000   Kurtosis:               9.84
Residual Mean:            -0.0034   Residual Std:         0.0496
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 12.0596 | RMSE = 14.6550 | MAPE = 2.80% | R² = 0.9648
==============================================================================
```

### 3.32. Log-SARIMA(0,1,1)x(0,1,0)_12
```text
Log-SARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 1
Seasonal order (P, D, Q, s): (0, 1, 0, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(0,1,1)x(0,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        170.744   AIC:       -337.488
BIC:                  -332.277   HQIC:      -335.379
Converged:                True   Run Time:   0.0577s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ma.L1               -0.3640     0.0827    -4.404     0.000     -0.526     -0.202
sigma2               0.0019     0.0002     7.969     0.000      0.001      0.002
==============================================================================
Ljung-Box (L=24) (Q):      45.26   Jarque-Bera (JB):       3.85
Prob(Q):                   0.0054   Prob(JB):             0.1460
Heteroskedasticity(H):       0.17   Skew:                   0.33
Prob(H) (two-sided):       0.0000   Kurtosis:               3.69
Residual Mean:             0.0002   Residual Std:         0.0439
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 37.3894 | RMSE = 43.7381 | MAPE = 8.14% | R² = 0.6866
==============================================================================
```

### 3.33. Log-SARIMA(0,1,1)x(0,1,1)_12
```text
Log-SARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 1
Seasonal order (P, D, Q, s): (0, 1, 1, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(0,1,1)x(0,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        164.875   AIC:       -323.749
BIC:                  -316.317   HQIC:      -320.755
Converged:                True   Run Time:   0.3027s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ma.L1               -0.3528     0.0959    -3.678     0.000     -0.541     -0.165
ma.S.L12            -0.5583     0.1090    -5.122     0.000     -0.772     -0.345
sigma2               0.0014     0.0002     7.630     0.000      0.001      0.002
==============================================================================
Ljung-Box (L=24) (Q):      18.84   Jarque-Bera (JB):       1.45
Prob(Q):                   0.7605   Prob(JB):             0.4849
Heteroskedasticity(H):       0.25   Skew:                   0.20
Prob(H) (two-sided):       0.0004   Kurtosis:               3.43
Residual Mean:             0.0008   Residual Std:         0.0385
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 12.9506 | RMSE = 16.1554 | MAPE = 2.89% | R² = 0.9572
==============================================================================
```

### 3.34. Log-SARIMA(0,1,1)x(1,1,0)_12
```text
Log-SARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 1
Seasonal order (P, D, Q, s): (1, 1, 0, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(0,1,1)x(1,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        165.656   AIC:       -325.312
BIC:                  -317.813   HQIC:      -322.288
Converged:                True   Run Time:   0.1740s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ma.L1               -0.4208     0.0942    -4.469     0.000     -0.605     -0.236
ar.S.L12            -0.4554     0.0829    -5.496     0.000     -0.618     -0.293
sigma2               0.0015     0.0002     7.249     0.000      0.001      0.002
==============================================================================
Ljung-Box (L=24) (Q):      25.81   Jarque-Bera (JB):     114.79
Prob(Q):                   0.3629   Prob(JB):             0.0000
Heteroskedasticity(H):       0.27   Skew:                  -0.83
Prob(H) (two-sided):       0.0005   Kurtosis:               7.92
Residual Mean:            -0.0032   Residual Std:         0.0470
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 23.5392 | RMSE = 29.1286 | MAPE = 5.08% | R² = 0.8610
==============================================================================
```

### 3.35. Log-SARIMA(0,1,1)x(1,1,1)_12
```text
Log-SARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 1
Seasonal order (P, D, Q, s): (1, 1, 1, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(0,1,1)x(1,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        162.500   AIC:       -317.000
BIC:                  -307.090   HQIC:      -313.007
Converged:                True   Run Time:   0.3220s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ma.L1               -0.3965     0.0937    -4.231     0.000     -0.580     -0.213
ar.S.L12            -0.3448     0.1766    -1.953     0.051     -0.691      0.001
ma.S.L12            -0.2041     0.2101    -0.972     0.331     -0.616      0.208
sigma2               0.0014     0.0002     7.350     0.000      0.001      0.002
==============================================================================
Ljung-Box (L=24) (Q):      12.19   Jarque-Bera (JB):   12849.48
Prob(Q):                   0.9777   Prob(JB):             0.0000
Heteroskedasticity(H):       0.25   Skew:                  -6.36
Prob(H) (two-sided):       0.0003   Kurtosis:              56.50
Residual Mean:            -0.0081   Residual Std:         0.0802
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 18.8799 | RMSE = 22.9472 | MAPE = 4.11% | R² = 0.9137
==============================================================================
```

### 3.36. Log-SARIMA(0,1,3)x(0,1,0)_12
```text
Log-SARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 3
Seasonal order (P, D, Q, s): (0, 1, 0, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(0,1,3)x(0,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        168.609   AIC:       -329.218
BIC:                  -318.878   HQIC:      -325.036
Converged:                True   Run Time:   0.1805s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ma.L1               -0.3160     0.0842    -3.753     0.000     -0.481     -0.151
ma.L2                0.0633     0.1066     0.594     0.553     -0.146      0.272
ma.L3               -0.2275     0.1203    -1.891     0.059     -0.463      0.008
sigma2               0.0019     0.0002     7.534     0.000      0.001      0.002
==============================================================================
Ljung-Box (L=24) (Q):      35.22   Jarque-Bera (JB):       1.88
Prob(Q):                   0.0651   Prob(JB):             0.3897
Heteroskedasticity(H):       0.18   Skew:                   0.19
Prob(H) (two-sided):       0.0000   Kurtosis:               3.54
Residual Mean:             0.0003   Residual Std:         0.0429
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 36.5048 | RMSE = 42.7426 | MAPE = 7.96% | R² = 0.7007
==============================================================================
```

### 3.37. Log-SARIMA(0,1,3)x(0,1,1)_12
```text
Log-SARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 3
Seasonal order (P, D, Q, s): (0, 1, 1, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(0,1,3)x(0,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        167.027   AIC:       -324.054
BIC:                  -311.782   HQIC:      -319.115
Converged:                True   Run Time:   0.4823s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ma.L1               -0.3432     0.1030    -3.331     0.001     -0.545     -0.141
ma.L2                0.0304     0.1244     0.244     0.807     -0.213      0.274
ma.L3               -0.2122     0.1312    -1.618     0.106     -0.469      0.045
ma.S.L12            -0.5572     0.1398    -3.985     0.000     -0.831     -0.283
sigma2               0.0012     0.0002     7.132     0.000      0.001      0.002
==============================================================================
Ljung-Box (L=24) (Q):      14.99   Jarque-Bera (JB):       3.43
Prob(Q):                   0.9210   Prob(JB):             0.1801
Heteroskedasticity(H):       0.32   Skew:                   0.09
Prob(H) (two-sided):       0.0033   Kurtosis:               3.88
Residual Mean:             0.0008   Residual Std:         0.0380
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 13.0714 | RMSE = 16.2571 | MAPE = 2.92% | R² = 0.9567
==============================================================================
```

### 3.38. Log-SARIMA(0,1,3)x(1,1,0)_12
```text
Log-SARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 3
Seasonal order (P, D, Q, s): (1, 1, 0, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(0,1,3)x(1,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        167.632   AIC:       -325.264
BIC:                  -312.765   HQIC:      -320.224
Converged:                True   Run Time:   0.2489s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ma.L1               -0.4031     0.1039    -3.878     0.000     -0.607     -0.199
ma.L2                0.1298     0.1192     1.089     0.276     -0.104      0.363
ma.L3               -0.2356     0.1271    -1.854     0.064     -0.485      0.013
ar.S.L12            -0.4379     0.0957    -4.575     0.000     -0.626     -0.250
sigma2               0.0014     0.0002     7.035     0.000      0.001      0.002
==============================================================================
Ljung-Box (L=24) (Q):      35.84   Jarque-Bera (JB):      47.35
Prob(Q):                   0.0568   Prob(JB):             0.0000
Heteroskedasticity(H):       0.26   Skew:                  -0.59
Prob(H) (two-sided):       0.0004   Kurtosis:               6.12
Residual Mean:            -0.0033   Residual Std:         0.0447
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 22.6677 | RMSE = 28.0558 | MAPE = 4.91% | R² = 0.8711
==============================================================================
```

### 3.39. Log-SARIMA(0,1,3)x(1,1,1)_12
```text
Log-SARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 3
Seasonal order (P, D, Q, s): (1, 1, 1, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(0,1,3)x(1,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        163.849   AIC:       -315.697
BIC:                  -300.971   HQIC:      -309.771
Converged:                True   Run Time:   0.5937s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ma.L1               -0.3575     0.0971    -3.680     0.000     -0.548     -0.167
ma.L2                0.0402     0.1166     0.345     0.730     -0.188      0.269
ma.L3               -0.1657     0.1299    -1.275     0.202     -0.420      0.089
ar.S.L12            -0.3373     0.1567    -2.152     0.031     -0.644     -0.030
ma.S.L12            -0.1815     0.2009    -0.903     0.366     -0.575      0.212
sigma2               0.0013     0.0002     6.673     0.000      0.001      0.002
==============================================================================
Ljung-Box (L=24) (Q):      14.48   Jarque-Bera (JB):   12521.98
Prob(Q):                   0.9350   Prob(JB):             0.0000
Heteroskedasticity(H):       0.30   Skew:                  -6.31
Prob(H) (two-sided):       0.0020   Kurtosis:              55.79
Residual Mean:            -0.0084   Residual Std:         0.0782
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 19.4650 | RMSE = 23.6646 | MAPE = 4.24% | R² = 0.9083
==============================================================================
```

### 3.40. Log-SARIMA(1,1,0)x(0,1,0)_12
```text
Log-SARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 0
Seasonal order (P, D, Q, s): (0, 1, 0, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(1,1,0)x(0,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        172.618   AIC:       -341.236
BIC:                  -336.005   HQIC:      -339.118
Converged:               False   Run Time:   0.0442s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.3216     0.0731    -4.402     0.000     -0.465     -0.178
sigma2               0.0019     0.0002     8.600     0.000      0.001      0.002
==============================================================================
Ljung-Box (L=24) (Q):      49.32   Jarque-Bera (JB):       4.62
Prob(Q):                   0.0017   Prob(JB):             0.0994
Heteroskedasticity(H):       0.16   Skew:                   0.25
Prob(H) (two-sided):       0.0000   Kurtosis:               3.91
Residual Mean:             0.0003   Residual Std:         0.0440
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 34.7076 | RMSE = 40.6761 | MAPE = 7.56% | R² = 0.7290
==============================================================================
Warnings:
  [1] Maximum Likelihood optimization failed to converge. Check mle_retvals
==============================================================================
```

### 3.41. Log-SARIMA(1,1,0)x(0,1,1)_12
```text
Log-SARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 0
Seasonal order (P, D, Q, s): (0, 1, 1, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(1,1,0)x(0,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        166.733   AIC:       -327.466
BIC:                  -320.000   HQIC:      -324.457
Converged:                True   Run Time:   0.1926s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.3459     0.0828    -4.178     0.000     -0.508     -0.184
ma.S.L12            -0.5630     0.1051    -5.354     0.000     -0.769     -0.357
sigma2               0.0014     0.0002     7.206     0.000      0.001      0.002
==============================================================================
Ljung-Box (L=24) (Q):       9.24   Jarque-Bera (JB):   11315.55
Prob(Q):                   0.9970   Prob(JB):             0.0000
Heteroskedasticity(H):       0.22   Skew:                  -6.07
Prob(H) (two-sided):       0.0001   Kurtosis:              53.15
Residual Mean:            -0.0055   Residual Std:         0.0737
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 12.3454 | RMSE = 15.1181 | MAPE = 2.81% | R² = 0.9626
==============================================================================
```

### 3.42. Log-SARIMA(1,1,0)x(1,1,0)_12
```text
Log-SARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 0
Seasonal order (P, D, Q, s): (1, 1, 0, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(1,1,0)x(1,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        163.825   AIC:       -321.650
BIC:                  -314.184   HQIC:      -318.641
Converged:                True   Run Time:   0.2045s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.3895     0.0837    -4.655     0.000     -0.554     -0.226
ar.S.L12            -0.4444     0.0759    -5.853     0.000     -0.593     -0.296
sigma2               0.0015     0.0002     7.011     0.000      0.001      0.002
==============================================================================
Ljung-Box (L=24) (Q):      35.26   Jarque-Bera (JB):    4490.39
Prob(Q):                   0.0647   Prob(JB):             0.0000
Heteroskedasticity(H):       0.22   Skew:                  -5.02
Prob(H) (two-sided):       0.0001   Kurtosis:              33.92
Residual Mean:            -0.0126   Residual Std:         0.1038
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 21.3432 | RMSE = 26.2526 | MAPE = 4.64% | R² = 0.8871
==============================================================================
```

### 3.43. Log-SARIMA(1,1,0)x(1,1,1)_12
```text
Log-SARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 0
Seasonal order (P, D, Q, s): (1, 1, 1, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(1,1,0)x(1,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        164.032   AIC:       -320.064
BIC:                  -310.109   HQIC:      -316.051
Converged:               False   Run Time:   0.4146s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.3708     0.0808    -4.587     0.000     -0.529     -0.212
ar.S.L12            -0.3527     0.1620    -2.178     0.029     -0.670     -0.035
ma.S.L12            -0.1774     0.2015    -0.880     0.379     -0.572      0.218
sigma2               0.0015     0.0002     7.022     0.000      0.001      0.002
==============================================================================
Ljung-Box (L=24) (Q):      29.04   Jarque-Bera (JB):    5767.16
Prob(Q):                   0.2187   Prob(JB):             0.0000
Heteroskedasticity(H):       0.21   Skew:                  -5.09
Prob(H) (two-sided):       0.0001   Kurtosis:              38.40
Residual Mean:            -0.0106   Residual Std:         0.0890
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 17.6965 | RMSE = 21.5139 | MAPE = 3.88% | R² = 0.9242
==============================================================================
Warnings:
  [1] Maximum Likelihood optimization failed to converge. Check mle_retvals
==============================================================================
```

### 3.44. Log-SARIMA(1,1,1)x(0,1,0)_12
```text
Log-SARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 1
Seasonal order (P, D, Q, s): (0, 1, 0, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(1,1,1)x(0,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        170.455   AIC:       -334.910
BIC:                  -327.094   HQIC:      -331.747
Converged:                True   Run Time:   0.0353s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.3241     0.3252    -0.997     0.319     -0.962      0.313
ma.L1                0.0003     0.3653     0.001     0.999     -0.716      0.716
sigma2               0.0019     0.0002     8.411     0.000      0.001      0.002
==============================================================================
Ljung-Box (L=24) (Q):      49.36   Jarque-Bera (JB):       4.62
Prob(Q):                   0.0017   Prob(JB):             0.0993
Heteroskedasticity(H):       0.16   Skew:                   0.25
Prob(H) (two-sided):       0.0000   Kurtosis:               3.91
Residual Mean:             0.0003   Residual Std:         0.0440
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 34.7275 | RMSE = 40.6987 | MAPE = 7.57% | R² = 0.7287
==============================================================================
```

### 3.45. Log-SARIMA(1,1,1)x(0,1,1)_12
```text
Log-SARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 1
Seasonal order (P, D, Q, s): (0, 1, 1, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(1,1,1)x(0,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        165.253   AIC:       -322.506
BIC:                  -312.597   HQIC:      -318.514
Converged:                True   Run Time:   0.4721s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.5330     0.2933    -1.818     0.069     -1.108      0.042
ma.L1                0.2344     0.3579     0.655     0.512     -0.467      0.936
ma.S.L12            -0.5736     0.1066    -5.383     0.000     -0.783     -0.365
sigma2               0.0013     0.0002     6.880     0.000      0.001      0.002
==============================================================================
Ljung-Box (L=24) (Q):       4.81   Jarque-Bera (JB):   24051.98
Prob(Q):                   1.0000   Prob(JB):             0.0000
Heteroskedasticity(H):       0.22   Skew:                  -8.09
Prob(H) (two-sided):       0.0001   Kurtosis:              76.47
Residual Mean:            -0.0093   Residual Std:         0.1085
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 12.1328 | RMSE = 14.8544 | MAPE = 2.79% | R² = 0.9639
==============================================================================
```

### 3.46. Log-SARIMA(1,1,1)x(1,1,0)_12
```text
Log-SARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 1
Seasonal order (P, D, Q, s): (1, 1, 0, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(1,1,1)x(1,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        163.827   AIC:       -319.654
BIC:                  -309.700   HQIC:      -315.642
Converged:                True   Run Time:   0.1297s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.3615     0.2607    -1.387     0.165     -0.872      0.149
ma.L1               -0.0328     0.3032    -0.108     0.914     -0.627      0.561
ar.S.L12            -0.4432     0.0771    -5.745     0.000     -0.594     -0.292
sigma2               0.0015     0.0002     6.949     0.000      0.001      0.002
==============================================================================
Ljung-Box (L=24) (Q):      36.09   Jarque-Bera (JB):    4070.17
Prob(Q):                   0.0538   Prob(JB):             0.0000
Heteroskedasticity(H):       0.22   Skew:                  -4.89
Prob(H) (two-sided):       0.0001   Kurtosis:              32.36
Residual Mean:            -0.0124   Residual Std:         0.1018
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 21.5015 | RMSE = 26.4595 | MAPE = 4.67% | R² = 0.8853
==============================================================================
```

### 3.47. Log-SARIMA(1,1,1)x(1,1,1)_12
```text
Log-SARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 1
Seasonal order (P, D, Q, s): (1, 1, 1, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(1,1,1)x(1,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        162.544   AIC:       -315.088
BIC:                  -302.702   HQIC:      -310.098
Converged:                True   Run Time:   0.5592s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1                0.1307     0.2665     0.491     0.624     -0.392      0.653
ma.L1               -0.5150     0.2445    -2.106     0.035     -0.994     -0.036
ar.S.L12            -0.3268     0.1794    -1.822     0.068     -0.678      0.025
ma.S.L12            -0.2340     0.2143    -1.092     0.275     -0.654      0.186
sigma2               0.0014     0.0002     7.082     0.000      0.001      0.002
==============================================================================
Ljung-Box (L=24) (Q):      18.11   Jarque-Bera (JB):    6884.52
Prob(Q):                   0.7975   Prob(JB):             0.0000
Heteroskedasticity(H):       0.26   Skew:                  -4.38
Prob(H) (two-sided):       0.0005   Kurtosis:              42.28
Residual Mean:            -0.0060   Residual Std:         0.0816
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 18.6625 | RMSE = 22.6612 | MAPE = 4.06% | R² = 0.9159
==============================================================================
```

### 3.48. Log-SARIMA(1,1,3)x(0,1,0)_12
```text
Log-SARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 3
Seasonal order (P, D, Q, s): (0, 1, 0, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(1,1,3)x(0,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        168.808   AIC:       -327.616
BIC:                  -314.691   HQIC:      -322.388
Converged:                True   Run Time:   0.1102s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.2325     0.4544    -0.512     0.609     -1.123      0.658
ma.L1               -0.1006     0.4542    -0.222     0.825     -0.991      0.790
ma.L2               -0.0070     0.2073    -0.034     0.973     -0.413      0.399
ma.L3               -0.2208     0.1243    -1.776     0.076     -0.464      0.023
sigma2               0.0019     0.0003     7.434     0.000      0.001      0.002
==============================================================================
Ljung-Box (L=24) (Q):      33.58   Jarque-Bera (JB):       1.41
Prob(Q):                   0.0924   Prob(JB):             0.4939
Heteroskedasticity(H):       0.19   Skew:                   0.17
Prob(H) (two-sided):       0.0000   Kurtosis:               3.46
Residual Mean:             0.0002   Residual Std:         0.0428
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 37.7755 | RMSE = 44.2038 | MAPE = 8.22% | R² = 0.6799
==============================================================================
```

### 3.49. Log-SARIMA(1,1,3)x(0,1,1)_12
```text
Log-SARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 3
Seasonal order (P, D, Q, s): (0, 1, 1, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(1,1,3)x(0,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        167.450   AIC:       -322.901
BIC:                  -308.175   HQIC:      -316.974
Converged:               False   Run Time:   0.7137s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1                0.3255     0.3494     0.931     0.352     -0.359      1.010
ma.L1               -0.6435     0.3586    -1.794     0.073     -1.346      0.059
ma.L2                0.1373     0.1682     0.816     0.414     -0.192      0.467
ma.L3               -0.2163     0.1465    -1.476     0.140     -0.504      0.071
ma.S.L12            -0.5837     0.1423    -4.102     0.000     -0.863     -0.305
sigma2               0.0012     0.0002     7.050     0.000      0.001      0.001
==============================================================================
Ljung-Box (L=24) (Q):       9.77   Jarque-Bera (JB):   12970.63
Prob(Q):                   0.9954   Prob(JB):             0.0000
Heteroskedasticity(H):       0.31   Skew:                   6.43
Prob(H) (two-sided):       0.0026   Kurtosis:              56.73
Residual Mean:             0.0065   Residual Std:         0.0762
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 12.0237 | RMSE = 14.6704 | MAPE = 2.79% | R² = 0.9647
==============================================================================
Warnings:
  [1] Maximum Likelihood optimization failed to converge. Check mle_retvals
==============================================================================
```

### 3.50. Log-SARIMA(1,1,3)x(1,1,0)_12
```text
Log-SARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 3
Seasonal order (P, D, Q, s): (1, 1, 0, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(1,1,3)x(1,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        165.852   AIC:       -319.704
BIC:                  -304.773   HQIC:      -313.686
Converged:                True   Run Time:   0.3640s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.0417     0.4862    -0.086     0.932     -0.995      0.911
ma.L1               -0.3655     0.4937    -0.740     0.459     -1.333      0.602
ma.L2                0.1140     0.2320     0.492     0.623     -0.341      0.569
ma.L3               -0.2325     0.1326    -1.754     0.079     -0.492      0.027
ar.S.L12            -0.4305     0.0966    -4.456     0.000     -0.620     -0.241
sigma2               0.0014     0.0002     6.976     0.000      0.001      0.002
==============================================================================
Ljung-Box (L=24) (Q):       9.85   Jarque-Bera (JB):   20277.27
Prob(Q):                   0.9951   Prob(JB):             0.0000
Heteroskedasticity(H):       0.27   Skew:                  -7.57
Prob(H) (two-sided):       0.0005   Kurtosis:              70.39
Residual Mean:            -0.0111   Residual Std:         0.1000
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 23.0962 | RMSE = 28.5426 | MAPE = 5.00% | R² = 0.8665
==============================================================================
```

### 3.51. Log-SARIMA(1,1,3)x(1,1,1)_12
```text
Log-SARIMA Model Summary - AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
==============================================================================
Run ID: dataset2_20261112_082858_8425b2 | Status: SUCCESS
Data: AirPassengers (Monthly Airline Passengers, 01/1949 - 12/1960)
Date range: 1949-01-01 to 1960-12-01 (Freq: MS)
Number of observations: 144 (Train: 115, Test: 29)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 3
Seasonal order (P, D, Q, s): (1, 1, 1, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(1,1,3)x(1,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        163.863   AIC:       -313.725
BIC:                  -296.545   HQIC:      -306.811
Converged:               False   Run Time:   0.7467s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1                0.1076     0.6065     0.177     0.859     -1.081      1.296
ma.L1               -0.4524     0.6190    -0.731     0.465     -1.666      0.761
ma.L2                0.0832     0.2542     0.327     0.743     -0.415      0.581
ma.L3               -0.1756     0.1305    -1.345     0.179     -0.431      0.080
ar.S.L12            -0.3460     0.1529    -2.264     0.024     -0.646     -0.046
ma.S.L12            -0.1879     0.1962    -0.958     0.338     -0.572      0.197
sigma2               0.0013     0.0002     6.623     0.000      0.001      0.002
==============================================================================
Ljung-Box (L=24) (Q):      16.99   Jarque-Bera (JB):    9530.92
Prob(Q):                   0.8490   Prob(JB):             0.0000
Heteroskedasticity(H):       0.30   Skew:                  -5.36
Prob(H) (two-sided):       0.0018   Kurtosis:              49.13
Residual Mean:            -0.0069   Residual Std:         0.0839
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=29):
  MAE = 18.5499 | RMSE = 22.5274 | MAPE = 4.05% | R² = 0.9169
==============================================================================
Warnings:
  [1] Maximum Likelihood optimization failed to converge. Check mle_retvals
==============================================================================
```
