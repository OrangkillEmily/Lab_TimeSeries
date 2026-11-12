# BÁO CÁO TỔNG HỢP MÔ HÌNH (MODEL SUMMARY) — MONTHLY RETAIL SALES (US, 07/1991 - 06/2008)

## 1. Thông tin Phiên Thực nghiệm & Bộ dữ liệu (Provenance & Dataset Info)
- **Run ID**: `dataset1_20261112_082025_d2a40e`
- **Thời điểm xuất báo cáo**: `2026-11-12 08:24:14`
- **Dataset Key**: `dataset1`
- **Tên bộ dữ liệu**: Monthly Retail Sales (US, 07/1991 - 06/2008)
- **Tổng số quan sát**: 204 tháng (1991-07-01 to 2008-06-01)
- **Tần suất**: `MS` (Monthly Start)
- **Tập Huấn luyện (Train - 80%)**: 163 quan sát
- **Tập Kiểm thử (Test - 20%)**: 41 quan sát
- **Tổng số mô hình ứng viên**: 78 (Thành công: 78, Thất bại: 0)

## 2. Bảng So sánh Tổng hợp các Mô hình Ứng viên (Model Comparison Table)

> **Nguyên tắc đối chiếu**: Tất cả chỉ số dự báo ngoài mẫu (`MAE`, `RMSE`, `MAPE (%)`, `R²`) đều được đánh giá trên **cùng tập Test và cùng thang đo gốc** (đã áp dụng biến đổi ngược $\exp(\cdot)$ cho các mô hình `Log-SARIMA`). Chỉ số `AIC` và `BIC` chỉ được so sánh trực tiếp trong cùng một nhóm `Scale_Group`.

| Model                        | Family     | Status   | Scale_Group        |     AIC |     BIC |   MAE |   RMSE |   MAPE (%) |      R² |   Ljung-Box p | White Noise      | Converged   |
|:-----------------------------|:-----------|:---------|:-------------------|--------:|--------:|------:|-------:|-----------:|--------:|--------------:|:-----------------|:------------|
| Log-SARIMA(2,1,0)x(0,1,1)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -395.75 | -384.07 |  1.18 |   1.48 |       6.39 |  0.8711 |        0.2012 | Pass (Đạt)       | True        |
| Log-SARIMA(0,1,3)x(0,1,1)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -387.07 | -372.58 |  1.18 |   1.49 |       6.37 |  0.8703 |        0.0281 | Fail (Không đạt) | True        |
| Log-SARIMA(1,1,1)x(0,1,1)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -396.57 | -384.92 |  1.18 |   1.49 |       6.4  |  0.8697 |        0.017  | Fail (Không đạt) | True        |
| Log-SARIMA(2,1,0)x(1,1,1)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -385.47 | -370.91 |  1.18 |   1.49 |       6.45 |  0.8694 |        0.5592 | Pass (Đạt)       | True        |
| Log-SARIMA(0,1,1)x(0,1,1)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -395.54 | -386.8  |  1.19 |   1.49 |       6.38 |  0.8698 |        0.0039 | Fail (Không đạt) | True        |
| Log-SARIMA(1,1,3)x(0,1,1)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -385.48 | -368.09 |  1.18 |   1.5  |       6.46 |  0.8688 |        0.0227 | Fail (Không đạt) | True        |
| Log-SARIMA(0,1,3)x(1,1,1)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -386.8  | -369.42 |  1.19 |   1.5  |       6.45 |  0.8687 |        0.0231 | Fail (Không đạt) | True        |
| Log-SARIMA(2,1,1)x(0,1,1)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -394.02 | -379.46 |  1.19 |   1.5  |       6.47 |  0.8685 |        0.0666 | Pass (Đạt)       | True        |
| Log-SARIMA(2,1,3)x(0,1,1)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -402.78 | -382.49 |  1.2  |   1.5  |       6.53 |  0.8686 |        0.8259 | Pass (Đạt)       | True        |
| Log-SARIMA(0,1,1)x(1,1,1)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -395.64 | -383.99 |  1.21 |   1.51 |       6.56 |  0.8665 |        0.012  | Fail (Không đạt) | False       |
| Log-SARIMA(0,1,0)x(1,1,1)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -327    | -318.24 |  1.23 |   1.51 |       6.49 |  0.8662 |        0      | Fail (Không đạt) | True        |
| Log-SARIMA(1,1,1)x(1,1,1)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -393.98 | -379.42 |  1.21 |   1.52 |       6.62 |  0.8648 |        0.0216 | Fail (Không đạt) | True        |
| Log-SARIMA(1,1,0)x(1,1,1)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -371.54 | -359.86 |  1.22 |   1.52 |       6.43 |  0.8653 |        0      | Fail (Không đạt) | True        |
| Log-SARIMA(1,1,3)x(1,1,1)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -383.1  | -362.82 |  1.22 |   1.53 |       6.67 |  0.8634 |        0.0601 | Pass (Đạt)       | True        |
| Log-SARIMA(2,1,3)x(1,1,1)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -395.65 | -372.47 |  1.25 |   1.54 |       6.83 |  0.8613 |        0.8207 | Pass (Đạt)       | False       |
| Log-SARIMA(2,1,1)x(1,1,1)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -391.17 | -373.7  |  1.31 |   1.58 |       6.79 |  0.8546 |        0.001  | Fail (Không đạt) | False       |
| Log-SARIMA(1,1,0)x(0,1,1)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -368.96 | -360.2  |  1.27 |   1.59 |       6.52 |  0.8528 |        0      | Fail (Không đạt) | True        |
| Log-SARIMA(0,1,0)x(0,1,1)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -320.46 | -314.62 |  1.32 |   1.65 |       6.71 |  0.8401 |        0      | Fail (Không đạt) | True        |
| Log-SARIMA(1,1,0)x(1,1,0)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -355.82 | -347.06 |  1.34 |   1.71 |       7.48 |  0.8289 |        0      | Fail (Không đạt) | True        |
| Log-SARIMA(0,1,0)x(1,1,0)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -310.84 | -304.98 |  1.42 |   1.8  |       7.91 |  0.81   |        0      | Fail (Không đạt) | True        |
| SARIMA(0,1,1)x(0,1,1)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  233.85 |  242.59 |  1.45 |   1.84 |       8.21 |  0.8012 |        0.1053 | Pass (Đạt)       | True        |
| SARIMA(2,1,3)x(0,1,1)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  223.07 |  243.35 |  1.48 |   1.87 |       8.38 |  0.7956 |        0.9712 | Pass (Đạt)       | True        |
| SARIMA(2,1,3)x(1,1,1)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  225.07 |  248.25 |  1.48 |   1.87 |       8.38 |  0.7958 |        0.9716 | Pass (Đạt)       | False       |
| SARIMA(0,1,1)x(1,1,1)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  235.32 |  246.97 |  1.47 |   1.88 |       8.38 |  0.7936 |        0.109  | Pass (Đạt)       | True        |
| SARIMA(1,1,0)x(0,1,1)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  258.39 |  267.15 |  1.47 |   1.91 |       8.48 |  0.7866 |        0      | Fail (Không đạt) | True        |
| SARIMA(1,1,1)x(0,1,1)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  229.45 |  241.1  |  1.47 |   1.91 |       8.48 |  0.7862 |        0.4558 | Pass (Đạt)       | True        |
| SARIMA(0,1,3)x(0,1,1)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  229.34 |  243.83 |  1.5  |   1.95 |       8.72 |  0.7767 |        0.6242 | Pass (Đạt)       | True        |
| SARIMA(1,1,1)x(1,1,1)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  231.11 |  245.67 |  1.5  |   1.95 |       8.69 |  0.7784 |        0.4142 | Pass (Đạt)       | True        |
| SARIMA(1,1,0)x(1,1,1)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  258.02 |  269.7  |  1.53 |   1.96 |       8.83 |  0.7751 |        0.0001 | Fail (Không đạt) | True        |
| SARIMA(0,1,3)x(1,1,1)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  230.7  |  248.09 |  1.53 |   1.98 |       8.88 |  0.7715 |        0.5653 | Pass (Đạt)       | True        |
| Log-SARIMA(2,1,3)x(1,1,0)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -394.35 | -373.96 |  1.57 |   1.99 |       8.66 |  0.7676 |        0.5022 | Pass (Đạt)       | False       |
| SARIMA(0,1,1)x(1,1,0)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  239.49 |  248.27 |  1.54 |   2    |       8.95 |  0.7656 |        0.0571 | Pass (Đạt)       | True        |
| SARIMA(1,1,3)x(0,1,1)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  230.78 |  248.17 |  1.54 |   2    |       8.98 |  0.7661 |        0.6654 | Pass (Đạt)       | True        |
| SARIMA(2,1,1)x(0,1,1)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  229.46 |  244.02 |  1.54 |   2    |       8.97 |  0.7666 |        0.7873 | Pass (Đạt)       | True        |
| SARIMA(0,1,0)x(0,1,1)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  313.37 |  319.21 |  1.57 |   2.02 |       9.12 |  0.7606 |        0      | Fail (Không đạt) | True        |
| SARIMA(1,1,3)x(1,1,1)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  232.26 |  252.55 |  1.57 |   2.03 |       9.16 |  0.7596 |        0.6315 | Pass (Đạt)       | True        |
| SARIMA(2,1,1)x(1,1,1)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  231.25 |  248.72 |  1.57 |   2.03 |       9.16 |  0.7596 |        0.7655 | Pass (Đạt)       | True        |
| Log-SARIMA(1,1,0)x(0,1,0)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -383.99 | -377.98 |  1.61 |   2.04 |       8.95 |  0.7569 |        0      | Fail (Không đạt) | True        |
| SARIMA(2,1,0)x(0,1,1)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  230.05 |  241.73 |  1.59 |   2.06 |       9.32 |  0.7527 |        0.7484 | Pass (Đạt)       | True        |
| SARIMA(2,1,3)x(1,1,0)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  232.21 |  252.6  |  1.61 |   2.08 |       9.43 |  0.7473 |        0.8744 | Pass (Đạt)       | True        |
| SARIMA(2,1,0)x(1,1,1)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  230.85 |  245.41 |  1.62 |   2.08 |       9.48 |  0.7467 |        0.7314 | Pass (Đạt)       | True        |
| SARIMA(0,1,0)x(1,1,1)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  312.58 |  321.34 |  1.63 |   2.08 |       9.51 |  0.7478 |        0      | Fail (Không đạt) | True        |
| Log-SARIMA(0,1,1)x(1,1,0)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -387.66 | -378.88 |  1.65 |   2.08 |       9.09 |  0.7467 |        0.0038 | Fail (Không đạt) | True        |
| Log-SARIMA(2,1,0)x(1,1,0)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -376.08 | -364.43 |  1.65 |   2.08 |       9.13 |  0.7475 |        0.0329 | Fail (Không đạt) | True        |
| SARIMA(1,1,0)x(1,1,0)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  262.04 |  270.8  |  1.62 |   2.09 |       9.46 |  0.7449 |        0      | Fail (Không đạt) | True        |
| SARIMA(1,1,1)x(1,1,0)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  233.84 |  245.52 |  1.63 |   2.1  |       9.52 |  0.7426 |        0.2629 | Pass (Đạt)       | True        |
| SARIMA(0,1,3)x(1,1,0)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  236.59 |  251.22 |  1.71 |   2.17 |       9.95 |  0.7236 |        0.3995 | Pass (Đạt)       | True        |
| Log-SARIMA(1,1,1)x(1,1,0)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -386.29 | -374.61 |  1.72 |   2.17 |       9.48 |  0.7239 |        0.0041 | Fail (Không đạt) | True        |
| Log-SARIMA(0,1,3)x(1,1,0)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -385.4  | -370.76 |  1.72 |   2.18 |       9.51 |  0.7219 |        0.016  | Fail (Không đạt) | True        |
| SARIMA(2,1,1)x(1,1,0)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  232.9  |  247.47 |  1.73 |   2.2  |      10.09 |  0.7177 |        0.5702 | Pass (Đạt)       | True        |
| SARIMA(1,1,3)x(1,1,0)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  236.12 |  253.64 |  1.74 |   2.21 |      10.17 |  0.7131 |        0.5172 | Pass (Đạt)       | True        |
| Log-SARIMA(2,1,1)x(1,1,0)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -380.96 | -366.4  |  1.75 |   2.21 |       9.63 |  0.7145 |        0.0087 | Fail (Không đạt) | True        |
| Log-SARIMA(1,1,3)x(1,1,0)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -383.24 | -365.72 |  1.78 |   2.25 |       9.81 |  0.7031 |        0.0075 | Fail (Không đạt) | True        |
| SARIMA(2,1,0)x(1,1,0)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  233.13 |  244.78 |  1.82 |   2.29 |      10.61 |  0.6936 |        0.436  | Pass (Đạt)       | True        |
| SARIMA(0,1,0)x(1,1,0)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  317.32 |  323.17 |  1.84 |   2.32 |      10.75 |  0.6849 |        0      | Fail (Không đạt) | True        |
| SARIMA(1,1,0)x(0,1,0)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  278.02 |  284.02 |  1.85 |   2.32 |      10.77 |  0.6835 |        0.0001 | Fail (Không đạt) | False       |
| SARIMA(0,1,1)x(0,1,0)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  251.6  |  257.59 |  1.87 |   2.34 |      10.87 |  0.6787 |        0.0654 | Pass (Đạt)       | True        |
| SARIMA(1,1,1)x(0,1,0)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  248.92 |  257.91 |  2.02 |   2.5  |      11.71 |  0.6353 |        0.2564 | Pass (Đạt)       | True        |
| SARIMA(2,1,3)x(0,1,0)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  246.42 |  264.32 |  2.08 |   2.55 |      12.04 |  0.6186 |        0.6411 | Pass (Đạt)       | True        |
| SARIMA(0,1,3)x(0,1,0)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  248.47 |  260.41 |  2.09 |   2.57 |      12.1  |  0.6135 |        0.2871 | Pass (Đạt)       | True        |
| SARIMA(2,1,1)x(0,1,0)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  249.82 |  261.81 |  2.14 |   2.62 |      12.38 |  0.5974 |        0.3366 | Pass (Đạt)       | True        |
| SARIMA(1,1,3)x(0,1,0)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  249.43 |  264.34 |  2.16 |   2.64 |      12.45 |  0.5927 |        0.3919 | Pass (Đạt)       | True        |
| SARIMA(2,1,0)x(0,1,0)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  250.4  |  259.39 |  2.27 |   2.75 |      13.07 |  0.5558 |        0.2134 | Pass (Đạt)       | True        |
| Log-SARIMA(2,1,0)x(0,1,0)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -403.27 | -394.28 |  2.54 |   2.99 |      13.67 |  0.4775 |        0.0506 | Pass (Đạt)       | True        |
| Log-SARIMA(0,1,1)x(0,1,0)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -409.37 | -403.37 |  2.55 |   3.01 |      13.69 |  0.4686 |        0.0373 | Fail (Không đạt) | True        |
| Log-SARIMA(0,1,3)x(0,1,0)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -401.83 | -389.9  |  2.57 |   3.03 |      13.77 |  0.4627 |        0.0489 | Fail (Không đạt) | True        |
| Log-SARIMA(1,1,1)x(0,1,0)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -408.24 | -399.25 |  2.68 |   3.14 |      14.33 |  0.422  |        0.0564 | Pass (Đạt)       | True        |
| Log-SARIMA(2,1,1)x(0,1,0)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -406.24 | -394.25 |  2.69 |   3.15 |      14.37 |  0.4192 |        0.0576 | Pass (Đạt)       | True        |
| Log-SARIMA(1,1,3)x(0,1,0)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -401.44 | -386.52 |  2.76 |   3.22 |      14.74 |  0.3921 |        0.1266 | Pass (Đạt)       | True        |
| Log-SARIMA(2,1,3)x(0,1,0)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -415.62 | -397.72 |  2.75 |   3.25 |      14.64 |  0.3823 |        0.3381 | Pass (Đạt)       | True        |
| MA(1) [ARIMA(0,1,1)]         | MA         | SUCCESS  | original_d1        |  587.92 |  594.09 |  3.46 |   4.15 |      18.39 | -0.0068 |        0      | Fail (Không đạt) | True        |
| AR(2) [ARIMA(2,1,0)]         | AR         | SUCCESS  | original_d1        |  593.58 |  602.84 |  3.5  |   4.15 |      19.29 | -0.0063 |        0      | Fail (Không đạt) | True        |
| AR(1) [ARIMA(1,1,0)]         | AR         | SUCCESS  | original_d1        |  597.93 |  604.1  |  3.56 |   4.21 |      19.99 | -0.0358 |        0      | Fail (Không đạt) | True        |
| ARIMA(2,1,3)                 | ARIMA      | SUCCESS  | original_d1        |  568    |  586.53 |  3.51 |   4.36 |      17.65 | -0.1119 |        0      | Fail (Không đạt) | True        |
| MA(3) [ARIMA(0,1,3)]         | MA         | SUCCESS  | original_d1        |  584.61 |  596.96 |  3.64 |   4.59 |      17.96 | -0.2352 |        0      | Fail (Không đạt) | True        |
| ARIMA(1,1,1)                 | ARIMA      | SUCCESS  | original_d1        |  582.47 |  591.73 |  3.67 |   4.62 |      18.04 | -0.252  |        0      | Fail (Không đạt) | True        |
| ARIMA(1,1,3)                 | ARIMA      | SUCCESS  | original_d1        |  586.4  |  601.84 |  3.67 |   4.63 |      18.05 | -0.256  |        0      | Fail (Không đạt) | True        |
| ARIMA(2,1,1)                 | ARIMA      | SUCCESS  | original_d1        |  584.46 |  596.81 |  3.67 |   4.63 |      18.06 | -0.2543 |        0      | Fail (Không đạt) | True        |

### Diễn giải Đa Tiêu chí Tự động (Run ID: `dataset1_20261112_082025_d2a40e` — Monthly Retail Sales (US, 07/1991 - 06/2008))

#### A. Mô hình Tốt nhất theo Từng Tiêu chí Độc lập (Criterion-Specific Leaders)
- **Tiêu chí 1 — Sai số RMSE trên tập Test thấp nhất (Thang đo gốc)**: **Log-SARIMA(2,1,0)x(0,1,1)_12** (`RMSE = 1.48`, `R² = 0.8711`).
- **Tiêu chí 2 — Sai số tuyệt đối MAE trên tập Test thấp nhất**: **Log-SARIMA(2,1,0)x(0,1,1)_12** (`MAE = 1.18`).
- **Tiêu chí 3 — Sai số phần trăm MAPE (%) trên tập Test thấp nhất**: **Log-SARIMA(0,1,3)x(0,1,1)_12** (`MAPE = 6.37%`).
- **Tiêu chí 4 — Mô hình thỏa mãn đồng thời Chuẩn Nhiễu trắng (Ljung-Box $p \ge 0.05$), Hội tụ và có RMSE thấp nhất**: **Log-SARIMA(2,1,0)x(0,1,1)_12**.
- **Tiêu chí 5 — Tiêu chuẩn Thông tin (AIC / BIC) trên Tập Train theo từng nhóm thang đo hợp lệ (`Scale_Group`)**:
  - Nhóm `log_d1_D1_s12`: AIC thấp nhất là **Log-SARIMA(2,1,3)x(0,1,0)_12** (`AIC = -415.62`, `Test RMSE = 3.25`); BIC thấp nhất là **Log-SARIMA(0,1,1)x(0,1,0)_12** (`BIC = -403.37`).
  - Nhóm `original_d1`: AIC thấp nhất là **ARIMA(2,1,3)** (`AIC = 568.00`, `Test RMSE = 4.36`); BIC thấp nhất là **ARIMA(2,1,3)** (`BIC = 586.53`).
  - Nhóm `original_d1_D1_s12`: AIC thấp nhất là **SARIMA(2,1,3)x(0,1,1)_12** (`AIC = 223.07`, `Test RMSE = 1.87`); BIC thấp nhất là **SARIMA(1,1,1)x(0,1,1)_12** (`BIC = 241.10`).

#### B. Kiểm định Phần dư & Trạng thái Hội tụ
- **Các mô hình vượt qua kiểm định Ljung-Box ($p \ge 0.05$, phần dư đạt chuẩn White Noise)**: `Log-SARIMA(2,1,0)x(0,1,1)_12`, `Log-SARIMA(2,1,0)x(1,1,1)_12`, `Log-SARIMA(2,1,1)x(0,1,1)_12`, `Log-SARIMA(2,1,3)x(0,1,1)_12`, `Log-SARIMA(1,1,3)x(1,1,1)_12`, `Log-SARIMA(2,1,3)x(1,1,1)_12`, `SARIMA(0,1,1)x(0,1,1)_12`, `SARIMA(2,1,3)x(0,1,1)_12`, `SARIMA(2,1,3)x(1,1,1)_12`, `SARIMA(0,1,1)x(1,1,1)_12`, `SARIMA(1,1,1)x(0,1,1)_12`, `SARIMA(0,1,3)x(0,1,1)_12`, `SARIMA(1,1,1)x(1,1,1)_12`, `SARIMA(0,1,3)x(1,1,1)_12`, `Log-SARIMA(2,1,3)x(1,1,0)_12`, `SARIMA(0,1,1)x(1,1,0)_12`, `SARIMA(1,1,3)x(0,1,1)_12`, `SARIMA(2,1,1)x(0,1,1)_12`, `SARIMA(1,1,3)x(1,1,1)_12`, `SARIMA(2,1,1)x(1,1,1)_12`, `SARIMA(2,1,0)x(0,1,1)_12`, `SARIMA(2,1,3)x(1,1,0)_12`, `SARIMA(2,1,0)x(1,1,1)_12`, `SARIMA(1,1,1)x(1,1,0)_12`, `SARIMA(0,1,3)x(1,1,0)_12`, `SARIMA(2,1,1)x(1,1,0)_12`, `SARIMA(1,1,3)x(1,1,0)_12`, `SARIMA(2,1,0)x(1,1,0)_12`, `SARIMA(0,1,1)x(0,1,0)_12`, `SARIMA(1,1,1)x(0,1,0)_12`, `SARIMA(2,1,3)x(0,1,0)_12`, `SARIMA(0,1,3)x(0,1,0)_12`, `SARIMA(2,1,1)x(0,1,0)_12`, `SARIMA(1,1,3)x(0,1,0)_12`, `SARIMA(2,1,0)x(0,1,0)_12`, `Log-SARIMA(2,1,0)x(0,1,0)_12`, `Log-SARIMA(1,1,1)x(0,1,0)_12`, `Log-SARIMA(2,1,1)x(0,1,0)_12`, `Log-SARIMA(1,1,3)x(0,1,0)_12`, `Log-SARIMA(2,1,3)x(0,1,0)_12`.
- **Các mô hình vi phạm kiểm định Ljung-Box ($p < 0.05$, phần dư còn tự tương quan)**: `Log-SARIMA(0,1,3)x(0,1,1)_12`, `Log-SARIMA(1,1,1)x(0,1,1)_12`, `Log-SARIMA(0,1,1)x(0,1,1)_12`, `Log-SARIMA(1,1,3)x(0,1,1)_12`, `Log-SARIMA(0,1,3)x(1,1,1)_12`, `Log-SARIMA(0,1,1)x(1,1,1)_12`, `Log-SARIMA(0,1,0)x(1,1,1)_12`, `Log-SARIMA(1,1,1)x(1,1,1)_12`, `Log-SARIMA(1,1,0)x(1,1,1)_12`, `Log-SARIMA(2,1,1)x(1,1,1)_12`, `Log-SARIMA(1,1,0)x(0,1,1)_12`, `Log-SARIMA(0,1,0)x(0,1,1)_12`, `Log-SARIMA(1,1,0)x(1,1,0)_12`, `Log-SARIMA(0,1,0)x(1,1,0)_12`, `SARIMA(1,1,0)x(0,1,1)_12`, `SARIMA(1,1,0)x(1,1,1)_12`, `SARIMA(0,1,0)x(0,1,1)_12`, `Log-SARIMA(1,1,0)x(0,1,0)_12`, `SARIMA(0,1,0)x(1,1,1)_12`, `Log-SARIMA(0,1,1)x(1,1,0)_12`, `Log-SARIMA(2,1,0)x(1,1,0)_12`, `SARIMA(1,1,0)x(1,1,0)_12`, `Log-SARIMA(1,1,1)x(1,1,0)_12`, `Log-SARIMA(0,1,3)x(1,1,0)_12`, `Log-SARIMA(2,1,1)x(1,1,0)_12`, `Log-SARIMA(1,1,3)x(1,1,0)_12`, `SARIMA(0,1,0)x(1,1,0)_12`, `SARIMA(1,1,0)x(0,1,0)_12`, `Log-SARIMA(0,1,1)x(0,1,0)_12`, `Log-SARIMA(0,1,3)x(0,1,0)_12`, `MA(1) [ARIMA(0,1,1)]`, `AR(2) [ARIMA(2,1,0)]`, `AR(1) [ARIMA(1,1,0)]`, `ARIMA(2,1,3)`, `MA(3) [ARIMA(0,1,3)]`, `ARIMA(1,1,1)`, `ARIMA(1,1,3)`, `ARIMA(2,1,1)`.
- **Các mô hình có cảnh báo không hội tụ (`Converged = False`)**: `Log-SARIMA(0,1,1)x(1,1,1)_12`, `Log-SARIMA(2,1,3)x(1,1,1)_12`, `Log-SARIMA(2,1,1)x(1,1,1)_12`, `SARIMA(2,1,3)x(1,1,1)_12`, `Log-SARIMA(2,1,3)x(1,1,0)_12`, `SARIMA(1,1,0)x(0,1,0)_12`.
- **Các mô hình gặp lỗi thực thi (`Status = FAILED`)**: Không có (0 lỗi).

#### C. Lưu ý Phương pháp luận (Tại sao không khẳng định một mô hình tốt nhất tuyệt đối duy nhất?)
1. **Không so sánh chéo AIC/BIC giữa các nhóm khác bậc sai phân hoặc khác phép biến đổi**: Giá trị Log-Likelihood của nhóm `log_d1_D1_s12` được tính trên biến $\ln(Y_t)$ nên có AIC âm, không thể đem so sánh trực tiếp với AIC dương của nhóm `original_d1_D1_s12` hoặc `original_d1`.
2. **Sự đánh đổi giữa Statistical Fit (AIC/BIC), Residual Adequacy (Ljung-Box) và Out-of-Sample Error (RMSE/MAE)**: Một mô hình có thể đạt RMSE thấp nhất trên tập Test nhờ biến đổi Logarit nhưng phần dư trên không gian Log vẫn còn tự tương quan nhẹ (như trên Dataset 1), hoặc một mô hình bậc cao có AIC thấp nhất nhưng tốn nhiều tham số hơn cấu hình tối giản (như `SARIMA(2,1,3)` so với `SARIMA(0,1,1)`). Do đó, việc lựa chọn mô hình cuối cùng phụ thuộc vào mục tiêu ưu tiên độ chính xác dự báo ngoài mẫu hay tính chặt chẽ của giả định nhiễu trắng.

## 3. Hồ sơ Thống kê Chi tiết Từng Mô hình (Individual Model Cards)

### 3.1. AR(1) [ARIMA(1,1,0)]
```text
AR Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 0
Transformation: Original Scale (Sai phân bậc d=1)
Model: AR(1) [ARIMA(1,1,0)]
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -296.965   AIC:        597.930
BIC:                   604.105   HQIC:       600.437
Converged:                True   Run Time:   0.0302s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.3145     0.0922    -3.410     0.001     -0.495     -0.134
sigma2               2.2880     0.1874    12.209     0.000      1.921      2.655
==============================================================================
Ljung-Box (L=24) (Q):     264.57   Jarque-Bera (JB):     367.67
Prob(Q):                   0.0000   Prob(JB):             0.0000
Heteroskedasticity(H):       5.20   Skew:                  -2.06
Prob(H) (two-sided):       0.0000   Kurtosis:               9.13
Residual Mean:             0.1363   Residual Std:         1.5112
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 3.5600 | RMSE = 4.2060 | MAPE = 19.99% | R² = -0.0358
==============================================================================
```

### 3.2. AR(2) [ARIMA(2,1,0)]
```text
AR Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 2 | Differencing order (d): 1 | MA order (q): 0
Transformation: Original Scale (Sai phân bậc d=1)
Model: AR(2) [ARIMA(2,1,0)]
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -293.789   AIC:        593.579
BIC:                   602.842   HQIC:       597.340
Converged:                True   Run Time:   0.0394s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.3782     0.1009    -3.750     0.000     -0.576     -0.181
ar.L2               -0.1966     0.0936    -2.100     0.036     -0.380     -0.013
sigma2               2.1990     0.2480     8.867     0.000      1.713      2.685
==============================================================================
Ljung-Box (L=24) (Q):     261.75   Jarque-Bera (JB):     247.23
Prob(Q):                   0.0000   Prob(JB):             0.0000
Heteroskedasticity(H):       5.26   Skew:                  -1.71
Prob(H) (two-sided):       0.0000   Kurtosis:               7.99
Residual Mean:             0.1596   Residual Std:         1.4789
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 3.4961 | RMSE = 4.1457 | MAPE = 19.29% | R² = -0.0063
==============================================================================
```

### 3.3. MA(1) [ARIMA(0,1,1)]
```text
MA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 1
Transformation: Original Scale (Sai phân bậc d=1)
Model: MA(1) [ARIMA(0,1,1)]
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -291.958   AIC:        587.917
BIC:                   594.092   HQIC:       590.424
Converged:                True   Run Time:   0.0312s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ma.L1               -0.5330     0.0511   -10.437     0.000     -0.633     -0.433
sigma2               2.1479     0.2079    10.330     0.000      1.740      2.555
==============================================================================
Ljung-Box (L=24) (Q):     266.74   Jarque-Bera (JB):     125.49
Prob(Q):                   0.0000   Prob(JB):             0.0000
Heteroskedasticity(H):       4.99   Skew:                  -1.27
Prob(H) (two-sided):       0.0000   Kurtosis:               6.49
Residual Mean:             0.2073   Residual Std:         1.4554
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 3.4598 | RMSE = 4.1468 | MAPE = 18.39% | R² = -0.0068
==============================================================================
```

### 3.4. MA(3) [ARIMA(0,1,3)]
```text
MA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 3
Transformation: Original Scale (Sai phân bậc d=1)
Model: MA(3) [ARIMA(0,1,3)]
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -288.304   AIC:        584.608
BIC:                   596.959   HQIC:       589.623
Converged:                True   Run Time:   0.0661s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ma.L1               -0.4814     0.1101    -4.374     0.000     -0.697     -0.266
ma.L2               -0.1517     0.1092    -1.389     0.165     -0.366      0.062
ma.L3               -0.0752     0.1508    -0.499     0.618     -0.371      0.220
sigma2               2.0503     0.2021    10.144     0.000      1.654      2.446
==============================================================================
Ljung-Box (L=24) (Q):     240.11   Jarque-Bera (JB):     112.16
Prob(Q):                   0.0000   Prob(JB):             0.0000
Heteroskedasticity(H):       5.45   Skew:                  -0.98
Prob(H) (two-sided):       0.0000   Kurtosis:               6.58
Residual Mean:             0.2985   Residual Std:         1.4050
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 3.6449 | RMSE = 4.5931 | MAPE = 17.96% | R² = -0.2352
==============================================================================
```

### 3.5. ARIMA(1,1,1)
```text
ARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 1
Transformation: Original Scale (Sai phân bậc d=1)
Model: ARIMA(1,1,1)
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -288.234   AIC:        582.467
BIC:                   591.730   HQIC:       586.228
Converged:                True   Run Time:   0.0429s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1                0.3322     0.2367     1.404     0.160     -0.132      0.796
ma.L1               -0.8059     0.1339    -6.017     0.000     -1.068     -0.543
sigma2               2.0487     0.1870    10.957     0.000      1.682      2.415
==============================================================================
Ljung-Box (L=24) (Q):     239.86   Jarque-Bera (JB):     117.63
Prob(Q):                   0.0000   Prob(JB):             0.0000
Heteroskedasticity(H):       5.46   Skew:                  -1.02
Prob(H) (two-sided):       0.0000   Kurtosis:               6.64
Residual Mean:             0.2983   Residual Std:         1.4045
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 3.6667 | RMSE = 4.6243 | MAPE = 18.04% | R² = -0.2520
==============================================================================
```

### 3.6. ARIMA(1,1,3)
```text
ARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 3
Transformation: Original Scale (Sai phân bậc d=1)
Model: ARIMA(1,1,3)
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -288.199   AIC:        586.397
BIC:                   601.835   HQIC:       592.665
Converged:                True   Run Time:   0.1446s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1                0.2569     1.8826     0.136     0.891     -3.433      3.947
ma.L1               -0.7338     1.8881    -0.389     0.698     -4.434      2.967
ma.L2               -0.0183     0.9316    -0.020     0.984     -1.844      1.808
ma.L3               -0.0326     0.3729    -0.088     0.930     -0.764      0.698
sigma2               2.0478     0.2051     9.983     0.000      1.646      2.450
==============================================================================
Ljung-Box (L=24) (Q):     239.63   Jarque-Bera (JB):     119.73
Prob(Q):                   0.0000   Prob(JB):             0.0000
Heteroskedasticity(H):       5.44   Skew:                  -1.03
Prob(H) (two-sided):       0.0000   Kurtosis:               6.67
Residual Mean:             0.2989   Residual Std:         1.4040
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 3.6712 | RMSE = 4.6316 | MAPE = 18.05% | R² = -0.2560
==============================================================================
```

### 3.7. ARIMA(2,1,1)
```text
ARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 2 | Differencing order (d): 1 | MA order (q): 1
Transformation: Original Scale (Sai phân bậc d=1)
Model: ARIMA(2,1,1)
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -288.230   AIC:        584.460
BIC:                   596.810   HQIC:       589.475
Converged:                True   Run Time:   0.0528s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1                0.3326     0.2785     1.194     0.232     -0.213      0.878
ar.L2                0.0075     0.1505     0.050     0.960     -0.287      0.302
ma.L1               -0.8081     0.1946    -4.153     0.000     -1.189     -0.427
sigma2               2.0487     0.1873    10.939     0.000      1.682      2.416
==============================================================================
Ljung-Box (L=24) (Q):     239.98   Jarque-Bera (JB):     118.64
Prob(Q):                   0.0000   Prob(JB):             0.0000
Heteroskedasticity(H):       5.45   Skew:                  -1.03
Prob(H) (two-sided):       0.0000   Kurtosis:               6.65
Residual Mean:             0.2980   Residual Std:         1.4045
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 3.6702 | RMSE = 4.6285 | MAPE = 18.06% | R² = -0.2543
==============================================================================
```

### 3.8. ARIMA(2,1,3)
```text
ARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 2 | Differencing order (d): 1 | MA order (q): 3
Transformation: Original Scale (Sai phân bậc d=1)
Model: ARIMA(2,1,3)
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -278.002   AIC:        568.004
BIC:                   586.530   HQIC:       575.526
Converged:                True   Run Time:   0.3858s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -1.5153     0.1197   -12.661     0.000     -1.750     -1.281
ar.L2               -0.5233     0.1174    -4.457     0.000     -0.753     -0.293
ma.L1                1.1663     0.0864    13.502     0.000      0.997      1.336
ma.L2               -0.4394     0.0998    -4.405     0.000     -0.635     -0.244
ma.L3               -0.6427     0.0447   -14.388     0.000     -0.730     -0.555
sigma2               1.7847     0.1604    11.124     0.000      1.470      2.099
==============================================================================
Ljung-Box (L=24) (Q):     234.21   Jarque-Bera (JB):      28.98
Prob(Q):                   0.0000   Prob(JB):             0.0000
Heteroskedasticity(H):       4.97   Skew:                  -0.28
Prob(H) (two-sided):       0.0000   Kurtosis:               4.99
Residual Mean:             0.2532   Residual Std:         1.3180
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 3.5149 | RMSE = 4.3578 | MAPE = 17.65% | R² = -0.1119
==============================================================================
```

### 3.9. SARIMA(0,1,0)x(0,1,1)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 0
Seasonal order (P, D, Q, s): (0, 1, 1, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(0,1,0)x(0,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -154.687   AIC:        313.374
BIC:                   319.214   HQIC:       315.747
Converged:                True   Run Time:   0.0699s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ma.S.L12            -0.2819     0.0558    -5.049     0.000     -0.391     -0.172
sigma2               0.5598     0.0669     8.370     0.000      0.429      0.691
==============================================================================
Ljung-Box (L=24) (Q):      96.38   Jarque-Bera (JB):       0.53
Prob(Q):                   0.0000   Prob(JB):             0.7656
Heteroskedasticity(H):       3.62   Skew:                   0.11
Prob(H) (two-sided):       0.0000   Kurtosis:               3.20
Residual Mean:             0.0193   Residual Std:         0.7247
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.5652 | RMSE = 2.0219 | MAPE = 9.12% | R² = 0.7606
==============================================================================
```

### 3.10. SARIMA(0,1,0)x(1,1,0)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 0
Seasonal order (P, D, Q, s): (1, 1, 0, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(0,1,0)x(1,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -156.659   AIC:        317.317
BIC:                   323.172   HQIC:       319.696
Converged:                True   Run Time:   0.0453s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.S.L12            -0.1245     0.0700    -1.780     0.075     -0.262      0.013
sigma2               0.5670     0.0667     8.504     0.000      0.436      0.698
==============================================================================
Ljung-Box (L=24) (Q):      97.60   Jarque-Bera (JB):       1.29
Prob(Q):                   0.0000   Prob(JB):             0.5243
Heteroskedasticity(H):       3.62   Skew:                   0.15
Prob(H) (two-sided):       0.0000   Kurtosis:               3.33
Residual Mean:             0.0177   Residual Std:         0.7316
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.8449 | RMSE = 2.3198 | MAPE = 10.75% | R² = 0.6849
==============================================================================
```

### 3.11. SARIMA(0,1,0)x(1,1,1)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 0
Seasonal order (P, D, Q, s): (1, 1, 1, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(0,1,0)x(1,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -153.289   AIC:        312.579
BIC:                   321.339   HQIC:       316.139
Converged:                True   Run Time:   0.1644s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.S.L12             0.2912     0.2160     1.348     0.178     -0.132      0.715
ma.S.L12            -0.5822     0.1940    -3.001     0.003     -0.962     -0.202
sigma2               0.5304     0.0625     8.480     0.000      0.408      0.653
==============================================================================
Ljung-Box (L=24) (Q):      89.30   Jarque-Bera (JB):       0.85
Prob(Q):                   0.0000   Prob(JB):             0.6537
Heteroskedasticity(H):       3.77   Skew:                   0.02
Prob(H) (two-sided):       0.0000   Kurtosis:               3.37
Residual Mean:             0.0209   Residual Std:         0.7076
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.6285 | RMSE = 2.0753 | MAPE = 9.51% | R² = 0.7478
==============================================================================
```

### 3.12. SARIMA(0,1,1)x(0,1,0)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 1
Seasonal order (P, D, Q, s): (0, 1, 0, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(0,1,1)x(0,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -123.798   AIC:        251.597
BIC:                   257.591   HQIC:       254.032
Converged:                True   Run Time:   0.0373s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ma.L1               -0.7426     0.0526   -14.118     0.000     -0.846     -0.640
sigma2               0.3116     0.0297    10.494     0.000      0.253      0.370
==============================================================================
Ljung-Box (L=24) (Q):      35.20   Jarque-Bera (JB):       7.43
Prob(Q):                   0.0654   Prob(JB):             0.0244
Heteroskedasticity(H):       3.35   Skew:                   0.06
Prob(H) (two-sided):       0.0000   Kurtosis:               4.08
Residual Mean:             0.0601   Residual Std:         0.5537
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.8672 | RMSE = 2.3425 | MAPE = 10.87% | R² = 0.6787
==============================================================================
```

### 3.13. SARIMA(0,1,1)x(0,1,1)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 1
Seasonal order (P, D, Q, s): (0, 1, 1, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(0,1,1)x(0,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -113.926   AIC:        233.852
BIC:                   242.590   HQIC:       237.403
Converged:                True   Run Time:   0.1020s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ma.L1               -0.7488     0.0599   -12.512     0.000     -0.866     -0.632
ma.S.L12            -0.3110     0.0705    -4.410     0.000     -0.449     -0.173
sigma2               0.3124     0.0364     8.575     0.000      0.241      0.384
==============================================================================
Ljung-Box (L=24) (Q):      32.94   Jarque-Bera (JB):       1.57
Prob(Q):                   0.1053   Prob(JB):             0.4558
Heteroskedasticity(H):       3.60   Skew:                   0.14
Prob(H) (two-sided):       0.0000   Kurtosis:               3.41
Residual Mean:             0.0620   Residual Std:         0.5399
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.4515 | RMSE = 1.8426 | MAPE = 8.21% | R² = 0.8012
==============================================================================
```

### 3.14. SARIMA(0,1,1)x(1,1,0)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 1
Seasonal order (P, D, Q, s): (1, 1, 0, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(0,1,1)x(1,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -116.743   AIC:        239.486
BIC:                   248.268   HQIC:       243.055
Converged:                True   Run Time:   0.0815s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ma.L1               -0.7453     0.0574   -12.990     0.000     -0.858     -0.633
ar.S.L12            -0.1679     0.0838    -2.003     0.045     -0.332     -0.004
sigma2               0.3161     0.0349     9.048     0.000      0.248      0.385
==============================================================================
Ljung-Box (L=24) (Q):      35.82   Jarque-Bera (JB):       3.21
Prob(Q):                   0.0571   Prob(JB):             0.2009
Heteroskedasticity(H):       3.41   Skew:                   0.14
Prob(H) (two-sided):       0.0001   Kurtosis:               3.66
Residual Mean:             0.0555   Residual Std:         0.5487
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.5350 | RMSE = 2.0008 | MAPE = 8.95% | R² = 0.7656
==============================================================================
```

### 3.15. SARIMA(0,1,1)x(1,1,1)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 1
Seasonal order (P, D, Q, s): (1, 1, 1, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(0,1,1)x(1,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -113.661   AIC:        235.323
BIC:                   246.973   HQIC:       240.057
Converged:                True   Run Time:   0.2079s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ma.L1               -0.7410     0.0608   -12.189     0.000     -0.860     -0.622
ar.S.L12             0.1515     0.2127     0.712     0.476     -0.265      0.568
ma.S.L12            -0.4348     0.2006    -2.167     0.030     -0.828     -0.042
sigma2               0.3061     0.0346     8.838     0.000      0.238      0.374
==============================================================================
Ljung-Box (L=24) (Q):      32.77   Jarque-Bera (JB):       2.32
Prob(Q):                   0.1090   Prob(JB):             0.3138
Heteroskedasticity(H):       3.67   Skew:                   0.05
Prob(H) (two-sided):       0.0000   Kurtosis:               3.60
Residual Mean:             0.0554   Residual Std:         0.5359
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.4688 | RMSE = 1.8774 | MAPE = 8.38% | R² = 0.7936
==============================================================================
```

### 3.16. SARIMA(0,1,3)x(0,1,0)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 3
Seasonal order (P, D, Q, s): (0, 1, 0, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(0,1,3)x(0,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -120.237   AIC:        248.473
BIC:                   260.408   HQIC:       253.323
Converged:                True   Run Time:   0.0768s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ma.L1               -0.8825     0.0834   -10.584     0.000     -1.046     -0.719
ma.L2                0.1804     0.1063     1.697     0.090     -0.028      0.389
ma.L3                0.0006     0.0941     0.006     0.995     -0.184      0.185
sigma2               0.3040     0.0290    10.481     0.000      0.247      0.361
==============================================================================
Ljung-Box (L=24) (Q):      27.38   Jarque-Bera (JB):       9.91
Prob(Q):                   0.2871   Prob(JB):             0.0070
Heteroskedasticity(H):       2.80   Skew:                  -0.13
Prob(H) (two-sided):       0.0005   Kurtosis:               4.23
Residual Mean:             0.0556   Residual Std:         0.5447
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 2.0912 | RMSE = 2.5692 | MAPE = 12.10% | R² = 0.6135
==============================================================================
```

### 3.17. SARIMA(0,1,3)x(0,1,1)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 3
Seasonal order (P, D, Q, s): (0, 1, 1, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(0,1,3)x(0,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -109.670   AIC:        229.340
BIC:                   243.830   HQIC:       235.228
Converged:                True   Run Time:   0.1970s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ma.L1               -0.9250     0.0923   -10.020     0.000     -1.106     -0.744
ma.L2                0.2484     0.1128     2.203     0.028      0.027      0.469
ma.L3               -0.0204     0.1070    -0.191     0.849     -0.230      0.189
ma.S.L12            -0.3174     0.0774    -4.099     0.000     -0.469     -0.166
sigma2               0.3006     0.0351     8.565     0.000      0.232      0.369
==============================================================================
Ljung-Box (L=24) (Q):      21.25   Jarque-Bera (JB):       2.33
Prob(Q):                   0.6242   Prob(JB):             0.3124
Heteroskedasticity(H):       3.10   Skew:                  -0.07
Prob(H) (two-sided):       0.0002   Kurtosis:               3.59
Residual Mean:             0.0532   Residual Std:         0.5300
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.5048 | RMSE = 1.9528 | MAPE = 8.72% | R² = 0.7767
==============================================================================
```

### 3.18. SARIMA(0,1,3)x(1,1,0)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 3
Seasonal order (P, D, Q, s): (1, 1, 0, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(0,1,3)x(1,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -113.294   AIC:        236.587
BIC:                   251.223   HQIC:       242.535
Converged:                True   Run Time:   0.1141s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ma.L1               -0.9166     0.0922    -9.940     0.000     -1.097     -0.736
ma.L2                0.2160     0.1210     1.785     0.074     -0.021      0.453
ma.L3                0.0040     0.1058     0.038     0.970     -0.203      0.211
ar.S.L12            -0.1906     0.0943    -2.021     0.043     -0.375     -0.006
sigma2               0.3004     0.0328     9.150     0.000      0.236      0.365
==============================================================================
Ljung-Box (L=24) (Q):      25.12   Jarque-Bera (JB):       4.04
Prob(Q):                   0.3995   Prob(JB):             0.1328
Heteroskedasticity(H):       2.98   Skew:                  -0.05
Prob(H) (two-sided):       0.0003   Kurtosis:               3.80
Residual Mean:             0.0515   Residual Std:         0.5361
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.7062 | RMSE = 2.1728 | MAPE = 9.95% | R² = 0.7236
==============================================================================
```

### 3.19. SARIMA(0,1,3)x(1,1,1)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 3
Seasonal order (P, D, Q, s): (1, 1, 1, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(0,1,3)x(1,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -109.351   AIC:        230.702
BIC:                   248.089   HQIC:       237.768
Converged:                True   Run Time:   0.3420s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ma.L1               -0.9158     0.0922    -9.929     0.000     -1.097     -0.735
ma.L2                0.2503     0.1137     2.202     0.028      0.028      0.473
ma.L3               -0.0287     0.1091    -0.263     0.792     -0.243      0.185
ar.S.L12             0.1078     0.2269     0.475     0.635     -0.337      0.553
ma.S.L12            -0.4016     0.2143    -1.874     0.061     -0.822      0.018
sigma2               0.2961     0.0340     8.706     0.000      0.229      0.363
==============================================================================
Ljung-Box (L=24) (Q):      22.23   Jarque-Bera (JB):       5.18
Prob(Q):                   0.5653   Prob(JB):             0.0749
Heteroskedasticity(H):       3.20   Skew:                  -0.17
Prob(H) (two-sided):       0.0002   Kurtosis:               3.85
Residual Mean:             0.0505   Residual Std:         0.5242
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.5288 | RMSE = 1.9757 | MAPE = 8.88% | R² = 0.7715
==============================================================================
```

### 3.20. SARIMA(1,1,0)x(0,1,0)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 0
Seasonal order (P, D, Q, s): (0, 1, 0, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(1,1,0)x(0,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -137.008   AIC:        278.017
BIC:                   284.025   HQIC:       280.458
Converged:               False   Run Time:   0.0433s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.5662     0.0598    -9.460     0.000     -0.683     -0.449
sigma2               0.3683     0.0348    10.595     0.000      0.300      0.436
==============================================================================
Ljung-Box (L=24) (Q):      60.31   Jarque-Bera (JB):       9.96
Prob(Q):                   0.0001   Prob(JB):             0.0069
Heteroskedasticity(H):       3.21   Skew:                  -0.21
Prob(H) (two-sided):       0.0001   Kurtosis:               4.19
Residual Mean:             0.0252   Residual Std:         0.6065
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.8502 | RMSE = 2.3250 | MAPE = 10.77% | R² = 0.6835
==============================================================================
Warnings:
  [1] Maximum Likelihood optimization failed to converge. Check mle_retvals
==============================================================================
```

### 3.21. SARIMA(1,1,0)x(0,1,1)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 0
Seasonal order (P, D, Q, s): (0, 1, 1, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(1,1,0)x(0,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -126.193   AIC:        258.386
BIC:                   267.146   HQIC:       261.946
Converged:                True   Run Time:   0.0901s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.5830     0.0617    -9.447     0.000     -0.704     -0.462
ma.S.L12            -0.3282     0.0650    -5.047     0.000     -0.456     -0.201
sigma2               0.3691     0.0438     8.426     0.000      0.283      0.455
==============================================================================
Ljung-Box (L=24) (Q):      66.99   Jarque-Bera (JB):       2.09
Prob(Q):                   0.0000   Prob(JB):             0.3517
Heteroskedasticity(H):       3.15   Skew:                  -0.21
Prob(H) (two-sided):       0.0002   Kurtosis:               3.39
Residual Mean:             0.0253   Residual Std:         0.5946
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.4737 | RMSE = 1.9090 | MAPE = 8.48% | R² = 0.7866
==============================================================================
```

### 3.22. SARIMA(1,1,0)x(1,1,0)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 0
Seasonal order (P, D, Q, s): (1, 1, 0, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(1,1,0)x(1,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -128.021   AIC:        262.041
BIC:                   270.801   HQIC:       265.601
Converged:                True   Run Time:   0.0701s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.5778     0.0660    -8.752     0.000     -0.707     -0.448
ar.S.L12            -0.1455     0.0881    -1.652     0.099     -0.318      0.027
sigma2               0.3795     0.0439     8.640     0.000      0.293      0.466
==============================================================================
Ljung-Box (L=24) (Q):      65.75   Jarque-Bera (JB):       4.08
Prob(Q):                   0.0000   Prob(JB):             0.1302
Heteroskedasticity(H):       2.98   Skew:                  -0.17
Prob(H) (two-sided):       0.0003   Kurtosis:               3.74
Residual Mean:             0.0218   Residual Std:         0.6026
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.6216 | RMSE = 2.0873 | MAPE = 9.46% | R² = 0.7449
==============================================================================
```

### 3.23. SARIMA(1,1,0)x(1,1,1)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 0
Seasonal order (P, D, Q, s): (1, 1, 1, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(1,1,0)x(1,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -125.011   AIC:        258.022
BIC:                   269.702   HQIC:       262.769
Converged:                True   Run Time:   0.1488s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.5821     0.0624    -9.326     0.000     -0.704     -0.460
ar.S.L12             0.2222     0.2004     1.109     0.267     -0.171      0.615
ma.S.L12            -0.5277     0.1832    -2.881     0.004     -0.887     -0.169
sigma2               0.3535     0.0402     8.794     0.000      0.275      0.432
==============================================================================
Ljung-Box (L=24) (Q):      60.66   Jarque-Bera (JB):       5.25
Prob(Q):                   0.0001   Prob(JB):             0.0723
Heteroskedasticity(H):       3.32   Skew:                  -0.30
Prob(H) (two-sided):       0.0001   Kurtosis:               3.70
Residual Mean:             0.0233   Residual Std:         0.5823
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.5297 | RMSE = 1.9600 | MAPE = 8.83% | R² = 0.7751
==============================================================================
```

### 3.24. SARIMA(1,1,1)x(0,1,0)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 1
Seasonal order (P, D, Q, s): (0, 1, 0, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(1,1,1)x(0,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -121.458   AIC:        248.915
BIC:                   257.907   HQIC:       252.569
Converged:                True   Run Time:   0.0581s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.2264     0.0934    -2.424     0.015     -0.409     -0.043
ma.L1               -0.6476     0.0705    -9.188     0.000     -0.786     -0.509
sigma2               0.3020     0.0281    10.754     0.000      0.247      0.357
==============================================================================
Ljung-Box (L=24) (Q):      28.09   Jarque-Bera (JB):      10.90
Prob(Q):                   0.2564   Prob(JB):             0.0043
Heteroskedasticity(H):       2.88   Skew:                  -0.14
Prob(H) (two-sided):       0.0003   Kurtosis:               4.29
Residual Mean:             0.0566   Residual Std:         0.5454
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 2.0200 | RMSE = 2.4959 | MAPE = 11.71% | R² = 0.6353
==============================================================================
```

### 3.25. SARIMA(1,1,1)x(0,1,1)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 1
Seasonal order (P, D, Q, s): (0, 1, 1, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(1,1,1)x(0,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -110.727   AIC:        229.454
BIC:                   241.105   HQIC:       234.189
Converged:                True   Run Time:   0.1475s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.2710     0.0917    -2.955     0.003     -0.451     -0.091
ma.L1               -0.6380     0.0758    -8.416     0.000     -0.787     -0.489
ma.S.L12            -0.3191     0.0714    -4.471     0.000     -0.459     -0.179
sigma2               0.2980     0.0340     8.774     0.000      0.231      0.365
==============================================================================
Ljung-Box (L=24) (Q):      24.10   Jarque-Bera (JB):       2.50
Prob(Q):                   0.4558   Prob(JB):             0.2871
Heteroskedasticity(H):       3.26   Skew:                  -0.06
Prob(H) (two-sided):       0.0001   Kurtosis:               3.62
Residual Mean:             0.0587   Residual Std:         0.5303
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.4727 | RMSE = 1.9107 | MAPE = 8.48% | R² = 0.7862
==============================================================================
```

### 3.26. SARIMA(1,1,1)x(1,1,0)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 1
Seasonal order (P, D, Q, s): (1, 1, 0, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(1,1,1)x(1,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -112.919   AIC:        233.838
BIC:                   245.518   HQIC:       238.584
Converged:                True   Run Time:   0.1200s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.2627     0.1016    -2.587     0.010     -0.462     -0.064
ma.L1               -0.6411     0.0749    -8.557     0.000     -0.788     -0.494
ar.S.L12            -0.1859     0.0931    -1.997     0.046     -0.368     -0.003
sigma2               0.3038     0.0334     9.099     0.000      0.238      0.369
==============================================================================
Ljung-Box (L=24) (Q):      27.93   Jarque-Bera (JB):       4.30
Prob(Q):                   0.2629   Prob(JB):             0.1164
Heteroskedasticity(H):       3.04   Skew:                  -0.06
Prob(H) (two-sided):       0.0002   Kurtosis:               3.82
Residual Mean:             0.0509   Residual Std:         0.5385
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.6315 | RMSE = 2.0969 | MAPE = 9.52% | R² = 0.7426
==============================================================================
```

### 3.27. SARIMA(1,1,1)x(1,1,1)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 1
Seasonal order (P, D, Q, s): (1, 1, 1, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(1,1,1)x(1,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -110.555   AIC:        231.109
BIC:                   245.673   HQIC:       237.028
Converged:                True   Run Time:   0.2031s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.2712     0.0940    -2.885     0.004     -0.455     -0.087
ma.L1               -0.6285     0.0776    -8.102     0.000     -0.781     -0.476
ar.S.L12             0.1034     0.2208     0.468     0.640     -0.329      0.536
ma.S.L12            -0.3944     0.2038    -1.935     0.053     -0.794      0.005
sigma2               0.2936     0.0328     8.941     0.000      0.229      0.358
==============================================================================
Ljung-Box (L=24) (Q):      24.84   Jarque-Bera (JB):       4.90
Prob(Q):                   0.4142   Prob(JB):             0.0863
Heteroskedasticity(H):       3.34   Skew:                  -0.15
Prob(H) (two-sided):       0.0001   Kurtosis:               3.83
Residual Mean:             0.0503   Residual Std:         0.5257
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.5024 | RMSE = 1.9453 | MAPE = 8.69% | R² = 0.7784
==============================================================================
```

### 3.28. SARIMA(1,1,3)x(0,1,0)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 3
Seasonal order (P, D, Q, s): (0, 1, 0, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(1,1,3)x(0,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -119.713   AIC:        249.425
BIC:                   264.343   HQIC:       255.487
Converged:                True   Run Time:   0.0971s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.6579     0.6410    -1.026     0.305     -1.914      0.598
ma.L1               -0.2045     0.6395    -0.320     0.749     -1.458      1.049
ma.L2               -0.4625     0.5086    -0.909     0.363     -1.459      0.534
ma.L3                0.1726     0.0868     1.988     0.047      0.002      0.343
sigma2               0.3017     0.0308     9.783     0.000      0.241      0.362
==============================================================================
Ljung-Box (L=24) (Q):      25.26   Jarque-Bera (JB):       6.29
Prob(Q):                   0.3919   Prob(JB):             0.0430
Heteroskedasticity(H):       2.81   Skew:                  -0.03
Prob(H) (two-sided):       0.0004   Kurtosis:               4.00
Residual Mean:             0.0552   Residual Std:         0.5430
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 2.1552 | RMSE = 2.6375 | MAPE = 12.45% | R² = 0.5927
==============================================================================
```

### 3.29. SARIMA(1,1,3)x(0,1,1)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 3
Seasonal order (P, D, Q, s): (0, 1, 1, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(1,1,3)x(0,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -109.390   AIC:        230.780
BIC:                   248.167   HQIC:       237.846
Converged:                True   Run Time:   0.3027s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.6427     0.8750    -0.734     0.463     -2.358      1.072
ma.L1               -0.2588     0.8623    -0.300     0.764     -1.949      1.431
ma.L2               -0.4173     0.7512    -0.555     0.579     -1.890      1.055
ma.L3                0.1882     0.1414     1.331     0.183     -0.089      0.465
ma.S.L12            -0.3069     0.0785    -3.912     0.000     -0.461     -0.153
sigma2               0.2994     0.0355     8.425     0.000      0.230      0.369
==============================================================================
Ljung-Box (L=24) (Q):      20.54   Jarque-Bera (JB):       1.25
Prob(Q):                   0.6654   Prob(JB):             0.5359
Heteroskedasticity(H):       3.10   Skew:                   0.04
Prob(H) (two-sided):       0.0002   Kurtosis:               3.44
Residual Mean:             0.0507   Residual Std:         0.5319
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.5418 | RMSE = 1.9987 | MAPE = 8.98% | R² = 0.7661
==============================================================================
```

### 3.30. SARIMA(1,1,3)x(1,1,0)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 3
Seasonal order (P, D, Q, s): (1, 1, 0, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(1,1,3)x(1,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -112.059   AIC:        236.118
BIC:                   253.638   HQIC:       243.237
Converged:                True   Run Time:   0.1954s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.6348     0.5973    -1.063     0.288     -1.806      0.536
ma.L1               -0.2636     0.5904    -0.446     0.655     -1.421      0.894
ma.L2               -0.4238     0.5029    -0.843     0.399     -1.410      0.562
ma.L3                0.1928     0.1063     1.813     0.070     -0.016      0.401
ar.S.L12            -0.1914     0.0948    -2.019     0.044     -0.377     -0.006
sigma2               0.2998     0.0338     8.861     0.000      0.233      0.366
==============================================================================
Ljung-Box (L=24) (Q):      23.04   Jarque-Bera (JB):       2.26
Prob(Q):                   0.5172   Prob(JB):             0.3230
Heteroskedasticity(H):       2.95   Skew:                   0.04
Prob(H) (two-sided):       0.0004   Kurtosis:               3.60
Residual Mean:             0.0473   Residual Std:         0.5380
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.7430 | RMSE = 2.2137 | MAPE = 10.17% | R² = 0.7131
==============================================================================
```

### 3.31. SARIMA(1,1,3)x(1,1,1)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 3
Seasonal order (P, D, Q, s): (1, 1, 1, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(1,1,3)x(1,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -109.131   AIC:        232.262
BIC:                   252.547   HQIC:       240.505
Converged:                True   Run Time:   0.3801s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.6413     0.7438    -0.862     0.389     -2.099      0.817
ma.L1               -0.2518     0.7271    -0.346     0.729     -1.677      1.173
ma.L2               -0.4105     0.6294    -0.652     0.514     -1.644      0.823
ma.L3                0.1818     0.1345     1.352     0.176     -0.082      0.445
ar.S.L12             0.0888     0.2306     0.385     0.700     -0.363      0.541
ma.S.L12            -0.3736     0.2166    -1.725     0.085     -0.798      0.051
sigma2               0.2954     0.0348     8.495     0.000      0.227      0.364
==============================================================================
Ljung-Box (L=24) (Q):      21.12   Jarque-Bera (JB):       2.67
Prob(Q):                   0.6315   Prob(JB):             0.2628
Heteroskedasticity(H):       3.20   Skew:                  -0.04
Prob(H) (two-sided):       0.0002   Kurtosis:               3.65
Residual Mean:             0.0455   Residual Std:         0.5274
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.5705 | RMSE = 2.0261 | MAPE = 9.16% | R² = 0.7596
==============================================================================
```

### 3.32. SARIMA(2,1,0)x(0,1,0)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 2 | Differencing order (d): 1 | MA order (q): 0
Seasonal order (P, D, Q, s): (0, 1, 0, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(2,1,0)x(0,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -122.202   AIC:        250.403
BIC:                   259.395   HQIC:       254.056
Converged:                True   Run Time:   0.0389s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.8049     0.0791   -10.181     0.000     -0.960     -0.650
ar.L2               -0.4268     0.0759    -5.626     0.000     -0.576     -0.278
sigma2               0.3053     0.0333     9.162     0.000      0.240      0.371
==============================================================================
Ljung-Box (L=24) (Q):      29.18   Jarque-Bera (JB):       3.46
Prob(Q):                   0.2134   Prob(JB):             0.1775
Heteroskedasticity(H):       2.46   Skew:                  -0.18
Prob(H) (two-sided):       0.0021   Kurtosis:               3.65
Residual Mean:             0.0386   Residual Std:         0.5496
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 2.2656 | RMSE = 2.7544 | MAPE = 13.07% | R² = 0.5558
==============================================================================
```

### 3.33. SARIMA(2,1,0)x(0,1,1)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 2 | Differencing order (d): 1 | MA order (q): 0
Seasonal order (P, D, Q, s): (0, 1, 1, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(2,1,0)x(0,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -111.025   AIC:        230.051
BIC:                   241.731   HQIC:       234.797
Converged:                True   Run Time:   0.1446s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.8448     0.0819   -10.309     0.000     -1.005     -0.684
ar.L2               -0.4494     0.0772    -5.823     0.000     -0.601     -0.298
ma.S.L12            -0.3240     0.0748    -4.331     0.000     -0.471     -0.177
sigma2               0.2956     0.0370     7.986     0.000      0.223      0.368
==============================================================================
Ljung-Box (L=24) (Q):      19.07   Jarque-Bera (JB):       0.11
Prob(Q):                   0.7484   Prob(JB):             0.9484
Heteroskedasticity(H):       2.86   Skew:                  -0.06
Prob(H) (two-sided):       0.0005   Kurtosis:               3.05
Residual Mean:             0.0380   Residual Std:         0.5355
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.5937 | RMSE = 2.0551 | MAPE = 9.32% | R² = 0.7527
==============================================================================
```

### 3.34. SARIMA(2,1,0)x(1,1,0)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 2 | Differencing order (d): 1 | MA order (q): 0
Seasonal order (P, D, Q, s): (1, 1, 0, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(2,1,0)x(1,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -112.566   AIC:        233.132
BIC:                   244.782   HQIC:       237.866
Converged:                True   Run Time:   0.1114s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.8399     0.0874    -9.607     0.000     -1.011     -0.669
ar.L2               -0.4536     0.0792    -5.730     0.000     -0.609     -0.298
ar.S.L12            -0.2033     0.0882    -2.305     0.021     -0.376     -0.030
sigma2               0.3065     0.0370     8.277     0.000      0.234      0.379
==============================================================================
Ljung-Box (L=24) (Q):      24.45   Jarque-Bera (JB):       0.29
Prob(Q):                   0.4360   Prob(JB):             0.8655
Heteroskedasticity(H):       2.67   Skew:                  -0.04
Prob(H) (two-sided):       0.0013   Kurtosis:               3.20
Residual Mean:             0.0315   Residual Std:         0.5436
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.8166 | RMSE = 2.2876 | MAPE = 10.61% | R² = 0.6936
==============================================================================
```

### 3.35. SARIMA(2,1,0)x(1,1,1)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 2 | Differencing order (d): 1 | MA order (q): 0
Seasonal order (P, D, Q, s): (1, 1, 1, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(2,1,0)x(1,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -110.423   AIC:        230.847
BIC:                   245.410   HQIC:       236.765
Converged:                True   Run Time:   0.2325s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.8430     0.0828   -10.185     0.000     -1.005     -0.681
ar.L2               -0.4426     0.0777    -5.693     0.000     -0.595     -0.290
ar.S.L12             0.0739     0.2120     0.349     0.727     -0.342      0.489
ma.S.L12            -0.3770     0.2024    -1.863     0.063     -0.774      0.020
sigma2               0.2935     0.0361     8.139     0.000      0.223      0.364
==============================================================================
Ljung-Box (L=24) (Q):      19.38   Jarque-Bera (JB):       0.71
Prob(Q):                   0.7314   Prob(JB):             0.7018
Heteroskedasticity(H):       2.97   Skew:                  -0.11
Prob(H) (two-sided):       0.0004   Kurtosis:               3.25
Residual Mean:             0.0327   Residual Std:         0.5295
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.6193 | RMSE = 2.0801 | MAPE = 9.48% | R² = 0.7467
==============================================================================
```

### 3.36. SARIMA(2,1,1)x(0,1,0)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 2 | Differencing order (d): 1 | MA order (q): 1
Seasonal order (P, D, Q, s): (0, 1, 0, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(2,1,1)x(0,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -120.912   AIC:        249.823
BIC:                   261.812   HQIC:       254.695
Converged:                True   Run Time:   0.0706s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.3517     0.1599    -2.200     0.028     -0.665     -0.038
ar.L2               -0.1409     0.1483    -0.950     0.342     -0.432      0.150
ma.L1               -0.5329     0.1586    -3.361     0.001     -0.844     -0.222
sigma2               0.2999     0.0307     9.764     0.000      0.240      0.360
==============================================================================
Ljung-Box (L=24) (Q):      26.33   Jarque-Bera (JB):       8.05
Prob(Q):                   0.3366   Prob(JB):             0.0179
Heteroskedasticity(H):       2.72   Skew:                  -0.12
Prob(H) (two-sided):       0.0006   Kurtosis:               4.11
Residual Mean:             0.0537   Residual Std:         0.5437
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 2.1414 | RMSE = 2.6221 | MAPE = 12.38% | R² = 0.5974
==============================================================================
```

### 3.37. SARIMA(2,1,1)x(0,1,1)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 2 | Differencing order (d): 1 | MA order (q): 1
Seasonal order (P, D, Q, s): (0, 1, 1, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(2,1,1)x(0,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -109.730   AIC:        229.460
BIC:                   244.023   HQIC:       235.378
Converged:                True   Run Time:   0.4559s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.4838     0.1668    -2.900     0.004     -0.811     -0.157
ar.L2               -0.2204     0.1392    -1.583     0.113     -0.493      0.052
ma.L1               -0.4383     0.1670    -2.625     0.009     -0.766     -0.111
ma.S.L12            -0.3190     0.0753    -4.236     0.000     -0.467     -0.171
sigma2               0.2937     0.0353     8.327     0.000      0.225      0.363
==============================================================================
Ljung-Box (L=24) (Q):      18.32   Jarque-Bera (JB):       1.16
Prob(Q):                   0.7873   Prob(JB):             0.5613
Heteroskedasticity(H):       3.01   Skew:                  -0.02
Prob(H) (two-sided):       0.0003   Kurtosis:               3.43
Residual Mean:             0.0505   Residual Std:         0.5288
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.5409 | RMSE = 1.9967 | MAPE = 8.97% | R² = 0.7666
==============================================================================
```

### 3.38. SARIMA(2,1,1)x(1,1,0)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 2 | Differencing order (d): 1 | MA order (q): 1
Seasonal order (P, D, Q, s): (1, 1, 0, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(2,1,1)x(1,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -111.451   AIC:        232.902
BIC:                   247.466   HQIC:       238.820
Converged:                True   Run Time:   0.1691s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.4549     0.1658    -2.743     0.006     -0.780     -0.130
ar.L2               -0.2086     0.1440    -1.449     0.147     -0.491      0.074
ma.L1               -0.4677     0.1631    -2.868     0.004     -0.787     -0.148
ar.S.L12            -0.2026     0.0945    -2.145     0.032     -0.388     -0.017
sigma2               0.3014     0.0351     8.598     0.000      0.233      0.370
==============================================================================
Ljung-Box (L=24) (Q):      22.15   Jarque-Bera (JB):       2.23
Prob(Q):                   0.5702   Prob(JB):             0.3282
Heteroskedasticity(H):       2.84   Skew:                  -0.01
Prob(H) (two-sided):       0.0007   Kurtosis:               3.60
Residual Mean:             0.0455   Residual Std:         0.5357
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.7284 | RMSE = 2.1956 | MAPE = 10.09% | R² = 0.7177
==============================================================================
```

### 3.39. SARIMA(2,1,1)x(1,1,1)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 2 | Differencing order (d): 1 | MA order (q): 1
Seasonal order (P, D, Q, s): (1, 1, 1, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(2,1,1)x(1,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -109.624   AIC:        231.247
BIC:                   248.723   HQIC:       238.349
Converged:                True   Run Time:   0.3615s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.4881     0.1701    -2.869     0.004     -0.821     -0.155
ar.L2               -0.2185     0.1406    -1.554     0.120     -0.494      0.057
ma.L1               -0.4272     0.1700    -2.513     0.012     -0.760     -0.094
ar.S.L12             0.0643     0.2225     0.289     0.773     -0.372      0.500
ma.S.L12            -0.3603     0.2097    -1.718     0.086     -0.771      0.051
sigma2               0.2904     0.0343     8.472     0.000      0.223      0.358
==============================================================================
Ljung-Box (L=24) (Q):      18.74   Jarque-Bera (JB):       2.79
Prob(Q):                   0.7655   Prob(JB):             0.2479
Heteroskedasticity(H):       3.09   Skew:                  -0.10
Prob(H) (two-sided):       0.0002   Kurtosis:               3.64
Residual Mean:             0.0444   Residual Std:         0.5233
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.5702 | RMSE = 2.0264 | MAPE = 9.16% | R² = 0.7596
==============================================================================
```

### 3.40. SARIMA(2,1,3)x(0,1,0)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 2 | Differencing order (d): 1 | MA order (q): 3
Seasonal order (P, D, Q, s): (0, 1, 0, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(2,1,3)x(0,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -117.211   AIC:        246.422
BIC:                   264.323   HQIC:       253.695
Converged:                True   Run Time:   0.1996s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -1.0179     0.1858    -5.478     0.000     -1.382     -0.654
ar.L2               -0.7086     0.1694    -4.182     0.000     -1.041     -0.376
ma.L1                0.1489     0.2102     0.708     0.479     -0.263      0.561
ma.L2                0.0093     0.1897     0.049     0.961     -0.363      0.381
ma.L3               -0.3478     0.1745    -1.994     0.046     -0.690     -0.006
sigma2               0.2913     0.0315     9.239     0.000      0.229      0.353
==============================================================================
Ljung-Box (L=24) (Q):      20.96   Jarque-Bera (JB):       5.48
Prob(Q):                   0.6411   Prob(JB):             0.0644
Heteroskedasticity(H):       2.81   Skew:                  -0.01
Prob(H) (two-sided):       0.0004   Kurtosis:               3.94
Residual Mean:             0.0559   Residual Std:         0.5336
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 2.0773 | RMSE = 2.5524 | MAPE = 12.04% | R² = 0.6186
==============================================================================
```

### 3.41. SARIMA(2,1,3)x(0,1,1)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 2 | Differencing order (d): 1 | MA order (q): 3
Seasonal order (P, D, Q, s): (0, 1, 1, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(2,1,3)x(0,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -104.535   AIC:        223.070
BIC:                   243.355   HQIC:       231.313
Converged:                True   Run Time:   0.7188s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.6735     0.0446   -15.091     0.000     -0.761     -0.586
ar.L2               -0.9015     0.0495   -18.209     0.000     -0.999     -0.805
ma.L1               -0.1583   231.6048    -0.001     0.999   -454.095    453.779
ma.L2                0.6101   677.6291     0.001     0.999  -1327.519   1328.739
ma.L3               -0.7086   596.4602    -0.001     0.999  -1169.749   1168.332
ma.S.L12            -0.3673     0.0705    -5.208     0.000     -0.506     -0.229
sigma2               0.2704   227.6402     0.001     0.999   -445.896    446.437
==============================================================================
Ljung-Box (L=24) (Q):      12.67   Jarque-Bera (JB):       0.36
Prob(Q):                   0.9712   Prob(JB):             0.8359
Heteroskedasticity(H):       2.62   Skew:                  -0.03
Prob(H) (two-sided):       0.0016   Kurtosis:               3.23
Residual Mean:             0.0528   Residual Std:         0.5151
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.4822 | RMSE = 1.8683 | MAPE = 8.38% | R² = 0.7956
==============================================================================
```

### 3.42. SARIMA(2,1,3)x(1,1,0)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 2 | Differencing order (d): 1 | MA order (q): 3
Seasonal order (P, D, Q, s): (1, 1, 0, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(2,1,3)x(1,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -109.103   AIC:        232.207
BIC:                   252.595   HQIC:       240.492
Converged:                True   Run Time:   0.3250s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.8912     0.2140    -4.165     0.000     -1.311     -0.472
ar.L2               -0.7078     0.1357    -5.215     0.000     -0.974     -0.442
ma.L1               -0.0115     0.2428    -0.048     0.962     -0.487      0.464
ma.L2                0.1421     0.2039     0.697     0.486     -0.258      0.542
ma.L3               -0.3612     0.1482    -2.437     0.015     -0.652     -0.071
ar.S.L12            -0.2181     0.1012    -2.156     0.031     -0.416     -0.020
sigma2               0.2903     0.0338     8.597     0.000      0.224      0.356
==============================================================================
Ljung-Box (L=24) (Q):      16.37   Jarque-Bera (JB):       1.62
Prob(Q):                   0.8744   Prob(JB):             0.4441
Heteroskedasticity(H):       3.06   Skew:                   0.10
Prob(H) (two-sided):       0.0003   Kurtosis:               3.47
Residual Mean:             0.0447   Residual Std:         0.5300
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.6132 | RMSE = 2.0774 | MAPE = 9.43% | R² = 0.7473
==============================================================================
```

### 3.43. SARIMA(2,1,3)x(1,1,1)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 2 | Differencing order (d): 1 | MA order (q): 3
Seasonal order (P, D, Q, s): (1, 1, 1, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(2,1,3)x(1,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -104.535   AIC:        225.070
BIC:                   248.253   HQIC:       234.491
Converged:               False   Run Time:   1.6169s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.6736     0.0447   -15.067     0.000     -0.761     -0.586
ar.L2               -0.9015     0.0495   -18.206     0.000     -0.999     -0.804
ma.L1               -0.1586     6.6605    -0.024     0.981    -13.213     12.896
ma.L2                0.6100    19.4683     0.031     0.975    -37.547     38.767
ma.L3               -0.7089    17.1342    -0.041     0.967    -34.291     32.873
ar.S.L12             0.0000     0.0075     0.004     0.997     -0.015      0.015
ma.S.L12            -0.3691     0.0739    -4.997     0.000     -0.514     -0.224
sigma2               0.2706     6.5433     0.041     0.967    -12.554     13.095
==============================================================================
Ljung-Box (L=24) (Q):      12.65   Jarque-Bera (JB):       0.35
Prob(Q):                   0.9716   Prob(JB):             0.8403
Heteroskedasticity(H):       2.62   Skew:                  -0.03
Prob(H) (two-sided):       0.0016   Kurtosis:               3.23
Residual Mean:             0.0528   Residual Std:         0.5151
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.4829 | RMSE = 1.8676 | MAPE = 8.38% | R² = 0.7958
==============================================================================
Warnings:
  [1] Maximum Likelihood optimization failed to converge. Check mle_retvals
==============================================================================
```

### 3.44. Log-SARIMA(0,1,0)x(0,1,1)_12
```text
Log-SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 0
Seasonal order (P, D, Q, s): (0, 1, 1, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(0,1,0)x(0,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        162.229   AIC:       -320.459
BIC:                  -314.619   HQIC:      -318.086
Converged:                True   Run Time:   0.1557s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ma.S.L12            -0.7584     0.0756   -10.027     0.000     -0.907     -0.610
sigma2               0.0053     0.0006     8.349     0.000      0.004      0.007
==============================================================================
Ljung-Box (L=24) (Q):     133.34   Jarque-Bera (JB):       1.78
Prob(Q):                   0.0000   Prob(JB):             0.4104
Heteroskedasticity(H):       0.97   Skew:                  -0.26
Prob(H) (two-sided):       0.9233   Kurtosis:               3.09
Residual Mean:             0.0006   Residual Std:         0.0747
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.3243 | RMSE = 1.6525 | MAPE = 6.71% | R² = 0.8401
==============================================================================
```

### 3.45. Log-SARIMA(0,1,0)x(1,1,0)_12
```text
Log-SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 0
Seasonal order (P, D, Q, s): (1, 1, 0, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(0,1,0)x(1,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        157.420   AIC:       -310.839
BIC:                  -304.985   HQIC:      -308.460
Converged:                True   Run Time:   0.0606s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.S.L12            -0.1283     0.0855    -1.501     0.133     -0.296      0.039
sigma2               0.0060     0.0008     7.836     0.000      0.004      0.007
==============================================================================
Ljung-Box (L=24) (Q):     107.85   Jarque-Bera (JB):       0.96
Prob(Q):                   0.0000   Prob(JB):             0.6175
Heteroskedasticity(H):       0.91   Skew:                  -0.17
Prob(H) (two-sided):       0.7556   Kurtosis:               2.82
Residual Mean:             0.0005   Residual Std:         0.0771
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.4227 | RMSE = 1.8015 | MAPE = 7.91% | R² = 0.8100
==============================================================================
```

### 3.46. Log-SARIMA(0,1,0)x(1,1,1)_12
```text
Log-SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 0
Seasonal order (P, D, Q, s): (1, 1, 1, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(0,1,0)x(1,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        166.498   AIC:       -326.996
BIC:                  -318.236   HQIC:      -323.436
Converged:                True   Run Time:   0.1971s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.S.L12             0.2663     0.1156     2.304     0.021      0.040      0.493
ma.S.L12            -0.8401     0.1342    -6.262     0.000     -1.103     -0.577
sigma2               0.0047     0.0005     8.846     0.000      0.004      0.006
==============================================================================
Ljung-Box (L=24) (Q):      98.22   Jarque-Bera (JB):       5.24
Prob(Q):                   0.0000   Prob(JB):             0.0728
Heteroskedasticity(H):       0.99   Skew:                  -0.39
Prob(H) (two-sided):       0.9706   Kurtosis:               3.48
Residual Mean:             0.0004   Residual Std:         0.0722
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.2301 | RMSE = 1.5117 | MAPE = 6.49% | R² = 0.8662
==============================================================================
```

### 3.47. Log-SARIMA(0,1,1)x(0,1,0)_12
```text
Log-SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 1
Seasonal order (P, D, Q, s): (0, 1, 0, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(0,1,1)x(0,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        206.684   AIC:       -409.368
BIC:                  -403.374   HQIC:      -406.933
Converged:                True   Run Time:   0.0529s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ma.L1               -0.7798     0.0618   -12.623     0.000     -0.901     -0.659
sigma2               0.0036     0.0003    12.272     0.000      0.003      0.004
==============================================================================
Ljung-Box (L=24) (Q):      37.69   Jarque-Bera (JB):      39.92
Prob(Q):                   0.0373   Prob(JB):             0.0000
Heteroskedasticity(H):       0.76   Skew:                  -0.52
Prob(H) (two-sided):       0.3463   Kurtosis:               5.30
Residual Mean:             0.0021   Residual Std:         0.0601
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 2.5495 | RMSE = 3.0126 | MAPE = 13.69% | R² = 0.4686
==============================================================================
```

### 3.48. Log-SARIMA(0,1,1)x(0,1,1)_12
```text
Log-SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 1
Seasonal order (P, D, Q, s): (0, 1, 1, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(0,1,1)x(0,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        200.769   AIC:       -395.537
BIC:                  -386.800   HQIC:      -391.987
Converged:                True   Run Time:   0.4509s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ma.L1               -0.8000     0.0539   -14.843     0.000     -0.906     -0.694
ma.S.L12            -0.6446     0.0612   -10.524     0.000     -0.765     -0.525
sigma2               0.0030     0.0003    10.315     0.000      0.002      0.004
==============================================================================
Ljung-Box (L=24) (Q):      46.50   Jarque-Bera (JB):      15.71
Prob(Q):                   0.0039   Prob(JB):             0.0004
Heteroskedasticity(H):       0.76   Skew:                  -0.50
Prob(H) (two-sided):       0.3588   Kurtosis:               4.24
Residual Mean:            -0.0031   Residual Std:         0.0560
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.1868 | RMSE = 1.4912 | MAPE = 6.38% | R² = 0.8698
==============================================================================
```

### 3.49. Log-SARIMA(0,1,1)x(1,1,0)_12
```text
Log-SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 1
Seasonal order (P, D, Q, s): (1, 1, 0, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(0,1,1)x(1,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        196.830   AIC:       -387.661
BIC:                  -378.879   HQIC:      -384.092
Converged:                True   Run Time:   0.1108s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ma.L1               -0.8068     0.0572   -14.115     0.000     -0.919     -0.695
ar.S.L12            -0.1949     0.0942    -2.068     0.039     -0.380     -0.010
sigma2               0.0034     0.0003    11.576     0.000      0.003      0.004
==============================================================================
Ljung-Box (L=24) (Q):      46.52   Jarque-Bera (JB):      26.71
Prob(Q):                   0.0038   Prob(JB):             0.0000
Heteroskedasticity(H):       0.81   Skew:                  -0.50
Prob(H) (two-sided):       0.4791   Kurtosis:               4.81
Residual Mean:            -0.0002   Residual Std:         0.0596
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.6454 | RMSE = 2.0798 | MAPE = 9.09% | R² = 0.7467
==============================================================================
```

### 3.50. Log-SARIMA(0,1,1)x(1,1,1)_12
```text
Log-SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 1
Seasonal order (P, D, Q, s): (1, 1, 1, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(0,1,1)x(1,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        201.822   AIC:       -395.645
BIC:                  -383.994   HQIC:      -390.910
Converged:               False   Run Time:   0.6065s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ma.L1               -0.8186     0.0605   -13.540     0.000     -0.937     -0.700
ar.S.L12             0.1864     0.1341     1.390     0.165     -0.077      0.449
ma.S.L12            -0.7508     0.1323    -5.674     0.000     -1.010     -0.491
sigma2               0.0028     0.0002    12.067     0.000      0.002      0.003
==============================================================================
Ljung-Box (L=24) (Q):      42.28   Jarque-Bera (JB):      65.26
Prob(Q):                   0.0120   Prob(JB):             0.0000
Heteroskedasticity(H):       0.78   Skew:                  -0.85
Prob(H) (two-sided):       0.4042   Kurtosis:               5.74
Residual Mean:            -0.0028   Residual Std:         0.0572
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.2054 | RMSE = 1.5102 | MAPE = 6.56% | R² = 0.8665
==============================================================================
Warnings:
  [1] Maximum Likelihood optimization failed to converge. Check mle_retvals
==============================================================================
```

### 3.51. Log-SARIMA(0,1,3)x(0,1,0)_12
```text
Log-SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 3
Seasonal order (P, D, Q, s): (0, 1, 0, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(0,1,3)x(0,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        204.917   AIC:       -401.834
BIC:                  -389.900   HQIC:      -396.985
Converged:                True   Run Time:   0.1290s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ma.L1               -0.8360     0.0824   -10.149     0.000     -0.998     -0.675
ma.L2                0.1117     0.1138     0.981     0.326     -0.111      0.335
ma.L3               -0.0444     0.0941    -0.472     0.637     -0.229      0.140
sigma2               0.0035     0.0003    12.431     0.000      0.003      0.004
==============================================================================
Ljung-Box (L=24) (Q):      36.52   Jarque-Bera (JB):      44.09
Prob(Q):                   0.0489   Prob(JB):             0.0000
Heteroskedasticity(H):       0.76   Skew:                  -0.54
Prob(H) (two-sided):       0.3427   Kurtosis:               5.43
Residual Mean:             0.0021   Residual Std:         0.0599
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 2.5666 | RMSE = 3.0293 | MAPE = 13.77% | R² = 0.4627
==============================================================================
```

### 3.52. Log-SARIMA(0,1,3)x(0,1,1)_12
```text
Log-SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 3
Seasonal order (P, D, Q, s): (0, 1, 1, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(0,1,3)x(0,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        198.533   AIC:       -387.065
BIC:                  -372.576   HQIC:      -381.177
Converged:                True   Run Time:   0.6267s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ma.L1               -0.9153     0.0737   -12.411     0.000     -1.060     -0.771
ma.L2                0.2200     0.1182     1.860     0.063     -0.012      0.452
ma.L3               -0.0870     0.1063    -0.818     0.413     -0.295      0.121
ma.S.L12            -0.6329     0.0656    -9.649     0.000     -0.761     -0.504
sigma2               0.0030     0.0003    10.273     0.000      0.002      0.004
==============================================================================
Ljung-Box (L=24) (Q):      38.88   Jarque-Bera (JB):      15.74
Prob(Q):                   0.0281   Prob(JB):             0.0004
Heteroskedasticity(H):       0.71   Skew:                  -0.41
Prob(H) (two-sided):       0.2518   Kurtosis:               4.36
Residual Mean:            -0.0031   Residual Std:         0.0566
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.1803 | RMSE = 1.4883 | MAPE = 6.37% | R² = 0.8703
==============================================================================
```

### 3.53. Log-SARIMA(0,1,3)x(1,1,0)_12
```text
Log-SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 3
Seasonal order (P, D, Q, s): (1, 1, 0, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(0,1,3)x(1,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        197.699   AIC:       -385.398
BIC:                  -370.762   HQIC:      -379.451
Converged:                True   Run Time:   0.3502s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ma.L1               -0.8985     0.0756   -11.891     0.000     -1.047     -0.750
ma.L2                0.1406     0.1252     1.123     0.261     -0.105      0.386
ma.L3               -0.0262     0.1021    -0.257     0.797     -0.226      0.174
ar.S.L12            -0.2081     0.0939    -2.217     0.027     -0.392     -0.024
sigma2               0.0033     0.0003    11.808     0.000      0.003      0.004
==============================================================================
Ljung-Box (L=24) (Q):      41.15   Jarque-Bera (JB):      29.97
Prob(Q):                   0.0160   Prob(JB):             0.0000
Heteroskedasticity(H):       0.77   Skew:                  -0.53
Prob(H) (two-sided):       0.3678   Kurtosis:               4.92
Residual Mean:            -0.0000   Residual Std:         0.0594
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.7239 | RMSE = 2.1794 | MAPE = 9.51% | R² = 0.7219
==============================================================================
```

### 3.54. Log-SARIMA(0,1,3)x(1,1,1)_12
```text
Log-SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 3
Seasonal order (P, D, Q, s): (1, 1, 1, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(0,1,3)x(1,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        199.401   AIC:       -386.803
BIC:                  -369.416   HQIC:      -379.737
Converged:                True   Run Time:   0.8663s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ma.L1               -0.8823     0.0906    -9.739     0.000     -1.060     -0.705
ma.L2                0.1671     0.1379     1.212     0.225     -0.103      0.437
ma.L3               -0.1165     0.1137    -1.024     0.306     -0.339      0.106
ar.S.L12             0.1907     0.1474     1.293     0.196     -0.098      0.480
ma.S.L12            -0.7487     0.1539    -4.864     0.000     -1.050     -0.447
sigma2               0.0028     0.0002    12.044     0.000      0.002      0.003
==============================================================================
Ljung-Box (L=24) (Q):      39.68   Jarque-Bera (JB):      72.45
Prob(Q):                   0.0231   Prob(JB):             0.0000
Heteroskedasticity(H):       0.78   Skew:                  -0.90
Prob(H) (two-sided):       0.4100   Kurtosis:               5.90
Residual Mean:            -0.0029   Residual Std:         0.0570
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.1940 | RMSE = 1.4975 | MAPE = 6.45% | R² = 0.8687
==============================================================================
```

### 3.55. Log-SARIMA(1,1,0)x(0,1,0)_12
```text
Log-SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 0
Seasonal order (P, D, Q, s): (0, 1, 0, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(1,1,0)x(0,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        193.995   AIC:       -383.989
BIC:                  -377.981   HQIC:      -381.548
Converged:                True   Run Time:   0.0634s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.5285     0.0731    -7.233     0.000     -0.672     -0.385
sigma2               0.0043     0.0004    10.615     0.000      0.004      0.005
==============================================================================
Ljung-Box (L=24) (Q):      66.27   Jarque-Bera (JB):      14.17
Prob(Q):                   0.0000   Prob(JB):             0.0008
Heteroskedasticity(H):       0.85   Skew:                  -0.42
Prob(H) (two-sided):       0.5648   Kurtosis:               4.24
Residual Mean:             0.0008   Residual Std:         0.0660
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.6088 | RMSE = 2.0377 | MAPE = 8.95% | R² = 0.7569
==============================================================================
```

### 3.56. Log-SARIMA(1,1,0)x(0,1,1)_12
```text
Log-SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 0
Seasonal order (P, D, Q, s): (0, 1, 1, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(1,1,0)x(0,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        187.478   AIC:       -368.956
BIC:                  -360.196   HQIC:      -365.396
Converged:                True   Run Time:   0.1871s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.5458     0.0767    -7.118     0.000     -0.696     -0.396
ma.S.L12            -0.6798     0.0633   -10.731     0.000     -0.804     -0.556
sigma2               0.0037     0.0004     9.193     0.000      0.003      0.005
==============================================================================
Ljung-Box (L=24) (Q):     103.72   Jarque-Bera (JB):       6.97
Prob(Q):                   0.0000   Prob(JB):             0.0307
Heteroskedasticity(H):       0.84   Skew:                  -0.45
Prob(H) (two-sided):       0.5669   Kurtosis:               3.56
Residual Mean:            -0.0015   Residual Std:         0.0646
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.2667 | RMSE = 1.5853 | MAPE = 6.52% | R² = 0.8528
==============================================================================
```

### 3.57. Log-SARIMA(1,1,0)x(1,1,0)_12
```text
Log-SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 0
Seasonal order (P, D, Q, s): (1, 1, 0, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(1,1,0)x(1,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        180.910   AIC:       -355.819
BIC:                  -347.059   HQIC:      -352.259
Converged:                True   Run Time:   0.1206s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.5403     0.0793    -6.813     0.000     -0.696     -0.385
ar.S.L12            -0.1415     0.0986    -1.435     0.151     -0.335      0.052
sigma2               0.0042     0.0004     9.589     0.000      0.003      0.005
==============================================================================
Ljung-Box (L=24) (Q):      84.01   Jarque-Bera (JB):      10.98
Prob(Q):                   0.0000   Prob(JB):             0.0041
Heteroskedasticity(H):       0.82   Skew:                  -0.48
Prob(H) (two-sided):       0.5045   Kurtosis:               3.92
Residual Mean:            -0.0008   Residual Std:         0.0676
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.3436 | RMSE = 1.7093 | MAPE = 7.48% | R² = 0.8289
==============================================================================
```

### 3.58. Log-SARIMA(1,1,0)x(1,1,1)_12
```text
Log-SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 0
Seasonal order (P, D, Q, s): (1, 1, 1, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(1,1,0)x(1,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        189.770   AIC:       -371.540
BIC:                  -359.861   HQIC:      -366.794
Converged:                True   Run Time:   0.3231s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.5288     0.0796    -6.640     0.000     -0.685     -0.373
ar.S.L12             0.2162     0.1386     1.560     0.119     -0.055      0.488
ma.S.L12            -0.7603     0.1377    -5.521     0.000     -1.030     -0.490
sigma2               0.0034     0.0003    11.228     0.000      0.003      0.004
==============================================================================
Ljung-Box (L=24) (Q):      69.43   Jarque-Bera (JB):      31.51
Prob(Q):                   0.0000   Prob(JB):             0.0000
Heteroskedasticity(H):       0.86   Skew:                  -0.69
Prob(H) (two-sided):       0.6035   Kurtosis:               4.78
Residual Mean:            -0.0021   Residual Std:         0.0637
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.2244 | RMSE = 1.5169 | MAPE = 6.43% | R² = 0.8653
==============================================================================
```

### 3.59. Log-SARIMA(1,1,1)x(0,1,0)_12
```text
Log-SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 1
Seasonal order (P, D, Q, s): (0, 1, 0, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(1,1,1)x(0,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        207.121   AIC:       -408.242
BIC:                  -399.250   HQIC:      -404.589
Converged:                True   Run Time:   0.0593s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.1044     0.1080    -0.967     0.333     -0.316      0.107
ma.L1               -0.7298     0.0844    -8.648     0.000     -0.895     -0.564
sigma2               0.0036     0.0003    12.455     0.000      0.003      0.004
==============================================================================
Ljung-Box (L=24) (Q):      35.87   Jarque-Bera (JB):      42.26
Prob(Q):                   0.0564   Prob(JB):             0.0000
Heteroskedasticity(H):       0.73   Skew:                  -0.52
Prob(H) (two-sided):       0.2718   Kurtosis:               5.38
Residual Mean:             0.0021   Residual Std:         0.0599
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 2.6788 | RMSE = 3.1419 | MAPE = 14.33% | R² = 0.4220
==============================================================================
```

### 3.60. Log-SARIMA(1,1,1)x(0,1,1)_12
```text
Log-SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 1
Seasonal order (P, D, Q, s): (0, 1, 1, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(1,1,1)x(0,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        202.283   AIC:       -396.567
BIC:                  -384.916   HQIC:      -391.832
Converged:                True   Run Time:   0.4414s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.1840     0.0979    -1.878     0.060     -0.376      0.008
ma.L1               -0.7325     0.0737    -9.945     0.000     -0.877     -0.588
ma.S.L12            -0.6231     0.0633    -9.838     0.000     -0.747     -0.499
sigma2               0.0029     0.0003    10.420     0.000      0.002      0.004
==============================================================================
Ljung-Box (L=24) (Q):      40.93   Jarque-Bera (JB):      19.90
Prob(Q):                   0.0170   Prob(JB):             0.0000
Heteroskedasticity(H):       0.71   Skew:                  -0.42
Prob(H) (two-sided):       0.2514   Kurtosis:               4.57
Residual Mean:            -0.0022   Residual Std:         0.0556
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.1795 | RMSE = 1.4918 | MAPE = 6.40% | R² = 0.8697
==============================================================================
```

### 3.61. Log-SARIMA(1,1,1)x(1,1,0)_12
```text
Log-SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 1
Seasonal order (P, D, Q, s): (1, 1, 0, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(1,1,1)x(1,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        197.146   AIC:       -386.293
BIC:                  -374.613   HQIC:      -381.546
Converged:                True   Run Time:   0.1887s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.1354     0.1030    -1.314     0.189     -0.337      0.067
ma.L1               -0.7531     0.0790    -9.527     0.000     -0.908     -0.598
ar.S.L12            -0.2145     0.0935    -2.295     0.022     -0.398     -0.031
sigma2               0.0033     0.0003    11.877     0.000      0.003      0.004
==============================================================================
Ljung-Box (L=24) (Q):      46.31   Jarque-Bera (JB):      30.32
Prob(Q):                   0.0041   Prob(JB):             0.0000
Heteroskedasticity(H):       0.78   Skew:                  -0.57
Prob(H) (two-sided):       0.3986   Kurtosis:               4.88
Residual Mean:            -0.0008   Residual Std:         0.0604
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.7180 | RMSE = 2.1715 | MAPE = 9.48% | R² = 0.7239
==============================================================================
```

### 3.62. Log-SARIMA(1,1,1)x(1,1,1)_12
```text
Log-SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 1
Seasonal order (P, D, Q, s): (1, 1, 1, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(1,1,1)x(1,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        201.992   AIC:       -393.985
BIC:                  -379.422   HQIC:      -388.067
Converged:                True   Run Time:   0.4022s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.0683     0.1031    -0.662     0.508     -0.270      0.134
ma.L1               -0.7873     0.0774   -10.166     0.000     -0.939     -0.636
ar.S.L12             0.1771     0.1391     1.273     0.203     -0.096      0.450
ma.S.L12            -0.7362     0.1390    -5.297     0.000     -1.009     -0.464
sigma2               0.0028     0.0002    12.024     0.000      0.002      0.003
==============================================================================
Ljung-Box (L=24) (Q):      39.96   Jarque-Bera (JB):      69.53
Prob(Q):                   0.0216   Prob(JB):             0.0000
Heteroskedasticity(H):       0.74   Skew:                  -0.87
Prob(H) (two-sided):       0.3206   Kurtosis:               5.85
Residual Mean:            -0.0028   Residual Std:         0.0569
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.2121 | RMSE = 1.5195 | MAPE = 6.62% | R² = 0.8648
==============================================================================
```

### 3.63. Log-SARIMA(1,1,3)x(0,1,0)_12
```text
Log-SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 3
Seasonal order (P, D, Q, s): (0, 1, 0, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(1,1,3)x(0,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        205.720   AIC:       -401.440
BIC:                  -386.522   HQIC:      -395.379
Converged:                True   Run Time:   0.2955s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.8237     0.2010    -4.098     0.000     -1.218     -0.430
ma.L1                0.0200     0.2334     0.086     0.932     -0.438      0.478
ma.L2               -0.6712     0.1446    -4.642     0.000     -0.955     -0.388
ma.L3                0.1105     0.0947     1.167     0.243     -0.075      0.296
sigma2               0.0035     0.0003    12.113     0.000      0.003      0.004
==============================================================================
Ljung-Box (L=24) (Q):      32.02   Jarque-Bera (JB):      34.70
Prob(Q):                   0.1266   Prob(JB):             0.0000
Heteroskedasticity(H):       0.77   Skew:                  -0.40
Prob(H) (two-sided):       0.3681   Kurtosis:               5.22
Residual Mean:             0.0023   Residual Std:         0.0599
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 2.7592 | RMSE = 3.2221 | MAPE = 14.74% | R² = 0.3921
==============================================================================
```

### 3.64. Log-SARIMA(1,1,3)x(0,1,1)_12
```text
Log-SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 3
Seasonal order (P, D, Q, s): (0, 1, 1, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(1,1,3)x(0,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        198.739   AIC:       -385.478
BIC:                  -368.091   HQIC:      -378.413
Converged:                True   Run Time:   0.8590s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.6567     0.7972    -0.824     0.410     -2.219      0.906
ma.L1               -0.2256     0.8085    -0.279     0.780     -1.810      1.359
ma.L2               -0.5087     0.6796    -0.748     0.454     -1.841      0.823
ma.L3                0.1261     0.0983     1.283     0.200     -0.067      0.319
ma.S.L12            -0.6135     0.0702    -8.740     0.000     -0.751     -0.476
sigma2               0.0030     0.0003    10.119     0.000      0.002      0.004
==============================================================================
Ljung-Box (L=24) (Q):      39.76   Jarque-Bera (JB):      47.40
Prob(Q):                   0.0227   Prob(JB):             0.0000
Heteroskedasticity(H):       0.69   Skew:                  -0.65
Prob(H) (two-sided):       0.2117   Kurtosis:               5.42
Residual Mean:            -0.0038   Residual Std:         0.0596
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.1850 | RMSE = 1.4967 | MAPE = 6.46% | R² = 0.8688
==============================================================================
```

### 3.65. Log-SARIMA(1,1,3)x(1,1,0)_12
```text
Log-SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 3
Seasonal order (P, D, Q, s): (1, 1, 0, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(1,1,3)x(1,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        197.618   AIC:       -383.236
BIC:                  -365.716   HQIC:      -376.117
Converged:                True   Run Time:   0.3462s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.5220     0.3458    -1.510     0.131     -1.200      0.156
ma.L1               -0.3435     0.3562    -0.964     0.335     -1.042      0.355
ma.L2               -0.4322     0.2935    -1.472     0.141     -1.007      0.143
ma.L3                0.1276     0.0874     1.460     0.144     -0.044      0.299
ar.S.L12            -0.2280     0.0941    -2.421     0.015     -0.412     -0.043
sigma2               0.0033     0.0003    11.822     0.000      0.003      0.004
==============================================================================
Ljung-Box (L=24) (Q):      44.04   Jarque-Bera (JB):      28.38
Prob(Q):                   0.0075   Prob(JB):             0.0000
Heteroskedasticity(H):       0.76   Skew:                  -0.54
Prob(H) (two-sided):       0.3578   Kurtosis:               4.84
Residual Mean:            -0.0013   Residual Std:         0.0613
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.7780 | RMSE = 2.2517 | MAPE = 9.81% | R² = 0.7031
==============================================================================
```

### 3.66. Log-SARIMA(1,1,3)x(1,1,1)_12
```text
Log-SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 1 | Differencing order (d): 1 | MA order (q): 3
Seasonal order (P, D, Q, s): (1, 1, 1, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(1,1,3)x(1,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        198.551   AIC:       -383.102
BIC:                  -362.817   HQIC:      -374.859
Converged:                True   Run Time:   1.0101s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.6849     0.4695    -1.459     0.145     -1.605      0.235
ma.L1               -0.1584     0.4639    -0.342     0.733     -1.068      0.751
ma.L2               -0.5567     0.3959    -1.406     0.160     -1.333      0.219
ma.L3                0.0644     0.0924     0.697     0.486     -0.117      0.246
ar.S.L12             0.1738     0.1499     1.160     0.246     -0.120      0.468
ma.S.L12            -0.7221     0.1435    -5.033     0.000     -1.003     -0.441
sigma2               0.0029     0.0003    11.102     0.000      0.002      0.003
==============================================================================
Ljung-Box (L=24) (Q):      35.59   Jarque-Bera (JB):     109.72
Prob(Q):                   0.0601   Prob(JB):             0.0000
Heteroskedasticity(H):       0.73   Skew:                  -1.04
Prob(H) (two-sided):       0.2862   Kurtosis:               6.64
Residual Mean:            -0.0047   Residual Std:         0.0607
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.2176 | RMSE = 1.5274 | MAPE = 6.67% | R² = 0.8634
==============================================================================
```

### 3.67. Log-SARIMA(2,1,0)x(0,1,0)_12
```text
Log-SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 2 | Differencing order (d): 1 | MA order (q): 0
Seasonal order (P, D, Q, s): (0, 1, 0, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(2,1,0)x(0,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        204.635   AIC:       -403.271
BIC:                  -394.279   HQIC:      -399.617
Converged:                True   Run Time:   0.0633s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.7378     0.0807    -9.145     0.000     -0.896     -0.580
ar.L2               -0.3881     0.0740    -5.248     0.000     -0.533     -0.243
sigma2               0.0037     0.0003    11.905     0.000      0.003      0.004
==============================================================================
Ljung-Box (L=24) (Q):      36.36   Jarque-Bera (JB):      29.32
Prob(Q):                   0.0506   Prob(JB):             0.0000
Heteroskedasticity(H):       0.71   Skew:                  -0.42
Prob(H) (two-sided):       0.2255   Kurtosis:               5.00
Residual Mean:             0.0012   Residual Std:         0.0607
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 2.5368 | RMSE = 2.9873 | MAPE = 13.67% | R² = 0.4775
==============================================================================
```

### 3.68. Log-SARIMA(2,1,0)x(0,1,1)_12
```text
Log-SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 2 | Differencing order (d): 1 | MA order (q): 0
Seasonal order (P, D, Q, s): (0, 1, 1, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(2,1,0)x(0,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        201.877   AIC:       -395.754
BIC:                  -384.074   HQIC:      -391.007
Converged:                True   Run Time:   0.3556s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.8013     0.0737   -10.879     0.000     -0.946     -0.657
ar.L2               -0.4343     0.0751    -5.784     0.000     -0.581     -0.287
ma.S.L12            -0.6145     0.0676    -9.092     0.000     -0.747     -0.482
sigma2               0.0030     0.0003    10.286     0.000      0.002      0.004
==============================================================================
Ljung-Box (L=24) (Q):      29.52   Jarque-Bera (JB):      41.36
Prob(Q):                   0.2012   Prob(JB):             0.0000
Heteroskedasticity(H):       0.69   Skew:                  -0.43
Prob(H) (two-sided):       0.2153   Kurtosis:               5.43
Residual Mean:            -0.0023   Residual Std:         0.0593
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.1781 | RMSE = 1.4839 | MAPE = 6.39% | R² = 0.8711
==============================================================================
```

### 3.69. Log-SARIMA(2,1,0)x(1,1,0)_12
```text
Log-SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 2 | Differencing order (d): 1 | MA order (q): 0
Seasonal order (P, D, Q, s): (1, 1, 0, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(2,1,0)x(1,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        192.040   AIC:       -376.080
BIC:                  -364.430   HQIC:      -371.346
Converged:                True   Run Time:   0.2554s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.7788     0.0778   -10.011     0.000     -0.931     -0.626
ar.L2               -0.4276     0.0762    -5.615     0.000     -0.577     -0.278
ar.S.L12            -0.2509     0.0928    -2.703     0.007     -0.433     -0.069
sigma2               0.0035     0.0003    11.197     0.000      0.003      0.004
==============================================================================
Ljung-Box (L=24) (Q):      38.22   Jarque-Bera (JB):      39.24
Prob(Q):                   0.0329   Prob(JB):             0.0000
Heteroskedasticity(H):       0.71   Skew:                  -0.59
Prob(H) (two-sided):       0.2556   Kurtosis:               5.21
Residual Mean:            -0.0024   Residual Std:         0.0646
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.6489 | RMSE = 2.0766 | MAPE = 9.13% | R² = 0.7475
==============================================================================
```

### 3.70. Log-SARIMA(2,1,0)x(1,1,1)_12
```text
Log-SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 2 | Differencing order (d): 1 | MA order (q): 0
Seasonal order (P, D, Q, s): (1, 1, 1, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(2,1,0)x(1,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        197.736   AIC:       -385.471
BIC:                  -370.908   HQIC:      -379.553
Converged:                True   Run Time:   0.9001s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.7437     0.0863    -8.614     0.000     -0.913     -0.574
ar.L2               -0.3684     0.0709    -5.199     0.000     -0.507     -0.230
ar.S.L12             0.1325     0.1503     0.882     0.378     -0.162      0.427
ma.S.L12            -0.6748     0.1447    -4.663     0.000     -0.958     -0.391
sigma2               0.0030     0.0003    11.274     0.000      0.003      0.004
==============================================================================
Ljung-Box (L=24) (Q):      22.34   Jarque-Bera (JB):      56.42
Prob(Q):                   0.5592   Prob(JB):             0.0000
Heteroskedasticity(H):       0.68   Skew:                  -0.64
Prob(H) (two-sided):       0.1979   Kurtosis:               5.72
Residual Mean:            -0.0037   Residual Std:         0.0599
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.1816 | RMSE = 1.4934 | MAPE = 6.45% | R² = 0.8694
==============================================================================
```

### 3.71. Log-SARIMA(2,1,1)x(0,1,0)_12
```text
Log-SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 2 | Differencing order (d): 1 | MA order (q): 1
Seasonal order (P, D, Q, s): (0, 1, 0, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(2,1,1)x(0,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        207.122   AIC:       -406.243
BIC:                  -394.254   HQIC:      -401.372
Converged:                True   Run Time:   0.1457s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.1077     0.1362    -0.791     0.429     -0.375      0.159
ar.L2               -0.0040     0.1275    -0.032     0.975     -0.254      0.246
ma.L1               -0.7268     0.1144    -6.351     0.000     -0.951     -0.502
sigma2               0.0036     0.0003    12.403     0.000      0.003      0.004
==============================================================================
Ljung-Box (L=24) (Q):      35.78   Jarque-Bera (JB):      42.15
Prob(Q):                   0.0576   Prob(JB):             0.0000
Heteroskedasticity(H):       0.73   Skew:                  -0.52
Prob(H) (two-sided):       0.2700   Kurtosis:               5.38
Residual Mean:             0.0021   Residual Std:         0.0599
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 2.6865 | RMSE = 3.1495 | MAPE = 14.37% | R² = 0.4192
==============================================================================
```

### 3.72. Log-SARIMA(2,1,1)x(0,1,1)_12
```text
Log-SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 2 | Differencing order (d): 1 | MA order (q): 1
Seasonal order (P, D, Q, s): (0, 1, 1, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(2,1,1)x(0,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        202.012   AIC:       -394.025
BIC:                  -379.462   HQIC:      -388.107
Converged:                True   Run Time:   0.7146s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.2558     0.1218    -2.101     0.036     -0.494     -0.017
ar.L2               -0.1026     0.1214    -0.845     0.398     -0.340      0.135
ma.L1               -0.6570     0.1012    -6.493     0.000     -0.855     -0.459
ma.S.L12            -0.6120     0.0657    -9.310     0.000     -0.741     -0.483
sigma2               0.0030     0.0003    10.327     0.000      0.002      0.004
==============================================================================
Ljung-Box (L=24) (Q):      35.12   Jarque-Bera (JB):      16.94
Prob(Q):                   0.0666   Prob(JB):             0.0002
Heteroskedasticity(H):       0.67   Skew:                  -0.37
Prob(H) (two-sided):       0.1820   Kurtosis:               4.47
Residual Mean:            -0.0024   Residual Std:         0.0557
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.1865 | RMSE = 1.4988 | MAPE = 6.47% | R² = 0.8685
==============================================================================
```

### 3.73. Log-SARIMA(2,1,1)x(1,1,0)_12
```text
Log-SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 2 | Differencing order (d): 1 | MA order (q): 1
Seasonal order (P, D, Q, s): (1, 1, 0, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(2,1,1)x(1,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        195.480   AIC:       -380.959
BIC:                  -366.396   HQIC:      -375.041
Converged:                True   Run Time:   0.5082s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.1754     0.1214    -1.445     0.149     -0.413      0.063
ar.L2               -0.0527     0.1270    -0.415     0.678     -0.302      0.196
ma.L1               -0.7221     0.1017    -7.099     0.000     -0.922     -0.523
ar.S.L12            -0.2247     0.0937    -2.399     0.016     -0.408     -0.041
sigma2               0.0033     0.0003    11.879     0.000      0.003      0.004
==============================================================================
Ljung-Box (L=24) (Q):      43.51   Jarque-Bera (JB):      32.85
Prob(Q):                   0.0087   Prob(JB):             0.0000
Heteroskedasticity(H):       0.76   Skew:                  -0.57
Prob(H) (two-sided):       0.3602   Kurtosis:               4.98
Residual Mean:            -0.0010   Residual Std:         0.0606
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.7464 | RMSE = 2.2080 | MAPE = 9.63% | R² = 0.7145
==============================================================================
```

### 3.74. Log-SARIMA(2,1,1)x(1,1,1)_12
```text
Log-SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 2 | Differencing order (d): 1 | MA order (q): 1
Seasonal order (P, D, Q, s): (1, 1, 1, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(2,1,1)x(1,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        201.586   AIC:       -391.172
BIC:                  -373.697   HQIC:      -384.071
Converged:               False   Run Time:   0.9163s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1                0.1268     0.0919     1.379     0.168     -0.053      0.307
ar.L2                0.2024     0.0666     3.040     0.002      0.072      0.333
ma.L1               -1.0363     0.0419   -24.724     0.000     -1.118     -0.954
ar.S.L12             0.3039     0.1102     2.756     0.006      0.088      0.520
ma.S.L12            -1.0788     0.2904    -3.714     0.000     -1.648     -0.510
sigma2               0.0020     0.0008     2.501     0.012      0.000      0.004
==============================================================================
Ljung-Box (L=24) (Q):      51.33   Jarque-Bera (JB):      62.76
Prob(Q):                   0.0010   Prob(JB):             0.0000
Heteroskedasticity(H):       0.94   Skew:                  -0.81
Prob(H) (two-sided):       0.8359   Kurtosis:               5.73
Residual Mean:            -0.0004   Residual Std:         0.0596
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.3134 | RMSE = 1.5758 | MAPE = 6.79% | R² = 0.8546
==============================================================================
Warnings:
  [1] Maximum Likelihood optimization failed to converge. Check mle_retvals
==============================================================================
```

### 3.75. Log-SARIMA(2,1,3)x(0,1,0)_12
```text
Log-SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 2 | Differencing order (d): 1 | MA order (q): 3
Seasonal order (P, D, Q, s): (0, 1, 0, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(2,1,3)x(0,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        213.812   AIC:       -415.623
BIC:                  -397.722   HQIC:      -408.349
Converged:                True   Run Time:   0.5283s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -1.1600     0.0107  -108.091     0.000     -1.181     -1.139
ar.L2               -0.9848     0.0128   -77.231     0.000     -1.010     -0.960
ma.L1                0.4065    24.5842     0.017     0.987    -47.778     48.591
ma.L2                0.1644    25.1305     0.007     0.995    -49.090     49.419
ma.L3               -0.7332    31.6265    -0.023     0.982    -62.720     61.254
sigma2               0.0030     0.1287     0.023     0.982     -0.249      0.255
==============================================================================
Ljung-Box (L=24) (Q):      26.30   Jarque-Bera (JB):      18.44
Prob(Q):                   0.3381   Prob(JB):             0.0001
Heteroskedasticity(H):       0.90   Skew:                  -0.34
Prob(H) (two-sided):       0.7040   Kurtosis:               4.58
Residual Mean:             0.0018   Residual Std:         0.0575
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 2.7467 | RMSE = 3.2480 | MAPE = 14.64% | R² = 0.3823
==============================================================================
```

### 3.76. Log-SARIMA(2,1,3)x(0,1,1)_12
```text
Log-SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 2 | Differencing order (d): 1 | MA order (q): 3
Seasonal order (P, D, Q, s): (0, 1, 1, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(2,1,3)x(0,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        208.388   AIC:       -402.777
BIC:                  -382.492   HQIC:      -394.533
Converged:                True   Run Time:   1.2709s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -1.1603     0.0086  -135.019     0.000     -1.177     -1.143
ar.L2               -0.9848     0.0123   -79.858     0.000     -1.009     -0.961
ma.L1                0.4073    69.2532     0.006     0.995   -135.327    136.141
ma.L2                0.0918    64.1699     0.001     0.999   -125.679    125.862
ma.L3               -0.7709    90.6283    -0.009     0.993   -178.399    176.857
ma.S.L12            -0.6007     0.0642    -9.352     0.000     -0.727     -0.475
sigma2               0.0025     0.2929     0.009     0.993     -0.572      0.577
==============================================================================
Ljung-Box (L=24) (Q):      17.51   Jarque-Bera (JB):     309.19
Prob(Q):                   0.8259   Prob(JB):             0.0000
Heteroskedasticity(H):       0.84   Skew:                  -1.54
Prob(H) (two-sided):       0.5639   Kurtosis:               9.32
Residual Mean:            -0.0059   Residual Std:         0.0624
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.1991 | RMSE = 1.4982 | MAPE = 6.53% | R² = 0.8686
==============================================================================
```

### 3.77. Log-SARIMA(2,1,3)x(1,1,0)_12
```text
Log-SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 2 | Differencing order (d): 1 | MA order (q): 3
Seasonal order (P, D, Q, s): (1, 1, 0, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(2,1,3)x(1,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        204.176   AIC:       -394.352
BIC:                  -373.964   HQIC:      -386.067
Converged:               False   Run Time:   1.3384s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -1.1633     0.0090  -129.350     0.000     -1.181     -1.146
ar.L2               -0.9881     0.0132   -74.646     0.000     -1.014     -0.962
ma.L1                0.3901    19.5954     0.020     0.984    -38.016     38.796
ma.L2                0.1094    18.7567     0.006     0.995    -36.653     36.872
ma.L3               -0.7687    25.9880    -0.030     0.976    -51.704     50.167
ar.S.L12            -0.3003     0.0807    -3.721     0.000     -0.458     -0.142
sigma2               0.0027     0.0927     0.030     0.976     -0.179      0.184
==============================================================================
Ljung-Box (L=24) (Q):      23.30   Jarque-Bera (JB):     307.65
Prob(Q):                   0.5022   Prob(JB):             0.0000
Heteroskedasticity(H):       0.92   Skew:                  -1.58
Prob(H) (two-sided):       0.7723   Kurtosis:               9.27
Residual Mean:            -0.0044   Residual Std:         0.0662
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.5670 | RMSE = 1.9921 | MAPE = 8.66% | R² = 0.7676
==============================================================================
Warnings:
  [1] Maximum Likelihood optimization failed to converge. Check mle_retvals
==============================================================================
```

### 3.78. Log-SARIMA(2,1,3)x(1,1,1)_12
```text
Log-SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261112_082025_d2a40e | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 2 | Differencing order (d): 1 | MA order (q): 3
Seasonal order (P, D, Q, s): (1, 1, 1, 12)
Transformation: Log Transformation ln(Y_t) + Inverse exp() | d=1, D=1, s=12
Model: Log-SARIMA(2,1,3)x(1,1,1)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:        205.826   AIC:       -395.653
BIC:                  -372.470   HQIC:      -386.232
Converged:               False   Run Time:   1.3248s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -1.1596     0.0121   -95.675     0.000     -1.183     -1.136
ar.L2               -0.9888     0.0151   -65.432     0.000     -1.018     -0.959
ma.L1                0.3888     1.4415     0.270     0.787     -2.436      3.214
ma.L2                0.0893     1.3272     0.067     0.946     -2.512      2.690
ma.L3               -0.7806     1.9235    -0.406     0.685     -4.551      2.990
ar.S.L12             0.1468     0.1566     0.938     0.348     -0.160      0.454
ma.S.L12            -0.6905     0.1419    -4.865     0.000     -0.969     -0.412
sigma2               0.0024     0.0060     0.404     0.686     -0.009      0.014
==============================================================================
Ljung-Box (L=24) (Q):      17.63   Jarque-Bera (JB):     322.79
Prob(Q):                   0.8207   Prob(JB):             0.0000
Heteroskedasticity(H):       0.83   Skew:                  -1.70
Prob(H) (two-sided):       0.5389   Kurtosis:               9.33
Residual Mean:            -0.0067   Residual Std:         0.0630
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.2473 | RMSE = 1.5390 | MAPE = 6.83% | R² = 0.8613
==============================================================================
Warnings:
  [1] Maximum Likelihood optimization failed to converge. Check mle_retvals
==============================================================================
```
