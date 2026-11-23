# BÁO CÁO TỔNG HỢP MÔ HÌNH (MODEL SUMMARY) — MONTHLY RETAIL SALES (US, 07/1991 - 06/2008)

## 1. Thông tin Phiên Thực nghiệm & Bộ dữ liệu (Provenance & Dataset Info)
- **Run ID**: `dataset1_20261123_154923_d2775a`
- **Thời điểm xuất báo cáo**: `2026-11-23 15:49:33`
- **Dataset Key**: `dataset1`
- **Tên bộ dữ liệu**: Monthly Retail Sales (US, 07/1991 - 06/2008)
- **Tổng số quan sát**: 204 tháng (1991-07-01 to 2008-06-01)
- **Tần suất**: `MS` (Monthly Start)
- **Tập Huấn luyện (Train - 80%)**: 163 quan sát
- **Tập Kiểm thử (Test - 20%)**: 41 quan sát
- **Tổng số mô hình ứng viên**: 27 (Thành công: 27, Thất bại: 0)

## 2. Bảng So sánh Tổng hợp các Mô hình Ứng viên (Model Comparison Table)

> **Nguyên tắc đối chiếu**: Tất cả chỉ số dự báo ngoài mẫu (`MAE`, `RMSE`, `MAPE (%)`, `R²`) đều được đánh giá trên **cùng tập Test và cùng thang đo gốc** (đã áp dụng biến đổi ngược $\exp(\cdot)$ cho các mô hình `Log-SARIMA`). Chỉ số `AIC` và `BIC` chỉ được so sánh trực tiếp trong cùng một nhóm `Scale_Group`.

| Model                        | Family     | Status   | Scale_Group        |     AIC |     BIC |   MAE |   RMSE |   MAPE (%) |      R² |   Ljung-Box p | White Noise      | Converged   |
|:-----------------------------|:-----------|:---------|:-------------------|--------:|--------:|------:|-------:|-----------:|--------:|--------------:|:-----------------|:------------|
| Log-SARIMA(1,1,1)x(0,1,1)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -396.57 | -384.92 |  1.18 |   1.49 |       6.4  |  0.8697 |        0.017  | Fail (Không đạt) | True        |
| Log-SARIMA(0,1,1)x(0,1,1)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -395.54 | -386.8  |  1.19 |   1.49 |       6.38 |  0.8698 |        0.0039 | Fail (Không đạt) | True        |
| Log-SARIMA(2,1,3)x(0,1,1)_12 | Log-SARIMA | SUCCESS  | log_d1_D1_s12      | -402.78 | -382.49 |  1.2  |   1.5  |       6.53 |  0.8686 |        0.8259 | Pass (Đạt)       | False       |
| SARIMA(0,1,1)x(0,1,1)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  233.85 |  242.59 |  1.45 |   1.84 |       8.21 |  0.8012 |        0.1053 | Pass (Đạt)       | True        |
| SARIMA(2,1,3)x(0,1,1)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  223.07 |  243.35 |  1.48 |   1.87 |       8.38 |  0.7956 |        0.9712 | Pass (Đạt)       | True        |
| SARIMA(1,1,0)x(0,1,1)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  258.39 |  267.15 |  1.47 |   1.91 |       8.48 |  0.7866 |        0      | Fail (Không đạt) | True        |
| SARIMA(1,1,1)x(0,1,1)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  229.45 |  241.1  |  1.47 |   1.91 |       8.48 |  0.7862 |        0.4558 | Pass (Đạt)       | True        |
| SARIMA(0,1,3)x(0,1,1)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  229.34 |  243.83 |  1.5  |   1.95 |       8.72 |  0.7767 |        0.6242 | Pass (Đạt)       | True        |
| SARIMA(2,1,1)x(0,1,1)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  229.46 |  244.02 |  1.54 |   2    |       8.97 |  0.7666 |        0.7873 | Pass (Đạt)       | True        |
| SARIMA(1,1,3)x(0,1,1)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  230.78 |  248.17 |  1.54 |   2    |       8.98 |  0.7661 |        0.6654 | Pass (Đạt)       | True        |
| SARIMA(0,1,0)x(0,1,1)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  313.37 |  319.21 |  1.57 |   2.02 |       9.12 |  0.7606 |        0      | Fail (Không đạt) | True        |
| SARIMA(2,1,0)x(0,1,1)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  230.05 |  241.73 |  1.59 |   2.06 |       9.32 |  0.7527 |        0.7484 | Pass (Đạt)       | True        |
| SARIMA(1,1,0)x(0,1,0)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  278.02 |  284.02 |  1.85 |   2.32 |      10.77 |  0.6835 |        0.0001 | Fail (Không đạt) | False       |
| SARIMA(0,1,1)x(0,1,0)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  251.6  |  257.59 |  1.87 |   2.34 |      10.87 |  0.6787 |        0.0654 | Pass (Đạt)       | True        |
| SARIMA(1,1,1)x(0,1,0)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  248.92 |  257.91 |  2.02 |   2.5  |      11.71 |  0.6353 |        0.2564 | Pass (Đạt)       | True        |
| SARIMA(2,1,3)x(0,1,0)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  246.42 |  264.32 |  2.08 |   2.55 |      12.04 |  0.6186 |        0.6411 | Pass (Đạt)       | True        |
| SARIMA(0,1,3)x(0,1,0)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  248.47 |  260.41 |  2.09 |   2.57 |      12.1  |  0.6135 |        0.2871 | Pass (Đạt)       | True        |
| SARIMA(0,1,0)x(0,1,0)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  333.47 |  336.48 |  2.11 |   2.59 |      12.21 |  0.6072 |        0      | Fail (Không đạt) | True        |
| SARIMA(2,1,1)x(0,1,0)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  249.82 |  261.81 |  2.14 |   2.62 |      12.38 |  0.5974 |        0.3366 | Pass (Đạt)       | True        |
| SARIMA(1,1,3)x(0,1,0)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  249.43 |  264.34 |  2.16 |   2.64 |      12.45 |  0.5927 |        0.3919 | Pass (Đạt)       | True        |
| SARIMA(2,1,0)x(0,1,0)_12     | SARIMA     | SUCCESS  | original_d1_D1_s12 |  250.4  |  259.39 |  2.27 |   2.75 |      13.07 |  0.5558 |        0.2134 | Pass (Đạt)       | True        |
| MA(1) [ARIMA(0,1,1)]         | MA         | SUCCESS  | original_d1        |  587.92 |  594.09 |  3.46 |   4.15 |      18.39 | -0.0068 |        0      | Fail (Không đạt) | True        |
| AR(2) [ARIMA(2,1,0)]         | AR         | SUCCESS  | original_d1        |  593.58 |  602.84 |  3.5  |   4.15 |      19.29 | -0.0063 |        0      | Fail (Không đạt) | True        |
| AR(1) [ARIMA(1,1,0)]         | AR         | SUCCESS  | original_d1        |  597.93 |  604.1  |  3.56 |   4.21 |      19.99 | -0.0358 |        0      | Fail (Không đạt) | True        |
| MA(3) [ARIMA(0,1,3)]         | MA         | SUCCESS  | original_d1        |  584.61 |  596.96 |  3.64 |   4.59 |      17.96 | -0.2352 |        0      | Fail (Không đạt) | True        |
| ARIMA(1,1,1)                 | ARIMA      | SUCCESS  | original_d1        |  582.47 |  591.73 |  3.67 |   4.62 |      18.04 | -0.252  |        0      | Fail (Không đạt) | True        |
| ARIMA(2,1,1)                 | ARIMA      | SUCCESS  | original_d1        |  584.46 |  596.81 |  3.67 |   4.63 |      18.06 | -0.2543 |        0      | Fail (Không đạt) | True        |

### Diễn giải Đa Tiêu chí Tự động (Run ID: `dataset1_20261123_154923_d2775a` — Monthly Retail Sales (US, 07/1991 - 06/2008))

#### A. Mô hình Tốt nhất theo Từng Tiêu chí Độc lập (Criterion-Specific Leaders)
- **Tiêu chí 1 — Sai số RMSE trên tập Test thấp nhất (Thang đo gốc)**: **Log-SARIMA(1,1,1)x(0,1,1)_12** (`RMSE = 1.49`, `R² = 0.8697`).
- **Tiêu chí 2 — Sai số tuyệt đối MAE trên tập Test thấp nhất**: **Log-SARIMA(1,1,1)x(0,1,1)_12** (`MAE = 1.18`).
- **Tiêu chí 3 — Sai số phần trăm MAPE (%) trên tập Test thấp nhất**: **Log-SARIMA(0,1,1)x(0,1,1)_12** (`MAPE = 6.38%`).
- **Tiêu chí 4 — Mô hình thỏa mãn đồng thời Chuẩn Nhiễu trắng (Ljung-Box $p \ge 0.05$), Hội tụ và có RMSE thấp nhất**: **SARIMA(0,1,1)x(0,1,1)_12**.
- **Tiêu chí 5 — Tiêu chuẩn Thông tin (AIC / BIC) trên Tập Train theo từng nhóm thang đo hợp lệ (`Scale_Group`)**:
  - Nhóm `log_d1_D1_s12`: AIC thấp nhất là **Log-SARIMA(2,1,3)x(0,1,1)_12** (`AIC = -402.78`, `Test RMSE = 1.50`); BIC thấp nhất là **Log-SARIMA(0,1,1)x(0,1,1)_12** (`BIC = -386.80`).
  - Nhóm `original_d1`: AIC thấp nhất là **ARIMA(1,1,1)** (`AIC = 582.47`, `Test RMSE = 4.62`); BIC thấp nhất là **ARIMA(1,1,1)** (`BIC = 591.73`).
  - Nhóm `original_d1_D1_s12`: AIC thấp nhất là **SARIMA(2,1,3)x(0,1,1)_12** (`AIC = 223.07`, `Test RMSE = 1.87`); BIC thấp nhất là **SARIMA(1,1,1)x(0,1,1)_12** (`BIC = 241.10`).

#### B. Kiểm định Phần dư & Trạng thái Hội tụ
- **Các mô hình vượt qua kiểm định Ljung-Box ($p \ge 0.05$, phần dư đạt chuẩn White Noise)**: `Log-SARIMA(2,1,3)x(0,1,1)_12`, `SARIMA(0,1,1)x(0,1,1)_12`, `SARIMA(2,1,3)x(0,1,1)_12`, `SARIMA(1,1,1)x(0,1,1)_12`, `SARIMA(0,1,3)x(0,1,1)_12`, `SARIMA(2,1,1)x(0,1,1)_12`, `SARIMA(1,1,3)x(0,1,1)_12`, `SARIMA(2,1,0)x(0,1,1)_12`, `SARIMA(0,1,1)x(0,1,0)_12`, `SARIMA(1,1,1)x(0,1,0)_12`, `SARIMA(2,1,3)x(0,1,0)_12`, `SARIMA(0,1,3)x(0,1,0)_12`, `SARIMA(2,1,1)x(0,1,0)_12`, `SARIMA(1,1,3)x(0,1,0)_12`, `SARIMA(2,1,0)x(0,1,0)_12`.
- **Các mô hình vi phạm kiểm định Ljung-Box ($p < 0.05$, phần dư còn tự tương quan)**: `Log-SARIMA(1,1,1)x(0,1,1)_12`, `Log-SARIMA(0,1,1)x(0,1,1)_12`, `SARIMA(1,1,0)x(0,1,1)_12`, `SARIMA(0,1,0)x(0,1,1)_12`, `SARIMA(1,1,0)x(0,1,0)_12`, `SARIMA(0,1,0)x(0,1,0)_12`, `MA(1) [ARIMA(0,1,1)]`, `AR(2) [ARIMA(2,1,0)]`, `AR(1) [ARIMA(1,1,0)]`, `MA(3) [ARIMA(0,1,3)]`, `ARIMA(1,1,1)`, `ARIMA(2,1,1)`.
- **Các mô hình có cảnh báo không hội tụ (`Converged = False`)**: `Log-SARIMA(2,1,3)x(0,1,1)_12`, `SARIMA(1,1,0)x(0,1,0)_12`.
- **Các mô hình gặp lỗi thực thi (`Status = FAILED`)**: Không có (0 lỗi).

#### C. Lưu ý Phương pháp luận (Tại sao không khẳng định một mô hình tốt nhất tuyệt đối duy nhất?)
1. **Không so sánh chéo AIC/BIC giữa các nhóm khác bậc sai phân hoặc khác phép biến đổi**: Giá trị Log-Likelihood của nhóm `log_d1_D1_s12` được tính trên biến $\ln(Y_t)$ nên có AIC âm, không thể đem so sánh trực tiếp với AIC dương của nhóm `original_d1_D1_s12` hoặc `original_d1`.
2. **Sự đánh đổi giữa Statistical Fit (AIC/BIC), Residual Adequacy (Ljung-Box) và Out-of-Sample Error (RMSE/MAE)**: Một mô hình có thể đạt RMSE thấp nhất trên tập Test nhờ biến đổi Logarit nhưng phần dư trên không gian Log vẫn còn tự tương quan nhẹ (như trên Dataset 1), hoặc một mô hình bậc cao có AIC thấp nhất nhưng tốn nhiều tham số hơn cấu hình tối giản (như `SARIMA(2,1,3)` so với `SARIMA(0,1,1)`). Do đó, việc lựa chọn mô hình cuối cùng phụ thuộc vào mục tiêu ưu tiên độ chính xác dự báo ngoài mẫu hay tính chặt chẽ của giả định nhiễu trắng.

## 3. Hồ sơ Thống kê Chi tiết Từng Mô hình (Individual Model Cards)

### 3.1. AR(1) [ARIMA(1,1,0)]
```text
AR Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261123_154923_d2775a | Status: SUCCESS
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
Converged:                True   Run Time:   0.0289s
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
Run ID: dataset1_20261123_154923_d2775a | Status: SUCCESS
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
Converged:                True   Run Time:   0.0396s
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
Run ID: dataset1_20261123_154923_d2775a | Status: SUCCESS
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
Converged:                True   Run Time:   0.0356s
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
Run ID: dataset1_20261123_154923_d2775a | Status: SUCCESS
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
Converged:                True   Run Time:   0.0546s
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
Run ID: dataset1_20261123_154923_d2775a | Status: SUCCESS
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
Converged:                True   Run Time:   0.0444s
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

### 3.6. ARIMA(2,1,1)
```text
ARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261123_154923_d2775a | Status: SUCCESS
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
Converged:                True   Run Time:   0.0496s
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

### 3.7. SARIMA(0,1,1)x(0,1,0)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261123_154923_d2775a | Status: SUCCESS
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
Converged:                True   Run Time:   0.0377s
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

### 3.8. SARIMA(1,1,1)x(0,1,0)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261123_154923_d2775a | Status: SUCCESS
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
Converged:                True   Run Time:   0.0441s
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

### 3.9. SARIMA(0,1,1)x(0,1,1)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261123_154923_d2775a | Status: SUCCESS
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
Converged:                True   Run Time:   0.0920s
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

### 3.10. SARIMA(1,1,0)x(0,1,1)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261123_154923_d2775a | Status: SUCCESS
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
Converged:                True   Run Time:   0.0851s
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

### 3.11. SARIMA(1,1,1)x(0,1,1)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261123_154923_d2775a | Status: SUCCESS
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
Converged:                True   Run Time:   0.1387s
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

### 3.12. SARIMA(0,1,3)x(0,1,1)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261123_154923_d2775a | Status: SUCCESS
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
Converged:                True   Run Time:   0.1934s
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
  MAE = 1.5048 | RMSE = 1.9527 | MAPE = 8.72% | R² = 0.7767
==============================================================================
```

### 3.13. SARIMA(2,1,1)x(0,1,1)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261123_154923_d2775a | Status: SUCCESS
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
Converged:                True   Run Time:   0.2557s
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

### 3.14. SARIMA(2,1,3)x(0,1,1)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261123_154923_d2775a | Status: SUCCESS
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
Converged:                True   Run Time:   0.6731s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.6735     0.0446   -15.091     0.000     -0.761     -0.586
ar.L2               -0.9016     0.0495   -18.209     0.000     -0.999     -0.805
ma.L1               -0.1584  4382.4369    -0.000     1.000  -8589.577   8589.260
ma.L2                0.6101 12822.4790     0.000     1.000 -25130.987  25132.207
ma.L3               -0.7086 11287.1445    -0.000     1.000 -22123.105  22121.688
ma.S.L12            -0.3672     0.0705    -5.206     0.000     -0.505     -0.229
sigma2               0.2704  4307.4137     0.000     1.000  -8442.105   8442.646
==============================================================================
Ljung-Box (L=24) (Q):      12.67   Jarque-Bera (JB):       0.36
Prob(Q):                   0.9712   Prob(JB):             0.8357
Heteroskedasticity(H):       2.62   Skew:                  -0.03
Prob(H) (two-sided):       0.0016   Kurtosis:               3.23
Residual Mean:             0.0528   Residual Std:         0.5151
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.4822 | RMSE = 1.8684 | MAPE = 8.38% | R² = 0.7956
==============================================================================
```

### 3.15. Log-SARIMA(0,1,1)x(0,1,1)_12
```text
Log-SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261123_154923_d2775a | Status: SUCCESS
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
Converged:                True   Run Time:   0.4422s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ma.L1               -0.7999     0.0539   -14.842     0.000     -0.906     -0.694
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

### 3.16. Log-SARIMA(1,1,1)x(0,1,1)_12
```text
Log-SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261123_154923_d2775a | Status: SUCCESS
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
Converged:                True   Run Time:   0.3887s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.1839     0.0979    -1.878     0.060     -0.376      0.008
ma.L1               -0.7326     0.0736    -9.949     0.000     -0.877     -0.588
ma.S.L12            -0.6232     0.0633    -9.840     0.000     -0.747     -0.499
sigma2               0.0029     0.0003    10.420     0.000      0.002      0.004
==============================================================================
Ljung-Box (L=24) (Q):      40.93   Jarque-Bera (JB):      19.90
Prob(Q):                   0.0170   Prob(JB):             0.0000
Heteroskedasticity(H):       0.71   Skew:                  -0.42
Prob(H) (two-sided):       0.2515   Kurtosis:               4.57
Residual Mean:            -0.0022   Residual Std:         0.0556
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.1795 | RMSE = 1.4918 | MAPE = 6.40% | R² = 0.8697
==============================================================================
```

### 3.17. Log-SARIMA(2,1,3)x(0,1,1)_12
```text
Log-SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261123_154923_d2775a | Status: SUCCESS
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
Converged:               False   Run Time:   1.1124s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -1.1603     0.0086  -134.940     0.000     -1.177     -1.143
ar.L2               -0.9848     0.0123   -79.802     0.000     -1.009     -0.961
ma.L1                0.4074    34.0356     0.012     0.990    -66.301     67.116
ma.L2                0.0919    31.5398     0.003     0.998    -61.725     61.909
ma.L3               -0.7708    44.5419    -0.017     0.986    -88.071     86.530
ma.S.L12            -0.6006     0.0643    -9.348     0.000     -0.727     -0.475
sigma2               0.0025     0.1440     0.017     0.986     -0.280      0.285
==============================================================================
Ljung-Box (L=24) (Q):      17.52   Jarque-Bera (JB):     309.20
Prob(Q):                   0.8259   Prob(JB):             0.0000
Heteroskedasticity(H):       0.84   Skew:                  -1.54
Prob(H) (two-sided):       0.5638   Kurtosis:               9.32
Residual Mean:            -0.0059   Residual Std:         0.0624
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 1.1991 | RMSE = 1.4982 | MAPE = 6.53% | R² = 0.8686
==============================================================================
Warnings:
  [1] Maximum Likelihood optimization failed to converge. Check mle_retvals
==============================================================================
```

### 3.18. SARIMA(0,1,0)x(0,1,0)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261123_154923_d2775a | Status: SUCCESS
Data: Monthly Retail Sales (US, 07/1991 - 06/2008)
Date range: 1991-07-01 to 2008-06-01 (Freq: MS)
Number of observations: 204 (Train: 163, Test: 41)
AR order (p): 0 | Differencing order (d): 1 | MA order (q): 0
Seasonal order (P, D, Q, s): (0, 1, 0, 12)
Transformation: Original Scale | d=1, D=1, s=12
Model: SARIMA(0,1,0)x(0,1,0)_12
==============================================================================
                SARIMAX / ARIMA Statistical Fit & Diagnostics                 
==============================================================================
Log Likelihood:       -165.737   AIC:        333.475
BIC:                   336.479   HQIC:       334.695
Converged:                True   Run Time:   0.0474s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
sigma2               0.5416     0.0552     9.813     0.000      0.433      0.650
==============================================================================
Ljung-Box (L=24) (Q):     100.59   Jarque-Bera (JB):       3.02
Prob(Q):                   0.0000   Prob(JB):             0.2212
Heteroskedasticity(H):       4.29   Skew:                   0.18
Prob(H) (two-sided):       0.0000   Kurtosis:               3.59
Residual Mean:             0.0171   Residual Std:         0.7359
------------------------------------------------------------------------------
OUT-OF-SAMPLE TEST METRICS (Original Scale, n_test=41):
  MAE = 2.1115 | RMSE = 2.5902 | MAPE = 12.21% | R² = 0.6072
==============================================================================
```

### 3.19. SARIMA(0,1,0)x(0,1,1)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261123_154923_d2775a | Status: SUCCESS
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
Converged:                True   Run Time:   0.0976s
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

### 3.20. SARIMA(0,1,3)x(0,1,0)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261123_154923_d2775a | Status: SUCCESS
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
Converged:                True   Run Time:   0.0832s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ma.L1               -0.8825     0.0834   -10.585     0.000     -1.046     -0.719
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

### 3.21. SARIMA(1,1,0)x(0,1,0)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261123_154923_d2775a | Status: SUCCESS
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
Converged:               False   Run Time:   0.0650s
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

### 3.22. SARIMA(1,1,3)x(0,1,0)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261123_154923_d2775a | Status: SUCCESS
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
Converged:                True   Run Time:   0.0852s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.6580     0.6410    -1.027     0.305     -1.914      0.598
ma.L1               -0.2045     0.6394    -0.320     0.749     -1.458      1.049
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

### 3.23. SARIMA(1,1,3)x(0,1,1)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261123_154923_d2775a | Status: SUCCESS
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
Converged:                True   Run Time:   0.2800s
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

### 3.24. SARIMA(2,1,0)x(0,1,0)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261123_154923_d2775a | Status: SUCCESS
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
Converged:                True   Run Time:   0.0245s
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

### 3.25. SARIMA(2,1,0)x(0,1,1)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261123_154923_d2775a | Status: SUCCESS
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
Converged:                True   Run Time:   0.1260s
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

### 3.26. SARIMA(2,1,1)x(0,1,0)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261123_154923_d2775a | Status: SUCCESS
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
Converged:                True   Run Time:   0.0634s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -0.3516     0.1599    -2.200     0.028     -0.665     -0.038
ar.L2               -0.1409     0.1483    -0.950     0.342     -0.432      0.150
ma.L1               -0.5329     0.1585    -3.361     0.001     -0.844     -0.222
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

### 3.27. SARIMA(2,1,3)x(0,1,0)_12
```text
SARIMA Model Summary - Monthly Retail Sales (US, 07/1991 - 06/2008)
==============================================================================
Run ID: dataset1_20261123_154923_d2775a | Status: SUCCESS
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
Converged:                True   Run Time:   0.1383s
------------------------------------------------------------------------------
param                  coef    std err         z     P>|z|     [0.025     0.975]
------------------------------------------------------------------------------
ar.L1               -1.0180     0.1858    -5.479     0.000     -1.382     -0.654
ar.L2               -0.7086     0.1694    -4.182     0.000     -1.041     -0.377
ma.L1                0.1490     0.2102     0.709     0.478     -0.263      0.561
ma.L2                0.0092     0.1897     0.049     0.961     -0.363      0.381
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
