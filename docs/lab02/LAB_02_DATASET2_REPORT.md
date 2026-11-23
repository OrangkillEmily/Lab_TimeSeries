# BÁO CÁO THỰC HÀNH LAB 02 — DATASET 02 — AIRPASSENGERS
> Run ID: `dataset2_20261123_154951_786eea` | Model Summary mở rộng (gồm AR/MA): `reports/model_summary/lab02/dataset2/dataset2_model_summary.md`

## STAGE 01: Khám phá dữ liệu (EDA đầy đủ) & Phân chia Train/Test (80/20)
| Chỉ tiêu | Giá trị |
| --- | --- |
| Số quan sát (n) / Missing / Trùng ngày | 144 tháng / 0 / 0 |
| Khoảng thời gian & Tần suất | 1949-01-01 → 1960-12-01 (Monthly Start - MS) |
| Trung bình (Mean) / Độ lệch chuẩn (Std) | 280.30 / 119.97 (CV = 42.8%) |
| Min / Q25 / Median / Q75 / Max | 104.00 (11/1949) / 180.00 / 265.50 / 360.50 / 622.00 (07/1960) |
| Độ lệch (Skewness) / Tăng trưởng đầu→cuối | 0.583 / gấp 3.86 lần |
| Tập Train (80%, shuffle=False) | 115 tháng (1949-01-01 → 1958-07-01) | Mean=239.95, Std=91.35 |
| Tập Test (20%, shuffle=False) | 29 tháng (1958-08-01 → 1960-12-01) | Mean=440.31, Std=79.51 |

[../figures/lab02/dataset2/report/01_original_series.png]
[../figures/lab02/dataset2/report/02_train_test_split.png]

## STAGE 02–03: Kiểm định ADF & Sai phân
| Chuỗi | ADF Stat | p-value | Kết luận |
| --- | --- | --- | --- |
| Chuỗi gốc (Train) | -0.3569 | 0.9171 | KHÔNG DỪNG |
| Sai phân bậc 1 (d=1) | -2.5396 | 0.1061 | KHÔNG DỪNG |
| Sai phân kết hợp (d=1, D=1, s=12) | -12.1763 | 0.0000 | DỪNG |

[../figures/lab02/dataset2/report/03_differencing.png]

## STAGE 04: Phân tích ACF & PACF
- `d=1` (±0.184): ACF sig = `[1, 4, 8, 9, 11, 12, 13, 16, 20, 23, 24, 32, 36]` | PACF sig = `[1, 2, 4, 7, 8, 10, 11, 12]`
- `d=1, D=1, s=12` (±0.194): ACF sig = `[1, 3, 23]` | PACF sig = `[1, 20, 23]`

[../figures/lab02/dataset2/report/04_acf_pacf_d1.png]
[../figures/lab02/dataset2/report/05_acf_pacf_d1_D1.png]

## STAGE 05–06: Đối chứng ARIMA vs SARIMA → Log-SARIMA
| Mô hình | AIC | BIC | MAE | RMSE | MAPE | R² | Ljung-Box p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ARIMA(1,1,1) | 1058.25 | 1066.45 | 85.25 | 97.5 | 21.29% | -0.56 | 0.0 |
| ARIMA(2,1,1) | 1058.65 | 1069.59 | 60.53 | 82.62 | 12.9% | -0.12 | 0.0 |
| SARIMA(1,1,3)x(0,1,1)_12 | 639.62 | 654.35 | 19.43 | 24.85 | 4.31% | 0.8988 | 0.4105 |
| Log-SARIMA(1,1,0)x(0,1,1)_12 | -327.47 | -320.0 | 12.35 | 15.12 | 2.81% | 0.9626 | 0.997 |

[../figures/lab02/dataset2/report/06_best_sarima_residuals.png]
[../figures/lab02/dataset2/report/08_arima_vs_sarima.png]

## STAGE 07: Log-SARIMA & Đối chứng 3 thế hệ
- Mô hình tối ưu: **`Log-SARIMA(1,1,0)x(0,1,1)_12`** (RMSE = `15.12`, MAPE = `2.81%`, R² = `0.9626`, Ljung-Box p = `0.997`).

[../figures/lab02/dataset2/report/09_log_transformation.png]
[../figures/lab02/dataset2/report/10_arima_vs_log_sarima_battle.png]
[../figures/lab02/dataset2/report/11_log_diagnostics_2x2.png]
