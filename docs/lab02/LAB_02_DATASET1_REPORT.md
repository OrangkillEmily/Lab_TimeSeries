# BÁO CÁO THỰC HÀNH LAB 02 — DATASET 01 — MONTHLY RETAIL SALES (US)
> Run ID: `dataset1_20261123_154923_d2775a` | Model Summary mở rộng (gồm AR/MA): `reports/model_summary/lab02/dataset1/dataset1_model_summary.md`

## STAGE 01: Khám phá dữ liệu (EDA đầy đủ) & Phân chia Train/Test (80/20)
| Chỉ tiêu | Giá trị |
| --- | --- |
| Số quan sát (n) / Missing / Trùng ngày | 204 tháng / 0 / 0 |
| Khoảng thời gian & Tần suất | 1991-07-01 → 2008-06-01 (Monthly Start - MS) |
| Trung bình (Mean) / Độ lệch chuẩn (Std) | 10.69 / 5.96 (CV = 55.7%) |
| Min / Q25 / Median / Q75 / Max | 2.81 (02/1992) / 5.84 / 9.32 / 14.29 / 29.67 (01/2008) |
| Độ lệch (Skewness) / Tăng trưởng đầu→cuối | 0.844 / gấp 5.51 lần |
| Tập Train (80%, shuffle=False) | 163 tháng (1991-07-01 → 2005-01-01) | Mean=8.48, Std=3.94 |
| Tập Test (20%, shuffle=False) | 41 tháng (2005-02-01 → 2008-06-01) | Mean=19.51, Std=4.18 |

[../figures/lab02/dataset1/report/01_original_series.png]
[../figures/lab02/dataset1/report/02_train_test_split.png]

## STAGE 02–03: Kiểm định ADF & Sai phân
| Chuỗi | ADF Stat | p-value | Kết luận |
| --- | --- | --- | --- |
| Chuỗi gốc (Train) | 2.2309 | 0.9989 | KHÔNG DỪNG |
| Sai phân bậc 1 (d=1) | -1.7061 | 0.4280 | KHÔNG DỪNG |
| Sai phân kết hợp (d=1, D=1, s=12) | -8.7879 | 0.0000 | DỪNG |

[../figures/lab02/dataset1/report/03_differencing.png]

## STAGE 04: Phân tích ACF & PACF
- `d=1` (±0.154): ACF sig = `[1, 11, 12, 13, 23, 24, 25, 35, 36]` | PACF sig = `[1, 2, 10, 11, 12, 13, 15]`
- `d=1, D=1, s=12` (±0.16): ACF sig = `[1, 3, 4, 7, 23, 24, 26, 27]` | PACF sig = `[1, 2, 4, 5, 25, 36]`

[../figures/lab02/dataset1/report/04_acf_pacf_d1.png]
[../figures/lab02/dataset1/report/05_acf_pacf_d1_D1.png]

## STAGE 05–06: Đối chứng ARIMA vs SARIMA
| Mô hình | AIC | BIC | MAE | RMSE | MAPE | R² | Ljung-Box p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ARIMA(1,1,1) | 582.47 | 591.73 | 3.67 | 4.62 | 18.04% | -0.25 | 0.0 |
| ARIMA(2,1,1) | 584.46 | 596.81 | 3.67 | 4.63 | 18.06% | -0.25 | 0.0 |
| SARIMA(0,1,1)x(0,1,1)_12 | 233.85 | 242.59 | 1.45 | 1.84 | 8.21% | 0.8012 | 0.1053 |

[../figures/lab02/dataset1/report/06_best_sarima_residuals.png]
[../figures/lab02/dataset1/report/08_arima_vs_sarima.png]
