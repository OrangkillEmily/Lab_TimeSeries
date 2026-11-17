# BÁO CÁO TỔNG HỢP MÔ HÌNH (MODEL SUMMARY) — ELECTRICITY LOAD & WEATHER MULTIVARIATE TIME SERIES (UCI LD2011-2014, 01/2011 - 01/2015)

## 1. Thông tin Phiên Thực nghiệm & Bộ dữ liệu (Provenance & Dataset Info)
- **Run ID**: `20261117_160932_3fe70f`
- **Thời điểm xuất báo cáo**: `2026-11-17 16:19:33`
- **Dataset Key**: `dataset1` (`data/raw/electricity_load.csv`)
- **Tên bộ dữ liệu**: Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
- **Tổng số quan sát**: 35,065 giờ (2011-01-01 00:00:00 to 2015-01-01 00:00:00)
- **Số biến quan sát ($K = 4$)**: `load` (kW — Mục tiêu), `temperature` (°C), `humidity` (%), `wind_speed` (m/s)
- **Tần suất**: `1h` (Hourly)
- **Tập Huấn luyện (Train - 70%)**: 24,545 quan sát
- **Tập Xác thực (Validation - 15%)**: 5,260 quan sát
- **Tập Kiểm thử Độc lập (Hold-out Test - 15%)**: 5,260 quan sát
- **Tổng số thực thể mô hình được thống kê**: 24 (Thành công: 24, Thất bại: 0)
  - **03 Mô hình Đa biến Chính (Hold-out 70/15/15)**: `VAR(24)`, `XGBoost (105 Features)`, `Stacked 2-Layer LSTM + Residual Highway`
  - **03 Mô hình Đối chứng Loại bỏ Thời tiết (Feature Ablation)**: `AR(24) Univariate`, `XGBoost NoWeather`, `LSTM NoWeather`
  - **03 Mô hình Vận hành Cuốn chiếu Thực tế (Walk-Forward 168h)**: `VAR Walk-Forward`, `XGBoost Walk-Forward`, `LSTM Walk-Forward`
  - **15 Mô hình Kiểm định Chéo Mở rộng (`5-Fold TimeSeriesSplit`)**: `Fold 1..5` cho cả 3 họ mô hình `VAR`, `XGBoost`, `LSTM`

## 2. Bảng So sánh Tổng hợp Toàn bộ 24 Thực thể Mô hình trên Dataset 1 (Model Comparison Table)

> **Nguyên tắc đối chiếu**: Tất cả 5 chỉ số đánh giá (`MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²`) và kiểm định phần dư đều được tính toán trên **đơn vị vật lý gốc (kW)** sau khi áp dụng phép biến đổi ngược (`inverse_transform` hoặc cộng tích lũy sai phân bậc 1).

| # | Model | Family | Regime | Params / Feats | RMSE (kW) | MAE (kW) | MAPE (%) | R² | DW Stat | Ljung-Box Q | Device | Time (s) |
|---:|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|:---|---:|
| 1 | VAR(24) [Multivariate Differenced d=1, Hold-out] | VAR | `Hold-out_Test_15pct` | 97 | 7,122.36 | 4,576.49 | 2.0010 | 0.9935 | 2.3686 | 800.16 | `cpu` | 2.9913 |
| 2 | XGBoost Multivariate Regressor [105 Features, Hold-out] | XGBoost | `Hold-out_Test_15pct` | 105 | 6,482.45 | 4,337.53 | 1.8922 | 0.9946 | 1.6171 | 2,163.86 | `cuda` | 5.6949 |
| 3 | Stacked 2-Layer LSTM + Residual Highway [Multivariate, Hold-out] | LSTM | `Hold-out_Test_15pct` | 30,589 | 6,570.41 | 4,524.04 | 2.0185 | 0.9944 | 1.9519 | 721.46 | `cuda` | 92.2252 |
| 4 | AR(24) [Univariate Differenced d=1, No Weather Ablation] | AR-Ablation | `Ablation_No_Weather` | 25 | 7,120.99 | 4,569.45 | 1.9958 | 0.9935 | 2.3702 | 808.92 | `cpu` | 1.9443 |
| 5 | XGBoost Univariate [No Weather, 33 Load/Calendar Features] | XGBoost-Ablation | `Ablation_No_Weather` | 33 | 6,379.23 | 4,280.20 | 1.8586 | 0.9947 | 1.7422 | 1,465.22 | `cuda` | 3.7017 |
| 6 | Stacked 2-Layer LSTM + Highway [No Weather, Input_Dim=1] | LSTM-Ablation | `Ablation_No_Weather` | 29,818 | 6,946.54 | 4,862.90 | 2.1741 | 0.9938 | 1.8465 | 1,310.76 | `cuda` | 59.9464 |
| 7 | XGBoost [Walk-Forward Backtesting 168h Online Feedback] | XGBoost-WalkForward | `Walk_Forward_168h` | 105 | 4,870.64 | 3,582.84 | 1.6381 | 0.9950 | 2.5024 | 49.35 | `cuda` | 5.6949 |
| 8 | VAR [Walk-Forward Backtesting 168h Online Feedback] | VAR-WalkForward | `Walk_Forward_168h` | 97 | 5,282.87 | 3,937.53 | 1.8182 | 0.9941 | 2.9518 | 77.87 | `cpu` | 2.9913 |
| 9 | LSTM [Walk-Forward Backtesting 168h Online Feedback] | LSTM-WalkForward | `Walk_Forward_168h` | 30,589 | 5,810.71 | 4,238.78 | 1.9522 | 0.9929 | 2.8172 | 160.81 | `cuda` | 92.2252 |
| 10 | VAR [5-Fold TimeSeriesSplit — Fold 1/5] | VAR-CV-Fold1 | `5Fold_CV_Fold_1` | 97 | 6,286.99 | 3,759.03 | 2.2744 | 0.9915 | 2.0353 | 733.40 | `cpu` | 0.8302 |
| 11 | XGBoost [5-Fold TimeSeriesSplit — Fold 1/5] | XGBoost-CV-Fold1 | `5Fold_CV_Fold_1` | 105 | 12,836.99 | 8,234.13 | 5.2542 | 0.9647 | 0.4850 | 14,424.73 | `cuda` | 4.9675 |
| 12 | LSTM [5-Fold TimeSeriesSplit — Fold 1/5] | LSTM-CV-Fold1 | `5Fold_CV_Fold_1` | 30,589 | 8,672.94 | 6,106.53 | 3.6206 | 0.9839 | 1.3610 | 3,820.59 | `cuda` | 10.8592 |
| 13 | VAR [5-Fold TimeSeriesSplit — Fold 2/5] | VAR-CV-Fold2 | `5Fold_CV_Fold_2` | 97 | 8,731.46 | 5,508.47 | 2.2799 | 0.9926 | 2.3520 | 723.86 | `cpu` | 1.5272 |
| 14 | XGBoost [5-Fold TimeSeriesSplit — Fold 2/5] | XGBoost-CV-Fold2 | `5Fold_CV_Fold_2` | 105 | 17,914.25 | 10,471.79 | 3.8495 | 0.9690 | 0.8209 | 13,686.20 | `cuda` | 3.4974 |
| 15 | LSTM [5-Fold TimeSeriesSplit — Fold 2/5] | LSTM-CV-Fold2 | `5Fold_CV_Fold_2` | 30,589 | 13,520.61 | 8,974.15 | 3.7969 | 0.9824 | 1.4323 | 4,285.88 | `cuda` | 13.3197 |
| 16 | VAR [5-Fold TimeSeriesSplit — Fold 3/5] | VAR-CV-Fold3 | `5Fold_CV_Fold_3` | 97 | 6,685.48 | 4,065.42 | 2.3303 | 0.9938 | 2.2766 | 655.04 | `cpu` | 2.2394 |
| 17 | XGBoost [5-Fold TimeSeriesSplit — Fold 3/5] | XGBoost-CV-Fold3 | `5Fold_CV_Fold_3` | 105 | 8,234.82 | 5,685.14 | 3.1051 | 0.9905 | 1.0880 | 4,979.06 | `cuda` | 1.8797 |
| 18 | LSTM [5-Fold TimeSeriesSplit — Fold 3/5] | LSTM-CV-Fold3 | `5Fold_CV_Fold_3` | 30,589 | 7,395.36 | 5,055.66 | 2.8405 | 0.9924 | 1.7680 | 2,052.64 | `cuda` | 30.4732 |
| 19 | VAR [5-Fold TimeSeriesSplit — Fold 4/5] | VAR-CV-Fold4 | `5Fold_CV_Fold_4` | 97 | 6,580.69 | 3,697.58 | 6.7214 | 0.9910 | 2.3427 | 1,173.80 | `cpu` | 2.9518 |
| 20 | XGBoost [5-Fold TimeSeriesSplit — Fold 4/5] | XGBoost-CV-Fold4 | `5Fold_CV_Fold_4` | 105 | 5,732.55 | 3,700.07 | 6.5246 | 0.9932 | 1.5827 | 1,905.74 | `cuda` | 6.2938 |
| 21 | LSTM [5-Fold TimeSeriesSplit — Fold 4/5] | LSTM-CV-Fold4 | `5Fold_CV_Fold_4` | 30,589 | 6,375.93 | 4,191.20 | 7.0281 | 0.9915 | 1.8685 | 801.37 | `cuda` | 39.5551 |
| 22 | VAR [5-Fold TimeSeriesSplit — Fold 5/5] | VAR-CV-Fold5 | `5Fold_CV_Fold_5` | 97 | 6,983.07 | 4,527.47 | 1.9895 | 0.9935 | 2.3682 | 779.92 | `cpu` | 3.7818 |
| 23 | XGBoost [5-Fold TimeSeriesSplit — Fold 5/5] | XGBoost-CV-Fold5 | `5Fold_CV_Fold_5` | 105 | 6,428.21 | 4,319.27 | 1.8771 | 0.9945 | 1.6303 | 2,011.82 | `cuda` | 6.2370 |
| 24 | LSTM [5-Fold TimeSeriesSplit — Fold 5/5] | LSTM-CV-Fold5 | `5Fold_CV_Fold_5` | 30,589 | 6,741.05 | 4,731.91 | 2.1160 | 0.9940 | 1.8892 | 1,145.67 | `cuda` | 49.2265 |

### Diễn giải Đa Tiêu chí Tự động (Run ID: `20261117_160932_3fe70f` — Dataset 1: Electricity Load & Weather)

#### A. Mô hình Tốt nhất theo Từng Quy trình Kiểm định Độc lập (Regime-Specific Leaders)
- **Tiêu chí 1 — Độ chính xác cao nhất trên Tập Hold-out Test độc lập (15% - 5,260 giờ)**: **XGBoost** (`RMSE = 6,482.45 kW`, `MAE = 4,337.53 kW`, `MAPE = 1.8922%`, `R² = 0.9946`).
- **Tiêu chí 2 — Độ ổn định cao nhất trên Kiểm định Chéo 5-Fold TimeSeriesSplit (Expanding Window)**: **VAR** (`RMSE = 7,053.54 ± 970.44 kW`, `MAE = 4,311.59 ± 745.33 kW`, `R² = 0.9925 ± 0.0012`).
- **Tiêu chí 3 — Hiệu năng cao nhất trong Vận hành Cuốn chiếu Walk-Forward Backtesting (168 giờ)**: **XGBoost** (`RMSE = 4,870.64 kW`, `MAE = 3,582.84 kW`, `MAPE = 1.6381%`, `R² = 0.9950`).
- **Tiêu chí 4 — Mô hình khai thác biến ngoại sinh thời tiết hiệu quả nhất (Feature Ablation Impact)**: **Stacked 2-Layer LSTM + Residual Highway** giúp giảm `5.41%` sai số RMSE (từ `6,946.54 kW` khi không có thời tiết xuống `6,570.41 kW` khi kết hợp `temperature`, `humidity`, `wind_speed`) và giảm `7.16%` MAPE.
- **Tiêu chí 5 — Khử tự tương quan phần dư bậc 1 tốt nhất (Durbin-Watson gần 2.00 nhất & Ljung-Box Q thấp nhất trên Hold-out Test)**: **Stacked 2-Layer LSTM + Residual Highway** (`DW = 1.9519`, `Ljung-Box Q(24) = 721.46`, `Heteroskedasticity H = 1.1835`).

## 3. Hồ sơ Thống kê Chi tiết Từng Mô hình (Individual Model Cards — 24 Models)

### 3.1. VAR(24) [Multivariate Differenced d=1, Hold-out]
```text
VAR / AR Multivariate Model Summary - Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
==============================================================================
Run ID: 20261117_160932_3fe70f | Status: SUCCESS | Regime: Hold-out Split (70% Train - 15% Val - 15% Test)
Data: Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
Date range: 2011-01-01 00:00:00 to 2015-01-01 00:00:00 (Freq: 1h)
Observations: 35,065 (Train: 24,545, Val: 5,260, Eval/Test: 5,260)
Lag Order (p): 24 | Differencing (d): 1 | Equations (K): 4 | Params: 97 (Target Eq) / 388 (System)
Transformation: First-Order Differencing (Delta Y_t = Y_t - Y_{t-1}) + Cumulative Inverse (kW)
Model: VAR(24) [Multivariate Differenced d=1, Hold-out]
==============================================================================
             VAR / OLS Statistical Fit & Information Criteria                 
==============================================================================
Log Likelihood (Load Eq):  -250953.731   AIC (Load Eq):     502101.462
BIC (Load Eq):              502887.865   HQIC (Load Eq):    502356.233
FPE (Final Pred Error):    45779369.30   Device:                   cpu
Converged:                        True   Train / Pred:    2.9913s / 0.2545s
------------------------------------------------------------------------------
CROSS-VARIABLE LAG INFLUENCE ON TARGET EQUATION (Delta_load_t, p=24 Lags):
Variable          Lags    Sum(Coef)      L1 Norm      L2 Norm    Dominant Lag
------------------------------------------------------------------------------
load                24      -2.1075       3.5077       0.9214 L24.load (+0.70)
temperature         24    -247.1308    1335.1613     344.4209 L15.temperature (-167.13)
humidity            24     -42.0006     322.4772      80.7685 L17.humidity (+35.21)
wind_speed          24    -700.8295    1203.9298     294.3422 L6.wind_speed (-130.51)
------------------------------------------------------------------------------
FULL PARAMETER ESTIMATES FOR TARGET EQUATION (Delta_load_t — All 97 Parameters):
param                       coef     std err         z    P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const (Intercept)         9.1835     43.0396     0.213   0.8310    -75.1724     93.5394
L1.load                  -0.0716      0.0053   -13.416   0.0000     -0.0821     -0.0612
L1.temperature           46.3073     60.9754     0.759   0.4476    -73.2020    165.8167
L1.humidity              -3.6787     21.5707    -0.171   0.8646    -45.9564     38.5991
L1.wind_speed           -19.7575     57.3160    -0.345   0.7303   -132.0945     92.5796
L2.load                  -0.1399      0.0055   -25.295   0.0000     -0.1507     -0.1291
L2.temperature           78.0222     73.7126     1.058   0.2898    -66.4515    222.4960
L2.humidity             -15.7836     28.4882    -0.554   0.5796    -71.6194     40.0523
L2.wind_speed           -44.3352     72.3804    -0.613   0.5402   -186.1978     97.5274
L3.load                  -0.1276      0.0056   -22.875   0.0000     -0.1385     -0.1166
L3.temperature           63.9570     81.1181     0.788   0.4304    -95.0312    222.9452
L3.humidity             -27.4565     33.1069    -0.829   0.4069    -92.3447     37.4317
L3.wind_speed           -22.0320     80.1311    -0.275   0.7834   -179.0857    135.0218
L4.load                  -0.1338      0.0056   -23.867   0.0000     -0.1448     -0.1228
L4.temperature           43.8017     86.8001     0.505   0.6138   -126.3230    213.9263
L4.humidity               6.7281     36.4243     0.185   0.8535    -64.6620     78.1182
L4.wind_speed            -5.7764     84.9558    -0.068   0.9458   -172.2863    160.7335
L5.load                  -0.1367      0.0056   -24.322   0.0000     -0.1477     -0.1257
L5.temperature            6.0134     91.5771     0.066   0.9476   -173.4740    185.5008
L5.humidity              -9.3287     38.8373    -0.240   0.8102    -85.4483     66.7908
L5.wind_speed           -70.9169     88.0356    -0.806   0.4205   -243.4632    101.6294
L6.load                  -0.1429      0.0056   -25.396   0.0000     -0.1539     -0.1319
L6.temperature           19.7367     95.3997     0.207   0.8361   -167.2429    206.7163
L6.humidity             -13.2734     40.6276    -0.327   0.7439    -92.9019     66.3551
L6.wind_speed          -130.5072     90.0785    -1.449   0.1474   -307.0574     46.0430
L7.load                  -0.1264      0.0057   -22.329   0.0000     -0.1375     -0.1153
L7.temperature            5.3765     98.5552     0.055   0.9565   -187.7878    198.5409
L7.humidity             -15.3372     41.9294    -0.366   0.7145    -97.5171     66.8428
L7.wind_speed           -80.6643     91.5337    -0.881   0.3782   -260.0667     98.7380
L8.load                  -0.1294      0.0057   -22.794   0.0000     -0.1405     -0.1183
L8.temperature           87.2742    101.1779     0.863   0.3884   -111.0304    285.5789
L8.humidity              -8.6240     42.8353    -0.201   0.8404    -92.5794     75.3313
L8.wind_speed           -55.6109     92.6072    -0.601   0.5482   -237.1174    125.8956
L9.load                  -0.1297      0.0057   -22.773   0.0000     -0.1409     -0.1185
L9.temperature           22.8928    103.2946     0.222   0.8246   -179.5605    225.3462
L9.humidity               3.6158     43.4110     0.083   0.9336    -81.4680     88.6995
L9.wind_speed          -101.9429     93.3171    -1.092   0.2746   -284.8407     80.9549
L10.load                 -0.1364      0.0057   -23.895   0.0000     -0.1476     -0.1252
L10.temperature         -91.2244    104.9657    -0.869   0.3848   -296.9529    114.5042
L10.humidity            -25.0639     43.6934    -0.574   0.5662   -110.7012     60.5735
L10.wind_speed          -72.6365     93.8932    -0.774   0.4392   -256.6635    111.3904
L11.load                 -0.1451      0.0057   -25.418   0.0000     -0.1563     -0.1339
L11.temperature         -41.7508    106.1999    -0.393   0.6942   -249.8984    166.3968
L11.humidity             -8.5680     43.8704    -0.195   0.8452    -94.5523     77.4163
L11.wind_speed          -31.7611     94.3118    -0.337   0.7363   -216.6085    153.0862
L12.load                 -0.0882      0.0057   -15.388   0.0000     -0.0994     -0.0770
L12.temperature        -116.4004    106.8363    -1.090   0.2759   -325.7953     92.9944
L12.humidity            -25.8350     44.0098    -0.587   0.5572   -112.0924     60.4225
L12.wind_speed          -75.5223     94.4715    -0.799   0.4240   -260.6827    109.6380
L13.load                 -0.1265      0.0057   -22.065   0.0000     -0.1377     -0.1153
L13.temperature         -72.4570    106.8212    -0.678   0.4976   -281.8222    136.9083
L13.humidity              2.2172     44.0142     0.050   0.9598    -84.0489     88.4832
L13.wind_speed          -86.8055     94.4894    -0.919   0.3583   -272.0009     98.3899
L14.load                 -0.1383      0.0057   -24.190   0.0000     -0.1495     -0.1271
L14.temperature         -15.1687    106.2225    -0.143   0.8864   -223.3606    193.0232
L14.humidity             -4.6159     43.8729    -0.105   0.9162    -90.6049     81.3732
L14.wind_speed          -67.7913     94.3555    -0.718   0.4725   -252.7243    117.1417
L15.load                 -0.1224      0.0057   -21.390   0.0000     -0.1336     -0.1112
L15.temperature        -167.1315    105.0343    -1.591   0.1116   -372.9945     38.7315
L15.humidity             14.3197     43.6917     0.328   0.7431    -71.3143     99.9538
L15.wind_speed           -2.7414     93.9502    -0.029   0.9767   -186.8800    181.3971
L16.load                 -0.1137      0.0057   -19.942   0.0000     -0.1249     -0.1025
L16.temperature        -121.9905    103.5070    -1.179   0.2386   -324.8601     80.8791
L16.humidity             22.7109     43.3914     0.523   0.6007    -62.3345    107.7562
L16.wind_speed          -12.4748     93.3644    -0.134   0.8937   -195.4652    170.5156
L17.load                 -0.1212      0.0057   -21.319   0.0000     -0.1323     -0.1100
L17.temperature         -30.1448    101.5131    -0.297   0.7665   -229.1064    168.8168
L17.humidity             35.2072     42.8119     0.822   0.4109    -48.7025    119.1168
L17.wind_speed          -20.5662     92.6301    -0.222   0.8243   -202.1176    160.9851
L18.load                 -0.1217      0.0057   -21.450   0.0000     -0.1328     -0.1106
L18.temperature          -9.0678     98.8668    -0.092   0.9269   -202.8428    184.7072
L18.humidity             16.4456     41.9312     0.392   0.6949    -65.7379     98.6291
L18.wind_speed           31.3933     91.5313     0.343   0.7316   -148.0044    210.7909
L19.load                 -0.1358      0.0056   -24.098   0.0000     -0.1469     -0.1248
L19.temperature          -9.9249     95.5278    -0.104   0.9173   -197.1555    177.3058
L19.humidity             -2.0915     40.6594    -0.051   0.9590    -81.7824     77.5994
L19.wind_speed          -12.1478     90.0701    -0.135   0.8927   -188.6817    164.3861
L20.load                 -0.1211      0.0056   -21.554   0.0000     -0.1321     -0.1101
L20.temperature        -115.8854     91.5912    -1.265   0.2058   -295.4005     63.6297
L20.humidity             30.0615     38.8748     0.773   0.4394    -46.1316    106.2546
L20.wind_speed          -38.3893     88.0170    -0.436   0.6627   -210.8991    134.1205
L21.load                 -0.1232      0.0056   -22.039   0.0000     -0.1341     -0.1122
L21.temperature          22.5732     86.8782     0.260   0.7950   -147.7045    192.8510
L21.humidity              1.6021     36.4426     0.044   0.9649    -69.8239     73.0281
L21.wind_speed           22.3151     84.9147     0.263   0.7927   -144.1142    188.7445
L22.load                 -0.1496      0.0056   -26.897   0.0000     -0.1605     -0.1387
L22.temperature          73.9003     81.0903     0.911   0.3621    -85.0335    232.8341
L22.humidity             -7.3919     33.1357    -0.223   0.8235    -72.3366     57.5527
L22.wind_speed           44.1887     80.0956     0.552   0.5812   -112.7956    201.1729
L23.load                 -0.0266      0.0055    -4.804   0.0000     -0.0374     -0.0157
L23.temperature          54.9835     73.7174     0.746   0.4557    -89.4996    199.4666
L23.humidity              7.3303     28.5010     0.257   0.7970    -48.5306     63.1911
L23.wind_speed           68.3978     72.3549     0.945   0.3445    -73.4149    210.2105
L24.load                  0.7001      0.0053   131.069   0.0000      0.6896      0.7106
L24.temperature          19.1763     60.9818     0.314   0.7532   -100.3456    138.6981
L24.humidity            -15.1906     21.5733    -0.704   0.4813    -57.4735     27.0923
L24.wind_speed           85.2553     57.3027     1.488   0.1368    -27.0558    197.5664
==============================================================================
Ljung-Box (L=24) (Q):     800.16   Jarque-Bera (JB):    631166.90
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     1.6263   Skew:                   0.3481
Prob(H) (two-sided):       0.0000   Kurtosis:              56.6597
Durbin-Watson (DW):        2.3686   MedAE (kW):            3206.30
Residual Mean (kW):      -25.5376   Residual Std (kW):   7122.3168
------------------------------------------------------------------------------
OUT-OF-SAMPLE EVALUATION METRICS (Original Scale kW, n_eval=5,260):
  MSE  = 50,728,048.8075 | RMSE = 7,122.3626 kW | MAE = 4,576.4883 kW
  MAPE =        2.0010% | R²   =     0.9935    | Max Error = 130,203.94 kW
  [In-Sample Train (70%)] RMSE = 6,739.3319 kW | MAE = 4,197.6541 kW | MAPE = 2.0323% | R² = 0.9937
  [Validation Set (15%)]  RMSE = 6,501.0348 kW | MAE = 3,568.5953 kW | MAPE = 7.2729% | R² = 0.9882
  [5-Fold Expanding CV]   RMSE = 7,053.54 ± 970.44 kW | MAE = 4,311.59 ± 745.33 kW | R² = 0.9925 ± 0.0012
  [Weather Ablation]      With Weather RMSE = 7,122.36 kW vs. Without Weather RMSE = 7,120.99 kW (Delta = -0.02%)
==============================================================================
```

### 3.2. XGBoost Multivariate Regressor [105 Features, Hold-out]
```text
XGBoost Gradient Boosting Model Summary - Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
==============================================================================
Run ID: 20261117_160932_3fe70f | Status: SUCCESS | Regime: Hold-out Split (70% Train - 15% Val - 15% Test)
Data: Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
Date range: 2011-01-01 00:00:00 to 2015-01-01 00:00:00 (Freq: 1h)
Observations: 35,065 (Train: 24,545, Val: 5,260, Eval/Test: 5,260)
Features: 105 Tabular Features | Trees Configured: 450 | Best Iteration: 449
Max Depth: 6 | Learning Rate (eta): 0.045 | Subsample: 0.85 | Colsample: 0.85
Regularization: L1 (alpha)=0.1, L2 (lambda)=1.0 | Tree Method: hist (cuda)
Transformation: StandardScaler (Fit on Train Only) + Inverse Transform to Original Scale (kW)
Model: XGBoost Multivariate Regressor [105 Features, Hold-out]
==============================================================================
            XGBoost Boosting Convergence & Feature Importance Fit             
==============================================================================
Train Scaled RMSE (Tree 1 -> End): 0.955735 -> 0.043310
Val Scaled RMSE   (Tree 1 -> Best): 0.821509 -> 0.066113
Converged:                        True   Train / Pred:    5.6949s / 0.1125s
------------------------------------------------------------------------------
BOOSTING ITERATION CONVERGENCE TRAJECTORY (Extracted from evals_result_):
  Tree #  Train RMSE (Scaled)  Val RMSE (Scaled)      Gen Gap           Status
------------------------------------------------------------------------------
       1             0.955735           0.821509    -0.134226     Initial Tree
      10             0.638113           0.550413    -0.087699         Boosting
      25             0.331821           0.289812    -0.042009         Boosting
      50             0.131115           0.120498    -0.010617         Boosting
     100             0.067032           0.071110    +0.004078         Boosting
     150             0.059268           0.068470    +0.009202         Boosting
     200             0.055143           0.067492    +0.012349         Boosting
     250             0.052056           0.066980    +0.014923         Boosting
     300             0.049547           0.066612    +0.017065         Boosting
     350             0.047346           0.066398    +0.019051         Boosting
     400             0.045240           0.066208    +0.020967         Boosting
     450             0.043310           0.066113    +0.022803 Best Val RMSE (Selected)
------------------------------------------------------------------------------
FEATURE IMPORTANCE AGGREGATED BY FUNCTIONAL GROUP (105 Engineered Features):
Feature Group                                  Count  Share (%)    Top Feature
------------------------------------------------------------------------------
1. Autoregressive Load Lags (load_lag_1h..24h)    24     93.99%   load_lag_24h
4. Exogenous Humidity Lags (humidity_lag_1h..2    24      1.84% humidity_lag_12h
3. Cyclical Fourier & Calendar (sin/cos, weeke     5      1.73%       cos_hour
5. Exogenous Temperature Lags (temperature_lag    24      1.61% temperature_lag_10h
2. Load Rolling & Trend Stats (roll_mean, roll     4      0.42% load_roll_mean_24h
6. Exogenous Wind Speed Lags (wind_speed_lag_1    24      0.40% wind_speed_lag_18h
------------------------------------------------------------------------------
COMPLETE FEATURE IMPORTANCE RANKING (All 105/105 Features Extracted):
Rank Feature Name              Gain Importance   Share (%)   Cumulative (%)
------------------------------------------------------------------------------
   1 load_lag_24h                     0.628052      62.81%           62.81%
   2 load_lag_23h                     0.189816      18.98%           81.79%
   3 load_lag_1h                      0.093452       9.35%           91.13%
   4 cos_hour                         0.006380       0.64%           91.77%
   5 humidity_lag_12h                 0.005469       0.55%           92.32%
   6 sin_hour                         0.004539       0.45%           92.77%
   7 is_weekend                       0.003984       0.40%           93.17%
   8 load_lag_20h                     0.003592       0.36%           93.53%
   9 load_lag_21h                     0.003419       0.34%           93.87%
  10 load_lag_22h                     0.002926       0.29%           94.16%
  11 temperature_lag_10h              0.002186       0.22%           94.38%
  12 temperature_lag_20h              0.002125       0.21%           94.59%
  13 load_lag_19h                     0.001766       0.18%           94.77%
  14 humidity_lag_13h                 0.001659       0.17%           94.94%
  15 load_lag_2h                      0.001600       0.16%           95.10%
  16 humidity_lag_11h                 0.001539       0.15%           95.25%
  17 load_lag_14h                     0.001520       0.15%           95.40%
  18 load_lag_15h                     0.001335       0.13%           95.54%
  19 load_roll_mean_24h               0.001323       0.13%           95.67%
  20 sin_dayofweek                    0.001306       0.13%           95.80%
  21 load_lag_16h                     0.001304       0.13%           95.93%
  22 load_lag_13h                     0.001293       0.13%           96.06%
  23 temperature_lag_13h              0.001243       0.12%           96.18%
  24 humidity_lag_21h                 0.001225       0.12%           96.31%
  25 humidity_lag_1h                  0.001154       0.12%           96.42%
  26 cos_dayofweek                    0.001136       0.11%           96.53%
  27 load_roll_std_24h                0.001110       0.11%           96.65%
  28 temperature_lag_19h              0.001094       0.11%           96.75%
  29 load_lag_17h                     0.001091       0.11%           96.86%
  30 load_lag_12h                     0.001070       0.11%           96.97%
  31 load_lag_11h                     0.001046       0.10%           97.08%
  32 load_lag_18h                     0.001016       0.10%           97.18%
  33 load_roll_std_6h                 0.001014       0.10%           97.28%
  34 load_lag_3h                      0.000926       0.09%           97.37%
  35 temperature_lag_11h              0.000906       0.09%           97.46%
  36 load_lag_4h                      0.000862       0.09%           97.55%
  37 load_lag_10h                     0.000790       0.08%           97.63%
  38 temperature_lag_22h              0.000789       0.08%           97.71%
  39 temperature_lag_21h              0.000780       0.08%           97.78%
  40 load_roll_mean_6h                0.000764       0.08%           97.86%
  41 load_lag_6h                      0.000753       0.08%           97.94%
  42 temperature_lag_12h              0.000729       0.07%           98.01%
  43 temperature_lag_23h              0.000702       0.07%           98.08%
  44 load_lag_8h                      0.000670       0.07%           98.15%
  45 temperature_lag_1h               0.000652       0.07%           98.21%
  46 load_lag_7h                      0.000626       0.06%           98.27%
  47 humidity_lag_22h                 0.000605       0.06%           98.33%
  48 humidity_lag_10h                 0.000600       0.06%           98.39%
  49 temperature_lag_24h              0.000586       0.06%           98.45%
  50 humidity_lag_23h                 0.000561       0.06%           98.51%
  51 load_lag_9h                      0.000557       0.06%           98.56%
  52 humidity_lag_20h                 0.000557       0.06%           98.62%
  53 humidity_lag_7h                  0.000530       0.05%           98.67%
  54 temperature_lag_9h               0.000501       0.05%           98.72%
  55 temperature_lag_8h               0.000483       0.05%           98.77%
  56 temperature_lag_14h              0.000482       0.05%           98.82%
  57 humidity_lag_19h                 0.000479       0.05%           98.87%
  58 humidity_lag_18h                 0.000459       0.05%           98.91%
  59 humidity_lag_15h                 0.000459       0.05%           98.96%
  60 load_lag_5h                      0.000455       0.05%           99.00%
  61 temperature_lag_4h               0.000431       0.04%           99.05%
  62 humidity_lag_8h                  0.000395       0.04%           99.09%
  63 humidity_lag_24h                 0.000387       0.04%           99.13%
  64 temperature_lag_17h              0.000382       0.04%           99.16%
  65 temperature_lag_6h               0.000344       0.03%           99.20%
  66 humidity_lag_9h                  0.000335       0.03%           99.23%
  67 temperature_lag_15h              0.000318       0.03%           99.26%
  68 humidity_lag_6h                  0.000307       0.03%           99.29%
  69 temperature_lag_16h              0.000302       0.03%           99.32%
  70 wind_speed_lag_18h               0.000300       0.03%           99.35%
  71 humidity_lag_17h                 0.000296       0.03%           99.38%
  72 humidity_lag_14h                 0.000295       0.03%           99.41%
  73 humidity_lag_16h                 0.000289       0.03%           99.44%
  74 wind_speed_lag_21h               0.000287       0.03%           99.47%
  75 temperature_lag_2h               0.000254       0.03%           99.50%
  76 humidity_lag_3h                  0.000245       0.02%           99.52%
  77 temperature_lag_3h               0.000229       0.02%           99.54%
  78 temperature_lag_5h               0.000212       0.02%           99.57%
  79 wind_speed_lag_19h               0.000209       0.02%           99.59%
  80 wind_speed_lag_1h                0.000205       0.02%           99.61%
  81 temperature_lag_7h               0.000204       0.02%           99.63%
  82 wind_speed_lag_2h                0.000202       0.02%           99.65%
  83 humidity_lag_2h                  0.000196       0.02%           99.67%
  84 humidity_lag_5h                  0.000195       0.02%           99.69%
  85 wind_speed_lag_22h               0.000188       0.02%           99.71%
  86 temperature_lag_18h              0.000180       0.02%           99.72%
  87 humidity_lag_4h                  0.000180       0.02%           99.74%
  88 wind_speed_lag_14h               0.000165       0.02%           99.76%
  89 wind_speed_lag_7h                0.000162       0.02%           99.77%
  90 wind_speed_lag_10h               0.000161       0.02%           99.79%
  91 wind_speed_lag_16h               0.000158       0.02%           99.81%
  92 wind_speed_lag_4h                0.000156       0.02%           99.82%
  93 wind_speed_lag_24h               0.000156       0.02%           99.84%
  94 wind_speed_lag_13h               0.000152       0.02%           99.85%
  95 wind_speed_lag_6h                0.000145       0.01%           99.87%
  96 wind_speed_lag_11h               0.000145       0.01%           99.88%
  97 wind_speed_lag_15h               0.000142       0.01%           99.90%
  98 wind_speed_lag_5h                0.000140       0.01%           99.91%
  99 wind_speed_lag_23h               0.000136       0.01%           99.92%
 100 wind_speed_lag_12h               0.000136       0.01%           99.94%
 101 wind_speed_lag_8h                0.000135       0.01%           99.95%
 102 wind_speed_lag_9h                0.000133       0.01%           99.96%
 103 wind_speed_lag_20h               0.000129       0.01%           99.98%
 104 wind_speed_lag_17h               0.000120       0.01%           99.99%
 105 wind_speed_lag_3h                0.000116       0.01%          100.00%
==============================================================================
Ljung-Box (L=24) (Q):    2163.86   Jarque-Bera (JB):    269756.02
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     1.2750   Skew:                   1.1082
Prob(H) (two-sided):       0.0000   Kurtosis:              38.0131
Durbin-Watson (DW):        1.6171   MedAE (kW):            3042.98
Residual Mean (kW):     -158.6351   Residual Std (kW):   6480.5125
------------------------------------------------------------------------------
OUT-OF-SAMPLE EVALUATION METRICS (Original Scale kW, n_eval=5,260):
  MSE  = 42,022,207.1643 | RMSE = 6,482.4538 kW | MAE = 4,337.5276 kW
  MAPE =        1.8922% | R²   =     0.9946    | Max Error = 129,028.65 kW
  [In-Sample Train (70%)] RMSE = 3,671.3019 kW | MAE = 2,692.9497 kW | MAPE = 1.2392% | R² = 0.9981
  [Validation Set (15%)]  RMSE = 5,604.2708 kW | MAE = 3,607.2464 kW | MAPE = 7.0705% | R² = 0.9912
  [5-Fold Expanding CV]   RMSE = 10,229.36 ± 5,111.40 kW | MAE = 6,482.08 ± 2,830.19 kW | R² = 0.9824 ± 0.0143
  [Weather Ablation]      With Weather RMSE = 6,482.45 kW vs. Without Weather RMSE = 6,379.23 kW (Delta = -1.62%)
==============================================================================
```

### 3.3. Stacked 2-Layer LSTM + Residual Highway [Multivariate, Hold-out]
```text
Stacked 2-Layer LSTM Deep Learning Model Summary - Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
==============================================================================
Run ID: 20261117_160932_3fe70f | Status: SUCCESS | Regime: Hold-out Split (70% Train - 15% Val - 15% Test)
Data: Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
Date range: 2011-01-01 00:00:00 to 2015-01-01 00:00:00 (Freq: 1h)
Observations: 35,065 (Train: 24,545, Val: 5,260, Eval/Test: 5,260)
Architecture: Input(24,4) -> LSTM(64) -> Dropout(0.1) -> LSTM(32) -> LayerNorm -> GELU -> Linear(1) + Highway(27->1)
Trainable Parameters: 30,589 | Optimizer: AdamW + CosineAnnealingLR + EarlyStopping
Transformation: StandardScaler (Fit on Train Only) + Inverse Transform (kW)
Model: Stacked 2-Layer LSTM + Residual Highway [Multivariate, Hold-out]
==============================================================================
          PyTorch Deep Learning Convergence & Parameter Anatomy               
==============================================================================
Initial Train Loss:           0.016259   Best Val Loss:       0.004957
Final Train Loss:             0.005305   Best Epoch:           27 / 30  
Hardware Device:                  cuda   Train / Pred:    92.2252s / 0.7567s
------------------------------------------------------------------------------
LAYER-BY-LAYER TENSOR PARAMETER ANATOMY (Extracted from lstm_model.pt):
Layer / Tensor Name        Shape             Params      Mean      Std           [Min, Max]
------------------------------------------------------------------------------
lstm1.weight_ih_l0         (256, 4)           1,024  -0.00198  0.11173     [-0.464, +0.330]
lstm1.weight_hh_l0         (256, 64)         16,384  +0.00103  0.10558     [-0.583, +0.512]
lstm1.bias_ih_l0           (256,)               256  +0.01927  0.05633     [-0.092, +0.243]
lstm1.bias_hh_l0           (256,)               256  +0.01927  0.05633     [-0.092, +0.243]
lstm2.weight_ih_l0         (128, 64)          8,192  -0.00034  0.14125     [-0.621, +0.676]
lstm2.weight_hh_l0         (128, 32)          4,096  +0.00617  0.12757     [-0.497, +0.596]
lstm2.bias_ih_l0           (128,)               128  -0.00120  0.04343     [-0.102, +0.165]
lstm2.bias_hh_l0           (128,)               128  -0.00120  0.04343     [-0.102, +0.165]
norm.weight                (32,)                 32  +0.95623  0.10525     [+0.710, +1.295]
norm.bias                  (32,)                 32  -0.06904  0.07343     [-0.309, +0.004]
head.weight                (1, 32)               32  +0.00796  0.08202     [-0.174, +0.212]
head.bias                  (1,)                   1  +0.00796  0.00000     [+0.008, +0.008]
highway.weight             (1, 27)               27  +0.03547  0.15168     [-0.075, +0.779]
highway.bias               (1,)                   1  +0.00796  0.00000     [+0.008, +0.008]
------------------------------------------------------------------------------
LEARNED AUTOREGRESSIVE RESIDUAL HIGHWAY WEIGHTS (All 27 Connections: 24 Load Lags + 3 Exog):
  t-24:+0.1644 | t-23:+0.0738 | t-22:-0.0340 | t-21:+0.0034 | t-20:+0.0222 | t-19:+0.0110
  t-18:+0.0066 | t-17:+0.0307 | t-16:-0.0185 | t-15:+0.0235 | t-14:+0.0087 | t-13:-0.0749
  t-12:+0.0481 | t-11:-0.0281 | t-10:-0.0226 | t-09:+0.0061 | t-08:-0.0168 | t-07:-0.0098
  t-06:+0.0055 | t-05:-0.0332 | t-04:-0.0132 | t-03:+0.0255 | t-02:-0.0049 | t-01:+0.7792
  Exogenous Step t: temp_t=+0.0115 | humidity_t=-0.0068 | wind_t=+0.0004 | bias=+0.0080
------------------------------------------------------------------------------
EPOCH-BY-EPOCH TRAINING & VALIDATION CONVERGENCE HISTORY:
Epoch    Train MSE      Val MSE    Train MAE      Val MAE             Status
------------------------------------------------------------------------------
    1     0.016259     0.009691     0.089218     0.060351                   
    2     0.012045     0.008947     0.075351     0.059780                   
    3     0.011105     0.008665     0.072109     0.059786                   
    4     0.010616     0.008472     0.070968     0.059671                   
    5     0.009995     0.007529     0.069129     0.055175                   
    6     0.009110     0.006536     0.065825     0.051541                   
    7     0.008008     0.006475     0.061754     0.052572                   
    8     0.007594     0.006711     0.060215     0.055340                   
    9     0.007201     0.005733     0.058652     0.049135                   
   10     0.006923     0.006210     0.057255     0.051682                   
   11     0.006787     0.005810     0.056667     0.049450                   
   12     0.006678     0.005765     0.056178     0.049969                   
   13     0.006391     0.005473     0.054647     0.047805                   
   14     0.006408     0.006965     0.054504     0.056557                   
   15     0.006255     0.005729     0.054068     0.049244                   
   16     0.006182     0.005532     0.053390     0.047783                   
   17     0.005963     0.005628     0.052450     0.048282                   
   18     0.005887     0.005575     0.052132     0.047650                   
   19     0.005858     0.005205     0.051669     0.046013                   
   20     0.005742     0.005434     0.051144     0.047059                   
   21     0.005698     0.005107     0.050823     0.045291                   
   22     0.005632     0.005035     0.050558     0.044806                   
   23     0.005548     0.005079     0.050015     0.045327                   
   24     0.005499     0.004981     0.049738     0.044990                   
   25     0.005457     0.005129     0.049555     0.045424                   
   26     0.005405     0.005019     0.049242     0.044714                   
   27     0.005368     0.004957     0.049028     0.044649 <-- Best Checkpoint
   28     0.005351     0.004991     0.048919     0.044671                   
   29     0.005316     0.004996     0.048724     0.044675                   
   30     0.005305     0.005018     0.048624     0.044685                   
==============================================================================
Ljung-Box (L=24) (Q):     721.46   Jarque-Bera (JB):    351072.00
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     1.1835   Skew:                   0.8881
Prob(H) (two-sided):       0.0004   Kurtosis:              42.9837
Durbin-Watson (DW):        1.9519   MedAE (kW):        3345.3542
Residual Mean (kW):   -363.1559   Residual Std:        6560.3620
------------------------------------------------------------------------------
OUT-OF-SAMPLE EVALUATION METRICS (Original Scale kW, n_eval=5,260):
  MSE  = 43,170,232.1359 | RMSE = 6,570.4058 kW | MAE = 4,524.0441 kW
  MAPE =        2.0185% | R²   =     0.9944     | Max Error = 128,134.0402 kW
  [In-Sample Train (70%)] RMSE = 6,104.3941 kW | MAE = 4,069.9218 kW | MAPE = 1.9624% | R² = 0.9948
  [Validation Set (15%)]  RMSE = 5,968.2838 kW | MAE = 3,784.7711 kW | MAPE = 7.1742% | R² = 0.9901
  [5-Fold Expanding CV]   RMSE = 8,541.18 ± 2,917.78 kW | MAE = 5,811.89 ± 1,900.76 kW | R² = 0.9888 ± 0.0053
  [Weather Ablation]      With Weather RMSE = 6,570.41 kW vs. Without Weather RMSE = 6,946.54 kW (Delta = +5.41%)
==============================================================================
```

### 3.4. AR(24) [Univariate Differenced d=1, No Weather Ablation]
```text
VAR / AR Multivariate Model Summary - Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
==============================================================================
Run ID: 20261117_160932_3fe70f | Status: SUCCESS | Regime: Feature Ablation Study (Univariate Load Only — No Weather)
Data: Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
Date range: 2011-01-01 00:00:00 to 2015-01-01 00:00:00 (Freq: 1h)
Observations: 35,065 (Train: 24,545, Val: 5,260, Eval/Test: 5,260)
Lag Order (p): 24 | Differencing (d): 1 | Equations (K): 1 | Params: 25 (Target Eq) / 25 (System)
Transformation: First-Order Differencing (Delta Y_t = Y_t - Y_{t-1}) + Cumulative Inverse (kW)
Model: AR(24) [Univariate Differenced d=1, No Weather Ablation]
==============================================================================
             VAR / OLS Statistical Fit & Information Criteria                 
==============================================================================
Log Likelihood (Load Eq):  -250953.731   AIC (Load Eq):     502101.462
BIC (Load Eq):              502887.865   HQIC (Load Eq):    502356.233
FPE (Final Pred Error):    45779369.30   Device:                   cpu
Converged:                        True   Train / Pred:    1.9443s / 0.1909s
------------------------------------------------------------------------------
KEY AUTOREGRESSIVE PARAMETER ESTIMATES (const, L1..L3, L6, L12, L23, L24):
param                       coef     std err         z    P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const (Intercept)         9.1835     43.0396     0.213   0.8310    -75.1724     93.5394
L1.load                  -0.0716      0.0053   -13.416   0.0000     -0.0821     -0.0612
L2.load                  -0.1399      0.0055   -25.295   0.0000     -0.1507     -0.1291
L3.load                  -0.1276      0.0056   -22.875   0.0000     -0.1385     -0.1166
L6.load                  -0.1429      0.0056   -25.396   0.0000     -0.1539     -0.1319
L12.load                 -0.0882      0.0057   -15.388   0.0000     -0.0994     -0.0770
L18.load                 -0.1217      0.0057   -21.450   0.0000     -0.1328     -0.1106
L23.load                 -0.0266      0.0055    -4.804   0.0000     -0.0374     -0.0157
L24.load                  0.7001      0.0053   131.069   0.0000      0.6896      0.7106
==============================================================================
Ljung-Box (L=24) (Q):     808.92   Jarque-Bera (JB):    637647.80
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     1.6264   Skew:                   0.3554
Prob(H) (two-sided):       0.0000   Kurtosis:              56.9344
Durbin-Watson (DW):        2.3702   MedAE (kW):            3190.84
Residual Mean (kW):      -25.1870   Residual Std (kW):   7120.9475
------------------------------------------------------------------------------
OUT-OF-SAMPLE EVALUATION METRICS (Original Scale kW, n_eval=5,260):
  MSE  = 50,708,528.4883 | RMSE = 7,120.9921 kW | MAE = 4,569.4519 kW
  MAPE =        1.9958% | R²   =     0.9935    | Max Error = 129,977.55 kW
  [Weather Ablation]      With Weather RMSE = 7,122.36 kW vs. Without Weather RMSE = 7,120.99 kW (Delta = -0.02%)
==============================================================================
```

### 3.5. XGBoost Univariate [No Weather, 33 Load/Calendar Features]
```text
XGBoost Gradient Boosting Model Summary - Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
==============================================================================
Run ID: 20261117_160932_3fe70f | Status: SUCCESS | Regime: Feature Ablation Study (Univariate Load Only — No Weather)
Data: Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
Date range: 2011-01-01 00:00:00 to 2015-01-01 00:00:00 (Freq: 1h)
Observations: 35,065 (Train: 24,545, Val: 5,260, Eval/Test: 5,260)
Features: 33 Tabular Features | Trees Configured: 450 | Best Iteration: 449
Max Depth: 6 | Learning Rate (eta): 0.045 | Subsample: 0.85 | Colsample: 0.85
Regularization: L1 (alpha)=0.1, L2 (lambda)=1.0 | Tree Method: hist (cuda)
Transformation: StandardScaler (Fit on Train Only) + Inverse Transform to Original Scale (kW)
Model: XGBoost Univariate [No Weather, 33 Load/Calendar Features]
==============================================================================
            XGBoost Boosting Convergence & Feature Importance Fit             
==============================================================================
Train Scaled RMSE (Tree 1 -> End): 0.955735 -> 0.043310
Val Scaled RMSE   (Tree 1 -> Best): 0.821509 -> 0.066113
Converged:                        True   Train / Pred:    3.7017s / 0.0844s
------------------------------------------------------------------------------
BOOSTING ITERATION CONVERGENCE TRAJECTORY (Extracted from evals_result_):
  Tree #  Train RMSE (Scaled)  Val RMSE (Scaled)      Gen Gap           Status
------------------------------------------------------------------------------
       1             0.955735           0.821509    -0.134226     Initial Tree
      10             0.638113           0.550413    -0.087699         Boosting
      25             0.331821           0.289812    -0.042009         Boosting
      50             0.131115           0.120498    -0.010617         Boosting
     100             0.067032           0.071110    +0.004078         Boosting
     150             0.059268           0.068470    +0.009202         Boosting
     200             0.055143           0.067492    +0.012349         Boosting
     250             0.052056           0.066980    +0.014923         Boosting
     300             0.049547           0.066612    +0.017065         Boosting
     350             0.047346           0.066398    +0.019051         Boosting
     400             0.045240           0.066208    +0.020967         Boosting
     450             0.043310           0.066113    +0.022803 Best Val RMSE (Selected)
------------------------------------------------------------------------------
FEATURE IMPORTANCE AGGREGATED BY FUNCTIONAL GROUP (105 Engineered Features):
Feature Group                                  Count  Share (%)    Top Feature
------------------------------------------------------------------------------
1. Autoregressive Load Lags (load_lag_1h..24h)    24     93.99%   load_lag_24h
4. Exogenous Humidity Lags (humidity_lag_1h..2    24      1.84% humidity_lag_12h
3. Cyclical Fourier & Calendar (sin/cos, weeke     5      1.73%       cos_hour
5. Exogenous Temperature Lags (temperature_lag    24      1.61% temperature_lag_10h
2. Load Rolling & Trend Stats (roll_mean, roll     4      0.42% load_roll_mean_24h
6. Exogenous Wind Speed Lags (wind_speed_lag_1    24      0.40% wind_speed_lag_18h
------------------------------------------------------------------------------
TOP 15 DOMINANT FEATURES BY GAIN IMPORTANCE:
Rank Feature Name              Gain Importance   Share (%)   Cumulative (%)
------------------------------------------------------------------------------
   1 load_lag_24h                     0.628052      62.81%           62.81%
   2 load_lag_23h                     0.189816      18.98%           81.79%
   3 load_lag_1h                      0.093452       9.35%           91.13%
   4 cos_hour                         0.006380       0.64%           91.77%
   5 humidity_lag_12h                 0.005469       0.55%           92.32%
   6 sin_hour                         0.004539       0.45%           92.77%
   7 is_weekend                       0.003984       0.40%           93.17%
   8 load_lag_20h                     0.003592       0.36%           93.53%
   9 load_lag_21h                     0.003419       0.34%           93.87%
  10 load_lag_22h                     0.002926       0.29%           94.16%
  11 temperature_lag_10h              0.002186       0.22%           94.38%
  12 temperature_lag_20h              0.002125       0.21%           94.59%
  13 load_lag_19h                     0.001766       0.18%           94.77%
  14 humidity_lag_13h                 0.001659       0.17%           94.94%
  15 load_lag_2h                      0.001600       0.16%           95.10%
==============================================================================
Ljung-Box (L=24) (Q):    1465.22   Jarque-Bera (JB):    305182.63
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     1.1334   Skew:                   1.2998
Prob(H) (two-sided):       0.0088   Kurtosis:              40.2251
Durbin-Watson (DW):        1.7422   MedAE (kW):            3009.11
Residual Mean (kW):      -80.1242   Residual Std (kW):   6378.7302
------------------------------------------------------------------------------
OUT-OF-SAMPLE EVALUATION METRICS (Original Scale kW, n_eval=5,260):
  MSE  = 40,694,620.0475 | RMSE = 6,379.2335 kW | MAE = 4,280.1958 kW
  MAPE =        1.8586% | R²   =     0.9947    | Max Error = 128,919.73 kW
  [Weather Ablation]      With Weather RMSE = 6,482.45 kW vs. Without Weather RMSE = 6,379.23 kW (Delta = -1.62%)
==============================================================================
```

### 3.6. Stacked 2-Layer LSTM + Highway [No Weather, Input_Dim=1]
```text
Stacked 2-Layer LSTM (Univariate Ablation) Deep Learning Model Summary - Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
==============================================================================
Run ID: 20261117_160932_3fe70f | Status: SUCCESS | Regime: Feature Ablation Study (Univariate Load Only — No Weather)
Data: Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
Date range: 2011-01-01 00:00:00 to 2015-01-01 00:00:00 (Freq: 1h)
Observations: 35,065 (Train: 24,545, Val: 5,260, Eval/Test: 5,260)
Architecture: Input(24,1) -> LSTM(64) -> Dropout(0.1) -> LSTM(32) -> LayerNorm -> GELU -> Linear(1) + Highway(24->1)
Trainable Parameters: 30,589 | Optimizer: AdamW + CosineAnnealingLR + EarlyStopping
Transformation: StandardScaler (Fit on Train Only) + Inverse Transform (kW)
Model: Stacked 2-Layer LSTM + Highway [No Weather, Input_Dim=1]
==============================================================================
          PyTorch Deep Learning Convergence & Parameter Anatomy               
==============================================================================
Initial Train Loss:           0.016259   Best Val Loss:       0.004957
Final Train Loss:             0.005305   Best Epoch:           27 / 30  
Hardware Device:                  cuda   Train / Pred:    59.9464s / 0.5675s
==============================================================================
Ljung-Box (L=24) (Q):    1310.76   Jarque-Bera (JB):    233844.03
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     1.1780   Skew:                   0.9055
Prob(H) (two-sided):       0.0006   Kurtosis:              35.6143
Durbin-Watson (DW):        1.8465   MedAE (kW):        3663.7805
Residual Mean (kW):   -874.3347   Residual Std:        6891.2972
------------------------------------------------------------------------------
OUT-OF-SAMPLE EVALUATION METRICS (Original Scale kW, n_eval=5,260):
  MSE  = 48,254,437.4219 | RMSE = 6,946.5414 kW | MAE = 4,862.9039 kW
  MAPE =        2.1741% | R²   =     0.9938     | Max Error = 124,491.6855 kW
  [Weather Ablation]      With Weather RMSE = 6,570.41 kW vs. Without Weather RMSE = 6,946.54 kW (Delta = +5.41%)
==============================================================================
```

### 3.7. XGBoost [Walk-Forward Backtesting 168h Online Feedback]
```text
XGBoost Gradient Boosting Model Summary - Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
==============================================================================
Run ID: 20261117_160932_3fe70f | Status: SUCCESS | Regime: Walk-Forward Backtesting (168 Hours Rolling 1-Step Feedback)
Data: Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
Date range: 2014-05-26 21:00:00 to 2014-06-02 20:00:00 (Freq: 1h)
Observations: 35,065 (Train: 24,545, Val: 5,260, Eval/Test: 168)
Features: 105 Tabular Features | Trees Configured: 450 | Best Iteration: 449
Max Depth: 6 | Learning Rate (eta): 0.045 | Subsample: 0.85 | Colsample: 0.85
Regularization: L1 (alpha)=0.1, L2 (lambda)=1.0 | Tree Method: hist (cuda)
Transformation: StandardScaler (Fit on Train Only) + Inverse Transform to Original Scale (kW)
Model: XGBoost [Walk-Forward Backtesting 168h Online Feedback]
==============================================================================
            XGBoost Boosting Convergence & Feature Importance Fit             
==============================================================================
Train Scaled RMSE (Tree 1 -> End): 0.955735 -> 0.043310
Val Scaled RMSE   (Tree 1 -> Best): 0.821509 -> 0.066113
Converged:                        True   Train / Pred:    5.6949s / 0.1125s
------------------------------------------------------------------------------
BOOSTING ITERATION CONVERGENCE TRAJECTORY (Extracted from evals_result_):
  Tree #  Train RMSE (Scaled)  Val RMSE (Scaled)      Gen Gap           Status
------------------------------------------------------------------------------
       1             0.955735           0.821509    -0.134226     Initial Tree
      10             0.638113           0.550413    -0.087699         Boosting
      25             0.331821           0.289812    -0.042009         Boosting
      50             0.131115           0.120498    -0.010617         Boosting
     100             0.067032           0.071110    +0.004078         Boosting
     150             0.059268           0.068470    +0.009202         Boosting
     200             0.055143           0.067492    +0.012349         Boosting
     250             0.052056           0.066980    +0.014923         Boosting
     300             0.049547           0.066612    +0.017065         Boosting
     350             0.047346           0.066398    +0.019051         Boosting
     400             0.045240           0.066208    +0.020967         Boosting
     450             0.043310           0.066113    +0.022803 Best Val RMSE (Selected)
------------------------------------------------------------------------------
FEATURE IMPORTANCE AGGREGATED BY FUNCTIONAL GROUP (105 Engineered Features):
Feature Group                                  Count  Share (%)    Top Feature
------------------------------------------------------------------------------
1. Autoregressive Load Lags (load_lag_1h..24h)    24     93.99%   load_lag_24h
4. Exogenous Humidity Lags (humidity_lag_1h..2    24      1.84% humidity_lag_12h
3. Cyclical Fourier & Calendar (sin/cos, weeke     5      1.73%       cos_hour
5. Exogenous Temperature Lags (temperature_lag    24      1.61% temperature_lag_10h
2. Load Rolling & Trend Stats (roll_mean, roll     4      0.42% load_roll_mean_24h
6. Exogenous Wind Speed Lags (wind_speed_lag_1    24      0.40% wind_speed_lag_18h
------------------------------------------------------------------------------
TOP 15 DOMINANT FEATURES BY GAIN IMPORTANCE:
Rank Feature Name              Gain Importance   Share (%)   Cumulative (%)
------------------------------------------------------------------------------
   1 load_lag_24h                     0.628052      62.81%           62.81%
   2 load_lag_23h                     0.189816      18.98%           81.79%
   3 load_lag_1h                      0.093452       9.35%           91.13%
   4 cos_hour                         0.006380       0.64%           91.77%
   5 humidity_lag_12h                 0.005469       0.55%           92.32%
   6 sin_hour                         0.004539       0.45%           92.77%
   7 is_weekend                       0.003984       0.40%           93.17%
   8 load_lag_20h                     0.003592       0.36%           93.53%
   9 load_lag_21h                     0.003419       0.34%           93.87%
  10 load_lag_22h                     0.002926       0.29%           94.16%
  11 temperature_lag_10h              0.002186       0.22%           94.38%
  12 temperature_lag_20h              0.002125       0.21%           94.59%
  13 load_lag_19h                     0.001766       0.18%           94.77%
  14 humidity_lag_13h                 0.001659       0.17%           94.94%
  15 load_lag_2h                      0.001600       0.16%           95.10%
==============================================================================
Ljung-Box (L=24) (Q):      49.35   Jarque-Bera (JB):        13.89
Prob(Q):                   0.0017   Prob(JB):               0.0010
Heteroskedasticity(H):     2.6184   Skew:                   0.5174
Prob(H) (two-sided):       0.0004   Kurtosis:               3.9556
Durbin-Watson (DW):        2.5024   MedAE (kW):            2621.39
Residual Mean (kW):      920.1650   Residual Std (kW):   4782.9278
------------------------------------------------------------------------------
OUT-OF-SAMPLE EVALUATION METRICS (Original Scale kW, n_eval=168):
  MSE  = 23,723,101.9398 | RMSE = 4,870.6367 kW | MAE = 3,582.8443 kW
  MAPE =        1.6381% | R²   =     0.9950    | Max Error =  17,249.55 kW
==============================================================================
```

### 3.8. VAR [Walk-Forward Backtesting 168h Online Feedback]
```text
VAR / AR Multivariate Model Summary - Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
==============================================================================
Run ID: 20261117_160932_3fe70f | Status: SUCCESS | Regime: Walk-Forward Backtesting (168 Hours Rolling 1-Step Feedback)
Data: Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
Date range: 2014-05-26 21:00:00 to 2014-06-02 20:00:00 (Freq: 1h)
Observations: 35,065 (Train: 24,545, Val: 5,260, Eval/Test: 168)
Lag Order (p): 24 | Differencing (d): 1 | Equations (K): 4 | Params: 97 (Target Eq) / 388 (System)
Transformation: First-Order Differencing (Delta Y_t = Y_t - Y_{t-1}) + Cumulative Inverse (kW)
Model: VAR [Walk-Forward Backtesting 168h Online Feedback]
==============================================================================
             VAR / OLS Statistical Fit & Information Criteria                 
==============================================================================
Log Likelihood (Load Eq):  -250953.731   AIC (Load Eq):     502101.462
BIC (Load Eq):              502887.865   HQIC (Load Eq):    502356.233
FPE (Final Pred Error):    45779369.30   Device:                   cpu
Converged:                        True   Train / Pred:    2.9913s / 0.2545s
------------------------------------------------------------------------------
CROSS-VARIABLE LAG INFLUENCE ON TARGET EQUATION (Delta_load_t, p=24 Lags):
Variable          Lags    Sum(Coef)      L1 Norm      L2 Norm    Dominant Lag
------------------------------------------------------------------------------
load                24      -2.1075       3.5077       0.9214 L24.load (+0.70)
temperature         24    -247.1308    1335.1613     344.4209 L15.temperature (-167.13)
humidity            24     -42.0006     322.4772      80.7685 L17.humidity (+35.21)
wind_speed          24    -700.8295    1203.9298     294.3422 L6.wind_speed (-130.51)
------------------------------------------------------------------------------
KEY AUTOREGRESSIVE PARAMETER ESTIMATES (const, L1..L3, L6, L12, L23, L24):
param                       coef     std err         z    P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const (Intercept)         9.1835     43.0396     0.213   0.8310    -75.1724     93.5394
L1.load                  -0.0716      0.0053   -13.416   0.0000     -0.0821     -0.0612
L1.temperature           46.3073     60.9754     0.759   0.4476    -73.2020    165.8167
L1.humidity              -3.6787     21.5707    -0.171   0.8646    -45.9564     38.5991
L1.wind_speed           -19.7575     57.3160    -0.345   0.7303   -132.0945     92.5796
L2.load                  -0.1399      0.0055   -25.295   0.0000     -0.1507     -0.1291
L3.load                  -0.1276      0.0056   -22.875   0.0000     -0.1385     -0.1166
L6.load                  -0.1429      0.0056   -25.396   0.0000     -0.1539     -0.1319
L12.load                 -0.0882      0.0057   -15.388   0.0000     -0.0994     -0.0770
L12.temperature        -116.4004    106.8363    -1.090   0.2759   -325.7953     92.9944
L12.humidity            -25.8350     44.0098    -0.587   0.5572   -112.0924     60.4225
L12.wind_speed          -75.5223     94.4715    -0.799   0.4240   -260.6827    109.6380
L18.load                 -0.1217      0.0057   -21.450   0.0000     -0.1328     -0.1106
L23.load                 -0.0266      0.0055    -4.804   0.0000     -0.0374     -0.0157
L24.load                  0.7001      0.0053   131.069   0.0000      0.6896      0.7106
L24.temperature          19.1763     60.9818     0.314   0.7532   -100.3456    138.6981
L24.humidity            -15.1906     21.5733    -0.704   0.4813    -57.4735     27.0923
L24.wind_speed           85.2553     57.3027     1.488   0.1368    -27.0558    197.5664
==============================================================================
Ljung-Box (L=24) (Q):      77.87   Jarque-Bera (JB):        19.92
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     3.1972   Skew:                   0.2272
Prob(H) (two-sided):       0.0000   Kurtosis:               4.6246
Durbin-Watson (DW):        2.9518   MedAE (kW):            3067.45
Residual Mean (kW):      379.6501   Residual Std (kW):   5269.2096
------------------------------------------------------------------------------
OUT-OF-SAMPLE EVALUATION METRICS (Original Scale kW, n_eval=168):
  MSE  = 27,908,703.5346 | RMSE = 5,282.8689 kW | MAE = 3,937.5303 kW
  MAPE =        1.8182% | R²   =     0.9941    | Max Error =  18,706.95 kW
==============================================================================
```

### 3.9. LSTM [Walk-Forward Backtesting 168h Online Feedback]
```text
Stacked 2-Layer LSTM (Walk-Forward 168h) Deep Learning Model Summary - Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
==============================================================================
Run ID: 20261117_160932_3fe70f | Status: SUCCESS | Regime: Walk-Forward Backtesting (168 Hours Rolling 1-Step Feedback)
Data: Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
Date range: 2014-05-26 21:00:00 to 2014-06-02 20:00:00 (Freq: 1h)
Observations: 35,065 (Train: 24,545, Val: 5,260, Eval/Test: 168)
Architecture: Stacked 2-Layer LSTM + Residual Highway + Online 1-Step Kalman/AR Feedback (gain=0.25)
Trainable Parameters: 30,589 | Optimizer: AdamW + CosineAnnealingLR + EarlyStopping
Transformation: StandardScaler (Fit on Train Only) + Inverse Transform (kW)
Model: LSTM [Walk-Forward Backtesting 168h Online Feedback]
==============================================================================
          PyTorch Deep Learning Convergence & Parameter Anatomy               
==============================================================================
Initial Train Loss:           0.016259   Best Val Loss:       0.004957
Final Train Loss:             0.005305   Best Epoch:           27 / 30  
Hardware Device:                  cuda   Train / Pred:    92.2252s / 0.7567s
==============================================================================
Ljung-Box (L=24) (Q):     160.81   Jarque-Bera (JB):        54.02
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     2.5308   Skew:                  -0.1508
Prob(H) (two-sided):       0.0007   Kurtosis:               5.7615
Durbin-Watson (DW):        2.8172   MedAE (kW):        3115.1892
Residual Mean (kW):    355.7829   Residual Std:        5799.8054
------------------------------------------------------------------------------
OUT-OF-SAMPLE EVALUATION METRICS (Original Scale kW, n_eval=168):
  MSE  = 33,764,323.7447 | RMSE = 5,810.7077 kW | MAE = 4,238.7778 kW
  MAPE =        1.9522% | R²   =     0.9929     | Max Error = 26,100.6662 kW
==============================================================================
```

### 3.10. VAR [5-Fold TimeSeriesSplit — Fold 1/5]
```text
5-Fold Expanding Window CV Model Summary - VAR [5-Fold TimeSeriesSplit — Fold 1/5]
==============================================================================
Run ID: 20261117_160932_3fe70f | Status: SUCCESS | Fold: 1 / 5 (Leakage-Free In-Fold Scaler)
Data: Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
Window Period: Train [2011-01-01 00:00:00..2011-09-01 12:00:00] -> Val [2011-09-01 13:00:00..2012-05-02 00:00:00]
Observations: Train^(k) = 5,845 hours | Val^(k) = 5,844 hours | Device: cpu
Model Family: VAR | Params / Features: 97 | Train Time: 0.8302s
==============================================================================
Ljung-Box (L=24) (Q):     733.40   Jarque-Bera (JB):   2373458.21
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     1.8266   Skew:                   0.3896
Prob(H) (two-sided):       0.0000   Kurtosis:             101.7251
Durbin-Watson (DW):        2.0353   MedAE (kW):            2645.73
Residual Mean (kW):      -63.7256   Residual Std (kW):   6286.6713
------------------------------------------------------------------------------
OUT-OF-SAMPLE FOLD 1 VALIDATION METRICS (Original Scale kW, n_val=5,844):
  MSE  = 39,526,297.0125 | RMSE = 6,286.9943 kW | MAE = 3,759.0343 kW
  MAPE =        2.2744% | R²   =     0.9915    | Max Error = 131,469.06 kW
  [5-Fold CV Mean ± Std]  RMSE = 7,053.54 ± 970.44 kW | R² = 0.9925 ± 0.0012
==============================================================================
```

### 3.11. XGBoost [5-Fold TimeSeriesSplit — Fold 1/5]
```text
5-Fold Expanding Window CV Model Summary - XGBoost [5-Fold TimeSeriesSplit — Fold 1/5]
==============================================================================
Run ID: 20261117_160932_3fe70f | Status: SUCCESS | Fold: 1 / 5 (Leakage-Free In-Fold Scaler)
Data: Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
Window Period: Train [2011-01-01 00:00:00..2011-09-01 12:00:00] -> Val [2011-09-01 13:00:00..2012-05-02 00:00:00]
Observations: Train^(k) = 5,845 hours | Val^(k) = 5,844 hours | Device: cuda
Model Family: XGBoost | Params / Features: 105 | Train Time: 4.9675s
==============================================================================
Ljung-Box (L=24) (Q):   14424.73   Jarque-Bera (JB):    113972.23
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     7.4358   Skew:                  -2.7737
Prob(H) (two-sided):       0.0000   Kurtosis:              23.9113
Durbin-Watson (DW):        0.4850   MedAE (kW):            5410.38
Residual Mean (kW):    -4603.5060   Residual Std (kW):  11983.1532
------------------------------------------------------------------------------
OUT-OF-SAMPLE FOLD 1 VALIDATION METRICS (Original Scale kW, n_val=5,844):
  MSE  = 164,788,227.0776 | RMSE = 12,836.9867 kW | MAE = 8,234.1254 kW
  MAPE =        5.2542% | R²   =     0.9647    | Max Error = 138,133.62 kW
  [5-Fold CV Mean ± Std]  RMSE = 10,229.36 ± 5,111.40 kW | R² = 0.9824 ± 0.0143
==============================================================================
```

### 3.12. LSTM [5-Fold TimeSeriesSplit — Fold 1/5]
```text
5-Fold Expanding Window CV Model Summary - LSTM [5-Fold TimeSeriesSplit — Fold 1/5]
==============================================================================
Run ID: 20261117_160932_3fe70f | Status: SUCCESS | Fold: 1 / 5 (Leakage-Free In-Fold Scaler)
Data: Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
Window Period: Train [2011-01-01 00:00:00..2011-09-01 12:00:00] -> Val [2011-09-01 13:00:00..2012-05-02 00:00:00]
Observations: Train^(k) = 5,845 hours | Val^(k) = 5,844 hours | Device: cuda
Model Family: LSTM | Params / Features: 30,589 | Train Time: 10.8592s
==============================================================================
Ljung-Box (L=24) (Q):    3820.59   Jarque-Bera (JB):     95732.99
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     1.8829   Skew:                  -0.7560
Prob(H) (two-sided):       0.0000   Kurtosis:              22.7704
Durbin-Watson (DW):        1.3610   MedAE (kW):            4664.71
Residual Mean (kW):    -1972.5290   Residual Std (kW):   8445.6538
------------------------------------------------------------------------------
OUT-OF-SAMPLE FOLD 1 VALIDATION METRICS (Original Scale kW, n_val=5,844):
  MSE  = 75,219,939.1400 | RMSE = 8,672.9429 kW | MAE = 6,106.5271 kW
  MAPE =        3.6206% | R²   =     0.9839    | Max Error = 134,121.20 kW
  [5-Fold CV Mean ± Std]  RMSE = 8,541.18 ± 2,917.78 kW | R² = 0.9888 ± 0.0053
==============================================================================
```

### 3.13. VAR [5-Fold TimeSeriesSplit — Fold 2/5]
```text
5-Fold Expanding Window CV Model Summary - VAR [5-Fold TimeSeriesSplit — Fold 2/5]
==============================================================================
Run ID: 20261117_160932_3fe70f | Status: SUCCESS | Fold: 2 / 5 (Leakage-Free In-Fold Scaler)
Data: Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
Window Period: Train [2011-01-01 00:00:00..2012-05-02 00:00:00] -> Val [2012-05-02 01:00:00..2012-12-31 12:00:00]
Observations: Train^(k) = 11,689 hours | Val^(k) = 5,844 hours | Device: cpu
Model Family: VAR | Params / Features: 97 | Train Time: 1.5272s
==============================================================================
Ljung-Box (L=24) (Q):     723.86   Jarque-Bera (JB):    157069.09
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     0.7015   Skew:                   0.1124
Prob(H) (two-sided):       0.0000   Kurtosis:              28.3968
Durbin-Watson (DW):        2.3520   MedAE (kW):            3541.17
Residual Mean (kW):      -11.0740   Residual Std (kW):   8731.4571
------------------------------------------------------------------------------
OUT-OF-SAMPLE FOLD 2 VALIDATION METRICS (Original Scale kW, n_val=5,844):
  MSE  = 76,238,466.1853 | RMSE = 8,731.4641 kW | MAE = 5,508.4725 kW
  MAPE =        2.2799% | R²   =     0.9926    | Max Error = 128,177.18 kW
  [5-Fold CV Mean ± Std]  RMSE = 7,053.54 ± 970.44 kW | R² = 0.9925 ± 0.0012
==============================================================================
```

### 3.14. XGBoost [5-Fold TimeSeriesSplit — Fold 2/5]
```text
5-Fold Expanding Window CV Model Summary - XGBoost [5-Fold TimeSeriesSplit — Fold 2/5]
==============================================================================
Run ID: 20261117_160932_3fe70f | Status: SUCCESS | Fold: 2 / 5 (Leakage-Free In-Fold Scaler)
Data: Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
Window Period: Train [2011-01-01 00:00:00..2012-05-02 00:00:00] -> Val [2012-05-02 01:00:00..2012-12-31 12:00:00]
Observations: Train^(k) = 11,689 hours | Val^(k) = 5,844 hours | Device: cuda
Model Family: XGBoost | Params / Features: 105 | Train Time: 3.4974s
==============================================================================
Ljung-Box (L=24) (Q):   13686.20   Jarque-Bera (JB):     14796.31
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     0.2452   Skew:                   1.6936
Prob(H) (two-sided):       0.0000   Kurtosis:              10.0208
Durbin-Watson (DW):        0.8209   MedAE (kW):            5276.89
Residual Mean (kW):     2856.8736   Residual Std (kW):  17684.9836
------------------------------------------------------------------------------
OUT-OF-SAMPLE FOLD 2 VALIDATION METRICS (Original Scale kW, n_val=5,844):
  MSE  = 320,920,372.4353 | RMSE = 17,914.2505 kW | MAE = 10,471.7941 kW
  MAPE =        3.8495% | R²   =     0.9690    | Max Error = 129,361.15 kW
  [5-Fold CV Mean ± Std]  RMSE = 10,229.36 ± 5,111.40 kW | R² = 0.9824 ± 0.0143
==============================================================================
```

### 3.15. LSTM [5-Fold TimeSeriesSplit — Fold 2/5]
```text
5-Fold Expanding Window CV Model Summary - LSTM [5-Fold TimeSeriesSplit — Fold 2/5]
==============================================================================
Run ID: 20261117_160932_3fe70f | Status: SUCCESS | Fold: 2 / 5 (Leakage-Free In-Fold Scaler)
Data: Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
Window Period: Train [2011-01-01 00:00:00..2012-05-02 00:00:00] -> Val [2012-05-02 01:00:00..2012-12-31 12:00:00]
Observations: Train^(k) = 11,689 hours | Val^(k) = 5,844 hours | Device: cuda
Model Family: LSTM | Params / Features: 30,589 | Train Time: 13.3197s
==============================================================================
Ljung-Box (L=24) (Q):    4285.88   Jarque-Bera (JB):     10749.08
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     0.4985   Skew:                   0.7477
Prob(H) (two-sided):       0.0000   Kurtosis:               9.4736
Durbin-Watson (DW):        1.4323   MedAE (kW):            5892.16
Residual Mean (kW):    -1588.5417   Residual Std (kW):  13426.9686
------------------------------------------------------------------------------
OUT-OF-SAMPLE FOLD 2 VALIDATION METRICS (Original Scale kW, n_val=5,844):
  MSE  = 182,806,950.9156 | RMSE = 13,520.6121 kW | MAE = 8,974.1465 kW
  MAPE =        3.7969% | R²   =     0.9824    | Max Error = 122,342.05 kW
  [5-Fold CV Mean ± Std]  RMSE = 8,541.18 ± 2,917.78 kW | R² = 0.9888 ± 0.0053
==============================================================================
```

### 3.16. VAR [5-Fold TimeSeriesSplit — Fold 3/5]
```text
5-Fold Expanding Window CV Model Summary - VAR [5-Fold TimeSeriesSplit — Fold 3/5]
==============================================================================
Run ID: 20261117_160932_3fe70f | Status: SUCCESS | Fold: 3 / 5 (Leakage-Free In-Fold Scaler)
Data: Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
Window Period: Train [2011-01-01 00:00:00..2012-12-31 12:00:00] -> Val [2012-12-31 13:00:00..2013-09-01 00:00:00]
Observations: Train^(k) = 17,533 hours | Val^(k) = 5,844 hours | Device: cpu
Model Family: VAR | Params / Features: 97 | Train Time: 2.2394s
==============================================================================
Ljung-Box (L=24) (Q):     655.04   Jarque-Bera (JB):    886098.23
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     2.9322   Skew:                   0.3514
Prob(H) (two-sided):       0.0000   Kurtosis:              63.3201
Durbin-Watson (DW):        2.2766   MedAE (kW):            2685.41
Residual Mean (kW):       33.7140   Residual Std (kW):   6685.3903
------------------------------------------------------------------------------
OUT-OF-SAMPLE FOLD 3 VALIDATION METRICS (Original Scale kW, n_val=5,844):
  MSE  = 44,695,579.5233 | RMSE = 6,685.4753 kW | MAE = 4,065.4171 kW
  MAPE =        2.3303% | R²   =     0.9938    | Max Error = 126,810.22 kW
  [5-Fold CV Mean ± Std]  RMSE = 7,053.54 ± 970.44 kW | R² = 0.9925 ± 0.0012
==============================================================================
```

### 3.17. XGBoost [5-Fold TimeSeriesSplit — Fold 3/5]
```text
5-Fold Expanding Window CV Model Summary - XGBoost [5-Fold TimeSeriesSplit — Fold 3/5]
==============================================================================
Run ID: 20261117_160932_3fe70f | Status: SUCCESS | Fold: 3 / 5 (Leakage-Free In-Fold Scaler)
Data: Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
Window Period: Train [2011-01-01 00:00:00..2012-12-31 12:00:00] -> Val [2012-12-31 13:00:00..2013-09-01 00:00:00]
Observations: Train^(k) = 17,533 hours | Val^(k) = 5,844 hours | Device: cuda
Model Family: XGBoost | Params / Features: 105 | Train Time: 1.8797s
==============================================================================
Ljung-Box (L=24) (Q):    4979.06   Jarque-Bera (JB):     59919.47
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     0.8807   Skew:                  -0.3799
Prob(H) (two-sided):       0.0051   Kurtosis:              18.6684
Durbin-Watson (DW):        1.0880   MedAE (kW):            3986.20
Residual Mean (kW):     1418.1105   Residual Std (kW):   8111.7931
------------------------------------------------------------------------------
OUT-OF-SAMPLE FOLD 3 VALIDATION METRICS (Original Scale kW, n_val=5,844):
  MSE  = 67,812,225.2409 | RMSE = 8,234.8179 kW | MAE = 5,685.1361 kW
  MAPE =        3.1051% | R²   =     0.9905    | Max Error = 128,596.29 kW
  [5-Fold CV Mean ± Std]  RMSE = 10,229.36 ± 5,111.40 kW | R² = 0.9824 ± 0.0143
==============================================================================
```

### 3.18. LSTM [5-Fold TimeSeriesSplit — Fold 3/5]
```text
5-Fold Expanding Window CV Model Summary - LSTM [5-Fold TimeSeriesSplit — Fold 3/5]
==============================================================================
Run ID: 20261117_160932_3fe70f | Status: SUCCESS | Fold: 3 / 5 (Leakage-Free In-Fold Scaler)
Data: Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
Window Period: Train [2011-01-01 00:00:00..2012-12-31 12:00:00] -> Val [2012-12-31 13:00:00..2013-09-01 00:00:00]
Observations: Train^(k) = 17,533 hours | Val^(k) = 5,844 hours | Device: cuda
Model Family: LSTM | Params / Features: 30,589 | Train Time: 30.4732s
==============================================================================
Ljung-Box (L=24) (Q):    2052.64   Jarque-Bera (JB):    163354.83
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     2.2516   Skew:                   0.1422
Prob(H) (two-sided):       0.0000   Kurtosis:              28.8994
Durbin-Watson (DW):        1.7680   MedAE (kW):            3782.25
Residual Mean (kW):     -206.4309   Residual Std (kW):   7392.4831
------------------------------------------------------------------------------
OUT-OF-SAMPLE FOLD 3 VALIDATION METRICS (Original Scale kW, n_val=5,844):
  MSE  = 54,691,420.3121 | RMSE = 7,395.3648 kW | MAE = 5,055.6623 kW
  MAPE =        2.8405% | R²   =     0.9924    | Max Error = 124,251.58 kW
  [5-Fold CV Mean ± Std]  RMSE = 8,541.18 ± 2,917.78 kW | R² = 0.9888 ± 0.0053
==============================================================================
```

### 3.19. VAR [5-Fold TimeSeriesSplit — Fold 4/5]
```text
5-Fold Expanding Window CV Model Summary - VAR [5-Fold TimeSeriesSplit — Fold 4/5]
==============================================================================
Run ID: 20261117_160932_3fe70f | Status: SUCCESS | Fold: 4 / 5 (Leakage-Free In-Fold Scaler)
Data: Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
Window Period: Train [2011-01-01 00:00:00..2013-09-01 00:00:00] -> Val [2013-09-01 01:00:00..2014-05-02 12:00:00]
Observations: Train^(k) = 23,377 hours | Val^(k) = 5,844 hours | Device: cpu
Model Family: VAR | Params / Features: 97 | Train Time: 2.9518s
==============================================================================
Ljung-Box (L=24) (Q):    1173.80   Jarque-Bera (JB):   2602224.42
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     0.7937   Skew:                   0.0159
Prob(H) (two-sided):       0.0000   Kurtosis:             106.3767
Durbin-Watson (DW):        2.3427   MedAE (kW):            2493.74
Residual Mean (kW):      -49.2612   Residual Std (kW):   6580.5062
------------------------------------------------------------------------------
OUT-OF-SAMPLE FOLD 4 VALIDATION METRICS (Original Scale kW, n_val=5,844):
  MSE  = 43,305,487.9519 | RMSE = 6,580.6905 kW | MAE = 3,697.5826 kW
  MAPE =        6.7214% | R²   =     0.9910    | Max Error = 130,849.01 kW
  [5-Fold CV Mean ± Std]  RMSE = 7,053.54 ± 970.44 kW | R² = 0.9925 ± 0.0012
==============================================================================
```

### 3.20. XGBoost [5-Fold TimeSeriesSplit — Fold 4/5]
```text
5-Fold Expanding Window CV Model Summary - XGBoost [5-Fold TimeSeriesSplit — Fold 4/5]
==============================================================================
Run ID: 20261117_160932_3fe70f | Status: SUCCESS | Fold: 4 / 5 (Leakage-Free In-Fold Scaler)
Data: Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
Window Period: Train [2011-01-01 00:00:00..2013-09-01 00:00:00] -> Val [2013-09-01 01:00:00..2014-05-02 12:00:00]
Observations: Train^(k) = 23,377 hours | Val^(k) = 5,844 hours | Device: cuda
Model Family: XGBoost | Params / Features: 105 | Train Time: 6.2938s
==============================================================================
Ljung-Box (L=24) (Q):    1905.74   Jarque-Bera (JB):   1329217.18
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     0.7203   Skew:                  -0.9809
Prob(H) (two-sided):       0.0000   Kurtosis:              76.8576
Durbin-Watson (DW):        1.5827   MedAE (kW):            2631.23
Residual Mean (kW):       71.4036   Residual Std (kW):   5732.1040
------------------------------------------------------------------------------
OUT-OF-SAMPLE FOLD 4 VALIDATION METRICS (Original Scale kW, n_val=5,844):
  MSE  = 32,862,114.7716 | RMSE = 5,732.5487 kW | MAE = 3,700.0679 kW
  MAPE =        6.5246% | R²   =     0.9932    | Max Error = 126,565.58 kW
  [5-Fold CV Mean ± Std]  RMSE = 10,229.36 ± 5,111.40 kW | R² = 0.9824 ± 0.0143
==============================================================================
```

### 3.21. LSTM [5-Fold TimeSeriesSplit — Fold 4/5]
```text
5-Fold Expanding Window CV Model Summary - LSTM [5-Fold TimeSeriesSplit — Fold 4/5]
==============================================================================
Run ID: 20261117_160932_3fe70f | Status: SUCCESS | Fold: 4 / 5 (Leakage-Free In-Fold Scaler)
Data: Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
Window Period: Train [2011-01-01 00:00:00..2013-09-01 00:00:00] -> Val [2013-09-01 01:00:00..2014-05-02 12:00:00]
Observations: Train^(k) = 23,377 hours | Val^(k) = 5,844 hours | Device: cuda
Model Family: LSTM | Params / Features: 30,589 | Train Time: 39.5551s
==============================================================================
Ljung-Box (L=24) (Q):     801.37   Jarque-Bera (JB):   1184615.29
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     0.8915   Skew:                  -0.2048
Prob(H) (two-sided):       0.0113   Kurtosis:              72.7480
Durbin-Watson (DW):        1.8685   MedAE (kW):            3071.81
Residual Mean (kW):       26.4820   Residual Std (kW):   6375.8794
------------------------------------------------------------------------------
OUT-OF-SAMPLE FOLD 4 VALIDATION METRICS (Original Scale kW, n_val=5,844):
  MSE  = 40,652,538.9355 | RMSE = 6,375.9344 kW | MAE = 4,191.1987 kW
  MAPE =        7.0281% | R²   =     0.9915    | Max Error = 131,686.59 kW
  [5-Fold CV Mean ± Std]  RMSE = 8,541.18 ± 2,917.78 kW | R² = 0.9888 ± 0.0053
==============================================================================
```

### 3.22. VAR [5-Fold TimeSeriesSplit — Fold 5/5]
```text
5-Fold Expanding Window CV Model Summary - VAR [5-Fold TimeSeriesSplit — Fold 5/5]
==============================================================================
Run ID: 20261117_160932_3fe70f | Status: SUCCESS | Fold: 5 / 5 (Leakage-Free In-Fold Scaler)
Data: Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
Window Period: Train [2011-01-01 00:00:00..2014-05-02 12:00:00] -> Val [2014-05-02 13:00:00..2015-01-01 00:00:00]
Observations: Train^(k) = 29,221 hours | Val^(k) = 5,844 hours | Device: cpu
Model Family: VAR | Params / Features: 97 | Train Time: 3.7818s
==============================================================================
Ljung-Box (L=24) (Q):     779.92   Jarque-Bera (JB):    657619.95
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     1.7357   Skew:                   0.3667
Prob(H) (two-sided):       0.0000   Kurtosis:              54.9631
Durbin-Watson (DW):        2.3682   MedAE (kW):            3163.45
Residual Mean (kW):      -36.5437   Residual Std (kW):   6982.9700
------------------------------------------------------------------------------
OUT-OF-SAMPLE FOLD 5 VALIDATION METRICS (Original Scale kW, n_val=5,844):
  MSE  = 48,763,206.1510 | RMSE = 6,983.0657 kW | MAE = 4,527.4671 kW
  MAPE =        1.9895% | R²   =     0.9935    | Max Error = 130,503.43 kW
  [5-Fold CV Mean ± Std]  RMSE = 7,053.54 ± 970.44 kW | R² = 0.9925 ± 0.0012
==============================================================================
```

### 3.23. XGBoost [5-Fold TimeSeriesSplit — Fold 5/5]
```text
5-Fold Expanding Window CV Model Summary - XGBoost [5-Fold TimeSeriesSplit — Fold 5/5]
==============================================================================
Run ID: 20261117_160932_3fe70f | Status: SUCCESS | Fold: 5 / 5 (Leakage-Free In-Fold Scaler)
Data: Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
Window Period: Train [2011-01-01 00:00:00..2014-05-02 12:00:00] -> Val [2014-05-02 13:00:00..2015-01-01 00:00:00]
Observations: Train^(k) = 29,221 hours | Val^(k) = 5,844 hours | Device: cuda
Model Family: XGBoost | Params / Features: 105 | Train Time: 6.2370s
==============================================================================
Ljung-Box (L=24) (Q):    2011.82   Jarque-Bera (JB):    267853.90
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     1.2991   Skew:                   1.1267
Prob(H) (two-sided):       0.0000   Kurtosis:              36.0899
Durbin-Watson (DW):        1.6303   MedAE (kW):            3021.13
Residual Mean (kW):     -170.6816   Residual Std (kW):   6425.9428
------------------------------------------------------------------------------
OUT-OF-SAMPLE FOLD 5 VALIDATION METRICS (Original Scale kW, n_val=5,844):
  MSE  = 41,321,873.6470 | RMSE = 6,428.2092 kW | MAE = 4,319.2741 kW
  MAPE =        1.8771% | R²   =     0.9945    | Max Error = 129,118.48 kW
  [5-Fold CV Mean ± Std]  RMSE = 10,229.36 ± 5,111.40 kW | R² = 0.9824 ± 0.0143
==============================================================================
```

### 3.24. LSTM [5-Fold TimeSeriesSplit — Fold 5/5]
```text
5-Fold Expanding Window CV Model Summary - LSTM [5-Fold TimeSeriesSplit — Fold 5/5]
==============================================================================
Run ID: 20261117_160932_3fe70f | Status: SUCCESS | Fold: 5 / 5 (Leakage-Free In-Fold Scaler)
Data: Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
Window Period: Train [2011-01-01 00:00:00..2014-05-02 12:00:00] -> Val [2014-05-02 13:00:00..2015-01-01 00:00:00]
Observations: Train^(k) = 29,221 hours | Val^(k) = 5,844 hours | Device: cuda
Model Family: LSTM | Params / Features: 30,589 | Train Time: 49.2265s
==============================================================================
Ljung-Box (L=24) (Q):    1145.67   Jarque-Bera (JB):    258945.34
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     1.1541   Skew:                   0.8519
Prob(H) (two-sided):       0.0016   Kurtosis:              35.5657
Durbin-Watson (DW):        1.8892   MedAE (kW):            3572.37
Residual Mean (kW):     -263.6258   Residual Std (kW):   6735.8968
------------------------------------------------------------------------------
OUT-OF-SAMPLE FOLD 5 VALIDATION METRICS (Original Scale kW, n_val=5,844):
  MSE  = 45,441,803.6795 | RMSE = 6,741.0536 kW | MAE = 4,731.9143 kW
  MAPE =        2.1160% | R²   =     0.9940    | Max Error = 125,415.27 kW
  [5-Fold CV Mean ± Std]  RMSE = 8,541.18 ± 2,917.78 kW | R² = 0.9888 ± 0.0053
==============================================================================
```
