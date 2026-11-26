# BÁO CÁO TỔNG HỢP MÔ HÌNH (MODEL SUMMARY) — ELECTRICITY LOAD & WEATHER MULTIVARIATE TIME SERIES (UCI LD2011-2014, 01/2011 - 01/2015)

## 1. Thông tin Phiên Thực nghiệm & Bộ dữ liệu (Provenance & Dataset Info)
- **Run ID**: `20261124_193611_5cd58d`
- **Thời điểm xuất báo cáo**: `2026-11-26 21:07:52`
- **Dataset Key**: `dataset1` (`data/raw/electricity_load.csv`)
- **Tên bộ dữ liệu**: Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
- **Tổng số quan sát**: 35,065 giờ (2011-01-01 00:00:00 to 2015-01-01 00:00:00)
- **Số biến quan sát ($K = 4$)**: `load` (kW — Mục tiêu), `temperature` (°C), `humidity` (%), `wind_speed` (m/s)
- **Tần suất**: `1h` (Hourly)
- **Tập Huấn luyện (Train - 70%)**: 24,545 quan sát
- **Tập Xác thực (Validation - 15%)**: 5,260 quan sát
- **Tập Kiểm thử Độc lập (Hold-out Test - 15%)**: 5,260 quan sát
- **Tổng số thực thể mô hình được thống kê**: 24 (Thành công: 24, Thất bại: 0)
  - **03 Mô hình Đa biến Chính (Hold-out 70/15/15)**: `VAR(24)`, `XGBoost (106 Features)`, `Stacked 2-Layer LSTM + Residual Highway`
  - **03 Mô hình Đối chứng Loại bỏ Thời tiết (Feature Ablation)**: `AR(24) Univariate`, `XGBoost NoWeather`, `LSTM NoWeather`
  - **03 Mô hình Vận hành Cuốn chiếu Thực tế (Walk-Forward 168h)**: `VAR Walk-Forward`, `XGBoost Walk-Forward`, `LSTM Walk-Forward`
  - **15 Mô hình Kiểm định Chéo Mở rộng (`5-Fold TimeSeriesSplit`)**: `Fold 1..5` cho cả 3 họ mô hình `VAR`, `XGBoost`, `LSTM`

## 2. Bảng So sánh Tổng hợp Toàn bộ 24 Thực thể Mô hình trên Dataset 1 (Model Comparison Table)

> **Nguyên tắc đối chiếu**: Tất cả 5 chỉ số đánh giá (`MSE`, `RMSE`, `MAE`, `MAPE (%)`, `R²`) và kiểm định phần dư đều được tính toán trên **đơn vị vật lý gốc (kW)** sau khi áp dụng phép biến đổi ngược (`inverse_transform` hoặc cộng tích lũy sai phân bậc 1).

| # | Model | Family | Regime | Params / Feats | RMSE (kW) | MAE (kW) | MAPE (%) | R² | DW Stat | Ljung-Box Q | Device | Time (s) |
|---:|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|:---|---:|
| 1 | VAR(24) [Multivariate Differenced d=1, Hold-out] | VAR | `Hold-out_Test_15pct` | 97 | 7,122.36 | 4,576.49 | 2.0010 | 0.9935 | 2.3686 | 800.16 | `cpu` | 3.0565 |
| 2 | XGBoost Multivariate Regressor [106 Features, Hold-out] | XGBoost | `Hold-out_Test_15pct` | 106 | 6,320.97 | 4,273.05 | 1.8683 | 0.9948 | 1.8247 | 1,165.65 | `cuda` | 5.2138 |
| 3 | Stacked 2-Layer LSTM + Residual Highway [Multivariate, Hold-out] | LSTM | `Hold-out_Test_15pct` | 30,589 | 6,570.41 | 4,524.04 | 2.0185 | 0.9944 | 1.9519 | 721.46 | `cuda` | 97.0233 |
| 4 | AR(24) [Univariate Differenced d=1, No Weather Ablation] | AR-Ablation | `Ablation_No_Weather` | 25 | 7,120.99 | 4,569.45 | 1.9958 | 0.9935 | 2.3702 | 808.92 | `cpu` | 1.9867 |
| 5 | XGBoost Univariate [No Weather, 34 Load/Calendar Features] | XGBoost-Ablation | `Ablation_No_Weather` | 34 | 6,285.65 | 4,270.67 | 1.8571 | 0.9949 | 1.8921 | 833.61 | `cuda` | 3.3890 |
| 6 | Stacked 2-Layer LSTM + Highway [No Weather, Input_Dim=1] | LSTM-Ablation | `Ablation_No_Weather` | 29,818 | 7,078.22 | 4,930.62 | 2.2122 | 0.9935 | 1.7710 | 1,407.58 | `cuda` | 63.0651 |
| 7 | XGBoost [Walk-Forward Backtesting 168h Online Feedback] | XGBoost-WalkForward | `Walk_Forward_168h` | 106 | 5,356.26 | 3,973.05 | 1.8012 | 0.9940 | 2.5319 | 64.42 | `cuda` | 5.2138 |
| 8 | VAR [Walk-Forward Backtesting 168h Online Feedback] | VAR-WalkForward | `Walk_Forward_168h` | 97 | 5,282.87 | 3,937.53 | 1.8182 | 0.9941 | 2.9518 | 77.87 | `cpu` | 3.0565 |
| 9 | LSTM [Walk-Forward Backtesting 168h Online Feedback] | LSTM-WalkForward | `Walk_Forward_168h` | 30,589 | 5,810.71 | 4,238.78 | 1.9522 | 0.9929 | 2.8172 | 160.81 | `cuda` | 97.0233 |
| 10 | VAR [5-Fold TimeSeriesSplit — Fold 1/5] | VAR-CV-Fold1 | `5Fold_CV_Fold_1` | 97 | 5,787.58 | 3,841.17 | 1.7099 | 0.9938 | 1.7117 | 648.28 | `cpu` | 0.7028 |
| 11 | XGBoost [5-Fold TimeSeriesSplit — Fold 1/5] | XGBoost-CV-Fold1 | `5Fold_CV_Fold_1` | 106 | 11,718.06 | 6,691.03 | 3.5454 | 0.9747 | 0.5293 | 15,585.70 | `cuda` | 6.1018 |
| 12 | LSTM [5-Fold TimeSeriesSplit — Fold 1/5] | LSTM-CV-Fold1 | `5Fold_CV_Fold_1` | 30,589 | 8,537.15 | 5,980.96 | 2.8019 | 0.9866 | 1.2214 | 5,160.10 | `cuda` | 9.2671 |
| 13 | VAR [5-Fold TimeSeriesSplit — Fold 2/5] | VAR-CV-Fold2 | `5Fold_CV_Fold_2` | 97 | 8,793.84 | 5,499.72 | 2.8433 | 0.9927 | 2.3749 | 795.16 | `cpu` | 1.2285 |
| 14 | XGBoost [5-Fold TimeSeriesSplit — Fold 2/5] | XGBoost-CV-Fold2 | `5Fold_CV_Fold_2` | 106 | 18,281.70 | 10,829.08 | 4.5498 | 0.9684 | 0.7210 | 13,999.57 | `cuda` | 5.0186 |
| 15 | LSTM [5-Fold TimeSeriesSplit — Fold 2/5] | LSTM-CV-Fold2 | `5Fold_CV_Fold_2` | 30,589 | 13,336.46 | 8,866.25 | 4.3257 | 0.9832 | 1.5095 | 4,222.09 | `cuda` | 10.0490 |
| 16 | VAR [5-Fold TimeSeriesSplit — Fold 3/5] | VAR-CV-Fold3 | `5Fold_CV_Fold_3` | 97 | 7,213.96 | 3,871.26 | 2.4843 | 0.9891 | 2.3936 | 918.61 | `cpu` | 1.7816 |
| 17 | XGBoost [5-Fold TimeSeriesSplit — Fold 3/5] | XGBoost-CV-Fold3 | `5Fold_CV_Fold_3` | 106 | 8,892.65 | 5,870.95 | 3.4442 | 0.9834 | 1.2754 | 4,569.64 | `cuda` | 2.7909 |
| 18 | LSTM [5-Fold TimeSeriesSplit — Fold 3/5] | LSTM-CV-Fold3 | `5Fold_CV_Fold_3` | 30,589 | 8,183.92 | 5,488.88 | 3.3260 | 0.9860 | 1.6549 | 1,893.45 | `cuda` | 26.6055 |
| 19 | VAR [5-Fold TimeSeriesSplit — Fold 4/5] | VAR-CV-Fold4 | `5Fold_CV_Fold_4` | 97 | 7,237.27 | 4,600.41 | 1.9333 | 0.9938 | 2.3878 | 632.65 | `cpu` | 2.4096 |
| 20 | XGBoost [5-Fold TimeSeriesSplit — Fold 4/5] | XGBoost-CV-Fold4 | `5Fold_CV_Fold_4` | 106 | 7,055.79 | 4,715.85 | 1.9782 | 0.9941 | 1.7396 | 715.67 | `cuda` | 4.0364 |
| 21 | LSTM [5-Fold TimeSeriesSplit — Fold 4/5] | LSTM-CV-Fold4 | `5Fold_CV_Fold_4` | 30,589 | 7,784.54 | 5,585.09 | 2.4227 | 0.9929 | 1.9134 | 1,094.08 | `cuda` | 34.7935 |
| 22 | VAR [5-Fold TimeSeriesSplit — Fold 5/5] | VAR-CV-Fold5 | `5Fold_CV_Fold_5` | 97 | 5,991.19 | 3,511.07 | 7.5620 | 0.9898 | 2.1684 | 925.27 | `cpu` | 2.9439 |
| 23 | XGBoost [5-Fold TimeSeriesSplit — Fold 5/5] | XGBoost-CV-Fold5 | `5Fold_CV_Fold_5` | 106 | 5,227.53 | 3,497.29 | 7.3346 | 0.9922 | 1.6790 | 1,147.97 | `cuda` | 4.0384 |
| 24 | LSTM [5-Fold TimeSeriesSplit — Fold 5/5] | LSTM-CV-Fold5 | `5Fold_CV_Fold_5` | 30,589 | 5,982.23 | 3,977.50 | 7.8120 | 0.9898 | 1.8342 | 950.64 | `cuda` | 42.9271 |

### Diễn giải Đa Tiêu chí Tự động (Run ID: `20261124_193611_5cd58d` — Dataset 1: Electricity Load & Weather)

#### A. Mô hình Tốt nhất theo Từng Quy trình Kiểm định Độc lập (Regime-Specific Leaders)
- **Tiêu chí 1 — Độ chính xác cao nhất trên Tập Hold-out Test độc lập (15% - 5,260 giờ)**: **XGBoost** (`RMSE = 6,320.97 kW`, `MAE = 4,273.05 kW`, `MAPE = 1.8683%`, `R² = 0.9948`).
- **Tiêu chí 2 — Độ ổn định cao nhất trên Kiểm định Chéo 5-Fold TimeSeriesSplit (Expanding Window)**: **VAR** (`RMSE = 7,004.77 ± 1,204.93 kW`, `MAE = 4,264.73 ± 796.89 kW`, `R² = 0.9918 ± 0.0022`).
- **Tiêu chí 3 — Hiệu năng cao nhất trong Vận hành Cuốn chiếu Walk-Forward Backtesting (168 giờ)**: **VAR** (`RMSE = 5,282.87 kW`, `MAE = 3,937.53 kW`, `MAPE = 1.8182%`, `R² = 0.9941`).
- **Tiêu chí 4 — Mô hình khai thác biến ngoại sinh thời tiết hiệu quả nhất (Feature Ablation Impact)**: **Stacked 2-Layer LSTM + Residual Highway** giúp giảm `7.17%` sai số RMSE (từ `7,078.22 kW` khi không có thời tiết xuống `6,570.41 kW` khi kết hợp `temperature`, `humidity`, `wind_speed`) và giảm `8.76%` MAPE.
- **Tiêu chí 5 — Khử tự tương quan phần dư bậc 1 tốt nhất (Durbin-Watson gần 2.00 nhất & Ljung-Box Q thấp nhất trên Hold-out Test)**: **Stacked 2-Layer LSTM + Residual Highway** (`DW = 1.9519`, `Ljung-Box Q(24) = 721.46`, `Heteroskedasticity H = 1.1835`).

## 3. Hồ sơ Thống kê Chi tiết Từng Mô hình (Individual Model Cards — 24 Models)

### 3.1. VAR(24) [Multivariate Differenced d=1, Hold-out]
```text
VAR / AR Multivariate Model Summary - Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
==============================================================================
Run ID: 20261124_193611_5cd58d | Status: SUCCESS | Regime: Hold-out Split (70% Train - 15% Val - 15% Test)
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
Converged:                        True   Train / Pred:    3.0565s / 0.2574s
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
  [5-Fold Expanding CV]   RMSE = 7,004.77 ± 1,204.93 kW | MAE = 4,264.73 ± 796.89 kW | R² = 0.9918 ± 0.0022
  [Weather Ablation]      With Weather RMSE = 7,122.36 kW vs. Without Weather RMSE = 7,120.99 kW (Delta = -0.02%)
==============================================================================
```

### 3.2. XGBoost Multivariate Regressor [106 Features, Hold-out]
```text
XGBoost Gradient Boosting Model Summary - Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
==============================================================================
Run ID: 20261124_193611_5cd58d | Status: SUCCESS | Regime: Hold-out Split (70% Train - 15% Val - 15% Test)
Data: Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
Date range: 2011-01-01 00:00:00 to 2015-01-01 00:00:00 (Freq: 1h)
Observations: 35,065 (Train: 24,545, Val: 5,260, Eval/Test: 5,260)
Features: 106 Tabular Features | Trees Configured: 450 | Best Iteration: 358
Max Depth: 6 | Learning Rate (eta): 0.045 | Subsample: 0.85 | Colsample: 0.85
Regularization: L1 (alpha)=0.1, L2 (lambda)=1.0 | Tree Method: hist (cuda)
Transformation: StandardScaler (Fit on Train Only) + Inverse Transform to Original Scale (kW)
Model: XGBoost Multivariate Regressor [106 Features, Hold-out]
==============================================================================
            XGBoost Boosting Convergence & Feature Importance Fit             
==============================================================================
Train Scaled RMSE (Tree 1 -> End): 0.955722 -> 0.043671
Val Scaled RMSE   (Tree 1 -> Best): 0.821423 -> 0.065047
Converged:                        True   Train / Pred:    5.2138s / 0.0995s
------------------------------------------------------------------------------
BOOSTING ITERATION CONVERGENCE TRAJECTORY (Extracted from evals_result_):
  Tree #  Train RMSE (Scaled)  Val RMSE (Scaled)      Gen Gap           Status
------------------------------------------------------------------------------
       1             0.955722           0.821423    -0.134300     Initial Tree
      10             0.638098           0.550294    -0.087804         Boosting
      25             0.331789           0.289683    -0.042106         Boosting
      50             0.130762           0.120341    -0.010421         Boosting
     100             0.064996           0.069367    +0.004371         Boosting
     150             0.056804           0.066319    +0.009515         Boosting
     200             0.052635           0.065599    +0.012964         Boosting
     250             0.049817           0.065331    +0.015514         Boosting
     300             0.047330           0.065137    +0.017807         Boosting
     350             0.045153           0.065067    +0.019914         Boosting
     359             0.044828           0.065047    +0.020219 Best Val RMSE (Selected)
     389             0.043671           0.065053    +0.021382         Boosting
------------------------------------------------------------------------------
FEATURE IMPORTANCE AGGREGATED BY FUNCTIONAL GROUP (106 Engineered Features):
Feature Group                                  Count  Share (%)    Top Feature
------------------------------------------------------------------------------
1. Autoregressive Load Lags (load_lag_1h..24h)    24     96.13%   load_lag_24h
4. Exogenous Humidity Lags (humidity_lag_1h..2    24      1.33% humidity_lag_12h
5. Exogenous Temperature Lags (temperature_lag    24      0.99% temperature_lag_13h
3. Cyclical Fourier & Calendar (sin/cos, weeke     5      0.81%       sin_hour
2. Load Rolling & Trend Stats (roll, trend_24h     5      0.46% load_trend_24h
6. Exogenous Wind Speed Lags (wind_speed_lag_1    24      0.28% wind_speed_lag_21h
------------------------------------------------------------------------------
COMPLETE FEATURE IMPORTANCE RANKING (All 106/106 Features Extracted):
Rank Feature Name              Gain Importance   Share (%)   Cumulative (%)
------------------------------------------------------------------------------
   1 load_lag_24h                     0.711656      71.17%           71.17%
   2 load_lag_23h                     0.146604      14.66%           85.83%
   3 load_lag_1h                      0.083991       8.40%           94.23%
   4 humidity_lag_12h                 0.003471       0.35%           94.57%
   5 sin_hour                         0.002987       0.30%           94.87%
   6 load_lag_22h                     0.002617       0.26%           95.13%
   7 load_lag_20h                     0.002380       0.24%           95.37%
   8 cos_hour                         0.002286       0.23%           95.60%
   9 load_lag_21h                     0.002129       0.21%           95.81%
  10 load_trend_24h                   0.001835       0.18%           96.00%
  11 humidity_lag_11h                 0.001674       0.17%           96.16%
  12 temperature_lag_13h              0.001343       0.13%           96.30%
  13 load_lag_2h                      0.001257       0.13%           96.42%
  14 is_weekend                       0.001213       0.12%           96.54%
  15 humidity_lag_13h                 0.001181       0.12%           96.66%
  16 temperature_lag_20h              0.001045       0.10%           96.77%
  17 load_lag_15h                     0.001001       0.10%           96.87%
  18 load_lag_19h                     0.000971       0.10%           96.96%
  19 humidity_lag_21h                 0.000915       0.09%           97.06%
  20 load_lag_13h                     0.000905       0.09%           97.15%
  21 load_roll_mean_24h               0.000889       0.09%           97.23%
  22 load_roll_std_24h                0.000881       0.09%           97.32%
  23 sin_dayofweek                    0.000875       0.09%           97.41%
  24 load_lag_16h                     0.000803       0.08%           97.49%
  25 temperature_lag_12h              0.000783       0.08%           97.57%
  26 load_lag_14h                     0.000757       0.08%           97.64%
  27 load_lag_17h                     0.000749       0.07%           97.72%
  28 cos_dayofweek                    0.000713       0.07%           97.79%
  29 load_lag_11h                     0.000711       0.07%           97.86%
  30 load_lag_3h                      0.000705       0.07%           97.93%
  31 load_roll_std_6h                 0.000691       0.07%           98.00%
  32 humidity_lag_10h                 0.000645       0.06%           98.07%
  33 load_lag_12h                     0.000636       0.06%           98.13%
  34 temperature_lag_11h              0.000617       0.06%           98.19%
  35 load_lag_18h                     0.000589       0.06%           98.25%
  36 load_lag_10h                     0.000585       0.06%           98.31%
  37 temperature_lag_10h              0.000540       0.05%           98.36%
  38 temperature_lag_22h              0.000528       0.05%           98.42%
  39 humidity_lag_19h                 0.000525       0.05%           98.47%
  40 temperature_lag_21h              0.000513       0.05%           98.52%
  41 load_lag_4h                      0.000489       0.05%           98.57%
  42 temperature_lag_23h              0.000469       0.05%           98.62%
  43 temperature_lag_1h               0.000461       0.05%           98.66%
  44 humidity_lag_22h                 0.000459       0.05%           98.71%
  45 load_lag_6h                      0.000457       0.05%           98.75%
  46 humidity_lag_23h                 0.000452       0.05%           98.80%
  47 load_lag_8h                      0.000449       0.04%           98.84%
  48 temperature_lag_19h              0.000419       0.04%           98.89%
  49 humidity_lag_24h                 0.000406       0.04%           98.93%
  50 temperature_lag_24h              0.000386       0.04%           98.96%
  51 humidity_lag_1h                  0.000368       0.04%           99.00%
  52 humidity_lag_20h                 0.000364       0.04%           99.04%
  53 temperature_lag_14h              0.000364       0.04%           99.07%
  54 load_lag_9h                      0.000334       0.03%           99.11%
  55 humidity_lag_6h                  0.000332       0.03%           99.14%
  56 humidity_lag_7h                  0.000318       0.03%           99.17%
  57 load_roll_mean_6h                0.000317       0.03%           99.20%
  58 load_lag_7h                      0.000313       0.03%           99.24%
  59 temperature_lag_17h              0.000299       0.03%           99.27%
  60 humidity_lag_8h                  0.000282       0.03%           99.29%
  61 temperature_lag_8h               0.000275       0.03%           99.32%
  62 humidity_lag_18h                 0.000258       0.03%           99.35%
  63 temperature_lag_15h              0.000249       0.02%           99.37%
  64 humidity_lag_15h                 0.000240       0.02%           99.40%
  65 temperature_lag_9h               0.000235       0.02%           99.42%
  66 temperature_lag_16h              0.000232       0.02%           99.44%
  67 load_lag_5h                      0.000232       0.02%           99.47%
  68 temperature_lag_4h               0.000221       0.02%           99.49%
  69 humidity_lag_16h                 0.000210       0.02%           99.51%
  70 wind_speed_lag_21h               0.000209       0.02%           99.53%
  71 humidity_lag_14h                 0.000203       0.02%           99.55%
  72 humidity_lag_17h                 0.000203       0.02%           99.57%
  73 wind_speed_lag_22h               0.000202       0.02%           99.59%
  74 temperature_lag_5h               0.000191       0.02%           99.61%
  75 humidity_lag_9h                  0.000189       0.02%           99.63%
  76 temperature_lag_6h               0.000188       0.02%           99.65%
  77 wind_speed_lag_18h               0.000187       0.02%           99.67%
  78 humidity_lag_3h                  0.000182       0.02%           99.68%
  79 wind_speed_lag_14h               0.000177       0.02%           99.70%
  80 wind_speed_lag_19h               0.000176       0.02%           99.72%
  81 humidity_lag_2h                  0.000176       0.02%           99.74%
  82 temperature_lag_3h               0.000155       0.02%           99.75%
  83 wind_speed_lag_2h                0.000154       0.02%           99.77%
  84 wind_speed_lag_1h                0.000152       0.02%           99.78%
  85 humidity_lag_4h                  0.000150       0.01%           99.80%
  86 temperature_lag_18h              0.000147       0.01%           99.81%
  87 temperature_lag_2h               0.000120       0.01%           99.82%
  88 wind_speed_lag_7h                0.000115       0.01%           99.84%
  89 temperature_lag_7h               0.000111       0.01%           99.85%
  90 humidity_lag_5h                  0.000108       0.01%           99.86%
  91 wind_speed_lag_15h               0.000108       0.01%           99.87%
  92 wind_speed_lag_10h               0.000102       0.01%           99.88%
  93 wind_speed_lag_20h               0.000098       0.01%           99.89%
  94 wind_speed_lag_24h               0.000097       0.01%           99.90%
  95 wind_speed_lag_23h               0.000093       0.01%           99.91%
  96 wind_speed_lag_12h               0.000090       0.01%           99.92%
  97 wind_speed_lag_5h                0.000089       0.01%           99.93%
  98 wind_speed_lag_9h                0.000089       0.01%           99.93%
  99 wind_speed_lag_8h                0.000087       0.01%           99.94%
 100 wind_speed_lag_4h                0.000086       0.01%           99.95%
 101 wind_speed_lag_13h               0.000085       0.01%           99.96%
 102 wind_speed_lag_6h                0.000083       0.01%           99.97%
 103 wind_speed_lag_3h                0.000081       0.01%           99.98%
 104 wind_speed_lag_17h               0.000080       0.01%           99.98%
 105 wind_speed_lag_11h               0.000079       0.01%           99.99%
 106 wind_speed_lag_16h               0.000071       0.01%          100.00%
==============================================================================
Ljung-Box (L=24) (Q):    1165.65   Jarque-Bera (JB):    367765.21
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     1.1191   Skew:                   1.6379
Prob(H) (two-sided):       0.0185   Kurtosis:              43.8324
Durbin-Watson (DW):        1.8247   MedAE (kW):            3044.20
Residual Mean (kW):     -111.2241   Residual Std (kW):   6319.9956
------------------------------------------------------------------------------
OUT-OF-SAMPLE EVALUATION METRICS (Original Scale kW, n_eval=5,260):
  MSE  = 39,954,715.6783 | RMSE = 6,320.9743 kW | MAE = 4,273.0535 kW
  MAPE =        1.8683% | R²   =     0.9948    | Max Error = 132,576.21 kW
  [In-Sample Train (70%)] RMSE = 3,800.0003 kW | MAE = 2,755.1280 kW | MAPE = 1.2562% | R² = 0.9980
  [Validation Set (15%)]  RMSE = 5,513.8774 kW | MAE = 3,559.0320 kW | MAPE = 7.0071% | R² = 0.9915
  [5-Fold Expanding CV]   RMSE = 10,235.15 ± 5,097.38 kW | MAE = 6,320.84 ± 2,793.39 kW | R² = 0.9826 ± 0.0111
  [Weather Ablation]      With Weather RMSE = 6,320.97 kW vs. Without Weather RMSE = 6,285.65 kW (Delta = -0.56%)
==============================================================================
```

### 3.3. Stacked 2-Layer LSTM + Residual Highway [Multivariate, Hold-out]
```text
Stacked 2-Layer LSTM Deep Learning Model Summary - Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
==============================================================================
Run ID: 20261124_193611_5cd58d | Status: SUCCESS | Regime: Hold-out Split (70% Train - 15% Val - 15% Test)
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
Hardware Device:                  cuda   Train / Pred:    97.0233s / 0.8064s
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
  [5-Fold Expanding CV]   RMSE = 8,764.86 ± 2,738.29 kW | MAE = 5,979.73 ± 1,784.49 kW | R² = 0.9877 ± 0.0037
  [Weather Ablation]      With Weather RMSE = 6,570.41 kW vs. Without Weather RMSE = 7,078.22 kW (Delta = +7.17%)
==============================================================================
```

### 3.4. AR(24) [Univariate Differenced d=1, No Weather Ablation]
```text
VAR / AR Multivariate Model Summary - Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
==============================================================================
Run ID: 20261124_193611_5cd58d | Status: SUCCESS | Regime: Feature Ablation Study (Univariate Load Only — No Weather)
Data: Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
Date range: 2011-01-01 00:00:00 to 2015-01-01 00:00:00 (Freq: 1h)
Observations: 35,065 (Train: 24,545, Val: 5,260, Eval/Test: 5,260)
Lag Order (p): 24 | Differencing (d): 1 | Equations (K): 1 | Params: 25 (Target Eq) / 25 (System)
Transformation: First-Order Differencing (Delta Y_t = Y_t - Y_{t-1}) + Cumulative Inverse (kW)
Model: AR(24) [Univariate Differenced d=1, No Weather Ablation]
==============================================================================
             VAR / OLS Statistical Fit & Information Criteria                 
==============================================================================
Log Likelihood (Load Eq):  -252304.446   AIC (Load Eq):     504658.892
BIC (Load Eq):              504861.573   HQIC (Load Eq):    504724.555
FPE (Final Pred Error):    50812036.41   Device:                   cpu
Converged:                        True   Train / Pred:    1.9867s / 0.1931s
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

### 3.5. XGBoost Univariate [No Weather, 34 Load/Calendar Features]
```text
XGBoost Gradient Boosting Model Summary - Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
==============================================================================
Run ID: 20261124_193611_5cd58d | Status: SUCCESS | Regime: Feature Ablation Study (Univariate Load Only — No Weather)
Data: Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
Date range: 2011-01-01 00:00:00 to 2015-01-01 00:00:00 (Freq: 1h)
Observations: 35,065 (Train: 24,545, Val: 5,260, Eval/Test: 5,260)
Features: 34 Tabular Features | Trees Configured: 450 | Best Iteration: 358
Max Depth: 6 | Learning Rate (eta): 0.045 | Subsample: 0.85 | Colsample: 0.85
Regularization: L1 (alpha)=0.1, L2 (lambda)=1.0 | Tree Method: hist (cuda)
Transformation: StandardScaler (Fit on Train Only) + Inverse Transform to Original Scale (kW)
Model: XGBoost Univariate [No Weather, 34 Load/Calendar Features]
==============================================================================
            XGBoost Boosting Convergence & Feature Importance Fit             
==============================================================================
Converged:                        True   Train / Pred:    3.3890s / 0.0747s
------------------------------------------------------------------------------
FEATURE IMPORTANCE AGGREGATED BY FUNCTIONAL GROUP (34 Engineered Features):
Feature Group                                  Count  Share (%)    Top Feature
------------------------------------------------------------------------------
1. Autoregressive Load Lags (load_lag_1h..24h)    24     98.70%   load_lag_24h
3. Cyclical Fourier & Calendar (sin/cos, weeke     5      0.83%       sin_hour
2. Load Rolling & Trend Stats (roll, trend_24h     5      0.47% load_trend_24h
------------------------------------------------------------------------------
TOP 15 DOMINANT FEATURES BY GAIN IMPORTANCE:
Rank Feature Name              Gain Importance   Share (%)   Cumulative (%)
------------------------------------------------------------------------------
   1 load_lag_24h                     0.711656      73.06%           73.06%
   2 load_lag_23h                     0.146604      15.05%           88.12%
   3 load_lag_1h                      0.083991       8.62%           96.74%
   4 sin_hour                         0.002987       0.31%           97.05%
   5 load_lag_22h                     0.002617       0.27%           97.31%
   6 load_lag_20h                     0.002380       0.24%           97.56%
   7 cos_hour                         0.002286       0.23%           97.79%
   8 load_lag_21h                     0.002129       0.22%           98.01%
   9 load_trend_24h                   0.001835       0.19%           98.20%
  10 load_lag_2h                      0.001257       0.13%           98.33%
  11 is_weekend                       0.001213       0.12%           98.45%
  12 load_lag_15h                     0.001001       0.10%           98.56%
  13 load_lag_19h                     0.000971       0.10%           98.66%
  14 load_lag_13h                     0.000905       0.09%           98.75%
  15 load_roll_mean_24h               0.000889       0.09%           98.84%
==============================================================================
Ljung-Box (L=24) (Q):     833.61   Jarque-Bera (JB):    387131.49
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     1.0626   Skew:                   1.6871
Prob(H) (two-sided):       0.2035   Kurtosis:              44.8926
Durbin-Watson (DW):        1.8921   MedAE (kW):            3046.86
Residual Mean (kW):      -61.9715   Residual Std (kW):   6285.3406
------------------------------------------------------------------------------
OUT-OF-SAMPLE EVALUATION METRICS (Original Scale kW, n_eval=5,260):
  MSE  = 39,509,346.8944 | RMSE = 6,285.6461 kW | MAE = 4,270.6723 kW
  MAPE =        1.8571% | R²   =     0.9949    | Max Error = 132,242.04 kW
  [Weather Ablation]      With Weather RMSE = 6,320.97 kW vs. Without Weather RMSE = 6,285.65 kW (Delta = -0.56%)
==============================================================================
```

### 3.6. Stacked 2-Layer LSTM + Highway [No Weather, Input_Dim=1]
```text
Stacked 2-Layer LSTM (Univariate Ablation) Deep Learning Model Summary - Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
==============================================================================
Run ID: 20261124_193611_5cd58d | Status: SUCCESS | Regime: Feature Ablation Study (Univariate Load Only — No Weather)
Data: Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
Date range: 2011-01-01 00:00:00 to 2015-01-01 00:00:00 (Freq: 1h)
Observations: 35,065 (Train: 24,545, Val: 5,260, Eval/Test: 5,260)
Architecture: Input(24,1) -> LSTM(64) -> Dropout(0.1) -> LSTM(32) -> LayerNorm -> GELU -> Linear(1) + Highway(24->1)
Trainable Parameters: 29,818 | Optimizer: AdamW + CosineAnnealingLR + EarlyStopping
Transformation: StandardScaler (Fit on Train Only) + Inverse Transform (kW)
Model: Stacked 2-Layer LSTM + Highway [No Weather, Input_Dim=1]
==============================================================================
          PyTorch Deep Learning Convergence & Parameter Anatomy               
==============================================================================
Hardware Device:                  cuda   Train / Pred:    63.0651s / 0.6048s
==============================================================================
Ljung-Box (L=24) (Q):    1407.58   Jarque-Bera (JB):    176311.86
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     1.1876   Skew:                   0.9160
Prob(H) (two-sided):       0.0003   Kurtosis:              31.3039
Durbin-Watson (DW):        1.7710   MedAE (kW):        3734.5786
Residual Mean (kW):    -74.1782   Residual Std:        7077.8314
------------------------------------------------------------------------------
OUT-OF-SAMPLE EVALUATION METRICS (Original Scale kW, n_eval=5,260):
  MSE  = 50,101,199.7840 | RMSE = 7,078.2201 kW | MAE = 4,930.6207 kW
  MAPE =        2.2122% | R²   =     0.9935     | Max Error = 124,077.7816 kW
  [Weather Ablation]      With Weather RMSE = 6,570.41 kW vs. Without Weather RMSE = 7,078.22 kW (Delta = +7.17%)
==============================================================================
```

### 3.7. XGBoost [Walk-Forward Backtesting 168h Online Feedback]
```text
XGBoost Gradient Boosting Model Summary - Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
==============================================================================
Run ID: 20261124_193611_5cd58d | Status: SUCCESS | Regime: Walk-Forward Backtesting (168 Hours Rolling 1-Step Feedback)
Data: Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
Date range: 2014-05-26 21:00:00 to 2014-06-02 20:00:00 (Freq: 1h)
Observations: 35,065 (Train: 24,545, Val: 5,260, Eval/Test: 168)
Features: 106 Tabular Features | Trees Configured: 450 | Best Iteration: 358
Max Depth: 6 | Learning Rate (eta): 0.045 | Subsample: 0.85 | Colsample: 0.85
Regularization: L1 (alpha)=0.1, L2 (lambda)=1.0 | Tree Method: hist (cuda)
Transformation: StandardScaler (Fit on Train Only) + Inverse Transform to Original Scale (kW)
Model: XGBoost [Walk-Forward Backtesting 168h Online Feedback]
==============================================================================
            XGBoost Boosting Convergence & Feature Importance Fit             
==============================================================================
Train Scaled RMSE (Tree 1 -> End): 0.955722 -> 0.043671
Val Scaled RMSE   (Tree 1 -> Best): 0.821423 -> 0.065047
Converged:                        True   Train / Pred:    5.2138s / 0.0995s
------------------------------------------------------------------------------
BOOSTING ITERATION CONVERGENCE TRAJECTORY (Extracted from evals_result_):
  Tree #  Train RMSE (Scaled)  Val RMSE (Scaled)      Gen Gap           Status
------------------------------------------------------------------------------
       1             0.955722           0.821423    -0.134300     Initial Tree
      10             0.638098           0.550294    -0.087804         Boosting
      25             0.331789           0.289683    -0.042106         Boosting
      50             0.130762           0.120341    -0.010421         Boosting
     100             0.064996           0.069367    +0.004371         Boosting
     150             0.056804           0.066319    +0.009515         Boosting
     200             0.052635           0.065599    +0.012964         Boosting
     250             0.049817           0.065331    +0.015514         Boosting
     300             0.047330           0.065137    +0.017807         Boosting
     350             0.045153           0.065067    +0.019914         Boosting
     359             0.044828           0.065047    +0.020219 Best Val RMSE (Selected)
     389             0.043671           0.065053    +0.021382         Boosting
------------------------------------------------------------------------------
FEATURE IMPORTANCE AGGREGATED BY FUNCTIONAL GROUP (106 Engineered Features):
Feature Group                                  Count  Share (%)    Top Feature
------------------------------------------------------------------------------
1. Autoregressive Load Lags (load_lag_1h..24h)    24     96.13%   load_lag_24h
4. Exogenous Humidity Lags (humidity_lag_1h..2    24      1.33% humidity_lag_12h
5. Exogenous Temperature Lags (temperature_lag    24      0.99% temperature_lag_13h
3. Cyclical Fourier & Calendar (sin/cos, weeke     5      0.81%       sin_hour
2. Load Rolling & Trend Stats (roll, trend_24h     5      0.46% load_trend_24h
6. Exogenous Wind Speed Lags (wind_speed_lag_1    24      0.28% wind_speed_lag_21h
------------------------------------------------------------------------------
TOP 15 DOMINANT FEATURES BY GAIN IMPORTANCE:
Rank Feature Name              Gain Importance   Share (%)   Cumulative (%)
------------------------------------------------------------------------------
   1 load_lag_24h                     0.711656      71.17%           71.17%
   2 load_lag_23h                     0.146604      14.66%           85.83%
   3 load_lag_1h                      0.083991       8.40%           94.23%
   4 humidity_lag_12h                 0.003471       0.35%           94.57%
   5 sin_hour                         0.002987       0.30%           94.87%
   6 load_lag_22h                     0.002617       0.26%           95.13%
   7 load_lag_20h                     0.002380       0.24%           95.37%
   8 cos_hour                         0.002286       0.23%           95.60%
   9 load_lag_21h                     0.002129       0.21%           95.81%
  10 load_trend_24h                   0.001835       0.18%           96.00%
  11 humidity_lag_11h                 0.001674       0.17%           96.16%
  12 temperature_lag_13h              0.001343       0.13%           96.30%
  13 load_lag_2h                      0.001257       0.13%           96.42%
  14 is_weekend                       0.001213       0.12%           96.54%
  15 humidity_lag_13h                 0.001181       0.12%           96.66%
==============================================================================
Ljung-Box (L=24) (Q):      64.42   Jarque-Bera (JB):        11.04
Prob(Q):                   0.0000   Prob(JB):               0.0040
Heteroskedasticity(H):     2.3835   Skew:                   0.4589
Prob(H) (two-sided):       0.0014   Kurtosis:               3.8571
Durbin-Watson (DW):        2.5319   MedAE (kW):            2659.04
Residual Mean (kW):      897.7841   Residual Std (kW):   5280.4863
------------------------------------------------------------------------------
OUT-OF-SAMPLE EVALUATION METRICS (Original Scale kW, n_eval=168):
  MSE  = 28,689,551.8835 | RMSE = 5,356.2629 kW | MAE = 3,973.0530 kW
  MAPE =        1.8012% | R²   =     0.9940    | Max Error =  18,006.66 kW
==============================================================================
```

### 3.8. VAR [Walk-Forward Backtesting 168h Online Feedback]
```text
VAR / AR Multivariate Model Summary - Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
==============================================================================
Run ID: 20261124_193611_5cd58d | Status: SUCCESS | Regime: Walk-Forward Backtesting (168 Hours Rolling 1-Step Feedback)
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
Converged:                        True   Train / Pred:    3.0565s / 0.2574s
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
Run ID: 20261124_193611_5cd58d | Status: SUCCESS | Regime: Walk-Forward Backtesting (168 Hours Rolling 1-Step Feedback)
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
Hardware Device:                  cuda   Train / Pred:    97.0233s / 0.8064s
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
Run ID: 20261124_193611_5cd58d | Status: SUCCESS | Fold: 1 / 5 (Leakage-Free In-Fold Scaler)
Data: Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
Window Period: Train [2011-01-01 00:00:00..2011-07-27 01:00:00] -> Val [2011-07-27 02:00:00..2012-02-19 00:00:00]
Observations: Train^(k) = 4,970 hours | Val^(k) = 4,967 hours | Device: cpu
Model Family: VAR | Params / Features: 97 | Train Time: 0.7028s
==============================================================================
Ljung-Box (L=24) (Q):     648.28   Jarque-Bera (JB):    101226.76
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     1.4332   Skew:                   0.4853
Prob(H) (two-sided):       0.0000   Kurtosis:              25.0947
Durbin-Watson (DW):        1.7117   MedAE (kW):            2671.57
Residual Mean (kW):      -99.5896   Residual Std (kW):   5786.7279
------------------------------------------------------------------------------
OUT-OF-SAMPLE FOLD 1 VALIDATION METRICS (Original Scale kW, n_val=4,967):
  MSE  = 33,496,137.4639 | RMSE = 5,787.5848 kW | MAE = 3,841.1737 kW
  MAPE =        1.7099% | R²   =     0.9938    | Max Error =  89,463.11 kW
  [5-Fold CV Mean ± Std]  RMSE = 7,004.77 ± 1,204.93 kW | R² = 0.9918 ± 0.0022
==============================================================================
```

### 3.11. XGBoost [5-Fold TimeSeriesSplit — Fold 1/5]
```text
5-Fold Expanding Window CV Model Summary - XGBoost [5-Fold TimeSeriesSplit — Fold 1/5]
==============================================================================
Run ID: 20261124_193611_5cd58d | Status: SUCCESS | Fold: 1 / 5 (Leakage-Free In-Fold Scaler)
Data: Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
Window Period: Train [2011-01-01 00:00:00..2011-07-27 01:00:00] -> Val [2011-07-27 02:00:00..2012-02-19 00:00:00]
Observations: Train^(k) = 4,970 hours | Val^(k) = 4,967 hours | Device: cuda
Model Family: XGBoost | Params / Features: 106 | Train Time: 6.1018s
==============================================================================
Ljung-Box (L=24) (Q):   15585.70   Jarque-Bera (JB):    187417.69
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):    12.5163   Skew:                  -3.7927
Prob(H) (two-sided):       0.0000   Kurtosis:              32.1212
Durbin-Watson (DW):        0.5293   MedAE (kW):            4155.35
Residual Mean (kW):    -2755.7360   Residual Std (kW):  11389.4223
------------------------------------------------------------------------------
OUT-OF-SAMPLE FOLD 1 VALIDATION METRICS (Original Scale kW, n_val=4,967):
  MSE  = 137,313,020.3513 | RMSE = 11,718.0638 kW | MAE = 6,691.0340 kW
  MAPE =        3.5454% | R²   =     0.9747    | Max Error = 126,647.05 kW
  [5-Fold CV Mean ± Std]  RMSE = 10,235.15 ± 5,097.38 kW | R² = 0.9826 ± 0.0111
==============================================================================
```

### 3.12. LSTM [5-Fold TimeSeriesSplit — Fold 1/5]
```text
5-Fold Expanding Window CV Model Summary - LSTM [5-Fold TimeSeriesSplit — Fold 1/5]
==============================================================================
Run ID: 20261124_193611_5cd58d | Status: SUCCESS | Fold: 1 / 5 (Leakage-Free In-Fold Scaler)
Data: Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
Window Period: Train [2011-01-01 00:00:00..2011-07-27 01:00:00] -> Val [2011-07-27 02:00:00..2012-02-19 00:00:00]
Observations: Train^(k) = 4,970 hours | Val^(k) = 4,967 hours | Device: cuda
Model Family: LSTM | Params / Features: 30,589 | Train Time: 9.2671s
==============================================================================
Ljung-Box (L=24) (Q):    5160.10   Jarque-Bera (JB):     13070.06
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     2.2902   Skew:                  -0.7043
Prob(H) (two-sided):       0.0000   Kurtosis:              10.8210
Durbin-Watson (DW):        1.2214   MedAE (kW):            4420.27
Residual Mean (kW):     -541.2321   Residual Std (kW):   8519.9778
------------------------------------------------------------------------------
OUT-OF-SAMPLE FOLD 1 VALIDATION METRICS (Original Scale kW, n_val=4,967):
  MSE  = 72,882,953.1316 | RMSE = 8,537.1513 kW | MAE = 5,980.9555 kW
  MAPE =        2.8019% | R²   =     0.9866    | Max Error =  67,703.68 kW
  [5-Fold CV Mean ± Std]  RMSE = 8,764.86 ± 2,738.29 kW | R² = 0.9877 ± 0.0037
==============================================================================
```

### 3.13. VAR [5-Fold TimeSeriesSplit — Fold 2/5]
```text
5-Fold Expanding Window CV Model Summary - VAR [5-Fold TimeSeriesSplit — Fold 2/5]
==============================================================================
Run ID: 20261124_193611_5cd58d | Status: SUCCESS | Fold: 2 / 5 (Leakage-Free In-Fold Scaler)
Data: Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
Window Period: Train [2011-01-01 00:00:00..2012-02-19 00:00:00] -> Val [2012-02-19 01:00:00..2012-09-12 23:00:00]
Observations: Train^(k) = 9,937 hours | Val^(k) = 4,967 hours | Device: cpu
Model Family: VAR | Params / Features: 97 | Train Time: 1.2285s
==============================================================================
Ljung-Box (L=24) (Q):     795.16   Jarque-Bera (JB):    224823.35
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     1.4943   Skew:                   0.1894
Prob(H) (two-sided):       0.0000   Kurtosis:              35.9572
Durbin-Watson (DW):        2.3749   MedAE (kW):            3713.68
Residual Mean (kW):       31.6079   Residual Std (kW):   8793.7818
------------------------------------------------------------------------------
OUT-OF-SAMPLE FOLD 2 VALIDATION METRICS (Original Scale kW, n_val=4,967):
  MSE  = 77,331,598.1471 | RMSE = 8,793.8386 kW | MAE = 5,499.7215 kW
  MAPE =        2.8433% | R²   =     0.9927    | Max Error = 134,139.59 kW
  [5-Fold CV Mean ± Std]  RMSE = 7,004.77 ± 1,204.93 kW | R² = 0.9918 ± 0.0022
==============================================================================
```

### 3.14. XGBoost [5-Fold TimeSeriesSplit — Fold 2/5]
```text
5-Fold Expanding Window CV Model Summary - XGBoost [5-Fold TimeSeriesSplit — Fold 2/5]
==============================================================================
Run ID: 20261124_193611_5cd58d | Status: SUCCESS | Fold: 2 / 5 (Leakage-Free In-Fold Scaler)
Data: Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
Window Period: Train [2011-01-01 00:00:00..2012-02-19 00:00:00] -> Val [2012-02-19 01:00:00..2012-09-12 23:00:00]
Observations: Train^(k) = 9,937 hours | Val^(k) = 4,967 hours | Device: cuda
Model Family: XGBoost | Params / Features: 106 | Train Time: 5.0186s
==============================================================================
Ljung-Box (L=24) (Q):   13999.57   Jarque-Bera (JB):      9625.28
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):    20.8340   Skew:                   1.5831
Prob(H) (two-sided):       0.0000   Kurtosis:               9.0402
Durbin-Watson (DW):        0.7210   MedAE (kW):            5447.09
Residual Mean (kW):     2807.9241   Residual Std (kW):  18064.7796
------------------------------------------------------------------------------
OUT-OF-SAMPLE FOLD 2 VALIDATION METRICS (Original Scale kW, n_val=4,967):
  MSE  = 334,220,698.1683 | RMSE = 18,281.7039 kW | MAE = 10,829.0768 kW
  MAPE =        4.5498% | R²   =     0.9684    | Max Error = 131,001.66 kW
  [5-Fold CV Mean ± Std]  RMSE = 10,235.15 ± 5,097.38 kW | R² = 0.9826 ± 0.0111
==============================================================================
```

### 3.15. LSTM [5-Fold TimeSeriesSplit — Fold 2/5]
```text
5-Fold Expanding Window CV Model Summary - LSTM [5-Fold TimeSeriesSplit — Fold 2/5]
==============================================================================
Run ID: 20261124_193611_5cd58d | Status: SUCCESS | Fold: 2 / 5 (Leakage-Free In-Fold Scaler)
Data: Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
Window Period: Train [2011-01-01 00:00:00..2012-02-19 00:00:00] -> Val [2012-02-19 01:00:00..2012-09-12 23:00:00]
Observations: Train^(k) = 9,937 hours | Val^(k) = 4,967 hours | Device: cuda
Model Family: LSTM | Params / Features: 30,589 | Train Time: 10.0490s
==============================================================================
Ljung-Box (L=24) (Q):    4222.09   Jarque-Bera (JB):     12240.29
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     3.8104   Skew:                   0.7302
Prob(H) (two-sided):       0.0000   Kurtosis:              10.5505
Durbin-Watson (DW):        1.5095   MedAE (kW):            6054.22
Residual Mean (kW):      317.8793   Residual Std (kW):  13332.6704
------------------------------------------------------------------------------
OUT-OF-SAMPLE FOLD 2 VALIDATION METRICS (Original Scale kW, n_val=4,967):
  MSE  = 177,861,148.2063 | RMSE = 13,336.4594 kW | MAE = 8,866.2547 kW
  MAPE =        4.3257% | R²   =     0.9832    | Max Error = 130,657.77 kW
  [5-Fold CV Mean ± Std]  RMSE = 8,764.86 ± 2,738.29 kW | R² = 0.9877 ± 0.0037
==============================================================================
```

### 3.16. VAR [5-Fold TimeSeriesSplit — Fold 3/5]
```text
5-Fold Expanding Window CV Model Summary - VAR [5-Fold TimeSeriesSplit — Fold 3/5]
==============================================================================
Run ID: 20261124_193611_5cd58d | Status: SUCCESS | Fold: 3 / 5 (Leakage-Free In-Fold Scaler)
Data: Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
Window Period: Train [2011-01-01 00:00:00..2012-09-12 23:00:00] -> Val [2012-09-13 00:00:00..2013-04-07 22:00:00]
Observations: Train^(k) = 14,904 hours | Val^(k) = 4,967 hours | Device: cpu
Model Family: VAR | Params / Features: 97 | Train Time: 1.7816s
==============================================================================
Ljung-Box (L=24) (Q):     918.61   Jarque-Bera (JB):   2087237.63
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     0.5462   Skew:                   0.0301
Prob(H) (two-sided):       0.0000   Kurtosis:             103.4256
Durbin-Watson (DW):        2.3936   MedAE (kW):            2445.71
Residual Mean (kW):     -105.9128   Residual Std (kW):   7213.1845
------------------------------------------------------------------------------
OUT-OF-SAMPLE FOLD 3 VALIDATION METRICS (Original Scale kW, n_val=4,967):
  MSE  = 52,041,247.5759 | RMSE = 7,213.9620 kW | MAE = 3,871.2630 kW
  MAPE =        2.4843% | R²   =     0.9891    | Max Error = 128,523.04 kW
  [5-Fold CV Mean ± Std]  RMSE = 7,004.77 ± 1,204.93 kW | R² = 0.9918 ± 0.0022
==============================================================================
```

### 3.17. XGBoost [5-Fold TimeSeriesSplit — Fold 3/5]
```text
5-Fold Expanding Window CV Model Summary - XGBoost [5-Fold TimeSeriesSplit — Fold 3/5]
==============================================================================
Run ID: 20261124_193611_5cd58d | Status: SUCCESS | Fold: 3 / 5 (Leakage-Free In-Fold Scaler)
Data: Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
Window Period: Train [2011-01-01 00:00:00..2012-09-12 23:00:00] -> Val [2012-09-13 00:00:00..2013-04-07 22:00:00]
Observations: Train^(k) = 14,904 hours | Val^(k) = 4,967 hours | Device: cuda
Model Family: XGBoost | Params / Features: 106 | Train Time: 2.7909s
==============================================================================
Ljung-Box (L=24) (Q):    4569.64   Jarque-Bera (JB):    152544.57
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     1.0952   Skew:                  -1.1675
Prob(H) (two-sided):       0.0643   Kurtosis:              30.0486
Durbin-Watson (DW):        1.2754   MedAE (kW):            4037.28
Residual Mean (kW):     1320.7753   Residual Std (kW):   8794.0221
------------------------------------------------------------------------------
OUT-OF-SAMPLE FOLD 3 VALIDATION METRICS (Original Scale kW, n_val=4,967):
  MSE  = 79,079,271.2600 | RMSE = 8,892.6527 kW | MAE = 5,870.9476 kW
  MAPE =        3.4442% | R²   =     0.9834    | Max Error = 129,752.69 kW
  [5-Fold CV Mean ± Std]  RMSE = 10,235.15 ± 5,097.38 kW | R² = 0.9826 ± 0.0111
==============================================================================
```

### 3.18. LSTM [5-Fold TimeSeriesSplit — Fold 3/5]
```text
5-Fold Expanding Window CV Model Summary - LSTM [5-Fold TimeSeriesSplit — Fold 3/5]
==============================================================================
Run ID: 20261124_193611_5cd58d | Status: SUCCESS | Fold: 3 / 5 (Leakage-Free In-Fold Scaler)
Data: Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
Window Period: Train [2011-01-01 00:00:00..2012-09-12 23:00:00] -> Val [2012-09-13 00:00:00..2013-04-07 22:00:00]
Observations: Train^(k) = 14,904 hours | Val^(k) = 4,967 hours | Device: cuda
Model Family: LSTM | Params / Features: 30,589 | Train Time: 26.6055s
==============================================================================
Ljung-Box (L=24) (Q):    1893.45   Jarque-Bera (JB):    245030.42
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     0.4990   Skew:                  -0.2947
Prob(H) (two-sided):       0.0000   Kurtosis:              37.4037
Durbin-Watson (DW):        1.6549   MedAE (kW):            4054.09
Residual Mean (kW):      -85.7942   Residual Std (kW):   8183.4698
------------------------------------------------------------------------------
OUT-OF-SAMPLE FOLD 3 VALIDATION METRICS (Original Scale kW, n_val=4,967):
  MSE  = 66,976,539.1108 | RMSE = 8,183.9195 kW | MAE = 5,488.8784 kW
  MAPE =        3.3260% | R²   =     0.9860    | Max Error = 127,752.50 kW
  [5-Fold CV Mean ± Std]  RMSE = 8,764.86 ± 2,738.29 kW | R² = 0.9877 ± 0.0037
==============================================================================
```

### 3.19. VAR [5-Fold TimeSeriesSplit — Fold 4/5]
```text
5-Fold Expanding Window CV Model Summary - VAR [5-Fold TimeSeriesSplit — Fold 4/5]
==============================================================================
Run ID: 20261124_193611_5cd58d | Status: SUCCESS | Fold: 4 / 5 (Leakage-Free In-Fold Scaler)
Data: Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
Window Period: Train [2011-01-01 00:00:00..2013-04-07 22:00:00] -> Val [2013-04-07 23:00:00..2013-10-31 21:00:00]
Observations: Train^(k) = 19,871 hours | Val^(k) = 4,967 hours | Device: cpu
Model Family: VAR | Params / Features: 97 | Train Time: 2.4096s
==============================================================================
Ljung-Box (L=24) (Q):     632.65   Jarque-Bera (JB):    238866.95
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     1.6673   Skew:                   0.6200
Prob(H) (two-sided):       0.0000   Kurtosis:              36.9506
Durbin-Watson (DW):        2.3878   MedAE (kW):            3136.56
Residual Mean (kW):       12.8795   Residual Std (kW):   7237.2587
------------------------------------------------------------------------------
OUT-OF-SAMPLE FOLD 4 VALIDATION METRICS (Original Scale kW, n_val=4,967):
  MSE  = 52,378,079.9660 | RMSE = 7,237.2702 kW | MAE = 4,600.4121 kW
  MAPE =        1.9333% | R²   =     0.9938    | Max Error = 111,877.32 kW
  [5-Fold CV Mean ± Std]  RMSE = 7,004.77 ± 1,204.93 kW | R² = 0.9918 ± 0.0022
==============================================================================
```

### 3.20. XGBoost [5-Fold TimeSeriesSplit — Fold 4/5]
```text
5-Fold Expanding Window CV Model Summary - XGBoost [5-Fold TimeSeriesSplit — Fold 4/5]
==============================================================================
Run ID: 20261124_193611_5cd58d | Status: SUCCESS | Fold: 4 / 5 (Leakage-Free In-Fold Scaler)
Data: Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
Window Period: Train [2011-01-01 00:00:00..2013-04-07 22:00:00] -> Val [2013-04-07 23:00:00..2013-10-31 21:00:00]
Observations: Train^(k) = 19,871 hours | Val^(k) = 4,967 hours | Device: cuda
Model Family: XGBoost | Params / Features: 106 | Train Time: 4.0364s
==============================================================================
Ljung-Box (L=24) (Q):     715.67   Jarque-Bera (JB):     85997.73
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     1.1127   Skew:                   1.4293
Prob(H) (two-sided):       0.0299   Kurtosis:              23.1832
Durbin-Watson (DW):        1.7396   MedAE (kW):            3288.00
Residual Mean (kW):      122.2813   Residual Std (kW):   7054.7331
------------------------------------------------------------------------------
OUT-OF-SAMPLE FOLD 4 VALIDATION METRICS (Original Scale kW, n_val=4,967):
  MSE  = 49,784,212.2314 | RMSE = 7,055.7928 kW | MAE = 4,715.8494 kW
  MAPE =        1.9782% | R²   =     0.9941    | Max Error = 115,494.00 kW
  [5-Fold CV Mean ± Std]  RMSE = 10,235.15 ± 5,097.38 kW | R² = 0.9826 ± 0.0111
==============================================================================
```

### 3.21. LSTM [5-Fold TimeSeriesSplit — Fold 4/5]
```text
5-Fold Expanding Window CV Model Summary - LSTM [5-Fold TimeSeriesSplit — Fold 4/5]
==============================================================================
Run ID: 20261124_193611_5cd58d | Status: SUCCESS | Fold: 4 / 5 (Leakage-Free In-Fold Scaler)
Data: Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
Window Period: Train [2011-01-01 00:00:00..2013-04-07 22:00:00] -> Val [2013-04-07 23:00:00..2013-10-31 21:00:00]
Observations: Train^(k) = 19,871 hours | Val^(k) = 4,967 hours | Device: cuda
Model Family: LSTM | Params / Features: 30,589 | Train Time: 34.7935s
==============================================================================
Ljung-Box (L=24) (Q):    1094.08   Jarque-Bera (JB):     56595.42
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     1.3448   Skew:                   0.9478
Prob(H) (two-sided):       0.0000   Kurtosis:              19.4277
Durbin-Watson (DW):        1.9134   MedAE (kW):            4381.60
Residual Mean (kW):     -377.4048   Residual Std (kW):   7775.3843
------------------------------------------------------------------------------
OUT-OF-SAMPLE FOLD 4 VALIDATION METRICS (Original Scale kW, n_val=4,967):
  MSE  = 60,599,035.9841 | RMSE = 7,784.5383 kW | MAE = 5,585.0868 kW
  MAPE =        2.4227% | R²   =     0.9929    | Max Error = 117,036.81 kW
  [5-Fold CV Mean ± Std]  RMSE = 8,764.86 ± 2,738.29 kW | R² = 0.9877 ± 0.0037
==============================================================================
```

### 3.22. VAR [5-Fold TimeSeriesSplit — Fold 5/5]
```text
5-Fold Expanding Window CV Model Summary - VAR [5-Fold TimeSeriesSplit — Fold 5/5]
==============================================================================
Run ID: 20261124_193611_5cd58d | Status: SUCCESS | Fold: 5 / 5 (Leakage-Free In-Fold Scaler)
Data: Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
Window Period: Train [2011-01-01 00:00:00..2013-10-31 21:00:00] -> Val [2013-10-31 22:00:00..2014-05-26 20:00:00]
Observations: Train^(k) = 24,838 hours | Val^(k) = 4,967 hours | Device: cpu
Model Family: VAR | Params / Features: 97 | Train Time: 2.9439s
==============================================================================
Ljung-Box (L=24) (Q):     925.27   Jarque-Bera (JB):   2660436.85
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     1.3319   Skew:                  -0.3868
Prob(H) (two-sided):       0.0000   Kurtosis:             116.3770
Durbin-Watson (DW):        2.1684   MedAE (kW):            2407.98
Residual Mean (kW):      -12.2415   Residual Std (kW):   5991.1754
------------------------------------------------------------------------------
OUT-OF-SAMPLE FOLD 5 VALIDATION METRICS (Original Scale kW, n_val=4,967):
  MSE  = 35,894,332.1253 | RMSE = 5,991.1879 kW | MAE = 3,511.0698 kW
  MAPE =        7.5620% | R²   =     0.9898    | Max Error = 130,954.28 kW
  [5-Fold CV Mean ± Std]  RMSE = 7,004.77 ± 1,204.93 kW | R² = 0.9918 ± 0.0022
==============================================================================
```

### 3.23. XGBoost [5-Fold TimeSeriesSplit — Fold 5/5]
```text
5-Fold Expanding Window CV Model Summary - XGBoost [5-Fold TimeSeriesSplit — Fold 5/5]
==============================================================================
Run ID: 20261124_193611_5cd58d | Status: SUCCESS | Fold: 5 / 5 (Leakage-Free In-Fold Scaler)
Data: Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
Window Period: Train [2011-01-01 00:00:00..2013-10-31 21:00:00] -> Val [2013-10-31 22:00:00..2014-05-26 20:00:00]
Observations: Train^(k) = 24,838 hours | Val^(k) = 4,967 hours | Device: cuda
Model Family: XGBoost | Params / Features: 106 | Train Time: 4.0384s
==============================================================================
Ljung-Box (L=24) (Q):    1147.97   Jarque-Bera (JB):   1199272.62
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     1.2621   Skew:                  -3.3366
Prob(H) (two-sided):       0.0000   Kurtosis:              78.8302
Durbin-Watson (DW):        1.6790   MedAE (kW):            2545.08
Residual Mean (kW):      122.5057   Residual Std (kW):   5226.0943
------------------------------------------------------------------------------
OUT-OF-SAMPLE FOLD 5 VALIDATION METRICS (Original Scale kW, n_val=4,967):
  MSE  = 27,327,069.5854 | RMSE = 5,227.5300 kW | MAE = 3,497.2875 kW
  MAPE =        7.3346% | R²   =     0.9922    | Max Error = 126,895.46 kW
  [5-Fold CV Mean ± Std]  RMSE = 10,235.15 ± 5,097.38 kW | R² = 0.9826 ± 0.0111
==============================================================================
```

### 3.24. LSTM [5-Fold TimeSeriesSplit — Fold 5/5]
```text
5-Fold Expanding Window CV Model Summary - LSTM [5-Fold TimeSeriesSplit — Fold 5/5]
==============================================================================
Run ID: 20261124_193611_5cd58d | Status: SUCCESS | Fold: 5 / 5 (Leakage-Free In-Fold Scaler)
Data: Electricity Load & Weather Multivariate Time Series (UCI LD2011-2014, 01/2011 - 01/2015)
Window Period: Train [2011-01-01 00:00:00..2013-10-31 21:00:00] -> Val [2013-10-31 22:00:00..2014-05-26 20:00:00]
Observations: Train^(k) = 24,838 hours | Val^(k) = 4,967 hours | Device: cuda
Model Family: LSTM | Params / Features: 30,589 | Train Time: 42.9271s
==============================================================================
Ljung-Box (L=24) (Q):     950.64   Jarque-Bera (JB):   1118923.11
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     1.4570   Skew:                  -1.0503
Prob(H) (two-sided):       0.0000   Kurtosis:              76.4990
Durbin-Watson (DW):        1.8342   MedAE (kW):            2896.01
Residual Mean (kW):     -359.2133   Residual Std (kW):   5971.4322
------------------------------------------------------------------------------
OUT-OF-SAMPLE FOLD 5 VALIDATION METRICS (Original Scale kW, n_val=4,967):
  MSE  = 35,787,037.1541 | RMSE = 5,982.2268 kW | MAE = 3,977.4965 kW
  MAPE =        7.8120% | R²   =     0.9898    | Max Error = 130,637.10 kW
  [5-Fold CV Mean ± Std]  RMSE = 8,764.86 ± 2,738.29 kW | R² = 0.9877 ± 0.0037
==============================================================================
```
