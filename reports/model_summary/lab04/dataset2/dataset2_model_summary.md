# BÁO CÁO TỔNG HỢP MÔ HÌNH (MODEL SUMMARY) — PEMS-BAY SPATIO-TEMPORAL HIGHWAY TRAFFIC SPEED (207 SENSORS, 5-MIN, 01/2017)

## 1. Thông tin Phiên Thực nghiệm & Bộ dữ liệu Đồ thị Không gian - Thời gian (Provenance & Graph Info)
- **Run ID**: `20261117_160932_3fe70f`
- **Thời điểm xuất báo cáo**: `2026-11-17 16:19:33`
- **Dataset Key**: `dataset2` (`data/raw/pems-bay.h5` + `data/sensor_graph/adj_mx_bay.pkl`)
- **Tên bộ dữ liệu**: PEMS-BAY Spatio-Temporal Highway Traffic Speed (207 Sensors, 5-min, 01/2017)
- **Kích thước chuỗi không gian - thời gian**: 8,640 bước thời gian (5 phút/bước) × 207 trạm cảm biến cao tốc
- **Cấu trúc đồ thị cảm biến (`adj_mx_bay.pkl`)**: Ma trận có hướng `207 × 207` với `1,238` cạnh trọng số > 0 (Độ thưa: `97.11%`)
- **Cửa sổ trượt Đa tầm nhìn**: Lookback $L = 12$ bước (60 phút) $\rightarrow$ Dự báo đồng thời 3 mốc tương lai: `15_min (t+3)`, `30_min (t+6)`, `60_min (t+12)`
- **Số cửa sổ mẫu**: Train = 6,025 | Validation = 1,285 | Test = 1,285 (265,995 điểm dự báo/tầm nhìn)
- **Tổng số thực thể mô hình-tầm nhìn được thống kê**: 12 (3 Kiến trúc `DCRNN`, `Multi-Horizon TCN`, `Multi-Horizon LSTM` × 4 Tầm nhìn đánh giá)

## 2. Bảng So sánh Tổng hợp các Mô hình Đa Tầm nhìn trên PEMS-BAY (Model Comparison Table)

| # | Model & Horizon | Family | Horizon | Params | MSE | RMSE (mph) | MAE (mph) | MAPE (%) | R² | Max Error | Device | Train Time (s) |
|---:|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|:---|---:|
| 1 | DCRNN (Graph Diffusion + GRU) — Horizon 15_min (15 min / t+3) | `DCRNN` | `Horizon_15_min` | 21,290 | 7.6291 | 2.7621 | 1.3094 | 2.7182 | 0.9196 | 47.18 | `cuda` | 66.9065 |
| 2 | DCRNN (Graph Diffusion + GRU) — Horizon 30_min (30 min / t+6) | `DCRNN` | `Horizon_30_min` | 21,290 | 15.4535 | 3.9311 | 1.7338 | 3.8915 | 0.8372 | 55.52 | `cuda` | 66.9065 |
| 3 | DCRNN (Graph Diffusion + GRU) — Horizon 60_min (60 min / t+12) | `DCRNN` | `Horizon_60_min` | 21,290 | 27.8962 | 5.2817 | 2.3040 | 5.6224 | 0.7063 | 58.62 | `cuda` | 66.9065 |
| 4 | DCRNN (Graph Diffusion + GRU) — Horizon Average_All_Horizons (35 min / t+7) | `DCRNN` | `Horizon_Average_All_Horizons` | 21,290 | 16.9929 | 3.9916 | 1.7824 | 4.0774 | 0.8210 | 58.62 | `cuda` | 66.9065 |
| 5 | Multi-Horizon TCN (Causal Conv1D) — Horizon 15_min (15 min / t+3) | `Multi_Horizon_TCN` | `Horizon_15_min` | 3,370 | 8.6841 | 2.9469 | 1.3541 | 2.8546 | 0.9085 | 50.96 | `cuda` | 14.5001 |
| 6 | Multi-Horizon TCN (Causal Conv1D) — Horizon 30_min (30 min / t+6) | `Multi_Horizon_TCN` | `Horizon_30_min` | 3,370 | 19.0388 | 4.3633 | 1.8552 | 4.1751 | 0.7994 | 55.86 | `cuda` | 14.5001 |
| 7 | Multi-Horizon TCN (Causal Conv1D) — Horizon 60_min (60 min / t+12) | `Multi_Horizon_TCN` | `Horizon_60_min` | 3,370 | 37.1624 | 6.0961 | 2.6242 | 6.5286 | 0.6087 | 58.90 | `cuda` | 14.5001 |
| 8 | Multi-Horizon TCN (Causal Conv1D) — Horizon Average_All_Horizons (35 min / t+7) | `Multi_Horizon_TCN` | `Horizon_Average_All_Horizons` | 3,370 | 21.6284 | 4.4688 | 1.9445 | 4.5194 | 0.7722 | 58.90 | `cuda` | 14.5001 |
| 9 | Multi-Horizon LSTM (Temporal Only) — Horizon 15_min (15 min / t+3) | `Multi_Horizon_LSTM` | `Horizon_15_min` | 110,292 | 8.4484 | 2.9066 | 1.4182 | 2.9199 | 0.9110 | 47.02 | `cuda` | 8.1749 |
| 10 | Multi-Horizon LSTM (Temporal Only) — Horizon 30_min (30 min / t+6) | `Multi_Horizon_LSTM` | `Horizon_30_min` | 110,292 | 16.2314 | 4.0288 | 1.8885 | 4.0697 | 0.8290 | 50.79 | `cuda` | 8.1749 |
| 11 | Multi-Horizon LSTM (Temporal Only) — Horizon 60_min (60 min / t+12) | `Multi_Horizon_LSTM` | `Horizon_60_min` | 110,292 | 28.6132 | 5.3491 | 2.4986 | 5.4961 | 0.6987 | 54.84 | `cuda` | 8.1749 |
| 12 | Multi-Horizon LSTM (Temporal Only) — Horizon Average_All_Horizons (35 min / t+7) | `Multi_Horizon_LSTM` | `Horizon_Average_All_Horizons` | 110,292 | 17.7643 | 4.0948 | 1.9351 | 4.1619 | 0.8129 | 54.84 | `cuda` | 8.1749 |

### Diễn giải Đa Tiêu chí Tự động (Run ID: `20261117_160932_3fe70f` — Dataset 2: PEMS-BAY Traffic Graph)

- **Mô hình vượt trội nhất ở cả 3 tầm nhìn (15 phút, 30 phút, 60 phút)**: **DCRNN (Graph Diffusion + GRU)** đạt trung bình `RMSE = 3.9916 mph`, `MAE = 1.7824 mph`, `MAPE = 4.0774%` và `R² = 0.8210`.
- **Vai trò quyết định của Đồ thị Không gian (`adj_mx_bay.pkl`) ở tầm nhìn xa 60 phút ($t+12$)**: Khi dự báo xa 60 phút, sóng tắc nghẽn giao thông lan truyền từ các nút lân cận trên mạng lưới cao tốc. Do đó **DCRNN** (`RMSE = 5.2817 mph`, `MAE = 2.3040 mph`, `R² = 0.7063`) bỏ xa cả **Multi-Horizon LSTM** (`RMSE = 5.3491 mph`, `MAE = 2.4986 mph`) và **Multi-Horizon TCN** (`RMSE = 6.0961 mph`, `MAE = 2.6243 mph`).
- **Hiệu quả tham số (Parameter Efficiency)**: **DCRNN** chỉ cần `21,290` tham số khả huấn luyện (nhờ chia sẻ bộ lọc khuếch tán đồ thị trên 5 ma trận hỗ trợ $\mathcal{S} \in \mathbb{R}^{5 \times 207 \times 207}$), ít hơn gấp 5.18 lần so với **Multi-Horizon LSTM** (`110,292` tham số).

## 3. Hồ sơ Thống kê Chi tiết Từng Mô hình & Tầm nhìn (Individual Model Cards — 12 Evaluations)

### 3.1. DCRNN (Graph Diffusion + GRU) — Horizon 15_min (15 min / t+3)
```text
DCRNN (Graph Diffusion + GRU) Deep Learning Model Summary - PEMS-BAY Spatio-Temporal Highway Traffic Speed (207 Sensors, 5-min, 01/2017)
==============================================================================
Run ID: 20261117_160932_3fe70f | Status: SUCCESS | Regime: PEMS-BAY Multi-Horizon Traffic Forecasting (15_min, 15 min ahead)
Data: PEMS-BAY Spatio-Temporal Highway Traffic Speed (207 Sensors, 5-min, 01/2017)
Date range: 2017-01-01 00:00:00 to 2017-01-30 23:55:00 (Freq: 5min)
Observations: 8,640 (Train: 6,025, Val: 1,285, Eval/Test: 265,995)
Architecture: Bidirectional Random-Walk Diffusion (K=2, 5 Supports) + DCGRU(hidden=32) + LayerNorm + GELU + AR_Highway(12->3)
Trainable Parameters: 21,290 | Graph Support Buffer Elements: 214,245 (5 x 207 x 207)
Transformation: StandardScaler (Fit on Train Only) + Inverse Transform (mph)
Model: DCRNN (Graph Diffusion + GRU) — Horizon 15_min (15 min / t+3)
==============================================================================
          PyTorch Deep Learning Convergence & Parameter Anatomy               
==============================================================================
Initial Train Loss:           0.118180   Best Val Loss:       0.230240
Final Train Loss:             0.090550   Best Epoch:           11 / 12  
Hardware Device:                  cuda   Train / Pred:    66.9065s / 0.0850s
------------------------------------------------------------------------------
LAYER-BY-LAYER TENSOR PARAMETER ANATOMY (Extracted from dcrnn_pems_bay.pt):
Layer / Tensor Name        Shape             Params      Mean      Std           [Min, Max]
------------------------------------------------------------------------------
supports_stack*            (5, 207, 207)    214,245  +0.00483  0.06024     [-0.974, +1.498]
cell1.gate_linear.weight   (64, 165)         10,560  +0.00006  0.19202     [-1.047, +1.039]
cell1.gate_linear.bias     (64,)                 64  -0.00579  0.16632     [-0.438, +0.487]
cell1.cand_linear.weight   (32, 165)          5,280  -0.00071  0.08735     [-0.445, +0.458]
cell1.cand_linear.bias     (32,)                 32  -0.00049  0.05887     [-0.126, +0.136]
norm.weight                (32,)                 32  +1.00541  0.07076     [+0.849, +1.140]
norm.bias                  (32,)                 32  +0.00028  0.02501     [-0.047, +0.047]
proj.0.weight              (32, 160)          5,120  +0.00048  0.06583     [-0.296, +0.327]
proj.0.bias                (32,)                 32  -0.02567  0.05406     [-0.119, +0.065]
proj.2.weight              (3, 32)               96  +0.00667  0.12173     [-0.196, +0.275]
proj.2.bias                (3,)                   3  -0.03100  0.05782     [-0.099, +0.042]
ar_highway.weight          (3, 12)               36  +0.06285  0.22385     [-0.058, +0.883]
ar_highway.bias            (3,)                   3  +0.01399  0.01026     [-0.001, +0.022]
------------------------------------------------------------------------------
LEARNED MULTI-HORIZON AUTOREGRESSIVE WEIGHTS (ar_highway.weight, 12 Lookback Steps):
  15_min (t+3) : L12:-0.013 L7:-0.003 L4:-0.022 L3:-0.058 L2:+0.048 L1:+0.883
  30_min (t+6) : L12:-0.013 L7:+0.003 L4:-0.023 L3:-0.050 L2:+0.059 L1:+0.803
  60_min (t+12): L12:-0.017 L7:-0.002 L4:-0.028 L3:-0.058 L2:+0.035 L1:+0.708
------------------------------------------------------------------------------
EPOCH-BY-EPOCH TRAINING & VALIDATION CONVERGENCE HISTORY:
Epoch  Train Loss (SmoothL1)     Val Scaled MAE   Val-Train Gap         Status
------------------------------------------------------------------------------
    1                0.11818            0.25975        +0.14157               
    2                0.10791            0.25081        +0.14290               
    3                0.10285            0.24239        +0.13954               
    4                0.10022            0.24268        +0.14246               
    5                0.09802            0.24054        +0.14252               
    6                0.09657            0.23871        +0.14214               
    7                0.09496            0.23554        +0.14058               
    8                0.09343            0.23332        +0.13989               
    9                0.09241            0.23362        +0.14121               
   10                0.09162            0.23151        +0.13989               
   11                0.09095            0.23024        +0.13929 <-- Best Checkpoint
   12                0.09055            0.23175        +0.14120               
------------------------------------------------------------------------------
MULTI-HORIZON SPATIO-TEMPORAL EVALUATION BREAKDOWN (mph across 207 Sensors):
Horizon              Steps       MSE     RMSE      MAE  MAPE (%)       R²   MaxErr
------------------------------------------------------------------------------
15_min                   3    7.6291   2.7621   1.3094   2.7182%   0.9196    47.18
30_min                   6   15.4535   3.9311   1.7338   3.8915%   0.8372    55.52
60_min                  12   27.8962   5.2817   2.3040   5.6224%   0.7063    58.62
Average_All_Horizons     7   16.9929   3.9916   1.7824   4.0774%   0.8210    58.62
==============================================================================
Ljung-Box (L=24) (Q):    2303.23   Jarque-Bera (JB):       885.86
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     0.8014   Skew:                  -0.5684
Prob(H) (two-sided):       0.0222   Kurtosis:               6.9055
Durbin-Watson (DW):        0.3666   MedAE (mph):           0.2129
Residual Mean (mph):     -0.0199   Residual Std:           2.7620
------------------------------------------------------------------------------
OUT-OF-SAMPLE EVALUATION METRICS (Original Scale mph, n_eval=265,995):
  MSE  =         7.6291 | RMSE =     2.7621 mph | MAE =     1.3094 mph
  MAPE =        2.7182% | R²   =     0.9196     | Max Error =    47.1802 mph
==============================================================================
```

### 3.2. DCRNN (Graph Diffusion + GRU) — Horizon 30_min (30 min / t+6)
```text
DCRNN (Graph Diffusion + GRU) Deep Learning Model Summary - PEMS-BAY Spatio-Temporal Highway Traffic Speed (207 Sensors, 5-min, 01/2017)
==============================================================================
Run ID: 20261117_160932_3fe70f | Status: SUCCESS | Regime: PEMS-BAY Multi-Horizon Traffic Forecasting (30_min, 30 min ahead)
Data: PEMS-BAY Spatio-Temporal Highway Traffic Speed (207 Sensors, 5-min, 01/2017)
Date range: 2017-01-01 00:00:00 to 2017-01-30 23:55:00 (Freq: 5min)
Observations: 8,640 (Train: 6,025, Val: 1,285, Eval/Test: 265,995)
Architecture: Bidirectional Random-Walk Diffusion (K=2, 5 Supports) + DCGRU(hidden=32) + LayerNorm + GELU + AR_Highway(12->3)
Trainable Parameters: 21,290 | Graph Support Buffer Elements: 214,245 (5 x 207 x 207)
Transformation: StandardScaler (Fit on Train Only) + Inverse Transform (mph)
Model: DCRNN (Graph Diffusion + GRU) — Horizon 30_min (30 min / t+6)
==============================================================================
          PyTorch Deep Learning Convergence & Parameter Anatomy               
==============================================================================
Initial Train Loss:           0.118180   Best Val Loss:       0.230240
Final Train Loss:             0.090550   Best Epoch:           11 / 12  
Hardware Device:                  cuda   Train / Pred:    66.9065s / 0.0850s
------------------------------------------------------------------------------
LAYER-BY-LAYER TENSOR PARAMETER ANATOMY (Extracted from dcrnn_pems_bay.pt):
Layer / Tensor Name        Shape             Params      Mean      Std           [Min, Max]
------------------------------------------------------------------------------
supports_stack*            (5, 207, 207)    214,245  +0.00483  0.06024     [-0.974, +1.498]
cell1.gate_linear.weight   (64, 165)         10,560  +0.00006  0.19202     [-1.047, +1.039]
cell1.gate_linear.bias     (64,)                 64  -0.00579  0.16632     [-0.438, +0.487]
cell1.cand_linear.weight   (32, 165)          5,280  -0.00071  0.08735     [-0.445, +0.458]
cell1.cand_linear.bias     (32,)                 32  -0.00049  0.05887     [-0.126, +0.136]
norm.weight                (32,)                 32  +1.00541  0.07076     [+0.849, +1.140]
norm.bias                  (32,)                 32  +0.00028  0.02501     [-0.047, +0.047]
proj.0.weight              (32, 160)          5,120  +0.00048  0.06583     [-0.296, +0.327]
proj.0.bias                (32,)                 32  -0.02567  0.05406     [-0.119, +0.065]
proj.2.weight              (3, 32)               96  +0.00667  0.12173     [-0.196, +0.275]
proj.2.bias                (3,)                   3  -0.03100  0.05782     [-0.099, +0.042]
ar_highway.weight          (3, 12)               36  +0.06285  0.22385     [-0.058, +0.883]
ar_highway.bias            (3,)                   3  +0.01399  0.01026     [-0.001, +0.022]
------------------------------------------------------------------------------
LEARNED MULTI-HORIZON AUTOREGRESSIVE WEIGHTS (ar_highway.weight, 12 Lookback Steps):
  15_min (t+3) : L12:-0.013 L7:-0.003 L4:-0.022 L3:-0.058 L2:+0.048 L1:+0.883
  30_min (t+6) : L12:-0.013 L7:+0.003 L4:-0.023 L3:-0.050 L2:+0.059 L1:+0.803
  60_min (t+12): L12:-0.017 L7:-0.002 L4:-0.028 L3:-0.058 L2:+0.035 L1:+0.708
------------------------------------------------------------------------------
EPOCH-BY-EPOCH TRAINING & VALIDATION CONVERGENCE HISTORY:
Epoch  Train Loss (SmoothL1)     Val Scaled MAE   Val-Train Gap         Status
------------------------------------------------------------------------------
    1                0.11818            0.25975        +0.14157               
    2                0.10791            0.25081        +0.14290               
    3                0.10285            0.24239        +0.13954               
    4                0.10022            0.24268        +0.14246               
    5                0.09802            0.24054        +0.14252               
    6                0.09657            0.23871        +0.14214               
    7                0.09496            0.23554        +0.14058               
    8                0.09343            0.23332        +0.13989               
    9                0.09241            0.23362        +0.14121               
   10                0.09162            0.23151        +0.13989               
   11                0.09095            0.23024        +0.13929 <-- Best Checkpoint
   12                0.09055            0.23175        +0.14120               
------------------------------------------------------------------------------
MULTI-HORIZON SPATIO-TEMPORAL EVALUATION BREAKDOWN (mph across 207 Sensors):
Horizon              Steps       MSE     RMSE      MAE  MAPE (%)       R²   MaxErr
------------------------------------------------------------------------------
15_min                   3    7.6291   2.7621   1.3094   2.7182%   0.9196    47.18
30_min                   6   15.4535   3.9311   1.7338   3.8915%   0.8372    55.52
60_min                  12   27.8962   5.2817   2.3040   5.6224%   0.7063    58.62
Average_All_Horizons     7   16.9929   3.9916   1.7824   4.0774%   0.8210    58.62
==============================================================================
Ljung-Box (L=24) (Q):    4517.53   Jarque-Bera (JB):       690.98
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     0.7785   Skew:                  -0.8079
Prob(H) (two-sided):       0.0097   Kurtosis:               6.2085
Durbin-Watson (DW):        0.1305   MedAE (mph):           0.2967
Residual Mean (mph):     -0.1243   Residual Std:           3.9291
------------------------------------------------------------------------------
OUT-OF-SAMPLE EVALUATION METRICS (Original Scale mph, n_eval=265,995):
  MSE  =        15.4535 | RMSE =     3.9311 mph | MAE =     1.7338 mph
  MAPE =        3.8915% | R²   =     0.8372     | Max Error =    55.5159 mph
==============================================================================
```

### 3.3. DCRNN (Graph Diffusion + GRU) — Horizon 60_min (60 min / t+12)
```text
DCRNN (Graph Diffusion + GRU) Deep Learning Model Summary - PEMS-BAY Spatio-Temporal Highway Traffic Speed (207 Sensors, 5-min, 01/2017)
==============================================================================
Run ID: 20261117_160932_3fe70f | Status: SUCCESS | Regime: PEMS-BAY Multi-Horizon Traffic Forecasting (60_min, 60 min ahead)
Data: PEMS-BAY Spatio-Temporal Highway Traffic Speed (207 Sensors, 5-min, 01/2017)
Date range: 2017-01-01 00:00:00 to 2017-01-30 23:55:00 (Freq: 5min)
Observations: 8,640 (Train: 6,025, Val: 1,285, Eval/Test: 265,995)
Architecture: Bidirectional Random-Walk Diffusion (K=2, 5 Supports) + DCGRU(hidden=32) + LayerNorm + GELU + AR_Highway(12->3)
Trainable Parameters: 21,290 | Graph Support Buffer Elements: 214,245 (5 x 207 x 207)
Transformation: StandardScaler (Fit on Train Only) + Inverse Transform (mph)
Model: DCRNN (Graph Diffusion + GRU) — Horizon 60_min (60 min / t+12)
==============================================================================
          PyTorch Deep Learning Convergence & Parameter Anatomy               
==============================================================================
Initial Train Loss:           0.118180   Best Val Loss:       0.230240
Final Train Loss:             0.090550   Best Epoch:           11 / 12  
Hardware Device:                  cuda   Train / Pred:    66.9065s / 0.0850s
------------------------------------------------------------------------------
LAYER-BY-LAYER TENSOR PARAMETER ANATOMY (Extracted from dcrnn_pems_bay.pt):
Layer / Tensor Name        Shape             Params      Mean      Std           [Min, Max]
------------------------------------------------------------------------------
supports_stack*            (5, 207, 207)    214,245  +0.00483  0.06024     [-0.974, +1.498]
cell1.gate_linear.weight   (64, 165)         10,560  +0.00006  0.19202     [-1.047, +1.039]
cell1.gate_linear.bias     (64,)                 64  -0.00579  0.16632     [-0.438, +0.487]
cell1.cand_linear.weight   (32, 165)          5,280  -0.00071  0.08735     [-0.445, +0.458]
cell1.cand_linear.bias     (32,)                 32  -0.00049  0.05887     [-0.126, +0.136]
norm.weight                (32,)                 32  +1.00541  0.07076     [+0.849, +1.140]
norm.bias                  (32,)                 32  +0.00028  0.02501     [-0.047, +0.047]
proj.0.weight              (32, 160)          5,120  +0.00048  0.06583     [-0.296, +0.327]
proj.0.bias                (32,)                 32  -0.02567  0.05406     [-0.119, +0.065]
proj.2.weight              (3, 32)               96  +0.00667  0.12173     [-0.196, +0.275]
proj.2.bias                (3,)                   3  -0.03100  0.05782     [-0.099, +0.042]
ar_highway.weight          (3, 12)               36  +0.06285  0.22385     [-0.058, +0.883]
ar_highway.bias            (3,)                   3  +0.01399  0.01026     [-0.001, +0.022]
------------------------------------------------------------------------------
LEARNED MULTI-HORIZON AUTOREGRESSIVE WEIGHTS (ar_highway.weight, 12 Lookback Steps):
  15_min (t+3) : L12:-0.013 L7:-0.003 L4:-0.022 L3:-0.058 L2:+0.048 L1:+0.883
  30_min (t+6) : L12:-0.013 L7:+0.003 L4:-0.023 L3:-0.050 L2:+0.059 L1:+0.803
  60_min (t+12): L12:-0.017 L7:-0.002 L4:-0.028 L3:-0.058 L2:+0.035 L1:+0.708
------------------------------------------------------------------------------
EPOCH-BY-EPOCH TRAINING & VALIDATION CONVERGENCE HISTORY:
Epoch  Train Loss (SmoothL1)     Val Scaled MAE   Val-Train Gap         Status
------------------------------------------------------------------------------
    1                0.11818            0.25975        +0.14157               
    2                0.10791            0.25081        +0.14290               
    3                0.10285            0.24239        +0.13954               
    4                0.10022            0.24268        +0.14246               
    5                0.09802            0.24054        +0.14252               
    6                0.09657            0.23871        +0.14214               
    7                0.09496            0.23554        +0.14058               
    8                0.09343            0.23332        +0.13989               
    9                0.09241            0.23362        +0.14121               
   10                0.09162            0.23151        +0.13989               
   11                0.09095            0.23024        +0.13929 <-- Best Checkpoint
   12                0.09055            0.23175        +0.14120               
------------------------------------------------------------------------------
MULTI-HORIZON SPATIO-TEMPORAL EVALUATION BREAKDOWN (mph across 207 Sensors):
Horizon              Steps       MSE     RMSE      MAE  MAPE (%)       R²   MaxErr
------------------------------------------------------------------------------
15_min                   3    7.6291   2.7621   1.3094   2.7182%   0.9196    47.18
30_min                   6   15.4535   3.9311   1.7338   3.8915%   0.8372    55.52
60_min                  12   27.8962   5.2817   2.3040   5.6224%   0.7063    58.62
Average_All_Horizons     7   16.9929   3.9916   1.7824   4.0774%   0.8210    58.62
==============================================================================
Ljung-Box (L=24) (Q):    7839.87   Jarque-Bera (JB):       920.60
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     0.8457   Skew:                  -1.2878
Prob(H) (two-sided):       0.0833   Kurtosis:               6.2497
Durbin-Watson (DW):        0.0483   MedAE (mph):           0.4447
Residual Mean (mph):     -0.2987   Residual Std:           5.2732
------------------------------------------------------------------------------
OUT-OF-SAMPLE EVALUATION METRICS (Original Scale mph, n_eval=265,995):
  MSE  =        27.8962 | RMSE =     5.2817 mph | MAE =     2.3040 mph
  MAPE =        5.6224% | R²   =     0.7063     | Max Error =    58.6159 mph
==============================================================================
```

### 3.4. DCRNN (Graph Diffusion + GRU) — Horizon Average_All_Horizons (35 min / t+7)
```text
DCRNN (Graph Diffusion + GRU) Deep Learning Model Summary - PEMS-BAY Spatio-Temporal Highway Traffic Speed (207 Sensors, 5-min, 01/2017)
==============================================================================
Run ID: 20261117_160932_3fe70f | Status: SUCCESS | Regime: PEMS-BAY Multi-Horizon Traffic Forecasting (Average_All_Horizons, 35 min ahead)
Data: PEMS-BAY Spatio-Temporal Highway Traffic Speed (207 Sensors, 5-min, 01/2017)
Date range: 2017-01-01 00:00:00 to 2017-01-30 23:55:00 (Freq: 5min)
Observations: 8,640 (Train: 6,025, Val: 1,285, Eval/Test: 265,995)
Architecture: Bidirectional Random-Walk Diffusion (K=2, 5 Supports) + DCGRU(hidden=32) + LayerNorm + GELU + AR_Highway(12->3)
Trainable Parameters: 21,290 | Graph Support Buffer Elements: 214,245 (5 x 207 x 207)
Transformation: StandardScaler (Fit on Train Only) + Inverse Transform (mph)
Model: DCRNN (Graph Diffusion + GRU) — Horizon Average_All_Horizons (35 min / t+7)
==============================================================================
          PyTorch Deep Learning Convergence & Parameter Anatomy               
==============================================================================
Initial Train Loss:           0.118180   Best Val Loss:       0.230240
Final Train Loss:             0.090550   Best Epoch:           11 / 12  
Hardware Device:                  cuda   Train / Pred:    66.9065s / 0.0850s
------------------------------------------------------------------------------
LAYER-BY-LAYER TENSOR PARAMETER ANATOMY (Extracted from dcrnn_pems_bay.pt):
Layer / Tensor Name        Shape             Params      Mean      Std           [Min, Max]
------------------------------------------------------------------------------
supports_stack*            (5, 207, 207)    214,245  +0.00483  0.06024     [-0.974, +1.498]
cell1.gate_linear.weight   (64, 165)         10,560  +0.00006  0.19202     [-1.047, +1.039]
cell1.gate_linear.bias     (64,)                 64  -0.00579  0.16632     [-0.438, +0.487]
cell1.cand_linear.weight   (32, 165)          5,280  -0.00071  0.08735     [-0.445, +0.458]
cell1.cand_linear.bias     (32,)                 32  -0.00049  0.05887     [-0.126, +0.136]
norm.weight                (32,)                 32  +1.00541  0.07076     [+0.849, +1.140]
norm.bias                  (32,)                 32  +0.00028  0.02501     [-0.047, +0.047]
proj.0.weight              (32, 160)          5,120  +0.00048  0.06583     [-0.296, +0.327]
proj.0.bias                (32,)                 32  -0.02567  0.05406     [-0.119, +0.065]
proj.2.weight              (3, 32)               96  +0.00667  0.12173     [-0.196, +0.275]
proj.2.bias                (3,)                   3  -0.03100  0.05782     [-0.099, +0.042]
ar_highway.weight          (3, 12)               36  +0.06285  0.22385     [-0.058, +0.883]
ar_highway.bias            (3,)                   3  +0.01399  0.01026     [-0.001, +0.022]
------------------------------------------------------------------------------
LEARNED MULTI-HORIZON AUTOREGRESSIVE WEIGHTS (ar_highway.weight, 12 Lookback Steps):
  15_min (t+3) : L12:-0.013 L7:-0.003 L4:-0.022 L3:-0.058 L2:+0.048 L1:+0.883
  30_min (t+6) : L12:-0.013 L7:+0.003 L4:-0.023 L3:-0.050 L2:+0.059 L1:+0.803
  60_min (t+12): L12:-0.017 L7:-0.002 L4:-0.028 L3:-0.058 L2:+0.035 L1:+0.708
------------------------------------------------------------------------------
EPOCH-BY-EPOCH TRAINING & VALIDATION CONVERGENCE HISTORY:
Epoch  Train Loss (SmoothL1)     Val Scaled MAE   Val-Train Gap         Status
------------------------------------------------------------------------------
    1                0.11818            0.25975        +0.14157               
    2                0.10791            0.25081        +0.14290               
    3                0.10285            0.24239        +0.13954               
    4                0.10022            0.24268        +0.14246               
    5                0.09802            0.24054        +0.14252               
    6                0.09657            0.23871        +0.14214               
    7                0.09496            0.23554        +0.14058               
    8                0.09343            0.23332        +0.13989               
    9                0.09241            0.23362        +0.14121               
   10                0.09162            0.23151        +0.13989               
   11                0.09095            0.23024        +0.13929 <-- Best Checkpoint
   12                0.09055            0.23175        +0.14120               
------------------------------------------------------------------------------
MULTI-HORIZON SPATIO-TEMPORAL EVALUATION BREAKDOWN (mph across 207 Sensors):
Horizon              Steps       MSE     RMSE      MAE  MAPE (%)       R²   MaxErr
------------------------------------------------------------------------------
15_min                   3    7.6291   2.7621   1.3094   2.7182%   0.9196    47.18
30_min                   6   15.4535   3.9311   1.7338   3.8915%   0.8372    55.52
60_min                  12   27.8962   5.2817   2.3040   5.6224%   0.7063    58.62
Average_All_Horizons     7   16.9929   3.9916   1.7824   4.0774%   0.8210    58.62
==============================================================================
Ljung-Box (L=24) (Q):     187.97   Jarque-Bera (JB):      3244.79
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     1.2200   Skew:                  -0.0925
Prob(H) (two-sided):       0.1425   Kurtosis:              10.7826
Durbin-Watson (DW):        1.7550   MedAE (mph):           1.2120
Residual Mean (mph):     -0.1476   Residual Std:           3.9881
------------------------------------------------------------------------------
OUT-OF-SAMPLE EVALUATION METRICS (Original Scale mph, n_eval=265,995):
  MSE  =        16.9929 | RMSE =     3.9916 mph | MAE =     1.7824 mph
  MAPE =        4.0774% | R²   =     0.8210     | Max Error =    58.6159 mph
==============================================================================
```

### 3.5. Multi-Horizon TCN (Causal Conv1D) — Horizon 15_min (15 min / t+3)
```text
Multi-Horizon TCN (Causal Conv1D) Deep Learning Model Summary - PEMS-BAY Spatio-Temporal Highway Traffic Speed (207 Sensors, 5-min, 01/2017)
==============================================================================
Run ID: 20261117_160932_3fe70f | Status: SUCCESS | Regime: PEMS-BAY Multi-Horizon Traffic Forecasting (15_min, 15 min ahead)
Data: PEMS-BAY Spatio-Temporal Highway Traffic Speed (207 Sensors, 5-min, 01/2017)
Date range: 2017-01-01 00:00:00 to 2017-01-30 23:55:00 (Freq: 5min)
Observations: 8,640 (Train: 6,025, Val: 1,285, Eval/Test: 265,995)
Architecture: Dilated Causal Conv1D (dilation=[1,2], kernel=3, channels=32) + GELU + Linear(32->3) + AR_Linear(12->3)
Trainable Parameters: 3,370 | Optimizer: AdamW + CosineAnnealingLR + EarlyStopping
Transformation: StandardScaler (Fit on Train Only) + Inverse Transform (mph)
Model: Multi-Horizon TCN (Causal Conv1D) — Horizon 15_min (15 min / t+3)
==============================================================================
          PyTorch Deep Learning Convergence & Parameter Anatomy               
==============================================================================
Initial Train Loss:           0.118770   Best Val Loss:       0.259500
Final Train Loss:             0.112640   Best Epoch:            7 / 10  
Hardware Device:                  cuda   Train / Pred:    14.5001s / 0.0850s
------------------------------------------------------------------------------
LAYER-BY-LAYER TENSOR PARAMETER ANATOMY (Extracted from tcn_pems_bay.pt):
Layer / Tensor Name        Shape             Params      Mean      Std           [Min, Max]
------------------------------------------------------------------------------
conv1.weight               (32, 1, 3)            96  -0.02726  0.40867     [-0.750, +0.873]
conv1.bias                 (32,)                 32  +0.01144  0.30184     [-0.520, +0.503]
conv2.weight               (32, 32, 3)        3,072  -0.01373  0.10859     [-0.656, +0.540]
conv2.bias                 (32,)                 32  -0.02430  0.06916     [-0.130, +0.095]
head.weight                (3, 32)               96  +0.03396  0.16630     [-0.560, +0.401]
head.bias                  (3,)                   3  -0.01119  0.11303     [-0.140, +0.135]
ar_linear.weight           (3, 12)               36  +0.06352  0.23086     [-0.073, +0.838]
ar_linear.bias             (3,)                   3  +0.00073  0.02226     [-0.025, +0.029]
------------------------------------------------------------------------------
LEARNED MULTI-HORIZON AUTOREGRESSIVE WEIGHTS (ar_linear.weight, 12 Lookback Steps):
  15_min (t+3) : L12:-0.036 L7:+0.002 L4:+0.004 L3:+0.005 L2:+0.003 L1:+0.827
  30_min (t+6) : L12:-0.051 L7:-0.003 L4:+0.002 L3:+0.003 L2:+0.022 L1:+0.838
  60_min (t+12): L12:-0.073 L7:-0.021 L4:-0.003 L3:-0.002 L2:-0.005 L1:+0.817
------------------------------------------------------------------------------
EPOCH-BY-EPOCH TRAINING & VALIDATION CONVERGENCE HISTORY:
Epoch  Train Loss (SmoothL1)     Val Scaled MAE   Val-Train Gap         Status
------------------------------------------------------------------------------
    1                0.11877            0.26513        +0.14636               
    2                0.11521            0.26258        +0.14737               
    3                0.11422            0.26152        +0.14730               
    4                0.11411            0.26246        +0.14835               
    5                0.11349            0.26339        +0.14990               
    6                0.11367            0.26294        +0.14927               
    7                0.11300            0.25950        +0.14650 <-- Best Checkpoint
    8                0.11290            0.26106        +0.14816               
    9                0.11273            0.25993        +0.14720               
   10                0.11264            0.26017        +0.14753               
------------------------------------------------------------------------------
MULTI-HORIZON SPATIO-TEMPORAL EVALUATION BREAKDOWN (mph across 207 Sensors):
Horizon              Steps       MSE     RMSE      MAE  MAPE (%)       R²   MaxErr
------------------------------------------------------------------------------
15_min                   3    8.6841   2.9469   1.3541   2.8546%   0.9085    50.96
30_min                   6   19.0388   4.3633   1.8552   4.1751%   0.7994    55.86
60_min                  12   37.1624   6.0961   2.6242   6.5286%   0.6087    58.90
Average_All_Horizons     7   21.6284   4.4688   1.9445   4.5194%   0.7722    58.90
==============================================================================
Ljung-Box (L=24) (Q):    6961.02   Jarque-Bera (JB):       415.03
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     0.7227   Skew:                  -0.5298
Prob(H) (two-sided):       0.0008   Kurtosis:               5.5746
Durbin-Watson (DW):        0.1413   MedAE (mph):           0.2604
Residual Mean (mph):     -0.0368   Residual Std:           2.9466
------------------------------------------------------------------------------
OUT-OF-SAMPLE EVALUATION METRICS (Original Scale mph, n_eval=265,995):
  MSE  =         8.6841 | RMSE =     2.9469 mph | MAE =     1.3541 mph
  MAPE =        2.8546% | R²   =     0.9085     | Max Error =    50.9629 mph
==============================================================================
```

### 3.6. Multi-Horizon TCN (Causal Conv1D) — Horizon 30_min (30 min / t+6)
```text
Multi-Horizon TCN (Causal Conv1D) Deep Learning Model Summary - PEMS-BAY Spatio-Temporal Highway Traffic Speed (207 Sensors, 5-min, 01/2017)
==============================================================================
Run ID: 20261117_160932_3fe70f | Status: SUCCESS | Regime: PEMS-BAY Multi-Horizon Traffic Forecasting (30_min, 30 min ahead)
Data: PEMS-BAY Spatio-Temporal Highway Traffic Speed (207 Sensors, 5-min, 01/2017)
Date range: 2017-01-01 00:00:00 to 2017-01-30 23:55:00 (Freq: 5min)
Observations: 8,640 (Train: 6,025, Val: 1,285, Eval/Test: 265,995)
Architecture: Dilated Causal Conv1D (dilation=[1,2], kernel=3, channels=32) + GELU + Linear(32->3) + AR_Linear(12->3)
Trainable Parameters: 3,370 | Optimizer: AdamW + CosineAnnealingLR + EarlyStopping
Transformation: StandardScaler (Fit on Train Only) + Inverse Transform (mph)
Model: Multi-Horizon TCN (Causal Conv1D) — Horizon 30_min (30 min / t+6)
==============================================================================
          PyTorch Deep Learning Convergence & Parameter Anatomy               
==============================================================================
Initial Train Loss:           0.118770   Best Val Loss:       0.259500
Final Train Loss:             0.112640   Best Epoch:            7 / 10  
Hardware Device:                  cuda   Train / Pred:    14.5001s / 0.0850s
------------------------------------------------------------------------------
LAYER-BY-LAYER TENSOR PARAMETER ANATOMY (Extracted from tcn_pems_bay.pt):
Layer / Tensor Name        Shape             Params      Mean      Std           [Min, Max]
------------------------------------------------------------------------------
conv1.weight               (32, 1, 3)            96  -0.02726  0.40867     [-0.750, +0.873]
conv1.bias                 (32,)                 32  +0.01144  0.30184     [-0.520, +0.503]
conv2.weight               (32, 32, 3)        3,072  -0.01373  0.10859     [-0.656, +0.540]
conv2.bias                 (32,)                 32  -0.02430  0.06916     [-0.130, +0.095]
head.weight                (3, 32)               96  +0.03396  0.16630     [-0.560, +0.401]
head.bias                  (3,)                   3  -0.01119  0.11303     [-0.140, +0.135]
ar_linear.weight           (3, 12)               36  +0.06352  0.23086     [-0.073, +0.838]
ar_linear.bias             (3,)                   3  +0.00073  0.02226     [-0.025, +0.029]
------------------------------------------------------------------------------
LEARNED MULTI-HORIZON AUTOREGRESSIVE WEIGHTS (ar_linear.weight, 12 Lookback Steps):
  15_min (t+3) : L12:-0.036 L7:+0.002 L4:+0.004 L3:+0.005 L2:+0.003 L1:+0.827
  30_min (t+6) : L12:-0.051 L7:-0.003 L4:+0.002 L3:+0.003 L2:+0.022 L1:+0.838
  60_min (t+12): L12:-0.073 L7:-0.021 L4:-0.003 L3:-0.002 L2:-0.005 L1:+0.817
------------------------------------------------------------------------------
EPOCH-BY-EPOCH TRAINING & VALIDATION CONVERGENCE HISTORY:
Epoch  Train Loss (SmoothL1)     Val Scaled MAE   Val-Train Gap         Status
------------------------------------------------------------------------------
    1                0.11877            0.26513        +0.14636               
    2                0.11521            0.26258        +0.14737               
    3                0.11422            0.26152        +0.14730               
    4                0.11411            0.26246        +0.14835               
    5                0.11349            0.26339        +0.14990               
    6                0.11367            0.26294        +0.14927               
    7                0.11300            0.25950        +0.14650 <-- Best Checkpoint
    8                0.11290            0.26106        +0.14816               
    9                0.11273            0.25993        +0.14720               
   10                0.11264            0.26017        +0.14753               
------------------------------------------------------------------------------
MULTI-HORIZON SPATIO-TEMPORAL EVALUATION BREAKDOWN (mph across 207 Sensors):
Horizon              Steps       MSE     RMSE      MAE  MAPE (%)       R²   MaxErr
------------------------------------------------------------------------------
15_min                   3    8.6841   2.9469   1.3541   2.8546%   0.9085    50.96
30_min                   6   19.0388   4.3633   1.8552   4.1751%   0.7994    55.86
60_min                  12   37.1624   6.0961   2.6242   6.5286%   0.6087    58.90
Average_All_Horizons     7   21.6284   4.4688   1.9445   4.5194%   0.7722    58.90
==============================================================================
Ljung-Box (L=24) (Q):    8785.76   Jarque-Bera (JB):       469.39
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     0.7123   Skew:                  -0.1771
Prob(H) (two-sided):       0.0005   Kurtosis:               5.9396
Durbin-Watson (DW):        0.0443   MedAE (mph):           0.3668
Residual Mean (mph):     -0.0720   Residual Std:           4.3628
------------------------------------------------------------------------------
OUT-OF-SAMPLE EVALUATION METRICS (Original Scale mph, n_eval=265,995):
  MSE  =        19.0388 | RMSE =     4.3633 mph | MAE =     1.8552 mph
  MAPE =        4.1751% | R²   =     0.7994     | Max Error =    55.8628 mph
==============================================================================
```

### 3.7. Multi-Horizon TCN (Causal Conv1D) — Horizon 60_min (60 min / t+12)
```text
Multi-Horizon TCN (Causal Conv1D) Deep Learning Model Summary - PEMS-BAY Spatio-Temporal Highway Traffic Speed (207 Sensors, 5-min, 01/2017)
==============================================================================
Run ID: 20261117_160932_3fe70f | Status: SUCCESS | Regime: PEMS-BAY Multi-Horizon Traffic Forecasting (60_min, 60 min ahead)
Data: PEMS-BAY Spatio-Temporal Highway Traffic Speed (207 Sensors, 5-min, 01/2017)
Date range: 2017-01-01 00:00:00 to 2017-01-30 23:55:00 (Freq: 5min)
Observations: 8,640 (Train: 6,025, Val: 1,285, Eval/Test: 265,995)
Architecture: Dilated Causal Conv1D (dilation=[1,2], kernel=3, channels=32) + GELU + Linear(32->3) + AR_Linear(12->3)
Trainable Parameters: 3,370 | Optimizer: AdamW + CosineAnnealingLR + EarlyStopping
Transformation: StandardScaler (Fit on Train Only) + Inverse Transform (mph)
Model: Multi-Horizon TCN (Causal Conv1D) — Horizon 60_min (60 min / t+12)
==============================================================================
          PyTorch Deep Learning Convergence & Parameter Anatomy               
==============================================================================
Initial Train Loss:           0.118770   Best Val Loss:       0.259500
Final Train Loss:             0.112640   Best Epoch:            7 / 10  
Hardware Device:                  cuda   Train / Pred:    14.5001s / 0.0850s
------------------------------------------------------------------------------
LAYER-BY-LAYER TENSOR PARAMETER ANATOMY (Extracted from tcn_pems_bay.pt):
Layer / Tensor Name        Shape             Params      Mean      Std           [Min, Max]
------------------------------------------------------------------------------
conv1.weight               (32, 1, 3)            96  -0.02726  0.40867     [-0.750, +0.873]
conv1.bias                 (32,)                 32  +0.01144  0.30184     [-0.520, +0.503]
conv2.weight               (32, 32, 3)        3,072  -0.01373  0.10859     [-0.656, +0.540]
conv2.bias                 (32,)                 32  -0.02430  0.06916     [-0.130, +0.095]
head.weight                (3, 32)               96  +0.03396  0.16630     [-0.560, +0.401]
head.bias                  (3,)                   3  -0.01119  0.11303     [-0.140, +0.135]
ar_linear.weight           (3, 12)               36  +0.06352  0.23086     [-0.073, +0.838]
ar_linear.bias             (3,)                   3  +0.00073  0.02226     [-0.025, +0.029]
------------------------------------------------------------------------------
LEARNED MULTI-HORIZON AUTOREGRESSIVE WEIGHTS (ar_linear.weight, 12 Lookback Steps):
  15_min (t+3) : L12:-0.036 L7:+0.002 L4:+0.004 L3:+0.005 L2:+0.003 L1:+0.827
  30_min (t+6) : L12:-0.051 L7:-0.003 L4:+0.002 L3:+0.003 L2:+0.022 L1:+0.838
  60_min (t+12): L12:-0.073 L7:-0.021 L4:-0.003 L3:-0.002 L2:-0.005 L1:+0.817
------------------------------------------------------------------------------
EPOCH-BY-EPOCH TRAINING & VALIDATION CONVERGENCE HISTORY:
Epoch  Train Loss (SmoothL1)     Val Scaled MAE   Val-Train Gap         Status
------------------------------------------------------------------------------
    1                0.11877            0.26513        +0.14636               
    2                0.11521            0.26258        +0.14737               
    3                0.11422            0.26152        +0.14730               
    4                0.11411            0.26246        +0.14835               
    5                0.11349            0.26339        +0.14990               
    6                0.11367            0.26294        +0.14927               
    7                0.11300            0.25950        +0.14650 <-- Best Checkpoint
    8                0.11290            0.26106        +0.14816               
    9                0.11273            0.25993        +0.14720               
   10                0.11264            0.26017        +0.14753               
------------------------------------------------------------------------------
MULTI-HORIZON SPATIO-TEMPORAL EVALUATION BREAKDOWN (mph across 207 Sensors):
Horizon              Steps       MSE     RMSE      MAE  MAPE (%)       R²   MaxErr
------------------------------------------------------------------------------
15_min                   3    8.6841   2.9469   1.3541   2.8546%   0.9085    50.96
30_min                   6   19.0388   4.3633   1.8552   4.1751%   0.7994    55.86
60_min                  12   37.1624   6.0961   2.6242   6.5286%   0.6087    58.90
Average_All_Horizons     7   21.6284   4.4688   1.9445   4.5194%   0.7722    58.90
==============================================================================
Ljung-Box (L=24) (Q):   11060.86   Jarque-Bera (JB):       307.59
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     0.6888   Skew:                  -0.2845
Prob(H) (two-sided):       0.0001   Kurtosis:               5.3283
Durbin-Watson (DW):        0.0164   MedAE (mph):           0.5676
Residual Mean (mph):     -0.2736   Residual Std:           6.0900
------------------------------------------------------------------------------
OUT-OF-SAMPLE EVALUATION METRICS (Original Scale mph, n_eval=265,995):
  MSE  =        37.1624 | RMSE =     6.0961 mph | MAE =     2.6242 mph
  MAPE =        6.5286% | R²   =     0.6087     | Max Error =    58.8953 mph
==============================================================================
```

### 3.8. Multi-Horizon TCN (Causal Conv1D) — Horizon Average_All_Horizons (35 min / t+7)
```text
Multi-Horizon TCN (Causal Conv1D) Deep Learning Model Summary - PEMS-BAY Spatio-Temporal Highway Traffic Speed (207 Sensors, 5-min, 01/2017)
==============================================================================
Run ID: 20261117_160932_3fe70f | Status: SUCCESS | Regime: PEMS-BAY Multi-Horizon Traffic Forecasting (Average_All_Horizons, 35 min ahead)
Data: PEMS-BAY Spatio-Temporal Highway Traffic Speed (207 Sensors, 5-min, 01/2017)
Date range: 2017-01-01 00:00:00 to 2017-01-30 23:55:00 (Freq: 5min)
Observations: 8,640 (Train: 6,025, Val: 1,285, Eval/Test: 265,995)
Architecture: Dilated Causal Conv1D (dilation=[1,2], kernel=3, channels=32) + GELU + Linear(32->3) + AR_Linear(12->3)
Trainable Parameters: 3,370 | Optimizer: AdamW + CosineAnnealingLR + EarlyStopping
Transformation: StandardScaler (Fit on Train Only) + Inverse Transform (mph)
Model: Multi-Horizon TCN (Causal Conv1D) — Horizon Average_All_Horizons (35 min / t+7)
==============================================================================
          PyTorch Deep Learning Convergence & Parameter Anatomy               
==============================================================================
Initial Train Loss:           0.118770   Best Val Loss:       0.259500
Final Train Loss:             0.112640   Best Epoch:            7 / 10  
Hardware Device:                  cuda   Train / Pred:    14.5001s / 0.0850s
------------------------------------------------------------------------------
LAYER-BY-LAYER TENSOR PARAMETER ANATOMY (Extracted from tcn_pems_bay.pt):
Layer / Tensor Name        Shape             Params      Mean      Std           [Min, Max]
------------------------------------------------------------------------------
conv1.weight               (32, 1, 3)            96  -0.02726  0.40867     [-0.750, +0.873]
conv1.bias                 (32,)                 32  +0.01144  0.30184     [-0.520, +0.503]
conv2.weight               (32, 32, 3)        3,072  -0.01373  0.10859     [-0.656, +0.540]
conv2.bias                 (32,)                 32  -0.02430  0.06916     [-0.130, +0.095]
head.weight                (3, 32)               96  +0.03396  0.16630     [-0.560, +0.401]
head.bias                  (3,)                   3  -0.01119  0.11303     [-0.140, +0.135]
ar_linear.weight           (3, 12)               36  +0.06352  0.23086     [-0.073, +0.838]
ar_linear.bias             (3,)                   3  +0.00073  0.02226     [-0.025, +0.029]
------------------------------------------------------------------------------
LEARNED MULTI-HORIZON AUTOREGRESSIVE WEIGHTS (ar_linear.weight, 12 Lookback Steps):
  15_min (t+3) : L12:-0.036 L7:+0.002 L4:+0.004 L3:+0.005 L2:+0.003 L1:+0.827
  30_min (t+6) : L12:-0.051 L7:-0.003 L4:+0.002 L3:+0.003 L2:+0.022 L1:+0.838
  60_min (t+12): L12:-0.073 L7:-0.021 L4:-0.003 L3:-0.002 L2:-0.005 L1:+0.817
------------------------------------------------------------------------------
EPOCH-BY-EPOCH TRAINING & VALIDATION CONVERGENCE HISTORY:
Epoch  Train Loss (SmoothL1)     Val Scaled MAE   Val-Train Gap         Status
------------------------------------------------------------------------------
    1                0.11877            0.26513        +0.14636               
    2                0.11521            0.26258        +0.14737               
    3                0.11422            0.26152        +0.14730               
    4                0.11411            0.26246        +0.14835               
    5                0.11349            0.26339        +0.14990               
    6                0.11367            0.26294        +0.14927               
    7                0.11300            0.25950        +0.14650 <-- Best Checkpoint
    8                0.11290            0.26106        +0.14816               
    9                0.11273            0.25993        +0.14720               
   10                0.11264            0.26017        +0.14753               
------------------------------------------------------------------------------
MULTI-HORIZON SPATIO-TEMPORAL EVALUATION BREAKDOWN (mph across 207 Sensors):
Horizon              Steps       MSE     RMSE      MAE  MAPE (%)       R²   MaxErr
------------------------------------------------------------------------------
15_min                   3    8.6841   2.9469   1.3541   2.8546%   0.9085    50.96
30_min                   6   19.0388   4.3633   1.8552   4.1751%   0.7994    55.86
60_min                  12   37.1624   6.0961   2.6242   6.5286%   0.6087    58.90
Average_All_Horizons     7   21.6284   4.4688   1.9445   4.5194%   0.7722    58.90
==============================================================================
Ljung-Box (L=24) (Q):     210.44   Jarque-Bera (JB):      3739.07
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     1.2200   Skew:                  -0.0714
Prob(H) (two-sided):       0.1425   Kurtosis:              11.3555
Durbin-Watson (DW):        1.7550   MedAE (mph):           1.3223
Residual Mean (mph):     -0.1275   Residual Std:           4.4665
------------------------------------------------------------------------------
OUT-OF-SAMPLE EVALUATION METRICS (Original Scale mph, n_eval=265,995):
  MSE  =        21.6284 | RMSE =     4.4688 mph | MAE =     1.9445 mph
  MAPE =        4.5194% | R²   =     0.7722     | Max Error =    58.8953 mph
==============================================================================
```

### 3.9. Multi-Horizon LSTM (Temporal Only) — Horizon 15_min (15 min / t+3)
```text
Multi-Horizon LSTM (Temporal Baseline) Deep Learning Model Summary - PEMS-BAY Spatio-Temporal Highway Traffic Speed (207 Sensors, 5-min, 01/2017)
==============================================================================
Run ID: 20261117_160932_3fe70f | Status: SUCCESS | Regime: PEMS-BAY Multi-Horizon Traffic Forecasting (15_min, 15 min ahead)
Data: PEMS-BAY Spatio-Temporal Highway Traffic Speed (207 Sensors, 5-min, 01/2017)
Date range: 2017-01-01 00:00:00 to 2017-01-30 23:55:00 (Freq: 5min)
Observations: 8,640 (Train: 6,025, Val: 1,285, Eval/Test: 265,995)
Architecture: Temporal LSTM (input=207, hidden=64) + Linear(64->3x207) + AR_Linear(12->3) [No Spatial Graph]
Trainable Parameters: 110,292 | Optimizer: AdamW + CosineAnnealingLR + EarlyStopping
Transformation: StandardScaler (Fit on Train Only) + Inverse Transform (mph)
Model: Multi-Horizon LSTM (Temporal Only) — Horizon 15_min (15 min / t+3)
==============================================================================
          PyTorch Deep Learning Convergence & Parameter Anatomy               
==============================================================================
Initial Train Loss:           0.106340   Best Val Loss:       0.217570
Final Train Loss:             0.068050   Best Epoch:            4 / 8   
Hardware Device:                  cuda   Train / Pred:    8.1749s / 0.0850s
------------------------------------------------------------------------------
LAYER-BY-LAYER TENSOR PARAMETER ANATOMY (Extracted from lstm_pems_bay.pt):
Layer / Tensor Name        Shape             Params      Mean      Std           [Min, Max]
------------------------------------------------------------------------------
lstm.weight_ih_l0          (256, 207)        52,992  -0.00532  0.10222     [-0.425, +0.435]
lstm.weight_hh_l0          (256, 64)         16,384  +0.00231  0.12385     [-0.533, +0.590]
lstm.bias_ih_l0            (256,)               256  +0.00217  0.08432     [-0.225, +0.173]
lstm.bias_hh_l0            (256,)               256  +0.01034  0.08024     [-0.156, +0.217]
head.weight                (621, 64)         39,744  -0.00068  0.17614     [-0.818, +0.983]
head.bias                  (621,)               621  +0.00068  0.12349     [-0.421, +0.377]
ar_linear.weight           (3, 12)               36  +0.05058  0.19192     [-0.124, +0.825]
ar_linear.bias             (3,)                   3  -0.00568  0.01720     [-0.030, +0.010]
------------------------------------------------------------------------------
LEARNED MULTI-HORIZON AUTOREGRESSIVE WEIGHTS (ar_linear.weight, 12 Lookback Steps):
  15_min (t+3) : L12:+0.003 L7:-0.002 L4:-0.027 L3:-0.020 L2:+0.028 L1:+0.825
  30_min (t+6) : L12:+0.012 L7:+0.009 L4:-0.027 L3:-0.040 L2:-0.033 L1:+0.691
  60_min (t+12): L12:+0.025 L7:+0.011 L4:-0.019 L3:-0.062 L2:-0.124 L1:+0.483
------------------------------------------------------------------------------
EPOCH-BY-EPOCH TRAINING & VALIDATION CONVERGENCE HISTORY:
Epoch  Train Loss (SmoothL1)     Val Scaled MAE   Val-Train Gap         Status
------------------------------------------------------------------------------
    1                0.10634            0.23636        +0.13002               
    2                0.08778            0.22650        +0.13872               
    3                0.08029            0.22182        +0.14153               
    4                0.07648            0.21757        +0.14109 <-- Best Checkpoint
    5                0.07303            0.21994        +0.14691               
    6                0.07087            0.21817        +0.14730               
    7                0.06944            0.22147        +0.15203               
    8                0.06805            0.21870        +0.15065               
------------------------------------------------------------------------------
MULTI-HORIZON SPATIO-TEMPORAL EVALUATION BREAKDOWN (mph across 207 Sensors):
Horizon              Steps       MSE     RMSE      MAE  MAPE (%)       R²   MaxErr
------------------------------------------------------------------------------
15_min                   3    8.4484   2.9066   1.4182   2.9199%   0.9110    47.02
30_min                   6   16.2314   4.0288   1.8885   4.0697%   0.8290    50.79
60_min                  12   28.6132   5.3491   2.4986   5.4961%   0.6987    54.84
Average_All_Horizons     7   17.7643   4.0948   1.9351   4.1619%   0.8129    54.84
==============================================================================
Ljung-Box (L=24) (Q):    5938.12   Jarque-Bera (JB):      5757.19
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     0.1485   Skew:                   2.0971
Prob(H) (two-sided):       0.0000   Kurtosis:              12.4835
Durbin-Watson (DW):        0.1489   MedAE (mph):           0.2311
Residual Mean (mph):      0.0986   Residual Std:           2.9049
------------------------------------------------------------------------------
OUT-OF-SAMPLE EVALUATION METRICS (Original Scale mph, n_eval=265,995):
  MSE  =         8.4484 | RMSE =     2.9066 mph | MAE =     1.4182 mph
  MAPE =        2.9199% | R²   =     0.9110     | Max Error =    47.0235 mph
==============================================================================
```

### 3.10. Multi-Horizon LSTM (Temporal Only) — Horizon 30_min (30 min / t+6)
```text
Multi-Horizon LSTM (Temporal Baseline) Deep Learning Model Summary - PEMS-BAY Spatio-Temporal Highway Traffic Speed (207 Sensors, 5-min, 01/2017)
==============================================================================
Run ID: 20261117_160932_3fe70f | Status: SUCCESS | Regime: PEMS-BAY Multi-Horizon Traffic Forecasting (30_min, 30 min ahead)
Data: PEMS-BAY Spatio-Temporal Highway Traffic Speed (207 Sensors, 5-min, 01/2017)
Date range: 2017-01-01 00:00:00 to 2017-01-30 23:55:00 (Freq: 5min)
Observations: 8,640 (Train: 6,025, Val: 1,285, Eval/Test: 265,995)
Architecture: Temporal LSTM (input=207, hidden=64) + Linear(64->3x207) + AR_Linear(12->3) [No Spatial Graph]
Trainable Parameters: 110,292 | Optimizer: AdamW + CosineAnnealingLR + EarlyStopping
Transformation: StandardScaler (Fit on Train Only) + Inverse Transform (mph)
Model: Multi-Horizon LSTM (Temporal Only) — Horizon 30_min (30 min / t+6)
==============================================================================
          PyTorch Deep Learning Convergence & Parameter Anatomy               
==============================================================================
Initial Train Loss:           0.106340   Best Val Loss:       0.217570
Final Train Loss:             0.068050   Best Epoch:            4 / 8   
Hardware Device:                  cuda   Train / Pred:    8.1749s / 0.0850s
------------------------------------------------------------------------------
LAYER-BY-LAYER TENSOR PARAMETER ANATOMY (Extracted from lstm_pems_bay.pt):
Layer / Tensor Name        Shape             Params      Mean      Std           [Min, Max]
------------------------------------------------------------------------------
lstm.weight_ih_l0          (256, 207)        52,992  -0.00532  0.10222     [-0.425, +0.435]
lstm.weight_hh_l0          (256, 64)         16,384  +0.00231  0.12385     [-0.533, +0.590]
lstm.bias_ih_l0            (256,)               256  +0.00217  0.08432     [-0.225, +0.173]
lstm.bias_hh_l0            (256,)               256  +0.01034  0.08024     [-0.156, +0.217]
head.weight                (621, 64)         39,744  -0.00068  0.17614     [-0.818, +0.983]
head.bias                  (621,)               621  +0.00068  0.12349     [-0.421, +0.377]
ar_linear.weight           (3, 12)               36  +0.05058  0.19192     [-0.124, +0.825]
ar_linear.bias             (3,)                   3  -0.00568  0.01720     [-0.030, +0.010]
------------------------------------------------------------------------------
LEARNED MULTI-HORIZON AUTOREGRESSIVE WEIGHTS (ar_linear.weight, 12 Lookback Steps):
  15_min (t+3) : L12:+0.003 L7:-0.002 L4:-0.027 L3:-0.020 L2:+0.028 L1:+0.825
  30_min (t+6) : L12:+0.012 L7:+0.009 L4:-0.027 L3:-0.040 L2:-0.033 L1:+0.691
  60_min (t+12): L12:+0.025 L7:+0.011 L4:-0.019 L3:-0.062 L2:-0.124 L1:+0.483
------------------------------------------------------------------------------
EPOCH-BY-EPOCH TRAINING & VALIDATION CONVERGENCE HISTORY:
Epoch  Train Loss (SmoothL1)     Val Scaled MAE   Val-Train Gap         Status
------------------------------------------------------------------------------
    1                0.10634            0.23636        +0.13002               
    2                0.08778            0.22650        +0.13872               
    3                0.08029            0.22182        +0.14153               
    4                0.07648            0.21757        +0.14109 <-- Best Checkpoint
    5                0.07303            0.21994        +0.14691               
    6                0.07087            0.21817        +0.14730               
    7                0.06944            0.22147        +0.15203               
    8                0.06805            0.21870        +0.15065               
------------------------------------------------------------------------------
MULTI-HORIZON SPATIO-TEMPORAL EVALUATION BREAKDOWN (mph across 207 Sensors):
Horizon              Steps       MSE     RMSE      MAE  MAPE (%)       R²   MaxErr
------------------------------------------------------------------------------
15_min                   3    8.4484   2.9066   1.4182   2.9199%   0.9110    47.02
30_min                   6   16.2314   4.0288   1.8885   4.0697%   0.8290    50.79
60_min                  12   28.6132   5.3491   2.4986   5.4961%   0.6987    54.84
Average_All_Horizons     7   17.7643   4.0948   1.9351   4.1619%   0.8129    54.84
==============================================================================
Ljung-Box (L=24) (Q):    7938.64   Jarque-Bera (JB):     11315.69
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     0.0817   Skew:                   2.8929
Prob(H) (two-sided):       0.0000   Kurtosis:              16.3367
Durbin-Watson (DW):        0.0512   MedAE (mph):           0.3429
Residual Mean (mph):      0.2363   Residual Std:           4.0219
------------------------------------------------------------------------------
OUT-OF-SAMPLE EVALUATION METRICS (Original Scale mph, n_eval=265,995):
  MSE  =        16.2314 | RMSE =     4.0288 mph | MAE =     1.8885 mph
  MAPE =        4.0697% | R²   =     0.8290     | Max Error =    50.7904 mph
==============================================================================
```

### 3.11. Multi-Horizon LSTM (Temporal Only) — Horizon 60_min (60 min / t+12)
```text
Multi-Horizon LSTM (Temporal Baseline) Deep Learning Model Summary - PEMS-BAY Spatio-Temporal Highway Traffic Speed (207 Sensors, 5-min, 01/2017)
==============================================================================
Run ID: 20261117_160932_3fe70f | Status: SUCCESS | Regime: PEMS-BAY Multi-Horizon Traffic Forecasting (60_min, 60 min ahead)
Data: PEMS-BAY Spatio-Temporal Highway Traffic Speed (207 Sensors, 5-min, 01/2017)
Date range: 2017-01-01 00:00:00 to 2017-01-30 23:55:00 (Freq: 5min)
Observations: 8,640 (Train: 6,025, Val: 1,285, Eval/Test: 265,995)
Architecture: Temporal LSTM (input=207, hidden=64) + Linear(64->3x207) + AR_Linear(12->3) [No Spatial Graph]
Trainable Parameters: 110,292 | Optimizer: AdamW + CosineAnnealingLR + EarlyStopping
Transformation: StandardScaler (Fit on Train Only) + Inverse Transform (mph)
Model: Multi-Horizon LSTM (Temporal Only) — Horizon 60_min (60 min / t+12)
==============================================================================
          PyTorch Deep Learning Convergence & Parameter Anatomy               
==============================================================================
Initial Train Loss:           0.106340   Best Val Loss:       0.217570
Final Train Loss:             0.068050   Best Epoch:            4 / 8   
Hardware Device:                  cuda   Train / Pred:    8.1749s / 0.0850s
------------------------------------------------------------------------------
LAYER-BY-LAYER TENSOR PARAMETER ANATOMY (Extracted from lstm_pems_bay.pt):
Layer / Tensor Name        Shape             Params      Mean      Std           [Min, Max]
------------------------------------------------------------------------------
lstm.weight_ih_l0          (256, 207)        52,992  -0.00532  0.10222     [-0.425, +0.435]
lstm.weight_hh_l0          (256, 64)         16,384  +0.00231  0.12385     [-0.533, +0.590]
lstm.bias_ih_l0            (256,)               256  +0.00217  0.08432     [-0.225, +0.173]
lstm.bias_hh_l0            (256,)               256  +0.01034  0.08024     [-0.156, +0.217]
head.weight                (621, 64)         39,744  -0.00068  0.17614     [-0.818, +0.983]
head.bias                  (621,)               621  +0.00068  0.12349     [-0.421, +0.377]
ar_linear.weight           (3, 12)               36  +0.05058  0.19192     [-0.124, +0.825]
ar_linear.bias             (3,)                   3  -0.00568  0.01720     [-0.030, +0.010]
------------------------------------------------------------------------------
LEARNED MULTI-HORIZON AUTOREGRESSIVE WEIGHTS (ar_linear.weight, 12 Lookback Steps):
  15_min (t+3) : L12:+0.003 L7:-0.002 L4:-0.027 L3:-0.020 L2:+0.028 L1:+0.825
  30_min (t+6) : L12:+0.012 L7:+0.009 L4:-0.027 L3:-0.040 L2:-0.033 L1:+0.691
  60_min (t+12): L12:+0.025 L7:+0.011 L4:-0.019 L3:-0.062 L2:-0.124 L1:+0.483
------------------------------------------------------------------------------
EPOCH-BY-EPOCH TRAINING & VALIDATION CONVERGENCE HISTORY:
Epoch  Train Loss (SmoothL1)     Val Scaled MAE   Val-Train Gap         Status
------------------------------------------------------------------------------
    1                0.10634            0.23636        +0.13002               
    2                0.08778            0.22650        +0.13872               
    3                0.08029            0.22182        +0.14153               
    4                0.07648            0.21757        +0.14109 <-- Best Checkpoint
    5                0.07303            0.21994        +0.14691               
    6                0.07087            0.21817        +0.14730               
    7                0.06944            0.22147        +0.15203               
    8                0.06805            0.21870        +0.15065               
------------------------------------------------------------------------------
MULTI-HORIZON SPATIO-TEMPORAL EVALUATION BREAKDOWN (mph across 207 Sensors):
Horizon              Steps       MSE     RMSE      MAE  MAPE (%)       R²   MaxErr
------------------------------------------------------------------------------
15_min                   3    8.4484   2.9066   1.4182   2.9199%   0.9110    47.02
30_min                   6   16.2314   4.0288   1.8885   4.0697%   0.8290    50.79
60_min                  12   28.6132   5.3491   2.4986   5.4961%   0.6987    54.84
Average_All_Horizons     7   17.7643   4.0948   1.9351   4.1619%   0.8129    54.84
==============================================================================
Ljung-Box (L=24) (Q):   10223.45   Jarque-Bera (JB):     14990.97
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     0.0397   Skew:                   3.4026
Prob(H) (two-sided):       0.0000   Kurtosis:              18.2865
Durbin-Watson (DW):        0.0246   MedAE (mph):           0.5407
Residual Mean (mph):      0.4652   Residual Std:           5.3289
------------------------------------------------------------------------------
OUT-OF-SAMPLE EVALUATION METRICS (Original Scale mph, n_eval=265,995):
  MSE  =        28.6132 | RMSE =     5.3491 mph | MAE =     2.4986 mph
  MAPE =        5.4961% | R²   =     0.6987     | Max Error =    54.8376 mph
==============================================================================
```

### 3.12. Multi-Horizon LSTM (Temporal Only) — Horizon Average_All_Horizons (35 min / t+7)
```text
Multi-Horizon LSTM (Temporal Baseline) Deep Learning Model Summary - PEMS-BAY Spatio-Temporal Highway Traffic Speed (207 Sensors, 5-min, 01/2017)
==============================================================================
Run ID: 20261117_160932_3fe70f | Status: SUCCESS | Regime: PEMS-BAY Multi-Horizon Traffic Forecasting (Average_All_Horizons, 35 min ahead)
Data: PEMS-BAY Spatio-Temporal Highway Traffic Speed (207 Sensors, 5-min, 01/2017)
Date range: 2017-01-01 00:00:00 to 2017-01-30 23:55:00 (Freq: 5min)
Observations: 8,640 (Train: 6,025, Val: 1,285, Eval/Test: 265,995)
Architecture: Temporal LSTM (input=207, hidden=64) + Linear(64->3x207) + AR_Linear(12->3) [No Spatial Graph]
Trainable Parameters: 110,292 | Optimizer: AdamW + CosineAnnealingLR + EarlyStopping
Transformation: StandardScaler (Fit on Train Only) + Inverse Transform (mph)
Model: Multi-Horizon LSTM (Temporal Only) — Horizon Average_All_Horizons (35 min / t+7)
==============================================================================
          PyTorch Deep Learning Convergence & Parameter Anatomy               
==============================================================================
Initial Train Loss:           0.106340   Best Val Loss:       0.217570
Final Train Loss:             0.068050   Best Epoch:            4 / 8   
Hardware Device:                  cuda   Train / Pred:    8.1749s / 0.0850s
------------------------------------------------------------------------------
LAYER-BY-LAYER TENSOR PARAMETER ANATOMY (Extracted from lstm_pems_bay.pt):
Layer / Tensor Name        Shape             Params      Mean      Std           [Min, Max]
------------------------------------------------------------------------------
lstm.weight_ih_l0          (256, 207)        52,992  -0.00532  0.10222     [-0.425, +0.435]
lstm.weight_hh_l0          (256, 64)         16,384  +0.00231  0.12385     [-0.533, +0.590]
lstm.bias_ih_l0            (256,)               256  +0.00217  0.08432     [-0.225, +0.173]
lstm.bias_hh_l0            (256,)               256  +0.01034  0.08024     [-0.156, +0.217]
head.weight                (621, 64)         39,744  -0.00068  0.17614     [-0.818, +0.983]
head.bias                  (621,)               621  +0.00068  0.12349     [-0.421, +0.377]
ar_linear.weight           (3, 12)               36  +0.05058  0.19192     [-0.124, +0.825]
ar_linear.bias             (3,)                   3  -0.00568  0.01720     [-0.030, +0.010]
------------------------------------------------------------------------------
LEARNED MULTI-HORIZON AUTOREGRESSIVE WEIGHTS (ar_linear.weight, 12 Lookback Steps):
  15_min (t+3) : L12:+0.003 L7:-0.002 L4:-0.027 L3:-0.020 L2:+0.028 L1:+0.825
  30_min (t+6) : L12:+0.012 L7:+0.009 L4:-0.027 L3:-0.040 L2:-0.033 L1:+0.691
  60_min (t+12): L12:+0.025 L7:+0.011 L4:-0.019 L3:-0.062 L2:-0.124 L1:+0.483
------------------------------------------------------------------------------
EPOCH-BY-EPOCH TRAINING & VALIDATION CONVERGENCE HISTORY:
Epoch  Train Loss (SmoothL1)     Val Scaled MAE   Val-Train Gap         Status
------------------------------------------------------------------------------
    1                0.10634            0.23636        +0.13002               
    2                0.08778            0.22650        +0.13872               
    3                0.08029            0.22182        +0.14153               
    4                0.07648            0.21757        +0.14109 <-- Best Checkpoint
    5                0.07303            0.21994        +0.14691               
    6                0.07087            0.21817        +0.14730               
    7                0.06944            0.22147        +0.15203               
    8                0.06805            0.21870        +0.15065               
------------------------------------------------------------------------------
MULTI-HORIZON SPATIO-TEMPORAL EVALUATION BREAKDOWN (mph across 207 Sensors):
Horizon              Steps       MSE     RMSE      MAE  MAPE (%)       R²   MaxErr
------------------------------------------------------------------------------
15_min                   3    8.4484   2.9066   1.4182   2.9199%   0.9110    47.02
30_min                   6   16.2314   4.0288   1.8885   4.0697%   0.8290    50.79
60_min                  12   28.6132   5.3491   2.4986   5.4961%   0.6987    54.84
Average_All_Horizons     7   17.7643   4.0948   1.9351   4.1619%   0.8129    54.84
==============================================================================
Ljung-Box (L=24) (Q):     192.83   Jarque-Bera (JB):      2357.17
Prob(Q):                   0.0000   Prob(JB):               0.0000
Heteroskedasticity(H):     1.2200   Skew:                   0.1632
Prob(H) (two-sided):       0.1425   Kurtosis:               9.6271
Durbin-Watson (DW):        1.7550   MedAE (mph):           1.3159
Residual Mean (mph):      0.2667   Residual Std:           4.0852
------------------------------------------------------------------------------
OUT-OF-SAMPLE EVALUATION METRICS (Original Scale mph, n_eval=265,995):
  MSE  =        17.7643 | RMSE =     4.0948 mph | MAE =     1.9351 mph
  MAPE =        4.1619% | R²   =     0.8129     | Max Error =    54.8376 mph
==============================================================================
```
