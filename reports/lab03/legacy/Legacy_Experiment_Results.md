# Legacy Experiment Results - Lab 03 (archived notebook)

> **Provenance:** This table is transcribed from the saved outputs and Markdown commentary in `notebooks/lab03/archive/Lab03_TimeSeries_legacy.ipynb`. It describes a previous run with library-based XGBoost / Linear Regression / Random Forest and a 80/20 time split. It is kept separate from the NumPy-from-scratch experiment in the current pipeline. Do not compare these values as a controlled benchmark: the split, feature set, implementations and tuning protocol differ.

## Data cleaning already executed

- Raw CSV rows: **121,273**.
- Duplicate timestamps: **4** (the original file was not sorted).
- Clean regular hourly series: **121,296 observations**, from **2004-10-01 01:00** to **2018-08-03 00:00**.
- Missing hourly periods reinserted: **27**; cleaned missing values remaining: **0**.
- Target: `AEP_MW`, unit: MW.

## Baseline and model results (legacy single-step test)

| Model | MAE (MW) | RMSE (MW) | MAPE | R² |
|---|---:|---:|---:|---:|
| Naive (lag_1) | 407.6 | 525.5 | 2.81% | 0.954 |
| Linear Regression (24 lags) | 147.9 | 203.4 | 1.02% | 0.993 |
| XGBoost default (24 lags) | 152.5 | 205.9 | 1.05% | 0.993 |
| Random Forest (24 lags) | - | 211.9 | - | - |
| XGBoost tuned (24 lags) | 138.5 | 189.0 | 0.96% | - |

The archived notebook reports that the tuned XGBoost reduced Test RMSE by about **8.2%** versus the default XGBoost. The same commentary notes that the default XGBoost did not outperform Linear Regression on that split.

## Advanced results already executed

- Rolling mean/std with 24- and 168-hour windows, shifted by one timestamp: on a common 24,226-row Test subset, XGBoost RMSE changed **205.9 → 203.7** and Linear Regression **203.4 → 203.0**. The four rolling features jointly accounted for approximately **0.8%** of XGBoost feature importance.
- Multi-step, 24-hour forecast: average RMSE across the 24 horizons was approximately **962.7** for recursive, **914.9** for direct, and **1,220.9** for seasonal-naive. The notebook explicitly notes the model configurations differed, so this is an indicative comparison rather than a perfectly matched ablation.
- Tuning: 8 sampled parameter combinations using 3-fold `TimeSeriesSplit` on Train only. The noted best configuration was `n_estimators=200`, `max_depth=8`, `learning_rate=0.05`, `subsample=1.0`, `colsample_bytree=1.0`, `min_child_weight=10`, with CV RMSE about **209.1**.
- Feature importance: `lag_1` and `lag_2` jointly contributed about **78.9%** in the legacy XGBoost run; lags 22-24 jointly contributed about **11.7%**.
- Lag sweep: the notebook reports Linear Regression RMSE about **149.5** at 168 lags, while untuned XGBoost was about **189.2** at 168 lags.

## Legacy split

- Train: **97,018 rows**, 2004-10-02 01:00 to 2015-10-27 10:00.
- Test: **24,254 rows**, 2015-10-27 11:00 to 2018-08-03 00:00.
- No shuffle was used. This is an 80/20 split, unlike the current standardized 70/15/15 Train/Validation/Test split.

## Reused artifacts

- Original notebook, including executed outputs: `notebooks/lab03/archive/Lab03_TimeSeries_legacy.ipynb`.
- Extracted prior figures: `images/lab03/legacy/`.
- Original helper source was supplied separately in the uploaded ZIP and is retained at `legacy/source/` for provenance; the new pipeline uses the scratch modules in `src/lab03/`.

## Interpretation limits

These are historical notebook results, not reruns from the present codebase. The exact environment, library versions and full raw source data are not bundled here. The Lab-03-only source package does not bundle `AEP_hourly.csv`. The current pipeline therefore stops by default until the real CSV is placed in `data/raw/`; synthetic data are used only when explicitly requested with `--synthetic` and are labelled as a demo.
