# Lab 03 Model Summary

**Run:** `20261125_045448`
**Dataset:** `AEP_hourly.csv`
**Cleaned observations:** 121,296 hourly rows
**Features:** 22
**Split:** Train 84,789 | Validation 18,169 | Test 18,170
**Champion:** Linear Regression, selected by lowest Validation RMSE

Test metrics below are held-out results; they were not used to select the champion.

## Candidate comparison

| Model | Champion | CV RMSE mean | CV RMSE std | Validation RMSE | Test MAE | Test RMSE | Test MAPE (%) | Test R2 | Fit (s) | Rows used |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Linear Regression | Yes | 277.589 | 29.939 | 246.137 | 171.907 | 228.494 | 1.170 | 0.992 | 0.051 | 102958 |
| CART | No | 513.221 | 37.486 | 466.009 | 375.121 | 465.906 | 2.558 | 0.965 | 0.351 | 2500 |
| Naive Last-Value | No | 575.926 | 33.729 | 534.446 | 417.325 | 534.753 | 2.853 | 0.954 | 0.000 | 102958 |
| Random Forest | No | 756.342 | 73.556 | 755.079 | 557.222 | 705.227 | 3.900 | 0.920 | 0.286 | 2500 |
| XGBoost-style | No | 1435.365 | 72.273 | 1458.039 | 1135.129 | 1396.239 | 8.191 | 0.684 | 0.175 | 5000 |

CV uses five expanding-window folds within Train. Candidate models are fitted on Train + Validation before Test scoring.

## Model files

Candidate models fitted on Train + Validation: `reports/lab03/notebook_runs/20261125_045448/models/`.
- `naive_last_value.pkl`: `reports/lab03/notebook_runs/20261125_045448/models/naive_last_value.pkl` (candidate Test-evaluation model).
- `linear_regression.pkl`: `reports/lab03/notebook_runs/20261125_045448/models/linear_regression.pkl` (candidate Test-evaluation model).
- `cart.pkl`: `reports/lab03/notebook_runs/20261125_045448/models/cart.pkl` (candidate Test-evaluation model).
- `random_forest.pkl`: `reports/lab03/notebook_runs/20261125_045448/models/random_forest.pkl` (candidate Test-evaluation model).
- `xgboost_style.pkl`: `reports/lab03/notebook_runs/20261125_045448/models/xgboost_style.pkl` (candidate Test-evaluation model).
- `champion_future_forecast.pkl`: `reports/lab03/notebook_runs/20261125_045448/models/champion_future_forecast.pkl` (refitted on all labeled rows after Test evaluation).

The forecast model is distinct from the Train + Validation candidate used to report Test metrics; it does not change those Test metrics.

## Logs and standard directories

- `models/lab03/`: 0 `.pkl` files currently present.
- `logs/lab03/`: 0 `.log` files currently present; this notebook does not create run logs.
- Run metadata: `reports/lab03/notebook_runs/20261125_045448/run_summary.json`.
- Run reports and figures: `reports/lab03/notebook_runs/20261125_045448/`.

See `docs/lab03/MODEL_SUMMARY_SPEC.md` for field definitions and interpretation.
