# Lab 03 Model Summary

Run: `20261125_071814`

Champion selected by Validation RMSE: **Linear Regression**.

Test metrics are for held-out Test and did not participate in selection.

| model | selected_champion | cv_RMSE_mean | cv_RMSE_std | validation_RMSE | test_MAE | test_RMSE | test_MAPE (%) | test_R2 | fit_seconds | training_rows_used |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Linear Regression | True | 277.589 | 29.939 | 246.137 | 171.907 | 228.494 | 1.170 | 0.992 | 0.044 | 102958 |
| CART | False | 513.221 | 37.486 | 466.009 | 375.121 | 465.906 | 2.558 | 0.965 | 0.236 | 2500 |
| Naive Last-Value | False | 575.926 | 33.729 | 534.446 | 417.325 | 534.753 | 2.853 | 0.954 | 0.000 | 102958 |
| Random Forest | False | 756.342 | 73.556 | 755.079 | 557.222 | 705.227 | 3.900 | 0.920 | 0.245 | 2500 |
| XGBoost-style | False | 1435.365 | 72.273 | 1458.039 | 1135.129 | 1396.239 | 8.191 | 0.684 | 0.158 | 5000 |

See `MODEL_SUMMARY_SPEC.md` in `docs/lab03/` for field definitions.
