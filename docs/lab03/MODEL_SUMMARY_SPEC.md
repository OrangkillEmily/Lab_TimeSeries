# Model Summary Specification — Lab 03

The notebook writes three synchronized views of the candidate comparison:

- `reports/model_summary/lab03/model_summary.csv`
- `reports/model_summary/lab03/model_summary.json`
- `reports/model_summary/lab03/model_summary.md`

It also writes `cv_fold_metrics.csv`, `validation_metrics.json` and `test_metrics.json`.

## Fields

| Field group | Meaning |
|---|---|
| `model` | Candidate model name |
| `cv_*_mean`, `cv_*_std` | Mean and standard deviation of fold metrics from 5-fold expanding-window CV on Train only |
| `validation_*` | Metric obtained by fitting on Train and predicting Validation |
| `test_*` | Final metric after refitting on Train+Validation and predicting held-out Test |
| `fit_seconds`, `predict_seconds_test` | Wall-clock fit/predict durations for the final refit |
| `training_rows_used` | Number of rows actually consumed by the estimator (can be capped for tree runtime) |
| `total_train_validation_rows` | Full size of the final fit partition |
| `feature_count` | Number of lag/rolling/calendar predictors |
| `model_file` | Relative path to the fitted estimator artifact |
| `model_parameters` | JSON serialization of model hyperparameters |

## Selection rule

Champion is selected using the minimum **Validation RMSE**. The model summary may be sorted by Test RMSE for reading convenience, but sorting does not change the already locked champion. Test values are never inputs to selection.

## Interpretation

- Lower MAE/RMSE means smaller errors in MW units; RMSE penalizes large misses more heavily.
- MAPE is percentage-based and omits target values equal to zero; read it carefully when actual values are close to zero.
- sMAPE uses the sum of absolute actual and predicted values in its denominator.
- R² is not a percentage accuracy score and may be negative.
- CV std is a stability diagnostic: a high value suggests performance varies by time period.
- `training_rows_used` must be read together with `total_train_validation_rows` when a runtime cap is enabled.

## Provenance

Legacy notebook scores are documented separately in `reports/lab03/legacy/Legacy_Experiment_Results.md`; do not merge them into the current scratch model summary because estimator libraries, split protocol and features differ.
