# Lab 03 Architecture and Data Flow

```text
AEP_hourly.csv
    |
    v
Data Ingestion -> Data Cleaning -> Regularized hourly Series
    |                                  |
    +----------------------------------+
                       |
                       v
             Lag/Rolling/Calendar Features
                       |
                       v
          Chronological 70/15/15 Split
              /           |          \
             v            v           v
          Train       Validation     Test
             |
             +--> 5-fold expanding-window CV
             |
             +--> Fit candidate estimators
                          |
                          v
               Validation RMSE Selection
                          |
                          v
                Refit on Train + Validation
                          |
                          v
               Held-out Test Evaluation
                          |
              +-----------+-----------+
              |                       |
              v                       v
       Rolling Test Predict     24-step Recursive Future
              |                       |
              +-----------+-----------+
                          |
                          v
     Metrics + Model Summary + Figures + Logs + Reports
```

## Separation of responsibilities

- `pipeline.py` coordinates phases and owns the run lifecycle.
- `src/lab03/data.py` owns raw data parsing, regularization and splitting.
- `src/lab03/features.py` owns causal feature generation.
- `src/lab03/cv.py` owns expanding-window folds.
- `src/lab03/lab03_models/` owns all estimators; each model is isolated in one module.
- `src/lab03/metrics.py` computes metrics independently of estimators.
- `src/lab03/plotting.py` saves visual artifacts.
- `src/lab03/lab03_utils/report.py` creates the three report formats.

## Leakage controls

1. Chronological split; no random shuffle.
2. CV is limited to Train and validates on blocks later than its training prefix.
3. Lag/rolling features use observations strictly before target timestamp.
4. Every model in a CV fold is newly initialized and fit only on fold train.
5. Candidate selection uses Validation RMSE; Test cannot select or tune a model.
6. Future recursive inference feeds predictions back into the lag history and never reads future actual values.

## Experiment provenance

The original executed notebook and its previous library-estimator results are preserved under `notebooks/lab03/archive/`, `legacy/` and `reports/lab03/legacy/`. Their previous split/modeling implementation differs from the new scratch pipeline and remains separate in all reports.
