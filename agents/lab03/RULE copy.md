# LAB 03 — IMPLEMENTATION AND EXPERIMENT RULES

## 1. Prevent data leakage
- Sort hourly timestamps; never shuffle a time-series split.
- Use Train/Validation/Test = 70/15/15 in chronological order.
- Use five expanding-window walk-forward folds inside Train only.
- Lag features at t may use only observations before t. Rolling mean/std must shift by one timestamp before aggregation.
- Fit every learned transformation/estimator inside the relevant training partition or fold.
- Choose the champion using Validation RMSE. Do not use Test to select models or hyperparameters.
- Distinguish rolling one-step Test scoring from fixed-origin recursive multi-step forecasting.

## 2. Model implementation
- Naive, Linear Regression, CART, Random Forest and the educational XGBoost-style booster each have a dedicated `.py` file under `src/lab03/lab03_models/`.
- Implement estimators and metrics directly with NumPy; do not call scikit-learn or XGBoost estimators.
- Explain the educational booster honestly: it is XGBoost-style, not the official XGBoost package.
- Fix random seeds and record model parameters, row caps and fit/predict timing.

## 3. Required run artifacts
- One timestamped log per run.
- Markdown, HTML and PDF reports.
- All figures saved as image files under `images/lab03/`.
- CSV/JSON/Markdown model summary and fold-level CV metrics.
- Cleaned data, feature partitions, predictions, recursive forecast and audit metadata.

## 4. Documentation
Every algorithm and every pipeline phase must document Purpose, Input - Output, How to do, Evaluation Metrics and Used Data.

## 5. Provenance
Keep original notebook outputs and library-based experiment results under the legacy folders. Never mix those scores with the new scratch-model experiment table.
