# Lab 03 Methodology — Hourly Energy Consumption Forecasting

## 1. Problem formulation

Given a univariate hourly load sequence `y_t`, estimate the next-hour value `y_t` from values available before t. The cleaned source target is `AEP_MW`; the time key is `Datetime`.

## 2. Data preparation

- Sort timestamps and remove duplicates.
- Regularize to an hourly index.
- Interpolate only internal gaps up to three consecutive hours.
- Fail if a longer gap or edge missing value remains.
- Keep the raw CSV unchanged and persist the cleaned series and audit separately.

## 3. Supervised features

For each target timestamp t, create lag features `y_(t-k)` for k in 1, 2, 3, 6, 12, 24, 48 and 168. Rolling mean/std use `y.shift(1)` before aggregation over 6, 12, 24 and 168 hours. Calendar variables hour/day-of-week/month are encoded with sine/cosine. Initial rows without enough history are dropped.

## 4. Temporal split and validation protocol

The supervised rows are split chronologically into 70% Train, 15% Validation and 15% Test. No shuffle is allowed. Five expanding-window folds are generated only inside Train. Each fold's training prefix precedes its validation block. This prevents training on observations that occur later than the validation period.

## 5. Candidate algorithms

- Naive Last-Value: predicts `lag_1`.
- Linear Regression: ordinary least squares with a small ridge stabilizer.
- CART: greedily maximizes reduction in squared error using quantile-based candidate thresholds.
- Random Forest: bootstrap aggregation of independent scratch CART trees with random feature subsets.
- XGBoost-style: sequential histogram gradient-boosted trees using first/second derivatives for squared error and L1/L2 regularization. This is an educational implementation, not the official XGBoost engine.

All four learned estimators are implemented directly with NumPy. Regression metrics are also implemented directly; scikit-learn/XGBoost estimators are not used.

## 6. Model selection and final test

Each candidate is fitted on Train and scored on the same Validation timestamps. The lowest Validation RMSE is champion. After champion is recorded, final candidates are refit on Train+Validation and evaluated on the untouched Test partition. Test scores do not alter selection.

## 7. Two forecast protocols

- Test rolling one-step: prior actual Test observations may form later lag features, simulating continuous hourly operation with an observation arriving each hour.
- Future recursive fixed-origin: only history available at the final origin is used; each new prediction is appended to history and becomes input for the next horizon step. Default horizon is 24 hours.

The two protocols must not be presented as equivalent.

## 8. Evaluation metrics

MAE, MSE and RMSE are in the target unit (MW); MAPE and sMAPE are percentage metrics; R² compares residual sum of squares with target variation. MAPE omits zero actual values and may be unstable near zero. R² may be negative out of sample.

## 9. Reproducibility and provenance

Random seeds are set in model configuration. Each run writes a timestamped log, run JSON, metric tables, model artifacts, figures and MD/HTML/PDF report. Original notebook experiments are retained separately; their estimator and split protocol differs from this scratch implementation.
