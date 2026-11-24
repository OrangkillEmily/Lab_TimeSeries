# Lab 03 Deliverable Checklist

## Architecture and source
- [x] Lab 03-only package; no Lab 01/02/04 implementation copied into the package.
- [x] Repository-style directories: `agents/lab03`, `docs/lab03`, `script/lab03`, `src/lab03`, `tests/lab03`, `data/processed/lab03`, `models/lab03`, `logs/lab03`, `reports/lab03`, `reports/model_summary/lab03`, `progress_results/lab03`, `notebooks/lab03`.
- [x] Root `config.yaml` contains a `lab03` section only.
- [x] Root `pipeline.py` orchestrates all phases.
- [x] Both `scripts/run_lab.py` (required entrypoint) and `script/lab03/run_lab03.py` call the same pipeline.
- [x] One estimator per `.py` file under `src/lab03/lab03_models/`.
- [x] New estimator code and regression metrics do not import scikit-learn or XGBoost.

## Methodology
- [x] Chronological Train/Validation/Test = 70/15/15, no shuffle.
- [x] Five expanding-window walk-forward folds, limited to Train.
- [x] Validation RMSE selects champion; Test is held out from selection.
- [x] Past-only lag and rolling features.
- [x] Rolling one-step Test protocol and 24-step recursive future inference are distinguished.

## Logs and outputs
- [x] Timestamped run log with START/SUCCESS and exception traceback on failure.
- [x] All generated PNG figures go into `images/lab03/`.
- [x] Report generator writes `.md`, `.html`, and `.pdf`.
- [x] `model_summary.csv`, `.json`, `.md`, and CV fold metrics are generated.
- [x] Cleaned data, split features, metrics and predictions are saved under Lab 03 directories.

## Documentation
- [x] Detailed documentation exists for each of the five algorithms.
- [x] Every algorithm document contains Purpose, Input - Output, How to do, Evaluation Metrics and Used Data.
- [x] Phase documents cover ingestion, cleaning, feature engineering, splitting, CV, training, model selection, inference, evaluation, reporting and experiment tracking.
- [x] Every phase document contains Purpose, Input - Output, How to do, Evaluation Metrics and Used Data.
- [x] Execution guide, file guide, architecture and model summary specification included.

## Data and provenance
- [x] Original executed notebook and legacy source code are preserved.
- [x] Prior notebook images and results are stored separately as legacy provenance.
- [x] Default run refuses to label synthetic data as AEP; synthetic data requires the explicit `--synthetic` option.
- [ ] User must place original `AEP_hourly.csv` into `data/raw/` before running a real-data experiment.

## Verified in the build environment
- [x] Unit tests pass.
- [x] Python source compilation passes.
- [x] Explicit synthetic fast smoke run completes all pipeline phases.
- [x] Smoke run creates all three report formats, model summary, log and 8 images.
- [x] Generated PDF rendered for layout inspection.
