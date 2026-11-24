# Lab 03 Deliverable Checklist

## Completed in the runnable notebook

- [x] Current notebook uses the available `src/lab03` data, features, CV, metrics and scratch-model modules.
- [x] Real AEP data is present at `data/raw/AEP_hourly.csv`; the notebook fails clearly instead of silently substituting synthetic data.
- [x] Chronological 70/15/15 split, five expanding-window CV folds on Train, Validation-based selection and held-out Test evaluation.
- [x] Past-only lag/rolling features and separate rolling Test versus 24-hour recursive forecast protocols.
- [x] Each run saves its data, figures, predictions, five candidate models, champion forecast model and run summary under `reports/lab03/notebook_runs/<run_id>/`.
- [x] Latest Markdown, HTML and PDF reports are written under `reports/lab03/`.
- [x] Latest `model_summary.csv`, `.json`, `.md`, CV fold metrics and Validation/Test JSON metrics are written under `reports/model_summary/lab03/`.
- [x] Model summary contains CV, Validation and Test metrics, timing, training-row counts, feature count, model parameters and model artifact paths.
- [x] Notebook was executed end-to-end with the bundled AEP data; reports and all five model-summary artifacts were checked.

## Remaining gaps

- [ ] A root `pipeline.py` and `scripts/run_lab.py` do not exist. The documented `script/lab03/run_lab03.py` imports the missing root pipeline, so it is not a working CLI entry point. Use the notebook for now.
- [ ] Root `config.yaml` has shared Lab01/Lab02/Lab04 sections but no Lab03 section. One existing Lab03 test still expects a Lab03-only configuration and therefore fails.
- [ ] The notebook does not yet write timestamped START/SUCCESS logs or progress-status files.
- [ ] The earlier legacy notebook is no longer present in the current worktree; its old `src.lab03_utils` imports were unavailable.

## Validation notes

- Notebook execution on the bundled real dataset succeeds and produces non-empty MD/HTML/PDF reports and model-summary outputs.
- The report-bundle unit test covers non-empty Markdown, HTML and PDF generation.
- The full Lab03 unit-test runner currently has one pre-existing configuration assertion failure; the other eight tests pass.
