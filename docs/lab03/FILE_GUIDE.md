# Lab 03 File Guide

| Path | Purpose |
|---|---|
| `config.yaml` | Lab 03-only configuration, including data schema, split, CV and model hyperparameters |
| `pipeline.py` | Single end-to-end orchestration layer |
| `scripts/run_lab.py` | Required one-command launcher |
| `script/lab03/run_lab03.py` | Repository-style Lab 03 launcher |
| `src/lab03/data.py` | Data loading, cleaning and chronological split |
| `src/lab03/features.py` | Lag, rolling, calendar features and recursive inference features |
| `src/lab03/cv.py` | Expanding-window walk-forward split and evaluation |
| `src/lab03/metrics.py` | NumPy regression metrics |
| `src/lab03/lab03_models/` | One self-implemented estimator per Python file |
| `src/lab03/lab03_utils/report.py` | Markdown/HTML/PDF report generator |
| `src/lab03/plotting.py` | Saves every generated plot under `images/lab03/` |
| `data/raw/` | User-supplied original data; never overwritten |
| `data/processed/lab03/` | Cleaned series, features, predictions and audit files |
| `models/lab03/` | Serialized fitted scratch models |
| `logs/lab03/` | One timestamped log file per run |
| `reports/lab03/` | Main report in MD/HTML/PDF and separate legacy notes |
| `reports/model_summary/lab03/` | Model summary tables and fold metrics |
| `images/lab03/` | All generated PNG figures and separately preserved legacy images |
| `progress_results/lab03/` | Latest-run status JSON |
| `docs/lab03/models/` | One detailed algorithm document per model |
| `docs/lab03/phases/` | Purpose/Input-Output/How-to/Evaluation/Used-Data for every pipeline phase |
| `notebooks/lab03/` | Original executed Lab 03 notebook and archived copy |
| `legacy/` | Original helper code/methodology preserved for provenance; not imported by the new pipeline |
| `tests/lab03/` | Unit tests and lab-specific test runner |
