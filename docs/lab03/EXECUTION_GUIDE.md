# Lab 03 Execution Guide

## Environment

- Python 3.10+
- Install packages from the root `requirements.txt`
- Keep the working directory at the Lab 03 project root
- Put `AEP_hourly.csv` in `data/raw/`

## Full run

```bash
python -m pip install -r requirements.txt
python scripts/run_lab.py
```

This command performs ingestion, cleaning, feature engineering, chronological split, 5-fold walk-forward CV, candidate training, validation-based selection, final refit, inference, evaluation, logging, figure creation, model summary and report generation.

## Alternatives

```bash
python scripts/run_lab.py --fast
python scripts/run_lab.py --data path/to/AEP_hourly.csv
python scripts/run_lab.py --synthetic --fast
python scripts/run_lab.py --no-cv --fast
```

- `--fast`: smaller tree ensembles/depth and row caps for development checks. Not the recommended final experiment configuration.
- `--data`: override the AEP CSV location.
- `--synthetic`: explicitly generate synthetic demo data if the real CSV is absent or to force a demo run.
- `--no-cv`: skips the required 5-fold stage; use only for debugging, not the final deliverable run.

## Alternative entry point

```bash
python script/lab03/run_lab03.py
```

Both entry points invoke the same root `pipeline.py`.

## Verify completion

A successful run must end with `LAB03 PIPELINE SUCCESS` in `logs/lab03/<run_id>.log`. Confirm the three reports, all PNG files, `model_summary` files, CV fold metrics, model `.pkl` files and `data/processed/lab03/run_result.json` exist. The report PDF should open in a PDF viewer.

## Run tests

```bash
python tests/lab03/run_all_tests.py
```
