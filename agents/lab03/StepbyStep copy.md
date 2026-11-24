# STEP-BY-STEP — LAB 03 PIPELINE

1. Put `AEP_hourly.csv` into `data/raw/` with `Datetime` and `AEP_MW` columns.
2. Install packages using `pip install -r requirements.txt`.
3. Run the full flow once using `python scripts/run_lab.py` (or `python script/lab03/run_lab03.py`).
4. The root `pipeline.py` runs ingestion, cleaning, causal features, 70/15/15 temporal split, 5-fold walk-forward CV, training, Validation selection, Train+Validation refit, Test evaluation, future recursive inference, logging, plotting and reporting.
5. Check `logs/lab03/`, `reports/lab03/`, `reports/model_summary/lab03/`, `models/lab03/`, `images/lab03/` and `data/processed/lab03/`.
6. Run tests using `python tests/lab03/run_all_tests.py`.

Use `python scripts/run_lab.py --synthetic --fast` only for a labelled code smoke test. Do not report synthetic metrics as AEP results.
