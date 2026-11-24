# Lab 03 Execution Guide

## Requirements

- Python 3.10+
- Dependencies from the repository root `requirements.txt`
- The supplied `data/raw/AEP_hourly.csv` file

## Run the notebook

Open a terminal at the repository root and run:

```bash
python -m pip install -r requirements.txt
python -m jupyter lab
```

Open `notebooks/lab03/Lab03_TimeSeries.ipynb` and choose **Run All**. The notebook locates the repository root from its working directory and stops with a clear error if the AEP dataset is missing. It does not silently substitute synthetic data.

The earlier notebook referred to old `src.lab03_utils` modules that are not part of the current source tree. Use the new notebook to run Lab 03.

## Workflow and outputs

The notebook loads and regularizes the hourly series, builds causal lag/rolling/calendar features, makes a chronological 70/15/15 split, runs five expanding-window folds on Train, selects a champion using Validation RMSE, evaluates candidates on the held-out Test set and produces a recursive 24-hour forecast.

Every execution creates a unique output directory:

```text
reports/lab03/notebook_runs/<run_id>/
```

It contains cleaned data, CV/validation/test metrics, Test predictions, the 24-hour forecast, a JSON run summary, figures and the fitted champion model. Test results are reported only after model selection and do not change the selected champion.

The latest report bundle is also written directly to `reports/lab03/` as `Lab03_Report.md`, `Lab03_Report.html` and `Lab03_Report.pdf`. Latest model-comparison files are written to `reports/model_summary/lab03/`: `model_summary.csv`, `model_summary.json`, `model_summary.md`, `cv_fold_metrics.csv`, `validation_metrics.json` and `test_metrics.json`.

The notebook is the supported Lab 03 entry point. The legacy CLI runner currently depends on a root `pipeline.py` that is not present.
