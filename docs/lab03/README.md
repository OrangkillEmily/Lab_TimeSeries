# Documentation Index — Lab 03

## Start here

1. [Lab 03 notebook](../../notebooks/lab03/Lab03_TimeSeries.ipynb) — runnable end-to-end experiment.
2. [Execution Guide](EXECUTION_GUIDE.md) — environment, notebook launch and completion checks.
3. [File Guide](FILE_GUIDE.md) — responsibilities of every directory and key module.
4. [Architecture](ARCHITECTURE.md) — pipeline phases and leakage controls.
5. [Methodology](methodology.md) — forecasting protocol, features and metric definitions.
6. [Model Summary Specification](MODEL_SUMMARY_SPEC.md) — generated model comparison fields and interpretation.
7. [Deliverable Checklist](DELIVERABLE_CHECKLIST.md) — requirement coverage.

## Algorithm documentation

Each model has a dedicated implementation file in `src/lab03/lab03_models/` and a corresponding detailed explanation:

- [Naive Last-Value Baseline](models/Naive_Baseline.md)
- [Linear Regression Scratch](models/Linear_Regression_Scratch.md)
- [CART Decision Tree Scratch](models/CART_Decision_Tree_Scratch.md)
- [Random Forest Scratch](models/Random_Forest_Scratch.md)
- [XGBoost-style Scratch](models/XGBoost_Style_Scratch.md)

Every model document contains Purpose, Input - Output, How to do, Evaluation Metrics and Used Data.

## Pipeline phase documentation

- [Data Ingestion](phases/Data_Ingestion_Phase.md)
- [Data Cleaning](phases/Data_Cleaning_Phase.md)
- [Feature Engineering](phases/Feature_Engineering_Phase.md)
- [Data Splitting](phases/Data_Splitting_Phase.md)
- [Cross Validation](phases/Cross_Validation_Phase.md)
- [Training](phases/Training_Phase.md)
- [Model Selection](phases/Model_Selection_Phase.md)
- [Inference](phases/Inference_Phase.md)
- [Evaluation](phases/Evaluation_Phase.md)
- [Reporting](phases/Reporting_Phase.md)
- [Experiment Tracking / Logging](phases/Experiment_Tracking_Phase.md)

Every phase document contains Purpose, Input - Output, How to do, Evaluation Metrics and Used Data.

## Legacy material

The earlier legacy notebook is not present in the current worktree. The runnable Lab03 implementation is [Lab03_TimeSeries.ipynb](../../notebooks/lab03/Lab03_TimeSeries.ipynb).
