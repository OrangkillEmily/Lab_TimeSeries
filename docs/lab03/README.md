# Documentation Index — Lab 03

## Start here

1. [Execution Guide](EXECUTION_GUIDE.md) — environment, commands and completion checks.
2. [File Guide](FILE_GUIDE.md) — responsibilities of every directory and key module.
3. [Architecture](ARCHITECTURE.md) — pipeline phases and leakage controls.
4. [Methodology](methodology.md) — forecasting protocol, features and metric definitions.
5. [Model Summary](model_summary.md) — output schema and model-selection interpretation.
6. [Deliverable Checklist](DELIVERABLE_CHECKLIST.md) — requirement coverage.

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

## Original Lab material

The supplied executed notebook, legacy helper code and previous experiment output remain preserved separately. See `notebooks/lab03/archive/`, `legacy/`, `images/lab03/legacy/` and `reports/lab03/legacy/`. These materials are provenance only and are not imported by the new pipeline.
