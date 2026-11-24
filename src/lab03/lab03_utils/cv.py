"""Expanding-window walk-forward CV utilities."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .metrics import regression_metrics


def expanding_window_splits(n_samples: int, n_splits: int = 5, gap: int = 0):
    """Purpose: generate ordered expanding train/validation fold indices.

    Input: sample count, number of folds and optional temporal gap.
    Output: iterator of (train_indices, validation_indices); every validation block follows train.
    """
    if n_splits < 2:
        raise ValueError("n_splits phải >= 2.")
    fold_size = n_samples // (n_splits + 1)
    if fold_size < 1:
        raise ValueError("Không đủ mẫu cho số fold yêu cầu.")
    initial_train = n_samples - n_splits * fold_size
    for fold in range(n_splits):
        train_end = initial_train + fold * fold_size
        val_start = train_end + int(gap)
        val_end = min(n_samples, initial_train + (fold + 1) * fold_size)
        if val_start >= val_end:
            continue
        yield np.arange(0, train_end), np.arange(val_start, val_end)


def evaluate_walk_forward(X, y, model_factories: dict, n_splits=5, logger=None):
    """Purpose: compare model factories on five chronological folds of the Train partition.

    Input: Train-only X/y and dictionary {model label: zero-argument model factory}.
    Output: fold-level metrics DataFrame plus aggregate mean/std DataFrame.
    """
    records = []
    for fold_id, (train_idx, val_idx) in enumerate(expanding_window_splits(len(X), n_splits), start=1):
        for name, factory in model_factories.items():
            model = factory()
            model.fit(X.iloc[train_idx].to_numpy(dtype=float), y.iloc[train_idx].to_numpy(dtype=float))
            pred = model.predict(X.iloc[val_idx].to_numpy(dtype=float))
            metrics = regression_metrics(y.iloc[val_idx].to_numpy(dtype=float), pred)
            records.append({"fold": fold_id, "model": name, "train_rows": len(train_idx),
                            "validation_rows": len(val_idx), **metrics})
            if logger:
                logger.info("CV fold=%s model=%s train=%s val=%s RMSE=%.4f",
                            fold_id, name, len(train_idx), len(val_idx), metrics["RMSE"])
    detail = pd.DataFrame(records)
    metric_columns = ["MAE", "MSE", "RMSE", "MAPE (%)", "sMAPE (%)", "R2"]
    aggregate = detail.groupby("model")[metric_columns].agg(["mean", "std"])
    return detail, aggregate
