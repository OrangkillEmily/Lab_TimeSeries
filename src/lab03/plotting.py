"""Matplotlib figures; every generated image is written under images/lab03."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_all_figures(series, X_train, X_val, X_test, y_test, predictions,
                     summary_df, cv_detail, champion_name, champion_model,
                     forecast, figure_dir: str | Path):
    """Purpose: create the visual evidence for all pipeline phases.

    Input: source series, split feature frames, test predictions, summary/CV data,
           the selected fitted model, future forecast, and output directory.
    Output: mapping of figure labels to PNG paths. No figure is shown interactively.
    """
    out = Path(figure_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    def save(fig, key, filename):
        target = out / filename
        fig.tight_layout()
        fig.savefig(target, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths[key] = str(target)

    fig, ax = plt.subplots(figsize=(13, 4.5))
    series.plot(ax=ax, linewidth=0.8)
    ax.set_title(f"Hourly target series - {series.name or 'target'}")
    ax.set_xlabel("Time")
    ax.set_ylabel(series.name or "target")
    ax.grid(alpha=0.25)
    save(fig, "series", "01_hourly_series.png")

    fig, ax = plt.subplots(figsize=(13, 3.8))
    for label, frame in (("Train", X_train), ("Validation", X_val), ("Test", X_test)):
        ax.axvspan(frame.index.min(), frame.index.max(), alpha=0.23, label=label)
    ax.plot(series.index, series.values, linewidth=0.55, color="black", alpha=0.75)
    ax.set_title("Chronological Train / Validation / Test split (70/15/15)")
    ax.set_xlabel("Time")
    ax.set_ylabel(series.name or "target")
    ax.legend()
    ax.grid(alpha=0.2)
    save(fig, "split", "02_temporal_split.png")

    n_show = min(24 * 7, len(y_test))
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.plot(y_test.index[:n_show], y_test.values[:n_show], label="Actual", linewidth=1.8)
    for name, pred in predictions.items():
        ax.plot(y_test.index[:n_show], np.asarray(pred)[:n_show], label=name,
                linewidth=1.0, alpha=0.88)
    ax.set_title("Rolling one-step test predictions - first 7 days")
    ax.set_xlabel("Time")
    ax.set_ylabel(series.name or "target")
    ax.legend(fontsize=7.5, ncol=2)
    ax.grid(alpha=0.25)
    save(fig, "forecast_test", "03_test_forecast.png")

    if champion_name in predictions:
        residual = y_test.to_numpy(dtype=float) - np.asarray(predictions[champion_name])
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        axes[0].plot(y_test.index, residual, linewidth=0.55)
        axes[0].axhline(0, linewidth=1, color="black")
        axes[0].set_title(f"Residual over time - {champion_name}")
        axes[0].set_xlabel("Time")
        axes[1].hist(residual, bins=45)
        axes[1].set_title("Residual distribution")
        axes[1].set_xlabel("Actual - prediction")
        save(fig, "residuals", "04_residual_diagnostics.png")

    if not summary_df.empty and "test_RMSE" in summary_df.columns:
        ordered = summary_df.sort_values("test_RMSE")
        fig, ax = plt.subplots(figsize=(11, 4.8))
        ax.bar(ordered["model"], ordered["test_RMSE"])
        ax.set_title("Model comparison on held-out Test (RMSE)")
        ax.set_ylabel("RMSE (lower is better)")
        ax.tick_params(axis="x", labelrotation=16)
        ax.grid(axis="y", alpha=0.25)
        save(fig, "model_comparison", "05_test_model_comparison.png")

    if not cv_detail.empty:
        pivot = cv_detail.pivot(index="fold", columns="model", values="RMSE")
        fig, ax = plt.subplots(figsize=(10.5, 4.8))
        pivot.plot(ax=ax, marker="o")
        ax.set_title("Expanding-window Walk-Forward CV - RMSE by fold")
        ax.set_xlabel("Fold")
        ax.set_ylabel("RMSE")
        ax.grid(alpha=0.25)
        save(fig, "cv", "06_walk_forward_cv.png")

    importance = getattr(champion_model, "feature_importances_", None)
    if importance is not None:
        values = np.asarray(importance, dtype=float)
        title = f"Split-gain feature importance - {champion_name}"
        xlabel = "Normalized split gain"
    else:
        coef = getattr(champion_model, "coef_", None)
        values = np.abs(np.asarray(coef, dtype=float)) if coef is not None else None
        title = f"Absolute linear coefficients - {champion_name}"
        xlabel = "Absolute coefficient (original feature units; not comparable across scales)"
    if values is not None and len(values) == len(X_train.columns):
        order = np.argsort(values)[-min(18, len(values)):]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(np.asarray(X_train.columns)[order], values[order])
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        save(fig, "importance", "07_feature_importance.png")

    if forecast is not None and len(forecast):
        tail = series.iloc[-min(24 * 7, len(series)):]
        fig, ax = plt.subplots(figsize=(12.5, 4.5))
        ax.plot(tail.index, tail.values, label="Observed history", linewidth=1.5)
        ax.plot(forecast.index, forecast.values, label="Recursive forecast", marker="o", linewidth=1.2)
        ax.set_title(f"Recursive future forecast - horizon={len(forecast)} hourly steps")
        ax.set_xlabel("Time")
        ax.set_ylabel(series.name or "target")
        ax.legend()
        ax.grid(alpha=0.25)
        save(fig, "future_forecast", "08_recursive_future_forecast.png")

    return paths
