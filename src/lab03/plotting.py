"""Matplotlib figures; every generated image is written under images/lab03."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_MODEL_LABELS = {
    "Naive Last-Value": "Giá trị gần nhất",
    "Linear Regression": "Hồi quy tuyến tính",
    "CART": "Cây hồi quy CART",
    "Random Forest": "Rừng ngẫu nhiên",
    "XGBoost-style": "Mô hình tăng cường kiểu XGBoost",
}


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
    ax.set_title(f"Chuỗi mục tiêu theo giờ - {series.name or 'mục tiêu'}")
    ax.set_xlabel("Thời gian")
    ax.set_ylabel(series.name or "Mục tiêu")
    ax.grid(alpha=0.25)
    save(fig, "series", "01_hourly_series.png")

    fig, ax = plt.subplots(figsize=(13, 3.8))
    for label, frame in (("Huấn luyện", X_train), ("Xác thực", X_val), ("Kiểm tra", X_test)):
        ax.axvspan(frame.index.min(), frame.index.max(), alpha=0.23, label=label)
    ax.plot(series.index, series.values, linewidth=0.55, color="black", alpha=0.75)
    ax.set_title("Chia Huấn luyện / Xác thực / Kiểm tra theo thời gian (70/15/15)")
    ax.set_xlabel("Thời gian")
    ax.set_ylabel(series.name or "Mục tiêu")
    ax.legend()
    ax.grid(alpha=0.2)
    save(fig, "split", "02_temporal_split.png")

    n_show = min(24 * 7, len(y_test))
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.plot(y_test.index[:n_show], y_test.values[:n_show], label="Thực tế", linewidth=1.8)
    for name, pred in predictions.items():
        ax.plot(y_test.index[:n_show], np.asarray(pred)[:n_show], label=_MODEL_LABELS.get(name, name),
                linewidth=1.0, alpha=0.88)
    ax.set_title("Dự báo cuốn chiếu trước một bước trên tập Kiểm tra - 7 ngày đầu")
    ax.set_xlabel("Thời gian")
    ax.set_ylabel(series.name or "Mục tiêu")
    ax.legend(fontsize=7.5, ncol=2)
    ax.grid(alpha=0.25)
    save(fig, "forecast_test", "03_test_forecast.png")

    if champion_name in predictions:
        residual = y_test.to_numpy(dtype=float) - np.asarray(predictions[champion_name])
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        axes[0].plot(y_test.index, residual, linewidth=0.55)
        axes[0].axhline(0, linewidth=1, color="black")
        axes[0].set_title(f"Phần dư theo thời gian - {_MODEL_LABELS.get(champion_name, champion_name)}")
        axes[0].set_xlabel("Thời gian")
        axes[1].hist(residual, bins=45)
        axes[1].set_title("Phân bố phần dư")
        axes[1].set_xlabel("Giá trị thực tế - dự báo")
        save(fig, "residuals", "04_residual_diagnostics.png")

    if not summary_df.empty and "test_RMSE" in summary_df.columns:
        ordered = summary_df.sort_values("test_RMSE")
        fig, ax = plt.subplots(figsize=(11, 4.8))
        ax.bar(ordered["model"].map(lambda name: _MODEL_LABELS.get(name, name)), ordered["test_RMSE"])
        ax.set_title("So sánh mô hình trên tập Kiểm tra (RMSE)")
        ax.set_ylabel("RMSE (thấp hơn là tốt hơn)")
        ax.tick_params(axis="x", labelrotation=16)
        ax.grid(axis="y", alpha=0.25)
        save(fig, "model_comparison", "05_test_model_comparison.png")

    if not cv_detail.empty:
        pivot = cv_detail.pivot(index="fold", columns="model", values="RMSE")
        pivot.columns = [_MODEL_LABELS.get(name, name) for name in pivot.columns]
        fig, ax = plt.subplots(figsize=(10.5, 4.8))
        pivot.plot(ax=ax, marker="o")
        ax.set_title("RMSE theo fold - kiểm định chéo cửa sổ mở rộng")
        ax.set_xlabel("Fold")
        ax.set_ylabel("RMSE")
        ax.grid(alpha=0.25)
        save(fig, "cv", "06_walk_forward_cv.png")

    importance = getattr(champion_model, "feature_importances_", None)
    if importance is not None:
        values = np.asarray(importance, dtype=float)
        title = f"Mức quan trọng đặc trưng theo độ lợi phép chia - {_MODEL_LABELS.get(champion_name, champion_name)}"
        xlabel = "Độ lợi phép chia đã chuẩn hóa"
    else:
        coef = getattr(champion_model, "coef_", None)
        values = np.abs(np.asarray(coef, dtype=float)) if coef is not None else None
        title = f"Trị tuyệt đối hệ số tuyến tính - {_MODEL_LABELS.get(champion_name, champion_name)}"
        xlabel = "Trị tuyệt đối hệ số (đơn vị đặc trưng gốc; không so sánh được giữa các thang đo)"
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
        ax.plot(tail.index, tail.values, label="Lịch sử quan sát", linewidth=1.5)
        ax.plot(forecast.index, forecast.values, label="Dự báo đệ quy", marker="o", linewidth=1.2)
        ax.set_title(f"Dự báo tương lai đệ quy - {len(forecast)} bước theo giờ")
        ax.set_xlabel("Thời gian")
        ax.set_ylabel(series.name or "Mục tiêu")
        ax.legend()
        ax.grid(alpha=0.25)
        save(fig, "future_forecast", "08_recursive_future_forecast.png")

    if champion_name in predictions:
        actual = np.asarray(y_test, dtype=float)
        pred = np.asarray(predictions[champion_name], dtype=float)
        residual = actual - pred

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        axes[0].scatter(actual, pred, s=8, alpha=0.28, edgecolors="none")
        limits = [min(actual.min(), pred.min()), max(actual.max(), pred.max())]
        axes[0].plot(limits, limits, color="tab:red", linestyle="--", linewidth=1.5,
                     label="Dự báo = thực tế")
        champion_label = _MODEL_LABELS.get(champion_name, champion_name)
        axes[0].set_title(f"Giá trị thực tế và dự báo - {champion_label}")
        axes[0].set_xlabel("Giá trị thực tế")
        axes[0].set_ylabel("Giá trị dự báo")
        axes[0].legend()

        axes[1].hist(residual, bins=50, color="tab:blue", alpha=0.82,
                     edgecolor="white", linewidth=0.25)
        axes[1].axvline(0, color="black", linewidth=1.0)
        axes[1].axvline(residual.mean(), color="tab:red", linestyle="--", linewidth=1.5,
                        label=f"Trung bình: {residual.mean():.1f}")
        axes[1].set_title("Phân bố phần dư")
        axes[1].set_xlabel("Giá trị thực tế - dự báo")
        axes[1].legend()
        for ax in axes:
            ax.grid(alpha=0.25)
        save(fig, "actual_vs_prediction", "09_actual_vs_prediction_residuals.png")

        diagnostic = pd.DataFrame({
            "actual": actual,
            "prediction": pred,
            "absolute_error": np.abs(residual),
        }, index=y_test.index)
        diagnostic["weekday"] = diagnostic.index.dayofweek
        diagnostic["hour"] = diagnostic.index.hour
        mae_matrix = diagnostic.pivot_table(
            index="weekday", columns="hour", values="absolute_error", aggfunc="mean"
        ).reindex(index=range(7), columns=range(24))
        fig, ax = plt.subplots(figsize=(13, 5))
        im = ax.imshow(mae_matrix.to_numpy(), aspect="auto", cmap="YlOrRd", origin="upper")
        ax.set_title(f"MAE trên tập Kiểm tra theo thứ và giờ - {champion_label}")
        ax.set_xlabel("Giờ")
        ax.set_ylabel("Thứ trong tuần")
        ax.set_xticks(range(24), range(24))
        ax.set_yticks(range(7), ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ nhật"])
        fig.colorbar(im, ax=ax, label="MAE")
        save(fig, "test_mae_weekday_hour", "10_test_mae_weekday_hour.png")

    return paths
