"""
Module trực quan hóa dữ liệu và kết quả thực nghiệm (`src/lab04/visualize.py`) cho dự án Lab 04 Time Series.

Trách nhiệm của module:
1. Sinh trọn bộ 16+ biểu đồ học thuật ở độ phân giải 300 DPI phục vụ Yêu cầu 1–12:
   - Nhóm EDA (Mục 3):
     (1) `load_timeseries_overview.png`     : Biểu đồ phụ tải điện theo thời gian & lát cắt 14 ngày
     (2) `load_vs_temperature_scatter.png`  : Biểu đồ tương quan phi tuyến Load vs Temperature & Lag Cross-Correlation
     (3) `correlation_heatmap.png`          : Ma trận tương quan Pearson 4 biến
     (4) `seasonality_daily_weekly.png`     : Hồ sơ mùa vụ chu kỳ ngày (24h) và chu kỳ tuần (7 ngày)
   - Nhóm Preprocessing & K-Fold (Mục 4 & Yêu cầu 4):
     (5) `train_val_test_split.png`         : Phân vùng thời gian Chronological Split (70% / 15% / 15%)
     (6) `kfold_splits_timeline.png`        : Sơ đồ 5-Fold Expanding Window (`TimeSeriesSplit`)
   - Nhóm Training (Mục 6, 7, 8):
     (7) `lstm_train_vs_val_loss.png`       : Đường cong hội tụ Train vs Validation MSE & MAE của LSTM (kiểm tra Overfitting)
     (8) `xgboost_feature_importance.png`   : Top 15 đặc trưng quan trọng nhất của XGBoost
     (9) `training_time_comparison.png`     : So sánh thời gian huấn luyện giữa các mô hình
   - Nhóm Evaluation, Walk-Forward & PEMS-BAY (Mục 9, 10, 12 & Yêu cầu 6):
     (10) `actual_vs_predicted_all.png`     : Đường thực tế vs dự báo của VAR, XGBoost, LSTM trên tập Test
     (11) `walk_forward_backtest.png`       : Kết quả dự báo cuốn chiếu Walk-Forward Backtesting (168 giờ)
     (12) `model_comparison_mse.png`        : Biểu đồ cột so sánh MSE
     (13) `model_comparison_rmse.png`       : Biểu đồ cột so sánh RMSE
     (14) `model_comparison_mae.png`        : Biểu đồ cột so sánh MAE
     (15) `model_comparison_mape.png`       : Biểu đồ cột so sánh MAPE (%)
     (16) `model_comparison_r2.png`         : Biểu đồ cột so sánh R² Score
     (17) `residual_plot_all.png` & `residual_distribution_all.png`: Phân tích phần dư
     (18) `kfold_cv_comparison.png`         : Độ ổn định RMSE & R² qua 5 Folds (`Mean ± Std`)
     (19) `pems_bay_horizon_comparison.png` : Đánh giá DCRNN trên PEMS-BAY ở 3 chân trời 15, 30, 60 phút
2. Lưu đồng thời tại `reports/lab04/figures/<filename>.png` và `reports/lab04/figures/<run_id>/<stage>/<filename>.png`.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.lab04.utils import ensure_dir, get_project_root, load_config

ROOT_DIR = get_project_root()
_CFG = load_config()
FIGURES_DIR = ROOT_DIR / _CFG.get("paths", {}).get("figures_dir", "reports/lab04/figures")

sns.set_theme(style="whitegrid", palette="deep")
MODEL_COLORS = {
    "VAR": "#dc2626",
    "XGBoost": "#059669",
    "LSTM": "#1d4ed8",
    "DCRNN": "#7c3aed",
}


def create_stage_figure_dir(run_id: str, stage: str) -> Path:
    """
    What: Khởi tạo thư mục lưu biểu đồ theo từng `run_id` và từng giai đoạn trong `reports/lab04/figures/`.
    Why: Đảm bảo lưu vết lịch sử hình ảnh của mọi phiên chạy mà không bị mất khi chạy lần mới.
    Input:
        - run_id (str): Mã định danh phiên chạy.
        - stage (str): Tên giai đoạn (`eda`, `preprocessing`, `training`, `evaluation`).
    Output:
        - Path: Đường dẫn thư mục đã tạo.
    How: Ghép `FIGURES_DIR / run_id / stage.lower()` và gọi `ensure_dir()`.
    """
    return ensure_dir(FIGURES_DIR / str(run_id) / stage.lower())


def _save_fig(fig: plt.Figure, filename: str, stage_dir: Optional[Path] = None, dpi: int = 300) -> List[str]:
    """
    What: Lưu đối tượng `Figure` vào `reports/lab04/figures/<filename>` và sao chép vào thư mục `run_id`.
    Why: Đảm bảo cả báo cáo tổng hợp lẫn báo cáo theo `Run_ID` đều truy cập được ảnh nét 300 DPI.
    Input:
        - fig (plt.Figure): Hình vẽ Matplotlib.
        - filename (str): Tên tệp `.png`.
        - stage_dir (Optional[Path]): Thư mục giai đoạn của `run_id`.
        - dpi (int): Độ phân giải (`300`).
    Output:
        - List[str]: Danh sách đường dẫn tương đối của tệp ảnh đã lưu.
    How: Gọi `fig.savefig(..., dpi=dpi, bbox_inches='tight')` và `shutil.copy2`.
    """
    ensure_dir(FIGURES_DIR)
    primary = FIGURES_DIR / filename
    fig.savefig(primary, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    saved = [str(primary.relative_to(ROOT_DIR)).replace("\\", "/")]
    if stage_dir is not None:
        ensure_dir(stage_dir)
        dst = stage_dir / filename
        shutil.copy2(primary, dst)
        saved.append(str(dst.relative_to(ROOT_DIR)).replace("\\", "/"))
    return saved


def plot_load_timeseries_overview(df: pd.DataFrame, stage_dir: Optional[Path] = None) -> List[str]:
    """
    What: Vẽ biểu đồ diễn biến phụ tải điện `load` theo thời gian (toàn chuỗi và lát cắt 14 ngày).
    Why: Thực hiện yêu cầu trực quan hóa thứ nhất của Mục 3 (`Biểu đồ load theo thời gian`).
    Input:
        - df (pd.DataFrame): Bảng dữ liệu 4 biến.
        - stage_dir (Optional[Path]): Thư mục lưu theo `run_id`.
    Output:
        - List[str]: Đường dẫn tệp `load_timeseries_overview.png`.
    How: Vẽ 2 subplot: subplot trên hiển thị toàn chuỗi kèm đường trung bình trượt 7 ngày;
         subplot dưới hiển thị chi tiết 336 giờ (14 ngày) để thấy rõ chu kỳ ngày/tuần.
    """
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5))
    axes[0].plot(df.index, df["load"], color="#1d4ed8", linewidth=0.8, alpha=0.75, label="Hourly Load (kW)")
    axes[0].plot(df.index, df["load"].rolling(168, min_periods=1).mean(), color="#dc2626", linewidth=2.0, label="7-Day Rolling Mean (168h)")
    axes[0].set_title("Diễn biến Phụ tải Điện năng (Electricity Load) trên Toàn Chuỗi Thời gian", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Load (kW)")
    axes[0].legend(loc="upper right")

    zoom_df = df.iloc[-min(336, len(df)):]
    axes[1].plot(zoom_df.index, zoom_df["load"], color="#059669", linewidth=1.5, label="Hourly Load (Zoom 14 Days)")
    axes[1].set_title("Lát cắt Chi tiết 14 Ngày Gần nhất (336 Giờ) — Thể hiện Rõ Chu kỳ Ngày (24h) & Tuần (168h)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Thời gian (Datetime)")
    axes[1].set_ylabel("Load (kW)")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    return _save_fig(fig, "load_timeseries_overview.png", stage_dir=stage_dir)


def plot_load_vs_temperature(
    df: pd.DataFrame,
    lag_corrs: Dict[str, Any],
    stage_dir: Optional[Path] = None,
) -> List[str]:
    """
    What: Vẽ biểu đồ phân tán `load` vs `temperature` và đường cong tương quan trễ (`Lag Cross-Correlation`).
    Why: Thực hiện yêu cầu trực quan hóa thứ hai của Mục 3 (`Biểu đồ load vs temperature`) và minh họa
         hiện tượng `Lag Heterogeneity` trong Slide 4 Chương 4–5.
    Input:
        - df (pd.DataFrame): Bảng dữ liệu 4 biến.
        - lag_corrs (Dict[str, Any]): Tương quan theo độ trễ từ `eda.py`.
        - stage_dir (Optional[Path]): Thư mục lưu theo `run_id`.
    Output:
        - List[str]: Đường dẫn tệp `load_vs_temperature_scatter.png`.
    How: Subplot trái vẽ scatter `temperature` vs `load` kèm đường hồi quy đa thức bậc 2;
         subplot phải vẽ tương quan $r(\tau)$ theo độ trễ $\tau = 0..24\text{h}$.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    sample_df = df.iloc[-min(2000, len(df)):]
    sc = axes[0].scatter(
        sample_df["temperature"],
        sample_df["load"],
        c=sample_df["humidity"],
        cmap="viridis",
        alpha=0.55,
        s=18,
    )
    z = np.polyfit(sample_df["temperature"].to_numpy(), sample_df["load"].to_numpy(), 2)
    p = np.poly1d(z)
    xs = np.linspace(float(sample_df["temperature"].min()), float(sample_df["temperature"].max()), 100)
    axes[0].plot(xs, p(xs), color="#dc2626", linewidth=2.4, label="Quadratic Nonlinear Trend")
    plt.colorbar(sc, ax=axes[0], label="Humidity (%)")
    axes[0].set_title("Mối quan hệ Phi tuyến giữa Nhiệt độ (Temperature) và Phụ tải Điện (Load)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Temperature (°C)")
    axes[0].set_ylabel("Electricity Load (kW)")
    axes[0].legend(loc="upper left")

    lags = list(range(0, 25))
    for w_col, color in [("temperature", "#dc2626"), ("humidity", "#1d4ed8"), ("wind_speed", "#d97706")]:
        corrs = [float(df["load"].corr(df[w_col].shift(l))) for l in lags]
        axes[1].plot(lags, corrs, marker="o", markersize=3.5, linewidth=1.8, color=color, label=f"load vs {w_col}")
    axes[1].axhline(0.0, color="#6b7280", linestyle="--", linewidth=1.0)
    axes[1].set_title("Tương quan Chéo theo Độ trễ (Lag Heterogeneity: τ = 0..24 giờ)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Độ trễ τ (Giờ)")
    axes[1].set_ylabel("Hệ số Tương quan Pearson r(τ)")
    axes[1].legend(loc="best")

    fig.tight_layout()
    return _save_fig(fig, "load_vs_temperature_scatter.png", stage_dir=stage_dir)


def plot_correlation_heatmap(corr_df: pd.DataFrame, stage_dir: Optional[Path] = None) -> List[str]:
    """
    What: Vẽ biểu đồ nhiệt ma trận tương quan (`Correlation Matrix`) giữa 4 biến.
    Why: Thực hiện yêu cầu trực quan hóa thứ ba của Mục 3 (`Ma trận tương quan`).
    Input:
        - corr_df (pd.DataFrame): Ma trận tương quan $4 \times 4$.
        - stage_dir (Optional[Path]): Thư mục lưu theo `run_id`.
    Output:
        - List[str]: Đường dẫn tệp `correlation_heatmap.png`.
    How: Dùng `seaborn.heatmap(..., annot=True, fmt='.3f', cmap='coolwarm')`.
    """
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    sns.heatmap(
        corr_df,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        linewidths=1.0,
        square=True,
        ax=ax,
    )
    ax.set_title("Ma trận Tương quan Pearson giữa Phụ tải Điện năng và Thời tiết", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _save_fig(fig, "correlation_heatmap.png", stage_dir=stage_dir)


def plot_seasonality_profiles(df: pd.DataFrame, stage_dir: Optional[Path] = None) -> List[str]:
    """
    What: Vẽ biểu đồ phân tích chu kỳ ngày (24h) và chu kỳ tuần (7 ngày trong tuần).
    Why: Minh họa trực quan kết quả kiểm tra Seasonality ở Mục 3 và trả lời Câu hỏi 4 Mục 11.
    Input:
        - df (pd.DataFrame): Bảng dữ liệu 4 biến.
        - stage_dir (Optional[Path]): Thư mục lưu theo `run_id`.
    Output:
        - List[str]: Đường dẫn tệp `seasonality_daily_weekly.png`.
    How: Subplot trái vẽ trung bình và dải độ lệch chuẩn phụ tải theo 24 giờ trong ngày;
         subplot phải vẽ phụ tải trung bình theo 7 ngày trong tuần (Thứ Hai -> Chủ Nhật).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    hourly_grp = df.groupby(df.index.hour)["load"].agg(["mean", "std"])
    hrs = hourly_grp.index.to_numpy()
    axes[0].plot(hrs, hourly_grp["mean"], marker="o", color="#1d4ed8", linewidth=2.2, label="Mean Load by Hour")
    axes[0].fill_between(
        hrs,
        hourly_grp["mean"] - hourly_grp["std"],
        hourly_grp["mean"] + hourly_grp["std"],
        color="#93c5fd",
        alpha=0.35,
        label="±1 Std Band",
    )
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].set_title("Chu kỳ Ngày (Daily Seasonality — 24 Giờ)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Giờ trong Ngày (0h – 23h)")
    axes[0].set_ylabel("Electricity Load (kW)")
    axes[0].legend(loc="upper left")

    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_grp = df.groupby(df.index.dayofweek)["load"].mean()
    colors = ["#2563eb" if i < 5 else "#d97706" for i in range(7)]
    bars = axes[1].bar(dow_names, dow_grp.to_numpy(), color=colors, edgecolor="black", alpha=0.85)
    for b in bars:
        axes[1].text(b.get_x() + b.get_width() / 2.0, b.get_height() + 8.0, f"{b.get_height():.0f}", ha="center", va="bottom", fontsize=9)
    axes[1].set_title("Chu kỳ Tuần (Weekly Seasonality — Ngày thường vs. Cuối tuần)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Ngày trong Tuần")
    axes[1].set_ylabel("Mean Electricity Load (kW)")

    fig.tight_layout()
    return _save_fig(fig, "seasonality_daily_weekly.png", stage_dir=stage_dir)


def plot_train_val_test_split(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    stage_dir: Optional[Path] = None,
) -> List[str]:
    """
    What: Vẽ biểu đồ phân chia chuỗi thời gian theo thứ tự `Train (70%) - Validation (15%) - Test (15%)`.
    Why: Minh chứng trực quan cho Mục 4.2 (chia theo thứ tự thời gian, không shuffle).
    Input:
        - df_train, df_val, df_test (pd.DataFrame): 3 tập dữ liệu đã chia.
        - stage_dir (Optional[Path]): Thư mục lưu theo `run_id`.
    Output:
        - List[str]: Đường dẫn tệp `train_val_test_split.png`.
    How: Vẽ 3 đoạn chuỗi thời gian liên tiếp bằng 3 màu phân biệt kèm vạch phân cách.
    """
    fig, ax = plt.subplots(figsize=(13.0, 4.6))
    ax.plot(df_train.index, df_train["load"], color="#1d4ed8", linewidth=0.9, label=f"Train Set (70% — {len(df_train)}h)")
    ax.plot(df_val.index, df_val["load"], color="#d97706", linewidth=0.9, label=f"Validation Set (15% — {len(df_val)}h)")
    ax.plot(df_test.index, df_test["load"], color="#059669", linewidth=0.9, label=f"Test Set (15% — {len(df_test)}h)")
    ax.axvline(df_val.index.min(), color="#111827", linestyle="--", linewidth=1.4)
    ax.axvline(df_test.index.min(), color="#111827", linestyle="--", linewidth=1.4)
    ax.set_title("Phân chia Tập Dữ liệu Chuỗi Thời gian: Train (70%) – Validation (15%) – Test (15%) (No Shuffle)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Thời gian (Datetime)")
    ax.set_ylabel("Electricity Load (kW)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return _save_fig(fig, "train_val_test_split.png", stage_dir=stage_dir)


def plot_kfold_splits_timeline(
    df_clean: pd.DataFrame,
    kfold_splits: List[Dict[str, Any]],
    stage_dir: Optional[Path] = None,
) -> List[str]:
    """
    What: Vẽ sơ đồ cấu trúc 5-Fold Expanding Window (`TimeSeriesSplit`) trên trục thời gian.
    Why: Minh họa trực quan cho Yêu cầu bổ sung số 4 (áp dụng K-Fold bên cạnh Train-Val-Test).
    Input:
        - df_clean (pd.DataFrame): Chuỗi thời gian gốc.
        - kfold_splits (List[Dict[str, Any]]): Thông tin 5 Folds.
        - stage_dir (Optional[Path]): Thư mục lưu theo `run_id`.
    Output:
        - List[str]: Đường dẫn tệp `kfold_splits_timeline.png`.
    How: Vẽ các thanh ngang biểu diễn phạm vi `Train_k` (xanh) và `Val_k` (cam) của từng Fold $k=1..5$.
    """
    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    for pkg in kfold_splits:
        k = int(pkg["fold"])
        n_tr = int(pkg["train_size"])
        n_va = int(pkg["val_size"])
        ax.barh(y=f"Fold {k}", width=n_tr, left=0, height=0.55, color="#2563eb", edgecolor="black", label="In-Fold Train (Fit Scaler)" if k == 1 else "")
        ax.barh(y=f"Fold {k}", width=n_va, left=n_tr, height=0.55, color="#f59e0b", edgecolor="black", label="In-Fold Validation" if k == 1 else "")
    ax.set_title("Cấu trúc Kiểm định Chéo 5-Fold Theo Thời gian (Expanding Window TimeSeriesSplit)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Chỉ số Bước Thời gian (Giờ)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _save_fig(fig, "kfold_splits_timeline.png", stage_dir=stage_dir)


def plot_lstm_training_curves(
    df_hist: pd.DataFrame,
    best_epoch: int,
    stage_dir: Optional[Path] = None,
) -> List[str]:
    """
    What: Vẽ đường cong hội tụ `Train vs. Validation MSE` và `MAE` của mạng `LSTM(64)->LSTM(32)->Dense(1)`.
    Why: Minh chứng trực quan cho Câu hỏi 5 ở Mục 11 (*"LSTM có bị overfitting không?"*).
    Input:
        - df_hist (pd.DataFrame): Bảng lịch sử huấn luyện LSTM qua các epochs.
        - best_epoch (int): Epoch đạt `val_mse` thấp nhất.
        - stage_dir (Optional[Path]): Thư mục lưu theo `run_id`.
    Output:
        - List[str]: Đường dẫn `lstm_train_vs_val_loss.png` và `lstm_training_loss_curve.png`.
    How: Vẽ 2 subplot đối chiếu `train_mse` vs `val_mse` và `train_mae` vs `val_mae` kèm vạch `Best Epoch`.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    ep = df_hist["epoch"]
    axes[0].plot(ep, df_hist["train_mse"], color="#1d4ed8", linewidth=2.0, label="Train MSE Loss")
    axes[0].plot(ep, df_hist["val_mse"], color="#dc2626", linewidth=2.0, label="Validation MSE Loss")
    axes[0].axvline(best_epoch, color="#059669", linestyle="--", linewidth=1.6, label=f"Best Epoch = {best_epoch}")
    axes[0].set_title("Đường cong Hội tụ Hàm mất mát MSE của LSTM (Scaled Space)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].legend(loc="upper right")

    axes[1].plot(ep, df_hist["train_mae"], color="#2563eb", linewidth=2.0, label="Train MAE")
    axes[1].plot(ep, df_hist["val_mae"], color="#d97706", linewidth=2.0, label="Validation MAE")
    axes[1].axvline(best_epoch, color="#059669", linestyle="--", linewidth=1.6, label=f"Best Epoch = {best_epoch}")
    axes[1].set_title("Đường cong Hội tụ MAE của LSTM & Điểm Dừng Sớm (EarlyStopping)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE (Scaled)")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    saved = _save_fig(fig, "lstm_train_vs_val_loss.png", stage_dir=stage_dir)
    # Lưu thêm tên tương thích với cấu trúc báo cáo
    shutil.copy2(FIGURES_DIR / "lstm_train_vs_val_loss.png", FIGURES_DIR / "lstm_training_loss_curve.png")
    return saved


def plot_xgboost_feature_importance(df_imp: pd.DataFrame, stage_dir: Optional[Path] = None) -> List[str]:
    """
    What: Vẽ biểu đồ Top 15 đặc trưng có độ quan trọng (`Importance`) cao nhất trong `XGBoost`.
    Why: Minh chứng định lượng cho Câu hỏi 1 & Câu hỏi 3 ở Mục 11 đề bài.
    Input:
        - df_imp (pd.DataFrame): Bảng `Feature Importance` từ `XGBoostForecaster`.
        - stage_dir (Optional[Path]): Thư mục lưu theo `run_id`.
    Output:
        - List[str]: Đường dẫn tệp `xgboost_feature_importance.png`.
    How: Vẽ biểu đồ cột ngang cho Top 15 đặc trưng đứng đầu.
    """
    top_df = df_imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    ax.barh(top_df["Feature"], top_df["Importance"], color="#059669", edgecolor="black", alpha=0.88)
    ax.set_title("Top 15 Đặc trưng Quan trọng nhất của Mô hình XGBoost (Lag 24h + Rolling + Weather)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Relative Feature Importance (Gain)")
    fig.tight_layout()
    return _save_fig(fig, "xgboost_feature_importance.png", stage_dir=stage_dir)


def plot_training_time_comparison(time_map: Dict[str, float], stage_dir: Optional[Path] = None) -> List[str]:
    """
    What: Vẽ biểu đồ cột so sánh thời gian huấn luyện (giây) giữa các mô hình.
    Why: Đánh giá chi phí tính toán và hiệu quả của cơ chế `Auto-GPU`.
    Input:
        - time_map (Dict[str, float]): Thời gian huấn luyện của từng mô hình.
        - stage_dir (Optional[Path]): Thư mục lưu theo `run_id`.
    Output:
        - List[str]: Đường dẫn tệp `training_time_comparison.png`.
    How: Vẽ biểu đồ cột kèm nhãn số giây trên đỉnh từng cột.
    """
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    names = list(time_map.keys())
    vals = [float(time_map[k]) for k in names]
    colors = [MODEL_COLORS.get(k.split()[0], "#4b5563") for k in names]
    bars = ax.bar(names, vals, color=colors, edgecolor="black", width=0.52, alpha=0.88)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2.0, b.get_height() + max(vals) * 0.02, f"{b.get_height():.2f}s", ha="center", va="bottom", fontweight="bold")
    ax.set_title("So sánh Thời gian Huấn luyện giữa các Mô hình (Auto-GPU Enabled)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Thời gian Huấn luyện (Giây)")
    ax.set_ylim(0, max(vals) * 1.18 if max(vals) > 0 else 1.0)
    fig.tight_layout()
    return _save_fig(fig, "training_time_comparison.png", stage_dir=stage_dir)


def plot_actual_vs_predicted_all(df_preds: pd.DataFrame, stage_dir: Optional[Path] = None) -> List[str]:
    """
    What: Vẽ biểu đồ so sánh giá trị phụ tải thực tế (`Actual_Load`) và dự báo của `VAR`, `XGBoost`, `LSTM`.
    Why: Đánh giá trực quan độ bám sát đỉnh/đáy phụ tải trên tập Test độc lập.
    Input:
        - df_preds (pd.DataFrame): Bảng dự đoán trên tập Test.
        - stage_dir (Optional[Path]): Thư mục lưu theo `run_id`.
    Output:
        - List[str]: Đường dẫn tệp `actual_vs_predicted_all.png`.
    How: Hiển thị lát cắt 168 giờ (7 ngày) trên tập Test kèm biểu đồ phân tán `Actual vs Predicted`.
    """
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8.0))
    sub = df_preds.iloc[: min(168, len(df_preds))]
    axes[0].plot(sub["datetime"], sub["Actual_Load"], color="#111827", linewidth=2.2, label="Actual Load (Ground Truth)")
    axes[0].plot(sub["datetime"], sub["VAR_Pred"], color=MODEL_COLORS["VAR"], linestyle="--", linewidth=1.4, label="VAR Forecast")
    axes[0].plot(sub["datetime"], sub["XGBoost_Pred"], color=MODEL_COLORS["XGBoost"], linewidth=1.6, label="XGBoost Forecast")
    axes[0].plot(sub["datetime"], sub["LSTM_Pred"], color=MODEL_COLORS["LSTM"], linewidth=1.6, label="LSTM Forecast")
    axes[0].set_title("Đối chiếu Phụ tải Thực tế và Dự báo của 3 Mô hình (VAR, XGBoost, LSTM — 168 Giờ Tập Test)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Electricity Load (kW)")
    axes[0].legend(loc="upper right", ncol=2)

    for m_name, col in [("VAR", "VAR_Pred"), ("XGBoost", "XGBoost_Pred"), ("LSTM", "LSTM_Pred")]:
        axes[1].scatter(df_preds["Actual_Load"], df_preds[col], alpha=0.4, s=16, color=MODEL_COLORS[m_name], label=m_name)
    mn = float(df_preds["Actual_Load"].min())
    mx = float(df_preds["Actual_Load"].max())
    axes[1].plot([mn, mx], [mn, mx], color="#111827", linestyle="--", linewidth=1.8, label="Ideal 1:1 Line")
    axes[1].set_title("Biểu đồ Phân tán Giá trị Thực tế vs. Dự đoán trên Toàn bộ Tập Test (15%)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Actual Electricity Load (kW)")
    axes[1].set_ylabel("Predicted Electricity Load (kW)")
    axes[1].legend(loc="upper left")

    fig.tight_layout()
    return _save_fig(fig, "actual_vs_predicted_all.png", stage_dir=stage_dir)


def plot_walk_forward_backtest(
    wf_df: pd.DataFrame,
    wf_metrics: Dict[str, Dict[str, float]],
    stage_dir: Optional[Path] = None,
) -> List[str]:
    """
    What: Vẽ biểu đồ kết quả dự báo cuốn chiếu `Walk-Forward Validation / Backtesting` (Mục 9).
    Why: Minh họa trực quan quá trình `Train -> predict t+1 -> thêm vào chuỗi -> predict t+2`.
    Input:
        - wf_df (pd.DataFrame): Chuỗi dự đoán Walk-Forward.
        - wf_metrics (Dict[str, Dict[str, float]]): Chỉ số đánh giá trên cửa sổ Walk-Forward.
        - stage_dir (Optional[Path]): Thư mục lưu theo `run_id`.
    Output:
        - List[str]: Đường dẫn tệp `walk_forward_backtest.png`.
    How: Vẽ đường thực tế và 3 đường dự báo cuốn chiếu kèm chú thích `RMSE` & `MAPE (%)`.
    """
    fig, ax = plt.subplots(figsize=(13.5, 5.0))
    ax.plot(wf_df["datetime"], wf_df["Actual_Load"], color="#111827", linewidth=2.2, label="Observed Load (Ground Truth)")
    for m_name in ["VAR", "XGBoost", "LSTM"]:
        col = f"{m_name}_WalkForward"
        rmse_v = wf_metrics[m_name]["RMSE"]
        mape_v = wf_metrics[m_name]["MAPE"]
        ax.plot(
            wf_df["datetime"],
            wf_df[col],
            color=MODEL_COLORS[m_name],
            linewidth=1.5,
            linestyle="--" if m_name == "VAR" else "-",
            label=f"{m_name} Walk-Forward (RMSE={rmse_v:.2f}, MAPE={mape_v:.2f}%)",
        )
    ax.set_title("Kết quả Kiểm thử Cuốn chiếu (Walk-Forward Backtesting — Mục 9)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Thời gian (Datetime)")
    ax.set_ylabel("Electricity Load (kW)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return _save_fig(fig, "walk_forward_backtest.png", stage_dir=stage_dir)


def plot_metric_comparison_bars(
    holdout_metrics: Dict[str, Dict[str, Any]],
    stage_dir: Optional[Path] = None,
) -> List[str]:
    """
    What: Sinh 5 biểu đồ cột riêng biệt cho từng chỉ số đánh giá (`MSE`, `RMSE`, `MAE`, `MAPE`, `R²`).
    Why: Đáp ứng Yêu cầu bổ sung số 6 và giữ đúng chuẩn trình bày của `Lab04_ML-main`.
    Input:
        - holdout_metrics (Dict[str, Dict[str, Any]]): Số liệu 5 chỉ số của `VAR`, `XGBoost`, `LSTM`.
        - stage_dir (Optional[Path]): Thư mục lưu theo `run_id`.
    Output:
        - List[str]: Danh sách 5 tệp `model_comparison_<metric>.png`.
    How: Vẽ từng biểu đồ cột có gắn giá trị số học chính xác trên đỉnh cột.
    """
    specs = [
        ("MSE", "So sánh Sai số Bình phương Trung bình (Test MSE ↓)", "MSE (kW²)", "model_comparison_mse.png"),
        ("RMSE", "So sánh Căn bậc hai Sai số Bình phương Trung bình (Test RMSE ↓)", "RMSE (kW)", "model_comparison_rmse.png"),
        ("MAE", "So sánh Sai số Tuyệt đối Trung bình (Test MAE ↓)", "MAE (kW)", "model_comparison_mae.png"),
        ("MAPE", "So sánh Sai số Phần trăm Tuyệt đối Trung bình (Test MAPE % ↓)", "MAPE (%)", "model_comparison_mape.png"),
        ("R2", "So sánh Hệ số Xác định (Test R² Score ↑)", "R² Score", "model_comparison_r2.png"),
    ]
    models = ["VAR", "XGBoost", "LSTM"]
    colors = [MODEL_COLORS[m] for m in models]
    saved_all: List[str] = []

    for key, title, ylabel, fname in specs:
        fig, ax = plt.subplots(figsize=(7.8, 4.6))
        vals = [float(holdout_metrics[m][key]) for m in models]
        bars = ax.bar(models, vals, color=colors, edgecolor="black", width=0.5, alpha=0.9)
        span = max(vals) if max(vals) > 0 else 1.0
        for b, v in zip(bars, vals):
            fmt_str = f"{v:.2f}%" if key == "MAPE" else (f"{v:.4f}" if key == "R2" else f"{v:.2f}")
            ax.text(b.get_x() + b.get_width() / 2.0, b.get_height() + span * 0.02, fmt_str, ha="center", va="bottom", fontweight="bold")
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, span * 1.18)
        fig.tight_layout()
        saved_all.extend(_save_fig(fig, fname, stage_dir=stage_dir))

    return saved_all


def plot_residual_analysis(df_preds: pd.DataFrame, stage_dir: Optional[Path] = None) -> List[str]:
    """
    What: Vẽ biểu đồ phần dư theo thời gian (`residual_plot_all.png`) và phân phối phần dư (`residual_distribution_all.png`).
    Why: Kiểm tra tính không chệch (Mean Residual $\approx 0$) và độ tập trung sai số của 3 mô hình.
    Input:
        - df_preds (pd.DataFrame): Bảng dự đoán và phần dư trên tập Test.
        - stage_dir (Optional[Path]): Thư mục lưu theo `run_id`.
    Output:
        - List[str]: Đường dẫn 2 tệp biểu đồ phần dư.
    How: Vẽ scatter/line phần dư và biểu đồ KDE/Histogram phân phối phần dư.
    """
    saved: List[str] = []
    fig1, ax1 = plt.subplots(figsize=(12.5, 4.6))
    for m_name in ["VAR", "XGBoost", "LSTM"]:
        ax1.plot(df_preds["datetime"], df_preds[f"{m_name}_Residual"], color=MODEL_COLORS[m_name], alpha=0.75, linewidth=1.0, label=f"{m_name} Residual")
    ax1.axhline(0.0, color="#111827", linestyle="--", linewidth=1.5)
    ax1.set_title("Biểu đồ Phần dư Dự báo (Residuals e_t = y_t - y_hat_t) trên Tập Test", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Thời gian (Datetime)")
    ax1.set_ylabel("Residual (kW)")
    ax1.legend(loc="upper right")
    fig1.tight_layout()
    saved.extend(_save_fig(fig1, "residual_plot_all.png", stage_dir=stage_dir))

    fig2, ax2 = plt.subplots(figsize=(10.0, 4.8))
    for m_name in ["VAR", "XGBoost", "LSTM"]:
        sns.kdeplot(df_preds[f"{m_name}_Residual"], ax=ax2, color=MODEL_COLORS[m_name], fill=True, alpha=0.2, linewidth=2.0, label=m_name)
    ax2.axvline(0.0, color="#111827", linestyle="--", linewidth=1.5)
    ax2.set_title("Phân phối Xác suất Phần dư (Residual Distribution) của VAR, XGBoost và LSTM", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Sai số Phần dư e_t (kW)")
    ax2.set_ylabel("Mật độ (Density)")
    ax2.legend(loc="upper right")
    fig2.tight_layout()
    saved.extend(_save_fig(fig2, "residual_distribution_all.png", stage_dir=stage_dir))
    return saved


def plot_kfold_cv_comparison(df_folds: pd.DataFrame, stage_dir: Optional[Path] = None) -> List[str]:
    """
    What: Vẽ biểu đồ so sánh `RMSE` và `R²` qua 5 Folds của `TimeSeriesSplit` cho `VAR`, `XGBoost`, `LSTM`.
    Why: Minh họa kết quả định lượng của Yêu cầu bổ sung số 4 (K-Fold Cross-Validation).
    Input:
        - df_folds (pd.DataFrame): Bảng kết quả từng Fold từ `evaluate.py`.
        - stage_dir (Optional[Path]): Thư mục lưu theo `run_id`.
    Output:
        - List[str]: Đường dẫn tệp `kfold_cv_comparison.png`.
    How: Subplot trái vẽ RMSE theo từng Fold (`Fold 1..5`); subplot phải vẽ R² Score qua 5 Folds.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    for m_name in ["VAR", "XGBoost", "LSTM"]:
        sub = df_folds[df_folds["Model"] == m_name]
        axes[0].plot(sub["Fold"], sub["RMSE"], marker="o", linewidth=2.0, color=MODEL_COLORS[m_name], label=f"{m_name} (Mean={sub['RMSE'].mean():.2f})")
        axes[1].plot(sub["Fold"], sub["R2"], marker="s", linewidth=2.0, color=MODEL_COLORS[m_name], label=f"{m_name} (Mean={sub['R2'].mean():.4f})")

    axes[0].set_xticks([1, 2, 3, 4, 5])
    axes[0].set_title("Độ ổn định RMSE qua 5-Fold TimeSeriesSplit (↓)", fontsize=11.5, fontweight="bold")
    axes[0].set_xlabel("Fold Index (Temporal Expanding Window)")
    axes[0].set_ylabel("Validation RMSE (kW)")
    axes[0].legend(loc="best")

    axes[1].set_xticks([1, 2, 3, 4, 5])
    axes[1].set_title("Độ ổn định R² Score qua 5-Fold TimeSeriesSplit (↑)", fontsize=11.5, fontweight="bold")
    axes[1].set_xlabel("Fold Index (Temporal Expanding Window)")
    axes[1].set_ylabel("Validation R² Score")
    axes[1].legend(loc="best")

    fig.tight_layout()
    return _save_fig(fig, "kfold_cv_comparison.png", stage_dir=stage_dir)


def plot_pems_bay_horizons(pems_eval: Dict[str, Any], stage_dir: Optional[Path] = None) -> List[str]:
    """
    What: Vẽ biểu đồ kết quả dự báo giao thông của `DCRNN` trên `PEMS-BAY` ở 3 chân trời 15, 30, 60 phút.
    Why: Minh họa kết quả Bài tập 12 nâng cao (`pems-bay.h5` + `sensor_graph/adj_mx_bay.pkl`).
    Input:
        - pems_eval (Dict[str, Any]): Kết quả đánh giá PEMS-BAY từ `evaluate.py`.
        - stage_dir (Optional[Path]): Thư mục lưu theo `run_id`.
    Output:
        - List[str]: Đường dẫn tệp `pems_bay_horizon_comparison.png`.
    How: Subplot trái vẽ biểu đồ cột `RMSE`, `MAE`, `MAPE (%)` tại 15m, 30m, 60m;
         subplot phải vẽ đường tốc độ thực tế vs dự báo 15m và 60m tại trạm cảm biến đại diện.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    rows = pems_eval["horizon_rows"]
    labels = [r["Horizon"] for r in rows]
    x = np.arange(len(labels))
    w = 0.25

    rmse_vals = [r["RMSE"] for r in rows]
    mae_vals = [r["MAE"] for r in rows]
    mape_vals = [r["MAPE"] for r in rows]

    axes[0].bar(x - w, rmse_vals, width=w, color="#1d4ed8", edgecolor="black", label="RMSE (mph)")
    axes[0].bar(x, mae_vals, width=w, color="#059669", edgecolor="black", label="MAE (mph)")
    axes[0].bar(x + w, mape_vals, width=w, color="#d97706", edgecolor="black", label="MAPE (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_title("Hiệu năng DCRNN trên PEMS-BAY theo Chân trời Dự báo (15, 30, 60 Phút)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Giá trị Sai số")
    axes[0].legend(loc="upper left")

    act_15 = pems_eval.get("sample_sensor_actual_15m", [])
    pred_15 = pems_eval.get("sample_sensor_pred_15m", [])
    pred_60 = pems_eval.get("sample_sensor_pred_60m", [])
    steps = np.arange(len(act_15))
    axes[1].plot(steps, act_15, color="#111827", linewidth=1.8, label="Actual Speed (mph)")
    axes[1].plot(steps, pred_15, color="#059669", linewidth=1.4, label="DCRNN 15-min Ahead (t+3)")
    axes[1].plot(steps, pred_60, color="#7c3aed", linestyle="--", linewidth=1.3, label="DCRNN 60-min Ahead (t+12)")
    axes[1].set_title("Dự báo Tốc độ Xe (mph) của DCRNN tại Trạm Cảm biến Cao tốc (24 Giờ = 288 Bước)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Bước thời gian (5 phút / bước)")
    axes[1].set_ylabel("Speed (mph)")
    axes[1].legend(loc="lower left")

    fig.tight_layout()
    return _save_fig(fig, "pems_bay_horizon_comparison.png", stage_dir=stage_dir)