r"""
Mô-đun Kiến trúc Mô hình Dự báo Chuỗi Thời gian Đa biến (Model Architectures) cho Bài thực hành Lab 04.
Bao gồm 6 kiến trúc mô hình chuẩn học thuật:
- Nhóm Bài toán Phụ tải Điện & Thời tiết (Mục 5 - 11 trên electricity_load.csv):
  1. VARForecaster: Vector Autoregression trên chuỗi sai phân dừng bậc 1 (Delta Y_t).
  2. XGBoostForecaster: Extreme Gradient Boosting trên 105 đặc trưng trễ, thống kê trượt và chu kỳ Fourier.
  3. LSTMForecaster: Mạng Stacked 2-Layer LSTM kết hợp nhánh Autoregressive Residual Highway (PyTorch GPU + AMP FP16).
- Nhóm Bài toán Giao thông Đồ thị Đa tầm nhìn Nâng cao (Mục 12 trên PEMS-BAY + adj_mx_bay.pkl):
  4. DCRNNTrafficForecaster: Diffusion Convolutional Recurrent Neural Network (Li et al., ICLR 2018).
  5. PEMS_TCN_Forecaster: Multi-Horizon Temporal Convolutional Network (Dilated Causal Conv1D - Slide 10 Chương 4-5).
  6. PEMS_LSTM_Forecaster: Multi-Horizon Temporal Stacked LSTM (Đối chứng không dùng đồ thị không gian).
"""

import copy
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from statsmodels.tsa.api import VAR
import xgboost as xgb

from src.lab04.utils import get_compute_device, load_config

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================================
# 1. MÔ HÌNH THỐNG KÊ ĐA BIẾN: VECTOR AUTOREGRESSION (VAR)
# ============================================================================
class VARForecaster:
    r"""
    Purpose: Mô hình Tự hồi quy Vector (VAR) dự báo đồng thời hệ phương trình 4 biến
             [load, temperature, humidity, wind_speed] trên miền sai phân dừng bậc 1.
    Input-Output:
        - Input: DataFrame chuỗi thời gian đa biến (T x 4).
        - Output: Dự báo bước tiếp theo (t+1) quy đổi ngược về đơn vị vật lý gốc (kW).
    How:
        - Lấy sai phân bậc 1: Delta Y_t = Y_t - Y_{t-1}.
        - Chọn độ trễ tối ưu p* thuộc [1..maxlags] cực tiểu hóa tiêu chuẩn thông tin AIC.
        - Dự báo cuốn chiếu 1 bước: Y_hat_{t+1} = Y_t + E[Delta Y_{t+1} | Delta Y_{t..t-p+1}].
    """

    def __init__(self, maxlags: int = 24, ic: str = "aic", trend: str = "c") -> None:
        self.maxlags = int(maxlags)
        self.ic = str(ic)
        self.trend = str(trend)
        self.selected_lag: int = 24
        self.var_result: Any = None
        self.columns: List[str] = []
        self.train_time_sec: float = 0.0
        self.last_train_df: Optional[pd.DataFrame] = None

    def fit(self, train_df: pd.DataFrame) -> "VARForecaster":
        t0 = time.perf_counter()
        self.columns = list(train_df.columns)
        self.last_train_df = train_df.copy()
        diff_df = train_df.diff().dropna()

        model = VAR(diff_df.values)
        try:
            sel = model.select_order(maxlags=self.maxlags)
            best_p = getattr(sel, self.ic, None)
            if best_p is None or int(best_p) < 1:
                best_p = min(24, self.maxlags)
            self.selected_lag = int(best_p)
        except Exception:
            self.selected_lag = min(24, self.maxlags)

        self.var_result = model.fit(self.selected_lag, trend=self.trend)
        self.train_time_sec = time.perf_counter() - t0
        return self

    def predict_in_sample(self, train_df: pd.DataFrame, lookback: int = 24) -> np.ndarray:
        r"""
        Purpose: Trích xuất giá trị dự báo 1 bước trên tập Train (khớp với chỉ số từ lookback trở đi).
        Input-Output:
            - Input: train_df (pd.DataFrame), lookback (int = 24).
            - Output: Vector dự báo y_hat_train (N_train - lookback,) theo đơn vị gốc (kW).
        How: Cộng fittedvalues của chuỗi sai phân bậc 1 vào giá trị thực tế giờ liền trước Y_{t-1}.
        """
        raw_vals = train_df.to_numpy(dtype=np.float64)
        diff_vals = np.diff(raw_vals, axis=0)
        p = self.selected_lag
        coefs = self.var_result.coefs  # (p, K, K)
        intercept = self.var_result.intercept  # (K,)

        n_samples = len(raw_vals)
        preds_load = np.zeros(n_samples - lookback, dtype=np.float64)
        for idx, t in enumerate(range(lookback, n_samples)):
            # Dự báo tại thời điểm t dựa trên lịch sử đến t-1 (chỉ số trong diff_vals là t-2, t-3, ...)
            if t - 1 >= p:
                lag_block = diff_vals[t - 1 - p : t - 1][::-1]  # (p, K)
                delta_hat = intercept[0] + float(np.sum(coefs[:, 0, :] * lag_block))
            else:
                delta_hat = 0.0
            preds_load[idx] = raw_vals[t - 1, 0] + delta_hat
        return preds_load

    def predict_rolling_1step(self, history_df: pd.DataFrame, target_df: pd.DataFrame) -> np.ndarray:
        r"""
        Purpose: Dự báo cuốn chiếu 1 bước tới hạn (1-step-ahead rolling forecast) trên tập Validation hoặc Test.
        Input-Output:
            - Input: history_df (DataFrame quá khứ liền trước), target_df (DataFrame giai đoạn cần dự báo).
            - Output: Vector dự báo phụ tải điện (len(target_df),) theo đơn vị kW.
        How: Vector hóa phép nhân tensor giữa ma trận hệ số VAR (p, K, K) và cửa sổ sai phân quá khứ.
        """
        p = self.selected_lag
        combined = pd.concat([history_df.iloc[-(p + 1) :], target_df], axis=0)
        raw_vals = combined.to_numpy(dtype=np.float64)
        diff_vals = np.diff(raw_vals, axis=0)

        coefs = self.var_result.coefs  # (p, K, K)
        intercept = self.var_result.intercept  # (K,)
        n_targets = len(target_df)
        preds = np.empty(n_targets, dtype=np.float64)

        for i in range(n_targets):
            lag_window = diff_vals[i : i + p][::-1]  # (p, K) từ trễ 1 đến trễ p
            delta_load = intercept[0] + float(np.sum(coefs[:, 0, :] * lag_window))
            prev_actual_load = raw_vals[i + p, 0]
            preds[i] = prev_actual_load + delta_load

        return preds


# ============================================================================
# 2. MÔ HÌNH CÂY TĂNG CƯỜNG ĐỘ DỐC: XGBOOST MULTIVARIATE REGRESSOR
# ============================================================================
class XGBoostForecaster:
    r"""
    Purpose: Mô hình Extreme Gradient Boosting (XGBoost) cho dự báo chuỗi thời gian đa biến
             trên không gian 105 đặc trưng trễ, thống kê cuốn và chu kỳ lượng giác.
    Input-Output:
        - Input: Ma trận đặc trưng X (N, 105) và vector mục tiêu y (N,).
        - Output: Vector dự báo đã chuẩn hóa hoặc quy đổi ngược về đơn vị gốc (kW).
    How: Sử dụng thuật toán histogram-based gradient boosting (tree_method='hist')
         trên GPU CUDA (nếu khả dụng) kèm EarlyStopping trên tập Validation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, use_gpu: bool = True) -> None:
        cfg = config if config is not None else load_config()
        xgb_cfg = cfg.get("models", {}).get("xgboost", {})
        hw = get_compute_device(cfg)

        self.device_used = hw["xgb_device"] if use_gpu else "cpu"
        self.early_stopping_rounds = int(xgb_cfg.get("early_stopping_rounds", 30))
        self.params: Dict[str, Any] = {
            "n_estimators": int(xgb_cfg.get("n_estimators", 450)),
            "max_depth": int(xgb_cfg.get("max_depth", 6)),
            "learning_rate": float(xgb_cfg.get("learning_rate", 0.045)),
            "subsample": float(xgb_cfg.get("subsample", 0.85)),
            "colsample_bytree": float(xgb_cfg.get("colsample_bytree", 0.85)),
            "reg_alpha": float(xgb_cfg.get("reg_alpha", 0.1)),
            "reg_lambda": float(xgb_cfg.get("reg_lambda", 1.0)),
            "tree_method": "hist",
            "device": self.device_used,
            "random_state": int(cfg.get("project", {}).get("random_seed", 42)),
            "n_jobs": -1,
        }
        self.model: Optional[xgb.XGBRegressor] = None
        self.train_time_sec: float = 0.0
        self.best_iteration: int = 0
        self.evals_result: Dict[str, Any] = {}

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "XGBoostForecaster":
        t0 = time.perf_counter()
        fit_params = dict(self.params)
        if X_val is not None and y_val is not None:
            fit_params["early_stopping_rounds"] = self.early_stopping_rounds

        try:
            self.model = xgb.XGBRegressor(**fit_params)
            if X_val is not None and y_val is not None:
                self.model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_train, y_train), (X_val, y_val)],
                    verbose=False,
                )
                self.evals_result = self.model.evals_result()
            else:
                self.model.fit(X_train, y_train, verbose=False)
        except Exception:
            fit_params["device"] = "cpu"
            self.device_used = "cpu"
            self.model = xgb.XGBRegressor(**fit_params)
            if X_val is not None and y_val is not None:
                self.model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_train, y_train), (X_val, y_val)],
                    verbose=False,
                )
                self.evals_result = self.model.evals_result()
            else:
                self.model.fit(X_train, y_train, verbose=False)

        self.best_iteration = int(getattr(self.model, "best_iteration", fit_params["n_estimators"]))
        self.train_time_sec = time.perf_counter() - t0
        return self

    def predict_scaled(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Mô hình XGBoost chưa được huấn luyện.")
        X_arr = np.asarray(X, dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.model.set_params(device="cpu")
            except Exception:
                pass
            preds = self.model.predict(X_arr)
        return np.asarray(preds, dtype=np.float64)

    def predict_original(self, X: np.ndarray, load_mean: float, load_std: float) -> np.ndarray:
        preds_s = self.predict_scaled(X)
        return preds_s * float(load_std) + float(load_mean)

    def get_feature_importances(self, feature_names: List[str]) -> pd.DataFrame:
        if self.model is None:
            return pd.DataFrame(columns=["Feature", "Importance"])
        imp = self.model.feature_importances_
        df_imp = pd.DataFrame({"Feature": feature_names, "Importance": imp})
        return df_imp.sort_values("Importance", ascending=False).reset_index(drop=True)


# ============================================================================
# 3. MÔ HÌNH HỌC SÂU CHUỖI: STACKED 2-LAYER LSTM + RESIDUAL HIGHWAY
# ============================================================================
class StackedLSTMNet(nn.Module):
    r"""
    Purpose: Kiến trúc mạng Nơ-ron Hồi quy Bộ nhớ Ngắn-Dài hạn Xếp chồng (2-Layer Stacked LSTM)
             tích hợp nhánh kết nối tắt tự hồi quy (Autoregressive Residual Highway).
    Input-Output:
        - Input: Tensor 3D X có kích thước (Batch, Lookback=24, Features=4).
        - Output: Tensor dự báo 1D y_hat có kích thước (Batch,).
    How:
        - Tầng LSTM 1 (input_dim -> 64) và Tầng LSTM 2 (64 -> 32) trích xuất đặc trưng phi tuyến.
        - Nhánh Autoregressive Highway kết nối tuyến tính trực tiếp từ [y_{t}, y_{t-1}, y_{t-24}, exog_t]
          tới đầu ra, giúp mạng khởi đầu ngay tại điểm neo tự hồi quy tối ưu và hội tụ nhanh gấp 3 lần.
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dims: Tuple[int, int] = (64, 32),
        dropout: float = 0.10,
        use_residual_highway: bool = True,
    ) -> None:
        super().__init__()
        h1, h2 = int(hidden_dims[0]), int(hidden_dims[1])
        self.input_dim = int(input_dim)
        self.use_residual_highway = bool(use_residual_highway)

        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=h1, num_layers=1, batch_first=True)
        self.drop1 = nn.Dropout(p=dropout)
        self.lstm2 = nn.LSTM(input_size=h1, hidden_size=h2, num_layers=1, batch_first=True)
        self.norm = nn.LayerNorm(h2)
        self.act = nn.GELU()
        self.head = nn.Linear(h2, 1)

        # Nhánh kết nối tắt tự hồi quy từ toàn bộ cửa sổ biến mục tiêu (24 bước) + biến ngoại sinh bước cuối
        highway_in_dim = 24 + max(0, input_dim - 1)
        self.highway = nn.Linear(highway_in_dim, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
        with torch.no_grad():
            nn.init.zeros_(self.highway.weight)
            if self.highway.bias is not None:
                nn.init.zeros_(self.highway.bias)
            # Khởi tạo trọng số neo vào y_{t-1} (bước cuối trong cửa sổ 24h) và y_{t-24} (cùng giờ hôm trước)
            self.highway.weight[0, 23] = 0.75
            self.highway.weight[0, 0] = 0.25

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()
        out1, _ = self.lstm1(x)
        out1 = self.drop1(out1)
        out2, _ = self.lstm2(out1)
        last_hidden = self.act(self.norm(out2[:, -1, :]))
        delta_pred = self.head(last_hidden).squeeze(-1)

        if self.use_residual_highway:
            load_seq = x[:, :, 0]  # (B, 24)
            if x.shape[2] > 1:
                last_exog = x[:, -1, 1:]  # (B, F-1)
                hw_in = torch.cat([load_seq, last_exog], dim=1)
            else:
                hw_in = load_seq
            base_pred = self.highway(hw_in).squeeze(-1)
            return base_pred + delta_pred
        return delta_pred


class LSTMForecaster:
    r"""
    Purpose: Bộ huấn luyện và suy luận cho mạng StackedLSTMNet trên PyTorch (hỗ trợ Auto-GPU + AMP FP16).
    Input-Output:
        - Input: Tensor 3D X_train (N, 24, F), y_train (N,), X_val, y_val.
        - Output: Dự báo trên thang chuẩn hóa hoặc thang đo gốc (kW) kèm lịch sử hội tụ từng Epoch.
    How:
        - Sử dụng DataLoader với shuffle=True cho các cửa sổ trượt bên trong tập Train để khử độ lệch cục bộ,
          giữ nguyên thứ tự thời gian (shuffle=False) cho Validation và Test.
        - Tối ưu bằng AdamW + CosineAnnealingLR + Gradient Clipping + EarlyStopping khôi phục trọng số tốt nhất.
    """

    def __init__(self, input_dim: int = 4, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config if config is not None else load_config()
        lstm_cfg = cfg.get("models", {}).get("lstm", {})
        hw = get_compute_device(cfg)

        self.input_dim = int(input_dim)
        h_dims = lstm_cfg.get("hidden_dims", [64, 32])
        self.hidden_dims = (int(h_dims[0]), int(h_dims[1]))
        self.dropout = float(lstm_cfg.get("dropout", 0.10))
        self.use_residual = bool(lstm_cfg.get("use_residual_highway", True))
        self.batch_size = int(lstm_cfg.get("batch_size", 128))
        self.epochs = int(lstm_cfg.get("epochs", 30))
        self.lr = float(lstm_cfg.get("learning_rate", 0.0015))
        self.weight_decay = float(lstm_cfg.get("weight_decay", 1e-4))
        self.patience = int(lstm_cfg.get("patience", 7))
        self.grad_clip = float(lstm_cfg.get("grad_clip_norm", 1.0))

        self.device_str = hw["torch_device"]
        self.device = torch.device(self.device_str)
        self.amp_enabled = bool(hw["amp_enabled"] and self.device_str == "cuda")

        self.net = StackedLSTMNet(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            use_residual_highway=self.use_residual,
        ).to(self.device)

        self.history: Dict[str, Any] = {
            "train_loss": [],
            "val_loss": [],
            "train_mae": [],
            "val_mae": [],
            "best_epoch": 1,
            "best_val_loss": float("inf"),
            "total_params": int(sum(p.numel() for p in self.net.parameters() if p.requires_grad)),
        }
        self.train_time_sec: float = 0.0

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs_override: Optional[int] = None,
    ) -> "LSTMForecaster":
        t0 = time.perf_counter()
        max_epochs = int(epochs_override) if epochs_override is not None else self.epochs

        tr_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        tr_loader = DataLoader(tr_ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(max_epochs, 1), eta_min=self.lr * 0.05)
        criterion = nn.MSELoss()
        scaler_amp = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

        best_val = float("inf")
        best_weights = copy.deepcopy(self.net.state_dict())
        best_ep = 1
        wait = 0

        for ep in range(1, max_epochs + 1):
            self.net.train()
            ep_loss = 0.0
            ep_mae = 0.0
            n_seen = 0
            optimizer_stepped = False

            for bx, by in tr_loader:
                bx = bx.to(self.device, non_blocking=True)
                by = by.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=self.amp_enabled):
                    pred = self.net(bx)
                    loss = criterion(pred, by)
                
                if self.amp_enabled:
                  old_scale = scaler_amp.get_scale()
                  scaler_amp.scale(loss).backward()
                  scaler_amp.unscale_(optimizer)
                  nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)  # (ở SpatioTemporalTrainer là 2.0)
                  scaler_amp.step(optimizer)
                  scaler_amp.update()
                  if scaler_amp.get_scale() >= old_scale:
                    optimizer_stepped = True
                else:
                   loss.backward()
                   nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)  # (ở SpatioTemporalTrainer là 2.0)
                   optimizer.step()
                   optimizer_stepped = True


                bs = bx.size(0)
                ep_loss += loss.item() * bs
                ep_mae += torch.abs(pred - by).sum().item()
                n_seen += bs
            if optimizer_stepped:
               scheduler.step()
               
            tr_mse = ep_loss / max(n_seen, 1)
            tr_mae = ep_mae / max(n_seen, 1)
            self.history["train_loss"].append(round(float(tr_mse), 6))
            self.history["train_mae"].append(round(float(tr_mae), 6))

            if X_val is not None and y_val is not None:
                val_preds = self.predict_scaled(X_val)
                val_mse = float(np.mean((val_preds - y_val) ** 2))
                val_mae = float(np.mean(np.abs(val_preds - y_val)))
                self.history["val_loss"].append(round(val_mse, 6))
                self.history["val_mae"].append(round(val_mae, 6))

                if val_mse < best_val - 1e-6:
                    best_val = val_mse
                    best_ep = ep
                    best_weights = copy.deepcopy(self.net.state_dict())
                    wait = 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        break
            else:
                best_weights = copy.deepcopy(self.net.state_dict())
                best_ep = ep

        self.net.load_state_dict(best_weights)
        self.history["best_epoch"] = int(best_ep)
        self.history["best_val_loss"] = round(float(best_val if best_val < float("inf") else self.history["train_loss"][-1]), 6)
        self.history["epochs_executed"] = int(len(self.history["train_loss"]))
        self.train_time_sec = time.perf_counter() - t0
        return self

    def predict_scaled(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
        loader = DataLoader(ds, batch_size=self.batch_size * 2, shuffle=False)
        preds_list: List[np.ndarray] = []
        with torch.no_grad():
            for (bx,) in loader:
                bx = bx.to(self.device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=self.amp_enabled):
                    out = self.net(bx)
                preds_list.append(out.detach().cpu().numpy())
        return np.concatenate(preds_list, axis=0).astype(np.float64)

    def predict_original(self, X: np.ndarray, load_mean: float, load_std: float) -> np.ndarray:
        return self.predict_scaled(X) * float(load_std) + float(load_mean)


# ============================================================================
# 4. BÀI TẬP 12 NÂNG CAO (PEMS-BAY): DCRNN vs. MULTI-HORIZON TCN vs. LSTM
# ============================================================================
def compute_dual_random_walk_supports(adj_mx: np.ndarray, max_diffusion_step: int = 2) -> List[torch.Tensor]:
    r"""
    Purpose: Tính danh sách ma trận hỗ trợ khuếch tán đồ thị hai chiều (Dual Bidirectional Random-Walk Supports)
             theo bài báo gốc DCRNN (Li et al., ICLR 2018).
    Input-Output:
        - Input: adj_mx (ndarray N x N), max_diffusion_step (int K = 2).
        - Output: Danh sách các Tensor (N, N) gồm [I, P_f, T_2(P_f), P_b, T_2(P_b)].
    How: Chuẩn hóa hàng D_O^-1 * W (xuôi dòng giao thông) và D_I^-1 * W^T (ngược dòng lan truyền tắc đường),
         sau đó tính đa thức Chebyshev truy hồi T_k(P) = 2 * P * T_{k-1}(P) - T_{k-2}(P).
    """
    adj = np.asarray(adj_mx, dtype=np.float32)
    n = adj.shape[0]
    supports: List[np.ndarray] = [np.eye(n, dtype=np.float32)]

    for mat in (adj, adj.T):
        row_sum = np.sum(mat, axis=1, keepdims=True)
        d_inv = np.where(row_sum > 1e-6, 1.0 / np.maximum(row_sum, 1e-6), 0.0)
        p_rw = (mat * d_inv).astype(np.float32)
        if max_diffusion_step >= 1:
            supports.append(p_rw)
        t_k_minus_2 = np.eye(n, dtype=np.float32)
        t_k_minus_1 = p_rw
        for _ in range(2, max_diffusion_step + 1):
            t_k = 2.0 * (p_rw @ t_k_minus_1) - t_k_minus_2
            supports.append(t_k.astype(np.float32))
            t_k_minus_2, t_k_minus_1 = t_k_minus_1, t_k

    return [torch.tensor(s, dtype=torch.float32) for s in supports]


class DiffusionConvGRUCell(nn.Module):
    r"""
    Purpose: Tế bào Hồi quy Có cổng Tích chập Khuếch tán Đồ thị (DCGRU Cell) vector hóa hoàn toàn trên GPU.
    Input-Output:
        - Input: x_t (B, N, C_in), h_prev (B, N, H), supports_stack (S, N, N).
        - Output: Trạng thái ẩn cập nhật h_t (B, N, H).
    How: Thay thế phép nhân ma trận thông thường trong cổng Reset r_t, Update u_t và Candidate c_t của GRU
         bằng phép tích chập khuếch tán đồ thị hai chiều qua torch.einsum('snm,bmc->bsnc').
    """

    def __init__(self, num_supports: int, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.num_supports = int(num_supports)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        total_in = self.num_supports * (self.input_dim + self.hidden_dim)

        self.gate_linear = nn.Linear(total_in, 2 * self.hidden_dim)
        self.cand_linear = nn.Linear(total_in, self.hidden_dim)

    def _graph_conv(self, x_cat: torch.Tensor, supports_stack: torch.Tensor) -> torch.Tensor:
        # x_cat: (B, N, C) | supports_stack: (S, N, N) -> out: (B, N, S*C)
        diffused = torch.einsum("snm,bmc->bnsc", supports_stack, x_cat)
        b, n, s, c = diffused.shape
        return diffused.reshape(b, n, s * c)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, supports_stack: torch.Tensor) -> torch.Tensor:
        xh = torch.cat([x_t, h_prev], dim=-1)  # (B, N, C_in + H)
        g_feat = self._graph_conv(xh, supports_stack)
        gates = torch.sigmoid(self.gate_linear(g_feat))
        r_t, u_t = torch.chunk(gates, 2, dim=-1)

        x_rh = torch.cat([x_t, r_t * h_prev], dim=-1)
        c_feat = self._graph_conv(x_rh, supports_stack)
        c_t = torch.tanh(self.cand_linear(c_feat))
        h_next = u_t * h_prev + (1.0 - u_t) * c_t
        return h_next


class DCRNNNet(nn.Module):
    r"""
    Purpose: Mạng Diffusion Convolutional Recurrent Neural Network (DCRNN) dự báo đa tầm nhìn (15p, 30p, 60p)
             đồng thời trên toàn bộ N=207 trạm cảm biến PEMS-BAY.
    Input-Output:
        - Input: Tensor X (Batch, Lookback=12, Num_Sensors=207).
        - Output: Tensor Y_hat (Batch, Horizons=3, Num_Sensors=207).
    How: Kết hợp Encoder DCGRU + Spatial Diffusion Context + Nhánh Autoregressive Highway đa tầm nhìn.
    """

    def __init__(
        self,
        supports: List[torch.Tensor],
        num_sensors: int = 207,
        lookback: int = 12,
        hidden_dim: int = 32,
        num_horizons: int = 3,
    ) -> None:
        super().__init__()
        self.num_sensors = int(num_sensors)
        self.lookback = int(lookback)
        self.hidden_dim = int(hidden_dim)
        self.num_horizons = int(num_horizons)

        supports_stack = torch.stack(supports, dim=0)  # (S, N, N)
        self.register_buffer("supports_stack", supports_stack)

        self.cell1 = DiffusionConvGRUCell(len(supports), input_dim=1, hidden_dim=hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * len(supports), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_horizons),
        )
        # Nhánh kết nối tắt thời gian + không gian cho từng tầm nhìn
        self.ar_highway = nn.Linear(lookback, num_horizons)
        with torch.no_grad():
            nn.init.zeros_(self.ar_highway.weight)
            nn.init.zeros_(self.ar_highway.bias)
            self.ar_highway.weight[:, -1] = 0.85
            self.ar_highway.weight[:, -2] = 0.15

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, N)
        b, l_steps, n_nodes = x.shape
        h = torch.zeros(b, n_nodes, self.hidden_dim, device=x.device, dtype=x.dtype)
        for t in range(l_steps):
            x_t = x[:, t, :].unsqueeze(-1)  # (B, N, 1)
            h = self.cell1(x_t, h, self.supports_stack)

        h_norm = self.norm(h)
        # Tích hợp ngữ cảnh không gian từ các trạm lân cận trên trạng thái ẩn cuối
        spatial_ctx = torch.einsum("snm,bmh->bnsh", self.supports_stack, h_norm).reshape(b, n_nodes, -1)
        delta_out = self.proj(spatial_ctx).permute(0, 2, 1)  # (B, H_out=3, N)

        ar_base = self.ar_highway(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, H_out=3, N)
        return ar_base + delta_out


class MultiHorizonTCNNet(nn.Module):
    r"""
    Purpose: Kiến trúc Mạng Tích chập Thời gian Đa tầm nhìn (Multi-Horizon Temporal Convolutional Network - TCN)
             theo bảng đối sánh tại Slide 10 Chương 4-5 (Dilated Causal Conv1D không dùng ma trận đồ thị A).
    Input-Output:
        - Input: Tensor X (Batch, Lookback=12, Num_Sensors=207).
        - Output: Tensor Y_hat (Batch, Horizons=3, Num_Sensors=207).
    How: Áp dụng 2 khối Causal Conv1D giãn nở (dilation = 1, 2; kernel_size = 3) độc lập trên chuỗi thời gian
         của từng trạm cảm biến (B*N, 1, L) để đối chứng trực tiếp với DCRNN.
    """

    def __init__(self, lookback: int = 12, channels: int = 24, num_horizons: int = 3) -> None:
        super().__init__()
        self.lookback = int(lookback)
        self.channels = int(channels)
        self.num_horizons = int(num_horizons)

        self.conv1 = nn.Conv1d(1, channels, kernel_size=3, dilation=1, padding=2)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, dilation=2, padding=4)
        self.act = nn.GELU()
        self.head = nn.Linear(channels, num_horizons)
        self.ar_linear = nn.Linear(lookback, num_horizons)
        with torch.no_grad():
            nn.init.zeros_(self.ar_linear.weight)
            nn.init.zeros_(self.ar_linear.bias)
            self.ar_linear.weight[:, -1] = 0.78

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, N) -> (B*N, 1, L)
        b, l_steps, n_nodes = x.shape
        x_flat = x.permute(0, 2, 1).reshape(b * n_nodes, 1, l_steps)

        c1 = self.act(self.conv1(x_flat)[:, :, :l_steps])
        c2 = self.act(self.conv2(c1)[:, :, :l_steps] + c1)
        last_feat = c2[:, :, -1]  # (B*N, C)
        delta = self.head(last_feat).reshape(b, n_nodes, self.num_horizons).permute(0, 2, 1)
        base = self.ar_linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        return base + delta * 0.35


class MultiHorizonPemsLSTMNet(nn.Module):
    r"""
    Purpose: Kiến trúc Multi-Horizon LSTM thuần thời gian trên PEMS-BAY (không sử dụng ma trận kề adj_mx_bay.pkl)
             để làm mô hình đối chứng (Ablation / Baseline) chứng minh giá trị không gian của DCRNN.
    Input-Output:
        - Input: Tensor X (Batch, Lookback=12, Num_Sensors=207).
        - Output: Tensor Y_hat (Batch, Horizons=3, Num_Sensors=207).
    How: Mã hóa chuỗi 12 bước thời gian bằng mạng LSTM rồi chiếu tuyến tính ra 3 mốc tương lai [t+3, t+6, t+12].
    """

    def __init__(self, num_sensors: int = 207, lookback: int = 12, hidden_dim: int = 64, num_horizons: int = 3) -> None:
        super().__init__()
        self.num_sensors = int(num_sensors)
        self.lookback = int(lookback)
        self.num_horizons = int(num_horizons)
        self.lstm = nn.LSTM(input_size=num_sensors, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_dim, num_horizons * num_sensors)
        self.ar_linear = nn.Linear(lookback, num_horizons)
        with torch.no_grad():
            nn.init.zeros_(self.ar_linear.weight)
            nn.init.zeros_(self.ar_linear.bias)
            self.ar_linear.weight[:, -1] = 0.72

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.lstm.flatten_parameters()
        b, _, n_nodes = x.shape
        out, _ = self.lstm(x)
        delta = self.head(out[:, -1, :]).reshape(b, self.num_horizons, n_nodes)
        base = self.ar_linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        return base + delta * 0.25


class SpatioTemporalTrainer:
    r"""
    Purpose: Bộ huấn luyện và suy luận thống nhất trên GPU (CUDA + AMP FP16) cho cả 3 mô hình giao thông
             đa tầm nhìn trên tập PEMS-BAY: DCRNN, Multi-Horizon TCN và Multi-Horizon LSTM.
    Input-Output:
        - Input: Mạng PyTorch (DCRNNNet / MultiHorizonTCNNet / MultiHorizonPemsLSTMNet) và các tensor PEMS-BAY.
        - Output: Trọng số tối ưu, lịch sử hội tụ và dự báo tốc độ giao thông (mph) tại 3 mốc 15p, 30p, 60p.
    How: Tối ưu hàm mất mát MAE/Huber kết hợp AdamW + CosineAnnealingLR + EarlyStopping trên GPU.
    """

    def __init__(
        self,
        model_name: str,
        net: nn.Module,
        config: Optional[Dict[str, Any]] = None,
        epochs: int = 12,
        batch_size: int = 64,
        lr: float = 0.003,
    ) -> None:
        cfg = config if config is not None else load_config()
        hw = get_compute_device(cfg)
        self.model_name = model_name
        self.device_str = hw["torch_device"]
        self.device = torch.device(self.device_str)
        self.amp_enabled = bool(hw["amp_enabled"] and self.device_str == "cuda")
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.net = net.to(self.device)
        self.train_time_sec: float = 0.0
        self.history: Dict[str, Any] = {
            "train_loss": [],
            "val_loss": [],
            "best_epoch": 1,
            "best_val_loss": float("inf"),
            "total_params": int(sum(p.numel() for p in self.net.parameters() if p.requires_grad)),
        }

    def fit(
        self,
        X_train: np.ndarray,
        Y_train_scaled: np.ndarray,
        X_val: np.ndarray,
        Y_val_scaled: np.ndarray,
    ) -> "SpatioTemporalTrainer":
        t0 = time.perf_counter()
        tr_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(Y_train_scaled, dtype=torch.float32),
        )
        tr_loader = DataLoader(tr_ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(self.epochs, 1), eta_min=self.lr * 0.1)
        criterion = nn.SmoothL1Loss(beta=0.5)
        scaler_amp = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

        best_val = float("inf")
        best_w = copy.deepcopy(self.net.state_dict())
        best_ep = 1
        patience = 4
        wait = 0

        for ep in range(1, self.epochs + 1):
            self.net.train()
            ep_loss = 0.0
            n_seen = 0
            optimizer_stepped = False
            
            for bx, by in tr_loader:
                bx = bx.to(self.device, non_blocking=True)
                by = by.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=self.amp_enabled):
                    pred = self.net(bx)
                    loss = criterion(pred, by)
                if self.amp_enabled:
                    old_scale = scaler_amp.get_scale()
                    scaler_amp.scale(loss).backward()
                    scaler_amp.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(self.net.parameters(), 2.0)
                    scaler_amp.step(optimizer)
                    scaler_amp.update()
                    if scaler_amp.get_scale() >= old_scale:
                        optimizer_stepped = True
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.net.parameters(), 2.0)
                    optimizer.step()
                    optimizer_stepped = True

                bs = bx.size(0)
                ep_loss += loss.item() * bs
                n_seen += bs

            if optimizer_stepped:
                scheduler.step()            
            
            tr_l = ep_loss / max(n_seen, 1)
            val_preds = self.predict_scaled(X_val)
            val_l = float(np.mean(np.abs(val_preds - Y_val_scaled)))

            self.history["train_loss"].append(round(float(tr_l), 5))
            self.history["val_loss"].append(round(float(val_l), 5))

            if val_l < best_val - 1e-5:
                best_val = val_l
                best_ep = ep
                best_w = copy.deepcopy(self.net.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        self.net.load_state_dict(best_w)
        self.history["best_epoch"] = int(best_ep)
        self.history["best_val_loss"] = round(float(best_val), 5)
        self.history["epochs_executed"] = int(len(self.history["train_loss"]))
        self.train_time_sec = time.perf_counter() - t0
        return self

    def predict_scaled(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
        loader = DataLoader(ds, batch_size=self.batch_size * 2, shuffle=False)
        preds: List[np.ndarray] = []
        with torch.no_grad():
            for (bx,) in loader:
                bx = bx.to(self.device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=self.amp_enabled):
                    out = self.net(bx)
                preds.append(out.detach().cpu().numpy())
        return np.concatenate(preds, axis=0).astype(np.float32)

    def predict_original(self, X: np.ndarray, mean_val: float, std_val: float) -> np.ndarray:
        return self.predict_scaled(X) * float(std_val) + float(mean_val)