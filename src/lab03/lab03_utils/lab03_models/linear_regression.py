"""Linear Regression implemented from first principles with NumPy only."""
from __future__ import annotations
import numpy as np


class LinearRegressionScratch:
    """Ordinary least-squares regression solved through centered normal equations.

    No scikit-learn estimator or high-level regression helper is used.
    """
    def __init__(self, ridge: float = 1e-8):
        self.ridge = float(ridge)
        self.coef_ = None
        self.intercept_ = None
        self.feature_mean_ = None
        self.feature_scale_ = None
        self.target_mean_ = None
        self.n_features_in_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if X.ndim != 2 or X.shape[0] != y.size or y.size == 0 or X.shape[1] == 0:
            raise ValueError("X phải là ma trận 2D và số dòng phải khớp y không rỗng.")
        self.n_features_in_ = X.shape[1]
        self.feature_mean_ = X.mean(axis=0)
        self.feature_scale_ = X.std(axis=0)
        self.feature_scale_[self.feature_scale_ < 1e-12] = 1.0
        Z = (X - self.feature_mean_) / self.feature_scale_
        self.target_mean_ = float(y.mean())
        yc = y - self.target_mean_
        gram = Z.T @ Z
        gram.flat[:: gram.shape[0] + 1] += self.ridge
        rhs = Z.T @ yc
        try:
            beta = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(gram) @ rhs
        self.coef_ = beta / self.feature_scale_
        self.intercept_ = self.target_mean_ - float(self.feature_mean_ @ self.coef_)
        return self

    def predict(self, X):
        if self.coef_ is None:
            raise RuntimeError("Model chưa được fit.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X @ self.coef_ + self.intercept_

    def get_params(self):
        return {"ridge": self.ridge}
