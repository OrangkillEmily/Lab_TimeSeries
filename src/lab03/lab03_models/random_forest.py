"""Random Forest regressor implemented from scratch with bootstrap CART trees."""
from __future__ import annotations
import numpy as np
from .decision_tree import DecisionTreeRegressorScratch


class RandomForestRegressorScratch:
    """Bagged regression trees with bootstrap sampling and random feature subsets."""
    def __init__(self, n_estimators=24, max_depth=7, min_samples_split=20,
                 min_samples_leaf=10, max_features="sqrt", max_samples=0.75,
                 max_thresholds=10, random_state=42, max_train_rows=15000):
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.max_samples = float(max_samples)
        self.max_thresholds = int(max_thresholds)
        self.random_state = int(random_state)
        self.max_train_rows = int(max_train_rows)
        self.trees_ = []
        self.feature_importances_ = None
        self.n_features_in_ = None
        self.n_training_rows_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if X.ndim != 2 or X.shape[0] != y.size or y.size == 0:
            raise ValueError("X/y không hợp lệ.")
        rng = np.random.default_rng(self.random_state)
        if self.max_train_rows and X.shape[0] > self.max_train_rows:
            # Evenly spaced subset preserves coverage of the complete training period.
            idx = np.linspace(0, X.shape[0] - 1, self.max_train_rows).astype(int)
            X, y = X[idx], y[idx]
        self.n_training_rows_ = int(y.size)
        self.n_features_in_ = X.shape[1]
        sample_n = max(2, min(y.size, int(np.ceil(y.size * self.max_samples))))
        self.trees_ = []
        importances = np.zeros(X.shape[1], dtype=np.float64)
        for tree_id in range(self.n_estimators):
            bootstrap = rng.integers(0, y.size, size=sample_n)
            tree = DecisionTreeRegressorScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                max_thresholds=self.max_thresholds,
                random_state=self.random_state + tree_id + 1,
                max_train_rows=0,
            )
            tree.fit(X[bootstrap], y[bootstrap])
            self.trees_.append(tree)
            importances += tree.feature_importances_
        total = importances.sum()
        self.feature_importances_ = importances / total if total > 0 else importances
        return self

    def predict(self, X):
        if not self.trees_:
            raise RuntimeError("Model chưa được fit.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        predictions = np.vstack([tree.predict(X) for tree in self.trees_])
        return predictions.mean(axis=0)

    def get_params(self):
        return {k: v for k, v in self.__dict__.items()
                if k in {"n_estimators", "max_depth", "min_samples_split", "min_samples_leaf",
                         "max_features", "max_samples", "max_thresholds", "random_state",
                         "max_train_rows"}}
