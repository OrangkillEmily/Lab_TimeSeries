"""CART regression tree implemented from first principles using NumPy."""
from __future__ import annotations
import numpy as np


class DecisionTreeRegressorScratch:
    """Regression tree using greedy variance-reduction splits.

    Candidate thresholds are quantiles computed inside each node. This is a
    transparent small-histogram approximation to exhaustive CART threshold search.
    """
    def __init__(self, max_depth=8, min_samples_split=20, min_samples_leaf=10,
                 max_features=None, max_thresholds=12, random_state=42, max_train_rows=15000):
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.max_thresholds = int(max_thresholds)
        self.random_state = int(random_state)
        self.max_train_rows = int(max_train_rows)
        self.n_training_rows_ = None
        self.root_ = None
        self.feature_importances_ = None
        self.n_features_in_ = None
        self._rng = None

    def _feature_count(self, p):
        if self.max_features is None or self.max_features == "all":
            return p
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(p)))
        if self.max_features == "log2":
            return max(1, int(np.log2(p)))
        if isinstance(self.max_features, float):
            return max(1, min(p, int(np.ceil(p * self.max_features))))
        return max(1, min(p, int(self.max_features)))

    def _build(self, X, y, rows, depth):
        values = y[rows]
        prediction = float(values.mean())
        node = {"leaf": True, "value": prediction, "feature": None, "threshold": None,
                "left": None, "right": None}
        if depth >= self.max_depth or rows.size < self.min_samples_split:
            return node
        parent_sse = float(np.sum((values - prediction) ** 2))
        if parent_sse <= 1e-12:
            return node

        p = X.shape[1]
        k = self._feature_count(p)
        features = self._rng.choice(p, size=k, replace=False)
        best_gain, best_feature, best_threshold = 0.0, None, None
        for feature in features:
            x = X[rows, feature]
            if np.min(x) == np.max(x):
                continue
            qs = np.linspace(0.1, 0.9, self.max_thresholds)
            thresholds = np.unique(np.quantile(x, qs))
            for threshold in thresholds:
                left_mask = x <= threshold
                nl = int(left_mask.sum())
                nr = rows.size - nl
                if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                    continue
                yl = values[left_mask]
                yr = values[~left_mask]
                ml, mr = float(yl.mean()), float(yr.mean())
                child_sse = float(np.sum((yl - ml) ** 2) + np.sum((yr - mr) ** 2))
                gain = parent_sse - child_sse
                if gain > best_gain:
                    best_gain, best_feature, best_threshold = gain, int(feature), float(threshold)
        if best_feature is None:
            return node
        x = X[rows, best_feature]
        left_rows = rows[x <= best_threshold]
        right_rows = rows[x > best_threshold]
        if left_rows.size < self.min_samples_leaf or right_rows.size < self.min_samples_leaf:
            return node
        node.update({"leaf": False, "feature": best_feature, "threshold": best_threshold,
                     "left": self._build(X, y, left_rows, depth + 1),
                     "right": self._build(X, y, right_rows, depth + 1)})
        self.feature_importances_[best_feature] += best_gain
        return node

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if X.ndim != 2 or X.shape[0] != y.size or y.size == 0:
            raise ValueError("X/y không hợp lệ.")
        if self.max_train_rows and X.shape[0] > self.max_train_rows:
            subset = np.linspace(0, X.shape[0] - 1, self.max_train_rows).astype(int)
            X, y = X[subset], y[subset]
        self.n_training_rows_ = int(y.size)
        self.n_features_in_ = X.shape[1]
        self._rng = np.random.default_rng(self.random_state)
        self.feature_importances_ = np.zeros(self.n_features_in_, dtype=np.float64)
        self.root_ = self._build(X, y, np.arange(y.size), 0)
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total
        return self

    def _predict_node(self, node, X, rows, out):
        if rows.size == 0:
            return
        if node["leaf"]:
            out[rows] = node["value"]
            return
        mask = X[rows, node["feature"]] <= node["threshold"]
        self._predict_node(node["left"], X, rows[mask], out)
        self._predict_node(node["right"], X, rows[~mask], out)

    def get_params(self):
        return {"max_depth": self.max_depth, "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf, "max_features": self.max_features,
                "max_thresholds": self.max_thresholds, "random_state": self.random_state,
                "max_train_rows": self.max_train_rows}

    def predict(self, X):
        if self.root_ is None:
            raise RuntimeError("Model chưa được fit.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        out = np.empty(X.shape[0], dtype=np.float64)
        self._predict_node(self.root_, X, np.arange(X.shape[0]), out)
        return out
