"""A compact XGBoost-style regressor implemented from scratch using NumPy.

The implementation uses the second-order Taylor objective, regularized leaf
weights, gain-based split selection, row subsampling, column subsampling and
quantile histogram bins. It is an educational implementation, not the official
XGBoost package or a bit-for-bit reproduction of its production engine.
"""
from __future__ import annotations
import numpy as np


class XGBoostScratch:
    def __init__(self, n_estimators=28, learning_rate=0.06, max_depth=3,
                 min_child_weight=5.0, subsample=0.8, colsample_bytree=0.8,
                 reg_lambda=1.0, reg_alpha=0.0, gamma=0.0, max_bins=16,
                 max_train_rows=18000, random_state=42):
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_child_weight = float(min_child_weight)
        self.subsample = float(subsample)
        self.colsample_bytree = float(colsample_bytree)
        self.reg_lambda = float(reg_lambda)
        self.reg_alpha = float(reg_alpha)
        self.gamma = float(gamma)
        self.max_bins = int(max_bins)
        self.max_train_rows = int(max_train_rows)
        self.random_state = int(random_state)
        self.base_score_ = None
        self.bin_edges_ = None
        self.trees_ = []
        self.feature_importances_ = None
        self.n_features_in_ = None
        self.n_training_rows_ = None

    def _soft_threshold(self, value):
        if value > self.reg_alpha:
            return value - self.reg_alpha
        if value < -self.reg_alpha:
            return value + self.reg_alpha
        return 0.0

    def _leaf_weight(self, grad_sum, hess_sum):
        return -self._soft_threshold(grad_sum) / (hess_sum + self.reg_lambda + 1e-12)

    def _score(self, grad_sum, hess_sum):
        g = self._soft_threshold(grad_sum)
        return (g * g) / (hess_sum + self.reg_lambda + 1e-12)

    def _make_bins(self, X):
        edges = []
        for j in range(X.shape[1]):
            quantiles = np.linspace(0.0, 1.0, self.max_bins + 1)[1:-1]
            e = np.unique(np.quantile(X[:, j], quantiles))
            edges.append(e.astype(np.float32))
        self.bin_edges_ = edges
        B = np.empty(X.shape, dtype=np.uint8)
        for j, edge in enumerate(edges):
            B[:, j] = np.searchsorted(edge, X[:, j], side="right").astype(np.uint8)
        return B

    def _build_tree(self, B, grad, hess, rows, features, depth, n_bins, importances):
        G = float(grad[rows].sum())
        H = float(hess[rows].sum())
        node = {"leaf": True, "weight": self._leaf_weight(G, H), "feature": None,
                "threshold_bin": None, "left": None, "right": None}
        if depth >= self.max_depth or rows.size < 2 * max(1, int(self.min_child_weight)):
            return node
        parent_score = self._score(G, H)
        best_gain, best_feature, best_bin = 0.0, None, None
        for feature in features:
            bins = B[rows, feature]
            g_hist = np.bincount(bins, weights=grad[rows], minlength=n_bins).astype(np.float64)
            h_hist = np.bincount(bins, weights=hess[rows], minlength=n_bins).astype(np.float64)
            g_left = np.cumsum(g_hist)[:-1]
            h_left = np.cumsum(h_hist)[:-1]
            g_right = G - g_left
            h_right = H - h_left
            valid = (h_left >= self.min_child_weight) & (h_right >= self.min_child_weight)
            if not np.any(valid):
                continue
            gains = 0.5 * (np.array([self._score(g, h) for g, h in zip(g_left, h_left)])
                           + np.array([self._score(g, h) for g, h in zip(g_right, h_right)])
                           - parent_score) - self.gamma
            gains[~valid] = -np.inf
            local = int(np.argmax(gains))
            if gains[local] > best_gain:
                best_gain = float(gains[local])
                best_feature = int(feature)
                best_bin = local
        if best_feature is None:
            return node
        left_mask = B[rows, best_feature] <= best_bin
        left_rows, right_rows = rows[left_mask], rows[~left_mask]
        if left_rows.size == 0 or right_rows.size == 0:
            return node
        node.update({"leaf": False, "feature": best_feature, "threshold_bin": int(best_bin),
                     "left": self._build_tree(B, grad, hess, left_rows, features, depth + 1, n_bins, importances),
                     "right": self._build_tree(B, grad, hess, right_rows, features, depth + 1, n_bins, importances)})
        importances[best_feature] += best_gain
        return node

    def _predict_tree(self, tree, B):
        out = np.zeros(B.shape[0], dtype=np.float64)
        def visit(node, rows):
            if rows.size == 0:
                return
            if node["leaf"]:
                out[rows] = node["weight"]
                return
            mask = B[rows, node["feature"]] <= node["threshold_bin"]
            visit(node["left"], rows[mask])
            visit(node["right"], rows[~mask])
        visit(tree, np.arange(B.shape[0]))
        return out

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if X.ndim != 2 or X.shape[0] != y.size or y.size == 0:
            raise ValueError("X/y không hợp lệ.")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("X/y có NaN hoặc Inf.")
        if self.max_train_rows and X.shape[0] > self.max_train_rows:
            idx = np.linspace(0, X.shape[0] - 1, self.max_train_rows).astype(int)
            X, y = X[idx], y[idx]
        self.n_training_rows_ = int(y.size)
        self.n_features_in_ = X.shape[1]
        rng = np.random.default_rng(self.random_state)
        B = self._make_bins(X)
        n_bins = max(2, max((int(b.max()) + 1 for b in B.T), default=2))
        self.base_score_ = float(y.mean())
        pred = np.full(y.size, self.base_score_, dtype=np.float64)
        self.trees_ = []
        importances = np.zeros(X.shape[1], dtype=np.float64)
        row_n = max(2, int(y.size * self.subsample))
        feature_n = max(1, int(np.ceil(X.shape[1] * self.colsample_bytree)))
        for _ in range(self.n_estimators):
            grad = pred - y                 # d(1/2 * (pred-y)^2) / d(pred)
            hess = np.ones_like(grad)       # second derivative for squared error
            rows = rng.choice(y.size, size=row_n, replace=False)
            features = rng.choice(X.shape[1], size=feature_n, replace=False)
            tree = self._build_tree(B, grad, hess, rows, features, 0, n_bins, importances)
            self.trees_.append(tree)
            pred += self.learning_rate * self._predict_tree(tree, B)
        total = importances.sum()
        self.feature_importances_ = importances / total if total > 0 else importances
        return self

    def predict(self, X):
        if self.base_score_ is None:
            raise RuntimeError("Model chưa được fit.")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        B = np.empty(X.shape, dtype=np.uint8)
        for j, edge in enumerate(self.bin_edges_):
            B[:, j] = np.searchsorted(edge, X[:, j], side="right").astype(np.uint8)
        pred = np.full(X.shape[0], self.base_score_, dtype=np.float64)
        for tree in self.trees_:
            pred += self.learning_rate * self._predict_tree(tree, B)
        return pred

    def get_params(self):
        return {k: v for k, v in self.__dict__.items() if k in {
            "n_estimators", "learning_rate", "max_depth", "min_child_weight", "subsample",
            "colsample_bytree", "reg_lambda", "reg_alpha", "gamma", "max_bins",
            "max_train_rows", "random_state"}}
