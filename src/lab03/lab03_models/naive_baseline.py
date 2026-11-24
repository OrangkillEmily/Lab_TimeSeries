"""Persistence baseline: next value equals the most recent observed value."""
import numpy as np


class NaiveLastValue:
    """Stateless lag-1 predictor used as a lower-complexity benchmark."""
    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        # The first feature must be lag_1 and therefore the most recent observation.
        return X[:, 0].copy()

    def get_params(self):
        return {"strategy": "last_observation"}
