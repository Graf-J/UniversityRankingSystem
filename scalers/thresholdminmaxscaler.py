import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ThresholdMinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float):
        self.threshold = threshold
        self._data_min = None
        self._data_max = None

    def fit(self, X, y=None):
        """Compute the minimum and maximum values of the difference to be used for later scaling."""
        X_abs_diff = np.abs(X - self.threshold)
        self._data_min = np.min(X_abs_diff, axis=0)
        self._data_max = np.max(X_abs_diff, axis=0)

        return self

    def transform(self, X):
        """Scale the differences to the [0, 1] range."""
        X_abs_diff = np.abs(X - self.threshold)
        X_normalized = (X_abs_diff - self._data_min) / (self._data_max - self._data_min)
        X_normalized_inverted = 1 - X_normalized

        return X_normalized_inverted

    def fit_transform(self, X, y=None):
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)