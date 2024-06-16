import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class InvertedMinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._data_min = None
        self._data_max = None

    def fit(self, X, y=None):
        """Compute the minimum and maximum values to be used for later scaling."""
        self._data_min = np.min(X, axis=0)
        self._data_max = np.max(X, axis=0)

        return self

    def transform(self, X):
        """Scale the data to the [0, 1] range and then invert it."""
        X_normalized = (X - self._data_min) / (self._data_max - self._data_min)
        X_normalized_inverted = 1 - X_normalized

        return X_normalized_inverted

    def fit_transform(self, X, y=None):
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)