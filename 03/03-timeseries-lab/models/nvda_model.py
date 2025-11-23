from __future__ import annotations

from typing import Optional
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class NVDATradingModel(BaseEstimator, RegressorMixin):
    """
    Trading model for NVDA using a VAR(0)-equivalent constant-mean model.
    Forecasts future values as the mean of the historical observed values.
    """

    def __init__(self):
        self._mean: Optional[float] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Fit the model by computing the historical mean.

        Parameters
        ----------
        X : np.ndarray
            Input series (dummy or actual data).
        y : np.ndarray, optional
            Target values. If None, X is used as the target.
        """
        if y is None:
            series = np.asarray(X, float).ravel()
        else:
            series = np.asarray(y, float).ravel()

        self._mean = float(np.mean(series))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Forecast len(X) steps ahead using the stored mean.

        Parameters
        ----------
        X : np.ndarray
            Dummy input; only its length is used.

        Returns
        -------
        np.ndarray
            Constant forecast vector of length len(X).
        """
        if self._mean is None:
            raise RuntimeError("Call fit() before predict().")

        n_steps = len(np.asarray(X))
        return np.full(n_steps, self._mean, dtype=float)

    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit the model and generate a forecast horizon equal to len(X).
        """
        self.fit(X, y)
        return self.predict(np.zeros_like(X))