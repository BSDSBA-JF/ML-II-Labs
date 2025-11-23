from __future__ import annotations

from typing import Optional
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class WMTTradingModel(BaseEstimator, RegressorMixin):
    """
    Trading model for WMT using a VAR(0)-equivalent forecasting strategy.

    Based on evaluation results (Random Walk with Drift excluded), the best model
    for WMT was VAR(0), which corresponds to forecasting using the historical mean
    of the observed series.
    """

    def __init__(self):
        self._mean: Optional[float] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Compute and store the historical mean of the series.

        Parameters
        ----------
        X : np.ndarray
            Input array; if y is None, this is treated as the target series.
        y : np.ndarray, optional
            Target series. If None, X is used.
        """
        if y is None:
            series = np.asarray(X, dtype=float).ravel()
        else:
            series = np.asarray(y, dtype=float).ravel()

        if series.size == 0:
            raise ValueError("Empty series provided to WMTTradingModel.fit().")

        self._mean = float(np.mean(series))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Produce forecasts. The length of X determines the forecast horizon.

        Parameters
        ----------
        X : np.ndarray
            Dummy array; only len(X) is used.

        Returns
        -------
        np.ndarray
            Forecast vector where each value equals the stored mean.
        """
        if self._mean is None:
            raise RuntimeError("Call fit() before predict().")

        horizon = len(np.asarray(X))
        return np.full(horizon, self._mean, dtype=float)

    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit the model then forecast a horizon equal to len(X).
        """
        self.fit(X, y)
        return self.predict(np.zeros_like(X))