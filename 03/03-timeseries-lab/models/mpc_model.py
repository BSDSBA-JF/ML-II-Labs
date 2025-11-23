from __future__ import annotations

from typing import Optional
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.tsa.arima.model import ARIMA


class MPCTradingModel(BaseEstimator, RegressorMixin):
    """
    Trading model for MPC using an ARIMA model, which achieved the best
    forecasting performance based on evaluation metrics.
    """

    def __init__(self, order: tuple[int, int, int] = (1, 1, 1)):
        self.order = order
        self._results = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Fit ARIMA on the series (y if provided, else X).
        """
        if y is None:
            series = np.asarray(X, dtype=float).ravel()
        else:
            series = np.asarray(y, dtype=float).ravel()

        model = ARIMA(series, order=self.order)
        self._results = model.fit()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Forecast len(X) steps ahead.
        """
        if self._results is None:
            raise RuntimeError("Call fit() before predict().")

        n_steps = len(np.asarray(X))
        fc = self._results.forecast(steps=n_steps)
        return np.asarray(fc, dtype=float)

    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit the model then forecast len(X) steps ahead.
        """
        self.fit(X, y)
        return self.predict(np.zeros_like(X))