from __future__ import annotations

from typing import Optional
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.tsa.arima.model import ARIMA


class MPCTradingModel(BaseEstimator, RegressorMixin):
    """
    Trading model for MPC using an ARIMA(0,1,0) model, which achieved the best
    forecasting performance based on evaluation metrics.

    Since this is a statistical model, the interface follows a simplified
    forecasting design using a `forecast(n_steps)` method instead of predict().
    """

    def __init__(self, order: tuple[int, int, int] = (0, 1, 0)):
        self.order = order
        self._results: Optional[object] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Fit ARIMA(0,1,0) on the series (y if provided, else X).
        """
        if y is None:
            series = np.asarray(X, dtype=float).ravel()
        else:
            series = np.asarray(y, dtype=float).ravel()

        if series.size == 0:
            raise ValueError("Empty series provided to MPCTradingModel.fit().")

        model = ARIMA(series, order=self.order)
        self._results = model.fit()
        return self

    def forecast(self, n_steps: int) -> np.ndarray:
        """
        Forecast forward n future steps.

        Parameters
        ----------
        n_steps : int
            Number of periods ahead to forecast.

        Returns
        -------
        np.ndarray
            Forecasted values.
        """
        if self._results is None:
            raise RuntimeError("Call fit() before forecast().")

        if n_steps <= 0:
            return np.array([], dtype=float)

        fc = self._results.forecast(steps=n_steps)
        return np.asarray(fc, dtype=float)