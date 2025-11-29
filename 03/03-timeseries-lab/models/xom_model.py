from __future__ import annotations

from typing import Optional
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class XOMTradingModel(BaseEstimator, RegressorMixin):
    """
    Trading model for XOM using a Random Walk with Drift process.

    This model was selected because it achieved the strongest predictive
    performance among candidate approaches, reflecting the persistent and
    trend-following behavior of XOM's price dynamics.
    """

    def __init__(self):
        self._drift: Optional[float] = None
        self._last_value: Optional[float] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Fit the Random Walk with Drift model by estimating the mean
        change between observations.

        Parameters
        ----------
        X : np.ndarray
            Input time series used if y is None.
        y : np.ndarray, optional
            Target series. If None, X is used.
        """
        if y is None:
            series = np.asarray(X, dtype=float).ravel()
        else:
            series = np.asarray(y, dtype=float).ravel()

        if len(series) < 2:
            raise ValueError("Series must contain at least two observations.")

        # Drift = mean of first differences
        diffs = np.diff(series)
        self._drift = float(np.mean(diffs))
        self._last_value = float(series[-1])

        return self

    def forecast(self, n_steps: int) -> np.ndarray:
        """
        Forecast future values assuming a Random Walk with constant drift.

        Parameters
        ----------
        n_steps : int
            Number of steps ahead to forecast.

        Returns
        -------
        np.ndarray
            Forecast values following RW-with-drift dynamics.
        """
        if self._drift is None or self._last_value is None:
            raise RuntimeError("Call fit() before forecast().")

        # Forecast: last_value + drift * step index
        steps = np.arange(1, n_steps + 1)
        return self._last_value + self._drift * steps