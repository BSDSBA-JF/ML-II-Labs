from __future__ import annotations

from typing import Optional
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression


class WMTTradingModel(BaseEstimator, RegressorMixin):
    """
    Trading model for WMT using a Linear Regression model on lagged values.

    Based on evaluation results, Linear
    Regression achieved the best forecasting performance compared to VAR(0),
    ARIMA, and other baseline models.
    """

    def __init__(self):
        self._model: Optional[LinearRegression] = None
        self._last_value: Optional[float] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Fit a linear autoregressive model using lag-1 features.

        Parameters
        ----------
        X : np.ndarray
            Input series, used if y is None.
        y : np.ndarray, optional
            Target series.
        """
        if y is None:
            series = np.asarray(X, dtype=float).ravel()
        else:
            series = np.asarray(y, dtype=float).ravel()

        if len(series) < 2:
            raise ValueError("Series must contain at least two values.")

        # Create lag structure: y[t-1] -> y[t]
        X_lag = series[:-1].reshape(-1, 1)
        y_target = series[1:]

        self._model = LinearRegression()
        self._model.fit(X_lag, y_target)

        self._last_value = float(series[-1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Recursively forecast len(X) steps ahead using the fitted model.

        Parameters
        ----------
        X : np.ndarray
            Dummy array — only its length determines horizon.

        Returns
        -------
        np.ndarray
            Forecasted values.
        """
        if self._model is None or self._last_value is None:
            raise RuntimeError("Call fit() before predict().")

        horizon = len(np.asarray(X))
        preds = []
        current_val = self._last_value

        for _ in range(horizon):
            next_val = float(self._model.predict([[current_val]])[0])
            preds.append(next_val)
            current_val = next_val

        return np.asarray(preds, dtype=float)

    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit the model and forecast a horizon equal to len(X).
        """
        self.fit(X, y)
        return self.predict(np.zeros_like(X))