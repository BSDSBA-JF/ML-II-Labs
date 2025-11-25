from __future__ import annotations

from typing import Optional
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor


class NVDATradingModel(BaseEstimator, RegressorMixin):
    """
    Trading model for NVDA using a Random Forest regressor with lag-based
    autoregressive learning. Random Forest provided the lowest forecasting
    error among candidate models,
    making it the best-performing approach for NVDA.
    """

    def __init__(self,
                 n_estimators: int = 300,
                 random_state: int = 42,
                 max_depth: Optional[int] = None):
        """
        Parameters
        ----------
        n_estimators : int
            Number of trees in the ensemble.
        random_state : int
            Seed for reproducibility.
        max_depth : int or None
            Maximum depth of trees. None means no max depth.
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_depth = max_depth

        self._model: Optional[RandomForestRegressor] = None
        self._last_value: Optional[float] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Fit a Random Forest model using lag-1 values as features.

        Parameters
        ----------
        X : np.ndarray
            Input series (used if y is None).
        y : np.ndarray, optional
            Target series. If None, X is treated as y.
        """
        if y is None:
            series = np.asarray(X, dtype=float).ravel()
        else:
            series = np.asarray(y, dtype=float).ravel()

        if len(series) < 2:
            raise ValueError("Series must contain at least two values.")

        # Lag mapping: previous value -> next value
        X_lag = series[:-1].reshape(-1, 1)
        y_target = series[1:]

        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            max_depth=self.max_depth
        )
        self._model.fit(X_lag, y_target)

        # Store last observed point for future recursive forecasting
        self._last_value = float(series[-1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Recursively predict len(X) future values.

        Parameters
        ----------
        X : np.ndarray
            Dummy input, only the length matters.

        Returns
        -------
        np.ndarray
            Forecasted values.
        """
        if self._model is None or self._last_value is None:
            raise RuntimeError("Call fit() before predict().")

        horizon = len(np.asarray(X))
        forecasts = []
        current_value = self._last_value

        for _ in range(horizon):
            next_val = float(self._model.predict([[current_value]])[0])
            forecasts.append(next_val)
            current_value = next_val

        return np.asarray(forecasts, dtype=float)

    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit the model and immediately generate a prediction horizon equal to len(X)."""
        self.fit(X, y)
        return self.predict(np.zeros_like(X))