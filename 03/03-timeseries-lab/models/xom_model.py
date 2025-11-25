from __future__ import annotations

from typing import Optional
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class XOMTradingModel(BaseEstimator, RegressorMixin):
    """
    Trading model for XOM using a VAR(1)-style forecasting structure.

    Although VAR is traditionally multivariate, XOM behaved consistently with a
    first-order lag dependence. Therefore, the VAR(1) logic is implemented as an
    AR(1) recurrence using the learned lag coefficient and intercept.
    """

    def __init__(self):
        self.coef_: Optional[float] = None      # lag coefficient
        self.intercept_: Optional[float] = None # constant term
        self.last_value_: Optional[float] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Fit a first-order autoregressive model (VAR(1) equivalent on univariate data).

        Parameters
        ----------
        X : np.ndarray
            Input time series.
        y : np.ndarray, optional
            Target series. If None, X is used.
        """
        if y is None:
            series = np.asarray(X, dtype=float).ravel()
        else:
            series = np.asarray(y, dtype=float).ravel()

        if len(series) < 2:
            raise ValueError("Series must contain at least two observations.")

        # Construct VAR(1) lag regression:  y[t] = intercept + coef * y[t-1]
        y_target = series[1:]
        y_lag = series[:-1]

        # Ordinary Least Squares for AR(1)
        A = np.column_stack([np.ones_like(y_lag), y_lag])
        params = np.linalg.lstsq(A, y_target, rcond=None)[0]

        self.intercept_, self.coef_ = float(params[0]), float(params[1])
        self.last_value_ = float(series[-1])

        return self

    def forecast(self, n_steps: int) -> np.ndarray:
        """
        Forecast future values recursively using VAR(1) dynamics.

        Parameters
        ----------
        n_steps : int
            Number of future steps to forecast.

        Returns
        -------
        np.ndarray
            Forecast array of length n_steps.
        """
        if self.coef_ is None or self.intercept_ is None or self.last_value_ is None:
            raise RuntimeError("Call fit() before forecast().")

        forecasts = []
        value = self.last_value_

        for _ in range(n_steps):
            value = self.intercept_ + self.coef_ * value
            forecasts.append(value)

        return np.asarray(forecasts, dtype=float)