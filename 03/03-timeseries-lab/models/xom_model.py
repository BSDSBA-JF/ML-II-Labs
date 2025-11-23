from __future__ import annotations

from typing import Optional
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.tsa.arima.model import ARIMA


class XOMTradingModel(BaseEstimator, RegressorMixin):
    """
    Trading model for XOM using an AR(1)-type model implemented via ARIMA(1,0,0).

    Random Walk with Drift aside, a VAR(1)-style dynamic (here approximated by
    ARIMA(1,0,0) on the univariate series) achieved the best performance for XOM.
    """

    def __init__(self, order: tuple[int, int, int] = (1, 0, 0)):
        """
        Parameters
        ----------
        order : tuple[int, int, int]
            The (p, d, q) order of the ARIMA model. Default is (1, 0, 0),
            corresponding to an AR(1) process.
        """
        self.order = order
        self._results: Optional[object] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Fit ARIMA(1,0,0) (or specified order) on the series.

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
            raise ValueError("Empty series provided to XOMTradingModel.fit().")

        model = ARIMA(series, order=self.order)
        self._results = model.fit()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Forecast len(X) steps ahead.

        Parameters
        ----------
        X : np.ndarray
            Dummy input; only its length is used to determine forecast horizon.

        Returns
        -------
        np.ndarray
            Forecasts of length len(X).
        """
        if self._results is None:
            raise RuntimeError("Call fit() before predict().")

        n_steps = int(len(np.asarray(X)))
        if n_steps <= 0:
            return np.array([], dtype=float)

        fc = self._results.forecast(steps=n_steps)
        return np.asarray(fc, dtype=float)

    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit the model then forecast len(X) steps ahead.
        """
        self.fit(X, y)
        return self.predict(np.zeros_like(X))
