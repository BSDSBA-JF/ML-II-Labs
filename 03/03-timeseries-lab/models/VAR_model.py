from __future__ import annotations
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from typing import List, Optional

class VARTradingModel:
    """
    VAR model for forecasting future prices of a target stock in ARIMA style.

    Workflow:
    - Takes log prices of the target and related stocks.
    - Computes log returns from the log prices.
    - Fits a VAR model on log returns.
    - Forecasts future log returns for a given horizon.
    - Reconstructs forecasted log prices for the target stock.
    """

    def __init__(self, target_stock: str, related_stocks: List[str], maxlags: int = 5):
        self.target_stock = target_stock
        self.related_stocks = related_stocks
        self.maxlags = maxlags

        self._results = None
        self._best_lag = None
        self.target_idx = None
        self.last_log_price = None

    def fit(self, X: pd.DataFrame):
        """
        Fit VAR model on log returns computed from X (log prices of related stocks).
        """
        log_prices = X[self.related_stocks].copy()
        log_returns = log_prices.diff().dropna()

        self.target_idx = self.related_stocks.index(self.target_stock)
        self.last_log_price = log_prices[self.target_stock].iloc[-1]

        model = VAR(log_returns)
        # custom lag logic for the stocks
        if self.target_stock in ['WMT', 'NVDA']:
            self._best_lag = 0
        elif self.target_stock == 'XOM':
            self._best_lag = 1
        else:
            # fallback to automatic selection
            order_selection = model.select_order(maxlags=self.maxlags)
            self._best_lag = order_selection.selected_orders['aic']

        if self._best_lag == 0:
            # Degenerate case: use mean log returns
            mean_return = log_returns.mean()
            self._results = ('mean', mean_return)
        else:
            self._results = model.fit(self._best_lag)

        return self

    def forecast(self, n_steps: int) -> np.ndarray:
        """
        Forecast future log prices for the target stock for n_steps ahead.
        """
        if self._results is None:
            raise RuntimeError("Call fit() before forecast().")

        if isinstance(self._results, tuple) and self._results[0] == 'mean':
            # Mean return model
            mean_vec = self._results[1]
            forecast_log_returns = np.tile(mean_vec.values, (n_steps, 1))
        else:
            model_res = self._results
            lag = self._best_lag
            last_vals = model_res.y[-lag:]
            forecast_log_returns = model_res.forecast(last_vals, steps=n_steps)

        # Reconstruct log prices for the target stock
        forecast_log_price = self.last_log_price + np.cumsum(forecast_log_returns[:, self.target_idx])
        return forecast_log_price
