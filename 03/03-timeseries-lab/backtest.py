from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from utils import TradingMetrics as trade_metrics


class Backtest:
    """Perform backtesting to assess the model's profitability on observed
    data.
    """

    def __init__(
        self, starting_capital: int = 10000, transaction_cost: float = 0.001
    ) -> None:
        self.starting_capital = starting_capital
        self.transaction_cost = transaction_cost
        self._reset()

    def _reset(self):
        """Reset to avoid errors for every run of the strategy."""
        self.cash = self.starting_capital
        self.position = 0
        self.portfolio_values = []
        self.trades = []
        self.trade_log = []

    def run_strategy(
        self,
        actual_prices: Union[np.ndarray, pd.Series],
        predicted_prices: Union[np.ndarray, pd.Series],
        default: bool = True,
        threshold: float = 0.01,
    ) -> Optional[List[Dict[str, float]]]:
        """Run the trading strategy.

        Parameters:
        -----------
        actual_prices (np.ndarray, pd.Series): Actual prices
        predicted_prices (np.ndarray, pd.Series): Predicted prices
        default (bool): Indicates if the default strategy would be done
            (default=True)
        threshold (float): Threshold for the price difference between the
            actual vs predicted prices (default=0.01 -> 1% difference).
        """
        self._reset()  # Fixed to avoid any issues with re-running the strategy

        # Convert pandas Series into an array
        actual_prices = self._check_input_instance(actual_prices)
        predicted_prices = self._check_input_instance(predicted_prices)

        if default:
            return self.default_strategy(
                actual_prices, predicted_prices, threshold=threshold
            )
        return self.custom_strategy(actual_prices, predicted_prices)

    def default_strategy(
        self,
        actual_prices: Union[np.ndarray, pd.Series],
        predicted_prices: Union[np.ndarray, pd.Series],
        threshold: float = 0.01,
    ):
        """Conduct a simple thresholding strategy. The rules are:
        - BUY: If predicted prices > actual prices * (1 + threshold) and no
            position
        - SELL: If predicted prices < actual prices * (1 - threshold) and holding
            a position
        - HOLD: Otherwise

        Parameters:
        -----------
        actual_prices (np.ndarray, pd.Series): Actual prices
        predicted_prices (np.ndarray, pd.Series): Predicted prices
        threshold (float): Acceptable threshold (in %) for actual vs predicted price
            difference. For example, if my model predicted a 1% higher price,
            we buy shares.
        """

        trade_length = len(actual_prices)

        for time in range(trade_length):
            action = "HOLD"
            shares_traded = 0

            current_price = actual_prices[time]
            predicted_price = predicted_prices[time]

            if predicted_price > current_price * (1 + threshold) and self.position == 0:
                # Buy as much shares as you can

                shares_to_buy = int(
                    self.cash / (current_price * (1 + self.transaction_cost))
                )

                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                    self.cash -= cost
                    self.position = shares_to_buy
                    action = "BUY"
                    shares_traded = shares_to_buy
            elif (
                predicted_price < current_price * (1 - threshold) and self.position > 0
            ):
                # Sell everything
                revenue = self.position * current_price * (1 - self.transaction_cost)
                profit = revenue - (
                    self.position * self.entry_price * (1 + self.transaction_cost)
                )
                self.trades.append(profit)
                self.cash += revenue
                action = "SELL"
                shares_traded = self.position
                self.position = 0

            if action == "BUY":
                self.entry_price = current_price

            # Compute portfolio value
            portfolio_value = self.cash + (self.position * current_price)
            self.portfolio_values.append(portfolio_value)

            # Log your trade
            self.trade_log.append(
                {
                    "Actual Price": current_price,
                    "Predicted Price": predicted_price,
                    "Action": action,
                    "Shares Traded": shares_traded,
                    "Position": self.position,
                    "Cash": self.cash,
                    "Portfolio Value": portfolio_value,
                }
            )
        return self.trade_log

    # ================================START==================================
    # =================DELETE THESE COMMENTS AFTER EDITING===================
    # FEEL FREE TO EDIT THE METHOD BELOW TO INCLUDE YOUR OWN TRADING STRATEGY
    # ENSURE YOU INCLUDE DOCSTRINGS AND TYPE ANNOTATIONS
    def custom_strategy(self, actual_prices, predicted_prices):
        """
        Strategy:
        - Buy as many shares as possible on the first day
        - Hold until the last day
        - Sell all shares on the last day
        """
        # Reset before running
        self._reset()
        actual_prices = self._check_input_instance(actual_prices)

        first_price = actual_prices[0]
        last_price = actual_prices[-1]

        # ---- BUY on first day ----
        self.position = int(self.cash / (first_price * (1 + self.transaction_cost)))
        cost = self.position * first_price * (1 + self.transaction_cost)
        self.cash -= cost
        self.entry_price = first_price

        portfolio_value = self.cash + self.position * first_price
        self.portfolio_values.append(portfolio_value)
        self.trade_log.append({
            "Actual Price": first_price,
            "Predicted Price": first_price,
            "Action": "BUY",
            "Shares Traded": self.position,
            "Position": self.position,
            "Cash": self.cash,
            "Portfolio Value": portfolio_value
        })

        # ---- HOLD until last day ----
        for price in actual_prices[1:-1]:
            portfolio_value = self.cash + self.position * price
            self.portfolio_values.append(portfolio_value)
            self.trade_log.append({
                "Actual Price": price,
                "Predicted Price": price,
                "Action": "HOLD",
                "Shares Traded": 0,
                "Position": self.position,
                "Cash": self.cash,
                "Portfolio Value": portfolio_value
            })

        # ---- SELL on last day ----
        shares_sold = self.position  # log actual sold shares
        revenue = shares_sold * last_price * (1 - self.transaction_cost)
        profit = revenue - (shares_sold * self.entry_price * (1 + self.transaction_cost))
        self.trades.append(profit)
        self.cash += revenue
        self.position = 0

        portfolio_value = self.cash
        self.portfolio_values.append(portfolio_value)
        self.trade_log.append({
            "Actual Price": last_price,
            "Predicted Price": last_price,
            "Action": "SELL",
            "Shares Traded": shares_sold,
            "Position": self.position,
            "Cash": self.cash,
            "Portfolio Value": portfolio_value
        })

        return self.trade_log

    # ================================END===================================

    def compute_trade_metrics(self, annual_risk_free_rate: float = 0.02):
        """Return a dictionary of all trade metrics."""
        returns = self._get_returns()

        return trade_metrics.compute_all_metrics(
            self.portfolio_values, returns, self.trades, annual_risk_free_rate
        )

    def _get_returns(self):
        """Compute the portfolio returns."""
        if len(self.portfolio_values) < 2:
            return np.array([])
        portfolio = np.array(self.portfolio_values)
        return np.diff(portfolio) / portfolio[:-1]

    def _check_input_instance(self, input_arr: Union[np.ndarray, pd.Series]):
        if isinstance(input_arr, pd.Series):
            return np.asarray(input_arr)
        return input_arr
