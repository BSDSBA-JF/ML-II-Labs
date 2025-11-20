from typing import Any, Dict, Optional, Type, Union

import numpy as np
import pandas as pd


class ForecastingMetrics:
    """Class object that contains static methods for relevant
    forecasting metrics.

    Example Usage:
    --------------
    >>> from utils import ForecastingMetrics as forecast_metrics
    >>> forecast_metrics.mae(y_true, y_pred)
    """

    @staticmethod
    def mae(
        y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]
    ) -> np.float32:
        """Compute the mean absolute error (MAE) of a model's predictions
        against the actual (true) values.
        """
        return np.float32(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def mse(
        y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]
    ) -> np.float32:
        """Compute the mean squared error (MSE) of a model's predictions
        against the actual (true) values.
        """
        return np.float32(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def rmse(
        y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]
    ) -> np.float32:
        """Compute the root mean squared error (RMSE) of a model's predictions
        against the actual (true) values.
        """
        return np.float32(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def mape(
        y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]
    ) -> np.float32:
        """Compute the mean absolute error (MAPE), in %, of a model's predictions
        against the actual (true) values.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        nonzero_filter = y_true != 0  # avoid division by 0
        y_true = y_true[nonzero_filter]
        y_pred = y_pred[nonzero_filter]
        return np.float32(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

    @staticmethod
    def smape(
        y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]
    ) -> np.float32:
        """Compute the symmetric mean absolute error (SMAPE), in %, of a
        model's predictions against the actual (true) values.
        """
        numerator = np.abs(y_true - y_pred)
        denominator = np.abs(y_true) + np.abs(y_pred)
        nonzero_filter = denominator != 0

        return np.float32(
            np.mean(numerator[nonzero_filter] / (denominator[nonzero_filter] / 2)) * 100
        )

    @staticmethod
    def direction_accuracy(
        y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]
    ) -> np.float32:
        """Compute the direction accuracy, in %, of your predictions against the
        actual(true) values.
        """
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        return np.float32(np.mean(true_direction == pred_direction) * 100)

    # =================DELETE THESE COMMENTS AFTER EDITING===================
    # FEEL FREE TO EDIT THE METHOD BELOW TO INCLUDE YOUR OTHER METRICS
    # YOU MAY ADD MORE METRICS FOR FORECASTING IF YOU DEEM NECESSARY

    @classmethod
    def compute_all_metrics(
        cls, y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]
    ) -> Dict[str, np.float32]:
        """Return a dictionary containing all of the metrics found within this
        class.
        """
        return {
            "MAE": cls.mae(y_true, y_pred),
            "MSE": cls.mse(y_true, y_pred),
            "RMSE": cls.rmse(y_true, y_pred),
            "MAPE": cls.mape(y_true, y_pred),
            "SMAPE": cls.smape(y_true, y_pred),
            "Directional Accuracy": cls.direction_accuracy(y_true, y_pred),
        }


class TradingMetrics:
    """Class object that contains static methods for relevant
    stock trading metrics.

    Example Usage:
    --------------
    >>> from utils import TradingMetrics as trade_metrics
    >>> trade_metrics.cumulative_return(portfolio_values)

    """

    @staticmethod
    def cumulative_return(portfolio_values):
        """Return the cumulative return of the portfolio (in %)."""
        return (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0] * 100

    @staticmethod
    def sharpe_ratio(returns, annual_risk_free_rate=0.02):
        """Compute the Sharpe Ratio"""
        # To get to the daily risk free rate, we divide the annual to 252
        # or the number of trading days
        daily_risk_free_rate = annual_risk_free_rate / 252

        if len(returns) == 0 or np.std(returns) == 0:
            return 0
        return (np.mean(returns) - daily_risk_free_rate) / np.std(returns)

    @staticmethod
    def max_drawdown(portfolio_values):
        """Compute the max drawdown (in %)."""
        period_peaks = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - period_peaks) / period_peaks
        return np.min(drawdowns) * 100

    @staticmethod
    def win_rate(trades):
        """Compute the win rate (in %)"""
        if len(trades) == 0:
            return 0
        wins = sum(1 for trade in trades if trade > 0)
        return wins / len(trades) * 100

    @staticmethod
    def profit_factor(trades):
        """Compute the profit factor"""
        if len(trades) == 0:
            return 0
        gross_profit = sum(trade for trade in trades if trade > 0)
        gross_loss = abs(sum(trade for trade in trades if trade < 0))

        if gross_loss == 0:
            return 0 if gross_profit < 0 else np.inf
        return gross_profit / gross_loss

    # =================DELETE THESE COMMENTS AFTER EDITING===================
    # FEEL FREE TO EDIT THE METHOD BELOW TO INCLUDE YOUR OTHER METRICS
    # YOU MAY ADD MORE METRICS FOR FORECASTING IF YOU DEEM NECESSARY

    @classmethod
    def compute_all_metrics(
        cls, portfolio_values, returns, trades, annual_risk_free_rate=0.02
    ):
        """Return a dictionary containing all of the metrics found in this
        class.
        """
        return {
            "Cumulative Return": cls.cumulative_return(portfolio_values),
            "Sharpe Ratio": cls.sharpe_ratio(returns, annual_risk_free_rate),
            "Max Drawdown": cls.max_drawdown(portfolio_values),
            "Win Rate (in %)": cls.win_rate(trades),
            "Profit Factor": cls.profit_factor(trades),
        }


class PortfolioEvaluator:
    """Class object that will evaluate your portfolio based on your backtesting
    strategy.
    """

    def __init__(
        self,
        starting_capital: int = 10000,
        allocation: Union[str, Dict[str, float]] = "equal",
    ) -> None:
        """Define the portfolio evaluator.

        Parameters:
        -----------
        starting_capital (int): The starting amount in $ of your overall
            modeling
        allocation (str | dict): The portfolio allocation in place. Note that
            there are only two valid inputs: 'equal' or a dictionary of
            stock weights that sum to 1. (default='equal')
        """
        self.starting_capital = starting_capital
        self.allocation = allocation

        self.stocks = {}

    def add_stock(self, stock_code: str, backtester: Type, model_name: str) -> None:
        """Add a stock to your portfolio, with the backtester.

        Parameters:
        -----------
        stock_code (str): The actual stock code.
        backtester (class instance): The backtester class. Ensure that your
            strategy has been run first.
        model_name (str): Name of your model.
        """
        self.stocks[stock_code] = {
            "backtester": backtester,
            "model": model_name,
            "metrics": backtester.compute_trade_metrics(),
        }

    def compute_portfolio_performance(self) -> Optional[Dict[str, Any]]:
        """Compute for portfolio performance."""
        n_stocks = len(self.stocks)

        if n_stocks == 0:
            print("No stocks in portfolio")
            return None

        if self.allocation == "equal":
            weights = {stock: 1 / n_stocks for stock in self.stocks.keys()}
        elif isinstance(self.allocation, dict):
            weights = self.allocation
        else:
            print("Allocation should either be 'equal' or a dictionary of weights.")
            return None

        if sum(weights.values()) != 1.0:
            raise ValueError("Weights should sum to 1.0.")

        # Get portfolio values over time
        max_length = max(
            len(data["backtester"].portfolio_values) for data in self.stocks.values()
        )
        portfolio_values = np.zeros(max_length)

        for stock, weight in weights.items():
            stock_backtest = self.stocks[stock]["backtester"]
            stock_values = np.array(stock_backtest.portfolio_values)

            # Scale stock values by weight
            allocated_capital = self.starting_capital * weight
            scaling_factor = allocated_capital / self.starting_capital

            weighted_values = stock_values * scaling_factor
            portfolio_values += weighted_values

        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        total_return = (
            (portfolio_values[-1] - self.starting_capital) / self.starting_capital * 100
        )
        sharpe_ratio = self._compute_sharpe(returns)
        max_drawdown = self._compute_mdd(portfolio_values)
        volatility = np.std(returns) * 100

        return {
            "initial_capital": self.starting_capital,
            "final_value": portfolio_values[-1],
            "total_return_%": total_return,
            "profit_$": portfolio_values[-1] - self.starting_capital,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown_%": max_drawdown,
            "volatility_%": volatility,
            "num_stocks": n_stocks,
            "portfolio_values": portfolio_values,
        }

    def show_report(self) -> None:
        """Show the porfolio report from your list of stocks."""
        print("=" * 80)
        print("PORTFOLIO EVALUATION REPORT")
        print("=" * 80)

        if len(self.stocks) == 0:
            print("No stocks in portfolio")
            return None

        # Show individual stock performance
        print("\nINDIVIDUAL STOCK PERFORMANCE:")
        print("-" * 80)
        print(
            f"{'Stock':<8} {'Model':<15} {'Return%':<12} {'Sharpe':<10} "
            f"{'MaxDD%':<10} {'WinRate%':<10} {'PF':<10}"
        )
        print("-" * 80)

        for stock, data in self.stocks.items():
            metrics = data["metrics"]
            print(
                f"{stock:<8} {data['model']:<15} "
                f"{metrics['Cumulative Return']:>9.2f}% "
                f"{metrics['Sharpe Ratio']:>9.2f} "
                f"{metrics['Max Drawdown']:>9.2f}%"
                f"{metrics['Win Rate (in %)']:>9.2f}%"
                f"{metrics['Profit Factor']:>9.2f}"
            )
        print("-" * 80)

        portfolio_metrics = self.compute_portfolio_performance()

        if portfolio_metrics:
            print(
                f"\nInitial Capital:       ${portfolio_metrics['initial_capital']:,.2f}"
            )
            print(f"Final Portfolio Value: ${portfolio_metrics['final_value']:,.2f}")
            print(f"Total Profit:          ${portfolio_metrics['profit_$']:,.2f}")
            print(f"Total Return:          {portfolio_metrics['total_return_%']:.2f}%")
            print(f"Sharpe Ratio:          {portfolio_metrics['sharpe_ratio']:.2f}")
            print(f"Maximum Drawdown:      {portfolio_metrics['max_drawdown_%']:.2f}%")
            print(f"Volatility:            {portfolio_metrics['volatility_%']:.2f}%")
        print("=" * 80)

    def _compute_sharpe(self, returns):
        return TradingMetrics().sharpe_ratio(returns)

    def _compute_mdd(self, portfolio_values):
        return TradingMetrics().max_drawdown(portfolio_values)
