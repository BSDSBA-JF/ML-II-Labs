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