from typing import Dict, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class FeatureBuilder:
    """
    Helper class to load character data, compute centrality features with EWMA,
    and prepare preprocessing pipelines for modeling.
    """

    def __init__(self, centrality_paths: Dict[str, str], alpha: float = 0.3):
        """
        Parameters
        ----------
        centrality_paths : Dict[str, str]
            Mapping from centrality names to CSV file paths.
        alpha : float, optional
            EWMA smoothing factor for centralities, default 0.3.
        """
        self.centrality_paths = centrality_paths
        self.alpha = alpha
        self.preprocessor = None

    def centrality_from_csv(self) -> pd.DataFrame:
        """
        Compute EWMA-smoothed centralities from CSVs and return merged DataFrame.
        """
        results = []
        for name, path in self.centrality_paths.items():
            df = pd.read_csv(path, index_col=0).iloc[:, :-1]
            df_filled = df.fillna(0)
            df_ewma = df_filled.T.ewm(alpha=self.alpha).mean().T
            df_final = df_ewma.iloc[:, -1].reset_index(name=name)
            df_final.rename(columns={"index": "Name"}, inplace=True)
            results.append(df_final)

        merged_df = results[0]
        for df_centrality in results[1:]:
            merged_df = merged_df.merge(df_centrality, on="Name")
        return merged_df

    def load_and_merge_features(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load character characteristics, merge with centralities, and split
        into features (X) and target (y).
        """
        df = pd.read_csv("data/characters/characteristics.csv")
        centrality_df = self.centrality_from_csv()
        df = df.merge(centrality_df, on="Name")

        X = df.drop(columns=["Name", "Use", "Own", "Pull Number"])
        y = df["Pull Number"]
        return X, y

    def build_preprocessor(self) -> ColumnTransformer:
        """
        Build a preprocessing pipeline for categorical, numeric, and passthrough columns.
        """
        categorical_cols = ["Element", "Weapon Type", "Body Type"]
        numeric_cols = ["Line Count"]
        passthrough_cols = [
            "Abyss Betweenness",
            "Abyss Eigenvector",
            "On-Field",
            "DPS",
            "Support",
            "Survivability",
            "Star",
            "Free",
        ]

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
                ("num", StandardScaler(), numeric_cols),
                ("passthrough", "passthrough", passthrough_cols),
            ]
        )
        return self.preprocessor


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
    
    @staticmethod
    def r2_score(
        y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]
    ) -> float:
        """
        Compute the R-squared (coefficient of determination) between predicted
        and actual values.

        Parameters
        ----------
        y_true : array-like
            True target values.
        y_pred : array-like
            Predicted target values.

        Returns
        -------
        float
            R-squared value.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot

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
            #"Directional Accuracy": cls.direction_accuracy(y_true, y_pred),
            "R2": cls.r2_score(y_true, y_pred)
        }