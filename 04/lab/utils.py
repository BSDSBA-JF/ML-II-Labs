# Standard library
from typing import Tuple

# Data handling
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# PyTorch
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


def train_and_evaluate_neural_network(
    alpha: float,
    n_hidden_layers: int,
    n_hidden_nodes: int,
    dropout_rate: float,
    epochs: int = 10_000,
    batch_size: int = 42,
) -> Tuple[nn.Module, float]:
    """
    Train and evaluate a neural network model for predicting Genshin pull counts
    using centrality-based and character-based features.

    This function treats the exponential weighting factor (alpha) used in the
    computation of centrality measures as a tunable hyperparameter, along with
    the neural network architecture (number of hidden layers, number of nodes
    per layer, and dropout rate). The model is trained using mini-batch gradient
    descent and evaluated on a held-out validation set.

    Parameters
    ----------
    alpha : float
        Exponential weighting factor used in computing centrality features.
        Higher values place more emphasis on recent events.
    n_hidden_layers : int
        Number of hidden layers in the neural network.
    n_hidden_nodes : int
        Number of neurons in each hidden layer.
    dropout_rate : float
        Dropout probability applied between hidden layers to reduce overfitting.
    epochs : int, default=10_000
        Number of training epochs.
    batch_size : int, default=42
        Number of samples per mini-batch.

    Returns
    -------
    Tuple[nn.Module, float]
        - Trained PyTorch model (`nn.Module`).
        - Validation loss (`float`) computed as mean squared error (MSE) on the
          hold-out validation set.
    """

    # --- Prepare centrality features ---
    abyss_ev = centrality_from_csv('data/abyss_eigenvector.csv', 'Abyss Eigenvector', alpha=alpha).iloc[:, 1]
    abyss_between = centrality_from_csv('data/abyss_betweenness.csv', 'Abyss Betweenness', alpha=alpha).iloc[:, 1]
    df_characteristics = pd.read_csv('data/characters/characteristics.csv')
    
    df_merged = pd.concat([df_characteristics, abyss_ev, abyss_between], axis=1)
    X = df_merged.drop(columns=["Name", "Use", "Own", "Pull Number"])
    y = df_merged["Pull Number"]
    
    # Train/validation split
    X_train_part, X_val, y_train_part, y_val = train_test_split(X, y, test_size=0.2, random_state=37)
    
    # Preprocessing
    # Categorical and numeric columns
    categorical_cols = ["Element", "Weapon Type", "Body Type"]
    standardized_cols = ['Line Count']
    passthrough_cos = ['Abyss Betweenness', 'Abyss Eigenvector', 'On-Field', 'DPS', 'Support', 'Survivability', 'Star', 'Free']

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), standardized_cols),
            ('passthrough', 'passthrough', passthrough_cos)
        ]
    )

    X_train_preprocessed = preprocessor.fit_transform(X_train_part)
    X_val_preprocessed = preprocessor.transform(X_val)
    
    X_torch = torch.tensor(X_train_preprocessed, dtype=torch.float32)
    y_torch = torch.tensor(y_train_part.values, dtype=torch.float32).unsqueeze(1)
    
    val_X_torch = torch.tensor(X_val_preprocessed, dtype=torch.float32)
    val_y_torch = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
    
    dataset = TensorDataset(X_torch, y_torch)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Model, loss, optimizer
    model = GenshinNetwork(n_input=X_torch.shape[1],
                            n_hidden_layers=n_hidden_layers,
                            n_hidden_nodes=n_hidden_nodes,
                            dropout_rate=dropout_rate)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
    
    # Evaluate on validation
    model.eval()
    with torch.no_grad():
        val_preds = model(val_X_torch)
        val_loss = criterion(val_preds, val_y_torch).item()
    
    return model, val_loss

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