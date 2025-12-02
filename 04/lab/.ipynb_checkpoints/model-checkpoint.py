import numpy as np
import pandas as pd
from typing import List, Optional, Union

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.utils.validation import check_is_fitted

import torch
import torch.nn as nn

class GenshinModel(BaseEstimator, RegressorMixin):
    def __init__(self, model_type="rf", random_state=42,
                 n_estimators=100, max_depth=None,
                 n_neighbors=5, nn_params=None):
        # Do not modify any parameter here
        self.model_type = model_type
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_neighbors = n_neighbors
        self.nn_params = nn_params
        self._model_ = None
        self.preprocessor = None

    def fit(self, X, y, categorical_cols=None):
        model_type = self.model_type.lower()  # safe to lowercase here

        # Preprocessing
        if isinstance(X, pd.DataFrame):
            if categorical_cols is None:
                categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
                ],
                remainder="passthrough"
            )
            X_transformed = self.preprocessor.fit_transform(X)
        else:
            X_transformed = X

        # Model creation using init params
        if model_type == "rf":
            self._model_ = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
        elif model_type == "knn":
            self._model_ = KNeighborsRegressor(n_neighbors=self.n_neighbors)
        elif model_type == "lr":
            self._model_ = LinearRegression()
        elif model_type == "nn":
            input_dim = X_transformed.shape[1]
            self._model_ = GenshinNetwork(n_input=input_dim, **(self.nn_params or {}))
            self._train_nn(X_transformed, y)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        if model_type in ["rf", "knn", "lr"]:
            self._model_.fit(X_transformed, y)

        return self

    def predict(self, X):
        check_is_fitted(self, "_model_")
        if self.preprocessor and isinstance(X, pd.DataFrame):
            X_transformed = self.preprocessor.transform(X)
        else:
            X_transformed = X

        if self.model_type.lower() in ["rf", "knn", "lr"]:
            return self._model_.predict(X_transformed)
        elif self.model_type.lower() == "nn":
            self._model_.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X_transformed, dtype=torch.float32)
                return self._model_(X_tensor).numpy()

class GenshinNetwork(nn.Module):
    """Feedforward neural network for regression."""
    
    def __init__(self, n_input=26, n_hidden_layers=4, n_hidden_nodes=10, dropout_rate=0.2):
        super().__init__()
        layers = []
        in_features = n_input
        
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(in_features, n_hidden_nodes))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_features = n_hidden_nodes
        
        layers.append(nn.Linear(in_features, 1))  # Output layer
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
