# model.py
from typing import List, Union, Optional
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# --- Custom transformer: date -> days_from_start (per id), then drop 'date' ---
class DaysFromStart(BaseEstimator, TransformerMixin):
    def __init__(self, id_col: str, date_col: str):
        self.id_col = id_col
        self.date_col = date_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.date_col] = pd.to_datetime(X[self.date_col], errors="coerce")
        X["days_from_start"] = (
            X.groupby(self.id_col)[self.date_col]
             .transform(lambda s: (s - s.min()).dt.days)
             .astype("float64")
        )
        return X.drop(columns=[self.date_col])


# --- Helper: select numeric columns except a given list ---
def numeric_except(exclude_cols: List[str]):
    def _sel(X):
        num_cols = X.select_dtypes(include=["number"]).columns.tolist()
        return [c for c in num_cols if c not in exclude_cols]
    return _sel


# --- Preprocessing-only pipeline (exported) ---
def make_preprocessor(id_col: str, date_col: str) -> Pipeline:
    coltx = ColumnTransformer(
        transformers=[
            ("mm_day", MinMaxScaler(), ["days_from_start"]),                          # scale time index
            ("num", StandardScaler(), numeric_except(exclude_cols=["days_from_start"])),
            ("cat", OneHotEncoder(handle_unknown="ignore"),
             selector(dtype_include=["object", "category"]))
        ],
        remainder="drop",
    )
    return Pipeline(steps=[
        ("days_from_start", DaysFromStart(id_col=id_col, date_col=date_col)),
        ("coltx", coltx),
    ])


# --- Full pipeline (preprocess + classifier) (exported) ---
def make_model_pipeline(
    id_col: str,
    date_col: str,
    classifier: Optional[object] = None,
    random_state: int = 1234,
) -> Pipeline:
    """
    Build a full sklearn Pipeline: preprocessing + classifier.
    If no classifier is provided, defaults to LogisticRegression.
    """
    if classifier is None:
        classifier = LogisticRegression(random_state=random_state, max_iter=1000)

    return Pipeline(steps=[
        ("preprocess", make_preprocessor(id_col=id_col, date_col=date_col)),
        ("clf", classifier),
    ])
