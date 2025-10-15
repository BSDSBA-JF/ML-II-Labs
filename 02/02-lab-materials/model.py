# model.py
from typing import Iterable, Optional, Union, Tuple, List

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted

from sklearn.metrics import (
    make_scorer, precision_score, recall_score, f1_score
)

# imbalanced-learn
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from imblearn.base import BaseSampler


# -----------------------------
# Helpers / custom transformers
# -----------------------------
class DaysFromStart(BaseEstimator, TransformerMixin):
    """Create global numeric 'days_from_start' from the date column and drop raw date."""
    def __init__(self, date_col: str):
        self.date_col = date_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.date_col] = pd.to_datetime(X[self.date_col], errors="coerce")
        global_min = X[self.date_col].min()
        X["days_from_start"] = (X[self.date_col] - global_min).dt.days.astype("float64")
        return X.drop(columns=[self.date_col])


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Optionally drop specific columns inside the pipeline."""
    def __init__(self, drop_cols: Optional[List[str]] = None):
        self.drop_cols = drop_cols or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.drop_cols, errors="ignore")


def _numeric_except(exclude_cols: List[str]):
    def _sel(X):
        num_cols = X.select_dtypes(include=["number"]).columns.tolist()
        return [c for c in num_cols if c not in exclude_cols]
    return _sel


def _make_ohe():
    """
    Return OneHotEncoder with dense output.
    Handles older sklearn versions where `sparse_output` may not exist.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # fallback for older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _coltx():
    """ColumnTransformer used after 'days_from_start' is created."""
    return ColumnTransformer(
        transformers=[
            ("mm_day", MinMaxScaler(), ["days_from_start"]),  # scale timeline to [0,1]
            ("num", StandardScaler(), _numeric_except(["days_from_start"])),
            ("cat", _make_ohe(), selector(dtype_include=["object", "category"])),
        ],
        remainder="drop",
    )


# -----------------------------
# Main estimator (no builders)
# -----------------------------
class ML2Detector(BaseEstimator, ClassifierMixin):
    """
    End-to-end estimator that includes:
      - optional column drop (safe even if you already dropped in the notebook)
      - time feature: global days_from_start (no id grouping)
      - ColumnTransformer: MinMax(days), StandardScaler(numerics), OHE(cats)
      - optional class-imbalance handling (samplers or class_weight)
      - pluggable classifier

    Parameters
    ----------
    date_col : str
        Raw timestamp/date column to transform into days_from_start.
    drop_cols : list[str] or None
        Columns to drop before preprocessing (ignored if already dropped upstream).
    classifier : str or sklearn estimator, default="logreg"
        "logreg" | "balanced_rf" | "easy_ensemble" | "rf" | sklearn-compatible estimator.
    imbalance : str or None
        None | "class_weight" | "ros" | "rus" | "smote" | "adasyn" | "smoteenn" | "smotetomek" | "cc_smote".
    random_state : int
        Random seed.
    """

    def __init__(
        self,
        date_col: str,
        drop_cols: Optional[List[str]] = None,
        classifier: Union[str, BaseEstimator] = "logreg",
        imbalance: Optional[str] = None,
        random_state: int = 1234,
    ):
        self.date_col = date_col
        self.drop_cols = drop_cols
        self.classifier = classifier
        self.imbalance = imbalance
        self.random_state = random_state

        self._pipeline_ = None  # set in fit

    # ------------- internal builders (kept inside the class) -------------
    def _make_preprocessor(self) -> SklearnPipeline:
        """Nested sklearn Pipeline (only used when NO sampler is present)."""
        steps = []
        # 1) global days_from_start (no id_col dependency)
        steps.append(("days_from_start", DaysFromStart(self.date_col)))
        # 2) optional drop (safe even if notebook already dropped)
        if self.drop_cols:
            steps.append(("drop_cols", ColumnDropper(self.drop_cols)))
        # 3) column-wise transforms
        steps.append(("coltx", _coltx()))
        return SklearnPipeline(steps=steps)

    def _preprocess_steps_flat(self):
        """Preprocessing steps as a flat list for ImbPipeline (sampler present)."""
        steps = [("days_from_start", DaysFromStart(self.date_col))]
        if self.drop_cols:
            steps.append(("drop_cols", ColumnDropper(self.drop_cols)))
        steps.append(("coltx", _coltx()))
        return steps

    def _make_classifier(self) -> BaseEstimator:
        if isinstance(self.classifier, str):
            if self.classifier == "logreg":
                return LogisticRegression(max_iter=2000, random_state=self.random_state)
            if self.classifier == "balanced_rf":
                return BalancedRandomForestClassifier(
                    n_estimators=400, random_state=self.random_state, n_jobs=-1
                )
            if self.classifier == "easy_ensemble":
                return EasyEnsembleClassifier(
                    n_estimators=20, random_state=self.random_state
                )
            if self.classifier == "rf":
                return RandomForestClassifier(
                    n_estimators=500, random_state=self.random_state, n_jobs=-1
                )
            raise ValueError(f"Unknown classifier string: {self.classifier}")
        return self.classifier  # custom sklearn estimator

    def _make_sampler(self) -> Optional[BaseSampler]:
        if self.imbalance is None:
            return None
        if self.imbalance == "ros":
            return RandomOverSampler(random_state=self.random_state)
        if self.imbalance == "rus":
            return RandomUnderSampler(random_state=self.random_state)
        if self.imbalance == "smote":
            return SMOTE(random_state=self.random_state)
        if self.imbalance == "adasyn":
            return ADASYN(random_state=self.random_state)
        if self.imbalance == "smoteenn":
            return SMOTEENN(random_state=self.random_state)
        if self.imbalance == "smotetomek":
            return SMOTETomek(random_state=self.random_state)
        if self.imbalance == "cc_smote":
            # two-stage: under-sample centroids then SMOTE
            return "cc_smote"  # sentinel, handled in fit
        if self.imbalance == "class_weight":
            return None  # handled by classifier params
        raise ValueError(f"Unknown imbalance option: {self.imbalance}")

    # --------------------------- API methods ---------------------------
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        sample_weight: Optional[Iterable[float]] = None,
    ):
        clf = self._make_classifier()
        sampler = self._make_sampler()

        # Handle class_weight request for supported classifiers
        if self.imbalance == "class_weight" and hasattr(clf, "set_params"):
            try:
                clf.set_params(class_weight="balanced")
            except ValueError:
                pass

        if sampler is None:
            # No sampler → we can safely use a regular sklearn Pipeline
            pre = self._make_preprocessor()
            self._pipeline_ = SklearnPipeline([
                ("preprocess", pre),
                ("clf", clf),
            ])
        else:
            # With sampler → flatten preprocessing steps (imblearn forbids nested Pipelines)
            steps = self._preprocess_steps_flat()
            if sampler == "cc_smote":
                steps += [
                    ("cc", ClusterCentroids(random_state=self.random_state)),
                    ("smote", SMOTE(random_state=self.random_state)),
                    ("clf", clf),
                ]
            else:
                steps += [
                    ("sampler", sampler),
                    ("clf", clf),
                ]
            self._pipeline_ = ImbPipeline(steps)

        if sample_weight is None:
            self._pipeline_.fit(X, y)
        else:
            # Try to pass sample_weight to the final estimator if supported
            try:
                self._pipeline_.fit(X, y, clf__sample_weight=sample_weight)
            except TypeError:
                self._pipeline_.fit(X, y)

        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        check_is_fitted(self, "_pipeline_")
        return self._pipeline_.predict(X)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        check_is_fitted(self, "_pipeline_")
        if hasattr(self._pipeline_[-1], "predict_proba"):
            return self._pipeline_.predict_proba(X)
        if hasattr(self._pipeline_[-1], "decision_function"):
            scores = self._pipeline_.decision_function(X)
            from scipy.special import expit
            if scores.ndim == 1:
                p1 = expit(scores)
                return np.vstack([1 - p1, p1]).T
        raise AttributeError("Final estimator has neither predict_proba nor decision_function.")

    # ---------- optional: CV experiment method ----------
    def experiment_fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        sampler: Optional[BaseSampler] = None,
        refit_metric: str = "f1",
        n_splits: int = 5,
        verbose: int = 1,
    ) -> dict:
        """
        Quick CV experiment wrapper that swaps a sampler (or none) and scores.
        Returns GridSearchCV.cv_results_ dict.
        """
        from sklearn.model_selection import StratifiedKFold, GridSearchCV

        clf = self._make_classifier()
        steps = self._preprocess_steps_flat()
        Pipe = ImbPipeline if sampler is not None else SklearnPipeline
        if sampler is not None and sampler != "cc_smote":
            steps.append(("sampler", sampler))
        elif sampler == "cc_smote":
            steps += [
                ("cc", ClusterCentroids(random_state=self.random_state)),
                ("smote", SMOTE(random_state=self.random_state)),
            ]
        steps.append(("clf", clf))
        pipe = Pipe(steps)

        scorers = {
            "precision": make_scorer(precision_score, average="macro", zero_division=0),
            "recall": make_scorer(recall_score, average="macro", zero_division=0),
            "f1": make_scorer(f1_score, average="macro", zero_division=0),
        }

        param_grid = {}
        if isinstance(clf, LogisticRegression):
            param_grid = {"clf__C": [0.1, 1.0, 10.0]}
        elif isinstance(clf, RandomForestClassifier):
            param_grid = {"clf__n_estimators": [300, 500]}
        elif isinstance(clf, BalancedRandomForestClassifier):
            param_grid = {"clf__n_estimators": [200, 400]}

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid or {"clf__dummy": [None]},
            scoring=scorers,
            refit=refit_metric,
            cv=cv,
            n_jobs=-1,
            verbose=verbose,
        )
        search.fit(X, y)
        return search.cv_results_
