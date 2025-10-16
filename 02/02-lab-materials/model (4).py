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

import shap
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from alibi.explainers import Counterfactual

class WhiteBox:
    """
    A white-box wrapper around the RandomForestClassifier.

    This class provides a simplified and transparent interface for fitting
    and predicting with a random forest model. It follows the scikit-learn
    estimator API design (fit/predict methods) and supports reproducibility
    and parallel computation.

    Parameters
    ----------
    random_state : int, default=39
        Controls the random seed for reproducibility of results.
    
    n_jobs : int or None, optional
        The number of CPU cores to use for parallel processing.
        If None, all available cores are used.

    Attributes
    ----------
    _model_ : RandomForestClassifier
        The underlying scikit-learn Random Forest model instance.
    """

    def __init__(self, random_state: int = 39, n_jobs: Optional[int] = None):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._model_ = RandomForestClassifier(
            n_estimators=500,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]):
        """
        Fit the Random Forest model to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        
        y : array-like of shape (n_samples,)
            Target labels corresponding to each training example.

        Returns
        -------
        self : WhiteBox
            Returns the fitted instance.
        """
        self._model_.fit(X, y)
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class labels for input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to predict on.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels for each input sample.
        """
        check_is_fitted(self._model_)
        return self._model_.predict(X)

# To Handle Class Imbalance
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


