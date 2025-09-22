from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    make_scorer,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.utils.validation import check_is_fitted

from imblearn.base import BaseSampler


class FraudDetector(BaseEstimator, ClassifierMixin):
    """
    A custom fraud detection model built on top of a Random Forest classifier.

    This class provides an extensible framework for experimenting with
    imbalanced classification problems, particularly in fraud detection.
    It integrates with scikit-learn and imbalanced-learn, allowing
    seamless use of pipelines, cross-validation, and custom scorers.

    Key Features
    ------------
    - Uses a Random Forest classifier with 500 estimators as a baseline.
    - Supports rebalancing strategies (e.g., SMOTE, undersampling,
      hybrid methods) via `experiment_fit`.
    - Provides custom scoring metrics tailored for fraud detection:
      * Absolute Net Value Gained (total amount of fraud correctly flagged).
      * Relative Net Value Gained (proportion of fraudulent value caught).
    - Compatible with scikit-learn model selection tools such as
      `GridSearchCV` and `Pipeline`.

    Parameters
    ----------
    random_state : int, default=39
        Random seed for reproducibility of the Random Forest model.
    n_jobs : int, optional
        Number of CPU cores to use for training the Random Forest.
        If None, scikit-learn defaults are used.

    Attributes
    ----------
    _model_ : RandomForestClassifier
        The underlying Random Forest classifier.
    """
    
    def __init__(self, random_state: int = 39, n_jobs: Optional[int] = None):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._model_ = RandomForestClassifier(
            n_estimators=500,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        sample_weight: Optional[Iterable[float]] = None,
    ):
        """
        Fit the model using a training X and y.

        Parameters:
        -----------
        X (np.ndarray | pd.DataFrame): The training data
        y (np.ndarray | pd.Sereis): The corresponding targets
        sample_weight (iterable): Sample class weights (default=None)
        """

        X_balanced, y_balanced = self._handle_imbalance(X, y) # you can modify the parameters here if needed
        if sample_weight is None:
            self._model_.fit(X_balanced, y_balanced)
        else:
            self._model_.fit(X_balanced, y_balanced, sample_weight=sample_weight)
        return self

    def experiment_fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        sample_strategy: Optional[BaseSampler] = None,
        sample_weight: Optional[Iterable[float]] = None,
    ) -> dict[str, np.ndarray]:
        """
        Fit an imbalanced-learn pipeline on the training data.

        This method is designed for experimentation. It allows you to
        easily switch between different class imbalance handling
        strategies (e.g., SMOTE, RandomUnderSampler) and apply optional
        sample weights when fitting the model.

        Parameters
        ----------
        X : np.ndarray | pd.DataFrame
            Training data.
        y : np.ndarray | pd.Series
            Target labels.
        sample_strategy : BaseSampler, optional
            Resampling strategy (default=None).
        sample_weight : iterable, optional
            Sample weights for fitting (default=None).

        Returns
        -------
        dict
            Contains the key:value pair of str to the mean validation scores
        """
        # Build pipeline
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=36)
    
        # Build pipeline
        steps = []
        if sample_strategy is not None:
            steps.append(("sampler", sample_strategy))  # e.g., SMOTE(), RandomUnderSampler()
        steps.append(("clf", self._model_))

        pipeline = Pipeline(steps=steps)

        # Define the scorers
        fcr_scorer = make_scorer(recall_score, pos_label=1)

        # Detection: macro recall
        detection_scorer = make_scorer(recall_score, average="macro")

        scorers = {
            "precision": make_scorer(precision_score, average="macro", zero_division=0),
            "recall": make_scorer(recall_score, average="macro", zero_division=0),
            "f1": make_scorer(f1_score, average="macro", zero_division=0),
            "fcr": fcr_scorer,
            "detection": detection_scorer,
            "abs_net_value": FraudDetector.abs_net_value_scorer,
            "rel_net_value": FraudDetector.rel_net_value_scorer,
        }

        # GridSearchCV
        if sample_weight is not None:
            param_grid = {
                "clf__n_estimators": [500],
                "clf__class_weight": ['balanced']
            }
        else:
            param_grid = {
                "clf__n_estimators": [500],
            }
            
        search = GridSearchCV(
            estimator=pipeline,
            scoring=scorers,
            param_grid=param_grid,
            refit="fcr",       
            cv=skf,
            n_jobs=-1,
            verbose=1
        )

        search.fit(X, y)

        return search.cv_results_ 


    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict the outcome from data X."""
        check_is_fitted(self, "_model_")
        return self._model_.predict(X)

    def fit_predict(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]
    ) -> np.ndarray:
        """
        Fit the model using a training X and y, and predict the outcome using
        X. Note that this would give the predicted labels during training.
        """
        return self.fit(X, y).predict(X)

    def _handle_imbalance(self, X, y):
        return X, y

    #### END MODIFY THIS METHOD

    # Helper functions
    @staticmethod
    def abs_net_value_scorer(
        estimator: BaseEstimator,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> float:
        """
        Custom scoring function to calculate the absolute net value gained
        from fraud detection.
    
        This metric sums the transaction amounts of fraud cases that are 
        correctly identified by the model.
    
        Parameters
        ----------
        estimator : BaseEstimator
            The trained estimator with a `predict` method.
        X : Union[pd.DataFrame, np.ndarray]
            Feature set containing transaction data. The last column is 
            assumed to represent the transaction amount.
        y : Union[pd.Series, np.ndarray]
            True class labels, where 1 indicates fraud and 0 indicates 
            non-fraud.
    
        Returns
        -------
        float
            The total transaction amount of correctly detected fraudulent 
            transactions.
        """
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        y_pred = estimator.predict(X)
        amount = X.iloc[:, -1]  # last column as transaction value
        caught_value = amount[(y == 1) & (y_pred == 1)].sum()
        return caught_value
    
    @staticmethod
    def rel_net_value_scorer(
        estimator: BaseEstimator,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> float:
        """
        Custom scoring function to calculate the relative net value gained
        from fraud detection.
    
        This metric represents the proportion of total fraudulent transaction 
        value that the model successfully detects.
    
        Parameters
        ----------
        estimator : BaseEstimator
            The trained estimator with a `predict` method.
        X : Union[pd.DataFrame, np.ndarray]
            Feature set containing transaction data. The last column is 
            assumed to represent the transaction amount.
        y : Union[pd.Series, np.ndarray]
            True class labels, where 1 indicates fraud and 0 indicates 
            non-fraud.
    
        Returns
        -------
        float
            The ratio of detected fraudulent transaction value to the total 
            fraudulent transaction value. Returns 0 if there are no fraud 
            cases in `y`.
        """
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
    
        y_pred = estimator.predict(X)
        amount = X.iloc[:, -1]  # last column as transaction value
        caught_value = amount[(y == 1) & (y_pred == 1)].sum()
        total_fraud = amount[y == 1].sum()
    
        return float(caught_value / total_fraud) if total_fraud != 0 else 0.0