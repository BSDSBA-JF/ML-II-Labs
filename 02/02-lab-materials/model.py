# model.py
from typing import Iterable, Optional, Union, Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
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

import dice_ml as dice
from dice_ml import Data, Model, Dice
from dice_ml.utils import helpers

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
        
    def _make_preprocessor(self, numerical_cols: List[str], categorical_cols: List[str]):
        """
        Creates a ColumnTransformer to preprocess numerical and categorical columns.
        """
        self.numerical_columns = numerical_cols
        self.categorical_columns = categorical_cols
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
            ],
            remainder='drop'
        )
        preprocessor.set_output(transform="pandas")
        return preprocessor

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None
    ):
        # Encode target
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        # Determine columns if not passed
        if numerical_cols is None:
            numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
        if categorical_cols is None:
            categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

        # Preprocess
        self.preprocessor = self._make_preprocessor(numerical_cols, categorical_cols)
        X_transformed = self.preprocessor.fit_transform(X)

        # Fit model
        self._model_.fit(X_transformed, y_encoded)

        # Save feature names
        self.feature_names = X_transformed.columns.tolist() if isinstance(X_transformed, pd.DataFrame) else [
            f'Feature_{i}' for i in range(X_transformed.shape[1])
        ]

        # Save training data (for counterfactuals)
        self.training_df = X.copy()
        self.training_df.fillna(0, inplace=True)
        self.training_df["target"] = y_encoded

        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class labels for input samples.
        """
        check_is_fitted(self._model_)
        
        # Transform X using the fitted preprocessor
        X_transformed = self.preprocessor.transform(X)
        
        # Predict using the transformed features
        y_pred_encoded = self._model_.predict(X_transformed)
        
        # Decode labels back to original strings if label_encoder exists
        if hasattr(self, 'label_encoder') and self.label_encoder is not None:
            return self.label_encoder.inverse_transform(y_pred_encoded)
        else:
            return y_pred_encoded

    
    def treeshap(self, X_explain: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Calculate SHAP values for tree-based models (TreeSHAP).

        TreeSHAP is a fast and exact method for computing SHAP values for
        tree-based models, explaining the contribution of each feature to the
        model's output (logit space, often log-odds for classification).

        Parameters
        ----------
        X_explain : array-like of shape (n_samples, n_features)
            Data for which to calculate SHAP values. Should be feature data, 
            not the target variable.

        Returns
        -------
        shap_values : ndarray of shape (n_samples, n_features) or 
                      (n_samples, n_classes, n_features)
            The SHAP values, where the last dimension corresponds to the features.
            For binary classification, it often returns (n_samples, 2, n_features), 
            but for interpretation, the positive class (index 1) is typically used.
        """

        explainer = shap.TreeExplainer(self._model_)
        shap_values = explainer.shap_values(X_explain)
        
        return shap_values
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class probabilities for input samples.
        """
        
        # Transform X using the fitted preprocessor
        X_transformed = self.preprocessor.transform(X)
        
        # Predict probabilities using the transformed features
        return self._model_.predict_proba(X_transformed)
    
    def counterfactual(
        self, 
        query_instance: Union[np.ndarray, pd.Series, pd.DataFrame], 
        target_class: int = 0, 
        num_cfs: int = 5
    ) -> Optional[pd.DataFrame]:
        """
        Generate diverse counterfactual explanations using DiCE.

        A counterfactual explanation shows what the feature values would need 
        to be changed to in order to change the model's prediction 
        (e.g., from 'not approved' to 'approved').

        Parameters
        ----------
        query_instance : array-like of shape (n_features,) or (1, n_features) 
            The single instance to explain (the 'factual' example).
        
        target_class : int, default=0
            The desired output class for the counterfactuals to achieve.
        
        num_cfs : int, default=5
            The number of diverse counterfactuals to generate.

        Returns
        -------
        counterfactuals : pd.DataFrame or None
            A DataFrame containing the generated counterfactuals, or None if an
            error occurs or DiCE fails to find CEs.
        """
        

        dice_model = dice.model.Model(model=self, backend="sklearn")

        # DiCE Data - uses the training data for finding viable counterfactuals
        dice_data = dice.Data(
            dataframe=self.training_df, 
            continuous_features=self.numerical_columns,
            categorical_features=self.categorical_columns, 
            outcome_name='target' 
        )

        explainer = dice.Dice(
            dice_data, 
            dice_model, 
            method='kdtree' 
        )

        dice_exp = explainer.generate_counterfactuals(
            query_instance, 
            total_CFs=num_cfs, 
            desired_class=target_class, 
        )
        

        if dice_exp:
            return dice_exp.visualize_as_dataframe(
                show_only_changes=True
            )
        else:
            print("Warning: DiCE could not find any counterfactuals.")
            return None
    
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


