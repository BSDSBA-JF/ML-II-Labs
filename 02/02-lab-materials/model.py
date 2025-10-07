# --- Python typing ---
from typing import Union, Optional, Dict, Any

# --- Numerical / Data ---
import numpy as np
import pandas as pd

# --- Scikit-learn ---
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.utils.validation import check_is_fitted

# --- Imbalanced-learn ---
from imblearn.pipeline import Pipeline as ImbPipeline  # for handling sampling inside pipeline
from imblearn.base import BaseSampler                  # base class for SMOTE, ADASYN, etc.
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Preprocessing
# If your preprocessor is e.g., a ColumnTransformer or StandardScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Columns for preprocessing
minmax_cols = ['age_days', 'feeding_frequency_per_day', 'urine_output_count', 'stool_count']
standardize_cols = [
    'gestational_age_weeks', 'weight_kg', 'length_cm', 'head_circumference_cm',
    'temperature_c', 'heart_rate_bpm', 'respiratory_rate_bpm', 'oxygen_saturation',
    'jaundice_level_mg_dl'
]
categorical_cols = ['gender', 'feeding_type', 'immunizations_done', 'reflexes_normal']


# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('minmax', MinMaxScaler(), minmax_cols),
        ('standard', StandardScaler(), standardize_cols),
        ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='drop'  # all other columns (unlisted) are automatically dropped
)

class WhiteBox(BaseEstimator, ClassifierMixin):    
    def __init__(self, preprocessor=preprocessor, random_state: int = 39, n_jobs: Optional[int] = None):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._model_ = RandomForestClassifier(
            n_estimators=500,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        self.preprocessor = preprocessor
        steps = []
        steps.append(("preprocess", self.preprocessor))
        steps.append(("clf", self._model_)) 
        self.pipeline = ImbPipeline(steps=steps)


    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
    ):
        """
        Fit the model using a training X and y.

        Parameters:
        -----------
        X (np.ndarray | pd.DataFrame): The training data
        y (np.ndarray | pd.Sereis): The corresponding targets
        sample_weight (iterable): Sample class weights (default=None)
        """

        self.pipeline.fit(X, y)
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict the outcome from data X."""
        check_is_fitted(self, "pipeline")
        return self.pipeline.predict(X)

    def fit_predict(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]
    ) -> np.ndarray:
        """
        Fit the model using a training X and y, and predict the outcome using
        X. Note that this would give the predicted labels during training.
        """
        return self.fit(X, y).predict(X)
    
    def cross_validate_pipeline(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        test_model,
        sample_strategy: Optional[BaseSampler] = None,
        n_splits: int = 5,
        random_state: int = 42,
        n_jobs: int = 1,
    ) -> Dict[str, any]:
        """
        Perform cross-validation on a pipeline that includes preprocessing, 
        optional resampling, and a classifier. Returns training and testing scores 
        for multiple metrics.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Feature matrix for training and validation.
        y : np.ndarray or pd.Series
            Target vector (binary labels, e.g., 'At Risk' and 'Healthy').
        test_model : estimator object
            The classifier to evaluate. Must follow scikit-learn API (fit/predict).
        sample_strategy : BaseSampler, optional (default=None)
            Optional sampling technique from imbalanced-learn (e.g., SMOTE, ADASYN,
            RandomOverSampler, RandomUnderSampler) to handle class imbalance.
        n_splits : int, default=5
            Number of folds for Stratified K-Fold cross-validation.
        random_state : int, default=42
            Random seed for reproducibility in splitting and sampling.
        n_jobs : int, default=1
            Number of CPU cores to use during cross-validation. -1 uses all cores.

        Returns
        -------
        cv_results : dict
            Dictionary containing training and testing scores for each fold and metric.
            Keys include: 'train_precision', 'test_precision', 'train_recall', 
            'test_recall', 'train_recall_at_risk', 'test_recall_at_risk', 
            'train_recall_healthy', 'test_recall_healthy', 'train_f1', 'test_f1'.
        """
        # Define scorers
        scorers = {
            "precision": make_scorer(precision_score, zero_division=0, average="binary", pos_label="At Risk"),
            "recall": make_scorer(recall_score, zero_division=0, average="binary", pos_label="At Risk"),
            "recall_at_risk": make_scorer(recall_score, zero_division=0, pos_label="At Risk"),
            "recall_healthy": make_scorer(recall_score, zero_division=0, pos_label="Healthy"),
            "f1": make_scorer(f1_score, zero_division=0, average="binary", pos_label="At Risk"),
        }

        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        # Build pipeline
        steps = []
        if self.preprocessor is not None:
            steps.append(("preprocess", self.preprocessor))
        if sample_strategy is not None:
            steps.append(("sampler", sample_strategy))
        steps.append(("clf", test_model))

        pipeline = ImbPipeline(steps=steps)

        # Fit pipeline (optional, not strictly needed for cross_validate)
        pipeline.fit(X, y)

        # Run cross-validation
        cv_results = cross_validate(
            estimator=pipeline,
            X=X,
            y=y,
            cv=skf,
            scoring=scorers,
            n_jobs=n_jobs,
            return_train_score=True
        )

        return cv_results
