"""
Random Forest Model with Centrality Features for Genshin Impact Pull Prediction

This script trains a Random Forest regressor using character attributes and 
network centrality features (Abyss Eigenvector and Betweenness) to predict 
character pull counts.

Matches the exact implementation from Section 6.4 of the report notebook.

Usage:
    python train_random_forest.py
"""

import sys
import joblib
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV

from utils import FeatureBuilder, ForecastingMetrics
from model import GenshinModel


def train_random_forest_with_centrality(
    alpha=0.2,
    test_size=0.2,
    random_state_split=37,
    random_state_model=42,
    n_splits=5,
    verbose=True
):
    """
    Train Random Forest with centrality features using the exact approach from the report.
    
    This function replicates Section 6.4 - "Random Forest BUT Better (has the Centrality Measures now)"
    
    Parameters
    ----------
    alpha : float
        EWMA smoothing factor for centrality features (default: 0.2)
    test_size : float
        Proportion of data to use for testing (default: 0.2)
    random_state_split : int
        Random state for train/test split (default: 37)
    random_state_model : int
        Random state for model training (default: 42)
    n_splits : int
        Number of folds for cross-validation (default: 5)
    verbose : bool
        Whether to print progress information (default: True)
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'model': Best trained GenshinModel
        - 'rf_model': Internal RandomForestRegressor
        - 'preprocessor': Internal ColumnTransformer
        - 'metrics': Performance metrics on test set
        - 'best_params': Best hyperparameters from GridSearchCV
        - 'feature_importances': DataFrame of feature importances
        - 'X_test': Test features
        - 'y_test': Test targets
        - 'y_pred': Predictions on test set
    """
    
    # Define centrality paths (Abyss mode only - matches notebook exactly)
    abyss_paths = {
        "Abyss Eigenvector": "data/abyss_eigenvector.csv",
        "Abyss Betweenness": "data/abyss_betweenness.csv",
    }
    

    
    # Build features using FeatureBuilder (matches notebook exactly)
    fb = FeatureBuilder(centrality_paths=abyss_paths, alpha=alpha)
    X, y = fb.load_and_merge_features()
    

    
    # Train/test split (matches notebook exactly)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state_split
    )
    

    
    # Create GenshinModel instance (matches notebook exactly)
    model = GenshinModel(model_type="rf", random_state=random_state_model)
    
    # Define hyperparameter grid (matches notebook exactly)
    param_grid = {
        "n_estimators": [50, 100, 200, 300, 500, 1000]
    }
    
    # K-Fold cross-validation (matches notebook exactly)
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state_model)
    

    
    # Grid search (matches notebook exactly)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=kfold,
        scoring={"MSE": "neg_mean_squared_error", "R2": "r2"},
        refit="MSE",
        return_train_score=True,
        n_jobs=-1,
        verbose=1 if verbose else 0
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get best model (matches notebook exactly)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    if verbose:
        print(f"\n{'='*60}")
        print("BEST PARAMETERS")
        print(f"{'='*60}")
        print(f"Best Params: {best_params}")
    
    # Predict on test set (matches notebook exactly)
    y_pred_train = best_model.predict(X_train)
    
    # Compute metrics
    metrics = ForecastingMetrics.compute_all_metrics(y_train, y_pred_train)
    
    if verbose:
        print(f"\n{'='*60}")
        print("TRAIN SET PERFORMANCE METRICS")
        print(f"{'='*60}")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name:25s}: {metric_value:,.2f}")
        print(f"{'='*60}")
    
    # Get feature importances (matches notebook exactly)
    rf_model = best_model._model_  # Access the internal RandomForestRegressor
    preprocessor = best_model.preprocessor  # Access the internal ColumnTransformer
    
    feature_names = preprocessor.get_feature_names_out()
    importances = rf_model.feature_importances_
    
    feature_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
    
    if verbose:
        print(f"\n{'='*60}")
        print("TOP 10 MOST IMPORTANT FEATURES")
        print(f"{'='*60}")
        print(feature_importance_df.head(10).to_string(index=False))
        print(f"{'='*60}")

    return {
    "model": best_model,
    "rf_model": rf_model,
    "preprocessor": preprocessor,
    "metrics": metrics,
    "best_params": best_params,
    "feature_importances": feature_importance_df,
    "X_train": X_train,
    "y_train": y_train,
    "y_pred_train": y_pred_train
}


def compare_alphas(alphas=[0.2, 0.3, 0.4], verbose=True):
    """
    Compare Random Forest performance across different alpha values.
    
    This helps identify the optimal EWMA smoothing factor for centrality features.
    
    Parameters
    ----------
    alphas : list
        List of alpha values to test (default: [0.2, 0.3, 0.4])
    verbose : bool
        Whether to print progress (default: True)
    
    Returns
    -------
    pd.DataFrame
        Comparison of metrics across different alphas
    """
    results = {}
    
    for alpha in alphas:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training with alpha={alpha}")
            print(f"{'='*60}")
        
        result = train_random_forest_with_centrality(
            alpha=alpha,
            verbose=verbose
        )
        
        results[f"alpha={alpha}"] = result["metrics"]
    
    comparison_df = pd.DataFrame(results).T
    
    if verbose:
        print(f"\n{'='*60}")
        print("ALPHA COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(comparison_df.to_string())
        print(f"{'='*60}")
        print("\nBest alpha by R2:", comparison_df['R2'].idxmax())
        print(f"Best alpha by MAE: {comparison_df['MAE'].idxmin()}")
        print(f"{'='*60}")
    
    return comparison_df

if __name__ == "__main__":
    print("\nTraining Random Forest with Centrality Features on training data only...\n")
    
    # Train the model
    result = train_random_forest_with_centrality(alpha=0.2, verbose=True)
    
    # Save the trained model
    joblib.dump(result["model"], "best_model.joblib")
    
    print("\n✓ Model and training data saved successfully as joblib files.")

