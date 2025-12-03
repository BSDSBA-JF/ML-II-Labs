"""
Inference script for the Random Forest + Centrality Genshin model.

- Loads the trained model from joblib.
- Rebuilds features using FeatureBuilder (same as train.py).
- Recreates the train/test split with the same random_state.
- Draws a random sample from the test set.
- Runs model.predict on that sample and prints predictions vs true values.
"""

import os
import sys

# Make sure we can import from the project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import FeatureBuilder
from model import GenshinModel  # not strictly needed at runtime, but nice for typing / clarity


# ----------------------------------------------------------------------
# Configuration (must match train.py)
# ----------------------------------------------------------------------
MODEL_PATH = "best_model.joblib"
ALPHA = 0.2
TEST_SIZE = 0.2
RANDOM_STATE_SPLIT = 37

ABYSS_PATHS = {
    "Abyss Eigenvector": "data/abyss_eigenvector.csv",
    "Abyss Betweenness": "data/abyss_betweenness.csv",
}


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def load_model(model_path: str = MODEL_PATH) -> GenshinModel:
    """
    Load the trained GenshinModel (Random Forest with centrality) from disk.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    model = joblib.load(model_path)
    return model


def load_features(
    alpha: float = ALPHA,
    test_size: float = TEST_SIZE,
    random_state_split: int = RANDOM_STATE_SPLIT,
):
    """
    Rebuild features using FeatureBuilder and perform the same train/test split
    as in train.py. Returns (X_train, X_test, y_train, y_test).
    """
    fb = FeatureBuilder(centrality_paths=ABYSS_PATHS, alpha=alpha)
    X, y = fb.load_and_merge_features()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state_split,
    )

    return X_train, X_test, y_train, y_test


def random_sample_predictions(
    model: GenshinModel,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_samples: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Take a random sample from the test set and run model.predict on it.

    Returns a DataFrame containing:
    - original feature columns
    - true target ('y_true')
    - predicted target ('y_pred')
    """
    # Sample rows from X_test
    sample_X = X_test.sample(n=n_samples, random_state=random_state)
    sample_idx = sample_X.index

    # Align y_test with that sample (same indices)
    sample_y_true = y_test.loc[sample_idx]

    # Run predictions
    sample_y_pred = model.predict(sample_X)

    # Build a nice result DataFrame
    result_df = sample_X.copy()
    result_df["y_true"] = sample_y_true.values
    result_df["y_pred"] = sample_y_pred

    return result_df


# ----------------------------------------------------------------------
# Main script entrypoint
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("INFERENCE: RANDOM FOREST WITH CENTRALITY FEATURES")
    print("Genshin Impact Pull Prediction")
    print("=" * 60 + "\n")

    # 1. Load model
    print(f"Loading model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    # 2. Load features and recreate train/test split
    print("Rebuilding features and train/test split...")
    X_train, X_test, y_train, y_test = load_features()

    # 3. Take a random sample from the test set and run predictions
    n_samples = 5
    print(f"\nTaking a random sample of {n_samples} rows from the test set...\n")
    sample_results = random_sample_predictions(
        model,
        X_test,
        y_test,
        n_samples=n_samples,
        random_state=42,
    )

    # 4. Show results
    print("=" * 60)
    print("SAMPLE PREDICTIONS (from test set)")
    print("=" * 60)
    # Only show the target + prediction by default to keep it short
    cols_to_show = ["y_true", "y_pred"]
    print(sample_results[cols_to_show].to_string(index=True))
    print("=" * 60 + "\n")