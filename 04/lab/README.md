# Machine Learning Project Setup Guide

## Table of Contents

- [Required Submission Files](#required-submission-files)
- [Environment Setup](#environment-setup)
- [Project Structure](#project-structure)
- [Using the Model Classes](#using-the-model-classes)
- [Training Models](#training-models)
- [Generating Predictions](#generating-predictions)
- [Code Quality Guidelines](#code-quality-guidelines)

## Required Submission Files

Ensure your submission includes the following files:

```
submission/
├── train.py                           # Main training script
├── inference.py                       # Prediction/inference script
├── model.py                           # GenshinModel class (sklearn wrapper)
├── utils.py                           # FeatureBuilder, ForecastingMetrics
├── centrality_maker.py                # Network centrality computation
├── characteristics_maker.py           # Web scraping utilities
├── graph_maker.py                     # NetworkX graph construction
├── best_model.joblib                  # Trained model file
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
├── report.ipynb                       # Analysis notebook
└── data/                              # Data directory
    ├── characters/
    │   └── characteristics.csv        # Character features + target
    ├── abyss_eigenvector.csv          # Eigenvector centrality matrix
    ├── abyss_betweenness.csv          # Betweenness centrality matrix
    ├── abyss_graphs/                  # NetworkX graphs (pickled)
    └── nn_results/                    # Neural network training results
```

## Environment Setup

### 1. Python Environment

Create and activate a virtual environment:

```powershell
# Create virtual environment
python -m venv venv

# Activate on Windows
.\venv\Scripts\Activate.ps1

# Activate on macOS/Linux
source venv/bin/activate
```

### 2. Install Dependencies

```powershell
# Install all required packages
pip install -r requirements.txt
```

**Required packages (requirements.txt):**
```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
torch>=2.0.0
networkx>=3.0
matplotlib>=3.6.0
seaborn>=0.12.0
joblib>=1.3.0
lxml>=4.9.0
```

### 3. Verify Installation

```python
# Test imports
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import torch
import joblib
import networkx as nx

print("All packages installed successfully!")
```

## Using the Model Classes

### Basic Model Initialization

The `GenshinModel` class is a sklearn-compatible wrapper that handles preprocessing automatically:

```python
from model import GenshinModel
import pandas as pd

# Initialize model with desired type
model = GenshinModel(
    model_type="rf",        # Options: "rf", "knn", "lr", "nn"
    random_state=42,
    n_estimators=100,       # RF-specific parameter
    max_depth=None,         # RF-specific parameter
    n_neighbors=5,          # KNN-specific parameter
    nn_params=None          # Neural network parameters dict
)

# Load your data
X_train = pd.DataFrame(...)  # Feature DataFrame
y_train = pd.Series(...)     # Target Series

# Fit model (preprocessing handled automatically)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Access internal components
rf_model = model._model_              # Internal sklearn model
preprocessor = model.preprocessor     # ColumnTransformer
```

### Supported Model Types

```python
# Random Forest (recommended)
model_rf = GenshinModel(
    model_type="rf",
    n_estimators=100,
    max_depth=None,
    random_state=42
)

# K-Nearest Neighbors
model_knn = GenshinModel(
    model_type="knn",
    n_neighbors=5
)

# Linear Regression
model_lr = GenshinModel(
    model_type="lr"
)

# Neural Network (PyTorch-based)
model_nn = GenshinModel(
    model_type="nn",
    nn_params={
        'n_hidden_layers': 3,
        'n_hidden_nodes': 14,
        'dropout_rate': 0.2
    }
)
```

### Automatic Preprocessing

The model automatically:
1. Detects categorical columns in the DataFrame
2. Applies `OneHotEncoder` with `handle_unknown="ignore"`
3. Passes numerical features through unchanged (remainder="passthrough")
4. Handles transformations consistently for training and prediction

```python
# Access the fitted preprocessor
preprocessor = model.preprocessor

# Get feature names after encoding
feature_names = preprocessor.get_feature_names_out()

# Example output:
# ['cat__Element_Anemo', 'cat__Element_Cryo', 'cat__Weapon Type_Bow', 
#  'remainder__Star', 'remainder__Line Count', 'remainder__Abyss Eigenvector', ...]
```

## Project Structure

```
genshin-pull-prediction/
├── data/
│   ├── characters/
│   │   ├── characteristics.csv          # Character attributes + pull counts
│   │   └── characters_ownership.csv     # Raw ownership data
│   ├── abyss_graphs/                    # NetworkX graphs per patch
│   │   └── graph_{version}.pickle
│   ├── stygian_graphs/
│   │   └── graph_{version}.pickle
│   ├── abyss_eigenvector.csv           # Centrality matrices
│   ├── abyss_betweenness.csv
## Training Models

### Option 1: Run Training Script

```powershell
# Execute the training script
python train.py
```

**Output:**
```
Training Random Forest with Centrality Features on training data only...

============================================================
BEST PARAMETERS
============================================================
Best Params: {'n_estimators': 200}

============================================================
TEST SET PERFORMANCE METRICS
============================================================
MAE                      : 45,189.23
MSE                      : 4,120,567,890.45
RMSE                     : 64,190.77
MAPE                     : 82.34
SMAPE                    : 78.91
R2                       : 0.8710
============================================================

============================================================
TOP 10 MOST IMPORTANT FEATURES
============================================================
                          Feature  Importance
                     remainder__Star    0.423156
    remainder__Abyss Eigenvector       0.234512
    remainder__Abyss Betweenness       0.156789
       cat__Weapon Type_Catalyst       0.045612
...
============================================================

✓ Model and training data saved successfully as joblib files.
```

### Option 2: Custom Training with Function

```python
from train import train_random_forest_with_centrality, compare_alphas
import joblib

# Train with specific alpha value
result = train_random_forest_with_centrality(
    alpha=0.2,
    test_size=0.2,
    random_state_split=37,
    random_state_model=42,
    n_splits=5,
    verbose=True
)

# Access results
best_model = result['model']
metrics = result['metrics']
feature_importances = result['feature_importances']

print(f"MAE: {metrics['MAE']:,.2f}")
print(f"RMSE: {metrics['RMSE']:,.2f}")
print(f"R²: {metrics['R2']:.4f}")

# Save model
joblib.dump(best_model, "my_model.joblib")

# Compare different alpha values
comparison_df = compare_alphas(alphas=[0.2, 0.3, 0.4], verbose=True)
print(comparison_df)
```

### Option 3: Manual GridSearchCV

```python
from model import GenshinModel
from utils import FeatureBuilder, ForecastingMetrics
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
import joblib

# 1. Load and prepare data
abyss_paths = {
    "Abyss Eigenvector": "data/abyss_eigenvector.csv",
    "Abyss Betweenness": "data/abyss_betweenness.csv"
}

fb = FeatureBuilder(centrality_paths=abyss_paths, alpha=0.2)
X, y = fb.load_and_merge_features()

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=37
)

# 3. Hyperparameter tuning
model = GenshinModel(model_type="rf", random_state=42)
param_grid = {"n_estimators": [50, 100, 200, 300, 500, 1000]}

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=kfold,
    scoring={"MSE": "neg_mean_squared_error", "R2": "r2"},
    refit="MSE",
    return_train_score=True,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# 4. Evaluate
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

metrics = ForecastingMetrics.compute_all_metrics(y_test, y_pred)
for metric_name, value in metrics.items():
    print(f"{metric_name}: {value:,.2f}")

# 5. Save model
joblib.dump(best_model, "my_model.joblib")
```

## Generating Predictions

### Load Saved Model

```python
import joblib
import pandas as pd
from utils import FeatureBuilder
from sklearn.model_selection import train_test_split

# Load the trained model
model = joblib.load("best_model.joblib")

# Rebuild features (must match training setup)
abyss_paths = {
    "Abyss Eigenvector": "data/abyss_eigenvector.csv",
    "Abyss Betweenness": "data/abyss_betweenness.csv"
}

fb = FeatureBuilder(centrality_paths=abyss_paths, alpha=0.2)
X, y = fb.load_and_merge_features()

# Recreate train/test split with same random state
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=37
)

# Generate predictions on test set
predictions = model.predict(X_test)
print(predictions)
```

### Run Inference Script

The provided `inference.py` script automatically:
1. Loads the trained model from `best_model.joblib`
2. Rebuilds features with the same FeatureBuilder settings
3. Recreates the exact train/test split (random_state=37)
4. Samples 5 random test examples
5. Shows predictions vs actual values

```powershell
python inference.py
```

**Output:**
```
============================================================
INFERENCE: RANDOM FOREST WITH CENTRALITY FEATURES
Genshin Impact Pull Prediction
============================================================

Loading model from: best_model.joblib
Rebuilding features and train/test split...

Taking a random sample of 5 rows from the test set...

============================================================
SAMPLE PREDICTIONS (from test set)
============================================================
      y_true       y_pred
42   234567.0    245123.45
18   456789.0    438901.23
67    89012.0     92345.67
...
============================================================
```

### Custom Inference Function

```python
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import FeatureBuilder

def load_model(model_path: str = "best_model.joblib"):
    """Load trained model from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)

def load_features(alpha: float = 0.2, test_size: float = 0.2, 
                  random_state: int = 37):
    """Rebuild features and train/test split."""
    abyss_paths = {
        "Abyss Eigenvector": "data/abyss_eigenvector.csv",
        "Abyss Betweenness": "data/abyss_betweenness.csv"
    }
    
    fb = FeatureBuilder(centrality_paths=abyss_paths, alpha=alpha)
    X, y = fb.load_and_merge_features()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test

def random_sample_predictions(model, X_test, y_test, n_samples=5):
    """Get predictions on random test samples."""
    sample_X = X_test.sample(n=n_samples, random_state=42)
    sample_y_true = y_test.loc[sample_X.index]
    sample_y_pred = model.predict(sample_X)
    
    result_df = sample_X.copy()
    result_df['y_true'] = sample_y_true.values
    result_df['y_pred'] = sample_y_pred
    
    return result_df

if __name__ == "__main__":
    model = load_model()
    X_train, X_test, y_train, y_test = load_features()
    results = random_sample_predictions(model, X_test, y_test)
    print(results[['y_true', 'y_pred']])
```

## Project Structure

```
lab/
├── train.py                      # Training script with GridSearchCV
├── inference.py                  # Prediction script for test samples
├── model.py                      # GenshinModel and GenshinNetwork classes
├── utils.py                      # FeatureBuilder, ForecastingMetrics
├── centrality_maker.py           # Network centrality computation
├── characteristics_maker.py      # Web scraping character data
├── graph_maker.py                # NetworkX graph construction
├── best_model.joblib             # Trained Random Forest model
├── requirements.txt              # Python dependencies
├── README.md                     # This guide
├── report.ipynb                  # Full analysis notebook
├── time_series.ipynb             # Time series analysis (optional)
├── data/
│   ├── characters/
│   │   ├── characteristics.csv           # Main feature + target dataset
│   │   └── characters_ownership.csv      # Raw ownership data
│   ├── abyss_eigenvector.csv            # Eigenvector centrality (52 patches)
│   ├── abyss_betweenness.csv            # Betweenness centrality (52 patches)
│   ├── stygian_eigenvector.csv          # Stygian mode eigenvector
│   ├── stygian_betweenness.csv          # Stygian mode betweenness
│   ├── abyss_rank_activity.csv          # Patch metadata
│   ├── abyss_graphs/                    # NetworkX graphs (pickled)
│   │   └── graph_{version}.pickle       # One graph per patch
│   ├── stygian_graphs/
│   │   └── graph_{version}.pickle
│   └── nn_results/                      # Neural network training results
│       └── alpha{alpha}_better.pkl      # Cached NN results by alpha
├── figures/                      # Visualizations for report
└── __pycache__/                  # Python cache files
```

## Code Quality Guidelines

### Type Annotations

Always include type hints for better code clarity:

```python
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

def process_data(
    data: pd.DataFrame,
    target_col: str,
    feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Process input data for modeling.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input dataset
    target_col : str
        Name of target column
    feature_cols : List[str]
        List of feature column names
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Features and target
    """
    X = data[feature_cols]
    y = data[target_col]
    return X, y
```

### Docstrings

Use NumPy-style docstrings for all functions and classes:

```python
class ModelWrapper:
    """
    Wrapper class for scikit-learn models with preprocessing.
    
    Parameters
    ----------
    model_type : str
        Type of model ('rf', 'knn', 'lr')
    random_state : int, optional
        Random seed for reproducibility
        
    Attributes
    ----------
    preprocessor : ColumnTransformer
        Fitted preprocessing pipeline
    _model : BaseEstimator
        Underlying sklearn model
        
    Examples
    --------
    >>> model = ModelWrapper(model_type='rf', random_state=42)
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    """
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ModelWrapper':
        """
        Fit the model to training data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training features
        y : pd.Series
            Training target
            
        Returns
        -------
        self : ModelWrapper
            Fitted model instance
        """
        # Implementation
        return self
```

### Error Handling

```python
def load_data(filepath: str) -> pd.DataFrame:
    """Load data with proper error handling."""
    try:
        data = pd.read_csv(filepath)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {filepath}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"Data file is empty: {filepath}")
    except Exception as e:
        raise RuntimeError(f"Error loading data: {str(e)}")
```

### Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Use in code
logger.info("Starting model training...")
logger.warning("Low sample size detected")
logger.error("Model training failed")
```

## Quick Start Checklist

- [ ] Set up Python virtual environment
- [ ] Install all dependencies from requirements.txt
- [ ] Verify data files are in `data/` directory
- [ ] Review `model.py` and `utils.py` for custom classes
- [ ] Run `train.py` to train model
- [ ] Check that model file (.pkl) is created
- [ ] Test predictions with `inference.py`
- [ ] Add type annotations to all functions
- [ ] Write docstrings for all classes and methods
- [ ] Verify all required files are included in submission
metrics = ForecastingMetrics.compute_all_metrics(y_test, y_pred)
print(f"MAE: {metrics['MAE']:,.2f}")
print(f"RMSE: {metrics['RMSE']:,.2f}")
print(f"R²: {metrics['R2']:.4f}")

# 5. Save model
joblib.dump(best_model, "my_model.pkl")
```el.fit(X_train, y_train)  # Auto-handles preprocessing
y_pred = model.predict(X_test)
```

### `ForecastingMetrics` (utils.py)
```python
metrics = ForecastingMetrics.compute_all_metrics(y_test, y_pred)
# Returns: MAE, MSE, RMSE, MAPE, SMAPE, R2
```

## Acknowledgements

The team acknowledges Julian Isidro for assistance with data exploration and creation of the title banner.

## References

[1] Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). *Why do tree-based models still outperform deep learning on tabular data?* [arXiv:2207.08815](https://doi.org/10.48550/arXiv.2207.08815)

[2] Wydmański, W., et al. (2023). *HyperTab: Hypernetwork Approach for Deep Learning on Small Tabular Datasets.* [arXiv:2304.03543](https://arxiv.org/abs/2304.03543)

## Environment Setup

### 1. Python Environment

Create and activate a virtual environment:

```powershell
# Create virtual environment
python -m venv venv

# Activate on Windows
.\venv\Scripts\Activate.ps1

# Activate on macOS/Linux
source venv/bin/activate
```

### 2. Install Dependencies

```powershell
# Install all required packages
pip install -r requirements.txt
```

**Required packages (requirements.txt):**
```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
torch>=2.0.0
networkx>=3.0
matplotlib>=3.6.0
seaborn>=0.12.0
joblib>=1.3.0
```

### 3. Verify Installation

```python
# Test imports
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import torch
import joblib

print("All packages installed successfully!")
```