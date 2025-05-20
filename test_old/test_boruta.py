#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.datasets import make_classification

# Generate a synthetic classification dataset
X, y = make_classification(
    n_samples=100,
    n_features=20,
    n_informative=8,
    n_redundant=2,
    n_repeated=0,
    n_classes=2,
    random_state=42
)

print(f"Generated dataset: X shape {X.shape}, y shape {y.shape}")
print(f"Class distribution: {np.bincount(y)}")

# Define RF base estimator for Boruta
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42,
    n_jobs=1  # Boruta requires single-threaded estimator
)

# Initialize Boruta
boruta_selector = BorutaPy(
    estimator=rf,
    n_estimators='auto',
    max_iter=50,
    random_state=42,
    verbose=2,  # Increased verbosity for debugging
    perc=95     # More lenient percentile
)

try:
    # Print the first few samples
    print("\nFirst 5 samples of X:")
    print(X[:5])
    print("\nFirst 5 elements of y:")
    print(y[:5])
    
    # Fit Boruta
    print("\nFitting Boruta...")
    boruta_selector.fit(X, y)
    
    # Get results
    print("\nBoruta Results:")
    print(f"Number of selected features: {np.sum(boruta_selector.support_)}")
    print(f"Selected features (mask): {boruta_selector.support_}")
    print(f"Feature ranking: {boruta_selector.ranking_}")
    
    # Extract selected features
    selected_features = np.where(boruta_selector.support_)[0]
    print(f"Selected feature indices: {selected_features}")
    
    # Transform data using selected features
    X_selected = X[:, selected_features]
    print(f"Transformed data shape: {X_selected.shape}")
    
    print("\nBoruta successfully completed!")
    
except Exception as e:
    print(f"\nError during Boruta execution: {str(e)}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc() 