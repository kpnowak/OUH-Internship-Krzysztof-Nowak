import os
import numpy as np
import pandas as pd
from Z_alg.models import cached_fit_transform_extractor_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Test with fake data
X = np.random.rand(20, 50)  # 20 samples, 50 features
y = np.array([0] * 10 + [1] * 10)  # Binary labels for testing

print("Testing feature extraction with force_n_components=True")
print(f"Input data shape: {X.shape}")

# Try with a high number of components that would normally be reduced
n_components = 15
print(f"Requested components: {n_components}")

# Test with force_n_components=False (default behavior)
print("\nTEST 1: Default behavior (auto-reduction)")
extractor, X_transformed = cached_fit_transform_extractor_classification(
    X, y, LDA(), n_components, force_n_components=False, 
    ds_name="test", modality_name="test_modality", fold_idx=0
)
print(f"Transformed data shape: {X_transformed.shape}")

# Test with force_n_components=True (our modification)
print("\nTEST 2: Modified behavior (force components)")
extractor, X_transformed = cached_fit_transform_extractor_classification(
    X, y, LDA(), n_components, force_n_components=True, 
    ds_name="test", modality_name="test_modality", fold_idx=1
)
print(f"Transformed data shape: {X_transformed.shape}")

print("\nTest completed!") 