#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from Z_alg.models import cached_fit_transform_extractor_regression

# Create simple test data
X = np.random.rand(20, 10)
y = np.random.rand(20)

# Test PCA extractor
extractor = PCA(random_state=42)
new_extractor, X_transformed = cached_fit_transform_extractor_regression(
    X, y, extractor, n_components=5, ds_name="test", modality_name="test_modality", fold_idx=0
)

print("Function executed successfully")
print(f"Input shape: {X.shape}")
print(f"Output shape: {X_transformed.shape}")
print(f"Components requested: 5, actual components: {new_extractor.n_components_}") 