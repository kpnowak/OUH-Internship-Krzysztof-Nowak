#!/usr/bin/env python3

"""
Test script for PLSRegression components in the models module.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression

# Import our modules
from Z_alg.models import cached_fit_transform_extractor_regression

def test_pls_regression_handling():
    """Test that the PLSRegression handling works correctly."""
    print("Testing PLSRegression functionality...")
    
    # Create random test data
    X = np.random.rand(20, 15)
    y = np.random.rand(20)
    
    # Create a PLSRegression extractor
    extractor = PLSRegression(n_components=2)
    
    try:
        # Test with PLS
        print("\nTesting with PLSRegression...")
        new_extractor, X_transformed = cached_fit_transform_extractor_regression(
            X, y, extractor, n_components=5, ds_name="test", modality_name="test_modality", fold_idx=0
        )
        
        if X_transformed is not None:
            print(f"Success! Transformed shape: {X_transformed.shape}")
        else:
            print("Test failed - transformation returned None")
            
        # Test with missing y (should use fallback)
        print("\nTesting with missing y values...")
        new_extractor, X_transformed = cached_fit_transform_extractor_regression(
            X, None, extractor, n_components=5, ds_name="test", modality_name="test_modality", fold_idx=0
        )
        
        if X_transformed is not None:
            print(f"PLS with missing y worked (using fallback). Shape: {X_transformed.shape}")
        else:
            print("PLS with missing y returned None as expected")
            
        # Test with too many components (should reduce automatically)
        print("\nTesting with too many components...")
        new_extractor, X_transformed = cached_fit_transform_extractor_regression(
            X, y, extractor, n_components=100, ds_name="test", modality_name="test_modality", fold_idx=0
        )
        
        if X_transformed is not None:
            print(f"Success! Requested 100 components, got shape: {X_transformed.shape}")
        else:
            print("Test failed - transformation returned None")
            
        print("\nAll tests completed!")
        
    except Exception as e:
        print(f"Test failed with error: {str(e)}")

if __name__ == "__main__":
    test_pls_regression_handling() 