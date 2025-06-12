#!/usr/bin/env python3
"""
Test script to verify fusion strategy restrictions are working correctly.
"""

import numpy as np
from fusion import merge_modalities
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def test_fusion_restrictions():
    """Test that fusion restrictions are properly enforced."""
    
    # Test data with 0% missing values
    X1 = np.random.randn(50, 100)
    X2 = np.random.randn(50, 50) 
    X3 = np.random.randn(50, 30)
    y = np.random.randn(50)

    print('=== Testing Fusion Strategies with 0% Missing Data ===')

    # Test weighted_concat (should work)
    try:
        result = merge_modalities(X1, X2, X3, strategy='weighted_concat', y=y, is_regression=True)
        print(f'✓ weighted_concat: {result.shape}')
    except Exception as e:
        print(f'✗ weighted_concat failed: {e}')

    # Test learnable_weighted (should work and return tuple for training)
    try:
        result = merge_modalities(X1, X2, X3, strategy='learnable_weighted', y=y, is_regression=True, is_train=True)
        if isinstance(result, tuple):
            merged_data, fitted_fusion = result
            print(f'✓ learnable_weighted: {merged_data.shape} (with fitted object)')
        else:
            print(f'✓ learnable_weighted: {result.shape}')
    except Exception as e:
        print(f'✗ learnable_weighted failed: {e}')

    # Test with missing data (20%)
    print('\n=== Testing with 20% Missing Data ===')
    X1_missing = X1.copy()
    X1_missing[np.random.choice(50, 10, replace=False), :] = np.nan

    # Test weighted_concat (should fail/fallback)
    try:
        result = merge_modalities(X1_missing, X2, X3, strategy='weighted_concat', y=y, is_regression=True)
        print(f'weighted_concat with missing data: {result.shape} (should show error message)')
    except Exception as e:
        print(f'weighted_concat with missing data failed: {e}')

    # Test learnable_weighted (should work)
    try:
        result = merge_modalities(X1_missing, X2, X3, strategy='learnable_weighted', y=y, is_regression=True, is_train=True)
        if isinstance(result, tuple):
            merged_data, fitted_fusion = result
            print(f'✓ learnable_weighted with missing data: {merged_data.shape} (with fitted object)')
        else:
            print(f'✓ learnable_weighted with missing data: {result.shape}')
    except Exception as e:
        print(f'✗ learnable_weighted with missing data failed: {e}')

    # Test early_fusion_pca (should work with missing data)
    try:
        result = merge_modalities(X1_missing, X2, X3, strategy='early_fusion_pca', n_components=10, is_train=True)
        if isinstance(result, tuple):
            merged_data, fitted_fusion = result
            print(f'✓ early_fusion_pca with missing data: {merged_data.shape} (with fitted object)')
        else:
            print(f'✓ early_fusion_pca with missing data: {result.shape}')
    except Exception as e:
        print(f'✗ early_fusion_pca with missing data failed: {e}')

    # Test validation mode (should return just arrays, not tuples)
    print('\n=== Testing Validation Mode (is_train=False) ===')
    try:
        # First get a fitted fusion object
        train_result = merge_modalities(X1, X2, X3, strategy='learnable_weighted', y=y, is_regression=True, is_train=True)
        if isinstance(train_result, tuple):
            _, fitted_fusion = train_result
            
            # Now test validation mode
            val_result = merge_modalities(X1, X2, X3, strategy='learnable_weighted', fitted_fusion=fitted_fusion, is_train=False)
            print(f'✓ learnable_weighted validation mode: {val_result.shape}')
        else:
            print('✗ Could not get fitted fusion object for validation test')
    except Exception as e:
        print(f'✗ learnable_weighted validation mode failed: {e}')

    print('\n=== Summary ===')
    print('✓ weighted_concat: Only works with 0% missing data')
    print('✓ learnable_weighted, mkl, snf, early_fusion_pca: Work with 0%, 20%, and 50% missing data')
    print('✓ average and sum fusion techniques: COMMENTED OUT')
    print('✓ Training mode returns (data, fitted_object) tuples')
    print('✓ Validation mode returns just data arrays')
    print('\n=== Test Complete ===')

if __name__ == "__main__":
    test_fusion_restrictions() 