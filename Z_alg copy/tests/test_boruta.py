#!/usr/bin/env python3
"""
Tests for the Boruta feature selection functionality.
"""

import pytest
import numpy as np
from Z_alg.utils_boruta import boruta_selector

def test_boruta_selector_classification():
    """Test Boruta selector with classification data."""
    # Create synthetic classification data
    np.random.seed(42)
    n_samples, n_features = 50, 20
    X = np.random.randn(n_samples, n_features)
    # Make 5 features important, the rest are noise
    y = (X[:, 0] + X[:, 1] + X[:, 5] > 0).astype(int)
    
    # Test with classifier mode
    n_features_to_select = 10
    selected_features = boruta_selector(
        X, y, k_features=n_features_to_select, task="clf", 
        random_state=42, max_iter=50
    )
    
    # Check that we get the expected number of features
    assert len(selected_features) == n_features_to_select
    # Check that the important features (0, 1, 5) are selected
    assert 0 in selected_features
    assert 1 in selected_features
    assert 5 in selected_features

def test_boruta_selector_regression():
    """Test Boruta selector with regression data."""
    # Create synthetic regression data
    np.random.seed(42)
    n_samples, n_features = 50, 20
    X = np.random.randn(n_samples, n_features)
    # Make 3 features important, the rest are noise
    y = 2 * X[:, 0] + X[:, 1] - 3 * X[:, 3] + 0.1 * np.random.randn(n_samples)
    
    # Test with regression mode
    n_features_to_select = 5
    selected_features = boruta_selector(
        X, y, k_features=n_features_to_select, task="reg", 
        random_state=42, max_iter=50
    )
    
    # Check that we get the expected number of features
    assert len(selected_features) == n_features_to_select
    # Check that the important features (0, 1, 3) are selected
    assert 0 in selected_features
    assert 1 in selected_features
    assert 3 in selected_features

def test_boruta_selector_invalid_task():
    """Test Boruta selector with invalid task type."""
    np.random.seed(42)
    X = np.random.randn(10, 5)
    y = np.random.randn(10)
    
    # Test with invalid task type
    with pytest.raises(ValueError):
        boruta_selector(X, y, k_features=3, task="invalid")

def test_boruta_selector_with_nans():
    """Test Boruta selector with NaN values."""
    np.random.seed(42)
    X = np.random.randn(10, 5)
    # Add some NaN values
    X[2, 1] = np.nan
    y = np.random.randn(10)
    
    # Should raise ValueError due to NaNs
    with pytest.raises(ValueError):
        boruta_selector(X, y, k_features=3, task="reg")

def test_boruta_selector_fixed_k_features():
    """Test that Boruta selector always returns exact k features."""
    # Create data where only 1 feature is really important
    np.random.seed(42)
    n_samples, n_features = 30, 10
    X = np.random.randn(n_samples, n_features)
    y = 5 * X[:, 2] + 0.1 * np.random.randn(n_samples)
    
    # Ask for 5 features
    k_features = 5
    selected_features = boruta_selector(
        X, y, k_features=k_features, task="reg", 
        random_state=42, max_iter=20, perc=95
    )
    
    # Boruta might only find 1 feature "important", but we should get exactly k_features
    assert len(selected_features) == k_features
    # The most important feature should be included
    assert 2 in selected_features 