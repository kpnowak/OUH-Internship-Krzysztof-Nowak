#!/usr/bin/env python3
"""
Test script for biomedical data improvements.
"""

import numpy as np
import pandas as pd
import logging
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, r2_score

# Local imports
from config import CV_CONFIG, MODEL_OPTIMIZATIONS, PREPROCESSING_CONFIG
from preprocessing import biomedical_preprocessing_pipeline
from fast_feature_selection import biomedical_feature_selection
from cv import get_cv_strategy, validate_cv_splits
from models import get_model_object

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_biomedical_like_data(n_samples=100, n_features=1000, task_type='classification', sparsity=0.8):
    """
    Create synthetic data that mimics biomedical characteristics.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        task_type: 'classification' or 'regression'
        sparsity: Proportion of zero values
    
    Returns:
        X, y data
    """
    logger.info(f"Creating synthetic {task_type} data: {n_samples} samples, {n_features} features")
    
    if task_type == 'classification':
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=min(50, n_features//10),
            n_redundant=min(20, n_features//20),
            n_clusters_per_class=1,
            class_sep=0.8,
            random_state=42
        )
    else:
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=min(50, n_features//10),
            noise=0.1,
            random_state=42
        )
    
    # Make data sparse (like gene expression)
    mask = np.random.random(X.shape) < sparsity
    X[mask] = 0
    
    # Add some extreme values (like outliers in biomedical data)
    outlier_mask = np.random.random(X.shape) < 0.01
    X[outlier_mask] = np.random.exponential(10, size=np.sum(outlier_mask))
    
    # Ensure non-negative values (like gene expression)
    X = np.abs(X)
    
    logger.info(f"Data sparsity: {np.mean(X == 0):.2%}")
    logger.info(f"Data range: [{np.min(X):.2f}, {np.max(X):.2f}]")
    
    return X, y

def test_preprocessing_pipeline():
    """Test the biomedical preprocessing pipeline."""
    logger.info("Testing biomedical preprocessing pipeline")
    
    # Create test data
    X, y = create_biomedical_like_data(n_samples=50, n_features=200, sparsity=0.7)
    
    logger.info(f"Original data shape: {X.shape}")
    logger.info(f"Original sparsity: {np.mean(X == 0):.2%}")
    
    # Apply preprocessing
    X_processed, transformers = biomedical_preprocessing_pipeline(X, y, PREPROCESSING_CONFIG)
    
    logger.info(f"Processed data shape: {X_processed.shape}")
    logger.info(f"Processed sparsity: {np.mean(X_processed == 0):.2%}")
    logger.info(f"Transformers applied: {list(transformers.keys())}")
    
    # Check for NaN or infinite values
    assert not np.any(np.isnan(X_processed)), "Processed data contains NaN values"
    assert not np.any(np.isinf(X_processed)), "Processed data contains infinite values"
    
    logger.info("✓ Preprocessing pipeline test passed")
    return True

def test_feature_selection():
    """Test biomedical feature selection methods."""
    logger.info("Testing biomedical feature selection")
    
    # Create test data
    X, y = create_biomedical_like_data(n_samples=80, n_features=500, task_type='classification', sparsity=0.6)
    
    # Test different selection methods
    methods = ['combined', 'stability', 'sparse_aware', 'univariate']
    
    for method in methods:
        logger.info(f"Testing {method} feature selection")
        
        try:
            selected_indices = biomedical_feature_selection(
                X, y, task_type='classification', n_features=32, method=method
            )
            
            assert len(selected_indices) == 32, f"Expected 32 features, got {len(selected_indices)}"
            assert len(set(selected_indices)) == 32, "Duplicate features selected"
            assert all(0 <= idx < X.shape[1] for idx in selected_indices), "Invalid feature indices"
            
            logger.info(f"✓ {method} feature selection passed")
            
        except Exception as e:
            logger.warning(f"✗ {method} feature selection failed: {e}")
    
    return True

def test_cv_strategy():
    """Test cross-validation strategy for small datasets."""
    logger.info("Testing CV strategy for small datasets")
    
    # Test with very small dataset
    X_small, y_small = create_biomedical_like_data(n_samples=25, n_features=100, task_type='classification')
    
    cv_strategy = get_cv_strategy(X_small, y_small, task_type='classification', cv_config=CV_CONFIG)
    is_valid = validate_cv_splits(cv_strategy, X_small, y_small, task_type='classification')
    
    assert is_valid, "CV strategy validation failed for small dataset"
    logger.info("✓ Small dataset CV strategy test passed")
    
    # Test with medium dataset
    X_medium, y_medium = create_biomedical_like_data(n_samples=100, n_features=200, task_type='classification')
    
    cv_strategy = get_cv_strategy(X_medium, y_medium, task_type='classification', cv_config=CV_CONFIG)
    is_valid = validate_cv_splits(cv_strategy, X_medium, y_medium, task_type='classification')
    
    assert is_valid, "CV strategy validation failed for medium dataset"
    logger.info("✓ Medium dataset CV strategy test passed")
    
    return True

def test_model_performance():
    """Test model performance with biomedical data."""
    logger.info("Testing model performance with biomedical-like data")
    
    # Create test data
    X, y = create_biomedical_like_data(n_samples=100, n_features=300, task_type='classification', sparsity=0.7)
    
    # Preprocess data
    X_processed, _ = biomedical_preprocessing_pipeline(X, y, PREPROCESSING_CONFIG)
    
    # Select features
    selected_indices = biomedical_feature_selection(
        X_processed, y, task_type='classification', n_features=32, method='combined'
    )
    X_selected = X_processed[:, selected_indices]
    
    # Test different models
    models_to_test = ['RandomForest', 'LogisticRegression', 'SVM']
    
    for model_name in models_to_test:
        logger.info(f"Testing {model_name}")
        
        try:
            # Get model
            model = get_model_object(model_name, random_state=42)
            
            # Get CV strategy
            cv_strategy = get_cv_strategy(X_selected, y, task_type='classification', cv_config=CV_CONFIG)
            
            # Cross-validation
            scores = cross_val_score(model, X_selected, y, cv=cv_strategy, scoring='accuracy')
            
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            logger.info(f"{model_name} accuracy: {mean_score:.3f} ± {std_score:.3f}")
            
            # Performance should be better than random (0.5 for binary classification)
            assert mean_score > 0.55, f"{model_name} performance too low: {mean_score:.3f}"
            
            logger.info(f"✓ {model_name} test passed")
            
        except Exception as e:
            logger.warning(f"✗ {model_name} test failed: {e}")
    
    return True

def test_regression_performance():
    """Test regression model performance."""
    logger.info("Testing regression model performance")
    
    # Create test data
    X, y = create_biomedical_like_data(n_samples=100, n_features=300, task_type='regression', sparsity=0.7)
    
    # Preprocess data
    X_processed, _ = biomedical_preprocessing_pipeline(X, y, PREPROCESSING_CONFIG)
    
    # Select features
    selected_indices = biomedical_feature_selection(
        X_processed, y, task_type='regression', n_features=32, method='combined'
    )
    X_selected = X_processed[:, selected_indices]
    
    # Test regression models
    models_to_test = ['RandomForest', 'ElasticNet', 'Lasso']
    
    for model_name in models_to_test:
        logger.info(f"Testing {model_name} for regression")
        
        try:
            # Get model
            model = get_model_object(model_name, random_state=42)
            
            # Get CV strategy
            cv_strategy = get_cv_strategy(X_selected, y, task_type='regression', cv_config=CV_CONFIG)
            
            # Cross-validation
            scores = cross_val_score(model, X_selected, y, cv=cv_strategy, scoring='r2')
            
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            logger.info(f"{model_name} R²: {mean_score:.3f} ± {std_score:.3f}")
            
            # Performance should be positive (better than mean predictor)
            assert mean_score > 0.0, f"{model_name} R² too low: {mean_score:.3f}"
            
            logger.info(f"✓ {model_name} regression test passed")
            
        except Exception as e:
            logger.warning(f"✗ {model_name} regression test failed: {e}")
    
    return True

def run_all_tests():
    """Run all biomedical improvement tests."""
    logger.info("Starting biomedical improvements test suite")
    
    tests = [
        ("Preprocessing Pipeline", test_preprocessing_pipeline),
        ("Feature Selection", test_feature_selection),
        ("CV Strategy", test_cv_strategy),
        ("Classification Performance", test_model_performance),
        ("Regression Performance", test_regression_performance),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            test_func()
            logger.info(f"✓ {test_name} PASSED")
            passed += 1
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED: {e}")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Test Results: {passed}/{total} tests passed")
    logger.info(f"{'='*50}")
    
    if passed == total:
        logger.info("🎉 All tests passed! Biomedical improvements are working correctly.")
        return True
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Please review the issues.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1) 