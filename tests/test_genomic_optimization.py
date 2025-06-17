#!/usr/bin/env python3
"""
Test script for genomic optimization approach.

This script validates the new genomic-optimized feature selection and modeling
approach using synthetic data that mimics real genomic characteristics.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, matthews_corrcoef
import logging
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fast_feature_selection import GenomicFeatureSelector
from models import get_genomic_optimized_models, validate_model_performance
from config import N_VALUES_LIST, PERFORMANCE_TARGETS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_genomic_like_data(n_samples=100, n_features=5000, n_informative=200, 
                           sparsity=0.3, noise=0.1, task='regression'):
    """
    Create synthetic data that mimics genomic characteristics.
    
    Parameters
    ----------
    n_samples : int
        Number of samples (typical for genomic studies: 50-500)
    n_features : int
        Number of features (typical for genomics: 1000-50000)
    n_informative : int
        Number of truly informative features
    sparsity : float
        Fraction of features that are zero (genomic data is often sparse)
    noise : float
        Noise level
    task : str
        'regression' or 'classification'
    
    Returns
    -------
    X, y : arrays
        Feature matrix and target vector
    """
    logger.info(f"Creating {task} dataset: {n_samples} samples, {n_features} features")
    
    if task == 'regression':
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            noise=noise,
            random_state=42
        )
    else:
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=0,
            n_clusters_per_class=1,
            class_sep=0.8,
            random_state=42
        )
    
    # Add sparsity to mimic genomic data
    if sparsity > 0:
        sparse_mask = np.random.random(X.shape) < sparsity
        X[sparse_mask] = 0
    
    # Add some biological-like structure
    # Some features should be correlated (gene co-expression)
    n_correlated_groups = 10
    group_size = 20
    for i in range(n_correlated_groups):
        start_idx = i * group_size
        end_idx = min(start_idx + group_size, n_features)
        if end_idx > start_idx + 1:
            # Make features in this group correlated
            base_feature = X[:, start_idx]
            for j in range(start_idx + 1, end_idx):
                correlation_strength = np.random.uniform(0.3, 0.8)
                noise_component = np.random.normal(0, 1, n_samples)
                X[:, j] = correlation_strength * base_feature + (1 - correlation_strength) * noise_component
    
    logger.info(f"Created dataset with sparsity: {np.mean(X == 0):.3f}")
    return X, y

def test_feature_selection_methods():
    """Test different genomic feature selection methods."""
    logger.info("=== Testing Genomic Feature Selection Methods ===")
    
    # Create test data
    X, y = create_genomic_like_data(n_samples=80, n_features=2000, task='regression')
    
    methods = [
        'genomic_ensemble',
        'biological_relevance', 
        'permissive_univariate',
        'stability_selection',
        'variance_f_test'
    ]
    
    results = {}
    
    for method in methods:
        logger.info(f"\nTesting method: {method}")
        
        try:
            # Test with different feature counts from our new range
            for n_features in [128, 256, 512]:
                selector = GenomicFeatureSelector(
                    method=method, 
                    n_features=n_features,
                    random_state=42
                )
                
                selector.fit(X, y, is_regression=True)
                selected_features = selector.get_selected_features()
                
                logger.info(f"  {n_features} features requested, {len(selected_features)} selected")
                
                # Store results
                if method not in results:
                    results[method] = {}
                results[method][n_features] = len(selected_features)
                
        except Exception as e:
            logger.error(f"Method {method} failed: {e}")
            results[method] = "FAILED"
    
    return results

def test_model_performance():
    """Test model performance with genomic optimization."""
    logger.info("\n=== Testing Model Performance ===")
    
    # Test regression
    logger.info("\n--- Regression Test ---")
    X_reg, y_reg = create_genomic_like_data(n_samples=100, n_features=3000, task='regression')
    
    # Use genomic feature selection
    selector = GenomicFeatureSelector(method='genomic_ensemble', n_features=256)
    X_reg_selected = selector.fit_transform(X_reg, y_reg, is_regression=True)
    
    logger.info(f"Selected {X_reg_selected.shape[1]} features for regression")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_reg_selected, y_reg, test_size=0.3, random_state=42
    )
    
    # Test genomic-optimized models
    models = get_genomic_optimized_models(
        n_features=X_reg_selected.shape[1], 
        n_samples=X_train.shape[0], 
        is_regression=True
    )
    
    regression_results = {}
    
    for model_name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            # Validate against targets
            validation = validate_model_performance(y_test, y_pred, is_regression=True, model_name=model_name)
            
            regression_results[model_name] = {
                'r2': r2,
                'rmse': rmse,
                'target_met': validation.get('r2_target_met', False),
                'adequate': validation.get('performance_adequate', False)
            }
            
            logger.info(f"  {model_name}: R² = {r2:.4f}, RMSE = {rmse:.4f}")
            
        except Exception as e:
            logger.error(f"Model {model_name} failed: {e}")
            regression_results[model_name] = "FAILED"
    
    # Test classification
    logger.info("\n--- Classification Test ---")
    X_clf, y_clf = create_genomic_like_data(n_samples=100, n_features=3000, task='classification')
    
    # Use genomic feature selection
    selector_clf = GenomicFeatureSelector(method='genomic_ensemble', n_features=256)
    X_clf_selected = selector_clf.fit_transform(X_clf, y_clf, is_regression=False)
    
    logger.info(f"Selected {X_clf_selected.shape[1]} features for classification")
    
    # Split data
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf_selected, y_clf, test_size=0.3, random_state=42
    )
    
    # Test genomic-optimized models
    models_clf = get_genomic_optimized_models(
        n_features=X_clf_selected.shape[1], 
        n_samples=X_train_clf.shape[0], 
        is_regression=False
    )
    
    classification_results = {}
    
    for model_name, model in models_clf.items():
        try:
            model.fit(X_train_clf, y_train_clf)
            y_pred_clf = model.predict(X_test_clf)
            
            accuracy = accuracy_score(y_test_clf, y_pred_clf)
            mcc = matthews_corrcoef(y_test_clf, y_pred_clf)
            
            # Validate against targets
            validation = validate_model_performance(y_test_clf, y_pred_clf, is_regression=False, model_name=model_name)
            
            classification_results[model_name] = {
                'accuracy': accuracy,
                'mcc': mcc,
                'target_met': validation.get('mcc_target_met', False),
                'adequate': validation.get('performance_adequate', False)
            }
            
            logger.info(f"  {model_name}: Accuracy = {accuracy:.4f}, MCC = {mcc:.4f}")
            
        except Exception as e:
            logger.error(f"Model {model_name} failed: {e}")
            classification_results[model_name] = "FAILED"
    
    return regression_results, classification_results

def test_configuration_values():
    """Test that our new configuration values are being used."""
    logger.info("\n=== Testing Configuration Values ===")
    
    # Test N_VALUES_LIST
    logger.info(f"N_VALUES_LIST: {N_VALUES_LIST}")
    expected_min = 128
    if min(N_VALUES_LIST) >= expected_min:
        logger.info(" N_VALUES_LIST uses genomic-appropriate ranges")
    else:
        logger.error(f" N_VALUES_LIST still too small, minimum should be >= {expected_min}")
    
    # Test performance targets
    logger.info(f"Regression R² target: {PERFORMANCE_TARGETS['regression']['r2_min']}")
    logger.info(f"Classification MCC target: {PERFORMANCE_TARGETS['classification']['mcc_min']}")
    
    if PERFORMANCE_TARGETS['regression']['r2_min'] >= 0.5:
        logger.info(" Regression targets are appropriately ambitious")
    else:
        logger.warning(" Regression targets might be too low")
    
    if PERFORMANCE_TARGETS['classification']['mcc_min'] >= 0.5:
        logger.info(" Classification targets are appropriately ambitious")
    else:
        logger.warning(" Classification targets might be too low")

def compare_old_vs_new_approach():
    """Compare old vs new approach on the same data."""
    logger.info("\n=== Comparing Old vs New Approach ===")
    
    # Create test data
    X, y = create_genomic_like_data(n_samples=80, n_features=2000, task='regression')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Old approach: 8 features, high regularization
    logger.info("\n--- Old Approach (8 features, high regularization) ---")
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.linear_model import ElasticNet
    
    old_selector = SelectKBest(f_regression, k=8)
    X_train_old = old_selector.fit_transform(X_train, y_train)
    X_test_old = old_selector.transform(X_test)
    
    old_model = ElasticNet(alpha=0.1, l1_ratio=0.5)  # Old high regularization
    old_model.fit(X_train_old, y_train)
    y_pred_old = old_model.predict(X_test_old)
    
    r2_old = r2_score(y_test, y_pred_old)
    logger.info(f"Old approach R²: {r2_old:.4f}")
    
    # New approach: 256 features, minimal regularization
    logger.info("\n--- New Approach (256 features, minimal regularization) ---")
    
    new_selector = GenomicFeatureSelector(method='genomic_ensemble', n_features=256)
    X_train_new = new_selector.fit_transform(X_train, y_train, is_regression=True)
    X_test_new = new_selector.transform(X_test)
    
    new_model = ElasticNet(alpha=0.001, l1_ratio=0.1)  # New minimal regularization
    new_model.fit(X_train_new, y_train)
    y_pred_new = new_model.predict(X_test_new)
    
    r2_new = r2_score(y_test, y_pred_new)
    logger.info(f"New approach R²: {r2_new:.4f}")
    
    improvement = r2_new - r2_old
    logger.info(f"\nImprovement: {improvement:.4f} ({improvement/abs(r2_old)*100:.1f}% relative)")
    
    if r2_new > r2_old:
        logger.info(" New approach shows improvement!")
    else:
        logger.warning(" New approach needs further tuning")
    
    return r2_old, r2_new

def main():
    """Run all tests."""
    logger.info("Starting Genomic Optimization Tests")
    logger.info("=" * 50)
    
    try:
        # Test configuration
        test_configuration_values()
        
        # Test feature selection
        fs_results = test_feature_selection_methods()
        
        # Test model performance
        reg_results, clf_results = test_model_performance()
        
        # Compare approaches
        r2_old, r2_new = compare_old_vs_new_approach()
        
        # Summary
        logger.info("\n" + "=" * 50)
        logger.info("SUMMARY OF RESULTS")
        logger.info("=" * 50)
        
        logger.info("\n Feature Selection Results:")
        for method, result in fs_results.items():
            if isinstance(result, dict):
                logger.info(f"  {method}:  Working")
            else:
                logger.info(f"  {method}:  {result}")
        
        logger.info("\n📈 Regression Performance:")
        for model, result in reg_results.items():
            if isinstance(result, dict):
                r2 = result['r2']
                status = " Target met" if result['target_met'] else ("🟡 Adequate" if result['adequate'] else " Poor")
                logger.info(f"  {model}: R² = {r2:.4f} {status}")
            else:
                logger.info(f"  {model}:  {result}")
        
        logger.info("\n Classification Performance:")
        for model, result in clf_results.items():
            if isinstance(result, dict):
                mcc = result['mcc']
                acc = result['accuracy']
                status = " Target met" if result['target_met'] else ("🟡 Adequate" if result['adequate'] else " Poor")
                logger.info(f"  {model}: MCC = {mcc:.4f}, Acc = {acc:.4f} {status}")
            else:
                logger.info(f"  {model}:  {result}")
        
        logger.info(f"\n🔄 Approach Comparison:")
        logger.info(f"  Old approach R²: {r2_old:.4f}")
        logger.info(f"  New approach R²: {r2_new:.4f}")
        logger.info(f"  Improvement: {r2_new - r2_old:.4f}")
        
        # Overall assessment
        logger.info("\n🏆 OVERALL ASSESSMENT:")
        
        # Check if any regression model meets targets
        reg_success = any(
            isinstance(result, dict) and result.get('target_met', False) 
            for result in reg_results.values()
        )
        
        # Check if any classification model meets targets
        clf_success = any(
            isinstance(result, dict) and result.get('target_met', False) 
            for result in clf_results.values()
        )
        
        if reg_success and clf_success:
            logger.info("🎉 EXCELLENT: Both regression and classification targets achieved!")
        elif reg_success or clf_success:
            logger.info(" GOOD: At least one task meets performance targets")
        elif r2_new > 0.1:  # Some meaningful signal
            logger.info("📈 PROMISING: Showing meaningful improvement, needs fine-tuning")
        else:
            logger.info(" NEEDS WORK: Further optimization required")
        
        logger.info("\n✨ Genomic optimization testing complete!")
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 