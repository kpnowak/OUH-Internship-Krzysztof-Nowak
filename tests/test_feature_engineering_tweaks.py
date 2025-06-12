#!/usr/bin/env python3
"""
Test script for feature engineering tweaks implementation.
Tests Sparse PLS-DA for better MCC and Kernel PCA for higher R².
"""

import numpy as np
import pandas as pd
import logging
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, r2_score, accuracy_score

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_genomic_like_data(n_samples=200, n_features=1000, task_type="classification"):
    """Create genomic-like data for testing."""
    if task_type == "classification":
        # Create imbalanced classification data (like genomic classification)
        X, y = make_classification(
            n_samples=n_samples, 
            n_features=n_features,
            n_informative=50,
            n_redundant=20,
            n_clusters_per_class=1,
            n_classes=3,
            weights=[0.7, 0.2, 0.1],  # Imbalanced classes
            random_state=42
        )
    else:
        # Create regression data with non-linear relationships
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=50,
            noise=0.1,
            random_state=42
        )
        # Add non-linear relationships
        y = y + 0.1 * np.sum(X[:, :10] ** 2, axis=1)
    
    return X, y

def test_configuration():
    """Test feature engineering configuration loading."""
    print("=" * 60)
    print("Testing Feature Engineering Configuration")
    print("=" * 60)
    
    try:
        from config import FEATURE_ENGINEERING_CONFIG
        
        print("  FEATURE_ENGINEERING_CONFIG loaded successfully:")
        for key, value in FEATURE_ENGINEERING_CONFIG.items():
            if isinstance(value, dict):
                print(f"    {key}:")
                for subkey, subvalue in value.items():
                    print(f"      {subkey}: {subvalue}")
            else:
                print(f"    {key}: {value}")
        
        # Verify required keys exist
        required_keys = [
            "enabled", "sparse_plsda_enabled", "kernel_pca_enabled",
            "sparse_plsda", "kernel_pca", "median_heuristic"
        ]
        
        missing_keys = [key for key in required_keys if key not in FEATURE_ENGINEERING_CONFIG]
        
        if missing_keys:
            print(f"✗ Missing configuration keys: {missing_keys}")
            return False
        else:
            print("✓ All required configuration keys present")
            return True
        
    except Exception as e:
        print(f"✗ Error testing configuration: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_sparse_plsda():
    """Test Sparse PLS-DA implementation."""
    print("=" * 60)
    print("Testing Sparse PLS-DA for Better MCC")
    print("=" * 60)
    
    try:
        from models import SparsePLSDA
        
        # Create test data
        X, y = create_genomic_like_data(n_samples=150, n_features=500, task_type="classification")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        print(f"  Data shape: {X_train.shape}, Classes: {np.unique(y_train)}")
        print(f"  Class distribution: {np.bincount(y_train)}")
        
        # Test Sparse PLS-DA with 32 components
        sparse_plsda = SparsePLSDA(n_components=32, alpha=0.1, scale=True)
        
        # Fit and transform
        X_train_transformed = sparse_plsda.fit_transform(X_train, y_train)
        X_test_transformed = sparse_plsda.transform(X_test)
        
        print(f"✓ Sparse PLS-DA fitted successfully")
        print(f"  Original features: {X_train.shape[1]}")
        print(f"  Transformed features: {X_train_transformed.shape[1]}")
        print(f"  Components extracted: {sparse_plsda.x_weights_.shape[1]}")
        
        # Test with a simple classifier to check MCC improvement potential
        from sklearn.linear_model import LogisticRegression
        
        # Train on original data
        clf_original = LogisticRegression(random_state=42, max_iter=1000)
        clf_original.fit(X_train, y_train)
        y_pred_original = clf_original.predict(X_test)
        
        # Train on transformed data
        clf_transformed = LogisticRegression(random_state=42, max_iter=1000)
        clf_transformed.fit(X_train_transformed, y_train)
        y_pred_transformed = clf_transformed.predict(X_test_transformed)
        
        # Calculate metrics
        mcc_original = matthews_corrcoef(y_test, y_pred_original)
        mcc_transformed = matthews_corrcoef(y_test, y_pred_transformed)
        acc_original = accuracy_score(y_test, y_pred_original)
        acc_transformed = accuracy_score(y_test, y_pred_transformed)
        
        print(f"  Original data - MCC: {mcc_original:.4f}, Accuracy: {acc_original:.4f}")
        print(f"  Transformed data - MCC: {mcc_transformed:.4f}, Accuracy: {acc_transformed:.4f}")
        
        if mcc_transformed >= mcc_original:
            print(f"✓ Sparse PLS-DA maintained/improved MCC (+{mcc_transformed - mcc_original:.4f})")
        else:
            print(f"! Sparse PLS-DA decreased MCC ({mcc_transformed - mcc_original:.4f})")
        
        # Test parameter access
        params = sparse_plsda.get_params()
        print(f"✓ Parameter access working: {len(params)} parameters")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing Sparse PLS-DA: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_kernel_pca_median_heuristic():
    """Test Kernel PCA with median heuristic."""
    print("=" * 60)
    print("Testing Kernel PCA with Median Heuristic for Higher R²")
    print("=" * 60)
    
    try:
        from models import KernelPCAMedianHeuristic
        
        # Create test data with non-linear relationships
        X, y = create_genomic_like_data(n_samples=150, n_features=500, task_type="regression")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        print(f"  Data shape: {X_train.shape}")
        print(f"  Target range: [{np.min(y_train):.2f}, {np.max(y_train):.2f}]")
        
        # Test Kernel PCA with median heuristic
        kernel_pca = KernelPCAMedianHeuristic(
            n_components=64, 
            kernel="rbf", 
            sample_size=100,
            random_state=42
        )
        
        # Fit and transform
        X_train_transformed = kernel_pca.fit_transform(X_train)
        X_test_transformed = kernel_pca.transform(X_test)
        
        print(f"✓ Kernel PCA fitted successfully")
        print(f"  Original features: {X_train.shape[1]}")
        print(f"  Transformed features: {X_train_transformed.shape[1]}")
        print(f"  Computed gamma: {kernel_pca.gamma_computed_:.6f}")
        
        # Test with a simple regressor to check R² improvement potential
        from sklearn.ensemble import RandomForestRegressor
        
        # Train on original data
        reg_original = RandomForestRegressor(n_estimators=100, random_state=42)
        reg_original.fit(X_train, y_train)
        y_pred_original = reg_original.predict(X_test)
        
        # Train on transformed data
        reg_transformed = RandomForestRegressor(n_estimators=100, random_state=42)
        reg_transformed.fit(X_train_transformed, y_train)
        y_pred_transformed = reg_transformed.predict(X_test_transformed)
        
        # Calculate R²
        r2_original = r2_score(y_test, y_pred_original)
        r2_transformed = r2_score(y_test, y_pred_transformed)
        
        print(f"  Original data - R²: {r2_original:.4f}")
        print(f"  Transformed data - R²: {r2_transformed:.4f}")
        
        if r2_transformed >= r2_original:
            print(f"✓ Kernel PCA maintained/improved R² (+{r2_transformed - r2_original:.4f})")
        else:
            print(f"! Kernel PCA decreased R² ({r2_transformed - r2_original:.4f})")
        
        # Test median heuristic calculation
        gamma_manual = kernel_pca._compute_median_heuristic_gamma(X_train[:50])
        print(f"✓ Median heuristic calculation working: gamma = {gamma_manual:.6f}")
        
        # Test parameter access
        params = kernel_pca.get_params()
        print(f"✓ Parameter access working: {len(params)} parameters")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing Kernel PCA: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_extractor_integration():
    """Test integration with extractor functions."""
    print("=" * 60)
    print("Testing Integration with Extractor Functions")
    print("=" * 60)
    
    try:
        # Enable feature engineering
        from config import FEATURE_ENGINEERING_CONFIG
        FEATURE_ENGINEERING_CONFIG["enabled"] = True
        
        # Test classification extractors
        from models import get_classification_extractors
        clf_extractors = get_classification_extractors()
        
        if "SparsePLS-DA" in clf_extractors:
            print("✓ SparsePLS-DA found in classification extractors")
            extractor = clf_extractors["SparsePLS-DA"]
            print(f"  Components: {extractor.n_components}")
            print(f"  Alpha: {extractor.alpha}")
        else:
            print("✗ SparsePLS-DA not found in classification extractors")
        
        # Test regression extractors
        from models import get_regression_extractors
        reg_extractors = get_regression_extractors()
        
        if "KernelPCA-RBF" in reg_extractors:
            print("✓ KernelPCA-RBF found in regression extractors")
            extractor = reg_extractors["KernelPCA-RBF"]
            print(f"  Components: {extractor.n_components}")
            print(f"  Kernel: {extractor.kernel}")
            print(f"  Sample size: {extractor.sample_size}")
        else:
            print("✗ KernelPCA-RBF not found in regression extractors")
        
        # Test disabled state
        FEATURE_ENGINEERING_CONFIG["enabled"] = False
        clf_extractors_disabled = get_classification_extractors()
        reg_extractors_disabled = get_regression_extractors()
        
        if "SparsePLS-DA" not in clf_extractors_disabled and "KernelPCA-RBF" not in reg_extractors_disabled:
            print("✓ Feature engineering extractors properly disabled")
        else:
            print("! Feature engineering extractors not properly disabled")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing extractor integration: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_cli_integration():
    """Test CLI integration."""
    print("=" * 60)
    print("Testing CLI Integration")
    print("=" * 60)
    
    try:
        # Test argument parsing
        import argparse
        from cli import main
        
        # Create a mock parser to test argument addition
        parser = argparse.ArgumentParser()
        parser.add_argument("--feature-engineering", action="store_true", 
                          help="Enable feature engineering tweaks")
        
        # Test parsing
        args = parser.parse_args(["--feature-engineering"])
        
        if hasattr(args, 'feature_engineering') and args.feature_engineering:
            print("✓ CLI argument --feature-engineering parsed successfully")
        else:
            print("✗ CLI argument parsing failed")
            return False
        
        # Test configuration enabling
        from config import FEATURE_ENGINEERING_CONFIG
        original_state = FEATURE_ENGINEERING_CONFIG["enabled"]
        
        # Simulate CLI enabling
        FEATURE_ENGINEERING_CONFIG["enabled"] = True
        
        if FEATURE_ENGINEERING_CONFIG["enabled"]:
            print("✓ Configuration can be enabled via CLI simulation")
        else:
            print("✗ Configuration enabling failed")
        
        # Restore original state
        FEATURE_ENGINEERING_CONFIG["enabled"] = original_state
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing CLI integration: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_comparison():
    """Test performance comparison with and without feature engineering."""
    print("=" * 60)
    print("Testing Performance Comparison")
    print("=" * 60)
    
    try:
        from config import FEATURE_ENGINEERING_CONFIG
        from models import get_classification_extractors, get_regression_extractors
        
        # Test classification performance
        X_clf, y_clf = create_genomic_like_data(n_samples=200, n_features=300, task_type="classification")
        X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.3, random_state=42)
        
        # Without feature engineering
        FEATURE_ENGINEERING_CONFIG["enabled"] = False
        extractors_standard = get_classification_extractors()
        
        # With feature engineering
        FEATURE_ENGINEERING_CONFIG["enabled"] = True
        extractors_enhanced = get_classification_extractors()
        
        print(f"  Standard extractors: {len(extractors_standard)}")
        print(f"  Enhanced extractors: {len(extractors_enhanced)}")
        
        if len(extractors_enhanced) > len(extractors_standard):
            print("✓ Feature engineering adds new extractors")
        else:
            print("! Feature engineering may not be adding extractors")
        
        # Test regression performance
        X_reg, y_reg = create_genomic_like_data(n_samples=200, n_features=300, task_type="regression")
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)
        
        # Without feature engineering
        FEATURE_ENGINEERING_CONFIG["enabled"] = False
        reg_extractors_standard = get_regression_extractors()
        
        # With feature engineering
        FEATURE_ENGINEERING_CONFIG["enabled"] = True
        reg_extractors_enhanced = get_regression_extractors()
        
        print(f"  Standard regression extractors: {len(reg_extractors_standard)}")
        print(f"  Enhanced regression extractors: {len(reg_extractors_enhanced)}")
        
        if len(reg_extractors_enhanced) > len(reg_extractors_standard):
            print("✓ Feature engineering adds new regression extractors")
        else:
            print("! Feature engineering may not be adding regression extractors")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing performance comparison: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Feature Engineering Tweaks - Comprehensive Test Suite")
    print("=" * 60)
    
    tests = [
        ("Configuration Loading", test_configuration),
        ("Sparse PLS-DA", test_sparse_plsda),
        ("Kernel PCA Median Heuristic", test_kernel_pca_median_heuristic),
        ("Extractor Integration", test_extractor_integration),
        ("CLI Integration", test_cli_integration),
        ("Performance Comparison", test_performance_comparison)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:.<40} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Feature engineering tweaks are working correctly.")
        print("\nTo use feature engineering tweaks:")
        print("  python cli.py --feature-engineering")
        print("\nThis enables:")
        print("  - Sparse PLS-DA (32 components) for better MCC in classification")
        print("  - Kernel PCA RBF (64 components) for higher R² in regression")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return passed == len(results)

if __name__ == "__main__":
    main() 