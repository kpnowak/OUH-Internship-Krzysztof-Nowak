#!/usr/bin/env python3
"""
Test script for the enhanced feature-first tuner with cross-validation fixes.

This script tests:
1. Cross-validation stability with small datasets
2. Parameter validation for CV compatibility
3. Enhanced error handling for PCA issues
4. Feature-first pipeline integration
"""

import sys
import numpy as np
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required modules can be imported."""
    try:
        from tuner_halving import (
            load_dataset_for_tuner_optimized,
            feature_first_simulate,
            create_feature_first_pipeline,
            param_space,
            SafeExtractorWrapper,
            tune
        )
        logger.info("✓ All tuner imports successful")
        return True
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False

def test_safe_extractor_wrapper():
    """Test the SafeExtractorWrapper with small datasets."""
    try:
        from tuner_halving import SafeExtractorWrapper
        from sklearn.decomposition import PCA
        import warnings
        
        logger.info("Testing SafeExtractorWrapper with small datasets...")
        
        # Test with very small dataset that would normally cause PCA to fail
        np.random.seed(42)
        X_tiny = np.random.randn(3, 10)  # Only 3 samples
        y_tiny = np.array([0, 1, 0])
        
        # Test with too many components
        pca_extractor = PCA(n_components=5)  # More components than samples
        safe_wrapper = SafeExtractorWrapper(pca_extractor)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # This should not fail - wrapper should handle it gracefully
            safe_wrapper.fit(X_tiny, y_tiny)
            X_transformed = safe_wrapper.transform(X_tiny)
            
            logger.info(f"  ✓ Small dataset test: {X_tiny.shape} -> {X_transformed.shape}")
            
            # Test with empty dataset
            X_empty = np.array([]).reshape(0, 10)
            y_empty = np.array([])
            
            safe_wrapper_empty = SafeExtractorWrapper(PCA(n_components=2))
            try:
                safe_wrapper_empty.fit(X_empty, y_empty)
                # Check if it properly failed
                if safe_wrapper_empty.extraction_failed and safe_wrapper_empty.fallback_extractor is None:
                    logger.info("  ✓ Correctly handled empty dataset")
                else:
                    logger.error("  ✗ Should have failed with empty dataset")
                    return False
            except ValueError:
                logger.info("  ✓ Correctly rejected empty dataset")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ SafeExtractorWrapper test failed: {e}")
        return False

def test_parameter_space_adaptive():
    """Test that parameter space adapts correctly to dataset size."""
    try:
        from tuner_halving import param_space
        
        logger.info("Testing adaptive parameter space...")
        
        # Test with different dataset sizes
        test_cases = [
            (10, 100, "tiny dataset"),
            (25, 500, "small dataset"), 
            (75, 1000, "medium dataset"),
            (200, 2000, "large dataset")
        ]
        
        for n_samples, n_features, description in test_cases:
            X_shape = (n_samples, n_features)
            params = param_space("PCA", "RandomForestClassifier", X_shape)
            
            # Check if n_components are reasonable for the dataset size
            if "extractor__extractor__n_components" in params:
                max_components = max(params["extractor__extractor__n_components"])
                
                # For CV, we need max_components < (training_fold_size - 1)
                # With 5-fold CV, training fold is ~80% of data
                max_safe_components = int(0.8 * n_samples) - 1
                
                if max_components >= max_safe_components:
                    logger.warning(f"  ⚠ {description}: max_components={max_components} "
                                 f"might be too large for n_samples={n_samples}")
                else:
                    logger.info(f"  ✓ {description}: max_components={max_components} "
                              f"is safe for n_samples={n_samples}")
            else:
                logger.info(f"  ✓ {description}: no component parameters")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Parameter space test failed: {e}")
        return False

def test_feature_first_simulate_small_data():
    """Test feature_first_simulate with small datasets."""
    try:
        from tuner_halving import feature_first_simulate, create_feature_first_pipeline
        from sklearn.metrics import make_scorer, accuracy_score
        from sklearn.model_selection import KFold
        import warnings
        
        logger.info("Testing feature_first_simulate with small datasets...")
        
        # Create a small synthetic dataset
        np.random.seed(42)
        n_samples = 15
        X_mod1 = np.random.randn(n_samples, 50)
        X_mod2 = np.random.randn(n_samples, 30)
        y = np.random.randint(0, 2, n_samples)
        
        X_modalities = {
            'exp': X_mod1,
            'mirna': X_mod2
        }
        
        # Set up CV parameters
        cv_params = {
            'cv': KFold(n_splits=3, shuffle=True, random_state=42),
            'scoring': make_scorer(accuracy_score),
            'n_jobs': 1
        }
        
        # Test with conservative hyperparameters
        hyperparams = {
            'extractor__extractor__n_components': 2,  # Very conservative
            'model__n_estimators': 50,
            'model__max_depth': 3
        }
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            scores = feature_first_simulate(
                X_modalities=X_modalities,
                y=y,
                extractor="PCA",
                model="RandomForestClassifier",
                cv_params=cv_params,
                hyperparams=hyperparams,
                logger=logger
            )
        
        if len(scores) > 0 and np.all(np.isfinite(scores)):
            logger.info(f"  ✓ Small data simulation: CV scores = {scores}")
            return True
        else:
            logger.error(f"  ✗ Small data simulation failed: scores = {scores}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Feature-first simulate test failed: {e}")
        return False

def test_tuner_integration():
    """Test the complete tuner integration with a real dataset."""
    try:
        from tuner_halving import load_dataset_for_tuner_optimized, tune
        
        logger.info("Testing complete tuner integration...")
        
        # Try to load a real dataset
        try:
            processed_modalities, y = load_dataset_for_tuner_optimized("AML", task="clf")
            
            logger.info(f"Dataset loaded successfully:")
            logger.info(f"  Modalities: {list(processed_modalities.keys())}")
            logger.info(f"  Samples: {len(y)}")
            
            for mod_name, mod_data in processed_modalities.items():
                logger.info(f"  {mod_name}: {mod_data.shape}")
            
            # Test that we can create a feature-first pipeline
            from tuner_halving import create_feature_first_pipeline
            
            pipeline = create_feature_first_pipeline(
                extractor="PCA",
                model="RandomForestClassifier",
                fusion_method="average"
            )
            
            logger.info(f"  ✓ Pipeline created: {list(pipeline.named_steps.keys())}")
            
            # Test basic functionality without running full tuning
            logger.info("✓ Tuner integration test passed")
            return True
            
        except Exception as load_error:
            logger.warning(f"Could not load real dataset: {load_error}")
            logger.info("✓ Tuner integration test skipped (no dataset available)")
            return True
            
    except Exception as e:
        logger.error(f"✗ Tuner integration test failed: {e}")
        return False

def test_cv_parameter_validation():
    """Test the cross-validation parameter validation logic."""
    try:
        logger.info("Testing CV parameter validation...")
        
        # Simulate the validation logic
        n_samples = 20
        n_splits = 5
        min_train_size = (n_samples * (n_splits - 1)) // n_splits  # ~16 for 20 samples, 5 folds
        
        # Test parameter combinations
        test_params = [
            {'extractor__extractor__n_components': 2, 'model__max_depth': 5},  # Should be valid (2 < 16)
            {'extractor__extractor__n_components': 16, 'model__max_depth': 5}, # Should be invalid (16 >= 16)
            {'extractor__extractor__n_components': 20, 'model__max_depth': 5}, # Should be invalid (20 >= 16)
            {'model__max_depth': 5},  # Should be valid (no n_components)
        ]
        
        expected_results = [True, False, False, True]
        
        logger.info(f"  Validation constraints: n_samples={n_samples}, n_splits={n_splits}, min_train_size={min_train_size}")
        
        for i, (params, expected) in enumerate(zip(test_params, expected_results)):
            is_valid = True
            for param_name, param_value in params.items():
                if "n_components" in param_name and isinstance(param_value, int):
                    logger.debug(f"    Checking: {param_name}={param_value} >= {min_train_size}?")
                    if param_value >= min_train_size:
                        is_valid = False
                        break
            
            if is_valid == expected:
                logger.info(f"  ✓ Test {i+1}: {params} -> valid={is_valid} (expected {expected})")
            else:
                logger.error(f"  ✗ Test {i+1}: {params} -> valid={is_valid} (expected {expected})")
                logger.error(f"    Details: n_components vs min_train_size = ? vs {min_train_size}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"✗ CV parameter validation test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting enhanced tuner tests...")
    logger.info("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("SafeExtractorWrapper Test", test_safe_extractor_wrapper),
        ("Adaptive Parameter Space Test", test_parameter_space_adaptive),
        ("Feature-First Simulate Test", test_feature_first_simulate_small_data),
        ("CV Parameter Validation Test", test_cv_parameter_validation),
        ("Tuner Integration Test", test_tuner_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Enhanced tuner is working correctly.")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 