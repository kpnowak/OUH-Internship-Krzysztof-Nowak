#!/usr/bin/env python3
"""
Test suite for fusion upgrades implementation.
"""

import numpy as np
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_configuration_loading():
    """Test configuration loading"""
    try:
        from config import FUSION_UPGRADES_CONFIG
        logger.info(" FUSION_UPGRADES_CONFIG loaded successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Configuration loading failed: {e}")
        return False

def test_attention_fuser():
    """Test AttentionFuser"""
    try:
        from fusion import AttentionFuser
        
        # Create test data
        np.random.seed(42)
        X1 = np.random.randn(50, 20)
        X2 = np.random.randn(50, 15)
        y = np.random.randn(50)
        
        # Test AttentionFuser
        fuser = AttentionFuser(hidden_dim=16, max_epochs=10)
        X_fused = fuser.fit_transform([X1, X2], y)
        
        logger.info(f" AttentionFuser successful: {X_fused.shape}")
        return True
    except Exception as e:
        logger.error(f"✗ AttentionFuser test failed: {e}")
        return False

def test_late_fusion_stacking():
    """Test LateFusionStacking"""
    try:
        from fusion import LateFusionStacking
        
        # Create test data
        np.random.seed(42)
        X1 = np.random.randn(50, 20)
        X2 = np.random.randn(50, 15)
        y = np.random.randint(0, 2, 50)
        
        # Test LateFusionStacking
        stacker = LateFusionStacking(is_regression=False, cv_folds=3)
        stacker.fit([X1, X2], y)
        predictions = stacker.predict([X1, X2])
        
        logger.info(f" LateFusionStacking successful: {predictions.shape}")
        return True
    except Exception as e:
        logger.error(f"✗ LateFusionStacking test failed: {e}")
        return False

def test_merge_modalities():
    """Test merge_modalities integration"""
    try:
        from fusion import merge_modalities
        
        # Create test data
        np.random.seed(42)
        X1 = np.random.randn(40, 15)
        X2 = np.random.randn(40, 10)
        y = np.random.randn(40)
        
        # Test attention_weighted
        result, fitted = merge_modalities(
            X1, X2,
            strategy="attention_weighted",
            y=y,
            is_regression=True,
            is_train=True,
            fusion_params={"hidden_dim": 8, "max_epochs": 5}
        )
        
        logger.info(f" merge_modalities successful: {result.shape}")
        return True
    except Exception as e:
        logger.error(f"✗ merge_modalities test failed: {e}")
        return False

if __name__ == "__main__":
    tests = [
        test_configuration_loading,
        test_attention_fuser,
        test_late_fusion_stacking,
        test_merge_modalities
    ]
    
    results = [test() for test in tests]
    passed = sum(results)
    total = len(results)
    
    logger.info(f"Results: {passed}/{total} tests passed")
    if passed == total:
        logger.info("🎉 All tests passed!")
        logger.info("Usage: python cli.py --fusion-upgrades")
    else:
        logger.error(" Some tests failed")
