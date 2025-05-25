#!/usr/bin/env python3
"""
Simple test to verify the classification algorithm with one selector and one extractor.
"""

import sys
import os
import logging
import numpy as np
import pandas as pd

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_simple_classification():
    """Test the classification pipeline with simplified configuration."""
    try:
        from models import (
            get_classification_selectors, get_classification_extractors,
            cached_fit_transform_selector_classification,
            cached_fit_transform_extractor_classification,
            validate_and_fix_shape_mismatch
        )
        
        logger.info("=== Testing Simple Classification Pipeline ===")
        
        # Get available selectors and extractors
        selectors = get_classification_selectors()
        extractors = get_classification_extractors()
        
        logger.info(f"Available selectors: {list(selectors.keys())}")
        logger.info(f"Available extractors: {list(extractors.keys())}")
        
        # Create synthetic data for testing
        np.random.seed(42)
        n_samples = 100
        n_features = 50
        n_classes = 3
        
        # Generate random features
        X = np.random.randn(n_samples, n_features)
        
        # Generate random classification labels
        y = np.random.randint(0, n_classes, size=n_samples)
        
        logger.info(f"Generated test data: X={X.shape}, y={y.shape}, n_classes={n_classes}")
        
        # Test shape validation
        X_val, y_val = validate_and_fix_shape_mismatch(X, y, "test_data")
        logger.info(f"After validation: X={X_val.shape}, y={y_val.shape}")
        
        # Test feature selection with fclassif
        logger.info("Testing feature selection...")
        selector_code = "fclassif"
        n_feats = 10
        
        selected_features, X_selected = cached_fit_transform_selector_classification(
            X_val, y_val, selector_code, n_feats, 
            ds_name="test", modality_name="synthetic", fold_idx=0
        )
        
        if selected_features is not None and X_selected is not None:
            logger.info(f"Feature selection successful: {np.sum(selected_features)} features selected, X_selected={X_selected.shape}")
        else:
            logger.error("Feature selection failed")
            return False
        
        # Test feature extraction with PCA
        logger.info("Testing feature extraction...")
        extractor = extractors["PCA"]
        n_components = 8
        
        fitted_extractor, X_extracted = cached_fit_transform_extractor_classification(
            X_val, y_val, extractor, n_components,
            ds_name="test", modality_name="synthetic", fold_idx=0
        )
        
        if fitted_extractor is not None and X_extracted is not None:
            logger.info(f"Feature extraction successful: X_extracted={X_extracted.shape}")
        else:
            logger.error("Feature extraction failed")
            return False
        
        # Test model training
        logger.info("Testing model training...")
        from models import get_model_object
        
        model = get_model_object("LogisticRegression", random_state=42)
        
        # Split data for training and validation
        split_idx = int(0.8 * len(X_extracted))
        X_train = X_extracted[:split_idx]
        X_test = X_extracted[split_idx:]
        y_train = y_val[:split_idx]
        y_test = y_val[split_idx:]
        
        logger.info(f"Train/test split: train={X_train.shape}, test={X_test.shape}")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Model trained successfully! Accuracy: {accuracy:.3f}")
        
        logger.info("=== All tests passed! ===")
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_classification()
    if success:
        print("✅ Simple classification test PASSED")
        sys.exit(0)
    else:
        print("❌ Simple classification test FAILED")
        sys.exit(1) 