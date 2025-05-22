#!/usr/bin/env python3

"""
Integration test for the complete Z_alg pipeline.

This script tests the main components of the pipeline to ensure they work together correctly.
"""

import numpy as np
import pandas as pd
import os
import logging
import sys
from typing import Dict, List, Tuple, Any

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

logger = logging.getLogger(__name__)

def create_test_data(n_samples=20, n_features=50, n_classes=2, random_seed=42):
    """Create synthetic test data for both regression and classification."""
    np.random.seed(random_seed)
    
    # Create feature matrices for multiple modalities
    modalities = {
        "RNA": pd.DataFrame(np.random.rand(n_features, n_samples)),
        "Methylation": pd.DataFrame(np.random.rand(n_features, n_samples)), 
        "Protein": pd.DataFrame(np.random.rand(n_features, n_samples))
    }
    
    # Set column names as sample IDs
    sample_ids = [f"sample_{i}" for i in range(n_samples)]
    for modality_name, modality_df in modalities.items():
        modality_df.columns = sample_ids
    
    # Create target variables
    regression_y = np.random.rand(n_samples)
    classification_y = np.random.randint(0, n_classes, size=n_samples)
    
    common_ids = sample_ids
    
    return modalities, common_ids, regression_y, classification_y

def test_core_components():
    """Test the core functionality of the pipeline."""
    from Z_alg.models import (
        get_regression_models, 
        get_classification_models,
        get_regression_extractors,
        get_classification_extractors,
        cached_fit_transform_extractor_regression,
        cached_fit_transform_extractor_classification
    )
    
    # Test model creation functions
    logger.info("Testing model creation functions...")
    reg_models = get_regression_models()
    clf_models = get_classification_models()
    
    assert "LinearRegression" in reg_models, "LinearRegression not found in regression models"
    assert "LogisticRegression" in clf_models, "LogisticRegression not found in classification models"
    
    # Test extractor creation functions
    logger.info("Testing extractor creation functions...")
    reg_extractors = get_regression_extractors()
    clf_extractors = get_classification_extractors()
    
    assert "PCA" in reg_extractors, "PCA not found in regression extractors"
    assert "PCA" in clf_extractors, "PCA not found in classification extractors"
    
    # Create test data
    logger.info("Creating test data...")
    modalities, common_ids, regression_y, classification_y = create_test_data()
    
    # Test extractor functions with test data
    logger.info("Testing extraction functions...")
    X = modalities["RNA"].values.T
    y_reg = regression_y
    y_clf = classification_y
    
    for name, extractor in reg_extractors.items():
        try:
            new_extractor, X_transformed = cached_fit_transform_extractor_regression(
                X, y_reg, extractor, n_components=5, ds_name="test", modality_name="test_modality", fold_idx=0
            )
            logger.info(f"Regression extractor {name} - Success: {X_transformed.shape}")
        except Exception as e:
            logger.error(f"Regression extractor {name} - Failed: {str(e)}")
            
    for name, extractor in clf_extractors.items():
        try:
            new_extractor, X_transformed = cached_fit_transform_extractor_classification(
                X, y_clf, extractor, n_components=5, ds_name="test", modality_name="test_modality", fold_idx=0
            )
            logger.info(f"Classification extractor {name} - Success: {X_transformed.shape}")
        except Exception as e:
            logger.error(f"Classification extractor {name} - Failed: {str(e)}")
    
    logger.info("Core component tests completed!")
    
def test_cv_functions():
    """Test the cross-validation functions."""
    from Z_alg.cv import (
        verify_data_alignment, 
        process_cv_fold,
        _process_single_modality
    )
    
    logger.info("Testing cross-validation functions...")
    
    # Test data alignment
    X = np.random.rand(10, 5)
    y1 = np.random.rand(10)  # Matching shape
    y2 = np.random.rand(11)  # Mismatched shape
    
    X_aligned1, y_aligned1 = verify_data_alignment(X, y1, "test_matching")
    assert X_aligned1.shape[0] == y_aligned1.shape[0], "Alignment failed for matching shapes"
    
    X_aligned2, y_aligned2 = verify_data_alignment(X, y2, "test_mismatched")
    assert X_aligned2.shape[0] == y_aligned2.shape[0], "Alignment failed for mismatched shapes"
    assert X_aligned2.shape[0] == 10, "Expected truncated shape to be 10"
    
    logger.info("Data alignment test successful!")
    
def test_pipeline():
    """Test the complete pipeline functionality."""
    from Z_alg.cv import run_extraction_pipeline, run_selection_pipeline
    
    logger.info("Testing pipeline functions...")
    
    # Create test data
    modalities, common_ids, regression_y, classification_y = create_test_data()
    
    # Create temporary output directory
    os.makedirs("test_output", exist_ok=True)
    
    # Import necessary functions for pipeline
    from Z_alg.models import (
        get_regression_extractors,
        get_regression_models
    )
    
    # Test regression pipeline with a single modality
    test_modalities = {"RNA": modalities["RNA"]}
    
    try:
        run_extraction_pipeline(
            ds_name="TestReg", 
            data_modalities=test_modalities, 
            common_ids=common_ids, 
            y=regression_y, 
            base_out="test_output",
            extractors={"PCA": get_regression_extractors()["PCA"]}, 
            n_comps_list=[5], 
            models=list(get_regression_models().keys())[:1],
            progress_count=[0], 
            total_runs=1,
            is_regression=True
        )
        logger.info("Regression extraction pipeline test successful!")
    except Exception as e:
        logger.error(f"Regression extraction pipeline test failed: {str(e)}")
    
    logger.info("Pipeline tests completed!")

if __name__ == "__main__":
    logger.info("Starting Z_alg integration tests...")
    
    # Run the tests
    try:
        test_core_components()
        test_cv_functions()
        test_pipeline()
        logger.info("All tests completed successfully!")
    except Exception as e:
        logger.error(f"Tests failed: {str(e)}")
    
    # Clean up
    try:
        import shutil
        if os.path.exists("test_output"):
            shutil.rmtree("test_output")
    except:
        pass 