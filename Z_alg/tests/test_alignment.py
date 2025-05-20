#!/usr/bin/env python3

"""
Test script for data alignment in the Z_alg pipeline.
"""

import numpy as np
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our alignment functions
from Z_alg._process_single_modality import verify_data_alignment, align_samples_to_modalities
from Z_alg.fusion import merge_modalities, ModalityImputer

def test_data_alignment():
    """Test the data alignment process with mock data."""
    logger.info("Testing data alignment...")
    
    # Create mock data with mismatched dimensions
    X1 = np.random.rand(11, 5)
    X2 = np.random.rand(10, 8)
    y = np.random.rand(10)
    
    logger.info(f"Initial shapes: X1={X1.shape}, X2={X2.shape}, y={y.shape}")
    
    # Test verify_data_alignment
    X1_aligned, y1_aligned = verify_data_alignment(X1, y, name="test_X1")
    logger.info(f"After alignment: X1_aligned={X1_aligned.shape}, y1_aligned={len(y1_aligned)}")
    
    # Test merge_modalities with mismatched array dimensions
    imputer = ModalityImputer()
    merged = merge_modalities(X1, X2, imputer=imputer, is_train=True)
    logger.info(f"After merging: merged={merged.shape}")
    
    # Test full alignment pipeline
    merged_aligned, y_aligned = verify_data_alignment(merged, y, name="merged")
    logger.info(f"Final shapes: merged_aligned={merged_aligned.shape}, y_aligned={len(y_aligned)}")
    
    # Verify the result
    assert merged_aligned.shape[0] == len(y_aligned), "Shapes should match after alignment"
    logger.info("Alignment test passed!")
    
    return merged_aligned, y_aligned

def test_id_alignment():
    """Test the ID-based sample alignment."""
    logger.info("Testing ID alignment...")
    
    # Create mock modality dataframes
    modality1 = pd.DataFrame(np.random.rand(5, 10), 
                           columns=[f"sample_{i}" for i in range(10)])
    modality2 = pd.DataFrame(np.random.rand(8, 8), 
                           columns=[f"sample_{i}" for i in range(2, 10)])
    
    # Create a mock data modalities dict
    data_modalities = {
        "modality1": modality1,
        "modality2": modality2
    }
    
    # Test ID alignment
    id_train = [f"sample_{i}" for i in range(8)]
    id_val = [f"sample_{i}" for i in range(8, 10)]
    
    valid_train_ids, valid_val_ids = align_samples_to_modalities(id_train, id_val, data_modalities)
    
    logger.info(f"Original train IDs: {len(id_train)}, valid train IDs: {len(valid_train_ids)}")
    logger.info(f"Original val IDs: {len(id_val)}, valid val IDs: {len(valid_val_ids)}")
    
    # Verify the result
    assert set(valid_train_ids) == set([f"sample_{i}" for i in range(2, 8)]), "Train IDs should be properly aligned"
    assert set(valid_val_ids) == set([f"sample_{i}" for i in range(8, 10)]), "Val IDs should be properly aligned"
    logger.info("ID alignment test passed!")
    
    return valid_train_ids, valid_val_ids

if __name__ == "__main__":
    try:
        # Test data alignment
        merged, y = test_data_alignment()
        logger.info(f"Data alignment successful: {merged.shape}, {len(y)}")
        
        # Test ID alignment
        train_ids, val_ids = test_id_alignment()
        logger.info(f"ID alignment successful: train={len(train_ids)}, val={len(val_ids)}")
        
        logger.info("All tests passed!")
    except Exception as e:
        logger.error(f"Error in tests: {str(e)}")
        raise 