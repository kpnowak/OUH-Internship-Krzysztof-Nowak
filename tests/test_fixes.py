#!/usr/bin/env python3
"""
Test script to verify that the error fixes work correctly.
"""

import sys
import logging
from pathlib import Path

# Add current directory to path
sys.path.append('.')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_config_loading():
    """Test that the configuration loads correctly."""
    try:
        from config import REGRESSION_DATASETS, CLASSIFICATION_DATASETS, WARNING_SUPPRESSION_CONFIG
        
        logger.info("✅ Configuration loaded successfully")
        logger.info(f"   - Regression datasets: {len(REGRESSION_DATASETS)}")
        logger.info(f"   - Classification datasets: {len(CLASSIFICATION_DATASETS)}")
        logger.info(f"   - Warning suppression enabled: {WARNING_SUPPRESSION_CONFIG.get('suppress_sklearn_warnings', False)}")
        
        # Check if Sarcoma is now included
        reg_names = [ds['name'] for ds in REGRESSION_DATASETS]
        if 'Sarcoma' in reg_names:
            logger.info("   - ✅ Sarcoma dataset is now included in regression datasets")
        else:
            logger.warning("   - ❌ Sarcoma dataset is still missing from regression datasets")
        
        return True
    except Exception as e:
        logger.error(f"❌ Configuration loading failed: {str(e)}")
        return False

def test_data_loading():
    """Test basic data loading functionality."""
    try:
        from data_io import load_dataset
        
        # Try to load a small dataset
        logger.info("Testing data loading with AML dataset...")
        modalities, y, common_ids = load_dataset(
            ds_name='aml',
            modalities=['exp'],  # Just load one modality for testing
            outcome_col='lab_procedure_bone_marrow_blast_cell_outcome_percent_value',
            task_type='regression'
        )
        
        if modalities and len(common_ids) > 0:
            logger.info(f"✅ Data loading successful: {len(common_ids)} samples, {list(modalities.keys())} modalities")
            return True
        else:
            logger.warning("❌ Data loading returned empty results")
            return False
            
    except Exception as e:
        logger.error(f"❌ Data loading failed: {str(e)}")
        return False

def test_mad_analysis():
    """Test MAD analysis functionality."""
    try:
        from mad_analysis import calculate_mad
        import numpy as np
        
        # Create test data
        test_data = np.random.rand(10, 20)  # 10 features, 20 samples
        mad_value = calculate_mad(test_data, axis=1)
        
        if isinstance(mad_value, (int, float)) and mad_value >= 0:
            logger.info(f"✅ MAD calculation successful: {mad_value:.6f}")
            return True
        else:
            logger.warning(f"❌ MAD calculation returned unexpected value: {mad_value}")
            return False
            
    except Exception as e:
        logger.error(f"❌ MAD calculation failed: {str(e)}")
        return False

def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("TESTING ERROR FIXES")
    logger.info("=" * 60)
    
    tests = [
        ("Configuration Loading", test_config_loading),
        ("Data Loading", test_data_loading),
        ("MAD Analysis", test_mad_analysis),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {str(e)}")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED - Fixes are working correctly!")
    else:
        logger.warning(f"⚠️  {total - passed} tests failed - Some issues may remain")
    
    logger.info("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 