#!/usr/bin/env python3
"""
Specific test for Sarcoma dataset to verify the fixes work correctly.
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

def test_sarcoma_loading():
    """Test loading the Sarcoma dataset specifically."""
    try:
        from data_io import load_dataset
        from config import REGRESSION_DATASETS
        
        # Find Sarcoma configuration
        sarcoma_config = None
        for ds_conf in REGRESSION_DATASETS:
            if ds_conf['name'] == 'Sarcoma':
                sarcoma_config = ds_conf
                break
        
        if not sarcoma_config:
            logger.error("❌ Sarcoma dataset not found in REGRESSION_DATASETS")
            return False
        
        logger.info("Testing Sarcoma dataset loading...")
        logger.info(f"Configuration: {sarcoma_config}")
        
        # Try to load the dataset
        modalities, y, common_ids = load_dataset(
            ds_name='sarcoma',
            modalities=['exp'],  # Just load one modality for testing
            outcome_col=sarcoma_config['outcome_col'],
            task_type='regression'
        )
        
        if modalities and len(common_ids) > 0:
            logger.info(f"✅ Sarcoma loading successful: {len(common_ids)} samples, {list(modalities.keys())} modalities")
            
            # Check that y is numeric
            import numpy as np
            if np.issubdtype(y.dtype, np.number):
                logger.info(f"✅ Outcome data is properly numeric: dtype={y.dtype}")
                logger.info(f"   - Mean: {y.mean():.3f}")
                logger.info(f"   - Std: {y.std():.3f}")
                logger.info(f"   - Range: [{y.min():.3f}, {y.max():.3f}]")
                return True
            else:
                logger.error(f"❌ Outcome data is not numeric: dtype={y.dtype}")
                return False
        else:
            logger.warning("❌ Sarcoma loading returned empty results")
            return False
            
    except Exception as e:
        logger.error(f"❌ Sarcoma loading failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_sarcoma_cli_processing():
    """Test Sarcoma dataset processing through CLI interface."""
    try:
        from cli import process_dataset
        from config import REGRESSION_DATASETS
        
        # Find Sarcoma configuration
        sarcoma_config = None
        for ds_conf in REGRESSION_DATASETS:
            if ds_conf['name'] == 'Sarcoma':
                sarcoma_config = ds_conf
                break
        
        if not sarcoma_config:
            logger.error("❌ Sarcoma dataset not found in REGRESSION_DATASETS")
            return False
        
        logger.info("Testing Sarcoma dataset processing through CLI...")
        
        # Process the dataset
        result = process_dataset(sarcoma_config, is_regression=True)
        
        if result:
            logger.info(f"✅ Sarcoma CLI processing successful")
            logger.info(f"   - Dataset name: {result['name']}")
            logger.info(f"   - Modalities: {list(result['modalities'].keys())}")
            logger.info(f"   - Common IDs: {len(result['common_ids'])}")
            logger.info(f"   - Target data type: {result['y_aligned'].dtype}")
            
            # Check that y_aligned is numeric
            import numpy as np
            if np.issubdtype(result['y_aligned'].dtype, np.number):
                logger.info(f"✅ CLI target data is properly numeric")
                return True
            else:
                logger.error(f"❌ CLI target data is not numeric: {result['y_aligned'].dtype}")
                return False
        else:
            logger.error("❌ Sarcoma CLI processing returned None")
            return False
            
    except Exception as e:
        logger.error(f"❌ Sarcoma CLI processing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run Sarcoma-specific tests."""
    logger.info("=" * 60)
    logger.info("TESTING SARCOMA DATASET FIXES")
    logger.info("=" * 60)
    
    tests = [
        ("Sarcoma Data Loading", test_sarcoma_loading),
        ("Sarcoma CLI Processing", test_sarcoma_cli_processing),
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
    logger.info(f"SARCOMA TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL SARCOMA TESTS PASSED - Fixes are working correctly!")
    else:
        logger.warning(f"⚠️  {total - passed} Sarcoma tests failed - Some issues may remain")
    
    logger.info("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 