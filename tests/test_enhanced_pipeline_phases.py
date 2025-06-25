#!/usr/bin/env python3
"""
Comprehensive Test for Enhanced Pipeline Phases.
Tests all 4 architectural improvements:
1. Early Data Quality Pipeline
2. Fusion-Aware Feature Selection
3. Centralized Missing Data Management
4. Coordinated Validation Framework
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
import sys
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_test_data(dataset_name: str = "TestDataset", 
                      n_samples: int = 100,
                      missing_rate: float = 0.1) -> Tuple[Dict[str, Tuple[np.ndarray, List[str]]], np.ndarray]:
    """Generate synthetic test data for pipeline testing."""
    
    np.random.seed(42)  # For reproducibility
    
    # Generate sample IDs
    sample_ids = [f"Sample_{i:03d}" for i in range(n_samples)]
    
    # Generate modality data
    modality_data_dict = {}
    
    # Gene expression modality (high dimensional)
    gene_exp = np.random.randn(n_samples, 500)
    # Add some missing values
    if missing_rate > 0:
        missing_mask = np.random.random((n_samples, 500)) < missing_rate
        gene_exp[missing_mask] = np.nan
    modality_data_dict['exp'] = (gene_exp, sample_ids.copy())
    
    # Methylation modality (medium dimensional)
    methylation = np.random.beta(0.5, 0.5, (n_samples, 200))
    # Add some missing values
    if missing_rate > 0:
        missing_mask = np.random.random((n_samples, 200)) < missing_rate
        methylation[missing_mask] = np.nan
    modality_data_dict['methy'] = (methylation, sample_ids.copy())
    
    # miRNA modality (low dimensional)
    mirna = np.random.lognormal(0, 1, (n_samples, 100))
    # Add some missing values
    if missing_rate > 0:
        missing_mask = np.random.random((n_samples, 100)) < missing_rate
        mirna[missing_mask] = np.nan
    modality_data_dict['mirna'] = (mirna, sample_ids.copy())
    
    # Generate target variable
    # For classification: binary target
    y = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    
    logger.info(f"Generated test data for {dataset_name}:")
    logger.info(f"  Samples: {n_samples}")
    logger.info(f"  Missing rate: {missing_rate:.1%}")
    logger.info(f"  Modalities: {list(modality_data_dict.keys())}")
    for name, (data, _) in modality_data_dict.items():
        logger.info(f"    {name}: {data.shape}")
    
    return modality_data_dict, y

def test_phase_1_data_quality():
    """Test Phase 1: Early Data Quality Pipeline."""
    logger.info("\n" + "="*60)
    logger.info("TESTING PHASE 1: Early Data Quality Pipeline")
    logger.info("="*60)
    
    try:
        from data_quality import run_early_data_quality_pipeline
        
        # Generate test data
        modality_data_dict, y = generate_test_data("Phase1Test", n_samples=50, missing_rate=0.05)
        
        # Test basic functionality
        logger.info("Test 1.1: Basic data quality assessment")
        quality_report, guidance = run_early_data_quality_pipeline(
            modality_data_dict, y, "Phase1Test", "classification"
        )
        
        assert 'overall_quality_score' in quality_report
        logger.info(f"✓ Quality score: {quality_report['overall_quality_score']:.3f}")
        
        logger.info(" Phase 1 tests PASSED")
        return True
        
    except ImportError as e:
        logger.error(f" Phase 1 module import failed: {e}")
        return False
    except Exception as e:
        logger.error(f" Phase 1 test failed: {e}")
        return False

def test_phase_2_fusion_aware():
    """Test Phase 2: Fusion-Aware Feature Selection."""
    logger.info("\n" + "="*60)
    logger.info("TESTING PHASE 2: Fusion-Aware Feature Selection")
    logger.info("="*60)
    
    try:
        from fusion_aware_preprocessing import determine_optimal_fusion_order
        
        # Test order determination for different fusion methods
        logger.info("Test 2.1: Optimal order determination")
        
        fusion_methods = ['average', 'sum', 'mkl', 'weighted_concat', 'early_fusion_pca']
        for method in fusion_methods:
            order = determine_optimal_fusion_order(method)
            logger.info(f"  {method}: {order}")
            assert order in ['scale_fuse_select', 'select_scale_fuse']
        
        logger.info(" Phase 2 tests PASSED")
        return True
        
    except ImportError as e:
        logger.error(f" Phase 2 module import failed: {e}")
        return False
    except Exception as e:
        logger.error(f" Phase 2 test failed: {e}")
        return False

def test_phase_3_missing_data():
    """Test Phase 3: Centralized Missing Data Management."""
    logger.info("\n" + "="*60)
    logger.info("TESTING PHASE 3: Centralized Missing Data Management")
    logger.info("="*60)
    
    try:
        from missing_data_handler import create_missing_data_handler
        
        # Test missing data analysis
        logger.info("Test 3.1: Missing data pattern analysis")
        
        # Low missing data
        low_missing_data, y_low = generate_test_data("LowMissing", n_samples=40, missing_rate=0.02)
        handler_low = create_missing_data_handler(strategy="auto")
        analysis_low = handler_low.analyze_missing_patterns(low_missing_data)
        
        assert analysis_low['overall_missing_percentage'] < 0.05
        logger.info(f"✓ Low missing data: {analysis_low['overall_missing_percentage']:.1%}")
        
        logger.info(" Phase 3 tests PASSED")
        return True
        
    except ImportError as e:
        logger.error(f" Phase 3 module import failed: {e}")
        return False
    except Exception as e:
        logger.error(f" Phase 3 test failed: {e}")
        return False

def test_phase_4_validation():
    """Test Phase 4: Coordinated Validation Framework."""
    logger.info("\n" + "="*60)
    logger.info("TESTING PHASE 4: Coordinated Validation Framework")
    logger.info("="*60)
    
    try:
        from validation_coordinator import create_validation_coordinator, ValidationSeverity
        
        # Test basic validation framework
        logger.info("Test 4.1: Basic validation framework")
        
        validator = create_validation_coordinator(fail_fast=False)
        assert validator.fail_fast == False
        
        # Test issue addition
        validator.add_issue(ValidationSeverity.INFO, "Test info message")
        validator.add_issue(ValidationSeverity.WARNING, "Test warning message")
        
        summary = validator.get_validation_summary()
        assert summary['total_issues'] == 2
        logger.info(f"✓ Validation issues tracked: {summary['total_issues']}")
        
        logger.info(" Phase 4 tests PASSED")
        return True
        
    except ImportError as e:
        logger.error(f" Phase 4 module import failed: {e}")
        return False
    except Exception as e:
        logger.error(f" Phase 4 test failed: {e}")
        return False

def test_integration():
    """Test integration of all phases."""
    logger.info("\n" + "="*60)
    logger.info("TESTING INTEGRATION: All Phases Together")
    logger.info("="*60)
    
    try:
        from enhanced_pipeline_integration import run_enhanced_preprocessing_pipeline
        
        # Test basic integration
        logger.info("Test 5.1: Basic pipeline integration")
        
        test_data, y_test = generate_test_data("IntegrationTest", n_samples=40, missing_rate=0.1)
        
        final_data, y_aligned, metadata = run_enhanced_preprocessing_pipeline(
            test_data, y_test, 
            fusion_method="average", 
            task_type="classification", 
            dataset_name="IntegrationTest"
        )
        
        assert isinstance(final_data, dict)
        assert len(y_aligned) > 0
        assert 'fusion_method' in metadata
        logger.info(f"✓ Pipeline completed successfully")
        logger.info(f"  Final data keys: {list(final_data.keys())}")
        logger.info(f"  Samples processed: {len(y_aligned)}")
        
        logger.info(" Integration tests PASSED")
        return True
        
    except ImportError as e:
        logger.error(f" Integration module import failed: {e}")
        return False
    except Exception as e:
        logger.error(f" Integration test failed: {e}")
        return False

def main():
    """Run all phase tests."""
    logger.info(" Starting Enhanced Pipeline Phase Testing")
    logger.info("Testing all 4 architectural improvements:")
    logger.info("1. Early Data Quality Pipeline")
    logger.info("2. Fusion-Aware Feature Selection")  
    logger.info("3. Centralized Missing Data Management")
    logger.info("4. Coordinated Validation Framework")
    logger.info("5. Integration Testing")
    
    results = []
    
    # Run individual phase tests
    results.append(("Phase 1 - Data Quality", test_phase_1_data_quality()))
    results.append(("Phase 2 - Fusion-Aware", test_phase_2_fusion_aware()))
    results.append(("Phase 3 - Missing Data", test_phase_3_missing_data()))
    results.append(("Phase 4 - Validation", test_phase_4_validation()))
    results.append(("Integration", test_integration()))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = " PASSED" if result else " FAILED"
        logger.info(f"{test_name:.<40} {status}")
        if result:
            passed += 1
    
    logger.info("-" * 60)
    logger.info(f"Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! Enhanced Pipeline is ready for use.")
        return True
    else:
        logger.info(f"  Some tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)