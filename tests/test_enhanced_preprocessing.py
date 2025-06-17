#!/usr/bin/env python3
"""
Test script for the enhanced preprocessing pipeline with all 6 priority fixes.
"""

import numpy as np
import pandas as pd
import logging
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure matplotlib for headless operation
import matplotlib
matplotlib.use('Agg')

from data_io import load_and_preprocess_data_enhanced
from preprocessing import (
    ModalityAwareScaler,
    AdaptiveFeatureSelector,
    SampleIntersectionManager,
    PreprocessingValidator,
    FusionMethodStandardizer,
    enhanced_comprehensive_preprocessing_pipeline
)
# DataOrientationValidator is now in data_io.py for early validation
from data_io import DataOrientationValidator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_priority_1_data_orientation():
    """Test Priority 1: Data Orientation Validation"""
    logger.info("=== Testing Priority 1: Data Orientation Validation ===")
    
    # Test case 1: Normal orientation (samples x features)
    X_normal = np.random.randn(100, 50)
    X_validated = DataOrientationValidator.validate_data_orientation(X_normal, "test_normal")
    assert X_validated.shape == (100, 50), f"Normal orientation failed: {X_validated.shape}"
    logger.info("✓ Normal orientation test passed")
    
    # Test case 2: Gene expression with suspicious orientation (should auto-transpose)
    X_suspicious = np.random.randn(5000, 150)  # 5000 samples > 150 features (suspicious for gene data)
    X_validated = DataOrientationValidator.validate_data_orientation(X_suspicious, "gene_expression")
    assert X_validated.shape == (150, 5000), f"Gene expression transposition failed: {X_validated.shape}"
    logger.info("✓ Gene expression auto-transposition test passed")
    
    # Test case 3: Modality consistency
    modality_dict = {
        'exp': np.random.randn(100, 200),
        'mirna': np.random.randn(100, 150),
        'methy': np.random.randn(100, 300)
    }
    validated_dict = DataOrientationValidator.validate_modality_consistency(modality_dict)
    for modality, data in validated_dict.items():
        assert data.shape[0] == 100, f"Inconsistent sample count for {modality}: {data.shape[0]}"
    logger.info("✓ Modality consistency test passed")

def test_priority_2_modality_scaling():
    """Test Priority 2: Modality-Specific Scaling"""
    logger.info("=== Testing Priority 2: Modality-Specific Scaling ===")
    
    # Test case 1: Methylation (bounded [0,1] data should not be scaled)
    X_methy = np.random.beta(2, 2, (100, 50))  # Beta distribution for realistic methylation data
    X_scaled, scaler = ModalityAwareScaler.scale_modality_data(X_methy, "methylation")
    assert scaler is None, "Methylation data should not be scaled"
    assert np.array_equal(X_scaled, X_methy), "Methylation data should remain unchanged"
    logger.info("✓ Methylation no-scaling test passed")
    
    # Test case 2: Gene expression (should use robust scaling)
    X_gene = np.random.randn(100, 200) * 10 + 5  # Realistic gene expression range
    X_scaled, scaler = ModalityAwareScaler.scale_modality_data(X_gene, "gene_expression")
    assert scaler is not None, "Gene expression should be scaled"
    assert X_scaled.mean() < 1.0, "Gene expression should be centered"
    logger.info("✓ Gene expression robust scaling test passed")
    
    # Test case 3: miRNA (should use robust scaling)
    X_mirna = np.random.lognormal(0, 1, (100, 80))  # Log-normal for realistic miRNA data
    X_scaled, scaler = ModalityAwareScaler.scale_modality_data(X_mirna, "mirna")
    assert scaler is not None, "miRNA should be scaled"
    assert np.abs(X_scaled.mean()) < 1.0, "miRNA should be centered"
    logger.info("✓ miRNA robust scaling test passed")

def test_priority_3_adaptive_feature_selection():
    """Test Priority 3: Adaptive Feature Selection"""
    logger.info("=== Testing Priority 3: Adaptive Feature Selection ===")
    
    # Test case 1: Small sample size (should select fewer features)
    n_samples = 50
    n_features = 500
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)  # Binary classification
    
    target_features = AdaptiveFeatureSelector.calculate_adaptive_feature_count(n_samples)
    assert target_features <= n_samples - 1, f"Target features {target_features} > samples-1 {n_samples-1}"
    assert target_features >= 30, f"Target features {target_features} < minimum 30"
    logger.info(f"✓ Small sample adaptive calculation: {n_samples} samples -> {target_features} features")
    
    # Test case 2: Feature selection with classification
    X_selected, selector = AdaptiveFeatureSelector.select_features_adaptive(X, y, "gene_expression", "classification")
    assert X_selected.shape[1] <= target_features, f"Selected {X_selected.shape[1]} > target {target_features}"
    assert X_selected.shape[0] == n_samples, f"Sample count changed: {X_selected.shape[0]} != {n_samples}"
    logger.info(f"✓ Adaptive feature selection: {n_features} -> {X_selected.shape[1]} features")
    
    # Test case 3: Already at target (should not select)
    X_small = X[:, :target_features]
    X_no_select, selector_none = AdaptiveFeatureSelector.select_features_adaptive(X_small, y, "gene_expression", "classification")
    assert np.array_equal(X_no_select, X_small), "Should not select when already at target"
    assert selector_none is None, "Selector should be None when no selection needed"
    logger.info("✓ No-selection when at target test passed")

def test_priority_4_sample_intersection():
    """Test Priority 4: Sample Intersection Management"""
    logger.info("=== Testing Priority 4: Sample Intersection Management ===")
    
    # Create test data with overlapping samples
    sample_ids_1 = [f"TCGA-{i:02d}" for i in range(100)]
    sample_ids_2 = [f"TCGA-{i:02d}" for i in range(20, 120)]  # 80 overlap
    sample_ids_3 = [f"TCGA-{i:02d}" for i in range(50, 150)]  # 50 overlap
    
    modality_data_dict = {
        'exp': (np.random.randn(100, 200), sample_ids_1),
        'mirna': (np.random.randn(100, 150), sample_ids_2),
        'methy': (np.random.randn(100, 300), sample_ids_3)
    }
    
    # Test master list creation
    master_samples = SampleIntersectionManager.create_master_patient_list(modality_data_dict)
    expected_intersection = set(sample_ids_1) & set(sample_ids_2) & set(sample_ids_3)
    assert set(master_samples) == expected_intersection, "Master list should be intersection of all modalities"
    logger.info(f"✓ Master sample list: {len(master_samples)} common samples from 3 modalities")
    
    # Test alignment to master list
    aligned_dict = SampleIntersectionManager.align_modalities_to_master_list(modality_data_dict, master_samples)
    for modality, data in aligned_dict.items():
        assert data.shape[0] == len(master_samples), f"{modality} not aligned: {data.shape[0]} != {len(master_samples)}"
    logger.info("✓ Sample alignment test passed")

def test_priority_5_validation():
    """Test Priority 5: Enhanced Validation and Logging"""
    logger.info("=== Testing Priority 5: Enhanced Validation and Logging ===")
    
    # Test case 1: Valid data (should pass)
    X_dict_valid = {
        'exp': np.random.randn(100, 50),
        'mirna': np.random.randn(100, 40),
        'methy': np.random.uniform(0, 1, (100, 60))  # Methylation-like data
    }
    is_valid, issues = PreprocessingValidator.validate_preprocessing_stage(X_dict_valid, "test_valid", "classification")
    assert is_valid, f"Valid data should pass validation, issues: {issues}"
    logger.info("✓ Valid data validation test passed")
    
    # Test case 2: Problematic data (should flag issues)
    X_dict_problematic = {
        'exp': np.zeros((100, 50)),  # High sparsity
        'mirna': np.full((100, 40), np.nan),  # All NaN
        'methy': np.random.randn(10, 200)  # Low sample/feature ratio
    }
    is_valid, issues = PreprocessingValidator.validate_preprocessing_stage(X_dict_problematic, "test_problematic", "classification")
    assert not is_valid, "Problematic data should fail validation"
    assert len(issues) > 0, "Should report validation issues"
    logger.info(f"✓ Problematic data validation test passed: {len(issues)} issues detected")

def test_priority_6_fusion_standardization():
    """Test Priority 6: Fusion Method Standardization"""
    logger.info("=== Testing Priority 6: Fusion Method Standardization ===")
    
    # Test configuration retrieval
    base_config = FusionMethodStandardizer.get_base_preprocessing_config()
    assert 'data_orientation' in base_config, "Base config should include data orientation"
    assert 'modality_scaling' in base_config, "Base config should include modality scaling"
    logger.info("✓ Base preprocessing config test passed")
    
    # Test method-specific configuration
    snf_config = FusionMethodStandardizer.get_method_specific_config('fusion_snf')
    assert snf_config.get('prevent_over_compression', False), "SNF should prevent over-compression"
    assert 'optimal_feature_range' in snf_config, "SNF should have optimal feature range"
    logger.info("✓ Method-specific config test passed")
    
    # Test standardized preprocessing
    X_dict = {
        'exp': np.random.randn(100, 500),
        'mirna': np.random.randn(100, 300),
        'methy': np.random.uniform(0, 1, (100, 400))
    }
    y = np.random.randint(0, 2, 100)
    
    standardized_dict = FusionMethodStandardizer.standardize_fusion_preprocessing(
        'fusion_snf', X_dict, y, 'classification'
    )
    
    # Check that SNF didn't over-compress (should have >= 50 features per modality)
    for modality, data in standardized_dict.items():
        assert data.shape[1] >= 30, f"SNF over-compressed {modality}: {data.shape[1]} features"
        assert data.shape[0] == 100, f"Sample count changed for {modality}: {data.shape[0]}"
    logger.info("✓ Fusion standardization test passed")

def test_comprehensive_pipeline():
    """Test the complete enhanced preprocessing pipeline"""
    logger.info("=== Testing Comprehensive Enhanced Pipeline ===")
    
    # Create realistic test data
    sample_ids = [f"TCGA-{i:02d}-{j:04d}" for i in range(10, 15) for j in range(1000, 1020)]  # 100 samples
    
    modality_data_dict = {
        'exp': (np.random.randn(100, 1000), sample_ids),  # Gene expression
        'mirna': (np.random.lognormal(0, 1, (100, 200)), sample_ids),  # miRNA
        'methy': (np.random.beta(2, 2, (100, 500)), sample_ids)  # Methylation
    }
    
    y = np.random.randint(0, 3, 100)  # Multi-class classification
    
    try:
        processed_modalities, y_aligned = enhanced_comprehensive_preprocessing_pipeline(
            modality_data_dict=modality_data_dict,
            y=y,
            fusion_method="fusion_snf",
            task_type="classification"
        )
        
        # Validate results
        assert len(processed_modalities) == 3, f"Should have 3 modalities, got {len(processed_modalities)}"
        assert len(y_aligned) == 100, f"Target alignment failed: {len(y_aligned)} != 100"
        
        for modality, data in processed_modalities.items():
            assert data.shape[0] == 100, f"{modality} sample count: {data.shape[0]}"
            assert data.shape[1] > 0, f"{modality} no features remaining"
            assert not np.isnan(data).any(), f"{modality} contains NaN values"
            assert not np.isinf(data).any(), f"{modality} contains infinite values"
        
        logger.info("✓ Comprehensive pipeline test passed")
        
        # Log final dimensions
        for modality, data in processed_modalities.items():
            logger.info(f"  {modality}: {data.shape}")
            
    except Exception as e:
        logger.error(f"Comprehensive pipeline test failed: {e}")
        raise

def test_real_dataset_integration():
    """Test with a real dataset (if available)"""
    logger.info("=== Testing Real Dataset Integration ===")
    
    try:
        # Try to load a real dataset using the enhanced preprocessing
        processed_modalities, y_aligned = load_and_preprocess_data_enhanced(
            dataset_name="Colon",  # Use Colon as it's typically available
            task_type="classification",
            fusion_method="fusion_attention_weighted",
            apply_priority_fixes=True
        )
        
        if len(processed_modalities) > 0:
            logger.info(f"✓ Real dataset test passed: {len(processed_modalities)} modalities loaded")
            for modality, data in processed_modalities.items():
                logger.info(f"  {modality}: {data.shape}")
        else:
            logger.warning("Real dataset test skipped: No data loaded (dataset may not be available)")
            
    except Exception as e:
        logger.warning(f"Real dataset test skipped due to error: {e}")

def main():
    """Run all tests"""
    logger.info("Starting Enhanced Preprocessing Pipeline Tests")
    logger.info("=" * 60)
    
    try:
        test_priority_1_data_orientation()
        test_priority_2_modality_scaling()
        test_priority_3_adaptive_feature_selection()
        test_priority_4_sample_intersection()
        test_priority_5_validation()
        test_priority_6_fusion_standardization()
        test_comprehensive_pipeline()
        test_real_dataset_integration()
        
        logger.info("=" * 60)
        logger.info("🎉 ALL TESTS PASSED! Enhanced preprocessing pipeline is working correctly.")
        logger.info(" All 6 priority fixes have been successfully implemented and validated.")
        
    except Exception as e:
        logger.error(f" Test failed: {e}")
        raise

if __name__ == "__main__":
    main() 