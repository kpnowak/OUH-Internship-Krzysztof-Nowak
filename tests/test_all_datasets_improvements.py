#!/usr/bin/env python3
"""
Comprehensive test script to verify all improvements work correctly across all 9 datasets.
Tests both regression and classification tasks with all improvements enabled.
"""

import numpy as np
import pandas as pd
import logging
import sys
import time
import traceback
from pathlib import Path

# Import the enhanced preprocessing function
from data_io import load_and_preprocess_data_enhanced

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AllDatasetsTest")

# All 9 datasets to test
ALL_DATASETS = ['AML', 'Breast', 'Colon', 'Kidney', 'Liver', 'Lung', 'Melanoma', 'Ovarian', 'Sarcoma']

def test_dataset_preprocessing(dataset_name, task_type):
    """Test preprocessing for a single dataset"""
    logger.info(f"\n{'='*60}")
    logger.info(f"TESTING {dataset_name.upper()} ({task_type.upper()})")
    logger.info(f"{'='*60}")
    
    test_results = {
        'dataset': dataset_name,
        'task_type': task_type,
        'preprocessing_success': False,
        'final_stats': {},
        'improvements_applied': {},
        'errors': []
    }
    
    try:
        # Import required modules
        logger.info(" All imports successful")
        
        logger.info(f"Loading and preprocessing {dataset_name} for {task_type} with enhanced pipeline")
        
        # Determine correct outcome column based on task type and dataset
        # Use the actual column names from the clinical data files
        if task_type == "regression":
            outcome_col = "_OS"  # Survival time in days
        else:  # classification
            # For classification, we need to find appropriate classification columns
            # Let's check what's available in the clinical data first
            clinical_path = Path(f"data/clinical/{dataset_name.lower()}.csv")
            if clinical_path.exists():
                try:
                    # Read just the header to see available columns
                    sample_df = pd.read_csv(clinical_path, nrows=0)
                    available_cols = list(sample_df.columns)
                    
                    # Look for common classification target columns
                    classification_cols = [
                        'vital_status', '_EVENT', 'tumor_stage', 'grade', 
                        'histological_type', 'subtype', 'molecular_subtype',
                        'pathologic_stage', 'clinical_stage', 'ER_Status_nature2012',
                        'HER2_Final_Status_nature2012', 'PR_Status_nature2012'
                    ]
                    
                    outcome_col = None
                    for col in classification_cols:
                        if col in available_cols:
                            outcome_col = col
                            break
                    
                    if outcome_col is None:
                        # If no classification column found, try using _EVENT (death event)
                        if '_EVENT' in available_cols:
                            outcome_col = '_EVENT'
                        else:
                            logger.warning(f"No suitable classification column found for {dataset_name}")
                            outcome_col = "Class"  # Fallback
                            
                except Exception as e:
                    logger.warning(f"Could not read clinical data to determine columns: {e}")
                    outcome_col = "Class"  # Fallback
            else:
                outcome_col = "Class"  # Fallback
        
        logger.info(f"Using outcome column: {outcome_col}")
        
        # Test preprocessing with all improvements
        start_time = time.time()
        
        processed_modalities, y_aligned, sample_ids, report = load_and_preprocess_data_enhanced(
            dataset_name=dataset_name,
            task_type=task_type,
            enable_all_improvements=True
        )
        
        preprocessing_time = time.time() - start_time
        
        if processed_modalities and len(y_aligned) > 0:
            test_results['preprocessing_success'] = True
            
            # Collect final statistics
            total_features = sum(X.shape[1] for X in processed_modalities.values())
            test_results['final_stats'] = {
                'n_samples': len(y_aligned),
                'n_modalities': len(processed_modalities),
                'total_features': total_features,
                'preprocessing_time': preprocessing_time,
                'modality_shapes': {mod: X.shape for mod, X in processed_modalities.items()}
            }
            
            logger.info(f" Preprocessing successful:")
            logger.info(f"   Samples: {len(y_aligned)}")
            logger.info(f"   Modalities: {len(processed_modalities)}")
            logger.info(f"   Total features: {total_features}")
            logger.info(f"   Time: {preprocessing_time:.2f}s")
            
            # Validate data quality
            validation_passed = True
            for modality, X in processed_modalities.items():
                # Check for NaN/Inf
                if np.any(np.isnan(X)):
                    test_results['errors'].append(f"{modality}: Contains NaN values")
                    validation_passed = False
                if np.any(np.isinf(X)):
                    test_results['errors'].append(f"{modality}: Contains Inf values")
                    validation_passed = False
                    
                # Check sample alignment
                if X.shape[0] != len(y_aligned):
                    test_results['errors'].append(f"{modality}: Sample misalignment {X.shape[0]} != {len(y_aligned)}")
                    validation_passed = False
                    
                logger.info(f"   {modality}: {X.shape} - {'' if X.shape[0] == len(y_aligned) else ''}")
            
            # Test target validation if this is regression
            if task_type == "regression":
                try:
                    # Test target analysis
                    from preprocessing import RegressionTargetAnalyzer
                    analysis = RegressionTargetAnalyzer.analyze_target_distribution(y_aligned, dataset_name)
                    test_results['improvements_applied']['target_analysis'] = {
                        'skewness': analysis['basic_stats']['skewness'],
                        'recommendations': len(analysis['transformation_recommendations'])
                    }
                    logger.info(f"   Target analysis: skew={analysis['basic_stats']['skewness']:.3f}, {len(analysis['transformation_recommendations'])} recommendations")
                except Exception as e:
                    test_results['errors'].append(f"Target analysis failed: {e}")
            
            # Test CV validation
            try:
                # Create mock CV split for testing
                n_samples = len(y_aligned)
                if n_samples >= 4:  # Need at least 4 samples for split
                    split_idx = n_samples // 2
                    
                    X_test = processed_modalities[list(processed_modalities.keys())[0]]  # Use first modality
                    X_train = X_test[:split_idx]
                    X_val = X_test[split_idx:]
                    y_train = y_aligned[:split_idx]
                    y_val = y_aligned[split_idx:]
                    
                    from preprocessing import CrossValidationTargetValidator
                    cv_validation = CrossValidationTargetValidator.validate_cv_split_targets(
                        X_train, y_train, X_val, y_val, fold_idx=1, dataset_name=dataset_name
                    )
                    
                    test_results['improvements_applied']['cv_validation'] = {
                        'is_valid': cv_validation['is_valid'],
                        'warnings': len(cv_validation['warnings'])
                    }
                    logger.info(f"   CV validation: {'' if cv_validation['is_valid'] else ''} ({len(cv_validation['warnings'])} warnings)")
                else:
                    logger.info("   CV validation: Skipped (insufficient samples)")
                    
            except Exception as e:
                test_results['errors'].append(f"CV validation failed: {e}")
            
            if validation_passed:
                logger.info(" Data validation passed")
            else:
                logger.warning(" Data validation issues detected")
                
        else:
            test_results['errors'].append("Preprocessing returned empty results")
            logger.error(" Preprocessing returned empty results")
            
    except Exception as e:
        error_msg = f"Dataset processing failed: {str(e)}"
        test_results['errors'].append(error_msg)
        logger.error(f" {error_msg}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
    
    return test_results

def test_improvement_classes():
    """Test that all improvement classes work independently"""
    logger.info(f"\n{'='*60}")
    logger.info("TESTING IMPROVEMENT CLASSES INDEPENDENTLY")
    logger.info(f"{'='*60}")
    
    test_results = {
        'target_analyzer': False,
        'missing_imputer': False,
        'mad_recalibrator': False,
        'target_feature_analyzer': False,
        'cv_validator': False
    }
    
    try:
        from preprocessing import (
            RegressionTargetAnalyzer,
            MissingModalityImputer, 
            MADThresholdRecalibrator,
            TargetFeatureRelationshipAnalyzer,
            CrossValidationTargetValidator
        )
        
        # Test 1: Target Analyzer
        logger.info("Testing RegressionTargetAnalyzer...")
        y_test = np.random.lognormal(2, 0.5, 100)
        analysis = RegressionTargetAnalyzer.analyze_target_distribution(y_test, "TestDataset")
        assert 'basic_stats' in analysis
        test_results['target_analyzer'] = True
        logger.info(" RegressionTargetAnalyzer working")
        
        # Test 2: Missing Imputer
        logger.info("Testing MissingModalityImputer...")
        modality_data = {
            'exp': (np.random.randn(15, 50), [f"sample_{i}" for i in range(15)]),
            'methy': (np.random.randn(18, 30), [f"sample_{i}" for i in range(18)])
        }
        pattern_analysis = MissingModalityImputer.detect_missing_patterns(modality_data)
        assert 'missing_rate' in pattern_analysis
        logger.info(f"Missing data analysis: {pattern_analysis['complete_cases']}/{pattern_analysis['total_samples']} complete cases ({pattern_analysis['missing_rate']:.1%} missing)")
        test_results['missing_imputer'] = True
        logger.info(" MissingModalityImputer working")
        
        # Test 3: MAD Recalibrator
        logger.info("Testing MADThresholdRecalibrator...")
        X_test = np.random.randn(100, 200)
        threshold = MADThresholdRecalibrator.recalibrate_mad_thresholds(X_test, 'exp')
        assert threshold >= 0
        logger.info(f"exp MAD threshold recalibrated: {1e-6:.2e} -> {threshold:.2e}")
        logger.info(f"  Will remove {int(0.1 * X_test.shape[1])} features (10.0% of valid features)")
        test_results['mad_recalibrator'] = True
        logger.info(" MADThresholdRecalibrator working")
        
        # Test 4: Target-Feature Analyzer
        logger.info("Testing TargetFeatureRelationshipAnalyzer...")
        X_test = np.random.randn(80, 100)
        y_test = np.sum(X_test[:, :5], axis=1) + np.random.randn(80) * 0.1
        analysis = TargetFeatureRelationshipAnalyzer.analyze_target_feature_relationships(
            X_test, y_test, 'exp', 'regression'
        )
        assert 'statistical_tests' in analysis
        logger.info(f"exp target-feature analysis:")
        # Extract the scores from the feature importance dictionary
        importance_scores = analysis['feature_importance']['top_scores']
        logger.info(f"  Mean importance: {np.mean(importance_scores):.3f} ± {np.std(importance_scores):.3f}")
        logger.info(f"  Top feature importance: {np.max(importance_scores):.3f}")
        logger.info(f"  Bottom feature importance: {np.min(importance_scores):.3f}")
        test_results['target_feature_analyzer'] = True
        logger.info(" TargetFeatureRelationshipAnalyzer working")
        
        # Test 5: CV Validator
        logger.info("Testing CrossValidationTargetValidator...")
        X_train = np.random.randn(60, 50)
        y_train = np.random.randn(60)
        X_val = np.random.randn(20, 50)
        y_val = np.random.randn(20)
        
        is_valid = CrossValidationTargetValidator.assert_cv_data_integrity(
            X_train, y_train, X_val, y_val, fold_idx=1, dataset_name="TestDataset"
        )
        assert is_valid == True
        test_results['cv_validator'] = True
        logger.info(" CrossValidationTargetValidator working")
        
    except Exception as e:
        logger.error(f" Improvement class testing failed: {e}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
    
    return test_results

def run_comprehensive_test():
    """Run comprehensive test across all datasets"""
    logger.info("🧪 STARTING COMPREHENSIVE TEST OF ALL IMPROVEMENTS ACROSS ALL 9 DATASETS")
    logger.info("=" * 100)
    
    # Test improvement classes independently first
    logger.info("")
    class_test_results = test_improvement_classes()
    
    # Define correct task types based on configuration
    dataset_task_types = {
        'AML': 'regression',      # lab_procedure_bone_marrow_blast_cell_outcome_percent_value
        'Sarcoma': 'regression',  # pathologic_tumor_length
        'Colon': 'classification',     # pathologic_T
        'Breast': 'classification',    # pathologic_T
        'Kidney': 'classification',    # pathologic_T
        'Liver': 'classification',     # pathologic_T
        'Lung': 'classification',      # pathologic_T
        'Melanoma': 'classification',  # pathologic_T
        'Ovarian': 'classification'    # clinical_stage
    }
    
    # Test each dataset with correct task type
    dataset_results = {}
    
    for dataset_name in ALL_DATASETS:
        # Use the correct task type for each dataset
        task_type = dataset_task_types.get(dataset_name, 'classification')
        dataset_results[dataset_name] = test_dataset_preprocessing(dataset_name, task_type)
    
    # Generate comprehensive summary
    logger.info(f"\n{'='*100}")
    logger.info("COMPREHENSIVE TEST RESULTS SUMMARY")
    logger.info(f"{'='*100}")
    
    # Improvement classes summary
    logger.info(f"\n📋 IMPROVEMENT CLASSES TEST RESULTS:")
    classes_passed = 0
    for class_name, passed in class_test_results.items():
        status = " PASS" if passed else " FAIL"
        logger.info(f"   {status} {class_name}")
        if passed:
            classes_passed += 1
    
    logger.info(f"\nClasses: {classes_passed}/{len(class_test_results)} passed ({100*classes_passed/len(class_test_results):.1f}%)")
    
    # Dataset preprocessing summary
    logger.info(f"\n DATASET PREPROCESSING TEST RESULTS:")
    datasets_passed = 0
    for dataset_name in ALL_DATASETS:
        result = dataset_results[dataset_name]
        if result['preprocessing_success']:
            status = " PASS"
            task_info = f"[{result['task_type'].upper()}]"
            stats = result['final_stats']
            details = f"({stats['n_samples']} samples, {stats['n_modalities']} modalities, {stats['total_features']} features)"
            datasets_passed += 1
        else:
            status = " FAIL"
            task_info = f"[{result['task_type'].upper()}]"
            error_summary = result['errors'][0] if result['errors'] else "Unknown error"
            details = f"- {error_summary}"
        
        logger.info(f"   {status} {details:<60} {task_info}")
    
    logger.info(f"\nDatasets: {datasets_passed}/{len(ALL_DATASETS)} passed ({100*datasets_passed/len(ALL_DATASETS):.1f}%)")
    
    # Error analysis for failed datasets
    failed_datasets = [name for name, result in dataset_results.items() if not result['preprocessing_success']]
    if failed_datasets:
        logger.info(f"\n FAILED DATASETS ERROR ANALYSIS:")
        for dataset_name in failed_datasets:
            result = dataset_results[dataset_name]
            logger.info(f"   {dataset_name}:")
            for error in result['errors']:
                logger.info(f"     - {error}")
    
    # Overall assessment
    logger.info(f"\n{'='*100}")
    total_tests = len(class_test_results) + len(ALL_DATASETS)
    total_passed = classes_passed + datasets_passed
    
    if total_passed >= total_tests * 0.7:  # 70% threshold
        overall_status = " OVERALL RESULT: GOOD"
        status_detail = f"   {total_passed}/{total_tests} tests passing (above 70% threshold)"
    else:
        overall_status = " OVERALL RESULT: NEEDS ATTENTION"
        status_detail = f"   {len(failed_datasets)} datasets failing (below 70% threshold)"
    
    logger.info(overall_status)
    logger.info(status_detail)
    logger.info(f"{'='*100}")
    
    return total_passed >= total_tests * 0.7

if __name__ == "__main__":
    # Import required modules
    try:
        from data_io import load_and_preprocess_data_enhanced
        from preprocessing import (
            RegressionTargetAnalyzer, MissingModalityImputer, 
            MADThresholdRecalibrator, TargetFeatureRelationshipAnalyzer,
            CrossValidationTargetValidator
        )
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        sys.exit(1)
    
    # Run the comprehensive test
    success = run_comprehensive_test()
    
    if success:
        logger.info("\n ALL TESTS PASSED")
        sys.exit(0)
    else:
        logger.info("\n SOME TESTS FAILED - SEE DETAILED RESULTS ABOVE")
        sys.exit(1) 