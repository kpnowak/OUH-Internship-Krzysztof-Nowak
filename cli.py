#!/usr/bin/env python3
"""
Command-line interface module for running the pipeline.
"""

import os
import sys
import time
import argparse
from typing import Dict, List, Optional, Any
import logging
import numpy as np
import pandas as pd

# Local imports
from config import (
    REGRESSION_DATASETS, CLASSIFICATION_DATASETS,
    MAX_COMPONENTS, MAX_FEATURES, N_VALUES_LIST, CV_CONFIG,
    MISSING_MODALITIES_CONFIG  # ADD missing modalities config import
)
from data_io import load_dataset, load_and_preprocess_data_enhanced
from models import (
    get_regression_extractors, get_regression_selectors,
    get_classification_extractors, get_classification_selectors,
    get_regression_models, get_classification_models
)
from cv import (
    run_extraction_pipeline, run_selection_pipeline
)
from utils import comprehensive_logger
from cv import merge_small_classes
from mad_analysis import run_mad_analysis
from logging_utils import (
    setup_logging_levels, log_pipeline_stage, log_mad_analysis_info,
    log_dataset_preparation, log_model_training_info, log_data_save_info, log_plot_save_info,
    log_timing_summary
)
# Import the enhanced 4-phase pipeline integration
from enhanced_pipeline_integration import run_enhanced_preprocessing_pipeline

# Remove any existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Set up logging to file and console with improved configuration
log_file = "debug.log"

# Create file handler that always logs everything to debug.log
file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)  # Always log everything to file
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s')
file_handler.setFormatter(file_formatter)

# Create console handler with default WARNING level (will be adjusted based on args)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)  # Default level, will be changed based on args
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Set up root logger
logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, console_handler])

# Suppress noisy third-party loggers
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.ticker').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('sklearn').setLevel(logging.WARNING)

# Additional matplotlib configuration for parallel processing
import matplotlib
matplotlib.use('Agg')  # Ensure non-interactive backend
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode
# Configure matplotlib to avoid tkinter issues in parallel processing
matplotlib.rcParams['figure.max_open_warning'] = 0  # Disable figure limit warnings
matplotlib.rcParams['agg.path.chunksize'] = 10000  # Optimize for large plots

logger = logging.getLogger(__name__)

def process_dataset(ds_conf: Dict[str, Any], is_regression: bool = True) -> Optional[Dict[str, Any]]:
    """
    Load and process a dataset based on its configuration.
    
    Parameters
    ----------
    ds_conf : Dict[str, Any]
        Dataset configuration dictionary
    is_regression : bool
        Whether this is a regression task
        
    Returns
    -------
    Optional[Dict[str, Any]]
        Dictionary containing processed dataset information or None if loading failed
    """
    comprehensive_logger.log_memory_usage(f"dataset_start_{ds_conf['name']}", force=True)
    
    try:
        # Extract the dataset name
        ds_name = ds_conf["name"]
        logger.info(f"\n>> Processing {ds_name} dataset...")

        
        # Load the dataset using the new optimized function
        ds_name = ds_conf["name"]
        modalities_list = list(ds_conf["modalities"].keys())
        # Convert modality names to short names for the new function
        modality_short_names = []
        for mod_name in modalities_list:
            if "Gene Expression" in mod_name or "exp" in mod_name.lower():
                modality_short_names.append("exp")
            elif "miRNA" in mod_name or "mirna" in mod_name.lower():
                modality_short_names.append("mirna")
            elif "Methylation" in mod_name or "methy" in mod_name.lower():
                modality_short_names.append("methy")
            else:
                # Default to the original name if no match
                modality_short_names.append(mod_name.lower())
        
        outcome_col = ds_conf["outcome_col"]
        task_type = 'regression' if is_regression else 'classification'
        

        
        # Load raw data first
        raw_modalities, y_raw, common_ids, is_regression_detected = load_dataset(
            ds_name.lower(), 
            modality_short_names, 
            outcome_col, 
            task_type,
            parallel=True,
            use_cache=True
        )
        
        if raw_modalities is None or len(common_ids) == 0:
            logger.warning(f"Error: Failed to load raw dataset {ds_name}")
            log_dataset_preparation(ds_name, {}, [], (), success=False)
            return None
        
        # Convert to enhanced pipeline format: Dict[str, Tuple[np.ndarray, List[str]]]
        modality_data_dict = {}
        for modality_name, modality_df in raw_modalities.items():
            # Convert DataFrame to numpy array (transpose to get samples x features)
            X = modality_df.T.values  # modality_df is features x samples
            modality_data_dict[modality_name] = (X, common_ids)
        
        # Apply the NEW 4-phase enhanced preprocessing pipeline
        try:
            # Determine optimal fusion method based on task type
            fusion_method = "snf" if task_type == "classification" else "weighted_concat"
            
            modalities_data, y_aligned, pipeline_metadata = run_enhanced_preprocessing_pipeline(
                modality_data_dict=modality_data_dict,
                y=y_raw.values,
                fusion_method=fusion_method,
                task_type=task_type,
                dataset_name=ds_name,
                enable_early_quality_check=True,
                enable_feature_first_order=True,
                enable_centralized_missing_data=True,
                enable_coordinated_validation=True
            )
            
            logger.info(f" 4-Phase Enhanced Pipeline completed successfully for {ds_name}")
            logger.info(f" Quality Score: {pipeline_metadata.get('quality_score', 'N/A')}")
            logger.info(f" Phases Enabled: {pipeline_metadata.get('phases_enabled', {})}")
            
        except Exception as e:
            logger.warning(f"4-Phase Enhanced Pipeline failed for {ds_name}: {str(e)}")
            logger.info(f"Falling back to standard preprocessing for {ds_name}")
            
            # Fallback to robust biomedical preprocessing
            try:
                from preprocessing import robust_biomedical_preprocessing_pipeline
                
                processed_modalities = {}
                for modality_name, (X, sample_ids) in modality_data_dict.items():
                    # Determine modality type
                    if 'exp' in modality_name.lower():
                        modality_type = 'gene_expression'
                    elif 'mirna' in modality_name.lower():
                        modality_type = 'mirna'
                    elif 'methy' in modality_name.lower():
                        modality_type = 'methylation'
                    else:
                        modality_type = 'unknown'
                    
                    # Apply robust preprocessing
                    X_processed, transformers, report = robust_biomedical_preprocessing_pipeline(
                        X, modality_type=modality_type
                    )
                    processed_modalities[modality_name] = X_processed
                    logger.info(f"Fallback preprocessing for {modality_name}: {X.shape} -> {X_processed.shape}")
                
                # Align targets
                n_samples = list(processed_modalities.values())[0].shape[0]
                y_aligned = y_aligned[:n_samples] if len(y_aligned) >= n_samples else y_aligned
                modalities_data = processed_modalities
                
                logger.info(f"Fallback robust preprocessing completed for {ds_name}")
            except Exception as e2:
                logger.error(f"Both enhanced and fallback preprocessing failed for {ds_name}: {str(e2)}")
                # Ultimate fallback to raw data
                modalities_data = raw_modalities
                y_aligned = y_raw.values
        
        # Check if loading was successful  
        if modalities_data is None:
            logger.warning(f"Error: Failed to load dataset {ds_name}")
            log_dataset_preparation(ds_name, {}, [], (), success=False)
            return None
        
        # Convert processed data back to DataFrame format for compatibility
        modalities = {}
        if isinstance(modalities_data, dict):
            for modality_name, processed_array in modalities_data.items():
                # Convert back to DataFrame format (features x samples)
                if isinstance(processed_array, np.ndarray):
                    # Create feature names and sample IDs for the processed data
                    n_samples, n_features = processed_array.shape
                    feature_names = [f"feature_{i}" for i in range(n_features)]
                    sample_ids = common_ids[:n_samples] if len(common_ids) >= n_samples else [f"sample_{i}" for i in range(n_samples)]
                    
                    # Create DataFrame (features x samples to match expected format)
                    modalities[modality_name] = pd.DataFrame(
                        processed_array.T,  # Transpose: samples x features -> features x samples
                        index=feature_names,
                        columns=sample_ids
                    )
                else:
                    # Already in correct format
                    modalities[modality_name] = processed_array
        else:
            logger.error(f"Unexpected modalities_data format: {type(modalities_data)}")
            return None
        
        # Convert pandas Series to numpy array for compatibility with existing code
        if hasattr(y_aligned, 'values'):
            y_aligned = y_aligned.values
        
        # For regression, ensure y_aligned is numeric
        if is_regression:
            if not np.issubdtype(y_aligned.dtype, np.number):
                logger.error(f"Regression target data is not numeric: dtype={y_aligned.dtype}")
                logger.error(f"Sample values: {y_aligned[:5] if len(y_aligned) > 0 else 'empty'}")
                logger.error(f"This indicates a data loading error - regression targets must be numeric")
                log_dataset_preparation(ds_name, modalities, common_ids, y_aligned.shape, success=False)
                return None
            
            # Ensure no NaN or infinite values
            if np.any(np.isnan(y_aligned)) or np.any(np.isinf(y_aligned)):
                logger.warning(f"Found NaN or infinite values in regression target, cleaning...")
                
                # Count the problematic values
                nan_count = np.sum(np.isnan(y_aligned))
                inf_count = np.sum(np.isinf(y_aligned))
                logger.warning(f"Found {nan_count} NaN values and {inf_count} infinite values in {ds_name} target")
                
                # For AML dataset, this is critical - log more details
                if ds_name.lower() == 'aml':
                    logger.error(f"CRITICAL: AML dataset has {nan_count} NaN values in target - this will cause model training failures!")
                    logger.error("This indicates the pipe-separated value extraction in data_io.py failed")
                    
                    # Show sample of problematic values
                    if nan_count > 0:
                        nan_indices = np.where(np.isnan(y_aligned))[0][:5]  # First 5 NaN indices
                        logger.error(f"Sample NaN indices: {nan_indices}")
                        logger.error(f"Corresponding sample IDs: {[common_ids[i] for i in nan_indices if i < len(common_ids)]}")
                
                # Clean the values
                original_length = len(y_aligned)
                y_aligned = np.nan_to_num(y_aligned, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Verify cleaning was successful
                remaining_nan = np.sum(np.isnan(y_aligned))
                remaining_inf = np.sum(np.isinf(y_aligned))
                
                if remaining_nan > 0 or remaining_inf > 0:
                    logger.error(f"CRITICAL: Cleaning failed! Still have {remaining_nan} NaN and {remaining_inf} infinite values")
                    logger.error("This will definitely cause 'Input contains NaN' errors in model training")
                    return None
                else:
                    logger.info(f"Successfully cleaned all NaN/infinite values in {ds_name} target")
                
    
        
        # Verify data integrity
        if len(common_ids) < 10:
            logger.warning(f"Warning: Too few samples ({len(common_ids)}) in {ds_name}, skipping")
            log_dataset_preparation(ds_name, modalities, common_ids, y_aligned.shape, success=False)
            return None
            
        if len(y_aligned) == 0:
            logger.warning(f"Error: No target values available for {ds_name}")
            log_dataset_preparation(ds_name, modalities, common_ids, (), success=False)
            return None
        
        # For classification datasets, validate class distribution
        # Use the detected task type to ensure consistency
        if not is_regression_detected:
            unique, counts = np.unique(y_aligned, return_counts=True)
            min_samples = np.min(counts)
            n_classes = len(unique)
            
            logger.info(f"Class distribution for {ds_name}: {dict(zip(unique, counts))}")
            logger.debug(f"[DATASET_PREP] {ds_name} - {n_classes} classes, min samples per class: {min_samples}")
            
            # Check for problematic class distributions
            if min_samples < 2:
                logger.info(f"Dataset {ds_name} has classes with < 2 samples")
                logger.info(f"Original class distribution: {dict(zip(unique, counts))}")
                
                if CV_CONFIG["merge_small_classes"]:
                    logger.info(f"Merging small classes to ensure proper cross-validation")
                    y_aligned, label_mapping = merge_small_classes(y_aligned, 2)
                    
                    # Log the new class distribution
                    unique, counts = np.unique(y_aligned, return_counts=True)
                    logger.info(f"After merging: {dict(zip(unique, counts))}")
                    logger.info(f"Label mapping: {label_mapping}")
                    
                    # Check if we still have enough classes
                    if len(unique) < 2:
                        logger.error(f"Dataset {ds_name} has insufficient classes after merging (< 2 classes)")
                        logger.error(f"Cannot proceed with this dataset")
                        log_dataset_preparation(ds_name, modalities, common_ids, y_aligned.shape, success=False)
                        return None
                else:
                    # If merging is disabled, filter out small classes
                    valid_classes = unique[counts >= 2]
                    if len(valid_classes) < 2:
                        logger.error(f"Dataset {ds_name} has insufficient valid classes for classification (< 2 classes with >= 2 samples)")
                        logger.error(f"Cannot proceed with this dataset")
                        log_dataset_preparation(ds_name, modalities, common_ids, y_aligned.shape, success=False)
                        return None
                    
                    # Filter samples to only include valid classes
                    valid_mask = np.isin(y_aligned, valid_classes)
                    filtered_common_ids = [common_ids[i] for i in range(len(common_ids)) if valid_mask[i]]
                    filtered_y_aligned = y_aligned[valid_mask]
                    
                    # Filter modalities to match the filtered samples
                    filtered_modalities = {}
                    for mod_name, mod_data in modalities.items():
                        filtered_modalities[mod_name] = mod_data[filtered_common_ids]
                    
                    # Update variables
                    common_ids = filtered_common_ids
                    y_aligned = filtered_y_aligned
                    modalities = filtered_modalities
                    
                    logger.info(f"After filtering: {len(common_ids)} samples, {len(np.unique(y_aligned))} classes")
        
        # Final validation after any filtering
        if not is_regression_detected:
            unique_final, counts_final = np.unique(y_aligned, return_counts=True)
            if len(unique_final) < 2:
                logger.error(f"Dataset {ds_name} has insufficient classes after processing (< 2)")
                log_dataset_preparation(ds_name, modalities, common_ids, y_aligned.shape, success=False)
                return None
            
            # Warn about very small classes that might still cause issues
            problematic_classes = unique_final[(counts_final >= 2) & (counts_final < 5)]
            if len(problematic_classes) > 0:
                logger.warning(f"Dataset {ds_name} has classes with few samples that may cause CV issues: {dict(zip(problematic_classes, counts_final[np.isin(unique_final, problematic_classes)]))}")
        
        for mod_name, mod_df in modalities.items():
            if mod_df.empty:
                logger.warning(f"Warning: Empty modality {mod_name} in {ds_name}")
        
        # Log successful dataset preparation
        log_dataset_preparation(ds_name, modalities, common_ids, y_aligned.shape, success=True)
        logger.debug(f"[DATASET_PREP] {ds_name} - Dataset preparation completed successfully")
        
        # Return the loaded data
        return {
            "name": ds_name,
            "modalities": modalities,
            "common_ids": common_ids,
            "y_aligned": y_aligned
        }
        
    except Exception as e:
        logger.error(f"Error processing dataset {ds_conf['name']}: {str(e)}")
        logger.debug(f"[DATASET_PREP] {ds_conf['name']} - Exception details: {str(e)}")
        import traceback
        logger.debug(f"[DATASET_PREP] {ds_conf['name']} - Traceback:\n{traceback.format_exc()}")
        log_dataset_preparation(ds_conf['name'], {}, [], (), success=False)
        return None

def process_regression_datasets(args):
    """Process all regression datasets."""
    regression_block_info = "=== REGRESSION BLOCK (AML, Sarcoma) ==="
    logger.info(regression_block_info)
    print(f"{regression_block_info}")
    reg_extractors = get_regression_extractors()
    reg_selectors = get_regression_selectors()
    reg_models = get_regression_models()  # CURRENT IMPLEMENTATION: Use getter function
    n_shared_list = N_VALUES_LIST.copy()  # Shared list for both extraction and selection
    
    # Check if a specific n_val is requested via command line
    if args.n_val:
        n_val = int(args.n_val)
        if n_val in n_shared_list:
            n_shared_list = [n_val]
            logger.info(f"Processing only n_val = {n_val} as requested")
        else:
            logger.warning(f"Requested n_val = {n_val} not in {n_shared_list}, using all values")
    
    # Calculate total runs (now just extractors + selectors) * len(n_shared_list)
    reg_total_runs = (
        len(REGRESSION_DATASETS) * 
        (len(reg_extractors) + len(reg_selectors)) *
        len(n_shared_list)
    )
    progress_count_reg = [0]
    
    for ds_conf in REGRESSION_DATASETS:
        result = process_dataset(ds_conf, is_regression=True)
        if result is None:
            continue
            
        ds_name = result["name"]
        modalities = result["modalities"]
        common_ids = result["common_ids"]
        y_aligned = result["y_aligned"]
        
        # Set up output directory
        base_out = os.path.join(
            ds_conf.get("output_dir", "output"), ds_name
        )
        os.makedirs(base_out, exist_ok=True)
        
        try:
            if args.sequential:
                # SEQUENTIAL: Process one extractor/selector at a time through all fusion techniques
                print(f"===> Processing with SEQUENTIAL architecture for dataset {ds_name}")
                logger.info(f"===> Using SEQUENTIAL architecture: One extractor/selector → All fusion techniques → All models")
                
                from cv import run_sequential_extraction_pipeline, run_sequential_selection_pipeline
                
                # First, run sequential extraction pipeline
                print(f"===> Processing SEQUENTIAL EXTRACTION for dataset {ds_name}")
                logger.info(f"===> Processing SEQUENTIAL EXTRACTION for dataset {ds_name}")
                
                run_sequential_extraction_pipeline(
                    ds_name, modalities, common_ids, y_aligned, base_out,
                    reg_extractors, n_shared_list, list(reg_models.keys()), progress_count_reg, reg_total_runs,
                    is_regression=True
                )
                
                print(f"===> COMPLETED SEQUENTIAL EXTRACTION for dataset {ds_name}")
                logger.info(f"===> COMPLETED SEQUENTIAL EXTRACTION for dataset {ds_name}")
                
                # Then, run sequential selection pipeline
                print(f"===> Processing SEQUENTIAL SELECTION for dataset {ds_name}")
                logger.info(f"===> Processing SEQUENTIAL SELECTION for dataset {ds_name}")
                
                run_sequential_selection_pipeline(
                    ds_name, modalities, common_ids, y_aligned, base_out,
                    reg_selectors, n_shared_list, list(reg_models.keys()), progress_count_reg, reg_total_runs,
                    is_regression=True
                )
                
                print(f"===> COMPLETED SEQUENTIAL SELECTION for dataset {ds_name}")
                logger.info(f"===> COMPLETED SEQUENTIAL SELECTION for dataset {ds_name}")
                
            elif args.fusion_first:
                # LEGACY: Fusion-First Architecture
                print(f"===> Processing with FUSION-FIRST architecture for dataset {ds_name}")
                logger.info(f"===> Using LEGACY FUSION-FIRST architecture: Fusion → Feature Processing → Model Training")
                
                # First, run extraction pipeline for all n_values
                print(f"===> Processing EXTRACTION for dataset {ds_name}")
                logger.info(f"===> Processing EXTRACTION for dataset {ds_name}")
                
                run_extraction_pipeline(
                    ds_name, modalities, common_ids, y_aligned, base_out,
                    reg_extractors, n_shared_list, list(reg_models.keys()), progress_count_reg, reg_total_runs,
                    is_regression=True
                )
                
                print(f"===> COMPLETED EXTRACTION for dataset {ds_name}")
                logger.info(f"===> COMPLETED EXTRACTION for dataset {ds_name}")
                
                # Then, run selection pipeline for all n_values
                print(f"===> Processing SELECTION for dataset {ds_name}")
                logger.info(f"===> Processing SELECTION for dataset {ds_name}")
                
                run_selection_pipeline(
                    ds_name, modalities, common_ids, y_aligned, base_out,
                    reg_selectors, n_shared_list, list(reg_models.keys()), progress_count_reg, reg_total_runs,
                    is_regression=True
                )
                
                print(f"===> COMPLETED SELECTION for dataset {ds_name}")
                logger.info(f"===> COMPLETED SELECTION for dataset {ds_name}")
                
            else:
                # DEFAULT: Feature-First Architecture
                print(f"===> Processing with FEATURE-FIRST architecture for dataset {ds_name}")
                logger.info(f"===> Using STANDARD FEATURE-FIRST architecture: Feature Processing → Fusion → Model Training")
                
                from feature_first_pipeline import run_feature_first_pipeline
                
                # Combine extractors and selectors
                all_algorithms = {**reg_extractors, **reg_selectors}
                
                # FIXED: Loop over all configured missing percentages
                for missing_percentage in MISSING_MODALITIES_CONFIG["missing_percentages"]:
                    print(f"===> Processing missing percentage: {missing_percentage*100:.0f}%")
                    logger.info(f"===> Processing missing percentage: {missing_percentage*100:.0f}%")
                    
                    # Run feature-first pipeline for this missing percentage
                    run_feature_first_pipeline(
                        ds_name=ds_name,
                        data_modalities=modalities,
                        common_ids=common_ids,
                        y=y_aligned,
                        base_out=base_out,
                        algorithms=all_algorithms,
                        n_values=n_shared_list,
                        models=list(reg_models.keys()),
                        is_regression=True,
                        missing_percentage=missing_percentage  # FIXED: Use actual missing percentage
                    )
                    
                    print(f"===> COMPLETED missing percentage {missing_percentage*100:.0f}% for dataset {ds_name}")
                    logger.info(f"===> COMPLETED missing percentage {missing_percentage*100:.0f}% for dataset {ds_name}")
                
                print(f"===> COMPLETED FEATURE-FIRST pipeline for dataset {ds_name}")
                logger.info(f"===> COMPLETED FEATURE-FIRST pipeline for dataset {ds_name}")
            
            # Combine best fold metrics from both extraction and selection
            print(f"===> Combining best fold metrics for dataset {ds_name}")
            logger.info(f"===> Combining best fold metrics for dataset {ds_name}")
            from cv import combine_best_fold_metrics
            combine_best_fold_metrics(ds_name, base_out)
            print(f"===> COMPLETED combining best fold metrics for dataset {ds_name}")
            logger.info(f"===> COMPLETED combining best fold metrics for dataset {ds_name}")
                
        except KeyboardInterrupt:
            logger.warning(f"KeyboardInterrupt during processing dataset {ds_name}. Aborting all processing.")
            raise  # Re-raise to abort all processing
        except Exception as e:
            logger.error(f"Error processing dataset {ds_name}: {str(e)}")
            logger.error(f"Continuing with next dataset")
            import traceback
            traceback.print_exc()
            continue  # Continue to next dataset

def process_classification_datasets(args):
    """Process all classification datasets."""
    classification_block_info = "\n=== CLASSIFICATION BLOCK (Breast, Colon, Kidney, etc.) ==="
    logger.info(classification_block_info)
    print(f"{classification_block_info}")
    clf_extractors = get_classification_extractors()
    clf_selectors = get_classification_selectors()
    clf_models = get_classification_models()
    n_shared_list = N_VALUES_LIST.copy()  # Shared list for both extraction and selection
    
    # Check if a specific n_val is requested via command line
    if args.n_val:
        n_val = int(args.n_val)
        if n_val in n_shared_list:
            n_shared_list = [n_val]
            logger.info(f"Processing only n_val = {n_val} as requested")
        else:
            logger.warning(f"Requested n_val = {n_val} not in {n_shared_list}, using all values")
    
    # Calculate total runs (now just extractors + selectors) * len(n_shared_list)
    clf_total_runs = (
        len(CLASSIFICATION_DATASETS) * 
        (len(clf_extractors) + len(clf_selectors)) *
        len(n_shared_list)
    )
    progress_count_clf = [0]
    
    for ds_conf in CLASSIFICATION_DATASETS:
        result = process_dataset(ds_conf, is_regression=False)
        if result is None:
            continue
            
        ds_name = result["name"]
        modalities = result["modalities"]
        common_ids = result["common_ids"]
        y_aligned = result["y_aligned"]
        
        # Set up output directory
        base_out = os.path.join(
            ds_conf.get("output_dir", "output"), ds_name
        )
        os.makedirs(base_out, exist_ok=True)
        
        try:
            if args.sequential:
                # SEQUENTIAL: Process one extractor/selector at a time through all fusion techniques
                print(f"===> Processing with SEQUENTIAL architecture for dataset {ds_name}")
                logger.info(f"===> Using SEQUENTIAL architecture: One extractor/selector → All fusion techniques → All models")
                
                from cv import run_sequential_extraction_pipeline, run_sequential_selection_pipeline
                
                # First, run sequential extraction pipeline
                print(f"===> Processing SEQUENTIAL EXTRACTION for dataset {ds_name}")
                logger.info(f"===> Processing SEQUENTIAL EXTRACTION for dataset {ds_name}")
                
                run_sequential_extraction_pipeline(
                    ds_name, modalities, common_ids, y_aligned, base_out,
                    clf_extractors, n_shared_list, list(clf_models.keys()), progress_count_clf, clf_total_runs,
                    is_regression=False
                )
                
                print(f"===> COMPLETED SEQUENTIAL EXTRACTION for dataset {ds_name}")
                logger.info(f"===> COMPLETED SEQUENTIAL EXTRACTION for dataset {ds_name}")
                
                # Then, run sequential selection pipeline
                print(f"===> Processing SEQUENTIAL SELECTION for dataset {ds_name}")
                logger.info(f"===> Processing SEQUENTIAL SELECTION for dataset {ds_name}")
                
                run_sequential_selection_pipeline(
                    ds_name, modalities, common_ids, y_aligned, base_out,
                    clf_selectors, n_shared_list, list(clf_models.keys()), progress_count_clf, clf_total_runs,
                    is_regression=False
                )
                
                print(f"===> COMPLETED SEQUENTIAL SELECTION for dataset {ds_name}")
                logger.info(f"===> COMPLETED SEQUENTIAL SELECTION for dataset {ds_name}")
                
            elif args.fusion_first:
                # LEGACY: Fusion-First Architecture
                print(f"===> Processing with FUSION-FIRST architecture for dataset {ds_name}")
                logger.info(f"===> Using LEGACY FUSION-FIRST architecture: Fusion → Feature Processing → Model Training")
                
                # First, run extraction pipeline for all n_values
                print(f"===> Processing EXTRACTION for dataset {ds_name}")
                logger.info(f"===> Processing EXTRACTION for dataset {ds_name}")
                
                run_extraction_pipeline(
                    ds_name, modalities, common_ids, y_aligned, base_out,
                    clf_extractors, n_shared_list, list(clf_models.keys()), progress_count_clf, clf_total_runs,
                    is_regression=False
                )
                
                print(f"===> COMPLETED EXTRACTION for dataset {ds_name}")
                logger.info(f"===> COMPLETED EXTRACTION for dataset {ds_name}")
                
                # Then, run selection pipeline for all n_values
                print(f"===> Processing SELECTION for dataset {ds_name}")
                logger.info(f"===> Processing SELECTION for dataset {ds_name}")
                
                run_selection_pipeline(
                    ds_name, modalities, common_ids, y_aligned, base_out,
                    clf_selectors, n_shared_list, list(clf_models.keys()), progress_count_clf, clf_total_runs,
                    is_regression=False
                )
                
                print(f"===> COMPLETED SELECTION for dataset {ds_name}")
                logger.info(f"===> COMPLETED SELECTION for dataset {ds_name}")
                
            else:
                # DEFAULT: Feature-First Architecture
                print(f"===> Processing with FEATURE-FIRST architecture for dataset {ds_name}")
                logger.info(f"===> Using STANDARD FEATURE-FIRST architecture: Feature Processing → Fusion → Model Training")
                
                from feature_first_pipeline import run_feature_first_pipeline
                
                # Combine extractors and selectors
                all_algorithms = {**clf_extractors, **clf_selectors}
                
                # FIXED: Loop over all configured missing percentages
                for missing_percentage in MISSING_MODALITIES_CONFIG["missing_percentages"]:
                    print(f"===> Processing missing percentage: {missing_percentage*100:.0f}%")
                    logger.info(f"===> Processing missing percentage: {missing_percentage*100:.0f}%")
                    
                    # Run feature-first pipeline for this missing percentage
                    run_feature_first_pipeline(
                        ds_name=ds_name,
                        data_modalities=modalities,
                        common_ids=common_ids,
                        y=y_aligned,
                        base_out=base_out,
                        algorithms=all_algorithms,
                        n_values=n_shared_list,
                        models=list(clf_models.keys()),
                        is_regression=False,
                        missing_percentage=missing_percentage  # FIXED: Use actual missing percentage
                    )
                    
                    print(f"===> COMPLETED missing percentage {missing_percentage*100:.0f}% for dataset {ds_name}")
                    logger.info(f"===> COMPLETED missing percentage {missing_percentage*100:.0f}% for dataset {ds_name}")
                
                print(f"===> COMPLETED FEATURE-FIRST pipeline for dataset {ds_name}")
                logger.info(f"===> COMPLETED FEATURE-FIRST pipeline for dataset {ds_name}")
            
            # Combine best fold metrics from both extraction and selection
            print(f"===> Combining best fold metrics for dataset {ds_name}")
            logger.info(f"===> Combining best fold metrics for dataset {ds_name}")
            from cv import combine_best_fold_metrics
            combine_best_fold_metrics(ds_name, base_out)
            print(f"===> COMPLETED combining best fold metrics for dataset {ds_name}")
            logger.info(f"===> COMPLETED combining best fold metrics for dataset {ds_name}")
                
        except KeyboardInterrupt:
            logger.warning(f"KeyboardInterrupt during processing dataset {ds_name}. Aborting all processing.")
            raise  # Re-raise to abort all processing
        except Exception as e:
            logger.error(f"Error processing dataset {ds_name}: {str(e)}")
            logger.error(f"Continuing with next dataset")
            import traceback
            traceback.print_exc()
            continue  # Continue to next dataset

def main():
    """Main function for running the pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Multi-modal machine learning pipeline for omics data")
    
    parser.add_argument(
        "--regression-only", action="store_true", help="Run only regression datasets"
    )
    parser.add_argument(
        "--classification-only", action="store_true", help="Run only classification datasets"
    )
    parser.add_argument(
        "--dataset", type=str, help="Run only a specific dataset by name"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode with more logging"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose mode with detailed logging"
    )
    parser.add_argument(
        "--n-val", type=int, help=f"Run only a specific n_val from {N_VALUES_LIST}"
    )
    parser.add_argument(
        "--mad-only", action="store_true", help="Run only MAD analysis without model training"
    )
    parser.add_argument(
        "--skip-mad", action="store_true", help="Skip MAD analysis and run only model training"
    )
    parser.add_argument(
        "--feature-engineering", action="store_true", 
        help="Enable feature engineering tweaks (Sparse PLS-DA for MCC, Kernel PCA for R²)"
    )
    parser.add_argument(
        "--fusion-upgrades", action="store_true",
        help="Enable fusion upgrades (Attention-weighted concatenation, Late-fusion stacking)"
    )
    parser.add_argument(
        "--fusion-first", action="store_true",
        help="Use legacy fusion-first architecture (Fusion->Feature Processing->Model Training)"
    )
    parser.add_argument(
        "--sequential", action="store_true",
        help="Use sequential processing (one extractor/selector at a time through all fusion techniques)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logging levels based on arguments
    setup_logging_levels(args)
    
    # Enable feature engineering if requested
    if args.feature_engineering:
        from config import FEATURE_ENGINEERING_CONFIG
        FEATURE_ENGINEERING_CONFIG["enabled"] = True
        logger.info("Feature engineering tweaks enabled via CLI")
        logger.info("  - Sparse PLS-DA (32 components) for better MCC in classification")
        logger.info("  - Kernel PCA RBF (64 components) for higher R² in regression")
    
    # Enable fusion upgrades if requested
    if args.fusion_upgrades:
        from config import FUSION_UPGRADES_CONFIG
        FUSION_UPGRADES_CONFIG["enabled"] = True
        logger.info("Fusion upgrades enabled via CLI")
        logger.info("  - Attention-weighted concatenation: Sample-specific weighting (AML R² +0.05, Colon MCC +0.04)")
        logger.info("  - Late-fusion stacking: Per-omic model predictions as meta-features")
    
    # Log startup information
    logger.info("=" * 70)
    logger.info("Multi-modal Machine Learning Pipeline for Omics Data")
    logger.info("=" * 70)
    logger.debug(f"Command line arguments: {vars(args)}")
    
    # Record the start time for total algorithm timing
    algorithm_start_time = time.time()
    start_time_formatted = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(algorithm_start_time))
    
    # Log algorithm start time
    startup_msg = f"Algorithm started at: {start_time_formatted}"
    logger.info(startup_msg)
    logger.info("=" * 70)
    print(f"\n{startup_msg}")
    print("=" * 70)
    
    # Handle MAD analysis
    if args.skip_mad:
        log_mad_analysis_info("MAD analysis skipped by user request")
    elif args.mad_only:
        log_mad_analysis_info("Running MAD analysis only (no model training)")
        log_pipeline_stage("MAD_ANALYSIS_START")
        try:
            run_mad_analysis(output_dir="output")
            log_mad_analysis_info("MAD analysis completed successfully")
            log_pipeline_stage("MAD_ANALYSIS_END", details="Completed successfully")
        except Exception as e:
            log_mad_analysis_info(f"MAD analysis failed: {str(e)}", level="error")
            logger.error(f"MAD analysis error details: {str(e)}")
            import traceback
            logger.debug(f"MAD analysis traceback:\n{traceback.format_exc()}")
        
        # Calculate and log total time for MAD-only run
        log_timing_summary(algorithm_start_time, "MAD-only analysis")
        return
    else:
        # Run MAD analysis before model training
        log_pipeline_stage("MAD_ANALYSIS_START")
        log_mad_analysis_info("Starting MAD analysis")
        try:
            run_mad_analysis(output_dir="output")
            log_mad_analysis_info("MAD analysis completed successfully")
            log_pipeline_stage("MAD_ANALYSIS_END", details="Completed successfully")
        except Exception as e:
            log_mad_analysis_info(f"MAD analysis failed: {str(e)}", level="error")
            logger.error(f"Error in MAD analysis: {str(e)}")
            logger.error("Continuing with model training...")
            import traceback
            logger.debug(f"MAD analysis traceback:\n{traceback.format_exc()}")
    
    # Start model training phase
    log_pipeline_stage("MODEL_TRAINING_START")
    
    # Process datasets based on arguments
    if args.dataset:
        # Find and process only the specified dataset
        log_pipeline_stage("SINGLE_DATASET", dataset=args.dataset.upper())
        process_single_dataset(args.dataset.lower(), args)
    elif args.regression_only:
        # Process only regression datasets
        log_pipeline_stage("REGRESSION_DATASETS_START")
        process_regression_datasets(args)
        log_pipeline_stage("REGRESSION_DATASETS_END")
    elif args.classification_only:
        # Process only classification datasets
        log_pipeline_stage("CLASSIFICATION_DATASETS_START")
        process_classification_datasets(args)
        log_pipeline_stage("CLASSIFICATION_DATASETS_END")
    else:
        # Process all datasets
        log_pipeline_stage("ALL_DATASETS_START")
        process_regression_datasets(args)
        process_classification_datasets(args)
        log_pipeline_stage("ALL_DATASETS_END")
    
    # Calculate total algorithm runtime and log comprehensive timing summary
    timing_info = log_timing_summary(algorithm_start_time, "Pipeline")
    
    # Log completion information to pipeline stage
    completion_msg = f"Pipeline completed in {timing_info['hours']}h {timing_info['minutes']}m {timing_info['seconds']}s"
    log_pipeline_stage("PIPELINE_COMPLETE", details=completion_msg)

def process_single_dataset(target_ds, args):
    """Process a single dataset specified by name."""
    found = False
    
    # First try regression datasets
    for ds_conf in REGRESSION_DATASETS:
        if ds_conf["name"].lower() == target_ds:
            logger.info(f"Processing single regression dataset: {ds_conf['name']}")
            result = process_dataset(ds_conf, is_regression=True)
            if result:
                ds_name = result["name"]
                base_out = os.path.join(
                    ds_conf.get("output_dir", "output"), ds_name
                )
                os.makedirs(base_out, exist_ok=True)
                
                # Get n_val list (filtered if requested via args)
                n_val_list = N_VALUES_LIST.copy()
                if args.n_val and args.n_val in n_val_list:
                    n_val_list = [args.n_val]
                
                if args.sequential:
                    # Use sequential processing
                    from cv import run_sequential_extraction_pipeline, run_sequential_selection_pipeline
                    
                    run_sequential_extraction_pipeline(
                        ds_name, result["modalities"], result["common_ids"], 
                        result["y_aligned"], base_out,
                        get_regression_extractors(), n_val_list, 
                        list(get_regression_models().keys()),
                        [0], 1, is_regression=True
                    )
                    
                    run_sequential_selection_pipeline(
                        ds_name, result["modalities"], result["common_ids"], 
                        result["y_aligned"], base_out,
                        get_regression_selectors(), n_val_list, 
                        list(get_regression_models().keys()),
                        [0], 1, is_regression=True
                    )
                else:
                    # Use standard processing
                    run_extraction_pipeline(
                        ds_name, result["modalities"], result["common_ids"], 
                        result["y_aligned"], base_out,
                        get_regression_extractors(), n_val_list, 
                        list(get_regression_models().keys()),  # CURRENT IMPLEMENTATION: Use getter function
                        [0], 1, is_regression=True
                    )
                    
                    run_selection_pipeline(
                        ds_name, result["modalities"], result["common_ids"], 
                        result["y_aligned"], base_out,
                        get_regression_selectors(), n_val_list, 
                        list(get_regression_models().keys()),  # CURRENT IMPLEMENTATION: Use getter function
                        [0], 1, is_regression=True
                    )
                
                # Combine best fold metrics from both extraction and selection
                from cv import combine_best_fold_metrics
                combine_best_fold_metrics(ds_name, base_out)
            found = True
            break
    
    # Then try classification datasets if not found
    if not found:
        for ds_conf in CLASSIFICATION_DATASETS:
            if ds_conf["name"].lower() == target_ds:
                logger.info(f"Processing single classification dataset: {ds_conf['name']}")
                result = process_dataset(ds_conf, is_regression=False)
                if result:
                    ds_name = result["name"]
                    base_out = os.path.join(
                        ds_conf.get("output_dir", "output"), ds_name
                    )
                    os.makedirs(base_out, exist_ok=True)
                    
                    # Get n_val list (filtered if requested via args)
                    n_val_list = N_VALUES_LIST.copy()
                    if args.n_val and args.n_val in n_val_list:
                        n_val_list = [args.n_val]
                    
                    if args.sequential:
                        # Use sequential processing
                        from cv import run_sequential_extraction_pipeline, run_sequential_selection_pipeline
                        
                        run_sequential_extraction_pipeline(
                            ds_name, result["modalities"], result["common_ids"], 
                            result["y_aligned"], base_out,
                            get_classification_extractors(), n_val_list, 
                            list(get_classification_models().keys()), 
                            [0], 1, is_regression=False
                        )
                        
                        run_sequential_selection_pipeline(
                            ds_name, result["modalities"], result["common_ids"], 
                            result["y_aligned"], base_out,
                            get_classification_selectors(), n_val_list, 
                            list(get_classification_models().keys()), 
                            [0], 1, is_regression=False
                        )
                    else:
                        # Use standard processing
                        run_extraction_pipeline(
                            ds_name, result["modalities"], result["common_ids"], 
                            result["y_aligned"], base_out,
                            get_classification_extractors(), n_val_list, 
                            list(get_classification_models().keys()), 
                            [0], 1, is_regression=False
                        )
                        
                        run_selection_pipeline(
                            ds_name, result["modalities"], result["common_ids"], 
                            result["y_aligned"], base_out,
                            get_classification_selectors(), n_val_list, 
                            list(get_classification_models().keys()), 
                            [0], 1, is_regression=False
                        )
                    
                    # Combine best fold metrics from both extraction and selection
                    from cv import combine_best_fold_metrics
                    combine_best_fold_metrics(ds_name, base_out)
                found = True
                break
    
    if not found:
        logger.error(f"Error: Dataset '{target_ds}' not found in configurations.")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\nPipeline interrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nError in main pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 