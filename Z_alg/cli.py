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

# Local imports
from Z_alg.config import (
    REGRESSION_DATASETS, CLASSIFICATION_DATASETS,
    MAX_COMPONENTS, MAX_FEATURES
)
from Z_alg.data_io import load_dataset
from Z_alg.models import (
    get_regression_extractors, get_regression_selectors,
    get_classification_extractors, get_classification_selectors,
    get_classification_models
)
from Z_alg.cv import (
    run_extraction_pipeline, run_selection_pipeline
)
from Z_alg.utils import comprehensive_logger
from Z_alg.mad_analysis import run_mad_analysis
from Z_alg.logging_utils import (
    setup_logging_levels, log_pipeline_stage, log_mad_analysis_info,
    log_dataset_preparation, log_model_training_info, log_data_save_info, log_plot_save_info,
    log_timing_summary
)

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
        logger.debug(f"[DATASET_PREP] {ds_name} - Starting dataset preparation")
        
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
        
        logger.debug(f"[DATASET_PREP] {ds_name} - Loading modalities: {modality_short_names}")
        logger.debug(f"[DATASET_PREP] {ds_name} - Outcome column: {outcome_col}, Task: {task_type}")
        
        # Call the new optimized load_dataset function
        modalities_data, y_aligned, common_ids = load_dataset(
            ds_name.lower(), 
            modality_short_names, 
            outcome_col, 
            task_type,
            parallel=True,
            use_cache=True
        )
        
        # Check if loading was successful
        if modalities_data is None or len(common_ids) == 0:
            logger.warning(f"Error: Failed to load dataset {ds_name}")
            log_dataset_preparation(ds_name, {}, [], (), success=False)
            return None
            
        # Data is already in the correct format
        modalities = modalities_data
        
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
                y_aligned = np.nan_to_num(y_aligned, nan=0.0, posinf=0.0, neginf=0.0)
                logger.debug(f"[DATASET_PREP] {ds_name} - Cleaned NaN/infinite values in target")
        
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
        if not is_regression:
            unique, counts = np.unique(y_aligned, return_counts=True)
            min_samples = np.min(counts)
            n_classes = len(unique)
            
            logger.info(f"Class distribution for {ds_name}: {dict(zip(unique, counts))}")
            logger.debug(f"[DATASET_PREP] {ds_name} - {n_classes} classes, min samples per class: {min_samples}")
            
            # Check for problematic class distributions
            classes_with_few_samples = unique[counts < 2]
            if len(classes_with_few_samples) > 0:
                logger.info(f"Dataset {ds_name} has classes with < 2 samples: {classes_with_few_samples}")
                logger.info(f"Original class distribution: {dict(zip(unique, counts))}")
                logger.info(f"Filtering out these classes to ensure proper cross-validation")
                logger.debug(f"[DATASET_PREP] {ds_name} - Filtering classes with insufficient samples")
                
                # Actually filter out classes with insufficient samples
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
                for mod_name, mod_df in modalities.items():
                    # Keep only columns (samples) that correspond to filtered_common_ids
                    available_cols = [col for col in filtered_common_ids if col in mod_df.columns]
                    if len(available_cols) > 0:
                        filtered_modalities[mod_name] = mod_df[available_cols]
                    else:
                        logger.warning(f"No samples remaining in modality {mod_name} after class filtering")
                
                # Update the dataset with filtered data
                modalities = filtered_modalities
                common_ids = filtered_common_ids
                y_aligned = filtered_y_aligned
                
                # Relabel classes to be consecutive integers starting from 0
                # Sort valid_classes to ensure consistent mapping
                valid_classes_sorted = np.sort(valid_classes)
                label_mapping = {old_label: new_label for new_label, old_label in enumerate(valid_classes_sorted)}
                y_aligned = np.array([label_mapping[label] for label in y_aligned])
                
                logger.info(f"Filtered dataset from {len(y_aligned) + np.sum(~valid_mask)} to {len(y_aligned)} samples")
                logger.info(f"Reduced from {n_classes} to {len(valid_classes)} classes")
                logger.info(f"Class relabeling mapping: {label_mapping}")
                logger.info(f"Final class distribution: {dict(zip(range(len(valid_classes_sorted)), np.bincount(y_aligned)))}")
                logger.debug(f"[DATASET_PREP] {ds_name} - Class filtering completed")
            else:
                logger.info(f"All classes have sufficient samples (>= 2)")
                logger.debug(f"[DATASET_PREP] {ds_name} - No class filtering needed")
        
        # Final validation after any filtering
        if not is_regression:
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
    logger.info("=== REGRESSION BLOCK (AML, Sarcoma) ===")
    reg_extractors = get_regression_extractors()
    reg_selectors = get_regression_selectors()
    reg_models = ["LinearRegression", "RandomForestRegressor", "ElasticNet"]
    n_shared_list = [8, 16, 32]  # Shared list for both extraction and selection
    
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
            # First, run extraction pipeline for all n_values
            print(f"===> Processing EXTRACTION for dataset {ds_name}")
            logger.info(f"===> Processing EXTRACTION for dataset {ds_name}")
            
            run_extraction_pipeline(
                ds_name, modalities, common_ids, y_aligned, base_out,
                reg_extractors, n_shared_list, reg_models, progress_count_reg, reg_total_runs,
                is_regression=True
            )
            
            print(f"===> COMPLETED EXTRACTION for dataset {ds_name}")
            logger.info(f"===> COMPLETED EXTRACTION for dataset {ds_name}")
            
            # Then, run selection pipeline for all n_values
            print(f"===> Processing SELECTION for dataset {ds_name}")
            logger.info(f"===> Processing SELECTION for dataset {ds_name}")
            
            run_selection_pipeline(
                ds_name, modalities, common_ids, y_aligned, base_out,
                reg_selectors, n_shared_list, reg_models, progress_count_reg, reg_total_runs,
                is_regression=True
            )
            
            print(f"===> COMPLETED SELECTION for dataset {ds_name}")
            logger.info(f"===> COMPLETED SELECTION for dataset {ds_name}")
            
            # Combine best fold metrics from both extraction and selection
            print(f"===> Combining best fold metrics for dataset {ds_name}")
            logger.info(f"===> Combining best fold metrics for dataset {ds_name}")
            from Z_alg.cv import combine_best_fold_metrics
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
    logger.info("\n=== CLASSIFICATION BLOCK (Breast, Colon, Kidney, etc.) ===")
    clf_extractors = get_classification_extractors()
    clf_selectors = get_classification_selectors()
    clf_models = get_classification_models()
    n_shared_list = [8, 16, 32]  # Shared list for both extraction and selection
    
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
            
            # Combine best fold metrics from both extraction and selection
            print(f"===> Combining best fold metrics for dataset {ds_name}")
            logger.info(f"===> Combining best fold metrics for dataset {ds_name}")
            from Z_alg.cv import combine_best_fold_metrics
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
        "--n-val", type=int, help="Run only a specific n_val (8, 16, or 32)"
    )
    parser.add_argument(
        "--mad-only", action="store_true", help="Run only MAD analysis without model training"
    )
    parser.add_argument(
        "--skip-mad", action="store_true", help="Skip MAD analysis and run only model training"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logging levels based on arguments
    setup_logging_levels(args)
    
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
                n_val_list = [8, 16, 32]
                if args.n_val and args.n_val in n_val_list:
                    n_val_list = [args.n_val]
                
                run_extraction_pipeline(
                    ds_name, result["modalities"], result["common_ids"], 
                    result["y_aligned"], base_out,
                    get_regression_extractors(), n_val_list, 
                    ["LinearRegression", "RandomForestRegressor", "ElasticNet"], 
                    [0], 1, is_regression=True
                )
                
                run_selection_pipeline(
                    ds_name, result["modalities"], result["common_ids"], 
                    result["y_aligned"], base_out,
                    get_regression_selectors(), n_val_list, 
                    ["LinearRegression", "RandomForestRegressor", "ElasticNet"], 
                    [0], 1, is_regression=True
                )
                
                # Combine best fold metrics from both extraction and selection
                from Z_alg.cv import combine_best_fold_metrics
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
                    n_val_list = [8, 16, 32]
                    if args.n_val and args.n_val in n_val_list:
                        n_val_list = [args.n_val]
                    
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
                    from Z_alg.cv import combine_best_fold_metrics
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