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

# Local imports
from Z_alg.config import (
    REGRESSION_DATASETS, CLASSIFICATION_DATASETS,
    MAX_COMPONENTS, MAX_FEATURES
)
from Z_alg.io import load_dataset
from Z_alg.models import (
    get_regression_extractors, get_regression_selectors,
    get_classification_extractors, get_classification_selectors,
    get_classification_models
)
from Z_alg.cv import (
    run_extraction_pipeline, run_selection_pipeline
)
from Z_alg.utils import log_resource_usage

# Remove any existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Set up logging to file and console
log_file = "debug.log"
file_handler = logging.FileHandler(log_file, mode='w')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
file_handler.setFormatter(file_formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
console_handler.setFormatter(console_formatter)

logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, console_handler])
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
    log_resource_usage(f"Starting {ds_conf['name']}")
    
    try:
        # Extract the dataset name
        ds_name = ds_conf["name"]
        logger.info(f"\n>> Processing {ds_name} dataset...")
        
        # Load the dataset
        result = load_dataset(ds_conf)
        
        if result is None:
            logger.warning(f"Error: Failed to load dataset {ds_name}")
            return None
            
        modalities, common_ids, y_aligned = result
        
        # Verify data integrity
        if len(common_ids) < 10:
            logger.warning(f"Warning: Too few samples ({len(common_ids)}) in {ds_name}, skipping")
            return None
            
        if not y_aligned.size:
            logger.warning(f"Error: No target values available for {ds_name}")
            return None
        
        for mod_name, mod_df in modalities.items():
            if mod_df.empty:
                logger.warning(f"Warning: Empty modality {mod_name} in {ds_name}")
                
        # Return the loaded data
        return {
            "name": ds_name,
            "modalities": modalities,
            "common_ids": common_ids,
            "y_aligned": y_aligned
        }
        
    except Exception as e:
        logger.error(f"Error processing dataset {ds_conf['name']}: {str(e)}")
        return None

def process_regression_datasets():
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
        
        # Run extraction and selection with matching n_val pairs
        for n_val in n_shared_list:
            try:
                # Add debug logging
                print(f"===> Processing n_val = {n_val} for dataset {ds_name}")
                logger.info(f"===> Processing n_val = {n_val} for dataset {ds_name}")
                
                # Run extraction pipeline with n_val as n_components
                run_extraction_pipeline(
                    ds_name, modalities, common_ids, y_aligned, base_out,
                    reg_extractors, [n_val], reg_models, progress_count_reg, reg_total_runs,
                    is_regression=True
                )
                
                # Run selection pipeline with the same n_val as n_features
                run_selection_pipeline(
                    ds_name, modalities, common_ids, y_aligned, base_out,
                    reg_selectors, [n_val], reg_models, progress_count_reg, reg_total_runs,
                    is_regression=True
                )
                
                print(f"===> COMPLETED n_val = {n_val} for dataset {ds_name}")
                logger.info(f"===> COMPLETED n_val = {n_val} for dataset {ds_name}")
            except KeyboardInterrupt:
                logger.warning(f"KeyboardInterrupt during n_val = {n_val}. Aborting all processing.")
                raise  # Re-raise to abort all processing
            except Exception as e:
                logger.error(f"Error processing n_val = {n_val}: {str(e)}")
                logger.error(f"Continuing with next n_val")
                import traceback
                traceback.print_exc()
                continue  # Continue to next n_val

def process_classification_datasets():
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
        
        # Run extraction and selection with matching n_val pairs
        for n_val in n_shared_list:
            try:
                # Add debug logging
                print(f"===> Processing n_val = {n_val} for dataset {ds_name}")
                logger.info(f"===> Processing n_val = {n_val} for dataset {ds_name}")
                
                # Run extraction pipeline with n_val as n_components
                run_extraction_pipeline(
                    ds_name, modalities, common_ids, y_aligned, base_out,
                    clf_extractors, [n_val], list(clf_models.keys()), progress_count_clf, clf_total_runs,
                    is_regression=False
                )
                
                # Run selection pipeline with the same n_val as n_features
                run_selection_pipeline(
                    ds_name, modalities, common_ids, y_aligned, base_out,
                    clf_selectors, [n_val], list(clf_models.keys()), progress_count_clf, clf_total_runs,
                    is_regression=False
                )
                
                print(f"===> COMPLETED n_val = {n_val} for dataset {ds_name}")
                logger.info(f"===> COMPLETED n_val = {n_val} for dataset {ds_name}")
            except KeyboardInterrupt:
                logger.warning(f"KeyboardInterrupt during n_val = {n_val}. Aborting all processing.")
                raise  # Re-raise to abort all processing
            except Exception as e:
                logger.error(f"Error processing n_val = {n_val}: {str(e)}")
                logger.error(f"Continuing with next n_val")
                import traceback
                traceback.print_exc()
                continue  # Continue to next n_val

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
        "--n-val", type=int, help="Run only a specific n_val (8, 16, or 32)"
    )
    
    global args
    args = parser.parse_args()
    
    # Set up debug mode if requested
    if args.debug:
        os.environ["DEBUG_RESOURCES"] = "1"
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)
        
    # Print welcome message
    logger.info("=" * 70)
    logger.info("Multi-modal Machine Learning Pipeline for Omics Data")
    logger.info("=" * 70)
    
    start_time = time.time()
    
    # Process datasets based on arguments
    if args.dataset:
        # Find and process only the specified dataset
        target_ds = args.dataset.lower()
        found = False
        
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
                    
                    run_extraction_pipeline(
                        ds_name, result["modalities"], result["common_ids"], 
                        result["y_aligned"], base_out,
                        get_regression_extractors(), [8, 16, 32], 
                        ["LinearRegression", "RandomForestRegressor", "ElasticNet"], 
                        [0], 1, is_regression=True
                    )
                    
                    run_selection_pipeline(
                        ds_name, result["modalities"], result["common_ids"], 
                        result["y_aligned"], base_out,
                        get_regression_selectors(), [8, 16, 32], 
                        ["LinearRegression", "RandomForestRegressor", "ElasticNet"], 
                        [0], 1, is_regression=True
                    )
                found = True
                break
                
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
                        
                        run_extraction_pipeline(
                            ds_name, result["modalities"], result["common_ids"], 
                            result["y_aligned"], base_out,
                            get_classification_extractors(), [8, 16, 32], 
                            list(get_classification_models().keys()), 
                            [0], 1, is_regression=False
                        )
                        
                        run_selection_pipeline(
                            ds_name, result["modalities"], result["common_ids"], 
                            result["y_aligned"], base_out,
                            get_classification_selectors(), [8, 16, 32], 
                            list(get_classification_models().keys()), 
                            [0], 1, is_regression=False
                        )
                    found = True
                    break
                    
        if not found:
            logger.error(f"Error: Dataset '{args.dataset}' not found in configurations.")
            sys.exit(1)
            
    elif args.regression_only:
        # Process only regression datasets
        process_regression_datasets()
    elif args.classification_only:
        # Process only classification datasets
        process_classification_datasets()
    else:
        # Process all datasets
        process_regression_datasets()
        process_classification_datasets()
    
    # Print completion information
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info("\n" + "=" * 70)
    logger.info(f"Pipeline completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    logger.info("=" * 70)
    print("\n" + "=" * 70)
    print(f"Pipeline completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print("=" * 70)

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