"""
Logging utilities for Z_alg package.

This module contains logging functions to avoid circular imports between cli.py and other modules.
"""

import logging
import os
import time

logger = logging.getLogger(__name__)


def setup_logging_levels(args):
    """
    Set up logging levels based on command line arguments.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments
    """
    # Get the console handler
    console_handler = None
    for handler in logging.root.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            console_handler = handler
            break
    
    if args.debug:
        # Debug mode: show everything in console and file
        os.environ["Z_ALG_DEBUG"] = "1"
        os.environ["DEBUG_RESOURCES"] = "1"
        if console_handler:
            console_handler.setLevel(logging.DEBUG)
        logger.info("Debug mode enabled - showing all debug information")
        
    elif args.verbose:
        # Verbose mode: show info and above in console, everything in file
        os.environ["Z_ALG_VERBOSE"] = "1"
        if console_handler:
            console_handler.setLevel(logging.INFO)
        logger.info("Verbose mode enabled - showing detailed information")
        
    else:
        # Normal mode: show only warnings and errors in console, everything in file
        if console_handler:
            console_handler.setLevel(logging.WARNING)


def log_pipeline_stage(stage_name: str, dataset: str = None, details: str = None):
    """
    Log pipeline stage information with consistent formatting.
    
    Parameters
    ----------
    stage_name : str
        Name of the pipeline stage
    dataset : str, optional
        Dataset being processed
    details : str, optional
        Additional details about the stage
    """
    if dataset and details:
        message = f"[{stage_name}] {dataset} | {details}"
    elif dataset:
        message = f"[{stage_name}] {dataset}"
    else:
        message = f"[{stage_name}] {details}" if details else f"[{stage_name}]"
    
    logger.info(message)
    # Also log to debug.log with more details
    logger.debug(f"Pipeline stage: {message}")


def log_mad_analysis_info(message: str, level: str = "info"):
    """
    Log MAD analysis specific information.
    
    Parameters
    ----------
    message : str
        Message to log
    level : str
        Logging level (info, warning, error, debug)
    """
    mad_message = f"[MAD_ANALYSIS] {message}"
    
    if level == "debug":
        logger.debug(mad_message)
    elif level == "info":
        logger.info(mad_message)
    elif level == "warning":
        logger.warning(mad_message)
    elif level == "error":
        logger.error(mad_message)


def log_dataset_preparation(dataset_name: str, modalities: dict, common_ids: list, y_shape: tuple, success: bool = True):
    """
    Log dataset preparation information.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    modalities : dict
        Dictionary of modality DataFrames
    common_ids : list
        List of common sample IDs
    y_shape : tuple
        Shape of target variable
    success : bool
        Whether preparation was successful
    """
    if success:
        logger.info(f"[DATASET_PREP] {dataset_name} prepared successfully")
        logger.info(f"[DATASET_PREP] {dataset_name} - {len(common_ids)} samples, {len(modalities)} modalities")
        logger.debug(f"[DATASET_PREP] {dataset_name} - Target shape: {y_shape}")
        
        for mod_name, mod_df in modalities.items():
            if mod_df is not None:
                logger.debug(f"[DATASET_PREP] {dataset_name} - {mod_name}: {mod_df.shape}")
            else:
                logger.warning(f"[DATASET_PREP] {dataset_name} - {mod_name}: None/Empty")
    else:
        logger.error(f"[DATASET_PREP] {dataset_name} preparation failed")


def log_model_training_info(model_name: str, dataset: str, fold_idx: int, train_samples: int, val_samples: int, success: bool = True, fallback: bool = False, error_msg: str = None):
    """
    Log model training information.
    
    Parameters
    ----------
    model_name : str
        Name of the model
    dataset : str
        Dataset name
    fold_idx : int
        Fold index
    train_samples : int
        Number of training samples
    val_samples : int
        Number of validation samples
    success : bool
        Whether training was successful
    fallback : bool
        Whether fallback strategy was used
    error_msg : str, optional
        Error message if training failed
    """
    fold_str = f"fold {fold_idx}" if fold_idx is not None else "training"
    
    if success:
        if fallback:
            logger.warning(f"[MODEL_TRAINING] {dataset} - {model_name} ({fold_str}) - Used fallback strategy")
        else:
            logger.info(f"[MODEL_TRAINING] {dataset} - {model_name} ({fold_str}) - Trained successfully")
        logger.debug(f"[MODEL_TRAINING] {dataset} - {model_name} ({fold_str}) - Train: {train_samples}, Val: {val_samples}")
    else:
        logger.error(f"[MODEL_TRAINING] {dataset} - {model_name} ({fold_str}) - Training failed")
        if error_msg:
            logger.error(f"[MODEL_TRAINING] {dataset} - {model_name} ({fold_str}) - Error: {error_msg}")


def log_data_save_info(dataset: str, file_type: str, file_path: str, success: bool = True, error_msg: str = None):
    """
    Log data saving information.
    
    Parameters
    ----------
    dataset : str
        Dataset name
    file_type : str
        Type of file being saved (metrics, plots, etc.)
    file_path : str
        Path where file was saved
    success : bool
        Whether saving was successful
    error_msg : str, optional
        Error message if saving failed
    """
    if success:
        logger.info(f"[DATA_SAVE] {dataset} - {file_type} saved to: {file_path}")
        logger.debug(f"[DATA_SAVE] {dataset} - {file_type} save operation completed successfully")
    else:
        logger.error(f"[DATA_SAVE] {dataset} - Failed to save {file_type} to: {file_path}")
        if error_msg:
            logger.error(f"[DATA_SAVE] {dataset} - {file_type} save error: {error_msg}")


def log_plot_save_info(dataset: str, plot_type: str, plot_path: str, success: bool = True, error_msg: str = None):
    """
    Log plot saving information.
    
    Parameters
    ----------
    dataset : str
        Dataset name
    plot_type : str
        Type of plot being saved
    plot_path : str
        Path where plot was saved
    success : bool
        Whether saving was successful
    error_msg : str, optional
        Error message if saving failed
    """
    if success:
        logger.info(f"[PLOT_SAVE] {dataset} - {plot_type} plot saved to: {plot_path}")
        logger.debug(f"[PLOT_SAVE] {dataset} - {plot_type} plot save operation completed successfully")
    else:
        logger.error(f"[PLOT_SAVE] {dataset} - Failed to save {plot_type} plot to: {plot_path}")
        if error_msg:
            logger.error(f"[PLOT_SAVE] {dataset} - {plot_type} plot save error: {error_msg}")


def log_timing_summary(start_time: float, operation_name: str = "Algorithm", print_to_console: bool = True):
    """
    Log timing summary with consistent formatting.
    
    Parameters
    ----------
    start_time : float
        Start time (from time.time())
    operation_name : str
        Name of the operation being timed
    print_to_console : bool
        Whether to also print to console
    """
    # Calculate total runtime
    total_elapsed_time = time.time() - start_time
    hours, remainder = divmod(total_elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Format times
    start_time_formatted = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    end_time_formatted = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    # Create timing summary
    timing_summary = (
        f"{operation_name} started: {start_time_formatted}\n"
        f"{operation_name} ended: {end_time_formatted}\n"
        f"Total runtime: {int(hours)}h {int(minutes)}m {int(seconds)}s"
    )
    
    # Log to debug.log
    logger.info("\n" + "=" * 70)
    logger.info(f"{operation_name.upper()} TIMING SUMMARY")
    logger.info("=" * 70)
    logger.info(timing_summary)
    logger.info("=" * 70)
    
    # Print to console if requested
    if print_to_console:
        print("\n" + "=" * 70)
        print(f"{operation_name.upper()} TIMING SUMMARY")
        print("=" * 70)
        print(timing_summary)
        print("=" * 70)
    
    return {
        'total_seconds': total_elapsed_time,
        'hours': int(hours),
        'minutes': int(minutes),
        'seconds': int(seconds),
        'start_time': start_time_formatted,
        'end_time': end_time_formatted,
        'summary': timing_summary
    } 