#!/usr/bin/env python3
"""
Cross-validation module for model training and evaluation.
"""

import os
import numpy as np
import pandas as pd
import warnings
from joblib import Parallel, delayed, parallel_config

# Suppress sklearn warnings if configured
try:
    from config import WARNING_SUPPRESSION_CONFIG
    if WARNING_SUPPRESSION_CONFIG.get("suppress_sklearn_warnings", True):
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
        warnings.filterwarnings("ignore", message="invalid value encountered in divide")
        warnings.filterwarnings("ignore", message="divide by zero encountered")
except ImportError:
    pass
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, matthews_corrcoef
)
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
import time
import joblib
import shutil
import glob
from sklearn.base import clone, BaseEstimator, TransformerMixin

# Import for hyperparameter loading
import json
import pathlib

# Local imports
from config import N_JOBS, DatasetConfig, FEATURE_EXTRACTION_CONFIG
from preprocessing import process_with_missing_modalities, CrossValidationTargetValidator
from fusion import merge_modalities, ModalityImputer
from utils import suppress_sklearn_warnings, safe_r2_score, safe_mcc_score
from models import (
    get_model_object, cached_fit_transform_selector_regression,
    transform_selector_regression, cached_fit_transform_selector_classification,
    transform_selector_classification, cached_fit_transform_extractor_classification,
    transform_extractor_classification, get_selector_object
)
# Removed unused legacy imports
from _process_single_modality import align_samples_to_modalities, verify_data_alignment
from logging_utils import log_pipeline_stage, log_data_save_info, log_model_training_info, log_plot_save_info

# Initialize logger
logger = logging.getLogger(__name__)

# Hyperparameter directory
HP_DIR = pathlib.Path("hp_best")

def load_best(dataset, extr_nm, model_nm, task):
    """
    Load best hyperparameters for a given dataset, extractor, and model combination.
    This function is a wrapper around the more comprehensive load_best_hyperparameters from models.py.
    
    Parameters
    ----------
    dataset : str
        Dataset name
    extr_nm : str
        Extractor name
    model_nm : str
        Model name
    task : str
        Task type ("reg" or "clf")
        
    Returns
    -------
    dict
        Best hyperparameters (raw format for backward compatibility), or empty dict if not found
    """
    from models import load_best_hyperparameters
    
    hyperparams = load_best_hyperparameters(dataset, extr_nm, model_nm, task)
    
    # Return raw parameters combining both extractor and model parameters for backward compatibility
    combined_params = {}
    
    # Add extractor parameters with extractor__extractor__ prefix (to match tuning format)
    for key, value in hyperparams['extractor_params'].items():
        combined_params[f'extractor__extractor__{key}'] = value
    
    # Add model parameters with model__ prefix
    for key, value in hyperparams['model_params'].items():
        combined_params[f'model__{key}'] = value
    
    return combined_params

# 1. Add a threshold for severe alignment loss at the top of the file:
SEVERE_ALIGNMENT_LOSS_THRESHOLD = 0.3  # 30%
MIN_SAMPLES_PER_FOLD = 5

# Add improved CV configuration at the top of the file
CV_CONFIG = {
    "min_samples_per_class_per_fold": 1,  # Reduced from 2 to 1 for very small datasets
    "min_total_samples_for_stratified": 6,  # Reduced from 10 to 6
    "min_samples_per_fold": 3,  # Reduced from 5 to 3
    "max_cv_splits": 5,  # Maximum number of CV splits
    "min_cv_splits": 2,  # Minimum number of CV splits
    "adaptive_min_samples": True,  # Adapt minimum samples based on dataset size
    "merge_small_classes": True,  # Enable merging of small classes
    "min_samples_for_merge": 2,  # Minimum samples before merging is considered
}

class SafeExtractorWrapper(BaseEstimator, TransformerMixin):
    """
    Wrapper to ensure extractors always produce 2-dimensional output.
    
    This prevents the "Found array with dim 3. ElasticNet expected <= 2" error
    by enforcing that all extractor outputs are properly shaped for downstream models.
    """
    
    def __init__(self, extractor):
        self.extractor = extractor
        
    def fit(self, X, y=None):
        """Fit the wrapped extractor."""
        self.extractor.fit(X, y)
        return self
        
    def transform(self, X):
        """Transform with safe dimensionality checking."""
        X_transformed = self.extractor.transform(X)
        
        # Ensure output is always 2-dimensional
        if X_transformed.ndim > 2:
            # Flatten extra dimensions while preserving the sample dimension
            original_shape = X_transformed.shape
            n_samples = original_shape[0]
            n_features = np.prod(original_shape[1:])  # Flatten all feature dimensions
            
            X_transformed = X_transformed.reshape(n_samples, n_features)
            
            logger.debug(f"SafeExtractorWrapper: Reshaped {original_shape} -> {X_transformed.shape}")
            
        elif X_transformed.ndim < 2:
            # Add feature dimension if needed
            if X_transformed.ndim == 1:
                X_transformed = X_transformed.reshape(-1, 1)
                
        # Final validation
        if X_transformed.ndim != 2:
            raise ValueError(f"SafeExtractorWrapper: Could not reshape to 2D. "
                           f"Got shape {X_transformed.shape} with {X_transformed.ndim} dimensions")
                           
        return X_transformed
        
    def fit_transform(self, X, y=None):
        """Fit and transform with safe dimensionality checking."""
        return self.fit(X, y).transform(X)
        
    def get_params(self, deep=True):
        """Get parameters from wrapped extractor."""
        if deep:
            params = self.extractor.get_params(deep=True)
            # Prefix with extractor__ to avoid conflicts
            return {f"extractor__{k}": v for k, v in params.items()}
        else:
            return {'extractor': self.extractor}
            
    def set_params(self, **params):
        """Set parameters on wrapped extractor."""
        extractor_params = {}
        for key, value in params.items():
            if key.startswith('extractor__'):
                # Remove the extractor__ prefix
                actual_key = key[len('extractor__'):]
                extractor_params[actual_key] = value
            elif key == 'extractor':
                self.extractor = value
                
        if extractor_params:
            self.extractor.set_params(**extractor_params)
            
        return self

def safe_wrap_extractor(extractor_obj):
    """
    Safely wrap extractor with SafeExtractorWrapper if needed.
    
    Parameters
    ----------
    extractor_obj : object
        The extractor object to wrap
        
    Returns
    -------
    object
        Safely wrapped extractor
    """
    # Check if this is already a SafeExtractorWrapper
    if isinstance(extractor_obj, SafeExtractorWrapper):
        return extractor_obj
    
    # Wrap the extractor with safety checks
    return SafeExtractorWrapper(extractor_obj)

def _process_single_modality(
    modality_name: str, 
    modality_df: pd.DataFrame, 
    id_train: List[str], 
    id_val: List[str], 
    idx_test: np.ndarray, 
    y_train: np.ndarray, 
    extr_obj: Any, 
    ncomps: int, 
    idx_to_id: Dict[int, str], 
    fold_idx: Optional[int] = None,
    is_regression: bool = True,
    dataset_name: Optional[str] = None  # Added dataset name parameter
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Process a single modality's data through extraction pipeline.
    
    Parameters
    ----------
    modality_name : str
        Name of the modality (e.g., 'exp', 'mirna', 'methy', 'clinical')
    modality_df : pd.DataFrame
        Modality data
    id_train : List[str]
        Training IDs
    id_val : List[str]
        Validation IDs
    idx_test : np.ndarray
        Test indices
    y_train : np.ndarray
        Training target values
    extr_obj : Any
        Extractor object
    ncomps : int
        Number of components
    idx_to_id : Dict[int, str]
        Mapping from index to ID
    fold_idx : Optional[int]
        Fold index
    is_regression : bool
        Whether this is a regression task
    dataset_name : Optional[str]
        Dataset name for hyperparameter loading (e.g., 'AML', 'Breast')
        
    Returns
    -------
    Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]
        Training, validation, and test data arrays
    """
    try:
        # Since samples are pre-filtered, all id_train and id_val should be available
        available_sample_ids = set(modality_df.columns)
        
        # Verify all requested samples are available
        missing_train = [id_ for id_ in id_train if id_ not in available_sample_ids]
        missing_val = [id_ for id_ in id_val if id_ not in available_sample_ids]
        
        if missing_train or missing_val:
            logger.error(f"CRITICAL: Pre-filtering failed for {modality_name} fold {fold_idx}: missing train={missing_train}, missing val={missing_val}")
            return None, None, None
        
        # Get test sample IDs
        test_ids = [idx_to_id[idx] for idx in idx_test if idx in idx_to_id and idx_to_id[idx] in available_sample_ids]
        
        # Extract data in the EXACT same order as the IDs
        try:
            df_train = modality_df.loc[:, id_train].transpose()
            df_val = modality_df.loc[:, id_val].transpose() 
            df_test = modality_df.loc[:, test_ids].transpose() if test_ids else pd.DataFrame()
        except Exception as e:
            logger.error(f"CRITICAL: Failed to extract modality data for {modality_name} fold {fold_idx}: {str(e)}")
            return None, None, None
        
        # Validate perfect alignment
        if df_train.shape[0] != len(y_train):
            logger.error(f"CRITICAL: Sample alignment failed in {modality_name} fold {fold_idx}: df_train={df_train.shape[0]}, y_train={len(y_train)}")
            return None, None, None
        
        if df_train.shape[0] != len(id_train):
            logger.error(f"CRITICAL: ID alignment failed in {modality_name} fold {fold_idx}: df_train={df_train.shape[0]}, id_train={len(id_train)}")
            return None, None, None
        
        # CRITICAL: Check for any NaN or infinite values that could cause sample loss
        train_data = df_train.values
        val_data = df_val.values
        
        # Check for problematic values
        train_nan_mask = np.isnan(train_data).all(axis=1)  # Samples with all NaN features
        train_inf_mask = np.isinf(train_data).all(axis=1)  # Samples with all inf features
        
        if train_nan_mask.any() or train_inf_mask.any():
            problematic_samples = train_nan_mask | train_inf_mask
            n_problematic = problematic_samples.sum()
            logger.warning(f"Found {n_problematic} problematic samples in {modality_name} fold {fold_idx} that could cause issues")
            
            # Replace problematic values to prevent sample loss
            train_data = np.nan_to_num(train_data, nan=0.0, posinf=0.0, neginf=0.0)
            val_data = np.nan_to_num(val_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Update dataframes
            df_train = pd.DataFrame(train_data, index=df_train.index, columns=df_train.columns)
            df_val = pd.DataFrame(val_data, index=df_val.index, columns=df_val.columns)
        
        # Calculate maximum allowed components before calling the extractor
        max_allowed = min(df_train.shape)  # min(n_samples, n_features)
        req = ncomps
        
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        from sklearn.feature_selection import SelectorMixin
        from boruta import BorutaPy
        from models import (
            cached_fit_transform_selector_regression, transform_selector_regression,
            cached_fit_transform_selector_classification, transform_selector_classification,
            cached_fit_transform_extractor_classification, transform_extractor_classification
        )

        # Store original shapes for validation
        expected_train_samples = df_train.shape[0]
        expected_val_samples = df_val.shape[0]

        # If extr_obj is a selector or selector code string, use selector pipeline
        if isinstance(extr_obj, (SelectorMixin, BorutaPy, str)) or (isinstance(extr_obj, dict) and 'type' in extr_obj):
            # Use the passed is_regression parameter instead of inferring from data type
            if is_regression:
                # Regression selector
                selected_features, X_tr = cached_fit_transform_selector_regression(
                    extr_obj, df_train, y_train, req, fold_idx=fold_idx, ds_name=modality_name
                )
                X_va = transform_selector_regression(df_val, selected_features)
                X_te = transform_selector_regression(df_test, selected_features) if not df_test.empty else np.array([])
            else:
                # Classification selector
                selected_features, X_tr = cached_fit_transform_selector_classification(
                    df_train, y_train, extr_obj, req, ds_name=None, modality_name=modality_name, fold_idx=fold_idx
                )
                X_va = transform_selector_classification(df_val, selected_features)
                X_te = transform_selector_classification(df_test, selected_features) if not df_test.empty else np.array([])
                
            # CRITICAL: Validate that no samples were lost during selection
            if X_tr.shape[0] != expected_train_samples:
                logger.error(f"CRITICAL: Sample loss during selection in {modality_name} fold {fold_idx}: expected {expected_train_samples}, got {X_tr.shape[0]}")
                return None, None, None
            if X_va.shape[0] != expected_val_samples:
                logger.error(f"CRITICAL: Val sample loss during selection in {modality_name} fold {fold_idx}: expected {expected_val_samples}, got {X_va.shape[0]}")
                return None, None, None
                
            # Replace any remaining NaNs with zeros
            X_tr = np.nan_to_num(X_tr, nan=0.0)
            X_va = np.nan_to_num(X_va, nan=0.0)
            if X_te is not None and X_te.size > 0:
                X_te = np.nan_to_num(X_te, nan=0.0)
            logger.debug(f"Processed {modality_name} (selector) fold {fold_idx} - Train: {X_tr.shape}, Val: {X_va.shape}")
            return X_tr, X_va, X_te

        # For LDA, components are limited by the number of classes
        if isinstance(extr_obj, LDA):
            max_allowed = min(max_allowed, len(np.unique(y_train)) - 1)
            
        # Check if requested components exceed maximum allowed
        if req > max_allowed:
            logger.debug(f"{modality_name}: clipping n_components {req}->{max_allowed}")
            req = max_allowed
        
        # Extract features - use the passed is_regression parameter instead of inferring from data type
        if is_regression:
            try:
                from models import cached_fit_transform_extractor_regression, transform_extractor_regression
                extractor, X_tr = cached_fit_transform_extractor_regression(
                    df_train.values, y_train, extr_obj, req, 
                    ds_name=dataset_name or modality_name, fold_idx=fold_idx, modality_name=modality_name
                )
                
                # Check if extraction was successful
                if extractor is None or X_tr is None:
                    logger.debug(f"Extraction failed for {modality_name} fold {fold_idx}")
                    return None, None, None
                
                # CRITICAL: Validate that no samples were lost during extraction
                if X_tr.shape[0] != expected_train_samples:
                    logger.error(f"CRITICAL: Sample loss during extraction in {modality_name} fold {fold_idx}: expected {expected_train_samples}, got {X_tr.shape[0]}")
                    return None, None, None
                
                # Only transform validation and test if we have data
                if not df_val.empty:
                    X_va = transform_extractor_regression(df_val.values, extractor)
                    if X_va is None:
                        logger.debug(f"Failed to transform validation data for {modality_name} fold {fold_idx}, using zeros")
                        X_va = np.zeros((df_val.shape[0], X_tr.shape[1]), dtype=np.float64)
                    # Validate val samples
                    if X_va.shape[0] != expected_val_samples:
                        logger.error(f"CRITICAL: Val sample loss during extraction in {modality_name} fold {fold_idx}: expected {expected_val_samples}, got {X_va.shape[0]}")
                        return None, None, None
                else:
                    X_va = np.array([], dtype=np.float64).reshape(0, X_tr.shape[1] if X_tr is not None else 0)
                
                if not df_test.empty:
                    X_te = transform_extractor_regression(df_test.values, extractor)
                    if X_te is None:
                        logger.debug(f"Failed to transform test data for {modality_name} fold {fold_idx}, using zeros")
                        X_te = np.zeros((df_test.shape[0], X_tr.shape[1]), dtype=np.float64)
                else:
                    X_te = np.array([], dtype=np.float64).reshape(0, X_tr.shape[1] if X_tr is not None else 0)
                
                logger.debug(f"Extraction successful for {modality_name} fold {fold_idx}: {X_tr.shape} from {df_train.shape}")
            except Exception as e:
                logger.error(f"Error extracting features for {modality_name} fold {fold_idx}: {str(e)}")
                return None, None, None
        else:
            # Classification
            try:
                from models import cached_fit_transform_extractor_classification, transform_extractor_classification
                extractor, X_tr = cached_fit_transform_extractor_classification(
                    df_train.values, y_train, extr_obj, req, 
                    ds_name=dataset_name or modality_name, fold_idx=fold_idx, modality_name=modality_name
                )
                
                # Check if extraction was successful
                if extractor is None or X_tr is None:
                    logger.debug(f"Extraction failed for {modality_name} fold {fold_idx}")
                    return None, None, None
                
                # CRITICAL: Validate that no samples were lost during extraction
                if X_tr.shape[0] != expected_train_samples:
                    logger.error(f"CRITICAL: Sample loss during extraction in {modality_name} fold {fold_idx}: expected {expected_train_samples}, got {X_tr.shape[0]}")
                    return None, None, None
                
                # Only transform validation and test if we have data
                if not df_val.empty:
                    X_va = transform_extractor_classification(df_val.values, extractor)
                    if X_va is None:
                        logger.debug(f"Failed to transform validation data for {modality_name} fold {fold_idx}, using zeros")
                        X_va = np.zeros((df_val.shape[0], X_tr.shape[1]), dtype=np.float64)
                    # Validate val samples
                    if X_va.shape[0] != expected_val_samples:
                        logger.error(f"CRITICAL: Val sample loss during extraction in {modality_name} fold {fold_idx}: expected {expected_val_samples}, got {X_va.shape[0]}")
                        return None, None, None
                else:
                    X_va = np.array([], dtype=np.float64).reshape(0, X_tr.shape[1] if X_tr is not None else 0)
                
                if not df_test.empty:
                    X_te = transform_extractor_classification(df_test.values, extractor)
                    if X_te is None:
                        logger.debug(f"Failed to transform test data for {modality_name} fold {fold_idx}, using zeros")
                        X_te = np.zeros((df_test.shape[0], X_tr.shape[1]), dtype=np.float64)
                else:
                    X_te = np.array([], dtype=np.float64).reshape(0, X_tr.shape[1] if X_tr is not None else 0)
                
                logger.debug(f"Extraction successful for {modality_name} fold {fold_idx}: {X_tr.shape} from {df_train.shape}")
            except Exception as e:
                logger.error(f"Error extracting features for {modality_name} fold {fold_idx}: {str(e)}")
                return None, None, None
        
        # Replace any NaN values with zeros
        X_tr = np.nan_to_num(X_tr, nan=0.0)
        X_va = np.nan_to_num(X_va, nan=0.0)
        if X_te is not None and X_te.size > 0:
            X_te = np.nan_to_num(X_te, nan=0.0)
        
        # Final validation: ensure we return exactly the expected number of samples
        if X_tr.shape[0] != len(id_train):
            logger.warning(f"Output alignment mismatch in {modality_name} fold {fold_idx}: expected {len(id_train)}, got {X_tr.shape[0]}")
            # Try to fix by truncating or padding
            if X_tr.shape[0] > len(id_train):
                logger.warning(f"Truncating training data from {X_tr.shape[0]} to {len(id_train)} samples")
                X_tr = X_tr[:len(id_train)]
            elif X_tr.shape[0] < len(id_train):
                logger.warning(f"Training data has fewer samples than expected - this may indicate sample loss during processing")
                # Update the expected target vector to match
                y_train = y_train[:X_tr.shape[0]]
        
        if X_va.shape[0] != len(id_val):
            logger.warning(f"Output val alignment mismatch in {modality_name} fold {fold_idx}: expected {len(id_val)}, got {X_va.shape[0]}")
            # Try to fix by truncating or padding
            if X_va.shape[0] > len(id_val):
                logger.warning(f"Truncating validation data from {X_va.shape[0]} to {len(id_val)} samples")
                X_va = X_va[:len(id_val)]
            elif X_va.shape[0] < len(id_val):
                logger.warning(f"Validation data has fewer samples than expected - this may indicate sample loss during processing")
        
        # Log shape information
        logger.debug(f"Successfully processed {modality_name} fold {fold_idx} - Train: {X_tr.shape}, Val: {X_va.shape}, Test: {X_te.shape if X_te is not None and X_te.size > 0 else '(0,0)'}")
        
        # Return the processed data
        return X_tr, X_va, X_te
    except Exception as e:
        logger.error(f"Error processing {modality_name} fold {fold_idx}: {str(e)}")
        return None, None, None

def process_cv_fold(
    train_idx,
    val_idx,
    idx_temp,
    idx_test,
    y_temp,
    y_test,
    data_modalities,
    models,
    extr_obj,
    ncomps,
    id_to_idx,
    idx_to_id,
    all_ids,
    missing_percentage,
    fold_idx,
    base_out,
    ds_name,
    extr_name,
    pipeline_type,
    is_regression=True,
    make_plots=True,
    plot_prefix_override=None,
    integration_technique="attention_weighted"  # CURRENT IMPLEMENTATION: Use specified fusion strategy
):
    """
    Process a single CV fold.
    
    Parameters
    ----------
    train_idx : array-like
        Training indices
    val_idx : array-like
        Validation indices
    idx_temp : array-like
        Temporary indices
    idx_test : array-like
        Test indices
    y_temp : array-like
        Temporary target values
    y_test : array-like
        Test target values
    data_modalities : Dict[str, pd.DataFrame]
        Dictionary of modality DataFrames
    models : List[str]
        List of model names
    extr_obj : Any
        Extractor object
    ncomps : int
        Number of components
    id_to_idx : Dict[str, int]
        Mapping from ID to index
    idx_to_id : Dict[int, str]
        Mapping from index to ID
    all_ids : List[str]
        List of all IDs
    missing_percentage : float
        Missing percentage
    fold_idx : int
        Fold index
    base_out : str
        Base output directory
    ds_name : str
        Dataset name
    extr_name : str
        Extractor name
    pipeline_type : str
        Pipeline type
    is_regression : bool
        Whether this is regression
    make_plots : bool
        Whether to make plots
    plot_prefix_override : str, optional
        Plot prefix override
    integration_technique : str
        Integration technique to use for merging modalities
        
    Returns
    -------
    Various
        Results depend on regression vs classification
    """
    try:
        logger.info(f"Starting fold {fold_idx} for {ds_name} with {extr_name} and n_comps={ncomps}")
        
        # Convert indices to IDs
        id_train = np.array([idx_to_id[i] for i in train_idx if i in idx_to_id])
        id_val = np.array([idx_to_id[i] for i in val_idx if i in idx_to_id])
        
        # Get target values
        y_train = y_temp[train_idx]
        y_val = y_temp[val_idx]
        
        # Create Series for easier access by ID
        y_train_series = pd.Series(y_train, index=id_train)
        y_val_series = pd.Series(y_val, index=id_val)
        
        # Process modalities with missing data to simulate real-world scenarios
        modified_modalities = process_with_missing_modalities(
            data_modalities, all_ids, missing_percentage, 
            random_state=fold_idx, min_overlap_ratio=0.3
        )
        
        # Use our alignment function to get initial common IDs across modalities
        valid_train_ids, valid_val_ids = align_samples_to_modalities(
            id_train, id_val, modified_modalities
        )
        
        # Return empty if insufficient common samples
        if len(valid_train_ids) < 5 or len(valid_val_ids) < 2:
            logger.warning(f"Insufficient common samples across modalities in fold {fold_idx}")
            if is_regression:
                return {}, {}
            else:
                return {}, {}, {}, {}
        
        # CRITICAL FIX: Determine exact common samples BEFORE any processing
        # This ensures all modalities use exactly the same samples
        
        # First, find samples that are available in ALL modalities
        modality_samples = {}
        for name, df in modified_modalities.items():
            available_samples = set(df.columns)
            # For each modality, find which of our train/val samples are actually available
            modality_train_available = [id_ for id_ in valid_train_ids if id_ in available_samples]
            modality_val_available = [id_ for id_ in valid_val_ids if id_ in available_samples]
            modality_samples[name] = {
                'train': set(modality_train_available),
                'val': set(modality_val_available)
            }
        
        # Find the intersection of available samples across ALL modalities
        if len(modality_samples) == 0:
            logger.warning(f"No modalities available for fold {fold_idx}")
            return {}, {}
        
        # Start with the first modality's samples
        first_modality = list(modality_samples.keys())[0]
        final_common_train = modality_samples[first_modality]['train']
        final_common_val = modality_samples[first_modality]['val']
        
        # Intersect with all other modalities
        for name, samples in modality_samples.items():
            final_common_train = final_common_train.intersection(samples['train'])
            final_common_val = final_common_val.intersection(samples['val'])
        
        # Convert back to lists and sort for consistency
        final_common_train = sorted(list(final_common_train))
        final_common_val = sorted(list(final_common_val))
        
        # Check if we have enough samples after intersection
        if len(final_common_train) < MIN_SAMPLES_PER_FOLD or len(final_common_val) < 1:
            logger.warning(f"Insufficient common samples across modalities in fold {fold_idx}: train={len(final_common_train)}, val={len(final_common_val)}")
            return {}, {}
        
        # Filter original target values to match the final common samples BEFORE processing
        # Create target series from the initial valid samples
        train_mask = np.isin(id_train, valid_train_ids)
        val_mask = np.isin(id_val, valid_val_ids)
        filtered_y_train = y_train[train_mask]
        filtered_y_val = y_val[val_mask]
        
        y_train_series = pd.Series(filtered_y_train, index=valid_train_ids)
        y_val_series = pd.Series(filtered_y_val, index=valid_val_ids)
        
        try:
            final_aligned_y_train = y_train_series.reindex(final_common_train).values
            final_aligned_y_val = y_val_series.reindex(final_common_val).values
            
            if np.isnan(final_aligned_y_train).any() or np.isnan(final_aligned_y_val).any():
                logger.error(f"NaN values in pre-filtered target vectors for fold {fold_idx}")
                return {}, {}
                
            logger.debug(f"Pre-filtered targets for fold {fold_idx}: train={len(final_aligned_y_train)}, val={len(final_aligned_y_val)}")
        except Exception as e:
            logger.error(f"Failed to pre-filter target vectors for fold {fold_idx}: {str(e)}")
            return {}, {}
        
        # Apply target outlier removal for regression tasks (only on training data)
        if is_regression:
            try:
                original_train_size = len(final_aligned_y_train)
                
                # Remove extreme outliers (>97.5th percentile) from training data only
                outlier_threshold = np.percentile(final_aligned_y_train, 97.5)
                outlier_mask = final_aligned_y_train <= outlier_threshold
                
                # Count outliers for logging
                n_outliers = np.sum(~outlier_mask)
                outlier_percentage = (n_outliers / original_train_size) * 100
                
                if n_outliers > 0:
                    # Filter training data and corresponding sample IDs
                    final_aligned_y_train = final_aligned_y_train[outlier_mask]
                    final_common_train_filtered = [final_common_train[i] for i, keep in enumerate(outlier_mask) if keep]
                    
                    # Update the training sample list
                    final_common_train = final_common_train_filtered
                    
                    logger.info(f"Fold {fold_idx}: Removed {n_outliers} extreme outliers (>{outlier_threshold:.2f}) "
                               f"from training set ({outlier_percentage:.1f}% of training data)")
                    logger.info(f"Training set size: {original_train_size} → {len(final_aligned_y_train)}")
                    
                    # Ensure we still have enough training samples
                    if len(final_aligned_y_train) < MIN_SAMPLES_PER_FOLD:
                        logger.warning(f"Insufficient training samples after outlier removal in fold {fold_idx}: "
                                     f"{len(final_aligned_y_train)} < {MIN_SAMPLES_PER_FOLD}")
                        return {}, {}
                else:
                    logger.debug(f"Fold {fold_idx}: No extreme outliers detected in training targets")
                    
                # Validation data keeps all samples (including outliers)
                logger.debug(f"Validation set unchanged: {len(final_aligned_y_val)} samples "
                            f"(outliers preserved for unbiased evaluation)")
                
            except Exception as e:
                logger.warning(f"Target outlier removal failed for fold {fold_idx}: {e}")
                # Continue without outlier removal if it fails
        
        # CORRECTED PIPELINE ORDER: FUSION FIRST, THEN FEATURE PROCESSING
        # Step 1: Extract RAW modality data (no feature processing)
        logger.info(f"PIPELINE ORDER: Fusion → Feature Processing (extractors/selectors)")
        logger.debug(f"Extracting raw data from {len(modified_modalities)} modalities with {len(final_common_train)} train and {len(final_common_val)} val samples")
        
        raw_modality_train = []
        raw_modality_val = []
        raw_modality_test = []
        
        for modality_name, modality_df in modified_modalities.items():
            logger.debug(f"Extracting raw data from {modality_name}")
            
            # Align modality data to sample IDs (but no feature processing)
            try:
                # Get training samples - ensure ID alignment  
                train_mask = modality_df.columns.isin(final_common_train)
                X_train_raw = modality_df.loc[:, train_mask].T.values  # Transpose to samples x features
                
                # Get validation samples  
                val_mask = modality_df.columns.isin(final_common_val)
                X_val_raw = modality_df.loc[:, val_mask].T.values
                
                # Get test samples
                test_ids = [idx_to_id[i] for i in idx_test if i in idx_to_id]
                test_mask = modality_df.columns.isin(test_ids)
                X_test_raw = modality_df.loc[:, test_mask].T.values if test_ids else np.array([])
                
                # Ensure correct sample order
                if X_train_raw.shape[0] == len(final_common_train):
                    # Reorder to match exact sample order
                    train_cols = modality_df.columns[train_mask]
                    train_order = [list(train_cols).index(sid) for sid in final_common_train if sid in train_cols]
                    if len(train_order) == len(final_common_train):
                        X_train_raw = X_train_raw[train_order]
                
                if X_val_raw.shape[0] == len(final_common_val):
                    # Reorder to match exact sample order
                    val_cols = modality_df.columns[val_mask]
                    val_order = [list(val_cols).index(sid) for sid in final_common_val if sid in val_cols]
                    if len(val_order) == len(final_common_val):
                        X_val_raw = X_val_raw[val_order]
                
                # Basic validation
                if X_train_raw.shape[0] != len(final_common_train):
                    logger.error(f"Training sample count mismatch for {modality_name}: expected {len(final_common_train)}, got {X_train_raw.shape[0]}")
                    continue
                if X_val_raw.shape[0] != len(final_common_val):
                    logger.error(f"Validation sample count mismatch for {modality_name}: expected {len(final_common_val)}, got {X_val_raw.shape[0]}")
                    continue
                    
                if X_train_raw.size == 0 or X_val_raw.size == 0:
                    logger.warning(f"Empty data arrays for {modality_name}")
                    continue
                    
                raw_modality_train.append(X_train_raw)
                raw_modality_val.append(X_val_raw)
                raw_modality_test.append(X_test_raw)
                
                logger.debug(f"Raw {modality_name}: Train {X_train_raw.shape}, Val {X_val_raw.shape}")
                
            except Exception as e:
                logger.error(f"Error extracting raw data from {modality_name}: {e}")
                continue
        
        # Validate we have raw modality data
        if not raw_modality_train or not raw_modality_val:
            logger.error(f"CRITICAL: No raw modality data available in fold {fold_idx}")
            return {}, {}
        
        # Step 2: Apply FUSION to raw modalities FIRST (before feature processing)
        logger.info(f"Applying fusion strategy '{integration_technique}' to raw modalities")
        
        # Create a new imputer instance for this fold
        from fusion import ModalityImputer
        fold_imputer = ModalityImputer()

        # Merge raw modalities - apply fusion BEFORE feature processing
        try:
            # Handle strategies that return tuples (fitted fusion objects) specially
            if integration_technique in ["early_fusion_pca", "learnable_weighted", "mkl", "snf", "attention_weighted", "late_fusion_stacking"]:
                # For training data: fit and get the fusion object
                train_result = merge_modalities(*raw_modality_train, imputer=fold_imputer, is_train=True, strategy=integration_technique, n_components=ncomps, y=final_aligned_y_train, is_regression=is_regression)
                
                # Handle tuple return values
                if isinstance(train_result, tuple):
                    X_train_merged, fitted_fusion = train_result
                else:
                    X_train_merged = train_result
                    fitted_fusion = None
                
                # For validation data: use the fitted fusion object
                X_val_merged = merge_modalities(*raw_modality_val, imputer=fold_imputer, is_train=False, strategy=integration_technique, n_components=ncomps, fitted_fusion=fitted_fusion, y=final_aligned_y_val, is_regression=is_regression)
                
                # Handle tuple return for validation (should just be array)
                if isinstance(X_val_merged, tuple):
                    X_val_merged = X_val_merged[0]  # Take just the array part
            else:
                # For simple strategies that don't return tuples
                X_train_merged = merge_modalities(*raw_modality_train, imputer=fold_imputer, is_train=True, strategy=integration_technique, n_components=ncomps, y=final_aligned_y_train, is_regression=is_regression)
                X_val_merged = merge_modalities(*raw_modality_val, imputer=fold_imputer, is_train=False, strategy=integration_technique, n_components=ncomps, y=final_aligned_y_val, is_regression=is_regression)
                
                # Handle unexpected tuple returns for simple strategies
                if isinstance(X_train_merged, tuple):
                    X_train_merged = X_train_merged[0]
                if isinstance(X_val_merged, tuple):
                    X_val_merged = X_val_merged[0]
            
            # Handle potential sample truncation from merge_modalities
            # If merge_modalities truncated samples due to mismatches, we need to update target vectors
            if X_train_merged.shape[0] != len(final_aligned_y_train):
                logger.warning(f"Training X/y length mismatch after fusion: X={X_train_merged.shape[0]} vs y={len(final_aligned_y_train)}")
                logger.warning("Truncating target vector to match fused data...")
                final_aligned_y_train = final_aligned_y_train[:X_train_merged.shape[0]]
                logger.info(f"Training targets truncated to {len(final_aligned_y_train)} samples")
            
            if X_val_merged.shape[0] != len(final_aligned_y_val):
                logger.warning(f"Validation X/y length mismatch after fusion: X={X_val_merged.shape[0]} vs y={len(final_aligned_y_val)}")
                logger.warning("Truncating target vector to match fused data...")
                final_aligned_y_val = final_aligned_y_val[:X_val_merged.shape[0]]
                logger.info(f"Validation targets truncated to {len(final_aligned_y_val)} samples")
            
            # Skip if no valid data after fusion
            if X_train_merged.size == 0 or X_val_merged.size == 0:
                logger.warning(f"No valid data after fusion in fold {fold_idx}")
                return {}, {}
                
            # Skip if too few samples after fusion
            if X_train_merged.shape[0] < MIN_SAMPLES_PER_FOLD:
                logger.warning(f"Skipping fold {fold_idx} for {ds_name}: too few samples after fusion ({X_train_merged.shape[0]} < {MIN_SAMPLES_PER_FOLD})")
                return {}, {}
                
            logger.debug(f"Fusion completed in fold {fold_idx}: Train {X_train_merged.shape}, Val {X_val_merged.shape}")
            
            # Update the aligned target vectors for the rest of the processing
            aligned_y_train = final_aligned_y_train
            aligned_y_val = final_aligned_y_val
            
        except Exception as e:
            logger.error(f"Error during fusion in fold {fold_idx}: {str(e)}")
            return {}, {}
        
        # Step 3: Apply feature processing (extractors/selectors) to FUSED data
        logger.info(f"Applying feature processing ({pipeline_type}) to fused data")
        
        # Save the number of features before extraction/selection
        original_n_features = ncomps  # This ensures n_features matches the intended value in metrics

        # Apply extraction/selection to FUSED data (correct order)
        # --- ENFORCE ncomps after reduction ---
        if is_regression:
            # Extraction pipeline
            if pipeline_type == "extraction":
                from models import cached_fit_transform_extractor_regression, transform_extractor_regression
                extractor, X_train_reduced = cached_fit_transform_extractor_regression(
                    X_train_merged, aligned_y_train, extr_obj, ncomps, ds_name=ds_name, modality_name=None, fold_idx=fold_idx
                )
                X_val_reduced = transform_extractor_regression(X_val_merged, extractor)
                # ENFORCE ncomps
                if X_train_reduced is not None and X_train_reduced.shape[1] > ncomps:
                    X_train_reduced = X_train_reduced[:, :ncomps]
                if X_val_reduced is not None and X_val_reduced.shape[1] > ncomps:
                    X_val_reduced = X_val_reduced[:, :ncomps]
                train_n_components = X_train_reduced.shape[1] if X_train_reduced is not None else -1
                final_X_train, final_X_val = X_train_reduced, X_val_reduced
            else:
                # Selection pipeline
                from models import cached_fit_transform_selector_regression, transform_selector_regression
                selected_features, X_train_reduced = cached_fit_transform_selector_regression(
                    extr_obj, X_train_merged, aligned_y_train, ncomps, fold_idx=fold_idx, ds_name=ds_name
                )
                X_val_reduced = transform_selector_regression(X_val_merged, selected_features)
                # ENFORCE ncomps
                if X_train_reduced is not None and X_train_reduced.shape[1] > ncomps:
                    X_train_reduced = X_train_reduced[:, :ncomps]
                if X_val_reduced is not None and X_val_reduced.shape[1] > ncomps:
                    X_val_reduced = X_val_reduced[:, :ncomps]
                train_n_components = X_train_reduced.shape[1] if X_train_reduced is not None else -1
                final_X_train, final_X_val = X_train_reduced, X_val_reduced
        else:
            # Classification
            if pipeline_type == "extraction":
                from models import cached_fit_transform_extractor_classification, transform_extractor_classification
                extractor, X_train_reduced = cached_fit_transform_extractor_classification(
                    X_train_merged, aligned_y_train, extr_obj, ncomps, ds_name=ds_name, modality_name=None, fold_idx=fold_idx
                )
                X_val_reduced = transform_extractor_classification(X_val_merged, extractor)
                # ENFORCE ncomps
                if X_train_reduced is not None and X_train_reduced.shape[1] > ncomps:
                    X_train_reduced = X_train_reduced[:, :ncomps]
                if X_val_reduced is not None and X_val_reduced.shape[1] > ncomps:
                    X_val_reduced = X_val_reduced[:, :ncomps]
                train_n_components = X_train_reduced.shape[1] if X_train_reduced is not None else -1
                final_X_train, final_X_val = X_train_reduced, X_val_reduced
            else:
                # Selection pipeline
                from models import cached_fit_transform_selector_classification, transform_selector_classification
                selected_features, X_train_reduced = cached_fit_transform_selector_classification(
                    X_train_merged, aligned_y_train, extr_obj, ncomps, ds_name=ds_name, modality_name=None, fold_idx=fold_idx
                )
                X_val_reduced = transform_selector_classification(X_val_merged, selected_features)
                # ENFORCE ncomps
                if X_train_reduced is not None and X_train_reduced.shape[1] > ncomps:
                    X_train_reduced = X_train_reduced[:, :ncomps]
                if X_val_reduced is not None and X_val_reduced.shape[1] > ncomps:
                    X_val_reduced = X_val_reduced[:, :ncomps]
                train_n_components = X_train_reduced.shape[1] if X_train_reduced is not None else -1
                final_X_train, final_X_val = X_train_reduced, X_val_reduced

        # Now pass n_features and train_n_components to the model training functions
        model_results = {}
        model_objects = {}
        model_yvals = {}
        model_ypreds = {}
        
        # Import the correct function based on the task type
        if is_regression:
            train_model = train_regression_model
        else:
            train_model = train_classification_model
            
        for model_name in models:
            try:
                # Apply robust data synchronization and NaN guarding before model training
                from models import synchronize_X_y_data, guard_against_target_nans
                
                # Guard against NaN values in targets and synchronize data
                final_X_train, final_y_train = guard_against_target_nans(
                    final_X_train, aligned_y_train, 
                    operation_name=f"training {model_name} (fold {fold_idx})"
                )
                final_X_val, final_y_val = guard_against_target_nans(
                    final_X_val, aligned_y_val, 
                    operation_name=f"validation {model_name} (fold {fold_idx})"
                )
                
                # Synchronize X and y data to ensure perfect alignment
                final_X_train, final_y_train = synchronize_X_y_data(
                    final_X_train, final_y_train, 
                    operation_name=f"training {model_name} (fold {fold_idx})"
                )
                final_X_val, final_y_val = synchronize_X_y_data(
                    final_X_val, final_y_val, 
                    operation_name=f"validation {model_name} (fold {fold_idx})"
                )
                
                # Only proceed if we have valid data (check before trying to get lengths)
                if (final_X_train is None or final_y_train is None or 
                    final_X_val is None or final_y_val is None):
                    logger.warning(f"Invalid data for {model_name} in fold {fold_idx}")
                    continue
                
                # Final hard check for alignment - this should never fail now
                if len(final_X_train) != len(final_y_train):
                    raise ValueError(f"Training X/y length mismatch after synchronization: {len(final_X_train)} vs {len(final_y_train)}")
                if len(final_X_val) != len(final_y_val):
                    raise ValueError(f"Validation X/y length mismatch after synchronization: {len(final_X_val)} vs {len(final_y_val)}")
                
                # For classification, check class distribution in this specific fold
                if not is_regression:
                    # Use adaptive minimum samples based on dataset size
                    if CV_CONFIG["adaptive_min_samples"]:
                        total_samples = len(final_y_train) + len(final_y_val)
                        if total_samples < 6:
                            adaptive_min_samples = 1  # Very small datasets
                        elif total_samples < 10:
                            adaptive_min_samples = 1  # Small datasets
                        else:
                            adaptive_min_samples = 2  # Normal datasets
                    else:
                        adaptive_min_samples = CV_CONFIG["min_samples_per_class_per_fold"]
                    
                    # Ensure class consistency between training and validation sets
                    final_train_classes = set(np.unique(final_y_train))
                    final_val_classes = set(np.unique(final_y_val))
                    
                    # Handle class mismatches BEFORE merging small classes
                    if final_train_classes != final_val_classes:
                        missing_in_val = final_train_classes - final_val_classes
                        extra_in_val = final_val_classes - final_train_classes
                        
                        if missing_in_val:
                            logger.debug(f"Fold {fold_idx}: Validation missing classes: {missing_in_val}")
                            # Remove training samples with classes not in validation
                            train_mask = np.isin(final_y_train, list(final_val_classes))
                            if not train_mask.all():
                                logger.debug(f"Fold {fold_idx}: Removing {(~train_mask).sum()} training samples with classes missing in validation")
                                final_X_train = final_X_train[train_mask]
                                final_y_train = final_y_train[train_mask]
                        
                        if extra_in_val:
                            logger.debug(f"Fold {fold_idx}: Mapping extra validation classes {extra_in_val} to training classes")
                            train_classes_list = sorted(list(final_train_classes))
                            for extra_class in extra_in_val:
                                nearest_class = min(train_classes_list, key=lambda x: abs(x - extra_class))
                                final_y_val = np.where(final_y_val == extra_class, nearest_class, final_y_val)
                        
                        # Remove validation samples with classes not in training (after mapping)
                        valid_mask = np.isin(final_y_val, list(final_train_classes))
                        if not valid_mask.all():
                            logger.debug(f"Fold {fold_idx}: Removing {(~valid_mask).sum()} validation samples with unmatched classes")
                            final_X_val = final_X_val[valid_mask]
                            final_y_val = final_y_val[valid_mask]
                    
                    # Check if we need to merge classes in this fold
                    if CV_CONFIG["merge_small_classes"]:
                        # Combine training and validation data to ensure consistent class merging
                        combined_y = np.concatenate([final_y_train, final_y_val])
                        unique_combined, counts_combined = np.unique(combined_y, return_counts=True)
                        min_combined_count = np.min(counts_combined)
                        
                        if min_combined_count < adaptive_min_samples:
                            logger.info(f"Fold {fold_idx}: Merging small classes (min_count={min_combined_count} < {adaptive_min_samples})")
                            
                            # Apply merging to combined data to ensure consistency
                            combined_y_merged, class_mapping = merge_small_classes(combined_y, adaptive_min_samples)
                            
                            # Split back into training and validation with consistent class structure
                            n_train = len(final_y_train)
                            final_y_train = combined_y_merged[:n_train]
                            final_y_val = combined_y_merged[n_train:]
                            
                            # Log the class distribution after merging
                            unique_train, counts_train = np.unique(final_y_train, return_counts=True)
                            unique_val, counts_val = np.unique(final_y_val, return_counts=True)
                            logger.info(f"Fold {fold_idx} after merging - Train classes: {dict(zip(unique_train, counts_train))}")
                            logger.info(f"Fold {fold_idx} after merging - Val classes: {dict(zip(unique_val, counts_val))}")
                    
                    # Final verification of class consistency
                    final_train_classes = set(np.unique(final_y_train))
                    final_val_classes = set(np.unique(final_y_val))
                    
                    if final_train_classes != final_val_classes:
                        logger.debug(f"Fold {fold_idx}: Final class mismatch - Train: {final_train_classes}, Val: {final_val_classes}")
                        # This should be very rare now, but handle it as a fallback
                        valid_mask = np.isin(final_y_val, list(final_train_classes))
                        if not valid_mask.all():
                            logger.debug(f"Fold {fold_idx}: Final cleanup - removing {(~valid_mask).sum()} validation samples")
                            final_X_val = final_X_val[valid_mask]
                            final_y_val = final_y_val[valid_mask]
                    
                    # Additional validation after merging
                    if len(np.unique(final_y_train)) < 2:
                        logger.warning(f"Skipping {model_name} in fold {fold_idx}: insufficient classes after merging ({len(np.unique(final_y_train))} classes)")
                        continue
                    
                    if final_X_train.shape[0] < 2:
                        logger.warning(f"Skipping {model_name} in fold {fold_idx}: insufficient training samples after merging ({final_X_train.shape[0]} samples)")
                        continue
                    
                    logger.debug(f"Processed data for {model_name} in fold {fold_idx}: Train {final_X_train.shape}, Val {final_X_val.shape}, Classes: {len(np.unique(final_y_train))}")
                
                # Train and evaluate
                if is_regression:
                    if plot_prefix_override:
                        current_plot_prefix = plot_prefix_override
                    else:
                        current_plot_prefix = f"{ds_name}_fold_{fold_idx}_{extr_name}_{ncomps}_{model_name}_{missing_percentage}_{integration_technique}"
                    model, metrics = train_model(
                        final_X_train, final_y_train, 
                        final_X_val, final_y_val,
                        model_name, 
                        out_dir=os.path.join(base_out, "plots"),
                        plot_prefix=current_plot_prefix,
                        fold_idx=fold_idx,
                        make_plots=make_plots,
                        n_features=original_n_features,
                        train_n_components=train_n_components,
                        extractor_name=extr_name,
                        dataset_name=ds_name  # Added for hyperparameter loading
                    )
                    model_results[model_name] = metrics
                    model_objects[model_name] = model
                else:
                    if plot_prefix_override:
                        current_plot_prefix = plot_prefix_override
                    else:
                        current_plot_prefix = f"{ds_name}_fold_{fold_idx}_{extr_name}_{ncomps}_{model_name}_{missing_percentage}_{integration_technique}"
                    model, metrics, y_val_out, y_pred_out = train_model(
                        final_X_train, final_y_train, 
                        final_X_val, final_y_val,
                        model_name, 
                        out_dir=os.path.join(base_out, "plots"),
                        plot_prefix=current_plot_prefix,
                        fold_idx=fold_idx,
                        make_plots=make_plots,
                        n_features=original_n_features,
                        train_n_components=train_n_components,
                        extractor_name=extr_name,
                        dataset_name=ds_name  # Added for hyperparameter loading
                    )
                    model_results[model_name] = metrics
                    model_objects[model_name] = model
                    model_yvals[model_name] = y_val_out
                    model_ypreds[model_name] = y_pred_out
            except Exception as e:
                logger.warning(f"Warning: Failed to train {model_name} in fold {fold_idx}: {str(e)}")
                continue
                
        # Check if any models were successfully trained
        if not model_results:
            logger.warning(f"No models were successfully trained in fold {fold_idx} for {ds_name}")
            if not is_regression:
                logger.warning(f"This may be due to insufficient class distribution in the fold")
        
        if is_regression:
            return model_results, model_objects
        else:
            return model_results, model_objects, model_yvals, model_ypreds
    except Exception as e:
        logger.warning(f"Warning: Error processing fold {fold_idx}: {str(e)}")
        if is_regression:
            return {}, {}
        else:
            return {}, {}, {}, {}

def run_extraction_pipeline(
    ds_name: str, 
    data_modalities: Dict[str, pd.DataFrame], 
    common_ids: List[str], 
    y: np.ndarray, 
    base_out: str,
    extractors: Dict[str, Any], 
    n_comps_list: List[int], 
    models: List[str],
    progress_count: List[int], 
    total_runs: int,
    is_regression: bool = True
):
    """
    Run extraction pipeline for a dataset with optimal n_components from hyperparameters.
    
    Parameters
    ----------
    ds_name : str
        Dataset name
    data_modalities : Dict[str, pd.DataFrame]
        Dictionary of modality DataFrames
    common_ids : List[str]
        List of common sample IDs
    y : np.ndarray
        Target values
    base_out : str
        Base output directory
    extractors : Dict[str, Any]
        Dictionary of extractors
    n_comps_list : List[int]
        List of n_components values (ignored for extraction, uses hyperparameters instead)
    models : List[str]
        List of model names to train
    progress_count : List[int]
        Progress counter [0]
    total_runs : int
        Total number of runs
    is_regression : bool
        Whether this is a regression task
    """
    from models import get_extraction_n_components_list
    
    # Get optimal n_components for each extractor from hyperparameters
    task = "reg" if is_regression else "clf"
    optimal_n_components = get_extraction_n_components_list(ds_name, extractors, task)
    
    logger.info(f"Using optimal n_components from hyperparameters for {ds_name}: {optimal_n_components}")
    
    _run_pipeline(
        ds_name=ds_name, 
        data_modalities=data_modalities, 
        common_ids=common_ids, 
        y=y, 
        base_out=base_out,
        transformers=extractors, 
        n_trans_list=optimal_n_components,  # Use extractor-specific optimal values
        models=models,
        progress_count=progress_count, 
        total_runs=total_runs,
        is_regression=is_regression, 
        pipeline_type="extraction"
    )

def run_selection_pipeline(
    ds_name: str, 
    data_modalities: Dict[str, pd.DataFrame], 
    common_ids: List[str], 
    y: np.ndarray, 
    base_out: str,
    selectors: Dict[str, str], 
    n_feats_list: List[int], 
    models: List[str],
    progress_count: List[int], 
    total_runs: int,
    is_regression: bool = True
):
    """
    Run selection pipeline for a dataset using specified n_features values [8, 16, 32].
    
    Parameters
    ----------
    ds_name : str
        Dataset name
    data_modalities : Dict[str, pd.DataFrame]
        Dictionary of modality DataFrames
    common_ids : List[str]
        List of common sample IDs
    y : np.ndarray
        Target values
    base_out : str
        Base output directory
    selectors : Dict[str, str]
        Dictionary of selectors
    n_feats_list : List[int]
        List of n_features values to test (typically [8, 16, 32])
    models : List[str]
        List of model names to train
    progress_count : List[int]
        Progress counter [0]
    total_runs : int
        Total number of runs
    is_regression : bool
        Whether this is a regression task
    """
    logger.info(f"Using specified n_features values for {ds_name} selection: {n_feats_list}")
    
    _run_pipeline(
        ds_name=ds_name, 
        data_modalities=data_modalities, 
        common_ids=common_ids, 
        y=y, 
        base_out=base_out,
        transformers=selectors, 
        n_trans_list=n_feats_list,  # Use specified n_features list [8, 16, 32]
        models=models,
        progress_count=progress_count, 
        total_runs=total_runs,
        is_regression=is_regression, 
        pipeline_type="selection"
    )

def train_regression_model(X_train, y_train, X_val, y_val, model_name, out_dir, plot_prefix, fold_idx=None, make_plots=True, n_features=None, train_n_components=None, extractor_name=None, dataset_name="unknown"):
    """Train regression model and evaluate it."""
    from models import get_model_object
    from plots import plot_regression_scatter, plot_regression_residuals, plot_feature_importance
    import os
    import numpy as np
    import time
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Extract dataset name from plot_prefix for logging
    dataset_name = plot_prefix.split('_')[0] if '_' in plot_prefix else "unknown"
    
    # Data should already be aligned by process_cv_fold, so just do basic validation
    if X_train is None or y_train is None or X_val is None or y_val is None:
        logger.warning(f"Invalid data for {model_name} in fold {fold_idx}")
        log_model_training_info(model_name, dataset_name, fold_idx, 0, 0, success=False, error_msg="Invalid input data")
        return None, {}
    
    # FIX C: Final X/y alignment guard just before model training
    # This stops the pipeline immediately if any future bug slips through
    if hasattr(X_train, 'index') and hasattr(y_train, 'index'):
        common_train = X_train.index.intersection(y_train.index)
        X_train = X_train.loc[common_train]
        y_train = y_train.loc[common_train]
    if hasattr(X_val, 'index') and hasattr(y_val, 'index'):
        common_val = X_val.index.intersection(y_val.index)
        X_val = X_val.loc[common_val]
        y_val = y_val.loc[common_val]
    
    # Final alignment check
    if len(X_train) != len(y_train):
        raise RuntimeError(f"X/y mismatch even after alignment: {len(X_train)} vs {len(y_train)}")
    if len(X_val) != len(y_val):
        raise RuntimeError(f"X_val/y_val mismatch even after alignment: {len(X_val)} vs {len(y_val)}")
    
    # IMPROVEMENT 5: Cross-Validation Target Validation
    try:
        CrossValidationTargetValidator.assert_cv_data_integrity(
            X_train, y_train, X_val, y_val, fold_idx, dataset_name
        )
        # Log detailed target validation for regression
        target_validation = CrossValidationTargetValidator.validate_cv_split_targets(
            X_train, y_train, X_val, y_val, fold_idx, dataset_name
        )
        if not target_validation['is_valid']:
            logger.warning(f"CV target validation warnings for fold {fold_idx}: {target_validation['warnings']}")
    except ValueError as e:
        logger.error(f"Critical CV target validation failed for fold {fold_idx}: {e}")
        log_model_training_info(model_name, dataset_name, fold_idx, X_train.shape[0], X_val.shape[0], success=False, error_msg=f"CV target validation failed: {e}")
        return None, {}
    except Exception as e:
        logger.warning(f"CV target validation error for fold {fold_idx}: {e}")
    
    # Basic sanity check without redundant alignment
    if X_train.shape[0] != len(y_train):
        logger.error(f"Data alignment error for {model_name} in fold {fold_idx}: X_train={X_train.shape[0]}, y_train={len(y_train)}")
        log_model_training_info(model_name, dataset_name, fold_idx, X_train.shape[0], 0, success=False, error_msg="Training data alignment error")
        return None, {}
    
    if X_val.shape[0] != len(y_val):
        logger.error(f"Data alignment error for {model_name} in fold {fold_idx}: X_val={X_val.shape[0]}, y_val={len(y_val)}")
        log_model_training_info(model_name, dataset_name, fold_idx, X_train.shape[0], X_val.shape[0], success=False, error_msg="Validation data alignment error")
        return None, {}

    # Log training start
    logger.debug(f"[MODEL_TRAINING] {dataset_name} - {model_name} (fold {fold_idx}) - Starting training with {X_train.shape[0]} train, {X_val.shape[0]} val samples")

    try:
        # Create the model (with early stopping enabled by default)
        model = get_model_object(model_name)
        
        # Load and apply tuned hyperparameters
        if extractor_name:
            best_params = load_best(dataset_name, extractor_name, model_name, "reg")
            if best_params:
                # Filter parameters for the model only (remove extractor params)
                model_params = {k.replace('model__', ''): v for k, v in best_params.items() 
                               if k.startswith('model__')}
                if model_params:
                    model.set_params(**model_params)
                    logger.info(f"Applied tuned hyperparameters for {dataset_name}_{extractor_name}_{model_name}: {model_params}")
                else:
                    logger.debug(f"No model hyperparameters found for {dataset_name}_{extractor_name}_{model_name}")
            else:
                logger.debug(f"No tuned hyperparameters found for {dataset_name}_{extractor_name}_{model_name}, using defaults")
        
        # Train the model with timing
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        
        # Check if fallback strategy was used
        fallback_used = False
        if hasattr(model, '_fallback_used'):
            fallback_used = model._fallback_used
        
        # Get early stopping information if available
        early_stopping_info = {}
        if hasattr(model, 'best_score_'):
            early_stopping_info = {
                'early_stopping_used': True,
                'best_validation_score': model.best_score_,
                'stopped_epoch': model.stopped_epoch_ or 'N/A',
                'early_stopping_history': model.history_ if hasattr(model, 'history_') else [],
                'patience_used': model.wait_ if hasattr(model, 'wait_') else None
            }
            logger.info(f"Early stopping for {model_name} (fold {fold_idx}): best score={model.best_score_:.4f}, stopped at epoch {model.stopped_epoch_ or 'N/A'}")
            logger.debug(f"[MODEL_TRAINING] {dataset_name} - {model_name} (fold {fold_idx}) - Early stopping used, best score: {model.best_score_:.4f}")
        else:
            early_stopping_info = {
                'early_stopping_used': False
            }
        
        # Make predictions with validation
        try:
            y_pred = model.predict(X_val)
            
            # Validate predictions are finite
            if not np.all(np.isfinite(y_pred)):
                logger.warning(f"Model {model_name} (fold {fold_idx}) produced non-finite predictions, cleaning...")
                y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=np.median(y_val), neginf=np.median(y_val))
            
            # Check for reasonable prediction variance
            if np.std(y_pred) == 0:
                logger.warning(f"Model {model_name} (fold {fold_idx}) produced constant predictions")
        
        except Exception as e:
            logger.error(f"Prediction failed for {model_name} (fold {fold_idx}): {str(e)}")
            # Create fallback predictions
            y_pred = np.full(len(y_val), np.median(y_val))
            fallback_used = True
        
        # Calculate metrics with error handling
        try:
            mse = mean_squared_error(y_val, y_pred)
            rmse = np.sqrt(mse) if mse >= 0 else 0.0
            mae = mean_absolute_error(y_val, y_pred)
            
            # Safe R² calculation using centralized safe scorer
            r2 = safe_r2_score(y_val, y_pred)
                
        except Exception as e:
            logger.warning(f"Metric calculation failed for {model_name} (fold {fold_idx}): {str(e)}")
            mse, rmse, mae, r2 = 999.0, 999.0, 999.0, -999.0
            fallback_used = True
        
        # Log successful training
        log_model_training_info(model_name, dataset_name, fold_idx, X_train.shape[0], X_val.shape[0], 
                               success=True, fallback=fallback_used)
        logger.debug(f"[MODEL_TRAINING] {dataset_name} - {model_name} (fold {fold_idx}) - Metrics: R2={r2:.4f}, RMSE={rmse:.4f}")
        
        # Use passed train_n_components if provided, else fallback to X_train.shape[1]
        if train_n_components is None:
            train_n_components = X_train.shape[1]
        
        # Create plots
        plot_success_count = 0
        plot_total_count = 0
        if out_dir and make_plots:
            os.makedirs(out_dir, exist_ok=True)
            
            # Scatter plot
            plot_total_count += 1
            scatter_path = os.path.join(out_dir, f"{plot_prefix}_scatter.png")
            if plot_regression_scatter(y_val, y_pred, f"{model_name} Scatter", scatter_path):
                plot_success_count += 1
                log_plot_save_info(dataset_name, "regression_scatter", scatter_path, success=True)
            else:
                log_plot_save_info(dataset_name, "regression_scatter", scatter_path, success=False)
            
            # Residuals plot
            plot_total_count += 1
            residuals_path = os.path.join(out_dir, f"{plot_prefix}_residuals.png")
            if plot_regression_residuals(y_val, y_pred, f"{model_name} Residuals", residuals_path):
                plot_success_count += 1
                log_plot_save_info(dataset_name, "regression_residuals", residuals_path, success=True)
            else:
                log_plot_save_info(dataset_name, "regression_residuals", residuals_path, success=False)
            
            # Feature importance plot
            if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
                plot_total_count += 1
                if hasattr(X_train, 'columns'):
                    feat_names = list(X_train.columns)
                else:
                    feat_names = [f"Feature {i}" for i in range(X_train.shape[1])]
                featimp_path = os.path.join(out_dir, f"{plot_prefix}_featimp.png")
                if plot_feature_importance(model, feat_names, f"{model_name} Feature Importance", featimp_path):
                    plot_success_count += 1
                    log_plot_save_info(dataset_name, "feature_importance", featimp_path, success=True)
                else:
                    log_plot_save_info(dataset_name, "feature_importance", featimp_path, success=False)
        
        # Log plot creation summary
        if make_plots and out_dir:
            logger.debug(f"[PLOT_SAVE] {dataset_name} - {model_name} (fold {fold_idx}) - Created {plot_success_count}/{plot_total_count} plots successfully")
        
        # Return model and metrics (including early stopping info)
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'train_time': train_time,
            'n_features': n_features if n_features is not None else -1,  # Original feature count or -1 if unknown
            'train_n_components': train_n_components,  # Actual feature count used in training
            **early_stopping_info  # Include early stopping metrics
        }
        
        return model, metrics
        
    except Exception as e:
        # Log training failure
        error_msg = str(e)
        log_model_training_info(model_name, dataset_name, fold_idx, X_train.shape[0], X_val.shape[0], 
                               success=False, error_msg=error_msg)
        logger.error(f"[MODEL_TRAINING] {dataset_name} - {model_name} (fold {fold_idx}) - Training failed: {error_msg}")
        import traceback
        logger.debug(f"[MODEL_TRAINING] {dataset_name} - {model_name} (fold {fold_idx}) - Traceback:\n{traceback.format_exc()}")
        return None, {}

def train_classification_model(X_train, y_train, X_val, y_val, model_name, out_dir, plot_prefix, fold_idx=None, make_plots=True, n_features=None, train_n_components=None, extractor_name=None, dataset_name="unknown"):
    """
    Train a classification model with proper handling of class counts.
    """
    import os
    import time
    import numpy as np
    from sklearn.metrics import accuracy_score, roc_auc_score
    from plots import plot_confusion_matrix, plot_roc_curve_binary, plot_roc_curve_multiclass
    
    # Extract dataset name from plot_prefix for logging
    dataset_name = plot_prefix.split('_')[0] if '_' in plot_prefix else "unknown"
    
    try:
        # IMPROVEMENT 5: Cross-Validation Target Validation for Classification
        try:
            CrossValidationTargetValidator.assert_cv_data_integrity(
                X_train, y_train, X_val, y_val, fold_idx, dataset_name
            )
            # Log detailed target validation for classification
            target_validation = CrossValidationTargetValidator.validate_cv_split_targets(
                X_train, y_train, X_val, y_val, fold_idx, dataset_name
            )
            if not target_validation['is_valid']:
                logger.warning(f"CV target validation warnings for fold {fold_idx}: {target_validation['warnings']}")
        except ValueError as e:
            logger.error(f"Critical CV target validation failed for fold {fold_idx}: {e}")
            return None, {}
        except Exception as e:
            logger.warning(f"CV target validation error for fold {fold_idx}: {e}")
        
        # Get the number of unique classes in the training data
        n_classes = len(np.unique(y_train))
        logger.info(f"Training data has {n_classes} unique classes")
        
        # Initialize model with correct number of classes
        if model_name == 'LogisticRegression':
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(max_iter=1000, solver='lbfgs')  # Removed multi_class parameter
        elif model_name == 'RandomForestClassifier':  # Fixed model name
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_name == 'SVC':  # Fixed model name
            from sklearn.svm import SVC
            model = SVC(probability=True, random_state=42)
        else:
            # Try to get model from models.py
            try:
                from models import get_model_object
                model = get_model_object(model_name)
                logger.info(f"Successfully created model {model_name} using get_model_object")
            except Exception as e:
                logger.error(f"Failed to create model {model_name}: {str(e)}")
                raise ValueError(f"Unknown model name: {model_name}")
        
        # Load and apply tuned hyperparameters
        if extractor_name:
            best_params = load_best(dataset_name, extractor_name, model_name, "clf")
            if best_params:
                # Filter parameters for the model only (remove extractor params)
                model_params = {k.replace('model__', ''): v for k, v in best_params.items() 
                               if k.startswith('model__')}
                if model_params:
                    model.set_params(**model_params)
                    logger.info(f"Applied tuned hyperparameters for {dataset_name}_{extractor_name}_{model_name}: {model_params}")
                else:
                    logger.debug(f"No model hyperparameters found for {dataset_name}_{extractor_name}_{model_name}")
            else:
                logger.debug(f"No tuned hyperparameters found for {dataset_name}_{extractor_name}_{model_name}, using defaults")
            
        # If using PCA, ensure we preserve enough components for all classes
        if train_n_components is not None:
            # Calculate the maximum possible components based on PCA constraints
            # PCA can use at most min(n_samples, n_features) components
            max_possible_components = min(X_train.shape[0], X_train.shape[1])
            
            # Calculate the ideal components (at least n_classes - 1 for LDA compatibility)
            ideal_components = max(n_classes - 1, train_n_components)
            
            # Use the minimum of ideal and possible components
            min_components = min(ideal_components, max_possible_components)
            
            # Only log debug message if we had to reduce from the ideal
            if ideal_components > max_possible_components:
                constraint_type = "samples" if X_train.shape[0] < X_train.shape[1] else "features"
                logger.debug(f"Adjusted PCA components from {ideal_components} to {min_components} (limited by {max_possible_components} available {constraint_type})")
            
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min_components)
            X_train = pca.fit_transform(X_train)
            X_val = pca.transform(X_val)
            logger.info(f"Applied PCA with {min_components} components")
            
        # Fit the model with timing
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Verify the model has the correct number of classes
        if hasattr(model, 'n_classes_'):
            if model.n_classes_ != n_classes:
                logger.warning(f"Model was trained with {model.n_classes_} classes but data has {n_classes} classes")
                # Retrain the model with correct number of classes
                model.fit(X_train, y_train)
                if model.n_classes_ != n_classes:
                    raise ValueError(f"Failed to train model with correct number of classes. Expected {n_classes}, got {model.n_classes_}")
        
        # Make predictions with validation
        try:
            y_pred = model.predict(X_val)
            
            # Validate predictions
            if not np.all(np.isfinite(y_pred)):
                logger.warning(f"Model {model_name} (fold {fold_idx}) produced non-finite predictions, using fallback...")
                # Use most frequent class as fallback
                from scipy.stats import mode
                fallback_class = mode(y_train)[0][0] if len(y_train) > 0 else 0
                y_pred = np.full(len(y_val), fallback_class)
                fallback_used = True
            
            # Validate predictions are in expected range
            valid_classes = np.unique(y_train)
            invalid_mask = ~np.isin(y_pred, valid_classes)
            if np.any(invalid_mask):
                logger.warning(f"Model {model_name} (fold {fold_idx}) predicted invalid classes, correcting...")
                y_pred[invalid_mask] = valid_classes[0]  # Use first valid class
                
        except Exception as e:
            logger.error(f"Prediction failed for {model_name} (fold {fold_idx}): {str(e)}")
            # Create fallback predictions
            from scipy.stats import mode
            fallback_class = mode(y_train)[0][0] if len(y_train) > 0 else 0
            y_pred = np.full(len(y_val), fallback_class)
            fallback_used = True
        
        # Get prediction probabilities with proper validation
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_val)
        elif hasattr(model, 'decision_function'):
            y_proba = model.decision_function(X_val)
            # Convert decision function output to probabilities for multi-class
            if y_proba.ndim == 1 and n_classes > 2:
                # For multi-class SVM, decision_function returns 1D array, convert to 2D
                y_proba_2d = np.zeros((len(y_proba), n_classes))
                # Use softmax to convert to probabilities
                from scipy.special import softmax
                y_proba_2d[:, 0] = y_proba
                y_proba = softmax(y_proba_2d, axis=1)
            elif y_proba.ndim == 2:
                # Multi-class decision function, convert to probabilities
                from scipy.special import softmax
                y_proba = softmax(y_proba, axis=1)
        else:
            # Fallback: create one-hot encoded probabilities from predictions
            y_proba = np.zeros((len(y_pred), n_classes))
            for i, pred in enumerate(y_pred):
                if 0 <= pred < n_classes:
                    y_proba[i, pred] = 1.0
                else:
                    # Handle case where prediction is outside expected range
                    y_proba[i, 0] = 1.0
        
        # Ensure prediction probabilities have the correct shape
        if y_proba.ndim == 1:
            if n_classes == 2:
                # Binary classification with 1D output
                y_proba_2d = np.zeros((len(y_proba), 2))
                y_proba_2d[:, 0] = 1 - y_proba
                y_proba_2d[:, 1] = y_proba
                y_proba = y_proba_2d
            else:
                # Multi-class with 1D output - convert to 2D
                y_proba_2d = np.zeros((len(y_proba), n_classes))
                y_proba_2d[:, 0] = y_proba
                y_proba = y_proba_2d
        
        # Validate prediction probabilities shape matches training classes
        if y_proba.shape[1] != n_classes:
            logger.warning(f"Prediction shape mismatch: model outputs {y_proba.shape[1]} classes, training has {n_classes} classes")
            if y_proba.shape[1] > n_classes:
                # Take only the first n_classes
                y_proba = y_proba[:, :n_classes]
            else:
                # Pad with uniform probabilities
                padded_proba = np.zeros((y_proba.shape[0], n_classes))
                padded_proba[:, :y_proba.shape[1]] = y_proba
                # Fill remaining with uniform probability
                remaining_prob = (1.0 - y_proba.sum(axis=1, keepdims=True)) / (n_classes - y_proba.shape[1])
                remaining_prob = np.maximum(remaining_prob, 0.01)
                padded_proba[:, y_proba.shape[1]:] = remaining_prob
                y_proba = padded_proba
        
        # Calculate comprehensive classification metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        from sklearn.metrics import matthews_corrcoef
        
        # Calculate accuracy
        accuracy = accuracy_score(y_val, y_pred)
        
        # Calculate precision, recall, F1 with proper averaging for multi-class
        try:
            if n_classes == 2:
                # Binary classification
                precision = precision_score(y_val, y_pred, average='binary', zero_division=0)
                recall = recall_score(y_val, y_pred, average='binary', zero_division=0)
                f1 = f1_score(y_val, y_pred, average='binary', zero_division=0)
            else:
                # Multi-class classification - use weighted average
                precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
        except Exception as e:
            logger.warning(f"Could not calculate precision/recall/F1 for {model_name}: {str(e)}")
            precision = recall = f1 = 0.0
        
        # Calculate Matthews Correlation Coefficient
        try:
            # Use safe MCC scorer to prevent warnings
            mcc = safe_mcc_score(y_val, y_pred)
        except Exception as e:
            logger.warning(f"Could not calculate MCC for {model_name}: {str(e)}")
            mcc = 0.0
        
        # Calculate AUC with enhanced error handling
        try:
            if n_classes == 2:
                # Binary classification
                auc = roc_auc_score(y_val, y_proba[:, 1])
            else:
                # Multi-class classification
                from plots import enhanced_roc_auc_score
                auc = enhanced_roc_auc_score(y_val, y_proba, multi_class='ovr', average='weighted')
        except Exception as e:
            logger.warning(f"Could not calculate AUC for {model_name}: {str(e)}")
            auc = 0.5
        
        # Best validation score (using accuracy as the primary metric)
        best_validation_score = accuracy
        
        # Early stopping metrics (not used in basic classification, but needed for CSV consistency)
        early_stopping_used = False
        stopped_epoch = 'N/A'
        patience_used = float('nan')
            
        # Create plots if requested
        if make_plots:
            # Create output directory if it doesn't exist
            os.makedirs(out_dir, exist_ok=True)
            
            # Calculate confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_val, y_pred)
            
            # Generate class labels
            unique_classes = sorted(np.unique(np.concatenate([y_train, y_val])))
            class_labels = [str(cls) for cls in unique_classes]
            
            # Plot confusion matrix
            cm_path = os.path.join(out_dir, f"{plot_prefix}_cm.png")
            plot_confusion_matrix(cm, class_labels, 
                                title=f"{model_name} Confusion Matrix",
                                out_path=cm_path)
            
            # Plot ROC curve
            if hasattr(model, 'predict_proba'):
                roc_path = os.path.join(out_dir, f"{plot_prefix}_roc.png")
                if n_classes == 2:
                    plot_roc_curve_binary(model, X_val, y_val, class_labels,
                                        title=f"{model_name} ROC Curve",
                                        out_path=roc_path)
                else:
                    plot_roc_curve_multiclass(model, X_val, y_val, class_labels,
                                            title=f"{model_name} ROC Curve",
                                            out_path=roc_path)
        
        # Return all expected values with comprehensive metrics
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mcc': mcc,
            'auc': auc,
            'train_time': train_time,
            'best_validation_score': best_validation_score,
            'early_stopping_used': early_stopping_used,
            'stopped_epoch': stopped_epoch,
            'patience_used': patience_used,
            'n_features': n_features if n_features is not None else -1,
            'train_n_components': train_n_components
        }
        
        return model, metrics, y_val, y_pred
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return None, {}, None, None

def check_and_filter_classes_in_fold(y_train, y_val, min_samples_per_class=2):
    """
    Check class distribution in CV fold and filter out classes with insufficient samples.
    
    Parameters
    ----------
    y_train : np.ndarray
        Training labels for this fold
    y_val : np.ndarray
        Validation labels for this fold  
    min_samples_per_class : int
        Minimum number of samples required per class
        
    Returns
    -------
    tuple
        (valid_train_mask, valid_val_mask, filtered_y_train, filtered_y_val, label_mapping)
        Returns None for all if no valid classes remain
    """
    # Check class distribution in training set
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_val, counts_val = np.unique(y_val, return_counts=True)
    
    # Check class distribution (debug logging removed for performance)
    
    # Find classes that have sufficient samples in training set
    valid_train_classes = unique_train[counts_train >= min_samples_per_class]
    
    # Find classes that appear in both training and validation sets
    common_classes = np.intersect1d(valid_train_classes, unique_val)
    
    # For training set, we need at least 2 samples per class (scikit-learn requirement)
    # For validation set, we need at least 1 sample per class (more lenient)
    valid_classes = []
    for cls in common_classes:
        train_count = counts_train[unique_train == cls][0] if cls in unique_train else 0
        val_count = counts_val[unique_val == cls][0] if cls in unique_val else 0
        
        # Always require at least 2 samples in training set for scikit-learn compatibility
        # But allow the validation set to have just 1 sample
        if train_count >= 2 and val_count >= 1:
            valid_classes.append(cls)
    
    valid_classes = np.array(valid_classes)
    
    if len(valid_classes) < 2:
        logger.warning(f"Insufficient valid classes in fold: only {len(valid_classes)} classes meet requirements "
                      f"(>= 2 train samples, >= 1 val sample)")

        return None, None, None, None, None
    
    # Filter samples to only include valid classes
    train_mask = np.isin(y_train, valid_classes)
    val_mask = np.isin(y_val, valid_classes)
    
    y_train_filtered = y_train[train_mask]
    y_val_filtered = y_val[val_mask]
    
    # Verify we still have sufficient samples after filtering
    if len(y_train_filtered) < 2 * len(valid_classes):
        logger.warning(f"Insufficient training samples after filtering: {len(y_train_filtered)} samples "
                      f"for {len(valid_classes)} classes (need >= {2 * len(valid_classes)})")
        return None, None, None, None, None
    
    if len(y_val_filtered) < len(valid_classes):
        logger.warning(f"Insufficient validation samples after filtering: {len(y_val_filtered)} samples "
                      f"for {len(valid_classes)} classes (need >= {len(valid_classes)})")
        return None, None, None, None, None
    
    # Create label mapping to consecutive integers
    valid_classes_sorted = np.sort(valid_classes)
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(valid_classes_sorted)}
    y_train_relabeled = np.array([label_mapping[label] for label in y_train_filtered])
    y_val_relabeled = np.array([label_mapping[label] for label in y_val_filtered])
    

    
    return train_mask, val_mask, y_train_relabeled, y_val_relabeled, label_mapping

def validate_cv_fold_quality(idx_temp, y_temp, cv_splitter, min_samples_per_class=None):
    """
    Validate that CV folds have sufficient samples per class for classification.
    
    Parameters
    ----------
    idx_temp : np.ndarray
        Indices for the training data
    y_temp : np.ndarray
        Labels for the training data
    cv_splitter : object
        CV splitter object (KFold or StratifiedKFold)
    min_samples_per_class : int
        Minimum samples required per class in each fold
        
    Returns
    -------
    bool
        True if folds are valid, False otherwise
    """
    try:
        # Use adaptive minimum samples if not specified
        # Always require at least 2 samples per class for scikit-learn compatibility
        if min_samples_per_class is None:
            if CV_CONFIG["adaptive_min_samples"]:
                total_samples = len(y_temp)
                if total_samples < 10:
                    min_samples_per_class = 2  # Very small datasets - still need 2 for sklearn
                elif total_samples < 20:
                    min_samples_per_class = 2  # Small datasets - still need 2 for sklearn
                else:
                    min_samples_per_class = 2  # Use 2 for all datasets for sklearn compatibility
            else:
                min_samples_per_class = CV_CONFIG["min_samples_per_class_per_fold"]
        
        valid_folds = 0
        total_folds = 0
        fold_details = []
        
        for train_idx, val_idx in cv_splitter.split(idx_temp, y_temp):
            total_folds += 1
            y_train_fold = y_temp[train_idx]
            y_val_fold = y_temp[val_idx]
            
            # Check if this fold has valid class distribution
            train_mask, val_mask, _, _, _ = check_and_filter_classes_in_fold(
                y_train_fold, y_val_fold, min_samples_per_class
            )
            
            fold_valid = train_mask is not None
            if fold_valid:
                valid_folds += 1
            
            # Collect fold details for debugging
            unique_train, counts_train = np.unique(y_train_fold, return_counts=True)
            unique_val, counts_val = np.unique(y_val_fold, return_counts=True)
            fold_details.append({
                'fold': total_folds,
                'valid': fold_valid,
                'train_classes': dict(zip(unique_train, counts_train)),
                'val_classes': dict(zip(unique_val, counts_val)),
                'train_samples': len(y_train_fold),
                'val_samples': len(y_val_fold)
            })
        
        # Calculate the minimum acceptable number of valid folds
        # For 2-fold CV, we need at least 1 valid fold
        # For 3+ fold CV, we need at least 2 valid folds
        min_required_folds = 1 if total_folds == 2 else 2
        is_valid = valid_folds >= min_required_folds
        
        if not is_valid:
            logger.warning(f"CV validation failed: only {valid_folds}/{total_folds} folds have sufficient class distribution "
                          f"(need >= {min_required_folds})")
            logger.debug("Fold details:")
            for detail in fold_details:
                logger.debug(f"  Fold {detail['fold']}: valid={detail['valid']}, "
                           f"train={detail['train_samples']} samples {detail['train_classes']}, "
                           f"val={detail['val_samples']} samples {detail['val_classes']}")
        else:
            logger.debug(f"CV validation passed: {valid_folds}/{total_folds} folds are valid")
        
        return is_valid
    except Exception as e:
        logger.warning(f"Error validating CV folds: {str(e)}")
        return False 

def get_optimal_cv_splits(y_temp, is_regression=False):
    """
    Determine optimal number of CV splits based on data characteristics.
    
    Parameters
    ----------
    y_temp : np.ndarray
        Target values
    is_regression : bool
        Whether this is a regression task
        
    Returns
    -------
    int
        Optimal number of CV splits
    """
    n_samples = len(y_temp)
    
    if is_regression:
        # For regression, base splits on sample size
        if n_samples < 15:
            return 2
        elif n_samples < 30:
            return 3
        else:
            return min(5, n_samples // 10)
    else:
        # For classification, consider class distribution
        unique, counts = np.unique(y_temp, return_counts=True)
        n_classes = len(unique)
        min_class_count = np.min(counts)
        
        # Calculate maximum possible splits while maintaining minimum samples per class per fold
        # Use adaptive minimum samples based on dataset size
        # Always require at least 2 samples per class for scikit-learn compatibility
        if CV_CONFIG["adaptive_min_samples"]:
            if n_samples < 10:
                adaptive_min_samples = 2  # Very small datasets - still need 2 for sklearn
            elif n_samples < 20:
                adaptive_min_samples = 2  # Small datasets - still need 2 for sklearn
            else:
                adaptive_min_samples = 2  # Use 2 for all datasets for sklearn compatibility
        else:
            adaptive_min_samples = CV_CONFIG["min_samples_per_class_per_fold"]
        
        max_splits_by_class = min_class_count // adaptive_min_samples
        
        # Also consider total sample size
        max_splits_by_total = n_samples // CV_CONFIG["min_samples_per_fold"]
        
        # Take the minimum of both constraints
        max_possible_splits = min(max_splits_by_class, max_splits_by_total)
        
        # Ensure we have at least minimum splits
        optimal_splits = max(CV_CONFIG["min_cv_splits"], min(max_possible_splits, CV_CONFIG["max_cv_splits"]))
        
        logger.debug(f"CV splits calculation: n_samples={n_samples}, n_classes={n_classes}, "
                    f"min_class_count={min_class_count}, adaptive_min_samples={adaptive_min_samples}, "
                    f"max_by_class={max_splits_by_class}, max_by_total={max_splits_by_total}, optimal={optimal_splits}")
        
        return optimal_splits

def create_robust_cv_splitter(idx_temp, y_temp, is_regression=False, sample_ids=None):
    """
    Create a robust CV splitter with enhanced strategies for stratified regression and grouped CV.
    
    Parameters
    ----------
    idx_temp : np.ndarray
        Training indices
    y_temp : np.ndarray
        Training labels
    is_regression : bool
        Whether this is a regression task
    sample_ids : List[str], optional
        Sample IDs for extracting patient groups (for grouped CV)
        
    Returns
    -------
    tuple
        (cv_splitter, n_splits, cv_type_used)
    """
    n_samples = len(y_temp)
    
    # Determine optimal number of splits
    optimal_splits = get_optimal_cv_splits(y_temp, is_regression)
    
    # Try enhanced CV strategies first
    try:
        task_type = 'regression' if is_regression else 'classification'
        
        # Use enhanced CV splitter with stratified regression and grouped CV
        cv_result = create_enhanced_cv_splitter(
            y=y_temp,
            sample_ids=sample_ids,
            task_type=task_type,
            n_splits=optimal_splits,
            use_stratified_regression=True,
            use_grouped_cv=True,
            random_state=0
        )
        
        # Unpack the result (handle different return formats)
        if len(cv_result) == 4:
            cv_splitter, strategy_desc, y_for_cv, groups = cv_result
        else:
            cv_splitter, strategy_desc = cv_result[:2]
            y_for_cv = y_temp
            groups = None
        
        # Validate the enhanced strategy
        if validate_enhanced_cv_strategy(cv_splitter, y_for_cv, groups, optimal_splits, task_type):
            logger.info(f"Enhanced CV strategy successful: {strategy_desc}")
            return cv_splitter, optimal_splits, strategy_desc
        else:
            logger.warning(f"Enhanced CV strategy validation failed: {strategy_desc}")
            
    except Exception as e:
        logger.warning(f"Enhanced CV strategy failed: {e}, falling back to standard approach")
    
    # Fallback to original robust CV logic
    if is_regression:
        cv_splitter = KFold(n_splits=optimal_splits, shuffle=True, random_state=0)
        return cv_splitter, optimal_splits, "KFold"
    
    # For classification, try multiple strategies
    unique, counts = np.unique(y_temp, return_counts=True)
    n_classes = len(unique)
    min_class_count = np.min(counts)
    
    # Use adaptive minimum samples based on dataset size
    if CV_CONFIG["adaptive_min_samples"]:
        if n_samples < 6:
            adaptive_min_samples = 1  # Very small datasets
        elif n_samples < 10:
            adaptive_min_samples = 1  # Small datasets
        else:
            adaptive_min_samples = 2  # Normal datasets
    else:
        adaptive_min_samples = CV_CONFIG["min_samples_per_class_per_fold"]
    
    # If we have very small classes and merging is enabled, merge them
    if CV_CONFIG["merge_small_classes"] and min_class_count < adaptive_min_samples:
        logger.info(f"Merging small classes (min_count={min_class_count} < {adaptive_min_samples})")
        y_temp, label_mapping = merge_small_classes(y_temp, adaptive_min_samples)
        unique, counts = np.unique(y_temp, return_counts=True)
        n_classes = len(unique)
        min_class_count = np.min(counts)
        logger.info(f"After merging: {n_classes} classes, min_count={min_class_count}")
    
    # Try stratified split first with proper warning suppression
    try:
        if n_samples >= CV_CONFIG["min_total_samples_for_stratified"] and min_class_count >= adaptive_min_samples:
            # Additional safety check: ensure each class has enough samples for the splits
            if min_class_count >= optimal_splits + 1:  # +1 for safety buffer
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", 
                                          message="The least populated class in y has only .* members, which is less than n_splits=.*", 
                                          category=UserWarning)
                    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.model_selection._split")
                    
                    cv_splitter = StratifiedKFold(n_splits=optimal_splits, shuffle=True, random_state=0)
                    return cv_splitter, optimal_splits, "StratifiedKFold"
            else:
                logger.debug(f"Insufficient samples per class ({min_class_count}) for StratifiedKFold with {optimal_splits} splits")
    except Exception as e:
        logger.warning(f"StratifiedKFold failed: {e}")
    
    # Fall back to regular KFold
    n_splits = min(optimal_splits, n_samples // 3)
    n_splits = max(2, n_splits)  # At least 2 splits
    cv_splitter = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    return cv_splitter, n_splits, "KFold"

def log_cv_fold_summary(ds_name, y_temp, cv_splitter, cv_type_used, n_splits):
    """
    Log a summary of CV fold characteristics for debugging.
    
    Parameters
    ----------
    ds_name : str
        Dataset name
    y_temp : np.ndarray
        Training labels
    cv_splitter : object
        CV splitter object
    cv_type_used : str
        Type of CV splitter used
    n_splits : int
        Number of CV splits
    """
    try:
        unique, counts = np.unique(y_temp, return_counts=True)
        total_samples = len(y_temp)
        n_classes = len(unique)
        min_class_count = np.min(counts)
        max_class_count = np.max(counts)
        
        logger.info(f"CV Summary for {ds_name}:")
        logger.info(f"  Total samples: {total_samples}")
        logger.info(f"  Classes: {n_classes}")
        logger.info(f"  Class distribution: {dict(zip(unique, counts))}")
        logger.info(f"  Min/Max class count: {min_class_count}/{max_class_count}")
        logger.info(f"  CV type: {cv_type_used} with {n_splits} splits")
        
        # Analyze expected fold sizes
        avg_fold_size = total_samples // n_splits
        logger.info(f"  Expected avg fold size: {avg_fold_size}")
        logger.info(f"  Expected min samples per class per fold: ~{min_class_count // n_splits}")
        
        # Check if this looks problematic
        if min_class_count < n_splits * 2:
            logger.warning(f"  WARNING: Smallest class ({min_class_count} samples) may cause issues with {n_splits}-fold CV")
        
    except Exception as e:
        logger.debug(f"Error logging CV summary: {str(e)}")

def combine_best_fold_metrics(ds_name: str, base_out: str):
    """
    Combine best fold metrics from both extraction and selection pipelines into a single CSV file per dataset.
    This function should be called after both extraction and selection pipelines are completed.
    
    Parameters
    ----------
    ds_name : str
        Dataset name
    base_out : str
        Base output directory
    """
    try:
        metrics_dir = os.path.join(base_out, "metrics")
        
        # Look for both extraction and selection best fold metrics files
        extraction_file = os.path.join(metrics_dir, f"{ds_name}_extraction_best_fold_metrics.csv")
        selection_file = os.path.join(metrics_dir, f"{ds_name}_selection_best_fold_metrics.csv")
        
        combined_data = []
        
        # Read extraction metrics if file exists
        if os.path.exists(extraction_file):
            try:
                ext_df = pd.read_csv(extraction_file)
                combined_data.append(ext_df)
                logger.info(f"Found {len(ext_df)} extraction best fold entries for {ds_name}")
            except Exception as e:
                logger.warning(f"Error reading extraction best fold metrics for {ds_name}: {str(e)}")
        
        # Read selection metrics if file exists
        if os.path.exists(selection_file):
            try:
                sel_df = pd.read_csv(selection_file)
                combined_data.append(sel_df)
                logger.info(f"Found {len(sel_df)} selection best fold entries for {ds_name}")
            except Exception as e:
                logger.warning(f"Error reading selection best fold metrics for {ds_name}: {str(e)}")
        
        # Combine and save if we have data
        if combined_data:
            combined_df = pd.concat(combined_data, ignore_index=True)
            
            # Create the combined filename
            combined_file = os.path.join(metrics_dir, f"{ds_name}_combined_best_fold_metrics.csv")
            
            # Save combined metrics
            combined_df.to_csv(combined_file, index=False)
            logger.info(f"Saved combined best fold metrics for {ds_name} to {combined_file} ({len(combined_df)} entries)")
            
            return combined_file
        else:
            logger.warning(f"No best fold metrics found for {ds_name}")
            return None
            
    except Exception as e:
        logger.error(f"Error combining best fold metrics for {ds_name}: {str(e)}")
        return None

def _run_pipeline(
    ds_name: str, 
    data_modalities: Dict[str, pd.DataFrame], 
    common_ids: List[str], 
    y: np.ndarray, 
    base_out: str,
    transformers: Dict[str, Any], 
    n_trans_list: Union[List[int], Dict[str, List[int]]], 
    models: List[str],
    progress_count: List[int], 
    total_runs: int,
    is_regression: bool = True, 
    pipeline_type: str = "extraction"
):
    # Suppress sklearn warnings early in pipeline
    suppress_sklearn_warnings()
    """
    Generic pipeline function handling both extraction and selection for
    both regression and classification tasks.
    
    Parameters
    ----------
    ds_name : str
        Dataset name
    data_modalities : Dict[str, pd.DataFrame]
        Dictionary of modality DataFrames
    common_ids : List[str]
        List of common sample IDs
    y : np.ndarray
        Target values
    base_out : str
        Base output directory
    transformers : Dict[str, Any]
        Dictionary of transformers (extractors or selectors)
    n_trans_list : Union[List[int], Dict[str, List[int]]]
        List of parameters for transformers (n_components or n_features)
        Can be a single list for all transformers, or a dict mapping transformer names to their specific lists
    models : List[str]
        List of model names to train
    progress_count : List[int]
        Progress counter [0]
    total_runs : int
        Total number of runs
    is_regression : bool
        Whether this is a regression task
    pipeline_type : str
        Type of pipeline ("extraction" or "selection")
        
    Returns
    -------
    None
    """
    # Log pipeline start
    task_type = "regression" if is_regression else "classification"
    log_pipeline_stage(f"{pipeline_type.upper()}_PIPELINE_START", dataset=ds_name, 
                      details=f"{task_type} with {len(transformers)} transformers, {len(n_trans_list) if isinstance(n_trans_list, dict) else len(n_trans_list)} parameters")
    logger.debug(f"[PIPELINE] {ds_name} - Starting {pipeline_type} pipeline for {task_type}")
    
    # Ensure output directories exist
    os.makedirs(base_out, exist_ok=True)
    for subdir in ["models", "metrics", "plots"]:
        subdir_path = os.path.join(base_out, subdir)
        os.makedirs(subdir_path, exist_ok=True)
        logger.debug(f"[PIPELINE] {ds_name} - Created directory: {subdir_path}")
    
    # Convert y to numpy array
    y_arr = np.array(y)
    
    # Warn if sample size is very small
    if len(y_arr) < 30:
        logger.warning(f"Sample size for {ds_name} is very small ({len(y_arr)} samples). Model performance may be unstable.")
        logger.debug(f"[PIPELINE] {ds_name} - Small sample size warning: {len(y_arr)} samples")
    
    # Create indices array for row-based indexing
    indices = np.arange(len(common_ids))
    
    # Create id to index and index to id mappings
    id_to_idx = {id_: idx for idx, id_ in enumerate(common_ids)}
    idx_to_id = {idx: id_ for id_, idx in id_to_idx.items()}
    
    logger.debug(f"[PIPELINE] {ds_name} - Created sample mappings for {len(common_ids)} samples")
    
    # Split with stratification if possible
    try:
        # For small datasets (< 15 samples), use a larger test proportion to ensure some test samples
        test_size = 0.3 if len(indices) < 15 else 0.2
        logger.debug(f"[PIPELINE] {ds_name} - Using test_size={test_size} for {len(indices)} samples")
        # For regression, bin continuous values for stratified split
        if is_regression:
            # Validate that y_arr contains numeric data
            if not np.issubdtype(y_arr.dtype, np.number):
                logger.error(f"Regression target data contains non-numeric values: dtype={y_arr.dtype}")
                logger.error(f"Sample values: {y_arr[:5] if len(y_arr) > 0 else 'empty'}")
                raise ValueError("Regression target data must be numeric")
            
            # Check for NaN or infinite values
            if np.any(np.isnan(y_arr)) or np.any(np.isinf(y_arr)):
                logger.warning(f"Found NaN or infinite values in regression target, cleaning...")
                y_arr = np.nan_to_num(y_arr, nan=0.0, posinf=0.0, neginf=0.0)
            
            n_bins = min(5, len(y_arr)//3)
            unique_vals = len(np.unique(y_arr))
            if n_bins < 2 or unique_vals < n_bins or len(indices) < 15:
                logger.info(f"Skipping stratification for regression: n_bins={n_bins}, unique_vals={unique_vals}, sample_size={len(indices)}")
                raise ValueError("Too few samples or unique values for stratified split in regression")
            
            try:
                y_bins = pd.qcut(y_arr, n_bins, labels=False, duplicates='drop')
                idx_temp, idx_test, y_temp, y_test = train_test_split(
                    indices, y_arr, test_size=test_size, random_state=0, stratify=y_bins
                )
            except Exception as e:
                logger.warning(f"Failed to create quantile bins for regression stratification: {str(e)}")
                raise ValueError("Could not create stratified split for regression")
        else:
            # For classification, check class distribution
            unique, counts = np.unique(y_arr, return_counts=True)
            min_samples = np.min(counts)
            n_classes = len(unique)
            
            logger.debug(f"Class distribution for {ds_name}: {dict(zip(unique, counts))}")
            
            # Check if data was properly pre-filtered (classes should be consecutive starting from 0)
            expected_classes = np.arange(n_classes)
            if not np.array_equal(unique, expected_classes):
                logger.info(f"Classes are not consecutive (expected {expected_classes}, got {unique}). This may indicate incomplete preprocessing.")
            
            # Calculate test set size and check if stratification is feasible
            test_samples = int(len(indices) * test_size)
            
            # Check if stratified splitting is feasible
            if test_samples < n_classes:
                logger.warning(f"Dataset {ds_name}: test_size={test_samples} < n_classes={n_classes}. Stratified split not feasible, using random split.")
                raise ValueError(f"Test size ({test_samples}) smaller than number of classes ({n_classes})")
            
            # If we still have problematic classes, it means the preprocessing in cli.py didn't work correctly
            if min_samples < 2:
                logger.warning(f"Dataset {ds_name} still has classes with < 2 samples after preprocessing: {dict(zip(unique, counts))}")
                logger.warning(f"Applying emergency class filtering in CV pipeline...")
                
                # Keep only classes with at least 2 samples
                valid_classes = unique[counts >= 2]
                if len(valid_classes) < 2:
                    logger.error(f"Too few valid classes for classification. Falling back to regular split.")
                    raise ValueError("Too few valid classes for stratified split")
                
                # Filter samples to only include valid classes
                valid_mask = np.isin(y_arr, valid_classes)
                indices_filtered = indices[valid_mask]
                y_arr_filtered = y_arr[valid_mask]
                
                # Relabel classes to be consecutive integers starting from 0
                valid_classes_sorted = np.sort(valid_classes)
                label_mapping = {old_label: new_label for new_label, old_label in enumerate(valid_classes_sorted)}
                y_arr_relabeled = np.array([label_mapping[label] for label in y_arr_filtered])
                
                logger.info(f"Emergency filtering: dataset from {len(y_arr)} to {len(y_arr_filtered)} samples, {n_classes} to {len(valid_classes)} classes")
                logger.info(f"Emergency relabeling: {label_mapping}")
                
                # Re-check if stratification is still feasible after filtering
                test_samples_filtered = int(len(indices_filtered) * test_size)
                if test_samples_filtered < len(valid_classes):
                    logger.warning(f"Even after filtering, test_size={test_samples_filtered} < n_classes={len(valid_classes)}. Using random split.")
                    raise ValueError(f"Test size still smaller than number of classes after filtering")
                
                idx_temp, idx_test, y_temp, y_test = train_test_split(
                    indices_filtered, y_arr_filtered, test_size=test_size, random_state=0, stratify=y_arr_relabeled
                )
            else:
                # Standard stratified split when all classes have sufficient samples
                logger.debug(f"All classes have sufficient samples for {ds_name}, proceeding with stratified split")
                idx_temp, idx_test, y_temp, y_test = train_test_split(
                    indices, y_arr, test_size=test_size, random_state=0, stratify=y_arr
                )
    except ValueError as e:
        logger.warning(f"Stratification failed for {ds_name}: {str(e)}. Falling back to regular split.")
        if not is_regression:
            unique, counts = np.unique(y_arr, return_counts=True)
            logger.warning(f"Class distribution for {ds_name}: {dict(zip(unique, counts))}")
        idx_temp, idx_test, y_temp, y_test = train_test_split(
            indices, y_arr, test_size=0.2, random_state=0
        )
    
    # Process each transformer and parameter combination
    from config import MISSING_MODALITIES_CONFIG
    
    # Save all results for batch processing
    all_pipeline_results = []
    
    for trans_name, trans_obj in transformers.items():
        # Handle both formats: List[int] for uniform n_values, or Dict[str, List[int]] for extractor-specific values
        if isinstance(n_trans_list, dict):
            # Use extractor-specific n_components/n_features
            n_values_for_this_transformer = n_trans_list.get(trans_name, [8])  # Default to [8] if not found
            logger.info(f"Using extractor-specific values for {trans_name}: {n_values_for_this_transformer}")
        else:
            # Use the same n_values for all transformers (backward compatibility)
            n_values_for_this_transformer = n_trans_list
            logger.debug(f"Using uniform values for {trans_name}: {n_values_for_this_transformer}")
        
        for n_val in n_values_for_this_transformer:
            try:
                # Update and report progress
                progress_count[0] += 1
                run_idx = progress_count[0]
                trans_type = "EXTRACT" if pipeline_type == "extraction" else "SELECT"
                task_type = "REG" if is_regression else "CLF"
                progress_msg = f"[{trans_type}-{task_type} CV] {run_idx}/{total_runs} => {ds_name} | {trans_name}-{n_val}"
                # Always show progress in terminal
                print(progress_msg)
                logger.info(progress_msg)
                log_pipeline_stage(f"{trans_type}_CV", dataset=ds_name, details=f"{trans_name}-{n_val} ({run_idx}/{total_runs})")

                # Use the enhanced robust CV splitter with sample IDs for grouped CV
                cv_splitter, n_splits, cv_type_used = create_robust_cv_splitter(idx_temp, y_temp, is_regression, sample_ids=common_ids)
                
                logger.info(f"Dataset: {ds_name}, using {cv_type_used} {n_splits}-fold CV with {len(idx_temp)} training samples")
                logger.debug(f"[PIPELINE] {ds_name} - {trans_name}-{n_val}: Using {cv_type_used} {n_splits}-fold CV")
                
                # Log detailed CV summary for classification tasks
                if not is_regression:
                    log_cv_fold_summary(ds_name, y_temp, cv_splitter, cv_type_used, n_splits)
                
                # Results storage
                pipeline_results = []
                
                # Process each missing percentage
                for missing_percentage in MISSING_MODALITIES_CONFIG["missing_percentages"]:
                    try:
                        # COMPREHENSIVE FUSION TESTING: Test all appropriate fusion techniques as specified in README
                        # Always test ALL fusion techniques regardless of FUSION_UPGRADES_CONFIG
                        if missing_percentage == 0.0:
                            # For 0% missing data, test ALL fusion techniques that work with clean data
                            # As specified in README: [attention_weighted, learnable_weighted, mkl, snf, early_fusion_pca]
                            integration_techniques = [
                                "attention_weighted", 
                                "learnable_weighted", 
                                "mkl", 
                                "snf", 
                                "early_fusion_pca"
                            ]
                            logger.info(f"Testing {len(integration_techniques)} fusion techniques for 0% missing data: {integration_techniques}")
                        else:
                            # For missing data (>0%), test only techniques that handle missing data
                            # As specified in README: [mkl, snf, early_fusion_pca]
                            integration_techniques = [
                                "mkl", 
                                "snf", 
                                "early_fusion_pca"
                            ]
                            logger.info(f"Testing {len(integration_techniques)} fusion techniques for {missing_percentage*100}% missing data: {integration_techniques}")
                        
                        # Process each integration technique
                        for integration_technique in integration_techniques:
                            try:
                                cv_results = []
                                model_candidates = {model_name: {"metric": None, "model": None, "fold_idx": None, "train_val": None} for model_name in models}
                                model_yvals_folds = {model_name: [] for model_name in models}
                                model_ypreds_folds = {model_name: [] for model_name in models}
                                train_val_data = []  # Store (train_idx, val_idx, ...) for each fold
                                
                                logger.info(f"Processing {ds_name} with {trans_name}-{n_val}, missing={missing_percentage}%, integration={integration_technique}")
                                
                                # Process CV folds
                                for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(idx_temp, y_temp)):
                                    try:
                                        train_val_data.append((train_idx, val_idx))
                                        with warnings.catch_warnings():
                                            warnings.simplefilter("ignore", UserWarning)
                                            if pipeline_type == "extraction":
                                                if is_regression:
                                                    result, model_objs = process_cv_fold(
                                                        train_idx, val_idx, idx_temp, idx_test, y_temp, y_test,
                                                        data_modalities, models, (clone(trans_obj) if hasattr(trans_obj, 'fit') else trans_obj), n_val, 
                                                        id_to_idx, idx_to_id, common_ids, missing_percentage,
                                                        fold_idx, base_out, ds_name, trans_name, pipeline_type,
                                                        is_regression,
                                                        make_plots=False,
                                                        integration_technique=integration_technique
                                                    )
                                                else:
                                                    result, model_objs, yvals, ypreds = process_cv_fold(
                                                        train_idx, val_idx, idx_temp, idx_test, y_temp, y_test,
                                                        data_modalities, models, (clone(trans_obj) if hasattr(trans_obj, 'fit') else trans_obj), n_val, 
                                                        id_to_idx, idx_to_id, common_ids, missing_percentage,
                                                        fold_idx, base_out, ds_name, trans_name, pipeline_type,
                                                        is_regression,
                                                        make_plots=False,
                                                        integration_technique=integration_technique
                                                    )
                                            else:
                                                if is_regression:
                                                    result, model_objs = process_cv_fold(
                                                        train_idx, val_idx, idx_temp, idx_test, y_temp, y_test,
                                                        data_modalities, models, (clone(trans_obj) if hasattr(trans_obj, 'fit') else trans_obj), n_val, 
                                                        id_to_idx, idx_to_id, common_ids, missing_percentage,
                                                        fold_idx, base_out, ds_name, trans_name, pipeline_type,
                                                        is_regression,
                                                        make_plots=False,
                                                        integration_technique=integration_technique
                                                    )
                                                else:
                                                    result, model_objs, yvals, ypreds = process_cv_fold(
                                                        train_idx, val_idx, idx_temp, idx_test, y_temp, y_test,
                                                        data_modalities, models, (clone(trans_obj) if hasattr(trans_obj, 'fit') else trans_obj), n_val, 
                                                        id_to_idx, idx_to_id, common_ids, missing_percentage,
                                                        fold_idx, base_out, ds_name, trans_name, pipeline_type,
                                                        is_regression,
                                                        make_plots=False,
                                                        integration_technique=integration_technique
                                                    )
                                        if result:
                                            cv_results.append(result)
                                            for model_name in models:
                                                if not is_regression and model_name in yvals and model_name in ypreds:
                                                    model_yvals_folds[model_name].append(yvals[model_name])
                                                    model_ypreds_folds[model_name].append(ypreds[model_name])
                                                
                                                # Add this missing code to update model_candidates based on model performance
                                                if model_name in result:
                                                    # Get key metric for comparing model performance
                                                    metric_name = 'r2' if is_regression else 'f1'  # Use R² for regression, F1 for classification
                                                    current_metric = result[model_name].get(metric_name)
                                                    
                                                    if current_metric is not None:
                                                        # For regression, higher R² is better; for classification, higher F1 is better
                                                        current_best = model_candidates[model_name]["metric"]
                                                        
                                                        # If we don't have a best model yet or this one is better, update
                                                        if current_best is None or current_metric > current_best:
                                                            model_candidates[model_name]["metric"] = current_metric
                                                            model_candidates[model_name]["model"] = model_objs.get(model_name)
                                                            model_candidates[model_name]["fold_idx"] = fold_idx
                                                            model_candidates[model_name]["train_val"] = (train_idx, val_idx)
                                    except Exception as e:
                                        logger.error(f"Error processing fold {fold_idx} for {ds_name} with {trans_name}-{n_val} (missing={missing_percentage}%, integration={integration_technique}): {str(e)}")
                                        continue  # Continue to next fold
                                
                                # After all folds, find the best fold and rerun only that fold with make_plots=True, saving model and plots
                                best_fold_metrics = {}  # Store best fold metrics for CSV saving
                                for model_name in models:
                                    try:
                                        best_model = model_candidates[model_name]["model"]
                                        best_metric = model_candidates[model_name]["metric"]
                                        best_fold_idx = model_candidates[model_name]["fold_idx"]
                                        if best_model is not None and best_fold_idx is not None:
                                            # Get the train/val indices for the best fold
                                            train_idx, val_idx = train_val_data[best_fold_idx]
                                            # Rerun process_cv_fold for the best fold, but with make_plots=True
                                            if is_regression:
                                                # Use a modified plot prefix with "best_fold", pipeline_type, missing_percentage, and integration technique
                                                best_plot_prefix = f"{ds_name}_best_fold_{pipeline_type}_{trans_name}_{n_val}_{model_name}_{missing_percentage}_{integration_technique}"
                                                
                                                best_fold_results, best_model_obj = process_cv_fold(
                                                    train_idx, val_idx, idx_temp, idx_test, y_temp, y_test,
                                                    data_modalities, [model_name], (clone(trans_obj) if hasattr(trans_obj, 'fit') else trans_obj), n_val, 
                                                    id_to_idx, idx_to_id, common_ids, missing_percentage,
                                                    best_fold_idx, base_out, ds_name, trans_name, pipeline_type,
                                                    is_regression,
                                                    make_plots=True,
                                                    plot_prefix_override=best_plot_prefix,
                                                    integration_technique=integration_technique
                                                )
                                            else:
                                                # Use a modified plot prefix with "best_fold", pipeline_type, missing_percentage, and integration technique
                                                best_plot_prefix = f"{ds_name}_best_fold_{pipeline_type}_{trans_name}_{n_val}_{model_name}_{missing_percentage}_{integration_technique}"
                                                
                                                best_fold_results, best_model_obj, _, _ = process_cv_fold(
                                                    train_idx, val_idx, idx_temp, idx_test, y_temp, y_test,
                                                    data_modalities, [model_name], (clone(trans_obj) if hasattr(trans_obj, 'fit') else trans_obj), n_val, 
                                                    id_to_idx, idx_to_id, common_ids, missing_percentage,
                                                    best_fold_idx, base_out, ds_name, trans_name, pipeline_type,
                                                    is_regression,
                                                    make_plots=True,
                                                    plot_prefix_override=best_plot_prefix,
                                                    integration_technique=integration_technique
                                                )
                                            
                                            # Store best fold metrics for this model
                                            if model_name in best_fold_results:
                                                best_fold_metrics[model_name] = best_fold_results[model_name].copy()
                                                best_fold_metrics[model_name]['best_fold_idx'] = best_fold_idx
                                            
                                            # Save the best model with integration technique in filename
                                            model_path = os.path.join(
                                                base_out, "models",
                                                f"best_model_{pipeline_type}_{model_name}_{trans_name}_{n_val}_{missing_percentage}_{integration_technique}.pkl"
                                            )
                                            # Save best_model_obj instead of best_model since it's the freshly retrained model
                                            # with make_plots=True
                                            if model_name in best_model_obj:
                                                joblib.dump(best_model_obj[model_name], model_path)
                                                logger.info(f"Saved best model for {model_name} to {model_path}")
                                            else:
                                                # Fallback to the original model if something went wrong
                                                joblib.dump(best_model, model_path)
                                                logger.info(f"Fallback: Saved original best model for {model_name} to {model_path}")
                                    except Exception as e:
                                        logger.error(f"Error processing best fold for model {model_name} with {trans_name}-{n_val} (missing={missing_percentage}%, integration={integration_technique}): {str(e)}")
                                        continue  # Continue to next model
                                
                                # Save best fold metrics to CSV
                                if best_fold_metrics:
                                    best_fold_results_list = []
                                    for model_name, metrics in best_fold_metrics.items():
                                        # Create result entry for best fold with consistent column naming
                                        best_fold_entry = {
                                            "Dataset": ds_name, 
                                            "Workflow": f"{pipeline_type.title()}-BestFold",
                                            "Algorithm": trans_name,
                                            "n_features": metrics.get('n_features', -1),
                                            "n_components": n_val,
                                            "train_n_components": metrics.get('train_n_components', -1),
                                            "integration_tech": integration_technique,
                                            "Model": model_name,
                                            "Missing_Percentage": missing_percentage,
                                            "best_fold_idx": metrics.get('best_fold_idx', -1)
                                        }
                                        
                                        # Add the performance metrics
                                        if is_regression:
                                            # For regression metrics
                                            best_fold_entry.update({
                                                'mse': metrics.get('mse', float('nan')),
                                                'rmse': metrics.get('rmse', float('nan')),
                                                'mae': metrics.get('mae', float('nan')),
                                                'r2': metrics.get('r2', float('nan')),
                                                'train_time': metrics.get('train_time', float('nan')),
                                                # Early stopping metrics
                                                'early_stopping_used': metrics.get('early_stopping_used', False),
                                                'best_validation_score': metrics.get('best_validation_score', float('nan')),
                                                'stopped_epoch': str(metrics.get('stopped_epoch', 'N/A')),
                                                'patience_used': metrics.get('patience_used', float('nan'))
                                            })
                                        else:
                                            # For classification metrics
                                            best_fold_entry.update({
                                                'accuracy': metrics.get('accuracy', float('nan')),
                                                'precision': metrics.get('precision', float('nan')),
                                                'recall': metrics.get('recall', float('nan')),
                                                'f1': metrics.get('f1', float('nan')),
                                                'auc': metrics.get('auc', float('nan')),
                                                'mcc': metrics.get('mcc', float('nan')),
                                                'train_time': metrics.get('train_time', float('nan')),
                                                # Early stopping metrics
                                                'early_stopping_used': metrics.get('early_stopping_used', False),
                                                'best_validation_score': metrics.get('best_validation_score', float('nan')),
                                                'stopped_epoch': str(metrics.get('stopped_epoch', 'N/A')),
                                                'patience_used': metrics.get('patience_used', float('nan'))
                                            })
                                        
                                        best_fold_results_list.append(best_fold_entry)
                                    
                                    # Save best fold metrics to CSV
                                    try:
                                        best_fold_metrics_file = os.path.join(
                                            base_out, "metrics", 
                                            f"{ds_name}_{pipeline_type}_best_fold_metrics.csv"
                                        )
                                        
                                        # Check if file exists
                                        file_exists = os.path.exists(best_fold_metrics_file)
                                        
                                        # Append results to CSV
                                        pd.DataFrame(best_fold_results_list).to_csv(
                                            best_fold_metrics_file,
                                            mode='a',
                                            header=not file_exists,
                                            index=False
                                        )
                                        logger.info(f"Saved {len(best_fold_results_list)} best fold results to {best_fold_metrics_file}")
                                    except Exception as e:
                                        logger.error(f"Error saving best fold metrics: {str(e)}")
                                
                                # Aggregate metrics across folds
                                cv_metrics = {}
                                for model_name in models:
                                    valid_results = []
                                    for i in range(n_splits):  # <- Use n_splits instead of cv.n_splits
                                        if i < len(cv_results) and model_name in cv_results[i]:
                                            valid_results.append(cv_results[i][model_name])
                                    
                                    if valid_results:
                                        # Average metrics across folds
                                        metric_keys = valid_results[0].keys()
                                        avg_metrics = {}
                                        
                                        for k in metric_keys:
                                            values = []
                                            for m in valid_results:
                                                if k in m:
                                                    val = m[k]
                                                    # Only include numeric values for averaging
                                                    if isinstance(val, (int, float, np.number)) and not (isinstance(val, float) and np.isnan(val)):
                                                        values.append(val)
                                            
                                            if values:
                                                # Average numeric values
                                                avg_metrics[k] = np.mean(values)
                                            elif k in valid_results[0]:
                                                # For non-numeric values, take the first occurrence
                                                # This handles early stopping info like 'early_stopping_used', 'stopped_epoch', etc.
                                                first_val = valid_results[0][k]
                                                if k == 'early_stopping_history':
                                                    # For history, take the longest one (best performing fold)
                                                    longest_history = max([m.get(k, []) for m in valid_results], key=len, default=[])
                                                    avg_metrics[k] = longest_history
                                                elif k == 'stopped_epoch':
                                                    # For stopped_epoch, take the average if numeric, else first non-N/A value
                                                    numeric_epochs = [m[k] for m in valid_results if k in m and isinstance(m[k], (int, float, np.number)) and not np.isnan(m[k])]
                                                    if numeric_epochs:
                                                        avg_metrics[k] = np.mean(numeric_epochs)
                                                    else:
                                                        avg_metrics[k] = first_val
                                                else:
                                                    # For other non-numeric values, take the first
                                                    avg_metrics[k] = first_val
                                        
                                        cv_metrics[model_name] = avg_metrics
                                
                                # Add combined results
                                for model_name, metrics in cv_metrics.items():
                                    # Add additional metrics to the result entry with consistent column naming
                                    result_entry = {
                                        "Dataset": ds_name, 
                                        "Workflow": f"{pipeline_type.title()}-CV",
                                        "Algorithm": trans_name,
                                        "n_features": metrics.get('n_features', -1),  # Original feature count
                                        "n_components": n_val,  # Intended number of components/features
                                        "train_n_components": metrics.get('train_n_components', -1),  # Actual components used in training
                                        "integration_tech": integration_technique,
                                        "Model": model_name,
                                        "Missing_Percentage": missing_percentage
                                    }
                                    
                                    # Add the performance metrics
                                    if is_regression:
                                        # For regression metrics
                                        result_entry.update({
                                            'mse': metrics.get('mse', float('nan')),
                                            'rmse': metrics.get('rmse', float('nan')),
                                            'mae': metrics.get('mae', float('nan')),
                                            'r2': metrics.get('r2', float('nan')),
                                            'train_time': metrics.get('train_time', float('nan')),
                                            # Early stopping metrics
                                            'early_stopping_used': metrics.get('early_stopping_used', False),
                                            'best_validation_score': metrics.get('best_validation_score', float('nan')),
                                            'stopped_epoch': str(metrics.get('stopped_epoch', 'N/A')),  # Convert to string for CSV
                                            'patience_used': metrics.get('patience_used', float('nan'))
                                        })
                                    else:
                                        # For classification metrics
                                        result_entry.update({
                                            'accuracy': metrics.get('accuracy', float('nan')),
                                            'precision': metrics.get('precision', float('nan')),
                                            'recall': metrics.get('recall', float('nan')),
                                            'f1': metrics.get('f1', float('nan')),
                                            'auc': metrics.get('auc', float('nan')),
                                            'mcc': metrics.get('mcc', float('nan')),
                                            'train_time': metrics.get('train_time', float('nan')),
                                            # Early stopping metrics
                                            'early_stopping_used': metrics.get('early_stopping_used', False),
                                            'best_validation_score': metrics.get('best_validation_score', float('nan')),
                                            'stopped_epoch': str(metrics.get('stopped_epoch', 'N/A')),  # Convert to string for CSV
                                            'patience_used': metrics.get('patience_used', float('nan'))
                                        })
                                    
                                    pipeline_results.append(result_entry)
                            except Exception as e:
                                logger.error(f"Error processing integration technique {integration_technique} for {ds_name} with {trans_name}-{n_val} (missing={missing_percentage}%): {str(e)}")
                                continue  # Continue to next integration technique
                    except Exception as e:
                        logger.error(f"Error processing missing percentage {missing_percentage} for {ds_name} with {trans_name}-{n_val}: {str(e)}")
                        continue  # Continue to next missing percentage
                
                # Add results to the batch for later processing
                all_pipeline_results.extend(pipeline_results)
                
                # Write results to file every 10 transformers or at the end
                # This reduces disk I/O while still providing regular checkpoints
                if len(all_pipeline_results) >= 20 or (trans_name == list(transformers.keys())[-1] and n_val == n_trans_list[-1]):
                    try:
                        # Save results to CSV in batches
                        if all_pipeline_results:
                            metrics_file = os.path.join(
                                base_out, "metrics", 
                                f"{ds_name}_{pipeline_type}_cv_metrics.csv"
                            )
                            
                            # Check if file exists
                            file_exists = os.path.exists(metrics_file)
                            
                            # Append results to CSV
                            pd.DataFrame(all_pipeline_results).to_csv(
                                metrics_file,
                                mode='a',
                                header=not file_exists,
                                index=False
                            )
                            logger.info(f"Saved {len(all_pipeline_results)} results to {metrics_file}")
                            
                            # Clear batch after saving
                            all_pipeline_results = []
                            
                            # Force garbage collection to free memory
                            import gc
                            gc.collect()
                    except Exception as e:
                        logger.error(f"Error saving metrics batch: {str(e)}")
                
            except KeyboardInterrupt:
                logger.warning(f"KeyboardInterrupt during {ds_name} with {trans_name}-{n_val}. Aborting all processing.")
                raise  # Re-raise to abort all processing
            except Exception as e:
                logger.error(f"Error processing {trans_name}-{n_val} for {ds_name}: {str(e)}")
                continue  # Continue to next n_val
    
    # Final save of any remaining results
    if all_pipeline_results:
        try:
            metrics_file = os.path.join(
                base_out, "metrics", 
                f"{ds_name}_{pipeline_type}_cv_metrics.csv"
            )
            
            # Check if file exists
            file_exists = os.path.exists(metrics_file)
            
            # Append results to CSV
            pd.DataFrame(all_pipeline_results).to_csv(
                metrics_file,
                mode='a',
                header=not file_exists,
                index=False
            )
            logger.info(f"Saved final {len(all_pipeline_results)} results to {metrics_file}")
        except Exception as e:
            logger.error(f"Error saving final metrics batch: {str(e)}")
    
    # Clean up resources
    import gc
    gc.collect()

def run_extraction_pipeline(
    ds_name: str, 
    data_modalities: Dict[str, pd.DataFrame], 
    common_ids: List[str], 
    y: np.ndarray, 
    base_out: str,
    extractors: Dict[str, Any], 
    n_comps_list: List[int], 
    models: List[str],
    progress_count: List[int], 
    total_runs: int,
    is_regression: bool = True
):
    """
    Run extraction pipeline for a dataset with optimal n_components from hyperparameters.
    
    Parameters
    ----------
    ds_name : str
        Dataset name
    data_modalities : Dict[str, pd.DataFrame]
        Dictionary of modality DataFrames
    common_ids : List[str]
        List of common sample IDs
    y : np.ndarray
        Target values
    base_out : str
        Base output directory
    extractors : Dict[str, Any]
        Dictionary of extractors
    n_comps_list : List[int]
        List of n_components values (ignored for extraction, uses hyperparameters instead)
    models : List[str]
        List of model names to train
    progress_count : List[int]
        Progress counter [0]
    total_runs : int
        Total number of runs
    is_regression : bool
        Whether this is a regression task
    """
    from models import get_extraction_n_components_list
    
    # Get optimal n_components for each extractor from hyperparameters
    task = "reg" if is_regression else "clf"
    optimal_n_components = get_extraction_n_components_list(ds_name, extractors, task)
    
    logger.info(f"Using optimal n_components from hyperparameters for {ds_name}: {optimal_n_components}")
    
    _run_pipeline(
        ds_name=ds_name, 
        data_modalities=data_modalities, 
        common_ids=common_ids, 
        y=y, 
        base_out=base_out,
        transformers=extractors, 
        n_trans_list=optimal_n_components,  # Use extractor-specific optimal values
        models=models,
        progress_count=progress_count, 
        total_runs=total_runs,
        is_regression=is_regression, 
        pipeline_type="extraction"
    )

def run_selection_pipeline(
    ds_name: str, 
    data_modalities: Dict[str, pd.DataFrame], 
    common_ids: List[str], 
    y: np.ndarray, 
    base_out: str,
    selectors: Dict[str, str], 
    n_feats_list: List[int], 
    models: List[str],
    progress_count: List[int], 
    total_runs: int,
    is_regression: bool = True
):
    """
    Run selection pipeline for a dataset using specified n_features values [8, 16, 32].
    
    Parameters
    ----------
    ds_name : str
        Dataset name
    data_modalities : Dict[str, pd.DataFrame]
        Dictionary of modality DataFrames
    common_ids : List[str]
        List of common sample IDs
    y : np.ndarray
        Target values
    base_out : str
        Base output directory
    selectors : Dict[str, str]
        Dictionary of selectors
    n_feats_list : List[int]
        List of n_features values to test (typically [8, 16, 32])
    models : List[str]
        List of model names to train
    progress_count : List[int]
        Progress counter [0]
    total_runs : int
        Total number of runs
    is_regression : bool
        Whether this is a regression task
    """
    logger.info(f"Using specified n_features values for {ds_name} selection: {n_feats_list}")
    
    _run_pipeline(
        ds_name=ds_name, 
        data_modalities=data_modalities, 
        common_ids=common_ids, 
        y=y, 
        base_out=base_out,
        transformers=selectors, 
        n_trans_list=n_feats_list,  # Use specified n_features list [8, 16, 32]
        models=models,
        progress_count=progress_count, 
        total_runs=total_runs,
        is_regression=is_regression, 
        pipeline_type="selection"
    )

def merge_small_classes(y: np.ndarray, min_samples: int = 2) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Merge small classes into the nearest larger class based on sample counts.
    
    Parameters
    ----------
    y : np.ndarray
        Class labels
    min_samples : int
        Minimum number of samples required to keep a class separate
        
    Returns
    -------
    Tuple[np.ndarray, Dict[int, int]]
        Tuple of (merged labels, mapping from old to new labels)
    """
    unique_classes, class_counts = np.unique(y, return_counts=True)
    n_classes = len(unique_classes)
    
    if n_classes <= 2:
        return y, {old: old for old in unique_classes}
    
    # Sort classes by count
    sorted_indices = np.argsort(class_counts)
    sorted_classes = unique_classes[sorted_indices]
    sorted_counts = class_counts[sorted_indices]
    
    # Initialize mapping
    label_mapping = {old: old for old in unique_classes}
    
    # Find classes to merge
    small_classes = sorted_classes[sorted_counts < min_samples]
    if len(small_classes) == 0:
        return y, label_mapping
    
    # For each small class, find the nearest larger class
    for small_class in small_classes:
        # Find the closest larger class
        larger_classes = sorted_classes[sorted_counts >= min_samples]
        if len(larger_classes) == 0:
            # If no larger classes, merge with the largest small class
            largest_small = sorted_classes[-1]
            label_mapping[small_class] = largest_small
        else:
            # Merge with the most similar class (using class index as similarity)
            closest_class = larger_classes[np.argmin(np.abs(larger_classes - small_class))]
            label_mapping[small_class] = closest_class
    
    # Apply mapping
    merged_y = np.array([label_mapping[label] for label in y])
    
    return merged_y, label_mapping

def cross_validate_model(X, y, model_name, n_splits=5, random_state=42, out_dir=None, make_plots=True, n_features=None, train_n_components=None):
    """
    Perform cross-validation for a classification model.
    """
    from sklearn.model_selection import StratifiedKFold, KFold
    from sklearn.metrics import accuracy_score, roc_auc_score
    import numpy as np
    import os
    import warnings
    
    # Check if stratification is viable
    unique_classes, class_counts = np.unique(y, return_counts=True)
    min_class_count = np.min(class_counts)
    
    # Initialize cross-validation with appropriate strategy
    if min_class_count >= n_splits + 1:  # +1 for safety buffer
        # Use StratifiedKFold with warning suppression
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", 
                                  message="The least populated class in y has only .* members, which is less than n_splits=.*", 
                                  category=UserWarning)
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.model_selection._split")
            
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            logger.debug(f"Using StratifiedKFold for cross_validate_model")
    else:
        # Fall back to KFold
        skf = KFold(n_splits=min(n_splits, len(y) // 2), shuffle=True, random_state=random_state)
        logger.debug(f"Using KFold for cross_validate_model (insufficient samples for stratification)")
    
    # Initialize metrics storage
    accuracies = []
    aucs = []
    models = []
    y_true_all = []
    y_pred_all = []
    
    # Perform cross-validation
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        # Split data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Create plot prefix
        plot_prefix = f"fold_{fold_idx}"
        if out_dir:
            plot_prefix = os.path.join(out_dir, plot_prefix)
        
        # Train model
        model, metrics, y_val_fold, y_pred_fold = train_classification_model(
            X_train, y_train, X_val, y_val,
            model_name=model_name,
            out_dir=out_dir,
            plot_prefix=plot_prefix,
            fold_idx=fold_idx,
            make_plots=make_plots,
            n_features=n_features,
            train_n_components=train_n_components,
            extractor_name=None  # This function doesn't have extractor context
        )
        
        if model is not None:
            # Store metrics
            accuracies.append(metrics['accuracy'])
            aucs.append(metrics['auc'])
            models.append(model)
            y_true_all.extend(y_val_fold)
            y_pred_all.extend(y_pred_fold)
        else:
            logger.warning(f"Warning: Failed to train {model_name} in fold {fold_idx}")
    
    # Calculate average metrics
    if accuracies:
        avg_accuracy = np.mean(accuracies)
        avg_auc = np.mean(aucs)
        std_accuracy = np.std(accuracies)
        std_auc = np.std(aucs)
        
        logger.info(f"Cross-validation results for {model_name}:")
        logger.info(f"Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
        logger.info(f"Average AUC: {avg_auc:.4f} ± {std_auc:.4f}")
        
        return {
            'models': models,
            'accuracies': accuracies,
            'aucs': aucs,
            'avg_accuracy': avg_accuracy,
            'std_accuracy': std_accuracy,
            'avg_auc': avg_auc,
            'std_auc': std_auc,
            'y_true': y_true_all,
            'y_pred': y_pred_all
        }
    else:
        logger.warning(f"No models were successfully trained for {model_name}")
        return None

# ============================================================================
# ENHANCED CROSS-VALIDATION STRATEGIES
# ============================================================================

def extract_patient_ids_from_samples(sample_ids: List[str]) -> List[str]:
    """
    Extract patient IDs from sample IDs for grouped cross-validation.
    
    For TCGA data, sample IDs typically have format: TCGA-XX-XXXX-XXX
    where the first 12 characters (TCGA-XX-XXXX) identify the patient.
    
    Parameters
    ----------
    sample_ids : List[str]
        List of sample IDs
        
    Returns
    -------
    List[str]
        List of patient IDs (same length as sample_ids)
    """
    patient_ids = []
    
    for sample_id in sample_ids:
        if isinstance(sample_id, str) and sample_id.startswith("TCGA"):
            # Extract patient ID from TCGA format: TCGA-XX-XXXX
            parts = sample_id.split("-")
            if len(parts) >= 3:
                patient_id = "-".join(parts[:3])  # TCGA-XX-XXXX
                patient_ids.append(patient_id)
            else:
                # Fallback if format is unexpected
                patient_ids.append(sample_id)
        else:
            # For non-TCGA data, use the full sample ID as patient ID
            patient_ids.append(sample_id)
    
    return patient_ids

def create_stratified_regression_bins(y: np.ndarray, n_bins: int = 4, min_samples_per_bin: int = 2) -> Tuple[np.ndarray, bool]:
    """
    Create stratified bins for regression targets to enable stratified K-fold.
    
    For AML blast percentage, this bins the continuous values into quartiles
    to ensure each fold has a comparable range of target values.
    
    Parameters
    ----------
    y : np.ndarray
        Continuous target values
    n_bins : int, default=4
        Number of bins to create (quartiles by default)
    min_samples_per_bin : int, default=2
        Minimum samples required per bin for stratification to be viable
        
    Returns
    -------
    Tuple[np.ndarray, bool]
        Binned target values and whether stratification is viable
    """
    try:
        # Use quantile-based binning to ensure equal-sized bins
        bins = np.quantile(y, np.linspace(0, 1, n_bins + 1))
        
        # Handle edge case where all values are the same
        if len(np.unique(bins)) == 1:
            logger.warning("All target values are identical, stratification not viable")
            return np.zeros(len(y), dtype=int), False
        
        # Create bins using digitize (returns bin indices)
        binned_y = np.digitize(y, bins[1:-1])  # Exclude first and last bin edges
        
        # Ensure bins are 0-indexed and within valid range
        binned_y = np.clip(binned_y, 0, n_bins - 1)
        
        # Check bin distribution for stratification viability
        unique_bins, bin_counts = np.unique(binned_y, return_counts=True)
        min_bin_count = np.min(bin_counts)
        
        # Check if stratification is viable
        stratification_viable = (min_bin_count >= min_samples_per_bin)
        
        if not stratification_viable:
            logger.warning(f"Stratification not viable: smallest bin has {min_bin_count} samples "
                          f"(minimum required: {min_samples_per_bin})")
            logger.debug(f"Bin distribution: bins={unique_bins}, counts={bin_counts}")
        else:
            logger.debug(f"Target stratification viable: {len(unique_bins)} bins with counts {bin_counts}")
        
        return binned_y, stratification_viable
        
    except Exception as e:
        logger.warning(f"Failed to create stratified bins: {e}")
        return y, False

def check_classification_stratification_viability(y: np.ndarray, n_splits: int = 3, min_samples_per_class: int = 2) -> bool:
    """
    Check if classification stratification is viable given the class distribution.
    
    Parameters
    ----------
    y : np.ndarray
        Classification target values
    n_splits : int
        Number of CV splits planned
    min_samples_per_class : int
        Minimum samples per class required for stratification
        
    Returns
    -------
    bool
        True if stratification is viable, False otherwise
    """
    unique_classes, class_counts = np.unique(y, return_counts=True)
    min_class_count = np.min(class_counts)
    
    # Need at least min_samples_per_class AND at least n_splits samples per class
    # Add buffer of 1 to be more conservative and avoid edge cases
    required_min = max(min_samples_per_class, n_splits + 1)  # +1 for safety buffer
    stratification_viable = (min_class_count >= required_min)
    
    if not stratification_viable:
        logger.info(f"Classification stratification not viable: smallest class has {min_class_count} samples "
                   f"(minimum required: {required_min} for {n_splits} splits). Using KFold instead.")
        logger.debug(f"Class distribution: classes={unique_classes}, counts={class_counts}")
    else:
        logger.debug(f"Classification stratification viable: {len(unique_classes)} classes with counts {class_counts}")
    
    return stratification_viable

def create_enhanced_cv_splitter(y: np.ndarray, 
                              sample_ids: Optional[List[str]] = None,
                              task_type: str = 'regression',
                              n_splits: int = 5,
                              use_stratified_regression: bool = True,
                              use_grouped_cv: bool = True,
                              random_state: int = 42) -> Tuple[Any, str]:
    """
    Create enhanced cross-validation splitter with stratified regression and grouped CV.
    
    Parameters
    ----------
    y : np.ndarray
        Target values
    sample_ids : List[str], optional
        Sample IDs for extracting patient groups
    task_type : str
        'regression' or 'classification'
    n_splits : int
        Number of CV folds
    use_stratified_regression : bool
        Whether to use stratified bins for regression
    use_grouped_cv : bool
        Whether to use grouped CV for patient replicates
    random_state : int
        Random state for reproducibility
        
    Returns
    -------
    Tuple[cv_splitter, strategy_description]
        CV splitter object and description of strategy used
    """
    import warnings
    from sklearn.model_selection import (
        KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold
    )
    
    # Suppress sklearn stratification warnings during CV setup - capture all relevant warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", 
                              message="The least populated class in y has only .* members, which is less than n_splits=.*", 
                              category=UserWarning)
        warnings.filterwarnings("ignore", 
                              message="The least populated class in y has only .* members", 
                              category=UserWarning)
        # Also suppress the module-specific warnings from sklearn
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.model_selection._split")
        
        return _create_enhanced_cv_splitter_impl(y, sample_ids, task_type, n_splits, 
                                                use_stratified_regression, use_grouped_cv, random_state)

def _create_enhanced_cv_splitter_impl(y: np.ndarray, 
                                    sample_ids: Optional[List[str]] = None,
                                    task_type: str = 'regression',
                                    n_splits: int = 5,
                                    use_stratified_regression: bool = True,
                                    use_grouped_cv: bool = True,
                                    random_state: int = 42):
    """Implementation of enhanced CV splitter with warnings suppressed."""
    from sklearn.model_selection import (
        KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold
    )
    
    n_samples = len(y)
    strategy_parts = []
    
    # Determine if we have patient groups
    groups = None
    if use_grouped_cv and sample_ids is not None:
        patient_ids = extract_patient_ids_from_samples(sample_ids)
        unique_patients = np.unique(patient_ids)
        
        # Only use grouped CV if we have multiple samples per patient
        if len(unique_patients) < len(sample_ids):
            groups = np.array([list(unique_patients).index(pid) for pid in patient_ids])
            n_groups = len(unique_patients)
            
            # Adjust n_splits if we have fewer groups than requested splits
            n_splits = min(n_splits, n_groups)
            strategy_parts.append(f"Grouped({n_groups} patients)")
            
            logger.info(f"Detected {len(sample_ids) - len(unique_patients)} patient replicates")
            logger.info(f"Using grouped CV with {n_groups} patient groups")
        else:
            logger.debug("No patient replicates detected, skipping grouped CV")
    
    # Handle regression vs classification
    if task_type == 'regression':
        if use_stratified_regression:
            # Create stratified bins for regression and check viability
            y_binned, stratification_viable = create_stratified_regression_bins(y, n_bins=4, min_samples_per_bin=n_splits)
            
            if stratification_viable:
                strategy_parts.append("Stratified(quartiles)")
                
                if groups is not None:
                    # Grouped + Stratified
                    cv_splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                    strategy_parts.append("StratifiedGroupKFold")
                    strategy_desc = " + ".join(strategy_parts)
                    
                    logger.info(f"Using {strategy_desc} for regression CV")
                    return cv_splitter, strategy_desc, y_binned, groups
                else:
                    # Stratified only with warning suppression
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", 
                                              message="The least populated class in y has only .* members, which is less than n_splits=.*", 
                                              category=UserWarning)
                        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.model_selection._split")
                        
                        cv_splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                        strategy_desc = "StratifiedKFold(quartiles)"
                        
                        logger.info(f"Using {strategy_desc} for regression CV")
                        return cv_splitter, strategy_desc, y_binned, None
            else:
                logger.info("Stratified regression not viable, falling back to non-stratified CV")
        
        # Fallback for regression: GroupKFold or regular KFold
        if groups is not None:
            cv_splitter = GroupKFold(n_splits=n_splits)
            strategy_desc = "GroupKFold"
            logger.info(f"Using {strategy_desc} for regression CV")
            return cv_splitter, strategy_desc, y, groups
        else:
            cv_splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            strategy_desc = "KFold"
            logger.info(f"Using {strategy_desc} for regression CV")
            return cv_splitter, strategy_desc, y, None
    
    else:  # Classification
        # Check if stratification is viable for classification
        stratification_viable = check_classification_stratification_viability(y, n_splits=n_splits)
        
        if stratification_viable:
            if groups is not None:
                # Grouped + Stratified for classification
                cv_splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                strategy_desc = "StratifiedGroupKFold"
                
                logger.info(f"Using {strategy_desc} for classification CV")
                return cv_splitter, strategy_desc, y, groups
            else:
                                    # Standard stratified classification with warning suppression
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", 
                                              message="The least populated class in y has only .* members, which is less than n_splits=.*", 
                                              category=UserWarning)
                        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.model_selection._split")
                        
                        cv_splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                        strategy_desc = "StratifiedKFold"
                        
                        logger.info(f"Using {strategy_desc} for classification CV")
                        return cv_splitter, strategy_desc, y, None
        else:
            logger.info("Classification stratification not viable, falling back to non-stratified CV")
            
            # Fallback to non-stratified CV
            if groups is not None:
                cv_splitter = GroupKFold(n_splits=n_splits)
                strategy_desc = "GroupKFold (stratification not viable)"
                logger.info(f"Using {strategy_desc} for classification CV")
                return cv_splitter, strategy_desc, y, groups
            else:
                cv_splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                strategy_desc = "KFold (stratification not viable)"
                logger.info(f"Using {strategy_desc} for classification CV")
                return cv_splitter, strategy_desc, y, None

def validate_enhanced_cv_strategy(cv_splitter, y_for_cv, groups, n_splits, task_type):
    """
    Validate that the enhanced CV strategy will work properly.
    
    Parameters
    ----------
    cv_splitter : sklearn CV splitter
        The cross-validation splitter
    y_for_cv : np.ndarray
        Target values (potentially binned for stratified regression)
    groups : np.ndarray or None
        Group labels for grouped CV
    n_splits : int
        Expected number of splits
    task_type : str
        'regression' or 'classification'
        
    Returns
    -------
    bool
        True if validation passes, False otherwise
    """
    import warnings
    
    # Suppress sklearn stratification warnings during validation - comprehensive suppression
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", 
                              message="The least populated class in y has only .* members, which is less than n_splits=.*", 
                              category=UserWarning)
        warnings.filterwarnings("ignore", 
                              message="The least populated class in y has only .* members", 
                              category=UserWarning)
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.model_selection._split")
        
        try:
            # Test the splitter
            if groups is not None:
                splits = list(cv_splitter.split(np.zeros((len(y_for_cv), 1)), y_for_cv, groups))
            else:
                splits = list(cv_splitter.split(np.zeros((len(y_for_cv), 1)), y_for_cv))
        
            actual_splits = len(splits)
            if actual_splits != n_splits:
                logger.warning(f"CV validation: expected {n_splits} splits, got {actual_splits}")
                return False
            
            # Check fold sizes
            fold_sizes = [len(train_idx) + len(val_idx) for train_idx, val_idx in splits]
            min_fold_size = min(fold_sizes)
            max_fold_size = max(fold_sizes)
            
            if min_fold_size < 10:
                logger.warning(f"CV validation: very small fold detected ({min_fold_size} samples)")
                return False
            
            if max_fold_size / min_fold_size > 2.0:
                logger.warning(f"CV validation: unbalanced folds (sizes: {min_fold_size}-{max_fold_size})")
            
            # Check class distribution for classification
            if task_type == 'classification':
                for fold_idx, (train_idx, val_idx) in enumerate(splits):
                    y_train_fold = y_for_cv[train_idx]
                    y_val_fold = y_for_cv[val_idx]
                    
                    # Check if all classes are represented in training
                    train_classes = set(y_train_fold)
                    val_classes = set(y_val_fold)
                    
                    if len(train_classes) < 2:
                        logger.warning(f"CV validation: fold {fold_idx} training set has only {len(train_classes)} classes")
                        return False
                    
                    if len(val_classes) < 1:
                        logger.warning(f"CV validation: fold {fold_idx} validation set has no samples")
                        return False
            
            logger.debug(f"CV validation passed: {actual_splits} folds, sizes {min_fold_size}-{max_fold_size}")
            return True
            
        except Exception as e:
            logger.error(f"CV validation failed: {e}")
            return False