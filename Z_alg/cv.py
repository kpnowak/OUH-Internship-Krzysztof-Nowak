#!/usr/bin/env python3
"""
Cross-validation module for model training and evaluation.
"""

import os
import numpy as np
import pandas as pd
import warnings
from joblib import Parallel, delayed, parallel_config
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
from sklearn.base import clone

# Local imports
from Z_alg.config import N_JOBS, DatasetConfig, FEATURE_EXTRACTION_CONFIG
from Z_alg.preprocessing import process_with_missing_modalities
from Z_alg.fusion import merge_modalities, ModalityImputer
from Z_alg.models import (
    get_model_object, cached_fit_transform_selector_regression,
    transform_selector_regression, cached_fit_transform_selector_classification,
    transform_selector_classification, cached_fit_transform_extractor_classification,
    transform_extractor_classification, get_selector_object
)
from Z_alg.utils import TParallel, heavy_cpu_section
from Z_alg._process_single_modality import align_samples_to_modalities, verify_data_alignment

# Initialize logger
logger = logging.getLogger(__name__)

# 1. Add a threshold for severe alignment loss at the top of the file:
SEVERE_ALIGNMENT_LOSS_THRESHOLD = 0.3  # 30%
MIN_SAMPLES_PER_FOLD = 5

# Add improved CV configuration at the top of the file
CV_CONFIG = {
    "min_samples_per_class_per_fold": 2,  # Minimum samples per class in each fold
    "min_total_samples_for_stratified": 10,  # Minimum total samples to attempt stratified CV
    "min_samples_per_fold": 5,  # Minimum total samples per fold
    "max_cv_splits": 5,  # Maximum number of CV splits
    "min_cv_splits": 2,  # Minimum number of CV splits
    "adaptive_min_samples": True,  # Adapt minimum samples based on dataset size
}

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
    is_regression: bool = True
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Process a single data modality within a CV fold.
    
    Parameters
    ----------
    modality_name : str
        Name of the modality
    modality_df : pd.DataFrame
        Modality data, with rows=genes/features, columns=samples
    id_train : List[str]
        Training sample IDs (pre-filtered to be available in this modality)
    id_val : List[str]
        Validation sample IDs (pre-filtered to be available in this modality)
    idx_test : np.ndarray
        Test indices
    y_train : np.ndarray
        Training labels/target values (pre-filtered to match id_train exactly)
    extr_obj : Any
        Extractor object, or name of extractor
    ncomps : int
        Number of components to extract
    idx_to_id : Dict[int, str]
        Mapping from index to sample ID
    fold_idx : Optional[int]
        CV fold index
    is_regression : bool
        Whether this is a regression task
        
    Returns
    -------
    Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]
        Tuple of (train, validation, test) arrays
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
        from Z_alg.models import (
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
                from Z_alg.models import cached_fit_transform_extractor_regression, transform_extractor_regression
                extractor, X_tr = cached_fit_transform_extractor_regression(
                    df_train.values, y_train, extr_obj, req, 
                    ds_name=modality_name, fold_idx=fold_idx
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
                extractor, X_tr = cached_fit_transform_extractor_classification(
                    df_train.values, y_train, extr_obj, req, 
                    modality_name=modality_name, fold_idx=fold_idx
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
            logger.error(f"CRITICAL: Output alignment failed in {modality_name} fold {fold_idx}: expected {len(id_train)}, got {X_tr.shape[0]}")
            return None, None, None
        
        if X_va.shape[0] != len(id_val):
            logger.error(f"CRITICAL: Output val alignment failed in {modality_name} fold {fold_idx}: expected {len(id_val)}, got {X_va.shape[0]}")
            return None, None, None
        
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
    plot_prefix_override=None
):
    """
    Process a single CV fold, handling all modalities and models.
    
    Parameters
    ----------
    train_idx : ndarray
        Training indices
    val_idx : ndarray
        Validation indices
    idx_temp : ndarray
        Temporary indices (train+val)
    idx_test : ndarray
        Test indices
    y_temp : ndarray
        Temporary target values (train+val)
    y_test : ndarray
        Test target values
    data_modalities : Dict[str, pd.DataFrame]
        Dictionary of modality DataFrames
    models : List[str]
        List of model names to train
    extr_obj : Any
        Extractor object
    ncomps : int
        Number of components to extract
    id_to_idx : Dict[str, int]
        Mapping from sample ID to index
    idx_to_id : Dict[int, str]
        Mapping from index to sample ID
    all_ids : List[str]
        List of all sample IDs
    missing_percentage : float
        Percentage of data to mark as missing
    fold_idx : int
        CV fold index
    base_out : str
        Base output directory
    ds_name : str
        Dataset name
    extr_name : str
        Extractor name
    pipeline_type : str
        Type of pipeline ("extraction" or "selection")
    is_regression : bool
        Whether this is a regression task
    make_plots : bool
        Whether to generate plots for the best fold
    plot_prefix_override : Optional[str]
        Override for the plot prefix
        
    Returns
    -------
    Dict[str, Dict[str, float]]
        Dictionary of model results
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
        
        # Process modalities in parallel, using the EXACT common sample sets
        from joblib import Parallel, delayed
        from Z_alg.config import JOBLIB_PARALLEL_CONFIG
        
        # Limit number of jobs to avoid excessive resource usage
        n_jobs = min(3, os.cpu_count() or 1)
        logger.debug(f"Processing {len(modified_modalities)} modalities with {len(final_common_train)} train and {len(final_common_val)} val samples")
        
        modality_results = Parallel(n_jobs=n_jobs, **JOBLIB_PARALLEL_CONFIG)(
            delayed(_process_single_modality)(
                name, 
                df, 
                final_common_train,  # Use final common samples
                final_common_val,    # Use final common samples
                idx_test, 
                final_aligned_y_train,  # Use pre-filtered targets
                (clone(extr_obj) if hasattr(extr_obj, 'fit') else extr_obj),  # fresh instance
                ncomps, 
                idx_to_id,
                fold_idx,
                is_regression
            )
            for name, df in modified_modalities.items()
        )

        # Filter out None results - all should have the same dimensions now
        valid_results = []
        expected_train_samples = len(final_common_train)
        expected_val_samples = len(final_common_val)
        
        for i, r in enumerate(modality_results):
            if r is not None and len(r) >= 3:  # Check we have at least train, val, test
                X_train, X_val, X_test = r[0], r[1], r[2]
                if X_train is not None and X_val is not None:
                    if X_train.size > 0 and X_val.size > 0:
                        # Verify dimensions match expected
                        if X_train.shape[0] != expected_train_samples:
                            logger.error(f"Modality {i} train samples mismatch: expected {expected_train_samples}, got {X_train.shape[0]}")
                            continue
                        if X_val.shape[0] != expected_val_samples:
                            logger.error(f"Modality {i} val samples mismatch: expected {expected_val_samples}, got {X_val.shape[0]}")
                            continue
                        valid_results.append((X_train, X_val, X_test))
                        logger.debug(f"Modality {i} validated: Train {X_train.shape}, Val {X_val.shape}")
                    else:
                        logger.warning(f"Modality {i} returned empty arrays")
                else:
                    logger.warning(f"Modality {i} returned None arrays")
            else:
                logger.warning(f"Modality {i} returned invalid result")

        if not valid_results:
            logger.warning(f"No valid data found for any modality in fold {fold_idx}")
            return {}, {}
        
        # Create a new imputer instance for this fold
        from Z_alg.fusion import ModalityImputer
        fold_imputer = ModalityImputer()

        # Merge modalities - all should have identical dimensions now
        try:
            X_train_merged = merge_modalities(*[r[0] for r in valid_results], imputer=fold_imputer, is_train=True)
            X_val_merged = merge_modalities(*[r[1] for r in valid_results], imputer=fold_imputer, is_train=False)
            
            # Final verification: merged arrays should match the target vectors exactly
            if X_train_merged.shape[0] != len(final_aligned_y_train):
                logger.error(f"CRITICAL: Final train alignment failed: X={X_train_merged.shape[0]}, y={len(final_aligned_y_train)}")
                return {}, {}
            
            if X_val_merged.shape[0] != len(final_aligned_y_val):
                logger.error(f"CRITICAL: Final val alignment failed: X={X_val_merged.shape[0]}, y={len(final_aligned_y_val)}")
                return {}, {}
            
            # Skip if no valid data after merging
            if X_train_merged.size == 0 or X_val_merged.size == 0:
                logger.warning(f"No valid data after merging in fold {fold_idx}")
                return {}, {}
                
            # Skip if too few samples after merging
            if X_train_merged.shape[0] < MIN_SAMPLES_PER_FOLD:
                logger.warning(f"Skipping fold {fold_idx} for {ds_name}: too few samples after merging ({X_train_merged.shape[0]} < {MIN_SAMPLES_PER_FOLD})")
                return {}, {}
                
            logger.debug(f"Perfect alignment achieved in fold {fold_idx}: Train {X_train_merged.shape}, Val {X_val_merged.shape}")
            
            # Update the aligned target vectors for the rest of the processing
            aligned_y_train = final_aligned_y_train
            aligned_y_val = final_aligned_y_val
            
        except Exception as e:
            logger.error(f"Error merging modalities in fold {fold_idx}: {str(e)}")
            return {}, {}
        
        # No need for verify_data_alignment calls here since we ensured perfect alignment
        # The arrays should be perfectly aligned at this point
        
        # Save the number of features before extraction/selection
        original_n_features = ncomps  # This ensures n_features matches the intended value in metrics

        # Apply extraction/selection to merged data
        # --- ENFORCE ncomps after reduction ---
        if is_regression:
            # Extraction pipeline
            if pipeline_type == "extraction":
                from Z_alg.models import cached_fit_transform_extractor_regression, transform_extractor_regression
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
                from Z_alg.models import cached_fit_transform_selector_regression, transform_selector_regression
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
                from Z_alg.models import cached_fit_transform_extractor_classification, transform_extractor_classification
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
                from Z_alg.models import cached_fit_transform_selector_classification, transform_selector_classification
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
                # Perform one final verification before training the model
                # This is critical to ensure X and y have the same number of samples
                from Z_alg.models import validate_and_fix_shape_mismatch
                final_X_train, final_y_train = validate_and_fix_shape_mismatch(
                    final_X_train, aligned_y_train, 
                    name=f"training data for {model_name} (fold {fold_idx})", 
                    fold_idx=fold_idx
                )
                final_X_val, final_y_val = validate_and_fix_shape_mismatch(
                    final_X_val, aligned_y_val, 
                    name=f"validation data for {model_name} (fold {fold_idx})", 
                    fold_idx=fold_idx
                )
                
                # Only proceed if we have valid data
                if (final_X_train is None or final_y_train is None or 
                    final_X_val is None or final_y_val is None):
                    logger.warning(f"Invalid data for {model_name} in fold {fold_idx}")
                    continue
                
                # For classification, check class distribution in this specific fold
                if not is_regression:
                    # Use adaptive minimum samples based on dataset size (consistent with CV splitter)
                    # Always require at least 2 samples per class for scikit-learn compatibility
                    if CV_CONFIG["adaptive_min_samples"]:
                        total_samples = len(final_y_train) + len(final_y_val)
                        if total_samples < 10:
                            adaptive_min_samples = 2  # Very small datasets - still need 2 for sklearn
                        elif total_samples < 20:
                            adaptive_min_samples = 2  # Small datasets - still need 2 for sklearn
                        else:
                            adaptive_min_samples = 2  # Use 2 for all datasets for sklearn compatibility
                    else:
                        adaptive_min_samples = CV_CONFIG["min_samples_per_class_per_fold"]
                    
                    train_mask, val_mask, y_train_filtered, y_val_filtered, label_mapping = check_and_filter_classes_in_fold(
                        final_y_train, final_y_val, min_samples_per_class=adaptive_min_samples
                    )
                    
                    if train_mask is None:
                        logger.warning(f"Skipping {model_name} in fold {fold_idx}: insufficient class distribution "
                                     f"(min_samples_per_class={adaptive_min_samples})")
                        continue
                    
                    # Filter the feature matrices to match the filtered labels
                    final_X_train = final_X_train[train_mask]
                    final_X_val = final_X_val[val_mask]
                    final_y_train = y_train_filtered
                    final_y_val = y_val_filtered
                    
                    # Additional validation after filtering
                    if len(np.unique(final_y_train)) < 2:
                        logger.warning(f"Skipping {model_name} in fold {fold_idx}: insufficient classes after filtering ({len(np.unique(final_y_train))} classes)")
                        continue
                    
                    if final_X_train.shape[0] < 2:
                        logger.warning(f"Skipping {model_name} in fold {fold_idx}: insufficient training samples after filtering ({final_X_train.shape[0]} samples)")
                        continue
                    
                    logger.debug(f"Filtered data for {model_name} in fold {fold_idx}: Train {final_X_train.shape}, Val {final_X_val.shape}, Classes: {len(np.unique(final_y_train))}")
                
                # Train and evaluate
                if is_regression:
                    if plot_prefix_override:
                        current_plot_prefix = plot_prefix_override
                    else:
                        current_plot_prefix = f"{ds_name}_fold_{fold_idx}_{extr_name}_{ncomps}_{model_name}"
                    model, metrics = train_model(
                        final_X_train, final_y_train, 
                        final_X_val, final_y_val,
                        model_name, 
                        out_dir=os.path.join(base_out, "plots"),
                        plot_prefix=current_plot_prefix,
                        fold_idx=fold_idx,
                        make_plots=make_plots,
                        n_features=original_n_features,
                        train_n_components=train_n_components
                    )
                    model_results[model_name] = metrics
                    model_objects[model_name] = model
                else:
                    if plot_prefix_override:
                        current_plot_prefix = plot_prefix_override
                    else:
                        current_plot_prefix = f"{ds_name}_fold_{fold_idx}_{extr_name}_{ncomps}_{model_name}"
                    model, metrics, y_val_out, y_pred_out = train_model(
                        final_X_train, final_y_train, 
                        final_X_val, final_y_val,
                        model_name, 
                        out_dir=os.path.join(base_out, "plots"),
                        plot_prefix=current_plot_prefix,
                        fold_idx=fold_idx,
                        make_plots=make_plots,
                        n_features=original_n_features,
                        train_n_components=train_n_components
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

def _run_pipeline(
    ds_name: str, 
    data_modalities: Dict[str, pd.DataFrame], 
    common_ids: List[str], 
    y: np.ndarray, 
    base_out: str,
    transformers: Dict[str, Any], 
    n_trans_list: List[int], 
    models: List[str],
    progress_count: List[int], 
    total_runs: int,
    is_regression: bool = True, 
    pipeline_type: str = "extraction"
):
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
    n_trans_list : List[int]
        List of parameters for transformers (n_components or n_features)
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
    # Ensure output directories exist
    os.makedirs(base_out, exist_ok=True)
    for subdir in ["models", "metrics", "plots"]:
        os.makedirs(os.path.join(base_out, subdir), exist_ok=True)
    
    # Convert y to numpy array
    y_arr = np.array(y)
    
    # Warn if sample size is very small
    if len(y_arr) < 30:
        logger.warning(f"Sample size for {ds_name} is very small ({len(y_arr)} samples). Model performance may be unstable.")
    
    # Create indices array for row-based indexing
    indices = np.arange(len(common_ids))
    
    # Create id to index and index to id mappings
    id_to_idx = {id_: idx for idx, id_ in enumerate(common_ids)}
    idx_to_id = {idx: id_ for id_, idx in id_to_idx.items()}
    
    # Split with stratification if possible
    try:
        # For small datasets (< 15 samples), use a larger test proportion to ensure some test samples
        test_size = 0.3 if len(indices) < 15 else 0.2
        # For regression, bin continuous values for stratified split
        if is_regression:
            n_bins = min(5, len(y_arr)//3)
            unique_vals = len(np.unique(y_arr))
            if n_bins < 2 or unique_vals < n_bins or len(indices) < 15:
                logger.info(f"Skipping stratification for regression: n_bins={n_bins}, unique_vals={unique_vals}, sample_size={len(indices)}")
                raise ValueError("Too few samples or unique values for stratified split in regression")
            y_bins = pd.qcut(y_arr, n_bins, labels=False, duplicates='drop')
            idx_temp, idx_test, y_temp, y_test = train_test_split(
                indices, y_arr, test_size=test_size, random_state=0, stratify=y_bins
            )
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
    from Z_alg.config import MISSING_MODALITIES_CONFIG
    
    # Save all results for batch processing
    all_pipeline_results = []
    
    for trans_name, trans_obj in transformers.items():
        for n_val in n_trans_list:
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

                # Use the new robust CV splitter
                cv_splitter, n_splits, cv_type_used = create_robust_cv_splitter(idx_temp, y_temp, is_regression)
                
                logger.info(f"Dataset: {ds_name}, using {cv_type_used} {n_splits}-fold CV with {len(idx_temp)} training samples")
                
                # Log detailed CV summary for classification tasks
                if not is_regression:
                    log_cv_fold_summary(ds_name, y_temp, cv_splitter, cv_type_used, n_splits)
                
                # Results storage
                pipeline_results = []
                
                # Process each missing percentage
                for missing_percentage in MISSING_MODALITIES_CONFIG["missing_percentages"]:
                    try:
                        cv_results = []
                        model_candidates = {model_name: {"metric": None, "model": None, "fold_idx": None, "train_val": None} for model_name in models}
                        model_yvals_folds = {model_name: [] for model_name in models}
                        model_ypreds_folds = {model_name: [] for model_name in models}
                        train_val_data = []  # Store (train_idx, val_idx, ...) for each fold
                        
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
                                                make_plots=False
                                            )
                                        else:
                                            result, model_objs, yvals, ypreds = process_cv_fold(
                                                train_idx, val_idx, idx_temp, idx_test, y_temp, y_test,
                                                data_modalities, models, (clone(trans_obj) if hasattr(trans_obj, 'fit') else trans_obj), n_val, 
                                                id_to_idx, idx_to_id, common_ids, missing_percentage,
                                                fold_idx, base_out, ds_name, trans_name, pipeline_type,
                                                is_regression,
                                                make_plots=False
                                            )
                                    else:
                                        if is_regression:
                                            result, model_objs = process_cv_fold(
                                                train_idx, val_idx, idx_temp, idx_test, y_temp, y_test,
                                                data_modalities, models, (clone(trans_obj) if hasattr(trans_obj, 'fit') else trans_obj), n_val, 
                                                id_to_idx, idx_to_id, common_ids, missing_percentage,
                                                fold_idx, base_out, ds_name, trans_name, pipeline_type,
                                                is_regression,
                                                make_plots=False
                                            )
                                        else:
                                            result, model_objs, yvals, ypreds = process_cv_fold(
                                                train_idx, val_idx, idx_temp, idx_test, y_temp, y_test,
                                                data_modalities, models, (clone(trans_obj) if hasattr(trans_obj, 'fit') else trans_obj), n_val, 
                                                id_to_idx, idx_to_id, common_ids, missing_percentage,
                                                fold_idx, base_out, ds_name, trans_name, pipeline_type,
                                                is_regression,
                                                make_plots=False
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
                                logger.error(f"Error processing fold {fold_idx} for {ds_name} with {trans_name}-{n_val} (missing={missing_percentage}): {str(e)}")
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
                                        # Use a modified plot prefix with "best_fold", pipeline_type, and missing_percentage
                                        best_plot_prefix = f"{ds_name}_best_fold_{pipeline_type}_{trans_name}_{n_val}_{model_name}_{missing_percentage}"
                                        
                                        best_fold_results, best_model_obj = process_cv_fold(
                                            train_idx, val_idx, idx_temp, idx_test, y_temp, y_test,
                                            data_modalities, [model_name], (clone(trans_obj) if hasattr(trans_obj, 'fit') else trans_obj), n_val, 
                                            id_to_idx, idx_to_id, common_ids, missing_percentage,
                                            best_fold_idx, base_out, ds_name, trans_name, pipeline_type,
                                            is_regression,
                                            make_plots=True,
                                            plot_prefix_override=best_plot_prefix
                                        )
                                    else:
                                        # Use a modified plot prefix with "best_fold", pipeline_type, and missing_percentage
                                        best_plot_prefix = f"{ds_name}_best_fold_{pipeline_type}_{trans_name}_{n_val}_{model_name}_{missing_percentage}"
                                        
                                        best_fold_results, best_model_obj, _, _ = process_cv_fold(
                                            train_idx, val_idx, idx_temp, idx_test, y_temp, y_test,
                                            data_modalities, [model_name], (clone(trans_obj) if hasattr(trans_obj, 'fit') else trans_obj), n_val, 
                                            id_to_idx, idx_to_id, common_ids, missing_percentage,
                                            best_fold_idx, base_out, ds_name, trans_name, pipeline_type,
                                            is_regression,
                                            make_plots=True,
                                            plot_prefix_override=best_plot_prefix
                                        )
                                    
                                    # Store best fold metrics for this model
                                    if model_name in best_fold_results:
                                        best_fold_metrics[model_name] = best_fold_results[model_name].copy()
                                        best_fold_metrics[model_name]['best_fold_idx'] = best_fold_idx
                                    
                                    # Save the best model
                                    model_path = os.path.join(
                                        base_out, "models",
                                        f"best_model_{pipeline_type}_{model_name}_{trans_name}_{n_val}_{missing_percentage}.pkl"
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
                                logger.error(f"Error processing best fold for model {model_name} with {trans_name}-{n_val} (missing={missing_percentage}): {str(e)}")
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
    Run extraction pipeline for a dataset.
    
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
        List of n_components values
    models : List[str]
        List of model names to train
    progress_count : List[int]
        Progress counter [0]
    total_runs : int
        Total number of runs
    is_regression : bool
        Whether this is a regression task
    """
    _run_pipeline(
        ds_name=ds_name, 
        data_modalities=data_modalities, 
        common_ids=common_ids, 
        y=y, 
        base_out=base_out,
        transformers=extractors, 
        n_trans_list=n_comps_list, 
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
    Run selection pipeline for a dataset.
    
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
        List of n_features values
    models : List[str]
        List of model names to train
    progress_count : List[int]
        Progress counter [0]
    total_runs : int
        Total number of runs
    is_regression : bool
        Whether this is a regression task
    """
    _run_pipeline(
        ds_name=ds_name, 
        data_modalities=data_modalities, 
        common_ids=common_ids, 
        y=y, 
        base_out=base_out,
        transformers=selectors, 
        n_trans_list=n_feats_list, 
        models=models,
        progress_count=progress_count, 
        total_runs=total_runs,
        is_regression=is_regression, 
        pipeline_type="selection"
    )

def train_regression_model(X_train, y_train, X_val, y_val, model_name, out_dir, plot_prefix, fold_idx=None, make_plots=True, n_features=None, train_n_components=None):
    """Train regression model and evaluate it."""
    from Z_alg.models import get_model_object
    from Z_alg.plots import plot_regression_scatter, plot_regression_residuals, plot_feature_importance
    import os
    import numpy as np
    import time
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Data should already be aligned by process_cv_fold, so just do basic validation
    if X_train is None or y_train is None or X_val is None or y_val is None:
        logger.warning(f"Invalid data for {model_name} in fold {fold_idx}")
        return None, {}
    
    # Basic sanity check without redundant alignment
    if X_train.shape[0] != len(y_train):
        logger.error(f"Data alignment error for {model_name} in fold {fold_idx}: X_train={X_train.shape[0]}, y_train={len(y_train)}")
        return None, {}
    
    if X_val.shape[0] != len(y_val):
        logger.error(f"Data alignment error for {model_name} in fold {fold_idx}: X_val={X_val.shape[0]}, y_val={len(y_val)}")
        return None, {}

    # Create the model (with early stopping enabled by default)
    model = get_model_object(model_name)
    
    # Train the model with timing
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    
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
    else:
        early_stopping_info = {
            'early_stopping_used': False
        }
    
    # Make predictions
    y_pred = model.predict(X_val)
    
    # Calculate metrics
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    # Use passed train_n_components if provided, else fallback to X_train.shape[1]
    if train_n_components is None:
        train_n_components = X_train.shape[1]
    
    # Create plots
    if out_dir and make_plots:
        os.makedirs(out_dir, exist_ok=True)
        plot_regression_scatter(y_val, y_pred, f"{model_name} Scatter", os.path.join(out_dir, f"{plot_prefix}_scatter.png"))
        plot_regression_residuals(y_val, y_pred, f"{model_name} Residuals", os.path.join(out_dir, f"{plot_prefix}_residuals.png"))
        # Feature importance plot
        if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
            if hasattr(X_train, 'columns'):
                feat_names = list(X_train.columns)
            else:
                feat_names = [f"Feature {i}" for i in range(X_train.shape[1])]
            plot_feature_importance(model, feat_names, f"{model_name} Feature Importance", os.path.join(out_dir, f"{plot_prefix}_featimp.png"))
    
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


def train_classification_model(X_train, y_train, X_val, y_val, model_name, out_dir, plot_prefix, fold_idx=None, make_plots=True, n_features=None, train_n_components=None):
    """Train classification model and evaluate it."""
    from Z_alg.models import get_model_object
    from Z_alg.plots import plot_confusion_matrix, plot_roc_curve_binary, plot_feature_importance
    import os
    import numpy as np
    import time
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, matthews_corrcoef
    
    # Data should already be aligned by process_cv_fold, so just do basic validation
    if X_train is None or y_train is None or X_val is None or y_val is None:
        logger.warning(f"Invalid data for {model_name} in fold {fold_idx}")
        return None, {}, None, None
    
    # Basic sanity check without redundant alignment
    if X_train.shape[0] != len(y_train):
        logger.error(f"Data alignment error for {model_name} in fold {fold_idx}: X_train={X_train.shape[0]}, y_train={len(y_train)}")
        return None, {}, None, None
    
    if X_val.shape[0] != len(y_val):
        logger.error(f"Data alignment error for {model_name} in fold {fold_idx}: X_val={X_val.shape[0]}, y_val={len(y_val)}")
        return None, {}, None, None

    # Additional safety checks for classification
    n_train_samples = len(y_train)
    n_val_samples = len(y_val)
    n_train_classes = len(np.unique(y_train))
    n_val_classes = len(np.unique(y_val))
    
    # Check minimum sample requirements
    if n_train_samples < 2:
        logger.warning(f"Insufficient training samples for {model_name} in fold {fold_idx}: {n_train_samples} < 2")
        return None, {}, None, None
    
    if n_val_samples < 1:
        logger.warning(f"Insufficient validation samples for {model_name} in fold {fold_idx}: {n_val_samples} < 1")
        return None, {}, None, None
    
    if n_train_classes < 2:
        logger.warning(f"Insufficient classes in training data for {model_name} in fold {fold_idx}: {n_train_classes} < 2")
        return None, {}, None, None
    
    # Warn about very small datasets
    if n_train_samples < 10:
        logger.warning(f"Very small training set for {model_name} in fold {fold_idx}: {n_train_samples} samples")
    
    # Create the model (with early stopping enabled by default)
    model = get_model_object(model_name)
    
    # Train the model with timing
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    
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
    else:
        early_stopping_info = {
            'early_stopping_used': False
        }
    
    # Make predictions
    y_pred = model.predict(X_val)
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_val, y_pred)
    
    # Calculate MCC (Matthews Correlation Coefficient)
    mcc = matthews_corrcoef(y_val, y_pred)
    
    # Use passed train_n_components if provided, else fallback to X_train.shape[1]
    if train_n_components is None:
        train_n_components = X_train.shape[1]
    
    # Try to calculate ROC AUC if applicable (binary classification with proba method)
    auc = 0.5
    try:
        if hasattr(model, 'predict_proba') and len(np.unique(y_val)) == 2:
            y_score = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_score)
    except Exception as e:
        logger.warning(f"Could not calculate AUC: {str(e)}")
    
    # Create plots if directory is provided
    if out_dir and make_plots:
        os.makedirs(out_dir, exist_ok=True)
        
        # Confusion matrix
        class_labels = sorted(np.unique(np.concatenate([y_train, y_val])))
        plot_confusion_matrix(cm, class_labels, f"{model_name} Confusion Matrix", 
                             os.path.join(out_dir, f"{plot_prefix}_confusion.png"))
        
        # ROC curve for binary classification
        if hasattr(model, 'predict_proba') and len(np.unique(y_val)) == 2:
            plot_roc_curve_binary(model, X_val, y_val, class_labels, 
                                 f"{model_name} ROC Curve", 
                                 os.path.join(out_dir, f"{plot_prefix}_roc.png"))
        # Feature importance plot
        if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
            if hasattr(X_train, 'columns'):
                feat_names = list(X_train.columns)
            else:
                feat_names = [f"Feature {i}" for i in range(X_train.shape[1])]
            plot_feature_importance(model, feat_names, f"{model_name} Feature Importance", os.path.join(out_dir, f"{plot_prefix}_featimp.png"))
    
    # Return model, metrics, y_val, y_pred (including early stopping info)
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'mcc': mcc,
        'train_time': train_time,
        'n_features': n_features if n_features is not None else -1,  # Original feature count or -1 if unknown
        'train_n_components': train_n_components,  # Actual feature count used in training
        **early_stopping_info  # Include early stopping metrics
    }
    
    return model, metrics, y_val, y_pred 

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
    
    # Log detailed class distribution for debugging
    logger.debug(f"Training set class distribution: {dict(zip(unique_train, counts_train))}")
    logger.debug(f"Validation set class distribution: {dict(zip(unique_val, counts_val))}")
    
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
        logger.debug(f"Available classes: train={unique_train}, val={unique_val}, "
                    f"valid_train_classes={valid_train_classes}, common={common_classes}")
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
    
    logger.debug(f"Successfully filtered fold: {len(valid_classes)} classes, "
                f"{len(y_train_filtered)} train samples, {len(y_val_filtered)} val samples")
    
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

def create_robust_cv_splitter(idx_temp, y_temp, is_regression=False):
    """
    Create a robust CV splitter that ensures good class distribution.
    
    Parameters
    ----------
    idx_temp : np.ndarray
        Training indices
    y_temp : np.ndarray
        Training labels
    is_regression : bool
        Whether this is a regression task
        
    Returns
    -------
    tuple
        (cv_splitter, n_splits, cv_type_used)
    """
    n_samples = len(y_temp)
    optimal_splits = get_optimal_cv_splits(y_temp, is_regression)
    
    if is_regression:
        cv_splitter = KFold(n_splits=optimal_splits, shuffle=True, random_state=0)
        return cv_splitter, optimal_splits, "KFold"
    
    # For classification, try multiple strategies
    unique, counts = np.unique(y_temp, return_counts=True)
    n_classes = len(unique)
    min_class_count = np.min(counts)
    
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
    
    logger.debug(f"Creating CV splitter for classification: n_samples={n_samples}, "
                f"n_classes={n_classes}, min_class_count={min_class_count}, "
                f"adaptive_min_samples={adaptive_min_samples}, optimal_splits={optimal_splits}")
    
    # Strategy 1: Try StratifiedKFold with optimal splits
    if (n_samples >= CV_CONFIG["min_total_samples_for_stratified"] and 
        min_class_count >= adaptive_min_samples * optimal_splits):
        
        try:
            cv_splitter = StratifiedKFold(n_splits=optimal_splits, shuffle=True, random_state=0)
            # Test if it works and produces good folds
            if validate_cv_fold_quality(idx_temp, y_temp, cv_splitter, adaptive_min_samples):
                return cv_splitter, optimal_splits, "StratifiedKFold"
            else:
                logger.debug("StratifiedKFold validation failed, trying reduced splits")
        except Exception as e:
            logger.debug(f"StratifiedKFold failed: {str(e)}")
    
    # Strategy 2: Try StratifiedKFold with reduced splits
    for n_splits in range(optimal_splits - 1, CV_CONFIG["min_cv_splits"] - 1, -1):
        if min_class_count >= adaptive_min_samples * n_splits:
            try:
                cv_splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
                if validate_cv_fold_quality(idx_temp, y_temp, cv_splitter, adaptive_min_samples):
                    logger.info(f"Using StratifiedKFold with reduced splits: {n_splits}")
                    return cv_splitter, n_splits, f"StratifiedKFold (reduced to {n_splits})"
            except Exception as e:
                logger.debug(f"StratifiedKFold with {n_splits} splits failed: {str(e)}")
                continue
    
    # Strategy 3: Fall back to KFold with optimal splits
    try:
        cv_splitter = KFold(n_splits=optimal_splits, shuffle=True, random_state=0)
        if validate_cv_fold_quality(idx_temp, y_temp, cv_splitter, adaptive_min_samples):
            logger.warning(f"Falling back to KFold with {optimal_splits} splits")
            return cv_splitter, optimal_splits, f"KFold (fallback)"
    except Exception as e:
        logger.debug(f"KFold with {optimal_splits} splits failed: {str(e)}")
    
    # Strategy 4: KFold with minimum splits (last resort)
    min_splits = CV_CONFIG["min_cv_splits"]
    cv_splitter = KFold(n_splits=min_splits, shuffle=True, random_state=0)
    logger.warning(f"Using minimum KFold splits: {min_splits} (last resort)")
    return cv_splitter, min_splits, f"KFold (minimum {min_splits})"

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