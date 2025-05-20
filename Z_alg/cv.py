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

# Initialize logger
logger = logging.getLogger(__name__)

# Import our alignment helpers
try:
    from Z_alg._process_single_modality import align_samples_to_modalities, verify_data_alignment
except ImportError:
    # Create inline versions if import fails
    logger.warning("Warning: Could not import helper modules, using inline functions")
    
    def align_samples_to_modalities(id_train, id_val, data_modalities):
        """Fallback implementation for alignment function"""
        valid_train_ids = set(id_train) 
        valid_val_ids = set(id_val)
        
        for name, df in data_modalities.items():
            if df is None or df.empty:
                continue
            avail_ids = set(df.columns)
            valid_train_ids = valid_train_ids.intersection(avail_ids)
            valid_val_ids = valid_val_ids.intersection(avail_ids)
        
        return sorted(list(valid_train_ids)), sorted(list(valid_val_ids))
    
    def verify_data_alignment(X, y, name="unnamed", fold_idx=None):
        """
        Verify that X and y have matching first dimensions and fix if needed.
        
        Parameters
        ----------
        X : array-like
            Feature matrix
        y : array-like
            Target vector
        name : str, default="unnamed"
            Name of the dataset (for logging)
        fold_idx : Optional[int]
            Fold index (for logging)
            
        Returns
        -------
        Tuple[array-like, array-like]
            Aligned X and y arrays
        """
        if X is None or y is None:
            return None, None
            
        if X.shape[0] != len(y):
            logger.warning(f"Warning: Shape mismatch for {name} (fold {fold_idx}): X={X.shape}, y={len(y)}")
            
            # Find the minimum length to truncate to
            min_samples = min(X.shape[0], len(y))
            
            # Truncate both arrays
            X_aligned = X[:min_samples]
            y_aligned = y[:min_samples]
            
            logger.info(f"Aligned shapes: X={X_aligned.shape}, y={len(y_aligned)}")
            return X_aligned, y_aligned
        
        # No adjustment needed
        return X, y

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
    fold_idx: Optional[int] = None
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
        Training sample IDs
    id_val : List[str]
        Validation sample IDs
    idx_test : np.ndarray
        Test indices
    y_train : np.ndarray
        Training labels/target values
    extr_obj : Any
        Extractor object, or name of extractor
    ncomps : int
        Number of components to extract
    idx_to_id : Dict[int, str]
        Mapping from index to sample ID
    fold_idx : Optional[int]
        CV fold index
        
    Returns
    -------
    Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]
        Tuple of (train, validation, test) arrays
    """
    try:
        # Create Series for training y with proper indices
        y_train_series = pd.Series(y_train, index=id_train)
        
        # Check for available samples in this modality
        avail_train_ids = [id_ for id_ in id_train if id_ in modality_df.columns]
        avail_val_ids = [id_ for id_ in id_val if id_ in modality_df.columns]
        avail_test_ids = [idx_to_id[idx] for idx in idx_test if idx in idx_to_id and idx_to_id[idx] in modality_df.columns]
        
        # Early return if not enough samples
        if len(avail_train_ids) < 5 or len(avail_val_ids) < 2:
            logger.info(f"Not enough valid samples for {modality_name} in fold {fold_idx} - Train: {len(avail_train_ids)}, Val: {len(avail_val_ids)}")
            return None, None, None
            
        # Get data for each split
        df_train = modality_df.loc[:, avail_train_ids].transpose()
        df_val = modality_df.loc[:, avail_val_ids].transpose()
        df_test = modality_df.loc[:, avail_test_ids].transpose() if avail_test_ids else pd.DataFrame()
        
        # Now get aligned y values AFTER sampling data - critical for alignment
        aligned_y_train = y_train_series.reindex(df_train.index).values
        
        # Confirm we have valid data after all processing
        if df_train.shape[0] != len(aligned_y_train) or np.isnan(df_train.values).any():
            logger.warning(f"Data alignment issue in {modality_name}: {df_train.shape[0]} rows vs {len(aligned_y_train)} labels in fold {fold_idx}")
            return None, None, None
        
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

        # If extr_obj is a selector or selector code string, use selector pipeline
        if isinstance(extr_obj, (SelectorMixin, BorutaPy, str)) or (isinstance(extr_obj, dict) and 'type' in extr_obj):
            # Regression or classification selector?
            # We'll use regression if y_train is float, classification if int or categorical
            if np.issubdtype(aligned_y_train.dtype, np.floating):
                # Regression selector
                selected_features, X_tr = cached_fit_transform_selector_regression(
                    extr_obj, df_train, aligned_y_train, req, fold_idx=fold_idx, ds_name=modality_name
                )
                X_va = transform_selector_regression(df_val, selected_features)
                X_te = transform_selector_regression(df_test, selected_features) if not df_test.empty else np.array([])
            else:
                # Classification selector
                selected_features, X_tr = cached_fit_transform_selector_classification(
                    df_train, aligned_y_train, extr_obj, req, ds_name=None, modality_name=modality_name, fold_idx=fold_idx
                )
                X_va = transform_selector_classification(df_val, selected_features)
                X_te = transform_selector_classification(df_test, selected_features) if not df_test.empty else np.array([])
            # Replace any remaining NaNs with zeros
            X_tr = np.nan_to_num(X_tr, nan=0.0)
            X_va = np.nan_to_num(X_va, nan=0.0)
            if X_te is not None and X_te.size > 0:
                X_te = np.nan_to_num(X_te, nan=0.0)
            logger.info(f"Successfully processed {modality_name} (selector) in fold {fold_idx} - Train: {X_tr.shape}, Val: {X_va.shape}, Test: {X_te.shape if X_te is not None and X_te.size > 0 else '(0,0)'}")
            return X_tr, X_va, X_te

        # For LDA, components are limited by the number of classes
        if isinstance(extr_obj, LDA):
            max_allowed = min(max_allowed, len(np.unique(aligned_y_train)) - 1)
            
        # Check if requested components exceed maximum allowed
        if req > max_allowed:
            logger.info(f"{modality_name}: clipping n_components {req}->{max_allowed}")
            req = max_allowed
        
        # Extract features - different process for classification vs regression
        if np.issubdtype(aligned_y_train.dtype, np.floating):
            # Regression
            try:
                from Z_alg.models import cached_fit_transform_extractor_regression, transform_extractor_regression
                extractor, X_tr = cached_fit_transform_extractor_regression(
                    df_train.values, aligned_y_train, extr_obj, req, 
                    ds_name=modality_name, fold_idx=fold_idx
                )
                # Only transform validation and test if we have data
                X_va = transform_extractor_regression(df_val.values, extractor) if not df_val.empty else np.array([])
                X_te = transform_extractor_regression(df_test.values, extractor) if not df_test.empty else np.array([])
                
                logger.info(f"Extraction successful for {modality_name} in fold {fold_idx}: {X_tr.shape} from {df_train.shape}")
            except Exception as e:
                logger.error(f"Error extracting features for {modality_name} in fold {fold_idx}: {str(e)}")
                return None, None, None
        else:
            # Classification
            try:
                extractor, X_tr = cached_fit_transform_extractor_classification(
                    df_train.values, aligned_y_train, extr_obj, req, 
                    modality_name=modality_name, fold_idx=fold_idx
                )
                # Only transform validation and test if we have data
                X_va = transform_extractor_classification(df_val.values, extractor) if not df_val.empty else np.array([])
                X_te = transform_extractor_classification(df_test.values, extractor) if not df_test.empty else np.array([])
                
                logger.info(f"Extraction successful for {modality_name} in fold {fold_idx}: {X_tr.shape} from {df_train.shape}")
            except Exception as e:
                logger.error(f"Error extracting features for {modality_name} in fold {fold_idx}: {str(e)}")
                return None, None, None
        
        # Replace any NaN values with zeros
        X_tr = np.nan_to_num(X_tr, nan=0.0)
        X_va = np.nan_to_num(X_va, nan=0.0)
        if X_te is not None and X_te.size > 0:
            X_te = np.nan_to_num(X_te, nan=0.0)
        
        # Log shape information
        logger.info(f"Successfully processed {modality_name} in fold {fold_idx} - Train: {X_tr.shape}, Val: {X_va.shape}, Test: {X_te.shape if X_te is not None and X_te.size > 0 else '(0,0)'}")
        
        # Return the processed data
        return X_tr, X_va, X_te
    except Exception as e:
        logger.error(f"Error processing {modality_name} in fold {fold_idx}: {str(e)}")
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
    is_regression=True
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
    is_regression : bool
        Whether this is a regression task
        
    Returns
    -------
    Dict[str, Dict[str, float]]
        Dictionary of model results
    """
    try:
        # CRITICAL FIX: Add detailed logging to track the pipeline
        logger.info(f"Starting fold {fold_idx} for {ds_name} with {extr_name} and n_comps={ncomps}")
        logger.info(f"Initial indices: train={len(train_idx)}, val={len(val_idx)}, test={len(idx_test)}")
        
        # Convert indices to IDs
        id_train = np.array([idx_to_id[i] for i in train_idx if i in idx_to_id])
        id_val = np.array([idx_to_id[i] for i in val_idx if i in idx_to_id])
        
        # Add detailed logging for initial conversion
        logger.info(f"After index->ID conversion in fold {fold_idx}: train_ids={len(id_train)}, val_ids={len(id_val)}")
        
        # Get target values
        y_train = y_temp[train_idx]
        y_val = y_temp[val_idx]
        
        # CRITICAL FIX: Confirm y values are correct length
        logger.info(f"Initial y vectors in fold {fold_idx}: y_train={len(y_train)}, y_val={len(y_val)}")
        
        # Create Series for easier access by ID
        y_train_series = pd.Series(y_train, index=id_train)
        y_val_series = pd.Series(y_val, index=id_val)
        
        # Process modalities with missing data to simulate real-world scenarios
        from Z_alg.preprocessing import process_with_missing_modalities
        modified_modalities = process_with_missing_modalities(
            data_modalities, all_ids, missing_percentage, 
            random_state=fold_idx, min_overlap_ratio=0.3
        )
        
        # CRITICAL FIX: Check sample counts after missing data simulation
        logger.info(f"After missing data simulation in fold {fold_idx}: {len(modified_modalities)} modalities")
        for name, df in modified_modalities.items():
            logger.info(f"  Modality {name} in fold {fold_idx}: {df.shape} with {len(df.columns)} samples")
        
        # Use our new alignment function to find common IDs
        valid_train_ids, valid_val_ids = align_samples_to_modalities(
            id_train, id_val, modified_modalities
        )
        
        # Return empty if insufficient common samples
        if len(valid_train_ids) < 5 or len(valid_val_ids) < 2:
            logger.warning(f"Warning: Insufficient common samples across modalities in fold {fold_idx}")
            return {}
        
        # CRITICAL FIX: Ensure strict alignment between IDs and y values
        # First, convert lists to ensure we're working with the same data type
        valid_train_ids = list(valid_train_ids)
        valid_val_ids = list(valid_val_ids)
        
        # Add detailed logging for aligned IDs
        logger.info(f"After ID alignment in fold {fold_idx}: train_ids={len(valid_train_ids)}, val_ids={len(valid_val_ids)}")
        
        # Ensure valid_train_ids only contains IDs that are in y_train_series
        valid_train_ids = [id_ for id_ in valid_train_ids if id_ in y_train_series.index]
        valid_val_ids = [id_ for id_ in valid_val_ids if id_ in y_val_series.index]
        
        # Log after filtering by y_train_series index
        logger.info(f"After y-series filtering in fold {fold_idx}: train_ids={len(valid_train_ids)}, val_ids={len(valid_val_ids)}")
        
        # Get aligned labels for the common samples
        aligned_y_train = y_train_series.loc[valid_train_ids].values
        aligned_y_val = y_val_series.loc[valid_val_ids].values
        
        # Log y vector sizes after alignment
        logger.info(f"After alignment to y_series in fold {fold_idx}: y_train={len(aligned_y_train)}, y_val={len(aligned_y_val)}")
        
        # CRITICAL FIX: Ensure valid_train_ids and aligned_y_train have exactly the same number of samples
        if len(valid_train_ids) != len(aligned_y_train):
            logger.error(f"Critical alignment error in fold {fold_idx}: train_ids={len(valid_train_ids)}, y_train={len(aligned_y_train)}")
            # Fix the mismatch by truncating to the smaller size
            min_len = min(len(valid_train_ids), len(aligned_y_train))
            valid_train_ids = valid_train_ids[:min_len]
            aligned_y_train = aligned_y_train[:min_len]
            
        # Same for validation
        if len(valid_val_ids) != len(aligned_y_val):
            logger.error(f"Critical alignment error in fold {fold_idx}: val_ids={len(valid_val_ids)}, y_val={len(aligned_y_val)}")
            # Fix the mismatch by truncating to the smaller size
            min_len = min(len(valid_val_ids), len(aligned_y_val))
            valid_val_ids = valid_val_ids[:min_len]
            aligned_y_val = aligned_y_val[:min_len]
        
        # Double check that we have the correct number of samples
        logger.info(f"Pre-processing alignment in fold {fold_idx}: train_ids={len(valid_train_ids)}, val_ids={len(valid_val_ids)}")
        logger.info(f"Pre-processing aligned y in fold {fold_idx}: y_train={len(aligned_y_train)}, y_val={len(aligned_y_val)}")
        
        # Final check - ensure we have enough samples to proceed
        if len(valid_train_ids) < 5 or len(valid_val_ids) < 2:
            logger.warning(f"Warning: Insufficient aligned samples in fold {fold_idx} after strict alignment")
            return {}
        
        # Special handling for selectors - they need more careful alignment
        # Selectors are identified by being a string or a dict with 'type' key
        selector_mode = isinstance(extr_obj, str) or (isinstance(extr_obj, dict) and 'type' in extr_obj)
        if selector_mode:
            logger.info(f"Selector mode detected in fold {fold_idx}: {extr_obj}")
            
            # For selectors, we need to ensure special alignment for multi-modality selectors
            from sklearn.feature_selection import SelectorMixin
            from boruta import BorutaPy
            selector_types = (SelectorMixin, BorutaPy, str)
            if isinstance(extr_obj, selector_types) or (isinstance(extr_obj, dict) and 'type' in extr_obj):
                logger.info(f"Before final selector alignment in fold {fold_idx}: train_ids={len(valid_train_ids)}, y_train={len(aligned_y_train)}")
    
        # Process modalities in parallel
        from joblib import Parallel, delayed
        from Z_alg.config import JOBLIB_PARALLEL_CONFIG
        
        # Limit number of jobs to avoid excessive resource usage
        n_jobs = min(3, os.cpu_count() or 1)
        logger.info(f"Processing {len(modified_modalities)} modalities in parallel with {n_jobs} jobs")
        
        modality_results = Parallel(n_jobs=n_jobs, **JOBLIB_PARALLEL_CONFIG)(
            delayed(_process_single_modality)(
                name, 
                df, 
                valid_train_ids,  # Use aligned IDs
                valid_val_ids,    # Use aligned IDs
                idx_test, 
                aligned_y_train,  # Use aligned labels
                extr_obj, 
                ncomps, 
                idx_to_id,
                fold_idx
            )
            for name, df in modified_modalities.items()
        )
    
        # Filter out None results and ensure all arrays have data
        valid_results = []
        for r in modality_results:
            if r is not None and all(x is not None for x in r[:2]):  # Only check train and val
                if all(x.size > 0 for x in r[:2]):  # Only check train and val
                    valid_results.append(r)
    
        if not valid_results:
            logger.warning(f"Warning: No valid data found for any modality in fold {fold_idx}")
            return {}
    
        # Create a new imputer instance for this fold
        from Z_alg.fusion import ModalityImputer
        fold_imputer = ModalityImputer()
    
        # Merge modalities with the fold-specific imputer - fit only once on training data
        try:
            from Z_alg.fusion import merge_modalities
            
            # Log shapes of individual modality results before merging
            logger.info(f"Before merging in fold {fold_idx}: {len(valid_results)} valid modalities")
            for i, r in enumerate(valid_results):
                logger.info(f"  Modality {i+1} in fold {fold_idx}: train={r[0].shape}, val={r[1].shape}")
            
            # Check for row count consistency before merging
            train_row_counts = [r[0].shape[0] for r in valid_results]
            val_row_counts = [r[1].shape[0] for r in valid_results]
            
            if len(set(train_row_counts)) > 1:
                logger.warning(f"Inconsistent train row counts before merging in fold {fold_idx}: {train_row_counts}")
                # Find the minimum number of rows and truncate all arrays
                min_train_rows = min(train_row_counts)
                valid_results = [(r[0][:min_train_rows], r[1], r[2]) for r in valid_results]
                
                # Update aligned_y_train to match
                if len(aligned_y_train) != min_train_rows:
                    logger.warning(f"Truncating aligned_y_train from {len(aligned_y_train)} to {min_train_rows} in fold {fold_idx}")
                    aligned_y_train = aligned_y_train[:min_train_rows]
            
            if len(set(val_row_counts)) > 1:
                logger.warning(f"Inconsistent validation row counts before merging in fold {fold_idx}: {val_row_counts}")
                # Find the minimum number of rows and truncate all arrays
                min_val_rows = min(val_row_counts)
                valid_results = [(r[0], r[1][:min_val_rows], r[2]) for r in valid_results]
                
                # Update aligned_y_val to match
                if len(aligned_y_val) != min_val_rows:
                    logger.warning(f"Truncating aligned_y_val from {len(aligned_y_val)} to {min_val_rows} in fold {fold_idx}")
                    aligned_y_val = aligned_y_val[:min_val_rows]
            
            # Critical fix: Also check if all arrays have the same number of rows as y
            min_train_samples = min([r[0].shape[0] for r in valid_results] + [len(aligned_y_train)])
            if min_train_samples < len(aligned_y_train):
                logger.info(f"Aligning dimensions in fold {fold_idx}: aligned_y_train from {len(aligned_y_train)} to {min_train_samples}")
                aligned_y_train = aligned_y_train[:min_train_samples]
                # Also update the arrays
                valid_results = [(r[0][:min_train_samples], r[1], r[2]) for r in valid_results]
            
            min_val_samples = min([r[1].shape[0] for r in valid_results] + [len(aligned_y_val)])
            if min_val_samples < len(aligned_y_val):
                logger.info(f"Aligning dimensions in fold {fold_idx}: aligned_y_val from {len(aligned_y_val)} to {min_val_samples}")
                aligned_y_val = aligned_y_val[:min_val_samples]
                # Also update the arrays
                valid_results = [(r[0], r[1][:min_val_samples], r[2]) for r in valid_results]
            
            X_train_merged = merge_modalities(*[r[0] for r in valid_results], imputer=fold_imputer, is_train=True)
            
            # Use the same fitted imputer for validation
            X_val_merged = merge_modalities(*[r[1] for r in valid_results], imputer=fold_imputer, is_train=False)
            
            # Final verification - these should be rare now
            if X_train_merged.shape[0] != len(aligned_y_train):
                logger.warning(f"Shape mismatch after merging in fold {fold_idx}: X_train={X_train_merged.shape}, y_train={len(aligned_y_train)}")
                # CRITICAL FIX: This is where the warnings are coming from
                # We must ensure the arrays have the same dimensions
                min_samples = min(X_train_merged.shape[0], len(aligned_y_train))
                X_train_merged = X_train_merged[:min_samples]
                aligned_y_train = aligned_y_train[:min_samples]
                # Verify that the shapes now match
                if X_train_merged.shape[0] != len(aligned_y_train):
                    logger.error(f"Failed to align arrays after truncation in fold {fold_idx}. This should never happen.")
                    return {}
                logger.info(f"Successfully fixed shape mismatch in fold {fold_idx}: X_train={X_train_merged.shape}, y_train={len(aligned_y_train)}")
            
            # Ensure X_val_merged and aligned_y_val have the same number of samples
            if X_val_merged.shape[0] != len(aligned_y_val):
                logger.warning(f"Shape mismatch after merging in fold {fold_idx}: X_val={X_val_merged.shape}, y_val={len(aligned_y_val)}")
                # CRITICAL FIX: Same for validation data
                min_samples = min(X_val_merged.shape[0], len(aligned_y_val))
                X_val_merged = X_val_merged[:min_samples]
                aligned_y_val = aligned_y_val[:min_samples]
                # Verify that the shapes now match
                if X_val_merged.shape[0] != len(aligned_y_val):
                    logger.error(f"Failed to align validation arrays after truncation in fold {fold_idx}. This should never happen.")
                    return {}
                logger.info(f"Successfully fixed shape mismatch in fold {fold_idx}: X_val={X_val_merged.shape}, y_val={len(aligned_y_val)}")
            
            # Log shapes after merging, before alignment
            logger.info(f"After merging in fold {fold_idx}: X_train={X_train_merged.shape}, X_val={X_val_merged.shape}")
            logger.info(f"After merging in fold {fold_idx}: y_train={len(aligned_y_train)}, y_val={len(aligned_y_val)}")
            
            # Skip if no valid data after merging
            if X_train_merged.size == 0 or X_val_merged.size == 0:
                logger.warning(f"Warning: No valid data after merging in fold {fold_idx}")
                return {}
        except Exception as e:
            logger.error(f"Error merging modalities in fold {fold_idx}: {str(e)}")
            return {}
        
        # Verify shapes before proceeding - using our helper function
        X_train_merged, aligned_y_train = verify_data_alignment(
            X_train_merged, aligned_y_train, 
            name=f"training data (fold {fold_idx})", 
            fold_idx=fold_idx
        )
        
        X_val_merged, aligned_y_val = verify_data_alignment(
            X_val_merged, aligned_y_val, 
            name=f"validation data (fold {fold_idx})", 
            fold_idx=fold_idx
        )
        
        # Log shapes after alignment
        logger.info(f"After first alignment in fold {fold_idx}: X_train={X_train_merged.shape if X_train_merged is not None else 'None'}, y_train={len(aligned_y_train) if aligned_y_train is not None else 'None'}")
        logger.info(f"After first alignment in fold {fold_idx}: X_val={X_val_merged.shape if X_val_merged is not None else 'None'}, y_val={len(aligned_y_val) if aligned_y_val is not None else 'None'}")
        
        # Skip fold if we don't have enough samples after alignment
        if (X_train_merged is None or aligned_y_train is None or 
            X_val_merged is None or aligned_y_val is None):
            logger.warning(f"Warning: Invalid data after alignment in fold {fold_idx}")
            return {}
            
        # Final validation - skip fold if we don't have enough samples after alignment
        if len(aligned_y_train) < 4 or len(aligned_y_val) < 2:
            logger.warning(f"Warning: Too few samples remaining after alignment in fold {fold_idx}")
            return {}
    
        # Train and evaluate models
        model_results = {}
        
        # Import the correct function based on the task type
        if is_regression:
            train_model = train_regression_model
        else:
            train_model = train_classification_model
            
        for model_name in models:
            try:
                # Perform one final verification before training the model
                # This is critical to ensure X and y have the same number of samples
                final_X_train, final_y_train = verify_data_alignment(
                    X_train_merged, aligned_y_train, 
                    name=f"training data for {model_name} (fold {fold_idx})", 
                    fold_idx=fold_idx
                )
                
                final_X_val, final_y_val = verify_data_alignment(
                    X_val_merged, aligned_y_val, 
                    name=f"validation data for {model_name} (fold {fold_idx})", 
                    fold_idx=fold_idx
                )
                
                # Log shapes before final model training
                logger.info(f"Final model data for {model_name} (fold {fold_idx}): X_train={final_X_train.shape if final_X_train is not None else 'None'}, y_train={len(final_y_train) if final_y_train is not None else 'None'}")
                logger.info(f"Final model data for {model_name} (fold {fold_idx}): X_val={final_X_val.shape if final_X_val is not None else 'None'}, y_val={len(final_y_val) if final_y_val is not None else 'None'}")
                
                # Only proceed if we have valid data
                if (final_X_train is None or final_y_train is None or 
                    final_X_val is None or final_y_val is None):
                    logger.warning(f"Warning: Invalid data for {model_name} in fold {fold_idx}")
                    continue
                    
                # Train and evaluate
                model, metrics = train_model(
                    final_X_train, final_y_train, 
                    final_X_val, final_y_val,
                    model_name, 
                    out_dir=os.path.join(base_out, "plots"),
                    plot_prefix=f"{ds_name}_fold_{fold_idx}_{extr_name}_{ncomps}_{model_name}",
                    fold_idx=fold_idx
                )
                
                # Store results if valid
                if model is not None and metrics:
                    model_results[model_name] = metrics
            except Exception as e:
                logger.warning(f"Warning: Failed to train {model_name} in fold {fold_idx}: {str(e)}")
                continue
                
        return model_results
    except Exception as e:
        logger.warning(f"Warning: Error processing fold {fold_idx}: {str(e)}")
        return {}

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
            idx_temp, idx_test, y_temp, y_test = train_test_split(
                indices, y_arr, test_size=test_size, random_state=0, stratify=y_arr
            )
    except ValueError as e:
        logger.warning(f"Stratification failed for {ds_name}: {str(e)}. Falling back to regular split. Unique values: {len(np.unique(y_arr))}, sample size: {len(indices)}")
        idx_temp, idx_test, y_temp, y_test = train_test_split(
            indices, y_arr, test_size=0.2, random_state=0
        )
    
    # Process each transformer and parameter combination
    from Z_alg.config import MISSING_MODALITIES_CONFIG
    
    for trans_name, trans_obj in transformers.items():
        for n_val in n_trans_list:
            # Update and report progress
            progress_count[0] += 1
            run_idx = progress_count[0]
            trans_type = "EXTRACT" if pipeline_type == "extraction" else "SELECT"
            task_type = "REG" if is_regression else "CLF"
            progress_msg = f"[{trans_type}-{task_type} CV] {run_idx}/{total_runs} => {ds_name} | {trans_name}-{n_val}"
            # Always show progress in terminal
            print(progress_msg)
            logger.info(progress_msg)

            # Choose appropriate CV splitter based on dataset size and task
            n_splits = 2 if len(idx_temp) < 15 else 3  # Use fewer splits for small datasets
            cv_cls = KFold if is_regression else StratifiedKFold
            cv = cv_cls(n_splits=n_splits, shuffle=True, random_state=0)
            
            logger.info(f"Dataset: {ds_name}, using {n_splits}-fold CV with {len(idx_temp)} training samples")
            
            # Results storage
            all_results = []
            
            # Process each missing percentage
            for missing_percentage in MISSING_MODALITIES_CONFIG["missing_percentages"]:
                cv_results = []
                
                # Process each CV fold
                for fold_idx, (train_idx, val_idx) in enumerate(cv.split(idx_temp, y_temp)):
                    # Suppress warnings during fold processing
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        
                        if pipeline_type == "extraction":
                            # For extraction pipeline
                            result = process_cv_fold(
                                train_idx, val_idx, idx_temp, idx_test, y_temp, y_test,
                                data_modalities, models, trans_obj, n_val, 
                                id_to_idx, idx_to_id, common_ids, missing_percentage,
                                fold_idx, base_out, ds_name, trans_name, is_regression
                            )
                        else:
                            # For selection pipeline, call process_cv_fold as well
                            result = process_cv_fold(
                                train_idx, val_idx, idx_temp, idx_test, y_temp, y_test,
                                data_modalities, models, trans_obj, n_val, 
                                id_to_idx, idx_to_id, common_ids, missing_percentage,
                                fold_idx, base_out, ds_name, trans_name, is_regression
                            )
                        
                    # Only add valid results
                    if result:
                        cv_results.append(result)
                
                # Skip if no valid results for this missing percentage
                if not cv_results:
                    continue
                
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
                        avg_metrics = {
                            k: np.mean([m[k] for m in valid_results if k in m and not np.isnan(m[k])]) 
                            for k in metric_keys
                        }
                        cv_metrics[model_name] = avg_metrics
                
                # Add combined results
                for model_name, metrics in cv_metrics.items():
                    result_entry = {
                        "Dataset": ds_name, 
                        "Workflow": f"{pipeline_type.title()}-CV",
                        f"{pipeline_type.title()[:-3]}tor": trans_name, 
                        f"n_{pipeline_type == 'extraction' and 'components' or 'features'}": n_val,
                        "Model": model_name,
                        "Missing_Percentage": missing_percentage,
                        **metrics
                    }
                    all_results.append(result_entry)
            
            # Save all results
            if all_results:
                metrics_file = os.path.join(
                    base_out, "metrics", 
                    f"{ds_name}_{pipeline_type}_cv_metrics.csv"
                )
                
                # Check if file exists
                file_exists = os.path.exists(metrics_file)
                
                # Append results to CSV
                pd.DataFrame(all_results).to_csv(
                    metrics_file,
                    mode='a',
                    header=not file_exists,
                    index=False
                )

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

def train_regression_model(X_train, y_train, X_val, y_val, model_name, out_dir, plot_prefix, fold_idx=None):
    """Train regression model and evaluate it."""
    from Z_alg.models import get_model_object
    from Z_alg.plots import plot_regression_scatter, plot_regression_residuals
    import os
    import numpy as np
    import time
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Final data alignment check before training
    from Z_alg._process_single_modality import verify_data_alignment
    X_train, y_train = verify_data_alignment(
        X_train, y_train, name=f"final training data for {model_name}", fold_idx=fold_idx
    )
    X_val, y_val = verify_data_alignment(
        X_val, y_val, name=f"final validation data for {model_name}", fold_idx=fold_idx
    )
    
    # Return if any data is invalid
    if X_train is None or y_train is None or X_val is None or y_val is None:
        logger.warning(f"Final data alignment failed for {model_name} in fold {fold_idx}")
        return None, {}
    
    # Create the model
    model = get_model_object(model_name)
    
    # Train the model with timing
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    
    # Make predictions
    y_pred = model.predict(X_val)
    
    # Calculate metrics
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    # Create plots
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        plot_regression_scatter(y_val, y_pred, f"{model_name} Scatter", os.path.join(out_dir, f"{plot_prefix}_scatter.png"))
        plot_regression_residuals(y_val, y_pred, f"{model_name} Residuals", os.path.join(out_dir, f"{plot_prefix}_residuals.png"))
    
    # Return model and metrics
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'train_time': train_time
    }
    
    return model, metrics


def train_classification_model(X_train, y_train, X_val, y_val, model_name, out_dir, plot_prefix, fold_idx=None):
    """Train classification model and evaluate it."""
    from Z_alg.models import get_model_object
    from Z_alg.plots import plot_confusion_matrix, plot_roc_curve_binary
    import os
    import numpy as np
    import time
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, matthews_corrcoef
    
    # Final data alignment check before training
    from Z_alg._process_single_modality import verify_data_alignment
    X_train, y_train = verify_data_alignment(
        X_train, y_train, name=f"final training data for {model_name}", fold_idx=fold_idx
    )
    X_val, y_val = verify_data_alignment(
        X_val, y_val, name=f"final validation data for {model_name}", fold_idx=fold_idx
    )
    
    # Return if any data is invalid
    if X_train is None or y_train is None or X_val is None or y_val is None:
        logger.warning(f"Final data alignment failed for {model_name} in fold {fold_idx}")
        return None, {}
    
    # Create the model
    model = get_model_object(model_name)
    
    # Train the model with timing
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    
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
    
    # Try to calculate ROC AUC if applicable (binary classification with proba method)
    auc = 0.5  # Default value
    try:
        if hasattr(model, 'predict_proba') and len(np.unique(y_val)) == 2:
            y_score = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_score)
    except Exception as e:
        logger.warning(f"Could not calculate AUC: {str(e)}")
    
    # Create plots if directory is provided
    if out_dir:
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
    
    # Return model and metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'mcc': mcc,
        'train_time': train_time
    }
    
    return model, metrics 