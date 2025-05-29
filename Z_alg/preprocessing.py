#!/usr/bin/env python3
"""
Preprocessing module for data preparation and cleaning functions.
"""

import numpy as np
import pandas as pd
import random
from typing import Dict, List, Optional, Union, Any, Tuple
import logging

# Local imports
from config import MAX_VARIABLE_FEATURES

logger = logging.getLogger(__name__)

def _keep_top_variable_rows(df: pd.DataFrame,
                          k: int = MAX_VARIABLE_FEATURES) -> pd.DataFrame:
    """
    Keep at most *k* rows with the highest variance across samples.

    The omics matrices in this project are all shaped (features × samples),
    so we compute row-variance. Sparse frames are handled efficiently
    with toarray() fallback if needed.

    Parameters
    ----------
    df : pd.DataFrame (features × samples)
    k  : int – number of rows to keep (default = 5 000)

    Returns
    -------
    pd.DataFrame containing ≤ k rows with highest variance
    """
    # Skip if the data frame is already small enough
    if df.shape[0] <= k:
        return df
    
    # Compute row-wise variance
    try:
        if hasattr(df, 'sparse') and df.sparse.density < 0.3:
            # For sparse DataFrames, compute variance accordingly
            variances = df.sparse.to_dense().var(axis=1)
        else:
            variances = df.var(axis=1)
    except Exception as e:
        # Fallback to a simple implementation
        logger.warning(f"Warning: Using fallback variance computation due to: {str(e)}")
        variances = np.nanvar(df.values, axis=1)
    
    # Get indices of top-k variable rows
    if len(variances) <= k:
        # If we have fewer rows than k, keep them all
        return df
    else:
        # Get indices of top-k rows by variance
        top_indices = variances.nlargest(k).index
        return df.loc[top_indices]

def fix_tcga_id_slicing(id_list: List[str]) -> List[str]:
    """
    Standardize TCGA patient IDs by slicing to maintain only the core part.
    
    Parameters
    ----------
    id_list : List[str]
        List of patient IDs
        
    Returns
    -------
    List[str]
        List of standardized patient IDs
    """
    fixed_ids = []
    for id_str in id_list:
        # Check if it's a TCGA ID (typically starts with TCGA)
        if isinstance(id_str, str) and id_str.startswith("TCGA"):
            # Keep only the first 12 characters, which identify the patient uniquely
            # Format typically: TCGA-XX-XXXX
            parts = id_str.split("-")
            if len(parts) >= 3:
                # Ensure we get the core patient ID (first 3 parts)
                fixed_id = "-".join(parts[:3])
                fixed_ids.append(fixed_id)
            else:
                # If the ID doesn't have enough parts, keep it as is
                fixed_ids.append(id_str)
        else:
            # Non-TCGA ID, keep it as is
            fixed_ids.append(id_str)
            
    return fixed_ids

def custom_parse_outcome(series: pd.Series, outcome_type: str) -> pd.Series:
    """
    Parse outcome data based on the specified type.
    
    Parameters
    ----------
    series : pd.Series
        Series containing outcome data
    outcome_type : str
        Type of outcome data ('os', 'pfs', 'response', etc.)
        
    Returns
    -------
    pd.Series
        Parsed outcome data
    """
    # If series is a numpy.ndarray, convert to pandas Series first
    if isinstance(series, np.ndarray):
        series = pd.Series(series)
    
    # Handle the case when we cannot check for NaN values with isna()
    if not hasattr(series, 'isna'):
        # Convert to pandas Series to ensure we have the isna method
        series = pd.Series(series)
    
    if outcome_type in ['os', 'pfs', 'survival', 'continuous']:
        # For continuous outcomes like survival time, convert to float
        return pd.to_numeric(series, errors='coerce')
    elif outcome_type in ['response', 'class', 'status']:
        # For categorical outcomes, handle various formats
        if all(isinstance(x, (int, float, np.number)) or 
               (isinstance(x, str) and x.isdigit()) for x in series if pd.notna(x)):
            # If all values are numeric or numeric strings, convert to integers
            return pd.to_numeric(series, errors='coerce').astype('Int64')
        else:
            # For text categories like "Responder"/"Non-responder", encode as categorical
            try:
                return pd.Categorical(series).codes
            except:
                # If categorical encoding fails, try to convert to string first
                return pd.Categorical(series.astype(str)).codes
    else:
        # Default handling for unknown types - try numeric conversion with fallback
        try:
            return pd.to_numeric(series, errors='coerce')
        except Exception as e:
            logger.warning(f"Failed to convert outcome to numeric: {str(e)}")
            # Last resort - convert to string and then categorical
            try:
                return pd.Categorical(series.astype(str)).codes
            except Exception as e2:
                logger.warning(f"Failed to convert outcome to categorical: {str(e2)}")
                return series

def safe_convert_to_numeric(X: Any) -> np.ndarray:
    """
    Safely convert data to numeric numpy array, handling various input types.
    
    Parameters
    ----------
    X : Any
        Input data to convert
        
    Returns
    -------
    np.ndarray
        Numeric numpy array (float64 for sklearn compatibility)
    """
    try:
        if isinstance(X, pd.DataFrame):
            # Convert DataFrame to numpy array, filling NaNs with 0
            # Use float64 for sklearn compatibility
            return X.fillna(0).astype(np.float64).values
        elif isinstance(X, pd.Series):
            # Convert Series to numpy array, filling NaNs with 0
            return X.fillna(0).astype(np.float64).values
        elif isinstance(X, list):
            # Convert list to numpy array - use float64 for sklearn compatibility
            X_np = np.array(X, dtype=np.float64)
            # Replace any NaNs with 0
            X_np = np.nan_to_num(X_np, nan=0.0)
            return X_np
        elif isinstance(X, np.ndarray):
            # Already a numpy array, ensure float64 and handle NaNs
            X_float = X.astype(np.float64)
            # Replace any NaNs with 0
            X_float = np.nan_to_num(X_float, nan=0.0)
            return X_float
        else:
            # Unsupported type, try to convert to numpy array with float64
            X_np = np.array(X, dtype=np.float64)
            # Replace any NaNs with 0
            X_np = np.nan_to_num(X_np, nan=0.0)
            return X_np
    except Exception as e:
        logger.error(f"Error in safe_convert_to_numeric: {str(e)}")
        # Last resort: empty array with appropriate shape - use float64
        return np.zeros((1, 1), dtype=np.float64)

def process_with_missing_modalities(data_modalities: Dict[str, pd.DataFrame], 
                                   all_ids: List[str],
                                   missing_percentage: float,
                                   random_state: Optional[int] = None,
                                   min_overlap_ratio: float = 0.3) -> Dict[str, pd.DataFrame]:
    """
    Process modalities by randomly marking some samples as missing.
    This simulates real-world scenarios where some samples might not have data for all modalities.
    
    Parameters
    ----------
    data_modalities : Dict[str, pd.DataFrame]
        Dictionary of modality DataFrames
    all_ids : List[str]
        List of all sample IDs
    missing_percentage : float
        Percentage of data to mark as missing (0.0 to 1.0)
    random_state : Optional[int]
        Random seed for reproducibility
    min_overlap_ratio : float
        Minimum ratio of samples that must be present in all modalities
        
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary of processed modality DataFrames
    """
    # If missing_percentage is 0, return original data
    if missing_percentage == 0.0:
        return data_modalities
    
    # Set random seed for reproducibility if provided
    if random_state is not None:
        np.random.seed(random_state)
    
    # Initialize result dictionary - we'll modify DataFrames in-place when possible
    modified_modalities = {}
    
    # Keep track of sample availability for efficient overlap calculation
    sample_availability = {id_: set() for id_ in all_ids}
    
    # Ensure we have IDs as a set for O(1) lookup
    all_ids_set = set(all_ids)
    
    # Process each modality
    for mod_name, mod_df in data_modalities.items():
        if mod_df is None or mod_df.empty:
            modified_modalities[mod_name] = mod_df
            continue
        
        # Get available samples in this modality
        avail_samples = set(mod_df.columns).intersection(all_ids_set)
        
        # Skip modalities with very few samples
        if len(avail_samples) < 5:
            modified_modalities[mod_name] = mod_df
            for id_ in avail_samples:
                sample_availability[id_].add(mod_name)
            continue
        
        # Decide how many samples to keep (non-missing)
        samples_to_keep = max(
            int(len(avail_samples) * (1.0 - missing_percentage)),
            5  # Ensure at least 5 samples remain
        )
        
        # Randomly select samples to keep
        if samples_to_keep < len(avail_samples):
            samples_list = list(avail_samples)
            np.random.shuffle(samples_list)  # In-place shuffle
            keep_samples = set(samples_list[:samples_to_keep])
            
            # Filter the modality to keep only selected samples
            # Use view when possible to avoid copying
            keep_cols = [col for col in mod_df.columns if col in keep_samples]
            modified_modalities[mod_name] = mod_df[keep_cols]
            
            # Update sample availability
            for id_ in keep_samples:
                sample_availability[id_].add(mod_name)
        else:
            # Keep all samples if we would keep more than available
            modified_modalities[mod_name] = mod_df
            for id_ in avail_samples:
                sample_availability[id_].add(mod_name)
    
    # Calculate the number of modalities each sample appears in
    sample_mod_counts = {id_: len(mods) for id_, mods in sample_availability.items()}
    
    # Find samples present in all modalities
    all_mod_count = len(data_modalities)
    samples_in_all = [id_ for id_, count in sample_mod_counts.items() if count == all_mod_count]
    
    # Check if we have enough overlap
    if len(samples_in_all) < max(5, min_overlap_ratio * len(all_ids)):
        # We need to adjust to ensure sufficient overlap
        modality_names = list(data_modalities.keys())
        
        # Find samples with high presence but not in all modalities
        near_complete_samples = [
            id_ for id_, count in sample_mod_counts.items() 
            if count >= all_mod_count - 1 and count < all_mod_count
        ]
        
        # For some of these samples, add them to all modalities they're missing from
        np.random.shuffle(near_complete_samples)
        samples_to_add = near_complete_samples[:min(
            len(near_complete_samples),
            max(5, int(min_overlap_ratio * len(all_ids))) - len(samples_in_all)
        )]
        
        # Add these samples to modalities they're missing from
        for id_ in samples_to_add:
            missing_mods = [mod for mod in modality_names if mod not in sample_availability[id_]]
            for mod_name in missing_mods:
                if mod_name in modified_modalities and modified_modalities[mod_name] is not None:
                    # Check if sample is available in original data
                    if id_ in data_modalities[mod_name].columns:
                        # Get the column data
                        col_data = data_modalities[mod_name][id_]
                        
                        # Ensure we're working with a copy to avoid SettingWithCopyWarning
                        if isinstance(modified_modalities[mod_name], pd.DataFrame):
                            # Create a proper copy and add the column using pd.concat to avoid warnings
                            current_df = modified_modalities[mod_name].copy()
                            
                            # Add the column using pd.concat instead of direct assignment
                            new_col_df = pd.DataFrame({id_: col_data})
                            modified_modalities[mod_name] = pd.concat([current_df, new_col_df], axis=1)
    
    return modified_modalities

def align_modality_data(modalities: Dict[str, pd.DataFrame], common_ids: List[str], target_col: str) -> List[str]:
    """
    Align modality data with common IDs, prioritizing samples that appear across all modalities.
    
    Parameters
    ----------
    modalities : Dict[str, pd.DataFrame]
        Dictionary of modality DataFrames
    common_ids : List[str]
        List of common IDs across modalities
    target_col : str
        Target column name in labels
        
    Returns
    -------
    List[str]
        List of aligned sample IDs
    """
    # If no modalities, return empty list
    if not modalities:
        return []
    
    # Track which samples appear in how many modalities
    overlap_counts = {}
    
    # Count how many modalities each sample appears in
    for mod_name, mod_df in modalities.items():
        if mod_df.empty:
            continue
        
        for sample_id in mod_df.columns:
            if sample_id in common_ids:
                overlap_counts[sample_id] = overlap_counts.get(sample_id, 0) + 1
    
    # Prioritize samples that appear in all modalities
    all_modalities_count = len([m for m in modalities.values() if not m.empty])
    
    # Get samples present in all modalities first, then samples with partial presence
    complete_samples = [id for id in common_ids if overlap_counts.get(id, 0) == all_modalities_count]
    
    # If we have enough complete samples, use only those
    if len(complete_samples) >= 5:
        logger.info(f"Found {len(complete_samples)} samples present in all {all_modalities_count} modalities")
        return sorted(complete_samples)
    
    # Otherwise, prioritize samples by how many modalities they appear in
    partial_samples = sorted(
        [(id, count) for id, count in overlap_counts.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    # Take samples with highest overlap first
    prioritized_samples = [id for id, _ in partial_samples if id in common_ids]
    
    logger.info(f"Using {len(prioritized_samples)} prioritized samples - complete samples: {len(complete_samples)}")
    return prioritized_samples

def normalize_sample_ids(ids: List[str], target_separator: str = '-') -> Dict[str, str]:
    """
    Normalize sample IDs by replacing different separators with a target separator.
    Useful for standardizing IDs across different data sources.
    
    Parameters
    ----------
    ids : List[str]
        List of sample IDs to normalize
    target_separator : str, default='-'
        The separator to use in the normalized IDs
        
    Returns
    -------
    Dict[str, str]
        Dictionary mapping original IDs to normalized IDs
    """
    separators = ['-', '.', '_', ' ', '+']
    id_map = {}
    
    for id_str in ids:
        if not isinstance(id_str, str):
            continue
            
        normalized = id_str
        # Replace all separators (except target) with the target separator
        for sep in separators:
            if sep != target_separator and sep in normalized:
                normalized = normalized.replace(sep, target_separator)
        
        # Only add to map if it actually changed
        if normalized != id_str:
            id_map[id_str] = normalized
    
    return id_map

def filter_rare_categories(series: pd.Series, min_count: int = 3) -> pd.Series:
    """
    Filter out rare categories from a Series.
    
    Parameters
    ----------
    series : pd.Series
        Series containing categorical data
    min_count : int
        Minimum count for a category to be retained
        
    Returns
    -------
    pd.Series
        Series with rare categories filtered out
    """
    if pd.api.types.is_numeric_dtype(series) or len(series) < 5:
        # Not categorical or too few samples
        return series
        
    # Count values and identify rare categories
    value_counts = series.value_counts()
    rare_categories = value_counts[value_counts < min_count].index.tolist()
    
    # If most categories are rare, don't filter
    if len(rare_categories) > len(value_counts) / 2:
        return series
        
    # Replace rare categories with NaN
    filtered = series.copy()
    filtered[filtered.isin(rare_categories)] = np.nan
    
    return filtered 