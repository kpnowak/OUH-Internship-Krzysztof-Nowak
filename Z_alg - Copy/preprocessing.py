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
from Z_alg.config import MAX_VARIABLE_FEATURES

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
    if outcome_type in ['os', 'pfs', 'survival']:
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
            return pd.Categorical(series).codes
    else:
        # Default handling for unknown types - try numeric conversion with fallback
        try:
            return pd.to_numeric(series, errors='coerce')
        except Exception:
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
        Numeric numpy array
    """
    try:
        if isinstance(X, pd.DataFrame):
            # Convert DataFrame to numpy array, filling NaNs with 0
            return X.fillna(0).values
        elif isinstance(X, pd.Series):
            # Convert Series to numpy array, filling NaNs with 0
            return X.fillna(0).values
        elif isinstance(X, list):
            # Convert list to numpy array
            X_np = np.array(X, dtype=np.float32)
            # Replace any NaNs with 0
            X_np = np.nan_to_num(X_np, nan=0.0)
            return X_np
        elif isinstance(X, np.ndarray):
            # Already a numpy array, just handle NaNs
            X_float = X.astype(np.float32)
            # Replace any NaNs with 0
            X_float = np.nan_to_num(X_float, nan=0.0)
            return X_float
        else:
            # Unsupported type, try to convert to numpy array
            X_np = np.array(X, dtype=np.float32)
            # Replace any NaNs with 0
            X_np = np.nan_to_num(X_np, nan=0.0)
            return X_np
    except Exception as e:
        logger.error(f"Error in safe_convert_to_numeric: {str(e)}")
        # Last resort: empty array with appropriate shape
        return np.zeros((1, 1), dtype=np.float32)

def process_with_missing_modalities(data_modalities: Dict[str, pd.DataFrame], 
                                   all_ids: List[str],
                                   missing_percentage: float,
                                   random_state: Optional[int] = None,
                                   min_overlap_ratio: float = 0.3) -> Dict[str, pd.DataFrame]:
    """
    Process modalities by randomly marking a percentage of sample data as missing,
    while ensuring a minimum overlap ratio across modalities.
    
    Parameters
    ----------
    data_modalities : Dict[str, pd.DataFrame]
        Dictionary of modality DataFrames
    all_ids : List[str]
        List of all sample IDs
    missing_percentage : float
        Percentage of data to mark as missing (0.0 - 1.0)
    random_state : Optional[int]
        Random seed for reproducibility
    min_overlap_ratio : float
        Minimum ratio of samples that should overlap across all modalities (default=0.3)
        
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary of modality DataFrames with missing data
    """
    # Skip if missing percentage is 0
    if missing_percentage <= 0:
        return data_modalities
    
    # Set random seed for reproducibility
    random.seed(random_state)
    
    # Check current overlap before introducing missing data
    common_samples = set()
    first = True
    
    for modality_df in data_modalities.values():
        if modality_df.empty or modality_df.shape[1] == 0:
            continue
            
        if first:
            common_samples = set(modality_df.columns)
            first = False
        else:
            common_samples = common_samples.intersection(set(modality_df.columns))
    
    current_overlap_ratio = len(common_samples) / len(all_ids) if all_ids else 0
    
    # Calculate the maximum missing percentage that maintains the minimum overlap
    adjusted_missing_percentage = missing_percentage
    
    # If current overlap is already close to minimum, reduce missing percentage
    if current_overlap_ratio <= min_overlap_ratio * 1.5:
        adjusted_missing_percentage = min(missing_percentage, (current_overlap_ratio - min_overlap_ratio) / 2)
        adjusted_missing_percentage = max(0, adjusted_missing_percentage)  # Ensure it's not negative
        
        if adjusted_missing_percentage < missing_percentage:
            logger.info(f"Overlap protection: Reducing missing percentage from {missing_percentage:.2f} to {adjusted_missing_percentage:.2f} to maintain sample overlap ratio")
    
    # Create a copy of the modalities to avoid modifying the originals
    modified_modalities = {}
    
    # Process each modality
    for modality_name, modality_df in data_modalities.items():
        # If the DataFrame is empty or has no columns, skip it
        if modality_df.empty or modality_df.shape[1] == 0:
            modified_modalities[modality_name] = modality_df
            continue
            
        # Get the columns (sample IDs) in this modality
        modality_cols = list(modality_df.columns)
        
        # First, identify columns that must be preserved (present in all modalities)
        core_columns = list(common_samples.intersection(modality_cols))
        other_columns = [col for col in modality_cols if col not in core_columns]
        
        # Calculate number of non-core samples to mark as missing
        n_missing = int(len(other_columns) * adjusted_missing_percentage)
        
        # Randomly select non-core columns to remove
        if n_missing > 0 and other_columns:
            cols_to_remove = random.sample(other_columns, min(n_missing, len(other_columns)))
            # Create a new DataFrame without the removed columns
            modified_df = modality_df.drop(columns=cols_to_remove)
            modified_modalities[modality_name] = modified_df
        else:
            # If no samples to remove, keep the original
            modified_modalities[modality_name] = modality_df
    
    # Double check the final overlap
    final_common_samples = set()
    first = True
    
    for modality_df in modified_modalities.values():
        if modality_df.empty or modality_df.shape[1] == 0:
            continue
            
        if first:
            final_common_samples = set(modality_df.columns)
            first = False
        else:
            final_common_samples = final_common_samples.intersection(set(modality_df.columns))
    
    final_overlap_ratio = len(final_common_samples) / len(all_ids) if all_ids else 0
    
    logger.info(f"Missing data applied: initial overlap={current_overlap_ratio:.2f}, final overlap={final_overlap_ratio:.2f}")
    
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