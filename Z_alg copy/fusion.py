#!/usr/bin/env python3
"""
Fusion module for multimodal data integration.
"""

import numpy as np
from typing import List, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)

class ModalityImputer:
    """
    Imputes missing values in a multimodal dataset.
    Simple implementation that replaces missing values with column means.
    """
    def __init__(self):
        """Initialize the imputer."""
        self.means_ = None
        
    def fit(self, X: np.ndarray) -> 'ModalityImputer':
        """
        Compute the column means of X for later use in transform.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data matrix.
            
        Returns
        -------
        self : object
            Returns self.
        """
        self.means_ = np.nanmean(X, axis=0)
        # Replace NaN means with 0 (in case entire column is NaN)
        self.means_ = np.nan_to_num(self.means_, nan=0)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Impute missing values in X using the column means from fit.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data matrix.
            
        Returns
        -------
        X_imputed : ndarray, shape (n_samples, n_features)
            Data with imputed values.
        """
        if self.means_ is None:
            raise ValueError("ModalityImputer has not been fitted yet. Call 'fit' first.")
        
        # Create output array
        X_imputed = X.copy()
        
        # Create mask of NaN values
        mask = np.isnan(X_imputed)
        
        # If there are any NaNs, replace them with column means
        if np.any(mask):
            # For each column with NaNs, replace with its mean
            for j in range(X_imputed.shape[1]):
                col_mask = mask[:, j]
                if np.any(col_mask):
                    # Use j-th mean to replace NaNs in j-th column
                    X_imputed[col_mask, j] = self.means_[j]
        
        return X_imputed
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit to X and transform X.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data matrix.
            
        Returns
        -------
        X_imputed : ndarray, shape (n_samples, n_features)
            Data with imputed values.
        """
        return self.fit(X).transform(X)


def _pad(arr: np.ndarray, target_cols: int) -> np.ndarray:
    """
    Pad array with zeros to reach target_cols columns.
    
    Parameters
    ----------
    arr : np.ndarray
        Input array
    target_cols : int
        Target number of columns
        
    Returns
    -------
    np.ndarray
        Padded array with target_cols columns
    """
    diff = target_cols - arr.shape[1]
    if diff <= 0:
        return arr
    return np.pad(arr, ((0, 0), (0, diff)), mode="constant", constant_values=0)


def merge_modalities(*arrays: np.ndarray, 
                    strategy: str = "concat", 
                    imputer: Optional[ModalityImputer] = None, 
                    is_train: bool = True) -> np.ndarray:
    """
    Merge an arbitrary number of numpy arrays (same number of rows).

    Parameters
    ----------
    *arrays : np.ndarray
        Variable-length list of 2-D arrays (or None/empty)
    strategy : str, default="concat"
        Merge strategy: 'concat' | 'average' | 'sum' | 'max'
    imputer : Optional[ModalityImputer]
        Optional ModalityImputer instance for handling missing values
    is_train : bool, default=True
        Whether this is training data (True) or validation/test data (False)

    Returns
    -------
    np.ndarray
        The merged matrix (float32).
    """
    # Skip None or empty arrays
    filtered_arrays = [arr for arr in arrays if arr is not None and arr.size > 0]
    
    # Check if we have any arrays to merge
    if not filtered_arrays:
        logger.warning("Warning: No valid arrays provided for merging")
        return np.zeros((0, 0), dtype=np.float32)
    
    # Convert all arrays to float32 numpy arrays and ensure they're 2D
    processed_arrays = []
    for i, arr in enumerate(filtered_arrays):
        try:
            # Convert to numpy array if not already
            arr_np = np.asarray(arr, dtype=np.float32)
            # Ensure 2D - if 1D, reshape to column vector
            if arr_np.ndim == 1:
                arr_np = arr_np.reshape(-1, 1)
            # For higher dimensions, flatten all but the first dimension
            elif arr_np.ndim > 2:
                original_shape = arr_np.shape
                arr_np = arr_np.reshape(original_shape[0], -1)
            processed_arrays.append(arr_np)
        except Exception as e:
            logger.error(f"Error processing array {i}: {str(e)}")
            # Skip problematic arrays
            continue
    
    # Find row counts and check for mismatches
    row_counts = [arr.shape[0] for arr in processed_arrays]
    
    # Check for mismatches in row counts - this is critical for proper alignment
    if len(set(row_counts)) > 1:
        min_rows = min(row_counts)
        max_rows = max(row_counts)
        logger.warning(f"Arrays have different row counts: min={min_rows}, max={max_rows}. Truncating to {min_rows} rows.")
        
        # Truncate all arrays to have the same number of rows
        processed_arrays = [arr[:min_rows] for arr in processed_arrays]
        
        # Double-check row counts after truncation
        new_row_counts = [arr.shape[0] for arr in processed_arrays]
        if len(set(new_row_counts)) > 1:
            logger.error(f"Failed to align arrays after truncation: {new_row_counts}. Using first array only.")
            # Last resort - use only the first array
            processed_arrays = [processed_arrays[0]]
    
    # Log array shapes for debugging
    shapes = [arr.shape for arr in processed_arrays]
    logger.info(f"Merging {len(processed_arrays)} arrays with shapes: {shapes}, strategy: {strategy}")
    
    if len(processed_arrays) == 0:
        logger.warning("No arrays to merge after processing")
        return np.zeros((0, 0), dtype=np.float32)
    
    # Merge based on strategy
    try:
        if strategy == "concat":
            # Concatenate along features dimension
            merged = np.column_stack(processed_arrays)
        elif strategy in ["average", "sum", "max"]:
            # For these strategies, we need arrays with the same shape
            # Find the max number of columns across all arrays
            max_cols = max(arr.shape[1] for arr in processed_arrays)
            
            # Pad each array to have the same number of columns
            padded_arrays = []
            for arr in processed_arrays:
                if arr.shape[1] < max_cols:
                    pad_width = max_cols - arr.shape[1]
                    padded = np.pad(arr, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
                    padded_arrays.append(padded)
                else:
                    padded_arrays.append(arr)

            # Stack arrays along a new axis (making a 3D array)
            stacked = np.stack(padded_arrays, axis=0)
            
            # Apply the appropriate operation along the first axis (modalities)
            if strategy == "average":
                merged = np.nanmean(stacked, axis=0)
            elif strategy == "sum":
                merged = np.nansum(stacked, axis=0)
            elif strategy == "max":
                merged = np.nanmax(stacked, axis=0)
        else:
            # Default to concatenation for unknown strategy
            logger.warning(f"Unknown merge strategy: {strategy}, using concat instead")
            merged = np.column_stack(processed_arrays)

        # Apply imputation if an imputer is provided
        if imputer is not None:
            try:
                if is_train:
                    merged = imputer.fit_transform(merged)
                else:
                    merged = imputer.transform(merged)
            except Exception as e:
                logger.warning(f"Imputation failed: {str(e)}, using original data")
                # Replace NaNs with 0 as a fallback
                merged = np.nan_to_num(merged, nan=0.0)
        else:
            # Always ensure there are no NaNs in the result
            merged = np.nan_to_num(merged, nan=0.0)
            
        # Last check for inf values
        if not np.isfinite(merged).all():
            logger.warning("Merged array contains inf values, replacing with 0.")
            merged = np.nan_to_num(merged, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Verify the merged array's shape and ensure it's not empty
        if merged.size == 0 or merged.shape[0] == 0:
            logger.warning("Merged array has 0 rows")
            return np.zeros((1, 1), dtype=np.float32)
            
        logger.info(f"Merged array shape: {merged.shape}")
        return merged
        
    except Exception as e:
        logger.error(f"Error in merge_modalities: {str(e)}")
        # Return a safe fallback array
        if processed_arrays:
            # Just return the first array if merging failed
            return processed_arrays[0]
        return np.zeros((1, 1), dtype=np.float32) 