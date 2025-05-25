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
    Memory efficient implementation that avoids unnecessary copies.
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
        # Calculate means along axis 0 (columnwise)
        # Use float32 to reduce memory usage
        self.means_ = np.nanmean(X, axis=0, dtype=np.float32)
        
        # Replace NaN means with 0 (in case entire column is NaN)
        # Use in-place operation to avoid creating new arrays
        np.nan_to_num(self.means_, copy=False, nan=0.0)
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
        
        # Check if there are any NaNs at all - avoid unnecessary operations
        if not np.isnan(X).any():
            return X
        
        # Create a copy of X with same dtype, preferably float32 to save memory
        X_imputed = X.copy().astype(np.float32, copy=False)
        
        # Find NaN positions
        nan_mask = np.isnan(X_imputed)
        
        # Only process columns that have NaNs
        nan_cols = np.where(nan_mask.any(axis=0))[0]
        
        # Process each column with NaNs individually to avoid creating large temporary arrays
        for col in nan_cols:
            # Get mask for this column
            col_mask = nan_mask[:, col]
            
            # Only replace values if there are NaNs
            if col_mask.any():
                X_imputed[col_mask, col] = self.means_[col]
        
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
        logger.warning("No valid arrays provided for merging")
        return np.zeros((0, 0), dtype=np.float32)
    
    # Convert all arrays to float32 numpy arrays and ensure they're 2D
    processed_arrays = []
    for i, arr in enumerate(filtered_arrays):
        try:
            # Convert to numpy array if not already - use float32 to reduce memory usage
            if not isinstance(arr, np.ndarray):
                arr_np = np.asarray(arr, dtype=np.float32)
            else:
                # If already numpy array, just ensure it's float32 without extra copy
                arr_np = arr if arr.dtype == np.float32 else arr.astype(np.float32)
                
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
    
    # If no arrays remain after processing, return empty array
    if not processed_arrays:
        logger.warning("No arrays to merge after processing")
        return np.zeros((0, 0), dtype=np.float32)
    
    # Find row counts and check for mismatches
    row_counts = [arr.shape[0] for arr in processed_arrays]
    
    # Check for mismatches in row counts - this is critical for proper alignment
    if len(set(row_counts)) > 1:
        min_rows = min(row_counts)
        max_rows = max(row_counts)
        logger.warning(f"Arrays have different row counts: min={min_rows}, max={max_rows}. Truncating to {min_rows} rows.")
        
        # Truncate all arrays to have the same number of rows - view when possible
        processed_arrays = [arr[:min_rows] for arr in processed_arrays]
    
    # Get final row count after truncation
    n_rows = processed_arrays[0].shape[0]
    
    # Merge based on strategy
    try:
        if strategy == "concat":
            # Concatenate along features dimension - most memory efficient
            merged = np.column_stack(processed_arrays)
        elif strategy in ["average", "sum", "max"]:
            # For these strategies, we need arrays with the same shape
            # Find the max number of columns across all arrays
            max_cols = max(arr.shape[1] for arr in processed_arrays)
            
            # Initialize result array - we'll fill it directly without intermediate arrays
            if strategy == "average" or strategy == "sum":
                # Use zeros for average/sum to accumulate values
                merged = np.zeros((n_rows, max_cols), dtype=np.float32)
                # Count number of non-NaN values per position for averaging
                counts = np.zeros((n_rows, max_cols), dtype=np.int32) if strategy == "average" else None
            else:  # strategy == "max"
                # Use negative infinity for max to find maximum values
                merged = np.full((n_rows, max_cols), -np.inf, dtype=np.float32)
            
            # Process each array individually to avoid stacking all arrays in memory
            for arr in processed_arrays:
                n_cols = arr.shape[1]
                
                if strategy == "average":
                    # Create mask for non-NaN values - avoid copying the array if possible
                    mask = ~np.isnan(arr)
                    # Only process columns with non-NaN values to avoid unnecessary operations
                    if np.any(mask):
                        # Add values to merged array where not NaN
                        np.add.at(merged[:, :n_cols], np.where(mask), arr[mask])
                        # Count non-NaN values for averaging
                        np.add.at(counts[:, :n_cols], np.where(mask), 1)
                elif strategy == "sum":
                    # Handle NaNs by replacing with 0 for summation - avoid copy if no NaNs
                    if np.isnan(arr).any():
                        arr_clean = np.nan_to_num(arr, nan=0.0, copy=True)
                    else:
                        arr_clean = arr
                    merged[:, :n_cols] += arr_clean
                elif strategy == "max":
                    # Handle NaNs by replacing with -inf for max - avoid copy if no NaNs
                    if np.isnan(arr).any():
                        arr_clean = np.nan_to_num(arr, nan=-np.inf, copy=True)
                    else:
                        arr_clean = arr
                    # Update merged with element-wise maximum
                    merged[:, :n_cols] = np.maximum(merged[:, :n_cols], arr_clean)
            
            # Finalize average calculation
            if strategy == "average":
                # Avoid division by zero by setting counts=1 where counts=0
                counts[counts == 0] = 1
                merged /= counts
                # Replace potential NaNs from division
                merged = np.nan_to_num(merged, nan=0.0, copy=False)
            elif strategy == "max":
                # Replace -inf values with 0 where no valid data existed
                merged[merged == -np.inf] = 0.0
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
                # Replace NaNs with 0 as a fallback - do in-place if possible
                np.nan_to_num(merged, nan=0.0, copy=False)
        else:
            # Always ensure there are no NaNs in the result - in-place operation
            np.nan_to_num(merged, nan=0.0, copy=False)
            
        # Last check for inf values - in-place operation
        if not np.isfinite(merged).all():
            np.nan_to_num(merged, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        
        # Verify the merged array's shape and ensure it's not empty
        if merged.size == 0 or merged.shape[0] == 0:
            logger.warning("Merged array has 0 rows")
            return np.zeros((1, 1), dtype=np.float32)
            
        logger.debug(f"Merged array shape: {merged.shape}")
        return merged
        
    except Exception as e:
        logger.error(f"Error in merge_modalities: {str(e)}")
        # Return a safe fallback array
        if processed_arrays:
            # Just return the first array if merging failed
            return processed_arrays[0]
        return np.zeros((1, 1), dtype=np.float32) 