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


class EarlyFusionPCA:
    """
    Early Fusion with PCA for multimodal data integration.
    Concatenates modalities and applies PCA for dimensionality reduction.
    """
    def __init__(self, n_components: int = None, random_state: int = 42):
        """
        Initialize EarlyFusionPCA.
        
        Parameters
        ----------
        n_components : int, optional
            Number of components to keep. If None, keeps all components.
        random_state : int
            Random state for reproducibility.
        """
        self.n_components = n_components
        self.random_state = random_state
        self.pca_ = None
        self.fitted_ = False
        
    def fit(self, *arrays: np.ndarray) -> 'EarlyFusionPCA':
        """
        Fit the EarlyFusionPCA on the concatenated modalities.
        
        Parameters
        ----------
        *arrays : np.ndarray
            Variable-length list of 2-D arrays (modalities)
            
        Returns
        -------
        self : object
            Returns self.
        """
        # First concatenate the arrays
        concatenated = self._concatenate_arrays(*arrays)
        
        if concatenated.size == 0:
            logger.warning("No valid data for EarlyFusionPCA fitting")
            return self
        
        # Determine optimal number of components
        max_components = min(concatenated.shape[0], concatenated.shape[1])
        if self.n_components is None:
            effective_components = max_components
        else:
            effective_components = min(self.n_components, max_components)
        
        # Ensure we have at least 1 component
        effective_components = max(1, effective_components)
        
        # Initialize and fit PCA
        from sklearn.decomposition import PCA
        self.pca_ = PCA(n_components=effective_components, random_state=self.random_state)
        
        try:
            self.pca_.fit(concatenated)
            self.fitted_ = True
            logger.debug(f"EarlyFusionPCA fitted with {effective_components} components on data shape {concatenated.shape}")
        except Exception as e:
            logger.error(f"Error fitting EarlyFusionPCA: {str(e)}")
            # Fallback: just store the concatenated data without PCA
            self.pca_ = None
            self.fitted_ = True
        
        return self
    
    def transform(self, *arrays: np.ndarray) -> np.ndarray:
        """
        Transform the concatenated modalities using fitted PCA.
        
        Parameters
        ----------
        *arrays : np.ndarray
            Variable-length list of 2-D arrays (modalities)
            
        Returns
        -------
        np.ndarray
            Transformed data.
        """
        if not self.fitted_:
            raise ValueError("EarlyFusionPCA has not been fitted yet. Call 'fit' first.")
        
        # Concatenate the arrays
        concatenated = self._concatenate_arrays(*arrays)
        
        if concatenated.size == 0:
            logger.warning("No valid data for EarlyFusionPCA transformation")
            return np.zeros((0, 1), dtype=np.float32)
        
        # Apply PCA transformation if available
        if self.pca_ is not None:
            try:
                transformed = self.pca_.transform(concatenated)
                return transformed.astype(np.float32)
            except Exception as e:
                logger.warning(f"Error in PCA transformation: {str(e)}, returning concatenated data")
                return concatenated
        else:
            # Return concatenated data if PCA failed during fitting
            return concatenated
    
    def fit_transform(self, *arrays: np.ndarray) -> np.ndarray:
        """
        Fit and transform the data.
        
        Parameters
        ----------
        *arrays : np.ndarray
            Variable-length list of 2-D arrays (modalities)
            
        Returns
        -------
        np.ndarray
            Transformed data.
        """
        return self.fit(*arrays).transform(*arrays)
    
    def _concatenate_arrays(self, *arrays: np.ndarray) -> np.ndarray:
        """
        Helper method to concatenate arrays safely.
        
        Parameters
        ----------
        *arrays : np.ndarray
            Variable-length list of 2-D arrays
            
        Returns
        -------
        np.ndarray
            Concatenated array.
        """
        # Skip None or empty arrays
        filtered_arrays = [arr for arr in arrays if arr is not None and arr.size > 0]
        
        if not filtered_arrays:
            return np.zeros((0, 0), dtype=np.float32)
        
        # Process arrays to ensure they're 2D and float32
        processed_arrays = []
        for i, arr in enumerate(filtered_arrays):
            try:
                # Convert to numpy array if not already
                if not isinstance(arr, np.ndarray):
                    arr_np = np.asarray(arr, dtype=np.float32)
                else:
                    arr_np = arr if arr.dtype == np.float32 else arr.astype(np.float32)
                    
                # Ensure 2D
                if arr_np.ndim == 1:
                    arr_np = arr_np.reshape(-1, 1)
                elif arr_np.ndim > 2:
                    original_shape = arr_np.shape
                    arr_np = arr_np.reshape(original_shape[0], -1)
                    
                processed_arrays.append(arr_np)
            except Exception as e:
                logger.error(f"Error processing array {i} in EarlyFusionPCA: {str(e)}")
                continue
        
        if not processed_arrays:
            return np.zeros((0, 0), dtype=np.float32)
        
        # Check for row count mismatches
        row_counts = [arr.shape[0] for arr in processed_arrays]
        if len(set(row_counts)) > 1:
            min_rows = min(row_counts)
            logger.warning(f"EarlyFusionPCA: Arrays have different row counts, truncating to {min_rows} rows")
            processed_arrays = [arr[:min_rows] for arr in processed_arrays]
        
        # Concatenate along features dimension
        try:
            concatenated = np.column_stack(processed_arrays)
            # Handle any remaining NaN or inf values
            concatenated = np.nan_to_num(concatenated, nan=0.0, posinf=0.0, neginf=0.0)
            return concatenated
        except Exception as e:
            logger.error(f"Error concatenating arrays in EarlyFusionPCA: {str(e)}")
            # Return the first array as fallback
            return processed_arrays[0] if processed_arrays else np.zeros((0, 0), dtype=np.float32)


def merge_modalities(*arrays: np.ndarray, 
                    strategy: str = "weighted_concat", 
                    imputer: Optional[ModalityImputer] = None, 
                    is_train: bool = True,
                    n_components: int = None) -> np.ndarray:
    """
    Merge an arbitrary number of numpy arrays (same number of rows).

    Parameters
    ----------
    *arrays : np.ndarray
        Variable-length list of 2-D arrays (or None/empty)
    strategy : str, default="weighted_concat"
        Merge strategy: 'weighted_concat' | 'average' | 'sum' | 'early_fusion_pca'
    imputer : Optional[ModalityImputer]
        Optional ModalityImputer instance for handling missing values
    is_train : bool, default=True
        Whether this is training data (True) or validation/test data (False)
    n_components : int, optional
        Number of components for EarlyFusionPCA (only used with early_fusion_pca strategy)

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
        if strategy == "weighted_concat":
            # Weighted concatenation - weight by inverse of feature count to balance modalities
            if len(processed_arrays) == 1:
                # If only one modality, just return it
                merged = processed_arrays[0]
            else:
                # Calculate weights based on inverse of feature counts
                feature_counts = [arr.shape[1] for arr in processed_arrays]
                total_features = sum(feature_counts)
                
                # Weight by inverse of feature count, normalized
                weights = [total_features / (len(processed_arrays) * count) for count in feature_counts]
                
                # Apply weights and concatenate
                weighted_arrays = []
                for arr, weight in zip(processed_arrays, weights):
                    weighted_arr = arr * weight
                    weighted_arrays.append(weighted_arr)
                
                merged = np.column_stack(weighted_arrays)
                logger.debug(f"Weighted concatenation with weights: {weights}")
                
        elif strategy == "early_fusion_pca":
            # Early Fusion with PCA
            early_fusion = EarlyFusionPCA(n_components=n_components, random_state=42)
            merged = early_fusion.fit_transform(*processed_arrays)
            logger.debug(f"EarlyFusionPCA applied with n_components={n_components}")
            
        elif strategy in ["average", "sum"]:
            # For these strategies, we need arrays with the same shape
            # Find the max number of columns across all arrays
            max_cols = max(arr.shape[1] for arr in processed_arrays)
            
            # Initialize result array - we'll fill it directly without intermediate arrays
            if strategy == "average" or strategy == "sum":
                # Use zeros for average/sum to accumulate values
                merged = np.zeros((n_rows, max_cols), dtype=np.float32)
                # Count number of non-NaN values per position for averaging
                counts = np.zeros((n_rows, max_cols), dtype=np.int32) if strategy == "average" else None
            
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
            
            # Finalize average calculation
            if strategy == "average":
                # Avoid division by zero by setting counts=1 where counts=0
                counts[counts == 0] = 1
                merged /= counts
                # Replace potential NaNs from division
                merged = np.nan_to_num(merged, nan=0.0, copy=False)
                
        else:
            # Default to weighted concatenation for unknown strategy
            logger.warning(f"Unknown merge strategy: {strategy}, using weighted_concat instead")
            # Fallback to simple concatenation if weighting fails
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
            
        logger.debug(f"Merged array shape: {merged.shape} using strategy: {strategy}")
        return merged
        
    except Exception as e:
        logger.error(f"Error in merge_modalities with strategy {strategy}: {str(e)}")
        # Return a safe fallback array
        if processed_arrays:
            # Just return the first array if merging failed
            return processed_arrays[0]
        return np.zeros((1, 1), dtype=np.float32) 