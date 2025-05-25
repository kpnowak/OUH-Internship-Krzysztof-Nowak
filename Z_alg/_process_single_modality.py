"""
Helper functions for modality processing to ensure proper sample alignment
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Any, Union

logger = logging.getLogger(__name__)

def align_samples_to_modalities(
    id_train: Union[List[str], np.ndarray], 
    id_val: Union[List[str], np.ndarray],
    data_modalities: Dict[str, pd.DataFrame]
) -> Tuple[List[str], List[str]]:
    """
    Find IDs that are common across all modalities to ensure proper alignment
    
    Parameters
    ----------
    id_train : Union[List[str], np.ndarray]
        List of training sample IDs
    id_val : Union[List[str], np.ndarray]
        List of validation sample IDs
    data_modalities : Dict[str, pd.DataFrame]
        Dictionary of modality DataFrames
        
    Returns
    -------
    Tuple[List[str], List[str]]
        Lists of valid training and validation IDs present in all modalities
    """
    # Convert to lists if numpy arrays
    id_train_list = list(id_train)
    id_val_list = list(id_val)
    
    # Start with all IDs
    valid_train_ids = set(id_train_list) 
    valid_val_ids = set(id_val_list)
    
    # Log initial counts
    logger.debug(f"Initial ID count: {len(valid_train_ids)} train, {len(valid_val_ids)} validation")
    
    # Find IDs available in each modality
    for name, df in data_modalities.items():
        if df is None or df.empty:
            logger.warning(f"Empty modality: {name}")
            continue
            
        avail_ids = set(df.columns)
        train_before = len(valid_train_ids)
        val_before = len(valid_val_ids)
        
        valid_train_ids = valid_train_ids.intersection(avail_ids)
        valid_val_ids = valid_val_ids.intersection(avail_ids)
        
        train_diff = train_before - len(valid_train_ids)
        val_diff = val_before - len(valid_val_ids)
        
        if train_diff > 0 or val_diff > 0:
            logger.debug(f"Modality {name} removed {train_diff} train and {val_diff} validation IDs")
    
    # Convert sets back to ordered lists (needed for consistent ordering)
    valid_train_ids = sorted(list(valid_train_ids))
    valid_val_ids = sorted(list(valid_val_ids))
    
    logger.info(f"Found {len(valid_train_ids)} training and {len(valid_val_ids)} validation samples present in all modalities")
    
    return valid_train_ids, valid_val_ids

def verify_data_alignment(X: np.ndarray, y: np.ndarray, name: str = "unnamed", fold_idx: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Verify and fix data alignment between features (X) and labels (y)
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels/targets
    name : str
        Name of the dataset (for logging)
    fold_idx : Optional[int]
        Fold index (for logging)
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Aligned X and y arrays
    """
    fold_str = f" in fold {fold_idx}" if fold_idx is not None else ""
    
    if X is None or y is None:
        logger.warning(f"Null data for {name}{fold_str}")
        return None, None
    
    # Convert y to numpy array if it's not already
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    
    # Ensure y is 1D
    if y.ndim > 1 and y.shape[1] == 1:
        y = y.ravel()
    
    # Log input shapes for debugging    
    logger.debug(f"Data alignment check for {name}{fold_str}: X shape={X.shape}, y shape=({len(y)},)")
        
    if X.shape[0] != len(y):
        # Shape mismatch detected
        logger.warning(f"Shape mismatch for {name}{fold_str}: X={X.shape}, y={len(y)}")
        
        # Use the minimum length and truncate both
        min_samples = min(X.shape[0], len(y))
        
        # Check if substantial data loss would occur (>25% samples lost)
        percent_loss = 100 * (max(X.shape[0], len(y)) - min_samples) / max(X.shape[0], len(y))
        if percent_loss > 25:
            logger.warning(f"Severe alignment loss for {name}{fold_str}: {percent_loss:.1f}% of samples lost")
        
        X_aligned = X[:min_samples]
        y_aligned = y[:min_samples]
        
        # Double check aligned shapes
        if X_aligned.shape[0] != len(y_aligned):
            logger.error(f"ERROR: Alignment failed for {name}{fold_str}: X={X_aligned.shape}, y={len(y_aligned)}")
            # Last resort - return None if we can't align
            if min_samples < 2:
                logger.error(f"Too few samples for {name}{fold_str}: {min_samples}")
                return None, None
        
        # Log details about the alignment operation
        logger.info(f"Aligned shapes for {name}{fold_str}: X={X_aligned.shape}, y={len(y_aligned)}, truncated {abs(X.shape[0] - len(y))} samples")
        return X_aligned, y_aligned
    
    # No alignment needed
    logger.debug(f"No alignment needed for {name}{fold_str}: X and y shapes match ({X.shape[0]} samples)")
    return X, y 