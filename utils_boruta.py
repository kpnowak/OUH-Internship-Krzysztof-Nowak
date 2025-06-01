#!/usr/bin/env python3
"""
Utilities for Boruta feature selection with improved stability.
"""

import numpy as np
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from typing import List, Literal, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)


def boruta_selector(X: np.ndarray, y: np.ndarray, n_feats: int, task: str = 'reg', random_state: int = 42) -> List[int]:
    """
    Apply Boruta feature selection with better robustness and error handling.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    n_feats : int
        Maximum number of features to select
    task : str
        Task type ('reg' for regression, 'clf' for classification)
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    List[int]
        List of selected feature indices
    """
    # Boruta is very sensitive to data issues - ensure clean data
    X_clean = np.nan_to_num(X, nan=0.0)
    
    # Ensure X and y have the same number of samples
    if X_clean.shape[0] != len(y):
        logger.warning(f"Shape mismatch in boruta_selector: X={X_clean.shape}, y={len(y)}")
        min_samples = min(X_clean.shape[0], len(y))
        X_clean = X_clean[:min_samples]
        y_clean = y[:min_samples]
    else:
        y_clean = y.copy()
    
    # Handle the case where we have too few samples
    min_samples_required = 10  # Boruta often fails with fewer samples
    if X_clean.shape[0] < min_samples_required:
        logger.warning(f"Too few samples for Boruta ({X_clean.shape[0]} < {min_samples_required}), falling back to feature importance")
        return _get_feature_importances(X_clean, y_clean, n_feats, task, random_state)
    
    # Handle the case where number of features is too low
    if X_clean.shape[1] <= n_feats:
        logger.info(f"Number of features ({X_clean.shape[1]}) is <= requested features ({n_feats}), selecting all")
        return list(range(X_clean.shape[1]))
        
    try:
        # Choose model based on task
        if task == 'reg':
            model = RandomForestRegressor(
                n_estimators=100, 
                max_depth=10,
                random_state=random_state, 
                n_jobs=1
            )
            percentile = 90  # Slightly more permissive
        else:  # Classification
            model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=10,
                class_weight='balanced',
                random_state=random_state, 
                n_jobs=1
            )
            percentile = 90  # Slightly more permissive
            
        # Make Boruta more robust by adjusting parameters
        boruta = BorutaPy(
            model,
            n_estimators='auto',
            verbose=0,
            random_state=random_state,
            max_iter=30,  # Reduced from 50 for faster execution
            perc=percentile,
            two_step=False  # More permissive selection
        )
        
        # Fit Boruta with error handling
        try:
            boruta.fit(X_clean, y_clean)
            
            # Get ranking and selected features
            ranks = boruta.ranking_
            support = boruta.support_
            
            # Get indices of selected features
            selected_indices = np.where(support)[0]
            
            # If Boruta selected some features but fewer than requested
            if 0 < len(selected_indices) < n_feats:
                # Use tentative features too
                tentative = boruta.support_weak_
                tentative_indices = np.where(tentative)[0]
                
                # Combine confirmed and tentative features
                combined_indices = np.union1d(selected_indices, tentative_indices)
                
                # If still not enough, add by importance
                if len(combined_indices) < n_feats:
                    remaining = n_feats - len(combined_indices)
                    ranks_arr = np.argsort(ranks)
                    unselected = np.setdiff1d(ranks_arr, combined_indices)
                    additional = unselected[:remaining]
                    selected_indices = np.union1d(combined_indices, additional)
                else:
                    selected_indices = combined_indices[:n_feats]
            
            # Limit the number of features if Boruta selected too many
            if len(selected_indices) > n_feats:
                # Use ranking for best features
                rank_indices = np.argsort(ranks)
                selected_indices = rank_indices[:n_feats]
            
            # If no features were selected by Boruta, take top N based on feature importance
            if len(selected_indices) == 0:
                logger.debug("Boruta didn't select any features, falling back to feature importance")
                selected_indices = _get_feature_importances(X_clean, y_clean, n_feats, task, random_state)
                
            # Convert numpy array to Python list before returning
            if isinstance(selected_indices, np.ndarray):
                return [int(i) for i in selected_indices]
            return selected_indices
            
        except ValueError as ve:
            logger.debug(f"Boruta value error: {str(ve)}, falling back to feature importance")
            return _get_feature_importances(X_clean, y_clean, n_feats, task, random_state)
    
    except Exception as e:
        logger.error(f"Error in Boruta feature selection: {str(e)}")
        # Return top features based on importance as fallback
        return _get_feature_importances(X_clean, y_clean, n_feats, task, random_state)


def _get_feature_importances(X: np.ndarray, y: np.ndarray, n_feats: int, task: str, random_state: int) -> List[int]:
    """
    Helper function to select features based on feature importance.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    n_feats : int
        Number of features to select
    task : str
        Task type ('reg' for regression, 'clf' for classification)
    random_state : int
        Random seed
        
    Returns
    -------
    List[int]
        List of selected feature indices
    """
    try:
        # Use appropriate model based on task
        if task == 'reg':
            model = RandomForestRegressor(
                n_estimators=100, 
                max_depth=10,
                random_state=random_state, 
                n_jobs=1
            )
        else:
            model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=10,
                class_weight='balanced',
                random_state=random_state, 
                n_jobs=1
            )
        
        # Limit feature count to available features
        n_feats = min(n_feats, X.shape[1])
        
        # Fit model
        model.fit(X, y)
        
        # Get feature importances and select top features
        importances = model.feature_importances_
        top_indices = np.argsort(-importances)[:n_feats]  # Descending order
        
        # Convert to Python list before returning (don't use tolist() which is only for numpy arrays)
        return [int(i) for i in top_indices]
    except Exception as e:
        logger.error(f"Error in feature importance fallback: {str(e)}")
        # Return first n_feats as last resort
        return list(range(min(n_feats, X.shape[1]))) 