#!/usr/bin/env python3
"""
Models module for extractors, selectors, and model creation functions.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Set, Union, Literal
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import (
    LinearRegression, Lasso, ElasticNet, LogisticRegression, Ridge
)
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
)
from sklearn.svm import SVR, SVC
from sklearn.decomposition import (
    PCA, NMF, FastICA, FactorAnalysis, KernelPCA
)
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import (
    mutual_info_regression, mutual_info_classif,
    f_regression, f_classif, chi2, SelectKBest,
    SelectFromModel
)
from sklearn.preprocessing import MinMaxScaler
from boruta import BorutaPy

# Local imports
from Z_alg.config import MODEL_OPTIMIZATIONS
from Z_alg.preprocessing import safe_convert_to_numeric
from Z_alg.utils_boruta import boruta_selector

# Caching for models and feature transformations
_selector_cache = {
    'sel_reg': {},
    'sel_clf': {},
}

_extractor_cache = {
    'ext_reg': {},
    'ext_clf': {},
}

# Flag for LDA warning
_SHOWN_LDA_MSG = False

logger = logging.getLogger(__name__)

# For compatibility with older sklearn versions
try:
    sklearn_version = "1.3.0"  # Default fallback
except:
    pass

def get_regression_extractors() -> Dict[str, Any]:
    """
    Get dictionary of regression feature extractors.
    
    Returns
    -------
    dict
        Dictionary mapping extractor names to initialized extractor objects
    """
    return {
        "PCA": PCA(random_state=42),
        "NMF": NMF(
            init='nndsvdar',
            random_state=42,
            max_iter=5000,  # Increased max iterations
            tol=1e-3,      # Relaxed tolerance
            beta_loss='frobenius',
            solver='mu'
        ),
        "ICA": FastICA(
            random_state=42,
            max_iter=5000,  # Increased max iterations
            tol=1e-3,      # Relaxed tolerance
            algorithm='parallel',
            whiten='unit-variance'
        ),
        "FA": FactorAnalysis(
            random_state=42,
            max_iter=5000,  # Increased max iterations
            tol=1e-3       # Relaxed tolerance
        ),
        "PLS": PLSRegression(
            n_components=8,
            max_iter=5000,  # Increased max iterations
            tol=1e-3       # Relaxed tolerance
        )
    }

def get_regression_selectors() -> Dict[str, str]:
    """
    Get dictionary of regression feature selectors.
    
    Returns
    -------
    dict
        Dictionary mapping selector names to selector codes
    """
    return {
        "MRMR": "mrmr_reg",
        "LASSO": "lasso",
        "ElasticNetFS": "enet",
        "f_regressionFS": "freg",
        "Boruta": "boruta_reg"
    }

def get_classification_extractors() -> Dict[str, Any]:
    """
    Get dictionary of classification feature extractors.
    
    Returns
    -------
    dict
        Dictionary mapping extractor names to initialized extractor objects
    """
    ica_params = {
        "max_iter": 1000,  # Reduced for faster execution
        "whiten": "unit-variance",
        "whiten_solver": "svd",
        "tol": 1e-2,       # Relaxed tolerance
        "algorithm": "parallel",
        "fun": "logcosh"   # Better convergence
    } if sklearn_version >= "1.3.0" else {
        "max_iter": 1000,  
        "whiten": "unit-variance",
        "tol": 1e-2,       
        "algorithm": "parallel",
        "fun": "logcosh"
    }
    
    return {
        "PCA": PCA(random_state=42),
        "ICA": FastICA(**ica_params, random_state=42),
        "LDA": LDA(),
        "FA": FactorAnalysis(
            n_components=10,
            max_iter=1000, 
            tol=1e-3,
            random_state=42
        ),
        "KPCA": KernelPCA(kernel='rbf', random_state=42)
    }

def get_classification_selectors() -> Dict[str, str]:
    """
    Get dictionary of classification feature selectors.
    
    Returns
    -------
    dict
        Dictionary mapping selector names to selector codes
    """
    return {
        "MRMR": "mrmr_clf",
        "fclassifFS": "fclassif",
        "LogisticL1": "logistic_l1",
        "Boruta": "boruta_clf",
        "Chi2FS": "chi2_selection"
    }

def get_selector_object(selector_code: str, n_feats: int):
    """Create appropriate selector object based on code."""
    if selector_code == "mrmr_reg":
        return SelectKBest(mutual_info_regression, k=n_feats)
    elif selector_code == "lasso":
        # Create a standalone Lasso model that we will fit manually
        # instead of letting SelectFromModel try to fit it
        return {"type": "lasso", "n_feats": n_feats}
    elif selector_code == "enet":
        # Create a standalone ElasticNet model that we will fit manually
        # instead of letting SelectFromModel try to fit it
        return {"type": "enet", "n_feats": n_feats}
    elif selector_code == "freg":
        return SelectKBest(f_regression, k=n_feats)
    elif selector_code == "boruta_reg":
        # Instead of using BorutaPy directly, use the dictionary approach
        # with a simpler implementation that uses boruta_selector
        return {"type": "boruta_reg", "n_feats": n_feats}
    elif selector_code == "mrmr_clf" or selector_code.upper() == "MRMR":
        return SelectKBest(mutual_info_classif, k=n_feats)
    elif selector_code == "fclassifFS" or selector_code == "fclassif":
        return SelectKBest(f_classif, k=n_feats)
    elif selector_code == "logistic_l1" or selector_code == "LogisticL1":
        # Similarly for logistic regression
        return {"type": "logistic_l1", "n_feats": n_feats}
    elif selector_code == "chi2_selection" or selector_code == "Chi2FS":
        return SelectKBest(chi2, k=n_feats)
    elif selector_code == "boruta_clf" or selector_code == "Boruta":
        # Similarly for Boruta classification
        return {"type": "boruta_clf", "n_feats": n_feats}
    else:
        # Fallback to appropriate default selector based on prefix
        if selector_code.startswith("f_") or "classif" in selector_code.lower():
            logger.warning(f"Unknown selector code: {selector_code}, using f_classif as fallback")
            return SelectKBest(f_classif, k=n_feats)
        else:
            logger.warning(f"Unknown selector code: {selector_code}, using mutual_info_classif as fallback")
            return SelectKBest(mutual_info_classif, k=n_feats)

def get_regression_models() -> Dict[str, Any]:
    """Get dictionary of regression models."""
    return {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5)
    }

def get_classification_models() -> Dict[str, Any]:
    """Get dictionary of classification models."""
    return {
        "LogisticRegression": LogisticRegression(
            random_state=42,
            penalty='l2',
            solver='liblinear'
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),
        "SVC": SVC(
            probability=True,
            random_state=42
        )
    }

def get_model_object(model_name: str, random_state: Optional[int] = None):
    """
    Create and return a model instance based on the model name.
    
    Parameters
    ----------
    model_name : str
        Name of the model to create
    random_state : Optional[int]
        Random state for reproducibility
        
    Returns
    -------
    model
        Initialized model instance
    """
    if model_name == "RandomForestRegressor":
        model_params = MODEL_OPTIMIZATIONS["RandomForestRegressor"].copy()
        if random_state is not None:
            model_params["random_state"] = random_state
        return RandomForestRegressor(**model_params)
    elif model_name == "RandomForestClassifier":
        model_params = MODEL_OPTIMIZATIONS["RandomForestClassifier"].copy()
        if random_state is not None:
            model_params["random_state"] = random_state
        return RandomForestClassifier(**model_params)
    elif model_name == "LinearRegression":
        return LinearRegression(**MODEL_OPTIMIZATIONS["LinearRegression"])
    elif model_name == "ElasticNet":
        return ElasticNet(alpha=1.0, l1_ratio=0.5)
    elif model_name == "SVR":
        model_params = MODEL_OPTIMIZATIONS["SVR"].copy()
        # SVR does NOT support random_state
        model_params.pop("random_state", None)
        return SVR(**model_params)
    elif model_name == "LogisticRegression":
        model_params = MODEL_OPTIMIZATIONS.get("LogisticRegression", {}).copy()
        if random_state is not None:
            model_params["random_state"] = random_state
        if "penalty" not in model_params:
            model_params["penalty"] = 'l2'
        if "solver" not in model_params:
            model_params["solver"] = 'liblinear'
        return LogisticRegression(**model_params)
    elif model_name == "SVC":
        model_params = MODEL_OPTIMIZATIONS.get("SVC", {}).copy()
        if random_state is not None:
            model_params["random_state"] = random_state
        if "kernel" not in model_params:
            model_params["kernel"] = 'rbf'
        if "probability" not in model_params:
            model_params["probability"] = True
        return SVC(**model_params)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def cached_fit_transform_selector_regression(selector, X, y, n_feats, fold_idx=None, ds_name=None):
    """
    Cached version of fit_transform for regression selectors.
    
    Parameters
    ----------
    selector : object or str
        Feature selector object or selector code string
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    n_feats : int
        Number of features to select
    fold_idx : Optional[int]
        Fold index (for caching)
    ds_name : Optional[str]
        Dataset name (for caching)
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Indices of selected features and transformed data
    """
    # Convert string selector to object if needed
    if isinstance(selector, str):
        selector = get_selector_object(selector, n_feats)
    
    # Use modality-independent key for more efficient caching
    if isinstance(selector, dict):
        # Handle dictionary for Lasso/ElasticNet
        key = f"{ds_name}_{fold_idx}_{selector['type']}_{n_feats}"
        selector_type = selector['type']
    else:
        key = f"{ds_name}_{fold_idx}_{selector.__class__.__name__}_{n_feats}"
        selector_type = None
    
    if key in _selector_cache['sel_reg']:
        return _selector_cache['sel_reg'][key]
    
    try:
        # Convert to numpy arrays
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        
        # Clean NaN values
        X_arr = np.nan_to_num(X_arr, nan=0.0)
        
        # Use the standardized verify_data_alignment function for consistency
        from Z_alg._process_single_modality import verify_data_alignment
        X_arr, y_arr = verify_data_alignment(
            X_arr, y_arr, 
            name=f"{ds_name if ds_name else 'unknown'} regression selector data", 
            fold_idx=fold_idx
        )
        
        # If alignment failed, return a fallback
        if X_arr is None or y_arr is None:
            logger.warning(f"Data alignment failure in selector for {ds_name}")
            # Fallback to first feature if we can
            if X is not None and X.shape[1] > 0:
                mask = np.zeros(X.shape[1], dtype=bool)
                mask[0] = True
                selected_features = mask
                transformed_X = X[:min(len(X), 10), [0]]  # Use up to 10 samples
                _selector_cache['sel_reg'][key] = (selected_features, transformed_X)
                return selected_features, transformed_X
            return None, None
        
        # Make sure n_feats is limited by data dimensions
        max_possible_feats = min(X_arr.shape[0], X_arr.shape[1])
        effective_n_feats = min(n_feats, max_possible_feats)
        
        if effective_n_feats < n_feats:
            logger.info(f"Limiting features from {n_feats} to {effective_n_feats} due to data dimensions")
        
        # Handle MRMR using our custom implementation
        if isinstance(selector, dict) and selector_type == "mrmr_reg":
            try:
                # Try to import our custom MRMR implementation
                from Z_alg.mrmr_helper import simple_mrmr
                logger.info(f"Using custom MRMR implementation for regression")
                
                # Get selected feature indices
                selected_indices = simple_mrmr(
                    X_arr, y_arr, 
                    n_selected_features=effective_n_feats,
                    is_regression=True
                )
                
                # Convert indices to feature array and transform
                selected_features = selected_indices
                X_selected = X_arr[:, selected_indices]
                
                # Cache and return
                _selector_cache['sel_reg'][key] = (selected_features, X_selected)
                return selected_features, X_selected
            except ImportError:
                logger.warning("Custom MRMR implementation not found, using mutual_info_regression")
                selector = SelectKBest(mutual_info_regression, k=effective_n_feats)
            except Exception as e:
                logger.warning(f"MRMR error: {str(e)}, using mutual_info_regression as fallback")
                selector = SelectKBest(mutual_info_regression, k=effective_n_feats)
        
        # Handle other dictionary-based selectors
        elif isinstance(selector, dict):
            if selector_type == "lasso":
                # Manually fit Lasso first
                estimator = Lasso(alpha=0.01, random_state=42)
                estimator.fit(X_arr, y_arr)
                
                # Now create SelectFromModel with the fitted estimator
                sfm = SelectFromModel(estimator, max_features=effective_n_feats, prefit=True)
                X_selected = sfm.transform(X_arr)
                selected_features = np.arange(X_arr.shape[1])[sfm.get_support()]
                
            elif selector_type == "enet":
                # Manually fit ElasticNet first
                estimator = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42)
                estimator.fit(X_arr, y_arr)
                
                # Now create SelectFromModel with the fitted estimator
                sfm = SelectFromModel(estimator, max_features=effective_n_feats, prefit=True)
                X_selected = sfm.transform(X_arr)
                selected_features = np.arange(X_arr.shape[1])[sfm.get_support()]
                
            elif selector_type == "boruta_reg":
                # Use the stable boruta_selector
                from Z_alg.utils_boruta import boruta_selector
                selected_features = boruta_selector(
                    X_arr, y_arr, n_feats=effective_n_feats, 
                    task="reg", random_state=42
                )
                X_selected = X_arr[:, selected_features]
                
            else:
                # Fallback to mutual_info_regression
                logger.warning(f"Unknown selector type: {selector_type}, using mutual_info_regression")
                selector = SelectKBest(mutual_info_regression, k=effective_n_feats)
                X_selected = selector.fit_transform(X_arr, y_arr)
                selected_features = np.arange(X_arr.shape[1])[selector.get_support()]
            
        # Special handling for Boruta
        elif isinstance(selector, BorutaPy):
            # Use the stable boruta_selector
            from Z_alg.utils_boruta import boruta_selector
            sel_idx = boruta_selector(
                X_arr, y_arr, n_feats=effective_n_feats, 
                task="reg", random_state=42
            )
            X_selected = X_arr[:, sel_idx]
            selected_features = sel_idx
        else:
            # Standard scikit-learn selector handling for SelectKBest
            try:
                # Make sure k is updated if needed
                if hasattr(selector, 'k'):
                    selector.k = effective_n_feats
                    
                X_selected = selector.fit_transform(X_arr, y_arr)
                selected_features = np.arange(X_arr.shape[1])[selector.get_support()]
            except Exception as e:
                logger.warning(f"Error with standard selector: {str(e)}, using mutual_info_regression")
                selector = SelectKBest(mutual_info_regression, k=effective_n_feats)
                X_selected = selector.fit_transform(X_arr, y_arr)
                selected_features = np.arange(X_arr.shape[1])[selector.get_support()]
        
        _selector_cache['sel_reg'][key] = (selected_features, X_selected)
        return selected_features, X_selected
    except Exception as e:
        logger.error(f"Error in feature selection: {str(e)}")
        # Return a reasonable fallback
        max_cols = min(n_feats, X.shape[1])
        selected_features = np.arange(max_cols)
        X_selected = X[:, selected_features]
        return selected_features, X_selected

def transform_selector_regression(X, selected_features):
    """
    Transform data using selected features for regression.
    
    Parameters
    ----------
    X : array-like
        Feature matrix
    selected_features : np.ndarray
        Indices of selected features
        
    Returns
    -------
    np.ndarray
        Transformed data with only selected features
    """
    try:
        # Handle None case
        if selected_features is None:
            logger.warning("Selected features is None, returning original X")
            if isinstance(X, pd.DataFrame):
                return X.values
            return X
            
        # Make sure selected_features is valid
        if len(selected_features) == 0:
            logger.warning("No features were selected, using first feature as fallback")
            if isinstance(X, pd.DataFrame):
                return X.iloc[:, 0:1].values
            return X[:, 0:1]
            
        # Make sure selected_features indices are valid for this X
        if isinstance(X, pd.DataFrame):
            # For DataFrame, check column count
            if max(selected_features) >= X.shape[1]:
                logger.warning(f"Selected feature indices exceed X dimensions: max index {max(selected_features)} vs {X.shape[1]} columns")
                # Take only valid indices
                valid_indices = [i for i in selected_features if i < X.shape[1]]
                if len(valid_indices) == 0:
                    # If no valid indices, take first column as fallback
                    return X.iloc[:, 0:1].values
                return X.iloc[:, valid_indices].values
            return X.iloc[:, selected_features].values
        else:
            # For numpy arrays
            if max(selected_features) >= X.shape[1]:
                logger.warning(f"Selected feature indices exceed X dimensions: max index {max(selected_features)} vs {X.shape[1]} columns")
                # Take only valid indices
                valid_indices = [i for i in selected_features if i < X.shape[1]]
                if len(valid_indices) == 0:
                    # If no valid indices, take first column as fallback
                    return X[:, 0:1]
                return X[:, valid_indices]
            return X[:, selected_features]
    except Exception as e:
        logger.error(f"Error in transform_selector_regression: {str(e)}")
        # Return a safe fallback - first column
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, 0:1].values
        return X[:, 0:1]

def select_optimal_components(X, n_requested, min_explained_variance=0.8, modality_name=None, fold_idx=None, force_n_components=False):
    """
    Select optimal number of components based on explained variance.
    
    Parameters
    ----------
    X : array-like
        Input data matrix
    n_requested : int
        Requested number of components
    min_explained_variance : float
        Minimum explained variance to target (default=0.8)
    modality_name : str, optional
        Modality name for logging
    fold_idx : int, optional
        Fold index for logging
    force_n_components : bool, optional
        If True, tries to use exactly n_requested components, only limited by mathematical constraints
        
    Returns
    -------
    int
        Optimal number of components
    """
    try:
        # Ensure X is suitable for PCA
        from sklearn.preprocessing import StandardScaler
        X_centered = StandardScaler(with_std=False).fit_transform(X)
        
        # First check mathematical upper limit (min dimension)
        max_possible = min(X.shape[0], X.shape[1])
        
        # If requested components exceed maximum possible, adjust
        if n_requested > max_possible:
            logger.debug(f"Data constraint: using {max_possible} components for {modality_name or 'unknown'} in fold {fold_idx or 'unknown'} (mathematical limit)")
            n_comps = max_possible
        else:
            n_comps = n_requested
        
        # If we have very few samples, just return the adjusted number
        if X.shape[0] < 10 or X.shape[1] < 10 or force_n_components:
            return n_comps
        
        # Run incremental PCA to determine optimal components
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(max_possible, 20)).fit(X_centered)  # Cap at 20 for efficiency
        
        # Calculate cumulative explained variance 
        cum_var_ratio = np.cumsum(pca.explained_variance_ratio_)
        
        # Find the components needed to reach target variance
        for i, var_ratio in enumerate(cum_var_ratio):
            if var_ratio >= min_explained_variance:
                variance_based_comps = i + 1
                break
        else:
            # If we can't reach the target variance, use all components
            variance_based_comps = len(cum_var_ratio)
        
        # Choose whichever is smaller: requested, variance-based, or mathematical limit
        optimal_comps = min(n_comps, variance_based_comps)
        
        # Log if we're significantly reducing from the requested number
        if optimal_comps < n_requested * 0.8 and modality_name:  # Only log if >20% reduction
            logger.info(f"Optimized components: {n_requested} -> {optimal_comps} for {modality_name} (fold {fold_idx}) based on {cum_var_ratio[optimal_comps-1]:.2f} explained variance")
        
        return optimal_comps
    except Exception as e:
        # If optimization fails, fall back to the safer lower option
        fallback = min(n_requested, max_possible)
        logger.warning(f"Component optimization failed: {str(e)}. Using {fallback} components.")
        return fallback

def cached_fit_transform_extractor_classification(X, y, extractor, n_components, force_n_components=False, ds_name=None, modality_name=None, fold_idx=None):
    """
    Cached version of fit_transform for classification extractors.
    
    Parameters
    ----------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    extractor : object
        Feature extractor object
    n_components : int
        Number of components to extract
    force_n_components : bool, default=False
        If True, tries to use exactly n_components without automatic reduction
    ds_name : str
        Dataset name (for caching)
    modality_name : str
        Modality name (for caching)
    fold_idx : Optional[int]
        Fold index (for caching)
        
    Returns
    -------
    Tuple[object, np.ndarray]
        Fitted extractor and transformed data
    """
    # Generate cache key
    key = f"{ds_name}_{fold_idx}_{modality_name}_{extractor.__class__.__name__}_{n_components}"
    
    # Memory management - limit cache size to prevent memory issues
    if len(_extractor_cache['ext_clf']) > 100:
        # Get keys sorted by timestamps to remove oldest entries
        cache_items = list(_extractor_cache['ext_clf'].items())
        sorted_items = sorted(cache_items, key=lambda item: item[1][2] if len(item[1]) > 2 else 0)
        
        # Remove oldest 30% of entries
        keys_to_remove = [item[0] for item in sorted_items[:int(len(sorted_items) * 0.3)]]
        for k in keys_to_remove:
            del _extractor_cache['ext_clf'][k]
        
        logger.info(f"Cache cleanup: removed {len(keys_to_remove)} oldest entries, {len(_extractor_cache['ext_clf'])} remain")
    
    # Check if result is in cache
    if key in _extractor_cache['ext_clf']:
        return _extractor_cache['ext_clf'][key][:2]  # Return only extractor and transformed data, not timestamp
    
    try:
        # Convert DataFrame to numpy if needed
        if isinstance(X, pd.DataFrame):
            X_safe = X.fillna(0).values
        else:
            X_safe = safe_convert_to_numeric(X)
            
        # For classification, y should be integer or categorical
        if y is not None:
            try:
                # For classification, try to ensure y is integer
                if np.issubdtype(y.dtype, np.floating):
                    logger.warning(f"Warning: Converting float y to integer for classification - {modality_name} in fold {fold_idx}")
                    y_safe = np.round(y).astype(np.int32)
                else:
                    y_safe = np.asarray(y)
            except:
                logger.warning(f"Warning: Could not convert y for {modality_name} in fold {fold_idx}")
                y_safe = y
        else:
            y_safe = None
            
        # Use the standardized verify_data_alignment function for consistency
        from Z_alg._process_single_modality import verify_data_alignment
        X_safe, y_safe = verify_data_alignment(
            X_safe, y_safe, 
            name=f"{modality_name} classification extractor data", 
            fold_idx=fold_idx
        )
        
        # If alignment failed, return a fallback
        if X_safe is None or y_safe is None:
            logger.warning(f"Data alignment failure in classification extractor for {modality_name}")
            return None, None
            
        # Calculate absolute maximum possible components (minimum dimension of data)
        absolute_max = min(X_safe.shape[0], X_safe.shape[1])
        
        # Check if requested components exceed the absolute maximum
        if n_components > absolute_max:
            # This is a hard constraint we can't exceed due to math limitations
            effective_n_components = absolute_max
            logger.debug(f"Mathematical constraint: using {effective_n_components} components for {modality_name} in fold {fold_idx}")
        else:
            # If force_n_components is True, use the requested number
            effective_n_components = n_components
        
        # LDA has special constraints regardless of force_n_components
        if isinstance(extractor, LDA):
            # LDA has strict component limitations based on number of classes
            n_classes = len(np.unique(y_safe))
            max_components = min(X_safe.shape[1], n_classes - 1)
            if effective_n_components > max_components:
                global _SHOWN_LDA_MSG
                if not _SHOWN_LDA_MSG:
                    logger.info(f"Note: LDA components are limited by classes-1. This is normal and not an error.")
                    _SHOWN_LDA_MSG = True
                effective_n_components = max_components
                # Use debug level instead of info to make it less noisy
                logger.debug(f"LDA constraint: using {effective_n_components} components for {modality_name} in fold {fold_idx}")
        
        # Create a new instance of the extractor to avoid modifying the original
        if isinstance(extractor, PCA):
            new_extractor = PCA(n_components=effective_n_components, random_state=42)
        elif isinstance(extractor, FastICA):
            try:
                new_extractor = FastICA(
                    n_components=effective_n_components,
                    random_state=42,
                    max_iter=1000,
                    tol=1e-3,
                    algorithm='parallel',
                    whiten='unit-variance'
                )
            except:
                logger.warning(f"Warning: FastICA configuration failed for {modality_name}, falling back to PCA")
                new_extractor = PCA(n_components=effective_n_components, random_state=42)
        elif isinstance(extractor, NMF):
            try:
                # For NMF, ensure all values are non-negative
                X_safe = np.maximum(X_safe, 0)
                new_extractor = NMF(
                    n_components=effective_n_components,
                    init='nndsvda',
                    solver='cd',
                    random_state=42,
                    max_iter=1000,
                    tol=1e-3
                )
            except:
                logger.warning(f"Warning: NMF configuration failed for {modality_name}, falling back to PCA")
                new_extractor = PCA(n_components=effective_n_components, random_state=42)
        elif isinstance(extractor, FactorAnalysis):
            try:
                new_extractor = FactorAnalysis(
                    n_components=effective_n_components,
                    max_iter=1000,
                    tol=1e-3,
                    random_state=42
                )
            except:
                logger.warning(f"Warning: FactorAnalysis configuration failed for {modality_name}, falling back to PCA")
                new_extractor = PCA(n_components=effective_n_components, random_state=42)
        elif isinstance(extractor, LDA):
            new_extractor = LDA(n_components=effective_n_components)
        elif isinstance(extractor, KernelPCA):
            new_extractor = KernelPCA(
                n_components=effective_n_components,
                kernel='rbf',
                random_state=42
            )
        elif extractor.__class__.__name__ == 'PLSRegression':
            try:
                # Use the already imported PLSRegression from the top of the file
                new_extractor = PLSRegression(n_components=effective_n_components, max_iter=500, tol=1e-3)
                if y_safe is None:
                    logger.warning(f"PLSRegression requires y values for fit_transform. Skipping extraction for {modality_name} in fold {fold_idx}.")
                    return None, None
                X_transformed = new_extractor.fit_transform(X_safe, y_safe)[0]
                logger.info(f"PLSRegression extraction successful for {modality_name} in fold {fold_idx}: {X_transformed.shape} from {X_safe.shape}")
                X_transformed = np.nan_to_num(X_transformed, nan=0.0)
                import time
                _extractor_cache['ext_clf'][key] = (new_extractor, X_transformed, time.time())
                return new_extractor, X_transformed
            except Exception as e:
                logger.warning(f"Warning: PLSRegression configuration failed for {modality_name}, falling back to PCA. Error: {str(e)}")
                new_extractor = PCA(n_components=effective_n_components, random_state=42)
        else:
            # Default to PCA as a safe fallback
            logger.warning(f"Unknown extractor type: {type(extractor)} for {modality_name}, falling back to PCA")
            new_extractor = PCA(n_components=effective_n_components, random_state=42)
        
        # Fit and transform
        try:
            X_transformed = new_extractor.fit_transform(X_safe, y_safe)
            logger.info(f"Extraction successful for {modality_name} in fold {fold_idx}: {X_transformed.shape} from {X_safe.shape}")
        except Exception as e:
            logger.error(f"Error in feature extraction for {modality_name}: {str(e)}, falling back to PCA")
            # If extraction fails, fall back to PCA which is more robust
            try:
                safe_n_components = min(effective_n_components, X_safe.shape[1], X_safe.shape[0])
                new_extractor = PCA(n_components=safe_n_components, random_state=42)
                X_transformed = new_extractor.fit_transform(X_safe)
            except Exception as e2:
                logger.error(f"PCA fallback also failed: {str(e2)}")
                return None, None
        
        # Handle any NaN values in the transformed data
        X_transformed = np.nan_to_num(X_transformed, nan=0.0)
        
        # Verify the transformed data
        if X_transformed.shape[0] != X_safe.shape[0]:
            logger.warning(f"Warning: Transformed data has different number of samples: {X_transformed.shape[0]} vs {X_safe.shape[0]} for {modality_name}")
            return None, None
        
        # Store result in cache with timestamp for cache management
        import time
        _extractor_cache['ext_clf'][key] = (new_extractor, X_transformed, time.time())
        return new_extractor, X_transformed
    except Exception as e:
        logger.error(f"Error in feature extraction for {modality_name} in fold {fold_idx}: {str(e)}")
        return None, None

def transform_extractor_classification(X, extractor):
    """
    Transform data using fitted extractor for classification.
    
    Parameters
    ----------
    X : array-like
        Feature matrix
    extractor : object
        Fitted extractor object
        
    Returns
    -------
    np.ndarray
        Transformed data
    """
    if extractor is None:
        return None
    
    try:
        # Convert DataFrame to numpy if needed
        if isinstance(X, pd.DataFrame):
            X_safe = X.fillna(0).values
        else:
            X_safe = safe_convert_to_numeric(X)
            
        # Transform the data
        if extractor.__class__.__name__ == 'PLSRegression':
            X_transformed = extractor.transform(X_safe)
        else:
            X_transformed = extractor.transform(X_safe)
        
        # Handle any NaN values in the transformed data
        X_transformed = np.nan_to_num(X_transformed, nan=0.0)
        
        return X_transformed
    except Exception as e:
        logger.error(f"Error in transform_extractor_classification: {str(e)}")
        return None

def cached_fit_transform_extractor_regression(X, y, extractor, n_components, force_n_components=False, ds_name=None, modality_name=None, fold_idx=None):
    """
    Cached version of fit_transform for regression extractors.
    
    Parameters
    ----------
    X : array-like
        Feature matrix
    y : array-like
        Target vector (continuous values for regression)
    extractor : object
        Feature extractor object
    n_components : int
        Number of components to extract
    force_n_components : bool, default=False
        If True, tries to use exactly n_components without automatic reduction
    ds_name : str
        Dataset name (for caching)
    modality_name : str
        Modality name (for caching)
    fold_idx : Optional[int]
        Fold index (for caching)
        
    Returns
    -------
    Tuple[object, np.ndarray]
        Fitted extractor and transformed data
    """
    # Generate cache key
    key = f"{ds_name}_{fold_idx}_{modality_name}_{extractor.__class__.__name__}_{n_components}"
    
    # Memory management - limit cache size to prevent memory issues
    if len(_extractor_cache['ext_reg']) > 100:
        # Get keys sorted by timestamps to remove oldest entries
        cache_items = list(_extractor_cache['ext_reg'].items())
        sorted_items = sorted(cache_items, key=lambda item: item[1][2] if len(item[1]) > 2 else 0)
        
        # Remove oldest 30% of entries
        keys_to_remove = [item[0] for item in sorted_items[:int(len(sorted_items) * 0.3)]]
        for k in keys_to_remove:
            del _extractor_cache['ext_reg'][k]
        
        logger.info(f"Cache cleanup: removed {len(keys_to_remove)} oldest entries, {len(_extractor_cache['ext_reg'])} remain")
    
    # Check if result is in cache
    if key in _extractor_cache['ext_reg']:
        return _extractor_cache['ext_reg'][key][:2]  # Return only extractor and transformed data, not timestamp
    
    try:
        # Convert DataFrame to numpy if needed
        if isinstance(X, pd.DataFrame):
            X_safe = X.fillna(0).values
        else:
            X_safe = safe_convert_to_numeric(X)
            
        # For regression, y should be numeric
        if y is not None:
            try:
                y_safe = np.asarray(y, dtype=np.float32)
            except:
                logger.warning(f"Warning: Could not convert y to float for {modality_name} in fold {fold_idx}")
                y_safe = y
        else:
            y_safe = None
            
        # Use the standardized verify_data_alignment function for consistency
        from Z_alg._process_single_modality import verify_data_alignment
        X_safe, y_safe = verify_data_alignment(
            X_safe, y_safe, 
            name=f"{modality_name} regression extractor data", 
            fold_idx=fold_idx
        )
        
        # If alignment failed, return a fallback
        if X_safe is None or y_safe is None:
            logger.warning(f"Data alignment failure in regression extractor for {modality_name}")
            return None, None
        
        # Calculate absolute maximum possible components (minimum dimension of data)
        absolute_max = min(X_safe.shape[0], X_safe.shape[1])
        
        # Check if requested components exceed the absolute maximum
        if n_components > absolute_max:
            # This is a hard constraint we can't exceed due to math limitations
            effective_n_components = absolute_max
            logger.debug(f"Mathematical constraint: using {effective_n_components} components for {modality_name} in fold {fold_idx}")
        else:
            # If force_n_components is True, use the requested number
            effective_n_components = n_components
        
        # Create a new instance of the extractor to avoid modifying the original
        if isinstance(extractor, PCA):
            new_extractor = PCA(n_components=effective_n_components, random_state=42)
        elif isinstance(extractor, FastICA):
            try:
                new_extractor = FastICA(
                    n_components=effective_n_components,
                    random_state=42,
                    max_iter=1000,
                    tol=1e-3,
                    algorithm='parallel',
                    whiten='unit-variance'
                )
            except:
                logger.warning(f"Warning: FastICA configuration failed for {modality_name}, falling back to PCA")
                new_extractor = PCA(n_components=effective_n_components, random_state=42)
        elif isinstance(extractor, NMF):
            try:
                # For NMF, ensure all values are non-negative
                X_safe = np.maximum(X_safe, 0)
                new_extractor = NMF(
                    n_components=effective_n_components,
                    init='nndsvda',
                    solver='cd',
                    random_state=42,
                    max_iter=1000,
                    tol=1e-3
                )
            except:
                logger.warning(f"Warning: NMF configuration failed for {modality_name}, falling back to PCA")
                new_extractor = PCA(n_components=effective_n_components, random_state=42)
        elif isinstance(extractor, FactorAnalysis):
            try:
                new_extractor = FactorAnalysis(
                    n_components=effective_n_components,
                    max_iter=1000,
                    tol=1e-3,
                    random_state=42
                )
            except:
                logger.warning(f"Warning: FactorAnalysis configuration failed for {modality_name}, falling back to PCA")
                new_extractor = PCA(n_components=effective_n_components, random_state=42)
        elif extractor.__class__.__name__ == 'PLSRegression':
            try:
                # Use the already imported PLSRegression from the top of the file
                new_extractor = PLSRegression(n_components=effective_n_components, max_iter=500, tol=1e-3)
                if y_safe is None:
                    logger.warning(f"PLSRegression requires y values for fit_transform. Skipping extraction for {modality_name} in fold {fold_idx}.")
                    return None, None
                X_transformed = new_extractor.fit_transform(X_safe, y_safe)[0]
                logger.info(f"PLSRegression extraction successful for {modality_name} in fold {fold_idx}: {X_transformed.shape} from {X_safe.shape}")
                X_transformed = np.nan_to_num(X_transformed, nan=0.0)
                import time
                _extractor_cache['ext_reg'][key] = (new_extractor, X_transformed, time.time())
                return new_extractor, X_transformed
            except Exception as e:
                logger.warning(f"Warning: PLSRegression configuration failed for {modality_name}, falling back to PCA. Error: {str(e)}")
                new_extractor = PCA(n_components=effective_n_components, random_state=42)
        elif isinstance(extractor, KernelPCA):
            new_extractor = KernelPCA(
                n_components=effective_n_components,
                kernel='rbf',
                random_state=42
            )
        else:
            # Default to PCA as a safe fallback
            logger.warning(f"Unknown extractor type: {type(extractor)} for {modality_name}, falling back to PCA")
            new_extractor = PCA(n_components=effective_n_components, random_state=42)
        
        # Fit and transform
        try:
            # For regression, y is generally not used in the transformation step except for specific algorithms
            if isinstance(new_extractor, PLSRegression) and y_safe is not None:
                # PLS specific handling already done above
                pass
            else:
                X_transformed = new_extractor.fit_transform(X_safe)
            
            logger.info(f"Extraction successful for {modality_name} in fold {fold_idx}: {X_transformed.shape} from {X_safe.shape}")
        except Exception as e:
            logger.error(f"Error in feature extraction for {modality_name}: {str(e)}, falling back to PCA")
            # If extraction fails, fall back to PCA which is more robust
            try:
                safe_n_components = min(effective_n_components, X_safe.shape[1], X_safe.shape[0])
                new_extractor = PCA(n_components=safe_n_components, random_state=42)
                X_transformed = new_extractor.fit_transform(X_safe)
            except Exception as e2:
                logger.error(f"PCA fallback also failed: {str(e2)}")
                return None, None
            
        # Handle any NaN values in the transformed data
        X_transformed = np.nan_to_num(X_transformed, nan=0.0)
        
        # Verify the transformed data
        if X_transformed.shape[0] != X_safe.shape[0]:
            logger.warning(f"Warning: Transformed data has different number of samples: {X_transformed.shape[0]} vs {X_safe.shape[0]} for {modality_name}")
            return None, None
        
        # Store result in cache with timestamp for cache management
        import time
        _extractor_cache['ext_reg'][key] = (new_extractor, X_transformed, time.time())
        return new_extractor, X_transformed
    except Exception as e:
        logger.error(f"Error in feature extraction for {modality_name} in fold {fold_idx}: {str(e)}")
        return None, None

def transform_extractor_regression(X, extractor):
    """
    Transform data using fitted extractor for regression.
    
    Parameters
    ----------
    X : array-like
        Feature matrix
    extractor : object
        Fitted feature extractor
        
    Returns
    -------
    np.ndarray
        Transformed data
    """
    if extractor is None:
        return None
    
    try:
        # Convert DataFrame to numpy if needed
        if isinstance(X, pd.DataFrame):
            X_safe = X.fillna(0).values
        else:
            X_safe = safe_convert_to_numeric(X)
            
        # Transform the data
        if extractor.__class__.__name__ == 'PLSRegression':
            X_transformed = extractor.transform(X_safe)
        else:
            X_transformed = extractor.transform(X_safe)
        
        # Handle any NaN values in the transformed data
        X_transformed = np.nan_to_num(X_transformed, nan=0.0)
            
        return X_transformed
    except Exception as e:
        logger.error(f"Error in transform_extractor_regression: {str(e)}")
        return None

def cached_fit_transform_selector_classification(X, y, selector_code, n_feats, ds_name, modality_name, fold_idx=None):
    """Cached version of fit_transform for classification selectors."""
    # Use modality-independent key for more efficient caching
    if isinstance(selector_code, dict):
        key = f"{ds_name}_{modality_name}_{selector_code['type']}_{n_feats}"
        selector_type = selector_code['type']
    elif isinstance(selector_code, str):
        key = f"{ds_name}_{modality_name}_{selector_code}_{n_feats}"
        selector_type = selector_code
    else:
        # Handle case where a full selector object is passed instead of a code
        if hasattr(selector_code, '__class__'):
            key = f"{ds_name}_{modality_name}_{selector_code.__class__.__name__}_{n_feats}"
            selector_type = selector_code.__class__.__name__
        else:
            key = f"{ds_name}_{modality_name}_unknown_{n_feats}"
            selector_type = "unknown"
            
    # Check cache
    if key in _selector_cache['sel_clf']:
        logger.debug(f"Cache hit: {key}")
        return _selector_cache['sel_clf'][key]
    
    # Process selector based on type
    selected_features = None
    transformed_X = None
    
    # Log what we're doing
    logger.debug(f"Starting selector: {selector_type}, n_feats: {n_feats}, shape: {X.shape}")
    
    try:
        # Get raw data as numpy array
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        
        # Clean NaN values
        X_arr = np.nan_to_num(X_arr, nan=0.0)
        
        # Use the standardized verify_data_alignment function for consistency
        from Z_alg._process_single_modality import verify_data_alignment
        X_arr, y_arr = verify_data_alignment(
            X_arr, y_arr, 
            name=f"{modality_name} selector data", 
            fold_idx=fold_idx
        )
        
        # If alignment failed, return a fallback
        if X_arr is None or y_arr is None:
            logger.warning(f"Data alignment failure in {selector_type} for {modality_name}")
            # Fallback to first feature if we can
            if X is not None and X.shape[1] > 0:
                mask = np.zeros(X.shape[1], dtype=bool)
                mask[0] = True
                selected_features = mask
                transformed_X = X[:min(len(X), 10), [0]]  # Use up to 10 samples
                _selector_cache['sel_clf'][key] = (selected_features, transformed_X)
                return selected_features, transformed_X
            return None, None
        
        # Use the correct selector
        if selector_type == 'mrmr_clf' or selector_type == 'MRMR':
            # Handle MRMR using our own implementation
            try:
                # Try to import our custom MRMR implementation
                try:
                    from Z_alg.mrmr_helper import simple_mrmr
                    logger.info(f"Using custom MRMR implementation for {modality_name}")
                    n_features = min(n_feats, X_arr.shape[1])
                    
                    # Get selected feature indices
                    selected_indices = simple_mrmr(
                        X_arr, y_arr, 
                        n_selected_features=n_features,
                        is_regression=False
                    )
                    
                    # Convert indices to boolean mask
                    mask = np.zeros(X_arr.shape[1], dtype=bool)
                    mask[selected_indices] = True
                    
                    # Limit to valid indices
                    if len(selected_indices) > 0:
                        selected_features = mask
                        transformed_X = X_arr[:, selected_indices]
                        
                        # Cache result
                        _selector_cache['sel_clf'][key] = (selected_features, transformed_X)
                        return selected_features, transformed_X
                except ImportError:
                    logger.warning("Custom MRMR implementation not found, falling back to SelectKBest")
            except Exception as e:
                logger.warning(f"MRMR error: {str(e)}, using SelectKBest with f_classif as fallback")
            
            # Fallback to f_classif if MRMR fails
            from sklearn.feature_selection import SelectKBest, f_classif
            selector = SelectKBest(f_classif, k=min(n_feats, X_arr.shape[1]))
            logger.info(f"Using SelectKBest with f_classif for {modality_name}")
            
        elif selector_type == 'fclassifFS' or selector_type == 'fclassif':
            # Convert to SelectKBest with f_classif
            from sklearn.feature_selection import SelectKBest, f_classif
            
            selector = SelectKBest(f_classif, k=min(n_feats, X_arr.shape[1]))
            logger.info(f"Using SelectKBest with f_classif for {modality_name}")
            
        elif selector_type in ['LogisticL1', 'logistic_l1', 'ElasticNet', 'RandomForest']:
            # Model-based feature selection
            from sklearn.feature_selection import SelectFromModel
            
            if selector_type == 'LogisticL1' or selector_type == 'logistic_l1':
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
            elif selector_type == 'ElasticNet':
                from sklearn.linear_model import ElasticNet
                model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
            elif selector_type == 'RandomForest':
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                
            # Create and fit model first to avoid SelectFromModel initialization issues
            try:
                model.fit(X_arr, y_arr)
                # Then create selector
                selector = SelectFromModel(model, max_features=min(n_feats, X_arr.shape[1]), threshold=-np.inf)
            except Exception as e:
                logger.warning(f"Model fitting failed ({selector_type}): {str(e)}, using first feature as fallback")
                # Fallback to first feature
                mask = np.zeros(X_arr.shape[1], dtype=bool)
                mask[0] = True
                selected_features = mask
                transformed_X = X_arr[:, [0]]
                
                # Cache result
                _selector_cache['sel_clf'][key] = (selected_features, transformed_X)
                return selected_features, transformed_X
            
        elif selector_type == 'boruta_clf' or selector_type == 'Boruta':
            # Use the improved boruta implementation
            from Z_alg.utils_boruta import boruta_selector
            selected_features = boruta_selector(
                X_arr, y_arr, n_feats=min(n_feats, X_arr.shape[1]), 
                task="clf", random_state=42
            )
            
            if selected_features is not None and len(selected_features) > 0:
                # Convert indices to boolean mask
                mask = np.zeros(X_arr.shape[1], dtype=bool)
                mask[selected_features] = True
                selected_features = mask
                transformed_X = X_arr[:, selected_features]
                
                # Cache result
                _selector_cache['sel_clf'][key] = (selected_features, transformed_X)
                return selected_features, transformed_X
            else:
                logger.warning("Boruta returned no features, falling back to first feature")
                # Fallback to first feature
                mask = np.zeros(X_arr.shape[1], dtype=bool)
                mask[0] = True
                selected_features = mask
                transformed_X = X_arr[:, [0]]
                
                # Cache result
                _selector_cache['sel_clf'][key] = (selected_features, transformed_X)
                return selected_features, transformed_X
        
        elif selector_type == 'chi2_selection' or selector_type == 'Chi2FS':
            # Convert to SelectKBest with chi2
            from sklearn.feature_selection import SelectKBest, chi2
            
            # Ensure data is positive for chi2
            X_arr = np.abs(X_arr)
                
            selector = SelectKBest(chi2, k=min(n_feats, X_arr.shape[1]))
            logger.info(f"Using SelectKBest with chi2 for {modality_name}")
            
        elif hasattr(selector_code, 'fit') and hasattr(selector_code, 'transform'):
            # Direct selector object
            selector = selector_code
            logger.info(f"Using selector object directly: {type(selector).__name__}")
        else:
            # Unknown selector - fallback to SelectKBest with f_classif
            logger.warning(f"Unknown selector code: {selector_type}, using f_classif as fallback")
            from sklearn.feature_selection import SelectKBest, f_classif
            selector = SelectKBest(f_classif, k=min(n_feats, X_arr.shape[1]))
        
        # Fit the selector if we don't already have results
        if selected_features is None or transformed_X is None:
            # Fit selector
            try:
                selector.fit(X_arr, y_arr)
            except Exception as e:
                logger.warning(f"Selector fitting failed: {str(e)}, using first feature as fallback")
                # Fallback to first feature
                mask = np.zeros(X_arr.shape[1], dtype=bool)
                mask[0] = True
                selected_features = mask
                transformed_X = X_arr[:, [0]]
                
                # Cache and return fallback
                _selector_cache['sel_clf'][key] = (selected_features, transformed_X)
                return selected_features, transformed_X
            
            # Get selected features - handling different selector types
            if hasattr(selector, 'get_support'):
                selected_features = selector.get_support()
            elif hasattr(selector, 'support_'):
                selected_features = selector.support_
            else:
                # Last resort - try to infer from coef_ or feature_importances_
                if hasattr(selector, 'estimator_') and hasattr(selector.estimator_, 'coef_'):
                    # For L1-based linear models
                    coefs = selector.estimator_.coef_
                    if coefs.ndim > 1:
                        coefs = np.sum(np.abs(coefs), axis=0)
                    # Select top n_feats features based on coefficient magnitude
                    top_indices = np.argsort(np.abs(coefs))[-min(n_feats, len(coefs)):]
                    mask = np.zeros(X_arr.shape[1], dtype=bool)
                    mask[top_indices] = True
                    selected_features = mask
                elif hasattr(selector, 'estimator_') and hasattr(selector.estimator_, 'feature_importances_'):
                    # For tree-based models
                    importances = selector.estimator_.feature_importances_
                    # Select top n_feats features based on importance
                    top_indices = np.argsort(importances)[-min(n_feats, len(importances)):]
                    mask = np.zeros(X_arr.shape[1], dtype=bool)
                    mask[top_indices] = True
                    selected_features = mask
                else:
                    # No clear way to get support - fallback to first feature
                    logger.warning("No features selected, using first feature as fallback")
                    mask = np.zeros(X_arr.shape[1], dtype=bool)
                    mask[0] = True
                    selected_features = mask
                    
            # Check if any features were selected
            if selected_features is None or np.sum(selected_features) == 0:
                logger.warning("No features were selected, using first feature as fallback")
                mask = np.zeros(X_arr.shape[1], dtype=bool)
                mask[0] = True
                selected_features = mask
            
            # Apply transformation
            try:
                if hasattr(selector, 'transform'):
                    transformed_X = selector.transform(X_arr)
                else:
                    # Manual transform using the support mask
                    transformed_X = X_arr[:, selected_features]
            except Exception as e:
                logger.warning(f"Transformation failed: {str(e)}, using manual indexing")
                # Fallback to manual indexing
                indices = np.where(selected_features)[0]
                if len(indices) > 0:
                    transformed_X = X_arr[:, indices]
                else:
                    # If no features selected, use first feature
                    transformed_X = X_arr[:, [0]]
                    selected_features = np.zeros(X_arr.shape[1], dtype=bool)
                    selected_features[0] = True
                    
        # Cache result
        _selector_cache['sel_clf'][key] = (selected_features, transformed_X)
        return selected_features, transformed_X
        
    except Exception as e:
        logger.warning(f"Error in selector processing: {str(e)}")
        # Return a safe fallback - first feature only
        try:
            mask = np.zeros(X.shape[1], dtype=bool)
            mask[0] = True
            return mask, X.values[:, [0]] if hasattr(X, 'values') else X[:, [0]]
        except:
            # Ultimate fallback - create a single random feature
            logger.error(f"Critical failure in feature selection, returning random feature")
            return np.array([True]), np.random.rand(X.shape[0], 1)
        
def transform_selector_classification(X, selected_features):
    """
    Transform data using selected features for classification.
    
    Parameters
    ----------
    X : array-like
        Feature matrix
    selected_features : np.ndarray
        Indices of selected features
        
    Returns
    -------
    np.ndarray
        Transformed data with only selected features
    """
    try:
        # Handle None case
        if selected_features is None:
            logger.warning("Selected features is None, returning original X")
            if isinstance(X, pd.DataFrame):
                return X.values
            return X
        # Make sure selected_features is valid
        if len(selected_features) == 0:
            logger.warning("No features were selected, using first feature as fallback")
            if isinstance(X, pd.DataFrame):
                return X.iloc[:, 0:1].values
            return X[:, 0:1]
            
        # Make sure selected_features indices are valid for this X
        if isinstance(X, pd.DataFrame):
            # For DataFrame, check column count
            if max(selected_features) >= X.shape[1]:
                logger.warning(f"Selected feature indices exceed X dimensions: max index {max(selected_features)} vs {X.shape[1]} columns")
                # Take only valid indices
                valid_indices = [i for i in selected_features if i < X.shape[1]]
                if len(valid_indices) == 0:
                    # If no valid indices, take first column as fallback
                    return X.iloc[:, 0:1].values
                return X.iloc[:, valid_indices].values
            return X.iloc[:, selected_features].values
        else:
            # For numpy arrays
            if max(selected_features) >= X.shape[1]:
                logger.warning(f"Selected feature indices exceed X dimensions: max index {max(selected_features)} vs {X.shape[1]} columns")
                # Take only valid indices
                valid_indices = [i for i in selected_features if i < X.shape[1]]
                if len(valid_indices) == 0:
                    # If no valid indices, take first column as fallback
                    return X[:, 0:1]
                return X[:, valid_indices]
            return X[:, selected_features] 
    except Exception as e:
        logger.error(f"Error in transform_selector_classification: {str(e)}")
        # Return a safe fallback - first column
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, 0:1].values
        return X[:, 0:1] 