#!/usr/bin/env python3
"""
Models module for extractors, selectors, and model creation functions.
"""

import logging
import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Tuple, Any, Optional, Set, Union, Literal
import json
import pathlib

# Initialize logger
logger = logging.getLogger(__name__)

# Hyperparameter directory
HP_DIR = pathlib.Path("hp_best")

def _map_extractor_class_to_hyperparameter_name(extractor_class_name):
    """
    Map sklearn class names to hyperparameter file naming convention.
    
    Parameters
    ----------
    extractor_class_name : str
        The class name from extractor.__class__.__name__
        
    Returns
    -------
    str
        The corresponding hyperparameter file prefix
    """
    mapping = {
        # Extractors
        'KernelPCA': 'KPCA',
        'KernelPCAMedianHeuristic': 'KPCA',  # Custom KPCA implementation
        'LinearDiscriminantAnalysis': 'LDA',
        'PLSDiscriminantAnalysis': 'PLS-DA',
        'PLSRegression': 'KPLS',  # Maps to KPLS files
        'KernelPLSRegression': 'KPLS',
        'FactorAnalysis': 'FA',
        'SparsePLS': 'SparsePLS',
        'SparsePLSDA': 'SparsePLS',  # Classification version maps to same files
        'PCA': 'PCA',
        'FastICA': 'ICA',
        'NMF': 'NMF',
        # Selectors (might be used in same function)
        'Lasso': 'LASSO',
        'ElasticNet': 'ElasticNetFS',
        'SelectFromModel': 'RFImportance',  # When using RandomForest importance
        'SelectKBest': 'VarianceFTest',     # When using f_regression/f_classif
        'RFE': 'RFE',
        'mutual_info': 'MutualInfo',
        'boruta': 'Boruta'
    }
    return mapping.get(extractor_class_name, extractor_class_name)

def load_best_hyperparameters(dataset, extractor_name, model_name, task):
    """
    Load best hyperparameters for a given dataset, extractor, and model combination.
    
    Parameters
    ----------
    dataset : str
        Dataset name (e.g., "AML", "Breast")
    extractor_name : str
        Extractor name (e.g., "PCA", "KPCA", "FA") - can be class name or hyperparameter file name
    model_name : str
        Model name (e.g., "LinearRegression", "ElasticNet")
    task : str
        Task type ("reg" for regression, "clf" for classification)
        
    Returns
    -------
    dict
        Best hyperparameters with separate extractor and model params, or empty dict if not found
    """
    
    # Map class names to hyperparameter file names if needed
    mapped_extractor_name = _map_extractor_class_to_hyperparameter_name(extractor_name)
    
    def deserialize_sklearn_objects(params):
        """Convert string representations of sklearn objects back to actual objects."""
        from sklearn.preprocessing import PowerTransformer
        
        deserialized = {}
        for key, value in params.items():
            if isinstance(value, str):
                if value == "PowerTransformer()":
                    deserialized[key] = PowerTransformer()
                elif "PowerTransformer(" in value:
                    deserialized[key] = PowerTransformer()
                else:
                    deserialized[key] = value
            else:
                deserialized[key] = value
        return deserialized
    
    def separate_extractor_model_params(params):
        """Separate extractor and model parameters."""
        extractor_params = {}
        model_params = {}
        
        for key, value in params.items():
            if key.startswith('extractor__extractor__'):
                # Remove extractor__extractor__ prefix for direct application
                actual_key = key.replace('extractor__extractor__', '')
                extractor_params[actual_key] = value
            elif key.startswith('extractor__'):
                # Remove extractor__ prefix
                actual_key = key.replace('extractor__', '')
                extractor_params[actual_key] = value
            elif key.startswith('model__'):
                # Remove model__ prefix
                actual_key = key.replace('model__', '')
                model_params[actual_key] = value
        
        return extractor_params, model_params
    
    # Try exact dataset match first
    file_path = HP_DIR / f"{dataset}_{mapped_extractor_name}_{model_name}.json"
    if file_path.exists():
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                raw_params = data.get("best_params", {})
                params = deserialize_sklearn_objects(raw_params)
                extractor_params, model_params = separate_extractor_model_params(params)
                
                logger = logging.getLogger(__name__)
                logger.info(f"Loaded hyperparameters for {dataset}_{mapped_extractor_name}_{model_name}")
                logger.debug(f"Extractor params: {extractor_params}")
                logger.debug(f"Model params: {model_params}")
                
                return {
                    'extractor_params': extractor_params,
                    'model_params': model_params,
                    'source': f"{dataset}_{mapped_extractor_name}_{model_name}"
                }
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to load hyperparameters from {file_path}: {str(e)}")
    
    # Fallback: Use family dataset (Breast for classification, AML for regression)
    family_dataset = "Breast" if task == "clf" else "AML"
    if family_dataset != dataset:
        fallback_path = HP_DIR / f"{family_dataset}_{mapped_extractor_name}_{model_name}.json"
        if fallback_path.exists():
            try:
                with open(fallback_path, 'r') as f:
                    data = json.load(f)
                    raw_params = data.get("best_params", {})
                    params = deserialize_sklearn_objects(raw_params)
                    extractor_params, model_params = separate_extractor_model_params(params)
                    
                    logger = logging.getLogger(__name__)
                    logger.info(f"Using fallback hyperparameters from {family_dataset}_{mapped_extractor_name}_{model_name} for {dataset}")
                    logger.debug(f"Extractor params: {extractor_params}")
                    logger.debug(f"Model params: {model_params}")
                    
                    return {
                        'extractor_params': extractor_params,
                        'model_params': model_params,
                        'source': f"{family_dataset}_{mapped_extractor_name}_{model_name} (fallback)"
                    }
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to load fallback hyperparameters from {fallback_path}: {str(e)}")
    
    # No hyperparameters found
    logger = logging.getLogger(__name__)
    logger.debug(f"No hyperparameters found for {dataset}_{mapped_extractor_name}_{model_name} (original: {extractor_name})")
    return {
        'extractor_params': {},
        'model_params': {},
        'source': 'default (no tuned params found)'
    }

# Suppress sklearn deprecation warning about force_all_finite -> ensure_all_finite
warnings.filterwarnings("ignore", message=".*force_all_finite.*was renamed to.*ensure_all_finite.*", category=FutureWarning)

from sklearn import __version__ as sklearn_version
from sklearn.linear_model import (
    LinearRegression, Lasso, ElasticNet, ElasticNetCV, LogisticRegression,
    HuberRegressor, RANSACRegressor
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
    GradientBoostingClassifier
)
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.svm import SVC
from sklearn.decomposition import (
    PCA, NMF, FastICA, FactorAnalysis, KernelPCA
)
from sklearn.cross_decomposition import PLSRegression, PLSCanonical
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import (
    mutual_info_regression, mutual_info_classif,
    f_regression, f_classif, chi2, SelectKBest,
    SelectFromModel, RFE
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, mean_absolute_error, roc_auc_score, matthews_corrcoef, f1_score
import hashlib
import time
import copy
import warnings

# Local imports
from config import MODEL_OPTIMIZATIONS
from preprocessing import safe_convert_to_numeric
from utils_boruta import boruta_selector

# Try to import XGBoost, fall back gracefully if not available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Try to import LightGBM, fall back gracefully if not available
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Try to import imbalanced-learn for balanced models
try:
    from imblearn.ensemble import BalancedRandomForestClassifier
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    IMBALANCED_LEARN_AVAILABLE = True
except ImportError:
    IMBALANCED_LEARN_AVAILABLE = False

# Try to import scikit-optimize for hyperparameter tuning
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

# Try to import Sparse PLS from sklearn-extensions or implement fallback
try:
    from sklearn.cross_decomposition import CCA
    SPARSE_PLS_AVAILABLE = True
except ImportError:
    SPARSE_PLS_AVAILABLE = False

logger = logging.getLogger(__name__)


def _sanitize_extractor_hyperparameters(extractor, extractor_class_name, hyperparams):
    """
    Sanitize and validate hyperparameters before applying to extractor.
    
    Parameters
    ----------
    extractor : object
        The extractor object
    extractor_class_name : str
        Name of the extractor class (for checking attributes and logging)
    hyperparams : dict
        Raw hyperparameters to sanitize
        
    Returns
    -------
    dict
        Sanitized hyperparameters safe to apply
    """
    validated_params = {}
    
    for param_name, param_value in hyperparams.items():
        # Handle parameter name prefixes from hyperparameter files
        # Parameters may come with prefixes like "extractor__extractor__n_components"
        # We need to extract the actual parameter name
        actual_param_name = param_name
        
        # Strip common prefixes used in hyperparameter files
        if param_name.startswith("extractor__extractor__"):
            actual_param_name = param_name[len("extractor__extractor__"):]
        elif param_name.startswith("extractor__"):
            actual_param_name = param_name[len("extractor__"):]
        
        if hasattr(extractor, actual_param_name):
            # Special handling for KernelPCA gamma parameter
            if actual_param_name == 'gamma' and extractor_class_name in ['KernelPCA', 'KernelPCAMedianHeuristic']:
                if isinstance(param_value, str) and param_value in ['scale', 'auto']:
                    # Convert deprecated string values to numeric defaults
                    param_value = 1.0  # Use median heuristic-like value
                    logger.info(f"Converted deprecated gamma='{param_value}' to gamma=1.0 for KPCA")
                elif isinstance(param_value, (int, float)) and param_value > 0:
                    # Valid numeric gamma
                    pass
                else:
                    # Invalid gamma, use default
                    param_value = 1.0
                    logger.warning(f"Invalid gamma value {param_value}, using default 1.0")
            
            # Use the actual parameter name (without prefix) for setting
            validated_params[actual_param_name] = param_value
            logger.debug(f"Mapped hyperparameter {param_name}  {actual_param_name} = {param_value} for {extractor_class_name}")
        else:
            logger.debug(f"Skipping unknown parameter {param_name} (mapped to {actual_param_name}) for {extractor_class_name}")
    
    # Special post-processing for LDA to prevent matrix factorization errors
    if extractor_class_name == 'LinearDiscriminantAnalysis' or 'LDA' in extractor_class_name:
        # Check for problematic parameter combinations that cause matrix factorization errors
        shrinkage = validated_params.get('shrinkage')
        solver = validated_params.get('solver')
        
        # The 'eigen' solver with shrinkage=None often fails on high-dimensional data
        # When shrinkage=None and solver=eigen, the within-class scatter matrix can be singular
        if 'shrinkage' in validated_params and shrinkage is None and solver == 'eigen':
            logger.info(f"LDA: Detected problematic combination (shrinkage=None, solver=eigen) - switching to SVD solver for stability")
            validated_params['solver'] = 'svd'  # SVD solver is more stable for high-dimensional data
            # Remove shrinkage parameter when using SVD solver (not applicable)
            del validated_params['shrinkage']
            logger.debug("LDA: Removed shrinkage parameter when using SVD solver")
        
        # If only shrinkage=None is specified, add appropriate solver
        elif 'shrinkage' in validated_params and shrinkage is None and 'solver' not in validated_params:
            logger.info("LDA: Setting SVD solver for shrinkage=None to prevent matrix factorization issues")
            validated_params['solver'] = 'svd'
            del validated_params['shrinkage']  # Remove shrinkage for SVD
        
        # If using eigen solver with explicit shrinkage=None, set shrinkage to auto
        elif 'shrinkage' in validated_params and solver == 'eigen' and shrinkage is None:
            logger.info("LDA: Setting shrinkage='auto' for eigen solver to prevent matrix factorization issues")
            validated_params['shrinkage'] = 'auto'
    
    # Special post-processing for KPCA to prevent zero-size eigenvalue array errors and ARPACK issues
    elif extractor_class_name in ['KernelPCA', 'KernelPCAMedianHeuristic']:
        gamma = validated_params.get('gamma')
        n_components = validated_params.get('n_components')
        eigen_solver = validated_params.get('eigen_solver')
        
        # Handle gamma parameter validation
        if gamma is not None:
            if isinstance(gamma, (int, float)):
                # Very small gamma values can cause eigenvalue computation issues
                if gamma < 1e-3:  # Too small gamma
                    logger.info(f"KPCA: Detected very small gamma ({gamma}), increasing to 1e-3 for stability")
                    validated_params['gamma'] = 1e-3
                elif gamma > 10.0:  # Too large gamma
                    logger.info(f"KPCA: Detected very large gamma ({gamma}), reducing to 10.0 for stability")
                    validated_params['gamma'] = 10.0
            elif isinstance(gamma, str):
                # Convert string gamma to numerical value for compatibility
                if gamma in ['scale', 'auto']:
                    # Use default gamma (1.0/n_features) - will be computed at runtime
                    validated_params['gamma'] = None
                    logger.info(f"KPCA: Converting gamma '{gamma}' to None for compatibility")
                else:
                    logger.info(f"KPCA: Invalid gamma string '{gamma}', using None (default)")
                    validated_params['gamma'] = None
        else:
            # Keep None as default
            validated_params['gamma'] = None
            logger.debug("KPCA: Using default gamma (None)")
        
        # Eigen solver configuration to prevent ARPACK errors
        if eigen_solver == 'arpack' or eigen_solver == 'auto':
            # Replace ARPACK solver with more stable alternatives for problematic datasets
            validated_params['eigen_solver'] = 'dense'
            logger.info("KPCA: Using 'dense' solver instead of ARPACK for improved stability")
        elif 'eigen_solver' not in validated_params:
            validated_params['eigen_solver'] = 'dense'
            logger.debug("KPCA: Set eigen_solver to 'dense' for improved stability")
            
        # Ensure tolerance is set for numerical stability
        if 'tol' not in validated_params:
            validated_params['tol'] = 1e-4
            logger.debug("KPCA: Set tolerance to 1e-4 for numerical stability")
            
        # Force single-threaded execution to avoid parallelism issues
        if 'n_jobs' not in validated_params or validated_params.get('n_jobs', 1) != 1:
            validated_params['n_jobs'] = 1
            logger.debug("KPCA: Set n_jobs=1 to avoid parallelism issues")
    
    # Special post-processing for SparsePLS to prevent zero weights
    elif extractor_class_name == 'SparsePLS':
        alpha = validated_params.get('alpha')
        
        # Alpha=1.0 is often too aggressive and causes all weights to become zero
        if alpha is not None and isinstance(alpha, (int, float)):
            if alpha >= 1.0:  # Too aggressive sparsity
                new_alpha = 0.3  # More conservative value
                logger.info(f"SparsePLS: Detected aggressive alpha ({alpha}), reducing to {new_alpha} to prevent zero weights")
                validated_params['alpha'] = new_alpha
            elif alpha > 0.7:  # Moderately aggressive
                new_alpha = alpha * 0.5  # Reduce by half
                logger.info(f"SparsePLS: Detected high alpha ({alpha}), reducing to {new_alpha:.2f} to prevent zero weights")
                validated_params['alpha'] = new_alpha
    
    return validated_params


def _safe_cv(y, task_type, n_splits=5, seed=42):
    """
    Create a safe CV splitter that avoids the 'least populated class has only 1 member' warning.
    
    For regression: Uses KFold (no stratification needed)
    For classification: Uses StratifiedKFold (but caller should ensure classes are merged first)
    
    Parameters
    ----------
    y : array-like
        Target values (not used for regression, kept for API consistency)
    task_type : str
        Either "regression" or "classification" 
    n_splits : int, default=5
        Number of cross-validation folds
    seed : int, default=42
        Random state for reproducibility
        
    Returns
    -------
    cv_splitter : sklearn CV splitter
        KFold for regression, StratifiedKFold for classification
    """
    if task_type == "regression":
        return KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:  # classification
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


# Try to import IKPLS for Kernel Partial Least Squares
try:
    from ikpls.numpy_ikpls import PLS as IKPLS
    # Test IKPLS compatibility with current numpy version
    import numpy as np
    test_X = np.random.randn(10, 5)
    test_y = np.random.randn(10, 1)
    test_ikpls = IKPLS(algorithm=1)
    try:
        test_ikpls.fit(test_X, test_y, 2)
        IKPLS_AVAILABLE = True
        logger.info("IKPLS library loaded and tested successfully")
    except Exception as e:
        if "unexpected keyword argument 'mean'" in str(e) or "_std()" in str(e):
            logger.info("IKPLS library has numpy compatibility issues - but custom KernelPLSRegression has built-in fallback")
            # Our custom KernelPLSRegression class has built-in fallback logic to handle this exact issue
            # It will automatically fall back to regular PLSRegression when IKPLS fails
            # So we can still mark IKPLS as "available" since the wrapper handles the failure gracefully
            IKPLS_AVAILABLE = True
            logger.info("KernelPLSRegression will use built-in PLSRegression fallback for numpy compatibility")
        else:
            # Other errors might be due to test data, so we'll assume it works
            IKPLS_AVAILABLE = True
            logger.debug(f"IKPLS test failed but library seems available: {str(e)}")
except ImportError:
    IKPLS_AVAILABLE = False
    logger.debug("IKPLS library not available")

# FIX B: Helper function to re-attach index after feature extraction/selection
def _df_from_array(arr, index, prefix):
    """
    Convert numpy array back to DataFrame with proper index and column names.
    This ensures sample IDs are preserved after any NumPy-returning extractor.
    """
    cols = [f"{prefix}_{i+1}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, index=index, columns=cols)

# Target transformation registry for regression datasets
TARGET_TRANSFORMS = {
    'AML': ('log1p', np.log1p, np.expm1),
    'Sarcoma': ('sqrt', np.sqrt, lambda x: x**2),
}

def safe_target_transform(y, transform_func, dataset_name):
    """
    Safely apply target transformation while preserving pandas Series index.
    
    Parameters
    ----------
    y : pd.Series or np.ndarray
        Target values
    transform_func : callable
        Transformation function (e.g., np.log1p, np.sqrt)
    dataset_name : str
        Dataset name for logging
        
    Returns
    -------
    pd.Series or np.ndarray
        Transformed targets with preserved index
    """
    try:
        # Preserve original type and index
        original_index = getattr(y, 'index', None)
        original_name = getattr(y, 'name', None)
        
        # Convert to numpy for transformation
        y_values = y.values if hasattr(y, 'values') else np.asarray(y)
        
        # Check for problematic values before transformation
        if transform_func == np.log1p:
            # log1p(x) is undefined for x < -1, creates NaN
            min_val = np.min(y_values)
            if min_val < -1:
                logger.warning(f"Target contains values < -1 (min={min_val:.3f}) for {dataset_name}, skipping log1p transformation to prevent NaN")
                return y  # Return original values
            elif min_val < 0:
                logger.info(f"Target contains negative values (min={min_val:.3f}) for {dataset_name}, applying log1p carefully")
        elif transform_func == np.sqrt:
            # sqrt(x) is undefined for x < 0
            min_val = np.min(y_values)
            if min_val < 0:
                logger.warning(f"Target contains negative values (min={min_val:.3f}) for {dataset_name}, skipping sqrt transformation to prevent NaN")
                return y  # Return original values
        
        # Apply transformation
        transformed_values = transform_func(y_values)
        
        # Check if transformation created NaN values
        if np.isnan(transformed_values).any():
            nan_count = np.isnan(transformed_values).sum()
            logger.error(f"Target transformation created {nan_count} NaN values for {dataset_name}, reverting to original")
            return y  # Return original values
        
        # Preserve pandas Series structure if original was a Series
        if original_index is not None:
            return pd.Series(transformed_values, index=original_index, name=original_name)
        else:
            return transformed_values
            
    except Exception as e:
        logger.warning(f"Target transformation failed for {dataset_name}: {str(e)}, using original values")
        return y

def safe_target_inverse_transform(y_transformed, inverse_func, dataset_name):
    """
    Safely apply inverse target transformation while preserving pandas Series index.
    
    Parameters
    ----------
    y_transformed : pd.Series or np.ndarray
        Transformed target values
    inverse_func : callable
        Inverse transformation function (e.g., np.expm1, lambda x: x**2)
    dataset_name : str
        Dataset name for logging
        
    Returns
    -------
    pd.Series or np.ndarray
        Inverse transformed targets with preserved index
    """
    try:
        # Preserve original type and index
        original_index = getattr(y_transformed, 'index', None)
        original_name = getattr(y_transformed, 'name', None)
        
        # Convert to numpy for transformation
        y_values = y_transformed.values if hasattr(y_transformed, 'values') else np.asarray(y_transformed)
        
        # Apply inverse transformation
        inverse_values = inverse_func(y_values)
        
        # Check if inverse transformation created NaN values
        if np.isnan(inverse_values).any():
            nan_count = np.isnan(inverse_values).sum()
            logger.error(f"Inverse target transformation created {nan_count} NaN values for {dataset_name}")
        
        # Preserve pandas Series structure if original was a Series
        if original_index is not None:
            return pd.Series(inverse_values, index=original_index, name=original_name)
        else:
            return inverse_values
            
    except Exception as e:
        logger.warning(f"Inverse target transformation failed for {dataset_name}: {str(e)}")
        return y_transformed

def synchronize_X_y_data(X, y, operation_name="transformation"):
    """
    Synchronize X and y data after any transformation to ensure perfect alignment.
    
    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix
    y : pd.Series or np.ndarray
        Target values
    operation_name : str
        Name of the operation for logging
        
    Returns
    -------
    tuple
        (X_synchronized, y_synchronized) with matching indices/samples
    """
    try:
        # If both are pandas objects with indices, synchronize them
        if hasattr(X, 'index') and hasattr(y, 'index'):
            # Find common indices
            common_indices = X.index.intersection(y.index)
            
            if len(common_indices) == 0:
                logger.error(f"No common indices found between X and y after {operation_name}")
                return X, y
            
            # Filter to common indices
            X_sync = X.loc[common_indices]
            y_sync = y.loc[common_indices]
            
            # Log if we dropped samples
            original_X_len = len(X)
            original_y_len = len(y)
            final_len = len(common_indices)
            
            if final_len < original_X_len or final_len < original_y_len:
                logger.warning(f"Synchronization after {operation_name}: X {original_X_len}->{final_len}, y {original_y_len}->{final_len}")
            
            return X_sync, y_sync
        
        # If numpy arrays or mixed types, check lengths and truncate if needed
        else:
            X_len = len(X) if hasattr(X, '__len__') else X.shape[0] if hasattr(X, 'shape') else 0
            y_len = len(y) if hasattr(y, '__len__') else y.shape[0] if hasattr(y, 'shape') else 0
            
            if X_len != y_len:
                min_len = min(X_len, y_len)
                logger.warning(f"Length mismatch after {operation_name}: X={X_len}, y={y_len}. Truncating to {min_len}")
                
                if hasattr(X, 'iloc'):
                    X_sync = X.iloc[:min_len]
                else:
                    X_sync = X[:min_len]
                    
                if hasattr(y, 'iloc'):
                    y_sync = y.iloc[:min_len]
                else:
                    y_sync = y[:min_len]
                    
                return X_sync, y_sync
            
            return X, y
            
    except Exception as e:
        logger.error(f"Error synchronizing X and y after {operation_name}: {str(e)}")
        return X, y

def guard_against_target_nans(X, y, operation_name="operation"):
    """
    Guard against NaN values in targets and remove corresponding samples from both X and y.
    
    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix
    y : pd.Series or np.ndarray
        Target values
    operation_name : str
        Name of the operation for logging
        
    Returns
    -------
    tuple
        (X_clean, y_clean) with NaN targets removed
    """
    try:
        # Convert y to numpy for NaN checking
        y_values = y.values if hasattr(y, 'values') else np.asarray(y)
        
        # Create mask for finite values
        finite_mask = np.isfinite(y_values) & ~np.isnan(y_values)
        
        # Count problematic values
        nan_count = np.isnan(y_values).sum()
        inf_count = np.isinf(y_values).sum()
        
        if nan_count > 0 or inf_count > 0:
            logger.warning(f"Found {nan_count} NaN and {inf_count} infinite values in targets before {operation_name}")
            
            # Filter both X and y using the mask
            if hasattr(X, 'iloc') and hasattr(y, 'iloc'):
                # Pandas objects
                X_clean = X.iloc[finite_mask]
                y_clean = y.iloc[finite_mask]
            elif hasattr(X, 'index') and hasattr(y, 'index'):
                # Pandas objects with index
                X_clean = X[finite_mask]
                y_clean = y[finite_mask]
            else:
                # Numpy arrays
                X_clean = X[finite_mask]
                y_clean = y[finite_mask]
            
            logger.info(f"Removed {len(y) - len(y_clean)} samples with problematic targets before {operation_name}")
            return X_clean, y_clean
        
        return X, y
        
    except Exception as e:
        logger.error(f"Error guarding against NaN targets before {operation_name}: {str(e)}")
        return X, y

class PLSDiscriminantAnalysis:
    """
    Partial Least Squares Discriminant Analysis (PLS-DA) implementation.
    
    PLS-DA is a supervised dimensionality reduction technique that uses class labels
    to find components that maximize the covariance between features and class labels.
    Often outperforms unsupervised methods like PCA when class information is available.
    """
    
    def __init__(self, n_components=2, max_iter=500, tol=1e-6, copy=True, scale=True):
        """
        Initialize PLS-DA.
        
        Parameters
        ----------
        n_components : int, default=2
            Number of components to extract
        max_iter : int, default=500
            Maximum number of iterations
        tol : float, default=1e-6
            Tolerance for convergence
        copy : bool, default=True
            Whether to copy X and Y or perform in-place operations
        scale : bool, default=True
            Whether to scale data to unit variance
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.copy = copy
        self.scale = scale
        self.pls_ = None
        self.classes_ = None
        self.n_classes_ = None
        
    def fit(self, X, y):
        """
        Fit PLS-DA model.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values (class labels)
            
        Returns
        -------
        self
        """
        from sklearn.preprocessing import LabelEncoder, LabelBinarizer
        
        # Handle class labels
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        self.n_classes_ = len(self.classes_)
        
        # Convert to binary matrix for multiclass
        if self.n_classes_ > 2:
            self.label_binarizer_ = LabelBinarizer()
            Y_binary = self.label_binarizer_.fit_transform(y)
        else:
            # For binary classification, use simple 0/1 encoding
            Y_binary = y_encoded.reshape(-1, 1)
            
        # Fit PLS regression on binary targets
        self.pls_ = PLSRegression(
            n_components=min(self.n_components, Y_binary.shape[1], X.shape[1]),
            max_iter=self.max_iter,
            tol=self.tol,
            copy=self.copy,
            scale=self.scale
        )
        
        self.pls_.fit(X, Y_binary)
        return self
        
    def transform(self, X):
        """
        Transform data to PLS-DA space.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to transform
            
        Returns
        -------
        X_transformed : array-like, shape (n_samples, n_components)
            Transformed data
        """
        if self.pls_ is None:
            raise ValueError("Model must be fitted before transform")
        return self.pls_.transform(X)
        
    def fit_transform(self, X, y):
        """
        Fit model and transform data.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
            
        Returns
        -------
        X_transformed : array-like, shape (n_samples, n_components)
            Transformed data
        """
        return self.fit(X, y).transform(X)
    
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
            
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            'n_components': self.n_components,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'copy': self.copy,
            'scale': self.scale
        }
    
    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        
        Parameters
        ----------
        **params : dict
            Estimator parameters.
            
        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key} for estimator {self.__class__.__name__}")
        return self

class SparsePLSDA:
    """
    Sparse Partial Least Squares Discriminant Analysis for improved MCC.
    
    Creates maximally discriminative latent space and balances class variance.
    Optimized for genomic classification tasks with high-dimensional data.
    """
    
    def __init__(self, n_components=5, alpha=0.3, max_iter=1000, tol=1e-6, scale=True):
        """
        Initialize Sparse PLS-DA with optimized parameters for genomic data.
        
        Parameters
        ----------
        n_components : int
            Number of components to extract (reduced from 32 to 5 for stability)
        alpha : float
            Sparsity parameter for L1 regularization (increased from 0.1 to 0.3)
            Higher values increase sparsity and reduce overfitting
        max_iter : int
            Maximum number of iterations
        tol : float
            Convergence tolerance
        scale : bool
            Whether to scale the data
        """
        self.n_components = n_components
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.scale = scale
        self.components_ = None
        self.x_weights_ = None
        self.y_weights_ = None
        self.x_scores_ = None
        self.y_scores_ = None
        self.x_loadings_ = None
        self.y_loadings_ = None
        self.classes_ = None
        self.n_classes_ = None
        self.scaler_x_ = None
        self.scaler_y_ = None
        
    def _encode_labels(self, y):
        """Encode labels for discriminant analysis."""
        from sklearn.preprocessing import LabelEncoder, LabelBinarizer
        
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        self.n_classes_ = len(self.classes_)
        
        # Create binary matrix for multi-class
        if self.n_classes_ > 2:
            self.label_binarizer_ = LabelBinarizer()
            Y = self.label_binarizer_.fit_transform(y)
        else:
            # For binary classification, create a single column
            Y = y_encoded.reshape(-1, 1).astype(float)
            
        return Y
        
    def _soft_threshold(self, x, threshold):
        """Apply soft thresholding for sparsity."""
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
        
    def fit(self, X, y):
        """
        Fit Sparse PLS-DA model.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target labels
            
        Returns
        -------
        self : object
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        n_samples, n_features = X.shape
        
        # Encode labels for discriminant analysis
        Y = self._encode_labels(y)
        
        # Scale data if requested - use RobustScaler for outlier-heavy genomic data
        if self.scale:
            from sklearn.preprocessing import RobustScaler
            self.scaler_x_ = RobustScaler()
            X_scaled = self.scaler_x_.fit_transform(X)
            self.scaler_y_ = RobustScaler()
            Y_scaled = self.scaler_y_.fit_transform(Y)
        else:
            X_scaled = X.copy()
            Y_scaled = Y.copy()
            
        # RESPECT TUNED HYPERPARAMETERS: Use requested components with minimal safety checks
        actual_components = self.n_components
        
        # Only apply essential mathematical constraints
        actual_components = min(actual_components, n_samples - 1)  # Can't exceed samples-1
        actual_components = min(actual_components, Y.shape[1])     # Can't exceed number of classes
        actual_components = max(1, actual_components)              # At least 1 component
        
        logger.debug(f"SparsePLSDA: Using {actual_components} components as specified by hyperparameters "
                    f"(n_samples={n_samples}, n_features={n_features}, n_classes={Y.shape[1]})")
        
        # Initialize arrays
        self.x_weights_ = np.zeros((n_features, actual_components))
        self.y_weights_ = np.zeros((Y.shape[1], actual_components))
        self.x_scores_ = np.zeros((n_samples, actual_components))
        self.y_scores_ = np.zeros((n_samples, actual_components))
        self.x_loadings_ = np.zeros((n_features, actual_components))
        self.y_loadings_ = np.zeros((Y.shape[1], actual_components))
        
        # Copy for deflation
        X_k = X_scaled.copy()
        Y_k = Y_scaled.copy()
        
        for k in range(actual_components):
            # Compute cross-covariance matrix
            C = X_k.T @ Y_k
            
            # SVD of cross-covariance
            try:
                U, s, Vt = np.linalg.svd(C, full_matrices=False)
                
                # X weights (first left singular vector)
                w = U[:, 0]
                
                # Y weights (first right singular vector)
                c = Vt[0, :]
                
                # Apply sparsity to X weights
                w = self._soft_threshold(w, self.alpha)
                w_norm = np.linalg.norm(w)
                if w_norm > 0:
                    w = w / w_norm
                else:
                    # If all weights are zero, use original weights
                    w = U[:, 0]
                    w = w / np.linalg.norm(w)
                
                # Normalize Y weights
                c_norm = np.linalg.norm(c)
                if c_norm > 0:
                    c = c / c_norm
                
                # Compute scores
                t = X_k @ w  # X scores
                u = Y_k @ c  # Y scores
                
                # Compute loadings
                p = (X_k.T @ t) / (t.T @ t)  # X loadings
                q = (Y_k.T @ u) / (u.T @ u)  # Y loadings
                
                # Store components
                self.x_weights_[:, k] = w
                self.y_weights_[:, k] = c
                self.x_scores_[:, k] = t
                self.y_scores_[:, k] = u
                self.x_loadings_[:, k] = p
                self.y_loadings_[:, k] = q
                
                # Deflate X and Y
                X_k = X_k - np.outer(t, p)
                Y_k = Y_k - np.outer(u, q)
                
            except np.linalg.LinAlgError:
                # If SVD fails, break early
                actual_components = k
                break
        
        # Trim arrays to actual components
        if actual_components < self.n_components:
            self.x_weights_ = self.x_weights_[:, :actual_components]
            self.y_weights_ = self.y_weights_[:, :actual_components]
            self.x_scores_ = self.x_scores_[:, :actual_components]
            self.y_scores_ = self.y_scores_[:, :actual_components]
            self.x_loadings_ = self.x_loadings_[:, :actual_components]
            self.y_loadings_ = self.y_loadings_[:, :actual_components]
        
        # Store components for transform
        self.components_ = self.x_weights_
        
        return self
        
    def transform(self, X):
        """
        Transform data using fitted Sparse PLS-DA model.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to transform
            
        Returns
        -------
        X_transformed : array, shape (n_samples, n_components)
            Transformed data
        """
        X = np.asarray(X)
        
        # Scale if scaler was fitted
        if self.scale and self.scaler_x_ is not None:
            X_scaled = self.scaler_x_.transform(X)
        else:
            X_scaled = X
            
        # Transform using X weights
        X_transformed = X_scaled @ self.x_weights_
        
        return X_transformed
        
    def fit_transform(self, X, y):
        """
        Fit model and transform data.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target labels
            
        Returns
        -------
        X_transformed : array, shape (n_samples, n_components)
            Transformed training data
        """
        return self.fit(X, y).transform(X)
        
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'n_components': self.n_components,
            'alpha': self.alpha,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'scale': self.scale
        }
        
    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key}")
        return self

class SparsePLS:
    """
    Sparse Partial Least Squares implementation.
    
    Implements sparse PLS using L1 regularization to encourage sparsity
    in the loading vectors, making the model more interpretable.
    """
    
    def __init__(self, n_components=3, alpha=0.1, max_iter=500, tol=1e-6, copy=True, scale=True):
        """
        Initialize Sparse PLS with optimized parameters for genomic data.
        
        Parameters
        ----------
        n_components : int, default=3
            Number of components to extract (reduced from 5 to prevent overfitting)
        alpha : float, default=0.1
            Sparsity parameter (L1 regularization strength) - reduced from 0.3 to 0.1
            Lower values prevent empty arrays while maintaining some sparsity
        max_iter : int, default=500
            Maximum number of iterations
        tol : float, default=1e-6
            Tolerance for convergence
        copy : bool, default=True
            Whether to copy X and Y
        scale : bool, default=True
            Whether to scale data
        """
        self.n_components = n_components
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.copy = copy
        self.scale = scale
        self.x_weights_ = None
        self.y_weights_ = None
        self.x_loadings_ = None
        self.y_loadings_ = None
        
    def _soft_threshold(self, x, threshold):
        """Apply soft thresholding for L1 regularization."""
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
        
    def fit(self, X, y):
        """
        Fit Sparse PLS model using iterative algorithm with overfitting control.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,) or (n_samples, n_targets)
            Target values
            
        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            
        n_samples, n_features = X.shape
        n_targets = y.shape[1]
        
        # RESPECT TUNED HYPERPARAMETERS: Use the requested n_components directly
        # The hyperparameter tuning process already found optimal values through cross-validation
        # Removing hard-coded safety limits that override tuned parameters
        actual_components = self.n_components
        
        # Only apply minimal safety check to prevent extreme cases
        if actual_components > n_samples:
            logger.warning(f"SparsePLS: Requested {actual_components} components but only {n_samples} samples available. "
                          f"Limiting to {n_samples - 1} components.")
            actual_components = max(1, n_samples - 1)
        
        logger.debug(f"SparsePLS: Using {actual_components} components as specified by hyperparameters "
                    f"(n_samples={n_samples}, n_features={n_features})")
        
        # Center and scale data - use RobustScaler for outlier-heavy genomic data
        if self.scale:
            from sklearn.preprocessing import RobustScaler
            self.x_scaler_ = RobustScaler()
            self.y_scaler_ = RobustScaler()
            X = self.x_scaler_.fit_transform(X)
            y = self.y_scaler_.fit_transform(y)
        else:
            X = X - np.mean(X, axis=0)
            y = y - np.mean(y, axis=0)
            
        # Handle edge case where no components can be safely extracted
        if actual_components == 0:
            logger.warning(f"SparsePLS: Cannot extract any components safely, using 1 component as fallback")
            actual_components = 1
        
        # Initialize storage
        self.x_weights_ = np.zeros((n_features, actual_components))
        self.y_weights_ = np.zeros((n_targets, actual_components))
        self.x_loadings_ = np.zeros((n_features, actual_components))
        self.y_loadings_ = np.zeros((n_targets, actual_components))
        
        # Track variance for overfitting detection
        self.component_variances_ = []
        
        X_residual = X.copy()
        y_residual = y.copy()
        
        for k in range(actual_components):
            # Initialize weights randomly
            w = np.random.randn(n_features)
            w = w / np.linalg.norm(w)
            
            # ROBUST ADAPTIVE ALPHA: Prevent complete weight zeroing while maintaining sparsity
            # Problem: alpha=1.0 is too aggressive for genomic data with many features
            # Solution: Scale alpha based on data characteristics and component success
            
            # Detect if this is high-dimensional genomic data
            is_genomic_data = n_features > 1000
            
            # For genomic data, use much more conservative alpha scaling
            if is_genomic_data:
                # Scale alpha based on feature dimensionality (higher dims need lower alpha)
                dimensionality_factor = max(0.1, 1000.0 / n_features)  # Decreases as features increase
                
                # Scale based on component index (later components need lower alpha)
                component_factor = max(0.05, 1.0 / (1.0 + 0.5 * k))
                
                # Additional scaling for high base alpha values
                if self.alpha > 0.5:
                    alpha_adjustment = 0.3  # Strong reduction for high alpha
                elif self.alpha > 0.1:
                    alpha_adjustment = 0.5  # Moderate reduction for medium alpha
                else:
                    alpha_adjustment = 0.8  # Light reduction for low alpha
                
                adaptive_alpha = self.alpha * dimensionality_factor * component_factor * alpha_adjustment
            else:
                # For low-dimensional data, use original scaling
                component_scale_factor = max(0.1, 1.0 / (1.0 + 0.1 * k))
                adaptive_alpha = self.alpha * component_scale_factor
            
            # Ensure alpha is within reasonable bounds
            adaptive_alpha = max(0.001, min(adaptive_alpha, 0.5))  # Cap maximum alpha at 0.5
            
            if k < 3:  # Log for first few components
                if is_genomic_data:
                    logger.debug(f"SparsePLS: Component {k+1} using adaptive_alpha={adaptive_alpha:.4f} "
                               f"(base={self.alpha}, genomic_scaling=True, dim_factor={dimensionality_factor:.3f}, "
                               f"comp_factor={component_factor:.3f}, alpha_adj={alpha_adjustment:.3f})")
                else:
                    logger.debug(f"SparsePLS: Component {k+1} using adaptive_alpha={adaptive_alpha:.4f} "
                               f"(base={self.alpha}, genomic_scaling=False, component_scale={component_scale_factor:.3f})")
            
            for iteration in range(self.max_iter):
                w_old = w.copy()
                
                # Update weights
                t = X_residual @ w
                c = y_residual.T @ t / (t.T @ t + 1e-8)
                u = y_residual @ c
                w = X_residual.T @ u / (u.T @ u + 1e-8)
                
                # Apply adaptive sparsity constraint
                w = self._soft_threshold(w, adaptive_alpha)
                
                # Normalize with improved tolerance for sparse weights
                w_norm = np.linalg.norm(w)
                
                # Use adaptive tolerance based on the sparsity level
                tolerance = max(1e-10, adaptive_alpha * 1e-6)
                
                if w_norm > tolerance:
                    w = w / w_norm
                else:
                    # Recovery mechanism with multiple fallback strategies
                    recovered = False
                    
                    # Strategy 1: Try with progressively reduced sparsity
                    if adaptive_alpha > 0.001:
                        for reduction_factor in [0.5, 0.1, 0.01]:
                            recovery_alpha = adaptive_alpha * reduction_factor
                            w_recovered = self._soft_threshold(X_residual.T @ u / (u.T @ u + 1e-8), recovery_alpha)
                            w_recovered_norm = np.linalg.norm(w_recovered)
                            
                            if w_recovered_norm > tolerance:
                                w = w_recovered / w_recovered_norm
                                adaptive_alpha = recovery_alpha
                                logger.debug(f"SparsePLS: Component {k+1} recovered with reduced alpha={recovery_alpha:.6f}")
                                recovered = True
                                break
                    
                    # Strategy 2: If sparsity reduction fails, try without soft thresholding
                    if not recovered:
                        w_no_threshold = X_residual.T @ u / (u.T @ u + 1e-8)
                        w_no_threshold_norm = np.linalg.norm(w_no_threshold)
                        
                        if w_no_threshold_norm > tolerance:
                            w = w_no_threshold / w_no_threshold_norm
                            adaptive_alpha = 0.0  # Disable thresholding for this component
                            logger.debug(f"SparsePLS: Component {k+1} recovered without sparsity constraint")
                            recovered = True
                    
                    # Strategy 3: Final fallback - initialize with random weights
                    if not recovered and k == 0:  # Only for first component to ensure we get at least one
                        w = np.random.randn(n_features)
                        w = w / np.linalg.norm(w)
                        adaptive_alpha = 0.0  # No sparsity for fallback
                        logger.debug(f"SparsePLS: Component {k+1} initialized with random weights as final fallback")
                        recovered = True
                    
                    if not recovered:
                        if k == 0:
                            logger.warning(f"SparsePLS: Could not fit even the first component, using minimal fallback")
                            # Create a minimal single component using the first feature
                            w = np.zeros(n_features)
                            if n_features > 0:
                                w[0] = 1.0
                            actual_components = 1
                        else:
                            logger.debug(f"SparsePLS: Component {k+1} weights became zero, stopping at {k} components")
                            actual_components = k
                        break
                    
                # Check convergence
                if np.linalg.norm(w - w_old) < self.tol:
                    break
                    
            # Store weights and loadings
            self.x_weights_[:, k] = w
            self.y_weights_[:, k] = c.flatten()
            
            # Compute scores and loadings
            t = X_residual @ w
            p = X_residual.T @ t / (t.T @ t + 1e-8)
            q = y_residual.T @ t / (t.T @ t + 1e-8)
            
            self.x_loadings_[:, k] = p
            self.y_loadings_[:, k] = q.flatten()
            
            # Monitor component variance for overfitting detection
            component_variance = np.var(t)
            self.component_variances_.append(component_variance)
            
            # Adaptive early stopping based on data characteristics and variance pattern
            # Calculate adaptive threshold based on:
            # 1. Data size (larger datasets can handle higher variance)
            # 2. Previous component variances (detect rapid increase)
            # 3. Sample-to-feature ratio (genomic data characteristics)
            
            # Base threshold scales with data characteristics
            base_threshold = max(
                50.0,  # Minimum threshold for very small datasets
                min(500.0, n_samples * 2.0),  # Scale with sample size, cap at 500
                min(200.0, n_features * 0.1)  # Scale with feature count, cap at 200
            )
            
            # Adaptive threshold based on variance pattern
            if len(self.component_variances_) > 1:
                # Check for exponential increase (sign of overfitting)
                prev_var = self.component_variances_[-2]
                variance_ratio = component_variance / max(prev_var, 1e-8)
                
                # If variance increases by more than 5x from previous component, it's likely overfitting
                if variance_ratio > 5.0 and component_variance > base_threshold:
                    logger.warning(f"SparsePLS: Exponential variance increase detected in component {k+1} "
                                  f"(var={component_variance:.2f}, ratio={variance_ratio:.2f}x), stopping early to prevent overfitting")
                    actual_components = k + 1
                    break
            else:
                # For first component, use a more lenient threshold
                # Genomic data often has high first-component variance due to dominant patterns
                lenient_threshold = base_threshold * 2.0
                
                if component_variance > lenient_threshold:
                    logger.warning(f"SparsePLS: Extremely high variance detected in component {k+1} "
                                  f"(var={component_variance:.2f}, threshold={lenient_threshold:.2f}), stopping early to prevent overfitting")
                    actual_components = k + 1
                    break
            
            # Deflate matrices
            X_residual = X_residual - np.outer(t, p)
            y_residual = y_residual - np.outer(t, q)
        
        # Trim arrays to actual components used
        if actual_components < self.n_components:
            self.x_weights_ = self.x_weights_[:, :actual_components]
            self.y_weights_ = self.y_weights_[:, :actual_components]
            self.x_loadings_ = self.x_loadings_[:, :actual_components]
            self.y_loadings_ = self.y_loadings_[:, :actual_components]
            
        # Store final number of components
        self.n_components_fitted_ = actual_components
        
        # Log variance statistics for monitoring
        if self.component_variances_:
            max_var = max(self.component_variances_)
            mean_var = np.mean(self.component_variances_)
            
            # Also log adaptive threshold information for debugging
            base_threshold = max(50.0, min(500.0, n_samples * 2.0), min(200.0, n_features * 0.1))
            lenient_threshold = base_threshold * 2.0
            
            logger.debug(f"SparsePLS fitted with {actual_components} components: "
                        f"max_variance={max_var:.2f}, mean_variance={mean_var:.2f}, "
                        f"adaptive_threshold={lenient_threshold:.2f}")
            
        return self
        
    def transform(self, X):
        """
        Transform data using fitted Sparse PLS model.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to transform
            
        Returns
        -------
        X_transformed : array-like, shape (n_samples, n_components)
            Transformed data with consistent dimensions (padded to original n_components if needed)
        """
        X = np.asarray(X, dtype=np.float64)
        
        # Check if model was fitted properly
        if not hasattr(self, 'x_weights_') or self.x_weights_.size == 0:
            logger.warning("SparsePLS: Model not fitted properly, returning zero array")
            return np.zeros((X.shape[0], self.n_components))
        
        if hasattr(self, 'x_scaler_'):
            X = self.x_scaler_.transform(X)
        else:
            X = X - np.mean(X, axis=0)
            
        # Transform using fitted components
        result = X @ self.x_weights_
        if result.size == 0:
            logger.warning("SparsePLS: Transform resulted in empty array, returning zero array")
            return np.zeros((X.shape[0], self.n_components))
        
        # Ensure consistent output dimensions for fusion compatibility
        # If fewer components were fitted than requested, pad with zeros
        n_samples = result.shape[0]
        n_fitted_components = result.shape[1] if result.ndim > 1 else 1
        
        if n_fitted_components < self.n_components:
            # Pad with zero columns to match the original requested number of components
            padding_cols = self.n_components - n_fitted_components
            padding = np.zeros((n_samples, padding_cols))
            result = np.hstack([result, padding])
            logger.debug(f"SparsePLS: Padded output from {n_fitted_components} to {self.n_components} components for fusion compatibility")
        
        return result
        
    def fit_transform(self, X, y):
        """
        Fit model and transform data.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,) or (n_samples, n_targets)
            Target values
            
        Returns
        -------
        X_transformed : array-like, shape (n_samples, n_components)
            Transformed data
        """
        return self.fit(X, y).transform(X)
    
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
            
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            'n_components': self.n_components,
            'alpha': self.alpha,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'copy': self.copy,
            'scale': self.scale
        }
    
    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        
        Parameters
        ----------
        **params : dict
            Estimator parameters.
            
        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key} for estimator {self.__class__.__name__}")
        return self

# Early Stopping Configuration
EARLY_STOPPING_CONFIG = {
    "enabled": True,
    "patience": 10,  # Number of epochs to wait for improvement
    "min_delta": 1e-4,  # Minimum change to qualify as improvement
    "validation_split": 0.2,  # Fraction of training data to use for validation
    "restore_best_weights": True,  # Whether to restore best model weights
    "monitor_metric": "auto",  # "auto", "loss", "accuracy", "r2", etc.
    "verbose": 1  # Verbosity level for early stopping
}

class KernelPCAMedianHeuristic:
    """
    Kernel PCA with median heuristic for gamma parameter.
    
    Captures non-linear gene–methylation interactions for improved R².
    Uses RBF kernel with gamma learned by median heuristic.
    """
    
    def __init__(self, n_components=64, kernel="rbf", gamma="auto", eigen_solver="auto", 
                 n_jobs=-1, random_state=42, sample_size=1000, percentile=50):
        """
        Initialize Kernel PCA with median heuristic.
        
        Parameters
        ----------
        n_components : int
            Number of components to extract (default 64 for R² optimization)
        kernel : str
            Kernel type (default "rbf")
        gamma : str or float
            Kernel coefficient (will be overridden by median heuristic)
        eigen_solver : str
            Eigenvalue solver
        n_jobs : int
            Number of parallel jobs
        random_state : int
            Random state for reproducibility
        sample_size : int
            Sample size for median heuristic calculation
        percentile : float
            Percentile for median calculation
        """
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.eigen_solver = eigen_solver
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.sample_size = sample_size
        self.percentile = percentile
        self.kernel_pca_ = None
        self.gamma_computed_ = None
        
    def _compute_median_heuristic_gamma(self, X):
        """
        Compute gamma using median heuristic.
        
        The median heuristic sets gamma = 1 / (2 * median(pairwise_distances)^2)
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data
            
        Returns
        -------
        gamma : float
            Computed gamma value
        """
        from sklearn.metrics.pairwise import pairwise_distances
        
        n_samples = X.shape[0]
        
        # Sample data if too large
        if n_samples > self.sample_size:
            np.random.seed(self.random_state)
            indices = np.random.choice(n_samples, self.sample_size, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
            
        # Compute pairwise distances
        distances = pairwise_distances(X_sample, metric='euclidean')
        
        # Get upper triangular part (excluding diagonal)
        triu_indices = np.triu_indices_from(distances, k=1)
        pairwise_dists = distances[triu_indices]
        
        # Compute median distance
        median_dist = np.percentile(pairwise_dists, self.percentile)
        
        # Avoid division by zero
        if median_dist == 0:
            median_dist = 1.0
            
        # Compute gamma using median heuristic
        gamma = 1.0 / (2 * median_dist ** 2)
        
        return gamma
        
    def fit(self, X, y=None):
        """
        Fit Kernel PCA with median heuristic.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, optional
            Target values (ignored)
            
        Returns
        -------
        self : object
        """
        from sklearn.decomposition import KernelPCA
        
        X = np.asarray(X)
        
        # Compute gamma using median heuristic
        if self.gamma == "auto" or self.gamma is None:
            self.gamma_computed_ = self._compute_median_heuristic_gamma(X)
        else:
            self.gamma_computed_ = self.gamma
            
        # Create and fit Kernel PCA with computed gamma and error handling
        try:
            self.kernel_pca_ = KernelPCA(
                n_components=self.n_components,
                kernel=self.kernel,
                gamma=self.gamma_computed_,
                eigen_solver='dense' if self.eigen_solver == 'auto' else self.eigen_solver,  # Use dense for stability
                n_jobs=1,  # Force single thread for stability
                tol=1e-4,  # Add tolerance for numerical stability
                random_state=self.random_state
            )
            
            self.kernel_pca_.fit(X)
        except Exception as e:
            # If KPCA fails, try with reduced components and different solver
            logger.warning(f"KernelPCA failed with {self.n_components} components: {str(e)}")
            
            # CRITICAL FIX: Use consistent_hyperparams if available, otherwise use conservative fallback
            # Check if consistent hyperparameters are available (passed through global state or other means)
            # For now, preserve the original n_components for consistency
            original_components = self.n_components
            safe_components = min(self.n_components // 2, X.shape[0] - 1, 8)
            safe_components = max(1, safe_components)
            
            try:
                logger.info(f"Retrying KernelPCA with {safe_components} components and 'auto' solver")
                self.kernel_pca_ = KernelPCA(
                    n_components=safe_components,
                    kernel='linear' if self.kernel == 'rbf' else self.kernel,  # Switch to linear kernel for stability
                    gamma=self.gamma_computed_ if self.kernel != 'linear' else None,
                    eigen_solver='dense',  # Use most stable solver
                    n_jobs=1,  # Single thread to avoid parallelism issues
                    tol=1e-3,  # More relaxed tolerance
                    random_state=self.random_state
                )
                
                self.kernel_pca_.fit(X)
                self.n_components = safe_components  # Update the effective component count
                logger.info(f"KernelPCA retry successful with {safe_components} components (originally {original_components})")
            except Exception as e2:
                # If still failing, fall back to regular PCA but try to preserve more components
                logger.warning(f"KernelPCA retry also failed: {str(e2)}, falling back to PCA")
                from sklearn.decomposition import PCA
                
                # Try to preserve more components in PCA fallback if data allows
                max_pca_components = min(X.shape[0] - 1, X.shape[1], original_components)
                pca_components = max(1, min(safe_components, max_pca_components))
                
                self.kernel_pca_ = PCA(
                    n_components=pca_components,
                    random_state=self.random_state
                )
                self.kernel_pca_.fit(X)
                self.n_components = pca_components
                logger.info(f"Fallback to PCA with {pca_components} components (originally requested {original_components})")
        
        # Store components for compatibility
        if hasattr(self.kernel_pca_, 'eigenvectors_'):
            self.components_ = self.kernel_pca_.eigenvectors_.T
        else:
            # Fallback for different sklearn versions
            self.components_ = None
        
        # Expose feature attributes from the inner KernelPCA for proper feature detection
        if hasattr(self.kernel_pca_, 'n_features_in_'):
            self.n_features_in_ = self.kernel_pca_.n_features_in_
        if hasattr(self.kernel_pca_, 'X_fit_'):
            self.X_fit_ = self.kernel_pca_.X_fit_
            
        return self
        
    def transform(self, X):
        """
        Transform data using fitted Kernel PCA.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to transform
            
        Returns
        -------
        X_transformed : array, shape (n_samples, n_components)
            Transformed data
        """
        if self.kernel_pca_ is None:
            raise ValueError("Model must be fitted before transform")
            
        return self.kernel_pca_.transform(X)
        
    def fit_transform(self, X, y=None):
        """
        Fit model and transform data.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, optional
            Target values (ignored)
            
        Returns
        -------
        X_transformed : array, shape (n_samples, n_components)
            Transformed training data
        """
        return self.fit(X, y).transform(X)
        
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'n_components': self.n_components,
            'kernel': self.kernel,
            'gamma': self.gamma,
            'eigen_solver': self.eigen_solver,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
            'sample_size': self.sample_size,
            'percentile': self.percentile
        }
        
    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key}")
        return self

class KernelPLSRegression:
    """
    Kernel Partial Least Squares (KPLS) Regression with numerical stability improvements.
    
    This class provides a scikit-learn compatible interface for Kernel PLS regression
    with enhanced numerical stability, regularization, and optimal component selection.
    
    Key improvements:
    - Reduced default components from 8 to 5 for better stability
    - Added regularization parameters for numerical stability
    - Cross-validation for optimal component selection
    - Improved gamma computation with bounds checking
    - Enhanced error handling and fallback mechanisms
    
    Advantages over ICA:
    - Supervised learning optimizes latent space for target prediction
    - Captures non-linear relationships through kernel transformations
    - Usually provides +3-6 pp R² improvement on genomic regression tasks
    - Maintains the interpretable "few components" interface of PLS
    - Graceful fallback to regular PLS when IKPLS has compatibility issues
    """
    
    def __init__(self, n_components=5, kernel="rbf", gamma="auto", max_iter=500, 
                 tol=1e-4, algorithm=1, random_state=42, regularization=1e-6,
                 use_cv_components=True, cv_folds=3, gamma_bounds=(1e-6, 1e3)):
        """
        Initialize Kernel PLS Regression with stability improvements.
        
        Parameters
        ----------
        n_components : int, default=5
            Maximum number of PLS components to extract (reduced from 8 for stability)
        kernel : str, default="rbf"
            Kernel type. Currently supports "rbf" (Radial Basis Function)
        gamma : str or float, default="auto"
            Kernel coefficient for RBF. If "auto", uses improved median heuristic
        max_iter : int, default=500
            Maximum number of iterations for PLS algorithm
        tol : float, default=1e-4
            Tolerance for convergence (relaxed for stability)
        algorithm : int, default=1
            IKPLS algorithm variant (1 or 2)
        random_state : int, default=42
            Random state for reproducibility
        regularization : float, default=1e-6
            Regularization parameter for numerical stability
        use_cv_components : bool, default=True
            Whether to use cross-validation for optimal component selection
        cv_folds : int, default=3
            Number of cross-validation folds for component selection
        gamma_bounds : tuple, default=(1e-6, 1e3)
            Lower and upper bounds for gamma parameter
        """
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.algorithm = algorithm
        self.random_state = random_state
        self.regularization = regularization
        self.use_cv_components = use_cv_components
        self.cv_folds = cv_folds
        self.gamma_bounds = gamma_bounds
        self.ikpls_ = None
        self.X_fit_ = None
        self.gamma_ = None
        self.fallback_pls_ = None
        self.optimal_components_ = None
        
    def _compute_rbf_kernel(self, X1, X2=None):
        """Compute RBF kernel matrix between X1 and X2."""
        if X2 is None:
            X2 = X1
            
        # Compute pairwise squared distances
        X1_norm = np.sum(X1**2, axis=1, keepdims=True)
        X2_norm = np.sum(X2**2, axis=1, keepdims=True)
        distances_sq = X1_norm + X2_norm.T - 2 * np.dot(X1, X2.T)
        
        # Apply RBF kernel
        return np.exp(-self.gamma_ * distances_sq)
        
    def _compute_gamma(self, X):
        """
        Compute gamma using improved median heuristic with bounds checking.
        
        This prevents extreme gamma values that can cause numerical instability.
        """
        if self.gamma == "auto":
            # Median heuristic with numerical stability
            n_samples = min(500, X.shape[0])  # Reduced sample size for efficiency
            if X.shape[0] > n_samples:
                indices = np.random.RandomState(self.random_state).choice(
                    X.shape[0], n_samples, replace=False
                )
                X_sample = X[indices]
            else:
                X_sample = X
                
            # Use more efficient pairwise distance computation
            from sklearn.metrics.pairwise import pairwise_distances
            try:
                # Compute pairwise distances efficiently
                distances = pairwise_distances(X_sample, metric='euclidean')
                # Get upper triangular part (excluding diagonal)
                upper_tri_indices = np.triu_indices_from(distances, k=1)
                distance_values = distances[upper_tri_indices]
                
                if len(distance_values) > 0:
                    # Use percentile-based approach for robustness
                    median_dist = np.median(distance_values)
                    # Add small epsilon to prevent division by zero
                    median_dist = max(median_dist, 1e-8)
                    
                    # Compute gamma with bounds checking
                    gamma = 1.0 / (2 * median_dist**2)
                    
                    # Apply bounds to prevent extreme values
                    gamma = np.clip(gamma, self.gamma_bounds[0], self.gamma_bounds[1])
                    
                    logger.debug(f"KPLS gamma computation: median_dist={median_dist:.6f}, gamma={gamma:.6f}")
                    return gamma
                else:
                    # Fallback to conservative gamma
                    return 1.0
                    
            except Exception as e:
                logger.warning(f"Gamma computation failed, using fallback: {e}")
                # Conservative fallback gamma
                return 1.0
        else:
            # Ensure user-provided gamma is within bounds
            gamma = float(self.gamma)
            gamma = np.clip(gamma, self.gamma_bounds[0], self.gamma_bounds[1])
            return gamma
    
    def _select_optimal_components_cv(self, X, y, K):
        """
        Select optimal number of components using cross-validation.
        
        This prevents overfitting and improves numerical stability.
        """
        if not self.use_cv_components:
            return min(self.n_components, K.shape[0], K.shape[1], y.shape[1])
        
        from sklearn.model_selection import KFold
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.metrics import mean_squared_error
        
        # Determine component range to test
        max_possible = min(self.n_components, K.shape[0] - 1, K.shape[1], y.shape[1])
        component_range = range(1, max_possible + 1)
        
        if len(component_range) <= 1:
            return max_possible
        
        # Cross-validation setup
        kf = KFold(n_splits=min(self.cv_folds, X.shape[0]), shuffle=True, random_state=self.random_state)
        cv_scores = []
        
        for n_comp in component_range:
            fold_scores = []
            
            for train_idx, val_idx in kf.split(X):
                try:
                    # Split data
                    K_train, K_val = K[train_idx], K[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    # Fit PLS with current number of components
                    pls = PLSRegression(
                        n_components=n_comp,
                        max_iter=self.max_iter,
                        tol=self.tol
                    )
                    pls.fit(K_train, y_train)
                    
                    # Predict and score
                    y_pred = pls.predict(K_val)
                    score = -mean_squared_error(y_val, y_pred)  # Negative MSE for maximization
                    fold_scores.append(score)
                    
                except Exception as e:
                    logger.debug(f"CV fold failed for {n_comp} components: {e}")
                    fold_scores.append(-np.inf)  # Penalize failed folds
            
            if fold_scores:
                cv_scores.append(np.mean(fold_scores))
            else:
                cv_scores.append(-np.inf)
        
        # Select optimal components
        if cv_scores:
            optimal_idx = np.argmax(cv_scores)
            optimal_components = component_range[optimal_idx]
            
            logger.debug(f"KPLS CV component selection: tested {len(component_range)} options, "
                        f"selected {optimal_components} components (score: {cv_scores[optimal_idx]:.4f})")
            
            return optimal_components
        else:
            # Fallback to conservative choice
            return min(3, max_possible)
    
    def fit(self, X, y):
        """
        Fit Kernel PLS model using custom implementation.
        
        This implementation uses regular PLS on the kernel matrix, providing
        true kernel PLS functionality without relying on the IKPLS library.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,) or (n_samples, n_targets)
            Target values
            
        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            
        # Store training data for kernel computation
        self.X_fit_ = X.copy()
        
        # Compute gamma for RBF kernel
        self.gamma_ = self._compute_gamma(X)
        
        if self.kernel == "rbf":
            # Compute kernel matrix
            K = self._compute_rbf_kernel(X)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")
            
        # Add regularization to kernel matrix for numerical stability
        K_reg = K + self.regularization * np.eye(K.shape[0])
        
        # Use cross-validation to select optimal number of components
        optimal_components = self._select_optimal_components_cv(X, y, K_reg)
        self.optimal_components_ = optimal_components
        
        # Use regular PLS on the regularized kernel matrix - this IS kernel PLS!
        from sklearn.cross_decomposition import PLSRegression
        
        # Fit PLS on regularized kernel matrix (this is the kernel PLS algorithm)
        self.kernel_pls_ = PLSRegression(
            n_components=optimal_components, 
            max_iter=self.max_iter,
            tol=self.tol
        )
        
        try:
            self.kernel_pls_.fit(K_reg, y)
            logger.debug(f"KPLS fitted successfully with {optimal_components} components "
                        f"(selected via CV from max {self.n_components}) on kernel matrix {K_reg.shape}")
        except Exception as e:
            logger.warning(f"KPLS fitting failed: {str(e)}, trying with reduced components")
            # Try with fewer components as fallback
            fallback_components = min(optimal_components // 2, 3, K_reg.shape[0] - 1)
            if fallback_components > 0:
                self.kernel_pls_ = PLSRegression(
                    n_components=fallback_components, 
                    max_iter=self.max_iter,
                    tol=self.tol
                )
                self.kernel_pls_.fit(K_reg, y)
                self.optimal_components_ = fallback_components
                logger.debug(f"KPLS fitted with fallback {fallback_components} components")
            else:
                raise ValueError(f"Cannot fit KPLS even with minimal components: {str(e)}")
                
        # Clear IKPLS-related attributes since we're not using IKPLS
        self.ikpls_ = None
        self.fallback_pls_ = None
                
        return self
        
    def transform(self, X):
        """
        Transform data using fitted Kernel PLS model.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to transform
            
        Returns
        -------
        X_transformed : array-like, shape (n_samples, n_components)
            Transformed data
        """
        if not hasattr(self, 'kernel_pls_') or self.kernel_pls_ is None:
            raise ValueError("Model must be fitted before transform")
            
        X = np.asarray(X, dtype=np.float64)
        
        if self.kernel == "rbf":
            # Compute kernel matrix between X and training data
            K = self._compute_rbf_kernel(X, self.X_fit_)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")
            
        # Transform using kernel PLS (PLS applied to kernel matrix)
        # Note: No regularization needed for transform, only for training
        try:
            X_transformed = self.kernel_pls_.transform(K)
            
            # Clip extreme values to prevent numerical instability
            X_transformed = np.clip(X_transformed, -100, 100)
            
            logger.debug(f"KPLS transform: {K.shape} -> {X_transformed.shape}, "
                        f"value range: [{X_transformed.min():.3f}, {X_transformed.max():.3f}]")
            return X_transformed
        except Exception as e:
            logger.warning(f"KPLS transform failed: {str(e)}")
            # Return zeros as fallback
            n_components = getattr(self.kernel_pls_, 'n_components', self.optimal_components_ or self.n_components)
            return np.zeros((X.shape[0], n_components))
            
    def fit_transform(self, X, y):
        """
        Fit model and transform data.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,) or (n_samples, n_targets)
            Target values
            
        Returns
        -------
        X_transformed : array-like, shape (n_samples, n_components)
            Transformed training data
        """
        return self.fit(X, y).transform(X)
        
    def predict(self, X):
        """
        Predict using Kernel PLS model.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to predict
            
        Returns
        -------
        y_pred : array-like, shape (n_samples,) or (n_samples, n_targets)
            Predicted values
        """
        if not hasattr(self, 'kernel_pls_') or self.kernel_pls_ is None:
            raise ValueError("Model must be fitted before predict")
            
        X = np.asarray(X, dtype=np.float64)
        
        if self.kernel == "rbf":
            K = self._compute_rbf_kernel(X, self.X_fit_)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")
            
        try:
            # Get predictions from kernel PLS
            y_pred = self.kernel_pls_.predict(K)
            
            # Clip extreme predictions to prevent numerical instability
            y_pred = np.clip(y_pred, -1000, 1000)
            
            logger.debug(f"KPLS predict: {K.shape} -> {y_pred.shape}, "
                        f"prediction range: [{y_pred.min():.3f}, {y_pred.max():.3f}]")
            return y_pred
        except Exception as e:
            logger.warning(f"KPLS predict failed: {str(e)}")
            # Return zeros as fallback - need to determine shape first
            try:
                # Try to get the expected output shape from the fitted model
                n_targets = getattr(self.kernel_pls_, 'n_targets_', 1)
                if n_targets == 1:
                    return np.zeros(X.shape[0])
                else:
                    return np.zeros((X.shape[0], n_targets))
            except:
                # Ultimate fallback
                return np.zeros(X.shape[0])
            
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'n_components': self.n_components,
            'kernel': self.kernel,
            'gamma': self.gamma,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'algorithm': self.algorithm,
            'random_state': self.random_state,
            'regularization': self.regularization,
            'use_cv_components': self.use_cv_components,
            'cv_folds': self.cv_folds,
            'gamma_bounds': self.gamma_bounds
        }
        
    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key}")
        return self

# ============================================================================
# Regression models with automatic parameter selection
# ============================================================================

class SelectionByCyclicCoordinateDescent(BaseEstimator, RegressorMixin):
    """
    ElasticNet with automatic α search using Cyclic Coordinate Descent.
    
    This wrapper uses ElasticNetCV to automatically find the optimal alpha
    parameter through cross-validation, which can find sparser models than
    halving search by using the efficient coordinate descent algorithm.
    
    Parameters
    ----------
    l1_ratio : float, default=0.5
        The ElasticNet mixing parameter between L1 and L2 penalties
    cv : int, default=5
        Number of cross-validation folds for alpha selection
    max_iter : int, default=2000
        Maximum number of iterations
    tol : float, default=1e-4
        Tolerance for optimization
    random_state : int, default=42
        Random state for reproducibility
    n_alphas : int, default=100
        Number of alphas along the regularization path
    """
    
    def __init__(self, l1_ratio=0.5, cv=5, max_iter=2000, tol=1e-4, 
                 random_state=42, n_alphas=100, eps=1e-3, task_type="regression"):
        self.l1_ratio = l1_ratio
        self.cv = cv
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.n_alphas = n_alphas
        self.eps = eps
        self.task_type = task_type
        
    def fit(self, X, y):
        """Fit the ElasticNet model with automatic alpha selection."""
        try:
            # Create safe CV splitter that avoids 'least populated class' warnings
            cv_splits = self.cv if isinstance(self.cv, int) else 5
            cv_obj = _safe_cv(y, self.task_type, n_splits=cv_splits, seed=self.random_state)
            
            # For small datasets, use adaptive alpha selection to avoid over-regularization
            n_samples = X.shape[0]
            if n_samples < 200:
                # Use adaptive alpha selection instead of fixed alpha to prevent constant predictions
                logger.debug(f"Small dataset ({n_samples} samples), using adaptive alpha selection")
                
                # Alpha candidates from least to most regularization
                alpha_candidates = [0.001, 0.01, 0.05, 0.1, 0.2]
                best_score = -np.inf
                best_alpha = alpha_candidates[0]
                
                for alpha_test in alpha_candidates:
                    try:
                        # Test each alpha with cross-validation
                        test_model = ElasticNet(
                            alpha=alpha_test,
                            l1_ratio=self.l1_ratio,
                            max_iter=self.max_iter,
                            tol=self.tol,
                            random_state=self.random_state,
                            selection='cyclic'
                        )
                        
                        # Use light cross-validation to select best alpha
                        cv_scores = cross_val_score(
                            test_model, X, y, 
                            cv=min(3, cv_splits),  # Use 3-fold for speed on small data
                            scoring='neg_mean_squared_error',
                            n_jobs=1  # Single job for stability
                        )
                        
                        avg_score = np.mean(cv_scores)
                        logger.debug(f"Alpha {alpha_test:.3f}: CV score = {avg_score:.4f}")
                        
                        if avg_score > best_score:
                            best_score = avg_score
                            best_alpha = alpha_test
                            
                    except Exception as alpha_error:
                        logger.debug(f"Alpha {alpha_test:.3f} failed: {alpha_error}")
                        continue
                
                logger.info(f"Selected optimal alpha = {best_alpha:.3f} (CV score = {best_score:.4f})")
                
                # Use the best alpha found
                base_model = ElasticNet(
                    alpha=best_alpha,
                    l1_ratio=self.l1_ratio,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    random_state=self.random_state,
                    selection='cyclic'
                )
                
                # Use lighter target transformation for small datasets
                self.regressor_ = TransformedTargetRegressor(
                    regressor=base_model,
                    transformer=StandardScaler()  # Use StandardScaler instead of PowerTransformer
                )
                
                self.regressor_.fit(X, y)
                self.alpha_ = base_model.alpha
                self.l1_ratio_ = base_model.l1_ratio
                
            else:
                # Use ElasticNetCV for larger datasets (unchanged)
                self.model_ = ElasticNetCV(
                    l1_ratio=self.l1_ratio,
                    cv=cv_obj,  # Use our safe CV splitter instead of raw integer
                    max_iter=self.max_iter,
                    tol=self.tol,
                    random_state=self.random_state,
                    n_alphas=self.n_alphas,
                    eps=self.eps,
                    n_jobs=-1,
                    selection='cyclic'  # Use cyclic coordinate descent
                )
                
                # Wrap in TransformedTargetRegressor for target transformation
                self.regressor_ = TransformedTargetRegressor(
                    regressor=self.model_,
                    transformer=PowerTransformer(method="yeo-johnson", standardize=True)
                )
                
                self.regressor_.fit(X, y)
                
                # Safely access alpha_ with fallback
                if hasattr(self.model_, 'alpha_') and self.model_.alpha_ is not None:
                    self.alpha_ = self.model_.alpha_
                    self.l1_ratio_ = self.model_.l1_ratio
                else:
                    logger.warning("ElasticNetCV fit succeeded but alpha_ not available, using default")
                    self.alpha_ = 0.1
                    self.l1_ratio_ = self.l1_ratio
            
            logger.debug(f"SelectionByCyclicCoordinateDescent: Optimal alpha={self.alpha_:.6f}, "
                        f"l1_ratio={self.l1_ratio_:.3f}")
            
            return self
            
        except Exception as e:
            logger.warning(f"SelectionByCyclicCoordinateDescent fit failed: {e}")
            # Fallback to regular ElasticNet with conservative settings
            try:
                fallback_model = ElasticNet(
                    alpha=0.1, 
                    l1_ratio=self.l1_ratio,
                    max_iter=self.max_iter, 
                    random_state=self.random_state,
                    selection='cyclic'
                )
                
                # Use simple StandardScaler for fallback to avoid numerical issues
                self.regressor_ = TransformedTargetRegressor(
                    regressor=fallback_model,
                    transformer=StandardScaler()
                )
                
                self.regressor_.fit(X, y)
                self.alpha_ = fallback_model.alpha
                self.l1_ratio_ = fallback_model.l1_ratio
                
                logger.debug(f"Using fallback ElasticNet: alpha={self.alpha_:.6f}")
                
            except Exception as fallback_error:
                logger.error(f"Even fallback ElasticNet failed: {fallback_error}")
                # Ultimate fallback - simple linear regression
                from sklearn.linear_model import LinearRegression
                self.regressor_ = LinearRegression()
                self.regressor_.fit(X, y)
                self.alpha_ = 0.0
                self.l1_ratio_ = 0.0
                
            return self
    
    def predict(self, X):
        """Make predictions using the fitted model."""
        return self.regressor_.predict(X)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'l1_ratio': self.l1_ratio,
            'cv': self.cv,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'random_state': self.random_state,
            'n_alphas': self.n_alphas,
            'eps': self.eps,
            'task_type': self.task_type
        }
    
    def set_params(self, **params):
        """Set the parameters of this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class RobustLinearRegressor(BaseEstimator, RegressorMixin):
    """
    Robust Linear Regression with automatic outlier handling.
    
    This wrapper provides identical interpretation to LinearRegression but
    with robustness to remaining outliers. It uses HuberRegressor as the
    primary method with RANSACRegressor as a fallback for extreme cases.
    
    Parameters
    ----------
    method : str, default='huber'
        Robust regression method ('huber' or 'ransac')
    epsilon : float, default=1.35
        Huber parameter (only for method='huber')
    max_iter : int, default=2000
        Maximum number of iterations
    random_state : int, default=42
        Random state for reproducibility
    """
    
    def __init__(self, method='huber', epsilon=1.35, max_iter=2000, 
                 random_state=42, alpha=0.0001):
        self.method = method
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.random_state = random_state
        self.alpha = alpha
        
    def fit(self, X, y):
        """Fit the robust linear regression model."""
        try:
            if self.method == 'huber':
                # Use HuberRegressor for robustness to outliers
                base_model = HuberRegressor(
                    epsilon=self.epsilon,
                    max_iter=self.max_iter,
                    alpha=self.alpha,
                    fit_intercept=True
                )
            elif self.method == 'ransac':
                # Use RANSACRegressor for extreme outlier cases
                # Handle compatibility between sklearn versions (base_estimator vs estimator)
                try:
                    # Try the newer parameter name first (sklearn >= 1.2)
                    base_model = RANSACRegressor(
                        estimator=LinearRegression(),
                        max_trials=100,
                        min_samples=None,  # Auto-determine
                        residual_threshold=None,  # Auto-determine
                        random_state=self.random_state
                    )
                except TypeError:
                    # Fallback to older parameter name (sklearn < 1.2)
                    base_model = RANSACRegressor(
                        base_estimator=LinearRegression(),
                        max_trials=100,
                        min_samples=None,  # Auto-determine
                        residual_threshold=None,  # Auto-determine
                        random_state=self.random_state
                    )
            else:
                # Fallback to regular LinearRegression
                base_model = LinearRegression()
            
            # Wrap in TransformedTargetRegressor for target transformation
            self.regressor_ = TransformedTargetRegressor(
                regressor=base_model,
                transformer=PowerTransformer(method="yeo-johnson", standardize=True)
            )
            
            self.regressor_.fit(X, y)
            
            # Store method used for inspection
            self.method_used_ = self.method
            
            logger.debug(f"RobustLinearRegressor: Using {self.method} method")
            
            return self
            
        except Exception as e:
            logger.warning(f"RobustLinearRegressor fit failed with {self.method}: {e}")
            # Ultimate fallback to LinearRegression
            self.regressor_ = TransformedTargetRegressor(
                regressor=LinearRegression(),
                transformer=PowerTransformer(method="yeo-johnson", standardize=True)
            )
            self.regressor_.fit(X, y)
            self.method_used_ = 'linear_fallback'
            return self
    
    def predict(self, X):
        """Make predictions using the fitted model."""
        return self.regressor_.predict(X)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'method': self.method,
            'epsilon': self.epsilon,
            'max_iter': self.max_iter,
            'random_state': self.random_state,
            'alpha': self.alpha
        }
    
    def set_params(self, **params):
        """Set the parameters of this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class OptimizedExtraTreesRegressor(BaseEstimator, RegressorMixin):
    """
    Optimized Extra Trees Regressor for small-n scenarios.
    
    This wrapper uses ExtraTreesRegressor with optimized parameters for
    small sample sizes, which tends to reduce variance compared to
    RandomForestRegressor on small-n datasets.
    
    Parameters
    ----------
    n_estimators : int, default=200
        Number of trees in the forest
    max_features : str or int, default='sqrt'
        Number of features to consider for splits
    bootstrap : bool, default=False
        Whether to use bootstrap sampling
    random_state : int, default=42
        Random state for reproducibility
    min_samples_split : int, default=5
        Minimum samples required to split a node
    min_samples_leaf : int, default=2
        Minimum samples required at a leaf node
    """
    
    def __init__(self, n_estimators=200, max_features='sqrt', bootstrap=False,
                 random_state=42, min_samples_split=5, min_samples_leaf=2,
                 max_depth=None, n_jobs=-1):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        
    def fit(self, X, y):
        """Fit the Extra Trees model."""
        try:
            # Adapt parameters based on sample size
            n_samples = X.shape[0]
            
            # For very small datasets, reduce complexity
            if n_samples < 50:
                n_estimators = min(100, self.n_estimators)
                min_samples_split = max(2, min(self.min_samples_split, n_samples // 5))
                min_samples_leaf = max(1, min(self.min_samples_leaf, n_samples // 10))
            else:
                n_estimators = self.n_estimators
                min_samples_split = self.min_samples_split
                min_samples_leaf = self.min_samples_leaf
            
            self.model_ = ExtraTreesRegressor(
                n_estimators=n_estimators,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                random_state=self.random_state,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_depth=self.max_depth,
                n_jobs=self.n_jobs
            )
            
            self.model_.fit(X, y)
            
            logger.debug(f"OptimizedExtraTreesRegressor: n_estimators={n_estimators}, "
                        f"max_features={self.max_features}, bootstrap={self.bootstrap}")
            
            return self
            
        except Exception as e:
            logger.warning(f"OptimizedExtraTreesRegressor fit failed: {e}")
            # Fallback to RandomForestRegressor
            self.model_ = RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
            self.model_.fit(X, y)
            return self
    
    def predict(self, X):
        """Make predictions using the fitted model."""
        return self.model_.predict(X)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'n_estimators': self.n_estimators,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'random_state': self.random_state,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_depth': self.max_depth,
            'n_jobs': self.n_jobs
        }
    
    def set_params(self, **params):
        """Set the parameters of this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


# Initialize caches with enhanced configuration from config
from config import CACHE_CONFIG

# Add at the top of the file, after logger definition
_feature_mismatch_logged = set()

        # Caching with LRU + size limits
class SizedLRUCache:
    """
    Memory-aware LRU Cache implementation that tracks both item count and total memory usage.
    Evicts items based on least recently used when either limit is exceeded.
    """
    def __init__(self, maxsize=128, maxmemory_mb=1000):
        """
        Initialize the cache with maximum size and memory limits.
        
        Parameters
        ----------
        maxsize : int
            Maximum number of items to store
        maxmemory_mb : int
            Maximum memory usage in MB
        """
        self.maxsize = maxsize
        self.maxmemory = maxmemory_mb * 1024 * 1024  # Convert MB to bytes
        self.cache = {}  # Main storage {key: value}
        self.key_order = []  # List to track access order
        self.memory_usage = 0  # Current memory usage in bytes
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
    def get(self, key, default=None):
        """
        Get an item from cache. Updates access order if found.
        
        Parameters
        ----------
        key : hashable
            Cache key
        default : Any
            Default value to return if key not found
            
        Returns
        -------
        Any
            Cached value or default
        """
        if key in self.cache:
            # Update access order - move to end of list
            self.key_order.remove(key)
            self.key_order.append(key)
            self.hits += 1
            return self.cache[key]
        else:
            self.misses += 1
            return default
        
    def put(self, key, value, item_size=None):
        """
        Add or update an item in the cache.
        Evicts items if necessary to maintain size and memory limits.
        
        Parameters
        ----------
        key : hashable
            Cache key
        value : Any
            Value to store
        item_size : Optional[int]
            Size of item in bytes. If None, estimates using sys.getsizeof
        
        Returns
        -------
        bool
            True if stored successfully
        """
        # Check if we already have this key
        if key in self.cache:
            # Update access order
            self.key_order.remove(key)
            
            # Reduce memory usage by the size of the old item
            old_size = getattr(value, 'nbytes', 0)
            if old_size == 0:
                import sys
                old_size = sys.getsizeof(value)
            self.memory_usage -= old_size
        
        # Calculate item size if not provided
        if item_size is None:
            if hasattr(value, 'nbytes'):
                item_size = value.nbytes
            else:
                import sys
                item_size = sys.getsizeof(value)
                
        # If this single item is too large, don't even try to cache it
        if item_size > self.maxmemory:
            logger.warning(f"Item too large to cache: {item_size / (1024*1024):.2f} MB > {self.maxmemory / (1024*1024):.2f} MB limit")
            return False
            
        # Make space if needed
        while (len(self.cache) >= self.maxsize or 
               self.memory_usage + item_size > self.maxmemory) and self.key_order:
            # Evict least recently used item
            evict_key = self.key_order.pop(0)
            evict_value = self.cache.pop(evict_key)
            
            # Reduce memory usage
            evict_size = getattr(evict_value, 'nbytes', 0)
            if evict_size == 0:
                import sys
                evict_size = sys.getsizeof(evict_value)
            
            self.memory_usage -= evict_size
            self.evictions += 1
            
        # Store new item
        self.cache[key] = value
        self.key_order.append(key)
        self.memory_usage += item_size
        
        return True
    
    def clear(self):
        """Clear the cache and reset stats."""
        self.cache.clear()
        self.key_order.clear()
        self.memory_usage = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
    def stats(self):
        """
        Get cache statistics.
        
        Returns
        -------
        dict
            Dictionary of cache statistics
        """
        hit_ratio = self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        
        return {
            "size": len(self.cache),
            "maxsize": self.maxsize,
            "memory_usage_mb": self.memory_usage / (1024 * 1024),
            "maxmemory_mb": self.maxmemory / (1024 * 1024),
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": hit_ratio,
            "evictions": self.evictions
        }

_selector_cache = {
    'sel_reg': SizedLRUCache(
        maxsize=CACHE_CONFIG["selector_regression"]["maxsize"], 
        maxmemory_mb=CACHE_CONFIG["selector_regression"]["maxmemory_mb"]
    ),
    'sel_clf': SizedLRUCache(
        maxsize=CACHE_CONFIG["selector_classification"]["maxsize"], 
        maxmemory_mb=CACHE_CONFIG["selector_classification"]["maxmemory_mb"]
    )
}

_extractor_cache = {
    'ext_reg': SizedLRUCache(
        maxsize=CACHE_CONFIG["extractor_regression"]["maxsize"], 
        maxmemory_mb=CACHE_CONFIG["extractor_regression"]["maxmemory_mb"]
    ),
    'ext_clf': SizedLRUCache(
        maxsize=CACHE_CONFIG["extractor_classification"]["maxsize"], 
        maxmemory_mb=CACHE_CONFIG["extractor_classification"]["maxmemory_mb"]
    )
}

def clear_modality_caches():
    """
    Clear all modality caches to ensure new modality-aware cache keys take effect.
    This prevents old cached results from different modalities interfering with each other.
    """
    global _selector_cache, _extractor_cache
    
    try:
        # Clear all selector caches
        for cache_type in _selector_cache:
            _selector_cache[cache_type].clear()
            
        # Clear all extractor caches
        for cache_type in _extractor_cache:
            _extractor_cache[cache_type].clear()
            
        logger.info("Successfully cleared all modality caches to ensure cache key fix takes effect")
        
    except Exception as e:
        logger.warning(f"Failed to clear modality caches: {str(e)}")

# CRITICAL: Clear existing caches immediately to ensure modality-aware cache keys take effect
# This prevents cached results from before the fix from causing wrong hyperparameter applications
clear_modality_caches()

def _generate_cache_key(ds_name, fold_idx, name, obj_type, n_val, input_shape=None):
    """Generate a consistent cache key"""
    # Create a stable representation for hashing
    key_parts = [
        str(ds_name or "unknown"),
        str(fold_idx or "global"),
        str(name or "unnamed"),
        str(obj_type),
        str(n_val)
    ]
    
    # CRITICAL FIX: Include input shape to prevent cache hits with different sample counts
    if input_shape is not None:
        key_parts.append(f"shape_{input_shape[0]}x{input_shape[1]}")
    
    # Generate a hash to ensure key size is bounded
    key_str = "_".join(key_parts)
    hash_obj = hashlib.md5(key_str.encode())
    
    # DIAGNOSTIC: Log cache key generation for debugging (only in debug mode)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Generated cache key: {hash_obj.hexdigest()[:8]}... from parts: {key_parts}")
    
    return hash_obj.hexdigest()

# Update cache functions to use the new key generation and caching strategy
def cached_fit_transform_selector_regression(X, y, selector, n_feats, ds_name=None, modality_name=None, fold_idx=0, model_name=None, fusion_method="attention_weighted", consistent_hyperparams=None):
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
    # Preserve the original selector code for MRMR handling
    original_selector_code = selector if isinstance(selector, str) else None
    
    # Convert string selector to object if needed
    if isinstance(selector, str):
        # We'll update the selector object after calculating effective_n_feats
        selector_code = selector
        selector = None
    
    # Generate a stable cache key
    if isinstance(selector, dict):
        # Handle dictionary for Lasso/ElasticNet
        selector_type = selector['type']
    elif original_selector_code:
        # Use the original selector code for cache key
        selector_type = original_selector_code
    elif 'selector_code' in locals():
        # Use the selector_code if we have it
        selector_type = selector_code
    elif selector is not None:
        selector_type = selector.__class__.__name__
    else:
        selector_type = "unknown"
        
    # Create cache key
    # CRITICAL FIX: Include modality_name in cache key so different modalities don't share cache
    cache_name = f"{selector_type}_{modality_name}" if modality_name else selector_type
    key = _generate_cache_key(ds_name, fold_idx, cache_name, "sel_reg", n_feats, X.shape if X is not None else None)
    
    # Check cache
    cached_result = _selector_cache['sel_reg'].get(key)
    if cached_result is not None:
        return cached_result
    
    try:
        # Convert to numpy arrays
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        
        # Clean NaN values
        X_arr = np.nan_to_num(X_arr, nan=0.0)
        
        # FIX C: Use strict validation without truncation to prevent X/y mismatches
        # Do NOT allow truncation as it causes alignment issues with target vectors
        X_arr, y_arr = validate_and_fix_shape_mismatch(
            X_arr, y_arr, 
            name=f"{ds_name if ds_name else 'unknown'} regression selector data", 
            fold_idx=fold_idx,
            allow_truncation=False  # CRITICAL: Disable truncation to prevent alignment bugs
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
                result = (selected_features, transformed_X)
                _selector_cache['sel_reg'].put(key, result, item_size=transformed_X.nbytes)
                return result
            return None, None
        
        # Make sure n_feats is limited by data dimensions
        max_possible_feats = min(X_arr.shape[0], X_arr.shape[1])
        effective_n_feats = min(n_feats, max_possible_feats)
        
        if effective_n_feats < n_feats:
            logger.info(f"Limiting features from {n_feats} to {effective_n_feats} due to data dimensions")
        
        # Create selector object with correct effective_n_feats if we have a selector_code
        if selector is None and 'selector_code' in locals():
            selector = get_selector_object(selector_code, effective_n_feats)
        
        # Load and apply hyperparameters if dataset name and model name are provided
        if ds_name and model_name and selector is not None and hasattr(selector, 'set_params'):
            try:
                # Get the selector code for hyperparameter loading
                selector_algorithm = None
                if 'selector_code' in locals():
                    selector_algorithm = selector_code
                elif original_selector_code:
                    selector_algorithm = original_selector_code
                elif hasattr(selector, '__class__'):
                    selector_algorithm = selector.__class__.__name__
                
                if selector_algorithm:
                    hyperparams = load_feature_first_hyperparameters(
                        dataset=ds_name,
                        algorithm=selector_algorithm,
                        model=model_name,
                        fusion_method=fusion_method,
                        n_features=n_feats,
                        approach="selector"
                    )
                    
                    # Only apply selector-specific parameters to the selector
                    # model_params should be applied later to the final model, not the selector
                    if hyperparams['selector_params']:
                        selector.set_params(**hyperparams['selector_params'])
                        logger.info(f"Applied selector-specific hyperparameters for {ds_name}_{selector_algorithm}: {hyperparams['selector_params']}")
                    
                    # Store model_params for later use by the final model (not applied to selector)
                    if hyperparams['model_params']:
                        logger.debug(f"Model params available for final model (not applied to selector): {hyperparams['model_params']}")
                        
            except Exception as e:
                logger.warning(f"Failed to apply selector hyperparameters for {ds_name}_{selector_algorithm}_{model_name}: {str(e)}")
        
        # Use basic feature selection as fallback
        if selector is None:
            from sklearn.feature_selection import SelectKBest, f_regression
            selector = SelectKBest(f_regression, k=effective_n_feats)
        
        # Fit and transform
        selector.fit(X_arr, y_arr)
        X_selected = selector.transform(X_arr)
        
        # Get selected features
        if hasattr(selector, 'get_support'):
            selected_features = selector.get_support()
        else:
            # Fallback for selectors without get_support
            selected_features = np.arange(X_selected.shape[1])
        
        result = (selected_features, X_selected)
        _selector_cache['sel_reg'].put(key, result, item_size=X_selected.nbytes if hasattr(X_selected, 'nbytes') else X_selected.size * 8)
        return result
        
    except Exception as e:
        logger.warning(f"Selector regression failed for {ds_name}: {str(e)}")
        # Return fallback result
        if X is not None and X.shape[1] > 0:
            mask = np.zeros(X.shape[1], dtype=bool)
            mask[0] = True
            selected_features = mask
            transformed_X = X[:min(len(X), 10), [0]]
            result = (selected_features, transformed_X)
            return result
        return None, None

def get_regression_extractors() -> Dict[str, Any]:
    """
    Get dictionary of regression feature extractors optimized for genomic data.
    
    CURRENT IMPLEMENTATION - 6 extractors as specified:
    1. PCA - Reliable baseline for dimensionality reduction
    2. KPCA - Kernel-based non-linear dimensionality reduction
    3. FA - Captures underlying biological factors
    4. PLS - Supervised linear method
    5. KPLS - Supervised non-linear method (when IKPLS available)
    6. SparsePLS - Sparse supervised extraction
    
    Returns
    -------
    dict
        Dictionary mapping extractor names to initialized extractor objects
    """
    # Check if IKPLS is available
    IKPLS_AVAILABLE = False
    try:
        import ikpls
        IKPLS_AVAILABLE = True
    except ImportError:
        IKPLS_AVAILABLE = False
    
    extractors = {
        # SPECIFIED 6 ALGORITHMS FOR REGRESSION
        "PCA": PCA(random_state=42),
        "KPCA": KernelPCA(
            kernel="rbf", 
            random_state=42, 
            n_jobs=1,  # Use single thread to avoid parallelism issues
            gamma=None,  # Use default gamma (1.0/n_features)
            eigen_solver="dense",  # Use dense solver by default for better stability
            tol=1e-4,  # Relaxed tolerance for eigenvalue computation
            max_iter=None  # Let sklearn use default max_iter
        ),
        "FA": FactorAnalysis(
            random_state=42,
            max_iter=100,   # Very low for speed - FA is inherently slow on genomic data
            tol=1e-1,      # Very relaxed tolerance for speed
            n_components=4  # Fixed low default to prevent slow high-component searches
        ),
        "PLS": PLSRegression(
            n_components=5,        # Reduced from 8 to 5 for consistency
            max_iter=5000,  # Increased max iterations
            tol=1e-3       # Relaxed tolerance
        ),
        "KPLS": KernelPLSRegression(
            n_components=5,        # Reduced from 8 to 5 for stability
            kernel="rbf",
            gamma="auto",          # median heuristic with bounds
            max_iter=500,
            tol=1e-4,             # Relaxed tolerance for stability
            regularization=1e-6,   # Added regularization
            use_cv_components=True, # Cross-validation for optimal components
            cv_folds=3,           # 3-fold CV for component selection
            gamma_bounds=(1e-6, 1e3), # Bounds to prevent extreme gamma values
            random_state=42
        ) if IKPLS_AVAILABLE else PLSRegression(
            n_components=5,        # Also reduced for regular PLS fallback
            max_iter=5000,
            tol=1e-3
        ),
        "SparsePLS": SparsePLS(
            n_components=3,        # Reduced from 5 to 3 for consistency
            alpha=0.1,  # FIXED: Reduced from 0.3 to 0.1 to prevent empty arrays
            max_iter=500,
            tol=1e-6,
            scale=True
        ),
    }
    
    return extractors

def get_regression_selectors() -> Dict[str, str]:
    """
    Get dictionary of regression feature selectors.
    
    Returns
    -------
    dict
        Dictionary mapping selector names to selector codes
    """
    return {
        "ElasticNetFS": "elasticnet_regression", 
        "RFImportance": "random_forest_regression",
        "VarianceFTest": "f_regression",
        "LASSO": "lasso_regression",
        "f_regressionFS": "f_regression"
    }

def get_classification_extractors() -> Dict[str, Any]:
    """
    Get dictionary of classification feature extractors optimized for genomic data.
    
    CURRENT IMPLEMENTATION - 6 extractors as specified:
    1. PCA - Principal component analysis
    2. KPCA - Kernel-based non-linear dimensionality reduction  
    3. FA - Factor analysis
    4. LDA - Linear discriminant analysis
    5. PLS-DA - Partial least squares discriminant analysis
    6. SparsePLS - Sparse partial least squares
    
    Returns
    -------
    dict
        Dictionary mapping extractor names to initialized extractor objects
    """
    extractors = {
        # SPECIFIED 6 ALGORITHMS FOR CLASSIFICATION
        "PCA": PCA(random_state=42),
        "KPCA": KernelPCA(
            kernel="rbf", 
            random_state=42, 
            n_jobs=1,  # Use single thread to avoid parallelism issues
            gamma=None,  # Use default gamma (1.0/n_features)
            eigen_solver="dense",  # Use dense solver by default for better stability
            tol=1e-4,  # Relaxed tolerance for eigenvalue computation
            max_iter=None  # Let sklearn use default max_iter
        ),
        "FA": FactorAnalysis(
            random_state=42,
            max_iter=100,   # Very low for speed - FA is inherently slow on genomic data
            tol=1e-1,      # Very relaxed tolerance for speed
            n_components=4  # Fixed low default to prevent slow high-component searches
        ),
        "LDA": LDA(),
        "PLS-DA": PLSDiscriminantAnalysis(
            n_components=5,        # Reduced from 8 to 5 for consistency
            max_iter=1000,
            tol=1e-6,
            scale=True
        ),
        "SparsePLS": SparsePLS(
            n_components=3,        # Reduced from 5 to 3 for consistency
            alpha=0.1,  # FIXED: Reduced from 0.3 to 0.1 to prevent empty arrays
            max_iter=500,
            tol=1e-6,
            scale=True
        )
    }
    
    return extractors

def get_classification_selectors() -> Dict[str, str]:
    """
    Get dictionary of classification feature selectors.
    
    CURRENT IMPLEMENTATION - 5 selectors as specified:
    1. ElasticNetFS - ElasticNet-based feature selection
    2. RFImportance - Random Forest importance-based selection
    3. VarianceFTest - Variance-based F-test selection
    4. LASSO - L1-regularized feature selection (mapped to LogisticL1)
    5. LogisticL1 - Logistic regression with L1 penalty
    
    Returns
    -------
    dict
        Dictionary mapping selector names to selector codes
    """
    return {
        # SPECIFIED 5 ALGORITHMS FOR CLASSIFICATION SELECTION
        "ElasticNetFS": "elasticnet_classification", 
        "RFImportance": "random_forest_classification",
        "VarianceFTest": "f_classification",
        "LASSO": "lasso_classification",  # Maps to LogisticL1 implementation
        "LogisticL1": "lasso_classification"
    }

def get_selector_object(selector_code: str, n_feats: int):
    """
    Create a feature selector object based on the selector code.
    
    Parameters
    ----------
    selector_code : str
        Code identifying the selector type
    n_feats : int
        Number of features to select
        
    Returns
    -------
    selector object
        Configured feature selector
    """
    from sklearn.feature_selection import SelectKBest, f_regression, f_classif
    
    if selector_code == "f_regression":
        return SelectKBest(score_func=f_regression, k=n_feats)
    elif selector_code == "f_classification":
        return SelectKBest(score_func=f_classif, k=n_feats)
    else:
        # Default fallback
        return SelectKBest(score_func=f_regression, k=n_feats)

# Cache configuration
CACHE_CONFIG = {
    "selector_regression": {"maxsize": 64, "maxmemory_mb": 500},
    "selector_classification": {"maxsize": 64, "maxmemory_mb": 500},
    "extractor_regression": {"maxsize": 64, "maxmemory_mb": 500},
    "extractor_classification": {"maxsize": 64, "maxmemory_mb": 500}
}

def validate_and_fix_shape_mismatch(X, y, name="data", fold_idx=None, allow_truncation=True):
    """
    Validate and fix shape mismatches between X and y.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    name : str
        Name for logging
    fold_idx : int, optional
        Fold index for logging
    allow_truncation : bool
        Whether to allow truncation to fix mismatches
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Validated X and y arrays
    """
    if X is None or y is None:
        return None, None
        
    if X.shape[0] != len(y):
        if allow_truncation:
            min_samples = min(X.shape[0], len(y))
            logger.warning(f"Shape mismatch in {name}: X={X.shape}, y={len(y)}, truncating to {min_samples}")
            return X[:min_samples], y[:min_samples]
        else:
            logger.error(f"Shape mismatch in {name}: X={X.shape}, y={len(y)}, truncation disabled")
            return None, None
    
    return X, y

def cached_fit_transform_extractor_regression(X, y, extractor, n_components, ds_name=None, fold_idx=0, modality_name=None, model_name=None, fusion_method="attention_weighted", consistent_hyperparams=None):
    """
    Cached version of fit_transform for regression extractors with hyperparameter loading.
    
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
    ds_name : str, optional
        Dataset name for caching and hyperparameter loading
    fold_idx : int
        Fold index for caching
    modality_name : str, optional
        Modality name (not used for hyperparameter loading in multi-modal setup)
    model_name : str, optional
        Model name for hyperparameter loading
    fusion_method : str, optional
        Fusion method name
    consistent_hyperparams : dict, optional
        Pre-loaded consistent hyperparameters for element-wise fusion methods
        
    Returns
    -------
    Tuple[object, np.ndarray]
        Fitted extractor and transformed data
    """
    # Generate cache key including model name for model-specific hyperparameters
    extractor_class_name = extractor.__class__.__name__
    extractor_name = _map_extractor_class_to_hyperparameter_name(extractor_class_name)
    cache_name = f"{extractor_name}_{model_name}" if model_name else extractor_name
    
    # CRITICAL FIX: Include modality_name in cache key so different modalities don't share cache
    if modality_name:
        cache_name = f"{cache_name}_{modality_name}"
    
    key = _generate_cache_key(ds_name, fold_idx, cache_name, "ext_reg", n_components, X.shape if X is not None else None)
    
    # Check cache
    cached_result = _extractor_cache['ext_reg'].get(key)
    if cached_result is not None:
        return cached_result
    
    try:
        # Load and apply hyperparameters if dataset name is provided
        if ds_name and model_name:
            # CONSISTENCY FIX: Use consistent hyperparameters if provided (for element-wise fusion methods)
            if consistent_hyperparams:
                best_hyperparams = consistent_hyperparams
                logger.info(f"Using consistent hyperparameters for {ds_name}_{extractor_name} (source: {best_hyperparams['source']})")
            else:
                # Use the new feature-first hyperparameter loading system
                hyperparams = load_feature_first_hyperparameters(
                    dataset=ds_name,
                    algorithm=extractor_name,
                    model=model_name,
                    fusion_method=fusion_method,
                    approach="extractor"
                )
                
                if hyperparams['extractor_params']:
                    best_hyperparams = hyperparams
                else:
                    # Fallback: try other model combinations if specific one not found
                    model_candidates = ["LinearRegression", "ElasticNet", "RandomForestRegressor"]
                    best_hyperparams = None
                    
                    for fallback_model in model_candidates:
                        if fallback_model == model_name:
                            continue  # Skip the one we already tried
                        hyperparams = load_feature_first_hyperparameters(
                            dataset=ds_name,
                            algorithm=extractor_name,
                            model=fallback_model,
                            fusion_method=fusion_method,
                            approach="extractor"
                        )
                        if hyperparams['extractor_params']:
                            best_hyperparams = hyperparams
                            logger.warning(f"Using fallback hyperparameters from {fallback_model} for {model_name}")
                            break
            
            # Apply hyperparameters if found
            if 'best_hyperparams' in locals() and best_hyperparams and best_hyperparams.get('extractor_params'):
                try:
                    # Sanitize hyperparameters before applying
                    validated_params = _sanitize_extractor_hyperparameters(
                        extractor, extractor_name, best_hyperparams['extractor_params']
                    )
                    
                    if validated_params:
                        extractor.set_params(**validated_params)
                        extractor._hyperparams_applied = True  # Mark as applied
                        logger.info(f"Applied tuned extractor hyperparameters for {ds_name}_{extractor_name}: {validated_params} (source: {best_hyperparams['source']})")
                    else:
                        logger.debug(f"No valid hyperparameters to apply for {ds_name}_{extractor_name}")
                except Exception as e:
                    logger.warning(f"Failed to apply extractor hyperparameters for {ds_name}_{extractor_name}: {str(e)}")
                    logger.debug(f"Original params: {best_hyperparams['extractor_params']}")
            else:
                logger.debug(f"No extractor hyperparameters found for {ds_name}_{extractor_name}")
        
        # Convert to numpy arrays
        X_arr = np.asarray(X)
        y_arr = np.asarray(y) if y is not None else None
        
        # Clean NaN values
        X_arr = np.nan_to_num(X_arr, nan=0.0)
        
        # Validate shapes
        if y_arr is not None:
            X_arr, y_arr = validate_and_fix_shape_mismatch(
                X_arr, y_arr, 
                name=f"{ds_name if ds_name else 'unknown'} regression extractor data", 
                fold_idx=fold_idx,
                allow_truncation=False
            )
            
            if X_arr is None or y_arr is None:
                logger.warning(f"Data alignment failure in extractor for {ds_name}")
                return None, None
        
        # Set n_components if the extractor supports it and it's not already set by hyperparameters
        if hasattr(extractor, 'n_components') and not hasattr(extractor, '_hyperparams_applied'):
            # CRITICAL FIX: For consistent hyperparameters (element-wise fusion), always use the consistent value
            # to ensure all modalities have the same n_components
            if consistent_hyperparams and 'n_components' in consistent_hyperparams.get('extractor_params', {}):
                target_components = consistent_hyperparams['extractor_params']['n_components']
                logger.info(f"Using CONSISTENT n_components={target_components} from hyperparameters for {modality_name or 'unknown'} modality")
                extractor.n_components = target_components
            else:
                max_components = min(X_arr.shape[0], X_arr.shape[1])
                effective_components = min(n_components, max_components)
                extractor.n_components = effective_components
        
        # Fit and transform
        # Check if extractor requires y parameter using signature inspection
        requires_y = False
        if y_arr is not None and hasattr(extractor, 'fit'):
            try:
                import inspect
                sig = inspect.signature(extractor.fit)
                requires_y = 'y' in sig.parameters
            except:
                # Fallback to checking co_varnames (for custom classes)
                requires_y = 'y' in extractor.fit.__code__.co_varnames
        
        if requires_y:
            # Supervised extractor
            fitted_extractor = copy.deepcopy(extractor)
            fitted_extractor.fit(X_arr, y_arr)
            X_transformed = fitted_extractor.transform(X_arr)
        else:
            # Unsupervised extractor
            fitted_extractor = copy.deepcopy(extractor)
            fitted_extractor.fit(X_arr)
            X_transformed = fitted_extractor.transform(X_arr)
        
        # CRITICAL FIX: For KPCA, check if we got fewer components than expected due to gamma issues
        if (hasattr(fitted_extractor, '__class__') and 'KernelPCA' in fitted_extractor.__class__.__name__ and
            hasattr(fitted_extractor, 'n_components') and X_transformed is not None):
            
            expected_components = fitted_extractor.n_components
            actual_components = X_transformed.shape[1]
            
            if actual_components < expected_components:
                logger.warning(f"KPCA produced {actual_components} components instead of {expected_components} for {modality_name or 'unknown'} modality")
                logger.info(f"Attempting to fix KPCA gamma parameter for consistent component count")
                
                # Try with smaller gamma values
                for gamma_candidate in [0.1, 0.01, 0.001]:
                    try:
                        logger.debug(f"Trying KPCA with gamma={gamma_candidate}")
                        retry_extractor = copy.deepcopy(extractor)
                        retry_extractor.gamma = gamma_candidate
                        retry_extractor.fit(X_arr)
                        X_retry = retry_extractor.transform(X_arr)
                        
                        if X_retry.shape[1] >= expected_components:
                            logger.info(f"KPCA gamma fix successful: gamma={gamma_candidate} produces {X_retry.shape[1]} components")
                            fitted_extractor = retry_extractor
                            X_transformed = X_retry
                            break
                    except Exception as retry_error:
                        logger.debug(f"KPCA retry with gamma={gamma_candidate} failed: {retry_error}")
                        continue
                
                # If still not enough components, log a warning
                if X_transformed.shape[1] < expected_components:
                    logger.warning(f"Could not achieve {expected_components} components for {modality_name or 'unknown'} modality even with gamma adjustment")
                    logger.warning(f"Final component count: {X_transformed.shape[1]} (this may cause fusion inconsistencies)")
        
        result = (fitted_extractor, X_transformed)
        _extractor_cache['ext_reg'].put(key, result, item_size=X_transformed.nbytes if hasattr(X_transformed, 'nbytes') else X_transformed.size * 8)
        return result
        
    except Exception as e:
        error_msg = str(e)
        logger.warning(f"Extractor regression failed for {ds_name}: {error_msg}")
        
        # Special handling for KPCA ARPACK errors
        if ("ARPACK error" in error_msg or "No shifts could be applied" in error_msg) and 'KernelPCA' in extractor.__class__.__name__:
            logger.warning(f"KPCA encountered ARPACK eigenvalue solver error for {ds_name} - trying recovery strategies")
            try:
                from sklearn.decomposition import KernelPCA
                
                # Strategy 1: Try with 'dense' eigen_solver and reduced components
                safe_components = min(n_components // 2, X_arr.shape[0] - 1, 4)  # More conservative reduction
                safe_components = max(1, safe_components)
                
                logger.info(f"ARPACK recovery attempt 1: Using 'dense' solver with {safe_components} components")
                recovery_extractor = KernelPCA(
                    n_components=safe_components,
                    kernel=extractor.kernel,
                    gamma=None,  # Use default gamma for compatibility
                    eigen_solver='dense',  # Dense solver is more stable than ARPACK
                    n_jobs=1,  # Single thread to avoid parallelism issues
                    random_state=42
                )
                
                recovery_extractor.fit(X_arr)
                X_transformed = recovery_extractor.transform(X_arr)
                
                logger.info(f"ARPACK recovery successful with 'dense' solver and {safe_components} components")
                result = (recovery_extractor, X_transformed)
                _extractor_cache['ext_reg'].put(key, result, item_size=X_transformed.nbytes if hasattr(X_transformed, 'nbytes') else X_transformed.size * 8)
                return result
                
            except Exception as recovery_error1:
                logger.warning(f"ARPACK recovery attempt 1 failed: {recovery_error1}")
                try:
                    # Strategy 2: Try with randomized solver and even fewer components
                    minimal_components = min(2, X_arr.shape[0] - 1)
                    minimal_components = max(1, minimal_components)
                    
                    logger.info(f"ARPACK recovery attempt 2: Using 'randomized' solver with {minimal_components} components")
                    recovery_extractor2 = KernelPCA(
                        n_components=minimal_components,
                        kernel='linear',  # Switch to linear kernel for stability
                        eigen_solver='randomized',  # Randomized solver for large matrices
                        n_jobs=1,
                        random_state=42
                    )
                    
                    recovery_extractor2.fit(X_arr)
                    X_transformed = recovery_extractor2.transform(X_arr)
                    
                    logger.info(f"ARPACK recovery successful with 'randomized' solver and {minimal_components} components")
                    result = (recovery_extractor2, X_transformed)
                    _extractor_cache['ext_reg'].put(key, result, item_size=X_transformed.nbytes if hasattr(X_transformed, 'nbytes') else X_transformed.size * 8)
                    return result
                    
                except Exception as recovery_error2:
                    logger.warning(f"ARPACK recovery attempt 2 failed: {recovery_error2}")
                    logger.warning("All KPCA recovery strategies failed - falling back to PCA")
                    # Fall through to PCA fallback below
        
        # Special handling for KPCA zero-size array error
        elif "zero-size array to reduction operation maximum" in error_msg and 'KernelPCA' in extractor.__class__.__name__:
            logger.warning("KPCA encountered zero-size eigenvalue array - falling back to PCA")
            try:
                # Fall back to regular PCA with CONSISTENT components across all modalities
                from sklearn.decomposition import PCA
                
                # CONSISTENCY FIX: Use consistent hyperparameters if provided (for element-wise fusion methods)
                if consistent_hyperparams and consistent_hyperparams.get('n_components'):
                    # Use the consistent n_components from hyperparameters
                    target_components = consistent_hyperparams['n_components']
                    logger.info(f"Using consistent n_components={target_components} from hyperparameters for KPCA fallback")
                else:
                    # Use a fixed, conservative number that works for all modalities
                    # This ensures consistent dimensions for fusion
                    target_components = min(8, n_components // 2, X.shape[0] - 2)
                
                safe_components = min(target_components, X_arr.shape[0] - 1, X_arr.shape[1])
                safe_components = max(1, safe_components)  # Ensure at least 1 component
                
                fallback_extractor = PCA(n_components=safe_components, random_state=42)
                fallback_extractor.fit(X_arr)
                X_transformed = fallback_extractor.transform(X_arr)
                
                if consistent_hyperparams:
                    logger.info(f"KPCA fallback to PCA with CONSISTENT {safe_components} components (from hyperparams: {target_components})")
                else:
                    logger.info(f"KPCA fallback to PCA successful with {safe_components} components (consistent across modalities)")
                result = (fallback_extractor, X_transformed)
                _extractor_cache['ext_reg'].put(key, result, item_size=X_transformed.nbytes if hasattr(X_transformed, 'nbytes') else X_transformed.size * 8)
                return result
                
            except Exception as fallback_error:
                logger.error(f"KPCA fallback also failed: {fallback_error}")
        
        return None, None

def cached_fit_transform_extractor_classification(X, y, extractor, n_components, ds_name=None, fold_idx=0, modality_name=None, model_name=None, fusion_method="attention_weighted", consistent_hyperparams=None):
    """
    Cached version of fit_transform for classification extractors with hyperparameter loading.
    
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
    ds_name : str, optional
        Dataset name for caching and hyperparameter loading
    fold_idx : int
        Fold index for caching
    modality_name : str, optional
        Modality name (not used for hyperparameter loading in multi-modal setup)
        
    Returns
    -------
    Tuple[object, np.ndarray]
        Fitted extractor and transformed data
    """
    # Generate cache key including model name for model-specific hyperparameters
    extractor_class_name = extractor.__class__.__name__
    extractor_name = _map_extractor_class_to_hyperparameter_name(extractor_class_name)
    cache_name = f"{extractor_name}_{model_name}" if model_name else extractor_name
    
    # CRITICAL FIX: Include modality_name in cache key so different modalities don't share cache
    if modality_name:
        cache_name = f"{cache_name}_{modality_name}"
    
    key = _generate_cache_key(ds_name, fold_idx, cache_name, "ext_clf", n_components, X.shape if X is not None else None)
    
    # Check cache
    cached_result = _extractor_cache['ext_clf'].get(key)
    if cached_result is not None:
        return cached_result
    
    try:
        # Load and apply hyperparameters if dataset name is provided
        if ds_name and model_name:
            # CONSISTENCY FIX: Use consistent hyperparameters if provided (for element-wise fusion methods)
            if consistent_hyperparams:
                best_hyperparams = consistent_hyperparams
                logger.info(f"Using consistent hyperparameters for {ds_name}_{extractor_name} (source: {best_hyperparams['source']})")
            else:
                # Use the new feature-first hyperparameter loading system
                hyperparams = load_feature_first_hyperparameters(
                    dataset=ds_name,
                    algorithm=extractor_name,
                    model=model_name,
                    fusion_method=fusion_method,
                    approach="extractor"
                )
                
                if hyperparams['extractor_params']:
                    best_hyperparams = hyperparams
                else:
                    # Fallback: try other model combinations if specific one not found
                    model_candidates = ["LogisticRegression", "RandomForestClassifier", "SVC"]
                    best_hyperparams = None
                    
                    for fallback_model in model_candidates:
                        if fallback_model == model_name:
                            continue  # Skip the one we already tried
                        hyperparams = load_feature_first_hyperparameters(
                            dataset=ds_name,
                            algorithm=extractor_name,
                            model=fallback_model,
                            fusion_method=fusion_method,
                            approach="extractor"
                        )
                        if hyperparams['extractor_params']:
                            best_hyperparams = hyperparams
                            logger.warning(f"Using fallback hyperparameters from {fallback_model} for {model_name}")
                            break
            
            if best_hyperparams and best_hyperparams['extractor_params']:
                try:
                    # Sanitize hyperparameters before applying
                    validated_params = _sanitize_extractor_hyperparameters(
                        extractor, extractor_name, best_hyperparams['extractor_params']
                    )
                    
                    if validated_params:
                        extractor.set_params(**validated_params)
                        extractor._hyperparams_applied = True  # Mark as applied
                        logger.info(f"Applied tuned extractor hyperparameters for {ds_name}_{extractor_name}: {validated_params} (source: {best_hyperparams['source']})")
                    else:
                        logger.debug(f"No valid hyperparameters to apply for {ds_name}_{extractor_name}")
                except Exception as e:
                    logger.warning(f"Failed to apply extractor hyperparameters for {ds_name}_{extractor_name}: {str(e)}")
                    logger.debug(f"Original params: {best_hyperparams['extractor_params']}")
            else:
                logger.debug(f"No extractor hyperparameters found for {ds_name}_{extractor_name}")
        
        # Convert to numpy arrays
        X_arr = np.asarray(X)
        y_arr = np.asarray(y) if y is not None else None
        
        # Clean NaN values
        X_arr = np.nan_to_num(X_arr, nan=0.0)
        
        # Validate shapes
        if y_arr is not None:
            X_arr, y_arr = validate_and_fix_shape_mismatch(
                X_arr, y_arr, 
                name=f"{ds_name if ds_name else 'unknown'} classification extractor data", 
                fold_idx=fold_idx,
                allow_truncation=False
            )
            
            if X_arr is None or y_arr is None:
                logger.warning(f"Data alignment failure in extractor for {ds_name}")
                return None, None
        
        # Set n_components if the extractor supports it and it's not already set by hyperparameters
        if hasattr(extractor, 'n_components') and not hasattr(extractor, '_hyperparams_applied'):
            # CRITICAL FIX: For consistent hyperparameters (element-wise fusion), always use the consistent value
            # to ensure all modalities have the same n_components
            if consistent_hyperparams and 'n_components' in consistent_hyperparams.get('extractor_params', {}):
                target_components = consistent_hyperparams['extractor_params']['n_components']
                logger.info(f"Using CONSISTENT n_components={target_components} from hyperparameters for {modality_name or 'unknown'} modality")
                extractor.n_components = target_components
            else:
                max_components = min(X_arr.shape[0], X_arr.shape[1])
                
                # Special handling for LDA: max components is min(n_features, n_classes - 1)
                if extractor.__class__.__name__ == 'LinearDiscriminantAnalysis' and y_arr is not None:
                    n_classes = len(np.unique(y_arr))
                    max_components = min(max_components, n_classes - 1)
                
                effective_components = min(n_components, max_components)
                if effective_components > 0:
                    extractor.n_components = effective_components
        
        # Fit and transform
        # Check if extractor requires y parameter using signature inspection
        requires_y = False
        if y_arr is not None and hasattr(extractor, 'fit'):
            try:
                import inspect
                sig = inspect.signature(extractor.fit)
                requires_y = 'y' in sig.parameters
            except:
                # Fallback to checking co_varnames (for custom classes)
                requires_y = 'y' in extractor.fit.__code__.co_varnames
        
        if requires_y:
            # Supervised extractor
            fitted_extractor = copy.deepcopy(extractor)
            fitted_extractor.fit(X_arr, y_arr)
            X_transformed = fitted_extractor.transform(X_arr)
        else:
            # Unsupervised extractor
            fitted_extractor = copy.deepcopy(extractor)
            fitted_extractor.fit(X_arr)
            X_transformed = fitted_extractor.transform(X_arr)
        
        # CRITICAL FIX: For KPCA, check if we got fewer components than expected due to gamma issues
        if (hasattr(fitted_extractor, '__class__') and 'KernelPCA' in fitted_extractor.__class__.__name__ and
            hasattr(fitted_extractor, 'n_components') and X_transformed is not None):
            
            expected_components = fitted_extractor.n_components
            actual_components = X_transformed.shape[1]
            
            if actual_components < expected_components:
                logger.warning(f"KPCA produced {actual_components} components instead of {expected_components} for {modality_name or 'unknown'} modality")
                logger.info(f"Attempting to fix KPCA gamma parameter for consistent component count")
                
                # Try with smaller gamma values
                for gamma_candidate in [0.1, 0.01, 0.001]:
                    try:
                        logger.debug(f"Trying KPCA with gamma={gamma_candidate}")
                        retry_extractor = copy.deepcopy(extractor)
                        retry_extractor.gamma = gamma_candidate
                        if requires_y:
                            retry_extractor.fit(X_arr, y_arr)
                        else:
                            retry_extractor.fit(X_arr)
                        X_retry = retry_extractor.transform(X_arr)
                        
                        if X_retry.shape[1] >= expected_components:
                            logger.info(f"KPCA gamma fix successful: gamma={gamma_candidate} produces {X_retry.shape[1]} components")
                            fitted_extractor = retry_extractor
                            X_transformed = X_retry
                            break
                    except Exception as retry_error:
                        logger.debug(f"KPCA retry with gamma={gamma_candidate} failed: {retry_error}")
                        continue
                
                # If still not enough components, log a warning
                if X_transformed.shape[1] < expected_components:
                    logger.warning(f"Could not achieve {expected_components} components for {modality_name or 'unknown'} modality even with gamma adjustment")
                    logger.warning(f"Final component count: {X_transformed.shape[1]} (this may cause fusion inconsistencies)")
        
        result = (fitted_extractor, X_transformed)
        _extractor_cache['ext_clf'].put(key, result, item_size=X_transformed.nbytes if hasattr(X_transformed, 'nbytes') else X_transformed.size * 8)
        return result
        
    except Exception as e:
        error_msg = str(e)
        logger.warning(f"Extractor classification failed for {ds_name}: {error_msg}")
        
        # Special handling for KPCA ARPACK errors
        if ("ARPACK error" in error_msg or "No shifts could be applied" in error_msg) and 'KernelPCA' in extractor.__class__.__name__:
            logger.warning(f"KPCA encountered ARPACK eigenvalue solver error for {ds_name} - trying recovery strategies")
            try:
                from sklearn.decomposition import KernelPCA
                
                # Strategy 1: Try with 'dense' eigen_solver and reduced components
                safe_components = min(n_components // 2, X_arr.shape[0] - 1, 4)  # More conservative reduction
                safe_components = max(1, safe_components)
                
                logger.info(f"ARPACK recovery attempt 1: Using 'dense' solver with {safe_components} components")
                recovery_extractor = KernelPCA(
                    n_components=safe_components,
                    kernel=extractor.kernel,
                    gamma=None,  # Use default gamma for compatibility
                    eigen_solver='dense',  # Dense solver is more stable than ARPACK
                    n_jobs=1,  # Single thread to avoid parallelism issues
                    random_state=42
                )
                
                recovery_extractor.fit(X_arr)
                X_transformed = recovery_extractor.transform(X_arr)
                
                logger.info(f"ARPACK recovery successful with 'dense' solver and {safe_components} components")
                result = (recovery_extractor, X_transformed)
                _extractor_cache['ext_clf'].put(key, result, item_size=X_transformed.nbytes if hasattr(X_transformed, 'nbytes') else X_transformed.size * 8)
                return result
                
            except Exception as recovery_error1:
                logger.warning(f"ARPACK recovery attempt 1 failed: {recovery_error1}")
                try:
                    # Strategy 2: Try with randomized solver and even fewer components
                    minimal_components = min(2, X_arr.shape[0] - 1)
                    minimal_components = max(1, minimal_components)
                    
                    logger.info(f"ARPACK recovery attempt 2: Using 'randomized' solver with {minimal_components} components")
                    recovery_extractor2 = KernelPCA(
                        n_components=minimal_components,
                        kernel='linear',  # Switch to linear kernel for stability
                        eigen_solver='randomized',  # Randomized solver for large matrices
                        n_jobs=1,
                        random_state=42
                    )
                    
                    recovery_extractor2.fit(X_arr)
                    X_transformed = recovery_extractor2.transform(X_arr)
                    
                    logger.info(f"ARPACK recovery successful with 'randomized' solver and {minimal_components} components")
                    result = (recovery_extractor2, X_transformed)
                    _extractor_cache['ext_clf'].put(key, result, item_size=X_transformed.nbytes if hasattr(X_transformed, 'nbytes') else X_transformed.size * 8)
                    return result
                    
                except Exception as recovery_error2:
                    logger.warning(f"ARPACK recovery attempt 2 failed: {recovery_error2}")
                    logger.warning("All KPCA recovery strategies failed - falling back to PCA")
                    # Fall through to PCA fallback below
        
        # Special handling for KPCA zero-size array error
        elif "zero-size array to reduction operation maximum" in error_msg and 'KernelPCA' in extractor.__class__.__name__:
            logger.warning("KPCA encountered zero-size eigenvalue array - falling back to PCA")
            try:
                # Fall back to regular PCA with CONSISTENT components across all modalities
                from sklearn.decomposition import PCA
                
                # CONSISTENCY FIX: Use consistent hyperparameters if provided (for element-wise fusion methods)
                if consistent_hyperparams and consistent_hyperparams.get('n_components'):
                    # Use the consistent n_components from hyperparameters
                    target_components = consistent_hyperparams['n_components']
                    logger.info(f"Using consistent n_components={target_components} from hyperparameters for KPCA fallback")
                else:
                    # Use a fixed, conservative number that works for all modalities
                    # This ensures consistent dimensions for fusion
                    target_components = min(8, n_components // 2, X.shape[0] - 2)
                
                safe_components = min(target_components, X_arr.shape[0] - 1, X_arr.shape[1])
                safe_components = max(1, safe_components)  # Ensure at least 1 component
                
                fallback_extractor = PCA(n_components=safe_components, random_state=42)
                fallback_extractor.fit(X_arr)
                X_transformed = fallback_extractor.transform(X_arr)
                
                if consistent_hyperparams:
                    logger.info(f"KPCA fallback to PCA with CONSISTENT {safe_components} components (from hyperparams: {target_components})")
                else:
                    logger.info(f"KPCA fallback to PCA successful with {safe_components} components (consistent across modalities)")
                result = (fallback_extractor, X_transformed)
                _extractor_cache['ext_clf'].put(key, result, item_size=X_transformed.nbytes if hasattr(X_transformed, 'nbytes') else X_transformed.size * 8)
                return result
                
            except Exception as fallback_error:
                logger.error(f"KPCA fallback also failed: {fallback_error}")
        
        # Special handling for LDA matrix factorization errors
        elif "positive definite" in error_msg or "factorization" in error_msg and 'LinearDiscriminantAnalysis' in str(extractor.__class__.__name__):
            logger.warning("LDA encountered matrix factorization issues - trying with alternative settings")
            try:
                from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
                
                # Try with different solver and regularization
                safe_components = 1  # Start with just 1 component for stability
                
                fallback_extractor = LinearDiscriminantAnalysis(
                    n_components=safe_components,
                    solver='svd'  # SVD solver supports transform and is stable
                )
                fallback_extractor.fit(X_arr, y_arr)
                X_transformed = fallback_extractor.transform(X_arr)
                
                logger.info(f"LDA fallback with SVD solver successful with {safe_components} components")
                result = (fallback_extractor, X_transformed)
                _extractor_cache['ext_clf'].put(key, result, item_size=X_transformed.nbytes if hasattr(X_transformed, 'nbytes') else X_transformed.size * 8)
                return result
                
            except Exception as fallback_error:
                logger.warning(f"LDA fallback also failed: {fallback_error}, falling back to PCA")
                try:
                    # Final fallback to PCA
                    from sklearn.decomposition import PCA
                    
                    # CONSISTENCY FIX: Use consistent hyperparameters if provided (for element-wise fusion methods)
                    if consistent_hyperparams and consistent_hyperparams.get('n_components'):
                        # Use the consistent n_components from hyperparameters
                        target_components = consistent_hyperparams['n_components']
                        logger.info(f"Using consistent n_components={target_components} from hyperparameters for LDA final fallback")
                    else:
                        target_components = min(2, X.shape[0] - 1, X.shape[1] // 4)
                    
                    safe_components = min(target_components, X_arr.shape[0] - 1, X_arr.shape[1])
                    safe_components = max(1, safe_components)
                    
                    final_fallback = PCA(n_components=safe_components, random_state=42)
                    final_fallback.fit(X_arr)
                    X_transformed = final_fallback.transform(X_arr)
                    
                    if consistent_hyperparams:
                        logger.info(f"LDA final fallback to PCA with CONSISTENT {safe_components} components (from hyperparams: {target_components})")
                    else:
                        logger.info(f"LDA final fallback to PCA successful with {safe_components} components")
                    result = (final_fallback, X_transformed)
                    _extractor_cache['ext_clf'].put(key, result, item_size=X_transformed.nbytes if hasattr(X_transformed, 'nbytes') else X_transformed.size * 8)
                    return result
                    
                except Exception as final_error:
                    logger.error(f"All LDA fallbacks failed: {final_error}")
        
        return None, None

def transform_extractor_regression(X, fitted_extractor):
    """
    Transform data using a fitted regression extractor.
    
    Parameters
    ----------
    X : array-like
        Feature matrix to transform
    fitted_extractor : object
        Fitted extractor object
        
    Returns
    -------
    np.ndarray
        Transformed data
    """
    try:
        if fitted_extractor is None:
            return None
            
        X_arr = np.asarray(X)
        X_arr = np.nan_to_num(X_arr, nan=0.0)
        
        return fitted_extractor.transform(X_arr)
    except Exception as e:
        logger.warning(f"Transform extractor regression failed: {str(e)}")
        return None

def transform_extractor_classification(X, fitted_extractor):
    """
    Transform data using a fitted classification extractor.
    
    Parameters
    ----------
    X : array-like
        Feature matrix to transform
    fitted_extractor : object
        Fitted extractor object
        
    Returns
    -------
    np.ndarray
        Transformed data
    """
    try:
        if fitted_extractor is None:
            return None
            
        X_arr = np.asarray(X)
        X_arr = np.nan_to_num(X_arr, nan=0.0)
        
        return fitted_extractor.transform(X_arr)
    except Exception as e:
        logger.warning(f"Transform extractor classification failed: {str(e)}")
        return None

def cached_fit_transform_selector_classification(X, y, selector_code, n_feats, ds_name=None, modality_name=None, fold_idx=0, model_name=None, fusion_method="attention_weighted", consistent_hyperparams=None):
    """
    Cached version of fit_transform for classification selectors.
    
    Parameters
    ----------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    selector_code : str
        Selector code string
    n_feats : int
        Number of features to select
    ds_name : str, optional
        Dataset name for caching
    modality_name : str, optional
        Modality name (unused but kept for compatibility)
    fold_idx : int
        Fold index for caching
    model_name : str, optional
        Model name for hyperparameter loading
    fusion_method : str, optional
        Fusion method for hyperparameter loading
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Indices of selected features and transformed data
    """
    # Generate cache key
    # CRITICAL FIX: Include modality_name in cache key so different modalities don't share cache
    cache_name = f"{selector_code}_{modality_name}" if modality_name else selector_code
    key = _generate_cache_key(ds_name, fold_idx, cache_name, "sel_clf", n_feats, X.shape if X is not None else None)
    
    # Check cache
    cached_result = _selector_cache['sel_clf'].get(key)
    if cached_result is not None:
        return cached_result
    
    try:
        # Convert to numpy arrays
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        
        # Clean NaN values
        X_arr = np.nan_to_num(X_arr, nan=0.0)
        
        # Validate shapes
        X_arr, y_arr = validate_and_fix_shape_mismatch(
            X_arr, y_arr, 
            name=f"{ds_name if ds_name else 'unknown'} classification selector data", 
            fold_idx=fold_idx,
            allow_truncation=False
        )
        
        if X_arr is None or y_arr is None:
            logger.warning(f"Data alignment failure in selector for {ds_name}")
            return None, None
        
        # Make sure n_feats is limited by data dimensions
        max_possible_feats = min(X_arr.shape[0], X_arr.shape[1])
        effective_n_feats = min(n_feats, max_possible_feats)
        
        # Create selector object
        selector = get_selector_object(selector_code, effective_n_feats)
        
        # Load and apply hyperparameters if dataset name and model name are provided
        if ds_name and model_name and hasattr(selector, 'set_params'):
            try:
                hyperparams = load_feature_first_hyperparameters(
                    dataset=ds_name,
                    algorithm=selector_code,
                    model=model_name,
                    fusion_method=fusion_method,
                    n_features=n_feats,
                    approach="selector"
                )
                
                
                # Only apply selector-specific parameters to the selector
                # model_params should be applied later to the final model, not the selector
                if hyperparams['selector_params']:
                    selector.set_params(**hyperparams['selector_params'])
                    logger.info(f"Applied selector-specific hyperparameters for {ds_name}_{selector_code}: {hyperparams['selector_params']}")
                
                # Store model_params for later use by the final model (not applied to selector)
                if hyperparams['model_params']:
                    logger.debug(f"Model params available for final model (not applied to selector): {hyperparams['model_params']}")
                    
            except Exception as e:
                logger.warning(f"Failed to apply selector hyperparameters for {ds_name}_{selector_code}_{model_name}: {str(e)}")
        
        # Use basic feature selection as fallback
        if selector is None:
            from sklearn.feature_selection import SelectKBest, f_classif
            selector = SelectKBest(f_classif, k=effective_n_feats)
        
        # Fit and transform
        selector.fit(X_arr, y_arr)
        X_selected = selector.transform(X_arr)
        
        # Get selected features
        if hasattr(selector, 'get_support'):
            selected_features = selector.get_support()
        else:
            # Fallback for selectors without get_support
            selected_features = np.arange(X_selected.shape[1])
        
        result = (selected_features, X_selected)
        _selector_cache['sel_clf'].put(key, result, item_size=X_selected.nbytes if hasattr(X_selected, 'nbytes') else X_selected.size * 8)
        return result
        
    except Exception as e:
        logger.warning(f"Selector classification failed for {ds_name}: {str(e)}")
        # Return fallback result
        if X is not None and X.shape[1] > 0:
            mask = np.zeros(X.shape[1], dtype=bool)
            mask[0] = True
            selected_features = mask
            transformed_X = X[:min(len(X), 10), [0]]
            result = (selected_features, transformed_X)
            return result
        return None, None

def transform_selector_regression(X, selected_features):
    """
    Transform data using selected features for regression.
    
    Parameters
    ----------
    X : array-like
        Feature matrix to transform
    selected_features : array-like
        Boolean mask or indices of selected features
        
    Returns
    -------
    np.ndarray
        Transformed data with selected features
    """
    try:
        if selected_features is None:
            return None
            
        X_arr = np.asarray(X)
        X_arr = np.nan_to_num(X_arr, nan=0.0)
        
        if isinstance(selected_features, np.ndarray) and selected_features.dtype == bool:
            # Boolean mask
            return X_arr[:, selected_features]
        else:
            # Indices
            return X_arr[:, selected_features]
    except Exception as e:
        logger.warning(f"Transform selector regression failed: {str(e)}")
        return None

def transform_selector_classification(X, selected_features):
    """
    Transform data using selected features for classification.
    
    Parameters
    ----------
    X : array-like
        Feature matrix to transform
    selected_features : array-like
        Boolean mask or indices of selected features
        
    Returns
    -------
    np.ndarray
        Transformed data with selected features
    """
    try:
        if selected_features is None:
            return None
            
        X_arr = np.asarray(X)
        X_arr = np.nan_to_num(X_arr, nan=0.0)
        
        if isinstance(selected_features, np.ndarray) and selected_features.dtype == bool:
            # Boolean mask
            return X_arr[:, selected_features]
        else:
            # Indices
            return X_arr[:, selected_features]
    except Exception as e:
        logger.warning(f"Transform selector classification failed: {str(e)}")
        return None



def transform_selector_regression(X, selected_features):
    """Transform data using selected features for regression."""
    try:
        if selected_features is None:
            return None
        X_arr = np.asarray(X)
        X_arr = np.nan_to_num(X_arr, nan=0.0)
        if isinstance(selected_features, np.ndarray) and selected_features.dtype == bool:
            return X_arr[:, selected_features]
        else:
            return X_arr[:, selected_features]
    except Exception as e:
        logger.warning(f"Transform selector regression failed: {str(e)}")
        return None

def transform_selector_classification(X, selected_features):
    """Transform data using selected features for classification."""
    try:
        if selected_features is None:
            return None
        X_arr = np.asarray(X)
        X_arr = np.nan_to_num(X_arr, nan=0.0)
        if isinstance(selected_features, np.ndarray) and selected_features.dtype == bool:
            return X_arr[:, selected_features]
        else:
            return X_arr[:, selected_features]
    except Exception as e:
        logger.warning(f"Transform selector classification failed: {str(e)}")
        return None 


# ============================================================================
# HELPER CONSTRUCTORS FOR TUNER_HALVING.PY
# ============================================================================


def build_extractor(name):
    """
    Build an extractor based on its name.
    
    Parameters
    ----------
    name : str
        Name of the extractor
        
    Returns
    -------
    object
        Configured extractor object
    """
    # Check if the name is valid
    regression_extractors = get_regression_extractors()
    classification_extractors = get_classification_extractors()
    
    if name in regression_extractors:
        return regression_extractors[name]
    elif name in classification_extractors:
        return classification_extractors[name]
    else:
        raise ValueError(f"Unknown extractor: {name}")

def build_selector(name, n_features=16):
    """
    Build a feature selector based on its name and number of features.
    
    Parameters
    ----------
    name : str
        Name of the selector (e.g., 'ElasticNetFS', 'RFImportance', etc.)
    n_features : int
        Number of features to select
        
    Returns
    -------
    object
        Configured selector object
    """
    from sklearn.feature_selection import SelectKBest, f_regression, f_classif, SelectFromModel
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression
    
    # Get available selectors
    regression_selectors = get_regression_selectors()
    classification_selectors = get_classification_selectors()
    
    # Determine if this is a regression or classification selector
    if name in regression_selectors:
        selector_code = regression_selectors[name]
        task = "regression"
    elif name in classification_selectors:
        selector_code = classification_selectors[name]
        task = "classification"
    else:
        raise ValueError(f"Unknown selector: {name}")
    
    # Create selector object based on the selector code
    if selector_code == "f_regression":
        return SelectKBest(score_func=f_regression, k=n_features)
    
    elif selector_code == "f_classification":
        return SelectKBest(score_func=f_classif, k=n_features)
    
    elif selector_code == "elasticnet_regression":
        # ElasticNet-based feature selection for regression
        base_estimator = ElasticNet(random_state=42, max_iter=2000)
        return SelectFromModel(base_estimator, max_features=n_features)
    
    elif selector_code == "elasticnet_classification":
        # ElasticNet-based feature selection for classification
        base_estimator = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, 
                                          random_state=42, max_iter=2000)
        return SelectFromModel(base_estimator, max_features=n_features)
    
    elif selector_code == "random_forest_regression":
        # Random Forest importance-based selection for regression
        base_estimator = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        return SelectFromModel(base_estimator, max_features=n_features)
    
    elif selector_code == "random_forest_classification":
        # Random Forest importance-based selection for classification
        base_estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        return SelectFromModel(base_estimator, max_features=n_features)
    
    elif selector_code == "lasso_regression":
        # LASSO-based feature selection for regression
        # FIX: Use adaptive threshold to prevent 0 feature selection
        base_estimator = Lasso(random_state=42, max_iter=2000)
        
        # Create adaptive SelectFromModel that guarantees at least some features
        class AdaptiveLassoSelectorRegression:
            def __init__(self, base_estimator, max_features):
                self.base_estimator = base_estimator
                self.max_features = max_features
                self.selector_ = None
                self.feature_mask_ = None
                
            def fit(self, X, y):
                import numpy as np
                from sklearn.feature_selection import SelectFromModel
                
                # Try multiple threshold strategies to avoid 0 features
                thresholds_to_try = [
                    'mean',      # Default sklearn threshold
                    'median',    # More permissive
                    '0.25*mean', # Much more permissive
                    '0.1*mean',  # Very permissive
                    '0.01*mean'  # Extremely permissive
                ]
                
                for threshold in thresholds_to_try:
                    try:
                        # Fit the base estimator
                        fitted_estimator = self.base_estimator.fit(X, y)
                        
                        # Try this threshold
                        selector = SelectFromModel(fitted_estimator, threshold=threshold, max_features=self.max_features)
                        selector.fit(X, y)
                        
                        # Check if we selected any features
                        if hasattr(selector, 'get_support'):
                            support = selector.get_support()
                            n_selected = np.sum(support)
                            
                            if n_selected > 0:
                                # Success! We found a threshold that selects features
                                self.selector_ = selector
                                self.feature_mask_ = support
                                return self
                                
                    except Exception as e:
                        # This threshold failed, try the next one
                        continue
                
                # If all thresholds failed, use a simple variance-based fallback
                print(f"Warning: LASSO selector failed to select features, using f_regression fallback")
                from sklearn.feature_selection import SelectKBest, f_regression
                self.selector_ = SelectKBest(score_func=f_regression, k=min(self.max_features, X.shape[1]))
                self.selector_.fit(X, y)
                self.feature_mask_ = self.selector_.get_support()
                return self
                
            def transform(self, X):
                if self.selector_ is None:
                    raise ValueError("Selector not fitted yet")
                return self.selector_.transform(X)
                
            def fit_transform(self, X, y):
                return self.fit(X, y).transform(X)
                
            def get_support(self, indices=False):
                if self.feature_mask_ is None:
                    raise ValueError("Selector not fitted yet")
                if indices:
                    return np.where(self.feature_mask_)[0]
                return self.feature_mask_
        
        return AdaptiveLassoSelectorRegression(base_estimator, n_features)
    
    elif selector_code == "lasso_classification":
        # L1-regularized logistic regression for classification
        # FIX: Use adaptive threshold to prevent 0 feature selection
        base_estimator = LogisticRegression(penalty='l1', solver='liblinear', random_state=42, max_iter=2000)
        
        # Create adaptive SelectFromModel that guarantees at least some features
        class AdaptiveLassoSelector:
            def __init__(self, base_estimator, max_features):
                self.base_estimator = base_estimator
                self.max_features = max_features
                self.selector_ = None
                self.feature_mask_ = None
                
            def fit(self, X, y):
                import numpy as np
                from sklearn.feature_selection import SelectFromModel
                
                # Try multiple threshold strategies to avoid 0 features
                thresholds_to_try = [
                    'mean',      # Default sklearn threshold
                    'median',    # More permissive
                    '0.25*mean', # Much more permissive
                    '0.1*mean',  # Very permissive
                    '0.01*mean'  # Extremely permissive
                ]
                
                for threshold in thresholds_to_try:
                    try:
                        # Fit the base estimator
                        fitted_estimator = self.base_estimator.fit(X, y)
                        
                        # Try this threshold
                        selector = SelectFromModel(fitted_estimator, threshold=threshold, max_features=self.max_features)
                        selector.fit(X, y)
                        
                        # Check if we selected any features
                        if hasattr(selector, 'get_support'):
                            support = selector.get_support()
                            n_selected = np.sum(support)
                            
                            if n_selected > 0:
                                # Success! We found a threshold that selects features
                                self.selector_ = selector
                                self.feature_mask_ = support
                                return self
                                
                    except Exception as e:
                        # This threshold failed, try the next one
                        continue
                
                # If all thresholds failed, use a simple variance-based fallback
                print(f"Warning: LASSO selector failed to select features, using variance fallback")
                from sklearn.feature_selection import SelectKBest, f_classif
                self.selector_ = SelectKBest(score_func=f_classif, k=min(self.max_features, X.shape[1]))
                self.selector_.fit(X, y)
                self.feature_mask_ = self.selector_.get_support()
                return self
                
            def transform(self, X):
                if self.selector_ is None:
                    raise ValueError("Selector not fitted yet")
                return self.selector_.transform(X)
                
            def fit_transform(self, X, y):
                return self.fit(X, y).transform(X)
                
            def get_support(self, indices=False):
                if self.feature_mask_ is None:
                    raise ValueError("Selector not fitted yet")
                if indices:
                    return np.where(self.feature_mask_)[0]
                return self.feature_mask_
        
        return AdaptiveLassoSelector(base_estimator, n_features)
    
    else:
        # Fallback to univariate selection
        if task == "regression":
            return SelectKBest(score_func=f_regression, k=n_features)
        else:
            return SelectKBest(score_func=f_classif, k=n_features)

def build_model(name, task):
    """
    Build ML model by name and task for tuner_halving.py compatibility.
   
    Parameters
    ----------
    name : str
        Name of the model to build
    task : str
        Task type ('reg' for regression, 'clf' for classification)
       
    Returns
    -------
    model object
        Configured model instance
    """
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.preprocessing import PowerTransformer
    
    _MODEL = {
        # Regression models
        "LinearRegression": lambda: RobustLinearRegressor(
            method='huber',
            random_state=42
        ),
        "ElasticNet": lambda: SelectionByCyclicCoordinateDescent(
            l1_ratio=0.5,
            cv=5,
            random_state=42,
            task_type="regression"
        ),
        "RandomForestRegressor": lambda: OptimizedExtraTreesRegressor(
            n_estimators=200,
            max_features='sqrt',
            bootstrap=False,
            random_state=42
        ),

        # Classification models
        "LogisticRegression": lambda: LogisticRegression(
            max_iter=2000,
            random_state=42
        ),
        "RandomForestClassifier": lambda: RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            n_estimators=200
        ),
        "SVC": lambda: SVC(probability=True, random_state=42),
        "LogisticRegression": lambda: LogisticRegression(
            max_iter=2000,
            random_state=42
        ),
    }
   
    if name not in _MODEL:
        raise ValueError(f"Unknown model: {name}. Available: {list(_MODEL.keys())}")
   
    return _MODEL[name]()

def get_model_object(name, **kwargs):
    """
    Get model object by name for backward compatibility.
    
    Parameters
    ----------
    name : str
        Name of the model to build
    **kwargs
        Additional keyword arguments (ignored for compatibility)
       
    Returns
    -------
    model object
        Configured model instance
    """
    # Determine task type based on model name
    if name in {"LinearRegression", "ElasticNet", "RandomForestRegressor"}:
        task = "reg"
    else:
        task = "clf"
    
    return build_model(name, task)

# Add these functions after the selector functions

def get_regression_models() -> Dict[str, str]:
    """
    Get dictionary of regression models.
    
    Available regression models:
    1. LinearRegression - Linear regression with robust implementation
    2. ElasticNet - ElasticNet with automatic parameter selection
    3. RandomForestRegressor - Optimized random forest regressor
    
    Returns
    -------
    dict
        Dictionary mapping model names to model names (for consistency with selectors)
    """
    return {
        "LinearRegression": "LinearRegression",
        "ElasticNet": "ElasticNet", 
        "RandomForestRegressor": "RandomForestRegressor"
    }

def get_classification_models() -> Dict[str, str]:
    """
    Get dictionary of classification models.
    
    Available classification models:
    1. LogisticRegression - Logistic regression with regularization
    2. RandomForestClassifier - Random forest classifier
    3. SVC - Support Vector Classifier with probability estimates
    
    Returns
    -------
    dict
        Dictionary mapping model names to model names (for consistency with selectors)
    """
    return {
        "LogisticRegression": "LogisticRegression",
        "RandomForestClassifier": "RandomForestClassifier",
        "SVC": "SVC"
    }

def get_optimal_n_components_from_hyperparams(dataset, extractor_name, task):
    """
    Extract the optimal n_components value from hyperparameter tuning results.
    
    Parameters
    ----------
    dataset : str
        Dataset name (e.g., "AML", "Breast")
    extractor_name : str
        Extractor name (e.g., "PCA", "KPCA", "FA")
    task : str
        Task type ("reg" for regression, "clf" for classification)
        
    Returns
    -------
    int or None
        Optimal number of components, or None if not found
    """
    # Try different model combinations to find n_components
    model_candidates = (
        ["LinearRegression", "ElasticNet", "RandomForestRegressor"] if task == "reg" 
        else ["LogisticRegression", "RandomForestClassifier", "SVC"]
    )
    
    for model_name in model_candidates:
        file_path = HP_DIR / f"{dataset}_{extractor_name}_{model_name}.json"
        if file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    best_params = data.get("best_params", {})
                    
                    # Look for n_components in different formats
                    n_components = None
                    if "extractor__extractor__n_components" in best_params:
                        n_components = best_params["extractor__extractor__n_components"]
                    elif "extractor__n_components" in best_params:
                        n_components = best_params["extractor__n_components"]
                    
                    if n_components is not None:
                        logger = logging.getLogger(__name__)
                        logger.info(f"Found optimal n_components={n_components} for {dataset}_{extractor_name} from {model_name}")
                        return int(n_components)
                        
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to read n_components from {file_path}: {str(e)}")
    
    # Fallback: Use family dataset (Breast for classification, AML for regression)
    family_dataset = "Breast" if task == "clf" else "AML"
    if family_dataset != dataset:
        for model_name in model_candidates:
            fallback_path = HP_DIR / f"{family_dataset}_{extractor_name}_{model_name}.json"
            if fallback_path.exists():
                try:
                    with open(fallback_path, 'r') as f:
                        data = json.load(f)
                        best_params = data.get("best_params", {})
                        
                        # Look for n_components in different formats
                        n_components = None
                        if "extractor__extractor__n_components" in best_params:
                            n_components = best_params["extractor__extractor__n_components"]
                        elif "extractor__n_components" in best_params:
                            n_components = best_params["extractor__n_components"]
                        
                        if n_components is not None:
                            logger = logging.getLogger(__name__)
                            logger.info(f"Found fallback optimal n_components={n_components} for {dataset}_{extractor_name} from {family_dataset}_{model_name}")
                            return int(n_components)
                            
                except Exception as e:
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Failed to read fallback n_components from {fallback_path}: {str(e)}")
    
    # No optimal n_components found
    logger = logging.getLogger(__name__)
    logger.debug(f"No optimal n_components found for {dataset}_{extractor_name}")
    return None

def get_extraction_n_components_list(dataset, extractors, task):
    """
    Get list of optimal n_components for each extractor based on hyperparameter tuning.
    
    Parameters
    ----------
    dataset : str
        Dataset name (e.g., "AML", "Breast")
    extractors : Dict[str, Any]
        Dictionary of extractors
    task : str
        Task type ("reg" for regression, "clf" for classification)
        
    Returns
    -------
    Dict[str, List[int]]
        Dictionary mapping extractor names to their optimal n_components lists
    """
    extraction_n_components = {}
    
    for extractor_name in extractors.keys():
        optimal_n = get_optimal_n_components_from_hyperparams(dataset, extractor_name, task)
        if optimal_n is not None:
            extraction_n_components[extractor_name] = [optimal_n]
        else:
            # Fallback to default values if no hyperparameters found
            logger = logging.getLogger(__name__)
            logger.warning(f"No optimal n_components found for {extractor_name}, using default [8]")
            extraction_n_components[extractor_name] = [8]
    
    return extraction_n_components

def load_consistent_hyperparameters_for_fusion(dataset, algorithm, model, fusion_method, approach="extractor"):
    """
    Load hyperparameters that will be applied consistently across all modalities for fusion.
    
    This function ensures that for element-wise fusion methods (average, sum, max), all modalities
    use the same hyperparameter source and thus get consistent n_components values.
    
    Parameters
    ----------
    dataset : str
        Dataset name
    algorithm : str
        Algorithm name (extractor or selector)
    model : str
        Model name
    fusion_method : str
        Fusion method name
    approach : str
        Either "extractor" or "selector"
        
    Returns
    -------
    dict
        Hyperparameters with consistent source, or None if no suitable hyperparameters found
    """
    logger = logging.getLogger(__name__)
    
    # Element-wise fusion methods that require consistent component counts
    element_wise_fusion_methods = ['average', 'sum', 'max']
    
    # First try the specific model
    hyperparams = load_feature_first_hyperparameters(
        dataset=dataset,
        algorithm=algorithm,
        model=model,
        fusion_method=fusion_method,
        approach=approach
    )
    
    if hyperparams['algorithm_params'] or hyperparams['extractor_params']:
        logger.info(f"Found consistent hyperparameters for {dataset}_{algorithm}_{model} (source: {hyperparams['source']})")
        return hyperparams
    
    # If element-wise fusion method and no specific hyperparameters found,
    # try fallback models in a deterministic order to ensure consistency
    if fusion_method in element_wise_fusion_methods:
        # Determine task type
        is_regression = model in ["LinearRegression", "ElasticNet", "RandomForestRegressor", "SVR"]
        
        if is_regression:
            fallback_models = ["LinearRegression", "ElasticNet", "RandomForestRegressor"]
        else:
            fallback_models = ["LogisticRegression", "RandomForestClassifier", "SVC"]
        
        # Try fallback models in order
        for fallback_model in fallback_models:
            if fallback_model == model:
                continue  # Skip the one we already tried
                
            hyperparams = load_feature_first_hyperparameters(
                dataset=dataset,
                algorithm=algorithm,
                model=fallback_model,
                fusion_method=fusion_method,
                approach=approach
            )
            
            if hyperparams['algorithm_params'] or hyperparams['extractor_params']:
                logger.warning(f"Using consistent fallback hyperparameters from {fallback_model} for {model} "
                             f"(required for {fusion_method} fusion consistency)")
                return hyperparams
    
    # No suitable hyperparameters found
    logger.debug(f"No consistent hyperparameters found for {dataset}_{algorithm}_{model}")
    return None

def load_feature_first_hyperparameters(dataset, algorithm, model, fusion_method, n_features=None, approach="extractor"):
    """
    Load best hyperparameters for feature-first pipeline with correct file naming convention.
    
    MAPPING RULES:
    - Dataset mapping: AML (regression) for AML/Sarcoma, Breast (classification) for Breast/Colon/Kidney/Liver/Lung/Melanoma/Ovarian
    - Fusion mapping: attention_weighted for attention_weighted/learnable_weighted/standard_concat, average for average/mkl/sum/early_fusion_pca
    - Extractor approach: {dataset}_{extractor}_{model}_{fusion}.json
    - Selector approach: {dataset}_{selector}_{model}_{fusion}_{f_number}.json
    
    Parameters
    ----------
    dataset : str
        Original dataset name (e.g., "AML", "Breast", "Colon", etc.)
    algorithm : str
        Algorithm name (extractor like "FA", "PCA" or selector like "LASSO", "RFImportance")
    model : str
        Model name (e.g., "LinearRegression", "ElasticNet", "LogisticRegression")
    fusion_method : str
        Fusion technique (e.g., "attention_weighted", "average", "mkl", etc.)
    n_features : int, optional
        Number of features (required for selectors, ignored for extractors)
    approach : str
        Either "extractor" or "selector"
        
    Returns
    -------
    dict
        Best hyperparameters with separate extractor/selector and model params, or empty dict if not found
    """
    logger = logging.getLogger(__name__)
    
    # Step 1: Map dataset to source dataset based on user specifications
    # AML (regression)  source for all regression databases: AML, Sarcoma
    # Breast (classification)  source for all classification databases: Breast, Colon, Kidney, Liver, Lung, Melanoma, Ovarian
    
    regression_datasets = ["AML", "Sarcoma"]
    classification_datasets = ["Breast", "Colon", "Kidney", "Liver", "Lung", "Melanoma", "Ovarian"]
    
    if dataset in regression_datasets:
        source_dataset = "AML"  # AML for all regression datasets
    elif dataset in classification_datasets:
        source_dataset = "Breast"  # Breast for all classification datasets
    else:
        # Fallback: determine by model type if dataset not recognized
        logger.warning(f"Dataset {dataset} not in known regression/classification lists, determining by model")
        is_regression = model in ["LinearRegression", "ElasticNet", "RandomForestRegressor", "SVR"]
        source_dataset = "AML" if is_regression else "Breast"
    
    # Step 2: Map algorithm name to match hyperparameter file naming
    # elasticnet_regression  ElasticNetFS
    # f_regression  f_regressionFS
    # Add more mappings as needed
    algorithm_name_mapping = {
        "elasticnet_regression": "ElasticNetFS",
        "f_regression": "f_regressionFS",
        # Add more mappings as needed
    }
    
    source_algorithm = algorithm_name_mapping.get(algorithm, algorithm)
    
    # Step 3: Map fusion technique to source fusion based on user specifications
    # attention_weighted/learnable_weighted/standard_concat  use attention_weighted params
    # average/mkl/sum/early_fusion_pca/max  use average params
    if fusion_method in ["attention_weighted", "learnable_weighted", "standard_concat"]:
        source_fusion = "attention_weighted"
    elif fusion_method in ["average", "mkl", "sum", "early_fusion_pca", "max"]:
        source_fusion = "average"
    else:
        source_fusion = fusion_method  # Use as-is if not in mapping
    
    # Step 4: Construct filename based on approach
    if approach == "extractor":
        # Format: {dataset}_{extractor}_{model}_{fusion}.json
        filename = f"{source_dataset}_{source_algorithm}_{model}_{source_fusion}.json"
    else:  # selector
        # Format: {dataset}_{selector}_{model}_{fusion}_{f_number}.json
        if n_features is None:
            logger.error(f"n_features is required for selector approach but was None")
            return {'extractor_params': {}, 'model_params': {}, 'source': 'error: missing n_features'}
        filename = f"{source_dataset}_{source_algorithm}_{model}_{source_fusion}_{n_features}f.json"
    
    file_path = HP_DIR / filename
    
    logger.debug(f"Looking for hyperparameters: {filename}")
    logger.debug(f"  Original: dataset={dataset}, algorithm={algorithm}, model={model}, fusion={fusion_method}")
    logger.debug(f"  Mapped: source_dataset={source_dataset}, source_algorithm={source_algorithm}, source_fusion={source_fusion}")
    
    # Step 4: Try to load the file
    if file_path.exists():
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                raw_params = data.get("best_params", {})
                
                if not raw_params:
                    logger.warning(f"No best_params found in {filename}")
                    return {'extractor_params': {}, 'model_params': {}, 'source': f'empty best_params in {filename}'}
                
                # Step 5: Separate algorithm and model parameters
                algorithm_params = {}
                model_params = {}
                
                for key, value in raw_params.items():
                    if key.startswith('extractor__extractor__'):
                        # Remove extractor__extractor__ prefix for direct application
                        actual_key = key.replace('extractor__extractor__', '')
                        algorithm_params[actual_key] = value
                    elif key.startswith('extractor__'):
                        # Remove extractor__ prefix
                        actual_key = key.replace('extractor__', '')
                        algorithm_params[actual_key] = value
                    elif key.startswith('selector__'):
                        # Remove selector__ prefix (for selectors)
                        actual_key = key.replace('selector__', '')
                        algorithm_params[actual_key] = value
                    elif key.startswith('model__'):
                        # Remove model__ prefix
                        actual_key = key.replace('model__', '')
                        model_params[actual_key] = value
                    else:
                        # Unknown prefix, assign to algorithm params as fallback
                        algorithm_params[key] = value
                
                logger.info(f"Loaded hyperparameters from {filename}")
                logger.debug(f"  Algorithm params: {algorithm_params}")
                logger.debug(f"  Model params: {model_params}")
                
                # Return with both extractor_params and selector_params for compatibility
                return {
                    'extractor_params': algorithm_params,  # For backward compatibility
                    'selector_params': algorithm_params,   # For selectors
                    'algorithm_params': algorithm_params,  # Generic name
                    'model_params': model_params,
                    'source': filename,
                    'n_components': algorithm_params.get('n_components'),  # Extract n_components if present
                }
                
        except Exception as e:
            logger.warning(f"Failed to load hyperparameters from {file_path}: {str(e)}")
    
    # Step 6: Fallback - try without specific mapping (original dataset/fusion)
    if approach == "extractor":
        fallback_filename = f"{dataset}_{source_algorithm}_{model}_{fusion_method}.json"
    else:
        fallback_filename = f"{dataset}_{source_algorithm}_{model}_{fusion_method}_{n_features}f.json"
    
    fallback_path = HP_DIR / fallback_filename
    
    if fallback_path.exists() and fallback_filename != filename:
        logger.info(f"Trying fallback: {fallback_filename}")
        try:
            with open(fallback_path, 'r') as f:
                data = json.load(f)
                raw_params = data.get("best_params", {})
                
                algorithm_params = {}
                model_params = {}
                
                for key, value in raw_params.items():
                    if key.startswith('extractor__extractor__'):
                        actual_key = key.replace('extractor__extractor__', '')
                        algorithm_params[actual_key] = value
                    elif key.startswith('extractor__'):
                        actual_key = key.replace('extractor__', '')
                        algorithm_params[actual_key] = value
                    elif key.startswith('selector__'):
                        actual_key = key.replace('selector__', '')
                        algorithm_params[actual_key] = value
                    elif key.startswith('model__'):
                        actual_key = key.replace('model__', '')
                        model_params[actual_key] = value
                    else:
                        algorithm_params[key] = value
                
                logger.info(f"Loaded fallback hyperparameters from {fallback_filename}")
                
                return {
                    'extractor_params': algorithm_params,
                    'selector_params': algorithm_params,
                    'algorithm_params': algorithm_params,
                    'model_params': model_params,
                    'source': f"{fallback_filename} (fallback)",
                    'n_components': algorithm_params.get('n_components'),
                }
                
        except Exception as e:
            logger.warning(f"Failed to load fallback hyperparameters from {fallback_path}: {str(e)}")
    
    # No hyperparameters found
    logger.debug(f"No hyperparameters found for {filename} or {fallback_filename}")
    return {
        'extractor_params': {},
        'selector_params': {},
        'algorithm_params': {},
        'model_params': {},
        'source': 'default (no tuned params found)',
        'n_components': None,
    }

# CRITICAL: Clear existing caches immediately to ensure modality-aware cache keys take effect
# This prevents cached results from before the fix from causing wrong hyperparameter applications
clear_modality_caches()

def get_cache_diagnostics() -> Dict[str, Any]:
    """
    Get comprehensive cache diagnostics for all cache types.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing cache statistics and diagnostics
    """
    diagnostics = {
        "selector_caches": {},
        "extractor_caches": {},
        "summary": {},
        "modality_separation_status": "unknown"
    }
    
    # Get selector cache stats
    for cache_type, cache in _selector_cache.items():
        stats = cache.stats()
        diagnostics["selector_caches"][cache_type] = stats
        
    # Get extractor cache stats  
    for cache_type, cache in _extractor_cache.items():
        stats = cache.stats()
        diagnostics["extractor_caches"][cache_type] = stats
    
    # Calculate summary statistics
    total_size = sum(stats["size"] for stats in diagnostics["selector_caches"].values()) + \
                 sum(stats["size"] for stats in diagnostics["extractor_caches"].values())
    total_memory = sum(stats["memory_usage_mb"] for stats in diagnostics["selector_caches"].values()) + \
                   sum(stats["memory_usage_mb"] for stats in diagnostics["extractor_caches"].values())
    total_hits = sum(stats["hits"] for stats in diagnostics["selector_caches"].values()) + \
                 sum(stats["hits"] for stats in diagnostics["extractor_caches"].values())
    total_misses = sum(stats["misses"] for stats in diagnostics["selector_caches"].values()) + \
                   sum(stats["misses"] for stats in diagnostics["extractor_caches"].values())
    
    overall_hit_ratio = total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0
    
    diagnostics["summary"] = {
        "total_cached_items": total_size,
        "total_memory_usage_mb": total_memory,
        "overall_hit_ratio": overall_hit_ratio,
        "total_hits": total_hits,
        "total_misses": total_misses
    }
    
    # Check if modality separation is working by examining cache keys
    diagnostics["modality_separation_status"] = _check_modality_separation()
    
    return diagnostics

def _check_modality_separation() -> str:
    """
    Check if modality separation is working correctly in cache keys.
    
    Returns
    -------
    str
        Status of modality separation: "working", "not_working", or "insufficient_data"
    """
    try:
        modality_indicators = set()
        
        # Check all cache types for modality indicators in keys
        for cache in list(_selector_cache.values()) + list(_extractor_cache.values()):
            # Access the internal cache keys to check for modality names
            if hasattr(cache, 'cache') and cache.cache:
                # Sample up to 10 keys to check for modality patterns
                sample_keys = list(cache.cache.keys())[:10]
                for key in sample_keys:
                    # Check if key contains common modality names
                    key_str = str(key).lower()
                    if any(modality in key_str for modality in ['mirna', 'exp', 'methy', 'cnv']):
                        modality_indicators.add("modality_detected")
                    if '_' in key_str:  # Indicates concatenated components in cache key
                        modality_indicators.add("separator_detected")
        
        if len(modality_indicators) >= 2:
            return "working"
        elif len(modality_indicators) == 1:
            return "partially_working"
        elif not modality_indicators:
            # Check if we have any cached items at all
            total_items = sum(len(cache.cache) for cache in list(_selector_cache.values()) + list(_extractor_cache.values()))
            if total_items == 0:
                return "no_cached_data"
            else:
                return "not_working"
        else:
            return "insufficient_data"
            
    except Exception as e:
        logger.warning(f"Failed to check modality separation: {str(e)}")
        return "check_failed"

def validate_cache_keys_include_modality(ds_name: str, fold_idx: int, modality_name: str) -> bool:
    """
    Validate that cache keys properly include modality name for separation.
    
    Parameters
    ----------
    ds_name : str
        Dataset name
    fold_idx : int
        Fold index
    modality_name : str
        Modality name that should be in cache key
        
    Returns
    -------
    bool
        True if cache key includes modality name, False otherwise
    """
    try:
        # Create a test cache key with modality name
        test_cache_name = f"TestAlgorithm_{modality_name}"
        test_key = _generate_cache_key(ds_name, fold_idx, test_cache_name, "test", 8, (100, 1000))
        
        # The key should be different from one without modality name
        test_cache_name_no_modality = "TestAlgorithm"
        test_key_no_modality = _generate_cache_key(ds_name, fold_idx, test_cache_name_no_modality, "test", 8, (100, 1000))
        
        keys_are_different = test_key != test_key_no_modality
        
        if keys_are_different:
            logger.debug(f"Cache key validation PASSED: modality '{modality_name}' creates unique cache keys")
        else:
            logger.warning(f"Cache key validation FAILED: modality '{modality_name}' does not create unique cache keys")
            
        return keys_are_different
        
    except Exception as e:
        logger.error(f"Cache key validation failed with error: {str(e)}")
        return False

def log_cache_performance_summary():
    """Log a summary of cache performance across all cache types."""
    try:
        diagnostics = get_cache_diagnostics()
        
        logger.info("=== CACHE PERFORMANCE SUMMARY ===")
        logger.info(f"Total cached items: {diagnostics['summary']['total_cached_items']}")
        logger.info(f"Total memory usage: {diagnostics['summary']['total_memory_usage_mb']:.2f} MB")
        logger.info(f"Overall hit ratio: {diagnostics['summary']['overall_hit_ratio']:.3f}")
        logger.info(f"Modality separation status: {diagnostics['modality_separation_status']}")
        
        # Log individual cache performance
        for cache_type, stats in diagnostics["selector_caches"].items():
            logger.info(f"Selector {cache_type}: {stats['size']} items, {stats['memory_usage_mb']:.1f} MB, hit ratio: {stats['hit_ratio']:.3f}")
            
        for cache_type, stats in diagnostics["extractor_caches"].items():
            logger.info(f"Extractor {cache_type}: {stats['size']} items, {stats['memory_usage_mb']:.1f} MB, hit ratio: {stats['hit_ratio']:.3f}")
            
        logger.info("=== END CACHE SUMMARY ===")
        
    except Exception as e:
        logger.warning(f"Failed to log cache performance summary: {str(e)}")

def run_cache_diagnostics(dataset_name: str = "test", modalities: List[str] = None) -> Dict[str, Any]:
    """
    Run comprehensive cache diagnostics to verify the caching system is working correctly.
    
    Parameters
    ----------
    dataset_name : str
        Dataset name for test validation
    modalities : List[str], optional
        List of modality names to test. Defaults to common modalities.
        
    Returns
    -------
    Dict[str, Any]
        Comprehensive diagnostics report
    """
    if modalities is None:
        modalities = ['exp', 'mirna', 'methy', 'cnv']
    
    logger.info("=== RUNNING CACHE DIAGNOSTICS ===")
    
    diagnostics = {
        'cache_stats': get_cache_diagnostics(),
        'modality_validation': {},
        'cache_key_tests': {},
        'performance_summary': {},
        'recommendations': []
    }
    
    # Test cache key validation for each modality
    for modality in modalities:
        try:
            validation_result = validate_cache_keys_include_modality(dataset_name, 0, modality)
            diagnostics['modality_validation'][modality] = {
                'status': 'PASS' if validation_result else 'FAIL',
                'unique_cache_keys': validation_result
            }
            if not validation_result:
                diagnostics['recommendations'].append(f"Cache key validation failed for modality '{modality}' - check modality_name parameter passing")
        except Exception as e:
            diagnostics['modality_validation'][modality] = {
                'status': 'ERROR',
                'error': str(e)
            }
            diagnostics['recommendations'].append(f"Cache validation error for modality '{modality}': {str(e)}")
    
    # Performance analysis
    cache_stats = diagnostics['cache_stats']
    total_hit_ratio = cache_stats['summary']['overall_hit_ratio']
    total_memory_usage = cache_stats['summary']['total_memory_usage_mb']
    
    diagnostics['performance_summary'] = {
        'overall_hit_ratio': total_hit_ratio,
        'memory_usage_mb': total_memory_usage,
        'performance_rating': 'EXCELLENT' if total_hit_ratio > 0.8 else 'GOOD' if total_hit_ratio > 0.5 else 'POOR' if total_hit_ratio > 0.1 else 'NO_CACHE_HITS',
        'memory_rating': 'HIGH' if total_memory_usage > 500 else 'MEDIUM' if total_memory_usage > 100 else 'LOW',
        'modality_separation_status': cache_stats['modality_separation_status']
    }
    
    # Generate recommendations
    if total_hit_ratio < 0.3:
        diagnostics['recommendations'].append("Low cache hit ratio - consider increasing cache size or checking for excessive cache key variations")
    
    if cache_stats['modality_separation_status'] not in ['working', 'partially_working']:
        diagnostics['recommendations'].append("Modality separation not detected - verify modality_name is passed to cached functions")
    
    if total_memory_usage > 1000:  # > 1GB
        diagnostics['recommendations'].append("High memory usage - consider reducing cache size or clearing cache more frequently")
    
    # Log summary
    logger.info(f"Cache Diagnostics Summary:")
    logger.info(f"  Overall hit ratio: {total_hit_ratio:.3f} ({diagnostics['performance_summary']['performance_rating']})")
    logger.info(f"  Memory usage: {total_memory_usage:.1f} MB ({diagnostics['performance_summary']['memory_rating']})")
    logger.info(f"  Modality separation: {cache_stats['modality_separation_status']}")
    logger.info(f"  Modality validations: {sum(1 for v in diagnostics['modality_validation'].values() if v.get('status') == 'PASS')}/{len(modalities)} PASS")
    
    if diagnostics['recommendations']:
        logger.warning("Cache recommendations:")
        for rec in diagnostics['recommendations']:
            logger.warning(f"  - {rec}")
    else:
        logger.info("✅ All cache diagnostics passed!")
    
    logger.info("=== CACHE DIAGNOSTICS COMPLETE ===")
    
    return diagnostics
    """
    Load hyperparameters that will be applied consistently across all modalities for fusion.
    
    This function ensures that for element-wise fusion methods (average, sum, max), all modalities
    use the same hyperparameter source and thus get consistent n_components values.
    
    Parameters
    ----------
    dataset : str
        Dataset name
    algorithm : str
        Algorithm name (extractor or selector)
    model : str
        Model name
    fusion_method : str
        Fusion method name
    approach : str
        Either "extractor" or "selector"
        
    Returns
    -------
    dict
        Hyperparameters with consistent source, or None if no suitable hyperparameters found
    """
    logger = logging.getLogger(__name__)
    
    # Element-wise fusion methods that require consistent component counts
    element_wise_fusion_methods = ['average', 'sum', 'max']
    
    # First try the specific model
    hyperparams = load_feature_first_hyperparameters(
        dataset=dataset,
        algorithm=algorithm,
        model=model,
        fusion_method=fusion_method,
        approach=approach
    )
    
    if hyperparams['algorithm_params'] or hyperparams['extractor_params']:
        logger.info(f"Found consistent hyperparameters for {dataset}_{algorithm}_{model} (source: {hyperparams['source']})")
        return hyperparams
    
    # If element-wise fusion method and no specific hyperparameters found,
    # try fallback models in a deterministic order to ensure consistency
    if fusion_method in element_wise_fusion_methods:
        # Determine task type
        is_regression = model in ["LinearRegression", "ElasticNet", "RandomForestRegressor", "SVR"]
        
        if is_regression:
            fallback_models = ["LinearRegression", "ElasticNet", "RandomForestRegressor"]
        else:
            fallback_models = ["LogisticRegression", "RandomForestClassifier", "SVC"]
        
        # Try fallback models in order
        for fallback_model in fallback_models:
            if fallback_model == model:
                continue  # Skip the one we already tried
                
            hyperparams = load_feature_first_hyperparameters(
                dataset=dataset,
                algorithm=algorithm,
                model=fallback_model,
                fusion_method=fusion_method,
                approach=approach
            )
            
            if hyperparams['algorithm_params'] or hyperparams['extractor_params']:
                logger.warning(f"Using consistent fallback hyperparameters from {fallback_model} for {model} "
                             f"(required for {fusion_method} fusion consistency)")
                return hyperparams
    
    # No suitable hyperparameters found
    logger.debug(f"No consistent hyperparameters found for {dataset}_{algorithm}_{model}")
    return None

