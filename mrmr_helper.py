"""
Optimized implementation of MRMR (Maximum Relevance Minimum Redundancy)
feature selection algorithm with significant speed improvements.
"""

import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger(__name__)

def fast_mutual_info_batch(X, y, is_regression=False, n_neighbors=3):
    """
    Fast batch calculation of mutual information with optimizations.
    
    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix, shape (n_samples, n_features)
    y : numpy.ndarray
        Target vector, shape (n_samples,)
    is_regression : bool
        Whether this is a regression problem
    n_neighbors : int
        Number of neighbors for MI estimation (lower = faster)
        
    Returns
    -------
    numpy.ndarray
        Mutual information scores for each feature
    """
    # Critical fix: ensure n_neighbors doesn't exceed available samples
    n_samples = X.shape[0]
    safe_neighbors = min(n_neighbors, max(1, n_samples - 1))
    
    # Additional safety for very small datasets
    if n_samples < 3:
        logger.warning(f"Very small dataset ({n_samples} samples) for mutual info, using n_neighbors=1")
        safe_neighbors = 1
    
    if is_regression:
        return mutual_info_regression(X, y, n_neighbors=safe_neighbors, random_state=42)
    else:
        return mutual_info_classif(X, y, n_neighbors=safe_neighbors, random_state=42)

def correlation_based_redundancy(X, selected_indices, candidate_indices):
    """
    Fast approximation of redundancy using correlation instead of MI.
    Much faster than MI for large feature sets.
    
    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix
    selected_indices : list
        Indices of already selected features
    candidate_indices : list
        Indices of candidate features
        
    Returns
    -------
    numpy.ndarray
        Redundancy scores for each candidate
    """
    if not selected_indices:
        return np.zeros(len(candidate_indices))
    
    try:
        # Calculate correlation between each candidate and each selected feature
        redundancy_scores = np.zeros(len(candidate_indices))
        
        for i, candidate_idx in enumerate(candidate_indices):
            candidate_feature = X[:, candidate_idx]
            
            # Calculate correlation with each selected feature
            correlations = []
            for selected_idx in selected_indices:
                selected_feature = X[:, selected_idx]
                
                # Check for constant features (zero variance) to avoid divide by zero
                candidate_std = np.std(candidate_feature)
                selected_std = np.std(selected_feature)
                
                if candidate_std == 0 or selected_std == 0:
                    # If either feature is constant, correlation is undefined
                    # Use 0 correlation (no redundancy) for constant features
                    correlations.append(0.0)
                else:
                    # Calculate correlation coefficient with warning suppression
                    try:
                        import warnings
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', message='invalid value encountered in divide')
                            warnings.filterwarnings('ignore', message='divide by zero encountered')
                            warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
                            
                            corr = np.corrcoef(candidate_feature, selected_feature)[0, 1]
                            
                            # Handle NaN correlations (should not happen with std check, but be safe)
                            if np.isnan(corr) or np.isinf(corr):
                                correlations.append(0.0)
                            else:
                                correlations.append(abs(corr))
                    except:
                        # If correlation calculation fails for any reason, use 0
                        correlations.append(0.0)
            
            # Average absolute correlation as redundancy measure
            redundancy_scores[i] = np.mean(correlations) if correlations else 0.0
        
        return redundancy_scores
        
    except Exception as e:
        logger.warning(f"Correlation-based redundancy failed: {str(e)}, using zeros")
        return np.zeros(len(candidate_indices))

def simple_mrmr(X, y, n_selected_features=10, is_regression=False, fast_mode=None, max_features_prefilter=None, n_neighbors=None):
    """
    Optimized MRMR implementation with significant speed improvements.
    
    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix, shape (n_samples, n_features)
    y : numpy.ndarray
        Target vector, shape (n_samples,)
    n_selected_features : int
        Number of features to select
    is_regression : bool
        Whether this is a regression problem (True) or classification (False)
    fast_mode : bool, optional
        Use fast approximations for better speed (uses MRMR_CONFIG if None)
    max_features_prefilter : int, optional
        Pre-filter to top N most relevant features before MRMR (uses MRMR_CONFIG if None)
    n_neighbors : int, optional
        Number of neighbors for MI estimation (uses MRMR_CONFIG if None)
        
    Returns
    -------
    numpy.ndarray
        Indices of selected features
    """
    # Import config here to avoid circular imports
    try:
        from config import MRMR_CONFIG
        # Use config defaults if parameters not specified
        if fast_mode is None:
            fast_mode = MRMR_CONFIG.get("fast_mode", True)
        if max_features_prefilter is None:
            max_features_prefilter = MRMR_CONFIG.get("max_features_prefilter", 1000)
        if n_neighbors is None:
            n_neighbors = MRMR_CONFIG.get("n_neighbors", 3)
        progress_logging = MRMR_CONFIG.get("progress_logging", True)
        fallback_on_error = MRMR_CONFIG.get("fallback_on_error", True)
    except ImportError:
        # Fallback if config is not available
        if fast_mode is None:
            fast_mode = True
        if max_features_prefilter is None:
            max_features_prefilter = 1000
        if n_neighbors is None:
            n_neighbors = 3
        progress_logging = True
        fallback_on_error = True
    
    try:
        n_samples, n_features = X.shape
        
        # Cap n_selected_features to the number of available features
        n_selected_features = min(n_selected_features, n_features)
        
        # Suppress numerical warnings that are expected in MRMR correlation calculations
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='invalid value encountered in divide')
            warnings.filterwarnings('ignore', message='divide by zero encountered')
            warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
            
            # OPTIMIZATION 1: Pre-filter features if we have too many
            # This dramatically reduces computation time for large feature sets
            if n_features > max_features_prefilter and max_features_prefilter > n_selected_features * 2:
                if progress_logging:
                    logger.info(f"MRMR pre-filtering: {n_features} -> {max_features_prefilter} features for faster computation")
                
                # Calculate relevance for all features
                all_relevance = fast_mutual_info_batch(X, y, is_regression, n_neighbors)
                
                # Keep top max_features_prefilter most relevant features
                top_indices = np.argsort(all_relevance)[-max_features_prefilter:]
                X = X[:, top_indices]
                feature_mapping = top_indices  # Map back to original indices
                n_features = max_features_prefilter
            else:
                feature_mapping = np.arange(n_features)  # No pre-filtering
            
            return _perform_mrmr_selection(X, y, n_selected_features, is_regression, fast_mode, 
                                         max_features_prefilter, n_neighbors, progress_logging, 
                                         feature_mapping)
                                         
    except Exception as e:
        logger.error(f"MRMR failed with error: {str(e)}")
        if fallback_on_error:
            logger.warning("MRMR falling back to mutual information selection")
            # Fallback to simple mutual information selection
            try:
                if is_regression:
                    from sklearn.feature_selection import SelectKBest, mutual_info_regression
                    selector = SelectKBest(mutual_info_regression, k=n_selected_features)
                else:
                    from sklearn.feature_selection import SelectKBest, mutual_info_classif
                    selector = SelectKBest(mutual_info_classif, k=n_selected_features)
                
                selector.fit(X, y)
                selected_mask = selector.get_support()
                return np.where(selected_mask)[0]
            except Exception as e2:
                logger.error(f"Fallback also failed: {str(e2)}, returning first {n_selected_features} features")
                return np.arange(min(n_selected_features, X.shape[1]))
        else:
            raise

def _perform_mrmr_selection(X, y, n_selected_features, is_regression, fast_mode, 
                           max_features_prefilter, n_neighbors, progress_logging, 
                           feature_mapping):
    """
    Internal function to perform MRMR selection with warning suppression.
    """
    n_samples, n_features = X.shape
        
    # Cap n_selected_features to the number of available features
    n_selected_features = min(n_selected_features, n_features)
    
    # OPTIMIZATION 2: Scale features once for consistent MI computation
    if fast_mode:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        
        # Additional preprocessing: handle any remaining problematic values
        # Replace any NaN, inf, or extremely small variance features
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Check for constant features and add small noise to avoid correlation issues
        feature_stds = np.std(X, axis=0)
        constant_features = feature_stds < 1e-10
        if np.any(constant_features):
            if progress_logging:
                logger.debug(f"MRMR: Adding small noise to {np.sum(constant_features)} constant features")
            # Add very small random noise to constant features
            noise = np.random.RandomState(42).normal(0, 1e-8, X.shape)
            X[:, constant_features] += noise[:, constant_features]
    
    # Calculate relevance (mutual information with target) - batch computation
    if progress_logging:
        logger.debug(f"MRMR: Computing relevance for {n_features} features")
    relevance = fast_mutual_info_batch(X, y, is_regression, n_neighbors)
    
    # Handle edge case: if all relevance scores are 0 or NaN
    if np.all(relevance <= 0) or np.any(np.isnan(relevance)):
        logger.warning("MRMR: All relevance scores are 0 or NaN, selecting first features")
        return feature_mapping[:n_selected_features]
    
    # Initialize with the most relevant feature
    selected_local = [np.argmax(relevance)]  # Local indices (in filtered X)
    not_selected = list(range(n_features))
    not_selected.remove(selected_local[0])
    
    if progress_logging:
        logger.debug(f"MRMR: Starting with feature {feature_mapping[selected_local[0]]} (relevance: {relevance[selected_local[0]]:.4f})")
    
    # OPTIMIZATION 3: Select remaining features with batched redundancy calculation
    for i in range(1, n_selected_features):
        # Break if no more features to select
        if not not_selected:
            break
            
        # OPTIMIZATION 4: Use correlation-based redundancy approximation in fast mode
        if fast_mode:
            redundancy = correlation_based_redundancy(X, selected_local, not_selected)
        else:
            # Original MI-based redundancy (slower but more accurate)
            redundancy = np.zeros(len(not_selected))
            for j, candidate in enumerate(not_selected):
                mi_sum = 0
                for selected_idx in selected_local:
                    f1 = X[:, candidate].reshape(-1, 1)
                    f2 = X[:, selected_idx].reshape(-1, 1)
                    if is_regression:
                        mi = mutual_info_regression(f1, f2.ravel())[0]
                    else:
                        mi = mutual_info_classif(f1, f2.ravel())[0]
                    mi_sum += mi
                redundancy[j] = mi_sum / len(selected_local)
        
        # Compute MRMR criterion (relevance - redundancy) for all candidates
        candidate_relevance = np.array([relevance[candidate] for candidate in not_selected])
        mrmr_scores = candidate_relevance - redundancy
        
        # Select feature with highest MRMR score
        best_idx = np.argmax(mrmr_scores)
        next_feature = not_selected[best_idx]
        selected_local.append(next_feature)
        not_selected.remove(next_feature)
        
        if progress_logging and (i <= 3 or i % max(1, n_selected_features // 4) == 0):  # Log progress
            logger.debug(f"MRMR: Selected feature {i+1}/{n_selected_features}: {feature_mapping[next_feature]} "
                        f"(relevance: {relevance[next_feature]:.4f}, redundancy: {redundancy[best_idx]:.4f}, "
                        f"MRMR: {mrmr_scores[best_idx]:.4f})")
    
    # Map back to original feature indices
    selected_original = feature_mapping[selected_local]
    
    if progress_logging:
        logger.info(f"MRMR completed: selected {len(selected_original)} features from {len(feature_mapping)} candidates")
    return np.array(selected_original)

# Backward compatibility alias
def mrmr_feature_selection(*args, **kwargs):
    """Backward compatibility alias for simple_mrmr."""
    return simple_mrmr(*args, **kwargs) 