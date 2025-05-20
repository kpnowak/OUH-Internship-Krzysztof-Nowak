"""
Simplified implementation of MRMR (Maximum Relevance Minimum Redundancy)
feature selection algorithm to avoid dependency on scikit-feature.
"""

import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

def simple_mrmr(X, y, n_selected_features=10, is_regression=False):
    """
    Simplified MRMR implementation that selects features with maximum relevance
    to the target and minimum redundancy between selected features.
    
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
        
    Returns
    -------
    numpy.ndarray
        Indices of selected features
    """
    n_samples, n_features = X.shape
    
    # Cap n_selected_features to the number of available features
    n_selected_features = min(n_selected_features, n_features)
    
    # Calculate relevance (mutual information with target)
    relevance = mutual_info_regression(X, y) if is_regression else mutual_info_classif(X, y)
    
    # Initialize with the most relevant feature
    selected = [np.argmax(relevance)]
    not_selected = list(range(n_features))
    not_selected.remove(selected[0])
    
    # Select remaining features
    for i in range(1, n_selected_features):
        # Break if no more features to select
        if not not_selected:
            break
            
        # Calculate redundancy - average MI between each candidate and selected features
        redundancy = np.zeros(len(not_selected))
        
        # For each candidate feature
        for j, candidate in enumerate(not_selected):
            # Calculate average MI with already selected features
            mi_sum = 0
            for selected_idx in selected:
                # Reshape to make sklearn's mutual_info functions happy
                f1 = X[:, candidate].reshape(-1, 1)
                f2 = X[:, selected_idx].reshape(-1, 1)
                mi = mutual_info_regression(f1, f2.ravel())[0]
                mi_sum += mi
            
            # Average redundancy
            redundancy[j] = mi_sum / len(selected) if selected else 0
        
        # Compute MRMR criterion (relevance - redundancy) for all candidates
        mrmr_scores = np.array([relevance[candidate] for candidate in not_selected]) - redundancy
        
        # Select feature with highest MRMR score
        next_feature = not_selected[np.argmax(mrmr_scores)]
        selected.append(next_feature)
        not_selected.remove(next_feature)
    
    return np.array(selected) 