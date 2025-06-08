"""
Fast Feature Selection Algorithms for High-Dimensional Genomic Data

This module provides efficient alternatives to MRMR for feature selection
in cancer genomics datasets with thousands of features and limited samples.
Optimized for TCGA multi-omics data characteristics.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest, f_regression, f_classif, chi2, 
    VarianceThreshold, RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import ElasticNet, LogisticRegression, Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import pearsonr, spearmanr
import logging
from typing import Union, Tuple, Optional, List
import warnings

logger = logging.getLogger(__name__)

class FastFeatureSelector:
    """
    Fast feature selection algorithms optimized for high-dimensional genomic data.
    
    Provides multiple algorithms as alternatives to MRMR with significantly
    faster computation times while maintaining good performance on cancer genomics data.
    """
    
    def __init__(self, method: str = "variance_f_test", n_features: int = 100, 
                 random_state: int = 42, **kwargs):
        """
        Initialize the fast feature selector.
        
        Parameters
        ----------
        method : str
            Feature selection method to use. Options:
            - 'variance_f_test': Variance threshold + F-test (recommended)
            - 'rf_importance': Random Forest feature importance
            - 'elastic_net': Elastic Net regularization
            - 'rfe_linear': Recursive Feature Elimination with linear model
            - 'correlation': Correlation-based selection (regression only)
            - 'chi2': Chi-square test (classification only)
            - 'lasso': LASSO regularization
            - 'combined_fast': Multi-step fast selection
        n_features : int
            Number of features to select
        random_state : int
            Random state for reproducibility
        **kwargs : dict
            Additional parameters for specific methods
        """
        self.method = method
        self.n_features = n_features
        self.random_state = random_state
        self.kwargs = kwargs
        self.selector_ = None
        self.selected_features_ = None
        self.feature_scores_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, is_regression: bool = True) -> 'FastFeatureSelector':
        """
        Fit the feature selector to the data.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape (n_samples, n_features)
        y : np.ndarray
            Target vector, shape (n_samples,)
        is_regression : bool
            Whether this is a regression task
            
        Returns
        -------
        self : FastFeatureSelector
            Fitted selector
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Validate inputs
        if X.shape[0] != len(y):
            raise ValueError("X and y must have the same number of samples")
        
        # Cap n_features to available features
        self.n_features = min(self.n_features, X.shape[1])
        
        # Select method based on task type and method name
        if self.method == "variance_f_test":
            self._fit_variance_f_test(X, y, is_regression)
        elif self.method == "rf_importance":
            self._fit_rf_importance(X, y, is_regression)
        elif self.method == "elastic_net":
            self._fit_elastic_net(X, y, is_regression)
        elif self.method == "rfe_linear":
            self._fit_rfe_linear(X, y, is_regression)
        elif self.method == "correlation" and is_regression:
            self._fit_correlation(X, y)
        elif self.method == "chi2" and not is_regression:
            self._fit_chi2(X, y)
        elif self.method == "lasso":
            self._fit_lasso(X, y, is_regression)
        elif self.method == "combined_fast":
            self._fit_combined_fast(X, y, is_regression)
        else:
            # Fallback to appropriate univariate method
            if is_regression:
                self._fit_variance_f_test(X, y, is_regression)
            else:
                self._fit_variance_f_test(X, y, is_regression)
                
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the data using selected features.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix to transform
            
        Returns
        -------
        np.ndarray
            Transformed feature matrix with selected features only
        """
        if self.selected_features_ is None:
            raise ValueError("Selector must be fitted before transform")
            
        X = np.asarray(X)
        return X[:, self.selected_features_]
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray, is_regression: bool = True) -> np.ndarray:
        """
        Fit the selector and transform the data in one step.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        is_regression : bool
            Whether this is a regression task
            
        Returns
        -------
        np.ndarray
            Transformed feature matrix
        """
        return self.fit(X, y, is_regression).transform(X)
    
    def get_selected_features(self) -> np.ndarray:
        """Get indices of selected features."""
        if self.selected_features_ is None:
            raise ValueError("Selector must be fitted first")
        return self.selected_features_
    
    def get_feature_scores(self) -> Optional[np.ndarray]:
        """Get feature importance scores if available."""
        return self.feature_scores_
    
    def _fit_variance_f_test(self, X: np.ndarray, y: np.ndarray, is_regression: bool):
        """Variance threshold followed by F-test (recommended for speed + performance)."""
        logger.debug(f"FastFS: Using variance + F-test selection for {X.shape[1]} features")
        
        # Get configuration
        try:
            from config import FAST_FEATURE_SELECTION_CONFIG
            variance_threshold = self.kwargs.get('variance_threshold', 
                                               FAST_FEATURE_SELECTION_CONFIG.get('variance_threshold', 0.01))
        except ImportError:
            variance_threshold = self.kwargs.get('variance_threshold', 0.01)
        
        # Step 1: Remove low-variance features (very fast)
        var_selector = VarianceThreshold(threshold=variance_threshold)
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            X_var = var_selector.fit_transform(X)
        
        var_features = np.where(var_selector.get_support())[0]
        logger.debug(f"FastFS: Variance filter kept {X_var.shape[1]}/{X.shape[1]} features")
        
        # Step 2: F-test on remaining features
        if X_var.shape[1] <= self.n_features:
            # If we have fewer features than requested, use all
            self.selected_features_ = var_features
            self.feature_scores_ = np.ones(len(var_features))
        else:
            # Select top features using F-test
            if is_regression:
                f_selector = SelectKBest(f_regression, k=self.n_features)
            else:
                f_selector = SelectKBest(f_classif, k=self.n_features)
            
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                f_selector.fit(X_var, y)
            
            # Map back to original feature indices
            f_features = np.where(f_selector.get_support())[0]
            self.selected_features_ = var_features[f_features]
            self.feature_scores_ = f_selector.scores_[f_features]
        
        logger.debug(f"FastFS: Selected {len(self.selected_features_)} features using variance + F-test")
    
    def _fit_rf_importance(self, X: np.ndarray, y: np.ndarray, is_regression: bool):
        """Random Forest feature importance selection."""
        logger.debug(f"FastFS: Using Random Forest importance for {X.shape[1]} features")
        
        # Get configuration
        try:
            from config import FAST_FEATURE_SELECTION_CONFIG
            n_estimators = self.kwargs.get('n_estimators', 
                                         FAST_FEATURE_SELECTION_CONFIG.get('rf_n_estimators', 50))
            max_depth = self.kwargs.get('max_depth', 
                                      FAST_FEATURE_SELECTION_CONFIG.get('rf_max_depth', 10))
        except ImportError:
            n_estimators = self.kwargs.get('n_estimators', 50)
            max_depth = self.kwargs.get('max_depth', 10)
        
        if is_regression:
            rf = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=self.random_state,
                n_jobs=-1
            )
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            rf.fit(X, y)
        
        # Get feature importances and select top features
        importances = rf.feature_importances_
        top_indices = np.argsort(importances)[-self.n_features:]
        
        self.selected_features_ = top_indices
        self.feature_scores_ = importances[top_indices]
        
        logger.debug(f"FastFS: Selected {len(self.selected_features_)} features using RF importance")
    
    def _fit_elastic_net(self, X: np.ndarray, y: np.ndarray, is_regression: bool):
        """Elastic Net regularization for feature selection."""
        logger.debug(f"FastFS: Using Elastic Net for {X.shape[1]} features")
        
        # Scale features for regularization
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Set regularization parameters
        alpha = self.kwargs.get('alpha', 0.01)
        l1_ratio = self.kwargs.get('l1_ratio', 0.5)
        
        if is_regression:
            model = ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                random_state=self.random_state,
                max_iter=1000
            )
        else:
            model = LogisticRegression(
                penalty='elasticnet',
                C=1/alpha,  # sklearn uses C = 1/alpha
                l1_ratio=l1_ratio,
                random_state=self.random_state,
                solver='saga',
                max_iter=1000
            )
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            model.fit(X_scaled, y)
        
        # Get non-zero coefficients
        if hasattr(model, 'coef_'):
            if model.coef_.ndim > 1:
                coef = np.abs(model.coef_).max(axis=0)  # Multi-class case
            else:
                coef = np.abs(model.coef_)
        else:
            coef = np.abs(model.feature_importances_)
        
        # Select top features with non-zero coefficients
        non_zero_mask = coef > 1e-10
        if np.sum(non_zero_mask) >= self.n_features:
            # Select top n_features from non-zero coefficients
            non_zero_indices = np.where(non_zero_mask)[0]
            non_zero_coef = coef[non_zero_mask]
            top_indices = non_zero_indices[np.argsort(non_zero_coef)[-self.n_features:]]
        else:
            # If fewer non-zero features, select top overall
            top_indices = np.argsort(coef)[-self.n_features:]
        
        self.selected_features_ = top_indices
        self.feature_scores_ = coef[top_indices]
        
        logger.debug(f"FastFS: Selected {len(self.selected_features_)} features using Elastic Net")
    
    def _fit_rfe_linear(self, X: np.ndarray, y: np.ndarray, is_regression: bool):
        """Recursive Feature Elimination with linear model."""
        logger.debug(f"FastFS: Using RFE with linear model for {X.shape[1]} features")
        
        # Use simple linear models for speed
        if is_regression:
            from sklearn.linear_model import LinearRegression
            estimator = LinearRegression()
        else:
            estimator = LogisticRegression(
                random_state=self.random_state,
                solver='liblinear',
                max_iter=1000
            )
        
        # Use step size for faster elimination
        step = max(1, (X.shape[1] - self.n_features) // 10)
        
        rfe = RFE(
            estimator=estimator,
            n_features_to_select=self.n_features,
            step=step
        )
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            rfe.fit(X, y)
        
        self.selected_features_ = np.where(rfe.support_)[0]
        self.feature_scores_ = rfe.ranking_[self.selected_features_]
        
        logger.debug(f"FastFS: Selected {len(self.selected_features_)} features using RFE")
    
    def _fit_correlation(self, X: np.ndarray, y: np.ndarray):
        """Correlation-based selection for regression."""
        logger.debug(f"FastFS: Using correlation selection for {X.shape[1]} features")
        
        correlation_method = self.kwargs.get('correlation_method', 'pearson')
        
        correlations = np.zeros(X.shape[1])
        
        for i in range(X.shape[1]):
            try:
                if correlation_method == 'pearson':
                    corr, _ = pearsonr(X[:, i], y)
                else:  # spearman
                    corr, _ = spearmanr(X[:, i], y)
                correlations[i] = abs(corr) if not np.isnan(corr) else 0
            except:
                correlations[i] = 0
        
        # Select top correlated features
        top_indices = np.argsort(correlations)[-self.n_features:]
        
        self.selected_features_ = top_indices
        self.feature_scores_ = correlations[top_indices]
        
        logger.debug(f"FastFS: Selected {len(self.selected_features_)} features using correlation")
    
    def _fit_chi2(self, X: np.ndarray, y: np.ndarray):
        """Chi-square test for classification."""
        logger.debug(f"FastFS: Using Chi-square test for {X.shape[1]} features")
        
        # Ensure non-negative features for chi2
        X_pos = X.copy()
        X_pos = X_pos - X_pos.min(axis=0) + 1e-10
        
        chi2_selector = SelectKBest(chi2, k=self.n_features)
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            chi2_selector.fit(X_pos, y)
        
        self.selected_features_ = np.where(chi2_selector.get_support())[0]
        self.feature_scores_ = chi2_selector.scores_[self.selected_features_]
        
        logger.debug(f"FastFS: Selected {len(self.selected_features_)} features using Chi-square")
    
    def _fit_lasso(self, X: np.ndarray, y: np.ndarray, is_regression: bool):
        """LASSO regularization for feature selection."""
        logger.debug(f"FastFS: Using LASSO for {X.shape[1]} features")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        alpha = self.kwargs.get('alpha', 0.01)
        
        if is_regression:
            model = Lasso(alpha=alpha, random_state=self.random_state, max_iter=1000)
        else:
            model = LogisticRegression(
                penalty='l1',
                C=1/alpha,
                random_state=self.random_state,
                solver='liblinear',
                max_iter=1000
            )
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            model.fit(X_scaled, y)
        
        # Get non-zero coefficients
        if hasattr(model, 'coef_'):
            if model.coef_.ndim > 1:
                coef = np.abs(model.coef_).max(axis=0)
            else:
                coef = np.abs(model.coef_)
        else:
            coef = np.abs(model.feature_importances_)
        
        # Select features with largest absolute coefficients
        top_indices = np.argsort(coef)[-self.n_features:]
        
        self.selected_features_ = top_indices
        self.feature_scores_ = coef[top_indices]
        
        logger.debug(f"FastFS: Selected {len(self.selected_features_)} features using LASSO")
    
    def _fit_combined_fast(self, X: np.ndarray, y: np.ndarray, is_regression: bool):
        """Multi-step fast selection combining multiple methods."""
        logger.debug(f"FastFS: Using combined fast selection for {X.shape[1]} features")
        
        # Step 1: Variance threshold (remove constant/low-variance features)
        var_threshold = self.kwargs.get('variance_threshold', 0.01)
        var_selector = VarianceThreshold(threshold=var_threshold)
        X_var = var_selector.fit_transform(X)
        var_features = np.where(var_selector.get_support())[0]
        
        if X_var.shape[1] <= self.n_features:
            self.selected_features_ = var_features
            self.feature_scores_ = np.ones(len(var_features))
            return
        
        # Step 2: Fast univariate selection to reduce to ~3x target features
        intermediate_features = min(self.n_features * 3, X_var.shape[1])
        
        if is_regression:
            univariate_selector = SelectKBest(f_regression, k=intermediate_features)
        else:
            univariate_selector = SelectKBest(f_classif, k=intermediate_features)
        
        X_uni = univariate_selector.fit_transform(X_var, y)
        uni_features = np.where(univariate_selector.get_support())[0]
        
        # Step 3: Random Forest importance on reduced set
        if is_regression:
            rf = RandomForestRegressor(
                n_estimators=30,
                max_depth=8,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            rf = RandomForestClassifier(
                n_estimators=30,
                max_depth=8,
                random_state=self.random_state,
                n_jobs=-1
            )
        
        rf.fit(X_uni, y)
        importances = rf.feature_importances_
        
        # Select final features
        final_indices = np.argsort(importances)[-self.n_features:]
        
        # Map back to original indices
        self.selected_features_ = var_features[uni_features[final_indices]]
        self.feature_scores_ = importances[final_indices]
        
        logger.debug(f"FastFS: Selected {len(self.selected_features_)} features using combined method")


def get_fast_selector_recommendations(n_samples: int, n_features: int, 
                                    is_regression: bool) -> List[str]:
    """
    Get recommended fast feature selection methods based on data characteristics.
    
    Parameters
    ----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features
    is_regression : bool
        Whether this is a regression task
        
    Returns
    -------
    List[str]
        Recommended methods in order of preference
    """
    recommendations = []
    
    # For very high-dimensional data (typical in genomics)
    if n_features > 10000:
        recommendations.extend(['variance_f_test', 'combined_fast', 'rf_importance'])
    elif n_features > 1000:
        recommendations.extend(['variance_f_test', 'rf_importance', 'elastic_net'])
    else:
        recommendations.extend(['rf_importance', 'elastic_net', 'rfe_linear'])
    
    # Add task-specific methods
    if is_regression:
        recommendations.append('correlation')
    else:
        recommendations.append('chi2')
    
    # Always include LASSO as an option
    recommendations.append('lasso')
    
    return recommendations[:5]  # Return top 5 recommendations


# Convenience functions for integration with existing code
def fast_feature_selection_regression(X: np.ndarray, y: np.ndarray, 
                                    n_features: int = 100, 
                                    method: str = "variance_f_test") -> np.ndarray:
    """
    Fast feature selection for regression tasks.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    n_features : int
        Number of features to select
    method : str
        Selection method
        
    Returns
    -------
    np.ndarray
        Indices of selected features
    """
    selector = FastFeatureSelector(method=method, n_features=n_features)
    selector.fit(X, y, is_regression=True)
    return selector.get_selected_features()


def fast_feature_selection_classification(X: np.ndarray, y: np.ndarray, 
                                        n_features: int = 100, 
                                        method: str = "variance_f_test") -> np.ndarray:
    """
    Fast feature selection for classification tasks.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    n_features : int
        Number of features to select
    method : str
        Selection method
        
    Returns
    -------
    np.ndarray
        Indices of selected features
    """
    selector = FastFeatureSelector(method=method, n_features=n_features)
    selector.fit(X, y, is_regression=False)
    return selector.get_selected_features()