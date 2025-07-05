"""
Fast feature selection algorithms optimized for genomic data.

This module provides genomic-optimized feature selection that:
1. Uses much larger feature sets (hundreds to thousands)
2. Employs ensemble methods for robustness
3. Minimizes aggressive filtering that loses biological signal
4. Implements biological relevance scoring
"""

import numpy as np
import pandas as pd
import warnings
from typing import Union, List, Tuple, Optional, Dict
import logging
from sklearn.feature_selection import (
    SelectKBest, SelectFromModel, RFE, VarianceThreshold,
    f_regression, f_classif, mutual_info_regression, mutual_info_classif, chi2
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr, spearmanr
from scipy import stats

logger = logging.getLogger(__name__)

def calculate_mad_per_feature(X: np.ndarray) -> np.ndarray:
    """
    Calculate MAD (Median Absolute Deviation) for each feature.
    
    MAD is more robust to outliers than variance as it uses median instead of mean
    and absolute instead of squared deviations.
    
    Args:
        X: Input data (samples × features)
    
    Returns:
        Array of MAD values for each feature
    """
    mad_values = []
    for i in range(X.shape[1]):
        feature_data = X[:, i]
        # Handle NaN values
        valid_data = feature_data[~np.isnan(feature_data)]
        
        if len(valid_data) == 0:
            mad_values.append(0.0)
        elif len(valid_data) == 1:
            mad_values.append(0.0)
        else:
            median_val = np.median(valid_data)
            mad = np.median(np.abs(valid_data - median_val))
            # Scale by 1.4826 to make MAD equivalent to standard deviation for normal distributions
            mad_scaled = mad * 1.4826
            mad_values.append(mad_scaled)
    
    return np.array(mad_values)

class MADThreshold:
    """
    MAD-based feature selector, similar to VarianceThreshold but using MAD.
    
    MAD (Median Absolute Deviation) is more robust to outliers than variance.
    """
    
    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold
        self.mad_values_ = None
        self.support_mask_ = None
    
    def fit(self, X: np.ndarray, y=None) -> 'MADThreshold':
        """Fit the MAD threshold selector."""
        X = np.asarray(X)
        self.mad_values_ = calculate_mad_per_feature(X)
        self.support_mask_ = self.mad_values_ > self.threshold
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data by selecting features above MAD threshold."""
        if self.support_mask_ is None:
            raise ValueError("This MADThreshold instance is not fitted yet.")
        
        X = np.asarray(X)
        return X[:, self.support_mask_]
    
    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def get_support(self, indices: bool = False) -> Union[np.ndarray, List[int]]:
        """Get support mask or indices."""
        if self.support_mask_ is None:
            raise ValueError("This MADThreshold instance is not fitted yet.")
        
        if indices:
            return np.where(self.support_mask_)[0]
        else:
            return self.support_mask_

class GenomicFeatureSelector:
    """
    Genomic-optimized feature selection with ensemble methods and biological relevance.
    
    This selector is designed specifically for genomic data where:
    - We need hundreds to thousands of features
    - Biological signal is often weak and distributed
    - Traditional aggressive filtering loses important information
    """
    
    def __init__(self, method: str = "genomic_ensemble", n_features: int = 512, 
                 random_state: int = 42, **kwargs):
        """
        Initialize the genomic feature selector.
        
        Parameters
        ----------
        method : str
            Feature selection method:
            - 'genomic_ensemble': Multi-method ensemble (recommended)
            - 'biological_relevance': Biology-informed selection
            - 'permissive_univariate': Very permissive statistical selection
            - 'stability_selection': Stability-based selection
            - 'variance_f_test': Variance + F-test (fast)
        n_features : int
            Number of features to select (default: 512 for genomic data)
        random_state : int
            Random state for reproducibility
        **kwargs : dict
            Additional parameters
        """
        self.method = method
        self.n_features = max(n_features, 50)  # Minimum 50 features for genomic data
        self.random_state = random_state
        self.kwargs = kwargs
        self.selector_ = None
        self.selected_features_ = None
        self.feature_scores_ = None
        self.ensemble_scores_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, is_regression: bool = True) -> 'GenomicFeatureSelector':
        """
        Fit the genomic feature selector.
        
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
        self : GenomicFeatureSelector
            Fitted selector
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Validate inputs
        if X.shape[0] != len(y):
            raise ValueError("X and y must have the same number of samples")
        
        # Cap n_features to available features but be very permissive
        self.n_features = min(self.n_features, int(X.shape[1] * 0.95))  # Use up to 95% of features
        
        logger.info(f"GenomicFS: Selecting {self.n_features} from {X.shape[1]} features using {self.method}")
        
        # Select method
        if self.method == "genomic_ensemble":
            self._fit_genomic_ensemble(X, y, is_regression)
        elif self.method == "biological_relevance":
            self._fit_biological_relevance(X, y, is_regression)
        elif self.method == "permissive_univariate":
            self._fit_permissive_univariate(X, y, is_regression)
        elif self.method == "stability_selection":
            self._fit_stability_selection(X, y, is_regression)
        elif self.method == "variance_f_test":
            self._fit_variance_f_test(X, y, is_regression)
        elif self.method == "chi2":
            self._fit_chi2(X, y, is_regression)
        elif self.method == "fast_elastic_net":
            self._fit_fast_elastic_net(X, y, is_regression)
        else:
            # Fallback to genomic ensemble
            logger.warning(f"Unknown method {self.method}, using genomic_ensemble")
            self._fit_genomic_ensemble(X, y, is_regression)
                
        logger.info(f"GenomicFS: Selected {len(self.selected_features_)} features")
        return self
    
    def _fit_genomic_ensemble(self, X: np.ndarray, y: np.ndarray, is_regression: bool):
        """
        Ensemble method combining multiple genomic-appropriate selectors.
        """
        logger.debug("GenomicFS: Using ensemble method for genomic data")
        
        methods = []
        
        # Method 1: Very permissive univariate selection (top 80% of features)
        try:
            n_univariate = min(int(X.shape[1] * 0.8), self.n_features * 3)
            if is_regression:
                selector = SelectKBest(f_regression, k=n_univariate)
            else:
                selector = SelectKBest(f_classif, k=n_univariate)
            
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                selector.fit(X, y)
            
            univariate_scores = np.zeros(X.shape[1])
            selected_indices = selector.get_support(indices=True)
            # Ensure we only use scores for the selected features
            if len(selected_indices) == len(selector.scores_):
                univariate_scores[selected_indices] = selector.scores_
            else:
                # Handle mismatch by using only the available scores
                min_len = min(len(selected_indices), len(selector.scores_))
                univariate_scores[selected_indices[:min_len]] = selector.scores_[:min_len]
            methods.append(('univariate', univariate_scores))
            
        except Exception as e:
            logger.warning(f"Univariate selection failed: {e}")
        
        # Method 2: Mutual information (captures non-linear relationships)
        try:
            if is_regression:
                mi_scores = mutual_info_regression(X, y, random_state=self.random_state)
            else:
                mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
            methods.append(('mutual_info', mi_scores))
            
        except Exception as e:
            logger.warning(f"Mutual information failed: {e}")
        
        # Method 3: Random Forest importance (captures feature interactions)
        try:
            if is_regression:
                rf = RandomForestRegressor(
                    n_estimators=100, max_depth=10, min_samples_split=5,
                    random_state=self.random_state, n_jobs=-1
                )
            else:
                rf = RandomForestClassifier(
                    n_estimators=100, max_depth=10, min_samples_split=5,
                    random_state=self.random_state, n_jobs=-1
                )
            
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                rf.fit(X, y)
            
            rf_scores = rf.feature_importances_
            methods.append(('random_forest', rf_scores))
            
        except Exception as e:
            logger.warning(f"Random Forest importance failed: {e}")
        
        # Method 4: Minimal regularization (very permissive)
        try:
            if is_regression:
                model = ElasticNet(alpha=0.2, l1_ratio=0.5, max_iter=1000, random_state=self.random_state)  # Stricter alpha regularization
            else:
                model = LogisticRegression(
                    C=10000.0, penalty='l1', solver='liblinear', 
                    max_iter=1000, random_state=self.random_state
                )
            
            # Scale features for regularization
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                model.fit(X_scaled, y)
            
            if hasattr(model, 'coef_'):
                if model.coef_.ndim > 1:
                    reg_scores = np.abs(model.coef_).max(axis=0)
                else:
                    reg_scores = np.abs(model.coef_)
            else:
                reg_scores = np.abs(model.feature_importances_)
            
            methods.append(('regularization', reg_scores))
            
        except Exception as e:
            logger.warning(f"Regularization method failed: {e}")
        
        # Method 5: Correlation-based (for regression)
        if is_regression:
            try:
                correlations = np.zeros(X.shape[1])
                for i in range(X.shape[1]):
                    try:
                        corr, _ = spearmanr(X[:, i], y)
                        correlations[i] = abs(corr) if not np.isnan(corr) else 0
                    except:
                        correlations[i] = 0
                
                methods.append(('correlation', correlations))
                
            except Exception as e:
                logger.warning(f"Correlation method failed: {e}")
        
        # Combine methods using rank aggregation
        if not methods:
            logger.error("All feature selection methods failed, using variance fallback")
            self._fit_variance_fallback(X)
            return
        
        # Rank aggregation with genomic-appropriate weighting
        combined_scores = self._combine_genomic_scores(methods, X.shape[1])
        
        # Select top features
        top_indices = np.argsort(combined_scores)[-self.n_features:]
        
        self.selected_features_ = top_indices
        self.feature_scores_ = combined_scores[top_indices]
        self.ensemble_scores_ = {name: scores for name, scores in methods}
        
        logger.info(f"GenomicFS: Ensemble used {len(methods)} methods")
    
    def _fit_biological_relevance(self, X: np.ndarray, y: np.ndarray, is_regression: bool):
        """
        Biology-informed feature selection considering genomic characteristics.
        """
        logger.debug("GenomicFS: Using biological relevance scoring")
        
        # Start with statistical significance
        if is_regression:
            selector = SelectKBest(f_regression, k=min(self.n_features * 5, X.shape[1]))
        else:
            selector = SelectKBest(f_classif, k=min(self.n_features * 5, X.shape[1]))
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            selector.fit(X, y)
        
        base_scores = selector.scores_
        
        # Add biological relevance factors
        bio_scores = np.copy(base_scores)
        
        # Factor 1: MAD stability (prefer features with consistent MAD - more robust than variance)
        mad_values = calculate_mad_per_feature(X)
        mad_stability = 1.0 / (1.0 + np.abs(mad_values - np.median(mad_values)))
        bio_scores *= (1.0 + 0.2 * mad_stability)
        
        # Factor 2: Non-zero expression (prefer features with more non-zero values)
        non_zero_ratio = np.mean(X != 0, axis=0)
        bio_scores *= (1.0 + 0.3 * non_zero_ratio)
        
        # Factor 3: Dynamic range (prefer features with good dynamic range)
        ranges = np.ptp(X, axis=0)  # peak-to-peak
        normalized_ranges = ranges / (np.max(ranges) + 1e-10)
        bio_scores *= (1.0 + 0.2 * normalized_ranges)
        
        # Select top features
        top_indices = np.argsort(bio_scores)[-self.n_features:]
        
        self.selected_features_ = top_indices
        self.feature_scores_ = bio_scores[top_indices]
        
        logger.info("GenomicFS: Applied biological relevance scoring")
    
    def _fit_permissive_univariate(self, X: np.ndarray, y: np.ndarray, is_regression: bool):
        """
        Very permissive univariate selection for genomic data.
        """
        logger.debug("GenomicFS: Using permissive univariate selection")
        
        # Use very low MAD threshold (more robust than variance)
        mad_threshold = self.kwargs.get('mad_threshold', 1e-6)
        mad_selector = MADThreshold(threshold=mad_threshold)
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            X_mad = mad_selector.fit_transform(X)
        
        mad_features = np.where(mad_selector.get_support())[0]
        
        if X_mad.shape[1] <= self.n_features:
            self.selected_features_ = mad_features
            self.feature_scores_ = np.ones(len(mad_features))
            return
        
        # Very permissive statistical selection
        if is_regression:
            selector = SelectKBest(f_regression, k=self.n_features)
        else:
            selector = SelectKBest(f_classif, k=self.n_features)
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            selector.fit(X_mad, y)
        
        # Map back to original indices
        selected_mad_indices = np.where(selector.get_support())[0]
        self.selected_features_ = mad_features[selected_mad_indices]
        self.feature_scores_ = selector.scores_
        
        logger.info("GenomicFS: Applied permissive univariate selection")
    
    def _fit_stability_selection(self, X: np.ndarray, y: np.ndarray, is_regression: bool):
        """
        Stability selection for robust feature selection.
        """
        logger.debug("GenomicFS: Using stability selection")
        
        n_bootstrap = self.kwargs.get('n_bootstrap', 50)
        selection_threshold = self.kwargs.get('selection_threshold', 0.3)  # Very permissive
        
        n_samples, n_features_total = X.shape
        selection_counts = np.zeros(n_features_total)
        
        # Bootstrap sampling with feature selection
        for i in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Feature selection on bootstrap sample
            try:
                if is_regression:
                    selector = SelectKBest(f_regression, k=min(self.n_features * 2, n_features_total))
                else:
                    selector = SelectKBest(f_classif, k=min(self.n_features * 2, n_features_total))
                
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    selector.fit(X_boot, y_boot)
                
                selected = selector.get_support(indices=True)
                selection_counts[selected] += 1
                
            except Exception as e:
                logger.warning(f"Bootstrap iteration {i} failed: {e}")
                continue
        
        # Select features that appear frequently
        selection_frequency = selection_counts / n_bootstrap
        stable_features = np.where(selection_frequency >= selection_threshold)[0]
        
        if len(stable_features) >= self.n_features:
            # Select top features by frequency
            top_indices = stable_features[np.argsort(selection_frequency[stable_features])[-self.n_features:]]
        else:
            # If not enough stable features, add more based on frequency
            all_indices = np.argsort(selection_frequency)[-self.n_features:]
            top_indices = all_indices
        
        self.selected_features_ = top_indices
        self.feature_scores_ = selection_frequency[top_indices]
        
        logger.info(f"GenomicFS: Stability selection with {len(stable_features)} stable features")
    
    def _fit_variance_f_test(self, X: np.ndarray, y: np.ndarray, is_regression: bool):
        """
        Fast MAD + F-test selection with genomic optimization (more robust than variance).
        """
        logger.debug("GenomicFS: Using MAD + F-test selection")
        
        # Very permissive MAD threshold (more robust than variance)
        mad_threshold = self.kwargs.get('mad_threshold', 1e-6)
        mad_selector = MADThreshold(threshold=mad_threshold)
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            X_mad = mad_selector.fit_transform(X)
        
        mad_features = np.where(mad_selector.get_support())[0]
        
        if X_mad.shape[1] <= self.n_features:
            self.selected_features_ = mad_features
            self.feature_scores_ = np.ones(len(mad_features))
            return
        
        # F-test selection
        if is_regression:
            f_selector = SelectKBest(f_regression, k=self.n_features)
        else:
            f_selector = SelectKBest(f_classif, k=self.n_features)
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            f_selector.fit(X_mad, y)
        
        # Map back to original indices
        f_features = np.where(f_selector.get_support())[0]
        self.selected_features_ = mad_features[f_features]
        self.feature_scores_ = f_selector.scores_
        
        logger.info("GenomicFS: Applied MAD + F-test selection")
    
    def _fit_chi2(self, X: np.ndarray, y: np.ndarray, is_regression: bool):
        """
        Chi-squared feature selection for classification tasks.
        """
        logger.debug("GenomicFS: Using Chi-squared selection")
        
        if is_regression:
            # Chi2 is not applicable for regression, fall back to F-test
            logger.warning("Chi2 not applicable for regression, using F-test instead")
            self._fit_variance_f_test(X, y, is_regression)
            return
        
        # Ensure non-negative values for chi2 (required by chi2 test)
        X_nonneg = X.copy()
        
        # Handle negative values by shifting to non-negative range
        min_vals = np.min(X_nonneg, axis=0)
        negative_features = min_vals < 0
        if np.any(negative_features):
            logger.debug(f"GenomicFS: Shifting {np.sum(negative_features)} features to non-negative range for chi2")
            X_nonneg[:, negative_features] -= min_vals[negative_features]
        
        # Apply MAD threshold first to remove constant features (more robust than variance)
        mad_threshold = self.kwargs.get('mad_threshold', 1e-6)
        mad_selector = MADThreshold(threshold=mad_threshold)
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            X_mad = mad_selector.fit_transform(X_nonneg)
        
        mad_features = np.where(mad_selector.get_support())[0]
        
        if X_mad.shape[1] <= self.n_features:
            self.selected_features_ = mad_features
            self.feature_scores_ = np.ones(len(mad_features))
            logger.info(f"GenomicFS: Chi2 selected {len(mad_features)} features (all after MAD filtering)")
            return
        
        # Apply chi2 selection
        try:
            from sklearn.feature_selection import SelectKBest, chi2
            
            chi2_selector = SelectKBest(chi2, k=self.n_features)
            
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                chi2_selector.fit(X_mad, y)
            
            # Map back to original indices
            chi2_features = np.where(chi2_selector.get_support())[0]
            self.selected_features_ = mad_features[chi2_features]
            self.feature_scores_ = chi2_selector.scores_
            
            logger.info(f"GenomicFS: Chi2 selected {len(self.selected_features_)} features")
            
        except Exception as e:
            logger.warning(f"Chi2 selection failed: {e}, falling back to F-test")
            # Fallback to F-test if chi2 fails
            self._fit_variance_f_test(X, y, is_regression)
    
    def _fit_fast_elastic_net(self, X: np.ndarray, y: np.ndarray, is_regression: bool):
        """
        Fast ElasticNet-based feature selection optimized for speed.
        """
        logger.debug("GenomicFS: Using fast ElasticNet selection")
        
        # Step 1: Quick MAD filtering to remove constant features (more robust than variance)
        mad_threshold = 1e-6
        mad_selector = MADThreshold(threshold=mad_threshold)
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            X_mad = mad_selector.fit_transform(X)
        
        mad_features = np.where(mad_selector.get_support())[0]
        
        if X_mad.shape[1] <= self.n_features:
            self.selected_features_ = mad_features
            self.feature_scores_ = np.ones(len(mad_features))
            logger.info(f"GenomicFS: Fast ElasticNet selected {len(mad_features)} features (all after MAD filtering)")
            return
        
        # Step 2: Fast univariate pre-filtering to reduce dimensionality
        # Select top 3x target features using F-test for speed
        prefilter_k = min(self.n_features * 3, X_mad.shape[1])
        
        try:
            if is_regression:
                prefilter_selector = SelectKBest(f_regression, k=prefilter_k)
            else:
                prefilter_selector = SelectKBest(f_classif, k=prefilter_k)
            
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                X_prefiltered = prefilter_selector.fit_transform(X_mad, y)
            
            prefilter_features = np.where(prefilter_selector.get_support())[0]
            
        except Exception as e:
            logger.warning(f"Pre-filtering failed: {e}, using all features")
            X_prefiltered = X_mad
            prefilter_features = np.arange(X_mad.shape[1])
        
        # Step 3: Fast ElasticNet with optimized parameters
        try:
            # Scale features for regularization
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_prefiltered)
            
            if is_regression:
                # Use very light regularization for speed
                model = ElasticNet(
                    alpha=0.2,        # Stricter regularization parameter
                    l1_ratio=0.5,     # Balanced L1/L2 regularization for feature selection
                    max_iter=500,     # Reduced iterations for speed
                    random_state=self.random_state,
                    selection='random'  # Random coordinate descent for speed
                )
            else:
                # Use LogisticRegression with L1 penalty
                model = LogisticRegression(
                    C=100.0,          # Light regularization (high C)
                    penalty='l1',
                    solver='liblinear',  # Fast solver for L1
                    max_iter=500,     # Reduced iterations for speed
                    random_state=self.random_state
                )
            
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                model.fit(X_scaled, y)
            
            # Get feature importance scores
            if hasattr(model, 'coef_'):
                if model.coef_.ndim > 1:
                    importance_scores = np.abs(model.coef_).max(axis=0)
                else:
                    importance_scores = np.abs(model.coef_)
            else:
                importance_scores = np.abs(model.feature_importances_)
            
            # Select top features
            top_prefilter_indices = np.argsort(importance_scores)[-self.n_features:]
            
            # Map back to original indices
            selected_mad_indices = prefilter_features[top_prefilter_indices]
            self.selected_features_ = mad_features[selected_mad_indices]
            self.feature_scores_ = importance_scores[top_prefilter_indices]
            
            logger.info(f"GenomicFS: Fast ElasticNet selected {len(self.selected_features_)} features")
            
        except Exception as e:
            logger.warning(f"Fast ElasticNet failed: {e}, falling back to F-test")
            # Fallback to simple F-test selection
            try:
                if is_regression:
                    fallback_selector = SelectKBest(f_regression, k=self.n_features)
                else:
                    fallback_selector = SelectKBest(f_classif, k=self.n_features)
                
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    fallback_selector.fit(X_mad, y)
                
                fallback_features = np.where(fallback_selector.get_support())[0]
                self.selected_features_ = mad_features[fallback_features]
                self.feature_scores_ = fallback_selector.scores_
                
                logger.info(f"GenomicFS: Fallback F-test selected {len(self.selected_features_)} features")
                
            except Exception as fallback_e:
                logger.error(f"Fallback also failed: {fallback_e}, using MAD-only selection")
                self._fit_mad_fallback(X)
    
    def _combine_genomic_scores(self, methods: List[Tuple[str, np.ndarray]], n_features: int) -> np.ndarray:
        """
        Combine scores from multiple methods using genomic-appropriate weighting.
        """
        if not methods:
            return np.zeros(n_features)
        
        # Normalize all scores to [0, 1]
        normalized_scores = []
        weights = []
        
        for method_name, scores in methods:
            # Handle NaN and infinite values
            scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Normalize to [0, 1]
            if np.max(scores) > np.min(scores):
                norm_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
            else:
                norm_scores = np.ones_like(scores) * 0.5
            
            normalized_scores.append(norm_scores)
            
            # Assign weights based on method reliability for genomic data
            if method_name == 'univariate':
                weights.append(0.3)  # High weight for statistical significance
            elif method_name == 'mutual_info':
                weights.append(0.25)  # Good for non-linear relationships
            elif method_name == 'random_forest':
                weights.append(0.2)  # Good for interactions
            elif method_name == 'regularization':
                weights.append(0.15)  # Lower weight due to potential over-regularization
            elif method_name == 'correlation':
                weights.append(0.1)  # Lowest weight for simple correlation
            else:
                weights.append(0.1)  # Default weight
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Combine scores
        combined = np.zeros(n_features)
        for i, (norm_scores, weight) in enumerate(zip(normalized_scores, weights)):
            combined += weight * norm_scores
        
        return combined
    
    def _fit_mad_fallback(self, X: np.ndarray):
        """
        Fallback method using only MAD (more robust than variance).
        """
        logger.warning("GenomicFS: Using MAD fallback method")
        
        mad_values = calculate_mad_per_feature(X)
        top_indices = np.argsort(mad_values)[-self.n_features:]
        
        self.selected_features_ = top_indices
        self.feature_scores_ = mad_values[top_indices]
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using selected features.
        
        Parameters
        ----------
        X : np.ndarray
            Input data
            
        Returns
        -------
        np.ndarray
            Transformed data with selected features
        """
        if self.selected_features_ is None:
            raise ValueError("Selector has not been fitted yet")
        
        return X[:, self.selected_features_]
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray, is_regression: bool = True) -> np.ndarray:
        """
        Fit selector and transform data.
        
        Parameters
        ----------
        X : np.ndarray
            Input data
        y : np.ndarray
            Target values
        is_regression : bool
            Whether this is regression
            
        Returns
        -------
        np.ndarray
            Transformed data with selected features
        """
        return self.fit(X, y, is_regression).transform(X)
    
    def get_selected_features(self) -> np.ndarray:
        """
        Get indices of selected features.
        
        Returns
        -------
        np.ndarray
            Indices of selected features
        """
        if self.selected_features_ is None:
            raise ValueError("Selector has not been fitted yet")
        
        return self.selected_features_
    
    def get_feature_scores(self) -> np.ndarray:
        """
        Get scores of selected features.
        
        Returns
        -------
        np.ndarray
            Scores of selected features
        """
        if self.feature_scores_ is None:
            raise ValueError("Selector has not been fitted yet")
        
        return self.feature_scores_
    
    def get_support(self, indices: bool = False) -> np.ndarray:
        """
        Get support mask or indices.
        
        Parameters
        ----------
        indices : bool
            Whether to return indices instead of mask
            
        Returns
        -------
        np.ndarray
            Support mask or indices
        """
        if self.selected_features_ is None:
            raise ValueError("Selector has not been fitted yet")
        
        if indices:
            return self.selected_features_
        else:
            # Create boolean mask (assuming we know the total number of features)
            # This is a limitation - we need to store the original feature count
            mask = np.zeros(np.max(self.selected_features_) + 1, dtype=bool)
            mask[self.selected_features_] = True
            return mask

# Convenience functions for backward compatibility

def fast_feature_selection_regression(X: np.ndarray, y: np.ndarray, 
                                    n_features: int = 512, 
                                    method: str = "genomic_ensemble") -> np.ndarray:
    """
    Fast feature selection for regression tasks optimized for genomic data.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    n_features : int
        Number of features to select (default: 512 for genomic data)
    method : str
        Selection method
        
    Returns
    -------
    np.ndarray
        Indices of selected features
    """
    selector = GenomicFeatureSelector(method=method, n_features=n_features)
    selector.fit(X, y, is_regression=True)
    return selector.get_selected_features()

def fast_feature_selection_classification(X: np.ndarray, y: np.ndarray, 
                                        n_features: int = 512, 
                                        method: str = "genomic_ensemble") -> np.ndarray:
    """
    Fast feature selection for classification tasks optimized for genomic data.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    n_features : int
        Number of features to select (default: 512 for genomic data)
    method : str
        Selection method
        
    Returns
    -------
    np.ndarray
        Indices of selected features
    """
    selector = GenomicFeatureSelector(method=method, n_features=n_features)
    selector.fit(X, y, is_regression=False)
    return selector.get_selected_features()

# Legacy compatibility
FastFeatureSelector = GenomicFeatureSelector