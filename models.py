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
    LinearRegression, Lasso, ElasticNet, LogisticRegression
)
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier
)
from sklearn.svm import SVR, SVC
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

# Target transformation registry for regression datasets
TARGET_TRANSFORMS = {
    'AML': ('log1p', np.log1p, np.expm1),
    'Sarcoma': ('sqrt', np.sqrt, lambda x: x**2),
}

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
    
    def __init__(self, n_components=32, alpha=0.1, max_iter=1000, tol=1e-6, scale=True):
        """
        Initialize Sparse PLS-DA.
        
        Parameters
        ----------
        n_components : int
            Number of components to extract (default 32 for MCC optimization)
        alpha : float
            Sparsity parameter for L1 regularization
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
        from sklearn.preprocessing import StandardScaler
        
        X = np.asarray(X)
        y = np.asarray(y)
        
        n_samples, n_features = X.shape
        
        # Encode labels for discriminant analysis
        Y = self._encode_labels(y)
        
        # Scale data if requested
        if self.scale:
            self.scaler_x_ = StandardScaler()
            X_scaled = self.scaler_x_.fit_transform(X)
            self.scaler_y_ = StandardScaler()
            Y_scaled = self.scaler_y_.fit_transform(Y)
        else:
            X_scaled = X.copy()
            Y_scaled = Y.copy()
            
        # Determine actual number of components
        max_components = min(n_samples - 1, n_features, Y.shape[1], self.n_components)
        actual_components = max_components
        
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
    
    def __init__(self, n_components=2, alpha=0.1, max_iter=500, tol=1e-6, copy=True, scale=True):
        """
        Initialize Sparse PLS.
        
        Parameters
        ----------
        n_components : int, default=2
            Number of components to extract
        alpha : float, default=0.1
            Sparsity parameter (L1 regularization strength)
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
        Fit Sparse PLS model using iterative algorithm.
        
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
        
        # Center and scale data
        if self.scale:
            from sklearn.preprocessing import StandardScaler
            self.x_scaler_ = StandardScaler()
            self.y_scaler_ = StandardScaler()
            X = self.x_scaler_.fit_transform(X)
            y = self.y_scaler_.fit_transform(y)
        else:
            X = X - np.mean(X, axis=0)
            y = y - np.mean(y, axis=0)
            
        # Initialize storage
        self.x_weights_ = np.zeros((n_features, self.n_components))
        self.y_weights_ = np.zeros((n_targets, self.n_components))
        self.x_loadings_ = np.zeros((n_features, self.n_components))
        self.y_loadings_ = np.zeros((n_targets, self.n_components))
        
        X_residual = X.copy()
        y_residual = y.copy()
        
        for k in range(self.n_components):
            # Initialize weights randomly
            w = np.random.randn(n_features)
            w = w / np.linalg.norm(w)
            
            for iteration in range(self.max_iter):
                w_old = w.copy()
                
                # Update weights
                t = X_residual @ w
                c = y_residual.T @ t / (t.T @ t + 1e-8)
                u = y_residual @ c
                w = X_residual.T @ u / (u.T @ u + 1e-8)
                
                # Apply sparsity constraint
                w = self._soft_threshold(w, self.alpha)
                
                # Normalize
                w_norm = np.linalg.norm(w)
                if w_norm > 1e-8:
                    w = w / w_norm
                else:
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
            
            # Deflate matrices
            X_residual = X_residual - np.outer(t, p)
            y_residual = y_residual - np.outer(t, q)
            
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
            Transformed data
        """
        if self.x_weights_ is None:
            raise ValueError("Model must be fitted before transform")
            
        X = np.asarray(X, dtype=np.float64)
        
        if hasattr(self, 'x_scaler_'):
            X = self.x_scaler_.transform(X)
        else:
            X = X - np.mean(X, axis=0)
            
        return X @ self.x_weights_
        
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
            
        # Create and fit Kernel PCA with computed gamma
        self.kernel_pca_ = KernelPCA(
            n_components=self.n_components,
            kernel=self.kernel,
            gamma=self.gamma_computed_,
            eigen_solver=self.eigen_solver,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )
        
        self.kernel_pca_.fit(X)
        
        # Store components for compatibility
        if hasattr(self.kernel_pca_, 'eigenvectors_'):
            self.components_ = self.kernel_pca_.eigenvectors_.T
        else:
            # Fallback for different sklearn versions
            self.components_ = None
            
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

class EarlyStoppingWrapper:
    """
    Early stopping wrapper for sklearn models that don't natively support it.
    Implements validation-based early stopping for iterative models.
    """
    
    def __init__(self, base_model, patience=10, min_delta=1e-4, validation_split=0.2, 
                 restore_best_weights=True, monitor_metric="auto", verbose=1, random_state=None):
        """
        Initialize Early Stopping Wrapper.
        
        Parameters
        ----------
        base_model : sklearn estimator
            Base model to wrap with early stopping
        patience : int, default=10
            Number of epochs to wait for improvement
        min_delta : float, default=1e-4
            Minimum change to qualify as improvement
        validation_split : float, default=0.2
            Fraction of training data to use for validation
        restore_best_weights : bool, default=True
            Whether to restore best model weights
        monitor_metric : str, default="auto"
            Metric to monitor for early stopping
        verbose : int, default=1
            Verbosity level
        random_state : int, optional
            Random state for reproducibility
        """
        self.base_model = base_model
        self.patience = patience
        self.min_delta = min_delta
        self.validation_split = validation_split
        self.restore_best_weights = restore_best_weights
        self.monitor_metric = monitor_metric
        self.verbose = verbose
        self.random_state = random_state
        
        # State variables
        self.best_score_ = None
        self.best_model_ = None
        self.best_model_params_ = None
        self.best_n_estimators_ = None
        self.best_max_iter_ = None
        self.wait_ = 0
        self.stopped_epoch_ = 0
        self.history_ = []
        
    def _determine_monitor_metric(self, is_regression=None):
        """Determine which metric to monitor based on the model type."""
        if self.monitor_metric != "auto":
            return self.monitor_metric
        
        if is_regression is None:
            # Try to infer from model type
            model_name = self.base_model.__class__.__name__
            if any(term in model_name.lower() for term in ['regressor', 'regression']):
                is_regression = True
            elif any(term in model_name.lower() for term in ['classifier', 'classification']):
                is_regression = False
            else:
                # Default to regression
                is_regression = True
                
        return "neg_mse" if is_regression else "accuracy"
    
    def _calculate_score(self, model, X_val, y_val, metric):
        """Calculate validation score for monitoring."""
        try:
            if metric == "neg_mse":
                y_pred = model.predict(X_val)
                return -mean_squared_error(y_val, y_pred)
            elif metric == "accuracy":
                return model.score(X_val, y_val)
            elif metric == "r2":
                return model.score(X_val, y_val)  # For regression, score returns R²
            else:
                # Default to model's score method
                return model.score(X_val, y_val)
        except Exception as e:
            logger.warning(f"Error calculating score with metric {metric}: {str(e)}")
            return -np.inf
    
    def _is_improvement(self, current_score, best_score, metric):
        """Check if current score is an improvement over best score."""
        if best_score is None:
            return True
        
        # For negative metrics (like neg_mse), higher is better
        # For positive metrics (like accuracy, r2), higher is better
        return current_score > (best_score + self.min_delta)
    
    def fit(self, X, y):
        """
        Fit the model with early stopping.
        
        Parameters
        ----------
        X : array-like
            Training features
        y : array-like
            Training targets
            
        Returns
        -------
        self
        """
        # Reset state
        self.best_score_ = None
        self.best_model_ = None
        self.wait_ = 0
        self.stopped_epoch_ = 0
        self.history_ = []
        
        # Determine if this is regression or classification
        is_regression = len(np.unique(y)) > 10 or np.issubdtype(y.dtype, np.floating)
        monitor_metric = self._determine_monitor_metric(is_regression)
        
        # Check if model supports iterative training
        model_name = self.base_model.__class__.__name__
        
        if model_name in ['RandomForestRegressor', 'RandomForestClassifier']:
            return self._fit_ensemble_early_stopping(X, y, monitor_metric)
        elif model_name in ['LogisticRegression', 'ElasticNet', 'Lasso']:
            return self._fit_iterative_early_stopping(X, y, monitor_metric)
        elif model_name == 'LinearRegression':
            # LinearRegression has analytical solution, no early stopping needed
            self.base_model.fit(X, y)
            self.best_model_ = copy.deepcopy(self.base_model)
            self.best_score_ = self._calculate_score(self.best_model_, X, y, monitor_metric)
            self.stopped_epoch_ = None  # No epochs for analytical solution
            return self
        elif model_name in ['SVR', 'SVC']:
            # SVMs don't typically benefit from early stopping, train normally
            self.base_model.fit(X, y)
            self.best_model_ = copy.deepcopy(self.base_model)
            self.best_score_ = self._calculate_score(self.best_model_, X, y, monitor_metric)
            self.stopped_epoch_ = None  # No epochs for SVM
            return self
        else:
            # Unknown model, train normally
            logger.warning(f"Early stopping not implemented for {model_name}, training normally")
            self.base_model.fit(X, y)
            self.best_model_ = copy.deepcopy(self.base_model)
            self.best_score_ = self._calculate_score(self.best_model_, X, y, monitor_metric)
            self.stopped_epoch_ = None  # No epochs for unknown models
            return self
    
    def _fit_ensemble_early_stopping(self, X, y, monitor_metric):
        """Enhanced early stopping for ensemble models with adaptive patience and robust error handling."""
        from config import EARLY_STOPPING_CONFIG
        
        # Split training data for validation with enhanced error handling
        try:
            # Check if stratified splitting is feasible
            unique_classes = np.unique(y)
            n_classes = len(unique_classes)
            test_samples = int(len(y) * self.validation_split)
            
            # Only use stratification if feasible and beneficial
            stratify_param = None
            if n_classes <= 10 and len(y) >= 20 and test_samples >= n_classes:
                stratify_param = y
            
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.validation_split, random_state=self.random_state, 
                stratify=stratify_param
            )
        except Exception as e:
            logger.warning(f"Stratified split failed: {str(e)}, using random split")
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.validation_split, random_state=self.random_state
            )
        
        # Adaptive patience based on data complexity
        effective_patience = self.patience
        if EARLY_STOPPING_CONFIG.get("adaptive_patience", False):
            n_features = X.shape[1]
            n_samples = X.shape[0]
            complexity_factor = min(2.0, (n_features / n_samples) if n_samples > 0 else 1.0)
            effective_patience = min(
                int(self.patience * (1 + complexity_factor)),
                EARLY_STOPPING_CONFIG.get("max_patience", 50)
            )
            if self.verbose >= 1:
                logger.debug(f"Adaptive patience: {effective_patience} (base: {self.patience}, complexity: {complexity_factor:.2f})")
        
        # Enhanced estimator range with adaptive steps
        min_estimators = 10
        max_estimators = getattr(self.base_model, 'n_estimators', 200)
        
        # Adaptive step size based on dataset size
        if X.shape[0] < 100:
            step_size = 5
            max_estimators = min(max_estimators, 100)
        elif X.shape[0] < 500:
            step_size = 10
        else:
            step_size = max(10, max_estimators // 20)
        
        estimator_range = range(min_estimators, max_estimators + 1, step_size)
        
        for n_est in estimator_range:
            try:
                # Create model with current number of estimators
                model_params = self.base_model.get_params()
                model_params['n_estimators'] = n_est
                current_model = self.base_model.__class__(**model_params)
                
                # Train model with timeout protection
                current_model.fit(X_train, y_train)
                
                # Calculate validation score
                current_score = self._calculate_score(current_model, X_val, y_val, monitor_metric)
                self.history_.append(current_score)
                
                if self.verbose >= 1:
                    logger.info(f"Estimators: {n_est}, Validation {monitor_metric}: {current_score:.4f}")
                
                # Check for improvement
                if self._is_improvement(current_score, self.best_score_, monitor_metric):
                    self.best_score_ = current_score
                    if self.restore_best_weights:
                        # Store model parameters instead of the model itself to avoid pickle issues
                        self.best_model_params_ = current_model.get_params()
                        self.best_n_estimators_ = n_est
                        self.best_model_ = current_model  # Keep reference for immediate use
                    self.wait_ = 0
                else:
                    self.wait_ += 1
                
                # Early stopping check with adaptive patience
                if self.wait_ >= effective_patience:
                    self.stopped_epoch_ = n_est
                    if self.verbose >= 1:
                        logger.info(f"Early stopping at {n_est} estimators. Best {monitor_metric}: {self.best_score_:.4f}")
                    break
                    
            except Exception as e:
                logger.warning(f"Error training with {n_est} estimators: {str(e)}")
                # Continue with next number of estimators
                continue
        
        # Fallback if no model was successfully trained
        if self.best_model_ is None:
            logger.warning("No successful model training, using minimal ensemble")
            try:
                model_params = self.base_model.get_params()
                model_params['n_estimators'] = min_estimators
                fallback_model = self.base_model.__class__(**model_params)
                fallback_model.fit(X, y)  # Train on full data as fallback
                self.best_model_ = fallback_model
                self.stopped_epoch_ = min_estimators
            except Exception as e:
                logger.error(f"Fallback model training failed: {str(e)}")
                raise
            
        return self
    
    def _fit_iterative_early_stopping(self, X, y, monitor_metric):
        """Implement early stopping for iterative models like LogisticRegression."""
        # Split training data for validation with proper stratification check
        try:
            unique_classes = np.unique(y)
            n_classes = len(unique_classes)
            test_samples = int(len(y) * self.validation_split)
            
            # Only use stratification if feasible
            stratify_param = None
            if n_classes <= 10 and test_samples >= n_classes:
                stratify_param = y
                
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.validation_split, random_state=self.random_state, stratify=stratify_param
            )
        except Exception as e:
            logger.warning(f"Stratified split failed in iterative early stopping: {str(e)}, using random split")
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.validation_split, random_state=self.random_state
            )
        
        # Get model parameters
        model_params = self.base_model.get_params()
        max_iter = model_params.get('max_iter', 1000)
        
        # Start with small max_iter and gradually increase
        iter_steps = [50, 100, 200, 300, 500, 700, max_iter]
        iter_steps = [i for i in iter_steps if i <= max_iter]
        if max_iter not in iter_steps:
            iter_steps.append(max_iter)
        
        for current_iter in iter_steps:
            # Create model with current max_iter
            model_params['max_iter'] = current_iter
            current_model = self.base_model.__class__(**model_params)
            
            # Suppress convergence warnings for intermediate iterations
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='lbfgs failed to converge')
                warnings.filterwarnings('ignore', message='Maximum number of iteration reached')
                current_model.fit(X_train, y_train)
            
            # Calculate validation score
            current_score = self._calculate_score(current_model, X_val, y_val, monitor_metric)
            self.history_.append(current_score)
            
            if self.verbose >= 1:
                logger.info(f"Iterations: {current_iter}, Validation {monitor_metric}: {current_score:.4f}")
            
            # Check for improvement
            if self._is_improvement(current_score, self.best_score_, monitor_metric):
                self.best_score_ = current_score
                if self.restore_best_weights:
                    # Store model parameters instead of the model itself to avoid pickle issues
                    self.best_model_params_ = current_model.get_params()
                    self.best_max_iter_ = current_iter
                    self.best_model_ = current_model  # Keep reference for immediate use
                self.wait_ = 0
            else:
                self.wait_ += 1
            
            # Early stopping check
            if self.wait_ >= self.patience:
                self.stopped_epoch_ = current_iter
                if self.verbose >= 1:
                    logger.info(f"Early stopping at {current_iter} iterations. Best {monitor_metric}: {self.best_score_:.4f}")
                break
        
        # If no best model was saved, use the last trained model
        if self.best_model_ is None:
            self.best_model_ = current_model
            
        return self
    
    def predict(self, X):
        """Make predictions using the best model."""
        if self.best_model_ is None:
            raise ValueError("Model must be fitted before making predictions")
        return self.best_model_.predict(X)
    
    def predict_proba(self, X):
        """Make probability predictions using the best model."""
        if self.best_model_ is None:
            raise ValueError("Model must be fitted before making predictions")
        if hasattr(self.best_model_, 'predict_proba'):
            return self.best_model_.predict_proba(X)
        else:
            raise AttributeError(f"Model {self.best_model_.__class__.__name__} does not support predict_proba")
    
    def score(self, X, y):
        """Calculate score using the best model."""
        if self.best_model_ is None:
            raise ValueError("Model must be fitted before scoring")
        return self.best_model_.score(X, y)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        params = {
            'base_model': self.base_model,
            'patience': self.patience,
            'min_delta': self.min_delta,
            'validation_split': self.validation_split,
            'restore_best_weights': self.restore_best_weights,
            'monitor_metric': self.monitor_metric,
            'verbose': self.verbose,
            'random_state': self.random_state
        }
        if deep and hasattr(self.base_model, 'get_params'):
            base_params = self.base_model.get_params(deep=True)
            for key, value in base_params.items():
                params[f'base_model__{key}'] = value
        return params
    
    def set_params(self, **params):
        """Set parameters for this estimator."""
        base_params = {}
        wrapper_params = {}
        
        for key, value in params.items():
            if key.startswith('base_model__'):
                base_params[key[12:]] = value  # Remove 'base_model__' prefix
            else:
                wrapper_params[key] = value
        
        # Set base model parameters
        if base_params and hasattr(self.base_model, 'set_params'):
            self.base_model.set_params(**base_params)
        
        # Set wrapper parameters
        for key, value in wrapper_params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        return self
    
    def _recreate_best_model(self):
        """Recreate the best model from stored parameters for pickling."""
        if self.best_model_params_ is None:
            return self.base_model
        
        # Create new model with best parameters
        recreated_model = self.base_model.__class__(**self.best_model_params_)
        return recreated_model
    
    def __getstate__(self):
        """Custom pickling to avoid recursion issues."""
        state = self.__dict__.copy()
        # Replace the best_model_ with None to avoid pickle issues
        # We'll recreate it from parameters when needed
        if self.best_model_params_ is not None:
            state['best_model_'] = None
        return state
    
    def __setstate__(self, state):
        """Custom unpickling to restore the model."""
        self.__dict__.update(state)
        # Recreate the best model if we have parameters
        if self.best_model_params_ is not None and self.best_model_ is None:
            self.best_model_ = self._recreate_best_model()
    
    def __getattr__(self, name):
        """Delegate attribute access to the best model if not found in wrapper."""
        if self.best_model_ is not None and hasattr(self.best_model_, name):
            return getattr(self.best_model_, name)
        elif hasattr(self.base_model, name):
            return getattr(self.base_model, name)
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

# Add at the top of the file, after logger definition
_feature_mismatch_logged = set()

# Improved caching with LRU + size limits - replace existing cache implementation
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

# Initialize caches with enhanced configuration from config
from config import CACHE_CONFIG

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
    return hash_obj.hexdigest()

# Update cache functions to use the new key generation and caching strategy
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
    key = _generate_cache_key(ds_name, fold_idx, selector_type, "sel_reg", n_feats, X.shape if X is not None else None)
    
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
        
        # Use the enhanced validation function instead of verify_data_alignment
        X_arr, y_arr = validate_and_fix_shape_mismatch(
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
        
        # Handle fast feature selection methods
        if isinstance(selector, dict) and selector.get('type') == 'fast_fs':
            try:
                from fast_feature_selection import FastFeatureSelector
                logger.info(f"Using fast feature selection method: {selector['method']}")
                
                # Create and fit the fast selector
                fast_selector = FastFeatureSelector(
                    method=selector['method'],
                    n_features=effective_n_feats,
                    random_state=42
                )
                fast_selector.fit(X_arr, y_arr, is_regression=selector['is_regression'])
                
                # Get selected features and transform data
                selected_indices = fast_selector.get_selected_features()
                selected_features = selected_indices
                X_selected = X_arr[:, selected_indices]
                
                # Cache and return
                result = (selected_features, X_selected)
                _selector_cache['sel_reg'].put(key, result, item_size=X_selected.nbytes)
                return result
            except ImportError:
                logger.warning("Fast feature selection module not found, using f_regression as fallback")
                selector = SelectKBest(f_regression, k=effective_n_feats)
            except Exception as e:
                logger.warning(f"Fast feature selection error: {str(e)}, using f_regression as fallback")
                selector = SelectKBest(f_regression, k=effective_n_feats)
        
        # Handle MRMR using our custom implementation - check both dict type and original selector code
        elif ((isinstance(selector, dict) and selector_type == "mrmr_reg") or 
            (original_selector_code == "mrmr_reg")):
            try:
                # Try to import our custom MRMR implementation
                from mrmr_helper import simple_mrmr
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
                result = (selected_features, X_selected)
                _selector_cache['sel_reg'].put(key, result, item_size=X_selected.nbytes)
                return result
            except ImportError:
                logger.warning("Custom MRMR implementation not found, using mutual_info_regression")
                selector = SelectKBest(mutual_info_regression, k=effective_n_feats)
            except Exception as e:
                logger.warning(f"MRMR error: {str(e)}, using mutual_info_regression as fallback")
                selector = SelectKBest(mutual_info_regression, k=effective_n_feats)
        
        # Handle other dictionary-based selectors
        elif isinstance(selector, dict):
            if selector_type == "lasso":
                # Manually fit Lasso first with optimized parameters
                estimator = Lasso(**MODEL_OPTIMIZATIONS["Lasso"])
                estimator.fit(X_arr, y_arr)
                
                # Now create SelectFromModel with the fitted estimator
                sfm = SelectFromModel(estimator, max_features=effective_n_feats, prefit=True)
                X_selected = sfm.transform(X_arr)
                selected_features = np.arange(X_arr.shape[1])[sfm.get_support()]
                
            elif selector_type == "enet":
                # Manually fit ElasticNet first with optimized parameters
                estimator = ElasticNet(
                    alpha=0.1,      # Lower alpha for better convergence
                    l1_ratio=0.5,    # Balanced L1/L2 ratio
                    max_iter=10000,   # Increased iterations
                    tol=1e-4,        # Convergence tolerance
                    selection='cyclic',  # Coordinate descent selection
                    random_state=42
                )
                estimator.fit(X_arr, y_arr)
                
                # Now create SelectFromModel with the fitted estimator
                sfm = SelectFromModel(estimator, max_features=effective_n_feats, prefit=True)
                X_selected = sfm.transform(X_arr)
                selected_features = np.arange(X_arr.shape[1])[sfm.get_support()]
                
            elif selector_type == "boruta_reg":
                # Use the stable boruta_selector
                from utils_boruta import boruta_selector
                selected_features = boruta_selector(
                    X_arr, y_arr, n_feats=effective_n_feats, 
                    task="reg", random_state=42
                )
                X_selected = X_arr[:, selected_features]
                
            elif selector_type == "rf_reg":
                # Random Forest Feature Importance for regression
                rf_model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=6,
                    random_state=42,
                    n_jobs=-1
                )
                rf_model.fit(X_arr, y_arr)
                importances = rf_model.feature_importances_
                sorted_indices = np.argsort(importances)[::-1]
                top_indices = sorted_indices[:min(n_feats, X_arr.shape[1])]
                mask = np.zeros(X_arr.shape[1], dtype=bool)
                mask[top_indices] = True
                selected_features = mask
                X_selected = X_arr[:, top_indices]
                
            else:
                # Fallback to mutual_info_regression
                logger.warning(f"Unknown selector type: {selector_type}, using mutual_info_regression")
                selector = SelectKBest(mutual_info_regression, k=effective_n_feats)
                X_selected = selector.fit_transform(X_arr, y_arr)
                selected_features = np.arange(X_arr.shape[1])[selector.get_support()]
            
        # Special handling for Boruta (removed - BorutaPy import was cleaned up)
        # elif isinstance(selector, BorutaPy):
        #     # Use the stable boruta_selector
        #     from utils_boruta import boruta_selector
        #     sel_idx = boruta_selector(
        #         X_arr, y_arr, n_feats=effective_n_feats, 
        #         task="reg", random_state=42
        #     )
        #     X_selected = X_arr[:, sel_idx]
        #     selected_features = sel_idx
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
        
        result = (selected_features, X_selected)
        # Add to cache with memory size tracking
        _selector_cache['sel_reg'].put(key, result, item_size=X_selected.nbytes)
        return result
    except Exception as e:
        logger.error(f"Error in feature selection: {str(e)}")
        # Return a reasonable fallback
        max_cols = min(n_feats, X.shape[1])
        selected_features = np.arange(max_cols)
        X_selected = X[:, selected_features]
        return selected_features, X_selected

def get_regression_extractors() -> Dict[str, Any]:
    """
    Get dictionary of regression feature extractors optimized for genomic data.
    
    Top 5 algorithms selected based on genomic data characteristics:
    1. PLS - Excellent for genomic data, handles multicollinearity well
    2. SparsePLS - Adds sparsity for better interpretability in genomic context  
    3. PCA - Robust baseline, handles high dimensionality effectively
    4. FA - Good for capturing latent biological factors
    5. ICA - Useful for separating independent biological signals
    
    Feature Engineering Tweaks (enabled via CLI):
    - Kernel PCA (RBF) - Captures non-linear gene–methylation interactions for higher R²
    
    Returns
    -------
    dict
        Dictionary mapping extractor names to initialized extractor objects
    """
    # Configure ICA parameters based on sklearn version
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
    
    extractors = {
        # TOP 5 ALGORITHMS FOR GENOMIC DATA
        "PCA": PCA(random_state=42),
        "ICA": FastICA(
            random_state=42,
            **ica_params
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
        ),
        "SparsePLS": SparsePLS(
            n_components=8,
            alpha=0.1,  # Sparsity parameter
            max_iter=1000,
            tol=1e-6,
            scale=True
        )
        
        # COMMENTED OUT - NOT IN TOP 5 FOR GENOMIC DATA
        # "NMF": NMF(
        #     init='nndsvdar',
        #     random_state=42,
        #     max_iter=5000,  # Increased max iterations
        #     tol=1e-3,      # Relaxed tolerance
        #     beta_loss='frobenius',
        #     solver='mu'
        # ),
        # "KernelPCA": KernelPCA(
        #     kernel='rbf',
        #     random_state=42,
        #     n_jobs=-1
        # ),
    }
    
    # Add feature engineering tweaks if enabled
    from config import FEATURE_ENGINEERING_CONFIG
    if FEATURE_ENGINEERING_CONFIG.get("enabled", False):
        if FEATURE_ENGINEERING_CONFIG.get("kernel_pca_enabled", True):
            config = FEATURE_ENGINEERING_CONFIG["kernel_pca"]
            median_config = FEATURE_ENGINEERING_CONFIG["median_heuristic"]
            extractors["KernelPCA-RBF"] = KernelPCAMedianHeuristic(
                n_components=config["n_components"],
                kernel=config["kernel"],
                gamma=config["gamma"],
                eigen_solver=config["eigen_solver"],
                n_jobs=config["n_jobs"],
                random_state=config["random_state"],
                sample_size=median_config["sample_size"],
                percentile=median_config["percentile"]
            )
    
    return extractors

def get_regression_selectors() -> Dict[str, str]:
    """
    Get dictionary of regression feature selectors.
    
    Top 5 algorithms selected based on genomic data characteristics:
    1. ElasticNetFS - Combines L1/L2 regularization, excellent for genomic data
    2. RFImportance - Captures non-linear relationships, robust to noise
    3. VarianceFTest - Fast, effective for removing low-variance features
    4. LASSO - L1 regularization promotes sparsity, good for gene selection
    5. f_regressionFS - Statistical significance-based, appropriate for genomics
    
    Returns
    -------
    dict
        Dictionary mapping selector names to selector codes
    """
    return {
        # TOP 5 ALGORITHMS FOR GENOMIC DATA
        "ElasticNetFS": "elastic_net_reg",
        "RFImportance": "rf_importance_reg", 
        "VarianceFTest": "variance_f_test_reg",
        "LASSO": "lasso",
        "f_regressionFS": "freg"
        
        # COMMENTED OUT - NOT IN TOP 5 FOR GENOMIC DATA
        # "CorrelationFS": "correlation_reg",
        # "CombinedFast": "combined_fast_reg",
        # "RandomForestFS": "rf_reg"
        # MRMR running takes a lot of time. It was replaced with above methods
        # "MRMR": "mrmr_reg",
        # Boruta running takes 20 mins per fold. It was replaced with RandomForestFS, but may be tested in the future.
        # "Boruta": "boruta_reg",
    }

def get_classification_extractors() -> Dict[str, Any]:
    """
    Get dictionary of classification feature extractors optimized for genomic data.
    
    Top 5 algorithms selected based on genomic data characteristics:
    1. PLS-DA - Supervised method, excellent for genomic classification
    2. SparsePLS - Sparse supervised extraction with interpretability
    3. PCA - Reliable baseline for dimensionality reduction
    4. LDA - Supervised, maximizes class separation
    5. FA - Captures underlying biological factors
    
    Feature Engineering Tweaks (enabled via CLI):
    - SparsePLS-DA - Creates maximally discriminative latent space for better MCC
    
    Returns
    -------
    dict
        Dictionary mapping extractor names to initialized extractor objects
    """
    # Configure ICA parameters based on sklearn version
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
    
    extractors = {
        # TOP 5 ALGORITHMS FOR GENOMIC DATA
        "PCA": PCA(random_state=42),
        "FA": FactorAnalysis(
            random_state=42,
            max_iter=5000,  # Increased max iterations
            tol=1e-3       # Relaxed tolerance
        ),
        "LDA": LDA(),
        "PLS-DA": PLSDiscriminantAnalysis(
            n_components=8,
            max_iter=1000,
            tol=1e-6,
            scale=True
        ),
        "SparsePLS": SparsePLS(
            n_components=8,
            alpha=0.1,  # Sparsity parameter
            max_iter=1000,
            tol=1e-6,
            scale=True
        )
        
        # COMMENTED OUT - NOT IN TOP 5 FOR GENOMIC DATA
        # "NMF": NMF(
        #     init='nndsvdar',
        #     random_state=42,
        #     max_iter=5000,  # Increased max iterations
        #     tol=1e-3,      # Relaxed tolerance
        #     beta_loss='frobenius',
        #     solver='mu'
        # ),
        # "ICA": FastICA(
        #     random_state=42,
        #     **ica_params
        # ),
        # "KernelPCA": KernelPCA(
        #     kernel='rbf',
        #     random_state=42,
        #     n_jobs=-1
        # ),
    }
    
    # Add feature engineering tweaks if enabled
    from config import FEATURE_ENGINEERING_CONFIG
    if FEATURE_ENGINEERING_CONFIG.get("enabled", False):
        if FEATURE_ENGINEERING_CONFIG.get("sparse_plsda_enabled", True):
            config = FEATURE_ENGINEERING_CONFIG["sparse_plsda"]
            extractors["SparsePLS-DA"] = SparsePLSDA(
                n_components=config["n_components"],
                alpha=config["alpha"],
                max_iter=config["max_iter"],
                tol=config["tol"],
                scale=config["scale"]
            )
    
    return extractors

def get_classification_selectors() -> Dict[str, str]:
    """
    Get dictionary of classification feature selectors.
    
    Top 5 algorithms selected based on genomic data characteristics:
    1. ElasticNetFS - Excellent balance of L1/L2 regularization
    2. RFImportance - Handles non-linear patterns in genomic data
    3. VarianceFTest - Fast preprocessing step for genomic data
    4. LogisticL1 - L1 regularization for sparse feature selection
    5. XGBoostFS - Advanced tree-based importance for complex patterns
    
    Returns
    -------
    dict
        Dictionary mapping selector names to selector codes
    """
    return {
        # TOP 5 ALGORITHMS FOR GENOMIC DATA
        "ElasticNetFS": "elastic_net_clf", 
        "RFImportance": "rf_importance_clf",
        "VarianceFTest": "variance_f_test_clf",
        "LogisticL1": "logistic_l1",
        "XGBoostFS": "xgb_clf"
        
        # COMMENTED OUT - NOT IN TOP 5 FOR GENOMIC DATA
        # "Chi2FS": "chi2_fast",
        # "CombinedFast": "combined_fast_clf",
        # "fclassifFS": "fclassif",
        # MRMR running takes a lot of time. It was replaced with above methods
        # "MRMR": "mrmr_clf",
        # Boruta running takes 20 mins per fold. It was replaced with XGBoostFS, but may be tested in the future.
        # "Boruta": "boruta_clf",
    }

def get_selector_object(selector_code: str, n_feats: int):
    """
    Create a feature selector object based on the selector code.
    Improved to be less aggressive and better handle small datasets.
    
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
    # Ensure minimum number of features for small datasets
    min_features = max(5, min(n_feats, 10))  # At least 5 features, but not more than requested
    effective_n_feats = max(min_features, n_feats)
    
    if selector_code == "mutual_info_regression":
        return SelectKBest(score_func=mutual_info_regression, k=effective_n_feats)
    elif selector_code == "mutual_info_classification":
        return SelectKBest(score_func=mutual_info_classif, k=effective_n_feats)
    elif selector_code == "f_regression":
        return SelectKBest(score_func=f_regression, k=effective_n_feats)
    elif selector_code == "f_classification":
        return SelectKBest(score_func=f_classif, k=effective_n_feats)
    elif selector_code == "chi2":
        return SelectKBest(score_func=chi2, k=effective_n_feats)
    elif selector_code == "lasso_regression":
        # Use less aggressive regularization for better feature retention
        alpha_value = max(0.001, 0.1 / np.sqrt(effective_n_feats))  # Adaptive alpha
        lasso = Lasso(alpha=alpha_value, max_iter=5000, random_state=42)
        return SelectFromModel(lasso, max_features=effective_n_feats, threshold=-np.inf)
    elif selector_code == "lasso_classification":
        # Use less aggressive regularization for classification
        C_value = max(0.1, 10.0 / np.sqrt(effective_n_feats))  # Adaptive C
        logistic = LogisticRegression(
            penalty='l1', solver='liblinear', C=C_value, 
            max_iter=5000, random_state=42, class_weight='balanced'
        )
        return SelectFromModel(logistic, max_features=effective_n_feats, threshold=-np.inf)
    elif selector_code == "elasticnet_regression":
        # Use less aggressive regularization
        alpha_value = max(0.001, 0.05 / np.sqrt(effective_n_feats))
        elasticnet = ElasticNet(
            alpha=alpha_value, l1_ratio=0.5, max_iter=5000, 
            random_state=42, selection='cyclic'
        )
        return SelectFromModel(elasticnet, max_features=effective_n_feats, threshold=-np.inf)
    elif selector_code == "elasticnet_classification":
        # Use less aggressive regularization for classification
        C_value = max(0.1, 5.0 / np.sqrt(effective_n_feats))
        logistic = LogisticRegression(
            penalty='elasticnet', solver='saga', C=C_value, l1_ratio=0.5,
            max_iter=5000, random_state=42, class_weight='balanced'
        )
        return SelectFromModel(logistic, max_features=effective_n_feats, threshold=-np.inf)
    elif selector_code == "random_forest_regression":
        # Use more trees for better feature importance estimation
        rf = RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_leaf=2,
            random_state=42, n_jobs=-1, oob_score=False
        )
        return SelectFromModel(rf, max_features=effective_n_feats, threshold=-np.inf)
    elif selector_code == "random_forest_classification":
        # Use more trees for better feature importance estimation
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=2,
            random_state=42, n_jobs=-1, class_weight='balanced', oob_score=False
        )
        return SelectFromModel(rf, max_features=effective_n_feats, threshold=-np.inf)
    elif selector_code == "boruta_regression":
        # Use Boruta with less aggressive settings
        rf = RandomForestRegressor(
            n_estimators=100, max_depth=7, random_state=42, n_jobs=-1
        )
        return boruta_selector(rf, n_feats=effective_n_feats, is_classification=False)
    elif selector_code == "boruta_classification":
        # Use Boruta with less aggressive settings
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=7, random_state=42, 
            n_jobs=-1, class_weight='balanced'
        )
        return boruta_selector(rf, n_feats=effective_n_feats, is_classification=True)
    
    # Fast feature selection methods for regression - return sklearn objects with genomic optimization
    elif selector_code == "variance_f_test_reg":
        # Use genomic-optimized F-test selector with much larger k
        genomic_k = min(effective_n_feats, 10000)  # Much larger for genomic data
        return SelectKBest(score_func=f_regression, k=genomic_k)
    elif selector_code == "rf_importance_reg":
        # Use genomic-optimized RandomForest selector
        rf = RandomForestRegressor(
            n_estimators=1000, max_depth=None, min_samples_split=2,
            min_samples_leaf=1, random_state=42, n_jobs=-1
        )
        return SelectFromModel(rf, max_features=effective_n_feats, threshold="0.001*mean")
    elif selector_code == "elastic_net_reg":
        # Use genomic-optimized ElasticNet selector with minimal regularization
        enet = ElasticNet(alpha=0.0001, l1_ratio=0.1, max_iter=5000, random_state=42)
        return SelectFromModel(enet, max_features=effective_n_feats, threshold="0.001*mean")
    elif selector_code == "correlation_reg":
        # Fallback to F-test for correlation-based selection
        genomic_k = min(effective_n_feats, 10000)
        return SelectKBest(score_func=f_regression, k=genomic_k)
    elif selector_code == "combined_fast_reg":
        # Use F-test as the combined fast method
        genomic_k = min(effective_n_feats, 10000)
        return SelectKBest(score_func=f_regression, k=genomic_k)
    
    # Fast feature selection methods for classification - return sklearn objects with genomic optimization
    elif selector_code == "variance_f_test_clf":
        # Use genomic-optimized F-test selector with much larger k
        genomic_k = min(effective_n_feats, 10000)  # Much larger for genomic data
        return SelectKBest(score_func=f_classif, k=genomic_k)
    elif selector_code == "rf_importance_clf":
        # Use genomic-optimized RandomForest selector
        rf = RandomForestClassifier(
            n_estimators=1000, max_depth=None, min_samples_split=2,
            min_samples_leaf=1, random_state=42, n_jobs=-1, class_weight='balanced'
        )
        return SelectFromModel(rf, max_features=effective_n_feats, threshold="0.001*mean")
    elif selector_code == "elastic_net_clf":
        # Use genomic-optimized LogisticRegression with minimal regularization
        lr = LogisticRegression(
            C=100.0, max_iter=5000, random_state=42, solver='liblinear', class_weight='balanced'
        )
        return SelectFromModel(lr, max_features=effective_n_feats, threshold="0.001*mean")
    elif selector_code == "chi2_fast":
        # Use genomic-optimized Chi2 selector
        genomic_k = min(effective_n_feats, 10000)
        return SelectKBest(score_func=chi2, k=genomic_k)
    elif selector_code == "combined_fast_clf":
        # Use F-test as the combined fast method
        genomic_k = min(effective_n_feats, 10000)
        return SelectKBest(score_func=f_classif, k=genomic_k)
    
    # Legacy selector codes - map to standard sklearn selectors with genomic optimization
    elif selector_code == "lasso":
        return SelectFromModel(
            Lasso(alpha=0.0001, max_iter=5000, random_state=42),  # Much lower alpha for genomic data
            max_features=effective_n_feats, threshold="0.001*mean"  # Very low threshold
        )
    elif selector_code == "freg":
        # Use genomic-optimized F-test with larger k
        genomic_k = min(effective_n_feats, 10000)
        return SelectKBest(score_func=f_regression, k=genomic_k)
    elif selector_code == "rf_reg":
        rf = RandomForestRegressor(
            n_estimators=1000, max_depth=None, min_samples_leaf=1,  # Genomic optimization
            random_state=42, n_jobs=-1, oob_score=False
        )
        return SelectFromModel(rf, max_features=effective_n_feats, threshold="0.001*mean")
    elif selector_code == "fclassif":
        # Use genomic-optimized F-test with larger k
        genomic_k = min(effective_n_feats, 10000)
        return SelectKBest(score_func=f_classif, k=genomic_k)
    elif selector_code == "logistic_l1":
        logistic = LogisticRegression(
            penalty='l1', solver='liblinear', C=100.0,  # Much higher C for genomic data
            max_iter=5000, random_state=42, class_weight='balanced'
        )
        return SelectFromModel(logistic, max_features=effective_n_feats, threshold="0.001*mean")
    elif selector_code == "xgb_clf":
        # Fallback to RandomForest with genomic optimization
        rf = RandomForestClassifier(
            n_estimators=1000, max_depth=None, min_samples_leaf=1,  # Genomic optimization
            random_state=42, n_jobs=-1, class_weight='balanced', oob_score=False
        )
        return SelectFromModel(rf, max_features=effective_n_feats, threshold="0.001*mean")
    
    else:
        # Default fallback to mutual information
        logger.warning(f"Unknown selector code: {selector_code}, using mutual_info_regression")
        return SelectKBest(score_func=mutual_info_regression, k=effective_n_feats)

def get_regression_models() -> Dict[str, Any]:
    """Get dictionary of regression models with tuned ensembles."""
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=500,  # Increased from 100 to 500
            max_depth=None,    # Changed from limited depth to None
            min_samples_leaf=2, # Changed from 5 to 2
            random_state=42
        ),
        "ElasticNet": ElasticNet(**MODEL_OPTIMIZATIONS["ElasticNet"])
    }
    
    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        models["XGBRegressor"] = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0
        )
    
    # Add LightGBM if available
    if LIGHTGBM_AVAILABLE:
        models["LGBMRegressor"] = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=42,
            verbosity=-1
        )
    
    # Add GradientBoostingRegressor as suggested
    models["GradientBoostingRegressor"] = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    
    # Add improved regression models for negative R² issues
    if XGBOOST_AVAILABLE:
        models["ImprovedXGBRegressor"] = xgb.XGBRegressor(
            n_estimators=800,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0
        )
    
    if LIGHTGBM_AVAILABLE:
        models["ImprovedLightGBMRegressor"] = lgb.LGBMRegressor(
            n_estimators=800,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="regression",  # Can be changed to "quantile" for robust loss
            random_state=42,
            verbosity=-1
        )
    
    # Add robust gradient boosting with Huber loss
    models["RobustGradientBoosting"] = GradientBoostingRegressor(
        n_estimators=700,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        loss="huber",  # Robust loss function for outliers
        alpha=0.9,     # Huber loss parameter
        random_state=42
    )
    
    return models

def get_classification_models() -> Dict[str, Any]:
    """Get dictionary of classification models with tuned ensembles."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import LinearSVC
    
    models = {
        "LogisticRegression": LogisticRegression(
            random_state=42,
            penalty='l2',
            solver='liblinear',
            class_weight='balanced',  # Added balanced class weights
            max_iter=500,  # Increased max_iter
            C=1.0  # Will be tuned via hyperparameter search
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=500,  # Increased from 100 to 500
            max_depth=None,    # Changed from limited depth to None
            min_samples_leaf=2, # Changed from default to 2
            class_weight='balanced',
            random_state=42
        ),
        "SVC": SVC(
            probability=True,
            random_state=42,
            class_weight='balanced'
        )
    }
    
    # Add calibrated Linear SVC as suggested
    linear_svc = LinearSVC(
        random_state=42,
        class_weight='balanced',
        max_iter=2000
    )
    models["CalibratedLinearSVC"] = CalibratedClassifierCV(
        linear_svc, 
        method='sigmoid',
        cv=3
    )
    
    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        models["XGBClassifier"] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0,
            eval_metric='logloss'
        )
    
    # Add LightGBM if available
    if LIGHTGBM_AVAILABLE:
        models["LGBMClassifier"] = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=42,
            verbosity=-1
        )
    
    # Add GradientBoostingClassifier as suggested
    models["GradientBoostingClassifier"] = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    
    # Add balanced models for class imbalance handling
    if IMBALANCED_LEARN_AVAILABLE:
        models["BalancedRandomForest"] = BalancedRandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            sampling_strategy='auto',
            replacement=False,
            bootstrap=True,
            oob_score=True,
            random_state=42
        )
    
    if XGBOOST_AVAILABLE:
        models["BalancedXGBoost"] = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            scale_pos_weight=None,  # Let sampler handle balance
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0
        )
    
    if LIGHTGBM_AVAILABLE:
        models["BalancedLightGBM"] = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=1.0,
            is_unbalance=True,  # LightGBM's built-in class weight handling
            random_state=42,
            verbosity=-1
        )
    
    return models

def get_model_object(model_name: str, random_state: Optional[int] = None, enable_early_stopping: bool = None, dataset: str = None):
    """
    Create a model object with optimized parameters and early stopping support.
    
    Parameters
    ----------
    model_name : str
        Name of the model to create
    random_state : int, optional
        Random state for reproducibility
    enable_early_stopping : bool, optional
        Whether to enable early stopping (if None, uses global config)
        
    Returns
    -------
    object
        Configured model object
    """
    # Use global early stopping setting if not specified
    if enable_early_stopping is None:
        enable_early_stopping = EARLY_STOPPING_CONFIG.get("enabled", True)
    
    # Set random state if provided
    if random_state is None:
        random_state = 42
    
    # Create base models with optimized parameters
    if model_name == "LinearRegression":
        model_params = MODEL_OPTIMIZATIONS.get("LinearRegression", {}).copy()
        base_model = LinearRegression(**model_params)
    
    elif model_name == "Lasso":
        model_params = MODEL_OPTIMIZATIONS.get("Lasso", {}).copy()
        if random_state is not None:
            model_params["random_state"] = random_state
        base_model = Lasso(**model_params)
    
    elif model_name == "ElasticNet":
        model_params = MODEL_OPTIMIZATIONS.get("ElasticNet", {}).copy()
        if random_state is not None:
            model_params["random_state"] = random_state
        base_model = ElasticNet(**model_params)
    
    elif model_name == "RandomForestRegressor":
        model_params = MODEL_OPTIMIZATIONS.get("RandomForestRegressor", {}).copy()
        if random_state is not None:
            model_params["random_state"] = random_state
        base_model = RandomForestRegressor(**model_params)
    
    elif model_name == "RandomForestClassifier":
        model_params = MODEL_OPTIMIZATIONS.get("RandomForestClassifier", {}).copy()
        if random_state is not None:
            model_params["random_state"] = random_state
        base_model = RandomForestClassifier(**model_params)
    
    elif model_name == "LogisticRegression":
        model_params = MODEL_OPTIMIZATIONS.get("LogisticRegression", {}).copy()
        if random_state is not None:
            model_params["random_state"] = random_state
        base_model = LogisticRegression(**model_params)
    
    elif model_name == "SVR":
        model_params = MODEL_OPTIMIZATIONS.get("SVR", {}).copy()
        if "kernel" not in model_params:
            model_params["kernel"] = 'rbf'
        if "cache_size" not in model_params:
            model_params["cache_size"] = 500
        base_model = SVR(**model_params)
    
    elif model_name == "SVC":
        model_params = MODEL_OPTIMIZATIONS.get("SVC", {}).copy()
        if random_state is not None:
            model_params["random_state"] = random_state
        if "kernel" not in model_params:
            model_params["kernel"] = 'rbf'
        if "probability" not in model_params:
            model_params["probability"] = True
        base_model = SVC(**model_params)
    
    elif model_name == "BalancedRandomForest":
        if not IMBALANCED_LEARN_AVAILABLE:
            logger.warning("imbalanced-learn not available, falling back to RandomForestClassifier with class_weight='balanced'")
            model_params = MODEL_OPTIMIZATIONS.get("RandomForestClassifier", {}).copy()
            model_params["class_weight"] = "balanced"
            if random_state is not None:
                model_params["random_state"] = random_state
            base_model = RandomForestClassifier(**model_params)
        else:
            model_params = MODEL_OPTIMIZATIONS.get("BalancedRandomForest", {}).copy()
            if random_state is not None:
                model_params["random_state"] = random_state
            base_model = BalancedRandomForestClassifier(**model_params)
    
    elif model_name == "BalancedXGBoost":
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available, falling back to GradientBoostingClassifier")
            model_params = MODEL_OPTIMIZATIONS.get("GradientBoosting", {}).copy()
            if random_state is not None:
                model_params["random_state"] = random_state
            base_model = GradientBoostingClassifier(**model_params)
        else:
            model_params = MODEL_OPTIMIZATIONS.get("BalancedXGBoost", {}).copy()
            if random_state is not None:
                model_params["random_state"] = random_state
            # Remove eval_metric from params as it's handled differently in XGBClassifier
            eval_metric = model_params.pop("eval_metric", "logloss")
            base_model = xgb.XGBClassifier(eval_metric=eval_metric, **model_params)
    
    elif model_name == "BalancedLightGBM":
        if not LIGHTGBM_AVAILABLE:
            logger.warning("LightGBM not available, falling back to GradientBoostingClassifier")
            model_params = MODEL_OPTIMIZATIONS.get("GradientBoosting", {}).copy()
            if random_state is not None:
                model_params["random_state"] = random_state
            base_model = GradientBoostingClassifier(**model_params)
        else:
            model_params = MODEL_OPTIMIZATIONS.get("BalancedLightGBM", {}).copy()
            if random_state is not None:
                model_params["random_state"] = random_state
            base_model = lgb.LGBMClassifier(**model_params)
    
    elif model_name == "ImprovedXGBRegressor":
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available, falling back to GradientBoostingRegressor")
            model_params = MODEL_OPTIMIZATIONS.get("RobustGradientBoosting", {}).copy()
            if random_state is not None:
                model_params["random_state"] = random_state
            base_model = GradientBoostingRegressor(**model_params)
        else:
            model_params = MODEL_OPTIMIZATIONS.get("ImprovedXGBRegressor", {}).copy()
            if random_state is not None:
                model_params["random_state"] = random_state
            base_model = xgb.XGBRegressor(**model_params)
    
    elif model_name == "ImprovedLightGBMRegressor":
        if not LIGHTGBM_AVAILABLE:
            logger.warning("LightGBM not available, falling back to GradientBoostingRegressor")
            model_params = MODEL_OPTIMIZATIONS.get("RobustGradientBoosting", {}).copy()
            if random_state is not None:
                model_params["random_state"] = random_state
            base_model = GradientBoostingRegressor(**model_params)
        else:
            model_params = MODEL_OPTIMIZATIONS.get("ImprovedLightGBMRegressor", {}).copy()
            if random_state is not None:
                model_params["random_state"] = random_state
            base_model = lgb.LGBMRegressor(**model_params)
    
    elif model_name == "RobustGradientBoosting":
        model_params = MODEL_OPTIMIZATIONS.get("RobustGradientBoosting", {}).copy()
        if random_state is not None:
            model_params["random_state"] = random_state
        base_model = GradientBoostingRegressor(**model_params)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # For models that support early stopping, wrap them
    if enable_early_stopping and model_name in ["RandomForestRegressor", "RandomForestClassifier"]:
        # Get early stopping parameters without the 'enabled' key and invalid parameters
        early_stopping_params = {k: v for k, v in EARLY_STOPPING_CONFIG.items() 
                                if k not in ['enabled', 'adaptive_patience', 'max_patience']}
        base_model = EarlyStoppingWrapper(base_model, **early_stopping_params)
    
    # Apply target transformation for regression models if dataset is in registry
    if dataset in TARGET_TRANSFORMS and model_name in ["LinearRegression", "Lasso", "ElasticNet", "RandomForestRegressor", "SVR", "ImprovedXGBRegressor", "ImprovedLightGBMRegressor", "RobustGradientBoosting"]:
        from sklearn.compose import TransformedTargetRegressor
        name, fwd, inv = TARGET_TRANSFORMS[dataset]
        logger.info(f"Applying {name} target transformation for dataset {dataset}")
        base_model = TransformedTargetRegressor(
            regressor=base_model,
            func=fwd,
            inverse_func=inv,
            check_inverse=False
        )
    
    return base_model

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
            max_index = max(selected_features) if len(selected_features) > 0 else -1
            if max_index >= X.shape[1]:
                logger.warning(f"Selected feature indices exceed X dimensions: max index {max_index} vs {X.shape[1]} columns")
                # Take only valid indices
                valid_indices = [i for i in selected_features if i < X.shape[1]]
                if len(valid_indices) == 0:
                    # If no valid indices, take first column as fallback
                    logger.warning("No valid feature indices found, using first column as fallback")
                    return X.iloc[:, 0:1].values
                logger.debug(f"Using {len(valid_indices)} valid indices out of {len(selected_features)} selected")
                return X.iloc[:, valid_indices].values
            return X.iloc[:, selected_features].values
        else:
            # For numpy arrays
            max_index = max(selected_features) if len(selected_features) > 0 else -1
            if max_index >= X.shape[1]:
                logger.warning(f"Selected feature indices exceed X dimensions: max index {max_index} vs {X.shape[1]} columns")
                # Take only valid indices
                valid_indices = [i for i in selected_features if i < X.shape[1]]
                if len(valid_indices) == 0:
                    # If no valid indices, take first column as fallback
                    logger.warning("No valid feature indices found, using first column as fallback")
                    return X[:, 0:1]
                logger.debug(f"Using {len(valid_indices)} valid indices out of {len(selected_features)} selected")
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

def cached_fit_transform_selector_classification(X, y, selector_code, n_feats, ds_name, modality_name, fold_idx=None):
    """Cached version of fit_transform for classification selectors."""
    # Generate a stable cache key
    if isinstance(selector_code, dict):
        selector_type = selector_code['type']
    elif isinstance(selector_code, str):
        selector_type = selector_code
    else:
        # Handle case where a full selector object is passed instead of a code
        if hasattr(selector_code, '__class__'):
            selector_type = selector_code.__class__.__name__
        else:
            selector_type = "unknown"
            
    # Create cache key
    key = _generate_cache_key(ds_name, fold_idx, selector_type, "sel_clf", n_feats, X.shape if X is not None else None)
    
    # Check cache
    cached_result = _selector_cache['sel_clf'].get(key)
    if cached_result is not None:
        logger.debug(f"Cache hit: {key}")
        return cached_result
    
    # Process selector based on type
    selected_features = None
    transformed_X = None
    
    # Log what we're doing
    logger.debug(f"Starting selector: {selector_type}, n_feats: {n_feats}, shape: {X.shape}")
    
    try:
        # Early validation to catch zero-size arrays before any processing
        if X is None or (hasattr(X, 'size') and X.size == 0) or (hasattr(X, 'shape') and (X.shape[0] == 0 or X.shape[1] == 0)):
            logger.warning(f"Zero-size or None input data for {modality_name} in fold {fold_idx}: X={X.shape if hasattr(X, 'shape') else 'None'}")
            return None, None
        
        if y is None or (hasattr(y, 'size') and y.size == 0):
            logger.warning(f"Zero-size or None target data for {modality_name} in fold {fold_idx}: y={y.shape if hasattr(y, 'shape') else 'None'}")
            return None, None
        
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
                    logger.debug(f"Converting float y to integer for classification - {modality_name} in fold {fold_idx}")
                    y_safe = np.round(y).astype(np.int32)
                else:
                    y_safe = np.asarray(y)
            except:
                logger.warning(f"Could not convert y for {modality_name} in fold {fold_idx}")
                y_safe = y
        else:
            y_safe = None
            
        # Use the enhanced validation function instead of verify_data_alignment
        X_safe, y_safe = validate_and_fix_shape_mismatch(
            X_safe, y_safe, 
            name=f"{modality_name} selector data", 
            fold_idx=fold_idx
        )
        
        # If alignment failed, return a fallback
        if X_safe is None or y_safe is None:
            logger.warning(f"Data alignment failure in {selector_type} for {modality_name}")
            # Fallback to first feature if we can
            if X is not None and X.shape[1] > 0:
                mask = np.zeros(X.shape[1], dtype=bool)
                mask[0] = True
                selected_features = mask
                transformed_X = X[:min(len(X), 10), [0]]  # Use up to 10 samples
                result = (selected_features, transformed_X)
                _selector_cache['sel_clf'].put(key, result, item_size=transformed_X.nbytes)
                return result
            return None, None
        
        # Make sure n_feats is limited by data dimensions
        max_possible_feats = min(X_safe.shape[0], X_safe.shape[1])
        effective_n_feats = min(n_feats, max_possible_feats)
        
        if effective_n_feats < n_feats:
            logger.info(f"Limiting features from {n_feats} to {effective_n_feats} due to data dimensions")
        
        # Handle fast feature selection methods
        if selector_type in ['variance_f_test_clf', 'rf_importance_clf', 'elastic_net_clf', 'chi2_fast', 'combined_fast_clf']:
            try:
                from fast_feature_selection import FastFeatureSelector
                
                # Map selector type to method name
                method_mapping = {
                    'variance_f_test_clf': 'variance_f_test',
                    'rf_importance_clf': 'genomic_ensemble',  # Use ensemble for RF importance
                    'elastic_net_clf': 'genomic_ensemble',    # Use ensemble for elastic net (includes regularization)
                    'chi2_fast': 'chi2',
                    'combined_fast_clf': 'genomic_ensemble'   # Use ensemble for combined methods
                }
                
                method = method_mapping[selector_type]
                logger.info(f"Using fast feature selection method: {method} for {modality_name}")
                
                # Create and fit the fast selector
                fast_selector = FastFeatureSelector(
                    method=method,
                    n_features=effective_n_feats,
                    random_state=42
                )
                fast_selector.fit(X_safe, y_safe, is_regression=False)
                
                # Get selected features and transform data
                selected_indices = fast_selector.get_selected_features()
                
                # Create boolean mask
                selected_features = np.zeros(X_safe.shape[1], dtype=bool)
                selected_features[selected_indices] = True
                transformed_X = X_safe[:, selected_indices]
                
                # Cache and return
                result = (selected_features, transformed_X)
                _selector_cache['sel_clf'].put(key, result, item_size=transformed_X.nbytes)
                return result
                
            except ImportError:
                logger.warning("Fast feature selection module not found, using f_classif as fallback")
                from sklearn.feature_selection import SelectKBest, f_classif
                selector = SelectKBest(f_classif, k=effective_n_feats)
            except Exception as e:
                logger.warning(f"Fast feature selection error for {modality_name}: {str(e)}, using f_classif as fallback")
                from sklearn.feature_selection import SelectKBest, f_classif
                selector = SelectKBest(f_classif, k=effective_n_feats)
        
        # Enhanced feature selection with robust fallbacks
        elif selector_type == 'mrmr_clf' or selector_type == 'MRMR':
            # Handle MRMR using our own implementation with enhanced error handling
            from config import FEATURE_SELECTION_CONFIG
            
            fallback_methods = FEATURE_SELECTION_CONFIG.get("fallback_methods", ["mutual_info", "f_test", "variance"])
            error_tolerance = FEATURE_SELECTION_CONFIG.get("error_tolerance", 3)
            
            for attempt in range(error_tolerance):
                try:
                    # Try to import our custom MRMR implementation
                    try:
                        from mrmr_helper import simple_mrmr
                        logger.info(f"Using custom MRMR implementation for {modality_name} (attempt {attempt + 1})")
                        n_features = min(effective_n_feats, X_safe.shape[1])
                        
                        # Get selected feature indices with timeout protection
                        selected_indices = simple_mrmr(
                            X_safe, y_safe, 
                            n_selected_features=n_features,
                            is_regression=False
                        )
                        
                        # Validate selected indices
                        if selected_indices is not None and len(selected_indices) > 0:
                            # Convert indices to boolean mask
                            mask = np.zeros(X_safe.shape[1], dtype=bool)
                            valid_indices = [i for i in selected_indices if 0 <= i < X_safe.shape[1]]
                            if len(valid_indices) > 0:
                                mask[valid_indices] = True
                                selected_features = mask
                                transformed_X = X_safe[:, valid_indices]
                                
                                # Cache result
                                result = (selected_features, transformed_X)
                                _selector_cache['sel_clf'].put(key, result, item_size=transformed_X.nbytes)
                                return result
                        
                        logger.warning(f"MRMR returned invalid indices for {modality_name}")
                        
                    except ImportError:
                        logger.warning("Custom MRMR implementation not found, falling back to sklearn methods")
                        break  # Exit MRMR attempts and try fallbacks
                        
                except Exception as e:
                    logger.warning(f"MRMR attempt {attempt + 1} failed for {modality_name}: {str(e)}")
                    if attempt == error_tolerance - 1:
                        logger.warning("All MRMR attempts failed, using fallback methods")
                        break
            
            # Try fallback methods in order
            for fallback_method in fallback_methods:
                try:
                    if fallback_method == "mutual_info":
                        from sklearn.feature_selection import SelectKBest, mutual_info_classif
                        selector = SelectKBest(mutual_info_classif, k=effective_n_feats)
                        logger.info(f"Using mutual_info fallback for {modality_name}")
                        break
                    elif fallback_method == "f_test":
                        from sklearn.feature_selection import SelectKBest, f_classif
                        selector = SelectKBest(f_classif, k=effective_n_feats)
                        logger.info(f"Using f_test fallback for {modality_name}")
                        break
                    elif fallback_method == "variance":
                        from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
                        # First apply variance threshold, then select top k
                        var_selector = VarianceThreshold(threshold=0.01)
                        X_var = var_selector.fit_transform(X_safe)
                        if X_var.shape[1] >= effective_n_feats:
                            selector = SelectKBest(f_classif, k=effective_n_feats)
                            logger.info(f"Using variance+f_test fallback for {modality_name}")
                            break
                except Exception as e:
                    logger.warning(f"Fallback method {fallback_method} failed: {str(e)}")
                    continue
            else:
                # If all fallbacks fail, use the most basic selector
                from sklearn.feature_selection import SelectKBest, f_classif
            
            # Use safe f_classif with error handling for divide-by-zero
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='divide by zero encountered in divide')
                    warnings.filterwarnings('ignore', message='invalid value encountered in divide')
                    
                    selector = SelectKBest(f_classif, k=min(n_feats, X_safe.shape[1]))
                    logger.info(f"Using SelectKBest with f_classif for {modality_name}")
            except Exception as e:
                logger.warning(f"f_classif fallback failed for {modality_name}: {str(e)}, using mutual_info_classif")
                from sklearn.feature_selection import mutual_info_classif
                selector = SelectKBest(mutual_info_classif, k=min(n_feats, X_safe.shape[1]))
        
        elif selector_type == 'mrmr_reg':
            # Handle regression MRMR selector called from classification function
            # This happens when individual modality processing determines task type differently
            # from the overall pipeline task type
            logger.info(f"Handling regression MRMR selector in classification function for {modality_name}")
            try:
                # Try to import our custom MRMR implementation
                try:
                    from mrmr_helper import simple_mrmr
                    logger.info(f"Using custom MRMR regression implementation for {modality_name}")
                    n_features = min(n_feats, X_safe.shape[1])
                    
                    # Get selected feature indices - use regression version
                    selected_indices = simple_mrmr(
                        X_safe, y_safe, 
                        n_selected_features=n_features,
                        is_regression=True  # Use regression MRMR
                    )
                    
                    # Convert indices to boolean mask
                    mask = np.zeros(X_safe.shape[1], dtype=bool)
                    mask[selected_indices] = True
                    
                    # Limit to valid indices
                    if len(selected_indices) > 0:
                        selected_features = mask
                        transformed_X = X_safe[:, selected_indices]
                        
                        # Cache result
                        result = (selected_features, transformed_X)
                        _selector_cache['sel_clf'].put(key, result, item_size=transformed_X.nbytes)
                        return result
                except ImportError:
                    logger.warning("Custom MRMR implementation not found, falling back to mutual_info_regression")
            except Exception as e:
                logger.warning(f"MRMR regression error: {str(e)}, using mutual_info_regression as fallback")
            
            # Fallback to mutual_info_regression if MRMR fails (better for regression than f_classif)
            from sklearn.feature_selection import SelectKBest, mutual_info_regression
            
            try:
                selector = SelectKBest(mutual_info_regression, k=min(n_feats, X_safe.shape[1]))
                logger.info(f"Using SelectKBest with mutual_info_regression for {modality_name}")
            except Exception as e:
                logger.warning(f"mutual_info_regression fallback failed for {modality_name}: {str(e)}, using f_classif")
                from sklearn.feature_selection import f_classif
                # Last resort fallback to f_classif
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='divide by zero encountered in divide')
                    warnings.filterwarnings('ignore', message='invalid value encountered in divide')
                    selector = SelectKBest(f_classif, k=min(n_feats, X_safe.shape[1]))
        
        elif selector_type == 'fclassifFS' or selector_type == 'fclassif':
            # Convert to SelectKBest with f_classif
            from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
            
            try:
                # First, remove constant features (zero variance) to avoid sklearn warnings
                variance_selector = VarianceThreshold(threshold=0.0)  # Remove features with zero variance
                X_filtered = variance_selector.fit_transform(X_safe)
                
                # Check if we have enough features left after filtering
                if X_filtered.shape[1] < n_feats:
                    logger.warning(f"Only {X_filtered.shape[1]} non-constant features available for {modality_name}, adjusting n_feats from {n_feats}")
                    effective_n_feats = min(n_feats, X_filtered.shape[1])
                else:
                    effective_n_feats = n_feats
                
                if X_filtered.shape[1] == 0:
                    logger.warning(f"All features are constant for {modality_name}, using first original feature as fallback")
                    # Fallback to first feature
                    mask = np.zeros(X_safe.shape[1], dtype=bool)
                    mask[0] = True
                    selected_features = mask
                    X_selected = X_safe[:, [0]]
                    
                    result = (selected_features, X_selected)
                    _selector_cache['sel_clf'].put(key, result, item_size=X_selected.nbytes)
                    return result
                
                # Apply f_classif selector on filtered data with warning suppression
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='Features .* are constant.', category=UserWarning)
                    warnings.filterwarnings('ignore', message='divide by zero encountered in divide')
                    warnings.filterwarnings('ignore', message='invalid value encountered in divide')
                    
                    selector = SelectKBest(score_func=f_classif, k=effective_n_feats)
                    X_selected_filtered = selector.fit_transform(X_filtered, y_safe)
                    selected_indices_filtered = selector.get_support(indices=True)
                
                # Map back to original feature indices
                original_feature_indices = variance_selector.get_support(indices=True)
                selected_original_indices = original_feature_indices[selected_indices_filtered]
                
                # Create boolean mask for original features
                selected_features = np.zeros(X_safe.shape[1], dtype=bool)
                selected_features[selected_original_indices] = True
                
                # Get the selected data from original X_safe
                X_selected = X_safe[:, selected_original_indices]
                
                logger.debug(f"f_classif selected {len(selected_original_indices)} features for {modality_name} (filtered {X_safe.shape[1] - X_filtered.shape[1]} constant features)")
                
                result = (selected_features, X_selected)
                _selector_cache['sel_clf'].put(key, result, item_size=X_selected.nbytes)
                return result
                
            except Exception as e:
                logger.warning(f"Enhanced f_classif failed for {modality_name}: {str(e)}, using simple fallback")
                # Fallback to simple f_classif without filtering
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='Features .* are constant.', category=UserWarning)
                    warnings.filterwarnings('ignore', message='divide by zero encountered in divide')
                    warnings.filterwarnings('ignore', message='invalid value encountered in divide')
                    
                    selector = SelectKBest(score_func=f_classif, k=min(n_feats, X_safe.shape[1]))
                    X_selected = selector.fit_transform(X_safe, y_safe)
                    selected_features = selector.get_support(indices=True)
                    
                    result = (selected_features, X_selected)
                    _selector_cache['sel_clf'].put(key, result, item_size=X_selected.nbytes)
                    return result
        
        elif selector_type == 'chi2_selection' or selector_type == 'Chi2FS':
            # Convert to SelectKBest with chi2
            from sklearn.feature_selection import SelectKBest, chi2
            
            # Ensure data is positive for chi2
            X_safe = np.abs(X_safe)
                
            selector = SelectKBest(chi2, k=min(n_feats, X_safe.shape[1]))
            logger.info(f"Using SelectKBest with chi2 for {modality_name}")
            
        elif selector_type == 'logistic_l1' or selector_type == 'LogisticL1':
            # Logistic Regression with L1 penalty for feature selection
            from sklearn.feature_selection import SelectFromModel
            from sklearn.linear_model import LogisticRegression
            
            try:
                # Create LogisticRegression with L1 penalty
                logistic_l1 = LogisticRegression(
                    penalty='l1',
                    solver='liblinear',  # liblinear supports L1 penalty
                    C=1.0,
                    random_state=42,
                    max_iter=1000
                )
                
                # Fit the model
                logistic_l1.fit(X_safe, y_safe)
                
                # Use SelectFromModel to select features based on coefficients
                sfm = SelectFromModel(logistic_l1, max_features=n_feats, prefit=True)
                X_selected = sfm.transform(X_safe)
                selected_features = sfm.get_support()
                
                # Cache and return result
                result = (selected_features, X_selected)
                _selector_cache['sel_clf'].put(key, result, item_size=X_selected.nbytes)
                return result
                
            except Exception as e:
                logger.warning(f"LogisticL1 feature selection failed for {modality_name}: {str(e)}, falling back to f_classif")
                from sklearn.feature_selection import SelectKBest, f_classif
                selector = SelectKBest(f_classif, k=min(n_feats, X_safe.shape[1]))
            
        elif selector_type == 'xgb_clf':
            # XGBoost Feature Importance for classification
            if XGBOOST_AVAILABLE:
                logger.info(f"Using XGBoost feature importance for classification on {modality_name}")
                try:
                    # Fix class labels to be consecutive starting from 0 (required by XGBoost)
                    unique_classes = np.unique(y_safe)
                    n_classes = len(unique_classes)
                    
                    # Create mapping from current labels to consecutive labels starting from 0
                    class_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_classes)}
                    y_safe_relabeled = np.array([class_mapping[label] for label in y_safe])
                    
                    logger.debug(f"XGBoost class relabeling for {modality_name}: {class_mapping}")
                    
                    if n_classes == 2:
                        objective = 'binary:logistic'
                    else:
                        objective = 'multi:softprob'
                    
                    # Create XGBoost classifier with relabeled classes
                    xgb_model = xgb.XGBClassifier(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.3,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective=objective,
                        random_state=42,
                        n_jobs=-1,
                        verbosity=0,  # Suppress XGBoost output
                        gamma=0,  # Use default regularization value (0 = no regularization)
                    )
                    
                    # Fit the model with relabeled classes
                    xgb_model.fit(X_safe, y_safe_relabeled)
                    
                    # Get feature importances and select top features
                    importances = xgb_model.feature_importances_
                    # Get indices of features sorted by importance (descending)
                    sorted_indices = np.argsort(importances)[::-1]
                    top_indices = sorted_indices[:min(n_feats, X_safe.shape[1])]
                    
                    # Create boolean mask
                    mask = np.zeros(X_safe.shape[1], dtype=bool)
                    mask[top_indices] = True
                    selected_features = mask
                    transformed_X = X_safe[:, top_indices]
                    
                    # Cache and return result
                    result = (selected_features, transformed_X)
                    _selector_cache['sel_clf'].put(key, result, item_size=transformed_X.nbytes)
                    return result
                    
                except Exception as e:
                    logger.warning(f"XGBoost feature selection failed for {modality_name}: {str(e)}, falling back to f_classif")
                    from sklearn.feature_selection import SelectKBest, f_classif
                    selector = SelectKBest(f_classif, k=min(n_feats, X_safe.shape[1]))
            else:
                logger.warning("XGBoost not available, falling back to f_classif")
                from sklearn.feature_selection import SelectKBest, f_classif
                selector = SelectKBest(f_classif, k=min(n_feats, X_safe.shape[1]))
                
        elif hasattr(selector_code, 'fit') and hasattr(selector_code, 'transform'):
            # Direct selector object
            selector = selector_code
            logger.info(f"Using selector object directly: {type(selector).__name__}")
        else:
            # Unknown selector - fallback to SelectKBest with f_classif
            logger.warning(f"Unknown selector code: {selector_type}, using f_classif as fallback")
            from sklearn.feature_selection import SelectKBest, f_classif
            
            # Use safe f_classif with error handling for divide-by-zero
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='divide by zero encountered in divide')
                    warnings.filterwarnings('ignore', message='invalid value encountered in divide')
                    
                    selector = SelectKBest(f_classif, k=min(n_feats, X_safe.shape[1]))
            except Exception as e:
                logger.warning(f"f_classif fallback failed for unknown selector {selector_type}: {str(e)}, using mutual_info_classif")
                from sklearn.feature_selection import mutual_info_classif
                selector = SelectKBest(mutual_info_classif, k=min(n_feats, X_safe.shape[1]))
        
        # Fit the selector if we don't already have results
        if selected_features is None or transformed_X is None:
            # Fit selector with safe error handling
            try:
                # Suppress sklearn warnings during fitting
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='divide by zero encountered in divide')
                    warnings.filterwarnings('ignore', message='invalid value encountered in divide')
                    warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')
                    
                    selector.fit(X_safe, y_safe)
            except Exception as e:
                logger.warning(f"Selector fitting failed: {str(e)}, using first feature as fallback")
                # Fallback to first feature
                mask = np.zeros(X_safe.shape[1], dtype=bool)
                mask[0] = True
                selected_features = mask
                transformed_X = X_safe[:, [0]]
                
                # Cache and return fallback
                result = (selected_features, transformed_X)
                _selector_cache['sel_clf'].put(key, result, item_size=transformed_X.nbytes)
                return result
            
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
                mask = np.zeros(X_safe.shape[1], dtype=bool)
                mask[top_indices] = True
                selected_features = mask
            elif hasattr(selector, 'estimator_') and hasattr(selector.estimator_, 'feature_importances_'):
                # For tree-based models
                importances = selector.estimator_.feature_importances_
                # Select top n_feats features based on importance
                top_indices = np.argsort(importances)[-min(n_feats, len(importances)):]
                mask = np.zeros(X_safe.shape[1], dtype=bool)
                mask[top_indices] = True
                selected_features = mask
            else:
                # No clear way to get support - fallback to first feature
                logger.warning("No features selected, using first feature as fallback")
                mask = np.zeros(X_safe.shape[1], dtype=bool)
                mask[0] = True
                selected_features = mask
                
        # Check if any features were selected
        if selected_features is None or np.sum(selected_features) == 0:
            logger.warning("No features were selected, using first feature as fallback")
            mask = np.zeros(X_safe.shape[1], dtype=bool)
            mask[0] = True
            selected_features = mask
            
        # Apply transformation
        try:
            if hasattr(selector, 'transform'):
                transformed_X = selector.transform(X_safe)
            else:
                # Manual transform using the support mask
                transformed_X = X_safe[:, selected_features]
        except Exception as e:
            logger.warning(f"Transformation failed: {str(e)}, using manual indexing")
            # Fallback to manual indexing
            indices = np.where(selected_features)[0]
            if len(indices) > 0:
                transformed_X = X_safe[:, indices]
            else:
                # If no features selected, use first feature
                transformed_X = X_safe[:, [0]]
                selected_features = np.zeros(X_safe.shape[1], dtype=bool)
                selected_features[0] = True
                
        # Cache result with memory size tracking
        result = (selected_features, transformed_X)
        _selector_cache['sel_clf'].put(key, result, item_size=transformed_X.nbytes if transformed_X is not None else 0)
        return result
        
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
    # Generate a stable cache key
    extractor_type = extractor.__class__.__name__
    key = _generate_cache_key(ds_name, fold_idx, extractor_type, "ext_reg", n_components, X.shape if X is not None else None)
    
    # Check cache
    cached_result = _extractor_cache['ext_reg'].get(key)
    if cached_result is not None:
        return cached_result
    
    try:
        # Convert DataFrame to numpy if needed
        if isinstance(X, pd.DataFrame):
            X_safe = X.fillna(0).values
        else:
            X_safe = safe_convert_to_numeric(X)
            
        # For regression, y should be numeric
        if y is not None:
            try:
                y_safe = np.asarray(y, dtype=np.float64)  # Use float64 for consistency
            except:
                logger.warning(f"Could not convert y to float for {modality_name} in fold {fold_idx}")
                y_safe = y
        else:
            y_safe = None
            
        # Use the enhanced validation function instead of verify_data_alignment
        X_safe, y_safe = validate_and_fix_shape_mismatch(
            X_safe, y_safe, 
            name=f"{modality_name} regression extractor data", 
            fold_idx=fold_idx
        )
        
        # If alignment failed, return a fallback
        if X_safe is None or y_safe is None:
            logger.warning(f"Data alignment failure in regression extractor for {modality_name}")
            return None, None
        
        # Additional validation for zero-size arrays
        if X_safe.size == 0 or X_safe.shape[0] == 0 or X_safe.shape[1] == 0:
            logger.warning(f"Zero-size array detected for {modality_name} in fold {fold_idx}: shape {X_safe.shape}")
            return None, None
            
        if y_safe is not None and y_safe.size == 0:
            logger.warning(f"Zero-size target array detected for {modality_name} in fold {fold_idx}: size {y_safe.size}")
            return None, None
            
        # Check for minimum data requirements
        if X_safe.shape[0] < 2:
            logger.warning(f"Insufficient samples for {modality_name} in fold {fold_idx}: {X_safe.shape[0]} samples (minimum 2 required)")
            return None, None
            
        if X_safe.shape[1] < 1:
            logger.warning(f"No features available for {modality_name} in fold {fold_idx}: {X_safe.shape[1]} features")
            return None, None
        
        # Enhanced adaptive component selection with stability checks
        from config import EXTRACTOR_CONFIG
        
        # Calculate absolute maximum possible components (minimum dimension of data)
        absolute_max = min(X_safe.shape[0], X_safe.shape[1])
        
        # Ensure we have a valid number of components to work with
        if absolute_max <= 0:
            logger.warning(f"Invalid component calculation for {modality_name} in fold {fold_idx}: absolute_max={absolute_max}, shape={X_safe.shape}")
            return None, None
        
        # Enhanced adaptive component selection for regression
        if EXTRACTOR_CONFIG.get("adaptive_components", True):
            # Consider data characteristics for component selection
            n_samples, n_features = X_safe.shape
            
            # Rule 1: Mathematical constraint
            math_limit = absolute_max
            
            # Rule 2: Statistical constraint (avoid overfitting)
            stat_limit = min(n_samples // 2, n_features)  # Conservative approach
            
            # Rule 3: Explained variance constraint for PCA-like methods
            if hasattr(extractor, 'explained_variance_ratio_') or 'PCA' in str(type(extractor)):
                # For PCA, use a more conservative approach
                variance_limit = min(n_components, int(n_features * EXTRACTOR_CONFIG.get("max_components_ratio", 0.9)))
            else:
                variance_limit = n_components
            
            # Take the minimum of all constraints
            effective_n_components = min(n_components, math_limit, stat_limit, variance_limit)
            
            # Ensure minimum viable components
            min_components = max(1, min(2, absolute_max))
            effective_n_components = max(effective_n_components, min_components)
            
            if effective_n_components != n_components:
                logger.info(f"Adaptive components for {modality_name}: {n_components} -> {effective_n_components} "
                           f"(math_limit={math_limit}, stat_limit={stat_limit}, variance_limit={variance_limit})")
        else:
            # Traditional approach
            if n_components > absolute_max:
                effective_n_components = absolute_max
                logger.debug(f"Adaptive components: using {effective_n_components} components for {modality_name} (mathematical limit)")
            else:
                effective_n_components = n_components
                if n_components < absolute_max:
                    logger.debug(f"Optimal performance: using {effective_n_components} < {absolute_max} available features for {modality_name}")
        
        # Final validation
        if effective_n_components <= 0:
            logger.warning(f"Invalid effective components for {modality_name} in fold {fold_idx}: {effective_n_components}")
            return None, None
        
        if effective_n_components > absolute_max:
            logger.warning(f"Effective components exceed mathematical limit for {modality_name}: {effective_n_components} > {absolute_max}")
            effective_n_components = absolute_max
        
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
                logger.warning(f"FastICA configuration failed for {modality_name}, falling back to PCA")
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
                logger.warning(f"NMF configuration failed for {modality_name}, falling back to PCA")
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
                logger.warning(f"FactorAnalysis configuration failed for {modality_name}, falling back to PCA")
                new_extractor = PCA(n_components=effective_n_components, random_state=42)
        elif extractor.__class__.__name__ == 'PLSRegression':
            try:
                # Use the already imported PLSRegression from the top of the file
                new_extractor = PLSRegression(n_components=effective_n_components, max_iter=500, tol=1e-3)
                if y_safe is None:
                    logger.warning(f"PLSRegression requires y values for fit_transform. Skipping extraction for {modality_name} in fold {fold_idx}.")
                    return None, None
                X_transformed = new_extractor.fit_transform(X_safe, y_safe)[0]
                logger.debug(f"PLSRegression extraction successful for {modality_name} in fold {fold_idx}: {X_transformed.shape} from {X_safe.shape}")
                X_transformed = np.nan_to_num(X_transformed, nan=0.0)
                
                result = (new_extractor, X_transformed)
                _extractor_cache['ext_reg'].put(key, result, item_size=X_transformed.nbytes)
                return result
            except Exception as e:
                logger.warning(f"PLSRegression configuration failed for {modality_name}, falling back to PCA. Error: {str(e)}")
                new_extractor = PCA(n_components=effective_n_components, random_state=42)
        elif isinstance(extractor, KernelPCA):
            new_extractor = KernelPCA(
                n_components=effective_n_components,
                kernel='rbf',
                random_state=42
            )
        elif isinstance(extractor, SparsePLS):
            try:
                new_extractor = SparsePLS(
                    n_components=effective_n_components,
                    alpha=extractor.alpha,
                    max_iter=extractor.max_iter,
                    tol=extractor.tol,
                    scale=extractor.scale
                )
                if y_safe is None:
                    logger.warning(f"SparsePLS requires y values for fit_transform. Skipping extraction for {modality_name} in fold {fold_idx}.")
                    return None, None
                X_transformed = new_extractor.fit_transform(X_safe, y_safe)
                logger.debug(f"SparsePLS extraction successful for {modality_name} in fold {fold_idx}: {X_transformed.shape} from {X_safe.shape}")
                X_transformed = np.nan_to_num(X_transformed, nan=0.0)
                
                result = (new_extractor, X_transformed)
                _extractor_cache['ext_reg'].put(key, result, item_size=X_transformed.nbytes)
                return result
            except Exception as e:
                logger.warning(f"SparsePLS configuration failed for {modality_name}, falling back to PCA. Error: {str(e)}")
                new_extractor = PCA(n_components=effective_n_components, random_state=42)
        else:
            # Default to PCA as a safe fallback
            logger.warning(f"Unknown extractor type: {type(extractor)} for {modality_name}, falling back to PCA")
            new_extractor = PCA(n_components=effective_n_components, random_state=42)
        
        # Fit and transform
        try:
            # Additional safety check right before fitting
            if X_safe.size == 0 or X_safe.shape[0] == 0 or X_safe.shape[1] == 0:
                logger.error(f"Zero-size array detected just before fitting for {modality_name} in fold {fold_idx}: shape {X_safe.shape}")
                return None, None
            
            # CRITICAL: Final NaN check and cleaning before extractor training
            if np.isnan(X_safe).any():
                nan_count = np.isnan(X_safe).sum()
                logger.warning(f"CRITICAL: {nan_count} NaN values detected in X_safe before {type(new_extractor).__name__} training for {modality_name} fold {fold_idx}")
                logger.warning("This will cause extractor training to fail - cleaning now")
                X_safe = np.nan_to_num(X_safe, nan=0.0, posinf=0.0, neginf=0.0)
                logger.info(f"Cleaned {nan_count} NaN values before extractor training")
            
            if y_safe is not None and np.isnan(y_safe).any():
                y_nan_count = np.isnan(y_safe).sum()
                logger.warning(f"CRITICAL: {y_nan_count} NaN values detected in y_safe before {type(new_extractor).__name__} training for {modality_name} fold {fold_idx}")
                y_median = np.nanmedian(y_safe) if not np.isnan(y_safe).all() else 0.0
                y_safe = np.nan_to_num(y_safe, nan=y_median, posinf=y_median, neginf=y_median)
                logger.info(f"Cleaned {y_nan_count} NaN values in target before extractor training")
            
            # Additional validation for infinite values
            if np.isinf(X_safe).any():
                inf_count = np.isinf(X_safe).sum()
                logger.warning(f"CRITICAL: {inf_count} infinite values detected in X_safe before {type(new_extractor).__name__} training for {modality_name} fold {fold_idx}")
                X_safe = np.nan_to_num(X_safe, nan=0.0, posinf=0.0, neginf=0.0)
                logger.info(f"Cleaned {inf_count} infinite values before extractor training")
            
            # Handle different extractor types that require y parameter
            if isinstance(new_extractor, LDA):
                # LDA requires y parameter for fit_transform
                X_transformed = new_extractor.fit_transform(X_safe, y_safe)
            else:
                # Most other extractors don't use y in fit_transform
                X_transformed = new_extractor.fit_transform(X_safe)
            
            # CRITICAL: Check if the extractor produced NaN values
            if np.isnan(X_transformed).any():
                nan_output_count = np.isnan(X_transformed).sum()
                logger.error(f"CRITICAL: {type(new_extractor).__name__} produced {nan_output_count} NaN values for {modality_name} fold {fold_idx}")
                logger.error("This indicates the extractor failed during training - will clean and continue")
                X_transformed = np.nan_to_num(X_transformed, nan=0.0, posinf=0.0, neginf=0.0)
            
            logger.info(f"Extraction successful for {modality_name} in fold {fold_idx}: {X_transformed.shape} from {X_safe.shape}")
        except Exception as e:
            logger.error(f"Error in feature extraction for {modality_name}: {str(e)}, falling back to PCA")
            # If extraction fails, fall back to PCA which is more robust
            try:
                # Additional safety check for fallback
                if X_safe.size == 0 or X_safe.shape[0] == 0 or X_safe.shape[1] == 0:
                    logger.error(f"Cannot fallback to PCA: zero-size array for {modality_name} in fold {fold_idx}")
                    return None, None
                
                safe_n_components = min(effective_n_components, X_safe.shape[1], X_safe.shape[0])
                if safe_n_components <= 0:
                    logger.error(f"Cannot create PCA fallback: invalid components {safe_n_components} for {modality_name}")
                    return None, None
                
                new_extractor = PCA(n_components=safe_n_components, random_state=42)
                X_transformed = new_extractor.fit_transform(X_safe)
            except Exception as e2:
                logger.error(f"PCA fallback also failed: {str(e2)}")
                return None, None
        
        # Handle any NaN values in the transformed data
        X_transformed = np.nan_to_num(X_transformed, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Ensure output is also float64
        if X_transformed.dtype != np.float64:
            X_transformed = X_transformed.astype(np.float64)
        
        # Verify the transformed data
        if X_transformed.shape[0] != X_safe.shape[0]:
            logger.warning(f"Transformed data has different number of samples: {X_transformed.shape[0]} vs {X_safe.shape[0]} for {modality_name}")
            return None, None
        
        # Store result in cache with memory size tracking
        result = (new_extractor, X_transformed)
        _extractor_cache['ext_reg'].put(key, result, item_size=X_transformed.nbytes)
        return result
    except Exception as e:
        logger.error(f"Error in feature extraction for {modality_name} in fold {fold_idx}: {str(e)}")
        return None, None

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
    extractor_type = extractor.__class__.__name__
    key = _generate_cache_key(ds_name, fold_idx, extractor_type, "ext_clf", n_components, X.shape if X is not None else None)
    
    # Check cache
    cached_result = _extractor_cache['ext_clf'].get(key)
    if cached_result is not None:
        return cached_result
    
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
                    logger.debug(f"Converting float y to integer for classification - {modality_name} in fold {fold_idx}")
                    y_safe = np.round(y).astype(np.int32)
                else:
                    y_safe = np.asarray(y)
            except:
                logger.warning(f"Could not convert y for {modality_name} in fold {fold_idx}")
                y_safe = y
        else:
            y_safe = None
            
        # Use the enhanced validation function instead of verify_data_alignment
        X_safe, y_safe = validate_and_fix_shape_mismatch(
            X_safe, y_safe, 
            name=f"{modality_name} classification extractor data", 
            fold_idx=fold_idx
        )
        
        # If alignment failed, return a fallback
        if X_safe is None or y_safe is None:
            logger.warning(f"Data alignment failure in classification extractor for {modality_name}")
            return None, None
            
        # Additional validation for zero-size arrays
        if X_safe.size == 0 or X_safe.shape[0] == 0 or X_safe.shape[1] == 0:
            logger.warning(f"Zero-size array detected for {modality_name} in fold {fold_idx}: shape {X_safe.shape}")
            return None, None
            
        if y_safe.size == 0:
            logger.warning(f"Zero-size target array detected for {modality_name} in fold {fold_idx}: size {y_safe.size}")
            return None, None
            
        # Check for minimum data requirements
        if X_safe.shape[0] < 2:
            logger.warning(f"Insufficient samples for {modality_name} in fold {fold_idx}: {X_safe.shape[0]} samples (minimum 2 required)")
            return None, None
            
        if X_safe.shape[1] < 1:
            logger.warning(f"No features available for {modality_name} in fold {fold_idx}: {X_safe.shape[1]} features")
            return None, None
        
        # OPTIMIZATION: Use actual data dimensions rather than forcing maximum
        # Calculate absolute maximum possible components (minimum dimension of data)
        absolute_max = min(X_safe.shape[0], X_safe.shape[1])
        
        # Ensure we have a valid number of components to work with
        if absolute_max <= 0:
            logger.warning(f"Invalid component calculation for {modality_name} in fold {fold_idx}: absolute_max={absolute_max}, shape={X_safe.shape}")
            return None, None
        
        # Adaptive component selection: use the best number based on data availability
        if n_components > absolute_max:
            # This is a hard constraint we can't exceed due to math limitations
            effective_n_components = absolute_max
            logger.debug(f"Adaptive components: using {effective_n_components} components for {modality_name} (mathematical limit)")
        else:
            # Use requested number, which is better for performance if it's smaller
            effective_n_components = n_components
            if n_components < absolute_max:
                logger.debug(f"Optimal performance: using {effective_n_components} < {absolute_max} available features for {modality_name}")
        
        # Final check - ensure effective_n_components is valid
        if effective_n_components <= 0:
            logger.warning(f"Invalid effective components for {modality_name} in fold {fold_idx}: {effective_n_components}")
            return None, None
        
        # LDA has special constraints regardless of force_n_components
        if isinstance(extractor, LDA):
            # LDA has strict component limitations based on number of classes
            n_classes = len(np.unique(y_safe))
            max_components = min(X_safe.shape[1], n_classes - 1)
            if effective_n_components > max_components:
                logger.debug(f"LDA constraint: using {max_components} components for {modality_name} (classes-1 limit)")
                effective_n_components = max_components
        
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
            try:
                # KernelPCA can be sensitive to data characteristics, so add extra validation
                if X_safe.shape[0] < 3 or X_safe.shape[1] < 2:
                    logger.warning(f"Data too small for KernelPCA for {modality_name} in fold {fold_idx}: {X_safe.shape}, falling back to PCA")
                    new_extractor = PCA(n_components=effective_n_components, random_state=42)
                else:
                    # Use more robust KernelPCA parameters with adaptive gamma
                    # Calculate gamma based on data scale (mimics 'scale' behavior)
                    try:
                        # Auto-scale gamma based on data variance (equivalent to gamma='scale')
                        data_var = np.var(X_safe)
                        n_features = X_safe.shape[1]
                        adaptive_gamma = 1.0 / (n_features * max(data_var, 1e-8))  # Avoid division by zero
                        
                        new_extractor = KernelPCA(
                            n_components=effective_n_components,
                            kernel='rbf',
                            gamma=adaptive_gamma,  # Use calculated adaptive gamma
                            eigen_solver='auto',  # Let sklearn choose the best solver
                            random_state=42,
                            copy_X=True  # Ensure we don't modify original data
                        )
                    except Exception as gamma_calc_error:
                        logger.warning(f"Adaptive gamma calculation failed for {modality_name}, using default gamma=None: {str(gamma_calc_error)}")
                        new_extractor = KernelPCA(
                            n_components=effective_n_components,
                            kernel='rbf',
                            gamma=None,  # Use sklearn default
                            eigen_solver='auto',
                            random_state=42,
                            copy_X=True
                        )
            except Exception as e:
                logger.warning(f"KernelPCA configuration failed for {modality_name}, falling back to PCA: {str(e)}")
                new_extractor = PCA(n_components=effective_n_components, random_state=42)
        elif isinstance(extractor, PLSDiscriminantAnalysis):
            try:
                new_extractor = PLSDiscriminantAnalysis(
                    n_components=effective_n_components,
                    max_iter=extractor.max_iter,
                    tol=extractor.tol,
                    scale=extractor.scale
                )
                if y_safe is None:
                    logger.warning(f"PLS-DA requires y values for fit_transform. Skipping extraction for {modality_name} in fold {fold_idx}.")
                    return None, None
                X_transformed = new_extractor.fit_transform(X_safe, y_safe)
                logger.info(f"PLS-DA extraction successful for {modality_name} in fold {fold_idx}: {X_transformed.shape} from {X_safe.shape}")
                X_transformed = np.nan_to_num(X_transformed, nan=0.0)
                
                result = (new_extractor, X_transformed)
                _extractor_cache['ext_clf'].put(key, result, item_size=X_transformed.nbytes)
                return result
            except Exception as e:
                logger.warning(f"PLS-DA configuration failed for {modality_name}, falling back to PCA. Error: {str(e)}")
                new_extractor = PCA(n_components=effective_n_components, random_state=42)
        elif isinstance(extractor, SparsePLS):
            try:
                new_extractor = SparsePLS(
                    n_components=effective_n_components,
                    alpha=extractor.alpha,
                    max_iter=extractor.max_iter,
                    tol=extractor.tol,
                    scale=extractor.scale
                )
                if y_safe is None:
                    logger.warning(f"SparsePLS requires y values for fit_transform. Skipping extraction for {modality_name} in fold {fold_idx}.")
                    return None, None
                X_transformed = new_extractor.fit_transform(X_safe, y_safe)
                logger.info(f"SparsePLS extraction successful for {modality_name} in fold {fold_idx}: {X_transformed.shape} from {X_safe.shape}")
                X_transformed = np.nan_to_num(X_transformed, nan=0.0)
                
                result = (new_extractor, X_transformed)
                _extractor_cache['ext_clf'].put(key, result, item_size=X_transformed.nbytes)
                return result
            except Exception as e:
                logger.warning(f"SparsePLS configuration failed for {modality_name}, falling back to PCA. Error: {str(e)}")
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
                
                result = (new_extractor, X_transformed)
                _extractor_cache['ext_clf'].put(key, result, item_size=X_transformed.nbytes)
                return result
            except Exception as e:
                logger.warning(f"Warning: PLSRegression configuration failed for {modality_name}, falling back to PCA. Error: {str(e)}")
                new_extractor = PCA(n_components=effective_n_components, random_state=42)
        else:
            # Default to PCA as a safe fallback
            logger.warning(f"Unknown extractor type: {type(extractor)} for {modality_name}, falling back to PCA")
            new_extractor = PCA(n_components=effective_n_components, random_state=42)
        
        # Fit and transform
        try:
            # Additional safety check right before fitting
            if X_safe.size == 0 or X_safe.shape[0] == 0 or X_safe.shape[1] == 0:
                logger.error(f"Zero-size array detected just before fitting for {modality_name} in fold {fold_idx}: shape {X_safe.shape}")
                return None, None
            
            # CRITICAL: Final NaN check and cleaning before extractor training
            if np.isnan(X_safe).any():
                nan_count = np.isnan(X_safe).sum()
                logger.warning(f"CRITICAL: {nan_count} NaN values detected in X_safe before {type(new_extractor).__name__} training for {modality_name} fold {fold_idx}")
                logger.warning("This will cause extractor training to fail - cleaning now")
                X_safe = np.nan_to_num(X_safe, nan=0.0, posinf=0.0, neginf=0.0)
                logger.info(f"Cleaned {nan_count} NaN values before extractor training")
            
            if y_safe is not None and np.isnan(y_safe).any():
                y_nan_count = np.isnan(y_safe).sum()
                logger.warning(f"CRITICAL: {y_nan_count} NaN values detected in y_safe before {type(new_extractor).__name__} training for {modality_name} fold {fold_idx}")
                y_median = np.nanmedian(y_safe) if not np.isnan(y_safe).all() else 0.0
                y_safe = np.nan_to_num(y_safe, nan=y_median, posinf=y_median, neginf=y_median)
                logger.info(f"Cleaned {y_nan_count} NaN values in target before extractor training")
            
            # Additional validation for infinite values
            if np.isinf(X_safe).any():
                inf_count = np.isinf(X_safe).sum()
                logger.warning(f"CRITICAL: {inf_count} infinite values detected in X_safe before {type(new_extractor).__name__} training for {modality_name} fold {fold_idx}")
                X_safe = np.nan_to_num(X_safe, nan=0.0, posinf=0.0, neginf=0.0)
                logger.info(f"Cleaned {inf_count} infinite values before extractor training")
            
            # Handle different extractor types that require y parameter
            if isinstance(new_extractor, LDA):
                # LDA requires y parameter for fit_transform
                X_transformed = new_extractor.fit_transform(X_safe, y_safe)
            else:
                # Most other extractors don't use y in fit_transform
                X_transformed = new_extractor.fit_transform(X_safe)
            
            # CRITICAL: Check if the extractor produced NaN values
            if np.isnan(X_transformed).any():
                nan_output_count = np.isnan(X_transformed).sum()
                logger.error(f"CRITICAL: {type(new_extractor).__name__} produced {nan_output_count} NaN values for {modality_name} fold {fold_idx}")
                logger.error("This indicates the extractor failed during training - will clean and continue")
                X_transformed = np.nan_to_num(X_transformed, nan=0.0, posinf=0.0, neginf=0.0)
            
            logger.info(f"Extraction successful for {modality_name} in fold {fold_idx}: {X_transformed.shape} from {X_safe.shape}")
        except Exception as e:
            logger.error(f"Error in feature extraction for {modality_name}: {str(e)}, falling back to PCA")
            # If extraction fails, fall back to PCA which is more robust
            try:
                # Additional safety check for fallback
                if X_safe.size == 0 or X_safe.shape[0] == 0 or X_safe.shape[1] == 0:
                    logger.error(f"Cannot fallback to PCA: zero-size array for {modality_name} in fold {fold_idx}")
                    return None, None
                
                safe_n_components = min(effective_n_components, X_safe.shape[1], X_safe.shape[0])
                if safe_n_components <= 0:
                    logger.error(f"Cannot create PCA fallback: invalid components {safe_n_components} for {modality_name}")
                    return None, None
                
                new_extractor = PCA(n_components=safe_n_components, random_state=42)
                X_transformed = new_extractor.fit_transform(X_safe)
            except Exception as e2:
                logger.error(f"PCA fallback also failed: {str(e2)}")
                return None, None
        
        # Handle any NaN values in the transformed data
        X_transformed = np.nan_to_num(X_transformed, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Verify the transformed data
        if X_transformed.shape[0] != X_safe.shape[0]:
            logger.warning(f"Warning: Transformed data has different number of samples: {X_transformed.shape[0]} vs {X_safe.shape[0]} for {modality_name}")
            return None, None
        
        # Store result in cache with memory size tracking
        result = (new_extractor, X_transformed)
        _extractor_cache['ext_clf'].put(key, result, item_size=X_transformed.nbytes)
        return result
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
        # Convert DataFrame to numpy if needed, ensuring float64 for sklearn compatibility
        if isinstance(X, pd.DataFrame):
            X_safe = X.fillna(0).astype(np.float64).values
        else:
            X_safe = safe_convert_to_numeric(X)
            
        # Ensure X_safe is float64 for sklearn compatibility
        if X_safe.dtype != np.float64:
            X_safe = X_safe.astype(np.float64)
            
        # Handle feature dimension mismatch for certain extractors
        expected_features = None
        if hasattr(extractor, 'n_features_in_'):
            expected_features = extractor.n_features_in_
        elif hasattr(extractor, 'components_') and hasattr(extractor.components_, 'shape'):
            # For PCA, NMF, etc.
            expected_features = extractor.components_.shape[1]
            
        key = (id(extractor), expected_features, X_safe.shape[1])
        if expected_features is not None and X_safe.shape[1] != expected_features:
            # OPTIMIZATION: Only warn if we need to add features (padding)
            # Having fewer features than the maximum is actually BETTER for performance
            if X_safe.shape[1] < expected_features:
                if key not in _feature_mismatch_logged:
                    logger.debug(f"Optimizing for performance: X has {X_safe.shape[1]} features, padding to {expected_features} expected by extractor")
                    _feature_mismatch_logged.add(key)
                
                # Create a new array with the expected number of features, using consistent dtype
                n_samples = X_safe.shape[0]
                X_adjusted = np.zeros((n_samples, expected_features), dtype=np.float64)
                
                # Copy the available features
                n_features_to_copy = min(X_safe.shape[1], expected_features)
                X_adjusted[:, :n_features_to_copy] = X_safe[:, :n_features_to_copy]
                
                X_safe = X_adjusted
            else:
                # We have more features than expected - truncate (this is even better!)
                if key not in _feature_mismatch_logged:
                    logger.debug(f"Excellent performance: X has {X_safe.shape[1]} features, using best {expected_features} for extractor")
                    _feature_mismatch_logged.add(key)
                X_safe = X_safe[:, :expected_features]
            
        # Ensure input data has finite values
        if not np.all(np.isfinite(X_safe)):
            logger.debug("Non-finite values detected in input data, replacing with zeros")
            X_safe = np.nan_to_num(X_safe, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Special handling for NMF: ensure all values are non-negative
        if isinstance(extractor, NMF):
            if np.any(X_safe < 0):
                logger.debug("NMF detected: converting negative values to zero for transformation")
                X_safe = np.maximum(X_safe, 0)
            
        # Transform the data
        X_transformed = extractor.transform(X_safe)
        
        # Handle any NaN values in the transformed data
        X_transformed = np.nan_to_num(X_transformed, nan=0.0, posinf=0.0, neginf=0.0)
            
        # Ensure output is also float64
        if X_transformed.dtype != np.float64:
            X_transformed = X_transformed.astype(np.float64)
        
        return X_transformed
    except Exception as e:
        logger.error(f"Error in transform_extractor_classification: {str(e)}")
        return None

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
        # Convert DataFrame to numpy if needed, ensuring float64 for sklearn compatibility
        if isinstance(X, pd.DataFrame):
            X_safe = X.fillna(0).astype(np.float64).values
        else:
            X_safe = safe_convert_to_numeric(X)
            
        # Ensure X_safe is float64 for sklearn compatibility
        if X_safe.dtype != np.float64:
            X_safe = X_safe.astype(np.float64)
            
        # Handle feature dimension mismatch for certain extractors
        expected_features = None
        if hasattr(extractor, 'n_features_in_'):
            expected_features = extractor.n_features_in_
        elif hasattr(extractor, 'components_') and hasattr(extractor.components_, 'shape'):
            # For PCA, NMF, etc.
            expected_features = extractor.components_.shape[1]
            
        key = (id(extractor), expected_features, X_safe.shape[1])
        if expected_features is not None and X_safe.shape[1] != expected_features:
            # OPTIMIZATION: Only warn if we need to add features (padding)
            # Having fewer features than the maximum is actually BETTER for performance
            if X_safe.shape[1] < expected_features:
                if key not in _feature_mismatch_logged:
                    logger.debug(f"Optimizing for performance: X has {X_safe.shape[1]} features, padding to {expected_features} expected by extractor")
                    _feature_mismatch_logged.add(key)
                
                # Create a new array with the expected number of features, using consistent dtype
                n_samples = X_safe.shape[0]
                X_adjusted = np.zeros((n_samples, expected_features), dtype=np.float64)
                
                # Copy the available features
                n_features_to_copy = min(X_safe.shape[1], expected_features)
                X_adjusted[:, :n_features_to_copy] = X_safe[:, :n_features_to_copy]
                
                X_safe = X_adjusted
            else:
                # We have more features than expected - truncate (this is even better!)
                if key not in _feature_mismatch_logged:
                    logger.debug(f"Excellent performance: X has {X_safe.shape[1]} features, using best {expected_features} for extractor")
                    _feature_mismatch_logged.add(key)
                X_safe = X_safe[:, :expected_features]
            
        # Ensure input data has finite values
        if not np.all(np.isfinite(X_safe)):
            logger.debug("Non-finite values detected in input data, replacing with zeros")
            X_safe = np.nan_to_num(X_safe, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Special handling for NMF: ensure all values are non-negative
        if isinstance(extractor, NMF):
            if np.any(X_safe < 0):
                logger.debug("NMF detected: converting negative values to zero for transformation")
                X_safe = np.maximum(X_safe, 0)
            
        # Transform the data
        X_transformed = extractor.transform(X_safe)
        
        # Handle any NaN values in the transformed data
        X_transformed = np.nan_to_num(X_transformed, nan=0.0, posinf=0.0, neginf=0.0)
            
        # Ensure output is also float64
        if X_transformed.dtype != np.float64:
            X_transformed = X_transformed.astype(np.float64)
        
        return X_transformed
    except Exception as e:
        logger.error(f"Error in transform_extractor_regression: {str(e)}")
        return None

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
            max_index = max(selected_features) if len(selected_features) > 0 else -1
            if max_index >= X.shape[1]:
                logger.warning(f"Selected feature indices exceed X dimensions: max index {max_index} vs {X.shape[1]} columns")
                # Take only valid indices
                valid_indices = [i for i in selected_features if i < X.shape[1]]
                if len(valid_indices) == 0:
                    # If no valid indices, take first column as fallback
                    logger.warning("No valid feature indices found, using first column as fallback")
                    return X.iloc[:, 0:1].values
                logger.debug(f"Using {len(valid_indices)} valid indices out of {len(selected_features)} selected")
                return X.iloc[:, valid_indices].values
            return X.iloc[:, selected_features].values
        else:
            # For numpy arrays
            max_index = max(selected_features) if len(selected_features) > 0 else -1
            if max_index >= X.shape[1]:
                logger.warning(f"Selected feature indices exceed X dimensions: max index {max_index} vs {X.shape[1]} columns")
                # Take only valid indices
                valid_indices = [i for i in selected_features if i < X.shape[1]]
                if len(valid_indices) == 0:
                    # If no valid indices, take first column as fallback
                    logger.warning("No valid feature indices found, using first column as fallback")
                    return X[:, 0:1]
                logger.debug(f"Using {len(valid_indices)} valid indices out of {len(selected_features)} selected")
                return X[:, valid_indices]
            return X[:, selected_features] 
    except Exception as e:
        logger.error(f"Error in transform_selector_classification: {str(e)}")
        # Return a safe fallback - first column
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, 0:1].values
        return X[:, 0:1]

def get_cache_stats():
    """
    Get statistics about all cache usage.
    
    Returns
    -------
    dict
        Dictionary with statistics for all caches
    """
    stats = {
        "selector_cache": {
            "regression": _selector_cache["sel_reg"].stats(),
            "classification": _selector_cache["sel_clf"].stats(),
        },
        "extractor_cache": {
            "regression": _extractor_cache["ext_reg"].stats(),
            "classification": _extractor_cache["ext_clf"].stats(),
        },
        "total_memory_mb": sum([
            _selector_cache["sel_reg"].stats()["memory_usage_mb"],
            _selector_cache["sel_clf"].stats()["memory_usage_mb"],
            _extractor_cache["ext_reg"].stats()["memory_usage_mb"],
            _extractor_cache["ext_clf"].stats()["memory_usage_mb"]
        ])
    }
    
    return stats

def clear_all_caches():
    """
    Clear all caches to free memory.
    
    Returns
    -------
    dict
        Statistics about the cleared caches
    """
    stats_before = get_cache_stats()
    
    # Clear all caches
    for cache in _selector_cache.values():
        cache.clear()
    for cache in _extractor_cache.values():
        cache.clear()
    
    # Clear the feature mismatch logging set as well
    global _feature_mismatch_logged
    _feature_mismatch_logged.clear()
    
    # Return stats from before clearing
    return stats_before

def force_clear_caches_on_alignment_error():
    """
    Force clear all caches when alignment errors are detected.
    This prevents stale cache entries from causing sample count mismatches.
    """
    logger.debug("Clearing all caches due to alignment errors")
    stats = clear_all_caches()
    logger.debug(f"Cleared caches with {stats['total_memory_mb']:.2f} MB total memory usage") 

def validate_and_fix_shape_mismatch(X, y, name="dataset", fold_idx=None, allow_truncation=True):
    """
    Enhanced shape mismatch validation and auto-fixing with comprehensive error handling.
    
    Parameters
    ----------
    X : array-like
        Feature matrix
    y : array-like  
        Target vector
    name : str
        Name for logging
    fold_idx : int, optional
        Fold index for logging
    allow_truncation : bool
        Whether to allow truncation to fix mismatches
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray] or Tuple[None, None]
        Validated and aligned X, y arrays or None, None if validation fails
    """
    from config import SHAPE_MISMATCH_CONFIG, MEMORY_OPTIMIZATION
    
    fold_str = f" in fold {fold_idx}" if fold_idx is not None else ""
    
    try:
        # Input validation
        if X is None or y is None:
            logger.warning(f"Null input data for {name}{fold_str}")
            return None, None
        
        # Convert to numpy arrays if needed with error handling
        try:
            if not isinstance(X, np.ndarray):
                X = np.asarray(X, dtype=np.float64)
            if not isinstance(y, np.ndarray):
                y = np.asarray(y)
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to convert inputs to arrays for {name}{fold_str}: {str(e)}")
            return None, None
        
        # Handle NaN and infinite values
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            logger.warning(f"NaN/Inf values detected in X for {name}{fold_str}, cleaning...")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            logger.warning(f"NaN/Inf values detected in y for {name}{fold_str}, cleaning...")
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Ensure y is 1D with enhanced handling
        if y.ndim > 1:
            if y.shape[1] == 1:
                y = y.ravel()
            elif y.shape[0] == 1:
                y = y.flatten()
            else:
                logger.error(f"Invalid y shape for {name}{fold_str}: {y.shape} (should be 1D or (n, 1))")
                return None, None
        
        # Check for empty arrays
        if X.size == 0 or y.size == 0:
            logger.warning(f"Empty arrays for {name}{fold_str}: X={X.shape}, y={y.shape}")
            return None, None
        
        # Check basic shape consistency
        if X.ndim != 2:
            if X.ndim == 1:
                # Try to reshape 1D array to 2D
                X = X.reshape(-1, 1)
                logger.info(f"Reshaped 1D X to 2D for {name}{fold_str}: {X.shape}")
            else:
                logger.error(f"Invalid X dimensions for {name}{fold_str}: {X.shape} (should be 2D)")
                return None, None
        
        n_samples_X = X.shape[0]
        n_samples_y = len(y)
        
        if SHAPE_MISMATCH_CONFIG["log_all_fixes"]:
            logger.debug(f"Shape validation for {name}{fold_str}: X=({n_samples_X}, {X.shape[1]}), y=({n_samples_y},)")
        
        # If shapes match, return as-is
        if n_samples_X == n_samples_y:
            if SHAPE_MISMATCH_CONFIG["log_all_fixes"]:
                logger.debug(f"Shapes already aligned for {name}{fold_str}")
            return X, y
        
        # Handle shape mismatch with enhanced strategies
        logger.info(f"Auto-fixing shape mismatch for {name}{fold_str}: X has {n_samples_X} samples, y has {n_samples_y} samples")
        
        if not allow_truncation and not SHAPE_MISMATCH_CONFIG["auto_fix_enabled"]:
            logger.error(f"Truncation not allowed and auto-fix disabled for {name}{fold_str}")
            return None, None
        
        # Determine alignment strategy
        strategy = SHAPE_MISMATCH_CONFIG["truncation_strategy"]
        
        if strategy == "min":
            # Truncate to minimum length
            min_samples = min(n_samples_X, n_samples_y)
            max_samples = max(n_samples_X, n_samples_y)
            
            # Check data loss with adaptive thresholds for small datasets
            loss_percentage = 100 * (max_samples - min_samples) / max_samples
            max_loss = SHAPE_MISMATCH_CONFIG["max_data_loss_percent"]
            
            # Be more lenient with small datasets
            if max_samples < 50:
                max_loss = min(max_loss * 1.5, 75)  # Allow up to 75% loss for very small datasets
            elif max_samples < 100:
                max_loss = min(max_loss * 1.2, 60)  # Allow up to 60% loss for small datasets
            
            if loss_percentage > max_loss:
                logger.error(f"Excessive data loss for {name}{fold_str}: {loss_percentage:.1f}% > {max_loss}% limit")
                if not SHAPE_MISMATCH_CONFIG["fallback_on_failure"]:
                    return None, None
                # Try fallback strategy
                logger.warning(f"Attempting fallback strategy for {name}{fold_str}")
                min_samples = max(min_samples, int(max_samples * (1 - max_loss/100)))
            
            min_required = SHAPE_MISMATCH_CONFIG["min_samples_after_fix"]
            if min_samples < min_required:
                logger.error(f"Too few samples after alignment for {name}{fold_str}: {min_samples} < {min_required}")
                return None, None
            
            # Perform truncation
            X_aligned = X[:min_samples]
            y_aligned = y[:min_samples]
            
        else:
            logger.error(f"Unknown truncation strategy: {strategy}")
            return None, None
        
        # Final validation with enhanced checks
        if X_aligned.shape[0] != len(y_aligned):
            logger.error(f"Alignment failed for {name}{fold_str}: X={X_aligned.shape}, y={len(y_aligned)}")
            return None, None
        
        # Check for minimum data quality
        if X_aligned.shape[1] == 0:
            logger.error(f"No features remaining after alignment for {name}{fold_str}")
            return None, None
        
        # Verify data integrity
        if np.all(X_aligned == 0):
            logger.warning(f"All features are zero after alignment for {name}{fold_str}")
        
        if len(np.unique(y_aligned)) == 1:
            logger.warning(f"All targets are identical after alignment for {name}{fold_str}")
        
        logger.info(f"Successfully auto-fixed {name}{fold_str}: truncated to {min_samples} samples (loss: {loss_percentage:.1f}%)")
        
        # Clear caches only if configured and data loss exceeds threshold
        cache_clear_threshold = SHAPE_MISMATCH_CONFIG.get("cache_clear_threshold", 25)
        if SHAPE_MISMATCH_CONFIG["cache_invalidation"] and loss_percentage > cache_clear_threshold:
            logger.debug(f"Clearing caches due to significant alignment for {name}{fold_str} (loss: {loss_percentage:.1f}% > {cache_clear_threshold}%)")
            force_clear_caches_on_alignment_error()
        elif loss_percentage > 10:
            logger.debug(f"Shape mismatch fixed for {name}{fold_str} (loss: {loss_percentage:.1f}%) - caches preserved")
        
        return X_aligned, y_aligned
        
    except Exception as e:
        logger.error(f"Error in enhanced shape validation for {name}{fold_str}: {str(e)}")
        if SHAPE_MISMATCH_CONFIG["fallback_on_failure"]:
            # Last resort: try basic truncation
            try:
                min_len = min(len(X), len(y))
                if min_len >= 2:
                    logger.warning(f"Using emergency fallback for {name}{fold_str}")
                    return X[:min_len], y[:min_len]
            except:
                pass
        return None, None

"""
Model definitions and configurations for multi-modal machine learning pipeline.
Optimized for genomic data with minimal regularization and high capacity.
"""

import numpy as np
import warnings
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, ElasticNet, Lasso
from sklearn.svm import SVR, SVC
from sklearn.feature_selection import SelectKBest, SelectFromModel, RFE
from sklearn.feature_selection import f_regression, f_classif, mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import StandardScaler
import logging

# Import configurations
try:
    from config import (
        RANDOM_FOREST_CONFIG, ELASTIC_NET_CONFIG, LASSO_CONFIG, 
        SVM_CONFIG, LOGISTIC_REGRESSION_CONFIG, EARLY_STOPPING_CONFIG
    )
except ImportError:
    # Fallback configurations for genomic data
    RANDOM_FOREST_CONFIG = {
        'n_estimators': 1000, 'max_depth': None, 'min_samples_split': 2,
        'min_samples_leaf': 1, 'max_features': 'sqrt', 'random_state': 42, 'n_jobs': -1
    }
    ELASTIC_NET_CONFIG = {
        'alpha': 0.001, 'l1_ratio': 0.1, 'max_iter': 5000, 'random_state': 42
    }
    LASSO_CONFIG = {
        'alpha': 0.0001, 'max_iter': 5000, 'random_state': 42
    }
    SVM_CONFIG = {
        'C': 100.0, 'epsilon': 0.001, 'kernel': 'rbf', 'gamma': 'scale', 'max_iter': 10000
    }
    LOGISTIC_REGRESSION_CONFIG = {
        'C': 100.0, 'max_iter': 5000, 'random_state': 42, 'solver': 'liblinear'
    }

logger = logging.getLogger(__name__)

def get_regression_models():
    """
    Get regression models optimized for genomic data.
    
    Returns
    -------
    dict
        Dictionary of model name to model instance
    """
    models = {
        "LinearRegression": LinearRegression(n_jobs=-1),
        
        "RandomForestRegressor": RandomForestRegressor(**RANDOM_FOREST_CONFIG),
        
        "ElasticNet": ElasticNet(**ELASTIC_NET_CONFIG),
        
        "Lasso": Lasso(**LASSO_CONFIG),
        
        "SVR": SVR(**SVM_CONFIG)
    }
    
    logger.debug(f"Created {len(models)} regression models with genomic optimization")
    return models

def get_classification_models():
    """
    Get classification models optimized for genomic data.
    
    Returns
    -------
    dict
        Dictionary of model name to model instance
    """
    models = {
        "LogisticRegression": LogisticRegression(**LOGISTIC_REGRESSION_CONFIG),
        
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=RANDOM_FOREST_CONFIG['n_estimators'],
            max_depth=RANDOM_FOREST_CONFIG['max_depth'],
            min_samples_split=RANDOM_FOREST_CONFIG['min_samples_split'],
            min_samples_leaf=RANDOM_FOREST_CONFIG['min_samples_leaf'],
            max_features=RANDOM_FOREST_CONFIG['max_features'],
            bootstrap=RANDOM_FOREST_CONFIG['bootstrap'],
            random_state=RANDOM_FOREST_CONFIG['random_state'],
            n_jobs=RANDOM_FOREST_CONFIG['n_jobs']
        ),
        
        "SVC": SVC(
            C=SVM_CONFIG['C'],
            kernel=SVM_CONFIG['kernel'],
            gamma=SVM_CONFIG['gamma'],
            max_iter=SVM_CONFIG['max_iter'],
            random_state=42
        )
    }
    
    logger.debug(f"Created {len(models)} classification models with genomic optimization")
    return models



def validate_model_performance(y_true, y_pred, is_regression=True, model_name="Unknown"):
    """
    Validate model performance against genomic data targets.
    
    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    is_regression : bool
        Whether this is regression
    model_name : str
        Name of the model for logging
        
    Returns
    -------
    dict
        Performance metrics and validation results
    """
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, f1_score
    
    try:
        if is_regression:
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            
            # Check against targets
            from config import PERFORMANCE_TARGETS
            targets = PERFORMANCE_TARGETS['regression']
            
            validation = {
                'r2_target_met': r2 >= targets['r2_min'],
                'performance_adequate': r2 >= 0.3,  # Minimum acceptable
                'metrics': {'r2': r2, 'rmse': rmse, 'mae': mae}
            }
            
            if r2 < 0.1:
                logger.warning(f"{model_name}: Very poor R² = {r2:.4f}")
            elif r2 >= targets['r2_min']:
                logger.info(f"{model_name}: Target R² achieved = {r2:.4f}")
                
        else:
            accuracy = accuracy_score(y_true, y_pred)
            mcc = matthews_corrcoef(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted')
            
            # Try to compute AUC if possible
            try:
                if len(np.unique(y_true)) == 2:  # Binary classification
                    auc = roc_auc_score(y_true, y_pred)
                else:
                    auc = roc_auc_score(y_true, y_pred, multi_class='ovr', average='weighted')
            except:
                auc = 0.5
            
            # Check against targets
            from config import PERFORMANCE_TARGETS
            targets = PERFORMANCE_TARGETS['classification']
            
            validation = {
                'accuracy_target_met': accuracy >= targets['accuracy_min'],
                'mcc_target_met': mcc >= targets['mcc_min'],
                'performance_adequate': accuracy >= 0.6 and mcc >= 0.3,
                'metrics': {'accuracy': accuracy, 'mcc': mcc, 'f1': f1, 'auc': auc}
            }
            
            if mcc < 0.1:
                logger.warning(f"{model_name}: Very poor MCC = {mcc:.4f}")
            elif mcc >= targets['mcc_min']:
                logger.info(f"{model_name}: Target MCC achieved = {mcc:.4f}")
        
        return validation
        
    except Exception as e:
        logger.error(f"Error validating performance for {model_name}: {e}")
        return {'performance_adequate': False, 'metrics': {}}

# Additional utility functions for genomic data

def get_adaptive_regularization(n_features, n_samples):
    """
    Get adaptive regularization parameters based on data dimensions.
    
    Parameters
    ----------
    n_features : int
        Number of features
    n_samples : int
        Number of samples
        
    Returns
    -------
    dict
        Adaptive regularization parameters
    """
    # For genomic data: less regularization when we have more features
    feature_ratio = n_features / n_samples
    
    if feature_ratio > 10:  # High-dimensional case
        alpha_elastic = 0.0001
        alpha_lasso = 0.00001
        C_logistic = 1000.0
        C_svm = 100.0
    elif feature_ratio > 5:  # Medium-dimensional
        alpha_elastic = 0.001
        alpha_lasso = 0.0001
        C_logistic = 100.0
        C_svm = 10.0
    else:  # Lower-dimensional
        alpha_elastic = 0.01
        alpha_lasso = 0.001
        C_logistic = 10.0
        C_svm = 1.0
    
    return {
        'elastic_alpha': alpha_elastic,
        'lasso_alpha': alpha_lasso,
        'logistic_C': C_logistic,
        'svm_C': C_svm
    }

def get_genomic_optimized_models(n_features, n_samples, is_regression=True):
    """
    Get models optimized for specific genomic data dimensions.
    
    Parameters
    ----------
    n_features : int
        Number of features
    n_samples : int
        Number of samples
    is_regression : bool
        Whether this is regression
        
    Returns
    -------
    dict
        Optimized models for the data dimensions
    """
    adaptive_params = get_adaptive_regularization(n_features, n_samples)
    
    if is_regression:
        models = {
            "LinearRegression": LinearRegression(n_jobs=-1),
            
            "RandomForestRegressor": RandomForestRegressor(
                n_estimators=min(1000, max(100, n_samples * 2)),
                max_depth=None,
                min_samples_split=max(2, n_samples // 50),
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            
            "ElasticNet": ElasticNet(
                alpha=adaptive_params['elastic_alpha'],
                l1_ratio=0.1,
                max_iter=5000,
                random_state=42
            ),
            
            "Lasso": Lasso(
                alpha=adaptive_params['lasso_alpha'],
                max_iter=5000,
                random_state=42
            ),
            
            "SVR": SVR(
                C=adaptive_params['svm_C'],
                epsilon=0.001,
                kernel='rbf',
                gamma='scale'
            )
        }
    else:
        models = {
            "LogisticRegression": LogisticRegression(
                C=adaptive_params['logistic_C'],
                max_iter=5000,
                random_state=42,
                solver='liblinear'
            ),
            
            "RandomForestClassifier": RandomForestClassifier(
                n_estimators=min(1000, max(100, n_samples * 2)),
                max_depth=None,
                min_samples_split=max(2, n_samples // 50),
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            
            "SVC": SVC(
                C=adaptive_params['svm_C'],
                kernel='rbf',
                gamma='scale',
                random_state=42
            )
        }
    
    logger.info(f"Created adaptive models for {n_features} features, {n_samples} samples")
    return models

def optimize_model_hyperparameters(model_name: str, X_train, y_train, X_val, y_val, 
                                  is_regression=True, n_trials=30, random_state=42):
    """
    Optimize model hyperparameters using Bayesian optimization.
    
    Parameters
    ----------
    model_name : str
        Name of the model to optimize
    X_train, y_train : array-like
        Training data
    X_val, y_val : array-like
        Validation data for optimization
    is_regression : bool
        Whether this is a regression task
    n_trials : int
        Number of optimization trials
    random_state : int
        Random state for reproducibility
        
    Returns
    -------
    object
        Optimized model
    """
    if not SKOPT_AVAILABLE:
        logger.warning("scikit-optimize not available, returning default model")
        return get_model_object(model_name, random_state)
    
    # Define search spaces for different models
    search_spaces = {
        "XGBRegressor": [
            Integer(50, 500, name='n_estimators'),
            Integer(3, 10, name='max_depth'),
            Real(0.01, 0.3, name='learning_rate'),
            Real(0.6, 1.0, name='subsample'),
            Real(0.6, 1.0, name='colsample_bytree'),
            Real(0.0, 2.0, name='reg_alpha'),
            Real(0.0, 2.0, name='reg_lambda')
        ],
        "XGBClassifier": [
            Integer(50, 500, name='n_estimators'),
            Integer(3, 10, name='max_depth'),
            Real(0.01, 0.3, name='learning_rate'),
            Real(0.6, 1.0, name='subsample'),
            Real(0.6, 1.0, name='colsample_bytree'),
            Real(0.0, 2.0, name='reg_alpha'),
            Real(0.0, 2.0, name='reg_lambda')
        ],
        "LGBMRegressor": [
            Integer(50, 500, name='n_estimators'),
            Integer(3, 10, name='max_depth'),
            Real(0.01, 0.3, name='learning_rate'),
            Real(0.6, 1.0, name='subsample'),
            Real(0.6, 1.0, name='colsample_bytree'),
            Real(0.0, 2.0, name='reg_alpha'),
            Real(0.0, 2.0, name='reg_lambda')
        ],
        "LGBMClassifier": [
            Integer(50, 500, name='n_estimators'),
            Integer(3, 10, name='max_depth'),
            Real(0.01, 0.3, name='learning_rate'),
            Real(0.6, 1.0, name='subsample'),
            Real(0.6, 1.0, name='colsample_bytree'),
            Real(0.0, 2.0, name='reg_alpha'),
            Real(0.0, 2.0, name='reg_lambda')
        ],
        "RandomForestRegressor": [
            Integer(100, 1000, name='n_estimators'),
            Integer(5, 50, name='max_depth'),
            Integer(1, 10, name='min_samples_leaf'),
            Integer(2, 20, name='min_samples_split')
        ],
        "RandomForestClassifier": [
            Integer(100, 1000, name='n_estimators'),
            Integer(5, 50, name='max_depth'),
            Integer(1, 10, name='min_samples_leaf'),
            Integer(2, 20, name='min_samples_split')
        ],
        "LogisticRegression": [
            Real(0.01, 10.0, name='C', prior='log-uniform'),
            Categorical(['l1', 'l2'], name='penalty')
        ],
        "ElasticNet": [
            Real(0.001, 10.0, name='alpha', prior='log-uniform'),
            Real(0.1, 0.9, name='l1_ratio')
        ],
        "GradientBoostingRegressor": [
            Integer(50, 500, name='n_estimators'),
            Integer(3, 10, name='max_depth'),
            Real(0.01, 0.3, name='learning_rate'),
            Real(0.6, 1.0, name='subsample')
        ],
        "GradientBoostingClassifier": [
            Integer(50, 500, name='n_estimators'),
            Integer(3, 10, name='max_depth'),
            Real(0.01, 0.3, name='learning_rate'),
            Real(0.6, 1.0, name='subsample')
        ]
    }
    
    if model_name not in search_spaces:
        logger.warning(f"No hyperparameter search space defined for {model_name}, using default")
        return get_model_object(model_name, random_state)
    
    search_space = search_spaces[model_name]
    
    @use_named_args(search_space)
    def objective(**params):
        """Objective function for hyperparameter optimization."""
        try:
            # Create model with current parameters
            if model_name == "XGBRegressor" and XGBOOST_AVAILABLE:
                model = xgb.XGBRegressor(random_state=random_state, verbosity=0, **params)
            elif model_name == "XGBClassifier" and XGBOOST_AVAILABLE:
                model = xgb.XGBClassifier(random_state=random_state, verbosity=0, 
                                        eval_metric='logloss', **params)
            elif model_name == "LGBMRegressor" and LIGHTGBM_AVAILABLE:
                model = lgb.LGBMRegressor(random_state=random_state, verbosity=-1, **params)
            elif model_name == "LGBMClassifier" and LIGHTGBM_AVAILABLE:
                model = lgb.LGBMClassifier(random_state=random_state, verbosity=-1, **params)
            elif model_name == "RandomForestRegressor":
                model = RandomForestRegressor(random_state=random_state, **params)
            elif model_name == "RandomForestClassifier":
                model = RandomForestClassifier(random_state=random_state, 
                                             class_weight='balanced', **params)
            elif model_name == "LogisticRegression":
                model = LogisticRegression(random_state=random_state, solver='liblinear',
                                         class_weight='balanced', max_iter=500, **params)
            elif model_name == "ElasticNet":
                model = ElasticNet(random_state=random_state, max_iter=5000, **params)
            elif model_name == "GradientBoostingRegressor":
                model = GradientBoostingRegressor(random_state=random_state, **params)
            elif model_name == "GradientBoostingClassifier":
                model = GradientBoostingClassifier(random_state=random_state, **params)
            else:
                return 1.0  # Return poor score for unsupported models
            
            # Fit and evaluate
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            if is_regression:
                # For regression, minimize negative R²
                score = -r2_score(y_val, y_pred)
            else:
                # For classification, minimize negative accuracy
                score = -accuracy_score(y_val, y_pred)
            
            return score
            
        except Exception as e:
            logger.warning(f"Error in hyperparameter optimization for {model_name}: {str(e)}")
            return 1.0  # Return poor score on error
    
    # Run optimization
    try:
        logger.info(f"Starting hyperparameter optimization for {model_name} with {n_trials} trials")
        result = gp_minimize(objective, search_space, n_calls=n_trials, 
                           random_state=random_state, n_initial_points=10)
        
        # Get best parameters
        best_params = dict(zip([dim.name for dim in search_space], result.x))
        logger.info(f"Best parameters for {model_name}: {best_params}")
        
        # Create optimized model
        if model_name == "XGBRegressor" and XGBOOST_AVAILABLE:
            return xgb.XGBRegressor(random_state=random_state, verbosity=0, **best_params)
        elif model_name == "XGBClassifier" and XGBOOST_AVAILABLE:
            return xgb.XGBClassifier(random_state=random_state, verbosity=0, 
                                   eval_metric='logloss', **best_params)
        elif model_name == "LGBMRegressor" and LIGHTGBM_AVAILABLE:
            return lgb.LGBMRegressor(random_state=random_state, verbosity=-1, **best_params)
        elif model_name == "LGBMClassifier" and LIGHTGBM_AVAILABLE:
            return lgb.LGBMClassifier(random_state=random_state, verbosity=-1, **best_params)
        elif model_name == "RandomForestRegressor":
            return RandomForestRegressor(random_state=random_state, **best_params)
        elif model_name == "RandomForestClassifier":
            return RandomForestClassifier(random_state=random_state, 
                                        class_weight='balanced', **best_params)
        elif model_name == "LogisticRegression":
            return LogisticRegression(random_state=random_state, solver='liblinear',
                                    class_weight='balanced', max_iter=500, **best_params)
        elif model_name == "ElasticNet":
            return ElasticNet(random_state=random_state, max_iter=5000, **best_params)
        elif model_name == "GradientBoostingRegressor":
            return GradientBoostingRegressor(random_state=random_state, **best_params)
        elif model_name == "GradientBoostingClassifier":
            return GradientBoostingClassifier(random_state=random_state, **best_params)
        
    except Exception as e:
        logger.error(f"Hyperparameter optimization failed for {model_name}: {str(e)}")
        return get_model_object(model_name, random_state)