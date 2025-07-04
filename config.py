#!/usr/bin/env python3
"""
Configuration module for the pipeline.
Contains constants and configurations used across the application.
"""

import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union

# Set Octave executable path for oct2py
# laptop 1 - my
os.environ["OCTAVE_EXECUTABLE"] = r"C:\Users\krzys\AppData\Local\Programs\GNU Octave\Octave-10.2.0\mingw64\bin\octave-cli.exe"
# Laptop 2 - Tata
#os.environ["OCTAVE_EXECUTABLE"] = r"C:\Program Files\GNU Octave\Octave-10.2.0\mingw64\bin\octave-cli.exe"


# Suppress convergence warnings for cleaner output
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
except ImportError:
    # Fallback for older sklearn versions
    warnings.filterwarnings('ignore', message='.*did not converge.*')
    warnings.filterwarnings('ignore', message='.*Objective did not converge.*')

# Constants
MAX_VARIABLE_FEATURES = None  # Let 4-phase pipeline handle feature selection intelligently
MAX_COMPONENTS = 128  # Maximum number of components for extractors
MAX_FEATURES = 512  # Maximum number of features for selectors
N_JOBS = min(os.cpu_count() or 8, 8)  # Number of parallel jobs
OMP_BLAS_THREADS = min(4, os.cpu_count() or 4)

# Feature and component selection values
N_VALUES_LIST = [8, 16, 32]  # Number of features/components to test

# Enhanced preprocessing parameters with sparsity and skewness handling
PREPROCESSING_CONFIG = {
    # Basic preprocessing parameters
    "mad_threshold": 1e-6,       # Threshold for MAD-based feature removal
    "correlation_threshold": 0.9,    # Threshold for removing highly correlated features
    "missing_threshold": 0.3,         # Threshold for missing value handling
    "outlier_threshold": 4.0,         # Standard deviations for outlier detection
    "log_transform": True,           # Apply log transformation
    "quantile_transform": True,      # Apply quantile transformation
    "remove_low_mad": True,
    "remove_highly_correlated": True,  # Remove highly correlated features
    "handle_outliers": True,
    "impute_missing": True,
    "scaling_method": 'robust',       # Robust scaling for genomic data
    "handle_missing": 'median',       # Median imputation for genomic data
    "outlier_method": 'iqr',
    "outlier_std_threshold": 4.0,     # Standard deviations for outlier threshold
    "normalize": True,
    "min_samples_per_feature": 3,    # Minimum samples required per feature
    "robust_scaling": True,          # Use robust scaling instead of standard scaling
    "feature_selection_method": "variance",  # Variance-based feature selection
    
    # Enhanced sparsity and skewness handling parameters
    "enhanced_sparsity_handling": True,     # Enable enhanced sparsity handling
    "sparsity_threshold": 0.9,              # Remove features with >90% zeros
    "min_expression_threshold": 0.1,        # Minimum meaningful expression level
    "smart_skewness_correction": True,      # Enable smart skewness correction
    "target_skewness_threshold": 0.5,       # Target |skewness| ≤ 0.5
    "enable_log_transform": True,           # Enable log1p transformation
    "enable_power_transform": True,         # Enable Yeo-Johnson/Box-Cox transformations
    "enable_quantile_transform": True,      # Enable quantile transformation as fallback
    
    # Numerical stability parameters
    "numerical_stability_checks": True,     # Enable comprehensive stability checks
    "adaptive_mad_threshold": True,    # Use adaptive MAD threshold selection
    "min_mad_threshold": 1e-10,        # Minimum MAD for numerical stability
    "max_mad_threshold": 1e-3,         # Maximum MAD threshold to consider
    "target_feature_removal_rate": 0.05,    # Target 5% of most problematic features for removal
    "min_samples_per_feature_stability": 3, # Minimum samples required for stable statistics
    "safe_statistical_computation": True,   # Use safe statistical computation methods
    "nan_handling_strategy": "remove",      # Strategy for NaN-producing features
    "zero_variance_handling": "remove",     # How to handle zero-variance features
    "constant_feature_handling": "remove",  # How to handle constant features
    "auto_remove_problematic_features": True,  # Automatically remove features that cause numerical instability
    
    # Advanced sparsity handling parameters
    "use_advanced_sparse_preprocessing": False,  # Enable advanced sparse preprocessing
    "min_non_zero_percentage": 0.05,        # Minimum percentage of non-zero values required
    "sparse_transform_method": "log1p_offset",  # Specialized transformation for sparse data
    "zero_inflation_handling": True,        # Handle zero-inflated distributions
    "outlier_capping_percentile": 99.0,     # Percentile for outlier capping in sparse data
    "target_sparsity_reduction": 0.10,      # Target percentage sparsity reduction
    
    # Aggressive dimensionality reduction parameters
    "use_aggressive_dimensionality_reduction": False,  # Enable aggressive feature reduction
    "gene_expression_target": 1500,         # Target features for gene expression
    "mirna_target": 150,                     # Target features for miRNA
    "methylation_target": 2000,             # Target features for methylation
    "dimensionality_selection_method": "hybrid",  # Selection method
    "variance_percentile": 75,               # Keep top 75% by variance in pre-filtering
    "enable_supervised_selection": True,     # Use target variable for feature selection when available
    
    # Robust scaling parameters
    "use_robust_scaling": True,              # Use RobustScaler instead of StandardScaler
    "robust_scaling_quantile_range": (25.0, 75.0),  # IQR range for RobustScaler
    "scaling_method": "robust",              # Scaling method options
    "apply_scaling_before_pca": True,        # Apply scaling before PCA/dimensionality reduction
    "clip_outliers_after_scaling": True,    # Clip extreme outliers after scaling
    "outlier_clip_range": (-5.0, 5.0),      # Range for outlier clipping after scaling
    
    # Final quantile normalization parameters
    "final_quantile_normalization": False,   # Apply quantile normalization as final step
    "quantile_n_quantiles": 1000,           # Number of quantiles for final transformation
    "quantile_output_distribution": "normal", # Output distribution
    "handle_missing_values": True,
    "missing_value_threshold": 0.5,
    "variance_threshold": 0.01,
    "outlier_detection": True,
    "outlier_method": "isolation_forest",
    "feature_selection": True,
    "feature_selection_k": "auto",
    "cross_validation_folds": 5,
    "random_state": 42,
    "n_jobs": -1,
    "use_cache": True,
    "cache_size": "4GB",
    "verbose": True,
    # Missing data indicators as features
    "add_missing_indicators": True,       # Add binary indicators for missing values
    "missing_indicator_threshold": 0.05,  # Only add indicators if >5% missing
    "missing_indicator_prefix": "missing_", # Prefix for indicator feature names
    "missing_indicator_sparse": True,     # Use sparse representation for indicators
}

# Modality-specific enhanced preprocessing configurations with numerical stability
ENHANCED_PREPROCESSING_CONFIGS = {
    "miRNA": {
        "enhanced_sparsity_handling": True,
        "sparsity_threshold": 0.9,          # Aggressive: remove features with >90% zeros
        "smart_skewness_correction": True,
        "target_skewness_threshold": 0.5,   # Target excellent skewness
        "enable_log_transform": True,
        "enable_power_transform": True,
        "enable_quantile_transform": True,
        "min_expression_threshold": 0.1,
        "handle_outliers": True,
        "outlier_threshold": 4.0,
        # Numerical stability for miRNA
        "mad_threshold": 1e-8,         # More aggressive for sparse miRNA data
        "adaptive_mad_threshold": True,
        "target_feature_removal_rate": 0.10, # Remove 10% of most problematic features
        "numerical_stability_checks": True,
        "safe_statistical_computation": True,
        # Advanced sparsity handling for miRNA
        "use_advanced_sparse_preprocessing": True,
        "min_non_zero_percentage": 0.1,     # Require >10% non-zero values
        "sparse_transform_method": "log1p_offset",  # Specialized sparse transformation
        "zero_inflation_handling": True,    # Handle zero-inflated distributions
        "outlier_capping_percentile": 99.5, # Cap extreme outliers in sparse data
        "target_sparsity_reduction": 0.15,  # Target >15% sparsity reduction
        # KNN imputation with biological similarity for miRNA
        "use_biological_knn_imputation": True,  # Enable domain-specific imputation
        "knn_neighbors": 5,                      # Number of neighbors for KNN
        "biological_similarity_weight": 0.7,    # Weight for biological vs distance similarity
        "knn_imputation_strategy": "biological", # Use biological similarity
        "fallback_imputation": "median",        # Fallback if biological KNN fails
        # Aggressive dimensionality reduction for miRNA
        "use_aggressive_dimensionality_reduction": True,
        "mirna_target": 150,                 # Target features for miRNA
        "dimensionality_selection_method": "hybrid",
        "enable_supervised_selection": True,
        # Robust scaling for miRNA
        "use_robust_scaling": True,
        "scaling_method": "robust",
        "robust_scaling_quantile_range": (10.0, 90.0),  # Wider range for sparse miRNA data
        "apply_scaling_before_pca": True,
        "clip_outliers_after_scaling": True,
        "outlier_clip_range": (-6.0, 6.0),  # Slightly wider range for miRNA outliers
        # Final quantile normalization for miRNA
        "final_quantile_normalization": False,   # Optional: can be enabled for miRNA
        "quantile_n_quantiles": 500,            # Fewer quantiles for sparse miRNA data
        "quantile_output_distribution": "normal", # Normal distribution for downstream ML
        "description": "Optimized for miRNA data with enhanced sparsity handling and biological KNN imputation"
    },
    "Gene Expression": {
        "enhanced_sparsity_handling": True,
        "sparsity_threshold": 0.85,         # Moderate: remove features with >85% zeros
        "smart_skewness_correction": True,
        "target_skewness_threshold": 0.7,   # Slightly more lenient
        "enable_log_transform": True,
        "enable_power_transform": False,    # Log transform usually sufficient
        "enable_quantile_transform": False,
        "min_expression_threshold": 0.05,
        "handle_outliers": True,
        "outlier_threshold": 5.0,
        # Numerical stability for gene expression
        "mad_threshold": 0.01,         # More aggressive MAD threshold
        "adaptive_mad_threshold": True,
        "target_feature_removal_rate": 0.05, # Remove 5% of most problematic features
        "numerical_stability_checks": True,
        "safe_statistical_computation": True,
        "auto_remove_problematic_features": True,  # Auto-remove unstable features
        "min_variance_threshold": 1e-6,  # More aggressive variance threshold for stability
        # Enhanced sparsity handling for gene expression
        "use_advanced_sparse_preprocessing": True,
        "min_non_zero_percentage": 0.05,    # Require >5% non-zero values
        "sparse_transform_method": "log1p_offset",
        "zero_inflation_handling": True,
        "outlier_capping_percentile": 99.0,
        "target_sparsity_reduction": 0.10,  # Target >10% sparsity reduction
        # KNN imputation with biological similarity for gene expression
        "use_biological_knn_imputation": True,  # Enable domain-specific imputation
        "knn_neighbors": 7,                      # More neighbors for gene expression
        "biological_similarity_weight": 0.6,    # Slightly less biological weight than miRNA
        "knn_imputation_strategy": "biological", # Use biological similarity
        "fallback_imputation": "median",        # Fallback if biological KNN fails
        # Aggressive dimensionality reduction for gene expression
        "use_aggressive_dimensionality_reduction": True,
        "gene_expression_target": 1500,     # Target features for gene expression
        "dimensionality_selection_method": "hybrid",
        "enable_supervised_selection": True,
        # Robust scaling for gene expression
        "use_robust_scaling": True,
        "scaling_method": "robust",
        "robust_scaling_quantile_range": (25.0, 75.0),  # Standard IQR for gene expression
        "apply_scaling_before_pca": True,
        "clip_outliers_after_scaling": True,
        "outlier_clip_range": (-5.0, 5.0),  # Standard range for gene expression
        # NEW: Enhanced outlier handling options for extreme expression values
        "use_log1p_preprocessing": False,    # Alternative: apply log1p to raw counts before scaling
        "adaptive_outlier_clipping": True,   # Use modality-specific clipping ranges
        # Final quantile normalization for gene expression
        "final_quantile_normalization": False,   # Optional: can be enabled for gene expression
        "quantile_n_quantiles": 1000,           # Full quantiles for gene expression
        "quantile_output_distribution": "normal", # Normal distribution for downstream ML
        "description": "Optimized for gene expression data - enhanced sparsity handling, biological KNN imputation, numerical stability, aggressive dimensionality reduction, and robust scaling"
    },
    "Methylation": {
        "enhanced_sparsity_handling": True,
        "sparsity_threshold": 0.95,         # Conservative: methylation can have legitimate zeros
        "smart_skewness_correction": False, # Methylation is typically 0-1 range, preserve distribution
        "log_transform": False,             # Don't log-transform methylation data
        "enable_power_transform": False,
        "min_expression_threshold": 0.0,   # No minimum threshold for methylation
        "handle_outliers": True,
        "outlier_threshold": 3.0,           # More sensitive outlier detection
        # Numerical stability for methylation (usually well-behaved)
        "mad_threshold": 1e-9,         # Conservative threshold for methylation
        "adaptive_mad_threshold": True,
        "target_feature_removal_rate": 0.02, # Remove only 2% of most problematic features
        "numerical_stability_checks": True,
        "safe_statistical_computation": True,
        "auto_remove_problematic_features": True,  # Auto-remove unstable features
        # Conservative sparsity handling for methylation
        "use_advanced_sparse_preprocessing": False,  # Methylation usually not sparse
        "min_non_zero_percentage": 0.01,    # Very conservative (1% non-zero)
        "sparse_transform_method": "asinh_sparse",  # Handles 0-1 range well
        "zero_inflation_handling": False,   # Methylation zeros are often meaningful
        "outlier_capping_percentile": 99.9, # Very conservative outlier capping
        "target_sparsity_reduction": 0.05,  # Target >5% sparsity reduction
        # NEW: Mean imputation for methylation (low missingness data)
        "use_mean_imputation": True,         # Use mean instead of median for methylation
        "imputation_strategy": "mean",       # Explicitly set mean imputation
        "fallback_imputation": "mean",       # Keep mean as fallback too
        # NEW: Variance-based dimensionality reduction for methylation (3956 features)
        "use_aggressive_dimensionality_reduction": True,
        "methylation_target": 2000,         # Reduce from 3956 using variance-based filtering
        "dimensionality_selection_method": "variance",  # Variance-focused as recommended
        "enable_supervised_selection": False,  # Conservative approach for methylation
        # NEW: Conservative robust scaling for methylation (0-1 range data)
        "use_robust_scaling": True,
        "scaling_method": "robust",
        "robust_scaling_quantile_range": (5.0, 95.0),  # Conservative range for 0-1 methylation data
        "apply_scaling_before_pca": True,
        "clip_outliers_after_scaling": False,  # Don't clip methylation data (0-1 range)
        "outlier_clip_range": (-3.0, 3.0),  # Conservative range (not used due to clip=False)
        "description": "Conservative preprocessing for methylation data - enhanced numerical stability, variance-based dimensionality reduction, and conservative robust scaling"
    }
}

# High-memory server optimization (60GB RAM available)
MEMORY_OPTIMIZATION = {
    "chunk_size": 10000,  # Much larger chunks for high-memory systems
    "cache_dir": "./.cache",  # Cache directory
    "cache_size": "8GB",  # Increased cache size per type for 60GB system
    #"total_cache_limit": "32GB",  # Use ~50% of available RAM for caching
    "auto_clear_threshold": 0.85,  # Higher threshold for high-memory systems
    "memory_monitor_interval": 60,  # Less frequent monitoring for stable systems
    "shape_mismatch_auto_fix": True,  # Enable automatic shape mismatch fixing
    "alignment_loss_threshold": 0.3,  # Keep conservative data loss threshold
    "min_samples_threshold": 2,  # Minimum samples required after alignment
}

# High-memory server caching configuration (60GB RAM available)
CACHE_CONFIG = {
    "selector_regression": {"maxsize": 32, "maxmemory_mb": 1000},  # 1GB limit
    "selector_classification": {"maxsize": 32, "maxmemory_mb": 1000},  # 1GB limit  
    "extractor_regression": {"maxsize": 32, "maxmemory_mb": 1000},  # 1GB limit
    "extractor_classification": {"maxsize": 32, "maxmemory_mb": 1000},  # 1GB limit
    "total_limit_mb": 4000,  # 4GB total limit (reduced from 8GB)
    #"selector_regression": {"maxsize": 128, "maxmemory_mb": 8000},  # 8GB limit per cache
    #"selector_classification": {"maxsize": 128, "maxmemory_mb": 8000},  # 8GB limit per cache
    #"extractor_regression": {"maxsize": 128, "maxmemory_mb": 8000},  # 8GB limit per cache
    #"extractor_classification": {"maxsize": 128, "maxmemory_mb": 8000},  # 8GB limit per cache
    #"total_limit_mb": 32000,  # 32GB total limit (~50% of 60GB RAM)
    "eviction_strategy": "lru",  # Least Recently Used eviction
    "memory_check_interval": 300,  # Check memory usage every 5 minutes
}

# High-memory parallel processing configuration
JOBLIB_PARALLEL_CONFIG = {
    'max_nbytes': '50M',  # Limit memory per worker
    #'max_nbytes': '2G',  # Increased memory per worker for 60GB system
    'prefer': 'threads',  # Prefer threads over processes
    'require': 'sharedmem',  # Require shared memory
    'verbose': 0  # No verbose output
}

# Model optimization settings - IMPROVED CONFIGURATIONS
MODEL_OPTIMIZATIONS = {
    "RandomForest": {
        "n_estimators": 100,         # Reduced for faster training
        "max_depth": 5,              # Shallow trees for high-dim data
        "min_samples_split": 10,     # Higher to prevent overfitting
        "min_samples_leaf": 5,       # Higher minimum leaf size
        "max_features": "sqrt",      # Square root of features
        "bootstrap": True,
        "oob_score": False,
        "class_weight": "balanced",
        "random_state": 42
    },
    "ElasticNet": {
        "alpha": 1.0,                # Reduced regularization for better performance
        "l1_ratio": 0.5,             # Balanced L1/L2 (will be tuned in log-space)
        "max_iter": 5000,            # More iterations
        "tol": 1e-4,
        "selection": "random",       # Random coordinate descent
        "random_state": 42
    },
    "Lasso": {
        "alpha": 1.0,                # Strong regularization
        "max_iter": 5000,
        "tol": 1e-4,
        "selection": "random",
        "random_state": 42
    },
    "LogisticRegression": {
        "penalty": "l2",             # L2 regularization for stability
        "solver": "liblinear",
        "C": 1.0,                    # Regularization strength
        "max_iter": 500,             # As suggested in the requirements
        "class_weight": "balanced",
        "random_state": 42
    },
    "SVM": {
        "C": 0.1,                    # Strong regularization
        "kernel": "linear",          # Linear for high-dim data
        "class_weight": "balanced",
        "probability": True,
        "random_state": 42
    },
    "GradientBoosting": {
        "n_estimators": 200,         # 100-300 trees as specified
        "learning_rate": 0.1,        # Moderate learning rate
        "max_depth": 3,              # Shallow trees for boosting
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "subsample": 1.0,
        "random_state": 42
    },

    "XGBoost": {
        "n_estimators": 200,         # Increased trees for better performance
        "max_depth": 6,              # Deeper trees for more complex patterns
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,            # L1 regularization
        "reg_lambda": 1.0,           # L2 regularization
        "scale_pos_weight": 1,       # Will be adjusted for imbalance
        "random_state": 42
    },
    "LightGBM": {
        "n_estimators": 200,         # Similar to XGBoost
        "max_depth": 6,              # Deeper trees
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,            # L1 regularization
        "reg_lambda": 1.0,           # L2 regularization
        "random_state": 42
    },
    "GradientBoosting": {
        "n_estimators": 200,         # Increased estimators
        "max_depth": 6,              # Deeper trees
        "learning_rate": 0.1,
        "subsample": 0.8,
        "random_state": 42
    },
    "BalancedRandomForest": {
        "n_estimators": 500,         # More trees for better performance
        "max_depth": None,           # No depth limit
        "sampling_strategy": "auto", # Automatic balancing
        "replacement": False,        # Without replacement
        "bootstrap": True,
        "oob_score": True,
        "random_state": 42
    },
    "BalancedXGBoost": {
        "n_estimators": 400,         # As specified in requirements
        "max_depth": 4,              # As specified in requirements
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",    # Keeps probability calibration
        "scale_pos_weight": None,    # Let sampler handle balance
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        "random_state": 42
    },
    "BalancedLightGBM": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        "is_unbalance": True,        # LightGBM's built-in class weight handling
        "random_state": 42
    },
    "ImprovedXGBRegressor": {
        "n_estimators": 800,         # Increased for better performance
        "max_depth": 4,              # Optimal depth for interactions
        "learning_rate": 0.05,       # Lower learning rate with more estimators
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,           # L2 regularization
        "random_state": 42
    },
    "ImprovedLightGBMRegressor": {
        "n_estimators": 800,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "objective": "regression",   # Can be changed to "quantile" for robust loss
        "random_state": 42
    },
    "RobustGradientBoosting": {
        "n_estimators": 700,         # As specified in requirements
        "max_depth": 3,              # As specified in requirements
        "learning_rate": 0.03,       # As specified in requirements
        "subsample": 0.8,
        "loss": "huber",             # Robust loss function for outliers
        "alpha": 0.9,                # Huber loss parameter
        "random_state": 42
    },
    "KPLS": {
        "gamma": ["auto", 1e-3, 1e-2],  # Kernel coefficient: auto (median heuristic), or fixed values
        "n_components": [4, 8, 16],     # Number of PLS components
        "algorithm": [1, 2],            # IKPLS algorithm variant
        "kernel": ["rbf"],              # Kernel type (currently only RBF supported)
        "max_iter": [500, 1000],        # Maximum iterations
        "random_state": 42
    }
}

# Enhanced early stopping configuration
EARLY_STOPPING_CONFIG = {
    "enabled": True,  # Enable/disable early stopping globally
    "patience": 50,  # Patience for early stopping
    "min_delta": 1e-6,  # Minimum improvement threshold
    "validation_split": 0.2,  # Fraction of training data to use for early stopping validation
    "restore_best_weights": True,  # Whether to restore best model weights
    "monitor_metric": "auto",  # Metric to monitor: "auto", "neg_mse", "accuracy", "r2"
    "verbose": 0,  # Verbosity level for early stopping (0=silent, 1=progress, 2=detailed)
    "adaptive_patience": True,  # Increase patience for complex models
    "max_patience": 100,  # Maximum patience for complex models
}

# Enhanced shape mismatch handling configuration
SHAPE_MISMATCH_CONFIG = {
    "auto_fix_enabled": True,  # Enable automatic shape mismatch fixing
    "max_data_loss_percent": 50,  # Maximum allowed data loss percentage
    "min_samples_after_fix": 2,  # Minimum samples required after fixing
    "truncation_strategy": "min",  # "min" = use minimum length, "intersection" = use sample intersection
    "log_all_fixes": True,  # Log all shape mismatch fixes
    "fallback_on_failure": True,  # Use fallback strategies when fixing fails
    "cache_invalidation": False,  # Disable frequent cache clearing - only clear on major errors
    "cache_clear_threshold": 25,  # Only clear caches if data loss > 25%
}

# Class imbalance handling configuration
CLASS_IMBALANCE_CONFIG = {
    "balance_enabled": True,  # Enable class balancing techniques
    "use_smote_undersampling": True,  # Use SMOTE + RandomUnderSampler pipeline
    "use_balanced_models": True,  # Use balanced-aware models
    "optimize_threshold_for_mcc": True,  # Optimize decision threshold for MCC
    "smote_k_neighbors": 5,  # SMOTE k_neighbors parameter
    "threshold_search_range": (0.1, 0.9),  # Range for threshold optimization
    "threshold_search_steps": 17,  # Number of steps in threshold search
    "min_samples_for_smote": 10,  # Minimum samples required to apply SMOTE
}

# Regression improvements configuration for negative R² issues
REGRESSION_IMPROVEMENTS_CONFIG = {
    "target_transformations_enabled": True,  # Enable target transformations
    "use_gradient_boosted_trees": True,  # Use XGBoost/LightGBM over RF/linear
    "use_robust_loss_functions": True,  # Use Huber/Quantile loss for outliers
    "hyperparameter_tuning_enabled": True,  # Enable Optuna tuning
    "n_trials": 30,  # Number of Optuna trials
    
    # Dataset-specific target transformations
    "target_transformations": {
        "aml": {
            "transform": "log1p",  # log1p for blast % (highly skewed)
            "inverse_transform": "expm1",
            "description": "Log1p transformation for AML blast % (highly skewed & heavy-tailed)"
        },
        "sarcoma": {
            "transform": "sqrt",  # sqrt for tumor length (right-skewed)
            "inverse_transform": "square",
            "description": "Square root transformation for Sarcoma tumor length (right-skewed)"
        }
    },
    
    # Optimized gradient boosting parameters
    "gradient_boosting_params": {
        "xgboost": {
            "n_estimators": 800,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "random_state": 42
        },
        "lightgbm": {
            "n_estimators": 800,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "random_state": 42
        },
        "gradient_boosting": {
            "n_estimators": 700,
            "max_depth": 3,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "loss": "huber",  # Robust loss function
            "random_state": 42
        }
    },
    
    # Robust loss function settings
    "robust_loss_settings": {
        "huber_alpha": 0.9,  # Huber loss parameter
        "quantile_alpha": 0.5,  # Quantile regression parameter
        "use_huber_for_outliers": True,  # Use Huber loss when outliers detected
        "outlier_detection_threshold": 3.0,  # IQR multiplier for outlier detection
    }
}

# Feature engineering tweaks configuration for improved metrics
FEATURE_ENGINEERING_CONFIG = {
    "enabled": False,  # Disabled by default, enabled via CLI
    "sparse_plsda_enabled": True,  # Sparse PLS-DA for better MCC
    "kernel_pca_enabled": True,    # Kernel PCA for higher R²
    
    # Sparse PLS-DA configuration for MCC improvement
    "sparse_plsda": {
        "n_components": 32,  # As specified in requirements
        "alpha": 0.1,        # Sparsity parameter
        "max_iter": 1000,
        "tol": 1e-6,
        "scale": True,
        "description": "Creates maximally discriminative latent space, balances class variance"
    },
    
    # Kernel PCA configuration for R² improvement  
    "kernel_pca": {
        "n_components": 64,  # As specified in requirements
        "kernel": "rbf",     # RBF kernel for non-linear interactions
        "gamma": "auto",     # Will be set to median heuristic
        "eigen_solver": "auto",
        "n_jobs": -1,
        "random_state": 42,
        "description": "Captures non-linear gene–methylation interactions feeding into boosted trees"
    },
    
    # Median heuristic for gamma calculation
    "median_heuristic": {
        "enabled": True,
        "sample_size": 1000,  # Sample size for gamma calculation
        "percentile": 50      # Median percentile
    }
}

# Fusion upgrades configuration for improved performance
FUSION_UPGRADES_CONFIG = {
    "enabled": False,  # Disabled by default, enabled via CLI
    
    # CURRENT IMPLEMENTATION - Specified fusion strategies only
    
    # Attention-weighted concatenation configuration
    "attention_weighted": {
        "enabled": True,
        "hidden_dim": 32,        # Hidden dimension for attention MLP
        "dropout_rate": 0.3,     # OPTIMIZED: Increased dropout for better regularization (0.1  0.3)
        "learning_rate": 0.001,  # Learning rate for optimization
        "max_epochs": 100,       # Maximum training epochs
        "patience": 10,          # Early stopping patience
        "random_state": 42,
        "description": "Sample-specific weighting improved AML R² +0.05 and Colon MCC +0.04"
    },
    
    # Learnable weighted fusion configuration
    "learnable_weighted": {
        "enabled": True,
        "cv_folds": 5,           # Cross-validation folds for weight learning
        "random_state": 42,
        "description": "OPTIMIZED: Learns optimal modality weights instead of using equal weights"
    },
    
    # Multiple Kernel Learning configuration
    "mkl": {
        "enabled": True,
        "n_components": 10,      # Number of components for kernel fusion
        "gamma": 1.0,           # Gamma parameter for RBF kernels
        "random_state": 42,
        "description": "Multiple Kernel Learning with RBF kernels"
    },
    
    # Similarity Network Fusion configuration
    "snf": {
        "enabled": True,
        "K": 30,                # Number of neighbors for similarity network
        "alpha": 0.8,           # Fusion parameter
        "T": 30,                # Number of iterations
        "random_state": 42,
        "description": "Similarity Network Fusion with spectral clustering"
    },
    
    # Early fusion PCA configuration
    "early_fusion_pca": {
        "enabled": True,
        "n_components": None,    # Automatically determined based on data
        "random_state": 42,
        "description": "PCA-based early integration"
    },
    
    # Strategy selection - CURRENT IMPLEMENTATION: Only specified strategies
    "default_strategy": "attention_weighted",     # Default strategy for clean data
    "fallback_strategy": "early_fusion_pca",     # Fallback when advanced strategies fail
    "auto_strategy_selection": True,             # Automatically select best strategy based on data
}

# Sample retention warning configuration
SAMPLE_RETENTION_CONFIG = {
    "suppress_warnings_for_datasets": ["colon", "kidney", "liver"],  # Datasets with expected low retention
    "low_retention_threshold": 40,  # Threshold for low retention warnings (%)
    "moderate_retention_threshold": 70,  # Threshold for moderate retention warnings (%)
    "log_retention_details": True,  # Log detailed retention information
    "expected_low_retention_message": "Expected low sample retention for this dataset type",
}

# Robust feature selection configuration - LESS AGGRESSIVE FEATURE SELECTION
FEATURE_SELECTION_CONFIG = {
    "min_features": 50,  # Much higher minimum
    "max_features_ratio": 0.95,  # Keep most features
    "mad_threshold": 0.0001,  # Very low MAD threshold (robust)
    "correlation_threshold": 0.99,  # Only remove highly correlated
    "missing_threshold": 0.8,  # Allow more missing data
    "mad_threshold": 0.01  # Lower MAD threshold
}

# Improved extractor stability configuration - BETTER COMPONENT SELECTION
EXTRACTOR_CONFIG = {
    "adaptive_components": True,  # Enable adaptive component selection
    "min_explained_variance": 0.85,  # Increased from 0.8 for better representation
    "max_components_ratio": 0.95,  # Increased from 0.9 to allow more components
    "fallback_to_pca": True,  # Fall back to PCA if other extractors fail
    "stability_checks": True,  # Perform stability checks on extracted features
    "numerical_stability": True,  # Enable numerical stability improvements
    "preserve_variance": True,  # Prioritize variance preservation
    "component_selection_method": "explained_variance",  # Use explained variance for selection
}

# Comprehensive logging configuration
LOGGING_CONFIG = {
    "level": "INFO",  # Default logging level
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "memory_logging": True,  # Log memory usage
    "performance_logging": True,  # Log performance metrics
    "shape_mismatch_logging": True,  # Log shape mismatch fixes
    "cache_logging": True,  # Log cache operations
    "feature_selection_logging": True,  # Log feature selection details
    "extractor_logging": True,  # Log extractor operations
    "model_training_logging": True,  # Log model training details
    "file_handler": True,
    "console_handler": True
}

# MRMR Feature Selection Configuration
MRMR_CONFIG = {
    "fast_mode": True,  # Use fast approximations (correlation instead of MI for redundancy)
    "max_features_prefilter": 1000,  # Pre-filter to top N features before MRMR (0 = no prefilter)
    "n_neighbors": 3,  # Number of neighbors for MI estimation (lower = faster)
    "progress_logging": True,  # Log MRMR selection progress
    "fallback_on_error": True  # Fall back to mutual_info if MRMR fails
}

# Fast Feature Selection Configuration (alternatives to MRMR)
FAST_FEATURE_SELECTION_CONFIG = {
    "enabled": True,  # Enable fast feature selection methods
    "default_method_regression": "variance_f_test",  # Default method for regression
    "default_method_classification": "variance_f_test",  # Default method for classification
    "mad_threshold": 0.0001,  # Very permissive (robust)
    "rf_n_estimators": 200,  # More trees for importance
    "rf_max_depth": 20,  # Deeper trees
    "correlation_method": "spearman",  # Better for genomic data
    "alpha": 0.0001,  # Minimal regularization
    "progress_logging": True,  # Log feature selection progress
    "fallback_on_error": True  # Fall back to univariate methods if fast methods fail
}

# Configuration for missing modalities simulation
MISSING_MODALITIES_CONFIG = {
    "enabled": True,
    "random_seed": 42,
    "cv_fold_seed_offset": 100,
    "missing_percentages": [0.0, 0.2, 0.5]
}

# Configuration for feature extraction behavior
FEATURE_EXTRACTION_CONFIG = {
    "force_n_components": True  # If True, tries to use the exact number of components requested
}

# Configuration for warning suppression
WARNING_SUPPRESSION_CONFIG = {
    "suppress_auc_warnings": True,  # Suppress AUC calculation warnings
    "suppress_alignment_warnings": False,  # Keep alignment warnings visible
    "suppress_class_warnings": True,  # Suppress class distribution warnings for known issues
    "suppress_sklearn_warnings": True,  # Suppress sklearn warnings
    "datasets_with_known_issues": ["Colon", "Lung"],  # Datasets with known small class issues
}

# Cross-validation configuration for small samples
CV_CONFIG = {
    "n_splits": 5,  # Standard 5-fold CV
    "shuffle": True,
    "random_state": 42,
    "stratify": True  # For classification
}

# Default ranges for different scenarios
DEFAULT_N_FEATURES_RANGE = [128, 256, 512, 1024]  # Genomic-appropriate ranges
DEFAULT_N_COMPONENTS_RANGE = [64, 128, 256]  # More components for complex data

# Random Forest - Optimized for genomic data
RANDOM_FOREST_CONFIG = {
    'n_estimators': 1000,  # More trees for complex patterns
    'max_depth': None,  # Allow deep trees for genomic complexity
    'min_samples_split': 2,  # Allow fine-grained splits
    'min_samples_leaf': 1,  # Allow detailed leaf nodes
    'max_features': 'sqrt',  # Good default for genomic data
    'bootstrap': True,
    'random_state': 42,
    'n_jobs': -1
}

# ElasticNet - Stricter regularization for better generalization
ELASTIC_NET_CONFIG = {
    'alpha': 0.3,  # OPTIMIZED: Stricter regularization (0.001  0.3, range 0.1-0.5)
    'l1_ratio': 0.5,  # Balanced L1/L2 regularization for feature selection
    'max_iter': 5000,  # More iterations for convergence
    'random_state': 42,
    'selection': 'random'
}

# Lasso - Minimal regularization
LASSO_CONFIG = {
    'alpha': 0.0001,  # Minimal regularization
    'max_iter': 5000,
    'random_state': 42,
    'selection': 'random'
}

# SVM - Optimized for high-dimensional data
SVM_CONFIG = {
    'C': 100.0,  # High C for complex patterns
    'epsilon': 0.001,  # Tight epsilon for precision
    'kernel': 'rbf',
    'gamma': 'scale',
    'max_iter': 10000
}

# Logistic Regression - Minimal regularization
LOGISTIC_REGRESSION_CONFIG = {
    'C': 100.0,  # Very low regularization
    'max_iter': 5000,
    'random_state': 42,
    'solver': 'liblinear'  # Good for high-dimensional data
}

# Neural network architecture for genomic data
NN_ARCHITECTURE_CONFIG = {
    'hidden_layers': [512, 256, 128],  # Larger networks
    'dropout_rate': 0.3,  # OPTIMIZED: Increased dropout for better regularization (0.1  0.3)
    'activation': 'relu',
    'batch_size': 32,
    'epochs': 200,  # More epochs
    'learning_rate': 0.001
}

# Target performance metrics for genomic data
PERFORMANCE_TARGETS = {
    'regression': {
        'r2_min': 0.5,  # Target R² > 0.5
        'rmse_max': 0.5,  # Relative to target range
        'mae_max': 0.3
    },
    'classification': {
        'accuracy_min': 0.7,  # Target accuracy > 70%
        'auc_min': 0.7,  # Target AUC > 0.7
        'mcc_min': 0.5,  # Target MCC > 0.5
        'f1_min': 0.6
    }
}

# Computational configuration
COMPUTATIONAL_CONFIG = {
    'n_jobs': -1,  # Use all available cores
    'memory_limit': '8GB',
    'timeout': 3600,  # 1 hour timeout per model
    'chunk_size': 1000,
    'parallel_backend': 'threading'
}

@dataclass
class DatasetConfig:
    """Dataset configuration dataclass."""
    name: str
    base_path: str
    modalities: Dict[str, str]
    outcome_file: str
    outcome_col: str
    id_col: str
    outcome_type: str = "os"
    output_dir: str = "output"
    fix_tcga_ids: bool = False
    nfeats_list: List[int] = field(default_factory=lambda: [32, 64, 128])  # Increased feature ranges
    ncomps_list: List[int] = field(default_factory=lambda: [32, 64, 128])  # Increased component ranges
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to dictionary."""
        return {
            "name": self.name,
            "base_path": self.base_path,
            "modalities": self.modalities,
            "outcome_file": self.outcome_file,
            "outcome_col": self.outcome_col,
            "id_col": self.id_col,
            "outcome_type": self.outcome_type,
            "output_dir": self.output_dir,
            "fix_tcga_ids": self.fix_tcga_ids,
            "nfeats_list": self.nfeats_list,
            "ncomps_list": self.ncomps_list
        }
    
    @staticmethod
    def get_config(dataset_name: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a dataset by name.
        
        Parameters
        ----------
        dataset_name : str
            Name of the dataset to get configuration for
            
        Returns
        -------
        Optional[Dict[str, Any]]
            Dataset configuration dictionary or None if not found
        """
        dataset_name_lower = dataset_name.lower()
        
        # Search in classification datasets
        for config_dict in CLASSIFICATION_DATASETS:
            if config_dict['name'].lower() == dataset_name_lower:
                return config_dict
        
        # Search in regression datasets
        for config_dict in REGRESSION_DATASETS:
            if config_dict['name'].lower() == dataset_name_lower:
                return config_dict
        
        return None

# Example dataset configurations
# Users should modify these with their actual datasets

# Regression datasets
REGRESSION_DATASETS = [    
    DatasetConfig(
        name="AML",
        base_path="data/aml",
        modalities={
            "Gene Expression": "exp.csv",
            "miRNA": "mirna.csv",
            "Methylation": "methy.csv"
        },
        outcome_file="data/clinical/aml.csv",
        outcome_col="lab_procedure_bone_marrow_blast_cell_outcome_percent_value",
        id_col="sampleID",
        outcome_type="continuous",
        fix_tcga_ids=True
    ).to_dict(),
    
    DatasetConfig(
        name="Sarcoma",
        base_path="data/sarcoma",
        modalities={
            "Gene Expression": "exp.csv",
            "miRNA": "mirna.csv",
            "Methylation": "methy.csv"
        },
        outcome_file="data/clinical/sarcoma.csv",
        outcome_col="pathologic_tumor_length",
        id_col="metsampleID",
        outcome_type="continuous",
        fix_tcga_ids=True
    ).to_dict()
]

"""Add commentMore actions
        DatasetConfig(
        name="TestRegression",
        base_path="test_data/regression",
        modalities={
            "Gene Expression": "exp.csv",
            "miRNA": "mirna.csv",
            "Methylation": "methy.csv"
        },
        outcome_file="clinical.csv",  # Fixed: should be in the same directory as modalities
        outcome_col="survival_time",
        id_col="sample_id",
        outcome_type="continuous",
        output_dir="output_regression",
        fix_tcga_ids=False  # Fixed: test data doesn't need TCGA ID fixing
    ).to_dict(),

    DatasetConfig(
        name="AML",
        base_path="data/aml",
        modalities={
            "Gene Expression": "exp.csv",
            "miRNA": "mirna.csv",
            "Methylation": "methy.csv"
        },
        outcome_file="data/clinical/aml.csv",
        outcome_col="lab_procedure_bone_marrow_blast_cell_outcome_percent_value",
        id_col="sampleID",
        outcome_type="continuous",
        fix_tcga_ids=True
    ).to_dict(),
    
    DatasetConfig(
        name="Sarcoma",
        base_path="data/sarcoma",
        modalities={
            "Gene Expression": "exp.csv",
            "miRNA": "mirna.csv",
            "Methylation": "methy.csv"
        },
        outcome_file="data/clinical/sarcoma.csv",
        outcome_col="pathologic_tumor_length",
        id_col="metsampleID",
        outcome_type="continuous",
        fix_tcga_ids=True
    ).to_dict()
]
"""

# Classification datasets
CLASSIFICATION_DATASETS = [    
    DatasetConfig(
        name="Colon",
        base_path="data/colon",
        modalities={
            "Gene Expression": "exp.csv",
            "miRNA": "mirna.csv",
            "Methylation": "methy.csv"
        },
        outcome_file="data/clinical/colon.csv",
        outcome_col="pathologic_T",
        id_col="sampleID",
        outcome_type="class",
        fix_tcga_ids=True
    ).to_dict(),

    DatasetConfig(
        name="Breast",
        base_path="data/breast",
        modalities={
            "Gene Expression": "exp.csv",
            "miRNA": "mirna.csv",
            "Methylation": "methy.csv"
        },
        outcome_file="data/clinical/breast.csv",
        outcome_col="pathologic_T",
        id_col="sampleID",
        outcome_type="class",
        fix_tcga_ids=True
    ).to_dict(),
    
    DatasetConfig(
        name="Kidney",
        base_path="data/kidney",
        modalities={
            "Gene Expression": "exp.csv",
            "miRNA": "mirna.csv",
            "Methylation": "methy.csv"
        },
        outcome_file="data/clinical/kidney.csv",
        outcome_col="pathologic_T",
        id_col="submitter_id.samples",
        outcome_type="class",
        fix_tcga_ids=True
    ).to_dict(),
    
    DatasetConfig(
        name="Liver",
        base_path="data/liver",
        modalities={
            "Gene Expression": "exp.csv",
            "miRNA": "mirna.csv",
            "Methylation": "methy.csv"
        },
        outcome_file="data/clinical/liver.csv",
        outcome_col="pathologic_T",
        id_col="sampleID",
        outcome_type="class",
        fix_tcga_ids=True
    ).to_dict(),
    
    DatasetConfig(
        name="Lung",
        base_path="data/lung",
        modalities={
            "Gene Expression": "exp.csv",
            "miRNA": "mirna.csv",
            "Methylation": "methy.csv"
        },
        outcome_file="data/clinical/lung.csv",
        outcome_col="pathologic_T",
        id_col="sampleID",
        outcome_type="class",
        fix_tcga_ids=True
    ).to_dict(),
    
    DatasetConfig(
        name="Melanoma",
        base_path="data/melanoma",
        modalities={
            "Gene Expression": "exp.csv",
            "miRNA": "mirna.csv",
            "Methylation": "methy.csv"
        },
        outcome_file="data/clinical/melanoma.csv",
        outcome_col="pathologic_T",
        id_col="sampleID",
        outcome_type="class",
        fix_tcga_ids=True
    ).to_dict(),
    
    DatasetConfig(
        name="Ovarian",
        base_path="data/ovarian",
        modalities={
            "Gene Expression": "exp.csv",
            "miRNA": "mirna.csv",
            "Methylation": "methy.csv"
        },
        outcome_file="data/clinical/ovarian.csv",
        outcome_col="clinical_stage",
        id_col="sampleID",
        outcome_type="class",
        fix_tcga_ids=True
    ).to_dict()
]

"""
    DatasetConfig(
        name="Colon",
        base_path="data/colon",
        modalities={
            "Gene Expression": "exp.csv",
            "miRNA": "mirna.csv",
            "Methylation": "methy.csv"
        },
        outcome_file="data/clinical/colon.csv",
        outcome_col="pathologic_T",
        id_col="sampleID",
        outcome_type="class",
        fix_tcga_ids=True
    ).to_dict(),

    DatasetConfig(
        name="Breast",
        base_path="data/breast",
        modalities={
            "Gene Expression": "exp.csv",
            "miRNA": "mirna.csv",
            "Methylation": "methy.csv"
        },
        outcome_file="data/clinical/breast.csv",
        outcome_col="pathologic_T",
        id_col="sampleID",
        outcome_type="class",
        fix_tcga_ids=True
    ).to_dict(),
    
    DatasetConfig(
        name="Kidney",
        base_path="data/kidney",
        modalities={
            "Gene Expression": "exp.csv",
            "miRNA": "mirna.csv",
            "Methylation": "methy.csv"
        },
        outcome_file="data/clinical/kidney.csv",
        outcome_col="pathologic_T",
        id_col="submitter_id.samples",
        outcome_type="class",
        fix_tcga_ids=True
    ).to_dict(),
    
    DatasetConfig(
        name="Liver",
        base_path="data/liver",
        modalities={
            "Gene Expression": "exp.csv",
            "miRNA": "mirna.csv",
            "Methylation": "methy.csv"
        },
        outcome_file="data/clinical/liver.csv",
        outcome_col="pathologic_T",
        id_col="sampleID",
        outcome_type="class",
        fix_tcga_ids=True
    ).to_dict(),
    
    DatasetConfig(
        name="Lung",
        base_path="data/lung",
        modalities={
            "Gene Expression": "exp.csv",
            "miRNA": "mirna.csv",
            "Methylation": "methy.csv"
        },
        outcome_file="data/clinical/lung.csv",
        outcome_col="pathologic_T",
        id_col="sampleID",
        outcome_type="class",
        fix_tcga_ids=True
    ).to_dict(),
    
    DatasetConfig(
        name="Melanoma",
        base_path="data/melanoma",
        modalities={
            "Gene Expression": "exp.csv",
            "miRNA": "mirna.csv",
            "Methylation": "methy.csv"
        },
        outcome_file="data/clinical/melanoma.csv",
        outcome_col="pathologic_T",
        id_col="sampleID",
        outcome_type="class",
        fix_tcga_ids=True
    ).to_dict(),
    
    DatasetConfig(
        name="Ovarian",
        base_path="data/ovarian",
        modalities={
            "Gene Expression": "exp.csv",
            "miRNA": "mirna.csv",
            "Methylation": "methy.csv"
        },
        outcome_file="data/clinical/ovarian.csv",
        outcome_col="clinical_stage",
        id_col="sampleID",
        outcome_type="class",
        fix_tcga_ids=True
    ).to_dict()
]
"""