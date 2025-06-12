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
os.environ["OCTAVE_EXECUTABLE"] = r"C:\Users\krzys\AppData\Local\Programs\GNU Octave\Octave-10.2.0\mingw64\bin\octave-cli.exe"

# Suppress convergence warnings for cleaner output
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
except ImportError:
    # Fallback for older sklearn versions
    warnings.filterwarnings('ignore', message='.*did not converge.*')
    warnings.filterwarnings('ignore', message='.*Objective did not converge.*')

# Constants
MAX_VARIABLE_FEATURES = 5000  # OPTIMIZED: Reduced from 50000 for faster processing
MAX_COMPONENTS = 128  # OPTIMIZED: Reduced from 256 for faster processing
MAX_FEATURES = 512  # OPTIMIZED: Reduced from 1024 for faster processing
N_JOBS = min(os.cpu_count() or 8, 8)  # Increased to 8 cores for server
OMP_BLAS_THREADS = min(4, os.cpu_count() or 4)

# Feature and component selection values - OPTIMIZED FOR SPEED
N_VALUES_LIST = [32, 64, 128]  # OPTIMIZED: Reduced from [64, 128, 256] for faster processing

# Additional preprocessing parameters for small sample, high-dimensional data
# OPTIMIZED: Simplified preprocessing for better performance
PREPROCESSING_CONFIG = {
    "variance_threshold": 0.001,       # OPTIMIZED: Increased from 0.001 for more aggressive filtering
    "correlation_threshold": 0.95,    # OPTIMIZED: Reduced from 0.98 (but correlation filtering is disabled)
    "missing_threshold": 0.3,         # OPTIMIZED: Reduced from 0.5 for cleaner data
    "outlier_threshold": 4.0,         # More lenient for biological data
    "log_transform": True,           # OPTIMIZED: Disabled by default for speed
    "quantile_transform": True,      # OPTIMIZED: Disabled by default for speed
    "remove_low_variance": True,
    "remove_highly_correlated": True,  # OPTIMIZED: Disabled for performance (expensive operation)
    "handle_outliers": True,
    "impute_missing": True,
    "scaling_method": 'robust',       # Robust to outliers in genomic data
    "handle_missing": 'median',       # Better for genomic data
    "outlier_method": 'iqr',
    "outlier_std_threshold": 4.0,     # OPTIMIZED: More permissive for speed
    "normalize": True,
    "min_samples_per_feature": 3,    # Minimum samples required per feature
    "robust_scaling": True,          # Use robust scaling instead of standard scaling
    "feature_selection_method": "variance",  # Use variance-based selection first
}

# High-memory server optimization (60GB RAM available)
MEMORY_OPTIMIZATION = {
    "chunk_size": 10000,  # Much larger chunks for high-memory systems
    "cache_dir": "./.cache",  # Cache directory
    "cache_size": "8GB",  # Increased cache size per type for 60GB system
    "total_cache_limit": "8GB",  # Use ~50% of available RAM for caching
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
        "penalty": "l2",             # Changed to l2 for better stability
        "solver": "liblinear",
        "C": 1.0,                    # Increased C for less regularization
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

# Enhanced early stopping configuration - MORE AGGRESSIVE EARLY STOPPING
EARLY_STOPPING_CONFIG = {
    "enabled": True,  # Enable/disable early stopping globally
    "patience": 50,  # Much more patience
    "min_delta": 1e-6,  # Very small improvement threshold
    "validation_split": 0.2,  # Fraction of training data to use for early stopping validation
    "restore_best_weights": True,  # Whether to restore best model weights
    "monitor_metric": "auto",  # Metric to monitor: "auto", "neg_mse", "accuracy", "r2"
    "verbose": 0,  # Verbosity level for early stopping (0=silent, 1=progress, 2=detailed)
    "adaptive_patience": True,  # Increase patience for complex models
    "max_patience": 100,  # Increased maximum patience
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
    
    # Attention-weighted concatenation configuration
    "attention_weighted": {
        "enabled": True,
        "hidden_dim": 32,        # Hidden dimension for attention MLP
        "dropout_rate": 0.1,     # Dropout rate for regularization
        "learning_rate": 0.001,  # Learning rate for optimization
        "max_epochs": 100,       # Maximum training epochs
        "patience": 10,          # Early stopping patience
        "random_state": 42,
        "description": "Sample-specific weighting improved AML R² +0.05 and Colon MCC +0.04"
    },
    
    # Late-fusion stacking configuration
    "late_fusion_stacking": {
        "enabled": True,
        "cv_folds": 5,           # Cross-validation folds for meta-features
        "base_models": None,     # Use default models (RF, ElasticNet/Logistic, SVR/SVC)
        "random_state": 42,
        "description": "Uses per-omic model predictions as features; helps when one modality dominates"
    },
    
    # Strategy selection
    "default_strategy": "attention_weighted",  # Default fusion strategy when enabled
    "fallback_strategy": "weighted_concat",    # Fallback when upgrades fail
    "auto_strategy_selection": True,           # Automatically select best strategy based on data
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
    "variance_threshold": 0.0001,  # Very low variance threshold
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
    "variance_threshold": 0.0001,  # Very permissive
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

# ElasticNet - Much less regularization for genomic data
ELASTIC_NET_CONFIG = {
    'alpha': 0.001,  # Very low regularization
    'l1_ratio': 0.1,  # Favor Ridge over Lasso
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
    'dropout_rate': 0.1,  # Minimal dropout
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
]

"""
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
]

"""
    DatasetConfig(
        name="TestClassification",
        base_path="test_data/classification",
        modalities={
            "Gene Expression": "exp.csv",
            "miRNA": "mirna.csv",
            "Methylation": "methy.csv"
        },
        outcome_file="clinical.csv",
        outcome_col="status",
        id_col="sample_id",
        outcome_type="class",
        output_dir="output_classification",
        fix_tcga_ids=True
    ).to_dict(),

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