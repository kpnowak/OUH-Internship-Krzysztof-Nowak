#!/usr/bin/env python3
"""
Configuration module for the pipeline.
Contains constants and configurations used across the application.
"""

import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union

# Suppress convergence warnings for cleaner output
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
except ImportError:
    # Fallback for older sklearn versions
    warnings.filterwarnings('ignore', message='.*did not converge.*')
    warnings.filterwarnings('ignore', message='.*Objective did not converge.*')

# Constants
MAX_VARIABLE_FEATURES = 5000  # Increased for high-memory server
MAX_COMPONENTS = 32  # Increased for high-memory server
MAX_FEATURES = 32    # Increased for high-memory server
N_JOBS = min(os.cpu_count() or 8, 8)  # Increased to 8 cores for server
OMP_BLAS_THREADS = min(4, os.cpu_count() or 4)

# High-memory server optimization (60GB RAM available)
MEMORY_OPTIMIZATION = {
    "chunk_size": 10000,  # Much larger chunks for high-memory systems
    "cache_dir": "./.cache",  # Cache directory
    "cache_size": "8GB",  # Increased cache size per type for 60GB system
    #"total_cache_limit": "8GB",  # Use ~50% of available RAM for caching
    "total_cache_limit": "32GB",  # Use ~50% of available RAM for caching
    "auto_clear_threshold": 0.85,  # Higher threshold for high-memory systems
    "memory_monitor_interval": 60,  # Less frequent monitoring for stable systems
    "shape_mismatch_auto_fix": True,  # Enable automatic shape mismatch fixing
    "alignment_loss_threshold": 0.3,  # Keep conservative data loss threshold
    "min_samples_threshold": 2,  # Minimum samples required after alignment
}

# High-memory server caching configuration (60GB RAM available)
CACHE_CONFIG = {
    #"selector_regression": {"maxsize": 32, "maxmemory_mb": 1000},  # 1GB limit
    #"selector_classification": {"maxsize": 32, "maxmemory_mb": 1000},  # 1GB limit  
    #"extractor_regression": {"maxsize": 32, "maxmemory_mb": 1000},  # 1GB limit
    #"extractor_classification": {"maxsize": 32, "maxmemory_mb": 1000},  # 1GB limit
    #"total_limit_mb": 4000,  # 4GB total limit (reduced from 8GB)
    "selector_regression": {"maxsize": 128, "maxmemory_mb": 8000},  # 8GB limit per cache
    "selector_classification": {"maxsize": 128, "maxmemory_mb": 8000},  # 8GB limit per cache
    "extractor_regression": {"maxsize": 128, "maxmemory_mb": 8000},  # 8GB limit per cache
    "extractor_classification": {"maxsize": 128, "maxmemory_mb": 8000},  # 8GB limit per cache
    "total_limit_mb": 32000,  # 32GB total limit (~50% of 60GB RAM)
    "eviction_strategy": "lru",  # Least Recently Used eviction
    "memory_check_interval": 300,  # Check memory usage every 5 minutes
}

# High-memory parallel processing configuration
JOBLIB_PARALLEL_CONFIG = {
    #'max_nbytes': '50M',  # Limit memory per worker
    'max_nbytes': '2G',  # Increased memory per worker for 60GB system
    'prefer': 'threads',  # Prefer threads over processes
    'require': 'sharedmem',  # Require shared memory
    'verbose': 0  # No verbose output
}

# Model optimization settings
MODEL_OPTIMIZATIONS = {
    "LinearRegression": {
        "fit_intercept": True,
        "n_jobs": 1
    },
    "RandomForestRegressor": {
        "n_estimators": 200,  # Restored to 200 for high-memory server
        "max_depth": 12,      # Increased depth for better performance with more memory
        "max_features": "sqrt",
        "min_samples_leaf": 2,  # Reduced for better model complexity
        "n_jobs": -1,  # Use all available cores
        "random_state": 42
    },
    "RandomForestClassifier": {
        "n_estimators": 200,  # Restored to 200 for high-memory server
        "max_depth": 12,      # Increased depth for better performance with more memory
        "max_features": "sqrt",
        "min_samples_leaf": 2,  # Reduced for better model complexity
        "class_weight": "balanced",
        "n_jobs": -1,  # Use all available cores
        "random_state": 42
    },
    "SVR": {
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
        "epsilon": 0.1,
        "tol": 1e-3,
        "cache_size": 500,
        "random_state": 42
    },
    "ElasticNet": {
        "alpha": 0.1,  # Higher regularization for better convergence
        "l1_ratio": 0.5,  # Balanced L1/L2 ratio
        "max_iter": 5000,  # Increased iterations for convergence
        "tol": 1e-4,  # Tolerance for convergence
        "selection": "cyclic",  # Coordinate descent selection
        "random_state": 42
    },
    "Lasso": {
        "alpha": 0.01,  # Lower alpha for better convergence (more aggressive than ElasticNet)
        "max_iter": 5000,  # Increased iterations for convergence
        "tol": 1e-4,  # Tolerance for convergence
        "selection": "cyclic",  # Coordinate descent selection
        "random_state": 42
    },
    "LogisticRegression": {
        "penalty": "l2",
        "C": 1.0,
        "solver": "liblinear",
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": 42
    },
    "SVC": {
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
        "probability": True,
        "class_weight": "balanced",
        "cache_size": 500,
        "random_state": 42
    }
}

# Enhanced early stopping configuration
EARLY_STOPPING_CONFIG = {
    "enabled": True,  # Enable/disable early stopping globally
    "patience": 10,  # Number of epochs to wait for improvement
    "min_delta": 1e-4,  # Minimum change to qualify as improvement
    "validation_split": 0.2,  # Fraction of training data to use for early stopping validation
    "restore_best_weights": True,  # Whether to restore best model weights
    "monitor_metric": "auto",  # Metric to monitor: "auto", "neg_mse", "accuracy", "r2"
    "verbose": 1,  # Verbosity level for early stopping (0=silent, 1=progress, 2=detailed)
    "adaptive_patience": True,  # Increase patience for complex models
    "max_patience": 50,  # Maximum patience for adaptive early stopping
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

# Sample retention warning configuration
SAMPLE_RETENTION_CONFIG = {
    "suppress_warnings_for_datasets": ["colon", "kidney", "liver"],  # Datasets with expected low retention
    "low_retention_threshold": 40,  # Threshold for low retention warnings (%)
    "moderate_retention_threshold": 70,  # Threshold for moderate retention warnings (%)
    "log_retention_details": True,  # Log detailed retention information
    "expected_low_retention_message": "Expected low sample retention for this dataset type",
}

# Robust feature selection configuration
FEATURE_SELECTION_CONFIG = {
    "fallback_enabled": True,  # Enable fallback feature selection methods
    "fallback_methods": ["mutual_info", "f_test", "variance"],  # Fallback methods in order
    "min_features": 1,  # Minimum number of features to select
    "max_features_ratio": 0.8,  # Maximum ratio of features to samples
    "error_tolerance": 3,  # Number of retries before giving up
    "adaptive_selection": True,  # Adapt selection based on data characteristics
}

# Improved extractor stability configuration  
EXTRACTOR_CONFIG = {
    "adaptive_components": True,  # Enable adaptive component selection
    "min_explained_variance": 0.8,  # Minimum explained variance for PCA
    "max_components_ratio": 0.9,  # Maximum ratio of components to features
    "fallback_to_pca": True,  # Fall back to PCA if other extractors fail
    "stability_checks": True,  # Perform stability checks on extracted features
    "numerical_stability": True,  # Enable numerical stability improvements
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
}

# MRMR Feature Selection Configuration
MRMR_CONFIG = {
    "fast_mode": True,  # Use fast approximations (correlation instead of MI for redundancy)
    "max_features_prefilter": 1000,  # Pre-filter to top N features before MRMR (0 = no prefilter)
    "n_neighbors": 3,  # Number of neighbors for MI estimation (lower = faster)
    "progress_logging": True,  # Log MRMR selection progress
    "fallback_on_error": True  # Fall back to mutual_info if MRMR fails
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
    nfeats_list: List[int] = field(default_factory=lambda: [8, 16, 32])
    ncomps_list: List[int] = field(default_factory=lambda: [4, 8, 16])
    
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
    DatasetConfig(
        name="TestRegression",
        base_path="test_data/regression",
        modalities={
            "Gene Expression": "exp.csv",
            "miRNA": "mirna.csv",
            "Methylation": "methy.csv"
        },
        outcome_file="clinical.csv",
        outcome_col="survival_time",
        id_col="sample_id",
        outcome_type="continuous",
        output_dir="output_regression",
        fix_tcga_ids=True
    ).to_dict()

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
    ).to_dict()

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