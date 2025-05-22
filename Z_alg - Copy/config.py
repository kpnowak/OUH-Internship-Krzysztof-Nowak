#!/usr/bin/env python3
"""
Configuration module for the pipeline.
Contains constants and configurations used across the application.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union

# Constants
MAX_VARIABLE_FEATURES = 5000
MAX_COMPONENTS = 32
MAX_FEATURES = 32
N_JOBS = min(os.cpu_count() or 4, 4)  # Limit to 4 cores
OMP_BLAS_THREADS = min(4, os.cpu_count() or 4)

# Memory optimization settings
MEMORY_OPTIMIZATION = {
    "chunk_size": 1000,  # Process data in chunks of this size
    "cache_dir": "./.cache",  # Cache directory
    "cache_size": "2GB"  # Maximum cache size
}

# Parallel processing configuration
JOBLIB_PARALLEL_CONFIG = {
    'max_nbytes': '50M',  # Limit memory per worker
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
        "n_estimators": 200,
        "max_depth": 10,
        "max_features": "sqrt",
        "min_samples_leaf": 2,
        "n_jobs": 1,
        "random_state": 42
    },
    "RandomForestClassifier": {
        "n_estimators": 200,
        "max_depth": 10,
        "max_features": "sqrt",
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "n_jobs": 1,
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