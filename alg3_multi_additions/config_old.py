#!/usr/bin/env python3

import os
import psutil
import platform
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union

# System-specific settings
TOTAL_RAM = psutil.virtual_memory().total
PHYS_CORES = psutil.cpu_count(logical=False)
OUTER_PROCS = max(1, int(PHYS_CORES * 0.9))
OMP_BLAS_THREADS = 1

# Memory optimization settings
MAX_ARRAY_SIZE = int(TOTAL_RAM * 0.8)  # 80% of RAM for arrays
CACHE_SIZE = int(TOTAL_RAM * 0.1)      # 10% of RAM for cache
SPARSE_THRESHOLD = 0.1  # Use sparse matrices if memory usage would be reduced by this factor
MEMORY_OPTIMIZATION = {
    "use_sparse": True,
    "dtype": "float32",
    "chunk_size": 50_000,  # Adjusted for Windows stability
    "max_memory_usage": 0.85,  # Conservative memory usage for Windows
    "max_array_size": MAX_ARRAY_SIZE,
    "cache_size": CACHE_SIZE
}

# Feature limits
MAX_VARIABLE_FEATURES = 5_000  # cap features to top-variance slice
MAX_COMPONENTS = 32
MAX_FEATURES = 32

# Parallelization settings
N_JOBS = OUTER_PROCS

# Thread control
os.environ.update({
    "OMP_NUM_THREADS": str(OMP_BLAS_THREADS),
    "OPENBLAS_NUM_THREADS": str(OMP_BLAS_THREADS),
    "MKL_NUM_THREADS": str(OMP_BLAS_THREADS),
    "NUMEXPR_NUM_THREADS": str(OMP_BLAS_THREADS),
})

# Joblib parallel config
JOBLIB_PARALLEL_CONFIG = {
    "n_jobs": OUTER_PROCS,
    "backend": "loky",
    "prefer": "processes",
    "batch_size": "auto",
    "verbose": 0,
    "max_nbytes": "100M",
    "temp_folder": os.path.join(os.getcwd(), "temp_joblib")
}

# Base path for test data
BASE_TEST_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test_data"))
# Base path for output
BASE_OUTPUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Regression datasets
REGRESSION_DATASETS = [
    DatasetConfig(
        name="TestRegression",
        base_path=os.path.join(BASE_TEST_PATH, "regression"),
        modalities={
            "Gene Expression": "exp.csv",
            "miRNA": "mirna.csv",
            "Methylation": "methy.csv"
        },
        outcome_file="clinical.csv",
        outcome_col="survival_time",
        id_col="sample_id",
        outcome_type="continuous",
        output_dir=os.path.join(BASE_OUTPUT_PATH, "output_regression"),
        fix_tcga_ids=True
    ).to_dict()
]

"""
    {
        "name": "AML",
        "clinical_file": "clinical/aml.csv",
        "omics_dir": "aml",
        "id_col": "sampleID",
        "outcome_col": "lab_procedure_bone_marrow_blast_cell_outcome_percent_value",
    },
    {
        "name": "Sarcoma",
        "clinical_file": "clinical/sarcoma.csv",
        "omics_dir": "sarcoma",
        "id_col": "metsampleID",
        "outcome_col": "pathologic_tumor_length",
    }
"""

# Classification datasets
CLASSIFICATION_DATASETS = [
    DatasetConfig(
        name="TestClassification",
        base_path=os.path.join(BASE_TEST_PATH, "classification"),
        modalities={
            "Gene Expression": "exp.csv",
            "miRNA": "mirna.csv",
            "Methylation": "methy.csv"
        },
        outcome_file="clinical.csv",
        outcome_col="status",
        id_col="sample_id",
        outcome_type="class",
        output_dir=os.path.join(BASE_OUTPUT_PATH, "output_classification"),
        fix_tcga_ids=True
    ).to_dict()
]

"""
    {
        "name": "Breast",
        "clinical_file": "clinical/breast.csv",
        "omics_dir": "breast",
        "id_col": "sampleID",
        "outcome_col": "pathologic_T",
    },
    ----------------------------------------
    {
        "name": "Colon",
        "clinical_file": "clinical/colon.csv",
        "omics_dir": "colon",
        "id_col": "sampleID",
        "outcome_col": "pathologic_T",
    },
    {
        "name": "Kidney",
        "clinical_file": "clinical/kidney.csv",
        "omics_dir": "kidney",
        "id_col": "submitter_id.samples",
        "outcome_col": "pathologic_T",
    },
    {
        "name": "Liver",
        "clinical_file": "clinical/liver.csv",
        "omics_dir": "liver",
        "id_col": "sampleID",
        "outcome_col": "pathologic_T",
    },
    {
        "name": "Lung",
        "clinical_file": "clinical/lung.csv",
        "omics_dir": "lung",
        "id_col": "sampleID",
        "outcome_col": "pathologic_T",
    },
    {
        "name": "Melanoma",
        "clinical_file": "clinical/melanoma.csv",
        "omics_dir": "melanoma",
        "id_col": "sampleID",
        "outcome_col": "pathologic_T",
    },
    {
        "name": "Ovarian",
        "clinical_file": "clinical/ovarian.csv",
        "omics_dir": "ovarian",
        "id_col": "sampleID",
        "outcome_col": "clinical_stage",
    }
"""

# Configuration for missing modalities simulation
MISSING_MODALITIES_CONFIG = {
    "enabled": True,  # Set to False to disable missing modalities simulation
    "missing_percentages": [0.0, 0.25],  # Percentages of samples with missing modalities
    "random_seed": 42,  # Base random seed for reproducibility
    "modality_names": ["Gene Expression", "Methylation", "miRNA"],  # Order must match data_modalities
    "cv_fold_seed_offset": 1_000  # Offset to add to random seed for each CV fold
}