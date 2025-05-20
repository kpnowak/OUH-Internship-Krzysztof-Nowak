#!/usr/bin/env python3

# Standard library imports
import os
import time
import threading
import warnings
from contextlib import contextmanager
from typing import Dict, List, Tuple, Optional, Union, Any

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import joblib
from joblib import Parallel, delayed, parallel_config, Memory
from threadpoolctl import threadpool_limits
import psutil
import seaborn as sns

# Scikit-learn imports
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import (
    LinearRegression, Lasso, ElasticNet, LogisticRegression
)
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    RandomForestRegressor as RF_for_BorutaReg,
    RandomForestClassifier as RF_for_BorutaClf
)
from sklearn.svm import SVR, SVC
from sklearn.decomposition import (
    PCA, NMF, FastICA, FactorAnalysis, KernelPCA
)
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, matthews_corrcoef, RocCurveDisplay
)
from sklearn.feature_selection import (
    mutual_info_regression, mutual_info_classif,
    f_regression, f_classif, chi2, SelectKBest,
    SelectFromModel
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from boruta import BorutaPy
from alg3_multi_additions.modality_imputer import ModalityImputer

# Constants and configuration
from config import (
    MAX_VARIABLE_FEATURES,
    MAX_COMPONENTS,
    MAX_FEATURES,
    MEMORY_OPTIMIZATION,
    JOBLIB_PARALLEL_CONFIG,
    OMP_BLAS_THREADS,
    REGRESSION_DATASETS,
    CLASSIFICATION_DATASETS,
    MISSING_MODALITIES_CONFIG,
    N_JOBS
)

# Update JOBLIB_PARALLEL_CONFIG for better memory management
JOBLIB_PARALLEL_CONFIG.update({
    'max_nbytes': '50M',  # Limit memory per worker
    'prefer': 'threads',  # Prefer threads over processes
    'require': 'sharedmem',  # Require shared memory
    'n_jobs': min(N_JOBS, os.cpu_count() or 1),  # Limit number of jobs
    'batch_size': 'auto',  # Automatic batch size
    'backend': 'threading'  # Use threading backend
})

# Local constants
DTYPE = np.float32
CHUNK_SIZE = MEMORY_OPTIMIZATION["chunk_size"]

# Memory caching setup with reduced size
MEM = Memory(location='./cache', verbose=0, bytes_limit="2GB")

# Model-specific optimizations with reduced memory usage
MODEL_OPTIMIZATIONS = {
    "RandomForest": {
        "n_estimators": 50,  # Reduced for memory efficiency
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "n_jobs": 1,
        "random_state": 42,
        "verbose": 0
    },
    "RandomForestClassifier": {
        "n_estimators": 50,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "n_jobs": 1,
        "random_state": 42
    },
    "LinearRegression": {
        "n_jobs": 1,
        "copy_X": False
    },
    "SVR": {
        "kernel": 'rbf',
        "cache_size": 200,  # Reduced cache size
        "max_iter": 1000,
        "tol": 1e-3
    }
}

# Thread control with reduced thread count
os.environ.update({
    "OMP_NUM_THREADS":      str(min(OMP_BLAS_THREADS, 4)),
    "OPENBLAS_NUM_THREADS": str(min(OMP_BLAS_THREADS, 4)),
    "MKL_NUM_THREADS":      str(min(OMP_BLAS_THREADS, 4)),
    "NUMEXPR_NUM_THREADS":  str(min(OMP_BLAS_THREADS, 4)),
})

# Add at the top of the file with other global variables
_selector_cache = {
    'sel_reg': {},
    'sel_clf': {},
    'extr_clf': {}
}

# ──────────────────────────────────────────────────────────────────────────────
#  🆕  GLOBAL SETTINGS
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
#  🆕  TOP-VARIABILITY FEATURE FILTER
# ──────────────────────────────────────────────────────────────────────────────
def _keep_top_variable_rows(df: pd.DataFrame,
                            k: int = MAX_VARIABLE_FEATURES) -> pd.DataFrame:
    """
    Keep at most *k* rows with the highest variance across samples.

    The omics matrices in this project are all shaped (features × samples),
    so we compute `row-variance`.  Sparse frames are handled efficiently
    with `toarray()` fallback if needed.

    Parameters
    ----------
    df : pd.DataFrame (features × samples)
    k  : int – number of rows to keep (default = 5 000)

    Returns
    -------
    pd.DataFrame containing ≤ k rows.
    """
    if df is None or df.shape[0] <= k:          # nothing to trim
        return df

    # fast path for dense / SparseDataFrame
    try:
        vr = df.var(axis=1, skipna=True).values
    except Exception:
        # sparse → dense only for the small variance vector
        vr = pd.DataFrame(df.to_numpy(copy=False)).var(axis=1).values

    top_idx = np.argpartition(vr, -k)[-k:]        # unsorted top-k idx
    return df.iloc[top_idx]

# ──────────────────────────────────────────────────────────────────────────────
#  📌 HARDWARE-AWARE TUNING  –  Xeon Silver 4114 × 2  (20C/40T, 59 GB RAM)
# ──────────────────────────────────────────────────────────────────────────────

# ----- Constants ------------------------------------------------------------

# ----- Thread-backend Parallel helper ----------------------------------------
def TParallel(*args, **kwargs):
    """Thread-backend Parallel that never spawns new processes."""
    # Check if we're already in a parallel context
    try:
        active_backend = joblib.parallel.get_active_backend()[0].backend
        if active_backend == "loky":
            # If we're in a process, use single thread
            kwargs["n_jobs"] = 1
    except Exception:
        pass
    return Parallel(*args, backend="threading", **kwargs)

# Wrap heavy BLAS work in threadpoolctl context
from contextlib import contextmanager
from threadpoolctl import threadpool_limits

@contextmanager
def heavy_cpu_section(num_threads=OMP_BLAS_THREADS):
    """Context manager to limit thread count in heavy CPU sections."""
    with threadpool_limits(limits=num_threads):
        yield

# ──────────────────────────────────────────────────────────────────────────────

# Resource monitoring function
def log_resource_usage(stage: str):
    """Log current resource usage for diagnostic purposes."""
    if os.environ.get("DEBUG_RESOURCES", "0") == "1":
        print(f"[INFO] {stage} - RAM used: {psutil.virtual_memory().percent}%, "
              f"CPU: {psutil.cpu_percent(interval=1)}%")

###############################################################################
# B) ID FIX & CUSTOM PARSE
###############################################################################

def fix_tcga_id_slicing(raw_str: str) -> str:
    """
    Convert e.g. 'TCGA-3C-AAAU-01-XYZ' => 'TCGA.3C.AAAU.01'
    by slicing to first 4 blocks if >=4, else rejoin with '.'.
    """
    if not isinstance(raw_str, str):
        return str(raw_str)
    s = raw_str.strip().strip('"')
    parts = s.split('-')
    if len(parts) >= 4:
        parts = parts[:4]
    return '.'.join(parts)

def custom_parse_outcome(val):
    """
    If '|' => take max of splitted floats,
    else parse float. If fails => np.nan
    """
    if isinstance(val, str):
        st = val.strip().strip('"')
        if '|' in st:
            try:
                return max(float(x) for x in st.split('|'))
            except:
                return np.nan
        else:
            try:
                return float(st)
            except:
                return np.nan
    else:
        return float(val) if pd.notna(val) else np.nan

###############################################################################
# C) LOADING & PREPARATION
###############################################################################

def try_read_file(path: str, clinical_cols: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    """
    Try to read a file, return None if it fails.

    Parameters
    ----------
    path            Path to the file
    clinical_cols   List of clinical column names to keep

    Returns
    -------
    pd.DataFrame | None
    """
    try:
        print(f"\nAttempting to read: {path}")
        
        # First check if file exists
        if not os.path.exists(path):
            print(f"File does not exist: {path}")
            return None
            
        # Read first few lines to check format
        with open(path, 'r') as f:
            first_lines = [next(f) for _ in range(3)]
        print(f"First few lines of {path}:")
        for line in first_lines:
            print(line.strip())
            
        if clinical_cols is None:
            # Try reading with different separators
            for sep in [',', '\t', ';', ' ']:
                try:
                    # First try standard reading
                    df = pd.read_csv(path, index_col=0, sep=sep)
                    if df.shape[1] > 0:  # Only use if we got some columns
                        print(f"Successfully read {path} with separator '{sep}', shape {df.shape}")
                        return df
                except Exception as e:
                    print(f"Failed to read with separator '{sep}': {str(e)}")
                    try:
                        # If standard reading fails, try reading with sample IDs in first line
                        df = pd.read_csv(path, sep=sep, header=None)
                        if df.shape[0] >= 2:  # Need at least 2 rows (sample IDs and feature names)
                            # First row contains sample IDs
                            sample_ids = df.iloc[0].values
                            # Second row contains feature names
                            feature_names = df.iloc[1].values
                            # Data starts from third row
                            data = df.iloc[2:].values
                            
                            # Create DataFrame with proper structure
                            df = pd.DataFrame(data, columns=sample_ids, index=feature_names)
                            
                            # Clean up column names (remove quotes and spaces)
                            df.columns = df.columns.str.replace('"', '').str.strip()
                            
                            # Convert data to numeric, handling any errors
                            df = df.apply(pd.to_numeric, errors='coerce')
                            
                            print(f"Successfully read {path} with custom format, shape {df.shape}")
                            return df
                    except Exception as e2:
                        print(f"Failed to read with custom format: {str(e2)}")
                        continue
            
            print(f"Failed to read {path} with any separator or format")
            return None
        
        # For clinical data
        header = pd.read_csv(path, nrows=0)
        actual_cols = header.columns.tolist()
        print(f"Available columns in {path}: {actual_cols}")
        
        # Filter clinical_cols to only include those that exist in the file
        valid_cols = [col for col in clinical_cols if col in actual_cols]
        
        if not valid_cols:
            print(f"Warning: None of the specified clinical columns found in {path}")
            return None
            
        # Read the file with only the valid columns, ensuring no duplicate index column
        usecols = list(dict.fromkeys([header.columns[0]] + valid_cols))
        df = pd.read_csv(path, index_col=0, usecols=usecols)
        print(f"Successfully read {path} with shape {df.shape}")
        
        # Handle categorical columns properly
        for col in df.columns:
            if df[col].dtype == 'object':
                # Keep categorical columns as is
                continue
            # Convert numeric columns to float32
            df[col] = df[col].astype(DTYPE)
            
        return df
    except Exception as e:
        print(f"Error reading {path}: {str(e)}")
        return None

def load_omics_and_clinical(ds_config, is_regression=True):
    odir = ds_config["omics_dir"]
    
    file_paths = [
        os.path.join(odir, "exp.csv"),
        os.path.join(odir, "methy.csv"),
        os.path.join(odir, "mirna.csv")
    ]
    
    print(f"\nLoading data from directory: {odir}")
    print(f"Looking for files: {file_paths}")
    
    # Read all files in parallel with threading backend
    results = TParallel(n_jobs=N_JOBS)(
        delayed(try_read_file)(path) for path in file_paths
    )
    
    exp_df, methy_df, mirna_df = results
    
    # Print shape information for debugging
    print("\nData shapes after loading:")
    print(f"Expression data: {exp_df.shape if exp_df is not None else 'None'}")
    print(f"Methylation data: {methy_df.shape if methy_df is not None else 'None'}")
    print(f"miRNA data: {mirna_df.shape if mirna_df is not None else 'None'}")
    
    # 🆕  limit each modality to the 5 000 most-variable features
    # Only process non-None dataframes
    if exp_df is not None:
        exp_df = _keep_top_variable_rows(exp_df, MAX_VARIABLE_FEATURES)
        print(f"Expression data after filtering: {exp_df.shape}")
    if methy_df is not None:
        methy_df = _keep_top_variable_rows(methy_df, MAX_VARIABLE_FEATURES)
        print(f"Methylation data after filtering: {methy_df.shape}")
    if mirna_df is not None:
        mirna_df = _keep_top_variable_rows(mirna_df, MAX_VARIABLE_FEATURES)
        print(f"miRNA data after filtering: {mirna_df.shape}")
    
    # Read clinical data with optimized settings
    # Handle potential key collision between id_col and outcome_col
    dtype_dict = {ds_config["id_col"]: str}
    if is_regression and ds_config["outcome_col"] != ds_config["id_col"]:
        dtype_dict[ds_config["outcome_col"]] = DTYPE
    
    clinical_df = pd.read_csv(
        ds_config["clinical_file"],
        sep=None,
        engine='python',
        dtype=dtype_dict,
        memory_map=True
    )
    print(f"Clinical data shape: {clinical_df.shape}")
    
    return exp_df, methy_df, mirna_df, clinical_df

def strip_and_slice_columns(col_list):
    newcols = []
    for c in col_list:
        s2 = c.strip().strip('"')
        s3 = fix_tcga_id_slicing(s2)
        newcols.append(s3)
    return newcols

def prepare_data(ds_config, exp_df, methy_df, mirna_df, clinical_df, is_regression=True):
    id_col = ds_config["id_col"]
    out_col = ds_config["outcome_col"]

    print("\nPreparing data:")
    print(f"ID column: {id_col}")
    print(f"Outcome column: {out_col}")

    # Create empty DataFrames for missing modalities
    if exp_df is None:
        exp_df = pd.DataFrame()
    if methy_df is None:
        methy_df = pd.DataFrame()
    if mirna_df is None:
        mirna_df = pd.DataFrame()

    # --- ONLY FOR KIDNEY: strip trailing 'A' from the ID column in memory ---
    if ds_config["name"] == "Kidney":
        def remove_trailing_A(s):
            if isinstance(s, str) and s.endswith('A'):
                return s[:-1]
            return s

        clinical_df_raw = clinical_df.copy()
        if id_col not in clinical_df_raw.columns:
            raise ValueError(f"ID col '{id_col}' not found in clinical data.")
        clinical_df_raw[id_col] = clinical_df_raw[id_col].apply(remove_trailing_A)
        clinical_df_raw[id_col] = clinical_df_raw[id_col].apply(fix_tcga_id_slicing)
    else:
        clinical_df_raw = clinical_df.copy()
        if id_col not in clinical_df_raw.columns:
            raise ValueError(f"ID col '{id_col}' not found in clinical data.")
        clinical_df_raw[id_col] = clinical_df_raw[id_col].apply(fix_tcga_id_slicing)

    if out_col not in clinical_df_raw.columns:
        raise ValueError(f"Outcome col '{out_col}' not found in clinical data.")

    clinical_df = clinical_df_raw.copy()

    # Parse outcomes depending on whether this is a regression or classification task.
    if is_regression:
        clinical_df[out_col] = clinical_df[out_col].apply(custom_parse_outcome)
        clinical_df = clinical_df.dropna(subset=[out_col]).copy()
        y = clinical_df[out_col].astype(float)
    else:
        raw_labels = clinical_df[out_col].astype(str).str.strip().str.replace('"', '')
        raw_labels = raw_labels.replace(['', 'NA', 'NaN', 'nan'], np.nan)
        clinical_df[out_col] = raw_labels
        clinical_df = clinical_df.dropna(subset=[out_col]).copy()
        clinical_df[out_col] = clinical_df[out_col].astype('category')
        y = clinical_df[out_col].cat.codes

    # Clean up column names in each omics dataset.
    exp_df.columns = strip_and_slice_columns(exp_df.columns)
    methy_df.columns = strip_and_slice_columns(methy_df.columns)
    mirna_df.columns = strip_and_slice_columns(mirna_df.columns)

    data_modalities = {
        "Gene Expression": exp_df,
        "Methylation": methy_df,
        "miRNA": mirna_df
    }

    # Drop empty modalities before intersection
    data_modalities = {n: df for n, df in data_modalities.items() if df.shape[1] > 0}
    
    print("\nModality shapes after cleaning:")
    for name, df in data_modalities.items():
        print(f"{name}: {df.shape}")
    
    # Guard against all empty modalities
    if not data_modalities:
        raise ValueError("No valid omics data found - all modalities are empty")

    # Intersection of sample IDs
    common_ids = set(clinical_df[id_col])
    print(f"\nInitial common IDs (from clinical): {len(common_ids)}")
    
    for name, df_mod in data_modalities.items():
        common_ids = common_ids.intersection(df_mod.columns)
        print(f"Common IDs after {name}: {len(common_ids)}")
    
    common_ids = sorted(list(common_ids))

    if len(common_ids) == 0 or y.shape[0] == 0:
        print(f"No overlapping or no valid samples => skipping {ds_config['name']}")
        return None, None, None, None

    clinical_filtered = clinical_df[clinical_df[id_col].isin(common_ids)].copy()
    clinical_filtered = clinical_filtered.sort_values(id_col).reset_index(drop=True)

    sub_mapping = {}
    for i, row in clinical_filtered.iterrows():
        sid = row[id_col]
        sub_mapping[sid] = row[out_col]

    final_y_vals = []
    for sid in common_ids:
        final_y_vals.append(sub_mapping[sid])
    y_series = pd.Series(final_y_vals, name="TARGET").reset_index(drop=True)

    print(f"\nFinal data shapes:")
    print(f"Clinical data: {clinical_filtered.shape}")
    print(f"Target values: {y_series.shape}")
    for name, df in data_modalities.items():
        print(f"{name}: {df.shape}")

    return data_modalities, common_ids, y_series, clinical_filtered

###############################################################################
# C) MISSING MODALITIES UTILITIES
###############################################################################

def process_with_missing_modalities(
    data_modalities: dict,
    common_ids: list,
    missing_percentage: float,
    fold_idx: int,
):
    """Return **copies** of the modality dataframes with NaNs for missing columns."""
    if not (MISSING_MODALITIES_CONFIG["enabled"] and missing_percentage):
        return {k: v.copy() for k, v in data_modalities.items()}

    n_samples = len(common_ids)
    n_modalities = len(data_modalities)  # Use actual number of modalities
    
    # Verify modality names match if config is provided
    if "modality_names" in MISSING_MODALITIES_CONFIG:
        config_modalities = set(MISSING_MODALITIES_CONFIG["modality_names"])
        actual_modalities = set(data_modalities.keys())
        if config_modalities != actual_modalities:
            raise ValueError(
                f"Modality mismatch: config has {config_modalities}, "
                f"but data has {actual_modalities}"
            )

    rng = np.random.RandomState(
        MISSING_MODALITIES_CONFIG["random_seed"]
        + fold_idx * MISSING_MODALITIES_CONFIG["cv_fold_seed_offset"]
    )

    availability = np.ones((n_samples, n_modalities), dtype=np.int8)
    n_to_modify = int(n_samples * missing_percentage)
    samples_to_modify = rng.choice(n_samples, n_to_modify, replace=False)
    for s in samples_to_modify:
        to_drop = rng.choice(
            n_modalities, rng.randint(1, n_modalities), replace=False
        )
        availability[s, to_drop] = 0

    id_to_idx = {sid: idx for idx, sid in enumerate(common_ids)}
    modified = {}
    for i, (name, df) in enumerate(data_modalities.items()):
        mod_df = df.copy()
        missing_cols = [sid for sid in common_ids if availability[id_to_idx[sid], i] == 0]
        if missing_cols:
            # **keep** the columns, just fill with NaNs so downstream code still sees them
            mod_df.loc[:, missing_cols] = np.nan
        modified[name] = mod_df
    return modified

def _pad(arr: np.ndarray, target_cols: int) -> np.ndarray:
    diff = target_cols - arr.shape[1]
    if diff <= 0:
        return arr
    return np.pad(arr, ((0, 0), (0, diff)), mode="constant", constant_values=0)

def merge_modalities(*arrays: np.ndarray, strategy: str = "concat", 
                    imputer: Optional[ModalityImputer] = None, 
                    is_train: bool = True) -> np.ndarray:
    """
    Merge an arbitrary number of numpy arrays (same number of rows).

    Parameters
    ----------
    *arrays : np.ndarray
        Variable-length list of 2-D arrays (or None/empty)
    strategy : str, default="concat"
        Merge strategy: 'concat' | 'average' | 'sum' | 'max'
    imputer : Optional[ModalityImputer]
        Optional ModalityImputer instance for handling missing values
    is_train : bool, default=True
        Whether this is training data (True) or validation/test data (False)

    Returns
    -------
    np.ndarray
        The merged matrix (float32). Empty (0, 0) array if nothing usable was supplied.

    Raises
    ------
    ValueError
        If an unknown merge strategy is specified
    IndexError
        If arrays have incompatible shapes for the chosen strategy
    """
    # keep only non-empty, non-None inputs
    valid = [a for a in arrays if a is not None and getattr(a, "size", 0) > 0]
    if not valid:
        return np.empty((0, 0), dtype=np.float32)

    # Create a new imputer instance if none provided
    if imputer is None:
        imputer = ModalityImputer()

    # Ensure all arrays have the same number of rows
    n_rows = valid[0].shape[0]
    if not all(a.shape[0] == n_rows for a in valid):
        raise ValueError("All arrays must have the same number of rows")

    if strategy == "concat":
        # For concatenation, we don't need to pad
        merged = np.concatenate(valid, axis=1).astype(np.float32, copy=False)
        if is_train:
            return imputer.fit_transform(merged)
        else:
            return imputer.transform(merged)

    # For element-wise operations, we need to ensure all arrays have the same number of columns
    # We'll use the maximum number of columns as the target
    target_cols = max(a.shape[1] for a in valid)
    
    # Pad each array to the target number of columns
    padded = []
    for arr in valid:
        if arr.shape[1] < target_cols:
            # Pad with zeros if needed
            pad_width = ((0, 0), (0, target_cols - arr.shape[1]))
            padded_arr = np.pad(arr, pad_width, mode='constant', constant_values=0)
        else:
            padded_arr = arr
        padded.append(padded_arr)

    if strategy == "average":
        stacked = np.stack(padded, axis=0)
        with np.errstate(invalid="ignore"):
            numer = np.nansum(stacked, axis=0)
            denom = np.maximum(1, np.sum(~np.isnan(stacked), axis=0))
        merged = (numer / denom).astype(np.float32, copy=False)
    elif strategy == "sum":
        stacked = np.stack(padded, axis=0)
        with np.errstate(invalid="ignore"):
            merged = np.nansum(stacked, axis=0)
            all_nan_mask = np.all(np.isnan(stacked), axis=0)
            merged[all_nan_mask] = 0
        merged = merged.astype(np.float32, copy=False)
    elif strategy == "max":
        stacked = np.stack(padded, axis=0)
        with np.errstate(invalid="ignore"):
            min_vals = np.nanmin(stacked, axis=0)
            merged = np.nanmax(stacked, axis=0)
            all_nan_mask = np.all(np.isnan(stacked), axis=0)
            merged[all_nan_mask] = min_vals[all_nan_mask]
        merged = merged.astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unknown merge strategy '{strategy}'")

    if is_train:
        return imputer.fit_transform(merged)
    else:
        return imputer.transform(merged)

###############################################################################
# D) EXTRACTORS & SELECTORS
###############################################################################

# For regression
def get_regression_extractors():
    return {
        "PCA": PCA(random_state=42),
        "NMF": NMF(
            init='nndsvdar',
            random_state=42,
            max_iter=5000,  # Increased max iterations
            tol=1e-3,      # Relaxed tolerance
            beta_loss='frobenius',
            solver='mu'
        ),
        "ICA": FastICA(
            random_state=42,
            max_iter=5000,  # Increased max iterations
            tol=1e-3,      # Relaxed tolerance
            algorithm='parallel',
            whiten='unit-variance'
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
        )
    }

def get_regression_selectors():
    return {
        "MRMR": "mrmr_reg",
        "LASSO": "lasso",
        "ElasticNetFS": "enet",
        "f_regressionFS": "freg",
        "Boruta": "boruta_reg"
    }

# For classification
def get_classification_extractors():
    ica_params = {
        "max_iter": 10000,
        "whiten": "unit-variance",
        "whiten_solver": "svd"
    } if sklearn_version >= "1.3.0" else {
        "max_iter": 10000,
        "whiten": "unit-variance"
    }
    
    return {
        "PCA": PCA(),
        "ICA": FastICA(**ica_params),
        "LDA": LDA(),
        "FA": FactorAnalysis(),
        "KPCA": KernelPCA(kernel='rbf')
    }

def get_classification_selectors():
    return {
        "MRMR": "mrmr_clf",
        "fclassifFS": "fclassif",
        "LogisticL1": "logistic_l1",
        "Boruta": "boruta_clf",
        "Chi2FS": "chi2_selection"
    }

###############################################################################
# E) PLOTTING HELPERS
###############################################################################

def verify_plot_exists(plot_path: str) -> bool:
    """
    Verify that a plot file exists and is not empty.
    
    Parameters
    ----------
    plot_path : str
        Path to the plot file
        
    Returns
    -------
    bool
        True if plot exists and is not empty, False otherwise
    """
    try:
        return os.path.exists(plot_path) and os.path.getsize(plot_path) > 0
    except Exception:
        return False

def plot_regression_scatter(y_test, y_pred, title, out_path):
    """Plot regression scatter plot and return success status."""
    # Check if all values are NaN
    if np.isnan(y_test).all() or np.isnan(y_pred).all():
        print(f"Warning: All values are NaN for {title}, skipping plot")
        return False
        
    try:
        fig, ax = plt.subplots(figsize=(5,5))
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.set_title(title + ": Actual vs. Predicted")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        mn = min(min(y_test), min(y_pred))
        mx = max(max(y_test), max(y_pred))
        ax.plot([mn, mx], [mn, mx], 'r--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
        return verify_plot_exists(out_path)
    except Exception as e:
        print(f"Error creating scatter plot for {title}: {str(e)}")
        return False

def plot_regression_residuals(y_test, y_pred, title, out_path):
    """Plot regression residuals and return success status."""
    # Check if all values are NaN
    if np.isnan(y_test).all() or np.isnan(y_pred).all():
        print(f"Warning: All values are NaN for {title}, skipping plot")
        return False
        
    try:
        residuals = y_test - y_pred
        fig, ax = plt.subplots(figsize=(5,5))
        ax.scatter(y_pred, residuals, alpha=0.5)
        ax.axhline(0, color='r', linestyle='--')
        ax.set_title(title + ": Residual Plot")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residual")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
        return verify_plot_exists(out_path)
    except Exception as e:
        print(f"Error creating residuals plot for {title}: {str(e)}")
        return False

def plot_confusion_matrix(cm, class_labels, title, out_path):
    """Plot confusion matrix and return success status."""
    try:
        fig, ax = plt.subplots(figsize=(5,5))
        sns.heatmap(cm, annot=True, fmt='d',
                    cmap='Blues', xticklabels=class_labels, yticklabels=class_labels, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
        return verify_plot_exists(out_path)
    except Exception as e:
        print(f"Error creating confusion matrix for {title}: {str(e)}")
        return False

def plot_roc_curve_binary(model, X_test, y_test, class_labels, title, out_path):
    """Plot ROC curve for binary classification and return success status."""
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        fig, ax = plt.subplots(figsize=(5,5))
        disp = RocCurveDisplay.from_predictions(y_test, y_proba, name='Binary ROC', ax=ax)
        ax.set_title(title + " - ROC Curve")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
        return verify_plot_exists(out_path)
    except Exception as e:
        print(f"Error creating ROC curve for {title}: {str(e)}")
        return False

###############################################################################
# F) TRAIN & EVAL: REGRESSION
###############################################################################

def cached_fit_transform_selector_regression(selector, X, y, n_feats, fold_idx=None, ds_name=None):
    """Cached version of fit_transform for regression selectors."""
    key = f"{ds_name}_{fold_idx}_{selector.__class__.__name__}_{n_feats}"
    
    if key in _selector_cache['sel_reg']:
        return _selector_cache['sel_reg'][key]
    
    try:
        # Convert DataFrame to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        X_selected = selector.fit_transform(X, y)
        selected_features = np.arange(X.shape[1])[selector.get_support()]
        
        _selector_cache['sel_reg'][key] = (selected_features, X_selected)
        return selected_features, X_selected
    except Exception as e:
        print(f"Error in feature selection: {str(e)}")
        return None, None

def process_modality(modality_name: str, modality_df: pd.DataFrame, id_train: List[str], 
                    id_val: List[str], idx_test: List[int], y_train: np.ndarray,
                    extr_obj: Any, ncomps: int, idx_to_id: Dict[int, str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Process a single modality with the given extractor."""
    try:
        # Get data for each split
        df_train = modality_df.loc[:, id_train].transpose()
        df_val = modality_df.loc[:, id_val].transpose()
        id_test = [idx_to_id[i] for i in idx_test]
        df_test = modality_df.loc[:, id_test].transpose()
        
        # Convert to numeric and handle NaN values
        df_train = df_train.apply(pd.to_numeric, errors='coerce')
        df_val = df_val.apply(pd.to_numeric, errors='coerce')
        df_test = df_test.apply(pd.to_numeric, errors='coerce')
        
        # Compute mean from training data for imputation
        train_mean = df_train.mean()
        
        # Impute NaN values using training mean
        df_train = df_train.fillna(train_mean)
        df_val = df_val.fillna(train_mean)
        df_test = df_test.fillna(train_mean)
        
        # Convert to numpy arrays and ensure float32
        X_train = df_train.values.astype(np.float32)
        X_val = df_val.values.astype(np.float32)
        X_test = df_test.values.astype(np.float32)
        
        # Validate n_components
        n_samples, n_features = X_train.shape
        ncomps = min(ncomps, min(n_samples, n_features))
        
        # Create new extractor instance with validated n_components
        if isinstance(extr_obj, NMF):
            extractor = NMF(
                n_components=ncomps,
                init='nndsvdar',
                random_state=42,
                max_iter=5000,
                tol=1e-3,
                beta_loss='frobenius',
                solver='mu'
            )
            # Ensure non-negative data for NMF
            X_train = np.maximum(X_train, 1e-10)
            X_val = np.maximum(X_val, 1e-10)
            X_test = np.maximum(X_test, 1e-10)
        elif isinstance(extr_obj, PCA):
            extractor = PCA(n_components=ncomps, random_state=42)
        elif isinstance(extr_obj, FastICA):
            extractor = FastICA(
                n_components=ncomps,
                random_state=42,
                max_iter=5000,
                tol=1e-3,
                algorithm='parallel',
                whiten='unit-variance'
            )
        elif isinstance(extr_obj, FactorAnalysis):
            extractor = FactorAnalysis(
                n_components=ncomps,
                random_state=42,
                max_iter=5000,
                tol=1e-3
            )
        elif isinstance(extr_obj, PLSRegression):
            extractor = PLSRegression(
                n_components=ncomps,
                max_iter=5000,
                tol=1e-3
            )
            # PLS requires target values
            if y_train is None:
                raise ValueError("Target values (y) are required for PLS")
            X_train = extractor.fit_transform(X_train, y_train)[0]
            X_val = extractor.transform(X_val)
            X_test = extractor.transform(X_test)
            return X_train, X_val, X_test
        else:
            raise ValueError(f"Unknown extractor type: {type(extr_obj)}")
        
        # Fit and transform with memory-efficient operations
        with heavy_cpu_section():
            X_train = extractor.fit_transform(X_train)
            X_val = extractor.transform(X_val)
            X_test = extractor.transform(X_test)
        
        return X_train, X_val, X_test
        
    except Exception as e:
        print(f"Error processing modality {modality_name}: {str(e)}")
        return None, None, None

def train_evaluate_model(model_name, model, X_train, y_train, X_val, fold_idx=None):
    """Train and evaluate a model."""
    try:
        if model_name == 'SVR':
            model = SVR(**MODEL_OPTIMIZATIONS["SVR"])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        return model, y_pred
    except Exception as e:
        print(f"Warning: Failed to train {model_name} in fold {fold_idx}: {str(e)}")
        return None, None

def process_cv_fold(
    train_idx,
    val_idx,
    idx_temp,
    idx_test,
    y_temp,
    y_test,
    data_modalities,
    reg_models,
    extr_obj,
    ncomps,
    id_to_idx,
    idx_to_id,
    all_ids,
    missing_percentage,
    fold_idx,
    base_out,
    ds_name,
    extr_name,
):
    """Process a single CV fold, only saving metrics."""
    id_train = [idx_to_id[i] for i in train_idx]
    id_val = [idx_to_id[i] for i in val_idx]
    y_train = y_temp[train_idx]

    # Use common_ids for missing modality simulation
    modified_modalities = process_with_missing_modalities(
        data_modalities, all_ids, missing_percentage, fold_idx
    )

    # Process modalities in parallel with threading backend
    modality_results = TParallel(n_jobs=N_JOBS)(
        delayed(process_modality)(name, df, id_train, id_val, idx_test, y_train, extr_obj, ncomps, idx_to_id)
        for name, df in modified_modalities.items()
    )

    # Filter out None results
    valid_results = [r for r in modality_results if r is not None and all(x is not None and x.size > 0 for x in r)]

    if not valid_results:
        print(f"Warning: No valid data found for any modality in fold {fold_idx}")
        return {}

    # Create a new imputer instance for this fold
    fold_imputer = ModalityImputer()

    # Merge modalities with the fold-specific imputer - fit only once on training data
    X_train_merged = merge_modalities(*[r[0] for r in valid_results], imputer=fold_imputer, is_train=True)
    
    # Use the same fitted imputer for validation and test
    X_val_merged = merge_modalities(*[r[1] for r in valid_results], imputer=fold_imputer, is_train=False)
    X_test_merged = merge_modalities(*[r[2] for r in valid_results], imputer=fold_imputer, is_train=False)

    # Skip if no valid data after merging
    if X_train_merged.size == 0 or X_val_merged.size == 0 or X_test_merged.size == 0:
        print(f"Warning: No valid data after merging in fold {fold_idx}")
        return {}

    # Train and evaluate models in parallel with threading backend
    model_results = {}
    for model_name in reg_models:
        try:
            if model_name == "RandomForest":
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    n_jobs=N_JOBS,
                    random_state=fold_idx
                )
            elif model_name == "LinearRegression":
                model = LinearRegression(n_jobs=N_JOBS)
            elif model_name == "SVR":
                model = SVR(**MODEL_OPTIMIZATIONS["SVR"])
            else:
                continue

            # Train the model with heavy CPU section
            with heavy_cpu_section():
                model.fit(X_train_merged, y_train)

            # Make predictions
            y_val_pred = model.predict(X_val_merged)

            # Calculate metrics
            metrics = {
                'mse': mean_squared_error(y_temp[val_idx], y_val_pred),
                'mae': mean_absolute_error(y_temp[val_idx], y_val_pred),
                'r2': r2_score(y_temp[val_idx], y_val_pred)
            }
            
            model_results[model_name] = metrics
        except Exception as e:
            print(f"Warning: Failed to train {model_name} in fold {fold_idx}: {str(e)}")
            continue

    return model_results

def process_reg_extraction_combo_cv(
    ds_name, extr_name, extr_obj, ncomps, reg_models,
    data_modalities, all_ids, y, base_out,
    progress_count, reg_total_runs, test_size=0.2, n_splits=3
):
    progress_count[0] += 1
    run_idx = progress_count[0]
    print(f"[EXTRACT-REG CV] {run_idx}/{reg_total_runs} => {ds_name} | {extr_name}-{ncomps}")

    # Ensure output directories exist
    os.makedirs(base_out, exist_ok=True)
    os.makedirs(os.path.join(base_out, "models"), exist_ok=True)
    os.makedirs(os.path.join(base_out, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(base_out, "plots"), exist_ok=True)

    # Convert IDs to numeric indices
    id_to_idx = {id_: idx for idx, id_ in enumerate(all_ids)}
    idx_to_id = {idx: id_ for id_, idx in id_to_idx.items()}
    
    # Create numeric indices array
    indices = np.arange(len(all_ids))
    y_arr = np.array(y)

    # Split indices and y values
    idx_temp, idx_test, y_temp, y_test = train_test_split(
        indices, y_arr, test_size=test_size, random_state=0
    )

    # Process CV folds for each missing percentage
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    all_results = []
    
    for missing_percentage in MISSING_MODALITIES_CONFIG["missing_percentages"]:
        cv_results = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(idx_temp)):
            # Suppress PLS warnings for this fold
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                result = process_cv_fold(
                    train_idx, val_idx, idx_temp, idx_test, y_temp, y_test,
                    data_modalities, reg_models, extr_obj, ncomps, id_to_idx, idx_to_id,
                    all_ids, missing_percentage, fold_idx, base_out, ds_name, extr_name
                )
            
            # Skip if no valid results
            if not result:
                continue
                
            cv_results.append(result)
        
        # Skip if no valid results for this missing percentage
        if not cv_results:
            continue
            
        # Aggregate results for this missing percentage
        cv_metrics = {}
        for model_name in reg_models:
            # Only include results for models that have predictions
            valid_results = []
            for i, (_, val_idx) in enumerate(cv.split(idx_temp)):
                if i < len(cv_results) and model_name in cv_results[i]:
                    try:
                        valid_results.append(cv_results[i][model_name])
                    except (KeyError, ValueError, TypeError) as e:
                        # These are expected errors when accessing metrics
                        print(f"Warning: Failed to get metrics for {model_name} in fold {i}: {e}")
                        continue
                    except Exception as e:
                        # Log unexpected errors but continue
                        print(f"Warning: Unexpected error getting metrics for {model_name} in fold {i}: {e}")
                        continue
            
            if valid_results:
                # Average metrics across folds
                avg_metrics = {
                    k: np.mean([m[k] for m in valid_results]) 
                    for k in valid_results[0].keys()
                }
                cv_metrics[model_name] = avg_metrics

        # Add results for this missing percentage
        for model_name in reg_models:
            if model_name in cv_metrics:
                avg_mets = {
                    "Dataset": ds_name, "Workflow": "Extraction-CV",
                    "Extractor": extr_name, "n_components": ncomps,
                    "Model": model_name,
                    "Missing_Percentage": missing_percentage,
                    **cv_metrics[model_name]  # Include all metrics
                }
                all_results.append(avg_mets)

    # Save all results if any were generated
    if all_results:
        pd.DataFrame(all_results).to_csv(
            os.path.join(base_out, "metrics", f"{ds_name}_extraction_cv_metrics.csv"),
            index=False
        )

    # Train and save final model with all training data
    if all_results:
        # Get best model based on average MSE across folds, filtering out NaN values
        candidates = [(model, np.mean([r['mse'] for r in all_results if r['Model'] == model])) 
                     for model in reg_models]
        valid_candidates = [(model, mse) for model, mse in candidates if not np.isnan(mse)]
        
        if not valid_candidates:
            print(f"Warning: No valid models found for {ds_name} with {extr_name}-{ncomps}")
            # Write empty metrics file with warning
            warning_metrics = pd.DataFrame([{
                "Dataset": ds_name,
                "Workflow": "Extraction-Final",
                "Extractor": extr_name,
                "n_components": ncomps,
                "Model": "NONE",
                "mse": np.nan,
                "mae": np.nan,
                "r2": np.nan,
                "Warning": "All models produced NaN metrics"
            }])
            warning_metrics.to_csv(
                os.path.join(base_out, "metrics", f"{ds_name}_extraction_final_metrics.csv"),
                index=False
            )
            return all_results
            
        best_model_name = min(valid_candidates, key=lambda x: x[1])[0]

        # Process all training data
        id_train = [idx_to_id[i] for i in idx_temp]
        id_test = [idx_to_id[i] for i in idx_test]
        y_train = y_temp

        # Process modalities with threading backend
        with parallel_config(backend="threading"):
            modality_results = Parallel(n_jobs=N_JOBS)(
                delayed(process_modality)(name, df, id_train, id_test, idx_test, y_train, extr_obj, ncomps, idx_to_id)
                for name, df in data_modalities.items()
            )

        # Filter and merge results
        valid_results = [r for r in modality_results if r is not None and all(x is not None and x.size > 0 for x in r)]
        if valid_results:
            # Create a new imputer instance for the final model
            final_imputer = ModalityImputer()
            
            # Merge modalities with the final imputer
            X_train_merged = merge_modalities(*[r[0] for r in valid_results], imputer=final_imputer, is_train=True)
            X_test_merged = merge_modalities(*[r[2] for r in valid_results], imputer=final_imputer, is_train=False)

            # Train final model
            if best_model_name == "RandomForest":
                final_model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    n_jobs=N_JOBS,
                    random_state=fold_idx
                )
            elif best_model_name == "LinearRegression":
                final_model = LinearRegression(n_jobs=N_JOBS)
            elif best_model_name == "SVR":
                final_model = SVR(**MODEL_OPTIMIZATIONS["SVR"])

            # Train with heavy CPU section
            with heavy_cpu_section():
                final_model.fit(X_train_merged, y_train)
            y_test_pred = final_model.predict(X_test_merged)

            # Save final model and imputer
            joblib.dump(
                (final_model, final_imputer),
                os.path.join(base_out, "models", 
                           f"{ds_name}_{extr_name}_{ncomps}_{best_model_name}_final.pkl")
            )

            # Generate and save plots
            plot_prefix = f"{ds_name}_{extr_name}_{ncomps}_{best_model_name}_final"
            
            # Scatter plot
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_test_pred, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            plt.xlabel('True Values')
            plt.ylabel('Predictions')
            plt.title(f'{plot_prefix} - Scatter Plot')
            plt.savefig(os.path.join(base_out, "plots", f'{plot_prefix}_scatter.png'))
            plt.close()

            # Residual plot
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test_pred, y_test - y_test_pred, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predictions')
            plt.ylabel('Residuals')
            plt.title(f'{plot_prefix} - Residuals Plot')
            plt.savefig(os.path.join(base_out, "plots", f'{plot_prefix}_residuals.png'))
            plt.close()

            # Save final metrics
            final_metrics = {
                "Dataset": ds_name,
                "Workflow": "Extraction-Final",
                "Extractor": extr_name,
                "n_components": ncomps,
                "Model": best_model_name,
                "mse": mean_squared_error(y_test, y_test_pred),
                "mae": mean_absolute_error(y_test, y_test_pred),
                "r2": r2_score(y_test, y_test_pred)
            }
            
            pd.DataFrame([final_metrics]).to_csv(
                os.path.join(base_out, "metrics", f"{ds_name}_extraction_final_metrics.csv"),
                index=False
            )
    
    return all_results

def process_reg_selection_combo_cv(
    ds_name, sel_name, sel_code, n_feats, reg_models,
    data_modalities, all_ids, y, base_out,
    progress_count, reg_total_runs, test_size=0.2, n_splits=3
):
    progress_count[0] += 1
    run_idx = progress_count[0]
    print(f"[SELECT-REG CV] {run_idx}/{reg_total_runs} => {ds_name} | {sel_name}-{n_feats}")

    all_ids_arr = np.array(all_ids)
    y_arr = np.array(y)

    id_temp, id_test, y_temp, y_test = train_test_split(
        all_ids_arr, y_arr, test_size=test_size, random_state=0
    )

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    cv_metrics = {}

    # Create a shared imputer instance for all folds
    shared_imputer = ModalityImputer()

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(id_temp)):
        id_train, id_val = id_temp[train_idx], id_temp[val_idx]
        y_train, y_val = y_temp[train_idx], y_temp[val_idx]

        train_list, val_list = [], []
        for modality_name, df_mod in data_modalities.items():
            df_train = df_mod.loc[:, id_train].transpose().apply(pd.to_numeric, errors='coerce').fillna(0)
            selector = get_selector_object(sel_code, n_feats)
            chosen_cols, X_tr = cached_fit_transform_selector_regression(
                selector, df_train, y_train, n_feats, fold_idx, ds_name
            )
            df_val = df_mod.loc[:, id_val].transpose().apply(pd.to_numeric, errors='coerce').fillna(0)
            X_va = transform_selector_regression(df_val, chosen_cols)
            train_list.append(np.array(X_tr))
            val_list.append(np.array(X_va))

        for merge_str in ["concat", "average", "sum", "max"]:
            try:
                # Use shared imputer for both train and validation
                X_tr_m = merge_modalities(*train_list, strategy=merge_str, imputer=shared_imputer, is_train=True)
                X_va_m = merge_modalities(*val_list, strategy=merge_str, imputer=shared_imputer, is_train=False)
            except Exception as e:
                print(f"Skipping merge '{merge_str}' fold {fold_idx}: {e}")
                continue

            # Create model instances from model names
            model_dict = {}
            for model_name in reg_models:
                if model_name == "RandomForest":
                    model_dict[model_name] = RandomForestRegressor(
                        n_estimators=100,
                        max_depth=None,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        n_jobs=N_JOBS,
                        random_state=fold_idx
                    )
                elif model_name == "LinearRegression":
                    model_dict[model_name] = LinearRegression(n_jobs=N_JOBS)
                elif model_name == "SVR":
                    model_dict[model_name] = SVR(**MODEL_OPTIMIZATIONS["SVR"])
                else:
                    raise ValueError(f"Unknown model: {model_name}")

            for model_name, model in model_dict.items():
                model, mets = train_regression_model(
                    X_tr_m, y_train, X_va_m, y_val,
                    model_name,
                    out_dir=None,
                    plot_prefix=""
                )
                key = (merge_str, model_name)
                cv_metrics.setdefault(key, []).append(mets)

    avg_cv_results = []
    for (merge_str, model_name), mets_list in cv_metrics.items():
        avg_mets = {k: np.mean([m[k] for m in mets_list]) for k in mets_list[0]}
        avg_mets.update({
            "Dataset": ds_name, "Workflow": "Selection-CV",
            "Selector": sel_name, "n_features": n_feats,
            "MergeStrategy": merge_str, "Model": model_name
        })

        train_list, test_list = [], []
        for modality_name, df_mod in data_modalities.items():
            df_train = df_mod.loc[:, id_temp].transpose().apply(pd.to_numeric, errors='coerce').fillna(0)
            chosen_cols, X_tr = cached_fit_transform_selector_regression(
                selector, df_train, y_temp, n_feats, fold_idx, ds_name
            )
            df_test = df_mod.loc[:, id_test].transpose().apply(pd.to_numeric, errors='coerce').fillna(0)
            X_te = transform_selector_regression(df_test, chosen_cols)
            train_list.append(np.array(X_tr))
            test_list.append(np.array(X_te))

        # Use shared imputer for final model
        X_tr_m = merge_modalities(*train_list, strategy=merge_str, imputer=shared_imputer, is_train=True)
        X_te_m = merge_modalities(*test_list, strategy=merge_str, imputer=shared_imputer, is_train=False)

        # Create model for final evaluation
        final_model, test_mets = train_regression_model(
            X_tr_m, y_temp, X_te_m, y_test,
            model_name,
            out_dir=os.path.join(base_out, "plots"),
            plot_prefix=f"{ds_name}_FINAL_SELECT_{sel_name}_{n_feats}_{merge_str}_{model_name}"
        )
        avg_mets = {f"Test_{k}": v for k, v in test_mets.items()}
        avg_cv_results.append(avg_mets)

        joblib.dump(final_model,
                    os.path.join(base_out, "models", f"{ds_name}_FINAL_SELECT_{sel_name}_{n_feats}_{merge_str}_{model_name}.pkl"))

    pd.DataFrame(avg_cv_results).to_csv(
        os.path.join(base_out, "metrics", f"{ds_name}_selection_cv_metrics.csv"),
        index=False
    )
    return avg_cv_results


###############################################################################
# L) HIGH‑LEVEL PROCESS FUNCTIONS (CLASSIFICATION) WITH CROSS‑VALIDATION
###############################################################################

def process_clf_extraction_combo_cv(
    ds_name, extr_name, extr_obj, ncomps, clf_models,
    data_modalities, all_ids, y, base_out,
    progress_count, clf_total_runs, test_size=0.2, n_splits=3
):
    progress_count[0] += 1
    run_idx = progress_count[0]
    print(f"[EXTRACT-CLF CV] {run_idx}/{clf_total_runs} => {ds_name} | {extr_name}-{ncomps}")

    all_ids_arr = np.array(all_ids)
    y_arr = np.array(y)

    id_temp, id_test, y_temp, y_test = train_test_split(
        all_ids_arr, y_arr, test_size=test_size, random_state=0
    )

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    cv_metrics = {}

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(id_temp)):
        id_train, id_val = id_temp[train_idx], id_temp[val_idx]
        y_train,  y_val  = y_temp[train_idx],  y_temp[val_idx]

        # Use process_with_missing_modalities for consistent missing modality simulation
        modified_modalities = process_with_missing_modalities(
            data_modalities, all_ids, MISSING_MODALITIES_CONFIG["missing_percentages"][0], fold_idx
        )

        train_list, val_list = [], []
        for modality_name, df_mod in modified_modalities.items():
            df_train = df_mod.loc[:, id_train].transpose().apply(pd.to_numeric, errors='coerce').fillna(0)
            fitted_extr, X_tr = cached_fit_transform_extractor_classification(
                df_train, y_train, extr_obj, ncomps, ds_name, modality_name
            )
            df_val = df_mod.loc[:, id_val].transpose().apply(pd.to_numeric, errors='coerce').fillna(0)
            X_va   = transform_extractor_classification(df_val, fitted_extr)
            train_list.append(X_tr)
            val_list.append(X_va)

        for merge_str in ["concat", "average", "sum", "max"]:
            try:
                # Create a shared imputer instance for this fold
                fold_imputer = ModalityImputer()
                X_tr_m = merge_modalities(*train_list, strategy=merge_str, imputer=fold_imputer, is_train=True)
                X_va_m = merge_modalities(*val_list, strategy=merge_str, imputer=fold_imputer, is_train=False)
            except Exception as e:
                print(f"Skipping merge '{merge_str}' fold {fold_idx}: {e}")
                continue

            # Create model instances from model names
            model_dict = {}
            for model_name in clf_models:
                if model_name == "LogisticRegression":
                    model_dict[model_name] = LogisticRegression(
                        penalty='l2',
                        solver='liblinear',
                        random_state=fold_idx
                    )
                elif model_name == "RandomForest":
                    model_dict[model_name] = RandomForestClassifier(
                        n_estimators=100,
                        random_state=fold_idx
                    )
                elif model_name == "SVC":
                    model_dict[model_name] = SVC(
                        kernel='rbf',
                        probability=True,
                        random_state=fold_idx
                    )
                else:
                    raise ValueError(f"Unknown classification model {model_name}")

            for model_name, model in model_dict.items():
                model, mets = train_classification_model(
                    X_tr_m, y_train, X_va_m, y_val,
                    model_name,
                    out_dir=os.path.join(base_out, "plots"),
                    plot_prefix=f"{ds_name}_fold_{fold_idx}_{extr_name}_{ncomps}_{merge_str}_{model_name}"
                )
                key = (merge_str, model_name)
                cv_metrics.setdefault(key, []).append(mets)

    avg_cv_results = []
    for (merge_str, model_name), mets_list in cv_metrics.items():
        avg_mets = {k: np.mean([m[k] for m in mets_list]) for k in mets_list[0]}
        avg_mets.update({
            "Dataset": ds_name, "Workflow": "Extraction-CV",
            "Extractor": extr_name, "n_components": ncomps,
            "MergeStrategy": merge_str, "Model": model_name
        })
        avg_cv_results.append(avg_mets)

        # Train final model
        if model_name == "LogisticRegression":
            final_model = LogisticRegression(penalty='l2', solver='liblinear', random_state=42)
        elif model_name == "RandomForest":
            final_model = RandomForestClassifier(**MODEL_OPTIMIZATIONS["RandomForest"])
        elif model_name == "SVC":
            final_model = SVC(kernel='rbf', probability=True, random_state=42)
        
        # Save the model
        joblib.dump(final_model,
                    os.path.join(base_out, "models", f"{ds_name}_FINAL_SELECT_{extr_name}_{ncomps}_{merge_str}_{model_name}.pkl"))

    pd.DataFrame(avg_cv_results).to_csv(
        os.path.join(base_out, "metrics", f"{ds_name}_extraction_cv_metrics.csv"),
        index=False
    )
    return avg_cv_results

def process_clf_selection_combo_cv(
    ds_name, sel_name, sel_code, n_feats, clf_models,
    data_modalities, all_ids, y, base_out,
    progress_count, clf_total_runs, test_size=0.2, n_splits=3
):
    progress_count[0] += 1
    run_idx = progress_count[0]
    print(f"[SELECT-CLF CV] {run_idx}/{clf_total_runs} => {ds_name} | {sel_name}-{n_feats}")

    all_ids_arr = np.array(all_ids)
    y_arr = np.array(y)

    id_temp, id_test, y_temp, y_test = train_test_split(
        all_ids_arr, y_arr, test_size=test_size, random_state=0
    )

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    cv_metrics = {}

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(id_temp)):
        id_train, id_val = id_temp[train_idx], id_temp[val_idx]
        y_train, y_val = y_temp[train_idx], y_temp[val_idx]

        # Use process_with_missing_modalities for consistent missing modality simulation
        modified_modalities = process_with_missing_modalities(
            data_modalities, all_ids, MISSING_MODALITIES_CONFIG["missing_percentages"][0], fold_idx
        )

        train_list, val_list = [], []
        for modality_name, df_mod in modified_modalities.items():
            df_train = df_mod.loc[:, id_train].transpose().apply(pd.to_numeric, errors='coerce').fillna(0)
            chosen_cols, X_tr = cached_fit_transform_selector_classification(
                df_train, y_train, sel_code, n_feats, ds_name, modality_name
            )
            df_val = df_mod.loc[:, id_val].transpose().apply(pd.to_numeric, errors='coerce').fillna(0)
            X_va = transform_selector_classification(df_val, chosen_cols)
            train_list.append(np.array(X_tr))
            val_list.append(np.array(X_va))

        for merge_str in ["concat", "average", "sum", "max"]:
            try:
                # Create a fresh imputer for each merge strategy
                fold_imputer = ModalityImputer()
                X_tr_m = merge_modalities(*train_list, strategy=merge_str, imputer=fold_imputer, is_train=True)
                X_va_m = merge_modalities(*val_list, strategy=merge_str, imputer=fold_imputer, is_train=False)
            except Exception as e:
                print(f"Skipping merge '{merge_str}' fold {fold_idx}: {e}")
                continue

            # Create model instances from model names
            model_dict = {}
            for model_name in clf_models:
                if model_name == "LogisticRegression":
                    model_dict[model_name] = LogisticRegression(
                        penalty='l2',
                        solver='liblinear',
                        random_state=fold_idx
                    )
                elif model_name == "RandomForest":
                    model_dict[model_name] = RandomForestClassifier(
                        n_estimators=100,
                        random_state=fold_idx
                    )
                elif model_name == "SVC":
                    model_dict[model_name] = SVC(
                        kernel='rbf',
                        probability=True,
                        random_state=fold_idx
                    )
                else:
                    raise ValueError(f"Unknown classification model {model_name}")

            for model_name, model in model_dict.items():
                model, mets = train_classification_model(
                    X_tr_m, y_train, X_va_m, y_val,
                    model_name,
                    out_dir=os.path.join(base_out, "plots"),
                    plot_prefix=f"{ds_name}_fold_{fold_idx}_{sel_name}_{n_feats}_{merge_str}_{model_name}"
                )
                key = (merge_str, model_name)
                cv_metrics.setdefault(key, []).append(mets)

    avg_cv_results = []
    for (merge_str, model_name), mets_list in cv_metrics.items():
        avg_mets = {k: np.mean([m[k] for m in mets_list]) for k in mets_list[0]}
        avg_mets.update({
            "Dataset": ds_name, "Workflow": "Selection-CV",
            "Selector": sel_name, "n_features": n_feats,
            "MergeStrategy": merge_str, "Model": model_name
        })
        avg_cv_results.append(avg_mets)

        # Train final model
        if model_name == "LogisticRegression":
            final_model = LogisticRegression(penalty='l2', solver='liblinear', random_state=42)
        elif model_name == "RandomForest":
            final_model = RandomForestClassifier(**MODEL_OPTIMIZATIONS["RandomForest"])
        elif model_name == "SVC":
            final_model = SVC(kernel='rbf', probability=True, random_state=42)
        
        # Save the model
        joblib.dump(final_model,
                    os.path.join(base_out, "models", f"{ds_name}_FINAL_SELECT_{sel_name}_{n_feats}_{merge_str}_{model_name}.pkl"))

    pd.DataFrame(avg_cv_results).to_csv(
        os.path.join(base_out, "metrics", f"{ds_name}_selection_cv_metrics.csv"),
        index=False
    )
    return avg_cv_results

###############################################################################
# M) MAIN
###############################################################################
def process_dataset(ds_conf, is_regression=True):
    """Process a single dataset with either regression or classification."""
    ds_name = ds_conf["name"]
    base_out = os.path.join("output_regression" if is_regression else "output_classification", ds_name)
    
    # Create output directories
    for subdir in ["", "models", "metrics", "plots"]:
        os.makedirs(os.path.join(base_out, subdir), exist_ok=True)
    
    print(f"\n--- Processing {ds_name} ({'Regression' if is_regression else 'Classification'}) ---")
    
    # Load and prepare data
    exp_df, methy_df, mirna_df, clinical_df = load_omics_and_clinical(ds_conf, is_regression)
    try:
        data_modalities, common_ids, y, clin_f = prepare_data(
            ds_conf, exp_df, methy_df, mirna_df, clinical_df, is_regression=is_regression
        )
    except ValueError as e:
        print(f"Skipping {ds_name} => {e}")
        return None
    
    if len(common_ids) == 0 or y.shape[0] == 0:
        print(f"No overlapping or no valid samples => skipping {ds_name}")
        return None
    
    return ds_name, data_modalities, common_ids, y, base_out

def run_extraction_pipeline(ds_name, data_modalities, common_ids, y, base_out, 
                          extractors, n_comps_list, models, progress_count, total_runs,
                          is_regression=True):
    """Run the extraction pipeline for a dataset."""
    extraction_jobs = [
        delayed(process_reg_extraction_combo_cv if is_regression else process_clf_extraction_combo_cv)(
            ds_name, extr_name, extr_obj, nc,
            models, data_modalities, common_ids, y, base_out,
            progress_count, total_runs, test_size=0.2, n_splits=3
        )
        for extr_name, extr_obj in extractors.items()
        for nc in n_comps_list
    ]
    
    # Process jobs in smaller batches to manage memory
    batch_size = 3  # Process 3 jobs at a time
    results = []
    for i in range(0, len(extraction_jobs), batch_size):
        batch = extraction_jobs[i:i + batch_size]
        batch_results = Parallel(**JOBLIB_PARALLEL_CONFIG)(batch)
        results.extend(batch_results)
        # Clear memory after each batch
        import gc
        gc.collect()
    
    return results

def run_selection_pipeline(ds_name, data_modalities, common_ids, y, base_out,
                         selectors, n_feats_list, models, progress_count, total_runs,
                         is_regression=True):
    """Run the selection pipeline for a dataset."""
    selection_jobs = [
        delayed(process_reg_selection_combo_cv if is_regression else process_clf_selection_combo_cv)(
            ds_name, sel_name, sel_code, nf,
            models, data_modalities, common_ids, y, base_out,
            progress_count, total_runs, test_size=0.2, n_splits=3
        )
        for sel_name, sel_code in selectors.items()
        for nf in n_feats_list
    ]
    
    # Process jobs in smaller batches to manage memory
    batch_size = 3  # Process 3 jobs at a time
    results = []
    for i in range(0, len(selection_jobs), batch_size):
        batch = selection_jobs[i:i + batch_size]
        batch_results = Parallel(**JOBLIB_PARALLEL_CONFIG)(batch)
        results.extend(batch_results)
        # Clear memory after each batch
        import gc
        gc.collect()
    
    return results

def process_regression_datasets():
    """Process all regression datasets."""
    print("=== REGRESSION BLOCK (AML, Sarcoma) ===")
    reg_extractors = get_regression_extractors()
    reg_selectors = get_regression_selectors()
    reg_models = ["LinearRegression", "RandomForest", "SVR"]
    n_comps_list = [8, 16, MAX_COMPONENTS]
    n_feats_list = [8, 16, MAX_FEATURES]
    
    reg_total_runs = (
        len(REGRESSION_DATASETS) * 
        (len(reg_extractors) * len(n_comps_list) + 
         len(reg_selectors) * len(n_feats_list))
    )
    progress_count_reg = [0]
    
    for ds_conf in REGRESSION_DATASETS:
        result = process_dataset(ds_conf, is_regression=True)
        if result is None:
            continue
            
        ds_name, data_modalities, common_ids, y, base_out = result
        
        # Run extraction and selection pipelines
        run_extraction_pipeline(
            ds_name, data_modalities, common_ids, y, base_out,
            reg_extractors, n_comps_list, reg_models,
            progress_count_reg, reg_total_runs, is_regression=True
        )
        
        run_selection_pipeline(
            ds_name, data_modalities, common_ids, y, base_out,
            reg_selectors, n_feats_list, reg_models,
            progress_count_reg, reg_total_runs, is_regression=True
        )

def process_classification_datasets():
    """Process all classification datasets."""
    print("\n=== CLASSIFICATION BLOCK (Breast, Colon, Kidney, etc.) ===")
    clf_extractors = get_classification_extractors()
    clf_selectors = get_classification_selectors()
    clf_models = ["LogisticRegression", "RandomForest", "SVC"]
    n_comps_list = [8, 16, MAX_COMPONENTS]
    n_feats_list = [8, 16, MAX_FEATURES]
    
    clf_total_runs = (
        len(CLASSIFICATION_DATASETS) * 
        (len(clf_extractors) * len(n_comps_list) + 
         len(clf_selectors) * len(n_feats_list))
    )
    progress_count_clf = [0]
    
    for ds_conf in CLASSIFICATION_DATASETS:
        result = process_dataset(ds_conf, is_regression=False)
        if result is None:
            continue
            
        ds_name, data_modalities, common_ids, y, base_out = result
        
        # Run extraction and selection pipelines
        run_extraction_pipeline(
            ds_name, data_modalities, common_ids, y, base_out,
            clf_extractors, n_comps_list, clf_models,
            progress_count_clf, clf_total_runs, is_regression=False
        )
        
        run_selection_pipeline(
            ds_name, data_modalities, common_ids, y, base_out,
            clf_selectors, n_feats_list, clf_models,
            progress_count_clf, clf_total_runs, is_regression=False
        )

def transform_selector_regression(X, selected_features):
    """Transform data using selected features for regression."""
    if isinstance(X, pd.DataFrame):
        return X.iloc[:, selected_features].values
    return X[:, selected_features]

def transform_selector_classification(X, selected_features):
    """Transform data using selected features for classification."""
    if isinstance(X, pd.DataFrame):
        return X.iloc[:, selected_features].values
    return X[:, selected_features]

def train_regression_model(X_train, y_train, X_val, y_val, model_name, out_dir=None, plot_prefix="", fold_idx=None):
    """Train and evaluate a regression model."""
    try:
        if model_name == "RandomForest":
            model = RandomForestRegressor(**MODEL_OPTIMIZATIONS["RandomForest"])
        elif model_name == "LinearRegression":
            model = LinearRegression(**MODEL_OPTIMIZATIONS["LinearRegression"])
        elif model_name == "SVR":
            model = SVR(**MODEL_OPTIMIZATIONS["SVR"])
        else:
            raise ValueError(f"Unknown model: {model_name}")

        with heavy_cpu_section():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

        metrics = {
            'mse': mean_squared_error(y_val, y_pred),
            'mae': mean_absolute_error(y_val, y_pred),
            'r2': r2_score(y_val, y_pred)
        }

        if out_dir:
            plot_regression_scatter(y_val, y_pred, plot_prefix, 
                                  os.path.join(out_dir, f"{plot_prefix}_scatter.png"))
            plot_regression_residuals(y_val, y_pred, plot_prefix,
                                    os.path.join(out_dir, f"{plot_prefix}_residuals.png"))

        return model, metrics
    except Exception as e:
        print(f"Warning: Failed to train {model_name} in fold {fold_idx}: {str(e)}")
        return None, {}

def train_classification_model(X_train, y_train, X_val, y_val, model_name, out_dir=None, plot_prefix="", fold_idx=None):
    """Train and evaluate a classification model."""
    try:
        if model_name == "LogisticRegression":
            model = LogisticRegression(penalty='l2', solver='liblinear', random_state=42)
        elif model_name == "RandomForest":
            model = RandomForestClassifier(**MODEL_OPTIMIZATIONS["RandomForest"])
        elif model_name == "SVC":
            model = SVC(kernel='rbf', probability=True, random_state=42)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        with heavy_cpu_section():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None

        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, average='weighted'),
            'recall': recall_score(y_val, y_pred, average='weighted'),
            'f1': f1_score(y_val, y_pred, average='weighted')
        }

        if y_proba is not None:
            metrics['auc'] = roc_auc_score(y_val, y_proba)

        if out_dir:
            cm = confusion_matrix(y_val, y_pred)
            class_labels = sorted(set(y_val))
            plot_confusion_matrix(cm, class_labels, plot_prefix,
                                os.path.join(out_dir, f"{plot_prefix}_confusion.png"))
            if y_proba is not None:
                plot_roc_curve_binary(model, X_val, y_val, class_labels, plot_prefix,
                                    os.path.join(out_dir, f"{plot_prefix}_roc.png"))

        return model, metrics
    except Exception as e:
        print(f"Warning: Failed to train {model_name} in fold {fold_idx}: {str(e)}")
        return None, {}

def cached_fit_transform_extractor_classification(X, y, extractor, n_components, ds_name, modality_name):
    """Cached version of fit_transform for classification extractors."""
    key = f"{ds_name}_{modality_name}_{extractor.__class__.__name__}_{n_components}"
    
    if key in _selector_cache:
        return _selector_cache[key]
    
    try:
        extractor.set_params(n_components=n_components)
        X_transformed = extractor.fit_transform(X)
        _selector_cache[key] = (extractor, X_transformed)
        return extractor, X_transformed
    except Exception as e:
        print(f"Error in feature extraction: {str(e)}")
        return None, None

def transform_extractor_classification(X, extractor):
    """Transform data using fitted extractor for classification."""
    try:
        return extractor.transform(X)
    except Exception as e:
        print(f"Error in transform: {str(e)}")
        return None

def cached_fit_transform_selector_classification(X, y, selector_code, n_feats, ds_name, modality_name):
    """Cached version of fit_transform for classification selectors."""
    key = f"{ds_name}_{modality_name}_{selector_code}_{n_feats}"
    
    if key in _selector_cache:
        return _selector_cache[key]
    
    try:
        if selector_code == "mrmr_clf":
            selector = SelectKBest(mutual_info_classif, k=n_feats)
        elif selector_code == "fclassif":
            selector = SelectKBest(f_classif, k=n_feats)
        elif selector_code == "logistic_l1":
            selector = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
        elif selector_code == "chi2_selection":
            selector = SelectKBest(chi2, k=n_feats)
        else:
            raise ValueError(f"Unknown selector code: {selector_code}")
            
        X_selected = selector.fit_transform(X, y)
        selected_features = np.arange(X.shape[1])[selector.get_support()]
        
        _selector_cache[key] = (selected_features, X_selected)
        return selected_features, X_selected
    except Exception as e:
        print(f"Error in feature selection: {str(e)}")
        return None, None

def get_selector_object(selector_code, n_feats):
    """Create appropriate selector object based on code."""
    if selector_code == "mrmr_reg":
        return SelectKBest(mutual_info_regression, k=n_feats)
    elif selector_code == "lasso":
        return SelectFromModel(Lasso(alpha=0.01, random_state=42), max_features=n_feats)
    elif selector_code == "enet":
        return SelectFromModel(ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42), max_features=n_feats)
    elif selector_code == "freg":
        return SelectKBest(f_regression, k=n_feats)
    elif selector_code == "mrmr_clf":
        return SelectKBest(mutual_info_classif, k=n_feats)
    elif selector_code == "fclassif":
        return SelectKBest(f_classif, k=n_feats)
    elif selector_code == "logistic_l1":
        return SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', random_state=42), max_features=n_feats)
    elif selector_code == "chi2_selection":
        return SelectKBest(chi2, k=n_feats)
    else:
        raise ValueError(f"Unknown selector code: {selector_code}")

def main():
    # Create and ensure temp folder exists
    temp_folder = os.path.join(os.getcwd(), "temp_joblib")
    os.makedirs(temp_folder, exist_ok=True)
    
    # Set matplotlib to use Agg backend
    plt.switch_backend('Agg')
    
    try:
        # Process regression datasets
        process_regression_datasets()
        
        # Process classification datasets
        process_classification_datasets()
        
        print("\nAll done! Regression outputs in 'output_regression/' and classification outputs in 'output_classification/'.")
    
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        raise
    finally:
        # Clean up joblib temp folder
        try:
            import shutil
            shutil.rmtree(temp_folder)
            print(f"\nCleaned up temporary folder: {temp_folder}")
        except Exception as e:
            print(f"\nWarning: Failed to clean up temporary folder {temp_folder}: {str(e)}")
        
        # Clear matplotlib figures
        plt.close('all')

if __name__ == "__main__":
    main()