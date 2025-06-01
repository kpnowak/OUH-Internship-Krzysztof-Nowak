#!/usr/bin/env python3

# Standard library imports
import os
import time
import threading
import warnings
from contextlib import contextmanager
from typing import Dict, List, Tuple, Optional, Union, Any

# Suppress common warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

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
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
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
from alg3_multi_additions.utils_boruta import boruta_selector

# Constants and configuration
from alg3_multi_additions.config_old import (
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

_extractor_cache = {
    'ext_reg': {},
    'ext_clf': {}
}

# Add flag for LDA warning
_SHOWN_LDA_MSG = False

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
    so we compute row-variance.  Sparse frames are handled efficiently
    with toarray() fallback if needed.

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

def try_read_file(path: str, clinical_cols: Optional[List[str]] = None, id_col: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Try to read a file, return None if it fails.

    Parameters
    ----------
    path            Path to the file
    clinical_cols   List of clinical column names to keep
    id_col         ID column name that must be preserved (if provided)

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
        
        # Ensure ID column is included if specified
        if id_col is not None and id_col not in clinical_cols:
            clinical_cols = [id_col] + clinical_cols
            print(f"Added ID column '{id_col}' to clinical columns")
        
        # Filter clinical_cols to only include those that exist in the file
        valid_cols = [col for col in clinical_cols if col in actual_cols]
        
        if not valid_cols:
            print(f"Warning: None of the specified clinical columns found in {path}")
            return None
            
        # Read the file with only the valid columns, without using index_col
        usecols = list(dict.fromkeys(valid_cols))
        df = pd.read_csv(path, usecols=usecols)
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
    """
    Load omics data and clinical data.
    
    Parameters
    ----------
    ds_config : dict
        Dataset configuration
    is_regression : bool, default=True
        Whether this is a regression task (True) or classification task (False)
    
    Returns
    -------
    data_modalities : dict
        Dictionary of modality DataFrames
    clinical_df : pd.DataFrame
        Clinical data DataFrame
    all_ids : list
        List of all patient IDs
    y : array-like
        Target values
    """
    ds_name = ds_config["name"]
    id_col = ds_config["id_col"]
    out_col = ds_config["outcome_col"]
    
    # Extract paths from config
    base_path = ds_config.get("base_path", "data")
    exp_path = os.path.join(base_path, ds_config.get("exp_path", ""))
    methy_path = os.path.join(base_path, ds_config.get("methy_path", ""))
    mirna_path = os.path.join(base_path, ds_config.get("mirna_path", ""))
    clinical_path = os.path.join(base_path, ds_config.get("clinical_path", ""))
    
    # Load clinical data with specified columns
    clinical_cols = ds_config.get("clinical_cols", None)
    clinical_df = try_read_file(clinical_path, clinical_cols, id_col)
    
    if clinical_df is None:
        raise ValueError(f"Failed to load clinical data from {clinical_path}")
    
    # Ensure ID column exists
    if id_col not in clinical_df.columns:
        # Maybe it ended up as the index?
        if clinical_df.index.name == id_col or clinical_df.index.name is None:
            clinical_df = clinical_df.reset_index().rename(columns={'index': id_col})
    
    if id_col not in clinical_df.columns:
        raise ValueError(
            f"ID column '{id_col}' is missing from clinical data. "
            f"Available columns: {clinical_df.columns.tolist()}. "
            f"Check clinical_cols in the dataset config."
        )
    
    # Rename columns if needed
    id_col_map = ds_config.get("id_col_map", {})
    for old_col, new_col in id_col_map.items():
        if old_col in clinical_df.columns:
            clinical_df.rename(columns={old_col: new_col}, inplace=True)
    
    # Handle outcome column
    if out_col not in clinical_df.columns:
        out_col_map = ds_config.get("out_col_map", {})
        for old_col, new_col in out_col_map.items():
            if old_col in clinical_df.columns:
                clinical_df[new_col] = clinical_df[old_col].apply(
                    lambda x: custom_parse_outcome(x)
                )
                break
    
    # Remove rows with missing target values
    clinical_df = clinical_df.dropna(subset=[out_col])
    
    # Filter clinical data if needed
    clinical_filter = ds_config.get("clinical_filter", None)
    if clinical_filter:
        for col, val in clinical_filter.items():
            if col in clinical_df.columns:
                clinical_df = clinical_df[clinical_df[col] == val]
    
    # Load omics data
    exp_df = try_read_file(exp_path)
    methy_df = try_read_file(methy_path)
    mirna_df = try_read_file(mirna_path)
    
    # Slice TCGA IDs if needed
    if ds_config.get("tcga_id_slicing", False):
        if exp_df is not None and exp_df.columns.str.match(r'TCGA-\w{2}-\w{4}').any():
            exp_df.columns = exp_df.columns.map(fix_tcga_id_slicing)
        if methy_df is not None and methy_df.columns.str.match(r'TCGA-\w{2}-\w{4}').any():
            methy_df.columns = methy_df.columns.map(fix_tcga_id_slicing)
        if mirna_df is not None and mirna_df.columns.str.match(r'TCGA-\w{2}-\w{4}').any():
            mirna_df.columns = mirna_df.columns.map(fix_tcga_id_slicing)
        clinical_df[id_col] = clinical_df[id_col].map(fix_tcga_id_slicing)
    
    # Create a dictionary of omics data modalities
    data_modalities = {}
    if exp_df is not None:
        data_modalities["Gene Expression"] = exp_df
    if methy_df is not None:
        data_modalities["Methylation"] = methy_df
    if mirna_df is not None:
        data_modalities["miRNA"] = mirna_df
    
    # For each modality, keep only the top variable features
    for name, df in data_modalities.items():
        # Apply variability filtering
        data_modalities[name] = _keep_top_variable_rows(df)
    
    # Get the list of all IDs
    all_ids = list(clinical_df[id_col])
    
    # Get target values
    y = clinical_df[out_col].values
    
    # For classification, ensure y is categorical
    if not is_regression:
        # Check if y is already categorical
        if not isinstance(y.dtype, pd.CategoricalDtype) and not pd.api.types.is_integer_dtype(y):
            # Convert to categorical
            y = pd.Categorical(y).codes
    
    return data_modalities, clinical_df, all_ids, y

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
        The merged matrix (float32).

    Raises
    ------
    ValueError
        If arrays have different row counts or if no usable arrays are provided.
    """
    # Skip None or empty arrays
    filtered_arrays = [arr for arr in arrays if arr is not None and arr.size > 0]
    
    # Check if we have any arrays to merge
    if not filtered_arrays:
        raise ValueError("No valid arrays provided for merging")
    
    # Convert all arrays to float32 numpy arrays and ensure they're 2D
    processed_arrays = []
    for arr in filtered_arrays:
        # Convert to numpy array if not already
        arr_np = np.asarray(arr, dtype=np.float32)
        # Ensure 2D - if 1D, reshape to column vector
        if arr_np.ndim == 1:
            arr_np = arr_np.reshape(-1, 1)
        # For higher dimensions, flatten all but the first dimension
        elif arr_np.ndim > 2:
            original_shape = arr_np.shape
            arr_np = arr_np.reshape(original_shape[0], -1)
        processed_arrays.append(arr_np)
    
    # Find row counts and check for mismatches
    row_counts = [arr.shape[0] for arr in processed_arrays]
    if len(set(row_counts)) > 1:
        print(f"WARNING: Arrays have different row counts: {row_counts}. Truncating to shortest length for alignment.")
        min_rows = min(row_counts)
        # Truncate all arrays to have the same number of rows
        processed_arrays = [arr[:min_rows] for arr in processed_arrays]
    
    # Merge based on strategy
    if strategy == "concat":
        # Concatenate along features dimension
        merged = np.column_stack(processed_arrays)
    elif strategy in ["average", "sum", "max"]:
        # For these strategies, we need arrays with the same shape
        # First get all arrays to the same number of columns by padding
        col_sizes = [arr.shape[1] for arr in processed_arrays]
        max_cols = max(col_sizes)
        
        # Pad each array to have the same number of columns
        padded_arrays = []
        for arr in processed_arrays:
            if arr.shape[1] < max_cols:
                pad_width = max_cols - arr.shape[1]
                padded = np.pad(arr, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
                padded_arrays.append(padded)
        else:
                padded_arrays.append(arr)

        # Stack arrays along a new axis (making a 3D array)
        stacked = np.stack(padded_arrays, axis=0)
        
        # Apply the appropriate operation along the first axis (modalities)
        if strategy == "average":
            merged = np.mean(stacked, axis=0)
        elif strategy == "sum":
            merged = np.sum(stacked, axis=0)
        elif strategy == "max":
            merged = np.max(stacked, axis=0)
        else:
            # Default to concatenation for unknown strategy
            print(f"Unknown merge strategy: {strategy}, using concat instead")
            merged = np.column_stack(processed_arrays)

    # Apply imputation if an imputer is provided
    if imputer is not None:
        if is_train:
            merged = imputer.fit_transform(merged)
        else:
            merged = imputer.transform(merged)
    
    # Final check and NaN handling
    merged = np.nan_to_num(merged, nan=0.0)
    
    return merged

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
        "max_iter": 1000,  # Reduced from 20000 for faster execution
        "whiten": "unit-variance",
        "whiten_solver": "svd",
        "tol": 1e-2,       # Relaxed tolerance from 1e-5
        "algorithm": "parallel",
        "fun": "logcosh"   # Changed from default to logcosh for better convergence
    } if sklearn_version >= "1.3.0" else {
        "max_iter": 1000,  # Reduced
        "whiten": "unit-variance",
        "tol": 1e-2,       # Relaxed tolerance
        "algorithm": "parallel",
        "fun": "logcosh"
    }
    
    return {
        "PCA": PCA(random_state=42),
        "ICA": FastICA(**ica_params, random_state=42),
        "LDA": LDA(),
        "FA": FactorAnalysis(
            n_components=10,
            max_iter=1000,  # Reduced from 20000
            tol=1e-3,       # Relaxed tolerance
            random_state=42
        ),
        "KPCA": KernelPCA(kernel='rbf', random_state=42)
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
        # Create a new figure for each plot
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
        
        # Create scatter plot
        scatter = ax.scatter(y_test, y_pred, alpha=0.5)
        
        # Add diagonal line
        mn = min(min(y_test), min(y_pred))
        mx = max(max(y_test), max(y_pred))
        ax.plot([mn, mx], [mn, mx], 'r--', alpha=0.7)
        
        # Set labels and title
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(title + ": Actual vs. Predicted")
        
        # Save and close
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)  # Explicitly close the figure
        
        return verify_plot_exists(out_path)
    except Exception as e:
        print(f"Error creating scatter plot for {title}: {str(e)}")
        plt.close('all')  # Ensure all figures are closed on error
        return False

def plot_regression_residuals(y_test, y_pred, title, out_path):
    """Plot regression residuals and return success status."""
    # Check if all values are NaN
    if np.isnan(y_test).all() or np.isnan(y_pred).all():
        print(f"Warning: All values are NaN for {title}, skipping plot")
        return False
        
    try:
        # Create a new figure for each plot
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
        
        # Calculate and plot residuals
        residuals = y_test - y_pred
        scatter = ax.scatter(y_pred, residuals, alpha=0.5)
        
        # Add horizontal line at y=0
        ax.axhline(0, color='r', linestyle='--')
        
        # Set labels and title
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residual")
        ax.set_title(title + ": Residual Plot")
        
        # Save and close
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)  # Explicitly close the figure
        
        return verify_plot_exists(out_path)
    except Exception as e:
        print(f"Error creating residuals plot for {title}: {str(e)}")
        plt.close('all')  # Ensure all figures are closed on error
        return False

def plot_confusion_matrix(cm, class_labels, title, out_path):
    """Plot confusion matrix and return success status."""
    try:
        # Create a new figure for each plot
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d',
                    cmap='Blues', xticklabels=class_labels, yticklabels=class_labels,
                    ax=ax)
        
        # Set labels and title
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        
        # Save and close
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)  # Explicitly close the figure
        
        return verify_plot_exists(out_path)
    except Exception as e:
        print(f"Error creating confusion matrix for {title}: {str(e)}")
        plt.close('all')  # Ensure all figures are closed on error
        return False

def plot_roc_curve_binary(model, X_test, y_test, class_labels, title, out_path):
    """Plot ROC curve for binary classification and return success status."""
    try:
        # Convert y_test to numeric if it's string
        if isinstance(y_test[0], str):
            y_test = np.array([int(x) for x in y_test])
            
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Create a new figure for each plot
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
        
        # Create ROC curve
        disp = RocCurveDisplay.from_predictions(
            y_test, 
            y_proba,
            name='Binary ROC',
            pos_label=1,  # Explicitly set positive label
            ax=ax
        )
        
        # Set title
        ax.set_title(title + " - ROC Curve")
        
        # Save and close
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)  # Explicitly close the figure
        
        return verify_plot_exists(out_path)
    except Exception as e:
        print(f"Error creating ROC curve for {title}: {str(e)}")
        plt.close('all')  # Ensure all figures are closed on error
        return False

###############################################################################
# F) TRAIN & EVAL: REGRESSION
###############################################################################

def cached_fit_transform_selector_regression(selector, X, y, n_feats, fold_idx=None, ds_name=None):
    """Cached version of fit_transform for regression selectors."""
    # Use modality-independent key for more efficient caching
    key = f"{ds_name}_{fold_idx}_{selector.__class__.__name__}_{n_feats}"
    
    if key in _selector_cache['sel_reg']:
        return _selector_cache['sel_reg'][key]
    
    try:
        # Convert DataFrame to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Special handling for Boruta
        if isinstance(selector, BorutaPy):
            # Use the new stable boruta_selector
            sel_idx = boruta_selector(
                X, y,
                k_features=n_feats,
                task="reg",
                random_state=42,
                max_iter=150,
                perc=85
            )
            X_selected = X[:, sel_idx]
            _selector_cache['sel_reg'][key] = (sel_idx, X_selected)
            return sel_idx, X_selected
        else:
            # Standard scikit-learn selector handling
            X_selected = selector.fit_transform(X, y)
            selected_features = np.arange(X.shape[1])[selector.get_support()]
        
        _selector_cache['sel_reg'][key] = (selected_features, X_selected)
        return selected_features, X_selected
    except Exception as e:
        print(f"Error in feature selection: {str(e)}")
        return None, None

def process_modality(modality_name, modality_df, id_train, id_val, idx_test, y_train, extr_obj, ncomps, idx_to_id, fold_idx=None):
    """Process a single modality with proper error handling."""
    try:
        # Create Series for training y with proper indices
        y_train_series = pd.Series(y_train, index=id_train)
        
        # First get data for each split with all available IDs
        df_train = modality_df.loc[:, [id_ for id_ in id_train if id_ in modality_df.columns]].transpose()
        df_val = modality_df.loc[:, [id_ for id_ in id_val if id_ in modality_df.columns]].transpose()
        df_test = modality_df.loc[:, [idx_to_id[idx] for idx in idx_test if idx_to_id[idx] in modality_df.columns]].transpose()
        
        # Skip if too few samples
        if df_train.shape[0] < 5 or df_val.shape[0] < 2:
            print(f"Not enough valid samples for {modality_name}")
            return None, None, None
        
        # Now get aligned y values AFTER sampling data - critical change
        # This ensures y_train aligns exactly with df_train index
        aligned_y_train = y_train_series.reindex(df_train.index).values
        
        # Check for any NaN values in aligned_y_train from reindexing
        if np.isnan(aligned_y_train).any():
            print(f"Warning: NaN values in aligned labels for {modality_name}")
            return None, None, None
        
        # Convert to numeric and handle NaN values - more aggressively replace NaNs with zeros
        df_train = df_train.apply(pd.to_numeric, errors='coerce').fillna(0)
        df_val = df_val.apply(pd.to_numeric, errors='coerce').fillna(0)
        df_test = df_test.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Ensure y values are also numeric if needed
        try:
            if not np.issubdtype(aligned_y_train.dtype, np.number):
                aligned_y_train = pd.factorize(aligned_y_train)[0]
        except:
            # If there's any issue, try direct conversion
            aligned_y_train = np.asarray(aligned_y_train, dtype=np.float32)
        
        # Confirm we have valid data after all processing
        if df_train.shape[0] != len(aligned_y_train):
            print(f"Data alignment issue in {modality_name}: {df_train.shape[0]} rows vs {len(aligned_y_train)} labels")
            return None, None, None
            
        # Get extractor and transform data with aligned y values
        extractor, X_tr = cached_fit_transform_extractor_classification(
            df_train, aligned_y_train, extr_obj, ncomps, None, modality_name, fold_idx
        )
        
        # Verify extraction worked
        if extractor is None or X_tr is None:
            print(f"Warning: Feature extraction failed for {modality_name}")
            return None, None, None
            
        # Transform validation and test data
        X_va = transform_extractor_classification(df_val, extractor)
        X_te = transform_extractor_classification(df_test, extractor)
        
        # Final validation and NaN check
        if X_va is None or X_te is None:
            print(f"Warning: Feature transformation failed for {modality_name}")
            return None, None, None
            
        # Replace any remaining NaNs with zeros
        X_tr = np.nan_to_num(X_tr, nan=0.0)
        X_va = np.nan_to_num(X_va, nan=0.0)
        X_te = np.nan_to_num(X_te, nan=0.0)
        
        # Return the processed data and the aligned_y_train for consistent use
        return X_tr, X_va, X_te
    except Exception as e:
        print(f"Error processing {modality_name}: {str(e)}")
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
    # Create explicit Series mapping between IDs and labels
    id_train = [idx_to_id[i] for i in train_idx]
    id_val = [idx_to_id[i] for i in val_idx]
    y_train = y_temp[train_idx]
    y_val = y_temp[val_idx]
    
    # Create Series with index=ID, value=label for easy alignment
    y_train_series = pd.Series(y_train, index=id_train)
    y_val_series = pd.Series(y_val, index=id_val)
    
    # Process with missing modalities
    modified_modalities = process_with_missing_modalities(
        data_modalities, all_ids, missing_percentage, fold_idx
    )
    
    # Find common IDs across all modalities for proper alignment
    valid_train_ids = set(id_train)
    valid_val_ids = set(id_val)
    
    for name, df in modified_modalities.items():
        valid_train_ids = valid_train_ids.intersection(df.columns)
        valid_val_ids = valid_val_ids.intersection(df.columns)
    
    # Return empty if no common samples
    if not valid_train_ids or not valid_val_ids:
        print(f"Warning: No common samples found across modalities in fold {fold_idx}")
        return {}
    
    # Sort IDs for consistency
    valid_train_ids = sorted(list(valid_train_ids))
    valid_val_ids = sorted(list(valid_val_ids))
    
    # Get aligned labels 
    aligned_y_train = y_train_series.loc[valid_train_ids].values
    aligned_y_val = y_val_series.loc[valid_val_ids].values

    # Process modalities in parallel with threading backend
    modality_results = TParallel(n_jobs=N_JOBS)(
        delayed(process_modality)(
            name, 
            df, 
            valid_train_ids,  # Use aligned IDs
            valid_val_ids,    # Use aligned IDs
            idx_test, 
            aligned_y_train,  # Use aligned labels
            extr_obj, 
            ncomps, 
            idx_to_id,
            fold_idx
        )
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
            # Use the fixed get_model_object function
            model = get_model_object(model_name, random_state=fold_idx)

            # Train the model with heavy CPU section
            with heavy_cpu_section():
                model.fit(X_train_merged, aligned_y_train)  # Use aligned training data

            # Make predictions
            y_val_pred = model.predict(X_val_merged)

            # Calculate metrics using aligned validation data
            metrics = {
                'mse': mean_squared_error(aligned_y_val, y_val_pred),
                'mae': mean_absolute_error(aligned_y_val, y_val_pred),
                'r2': r2_score(aligned_y_val, y_val_pred)
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

    # Split indices and y values with stratification
    idx_temp, idx_test, y_temp, y_test = train_test_split(
        indices, y_arr, test_size=test_size, random_state=0, stratify=y_arr
    )

    # Process CV folds for each missing percentage
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    all_results = []
    
    for missing_percentage in MISSING_MODALITIES_CONFIG["missing_percentages"]:
        cv_results = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(idx_temp, y_temp)):
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
            for i, (_, val_idx) in enumerate(cv.split(idx_temp, y_temp)):
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
        metrics_file = os.path.join(base_out, "metrics", f"{ds_name}_extraction_cv_metrics.csv")
        # Check if file exists to determine if we need to write header
        file_exists = os.path.exists(metrics_file)
        # Append results to CSV
        pd.DataFrame(all_results).to_csv(
            metrics_file,
            mode='a',
            header=not file_exists,
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
            final_metrics_file = os.path.join(base_out, "metrics", f"{ds_name}_extraction_final_metrics.csv")
            file_exists = os.path.exists(final_metrics_file)
            warning_metrics.to_csv(
                final_metrics_file,
                mode='a',
                header=not file_exists,
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
                delayed(process_modality)(name, df, id_train, id_test, idx_test, y_train, extr_obj, ncomps, idx_to_id, fold_idx)
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
            
            final_metrics_file = os.path.join(base_out, "metrics", f"{ds_name}_extraction_final_metrics.csv")
            file_exists = os.path.exists(final_metrics_file)
            pd.DataFrame([final_metrics]).to_csv(
                final_metrics_file,
                mode='a',
                header=not file_exists,
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
    all_results = []

    # Process CV folds for each missing percentage
    for missing_percentage in MISSING_MODALITIES_CONFIG["missing_percentages"]:
        cv_results = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(id_temp)):
            try:
                id_train = all_ids_arr[id_temp[train_idx]]
                id_val = all_ids_arr[id_temp[val_idx]]
                y_train = y_temp[train_idx]
                y_val = y_temp[val_idx]

                # Use process_with_missing_modalities for consistent missing modality simulation
                modified_modalities = process_with_missing_modalities(
                    data_modalities, all_ids, MISSING_MODALITIES_CONFIG["missing_percentages"][0], fold_idx
                )

                train_list, val_list = [], []
                valid_train_ids = set(id_train)
                valid_val_ids = set(id_val)
                
                # First pass: collect all valid IDs across modalities
                for modality_name, df_mod in modified_modalities.items():
                    valid_train_ids = valid_train_ids.intersection(set(df_mod.columns))
                    valid_val_ids = valid_val_ids.intersection(set(df_mod.columns))
                
                # Convert back to lists and sort for consistency
                valid_train_ids = sorted(list(valid_train_ids))
                valid_val_ids = sorted(list(valid_val_ids))
                
                if not valid_train_ids or not valid_val_ids:
                    print(f"Warning: No common IDs found across modalities in fold {fold_idx}")
                    continue

                # Get indices for valid IDs in the original arrays
                valid_train_indices = [i for i, id_ in enumerate(id_train) if id_ in valid_train_ids]
                valid_val_indices = [i for i, id_ in enumerate(id_val) if id_ in valid_val_ids]

                # Update y values using the valid indices
                y_train = y_train[valid_train_indices]
                y_val = y_val[valid_val_indices]

                # Verify lengths match
                if len(y_train) != len(valid_train_ids) or len(y_val) != len(valid_val_ids):
                    print(f"Warning: Length mismatch after alignment in fold {fold_idx}")
                    continue

                # Second pass: process each modality with common IDs
                for modality_name, df_mod in modified_modalities.items():
                    try:
                        df_train = df_mod.loc[:, valid_train_ids].transpose()
                        df_val = df_mod.loc[:, valid_val_ids].transpose()
                        
                        # Convert to numeric and handle NaN values
                        df_train = df_train.apply(pd.to_numeric, errors='coerce').fillna(0)
                        df_val = df_val.apply(pd.to_numeric, errors='coerce').fillna(0)
                        
                        # Create the selector object before using it
                        selector = get_selector_object(sel_code, n_feats)
                        
                        # Get selector and transform data
                        chosen_cols, X_tr = cached_fit_transform_selector_regression(
                            selector, df_train, y_train, n_feats, fold_idx, ds_name
                        )
                        
                        if chosen_cols is None or X_tr is None:
                            print(f"Warning: Feature selection failed for {modality_name} in fold {fold_idx}")
                            continue
                            
                        X_va = transform_selector_regression(df_val, chosen_cols)
                        
                        if X_va is None:
                            print(f"Warning: Feature transformation failed for {modality_name} in fold {fold_idx}")
                            continue
                        
                        train_list.append(np.array(X_tr))
                        val_list.append(np.array(X_va))
                        
                    except Exception as e:
                        print(f"Error processing {modality_name} in fold {fold_idx}: {str(e)}")
                        continue

                if not train_list or not val_list:
                    print(f"Warning: No valid data for any modality in fold {fold_idx}")
                    continue

                for merge_str in ["concat", "average", "sum", "max"]:
                    try:
                        # Create a shared imputer instance for this fold
                        fold_imputer = ModalityImputer()
                        X_tr_m = merge_modalities(*train_list, strategy=merge_str, imputer=fold_imputer, is_train=True)
                        X_va_m = merge_modalities(*val_list, strategy=merge_str, imputer=fold_imputer, is_train=False)
                        
                        if X_tr_m is None or X_va_m is None:
                            print(f"Warning: Merging failed for {merge_str} in fold {fold_idx}")
                            continue

                        # Sanity check before training
                        if X_tr_m.shape[0] != len(y_train) or np.isnan(X_tr_m).any():
                            print(f"Skipping invalid data configuration in fold {fold_idx}")
                            continue
                            
                        # Create model instances from model names
                        model_dict = {}
                        results = []  # Initialize results list
                        for model_name in reg_models:
                            model_dict[model_name] = get_model_object(model_name, random_state=fold_idx)
                            
                        for model_name, model in model_dict.items():
                            try:
                                # Train and evaluate model
                                model.fit(X_tr_m, y_train)
                                y_pred = model.predict(X_va_m)
                                score = r2_score(y_val, y_pred)
                                
                                # Store results
                                results.append({
                                    'fold': fold_idx,
                                    'merge_strategy': merge_str,
                                    'model': model_name,
                                    'score': score
                                })
                            except Exception as e:
                                print(f"Warning: Failed to train/evaluate {model_name} in fold {fold_idx}: {str(e)}")
                                continue
                                
                    except Exception as e:
                        print(f"Skipping merge '{merge_str}' fold {fold_idx}: {e}")
                        continue
                        
            except Exception as e:
                print(f"Warning: Failed to process fold {fold_idx}: {str(e)}")
                continue

        # Skip if no valid results for this missing percentage
        if not cv_results:
            continue
            
        # Aggregate results for this missing percentage
        for merge_str in ["concat", "average", "sum", "max"]:
            for model_name in reg_models:
                # Only include results for models that have predictions
                valid_results = [r for r in cv_results if r["MergeStrategy"] == merge_str and r["Model"] == model_name]
                
                if valid_results:
                    # Average metrics across folds
                    avg_metrics = {
                        k: np.mean([m[k] for m in valid_results]) 
                        for k in valid_results[0].keys() if k not in ["MergeStrategy", "Model", "Missing_Percentage"]
                    }
                    avg_metrics.update({
                        "Dataset": ds_name, "Workflow": "Selection-CV",
                        "Selector": sel_name, "n_features": n_feats,
                        "MergeStrategy": merge_str, "Model": model_name,
                        "Missing_Percentage": missing_percentage
                    })
                    all_results.append(avg_metrics)

    # Save all results if any were generated
    if all_results:
        metrics_file = os.path.join(base_out, "metrics", f"{ds_name}_selection_cv_metrics.csv")
        # Check if file exists to determine if we need to write header
        file_exists = os.path.exists(metrics_file)
        # Append results to CSV
        pd.DataFrame(all_results).to_csv(
            metrics_file,
            mode='a',
            header=not file_exists,
            index=False
        )

    # Train and save final model with all training data
    if all_results:
        # Get best model based on average MSE across folds, filtering out NaN values
        candidates = [(model, np.mean([r['mse'] for r in all_results if r['Model'] == model])) 
                     for model in reg_models]
        valid_candidates = [(model, mse) for model, mse in candidates if not np.isnan(mse)]
        
        if not valid_candidates:
            print(f"Warning: No valid models found for {ds_name} with {sel_name}-{n_feats}")
            # Write empty metrics file with warning
            warning_metrics = pd.DataFrame([{
                "Dataset": ds_name,
                "Workflow": "Selection-Final",
                "Selector": sel_name,
                "n_features": n_feats,
                "Model": "NONE",
                "mse": np.nan,
                "mae": np.nan,
                "r2": np.nan,
                "Warning": "All models produced NaN metrics"
            }])
            final_metrics_file = os.path.join(base_out, "metrics", f"{ds_name}_selection_final_metrics.csv")
            file_exists = os.path.exists(final_metrics_file)
            warning_metrics.to_csv(
                final_metrics_file,
                mode='a',
                header=not file_exists,
                index=False
            )
            return all_results
            
        best_model_name = min(valid_candidates, key=lambda x: x[1])[0]

        # Process all training data
        id_train = id_temp
        id_test = id_test
        y_train = y_temp

        # Process modalities with threading backend
        with parallel_config(backend="threading"):
            train_list, test_list = [], []
            for modality_name, df_mod in data_modalities.items():
                try:
                    # Ensure all IDs exist in the modality DataFrame
                    valid_train_ids = [id_ for id_ in id_train if id_ in df_mod.columns]
                    valid_test_ids = [id_ for id_ in id_test if id_ in df_mod.columns]
                    
                    if not valid_train_ids or not valid_test_ids:
                        print(f"Warning: No valid IDs found for {modality_name} in final model")
                        continue
                    
                    df_train = df_mod.loc[:, valid_train_ids].transpose()
                    df_test = df_mod.loc[:, valid_test_ids].transpose()
                    
                    # Convert to numeric and handle NaN values
                    df_train = df_train.apply(pd.to_numeric, errors='coerce').fillna(0)
                    df_test = df_test.apply(pd.to_numeric, errors='coerce').fillna(0)
                    
                    # Get selector and transform data
                    selector = get_selector_object(sel_code, n_feats)
                    chosen_cols, X_tr = cached_fit_transform_selector_regression(
                        selector, df_train, y_train, n_feats, None, ds_name
                    )
                    
                    if chosen_cols is None or X_tr is None:
                        print(f"Warning: Feature selection failed for {modality_name} in final model")
                        continue
                        
                    X_te = transform_selector_regression(df_test, chosen_cols)
                    
                    if X_te is None:
                        print(f"Warning: Feature transformation failed for {modality_name} in final model")
                        continue
                    
                    train_list.append(np.array(X_tr))
                    test_list.append(np.array(X_te))
                    
                except Exception as e:
                    print(f"Error processing {modality_name} in final model: {str(e)}")
                    continue

            if train_list and test_list:
                # Create a new imputer instance for the final model
                final_imputer = ModalityImputer()
                
                # Merge modalities with the final imputer
                X_train_merged = merge_modalities(*train_list, strategy="concat", imputer=final_imputer, is_train=True)
                X_test_merged = merge_modalities(*test_list, strategy="concat", imputer=final_imputer, is_train=False)

                # Train final model
                if best_model_name == "RandomForest":
                    final_model = RandomForestRegressor(
                        n_estimators=100,
                        max_depth=None,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        n_jobs=N_JOBS,
                        random_state=42
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
                               f"{ds_name}_{sel_name}_{n_feats}_{best_model_name}_final.pkl")
                )

                # Generate and save plots
                plot_prefix = f"{ds_name}_{sel_name}_{n_feats}_{best_model_name}_final"
                
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
                    "Workflow": "Selection-Final",
                    "Selector": sel_name,
                    "n_features": n_feats,
                    "Model": best_model_name,
                    "mse": mean_squared_error(y_test, y_test_pred),
                    "mae": mean_absolute_error(y_test, y_test_pred),
                    "r2": r2_score(y_test, y_test_pred)
                }
                
                final_metrics_file = os.path.join(base_out, "metrics", f"{ds_name}_selection_final_metrics.csv")
                file_exists = os.path.exists(final_metrics_file)
                pd.DataFrame([final_metrics]).to_csv(
                    final_metrics_file,
                    mode='a',
                    header=not file_exists,
                    index=False
                )
    
    return all_results

###############################################################################
# L) HIGH‑LEVEL PROCESS FUNCTIONS (CLASSIFICATION) WITH CROSS‑VALIDATION
###############################################################################

def cached_fit_transform_extractor_classification(X, y, extractor, n_components, ds_name, modality_name, fold_idx=None):
    """Cached version of fit_transform for classification extractors."""
    key = f"{ds_name}_{fold_idx}_{modality_name}_{extractor.__class__.__name__}_{n_components}"
    
    if key in _extractor_cache['ext_clf']:
        return _extractor_cache['ext_clf'][key]
    
    try:
        # Convert DataFrame to numpy if needed
        if isinstance(X, pd.DataFrame):
            X_safe = X.fillna(0).values
        else:
            X_safe = safe_convert_to_numeric(X)
            
        # Make sure y is suitable for classification
        try:
            if not np.issubdtype(y.dtype, np.number):
                y_safe = pd.factorize(y)[0]
            else:
                y_safe = y
        except:
            # If factorize fails, try direct conversion
            y_safe = np.asarray(y, dtype=np.int32)
            
        # Ensure X and y have the same number of samples
        if X_safe.shape[0] != len(y_safe):
            print(f"Warning: X shape {X_safe.shape} doesn't match y length {len(y_safe)}")
            return None, None
            
        # Adjust n_components based on extractor type and data constraints
        if isinstance(extractor, LDA):
            # LDA has strict component limitations
            n_classes = len(np.unique(y_safe))
            max_components = min(X_safe.shape[1], n_classes - 1)
            if n_components > max_components:
                global _SHOWN_LDA_MSG
                if not _SHOWN_LDA_MSG:
                    print(f"Warning: Reducing n_components for LDA (limited by classes-1) - message shown once")
                    _SHOWN_LDA_MSG = True
                n_components = max_components
        elif isinstance(extractor, (PCA, FastICA, FactorAnalysis)):
            # For other extractors, limit by min of samples and features
            max_components = min(X_safe.shape[0], X_safe.shape[1])
            if n_components > max_components:
                print(f"Warning: Reducing n_components from {n_components} to {max_components} for {extractor.__class__.__name__}")
                n_components = max_components
                
        # Create a new instance of the extractor to avoid modifying the original
        if isinstance(extractor, PCA):
            new_extractor = PCA(n_components=n_components, random_state=42)
        elif isinstance(extractor, FastICA):
            try:
                new_extractor = FastICA(
                n_components=n_components,
                random_state=42,
                    max_iter=1000,  # Reduced from 10000 to prevent non-convergence
                    tol=1e-3,       # Relaxed tolerance
                algorithm='parallel',
                whiten='unit-variance'
            )
            except:
                # Fallback to PCA which is more stable
                print(f"Warning: FastICA configuration failed, falling back to PCA")
                new_extractor = PCA(n_components=n_components, random_state=42)
        elif isinstance(extractor, FactorAnalysis):
            new_extractor = FactorAnalysis(
                n_components=n_components,
                max_iter=1000,  # Reduced from 10000
                tol=1e-3        # Relaxed tolerance
            )
        elif isinstance(extractor, LDA):
            new_extractor = LDA(n_components=n_components)
        elif isinstance(extractor, KernelPCA):
            new_extractor = KernelPCA(
                n_components=n_components,
                kernel='rbf',
                random_state=42
            )
        else:
            # Default to PCA as a safe fallback
            print(f"Unknown extractor type: {type(extractor)}, falling back to PCA")
            new_extractor = PCA(n_components=n_components, random_state=42)
            
        # Fit and transform
        try:
            X_transformed = new_extractor.fit_transform(X_safe, y_safe)
        except Exception as e:
            print(f"Error in feature extraction: {str(e)}, falling back to PCA")
            # If extraction fails, fall back to PCA which is more robust
            new_extractor = PCA(n_components=min(n_components, X_safe.shape[1], X_safe.shape[0]), random_state=42)
            X_transformed = new_extractor.fit_transform(X_safe)
        
        # Handle any NaN values in the transformed data
        X_transformed = np.nan_to_num(X_transformed, nan=0.0)
        
        # Verify the transformed data
        if X_transformed.shape[0] != X_safe.shape[0]:
            print(f"Warning: Transformed data has different number of samples: {X_transformed.shape[0]} vs {X_safe.shape[0]}")
            return None, None
            
        _extractor_cache['ext_clf'][key] = (new_extractor, X_transformed)
        return new_extractor, X_transformed
    except Exception as e:
        print(f"Error in feature extraction: {str(e)}")
        return None, None

def transform_extractor_classification(X, extractor):
    """Transform data using fitted extractor for classification."""
    try:
        # Convert and clean input data
        if isinstance(X, pd.DataFrame):
            X_safe = X.fillna(0).values
        else:
            X_safe = safe_convert_to_numeric(X)
            
        # Transform the data
        try:
            X_transformed = extractor.transform(X_safe)
        except Exception as e:
            print(f"Error in transform: {str(e)}")
            return None
            
        # Handle any NaN values in the transformed data
        X_transformed = np.nan_to_num(X_transformed, nan=0.0)
        
        # Verify the transformed data
        if X_transformed.shape[0] != X_safe.shape[0]:
            print(f"Warning: Transformed data has different number of samples: {X_transformed.shape[0]} vs {X_safe.shape[0]}")
            return None
            
        return X_transformed
    except Exception as e:
        print(f"Error in transform: {str(e)}")
        return None

def process_clf_extraction_combo_cv(
    ds_name, extr_name, extr_obj, ncomps, clf_models,
    data_modalities, common_ids, y, base_out,
    progress_count, clf_total_runs, test_size=0.2, n_splits=3
):
    progress_count[0] += 1
    run_idx = progress_count[0]
    print(f"[EXTRACT-CLF CV] {run_idx}/{clf_total_runs} => {ds_name} | {extr_name}-{ncomps}")

    # Convert to numpy arrays and ensure consistent indexing
    y_arr = np.array(y)
    
    # Create a mapping from indices to IDs
    idx_to_id = {idx: id_ for idx, id_ in enumerate(common_ids)}
    
    # Create indices array for row-based indexing
    indices = np.arange(len(common_ids))
    
    # Split using stratification for classification
    idx_temp, idx_test, y_temp, y_test = train_test_split(
        indices, y_arr, test_size=test_size, random_state=0, stratify=y_arr
    )

    # Use StratifiedKFold for balanced class distribution
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    cv_metrics = {}

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(idx_temp, y_temp)):
        # Get training and validation data for this fold
        X_train_indices = idx_temp[train_idx]
        X_val_indices = idx_temp[val_idx]
        y_train = y_temp[train_idx]
        y_val = y_temp[val_idx]

        # Skip fold if it doesn't have at least 2 classes in both train and val sets
        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            print(f"Skipping fold {fold_idx} - not enough classes after split")
            continue

        # Process each modality
        train_list, val_list = [], []
        for modality_name, df in data_modalities.items():
            try:
                # Extract features using indices
                X_train = df.iloc[X_train_indices]
                X_val = df.iloc[X_val_indices]
                    
                # Get extractor and transform data
                extractor, X_tr = cached_fit_transform_extractor_classification(
                    X_train, y_train, extr_obj, ncomps, ds_name, modality_name, fold_idx
                )
                
                if extractor is None or X_tr is None:
                    print(f"Warning: Feature extraction failed for {modality_name} in fold {fold_idx}")
                    continue
                    
                X_va = transform_extractor_classification(X_val, extractor)
                if X_va is None:
                    print(f"Warning: Feature transformation failed for {modality_name} in fold {fold_idx}")
                    continue
                    
                train_list.append(X_tr)
                val_list.append(X_va)
                
            except Exception as e:
                print(f"Warning: Failed to process {modality_name} in fold {fold_idx}: {str(e)}")
                continue

        if not train_list or not val_list:
            print(f"Warning: No valid data for any modality in fold {fold_idx}")
            continue

        # Process each merge strategy
        for merge_str in ["concat", "average", "sum", "max"]:
            try:
                # Create a shared imputer instance for this fold
                fold_imputer = ModalityImputer()
                
                # Merge modalities
                X_tr_m = merge_modalities(*train_list, strategy=merge_str, imputer=fold_imputer, is_train=True)
                X_va_m = merge_modalities(*val_list, strategy=merge_str, imputer=fold_imputer, is_train=False)
                
                # Sanity check
                if X_tr_m.shape[0] != len(y_train):
                    print(f"Warning: Shape mismatch after merging in fold {fold_idx}: {X_tr_m.shape[0]} vs {len(y_train)}")
                    continue

                # Process each model
                for model_name in clf_models:
                    try:
                        model = get_model_object(model_name, random_state=fold_idx)
                        model, metrics = train_classification_model(
                            X_tr_m, y_train, X_va_m, y_val, model_name,
                            out_dir=os.path.join(base_out, "plots"),
                            plot_prefix=f"{ds_name}_fold_{fold_idx}_{extr_name}_{ncomps}_{merge_str}_{model_name}",
                            fold_idx=fold_idx
                        )
                        
                        if model is not None and metrics:
                            key = (merge_str, model_name)
                            cv_metrics.setdefault(key, []).append(metrics)
                    except Exception as e:
                        print(f"Warning: Failed to train {model_name} in fold {fold_idx}: {str(e)}")
                        continue
            except ValueError as e:
                print(f"Warning: {str(e)} in fold {fold_idx}")
                continue
            except Exception as e:
                print(f"Warning: Failed to process merge strategy {merge_str} in fold {fold_idx}: {str(e)}")
                continue

    # Calculate and save average metrics
    avg_cv_results = []
    for (merge_str, model_name), mets_list in cv_metrics.items():
        if not mets_list:
            continue
            
        try:
            # Calculate mean for each metric, handling NaN values
            avg_mets = {}
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
                values = [m[metric] for m in mets_list if metric in m and not np.isnan(m[metric])]
                avg_mets[metric] = np.mean(values) if values else np.nan
                
            avg_mets.update({
                "Dataset": ds_name, "Workflow": "Extraction-CV",
                "Extractor": extr_name, "n_components": ncomps,
                "MergeStrategy": merge_str, "Model": model_name
            })
            avg_cv_results.append(avg_mets)
        except Exception as e:
            print(f"Warning: Failed to process metrics for {merge_str}-{model_name}: {str(e)}")
            continue

    # Save metrics to file
    if avg_cv_results:
        metrics_file = os.path.join(base_out, "metrics", f"{ds_name}_extraction_cv_metrics.csv")
        file_exists = os.path.exists(metrics_file)
        pd.DataFrame(avg_cv_results).to_csv(
            metrics_file,
            mode='a',
            header=not file_exists,
            index=False
        )

def process_clf_selection_combo_cv(
    ds_name, sel_name, sel_code, n_feats, clf_models,
    data_modalities, common_ids, y, base_out,
    progress_count, clf_total_runs, test_size=0.2, n_splits=3
):
    # Track and report progress
    progress_count[0] += 1
    run_idx = progress_count[0]
    print(f"[SELECT-CLF CV] {run_idx}/{clf_total_runs} => {ds_name} | {sel_name}-{n_feats}")

    # Add debugging for Boruta
    if sel_code == "boruta_clf":
        print(f"Processing Boruta selector with n_feats={n_feats}")
    
    # Convert to numpy array and create Series for easy indexing
    y_arr = np.array(y)
    y_series = pd.Series(y_arr, index=common_ids)
    
    # Create indices array for row-based indexing
    indices = np.arange(len(common_ids))
    ids_arr = np.array(common_ids)
    
    # Split with stratification for classification
    idx_temp, idx_test, y_temp, y_test = train_test_split(
        indices, y_arr, test_size=test_size, random_state=0, stratify=y_arr
    )
    
    # Get IDs for the temp set (train+val) - needed for further indexing
    ids_temp = ids_arr[idx_temp]
    
    # Use StratifiedKFold for classification to ensure class balance
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    cv_splits = list(cv.split(idx_temp, y_temp))
    
    cv_metrics = {}
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        # Get indices for this fold
        train_indices = idx_temp[train_idx]
        val_indices = idx_temp[val_idx]
        
        # Get corresponding IDs for this fold
        train_ids = ids_arr[train_indices]
        val_ids = ids_arr[val_indices]
        
        # Get target values
        train_y = y_arr[train_indices]
        val_y = y_arr[val_indices]
        
        # Skip if not enough classes in either set
        if len(np.unique(train_y)) < 2 or len(np.unique(val_y)) < 2:
            print(f"Skipping fold {fold_idx} - not enough classes after split")
            continue

        # Debug print for Boruta
        if sel_code == "boruta_clf":
            print(f"Fold {fold_idx}: train_shape={len(train_ids)}, val_shape={len(val_ids)}")
        
        # Process each modality
        train_list, val_list = [], []
        
        for modality_name, df in data_modalities.items():
            try:
                # Extract data using indices
                df_train = df.iloc[train_indices]
                df_val = df.iloc[val_indices]
                
                # Debug print for Boruta
                if sel_code == "boruta_clf":
                    print(f"  {modality_name}: df_train={df_train.shape}, aligned_y={len(train_y)}")
                
                # Apply feature selection
                chosen_cols, X_tr = cached_fit_transform_selector_classification(
                    df_train, train_y, sel_code, n_feats, ds_name, modality_name, fold_idx
                )
                
                if chosen_cols is None or X_tr is None:
                    print(f"Warning: Feature selection failed for {modality_name} in fold {fold_idx}")
                    continue
                    
                X_va = transform_selector_classification(df_val, chosen_cols)
                
                if X_va is None:
                    print(f"Warning: Feature transformation failed for {modality_name} in fold {fold_idx}")
                    continue
                
                # Add to result lists
                train_list.append(X_tr)
                val_list.append(X_va)
            except Exception as e:
                print(f"Warning: Error processing {modality_name} in fold {fold_idx}: {str(e)}")
                continue

        # Skip if no valid modalities
        if not train_list or not val_list:
            print(f"Warning: No valid data for any modality in fold {fold_idx}")
            continue

        # Debug print for Boruta
        if sel_code == "boruta_clf":
            print(f"  Valid modalities: {len(train_list)}")
            
        # Process for each merge strategy
        for merge_str in ["concat", "average", "sum", "max"]:
            try:
                # Create a fold-specific imputer
                fold_imputer = ModalityImputer()
                
                # Merge modalities
                X_tr_m = merge_modalities(*train_list, strategy=merge_str, imputer=fold_imputer, is_train=True)
                X_va_m = merge_modalities(*val_list, strategy=merge_str, imputer=fold_imputer, is_train=False)
                
                # Sanity check
                if X_tr_m.shape[0] != len(train_y):
                    print(f"Warning: Shape mismatch after merging in fold {fold_idx}: {X_tr_m.shape[0]} vs {len(train_y)}")
                    continue

                # Process each model
                for model_name in clf_models:
                    try:
                        # Create model
                        model = get_model_object(model_name, random_state=fold_idx)
                        
                        # Train and evaluate
                        model, metrics = train_classification_model(
                            X_tr_m, train_y, X_va_m, val_y,
                            model_name,
                            out_dir=os.path.join(base_out, "plots"),
                            plot_prefix=f"{ds_name}_fold_{fold_idx}_{sel_name}_{n_feats}_{merge_str}_{model_name}",
                            fold_idx=fold_idx
                        )
                        
                        # Store results
                        if model is not None and metrics:
                            key = (merge_str, model_name)
                            cv_metrics.setdefault(key, []).append(metrics)
                    except Exception as e:
                        print(f"Warning: Failed to train {model_name} in fold {fold_idx} with {merge_str}: {str(e)}")
                        continue
            except ValueError as e:
                print(f"Warning: {str(e)} in fold {fold_idx}")
                continue
            except Exception as e:
                print(f"Warning: Failed to process merge strategy {merge_str} in fold {fold_idx}: {str(e)}")
                continue

    # Calculate and save average metrics
    avg_cv_results = []
    for (merge_str, model_name), mets_list in cv_metrics.items():
        if not mets_list:
            continue
            
        try:
            # Calculate mean for each metric, handling NaN values
            avg_mets = {}
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
                values = [m[metric] for m in mets_list if metric in m and not np.isnan(m[metric])]
                avg_mets[metric] = np.mean(values) if values else np.nan
                
            avg_mets.update({
                "Dataset": ds_name, "Workflow": "Selection-CV",
                "Selector": sel_name, "n_features": n_feats,
                "MergeStrategy": merge_str, "Model": model_name
            })
            avg_cv_results.append(avg_mets)
        except Exception as e:
            print(f"Warning: Failed to process metrics for {merge_str}-{model_name}: {str(e)}")
            continue

    # Save results to file
    if avg_cv_results:
        metrics_file = os.path.join(base_out, "metrics", f"{ds_name}_selection_cv_metrics.csv")
        file_exists = os.path.exists(metrics_file)
        pd.DataFrame(avg_cv_results).to_csv(
            metrics_file,
            mode='a',
            header=not file_exists,
            index=False
        )

    # Debug summary for Boruta
    if sel_code == "boruta_clf":
        print(f"Finished {sel_name}-{n_feats}: generated {len(avg_cv_results)} metric rows")

###############################################################################
# M) MAIN
###############################################################################
def process_dataset(ds_conf, is_regression=True):
    """Process a single dataset with either regression or classification."""
    # Clear the caches to avoid sharing between datasets
    _selector_cache['sel_clf'].clear()
    _extractor_cache['ext_clf'].clear()
    
    ds_name = ds_conf["name"]
    base_out = os.path.join("output_regression" if is_regression else "output_classification", ds_name)
    
    # Create output directories
    for subdir in ["", "models", "metrics", "plots"]:
        os.makedirs(os.path.join(base_out, subdir), exist_ok=True)
    
    print(f"\n--- Processing {ds_name} ({'Regression' if is_regression else 'Classification'}) ---")
    
    # Load data
    data_modalities, clinical_df, all_ids, y = load_omics_and_clinical(ds_conf, is_regression)
    
    # Perform global ID intersection across all modalities
    common_ids = set(all_ids)
    for df in data_modalities.values():
        common_ids &= set(df.columns)
    
    # Sort for consistent ordering
    common_ids = pd.Index(sorted(common_ids))
    
    if len(common_ids) < 2:
        print(f"Error: No samples appear in all modalities for {ds_name}")
        return
    
    # Create aligned matrices with the same row order
    modalities_aligned = {}
    for name, df in data_modalities.items():
        modalities_aligned[name] = df.loc[:, common_ids].transpose()
        modalities_aligned[name] = modalities_aligned[name].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
    
    # Create aligned y with the same index
    y_series = pd.Series(y, index=all_ids)
    y_aligned = y_series.loc[common_ids]
    
    # Ensure aligned data has at least 2 classes for classification
    if not is_regression and len(np.unique(y_aligned)) < 2:
        print(f"Error: Only one class present in {ds_name} after alignment")
        return
    
    print(f"After global alignment: {len(common_ids)} samples with complete data across all modalities")
    
    # Extract relevant data shapes and progress
    progress_count = [0]
    
    # Run extraction pipeline
    if is_regression:
        extractors = get_regression_extractors()
        n_comps_list = ds_conf.get("ncomps_list", [4, 8, 16])  # Default if not specified
        models = ["LinearRegression", "RandomForest", "SVR"]
        total_runs = len(extractors) * len(n_comps_list)
        
        run_extraction_pipeline(
            ds_name, modalities_aligned, common_ids, y_aligned, base_out,
            extractors, n_comps_list, models, progress_count, total_runs,
            is_regression=True
        )
        
        # Reset progress counter before selection pipeline
        progress_count[0] = 0
        
        # Run selector pipeline with regression selectors
        selectors = get_regression_selectors()
        n_feats_list = ds_conf.get("nfeats_list", [4, 8, 16])  # Default if not specified
        run_selection_pipeline(
            ds_name, modalities_aligned, common_ids, y_aligned, base_out,
            selectors, n_feats_list, models, progress_count, total_runs,
            is_regression=True
        )
    else:
        extractors = get_classification_extractors()
        n_comps_list = ds_conf.get("ncomps_list", [4, 8, 16])  # Default if not specified
        models = ["LogisticRegression", "RandomForestClassifier", "SVC"]
        total_runs = len(extractors) * len(n_comps_list)
        
        run_extraction_pipeline(
            ds_name, modalities_aligned, common_ids, y_aligned, base_out,
            extractors, n_comps_list, models, progress_count, total_runs,
            is_regression=False
        )
        
        # Reset progress counter before selection pipeline
        progress_count[0] = 0
        
        # Run selector pipeline with classification selectors
        selectors = get_classification_selectors()
        n_feats_list = ds_conf.get("nfeats_list", [4, 8, 16])  # Default if not specified
        run_selection_pipeline(
            ds_name, modalities_aligned, common_ids, y_aligned, base_out,
            selectors, n_feats_list, models, progress_count, total_runs,
            is_regression=False
        )

def run_extraction_pipeline(ds_name, data_modalities, common_ids, y, base_out,
                          extractors, n_comps_list, models, progress_count, total_runs,
                          is_regression=True):
    """Run extraction pipeline for a dataset."""
    # Convert to numpy array
    y_arr = np.array(y)
    
    # Create indices array for row-based indexing
    indices = np.arange(len(common_ids))
    
    # Split using stratification if possible
    if is_regression:
        # For regression, bin continuous values for stratified split
        try:
            y_bins = pd.qcut(y_arr, min(10, len(y_arr)//3), labels=False, duplicates='drop')
            idx_temp, idx_test, y_temp, y_test = train_test_split(
                indices, y_arr, test_size=0.2, random_state=0, stratify=y_bins
            )
        except ValueError:
            # Fall back to regular split if stratification fails
            idx_temp, idx_test, y_temp, y_test = train_test_split(
                indices, y_arr, test_size=0.2, random_state=0
            )
    else:
        # For classification, use direct stratification
        idx_temp, idx_test, y_temp, y_test = train_test_split(
            indices, y_arr, test_size=0.2, random_state=0, stratify=y_arr
        )
    
    # Process each extractor+ncomps combination
    for extr_name, extr_obj in extractors.items():
        for n_comps in n_comps_list:
            if is_regression:
                process_reg_extraction_combo_cv(
                    ds_name, extr_name, extr_obj, n_comps, models,
                    data_modalities, common_ids, y, base_out,
                    progress_count, total_runs
                )
            else:
                process_clf_extraction_combo_cv(
                    ds_name, extr_name, extr_obj, n_comps, models,
                    data_modalities, common_ids, y, base_out,
                    progress_count, total_runs
                )

def run_selection_pipeline(ds_name, data_modalities, common_ids, y, base_out,
                         selectors, n_feats_list, models, progress_count, total_runs,
                         is_regression=True):
    """Run selection pipeline for a dataset."""
    # Convert to numpy array
    y_arr = np.array(y)
    
    # Create indices array for row-based indexing
    indices = np.arange(len(common_ids))
    
    # Split using stratification if possible
    if is_regression:
        # For regression, bin continuous values for stratified split
        try:
            y_bins = pd.qcut(y_arr, min(10, len(y_arr)//3), labels=False, duplicates='drop')
            idx_temp, idx_test, y_temp, y_test = train_test_split(
                indices, y_arr, test_size=0.2, random_state=0, stratify=y_bins
            )
        except ValueError:
            # Fall back to regular split if stratification fails
            idx_temp, idx_test, y_temp, y_test = train_test_split(
                indices, y_arr, test_size=0.2, random_state=0
            )
    else:
        # For classification, use direct stratification
        idx_temp, idx_test, y_temp, y_test = train_test_split(
            indices, y_arr, test_size=0.2, random_state=0, stratify=y_arr
        )
    
    # Process each selector+nfeats combination
    for sel_name, sel_code in selectors.items():
        for n_feats in n_feats_list:
            if is_regression:
                process_reg_selection_combo_cv(
                    ds_name, sel_name, sel_code, n_feats, models,
                    data_modalities, common_ids, y, base_out,
                    progress_count, total_runs
                )
            else:
                process_clf_selection_combo_cv(
                    ds_name, sel_name, sel_code, n_feats, models,
                    data_modalities, common_ids, y, base_out,
                    progress_count, total_runs
        )

def transform_selector_regression(X, selected_features):
    """Transform data using selected features for regression."""
    try:
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, selected_features].values
        return X[:, selected_features]
    except Exception as e:
        print(f"Error in transform: {str(e)}")
        return None

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
    # Initialize default metrics dictionary
    default_metrics = {
        'accuracy': np.nan,
        'precision': np.nan,
        'recall': np.nan,
        'f1': np.nan,
        'auc': np.nan
    }
    
    try:
        # Ensure inputs are proper numeric arrays
        X_train = safe_convert_to_numeric(X_train)
        X_val = safe_convert_to_numeric(X_val)
        
        # Handle potentially non-numeric targets
        try:
            if not np.issubdtype(y_train.dtype, np.number):
                y_train = pd.factorize(y_train)[0]
            if not np.issubdtype(y_val.dtype, np.number):
                y_val = pd.factorize(y_val)[0]
        except:
            # Direct conversion if factorize fails
            y_train = np.asarray(y_train, dtype=np.int32)
            y_val = np.asarray(y_val, dtype=np.int32)
            
        if X_train.size == 0 or X_val.size == 0:
            print(f"Warning: Empty input data for {model_name} in fold {fold_idx}")
            return None, default_metrics
            
        if model_name == "LogisticRegression":
            model = LogisticRegression(penalty='l2', solver='liblinear', random_state=42)
        elif model_name == "RandomForestClassifier":
            model = RandomForestClassifier(**MODEL_OPTIMIZATIONS["RandomForestClassifier"])
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
            'precision': precision_score(y_val, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_val, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_val, y_pred, average='weighted', zero_division=0)
        }

        if y_proba is not None:
            try:
                metrics['auc'] = roc_auc_score(y_val, y_proba)
            except Exception as e:
                print(f"Warning: Could not calculate AUC for {model_name} in fold {fold_idx}: {str(e)}")
                metrics['auc'] = np.nan

        if out_dir:
            try:
                cm = confusion_matrix(y_val, y_pred)
                class_labels = sorted(set(y_val))
                plot_confusion_matrix(cm, class_labels, plot_prefix,
                                    os.path.join(out_dir, f"{plot_prefix}_confusion.png"))
                if y_proba is not None:
                    plot_roc_curve_binary(model, X_val, y_val, class_labels, plot_prefix,
                                        os.path.join(out_dir, f"{plot_prefix}_roc.png"))
            except Exception as e:
                print(f"Warning: Could not create plots for {model_name} in fold {fold_idx}: {str(e)}")

        return model, metrics
    except Exception as e:
        print(f"Warning: Failed to train {model_name} in fold {fold_idx}: {str(e)}")
        return None, default_metrics

def safe_isnan(x):
    """Safely check for NaN values in arrays of any type."""
    try:
        if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
            return x.isna()
        elif hasattr(x, 'dtype') and np.issubdtype(x.dtype, np.number):
            return np.isnan(x)
        elif isinstance(x, (list, tuple)):
            try:
                # Try to convert to numeric array
                return np.isnan(np.asarray(x, dtype=np.float32))
            except:
                # If conversion fails, assume no NaNs
                return np.zeros(len(x), dtype=bool)
        else:
            # For non-numeric types, assume no NaNs
            return np.zeros(1, dtype=bool) if np.isscalar(x) else np.zeros(x.shape, dtype=bool)
    except:
        # Fallback for any other cases
        return False
            
def safe_convert_to_numeric(x):
    """Safely convert any array-like to numeric array, replacing non-numeric with 0."""
    try:
        if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
            return x.fillna(0).astype(np.float32).values
        elif hasattr(x, 'dtype') and np.issubdtype(x.dtype, np.number):
            return x.astype(np.float32)
        else:
            try:
                # Try to convert to numeric array
                arr = np.asarray(x, dtype=np.float32)
                # Replace any NaN values with 0
                return np.nan_to_num(arr, nan=0.0)
            except:
                # If conversion fails completely, return array of zeros
                shape = (len(x),) if hasattr(x, '__len__') else (1,)
                return np.zeros(shape, dtype=np.float32)
    except:
        # Return empty array as last resort
        return np.zeros(1, dtype=np.float32)

def cached_fit_transform_selector_classification(X, y, selector_code, n_feats, ds_name, modality_name, fold_idx=None):
    """Cached version of fit_transform for classification selectors."""
    # Use modality-independent key for more efficient caching
    key = f"{ds_name}_{modality_name}_{selector_code}_{n_feats}"
    
    if key in _selector_cache['sel_clf']:
        return _selector_cache['sel_clf'][key]
    
    try:
        # Convert DataFrame to numpy if needed and handle NaNs
        if isinstance(X, pd.DataFrame):
            X_numpy = X.fillna(0).values
        else:
            # Make sure X is numeric
            X_numpy = safe_convert_to_numeric(X)

        # Make sure y is numeric for classification (if it's categorical/string)
        if not hasattr(y, 'dtype') or not np.issubdtype(y.dtype, np.number):
            try:
                y = pd.factorize(y)[0]
            except:
                # If factorize fails, try to convert directly
                try:
                    y = np.asarray(y, dtype=np.int32)
                except:
                    print(f"Error: Unable to convert target to numeric")
                    raise ValueError("Target labels must be convertible to numeric")
        
        if selector_code == "boruta_clf":
            try:
                # Check if we have enough unique classes (at least 2) for classification
                unique_classes = np.unique(y)
                if len(unique_classes) < 2:
                    raise ValueError(f"Need at least 2 classes for classification, got {len(unique_classes)}")
                    
                # Use the new stable boruta_selector
                sel_idx = boruta_selector(
                    X_numpy, y,
                    k_features=n_feats,
                    task="clf",
                    random_state=42,
                    max_iter=150,
                    perc=85
                )
                X_selected = X_numpy[:, sel_idx]
                _selector_cache['sel_clf'][key] = (sel_idx, X_selected)
                return sel_idx, X_selected
                
            except (RuntimeError, ValueError) as err:
                print(f"Boruta skipped: {err}, using mutual_info_classif instead.")
                # Use a stable alternative
                selector = SelectKBest(mutual_info_classif, k=min(n_feats, X_numpy.shape[1]))
                X_selected = selector.fit_transform(X_numpy, y)
                selected_features = np.arange(X_numpy.shape[1])[selector.get_support()]
                
                _selector_cache['sel_clf'][key] = (selected_features, X_selected)
                return selected_features, X_selected
        elif selector_code == "mrmr_clf":
            selector = SelectKBest(mutual_info_classif, k=min(n_feats, X_numpy.shape[1]))
        elif selector_code == "fclassif":
            selector = SelectKBest(f_classif, k=min(n_feats, X_numpy.shape[1]))
        elif selector_code == "logistic_l1":
            selector = SelectFromModel(
                LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000, random_state=42),
                max_features=min(n_feats, X_numpy.shape[1]))
        elif selector_code == "chi2_selection":
            # Scale data to non-negative range for chi2
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X_numpy)
            selector = SelectKBest(chi2, k=min(n_feats, X_numpy.shape[1]))
            X_selected = selector.fit_transform(X_scaled, y)
            selected_features = np.arange(X_numpy.shape[1])[selector.get_support()]
            _selector_cache['sel_clf'][key] = (selected_features, X_selected)
            return selected_features, X_selected
        else:
            # Default to mutual_info_classif if unknown selector
            print(f"Unknown selector code: {selector_code}, using mutual_info_classif")
            selector = SelectKBest(mutual_info_classif, k=min(n_feats, X_numpy.shape[1]))
            
        # Standard selector handling
        X_selected = selector.fit_transform(X_numpy, y)
        selected_features = np.arange(X_numpy.shape[1])[selector.get_support()]
        
        _selector_cache['sel_clf'][key] = (selected_features, X_selected)
        return selected_features, X_selected
    except Exception as e:
        print(f"Error in feature selection: {str(e)}")
        # Return safe fallback values
        if isinstance(X, pd.DataFrame):
            X_numpy = X.fillna(0).values
        else:
            X_numpy = safe_convert_to_numeric(X)
            
        # Select first n_feats columns (or all if less than n_feats)
        max_cols = min(n_feats, X_numpy.shape[1])
        selected_features = np.arange(max_cols)
        X_selected = X_numpy[:, selected_features]
        return selected_features, X_selected

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
    elif selector_code == "boruta_reg":
        return BorutaPy(
            estimator=RF_for_BorutaReg(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                n_jobs=1  # Boruta requires single-threaded estimator
            ),
            n_estimators='auto',
            max_iter=100,
            random_state=42,
            verbose=0  # Reduce verbosity
        )
    elif selector_code == "mrmr_clf":
        return SelectKBest(mutual_info_classif, k=n_feats)
    elif selector_code == "fclassif":
        return SelectKBest(f_classif, k=n_feats)
    elif selector_code == "logistic_l1":
        return SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', random_state=42), max_features=n_feats)
    elif selector_code == "chi2_selection":
        return SelectKBest(chi2, k=n_feats)
    elif selector_code == "boruta_clf":
        return BorutaPy(
            estimator=RF_for_BorutaClf(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                n_jobs=1  # Boruta requires single-threaded estimator
            ),
            n_estimators='auto',
            max_iter=100,
            random_state=42,
            verbose=0  # Reduce verbosity
        )
    else:
        raise ValueError(f"Unknown selector code: {selector_code}")

def get_model_object(model_name, random_state=None):
    """Create and return a model instance based on the model name."""
    if model_name == "RandomForest":
        # Create a copy of the optimization dict and update random_state only if provided
        model_params = MODEL_OPTIMIZATIONS["RandomForest"].copy()
        if random_state is not None:
            model_params["random_state"] = random_state
        return RandomForestRegressor(**model_params)
    elif model_name == "RandomForestClassifier":
        model_params = MODEL_OPTIMIZATIONS["RandomForestClassifier"].copy()
        if random_state is not None:
            model_params["random_state"] = random_state
        return RandomForestClassifier(**model_params)
    elif model_name == "LinearRegression":
        # LinearRegression doesn't use random_state
        return LinearRegression(**MODEL_OPTIMIZATIONS["LinearRegression"])
    elif model_name == "SVR":
        # Create a copy of SVR params and update random_state if provided
        model_params = MODEL_OPTIMIZATIONS["SVR"].copy()
        if random_state is not None:
            model_params["random_state"] = random_state
        return SVR(**model_params)
    elif model_name == "LogisticRegression":
        # Create a copy of LogisticRegression params if they exist in MODEL_OPTIMIZATIONS
        model_params = MODEL_OPTIMIZATIONS.get("LogisticRegression", {}).copy()
        if random_state is not None:
            model_params["random_state"] = random_state
        # Set default parameters if not in MODEL_OPTIMIZATIONS
        if "penalty" not in model_params:
            model_params["penalty"] = 'l2'
        if "solver" not in model_params:
            model_params["solver"] = 'liblinear'
        return LogisticRegression(**model_params)
    elif model_name == "SVC":
        # Create a copy of SVC params if they exist in MODEL_OPTIMIZATIONS
        model_params = MODEL_OPTIMIZATIONS.get("SVC", {}).copy()
        if random_state is not None:
            model_params["random_state"] = random_state
        # Set default parameters if not in MODEL_OPTIMIZATIONS
        if "kernel" not in model_params:
            model_params["kernel"] = 'rbf'
        if "probability" not in model_params:
            model_params["probability"] = True
        return SVC(**model_params)
    else:
        raise ValueError(f"Unknown model: {model_name}")

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

def process_regression_datasets():
    """Process all regression datasets."""
    print("=== REGRESSION BLOCK (AML, Sarcoma) ===")
    reg_extractors = get_regression_extractors()
    reg_selectors = get_regression_selectors()
    reg_models = ["LinearRegression", "RandomForest", "SVR"]
    n_comps_list = [4, 8, 16]  # Reduced from [8, 16, MAX_COMPONENTS]
    n_feats_list = [4, 8, 16]  # Reduced from [8, 16, MAX_FEATURES]
    
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
    clf_models = ["LogisticRegression", "RandomForestClassifier", "SVC"]
    n_comps_list = [4, 8, 16]  # Reduced from [8, 16, MAX_COMPONENTS]
    n_feats_list = [4, 8, 16]  # Reduced from [8, 16, MAX_FEATURES]
    
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

if __name__ == "__main__":
    main()