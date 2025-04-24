#!/usr/bin/env python3

import os
import platform
import time
import hashlib
import joblib
import numpy as np
import pandas as pd
from functools import lru_cache
import psutil
from math import floor
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Type hints
from typing import Dict, List, Tuple, Optional, Union, Any

# For parallelization
from joblib import Parallel, delayed, parallel_config
import multiprocessing

# ──────────────────────────────────────────────────────────────────────────────
#  📌 HARDWARE-AWARE TUNING  –  Xeon Silver 4114 × 2  (20C/40T, 59 GB RAM)
# ──────────────────────────────────────────────────────────────────────────────

# ----- Memory limits --------------------------------------------------------
TOTAL_RAM = psutil.virtual_memory().total
MAX_ARRAY_SIZE = int(TOTAL_RAM * 0.8)  # 80% of RAM for arrays
CACHE_SIZE = int(TOTAL_RAM * 0.1)      # 10% of RAM for cache

# Memory optimization settings for VM with 58.6GB RAM
MEMORY_OPTIMIZATION = {
    "use_sparse": True,
    "dtype": np.float32,
    "chunk_size": 50_000,  # Adjusted for Windows stability
    "max_memory_usage": 0.85,  # Conservative memory usage for Windows
    "max_array_size": MAX_ARRAY_SIZE,  # Maximum array size (80% of RAM)
    "cache_size": CACHE_SIZE  # Cache size (10% of RAM)
}

# ----- Constants ------------------------------------------------------------
MAX_COMPONENTS = 1_024
MAX_FEATURES = 1_024
CHUNK_SIZE = MEMORY_OPTIMIZATION["chunk_size"]

# ----- CPU layout ------------------------------------------------------------
PHYS_CORES   = psutil.cpu_count(logical=False)      # 20
OUTER_PROCS  = max(1, int(PHYS_CORES*0.9))          # 18
OMP_BLAS_THREADS = 1                                # ✨ NEW

N_JOBS = OUTER_PROCS

# ----- Thread control --------------------------------------------------------
os.environ.update({
    "OMP_NUM_THREADS":      str(OMP_BLAS_THREADS),
    "OPENBLAS_NUM_THREADS": str(OMP_BLAS_THREADS),
    "MKL_NUM_THREADS":      str(OMP_BLAS_THREADS),
    "NUMEXPR_NUM_THREADS":  str(OMP_BLAS_THREADS),
})

# ----- Joblib default config -------------------------------------------------
JOBLIB_PARALLEL_CONFIG = {
    "n_jobs": OUTER_PROCS,
    "backend": "loky",
    "prefer": "processes",
    "batch_size": "auto",
    "verbose": 0,
    "max_nbytes": "100M",
    "temp_folder": os.path.join(os.getcwd(), "temp_joblib")
}

# ----- Thread-backend Parallel helper ----------------------------------------
def TParallel(*args, **kwargs):
    """Thread-backend Parallel that never spawns new processes."""
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

# Model-specific optimizations
MODEL_OPTIMIZATIONS = {
    "RandomForest": {
        "n_estimators": 500,  # Adjusted for Windows stability
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "n_jobs": N_JOBS,
        "random_state": 0,
        "verbose": 0,
        "warm_start": True,
        "bootstrap": True,
        "oob_score": True
    },
    "LinearRegression": {
        "n_jobs": N_JOBS,
        "copy_X": False
    },
    "SVR": {
        "kernel": 'rbf',
        "cache_size": 5_000,  # Adjusted for Windows stability
        "max_iter": -1,
        "tol": 1e-3
    }
}

# Regression models
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC

# Dimensionality Reduction (for regression)
from sklearn.decomposition import PCA, NMF, FastICA, FactorAnalysis, IncrementalPCA, MiniBatchNMF
from sklearn.cross_decomposition import PLSRegression

# Dimensionality Reduction (for classification)
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Data scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Splits & regression metrics
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt

# Classification metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, matthews_corrcoef, RocCurveDisplay
)

# Feature selection
from sklearn.feature_selection import (
    mutual_info_regression, mutual_info_classif,
    f_regression, f_classif, chi2, SelectKBest
)
from boruta import BorutaPy

# For boruta, we need separate (regressor/classifier) random forests
from sklearn.ensemble import RandomForestRegressor as RF_for_BorutaReg
from sklearn.ensemble import RandomForestClassifier as RF_for_BorutaClf

# For plotting
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse
import warnings
from pandas.api.types import is_sparse

# Memory caching setup
from joblib import Memory
MEM = Memory(location='./cache', verbose=0)

# -----------------------------------------------------------------------------
# 4. Pipeline helpers – no leakage, consistent dims
# -----------------------------------------------------------------------------
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import (
    PCA, NMF, FastICA, FactorAnalysis, KernelPCA
)
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import (
    SelectKBest, f_regression, f_classif, chi2
)
from sklearn.linear_model import Lasso, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC

# extractors ------------------------------------------------------------------
EXTRACTOR_FACTORY = {
    "PCA": lambda n: PCA(n_components=n, random_state=0, svd_solver='auto'),
    "NMF": lambda n: NMF(n_components=n, init="nndsvda", max_iter=10_000, random_state=0, solver='mu'),
    "ICA": lambda n: FastICA(n_components=n, max_iter=10_000, random_state=0, whiten='unit-variance', tol=1e-4),
    "FA": lambda n: FactorAnalysis(n_components=n, random_state=0),
    "KPCA": lambda n: KernelPCA(n_components=n, kernel="rbf"),
    "PLS": lambda n: PLSRegression(n_components=n, max_iter=1_000),
    "IncrementalPCA": lambda n: IncrementalPCA(n_components=n, batch_size=1000),
    "MiniBatchNMF": lambda n: MiniBatchNMF(n_components=n, init="nndsvda", batch_size=1000, random_state=0)
}

def fit_transform_extractor_classification(X_train, y_train, extractor, n_components):
    """Fit and transform data using the specified extractor for classification tasks."""
    # scaling
    if extractor.__class__.__name__ == "NMF":
        scl = MinMaxScaler(clip=True)
    else:
        scl = StandardScaler()

    X_train_scl = scl.fit_transform(X_train)

    if hasattr(extractor, "random_state"):
        extractor.random_state = 0

    if isinstance(extractor, LDA):
        n_classes = len(np.unique(y_train))
        max_lda = n_classes - 1
        if max_lda < 1:
            X_train_red = None
        else:
            n_components = min(n_components, max_lda)
            extractor.n_components = n_components
            X_train_red = extractor.fit_transform(X_train_scl, y_train)
    elif isinstance(extractor, KernelPCA):
        extractor.n_components = n_components
        X_train_red = extractor.fit_transform(X_train_scl)
    else:
        if hasattr(extractor, "n_components"):
            extractor.n_components = n_components
        X_train_red = extractor.fit_transform(X_train_scl)

    fitted_extractor = {
        "scaler": scl,
        "extractor": extractor
    }
    return fitted_extractor, X_train_red

def transform_extractor_classification(X_test, fitted_extractor):
    """Transform test data using a fitted extractor for classification tasks."""
    scl = fitted_extractor["scaler"]
    extractor = fitted_extractor["extractor"]
    X_test_scl = scl.transform(X_test)
    # Ensure the number of features matches what the extractor was trained on
    if X_test_scl.shape[1] != extractor.n_features_in_:
        raise ValueError(f"X has {X_test_scl.shape[1]} features, but {extractor.__class__.__name__} is expecting {extractor.n_features_in_} features as input")
    X_test_red = extractor.transform(X_test_scl)
    return X_test_red

def fit_transform_selector_classification(X_train, y_train, selector_code, n_feats):
    """Fit and transform data using the specified selector for classification tasks."""
    if selector_code=="mrmr_clf":
        mi = mutual_info_classif(X_train, y_train, random_state=0)
        idx = np.argsort(mi)[::-1]
        top_idx = idx[:n_feats]
        return list(top_idx), X_train.iloc[:, top_idx]

    elif selector_code=="fclassif":
        Fv, pv = f_classif(X_train, y_train)
        idx = np.argsort(Fv)[::-1]
        top_idx = idx[:n_feats]
        return list(top_idx), X_train.iloc[:, top_idx]

    elif selector_code=="logistic_l1":
        lr = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=0)
        lr.fit(X_train, y_train)
        coefs = np.abs(lr.coef_).sum(axis=0)
        idx = np.argsort(coefs)[::-1]
        top_idx = idx[:n_feats]
        return list(top_idx), X_train.iloc[:, top_idx]

    elif selector_code=="boruta_clf":
        rf = RF_for_BorutaClf(n_estimators=100, random_state=0)
        bor = BorutaPy(rf, n_estimators='auto', random_state=0)
        bor.fit(X_train.values, y_train.values)
        mask = bor.support_
        chosen_cols = np.where(mask)[0]
        if len(chosen_cols) > n_feats:
            ranks = bor.ranking_
            chosen_ranks = sorted(zip(chosen_cols, ranks[chosen_cols]), key=lambda x: x[1])
            chosen_cols = [x[0] for x in chosen_ranks[:n_feats]]
        return list(chosen_cols), X_train.iloc[:, chosen_cols]

    elif selector_code=="chi2_selection":
        X_clipped = np.clip(X_train, 0, None)
        sel = SelectKBest(chi2, k=min(n_feats, X_train.shape[1]))
        sel.fit(X_clipped, y_train)
        mask = sel.get_support()
        chosen_cols = np.where(mask)[0]
        return list(chosen_cols), X_train.iloc[:, chosen_cols]

    else:
        # fallback => no selection
        return list(range(X_train.shape[1])), X_train

def transform_selector_classification(X_test, chosen_cols):
    """Transform test data using a fitted selector for classification tasks."""
    return X_test.iloc[:, chosen_cols]

def train_classification_model(X_train, y_train, X_test, y_test,
                            model_name, out_dir=None, plot_prefix=""):
    """Train and evaluate a classification model."""
    if model_name=="LogisticRegression":
        model = LogisticRegression(penalty='l2', solver='liblinear', random_state=0)
    elif model_name=="RandomForest":
        model = RandomForestClassifier(n_estimators=100, random_state=0)
    elif model_name=="SVC":
        model = SVC(kernel='rbf', probability=True, random_state=0)
    else:
        raise ValueError(f"Unknown classification model {model_name}")

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall    = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1v       = f1_score(y_test, y_pred, average='weighted')
    mcc       = matthews_corrcoef(y_test, y_pred)

    try:
        unique_cl = np.unique(y_test)
        if len(unique_cl)==2:
            y_proba = model.predict_proba(X_test)[:, 1]
            aucv = roc_auc_score(y_test, y_proba)
        else:
            aucv = np.nan
    except:
        aucv = np.nan

    cm = confusion_matrix(y_test, y_pred)
    if out_dir and plot_prefix:
        os.makedirs(out_dir, exist_ok=True)
        cm_path = os.path.join(out_dir, f"{plot_prefix}_CM.png")
        str_labels = [str(lb) for lb in sorted(np.unique(y_train))]
        plot_confusion_matrix(cm, str_labels, plot_prefix, cm_path)

        if len(np.unique(y_test)) == 2:
            roc_path = os.path.join(out_dir, f"{plot_prefix}_ROC.png")
            plot_roc_curve_binary(model, X_test, y_test, str_labels, plot_prefix, roc_path)

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1v,
        "MCC": mcc,
        "AUROC": aucv,
        "Train_Time_Seconds": train_time
    }
    return model, metrics

def fit_transform_extractor_regression(X_train, y_train, extractor, n_components):
    """Fit and transform data using the specified extractor for regression tasks."""
    # scaling
    if extractor.__class__.__name__ == "NMF":
        scl = MinMaxScaler(clip=True)
    else:
        scl = StandardScaler()

    X_train_scl = scl.fit_transform(X_train)

    if hasattr(extractor, "random_state"):
        extractor.random_state = 0

    if isinstance(extractor, PLSRegression):
        extractor.n_components = n_components
        X_train_red = extractor.fit_transform(X_train_scl, y_train)[0]
    else:
        if hasattr(extractor, "n_components"):
            extractor.n_components = n_components
        X_train_red = extractor.fit_transform(X_train_scl)

    fitted_extractor = {
        "scaler": scl,
        "extractor": extractor
    }
    return fitted_extractor, X_train_red

def transform_extractor_regression(X_test, fitted_extractor):
    """Transform test data using a fitted extractor for regression tasks."""
    scl = fitted_extractor["scaler"]
    extractor = fitted_extractor["extractor"]
    X_test_scl = scl.transform(X_test)
    # Ensure the number of features matches what the extractor was trained on
    if X_test_scl.shape[1] != extractor.n_features_in_:
        raise ValueError(f"X has {X_test_scl.shape[1]} features, but {extractor.__class__.__name__} is expecting {extractor.n_features_in_} features as input")
    X_test_red = extractor.transform(X_test_scl)
    return X_test_red

def fit_transform_selector_regression(X_train, y_train, selector_code, n_feats):
    """Fit and transform data using the specified selector for regression tasks."""
    if selector_code=="mrmr_reg":
        mi = mutual_info_regression(X_train, y_train, random_state=0)
        idx = np.argsort(mi)[::-1]
        top_idx = idx[:n_feats]
        return list(top_idx), X_train.iloc[:, top_idx]

    elif selector_code=="freg":
        Fv, pv = f_regression(X_train, y_train)
        idx = np.argsort(Fv)[::-1]
        top_idx = idx[:n_feats]
        return list(top_idx), X_train.iloc[:, top_idx]

    elif selector_code=="lasso":
        lasso = Lasso(alpha=0.01, max_iter=10_000, random_state=0)
        lasso.fit(X_train, y_train)
        coefs = np.abs(lasso.coef_)
        idx = np.argsort(coefs)[::-1]
        top_idx = idx[:n_feats]
        return list(top_idx), X_train.iloc[:, top_idx]

    elif selector_code=="enet":
        enet = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10_000, random_state=0)
        enet.fit(X_train, y_train)
        coefs = np.abs(enet.coef_)
        idx = np.argsort(coefs)[::-1]
        top_idx = idx[:n_feats]
        return list(top_idx), X_train.iloc[:, top_idx]

    elif selector_code=="boruta_reg":
        rf = RF_for_BorutaReg(n_estimators=100, random_state=0)
        bor = BorutaPy(rf, n_estimators='auto', random_state=0)
        bor.fit(X_train.values, y_train.values)
        mask = bor.support_
        chosen_cols = np.where(mask)[0]
        if len(chosen_cols) > n_feats:
            ranks = bor.ranking_
            chosen_ranks = sorted(zip(chosen_cols, ranks[chosen_cols]), key=lambda x: x[1])
            chosen_cols = [x[0] for x in chosen_ranks[:n_feats]]
        return list(chosen_cols), X_train.iloc[:, chosen_cols]

    else:
        # fallback => no selection
        return list(range(X_train.shape[1])), X_train

def transform_selector_regression(X_test, chosen_cols):
    """Transform test data using a fitted selector for regression tasks."""
    return X_test.iloc[:, chosen_cols]

def train_regression_model(X_train, y_train, X_test, y_test,
                         model_name, out_dir=None, plot_prefix=""):
    """Train and evaluate a regression model."""
    if model_name=="LinearRegression":
        model = LinearRegression(n_jobs=N_JOBS)
    elif model_name=="RandomForest":
        model = RandomForestRegressor(n_estimators=100, random_state=0)
    elif model_name=="SVR":
        model = SVR(kernel='rbf', cache_size=2000)
    else:
        raise ValueError(f"Unknown regression model {model_name}")

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    if out_dir and plot_prefix:
        os.makedirs(out_dir, exist_ok=True)
        scatter_path = os.path.join(out_dir, f"{plot_prefix}_scatter.png")
        plot_regression_scatter(y_test, y_pred, plot_prefix, scatter_path)
        
        residuals_path = os.path.join(out_dir, f"{plot_prefix}_residuals.png")
        plot_regression_residuals(y_test, y_pred, plot_prefix, residuals_path)

    metrics = {
        "MSE": mse,
        "RMSE": sqrt(mse),
        "MAE": mae,
        "R2": r2,
        "Train_Time_Seconds": train_time
    }
    return model, metrics

@MEM.cache
def build_extraction_pipeline(name: str, n_components: int):
    scaler = MinMaxScaler() if name in ["NMF", "MiniBatchNMF"] else StandardScaler()
    extractor = EXTRACTOR_FACTORY[name](n_components)
    return Pipeline([("sc", scaler), ("extr", extractor)])

# selectors -------------------------------------------------------------------
SELECTOR_FACTORY = {
    "f_reg": lambda k: SelectKBest(f_regression, k=k),
    "f_clf": lambda k: SelectKBest(f_classif, k=k),
    "chi2": lambda k: SelectKBest(chi2, k=k),
}

@MEM.cache
def build_selection_pipeline(code: str, k: int, task: str):
    scaler = StandardScaler()
    if code in {"lasso", "enet", "logistic_l1"}:
        if code == "lasso":
            base = Lasso(alpha=0.01, max_iter=10_000, random_state=0)
        elif code == "enet":
            base = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10_000, random_state=0)
        else:  # logistic_l1
            base = LogisticRegression(penalty="l1", solver="liblinear", random_state=0)
        return Pipeline([("sc", scaler), ("sel", base)])
    if code in SELECTOR_FACTORY:
        return Pipeline([("sc", scaler), ("sel", SELECTOR_FACTORY[code](k))])
    raise ValueError(code)

###############################################################################
# A) CONFIG OF DATASETS
###############################################################################

REGRESSION_DATASETS = [
    {
        "name": "AML",
        "clinical_file": "clinical/aml.csv",
        "omics_dir": "aml",
        "id_col": "sampleID",
        "outcome_col": "lab_procedure_bone_marrow_blast_cell_outcome_percent_value",
    }
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

CLASSIFICATION_DATASETS = [
    {
        "name": "Colon",
        "clinical_file": "clinical/colon.csv",
        "omics_dir": "colon",
        "id_col": "sampleID",
        "outcome_col": "pathologic_T",
    }
]
"""
    {
        "name": "Breast",
        "clinical_file": "clinical/breast.csv",
        "omics_dir": "breast",
        "id_col": "sampleID",
        "outcome_col": "pathologic_T",
    },
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

###############################################################################
# B) MISSING MODALITIES CONFIG
###############################################################################

# Configuration for missing modalities simulation
MISSING_MODALITIES_CONFIG = {
    "enabled": True,  # Set to False to disable missing modalities simulation
    "missing_percentages": [0.0, 0.25],  # Percentages of samples with missing modalities
    "random_seed": 42,  # Base random seed for reproducibility
    "modality_names": ["Gene Expression", "Methylation", "miRNA"],  # Order must match data_modalities
    "cv_fold_seed_offset": 1_000  # Offset to add to random seed for each CV fold
}

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

def try_read_file(file_path):
    # Try different separators with optimized data types
    for sep in ['\t', ',', ';']:
        try:
            # First try with C engine and optimized settings
            df = pd.read_csv(
                file_path,
                sep=sep,
                index_col=0,
                header=0,
                dtype=MEMORY_OPTIMIZATION["dtype"],
                low_memory=False,
                engine='c',
                memory_map=True,
                chunksize=MEMORY_OPTIMIZATION["chunk_size"]
            )
            
            # If chunksize was used, concatenate chunks
            if isinstance(df, pd.io.parsers.TextFileReader):
                chunks = []
                for chunk in df:
                    # Only convert to sparse if memory usage is high and not already sparse
                    if (MEMORY_OPTIMIZATION["use_sparse"] and 
                        chunk.memory_usage().sum() > 1e6 and 
                        not is_sparse(chunk)):
                        # Convert to numpy array first, then to sparse
                        chunk = pd.DataFrame.sparse.from_spmatrix(
                            scipy.sparse.csr_matrix(chunk.values),
                            index=chunk.index,
                            columns=chunk.columns
                        )
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=False)
            
            if not df.empty and len(df.columns) > 0:
                # Convert to sparse if beneficial and not already sparse
                if (MEMORY_OPTIMIZATION["use_sparse"] and 
                    df.memory_usage().sum() > 1e6 and 
                    not is_sparse(df)):
                    df = pd.DataFrame.sparse.from_spmatrix(
                        scipy.sparse.csr_matrix(df.values),
                        index=df.index,
                        columns=df.columns
                    )
                # Force 32-bit storage
                df = df.astype(np.float32, copy=False)
                return df
        except Exception as e:
            continue
    
    try:
        # Fallback to python engine without low_memory parameter
        df = pd.read_csv(
            file_path,
            sep=None,
            engine='python',
            index_col=0,
            header=0,
            dtype=MEMORY_OPTIMIZATION["dtype"],
            memory_map=True,
            chunksize=MEMORY_OPTIMIZATION["chunk_size"]
        )
        
        # If chunksize was used, concatenate chunks
        if isinstance(df, pd.io.parsers.TextFileReader):
            chunks = []
            for chunk in df:
                # Only convert to sparse if memory usage is high and not already sparse
                if (MEMORY_OPTIMIZATION["use_sparse"] and 
                    chunk.memory_usage().sum() > 1e6 and 
                    not is_sparse(chunk)):
                    # Convert to numpy array first, then to sparse
                    chunk = pd.DataFrame.sparse.from_spmatrix(
                        scipy.sparse.csr_matrix(chunk.values),
                        index=chunk.index,
                        columns=chunk.columns
                    )
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=False)
            
        if not df.empty and len(df.columns) > 0:
            # Convert to sparse if beneficial and not already sparse
            if (MEMORY_OPTIMIZATION["use_sparse"] and 
                df.memory_usage().sum() > 1e6 and 
                not is_sparse(df)):
                df = pd.DataFrame.sparse.from_spmatrix(
                    scipy.sparse.csr_matrix(df.values),
                    index=df.index,
                    columns=df.columns
                )
            # Force 32-bit storage
            df = df.astype(np.float32, copy=False)
            return df
    except Exception as e:
        raise ValueError(f"Failed to read {os.path.basename(file_path)}: {str(e)}")

def load_omics_and_clinical(ds_config):
    odir = ds_config["omics_dir"]
    
    file_paths = [
        os.path.join(odir, "exp.csv"),
        os.path.join(odir, "methy.csv"),
        os.path.join(odir, "mirna.csv")
    ]
    
    # Read all files in parallel with threading backend
    results = TParallel(n_jobs=N_JOBS)(
        delayed(try_read_file)(path) for path in file_paths
    )
    
    exp_df, methy_df, mirna_df = results
    
    # Read clinical data with optimized settings
    clinical_df = pd.read_csv(
        ds_config["clinical_file"],
        sep=None,
        engine='python',
        dtype={'sampleID': str, ds_config["outcome_col"]: MEMORY_OPTIMIZATION["dtype"]},
        memory_map=True
    )
    
    return exp_df, methy_df, mirna_df, clinical_df

def strip_and_slice_columns(col_list):
    newcols = []
    for c in col_list:
        s2 = c.strip().strip('"')
        s3 = fix_tcga_id_slicing(s2)
        newcols.append(s3)
    return newcols

def prepare_data(ds_config, exp_df, methy_df, mirna_df, is_regression=True):
    id_col = ds_config["id_col"]
    out_col = ds_config["outcome_col"]

    # --- ONLY FOR KIDNEY: strip trailing 'A' from the ID column in memory ---
    if ds_config["name"] == "Kidney":
        def remove_trailing_A(s):
            if isinstance(s, str) and s.endswith('A'):
                return s[:-1]
            return s

        clinical_df_raw = pd.read_csv(ds_config["clinical_file"], sep=None, engine='python')
        if id_col not in clinical_df_raw.columns:
            raise ValueError(f"ID col '{id_col}' not found in {ds_config['clinical_file']}.")
        clinical_df_raw[id_col] = clinical_df_raw[id_col].apply(remove_trailing_A)
        clinical_df_raw[id_col] = clinical_df_raw[id_col].apply(fix_tcga_id_slicing)
    else:
        clinical_df_raw = pd.read_csv(ds_config["clinical_file"], sep=None, engine='python')
        if id_col not in clinical_df_raw.columns:
            raise ValueError(f"ID col '{id_col}' not found in {ds_config['clinical_file']}.")
        clinical_df_raw[id_col] = clinical_df_raw[id_col].apply(fix_tcga_id_slicing)

    if out_col not in clinical_df_raw.columns:
        raise ValueError(f"Outcome col '{out_col}' not found in {ds_config['clinical_file']}.")

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

    # Intersection of sample IDs
    common_ids = set(clinical_df[id_col])
    for df_mod in data_modalities.values():
        common_ids = common_ids.intersection(df_mod.columns)
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

    return data_modalities, common_ids, y_series, clinical_filtered

###############################################################################
# C) MISSING MODALITIES UTILITIES
###############################################################################

def create_modality_availability_matrix(n_samples, n_modalities, missing_percentage, random_seed):
    """
    Create a binary matrix indicating which modalities are available for each sample.
    
    Args:
        n_samples: Number of samples
        n_modalities: Number of modalities
        missing_percentage: Percentage of samples that should have at least one missing modality
        random_seed: Random seed for reproducibility
    
    Returns:
        numpy.ndarray: Binary matrix of shape (n_samples, n_modalities)
    """
    np.random.seed(random_seed)
    
    # Create matrix of ones
    availability_matrix = np.ones((n_samples, n_modalities), dtype=np.int8)
    
    if missing_percentage > 0:
        # Calculate number of samples to modify
        n_samples_to_modify = int(n_samples * missing_percentage)
        
        # Randomly select samples to modify
        samples_to_modify = np.random.choice(n_samples, n_samples_to_modify, replace=False)
        
        # For each selected sample, randomly set some modalities to 0
        for sample_idx in samples_to_modify:
            # Ensure at least one modality remains (don't drop all modalities)
            n_modalities_to_drop = np.random.randint(1, n_modalities)
            modalities_to_drop = np.random.choice(n_modalities, n_modalities_to_drop, replace=False)
            availability_matrix[sample_idx, modalities_to_drop] = 0
    
    return availability_matrix

def apply_missing_modalities(data_modalities, availability_matrix, modality_names):
    """
    Apply missing modalities to the data based on the availability matrix.
    
    Args:
        data_modalities: Dictionary of modality dataframes
        availability_matrix: Binary matrix indicating modality availability
        modality_names: List of modality names in the same order as availability_matrix columns
    
    Returns:
        dict: Modified data_modalities with missing modalities
    """
    modified_data = {}
    
    for i, modality_name in enumerate(modality_names):
        if modality_name in data_modalities:
            df = data_modalities[modality_name].copy()
            # Get samples that should have this modality missing
            missing_samples = np.where(availability_matrix[:, i] == 0)[0]
            if len(missing_samples) > 0:
                # Handle sparse arrays
                if is_sparse(df):
                    # Convert to dense for modification
                    df = df.sparse.to_dense()
                    # Set values to NaN
                    df.iloc[missing_samples, :] = np.nan
                    # Convert back to sparse if beneficial
                    if df.memory_usage().sum() > 1e6:  # Only convert back if memory usage is high
                        df = df.astype(pd.SparseDtype("float64", np.nan))
                else:
                    # Regular dense array handling
                    df.iloc[missing_samples, :] = np.nan
            modified_data[modality_name] = df
    
    return modified_data

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
    n_modalities = len(MISSING_MODALITIES_CONFIG["modality_names"])

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

from sklearn.impute import SimpleImputer
_imputer = SimpleImputer(strategy="mean")

def _impute_nan(arr: np.ndarray) -> np.ndarray:
    """Impute NaN values in a numpy array using mean imputation."""
    if np.isnan(arr).any():
        return _imputer.fit_transform(arr)
    return arr

def merge_modalities(*arrays, strategy: str = "concat") -> np.ndarray:
    """
    Merge an arbitrary number of numpy arrays (same number of rows).

    Parameters
    ----------
    *arrays     Variable-length list of 2-D arrays (or None/empty).
    strategy    'concat' | 'average' | 'sum' | 'max'

    Returns
    -------
    np.ndarray  The merged matrix (float32).  Empty (0, 0) array if nothing
                usable was supplied.
    """
    # keep only non-empty, non-None inputs
    valid = [a for a in arrays if a is not None and getattr(a, "size", 0) > 0]
    if not valid:
        return np.empty((0, 0), dtype=np.float32)

    if strategy == "concat":
        merged = np.concatenate(valid, axis=1).astype(np.float32, copy=False)
        return _impute_nan(merged)

    # for element-wise operations the matrices must have equal column count
    target = max(a.shape[1] for a in valid)
    padded = [_pad(a, target) for a in valid]

    if strategy == "average":
        count = sum((a != 0).astype(int) for a in padded)
        count[count == 0] = 1          # avoid division by 0
        merged = (sum(padded) / count).astype(np.float32, copy=False)
        return _impute_nan(merged)
    if strategy == "sum":
        merged = sum(padded).astype(np.float32, copy=False)
        return _impute_nan(merged)
    if strategy == "max":
        merged = np.maximum.reduce(padded).astype(np.float32, copy=False)
        return _impute_nan(merged)

    raise ValueError(f"Unknown merge strategy '{strategy}'")

###############################################################################
# D) EXTRACTORS & SELECTORS
###############################################################################

# For regression
def get_regression_extractors():
    return {
        "PCA": PCA(),
        "NMF": NMF(max_iter=10000, init='nndsvda'),
        "ICA": FastICA(max_iter=10000, tol=1e-4, whiten='unit-variance'),
        "FA": FactorAnalysis(),
        "PLS": PLSRegression(max_iter=1000, tol=1e-6)
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
    return {
        "PCA": PCA(),
        "ICA": FastICA(max_iter=10000, tol=1e-2),
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

def plot_regression_scatter(y_test, y_pred, title, out_path):
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

def plot_regression_residuals(y_test, y_pred, title, out_path):
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

def plot_confusion_matrix(cm, class_labels, title, out_path):
    fig, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt='d',
                cmap='Blues', xticklabels=class_labels, yticklabels=class_labels, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)

def plot_roc_curve_binary(model, X_test, y_test, class_labels, title, out_path):
    y_proba = model.predict_proba(X_test)[:, 1]
    fig, ax = plt.subplots(figsize=(5,5))
    disp = RocCurveDisplay.from_predictions(y_test, y_proba, name='Binary ROC', ax=ax)
    ax.set_title(title + " - ROC Curve")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)

###############################################################################
# F) TRAIN & EVAL: REGRESSION
###############################################################################

def process_modality(modality_name, modality_df, id_train, id_val, idx_test, y_train, extr_obj, ncomps, idx_to_id):
    """Process a single modality with proper parallel processing hierarchy."""
    # Process data in chunks for better memory management
    def process_chunk(chunk_ids, is_train=False):
        # Filter out IDs that don't exist in the DataFrame
        valid_ids = [id_ for id_ in chunk_ids if id_ in modality_df.columns]
        if not valid_ids:
            return None
        
        try:
            chunk_data = modality_df.loc[:, valid_ids].transpose()
            chunk_data = chunk_data.fillna(0)
            return chunk_data.values.astype(MEMORY_OPTIMIZATION["dtype"])
        except Exception as e:
            print(f"Warning: Error processing chunk for {modality_name}: {str(e)}")
            return None

    # Split IDs into larger chunks for better parallelization
    train_chunks = [id_train[i:i + CHUNK_SIZE] for i in range(0, len(id_train), CHUNK_SIZE)]
    val_chunks = [id_val[i:i + CHUNK_SIZE] for i in range(0, len(id_val), CHUNK_SIZE)]
    test_chunks = [[idx_to_id[idx] for idx in idx_test[i:i + CHUNK_SIZE]] 
                   for i in range(0, len(idx_test), CHUNK_SIZE)]

    # Process chunks in parallel with threading backend
    with parallel_config(backend="threading"):
        X_train_chunks = Parallel(n_jobs=N_JOBS)(
            delayed(process_chunk)(chunk, True) for chunk in train_chunks
        )
        X_val_chunks = Parallel(n_jobs=N_JOBS)(
            delayed(process_chunk)(chunk) for chunk in val_chunks
        )
        X_test_chunks = Parallel(n_jobs=N_JOBS)(
            delayed(process_chunk)(chunk) for chunk in test_chunks
        )

    # Filter out None chunks
    X_train_chunks = [chunk for chunk in X_train_chunks if chunk is not None and chunk.shape[0] > 0]
    X_val_chunks = [chunk for chunk in X_val_chunks if chunk is not None and chunk.shape[0] > 0]
    X_test_chunks = [chunk for chunk in X_test_chunks if chunk is not None and chunk.shape[0] > 0]

    # Check if we have any valid data
    if not X_train_chunks or not X_val_chunks or not X_test_chunks:
        print(f"Warning: No valid data found for modality {modality_name}")
        return None, None, None

    try:
        # Concatenate chunks with memory optimization
        X_train_np = np.vstack(X_train_chunks).astype(MEMORY_OPTIMIZATION["dtype"])
        X_val_np = np.vstack(X_val_chunks).astype(MEMORY_OPTIMIZATION["dtype"])
        X_test_np = np.vstack(X_test_chunks).astype(MEMORY_OPTIMIZATION["dtype"])

        # Create extractor copy for safety
        extr_copy = type(extr_obj)()
        for key, value in extr_obj.get_params().items():
            if hasattr(extr_copy, key):
                setattr(extr_copy, key, value)

        if hasattr(extr_copy, "n_components"):
            if isinstance(extr_copy, PLSRegression):
                max_components = min(X_train_np.shape[0] - 1, X_train_np.shape[1], ncomps)
                extr_copy.n_components = max_components
                extr_copy.max_iter = 1000
                extr_copy.tol = 1e-6
            else:
                extr_copy.n_components = min(ncomps, X_train_np.shape[1], X_train_np.shape[0])

        if hasattr(extr_copy, "random_state"):
            extr_copy.random_state = 0

        # Choose scaler based on extractor type
        if extr_copy.__class__.__name__ == "NMF":
            train_scaler = MinMaxScaler(feature_range=(0, 1))
            val_scaler = MinMaxScaler(feature_range=(0, 1))
            test_scaler = MinMaxScaler(feature_range=(0, 1))
        else:
            train_scaler = StandardScaler()
            val_scaler = StandardScaler()
            test_scaler = StandardScaler()

        # Scale the data
        X_train_scaled = train_scaler.fit_transform(X_train_np)
        X_val_scaled = val_scaler.fit_transform(X_val_np)
        X_test_scaled = test_scaler.fit_transform(X_test_np)

        # For PLS => pass y
        if isinstance(extr_copy, PLSRegression):
            Y_train_arr = y_train.reshape(-1, 1).astype(MEMORY_OPTIMIZATION["dtype"])
            X_train_trans = extr_copy.fit_transform(X_train_scaled, Y_train_arr)[0]
            
            val_extr = PLSRegression(
                n_components=min(X_val_scaled.shape[0] - 1, X_val_scaled.shape[1], extr_copy.n_components),
                max_iter=1000,
                tol=1e-6
            )
            if len(Y_train_arr) > X_val_scaled.shape[0]:
                Y_val_arr = Y_train_arr[:X_val_scaled.shape[0]]
            else:
                Y_val_arr = np.pad(Y_train_arr, ((0, max(0, X_val_scaled.shape[0] - len(Y_train_arr))), (0, 0)), 'constant')
            
            val_extr.fit(X_val_scaled, Y_val_arr)
            X_val_trans = val_extr.transform(X_val_scaled)
            
            test_extr = PLSRegression(
                n_components=min(X_test_scaled.shape[0] - 1, X_test_scaled.shape[1], extr_copy.n_components),
                max_iter=1000,
                tol=1e-6
            )
            if len(Y_train_arr) > X_test_scaled.shape[0]:
                Y_test_arr = Y_train_arr[:X_test_scaled.shape[0]]
            else:
                Y_test_arr = np.pad(Y_train_arr, ((0, max(0, X_test_scaled.shape[0] - len(Y_train_arr))), (0, 0)), 'constant')
            
            test_extr.fit(X_test_scaled, Y_test_arr)
            X_test_trans = test_extr.transform(X_test_scaled)
        else:
            X_train_trans = extr_copy.fit_transform(X_train_scaled)
            X_val_trans = extr_copy.transform(X_val_scaled)
            X_test_trans = extr_copy.transform(X_test_scaled)

        # Ensure all have the same number of components
        target_comps = min(X_train_trans.shape[1], X_val_trans.shape[1], X_test_trans.shape[1])
        X_train_trans = X_train_trans[:, :target_comps].astype(MEMORY_OPTIMIZATION["dtype"])
        X_val_trans = X_val_trans[:, :target_comps].astype(MEMORY_OPTIMIZATION["dtype"])
        X_test_trans = X_test_trans[:, :target_comps].astype(MEMORY_OPTIMIZATION["dtype"])

        return X_train_trans, X_val_trans, X_test_trans
    except Exception as e:
        print(f"Error processing modality {modality_name}: {str(e)}")
        return None, None, None

def train_evaluate_model(model_name, model, X_train, y_train, X_val):
    """
    Train and evaluate a model on the given data.
    Returns the model name and predictions.
    """
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_val_pred = model.predict(X_val)
    
    return model_name, y_val_pred

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

    # Merge modalities
    X_train_merged = merge_modalities(*[r[0] for r in valid_results])
    X_val_merged = merge_modalities(*[r[1] for r in valid_results])
    X_test_merged = merge_modalities(*[r[2] for r in valid_results])

    # Skip if no valid data after merging
    if X_train_merged.size == 0 or X_val_merged.size == 0 or X_test_merged.size == 0:
        print(f"Warning: No valid data after merging in fold {fold_idx}")
        return {}

    # Train and evaluate models in parallel with threading backend
    model_results = {}
    for model_name in reg_models:
        try:
            if model_name == "RandomForest":
                model = RandomForestRegressor(**MODEL_OPTIMIZATIONS["RandomForest"])
            elif model_name == "LinearRegression":
                model = LinearRegression(**MODEL_OPTIMIZATIONS["LinearRegression"])
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
                    except Exception as e:
                        print(f"Warning: Failed to get metrics for {model_name} in fold {i}: {e}")
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
        # Get best model based on average MSE across folds
        best_model_name = min(
            [(model, np.mean([r['mse'] for r in all_results if r['Model'] == model])) 
             for model in reg_models],
            key=lambda x: x[1]
        )[0]

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
            X_train_merged = merge_modalities(*[r[0] for r in valid_results])
            X_test_merged = merge_modalities(*[r[2] for r in valid_results])

            # Train final model
            if best_model_name == "RandomForest":
                final_model = RandomForestRegressor(**MODEL_OPTIMIZATIONS["RandomForest"])
            elif best_model_name == "LinearRegression":
                final_model = LinearRegression(**MODEL_OPTIMIZATIONS["LinearRegression"])
            elif best_model_name == "SVR":
                final_model = SVR(**MODEL_OPTIMIZATIONS["SVR"])

            # Train with heavy CPU section
            with heavy_cpu_section():
                final_model.fit(X_train_merged, y_train)
            y_test_pred = final_model.predict(X_test_merged)

            # Save final model
            joblib.dump(
                final_model,
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
    y_arr       = np.array(y)

    id_temp, id_test, y_temp, y_test = train_test_split(
        all_ids_arr, y_arr, test_size=test_size, random_state=0
    )

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    cv_metrics = {}

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(id_temp)):
        id_train, id_val = id_temp[train_idx], id_temp[val_idx]
        y_train,  y_val  = y_temp[train_idx],  y_temp[val_idx]

        train_list, val_list = [], []
        for modality_name, df_mod in data_modalities.items():
            df_train = df_mod.loc[:, id_train].transpose().apply(pd.to_numeric, errors='coerce').fillna(0)
            chosen, X_tr = cached_fit_transform_selector_regression(
                df_train, pd.Series(y_train), sel_code, n_feats, ds_name, modality_name
            )
            df_val   = df_mod.loc[:, id_val].transpose().apply(pd.to_numeric, errors='coerce').fillna(0)
            X_va     = transform_selector_regression(df_val, chosen)
            train_list.append(np.array(X_tr))
            val_list.append(np.array(X_va))

        for merge_str in ["concat", "average", "sum", "max"]:
            try:
                X_tr_m = merge_modalities(*train_list, strategy=merge_str)
                X_va_m = merge_modalities(*val_list,   strategy=merge_str)
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
                        random_state=0
                    )
                elif model_name == "LinearRegression":
                    model_dict[model_name] = LinearRegression(n_jobs=N_JOBS)
                elif model_name == "SVR":
                    model_dict[model_name] = SVR(kernel='rbf', cache_size=2000)
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
                df_train, pd.Series(y_temp), sel_code, n_feats, ds_name, modality_name
            )
            df_test  = df_mod.loc[:, id_test].transpose().apply(pd.to_numeric, errors='coerce').fillna(0)
            X_te     = transform_selector_regression(df_test, chosen_cols)
            train_list.append(np.array(X_tr))
            test_list.append(np.array(X_te))

        X_tr_m = merge_modalities(*train_list, strategy=merge_str)
        X_te_m = merge_modalities(*test_list,  strategy=merge_str)

        # Create model for final evaluation
        final_model, test_mets = train_regression_model(
            X_tr_m, y_temp, X_te_m, y_test,
            model_name,
            out_dir=os.path.join(base_out, "plots"),
            plot_prefix=f"{ds_name}_FINAL_SELECT_{sel_name}_{n_feats}_{merge_str}_{model_name}"
        )
        avg_mets.update({f"Test_{k}": v for k, v in test_mets.items()})
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

        train_list, val_list = [], []
        for modality_name, df_mod in data_modalities.items():
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
                X_tr_m = merge_modalities(*train_list, strategy=merge_str)
                X_va_m = merge_modalities(*val_list,   strategy=merge_str)
            except Exception as e:
                print(f"Skipping merge '{merge_str}' fold {fold_idx}: {e}")
                continue

            # Create model instances from model names
            model_dict = {}
            for model_name in clf_models:
                if model_name == "LogisticRegression":
                    model_dict[model_name] = LogisticRegression(penalty='l2', solver='liblinear', random_state=0)
                elif model_name == "RandomForest":
                    model_dict[model_name] = RandomForestClassifier(n_estimators=100, random_state=0)
                elif model_name == "SVC":
                    model_dict[model_name] = SVC(kernel='rbf', probability=True, random_state=0)
                else:
                    raise ValueError(f"Unknown classification model {model_name}")

            for model_name, model in model_dict.items():
                model, mets = train_classification_model(
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
            "Dataset": ds_name, "Workflow": "Extraction-CV",
            "Extractor": extr_name, "n_components": ncomps,
            "MergeStrategy": merge_str, "Model": model_name
        })

        train_list, test_list = [], []
        for modality_name, df_mod in data_modalities.items():
            df_train = df_mod.loc[:, id_temp].transpose().apply(pd.to_numeric, errors='coerce').fillna(0)
            fitted_extr, X_tr = cached_fit_transform_extractor_classification(
                df_train, y_temp, extr_obj, ncomps, ds_name, modality_name
            )
            df_test = df_mod.loc[:, id_test].transpose().apply(pd.to_numeric, errors='coerce').fillna(0)
            X_te    = transform_extractor_classification(df_test, fitted_extr)
            train_list.append(X_tr)
            test_list.append(X_te)

        X_tr_m = merge_modalities(*train_list, strategy=merge_str)
        X_te_m = merge_modalities(*test_list,  strategy=merge_str)

        # Create model for final evaluation
        final_model, test_mets = train_classification_model(
            X_tr_m, y_temp, X_te_m, y_test,
            model_name,
            out_dir=os.path.join(base_out, "plots"),
            plot_prefix=f"{ds_name}_FINAL_EXTRACT_{extr_name}_{ncomps}_{merge_str}_{model_name}"
        )
        avg_mets.update({f"Test_{k}": v for k, v in test_mets.items()})
        avg_cv_results.append(avg_mets)

        joblib.dump(final_model,
                    os.path.join(base_out, "models", f"{ds_name}_FINAL_EXTRACT_{extr_name}_{ncomps}_{merge_str}_{model_name}.pkl"))

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
        y_train,  y_val  = y_temp[train_idx],  y_temp[val_idx]

        train_list, val_list = [], []
        for modality_name, df_mod in data_modalities.items():
            df_train = df_mod.loc[:, id_train].transpose().apply(pd.to_numeric, errors='coerce').fillna(0)
            chosen_cols, X_tr = cached_fit_transform_selector_classification(
                df_train, y_train, sel_code, n_feats, ds_name, modality_name
            )
            df_val = df_mod.loc[:, id_val].transpose().apply(pd.to_numeric, errors='coerce').fillna(0)
            X_va   = transform_selector_classification(df_val, chosen_cols)
            train_list.append(np.array(X_tr))
            val_list.append(np.array(X_va))

        for merge_str in ["concat", "average", "sum", "max"]:
            try:
                X_tr_m = merge_modalities(*train_list, strategy=merge_str)
                X_va_m = merge_modalities(*val_list,   strategy=merge_str)
            except Exception as e:
                print(f"Skipping merge '{merge_str}' fold {fold_idx}: {e}")
                continue

            # Create model instances from model names
            model_dict = {}
            for model_name in clf_models:
                if model_name == "LogisticRegression":
                    model_dict[model_name] = LogisticRegression(penalty='l2', solver='liblinear', random_state=0)
                elif model_name == "RandomForest":
                    model_dict[model_name] = RandomForestClassifier(n_estimators=100, random_state=0)
                elif model_name == "SVC":
                    model_dict[model_name] = SVC(kernel='rbf', probability=True, random_state=0)
                else:
                    raise ValueError(f"Unknown classification model {model_name}")

            for model_name, model in model_dict.items():
                model, mets = train_classification_model(
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
            chosen_cols, X_tr = cached_fit_transform_selector_classification(
                df_train, y_temp, sel_code, n_feats, ds_name, modality_name
            )
            df_test = df_mod.loc[:, id_test].transpose().apply(pd.to_numeric, errors='coerce').fillna(0)
            X_te    = transform_selector_classification(df_test, chosen_cols)
            train_list.append(np.array(X_tr))
            test_list.append(np.array(X_te))

        X_tr_m = merge_modalities(*train_list, strategy=merge_str)
        X_te_m = merge_modalities(*test_list,  strategy=merge_str)

        # Create model for final evaluation
        final_model, test_mets = train_classification_model(
            X_tr_m, y_temp, X_te_m, y_test,
            model_name,
            out_dir=os.path.join(base_out, "plots"),
            plot_prefix=f"{ds_name}_FINAL_SELECT_{sel_name}_{n_feats}_{merge_str}_{model_name}"
        )
        avg_mets.update({f"Test_{k}": v for k, v in test_mets.items()})
        avg_cv_results.append(avg_mets)

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
def main():
    # Parameters for cross-validation splits
    TEST_SIZE = 0.2   # Hold-out test set fraction
    N_SPLITS = 3      # Number of CV folds

    # 1) REGRESSION block
    reg_extractors = get_regression_extractors()
    reg_selectors  = get_regression_selectors()
    reg_models     = ["LinearRegression", "RandomForest", "SVR"]
    n_comps_list   = [256, 512, MAX_COMPONENTS]  # Reduced number of components
    n_feats_list   = [256, 512, MAX_FEATURES]    # Reduced number of features

    n_extract_runs = (
        len(REGRESSION_DATASETS) * len(reg_extractors) * len(n_comps_list)
    )
    n_select_runs = (
        len(REGRESSION_DATASETS) * len(reg_selectors) * len(n_feats_list)
    )
    reg_total_runs = n_extract_runs + n_select_runs
    progress_count_reg = [0]

    print("=== REGRESSION BLOCK (AML, Sarcoma) ===")
    for ds_conf in REGRESSION_DATASETS:
        ds_name = ds_conf["name"]
        base_out = os.path.join("output_regression", ds_name)
        os.makedirs(base_out, exist_ok=True)
        os.makedirs(os.path.join(base_out, "models"), exist_ok=True)
        os.makedirs(os.path.join(base_out, "metrics"), exist_ok=True)
        os.makedirs(os.path.join(base_out, "plots"), exist_ok=True)

        print(f"\n--- Processing {ds_name} (Regression) ---")

        # load
        exp_df, methy_df, mirna_df, clinical_df = load_omics_and_clinical(ds_conf)
        # prepare
        try:
            data_modalities, common_ids, y, clin_f = prepare_data(
                ds_conf, exp_df, methy_df, mirna_df, is_regression=True
            )
        except ValueError as e:
            print(f"Skipping {ds_name} => {e}")
            continue

        if len(common_ids) == 0 or y.shape[0] == 0:
            print(f"No overlapping or no valid samples => skipping {ds_name}")
            continue

        # A) Extraction with CV
        extraction_jobs = [
            delayed(process_reg_extraction_combo_cv)(
                ds_name, extr_name, extr_obj, nc,
                reg_models, data_modalities, common_ids, y, base_out,
                progress_count_reg, reg_total_runs, test_size=TEST_SIZE, n_splits=N_SPLITS
            )
            for extr_name, extr_obj in reg_extractors.items()
            for nc in n_comps_list
        ]

        all_extraction_results = Parallel(**JOBLIB_PARALLEL_CONFIG)(extraction_jobs)

        # B) Selection with CV
        selection_jobs = [
            delayed(process_reg_selection_combo_cv)(
                ds_name, sel_name, sel_code, nf,
                reg_models, data_modalities, common_ids, y, base_out,
                progress_count_reg, reg_total_runs, test_size=TEST_SIZE, n_splits=N_SPLITS
            )
            for sel_name, sel_code in reg_selectors.items()
            for nf in n_feats_list
        ]

        all_selection_results = Parallel(**JOBLIB_PARALLEL_CONFIG)(selection_jobs)

    # 2) CLASSIFICATION block
    clf_extractors = get_classification_extractors()
    clf_selectors  = get_classification_selectors()
    clf_models     = ["LogisticRegression", "RandomForest", "SVC"]
    n_comps_list_clf = [256, 512, MAX_COMPONENTS]  # Reduced number of components
    n_feats_list_clf = [256, 512, MAX_FEATURES]    # Reduced number of features

    n_extract_runs_clf = (
        len(CLASSIFICATION_DATASETS) * len(clf_extractors) * len(n_comps_list_clf)
    )
    n_select_runs_clf = (
        len(CLASSIFICATION_DATASETS) * len(clf_selectors) * len(n_feats_list_clf)
    )
    clf_total_runs = n_extract_runs_clf + n_select_runs_clf
    progress_count_clf = [0]

    print("\n=== CLASSIFICATION BLOCK (Breast, Colon, Kidney, etc.) ===")
    for ds_conf in CLASSIFICATION_DATASETS:
        ds_name = ds_conf["name"]
        base_out = os.path.join("output_classification", ds_name)
        os.makedirs(base_out, exist_ok=True)
        os.makedirs(os.path.join(base_out, "models"), exist_ok=True)
        os.makedirs(os.path.join(base_out, "metrics"), exist_ok=True)
        os.makedirs(os.path.join(base_out, "plots"), exist_ok=True)

        print(f"\n--- Processing {ds_name} (Classification) ---")

        # load
        exp_df, methy_df, mirna_df, clinical_df = load_omics_and_clinical(ds_conf)
        # prepare
        try:
            data_modalities, common_ids, y, clin_f = prepare_data(
                ds_conf, exp_df, methy_df, mirna_df, is_regression=False
            )
        except ValueError as e:
            print(f"Skipping {ds_name} => {e}")
            continue

        if len(common_ids) == 0 or y.shape[0] == 0:
            print(f"No overlapping or no valid samples => skipping {ds_name}")
            continue

        # A) Extraction with CV
        extraction_jobs = [
            delayed(process_clf_extraction_combo_cv)(
                ds_name, extr_name, extr_obj, nc,
                clf_models, data_modalities, common_ids, y, base_out,
                progress_count_clf, clf_total_runs, test_size=TEST_SIZE, n_splits=N_SPLITS
            )
            for extr_name, extr_obj in clf_extractors.items()
            for nc in n_comps_list_clf
        ]

        all_extraction_results = Parallel(**JOBLIB_PARALLEL_CONFIG)(extraction_jobs)

        # B) Selection with CV
        selection_jobs = [
            delayed(process_clf_selection_combo_cv)(
                ds_name, sel_name, sel_code, nf,
                clf_models, data_modalities, common_ids, y, base_out,
                progress_count_clf, clf_total_runs, test_size=TEST_SIZE, n_splits=N_SPLITS
            )
            for sel_name, sel_code in clf_selectors.items()
            for nf in n_feats_list_clf
        ]

        all_selection_results = Parallel(**JOBLIB_PARALLEL_CONFIG)(selection_jobs)

    print("\nAll done! Regression outputs in 'output_regression/' and classification outputs in 'output_classification/'.")

# Cache helpers – smaller keys (SHA1 of columns)
class _Cache:
    def __init__(self):
        self._d = {}

    def get(self, k):
        return self._d.get(k)

    def set(self, k, v):
        self._d[k] = v

def _cols_hash(cols):
    return hashlib.sha1(",".join(map(str, cols)).encode()).hexdigest()

# Create global cache instances
extractor_regression_cache = _Cache()
selector_regression_cache = _Cache()
extractor_classification_cache = _Cache()
selector_classification_cache = _Cache()

def cached_fit_transform_extractor_regression(X_train, y_train, extractor, n_components, dataset_name, modality_name):
    # Create a new instance of the extractor for each modality
    extractor = type(extractor)(**extractor.get_params())
    # Include dataset name, modality name, and number of features in the cache key
    key = (id(extractor), n_components, X_train.shape[1], dataset_name, modality_name, _cols_hash(X_train.columns), _cols_hash(y_train))
    cached_result = extractor_regression_cache.get(key)
    if cached_result is not None:
        return cached_result
    result = fit_transform_extractor_regression(X_train, y_train, extractor, n_components)
    extractor_regression_cache.set(key, result)
    return result

def cached_fit_transform_selector_regression(X_train, y_train, selector_code, n_feats, dataset_name, modality_name):
    # Include dataset name, modality name, and number of features in the cache key
    key = (selector_code, n_feats, X_train.shape[1], dataset_name, modality_name, _cols_hash(X_train.columns), _cols_hash(y_train))
    cached_result = selector_regression_cache.get(key)
    if cached_result is not None:
        return cached_result
    result = fit_transform_selector_regression(X_train, y_train, selector_code, n_feats)
    selector_regression_cache.set(key, result)
    return result

def cached_fit_transform_extractor_classification(X_train, y_train, extractor, n_components, dataset_name, modality_name):
    # Create a new instance of the extractor for each modality
    extractor = type(extractor)(**extractor.get_params())
    # Include dataset name, modality name, and number of features in the cache key
    key = (id(extractor), n_components, X_train.shape[1], dataset_name, modality_name, _cols_hash(X_train.columns), _cols_hash(y_train))
    cached_result = extractor_classification_cache.get(key)
    if cached_result is not None:
        return cached_result
    result = fit_transform_extractor_classification(X_train, y_train, extractor, n_components)
    extractor_classification_cache.set(key, result)
    return result

def cached_fit_transform_selector_classification(X_train, y_train, selector_code, n_feats, dataset_name, modality_name):
    # Include dataset name, modality name, and number of features in the cache key
    key = (selector_code, n_feats, X_train.shape[1], dataset_name, modality_name, _cols_hash(X_train.columns), _cols_hash(y_train))
    cached_result = selector_classification_cache.get(key)
    if cached_result is not None:
        return cached_result
    result = fit_transform_selector_classification(X_train, y_train, selector_code, n_feats)
    selector_classification_cache.set(key, result)
    return result

###############################################################################
# M) MAIN
###############################################################################
if __name__ == "__main__":
    main()