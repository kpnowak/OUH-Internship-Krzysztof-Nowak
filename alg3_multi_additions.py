#!/usr/bin/env python3

import os
import time
import joblib
import numpy as np
import pandas as pd

# For parallelization
from joblib import Parallel, delayed

# Regression models
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC

# Dimensionality Reduction (for regression)
from sklearn.decomposition import PCA, NMF, FastICA, FactorAnalysis
from sklearn.cross_decomposition import PLSRegression

# Dimensionality Reduction (for classification)
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Data scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Splits & regression metrics
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

# Classification metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, matthews_corrcoef, RocCurveDisplay
)

# Feature selection
from sklearn.feature_selection import (
    mutual_info_regression, f_regression,
    mutual_info_classif, f_classif, chi2, SelectKBest
)
from boruta import BorutaPy

# For boruta, we need separate (regressor/classifier) random forests
from sklearn.ensemble import RandomForestRegressor as RF_for_BorutaReg
from sklearn.ensemble import RandomForestClassifier as RF_for_BorutaClf

# For plotting
import matplotlib.pyplot as plt
import seaborn as sns

###############################################################################
# A) CONFIG OF DATASETS
###############################################################################

REGRESSION_DATASETS = [
    {
        "name": "Sarcoma",
        "clinical_file": "clinical/sarcoma.csv",
        "omics_dir": "sarcoma",
        "id_col": "metsampleID",
        "outcome_col": "pathologic_tumor_length",
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
        "name": "Liver",
        "clinical_file": "clinical/liver.csv",
        "omics_dir": "liver",
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

def load_omics_and_clinical(ds_config):
    odir = ds_config["omics_dir"]
    # Read the omics CSV files without specifying an index column,
    # so that the first row is used as the header.
    exp_df   = pd.read_csv(os.path.join(odir, "exp.csv"), sep=None, engine='python')
    methy_df = pd.read_csv(os.path.join(odir, "methy.csv"), sep=None, engine='python')
    mirna_df = pd.read_csv(os.path.join(odir, "mirna.csv"), sep=None, engine='python')

    # Delete the first column from each DataFrame.
    # The header row remains unchanged.
    exp_df   = exp_df.iloc[:, 1:]
    methy_df = methy_df.iloc[:, 1:]
    mirna_df = mirna_df.iloc[:, 1:]

    clinical_df = pd.read_csv(ds_config["clinical_file"], sep=None, engine='python')
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
    exp_df.columns   = strip_and_slice_columns(exp_df.columns)
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
# D) EXTRACTORS & SELECTORS
###############################################################################

# For regression
def get_regression_extractors():
    return {
        "PCA": PCA(),
        "NMF": NMF(max_iter=10000, init='nndsvda'),
        "ICA": FastICA(max_iter=10000, tol=1e-2),
        "FA": FactorAnalysis(),
        "PLS": PLSRegression()
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
    plt.figure(figsize=(5,5))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.title(title + ": Actual vs. Predicted")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    mn = min(min(y_test), min(y_pred))
    mx = max(max(y_test), max(y_pred))
    plt.plot([mn, mx], [mn, mx], 'r--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_regression_residuals(y_test, y_pred, title, out_path):
    residuals = y_test - y_pred
    plt.figure(figsize=(5,5))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color='r', linestyle='--')
    plt.title(title + ": Residual Plot")
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_confusion_matrix(cm, class_labels, title, out_path):
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt='d',
                cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_roc_curve_binary(model, X_test, y_test, class_labels, title, out_path):
    y_proba = model.predict_proba(X_test)[:, 1]
    disp = RocCurveDisplay.from_predictions(y_test, y_proba, name='Binary ROC')
    disp.ax_.set_title(title + " - ROC Curve")
    plt.savefig(out_path)
    plt.close()

###############################################################################
# F) TRAIN & EVAL: REGRESSION
###############################################################################

def train_regression_model(X_train, y_train, X_test, y_test,
                           model_name, out_dir=None, plot_prefix=""):
    if model_name=="LinearRegression":
        model = LinearRegression()
    elif model_name=="RandomForest":
        model = RandomForestRegressor(n_estimators=100, random_state=0)
    elif model_name=="SVR":
        model = SVR(kernel='rbf')
    else:
        raise ValueError(f"Unknown regression model {model_name}")

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)

    if out_dir and plot_prefix:
        scatter_path = os.path.join(out_dir, f"{plot_prefix}_scatter.png")
        plot_regression_scatter(y_test, y_pred, plot_prefix, scatter_path)

        resid_path = os.path.join(out_dir, f"{plot_prefix}_residuals.png")
        plot_regression_residuals(y_test, y_pred, plot_prefix, resid_path)

    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "Train_Time_Seconds": train_time
    }
    return model, metrics

###############################################################################
# G) TRAIN & EVAL: CLASSIFICATION
###############################################################################

def train_classification_model(X_train, y_train, X_test, y_test,
                               model_name, out_dir=None, plot_prefix=""):
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

###############################################################################
# H) FEATURE EXTRACTION/SELECTION FIT+TRANSFORM UTILS (REGRESSION)
###############################################################################

def fit_transform_extractor_regression(X_train, y_train, extractor, n_components):
    # Some extractors (NMF) want non-negative => scale with MinMax
    if extractor.__class__.__name__ == "NMF":
        scl = MinMaxScaler(clip=True)
    else:
        scl = StandardScaler()

    X_train_scl = scl.fit_transform(X_train)

    if hasattr(extractor, "random_state"):
        extractor.random_state = 0
    if hasattr(extractor, "n_components"):
        extractor.n_components = n_components

    # For PLS => pass y
    if isinstance(extractor, PLSRegression):
        Y_train_arr = y_train.values.reshape(-1, 1)
        X_train_red = extractor.fit_transform(X_train_scl, Y_train_arr)[0]
    else:
        X_train_red = extractor.fit_transform(X_train_scl)

    fitted_extractor = {
        "scaler": scl,
        "extractor": extractor
    }
    return fitted_extractor, X_train_red

def transform_extractor_regression(X_test, fitted_extractor):
    scl = fitted_extractor["scaler"]
    extractor = fitted_extractor["extractor"]
    X_test_scl = scl.transform(X_test)
    if isinstance(extractor, PLSRegression):
        X_test_red = extractor.transform(X_test_scl)
    else:
        X_test_red = extractor.transform(X_test_scl)
    return X_test_red

def fit_transform_selector_regression(X_train, y_train, selector_code, n_feats):
    if selector_code == "mrmr_reg":
        mi = mutual_info_regression(X_train, y_train, random_state=0)
        idx = np.argsort(mi)[::-1]  # descending
        top_idx = idx[:n_feats]
        return list(top_idx), X_train.iloc[:, top_idx]

    elif selector_code == "lasso":
        lasso = Lasso(alpha=0.01, max_iter=10000, random_state=0)
        lasso.fit(X_train, y_train)
        coefs = lasso.coef_
        idx = np.argsort(np.abs(coefs))[::-1]
        top_idx = idx[:n_feats]
        return list(top_idx), X_train.iloc[:, top_idx]

    elif selector_code == "enet":
        en = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000, random_state=0)
        en.fit(X_train, y_train)
        c = en.coef_
        idx = np.argsort(np.abs(c))[::-1]
        top_idx = idx[:n_feats]
        return list(top_idx), X_train.iloc[:, top_idx]

    elif selector_code == "freg":
        Fv, pv = f_regression(X_train, y_train)
        idx = np.argsort(Fv)[::-1]
        top_idx = idx[:n_feats]
        return list(top_idx), X_train.iloc[:, top_idx]

    elif selector_code == "boruta_reg":
        rf = RF_for_BorutaReg(n_estimators=100, random_state=0)
        bor = BorutaPy(rf, n_estimators='auto', random_state=0)
        bor.fit(X_train.values, y_train.values)
        mask = bor.support_
        chosen = np.where(mask)[0]
        if len(chosen) > n_feats:
            ranks = bor.ranking_
            chosen_ranks = sorted(zip(chosen, ranks[chosen]), key=lambda x: x[1])
            chosen = [x[0] for x in chosen_ranks[:n_feats]]
        return list(chosen), X_train.iloc[:, chosen]

    else:
        # fallback => no selection
        return list(range(X_train.shape[1])), X_train

def transform_selector_regression(X_test, chosen_cols):
    return X_test.iloc[:, chosen_cols]

###############################################################################
# I) FEATURE EXTRACTION/SELECTION FIT+TRANSFORM UTILS (CLASSIFICATION)
###############################################################################

def fit_transform_extractor_classification(X_train, y_train, extractor, n_components):
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
    scl = fitted_extractor["scaler"]
    extractor = fitted_extractor["extractor"]
    X_test_scl = scl.transform(X_test)
    X_test_red = extractor.transform(X_test_scl)
    return X_test_red

def fit_transform_selector_classification(X_train, y_train, selector_code, n_feats):
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
    return X_test.iloc[:, chosen_cols]

###############################################################################
# J) MERGING STRATEGIES
###############################################################################

def pad_to_shape(arr, target_cols):
    """Pads array 'arr' with zeros on the right to reach 'target_cols' columns."""
    current_cols = arr.shape[1]
    if current_cols < target_cols:
        pad_width = target_cols - current_cols
        return np.pad(arr, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
    return arr

def merge_modalities(mod1, mod2, mod3, strategy="concat"):
    """
    Merge three numpy arrays (same # of rows).
      - 'concat'  => column-wise concatenation (works with mismatched shapes)
      - 'average' => element-wise average (requires same shape; pads if needed)
      - 'sum'     => element-wise sum (requires same shape; pads if needed)
      - 'max'     => element-wise max (requires same shape; pads if needed)
    """
    if strategy == "concat":
        return np.concatenate([mod1, mod2, mod3], axis=1)
    else:
        # For element-wise operations, determine the maximum number of columns among the three arrays.
        target_cols = max(mod1.shape[1], mod2.shape[1], mod3.shape[1])
        mod1_p = pad_to_shape(mod1, target_cols)
        mod2_p = pad_to_shape(mod2, target_cols)
        mod3_p = pad_to_shape(mod3, target_cols)
        
        if strategy == "average":
            return (mod1_p + mod2_p + mod3_p) / 3.0
        elif strategy == "sum":
            return mod1_p + mod2_p + mod3_p
        elif strategy == "max":
            return np.maximum(mod1_p, np.maximum(mod2_p, mod3_p))
        else:
            raise ValueError(f"Unknown merging strategy {strategy}")

###############################################################################
# K) HIGH-LEVEL PROCESS FUNCTIONS (REGRESSION) WITH CROSS-VALIDATION
###############################################################################

def process_reg_extraction_combo_cv(
    ds_name, extr_name, extr_obj, ncomps, reg_models,
    data_modalities, all_ids, y, base_out,
    progress_count, reg_total_runs, test_size=0.2, n_splits=5
):
    progress_count[0] += 1
    run_idx = progress_count[0]
    print(f"[EXTRACT-REG CV] {run_idx}/{reg_total_runs} => {ds_name} | {extr_name}-{ncomps}")

    all_ids_arr = np.array(all_ids)
    y_arr = np.array(y)

    # Outer split: hold-out test set and temporary set for CV
    id_temp, id_test, y_temp, y_test = train_test_split(
        all_ids_arr, y_arr, test_size=test_size, random_state=0
    )

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    cv_metrics = {}  # key: (merge_strategy, model_name) -> list of metric dicts

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(id_temp)):
        id_train = id_temp[train_idx]
        id_val   = id_temp[val_idx]
        y_train = y_temp[train_idx]
        y_val   = y_temp[val_idx]

        train_transformed_list = []
        val_transformed_list = []
        for mod_name, df_mod in data_modalities.items():
            # Process training set for this fold
            df_train = df_mod.loc[:, id_train].transpose()
            df_train = df_train.apply(pd.to_numeric, errors='coerce').fillna(0)
            fitted_extr, X_train_mod = fit_transform_extractor_regression(
                df_train, pd.Series(y_train), extr_obj, ncomps
            )
            # Process validation set for this fold
            df_val = df_mod.loc[:, id_val].transpose()
            df_val = df_val.apply(pd.to_numeric, errors='coerce').fillna(0)
            X_val_mod = transform_extractor_regression(df_val, fitted_extr)
            train_transformed_list.append(X_train_mod)
            val_transformed_list.append(X_val_mod)

        if len(train_transformed_list)==0:
            continue

        merging_strategies = ["concat", "average", "sum", "max"]
        for merge_str in merging_strategies:
            try:
                X_train_merged = merge_modalities(
                    train_transformed_list[0],
                    train_transformed_list[1],
                    train_transformed_list[2],
                    merge_str
                )
                X_val_merged = merge_modalities(
                    val_transformed_list[0],
                    val_transformed_list[1],
                    val_transformed_list[2],
                    merge_str
                )
            except Exception as e:
                print(f"Skipping merging strategy '{merge_str}' in fold {fold_idx} due to error: {e}")
                continue

            for model_name in reg_models:
                plot_prefix = f"{ds_name}_EXTRACT_{extr_name}_{ncomps}_{merge_str}_{model_name}_fold{fold_idx}"
                model, mets = train_regression_model(
                    X_train_merged, y_train, X_val_merged, y_val,
                    model_name,
                    out_dir=os.path.join(base_out, "plots"),
                    plot_prefix=plot_prefix
                )
                key = (merge_str, model_name)
                cv_metrics.setdefault(key, []).append(mets)

    # Average the CV metrics over folds and perform final evaluation on the hold-out test set
    avg_cv_results = []
    for (merge_str, model_name), mets_list in cv_metrics.items():
        avg_mets = {k: np.mean([m[k] for m in mets_list]) for k in mets_list[0].keys()}
        avg_mets.update({
            "Dataset": ds_name,
            "Workflow": "Extraction-CV",
            "Extractor": extr_name,
            "n_components": ncomps,
            "MergeStrategy": merge_str,
            "Model": model_name
        })
        # Final evaluation on hold-out test set using full temp set (id_temp)
        train_transformed_list = []
        test_transformed_list = []
        for mod_name, df_mod in data_modalities.items():
            df_train = df_mod.loc[:, id_temp].transpose()
            df_train = df_train.apply(pd.to_numeric, errors='coerce').fillna(0)
            fitted_extr, X_train_mod = fit_transform_extractor_regression(
                df_train, pd.Series(y_temp), extr_obj, ncomps
            )
            df_test = df_mod.loc[:, id_test].transpose()
            df_test = df_test.apply(pd.to_numeric, errors='coerce').fillna(0)
            X_test_mod = transform_extractor_regression(df_test, fitted_extr)
            train_transformed_list.append(X_train_mod)
            test_transformed_list.append(X_test_mod)
        try:
            X_train_merged = merge_modalities(
                train_transformed_list[0],
                train_transformed_list[1],
                train_transformed_list[2],
                merge_str
            )
            X_test_merged = merge_modalities(
                test_transformed_list[0],
                test_transformed_list[1],
                test_transformed_list[2],
                merge_str
            )
        except Exception as e:
            print(f"Skipping final test evaluation for merging strategy '{merge_str}' due to error: {e}")
            continue

        final_model, test_mets = train_regression_model(
            X_train_merged, y_temp, X_test_merged, y_test,
            model_name,
            out_dir=os.path.join(base_out, "plots"),
            plot_prefix=f"{ds_name}_FINAL_EXTRACT_{extr_name}_{ncomps}_{merge_str}_{model_name}"
        )
        for k, v in test_mets.items():
            avg_mets[f"Test_{k}"] = v
        avg_cv_results.append(avg_mets)

        # Optionally save the final model
        mfname = f"{ds_name}_FINAL_EXTRACT_{extr_name}_{ncomps}_{merge_str}_{model_name}.pkl"
        mp = os.path.join(base_out, "models", mfname)
        joblib.dump(final_model, mp)

    df_avg = pd.DataFrame(avg_cv_results)
    metrics_file = os.path.join(base_out, "metrics", f"{ds_name}_extraction_cv_metrics.csv")
    df_avg.to_csv(metrics_file, index=False)

    return avg_cv_results

def process_reg_selection_combo_cv(
    ds_name, sel_name, sel_code, n_feats, reg_models,
    data_modalities, all_ids, y, base_out,
    progress_count, reg_total_runs, test_size=0.2, n_splits=5
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

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(id_temp)):
        id_train = id_temp[train_idx]
        id_val   = id_temp[val_idx]
        y_train = y_temp[train_idx]
        y_val   = y_temp[val_idx]

        train_transformed_list = []
        val_transformed_list = []
        for mod_name, df_mod in data_modalities.items():
            df_train = df_mod.loc[:, id_train].transpose()
            df_train = df_train.apply(pd.to_numeric, errors='coerce').fillna(0)
            chosen_cols, X_train_mod = fit_transform_selector_regression(
                df_train, pd.Series(y_train), sel_code, n_feats
            )
            df_val = df_mod.loc[:, id_val].transpose()
            df_val = df_val.apply(pd.to_numeric, errors='coerce').fillna(0)
            X_val_mod = transform_selector_regression(df_val, chosen_cols)
            train_transformed_list.append(np.array(X_train_mod))
            val_transformed_list.append(np.array(X_val_mod))

        merging_strategies = ["concat", "average", "sum", "max"]
        for merge_str in merging_strategies:
            try:
                X_train_merged = merge_modalities(
                    train_transformed_list[0],
                    train_transformed_list[1],
                    train_transformed_list[2],
                    merge_str
                )
                X_val_merged = merge_modalities(
                    val_transformed_list[0],
                    val_transformed_list[1],
                    val_transformed_list[2],
                    merge_str
                )
            except Exception as e:
                print(f"Skipping merging strategy '{merge_str}' in fold {fold_idx} due to error: {e}")
                continue

            for model_name in reg_models:
                plot_prefix = f"{ds_name}_SELECT_{sel_name}_{n_feats}_{merge_str}_{model_name}_fold{fold_idx}"
                model, mets = train_regression_model(
                    X_train_merged, y_train, X_val_merged, y_val,
                    model_name,
                    out_dir=os.path.join(base_out, "plots"),
                    plot_prefix=plot_prefix
                )
                key = (merge_str, model_name)
                cv_metrics.setdefault(key, []).append(mets)

    avg_cv_results = []
    for (merge_str, model_name), mets_list in cv_metrics.items():
        avg_mets = {k: np.mean([m[k] for m in mets_list]) for k in mets_list[0].keys()}
        avg_mets.update({
            "Dataset": ds_name,
            "Workflow": "Selection-CV",
            "Selector": sel_name,
            "n_features": n_feats,
            "MergeStrategy": merge_str,
            "Model": model_name
        })
        # Final evaluation on hold-out test set
        train_transformed_list = []
        test_transformed_list = []
        for mod_name, df_mod in data_modalities.items():
            df_train = df_mod.loc[:, id_temp].transpose()
            df_train = df_train.apply(pd.to_numeric, errors='coerce').fillna(0)
            chosen_cols, X_train_mod = fit_transform_selector_regression(
                df_train, pd.Series(y_temp), sel_code, n_feats
            )
            df_test = df_mod.loc[:, id_test].transpose()
            df_test = df_test.apply(pd.to_numeric, errors='coerce').fillna(0)
            X_test_mod = transform_selector_regression(df_test, chosen_cols)
            train_transformed_list.append(np.array(X_train_mod))
            test_transformed_list.append(np.array(X_test_mod))
        try:
            X_train_merged = merge_modalities(
                train_transformed_list[0],
                train_transformed_list[1],
                train_transformed_list[2],
                merge_str
            )
            X_test_merged = merge_modalities(
                test_transformed_list[0],
                test_transformed_list[1],
                test_transformed_list[2],
                merge_str
            )
        except Exception as e:
            print(f"Skipping final test evaluation for merging strategy '{merge_str}' due to error: {e}")
            continue

        final_model, test_mets = train_regression_model(
            X_train_merged, y_temp, X_test_merged, y_test,
            model_name,
            out_dir=os.path.join(base_out, "plots"),
            plot_prefix=f"{ds_name}_FINAL_SELECT_{sel_name}_{n_feats}_{merge_str}_{model_name}"
        )
        for k, v in test_mets.items():
            avg_mets[f"Test_{k}"] = v
        avg_cv_results.append(avg_mets)

        mfname = f"{ds_name}_FINAL_SELECT_{sel_name}_{n_feats}_{merge_str}_{model_name}.pkl"
        mp = os.path.join(base_out, "models", mfname)
        joblib.dump(final_model, mp)

    df_avg = pd.DataFrame(avg_cv_results)
    metrics_file = os.path.join(base_out, "metrics", f"{ds_name}_selection_cv_metrics.csv")
    df_avg.to_csv(metrics_file, index=False)

    return avg_cv_results

###############################################################################
# L) HIGH-LEVEL PROCESS FUNCTIONS (CLASSIFICATION) WITH CROSS-VALIDATION
###############################################################################

def process_clf_extraction_combo_cv(
    ds_name, extr_name, extr_obj, ncomps, clf_models,
    data_modalities, all_ids, y, base_out,
    progress_count, clf_total_runs, test_size=0.2, n_splits=5
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
        id_train = id_temp[train_idx]
        id_val   = id_temp[val_idx]
        y_train = y_temp[train_idx]
        y_val   = y_temp[val_idx]

        train_transformed_list = []
        val_transformed_list = []
        for mod_name, df_mod in data_modalities.items():
            df_train = df_mod.loc[:, id_train].transpose()
            df_train = df_train.apply(pd.to_numeric, errors='coerce').fillna(0)
            fitted_extr, X_train_mod = fit_transform_extractor_classification(
                df_train, y_train, extr_obj, ncomps
            )
            df_val = df_mod.loc[:, id_val].transpose()
            df_val = df_val.apply(pd.to_numeric, errors='coerce').fillna(0)
            X_val_mod = transform_extractor_classification(df_val, fitted_extr)
            train_transformed_list.append(X_train_mod)
            val_transformed_list.append(X_val_mod)

        if len(train_transformed_list)==0:
            continue

        merging_strategies = ["concat", "average", "sum", "max"]
        for merge_str in merging_strategies:
            try:
                X_train_merged = merge_modalities(
                    train_transformed_list[0],
                    train_transformed_list[1],
                    train_transformed_list[2],
                    merge_str
                )
                X_val_merged = merge_modalities(
                    val_transformed_list[0],
                    val_transformed_list[1],
                    val_transformed_list[2],
                    merge_str
                )
            except Exception as e:
                print(f"Skipping merging strategy '{merge_str}' in fold {fold_idx} due to error: {e}")
                continue

            for model_name in clf_models:
                plot_prefix = f"{ds_name}_EXTRACT_{extr_name}_{ncomps}_{merge_str}_{model_name}_fold{fold_idx}"
                model, mets = train_classification_model(
                    X_train_merged, y_train, X_val_merged, y_val,
                    model_name,
                    out_dir=os.path.join(base_out, "plots"),
                    plot_prefix=plot_prefix
                )
                key = (merge_str, model_name)
                cv_metrics.setdefault(key, []).append(mets)

    avg_cv_results = []
    for (merge_str, model_name), mets_list in cv_metrics.items():
        avg_mets = {k: np.mean([m[k] for m in mets_list]) for k in mets_list[0].keys()}
        avg_mets.update({
            "Dataset": ds_name,
            "Workflow": "Extraction-CV",
            "Extractor": extr_name,
            "n_components": ncomps,
            "MergeStrategy": merge_str,
            "Model": model_name
        })
        # Final evaluation on hold-out test set
        train_transformed_list = []
        test_transformed_list = []
        for mod_name, df_mod in data_modalities.items():
            df_train = df_mod.loc[:, id_temp].transpose()
            df_train = df_train.apply(pd.to_numeric, errors='coerce').fillna(0)
            fitted_extr, X_train_mod = fit_transform_extractor_classification(
                df_train, y_temp, extr_obj, ncomps
            )
            df_test = df_mod.loc[:, id_test].transpose()
            df_test = df_test.apply(pd.to_numeric, errors='coerce').fillna(0)
            X_test_mod = transform_extractor_classification(df_test, fitted_extr)
            train_transformed_list.append(X_train_mod)
            test_transformed_list.append(X_test_mod)
        try:
            X_train_merged = merge_modalities(
                train_transformed_list[0],
                train_transformed_list[1],
                train_transformed_list[2],
                merge_str
            )
            X_test_merged = merge_modalities(
                test_transformed_list[0],
                test_transformed_list[1],
                test_transformed_list[2],
                merge_str
            )
        except Exception as e:
            print(f"Skipping final test evaluation for merging strategy '{merge_str}' due to error: {e}")
            continue

        final_model, test_mets = train_classification_model(
            X_train_merged, y_temp, X_test_merged, y_test,
            model_name,
            out_dir=os.path.join(base_out, "plots"),
            plot_prefix=f"{ds_name}_FINAL_EXTRACT_{extr_name}_{ncomps}_{merge_str}_{model_name}"
        )
        for k, v in test_mets.items():
            avg_mets[f"Test_{k}"] = v
        avg_cv_results.append(avg_mets)

        mfname = f"{ds_name}_FINAL_EXTRACT_{extr_name}_{ncomps}_{merge_str}_{model_name}.pkl"
        mp = os.path.join(base_out, "models", mfname)
        joblib.dump(final_model, mp)

    df_avg = pd.DataFrame(avg_cv_results)
    metrics_file = os.path.join(base_out, "metrics", f"{ds_name}_extraction_cv_metrics.csv")
    df_avg.to_csv(metrics_file, index=False)

    return avg_cv_results

def process_clf_selection_combo_cv(
    ds_name, sel_name, sel_code, n_feats, clf_models,
    data_modalities, all_ids, y, base_out,
    progress_count, clf_total_runs, test_size=0.2, n_splits=5
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
        id_train = id_temp[train_idx]
        id_val   = id_temp[val_idx]
        y_train = y_temp[train_idx]
        y_val   = y_temp[val_idx]

        train_transformed_list = []
        val_transformed_list = []
        for mod_name, df_mod in data_modalities.items():
            df_train = df_mod.loc[:, id_train].transpose()
            df_train = df_train.apply(pd.to_numeric, errors='coerce').fillna(0)
            chosen_cols, X_train_mod = fit_transform_selector_classification(
                df_train, y_train, sel_code, n_feats
            )
            df_val = df_mod.loc[:, id_val].transpose()
            df_val = df_val.apply(pd.to_numeric, errors='coerce').fillna(0)
            X_val_mod = transform_selector_classification(df_val, chosen_cols)
            train_transformed_list.append(np.array(X_train_mod))
            val_transformed_list.append(np.array(X_val_mod))
        merging_strategies = ["concat", "average", "sum", "max"]
        for merge_str in merging_strategies:
            try:
                X_train_merged = merge_modalities(
                    train_transformed_list[0],
                    train_transformed_list[1],
                    train_transformed_list[2],
                    merge_str
                )
                X_val_merged = merge_modalities(
                    val_transformed_list[0],
                    val_transformed_list[1],
                    val_transformed_list[2],
                    merge_str
                )
            except Exception as e:
                print(f"Skipping merging strategy '{merge_str}' in fold {fold_idx} due to error: {e}")
                continue

            for model_name in clf_models:
                plot_prefix = f"{ds_name}_SELECT_{sel_name}_{n_feats}_{merge_str}_{model_name}_fold{fold_idx}"
                model, mets = train_classification_model(
                    X_train_merged, y_train, X_val_merged, y_val,
                    model_name,
                    out_dir=os.path.join(base_out, "plots"),
                    plot_prefix=plot_prefix
                )
                key = (merge_str, model_name)
                cv_metrics.setdefault(key, []).append(mets)

    avg_cv_results = []
    for (merge_str, model_name), mets_list in cv_metrics.items():
        avg_mets = {k: np.mean([m[k] for m in mets_list]) for k in mets_list[0].keys()}
        avg_mets.update({
            "Dataset": ds_name,
            "Workflow": "Selection-CV",
            "Selector": sel_name,
            "n_features": n_feats,
            "MergeStrategy": merge_str,
            "Model": model_name
        })
        # Final evaluation on hold-out test set
        train_transformed_list = []
        test_transformed_list = []
        for mod_name, df_mod in data_modalities.items():
            df_train = df_mod.loc[:, id_temp].transpose()
            df_train = df_train.apply(pd.to_numeric, errors='coerce').fillna(0)
            chosen_cols, X_train_mod = fit_transform_selector_classification(
                df_train, y_temp, sel_code, n_feats
            )
            df_test = df_mod.loc[:, id_test].transpose()
            df_test = df_test.apply(pd.to_numeric, errors='coerce').fillna(0)
            X_test_mod = transform_selector_classification(df_test, chosen_cols)
            train_transformed_list.append(np.array(X_train_mod))
            test_transformed_list.append(np.array(X_test_mod))
        try:
            X_train_merged = merge_modalities(
                train_transformed_list[0],
                train_transformed_list[1],
                train_transformed_list[2],
                merge_str
            )
            X_test_merged = merge_modalities(
                test_transformed_list[0],
                test_transformed_list[1],
                test_transformed_list[2],
                merge_str
            )
        except Exception as e:
            print(f"Skipping final test evaluation for merging strategy '{merge_str}' due to error: {e}")
            continue

        final_model, test_mets = train_classification_model(
            X_train_merged, y_temp, X_test_merged, y_test,
            model_name,
            out_dir=os.path.join(base_out, "plots"),
            plot_prefix=f"{ds_name}_FINAL_SELECT_{sel_name}_{n_feats}_{merge_str}_{model_name}"
        )
        for k, v in test_mets.items():
            avg_mets[f"Test_{k}"] = v
        avg_cv_results.append(avg_mets)

        mfname = f"{ds_name}_FINAL_SELECT_{sel_name}_{n_feats}_{merge_str}_{model_name}.pkl"
        mp = os.path.join(base_out, "models", mfname)
        joblib.dump(final_model, mp)

    df_avg = pd.DataFrame(avg_cv_results)
    metrics_file = os.path.join(base_out, "metrics", f"{ds_name}_selection_cv_metrics.csv")
    df_avg.to_csv(metrics_file, index=False)

    return avg_cv_results

###############################################################################
# M) MAIN
###############################################################################
def main():
    # Parameters for cross-validation splits
    TEST_SIZE = 0.2   # Hold-out test set fraction
    N_SPLITS = 5      # Number of CV folds

    # 1) REGRESSION block
    reg_extractors = get_regression_extractors()
    reg_selectors  = get_regression_selectors()
    reg_models     = ["LinearRegression", "RandomForest", "SVR"]
    n_comps_list   = [8,16,32,64,128]
    n_feats_list   = [8,16,32,64,128]

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
        extraction_jobs = (
            delayed(process_reg_extraction_combo_cv)(
                ds_name, extr_name, extr_obj, nc,
                reg_models, data_modalities, common_ids, y, base_out,
                progress_count_reg, reg_total_runs, test_size=TEST_SIZE, n_splits=N_SPLITS
            )
            for extr_name, extr_obj in reg_extractors.items()
            for nc in n_comps_list
        )
        all_extraction_results = Parallel(n_jobs=-1)(extraction_jobs)
        # Flatten results if needed

        # B) Selection with CV
        selection_jobs = (
            delayed(process_reg_selection_combo_cv)(
                ds_name, sel_name, sel_code, nf,
                reg_models, data_modalities, common_ids, y, base_out,
                progress_count_reg, reg_total_runs, test_size=TEST_SIZE, n_splits=N_SPLITS
            )
            for sel_name, sel_code in reg_selectors.items()
            for nf in n_feats_list
        )
        all_selection_results = Parallel(n_jobs=-1)(selection_jobs)

    # 2) CLASSIFICATION block
    clf_extractors = get_classification_extractors()
    clf_selectors  = get_classification_selectors()
    clf_models     = ["LogisticRegression", "RandomForest", "SVC"]
    n_comps_list_clf = [8,16,32,64,128]
    n_feats_list_clf = [8,16,32,64,128]

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
        extraction_jobs = (
            delayed(process_clf_extraction_combo_cv)(
                ds_name, extr_name, extr_obj, nc,
                clf_models, data_modalities, common_ids, y, base_out,
                progress_count_clf, clf_total_runs, test_size=TEST_SIZE, n_splits=N_SPLITS
            )
            for extr_name, extr_obj in clf_extractors.items()
            for nc in n_comps_list_clf
        )
        all_extraction_results = Parallel(n_jobs=-1)(extraction_jobs)

        # B) Selection with CV
        selection_jobs = (
            delayed(process_clf_selection_combo_cv)(
                ds_name, sel_name, sel_code, nf,
                clf_models, data_modalities, common_ids, y, base_out,
                progress_count_clf, clf_total_runs, test_size=TEST_SIZE, n_splits=N_SPLITS
            )
            for sel_name, sel_code in clf_selectors.items()
            for nf in n_feats_list_clf
        )
        all_selection_results = Parallel(n_jobs=-1)(selection_jobs)

    print("\nAll done! Regression outputs in 'output_regression/' and classification outputs in 'output_classification/'.")

if __name__=="__main__":
    main()