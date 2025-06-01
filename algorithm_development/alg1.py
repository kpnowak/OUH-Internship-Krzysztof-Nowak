#!/usr/bin/env python3

import os
import time
import joblib
import numpy as np
import pandas as pd

# Regression models
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Dimensionality Reduction (for regression)
from sklearn.decomposition import PCA, NMF, FastICA, FactorAnalysis
from sklearn.cross_decomposition import PLSRegression

# Dimensionality Reduction (for classification)
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Data scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Splits & regression metrics
from sklearn.model_selection import train_test_split
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
]

CLASSIFICATION_DATASETS = [
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
]

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
    exp_df   = pd.read_csv(os.path.join(odir,"exp.csv"),   sep=None, engine='python', index_col=0)
    methy_df = pd.read_csv(os.path.join(odir,"methy.csv"), sep=None, engine='python', index_col=0)
    mirna_df = pd.read_csv(os.path.join(odir,"mirna.csv"), sep=None, engine='python', index_col=0)

    clinical_df = pd.read_csv(ds_config["clinical_file"], sep=None, engine='python')
    return exp_df, methy_df, mirna_df, clinical_df

def strip_and_slice_columns(col_list):
    newcols=[]
    for c in col_list:
        s2 = c.strip().strip('"')
        s3 = fix_tcga_id_slicing(s2)
        newcols.append(s3)
    return newcols

def prepare_data(ds_config, exp_df, methy_df, mirna_df, is_regression=True):
    id_col = ds_config["id_col"]
    out_col= ds_config["outcome_col"]

    # --- ONLY FOR KIDNEY: strip trailing 'A' from the ID column in memory ---
    if ds_config["name"] == "Kidney":
        ### NEW FOR KIDNEY ###
        # If the last character is 'A', remove it 
        # (no writeback to CSV; just fix in-memory).
        def remove_trailing_A(s):
            if isinstance(s, str) and s.endswith('A'):
                return s[:-1]
            return s
        
        # Apply this only to the ID column:
        # e.g. "TCGA-6D-AA2E-01A" => "TCGA-6D-AA2E-01"
        # so that fix_tcga_id_slicing() becomes "TCGA.6D.AA2E.01"
        clinical_df_raw = pd.read_csv(ds_config["clinical_file"], sep=None, engine='python')
        clinical_df_raw[id_col] = clinical_df_raw[id_col].apply(remove_trailing_A)
    else:
        # If it's not Kidney, load normally
        clinical_df_raw = pd.read_csv(ds_config["clinical_file"], sep=None, engine='python')

    # Now proceed as usual with fix_tcga_id_slicing on clinical_df_raw
    if id_col not in clinical_df_raw.columns:
        raise ValueError(f"ID col '{id_col}' not found in {ds_config['clinical_file']}.")

    # Use the same variable name as before
    clinical_df = clinical_df_raw.copy()
    clinical_df[id_col] = clinical_df[id_col].apply(fix_tcga_id_slicing)

    if out_col not in clinical_df.columns:
        raise ValueError(f"Outcome col '{out_col}' not found in {ds_config['clinical_file']}.")

    # Regression vs classification logic remains the same:
    if is_regression:
        clinical_df[out_col] = clinical_df[out_col].apply(custom_parse_outcome)
        clinical_df = clinical_df.dropna(subset=[out_col]).copy()
        y = clinical_df[out_col].astype(float)
    else:
        raw_labels = clinical_df[out_col].astype(str).str.strip().str.replace('"','')
        raw_labels = raw_labels.replace(['','NA','NaN','nan'], np.nan)
        clinical_df[out_col] = raw_labels
        clinical_df = clinical_df.dropna(subset=[out_col]).copy()
        clinical_df[out_col] = clinical_df[out_col].astype('category')
        y = clinical_df[out_col].cat.codes

    # Next, fix columns in expression, methylation, mirna data
    exp_df.columns   = strip_and_slice_columns(exp_df.columns)
    methy_df.columns = strip_and_slice_columns(methy_df.columns)
    mirna_df.columns = strip_and_slice_columns(mirna_df.columns)

    data_modalities = {
        "Gene Expression": exp_df,
        "Methylation": methy_df,
        "miRNA": mirna_df
    }

    # Intersection and final Y
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
        "NMF": NMF(max_iter=5000, init='nndsvda'),
        "ICA": FastICA(max_iter=5000, tol=1e-3),
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
        "ICA": FastICA(max_iter=5000, tol=1e-3),
        "LDA": LDA(),  # We'll handle n_components constraints in code
        "FA": FactorAnalysis(),
        "KPCA": KernelPCA(kernel='rbf')  
    }

def get_classification_selectors():
    return {
        "MRMR": "mrmr_clf",
        "f_classifFS": "fclassif",
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
    plt.plot([mn,mx],[mn,mx], 'r--', alpha=0.7)
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
# F) REGRESSION TRAIN & EVAL
###############################################################################

def train_regression_model(X, y, model_name, random_state=0, out_dir=None, plot_prefix=""):
    # final alignment
    if X.shape[0] != len(y):
        nrow = min(X.shape[0], len(y))
        X = X.iloc[:nrow, :].reset_index(drop=True)
        y = y.iloc[:nrow].reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    if model_name=="LinearRegression":
        model = LinearRegression()
    elif model_name=="RandomForest":
        model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    elif model_name=="SVR":
        model = SVR(kernel='rbf')
    else:
        raise ValueError(f"Unknown regression model {model_name}")

    t0=time.time()
    model.fit(X_train, y_train)
    train_time=time.time()-t0

    y_pred = model.predict(X_test)
    mae=mean_absolute_error(y_test, y_pred)
    mse=mean_squared_error(y_test, y_pred)
    rmse=sqrt(mse)

    if out_dir and plot_prefix:
        scatter_path = os.path.join(out_dir, f"{plot_prefix}_scatter.png")
        plot_regression_scatter(y_test, y_pred, plot_prefix, scatter_path)

        resid_path = os.path.join(out_dir, f"{plot_prefix}_residuals.png")
        plot_regression_residuals(y_test, y_pred, plot_prefix, resid_path)

    metrics={
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "Train_Time_Seconds": train_time
    }
    return model, metrics

###############################################################################
# G) CLASSIFICATION TRAIN & EVAL
###############################################################################

def train_classification_model(X, y, model_name, random_state=0, out_dir=None, plot_prefix=""):
    if X.shape[0] != len(y):
        nrow = min(X.shape[0], len(y))
        X = X.iloc[:nrow, :].reset_index(drop=True)
        y = y.iloc[:nrow].reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    if model_name=="LogisticRegression":
        model=LogisticRegression(penalty='l2', solver='liblinear', random_state=random_state)
    elif model_name=="RandomForest":
        model=RandomForestClassifier(n_estimators=100, random_state=random_state)
    elif model_name=="SVC":
        model=SVC(kernel='rbf', probability=True, random_state=random_state)
    else:
        raise ValueError(f"Unknown classification model {model_name}")

    t0=time.time()
    model.fit(X_train, y_train)
    train_time=time.time()-t0

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision= precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall   = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1       = f1_score(y_test, y_pred, average='weighted')
    mcc      = matthews_corrcoef(y_test, y_pred)

    try:
        unique_cl = np.unique(y_test)
        if len(unique_cl)==2:
            y_proba=model.predict_proba(X_test)[:,1]
            aucv=roc_auc_score(y_test, y_proba)
        else:
            aucv=np.nan
    except:
        aucv=np.nan

    cm=confusion_matrix(y_test, y_pred)
    if out_dir and plot_prefix:
        cm_path = os.path.join(out_dir, f"{plot_prefix}_CM.png")
        str_labels = [str(lb) for lb in sorted(np.unique(y))]
        plot_confusion_matrix(cm, str_labels, plot_prefix, cm_path)

        if len(np.unique(y_test))==2:
            roc_path = os.path.join(out_dir, f"{plot_prefix}_ROC.png")
            plot_roc_curve_binary(model, X_test, y_test, str_labels, plot_prefix, roc_path)

    metrics={
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": mcc,
        "AUROC": aucv,
        "Train_Time_Seconds": train_time
    }
    return model, metrics

###############################################################################
# H) INTEGRATION UTILS
###############################################################################

def run_extraction_integration_regression(data_modalities, common_ids, y, extract_algo, ncomp, random_state=0):
    pieces=[]
    for mod_name, df_mod in data_modalities.items():
        df_f = df_mod[common_ids].copy()  
        df_f = df_f.loc[:, sorted(df_f.columns)]
        X_mod = df_f.transpose()
        X_mod = X_mod.apply(pd.to_numeric, errors='coerce').reset_index(drop=True)

        # scale
        if extract_algo.__class__.__name__=="NMF":
            scl=MinMaxScaler()
        else:
            scl=StandardScaler()
        X_scl=scl.fit_transform(X_mod)

        if hasattr(extract_algo,"random_state"):
            extract_algo.random_state=random_state
        if hasattr(extract_algo,"n_components"):
            extract_algo.n_components=ncomp

        if isinstance(extract_algo, PLSRegression):
            Y_arr = y.values.reshape(-1,1)
            X_red,_ = extract_algo.fit_transform(X_scl, Y_arr)
        else:
            X_red = extract_algo.fit_transform(X_scl)

        if isinstance(X_red, tuple):
            X_red=X_red[0]

        colnames=[f"{mod_name}_{extract_algo.__class__.__name__}_{i}" for i in range(X_red.shape[1])]
        df_piece=pd.DataFrame(X_red, columns=colnames)
        pieces.append(df_piece)

    integrated_X=pd.concat(pieces, axis=1)
    return integrated_X

def run_selection_integration_regression(data_modalities, common_ids, y, selection_code, nfeat, random_state=0):
    pieces=[]
    for mod_name, df_mod in data_modalities.items():
        df_f = df_mod[common_ids].copy()
        df_f = df_f.loc[:, sorted(df_f.columns)]
        X_mod = df_f.transpose()
        X_mod = X_mod.apply(pd.to_numeric, errors='coerce').reset_index(drop=True)

        if selection_code=="mrmr_reg":
            mi=mutual_info_regression(X_mod,y, random_state=random_state)
            if X_mod.shape[1]<=nfeat:
                piece_df=X_mod
            else:
                top_idx=np.argsort(mi)[-nfeat:]
                piece_df=X_mod.iloc[:, top_idx]
        elif selection_code=="lasso":
            lasso=Lasso(alpha=0.01, max_iter=10000, random_state=random_state)
            lasso.fit(X_mod,y)
            coefs=lasso.coef_
            if X_mod.shape[1]<=nfeat:
                piece_df=X_mod
            else:
                ridx=np.argsort(np.abs(coefs))[-nfeat:]
                piece_df=X_mod.iloc[:, ridx]
        elif selection_code=="enet":
            en=ElasticNet(alpha=0.01,l1_ratio=0.5, max_iter=10000, random_state=random_state)
            en.fit(X_mod,y)
            c=en.coef_
            if X_mod.shape[1]<=nfeat:
                piece_df=X_mod
            else:
                ridx=np.argsort(np.abs(c))[-nfeat:]
                piece_df=X_mod.iloc[:, ridx]
        elif selection_code=="freg":
            Fv,pv=f_regression(X_mod,y)
            if X_mod.shape[1]<=nfeat:
                piece_df=X_mod
            else:
                ridx=np.argsort(Fv)[-nfeat:]
                piece_df=X_mod.iloc[:, ridx]
        elif selection_code=="boruta_reg":
            rf=RF_for_BorutaReg(n_estimators=100, random_state=random_state)
            bor=BorutaPy(rf, n_estimators='auto', random_state=random_state)
            bor.fit(X_mod.values, y.values)
            mask=bor.support_
            chosen_cols=X_mod.columns[mask]
            if len(chosen_cols)>nfeat:
                ranks=bor.ranking_
                chosen_ranks=[(c,ranks[X_mod.columns.get_loc(c)]) for c in chosen_cols]
                sr=sorted(chosen_ranks, key=lambda x:x[1])
                final_cols=[sc[0] for sc in sr[:nfeat]]
                piece_df=X_mod[final_cols]
            else:
                piece_df=X_mod[chosen_cols]
        else:
            piece_df=X_mod

        new_cols=[f"{mod_name}_{selection_code}_{c}" for c in piece_df.columns]
        piece_df.columns=new_cols
        pieces.append(piece_df)

    integrated_X=pd.concat(pieces, axis=1)
    return integrated_X

def run_extraction_integration_classification(data_modalities, common_ids, y, extract_algo, ncomp, random_state=0):
    pieces=[]
    for mod_name, df_mod in data_modalities.items():
        df_f=df_mod[common_ids].copy()
        df_f=df_f.loc[:, sorted(df_f.columns)]
        X_mod=df_f.transpose()
        X_mod=X_mod.apply(pd.to_numeric, errors='coerce').reset_index(drop=True)

        if extract_algo.__class__.__name__=="NMF":
            scl=MinMaxScaler()
        else:
            scl=StandardScaler()
        X_scl=scl.fit_transform(X_mod)

        if hasattr(extract_algo,"random_state"):
            extract_algo.random_state=random_state

        # For LDA => ensure n_components <= min(#features, #classes-1)
        # or else reduce ncomp or skip
        if isinstance(extract_algo, LDA):
            n_classes = len(np.unique(y))
            max_lda = min(X_scl.shape[1], n_classes - 1)
            if max_lda < 1:
                # means we cannot do LDA at all => skip this modality
                # return empty piece
                continue
            if ncomp>max_lda:
                # reduce
                ncomp = max_lda
            extract_algo.n_components = ncomp
            X_red=extract_algo.fit_transform(X_scl, y)
        elif isinstance(extract_algo, KernelPCA):
            extract_algo.n_components = ncomp
            X_red=extract_algo.fit_transform(X_scl)
        else:
            if hasattr(extract_algo,"n_components"):
                extract_algo.n_components=ncomp
            X_red=extract_algo.fit_transform(X_scl)

        if isinstance(X_red, tuple):
            X_red=X_red[0]

        colnames=[f"{mod_name}_{extract_algo.__class__.__name__}_{i}" for i in range(X_red.shape[1])]
        df_piece=pd.DataFrame(X_red, columns=colnames)
        pieces.append(df_piece)

    if len(pieces)==0:
        # means we couldn't do LDA or others => return an empty df
        return pd.DataFrame()
    integrated_X=pd.concat(pieces, axis=1)
    return integrated_X

def run_selection_integration_classification(data_modalities, common_ids, y, selection_code, nfeat, random_state=0):
    pieces=[]
    for mod_name, df_mod in data_modalities.items():
        df_f=df_mod[common_ids].copy()
        df_f=df_f.loc[:, sorted(df_f.columns)]
        X_mod=df_f.transpose()
        X_mod=X_mod.apply(pd.to_numeric, errors='coerce').reset_index(drop=True)

        if selection_code=="mrmr_clf":
            mi=mutual_info_classif(X_mod, y, random_state=random_state)
            if X_mod.shape[1]<=nfeat:
                piece_df=X_mod
            else:
                tidx=np.argsort(mi)[-nfeat:]
                piece_df=X_mod.iloc[:, tidx]
        elif selection_code=="fclassif":
            Fv,pv=f_classif(X_mod,y)
            if X_mod.shape[1]<=nfeat:
                piece_df=X_mod
            else:
                ridx=np.argsort(Fv)[-nfeat:]
                piece_df=X_mod.iloc[:, ridx]
        elif selection_code=="logistic_l1":
            lr=LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=random_state)
            lr.fit(X_mod,y)
            coefs=np.abs(lr.coef_).sum(axis=0)  # sum across classes if multi
            if X_mod.shape[1]<=nfeat:
                piece_df=X_mod
            else:
                ridx=np.argsort(coefs)[-nfeat:]
                piece_df=X_mod.iloc[:, ridx]
        elif selection_code=="boruta_clf":
            rf=RF_for_BorutaClf(n_estimators=100, random_state=random_state)
            bor=BorutaPy(rf, n_estimators='auto', random_state=random_state)
            bor.fit(X_mod.values, y.values)
            mask=bor.support_
            chosen_cols=X_mod.columns[mask]
            if len(chosen_cols)>nfeat:
                ranks=bor.ranking_
                chosen_ranks=[(c,ranks[X_mod.columns.get_loc(c)]) for c in chosen_cols]
                sr=sorted(chosen_ranks, key=lambda x:x[1])
                final_cols=[sc[0] for sc in sr[:nfeat]]
                piece_df=X_mod[final_cols]
            else:
                piece_df=X_mod[chosen_cols]
        elif selection_code=="chi2_selection":
            X_clipped=np.clip(X_mod,0,None)
            sel=SelectKBest(chi2, k=min(nfeat, X_mod.shape[1]))
            sel.fit(X_clipped, y)
            mask=sel.get_support()
            chosen_cols=X_mod.columns[mask]
            piece_df=X_mod[chosen_cols]
        else:
            piece_df=X_mod

        new_cols=[f"{mod_name}_{selection_code}_{c}" for c in piece_df.columns]
        piece_df.columns=new_cols
        pieces.append(piece_df)

    if len(pieces)==0:
        # means something was empty => return empty df
        return pd.DataFrame()
    integrated_X=pd.concat(pieces, axis=1)
    return integrated_X

###############################################################################
# I) MAIN
###############################################################################

def main():

    # 1) REGRESSION block
    reg_extractors = get_regression_extractors()  
    reg_selectors  = get_regression_selectors()   
    reg_models     = ["LinearRegression", "RandomForest", "SVR"]
    n_comps_list   = [8,16,32,64,128]
    n_feats_list   = [8,16,32,64,128]

    n_modalities = 3  
    n_extract_runs = (
        len(REGRESSION_DATASETS)*len(reg_extractors)
        *len(n_comps_list)*len(reg_models)*n_modalities
    )
    n_select_runs = (
        len(REGRESSION_DATASETS)*len(reg_selectors)
        *len(n_feats_list)*len(reg_models)*n_modalities
    )
    reg_total_runs = n_extract_runs + n_select_runs

    progress_count=[0]

    print("=== REGRESSION BLOCK (AML, Sarcoma) ===")
    for ds_conf in REGRESSION_DATASETS:
        ds_name = ds_conf["name"]
        base_out = os.path.join("output_regression", ds_name)
        os.makedirs(base_out, exist_ok=True)
        os.makedirs(os.path.join(base_out,"models"), exist_ok=True)
        os.makedirs(os.path.join(base_out,"metrics"), exist_ok=True)
        os.makedirs(os.path.join(base_out,"plots"), exist_ok=True)

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

        if len(common_ids)==0 or y.shape[0]==0:
            print(f"No overlapping or no valid samples => skipping {ds_name}")
            continue

        # A) EXTRACTION
        extraction_results=[]
        for extr_name, extr_obj in reg_extractors.items():
            for nc in n_comps_list:
                progress_count[0]+=1
                run_idx=progress_count[0]
                print(f"[EXTRACT-REG] {run_idx}/{reg_total_runs} => {ds_name} | {extr_name}-{nc}")

                import copy
                extr_instance=copy.deepcopy(extr_obj)

                integrated_X = run_extraction_integration_regression(
                    data_modalities, common_ids, y, extr_instance, nc
                )
                if integrated_X.shape[0]==0 or integrated_X.shape[1]==0:
                    print(f"   -> Extraction got empty DataFrame => skipping.")
                    continue

                # train 3 models
                for model_name in reg_models:
                    plot_prefix = f"{ds_name}_EXTRACT_{extr_name}_{nc}_{model_name}"
                    model, mets = train_regression_model(
                        integrated_X, y, model_name,
                        out_dir=os.path.join(base_out,"plots"),
                        plot_prefix=plot_prefix
                    )
                    row = {
                        "Dataset": ds_name,
                        "Workflow": "Extraction",
                        "Extractor": extr_name,
                        "n_components": nc,
                        "Model": model_name,
                        **mets
                    }
                    extraction_results.append(row)
                    # save model
                    mfname = f"{ds_name}_EXTRACT_{extr_name}_{nc}_{model_name}.pkl"
                    mp = os.path.join(base_out,"models",mfname)
                    joblib.dump(model, mp)

        # save extraction metrics
        ext_df = pd.DataFrame(extraction_results)
        ext_df.to_csv(
            os.path.join(base_out,"metrics", f"{ds_name}_extraction_metrics.csv"),
            index=False
        )

        # B) SELECTION
        selection_results=[]
        for sel_name, sel_code in reg_selectors.items():
            for nf in n_feats_list:
                progress_count[0]+=1
                run_idx=progress_count[0]
                print(f"[SELECT-REG] {run_idx}/{reg_total_runs} => {ds_name} | {sel_name}-{nf}")

                integrated_X = run_selection_integration_regression(
                    data_modalities, common_ids, y, sel_code, nf
                )
                if integrated_X.shape[0]==0 or integrated_X.shape[1]==0:
                    print(f"   -> Selection got empty DataFrame => skipping.")
                    continue

                for model_name in reg_models:
                    plot_prefix = f"{ds_name}_SELECT_{sel_name}_{nf}_{model_name}"
                    model, mets = train_regression_model(
                        integrated_X, y, model_name,
                        out_dir=os.path.join(base_out,"plots"),
                        plot_prefix=plot_prefix
                    )
                    row={
                        "Dataset": ds_name,
                        "Workflow": "Selection",
                        "Selector": sel_name,
                        "n_features": nf,
                        "Model": model_name,
                        **mets
                    }
                    selection_results.append(row)
                    mfname = f"{ds_name}_SELECT_{sel_name}_{nf}_{model_name}.pkl"
                    mp = os.path.join(base_out,"models", mfname)
                    joblib.dump(model, mp)

        sel_df=pd.DataFrame(selection_results)
        sel_df.to_csv(
            os.path.join(base_out,"metrics", f"{ds_name}_selection_metrics.csv"),
            index=False
        )

    # 2) CLASSIFICATION block
    clf_extractors = get_classification_extractors()
    clf_selectors  = get_classification_selectors()
    clf_models     = ["LogisticRegression","RandomForest","SVC"]
    n_comps_list_clf = [8,16,32,64,128]
    n_feats_list_clf = [8,16,32,64,128]

    n_extract_runs_clf = (
        len(CLASSIFICATION_DATASETS)*len(clf_extractors)
        *len(n_comps_list_clf)*len(clf_models)*n_modalities
    )
    n_select_runs_clf = (
        len(CLASSIFICATION_DATASETS)*len(clf_selectors)
        *len(n_feats_list_clf)*len(clf_models)*n_modalities
    )
    clf_total_runs = n_extract_runs_clf + n_select_runs_clf
    progress_count_clf=[0]

    print("\n=== CLASSIFICATION BLOCK (Breast, Colon, Kidney, Liver, Lung, Melanoma, Ovarian) ===")
    for ds_conf in CLASSIFICATION_DATASETS:
        ds_name = ds_conf["name"]
        base_out = os.path.join("output_classification", ds_name)
        os.makedirs(base_out, exist_ok=True)
        os.makedirs(os.path.join(base_out,"models"), exist_ok=True)
        os.makedirs(os.path.join(base_out,"metrics"), exist_ok=True)
        os.makedirs(os.path.join(base_out,"plots"), exist_ok=True)

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

        if len(common_ids)==0 or y.shape[0]==0:
            print(f"No overlapping or no valid samples => skipping {ds_name}")
            continue

        # A) extraction
        extraction_results=[]
        for extr_name, extr_obj in clf_extractors.items():
            for nc in n_comps_list_clf:
                progress_count_clf[0]+=1
                run_idx=progress_count_clf[0]
                print(f"[EXTRACT-CLF] {run_idx}/{clf_total_runs} => {ds_name} | {extr_name}-{nc}")

                import copy
                extr_instance=copy.deepcopy(extr_obj)
                integrated_X=run_extraction_integration_classification(
                    data_modalities, common_ids, y, extr_instance, nc, random_state=0
                )
                if integrated_X.shape[0]==0 or integrated_X.shape[1]==0:
                    print("   -> Extraction got empty df => skipping.")
                    continue

                for model_name in clf_models:
                    plot_prefix = f"{ds_name}_EXTRACT_{extr_name}_{nc}_{model_name}"
                    model, mets= train_classification_model(
                        integrated_X, y, model_name,
                        out_dir=os.path.join(base_out,"plots"),
                        plot_prefix=plot_prefix
                    )
                    row={
                        "Dataset": ds_name,
                        "Workflow": "Extraction",
                        "Extractor": extr_name,
                        "n_components": nc,
                        "Model": model_name,
                        **mets
                    }
                    extraction_results.append(row)

                    mfname=f"{ds_name}_EXTRACT_{extr_name}_{nc}_{model_name}.pkl"
                    mp=os.path.join(base_out,"models", mfname)
                    joblib.dump(model, mp)

        ext_df=pd.DataFrame(extraction_results)
        ext_df.to_csv(
            os.path.join(base_out,"metrics", f"{ds_name}_extraction_metrics.csv"),
            index=False
        )

        # B) selection
        selection_results=[]
        for sel_name, sel_code in clf_selectors.items():
            for nf in n_feats_list_clf:
                progress_count_clf[0]+=1
                run_idx=progress_count_clf[0]
                print(f"[SELECT-CLF] {run_idx}/{clf_total_runs} => {ds_name} | {sel_name}-{nf}")

                integrated_X=run_selection_integration_classification(
                    data_modalities, common_ids, y, sel_code, nf, random_state=0
                )
                if integrated_X.shape[0]==0 or integrated_X.shape[1]==0:
                    print("   -> Selection got empty df => skipping.")
                    continue

                for model_name in clf_models:
                    plot_prefix = f"{ds_name}_SELECT_{sel_name}_{nf}_{model_name}"
                    model, mets= train_classification_model(
                        integrated_X, y, model_name,
                        out_dir=os.path.join(base_out,"plots"),
                        plot_prefix=plot_prefix
                    )
                    row={
                        "Dataset": ds_name,
                        "Workflow": "Selection",
                        "Selector": sel_name,
                        "n_features": nf,
                        "Model": model_name,
                        **mets
                    }
                    selection_results.append(row)
                    mfname=f"{ds_name}_SELECT_{sel_name}_{nf}_{model_name}.pkl"
                    mp=os.path.join(base_out,"models", mfname)
                    joblib.dump(model, mp)

        sel_df=pd.DataFrame(selection_results)
        sel_df.to_csv(
            os.path.join(base_out,"metrics", f"{ds_name}_selection_metrics.csv"),
            index=False
        )

    print("\nAll done! Regression outputs in 'output_regression/' and classification outputs in 'output_classification/'.")


if __name__=="__main__":
    main()