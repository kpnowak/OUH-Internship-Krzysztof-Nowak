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

# Dimensionality Reduction
from sklearn.decomposition import PCA, NMF, FastICA, FactorAnalysis
from sklearn.cross_decomposition import PLSRegression

# Data scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Train/test splits & regression metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

# Feature selection
from sklearn.feature_selection import mutual_info_regression, f_regression
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor as RF_for_Boruta

# For parallelization
from joblib import Parallel, delayed

###############################################################################
# 1) Configuration for Multiple Datasets
###############################################################################

DATASETS = [
    {
        "name": "AML",
        "clinical_file": "clinical/aml.csv",
        "omics_dir": "aml",
        "id_col": "sampleID",
        "outcome_col": "lab_procedure_bone_marrow_blast_cell_outcome_percent_value"
    },
    {
        "name": "Breast",
        "clinical_file": "clinical/breast.csv",
        "omics_dir": "breast",
        "id_col": "sampleID",
        "outcome_col": "pathologic_T"
    },
    {
        "name": "Colon",
        "clinical_file": "clinical/colon.csv",
        "omics_dir": "colon",
        "id_col": "sampleID",
        "outcome_col": "pathologic_T"
    },
    {
        "name": "Kidney",
        "clinical_file": "clinical/kidney.csv",
        "omics_dir": "kidney",
        "id_col": "submitter_id.samples",
        "outcome_col": "pathologic_T"
    },
    {
        "name": "Liver",
        "clinical_file": "clinical/liver.csv",
        "omics_dir": "liver",
        "id_col": "sampleID",
        "outcome_col": "pathologic_T"
    },
    {
        "name": "Lung",
        "clinical_file": "clinical/lung.csv",
        "omics_dir": "lung",
        "id_col": "sampleID",
        "outcome_col": "pathologic_T"
    },
    {
        "name": "Melanoma",
        "clinical_file": "clinical/melanoma.csv",
        "omics_dir": "melanoma",
        "id_col": "sampleID",
        "outcome_col": "pathologic_T"
    }
]

###############################################################################
# 2) Custom parse function
###############################################################################

def custom_parse_func(value):
    """
    Converts the outcome column from string to float.
    If it contains '|', pick the maximum. Otherwise parse single float.
    If fails, return NaN.
    """
    if isinstance(value, str):
        if '|' in value:
            parts = value.split('|')
            try:
                floats = [float(x) for x in parts]
                return max(floats)
            except:
                return np.nan
        else:
            try:
                return float(value)
            except:
                return np.nan
    else:
        return float(value) if pd.notna(value) else np.nan

###############################################################################
# 3) Data Loading
###############################################################################

def load_dataset(ds_config):
    """
    Reads omics + clinical for one dataset (ds_config) from CSV-like files 
    with sep=None and engine='python'.
    """
    omics_dir = ds_config["omics_dir"]
    exp_df   = pd.read_csv(f"{omics_dir}/exp.csv",    sep=None, engine='python', index_col=0)
    methy_df = pd.read_csv(f"{omics_dir}/methy.csv",  sep=None, engine='python', index_col=0)
    mirna_df = pd.read_csv(f"{omics_dir}/mirna.csv",  sep=None, engine='python', index_col=0)

    clinical_df = pd.read_csv(ds_config["clinical_file"], sep=None, engine='python')

    return exp_df, methy_df, mirna_df, clinical_df


def prepare_data(ds_config, exp_df, methy_df, mirna_df, clinical_df):
    """
    1) Convert ID col from '-' to '.' if needed
    2) Parse outcome_col with custom_parse_func
    3) Drop NaNs
    4) Intersect with omics columns => build final y
    """
    id_col      = ds_config["id_col"]
    outcome_col = ds_config["outcome_col"]

    # fix ID col from '-' to '.'
    clinical_df[id_col] = clinical_df[id_col].astype(str).str.replace('-', '.')

    # parse outcome
    if outcome_col not in clinical_df.columns:
        raise ValueError(f"Outcome col '{outcome_col}' not found in {ds_config['clinical_file']}")

    clinical_df[outcome_col] = clinical_df[outcome_col].apply(custom_parse_func)
    # drop NaN in outcome
    clinical_df = clinical_df.dropna(subset=[outcome_col]).copy()

    # fix omics columns from '-' to '.' if needed
    exp_df.columns   = exp_df.columns.str.replace('-', '.')
    methy_df.columns = methy_df.columns.str.replace('-', '.')
    mirna_df.columns = mirna_df.columns.str.replace('-', '.')

    data_modalities = {
        "Gene Expression": exp_df,
        "Methylation": methy_df,
        "miRNA": mirna_df
    }

    # Intersect
    common_ids = set(clinical_df[id_col])
    for mod_df in data_modalities.values():
        common_ids = common_ids.intersection(mod_df.columns)
    common_ids = sorted(list(common_ids))

    clinical_filtered = clinical_df[clinical_df[id_col].isin(common_ids)].copy()
    clinical_filtered = clinical_filtered.sort_values(id_col).reset_index(drop=True)

    y = clinical_filtered[outcome_col].astype(float)

    return data_modalities, common_ids, y, clinical_filtered

###############################################################################
# 4) Regression training
###############################################################################

def train_and_evaluate_regression(X, y, model_name='LinearRegression', random_state=0):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    if model_name == 'LinearRegression':
        model = LinearRegression()
    elif model_name == 'RandomForest':
        model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    elif model_name == 'SVR':
        model = SVR(kernel='rbf')
    else:
        raise ValueError(f"Unknown model {model_name}")

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)

    return model, {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'Train_Time_Seconds': train_time
    }

###############################################################################
# 5) Extractors & Workflows
###############################################################################

def get_extraction_algos():
    return {
        'PCA': PCA(),
        'NMF': NMF(max_iter=5000, init='nndsvda'),
        'ICA': FastICA(max_iter=5000, tol=0.001),
        'FA': FactorAnalysis(),
        'PLS': PLSRegression()
    }

def apply_extraction(alg_name, alg_obj, X, y=None, n_components=8, random_state=0):
    if alg_name == 'NMF':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    if hasattr(alg_obj, 'random_state'):
        alg_obj.random_state = random_state
    if hasattr(alg_obj, 'n_components'):
        alg_obj.n_components = n_components

    if alg_name == 'PLS':
        Y_array = np.asarray(y).reshape(-1, 1)
        X_reduced, _ = alg_obj.fit_transform(X_scaled, Y_array)
    else:
        X_reduced = alg_obj.fit_transform(X_scaled)

    # If the estimator returns a tuple instead of a 2D array
    if isinstance(X_reduced, tuple):
        X_reduced = X_reduced[0]

    cols = [f"{alg_name}_comp_{i}" for i in range(n_components)]
    X_reduced_df = pd.DataFrame(X_reduced, columns=cols, index=X.index)
    return X_reduced_df

def run_extraction_workflow(modality_name, mod_df, y, dataset_name, output_dir,
                            progress_counter, total_runs):
    results = []
    extr_algos = get_extraction_algos()
    n_comps_list = [8,16,32,64,128]
    reg_models = ['LinearRegression','RandomForest','SVR']

    # Build all combos
    combos = []
    for ext_name, ext_obj in extr_algos.items():
        for n_comp in n_comps_list:
            combos.append((ext_name, ext_obj, n_comp))

    def process_extraction_combo(ext_name, ext_obj, n_comp):
        # Increment the shared counter to track total progress (non-threadsafe, but works in practice)
        progress_counter[0] += 1
        current_run = progress_counter[0]
        print(f"[EXTRACTION] Running {current_run}/{total_runs} => "
              f"{dataset_name} | {modality_name} | {ext_name}-{n_comp}")

        combo_results = []
        try:
            X_reduced = apply_extraction(ext_name, ext_obj, mod_df, y, n_components=n_comp)
        except Exception as e:
            print(f"Skipping {dataset_name} -> {modality_name}-{ext_name}-{n_comp} due to error: {e}")
            return combo_results  # empty

        for model_name in reg_models:
            model, metrics_dict = train_and_evaluate_regression(X_reduced, y, model_name)
            model_fname = f"{dataset_name}_EXTRACT_{modality_name}_{ext_name}_{n_comp}_{model_name}.pkl"
            model_path = os.path.join(output_dir, "models", model_fname)
            joblib.dump(model, model_path)

            result_entry = {
                'Dataset': dataset_name,
                'Workflow': 'Extraction',
                'Modality': modality_name,
                'Algorithm': ext_name,
                'n_components': n_comp,
                'Model': model_name,
                **metrics_dict
            }
            combo_results.append(result_entry)

        return combo_results

    # Parallelize the extraction combos
    parallel_results = Parallel(n_jobs=-1)(
        delayed(process_extraction_combo)(ext_name, ext_obj, n_comp)
        for (ext_name, ext_obj, n_comp) in combos
    )

    # Flatten results
    for rlist in parallel_results:
        results.extend(rlist)

    return results

###############################################################################
# 6) Selectors & Workflow
###############################################################################

def get_selection_algos():
    return {
        'MRMR': 'mrmr',
        'LASSO': 'lasso',
        'ElasticNetFS': 'enet',
        'f_regressionFS': 'freg',
        'Boruta': 'boruta'
    }

def apply_selection_with_n(method, X, y, n_features=8, random_state=0):
    if X.shape[1] <= n_features:
        return X

    method = method.lower()
    try:
        if method == 'mrmr':
            mi = mutual_info_regression(X, y, random_state=random_state)
            top_idx = np.argsort(mi)[-n_features:]
            return X.iloc[:, top_idx]

        elif method == 'lasso':
            lasso = Lasso(alpha=0.01, max_iter=10000, random_state=random_state)
            lasso.fit(X, y)
            coefs = lasso.coef_
            rank_idx = np.argsort(np.abs(coefs))[-n_features:]
            return X.iloc[:, rank_idx]

        elif method == 'enet':
            enet = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000, random_state=random_state)
            enet.fit(X, y)
            coefs = enet.coef_
            rank_idx = np.argsort(np.abs(coefs))[-n_features:]
            return X.iloc[:, rank_idx]

        elif method == 'freg':
            Fvals, pvals = f_regression(X, y)
            rank_idx = np.argsort(Fvals)[-n_features:]
            return X.iloc[:, rank_idx]

        elif method == 'boruta':
            rf_est = RF_for_Boruta(n_estimators=100, random_state=random_state)
            bor = BorutaPy(rf_est, n_estimators='auto', random_state=random_state)
            bor.fit(X.values, y.values.astype(float))
            mask = bor.support_
            chosen_cols = X.columns[mask]
            # If Boruta returns more than n_features, pick top N by rank
            if len(chosen_cols) > n_features:
                ranks = bor.ranking_
                chosen_ranks = [(c, ranks[X.columns.get_loc(c)]) for c in chosen_cols]
                sorted_cols = sorted(chosen_ranks, key=lambda x: x[1])
                final_cols = [sc[0] for sc in sorted_cols[:n_features]]
                return X[final_cols]
            else:
                return X[chosen_cols]
        else:
            print(f"Unknown FS method: {method}")
            return X

    except Exception as e:
        print(f"{method} FS failed: {e}")
        return X

def run_selection_workflow(modality_name, mod_df, y, dataset_name, output_dir,
                           progress_counter, total_runs):
    results = []
    sel_algos = get_selection_algos()
    n_feats_list = [8,16,32,64,128]
    reg_models = ['LinearRegression','RandomForest','SVR']

    # Build combos
    combos = []
    for sel_name, sel_method in sel_algos.items():
        for n_feat in n_feats_list:
            combos.append((sel_name, sel_method, n_feat))

    def process_selection_combo(sel_name, sel_method, n_feat):
        # Increment shared counter
        progress_counter[0] += 1
        current_run = progress_counter[0]
        print(f"[SELECTION] Running {current_run}/{total_runs} => "
              f"{dataset_name} | {modality_name} | {sel_name}-{n_feat}")

        combo_results = []
        X_selected = apply_selection_with_n(sel_method, mod_df, y, n_features=n_feat)
        if X_selected.shape[1] == 0:
            print(f"Skipping {dataset_name} -> {modality_name}-{sel_name}-{n_feat}: No features selected.")
            return combo_results  # empty

        for model_name in reg_models:
            model, metrics_dict = train_and_evaluate_regression(X_selected, y, model_name)
            model_fname = f"{dataset_name}_SELECT_{modality_name}_{sel_name}_{n_feat}_{model_name}.pkl"
            model_path = os.path.join(output_dir, "models", model_fname)
            joblib.dump(model, model_path)

            result_entry = {
                'Dataset': dataset_name,
                'Workflow': 'Selection',
                'Modality': modality_name,
                'Algorithm': sel_name,
                'n_features': n_feat,
                'Model': model_name,
                **metrics_dict
            }
            combo_results.append(result_entry)

        return combo_results

    # Parallelize selection combos
    parallel_results = Parallel(n_jobs=-1)(
        delayed(process_selection_combo)(sel_name, sel_method, n_feat)
        for (sel_name, sel_method, n_feat) in combos
    )

    # Flatten
    for rlist in parallel_results:
        results.extend(rlist)

    return results

###############################################################################
# 7) Main
###############################################################################

def main():
    """
    Two separate workflows for each dataset:
      - Extraction => regression
      - Selection => regression
    We will print progress like '25/300' to inform the user which run is occurring.
    """

    # -------------------------------------------------------------------------
    # 1) Compute the TOTAL number of runs across all datasets
    #    so we can do "current_run / total_runs" in prints.
    # -------------------------------------------------------------------------
    extraction_algos = list(get_extraction_algos().keys())
    selection_algos = list(get_selection_algos().keys())
    n_comps_list = [8,16,32,64,128]
    n_feats_list = [8,16,32,64,128]
    reg_models = ['LinearRegression','RandomForest','SVR']

    # We need the total # of extraction combos for a single dataset & single modality:
    #    (#extr_algos) * (#n_comps_list) * (#reg_models)
    # and the total # of selection combos for a single dataset & single modality:
    #    (#sel_algos) * (#n_feats_list) * (#reg_models).
    # Then multiply each by (# of modalities) and sum for (# of datasets).

    # For simplicity, let's assume each dataset has the same 3 modalities:
    #    "Gene Expression", "Methylation", "miRNA"
    # If you have a variable # of modalities per dataset, you'd just compute that inside the loop.
    n_modalities = 3  # "Gene Expression", "Methylation", "miRNA"

    single_dataset_extraction = len(extraction_algos) * len(n_comps_list) * len(reg_models) * n_modalities
    single_dataset_selection  = len(selection_algos)   * len(n_feats_list)  * len(reg_models) * n_modalities

    total_runs = (single_dataset_extraction + single_dataset_selection) * len(DATASETS)

    # We'll store the current progress in a single-element list so it can be mutated
    # by parallel processes. This is not strictly thread-safe, but works OK in practice.
    progress_counter = [0]

    # -------------------------------------------------------------------------
    # 2) Main Loop
    # -------------------------------------------------------------------------
    for ds_idx, ds in enumerate(DATASETS, start=1):
        dataset_name = ds["name"]

        # Make subfolders
        base_out_dir = os.path.join("output", dataset_name)
        os.makedirs(os.path.join(base_out_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(base_out_dir, "metrics"), exist_ok=True)
        os.makedirs(os.path.join(base_out_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(base_out_dir, "logs"), exist_ok=True)

        print(f"\n=== Processing dataset {ds_idx}/{len(DATASETS)}: {dataset_name} ===")

        # 1) Load data
        exp_df, methy_df, mirna_df, clinical_df = load_dataset(ds)

        # 2) Prepare data
        data_modalities, common_ids, y, clinical_filtered = prepare_data(
            ds, exp_df, methy_df, mirna_df, clinical_df
        )

        extraction_results = []
        selection_results = []

        # Loop over each modality
        for modality_name, df_modality in data_modalities.items():
            # Filter columns by common_ids
            df_modality_filtered = df_modality[common_ids].copy()
            df_modality_filtered = df_modality_filtered.loc[:, sorted(df_modality_filtered.columns)]
            X_modality = df_modality_filtered.transpose().reset_index(drop=True)
            X_modality = X_modality.apply(pd.to_numeric, errors='coerce')

            # A) EXTRACT WORKFLOW
            ext_res = run_extraction_workflow(
                modality_name, X_modality, y, dataset_name, base_out_dir,
                progress_counter, total_runs
            )
            extraction_results.extend(ext_res)

            # B) SELECT WORKFLOW
            sel_res = run_selection_workflow(
                modality_name, X_modality, y, dataset_name, base_out_dir,
                progress_counter, total_runs
            )
            selection_results.extend(sel_res)

        # Save CSVs for that dataset
        extraction_df = pd.DataFrame(extraction_results)
        extraction_csv = os.path.join(base_out_dir, "metrics", f"{dataset_name}_extraction_metrics.csv")
        extraction_df.to_csv(extraction_csv, index=False)

        selection_df = pd.DataFrame(selection_results)
        selection_csv = os.path.join(base_out_dir, "metrics", f"{dataset_name}_selection_metrics.csv")
        selection_df.to_csv(selection_csv, index=False)

        print(f"Finished dataset {dataset_name}. Metrics in {base_out_dir}/metrics/.")


if __name__ == "__main__":
    main()