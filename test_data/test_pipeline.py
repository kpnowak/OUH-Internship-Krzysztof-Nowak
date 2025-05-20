#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA

# Add parent directory to path to import the main script
sys.path.append(str(Path(__file__).parent.parent))

from alg3_multi_additions.alg3_multi_additions_CPU import (
    process_modality,
    process_reg_selection_combo_cv,
    process_clf_selection_combo_cv
)

def test_regression_pipeline():
    """Test the regression pipeline with synthetic data."""
    print("\nTesting Regression Pipeline...")
    
    # Load data
    data_dir = Path('test_data/regression')
    exp_data = pd.read_csv(data_dir / 'exp.csv', index_col=0)
    methy_data = pd.read_csv(data_dir / 'methy.csv', index_col=0)
    mirna_data = pd.read_csv(data_dir / 'mirna.csv', index_col=0)
    clinical_data = pd.read_csv(data_dir / 'clinical.csv')
    all_ids = list(clinical_data['sample_id'])
    y = clinical_data['survival_time'].values
    data_modalities = {
        'exp': exp_data,
        'methy': methy_data,
        'mirna': mirna_data
    }
    # Dummy values for required arguments
    ds_name = 'test_regression'
    sel_name = 'f_classif'
    sel_code = 'f_classif'
    n_feats = 10
    reg_models = ['RandomForest']
    base_out = None
    progress_count = [0]
    reg_total_runs = 1
    # Test feature selection (main pipeline)
    try:
        result = process_reg_selection_combo_cv(
            ds_name, sel_name, sel_code, n_feats, reg_models,
            data_modalities, all_ids, y, base_out, progress_count, reg_total_runs, test_size=0.2, n_splits=3
        )
        print("Successfully completed regression selection")
    except Exception as e:
        print(f"Error in regression selection: {str(e)}")
    # Test feature extraction (single modality)
    print("\nTesting Feature Extraction (single modality)...")
    modality_name = 'exp'
    modality_df = exp_data
    id_train = all_ids[:40]
    id_val = all_ids[40:50]
    idx_test = list(range(0, 10))
    y_train = y[:40]
    extr_obj = PCA(n_components=5)
    ncomps = 5
    idx_to_id = {i: sid for i, sid in enumerate(all_ids)}
    try:
        X_train, X_val, X_test = process_modality(
            modality_name, modality_df, id_train, id_val, idx_test, y_train, extr_obj, ncomps, idx_to_id
        )
        print(f"Feature extraction output shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    except Exception as e:
        print(f"Error in feature extraction: {str(e)}")

def test_classification_pipeline():
    """Test the classification pipeline with synthetic data."""
    print("\nTesting Classification Pipeline...")
    
    # Load data
    data_dir = Path('test_data/classification')
    exp_data = pd.read_csv(data_dir / 'exp.csv', index_col=0)
    methy_data = pd.read_csv(data_dir / 'methy.csv', index_col=0)
    mirna_data = pd.read_csv(data_dir / 'mirna.csv', index_col=0)
    clinical_data = pd.read_csv(data_dir / 'clinical.csv')
    all_ids = list(clinical_data['sample_id'])
    y = clinical_data['status'].values
    data_modalities = {
        'exp': exp_data,
        'methy': methy_data,
        'mirna': mirna_data
    }
    ds_name = 'test_classification'
    sel_name = 'f_classif'
    sel_code = 'f_classif'
    n_feats = 10
    clf_models = ['LogisticRegression']
    base_out = None
    progress_count = [0]
    clf_total_runs = 1
    try:
        result = process_clf_selection_combo_cv(
            ds_name, sel_name, sel_code, n_feats, clf_models,
            data_modalities, all_ids, y, base_out, progress_count, clf_total_runs, test_size=0.2, n_splits=3
        )
        print("Successfully completed classification selection")
    except Exception as e:
        print(f"Error in classification selection: {str(e)}")
    # Test feature extraction (single modality)
    print("\nTesting Feature Extraction (single modality)...")
    modality_name = 'exp'
    modality_df = exp_data
    id_train = all_ids[:40]
    id_val = all_ids[40:50]
    idx_test = list(range(0, 10))
    y_train = y[:40]
    extr_obj = PCA(n_components=5)
    ncomps = 5
    idx_to_id = {i: sid for i, sid in enumerate(all_ids)}
    try:
        X_train, X_val, X_test = process_modality(
            modality_name, modality_df, id_train, id_val, idx_test, y_train, extr_obj, ncomps, idx_to_id
        )
        print(f"Feature extraction output shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    except Exception as e:
        print(f"Error in feature extraction: {str(e)}")

if __name__ == "__main__":
    # First generate test data if it doesn't exist
    if not os.path.exists('test_data/regression/exp.csv'):
        print("Generating test data...")
        from generate_test_data import generate_synthetic_data
        generate_synthetic_data()
    
    # Run tests
    test_regression_pipeline()
    test_classification_pipeline() 