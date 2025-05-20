#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression, make_classification

# Set random seed for reproducibility
np.random.seed(42)

def generate_synthetic_data(n_samples=50, n_features=100, n_informative=20):
    """Generate synthetic data for both regression and classification tasks."""
    
    # Create output directories
    os.makedirs('test_data/regression', exist_ok=True)
    os.makedirs('test_data/classification', exist_ok=True)
    
    # Generate regression data
    X_reg, y_reg = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=0.1,
        random_state=42
    )
    
    # Generate classification data
    X_clf, y_clf = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_classes=2,
        random_state=42
    )
    
    # Create sample IDs
    sample_ids = [f'SAMPLE_{i:03d}' for i in range(n_samples)]
    
    # Create feature names
    feature_names = [f'FEATURE_{i:03d}' for i in range(n_features)]
    
    # Save regression data
    # Expression data
    exp_df = pd.DataFrame(X_reg.T, columns=sample_ids, index=feature_names)
    exp_df.to_csv('test_data/regression/exp.csv')
    
    # Methylation data (slightly modified version of expression)
    methy_df = pd.DataFrame((X_reg + np.random.normal(0, 0.1, X_reg.shape)).T, 
                           columns=sample_ids, index=feature_names)
    methy_df.to_csv('test_data/regression/methy.csv')
    
    # miRNA data (another modified version)
    mirna_df = pd.DataFrame((X_reg + np.random.normal(0, 0.2, X_reg.shape)).T, 
                           columns=sample_ids, index=feature_names)
    mirna_df.to_csv('test_data/regression/mirna.csv')
    
    # Clinical data for regression
    clinical_df = pd.DataFrame({
        'sample_id': sample_ids,
        'survival_time': y_reg
    })
    clinical_df.to_csv('test_data/regression/clinical.csv', index=False)
    
    # Save classification data
    # Expression data
    exp_df = pd.DataFrame(X_clf.T, columns=sample_ids, index=feature_names)
    exp_df.to_csv('test_data/classification/exp.csv')
    
    # Methylation data
    methy_df = pd.DataFrame((X_clf + np.random.normal(0, 0.1, X_clf.shape)).T, 
                           columns=sample_ids, index=feature_names)
    methy_df.to_csv('test_data/classification/methy.csv')
    
    # miRNA data
    mirna_df = pd.DataFrame((X_clf + np.random.normal(0, 0.2, X_clf.shape)).T, 
                           columns=sample_ids, index=feature_names)
    mirna_df.to_csv('test_data/classification/mirna.csv')
    
    # Clinical data for classification
    clinical_df = pd.DataFrame({
        'sample_id': sample_ids,
        'status': y_clf
    })
    clinical_df.to_csv('test_data/classification/clinical.csv', index=False)
    
    print("Generated test data:")
    print(f"- {n_samples} samples")
    print(f"- {n_features} features")
    print(f"- {n_informative} informative features")
    print("\nFiles created:")
    print("Regression:")
    print("- test_data/regression/exp.csv")
    print("- test_data/regression/methy.csv")
    print("- test_data/regression/mirna.csv")
    print("- test_data/regression/clinical.csv")
    print("\nClassification:")
    print("- test_data/classification/exp.csv")
    print("- test_data/classification/methy.csv")
    print("- test_data/classification/mirna.csv")
    print("- test_data/classification/clinical.csv")

if __name__ == "__main__":
    generate_synthetic_data() 