import os
import time
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, NMF, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from boruta import BorutaPy
from skrebate import ReliefF
from sklearn.model_selection import StratifiedKFold

# Load datasets
def load_data():
    exp_df = pd.read_csv('sarcoma/exp', sep=None, engine='python', index_col=0)
    methy_df = pd.read_csv('sarcoma/methy', sep=None, engine='python', index_col=0)
    mirna_df = pd.read_csv('sarcoma/mirna', sep=None, engine='python', index_col=0)
    survival_df = pd.read_csv('sarcoma/survival', sep=None, engine='python')
    return exp_df, methy_df, mirna_df, survival_df

# Prepare data
def prepare_data(exp_df, methy_df, mirna_df, survival_df):
    survival_df['PatientID'] = survival_df['PatientID'].str.replace('-', '.')
    data_modalities = {'Gene Expression': exp_df, 'Methylation': methy_df, 'miRNA': mirna_df}
    # Find common patients across all datasets
    common_patients = set(survival_df['PatientID'])
    for modality_df in data_modalities.values():
        common_patients = common_patients.intersection(modality_df.columns)
    common_patients = list(common_patients)
    # Filter survival data
    survival_filtered = survival_df[survival_df['PatientID'].isin(common_patients)]
    survival_filtered = survival_filtered.sort_values('PatientID').reset_index(drop=True)
    y = survival_filtered['Death'].astype(int)
    return data_modalities, common_patients, y

# Define algorithms
def get_algorithms():
    dim_reduction_algorithms = {
        'PCA': PCA(),
        'NMF': NMF(max_iter=1000, init='nndsvda'),
        'LDA': LDA(),
        'ICA': FastICA(max_iter=2000, tol=0.001),
        't-SNE': TSNE(perplexity=30)
    }

    feature_selection_algorithms = {
        'MRMR': 'mutual_info',
        'LASSO': 'lasso',
        'Logistic_L1': 'logistic_l1',
        'ReliefF': 'relieff',
        'Boruta': 'boruta',
        # 'RFECV_LogisticRegression': 'rfecv'
    }
    return dim_reduction_algorithms, feature_selection_algorithms

# Integration techniques
def integrate_features(feature_sets, method='concatenate'):
    if method == 'concatenate':
        # Combine all features
        integrated_features = list(set().union(*feature_sets))
    elif method == 'average':
        # Features selected most frequently
        all_features = [feature for feature_set in feature_sets for feature in feature_set]
        feature_counts = pd.Series(all_features).value_counts()
        threshold = len(feature_sets) / 2  # Appear in at least half the runs
        integrated_features = feature_counts[feature_counts >= threshold].index.tolist()
    elif method == 'intersection':
        # Features common to all sets
        integrated_features = list(set.intersection(*map(set, feature_sets)))
    else:
        raise ValueError("Invalid integration method.")
    return integrated_features[:10]  # Return top 10 features

# Apply algorithms
def apply_algorithms(modality_name, X, y, dim_reduction_algorithms, feature_selection_algorithms, results_list):
    print(f"\nProcessing modality: {modality_name}")
    modality_results = []

    # Dimensions to iterate over
    dimensions = [16, 32, 64]
    integration_methods = ['concatenate', 'average', 'intersection']

    # Feature Extraction
    for alg_name, alg in dim_reduction_algorithms.items():
        for n_components in dimensions:
            runtimes = []
            selected_features_runs = []
            print(f"Running {alg_name} with n_components={n_components}")
            for i in range(1, 11):
                start_time = time.time()
                # Clone the algorithm to avoid state carry-over
                alg_instance = alg
                # Set random state if applicable
                if hasattr(alg_instance, 'random_state'):
                    alg_instance.random_state = i
                # Set n_components if applicable
                if alg_name == 't-SNE':
                    alg_instance.n_components = 2  # t-SNE typically uses 2 components
                elif alg_name == 'LDA':
                    alg_instance.n_components = min(len(np.unique(y)) - 1, X.shape[1])
                else:
                    alg_instance.n_components = n_components
                # Scaling
                if alg_name == 'NMF':
                    scaler = MinMaxScaler()
                else:
                    scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                # Algorithm-specific processing
                if alg_name == 'LDA':
                    X_transformed = alg_instance.fit_transform(X_scaled, y)
                    # Extract top features
                    feature_importances = np.abs(alg_instance.coef_).flatten()
                elif alg_name == 't-SNE':
                    X_transformed = alg_instance.fit_transform(X_scaled)
                    feature_importances = None  # t-SNE does not provide feature importances
                else:
                    X_transformed = alg_instance.fit_transform(X_scaled)
                    components = alg_instance.components_
                    if alg_name in ['PCA', 'ICA']:
                        feature_importances = np.sum(np.abs(components), axis=0)
                    elif alg_name == 'NMF':
                        feature_importances = np.sum(components, axis=0)
                # Get top features
                if feature_importances is not None:
                    top_indices = np.argsort(feature_importances)[-n_components:]
                    top_features = X.columns[top_indices].tolist()
                    selected_features_runs.append(top_features)
                else:
                    selected_features_runs.append([])
                runtime = time.time() - start_time
                runtimes.append(runtime)
                print(f"{alg_name} run {i}/10 done: time {runtime:.2f} seconds")
            # Integration of features
            for method in integration_methods:
                integrated_features = integrate_features(selected_features_runs, method=method)
                avg_runtime = np.mean(runtimes)
                modality_results.append({
                    'Modality': modality_name,
                    'Algorithm': alg_name,
                    'Dimension': n_components,
                    'Integration_Method': method,
                    'Average_Runtime': avg_runtime,
                    'Selected_Features': integrated_features
                })

    # Feature Selection
    for alg_name, method in feature_selection_algorithms.items():
        runtimes = []
        selected_features_runs = []
        print(f"Running {alg_name}")
        for i in range(1, 11):
            start_time = time.time()
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            selected_features = []
            if method == 'mutual_info':
                mi = mutual_info_classif(X_scaled, y)
                # Select features for each dimension
                for n_features in dimensions:
                    top_indices = np.argsort(mi)[-n_features:]
                    top_features = X.columns[top_indices].tolist()
                    selected_features_runs.append(top_features)
            elif method == 'lasso':
                lasso = Lasso(alpha=0.01, max_iter=10000, random_state=i)
                lasso.fit(X_scaled, y)
                coef = lasso.coef_
                selected_features = X.columns[coef != 0].tolist()
                selected_features_runs.append(selected_features)
            elif method == 'logistic_l1':
                log_reg = LogisticRegression(
                    penalty='l1',
                    solver='liblinear',
                    C=0.1,
                    max_iter=10000,
                    random_state=i
                )
                log_reg.fit(X_scaled, y)
                coef = log_reg.coef_.flatten()
                selected_features = X.columns[coef != 0].tolist()
                selected_features_runs.append(selected_features)
            elif method == 'relieff':
                relief = ReliefF(n_neighbors=10)
                relief.fit(X_scaled, y)
                # Select features for each dimension
                for n_features in dimensions:
                    top_indices = np.argsort(relief.feature_importances_)[-n_features:]
                    top_features = X.columns[top_indices].tolist()
                    selected_features_runs.append(top_features)
            elif method == 'boruta':
                rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=i)
                boruta = BorutaPy(rf, n_estimators='auto', random_state=i)
                boruta.fit(X_scaled, y)
                selected_features = X.columns[boruta.support_].tolist()
                selected_features_runs.append(selected_features)
            runtime = time.time() - start_time
            runtimes.append(runtime)
            print(f"{alg_name} run {i}/10 done: time {runtime:.2f} seconds")
        # Integration of features
        for method in integration_methods:
            integrated_features = integrate_features(selected_features_runs, method=method)
            avg_runtime = np.mean(runtimes)
            modality_results.append({
                'Modality': modality_name,
                'Algorithm': alg_name,
                'Dimension': 'Variable',  # Variable dimensions for feature selection
                'Integration_Method': method,
                'Average_Runtime': avg_runtime,
                'Selected_Features': integrated_features
            })

    results_list.extend(modality_results)

# Main execution
def main():
    exp_df, methy_df, mirna_df, survival_df = load_data()
    data_modalities, common_patients, y = prepare_data(exp_df, methy_df, mirna_df, survival_df)
    dim_reduction_algorithms, feature_selection_algorithms = get_algorithms()
    results_list = []

    for modality_name, modality_df in data_modalities.items():
        # Filter modality data
        modality_df = modality_df[common_patients]
        modality_df = modality_df.loc[:, sorted(modality_df.columns)]
        X = modality_df.transpose()
        X = X.reset_index(drop=True)
        X.columns = X.columns.astype(str)  # Ensure feature names are strings
        apply_algorithms(modality_name, X, y, dim_reduction_algorithms, feature_selection_algorithms, results_list)

    # Save results
    results_df = pd.DataFrame(results_list)
    results_df.to_csv('algorithm_results.csv', index=False)
    print("\nAll results have been saved to 'algorithm_results.csv'.")

if __name__ == "__main__":
    main()