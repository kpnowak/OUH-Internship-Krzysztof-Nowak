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
        'PCA': PCA(n_components=10),
        'NMF': NMF(n_components=10, max_iter=1000, init='nndsvda'),
        'LDA': LDA(),
        'ICA': FastICA(n_components=10, max_iter=2000, tol=0.001),
        't-SNE': TSNE(n_components=2, perplexity=30)
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

# Apply algorithms
def apply_algorithms(modality_name, X, y, dim_reduction_algorithms, feature_selection_algorithms, results_list):
    print(f"\nProcessing modality: {modality_name}")
    modality_results = []

    # Feature Extraction
    for alg_name, alg in dim_reduction_algorithms.items():
        runtimes = []
        selected_features_list = []
        print(f"Running {alg_name}")
        for i in range(1, 11):
            start_time = time.time()
            # Set random state if applicable
            if hasattr(alg, 'random_state'):
                alg.random_state = i
            # Scaling
            if alg_name == 'NMF':
                scaler = MinMaxScaler()
            else:
                scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            # Algorithm-specific processing
            if alg_name == 'LDA':
                n_components = min(len(np.unique(y)) - 1, X_scaled.shape[1])
                alg.n_components = n_components
                X_transformed = alg.fit_transform(X_scaled, y)
                # Extract top features
                feature_importances = np.abs(alg.coef_).flatten()
                top_features = X.columns[np.argsort(feature_importances)[-10:]].tolist()
            elif alg_name == 't-SNE':
                X_transformed = alg.fit_transform(X_scaled)
                top_features = []  # t-SNE does not provide feature importances
            else:
                X_transformed = alg.fit_transform(X_scaled)
                # Extract top features
                components = alg.components_
                if alg_name in ['PCA', 'ICA']:
                    feature_importances = np.sum(np.abs(components), axis=0)
                elif alg_name == 'NMF':
                    feature_importances = np.sum(components, axis=0)
                top_features = X.columns[np.argsort(feature_importances)[-10:]].tolist()
            selected_features_list.extend(top_features)
            runtime = time.time() - start_time
            runtimes.append(runtime)
            print(f"{alg_name} run {i}/10 done: time {runtime:.2f} seconds")
        # Record results
        avg_runtime = np.mean(runtimes)
        if selected_features_list:
            feature_counts = pd.Series(selected_features_list).value_counts()
            top_features_overall = feature_counts.head(10).index.tolist()
        else:
            top_features_overall = []
        modality_results.append({
            'Modality': modality_name,
            'Algorithm': alg_name,
            'Average_Runtime': avg_runtime,
            'Selected_Features': top_features_overall
        })

    # Feature Selection
    for alg_name, method in feature_selection_algorithms.items():
        runtimes = []
        print(f"Running {alg_name}")
        selected_features_list = []
        for i in range(1, 11):
            start_time = time.time()
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            selected_features = []
            if method == 'mutual_info':
                mi = mutual_info_classif(X_scaled, y)
                selected_features = X.columns[np.argsort(mi)[-10:]].tolist()
            elif method == 'lasso':
                lasso = Lasso(alpha=0.01, max_iter=10000, random_state=i)
                lasso.fit(X_scaled, y)
                coef = lasso.coef_
                selected_features = X.columns[coef != 0].tolist()
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
            elif method == 'relieff':
                relief = ReliefF(n_neighbors=10)
                relief.fit(X_scaled, y)
                selected_features = X.columns[np.argsort(relief.feature_importances_)[-10:]].tolist()
            elif method == 'boruta':
                rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=i)
                boruta = BorutaPy(rf, n_estimators='auto', random_state=i)
                boruta.fit(X_scaled, y)
                selected_features = X.columns[boruta.support_].tolist()
            # RFECV chere
            selected_features_list.extend(selected_features)
            runtime = time.time() - start_time
            runtimes.append(runtime)
            print(f"{alg_name} run {i}/10 done: time {runtime:.2f} seconds")
        avg_runtime = np.mean(runtimes)
        if selected_features_list:
            feature_counts = pd.Series(selected_features_list).value_counts()
            top_features = feature_counts.head(10).index.tolist()
        else:
            top_features = []
        modality_results.append({
            'Modality': modality_name,
            'Algorithm': alg_name,
            'Average_Runtime': avg_runtime,
            'Selected_Features': top_features
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