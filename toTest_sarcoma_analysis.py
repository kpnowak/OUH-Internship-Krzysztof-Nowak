# Import necessary libraries
import os
import pandas as pd
import numpy as np
import ast  # For parsing string representation of lists
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.decomposition import PCA, NMF, FastICA
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, r2_score
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC, SVR
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
import gseapy as gp
import networkx as nx
import requests

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Define paths to data files (update these paths as necessary)
data_dir = 'sarcoma'  # Directory containing your data files

# Create directories to save outputs
output_dir = 'output'
plots_dir = os.path.join(output_dir, 'plots')
data_output_dir = os.path.join(output_dir, 'data')

os.makedirs(plots_dir, exist_ok=True)
os.makedirs(data_output_dir, exist_ok=True)

# Load the data files
def load_data():
    # Survival data
    survival_df = pd.read_csv(
        os.path.join(data_dir, 'survival'),
        sep='\t',
        header=0
    )

    # Gene Expression data
    exp_df = pd.read_csv(
        os.path.join(data_dir, 'exp'),
        sep=' ',
        header=0,
        index_col=0  # Assuming gene names are in the first column
    )

    # Clean up sample IDs in exp_df
    exp_df.columns = exp_df.columns.str.replace('"', '').str.replace('.', '-').str.strip().str.upper()

    # Transpose the DataFrame so that samples are rows and genes are columns
    exp_df = exp_df.T

    # Convert data to numeric, handling any errors
    exp_df = exp_df.apply(pd.to_numeric, errors='coerce')

    # Set index name
    exp_df.index.name = 'SampleID'

    # Methylation data
    methy_df = pd.read_csv(
        os.path.join(data_dir, 'methy'),
        sep=' ',
        header=0,
        index_col=0  # Assuming probe IDs are in the first column
    )

    # Clean up sample IDs in methy_df
    methy_df.columns = methy_df.columns.str.replace('"', '').str.replace('.', '-').str.strip().str.upper()

    # Transpose the DataFrame so that samples are rows and probes are columns
    methy_df = methy_df.T

    # Convert data to numeric, handling any errors
    methy_df = methy_df.apply(pd.to_numeric, errors='coerce')

    # Set index name
    methy_df.index.name = 'SampleID'

    # miRNA data
    mirna_df = pd.read_csv(
        os.path.join(data_dir, 'mirna'),
        sep=' ',
        header=0,
        index_col=0  # Assuming miRNA IDs are in the first column
    )

    # Clean up sample IDs in mirna_df
    mirna_df.columns = mirna_df.columns.str.replace('"', '').str.replace('.', '-').str.strip().str.upper()

    # Transpose the DataFrame so that samples are rows and miRNAs are columns
    mirna_df = mirna_df.T

    # Convert data to numeric, handling any errors
    mirna_df = mirna_df.apply(pd.to_numeric, errors='coerce')

    # Set index name
    mirna_df.index.name = 'SampleID'

    # Data cleaning: Remove double quotes and set index in survival_df
    survival_df.columns = survival_df.columns.str.replace('"', '')
    survival_df['PatientID'] = survival_df['PatientID'].astype(str).str.replace('"', '').str.strip().str.upper()
    survival_df['Survival'] = pd.to_numeric(survival_df['Survival'].astype(str).str.replace('"', ''), errors='coerce')
    survival_df['Death'] = pd.to_numeric(survival_df['Death'].astype(str).str.replace('"', ''), errors='coerce')

    return exp_df, methy_df, mirna_df, survival_df

# Preprocess data: Align samples and create labels
def preprocess_data(X_df, survival_df):
    # Standardize Patient IDs in survival_df
    survival_df['PatientID'] = survival_df['PatientID'].astype(str).str.strip().str.upper()

    # Optionally, slice IDs to a common length if necessary
    X_df.index = X_df.index.str[:12]
    survival_df['PatientID'] = survival_df['PatientID'].str[:12]

    # Merge with survival data
    merged_df = X_df.merge(
        survival_df,
        how='inner',
        left_index=True,
        right_on='PatientID'
    )

    # Check if merged_df is empty
    if merged_df.empty:
        print("Warning: The merged DataFrame is empty. No matching IDs found.")
        print("Please check the ID formats in your data.")
        exit()

    # Create labels (binary classification: Death)
    y_classification = merged_df['Death'].astype(int)
    # For regression tasks (predicting Survival time)
    y_regression = merged_df['Survival']

    X = merged_df.drop(columns=['PatientID', 'Survival', 'Death'])

    return X, y_classification, y_regression

# Statistical analysis
def statistical_analysis(X, y, selected_features, modality_name, algorithm):
    print(f"\nStatistical Analysis for {modality_name} Modality using {algorithm}:")
    results = []
    for feature in selected_features:
        if feature in X.columns:
            group0 = X[y == 0][feature]
            group1 = X[y == 1][feature]
            stat, p_value = ttest_ind(group0, group1, equal_var=False, nan_policy='omit')
            results.append({
                'Feature': feature,
                'Statistic': stat,
                'P-value': p_value
            })
        else:
            print(f"Feature {feature} not found in X columns.")
    if results:
        results_df = pd.DataFrame(results)
        corrected_pvals = multipletests(results_df['P-value'], method='fdr_bh')[1]
        results_df['Adjusted_P-value'] = corrected_pvals
        results_df['Significant'] = results_df['Adjusted_P-value'] < 0.05
        results_df = results_df.sort_values('Adjusted_P-value')

        # Save results to CSV
        results_df.to_csv(os.path.join(data_output_dir, f'statistical_analysis_{modality_name}_{algorithm}.csv'), index=False)

        print(results_df)
        return results_df
    else:
        print("No valid features for statistical analysis.")
        return pd.DataFrame()

# Model building and evaluation
def model_building_and_evaluation(X, y_classification, y_regression, selected_features, modality_name, algorithm):
    print(f"\nModel Evaluation for {modality_name} Modality using {algorithm}:")
    X_selected = X[selected_features].dropna(axis=1, how='any')

    if X_selected.empty:
        print("No features available for model building after dropping NaNs.")
        return

    models = [
        ('Linear Regression', LinearRegression(), 'regression'),
        ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42), 'classification'),
        ('SVM', SVC(kernel='linear', probability=True, random_state=42), 'classification')
    ]

    for model_name, model, task_type in models:
        print(f"\nUsing {model_name}:")
        if task_type == 'classification':
            clf = model
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            roc_auc_scores = []
            accuracy_scores = []
            for train_idx, test_idx in cv.split(X_selected, y_classification):
                X_train, X_test = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
                y_train, y_test = y_classification.iloc[train_idx], y_classification.iloc[test_idx]
                clf.fit(X_train, y_train)
                y_pred_proba = clf.predict_proba(X_test)[:, 1]
                y_pred = clf.predict(X_test)
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                acc = accuracy_score(y_test, y_pred)
                roc_auc_scores.append(roc_auc)
                accuracy_scores.append(acc)
            print(f"Average ROC-AUC: {np.mean(roc_auc_scores):.4f}")
            print(f"Average Accuracy: {np.mean(accuracy_scores):.4f}")

            # Save model performance
            with open(os.path.join(data_output_dir, f'model_performance_{modality_name}_{algorithm}_{model_name}.txt'), 'w') as f:
                f.write(f"Average ROC-AUC: {np.mean(roc_auc_scores):.4f}\n")
                f.write(f"Average Accuracy: {np.mean(accuracy_scores):.4f}\n")
        elif task_type == 'regression':
            reg = model
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            mse_scores = []
            r2_scores = []
            for train_idx, test_idx in cv.split(X_selected):
                X_train, X_test = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
                y_train, y_test = y_regression.iloc[train_idx], y_regression.iloc[test_idx]
                reg.fit(X_train, y_train)
                y_pred = reg.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                mse_scores.append(mse)
                r2_scores.append(r2)
            print(f"Average MSE: {np.mean(mse_scores):.4f}")
            print(f"Average R2 Score: {np.mean(r2_scores):.4f}")

            # Save model performance
            with open(os.path.join(data_output_dir, f'model_performance_{modality_name}_{algorithm}_{model_name}.txt'), 'w') as f:
                f.write(f"Average MSE: {np.mean(mse_scores):.4f}\n")
                f.write(f"Average R2 Score: {np.mean(r2_scores):.4f}\n")

    # Survival analysis (if applicable)
    try:
        survival_data = X_selected.copy()
        survival_data['Survival'] = y_regression
        survival_data['Death'] = y_classification
        cph = CoxPHFitter()
        cph.fit(survival_data, duration_col='Survival', event_col='Death')
        c_index = concordance_index(survival_data['Survival'], -cph.predict_partial_hazard(survival_data), survival_data['Death'])
        print(f"Cox Model Concordance Index: {c_index:.4f}")

        # Save survival analysis results
        with open(os.path.join(data_output_dir, f'survival_analysis_{modality_name}_{algorithm}.txt'), 'w') as f:
            f.write(f"Cox Model Concordance Index: {c_index:.4f}\n")
    except Exception as e:
        print("Survival analysis could not be performed:", e)

# Data visualization
def visualize_results(X, y_classification, selected_features, modality_name, algorithm, tsne_results=None):
    print(f"\nVisualization for {modality_name} Modality using {algorithm}:")
    X_selected = X[selected_features].dropna(axis=1, how='any')

    if X_selected.empty:
        print("No features available for visualization after dropping NaNs.")
        return

    # Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(X_selected.T, cmap='viridis')
    plt.title(f'Heatmap of Selected Features ({modality_name} - {algorithm})')
    plt.xlabel('Samples')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'heatmap_{modality_name}_{algorithm}.png'))
    plt.close()

    # Box Plots
    for feature in selected_features:
        if feature in X_selected.columns:
            plt.figure(figsize=(6, 4))
            sns.boxplot(x=y_classification, y=X_selected[feature])
            plt.title(f'Box Plot of {feature} by Class ({modality_name} - {algorithm})')
            plt.xlabel('Class')
            plt.ylabel('Feature Value')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'boxplot_{modality_name}_{algorithm}_{feature}.png'))
            plt.close()

    # PCA Plot
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_selected)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_classification)
    plt.title(f'PCA Plot ({modality_name} - {algorithm})')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(title='Class')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'pca_{modality_name}_{algorithm}.png'))
    plt.close()

    # t-SNE Plot
    if tsne_results is not None:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=y_classification)
        plt.title(f't-SNE Plot ({modality_name} - {algorithm})')
        plt.xlabel('Dim 1')
        plt.ylabel('Dim 2')
        plt.legend(title='Class')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'tsne_{modality_name}_{algorithm}.png'))
        plt.close()

# Pathway enrichment analysis
def pathway_enrichment(selected_genes, modality_name, algorithm):
    print(f"\nPathway Enrichment Analysis for {modality_name} Modality using {algorithm}:")
    try:
        enr = gp.enrichr(gene_list=selected_genes,
                         gene_sets=['KEGG_2019_Human', 'Reactome_2016'],
                         organism='Human',
                         outdir=None)
        # Save enrichment results
        enr.results.to_csv(os.path.join(data_output_dir, f'pathway_enrichment_{modality_name}_{algorithm}.csv'), index=False)

        # Display top pathways
        for gene_set in enr.results['Gene_set'].unique():
            print(f"\nTop pathways in {gene_set}:")
            top_pathways = enr.results[enr.results['Gene_set'] == gene_set].head(5)
            print(top_pathways[['Term', 'Adjusted P-value', 'Overlap', 'Combined Score']])
    except Exception as e:
        print("Pathway enrichment analysis could not be performed:", e)

# Network analysis
def network_analysis(selected_genes, modality_name, algorithm):
    print(f"\nNetwork Analysis for {modality_name} Modality using {algorithm}:")
    try:
        # Build STRING query
        string_api_url = "https://string-db.org/api"
        output_format = "tsv-no-header"
        method = "network"
        params = {
            "identifiers": "%0d".join(selected_genes),
            "species": 9606,  # Human
            "caller_identity": "YourAppName"
        }
        request_url = "/".join([string_api_url, output_format, method])
        response = requests.post(request_url, data=params)
        # Parse the response
        interactions = []
        for line in response.text.strip().split("\n"):
            l = line.strip().split("\t")
            p1, p2, score = l[2], l[3], float(l[5])
            interactions.append((p1, p2, {'weight': score}))
        # Build the graph
        G = nx.Graph()
        G.add_edges_from(interactions)
        # Plot the network
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G, k=0.5)
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue')
        nx.draw_networkx_edges(G, pos, alpha=0.5)
        nx.draw_networkx_labels(G, pos, font_size=10)
        plt.title(f'Protein-Protein Interaction Network ({modality_name} - {algorithm})')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'network_{modality_name}_{algorithm}.png'))
        plt.close()
    except Exception as e:
        print("Network analysis could not be performed:", e)

# Main execution
if __name__ == "__main__":
    # Load data
    exp_df, methy_df, mirna_df, survival_df = load_data()

    # Preprocess data
    X_exp, y_exp_classification, y_exp_regression = preprocess_data(exp_df, survival_df)
    X_methy, y_methy_classification, y_methy_regression = preprocess_data(methy_df, survival_df)
    X_mirna, y_mirna_classification, y_mirna_regression = preprocess_data(mirna_df, survival_df)

    # Load the algorithm results from CSV
    algorithm_results = pd.read_csv('algorithm_results.csv')

    # Prepare data structures
    modalities = {
        'Gene Expression': (X_exp, y_exp_classification, y_exp_regression),
        'Methylation': (X_methy, y_methy_classification, y_methy_regression),
        'miRNA': (X_mirna, y_mirna_classification, y_mirna_regression)
    }

    # Process each modality and algorithm
    for modality_name, (X, y_classification, y_regression) in modalities.items():
        modality_results = algorithm_results[algorithm_results['Modality'] == modality_name]
        for index, row in modality_results.iterrows():
            algorithm = row['Algorithm']
            avg_runtime = row['Average_Runtime']
            selected_features_str = row['Selected_Features']

            # Parse the selected features string into a list
            selected_features = ast.literal_eval(selected_features_str)

            print(f"\nAlgorithm: {algorithm}")
            print(f"Original Selected Features: {selected_features}")

            # Adjust selected_features and X.columns to match
            if modality_name == 'Gene Expression':
                # Remove IDs from selected_features and X.columns
                selected_features = [feat.split('.')[0] for feat in selected_features]
                X.columns = [str(col).split('.')[0] for col in X.columns]
            elif modality_name == 'Methylation':
                # Methylation feature names may not need adjustment
                pass
            elif modality_name == 'miRNA':
                # Adjust miRNA names if necessary
                selected_features = [feat.replace('hsa.', '') for feat in selected_features]
                X.columns = [str(col).replace('hsa.', '') for col in X.columns]

            print(f"Adjusted Selected Features: {selected_features}")

            # Check for missing features
            missing_features = [feat for feat in selected_features if feat not in X.columns]
            if missing_features:
                print(f"Warning: The following selected features are not found in X.columns:\n{missing_features}")
                # Remove missing features
                selected_features = [feat for feat in selected_features if feat in X.columns]

            if not selected_features:
                print("No valid features found after adjustment. Skipping this algorithm.")
                continue

            # Handle potential duplicates in X.columns
            if X.columns.duplicated().any():
                print("Warning: Duplicate column names found after adjustment. Removing duplicates.")
                X = X.loc[:, ~pd.Index(X.columns).duplicated()]

            # Proceed with analysis
            with open(os.path.join(data_output_dir, f'selected_features_{modality_name}_{algorithm}.txt'), 'w') as f:
                f.write("\n".join(selected_features))

            # Statistical analysis
            results_df = statistical_analysis(X, y_classification, selected_features, modality_name, algorithm)

            # Model building and evaluation
            model_building_and_evaluation(X, y_classification, y_regression, selected_features, modality_name, algorithm)

            # Visualization
            visualize_results(X, y_classification, selected_features, modality_name, algorithm)

            # If gene expression modality, perform pathway and network analysis
            if modality_name == 'Gene Expression':
                pathway_enrichment(selected_features, modality_name, algorithm)
                network_analysis(selected_features, modality_name, algorithm)