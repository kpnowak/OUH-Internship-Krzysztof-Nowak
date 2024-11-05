import os
import time
import pandas as pd
from sklearn.decomposition import PCA, NMF, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, mutual_info_classif
from boruta import BorutaPy
from skrebate import ReliefF
from sklearn.model_selection import StratifiedKFold
import numpy as np

# Load datasets
exp_df = pd.read_csv('sarcoma/exp', sep=None, engine='python')
methy_df = pd.read_csv('sarcoma/methy', sep=None, engine='python')
mirna_df = pd.read_csv('sarcoma/mirna', sep=None, engine='python')
survival_df = pd.read_csv('sarcoma/survival', sep=None, engine='python')

# Prepare data
survival_df['PatientID'] = survival_df['PatientID'].str.replace('-', '.')
common_patients = list(set(exp_df.columns).intersection(set(methy_df.columns)).intersection(set(mirna_df.columns)))
survival_filtered = survival_df[survival_df['PatientID'].isin(common_patients)]
y = survival_filtered['Death']
X_combined = pd.concat([exp_df[common_patients].T, methy_df[common_patients].T, mirna_df[common_patients].T], axis=1)
X_combined = X_combined.loc[survival_filtered['PatientID']]
X_combined = X_combined.reset_index(drop=True)
y = y.reset_index(drop=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

# Function to save output with runtime information
def save_output(data, folder_name, run_num, runtime):
    output_dir = os.path.join(folder_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f'output_{run_num}.csv')
    data['Runtime_seconds'] = runtime
    data.to_csv(output_path, index=False)
    print(f"Saved output to {output_path}")

# Running algorithms
for i in range(1, 11):
    """
    # PCA
    start_time = time.time()
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X_scaled)
    runtime = time.time() - start_time
    pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(10)])
    save_output(pca_df, "PCA", i, runtime)

    # NMF
    start_time = time.time()
    nmf = NMF(n_components=10, random_state=i, max_iter=1000)
    X_nmf = nmf.fit_transform(MinMaxScaler().fit_transform(X_combined))
    runtime = time.time() - start_time
    nmf_df = pd.DataFrame(X_nmf, columns=[f'NMF_{j+1}' for j in range(10)])
    save_output(nmf_df, "NMF", i, runtime)

    # LDA
    start_time = time.time()
    lda = LDA(n_components=min(len(y.unique()) - 1, X_scaled.shape[1]))
    X_lda = lda.fit_transform(X_scaled, y)
    runtime = time.time() - start_time
    lda_df = pd.DataFrame(X_lda, columns=[f'LDA_{k+1}' for k in range(X_lda.shape[1])])
    save_output(lda_df, "LDA", i, runtime)

    # ICA
    start_time = time.time()
    ica = FastICA(n_components=10, random_state=i, max_iter=2000, tol=0.001)
    X_ica = ica.fit_transform(X_scaled)
    runtime = time.time() - start_time
    ica_df = pd.DataFrame(X_ica, columns=[f'ICA_{l+1}' for l in range(10)])
    save_output(ica_df, "ICA", i, runtime)

    # t-SNE
    start_time = time.time()
    tsne = TSNE(n_components=2, perplexity=30, random_state=i)
    X_tsne = tsne.fit_transform(X_scaled)
    runtime = time.time() - start_time
    tsne_df = pd.DataFrame(X_tsne, columns=['t-SNE_1', 't-SNE_2'])
    save_output(tsne_df, "tSNE", i, runtime)

    # Feature Selection:
    # MRMR
    start_time = time.time()
    mi = mutual_info_classif(X_scaled, y)
    runtime = time.time() - start_time
    mrmr_df = pd.DataFrame({"Feature": X_combined.columns, "Mutual_Info": mi}).sort_values(by="Mutual_Info", ascending=False).head(10)
    save_output(mrmr_df, "MRMR", i, runtime)

    # LASSO
    start_time = time.time()
    lasso = Lasso(alpha=0.1, random_state=i)
    lasso.fit(X_scaled, y)
    runtime = time.time() - start_time
    lasso_df = pd.DataFrame({"Feature": X_combined.columns, "Coefficient": lasso.coef_}).query("Coefficient != 0")
    save_output(lasso_df, "LASSO", i, runtime)

    # ReliefF
    start_time = time.time()
    relief = ReliefF(n_neighbors=10)
    relief.fit(X_scaled, y)
    runtime = time.time() - start_time
    relief_df = pd.DataFrame({"Feature": X_combined.columns, "Score": relief.feature_importances_}).sort_values(by="Score", ascending=False).head(10)
    save_output(relief_df, "ReliefF", i, runtime)

    # Boruta
    start_time = time.time()
    rf = RandomForestClassifier(n_estimators=100, random_state=i)
    boruta = BorutaPy(rf, n_estimators='auto', random_state=i)
    boruta.fit(X_scaled, y)
    runtime = time.time() - start_time
    boruta_df = pd.DataFrame(X_combined.columns[boruta.support_], columns=["Selected_Features"])
    save_output(boruta_df, "Boruta", i, runtime)

    # RFECV - Standard RFECV with RandomForest
    start_time = time.time()
    rf = RandomForestClassifier(n_estimators=100, random_state=1)
    rfecv = RFECV(estimator=rf, step=1, cv=StratifiedKFold(5), scoring='accuracy', n_jobs=-1)
    rfecv.fit(X_scaled, y)
    runtime = time.time() - start_time
    rfecv_df = pd.DataFrame(X_combined.columns[rfecv.support_], columns=["Selected_Features"])
    save_output(rfecv_df, "RFECV_RandomForest", i, runtime)
"""
    # RFECV - Alternative with Logistic Regression for faster results
    start_time = time.time()
    log_reg = LogisticRegression(max_iter=1000)
    rfecv_alt = RFECV(estimator=log_reg, step=5, cv=3, scoring='accuracy', n_jobs=-1)
    rfecv_alt.fit(X_scaled, y)
    runtime = time.time() - start_time
    rfecv_alt_df = pd.DataFrame(X_combined.columns[rfecv_alt.support_], columns=["Selected_Features"])
    save_output(rfecv_alt_df, "RFECV_LogisticRegression", i, runtime)