import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load datasets
exp_df = pd.read_csv('sarcoma/exp', sep=None, engine='python')
methy_df = pd.read_csv('sarcoma/methy', sep=None, engine='python')
mirna_df = pd.read_csv('sarcoma/mirna', sep=None, engine='python')
survival_df = pd.read_csv('sarcoma/survival', sep=None, engine='python')

# Step 1: Convert patient IDs in survival_df to match the format in exp_df, methy_df, and mirna_df
survival_df['PatientID'] = survival_df['PatientID'].str.replace('-', '.')

# Step 2: Find the common patient IDs across all datasets
common_patients = list(set(exp_df.columns).intersection(set(methy_df.columns)).intersection(set(mirna_df.columns)))

# Step 3: Filter the survival dataset to include only common patients
survival_filtered = survival_df[survival_df['PatientID'].isin(common_patients)]
y = survival_filtered['Death']

# Step 4: Combine the datasets (expression, methylation, miRNA) using the common patient IDs
X_combined = pd.concat([exp_df[common_patients].T, methy_df[common_patients].T, mirna_df[common_patients].T], axis=1)

# Ensure that X_combined is aligned with survival_filtered
X_combined = X_combined.loc[survival_filtered['PatientID']]

# Fixing index alignment: reset indices to ensure proper alignment
X_combined = X_combined.reset_index(drop=True)
y = y.reset_index(drop=True)

# Step 5: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

# Step 6: Apply PCA feature extraction
n_components = 10  # Set the number of principal components to keep
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

# Step 7: Print the explained variance ratio for each principal component
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained Variance Ratio for each component:", explained_variance_ratio)
print("Total explained variance by selected components:", sum(explained_variance_ratio))

# Save the PCA-transformed data
output_dir = 'PCA'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_path = os.path.join(output_dir, 'pca_output.csv')
pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
pca_df['Death'] = y.reset_index(drop=True)
pca_df.to_csv(output_path, index=False)

print(f"PCA-transformed data saved to {output_path}")