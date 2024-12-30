import os
import pandas as pd
from sklearn.manifold import TSNE
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
X_combined = X_combined.loc[survival_filtered['PatientID']]

# Fixing index alignment: reset indices to ensure proper alignment
X_combined = X_combined.reset_index(drop=True)
y = y.reset_index(drop=True)

# Step 5: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

# Step 6: Apply t-SNE feature extraction
tsne = TSNE(n_components=2, perplexity=30, random_state=1)  # 2D projection for visualization
X_tsne = tsne.fit_transform(X_scaled)

# Step 7: Save and print the results
print("t-SNE Components Shape:", X_tsne.shape)
output_dir = 'tSNE'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_path = os.path.join(output_dir, 'tsne_output.csv')
tsne_df = pd.DataFrame(X_tsne, columns=['t-SNE_1', 't-SNE_2'])
tsne_df['Death'] = y.reset_index(drop=True)
tsne_df.to_csv(output_path, index=False)

print(f"t-SNE-transformed data saved to {output_path}")