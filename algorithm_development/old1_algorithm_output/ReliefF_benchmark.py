import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from skrebate import ReliefF

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

# Step 5: Ensure that X_combined is aligned with survival_filtered
X_combined = X_combined.loc[survival_filtered['PatientID']]

# Fixing index alignment: reset indices to ensure proper alignment
X_combined = X_combined.reset_index(drop=True)
y = y.reset_index(drop=True)

# Step 6: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

# Step 7: Apply ReliefF feature selection
relieff = ReliefF(n_neighbors=10, n_features_to_select=20)  # 10 neighbors, selecting top 20 features
relieff.fit(X_scaled, y)

# Step 8: Extract the selected features
selected_features = X_combined.columns[relieff.top_features_[:20]]

# Step 9: Print and save the selected features
print("Selected Features:", selected_features)

# Save the selected features to a file in the 'ReliefF' folder
output_dir = 'ReliefF'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_path = os.path.join(output_dir, 'output.txt')
with open(output_path, 'w') as f:
    for feature in selected_features:
        f.write(f"{feature}\n")

print(f"Selected features saved to {output_path}")