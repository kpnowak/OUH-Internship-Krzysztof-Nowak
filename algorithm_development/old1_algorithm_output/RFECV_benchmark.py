from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import os

# Load datasets
exp_df = pd.read_csv('sarcoma/exp', sep=None, engine='python')
methy_df = pd.read_csv('sarcoma/methy', sep=None, engine='python')
mirna_df = pd.read_csv('sarcoma/mirna', sep=None, engine='python')
survival_df = pd.read_csv('sarcoma/survival', sep=None, engine='python')

# Step 1: Align Patient IDs
survival_df['PatientID'] = survival_df['PatientID'].str.replace('-', '.')
common_patients = list(set(exp_df.columns).intersection(set(methy_df.columns)).intersection(set(mirna_df.columns)))

# Step 2: Filter for common patients
survival_filtered = survival_df[survival_df['PatientID'].isin(common_patients)]
y = survival_filtered['Death']
X_combined = pd.concat([exp_df[common_patients].T, methy_df[common_patients].T, mirna_df[common_patients].T], axis=1)
X_combined = X_combined.loc[survival_filtered['PatientID']]

# Fixing index alignment
X_combined = X_combined.reset_index(drop=True)
y = y.reset_index(drop=True)

# Step 3: Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

# Step 4: Apply RFECV
rf = RandomForestClassifier(n_estimators=100, random_state=1)
rfecv = RFECV(estimator=rf, step=1, cv=StratifiedKFold(5), scoring='accuracy', n_jobs=-1)
rfecv.fit(X_scaled, y)

# Step 5: Get selected features
selected_features = X_combined.columns[rfecv.support_]

# Step 6: Print and save selected features
print("Selected Features:", selected_features)
output_dir = 'RFECV'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_path = os.path.join(output_dir, 'output.txt')
with open(output_path, 'w') as f:
    for feature in selected_features:
        f.write(f"{feature}\n")

print(f"Selected features saved to {output_path}")