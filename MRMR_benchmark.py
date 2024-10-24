import os
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
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

# Step 5: Standardize the features (if there are valid samples)
if X_combined.shape[0] > 0:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    # Step 6: Calculate mutual information (relevance) between features and target
    mi = mutual_info_classif(X_scaled, y, random_state=1)
    feature_relevance = pd.Series(mi, index=X_combined.columns)

    # Step 7: Initialize MRMR feature selection
    selected_features = []
    n_features_to_select = 20  # You can adjust this number based on the desired number of features

    for _ in range(n_features_to_select):
        if len(selected_features) == 0:
            # First feature is the one with the highest relevance
            next_feature = feature_relevance.idxmax()
        else:
            # Calculate redundancy: correlate the current feature with already selected features
            redundancy = X_combined[selected_features].corrwith(X_combined).abs().mean(axis=0)
            # MRMR criterion: maximize relevance while minimizing redundancy
            mrmr_score = feature_relevance - redundancy
            next_feature = mrmr_score.idxmax()
        
        selected_features.append(next_feature)
        feature_relevance.drop(next_feature, inplace=True)

    # Step 8: Save the selected features to a file in the 'MRMR' folder
    output_dir = 'MRMR'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, 'output.txt')
    with open(output_path, 'w') as f:
        for feature in selected_features:
            f.write(f"{feature}\n")

    print(f"Selected features saved to {output_path}")
else:
    print("Error: No samples available for scaling and MRMR feature selection.")