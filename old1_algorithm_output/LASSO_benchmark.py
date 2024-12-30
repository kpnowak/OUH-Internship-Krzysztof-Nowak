import os
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

# Load datasets
exp_df = pd.read_csv('sarcoma/exp', sep=None, engine='python')
methy_df = pd.read_csv('sarcoma/methy', sep=None, engine='python')
mirna_df = pd.read_csv('sarcoma/mirna', sep=None, engine='python')
survival_df = pd.read_csv('sarcoma/survival', sep=None, engine='python')

# Convert patient IDs in survival_df to match the format in exp_df, methy_df, and mirna_df
survival_df['PatientID'] = survival_df['PatientID'].str.replace('-', '.')

# Align the datasets by common patient IDs
common_patients = list(set(exp_df.columns).intersection(set(methy_df.columns)).intersection(set(mirna_df.columns)))

# Filter survival data for common patients
survival_filtered = survival_df[survival_df['PatientID'].isin(common_patients)]
y = survival_filtered['Death']

# Combine datasets by common patients
X_combined = pd.concat([exp_df[common_patients].T, methy_df[common_patients].T, mirna_df[common_patients].T], axis=1)

# Ensure that X_combined is aligned with survival_filtered
X_combined = X_combined.loc[survival_filtered['PatientID']]

# Standardize the features if there are valid rows
if X_combined.shape[0] > 0:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    # Apply LASSO with cross-validation to find the best lambda (alpha)
    lasso = LassoCV(cv=5, random_state=1).fit(X_scaled, y)

    # Extract the features with non-zero coefficients (selected by LASSO)
    selected_features = X_combined.columns[(lasso.coef_ != 0)]

    # Show the selected features
    print("Selected Features:", selected_features)

    # Create the output directory if it doesn't exist
    output_dir = 'LASSO'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the selected features to a file in the 'LASSO' folder
    output_path = os.path.join(output_dir, 'output.txt')
    with open(output_path, 'w') as f:
        for feature in selected_features:
            f.write(f"{feature}\n")
    
    print(f"Selected features saved to {output_path}")
else:
    print("Error: No samples available after alignment.")