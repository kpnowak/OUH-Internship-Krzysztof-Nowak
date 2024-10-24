import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

# Load datasets
exp_df = pd.read_csv('sarcoma/exp', sep=None, engine='python')
methy_df = pd.read_csv('sarcoma/methy', sep=None, engine='python')
mirna_df = pd.read_csv('sarcoma/mirna', sep=None, engine='python')
survival_df = pd.read_csv('sarcoma/survival', sep=None, engine='python')

# Debug: Check the structure of the survival dataframe
print("Survival DataFrame:")
print(survival_df.head())

# Ensure alignment between survival data and combined datasets by filtering common patients
common_patients = list(set(exp_df.columns).intersection(set(methy_df.columns)).intersection(set(mirna_df.columns)))

# Debug: Check the common patients
print("Number of common patients:", len(common_patients))
print("Common Patients List:", common_patients[:5])  # Display the first 5 common patients for verification

# Filter survival data for common patients
survival_filtered = survival_df[survival_df['PatientID'].isin(common_patients)]
y = survival_filtered['Death']

# Debug: Check if we have filtered survival data properly
print("Survival Data after filtering:")
print(survival_filtered.head())

# Combine datasets by common patients
X_combined = pd.concat([exp_df[common_patients].T, methy_df[common_patients].T, mirna_df[common_patients].T], axis=1)

# Debug: Check the structure of the combined dataset
print("Combined Dataset:")
print(X_combined.head())

# Ensure that the number of rows in X_combined matches y
X_combined = X_combined.loc[survival_filtered['PatientID']]

# Debug: Check the shape of X_combined after alignment
print("Shape of X_combined after alignment:", X_combined.shape)

# Standardize features
if X_combined.shape[0] > 0:  # Check if there are samples to process
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    # Calculate mutual information (relevance)
    mi = mutual_info_classif(X_scaled, y, random_state=1)
    feature_relevance = pd.Series(mi, index=X_combined.columns)

    # MRMR feature selection
    selected_features = []
    n_features_to_select = 20
    for _ in range(n_features_to_select):
        if len(selected_features) == 0:
            next_feature = feature_relevance.idxmax()
        else:
            redundancy = X_combined[selected_features].corrwith(X_combined).abs().mean(axis=1)
            mrmr_score = feature_relevance - redundancy
            next_feature = mrmr_score.idxmax()
        
        selected_features.append(next_feature)
        feature_relevance.drop(next_feature, inplace=True)

    # Print selected features
    print("Selected Features:", selected_features)
else:
    print("No samples available for scaling and MRMR feature selection.")