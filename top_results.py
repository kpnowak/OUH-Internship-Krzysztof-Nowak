import pandas as pd

# Load the CSV data; adjust the file name/path as needed.
df = pd.read_csv('all_results_combined.csv')  # replace with your actual file name or data source

# --- Overall Top 5 Results ---

# Regression: best = lowest RMSE
df_reg = df[df['Task_Type'] == 'Regression'].copy()
df_reg = df_reg.dropna(subset=['RMSE'])
df_reg_sorted = df_reg.sort_values('RMSE')
top5_regression = df_reg_sorted.head(5)

# Classification: best = highest MCC
df_clf = df[df['Task_Type'] == 'Classification'].copy()
df_clf = df_clf.dropna(subset=['MCC'])
df_clf_sorted = df_clf.sort_values('MCC', ascending=False)
top5_classification = df_clf_sorted.head(5)

# Save the overall top 5 results to CSV files
top5_regression.to_csv('top5_regression.csv', index=False)
top5_classification.to_csv('top5_classification.csv', index=False)

print("Saved overall top 5 regression results to 'top5_regression.csv'")
print("Saved overall top 5 classification results to 'top5_classification.csv'")

# --- Top 5 Results for Each Cancer Type ---

# Assume the 'Dataset' column contains the cancer type (e.g., AML, Sarcoma, Breast, etc.)
cancer_types = df['Dataset'].unique()
# Dictionary to store top-5 results per cancer type
top_results_by_cancer = {}

for cancer in cancer_types:
    # Filter rows for the current cancer type
    df_cancer = df[df['Dataset'] == cancer].copy()
    # Determine the task type; assume all rows in the cancer type have the same Task_Type
    task = df_cancer['Task_Type'].iloc[0]
    if task == 'Regression':
        df_cancer = df_cancer.dropna(subset=['RMSE'])
        df_cancer_sorted = df_cancer.sort_values('RMSE')
    elif task == 'Classification':
        df_cancer = df_cancer.dropna(subset=['MCC'])
        df_cancer_sorted = df_cancer.sort_values('MCC', ascending=False)
    else:
        continue  # skip if Task_Type is not recognized
    
    top5 = df_cancer_sorted.head(5)
    top_results_by_cancer[cancer] = top5

    # Save the top-5 results for this cancer type to a CSV file named "top5_<cancer>.csv"
    filename = f"top5_{cancer}.csv"
    top5.to_csv(filename, index=False)
    print(f"Saved top 5 {cancer} {task} results to '{filename}'")

# Optionally, if you want to combine all cancer types into a single CSV with an extra column:
combined = pd.concat(
    [df.assign(Cancer_Type=cancer) for cancer, df in top_results_by_cancer.items()],
    ignore_index=True
)
combined.to_csv("top5_by_cancer.csv", index=False)
print("Saved combined top 5 results by cancer type to 'top5_by_cancer.csv'")