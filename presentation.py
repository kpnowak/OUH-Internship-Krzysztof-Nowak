#!/usr/bin/env python3

import os
import pandas as pd

def main():
    # You can modify these if needed, or discover subfolders automatically.
    regression_datasets = ["AML", "Sarcoma"]
    classification_datasets = ["Breast", "Colon", "Kidney", "Liver", "Lung", "Melanoma", "Ovarian"]
    
    # Base directories where the metrics are stored:
    base_reg_dir = "output_regression"
    base_clf_dir = "output_classification"
    
    all_dfs = []
    
    # 1) Gather Regression CSVs
    for ds in regression_datasets:
        metrics_dir = os.path.join(base_reg_dir, ds, "metrics")
        if not os.path.isdir(metrics_dir):
            continue
        # Collect all CSV files that end with _metrics.csv
        for fname in os.listdir(metrics_dir):
            if fname.endswith("_metrics.csv"):
                full_path = os.path.join(metrics_dir, fname)
                df = pd.read_csv(full_path)
                # Optionally add columns for clarity if you want:
                df["Task_Type"] = "Regression"
                df["Dataset_Name"] = ds
                df["Metrics_File"] = fname
                all_dfs.append(df)
    
    # 2) Gather Classification CSVs
    for ds in classification_datasets:
        metrics_dir = os.path.join(base_clf_dir, ds, "metrics")
        if not os.path.isdir(metrics_dir):
            continue
        # Collect all CSV files that end with _metrics.csv
        for fname in os.listdir(metrics_dir):
            if fname.endswith("_metrics.csv"):
                full_path = os.path.join(metrics_dir, fname)
                df = pd.read_csv(full_path)
                # Optionally add columns for clarity if you want:
                df["Task_Type"] = "Classification"
                df["Dataset_Name"] = ds
                df["Metrics_File"] = fname
                all_dfs.append(df)
    
    # 3) Concatenate everything into one DataFrame
    if not all_dfs:
        print("No metrics files found. Please ensure your output folders contain CSVs.")
        return
    big_df = pd.concat(all_dfs, axis=0, ignore_index=True)
    
    # 4) Calculate total train time
    # Make sure the Train_Time_Seconds column exists in your CSV files
    if "Train_Time_Seconds" in big_df.columns:
        total_seconds = big_df["Train_Time_Seconds"].sum()
        total_hours = total_seconds / 3600.0
        total_days = total_hours / 24.0
        print("=== TOTAL RUNTIME SUMMARY ===")
        print(f"Train_Time_Seconds (sum): {total_seconds:,.2f} s")
        print(f"In hours: {total_hours:,.2f} h")
        print(f"In days:  {total_days:,.2f} d")
    else:
        print("WARNING: No 'Train_Time_Seconds' column found. Cannot sum up total run time.")
    
    # 5) Save one single CSV file with all results
    out_csv = "all_results_combined.csv"
    big_df.to_csv(out_csv, index=False)
    print(f"\nAll data saved to: {out_csv}")
    print("Script complete.")


if __name__ == "__main__":
    main()