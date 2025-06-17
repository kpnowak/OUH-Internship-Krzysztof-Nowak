import pandas as pd
import os

def combine_cv_metrics(selection_file, extraction_file, output_file):
    """
    Combine selection and extraction CV metrics files into one combined file.
    
    Args:
        selection_file (str): Path to the selection CV metrics file
        extraction_file (str): Path to the extraction CV metrics file  
        output_file (str): Path where the combined file will be saved
    """
    
    # Read both CSV files
    print(f"Reading selection metrics from: {selection_file}")
    selection_df = pd.read_csv(selection_file)
    
    print(f"Reading extraction metrics from: {extraction_file}")
    extraction_df = pd.read_csv(extraction_file)
    
    # Combine the dataframes
    print("Combining dataframes...")
    combined_df = pd.concat([selection_df, extraction_df], ignore_index=True)
    
    # Sort by Dataset, Workflow, Algorithm for better organization
    print("Sorting combined data...")
    combined_df = combined_df.sort_values(['Dataset', 'Workflow', 'Algorithm', 'n_features', 'integration_tech', 'Model'])
    
    # Reset index
    combined_df = combined_df.reset_index(drop=True)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Save the combined file
    print(f"Saving combined metrics to: {output_file}")
    combined_df.to_csv(output_file, index=False)
    
    # Print summary statistics
    print("\n=== COMBINATION SUMMARY ===")
    print(f"Selection metrics rows: {len(selection_df)}")
    print(f"Extraction metrics rows: {len(extraction_df)}")
    print(f"Combined metrics rows: {len(combined_df)}")
    print(f"Total columns: {len(combined_df.columns)}")
    
    print("\nWorkflow distribution in combined file:")
    print(combined_df['Workflow'].value_counts())
    
    print("\nAlgorithm distribution in combined file:")
    print(combined_df['Algorithm'].value_counts())
    
    print(f"\nCombined file saved successfully at: {output_file}")
    
    return combined_df

if __name__ == "__main__":
    # Define file paths
    base_dir = "output_main_without_mrmr/Sarcoma/metrics"
    selection_file = os.path.join(base_dir, "Sarcoma_selection_cv_metrics.csv")
    extraction_file = os.path.join(base_dir, "Sarcoma_extraction_cv_metrics.csv")
    output_file = os.path.join(base_dir, "Sarcoma_combined_cv_metrics.csv")
    
    # Check if input files exist
    if not os.path.exists(selection_file):
        print(f"Error: Selection file not found at {selection_file}")
        exit(1)
        
    if not os.path.exists(extraction_file):
        print(f"Error: Extraction file not found at {extraction_file}")
        exit(1)
    
    # Combine the files
    try:
        combined_df = combine_cv_metrics(selection_file, extraction_file, output_file)
        print("\n Successfully combined CV metrics files!")
        
    except Exception as e:
        print(f" Error combining files: {str(e)}")
        exit(1) 