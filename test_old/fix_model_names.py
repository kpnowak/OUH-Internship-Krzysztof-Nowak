#!/usr/bin/env python3
"""
Script to fix inconsistent model names in CSV files.
This normalizes model names to match the normalized names in models.py
"""

import os
import pandas as pd
import glob
from Z_alg.models import get_normalized_model_name

def fix_model_names(directory):
    """
    Find all CSV files in the directory and fix inconsistent model names.
    
    Parameters:
    -----------
    directory : str
        Directory containing CSV files to process
    """
    # Find all CSV files in the directory and subdirectories
    csv_files = glob.glob(f"{directory}/**/*.csv", recursive=True)
    
    for file_path in csv_files:
        print(f"Processing {file_path}...")
        
        try:
            # Try to read with pandas first
            try:
                df = pd.read_csv(file_path)
                
                # Check if the file has a Model column
                if 'Model' in df.columns:
                    # Check if any model names need to be normalized
                    models_changed = False
                    
                    # Create a new column with normalized model names
                    normalized_models = df['Model'].apply(get_normalized_model_name)
                    
                    # Check if any names are different from the original
                    if not (normalized_models == df['Model']).all():
                        # Update the models
                        df['Model'] = normalized_models
                        models_changed = True
                        print(f"  Fixed model names using normalization function")
                    
                    # Save file if changes were made
                    if models_changed:
                        # Create backup
                        backup_path = f"{file_path}.model.bak"
                        os.rename(file_path, backup_path)
                        
                        # Save corrected file
                        df.to_csv(file_path, index=False)
                        print(f"  Saved corrected file (backup at {backup_path})")
                    else:
                        print(f"  No model name changes needed")
                else:
                    print(f"  No 'Model' column found in file")
                    
            except pd.errors.ParserError as e:
                print(f"  Error with pandas parser: {str(e)}")
                print(f"  Trying manual CSV processing...")
                
                # Manual CSV processing for files with inconsistent number of fields
                with open(file_path, 'r', newline='') as csvfile:
                    content = csvfile.read()
                
                # The manual processing is more limited since we can't easily use
                # the function without parsing the CSV correctly
                # We'll use a simple mapping based on the known conversions
                model_mappings = {
                    'RandomForest': 'RandomForestRegressor',
                    'randomforest': 'RandomForestRegressor',
                    'SVM': 'SVC',
                    'svm': 'SVC'
                }
                    
                # Check each model name for replacement
                content_changed = False
                for old_name, new_name in model_mappings.items():
                    if f",{old_name}," in content:
                        content = content.replace(f",{old_name},", f",{new_name},")
                        content_changed = True
                        print(f"  Fixed '{old_name}' -> '{new_name}'")
                
                # Save file if changes were made
                if content_changed:
                    # Create backup
                    backup_path = f"{file_path}.model.bak"
                    os.rename(file_path, backup_path)
                    
                    # Save corrected file
                    with open(file_path, 'w', newline='') as csvfile:
                        csvfile.write(content)
                    print(f"  Saved corrected file (backup at {backup_path})")
                else:
                    print(f"  No model name changes needed")
                
        except Exception as e:
            print(f"  Error processing {file_path}: {str(e)}")

if __name__ == "__main__":
    # Fix model names in both regression and classification output directories
    for directory in ["output_regression", "output_classification", "output_algorithm_multicore"]:
        if os.path.exists(directory):
            print(f"\nProcessing directory: {directory}")
            fix_model_names(directory)
        else:
            print(f"\nDirectory not found: {directory}")
    
    print("\nModel name normalization complete!") 