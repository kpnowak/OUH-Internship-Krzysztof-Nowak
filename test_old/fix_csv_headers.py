#!/usr/bin/env python3
"""
Script to fix misspelled headers in CSV files.
This corrects 'Extracttor' -> 'Extractor' and 'Selecttor' -> 'Selector'.
"""

import os
import pandas as pd
import glob
import csv

def fix_csv_headers(directory):
    """
    Find all CSV files in the directory and fix misspelled headers.
    
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
                
                # Check if columns need to be renamed
                columns_renamed = False
                
                if 'Extracttor' in df.columns:
                    df.rename(columns={'Extracttor': 'Extractor'}, inplace=True)
                    columns_renamed = True
                    print(f"  Fixed 'Extracttor' -> 'Extractor'")
                    
                if 'Selecttor' in df.columns:
                    df.rename(columns={'Selecttor': 'Selector'}, inplace=True)
                    columns_renamed = True
                    print(f"  Fixed 'Selecttor' -> 'Selector'")
                
                # Save file if changes were made
                if columns_renamed:
                    # Create backup
                    backup_path = f"{file_path}.bak"
                    os.rename(file_path, backup_path)
                    
                    # Save corrected file
                    df.to_csv(file_path, index=False)
                    print(f"  Saved corrected file (backup at {backup_path})")
                else:
                    print(f"  No changes needed")
                    
            except pd.errors.ParserError as e:
                print(f"  Error with pandas parser: {str(e)}")
                print(f"  Trying manual CSV processing...")
                
                # Manual CSV processing for files with inconsistent number of fields
                with open(file_path, 'r', newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    header = next(reader)
                    rows = list(reader)
                
                # Check if columns need to be renamed
                columns_renamed = False
                
                if 'Extracttor' in header:
                    header[header.index('Extracttor')] = 'Extractor'
                    columns_renamed = True
                    print(f"  Fixed 'Extracttor' -> 'Extractor'")
                    
                if 'Selecttor' in header:
                    header[header.index('Selecttor')] = 'Selector'
                    columns_renamed = True
                    print(f"  Fixed 'Selecttor' -> 'Selector'")
                
                # Save file if changes were made
                if columns_renamed:
                    # Create backup
                    backup_path = f"{file_path}.bak"
                    os.rename(file_path, backup_path)
                    
                    # Save corrected file
                    with open(file_path, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(header)
                        writer.writerows(rows)
                    print(f"  Saved corrected file (backup at {backup_path})")
                else:
                    print(f"  No changes needed")
                
        except Exception as e:
            print(f"  Error processing {file_path}: {str(e)}")

if __name__ == "__main__":
    # Fix headers in both regression and classification output directories
    for directory in ["output_regression", "output_classification", "output_algorithm_multicore"]:
        if os.path.exists(directory):
            print(f"\nProcessing directory: {directory}")
            fix_csv_headers(directory)
        else:
            print(f"\nDirectory not found: {directory}")
    
    print("\nHeader correction complete!") 