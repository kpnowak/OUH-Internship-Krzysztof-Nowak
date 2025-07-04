import pandas as pd
import numpy as np
import os
import csv
from collections import Counter

def load_dataset_configs():
    """Load dataset configurations manually"""
    configs = {
        'aml': {
            'type': 'regression',
            'outcome_column': 'lab_procedure_bone_marrow_blast_cell_outcome_percent_value'
        },
        'sarcoma': {
            'type': 'regression', 
            'outcome_column': 'pathologic_tumor_length'
        },
        'breast': {
            'type': 'classification',
            'outcome_column': 'pathologic_T'
        },
        'colon': {
            'type': 'classification',
            'outcome_column': 'pathologic_T'
        },
        'kidney': {
            'type': 'classification',
            'outcome_column': 'pathologic_T'
        },
        'liver': {
            'type': 'classification',
            'outcome_column': 'pathologic_T'
        },
        'lung': {
            'type': 'classification',
            'outcome_column': 'pathologic_T'
        },
        'melanoma': {
            'type': 'classification',
            'outcome_column': 'pathologic_T'
        },
        'ovarian': {
            'type': 'classification',
            'outcome_column': 'clinical_stage'
        }
    }
    return configs

def detect_separator_and_load(file_path):
    """Detect separator and load CSV file properly"""
    if not os.path.exists(file_path):
        return None
    
    try:
        # First, try to read the first line to understand the format
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
        
        # Check if it's space-separated with quotes (like our gene expression files)
        if first_line.count('"') > 10:  # Many quoted values suggest space separation
            print(f"  Detected space-separated format with quotes for {file_path}")
            # Use pandas with space separator and handle quotes
            df = pd.read_csv(file_path, sep=' ', quoting=csv.QUOTE_ALL)
            print(f"  Loaded {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        
        # Try different separators
        separators = [',', '\t', ';', '|']
        for sep in separators:
            if sep in first_line:
                print(f"  Detected separator '{sep}' for {file_path}")
                df = pd.read_csv(file_path, sep=sep)
                print(f"  Loaded {df.shape[0]} rows, {df.shape[1]} columns")
                return df
        
        # Default to comma if no separator detected
        print(f"  No separator detected, trying comma for {file_path}")
        df = pd.read_csv(file_path)
        print(f"  Loaded {df.shape[0]} rows, {df.shape[1]} columns")
        return df
        
    except Exception as e:
        print(f"  Error loading {file_path}: {e}")
        return None

def calculate_missing_percentage(df, exclude_first_col=True):
    """Calculate missing data percentage - includes both NaN and 0 values"""
    if df is None or df.empty:
        return 0.0
    
    # Exclude first column (usually ID) from missing calculation
    data_cols = df.iloc[:, 1:] if exclude_first_col and df.shape[1] > 1 else df
    
    if data_cols.empty:
        return 0.0
    
    # Calculate missing percentage - count both NaN and 0 values as missing
    total_values = data_cols.size
    nan_values = data_cols.isnull().sum().sum()
    zero_values = (data_cols == 0).sum().sum()
    missing_values = nan_values + zero_values
    
    return (missing_values / total_values) * 100 if total_values > 0 else 0.0

def analyze_data_imbalance(clinical_df, outcome_column, dataset_type, dataset_name=None):
    """Analyze data imbalance for regression or classification"""
    if clinical_df is None or outcome_column not in clinical_df.columns:
        return "No outcome data available"
    
    outcome_data = clinical_df[outcome_column].dropna()
    
    if outcome_data.empty:
        return "No valid outcome values"
    
    if dataset_type == 'regression':
        # For regression, create bins and show ranges with counts
        try:
            outcome_numeric = pd.to_numeric(outcome_data, errors='coerce').dropna()
            if outcome_numeric.empty:
                return "No numeric outcome values"
            
            # Define custom bin boundaries for specific datasets
            if dataset_name and dataset_name.lower() == 'aml':
                bin_edges = [0, 20, 40, 60, 80, 100]
            elif dataset_name and dataset_name.lower() == 'sarcoma':
                bin_edges = [1, 8, 16, 24, 32, 40]
            else:
                # Default: Create 5 equal-width bins
                bins = pd.cut(outcome_numeric, bins=5, include_lowest=True)
                bin_counts = bins.value_counts().sort_index()
                
                # Format as "range: count"
                ranges = []
                for interval, count in bin_counts.items():
                    left = round(interval.left, 2)
                    right = round(interval.right, 2)
                    ranges.append(f"[{left}-{right}]: {count}")
                
                return "; ".join(ranges)
            
            # Use custom bin edges for AML and Sarcoma
            bins = pd.cut(outcome_numeric, bins=bin_edges, include_lowest=True, right=False)
            bin_counts = bins.value_counts().sort_index()
            
            # Format as "range: count"
            ranges = []
            for interval, count in bin_counts.items():
                left = int(interval.left)
                right = int(interval.right)
                ranges.append(f"[{left}-{right}]: {count}")
            
            return "; ".join(ranges)
        except Exception as e:
            return f"Error in regression analysis: {e}"
    
    else:  # classification
        # For classification, show class counts
        class_counts = outcome_data.value_counts()
        class_info = [f"{cls}: {count}" for cls, count in class_counts.items()]
        return "; ".join(class_info)

def get_file_size_mb(file_path):
    """Get file size in MB"""
    if os.path.exists(file_path):
        size_bytes = os.path.getsize(file_path)
        return round(size_bytes / (1024 * 1024), 2)
    return 0

def analyze_dataset(dataset_name, config):
    """Analyze a single dataset"""
    print(f"\n=== Analyzing {dataset_name.upper()} ===")
    
    # File paths
    base_path = f"data/{dataset_name}"
    clinical_path = f"data/clinical/{dataset_name}.csv"
    exp_path = f"{base_path}/exp.csv"
    mirna_path = f"{base_path}/mirna.csv"
    methy_path = f"{base_path}/methy.csv"
    
    # Initialize results
    result = {
        'Dataset': dataset_name.upper(),
        'Type': config['type'],
        'Outcome_Column': config['outcome_column']
    }
    
    # Load clinical data
    print(f"Loading clinical data from {clinical_path}")
    print(f"  File size: {get_file_size_mb(clinical_path)} MB")
    clinical_df = detect_separator_and_load(clinical_path)
    
    if clinical_df is not None:
        result['Clinical_Samples'] = len(clinical_df)
        print(f"  Clinical samples: {len(clinical_df)}")
        
        # Analyze data imbalance
        imbalance = analyze_data_imbalance(clinical_df, config['outcome_column'], config['type'], dataset_name)
        result['Data_Imbalance'] = imbalance
        print(f"  Data imbalance: {imbalance}")
    else:
        result['Clinical_Samples'] = 0
        result['Data_Imbalance'] = "No clinical data"
    
    # Analyze each modality
    modalities = [
        ('Gene Expression', exp_path, 'GeneExp'),
        ('miRNA', mirna_path, 'miRNA'), 
        ('Methylation', methy_path, 'Methylation')
    ]
    
    # Store data for combined missing calculation
    all_data_values = []
    
    for modality_name, file_path, prefix in modalities:
        print(f"\nAnalyzing {modality_name} from {file_path}")
        print(f"  File size: {get_file_size_mb(file_path)} MB")
        
        df = detect_separator_and_load(file_path)
        
        if df is not None:
            # Samples = number of columns (excluding first column which is usually gene/feature ID)
            samples = df.shape[1] - 1 if df.shape[1] > 1 else 0
            # Features = number of rows (genes/miRNAs/CpG sites)
            features = df.shape[0]
            
            result[f'{prefix}_Samples'] = samples
            result[f'{prefix}_Features'] = features
            
            # Calculate missing percentage (excluding first column)
            missing_pct = calculate_missing_percentage(df, exclude_first_col=True)
            result[f'{prefix}_Missing_Pct'] = round(missing_pct, 2)
            
            print(f"  Samples: {samples}, Features: {features}")
            print(f"  Missing percentage (zeros + NaNs): {missing_pct:.2f}%")
            
            # Collect data for combined missing calculation (excluding first column)
            data_portion = df.iloc[:, 1:] if df.shape[1] > 1 else df
            if not data_portion.empty:
                all_data_values.extend(data_portion.values.flatten())
        else:
            result[f'{prefix}_Samples'] = 0
            result[f'{prefix}_Features'] = 0
            result[f'{prefix}_Missing_Pct'] = 0.0
            print(f"  No data available")
    
    # Calculate combined missing percentage across all modalities
    if all_data_values:
        total_values = len(all_data_values)
        # Count NaN and 0 values
        nan_count = sum(1 for x in all_data_values if pd.isna(x))
        zero_count = sum(1 for x in all_data_values if not pd.isna(x) and x == 0)
        missing_count = nan_count + zero_count
        
        combined_missing = (missing_count / total_values) * 100
        result['Combined_Missing_Pct'] = round(combined_missing, 2)
        print(f"\nCombined missing percentage (zeros + NaNs): {combined_missing:.2f}%")
        print(f"  Total values: {total_values:,}, Missing: {missing_count:,} (NaNs: {nan_count:,}, Zeros: {zero_count:,})")
    else:
        result['Combined_Missing_Pct'] = 0.0
    
    return result

def main():
    """Main function to create the summary table"""
    print("Creating raw data characteristics summary...")
    
    # Load configurations
    configs = load_dataset_configs()
    
    # Analyze all datasets
    results = []
    for dataset_name, config in configs.items():
        try:
            result = analyze_dataset(dataset_name, config)
            results.append(result)
        except Exception as e:
            print(f"Error analyzing {dataset_name}: {e}")
            # Add empty result to maintain structure
            results.append({
                'Dataset': dataset_name.upper(),
                'Type': config['type'],
                'Outcome_Column': config['outcome_column'],
                'Clinical_Samples': 0,
                'GeneExp_Samples': 0,
                'GeneExp_Features': 0,
                'GeneExp_Missing_Pct': 0.0,
                'miRNA_Samples': 0,
                'miRNA_Features': 0,
                'miRNA_Missing_Pct': 0.0,
                'Methylation_Samples': 0,
                'Methylation_Features': 0,
                'Methylation_Missing_Pct': 0.0,
                'Combined_Missing_Pct': 0.0,
                'Data_Imbalance': f"Error: {e}"
            })
    
    # Create DataFrame and save to CSV
    df_results = pd.DataFrame(results)
    
    # Reorder columns for better presentation
    column_order = [
        'Dataset', 'Type', 'Outcome_Column', 'Clinical_Samples',
        'GeneExp_Samples', 'GeneExp_Features', 'GeneExp_Missing_Pct',
        'miRNA_Samples', 'miRNA_Features', 'miRNA_Missing_Pct',
        'Methylation_Samples', 'Methylation_Features', 'Methylation_Missing_Pct',
        'Combined_Missing_Pct', 'Data_Imbalance'
    ]
    
    df_results = df_results[column_order]
    
    # Save to CSV
    output_file = 'raw_data_characteristics_final.csv'
    df_results.to_csv(output_file, index=False)
    
    print(f"\n=== SUMMARY COMPLETE ===")
    print(f"Results saved to: {output_file}")
    print(f"Total datasets analyzed: {len(results)}")
    print(f"Regression datasets: {sum(1 for r in results if r['Type'] == 'regression')}")
    print(f"Classification datasets: {sum(1 for r in results if r['Type'] == 'classification')}")
    
    # Display the results
    print(f"\nPreview of results:")
    print(df_results.to_string(index=False))

if __name__ == "__main__":
    main() 