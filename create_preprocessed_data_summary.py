#!/usr/bin/env python3
"""
Preprocessed Data Characteristics Analyzer

This script computes preprocessing for all datasets used in the main pipeline and saves 
the results in a CSV file similar to raw_data_characteristics_final.csv, but with 
additional preprocessing metrics including:

- Sample retention %
- Feature retention % 
- Original variance and scaled variance for each modality
- Quality scores from the 4-phase enhanced preprocessing pipeline

Datasets analyzed:
- Regression: AML, Sarcoma
- Classification: Colon, Breast, Kidney, Liver, Lung, Melanoma, Ovarian

The preprocessing is done using the same 4-phase enhanced pipeline as in the main pipeline.
"""

import pandas as pd
import numpy as np
import os
import csv
import time
import logging
import sys
from collections import Counter
from datetime import datetime

def load_dataset_configs():
    """Load dataset configurations from the main pipeline's config.py"""
    from config import DatasetConfig
    
    # Get all available datasets from the main pipeline configuration
    datasets = ['aml', 'sarcoma', 'breast', 'colon', 'kidney', 'liver', 'lung', 'melanoma', 'ovarian']
    
    configs = {}
    for dataset_name in datasets:
        config = DatasetConfig.get_config(dataset_name.lower())
        if config:
            # Determine task type using same logic as main pipeline
            outcome_col = config.get('outcome_col', '')
            if 'blast' in outcome_col.lower() or 'length' in outcome_col.lower():
                task_type = 'regression'
            else:
                task_type = 'classification'
            
            configs[dataset_name] = {
                'type': task_type,
                'outcome_column': outcome_col,
                'config': config  # Store the full config for reference
            }
    
    return configs

def setup_logging():
    """Setup logging for the preprocessing analysis."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        handlers=[
            logging.FileHandler('preprocessed_data_analysis.log', mode='w'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

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

def calculate_array_missing_percentage(array):
    """Calculate missing data percentage for numpy array - includes both NaN and 0 values"""
    if array is None or array.size == 0:
        return 0.0
    
    total_values = array.size
    nan_values = np.isnan(array).sum()
    zero_values = (array == 0).sum()
    missing_values = nan_values + zero_values
    
    return (missing_values / total_values) * 100 if total_values > 0 else 0.0

def calculate_variance_statistics(array):
    """Calculate variance statistics for an array"""
    if array is None or array.size == 0:
        return 0.0, 0.0
    
    # Calculate feature-wise variances
    if array.ndim == 1:
        return np.var(array), np.var(array)
    elif array.ndim == 2:
        feature_variances = np.var(array, axis=0)
        mean_variance = np.mean(feature_variances)
        return mean_variance, mean_variance
    else:
        return 0.0, 0.0

def analyze_data_imbalance(clinical_df, outcome_column, dataset_type, dataset_name=None):
    """Analyze data imbalance for regression or classification (same as original)"""
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

def analyze_data_imbalance_from_series(y_series, dataset_type, dataset_name=None):
    """Analyze data imbalance directly from a pandas Series (for regression or classification)"""
    if y_series is None or y_series.empty:
        return "No outcome data available"
    
    outcome_data = y_series.dropna()
    
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

def load_raw_dataset(dataset_name, config, logger):
    """Load raw dataset using the EXACT same approach as the main pipeline"""
    try:
        from data_io import load_dataset
        from config import DatasetConfig
        
        logger.info(f"Loading raw dataset: {dataset_name}")
        
        # Get configuration for the dataset (same as main pipeline)
        dataset_config = DatasetConfig.get_config(dataset_name.lower())
        if not dataset_config:
            logger.error(f"No configuration found for dataset: {dataset_name}")
            return None, None, None, None
        
        # Map modality names to short names (EXACT same logic as main pipeline)
        modality_mapping = {
            "Gene Expression": "exp",
            "miRNA": "mirna", 
            "Methylation": "methy"
        }
        
        modality_short_names = []
        for full_name in dataset_config['modalities'].keys():
            short_name = modality_mapping.get(full_name, full_name.lower())
            modality_short_names.append(short_name)
        
        # Determine task type (same logic as main pipeline)
        outcome_col = dataset_config.get('outcome_col', '')
        if 'blast' in outcome_col.lower() or 'length' in outcome_col.lower():
            task_type = 'regression'
        else:
            task_type = 'classification'
        
        logger.info(f"Task type: {task_type}, Outcome column: {outcome_col}")
        logger.info(f"Modalities to load: {modality_short_names}")
        
        # Load dataset using the main pipeline's approach
        # This function already handles clinical data loading with multiple parsing strategies
        modalities_data, y_series, common_ids, is_regression = load_dataset(
            ds_name=dataset_name.lower(),
            modalities=modality_short_names,
            outcome_col=dataset_config['outcome_col'],
            task_type=task_type,
            parallel=True,
            use_cache=True
        )
        
        if not modalities_data or y_series is None:
            logger.error(f"Failed to load dataset: {dataset_name}")
            return None, None, None, None
            
        logger.info(f"Successfully loaded {dataset_name} with {len(common_ids)} common samples")
        logger.info(f"Raw data loaded: {len(modalities_data)} modalities")
        
        return modalities_data, y_series, common_ids, is_regression
        
    except Exception as e:
        logger.error(f"Error loading raw dataset {dataset_name}: {e}")
        return None, None, None, None

def apply_preprocessing_pipeline(modalities_data, y_series, common_ids, dataset_name, task_type, logger):
    """Apply the 4-phase enhanced preprocessing pipeline"""
    try:
        from enhanced_pipeline_integration import run_enhanced_preprocessing_pipeline
        
        logger.info(f"Applying 4-phase enhanced preprocessing pipeline for {dataset_name}")
        
        # Convert to enhanced pipeline format: Dict[str, Tuple[np.ndarray, List[str]]]
        modality_data_dict = {}
        for modality_name, modality_df in modalities_data.items():
            # Convert DataFrame to numpy array (transpose to get samples x features)
            X_modality = modality_df.T.values  # modality_df is features x samples
            modality_data_dict[modality_name] = (X_modality, common_ids)
            logger.info(f"  Raw {modality_name} shape: {X_modality.shape}")
        
        # Apply 4-phase enhanced preprocessing pipeline WITHOUT fusion (feature-first approach)
        logger.info(f"Applying 4-phase preprocessing WITHOUT fusion for feature-first architecture...")
        
        processed_modalities, y_aligned, pipeline_metadata = run_enhanced_preprocessing_pipeline(
            modality_data_dict=modality_data_dict,
            y=y_series.values,
            fusion_method="average",  # Any fusion method - feature_first_order=True returns separate modalities
            task_type=task_type,
            dataset_name=dataset_name,
            enable_early_quality_check=True,
            enable_feature_first_order=True,  # CRITICAL: This returns separate modalities (not fused)
            enable_centralized_missing_data=True,
            enable_coordinated_validation=True
        )
        
        logger.info(f"4-phase preprocessing completed with quality score: {pipeline_metadata.get('quality_score', 'N/A')}")
        logger.info(f"Preprocessed modalities (separate arrays for feature-first):")
        
        for modality_name, modality_array in processed_modalities.items():
            logger.info(f"  Preprocessed {modality_name} shape: {modality_array.shape}")
        
        # Final validation and cleaning (EXACT same as main pipeline)
        for modality_name, X_mod in processed_modalities.items():
            if np.any(np.isnan(X_mod)) or np.any(np.isinf(X_mod)):
                logger.warning(f"Found NaN/Inf values in processed {modality_name}, cleaning...")
                processed_modalities[modality_name] = np.nan_to_num(X_mod, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if np.any(np.isnan(y_aligned)) or np.any(np.isinf(y_aligned)):
            logger.warning("Found NaN/Inf values in target, cleaning...")
            y_aligned = np.nan_to_num(y_aligned, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return processed_modalities, y_aligned, pipeline_metadata
        
    except Exception as e:
        logger.error(f"Failed to apply preprocessing pipeline for {dataset_name}: {e}")
        return None, None, None

def analyze_dataset_with_preprocessing(dataset_name, config, logger):
    """Analyze a single dataset with raw and preprocessed data comparison"""
    logger.info(f"\n=== Analyzing {dataset_name.upper()} with Preprocessing ===")
    
    # Initialize results
    result = {
        'Dataset': dataset_name.upper(),
        'Type': config['type'],
        'Outcome_Column': config['outcome_column']
    }
    
    # Step 1: Load raw dataset
    modalities_data, y_series, common_ids, is_regression = load_raw_dataset(dataset_name, config, logger)
    
    if not modalities_data or y_series is None:
        logger.error(f"Failed to load raw data for {dataset_name}")
        # Return empty result with error indicators
        result.update({
            'Clinical_Samples': 0,
            'GeneExp_Samples_Raw': 0, 'GeneExp_Features_Raw': 0, 'GeneExp_Missing_Pct_Raw': 0.0,
            'miRNA_Samples_Raw': 0, 'miRNA_Features_Raw': 0, 'miRNA_Missing_Pct_Raw': 0.0,
            'Methylation_Samples_Raw': 0, 'Methylation_Features_Raw': 0, 'Methylation_Missing_Pct_Raw': 0.0,
            'Combined_Missing_Pct_Raw': 0.0,
            'GeneExp_Samples_Processed': 0, 'GeneExp_Features_Processed': 0, 'GeneExp_Missing_Pct_Processed': 0.0,
            'miRNA_Samples_Processed': 0, 'miRNA_Features_Processed': 0, 'miRNA_Missing_Pct_Processed': 0.0,
            'Methylation_Samples_Processed': 0, 'Methylation_Features_Processed': 0, 'Methylation_Missing_Pct_Processed': 0.0,
            'Combined_Missing_Pct_Processed': 0.0,
            'Sample_Retention_Pct': 0.0,
            'GeneExp_Feature_Retention_Pct': 0.0, 'miRNA_Feature_Retention_Pct': 0.0, 'Methylation_Feature_Retention_Pct': 0.0,
            'GeneExp_Original_Variance': 0.0, 'GeneExp_Scaled_Variance': 0.0,
            'miRNA_Original_Variance': 0.0, 'miRNA_Scaled_Variance': 0.0,
            'Methylation_Original_Variance': 0.0, 'Methylation_Scaled_Variance': 0.0,
            'Data_Imbalance': "Failed to load data",
            'Quality_Score': 0.0,
            'Processing_Status': "FAILED"
        })
        return result
    
    # Step 2: Get clinical data sample count from loaded data
    # The y_series already contains the clinical data with proper sample intersection
    result['Clinical_Samples'] = len(y_series)
    logger.info(f"Clinical samples (with valid outcomes): {len(y_series)}")
    
    # Step 3: Analyze raw modalities
    logger.info("Analyzing raw modalities...")
    
    # Map modality short names to display names
    modality_name_mapping = {
        'exp': 'GeneExp',
        'mirna': 'miRNA',
        'methy': 'Methylation'
    }
    
    raw_modality_stats = {}
    for modality_short_name, modality_df in modalities_data.items():
        modality_display_name = modality_name_mapping.get(modality_short_name, modality_short_name.capitalize())
        
        # Raw statistics
        raw_samples = modality_df.shape[1] - 1 if modality_df.shape[1] > 1 else 0  # Columns are samples (excluding ID)
        raw_features = modality_df.shape[0]  # Rows are features
        raw_missing_pct = calculate_missing_percentage(modality_df, exclude_first_col=True)
        
        # Calculate original variance (transpose to get samples x features)
        X_raw = modality_df.T.values  # transpose to samples x features
        if X_raw.shape[1] > 0:  # Exclude first column (ID)
            X_raw_data = X_raw[:, 1:] if X_raw.shape[1] > 1 else X_raw
            raw_variance, _ = calculate_variance_statistics(X_raw_data)
        else:
            raw_variance = 0.0
        
        result[f'{modality_display_name}_Samples_Raw'] = raw_samples
        result[f'{modality_display_name}_Features_Raw'] = raw_features
        result[f'{modality_display_name}_Missing_Pct_Raw'] = round(raw_missing_pct, 2)
        result[f'{modality_display_name}_Original_Variance'] = round(raw_variance, 4)
        
        raw_modality_stats[modality_short_name] = {
            'samples': raw_samples,
            'features': raw_features,
            'variance': raw_variance
        }
        
        logger.info(f"  Raw {modality_display_name}: {raw_samples} samples, {raw_features} features, {raw_missing_pct:.2f}% missing, variance: {raw_variance:.4f}")
    
    # Calculate combined raw missing percentage
    all_raw_values = []
    for modality_df in modalities_data.values():
        data_portion = modality_df.iloc[:, 1:] if modality_df.shape[1] > 1 else modality_df
        if not data_portion.empty:
            all_raw_values.extend(data_portion.values.flatten())
    
    if all_raw_values:
        total_values = len(all_raw_values)
        nan_count = sum(1 for x in all_raw_values if pd.isna(x))
        zero_count = sum(1 for x in all_raw_values if not pd.isna(x) and x == 0)
        missing_count = nan_count + zero_count
        combined_raw_missing = (missing_count / total_values) * 100
        result['Combined_Missing_Pct_Raw'] = round(combined_raw_missing, 2)
    else:
        result['Combined_Missing_Pct_Raw'] = 0.0
    
    # Step 4: Apply preprocessing pipeline
    logger.info("Applying preprocessing pipeline...")
    
    processed_modalities, y_aligned, pipeline_metadata = apply_preprocessing_pipeline(
        modalities_data, y_series, common_ids, dataset_name, config['type'], logger
    )
    
    if not processed_modalities or y_aligned is None:
        logger.error(f"Failed to apply preprocessing for {dataset_name}")
        # Fill processed columns with zeros and mark as failed
        for modality_short_name in modalities_data.keys():
            modality_display_name = modality_name_mapping.get(modality_short_name, modality_short_name.capitalize())
            result[f'{modality_display_name}_Samples_Processed'] = 0
            result[f'{modality_display_name}_Features_Processed'] = 0
            result[f'{modality_display_name}_Missing_Pct_Processed'] = 0.0
            result[f'{modality_display_name}_Feature_Retention_Pct'] = 0.0
            result[f'{modality_display_name}_Scaled_Variance'] = 0.0
        
        result.update({
            'Combined_Missing_Pct_Processed': 0.0,
            'Sample_Retention_Pct': 0.0,
            'Quality_Score': 0.0,
            'Processing_Status': "FAILED"
        })
    else:
        # Step 5: Analyze processed modalities
        logger.info("Analyzing processed modalities...")
        
        processed_modality_stats = {}
        for modality_short_name, modality_array in processed_modalities.items():
            modality_display_name = modality_name_mapping.get(modality_short_name, modality_short_name.capitalize())
            
            # Processed statistics
            processed_samples = modality_array.shape[0]
            processed_features = modality_array.shape[1]
            processed_missing_pct = calculate_array_missing_percentage(modality_array)
            
            # Calculate scaled variance
            scaled_variance, _ = calculate_variance_statistics(modality_array)
            
            result[f'{modality_display_name}_Samples_Processed'] = processed_samples
            result[f'{modality_display_name}_Features_Processed'] = processed_features
            result[f'{modality_display_name}_Missing_Pct_Processed'] = round(processed_missing_pct, 2)
            result[f'{modality_display_name}_Scaled_Variance'] = round(scaled_variance, 4)
            
            processed_modality_stats[modality_short_name] = {
                'samples': processed_samples,
                'features': processed_features,
                'variance': scaled_variance
            }
            
            logger.info(f"  Processed {modality_display_name}: {processed_samples} samples, {processed_features} features, {processed_missing_pct:.2f}% missing, variance: {scaled_variance:.4f}")
        
        # Calculate combined processed missing percentage
        all_processed_values = []
        for modality_array in processed_modalities.values():
            all_processed_values.extend(modality_array.flatten())
        
        if all_processed_values:
            total_values = len(all_processed_values)
            nan_count = sum(1 for x in all_processed_values if np.isnan(x))
            zero_count = sum(1 for x in all_processed_values if not np.isnan(x) and x == 0)
            missing_count = nan_count + zero_count
            combined_processed_missing = (missing_count / total_values) * 100
            result['Combined_Missing_Pct_Processed'] = round(combined_processed_missing, 2)
        else:
            result['Combined_Missing_Pct_Processed'] = 0.0
        
        # Step 6: Calculate retention percentages
        logger.info("Calculating retention percentages...")
        
        # Sample retention (based on the first modality or y_aligned)
        original_samples = len(common_ids) if common_ids else 0
        processed_samples = len(y_aligned) if y_aligned is not None else 0
        sample_retention = (processed_samples / original_samples * 100) if original_samples > 0 else 0.0
        result['Sample_Retention_Pct'] = round(sample_retention, 2)
        
        # Feature retention for each modality
        for modality_short_name in modalities_data.keys():
            modality_display_name = modality_name_mapping.get(modality_short_name, modality_short_name.capitalize())
            
            raw_features = raw_modality_stats[modality_short_name]['features']
            processed_features = processed_modality_stats.get(modality_short_name, {}).get('features', 0)
            
            feature_retention = (processed_features / raw_features * 100) if raw_features > 0 else 0.0
            result[f'{modality_display_name}_Feature_Retention_Pct'] = round(feature_retention, 2)
        
        # Quality score from pipeline metadata
        quality_score = pipeline_metadata.get('quality_score', 0.0) if pipeline_metadata else 0.0
        result['Quality_Score'] = round(quality_score, 4)
        result['Processing_Status'] = "SUCCESS"
        
        logger.info(f"Sample retention: {sample_retention:.2f}% ({original_samples} -> {processed_samples})")
        logger.info(f"Quality score: {quality_score:.4f}")
    
    # Step 7: Analyze data imbalance (using the loaded y_series data)
    if y_series is not None and len(y_series) > 0:
        imbalance = analyze_data_imbalance_from_series(y_series, config['type'], dataset_name)
        result['Data_Imbalance'] = imbalance
    else:
        result['Data_Imbalance'] = "No outcome data available"
    
    logger.info(f"Analysis completed for {dataset_name}")
    return result

def main():
    """Main function to create the preprocessed data summary table"""
    logger = setup_logging()
    logger.info("Creating preprocessed data characteristics summary...")
    logger.info(f"Analysis started at: {datetime.now()}")
    
    # Load configurations
    configs = load_dataset_configs()
    
    # Analyze all datasets
    results = []
    total_datasets = len(configs)
    
    for i, (dataset_name, config) in enumerate(configs.items(), 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing dataset {i}/{total_datasets}: {dataset_name.upper()}")
        logger.info(f"{'='*60}")
        
        try:
            start_time = time.time()
            result = analyze_dataset_with_preprocessing(dataset_name, config, logger)
            elapsed_time = time.time() - start_time
            
            result['Processing_Time_Seconds'] = round(elapsed_time, 2)
            results.append(result)
            
            logger.info(f"Dataset {dataset_name} completed in {elapsed_time:.2f} seconds")
            
            if result['Processing_Status'] == "SUCCESS":
                logger.info(f"✓ SUCCESS: {dataset_name}")
            else:
                logger.error(f"✗ FAILED: {dataset_name}")
                
        except Exception as e:
            logger.error(f"Error analyzing {dataset_name}: {e}")
            # Add error result to maintain structure
            error_result = {
                'Dataset': dataset_name.upper(),
                'Type': config['type'],
                'Outcome_Column': config['outcome_column'],
                'Processing_Status': "ERROR",
                'Processing_Time_Seconds': 0.0,
                'Error_Message': str(e)
            }
            
            # Fill other required columns with zeros
            modality_names = ['GeneExp', 'miRNA', 'Methylation']
            for modality in modality_names:
                error_result.update({
                    f'{modality}_Samples_Raw': 0, f'{modality}_Features_Raw': 0, f'{modality}_Missing_Pct_Raw': 0.0,
                    f'{modality}_Samples_Processed': 0, f'{modality}_Features_Processed': 0, f'{modality}_Missing_Pct_Processed': 0.0,
                    f'{modality}_Feature_Retention_Pct': 0.0,
                    f'{modality}_Original_Variance': 0.0, f'{modality}_Scaled_Variance': 0.0
                })
            
            error_result.update({
                'Clinical_Samples': 0,
                'Combined_Missing_Pct_Raw': 0.0, 'Combined_Missing_Pct_Processed': 0.0,
                'Sample_Retention_Pct': 0.0, 'Quality_Score': 0.0,
                'Data_Imbalance': f"Error: {e}"
            })
            
            results.append(error_result)
    
    # Create DataFrame and save to CSV
    df_results = pd.DataFrame(results)
    
    # Reorder columns for better presentation
    column_order = [
        'Dataset', 'Type', 'Outcome_Column', 'Clinical_Samples',
        # Raw data columns
        'GeneExp_Samples_Raw', 'GeneExp_Features_Raw', 'GeneExp_Missing_Pct_Raw',
        'miRNA_Samples_Raw', 'miRNA_Features_Raw', 'miRNA_Missing_Pct_Raw',
        'Methylation_Samples_Raw', 'Methylation_Features_Raw', 'Methylation_Missing_Pct_Raw',
        'Combined_Missing_Pct_Raw',
        # Processed data columns
        'GeneExp_Samples_Processed', 'GeneExp_Features_Processed', 'GeneExp_Missing_Pct_Processed',
        'miRNA_Samples_Processed', 'miRNA_Features_Processed', 'miRNA_Missing_Pct_Processed',
        'Methylation_Samples_Processed', 'Methylation_Features_Processed', 'Methylation_Missing_Pct_Processed',
        'Combined_Missing_Pct_Processed',
        # Retention percentages
        'Sample_Retention_Pct',
        'GeneExp_Feature_Retention_Pct', 'miRNA_Feature_Retention_Pct', 'Methylation_Feature_Retention_Pct',
        # Variance statistics
        'GeneExp_Original_Variance', 'GeneExp_Scaled_Variance',
        'miRNA_Original_Variance', 'miRNA_Scaled_Variance',
        'Methylation_Original_Variance', 'Methylation_Scaled_Variance',
        # Other information
        'Data_Imbalance', 'Quality_Score', 'Processing_Status', 'Processing_Time_Seconds'
    ]
    
    # Only include columns that exist in the dataframe
    available_columns = [col for col in column_order if col in df_results.columns]
    df_results = df_results[available_columns]
    
    # Save to CSV
    output_file = 'preprocessed_data_characteristics.csv'
    df_results.to_csv(output_file, index=False)
    
    # Calculate summary statistics
    successful = df_results[df_results['Processing_Status'] == 'SUCCESS']
    failed = df_results[df_results['Processing_Status'] != 'SUCCESS']
    
    total_time = df_results['Processing_Time_Seconds'].sum()
    
    logger.info(f"\n{'='*60}")
    logger.info("ANALYSIS COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"Total datasets analyzed: {len(results)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    logger.info(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    logger.info(f"Analysis completed at: {datetime.now()}")
    
    # Display summary statistics for successful datasets
    if len(successful) > 0:
        logger.info(f"\nSUMMARY STATISTICS (Successful datasets):")
        logger.info(f"Average sample retention: {successful['Sample_Retention_Pct'].mean():.2f}%")
        
        # Feature retention by modality
        for modality in ['GeneExp', 'miRNA', 'Methylation']:
            col = f'{modality}_Feature_Retention_Pct'
            if col in successful.columns:
                avg_retention = successful[col].mean()
                logger.info(f"Average {modality} feature retention: {avg_retention:.2f}%")
        
        # Quality scores
        avg_quality = successful['Quality_Score'].mean()
        logger.info(f"Average quality score: {avg_quality:.4f}")
        
        # Variance improvements
        logger.info(f"\nVARIANCE STATISTICS:")
        for modality in ['GeneExp', 'miRNA', 'Methylation']:
            orig_col = f'{modality}_Original_Variance'
            scaled_col = f'{modality}_Scaled_Variance'
            if orig_col in successful.columns and scaled_col in successful.columns:
                avg_orig = successful[orig_col].mean()
                avg_scaled = successful[scaled_col].mean()
                logger.info(f"{modality}: Original variance: {avg_orig:.4f}, Scaled variance: {avg_scaled:.4f}")
    
    # Display the results preview
    logger.info(f"\nPreview of results:")
    print("\n" + df_results.to_string(index=False, max_rows=10, max_cols=15))
    
    if len(df_results) > 10:
        logger.info(f"\n... showing first 10 rows of {len(df_results)} total datasets")

if __name__ == "__main__":
    main() 