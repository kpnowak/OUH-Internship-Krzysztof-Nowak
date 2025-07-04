#!/usr/bin/env python3
"""
Classification Results Analyzer

This script analyzes results from classification experiments on cancer datasets
(Colon, Breast, Kidney, Liver, Lung, Melanoma, Ovarian), generating comprehensive 
CSV rankings based on MCC (Matthews Correlation Coefficient) performance across 
different missing data percentages.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Any
import warnings

warnings.filterwarnings('ignore')


class ClassificationResultsAnalyzer:
    """Analyzes classification experiment results and generates comprehensive rankings."""
    
    def __init__(self, datasets: List[str] = None):
        """
        Initialize the analyzer.
        
        Args:
            datasets: List of dataset names to analyze (default: classification datasets)
        """
        self.datasets = datasets or ['Colon', 'Breast', 'Kidney', 'Liver', 'Lung', 'Melanoma', 'Ovarian']
        self.missing_percentages = [0, 20, 50]
        self.data = []
        
        # Define extractors and selectors based on algorithm types for classification
        self.extractors = ['PCA', 'KPCA', 'FA', 'PLS-DA', 'LDA', 'SparsePLS']
        self.selectors = ['f_classifFS', 'VarianceFTest', 'LASSO', 'RFImportance', 'ElasticNetFS', 'LogisticL1']
        
    def load_data(self, base_path: str = "output") -> None:
        """
        Load all JSON result files.
        
        Args:
            base_path: Base directory containing the result files
        """
        print("Loading classification data...")
        
        for dataset in self.datasets:
            for missing_pct in self.missing_percentages:
                file_path = f"{base_path}/{dataset}/{dataset}_feature_first_results_{missing_pct}pct_missing.json"
                
                try:
                    with open(file_path, 'r') as f:
                        experiments = json.load(f)
                    
                    print(f"Loaded {len(experiments)} experiments from {file_path}")
                    
                    # Process each experiment
                    for exp in experiments:
                        # Extract relevant information
                        record = {
                            'dataset': exp.get('dataset', dataset),
                            'missing_percentage': exp.get('missing_percentage', missing_pct/100),
                            'algorithm': exp.get('algorithm', exp.get('algorithm_name', 'Unknown')),
                            'fusion_method': exp.get('fusion_method', 'Unknown'),
                            'model': exp.get('model', exp.get('model_name', 'Unknown')),
                            'test_mcc': exp.get('mean_scores', {}).get('test_mcc', np.nan),
                            'std_test_mcc': exp.get('std_scores', {}).get('test_mcc', np.nan),
                            'test_accuracy': exp.get('mean_scores', {}).get('test_accuracy', np.nan),
                            'std_test_accuracy': exp.get('std_scores', {}).get('test_accuracy', np.nan),
                            'test_f1': exp.get('mean_scores', {}).get('test_f1', np.nan),
                            'std_test_f1': exp.get('std_scores', {}).get('test_f1', np.nan),
                            'test_precision': exp.get('mean_scores', {}).get('test_precision', np.nan),
                            'std_test_precision': exp.get('std_scores', {}).get('test_precision', np.nan),
                            'test_recall': exp.get('mean_scores', {}).get('test_recall', np.nan),
                            'std_test_recall': exp.get('std_scores', {}).get('test_recall', np.nan),
                            'fit_time': exp.get('mean_scores', {}).get('fit_time', np.nan),
                            'std_fit_time': exp.get('std_scores', {}).get('fit_time', np.nan),
                            'n_features': exp.get('n_features', 0),
                            'n_samples': exp.get('n_samples', 0),
                            'n_value': exp.get('n_value', exp.get('n_features_components', 0))
                        }
                        
                        # Create combination identifier
                        record['combination'] = f"{record['algorithm']}_{record['fusion_method']}_{record['model']}"
                        
                        # Check if algorithm is an extractor or selector
                        if record['algorithm'] in self.extractors:
                            # For extractors, get number of components from processed_modalities_shapes
                            processed_shapes = exp.get('processed_modalities_shapes', {})
                            mirna_shape = processed_shapes.get('mirna', [])
                            if len(mirna_shape) >= 2:
                                n_components = mirna_shape[1]
                                record['combination'] += f"_{n_components}c"
                        elif record['algorithm'] in self.selectors:
                            # For selectors, use n_value (number of features)
                            if record.get('n_value', 0) > 0:
                                record['combination'] += f"_{record['n_value']}f"
                        
                        self.data.append(record)
                        
                except FileNotFoundError:
                    print(f"Warning: File not found - {file_path}")
                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON in {file_path}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        self.df = pd.DataFrame(self.data)
        
        # Convert missing percentage to percentage format for display
        self.df['missing_pct_display'] = (self.df['missing_percentage'] * 100).astype(int)
        
        print(f"Total experiments loaded: {len(self.df)}")
        print(f"Datasets: {self.df['dataset'].unique()}")
        print(f"Missing percentages: {sorted(self.df['missing_pct_display'].unique())}")
        
    def create_top_combinations_overall(self, top_n: int = 20, dataset: str = None) -> pd.DataFrame:
        """Create ranking of top combinations overall across all conditions for a specific dataset."""
        dataset_label = f" for {dataset}" if dataset else ""
        print(f"Creating top {top_n} combinations overall{dataset_label}...")
        
        # Filter by dataset if specified
        data = self.df if dataset is None else self.df[self.df['dataset'] == dataset]
        
        # Filter valid MCC scores
        valid_data = data[~data['test_mcc'].isna() & 
                         (data['test_mcc'] != np.inf) & 
                         (data['test_mcc'] != -np.inf)]
        
        # Sort by test_mcc descending and take top N
        top_combinations = valid_data.nlargest(top_n, 'test_mcc').copy()
        
        # Create output dataframe
        result = top_combinations[[
            'dataset', 'missing_pct_display', 'algorithm', 'fusion_method', 
            'model', 'combination', 'test_mcc', 'std_test_mcc', 'test_accuracy', 'std_test_accuracy',
            'test_f1', 'std_test_f1', 'test_precision', 'std_test_precision', 'test_recall', 'std_test_recall',
            'fit_time', 'std_fit_time', 'n_features', 'n_samples'
        ]].copy()
        
        result['rank'] = range(1, len(result) + 1)
        
        # Reorder columns
        result = result[['rank', 'dataset', 'missing_pct_display', 'algorithm', 
                        'fusion_method', 'model', 'combination', 'test_mcc', 'std_test_mcc',
                        'test_accuracy', 'std_test_accuracy', 'test_f1', 'std_test_f1', 
                        'test_precision', 'std_test_precision', 'test_recall', 'std_test_recall',
                        'fit_time', 'std_fit_time', 'n_features', 'n_samples']]
        
        return result
        
    def create_worst_combinations_overall(self, worst_n: int = 50, dataset: str = None) -> pd.DataFrame:
        """Create ranking of worst combinations overall across all conditions for a specific dataset."""
        dataset_label = f" for {dataset}" if dataset else ""
        print(f"Creating worst {worst_n} combinations overall{dataset_label}...")
        
        # Filter by dataset if specified
        data = self.df if dataset is None else self.df[self.df['dataset'] == dataset]
        
        # Filter valid MCC scores and only include MCC >= -1.0
        valid_data = data[~data['test_mcc'].isna() & 
                         (data['test_mcc'] != np.inf) & 
                         (data['test_mcc'] != -np.inf) &
                         (data['test_mcc'] >= -1.0)]
        
        # Sort by test_mcc ascending and take worst N
        worst_combinations = valid_data.nsmallest(worst_n, 'test_mcc').copy()
        
        # Create output dataframe
        result = worst_combinations[[
            'dataset', 'missing_pct_display', 'algorithm', 'fusion_method', 
            'model', 'combination', 'test_mcc', 'std_test_mcc', 'test_accuracy', 'std_test_accuracy',
            'test_f1', 'std_test_f1', 'test_precision', 'std_test_precision', 'test_recall', 'std_test_recall',
            'fit_time', 'std_fit_time', 'n_features', 'n_samples'
        ]].copy()
        
        result['rank'] = range(1, len(result) + 1)
        
        # Reorder columns
        result = result[['rank', 'dataset', 'missing_pct_display', 'algorithm', 
                        'fusion_method', 'model', 'combination', 'test_mcc', 'std_test_mcc',
                        'test_accuracy', 'std_test_accuracy', 'test_f1', 'std_test_f1', 
                        'test_precision', 'std_test_precision', 'test_recall', 'std_test_recall',
                        'fit_time', 'std_fit_time', 'n_features', 'n_samples']]
        
        return result
        
    def create_worst_combinations_by_missing_pct(self, worst_n: int = 50, dataset: str = None) -> Dict[int, pd.DataFrame]:
        """Create rankings of worst combinations for each missing percentage separately for a specific dataset."""
        dataset_label = f" for {dataset}" if dataset else ""
        print(f"Creating worst {worst_n} combinations for each missing percentage{dataset_label}...")
        
        # Filter by dataset if specified
        data = self.df if dataset is None else self.df[self.df['dataset'] == dataset]
        
        results = {}
        
        for missing_pct in sorted(data['missing_pct_display'].unique()):
            subset = data[data['missing_pct_display'] == missing_pct]
            
            # Filter valid MCC scores and only include MCC >= -1.0
            valid_data = subset[~subset['test_mcc'].isna() & 
                              (subset['test_mcc'] != np.inf) & 
                              (subset['test_mcc'] != -np.inf) &
                              (subset['test_mcc'] >= -1.0)]
            
            # Sort by test_mcc ascending and take worst N
            worst_combinations = valid_data.nsmallest(worst_n, 'test_mcc').copy()
            
            # Create output dataframe
            result = worst_combinations[[
                'dataset', 'algorithm', 'fusion_method', 'model', 'combination',
                'test_mcc', 'std_test_mcc', 'test_accuracy', 'std_test_accuracy',
                'test_f1', 'std_test_f1', 'test_precision', 'std_test_precision',
                'test_recall', 'std_test_recall', 'fit_time', 'std_fit_time',
                'n_features', 'n_samples'
            ]].copy()
            
            result['rank'] = range(1, len(result) + 1)
            result['missing_percentage'] = missing_pct
            
            # Reorder columns
            result = result[['rank', 'missing_percentage', 'dataset', 'algorithm', 
                            'fusion_method', 'model', 'combination', 'test_mcc', 'std_test_mcc',
                            'test_accuracy', 'std_test_accuracy', 'test_f1', 'std_test_f1', 
                            'test_precision', 'std_test_precision', 'test_recall', 'std_test_recall',
                            'fit_time', 'std_fit_time', 'n_features', 'n_samples']]
            
            results[missing_pct] = result
            
        return results
        
    def create_top_combinations_by_missing_pct(self, top_n: int = 20, dataset: str = None) -> Dict[int, pd.DataFrame]:
        """Create rankings of top combinations for each missing percentage separately for a specific dataset."""
        dataset_label = f" for {dataset}" if dataset else ""
        print(f"Creating top {top_n} combinations for each missing percentage{dataset_label}...")
        
        # Filter by dataset if specified
        data = self.df if dataset is None else self.df[self.df['dataset'] == dataset]
        
        results = {}
        
        for missing_pct in sorted(data['missing_pct_display'].unique()):
            subset = data[data['missing_pct_display'] == missing_pct]
            
            # Filter valid MCC scores
            valid_data = subset[~subset['test_mcc'].isna() & 
                              (subset['test_mcc'] != np.inf) & 
                              (subset['test_mcc'] != -np.inf)]
            
            # Sort by test_mcc descending and take top N
            top_combinations = valid_data.nlargest(top_n, 'test_mcc').copy()
            
            # Create output dataframe
            result = top_combinations[[
                'dataset', 'algorithm', 'fusion_method', 'model', 'combination',
                'test_mcc', 'std_test_mcc', 'test_accuracy', 'std_test_accuracy',
                'test_f1', 'std_test_f1', 'test_precision', 'std_test_precision',
                'test_recall', 'std_test_recall', 'fit_time', 'std_fit_time',
                'n_features', 'n_samples'
            ]].copy()
            
            result['rank'] = range(1, len(result) + 1)
            result['missing_percentage'] = missing_pct
            
            # Reorder columns
            result = result[['rank', 'missing_percentage', 'dataset', 'algorithm', 
                            'fusion_method', 'model', 'combination', 'test_mcc', 'std_test_mcc',
                            'test_accuracy', 'std_test_accuracy', 'test_f1', 'std_test_f1', 
                            'test_precision', 'std_test_precision', 'test_recall', 'std_test_recall',
                            'fit_time', 'std_fit_time', 'n_features', 'n_samples']]
            
            results[missing_pct] = result
            
        return results
        
    def create_fusion_technique_rankings(self, dataset: str = None) -> pd.DataFrame:
        """Create rankings of fusion techniques by missing percentage for a specific dataset."""
        dataset_label = f" for {dataset}" if dataset else ""
        print(f"Creating fusion technique rankings{dataset_label}...")
        
        # Filter by dataset if specified
        data = self.df if dataset is None else self.df[self.df['dataset'] == dataset]
        
        # Filter out MCC values lower than -1 before calculating averages
        filtered_df = data[data['test_mcc'] >= -1.0].copy()
        
        # Group by fusion method and missing percentage
        fusion_stats = filtered_df.groupby(['fusion_method', 'missing_pct_display']).agg({
            'test_mcc': ['mean', 'std', 'count'],
            'test_accuracy': 'mean',
            'test_f1': 'mean',
            'test_precision': 'mean',
            'test_recall': 'mean',
            'fit_time': 'mean'
        }).round(4)
        
        # Flatten column names
        fusion_stats.columns = ['_'.join(col).strip() for col in fusion_stats.columns]
        fusion_stats = fusion_stats.reset_index()
        
        # Rename columns for clarity
        fusion_stats.rename(columns={
            'test_mcc_mean': 'avg_test_mcc',
            'test_mcc_std': 'std_test_mcc',
            'test_mcc_count': 'n_experiments',
            'test_accuracy_mean': 'avg_test_accuracy',
            'test_f1_mean': 'avg_test_f1',
            'test_precision_mean': 'avg_test_precision',
            'test_recall_mean': 'avg_test_recall',
            'fit_time_mean': 'avg_fit_time'
        }, inplace=True)
        
        # Create rankings for each missing percentage
        results = []
        for missing_pct in sorted(fusion_stats['missing_pct_display'].unique()):
            subset = fusion_stats[fusion_stats['missing_pct_display'] == missing_pct].copy()
            subset['rank'] = subset['avg_test_mcc'].rank(method='dense', ascending=False)
            subset = subset.sort_values('rank')
            results.append(subset)
        
        result_df = pd.concat(results, ignore_index=True)
        
        # Reorder columns
        result_df = result_df[['missing_pct_display', 'rank', 'fusion_method', 
                              'avg_test_mcc', 'std_test_mcc', 'n_experiments',
                              'avg_test_accuracy', 'avg_test_f1', 'avg_test_precision',
                              'avg_test_recall', 'avg_fit_time']]
        
        return result_df
        
    def create_extractor_selector_rankings(self, dataset: str = None) -> pd.DataFrame:
        """Create rankings of extractors and selectors combined by missing percentage for a specific dataset."""
        dataset_label = f" for {dataset}" if dataset else ""
        print(f"Creating extractor/selector rankings{dataset_label}...")
        
        # Filter by dataset if specified
        data = self.df if dataset is None else self.df[self.df['dataset'] == dataset]
        
        # Filter out MCC values lower than -1 before calculating averages
        filtered_df = data[data['test_mcc'] >= -1.0].copy()
        
        # Group by algorithm and missing percentage
        algo_stats = filtered_df.groupby(['algorithm', 'missing_pct_display']).agg({
            'test_mcc': ['mean', 'std', 'count'],
            'test_accuracy': 'mean',
            'test_f1': 'mean',
            'test_precision': 'mean',
            'test_recall': 'mean',
            'fit_time': 'mean'
        }).round(4)
        
        # Flatten column names
        algo_stats.columns = ['_'.join(col).strip() for col in algo_stats.columns]
        algo_stats = algo_stats.reset_index()
        
        # Rename columns for clarity
        algo_stats.rename(columns={
            'test_mcc_mean': 'avg_test_mcc',
            'test_mcc_std': 'std_test_mcc',
            'test_mcc_count': 'n_experiments',
            'test_accuracy_mean': 'avg_test_accuracy',
            'test_f1_mean': 'avg_test_f1',
            'test_precision_mean': 'avg_test_precision',
            'test_recall_mean': 'avg_test_recall',
            'fit_time_mean': 'avg_fit_time'
        }, inplace=True)
        
        # Create rankings for each missing percentage
        results = []
        for missing_pct in sorted(algo_stats['missing_pct_display'].unique()):
            subset = algo_stats[algo_stats['missing_pct_display'] == missing_pct].copy()
            subset['rank'] = subset['avg_test_mcc'].rank(method='dense', ascending=False)
            subset = subset.sort_values('rank')
            results.append(subset)
        
        result_df = pd.concat(results, ignore_index=True)
        
        # Reorder columns
        result_df = result_df[['missing_pct_display', 'rank', 'algorithm', 
                              'avg_test_mcc', 'std_test_mcc', 'n_experiments',
                              'avg_test_accuracy', 'avg_test_f1', 'avg_test_precision',
                              'avg_test_recall', 'avg_fit_time']]
        
        return result_df
        
    def create_extractor_rankings(self, dataset: str = None) -> pd.DataFrame:
        """Create rankings of extractors only by missing percentage for a specific dataset."""
        dataset_label = f" for {dataset}" if dataset else ""
        print(f"Creating extractor rankings{dataset_label}...")
        
        # Filter by dataset if specified
        data = self.df if dataset is None else self.df[self.df['dataset'] == dataset]
        
        # Filter out MCC values lower than -1 and keep only extractor algorithms
        filtered_df = data[(data['test_mcc'] >= -1.0) & 
                          (data['algorithm'].isin(self.extractors))].copy()
        
        # Group by extractor algorithm and missing percentage
        extractor_stats = filtered_df.groupby(['algorithm', 'missing_pct_display']).agg({
            'test_mcc': ['mean', 'std', 'count'],
            'test_accuracy': 'mean',
            'test_f1': 'mean',
            'test_precision': 'mean',
            'test_recall': 'mean',
            'fit_time': 'mean'
        }).round(4)
        
        # Flatten column names
        extractor_stats.columns = ['_'.join(col).strip() for col in extractor_stats.columns]
        extractor_stats = extractor_stats.reset_index()
        
        # Rename columns for clarity
        extractor_stats.rename(columns={
            'test_mcc_mean': 'avg_test_mcc',
            'test_mcc_std': 'std_test_mcc',
            'test_mcc_count': 'n_experiments',
            'test_accuracy_mean': 'avg_test_accuracy',
            'test_f1_mean': 'avg_test_f1',
            'test_precision_mean': 'avg_test_precision',
            'test_recall_mean': 'avg_test_recall',
            'fit_time_mean': 'avg_fit_time'
        }, inplace=True)
        
        # Create rankings for each missing percentage
        results = []
        for missing_pct in sorted(extractor_stats['missing_pct_display'].unique()):
            subset = extractor_stats[extractor_stats['missing_pct_display'] == missing_pct].copy()
            subset['rank'] = subset['avg_test_mcc'].rank(method='dense', ascending=False)
            subset = subset.sort_values('rank')
            results.append(subset)
        
        result_df = pd.concat(results, ignore_index=True)
        
        # Reorder columns
        result_df = result_df[['missing_pct_display', 'rank', 'algorithm', 
                              'avg_test_mcc', 'std_test_mcc', 'n_experiments',
                              'avg_test_accuracy', 'avg_test_f1', 'avg_test_precision',
                              'avg_test_recall', 'avg_fit_time']]
        
        return result_df
        
    def create_selector_rankings(self, dataset: str = None) -> pd.DataFrame:
        """Create rankings of selectors only by missing percentage for a specific dataset."""
        dataset_label = f" for {dataset}" if dataset else ""
        print(f"Creating selector rankings{dataset_label}...")
        
        # Filter by dataset if specified
        data = self.df if dataset is None else self.df[self.df['dataset'] == dataset]
        
        # Filter out MCC values lower than -1 and keep only selector algorithms
        filtered_df = data[(data['test_mcc'] >= -1.0) & 
                          (data['algorithm'].isin(self.selectors))].copy()
        
        # Group by selector algorithm and missing percentage
        selector_stats = filtered_df.groupby(['algorithm', 'missing_pct_display']).agg({
            'test_mcc': ['mean', 'std', 'count'],
            'test_accuracy': 'mean',
            'test_f1': 'mean',
            'test_precision': 'mean',
            'test_recall': 'mean',
            'fit_time': 'mean'
        }).round(4)
        
        # Flatten column names
        selector_stats.columns = ['_'.join(col).strip() for col in selector_stats.columns]
        selector_stats = selector_stats.reset_index()
        
        # Rename columns for clarity
        selector_stats.rename(columns={
            'test_mcc_mean': 'avg_test_mcc',
            'test_mcc_std': 'std_test_mcc',
            'test_mcc_count': 'n_experiments',
            'test_accuracy_mean': 'avg_test_accuracy',
            'test_f1_mean': 'avg_test_f1',
            'test_precision_mean': 'avg_test_precision',
            'test_recall_mean': 'avg_test_recall',
            'fit_time_mean': 'avg_fit_time'
        }, inplace=True)
        
        # Create rankings for each missing percentage
        results = []
        for missing_pct in sorted(selector_stats['missing_pct_display'].unique()):
            subset = selector_stats[selector_stats['missing_pct_display'] == missing_pct].copy()
            subset['rank'] = subset['avg_test_mcc'].rank(method='dense', ascending=False)
            subset = subset.sort_values('rank')
            results.append(subset)
        
        result_df = pd.concat(results, ignore_index=True)
        
        # Reorder columns
        result_df = result_df[['missing_pct_display', 'rank', 'algorithm', 
                              'avg_test_mcc', 'std_test_mcc', 'n_experiments',
                              'avg_test_accuracy', 'avg_test_f1', 'avg_test_precision',
                              'avg_test_recall', 'avg_fit_time']]
        
        return result_df
        
    def create_model_rankings(self, dataset: str = None) -> pd.DataFrame:
        """Create rankings of models by missing percentage for a specific dataset."""
        dataset_label = f" for {dataset}" if dataset else ""
        print(f"Creating model rankings{dataset_label}...")
        
        # Filter by dataset if specified
        data = self.df if dataset is None else self.df[self.df['dataset'] == dataset]
        
        # Filter out MCC values lower than -1 before calculating averages
        filtered_df = data[data['test_mcc'] >= -1.0].copy()
        
        # Group by model and missing percentage
        model_stats = filtered_df.groupby(['model', 'missing_pct_display']).agg({
            'test_mcc': ['mean', 'std', 'count'],
            'test_accuracy': 'mean',
            'test_f1': 'mean',
            'test_precision': 'mean',
            'test_recall': 'mean',
            'fit_time': 'mean'
        }).round(4)
        
        # Flatten column names
        model_stats.columns = ['_'.join(col).strip() for col in model_stats.columns]
        model_stats = model_stats.reset_index()
        
        # Rename columns for clarity
        model_stats.rename(columns={
            'test_mcc_mean': 'avg_test_mcc',
            'test_mcc_std': 'std_test_mcc',
            'test_mcc_count': 'n_experiments',
            'test_accuracy_mean': 'avg_test_accuracy',
            'test_f1_mean': 'avg_test_f1',
            'test_precision_mean': 'avg_test_precision',
            'test_recall_mean': 'avg_test_recall',
            'fit_time_mean': 'avg_fit_time'
        }, inplace=True)
        
        # Create rankings for each missing percentage
        results = []
        for missing_pct in sorted(model_stats['missing_pct_display'].unique()):
            subset = model_stats[model_stats['missing_pct_display'] == missing_pct].copy()
            subset['rank'] = subset['avg_test_mcc'].rank(method='dense', ascending=False)
            subset = subset.sort_values('rank')
            results.append(subset)
        
        result_df = pd.concat(results, ignore_index=True)
        
        # Reorder columns
        result_df = result_df[['missing_pct_display', 'rank', 'model', 
                              'avg_test_mcc', 'std_test_mcc', 'n_experiments',
                              'avg_test_accuracy', 'avg_test_f1', 'avg_test_precision',
                              'avg_test_recall', 'avg_fit_time']]
        
        return result_df
        
    def save_results(self, output_dir: str = "results") -> None:
        """Save all analysis results to CSV files, organized by dataset."""
        print(f"Saving classification results to {output_dir}...")
        
        # Create main output directory and classification subdirectory
        classification_dir = f"{output_dir}/classification"
        Path(classification_dir).mkdir(parents=True, exist_ok=True)
        
        # Analyze each dataset separately
        for dataset in self.datasets:
            dataset_dir = f"{classification_dir}/{dataset}"
            Path(dataset_dir).mkdir(exist_ok=True)
            
            print(f"\n=== ANALYZING {dataset} CLASSIFICATION DATASET ===")
            
            # 1. Top 50 combinations overall for this dataset
            top_overall = self.create_top_combinations_overall(50, dataset)
            top_overall.to_csv(f"{dataset_dir}/top_50_combinations_overall.csv", index=False)
            
            # 2. Top 50 combinations by missing percentage for this dataset
            top_by_missing = self.create_top_combinations_by_missing_pct(50, dataset)
            for missing_pct, df in top_by_missing.items():
                df.to_csv(f"{dataset_dir}/top_50_combinations_{missing_pct}pct_missing.csv", index=False)
            
            # 3. Worst 50 combinations overall for this dataset
            worst_overall = self.create_worst_combinations_overall(50, dataset)
            worst_overall.to_csv(f"{dataset_dir}/worst_50_combinations_overall.csv", index=False)
            
            # 4. Worst 50 combinations by missing percentage for this dataset
            worst_by_missing = self.create_worst_combinations_by_missing_pct(50, dataset)
            for missing_pct, df in worst_by_missing.items():
                df.to_csv(f"{dataset_dir}/worst_50_combinations_{missing_pct}pct_missing.csv", index=False)
            
            # 5. Fusion technique rankings for this dataset
            fusion_rankings = self.create_fusion_technique_rankings(dataset)
            fusion_rankings.to_csv(f"{dataset_dir}/fusion_technique_rankings.csv", index=False)
            
            # 6. Extractor and selector rankings (combined) for this dataset
            extractor_selector_rankings = self.create_extractor_selector_rankings(dataset)
            extractor_selector_rankings.to_csv(f"{dataset_dir}/extractor_selector_rankings.csv", index=False)
            
            # 7. Extractor rankings for this dataset
            extractor_rankings = self.create_extractor_rankings(dataset)
            extractor_rankings.to_csv(f"{dataset_dir}/extractor_rankings.csv", index=False)
            
            # 8. Selector rankings for this dataset
            selector_rankings = self.create_selector_rankings(dataset)
            selector_rankings.to_csv(f"{dataset_dir}/selector_rankings.csv", index=False)
            
            # 9. Model rankings for this dataset
            model_rankings = self.create_model_rankings(dataset)
            model_rankings.to_csv(f"{dataset_dir}/model_rankings.csv", index=False)
            
            # Print dataset-specific statistics
            dataset_df = self.df[self.df['dataset'] == dataset]
            filtered_dataset_df = dataset_df[dataset_df['test_mcc'] >= -1.0]
            
            print(f"Results saved to {dataset_dir}/")
            print(f"Total experiments: {len(dataset_df)}")
            print(f"Experiments included in averages: {len(filtered_dataset_df)} ({100*len(filtered_dataset_df)/len(dataset_df):.1f}%)")
            if len(dataset_df) > 0:
                print(f"Best MCC score: {dataset_df['test_mcc'].max():.4f}")
            if len(filtered_dataset_df) > 0:
                print(f"Mean MCC score (filtered): {filtered_dataset_df['test_mcc'].mean():.4f}")
            
            # Show top combination for this dataset
            if len(dataset_df) > 0:
                best_combination = dataset_df.loc[dataset_df['test_mcc'].idxmax()]
                print(f"Best combination: {best_combination['algorithm']} + {best_combination['fusion_method']} + {best_combination['model']} (MCC={best_combination['test_mcc']:.4f})")
        
        # Create combined rankings across all classification datasets
        print(f"\n=== CREATING COMBINED CLASSIFICATION RANKINGS ===")
        combined_dir = f"{classification_dir}/combined"
        Path(combined_dir).mkdir(exist_ok=True)
        
        # 1. Combined top 50 combinations overall
        combined_top_overall = self.create_top_combinations_overall(50)
        combined_top_overall.to_csv(f"{combined_dir}/top_50_combinations_overall.csv", index=False)
        
        # 2. Combined top 50 combinations by missing percentage
        combined_top_by_missing = self.create_top_combinations_by_missing_pct(50)
        for missing_pct, df in combined_top_by_missing.items():
            df.to_csv(f"{combined_dir}/top_50_combinations_{missing_pct}pct_missing.csv", index=False)
        
        # 3. Combined worst 50 combinations overall
        combined_worst_overall = self.create_worst_combinations_overall(50)
        combined_worst_overall.to_csv(f"{combined_dir}/worst_50_combinations_overall.csv", index=False)
        
        # 4. Combined worst 50 combinations by missing percentage
        combined_worst_by_missing = self.create_worst_combinations_by_missing_pct(50)
        for missing_pct, df in combined_worst_by_missing.items():
            df.to_csv(f"{combined_dir}/worst_50_combinations_{missing_pct}pct_missing.csv", index=False)
        
        # 5. Combined fusion technique rankings
        combined_fusion_rankings = self.create_fusion_technique_rankings()
        combined_fusion_rankings.to_csv(f"{combined_dir}/fusion_technique_rankings.csv", index=False)
        
        # 6. Combined extractor and selector rankings
        combined_extractor_selector_rankings = self.create_extractor_selector_rankings()
        combined_extractor_selector_rankings.to_csv(f"{combined_dir}/extractor_selector_rankings.csv", index=False)
        
        # 7. Combined extractor rankings
        combined_extractor_rankings = self.create_extractor_rankings()
        combined_extractor_rankings.to_csv(f"{combined_dir}/extractor_rankings.csv", index=False)
        
        # 8. Combined selector rankings
        combined_selector_rankings = self.create_selector_rankings()
        combined_selector_rankings.to_csv(f"{combined_dir}/selector_rankings.csv", index=False)
        
        # 9. Combined model rankings
        combined_model_rankings = self.create_model_rankings()
        combined_model_rankings.to_csv(f"{combined_dir}/model_rankings.csv", index=False)
        
        print(f"Combined classification rankings saved to {combined_dir}/")
        
        print("\n=== ALL CLASSIFICATION RESULTS SAVED SUCCESSFULLY! ===")
        
        # Print overall summary
        print(f"\n=== OVERALL CLASSIFICATION SUMMARY ===")
        print(f"Total experiments analyzed: {len(self.df)}")
        print(f"Datasets analyzed: {', '.join(self.datasets)}")
        print(f"Missing percentages: {', '.join(map(str, sorted(self.df['missing_pct_display'].unique())))}")
        all_algorithms = sorted(self.df['algorithm'].unique())
        print(f"Algorithms ({len(all_algorithms)}): {', '.join(all_algorithms)}")
        print(f"  - Extractors ({len(self.extractors)}): {', '.join(sorted(self.extractors))}")
        print(f"  - Selectors ({len(self.selectors)}): {', '.join(sorted(self.selectors))}")
        print(f"Fusion methods ({len(self.df['fusion_method'].unique())}): {', '.join(sorted(self.df['fusion_method'].unique()))}")
        print(f"Models ({len(self.df['model'].unique())}): {', '.join(sorted(self.df['model'].unique()))}")
        
        # Show overall best combination
        if len(self.df) > 0:
            best_combination = self.df.loc[self.df['test_mcc'].idxmax()]
            print(f"\nOverall best classification combination:")
            print(f"  Dataset: {best_combination['dataset']}")
            print(f"  Missing %: {best_combination['missing_pct_display']}")
            print(f"  Algorithm: {best_combination['algorithm']}")
            print(f"  Fusion: {best_combination['fusion_method']}")
            print(f"  Model: {best_combination['model']}")
            print(f"  MCC Score: {best_combination['test_mcc']:.4f}")
        
        print(f"\nResults structure:")
        print(f"  - Individual dataset rankings: {classification_dir}/[dataset]/")
        print(f"  - Combined classification rankings: {classification_dir}/combined/")
        print(f"\nRanking files generated for each dataset:")
        for dataset in self.datasets:
            print(f"  {dataset}:")
            print(f"    - Extractor rankings: {len(self.extractors)} × 3 missing % = {len(self.extractors)*3} rows")
            print(f"    - Selector rankings: {len(self.selectors)} × 3 missing % = {len(self.selectors)*3} rows")
            print(f"    - Combined rankings: {len(self.extractors)+len(self.selectors)} × 3 missing % = {(len(self.extractors)+len(self.selectors))*3} rows")


def main():
    """Main function to run the classification analysis."""
    parser = argparse.ArgumentParser(description='Analyze classification results from cancer datasets')
    parser.add_argument('--datasets', nargs='+', 
                        default=['Colon', 'Breast', 'Kidney', 'Liver', 'Lung', 'Melanoma', 'Ovarian'],
                        help='Datasets to analyze (default: all classification datasets)')
    parser.add_argument('--input-dir', default='output',
                        help='Input directory containing result files (default: output)')
    parser.add_argument('--output-dir', default='results',
                        help='Output directory for analysis results (default: results)')
    
    args = parser.parse_args()
    
    # Create analyzer and run analysis
    analyzer = ClassificationResultsAnalyzer(datasets=args.datasets)
    analyzer.load_data(args.input_dir)
    analyzer.save_results(args.output_dir)
    

if __name__ == "__main__":
    main() 