#!/usr/bin/env python3
"""
Fitting Time Analysis Script

This script analyzes the fitting time ranges for different combinations of:
- Extractors (PCA, KPCA, FA, PLS, KPLS, SparsePLS, LDA, PLS-DA)
- Selectors (ElasticNetFS, RFImportance, VarianceFTest, LASSO, f_regressionFS, LogisticL1)
- Fusion Techniques (average, attention_weighted, mkl, learnable_weighted, standard_concat, early_fusion_pca, sum, max)
- Models (LinearRegression, ElasticNet, RandomForestRegressor, LogisticRegression, RandomForestClassifier, SVC)

The analysis covers both regression and classification tasks across different missing data percentages.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings

warnings.filterwarnings('ignore')

class FittingTimeAnalyzer:
    """Analyzes fitting time ranges for different algorithm combinations."""
    
    def __init__(self):
        """Initialize the analyzer."""
        self.regression_datasets = ['AML', 'Sarcoma']
        self.classification_datasets = ['Breast', 'Colon', 'Kidney', 'Liver', 'Lung', 'Melanoma', 'Ovarian']
        self.missing_percentages = [0, 20, 50]
        
        # Define algorithm categories
        self.extractors = ['PCA', 'KPCA', 'FA', 'PLS', 'KPLS', 'SparsePLS', 'LDA', 'PLS-DA']
        self.selectors = ['ElasticNetFS', 'RFImportance', 'VarianceFTest', 'LASSO', 'f_regressionFS', 'LogisticL1']
        self.fusion_methods = ['average', 'attention_weighted', 'mkl', 'learnable_weighted', 'standard_concat', 'early_fusion_pca', 'sum', 'max']
        self.regression_models = ['LinearRegression', 'ElasticNet', 'RandomForestRegressor']
        self.classification_models = ['LogisticRegression', 'RandomForestClassifier', 'SVC']
        
        self.data = []
        
    def load_output_data(self, base_path: str = "output") -> None:
        """Load all JSON result files from the output directory."""
        print("Loading output data...")
        
        # Load regression data
        for dataset in self.regression_datasets:
            for missing_pct in self.missing_percentages:
                file_path = f"{base_path}/{dataset}/{dataset}_feature_first_results_{missing_pct}pct_missing.json"
                self._load_dataset_file(file_path, dataset, missing_pct, "regression")
        
        # Load classification data
        for dataset in self.classification_datasets:
            for missing_pct in self.missing_percentages:
                file_path = f"{base_path}/{dataset}/{dataset}_feature_first_results_{missing_pct}pct_missing.json"
                self._load_dataset_file(file_path, dataset, missing_pct, "classification")
        
        self.df = pd.DataFrame(self.data)
        print(f"Total experiments loaded: {len(self.df)}")
        
    def _load_dataset_file(self, file_path: str, dataset: str, missing_pct: int, task_type: str) -> None:
        """Load a single dataset file."""
        try:
            with open(file_path, 'r') as f:
                experiments = json.load(f)
            
            print(f"Loaded {len(experiments)} experiments from {file_path}")
            
            for exp in experiments:
                record = {
                    'dataset': dataset,
                    'task_type': task_type,
                    'missing_percentage': missing_pct / 100,
                    'algorithm': exp.get('algorithm', exp.get('algorithm_name', 'Unknown')),
                    'fusion_method': exp.get('fusion_method', 'Unknown'),
                    'model': exp.get('model', exp.get('model_name', 'Unknown')),
                    'fit_time': exp.get('mean_scores', {}).get('fit_time', np.nan),
                    'std_fit_time': exp.get('std_scores', {}).get('fit_time', np.nan),
                    'n_features': exp.get('n_features', 0),
                    'n_samples': exp.get('n_samples', 0),
                    'n_value': exp.get('n_value', exp.get('n_features_components', 0))
                }
                
                # Add performance metrics
                if task_type == "regression":
                    record['test_r2'] = exp.get('mean_scores', {}).get('test_r2', np.nan)
                else:
                    record['test_mcc'] = exp.get('mean_scores', {}).get('test_mcc', np.nan)
                
                # Categorize algorithm
                if record['algorithm'] in self.extractors:
                    record['algorithm_type'] = 'extractor'
                elif record['algorithm'] in self.selectors:
                    record['algorithm_type'] = 'selector'
                else:
                    record['algorithm_type'] = 'unknown'
                
                self.data.append(record)
                
        except FileNotFoundError:
            print(f"Warning: File not found - {file_path}")
        except json.JSONDecodeError:
            print(f"Warning: Invalid JSON in {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    def analyze_fitting_time_ranges(self) -> Dict[str, Any]:
        """Analyze fitting time ranges for different combinations."""
        print("\n=== FITTING TIME RANGE ANALYSIS ===")
        
        analysis = {
            'overall_summary': {},
            'by_algorithm_type': {},
            'by_fusion_method': {},
            'by_model': {},
            'by_task_type': {},
            'by_missing_percentage': {},
            'extreme_cases': {}
        }
        
        # Overall summary
        valid_fit_times = self.df[self.df['fit_time'].notna() & (self.df['fit_time'] > 0)]
        analysis['overall_summary'] = {
            'total_experiments': len(self.df),
            'valid_fit_time_experiments': len(valid_fit_times),
            'min_fit_time': valid_fit_times['fit_time'].min(),
            'max_fit_time': valid_fit_times['fit_time'].max(),
            'mean_fit_time': valid_fit_times['fit_time'].mean(),
            'median_fit_time': valid_fit_times['fit_time'].median(),
            'std_fit_time': valid_fit_times['fit_time'].std()
        }
        
        # By algorithm type
        for algo_type in ['extractor', 'selector']:
            subset = valid_fit_times[valid_fit_times['algorithm_type'] == algo_type]
            if len(subset) > 0:
                analysis['by_algorithm_type'][algo_type] = {
                    'count': len(subset),
                    'min_fit_time': subset['fit_time'].min(),
                    'max_fit_time': subset['fit_time'].max(),
                    'mean_fit_time': subset['fit_time'].mean(),
                    'median_fit_time': subset['fit_time'].median(),
                    'std_fit_time': subset['fit_time'].std()
                }
        
        # By fusion method
        for fusion in self.fusion_methods:
            subset = valid_fit_times[valid_fit_times['fusion_method'] == fusion]
            if len(subset) > 0:
                analysis['by_fusion_method'][fusion] = {
                    'count': len(subset),
                    'min_fit_time': subset['fit_time'].min(),
                    'max_fit_time': subset['fit_time'].max(),
                    'mean_fit_time': subset['fit_time'].mean(),
                    'median_fit_time': subset['fit_time'].median(),
                    'std_fit_time': subset['fit_time'].std()
                }
        
        # By model
        all_models = self.regression_models + self.classification_models
        for model in all_models:
            subset = valid_fit_times[valid_fit_times['model'] == model]
            if len(subset) > 0:
                analysis['by_model'][model] = {
                    'count': len(subset),
                    'min_fit_time': subset['fit_time'].min(),
                    'max_fit_time': subset['fit_time'].max(),
                    'mean_fit_time': subset['fit_time'].mean(),
                    'median_fit_time': subset['fit_time'].median(),
                    'std_fit_time': subset['fit_time'].std()
                }
        
        # By task type
        for task_type in ['regression', 'classification']:
            subset = valid_fit_times[valid_fit_times['task_type'] == task_type]
            if len(subset) > 0:
                analysis['by_task_type'][task_type] = {
                    'count': len(subset),
                    'min_fit_time': subset['fit_time'].min(),
                    'max_fit_time': subset['fit_time'].max(),
                    'mean_fit_time': subset['fit_time'].mean(),
                    'median_fit_time': subset['fit_time'].median(),
                    'std_fit_time': subset['fit_time'].std()
                }
        
        # By missing percentage
        for missing_pct in [0, 20, 50]:
            subset = valid_fit_times[valid_fit_times['missing_percentage'] == missing_pct / 100]
            if len(subset) > 0:
                analysis['by_missing_percentage'][f"{missing_pct}%"] = {
                    'count': len(subset),
                    'min_fit_time': subset['fit_time'].min(),
                    'max_fit_time': subset['fit_time'].max(),
                    'mean_fit_time': subset['fit_time'].mean(),
                    'median_fit_time': subset['fit_time'].median(),
                    'std_fit_time': subset['fit_time'].std()
                }
        
        # Extreme cases (fastest and slowest)
        fastest = valid_fit_times.loc[valid_fit_times['fit_time'].idxmin()]
        slowest = valid_fit_times.loc[valid_fit_times['fit_time'].idxmax()]
        
        analysis['extreme_cases'] = {
            'fastest': {
                'fit_time': fastest['fit_time'],
                'dataset': fastest['dataset'],
                'algorithm': fastest['algorithm'],
                'fusion_method': fastest['fusion_method'],
                'model': fastest['model'],
                'task_type': fastest['task_type']
            },
            'slowest': {
                'fit_time': slowest['fit_time'],
                'dataset': slowest['dataset'],
                'algorithm': slowest['algorithm'],
                'fusion_method': slowest['fusion_method'],
                'model': slowest['model'],
                'task_type': slowest['task_type']
            }
        }
        
        return analysis
    
    def analyze_individual_algorithms(self) -> Dict[str, Any]:
        """Analyze fitting time ranges for individual algorithms."""
        print("\n=== INDIVIDUAL ALGORITHM ANALYSIS ===")
        
        analysis = {}
        valid_fit_times = self.df[self.df['fit_time'].notna() & (self.df['fit_time'] > 0)]
        
        # Analyze extractors
        for extractor in self.extractors:
            subset = valid_fit_times[valid_fit_times['algorithm'] == extractor]
            if len(subset) > 0:
                analysis[extractor] = {
                    'type': 'extractor',
                    'count': len(subset),
                    'min_fit_time': subset['fit_time'].min(),
                    'max_fit_time': subset['fit_time'].max(),
                    'mean_fit_time': subset['fit_time'].mean(),
                    'median_fit_time': subset['fit_time'].median(),
                    'std_fit_time': subset['fit_time'].std(),
                    'regression_count': len(subset[subset['task_type'] == 'regression']),
                    'classification_count': len(subset[subset['task_type'] == 'classification'])
                }
        
        # Analyze selectors
        for selector in self.selectors:
            subset = valid_fit_times[valid_fit_times['algorithm'] == selector]
            if len(subset) > 0:
                analysis[selector] = {
                    'type': 'selector',
                    'count': len(subset),
                    'min_fit_time': subset['fit_time'].min(),
                    'max_fit_time': subset['fit_time'].max(),
                    'mean_fit_time': subset['fit_time'].mean(),
                    'median_fit_time': subset['fit_time'].median(),
                    'std_fit_time': subset['fit_time'].std(),
                    'regression_count': len(subset[subset['task_type'] == 'regression']),
                    'classification_count': len(subset[subset['task_type'] == 'classification'])
                }
        
        return analysis
    
    def print_analysis_results(self, analysis: Dict[str, Any], algorithm_analysis: Dict[str, Any]) -> None:
        """Print the analysis results in a formatted way."""
        
        print("\n" + "="*80)
        print("FITTING TIME RANGE ANALYSIS RESULTS")
        print("="*80)
        
        # Overall summary
        print("\n📊 OVERALL SUMMARY:")
        overall = analysis['overall_summary']
        print(f"  Total experiments: {overall['total_experiments']}")
        print(f"  Valid fit time experiments: {overall['valid_fit_time_experiments']}")
        print(f"  Fit time range: {overall['min_fit_time']:.4f}s - {overall['max_fit_time']:.4f}s")
        print(f"  Mean fit time: {overall['mean_fit_time']:.4f}s")
        print(f"  Median fit time: {overall['median_fit_time']:.4f}s")
        print(f"  Standard deviation: {overall['std_fit_time']:.4f}s")
        
        # By algorithm type
        print("\n🔧 BY ALGORITHM TYPE:")
        for algo_type, stats in analysis['by_algorithm_type'].items():
            print(f"  {algo_type.title()}:")
            print(f"    Count: {stats['count']}")
            print(f"    Range: {stats['min_fit_time']:.4f}s - {stats['max_fit_time']:.4f}s")
            print(f"    Mean: {stats['mean_fit_time']:.4f}s")
            print(f"    Median: {stats['median_fit_time']:.4f}s")
        
        # By fusion method
        print("\n🔗 BY FUSION METHOD:")
        for fusion, stats in analysis['by_fusion_method'].items():
            print(f"  {fusion}:")
            print(f"    Count: {stats['count']}")
            print(f"    Range: {stats['min_fit_time']:.4f}s - {stats['max_fit_time']:.4f}s")
            print(f"    Mean: {stats['mean_fit_time']:.4f}s")
            print(f"    Median: {stats['median_fit_time']:.4f}s")
        
        # By model
        print("\n🤖 BY MODEL:")
        for model, stats in analysis['by_model'].items():
            print(f"  {model}:")
            print(f"    Count: {stats['count']}")
            print(f"    Range: {stats['min_fit_time']:.4f}s - {stats['max_fit_time']:.4f}s")
            print(f"    Mean: {stats['mean_fit_time']:.4f}s")
            print(f"    Median: {stats['median_fit_time']:.4f}s")
        
        # By task type
        print("\n📈 BY TASK TYPE:")
        for task_type, stats in analysis['by_task_type'].items():
            print(f"  {task_type.title()}:")
            print(f"    Count: {stats['count']}")
            print(f"    Range: {stats['min_fit_time']:.4f}s - {stats['max_fit_time']:.4f}s")
            print(f"    Mean: {stats['mean_fit_time']:.4f}s")
            print(f"    Median: {stats['median_fit_time']:.4f}s")
        
        # By missing percentage
        print("\n📊 BY MISSING DATA PERCENTAGE:")
        for missing_pct, stats in analysis['by_missing_percentage'].items():
            print(f"  {missing_pct} missing data:")
            print(f"    Count: {stats['count']}")
            print(f"    Range: {stats['min_fit_time']:.4f}s - {stats['max_fit_time']:.4f}s")
            print(f"    Mean: {stats['mean_fit_time']:.4f}s")
            print(f"    Median: {stats['median_fit_time']:.4f}s")
        
        # Extreme cases
        print("\n⚡ EXTREME CASES:")
        fastest = analysis['extreme_cases']['fastest']
        slowest = analysis['extreme_cases']['slowest']
        
        print(f"  Fastest: {fastest['fit_time']:.4f}s")
        print(f"    Dataset: {fastest['dataset']}")
        print(f"    Algorithm: {fastest['algorithm']}")
        print(f"    Fusion: {fastest['fusion_method']}")
        print(f"    Model: {fastest['model']}")
        print(f"    Task: {fastest['task_type']}")
        
        print(f"  Slowest: {slowest['fit_time']:.4f}s")
        print(f"    Dataset: {slowest['dataset']}")
        print(f"    Algorithm: {slowest['algorithm']}")
        print(f"    Fusion: {slowest['fusion_method']}")
        print(f"    Model: {slowest['model']}")
        print(f"    Task: {slowest['task_type']}")
        
        # Individual algorithm analysis
        print("\n🔍 INDIVIDUAL ALGORITHM ANALYSIS:")
        print("\nExtractors:")
        for algo, stats in algorithm_analysis.items():
            if stats['type'] == 'extractor':
                print(f"  {algo}:")
                print(f"    Count: {stats['count']} (R: {stats['regression_count']}, C: {stats['classification_count']})")
                print(f"    Range: {stats['min_fit_time']:.4f}s - {stats['max_fit_time']:.4f}s")
                print(f"    Mean: {stats['mean_fit_time']:.4f}s")
                print(f"    Median: {stats['median_fit_time']:.4f}s")
        
        print("\nSelectors:")
        for algo, stats in algorithm_analysis.items():
            if stats['type'] == 'selector':
                print(f"  {algo}:")
                print(f"    Count: {stats['count']} (R: {stats['regression_count']}, C: {stats['classification_count']})")
                print(f"    Range: {stats['min_fit_time']:.4f}s - {stats['max_fit_time']:.4f}s")
                print(f"    Mean: {stats['mean_fit_time']:.4f}s")
                print(f"    Median: {stats['median_fit_time']:.4f}s")
        
        print("\n" + "="*80)
    
    def save_analysis_results(self, analysis: Dict[str, Any], algorithm_analysis: Dict[str, Any], output_file: str = "fitting_time_analysis_results.json") -> None:
        """Save analysis results to JSON file."""
        results = {
            'analysis': analysis,
            'algorithm_analysis': algorithm_analysis,
            'summary': {
                'total_experiments': len(self.df),
                'valid_fit_time_experiments': len(self.df[self.df['fit_time'].notna() & (self.df['fit_time'] > 0)]),
                'datasets_analyzed': list(self.df['dataset'].unique()),
                'task_types': list(self.df['task_type'].unique()),
                'missing_percentages': list(self.df['missing_percentage'].unique())
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Analysis results saved to: {output_file}")

def main():
    """Main function to run the fitting time analysis."""
    analyzer = FittingTimeAnalyzer()
    
    # Load data
    analyzer.load_output_data()
    
    # Perform analysis
    analysis = analyzer.analyze_fitting_time_ranges()
    algorithm_analysis = analyzer.analyze_individual_algorithms()
    
    # Print results
    analyzer.print_analysis_results(analysis, algorithm_analysis)
    
    # Save results
    analyzer.save_analysis_results(analysis, algorithm_analysis)

if __name__ == "__main__":
    main() 