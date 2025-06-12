#!/usr/bin/env python3
"""
Test script for fast feature selection alternatives to MRMR.

This script demonstrates the speed and effectiveness of different feature selection
methods on simulated high-dimensional genomic data similar to TCGA datasets.
"""

import numpy as np
import pandas as pd
import time
import logging
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_genomic_like_data(n_samples=200, n_features=10000, task='classification', 
                           noise_level=0.1, informative_ratio=0.05):
    """
    Create synthetic data that mimics TCGA genomic characteristics.
    
    Parameters
    ----------
    n_samples : int
        Number of samples (typical TCGA: 100-500)
    n_features : int  
        Number of features (typical genomics: 10k-50k)
    task : str
        'classification' or 'regression'
    noise_level : float
        Amount of noise to add
    informative_ratio : float
        Ratio of informative features
        
    Returns
    -------
    X, y : arrays
        Feature matrix and target vector
    """
    n_informative = max(1, int(n_features * informative_ratio))
    n_redundant = max(0, int(n_informative * 0.2))  # Some redundant features
    
    if task == 'classification':
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_clusters_per_class=1,
            class_sep=0.8,
            random_state=42
        )
        # Make data sparse (like gene expression)
        X = np.abs(X)  # Ensure non-negative
        sparse_mask = np.random.random(X.shape) < 0.7  # 70% sparsity
        X[sparse_mask] = 0
        
    else:  # regression
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            noise=noise_level,
            random_state=42
        )
        # Make data sparse and non-negative
        X = np.abs(X)
        sparse_mask = np.random.random(X.shape) < 0.7
        X[sparse_mask] = 0
        
    return X, y

def benchmark_feature_selection_methods(X, y, task='classification', n_features_to_select=100):
    """
    Benchmark different feature selection methods.
    
    Parameters
    ----------
    X : array
        Feature matrix
    y : array
        Target vector
    task : str
        'classification' or 'regression'
    n_features_to_select : int
        Number of features to select
        
    Returns
    -------
    dict
        Results dictionary with timing and performance metrics
    """
    from fast_feature_selection import FastFeatureSelector, get_fast_selector_recommendations
    
    # Get recommended methods for this data
    recommendations = get_fast_selector_recommendations(
        n_samples=X.shape[0], 
        n_features=X.shape[1], 
        is_regression=(task == 'regression')
    )
    
    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y if task == 'classification' else None
    )
    
    results = {}
    
    # Test each recommended method
    for method in recommendations:
        logger.info(f"Testing {method} for {task}")
        
        try:
            # Time the feature selection
            start_time = time.time()
            
            selector = FastFeatureSelector(
                method=method, 
                n_features=n_features_to_select,
                random_state=42
            )
            
            # Fit and transform
            X_train_selected = selector.fit_transform(
                X_train, y_train, 
                is_regression=(task == 'regression')
            )
            X_test_selected = selector.transform(X_test)
            
            selection_time = time.time() - start_time
            
            # Evaluate performance with a simple model
            start_time = time.time()
            
            if task == 'classification':
                model = RandomForestClassifier(n_estimators=50, random_state=42)
                model.fit(X_train_selected, y_train)
                y_pred = model.predict(X_test_selected)
                performance = accuracy_score(y_test, y_pred)
                metric_name = 'accuracy'
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                model.fit(X_train_selected, y_train)
                y_pred = model.predict(X_test_selected)
                performance = r2_score(y_test, y_pred)
                metric_name = 'r2_score'
            
            model_time = time.time() - start_time
            
            results[method] = {
                'selection_time': selection_time,
                'model_time': model_time,
                'total_time': selection_time + model_time,
                'performance': performance,
                'metric': metric_name,
                'n_features_selected': X_train_selected.shape[1],
                'success': True
            }
            
            logger.info(f"{method}: {selection_time:.2f}s selection, "
                       f"{performance:.3f} {metric_name}")
            
        except Exception as e:
            logger.error(f"Error with {method}: {str(e)}")
            results[method] = {
                'selection_time': float('inf'),
                'model_time': float('inf'), 
                'total_time': float('inf'),
                'performance': 0.0,
                'metric': metric_name if 'metric_name' in locals() else 'unknown',
                'n_features_selected': 0,
                'success': False,
                'error': str(e)
            }
    
    # Test MRMR for comparison (if available)
    try:
        logger.info(f"Testing MRMR for {task} (for comparison)")
        start_time = time.time()
        
        from mrmr_helper import simple_mrmr
        selected_indices = simple_mrmr(
            X_train, y_train,
            n_selected_features=n_features_to_select,
            is_regression=(task == 'regression')
        )
        
        X_train_mrmr = X_train[:, selected_indices]
        X_test_mrmr = X_test[:, selected_indices]
        
        mrmr_selection_time = time.time() - start_time
        
        # Evaluate MRMR performance
        start_time = time.time()
        
        if task == 'classification':
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_train_mrmr, y_train)
            y_pred = model.predict(X_test_mrmr)
            mrmr_performance = accuracy_score(y_test, y_pred)
        else:
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X_train_mrmr, y_train)
            y_pred = model.predict(X_test_mrmr)
            mrmr_performance = r2_score(y_test, y_pred)
        
        mrmr_model_time = time.time() - start_time
        
        results['MRMR'] = {
            'selection_time': mrmr_selection_time,
            'model_time': mrmr_model_time,
            'total_time': mrmr_selection_time + mrmr_model_time,
            'performance': mrmr_performance,
            'metric': metric_name,
            'n_features_selected': len(selected_indices),
            'success': True
        }
        
        logger.info(f"MRMR: {mrmr_selection_time:.2f}s selection, "
                   f"{mrmr_performance:.3f} {metric_name}")
        
    except Exception as e:
        logger.warning(f"MRMR not available or failed: {str(e)}")
        
    return results

def plot_benchmark_results(results, task, save_path=None):
    """Plot benchmark results."""
    # Filter successful results
    successful_results = {k: v for k, v in results.items() if v.get('success', False)}
    
    if not successful_results:
        logger.warning("No successful results to plot")
        return
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Feature Selection Benchmark - {task.title()}', fontsize=16)
    
    methods = list(successful_results.keys())
    selection_times = [successful_results[m]['selection_time'] for m in methods]
    total_times = [successful_results[m]['total_time'] for m in methods]
    performances = [successful_results[m]['performance'] for m in methods]
    
    # 1. Selection time comparison
    bars1 = ax1.bar(methods, selection_times, color='skyblue', alpha=0.7)
    ax1.set_title('Feature Selection Time')
    ax1.set_ylabel('Time (seconds)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, time_val in zip(bars1, selection_times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{time_val:.2f}s', ha='center', va='bottom')
    
    # 2. Total time comparison
    bars2 = ax2.bar(methods, total_times, color='lightcoral', alpha=0.7)
    ax2.set_title('Total Time (Selection + Model Training)')
    ax2.set_ylabel('Time (seconds)')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, time_val in zip(bars2, total_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{time_val:.2f}s', ha='center', va='bottom')
    
    # 3. Performance comparison
    metric_name = list(successful_results.values())[0]['metric']
    bars3 = ax3.bar(methods, performances, color='lightgreen', alpha=0.7)
    ax3.set_title(f'Model Performance ({metric_name})')
    ax3.set_ylabel(metric_name.replace('_', ' ').title())
    ax3.tick_params(axis='x', rotation=45)
    
    for bar, perf_val in zip(bars3, performances):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{perf_val:.3f}', ha='center', va='bottom')
    
    # 4. Speed vs Performance scatter
    ax4.scatter(selection_times, performances, s=100, alpha=0.7, c='purple')
    for i, method in enumerate(methods):
        ax4.annotate(method, (selection_times[i], performances[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax4.set_xlabel('Selection Time (seconds)')
    ax4.set_ylabel(f'Performance ({metric_name})')
    ax4.set_title('Speed vs Performance Trade-off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()

def main():
    """Main function to run the benchmark."""
    logger.info("Starting Fast Feature Selection Benchmark")
    
    # Test parameters
    test_configs = [
        {
            'name': 'Small Classification',
            'n_samples': 150,
            'n_features': 1000,
            'task': 'classification',
            'n_select': 50
        },
        {
            'name': 'Large Classification (TCGA-like)',
            'n_samples': 300,
            'n_features': 20000,
            'task': 'classification', 
            'n_select': 100
        },
        {
            'name': 'Small Regression',
            'n_samples': 150,
            'n_features': 1000,
            'task': 'regression',
            'n_select': 50
        },
        {
            'name': 'Large Regression (TCGA-like)',
            'n_samples': 300,
            'n_features': 20000,
            'task': 'regression',
            'n_select': 100
        }
    ]
    
    all_results = {}
    
    for config in test_configs:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running benchmark: {config['name']}")
        logger.info(f"Samples: {config['n_samples']}, Features: {config['n_features']}")
        logger.info(f"Task: {config['task']}, Selecting: {config['n_select']} features")
        logger.info(f"{'='*50}")
        
        # Create synthetic data
        X, y = create_genomic_like_data(
            n_samples=config['n_samples'],
            n_features=config['n_features'],
            task=config['task']
        )
        
        logger.info(f"Created data: X.shape={X.shape}, y.shape={y.shape}")
        logger.info(f"Data sparsity: {(X == 0).mean():.1%}")
        
        # Run benchmark
        results = benchmark_feature_selection_methods(
            X, y, 
            task=config['task'],
            n_features_to_select=config['n_select']
        )
        
        all_results[config['name']] = results
        
        # Print summary
        logger.info(f"\nResults for {config['name']}:")
        for method, result in results.items():
            if result.get('success', False):
                logger.info(f"  {method:15s}: {result['selection_time']:6.2f}s selection, "
                           f"{result['performance']:6.3f} {result['metric']}")
            else:
                logger.info(f"  {method:15s}: FAILED - {result.get('error', 'Unknown error')}")
        
        # Create plot for this configuration
        plot_save_path = f"benchmark_{config['name'].lower().replace(' ', '_')}.png"
        plot_benchmark_results(results, config['name'], plot_save_path)
    
    # Summary comparison
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY: Fast Feature Selection vs MRMR")
    logger.info(f"{'='*60}")
    
    for config_name, results in all_results.items():
        logger.info(f"\n{config_name}:")
        
        if 'MRMR' in results and results['MRMR'].get('success', False):
            mrmr_time = results['MRMR']['selection_time']
            mrmr_perf = results['MRMR']['performance']
            
            logger.info(f"  MRMR baseline: {mrmr_time:.2f}s, {mrmr_perf:.3f}")
            
            # Find fastest method with comparable performance
            fast_methods = {k: v for k, v in results.items() 
                          if k != 'MRMR' and v.get('success', False)}
            
            if fast_methods:
                # Sort by speed
                sorted_methods = sorted(fast_methods.items(), 
                                      key=lambda x: x[1]['selection_time'])
                
                fastest = sorted_methods[0]
                speedup = mrmr_time / fastest[1]['selection_time']
                perf_diff = fastest[1]['performance'] - mrmr_perf
                
                logger.info(f"  Fastest alternative: {fastest[0]}")
                logger.info(f"    Time: {fastest[1]['selection_time']:.2f}s "
                           f"({speedup:.1f}x faster)")
                logger.info(f"    Performance: {fastest[1]['performance']:.3f} "
                           f"({perf_diff:+.3f} vs MRMR)")
        else:
            logger.info("  MRMR not available for comparison")
    
    logger.info(f"\n{'='*60}")
    logger.info("Benchmark completed!")
    logger.info("Recommendation: Use 'variance_f_test' or 'rf_importance' for best speed/performance balance")
    logger.info(f"{'='*60}")

if __name__ == "__main__":
    main()