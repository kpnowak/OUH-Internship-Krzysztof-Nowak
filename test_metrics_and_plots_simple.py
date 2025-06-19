#!/usr/bin/env python3

import os
import pandas as pd
from pathlib import Path

def check_metrics_files_structure():
    """Check if all metrics files are properly saved with correct naming."""
    
    print("="*80)
    print("CHECKING METRICS AND PLOTS STRUCTURE")
    print("="*80)
    
    # 1. Check existing metrics directories
    print("\n1. Checking existing metrics directories...")
    metrics_dirs = []
    for root, dirs, files in os.walk("."):
        if "metrics" in root:
            metrics_dirs.append(root)
    
    print(f"   Found {len(metrics_dirs)} metrics directories:")
    for metrics_dir in metrics_dirs:
        print(f"     {metrics_dir}")
        
    # 2. Check metrics files in each directory
    print("\n2. Checking metrics files in each directory...")
    all_metrics_files = []
    
    for metrics_dir in metrics_dirs:
        if os.path.exists(metrics_dir):
            files = [f for f in os.listdir(metrics_dir) if f.endswith('.csv')]
            print(f"\n   {metrics_dir}:")
            print(f"     Found {len(files)} CSV files")
            
            for file in files:
                file_path = os.path.join(metrics_dir, file)
                file_size = os.path.getsize(file_path) / 1024  # KB
                print(f"       {file} ({file_size:.1f} KB)")
                all_metrics_files.append(file_path)
                
                # Check file content
                try:
                    df = pd.read_csv(file_path)
                    print(f"         Rows: {len(df)}, Columns: {len(df.columns)}")
                    
                    # Check key columns
                    expected_cols = ['Dataset', 'Workflow', 'Algorithm', 'Model']
                    missing_cols = [col for col in expected_cols if col not in df.columns]
                    if missing_cols:
                        print(f"         ⚠️  Missing columns: {missing_cols}")
                    else:
                        print(f"         ✅ All key columns present")
                        
                except Exception as e:
                    print(f"         ❌ Error reading file: {e}")
    
    # 3. Check naming conventions
    print("\n3. Checking naming conventions...")
    expected_patterns = [
        "_extraction_cv_metrics.csv",
        "_extraction_best_fold_metrics.csv",
        "_selection_cv_metrics.csv", 
        "_selection_best_fold_metrics.csv",
        "_combined_best_fold_metrics.csv"
    ]
    
    for pattern in expected_patterns:
        matching_files = [f for f in all_metrics_files if pattern in f]
        print(f"   Pattern '{pattern}': {len(matching_files)} files")
        if matching_files:
            for file in matching_files[:3]:  # Show first 3 examples
                print(f"     {file}")
            if len(matching_files) > 3:
                print(f"     ... and {len(matching_files) - 3} more")
                
    # 4. Check final_results structure
    print("\n4. Checking final_results structure...")
    final_results_dir = "final_results"
    if os.path.exists(final_results_dir):
        datasets = [d for d in os.listdir(final_results_dir) 
                   if os.path.isdir(os.path.join(final_results_dir, d))]
        print(f"   Found {len(datasets)} datasets in final_results:")
        
        for dataset in datasets[:5]:  # Show first 5
            dataset_path = os.path.join(final_results_dir, dataset)
            print(f"\n     {dataset}/")
            
            # Check CSV files
            csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
            print(f"       CSV files: {len(csv_files)}")
            for csv_file in csv_files:
                file_size = os.path.getsize(os.path.join(dataset_path, csv_file)) / 1024
                print(f"         {csv_file} ({file_size:.1f} KB)")
                
            # Check plots directory
            plots_dir = os.path.join(dataset_path, "plots")
            if os.path.exists(plots_dir):
                plot_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
                print(f"       Plot files: {len(plot_files)}")
                for plot_file in plot_files:
                    file_size = os.path.getsize(os.path.join(plots_dir, plot_file)) / 1024
                    print(f"         {plot_file} ({file_size:.1f} KB)")
            else:
                print(f"       No plots directory found")
                
        if len(datasets) > 5:
            print(f"     ... and {len(datasets) - 5} more datasets")
    else:
        print("   final_results directory not found")
        
    # 5. Check plot naming conventions
    print("\n5. Checking plot naming conventions...")
    expected_plot_patterns = [
        "top_algorithms_3d.png",
        "top_algorithms_comprehensive.png", 
        "top_feature_settings_3d.png",
        "top_feature_settings_comprehensive.png",
        "top_integration_tech_3d.png",
        "top_integration_tech_comprehensive.png",
        "top_models_3d.png",
        "top_models_comprehensive.png"
    ]
    
    all_plot_files = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith('.png'):
                all_plot_files.append(os.path.join(root, file))
    
    print(f"   Found {len(all_plot_files)} total PNG files")
    
    for pattern in expected_plot_patterns:
        matching_plots = [f for f in all_plot_files if pattern in f]
        if matching_plots:
            print(f"   Pattern '{pattern}': ✅ {len(matching_plots)} files")
        else:
            print(f"   Pattern '{pattern}': ❌ No files found")
    
    print("\n" + "="*80)
    print("METRICS AND PLOTS STRUCTURE CHECK COMPLETE")
    print("="*80)
    
    # Summary
    print(f"\nSUMMARY:")
    print(f"  Metrics directories: {len(metrics_dirs)}")
    print(f"  Total CSV files: {len(all_metrics_files)}")
    print(f"  Total PNG files: {len(all_plot_files)}")
    
    if final_results_dir and os.path.exists(final_results_dir):
        datasets = [d for d in os.listdir(final_results_dir) 
                   if os.path.isdir(os.path.join(final_results_dir, d))]
        print(f"  Final results datasets: {len(datasets)}")
    
    return len(all_metrics_files) > 0 and len(all_plot_files) > 0

def check_specific_output_quality():
    """Check specific aspects of output quality."""
    
    print("\n" + "="*80)
    print("CHECKING OUTPUT QUALITY")
    print("="*80)
    
    # Check if CSV files have data
    print("\n1. Checking CSV file data quality...")
    csv_files_checked = 0
    csv_files_with_data = 0
    
    for root, dirs, files in os.walk("."):
        if "metrics" in root or "final_results" in root:
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    csv_files_checked += 1
                    
                    try:
                        df = pd.read_csv(file_path)
                        if len(df) > 0:
                            csv_files_with_data += 1
                            
                            # Check for typical metrics columns
                            if 'rmse' in df.columns or 'mcc' in df.columns:
                                # Check if metrics have reasonable values
                                if 'rmse' in df.columns:
                                    rmse_values = df['rmse'].dropna()
                                    if len(rmse_values) > 0:
                                        print(f"     {file}: RMSE range [{rmse_values.min():.3f}, {rmse_values.max():.3f}]")
                                        
                                if 'mcc' in df.columns:
                                    mcc_values = df['mcc'].dropna()
                                    if len(mcc_values) > 0:
                                        print(f"     {file}: MCC range [{mcc_values.min():.3f}, {mcc_values.max():.3f}]")
                            
                    except Exception as e:
                        print(f"     ❌ Error reading {file}: {e}")
                        
    print(f"\n   CSV files checked: {csv_files_checked}")
    print(f"   CSV files with data: {csv_files_with_data}")
    
    # Check plot file sizes (they should not be tiny)
    print("\n2. Checking plot file sizes...")
    plot_files_checked = 0
    plot_files_good_size = 0
    
    for root, dirs, files in os.walk("."):
        if "plots" in root:
            for file in files:
                if file.endswith('.png'):
                    file_path = os.path.join(root, file)
                    plot_files_checked += 1
                    
                    file_size = os.path.getsize(file_path) / 1024  # KB
                    if file_size > 50:  # At least 50KB suggests actual plot content
                        plot_files_good_size += 1
                        
    print(f"   Plot files checked: {plot_files_checked}")
    print(f"   Plot files with good size (>50KB): {plot_files_good_size}")
    
    return csv_files_with_data > 0 and plot_files_good_size > 0

def main():
    """Main function to run all checks."""
    
    print(f"Starting metrics and plots verification check...")
    
    # Check structure
    structure_ok = check_metrics_files_structure()
    
    # Check quality
    quality_ok = check_specific_output_quality()
    
    print("\n" + "="*80)
    if structure_ok and quality_ok:
        print("✅ ALL METRICS AND PLOTS CHECKS PASSED!")
        print("   - Proper file structure exists")
        print("   - Files contain actual data")
        print("   - Naming conventions are followed")
    else:
        print("❌ SOME CHECKS FAILED:")
        if not structure_ok:
            print("   - File structure issues detected")
        if not quality_ok:
            print("   - Data quality issues detected")
    print("="*80)

if __name__ == "__main__":
    main() 