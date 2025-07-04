#!/usr/bin/env python3
"""
Create MCC vs Missing Percentage Plot for Worst 50 Classification Combinations

This script creates a scatter plot showing the relationship between MCC values and 
missing data percentage for the worst 50 performing classification combinations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_mcc_vs_missing_plot_worst():
    """Create scatter plot of MCC vs missing percentage for worst 50 combinations."""
    
    # Read the worst combinations CSV file
    df = pd.read_csv('results/classification/combined/worst_50_combinations_overall.csv')
    
    # Take only the worst 20 combinations (bottom 20 since they're already ranked worst first)
    worst_20 = df.head(20).copy()
    
    print(f"Creating plot for worst 20 combinations...")
    print(f"MCC range: {worst_20['test_mcc'].min():.4f} to {worst_20['test_mcc'].max():.4f}")
    print(f"Missing % range: {worst_20['missing_pct_display'].min()}% to {worst_20['missing_pct_display'].max()}%")
    
    # Create the plot with larger size for better label visibility
    plt.figure(figsize=(20, 14))
    
    # Create scatter plot with larger, more visible points
    # Use a different colormap (Reds) to indicate poor performance
    scatter = plt.scatter(worst_20['missing_pct_display'], worst_20['test_mcc'], 
                         c=worst_20['test_mcc'], cmap='Reds', s=150, alpha=0.9, edgecolors='black', linewidth=2, zorder=5)
    
    # Build a mapping from x to all y-values and their corresponding row indices
    x_to_yrows = {}
    for idx, row in worst_20.iterrows():
        x = row['missing_pct_display']
        y = row['test_mcc']
        if x not in x_to_yrows:
            x_to_yrows[x] = []
        x_to_yrows[x].append((y, idx))

    # For each x, sort the y-values and annotate with two-directional overlap avoidance
    for x, yrows in x_to_yrows.items():
        # Sort by y-value
        yrows_sorted = sorted(yrows, key=lambda tup: tup[0])
        n = len(yrows_sorted)
        for i, (y, idx) in enumerate(yrows_sorted):
            row = worst_20.loc[idx]
            xytext = [-5, 5] if x == 50 else [5, 5]
            ha = 'right' if x == 50 else 'left'
            # Check previous
            if i > 0 and abs(y - yrows_sorted[i-1][0]) < 0.1:
                xytext[1] -= 10  # Move down if too close to previous
            # Check next
            if i < n - 1 and abs(y - yrows_sorted[i+1][0]) < 0.1:
                xytext[1] += 10  # Move up if too close to next
            plt.annotate(
                row['combination'],
                (x, y),
                xytext=tuple(xytext), textcoords='offset points',
                fontsize=8, ha=ha, va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='gray', alpha=0.7),
                zorder=3
            )
    
    # Customize the plot
    plt.xlabel('Missing Data Percentage (%)', fontsize=12, fontweight='bold')
    plt.ylabel('MCC Score', fontsize=12, fontweight='bold')
    plt.title('Worst 20 Classification Combinations: MCC vs Missing Data (%)', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('MCC Score', fontsize=12, fontweight='bold')
    
    # Set axis limits with generous padding for labels
    x_margin = 8
    
    mcc_min = worst_20['test_mcc'].min()
    mcc_max = worst_20['test_mcc'].max()
    mcc_range = mcc_max - mcc_min
    
    # If the range is very small, use a fixed range for better visualization
    if mcc_range < 0.01:
        y_margin = 0.05
        plt.ylim(mcc_min - y_margin, mcc_max + y_margin)
    else:
        y_margin = mcc_range * 0.1
        plt.ylim(mcc_min - y_margin, mcc_max + y_margin)
    
    plt.xlim(worst_20['missing_pct_display'].min() - x_margin, 
             worst_20['missing_pct_display'].max() + x_margin)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Set x-axis ticks to show all missing percentages
    unique_missing_pct = sorted(worst_20['missing_pct_display'].unique())
    plt.xticks(unique_missing_pct)
    
    # Adjust layout with padding for labels
    plt.tight_layout(pad=2.0)
    
    # Save the plot with extra padding to ensure labels are not cut off
    output_file = 'results/classification/combined/mcc_vs_missing_percentage_worst20.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.5)
    print(f"Plot saved as: {output_file}")
    
    # Show statistics
    print(f"\nWorst 20 Combinations Statistics:")
    print(f"Worst MCC: {worst_20['test_mcc'].min():.6f} at {worst_20.loc[worst_20['test_mcc'].idxmin(), 'missing_pct_display']}% missing")
    print(f"Best MCC in worst 20: {worst_20['test_mcc'].max():.6f} at {worst_20.loc[worst_20['test_mcc'].idxmax(), 'missing_pct_display']}% missing")
    
    # Count by missing percentage
    missing_counts = worst_20['missing_pct_display'].value_counts().sort_index()
    print(f"\nDistribution by missing percentage:")
    for missing_pct, count in missing_counts.items():
        print(f"  {missing_pct}%: {count} combinations")
    
    # Show most common algorithms
    algo_counts = worst_20['algorithm'].value_counts()
    print(f"\nMost common algorithms in worst 20:")
    for algo, count in algo_counts.items():
        print(f"  {algo}: {count} combinations")
    
    # Show most common fusion methods
    fusion_counts = worst_20['fusion_method'].value_counts()
    print(f"\nMost common fusion methods in worst 20:")
    for fusion, count in fusion_counts.items():
        print(f"  {fusion}: {count} combinations")
    
    # Show most common models
    model_counts = worst_20['model'].value_counts()
    print(f"\nMost common models in worst 20:")
    for model, count in model_counts.items():
        print(f"  {model}: {count} combinations")
    
    # Show dataset distribution
    dataset_counts = worst_20['dataset'].value_counts()
    print(f"\nDataset distribution in worst 20:")
    for dataset, count in dataset_counts.items():
        print(f"  {dataset}: {count} combinations")
    
    plt.show()

if __name__ == "__main__":
    create_mcc_vs_missing_plot_worst() 