#!/usr/bin/env python3
"""
Create R² vs Missing Percentage Plot for Worst 50 Regression Combinations

This script creates a scatter plot showing the relationship between R² values and 
missing data percentage for the worst 50 performing regression combinations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_r2_vs_missing_plot_worst():
    """Create scatter plot of R² vs missing percentage for worst 50 combinations."""
    
    # Read the worst combinations CSV file
    df = pd.read_csv('results/regression/combined/worst_50_combinations_overall.csv')
    
    # Take only the worst 20 combinations (bottom 20 since they're already ranked worst first)
    worst_20 = df.head(20).copy()
    
    print(f"Creating plot for worst 20 combinations...")
    print(f"R² range: {worst_20['test_r2'].min():.4f} to {worst_20['test_r2'].max():.4f}")
    print(f"Missing % range: {worst_20['missing_pct_display'].min()}% to {worst_20['missing_pct_display'].max()}%")
    
    # Create the plot with larger size for better label visibility
    plt.figure(figsize=(20, 14))
    
    # Create scatter plot with larger, more visible points
    # Use a different colormap (Reds) to indicate poor performance
    scatter = plt.scatter(worst_20['missing_pct_display'], worst_20['test_r2'], 
                         c=worst_20['test_r2'], cmap='Reds', s=150, alpha=0.9, edgecolors='black', linewidth=2, zorder=5)
    
    # Define label positions for each missing percentage
    label_x_map = {0: 2, 20: 22, 50: 48}
    
    # Group by missing percentage and process each group
    for missing_pct, label_x in label_x_map.items():
        group = worst_20[worst_20['missing_pct_display'] == missing_pct].sort_values('test_r2', ascending=False).reset_index(drop=True)
        n = len(group)
        if n == 0:
            continue
        
        # Calculate dynamic gap based on the plot's y-range for proper visual separation
        plot_y_range = worst_20['test_r2'].max() - worst_20['test_r2'].min()
        if plot_y_range < 0.01:
            # For very tight ranges, use a fixed larger gap
            gap = 0.01
        else:
            # Use 15% of the plot range divided by max possible labels per group
            gap = plot_y_range * 0.15 / max(1, n)
        
        for i, row in enumerate(group.itertuples()):
            # Start from the top of the group and stack downward
            label_y = group['test_r2'].max() - i * gap
            # Ensure label stays within plot bounds
            label_y = max(min(label_y, worst_20['test_r2'].max() + 0.005), worst_20['test_r2'].min() - 0.005)
            
            if missing_pct == 50:
                ha = 'right'
                relpos = (1, 0.5)
            else:
                ha = 'left'
                relpos = (0, 0.5)
            
            plt.annotate(
                row.combination,
                xy=(row.missing_pct_display, row.test_r2),
                xytext=(label_x, label_y), textcoords='data',
                fontsize=8, ha=ha, va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='gray', alpha=0.7, relpos=relpos),
                zorder=3
            )
    
    # Customize the plot
    plt.xlabel('Missing Data Percentage (%)', fontsize=12, fontweight='bold')
    plt.ylabel('R² Score', fontsize=12, fontweight='bold')
    plt.title('Worst 20 Regression Combinations: R² vs Missing Data (%)', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('R² Score', fontsize=12, fontweight='bold')
    
    # Set axis limits with generous padding for labels
    x_margin = 8
    
    # For negative R² values, use a fixed range to make the plot more readable
    # Similar to how the top performers work well with their positive range
    r2_min = worst_20['test_r2'].min()
    r2_max = worst_20['test_r2'].max()
    r2_range = r2_max - r2_min
    
    # If the range is very small (like in our case), use a fixed range
    if r2_range < 0.01:
        # Use a fixed range of 0.1 for better visualization
        y_margin = 0.05  # 0.05 on each side
        plt.ylim(r2_min - y_margin, r2_max + y_margin)
    else:
        # Use the original logic for larger ranges
        y_margin = r2_range * 0.1
        plt.ylim(r2_min - y_margin, r2_max + y_margin)
    
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
    output_file = 'results/regression/combined/r2_vs_missing_percentage_worst20.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.5)
    print(f"Plot saved as: {output_file}")
    
    # Show statistics
    print(f"\nWorst 20 Combinations Statistics:")
    print(f"Worst R²: {worst_20['test_r2'].min():.6f} at {worst_20.loc[worst_20['test_r2'].idxmin(), 'missing_pct_display']}% missing")
    print(f"Best R² in worst 20: {worst_20['test_r2'].max():.6f} at {worst_20.loc[worst_20['test_r2'].idxmax(), 'missing_pct_display']}% missing")
    
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
    create_r2_vs_missing_plot_worst() 