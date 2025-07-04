#!/usr/bin/env python3
"""
Create R² vs Missing Percentage Plot for Top 20 Regression Combinations

This script creates a scatter plot showing the relationship between R² values and 
missing data percentage for the top 20 performing regression combinations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_r2_vs_missing_plot():
    """Create scatter plot of R² vs missing percentage for top 20 combinations."""
    
    # Read the top combinations CSV file
    df = pd.read_csv('results/regression/combined/top_50_combinations_overall.csv')
    
    # Take only the top 20 combinations
    top_20 = df.head(20).copy()
    
    print(f"Creating plot for top 20 combinations...")
    print(f"R² range: {top_20['test_r2'].min():.4f} to {top_20['test_r2'].max():.4f}")
    print(f"Missing % range: {top_20['missing_pct_display'].min()}% to {top_20['missing_pct_display'].max()}%")
    
    # Create the plot with larger size for better label visibility
    plt.figure(figsize=(20, 14))
    
    # Create scatter plot with larger, more visible points
    scatter = plt.scatter(top_20['missing_pct_display'], top_20['test_r2'], 
                         c=top_20['test_r2'], cmap='viridis', s=150, alpha=0.9, edgecolors='black', linewidth=2, zorder=5)
    
    # Used positions for overlap avoidance
    used_positions = {}
    for idx, row in top_20.iterrows():
        x = row['missing_pct_display']
        y = row['test_r2']
        # Default label position
        if x == 50:
            xytext = (-5, 5)
            ha = 'right'
        else:
            xytext = (5, 5)
            ha = 'left'
        # Overlap avoidance: move label lower if y is close to previous at same x
        key = x
        if key not in used_positions:
            used_positions[key] = []
        overlap_count = sum(abs(y - prev_y) < 0.0002 for prev_y in used_positions[key])
        if overlap_count > 0:
            xytext = (xytext[0], xytext[1] - 10 * overlap_count)
        used_positions[key].append(y)
        plt.annotate(
            row['combination'],
            (x, y),
            xytext=xytext, textcoords='offset points',
            fontsize=8, ha=ha, va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='gray', alpha=0.7),
            zorder=3
        )
    
    # Customize the plot
    plt.xlabel('Missing Data Percentage (%)', fontsize=12, fontweight='bold')
    plt.ylabel('R² Score', fontsize=12, fontweight='bold')
    plt.title('Top 20 Regression Combinations: R² vs Missing Data (%)', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('R² Score', fontsize=12, fontweight='bold')
    
    # Set axis limits with generous padding for labels
    x_margin = 8
    y_margin = (top_20['test_r2'].max() - top_20['test_r2'].min()) * 0.1
    plt.xlim(top_20['missing_pct_display'].min() - x_margin, 
             top_20['missing_pct_display'].max() + x_margin)
    plt.ylim(top_20['test_r2'].min() - y_margin, 
             top_20['test_r2'].max() + y_margin)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Set x-axis ticks to show all missing percentages
    unique_missing_pct = sorted(top_20['missing_pct_display'].unique())
    plt.xticks(unique_missing_pct)
    
    # Adjust layout with padding for labels
    plt.tight_layout(pad=2.0)
    
    # Save the plot with extra padding to ensure labels are not cut off
    output_file = 'results/regression/combined/r2_vs_missing_percentage_top20.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.5)
    print(f"Plot saved as: {output_file}")
    
    # Show statistics
    print(f"\nTop 20 Combinations Statistics:")
    print(f"Best R²: {top_20['test_r2'].max():.6f} at {top_20.loc[top_20['test_r2'].idxmax(), 'missing_pct_display']}% missing")
    print(f"Worst R² in top 20: {top_20['test_r2'].min():.6f} at {top_20.loc[top_20['test_r2'].idxmin(), 'missing_pct_display']}% missing")
    
    # Count by missing percentage
    missing_counts = top_20['missing_pct_display'].value_counts().sort_index()
    print(f"\nDistribution by missing percentage:")
    for missing_pct, count in missing_counts.items():
        print(f"  {missing_pct}%: {count} combinations")
    
    # Show most common algorithms
    algo_counts = top_20['algorithm'].value_counts()
    print(f"\nMost common algorithms in top 20:")
    for algo, count in algo_counts.items():
        print(f"  {algo}: {count} combinations")
    
    # Show most common fusion methods
    fusion_counts = top_20['fusion_method'].value_counts()
    print(f"\nMost common fusion methods in top 20:")
    for fusion, count in fusion_counts.items():
        print(f"  {fusion}: {count} combinations")
    
    # Show most common models
    model_counts = top_20['model'].value_counts()
    print(f"\nMost common models in top 20:")
    for model, count in model_counts.items():
        print(f"  {model}: {count} combinations")
    
    plt.show()

if __name__ == "__main__":
    create_r2_vs_missing_plot() 