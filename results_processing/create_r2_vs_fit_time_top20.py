import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_r2_vs_fit_time_plot():
    """Create scatter plot of R² vs fit_time for top 20 regression combinations."""
    
    # Read the top combinations CSV file
    df = pd.read_csv('results/regression/combined/top_50_combinations_overall.csv')
    
    # Take only the top 20 combinations
    top_20 = df.head(20).copy()
    
    print(f"Creating plot for top 20 combinations...")
    print(f"R² range: {top_20['test_r2'].min():.4f} to {top_20['test_r2'].max():.4f}")
    print(f"Fit time range: {top_20['fit_time'].min():.4f} to {top_20['fit_time'].max():.4f}")
    
    # Create the plot with larger size for better label visibility
    plt.figure(figsize=(20, 14))
    
    # Create scatter plot with larger, more visible points
    scatter = plt.scatter(top_20['fit_time'], top_20['test_r2'], 
                         c=top_20['test_r2'], cmap='viridis', s=150, alpha=0.9, edgecolors='black', linewidth=2, zorder=5)
    
    # Used positions for overlap avoidance
    used_positions = {}
    for idx, row in top_20.iterrows():
        x = row['fit_time']
        y = row['test_r2']
        # Custom label position and alignment
        label_x = x + 0.002
        ha = 'left'
        relpos = (0, 0.5)
        # Overlap avoidance: move label lower if y is close to previous at same x
        key = round(x, 4)
        if key not in used_positions:
            used_positions[key] = []
        overlap_count = sum(abs(y - prev_y) < 0.0002 for prev_y in used_positions[key])
        label_y = y - 3 * overlap_count * 0.001  # tighter stacking
        used_positions[key].append(y)
        plt.annotate(
            row['combination'],
            (x, y),
            xytext=(label_x, label_y), textcoords='data',
            fontsize=8, ha=ha, va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='gray', alpha=0.7, relpos=relpos),
            zorder=3
        )
    
    # Customize the plot
    plt.xlabel('Average Fit Time', fontsize=12, fontweight='bold')
    plt.ylabel('R²', fontsize=12, fontweight='bold')
    plt.title('Top 20 Regression Combinations: R² vs Average Fit Time', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('R²', fontsize=12, fontweight='bold')
    
    # Set axis limits with generous padding for labels
    x_margin = (top_20['fit_time'].max() - top_20['fit_time'].min()) * 0.1
    y_margin = (top_20['test_r2'].max() - top_20['test_r2'].min()) * 0.1
    plt.xlim(top_20['fit_time'].min() - x_margin, 
             top_20['fit_time'].max() + x_margin)
    plt.ylim(top_20['test_r2'].min() - y_margin, 
             top_20['test_r2'].max() + y_margin)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Set x-axis ticks (optional: can be omitted for fit time)
    # unique_fit_time = sorted(top_20['fit_time'].unique())
    # plt.xticks(unique_fit_time)
    
    # Adjust layout with padding for labels
    plt.tight_layout(pad=2.0)
    
    # Save the plot with extra padding to ensure labels are not cut off
    output_file = 'results/regression/combined/r2_vs_fit_time_top20.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.5)
    print(f"Plot saved as: {output_file}")
    
    plt.show()

if __name__ == "__main__":
    create_r2_vs_fit_time_plot() 