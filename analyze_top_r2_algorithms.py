import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_and_combine_data():
    """Load both CSV files and combine them."""
    # Load the datasets
    extraction_df = pd.read_csv('output_main_without_mrmr/AML/metrics/AML_extraction_cv_metrics.csv')
    selection_df = pd.read_csv('output_main_without_mrmr/AML/metrics/AML_selection_cv_metrics.csv')
    
    # Combine the datasets
    combined_df = pd.concat([extraction_df, selection_df], ignore_index=True)
    
    return combined_df

def find_top_r2_algorithms(df, top_n=10):
    """Find the top N algorithms based on R² values."""
    # Sort by R² in descending order and get top N
    top_df = df.nlargest(top_n, 'r2').copy()
    
    # Create a label for each point combining the required information
    top_df['label'] = (top_df['Workflow'] + '_' + 
                      top_df['Algorithm'] + '_' + 
                      top_df['integration_tech'] + '_' + 
                      top_df['Model'] + '_' + 
                      top_df['n_features'].astype(str) + 'f')
    
    return top_df

def create_visualizations(top_df):
    """Create the requested visualizations."""
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: R² vs Missing Percentage
    scatter1 = ax1.scatter(top_df['Missing_Percentage'], top_df['r2'], 
                          s=100, alpha=0.7, c=range(len(top_df)), cmap='tab10')
    ax1.set_xlabel('Missing Percentage', fontsize=12)
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title('Top 10 Algorithms: R² vs Missing Percentage', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add labels for each point
    for i, row in top_df.iterrows():
        ax1.annotate(row['label'], 
                    (row['Missing_Percentage'], row['r2']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, rotation=45, ha='left')
    
    # Plot 2: R² vs Time
    scatter2 = ax2.scatter(top_df['train_time'], top_df['r2'], 
                          s=100, alpha=0.7, c=range(len(top_df)), cmap='tab10')
    ax2.set_xlabel('Training Time (seconds)', fontsize=12)
    ax2.set_ylabel('R² Score', fontsize=12)
    ax2.set_title('Top 10 Algorithms: R² vs Training Time', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')  # Log scale for better visualization of time differences
    ax2.grid(True, alpha=0.3)
    
    # Add labels for each point
    for i, row in top_df.iterrows():
        ax2.annotate(row['label'], 
                    (row['train_time'], row['r2']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, rotation=45, ha='left')
    
    plt.tight_layout()
    plt.savefig('top_10_r2_algorithms_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_top_algorithms_summary(top_df):
    """Print a summary of the top algorithms."""
    print("="*80)
    print("TOP 10 ALGORITHMS BY R² SCORE")
    print("="*80)
    
    for i, row in top_df.iterrows():
        print(f"\nRank {len(top_df) - list(top_df.index).index(i)}: R² = {row['r2']:.4f}")
        print(f"  Workflow: {row['Workflow']}")
        print(f"  Algorithm: {row['Algorithm']}")
        print(f"  Integration Tech: {row['integration_tech']}")
        print(f"  Model: {row['Model']}")
        print(f"  N Features: {row['n_features']}")
        print(f"  Missing Percentage: {row['Missing_Percentage']}%")
        print(f"  Training Time: {row['train_time']:.4f} seconds")
        print(f"  MSE: {row['mse']:.4f}")
        print(f"  RMSE: {row['rmse']:.4f}")
        print(f"  MAE: {row['mae']:.4f}")

def create_detailed_comparison_table(top_df):
    """Create a detailed comparison table."""
    # Select relevant columns for display
    display_cols = ['Workflow', 'Algorithm', 'integration_tech', 'Model', 
                   'n_features', 'Missing_Percentage', 'r2', 'train_time', 
                   'mse', 'rmse', 'mae']
    
    comparison_df = top_df[display_cols].copy()
    comparison_df = comparison_df.round(4)
    
    print("\n" + "="*120)
    print("DETAILED COMPARISON TABLE")
    print("="*120)
    print(comparison_df.to_string(index=False))
    
    # Save to CSV
    comparison_df.to_csv('top_10_r2_algorithms_comparison.csv', index=False)
    print(f"\nDetailed comparison saved to: top_10_r2_algorithms_comparison.csv")

def main():
    """Main function to execute the analysis."""
    print("Loading and combining data...")
    combined_df = load_and_combine_data()
    
    print(f"Total number of experiments: {len(combined_df)}")
    print(f"R² range: {combined_df['r2'].min():.4f} to {combined_df['r2'].max():.4f}")
    
    print("\nFinding top 10 algorithms by R² score...")
    top_df = find_top_r2_algorithms(combined_df, top_n=10)
    
    print("\nCreating visualizations...")
    create_visualizations(top_df)
    
    print_top_algorithms_summary(top_df)
    create_detailed_comparison_table(top_df)
    
    print(f"\nVisualization saved as: top_10_r2_algorithms_analysis.png")

if __name__ == "__main__":
    main() 