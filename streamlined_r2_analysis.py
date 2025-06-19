import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import textwrap

def load_and_combine_data():
    """Load both CSV files and combine them."""
    # Load the datasets
    extraction_df = pd.read_csv('output/AML/metrics/AML_extraction_cv_metrics.csv')
    selection_df = pd.read_csv('output/AML/metrics/AML_selection_cv_metrics.csv')
    
    # Combine the datasets
    combined_df = pd.concat([extraction_df, selection_df], ignore_index=True)
    
    return combined_df

def wrap_text(text, width=15):
    """Wrap text to specified width."""
    if len(str(text)) <= width:
        return str(text)
    return '\n'.join(textwrap.wrap(str(text), width=width))

def find_best_performers(df, top_n=10):
    """Find the best performing algorithms (highest R²)."""
    # Sort by R² in descending order and get top N
    top_df = df.nlargest(top_n, 'r2').copy()
    
    # Create wrapped labels for better display
    top_df['wrapped_model'] = top_df['Model'].apply(lambda x: wrap_text(x, 15))
    top_df['wrapped_algorithm'] = top_df['Algorithm'].apply(lambda x: wrap_text(x, 12))
    top_df['wrapped_integration'] = top_df['integration_tech'].apply(lambda x: wrap_text(x, 12))
    
    # Create comprehensive label with wrapping
    top_df['display_label'] = (top_df['wrapped_algorithm'] + '\n' + 
                              top_df['wrapped_model'] + '\n' + 
                              top_df['n_features'].astype(str) + 'f')
    
    return top_df

def create_streamlined_visualizations(top_df, df):
    """Create the 3 requested visualizations with proper text wrapping."""
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle('', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Define colors for consistent visualization
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_df)))
    
    # 1. Top 10: R² vs Missing Percentage
    ax1 = axes[0]
    scatter1 = ax1.scatter(top_df['Missing_Percentage'], top_df['r2'], 
                          s=200, alpha=0.8, c=colors, edgecolors='black', linewidth=2)
    ax1.set_xlabel('Missing Percentage (%)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('R² Score', fontsize=14, fontweight='bold')
    ax1.set_title('Top 10 Algorithms:\nR² vs Missing Percentage', fontsize=16, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.4)
    
    # Add wrapped labels for top performers
    for i, row in top_df.iterrows():
        ax1.annotate(row['display_label'], 
                    (row['Missing_Percentage'], row['r2']),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=9, ha='left', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor='white', 
                             alpha=0.8, edgecolor='gray'),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1'))
    
    # Set better axis limits and formatting
    ax1.set_xlim(-0.1, 0.6)
    y_range = top_df['r2'].max() - top_df['r2'].min()
    ax1.set_ylim(top_df['r2'].min() - 0.1*y_range, top_df['r2'].max() + 0.2*y_range)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    # 2. Top 10: R² vs Training Time
    ax2 = axes[1]
    scatter2 = ax2.scatter(top_df['train_time'], top_df['r2'], 
                          s=200, alpha=0.8, c=colors, edgecolors='black', linewidth=2)
    ax2.set_xlabel('Training Time (seconds)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('R² Score', fontsize=14, fontweight='bold')
    ax2.set_title('Top 10 Algorithms:\nR² vs Training Time', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.4)
    
    # Add wrapped labels for top performers
    for i, row in top_df.iterrows():
        ax2.annotate(row['display_label'], 
                    (row['train_time'], row['r2']),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=9, ha='left', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor='white', 
                             alpha=0.8, edgecolor='gray'),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1'))
    
    # Set better axis limits and formatting
    ax2.set_xlim(0.001, top_df['train_time'].max() * 2)
    ax2.set_ylim(top_df['r2'].min() - 0.1*y_range, top_df['r2'].max() + 0.2*y_range)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    
    # 3. Average R² by Missing Percentage (All Data)
    ax3 = axes[2]
    missing_performance = df.groupby('Missing_Percentage')['r2'].mean().sort_values(ascending=True)
    missing_performance.plot(kind='bar', ax=ax3, color='#97a5e8')
    ax3.set_xlabel('Missing Percentage (%)', fontsize=12)
    ax3.set_ylabel('Average R² Score', fontsize=12)
    ax3.set_title('Average R² by Missing Data Percentage', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(axis='x', rotation=0)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig('streamlined_r2_analysis.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()

def print_summary_statistics(top_df, df):
    """Print key summary statistics."""
    print("="*80)
    print("STREAMLINED R² ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"   Total experiments: {len(df)}")
    print(f"   R² range: {df['r2'].min():.4f} to {df['r2'].max():.4f}")
    print(f"   Average R²: {df['r2'].mean():.4f}")
    
    print(f"\n🏆 TOP 10 BEST PERFORMERS:")
    for i, (idx, row) in enumerate(top_df.iterrows(), 1):
        print(f"   {i:2d}. R² = {row['r2']:8.4f} | {row['Algorithm']:8s} | {row['Model'][:20]:20s} | {row['n_features']:4.0f}f | {row['Missing_Percentage']:4.1f}% missing | {row['train_time']:8.4f}s")
    
    print(f"\n📈 MISSING DATA IMPACT:")
    missing_impact = df.groupby('Missing_Percentage')['r2'].agg(['count', 'mean', 'std']).round(4)
    print(missing_impact)
    
    print(f"\n⏱️ TRAINING TIME ANALYSIS (Top 10):")
    print(f"   Fastest: {top_df['train_time'].min():.4f} seconds")
    print(f"   Slowest: {top_df['train_time'].max():.4f} seconds")
    print(f"   Average: {top_df['train_time'].mean():.4f} seconds")

def save_results(top_df, df):
    """Save the analysis results."""
    # Save top 10 with key metrics
    results_df = top_df[['Workflow', 'Algorithm', 'integration_tech', 'Model', 
                        'n_features', 'Missing_Percentage', 'r2', 'train_time', 
                        'mse', 'rmse', 'mae']].round(4)
    results_df.to_csv('streamlined_top_10_analysis.csv', index=False)
    
    # Save missing data impact analysis
    missing_analysis = df.groupby('Missing_Percentage')['r2'].agg(['count', 'mean', 'std']).round(4)
    missing_analysis.to_csv('missing_data_impact_analysis.csv')
    
    print(f"\n✅ Results saved:")
    print(f"   📊 streamlined_r2_analysis.png")
    print(f"   📋 streamlined_top_10_analysis.csv")
    print(f"   📈 missing_data_impact_analysis.csv")

def main():
    """Main function to execute the streamlined analysis."""
    print("🚀 Starting Streamlined R² Analysis...")
    print("   Focus: Top 10 performers with 3 key visualizations")
    
    # Load data
    combined_df = load_and_combine_data()
    
    # Find top performers
    print(f"\n🔍 Identifying top 10 algorithms from {len(combined_df)} experiments...")
    top_df = find_best_performers(combined_df, top_n=10)
    
    # Create streamlined visualizations
    print(f"\n📊 Creating streamlined visualizations...")
    create_streamlined_visualizations(top_df, combined_df)
    
    # Print summary
    print_summary_statistics(top_df, combined_df)
    
    # Save results
    save_results(top_df, combined_df)

if __name__ == "__main__":
    main() 