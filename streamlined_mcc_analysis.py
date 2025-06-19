import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import textwrap
import os
import glob

def load_and_combine_classification_data():
    """Load all classification CSV files from final_results and combine them."""
    combined_data = []
    
    # Classification datasets
    classification_datasets = ['Breast', 'Colon']
    
    print("Loading classification data...")
    for dataset in classification_datasets:
        for missing_pct in [0, 20, 50]:
            file_path = f'final_results/{dataset}/all_runs_ranked_{missing_pct}pct_missing.csv'
            if os.path.exists(file_path):
                print(f"  Loading {file_path}...")
                df = pd.read_csv(file_path)
                df['Dataset'] = dataset
                df['Missing_Percentage'] = missing_pct
                combined_data.append(df)
            else:
                print(f"  Warning: {file_path} not found")
    
    if combined_data:
        combined_df = pd.concat(combined_data, ignore_index=True)
        print(f"  Total experiments loaded: {len(combined_df)}")
        return combined_df
    else:
        raise FileNotFoundError("No classification data files found!")

def wrap_text(text, width=15):
    """Wrap text to specified width."""
    if len(str(text)) <= width:
        return str(text)
    return '\n'.join(textwrap.wrap(str(text), width=width))

def find_best_mcc_performers(df, top_n=10):
    """Find the best performing algorithms (highest MCC)."""
    # Sort by MCC in descending order and get top N
    top_df = df.nlargest(top_n, 'mcc').copy()
    
    # Create wrapped labels for better display
    top_df['wrapped_model'] = top_df['Model'].apply(lambda x: wrap_text(x, 15))
    top_df['wrapped_algorithm'] = top_df['Algorithm'].apply(lambda x: wrap_text(x, 12))
    top_df['wrapped_integration'] = top_df['integration_tech'].apply(lambda x: wrap_text(x, 12))
    
    # Create comprehensive label with wrapping
    top_df['display_label'] = (top_df['wrapped_algorithm'] + '\n' + 
                              top_df['wrapped_model'] + '\n' + 
                              top_df['n_features'].astype(str) + 'f')
    
    return top_df

def create_streamlined_mcc_visualizations(top_df, df):
    """Create the 3 requested visualizations with proper text wrapping focused on MCC."""
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle('', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Define colors for consistent visualization
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_df)))
    
    # 1. Top 10: MCC vs Missing Percentage
    ax1 = axes[0]
    scatter1 = ax1.scatter(top_df['Missing_Percentage'], top_df['mcc'], 
                          s=200, alpha=0.8, c=colors, edgecolors='black', linewidth=2)
    ax1.set_xlabel('Missing Percentage (%)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('MCC Score', fontsize=14, fontweight='bold')
    ax1.set_title('Top 10 Algorithms:\nMCC vs Missing Percentage', fontsize=16, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.4)
    
    # Add wrapped labels for top performers
    for i, row in top_df.iterrows():
        ax1.annotate(row['display_label'], 
                    (row['Missing_Percentage'], row['mcc']),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=9, ha='left', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor='white', 
                             alpha=0.8, edgecolor='gray'),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1'))
    
    # Set better axis limits and formatting
    ax1.set_xlim(-0.1, 55)
    mcc_range = top_df['mcc'].max() - top_df['mcc'].min()
    ax1.set_ylim(top_df['mcc'].min() - 0.1*mcc_range, top_df['mcc'].max() + 0.2*mcc_range)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    # 2. Top 10: MCC vs Training Time
    ax2 = axes[1]
    scatter2 = ax2.scatter(top_df['train_time'], top_df['mcc'], 
                          s=200, alpha=0.8, c=colors, edgecolors='black', linewidth=2)
    ax2.set_xlabel('Training Time (seconds)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('MCC Score', fontsize=14, fontweight='bold')
    ax2.set_title('Top 10 Algorithms:\nMCC vs Training Time', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.4)
    
    # Add wrapped labels for top performers
    for i, row in top_df.iterrows():
        ax2.annotate(row['display_label'], 
                    (row['train_time'], row['mcc']),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=9, ha='left', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor='white', 
                             alpha=0.8, edgecolor='gray'),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1'))
    
    # Set better axis limits and formatting
    ax2.set_xlim(0.001, top_df['train_time'].max() * 2)
    ax2.set_ylim(top_df['mcc'].min() - 0.1*mcc_range, top_df['mcc'].max() + 0.2*mcc_range)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    
    # 3. Average MCC by Missing Percentage (All Data)
    ax3 = axes[2]
    missing_performance = df.groupby('Missing_Percentage')['mcc'].mean().sort_values(ascending=True)
    missing_performance.plot(kind='bar', ax=ax3, color='#97a5e8')
    ax3.set_xlabel('Missing Percentage (%)', fontsize=12)
    ax3.set_ylabel('Average MCC Score', fontsize=12)
    ax3.set_title('Average MCC by Missing Data Percentage', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(axis='x', rotation=0)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig('streamlined_mcc_analysis.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()

def print_mcc_summary_statistics(top_df, df):
    """Print key summary statistics for MCC analysis."""
    print("="*80)
    print("STREAMLINED MCC ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"   Total experiments: {len(df)}")
    print(f"   MCC range: {df['mcc'].min():.4f} to {df['mcc'].max():.4f}")
    print(f"   Average MCC: {df['mcc'].mean():.4f}")
    print(f"   Datasets: {', '.join(df['Dataset'].unique())}")
    
    print(f"\n🏆 TOP 10 BEST MCC PERFORMERS:")
    for i, (idx, row) in enumerate(top_df.iterrows(), 1):
        print(f"   {i:2d}. MCC = {row['mcc']:8.4f} | {row['Dataset']:8s} | {row['Algorithm']:8s} | {row['Model'][:20]:20s} | {row['n_features']:4.0f}f | {row['Missing_Percentage']:4.1f}% missing | {row['train_time']:8.4f}s")
    
    print(f"\n📈 MISSING DATA IMPACT ON MCC:")
    missing_impact = df.groupby('Missing_Percentage')['mcc'].agg(['count', 'mean', 'std']).round(4)
    print(missing_impact)
    
    print(f"\n📊 DATASET COMPARISON:")
    dataset_performance = df.groupby('Dataset')['mcc'].agg(['count', 'mean', 'std', 'max']).round(4)
    print(dataset_performance)
    
    print(f"\n⏱️ TRAINING TIME ANALYSIS (Top 10):")
    print(f"   Fastest: {top_df['train_time'].min():.4f} seconds")
    print(f"   Slowest: {top_df['train_time'].max():.4f} seconds")
    print(f"   Average: {top_df['train_time'].mean():.4f} seconds")
    
    print(f"\n🎯 CLASSIFICATION METRICS CORRELATION (Top 10):")
    metrics_corr = top_df[['mcc', 'accuracy', 'precision', 'recall', 'f1', 'auc']].corr()['mcc'].round(4)
    print("   MCC correlation with other metrics:")
    for metric, corr in metrics_corr.items():
        if metric != 'mcc':
            print(f"     {metric:12s}: {corr:6.4f}")

def save_mcc_results(top_df, df):
    """Save the MCC analysis results."""
    # Save top 10 with key metrics
    results_df = top_df[['Dataset', 'Workflow', 'Algorithm', 'integration_tech', 'Model', 
                        'n_features', 'Missing_Percentage', 'mcc', 'accuracy', 'precision', 
                        'recall', 'f1', 'auc', 'train_time']].round(4)
    results_df.to_csv('streamlined_top_10_mcc_analysis.csv', index=False)
    
    # Save missing data impact analysis
    missing_analysis = df.groupby('Missing_Percentage')['mcc'].agg(['count', 'mean', 'std']).round(4)
    missing_analysis.to_csv('missing_data_impact_mcc_analysis.csv')
    
    # Save dataset comparison
    dataset_analysis = df.groupby('Dataset')[['mcc', 'accuracy', 'f1', 'auc']].agg(['mean', 'std', 'max']).round(4)
    dataset_analysis.to_csv('dataset_mcc_comparison.csv')
    
    # Save comprehensive performance statistics
    comprehensive_stats = df.describe()[['mcc', 'accuracy', 'precision', 'recall', 'f1', 'auc', 'train_time']].round(4)
    comprehensive_stats.to_csv('comprehensive_mcc_performance_statistics.csv')
    
    print(f"\n✅ Results saved:")
    print(f"   📊 streamlined_mcc_analysis.png")
    print(f"   📋 streamlined_top_10_mcc_analysis.csv")
    print(f"   📈 missing_data_impact_mcc_analysis.csv")
    print(f"   📊 dataset_mcc_comparison.csv")
    print(f"   📈 comprehensive_mcc_performance_statistics.csv")

def analyze_algorithm_patterns(df):
    """Analyze patterns in algorithm performance."""
    print(f"\n🔬 ALGORITHM PERFORMANCE PATTERNS:")
    
    # Best algorithms by average MCC
    algo_performance = df.groupby('Algorithm')['mcc'].agg(['count', 'mean', 'std', 'max']).round(4)
    algo_performance = algo_performance.sort_values('mean', ascending=False)
    print(f"\n   Top 5 Algorithms by Average MCC:")
    for i, (algo, stats) in enumerate(algo_performance.head().iterrows(), 1):
        print(f"   {i}. {algo:15s}: Avg={stats['mean']:6.4f}, Max={stats['max']:6.4f}, Count={stats['count']:3.0f}")
    
    # Best integration techniques
    integration_performance = df.groupby('integration_tech')['mcc'].agg(['count', 'mean', 'std', 'max']).round(4)
    integration_performance = integration_performance.sort_values('mean', ascending=False)
    print(f"\n   Top 5 Integration Techniques by Average MCC:")
    for i, (tech, stats) in enumerate(integration_performance.head().iterrows(), 1):
        print(f"   {i}. {tech:20s}: Avg={stats['mean']:6.4f}, Max={stats['max']:6.4f}, Count={stats['count']:3.0f}")
    
    # Best models
    model_performance = df.groupby('Model')['mcc'].agg(['count', 'mean', 'std', 'max']).round(4)
    model_performance = model_performance.sort_values('mean', ascending=False)
    print(f"\n   Top 3 Models by Average MCC:")
    for i, (model, stats) in enumerate(model_performance.head(3).iterrows(), 1):
        print(f"   {i}. {model:25s}: Avg={stats['mean']:6.4f}, Max={stats['max']:6.4f}, Count={stats['count']:3.0f}")

def main():
    """Main function to execute the streamlined MCC analysis."""
    print("🚀 Starting Streamlined MCC Analysis...")
    print("   Focus: Top 10 performers with 3 key visualizations")
    print("   Target: Classification algorithms using Matthews Correlation Coefficient")
    
    # Load data
    try:
        combined_df = load_and_combine_classification_data()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    
    # Check if MCC column exists
    if 'mcc' not in combined_df.columns:
        print("❌ Error: MCC column not found in the data!")
        print(f"Available columns: {list(combined_df.columns)}")
        return
    
    # Remove rows with NaN MCC values
    initial_count = len(combined_df)
    combined_df = combined_df.dropna(subset=['mcc'])
    final_count = len(combined_df)
    if initial_count != final_count:
        print(f"   Removed {initial_count - final_count} rows with missing MCC values")
    
    # Find top performers
    print(f"\n🔍 Identifying top 10 algorithms from {len(combined_df)} experiments...")
    top_df = find_best_mcc_performers(combined_df, top_n=10)
    
    # Create streamlined visualizations
    print(f"\n📊 Creating streamlined MCC visualizations...")
    create_streamlined_mcc_visualizations(top_df, combined_df)
    
    # Print summary
    print_mcc_summary_statistics(top_df, combined_df)
    
    # Analyze algorithm patterns
    analyze_algorithm_patterns(combined_df)
    
    # Save results
    save_mcc_results(top_df, combined_df)
    
    print(f"\n🎉 MCC Analysis Complete!")

if __name__ == "__main__":
    main() 