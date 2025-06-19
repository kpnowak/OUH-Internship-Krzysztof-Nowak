import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def load_and_combine_data():
    """Load both CSV files and combine them."""
    # Load the datasets
    extraction_df = pd.read_csv('output_main_without_mrmr/AML/metrics/AML_extraction_cv_metrics.csv')
    selection_df = pd.read_csv('output_main_without_mrmr/AML/metrics/AML_selection_cv_metrics.csv')
    
    # Combine the datasets
    combined_df = pd.concat([extraction_df, selection_df], ignore_index=True)
    
    return combined_df, extraction_df, selection_df

def analyze_r2_distribution(df):
    """Analyze the distribution of R² values."""
    print("="*80)
    print("R² DISTRIBUTION ANALYSIS")
    print("="*80)
    
    print(f"Total experiments: {len(df)}")
    print(f"R² Statistics:")
    print(f"  Mean: {df['r2'].mean():.4f}")
    print(f"  Median: {df['r2'].median():.4f}")
    print(f"  Std Dev: {df['r2'].std():.4f}")
    print(f"  Min: {df['r2'].min():.4f}")
    print(f"  Max: {df['r2'].max():.4f}")
    print(f"  25th percentile: {df['r2'].quantile(0.25):.4f}")
    print(f"  75th percentile: {df['r2'].quantile(0.75):.4f}")
    
    # Count positive vs negative R²
    positive_r2 = (df['r2'] > 0).sum()
    negative_r2 = (df['r2'] <= 0).sum()
    print(f"\nR² Value Distribution:")
    print(f"  Positive R²: {positive_r2} ({positive_r2/len(df)*100:.1f}%)")
    print(f"  Negative R²: {negative_r2} ({negative_r2/len(df)*100:.1f}%)")
    
    # Analyze by workflow
    print(f"\nR² by Workflow:")
    workflow_stats = df.groupby('Workflow')['r2'].agg(['count', 'mean', 'std', 'min', 'max'])
    print(workflow_stats.round(4))

def find_best_performers(df, top_n=10):
    """Find the best performing algorithms (highest R²)."""
    # Sort by R² in descending order and get top N
    top_df = df.nlargest(top_n, 'r2').copy()
    
    # Create a comprehensive label for each point
    top_df['short_label'] = (top_df['Algorithm'] + '_' + 
                            top_df['Model'].str.replace('Regressor', 'R').str.replace('Regression', 'LR') + 
                            '_' + top_df['n_features'].astype(str) + 'f')
    
    top_df['full_label'] = (top_df['Workflow'] + '_' + 
                           top_df['Algorithm'] + '_' + 
                           top_df['integration_tech'] + '_' + 
                           top_df['Model'] + '_' + 
                           top_df['n_features'].astype(str) + 'f')
    
    return top_df

def create_comprehensive_visualizations(top_df, df):
    """Create comprehensive visualizations."""
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(24, 16))
    
    # Create a grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Top 10: R² vs Missing Percentage
    ax1 = fig.add_subplot(gs[0, 0])
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_df)))
    scatter1 = ax1.scatter(top_df['Missing_Percentage'], top_df['r2'], 
                          s=150, alpha=0.8, c=colors, edgecolors='black', linewidth=1)
    ax1.set_xlabel('Missing Percentage (%)', fontsize=12)
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title('Top 10: R² vs Missing Percentage', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add labels for top performers
    for i, row in top_df.iterrows():
        ax1.annotate(row['short_label'], 
                    (row['Missing_Percentage'], row['r2']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, rotation=0, ha='left',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7))
    
    # 2. Top 10: R² vs Training Time
    ax2 = fig.add_subplot(gs[0, 1])
    scatter2 = ax2.scatter(top_df['train_time'], top_df['r2'], 
                          s=150, alpha=0.8, c=colors, edgecolors='black', linewidth=1)
    ax2.set_xlabel('Training Time (seconds)', fontsize=12)
    ax2.set_ylabel('R² Score', fontsize=12)
    ax2.set_title('Top 10: R² vs Training Time', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    
    # Add labels for top performers
    for i, row in top_df.iterrows():
        ax2.annotate(row['short_label'], 
                    (row['train_time'], row['r2']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, rotation=0, ha='left',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7))
    
    # 3. R² Distribution Histogram
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(df['r2'], bins=50, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('R² Score', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('R² Score Distribution (All Experiments)', fontsize=14, fontweight='bold')
    ax3.axvline(df['r2'].mean(), color='red', linestyle='--', label=f'Mean: {df["r2"].mean():.3f}')
    ax3.axvline(df['r2'].median(), color='green', linestyle='--', label=f'Median: {df["r2"].median():.3f}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Performance by Algorithm
    ax4 = fig.add_subplot(gs[1, 0])
    algorithm_performance = df.groupby('Algorithm')['r2'].mean().sort_values(ascending=True)
    algorithm_performance.plot(kind='barh', ax=ax4)
    ax4.set_xlabel('Average R² Score', fontsize=12)
    ax4.set_title('Average R² by Algorithm', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # 5. Performance by Model
    ax5 = fig.add_subplot(gs[1, 1])
    model_performance = df.groupby('Model')['r2'].mean().sort_values(ascending=True)
    model_performance.plot(kind='barh', ax=ax5)
    ax5.set_xlabel('Average R² Score', fontsize=12)
    ax5.set_title('Average R² by Model', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='x')
    
    # 6. Performance by Integration Technique
    ax6 = fig.add_subplot(gs[1, 2])
    integration_performance = df.groupby('integration_tech')['r2'].mean().sort_values(ascending=True)
    integration_performance.plot(kind='barh', ax=ax6)
    ax6.set_xlabel('Average R² Score', fontsize=12)
    ax6.set_title('Average R² by Integration Technique', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='x')
    
    # 7. MSE vs R² relationship
    ax7 = fig.add_subplot(gs[2, 0])
    # Filter out extreme outliers for better visualization
    q99 = df['mse'].quantile(0.99)
    filtered_df = df[df['mse'] <= q99]
    ax7.scatter(filtered_df['mse'], filtered_df['r2'], alpha=0.5, s=20)
    ax7.set_xlabel('MSE (99th percentile cutoff)', fontsize=12)
    ax7.set_ylabel('R² Score', fontsize=12)
    ax7.set_title('MSE vs R² Relationship', fontsize=14, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    # 8. Performance by Number of Features
    ax8 = fig.add_subplot(gs[2, 1])
    feature_performance = df.groupby('n_features')['r2'].mean().sort_values(ascending=True)
    feature_performance.plot(kind='bar', ax=ax8)
    ax8.set_xlabel('Number of Features', fontsize=12)
    ax8.set_ylabel('Average R² Score', fontsize=12)
    ax8.set_title('Average R² by Number of Features', fontsize=14, fontweight='bold')
    ax8.grid(True, alpha=0.3, axis='y')
    ax8.tick_params(axis='x', rotation=0)
    
    # 9. Performance by Missing Percentage
    ax9 = fig.add_subplot(gs[2, 2])
    missing_performance = df.groupby('Missing_Percentage')['r2'].mean().sort_values(ascending=True)
    missing_performance.plot(kind='bar', ax=ax9)
    ax9.set_xlabel('Missing Percentage (%)', fontsize=12)
    ax9.set_ylabel('Average R² Score', fontsize=12)
    ax9.set_title('Average R² by Missing Data Percentage', fontsize=14, fontweight='bold')
    ax9.grid(True, alpha=0.3, axis='y')
    ax9.tick_params(axis='x', rotation=0)
    
    plt.suptitle('Comprehensive R² Analysis: AML Dataset', fontsize=20, fontweight='bold', y=0.98)
    plt.savefig('comprehensive_r2_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_detailed_summary_report(top_df, df, extraction_df, selection_df):
    """Create a detailed summary report."""
    print("\n" + "="*100)
    print("COMPREHENSIVE ANALYSIS REPORT")
    print("="*100)
    
    # Overall statistics
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"   Total experiments: {len(df)}")
    print(f"   Extraction experiments: {len(extraction_df)}")
    print(f"   Selection experiments: {len(selection_df)}")
    
    # Best performers
    print(f"\n🏆 TOP 10 BEST PERFORMING ALGORITHMS:")
    for i, (idx, row) in enumerate(top_df.iterrows(), 1):
        print(f"   {i:2d}. R² = {row['r2']:8.4f} | {row['Algorithm']:12s} | {row['Model']:20s} | {row['n_features']:4.0f}f | {row['Missing_Percentage']:4.1f}% missing")
    
    # Algorithm analysis
    print(f"\n🔬 ALGORITHM PERFORMANCE RANKING:")
    algorithm_stats = df.groupby('Algorithm')['r2'].agg(['count', 'mean', 'std', 'max']).round(4)
    algorithm_stats = algorithm_stats.sort_values('mean', ascending=False)
    print(algorithm_stats)
    
    # Model analysis
    print(f"\n🤖 MODEL PERFORMANCE RANKING:")
    model_stats = df.groupby('Model')['r2'].agg(['count', 'mean', 'std', 'max']).round(4)
    model_stats = model_stats.sort_values('mean', ascending=False)
    print(model_stats)
    
    # Integration technique analysis
    print(f"\n🔗 INTEGRATION TECHNIQUE PERFORMANCE:")
    integration_stats = df.groupby('integration_tech')['r2'].agg(['count', 'mean', 'std', 'max']).round(4)
    integration_stats = integration_stats.sort_values('mean', ascending=False)
    print(integration_stats)
    
    # Missing data impact
    print(f"\n❓ MISSING DATA IMPACT:")
    missing_stats = df.groupby('Missing_Percentage')['r2'].agg(['count', 'mean', 'std']).round(4)
    print(missing_stats)
    
    # Feature count impact
    print(f"\n📏 FEATURE COUNT IMPACT:")
    feature_stats = df.groupby('n_features')['r2'].agg(['count', 'mean', 'std']).round(4)
    print(feature_stats)
    
    # Workflow comparison
    print(f"\n⚙️ WORKFLOW COMPARISON:")
    workflow_stats = df.groupby('Workflow')['r2'].agg(['count', 'mean', 'std', 'max']).round(4)
    print(workflow_stats)

def save_detailed_results(top_df, df):
    """Save detailed results to files."""
    # Save top performers with all details
    detailed_top = top_df[['Workflow', 'Algorithm', 'integration_tech', 'Model', 
                          'n_features', 'Missing_Percentage', 'r2', 'train_time', 
                          'mse', 'rmse', 'mae', 'early_stopping_used', 
                          'best_validation_score']].round(4)
    detailed_top.to_csv('top_10_detailed_analysis.csv', index=False)
    
    # Save comprehensive statistics
    stats_summary = []
    
    # Algorithm stats
    algorithm_stats = df.groupby('Algorithm')['r2'].agg(['count', 'mean', 'std', 'min', 'max']).round(4)
    algorithm_stats['category'] = 'Algorithm'
    algorithm_stats['name'] = algorithm_stats.index
    stats_summary.append(algorithm_stats.reset_index(drop=True))
    
    # Model stats
    model_stats = df.groupby('Model')['r2'].agg(['count', 'mean', 'std', 'min', 'max']).round(4)
    model_stats['category'] = 'Model'
    model_stats['name'] = model_stats.index
    stats_summary.append(model_stats.reset_index(drop=True))
    
    # Integration stats
    integration_stats = df.groupby('integration_tech')['r2'].agg(['count', 'mean', 'std', 'min', 'max']).round(4)
    integration_stats['category'] = 'Integration'
    integration_stats['name'] = integration_stats.index
    stats_summary.append(integration_stats.reset_index(drop=True))
    
    # Combine all stats
    all_stats = pd.concat(stats_summary, ignore_index=True)
    all_stats = all_stats[['category', 'name', 'count', 'mean', 'std', 'min', 'max']]
    all_stats.to_csv('comprehensive_performance_statistics.csv', index=False)

def main():
    """Main function to execute the comprehensive analysis."""
    print("🚀 Starting Comprehensive R² Analysis...")
    
    # Load data
    combined_df, extraction_df, selection_df = load_and_combine_data()
    
    # Analyze R² distribution
    analyze_r2_distribution(combined_df)
    
    # Find top performers
    print(f"\n🔍 Finding top 10 algorithms...")
    top_df = find_best_performers(combined_df, top_n=10)
    
    # Create visualizations
    print(f"\n📊 Creating comprehensive visualizations...")
    create_comprehensive_visualizations(top_df, combined_df)
    
    # Create detailed report
    create_detailed_summary_report(top_df, combined_df, extraction_df, selection_df)
    
    # Save results
    save_detailed_results(top_df, combined_df)
    
    print(f"\n✅ Analysis complete! Files saved:")
    print(f"   📊 comprehensive_r2_analysis.png")
    print(f"   📋 top_10_detailed_analysis.csv")
    print(f"   📈 comprehensive_performance_statistics.csv")

if __name__ == "__main__":
    main() 