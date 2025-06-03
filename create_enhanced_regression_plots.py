import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import seaborn as sns

# Set up the plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10

def create_comprehensive_plots(df, title_base, label_column, output_prefix):
    """Create both 3D and 2D plots for comprehensive analysis"""
    
    # Extract data
    rmse = df['baseline_rmse'].values
    robustness = df['robustness'].values
    train_time = df['train_time_avg'].values
    labels = df[label_column].values
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 3D Plot
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    colors = plt.cm.viridis(np.linspace(0, 1, len(df)))
    scatter = ax1.scatter(rmse, robustness, train_time, 
                         c=colors, s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    for i, label in enumerate(labels):
        ax1.text(rmse[i], robustness[i], train_time[i], f'  {label}', 
                fontsize=7, ha='left', va='bottom')
    
    ax1.set_xlabel('RMSE', fontsize=10)
    ax1.set_ylabel('Robustness', fontsize=10)
    ax1.set_zlabel('Training Time (s)', fontsize=10)
    ax1.set_title(f'{title_base}\n3D View', fontsize=12, fontweight='bold')
    ax1.view_init(elev=20, azim=45)
    ax1.grid(True, alpha=0.3)
    
    # 2. RMSE vs Robustness
    ax2 = fig.add_subplot(2, 3, 2)
    scatter2 = ax2.scatter(rmse, robustness, c=colors, s=100, alpha=0.7, edgecolors='black')
    for i, label in enumerate(labels):
        ax2.annotate(label, (rmse[i], robustness[i]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8, ha='left')
    ax2.set_xlabel('RMSE (Lower is Better)', fontsize=10)
    ax2.set_ylabel('Robustness (Higher is Better)', fontsize=10)
    ax2.set_title('RMSE vs Robustness', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. RMSE vs Training Time
    ax3 = fig.add_subplot(2, 3, 3)
    scatter3 = ax3.scatter(rmse, train_time, c=colors, s=100, alpha=0.7, edgecolors='black')
    for i, label in enumerate(labels):
        ax3.annotate(label, (rmse[i], train_time[i]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8, ha='left')
    ax3.set_xlabel('RMSE (Lower is Better)', fontsize=10)
    ax3.set_ylabel('Training Time (seconds)', fontsize=10)
    ax3.set_title('RMSE vs Training Time', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Robustness vs Training Time
    ax4 = fig.add_subplot(2, 3, 4)
    scatter4 = ax4.scatter(robustness, train_time, c=colors, s=100, alpha=0.7, edgecolors='black')
    for i, label in enumerate(labels):
        ax4.annotate(label, (robustness[i], train_time[i]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8, ha='left')
    ax4.set_xlabel('Robustness (Higher is Better)', fontsize=10)
    ax4.set_ylabel('Training Time (seconds)', fontsize=10)
    ax4.set_title('Robustness vs Training Time', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Performance Summary Bar Chart
    ax5 = fig.add_subplot(2, 3, 5)
    # Normalize metrics for comparison (0-1 scale)
    rmse_norm = 1 - (rmse - rmse.min()) / (rmse.max() - rmse.min()) if rmse.max() != rmse.min() else np.ones_like(rmse)
    robustness_norm = robustness
    time_norm = 1 - (train_time - train_time.min()) / (train_time.max() - train_time.min()) if train_time.max() != train_time.min() else np.ones_like(train_time)
    
    x = np.arange(len(labels))
    width = 0.25
    
    bars1 = ax5.bar(x - width, rmse_norm, width, label='RMSE (normalized)', alpha=0.8)
    bars2 = ax5.bar(x, robustness_norm, width, label='Robustness', alpha=0.8)
    bars3 = ax5.bar(x + width, time_norm, width, label='Speed (normalized)', alpha=0.8)
    
    ax5.set_xlabel('Methods', fontsize=10)
    ax5.set_ylabel('Normalized Score (Higher is Better)', fontsize=10)
    ax5.set_title('Performance Comparison\n(All metrics normalized 0-1)', fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(labels, rotation=45, ha='right')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Composite Score Analysis
    ax6 = fig.add_subplot(2, 3, 6)
    if 'composite_score' in df.columns:
        composite_scores = df['composite_score'].values
        bars = ax6.bar(range(len(labels)), composite_scores, color=colors, alpha=0.8, edgecolor='black')
        ax6.set_xlabel('Methods', fontsize=10)
        ax6.set_ylabel('Composite Score', fontsize=10)
        ax6.set_title('Overall Performance Score', fontsize=12, fontweight='bold')
        ax6.set_xticks(range(len(labels)))
        ax6.set_xticklabels(labels, rotation=45, ha='right')
        ax6.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, composite_scores)):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=8)
    else:
        ax6.text(0.5, 0.5, 'Composite Score\nNot Available', ha='center', va='center',
                transform=ax6.transAxes, fontsize=12)
        ax6.set_title('Composite Score Analysis', fontsize=12, fontweight='bold')
    
    plt.suptitle(f'{title_base} - Comprehensive Analysis', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save the plot
    output_path = f'{output_prefix}_comprehensive.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of showing
    print(f"Comprehensive plot saved to: {output_path}")

def create_summary_table(df, name, label_column):
    """Create a summary table with key metrics"""
    print(f"\n{name.upper()} ANALYSIS:")
    print("-" * 50)
    
    # Sort by composite score if available, otherwise by RMSE
    if 'composite_score' in df.columns:
        df_sorted = df.sort_values('composite_score', ascending=False)
        print("Ranked by Composite Score (Higher is Better):")
    else:
        df_sorted = df.sort_values('baseline_rmse', ascending=True)
        print("Ranked by RMSE (Lower is Better):")
    
    for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
        label = row[label_column]
        rmse = row['baseline_rmse']
        robustness = row['robustness']
        train_time = row['train_time_avg']
        
        print(f"{i:2d}. {label:<20} | RMSE: {rmse:8.2f} | Robustness: {robustness:.3f} | Time: {train_time:.2f}s", end="")
        
        if 'composite_score' in df.columns:
            composite = row['composite_score']
            print(f" | Score: {composite:.3f}")
        else:
            print()

def process_dataset(dataset_name):
    """Process a single dataset (AML or Sarcoma)"""
    print(f"\n{'='*60}")
    print(f"PROCESSING {dataset_name.upper()} DATASET - COMPREHENSIVE ANALYSIS")
    print(f"{'='*60}")
    
    # Create output directory if it doesn't exist
    output_dir = f"final_results/{dataset_name}/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Top Algorithms
    print(f"\n1. Processing {dataset_name} top algorithms...")
    df_algorithms = pd.read_csv(f'final_results/{dataset_name}/top_algorithms.csv')
    create_comprehensive_plots(df_algorithms, 
                              f'{dataset_name}: Algorithm Performance Analysis',
                              'Algorithm',
                              f'{output_dir}/top_algorithms')
    create_summary_table(df_algorithms, f'{dataset_name} Algorithms', 'Algorithm')
    
    # Plot 2: Top Feature Settings
    print(f"\n2. Processing {dataset_name} top feature settings...")
    df_features = pd.read_csv(f'final_results/{dataset_name}/top_feature_settings.csv')
    df_features['label'] = df_features['n_features'].astype(int).astype(str) + ' features'
    create_comprehensive_plots(df_features,
                              f'{dataset_name}: Feature Settings Performance Analysis',
                              'label',
                              f'{output_dir}/top_feature_settings')
    create_summary_table(df_features, f'{dataset_name} Feature Settings', 'label')
    
    # Plot 3: Top Integration Techniques
    print(f"\n3. Processing {dataset_name} top integration techniques...")
    df_integration = pd.read_csv(f'final_results/{dataset_name}/top_integration_tech.csv')
    create_comprehensive_plots(df_integration,
                              f'{dataset_name}: Integration Technique Performance Analysis',
                              'integration_tech',
                              f'{output_dir}/top_integration_tech')
    create_summary_table(df_integration, f'{dataset_name} Integration Techniques', 'integration_tech')
    
    # Plot 4: Top Models
    print(f"\n4. Processing {dataset_name} top models...")
    df_models = pd.read_csv(f'final_results/{dataset_name}/top_models.csv')
    create_comprehensive_plots(df_models,
                              f'{dataset_name}: Model Performance Analysis',
                              'Model',
                              f'{output_dir}/top_models')
    create_summary_table(df_models, f'{dataset_name} Models', 'Model')

def main():
    print("Creating comprehensive plots for regression datasets (AML and Sarcoma)...")
    
    # Process both datasets
    datasets = ['AML', 'Sarcoma']
    
    for dataset in datasets:
        try:
            process_dataset(dataset)
        except Exception as e:
            print(f"Error processing {dataset}: {str(e)}")
            continue
    
    print(f"\n{'='*60}")
    print("ALL COMPREHENSIVE PLOTS CREATED SUCCESSFULLY!")
    print(f"{'='*60}")
    
    # Overall insights
    print("\nKEY INSIGHTS:")
    print("-" * 20)
    print("• Lower RMSE values indicate better prediction accuracy")
    print("• Higher robustness values indicate more stable performance")
    print("• Training time shows computational efficiency")
    print("• Composite scores combine all metrics for overall ranking")
    
    print("\nPlots saved in:")
    for dataset in datasets:
        print(f"  - final_results/{dataset}/plots/")
    print("\nFiles created for each dataset:")
    print("  - top_algorithms_comprehensive.png")
    print("  - top_feature_settings_comprehensive.png") 
    print("  - top_integration_tech_comprehensive.png")
    print("  - top_models_comprehensive.png")

if __name__ == "__main__":
    main() 