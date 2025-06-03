import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

# Set up the plotting style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (16, 12)  # Even larger figure size
plt.rcParams['font.size'] = 10

def create_3d_plot(df, title, label_column, output_path):
    """Create a 3D scatter plot showing RMSE vs Robustness vs Training Time"""
    
    fig = plt.figure(figsize=(16, 12))  # Larger figure size
    # Adjust subplot positioning: [left, bottom, width, height]
    # Larger left/right margins (0.2 left, 0.15 right), smaller top/bottom margins
    ax = fig.add_subplot(111, projection='3d', position=[0.2, 0.05, 0.9, 0.9])
    
    # Extract data
    rmse = df['baseline_rmse'].values
    robustness = df['robustness'].values
    train_time = df['train_time_avg'].values
    labels = df[label_column].values
    
    # Create color map
    colors = plt.cm.viridis(np.linspace(0, 1, len(df)))
    
    # Create 3D scatter plot
    scatter = ax.scatter(rmse, robustness, train_time, 
                        c=colors, s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Add labels for each point
    for i, label in enumerate(labels):
        ax.text(rmse[i], robustness[i], train_time[i], f'  {label}', 
                fontsize=8, ha='left', va='bottom')
    
    # Set labels and title with increased padding
    ax.set_xlabel('RMSE (Lower is Better)', fontsize=12, labelpad=20)
    ax.set_ylabel('Robustness (Higher is Better)', fontsize=12, labelpad=20)
    ax.set_zlabel('Training Time (seconds)', fontsize=12, labelpad=20)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=30)
    
    # Improve the view angle
    ax.view_init(elev=20, azim=45)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Save the plot with extra padding and margins
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()  # Close the figure instead of showing it
    print(f"Plot saved to: {output_path}")

def process_dataset(dataset_name):
    """Process a single dataset (AML or Sarcoma)"""
    print(f"\n{'='*60}")
    print(f"PROCESSING {dataset_name.upper()} DATASET")
    print(f"{'='*60}")
    
    # Create output directory if it doesn't exist
    output_dir = f"final_results/{dataset_name}/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Top Algorithms
    print(f"\nCreating plot for {dataset_name} top algorithms...")
    df_algorithms = pd.read_csv(f'final_results/{dataset_name}/top_algorithms.csv')
    create_3d_plot(df_algorithms, 
                   f'{dataset_name}: Algorithm Performance Analysis\n(RMSE vs Robustness vs Training Time)',
                   'Algorithm',
                   f'{output_dir}/top_algorithms_3d.png')
    
    # Plot 2: Top Feature Settings
    print(f"Creating plot for {dataset_name} top feature settings...")
    df_features = pd.read_csv(f'final_results/{dataset_name}/top_feature_settings.csv')
    # Create a label combining n_features and n_components
    df_features['label'] = df_features['n_features'].astype(int).astype(str) + ' features'
    create_3d_plot(df_features,
                   f'{dataset_name}: Feature Settings Performance Analysis\n(RMSE vs Robustness vs Training Time)',
                   'label',
                   f'{output_dir}/top_feature_settings_3d.png')
    
    # Plot 3: Top Integration Techniques
    print(f"Creating plot for {dataset_name} top integration techniques...")
    df_integration = pd.read_csv(f'final_results/{dataset_name}/top_integration_tech.csv')
    create_3d_plot(df_integration,
                   f'{dataset_name}: Integration Technique Performance Analysis\n(RMSE vs Robustness vs Training Time)',
                   'integration_tech',
                   f'{output_dir}/top_integration_tech_3d.png')
    
    # Plot 4: Top Models
    print(f"Creating plot for {dataset_name} top models...")
    df_models = pd.read_csv(f'final_results/{dataset_name}/top_models.csv')
    create_3d_plot(df_models,
                   f'{dataset_name}: Model Performance Analysis\n(RMSE vs Robustness vs Training Time)',
                   'Model',
                   f'{output_dir}/top_models_3d.png')
    
    print(f"\nAll {dataset_name} plots have been created successfully!")
    
    # Print summary statistics
    print(f"\n{dataset_name.upper()} SUMMARY STATISTICS:")
    print("-" * 50)
    
    datasets = [
        ("Algorithms", df_algorithms),
        ("Feature Settings", df_features),
        ("Integration Techniques", df_integration),
        ("Models", df_models)
    ]
    
    for name, df in datasets:
        print(f"\n{name}:")
        print(f"  RMSE range: {df['baseline_rmse'].min():.2f} - {df['baseline_rmse'].max():.2f}")
        print(f"  Robustness range: {df['robustness'].min():.3f} - {df['robustness'].max():.3f}")
        print(f"  Training time range: {df['train_time_avg'].min():.2f} - {df['train_time_avg'].max():.2f} seconds")

def main():
    print("Creating 3D plots for regression datasets (AML and Sarcoma)...")
    
    # Process both datasets
    datasets = ['AML', 'Sarcoma']
    
    for dataset in datasets:
        try:
            process_dataset(dataset)
        except Exception as e:
            print(f"Error processing {dataset}: {str(e)}")
            continue
    
    print(f"\n{'='*60}")
    print("ALL DATASETS PROCESSED SUCCESSFULLY!")
    print(f"{'='*60}")
    print("\nPlots saved in:")
    for dataset in datasets:
        print(f"  - final_results/{dataset}/plots/")
    print("\nFiles created for each dataset:")
    print("  - top_algorithms_3d.png")
    print("  - top_feature_settings_3d.png") 
    print("  - top_integration_tech_3d.png")
    print("  - top_models_3d.png")

if __name__ == "__main__":
    main() 