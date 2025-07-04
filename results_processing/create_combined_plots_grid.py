#!/usr/bin/env python3
"""
Combined Plots Generator - Creates 10 plots in a 5x2 grid

This script generates all visualization plots in a single image:
- Row 1: MCC vs Fit Time (All Categories), MCC vs Fit Time (Top 20)
- Row 2: MCC vs Missing (All Categories), MCC vs Missing (Worst 20)
- Row 3: MCC vs Missing (Top 20), R² vs Fit Time (All Categories)
- Row 4: R² vs Fit Time (Top 20), R² vs Missing (All Categories)
- Row 5: R² vs Missing (Worst 20), R² vs Missing (Top 20)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Define constants
EXTRACTORS_CLASSIFICATION = ['PCA', 'KPCA', 'FA', 'LDA', 'PLS-DA', 'SparsePLS']
SELECTORS_CLASSIFICATION = ['ElasticNetFS', 'RFImportance', 'VarianceFTest', 'LASSO', 'LogisticL1']
EXTRACTORS_REGRESSION = ['PCA', 'KPCA', 'FA', 'PLS', 'KPLS', 'SparsePLS']
SELECTORS_REGRESSION = ['ElasticNetFS', 'RFImportance', 'VarianceFTest', 'LASSO', 'f_regressionFS']

# Color schemes
colors = {
    'extractor': 'royalblue',
    'selector': 'gold',
    'fusion': 'forestgreen',
    'model': 'firebrick'
}
light_bg = {
    'extractor': '#cfe2ff',
    'selector': '#fff9cc',
    'fusion': '#d6f5d6',
    'model': '#ffd6d6'
}
labels = {
    'extractor': 'Extractor',
    'selector': 'Selector',
    'fusion': 'Fusion Technique',
    'model': 'Model'
}

def load_classification_data():
    """Load classification data files."""
    try:
        extractor_selector = pd.read_csv('results/classification/combined/extractor_selector_rankings.csv')
        fusion = pd.read_csv('results/classification/combined/fusion_technique_rankings.csv')
        model = pd.read_csv('results/classification/combined/model_rankings.csv')
        top_50 = pd.read_csv('results/classification/combined/top_50_combinations_overall.csv')
        worst_50 = pd.read_csv('results/classification/combined/worst_50_combinations_overall.csv')
        return extractor_selector, fusion, model, top_50, worst_50
    except FileNotFoundError as e:
        print(f"Classification data file not found: {e}")
        return None, None, None, None, None

def load_regression_data():
    """Load regression data files."""
    try:
        extractor_selector = pd.read_csv('results/regression/combined/extractor_selector_rankings.csv')
        fusion = pd.read_csv('results/regression/combined/fusion_technique_rankings.csv')
        model = pd.read_csv('results/regression/combined/model_rankings.csv')
        top_50 = pd.read_csv('results/regression/combined/top_50_combinations_overall.csv')
        worst_50 = pd.read_csv('results/regression/combined/worst_50_combinations_overall.csv')
        return extractor_selector, fusion, model, top_50, worst_50
    except FileNotFoundError as e:
        print(f"Regression data file not found: {e}")
        return None, None, None, None, None

def prepare_category_data(extractor_selector, fusion, model, extractors, selectors, x_col, y_col):
    """Prepare data with category assignments."""
    data = []
    for idx, row in extractor_selector.iterrows():
        algo = row['algorithm']
        if algo in extractors:
            cat = 'extractor'
        elif algo in selectors:
            cat = 'selector'
        else:
            cat = 'extractor_selector'
        data.append({
            'x': row[x_col],
            'y': row[y_col],
            'name': algo,
            'cat': cat
        })
    for idx, row in fusion.iterrows():
        data.append({
            'x': row[x_col],
            'y': row[y_col],
            'name': row['fusion_method'],
            'cat': 'fusion'
        })
    for idx, row in model.iterrows():
        data.append({
            'x': row[x_col],
            'y': row[y_col],
            'name': row['model'],
            'cat': 'model'
        })
    return pd.DataFrame(data)

def plot_category_scatter(ax, all_df, x_label, y_label, title, label_x_map=None, show_legend=False):
    """Create category-based scatter plot."""
    # Plot dots
    for cat in ['extractor', 'selector', 'fusion', 'model']:
        sub = all_df[all_df['cat'] == cat]
        if len(sub) > 0:
            ax.scatter(sub['x'], sub['y'], color=colors[cat], s=60, 
                      edgecolors='black', linewidths=0.8, zorder=5, label=None)
    
    # Add labels
    if label_x_map:
        # For missing percentage plots
        for missing_pct, label_x in label_x_map.items():
            group = all_df[all_df['x'] == missing_pct].sort_values('y', ascending=False).reset_index(drop=True)
            n = len(group)
            if n == 0:
                continue
            gap = 0.015
            for i, row in enumerate(group.itertuples()):
                label_y = group['y'].max() - i * gap
                label_y = max(min(label_y, all_df['y'].max()-gap/2), all_df['y'].min()+gap/2)
                if missing_pct == 50:
                    ha = 'right'
                    relpos = (1, 0.5)
                else:
                    ha = 'left'
                    relpos = (0, 0.5)
                ax.annotate(
                    row.name,
                    xy=(row.x, row.y),
                    xytext=(label_x, label_y), textcoords='data',
                    fontsize=6, ha=ha, va='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=light_bg[row.cat], alpha=0.8),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', 
                                  color=colors[row.cat], alpha=0.7, relpos=relpos),
                    zorder=3
                )
    else:
        # For fit time plots
        for cat in ['extractor', 'selector', 'fusion', 'model']:
            group = all_df[all_df['cat'] == cat].sort_values('y', ascending=False).reset_index(drop=True)
            n = len(group)
            if n == 0:
                continue
            gap = 0.015
            for i, row in enumerate(group.itertuples()):
                label_y = group['y'].max() - i * gap
                label_y = max(min(label_y, all_df['y'].max()-gap/2), all_df['y'].min()+gap/2)
                ha = 'left'
                relpos = (0, 0.5)
                ax.annotate(
                    row.name,
                    xy=(row.x, row.y),
                    xytext=(row.x + 0.005 + (i % 2) * 0.01, label_y), textcoords='data',
                    fontsize=6, ha=ha, va='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=light_bg[row.cat], alpha=0.8),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', 
                                  color=colors[row.cat], alpha=0.7, relpos=relpos),
                    zorder=3
                )
    
    # Add legend if requested
    if show_legend:
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label=labels[k], 
                   markerfacecolor=colors[k], markeredgecolor='black', markersize=8)
            for k in ['extractor', 'selector', 'fusion', 'model']
        ]
        ax.legend(handles=legend_elements, fontsize=8, loc='best')
    
    ax.set_xlabel(x_label, fontsize=10, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)

def plot_top_combinations(ax, df, x_col, y_col, x_label, y_label, title, cmap='viridis'):
    """Create top combinations scatter plot."""
    top_20 = df.head(20).copy()
    
    scatter = ax.scatter(top_20[x_col], top_20[y_col], 
                        c=top_20[y_col], cmap=cmap, s=60, alpha=0.9, 
                        edgecolors='black', linewidth=0.8, zorder=5)
    
    # Add selected labels for clarity (every 3rd point to avoid overcrowding)
    for i, (idx, row) in enumerate(top_20.iterrows()):
        if i % 3 == 0:  # Show every 3rd label
            x = row[x_col]
            y = row[y_col]
            if x_col == 'missing_pct_display':
                if x == 50:
                    xytext = (-3, 3)
                    ha = 'right'
                else:
                    xytext = (3, 3)
                    ha = 'left'
            else:
                xytext = (3, 3)
                ha = 'left'
            
            ax.annotate(
                row['combination'][:20] + ('...' if len(row['combination']) > 20 else ''),
                (x, y),
                xytext=xytext, textcoords='offset points',
                fontsize=5, ha=ha, va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', 
                              color='gray', alpha=0.7),
                zorder=3
            )
    
    ax.set_xlabel(x_label, fontsize=10, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)

def plot_worst_combinations(ax, df, x_col, y_col, x_label, y_label, title, cmap='Reds'):
    """Create worst combinations scatter plot."""
    worst_20 = df.head(20).copy()
    
    scatter = ax.scatter(worst_20[x_col], worst_20[y_col], 
                        c=worst_20[y_col], cmap=cmap, s=60, alpha=0.9, 
                        edgecolors='black', linewidth=0.8, zorder=5)
    
    # Add selected labels for clarity (every 4th point to avoid overcrowding)
    for i, (idx, row) in enumerate(worst_20.iterrows()):
        if i % 4 == 0:  # Show every 4th label
            x = row[x_col]
            y = row[y_col]
            if x == 50:
                xytext = (-3, 3)
                ha = 'right'
            else:
                xytext = (3, 3)
                ha = 'left'
            
            ax.annotate(
                row['combination'][:20] + ('...' if len(row['combination']) > 20 else ''),
                (x, y),
                xytext=xytext, textcoords='offset points',
                fontsize=5, ha=ha, va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', 
                              color='gray', alpha=0.7),
                zorder=3
            )
    
    ax.set_xlabel(x_label, fontsize=10, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)

def create_combined_plots():
    """Create all 10 plots in a 5x2 grid."""
    
    print("Loading data...")
    # Load classification data
    class_es, class_fusion, class_model, class_top50, class_worst50 = load_classification_data()
    # Load regression data  
    reg_es, reg_fusion, reg_model, reg_top50, reg_worst50 = load_regression_data()
    
    if any(x is None for x in [class_es, class_fusion, class_model, class_top50, class_worst50]):
        print("Error: Could not load classification data files")
        return
    
    if any(x is None for x in [reg_es, reg_fusion, reg_model, reg_top50, reg_worst50]):
        print("Error: Could not load regression data files")
        return
    
    print("Creating combined plots...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(5, 2, figsize=(20, 25))
    fig.suptitle('Comprehensive Analysis: Classification and Regression Performance Plots', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Row 1, Col 1: MCC vs Fit Time (All Categories)
    print("Generating plot 1/10: MCC vs Fit Time (All Categories)")
    class_all_df = prepare_category_data(class_es, class_fusion, class_model, 
                                        EXTRACTORS_CLASSIFICATION, SELECTORS_CLASSIFICATION, 
                                        'avg_fit_time', 'avg_test_mcc')
    plot_category_scatter(axes[0,0], class_all_df, 'Average Fit Time', 'Average MCC',
                         'MCC vs Fit Time by Category', show_legend=True)
    
    # Row 1, Col 2: MCC vs Fit Time (Top 20)
    print("Generating plot 2/10: MCC vs Fit Time (Top 20)")
    plot_top_combinations(axes[0,1], class_top50, 'fit_time', 'test_mcc',
                         'Average Fit Time', 'MCC', 'Top 20 Classification: MCC vs Fit Time')
    
    # Row 2, Col 1: MCC vs Missing (All Categories)
    print("Generating plot 3/10: MCC vs Missing (All Categories)")
    class_missing_df = prepare_category_data(class_es, class_fusion, class_model,
                                            EXTRACTORS_CLASSIFICATION, SELECTORS_CLASSIFICATION,
                                            'missing_pct_display', 'avg_test_mcc')
    label_x_map = {0: 5, 20: 25, 50: 40}
    plot_category_scatter(axes[1,0], class_missing_df, 'Missing Data (%)', 'Average MCC',
                         'MCC vs Missing Data by Category', label_x_map)
    
    # Row 2, Col 2: MCC vs Missing (Worst 20)
    print("Generating plot 4/10: MCC vs Missing (Worst 20)")
    plot_worst_combinations(axes[1,1], class_worst50, 'missing_pct_display', 'test_mcc',
                           'Missing Data (%)', 'MCC', 'Worst 20 Classification: MCC vs Missing Data')
    
    # Row 3, Col 1: MCC vs Missing (Top 20)
    print("Generating plot 5/10: MCC vs Missing (Top 20)")
    plot_top_combinations(axes[2,0], class_top50, 'missing_pct_display', 'test_mcc',
                         'Missing Data (%)', 'MCC', 'Top 20 Classification: MCC vs Missing Data')
    
    # Row 3, Col 2: R² vs Fit Time (All Categories)
    print("Generating plot 6/10: R² vs Fit Time (All Categories)")
    reg_all_df = prepare_category_data(reg_es, reg_fusion, reg_model,
                                      EXTRACTORS_REGRESSION, SELECTORS_REGRESSION,
                                      'avg_fit_time', 'avg_test_r2')
    plot_category_scatter(axes[2,1], reg_all_df, 'Average Fit Time', 'Average R²',
                         'R² vs Fit Time by Category', show_legend=True)
    
    # Row 4, Col 1: R² vs Fit Time (Top 20)
    print("Generating plot 7/10: R² vs Fit Time (Top 20)")
    plot_top_combinations(axes[3,0], reg_top50, 'fit_time', 'test_r2',
                         'Average Fit Time', 'R²', 'Top 20 Regression: R² vs Fit Time')
    
    # Row 4, Col 2: R² vs Missing (All Categories)
    print("Generating plot 8/10: R² vs Missing (All Categories)")
    reg_missing_df = prepare_category_data(reg_es, reg_fusion, reg_model,
                                          EXTRACTORS_REGRESSION, SELECTORS_REGRESSION,
                                          'missing_pct_display', 'avg_test_r2')
    plot_category_scatter(axes[3,1], reg_missing_df, 'Missing Data (%)', 'Average R²',
                         'R² vs Missing Data by Category', label_x_map)
    
    # Row 5, Col 1: R² vs Missing (Worst 20)
    print("Generating plot 9/10: R² vs Missing (Worst 20)")
    plot_worst_combinations(axes[4,0], reg_worst50, 'missing_pct_display', 'test_r2',
                           'Missing Data (%)', 'R²', 'Worst 20 Regression: R² vs Missing Data')
    
    # Row 5, Col 2: R² vs Missing (Top 20)
    print("Generating plot 10/10: R² vs Missing (Top 20)")
    plot_top_combinations(axes[4,1], reg_top50, 'missing_pct_display', 'test_r2',
                         'Missing Data (%)', 'R²', 'Top 20 Regression: R² vs Missing Data')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save the combined plot
    output_file = 'results/combined_analysis_plots_grid.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.5)
    print(f"\nCombined plots saved as: {output_file}")
    
    plt.show()
    
    print("\nAll 10 plots generated successfully!")

if __name__ == "__main__":
    create_combined_plots() 