import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Define extractor and selector lists
EXTRACTORS = ['PCA', 'KPCA', 'FA', 'PLS', 'KPLS', 'SparsePLS']
SELECTORS = ['ElasticNetFS', 'RFImportance', 'VarianceFTest', 'LASSO', 'f_regressionFS']

# Load data
extractor_selector = pd.read_csv('results/regression/combined/extractor_selector_rankings.csv')
fusion = pd.read_csv('results/regression/combined/fusion_technique_rankings.csv')
model = pd.read_csv('results/regression/combined/model_rankings.csv')

plt.figure(figsize=(18, 10))

# Assign category and color
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

data = []
for idx, row in extractor_selector.iterrows():
    algo = row['algorithm']
    if algo in EXTRACTORS:
        cat = 'extractor'
    elif algo in SELECTORS:
        cat = 'selector'
    else:
        cat = 'extractor_selector'  # fallback, shouldn't happen
    data.append({
        'x': row['avg_fit_time'],
        'y': row['avg_test_r2'],
        'name': algo,
        'cat': cat
    })
for idx, row in fusion.iterrows():
    data.append({
        'x': row['avg_fit_time'],
        'y': row['avg_test_r2'],
        'name': row['fusion_method'],
        'cat': 'fusion'
    })
for idx, row in model.iterrows():
    data.append({
        'x': row['avg_fit_time'],
        'y': row['avg_test_r2'],
        'name': row['model'],
        'cat': 'model'
    })
all_df = pd.DataFrame(data)

# Plot dots at true positions
for cat in ['extractor', 'selector', 'fusion', 'model']:
    sub = all_df[all_df['cat'] == cat]
    plt.scatter(sub['x'], sub['y'], color=colors[cat], s=120, edgecolors='black', linewidths=1.5, zorder=5, label=None)

# For each category, stack labels vertically with overlap avoidance
for cat in ['extractor', 'selector', 'fusion', 'model']:
    group = all_df[all_df['cat'] == cat].sort_values('y', ascending=False).reset_index(drop=True)
    n = len(group)
    if n == 0:
        continue
    gap = 0.025
    for i, row in enumerate(group.itertuples()):
        label_y = group['y'].max() - i * gap
        label_y = max(min(label_y, all_df['y'].max()-gap/2), all_df['y'].min()+gap/2)
        ha = 'left'
        relpos = (0, 0.5)
        plt.annotate(
            row.name,
            xy=(row.x, row.y),
            xytext=(row.x + 0.01, label_y), textcoords='data',
            fontsize=9, ha=ha, va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=light_bg[row.cat], alpha=0.9),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color=colors[row.cat], alpha=0.7, relpos=relpos),
            zorder=3
        )

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=labels[k], markerfacecolor=colors[k], markeredgecolor='black', markersize=12)
    for k in ['extractor', 'selector', 'fusion', 'model']
]
plt.legend(handles=legend_elements, fontsize=12, loc='lower left', title='')

plt.xlabel('Average Fit Time', fontsize=13, fontweight='bold')
plt.ylabel('Average R² Score', fontsize=13, fontweight='bold')
plt.title('Average R² vs Average Fit Time by Category', fontsize=15, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3)
plt.tight_layout(pad=2.0)
plt.savefig('results/regression/combined/r2_vs_fit_time_all_categories.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
plt.show() 