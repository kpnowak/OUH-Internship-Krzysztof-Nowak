import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Define paths for AML and Sarcoma regression results
base_dirs = [
    'results/regression/AML/',
    'results/regression/Sarcoma/'
]
missing_pcts = [0, 20, 50]

all_r2 = {pct: [] for pct in missing_pcts}

# Collect all R2 values for each missing percentage
for base_dir in base_dirs:
    for pct in missing_pcts:
        pattern = os.path.join(base_dir, f'all_runs_ranked_{pct}pct_missing.csv')
        files = glob.glob(pattern)
        for file in files:
            df = pd.read_csv(file)
            # Only consider R2 >= -1
            r2s = df['test_r2'][df['test_r2'] >= -1]
            all_r2[pct].extend(r2s.tolist())

# Calculate mean and std for each missing percentage
means = []
stds = []
for pct in missing_pcts:
    vals = all_r2[pct]
    means.append(pd.Series(vals).mean())
    stds.append(pd.Series(vals).std())

# Bar plot
plt.figure(figsize=(8, 6))
plt.bar([str(p) + '%' for p in missing_pcts], means, yerr=stds, capsize=8, color=['royalblue', 'gold', 'firebrick'], alpha=0.8)
plt.ylabel('Average R²', fontsize=13, fontweight='bold')
plt.xlabel('Missing Data Percentage (%)', fontsize=13, fontweight='bold')
plt.title('Average R² vs Missing Data Percentage (All Regression Results)', fontsize=15, fontweight='bold', pad=20)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout(pad=2.0)
plt.savefig('results/regression/combined/r2_vs_missing_barplot.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
plt.show() 