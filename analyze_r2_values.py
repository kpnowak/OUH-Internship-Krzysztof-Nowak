import pandas as pd
import numpy as np

# Load data
extraction_df = pd.read_csv('output_main_without_mrmr/AML/metrics/AML_extraction_cv_metrics.csv')
selection_df = pd.read_csv('output_main_without_mrmr/AML/metrics/AML_selection_cv_metrics.csv')
combined_df = pd.concat([extraction_df, selection_df], ignore_index=True)

print("="*80)
print("UNDERSTANDING EXTREME NEGATIVE R² VALUES")
print("="*80)

# Basic R² statistics
print(f"R² Statistics:")
print(f"  Total experiments: {len(combined_df)}")
print(f"  Mean R²: {combined_df['r2'].mean():.6f}")
print(f"  Median R²: {combined_df['r2'].median():.6f}")
print(f"  Min R²: {combined_df['r2'].min():.2f}")
print(f"  Max R²: {combined_df['r2'].max():.6f}")
print(f"  Std R²: {combined_df['r2'].std():.2f}")

# Check extreme cases
extreme_negative = combined_df[combined_df['r2'] < -100]
very_extreme = combined_df[combined_df['r2'] < -1000]
super_extreme = combined_df[combined_df['r2'] < -100000]

print(f"\nExtreme R² Cases:")
print(f"  R² < -100: {len(extreme_negative)} cases ({len(extreme_negative)/len(combined_df)*100:.1f}%)")
print(f"  R² < -1000: {len(very_extreme)} cases ({len(very_extreme)/len(combined_df)*100:.1f}%)")
print(f"  R² < -100,000: {len(super_extreme)} cases ({len(super_extreme)/len(combined_df)*100:.1f}%)")

# Show the worst cases
print(f"\n5 Worst R² Values:")
worst_cases = combined_df.nsmallest(5, 'r2')[['Workflow', 'Algorithm', 'Model', 'r2', 'mse', 'Missing_Percentage']]
for idx, row in worst_cases.iterrows():
    print(f"  R²={row['r2']:.1f} | {row['Workflow']} | {row['Algorithm']} | {row['Model']} | MSE={row['mse']:.1f}")

# Analyze MSE vs R² relationship 
print(f"\nMSE Statistics:")
print(f"  Min MSE: {combined_df['mse'].min():.2f}")
print(f"  Max MSE: {combined_df['mse'].max():.2f}")
print(f"  Mean MSE: {combined_df['mse'].mean():.2f}")

# Check for any positive R²
positive_r2 = combined_df[combined_df['r2'] > 0]
near_zero_r2 = combined_df[(combined_df['r2'] >= -0.1) & (combined_df['r2'] <= 0)]

print(f"\nR² Distribution:")
print(f"  Positive R²: {len(positive_r2)} cases ({len(positive_r2)/len(combined_df)*100:.1f}%)")
print(f"  Near zero (-0.1 to 0): {len(near_zero_r2)} cases ({len(near_zero_r2)/len(combined_df)*100:.1f}%)")

print(f"\n" + "="*80)
print("EXPLANATION OF EXTREME NEGATIVE R² VALUES")
print("="*80)
print("""
R² (coefficient of determination) measures how well predictions fit the actual data:

• R² = 1.0    → Perfect predictions (all variance explained)
• R² = 0.0    → Model performs as well as predicting the mean
• R² < 0.0    → Model performs WORSE than predicting the mean

EXTREME NEGATIVE VALUES like -3,077,686 mean:
1. The model's predictions are catastrophically bad
2. The Mean Squared Error (MSE) is orders of magnitude larger than the variance of the target
3. The model is essentially making random or completely wrong predictions

POSSIBLE CAUSES:
• Data preprocessing issues (wrong scaling, NaN handling)
• Feature extraction creating meaningless features  
• Model hyperparameters completely unsuitable for the data
• Overfitting on training set, terrible generalization
• Bug in the pipeline or evaluation code
• Target variable distribution issues

MATHEMATICAL INTERPRETATION:
R² = 1 - (MSE / variance_of_targets)
When MSE >> variance_of_targets, R² becomes very negative.

For AML dataset, this suggests the current pipeline is fundamentally 
unsuitable for this regression task and needs major debugging.
""") 