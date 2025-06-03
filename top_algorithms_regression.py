import pandas as pd
import numpy as np
from pathlib import Path

# ------------------------------------------------------------
# CONFIGURATION – tweak if desired
# ------------------------------------------------------------
CSV_PATH = "output_main_without_mrmr/Sarcoma/metrics/Sarcoma_combined_cv_metrics.csv"   # uploaded file
TIME_CAP  = 300      # soft cap  (s) – affects ordering, not exclusion
TIME_HARD = 1200     # hard cap  (s) – rows above are discarded
EPS       = 1e-8     # to avoid div-by-zero

TOP_N_ALGOS   = 10
TOP_N_FEAT    = 5
TOP_N_INTEGR  = 5
TOP_N_MODELS  = 8

# NEW: Weighted scoring configuration
USE_WEIGHTED_SCORING = True  # Set to True to use weighted scoring instead of lexicographic

# Separate weight configurations for different ranking types
BASELINE_METRIC_WEIGHTS = {
    "baseline_rmse": 0.4,    # 40% - Lower is better
    "baseline_r2": 0.15,      # 15% - Higher is better  
    "baseline_mae": 0.1,     # 10% - Lower is better
    "robustness": 0.25,       # 25% - Higher is better
    "train_time_avg": 0.1     # 10% - Lower is better
}

# Weights for missing percentage rankings (using actual row metrics)
ACTUAL_METRIC_WEIGHTS = {
    "rmse": 0.4,             # 40% - Lower is better
    "r2": 0.15,              # 15% - Higher is better  
    "mae": 0.1,              # 10% - Lower is better
    "robustness": 0.25,      # 25% - Higher is better
    "train_time": 0.1        # 10% - Lower is better
}

SAVE_DIR = Path("final_results/Sarcoma")  # where result CSVs will be written
# ------------------------------------------------------------

df = pd.read_csv(CSV_PATH)

# Validate weights if using weighted scoring
if USE_WEIGHTED_SCORING:
    # Validate baseline metric weights
    total_weight = sum(BASELINE_METRIC_WEIGHTS.values())
    if abs(total_weight - 1.0) > 1e-6:
        print(f"Warning: Baseline metric weights sum to {total_weight:.4f}, not 1.0")
        print("Normalizing baseline weights...")
        BASELINE_METRIC_WEIGHTS = {k: v/total_weight for k, v in BASELINE_METRIC_WEIGHTS.items()}
    
    # Validate actual metric weights
    total_weight_actual = sum(ACTUAL_METRIC_WEIGHTS.values())
    if abs(total_weight_actual - 1.0) > 1e-6:
        print(f"Warning: Actual metric weights sum to {total_weight_actual:.4f}, not 1.0")
        print("Normalizing actual weights...")
        ACTUAL_METRIC_WEIGHTS = {k: v/total_weight_actual for k, v in ACTUAL_METRIC_WEIGHTS.items()}
    
    print("=== USING WEIGHTED SCORING ===")
    print("Baseline metric weights (for top summaries):")
    for metric, weight in BASELINE_METRIC_WEIGHTS.items():
        print(f"  {metric}: {weight:.1%}")
    print("Actual metric weights (for missing % rankings):")
    for metric, weight in ACTUAL_METRIC_WEIGHTS.items():
        print(f"  {metric}: {weight:.1%}")
    print()
else:
    print("=== USING LEXICOGRAPHIC RANKING ===")
    print("Priority order: baseline_rmse → baseline_r2 → baseline_mae → robustness → train_time_avg")
    print()

# Create SAVE_DIR if it doesn't exist
SAVE_DIR.mkdir(exist_ok=True)

# Helper function to calculate robustness for a given configuration
def calculate_robustness(df, dataset, workflow, algorithm, integration_tech=None, model=None):
    """Calculate robustness for a specific configuration"""
    base_mask = (
        (df.Dataset == dataset) &
        (df.Workflow == workflow) &
        (df.Algorithm == algorithm) &
        (df.Missing_Percentage == 0)
    )
    miss_mask = (
        (df.Dataset == dataset) &
        (df.Workflow == workflow) &
        (df.Algorithm == algorithm) &
        (df.Missing_Percentage > 0)
    )
    
    # Add additional filters if provided
    if integration_tech is not None:
        base_mask = base_mask & (df.integration_tech == integration_tech)
        miss_mask = miss_mask & (df.integration_tech == integration_tech)
    
    if model is not None:
        base_mask = base_mask & (df.Model == model)
        miss_mask = miss_mask & (df.Model == model)

    # Calculate baseline metrics (0% missing)
    R0 = df.loc[base_mask, "rmse"].mean()
    MAE0 = df.loc[base_mask, "mae"].mean()
    R2_0 = df.loc[base_mask, "r2"].mean()
    
    # Calculate missing data metrics
    Rmis = df.loc[miss_mask, "rmse"].mean()
    MAEmis = df.loc[miss_mask, "mae"].mean()
    R2mis = df.loc[miss_mask, "r2"].mean()
    
    # Handle NaN values (no missing data available)
    if np.isnan(Rmis):
        Rmis = R0
    if np.isnan(MAEmis):
        MAEmis = MAE0
    if np.isnan(R2mis):
        R2mis = R2_0

    # Calculate robustness for each metric
    rmse_robustness = 1 - max((Rmis - R0) / (R0 + EPS), 0)
    mae_robustness = 1 - max((MAEmis - MAE0) / (MAE0 + EPS), 0)
    
    # Fixed R² robustness calculation - consistent handling regardless of sign
    # For R², we want to measure how much the score drops due to missing data
    # Use absolute difference normalized by the range of possible degradation
    r2_degradation = R2_0 - R2mis  # Positive when performance degrades
    
    # Normalize by a reasonable scale - use max of absolute baseline value or 1.0
    # This ensures we don't get extreme values when baseline R² is close to 0
    r2_scale = max(abs(R2_0), 1.0)
    r2_robustness = 1 - max(r2_degradation / r2_scale, 0)
    
    # Composite robustness: average of all three metrics
    robustness = (rmse_robustness + mae_robustness + r2_robustness) / 3.0
    
    return robustness

# Helper function to apply scoring (weighted or lexicographic)
def apply_scoring(data, use_weighted=USE_WEIGHTED_SCORING):
    """Apply scoring to grouped data"""
    if use_weighted:
        data_norm = data.copy()
        
        # Normalize metrics
        for col in ["baseline_rmse", "baseline_mae", "train_time_avg"]:  # Lower is better
            min_val, max_val = data[col].min(), data[col].max()
            if max_val > min_val:
                data_norm[f"{col}_norm"] = 1 - (data[col] - min_val) / (max_val - min_val)
            else:
                data_norm[f"{col}_norm"] = 1.0
        
        for col in ["baseline_r2", "robustness"]:  # Higher is better
            min_val, max_val = data[col].min(), data[col].max()
            if max_val > min_val:
                data_norm[f"{col}_norm"] = (data[col] - min_val) / (max_val - min_val)
            else:
                data_norm[f"{col}_norm"] = 1.0
        
        # Calculate weighted composite score
        data_norm["composite_score"] = (
            BASELINE_METRIC_WEIGHTS["baseline_rmse"] * data_norm["baseline_rmse_norm"] +
            BASELINE_METRIC_WEIGHTS["baseline_r2"] * data_norm["baseline_r2_norm"] +
            BASELINE_METRIC_WEIGHTS["baseline_mae"] * data_norm["baseline_mae_norm"] +
            BASELINE_METRIC_WEIGHTS["robustness"] * data_norm["robustness_norm"] +
            BASELINE_METRIC_WEIGHTS["train_time_avg"] * data_norm["train_time_avg_norm"]
        )
        
        return data_norm.sort_values("composite_score", ascending=False)
    else:
        return data.sort_values(
            by=["baseline_rmse", "baseline_r2", "baseline_mae", "robustness", "train_time_avg"],
            ascending=[True, False, True, False, True])

# ============================================================================
# 1. TOP ALGORITHMS - Average baseline metrics from 0% missing data only
# ============================================================================
print("=== CALCULATING TOP ALGORITHMS ===")

# Get only 0% missing data
df_0pct = df[df.Missing_Percentage == 0].copy()

# Group by Algorithm and Workflow to capture which workflow each algorithm belongs to
algo_groups = []
for (algorithm, workflow), algo_data in df_0pct.groupby(['Algorithm', 'Workflow']):
    
    # Calculate average baseline metrics (from 0% missing data)
    baseline_rmse = algo_data['rmse'].mean()
    baseline_mae = algo_data['mae'].mean()
    baseline_r2 = algo_data['r2'].mean()
    train_time_avg = algo_data['train_time'].mean()
    
    # Skip if training time exceeds hard limit
    if train_time_avg > TIME_HARD:
        continue
    
    # Calculate average robustness across all configurations for this algorithm-workflow combination
    robustness_values = []
    for _, row in algo_data.iterrows():
        # Fixed: Pass all relevant parameters for consistent robustness calculation
        rob = calculate_robustness(df, row.Dataset, row.Workflow, row.Algorithm,
                                 integration_tech=row.integration_tech, model=row.Model)
        robustness_values.append(rob)
    
    avg_robustness = np.mean(robustness_values) if robustness_values else 0.0
    
    algo_groups.append({
        'Algorithm': algorithm,
        'Workflow': workflow,
        'baseline_rmse': baseline_rmse,
        'baseline_mae': baseline_mae,
        'baseline_r2': baseline_r2,
        'robustness': avg_robustness,
        'train_time_avg': train_time_avg
    })

algo_grouped = pd.DataFrame(algo_groups)
algo_best = apply_scoring(algo_grouped).head(TOP_N_ALGOS)

# ============================================================================
# 2. TOP FEATURES - Average baseline metrics from 0% missing data only
# ============================================================================
print("=== CALCULATING TOP FEATURES ===")

# Group by (n_features, n_components) and calculate average baseline metrics
feat_groups = []
for (n_features, n_components), group in df_0pct.groupby(['n_features', 'n_components']):
    # Calculate average baseline metrics (from 0% missing data)
    baseline_rmse = group['rmse'].mean()
    baseline_mae = group['mae'].mean()
    baseline_r2 = group['r2'].mean()
    train_time_avg = group['train_time'].mean()
    
    # Skip if training time exceeds hard limit
    if train_time_avg > TIME_HARD:
        continue
    
    # Calculate average robustness across all configurations for this feature setting
    robustness_values = []
    for _, row in group.iterrows():
        # Fixed: Pass all relevant parameters for consistent robustness calculation
        rob = calculate_robustness(df, row.Dataset, row.Workflow, row.Algorithm, 
                                 integration_tech=row.integration_tech, model=row.Model)
        robustness_values.append(rob)
    
    avg_robustness = np.mean(robustness_values) if robustness_values else 0.0
    
    feat_groups.append({
        'n_features': n_features,
        'n_components': n_components,
        'algorithms_count': len(group['Algorithm'].unique()),  # Show how many algorithms contributed
        'baseline_rmse': baseline_rmse,
        'baseline_mae': baseline_mae,
        'baseline_r2': baseline_r2,
        'robustness': avg_robustness,
        'train_time_avg': train_time_avg
    })

feat_grouped = pd.DataFrame(feat_groups)
feat_best = apply_scoring(feat_grouped).head(TOP_N_FEAT)

# ============================================================================
# 3. TOP INTEGRATION TECHNIQUES - Average baseline metrics from 0% missing data only
# ============================================================================
print("=== CALCULATING TOP INTEGRATION TECHNIQUES ===")

# Group by integration_tech and calculate average baseline metrics
integr_groups = []
for integration_tech in df_0pct['integration_tech'].unique():
    integr_data = df_0pct[df_0pct['integration_tech'] == integration_tech]
    
    # Calculate average baseline metrics (from 0% missing data)
    baseline_rmse = integr_data['rmse'].mean()
    baseline_mae = integr_data['mae'].mean()
    baseline_r2 = integr_data['r2'].mean()
    train_time_avg = integr_data['train_time'].mean()
    
    # Skip if training time exceeds hard limit
    if train_time_avg > TIME_HARD:
        continue
    
    # Calculate average robustness across all configurations for this integration technique
    robustness_values = []
    for _, row in integr_data.iterrows():
        # Fixed: Pass all relevant parameters for consistent robustness calculation
        rob = calculate_robustness(df, row.Dataset, row.Workflow, row.Algorithm, 
                                 integration_tech=row.integration_tech, model=row.Model)
        robustness_values.append(rob)
    
    avg_robustness = np.mean(robustness_values) if robustness_values else 0.0
    
    integr_groups.append({
        'integration_tech': integration_tech,
        'algorithms_count': len(integr_data['Algorithm'].unique()),  # Show how many algorithms contributed
        'baseline_rmse': baseline_rmse,
        'baseline_mae': baseline_mae,
        'baseline_r2': baseline_r2,
        'robustness': avg_robustness,
        'train_time_avg': train_time_avg
    })

integr_grouped = pd.DataFrame(integr_groups)
integr_best = apply_scoring(integr_grouped).head(TOP_N_INTEGR)

# ============================================================================
# 4. TOP MODELS - Average baseline metrics from 0% missing data only
# ============================================================================
print("=== CALCULATING TOP MODELS ===")

# Group by Model and calculate average baseline metrics
model_groups = []
for model in df_0pct['Model'].unique():
    model_data = df_0pct[df_0pct['Model'] == model]
    
    # Calculate average baseline metrics (from 0% missing data)
    baseline_rmse = model_data['rmse'].mean()
    baseline_mae = model_data['mae'].mean()
    baseline_r2 = model_data['r2'].mean()
    train_time_avg = model_data['train_time'].mean()
    
    # Skip if training time exceeds hard limit
    if train_time_avg > TIME_HARD:
        continue
    
    # Calculate average robustness across all configurations for this model
    robustness_values = []
    for _, row in model_data.iterrows():
        # Fixed: Pass all relevant parameters for consistent robustness calculation
        rob = calculate_robustness(df, row.Dataset, row.Workflow, row.Algorithm, 
                                 integration_tech=row.integration_tech, model=row.Model)
        robustness_values.append(rob)
    
    avg_robustness = np.mean(robustness_values) if robustness_values else 0.0
    
    model_groups.append({
        'Model': model,
        'algorithms_count': len(model_data['Algorithm'].unique()),  # Show how many algorithms contributed
        'baseline_rmse': baseline_rmse,
        'baseline_mae': baseline_mae,
        'baseline_r2': baseline_r2,
        'robustness': avg_robustness,
        'train_time_avg': train_time_avg
    })

model_grouped = pd.DataFrame(model_groups)
model_best = apply_scoring(model_grouped).head(TOP_N_MODELS)

# ============================================================================
# 5. CALCULATE RANKINGS FOR EACH MISSING PERCENTAGE
# ============================================================================
print("\n=== CREATING MISSING PERCENTAGE RANKINGS ===")

missing_percentages = [0.0, 0.2, 0.5]  # 0%, 20%, 50%
ranking_paths = []

for missing_pct in missing_percentages:
    print(f"Processing {missing_pct*100:.0f}% missing data...")
    
    # Filter data for this missing percentage
    df_missing = df[df.Missing_Percentage == missing_pct].copy()
    
    if len(df_missing) == 0:
        print(f"  No data found for {missing_pct*100:.0f}% missing")
        continue
    
    # For each row, calculate robustness and add baseline metrics
    # Note: Also applies TIME_HARD filtering for consistency with top summaries
    enriched_rows = []
    for _, row in df_missing.iterrows():
        row_copy = row.copy()
        
        # Calculate robustness for this specific configuration
        # Fixed: Pass all relevant parameters for consistent robustness calculation
        robustness = calculate_robustness(df, row.Dataset, row.Workflow, row.Algorithm,
                                        integration_tech=row.integration_tech, model=row.Model)
        
        # Get baseline metrics (0% missing) for this configuration
        base_mask = (
            (df.Dataset == row.Dataset) &
            (df.Workflow == row.Workflow) &
            (df.Algorithm == row.Algorithm) &
            (df.Missing_Percentage == 0)
        )
        
        baseline_rmse = df.loc[base_mask, "rmse"].mean()
        baseline_mae = df.loc[base_mask, "mae"].mean()
        baseline_r2 = df.loc[base_mask, "r2"].mean()
        train_time_avg = df.loc[base_mask, "train_time"].mean()
        
        # Skip if training time exceeds hard limit (consistent with other sections)
        if train_time_avg > TIME_HARD:
            continue
        
        # Add enriched metrics to row
        row_copy["baseline_rmse"] = baseline_rmse
        row_copy["baseline_mae"] = baseline_mae
        row_copy["baseline_r2"] = baseline_r2
        row_copy["robustness"] = robustness
        row_copy["train_time_avg"] = train_time_avg
        
        enriched_rows.append(row_copy)
    
    df_missing_enriched = pd.DataFrame(enriched_rows)
    
    # Apply ranking using ACTUAL ROW METRICS (not baseline)
    if USE_WEIGHTED_SCORING:
        df_missing_norm = df_missing_enriched.copy()
        
        # Normalize using ACTUAL row metrics
        for col in ["rmse", "mae", "train_time"]:  # Lower is better
            min_val, max_val = df_missing_enriched[col].min(), df_missing_enriched[col].max()
            if max_val > min_val:
                df_missing_norm[f"{col}_norm"] = 1 - (df_missing_enriched[col] - min_val) / (max_val - min_val)
            else:
                df_missing_norm[f"{col}_norm"] = 1.0
        
        for col in ["r2", "robustness"]:  # Higher is better
            min_val, max_val = df_missing_enriched[col].min(), df_missing_enriched[col].max()
            if max_val > min_val:
                df_missing_norm[f"{col}_norm"] = (df_missing_enriched[col] - min_val) / (max_val - min_val)
            else:
                df_missing_norm[f"{col}_norm"] = 1.0
        
        # Calculate weighted composite score using ACTUAL row metrics
        df_missing_norm["composite_score"] = (
            ACTUAL_METRIC_WEIGHTS["rmse"] * df_missing_norm["rmse_norm"] +
            ACTUAL_METRIC_WEIGHTS["r2"] * df_missing_norm["r2_norm"] +
            ACTUAL_METRIC_WEIGHTS["mae"] * df_missing_norm["mae_norm"] +
            ACTUAL_METRIC_WEIGHTS["robustness"] * df_missing_norm["robustness_norm"] +
            ACTUAL_METRIC_WEIGHTS["train_time"] * df_missing_norm["train_time_norm"]
        )
        
        # Add rank column
        df_missing_norm["rank"] = df_missing_norm["composite_score"].rank(ascending=False, method="min").astype(int)
        df_missing_ranked = df_missing_norm.sort_values("composite_score", ascending=False)
    else:
        # Lexicographic ranking using ACTUAL row metrics
        df_missing_ranked = df_missing_enriched.sort_values(
            by=["rmse", "r2", "mae", "robustness", "train_time"],
            ascending=[True, False, True, False, True])
        df_missing_ranked["rank"] = range(1, len(df_missing_ranked) + 1)
    
    # Save to CSV
    missing_pct_str = f"{missing_pct*100:.0f}pct"
    ranking_path = SAVE_DIR / f"all_runs_ranked_{missing_pct_str}_missing.csv"
    df_missing_ranked.to_csv(ranking_path, index=False)
    ranking_paths.append(ranking_path)
    
    print(f"  Saved {len(df_missing_ranked)} ranked runs to: {ranking_path}")

# ============================================================================
# 6. SAVE TOP RESULTS AND DISPLAY
# ============================================================================

# How many times does the BEST algorithm appear in other leaderboards?
top_algo_name = algo_best.iloc[0]["Algorithm"]
# Note: Features, Integration, and Models rankings now show averages across all algorithms
# so we can't count algorithm appearances in those rankings anymore
appear_count = 0  # Since other rankings are now algorithm-agnostic

# Save CSVs
algo_path   = SAVE_DIR / "top_algorithms.csv"
feat_path   = SAVE_DIR / "top_feature_settings.csv"
integr_path = SAVE_DIR / "top_integration_tech.csv"
model_path  = SAVE_DIR / "top_models.csv"

algo_best  .to_csv(algo_path,   index=False)
feat_best  .to_csv(feat_path,   index=False)
integr_best.to_csv(integr_path, index=False)
model_best .to_csv(model_path,  index=False)

# Display results
print("\n=== TOP ALGORITHMS ===")
if USE_WEIGHTED_SCORING:
    print(algo_best[["Algorithm","Workflow","composite_score","baseline_rmse","baseline_r2","baseline_mae","robustness","train_time_avg"]])
else:
    print(algo_best[["Algorithm","Workflow","baseline_rmse","baseline_r2","baseline_mae","robustness","train_time_avg"]])

print("\n=== TOP FEATURE / COMPONENT SETTINGS ===")
if USE_WEIGHTED_SCORING:
    print(feat_best[["n_features","n_components","algorithms_count","composite_score",
                     "baseline_rmse","baseline_r2","baseline_mae","robustness","train_time_avg"]])
else:
    print(feat_best[["n_features","n_components","algorithms_count",
                     "baseline_rmse","baseline_r2","baseline_mae","robustness","train_time_avg"]])

print("\n=== TOP INTEGRATION TECHNIQUES ===")
if USE_WEIGHTED_SCORING:
    print(integr_best[["integration_tech","algorithms_count","composite_score",
                       "baseline_rmse","baseline_r2","baseline_mae","robustness","train_time_avg"]])
else:
    print(integr_best[["integration_tech","algorithms_count",
                       "baseline_rmse","baseline_r2","baseline_mae","robustness","train_time_avg"]])

print("\n=== TOP MODELS ===")
if USE_WEIGHTED_SCORING:
    print(model_best[["Model","algorithms_count","composite_score","baseline_rmse","baseline_r2","baseline_mae",
                      "robustness","train_time_avg"]])
else:
    print(model_best[["Model","algorithms_count","baseline_rmse","baseline_r2","baseline_mae",
                      "robustness","train_time_avg"]])

print(f"\nNote: Features, Integration Techniques, and Models rankings show averages across all algorithms.")
print(f"The best algorithm '{top_algo_name}' is ranked #1 in the Algorithm-specific ranking.")

print("\nCSV files written to:")
print(f"  Top summaries: {algo_path}, {feat_path}, {integr_path}, {model_path}")
if ranking_paths:
    print(f"  Missing % rankings: {', '.join(str(p) for p in ranking_paths)}")

print(f"\nCreated {len(ranking_paths)} additional ranking files.")