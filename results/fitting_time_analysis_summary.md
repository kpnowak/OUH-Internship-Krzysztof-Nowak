# Fitting Time Range Analysis Summary

## Overview
This analysis examines the fitting time ranges for different combinations of extractors, selectors, fusion techniques, and models across 10,206 experiments from 9 datasets (2 regression, 7 classification) with varying missing data percentages (0%, 20%, 50%).

## Key Findings

### Overall Statistics
- **Total Experiments**: 10,206
- **Fit Time Range**: 0.0008s - 42.5931s
- **Mean Fit Time**: 0.1477s
- **Median Fit Time**: 0.0628s
- **Standard Deviation**: 0.4586s

### By Algorithm Type

#### Extractors (2,916 experiments)
- **Range**: 0.0008s - 42.5931s
- **Mean**: 0.1054s
- **Median**: 0.0422s
- **Fastest**: LDA (0.0668s mean)
- **Slowest**: SparsePLS (0.2069s mean)

#### Selectors (7,290 experiments)
- **Range**: 0.0009s - 1.3610s
- **Mean**: 0.1647s
- **Median**: 0.0775s
- **Fastest**: ElasticNetFS (0.1034s mean)
- **Slowest**: VarianceFTest (0.1912s mean)

### By Fusion Method

| Fusion Method | Count | Range | Mean | Median |
|---------------|-------|-------|------|--------|
| **average** | 1,701 | 0.0008s - 0.7720s | 0.1244s | 0.0443s |
| **attention_weighted** | 567 | 0.0012s - 2.0606s | 0.1535s | 0.0524s |
| **mkl** | 1,701 | 0.0008s - 1.2993s | 0.1615s | 0.0831s |
| **learnable_weighted** | 567 | 0.0014s - 1.8718s | 0.1535s | 0.0532s |
| **standard_concat** | 567 | 0.0010s - 1.8238s | 0.1708s | 0.0870s |
| **early_fusion_pca** | 1,701 | 0.0008s - 0.7671s | 0.1345s | 0.0610s |
| **sum** | 1,701 | 0.0008s - 42.5931s | 0.1775s | 0.0805s |
| **max** | 1,701 | 0.0008s - 0.7864s | 0.1293s | 0.0496s |

### By Model

#### Regression Models
| Model | Count | Range | Mean | Median |
|-------|-------|-------|------|--------|
| **LinearRegression** | 756 | 0.0035s - 0.8954s | 0.1266s | 0.0865s |
| **ElasticNet** | 756 | 0.0319s - 0.2311s | 0.0501s | 0.0397s |
| **RandomForestRegressor** | 756 | 0.1050s - 0.6733s | 0.2552s | 0.2862s |

#### Classification Models
| Model | Count | Range | Mean | Median |
|-------|-------|-------|------|--------|
| **LogisticRegression** | 2,646 | 0.0008s - 2.0606s | 0.0128s | 0.0064s |
| **RandomForestClassifier** | 2,646 | 0.0779s - 42.5931s | 0.3493s | 0.2497s |
| **SVC** | 2,646 | 0.0026s - 0.8852s | 0.0844s | 0.0464s |

### By Task Type

| Task Type | Count | Range | Mean | Median |
|-----------|-------|-------|------|--------|
| **Regression** | 2,268 | 0.0035s - 0.8954s | 0.1440s | 0.0903s |
| **Classification** | 7,938 | 0.0008s - 42.5931s | 0.1488s | 0.0482s |

### By Missing Data Percentage

| Missing % | Count | Range | Mean | Median |
|-----------|-------|-------|------|--------|
| **0%** | 4,536 | 0.0008s - 2.0606s | 0.1473s | 0.0619s |
| **20%** | 2,835 | 0.0008s - 1.2219s | 0.1420s | 0.0639s |
| **50%** | 2,835 | 0.0008s - 42.5931s | 0.1542s | 0.0628s |

## Individual Algorithm Performance

### Extractors (Ranked by Mean Fit Time)

1. **LDA** (378 experiments)
   - Range: 0.0008s - 0.2529s
   - Mean: 0.0668s
   - Median: 0.0199s

2. **KPCA** (486 experiments)
   - Range: 0.0013s - 0.5211s
   - Mean: 0.0740s
   - Median: 0.0344s

3. **PLS-DA** (378 experiments)
   - Range: 0.0008s - 0.2385s
   - Mean: 0.0745s
   - Median: 0.0377s

4. **PCA** (486 experiments)
   - Range: 0.0010s - 0.3667s
   - Mean: 0.0821s
   - Median: 0.0464s

5. **FA** (486 experiments)
   - Range: 0.0008s - 0.3612s
   - Mean: 0.0959s
   - Median: 0.0505s

6. **KPLS** (108 experiments)
   - Range: 0.0100s - 0.3442s
   - Mean: 0.1266s
   - Median: 0.0481s

7. **PLS** (108 experiments)
   - Range: 0.0035s - 0.5008s
   - Mean: 0.1597s
   - Median: 0.0417s

8. **SparsePLS** (486 experiments)
   - Range: 0.0010s - 42.5931s
   - Mean: 0.2069s
   - Median: 0.0655s

### Selectors (Ranked by Mean Fit Time)

1. **ElasticNetFS** (1,458 experiments)
   - Range: 0.0024s - 0.8577s
   - Mean: 0.1034s
   - Median: 0.0786s

2. **RFImportance** (1,458 experiments)
   - Range: 0.0009s - 1.0876s
   - Mean: 0.1591s
   - Median: 0.0778s

3. **VarianceFTest** (1,458 experiments)
   - Range: 0.0010s - 1.3610s
   - Mean: 0.1912s
   - Median: 0.0765s

4. **LASSO** (1,458 experiments)
   - Range: 0.0020s - 0.8475s
   - Mean: 0.1831s
   - Median: 0.0542s

5. **LogisticL1** (1,134 experiments)
   - Range: 0.0020s - 0.8482s
   - Mean: 0.1914s
   - Median: 0.0542s

## Extreme Cases

### Fastest Combination
- **Fit Time**: 0.0008s
- **Dataset**: Ovarian
- **Algorithm**: LDA
- **Fusion**: average
- **Model**: LogisticRegression
- **Task**: classification

### Slowest Combination
- **Fit Time**: 42.5931s
- **Dataset**: Melanoma
- **Algorithm**: SparsePLS
- **Fusion**: sum
- **Model**: RandomForestClassifier
- **Task**: classification

## Key Insights

1. **Algorithm Type Impact**: Extractors are generally faster (mean: 0.1054s) than selectors (mean: 0.1647s)

2. **Model Performance**: 
   - LogisticRegression is the fastest model (mean: 0.0128s)
   - RandomForestClassifier is the slowest (mean: 0.3493s)
   - ElasticNet is the fastest regression model (mean: 0.0501s)

3. **Fusion Method Efficiency**:
   - `average` fusion is the fastest (mean: 0.1244s)
   - `sum` fusion has the highest variance due to extreme outliers
   - `standard_concat` is the slowest among common methods (mean: 0.1708s)

4. **Missing Data Impact**: Fitting times are relatively consistent across missing data percentages, with 50% missing data showing slightly higher mean times due to outliers

5. **Task Type**: Classification tasks have higher variance due to RandomForestClassifier outliers, but similar median times to regression

6. **Outlier Analysis**: The extreme case (42.59s) involves SparsePLS + RandomForestClassifier + sum fusion, suggesting this combination should be avoided for time-sensitive applications

## Recommendations

1. **For Speed-Critical Applications**:
   - Use LDA or KPCA extractors
   - Prefer LogisticRegression or ElasticNet models
   - Use `average` or `max` fusion methods
   - Avoid RandomForestClassifier with SparsePLS

2. **For Balanced Performance**:
   - Use PCA or FA extractors
   - Consider SVC for classification
   - Use `early_fusion_pca` or `attention_weighted` fusion

3. **Avoid Combinations**:
   - SparsePLS + RandomForestClassifier + sum fusion (extremely slow)
   - RandomForestClassifier with complex fusion methods
   - VarianceFTest with large datasets (high variance) 