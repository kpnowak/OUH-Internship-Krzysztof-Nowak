# Fusion Enhancements for Multi-Modal Genomic Data

This document describes the three major enhancements implemented in the `fusion.py` module for improved multi-modal data integration in genomic analysis.

## Overview

The fusion module has been enhanced with three advanced fusion strategies:

1. **Learnable Weighted Fusion** - Automatically learns optimal weights based on modality performance
2. **Multiple-Kernel Learning (MKL)** - Combines RBF kernels from each modality optimally
3. **Similarity Network Fusion (SNF)** - Creates and fuses similarity networks with spectral clustering

## Important Restrictions

### **Fusion Strategy Usage by Missing Data Percentage:**

- **weighted_concat**: Only allowed with **0% missing data**
- **learnable_weighted, mkl, snf, early_fusion_pca**: Allowed with **0%, 20%, and 50% missing data**
- **average, sum**: **COMMENTED OUT** - These fusion techniques have been disabled

### **Missing Data Policy:**
If you attempt to use `weighted_concat` with missing data, the system will:
1. Log an error message explaining the restriction
2. Fall back to simple concatenation for compatibility
3. Recommend using alternative fusion strategies (learnable_weighted, mkl, snf, or early_fusion_pca)

## 1. Learnable Weighted Fusion

### Description
Replaces static scalar weights with learnable weights proportional to each modality's standalone validation performance (AUC for classification, R² for regression).

### Formula
```
w_i = perf_i / Σ perf_i
X_fused = np.hstack([w_i * Z_i for each modality i])
```

### Usage
```python
from fusion import merge_modalities

# For data with 0% missing values
merged_data = merge_modalities(
    modality1, modality2, modality3,
    strategy="learnable_weighted",
    y=target_values,
    is_regression=True,
    fusion_params={'cv_folds': 3, 'random_state': 42}
)

# For data with missing values (20% or 50%)
merged_data = merge_modalities(
    modality1, modality2, modality3,
    strategy="learnable_weighted",
    y=target_values,
    is_regression=True,
    fusion_params={'cv_folds': 3, 'random_state': 42}
)
```

### Key Features
- **Automatic weight learning** based on cross-validation performance
- **Performance-based weighting**: Better-performing modalities get higher weights
- **Robust fallbacks**: Falls back to equal weights if performance estimation fails
- **Works with missing data**: Unlike weighted_concat, this strategy handles missing values

## 2. Multiple-Kernel Learning (MKL)

### Description
For each omic modality, builds an RBF kernel and combines them optimally using kernel learning techniques. Works especially well with SVM/SVR and avoids dimensionality issues.

### Usage
```python
from fusion import MultipleKernelLearning

# Initialize MKL
mkl = MultipleKernelLearning(
    is_regression=True,
    n_components=10,
    gamma=1.0,
    random_state=42
)

# Fit and transform
X_fused = mkl.fit_transform([modality1, modality2, modality3], y)

# Or use in merge_modalities
merged_data = merge_modalities(
    modality1, modality2, modality3,
    strategy="mkl",
    y=target_values,
    fusion_params={'n_components': 10, 'gamma': 1.0}
)
```

### Key Features
- **Kernel-based fusion**: Builds RBF kernels for each modality
- **Optimal combination**: Uses kernel learning to find optimal weights
- **SVM/SVR compatible**: Works seamlessly with kernel-based models
- **Dimensionality robust**: Avoids curse of dimensionality issues
- **Missing data support**: Handles 0%, 20%, and 50% missing data

## 3. Similarity Network Fusion (SNF)

### Description
Leverages the `snfpy` package to create similarity networks for each modality and fuses them. Can be used with spectral clustering (unsupervised) or pre-computed-kernel SVC/SVR (supervised).

### Usage
```python
from fusion import SimilarityNetworkFusion

# Initialize SNF
snf = SimilarityNetworkFusion(
    K=20,                    # Number of neighbors
    alpha=0.5,               # Hyperparameter for fusion
    T=20,                    # Number of iterations
    use_spectral_clustering=True,
    n_clusters=3,
    random_state=42
)

# Fit and transform
X_fused = snf.fit_transform([modality1, modality2, modality3], y)

# Or use in merge_modalities
merged_data = merge_modalities(
    modality1, modality2, modality3,
    strategy="snf",
    y=target_values,
    fusion_params={'K': 20, 'alpha': 0.5, 'T': 20}
)
```

### Key Features
- **Network-based fusion**: Creates and fuses similarity networks
- **Flexible clustering**: Supports both supervised and unsupervised approaches
- **Spectral clustering**: Uses advanced clustering techniques
- **Kernel compatibility**: Works with pre-computed kernels for SVC/SVR
- **Missing data support**: Handles 0%, 20%, and 50% missing data

## Integration with Main Pipeline

### Updated merge_modalities Function

The main `merge_modalities` function now supports all new strategies:

```python
def merge_modalities(*arrays, 
                    strategy="weighted_concat",  # RESTRICTED to 0% missing data
                    y=None,                      # Required for learnable weights
                    is_regression=True,
                    fusion_params=None):
```

### Strategy Selection Guidelines

**For 0% Missing Data:**
- `weighted_concat` - Fast, traditional approach
- `learnable_weighted` - Performance-optimized weights
- `mkl` - Kernel-based fusion
- `snf` - Network-based fusion
- `early_fusion_pca` - PCA-based dimensionality reduction

**For 20% or 50% Missing Data:**
- `learnable_weighted` - **Recommended** for performance-based fusion
- `mkl` - Good for kernel-based models
- `snf` - Good for network-based analysis
- `early_fusion_pca` - Good for dimensionality reduction

**NOT AVAILABLE:**
- `average` - Commented out
- `sum` - Commented out
- `weighted_concat` - Restricted to 0% missing data only

## Dependencies

### Required
- `numpy`
- `scikit-learn`
- `scipy`

### Optional (for advanced features)
- `snfpy` - For Similarity Network Fusion
- `mklaren` - For Multiple-Kernel Learning (if using EasyMKL)

## Installation

```bash
# Install required dependencies
pip install numpy scikit-learn scipy

# Install optional dependencies for advanced features
pip install snfpy  # For SNF support
```

## Performance Considerations

### Computational Complexity
- **learnable_weighted**: O(k × CV × model_training) where k = number of modalities
- **mkl**: O(n² × k) for kernel computation + optimization
- **snf**: O(n² × k × T) where T = number of iterations

### Memory Usage
- **learnable_weighted**: Moderate (stores performance metrics)
- **mkl**: High (stores kernel matrices)
- **snf**: High (stores similarity networks)

### Recommendations
- For **large datasets**: Use `learnable_weighted`
- For **kernel methods**: Use `mkl`
- For **network analysis**: Use `snf`
- For **speed**: Use `weighted_concat` (0% missing data only)

## Error Handling

The enhanced fusion module includes robust error handling:

1. **Missing data validation**: Prevents `weighted_concat` with missing data
2. **Fallback mechanisms**: Graceful degradation when advanced methods fail
3. **Input validation**: Comprehensive checks for data consistency
4. **Informative logging**: Clear error messages and recommendations

## Example: Complete Workflow

```python
import numpy as np
from fusion import merge_modalities

# Example with genomic data (0% missing)
gene_expr = np.random.randn(100, 1000)    # 100 samples, 1000 genes
mirna_expr = np.random.randn(100, 200)    # 100 samples, 200 miRNAs
clinical = np.random.randn(100, 50)       # 100 samples, 50 clinical features
target = np.random.randn(100)             # Regression target

# Use learnable weighted fusion
fused_data = merge_modalities(
    gene_expr, mirna_expr, clinical,
    strategy="learnable_weighted",
    y=target,
    is_regression=True,
    fusion_params={'cv_folds': 5, 'random_state': 42}
)

print(f"Original shapes: {gene_expr.shape}, {mirna_expr.shape}, {clinical.shape}")
print(f"Fused shape: {fused_data.shape}")
```

## Troubleshooting

### Common Issues

1. **"weighted_concat strategy is only allowed with 0% missing data"**
   - **Solution**: Use `learnable_weighted`, `mkl`, `snf`, or `early_fusion_pca` instead

2. **"snfpy not available"**
   - **Solution**: Install with `pip install snfpy` or use alternative strategies

3. **Memory errors with large datasets**
   - **Solution**: Use `learnable_weighted` instead of `mkl` or `snf`

4. **Poor fusion performance**
   - **Solution**: Try different strategies; `learnable_weighted` often works best

## Future Enhancements

Planned improvements include:
- Adaptive strategy selection based on data characteristics
- Hierarchical fusion for very large datasets
- Integration with deep learning approaches
- Support for temporal/sequential data fusion 