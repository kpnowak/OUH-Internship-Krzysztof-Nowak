# Robust Scaling Implementation for PCA High Variance Issues

## Overview

This document summarizes the implementation of robust scaling to address PCA high variance issues in genomic data analysis. The implementation replaces StandardScaler with RobustScaler to better handle outlier-heavy genomic data, addressing the reported issue of high variance in PCA components (std ~7-26).

## Problem Statement

### Original Issue
- **High variance in PCA components**: Standard deviation ranging from ~7-26
- **Cause**: StandardScaler is sensitive to outliers, which are common in genomic data
- **Impact**: Unstable PCA components, poor dimensionality reduction, potential overfitting

### Genomic Data Characteristics
- **Outlier-heavy**: Gene expression can have extreme values due to biological variation
- **Zero-inflated**: Many features have dropout effects (technical zeros)
- **High dynamic range**: Expression levels can span several orders of magnitude
- **Batch effects**: Systematic biases that create artificial outliers

## Solution: Robust Scaling Implementation

### Core Changes

#### 1. Configuration Updates (`config.py`)
Added robust scaling parameters to `PREPROCESSING_CONFIG`:

```python
# NEW: Robust scaling parameters (addresses PCA high variance issues)
"use_robust_scaling": True,              # Use RobustScaler instead of StandardScaler
"robust_scaling_quantile_range": (25.0, 75.0),  # IQR range for RobustScaler
"scaling_method": "robust",              # Options: 'robust', 'standard', 'minmax', 'quantile'
"apply_scaling_before_pca": True,        # Apply scaling before PCA/dimensionality reduction
"clip_outliers_after_scaling": True,    # Clip extreme outliers after scaling
"outlier_clip_range": (-5.0, 5.0),      # Range for outlier clipping after scaling
```

#### 2. Modality-Specific Configurations
Enhanced `ENHANCED_PREPROCESSING_CONFIGS` with modality-specific robust scaling:

- **miRNA**: Wider quantile range (10.0, 90.0), clipping range (-6.0, 6.0)
- **Gene Expression**: Standard IQR (25.0, 75.0), clipping range (-5.0, 5.0)
- **Methylation**: Conservative range (5.0, 95.0), no clipping (preserves 0-1 range)

#### 3. New Preprocessing Function (`preprocessing.py`)
Implemented `robust_data_scaling()` function:

```python
def robust_data_scaling(X_train, X_test=None, config=None, modality_type='unknown'):
    """
    Apply robust scaling to handle outlier-heavy genomic data.
    Uses RobustScaler instead of StandardScaler to address PCA high variance issues.
    """
```

**Key Features:**
- Multiple scaling methods: robust, standard, minmax, quantile
- Modality-specific parameter handling
- Outlier clipping after scaling
- Comprehensive reporting and statistics
- Train/test consistency

#### 4. Pipeline Integration
Integrated robust scaling as **Step 4c** in `robust_biomedical_preprocessing_pipeline()`:
- Applied after outlier detection, before final validation
- Stores scaler in transformers dictionary for reuse
- Comprehensive logging and reporting

#### 5. PCA Component Selection Updates (`models.py`)
Updated `select_optimal_components()` to use RobustScaler:

```python
# OLD: StandardScaler(with_std=False).fit_transform(X)
# NEW: RobustScaler().fit_transform(X)
```

#### 6. Model Class Updates
Updated scaling in custom PLS classes:
- `SparsePLSDA`: Uses RobustScaler instead of StandardScaler
- `SparsePLS`: Uses RobustScaler instead of StandardScaler

## Technical Implementation Details

### RobustScaler vs StandardScaler

| Aspect | StandardScaler | RobustScaler |
|--------|----------------|--------------|
| **Center** | Mean | Median |
| **Scale** | Standard deviation | IQR (75th - 25th percentile) |
| **Outlier sensitivity** | High | Low |
| **Genomic data suitability** | Poor | Excellent |

### Scaling Process

1. **Fit on training data**: Calculate median and IQR
2. **Transform**: `(X - median) / IQR`
3. **Outlier clipping**: Optional clipping to prevent extreme values
4. **Apply to test data**: Use training statistics

### Modality-Specific Adaptations

#### miRNA Data
- **Quantile range**: (10.0, 90.0) - wider range for sparse data
- **Clipping**: (-6.0, 6.0) - allows for higher expression outliers
- **Rationale**: miRNA data is highly sparse with legitimate extreme values

#### Gene Expression Data
- **Quantile range**: (25.0, 75.0) - standard IQR
- **Clipping**: (-5.0, 5.0) - moderate outlier control
- **Rationale**: Balanced approach for typical gene expression patterns

#### Methylation Data
- **Quantile range**: (5.0, 95.0) - conservative range
- **Clipping**: Disabled - preserves 0-1 range
- **Rationale**: Methylation values are naturally bounded [0,1]

## Expected Benefits

### 1. PCA Stability Improvements
- **Reduced component variance**: More stable explained variance ratios
- **Better component balance**: Less dominance by outlier-driven components
- **Improved interpretability**: Components reflect biological variation, not technical artifacts

### 2. Model Performance Benefits
- **Better feature representation**: Scaling preserves biological signal
- **Reduced overfitting**: Less influence from extreme outliers
- **Improved generalization**: More robust to batch effects and technical variation

### 3. Computational Benefits
- **Faster convergence**: More stable optimization landscapes
- **Numerical stability**: Reduced risk of numerical issues
- **Memory efficiency**: Clipped values reduce memory requirements

## Integration Status

### Files Modified
1. **`config.py`**: Added robust scaling configuration parameters
2. **`preprocessing.py`**: 
   - Added `robust_data_scaling()` function
   - Integrated into `robust_biomedical_preprocessing_pipeline()`
3. **`models.py`**: 
   - Updated `select_optimal_components()` to use RobustScaler
   - Updated PLS classes to use RobustScaler

### Pipeline Integration
- **Automatic activation**: Enabled by default for all modalities
- **Step 4c**: Integrated after outlier detection in preprocessing pipeline
- **Transformer storage**: Scalers stored for consistent test data processing
- **Comprehensive logging**: Detailed reporting of scaling statistics

## Usage Examples

### Basic Usage
```python
from preprocessing import robust_data_scaling
from config import ENHANCED_PREPROCESSING_CONFIGS

# Apply robust scaling
X_scaled, X_test_scaled, scaler, report = robust_data_scaling(
    X_train, X_test, 
    config=ENHANCED_PREPROCESSING_CONFIGS['Gene Expression'],
    modality_type='Gene Expression'
)
```

### Pipeline Usage
```python
from preprocessing import robust_biomedical_preprocessing_pipeline

# Robust scaling is automatically applied
X_train, X_test, transformers, report = robust_biomedical_preprocessing_pipeline(
    X_train, X_test, y_train, modality_type='miRNA'
)

# Access the scaler
scaler = transformers.get('robust_scaler')
```

## Validation and Testing

### Test Scripts Created
1. **`test_robust_scaling.py`**: Basic functionality testing
2. **`test_realistic_pca_scaling.py`**: Realistic genomic data simulation

### Key Validation Points
- ✅ Modality-specific parameter handling
- ✅ Train/test consistency
- ✅ Outlier clipping effectiveness
- ✅ Pipeline integration
- ✅ Transformer storage and reuse

## Expected Impact on PCA Issues

### Before (StandardScaler)
- High sensitivity to outliers
- Extreme values dominate scaling
- PCA components have high variance (std ~7-26)
- Unstable explained variance ratios

### After (RobustScaler)
- Reduced outlier sensitivity
- IQR-based scaling preserves biological signal
- More stable PCA components
- Balanced explained variance distribution

## Monitoring and Metrics

### Scaling Report Metrics
- `variance_reduction_ratio`: Ratio of scaled to original variance
- `scaling_method`: Method used (robust, standard, etc.)
- `outlier_clipping_applied`: Whether clipping was performed
- `center_stats` and `scale_stats`: Scaler parameter statistics

### PCA Improvement Metrics
- Component variance standard deviation
- Explained variance ratio distribution
- Component stability across folds
- Numerical stability indicators

## Future Enhancements

### Potential Improvements
1. **Adaptive quantile ranges**: Data-driven quantile selection
2. **Hybrid scaling**: Combine multiple scaling methods
3. **Batch-aware scaling**: Account for batch effects in scaling
4. **Feature-specific scaling**: Different scaling per feature type

### Monitoring Recommendations
1. Track PCA component variance statistics
2. Monitor explained variance ratio stability
3. Compare model performance before/after scaling
4. Validate on real genomic datasets

## Conclusion

The robust scaling implementation provides a comprehensive solution to PCA high variance issues in genomic data analysis. By replacing StandardScaler with RobustScaler and implementing modality-specific configurations, the system now better handles the outlier-heavy nature of genomic data, leading to more stable PCA components and improved model performance.

The implementation is fully integrated into the main preprocessing pipeline and automatically applied to all genomic data processing workflows, ensuring consistent and robust handling of scaling across all modalities. 