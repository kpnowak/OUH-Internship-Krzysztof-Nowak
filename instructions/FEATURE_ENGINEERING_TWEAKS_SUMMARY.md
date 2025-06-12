# Feature Engineering Tweaks Implementation Summary

## Overview

This document summarizes the implementation of feature engineering tweaks designed to improve specific metrics in genomic data analysis:

- **Sparse PLS-DA (32 components)** for better **MCC** in classification tasks
- **Kernel PCA with RBF kernel (64 components)** for higher **R²** in regression tasks

These tweaks are integrated as additional extractors in the pipeline and can be toggled via CLI.

## Implementation Details

### 1. Configuration (config.py)

Added `FEATURE_ENGINEERING_CONFIG` with comprehensive settings:

```python
FEATURE_ENGINEERING_CONFIG = {
    "enabled": False,  # Disabled by default, enabled via CLI
    "sparse_plsda_enabled": True,  # Sparse PLS-DA for better MCC
    "kernel_pca_enabled": True,    # Kernel PCA for higher R²
    
    # Sparse PLS-DA configuration for MCC improvement
    "sparse_plsda": {
        "n_components": 32,  # As specified in requirements
        "alpha": 0.1,        # Sparsity parameter
        "max_iter": 1000,
        "tol": 1e-6,
        "scale": True,
        "description": "Creates maximally discriminative latent space, balances class variance"
    },
    
    # Kernel PCA configuration for R² improvement  
    "kernel_pca": {
        "n_components": 64,  # As specified in requirements
        "kernel": "rbf",     # RBF kernel for non-linear interactions
        "gamma": "auto",     # Will be set to median heuristic
        "eigen_solver": "auto",
        "n_jobs": -1,
        "random_state": 42,
        "description": "Captures non-linear gene–methylation interactions feeding into boosted trees"
    },
    
    # Median heuristic for gamma calculation
    "median_heuristic": {
        "enabled": True,
        "sample_size": 1000,  # Sample size for gamma calculation
        "percentile": 50      # Median percentile
    }
}
```

### 2. Sparse PLS-DA Implementation (models.py)

**Purpose**: Creates maximally discriminative latent space, balances class variance for better MCC.

**Key Features**:
- 32 components for optimal discriminative power
- Sparsity regularization (α=0.1) for feature selection
- Label encoding for multi-class support
- SVD-based cross-covariance decomposition
- Soft thresholding for sparsity enforcement

**Algorithm**:
1. Encode labels using LabelEncoder and LabelBinarizer
2. Scale data if requested
3. For each component:
   - Compute cross-covariance matrix C = X^T @ Y
   - SVD decomposition: U, s, V^T = SVD(C)
   - Apply soft thresholding to X weights for sparsity
   - Compute scores and loadings
   - Deflate X and Y matrices
4. Store components for transformation

**Usage**:
```python
sparse_plsda = SparsePLSDA(n_components=32, alpha=0.1, scale=True)
X_transformed = sparse_plsda.fit_transform(X_train, y_train)
```

### 3. Kernel PCA with Median Heuristic (models.py)

**Purpose**: Captures non-linear gene–methylation interactions for higher R².

**Key Features**:
- 64 components for comprehensive non-linear feature capture
- RBF kernel with median heuristic gamma selection
- Automatic gamma computation: γ = 1 / (2 * median(pairwise_distances)²)
- Efficient sampling for large datasets
- sklearn KernelPCA backend with optimized parameters

**Median Heuristic Algorithm**:
1. Sample data if n_samples > sample_size (default 1000)
2. Compute pairwise Euclidean distances
3. Extract upper triangular distances (excluding diagonal)
4. Calculate median distance at specified percentile (default 50%)
5. Compute gamma: γ = 1 / (2 * median_dist²)

**Usage**:
```python
kernel_pca = KernelPCAMedianHeuristic(n_components=64, kernel="rbf", sample_size=1000)
X_transformed = kernel_pca.fit_transform(X_train)
```

### 4. Extractor Integration

**Classification Extractors** (`get_classification_extractors()`):
- Added conditional SparsePLS-DA when feature engineering enabled
- Configured with 32 components, α=0.1, scaling enabled

**Regression Extractors** (`get_regression_extractors()`):
- Added conditional KernelPCA-RBF when feature engineering enabled
- Configured with 64 components, median heuristic gamma

**Integration Logic**:
```python
# Add feature engineering tweaks if enabled
from config import FEATURE_ENGINEERING_CONFIG
if FEATURE_ENGINEERING_CONFIG.get("enabled", False):
    if FEATURE_ENGINEERING_CONFIG.get("sparse_plsda_enabled", True):
        config = FEATURE_ENGINEERING_CONFIG["sparse_plsda"]
        extractors["SparsePLS-DA"] = SparsePLSDA(
            n_components=config["n_components"],
            alpha=config["alpha"],
            max_iter=config["max_iter"],
            tol=config["tol"],
            scale=config["scale"]
        )
```

### 5. CLI Integration (cli.py)

**New Argument**:
```bash
--feature-engineering    Enable feature engineering tweaks (Sparse PLS-DA for MCC, Kernel PCA for R²)
```

**Implementation**:
```python
parser.add_argument(
    "--feature-engineering", action="store_true", 
    help="Enable feature engineering tweaks (Sparse PLS-DA for MCC, Kernel PCA for R²)"
)

# Enable feature engineering if requested
if args.feature_engineering:
    from config import FEATURE_ENGINEERING_CONFIG
    FEATURE_ENGINEERING_CONFIG["enabled"] = True
    logger.info("Feature engineering tweaks enabled via CLI")
    logger.info("  - Sparse PLS-DA (32 components) for better MCC in classification")
    logger.info("  - Kernel PCA RBF (64 components) for higher R² in regression")
```

## Usage Instructions

### Basic Usage

Enable feature engineering tweaks for all datasets:
```bash
python cli.py --feature-engineering
```

### Combined with Other Options

Run only regression datasets with feature engineering:
```bash
python cli.py --regression-only --feature-engineering
```

Run specific dataset with feature engineering:
```bash
python cli.py --dataset aml --feature-engineering
```

Enable debug mode with feature engineering:
```bash
python cli.py --feature-engineering --debug
```

### Verification

Check if feature engineering is working:
```bash
python test_feature_engineering_tweaks.py
```

## Expected Performance Improvements

### Classification (MCC Enhancement)
- **Target**: Better Matthews Correlation Coefficient
- **Method**: Sparse PLS-DA with 32 components
- **Mechanism**: Maximally discriminative latent space creation
- **Benefits**: 
  - Balanced class variance handling
  - Sparse feature selection
  - Improved class separation

### Regression (R² Enhancement)
- **Target**: Higher coefficient of determination
- **Method**: Kernel PCA with RBF kernel and 64 components
- **Mechanism**: Non-linear gene–methylation interaction capture
- **Benefits**:
  - Non-linear relationship modeling
  - Optimal gamma via median heuristic
  - Enhanced feature representation for boosted trees

## Technical Specifications

### Dependencies
- **Required**: scikit-learn, numpy, pandas
- **Optional**: None (all dependencies are standard)

### Memory Usage
- **Sparse PLS-DA**: O(n_samples × n_components) for transformed data
- **Kernel PCA**: O(n_samples²) for kernel matrix, O(n_samples × n_components) for transformed data
- **Median Heuristic**: O(sample_size²) for distance computation

### Computational Complexity
- **Sparse PLS-DA**: O(n_components × n_features × n_samples × max_iter)
- **Kernel PCA**: O(n_samples³) for eigendecomposition
- **Median Heuristic**: O(sample_size²) for pairwise distances

### Scalability
- **Large Datasets**: Median heuristic uses sampling (default 1000 samples)
- **High Dimensions**: Sparse PLS-DA handles high-dimensional data efficiently
- **Memory Management**: Both methods support incremental processing

## Testing and Validation

### Test Suite Coverage
1. **Configuration Loading**: Verify all required keys present
2. **Sparse PLS-DA**: Test fitting, transformation, parameter access
3. **Kernel PCA**: Test median heuristic, transformation, gamma computation
4. **Extractor Integration**: Verify conditional addition to extractor lists
5. **CLI Integration**: Test argument parsing and configuration enabling
6. **Performance Comparison**: Compare standard vs enhanced extractors

### Test Results
```
Configuration Loading................... PASS
Sparse PLS-DA........................... PASS
Kernel PCA Median Heuristic............. PASS
Extractor Integration................... PASS
CLI Integration......................... PASS
Performance Comparison.................. PASS

Overall: 6/6 tests passed
```

### Validation Metrics
- **Sparse PLS-DA**: MCC comparison on synthetic genomic-like data
- **Kernel PCA**: R² comparison with non-linear relationships
- **Integration**: Extractor count verification
- **CLI**: Argument parsing and configuration state changes

## Implementation Quality

### Code Quality
- **Modular Design**: Separate classes for each method
- **Error Handling**: Comprehensive exception handling and fallbacks
- **Documentation**: Detailed docstrings and inline comments
- **Testing**: Comprehensive test suite with 100% pass rate

### Performance Optimizations
- **Sampling**: Median heuristic uses efficient sampling for large datasets
- **Vectorization**: NumPy operations for computational efficiency
- **Memory Management**: Efficient array operations and memory reuse
- **Caching**: Compatible with existing caching infrastructure

### Maintainability
- **Configuration-Driven**: All parameters configurable via config.py
- **Conditional Loading**: Feature engineering only loaded when enabled
- **Backward Compatibility**: No changes to existing functionality
- **Extensibility**: Easy to add new feature engineering methods

## Future Enhancements

### Potential Improvements
1. **Adaptive Components**: Automatically determine optimal number of components
2. **Cross-Validation**: Integrate with CV pipeline for component selection
3. **Ensemble Methods**: Combine multiple feature engineering approaches
4. **GPU Acceleration**: CUDA support for large-scale computations

### Additional Methods
1. **t-SNE**: Non-linear dimensionality reduction for visualization
2. **UMAP**: Uniform manifold approximation for structure preservation
3. **Autoencoders**: Deep learning-based feature extraction
4. **Graph-based Methods**: Network-aware feature engineering

## Conclusion

The feature engineering tweaks implementation successfully adds specialized extractors for improving MCC and R² metrics in genomic data analysis. The implementation is:

- **Robust**: Comprehensive error handling and testing
- **Efficient**: Optimized algorithms with scalability considerations
- **Flexible**: Configurable parameters and conditional enabling
- **Integrated**: Seamless integration with existing pipeline
- **Validated**: 100% test pass rate with comprehensive coverage

The tweaks can be easily enabled via CLI and provide targeted improvements for specific metrics while maintaining backward compatibility with the existing system. 