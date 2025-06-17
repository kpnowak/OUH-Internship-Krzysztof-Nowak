# Enhanced Missing Data Handling Implementation

## Overview

This implementation extends the `ModalityImputer` class in `fusion.py` with robust missing data handling strategies as requested in the enhancement specification. The implementation includes:

1. **KNN imputation (k=5)** for moderate missing data to preserve local structure
2. **Iterative Imputer with ExtraTrees** for highly missing blocks (>50%)
3. **Late-fusion fallback** for samples lacking entire modalities
4. **Adaptive strategy selection** based on missing data characteristics

## Implementation Details

### 1. Enhanced ModalityImputer Class

The `ModalityImputer` class now supports multiple imputation strategies:

```python
class ModalityImputer:
    def __init__(self, strategy='adaptive', k_neighbors=5, 
                 high_missing_threshold=0.5, random_state=42):
        """
        Enhanced imputer with multiple strategies:
        - 'mean': Simple mean imputation (fast)
        - 'knn': KNN imputation (k=5) for preserving local structure
        - 'iterative': Iterative imputation with ExtraTrees for high missing data
        - 'adaptive': Automatically chooses strategy based on missing percentage
        """
```

#### Strategy Selection Logic (Adaptive Mode):
- **0% missing**: No imputation needed
- **< 10% missing**: Mean imputation (fast for low missing data)
- **10-50% missing**: KNN imputation (preserves local structure)
- **> 50% missing**: Iterative imputation with ExtraTrees (robust for high missing data)

### 2. Late-Fusion Fallback

New `LateFusionFallback` class handles samples with missing entire modalities:

```python
class LateFusionFallback:
    def __init__(self, is_regression=True, reliability_metric='auto', 
                 min_modalities=1, random_state=42):
        """
        Late-fusion fallback for missing entire modalities.
        Uses only available modalities and weights by individual reliability.
        """
```

#### Key Features:
- **Individual modality models**: Fits separate models for each modality
- **Reliability assessment**: Uses cross-validation to assess modality performance
- **Weighted predictions**: Combines predictions weighted by reliability scores
- **Graceful degradation**: Works with any subset of available modalities

### 3. Enhanced Fusion Integration

The `merge_modalities()` function now includes:

#### Automatic Imputation:
```python
# Auto-create enhanced imputer if missing data is detected
if has_missing_values and imputer is None:
    auto_imputer = ModalityImputer(strategy='adaptive')
    merged = auto_imputer.fit_transform(merged)
```

#### Strategy Restrictions:
- **weighted_concat**: Restricted to 0% missing data only
- **Other strategies**: Support 0%, 20%, and 50% missing data levels

### 4. Helper Functions

#### Missing Modality Detection:
```python
def detect_missing_modalities(modalities, missing_threshold=0.9):
    """Detect which modalities have excessive missing data (>90%)"""
    return available_modalities, missing_modalities
```

#### Strategy Recommendation:
```python
def get_recommended_fusion_strategy(missing_percentage, has_targets=True, n_modalities=2):
    """Get recommended fusion strategy based on data characteristics"""
    # Returns optimal strategy based on missing data level and available information
```

#### Enhanced Imputer Factory:
```python
def create_enhanced_imputer(strategy='adaptive', **kwargs):
    """Factory function to create optimally configured imputer"""
    return ModalityImputer(strategy=strategy, **kwargs)
```

## Usage Examples

### Basic Enhanced Imputation

```python
from fusion import ModalityImputer

# Create adaptive imputer
imputer = ModalityImputer(strategy='adaptive', k_neighbors=5)

# Fit and transform data with missing values
imputed_data = imputer.fit_transform(data_with_missing)

# Check what strategy was chosen
strategy_info = imputer.get_strategy_info()
print(f"Chosen strategy: {strategy_info['chosen_strategy']}")
print(f"Missing percentage: {strategy_info['missing_percentage']:.2f}%")
```

### Late-Fusion Fallback

```python
from fusion import handle_missing_modalities_with_late_fusion

# Handle missing entire modalities
late_fusion = handle_missing_modalities_with_late_fusion(
    modalities, y, is_regression=True, 
    modality_names=['Gene_Expression', 'Methylation', 'miRNA']
)

# Make predictions with partial modalities
predictions = late_fusion.predict([modality1, None, modality3])  # modality2 missing
```

### Integrated Fusion with Enhanced Imputation

```python
from fusion import merge_modalities, create_enhanced_imputer

# Create enhanced imputer
imputer = create_enhanced_imputer(strategy='adaptive')

# Merge modalities with automatic imputation
merged_data = merge_modalities(
    modality1, modality2, modality3,
    strategy='learnable_weighted',
    imputer=imputer,
    y=targets,
    is_regression=True
)
```

### Strategy Recommendation

```python
from fusion import get_recommended_fusion_strategy

# Get recommended strategy based on data characteristics
missing_pct = 25.0  # 25% missing data
recommended = get_recommended_fusion_strategy(
    missing_pct, has_targets=True, n_modalities=3
)
print(f"Recommended strategy: {recommended}")  # Output: 'learnable_weighted'
```

## Performance Characteristics

### Imputation Strategy Performance:

| Strategy | Missing Data Range | Speed | Quality | Use Case |
|----------|-------------------|-------|---------|----------|
| Mean | 0-10% | Very Fast | Good | Low missing data |
| KNN | 10-50% | Moderate | Very Good | Preserves local structure |
| Iterative | >50% | Slow | Excellent | High missing data |
| Adaptive | Any | Variable | Optimal | Automatic selection |

### Fusion Strategy Compatibility:

| Strategy | Missing Data Support | Restriction |
|----------|---------------------|-------------|
| weighted_concat |  | 0% missing only |
| learnable_weighted |  | 0%, 20%, 50% |
| mkl |  | 0%, 20%, 50% |
| snf |  | 0%, 20%, 50% |
| early_fusion_pca |  | 0%, 20%, 50% |

## Technical Implementation Notes

### Dependencies:
- `sklearn.impute.KNNImputer` for KNN imputation
- `sklearn.impute.IterativeImputer` for iterative imputation
- `sklearn.ensemble.ExtraTreesRegressor` as estimator for iterative imputation

### Error Handling:
- Graceful fallback to mean imputation if advanced methods fail
- Comprehensive logging of strategy selection and performance
- Robust handling of edge cases (empty arrays, all-NaN columns)

### Memory Optimization:
- Uses `float32` dtype to reduce memory usage
- In-place operations where possible
- Efficient handling of large sparse matrices

### Validation:
- Automatic detection of missing data patterns
- Verification that no NaNs remain after imputation
- Cross-validation for reliability assessment in late-fusion

## Integration with Existing Pipeline

The enhanced missing data handling is fully integrated with the existing genomic analysis pipeline:

1. **Automatic Detection**: Missing data is automatically detected during fusion
2. **Strategy Selection**: Optimal imputation strategy is chosen based on data characteristics
3. **Seamless Integration**: Works with all existing fusion strategies (with appropriate restrictions)
4. **Backward Compatibility**: Existing code continues to work without modification

## Benefits for Genomic Data Analysis

1. **Improved Data Quality**: KNN and iterative imputation preserve biological relationships
2. **Robust Handling**: Can handle extreme missing data scenarios (>50% missing)
3. **Adaptive Behavior**: Automatically selects optimal strategy for each dataset
4. **Late-Fusion Capability**: Handles samples with missing entire omics modalities
5. **Performance Optimization**: Balances imputation quality with computational efficiency

This implementation significantly enhances the robustness of multi-modal genomic data analysis by providing sophisticated missing data handling while maintaining computational efficiency and ease of use. 