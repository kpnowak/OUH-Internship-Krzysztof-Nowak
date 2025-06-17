# K-Neighbors Fixes Summary

## Problem Description
The system was experiencing widespread "Expected n_neighbors <= n_samples_fit" errors, particularly in the Colon dataset classification pipeline. The errors were occurring in multiple places:

1. **KNNImputer**: Trying to use 5 neighbors when only 2-5 samples available
2. **SMOTE**: Trying to use 5 neighbors for oversampling with very small class sizes
3. **Mutual Info Functions**: Using n_neighbors=3+ when fewer samples available

## Root Cause Analysis
The issue was that various algorithms were using fixed k_neighbors parameters (typically 5) without checking if enough samples were available. When datasets have very few samples (2-5), these algorithms fail because:

- KNNImputer needs at least k+1 samples to find k neighbors
- SMOTE needs at least k+1 samples in the smallest class
- Mutual info estimation needs at least k+1 samples total

## Fixes Implemented

### 1. **KNNImputer Fix** (`fusion.py`)
**Location**: `ModalityImputer.fit()` method, lines ~115-130

**Problem**: KNNImputer using k_neighbors=5 with only 2-5 samples
**Solution**: 
- Calculate safe_neighbors = min(k_neighbors, max(1, n_samples - 1))
- Fall back to mean imputation if fewer than 3 samples
- Add comprehensive logging for debugging

```python
# Critical fix: ensure n_neighbors is valid for the dataset size
max_neighbors = max(1, X.shape[0] - 1)  # At least 1, at most n_samples - 1
safe_neighbors = min(self.k_neighbors, max_neighbors)

# Additional safety: if we have very few samples, fall back to mean imputation
if X.shape[0] < 3:  # Need at least 3 samples for meaningful KNN
    logger.warning(f"Too few samples ({X.shape[0]}) for KNN imputation, falling back to mean")
    self.chosen_strategy_ = 'mean'
    # ... fallback to mean imputation
else:
    logger.debug(f"KNN imputation: using {safe_neighbors} neighbors for {X.shape[0]} samples")
    self.knn_imputer_ = KNNImputer(n_neighbors=safe_neighbors, ...)
```

### 2. **Mutual Info Fix** (`mrmr_helper.py`)
**Location**: `fast_mutual_info_batch()` function, lines ~25-40

**Problem**: mutual_info_classif/regression using n_neighbors without validation
**Solution**:
- Calculate safe_neighbors = min(n_neighbors, max(1, n_samples - 1))
- Use n_neighbors=1 for very small datasets (< 3 samples)
- Apply to both classification and regression

```python
# Critical fix: ensure n_neighbors doesn't exceed available samples
n_samples = X.shape[0]
safe_neighbors = min(n_neighbors, max(1, n_samples - 1))

# Additional safety for very small datasets
if n_samples < 3:
    logger.warning(f"Very small dataset ({n_samples} samples) for mutual info, using n_neighbors=1")
    safe_neighbors = 1

if is_regression:
    return mutual_info_regression(X, y, n_neighbors=safe_neighbors, random_state=42)
else:
    return mutual_info_classif(X, y, n_neighbors=safe_neighbors, random_state=42)
```

### 3. **SMOTE Fix** (`cv.py`)
**Location**: `create_balanced_pipeline()` function, lines ~75-110

**Problem**: SMOTE using k_neighbors=5 when class sizes are smaller
**Solution**:
- Created SafeSMOTE wrapper class that dynamically adjusts k_neighbors
- Check smallest class size and adjust k_neighbors accordingly
- Skip SMOTE entirely if smallest class has < 3 samples

```python
class SafeSMOTE(SMOTE):
    def fit_resample(self, X, y):
        # Calculate safe k_neighbors based on the smallest class size
        from collections import Counter
        class_counts = Counter(y)
        min_class_size = min(class_counts.values())
        
        # Ensure k_neighbors doesn't exceed available samples
        safe_k_neighbors = min(self.k_neighbors, max(1, min_class_size - 1))
        
        # If we have very few samples, skip SMOTE
        if min_class_size < 3:
            logger.warning(f"Too few samples in smallest class ({min_class_size}) for SMOTE, skipping oversampling")
            return X, y
        
        # Update k_neighbors if needed
        if safe_k_neighbors != self.k_neighbors:
            logger.debug(f"SMOTE: adjusting k_neighbors from {self.k_neighbors} to {safe_k_neighbors} (min_class_size={min_class_size})")
            self.k_neighbors = safe_k_neighbors
        
        return super().fit_resample(X, y)
```

## Testing Results

All fixes were tested with very small datasets:

1.  **KNNImputer**: Successfully handled 2 samples by falling back to mean imputation
2.  **Mutual Info**: Successfully handled 3 samples for both classification and regression  
3.  **SMOTE**: Successfully handled 4 samples (2 per class) by skipping oversampling
4.  **Fusion**: Successfully handled 5 samples with 2-fold CV

## Impact

### Before Fixes:
- Models failing with "Expected n_neighbors <= n_samples_fit" errors
- Complete pipeline failures for small datasets
- No models being trained successfully

### After Fixes:
- All algorithms gracefully handle small datasets
- Automatic fallback strategies when k_neighbors is too large
- Models train successfully even with 2-5 samples
- Comprehensive logging for debugging

## Configuration

The fixes respect existing configuration parameters:
- `k_neighbors` in ModalityImputer (default: 5)
- `smote_k_neighbors` in CLASS_IMBALANCE_CONFIG (default: 5)  
- `n_neighbors` in MRMR_CONFIG (default: 3)

But now automatically adjust these values when datasets are too small.

## Files Modified

1. **fusion.py**: Enhanced KNNImputer in ModalityImputer class
2. **mrmr_helper.py**: Enhanced mutual_info functions with safe n_neighbors
3. **cv.py**: Enhanced SMOTE with SafeSMOTE wrapper class

## Backward Compatibility

All fixes are backward compatible:
- Existing behavior preserved for normal-sized datasets
- Only activates safety measures for very small datasets
- No changes to public APIs or configuration options

## Summary

These fixes ensure that the machine learning pipeline can handle datasets of any size, from very small (2-5 samples) to large datasets, without encountering k_neighbors-related errors. The system now automatically adapts the neighbor parameters based on data availability while maintaining optimal performance for larger datasets. 