# Genomic Data Optimization Strategy

## Problem Analysis

The current pipeline is producing very poor performance metrics:
- **Regression**: R² scores mostly negative (-0.05 to -1.5), indicating worse than mean predictor
- **Classification**: MCC around 0 or negative, accuracy around 0.5 (random chance)

## Root Causes Identified

### 1. **Catastrophically Small Feature Sets**
- Current: N_VALUES_LIST = [4, 8, 16, 32] 
- **Problem**: Genomic data requires hundreds to thousands of features to capture biological complexity
- **Solution**: Increased to [128, 256, 512, 1024]

### 2. **Over-Aggressive Feature Selection**
- Current: Selecting only 8 features from thousands
- **Problem**: Losing all biological signal through excessive filtering
- **Solution**: Use 95% of available features, minimal variance thresholds

### 3. **Excessive Regularization**
- Current: High alpha values (0.1, 0.01) causing over-regularization
- **Problem**: Models too constrained to learn genomic patterns
- **Solution**: Minimal regularization (alpha: 0.001, 0.0001)

### 4. **Insufficient Model Complexity**
- Current: Small Random Forests (100 trees, depth 12)
- **Problem**: Insufficient capacity for genomic complexity
- **Solution**: Large models (1000 trees, unlimited depth)

## Radical Solutions Implemented

### 1. **Genomic-Scale Feature Selection**

```python
# NEW: Genomic Feature Selector
class GenomicFeatureSelector:
    - Uses ensemble methods combining multiple selection approaches
    - Biological relevance scoring
    - Stability selection with bootstrap
    - Very permissive thresholds (variance: 1e-6)
    - Default: 512 features instead of 8
```

### 2. **Minimal Regularization Models**

```python
# ElasticNet: alpha=0.001 (was 0.1)
# Lasso: alpha=0.0001 (was 0.01) 
# LogisticRegression: C=100.0 (was 1.0)
# SVM: C=100.0, epsilon=0.001
```

### 3. **High-Capacity Models**

```python
# RandomForest: 
- n_estimators=1000 (was 200)
- max_depth=None (was 12)
- min_samples_split=2 (was 5)
- min_samples_leaf=1 (was 2)
```

### 4. **Adaptive Model Configuration**

```python
def get_adaptive_regularization(n_features, n_samples):
    feature_ratio = n_features / n_samples
    if feature_ratio > 10:  # High-dimensional genomic case
        return {
            'elastic_alpha': 0.0001,
            'lasso_alpha': 0.00001,
            'C_logistic': 1000.0,
            'C_svm': 100.0
        }
```

### 5. **Biological Relevance Scoring**

```python
def _fit_biological_relevance(self, X, y, is_regression):
    # Factor 1: Variance stability
    variance_stability = 1.0 / (1.0 + np.abs(variances - np.median(variances)))
    
    # Factor 2: Non-zero expression ratio
    non_zero_ratio = np.mean(X != 0, axis=0)
    
    # Factor 3: Dynamic range
    ranges = np.ptp(X, axis=0)
    
    # Combined biological score
    bio_scores *= (1.0 + 0.2 * variance_stability)
    bio_scores *= (1.0 + 0.3 * non_zero_ratio)
    bio_scores *= (1.0 + 0.2 * normalized_ranges)
```

## Performance Targets

### Regression Targets
- **R² ≥ 0.5** (currently negative)
- **RMSE ≤ 0.5** (relative to target range)
- **MAE ≤ 0.3**

### Classification Targets  
- **MCC ≥ 0.5** (currently ~0)
- **Accuracy ≥ 0.7** (currently ~0.5)
- **AUC ≥ 0.7**
- **F1 ≥ 0.6**

## Implementation Strategy

### Phase 1: Core Configuration Changes 
- [x] Updated N_VALUES_LIST to genomic scale
- [x] Increased MAX_FEATURES to 1024
- [x] Reduced regularization across all models
- [x] Enhanced model complexity parameters

### Phase 2: Advanced Feature Selection 
- [x] Implemented GenomicFeatureSelector with ensemble methods
- [x] Added biological relevance scoring
- [x] Implemented stability selection
- [x] Very permissive variance thresholds

### Phase 3: Model Optimization 
- [x] Adaptive regularization based on data dimensions
- [x] High-capacity model configurations
- [x] Performance validation against targets

### Phase 4: Integration & Testing
- [ ] Update CLI to use new configurations
- [ ] Test on sample datasets
- [ ] Validate performance improvements
- [ ] Fine-tune parameters based on results

## Expected Performance Improvements

### Before (Current Results)
```
AML Regression:
- R²: -0.24 to -1.5 (worse than mean)
- Models failing to learn any signal

Colon Classification:  
- MCC: ~0 (random performance)
- Accuracy: ~0.5 (random chance)
```

### After (Expected Results)
```
AML Regression:
- R²: 0.3 to 0.7 (meaningful predictive power)
- RMSE: Significantly reduced
- Models capturing genomic patterns

Colon Classification:
- MCC: 0.3 to 0.6 (good discrimination)
- Accuracy: 0.65 to 0.8 (strong performance)
- AUC: 0.7+ (good ranking ability)
```

## Key Principles for Genomic Data

### 1. **More Features, Not Fewer**
- Genomic signal is distributed across many features
- Biological pathways involve hundreds of genes
- Traditional "curse of dimensionality" doesn't apply the same way

### 2. **Minimal Regularization**
- Genomic patterns are often weak and complex
- Over-regularization destroys biological signal
- Better to overfit slightly than underfit severely

### 3. **Ensemble Approaches**
- No single method captures all genomic complexity
- Combine multiple selection methods
- Weight methods by genomic relevance

### 4. **Biological Awareness**
- Consider expression patterns (non-zero ratios)
- Account for dynamic range and variance stability
- Prefer features with biological plausibility

### 5. **High Model Capacity**
- Use large Random Forests (1000+ trees)
- Allow deep trees (unlimited depth)
- Enable complex feature interactions

## Validation Metrics

### Regression Validation
```python
def validate_regression_performance(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    if r2 >= 0.5:
        return "EXCELLENT - Target achieved"
    elif r2 >= 0.3:
        return "GOOD - Meaningful signal"
    elif r2 >= 0.1:
        return "FAIR - Some signal"
    else:
        return "POOR - No meaningful signal"
```

### Classification Validation
```python
def validate_classification_performance(y_true, y_pred):
    mcc = matthews_corrcoef(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    
    if mcc >= 0.5 and accuracy >= 0.7:
        return "EXCELLENT - Target achieved"
    elif mcc >= 0.3 and accuracy >= 0.6:
        return "GOOD - Strong performance"
    elif mcc >= 0.1 and accuracy >= 0.55:
        return "FAIR - Some discrimination"
    else:
        return "POOR - Random performance"
```

## Next Steps

1. **Test the new configuration** with n_features=128 (smallest new value)
2. **Monitor performance metrics** against targets
3. **Iteratively adjust** if needed:
   - Increase feature counts further if still poor
   - Reduce regularization even more if necessary
   - Add more biological relevance factors

4. **Scale up gradually**:
   - Start with 128 features
   - Move to 256, then 512, then 1024
   - Find optimal balance for each dataset

## Success Criteria

### Minimum Acceptable Performance
- **Regression**: R² ≥ 0.3 (10x improvement from current)
- **Classification**: MCC ≥ 0.3, Accuracy ≥ 0.6

### Target Performance  
- **Regression**: R² ≥ 0.5, RMSE ≤ 0.5
- **Classification**: MCC ≥ 0.5, Accuracy ≥ 0.7

### Stretch Goals
- **Regression**: R² ≥ 0.7
- **Classification**: MCC ≥ 0.6, Accuracy ≥ 0.8

This strategy represents a fundamental shift from conservative, small-scale feature selection to genomic-appropriate, large-scale, biologically-informed modeling. 