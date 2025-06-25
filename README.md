# Multi-Omics Data Fusion Optimization Pipeline

## Project Overview

This repository contains a comprehensive machine learning pipeline for multi-omics data fusion optimization using intermediate integration techniques. This project is part of a Bachelor's Thesis in Artificial Intelligence at VU Amsterdam, contributing to a larger research initiative focused on developing advanced machine learning models for early and accurate cancer detection.

This project investigates feature *extraction* and *selection* algorithms for multi-omics cancer data. The aim is to identify the most effective methods, validate them experimentally, and design a new algorithm tailored to this data type.

### Research Context

This work is part of a broader research project aimed at creating innovative machine learning models that can detect cancer faster and more accurately in patients by leveraging multiple types of biological data. The research explores how different data integration strategies and feature extraction/selection algorithms perform when working with multi-modal omics data, which is crucial for understanding complex biological processes and disease mechanisms.

### Objectives

1. **Survey** the state-of-the-art extraction and selection algorithms used in machine learning.
2. **Select** the algorithms most suitable for multi-omics data.
3. **Evaluate** those algorithms on benchmark cancer datasets, using a comprehensive, multi-factor experiment.
4. **Analyse** the results to determine which methods perform best and **explain why**.
5. **Design** a purpose-built extraction or selection algorithm optimised for multi-omics cancer data. *(Future Work)*

### Project Purpose

The primary goal of this project is to develop a specialized feature extraction and selection algorithm specifically optimized for cancer detection machine learning models. To achieve this objective, the project conducts comprehensive research and comparative analysis of existing algorithms to identify the most effective approaches for multi-omics cancer data.

The research methodology involves systematically evaluating state-of-the-art algorithms across different parameter configurations for both classification and regression tasks:

- **Feature Extraction Algorithms**: PCA, KPLS, Factor Analysis, PLS/PLS-DA, SparsePLS
- **Feature Selection Algorithms**: ElasticNetFS, Random Forest Importance, Variance F-test, LASSO, f_regressionFS, LogisticL1, XGBoostFS
- **Machine Learning Models**: Linear/Logistic Regression, Random Forest, ElasticNet, SVM
- **Advanced Integration Strategies**: Attention-weighted fusion, learnable weighted fusion, MKL, average fusion, sum fusion, early fusion PCA
- **Missing Data Handling**: Modality-specific imputation, missing data indicators as features
- **Parameter Variations**: Different numbers of components/features (8, 16, 32) and missing data percentages (0%, 20%, 50%)

This extensive benchmarking serves as the foundation for designing a novel algorithm that leverages the strengths of existing methods while addressing the unique challenges of multi-omics cancer data integration and feature optimization.

### Data Types

The pipeline works with **multi-omics cancer data**, specifically:

- **Gene Expression Data (exp.csv)**: Transcriptomic profiles measuring mRNA expression levels
- **miRNA Data (mirna.csv)**: MicroRNA expression profiles for post-transcriptional regulation analysis  
- **Methylation Data (methy.csv)**: DNA methylation patterns indicating epigenetic modifications
- **Clinical Data**: Patient outcomes and clinical variables for supervised learning

This multi-modal approach captures different layers of biological information, providing a comprehensive view of the molecular landscape in cancer patients.

## Current Pipeline Architecture

### Feature-First Processing (Default Implementation)

The pipeline implements a **feature-first architecture** where feature extraction and selection algorithms are applied to each modality separately **before** fusion occurs. This represents the current state-of-the-art approach for multi-omics data processing:

```
Raw Multi-Omics Data
    ↓
Feature Processing (Applied to each modality separately)
    ├── Gene Expression: PCA/ElasticNet/etc. → Processed Gene Features
    ├── miRNA: PCA/ElasticNet/etc. → Processed miRNA Features  
    └── Methylation: PCA/ElasticNet/etc. → Processed Methylation Features
    ↓
Fusion (Combine processed features from all modalities)
    ↓ 
Model Training (Train on fused processed features)
```

**Key Benefits of Feature-First Architecture:**
- **Modality-Specific Optimization**: Each data type gets specialized preprocessing
- **Better Feature Quality**: Fusion works with clean, optimized features
- **Improved Interpretability**: Clear understanding of individual modality contributions
- **Computational Efficiency**: Parallel processing of modalities possible

### Fusion Workflow in Feature-First Architecture

In the feature-first approach, fusion techniques operate on already-processed features:

```
Input: Raw Multi-Omics Data
    ├── Gene Expression (20,000+ features)
    ├── miRNA (2,000+ features)  
    └── Methylation (25,000+ features)
    
Step 1: Feature Processing (Applied separately to each modality)
    ├── Gene Expression → PCA/ElasticNet → 32 processed features
    ├── miRNA → PCA/ElasticNet → 32 processed features
    └── Methylation → PCA/ElasticNet → 32 processed features
    
Step 2: Fusion (Combine processed features)
    Input: [gene_32_features, mirna_32_features, methy_32_features]
    ↓
    Apply Fusion Technique:
    ├── Average: (gene + mirna + methy) / 3 → 32 fused features
    ├── Sum: gene + mirna + methy → 32 fused features  
    ├── Attention: weighted combination → 32 fused features
    ├── MKL: kernel-based combination → 32 fused features
    ├── Learnable: performance-weighted → 32 fused features
    ├── Standard Concat: [gene | mirna | methy] → 96 fused features
    └── Early PCA: concatenate then PCA → configurable fused features
    
Step 3: Model Training
    Input: Fused features (typically 32-64 dimensions)
    ↓
    Train: LinearRegression/RandomForest/ElasticNet
```

This approach ensures that fusion techniques work with:
- **Clean, optimized features** rather than noisy raw data
- **Consistent dimensionality** across modalities (e.g., 32 features each)
- **Biologically meaningful representations** from each modality
- **Computationally manageable** feature spaces

## Current Pipeline Features

### 🔬 **4-Phase Enhanced Pipeline Architecture**
- **Phase 1 - Early Data Quality Assessment**: Comprehensive data validation and quality scoring
- **Phase 2 - Feature-First Processing**: Feature extraction/selection applied to each modality separately before fusion
- **Phase 3 - Centralized Missing Data Management**: Intelligent imputation and missing data handling
- **Phase 4 - Coordinated Validation**: Enhanced cross-validation with numerical stability

### 🧩 **Missing Data-Adaptive Fusion Strategies**
- **Clean Data (0% missing)**: 7 advanced methods - attention_weighted, learnable_weighted, mkl, average, sum, standard_concat, early_fusion_pca
- **Missing Data (>0% missing)**: 4 robust methods - mkl, average, sum, early_fusion_pca
- **Attention-Weighted Fusion**: Neural attention mechanisms for sample-specific weighting
- **Multiple Kernel Learning (MKL)**: RBF kernel-based fusion with automatic kernel weighting
- **Average Fusion**: Simple averaging of modality features for robust baseline fusion
- **Sum Fusion**: Simple summation of modality features for additive combination

### 🧬 **Enhanced Modality-Specific Preprocessing**
- **Gene Expression**: Robust biomedical preprocessing with log transformation and robust scaling
- **miRNA**: Advanced sparsity handling (>90% zeros), biological KNN imputation, zero-inflation modeling
- **Methylation**: Conservative preprocessing with mean imputation and outlier capping
- **Cross-Modality Features**: Data orientation validation, numerical stability checks, adaptive MAD thresholds

### 📊 **Comprehensive Data Quality Analysis**
- **Quality Scoring**: Automated assessment of data quality with detailed reporting
- **Missing Pattern Analysis**: Intelligent detection and handling of missing data patterns
- **Numerical Stability**: Automatic removal of problematic features that cause NaN/inf values
- **Preprocessing Guidance**: Data-driven recommendations for optimal preprocessing strategies

### 🎯 **Task-Appropriate Cross-Validation Strategies**
- **Regression CV**: Adaptive strategies based on dataset size
  - Small datasets (<100 samples): `RepeatedKFold(2-3 splits, 5 repeats)`
  - Medium datasets (100-200 samples): `KFold(3-5 splits)`
  - Large datasets (>200 samples): `KFold(5 splits)`
  - With patient groups: `GroupKFold`
- **Classification CV**: Proper stratified approaches
  - Standard: `StratifiedKFold` for balanced class distribution
  - With patient groups: `StratifiedGroupKFold`
  - Fallbacks: `KFold/GroupKFold` when stratification not viable

### 🛡️ **Enhanced Algorithm Robustness**
- **ElasticNet Optimization**: Small dataset detection with fallback strategies
  - Small datasets (<200 samples): Fixed alpha ElasticNet with StandardScaler
  - Large datasets: ElasticNetCV with PowerTransformer
  - Multi-level fallbacks: ElasticNetCV → ElasticNet → LinearRegression
- **Numerical Stability**: Safe attribute access with robust error handling
- **Cross-Validation Compatibility**: Proper sklearn usage without forced adaptations

### ⚡ **Performance Optimizations**
- **Enhanced Feature Selection**: MAD thresholds (0.05), correlation removal (0.90), sparsity filtering (0.9)
- **Stricter Regularization**: ElasticNet alpha range (0.1-0.5) for better generalization
- **Numerical Stability**: Automatic detection and removal of problematic features
- **Memory Optimization**: Intelligent caching and parallel processing

## Experimental Design

> **Full code and preliminary results are available in the GitHub repository**
> **OUH-Internship-Krzysztof-Nowak**.

The pipeline systematically evaluates all combinations of algorithms and parameters using the **feature-first architecture** with the following enhanced experimental structure:

```python
# Regression branch algorithms (CURRENT IMPLEMENTATION)
REGRESSION_EXTRACTORS = [PCA, KPCA, FA, PLS, KPLS, SparsePLS]
REGRESSION_SELECTORS = [ElasticNetFS, RFImportance, VarianceFTest, LASSO, f_regressionFS]
REGRESSION_MODELS = [LinearRegression, ElasticNet, RandomForestRegressor]

# Classification branch algorithms (CURRENT IMPLEMENTATION)
CLASSIFICATION_EXTRACTORS = [PCA, KPCA, FA, LDA, PLS-DA, SparsePLS]
CLASSIFICATION_SELECTORS = [ElasticNetFS, RFImportance, VarianceFTest, LASSO, LogisticL1]
CLASSIFICATION_MODELS = [LogisticRegression, RandomForestClassifier, SVC]

# Missing data-adaptive fusion strategies (CURRENT IMPLEMENTATION)
FUSION_STRATEGIES_CLEAN_DATA = {
    'attention_weighted': 'Sample-specific attention weighting',
    'learnable_weighted': 'Performance-based modality weighting', 
    'mkl': 'Multiple Kernel Learning with RBF kernels',
    'average': 'Simple averaging for robust baseline fusion',
    'sum': 'Simple summation for additive combination',
    'standard_concat': 'Standard concatenation of processed features',
    'early_fusion_pca': 'PCA-based early integration'
}

FUSION_STRATEGIES_MISSING_DATA = {
    'mkl': 'Multiple Kernel Learning (robust to missing data)',
    'average': 'Simple averaging (handles missing data)',
    'sum': 'Simple summation (handles missing data)',
    'early_fusion_pca': 'PCA-based early integration (robust)'
}

# CURRENT Experimental loop for each dataset - FEATURE PROCESSING FIRST, then Fusion
for MISSING in [0, 0.20, 0.50]:  # Missing data scenarios first
    # STEP 1: Select fusion strategy based on missing data percentage
    if MISSING == 0:
        INTEGRATIONS = [attention_weighted, learnable_weighted, 
                       mkl, average, sum, standard_concat, early_fusion_pca]  # 7 methods for clean data
    else:  # missing data scenarios
        INTEGRATIONS = [mkl, average, sum, early_fusion_pca]  # 4 robust methods for missing data
            
    for ALGORITHM in EXTRACTORS + SELECTORS:  # Apply feature processing to raw modalities FIRST
        for N_FEATURES in [8, 16, 32]:  # For selection methods only
            for INTEGRATION in INTEGRATIONS:  # Then apply fusion to processed features SECOND
                for MODEL in TASK_SPECIFIC_MODELS:
                    run_experiment(
                        # CURRENT ORDER: Feature Processing → Fusion → Model Training
                        missing_rate=MISSING,           # 1. Missing data scenario
                        algorithm=ALGORITHM,            # 2. Feature processing applied to raw modalities FIRST
                        n_features=N_FEATURES if ALGORITHM in SELECTORS else None,  # Fixed for selectors
                        n_components=None if ALGORITHM in SELECTORS else "optimized",  # Tuned for extractors
                        integration=INTEGRATION,        # 3. Fusion applied to processed features SECOND
                        model=MODEL,                    # 4. Model training on fused processed features
                        # Pipeline configuration
                        enable_early_quality_check=True,
                        enable_feature_first_order=True,  # Feature processing applied to raw modalities first
                        enable_centralized_missing_data=True,
                        enable_coordinated_validation=True
                    )
```

### Current Experimental Features:
- **Missing Data-Based Strategy Selection**: 7 fusion methods for clean data (0% missing); 4 robust methods for missing data scenarios
- **Feature-First Pipeline Order**: Feature processing applied FIRST to raw modalities, then fusion applied to processed features
- **4-Phase Pipeline Integration**: Early quality assessment, feature-first processing, missing data management, coordinated validation
- **Enhanced Data Quality Analysis**: Comprehensive data quality assessment with automated reporting
- **Modality-Specific Processing**: Tailored preprocessing configurations for gene expression, miRNA, and methylation data
- **Robust Cross-Validation**: Enhanced CV with patient-level grouping and numerical stability checks
- **Memory & Performance Monitoring**: Real-time resource tracking and intelligent caching

This comprehensive experimental design ensures systematic evaluation across:
- **Feature extraction/selection algorithms** (6 extractors + 5 selectors for regression; 6 extractors + 5 selectors for classification)
- **Feature/component optimization**: 
  - **Selection methods**: Fixed at 8, 16, 32 features for systematic comparison
  - **Extraction methods**: Optimal number of components determined through hyperparameter tuning
- **Missing data scenarios** (0%, 20%, 50% missing modalities)
- **Missing data-adaptive integration strategies** (7 methods for clean data, 4 methods for missing data scenarios)
- **Feature-First Pipeline Architecture**: Feature Processing → Fusion → Model Training (current optimal order for multi-modal genomics)
- **Predictive models** with hyperparameter optimization and numerical stability

## Deliverables

1. **Literature review** summarising extraction and selection methods for multi-omics data.
2. **Experimental report** (methods, code links, and results tables/plots) highlighting the top-performing algorithm combinations and explaining their success.
3. **Recommendation** of the algorithms most suitable for multi-omics cancer studies, with justification.
4. **New algorithm** specifically designed and empirically validated for this data. *(Future Work)*

## Algorithm Architecture

### Enhanced Pipeline Workflow

1. **Phase 1 - Early Data Quality Assessment**: 
   - Automated data quality evaluation with comprehensive scoring
   - Data orientation validation (samples × features) for genomic data
   - Sample ID standardization and alignment across modalities
   - Quality-based preprocessing guidance and strategy recommendations

2. **Phase 2 - Feature-First Processing** (CURRENT ARCHITECTURE):
   - Feature extraction/selection applied to each raw modality separately BEFORE fusion
   - Modality-specific preprocessing configurations (gene expression, miRNA, methylation)
   - Numerical stability checks with automatic problematic feature removal
   - Robust biomedical preprocessing pipeline with enhanced sparsity handling

3. **Phase 3 - Centralized Missing Data Management**:
   - Intelligent missing data pattern analysis and strategy selection
   - Adaptive imputation methods based on data characteristics and modality type
   - Missing modality simulation for robustness testing (0%, 20%, 50%)
   - Cross-validation compatible missing data handling

4. **Phase 4 - Coordinated Validation Framework**:
   - **Task-Appropriate CV Strategies**: Proper regression (KFold/RepeatedKFold) vs classification (StratifiedKFold) approaches
   - **Adaptive CV Selection**: Dataset size-based strategy selection for optimal stability
   - **Enhanced Patient Grouping**: GroupKFold and StratifiedGroupKFold for patient-level validation
   - **Numerical Stability**: Comprehensive validation throughout the pipeline
   - **Sample Alignment**: Verification across processing steps with robust error handling

5. **Feature Extraction/Selection** (Applied FIRST - to Raw Modalities):
   - **Extraction Pipeline**: Dimensionality reduction (PCA, KPCA, PLS, KPLS, SparsePLS, Factor Analysis) applied to each modality separately
   - **Selection Pipeline**: Feature selection (ElasticNetFS, Random Forest Importance, Variance F-test, LASSO) applied to each modality separately
   - **Modality-Specific Processing**: Each modality processed independently with optimal parameters
   - **Intelligent Caching**: Cached results for expensive extraction/selection operations

6. **Multi-Modal Fusion** (Applied SECOND - to Processed Features):
   - **Clean Data (0% missing)**: 7 fusion methods tested - attention_weighted, learnable_weighted, mkl, average, sum, standard_concat, early_fusion_pca
   - **Missing Data (>0% missing)**: 4 robust fusion methods - mkl, average, sum, early_fusion_pca  
   - **Strategy Selection**: Automatic based on missing data percentage, not task type
   - **Processed Feature Fusion**: Fusion applied to already-processed features from each modality
   - **Robust Fallbacks**: Graceful degradation when advanced fusion methods fail

7. **Model Training & Evaluation**:
   - **Task-Appropriate Cross-Validation**: Regression uses KFold/RepeatedKFold, classification uses StratifiedKFold
   - **Adaptive CV Strategy**: Dataset size-based selection for optimal stability and reliability
   - **Enhanced Patient Grouping**: GroupKFold and StratifiedGroupKFold for patient-level validation
   - **Hyperparameter Optimization**: Pre-tuned parameters from `hp_best/` including optimal component counts for extraction methods
   - **Systematic Feature Evaluation**: Fixed feature counts (8, 16, 32) for selection methods to enable fair comparison
   - **Algorithm Robustness**: Multi-level fallbacks (ElasticNetCV → ElasticNet → LinearRegression) for numerical stability
   - Comprehensive evaluation metrics with enhanced AUC calculation for imbalanced datasets

8. **Results Analysis & Visualization**:
   - Automated generation of performance plots (scatter, residuals, ROC, confusion matrices)
   - Algorithm ranking and performance comparison across all experimental conditions
   - Statistical significance testing with critical difference analysis
   - Comprehensive results storage in `final_results/` with detailed metrics

### Supported Algorithms

#### Feature Extraction Methods (CURRENT IMPLEMENTATION)

**Regression Extractors (6)**: PCA, KPCA, Factor Analysis, PLS, KPLS, SparsePLS
**Classification Extractors (6)**: PCA, KPCA, Factor Analysis, LDA, PLS-DA, SparsePLS

- **PCA (Principal Component Analysis)**: Linear dimensionality reduction with variance maximization
- **KPCA (Kernel PCA)**: Non-linear dimensionality reduction with RBF kernel and median heuristic
- **Factor Analysis**: Latent factor modeling for hidden structure discovery
- **PLS (Partial Least Squares)**: Supervised dimensionality reduction for regression tasks
- **KPLS (Kernel PLS)**: Non-linear kernel-based partial least squares with cross-validation optimization (regression only)
- **LDA (Linear Discriminant Analysis)**: Supervised dimensionality reduction for classification tasks (classification only)
- **PLS-DA (PLS Discriminant Analysis)**: Supervised dimensionality reduction for classification tasks
- **SparsePLS**: Sparse partial least squares with automatic sparsity selection

#### Feature Selection Methods (CURRENT IMPLEMENTATION)

**Regression Selectors (5)**: ElasticNetFS, RFImportance, VarianceFTest, LASSO, f_regressionFS
**Classification Selectors (5)**: ElasticNetFS, RFImportance, VarianceFTest, LASSO, LogisticL1

- **ElasticNetFS**: ElasticNet-based feature selection with L1/L2 regularization and cross-validation
- **RFImportance (Random Forest Importance)**: Tree-based feature importance ranking with ensemble voting
- **VarianceFTest**: Variance-based F-test feature selection for statistical significance
- **LASSO**: L1-regularized linear model with automatic regularization parameter selection
- **f_regressionFS**: F-test based regression feature selection with statistical validation (regression only)
- **LogisticL1**: L1-regularized logistic regression for classification feature selection (classification only)

#### Data Fusion Methods (CURRENT IMPLEMENTATION)

**Clean Data Fusion (0% missing) - 7 methods tested:**
- **Attention-Weighted Fusion**: Neural attention mechanisms for sample-specific modality weighting
- **Learnable Weighted Fusion**: Cross-validation based performance weighting of modalities  
- **MKL (Multiple Kernel Learning)**: RBF kernel-based fusion with automatic kernel parameter optimization
- **Average Fusion**: Simple averaging of modality features for robust baseline fusion
- **Sum Fusion**: Simple summation of modality features for additive combination
- **Standard Concatenation**: Direct concatenation of processed features without dimensionality reduction
- **Early Fusion PCA**: Dimensionality reduction applied to concatenated processed features

**Missing Data Fusion (>0% missing) - 4 robust methods tested:**
- **MKL (Multiple Kernel Learning)**: Robust to missing data with kernel approximation
- **Average Fusion**: Simple averaging that handles missing data gracefully
- **Sum Fusion**: Simple summation that handles missing data gracefully
- **Early Fusion PCA**: Simple and robust concatenation with PCA dimensionality reduction

**Note**: In the current feature-first architecture, all fusion methods work on already-processed features from each modality, not on raw data.

#### Detailed Fusion Technique Explanations

##### 1. **Attention-Weighted Fusion**
**How it works**: Uses neural attention mechanisms to learn sample-specific weights for each modality. For each sample, the attention network analyzes the feature patterns and assigns different importance weights to each modality.

**Technical details**: 
- Implements a multi-layer perceptron (MLP) with configurable hidden dimensions (default: 32)
- Uses dropout (default: 0.3) for regularization
- Optimized with Adam optimizer (learning rate: 0.001)
- Training with early stopping (patience: 10 epochs, max: 100 epochs)

**When to use**: 
- When modalities have varying importance across different samples
- For complex datasets where simple averaging is insufficient
- When you have sufficient training data to learn attention patterns

**Advantages**: Adaptive, learns from data, can capture complex modality interactions
**Disadvantages**: Requires more computational resources, needs sufficient training data

##### 2. **Learnable Weighted Fusion**
**How it works**: Determines optimal static weights for each modality through cross-validation performance evaluation. Each modality gets a fixed weight based on its contribution to predictive performance.

**Technical details**:
- Uses 3-fold cross-validation by default (adaptive based on dataset size)
- Evaluates each modality individually using the target model
- Assigns weights proportional to individual modality performance
- Supports both regression (R²) and classification (AUC) scoring

**When to use**:
- When modalities have consistently different predictive power
- For interpretable fusion where you want to understand modality importance
- When computational efficiency is important

**Advantages**: Interpretable weights, computationally efficient, stable performance
**Disadvantages**: Static weights across all samples, requires cross-validation overhead

##### 3. **Multiple Kernel Learning (MKL)**
**How it works**: Creates separate RBF (Radial Basis Function) kernels for each modality and learns optimal kernel weights. Combines kernel similarities rather than raw features.

**Technical details**:
- Uses RBF kernels with gamma parameter optimization
- Implements kernel weight learning through performance-based optimization
- Falls back to kernel approximation when full kernel computation is expensive
- Handles missing modalities through kernel interpolation

**When to use**:
- For non-linear relationships between modalities
- When modalities have different scales and distributions
- For robust handling of missing data scenarios

**Advantages**: Handles non-linear relationships, robust to missing data, theoretically grounded
**Disadvantages**: Computationally intensive, requires kernel parameter tuning

##### 4. **Average Fusion**
**How it works**: Simple element-wise averaging of processed features from all modalities. Each modality contributes equally to the final representation.

**Technical details**:
- Computes element-wise mean: `fused_features = (mod1_features + mod2_features + mod3_features) / 3`
- Handles missing modalities by averaging only available modalities
- No parameters to tune, deterministic output
- Memory efficient implementation

**When to use**:
- As a robust baseline for comparison
- When all modalities are equally important
- For quick prototyping and testing
- When interpretability is crucial

**Advantages**: Simple, interpretable, robust, no hyperparameters, fast computation
**Disadvantages**: Assumes equal modality importance, may not capture complex relationships

##### 5. **Sum Fusion**
**How it works**: Simple element-wise summation of processed features from all modalities. Creates an additive combination where features accumulate.

**Technical details**:
- Computes element-wise sum: `fused_features = mod1_features + mod2_features + mod3_features`
- Handles missing modalities by summing only available modalities
- Preserves feature magnitude (unlike averaging)
- No normalization applied

**When to use**:
- When features represent counts or additive quantities
- For preserving the absolute magnitude of feature contributions
- As an alternative baseline to averaging
- When modality contributions should accumulate

**Advantages**: Preserves feature magnitude, simple implementation, fast computation
**Disadvantages**: Can amplify noise, sensitive to feature scaling, assumes additive relationships

##### 6. **Standard Concatenation**
**How it works**: Direct horizontal concatenation of processed features from all modalities without any transformation or dimensionality reduction. Creates a unified high-dimensional representation preserving all information.

**Technical details**:
- Concatenates features: `[mod1_features | mod2_features | mod3_features]`
- No transformation applied, preserves all processed features
- Output dimensionality = sum of input modality dimensions
- Memory efficient implementation using numpy/pandas concatenation
- Maintains original feature interpretability

**When to use**:
- When you want to preserve all processed feature information
- For downstream algorithms that can handle higher dimensions
- When interpretability of individual modality features is important
- As a baseline before applying dimensionality reduction

**Advantages**: Preserves all information, maintains interpretability, simple implementation, no information loss
**Disadvantages**: High dimensionality, potential curse of dimensionality, no modality weighting

##### 7. **Early Fusion PCA**
**How it works**: Concatenates processed features from all modalities and applies Principal Component Analysis for dimensionality reduction. Creates a unified low-dimensional representation.

**Technical details**:
- Concatenates features: `[mod1_features | mod2_features | mod3_features]`
- Applies PCA with configurable number of components (default: optimized based on explained variance)
- Uses scikit-learn's PCA implementation with SVD solver
- Handles missing modalities by imputing before concatenation

**When to use**:
- When you need dimensionality reduction of the fused representation
- For capturing global patterns across all modalities
- When memory constraints require lower-dimensional features
- As a preprocessing step for downstream algorithms

**Advantages**: Reduces dimensionality, captures global variance, computationally efficient
**Disadvantages**: Linear transformation only, may lose modality-specific patterns, requires complete data

#### Mathematical Formulations

For mathematical clarity, let's define the fusion operations where we have M modalities with processed features X₁, X₂, ..., Xₘ ∈ ℝⁿˣᵈ (n samples, d features each):

**Average Fusion:**
```
F_avg = (1/M) × Σᵢ₌₁ᴹ Xᵢ
```

**Sum Fusion:**
```
F_sum = Σᵢ₌₁ᴹ Xᵢ
```

**Learnable Weighted Fusion:**
```
F_weighted = Σᵢ₌₁ᴹ wᵢ × Xᵢ
where wᵢ = performance_score(Xᵢ, y) / Σⱼ performance_score(Xⱼ, y)
```

**Attention-Weighted Fusion:**
```
F_attention = Σᵢ₌₁ᴹ αᵢ(x) × Xᵢ
where αᵢ(x) = softmax(MLP([X₁, X₂, ..., Xₘ]))ᵢ
```

**MKL Fusion:**
```
F_mkl = Σᵢ₌₁ᴹ βᵢ × K(Xᵢ, Xᵢ)
where K(Xᵢ, Xᵢ) is the RBF kernel matrix and βᵢ are learned kernel weights
```

**Standard Concatenation:**
```
F_concat = [X₁ | X₂ | ... | Xₘ]
where [X₁ | X₂ | ... | Xₘ] represents horizontal concatenation
```

**Early Fusion PCA:**
```
F_pca = PCA([X₁ | X₂ | ... | Xₘ])
where [X₁ | X₂ | ... | Xₘ] represents horizontal concatenation
```

#### Fusion Strategy Selection Guide

| Fusion Method | Complexity | Performance | Interpretability | Missing Data | Computational Cost |
|---------------|------------|-------------|------------------|--------------|-------------------|
| **Average** | Low | Good baseline | High | Excellent | Very Low |
| **Sum** | Low | Good baseline | High | Excellent | Very Low |
| **Standard Concat** | Low | Good | High | Poor | Very Low |
| **Learnable Weighted** | Medium | Good | Medium | Good | Medium |
| **Early Fusion PCA** | Medium | Good | Medium | Fair | Medium |
| **MKL** | High | Excellent | Low | Excellent | High |
| **Attention-Weighted** | High | Excellent | Low | Good | High |

#### Practical Fusion Selection Guidelines

**For beginners and baseline comparisons:**
- Start with **Average Fusion** - simple, robust, and interpretable
- Use **Sum Fusion** as an alternative baseline
- Try **Standard Concatenation** when you want to preserve all feature information
- These provide solid performance benchmarks with zero hyperparameter tuning

**For improved performance with moderate complexity:**
- Use **Learnable Weighted Fusion** when modalities have different predictive power
- Use **Early Fusion PCA** when dimensionality reduction is beneficial
- Both offer good performance improvements over simple baselines

**For maximum performance with advanced techniques:**
- Use **Attention-Weighted Fusion** for complex datasets with sample-varying modality importance
- Use **MKL** for non-linear relationships and robust missing data handling
- These require more computational resources but often achieve best performance

**For missing data scenarios:**
- **Best**: MKL (excellent missing data handling)
- **Good**: Average, Sum (simple and robust)
- **Fair**: Early Fusion PCA (requires imputation)
- **Avoid**: Standard Concatenation, Attention-Weighted (poor missing data handling)

**For computational constraints:**
- **Fastest**: Average, Sum, Standard Concatenation (milliseconds)
- **Medium**: Learnable Weighted, Early Fusion PCA (seconds)
- **Slowest**: MKL, Attention-Weighted (minutes for large datasets)

#### Machine Learning Models
- **Regression**: Linear Regression, ElasticNet, Random Forest Regressor (with pre-tuned hyperparameters)
- **Classification**: Logistic Regression, Random Forest Classifier (with pre-tuned hyperparameters)

## Datasets

The pipeline includes multiple cancer datasets from The Cancer Genome Atlas (TCGA):

### Regression Tasks
- **AML (Acute Myeloid Leukemia)**: Predicting blast cell percentage
- **Sarcoma**: Predicting tumor length

### Classification Tasks
- **Breast, Colon, Kidney, Liver, Lung, Melanoma** and **Ovarian** datasets for pathologic T-stage and clinical stage classification

All datasets originate from:
Rappoport & Shamir (2018), *Multi-omic and multi-view clustering algorithms: review and cancer benchmark*, **Nucleic Acids Research**, 46 (20), 10546–10562.
Download link: [https://acgt.cs.tau.ac.il/multi_omic_benchmark/download.html](https://acgt.cs.tau.ac.il/multi_omic_benchmark/download.html)

### Data Structure
Each dataset contains:
```
data/
├── {cancer_type}/
│   ├── exp.csv          # Gene expression data
│   ├── mirna.csv        # miRNA expression data
│   └── methy.csv        # Methylation data
└── clinical/
    └── {cancer_type}.csv # Clinical outcomes
```

## Usage

### Basic Execution

Run the complete pipeline with all datasets and algorithms:

#### Feature-First Architecture (DEFAULT - CURRENT IMPLEMENTATION)
```bash
# Use the current Feature Processing → Fusion → Model Training order
python main.py
```

#### Fusion-First Architecture (LEGACY)
```bash
# Use the legacy Fusion → Feature Processing → Model Training order (for comparison/research)
python main.py --fusion-first
```

### Execution Options

#### Dataset-Specific Execution
```bash
# Run only regression datasets (AML, Sarcoma)
python main.py --regression-only

# Run only classification datasets (Breast, Colon, etc.)
python main.py --classification-only

# Run a specific dataset
python main.py --dataset AML
python main.py --dataset Breast
```

#### Analysis Type Control
```bash
# Run only MAD analysis (no model training)
python main.py --mad-only

# Skip MAD analysis (only model training)
python main.py --skip-mad

# Run everything (default behavior)
python main.py
```

#### Parameter Control
```bash
# Run with specific number of components/features
python main.py --n-val 8   # Only 8 components/features
python main.py --n-val 16  # Only 16 components/features
python main.py --n-val 32  # Only 32 components/features
```

#### Logging Control
```bash
# Debug mode (most verbose)
python main.py --debug

# Verbose mode (detailed information)
python main.py --verbose

# Warning mode (default - minimal output)
python main.py
```

#### Combined Options
```bash
# Example: Run only AML dataset with debug logging (feature-first by default)
python main.py --dataset AML --debug --n-val 16

# Example: Run only classification with verbose logging, skip MAD (feature-first by default)
python main.py --classification-only --verbose --skip-mad

# Example: Compare both architectures on the same dataset
python main.py --dataset Breast                # Feature-first (current implementation)
python main.py --dataset Breast --fusion-first # Fusion-first (legacy for comparison)
```

### Advanced Configuration

For advanced users, you can modify the configuration in `config.py`:

#### Core Pipeline Settings
- **Missing data percentages**: Modify `MISSING_MODALITIES_CONFIG["missing_percentages"]`
- **Algorithm selection**: Enable/disable algorithms in `get_*_extractors()` and `get_*_selectors()` functions
- **Model parameters**: Adjust `MODEL_OPTIMIZATIONS` dictionary
- **Memory settings**: Modify `MEMORY_OPTIMIZATION` and `CACHE_CONFIG`

#### Enhanced Preprocessing Configuration
- **Modality-Specific Settings**: Customize `ENHANCED_PREPROCESSING_CONFIGS` for each data type:
  ```python
  # Example: miRNA-specific configuration
  "miRNA": {
      "enhanced_sparsity_handling": True,
      "sparsity_threshold": 0.9,
      "use_biological_knn_imputation": True,
      "knn_neighbors": 5,
      "zero_inflation_handling": True,
      "mad_threshold": 1e-8
  }
  ```

#### Missing Data Intelligence
- **Missing Data Indicators**: Configure `PREPROCESSING_CONFIG`:
  ```python
  "add_missing_indicators": True,
  "missing_indicator_threshold": 0.05,  # 5% missing threshold
  "missing_indicator_prefix": "missing_",
  "missing_indicator_sparse": True
  ```

#### Fusion Strategy Settings
- **Attention Fusion**: Customize `FUSION_UPGRADES_CONFIG["attention_weighted"]`:
  ```python
  "hidden_dim": 32,
  "dropout_rate": 0.3,
  "learning_rate": 0.001,
  "max_epochs": 100
  ```

#### Feature Selection Optimization
- **Enhanced Thresholds**: Adjust feature selection parameters:
  ```python
  "correlation_threshold": 0.90,  # More aggressive correlation removal
  "mad_threshold": 0.05,          # Stricter MAD filtering  
  "sparsity_threshold": 0.9       # Higher sparsity removal
  ```

## Installation

### Prerequisites

- **Python 3.8 or higher**
- **pip** (Python package installer)
- **Git** (for cloning the repository)

### Quick Installation

1. **Clone the repository**:
```bash
git clone https://github.com/kpnowak/OUH-Internship-Krzysztof-Nowak.git
cd OUH-Internship-Krzysztof-Nowak
```

2. **Install dependencies**:

#### Option A: Convenience Script (Recommended)
```bash
python install.py
```
This interactive script will guide you through the installation process and automatically run tests.

#### Option B: Manual Installation
```bash
# Install core dependencies
pip install -r setup_and_info/requirements.txt

# Or install in development mode (recommended)
cd setup_and_info
pip install -e .
```

### Installation Options

#### Basic Installation (Core Dependencies Only)
```bash
cd setup_and_info
pip install -e .
```

This installs the essential dependencies:
- numpy (≥1.21.0) - Numerical computing
- pandas (≥1.3.0) - Data manipulation
- scipy (≥1.7.0) - Scientific computing
- scikit-learn (≥1.0.0) - Machine learning algorithms
- matplotlib (≥3.5.0) - Plotting
- seaborn (≥0.11.0) - Statistical visualization
- joblib (≥1.1.0) - Parallel processing
- threadpoolctl (≥3.0.0) - Thread control
- psutil (≥5.8.0) - System monitoring
- boruta (≥0.3.0) - Feature selection

#### Installation with Visualization Support
```bash
cd setup_and_info
pip install -e ".[visualization]"
```

Adds enhanced visualization capabilities:
- scikit-posthocs (≥0.6.0) - Critical difference diagrams for MAD analysis

#### Development Installation
```bash
cd setup_and_info
pip install -e ".[development]"
```

Includes development tools:
- pytest (≥6.0.0) - Testing framework
- pytest-cov (≥2.12.0) - Coverage reporting
- black (≥21.0.0) - Code formatting
- flake8 (≥3.9.0) - Linting
- mypy (≥0.910) - Type checking

#### Full Installation
```bash
cd setup_and_info
pip install -e ".[all]"
```

Installs all optional dependencies.

### Alternative Installation Methods

#### Using requirements.txt
```bash
# Core dependencies
pip install -r setup_and_info/requirements.txt

# Development dependencies (includes core + dev tools)
pip install -r setup_and_info/requirements-dev.txt
```

#### Using conda
```bash
# Create conda environment
conda create -n data_fusion python=3.9
conda activate data_fusion

# Install dependencies
pip install -r setup_and_info/requirements.txt
cd setup_and_info
pip install -e .
```

### Installation Verification

Run the comprehensive installation test:
```bash
python setup_and_info/test_installation.py
```

This script verifies:
-  Python version compatibility (3.8+)
-  All core dependencies
-  module imports
-  Basic functionality
-  Command-line interface
-  Optional dependencies (warnings if missing)

#### Quick Verification
```bash
# Test basic functionality
python -c "print('Multi-omics pipeline installed successfully!')"

# Test CLI
python main.py --help

# Test MAD analysis (if visualization dependencies installed)
python main.py --mad-only
```

### Troubleshooting

#### Common Issues

1. **Python Version Error**:
   ```bash
   # Check Python version
   python --version
   # Should be 3.8 or higher
   ```

2. **Missing Dependencies**:
   ```bash
   # Reinstall with verbose output
   cd setup_and_info
   pip install -e . -v
   ```

3. **Permission Errors**:
   ```bash
   # Install in user directory
   cd setup_and_info
   pip install -e . --user
   ```

4. **Memory Issues**:
   - Reduce `CACHE_CONFIG["total_limit_mb"]` in `config.py`
   - Use `--n-val 8` for smaller parameter space

5. **Cross-Validation Warnings**:
   - The pipeline now uses proper CV strategies (no more StratifiedKFold warnings for regression)
   - ElasticNet errors are handled with automatic fallbacks

6. **Small Dataset Issues**:
   - Automatic detection and optimization for datasets <200 samples
   - Enhanced numerical stability with appropriate preprocessing

#### Getting Help

- Check the installation test output: `python setup_and_info/test_installation.py`
- Review log files: `debug.log` (created during execution)
- Enable debug mode: `python main.py --debug`
- Recent improvements address most sklearn compatibility warnings

## Output Structure

The pipeline generates comprehensive outputs:

```
output/
├── {dataset_name}/
│   ├── metrics/
│   │   ├── {dataset}_extraction_cv_metrics.csv
│   │   ├── {dataset}_selection_cv_metrics.csv
│   │   ├── {dataset}_extraction_best_fold_metrics.csv
│   │   ├── {dataset}_selection_best_fold_metrics.csv
│   │   └── {dataset}_combined_best_fold_metrics.csv
│   ├── models/
│   │   └── best_model_*.pkl
│   └── plots/
│       ├── *_scatter.png
│       ├── *_residuals.png
│       ├── *_confusion.png
│       ├── *_roc.png
│       └── *_featimp.png
└── mad_analysis/
    ├── mad_metrics.csv
    ├── critical_difference_*.png
    └── statistics_table.csv
```

## Performance Considerations

### System Requirements
- **Memory Usage**: Optimized for high-memory systems (8GB+ RAM recommended, 16GB+ for large datasets)
- **CPU**: Multi-core processing with parallel feature extraction/selection
- **Storage**: SSD recommended for faster data I/O and caching

### Performance Optimizations
- **Intelligent Caching**: Feature extraction/selection results cached to avoid redundant computations
- **Parallel Processing**: Utilizes multiple CPU cores for cross-validation and algorithm evaluation
- **Memory Management**: Automatic cache clearing and memory monitoring (60GB RAM systems supported)
- **Early Stopping**: Prevents overfitting in neural networks and iterative algorithms

### Enhanced Efficiency Features
- **Task-Appropriate CV**: Proper regression/classification strategies eliminate warnings and improve stability
- **Adaptive Algorithm Selection**: Automatic detection of small datasets (<200 samples) for optimal preprocessing
- **Numerical Stability**: Automatic detection and removal of problematic features reduces computational overhead
- **Robust Error Handling**: Multi-level fallbacks prevent pipeline failures from algorithm-specific issues
- **Adaptive Preprocessing**: Modality-specific optimizations reduce processing time
- **Sample Alignment**: Robust handling of dimension mismatches prevents pipeline failures
- **Sparse Data Optimization**: Efficient handling of high-sparsity genomic data (>90% zeros)

### Computational Complexity
- **Feature Selection**: O(n_features × n_algorithms × k_folds) with intelligent pruning
- **Fusion Methods**: O(n_modalities × n_samples × fusion_complexity)
- **Missing Data Indicators**: O(n_features × missing_threshold) with sparse representation
- **Cross-Validation**: Parallelized across folds and algorithms for optimal throughput

## Repository Structure

```
OUH-Internship-Krzysztof-Nowak/
├── install.py                          # Convenience installation script
├── main.py                             # Main pipeline entry point
├── cli.py                              # Command-line interface
├── config.py                           # Configuration settings and dataset definitions
├── data_io.py                          # Data loading, I/O operations, and orientation validation  
├── preprocessing.py                    # Biomedical preprocessing and transformations
├── fusion.py                           # Multi-modal data fusion strategies
├── models.py                           # ML models, feature extraction/selection, and caching
├── cv.py                               # Cross-validation pipeline and model training
├── enhanced_pipeline_integration.py    # 4-phase enhanced pipeline coordinator
├── data_quality_analyzer.py            # Comprehensive data quality analysis
├── enhanced_evaluation.py              # Enhanced evaluation metrics and plotting
├── missing_data_handler.py             # Centralized missing data management
├── fusion_aware_preprocessing.py       # Fusion-first processing (legacy module name)
├── validation_coordinator.py           # Coordinated validation framework
├── plots.py                            # Basic visualization functions
├── mad_analysis.py                     # Statistical analysis and comparison
├── utils.py                            # Utility functions and performance monitoring
├── logging_utils.py                    # Enhanced logging and performance tracking
├── tuner_halving.py                    # Hyperparameter optimization
├── samplers.py                         # Data sampling and cross-validation strategies
├── fast_feature_selection.py          # Optimized feature selection methods
├── __init__.py                         # Package initialization
├── hp_best/                            # Pre-tuned hyperparameters for optimal performance
├── tuner_logs/                         # Hyperparameter tuning logs and progress tracking
├── data_quality_analysis/              # Comprehensive data quality reports
│   ├── classification/                # Classification task quality analysis
│   ├── regression/                    # Regression task quality analysis
│   ├── plots/                         # Data quality visualization plots
│   └── summary/                       # Overall quality summary reports
├── setup_and_info/                     # Setup and documentation files
│   ├── setup.py                       # Package installation script
│   ├── pyproject.toml                 # Modern Python packaging
│   ├── requirements.txt               # Core dependencies
│   ├── requirements-dev.txt           # Development dependencies
│   ├── MANIFEST.in                    # Package manifest
│   ├── test_installation.py           # Installation verification
│   ├── DEPENDENCIES_SUMMARY.md        # Dependencies documentation
│   └── MRMR_FIX_SUMMARY.md           # MRMR implementation notes
├── final_results/                      # Final experimental results
│   ├── AML/                           # AML dataset results
│   ├── Sarcoma/                       # Sarcoma dataset results
│   ├── Breast/                        # Breast cancer results
│   ├── Colon/                         # Colon cancer results
│   ├── Kidney/                        # Kidney cancer results
│   ├── Liver/                         # Liver cancer results
│   ├── Lung/                          # Lung cancer results
│   ├── Melanoma/                      # Melanoma results
│   └── Ovarian/                       # Ovarian cancer results
├── data/                              # Dataset storage
│   ├── aml/                           # AML dataset files
│   ├── sarcoma/                       # Sarcoma dataset files
│   ├── breast/                        # Breast cancer dataset files
│   ├── colon/                         # Colon cancer dataset files
│   ├── kidney/                        # Kidney cancer dataset files
│   ├── liver/                         # Liver cancer dataset files
│   ├── lung/                          # Lung cancer dataset files
│   ├── melanoma/                      # Melanoma dataset files
│   ├── ovarian/                       # Ovarian cancer dataset files
│   └── clinical/                      # Clinical data files
├── output_main_without_mrmr/          # Pipeline outputs without MRMR
├── debug_logs/                        # Debug and logging files
├── tests/                             # Unit tests
├── test_data/                         # Test datasets
│   ├── classification/                # Classification test data
│   └── regression/                    # Regression test data
├── .cache/                            # Cache directory
├── .gitignore                         # Git ignore rules
├── .gitattributes                     # Git attributes
└── README.md                          # This file
```

## Recent Pipeline Enhancements

### Version 4.0 - Feature-First Architecture Implementation (CURRENT)
- ✅ **CURRENT: Feature-First Architecture**: Complete implementation of Feature Processing → Fusion → Model Training pipeline order
- ✅ **Dual Architecture Support**: Feature-first as default, legacy fusion-first via `--fusion-first` flag
- ✅ **Modality-Specific Processing**: Apply extractors/selectors to each modality independently before fusion
- ✅ **Enhanced Experimental Loop**: Algorithm → Features → Fusion → Model order for comprehensive evaluation
- ✅ **Backward Compatibility**: Legacy fusion-first architecture remains available for comparison

### Version 3.1 - Enhanced CV Strategies & Algorithm Robustness
- ✅ **Task-Appropriate Cross-Validation**: Proper regression (KFold/RepeatedKFold) vs classification (StratifiedKFold) strategies
- ✅ **Adaptive CV Selection**: Dataset size-based strategy selection for optimal numerical stability
- ✅ **ElasticNet Robustness**: Small dataset detection with multi-level fallback strategies (ElasticNetCV → ElasticNet → LinearRegression)
- ✅ **Enhanced Error Handling**: Safe attribute access and robust sklearn compatibility
- ✅ **Numerical Stability**: Comprehensive safeguards throughout the pipeline

### Version 3.0 - 4-Phase Enhanced Pipeline Architecture
- ✅ **4-Phase Integration**: Early quality assessment, feature-first processing, centralized missing data, coordinated validation
- ✅ **Feature-First Pipeline Order**: Feature processing applied FIRST to raw modalities, then fusion applied to processed features
- ✅ **Missing Data-Adaptive Fusion**: 7 fusion methods for clean data, 4 robust methods for missing data scenarios
- ✅ **Comprehensive Data Quality Analysis**: Automated quality scoring with detailed reporting and preprocessing guidance
- ✅ **Enhanced Cross-Validation**: Patient-level grouping, numerical stability checks, and robust fold creation
- ✅ **Pre-Tuned Hyperparameters**: Optimized parameters stored in `hp_best/` for immediate high performance

### Version 2.5 - Advanced Data Quality & Stability
- ✅ **Data Orientation Validation**: Automatic detection and correction of transposed data matrices
- ✅ **Numerical Stability Framework**: Comprehensive NaN/inf detection and prevention
- ✅ **Modality-Specific Preprocessing**: Tailored configurations for gene expression, miRNA, and methylation data
- ✅ **Enhanced Missing Data Handling**: Intelligent pattern analysis and adaptive imputation strategies

### Version 2.0 - Multi-Modal Fusion Integration  
- ✅ **Advanced Fusion Methods**: MKL, attention-weighted, learnable weighted, average, sum, and early fusion PCA
- ✅ **Fusion-First Architecture**: Fusion applied to raw modalities before feature processing
- ✅ **Performance Monitoring**: Real-time memory usage tracking and computational efficiency optimization
- ✅ **Intelligent Caching**: LRU caching system for expensive extraction/selection operations

## Architecture Comparison

### Feature-First vs Fusion-First

| Aspect | Feature-First (NEW) | Fusion-First (LEGACY) |
|--------|---------------------|------------------------|
| **Pipeline Order** | Raw Data → Feature Processing → Fusion → Model Training | Raw Data → Fusion → Feature Processing → Model Training |
| **Processing Scope** | Each modality processed independently | Fused data processed as single unit |
| **Feature Quality** | Modality-specific feature optimization | Generic feature processing on fused data |
| **Computational Efficiency** | Parallel modality processing possible | Sequential processing required |
| **Interpretability** | Clear modality contributions | Mixed modality features harder to interpret |
| **Scalability** | Better for large feature spaces | Can struggle with high-dimensional fused data |
| **Fusion Quality** | Works with optimized features | Works with raw noisy data |
| **Current Status** | **CURRENT IMPLEMENTATION** | **LEGACY SUPPORT** |
| **Usage** | `python main.py` (default) | `python main.py --fusion-first` |

### When to Use Feature-First Architecture (RECOMMENDED)

- **Standard use cases**: This is the current default and recommended approach
- **Large feature spaces**: When individual modalities have thousands of features
- **Modality-specific optimization**: When different data types need specialized processing
- **Interpretable results**: When understanding individual modality contributions is important
- **Computational constraints**: When parallel processing of modalities is beneficial
- **Quality fusion**: When fusion should work with clean, optimized features

### When to Use Fusion-First Architecture (LEGACY)

- **Legacy compatibility**: When reproducing previous results or research comparisons
- **Simple fusion methods**: When using basic concatenation or averaging
- **Research purposes**: When comparing different architectural approaches
- **Exploratory analysis**: When testing different fusion strategies quickly

## Contributing

This project is part of ongoing research. For questions or contributions, please contact the research team.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@misc{data_fusion_optimization_2025,
  title={Multi-Omics Data Fusion Optimization using Intermediate Integration Techniques},
  author={[Krzysztof Nowak]},
  year={2025},
  institution={VU Amsterdam},
  type={Bachelor's Thesis}
}
```

## Pipeline Implementation Status

### Current Implementation (Version 4.0)

✅ **Feature-First Architecture**: The pipeline currently implements feature-first processing as the default approach
- Feature extraction/selection applied to each modality separately **before** fusion
- Use `python main.py` to run with the current implementation
- Optimal for multi-omics data with modality-specific preprocessing needs

✅ **Legacy Support**: Fusion-first architecture available for research comparison
- Use `python main.py --fusion-first` to run the legacy implementation
- Maintained for backward compatibility and research purposes

✅ **Memory Optimization**: Sequential processing available for memory-constrained environments
- Use `python main.py --sequential` for memory-friendly processing
- Processes one algorithm at a time through all fusion techniques

### Recommended Usage

For **standard research and analysis**: Use the default feature-first implementation:
```bash
python main.py  # Feature-first architecture (recommended)
```

For **memory-constrained systems**: Use sequential processing:
```bash
python main.py --sequential  # Memory-optimized feature-first
```

For **research comparison**: Legacy fusion-first is available:
```bash
python main.py --fusion-first  # Legacy implementation
```

## Acknowledgments

- VU Amsterdam Faculty of Science
- Research supervisors and collaborators
- The Cancer Genome Atlas (TCGA) for providing the datasets
- Open-source scientific Python community 

## Version 4.0 - Sequential Processing Architecture

This pipeline supports multiple processing architectures:

### Processing Architectures

| Architecture | Processing Order | Usage | Memory Management |
|-------------|------------------|-------|-------------------|
| **Feature-First** (CURRENT DEFAULT) | Raw Data -> Feature Processing -> Fusion -> Model Training | `python main.py` | Efficient, processes all modalities together |
| **Sequential** (MEMORY-OPTIMIZED) | Raw Data -> One Extractor/Selector -> All Fusion Techniques -> All Models (repeat) | `python main.py --sequential` | Memory-friendly, processes one algorithm at a time |
| **Fusion-First** (LEGACY) | Raw Data -> Fusion -> Feature Processing -> Model Training | `python main.py --fusion-first` | Legacy compatibility |

### Sequential Processing Benefits

The new **Sequential Processing** mode provides:

1. **Better Memory Management**: Processes one extractor/selector at a time, preventing out-of-memory errors
2. **Clear Progress Tracking**: Shows exactly which algorithm and fusion technique is being processed
3. **Systematic Processing**: Completes all fusion techniques and models for one algorithm before moving to the next
4. **Terminal Visibility**: Clear progress indicators showing current processing status

### Sequential Processing Order

When using `--sequential`, the pipeline processes in this order:

```
Dataset (e.g., AML regression)
├── Missing Percentage: 0%
│   ├── Extractor: PCA
│   │   ├── Parameter: 8 components
│   │   │   ├── Fusion: attention_weighted -> Models: [LinearRegression, ElasticNet, RandomForest]
│   │   │   ├── Fusion: learnable_weighted -> Models: [LinearRegression, ElasticNet, RandomForest]
│   │   │   ├── Fusion: mkl -> Models: [LinearRegression, ElasticNet, RandomForest]
│   │   │   ├── Fusion: average -> Models: [LinearRegression, ElasticNet, RandomForest]
│   │   │   ├── Fusion: sum -> Models: [LinearRegression, ElasticNet, RandomForest]
│   │   │   └── Fusion: early_fusion_pca -> Models: [LinearRegression, ElasticNet, RandomForest]
│   │   ├── Parameter: 16 components
│   │   │   └── (same fusion techniques and models)
│   │   └── Parameter: 32 components
│   │       └── (same fusion techniques and models)
│   ├── Extractor: KPCA
│   │   └── (same parameter and fusion processing)
│   └── (continue for all extractors)
├── Missing Percentage: 20%
│   └── (same processing with missing-data compatible fusion techniques)
└── Missing Percentage: 50%
    └── (same processing with missing-data compatible fusion techniques)
```

### Terminal Output Example

```bash
================================================================================
🚀 STARTING SEQUENTIAL EXTRACTION PIPELINE
📊 Dataset: AML (regression)
🔧 Transformers: ['PCA', 'KPCA', 'PLS', 'SparsePLS', 'FA']
🎯 Models: ['LinearRegression', 'ElasticNet', 'RandomForestRegressor']
================================================================================

📈 PROCESSING MISSING DATA: 0%
────────────────────────────────────────────────────────────────

🔧 PROCESSING TRANSFORMER: PCA
────────────────────────────────────────

  📊 Processing PCA-8
  🎯 [EXTRACT-REG CV] 1/45 => AML | PCA-8 | Missing: 0%
    🔗 Fusion techniques for 0% missing: 7 methods

    🔗 [1/7] Fusion: attention_weighted
      🤖 Training models: ['LinearRegression', 'ElasticNet', 'RandomForestRegressor']
        [1/3] Model: LinearRegression -> R2=0.234
        [2/3] Model: ElasticNet -> R2=0.189
        [3/3] Model: RandomForestRegressor -> R2=0.267
      ✅ Completed fusion: attention_weighted

    🔗 [2/7] Fusion: learnable_weighted
      🤖 Training models: ['LinearRegression', 'ElasticNet', 'RandomForestRegressor']
        [1/3] Model: LinearRegression -> R2=0.241
        [2/3] Model: ElasticNet -> R2=0.195
        [3/3] Model: RandomForestRegressor -> R2=0.273
      ✅ Completed fusion: learnable_weighted

    🔗 [3/7] Fusion: mkl
      🤖 Training models: ['LinearRegression', 'ElasticNet', 'RandomForestRegressor']
        [1/3] Model: LinearRegression -> R2=0.228
        [2/3] Model: ElasticNet -> R2=0.202
        [3/3] Model: RandomForestRegressor -> R2=0.251
      ✅ Completed fusion: mkl

    🔗 [4/7] Fusion: average
      🤖 Training models: ['LinearRegression', 'ElasticNet', 'RandomForestRegressor']
        [1/3] Model: LinearRegression -> R2=0.215
        [2/3] Model: ElasticNet -> R2=0.178
        [3/3] Model: RandomForestRegressor -> R2=0.239
      ✅ Completed fusion: average

    🔗 [5/7] Fusion: sum
      🤖 Training models: ['LinearRegression', 'ElasticNet', 'RandomForestRegressor']
        [1/3] Model: LinearRegression -> R2=0.218
        [2/3] Model: ElasticNet -> R2=0.181
        [3/3] Model: RandomForestRegressor -> R2=0.242
      ✅ Completed fusion: sum

    🔗 [6/7] Fusion: standard_concat
      🤖 Training models: ['LinearRegression', 'ElasticNet', 'RandomForestRegressor']
        [1/3] Model: LinearRegression -> R2=0.231
        [2/3] Model: ElasticNet -> R2=0.186
        [3/3] Model: RandomForestRegressor -> R2=0.249
      ✅ Completed fusion: standard_concat

    🔗 [7/7] Fusion: early_fusion_pca
      🤖 Training models: ['LinearRegression', 'ElasticNet', 'RandomForestRegressor']
        [1/3] Model: LinearRegression -> R2=0.223
        [2/3] Model: ElasticNet -> R2=0.194
        [3/3] Model: RandomForestRegressor -> R2=0.255
      ✅ Completed fusion: early_fusion_pca

  ✅ Completed PCA-8

✅ Completed transformer: PCA
✅ Completed missing percentage: 0%

🎉 SEQUENTIAL PIPELINE COMPLETED FOR AML
================================================================================
```

## Usage Examples

### Basic Usage
```bash
# Default feature-first processing (CURRENT IMPLEMENTATION)
python main.py

# Sequential processing (memory-friendly)
python main.py --sequential

# Legacy fusion-first processing (for research comparison)
python main.py --fusion-first
```

### Advanced Usage
```bash
# Sequential processing for specific dataset
python main.py --sequential --dataset AML

# Sequential processing for regression only
python main.py --sequential --regression-only

# Sequential processing with specific n_val
python main.py --sequential --n-val 32
```

## Configuration

### Dataset Configuration

The pipeline supports both regression and classification datasets:

- **Regression datasets**: AML, Sarcoma, etc. (continuous outcomes)
- **Classification datasets**: Breast, Colon, Kidney, etc. (categorical outcomes)

### Feature Extraction and Selection

- **Extractors**: PCA, Kernel PCA, PLS, Sparse PLS, Factor Analysis
- **Selectors**: Univariate selection, RFE, LASSO-based selection
- **Components/Features**: Configurable via `N_VALUES_LIST` in `config.py`

### Fusion Techniques

- **For clean data (0% missing)**: attention_weighted, learnable_weighted, mkl, average, sum, standard_concat, early_fusion_pca (7 methods)
- **For missing data (>0% missing)**: mkl, average, sum, early_fusion_pca (4 methods)

### Machine Learning Models

- **Regression**: Linear Regression, Elastic Net, Random Forest Regressor
- **Classification**: Logistic Regression, SVM, Random Forest Classifier

## Memory Management

The sequential processing mode addresses memory issues by:

1. **Processing one algorithm at a time**: Prevents memory accumulation from parallel processing
2. **Clearing memory between algorithms**: Explicit garbage collection between major processing steps
3. **Reduced concurrent operations**: Minimizes simultaneous memory-intensive operations
4. **Progress tracking**: Clear indication of current processing status to identify memory bottlenecks

## Output Structure

```
output/
├── AML/
│   ├── models/
│   │   ├── best_model_extraction_LinearRegression_PCA_8_0.0_attention_weighted.pkl
│   │   ├── best_model_extraction_ElasticNet_PCA_8_0.0_attention_weighted.pkl
│   │   └── ...
│   ├── metrics/
│   │   ├── AML_best_fold_extraction_PCA_8_0.0_attention_weighted_metrics.csv
│   │   └── ...
│   └── plots/
│       ├── AML_best_fold_extraction_PCA_8_LinearRegression_0.0_attention_weighted_scatter.png
│       └── ...
└── ...
``` 