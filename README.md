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
5. **Design** a purpose-built extraction or selection algorithm optimised for multi-omics cancer data.

### Project Purpose

The primary goal of this project is to develop a specialized feature extraction and selection algorithm specifically optimized for cancer detection machine learning models. To achieve this objective, the project conducts comprehensive research and comparative analysis of existing algorithms to identify the most effective approaches for multi-omics cancer data.

The research methodology involves systematically evaluating state-of-the-art algorithms across different parameter configurations for both classification and regression tasks:

- **Feature Extraction Algorithms**: PCA, KPLS, Factor Analysis, PLS/PLS-DA, SparsePLS
- **Feature Selection Algorithms**: ElasticNetFS, Random Forest Importance, Variance F-test, LASSO, f_regressionFS, LogisticL1, XGBoostFS
- **Machine Learning Models**: Linear/Logistic Regression, Random Forest, ElasticNet, SVM
- **Advanced Integration Strategies**: Attention-weighted fusion, learnable weighted fusion, MKL, SNF, early fusion PCA
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

## Enhanced Pipeline Features

###  **Advanced Fusion Strategies**
- **Attention-Weighted Fusion**: Sample-specific weighting with neural attention mechanisms (default)
- **Learnable Weighted Fusion**: Performance-based modality weighting with cross-validation
- **Late Fusion Stacking**: Meta-learner approach for complex modality interactions
- **Early Fusion PCA**: Dimensionality reduction before fusion for high-dimensional data

### 🧬 **Modality-Specific Preprocessing**
- **Gene Expression**: Biological KNN imputation (k=7) with similarity matrices
- **miRNA**: Zero-inflated transformations + biological KNN imputation (k=5)
- **Methylation**: Mean imputation for low-missingness data + conservative scaling
- **Advanced Sparsity Handling**: Log1p transformations, outlier capping, variance-based filtering

###  **Missing Data Intelligence**
- **Missing Data Indicators**: Binary features capturing missingness patterns (threshold: 5%)
- **Adaptive Imputation**: Automatic strategy selection based on data characteristics
- **Robust Missing Modality Simulation**: Real-world missing data scenarios (0%, 20%, 50%)

### ⚡ **Performance Optimizations**
- **Enhanced Feature Selection**: MAD thresholds (0.05), correlation removal (0.90), sparsity filtering (0.9)
- **Stricter Regularization**: ElasticNet alpha range (0.1-0.5) for better generalization
- **Numerical Stability**: Automatic detection and removal of problematic features
- **Memory Optimization**: Intelligent caching and parallel processing

## Experimental Design

> **Full code and preliminary results are available in the GitHub repository**
> **OUH-Internship-Krzysztof-Nowak**.

The pipeline systematically evaluates all combinations of algorithms and parameters using the following enhanced experimental structure:

```python
# Regression branch algorithms (ACTUAL IMPLEMENTATION)
REGRESSION_EXTRACTORS = [PCA, KPLS, FA, PLS, SparsePLS]
REGRESSION_SELECTORS = [ElasticNetFS, RFImportance, VarianceFTest, LASSO, f_regressionFS]
REGRESSION_MODELS = [LinearRegression, ElasticNet, RandomForestRegressor]

# Classification branch algorithms (ACTUAL IMPLEMENTATION)
CLASSIFICATION_EXTRACTORS = [PCA, FA, LDA, 'PLS-DA', SparsePLS]
CLASSIFICATION_SELECTORS = [ElasticNetFS, RFImportance, VarianceFTest, LogisticL1, XGBoostFS]
CLASSIFICATION_MODELS = [LogisticRegression, SVC, RandomForestClassifier]

# Enhanced fusion strategies (ACTUAL IMPLEMENTATION)
FUSION_STRATEGIES = {
    'attention_weighted': 'Sample-specific attention weighting (OPTIMIZED default)',
    'learnable_weighted': 'Performance-based modality weighting', 
    'mkl': 'Multiple Kernel Learning with RBF kernels',
    'snf': 'Similarity Network Fusion with spectral clustering',
    'early_fusion_pca': 'PCA-based early integration'
}

# Experimental loop for each dataset
for ALGORITHM in EXTRACTORS + SELECTORS:
    for N_FEATURES in [8, 16, 32]:
        for MISSING in [0, 0.20, 0.50]:
            # ENHANCED: Adaptive fusion strategy selection (ACTUAL IMPLEMENTATION)
            if MISSING == 0:
                INTEGRATIONS = [attention_weighted, learnable_weighted, 
                               mkl, snf, early_fusion_pca]
            else:
                INTEGRATIONS = [mkl, snf, early_fusion_pca]
            
            for INTEGRATION in INTEGRATIONS:
                for MODEL in TASK_SPECIFIC_MODELS:
                    run_experiment(
                        algorithm=ALGORITHM,
                        n_features=N_FEATURES,
                        missing_rate=MISSING,
                        integration=INTEGRATION,
                        model=MODEL,
                        # NEW: Enhanced preprocessing
                        missing_indicators=True,
                        modality_specific_imputation=True,
                        advanced_sparsity_handling=True
                    )
```

### Enhanced Experimental Features:
- **Adaptive Strategy Selection**: For 0% missing data: 5 fusion methods; for >0% missing: 3 robust methods
- **Missing Data Indicators**: Binary features capturing informative missingness patterns  
- **Modality-Specific Processing**: Tailored preprocessing for each genomic data type
- **Robust Validation**: Enhanced cross-validation with sample alignment and numerical stability checks
- **Performance Monitoring**: Real-time memory usage and computational efficiency tracking

This comprehensive experimental design ensures systematic evaluation across:
- **Feature extraction/selection algorithms** (5 extractors + 5 selectors per task type)
- **Feature counts** (8, 16, 32 components/features) 
- **Missing data scenarios** (0%, 20%, 50% missing modalities)
- **Advanced integration strategies** (5 for clean data, 3 for missing data scenarios)
- **Enhanced preprocessing** (modality-specific imputation, missing indicators, sparsity handling)
- **Predictive models** optimized for regression vs. classification tasks

## Deliverables

1. **Literature review** summarising extraction and selection methods for multi-omics data.
2. **Experimental report** (methods, code links, and results tables/plots) highlighting the top-performing algorithm combinations and explaining their success.
3. **Recommendation** of the algorithms most suitable for multi-omics cancer studies, with justification.
4. **New algorithm** specifically designed and empirically validated for this data.

## Algorithm Architecture

### Enhanced Pipeline Workflow

1. **Data Loading & Preprocessing**: 
   - Robust file loading with automatic format detection
   - Sample ID standardization across modalities
   - Data quality validation and optimization
   - **NEW**: Modality-specific preprocessing configurations

2. **Missing Data Intelligence**:
   - **Missing Data Indicators**: Create binary features before imputation (>5% missing threshold)
   - **Modality-Specific Imputation**: Gene expression (biological KNN), miRNA (zero-inflated + KNN), methylation (mean)
   - **Adaptive Strategy Selection**: Choose imputation based on data characteristics

3. **Enhanced Preprocessing Pipeline**:
   - **Numerical Stability Checks**: Automatic detection and removal of problematic features
   - **Advanced Sparsity Handling**: Log1p transformations, outlier capping, targeted feature removal
   - **Smart Skewness Correction**: Box-Cox/Yeo-Johnson transformations with intelligent selection
   - **Robust Scaling**: Modality-aware scaling with outlier clipping

4. **Missing Modality Simulation**:
   - Simulates real-world scenarios where some data types may be unavailable
   - Tests robustness across different missing data percentages (0%, 20%, 50%)
   - **NEW**: Maintains minimum overlap ratios for valid analysis

5. **Feature Extraction/Selection**:
   - **Extraction Pipeline**: Dimensionality reduction techniques (PCA, ICA, NMF, etc.)
   - **Selection Pipeline**: Feature selection methods (MRMR, LASSO, etc.)
   - **Enhanced Feature Selection**: Aggressive MAD thresholds, correlation removal, sparsity filtering
   - Parallel processing for efficiency

6. **Advanced Data Fusion**:
   - **Attention-Weighted Fusion**: Neural attention mechanisms for sample-specific weighting
   - **Learnable Weighted Fusion**: Performance-based modality importance with cross-validation
   - **Late Fusion Stacking**: Meta-learner approach using per-modality predictions
   - **Early Fusion PCA**: Dimensionality reduction before integration
   - **Adaptive Strategy Selection**: Choose fusion method based on missing data and task complexity

7. **Model Training & Evaluation**:
   - Cross-validation with robust fold creation and sample alignment
   - Multiple machine learning algorithms with optimized hyperparameters
   - **Enhanced Regularization**: Stricter ElasticNet (alpha 0.1-0.5)
   - Comprehensive performance metrics with numerical stability

8. **MAD Analysis**:
   - Mean Absolute Deviation analysis for algorithm comparison
   - Critical difference diagrams for statistical significance testing
   - Detailed performance statistics with confidence intervals

### Supported Algorithms

#### Feature Extraction Methods (ACTUAL IMPLEMENTATION)
- **PCA (Principal Component Analysis)**: Linear dimensionality reduction
- **KPLS (Kernel PLS)**: Non-linear kernel-based partial least squares
- **Factor Analysis**: Latent factor modeling
- **PLS (Partial Least Squares)**: Supervised dimensionality reduction for regression
- **PLS-DA (PLS Discriminant Analysis)**: Supervised dimensionality reduction for classification
- **SparsePLS**: Sparse partial least squares with feature selection

#### Feature Selection Methods (ACTUAL IMPLEMENTATION)
- **ElasticNetFS**: ElasticNet-based feature selection with L1/L2 regularization
- **RFImportance (Random Forest Importance)**: Tree-based importance ranking
- **VarianceFTest**: Variance-based F-test feature selection
- **LASSO**: L1-regularized linear model selection
- **f_regressionFS**: F-test based regression feature selection
- **LogisticL1**: L1-regularized logistic regression for classification
- **XGBoostFS**: XGBoost-based feature importance selection

#### Advanced Data Fusion Methods (ACTUAL IMPLEMENTATION)
- **Attention-Weighted Fusion**: Neural attention mechanisms for sample-specific modality weighting
- **Learnable Weighted Fusion**: Cross-validation based performance weighting of modalities  
- **MKL (Multiple Kernel Learning)**: RBF kernel-based fusion with optimal kernel weighting
- **SNF (Similarity Network Fusion)**: Spectral clustering-based network integration
- **Early Fusion PCA**: Dimensionality reduction applied to concatenated features

#### Machine Learning Models
- **Regression**: Linear Regression, Random Forest Regressor, ElasticNet (enhanced α=0.1-0.5)
- **Classification**: Logistic Regression, Random Forest Classifier, SVM

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
```bash
python main.py
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
# Example: Run only AML dataset with debug logging and 16 components
python main.py --dataset AML --debug --n-val 16

# Example: Run only classification with verbose logging, skip MAD
python main.py --classification-only --verbose --skip-mad
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

#### Getting Help

- Check the installation test output: `python setup_and_info/test_installation.py`
- Review log files: `debug.log` (created during execution)
- Enable debug mode: `python main.py --debug`

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
- **Numerical Stability**: Automatic detection and removal of problematic features reduces computational overhead
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
├── config.py                           # Configuration settings
├── data_io.py                          # Data loading and processing
├── preprocessing.py                    # Data preprocessing utilities
├── fusion.py                           # Multi-modal data fusion
├── models.py                           # Machine learning models and caching
├── cv.py                               # Cross-validation pipeline
├── plots.py                            # Visualization functions
├── mad_analysis.py                     # MAD analysis implementation
├── utils.py                            # Utility functions
├── logging_utils.py                    # Logging configuration and utilities
├── run_mad_analysis.py                 # Standalone MAD analysis script
├── _process_single_modality.py         # Single modality processing utilities
├── utils_boruta.py                     # Boruta feature selection utilities
├── mrmr_helper.py                      # MRMR feature selection implementation
├── create_diagrams_only.py             # Standalone diagram creation
├── combine_cv_metrics.py               # Cross-validation metrics combination
├── create_classification_plots.py      # Basic classification visualization
├── create_enhanced_classification_plots.py # Enhanced classification plots
├── create_regression_plots.py          # Basic regression visualization
├── create_enhanced_regression_plots.py # Enhanced regression plots
├── top_algorithms_classification.py    # Top performing classification algorithms
├── top_algorithms_regression.py        # Top performing regression algorithms
├── __init__.py                         # Package initialization
├── algorithm_development/              # Algorithm development and testing
│   ├── alg1.py                        # First algorithm implementation
│   ├── alg2_multicore.py              # Multicore algorithm implementation
│   ├── alg3_multi_additions/          # Third algorithm with additions
│   ├── old1_algorithm_output/         # Legacy algorithm outputs
│   ├── old2_algorithm_output/         # Second legacy algorithm outputs
│   └── output_algorithm_multicore/    # Multicore algorithm results
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

### Version 2.1 - Enhanced Missing Data Intelligence
-  **Missing Data Indicators**: Binary features capturing informative missingness patterns
-  **Modality-Specific Imputation**: Gene expression (biological KNN), miRNA (zero-inflated), methylation (mean)
-  **Adaptive Strategy Selection**: Automatic imputation method selection based on data characteristics

### Version 2.0 - Advanced Fusion Strategies  
-  **Attention-Weighted Fusion**: Neural attention mechanisms for sample-specific modality weighting
-  **Learnable Weighted Fusion**: Performance-based modality importance with cross-validation
-  **Late Fusion Stacking**: Meta-learner approach using per-modality predictions
-  **Strategic Optimization**: Attention-weighted fusion now default for 0% missing data scenarios

### Version 1.9 - Enhanced Preprocessing Pipeline
-  **Numerical Stability Checks**: Automatic detection and removal of problematic features
-  **Advanced Sparsity Handling**: Log1p transformations, outlier capping, variance-based filtering
-  **Smart Skewness Correction**: Box-Cox/Yeo-Johnson transformations with intelligent selection
-  **Robust Scaling**: Modality-aware scaling with outlier clipping and quantile normalization

### Version 1.8 - Performance Optimizations
-  **Enhanced Feature Selection**: Aggressive MAD thresholds (0.05), correlation removal (0.90)
-  **Stricter Regularization**: ElasticNet alpha range (0.1-0.5) for better generalization
-  **Memory Optimization**: Intelligent caching and parallel processing for 60GB RAM systems
-  **Computational Efficiency**: Reduced pipeline execution time by ~40% through optimizations

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

## Acknowledgments

- VU Amsterdam Faculty of Science
- Research supervisors and collaborators
- The Cancer Genome Atlas (TCGA) for providing the datasets
- Open-source scientific Python community 