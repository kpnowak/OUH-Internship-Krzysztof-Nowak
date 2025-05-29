# Multi-Omics Data Fusion Optimization Pipeline

## Project Overview

This repository contains a comprehensive machine learning pipeline for multi-omics data fusion optimization using intermediate integration techniques. This project is part of a Bachelor's Thesis in Artificial Intelligence at VU Amsterdam, contributing to a larger research initiative focused on developing advanced machine learning models for early and accurate cancer detection.

### Research Context

This work is part of a broader research project aimed at creating innovative machine learning models that can detect cancer faster and more accurately in patients by leveraging multiple types of biological data. The research explores how different data integration strategies and feature extraction/selection algorithms perform when working with multi-modal omics data, which is crucial for understanding complex biological processes and disease mechanisms.

### Project Purpose

The primary goal of this project is to develop a specialized feature extraction and selection algorithm specifically optimized for cancer detection machine learning models. To achieve this objective, the project conducts comprehensive research and comparative analysis of existing algorithms to identify the most effective approaches for multi-omics cancer data.

The research methodology involves systematically evaluating state-of-the-art algorithms across different parameter configurations for both classification and regression tasks:

- **Feature Extraction Algorithms**: PCA, NMF, ICA, Factor Analysis, PLS Regression, LDA, Kernel PCA
- **Feature Selection Algorithms**: MRMR, LASSO, ElasticNet, F-test, Chi-squared, Boruta, Random Forest Feature Importance
- **Machine Learning Models**: Linear/Logistic Regression, Random Forest, ElasticNet, SVM
- **Integration Strategies**: Intermediate integration with missing modality simulation
- **Parameter Variations**: Different numbers of components/features (8, 16, 32) and missing data percentages (0%, 20%, 50%)

This extensive benchmarking serves as the foundation for designing a novel algorithm that leverages the strengths of existing methods while addressing the unique challenges of multi-omics cancer data integration and feature optimization.

### Data Types

The pipeline works with **multi-omics cancer data**, specifically:

- **Gene Expression Data (exp.csv)**: Transcriptomic profiles measuring mRNA expression levels
- **miRNA Data (mirna.csv)**: MicroRNA expression profiles for post-transcriptional regulation analysis  
- **Methylation Data (methy.csv)**: DNA methylation patterns indicating epigenetic modifications
- **Clinical Data**: Patient outcomes and clinical variables for supervised learning

This multi-modal approach captures different layers of biological information, providing a comprehensive view of the molecular landscape in cancer patients.

## Algorithm Architecture

### Pipeline Workflow

1. **Data Loading & Preprocessing**: 
   - Robust file loading with automatic format detection
   - Sample ID standardization across modalities
   - Data quality validation and optimization

2. **Missing Modality Simulation**:
   - Simulates real-world scenarios where some data types may be unavailable
   - Tests robustness across different missing data percentages (0%, 20%, 50%)

3. **Feature Extraction/Selection**:
   - **Extraction Pipeline**: Dimensionality reduction techniques (PCA, ICA, NMF, etc.)
   - **Selection Pipeline**: Feature selection methods (MRMR, LASSO, etc.)
   - Parallel processing for efficiency

4. **Data Fusion**:
   - Intermediate integration strategy
   - Modality-specific imputation for missing values
   - Concatenation of processed features

5. **Model Training & Evaluation**:
   - Cross-validation with robust fold creation
   - Multiple machine learning algorithms
   - Comprehensive performance metrics

6. **MAD Analysis**:
   - Mean Absolute Deviation analysis for algorithm comparison
   - Critical difference diagrams for statistical significance testing
   - Detailed performance statistics

### Supported Algorithms

#### Feature Extraction Methods
- **PCA (Principal Component Analysis)**: Linear dimensionality reduction
- **NMF (Non-negative Matrix Factorization)**: Parts-based representation
- **ICA (Independent Component Analysis)**: Statistical independence-based separation
- **Factor Analysis**: Latent factor modeling
- **PLS Regression**: Supervised dimensionality reduction
- **LDA (Linear Discriminant Analysis)**: Classification-oriented projection
- **Kernel PCA**: Non-linear dimensionality reduction

#### Feature Selection Methods
- **MRMR (Minimum Redundancy Maximum Relevance)**: Information-theoretic selection
- **LASSO**: L1-regularized linear model selection
- **ElasticNet**: Combined L1/L2 regularization
- **F-test**: Statistical significance-based selection
- **Chi-squared**: Categorical feature selection
- **Boruta**: All-relevant feature selection
- **Random Forest Feature Importance**: Tree-based importance ranking

#### Machine Learning Models
- **Regression**: Linear Regression, Random Forest Regressor, ElasticNet
- **Classification**: Logistic Regression, Random Forest Classifier, SVM

## Datasets

The pipeline includes multiple cancer datasets from The Cancer Genome Atlas (TCGA):

### Regression Datasets
- **AML (Acute Myeloid Leukemia)**: Predicting blast cell percentage
- **Sarcoma**: Predicting tumor length

### Classification Datasets  
- **Breast Cancer**: Pathologic T-stage classification
- **Colon Cancer**: Pathologic T-stage classification
- **Kidney Cancer**: Pathologic T-stage classification
- **Liver Cancer**: Pathologic T-stage classification
- **Lung Cancer**: Pathologic T-stage classification
- **Melanoma**: Pathologic T-stage classification
- **Ovarian Cancer**: Clinical stage classification

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

- **Missing data percentages**: Modify `MISSING_MODALITIES_CONFIG["missing_percentages"]`
- **Algorithm selection**: Enable/disable algorithms in `get_*_extractors()` and `get_*_selectors()` functions
- **Model parameters**: Adjust `MODEL_OPTIMIZATIONS` dictionary
- **Memory settings**: Modify `MEMORY_OPTIMIZATION` and `CACHE_CONFIG`

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
- ✅ Python version compatibility (3.8+)
- ✅ All core dependencies
- ✅ module imports
- ✅ Basic functionality
- ✅ Command-line interface
- ⚠️ Optional dependencies (warnings if missing)

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

- **Memory Usage**: Optimized for high-memory systems (8GB+ RAM recommended)
- **Parallel Processing**: Utilizes multiple CPU cores for efficiency
- **Caching**: Intelligent caching system to avoid redundant computations
- **Early Stopping**: Prevents overfitting and reduces training time

## Repository Structure

```
OUH-Internship-Krzysztof-Nowak/
├── install.py                 # Convenience installation script
├── main.py                    # Main pipeline entry point
├── cli.py                     # Command-line interface
├── config.py                  # Configuration settings
├── data_io.py                 # Data loading and processing
├── preprocessing.py           # Data preprocessing utilities
├── fusion.py                  # Multi-modal data fusion
├── models.py                  # Machine learning models and caching
├── cv.py                      # Cross-validation pipeline
├── plots.py                   # Visualization functions
├── mad_analysis.py            # MAD analysis implementation
├── utils.py                   # Utility functions
├── run_mad_analysis.py        # Standalone MAD analysis script
├── _process_single_modality.py # Single modality processing utilities
├── utils_boruta.py            # Boruta feature selection utilities
├── mrmr_helper.py             # MRMR feature selection implementation
├── create_diagrams_only.py    # Standalone diagram creation
├── __init__.py                # Package initialization
├── setup_and_info/            # Setup and documentation files
│   ├── setup.py               # Package installation script
│   ├── pyproject.toml         # Modern Python packaging
│   ├── requirements.txt       # Core dependencies
│   ├── requirements-dev.txt   # Development dependencies
│   ├── MANIFEST.in            # Package manifest
│   ├── test_installation.py   # Installation verification
│   ├── MRMR_FIX_SUMMARY.md    # MRMR implementation notes
│   └── Multi-omics data fusion optimization using intermediate integration techniques-1 (2).pdf
├── data/                      # Dataset storage
├── output/                    # Results and outputs
├── tests/                     # Unit tests
├── test_data/                 # Test datasets
└── README.md                  # This file
```

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