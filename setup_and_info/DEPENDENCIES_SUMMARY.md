# Dependencies Summary for Multi-Omics Data Fusion Pipeline

This document provides a comprehensive overview of all dependencies required by the algorithms in this pipeline.

## Core Dependencies (Required)

These packages are essential for the pipeline to function and are automatically installed with the basic installation:

### Numerical Computing and Data Manipulation
- **numpy>=1.21.0** - Fundamental package for scientific computing
- **pandas>=1.3.0** - Data manipulation and analysis library
- **scipy>=1.7.0** - Scientific computing library

### Machine Learning and Statistical Analysis
- **scikit-learn>=1.0.0** - Machine learning library providing:
  - Feature selection algorithms (mutual_info, f_test, chi2, variance threshold)
  - Dimensionality reduction (PCA, LDA, ICA, FastICA)
  - Regression models (LinearRegression, ElasticNet, Lasso, SVR)
  - Classification models (LogisticRegression, SVC, RandomForestClassifier)
  - Cross-validation and model selection utilities
  - Preprocessing tools (StandardScaler, MinMaxScaler)
  - Metrics for evaluation

### Visualization and Plotting
- **matplotlib>=3.5.0** - Plotting library for creating visualizations
- **seaborn>=0.11.0** - Statistical data visualization library

### Parallel Processing and Performance
- **joblib>=1.1.0** - Parallel computing and caching utilities
- **threadpoolctl>=3.0.0** - Thread pool control for numerical libraries
- **psutil>=5.8.0** - System and process monitoring

### Feature Selection Algorithms
- **boruta>=0.3.0** - Boruta feature selection algorithm implementation

## Optional Dependencies

### Enhanced Visualization (--visualization option)
- **scikit-posthocs>=0.6.0** - Post-hoc statistical tests and critical difference diagrams for MAD analysis

### Development Tools (--development option)
- **pytest>=6.0.0** - Testing framework
- **pytest-cov>=2.12.0** - Coverage testing
- **black>=21.0.0** - Code formatting
- **flake8>=3.9.0** - Code linting
- **mypy>=0.910** - Type checking

## Standard Library Modules (Included with Python)

The following modules are part of Python's standard library and do not require separate installation:

### Core System Modules
- `os` - Operating system interface
- `sys` - System-specific parameters and functions
- `logging` - Logging facility
- `time` - Time-related functions
- `warnings` - Warning control

### Data Structures and Utilities
- `typing` - Type hints support
- `collections` - Specialized container datatypes
- `itertools` - Iterator functions
- `functools` - Higher-order functions and operations on callable objects
- `contextlib` - Context management utilities

### String and Pattern Processing
- `re` - Regular expression operations
- `difflib` - Helpers for computing deltas (used for fuzzy ID matching)
- `hashlib` - Secure hash and message digest algorithms

### Concurrency and Performance
- `threading` - Thread-based parallelism
- `gc` - Garbage collection interface
- `copy` - Shallow and deep copy operations

### File and Path Operations
- `pathlib` - Object-oriented filesystem paths

### Command Line and Process Management
- `argparse` - Command-line argument parsing
- `subprocess` - Subprocess management
- `importlib` - Import utilities

### Date and Time
- `datetime` - Date and time handling
- `random` - Random number generation

### Error Handling and Debugging
- `traceback` - Print or retrieve a stack traceback

### Data Classes
- `dataclasses` - Data class utilities (Python 3.7+)

## Algorithm-Specific Dependencies

### Feature Selection Algorithms Used
1. **Mutual Information** (scikit-learn)
2. **F-test** (scikit-learn)
3. **Chi-squared test** (scikit-learn)
4. **Variance Threshold** (scikit-learn)
5. **Boruta** (boruta package)
6. **MRMR** (custom implementation using scikit-learn)
7. **Lasso-based selection** (scikit-learn)
8. **Random Forest-based selection** (scikit-learn)

### Dimensionality Reduction Algorithms Used
1. **Principal Component Analysis (PCA)** (scikit-learn)
2. **Linear Discriminant Analysis (LDA)** (scikit-learn)
3. **Independent Component Analysis (ICA)** (scikit-learn)
4. **FastICA** (scikit-learn)
5. **Partial Least Squares (PLS)** (scikit-learn)

### Machine Learning Models Used
1. **Linear Regression** (scikit-learn)
2. **Elastic Net** (scikit-learn)
3. **Lasso Regression** (scikit-learn)
4. **Support Vector Regression (SVR)** (scikit-learn)
5. **Random Forest Regressor** (scikit-learn)
6. **Logistic Regression** (scikit-learn)
7. **Support Vector Classifier (SVC)** (scikit-learn)
8. **Random Forest Classifier** (scikit-learn)

### Statistical Analysis
1. **MAD (Median Absolute Deviation) Analysis** (scipy.stats)
2. **Critical Difference Diagrams** (scikit-posthocs - optional)
3. **Cross-validation** (scikit-learn)
4. **Performance metrics** (scikit-learn)

## Installation Commands

### Basic Installation
```bash
python install.py
# Choose option 1 for basic installation
```

### Full Installation (Recommended)
```bash
python install.py
# Choose option 4 for full installation
```

### Manual Installation
```bash
# Core dependencies
pip install -r setup_and_info/requirements.txt

# Development dependencies (optional)
pip install -r setup_and_info/requirements-dev.txt
```

## Verification

After installation, run the verification script:
```bash
python setup_and_info/test_installation.py
```

Or use the interactive installer's verification option:
```bash
python install.py
# Choose to run verification tests when prompted
```

## Troubleshooting

If you encounter issues with specific packages:

1. **Boruta installation issues**: Try installing with conda instead of pip
2. **Scikit-posthocs issues**: This is optional and only needed for MAD analysis diagrams
3. **Memory issues**: The pipeline is optimized for high-memory systems (8GB+ RAM recommended)
4. **Threading issues**: Adjust `N_JOBS` in `config.py` if you encounter threading problems

For more detailed troubleshooting, see the main README.md file. 