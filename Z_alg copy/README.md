# Multi-modal Machine Learning Pipeline

This package contains a modular machine learning pipeline for processing multi-modal omics data. It includes components for data loading, preprocessing, feature extraction, feature selection, model training, and evaluation.

## Structure

The package is organized into the following modules:

- `config.py` - Configuration settings and dataset definitions
- `io.py` - Data loading and saving functions
- `preprocessing.py` - Data preprocessing and cleaning utilities
- `fusion.py` - Multi-modal data integration
- `models.py` - Feature extraction/selection and model creation
- `cv.py` - Cross-validation and evaluation framework
- `plots.py` - Visualization utilities
- `cli.py` - Command-line interface
- `utils.py` - General utilities (threading, resource management)
- `utils_boruta.py` - Optimized implementation of Boruta feature selection
- `main.py` - Main entry point for running the pipeline

## Features

- Support for regression and classification tasks
- Multiple feature extraction methods: PCA, ICA, NMF, etc.
- Multiple feature selection methods: MRMR, Lasso, Boruta, etc.
- Multiple ML models: Linear/Logistic Regression, RandomForest, SVM, etc.
- Cross-validation with performance metrics
- Results plotting and visualization
- Robust error handling and cache management
- Thread-safe parallel processing
- Missing data simulation for robustness testing

## Usage

```bash
# Run the full pipeline with all datasets
python -m Z_alg.main

# Run only regression datasets
python -m Z_alg.main --regression-only

# Run only classification datasets
python -m Z_alg.main --classification-only

# Run a specific dataset
python -m Z_alg.main --dataset <dataset_name>

# Run with debug output
python -m Z_alg.main --debug
```

## Adding New Datasets

To add a new dataset, modify `config.py` to include a new entry in either `REGRESSION_DATASETS` or `CLASSIFICATION_DATASETS` with the following format:

```python
{
    "name": "dataset_name",
    "base_path": "path/to/dataset",
    "modalities": {
        "modality1": "path/to/modality1.csv",
        "modality2": "path/to/modality2.csv",
        # ...
    },
    "outcome_file": "path/to/outcomes.csv",
    "outcome_col": "target_column_name",
    "outcome_type": "os" | "pfs" | "response" | "class",
    "output_dir": "path/to/output"
}
```

## Testing

The package includes automated tests in the `tests/` directory. Run the tests with pytest:

```bash
cd Z_alg
pytest
```

## Dependencies

- Python 3.6+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Threadpoolctl
- BorutaPy

## License

This project is licensed under the MIT License. 