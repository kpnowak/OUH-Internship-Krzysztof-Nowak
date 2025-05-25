# Z_alg: Multi-Modal Analysis Pipeline

This repository contains a pipeline for multi-modal data analysis, particularly focused on integrating multiple omics data types for predictive modeling tasks.

## Recent Improvements

### Enhanced Data Loading Capabilities

The data loading pipeline has been significantly improved to handle various file formats and inconsistencies in sample identifiers:

1. **Robust file loading**: 
   - Multiple delimiters (`,`, `\t`, ` `, `;`) are automatically detected
   - Multiple encodings are tried automatically (utf-8, latin1, iso-8859-1, cp1252)
   - Improved error handling and detailed logging

2. **Sample ID Standardization**:
   - Automatic detection of sample ID format differences between modalities
   - Standardization of IDs using consistent separators (converting between hyphens, dots, underscores)
   - Fallback mechanisms for more complex ID matching when simple replacement fails
   - Added `normalize_sample_ids` utility function for consistent ID handling

3. **Improved Error Handling**:
   - More robust handling of edge cases like numpy arrays without isna() method
   - Better error reporting and diagnostic logging
   - Graceful failure with informative error messages

### Usage

To run the pipeline with default settings:

```bash
python -m Z_alg.main
```

To specify a specific dataset:

```bash
python -m Z_alg.cli --dataset AML
```

### Available Datasets

- AML: Acute Myeloid Leukemia dataset with gene expression, miRNA, and methylation data
- Colon: Colon cancer dataset with gene expression, miRNA, and methylation data
- (Additional datasets are available but commented out in the config file)

## Configuration

Dataset configurations are defined in `config.py`. Each dataset requires:

- Base path
- Modality paths (paths to each omics data file)
- Outcome file path
- Outcome column name
- ID column name
- Outcome type (continuous/class)

Example configuration:

```python
DatasetConfig(
    name="AML",
    base_path="data/aml",
    modalities={
        "Gene Expression": "exp.csv",
        "miRNA": "mirna.csv",
        "Methylation": "methy.csv"
    },
    outcome_file="data/clinical/aml.csv",
    outcome_col="lab_procedure_bone_marrow_blast_cell_outcome_percent_value",
    id_col="sampleID",
    outcome_type="continuous",
    fix_tcga_ids=True
)
```

## Dependencies

- Python 3.6+
- pandas
- numpy
- scikit-learn
- joblib (for parallelization)

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

## License

This project is licensed under the MIT License. 