# Data Directory Structure

This directory contains the multi-omics cancer datasets used by the pipeline. The data is organized by cancer type and modality.

## Directory Structure

```
data/
├── aml/                    # Acute Myeloid Leukemia (Regression)
│   ├── exp.csv            # Gene expression data
│   ├── mirna.csv          # miRNA expression data
│   └── methy.csv          # Methylation data
├── breast/                # Breast Cancer (Classification)
│   ├── exp.csv
│   ├── mirna.csv
│   └── methy.csv
├── colon/                 # Colon Cancer (Classification)
│   ├── exp.csv
│   ├── mirna.csv
│   └── methy.csv
├── kidney/                # Kidney Cancer (Classification)
│   ├── exp.csv
│   ├── mirna.csv
│   └── methy.csv
├── liver/                 # Liver Cancer (Classification)
│   ├── exp.csv
│   ├── mirna.csv
│   └── methy.csv
├── lung/                  # Lung Cancer (Classification)
│   ├── exp.csv
│   ├── mirna.csv
│   └── methy.csv
├── melanoma/              # Melanoma (Classification)
│   ├── exp.csv
│   ├── mirna.csv
│   └── methy.csv
├── ovarian/               # Ovarian Cancer (Classification)
│   ├── exp.csv
│   ├── mirna.csv
│   └── methy.csv
├── sarcoma/               # Sarcoma (Regression)
│   ├── exp.csv
│   ├── mirna.csv
│   └── methy.csv
└── clinical/              # Clinical outcome data
    ├── aml.csv
    ├── breast.csv
    ├── colon.csv
    ├── kidney.csv
    ├── liver.csv
    ├── lung.csv
    ├── melanoma.csv
    ├── ovarian.csv
    └── sarcoma.csv
```

## Data Format Requirements

### Omics Data Files (exp.csv, mirna.csv, methy.csv)

Each omics data file should be a CSV file with:
- **Rows**: Samples/patients
- **Columns**: Features (genes, miRNAs, methylation sites)
- **First column**: Sample IDs (must match clinical data)
- **Header**: Feature names

Example format:
```csv
sample_id,feature_1,feature_2,feature_3,...
TCGA-XX-XXXX,0.123,0.456,0.789,...
TCGA-YY-YYYY,0.234,0.567,0.890,...
...
```

### Clinical Data Files (clinical/*.csv)

Clinical outcome files should contain:
- **sample_id**: Patient identifiers (matching omics data)
- **target**: Outcome variable for prediction

#### Regression Datasets
- **AML**: `blast_percentage` (continuous target)
- **Sarcoma**: `tumor_length` (continuous target)

#### Classification Datasets
- **Breast, Colon, Kidney, Liver, Lung, Melanoma**: `pathologic_T` (categorical target)
- **Ovarian**: `clinical_stage` (categorical target)

Example clinical data format:
```csv
sample_id,target,additional_clinical_variables...
TCGA-XX-XXXX,T2,age,gender,stage,...
TCGA-YY-YYYY,T1,age,gender,stage,...
...
```

## Data Sources

The datasets are derived from The Cancer Genome Atlas (TCGA) and include:

### Gene Expression Data (exp.csv)
- **Type**: RNA-seq gene expression
- **Processing**: Log2-transformed, normalized
- **Features**: ~20,000 genes
- **Format**: Continuous values

### miRNA Data (mirna.csv)
- **Type**: miRNA expression profiles
- **Processing**: Log2-transformed, normalized
- **Features**: ~500-1000 miRNAs
- **Format**: Continuous values

### Methylation Data (methy.csv)
- **Type**: DNA methylation beta values
- **Processing**: Beta values (0-1 range)
- **Features**: ~25,000 CpG sites
- **Format**: Continuous values (0-1)

## Data Quality Requirements

### Sample ID Consistency
- All sample IDs must be consistent across modalities and clinical data
- Use TCGA barcode format: `TCGA-XX-XXXX`
- No missing sample IDs

### Missing Values
- Missing values should be represented as `NaN` or empty cells
- The pipeline handles missing values through imputation
- Excessive missing data (>50% per feature) may affect results

### Data Types
- Omics data: Numeric (float)
- Clinical targets: Numeric (regression) or categorical (classification)
- Sample IDs: String

## Data Preprocessing

The pipeline automatically performs:

1. **Sample Alignment**: Matches samples across modalities
2. **Missing Value Handling**: Imputation strategies
3. **Normalization**: Feature scaling and standardization
4. **Quality Control**: Removes low-variance features

## Adding New Datasets

To add a new cancer type:

1. Create a new directory: `data/new_cancer_type/`
2. Add the three omics files: `exp.csv`, `mirna.csv`, `methy.csv`
3. Add clinical data: `clinical/new_cancer_type.csv`
4. Update the configuration in `config.py`
5. Ensure data format matches the requirements above

## Data Privacy and Ethics

- All data should be de-identified
- Follow institutional and ethical guidelines
- Ensure proper data use agreements are in place
- TCGA data usage follows NIH guidelines

## Troubleshooting

### Common Issues

1. **Sample ID Mismatch**: Ensure consistent sample IDs across all files
2. **Missing Files**: All three omics files must be present for each cancer type
3. **Format Errors**: Check CSV format, headers, and data types
4. **Large File Sizes**: Consider data compression or chunked processing

### Validation

Run the installation verification script to check data integrity:
```bash
python setup_and_info/test_installation.py
```

## Contact

For questions about data format or issues with specific datasets, please refer to the main project documentation. 