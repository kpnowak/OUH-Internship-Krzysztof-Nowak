# Raw Data Summary - Multi-Modal Cancer Genomics Pipeline

## Overview
This table summarizes the key characteristics of the raw data **before any preprocessing** used in the multi-modal cancer genomics pipeline.

## Dataset Summary Table

| **Characteristic** | **Details** |
|-------------------|-------------|
| **Data Source** | The Cancer Genome Atlas (TCGA) |
| **Data Type** | Multi-omics cancer genomics data |
| **Total Datasets** | 9 cancer types |
| **Task Types** | 2 Regression + 7 Classification |
| **Modalities per Dataset** | 3 (Gene Expression, miRNA, Methylation) |
| **Sample ID Format** | TCGA barcode format: `TCGA-XX-XXXX` |
| **File Format** | CSV files with headers |

## Cancer Types and Tasks

| **Cancer Type** | **Task Type** | **Target Variable** | **Target Description** |
|----------------|---------------|-------------------|---------------------|
| **AML** | Regression | `blast_percentage` | Bone marrow blast cell percentage (continuous) |
| **Sarcoma** | Regression | `tumor_length` | Pathologic tumor length (continuous) |
| **Breast** | Classification | `pathologic_T` | Tumor stage classification (T1, T2, T3, T4) |
| **Colon** | Classification | `pathologic_T` | Tumor stage classification (T1, T2, T3, T4) |
| **Kidney** | Classification | `pathologic_T` | Tumor stage classification (T1, T2, T3, T4) |
| **Liver** | Classification | `pathologic_T` | Tumor stage classification (T1, T2, T3, T4) |
| **Lung** | Classification | `pathologic_T` | Tumor stage classification (T1, T2, T3, T4) |
| **Melanoma** | Classification | `pathologic_T` | Tumor stage classification (T1, T2, T3, T4) |
| **Ovarian** | Classification | `clinical_stage` | Clinical stage classification (I, II, III, IV) |

## Data Modalities (Raw Format)

| **Modality** | **Data Type** | **Typical Features** | **Value Range** | **Processing State** |
|-------------|---------------|-------------------|-----------------|-------------------|
| **Gene Expression** | RNA-seq | ~20,000 genes | Continuous (>0) | Log2-transformed, normalized |
| **miRNA** | miRNA expression | ~500-1,000 miRNAs | Continuous (>0) | Log2-transformed, normalized |
| **Methylation** | DNA methylation | ~25,000 CpG sites | Beta values [0-1] | Raw beta values |


## Raw Data Characteristics

| **Aspect** | **Gene Expression** | **miRNA** | **Methylation** |
|-----------|-------------------|-----------|----------------|
| **Typical Sample Count** | 200-800 per cancer type | 200-800 per cancer type | 200-800 per cancer type |
| **Typical Feature Count** | ~20,000 genes | ~500-1,000 miRNAs | ~25,000 CpG sites |
| **Value Distribution** | Log-normal (already log2) | Log-normal (already log2) | Beta distribution [0,1] |
| **Sparsity Level** | Low (~5-10% zeros) | High (~40-60% zeros) | Low (~1-5% zeros) |
| **Missing Values** | Variable (0-30%) | Variable (0-40%) | Variable (0-20%) |
| **Outlier Presence** | High (biological variation) | Very High (sparse data) | Moderate |
| **Data Scale** | Log2 scale (~0-20) | Log2 scale (~0-18) | Bounded [0,1] |

## Data Quality Issues (Pre-Preprocessing)

| **Issue Type** | **Description** | **Frequency** | **Impact** |
|---------------|----------------|---------------|-----------|
| **Missing Values** | NaN or empty cells | 5-40% per feature | Requires imputation |
| **Sample Misalignment** | Different samples across modalities | Common | Intersection optimization needed |
| **ID Format Inconsistency** | Various TCGA ID formats | Very Common | Standardization required |
| **File Corruption** | Malformed CSV headers/data | Occasional | Robust parsing needed |
| **Data Orientation** | Incorrect transposition | Common | Validation and correction |
| **Extreme Outliers** | Biological or technical outliers | Common | Clipping/scaling needed |
| **Zero Inflation** | Many zero values (especially miRNA) | High in miRNA | Specialized handling |
| **Constant Features** | No variation across samples | Rare | Removal required |

## Expected Sample Sizes (Typical)

| **Cancer Type** | **Expected Samples** | **Typical Range** |
|----------------|-------------------|------------------|
| **Breast** | ~1,000 | 800-1,200 |
| **Lung** | ~500 | 400-600 |
| **Colon** | ~400 | 300-500 |
| **Kidney** | ~500 | 400-600 |
| **Liver** | ~350 | 250-400 |
| **Melanoma** | ~450 | 350-550 |
| **Ovarian** | ~600 | 500-700 |
| **AML** | ~150 | 100-200 |
| **Sarcoma** | ~250 | 200-300 |

## Key Preprocessing Requirements

| **Requirement** | **Reason** | **Priority** |
|----------------|-----------|-------------|
| **Sample ID Standardization** | TCGA format variations | Critical |
| **Data Orientation Validation** | Prevent transposition errors | Critical |
| **Missing Value Handling** | Variable missingness patterns | High |
| **Modality-Specific Scaling** | Different data distributions | High |
| **Outlier Management** | Biological/technical outliers | High |
| **Feature Selection** | High-dimensional data | Medium |
| **Sample Intersection** | Multi-modal alignment | Critical |

---

**Note**: This table describes the raw data characteristics **before** any preprocessing. The preprocessing pipeline handles all these issues automatically to produce clean, aligned, and analysis-ready datasets. 