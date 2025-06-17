# Data Quality Analyzer for Pre-Model Training Data

## 🎯 Purpose
This script analyzes the quality of data **right before it gets fed into the ML models** during training. It runs independently from the main pipeline to allow for hyperparameter adjustments without affecting the main training process.

## 📊 What Gets Analyzed

### **All 9 Datasets:**
- **Regression (2)**: AML, Sarcoma
- **Classification (7)**: Colon, Breast, Kidney, Liver, Lung, Melanoma, Ovarian

### **All Algorithm Combinations:**

#### **Regression Algorithms:**
- **Extractors (6)**: PCA, KPLS, FA, PLS, SparsePLS, **KernelPCA-RBF** (feature engineering)
- **Selectors (5)**: ElasticNetFS, RFImportance, VarianceFTest, LASSO, f_regressionFS

#### **Classification Algorithms:**
- **Extractors (6)**: PCA, FA, LDA, PLS-DA, SparsePLS, **SparsePLS-DA** (feature engineering)
- **Selectors (5)**: ElasticNetFS, RFImportance, VarianceFTest, LogisticL1, XGBoostFS

#### **Fusion Techniques (7):**
- **weighted_concat** (default concatenation)
- **learnable_weighted** (learnable weighting)
- **attention_weighted** (attention mechanism)
- **late_fusion_stacking** (meta-learning approach)
- **mkl** (Multiple Kernel Learning)
- **snf** (Similarity Network Fusion)
- **early_fusion_pca** (PCA-based early fusion)

#### **Multiple Configurations:**
- **N_components values**: [32, 64, 128]
- **Missing data scenarios**: [0%, 20%, 50%] (configurable)
- **Train/test splits** for comprehensive evaluation

## 📈 Quality Metrics Calculated

For every combination, the analyzer calculates:
- **Zero appearance percentage** - How much of the data is exactly zero
- **Mean and standard deviation** - Central tendency and spread
- **Min/max values** - Data range
- **Variance** - Data variability
- **Skewness and kurtosis** - Distribution shape
- **Number of features and samples** - Dimensionality
- **Missing value percentage** - Data completeness
- **Outlier percentage** - Data quality issues
- **Percentiles (25th, 50th, 75th)** - Distribution characteristics
- **Interquartile range (IQR)** - Robust spread measure

## 🚀 How to Run

### **Simple Execution:**
```bash
python data_quality_analyzer.py
```

### **Expected Output:**
- **5,670 total algorithm combinations** will be tested (7× increase!)
- Progress logging with detailed information
- Results saved to structured directories

## 📁 Output Structure

```
data_quality_analysis/
├── regression/
│   ├── AML_detailed_analysis.json
│   ├── AML_summary.csv
│   ├── Sarcoma_detailed_analysis.json
│   └── Sarcoma_summary.csv
├── classification/
│   ├── Colon_detailed_analysis.json
│   ├── Colon_summary.csv
│   ├── Breast_detailed_analysis.json
│   ├── Breast_summary.csv
│   └── ... (for all 7 classification datasets)
└── summary/
    ├── overall_data_quality_summary.csv
    └── summary_statistics.csv
```

## 📄 Output Files

### **Detailed JSON Files** (`*_detailed_analysis.json`)
Complete analysis with nested structure showing:
- Dataset metadata
- Stage-by-stage analysis (raw -> fusion -> extraction -> selection)
- Train/test metrics for each combination
- Timestamp and configuration info

### **Summary CSV Files** (`*_summary.csv`)
Flattened tabular format with one row per:
- Dataset × Stage × Scenario × Technique × Process × Method × Split

### **Overall Summary** (`overall_data_quality_summary.csv`)
Combined data from all datasets for cross-dataset analysis

### **Summary Statistics** (`summary_statistics.csv`)
Aggregated statistics grouped by algorithm combinations

## 💡 Key Features

### **✅ Independent Analysis**
- Runs separately from main training pipeline
- No interference with existing hyperparameter settings
- Safe to run alongside main training

### **✅ Comprehensive Coverage**
- Tests **ALL** extraction/selection algorithm combinations
- Includes feature engineering algorithms
- Covers multiple fusion strategies
- Tests different missing data scenarios

### **✅ Critical Timing**
- Analyzes data at the exact point where `final_X_train` and `final_X_val` are passed to models
- Captures quality **right before fitting into the model**
- Reveals data quality issues that directly impact model performance

### **✅ Production-Ready**
- Robust error handling
- Comprehensive logging
- Memory-efficient processing
- Structured output for analysis

##  Configuration

The analyzer automatically:
- Enables feature engineering algorithms
- Uses the same configurations as the main pipeline
- Respects all missing data and fusion upgrade settings
- Follows the same data processing pipeline

## 📋 Use Cases

1. **Algorithm Selection**: Compare data quality across different extraction/selection methods
2. **Hyperparameter Tuning**: Understand how different n_components values affect data quality
3. **Fusion Strategy Evaluation**: Assess which fusion techniques preserve data quality best
4. **Missing Data Impact**: Analyze how missing data scenarios affect final data quality
5. **Dataset Comparison**: Compare data quality characteristics across different cancer types
6. **Quality Monitoring**: Track data quality trends across pipeline modifications

## ⚡ Performance

- **Expected Runtime**: Variable depending on dataset size and algorithm complexity
- **Memory Usage**: Optimized for efficient processing
- **Parallel Processing**: Utilizes available system resources
- **Background Execution**: Can run while other analyses are performed

## 🎯 Critical Point Analysis

This analyzer captures data quality at the **most critical point** in your pipeline - right before the data enters the machine learning models. This is where all preprocessing, fusion, extraction, and selection steps have been completed, making it the perfect place to assess the final data quality that will determine model performance.

---

**Ready to run!** Execute `python data_quality_analyzer.py` to start your comprehensive data quality analysis across all 5,670 algorithm combinations. 