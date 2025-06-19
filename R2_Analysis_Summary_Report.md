# Comprehensive R² Analysis Report: AML Dataset Performance Evaluation

## Executive Summary

This report presents a comprehensive analysis of 486 machine learning experiments conducted on the AML (Acute Myeloid Leukemia) dataset, comparing R² performance across different algorithms, models, integration techniques, and data conditions.

## Key Findings

### 🚨 Critical Performance Issue
All 486 experiments resulted in **negative R² values**, ranging from -0.0731 to -538,230,550.2. This indicates that **all models performed worse than a simple mean predictor**, suggesting fundamental issues with the current modeling approach for this dataset.

### 🏆 Top 10 Best Performing Algorithms

| Rank | R² Score | Algorithm | Model | Features | Missing % | Integration Tech |
|------|----------|-----------|-------|----------|-----------|-----------------|
| 1 | -0.0731 | PLS | RandomForestRegressor | 16 | 0.5% | weighted_concat |
| 2 | -0.0811 | LASSO | RandomForestRegressor | 8 | 0.0% | sum |
| 3 | -0.0876 | LASSO | RandomForestRegressor | 32 | 0.0% | sum |
| 4 | -0.0935 | ICA | RandomForestRegressor | 32 | 0.0% | sum |
| 5 | -0.0939 | ICA | LinearRegression | 32 | 0.5% | weighted_concat |
| 6 | -0.1208 | ICA | RandomForestRegressor | 32 | 0.0% | early_fusion_pca |
| 7 | -0.1237 | NMF | RandomForestRegressor | 16 | 0.0% | early_fusion_pca |
| 8 | -0.1275 | NMF | ElasticNet | 16 | 0.0% | average |
| 9 | -0.1294 | PLS | LinearRegression | 8 | 0.5% | weighted_concat |
| 10 | -0.1307 | PLS | ElasticNet | 8 | 0.5% | weighted_concat |

## Performance Analysis by Component

### 📊 Algorithm Performance Ranking
1. **FA (Factor Analysis)**: Best average performance (-0.45)
2. **ICA (Independent Component Analysis)**: Second best (-0.55)
3. **NMF (Non-negative Matrix Factorization)**: Third (-0.70)
4. **PCA (Principal Component Analysis)**: Fourth (-0.96)
5. **PLS (Partial Least Squares)**: Fifth (-3.45)
6. **ElasticNetFS**: Sixth (-3.86)
7. **f_regressionFS**: Seventh (-4.10)
8. **LASSO**: Eighth (-10.93)
9. **RandomForestFS**: Worst (-31,602,074.56)

### 🤖 Model Performance Ranking
1. **RandomForestRegressor**: Best performance (-0.34 average)
2. **ElasticNet**: Poor performance (-1,694,583.84 average)
3. **LinearRegression**: Worst performance (-8,839,449.01 average)

### 🔗 Integration Technique Performance
1. **average**: Best performance (-20.87)
2. **sum**: Second best (-185.56)
3. **early_fusion_pca**: Third (-350.58)
4. **weighted_concat**: Worst (-7,022,503.12)

### ⚙️ Workflow Comparison
- **Extraction-CV**: Much better performance (-1.22 average)
- **Selection-CV**: Poor performance (-7,900,523 average)

## Data Condition Analysis

### Missing Data Impact
- **0.0% missing**: -3,077,687 average R²
- **0.2% missing**: -2,001,649 average R²
- **0.5% missing**: -6,755,671 average R²

### Feature Count Impact
- **16 features**: Best performance (-931,102 average)
- **8 features**: Second best (-2,770,907 average)
- **32 features**: Worst (-6,832,016 average)

## Critical Issues Identified

### 1. **Severe Model Underperformance**
- All R² values are negative, indicating models perform worse than predicting the mean
- Some experiments show catastrophically poor performance (R² < -1,000,000)

### 2. **Algorithm-Specific Problems**
- **RandomForestFS** shows extremely poor performance with massive negative R² values
- **LASSO** and selection-based methods generally underperform
- **LinearRegression** combined with certain techniques produces very poor results

### 3. **Integration Technique Issues**
- **weighted_concat** consistently produces the worst results
- Simpler integration methods (**average**, **sum**) perform better

### 4. **Workflow Impact**
- **Selection-CV** workflow significantly underperforms compared to **Extraction-CV**

## Recommendations

### Immediate Actions
1. **Data Quality Review**: Investigate potential data leakage, scaling issues, or target variable problems
2. **Model Validation**: Review cross-validation setup and ensure proper train/test splits
3. **Feature Engineering**: Reconsider feature preprocessing and scaling strategies
4. **Target Variable Analysis**: Verify target variable distribution and outliers

### Methodological Improvements
1. **Focus on Extraction-CV**: Prioritize extraction-based workflows over selection-based
2. **Use RandomForestRegressor**: This model consistently outperforms others
3. **Simplify Integration**: Use **average** or **sum** instead of complex integration techniques
4. **Optimal Feature Count**: Target 16 features for best performance
5. **Algorithm Selection**: Prioritize FA, ICA, and NMF algorithms

### Technical Recommendations
1. **Avoid RandomForestFS**: This feature selection method shows catastrophic performance
2. **Minimize weighted_concat**: This integration technique consistently underperforms
3. **Handle Missing Data**: 0.2% missing data shows better performance than 0% or 0.5%

## Visualization Outputs

The analysis generated comprehensive visualizations showing:
- **R² vs Missing Percentage**: Top performers cluster around specific missing data percentages
- **R² vs Training Time**: No clear correlation between training time and performance
- **Performance Distributions**: Clear separation between algorithm families
- **Component Analysis**: Detailed breakdowns by each experimental component

## Files Generated
- `comprehensive_r2_analysis.png`: Complete visualization dashboard
- `top_10_detailed_analysis.csv`: Detailed metrics for best performers
- `comprehensive_performance_statistics.csv`: Statistical summaries by component
- `top_10_r2_algorithms_comparison.csv`: Simple comparison table

## Conclusion

The analysis reveals significant systematic issues with the current modeling approach on the AML dataset. While some configurations perform relatively better than others, the universal negative R² values indicate fundamental problems that require immediate investigation and methodological revision.

**Next Steps**: Focus on data quality assessment, model validation procedures, and systematic debugging of the preprocessing pipeline before proceeding with further experiments. 