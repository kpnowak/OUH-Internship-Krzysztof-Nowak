# 📊 **tuner_halving.py Comprehensive Logging Guide**

## 🎯 **Overview**

The enhanced `tuner_halving.py` now includes comprehensive logging functionality that tracks every stage of the hyperparameter tuning process, captures warnings and errors, and provides detailed insights into the pipeline execution.

---

## 🗂️ **Log File Organization**

### **Directory Structure**
```
tuner_logs/
├── tuner_session_YYYYMMDD_HHMMSS.log     # Session-level logs (batch operations)
└── tuner_DATASET_EXTRACTOR_MODEL_YYYYMMDD_HHMMSS.log  # Individual run logs
```

### **Log File Types**

#### **1. Session Logs** (`tuner_session_*.log`)
- **Generated for**: Batch operations (`--all` flag) and general session tracking
- **Contains**: Overall progress, batch summaries, subprocess coordination
- **Example**: `tuner_session_20250617_193728.log`

#### **2. Individual Run Logs** (`tuner_DATASET_EXTRACTOR_MODEL_*.log`)
- **Generated for**: Single model tuning runs (`--single` mode)
- **Contains**: Detailed stage-by-stage execution for specific combinations
- **Example**: `tuner_AML_PCA_ElasticNet_20250617_193728.log`

---

## 📋 **Logging Levels**

### **Available Log Levels**
```bash
--log-level DEBUG    # Most verbose (includes parameter spaces, detailed traces)
--log-level INFO     # Standard logging (default, stage tracking, results)
--log-level WARNING  # Warnings and errors only
--log-level ERROR    # Errors only
```

### **Log Level Details**

#### **DEBUG Level**
- **Parameter spaces**: Full hyperparameter grids
- **Data shapes**: Detailed data dimensions at each stage
- **Subprocess output**: Complete stdout/stderr from subprocesses
- **Internal operations**: Function calls, variable states

#### **INFO Level** (Default)
- **Stage tracking**: Clear progression through pipeline phases
- **Results**: Best scores, hyperparameters, timing information
- **Status updates**: Success/failure notifications
- **File operations**: Saved output files

#### **WARNING Level**
- **Model warnings**: sklearn warnings (e.g., convergence issues)
- **Data quality issues**: NaN/Inf detection and handling
- **Configuration warnings**: Mismatched settings

#### **ERROR Level**
- **Critical failures**: Pipeline crashes, file I/O errors
- **Exception details**: Full tracebacks with context
- **Subprocess failures**: Timeout and execution errors

---

## 🔄 **Stage Tracking**

### **Pipeline Stages Logged**

#### **1. TUNING_INITIALIZATION**
```
STAGE: TUNING_INITIALIZATION
dataset: AML
task: reg
extractor: PCA
model: ElasticNet
seed: 42
cv_folds: 3
```

#### **2. DATA_LOADING**
```
STAGE: DATA_LOADING
pipeline_type: 4-Phase Enhanced Pipeline
phases: [Phase 1: Early Data Quality Assessment, ...]
```

#### **3. DATA_VALIDATION**
```
STAGE: DATA_VALIDATION
✓ Data validation completed successfully
 Found NaN/Inf in preprocessed data, cleaning...
```

#### **4. SAMPLER_SETUP**
```
STAGE: SAMPLER_SETUP
Class distribution: {0: 85, 1: 85}
Using sampler: SMOTE
```

#### **5. PIPELINE_CONSTRUCTION**
```
STAGE: PIPELINE_CONSTRUCTION
Built extractor: PCA
Built model: TransformedTargetRegressor
Created sklearn Pipeline
```

#### **6. SCORER_CV_SETUP**
```
STAGE: SCORER_CV_SETUP
Using scorer: R² score
Using CV strategy: KFold with 3 folds
```

#### **7. PARAMETER_SPACE_GENERATION**
```
STAGE: PARAMETER_SPACE_GENERATION
Enhanced parameter combinations: 216
Parameter space: {'extractor__n_components': [2, 4, 8], ...}
```

#### **8. SEARCH_STRATEGY_SELECTION**
```
STAGE: SEARCH_STRATEGY_SELECTION
Using HalvingRandomSearchCV
```

#### **9. HYPERPARAMETER_SEARCH**
```
STAGE: HYPERPARAMETER_SEARCH
search_type: HalvingRandomSearchCV
n_combinations: 216
n_jobs: 2
backend: threading
```

#### **10. RESULTS_PROCESSING**
```
STAGE: RESULTS_PROCESSING
SAVED hp_best\AML_PCA_ElasticNet.json
Best Score: -0.0228
Search Time: 2.2s
```

#### **11. TUNING_COMPLETED/FAILED**
```
STAGE: TUNING_COMPLETED
status: SUCCESS
best_score: -0.0228313406308492
output_file: hp_best\AML_PCA_ElasticNet.json
total_time_seconds: 2.175243377685547
```

---

## 🚨 **Error Logging**

### **Error Context Tracking**
When errors occur, the system logs comprehensive context:

```
================================================================================
ERROR OCCURRED
================================================================================
Error Type: ValueError
Error Message: n_components=8 must be between 0 and min(n_samples, n_features)=3
Time: 2025-06-17 19:37:56
Context Information:
  operation: model_building
  extractor: PCA
  model: ElasticNet
  combination_number: 2/18
Full Traceback:
[Complete Python traceback...]
================================================================================
```

### **Common Error Types Tracked**
1. **Parameter validation errors** (PCA components vs. data size)
2. **Model convergence issues** (ElasticNet, SVC)
3. **Memory/timeout errors** (Large datasets)
4. **Data quality issues** (NaN/Inf values)
5. **Subprocess failures** (Timeout, crash)

---

## ️ **Warning Capture**

### **sklearn Warnings**
All sklearn warnings are automatically captured:
```
WARNING | py.warnings | R^2 score is not well-defined with less than two samples
WARNING | py.warnings | Convergence warning for ElasticNet
WARNING | py.warnings | PCA components exceed available features
```

### **Custom Warnings**
Pipeline-specific warnings:
```
WARNING | Found NaN/Inf in preprocessed data, cleaning...
WARNING | Skipping sampler due to small class size (5 < 6)
WARNING | Using conservative parameter ranges for small dataset
```

---

## 📊 **Batch Operation Logging**

### **Batch Progress Tracking**
```
STAGE: BATCH_TUNING_INITIALIZATION
dataset: AML
total_combinations: 18
extractors: ['PCA', 'KPCA', 'KPLS', 'FA', 'PLS', 'SparsePLS']
models: ['LinearRegression', 'ElasticNet', 'RandomForestRegressor']
subprocess_isolation: True
timeout_minutes: 30

STAGE: COMBINATION_1_OF_18
extractor: PCA
model: LinearRegression
✓ COMPLETED (1/18): PCA + LinearRegression

STAGE: COMBINATION_2_OF_18
extractor: PCA
model: ElasticNet
✗ FAILED (2/18): PCA + ElasticNet

STAGE: BATCH_TUNING_COMPLETED
total_time_minutes: 15.3
successful_combinations: 16
failed_combinations: 2
success_rate: 88.9%
```

---

## 🛠️ **Usage Examples**

### **1. Single Model with Detailed Logging**
```bash
python tuner_halving.py --dataset AML --extractor PCA --model ElasticNet --log-level DEBUG
```
**Generates**: Detailed single-run log with full parameter spaces and debug info

### **2. Batch Operation with Standard Logging**
```bash
python tuner_halving.py --dataset Breast --all --log-level INFO
```
**Generates**: Session log tracking all 18 combinations with progress updates

### **3. Error-Only Logging for Production**
```bash
python tuner_halving.py --dataset Colon --all --log-level ERROR --no-subprocess
```
**Generates**: Minimal log with only critical errors and failures

### **4. Warning-Level Logging for Monitoring**
```bash
python tuner_halving.py --dataset Kidney --extractor FA --model SVC --log-level WARNING
```
**Generates**: Log with warnings and errors, useful for monitoring issues

---

## 📈 **Log Analysis Tips**

### **Finding Successful Runs**
```bash
grep "TUNING_COMPLETED" tuner_logs/*.log
grep "SUCCESS" tuner_logs/*.log
```

### **Identifying Failures**
```bash
grep "ERROR OCCURRED" tuner_logs/*.log
grep "FAILED" tuner_logs/*.log
```

### **Performance Analysis**
```bash
grep "total_time_seconds" tuner_logs/*.log
grep "Best Score" tuner_logs/*.log
```

### **Warning Analysis**
```bash
grep "WARNING" tuner_logs/*.log
grep "n_components.*must be between" tuner_logs/*.log
```

---

## 🔧 **Advanced Features**

### **1. Automatic Warning Capture**
- All Python warnings are automatically logged to files
- sklearn warnings are preserved with full context
- Custom warnings include pipeline-specific information

### **2. Subprocess Isolation Logging**
- Each subprocess execution is tracked
- Timeout detection and reporting
- Stdout/stderr capture from subprocesses

### **3. Context-Aware Error Reporting**
- Errors include operation context
- Dataset, model, and combination information
- Timing and resource usage data

### **4. Performance Metrics**
- Search time tracking
- Parameter combination counting
- Success/failure rate calculation

---

## 📝 **Log File Format**

### **File Format**
```
TIMESTAMP | LEVEL | LOGGER_NAME | FUNCTION:LINE | MESSAGE
```

### **Example Entry**
```
2025-06-17 19:37:56,163 | INFO | tuner_session | tune:558 | Using HalvingRandomSearchCV
```

### **Components**
- **TIMESTAMP**: Precise timing (millisecond accuracy)
- **LEVEL**: INFO, DEBUG, WARNING, ERROR
- **LOGGER_NAME**: tuner_session or tuner_DATASET_EXTRACTOR_MODEL
- **FUNCTION:LINE**: Source code location
- **MESSAGE**: Detailed information

---

## ✅ **Benefits**

### **1. Complete Traceability**
- Every stage of the pipeline is logged
- Full context for debugging failures
- Performance monitoring and optimization

### **2. Production Monitoring**
- Automated error detection
- Warning trend analysis
- Success rate tracking

### **3. Research Insights**
- Parameter space exploration tracking
- Model performance comparison
- Pipeline optimization opportunities

### **4. Debugging Support**
- Detailed error context
- Step-by-step execution tracking
- Resource usage monitoring

---

## 🎯 **Best Practices**

### **1. Log Level Selection**
- **Development**: Use `DEBUG` for detailed analysis
- **Production**: Use `INFO` for standard monitoring
- **Monitoring**: Use `WARNING` for issue detection
- **Critical Systems**: Use `ERROR` for minimal logging

### **2. Log File Management**
- Regularly archive old log files
- Monitor disk space usage
- Use log rotation for long-running operations

### **3. Error Analysis**
- Review error logs regularly
- Track error patterns and trends
- Use context information for debugging

### **4. Performance Monitoring**
- Track timing information
- Monitor success rates
- Identify bottlenecks and optimization opportunities 