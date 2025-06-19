"""
Enhanced Halving tuner: finds best hyper-params for any dataset with 4-phase preprocessing
and saves them to hp_best/<dataset>_<extractor>_<model>.json

Features:
- 4-Phase Enhanced Pipeline Integration (preprocessing before tuning)
- Subprocess isolation for each combination (prevents memory leaks and crashes)
- Timeout protection (30min per combination)
- Auto-detection of dataset type (regression/classification)
- Support for all cancer datasets from config.py
- Robust error handling and monitoring
- Hyperparameters optimized on the SAME preprocessed data as main pipeline
- Comprehensive logging with stage tracking and error reporting
"""

import json, pathlib, argparse, numpy as np, joblib, time, sys, subprocess, warnings
import logging, gc, signal
from datetime import datetime
from itertools import product
from sklearn.experimental import enable_halving_search_cv  # Enable experimental feature
from sklearn.model_selection import HalvingRandomSearchCV, KFold, StratifiedKFold, GridSearchCV, GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, r2_score, matthews_corrcoef, mean_absolute_error
from data_io import load_dataset_for_tuner

# Optimized loader for already processed data
def load_dataset_for_tuner_optimized(dataset_name, task=None):
    """
    Load dataset with FULL 4-phase preprocessing AND fusion for tuner_halving.py.
    
    This loads the SAME fused and processed data that the main pipeline uses,
    resulting in much smaller feature counts for efficient hyperparameter tuning.
    
    Expected feature counts after 4-phase + fusion:
    - Breast: ~142 features per modality → ~200-500 fused features (vs 20,000+)
    - Colon: ~142 features per modality → ~200-500 fused features
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g., 'AML', 'Breast', etc.)
    task : str
        Task type ('reg' or 'clf')
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Fully processed and fused features (X) and targets (y) as numpy arrays
    """
    import numpy as np
    from config import DatasetConfig
    from enhanced_pipeline_integration import run_enhanced_preprocessing_pipeline
    import logging
    
    logger = logging.getLogger(__name__)
    logger.info(f"Loading dataset {dataset_name} with 4-phase preprocessing AND fusion (task: {task})")
    
    # Determine task type
    if task is None:
        config = DatasetConfig.get_config(dataset_name.lower())
        if not config:
            raise ValueError(f"No configuration found for dataset: {dataset_name}")
        
        outcome_col = config.get('outcome_col', '')
        if 'blast' in outcome_col.lower() or 'length' in outcome_col.lower():
            task_type = 'regression'
        else:
            task_type = 'classification'
    else:
        task_type = 'regression' if task == 'reg' else 'classification'
    
    # Get configuration for the dataset
    config = DatasetConfig.get_config(dataset_name.lower())
    if not config:
        raise ValueError(f"No configuration found for dataset: {dataset_name}")
    
    # Map modality names to short names (same as main pipeline)
    modality_mapping = {
        "Gene Expression": "exp",
        "miRNA": "mirna", 
        "Methylation": "methy"
    }
    
    modality_short_names = []
    for full_name in config['modalities'].keys():
        short_name = modality_mapping.get(full_name, full_name.lower())
        modality_short_names.append(short_name)
    
    # Load raw dataset first
    from data_io import load_dataset
    modalities_data, y_series, common_ids = load_dataset(
        ds_name=dataset_name.lower(),
        modalities=modality_short_names,
        outcome_col=config['outcome_col'],
        task_type=task_type,
        parallel=True,
        use_cache=True
    )
    
    if not modalities_data or y_series is None:
        raise ValueError(f"Failed to load data for {dataset_name}")
    
    logger.info(f"Raw data loaded: {len(modalities_data)} modalities with {len(common_ids)} common samples")
    
    # Convert to enhanced pipeline format: Dict[str, Tuple[np.ndarray, List[str]]]
    modality_data_dict = {}
    for modality_name, modality_df in modalities_data.items():
        # Convert DataFrame to numpy array (transpose to get samples x features)
        X_modality = modality_df.T.values  # modality_df is features x samples
        modality_data_dict[modality_name] = (X_modality, common_ids)
        logger.info(f"  Raw {modality_name} shape: {X_modality.shape}")
    
    # Apply 4-phase enhanced preprocessing pipeline WITH fusion
    fusion_method = "snf" if task_type == "classification" else "weighted_concat"
    logger.info(f"Applying 4-phase preprocessing with {fusion_method} fusion...")
    
    processed_modalities, y_aligned, pipeline_metadata = run_enhanced_preprocessing_pipeline(
        modality_data_dict=modality_data_dict,
        y=y_series.values,
        fusion_method=fusion_method,
        task_type=task_type,
        dataset_name=dataset_name,
        enable_early_quality_check=True,
        enable_fusion_aware_order=True,
        enable_centralized_missing_data=True,
        enable_coordinated_validation=True
    )
    
    logger.info(f"4-phase preprocessing completed with quality score: {pipeline_metadata.get('quality_score', 'N/A')}")
    
    # The processed_modalities should now be fused and ready for tuning
    # Each modality should have been processed and potentially fused together
    if len(processed_modalities) == 1:
        # Single fused modality
        modality_name = list(processed_modalities.keys())[0]
        X = processed_modalities[modality_name]
        logger.info(f"Single fused modality: {modality_name} shape {X.shape}")
    else:
        # Multiple processed modalities - concatenate them (they should be much smaller now)
        X_parts = []
        modality_info = []
        
        for modality_name, modality_array in processed_modalities.items():
            X_parts.append(modality_array)
            modality_info.append(f"{modality_name}: {modality_array.shape[1]} features")
            logger.info(f"  Processed {modality_name} shape: {modality_array.shape}")
        
        # Concatenate horizontally (samples x all_processed_features)
        X = np.concatenate(X_parts, axis=1)
        logger.info(f"Concatenated processed modalities: {', '.join(modality_info)}")
    
    y = y_aligned
    
    logger.info(f"Dataset {dataset_name} loaded with 4-phase preprocessing + fusion:")
    logger.info(f"  Final X shape: {X.shape}")
    logger.info(f"  Final y shape: {y.shape}")
    logger.info(f"  Fusion method: {fusion_method}")
    logger.info(f"  Quality score: {pipeline_metadata.get('quality_score', 'N/A')}")
    
    # Calculate original feature count
    original_features = sum(arr[0].shape[1] for arr in modality_data_dict.values())
    logger.info(f"  Feature reduction: {original_features} → {X.shape[1]} features ({original_features - X.shape[1]} reduced)")
    
    # Final validation
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        logger.warning("Found NaN/Inf values in processed data, cleaning...")
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        logger.warning("Found NaN/Inf values in target, cleaning...")
        y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return X, y
from samplers import safe_sampler
from models import build_extractor, build_model
from sklearn.preprocessing import PowerTransformer
from sklearn.base import BaseEstimator, TransformerMixin

# Suppress specific sklearn warnings about undefined R² scores and non-finite values
warnings.filterwarnings('ignore', 
                       message=r'R\^2 score is not well-defined with less than two samples.',
                       category=UserWarning,
                       module='sklearn.metrics._regression')

# Suppress warnings about non-finite test scores during hyperparameter search
warnings.filterwarnings('ignore',
                       message=r'One or more of the test scores are non-finite.*',
                       category=UserWarning,
                       module='sklearn.model_selection._search')

# Suppress convergence warnings for small datasets
warnings.filterwarnings('ignore',
                       message=r'.*did not converge.*',
                       category=UserWarning,
                       module='sklearn')

# Suppress linalg warnings for singular matrices in small datasets
warnings.filterwarnings('ignore',
                       message=r'.*Singular matrix.*',
                       category=RuntimeWarning,
                       module='scipy.linalg')

warnings.filterwarnings('ignore',
                       message=r'.*Matrix is not positive definite.*',
                       category=RuntimeWarning,
                       module='scipy.linalg')

# Create directories
HP_DIR = pathlib.Path("hp_best"); HP_DIR.mkdir(exist_ok=True)
LOG_DIR = pathlib.Path("tuner_logs"); LOG_DIR.mkdir(exist_ok=True)

SEED   = 42
N_ITER = 32          # candidates; Halving keeps ~⅓ each rung
CV_INNER = 3
TIMEOUT_MINUTES = 30  # Timeout per combination
SEARCH_TIMEOUT_MINUTES = 10  # Timeout for individual hyperparameter search

# Minimum samples per fold for reliable metrics
MIN_SAMPLES_PER_FOLD = 5

# Initialize a default logger for cases where logger is not available
logger = logging.getLogger(__name__)

# Setup comprehensive logging
def setup_logging(dataset_name=None, extractor=None, model=None, log_level=logging.INFO):
    """
    Setup comprehensive logging for tuner_halving.py with stage tracking.
    
    Creates both a general log file and specific log files for individual runs.
    """
    # Create timestamp for this session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine log file names
    if dataset_name and extractor and model:
        # Specific log for individual tuning run
        log_filename = f"tuner_{dataset_name}_{extractor}_{model}_{timestamp}.log"
        logger_name = f"tuner_{dataset_name}_{extractor}_{model}"
    else:
        # General log for overall session
        log_filename = f"tuner_session_{timestamp}.log"
        logger_name = "tuner_session"
    
    log_path = LOG_DIR / log_filename
    
    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s'
    )
    
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s'
    )
    
    # File handler (detailed logging)
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Console handler (less verbose)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Capture warnings
    logging.captureWarnings(True)
    warnings_logger = logging.getLogger('py.warnings')
    warnings_logger.addHandler(file_handler)
    
    logger.info(f"Logging initialized: {log_path}")
    logger.info(f"Session started at: {datetime.now()}")
    
    return logger, log_path

def log_stage(logger, stage_name, details=None, level=logging.INFO):
    """
    Log a specific stage of the tuning process with optional details.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger instance
    stage_name : str
        Name of the current stage
    details : dict, optional
        Additional details to log
    level : int
        Logging level
    """
    separator = "=" * 60
    logger.log(level, f"\n{separator}")
    logger.log(level, f"STAGE: {stage_name}")
    logger.log(level, f"TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if details:
        for key, value in details.items():
            logger.log(level, f"{key}: {value}")
    
    logger.log(level, separator)

def log_error_with_context(logger, error, context=None):
    """
    Log an error with additional context information.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger instance
    error : Exception
        The error that occurred
    context : dict, optional
        Additional context information
    """
    import traceback
    
    logger.error("=" * 80)
    logger.error("ERROR OCCURRED")
    logger.error("=" * 80)
    logger.error(f"Error Type: {type(error).__name__}")
    logger.error(f"Error Message: {str(error)}")
    logger.error(f"Time: {datetime.now()}")
    
    if context:
        logger.error("Context Information:")
        for key, value in context.items():
            logger.error(f"  {key}: {value}")
    
    logger.error("Full Traceback:")
    logger.error(traceback.format_exc())
    logger.error("=" * 80)

# Available datasets and their tasks (from config.py)
DATASET_INFO = {
    # Regression datasets (survival/continuous outcomes) 
    "AML": "reg",      # lab_procedure_bone_marrow_blast_cell_outcome_percent_value
    "Sarcoma": "reg",  # pathologic_tumor_length
    
    # Classification datasets
    "Colon": "clf",    # pathologic_T
    "Breast": "clf",   # pathologic_T  
    "Kidney": "clf",   # pathologic_T
    "Liver": "clf",    # pathologic_T
    "Lung": "clf",     # pathologic_T
    "Melanoma": "clf", # pathologic_T
    "Ovarian": "clf"   # clinical_stage
}

# Available extractors and models by task (aligned with main pipeline)
REGRESSION_EXTRACTORS = ["PCA", "KPCA", "KPLS", "FA", "PLS", "SparsePLS"]  # 6 extractors
REGRESSION_MODELS = ["LinearRegression", "ElasticNet", "RandomForestRegressor"]
CLASSIFICATION_EXTRACTORS = ["PCA", "KPCA", "FA", "LDA", "PLS-DA", "SparsePLS"]  # 6 extractors
CLASSIFICATION_MODELS = ["LogisticRegression", "SVC", "RandomForestClassifier"]

# ------------- Dataset type detection ----------------------
def detect_dataset_task(dataset):
    """Auto-detect if dataset is regression or classification from config.py."""
    if dataset in DATASET_INFO:
        return DATASET_INFO[dataset]
    
    # Try to load configuration from config.py
    try:
        from config import DatasetConfig
        config = DatasetConfig.get_config(dataset.lower())
        if config:
            outcome_type = config.get('outcome_type', 'class')
            if outcome_type in ['continuous', 'survival']:
                return "reg"
            else:
                return "clf"
    except Exception as e:
        print(f"Warning: Could not auto-detect task for {dataset}: {e}")
    
    # Default to regression if uncertain
    return "reg"

# ------------- Enhanced search space for 4-phase preprocessed data ----------------------------------
def param_space(extr, mdl, X_shape=None):
    """
    Ultra-conservative parameter space for Windows resource management.
    
    Since data is now fully preprocessed with:
    - Phase 1: Early Data Quality Assessment
    - Phase 2: Fusion-Aware Preprocessing  
    - Phase 3: Centralized Missing Data Management
    - Phase 4: Coordinated Validation Framework
    
    We use ultra-conservative parameter ranges to prevent Windows system crashes.
    
    Parameters
    ----------
    extr : str
        Extractor name
    mdl : str
        Model name
    X_shape : tuple, optional
        Shape of the data (n_samples, n_features) for adaptive parameters
    """
    p = {}
    
    # No more high-risk restrictions needed due to proper preprocessing reducing features dramatically
    is_high_risk = False  # All combinations are now safe with 700-feature data instead of 20,000+
    
    # Adaptive component selection based on actual data dimensions
    if X_shape is not None:
        n_samples, n_features = X_shape
        
        # Calculate cross-validation fold size for safety constraints
        cv_fold_size = n_samples // CV_INNER
        
        # Reasonable component ranges now that we have properly preprocessed data (~700 features)
        # Much more aggressive than before since feature count is 96.5% smaller
        if n_samples >= 150:
            # Good range for larger datasets: [4, 8, 16, 32]
            base_components = [4, 8, 16, 32]
        else:
            # Standard range for smaller datasets: [2, 4, 8, 16]  
            base_components = [2, 4, 8, 16]
        
        # Apply safety constraints while maintaining reasonable minimums
        # For KPCA and kernel methods, be more conservative due to numerical issues
        min_cv_samples = max(1, cv_fold_size // 2)  # More conservative for stability
        max_safe_components = min(min_cv_samples - 1, n_features // 2, 32)  # More conservative cap
        
        # Special handling for kernel methods that are prone to numerical issues
        if extr in {"KPCA", "KPLS"}:
            # Be extra conservative with kernel methods
            max_safe_components = min(max_safe_components, cv_fold_size // 3, 16)
            
        # Filter components based on constraints, but keep at least one valid option
        component_options = [c for c in base_components if c <= max_safe_components]
        
        # Fallback logic - ensure we always have at least one reasonable option
        if not component_options:
            if max_safe_components >= 4:
                component_options = [4]
            elif max_safe_components >= 2:
                component_options = [2]
            else:
                component_options = [1]  # Last resort for very small datasets
        
        # Add smaller component as fallback only if needed
        if component_options and min(component_options) > 4 and max_safe_components >= 4:
            component_options = [4] + component_options
            
        # Log component selection reasoning
        current_logger = logging.getLogger(__name__)
        current_logger.debug(f"Component selection for {extr}: samples={n_samples}, features={n_features}, "
                    f"cv_fold_size={cv_fold_size}, max_safe={max_safe_components}, "
                    f"selected={component_options}")
    else:
        # Default ranges when no shape provided - use reasonable baseline
        component_options = [2, 4, 8, 16]  # Standard range for preprocessed data
    
    # Extractor parameters - adaptive for preprocessed data
    # Note: Parameters now have extractor__extractor__ prefix due to SafeExtractorWrapper
    if extr in {"PCA","KPCA","KPLS","PLS","SparsePLS","PLS-DA","Sparse PLS-DA"}:
        p["extractor__extractor__n_components"] = component_options
    
    if extr in {"KPCA","KPLS"}:
        # Conservative gamma range for kernel methods to avoid numerical issues
        # Avoid very small gamma values that can cause singular matrices
        if X_shape is not None:
            n_samples = X_shape[0]
            # Scale gamma based on data size - smaller datasets need larger gamma
            if n_samples < 100:
                # For small datasets, use more conservative gamma range
                p["extractor__extractor__gamma"] = [0.01, 0.1, 1.0]
            else:
                # For larger datasets, can use wider range but still conservative
                p["extractor__extractor__gamma"] = [0.001, 0.01, 0.1, 1.0]
        else:
            # Default conservative range
            p["extractor__extractor__gamma"] = [0.01, 0.1, 1.0]
    
    if extr in {"SparsePLS","Sparse PLS-DA"}:
        # Conservative alpha values for sparse methods to reduce resource usage
        p["extractor__extractor__alpha"] = np.logspace(-1, 0, 3)  # Further reduced: 3 values instead of 4
    
    # Reasonable parameters for FA with preprocessed data
    if extr == "FA":
        p["extractor__extractor__n_components"] = component_options
        p["extractor__extractor__max_iter"] = [1000, 3000]  # Restored to 2 options
        p["extractor__extractor__tol"] = [1e-3, 1e-2]  # Restored to 2 options
    
    # LDA parameters with compatibility fixes for preprocessed data
    if extr == "LDA":
        # Use lsqr solver which supports shrinkage for better hyperparameter exploration
        p["extractor__extractor__solver"] = ["lsqr", "svd"]  # Both solvers
        p["extractor__extractor__shrinkage"] = [None, "auto"]  # Shrinkage options (only with lsqr)
    
    # Model parameters - enhanced for preprocessed data
    if mdl == "RandomForestRegressor":
        # RandomForestRegressor now uses OptimizedExtraTreesRegressor
        if X_shape is not None and X_shape[0] < 50:  # Small sample size
            p.update({
                "model__n_estimators": [50, 100, 150],  # Fewer estimators for small data
                "model__max_features": ["sqrt", "log2", 0.5],  # Feature sampling options
                "model__bootstrap": [False, True],  # Extra Trees typically uses False
                "model__min_samples_split": [2, 5, 10],  # Conservative splits
                "model__min_samples_leaf": [1, 2, 3],  # Prevent overfitting
                "model__max_depth": [3, 5, None],  # Shallow to moderate depth
            })
        else:
            p.update({
                "model__n_estimators": [100, 200, 300],  # More estimators for larger data
                "model__max_features": ["sqrt", "log2", 0.3, 0.5],  # More feature sampling options
                "model__bootstrap": [False, True],  # Extra Trees typically uses False
                "model__min_samples_split": [2, 5, 10],  # Various split options
                "model__min_samples_leaf": [1, 2],  # Less restrictive for larger data
                "model__max_depth": [None, 10, 15],  # Deeper trees for larger data
            })
    
    if mdl == "RandomForestClassifier":
        # Classification RandomForest (unchanged)
        if X_shape is not None and X_shape[0] < 50:  # Small sample size
            p.update({
                "model__n_estimators": [50, 100],  # Fewer estimators for small data
                "model__max_depth": [None, 3],  # Shallower trees
                "model__min_samples_leaf": [1, 2],  # Less restrictive
                "model__min_samples_split": [2],  # Less restrictive
                "model__max_features": ["sqrt"]  # Simple feature sampling
            })
        else:
            p.update({
                "model__n_estimators": [100, 200],  # Restored to 2 options
                "model__max_depth": [None, 5, 10],  # Restored to 3 options
                "model__min_samples_leaf": [1, 2],  # Restored to 2 options
                "model__min_samples_split": [2, 5],  # Restored to 2 options  
                "model__max_features": ["sqrt", "log2"]  # Restored to 2 options
            })
    
    if mdl in {"ElasticNet"}:
        # ElasticNet now uses SelectionByCyclicCoordinateDescent with automatic alpha search
        p.update({
            "model__l1_ratio": np.linspace(0.1, 0.9, 5),  # L1/L2 mixing parameter
            "model__cv": [3, 5],  # Cross-validation folds for alpha selection
            "model__n_alphas": [50, 100],  # Number of alphas to try
            "model__eps": [1e-3, 1e-4],  # Alpha grid spacing
            "model__max_iter": [1000, 2000],  # Maximum iterations
        })
    
    if mdl == "SVC":
        # Reasonable SVC parameters now that we have properly preprocessed data
        p.update({
            "model__C": np.logspace(-1, 1, 4),  # Restored to 4 C values  
            "model__gamma": np.logspace(-3, 0, 4),  # Restored to 4 gamma values
            "model__kernel": ["rbf", "linear"]  # Both kernels since data is much smaller
        })
    
    # Enhanced parameters for LinearRegression (now RobustLinearRegressor)
    if mdl == "LinearRegression":
        p.update({
            "model__method": ["huber", "ransac"],  # Robust regression methods
            "model__epsilon": [1.35, 1.5, 2.0],  # Huber parameter (for huber method)
            "model__alpha": [0.0001, 0.001, 0.01],  # Regularization parameter
            "model__max_iter": [1000, 2000],  # Maximum iterations
        })
        # Remove positive constraint for small datasets as it can be too restrictive
    
    # Enhanced parameters for LogisticRegression
    if mdl == "LogisticRegression":
        p.update({
            "model__C": np.logspace(-1, 1, 4),  # Smaller C range
            "model__penalty": ["l1", "l2"],  # Stable penalties
            "model__solver": ["liblinear", "saga"],  # Stable solvers
            "model__max_iter": [1000, 2000],  # Sufficient iterations
            "model__class_weight": [None, "balanced"]  # Class weighting
        })
    

    
    return p

def count_parameter_combinations(param_dict):
    """Count total number of parameter combinations."""
    if not param_dict:
        return 1
    
    total = 1
    for param_values in param_dict.values():
        total *= len(param_values)
    return total

# ------------- subprocess isolation ----------------------
def run_tuning_subprocess(dataset, task, extractor, model, logger=None):
    """Run tuning for a single combination in subprocess with timeout and logging."""
    if logger is None:
        logger = logging.getLogger("tuner_session")
    
    log_stage(logger, "SUBPROCESS_EXECUTION", {
        "dataset": dataset,
        "task": task,
        "extractor": extractor,
        "model": model,
        "timeout_minutes": TIMEOUT_MINUTES
    })
    
    cmd = [
        sys.executable, __file__,
        "--dataset", dataset,
        "--task", task,
        "--extractor", extractor,
        "--model", model,
        "--single"  # Flag to indicate single combination mode
    ]
    
    logger.info(f"Executing subprocess command: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_MINUTES*60)
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            logger.info(f"SUCCESS ({elapsed:.1f}s): {extractor} + {model}")
            if result.stdout.strip():
                logger.debug(f"Subprocess stdout:\n{result.stdout}")
        else:
            logger.error(f"FAILED ({elapsed:.1f}s): {extractor} + {model}")
            if result.stderr.strip():
                logger.error(f"Subprocess stderr:\n{result.stderr}")
            if result.stdout.strip():
                logger.debug(f"Subprocess stdout:\n{result.stdout}")
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        logger.error(f"TIMEOUT ({elapsed/60:.1f}min): {extractor} + {model}")
        logger.error(f"Process exceeded {TIMEOUT_MINUTES} minute timeout")
        return False
    except Exception as e:
        elapsed = time.time() - start_time
        log_error_with_context(logger, e, {
            "operation": "subprocess_execution",
            "dataset": dataset,
            "extractor": extractor,
            "model": model,
            "elapsed_time": f"{elapsed:.1f}s"
        })
        return False

# ------------- main tune routine with 4-phase preprocessing -----------------------------
def tune(dataset, task, extractor, model, logger=None):
    """Enhanced hyperparameter tuning with 4-phase pipeline integration and Windows resource management."""
    
    # Setup logging if not provided
    if logger is None:
        logger = setup_logging(dataset, extractor, model)
    
    # Monitor system resources on Windows
    initial_memory = None
    try:
        import psutil
        initial_memory = psutil.virtual_memory().percent
        logger.info(f"Initial memory usage: {initial_memory:.1f}%")
        
        # Safety check: abort if memory usage is already too high
        if initial_memory > 97.0:
            logger.error(f"Memory usage too high ({initial_memory:.1f}%) - aborting to prevent system crash")
            return False
            
    except ImportError:
        logger.debug("psutil not available for resource monitoring")
    
    # Suppress sklearn warnings at the start
    import warnings
    warnings.filterwarnings("ignore", 
                          message="The least populated class in y has only .* members", 
                          category=UserWarning)
    warnings.filterwarnings("ignore", 
                          message=".*stratified.*", 
                          category=UserWarning)
    
    # Stage 1: Initialization and Validation
    log_stage(logger, "TUNING_INITIALIZATION", {
        "dataset": dataset,
        "task": task,
        "extractor": extractor,
        "model": model,
        "seed": SEED,
        "cv_folds": CV_INNER
    })
    
    try:
        # Stage 1: Data Loading
        log_stage(logger, "DATA_LOADING", {
            "pipeline_type": "4-Phase Enhanced Pipeline",
            "phases": [
                "Phase 1: Early Data Quality Assessment",
                "Phase 2: Fusion-Aware Preprocessing",
                "Phase 3: Centralized Missing Data Management", 
                "Phase 4: Coordinated Validation Framework"
            ]
        })
        
        logger.info(f"Loading {dataset} with 4-Phase Enhanced Pipeline...")
        
        # Load data with FULL 4-phase preprocessing AND fusion (same as main pipeline)
        X, y = load_dataset_for_tuner_optimized(dataset, task=task)
        
        # Try to get real sample IDs for enhanced CV
        sample_ids = None
        try:
            # Load the dataset again to get sample IDs
            from data_io import load_dataset
            from config import DatasetConfig
            
            config = DatasetConfig.get_config(dataset.lower())
            if config:
                # Map modality names to short names
                modality_mapping = {
                    "Gene Expression": "exp",
                    "miRNA": "mirna", 
                    "Methylation": "methy"
                }
                
                modality_short_names = []
                for full_name in config['modalities'].keys():
                    short_name = modality_mapping.get(full_name, full_name.lower())
                    modality_short_names.append(short_name)
                
                # Load dataset to get sample IDs
                modalities_data, y_series, common_ids = load_dataset(
                    ds_name=dataset.lower(),
                    modalities=modality_short_names,
                    outcome_col=config['outcome_col'],
                    task_type=task,
                    parallel=True,
                    use_cache=True
                )
                
                if common_ids and len(common_ids) == len(y):
                    sample_ids = common_ids
                    logger.info(f"Loaded {len(sample_ids)} real sample IDs for enhanced CV")
                
        except Exception as e:
            logger.debug(f"Could not load real sample IDs: {e}, will use fallback approach")
        
        logger.info(f"Data loaded successfully:")
        logger.info(f"  X shape: {X.shape}")
        logger.info(f"  y shape: {y.shape}")
        logger.info(f"  Features after 4-phase preprocessing: {X.shape[1]}")
        
        # Compute baseline MAE for regression tasks
        baseline_mae = None
        if task == "reg":
            baseline_mae = compute_baseline_mae(y)
            logger.info(f"  Baseline MAE (mean prediction): {baseline_mae:.4f}")
        
        # Stage 2: Data Validation
        log_stage(logger, "DATA_VALIDATION")
        
        # Validate data quality after preprocessing
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            logger.warning("Found NaN/Inf in preprocessed data, cleaning...")
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            logger.warning("Found NaN/Inf in targets, cleaning...")
            y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
        
        logger.info("Data validation completed successfully")
        
        # Apply target outlier removal for regression tasks
        if task == "reg":
            try:
                original_size = len(y)
                
                # Remove extreme outliers (>97.5th percentile) from the dataset
                # Note: In tuner, we apply this to the full dataset since we don't have separate train/test splits yet
                outlier_threshold = np.percentile(y, 97.5)
                outlier_mask = y <= outlier_threshold
                
                # Count outliers for logging
                n_outliers = np.sum(~outlier_mask)
                outlier_percentage = (n_outliers / original_size) * 100
                
                if n_outliers > 0:
                    # Filter both features and targets
                    X = X[outlier_mask]
                    y = y[outlier_mask]
                    
                    # Update sample IDs if available
                    if sample_ids is not None and len(sample_ids) == original_size:
                        sample_ids = [sample_ids[i] for i, keep in enumerate(outlier_mask) if keep]
                        logger.info(f"Updated sample IDs after outlier removal: {len(sample_ids)} samples")
                    
                    logger.info(f"Removed {n_outliers} extreme outliers (>{outlier_threshold:.2f}) "
                               f"from dataset ({outlier_percentage:.1f}% of data)")
                    logger.info(f"Dataset size: {original_size} → {len(y)}")
                    
                    # Recompute baseline MAE after outlier removal
                    baseline_mae = compute_baseline_mae(y)
                    logger.info(f"Updated baseline MAE after outlier removal: {baseline_mae:.4f}")
                    
                    # Ensure we still have enough samples (need at least 15 samples for 3 folds)
                    min_required_samples = 15
                    if len(y) < min_required_samples:
                        logger.error(f"Insufficient samples after outlier removal: {len(y)} < {min_required_samples}")
                        return False
                else:
                    logger.debug("No extreme outliers detected in targets")
                    
                # Log final data shapes after outlier removal
                logger.info(f"Final data shapes after outlier removal:")
                logger.info(f"  X shape: {X.shape}")
                logger.info(f"  y shape: {y.shape}")
                
            except Exception as e:
                logger.warning(f"Target outlier removal failed: {e}")
                # Continue without outlier removal if it fails
        
        # Stage 3: Sampler Setup
        log_stage(logger, "SAMPLER_SETUP")
        
        sampler = None
        if task == "clf":
            try:
                # Check class distribution first
                unique_classes, class_counts = np.unique(y, return_counts=True)
                min_class_size = class_counts.min()
                n_classes = len(class_counts)
                
                logger.info(f"Class distribution: {dict(zip(unique_classes, class_counts))}")
                logger.info(f"Minimum class size: {min_class_size}")
                
                # Very conservative threshold for sampling:
                # Need at least CV_INNER * 4 samples per class to ensure robust CV
                # This accounts for worst-case CV splits and SMOTE neighbor requirements
                conservative_threshold = CV_INNER * 4
                
                # Additional check: if we have many classes with small sizes, be extra careful
                small_classes = np.sum(class_counts < conservative_threshold)
                if small_classes > n_classes / 2:  # More than half the classes are small
                    logger.warning(f"Too many small classes ({small_classes}/{n_classes}) - disabling sampling")
                    sampler = None
                elif min_class_size >= conservative_threshold:
                    sampler = safe_sampler(y)
                    if sampler is not None:
                        logger.info(f"Using sampler: {type(sampler).__name__}")
                    else:
                        logger.info("No sampler needed/available")
                else:
                    logger.warning(f"Skipping sampler due to small class size ({min_class_size} < {conservative_threshold})")
                    logger.info("Small classes may not have enough samples for robust SMOTE during CV")
                    sampler = None
            except Exception as e:
                log_error_with_context(logger, e, {"operation": "sampler_creation"})
                sampler = None

        # Stage 4: Pipeline Construction
        log_stage(logger, "PIPELINE_CONSTRUCTION")
        
        # Build pipeline steps
        steps = []
        if sampler: 
            steps.append(("sampler", sampler))
            logger.info("Added sampler to pipeline")
        
        try:
            extractor_obj = build_extractor(extractor)
            model_obj = build_model(model, task)
            
            # Wrap extractor with safe dimensionality checking
            safe_extractor = SafeExtractorWrapper(extractor_obj)
            
            steps.extend([
                ("extractor", safe_extractor),
                ("model", model_obj)
            ])
            
            logger.info(f"Built extractor: {type(extractor_obj).__name__} (wrapped for safety)")
            logger.info(f"Built model: {type(model_obj).__name__}")
            
        except Exception as e:
            log_error_with_context(logger, e, {
                "operation": "model_building",
                "extractor": extractor,
                "model": model
            })
            return False
        
        # Use imblearn Pipeline if we have a sampler, otherwise use sklearn Pipeline
        if sampler:
            from imblearn.pipeline import Pipeline as ImbPipeline
            pipe = ImbPipeline(steps)
            logger.info("Created imblearn Pipeline with sampler")
        else:
            pipe = Pipeline(steps)
            logger.info("Created sklearn Pipeline")

        # Stage 5: Scorer and CV Setup
        log_stage(logger, "SCORER_CV_SETUP")
        
        # Choose primary scorer and setup secondary scoring for regression
        if task == "reg":
            # Primary scorer: R² (optimize for this)
            scorer = make_scorer(safe_r2_score, greater_is_better=True)
            # Secondary scorer: MAE (track but don't optimize)
            mae_scorer = make_scorer(safe_mae_score, greater_is_better=True)
            scoring = {'r2': scorer, 'mae': mae_scorer}
            refit_scorer = 'r2'  # Optimize for R²
            logger.info("Using primary scorer: R² score (optimized)")
            logger.info("Using secondary scorer: MAE (tracked)")
        else:
            scorer = make_scorer(safe_mcc_score, greater_is_better=True)
            scoring = scorer
            refit_scorer = True
            logger.info("Using scorer: Matthews correlation coefficient")

        # Enhanced cross-validation strategy with stratified regression and grouped CV
        n_samples = len(y)
        
        # Use enhanced CV splitter with real sample IDs
        try:
            from cv import create_enhanced_cv_splitter, validate_enhanced_cv_strategy, MIN_SAMPLES_PER_FOLD
            
            task_type = 'regression' if task == 'reg' else 'classification'
            max_safe_folds = max(2, min(CV_INNER, n_samples // MIN_SAMPLES_PER_FOLD))
            
            # Create enhanced CV splitter
            cv_result = create_enhanced_cv_splitter(
                y=y,
                sample_ids=sample_ids,
                task_type=task_type,
                n_splits=max_safe_folds,
                use_stratified_regression=True,
                use_grouped_cv=True,
                random_state=SEED
            )
            
            # Unpack result
            if len(cv_result) == 4:
                cv_inner, strategy_desc, y_for_cv, groups = cv_result
            else:
                cv_inner, strategy_desc = cv_result[:2]
                y_for_cv = y
                groups = None
            
            # Validate the enhanced strategy
            if validate_enhanced_cv_strategy(cv_inner, y_for_cv, groups, max_safe_folds, task_type):
                logger.info(f"Enhanced CV strategy: {strategy_desc}")
                logger.info(f"Dataset: {n_samples} samples, {max_safe_folds} folds")
                
                # Log additional info for grouped CV
                if groups is not None:
                    n_groups = len(np.unique(groups))
                    n_replicates = n_samples - n_groups
                    logger.info(f"Grouped CV: {n_groups} patient groups, {n_replicates} replicates")
                
                # Log additional info for stratified regression
                if task == 'reg' and hasattr(cv_inner, '__class__') and 'Stratified' in cv_inner.__class__.__name__:
                    unique_bins, bin_counts = np.unique(y_for_cv, return_counts=True)
                    logger.info(f"Stratified regression: {len(unique_bins)} quartile bins with counts {bin_counts}")
                    
            else:
                logger.warning("Enhanced CV validation failed, falling back to standard approach")
                raise Exception("Enhanced CV validation failed")
                
        except Exception as e:
            logger.warning(f"Enhanced CV failed: {e}, using standard approach")
            
            # Fallback to standard CV approach
            if task == "reg":
                max_safe_folds = max(2, min(CV_INNER, n_samples // MIN_SAMPLES_PER_FOLD))
                cv_inner = KFold(max_safe_folds, shuffle=True, random_state=SEED)
                logger.info(f"Fallback: Using KFold CV with {max_safe_folds} folds (dataset: {n_samples} samples)")
            else:
                # For classification, check minimum class sizes
                unique_classes, class_counts = np.unique(y, return_counts=True)
                min_class_size = class_counts.min()
                
                # Ensure each fold has at least MIN_SAMPLES_PER_FOLD samples from each class
                max_safe_folds = max(2, min(CV_INNER, min_class_size // 2))
                cv_inner = StratifiedKFold(max_safe_folds, shuffle=True, random_state=SEED)
                logger.info(f"Fallback: Using StratifiedKFold CV with {max_safe_folds} folds")
                logger.info(f"Dataset: {n_samples} samples, min class size: {min_class_size}")
        
        # Log safety information
        estimated_min_fold_size = n_samples // cv_inner.n_splits
        if estimated_min_fold_size < MIN_SAMPLES_PER_FOLD:
            logger.warning(f"Small fold size detected (~{estimated_min_fold_size} samples/fold). "
                          f"Using safe scorers to prevent undefined metrics.")

        # Stage 6: Parameter Space Generation
        log_stage(logger, "PARAMETER_SPACE_GENERATION")
        
        # Get enhanced parameter space for preprocessed data
        params = param_space(extractor, model, X.shape)
        n_combinations = count_parameter_combinations(params)
        
        logger.info(f"Enhanced parameter combinations: {n_combinations}")
        logger.debug(f"Parameter space: {params}")
        
        # Stage 7: Search Strategy Selection
        log_stage(logger, "SEARCH_STRATEGY_SELECTION")
        
        # Choose search strategy based on parameter space size and dataset characteristics
        use_halving = (n_combinations > 20) and (estimated_min_fold_size >= MIN_SAMPLES_PER_FOLD)
        
        if not use_halving:
            logger.info("Using GridSearchCV (exhaustive search)")
            logger.info(f"Reason: {'Small parameter space' if n_combinations <= 20 else 'Small dataset - halving not suitable'}")
            search = GridSearchCV(
                estimator=pipe,
                param_grid=params,
                scoring=scoring,
                cv=cv_inner,
                refit=refit_scorer,
                n_jobs=1,  # Single job to prevent Windows resource exhaustion
                verbose=1
            )
        else:
            logger.info("Using HalvingRandomSearchCV with enhanced settings")
            
            # Enhanced halving configuration - use full dataset in later rungs
            min_resources_per_fold = MIN_SAMPLES_PER_FOLD * cv_inner.n_splits
            # Use full dataset size as max_resources to prevent judging models on small samples
            full_dataset_resources = n_samples
            
            logger.info(f"Enhanced halving configuration:")
            logger.info(f"  - Min resources per fold: {min_resources_per_fold}")
            logger.info(f"  - Max resources (full dataset): {full_dataset_resources}")
            logger.info(f"  - Factor: 2 (conservative, later rungs see full dataset)")
            
            # HalvingRandomSearchCV doesn't support multiple scorers like GridSearchCV
            # So we use the primary scorer only and compute MAE separately
            primary_scorer = scorer if task == "clf" else make_scorer(safe_r2_score, greater_is_better=True)
            
            search = HalvingRandomSearchCV(
                estimator = pipe,
                param_distributions = params,
                n_candidates="exhaust",
                factor = 2,  # Keep factor=2 as specified
                resource = "n_samples",
                max_resources = full_dataset_resources,  # Use full dataset size
                min_resources = min_resources_per_fold,  # Ensure minimum viable folds
                random_state = SEED,
                scoring = primary_scorer,  # Use primary scorer only for HalvingRandomSearchCV
                cv = cv_inner,
                refit = True,  # Always refit for halving search
                n_jobs = 1,  # Single job to prevent Windows resource exhaustion
                verbose = 1
            )

        # Stage 8: Hyperparameter Search Execution
        log_stage(logger, "HYPERPARAMETER_SEARCH", {
            "search_type": "GridSearchCV" if n_combinations <= 20 else "HalvingRandomSearchCV",
            "n_combinations": n_combinations,
            "n_jobs": 1,
            "backend": "sequential"
        })
        
        logger.info(f"Starting hyperparameter search for {extractor} + {model}...")
        search_start_time = time.time()
        
        # Use sequential processing to prevent Windows resource exhaustion
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", FutureWarning)
            
            try:
                # Force garbage collection and memory check before resource-intensive operation
                gc.collect()
                
                # Additional memory safety check before starting search
                try:
                    import psutil
                    current_memory = psutil.virtual_memory().percent
                    if current_memory > 96.0:
                        logger.error(f"Memory usage too high before search ({current_memory:.1f}%) - aborting")
                        return False
                    logger.info(f"Memory usage before search: {current_memory:.1f}%")
                except ImportError:
                    pass
                
                # Set up timeout handler for Windows safety
                class TimeoutException(Exception):
                    pass
                
                def timeout_handler(signum, frame):
                    raise TimeoutException("Hyperparameter search timed out")
                
                # Set alarm for search timeout (if supported on Windows)
                try:
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(SEARCH_TIMEOUT_MINUTES * 60)
                    
                    search.fit(X, y)
                    
                    signal.alarm(0)  # Cancel alarm
                except (AttributeError, OSError):
                    # SIGALRM not supported on Windows, proceed without timeout
                    search.fit(X, y)
                
                # Force garbage collection after search to free memory
                gc.collect()
                
            except Exception as e:
                logger.error(f"Hyperparameter search failed: {str(e)}")
                # Clean up resources on failure
                gc.collect()
                return False
        
        search_elapsed = time.time() - search_start_time
        logger.info(f"Hyperparameter search completed in {search_elapsed:.1f} seconds")
        
        # Validate search results
        if not hasattr(search, 'best_score_') or search.best_score_ is None:
            logger.error("Search completed but no best score found")
            return False
        
        # Handle multiple scorers for regression
        best_r2_score = None
        best_mae_score = None
        
        if task == "reg" and hasattr(search, 'cv_results_'):
            # Extract both R² and MAE scores
            try:
                best_r2_score = search.best_score_  # This is the R² score we optimized for
                
                # For GridSearchCV with multiple scorers, extract MAE from cv_results_
                if 'mean_test_mae' in search.cv_results_:
                    best_index = search.best_index_
                    best_mae_score = search.cv_results_['mean_test_mae'][best_index]
                    # Convert from negative MAE back to positive MAE
                    actual_mae = -best_mae_score if best_mae_score is not None else None
                    
                    logger.info(f"Best R² score: {best_r2_score:.4f}")
                    logger.info(f"Best MAE score: {actual_mae:.4f}" + 
                               (f" (baseline: {baseline_mae:.4f})" if baseline_mae is not None else ""))
                    
                    # Check if MAE improved over baseline
                    if baseline_mae is not None and actual_mae is not None:
                        mae_improvement = baseline_mae - actual_mae
                        mae_improvement_pct = (mae_improvement / baseline_mae) * 100
                        logger.info(f"MAE improvement: {mae_improvement:.4f} ({mae_improvement_pct:+.1f}%)")
                        
                        if mae_improvement > 0:
                            logger.info("✓ MAE improved over baseline")
                        else:
                            logger.warning(" MAE did not improve over baseline")
                else:
                    # For HalvingRandomSearchCV, compute MAE manually using the best estimator
                    logger.info(f"Best R² score: {best_r2_score:.4f}")
                    if hasattr(search, 'best_estimator_'):
                        try:
                            # Use cross-validation to compute MAE with the best parameters
                            from sklearn.model_selection import cross_val_score
                            
                            # Suppress sklearn warnings during MAE computation
                            with warnings.catch_warnings():
                                warnings.filterwarnings("ignore", 
                                                      message="The least populated class in y has only .* members", 
                                                      category=UserWarning)
                                mae_scores = cross_val_score(
                                    search.best_estimator_, X, y, 
                                    cv=cv_inner, 
                                    scoring=make_scorer(safe_mae_score, greater_is_better=True),
                                    n_jobs=1  # Single job to avoid conflicts
                                )
                            
                            actual_mae = -np.mean(mae_scores)  # Convert back to positive MAE
                            best_mae_score = -actual_mae  # Store as negative for consistency
                            
                            logger.info(f"Best MAE score (computed): {actual_mae:.4f}" + 
                                       (f" (baseline: {baseline_mae:.4f})" if baseline_mae is not None else ""))
                            
                            # Check if MAE improved over baseline
                            if baseline_mae is not None:
                                mae_improvement = baseline_mae - actual_mae
                                mae_improvement_pct = (mae_improvement / baseline_mae) * 100
                                logger.info(f"MAE improvement: {mae_improvement:.4f} ({mae_improvement_pct:+.1f}%)")
                                
                                if mae_improvement > 0:
                                    logger.info("✓ MAE improved over baseline")
                                else:
                                    logger.warning(" MAE did not improve over baseline")
                        except Exception as e:
                            logger.warning(f"Could not compute MAE manually: {e}")
                            logger.info(f"Best R² score: {best_r2_score:.4f}")
                    else:
                        logger.info(f"Best R² score: {best_r2_score:.4f}")
                        logger.warning("No best estimator available for MAE computation")
                        
            except Exception as e:
                logger.warning(f"Could not extract scoring results: {e}")
                logger.info(f"Best score: {search.best_score_:.4f}")
        
        if not np.isfinite(search.best_score_):
            logger.warning(f"Search completed but best score is non-finite: {search.best_score_}")
            # Continue anyway - we'll record this in the results
        
        # Check for successful parameter combinations
        if hasattr(search, 'cv_results_'):
            # Handle both single and multiple scorers
            if task == "reg":
                # For regression with multiple scorers, use the primary scorer (R²)
                if 'mean_test_r2' in search.cv_results_:
                    mean_test_scores = search.cv_results_['mean_test_r2']
                elif 'mean_test_score' in search.cv_results_:
                    mean_test_scores = search.cv_results_['mean_test_score']
                else:
                    logger.warning("Could not find test scores in cv_results_")
                    mean_test_scores = None
            else:
                # For classification with single scorer
                if 'mean_test_score' in search.cv_results_:
                    mean_test_scores = search.cv_results_['mean_test_score']
                else:
                    logger.warning("Could not find test scores in cv_results_")
                    mean_test_scores = None
            
            if mean_test_scores is not None:
                finite_scores = np.isfinite(mean_test_scores)
                n_successful = np.sum(finite_scores)
                n_total = len(mean_test_scores)
                
                logger.info(f"Successful parameter combinations: {n_successful}/{n_total} ({n_successful/n_total:.1%})")
                
                if n_successful == 0:
                    logger.error("No parameter combinations produced finite scores")
                    return False
                elif n_successful < n_total * 0.1:  # Less than 10% success rate
                    logger.warning(f"Low success rate for parameter combinations: {n_successful}/{n_total}")
            else:
                logger.warning("Could not validate parameter combinations - continuing anyway")
        
        logger.info("Hyperparameter search validation completed")

        # Stage 9: Results Processing and Saving
        log_stage(logger, "RESULTS_PROCESSING")
        
        # Save results with enhanced metadata
        best = {
            "best_params": search.best_params_,
            "best_score": search.best_score_,
            "dataset": dataset,
            "task": task,
            "extractor": extractor,
            "model": model,
            "preprocessing": "4-phase-enhanced",
            "data_shape": X.shape,
            "n_parameter_combinations": n_combinations,
            "search_method": "GridSearchCV" if n_combinations <= 20 else "HalvingRandomSearchCV",
            "cv_folds": CV_INNER,
            "search_time_seconds": search_elapsed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Add regression-specific metrics
        if task == "reg":
            best["best_r2_score"] = best_r2_score
            best["best_mae_score"] = -best_mae_score if best_mae_score is not None else None
            best["baseline_mae"] = baseline_mae
            if baseline_mae is not None and best_mae_score is not None:
                actual_mae = -best_mae_score
                mae_improvement = baseline_mae - actual_mae
                best["mae_improvement"] = mae_improvement
                best["mae_improvement_pct"] = (mae_improvement / baseline_mae) * 100
            best["scoring_method"] = "Multi-metric: R² (optimized) + MAE (tracked)"
        else:
            best["scoring_method"] = "Matthews correlation coefficient"

        fp = HP_DIR/f"{dataset}_{extractor}_{model}.json"
        
        # Convert non-serializable objects to string representations
        def make_json_serializable(obj):
            """Convert sklearn objects and other non-serializable objects to strings."""
            if hasattr(obj, '__class__') and hasattr(obj, '__module__'):
                # Check if it's a sklearn object or other complex object
                if 'sklearn' in str(type(obj)) or not isinstance(obj, (int, float, str, bool, list, dict, type(None))):
                    return str(obj)
            return obj
        
        # Make best_params JSON serializable
        serializable_best_params = {}
        for key, value in search.best_params_.items():
            serializable_best_params[key] = make_json_serializable(value)
        
        # Update the best dict with serializable parameters
        best["best_params"] = serializable_best_params
        
        # Attempt to save with better error handling
        try:
            with open(fp, "w") as f:
                json.dump(best, f, indent=2, default=str)
        except TypeError as e:
            logger.warning(f"JSON serialization failed: {e}")
            logger.info("Attempting fallback serialization...")
            
            # Fallback: convert all non-basic types to strings
            def deep_serialize(obj):
                if isinstance(obj, dict):
                    return {k: deep_serialize(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [deep_serialize(item) for item in obj]
                elif isinstance(obj, (int, float, str, bool, type(None))):
                    return obj
                else:
                    return str(obj)
            
            serialized_best = deep_serialize(best)
            with open(fp, "w") as f:
                json.dump(serialized_best, f, indent=2)
            logger.info("Fallback serialization successful")
        
        logger.info(f"SAVED {fp}")
        logger.info(f"  Best Score: {best['best_score']:.4f}")
        logger.info(f"  Data Shape: {X.shape}")
        logger.info(f"  Search Time: {search_elapsed:.1f}s")
        logger.info(f"  Preprocessing: 4-Phase Enhanced Pipeline")
        
        # Log best parameters
        logger.info("Best Parameters:")
        for param, value in search.best_params_.items():
            logger.info(f"  {param}: {value}")
        
        log_stage(logger, "TUNING_COMPLETED", {
            "status": "SUCCESS",
            "best_score": best['best_score'],
            "output_file": str(fp),
            "total_time_seconds": search_elapsed
        })
        
        # Final resource cleanup for Windows
        try:
            import psutil
            final_memory = psutil.virtual_memory().percent
            logger.info(f"Final memory usage: {final_memory:.1f}%")
            logger.info(f"Memory increase: {final_memory - initial_memory:.1f}%")
        except (ImportError, NameError):
            pass
        
        # Force final garbage collection
        gc.collect()
        
        return True
        
    except Exception as e:
        log_error_with_context(logger, e, {
            "operation": "tune_function",
            "dataset": dataset,
            "task": task,
            "extractor": extractor,
            "model": model
        })
        
        log_stage(logger, "TUNING_FAILED", {
            "status": "FAILED",
            "error_type": type(e).__name__,
            "error_message": str(e)
        }, level=logging.ERROR)
        
        # Force garbage collection on failure to cleanup resources
        gc.collect()
        
        return False

def tune_all_combinations(dataset, task, use_subprocess=True):
    """Run tuning for all extractor-model combinations for a dataset with 4-phase preprocessing and comprehensive logging."""
    
    # Setup session logger
    session_logger, session_log_path = setup_logging()
    
    if task == "reg":
        extractors = REGRESSION_EXTRACTORS
        models = REGRESSION_MODELS
    else:
        extractors = CLASSIFICATION_EXTRACTORS
        models = CLASSIFICATION_MODELS
    
    total_combinations = len(extractors) * len(models)
    
    log_stage(session_logger, "BATCH_TUNING_INITIALIZATION", {
        "dataset": dataset,
        "task": task,
        "total_combinations": total_combinations,
        "extractors": extractors,
        "models": models,
        "subprocess_isolation": use_subprocess,
        "timeout_minutes": TIMEOUT_MINUTES,
        "session_log": str(session_log_path)
    })
    
    successful = 0
    failed = 0
    start_time = time.time()
    
    for i, (extractor, model) in enumerate(product(extractors, models), 1):
        log_stage(session_logger, f"COMBINATION_{i}_OF_{total_combinations}", {
            "extractor": extractor,
            "model": model,
            "dataset": dataset,
            "preprocessing": "4-Phase Enhanced Pipeline"
        })
        
        if use_subprocess:
            success = run_tuning_subprocess(dataset, task, extractor, model, session_logger)
        else:
            # Direct execution (for debugging)
            try:
                success = tune(dataset, task, extractor, model, session_logger)
            except Exception as e:
                log_error_with_context(session_logger, e, {
                    "operation": "direct_tune_execution",
                    "combination": f"{extractor}+{model}",
                    "combination_number": f"{i}/{total_combinations}"
                })
                success = False
        
        if success:
            successful += 1
            session_logger.info(f"✓ COMPLETED ({i}/{total_combinations}): {extractor} + {model}")
        else:
            failed += 1
            session_logger.error(f"✗ FAILED ({i}/{total_combinations}): {extractor} + {model}")
    
    total_time = time.time() - start_time
    
    log_stage(session_logger, "BATCH_TUNING_COMPLETED", {
        "dataset": dataset,
        "total_time_minutes": f"{total_time/60:.1f}",
        "successful_combinations": successful,
        "failed_combinations": failed,
        "total_combinations": total_combinations,
        "success_rate": f"{successful/total_combinations:.1%}"
    })
    
    if successful > 0:
        # List generated files
        dataset_files = list(HP_DIR.glob(f"{dataset}_*.json"))
        if dataset_files:
            session_logger.info("Generated hyperparameter files:")
            for f in sorted(dataset_files):
                session_logger.info(f"  - {f.name}")
                
        # Show sample of best hyperparameters
        session_logger.info("Sample of optimized hyperparameters:")
        for f in sorted(dataset_files)[:3]:  # Show first 3
            try:
                with open(f, 'r') as file:
                    data = json.load(file)
                    session_logger.info(f"  {f.name}: Score={data['best_score']:.4f}")
            except:
                pass

def list_available_datasets():
    """List all available datasets and their auto-detected tasks from config.py."""
    print("Available datasets (from config.py):")
    print("=" * 50)
    
    # Get datasets from config
    try:
        from config import REGRESSION_DATASETS, CLASSIFICATION_DATASETS
        
        print("REGRESSION DATASETS:")
        for dataset_config in REGRESSION_DATASETS:
            name = dataset_config['name']
            outcome_col = dataset_config['outcome_col']
            print(f"  - {name:<12} ({outcome_col})")
        
        print("\nCLASSIFICATION DATASETS:")
        for dataset_config in CLASSIFICATION_DATASETS:
            name = dataset_config['name']
            outcome_col = dataset_config['outcome_col']
            print(f"  - {name:<12} ({outcome_col})")
            
    except ImportError:
        print("Could not load config.py, using hardcoded dataset info:")
        for dataset, task in DATASET_INFO.items():
            task_name = "regression" if task == "reg" else "classification"
            print(f"  - {dataset:<12} ({task_name})")

# ------------- Enhanced metrics with safer R² calculation ----------------------
def safe_r2_score(y_true, y_pred, **kwargs):
    """
    Safe R² scorer that handles edge cases with small sample sizes and model failures.
    Returns a meaningful fallback score when R² is undefined or when predictions are invalid.
    """
    # Check for minimum sample size
    if len(y_true) < 2:
        return -999.0
    
    # Validate predictions are finite and not all the same
    if not np.all(np.isfinite(y_pred)):
        return -999.0
    
    # Check if all predictions are identical (no variation)
    if np.std(y_pred) == 0:
        # If predictions have no variation, R² is typically very poor
        return -10.0
    
    # Check if targets have variation
    if np.std(y_true) == 0:
        # If targets have no variation, R² is undefined, but we can check prediction accuracy
        if np.std(y_pred - y_true) == 0:
            return 1.0  # Perfect prediction of constant
        else:
            return -1.0  # Poor prediction of constant
    
    try:
        r2 = r2_score(y_true, y_pred, **kwargs)
        # Check if R² is finite
        if not np.isfinite(r2):
            return -999.0
        # Cap extremely negative R² values for numerical stability
        return max(r2, -100.0)
    except Exception:
        return -999.0

def safe_mcc_score(y_true, y_pred, **kwargs):
    """
    Safe Matthews correlation coefficient that handles edge cases.
    """
    if len(y_true) < 2:
        return -1.0
    
    # Validate predictions
    if not np.all(np.isfinite(y_pred)):
        return -1.0
    
    # For binary classification, ensure we have both classes
    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)
    
    if len(unique_true) < 2:
        # Only one class in true labels
        if len(unique_pred) == 1 and unique_pred[0] in unique_true:
            return 1.0  # Perfect prediction of single class
        else:
            return -1.0  # Poor prediction
    
    try:
        mcc = matthews_corrcoef(y_true, y_pred, **kwargs)
        if not np.isfinite(mcc):
            return -1.0
        return mcc
    except Exception:
        return -1.0

def safe_mae_score(y_true, y_pred, **kwargs):
    """
    Safe Mean Absolute Error that handles edge cases.
    Returns negative MAE for sklearn compatibility (higher is better).
    """
    if len(y_true) < 2:
        return -999.0
    
    # Validate predictions are finite
    if not np.all(np.isfinite(y_pred)):
        return -999.0
    
    # Validate targets are finite  
    if not np.all(np.isfinite(y_true)):
        return -999.0
    
    try:
        mae = mean_absolute_error(y_true, y_pred, **kwargs)
        if not np.isfinite(mae):
            return -999.0
        return -mae  # Negative because sklearn scorers assume higher is better
    except Exception:
        return -999.0

def compute_baseline_mae(y):
    """
    Compute baseline MAE using mean prediction.
    This serves as a reference point for MAE improvement.
    """
    try:
        if len(y) < 2:
            return np.inf
        
        y_mean = np.mean(y)
        baseline_mae = mean_absolute_error(y, np.full_like(y, y_mean))
        return baseline_mae
    except Exception:
        return np.inf

def safe_pipeline_predict(pipeline, X, is_regression=True):
    """
    Safely make predictions with a pipeline, handling common failure modes.
    
    Parameters
    ----------
    pipeline : sklearn Pipeline
        The fitted pipeline
    X : np.ndarray
        Input features
    is_regression : bool
        Whether this is a regression task
        
    Returns
    -------
    np.ndarray or None
        Predictions, or None if prediction failed
    """
    try:
        # Validate input data
        if X.size == 0 or not np.all(np.isfinite(X)):
            return None
        
        # Make predictions
        if is_regression:
            y_pred = pipeline.predict(X)
        else:
            if hasattr(pipeline, 'predict_proba'):
                try:
                    y_proba = pipeline.predict_proba(X)
                    if np.any(np.isnan(y_proba)) or not np.all(np.isfinite(y_proba)):
                        # Fall back to direct predictions
                        y_pred = pipeline.predict(X)
                    else:
                        # Use probabilities for scoring if available
                        if y_proba.shape[1] == 2:
                            y_pred = y_proba[:, 1]  # Positive class probability
                        else:
                            y_pred = pipeline.predict(X)
                except:
                    y_pred = pipeline.predict(X)
            else:
                y_pred = pipeline.predict(X)
        
        # Validate predictions
        if not np.all(np.isfinite(y_pred)):
            return None
            
        return y_pred
        
    except Exception as e:
        return None

def validate_pipeline_step(pipeline, X_train, y_train, X_val, y_val, step_name="pipeline", is_regression=True):
    """
    Validate that a pipeline step produces valid results.
    
    Parameters
    ----------
    pipeline : sklearn Pipeline
        The pipeline to validate
    X_train, y_train : np.ndarray
        Training data
    X_val, y_val : np.ndarray
        Validation data
    step_name : str
        Name for logging
    is_regression : bool
        Whether this is regression
        
    Returns
    -------
    bool
        True if pipeline is valid, False otherwise
    """
    try:
        # Check input data validity
        if not np.all(np.isfinite(X_train)) or not np.all(np.isfinite(y_train)):
            return False
        if not np.all(np.isfinite(X_val)) or not np.all(np.isfinite(y_val)):
            return False
        
        # Check for minimum samples
        if X_train.shape[0] < 2 or X_val.shape[0] < 1:
            return False
        
        # Check for feature variation
        if X_train.shape[1] > 0 and np.all(np.std(X_train, axis=0) == 0):
            return False  # All features are constant
        
        # Try fitting
        pipeline.fit(X_train, y_train)
        
        # Try predicting
        train_pred = safe_pipeline_predict(pipeline, X_train, is_regression)
        val_pred = safe_pipeline_predict(pipeline, X_val, is_regression)
        
        if train_pred is None or val_pred is None:
            return False
        
        # Additional validation for classification
        if not is_regression:
            # Check if we have reasonable class predictions
            unique_train_true = len(np.unique(y_train))
            unique_val_true = len(np.unique(y_val))
            
            if unique_train_true < 2 and unique_val_true < 2:
                return False  # Not enough class diversity
        
        return True
        
    except Exception:
        return False

class SafeExtractorWrapper(BaseEstimator, TransformerMixin):
    """
    Wrapper to ensure extractors always produce 2-dimensional output and handle failures gracefully.
    
    This prevents the "Found array with dim 3. ElasticNet expected <= 2" error
    by enforcing that all extractor outputs are properly shaped for downstream models.
    Also provides fallback mechanisms for extractors that fail during fitting.
    """
    
    def __init__(self, extractor):
        self.extractor = extractor
        self.fallback_extractor = None
        self.extraction_failed = False
        
    def fit(self, X, y=None):
        """Fit the wrapped extractor with fallback handling."""
        try:
            self.extractor.fit(X, y)
            self.extraction_failed = False
        except Exception as e:
            # Get logger instance safely
            current_logger = logging.getLogger(__name__)
            error_msg = str(e)
            
            # Handle common extractor failures
            current_logger.warning(f"Primary extractor {type(self.extractor).__name__} failed: {error_msg}")
            
            # Special handling for KPCA zero-size array error
            if "zero-size array to reduction operation maximum" in error_msg:
                current_logger.warning("KPCA encountered zero-size eigenvalue array - likely due to insufficient data variation or too many components")
            elif "Matrix is not positive definite" in error_msg:
                current_logger.warning("KPCA encountered singular matrix - likely due to poor kernel parameter choice")
            
            # Create appropriate fallback based on extractor type
            if hasattr(self.extractor, 'n_components'):
                n_components = getattr(self.extractor, 'n_components', 2)
                
                # For KPCA failures, fall back to regular PCA
                if 'KernelPCA' in type(self.extractor).__name__ or 'KPCA' in type(self.extractor).__name__:
                    from sklearn.decomposition import PCA
                    # Use very conservative components for KPCA failures
                    safe_components = min(n_components // 2, X.shape[0] - 2, X.shape[1] // 4, 5)
                    safe_components = max(1, safe_components)  # Ensure at least 1 component
                    self.fallback_extractor = PCA(n_components=safe_components, random_state=42)
                    current_logger.info(f"Using conservative PCA fallback with {safe_components} components for KPCA failure")
                
                # For other component-based extractors, try PCA
                else:
                    from sklearn.decomposition import PCA
                    safe_components = min(n_components, X.shape[0] - 1, X.shape[1], 5)
                    safe_components = max(1, safe_components)
                    self.fallback_extractor = PCA(n_components=safe_components, random_state=42)
                    current_logger.info(f"Using PCA fallback with {safe_components} components")
            else:
                # For non-component extractors, use simple PCA
                from sklearn.decomposition import PCA
                safe_components = min(5, X.shape[0] - 1, X.shape[1])
                safe_components = max(1, safe_components)
                self.fallback_extractor = PCA(n_components=safe_components, random_state=42)
                current_logger.info(f"Using basic PCA fallback with {safe_components} components")
            
            try:
                self.fallback_extractor.fit(X, y)
                self.extraction_failed = True
                current_logger.info("Fallback extractor fitted successfully")
            except Exception as fallback_error:
                current_logger.error(f"Fallback extractor also failed: {fallback_error}")
                # Last resort: use identity transformation (first few features)
                self.extraction_failed = True
                self.fallback_extractor = None
        
        return self
        
    def transform(self, X):
        """Transform with safe dimensionality checking and fallback handling."""
        current_logger = logging.getLogger(__name__)
        
        if self.extraction_failed:
            if self.fallback_extractor is not None:
                try:
                    X_transformed = self.fallback_extractor.transform(X)
                except Exception as e:
                    current_logger.warning(f"Fallback transform failed: {e}, using identity transform")
                    # Last resort: return first few columns as features
                    n_features = min(5, X.shape[1])
                    X_transformed = X[:, :n_features]
            else:
                # Identity transformation fallback
                current_logger.warning("Using identity transform (first 5 features)")
                n_features = min(5, X.shape[1])
                X_transformed = X[:, :n_features]
        else:
            try:
                X_transformed = self.extractor.transform(X)
            except Exception as e:
                current_logger.warning(f"Transform failed: {e}, attempting fallback")
                if self.fallback_extractor is not None:
                    try:
                        X_transformed = self.fallback_extractor.transform(X)
                    except Exception as fallback_e:
                        current_logger.warning(f"Fallback transform also failed: {fallback_e}, using identity")
                        n_features = min(5, X.shape[1])
                        X_transformed = X[:, :n_features]
                else:
                    n_features = min(5, X.shape[1])
                    X_transformed = X[:, :n_features]
        
        # Ensure output is always 2-dimensional
        if X_transformed.ndim > 2:
            # Flatten extra dimensions while preserving the sample dimension
            original_shape = X_transformed.shape
            n_samples = original_shape[0]
            n_features = np.prod(original_shape[1:])  # Flatten all feature dimensions
            
            X_transformed = X_transformed.reshape(n_samples, n_features)
            
            current_logger.debug(f"SafeExtractorWrapper: Reshaped {original_shape} -> {X_transformed.shape}")
            
        elif X_transformed.ndim < 2:
            # Add feature dimension if needed
            if X_transformed.ndim == 1:
                X_transformed = X_transformed.reshape(-1, 1)
        
        # Handle empty or invalid arrays
        if X_transformed.size == 0 or X_transformed.shape[1] == 0:
            current_logger.warning("Empty transformation result, using identity fallback")
            n_features = min(5, X.shape[1])
            X_transformed = X[:, :n_features]
        
        # Final validation
        if X_transformed.ndim != 2:
            raise ValueError(f"SafeExtractorWrapper: Could not reshape to 2D. "
                           f"Got shape {X_transformed.shape} with {X_transformed.ndim} dimensions")
                           
        return X_transformed
        
    def fit_transform(self, X, y=None):
        """Fit and transform with safe dimensionality checking and fallback handling."""
        return self.fit(X, y).transform(X)
        
    def get_params(self, deep=True):
        """Get parameters from wrapped extractor."""
        if deep:
            params = self.extractor.get_params(deep=True)
            # Prefix with extractor__ to avoid conflicts
            return {f"extractor__{k}": v for k, v in params.items()}
        else:
            return {'extractor': self.extractor}
            
    def set_params(self, **params):
        """Set parameters on wrapped extractor."""
        extractor_params = {}
        for key, value in params.items():
            if key.startswith('extractor__'):
                # Remove the extractor__ prefix
                actual_key = key[len('extractor__'):]
                extractor_params[actual_key] = value
            elif key == 'extractor':
                self.extractor = value
                
        if extractor_params:
            self.extractor.set_params(**extractor_params)
            
        return self

# ------------- CLI -------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced 4-Phase Halving Tuner with Comprehensive Logging")
    parser.add_argument("--dataset", help="Dataset name")
    parser.add_argument("--task", choices=["reg", "clf"], help="Task type (reg/clf)")
    parser.add_argument("--extractor", help="Feature extractor")
    parser.add_argument("--model", help="Model name")
    parser.add_argument("--single", action="store_true", help="Single tuning mode (used by subprocess)")
    parser.add_argument("--no-subprocess", action="store_true", help="Disable subprocess isolation")
    parser.add_argument("--all", action="store_true", help="Run all extractor-model combinations for dataset")
    parser.add_argument("--list-datasets", action="store_true", help="List all available datasets")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Convert log level string to logging constant
    log_level = getattr(logging, args.log_level.upper())
    
    if args.list_datasets:
        list_available_datasets()
        sys.exit(0)
    
    if not args.dataset:
        parser.error("--dataset is required")
    
    # Validate dataset exists
    if args.dataset not in DATASET_INFO:
        print(f"Error: Dataset '{args.dataset}' not found.")
        list_available_datasets()
        sys.exit(1)
    
    # Auto-detect task if not specified
    if not args.task:
        args.task = detect_dataset_task(args.dataset)
        task_name = "regression" if args.task == "reg" else "classification"
        print(f"Auto-detected task for {args.dataset}: {task_name}")
    
    # Validate task matches dataset
    expected_task = DATASET_INFO[args.dataset]
    if args.task != expected_task:
        task_name = "regression" if expected_task == "reg" else "classification"
        print(f"Warning: {args.dataset} is a {task_name} dataset, using {expected_task}")
        args.task = expected_task
    
    # Single tuning mode (called by subprocess)
    if args.single:
        if not args.extractor or not args.model:
            parser.error("--single mode requires --extractor and --model")
        
        # Setup specific logger for single run
        logger, log_path = setup_logging(args.dataset, args.extractor, args.model, log_level)
        logger.info(f"Starting single tuning run: {args.dataset} - {args.extractor} - {args.model}")
        logger.info(f"Log file: {log_path}")
        
        success = tune(args.dataset, args.task, args.extractor, args.model, logger)
        sys.exit(0 if success else 1)
    
    # Setup session logger for batch operations
    session_logger, session_log_path = setup_logging(log_level=log_level)
    session_logger.info(f"Tuner session started with log level: {args.log_level}")
    session_logger.info(f"Session log file: {session_log_path}")
    
    # Batch mode: run all combinations
    if args.all:
        session_logger.info(f"Running ALL combinations for {args.dataset} with 4-Phase Enhanced Pipeline...")
        tune_all_combinations(args.dataset, args.task, not args.no_subprocess)
        sys.exit(0)
    
    # Single combination mode
    if not args.extractor or not args.model:
        parser.error("Single mode requires --extractor and --model")
    
    if args.no_subprocess:
        success = tune(args.dataset, args.task, args.extractor, args.model, session_logger)
    else:
        success = run_tuning_subprocess(args.dataset, args.task, args.extractor, args.model, session_logger)
    
    sys.exit(0 if success else 1)
