"""
Enhanced Halving tuner: finds best hyper-params for any dataset with 4-phase preprocessing
Supports two approaches:
1. Extractor approach: Tunes extractor + model parameters -> hp_best/<dataset>_<extractor>_<model>_<fusion>.json
2. Selector approach: Tunes model parameters with fixed feature selection -> hp_best/<dataset>_<model>_<fusion>_<n_features>f.json

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
import logging, gc, signal, pickle, hashlib
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
    Load dataset with FULL 4-phase preprocessing for tuner_halving.py in FEATURE-FIRST format.
    
    This loads preprocessed modalities SEPARATELY (not fused) to match the main pipeline's 
    feature-first architecture where:
    1. Each modality is preprocessed separately (miRNA->150, exp->1500, methy->2000 features)
    2. Feature extraction/selection is applied to each modality separately
    3. Fusion is applied to processed features
    4. Model training on fused features
    
    Expected feature counts after 4-phase preprocessing:
    - miRNA: 377 -> 150 features (aggressive dimensionality reduction)
- Gene Expression: 4987 -> 1500 features (aggressive dimensionality reduction)
- Methylation: 3956 -> 2000 features (aggressive dimensionality reduction)
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g., 'AML', 'Breast', etc.)
    task : str
        Task type ('reg' or 'clf')
        
    Returns
    -------
    Tuple[Dict[str, np.ndarray], np.ndarray]
        Preprocessed modalities as separate arrays (for feature-first pipeline) and targets
    """
    import numpy as np
    from config import DatasetConfig
    from enhanced_pipeline_integration import run_enhanced_preprocessing_pipeline
    import logging
    
    logger = logging.getLogger(__name__)
    logger.info(f"Loading dataset {dataset_name} with 4-phase preprocessing for FEATURE-FIRST pipeline (task: {task})")
    
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
    modalities_data, y_series, common_ids, is_regression = load_dataset(
        ds_name=dataset_name.lower(),
        modalities=modality_short_names,
        outcome_col=config['outcome_col'],
        task_type=task_type,
        parallel=True,
        use_cache=True
    )
    
    logger.info(f"Load dataset result: modalities_data={modalities_data is not None}, y_series={y_series is not None}, is_regression={is_regression}")
    
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
    
    # Apply 4-phase enhanced preprocessing pipeline WITHOUT fusion (feature-first approach)
    # NOTE: We don't want fusion here - we want separate preprocessed modalities
    logger.info(f"Applying 4-phase preprocessing WITHOUT fusion for feature-first architecture...")
    
    processed_modalities, y_aligned, pipeline_metadata = run_enhanced_preprocessing_pipeline(
        modality_data_dict=modality_data_dict,
        y=y_series.values,
        fusion_method="average",  # Any fusion method - feature_first_order=True returns separate modalities
        task_type=task_type,
        dataset_name=dataset_name,
        enable_early_quality_check=True,
        enable_feature_first_order=True,  # CRITICAL: This returns separate modalities (not fused)
        enable_centralized_missing_data=True,
        enable_coordinated_validation=True
    )
    
    logger.info(f"4-phase preprocessing completed with quality score: {pipeline_metadata.get('quality_score', 'N/A')}")
    logger.info(f"Preprocessed modalities (separate arrays for feature-first):")
    
    # The processed_modalities should now contain separate preprocessed modalities
    # with the target feature counts applied: miRNA->150, exp->1500, methy->2000
    for modality_name, modality_array in processed_modalities.items():
        logger.info(f"  Preprocessed {modality_name} shape: {modality_array.shape}")
        
        # Validate expected feature counts
        expected_features = {
            'mirna': 150,
            'exp': 1500, 
            'methy': 2000
        }
        
        if modality_name in expected_features:
            actual_features = modality_array.shape[1]
            expected = expected_features[modality_name]
            if actual_features > expected * 1.5:  # Allow some tolerance
                logger.warning(f"  {modality_name}: Expected ~{expected} features, got {actual_features}")
            else:
                logger.info(f"  {modality_name}: Feature count within expected range (~{expected})")
    
    y = y_aligned
    
    logger.info(f"Dataset {dataset_name} loaded with 4-phase preprocessing for feature-first pipeline:")
    logger.info(f"  Preprocessed modalities: {list(processed_modalities.keys())}")
    logger.info(f"  Total samples: {len(y)}")
    logger.info(f"  Quality score: {pipeline_metadata.get('quality_score', 'N/A')}")
    
    # Final validation
    for modality_name, X_mod in processed_modalities.items():
        if np.any(np.isnan(X_mod)) or np.any(np.isinf(X_mod)):
            logger.warning(f"Found NaN/Inf values in processed {modality_name}, cleaning...")
            processed_modalities[modality_name] = np.nan_to_num(X_mod, nan=0.0, posinf=1e6, neginf=-1e6)
    
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        logger.warning("Found NaN/Inf values in target, cleaning...")
        y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return processed_modalities, y
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
CACHE_DIR = pathlib.Path("preprocessing_cache"); CACHE_DIR.mkdir(exist_ok=True)

SEED   = 42
N_ITER = 32          # candidates; Halving keeps ~⅓ each rung
CV_INNER = 3
TIMEOUT_MINUTES = 30  # Timeout per combination
SEARCH_TIMEOUT_MINUTES = 10  # Timeout for individual hyperparameter search

# Minimum samples per fold for reliable metrics
MIN_SAMPLES_PER_FOLD = 5

# Preprocessing cache settings
ENABLE_PREPROCESSING_CACHE = True
CACHE_VERSION = "v1.0"  # Increment when preprocessing changes

# Initialize a default logger for cases where logger is not available
logger = logging.getLogger(__name__)

# ------------- Preprocessing Cache System ----------------------
def get_cache_key(dataset_name, task):
    """Generate a unique cache key for dataset + task + preprocessing version."""
    key_data = f"{dataset_name}_{task}_{CACHE_VERSION}"
    return hashlib.md5(key_data.encode()).hexdigest()

def save_preprocessing_cache(dataset_name, task, processed_modalities, y, sample_ids, baseline_mae):
    """Save preprocessed data to cache."""
    if not ENABLE_PREPROCESSING_CACHE:
        return
    
    try:
        cache_key = get_cache_key(dataset_name, task)
        cache_file = CACHE_DIR / f"{cache_key}.pkl"
        
        cache_data = {
            'dataset_name': dataset_name,
            'task': task,
            'processed_modalities': processed_modalities,
            'y': y,
            'sample_ids': sample_ids,
            'baseline_mae': baseline_mae,
            'timestamp': time.time(),
            'cache_version': CACHE_VERSION
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"Preprocessing cache saved: {cache_file}")
        
    except Exception as e:
        logger.warning(f"Failed to save preprocessing cache: {e}")

def load_preprocessing_cache(dataset_name, task):
    """Load preprocessed data from cache if available."""
    if not ENABLE_PREPROCESSING_CACHE:
        return None
    
    try:
        cache_key = get_cache_key(dataset_name, task)
        cache_file = CACHE_DIR / f"{cache_key}.pkl"
        
        if not cache_file.exists():
            return None
        
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Validate cache version
        if cache_data.get('cache_version') != CACHE_VERSION:
            logger.info(f"Cache version mismatch, ignoring cache: {cache_file}")
            return None
        
        # Check if cache is not too old (7 days)
        cache_age = time.time() - cache_data.get('timestamp', 0)
        if cache_age > 7 * 24 * 3600:  # 7 days
            logger.info(f"Cache too old ({cache_age/3600:.1f}h), ignoring: {cache_file}")
            return None
        
        logger.info(f"Preprocessing cache loaded: {cache_file}")
        return cache_data
        
    except Exception as e:
        logger.warning(f"Failed to load preprocessing cache: {e}")
        return None

def load_dataset_for_tuner_cached(dataset_name, task=None):
    """
    Load dataset with caching support to avoid redundant preprocessing.
    
    This function first checks if preprocessed data exists in cache.
    If not, it runs the full preprocessing and caches the result.
    """
    # Try to load from cache first
    cache_data = load_preprocessing_cache(dataset_name, task)
    if cache_data is not None:
        logger.info(f"Using cached preprocessing for {dataset_name} (task: {task})")
        return (
            cache_data['processed_modalities'],
            cache_data['y'],
            cache_data.get('sample_ids'),
            cache_data.get('baseline_mae')
        )
    
    # Cache miss - run full preprocessing
    logger.info(f"Cache miss for {dataset_name} (task: {task}), running full preprocessing...")
    processed_modalities, y = load_dataset_for_tuner_optimized(dataset_name, task)
    
    # Calculate baseline MAE and sample IDs
    sample_ids = list(range(len(y)))  # Simple sample IDs
    baseline_mae = np.mean(np.abs(y - np.mean(y))) if len(y) > 0 else 0.0
    
    # Save to cache for future use
    save_preprocessing_cache(dataset_name, task, processed_modalities, y, sample_ids, baseline_mae)
    
    return processed_modalities, y, sample_ids, baseline_mae

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

# Fixed feature counts for main pipeline integration
FEATURE_COUNTS = [8, 16, 32]  # Fixed feature selection counts from main pipeline

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

# Available extractors, selectors, and models by task (aligned with main pipeline)
REGRESSION_EXTRACTORS = ["PCA", "KPCA", "FA", "PLS", "KPLS", "SparsePLS"]  # 6 extractors - CURRENT IMPLEMENTATION
REGRESSION_SELECTORS = ["ElasticNetFS", "RFImportance", "VarianceFTest", "LASSO", "f_regressionFS"]  # 5 selectors - OPTION B IMPLEMENTATION
REGRESSION_MODELS = ["LinearRegression", "ElasticNet", "RandomForestRegressor"]  # 3 models - CURRENT IMPLEMENTATION

CLASSIFICATION_EXTRACTORS = ["PCA", "KPCA", "FA", "LDA", "PLS-DA", "SparsePLS"]  # 6 extractors - CURRENT IMPLEMENTATION
CLASSIFICATION_SELECTORS = ["ElasticNetFS", "RFImportance", "VarianceFTest", "LASSO", "LogisticL1"]  # 5 selectors - OPTION B IMPLEMENTATION  
CLASSIFICATION_MODELS = ["LogisticRegression", "RandomForestClassifier", "SVC"]  # 3 models - CURRENT IMPLEMENTATION

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

# ------------- Parameter space functions for both approaches ----------------------------------
def param_space_extractors(extr, mdl, X_shape=None):
    """
    Parameter space for EXTRACTOR-based tuning (original approach).
    
    Tunes both extractor parameters (n_components, etc.) and model parameters.
    This is for the traditional feature extraction pipeline.
    
    Parameters
    ----------
    extr : str
        Extractor name (PCA, KPCA, FA, etc.)
    mdl : str
        Model name
    X_shape : tuple, optional
        Shape of the data (n_samples, n_features) for adaptive parameters
    """
    p = {}
    
    # Feature extraction parameter tuning (original approach)
    n_samples = X_shape[0] if X_shape is not None else 100
    n_features = X_shape[1] if X_shape is not None else 3650
    
    current_logger = logging.getLogger(__name__)
    current_logger.debug(f"Extractor+Model parameter space: {extr}+{mdl}, samples={n_samples}, features={n_features}")
    
    # Calculate cross-validation fold size for safety constraints
    cv_fold_size = n_samples // CV_INNER
    
    # Component ranges adapted for datasets and cross-validation stability
    if n_samples >= 150:
        base_components = [8, 16, 32, 64, 128]
    elif n_samples >= 100:
        base_components = [4, 8, 16, 32, 64]
    elif n_samples >= 50:
        base_components = [2, 4, 8, 16, 32]
    elif n_samples >= 20:
        base_components = [1, 2, 4, 8]
    else:
        base_components = [1, 2]
    
    # Safety constraints for cross-validation
    min_cv_samples = max(3, cv_fold_size // 2)
    max_safe_components = min(
        min_cv_samples - 1,
        n_features // 2,
        64
    )
    
    if n_samples < 30:
        max_safe_components = min(max_safe_components, n_samples // 4, 8)
    
    # Special handling for kernel methods
    if extr in {"KPCA", "KPLS"}:
        max_safe_components = min(max_safe_components, cv_fold_size // 2, 64)
    
    # Filter components based on constraints
    component_options = [c for c in base_components if c <= max_safe_components]
    
    if not component_options:
        if max_safe_components >= 8:
            component_options = [8]
        elif max_safe_components >= 4:
            component_options = [4]
        elif max_safe_components >= 2:
            component_options = [2]
        else:
            component_options = [1]
    
    # Add smaller component as fallback
    if component_options and min(component_options) > 8 and max_safe_components >= 8:
        component_options = [8] + component_options
    
    # Extractor parameters (with extractor__extractor__ prefix due to SafeExtractorWrapper)
    if extr in {"PCA","KPCA","KPLS","PLS","SparsePLS","PLS-DA","Sparse PLS-DA"}:
        p["extractor__extractor__n_components"] = component_options
    
    if extr in {"KPCA","KPLS"}:
        if n_samples < 100:
            p["extractor__extractor__gamma"] = [0.01, 0.1, 1.0]
        else:
            p["extractor__extractor__gamma"] = [0.001, 0.01, 0.1, 1.0]
    
    if extr in {"SparsePLS","Sparse PLS-DA"}:
        p["extractor__extractor__alpha"] = np.logspace(-1, 0, 3)
    
    if extr == "FA":
        # Very aggressive FA optimization for speed - FA is inherently slow on high-dimensional data
        safe_fa_components = [min(c, 8) for c in component_options if c <= 8]  # Cap FA at 8 components max
        if not safe_fa_components:
            safe_fa_components = [2, 4]  # Fallback to very small components
        p["extractor__extractor__n_components"] = safe_fa_components
        p["extractor__extractor__max_iter"] = [50, 100]  # Very low iterations for speed
        p["extractor__extractor__tol"] = [1e-1, 1e0]  # Very relaxed tolerance for speed
    
    if extr == "LDA":
        p["extractor__extractor__solver"] = ["lsqr", "svd", "eigen"]
        p["extractor__extractor__shrinkage"] = [None, "auto"]
    
    # Model parameters for extractor-based approach
    _add_model_parameters(p, mdl, n_samples)
    
    return p

def param_space_selectors(selector, mdl, n_features=None, X_shape=None):
    """
    Parameter space for SELECTOR-based tuning (Option B implementation).
    
    Optimizes model hyperparameters separately for each selector type, recognizing that
    different selectors produce features with different statistical properties.
    
    Parameters
    ----------
    selector : str
        Selector name (ElasticNetFS, RFImportance, VarianceFTest, LASSO, f_regressionFS/LogisticL1)
    mdl : str
        Model name
    n_features : int, optional
        Number of features after selection (8, 16, or 32)
    X_shape : tuple, optional
        Shape of the data (n_samples, total_features_before_selection)
    """
    p = {}
    
    # Get sample size for adaptive parameter selection
    n_samples = X_shape[0] if X_shape is not None else 100
    effective_features = n_features if n_features is not None else 16
    
    current_logger = logging.getLogger(__name__)
    current_logger.debug(f"Selector+Model parameter space: {selector}+{mdl}, samples={n_samples}, selected_features={effective_features}")
    
    # NOTE: Feature selection count is fixed, but model hyperparameters are optimized
    # separately for each selector type due to different feature characteristics
    
    # Model parameters for selector-based approach (optimized for selector-specific feature properties)
    _add_model_parameters_for_selectors(p, selector, mdl, n_samples, effective_features)
    
    return p

def _add_model_parameters(p, mdl, n_samples):
    """Add model parameters for extractor-based tuning (traditional approach)."""
    # Model parameters for traditional feature extraction pipeline (higher dimensional features)
    
    if mdl == "RandomForestRegressor":
        # Optimized RandomForest for faster tuning while maintaining performance
        # Removed slowest combinations: n_estimators=500, max_depth=None/20
        if n_samples < 50:
            p.update({
                "model__n_estimators": [50, 100, 200],  # Reduced max estimators
                "model__max_features": ["sqrt", "log2", 0.3, 0.5],
                "model__bootstrap": [True, False],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 3],
                "model__max_depth": [10, 15, 20],  # Removed None (unlimited depth)
            })
        else:
            p.update({
                "model__n_estimators": [100, 200, 300],  # Reduced from [200, 300, 500]
                "model__max_features": ["sqrt", "log2", 0.2, 0.3, 0.5],
                "model__bootstrap": [True, False],
                "model__min_samples_split": [2, 5, 10, 15],
                "model__min_samples_leaf": [1, 2, 3],
                "model__max_depth": [10, 15, 20],  # Removed None (unlimited depth)
            })
    
    if mdl == "RandomForestClassifier":
        # Optimized RandomForest classification for faster tuning while maintaining performance
        # Removed slowest combinations: n_estimators=500, max_depth=None
        if n_samples < 50:
            p.update({
                "model__n_estimators": [50, 100, 200],  # Reduced max estimators
                "model__max_depth": [5, 10, 15],  # Removed None (unlimited depth)
                "model__min_samples_leaf": [1, 2, 3],
                "model__min_samples_split": [2, 5],
                "model__max_features": ["sqrt", "log2", 0.5],
                "model__class_weight": [None, "balanced"]
            })
        else:
            p.update({
                "model__n_estimators": [100, 200, 300],  # Reduced from [200, 300, 500]
                "model__max_depth": [10, 15, 20],  # Removed None (unlimited depth)
                "model__min_samples_leaf": [1, 2, 3],
                "model__min_samples_split": [2, 5, 10],
                "model__max_features": ["sqrt", "log2", 0.3, 0.5],
                "model__class_weight": [None, "balanced"]
            })
    
    if mdl == "ElasticNet":
        # Traditional ElasticNet for extracted features
        p.update({
            "model__l1_ratio": np.linspace(0.05, 0.95, 7),
            "model__cv": [3, 5, 7],
            "model__n_alphas": [50, 100, 200],
            "model__eps": [1e-4, 1e-5],
            "model__max_iter": [1000, 2000, 3000],
        })
    
    if mdl == "SVC":
        # Traditional SVC for extracted features
        p.update({
            "model__C": np.logspace(-2, 2, 6),
            "model__gamma": np.logspace(-4, 1, 6),
            "model__kernel": ["rbf", "linear", "poly"],
            "model__class_weight": [None, "balanced"]
        })
    
    if mdl == "LinearRegression":
        # Traditional robust linear regression
        p.update({
            "model__method": ["huber", "ransac", "theil_sen"],
            "model__epsilon": [1.2, 1.35, 1.5, 2.0],
            "model__alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1],
            "model__max_iter": [1000, 2000, 3000],
        })
    
    if mdl == "LogisticRegression":
        # Traditional logistic regression for extracted features with compatible solver/penalty combinations
        p.update({
            "model__C": np.logspace(-2, 2, 6),
            "model__penalty": ["l1", "l2"],  # Remove elasticnet to avoid solver conflicts
            "model__solver": ["liblinear", "saga"],  # Remove lbfgs to avoid elasticnet conflicts
            "model__max_iter": [1000, 2000, 3000],
            "model__class_weight": [None, "balanced"]
        })

def _add_model_parameters_for_selectors(p, selector, mdl, n_samples, n_features):
    """Add model parameters for selector-based tuning (optimized for selector-specific feature properties)."""
    # Model parameters optimized for low-dimensional selected features [8, 16, 32]
    # Different selectors produce features with different statistical properties:
    # - ElasticNetFS: Correlated features with L1+L2 regularization
    # - RFImportance: Interaction-rich features from tree splits
    # - VarianceFTest: Univariately significant features
    # - LASSO: Sparse, uncorrelated features
    # - f_regressionFS/LogisticL1: Linearly predictive features
    
    if mdl == "RandomForestRegressor":
        # Optimized RandomForest for faster tuning with selected features
        # Reduced n_estimators and removed max_depth=None for performance
        base_params = {
            "model__n_estimators": [100, 200, 300] if n_samples >= 50 else [50, 100, 200],  # Reduced
            "model__bootstrap": [True, False],
            "model__min_samples_split": [2, 5, 8] if n_samples >= 50 else [2, 3, 5],
            "model__min_samples_leaf": [1, 2, 3] if n_samples >= 50 else [1, 2],
            "model__max_depth": [8, 15, 20] if n_samples >= 50 else [5, 10, 15],  # Removed None
        }
        
        # Selector-specific optimizations (performance-optimized)
        if selector in ["RFImportance"]:
            # RFImportance selects interaction-rich features - use more trees, deeper trees (but limited)
            base_params.update({
                "model__n_estimators": [200, 300, 400] if n_samples >= 50 else [100, 200, 300],  # Reduced
                "model__max_features": [None, "sqrt", 0.8],  # Use more features for interactions
                "model__max_depth": [15, 20, 25] if n_samples >= 50 else [10, 15, 20],  # Removed None
            })
        elif selector in ["ElasticNetFS"]:
            # ElasticNetFS selects correlated features - reduce overfitting
            base_params.update({
                "model__max_features": ["sqrt", 0.5, 0.7],  # Reduce feature sampling
                "model__min_samples_split": [5, 8, 10] if n_samples >= 50 else [3, 5, 8],
                "model__min_samples_leaf": [2, 3, 4] if n_samples >= 50 else [2, 3],
            })
        elif selector in ["LASSO"]:
            # LASSO selects sparse, uncorrelated features - can use all features
            base_params.update({
                "model__max_features": [None, "sqrt", 0.9],  # Use more features
                "model__bootstrap": [False, True],  # Prefer no bootstrap for sparse features
            })
        elif selector in ["VarianceFTest", "f_regressionFS"]:
            # Univariate selectors - standard RF settings
            base_params.update({
                "model__max_features": [None, "sqrt", 0.7, 0.9],
            })
        else:
            # Default case
            base_params.update({
                "model__max_features": [None, "sqrt", 0.7, 0.9],
            })
        
        p.update(base_params)
    
    if mdl == "RandomForestClassifier":
        # Optimized RandomForest classification for faster tuning with selected features
        # Reduced n_estimators and removed max_depth=None for performance
        base_params = {
            "model__n_estimators": [100, 200, 300] if n_samples >= 50 else [50, 100, 200],  # Reduced
            "model__max_depth": [8, 12, 20] if n_samples >= 50 else [5, 8, 12],  # Removed None
            "model__min_samples_leaf": [1, 2, 3],
            "model__min_samples_split": [2, 5, 8] if n_samples >= 50 else [2, 3, 5],
            "model__class_weight": [None, "balanced"]
        }
        
        # Selector-specific optimizations (performance-optimized)
        if selector in ["RFImportance"]:
            # RFImportance selects interaction-rich features - use more trees, deeper trees (but limited)
            base_params.update({
                "model__n_estimators": [200, 300, 400] if n_samples >= 50 else [100, 200, 300],  # Reduced
                "model__max_features": [None, "sqrt", 0.8],  # Use more features for interactions
                "model__max_depth": [15, 20, 25] if n_samples >= 50 else [10, 15, 20],  # Removed None
            })
        elif selector in ["ElasticNetFS"]:
            # ElasticNetFS selects correlated features - reduce overfitting
            base_params.update({
                "model__max_features": ["sqrt", 0.5, 0.7],  # Reduce feature sampling
                "model__min_samples_split": [5, 8, 10] if n_samples >= 50 else [3, 5, 8],
                "model__min_samples_leaf": [2, 3, 4] if n_samples >= 50 else [2, 3],
                "model__class_weight": ["balanced"],  # Often needed for correlated features
            })
        elif selector in ["LASSO", "LogisticL1"]:
            # LASSO/LogisticL1 selects sparse, uncorrelated features - can use all features
            base_params.update({
                "model__max_features": [None, "sqrt", 0.9],  # Use more features
                "model__bootstrap": [False, True],  # Prefer no bootstrap for sparse features
            })
        elif selector in ["VarianceFTest"]:
            # Univariate F-test selectors - standard RF settings
            base_params.update({
                "model__max_features": [None, "sqrt", 0.7, 0.9],
            })
        else:
            # Default case
            base_params.update({
                "model__max_features": [None, "sqrt", 0.7, 0.9],
            })
        
        p.update(base_params)
    
    if mdl == "ElasticNet":
        # ElasticNet optimized for selector-specific feature characteristics
        base_params = {
            "model__cv": [3, 5, 7],
            "model__n_alphas": [50, 100, 150],
            "model__eps": [1e-4, 1e-3],
            "model__max_iter": [1000, 2000, 3000],
            "model__selection": ["cyclic", "random"]
        }
        
        # Selector-specific optimizations
        if selector in ["ElasticNetFS"]:
            # ElasticNetFS already uses ElasticNet - use different l1_ratio ranges
            base_params.update({
                "model__l1_ratio": np.linspace(0.3, 0.9, 4),  # Prefer more L1 (different from selector)
                "model__max_iter": [2000, 3000, 4000],  # More iterations for convergence
            })
        elif selector in ["LASSO"]:
            # LASSO selects sparse features - use more L2 regularization for diversity
            base_params.update({
                "model__l1_ratio": np.linspace(0.1, 0.6, 4),  # More L2 for complementary regularization
            })
        elif selector in ["RFImportance"]:
            # RFImportance selects interaction features - balanced L1/L2
            base_params.update({
                "model__l1_ratio": np.linspace(0.2, 0.8, 4),  # Balanced regularization
            })
        elif selector in ["VarianceFTest", "f_regressionFS"]:
            # Univariate selectors - standard ElasticNet
            base_params.update({
                "model__l1_ratio": np.linspace(0.1, 0.9, 5),  # Full range
            })
        else:
            # Default case
            base_params.update({
                "model__l1_ratio": np.linspace(0.1, 0.9, 5),
            })
        
        p.update(base_params)
    
    if mdl == "SVC":
        # SVC optimized for few selected features
        p.update({
            "model__C": np.logspace(-1, 2, 5),
            "model__gamma": ["scale", "auto"] + list(np.logspace(-3, 1, 4)),
            "model__kernel": ["rbf", "linear", "poly"],
            "model__degree": [2, 3, 4],
            "model__class_weight": [None, "balanced"]
        })
    
    if mdl == "LinearRegression":
        # Linear regression optimized for few selected features
        p.update({
            "model__method": ["huber", "ransac", "theil_sen"],
            "model__epsilon": [1.1, 1.35, 1.5, 2.0],
            "model__alpha": [0.0001, 0.001, 0.01],
            "model__max_iter": [1000, 2000],
            "model__random_state": [42]
        })
    
    if mdl == "LogisticRegression":
        # Logistic regression optimized for selector-specific feature characteristics
        base_params = {
            "model__max_iter": [1000, 2000, 3000],
            "model__class_weight": [None, "balanced"],
            "model__random_state": [42]
        }
        
        # Selector-specific optimizations with compatible solver/penalty combinations
        if selector in ["ElasticNetFS"]:
            # ElasticNetFS selects correlated features - use L2 penalty to handle correlation
            base_params.update({
                "model__C": np.logspace(-1, 1, 4),  # Moderate regularization
                "model__penalty": ["l2"],  # Only L2 to avoid solver conflicts
                "model__solver": ["lbfgs"],  # Compatible with L2
            })
        elif selector in ["LASSO", "LogisticL1"]:
            # LASSO/LogisticL1 selects sparse features - use different regularization
            base_params.update({
                "model__C": np.logspace(-1, 2, 5),  # Wider C range
                "model__penalty": ["l2"],  # Only L2 to avoid solver conflicts
                "model__solver": ["lbfgs"],  # Compatible with L2
            })
        elif selector in ["RFImportance"]:
            # RFImportance selects interaction features - use compatible solver/penalty combinations
            base_params.update({
                "model__C": np.logspace(-1, 2, 5),
                "model__penalty": ["l1", "l2"],  # Remove elasticnet to avoid solver conflicts
                "model__solver": ["liblinear", "saga"],  # Compatible solvers
            })
        elif selector in ["VarianceFTest"]:
            # Univariate F-test selectors - use compatible solver/penalty combinations
            base_params.update({
                "model__C": np.logspace(-1, 2, 5),
                "model__penalty": ["l1", "l2"],  # Remove elasticnet to avoid solver conflicts
                "model__solver": ["liblinear", "saga"],  # Compatible solvers
            })
        else:
            # Default case - use compatible solver/penalty combinations
            base_params.update({
                "model__C": np.logspace(-1, 2, 5),
                "model__penalty": ["l1", "l2"],  # Remove elasticnet to avoid solver conflicts
                "model__solver": ["liblinear", "saga"],  # Compatible solvers
            })
        
        p.update(base_params)
    

    
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
def run_tuning_subprocess_extractors(dataset, task, extractor, model, fusion_method="average", logger=None):
    """Run extractor tuning for a single combination in subprocess with timeout and logging."""
    if logger is None:
        logger = logging.getLogger("tuner_session")
    
    log_stage(logger, "SUBPROCESS_EXECUTION", {
        "dataset": dataset,
        "task": task,
        "extractor": extractor,
        "model": model,
        "fusion_method": fusion_method,
        "timeout_minutes": TIMEOUT_MINUTES
    })
    
    cmd = [
        sys.executable, __file__,
        "--dataset", dataset,
        "--task", task,
        "--approach", "extractors",
        "--extractor", extractor,
        "--model", model,
        "--fusion", fusion_method,
        "--single"  # Flag to indicate single combination mode
    ]
    
    logger.info(f"Executing subprocess command: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_MINUTES*60)
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            logger.info(f"SUCCESS ({elapsed:.1f}s): {extractor} + {model} + {fusion_method}")
            if result.stdout.strip():
                logger.debug(f"Subprocess stdout:\n{result.stdout}")
        else:
            logger.error(f"FAILED ({elapsed:.1f}s): {extractor} + {model} + {fusion_method}")
            if result.stderr.strip():
                logger.error(f"Subprocess stderr:\n{result.stderr}")
            if result.stdout.strip():
                logger.debug(f"Subprocess stdout:\n{result.stdout}")
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        logger.error(f"TIMEOUT ({elapsed/60:.1f}min): {extractor} + {model} + {fusion_method}")
        logger.error(f"Process exceeded {TIMEOUT_MINUTES} minute timeout")
        return False
    except Exception as e:
        elapsed = time.time() - start_time
        log_error_with_context(logger, e, {
            "operation": "subprocess_execution",
            "dataset": dataset,
            "extractor": extractor,
            "model": model,
            "fusion_method": fusion_method,
            "elapsed_time": f"{elapsed:.1f}s"
        })
        return False

def run_tuning_subprocess_selectors(dataset, task, selector, model, fusion_method="average", n_features=16, logger=None):
    """Run tuning for a single combination in subprocess with timeout and logging."""
    if logger is None:
        logger = logging.getLogger("tuner_session")
    
    log_stage(logger, "SUBPROCESS_EXECUTION", {
        "dataset": dataset,
        "task": task,
        "selector": selector,
        "model": model,
        "fusion_method": fusion_method,
        "n_features": n_features,
        "timeout_minutes": TIMEOUT_MINUTES
    })
    
    cmd = [
        sys.executable, __file__,
        "--dataset", dataset,
        "--task", task,
        "--approach", "selectors",
        "--selector", selector,
        "--model", model,
        "--fusion", fusion_method,
        "--n-features", str(n_features),
        "--single"  # Flag to indicate single combination mode
    ]
    
    logger.info(f"Executing subprocess command: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_MINUTES*60)
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            logger.info(f"SUCCESS ({elapsed:.1f}s): {model} + {fusion_method} + {n_features}f")
            if result.stdout.strip():
                logger.debug(f"Subprocess stdout:\n{result.stdout}")
        else:
            logger.error(f"FAILED ({elapsed:.1f}s): {model} + {fusion_method} + {n_features}f")
            if result.stderr.strip():
                logger.error(f"Subprocess stderr:\n{result.stderr}")
            if result.stdout.strip():
                logger.debug(f"Subprocess stdout:\n{result.stdout}")
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        logger.error(f"TIMEOUT ({elapsed/60:.1f}min): {model} + {fusion_method} + {n_features}f")
        logger.error(f"Process exceeded {TIMEOUT_MINUTES} minute timeout")
        return False
    except Exception as e:
        elapsed = time.time() - start_time
        log_error_with_context(logger, e, {
            "operation": "subprocess_execution",
            "dataset": dataset,
            "model": model,
            "fusion_method": fusion_method,
            "n_features": n_features,
            "elapsed_time": f"{elapsed:.1f}s"
        })
        return False

# ------------- main tune routine with 4-phase preprocessing -----------------------------
def create_multimodal_pipeline(extractor, model, fusion_method="average"):
    """
    Create a multi-modal pipeline that processes each modality separately.
    
    The correct pipeline architecture:
    1. Extract/Select per modality: PCA on mirna → (n_samples, n_components), PCA on exp → (n_samples, n_components), etc.
    2. Fusion: Combine extracted features using fusion method
    3. Model training: Train on fused features
    
    This cannot be done with sklearn Pipeline since it doesn't handle multiple input matrices.
    Instead, we return a custom class that handles the multi-modal workflow.
    """
    from models import build_extractor, build_model
    
    class MultiModalPipeline:
        def __init__(self, extractor_name, model_name, fusion_method="average"):
            self.extractor_name = extractor_name
            self.model_name = model_name
            self.fusion_method = fusion_method
            self.extractors = {}  # One extractor per modality
            self.model = None
            self.modality_names = None
            
        def fit(self, X_modalities, y):
            """
            Fit the multi-modal pipeline.
            
            Parameters:
            - X_modalities: dict of {modality_name: X_array}
            - y: target array
            """
            import numpy as np
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from models import build_extractor, build_model
            
            self.modality_names = list(X_modalities.keys())
            
            # Step 1: Fit extractors separately for each modality
            extracted_features = {}
            for modality_name, X_modality in X_modalities.items():
                # Build and fit extractor for this modality
                extractor_obj = build_extractor(self.extractor_name)
                if extractor_obj is None:
                    raise ValueError(f"Failed to build extractor: {self.extractor_name}")
                extractor_obj.fit(X_modality, y)
                self.extractors[modality_name] = extractor_obj
                
                # Extract features
                X_extracted = extractor_obj.transform(X_modality)
                extracted_features[modality_name] = X_extracted
            
            # Step 2: Fuse extracted features
            X_fused = self._fuse_features(extracted_features)
            
            # Step 3: Fit model on fused features
            task_type = "reg" if "Regressor" in self.model_name else "clf"
            self.model = build_model(self.model_name, task_type)
            if self.model is None:
                raise ValueError(f"Failed to build model: {self.model_name}")
            self.model.fit(X_fused, y)
            
            return self
        
        def predict(self, X_modalities):
            """
            Make predictions with the multi-modal pipeline.
            
            Parameters:
            - X_modalities: dict of {modality_name: X_array}
            """
            import numpy as np
            
            # Step 1: Extract features from each modality
            extracted_features = {}
            for modality_name, X_modality in X_modalities.items():
                if modality_name not in self.extractors:
                    raise ValueError(f"Modality {modality_name} not found in fitted extractors")
                
                X_extracted = self.extractors[modality_name].transform(X_modality)
                extracted_features[modality_name] = X_extracted
            
            # Step 2: Fuse extracted features
            X_fused = self._fuse_features(extracted_features)
            
            # Step 3: Make predictions
            return self.model.predict(X_fused)
        
        def _fuse_features(self, extracted_features):
            """
            Fuse extracted features from multiple modalities using proper fusion techniques.
            
            Parameters:
            - extracted_features: dict of {modality_name: extracted_X_array}
            
            Returns:
            - X_fused: fused feature matrix
            """
            import numpy as np
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from fusion import merge_modalities
            
            modality_arrays = list(extracted_features.values())
            
            if len(modality_arrays) == 1:
                return modality_arrays[0]
            
            # Use proper fusion from fusion.py
            try:
                # For multi-modal pipeline, we don't have y during transform, so use simpler fusion
                if self.fusion_method == "average":
                    # Proper average fusion: element-wise mean after scaling
                    from sklearn.preprocessing import RobustScaler
                    scaled_arrays = []
                    
                    for i, arr in enumerate(modality_arrays):
                        scaler = RobustScaler()
                        try:
                            arr_scaled = scaler.fit_transform(arr)
                            arr_scaled = np.clip(arr_scaled, -5, 5)  # Clip outliers
                            scaled_arrays.append(arr_scaled.astype(np.float32))
                        except Exception as e:
                            # Fallback: use original array
                            scaled_arrays.append(arr.astype(np.float32))
                    
                    # Element-wise average
                    fused = np.mean(scaled_arrays, axis=0)
                    return fused
                    
                elif self.fusion_method == "attention_weighted":
                    # For attention-weighted, we need concatenation since we don't have y for attention computation
                    # This is a limitation - true attention needs targets during fitting
                    # Fall back to variance-based weighting then concatenation
                    modality_weights = []
                    for arr in modality_arrays:
                        feature_variances = np.var(arr, axis=0)
                        mean_variance = np.mean(feature_variances)
                        modality_weights.append(mean_variance)
                    
                    # Normalize weights
                    total_weight = sum(modality_weights)
                    if total_weight > 0:
                        modality_weights = [w / total_weight for w in modality_weights]
                    else:
                        modality_weights = [1.0 / len(modality_arrays)] * len(modality_arrays)
                    
                    # Apply weights and concatenate
                    weighted_modalities = []
                    for arr, weight in zip(modality_arrays, modality_weights):
                        weighted_modalities.append(arr * weight)
                    
                    return np.column_stack(weighted_modalities)
                
                else:
                    raise ValueError(f"Unsupported fusion method: {self.fusion_method}")
                    
            except Exception as e:
                # Fallback to concatenation if fusion fails
                print(f"Warning: Fusion failed ({e}), falling back to concatenation")
                return np.column_stack(modality_arrays)
        
        def set_params(self, **params):
            """
            Set parameters for the pipeline components.
            
            Parameters should be in format:
            - extractor__param_name: parameter for all extractors
            - model__param_name: parameter for the model
            """
            extractor_params = {}
            model_params = {}
            
            for param_name, param_value in params.items():
                if param_name.startswith('extractor__'):
                    # Remove the extractor__ prefix
                    actual_param = param_name.replace('extractor__', '')
                    extractor_params[actual_param] = param_value
                elif param_name.startswith('model__'):
                    # Remove the model__ prefix
                    actual_param = param_name.replace('model__', '')
                    model_params[actual_param] = param_value
            
            # Apply extractor parameters to all extractors
            if extractor_params:
                for extractor_obj in self.extractors.values():
                    extractor_obj.set_params(**extractor_params)
            
            # Apply model parameters
            if model_params and self.model is not None:
                self.model.set_params(**model_params)
            
            return self
        
        def get_params(self, deep=True):
            """Get parameters for the pipeline."""
            params = {}
            
            # Get extractor parameters (use first extractor as template)
            if self.extractors:
                first_extractor = list(self.extractors.values())[0]
                extractor_params = first_extractor.get_params(deep=deep)
                for param_name, param_value in extractor_params.items():
                    params[f'extractor__{param_name}'] = param_value
            
            # Get model parameters
            if self.model is not None:
                model_params = self.model.get_params(deep=deep)
                for param_name, param_value in model_params.items():
                    params[f'model__{param_name}'] = param_value
            
            return params
    
    return MultiModalPipeline(extractor, model, fusion_method)

def create_multimodal_selector_pipeline(selector, model, fusion_method="average", n_features=16):
    """
    Create a multi-modal selector pipeline that processes each modality separately.
    
    The correct selector pipeline architecture:
    1. Select per modality: selector(mirna) → (n_samples, n_features), selector(exp) → (n_samples, n_features), etc.
    2. Fusion: Combine selected features using fusion method
    3. Model training: Train on fused features
    
    This is similar to the extractor pipeline but uses feature selectors instead of extractors.
    """
    from models import build_selector, build_model
    
    class MultiModalSelectorPipeline:
        def __init__(self, selector_name, model_name, fusion_method="average", n_features=16):
            self.selector_name = selector_name
            self.model_name = model_name
            self.fusion_method = fusion_method
            self.n_features = n_features
            self.selectors = {}  # One selector per modality
            self.model = None
            self.modality_names = None
            
        def fit(self, X_modalities, y):
            """
            Fit the multi-modal selector pipeline.
            
            Parameters:
            - X_modalities: dict of {modality_name: X_array}
            - y: target array
            """
            import numpy as np
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from models import build_selector, build_model
            
            self.modality_names = list(X_modalities.keys())
            
            # Step 1: Fit selectors separately for each modality
            selected_features = {}
            for modality_name, X_modality in X_modalities.items():
                # Build and fit selector for this modality
                selector_obj = build_selector(self.selector_name, self.n_features)
                if selector_obj is None:
                    raise ValueError(f"Failed to build selector: {self.selector_name}")
                selector_obj.fit(X_modality, y)
                self.selectors[modality_name] = selector_obj
                
                # Select features
                X_selected = selector_obj.transform(X_modality)
                selected_features[modality_name] = X_selected
            
            # Step 2: Fuse selected features
            X_fused = self._fuse_features(selected_features)
            
            # Step 3: Fit model on fused features
            task_type = "reg" if "Regressor" in self.model_name else "clf"
            self.model = build_model(self.model_name, task_type)
            if self.model is None:
                raise ValueError(f"Failed to build model: {self.model_name}")
            self.model.fit(X_fused, y)
            
            return self
        
        def predict(self, X_modalities):
            """
            Make predictions with the multi-modal selector pipeline.
            
            Parameters:
            - X_modalities: dict of {modality_name: X_array}
            """
            import numpy as np
            
            # Step 1: Select features from each modality
            selected_features = {}
            for modality_name, X_modality in X_modalities.items():
                if modality_name not in self.selectors:
                    raise ValueError(f"Modality {modality_name} not found in fitted selectors")
                
                X_selected = self.selectors[modality_name].transform(X_modality)
                selected_features[modality_name] = X_selected
            
            # Step 2: Fuse selected features
            X_fused = self._fuse_features(selected_features)
            
            # Step 3: Make predictions
            return self.model.predict(X_fused)
        
        def _fuse_features(self, selected_features):
            """
            Fuse selected features from multiple modalities using proper fusion techniques.
            
            Parameters:
            - selected_features: dict of {modality_name: selected_X_array}
            
            Returns:
            - X_fused: fused feature matrix
            """
            import numpy as np
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from fusion import merge_modalities
            
            modality_arrays = list(selected_features.values())
            
            if len(modality_arrays) == 1:
                return modality_arrays[0]
            
            # Use proper fusion from fusion.py
            try:
                # For multi-modal pipeline, we don't have y during transform, so use simpler fusion
                if self.fusion_method == "average":
                    # Proper average fusion: element-wise mean after scaling
                    from sklearn.preprocessing import RobustScaler
                    scaled_arrays = []
                    
                    for i, arr in enumerate(modality_arrays):
                        scaler = RobustScaler()
                        try:
                            arr_scaled = scaler.fit_transform(arr)
                            arr_scaled = np.clip(arr_scaled, -5, 5)  # Clip outliers
                            scaled_arrays.append(arr_scaled.astype(np.float32))
                        except Exception as e:
                            # Fallback: use original array
                            scaled_arrays.append(arr.astype(np.float32))
                    
                    # Element-wise average
                    fused = np.mean(scaled_arrays, axis=0)
                    return fused
                    
                elif self.fusion_method == "attention_weighted":
                    # For attention-weighted, we need concatenation since we don't have y for attention computation
                    # This is a limitation - true attention needs targets during fitting
                    # Fall back to variance-based weighting then concatenation
                    modality_weights = []
                    for arr in modality_arrays:
                        feature_variances = np.var(arr, axis=0)
                        mean_variance = np.mean(feature_variances)
                        modality_weights.append(mean_variance)
                    
                    # Normalize weights
                    total_weight = sum(modality_weights)
                    if total_weight > 0:
                        modality_weights = [w / total_weight for w in modality_weights]
                    else:
                        modality_weights = [1.0 / len(modality_arrays)] * len(modality_arrays)
                    
                    # Apply weights and concatenate
                    weighted_modalities = []
                    for arr, weight in zip(modality_arrays, modality_weights):
                        weighted_modalities.append(arr * weight)
                    
                    return np.column_stack(weighted_modalities)
                
                else:
                    raise ValueError(f"Unsupported fusion method: {self.fusion_method}")
                    
            except Exception as e:
                # Fallback to concatenation if fusion fails
                print(f"Warning: Fusion failed ({e}), falling back to concatenation")
                return np.column_stack(modality_arrays)
        
        def set_params(self, **params):
            """
            Set parameters for the pipeline components.
            
            Parameters should be in format:
            - selector__param_name: parameter for all selectors (not supported for fixed selectors)
            - model__param_name: parameter for the model
            """
            model_params = {}
            
            for param_name, param_value in params.items():
                if param_name.startswith('model__'):
                    # Remove the model__ prefix
                    actual_param = param_name.replace('model__', '')
                    model_params[actual_param] = param_value
                elif param_name.startswith('selector__'):
                    # For fixed selectors, we don't change selector parameters during tuning
                    # The selector parameters (like n_features) are fixed
                    pass
            
            # Apply model parameters
            if model_params and self.model is not None:
                self.model.set_params(**model_params)
            
            return self
        
        def get_params(self, deep=True):
            """Get parameters for the pipeline."""
            params = {}
            
            # For selectors, we don't expose selector parameters since they're fixed
            # Only model parameters are tuned
            
            # Get model parameters
            if self.model is not None:
                model_params = self.model.get_params(deep=deep)
                for param_name, param_value in model_params.items():
                    params[f'model__{param_name}'] = param_value
            
            return params
    
    return MultiModalSelectorPipeline(selector, model, fusion_method, n_features)

def feature_first_simulate(X_modalities, y, model, cv_params, hyperparams, logger, fusion_method="average", n_features=16):
    """
    Simulate model training with fixed feature selection from the main pipeline.
    
    This matches the main pipeline architecture where:
    1. Each modality gets preprocessed separately (already done - we receive preprocessed modalities)
    2. Feature extraction/selection applied externally to get exactly n_features
    3. Fusion applied to selected features
    4. Model training on fused features (tuned hyperparameters here)
    
    Parameters:
    - X_modalities: dict of {modality_name: preprocessed_X_array} from 4-phase preprocessing
    - y: target array
    - model: model name (RandomForestClassifier, etc.)
    - cv_params: cross-validation parameters
    - hyperparams: hyperparameters to test
    - logger: logger instance
    - fusion_method: fusion method for combining modalities
    - n_features: fixed number of features to simulate (8, 16, or 32)
    
    Returns:
    - Cross-validation scores
    """
    try:
        import numpy as np
        import warnings
        from sklearn.model_selection import cross_val_score
        from sklearn.base import clone
        import logging
        
        # Handle None logger
        if logger is None:
            logger = logging.getLogger(__name__)
        
        # Step 1: Simulate feature selection to get exactly n_features from each modality
        # This simulates the main pipeline where feature extraction/selection reduces each modality
        selected_modalities = {}
        for modality_name, modality_array in X_modalities.items():
            n_available = modality_array.shape[1]
            
            # Handle case where n_features is None (for extractors approach)
            if n_features is None:
                # For extractors, use all available features (no fixed feature count)
                selected_X = modality_array
                logger.debug(f"Extractor approach: Using all {n_available} features from {modality_name}")
            elif n_available >= n_features:
                # Select top n_features (simulate variance-based selection)
                feature_variances = np.var(modality_array, axis=0)
                top_indices = np.argsort(feature_variances)[-n_features:]  # Top variance features
                selected_X = modality_array[:, top_indices]
                logger.debug(f"Feature selection for {modality_name}: {modality_array.shape[1]} -> {n_features}")
            else:
                # If fewer features available, use all and pad with zeros
                logger.debug(f"Modality {modality_name} has only {n_available} features, less than target {n_features}")
                padding = np.zeros((modality_array.shape[0], n_features - n_available))
                selected_X = np.column_stack([modality_array, padding])
                logger.debug(f"Feature selection for {modality_name}: {modality_array.shape[1]} -> {n_features} (padded)")
            
            selected_modalities[modality_name] = selected_X
        
        # Step 2: Apply fusion to selected features based on fusion_method
        if fusion_method == "average":
            # Simple concatenation (current implementation)
            modality_arrays = list(selected_modalities.values())
            if len(modality_arrays) == 1:
                X_input = modality_arrays[0]
            else:
                X_input = np.column_stack(modality_arrays)
        
        elif fusion_method == "attention_weighted":
            # Attention-weighted fusion - compute attention weights based on modality variance
            modality_arrays = list(selected_modalities.values())
            modality_names = list(selected_modalities.keys())
            
            if len(modality_arrays) == 1:
                X_input = modality_arrays[0]
                logger.debug("Single modality - attention weights not applicable")
            else:
                # Calculate attention weights based on modality feature variance
                modality_weights = []
                for i, (name, arr) in enumerate(zip(modality_names, modality_arrays)):
                    # Use feature variance as attention signal
                    feature_variances = np.var(arr, axis=0)
                    mean_variance = np.mean(feature_variances)
                    modality_weights.append(mean_variance)
                
                # Normalize weights to sum to 1
                total_weight = sum(modality_weights)
                if total_weight > 0:
                    modality_weights = [w / total_weight for w in modality_weights]
                else:
                    # Fallback to equal weights if all variances are zero
                    modality_weights = [1.0 / len(modality_arrays)] * len(modality_arrays)
                
                logger.debug(f"Attention weights: {dict(zip(modality_names, modality_weights))}")
                
                # Apply attention weights to each modality
                weighted_modalities = []
                for arr, weight in zip(modality_arrays, modality_weights):
                    weighted_modalities.append(arr * weight)
                
                # Concatenate weighted modalities
                X_input = np.column_stack(weighted_modalities)
        
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")
        
        logger.debug(f"Input after feature selection and fusion ({fusion_method}): {X_input.shape}")
        logger.debug(f"Selected modality shapes: {[(name, arr.shape) for name, arr in selected_modalities.items()]}")
        logger.debug(f"Target feature count per modality: {n_features if n_features is not None else 'Variable (extractor-dependent)'}")
        
        # Step 3: Validate dataset size for cross-validation
        n_samples, n_total_features = X_input.shape
        cv_splitter = cv_params.get('cv')
        
        # Check if we have enough samples for meaningful CV
        if hasattr(cv_splitter, 'n_splits'):
            n_splits = cv_splitter.n_splits
        else:
            # Try to get n_splits from the splitter
            try:
                n_splits = cv_splitter.get_n_splits(X_input, y)
            except:
                n_splits = 5  # Default fallback
        
        min_samples_per_fold = n_samples // n_splits
        
        # CRITICAL FIX: If folds are too small, use a safer CV strategy
        if min_samples_per_fold < 3:
            logger.warning(f"Very small folds detected (~{min_samples_per_fold} samples/fold). "
                          f"Using safer CV strategy for {n_samples} samples.")
            
            # Use a more conservative CV approach for very small datasets
            if n_samples < 10:
                # For very small datasets, use Leave-One-Out or simple split
                logger.warning(f"Dataset too small ({n_samples} samples) for reliable CV, using simple validation")
                from sklearn.model_selection import train_test_split
                
                # Use a simple train-test split instead of CV
                try:
                    # Try stratified split for classification
                    stratify = None
                    if len(np.unique(y)) < n_samples / 2:  # Likely classification
                        unique_classes, class_counts = np.unique(y, return_counts=True)
                        if np.min(class_counts) >= 2:  # Each class needs at least 2 samples
                            stratify = y
                    
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_input, y, test_size=0.3, random_state=42, stratify=stratify
                    )
                    
                    # Create model directly (no extractors needed - features already selected)
                    from models import build_model
                    model_obj = build_model(model, "reg" if "Regressor" in model else "clf")
                    
                    # Apply hyperparameters and fit
                    model_obj.set_params(**hyperparams)
                    model_obj.fit(X_train, y_train)
                    y_pred = model_obj.predict(X_val)
                    
                    # Calculate score using the same scorer as CV
                    scorer = cv_params.get('scoring')
                    if scorer:
                        score = scorer._score_func(y_val, y_pred)
                    else:
                        # Fallback scoring
                        if len(np.unique(y)) < n_samples / 2:  # Classification
                            from sklearn.metrics import accuracy_score
                            score = accuracy_score(y_val, y_pred)
                        else:  # Regression
                            from sklearn.metrics import r2_score
                            score = r2_score(y_val, y_pred)
                    
                    logger.debug(f"Simple validation score: {score:.4f}")
                    return np.array([score])
                    
                except Exception as split_error:
                    logger.warning(f"Simple validation failed: {split_error}, returning poor score")
                    return np.array([-1.0])
            
            else:
                # For small but not tiny datasets, use safer CV
                from sklearn.model_selection import KFold
                safe_splits = max(2, min(3, n_samples // 4))
                cv_params = cv_params.copy()
                cv_params['cv'] = KFold(n_splits=safe_splits, shuffle=True, random_state=42)
                logger.info(f"Using safer {safe_splits}-fold CV for small dataset")
        
        # Step 4: Create model directly (no extractors - features already selected to n_features)
        try:
            from models import build_model
            model_obj = build_model(model, "reg" if "Regressor" in model else "clf")
        except Exception as model_error:
            logger.error(f"Failed to create model {model}: {model_error}")
            return np.array([-1.0])
        
        # Step 5: Apply all hyperparameters to the model
        # Clean parameter names by removing model__ prefix
        clean_hyperparams = {}
        for param_name, param_value in hyperparams.items():
            if param_name.startswith("model__"):
                clean_param_name = param_name.replace("model__", "")
                clean_hyperparams[clean_param_name] = param_value
            else:
                clean_hyperparams[param_name] = param_value
        
        try:
            model_obj.set_params(**clean_hyperparams)
            logger.debug(f"Applied hyperparameters: {clean_hyperparams}")
        except Exception as param_error:
            logger.warning(f"Failed to set hyperparameters: {param_error}")
            # Try with a subset of parameters that are valid
            valid_params = {}
            for param_name, param_value in clean_hyperparams.items():
                try:
                    model_obj.set_params(**{param_name: param_value})
                    valid_params[param_name] = param_value
                except:
                    logger.debug(f"Skipping invalid parameter: {param_name}={param_value}")
            
            if valid_params:
                model_obj.set_params(**valid_params)
                logger.debug(f"Applied valid hyperparameters: {valid_params}")
            else:
                logger.warning("No valid hyperparameters found, using defaults")
        
        logger.debug(f"Model created: {model}")
        logger.debug(f"Input shape: {X_input.shape} (n_features per modality: {n_features})")
        logger.debug(f"Hyperparameters: {hyperparams}")
        
        # Step 6: Enhanced cross-validation with direct model
        try:
            # Validate the CV splitter with the actual data
            cv_splitter = cv_params.get('cv')
            scorer = cv_params.get('scoring')
            
            # Pre-validate CV splits to catch issues early
            split_sizes = []
            try:
                for train_idx, val_idx in cv_splitter.split(X_input, y):
                    split_sizes.append((len(train_idx), len(val_idx)))
                    
                    # Check for empty splits
                    if len(train_idx) == 0 or len(val_idx) == 0:
                        raise ValueError(f"Empty CV split detected: train={len(train_idx)}, val={len(val_idx)}")
                    
                    # Check for minimum size requirement
                    if len(train_idx) < 2:
                        raise ValueError(f"Training split too small: {len(train_idx)} samples")
                
                logger.debug(f"CV split sizes: {split_sizes}")
                
            except Exception as split_validation_error:
                logger.error(f"CV split validation failed: {split_validation_error}")
                return np.array([-1.0])
            
            # Run cross-validation with the direct model
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                
                scores = cross_val_score(
                    estimator=model_obj,
                    X=X_input,
                    y=y,
                    **cv_params
                )
            
            # Validate scores
            if len(scores) == 0:
                logger.warning("No CV scores returned")
                return np.array([-1.0])
            
            # Filter out invalid scores
            valid_scores = scores[np.isfinite(scores)]
            if len(valid_scores) == 0:
                logger.warning("All CV scores were invalid (NaN/Inf)")
                return np.array([-1.0])
            
            if len(valid_scores) < len(scores):
                logger.warning(f"Some CV scores were invalid: {len(valid_scores)}/{len(scores)} valid")
            
            logger.debug(f"CV scores: {valid_scores} (mean: {np.mean(valid_scores):.4f})")
            return valid_scores
            
        except Exception as cv_error:
            logger.error(f"Cross-validation failed: {cv_error}")
            import traceback
            logger.debug(f"CV error traceback: {traceback.format_exc()}")
            
            # Try to provide more specific error information
            if "array of size 0" in str(cv_error) or "zero-size array" in str(cv_error):
                logger.error("CV failed due to empty arrays - likely caused by very small CV folds")
            elif "n_components" in str(cv_error):
                logger.error("CV failed due to PCA component issues - likely too many components for small data")
            
            return np.array([-1.0])
        
    except Exception as e:
        logger.error(f"Feature-first simulation failed: {e}")
        import traceback
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        return np.array([-1.0])  # Return poor score on failure

def tune_extractors(dataset, task, extractor, model, fusion_method="average", logger=None):
    """
    Hyperparameter tuning for EXTRACTOR+MODEL approach (original approach).
    
    Tunes both extractor parameters (n_components, etc.) and model parameters.
    This is for the traditional feature extraction pipeline.
    """
    return _tune_internal(dataset, task, extractor, model, fusion_method, None, "extractors", logger)

def tune_selectors(dataset, task, selector, model, fusion_method="average", n_features=16, logger=None):
    """
    Hyperparameter tuning for SELECTOR+MODEL approach (Option B implementation).
    
    Optimizes model hyperparameters separately for each selector type, recognizing that
    different selectors produce features with different statistical properties.
    Feature selection is handled separately with fixed counts [8, 16, 32].
    """
    return _tune_internal(dataset, task, selector, model, fusion_method, n_features, "selectors", logger)

def _tune_internal(dataset, task, extractor, model, fusion_method="average", n_features=16, approach="selectors", logger=None):
    """Enhanced hyperparameter tuning with Windows resource management for both approaches."""
    
    # Setup logging if not provided
    if logger is None:
        if approach == "extractors":
            logger, log_path = setup_logging(dataset, extractor, model, log_level=logging.INFO)
        else:  # selectors
            logger, log_path = setup_logging(dataset, "FixedFeatures", model, log_level=logging.INFO)
    
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
    if approach == "extractors":
        log_stage(logger, "TUNING_INITIALIZATION", {
            "dataset": dataset,
            "task": task,
            "extractor": extractor,
            "model": model,
            "fusion_method": fusion_method,
            "pipeline_type": "Extractor + Model Tuning",
            "seed": SEED,
            "cv_folds": CV_INNER
        })
    else:  # selectors
        log_stage(logger, "TUNING_INITIALIZATION", {
            "dataset": dataset,
            "task": task,
            "model": model,
            "fusion_method": fusion_method,
            "n_features": n_features,
            "pipeline_type": "Fixed Feature Selection + Model Tuning",
            "seed": SEED,
            "cv_folds": CV_INNER
        })
    
    try:
        # Stage 1: Data Loading with Feature-First Pipeline
        log_stage(logger, "DATA_LOADING", {
            "pipeline_type": "4-Phase Enhanced Pipeline + Feature-First Architecture",
            "phases": [
                "Phase 1: Early Data Quality Assessment",
                "Phase 2: Modality-Specific Preprocessing (miRNA->150, exp->1500, methy->2000)",
                "Phase 3: Centralized Missing Data Management", 
                "Phase 4: Coordinated Validation Framework"
            ],
            "architecture": "Feature-First: Preprocessing -> Feature Processing -> Fusion -> Model"
        })
        
        logger.info(f"Loading {dataset} with Feature-First 4-Phase Enhanced Pipeline...")
        
        # Load preprocessed modalities separately (not fused) for feature-first pipeline
        # Use cached version to avoid redundant preprocessing across combinations
        processed_modalities, y, sample_ids_cached, baseline_mae = load_dataset_for_tuner_cached(dataset, task=task)
        
        logger.info(f"Feature-first data loaded successfully:")
        logger.info(f"  Preprocessed modalities: {list(processed_modalities.keys())}")
        logger.info(f"  Total samples: {len(y)}")
        
        # Log modality shapes after preprocessing
        total_preprocessed_features = 0
        for modality_name, modality_array in processed_modalities.items():
            logger.info(f"  {modality_name}: {modality_array.shape}")
            total_preprocessed_features += modality_array.shape[1]
        
        logger.info(f"  Total preprocessed features: {total_preprocessed_features}")
        
        # Use cached baseline MAE if available, otherwise compute it
        if baseline_mae is None and task == "reg":
            baseline_mae = compute_baseline_mae(y)
        
        if baseline_mae is not None:
            logger.info(f"  Baseline MAE (mean prediction): {baseline_mae:.4f}")
        
        # Stage 2: Approach-Specific Preparation
        if approach == "extractors":
            log_stage(logger, "EXTRACTOR_PIPELINE_PREPARATION", {
                "step_1": "Preprocessed modalities ready for extractor-based pipeline",
                "step_2": "Feature extraction with variable components per modality",
                "step_3": "Traditional sklearn pipeline with extractor+model tuning"
            })
            
            logger.info("Extractor pipeline preparation completed")
            logger.info("Note: Feature extraction components will be tuned dynamically")
            logger.info("Each hyperparameter test will use: extractor -> fusion -> model pipeline")
        else:  # selectors
            log_stage(logger, "FIXED_FEATURE_PREPARATION", {
                "step_1": "Preprocessed modalities ready for fixed feature selection simulation",
                "step_2": f"Each modality will be reduced to exactly {n_features} features",
                "step_3": "Fusion and model training with optimized hyperparameters"
            })
            
            logger.info("Fixed feature selection preparation completed")
            logger.info(f"Note: Modalities will be reduced to {n_features} features each")
            logger.info("Each hyperparameter test will simulate: feature selection -> fusion -> model training")
        
        # Use sample IDs from cache if available, otherwise try to load them
        sample_ids = sample_ids_cached
        if sample_ids is None:
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
                    task_type_for_ids = 'regression' if task == 'reg' else 'classification'
                    modalities_data, y_series, common_ids, is_regression = load_dataset(
                        ds_name=dataset.lower(),
                        modalities=modality_short_names,
                        outcome_col=config['outcome_col'],
                        task_type=task_type_for_ids,
                        parallel=True,
                        use_cache=True
                    )
                    
                    if common_ids and len(common_ids) == len(y):
                        sample_ids = common_ids
                        logger.info(f"Loaded {len(sample_ids)} real sample IDs for enhanced CV")
                    
            except Exception as e:
                logger.debug(f"Could not load real sample IDs: {e}, will use fallback approach")
        else:
            logger.info(f"Using cached sample IDs: {len(sample_ids)} samples")
        
        # Stage 3: Data Validation
        log_stage(logger, "DATA_VALIDATION")
        
        # Validate data quality of preprocessed modalities
        for modality_name, modality_array in processed_modalities.items():
            if np.any(np.isnan(modality_array)) or np.any(np.isinf(modality_array)):
                logger.warning(f"Found NaN/Inf in {modality_name}, cleaning...")
                processed_modalities[modality_name] = np.nan_to_num(modality_array, nan=0.0, posinf=1e6, neginf=-1e6)
        
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
                    # Filter both modalities and targets
                    for modality_name in processed_modalities:
                        processed_modalities[modality_name] = processed_modalities[modality_name][outlier_mask]
                    y = y[outlier_mask]
                    
                    # Update sample IDs if available
                    if sample_ids is not None and len(sample_ids) == original_size:
                        sample_ids = [sample_ids[i] for i, keep in enumerate(outlier_mask) if keep]
                        logger.info(f"Updated sample IDs after outlier removal: {len(sample_ids)} samples")
                    
                    logger.info(f"Removed {n_outliers} extreme outliers (>{outlier_threshold:.2f}) "
                               f"from dataset ({outlier_percentage:.1f}% of data)")
                    logger.info(f"Dataset size: {original_size} -> {len(y)}")
                    
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
                for modality_name, modality_array in processed_modalities.items():
                    logger.info(f"  {modality_name} shape: {modality_array.shape}")
                logger.info(f"  y shape: {y.shape}")
                
            except Exception as e:
                logger.warning(f"Target outlier removal failed: {e}")
                # Continue without outlier removal if it fails
        
        # Stage 4: Sampler Setup
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

        # Stage 5: Fixed Feature Selection Integration Setup
        log_stage(logger, "FIXED_FEATURE_INTEGRATION_SETUP")
        
        # For fixed feature selection, we don't need extractors - features are pre-selected
        # Instead, we create models directly and tune their hyperparameters
        logger.info("Setting up fixed feature selection simulation for hyperparameter tuning")
        logger.info("Note: Feature selection is simulated, focus is on model hyperparameter optimization")
        
        # Validate that we can build the model
        try:
            # Test that we can create the model
            from models import build_model
            test_model = build_model(model, "reg" if task == "reg" else "clf")
            logger.info(f"Validated model component:")
            logger.info(f"  - Model: {model}")
            logger.info(f"  - Model type: {type(test_model).__name__}")
            logger.info(f"  - Fixed features per modality: {n_features}")
            
        except Exception as e:
            log_error_with_context(logger, e, {
                "operation": "model_validation",
                "model": model,
                "n_features": n_features
            })
            return False
        
        # Note: No extractors needed - features are pre-selected to n_features per modality

        # Stage 6: Fixed Feature Selection CV Setup
        log_stage(logger, "FIXED_FEATURE_CV_SETUP")
        
        # Setup CV parameters for feature_first_simulate function
        if task == "reg":
            primary_scoring = make_scorer(safe_r2_score, greater_is_better=True)
            logger.info("Using scorer for feature-first simulation: R² score")
        else:
            primary_scoring = make_scorer(safe_mcc_score, greater_is_better=True)
            logger.info("Using scorer for feature-first simulation: Matthews correlation coefficient")

        # Enhanced cross-validation strategy with stratified regression and grouped CV
        n_samples = len(y)
        
        # Use enhanced CV splitter with real sample IDs
        try:
            from cv import create_enhanced_cv_splitter, validate_enhanced_cv_strategy, MIN_SAMPLES_PER_FOLD
            
            task_type = 'regression' if task == 'reg' else 'classification'
            max_safe_folds = max(2, min(CV_INNER, n_samples // MIN_SAMPLES_PER_FOLD))
            
            # Create enhanced CV splitter with proper regression/classification strategies
            cv_result = create_enhanced_cv_splitter(
                y=y,
                sample_ids=sample_ids,
                task_type=task_type,
                n_splits=max_safe_folds,
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
                
                # Log additional info for regression CV
                if task == 'reg':
                    logger.info(f"Regression CV strategy: {strategy_desc}")
                    
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

        # Stage 7: Parameter Space Generation
        log_stage(logger, "PARAMETER_SPACE_GENERATION")
        
        # Calculate data dimensions for parameter space (after feature selection simulation)
        total_samples = len(y)
        n_modalities = len(processed_modalities)
        
        # For extractors, we don't have fixed n_features (it's variable based on extractor parameters)
        # For selectors, we use the fixed n_features parameter
        if approach == "extractors":
            # For extractors, use total input features as a rough estimate for parameter space
            total_input_features = sum(mod.shape[1] for mod in processed_modalities.values())
            data_shape = (total_samples, total_input_features)
        else:  # selectors
            total_features_after_selection = n_modalities * n_features
            data_shape = (total_samples, total_features_after_selection)
        
        # Get parameter space based on approach
        if approach == "extractors":
            params = param_space_extractors(extractor, model, data_shape)
        else:  # selectors
            # For selectors approach, extractor parameter contains the selector name
            selector = extractor  # In selectors approach, extractor parameter holds the selector name
            params = param_space_selectors(selector, model, n_features, data_shape)
        n_combinations = count_parameter_combinations(params)
        
        logger.info(f"Model parameter combinations: {n_combinations}")
        logger.info(f"Data shape for parameter space: {data_shape} (samples, features)")
        
        if approach == "extractors":
            logger.info(f"Extractor approach: Variable features per modality, Total modalities: {n_modalities}")
            logger.info(f"Total input features: {data_shape[1]}")
        else:  # selectors
            logger.info(f"Selector approach: {n_features} features per modality, Total modalities: {n_modalities}")
            logger.info(f"Total selected features: {data_shape[1]}")
        
        logger.debug(f"Parameter space: {params}")
        
        # Stage 8: Approach-Specific Parameter Search Strategy
        if approach == "extractors":
            log_stage(logger, "EXTRACTOR_PARAMETER_SEARCH")
            
            logger.info("Using traditional sklearn pipeline parameter search for extractors")
            logger.info(f"Parameter combinations to evaluate: {n_combinations}")
            logger.info("Note: Each combination tested with sklearn GridSearchCV using extractor pipelines")
        else:  # selectors
            log_stage(logger, "FIXED_FEATURE_PARAMETER_SEARCH")
            
            logger.info("Using custom fixed feature selection parameter search")
            logger.info(f"Parameter combinations to evaluate: {n_combinations}")
            logger.info("Note: Each combination tested with feature_first_simulate() using fixed feature counts")
        
        # Setup approach-specific search parameters
        if approach == "extractors":
            # Multi-modal pipeline approach for extractors
            logger.info("Setting up multi-modal pipeline for extractor approach...")
            
            # Log the correct data shapes for each modality
            for modality_name, modality_data in processed_modalities.items():
                logger.info(f"Input {modality_name} shape: {modality_data.shape}")
            
            # Create custom multi-modal grid search
            from sklearn.model_selection import ParameterGrid
            
            # Limit parameter combinations for Windows stability
            max_combinations = 30  # Conservative limit for extractors
            param_combinations = list(ParameterGrid(params))
            
            if len(param_combinations) > max_combinations:
                logger.warning(f"Too many combinations ({len(param_combinations)}) - sampling {max_combinations}")
                np.random.seed(SEED)
                selected_indices = np.random.choice(len(param_combinations), max_combinations, replace=False)
                param_combinations = [param_combinations[i] for i in selected_indices]
                logger.info(f"Reduced to {len(param_combinations)} parameter combinations")
            
            # Stage 9: Multi-Modal Pipeline Search Execution
            log_stage(logger, "MULTIMODAL_SEARCH_EXECUTION", {
                "search_type": "Custom Multi-Modal Grid Search",
                "n_combinations": len(param_combinations),
                "backend": "sequential"
            })
            
            logger.info(f"Starting multi-modal pipeline search for {extractor} + {model}...")
            search_start_time = time.time()
            
            try:
                # Force garbage collection before search
                gc.collect()
                
                # Initialize tracking variables
                best_score = float('-inf')
                best_params = None
                results_list = []
                
                # Iterate through parameter combinations
                for i, param_combination in enumerate(param_combinations):
                    try:
                        logger.debug(f"Testing combination {i+1}/{len(param_combinations)}: {param_combination}")
                        
                        # Create and configure pipeline
                        pipeline = create_multimodal_pipeline(extractor, model, fusion_method)
                        pipeline.set_params(**param_combination)
                        
                        # Perform cross-validation
                        cv_scores = []
                        for train_idx, val_idx in cv_inner.split(list(processed_modalities.values())[0], y):
                            # Split modalities for this fold
                            X_train_modalities = {}
                            X_val_modalities = {}
                            for modality_name, modality_data in processed_modalities.items():
                                X_train_modalities[modality_name] = modality_data[train_idx]
                                X_val_modalities[modality_name] = modality_data[val_idx]
                            
                            y_train, y_val = y[train_idx], y[val_idx]
                            
                            # Fit and predict
                            pipeline.fit(X_train_modalities, y_train)
                            y_pred = pipeline.predict(X_val_modalities)
                            
                            # Calculate score
                            if primary_scoring == 'r2':
                                from sklearn.metrics import r2_score
                                score = r2_score(y_val, y_pred)
                            elif primary_scoring == 'neg_mean_absolute_error':
                                from sklearn.metrics import mean_absolute_error
                                score = -mean_absolute_error(y_val, y_pred)
                            else:
                                # Use scorer function
                                score = primary_scoring._score_func(y_val, y_pred)
                            
                            cv_scores.append(score)
                        
                        # Calculate mean score
                        mean_score = np.mean(cv_scores)
                        std_score = np.std(cv_scores)
                        
                        # Track result
                        results_list.append({
                            'params': param_combination.copy(),
                            'mean_score': mean_score,
                            'std_score': std_score,
                            'cv_scores': cv_scores.copy()
                        })
                        
                        # Check if this is the best so far
                        if np.isfinite(mean_score) and mean_score > best_score:
                            best_score = mean_score
                            best_params = param_combination.copy()
                            logger.info(f"New best score: {best_score:.4f} ± {std_score:.4f}")
                            logger.debug(f"Best params so far: {best_params}")
                        
                        # Periodic memory check
                        if (i + 1) % 10 == 0:
                            gc.collect()
                            
                    except Exception as e:
                        logger.warning(f"Parameter combination {i+1} failed: {str(e)}")
                        continue
                
                search_elapsed = time.time() - search_start_time
                logger.info(f"Multi-modal pipeline search completed in {search_elapsed:.1f} seconds")
                logger.info(f"Best score: {best_score:.4f}")
                logger.info(f"Best parameters: {best_params}")
                
                # Create results object for compatibility
                class MultiModalSearchResults:
                    def __init__(self, best_params, best_score, results_list):
                        self.best_params_ = best_params
                        self.best_score_ = best_score
                        self.cv_results_ = results_list
                
                search_results = MultiModalSearchResults(best_params, best_score, results_list)
                
            except Exception as e:
                logger.error(f"Multi-modal pipeline search failed: {str(e)}")
                gc.collect()
                return False
        
        else:  # selectors approach
            # Multi-modal selector pipeline approach for selectors
            logger.info("Setting up multi-modal selector pipeline for selectors approach...")
            
            # Log the correct data shapes for each modality
            for modality_name, modality_data in processed_modalities.items():
                logger.info(f"Input {modality_name} shape: {modality_data.shape}")
            
            # Create custom multi-modal selector grid search
            from sklearn.model_selection import ParameterGrid
            
            # Limit parameter combinations for Windows stability
            max_combinations = 30  # Conservative limit for selectors
            param_combinations = list(ParameterGrid(params))
            
            if len(param_combinations) > max_combinations:
                logger.warning(f"Too many combinations ({len(param_combinations)}) - sampling {max_combinations}")
                np.random.seed(SEED)
                selected_indices = np.random.choice(len(param_combinations), max_combinations, replace=False)
                param_combinations = [param_combinations[i] for i in selected_indices]
                logger.info(f"Reduced to {len(param_combinations)} parameter combinations")
            
            # Stage 9: Multi-Modal Selector Pipeline Search Execution
            log_stage(logger, "MULTIMODAL_SELECTOR_SEARCH_EXECUTION", {
                "search_type": "Custom Multi-Modal Selector Grid Search",
                "n_combinations": len(param_combinations),
                "backend": "sequential"
            })
            
            logger.info(f"Starting multi-modal selector pipeline search for {extractor} + {model}...")
            search_start_time = time.time()
            
            try:
                # Force garbage collection before search
                gc.collect()
                
                # Initialize tracking variables
                best_score = float('-inf')
                best_params = None
                results_list = []
                
                # Iterate through parameter combinations
                for i, param_combination in enumerate(param_combinations):
                    try:
                        logger.debug(f"Testing combination {i+1}/{len(param_combinations)}: {param_combination}")
                        
                        # Create and configure selector pipeline
                        selector = extractor  # In selectors approach, extractor parameter holds the selector name
                        pipeline = create_multimodal_selector_pipeline(selector, model, fusion_method, n_features)
                        pipeline.set_params(**param_combination)
                        
                        # Perform cross-validation
                        cv_scores = []
                        for train_idx, val_idx in cv_inner.split(list(processed_modalities.values())[0], y):
                            # Split modalities for this fold
                            X_train_modalities = {}
                            X_val_modalities = {}
                            for modality_name, modality_data in processed_modalities.items():
                                X_train_modalities[modality_name] = modality_data[train_idx]
                                X_val_modalities[modality_name] = modality_data[val_idx]
                            
                            y_train, y_val = y[train_idx], y[val_idx]
                            
                            # Fit and predict
                            pipeline.fit(X_train_modalities, y_train)
                            y_pred = pipeline.predict(X_val_modalities)
                            
                            # Calculate score
                            if primary_scoring == 'r2':
                                from sklearn.metrics import r2_score
                                score = r2_score(y_val, y_pred)
                            elif primary_scoring == 'neg_mean_absolute_error':
                                from sklearn.metrics import mean_absolute_error
                                score = -mean_absolute_error(y_val, y_pred)
                            else:
                                # Use scorer function
                                score = primary_scoring._score_func(y_val, y_pred)
                            
                            cv_scores.append(score)
                        
                        # Calculate mean score
                        mean_score = np.mean(cv_scores)
                        std_score = np.std(cv_scores)
                        
                        # Track result
                        results_list.append({
                            'params': param_combination.copy(),
                            'mean_score': mean_score,
                            'std_score': std_score,
                            'cv_scores': cv_scores.copy()
                        })
                        
                        # Check if this is the best so far
                        if np.isfinite(mean_score) and mean_score > best_score:
                            best_score = mean_score
                            best_params = param_combination.copy()
                            logger.info(f"New best score: {best_score:.4f} ± {std_score:.4f}")
                            logger.debug(f"Best params so far: {best_params}")
                        
                        # Periodic memory check
                        if (i + 1) % 10 == 0:
                            gc.collect()
                            
                    except Exception as e:
                        logger.warning(f"Parameter combination {i+1} failed: {str(e)}")
                        continue
                
                search_elapsed = time.time() - search_start_time
                logger.info(f"Multi-modal selector pipeline search completed in {search_elapsed:.1f} seconds")
                logger.info(f"Best score: {best_score:.4f}")
                logger.info(f"Best parameters: {best_params}")
                
                # Create results object for compatibility
                class MultiModalSelectorSearchResults:
                    def __init__(self, best_params, best_score, results_list):
                        self.best_params_ = best_params
                        self.best_score_ = best_score
                        self.cv_results_ = results_list
                
                search_results = MultiModalSelectorSearchResults(best_params, best_score, results_list)
                
            except Exception as e:
                logger.error(f"Multi-modal selector pipeline search failed: {str(e)}")
                gc.collect()
                return False
        
        # Unified result handling for both approaches
        search = search_results
        
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
                            # NOTE: This section is disabled for feature-first pipeline
                            # because search.best_estimator_ doesn't exist in our custom search
                            # and we don't have a single X matrix (we use separate modalities)
                            logger.debug("Traditional MAE computation disabled for feature-first pipeline")
                            
                            # # Use cross-validation to compute MAE with the best parameters
                            # from sklearn.model_selection import cross_val_score
                            # 
                            # # Suppress sklearn warnings during MAE computation
                            # with warnings.catch_warnings():
                            #     warnings.filterwarnings("ignore", 
                            #                           message="The least populated class in y has only .* members", 
                            #                           category=UserWarning)
                            #     mae_scores = cross_val_score(
                            #         search.best_estimator_, X, y, 
                            #         cv=cv_inner, 
                            #         scoring=make_scorer(safe_mae_score, greater_is_better=True),
                            #         n_jobs=1  # Single job to avoid conflicts
                            #     )
                            # 
                            # actual_mae = -np.mean(mae_scores)  # Convert back to positive MAE
                            # best_mae_score = -actual_mae  # Store as negative for consistency
                            # 
                            # logger.info(f"Best MAE score (computed): {actual_mae:.4f}" + 
                            #            (f" (baseline: {baseline_mae:.4f})" if baseline_mae is not None else ""))
                            # 
                            # # Check if MAE improved over baseline
                            # if baseline_mae is not None:
                            #     mae_improvement = baseline_mae - actual_mae
                            #     mae_improvement_pct = (mae_improvement / baseline_mae) * 100
                            #     logger.info(f"MAE improvement: {mae_improvement:.4f} ({mae_improvement_pct:+.1f}%)")
                            #     
                            #     if mae_improvement > 0:
                            #         logger.info("✓ MAE improved over baseline")
                            #     else:
                            #         logger.warning(" MAE did not improve over baseline")
                        except Exception as e:
                            logger.warning(f"Could not compute MAE manually: {e}")
                            logger.info(f"Best R² score: {best_r2_score:.4f}")
                    else:
                        logger.info(f"Best R² score: {best_r2_score:.4f}")
                        logger.warning("No best estimator available for MAE computation (expected for feature-first pipeline)")
                        
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

        # Stage 10: Results Processing and Saving
        log_stage(logger, "RESULTS_PROCESSING")
        
        # Save results with enhanced metadata for fixed feature selection
        best = {
            "best_params": search.best_params_,
            "best_score": search.best_score_,
            "dataset": dataset,
            "task": task,
            "model": model,
            "fusion_method": fusion_method,
            "n_features": n_features,
            "preprocessing": "fixed-feature-selection-4-phase-enhanced",
            "data_shape": {
                "total_samples": len(y),
                "n_features_per_modality": n_features,
                "n_modalities": len(processed_modalities),
                "total_features_after_selection": len(processed_modalities) * n_features if n_features is not None else "Variable (extractor-dependent)",
                "modality_shapes_before_selection": {name: arr.shape for name, arr in processed_modalities.items()}
            },
            "n_parameter_combinations": len(param_combinations),
            "search_method": "Custom Fixed Feature Selection Grid Search",
            "cv_folds": cv_inner.n_splits if hasattr(cv_inner, 'n_splits') else CV_INNER,
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

        # Generate filename based on approach
        if approach == "extractors":
            fp = HP_DIR/f"{dataset}_{extractor}_{model}_{fusion_method}.json"
        else:  # selectors
            # Include selector in filename for Option B implementation
            selector = extractor  # In selectors approach, extractor parameter holds the selector name
            fp = HP_DIR/f"{dataset}_{selector}_{model}_{fusion_method}_{n_features}f.json"
        
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
        logger.info(f"  Features per modality: {n_features if n_features is not None else 'Variable (extractor-dependent)'}, Total modalities: {len(processed_modalities)}")
        logger.info(f"  Total features after selection: {len(processed_modalities) * n_features if n_features is not None else 'Variable (extractor-dependent)'}")
        logger.info(f"  Samples: {len(y)}")
        logger.info(f"  Search Time: {search_elapsed:.1f}s")
        logger.info(f"  Preprocessing: Fixed Feature Selection + 4-Phase Enhanced Pipeline")
        
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

def ensure_preprocessing_cache(dataset, task, logger=None):
    """Ensure preprocessing cache exists for a dataset, creating it if necessary."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    cache_data = load_preprocessing_cache(dataset, task)
    if cache_data is None:
        logger.info(f"Pre-caching preprocessing for {dataset} (task: {task})...")
        start_time = time.time()
        
        # Run preprocessing once and cache it
        processed_modalities, y, sample_ids, baseline_mae = load_dataset_for_tuner_cached(dataset, task)
        
        cache_time = time.time() - start_time
        logger.info(f"Preprocessing cached in {cache_time:.1f}s - will be reused for all combinations")
        
        return processed_modalities, y, sample_ids, baseline_mae
    else:
        logger.info(f"Preprocessing cache already exists for {dataset} (task: {task})")
        return (
            cache_data['processed_modalities'],
            cache_data['y'],
            cache_data.get('sample_ids'),
            cache_data.get('baseline_mae')
        )

def tune_all_extractors(dataset, task, use_subprocess=True):
    """Run tuning for all extractor combinations with both fusion methods."""
    
    # Setup session logger
    session_logger, session_log_path = setup_logging()
    
    # Pre-cache preprocessing to avoid redundant work
    session_logger.info(f"Ensuring preprocessing cache for {dataset}...")
    ensure_preprocessing_cache(dataset, task, session_logger)
    
    if task == "reg":
        extractors = REGRESSION_EXTRACTORS
        models = REGRESSION_MODELS
    else:
        extractors = CLASSIFICATION_EXTRACTORS
        models = CLASSIFICATION_MODELS
    
    # Both fusion methods to test
    fusion_methods = ["average", "attention_weighted"]
    
    total_combinations = len(extractors) * len(models) * len(fusion_methods)
    
    log_stage(session_logger, "BATCH_EXTRACTOR_TUNING_INITIALIZATION", {
        "dataset": dataset,
        "task": task,
        "total_combinations": total_combinations,
        "extractors": extractors,
        "models": models,
        "fusion_methods": fusion_methods,
        "subprocess_isolation": use_subprocess,
        "timeout_minutes": TIMEOUT_MINUTES,
        "session_log": str(session_log_path)
    })
    
    successful = 0
    failed = 0
    start_time = time.time()
    
    for i, (extractor, model, fusion_method) in enumerate(product(extractors, models, fusion_methods), 1):
        log_stage(session_logger, f"COMBINATION_{i}_OF_{total_combinations}", {
            "extractor": extractor,
            "model": model,
            "fusion_method": fusion_method,
            "dataset": dataset,
            "preprocessing": "Extractor + Model Tuning"
        })
        
        if use_subprocess:
            success = run_tuning_subprocess_extractors(dataset, task, extractor, model, fusion_method, session_logger)
        else:
            try:
                success = tune_extractors(dataset, task, extractor, model, fusion_method, session_logger)
            except Exception as e:
                log_error_with_context(session_logger, e, {
                    "operation": "direct_tune_extractors_execution",
                    "combination": f"{extractor}+{model}+{fusion_method}",
                    "combination_number": f"{i}/{total_combinations}"
                })
                success = False
        
        if success:
            successful += 1
            session_logger.info(f"✓ COMPLETED ({i}/{total_combinations}): {extractor} + {model} + {fusion_method}")
        else:
            failed += 1
            session_logger.error(f"✗ FAILED ({i}/{total_combinations}): {extractor} + {model} + {fusion_method}")
    
    total_time = time.time() - start_time
    
    log_stage(session_logger, "BATCH_EXTRACTOR_TUNING_COMPLETED", {
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

def tune_all_selectors(dataset, task, use_subprocess=True):
    """Run tuning for all selector+model combinations with fixed feature counts, both fusion methods (Option B implementation)."""
    
    # Setup session logger
    session_logger, session_log_path = setup_logging()
    
    # Pre-cache preprocessing to avoid redundant work
    session_logger.info(f"Ensuring preprocessing cache for {dataset}...")
    ensure_preprocessing_cache(dataset, task, session_logger)
    
    if task == "reg":
        selectors = REGRESSION_SELECTORS
        models = REGRESSION_MODELS
    else:
        selectors = CLASSIFICATION_SELECTORS
        models = CLASSIFICATION_MODELS
    
    # Both fusion methods to test
    fusion_methods = ["average", "attention_weighted"]
    
    # Fixed feature counts to test
    feature_counts = FEATURE_COUNTS  # [8, 16, 32]
    
    total_combinations = len(selectors) * len(models) * len(fusion_methods) * len(feature_counts)
    
    log_stage(session_logger, "BATCH_SELECTOR_TUNING_INITIALIZATION", {
        "dataset": dataset,
        "task": task,
        "total_combinations": total_combinations,
        "selectors": selectors,
        "models": models,
        "fusion_methods": fusion_methods,
        "feature_counts": feature_counts,
        "subprocess_isolation": use_subprocess,
        "timeout_minutes": TIMEOUT_MINUTES,
        "session_log": str(session_log_path),
        "implementation": "Option B - Selector-specific hyperparameter optimization"
    })
    
    successful = 0
    failed = 0
    start_time = time.time()
    
    for i, (selector, model, fusion_method, n_features) in enumerate(product(selectors, models, fusion_methods, feature_counts), 1):
        log_stage(session_logger, f"COMBINATION_{i}_OF_{total_combinations}", {
            "selector": selector,
            "model": model,
            "fusion_method": fusion_method,
            "n_features": n_features,
            "dataset": dataset,
            "preprocessing": "Selector-specific Feature Selection + 4-Phase Enhanced Pipeline"
        })
        
        if use_subprocess:
            success = run_tuning_subprocess_selectors(dataset, task, selector, model, fusion_method, n_features, session_logger)
        else:
            # Direct execution (for debugging)
            try:
                success = tune_selectors(dataset, task, selector, model, fusion_method, n_features, session_logger)
            except Exception as e:
                log_error_with_context(session_logger, e, {
                    "operation": "direct_tune_selectors_execution",
                    "combination": f"{selector}+{model}+{fusion_method}+{n_features}f",
                    "combination_number": f"{i}/{total_combinations}"
                })
                success = False
        
        if success:
            successful += 1
            session_logger.info(f"✓ COMPLETED ({i}/{total_combinations}): {selector} + {model} + {fusion_method} + {n_features}f")
        else:
            failed += 1
            session_logger.error(f"✗ FAILED ({i}/{total_combinations}): {selector} + {model} + {fusion_method} + {n_features}f")
    
    total_time = time.time() - start_time
    
    log_stage(session_logger, "BATCH_SELECTOR_TUNING_COMPLETED", {
        "dataset": dataset,
        "total_time_minutes": f"{total_time/60:.1f}",
        "successful_combinations": successful,
        "failed_combinations": failed,
        "total_combinations": total_combinations,
        "success_rate": f"{successful/total_combinations:.1%}",
        "implementation": "Option B - Each selector gets separate hyperparameter optimization"
    })
    
    if successful > 0:
        # List generated files
        dataset_files = list(HP_DIR.glob(f"{dataset}_*.json"))
        if dataset_files:
            session_logger.info("Generated hyperparameter files:")
            for f in sorted(dataset_files):
                session_logger.info(f"  - {f.name}")
                
        # Show sample of best hyperparameters
        session_logger.info("Sample of selector-specific optimized hyperparameters:")
        for f in sorted(dataset_files)[:5]:  # Show first 5
            try:
                with open(f, 'r') as file:
                    data = json.load(file)
                    n_feat = data.get('n_features', 'N/A')
                    selector_name = f.name.split('_')[1] if '_' in f.name else 'Unknown'
                    session_logger.info(f"  {f.name}: Selector={selector_name}, Score={data['best_score']:.4f}, Features={n_feat}")
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
    
    The Matthews correlation coefficient (MCC) is particularly important for genomic/biomedical data
    because it provides a balanced measure of classification quality even with severe class imbalance,
    which is common in cancer genomics where:
    
    1. **Patient outcome prediction**: Often have unequal numbers of good vs poor outcomes
    2. **Biomarker discovery**: Most genes/features are non-informative (negative class >> positive class)  
    3. **Multi-omics integration**: Different data types may have different class distributions
    4. **Small sample studies**: Limited patients available, leading to extreme ratios
    
    MCC accounts for all four confusion matrix categories (TP, TN, FP, FN) and produces a 
    correlation coefficient between observed and predicted classifications:
    - MCC = +1: Perfect prediction
    - MCC = 0: Random performance  
    - MCC = -1: Perfect inverse prediction
    
    This is more robust than accuracy, precision, recall, or F1-score for imbalanced datasets
    commonly found in genomic studies.
    
    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_pred : array-like  
        Predicted binary labels
    **kwargs : dict
        Additional arguments passed to matthews_corrcoef
        
    Returns
    -------
    float
        MCC score, with fallbacks for edge cases:
        - Returns -1.0 for invalid inputs (worst performance)
        - Returns 0.0 for undefined cases (random performance)
        - Returns 1.0 for perfect single-class prediction
    """
    import numpy as np
    try:
        # Basic validation
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
        
        # Calculate MCC using sklearn's implementation
        mcc = matthews_corrcoef(y_true, y_pred, **kwargs)
        
        # Handle NaN cases (can occur with extreme class imbalance)
        if np.isnan(mcc):
            return 0.0  # No correlation (random performance)
        
        if not np.isfinite(mcc):
            return -1.0
            
        return mcc
        
    except Exception as e:
        # Fallback for any MCC calculation issues
        return -1.0  # Worst performance as fallback

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
            # Pre-validate input data
            if X.shape[0] == 0:
                raise ValueError("Cannot fit extractor on empty dataset")
            
            if X.shape[1] == 0:
                raise ValueError("Cannot fit extractor on dataset with no features")
            
            # Check for minimum samples requirement
            if X.shape[0] < 2:
                raise ValueError(f"Cannot fit extractor on dataset with only {X.shape[0]} samples (need at least 2)")
                
            self.extractor.fit(X, y)
            self.extraction_failed = False
        except Exception as e:
            # Get logger instance safely
            current_logger = logging.getLogger(__name__)
            error_msg = str(e)
            
            # Handle common extractor failures
            current_logger.warning(f"Primary extractor {type(self.extractor).__name__} failed: {error_msg}")
            current_logger.debug(f"Input shape: {X.shape}")
            
            # Special handling for common failure modes
            if "zero-size array" in error_msg or "empty array" in error_msg:
                current_logger.warning("Extractor encountered empty array - likely due to very small CV fold")
            elif "n_components" in error_msg and "greater than" in error_msg:
                current_logger.warning("Too many components requested for dataset size")
            elif "Matrix is not positive definite" in error_msg:
                current_logger.warning("KPCA encountered singular matrix - likely due to poor kernel parameter choice")
            elif "zero-size array to reduction operation maximum" in error_msg:
                current_logger.warning("KPCA encountered zero-size eigenvalue array - likely due to insufficient data variation")
            
            # Create appropriate fallback based on extractor type and data constraints
            n_samples, n_features = X.shape
            
            if hasattr(self.extractor, 'n_components'):
                n_components = getattr(self.extractor, 'n_components', 2)
                
                # Calculate safe number of components based on data size
                # For PCA: components <= min(n_samples-1, n_features)
                max_safe_components = min(n_samples - 1, n_features, 10)  # Cap at 10 for computational efficiency
                
                # For KPCA failures, fall back to regular PCA
                if 'KernelPCA' in type(self.extractor).__name__ or 'KPCA' in type(self.extractor).__name__:
                    from sklearn.decomposition import PCA
                    # Use very conservative components for KPCA failures
                    safe_components = min(n_components // 2, max_safe_components, 3)
                    safe_components = max(1, safe_components)  # Ensure at least 1 component
                    self.fallback_extractor = PCA(n_components=safe_components, random_state=42)
                    current_logger.info(f"Using conservative PCA fallback with {safe_components} components for KPCA failure")
                
                # For other component-based extractors, try PCA with safe components
                else:
                    from sklearn.decomposition import PCA
                    safe_components = min(n_components, max_safe_components, 5)
                    safe_components = max(1, safe_components)
                    self.fallback_extractor = PCA(n_components=safe_components, random_state=42)
                    current_logger.info(f"Using PCA fallback with {safe_components} components")
            else:
                # For non-component extractors, use simple PCA
                from sklearn.decomposition import PCA
                max_safe_components = min(n_samples - 1, n_features, 5)
                safe_components = max(1, max_safe_components)
                self.fallback_extractor = PCA(n_components=safe_components, random_state=42)
                current_logger.info(f"Using basic PCA fallback with {safe_components} components")
            
            try:
                # Validate fallback extractor can work with the data
                if X.shape[0] >= 2 and X.shape[1] >= 1:
                    self.fallback_extractor.fit(X, y)
                    self.extraction_failed = True
                    current_logger.info("Fallback extractor fitted successfully")
                else:
                    # Data too small even for fallback
                    current_logger.error(f"Dataset too small for any extractor: {X.shape}")
                    self.extraction_failed = True
                    self.fallback_extractor = None
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
    parser = argparse.ArgumentParser(description="Multi-Omics Hyperparameter Tuner - Supports both Extractor and Selector approaches")
    parser.add_argument("--dataset", help="Dataset name")
    parser.add_argument("--task", choices=["reg", "clf"], help="Task type (reg/clf)")
    parser.add_argument("--approach", choices=["extractors", "selectors"], 
                       help="Tuning approach: 'extractors' (tune extractors+models) or 'selectors' (tune models with fixed feature selection)")
    
    # Extractor-based arguments
    parser.add_argument("--extractor", help="Extractor name (required for --approach extractors)")
    
    # Selector-based arguments
    parser.add_argument("--selector", help="Selector name (required for --approach selectors)")  
    parser.add_argument("--n-features", type=int, choices=FEATURE_COUNTS, default=16, 
                       help="Number of features per modality (for --approach selectors)")
    
    # Common arguments
    parser.add_argument("--model", help="Model name")
    parser.add_argument("--fusion", choices=["average", "attention_weighted"], default="average", help="Fusion method")
    parser.add_argument("--single", action="store_true", help="Single tuning mode (used by subprocess)")
    parser.add_argument("--no-subprocess", action="store_true", help="Disable subprocess isolation")
    parser.add_argument("--all", action="store_true", help="Run all combinations for dataset")
    parser.add_argument("--list-datasets", action="store_true", help="List all available datasets")
    parser.add_argument("--no-cache", action="store_true", help="Disable preprocessing cache (slower but ensures fresh preprocessing)")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Convert log level string to logging constant
    log_level = getattr(logging, args.log_level.upper())
    
    # Handle cache option
    if args.no_cache:
        ENABLE_PREPROCESSING_CACHE = False
        print("Preprocessing cache disabled - will run full preprocessing for each combination")
    
    if args.list_datasets:
        list_available_datasets()
        sys.exit(0)
    
    if not args.approach and not args.list_datasets:
        parser.error("--approach is required (unless using --list-datasets)")
    
    if not args.dataset and not args.list_datasets:
        parser.error("--dataset is required (unless using --list-datasets)")
    
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
    
    # Validate approach-specific arguments
    if args.approach == "extractors":
        # Only require --extractor for non-batch mode
        if not args.all and not args.extractor:
            parser.error("--approach extractors requires --extractor (unless using --all)")
        
        # Validate extractor for task (only if extractor is specified)
        if args.extractor:
            if args.task == "reg":
                if args.extractor not in REGRESSION_EXTRACTORS:
                    parser.error(f"Extractor '{args.extractor}' not valid for regression. Choose from: {REGRESSION_EXTRACTORS}")
            else:  # classification
                if args.extractor not in CLASSIFICATION_EXTRACTORS:
                    parser.error(f"Extractor '{args.extractor}' not valid for classification. Choose from: {CLASSIFICATION_EXTRACTORS}")
    
    elif args.approach == "selectors":
        # Only require --selector for non-batch mode
        if not args.all and not args.selector:
            parser.error("--approach selectors requires --selector (unless using --all)")
        
        # Validate selector for task (only if selector is specified)
        if args.selector:
            if args.task == "reg":
                if args.selector not in REGRESSION_SELECTORS:
                    parser.error(f"Selector '{args.selector}' not valid for regression. Choose from: {REGRESSION_SELECTORS}")
            else:  # classification
                if args.selector not in CLASSIFICATION_SELECTORS:
                    parser.error(f"Selector '{args.selector}' not valid for classification. Choose from: {CLASSIFICATION_SELECTORS}")
        
        if args.extractor:
            print("Warning: --extractor ignored for --approach selectors")
            args.extractor = None
    
    # Single tuning mode (called by subprocess)
    if args.single:
        if not args.model:
            parser.error("--single mode requires --model")
        
        # Setup specific logger for single run
        if args.approach == "extractors":
            logger, log_path = setup_logging(args.dataset, args.extractor, args.model, log_level)
            logger.info(f"Starting single extractor tuning run: {args.dataset} - {args.extractor} - {args.model} - {args.fusion}")
            success = tune_extractors(args.dataset, args.task, args.extractor, args.model, args.fusion, logger)
        else:  # selectors
            logger, log_path = setup_logging(args.dataset, args.selector, args.model, log_level)
            logger.info(f"Starting single selector tuning run: {args.dataset} - {args.selector} - {args.model} - {args.fusion} - {args.n_features}f")
            success = tune_selectors(args.dataset, args.task, args.selector, args.model, args.fusion, args.n_features, logger)
        
        logger.info(f"Log file: {log_path}")
        sys.exit(0 if success else 1)
    
    # Setup session logger for batch operations
    session_logger, session_log_path = setup_logging(log_level=log_level)
    session_logger.info(f"Tuner session started with log level: {args.log_level}")
    session_logger.info(f"Session log file: {session_log_path}")
    
    # Batch mode: run all combinations
    if args.all:
        if args.approach == "extractors":
            session_logger.info(f"Running ALL extractor combinations for {args.dataset}...")
            tune_all_extractors(args.dataset, args.task, not args.no_subprocess)
        else:  # selectors
            session_logger.info(f"Running ALL selector combinations for {args.dataset} with Fixed Feature Selection...")
            tune_all_selectors(args.dataset, args.task, not args.no_subprocess)
        sys.exit(0)
    
    # Single combination mode
    if not args.model:
        parser.error("Single mode requires --model")
    
    # Additional validation for single mode
    if args.approach == "extractors" and not args.extractor:
        parser.error("Single extractor mode requires --extractor")
    if args.approach == "selectors" and not args.selector:
        parser.error("Single selector mode requires --selector")
    
    if args.approach == "extractors":
        if args.no_subprocess:
            success = tune_extractors(args.dataset, args.task, args.extractor, args.model, args.fusion, session_logger)
        else:
            success = run_tuning_subprocess_extractors(args.dataset, args.task, args.extractor, args.model, args.fusion, session_logger)
    else:  # selectors
        if args.no_subprocess:
            success = tune_selectors(args.dataset, args.task, args.selector, args.model, args.fusion, args.n_features, session_logger)
        else:
            success = run_tuning_subprocess_selectors(args.dataset, args.task, args.selector, args.model, args.fusion, args.n_features, session_logger)
    
    sys.exit(0 if success else 1)