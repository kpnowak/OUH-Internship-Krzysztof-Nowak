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
"""

import json, pathlib, argparse, numpy as np, joblib, time, sys, subprocess
from itertools import product
from sklearn.experimental import enable_halving_search_cv  # Enable experimental feature
from sklearn.model_selection import HalvingRandomSearchCV, KFold, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, r2_score, matthews_corrcoef
from data_io import load_dataset_for_tuner
from samplers import safe_sampler
from models import build_extractor, build_model
from sklearn.preprocessing import PowerTransformer

HP_DIR = pathlib.Path("hp_best"); HP_DIR.mkdir(exist_ok=True)
SEED   = 42
N_ITER = 32          # candidates; Halving keeps ~⅓ each rung
CV_INNER = 3
TIMEOUT_MINUTES = 30  # Timeout per combination

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
    Enhanced parameter space for models trained on 4-phase preprocessed data.
    
    Since data is now fully preprocessed with:
    - Phase 1: Early Data Quality Assessment
    - Phase 2: Fusion-Aware Preprocessing  
    - Phase 3: Centralized Missing Data Management
    - Phase 4: Coordinated Validation Framework
    
    We can use more aggressive hyperparameter ranges, but need to be adaptive
    to the actual data dimensions after preprocessing.
    
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
    
    # Adaptive component selection based on actual data dimensions
    if X_shape is not None:
        n_samples, n_features = X_shape
        # Be very conservative for CV - each fold might have as few as n_samples // CV_INNER
        # And HalvingRandomSearchCV starts with even fewer samples
        min_cv_samples = max(1, n_samples // (CV_INNER * 3))  # Very conservative estimate
        max_components = min(min_cv_samples - 1, n_features, 16)  # Cap at 16 for safety
        
        # Ultra-conservative component ranges for small CV splits
        if max_components >= 8:
            component_options = [2, 4, 8]
        elif max_components >= 4:
            component_options = [2, 4]
        elif max_components >= 2:
            component_options = [2]
        else:
            component_options = [1]  # Fallback for very small datasets
    else:
        # Default conservative ranges if no shape provided
        component_options = [2, 4]
    
    # Extractor parameters - adaptive for preprocessed data
    if extr in {"PCA","KPCA","KPLS","PLS","SparsePLS","PLS-DA","Sparse PLS-DA"}:
        p["extractor__n_components"] = component_options
    
    if extr in {"KPCA","KPLS"}:
        # Wider gamma range for kernel methods on preprocessed data
        p["extractor__gamma"] = np.logspace(-3, 0, 4)  # Reduced range and count for stability
    
    if extr in {"SparsePLS","PLS-DA","Sparse PLS-DA"}:
        # More alpha values for sparse methods
        p["extractor__alpha"] = np.logspace(-2, 0, 4)  # Reduced range and count for stability
    
    # Enhanced parameters for FA
    if extr == "FA":
        p["extractor__n_components"] = component_options
        p["extractor__max_iter"] = [1000, 3000]  # Reduced options for speed
        p["extractor__tol"] = [1e-3, 1e-2]  # More tolerant for small data
    
    # Enhanced parameters for LDA
    if extr == "LDA":
        p["extractor__solver"] = ["svd", "lsqr"]  # Reduced solvers for stability
        p["extractor__shrinkage"] = [None, "auto", 0.1]  # Reduced options
    
    # Model parameters - enhanced for preprocessed data
    if mdl.startswith("RandomForest"):
        # Adaptive parameters based on data size
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
                "model__n_estimators": [100, 200],  # Fewer estimators for speed
                "model__max_depth": [None, 5, 10],  # Reasonable depths
                "model__min_samples_leaf": [1, 2],  # Less options
                "model__min_samples_split": [2, 5],  # Less options
                "model__max_features": ["sqrt", "log2"]  # Simple feature sampling
            })
    
    if mdl in {"ElasticNet"}:
        # ElasticNet is now wrapped in TransformedTargetRegressor, so use regressor__ prefix
        p.update({
            "model__regressor__alpha": np.logspace(-3, 1, 6),  # Reduced range for ElasticNet
            "model__regressor__l1_ratio": np.linspace(0.1, 0.9, 3),  # Fewer options
            "model__regressor__max_iter": [1000, 2000],  # Sufficient iterations
            # Optional: Add transformer options to let tuner decide
            "model__transformer": [None, PowerTransformer(method="yeo-johnson", standardize=True)]
        })
    
    if mdl == "SVC":
        p.update({
            "model__C": np.logspace(-1, 1, 4),  # Smaller C range
            "model__gamma": np.logspace(-3, 0, 4),  # Smaller gamma range
            "model__kernel": ["rbf", "linear"]  # Stable kernels for small data
        })
    
    # Enhanced parameters for LinearRegression
    if mdl == "LinearRegression":
        # LinearRegression is now wrapped in TransformedTargetRegressor, so use regressor__ prefix
        p.update({
            "model__regressor__fit_intercept": [True, False],
            # Optional: Add transformer options to let tuner decide
            "model__transformer": [None, PowerTransformer(method="yeo-johnson", standardize=True)]
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
def run_tuning_subprocess(dataset, task, extractor, model):
    """Run tuning for a single combination in subprocess with timeout."""
    cmd = [
        sys.executable, __file__,
        "--dataset", dataset,
        "--task", task,
        "--extractor", extractor,
        "--model", model,
        "--single"  # Flag to indicate single combination mode
    ]
    
    print(f"\n{'='*60}")
    print(f"Running: {extractor} + {model} (4-Phase Preprocessed)")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_MINUTES*60)
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"SUCCESS ({elapsed:.1f}s): {extractor} + {model}")
            if result.stdout.strip():
                print("Output:", result.stdout.strip())
        else:
            print(f"FAILED ({elapsed:.1f}s): {extractor} + {model}")
            if result.stderr.strip():
                print("Error:", result.stderr.strip())
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT ({TIMEOUT_MINUTES}min): {extractor} + {model}")
        return False
    except Exception as e:
        print(f"EXCEPTION: {extractor} + {model} - {str(e)}")
        return False

# ------------- main tune routine with 4-phase preprocessing -----------------------------
def tune(dataset, task, extractor, model):
    """
    Enhanced tuning function with 4-phase preprocessing integration.
    
    This ensures hyperparameters are optimized on the SAME preprocessed data
    that the main pipeline uses, providing consistent and meaningful optimization.
    """
    try:
        print(f"Loading {dataset} with 4-Phase Enhanced Pipeline...")
        print("=" * 60)
        print("Pipeline Phases:")
        print("  Phase 1: Early Data Quality Assessment")
        print("  Phase 2: Fusion-Aware Preprocessing")
        print("  Phase 3: Centralized Missing Data Management")
        print("  Phase 4: Coordinated Validation Framework")
        print("=" * 60)
        
        # Load data with FULL 4-phase preprocessing (same as main pipeline)
        X, y = load_dataset_for_tuner(dataset, task=task)
        
        print(f"Data loaded successfully:")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Features after 4-phase preprocessing: {X.shape[1]}")
        
        # Validate data quality after preprocessing
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            print("Warning: Found NaN/Inf in preprocessed data, cleaning...")
            X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            print("Warning: Found NaN/Inf in targets, cleaning...")
            y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Get sampler for classification if needed
        sampler = None
        if task == "clf":
            try:
                # Check class distribution first
                unique_classes, class_counts = np.unique(y, return_counts=True)
                min_class_size = class_counts.min()
                
                print(f"  Class distribution: {dict(zip(unique_classes, class_counts))}")
                print(f"  Minimum class size: {min_class_size}")
                
                # Only use sampler if we have reasonable class sizes for cross-validation
                # Need at least CV_INNER samples per class for robust CV
                if min_class_size >= CV_INNER * 2:  # Conservative threshold
                    sampler = safe_sampler(y)
                    if sampler is not None:
                        print(f"  Using sampler: {type(sampler).__name__}")
                    else:
                        print("  No sampler needed/available")
                else:
                    print(f"  Skipping sampler due to small class size ({min_class_size} < {CV_INNER * 2})")
                    sampler = None
            except Exception as e:
                print(f"  Warning: Could not create sampler: {e}")
                sampler = None

        # Build pipeline steps
        steps = []
        if sampler: 
            steps.append(("sampler", sampler))
        steps.extend([
            ("extractor", build_extractor(extractor)),
            ("model", build_model(model, task))
        ])
        
        # Use imblearn Pipeline if we have a sampler, otherwise use sklearn Pipeline
        if sampler:
            from imblearn.pipeline import Pipeline as ImbPipeline
            pipe = ImbPipeline(steps)
        else:
            pipe = Pipeline(steps)

        # Choose scorer
        scorer = make_scorer(r2_score) if task=="reg" \
                 else make_scorer(matthews_corrcoef)

        # Cross-validation strategy
        cv_inner = KFold(CV_INNER, shuffle=True, random_state=SEED) \
                   if task=="reg" else StratifiedKFold(CV_INNER, shuffle=True, random_state=SEED)

        # Get enhanced parameter space for preprocessed data
        params = param_space(extractor, model, X.shape)
        n_combinations = count_parameter_combinations(params)
        
        print(f"Enhanced parameter combinations: {n_combinations}")
        
        # Use GridSearchCV for small parameter spaces, HalvingRandomSearchCV for larger ones
        if n_combinations <= 20:  # Increased threshold for enhanced params
            print("Using GridSearchCV (exhaustive search)")
            search = GridSearchCV(
                estimator=pipe,
                param_grid=params,
                scoring=scorer,
                cv=cv_inner,
                refit=True,
                n_jobs=2,  # Reduced for Windows compatibility
                verbose=1
            )
        else:
            print("Using HalvingRandomSearchCV")
            search = HalvingRandomSearchCV(
                estimator = pipe,
                param_distributions = params,
                n_candidates="exhaust",
                factor = 3,                       # 16 → 6 → 2
                resource = "n_samples",
                max_resources = "auto",
                random_state = SEED,
                scoring = scorer,
                cv = cv_inner,
                refit = True,
                n_jobs = 2,  # Reduced for Windows compatibility
                verbose = 1
            )

        print(f"Starting hyperparameter search for {extractor} + {model}...")
        
        # Use threading backend for better Windows compatibility
        with joblib.parallel_backend("threading"):
            search.fit(X, y)

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
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        fp = HP_DIR/f"{dataset}_{extractor}_{model}.json"
        json.dump(best, open(fp, "w"), indent=2)
        print(f"SAVED {fp}")
        print(f"  Best Score: {best['best_score']:.4f}")
        print(f"  Data Shape: {X.shape}")
        print(f"  Preprocessing: 4-Phase Enhanced Pipeline")
        return True
        
    except Exception as e:
        print(f"Error in tune(): {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def tune_all_combinations(dataset, task, use_subprocess=True):
    """Run tuning for all extractor-model combinations for a dataset with 4-phase preprocessing."""
    if task == "reg":
        extractors = REGRESSION_EXTRACTORS
        models = REGRESSION_MODELS
    else:
        extractors = CLASSIFICATION_EXTRACTORS
        models = CLASSIFICATION_MODELS
    
    total_combinations = len(extractors) * len(models)
    print(f"{'='*80}")
    print(f"4-PHASE ENHANCED HYPERPARAMETER TUNING")
    print(f"{'='*80}")
    print(f"Dataset: {dataset} ({task})")
    print(f"Preprocessing: 4-Phase Enhanced Pipeline")
    print(f"  Phase 1: Early Data Quality Assessment")
    print(f"  Phase 2: Fusion-Aware Preprocessing")
    print(f"  Phase 3: Centralized Missing Data Management")
    print(f"  Phase 4: Coordinated Validation Framework")
    print(f"Extractors: {extractors}")
    print(f"Models: {models}")
    print(f"Total combinations: {total_combinations}")
    print(f"Subprocess isolation: {'ON' if use_subprocess else 'OFF'}")
    print(f"Timeout per combination: {TIMEOUT_MINUTES} minutes")
    print(f"{'='*80}")
    
    successful = 0
    failed = 0
    start_time = time.time()
    
    for i, (extractor, model) in enumerate(product(extractors, models), 1):
        print(f"\n{'='*80}")
        print(f"Progress: {i}/{total_combinations} - {dataset} - {extractor} + {model}")
        print(f"4-Phase Preprocessing: ENABLED")
        print(f"{'='*80}")
        
        if use_subprocess:
            success = run_tuning_subprocess(dataset, task, extractor, model)
        else:
            # Direct execution (for debugging)
            try:
                success = tune(dataset, task, extractor, model)
            except Exception as e:
                success = False
                print(f"FAILED: {extractor} + {model} - {str(e)}")
        
        if success:
            successful += 1
            print(f" COMPLETED: {extractor} + {model}")
        else:
            failed += 1
            print(f" FAILED: {extractor} + {model}")
    
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY for {dataset} (4-Phase Enhanced)")
    print(f"{'='*80}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Successful: {successful}/{total_combinations}")
    print(f"Failed: {failed}/{total_combinations}")
    print(f"Success rate: {successful/total_combinations:.1%}")
    
    if successful > 0:
        # List generated files
        dataset_files = list(HP_DIR.glob(f"{dataset}_*.json"))
        if dataset_files:
            print(f"\nGenerated hyperparameter files:")
            for f in sorted(dataset_files):
                print(f"  - {f.name}")
                
        # Show sample of best hyperparameters
        print(f"\nSample of optimized hyperparameters:")
        for f in sorted(dataset_files)[:3]:  # Show first 3
            try:
                with open(f, 'r') as file:
                    data = json.load(file)
                    print(f"  {f.name}: Score={data['best_score']:.4f}")
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

# ------------- CLI -------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced 4-Phase Halving Tuner")
    parser.add_argument("--dataset", help="Dataset name")
    parser.add_argument("--task", choices=["reg", "clf"], help="Task type (reg/clf)")
    parser.add_argument("--extractor", help="Feature extractor")
    parser.add_argument("--model", help="Model name")
    parser.add_argument("--single", action="store_true", help="Single tuning mode (used by subprocess)")
    parser.add_argument("--no-subprocess", action="store_true", help="Disable subprocess isolation")
    parser.add_argument("--all", action="store_true", help="Run all extractor-model combinations for dataset")
    parser.add_argument("--list-datasets", action="store_true", help="List all available datasets")
    
    args = parser.parse_args()
    
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
        
        success = tune(args.dataset, args.task, args.extractor, args.model)
        sys.exit(0 if success else 1)
    
    # Batch mode: run all combinations
    if args.all:
        print(f"\n Running ALL combinations for {args.dataset} with 4-Phase Enhanced Pipeline...")
        tune_all_combinations(args.dataset, args.task, not args.no_subprocess)
        sys.exit(0)
    
    # Single combination mode
    if not args.extractor or not args.model:
        parser.error("Single mode requires --extractor and --model")
    
    if args.no_subprocess:
        success = tune(args.dataset, args.task, args.extractor, args.model)
    else:
        success = run_tuning_subprocess(args.dataset, args.task, args.extractor, args.model)
    
    sys.exit(0 if success else 1)
