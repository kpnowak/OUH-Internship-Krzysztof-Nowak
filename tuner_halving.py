"""
Enhanced Halving tuner: finds best hyper-params for any dataset with subprocess isolation
and saves them to hp_best/<dataset>_<extractor>_<model>.json

Features:
- Subprocess isolation for each combination (prevents memory leaks and crashes)
- Timeout protection (30min per combination)
- Auto-detection of dataset type (regression/classification)
- Support for all cancer datasets
- Robust error handling and monitoring
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

HP_DIR = pathlib.Path("hp_best"); HP_DIR.mkdir(exist_ok=True)
SEED   = 42
N_ITER = 32          # candidates; Halving keeps ~⅓ each rung
CV_INNER = 3
TIMEOUT_MINUTES = 30  # Timeout per combination

# Available datasets and their tasks (based on config.py configurations)
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

# Available extractors and models by task
REGRESSION_EXTRACTORS = ["PCA", "KPCA", "KPLS", "FA", "PLS", "SparsePLS"]  # 6 extractors (removed LDA)
REGRESSION_MODELS = ["LinearRegression", "ElasticNet", "RandomForestRegressor"]
CLASSIFICATION_EXTRACTORS = ["PCA", "KPCA", "FA", "LDA", "PLS-DA", "SparsePLS"]  # 6 extractors (LDA only for classification)
CLASSIFICATION_MODELS = ["LogisticRegression", "SVC", "RandomForestClassifier"]

# ------------- Dataset type detection ----------------------
def detect_dataset_task(dataset):
    """Auto-detect if dataset is regression or classification."""
    if dataset in DATASET_INFO:
        return DATASET_INFO[dataset]
    
    # Try to load and inspect the data
    try:
        from data_io import load_dataset_for_tuner
        # Try regression first
        try:
            X, y = load_dataset_for_tuner(dataset, task="reg")
            if len(np.unique(y)) > 10:  # Likely continuous
                return "reg"
        except:
            pass
        
        # Try classification
        try:
            X, y = load_dataset_for_tuner(dataset, task="clf")
            if len(np.unique(y)) <= 10:  # Likely categorical
                return "clf"
        except:
            pass
            
    except Exception as e:
        print(f"Warning: Could not auto-detect task for {dataset}: {e}")
    
    # Default to regression if uncertain
    return "reg"

# ------------- search space ----------------------------------
def param_space(extr, mdl):
    p = {}
    # extractor - use very small component ranges for tiny datasets
    if extr in {"PCA","KPCA","KPLS","PLS","SparsePLS","PLS-DA","Sparse PLS-DA"}:
        p["extractor__n_components"] = [2,3]  # Very small range for tiny datasets
    if extr in {"KPCA","KPLS"}:
        p["extractor__gamma"] = np.logspace(-4,-1,6)
    if extr in {"SparsePLS","PLS-DA","Sparse PLS-DA"}:
        p["extractor__alpha"] = np.logspace(-3,0,5)
    
    # Add parameters for FA to increase search space
    if extr == "FA":
        p["extractor__n_components"] = [2,3]
        p["extractor__max_iter"] = [1000, 3000, 5000]  # Add iterations parameter
        p["extractor__tol"] = [1e-4, 1e-3, 1e-2]  # Add tolerance parameter
    
    # LDA has limited hyperparameters, add what we can
    if extr == "LDA":
        p["extractor__solver"] = ["svd", "lsqr"]  # Add solver options
        p["extractor__shrinkage"] = [None, "auto", 0.1, 0.5]  # Add shrinkage (for lsqr)
    
    # model
    if mdl.startswith("RandomForest"):
        p.update({
            "model__n_estimators": [200,400,800],
            "model__max_depth": [None,5,15],
            "model__min_samples_leaf": [1,2,4],
        })
    if mdl in {"ElasticNet"}:
        p.update({
            "model__alpha": np.logspace(-4,1,8),
            "model__l1_ratio": np.linspace(0.1,0.9,5),
        })
    if mdl == "SVC":
        p.update({
            "model__C": np.logspace(-2,2,8),
            "model__gamma": np.logspace(-4,-1,6),
        })
    
    # Add parameters for LinearRegression to increase search space
    if mdl == "LinearRegression":
        p["model__fit_intercept"] = [True, False]
        p["model__copy_X"] = [True, False]
    
    # Add parameters for LogisticRegression
    if mdl == "LogisticRegression":
        p["model__C"] = np.logspace(-2, 2, 8)
        p["model__penalty"] = ["l1", "l2", "elasticnet"]
        p["model__solver"] = ["liblinear", "saga"]
    
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
    print(f"Running: {extractor} + {model}")
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

# ------------- main tune routine -----------------------------
def tune(dataset, task, extractor, model):
    """Direct tuning function (used by subprocess)."""
    try:
        X, y = load_dataset_for_tuner(dataset, task=task)          # drops NaN targets
        sampler = safe_sampler(y) if task=="clf" else None

        steps = []
        if sampler: 
            steps.append(("sampler", sampler))
        steps.extend([("extractor", build_extractor(extractor)),
                      ("model",     build_model(model, task))])
        
        # Use imblearn Pipeline if we have a sampler, otherwise use sklearn Pipeline
        if sampler:
            from imblearn.pipeline import Pipeline as ImbPipeline
            pipe = ImbPipeline(steps)
        else:
            pipe = Pipeline(steps)

        scorer = make_scorer(r2_score) if task=="reg" \
                 else make_scorer(matthews_corrcoef)

        cv_inner = KFold(CV_INNER, shuffle=True, random_state=SEED) \
                   if task=="reg" else StratifiedKFold(CV_INNER, shuffle=True, random_state=SEED)

        # Get parameter space and decide on search strategy
        params = param_space(extractor, model)
        n_combinations = count_parameter_combinations(params)
        
        print(f"Parameter combinations: {n_combinations}")
        
        # Use GridSearchCV for small parameter spaces, HalvingRandomSearchCV for larger ones
        if n_combinations <= 10:
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

        # Use threading backend for better Windows compatibility
        with joblib.parallel_backend("threading"):
            search.fit(X, y)

        best = {"best_params": search.best_params_,
                "best_score":  search.best_score_}

        fp = HP_DIR/f"{dataset}_{extractor}_{model}.json"
        json.dump(best, open(fp, "w"), indent=2)
        print(f"SAVED {fp}  score={best['best_score']:.3f}")
        return True
        
    except Exception as e:
        print(f"Error in tune(): {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def tune_all_combinations(dataset, task, use_subprocess=True):
    """Run tuning for all extractor-model combinations for a dataset."""
    if task == "reg":
        extractors = REGRESSION_EXTRACTORS
        models = REGRESSION_MODELS
    else:
        extractors = CLASSIFICATION_EXTRACTORS
        models = CLASSIFICATION_MODELS
    
    total_combinations = len(extractors) * len(models)
    print(f"Running hyperparameter tuning for {dataset} ({task})")
    print(f"Extractors: {extractors}")
    print(f"Models: {models}")
    print(f"Total combinations: {total_combinations}")
    print(f"Subprocess isolation: {'ON' if use_subprocess else 'OFF'}")
    print(f"Timeout per combination: {TIMEOUT_MINUTES} minutes")
    
    successful = 0
    failed = 0
    start_time = time.time()
    
    for i, (extractor, model) in enumerate(product(extractors, models), 1):
        print(f"\n{'='*80}")
        print(f"Progress: {i}/{total_combinations} - {dataset} - {extractor} + {model}")
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
            print(f"COMPLETED: {extractor} + {model}")
        else:
            failed += 1
    
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY for {dataset}")
    print(f"{'='*80}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Successful: {successful}/{total_combinations}")
    print(f"Failed: {failed}/{total_combinations}")
    
    if successful > 0:
        # List generated files
        dataset_files = list(HP_DIR.glob(f"{dataset}_*.json"))
        if dataset_files:
            print(f"\nGenerated hyperparameter files:")
            for f in sorted(dataset_files):
                print(f"  - {f.name}")

def list_available_datasets():
    """List all available datasets and their auto-detected tasks."""
    print("Available datasets:")
    for dataset, task in DATASET_INFO.items():
        task_name = "regression" if task == "reg" else "classification"
        print(f"  - {dataset:<12} ({task_name})")

# ------------- CLI -------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced halving tuner for any dataset")
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
        print("Available datasets:")
        for dataset, task in DATASET_INFO.items():
            task_name = "regression" if task == "reg" else "classification"
            print(f"  - {dataset:<10} ({task_name})")
        sys.exit(0)
    
    if not args.dataset:
        parser.error("--dataset is required")
    
    if args.dataset not in DATASET_INFO:
        print(f"Error: Dataset '{args.dataset}' not found.")
        print("Available datasets:")
        for dataset, task in DATASET_INFO.items():
            task_name = "regression" if task == "reg" else "classification"
            print(f"  - {dataset:<10} ({task_name})")
        sys.exit(1)
    
    # Auto-detect task if not specified
    if not args.task:
        args.task = DATASET_INFO[args.dataset]
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
        print(f"\nRunning ALL combinations for {args.dataset} dataset...")
        print("=" * 80)
        
        total_combinations = len(REGRESSION_EXTRACTORS) * len(REGRESSION_MODELS)
        successful = 0
        failed = 0
        
        start_time = time.time()
        
        for i, (extractor, model) in enumerate(product(REGRESSION_EXTRACTORS, REGRESSION_MODELS), 1):
            print(f"\n[{i}/{total_combinations}] {extractor} + {model}")
            print("-" * 60)
            
            if args.no_subprocess:
                # Direct execution
                success = tune(args.dataset, args.task, extractor, model)
            else:
                # Subprocess execution with timeout
                success = run_tuning_subprocess(args.dataset, args.task, extractor, model)
            
            if success:
                successful += 1
            else:
                failed += 1
        
        elapsed = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"BATCH TUNING COMPLETE for {args.dataset}")
        print(f"Total time: {elapsed/60:.1f} minutes")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Success rate: {successful/total_combinations:.1%}")
        print(f"Results saved in: hp_best/")
        
        sys.exit(0)
    
    # Single combination mode
    if not args.extractor or not args.model:
        parser.error("Single mode requires --extractor and --model")
    
    if args.no_subprocess:
        success = tune(args.dataset, args.task, args.extractor, args.model)
    else:
        success = run_tuning_subprocess(args.dataset, args.task, args.extractor, args.model)
    
    sys.exit(0 if success else 1)
