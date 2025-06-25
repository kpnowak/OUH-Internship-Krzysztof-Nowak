#!/usr/bin/env python3
"""
Feature-First Pipeline Module.
Implements the new architecture where feature processing is applied to each modality separately
before fusion, rather than applying fusion first and then feature processing.

NEW ORDER: Raw Data → Feature Processing → Fusion → Model Training
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedKFold

# Local imports
from models import (
    cached_fit_transform_extractor_regression, transform_extractor_regression,
    cached_fit_transform_selector_regression, transform_selector_regression,
    cached_fit_transform_extractor_classification, transform_extractor_classification,
    cached_fit_transform_selector_classification, transform_selector_classification
)
from fusion import merge_modalities
from utils import suppress_sklearn_warnings, comprehensive_logger

logger = logging.getLogger(__name__)

class FeatureFirstPipeline:
    """
    Pipeline that applies feature processing to each modality separately first,
    then applies fusion to the processed features.
    """
    
    def __init__(self, 
                 fusion_method: str = "average",
                 task_type: str = "classification",
                 missing_percentage: float = 0.0):
        """
        Initialize feature-first pipeline.
        
        Parameters
        ----------
        fusion_method : str
            Fusion method to apply to processed features
        task_type : str
            Task type (classification or regression)
        missing_percentage : float
            Percentage of missing data for fusion strategy selection
        """
        self.fusion_method = fusion_method
        self.task_type = task_type
        self.missing_percentage = missing_percentage
        self.is_regression = (task_type == "regression")
        
        # Select appropriate fusion strategies based on missing data
        if missing_percentage == 0.0:
            # Clean data: all 6 fusion methods available (matching fusion-first pipeline)
            self.available_fusion_methods = [
                'attention_weighted', 'learnable_weighted', 'mkl', 'average', 'sum', 'early_fusion_pca'
            ]
        else:
            # Missing data: only 4 robust methods (matching fusion-first pipeline)
            self.available_fusion_methods = ['mkl', 'average', 'sum', 'early_fusion_pca']
        
        logger.info(f"FeatureFirstPipeline initialized: {fusion_method}, {task_type}, {missing_percentage:.1%} missing")
        logger.info(f"Available fusion methods: {self.available_fusion_methods}")
    
    def run_feature_first_experiment(self,
                                   data_modalities: Dict[str, pd.DataFrame],
                                   y: np.ndarray,
                                   common_ids: List[str],
                                   algorithm_name: str,
                                   algorithm_obj: Any,
                                   n_features_or_components: int,
                                   model_name: str,
                                   model_obj: Any,
                                   cv_strategy: Any,
                                   cv_groups: Any,
                                   dataset_name: str = "unknown",
                                   fold_idx: int = 0) -> Dict[str, Any]:
        """
        Run a single feature-first experiment.
        
        NEW ARCHITECTURE:
        1. Apply feature processing to each modality separately
        2. Apply fusion to processed features
        3. Train model on fused processed features
        
        Parameters
        ----------
        data_modalities : Dict[str, pd.DataFrame]
            Raw modality data
        y : np.ndarray
            Target values
        common_ids : List[str]
            Common sample IDs
        algorithm_name : str
            Name of feature processing algorithm
        algorithm_obj : Any
            Feature processing algorithm object
        n_features_or_components : int
            Number of features/components to extract/select
        model_name : str
            Name of model
        model_obj : Any
            Model object
        cv_strategy : Any
            Cross-validation strategy
        cv_groups : Any
            Cross-validation groups
        dataset_name : str
            Dataset name
        fold_idx : int
            Fold index
            
        Returns
        -------
        Dict[str, Any]
            Experiment results
        """
        
        logger.info(f"Running feature-first experiment: {algorithm_name} → {self.fusion_method} → {model_name}")
        
        try:
            # Step 1: Apply feature processing to each modality separately
            processed_modalities = self._apply_feature_processing_to_modalities(
                data_modalities, y, common_ids, algorithm_name, algorithm_obj, 
                n_features_or_components, dataset_name, fold_idx, model_name
            )
            
            if not processed_modalities:
                logger.warning("No processed modalities available")
                return {"error": "No processed modalities"}
            
            # Step 2: Apply fusion to processed features
            fused_features = self._apply_fusion_to_processed_features(
                processed_modalities, y, dataset_name, fold_idx
            )
            
            if fused_features is None or fused_features.size == 0:
                logger.warning("Fusion resulted in empty features")
                return {"error": "Empty fused features"}
            
            # Step 3: Train model on fused processed features
            results = self._train_model_on_fused_features(
                fused_features, y, model_name, model_obj, cv_strategy, cv_groups, 
                dataset_name, algorithm_name, fold_idx
            )
            
            # Add metadata
            results.update({
                'pipeline_order': 'feature_first',
                'algorithm_name': algorithm_name,
                'fusion_method': self.fusion_method,
                'model_name': model_name,
                'n_features_components': n_features_or_components,
                'processed_modalities_shapes': {name: arr.shape for name, arr in processed_modalities.items()},
                'fused_features_shape': fused_features.shape
            })
            
            return results
            
        except Exception as e:
            logger.error(f"Feature-first experiment failed: {str(e)}")
            return {"error": str(e)}
    
    def _apply_feature_processing_to_modalities(self,
                                              data_modalities: Dict[str, pd.DataFrame],
                                              y: np.ndarray,
                                              common_ids: List[str],
                                              algorithm_name: str,
                                              algorithm_obj: Any,
                                              n_features_or_components: int,
                                              dataset_name: str,
                                              fold_idx: int,
                                              model_name: str) -> Dict[str, np.ndarray]:
        """
        Apply feature extraction or selection to each modality separately.
        
        Parameters
        ----------
        data_modalities : Dict[str, pd.DataFrame]
            Raw modality data (features x samples)
        y : np.ndarray
            Target values
        common_ids : List[str]
            Common sample IDs
        algorithm_name : str
            Name of feature processing algorithm
        algorithm_obj : Any
            Feature processing algorithm object
        n_features_or_components : int
            Number of features/components
        dataset_name : str
            Dataset name
        fold_idx : int
            Fold index
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary mapping modality names to processed feature arrays
        """
        
        processed_modalities = {}
        
        for modality_name, modality_df in data_modalities.items():
            try:
                logger.info(f"Processing {modality_name} with {algorithm_name}")
                
                # Convert DataFrame to numpy array (transpose to get samples x features)
                X_modality = modality_df.T.values  # modality_df is features x samples
                
                # Align with common samples
                modality_ids = modality_df.columns.tolist()
                id_to_idx = {sid: i for i, sid in enumerate(modality_ids)}
                aligned_indices = [id_to_idx[sid] for sid in common_ids if sid in id_to_idx]
                
                if not aligned_indices:
                    logger.warning(f"No aligned samples for {modality_name}")
                    continue
                
                X_aligned = X_modality[aligned_indices]
                y_aligned = y[:len(aligned_indices)]
                
                logger.info(f"Aligned {modality_name}: {X_aligned.shape}")
                
                # Apply feature processing
                if self.is_regression:
                    if "selection" in algorithm_name.lower() or algorithm_name in ['ElasticNetFS', 'RFImportance', 'VarianceFTest', 'LASSO', 'f_regressionFS']:
                        # Selection algorithm
                        selector, X_processed = cached_fit_transform_selector_regression(
                            X_aligned, y_aligned, algorithm_obj, n_features_or_components,
                            ds_name=dataset_name, modality_name=modality_name, fold_idx=fold_idx
                        )
                    else:
                        # Extraction algorithm
                        extractor, X_processed = cached_fit_transform_extractor_regression(
                            X_aligned, y_aligned, algorithm_obj, n_features_or_components,
                            ds_name=dataset_name, modality_name=modality_name, fold_idx=fold_idx,
                            model_name=model_name
                        )
                else:
                    if "selection" in algorithm_name.lower() or algorithm_name in ['ElasticNetFS', 'RFImportance', 'VarianceFTest', 'LASSO', 'LogisticL1']:
                        # Selection algorithm
                        selector, X_processed = cached_fit_transform_selector_classification(
                            X_aligned, y_aligned, algorithm_obj, n_features_or_components,
                            ds_name=dataset_name, modality_name=modality_name, fold_idx=fold_idx
                        )
                    else:
                        # Extraction algorithm
                        extractor, X_processed = cached_fit_transform_extractor_classification(
                            X_aligned, y_aligned, algorithm_obj, n_features_or_components,
                            ds_name=dataset_name, modality_name=modality_name, fold_idx=fold_idx,
                            model_name=model_name
                        )
                
                if X_processed is not None and X_processed.size > 0:
                    processed_modalities[modality_name] = X_processed
                    logger.info(f"Processed {modality_name}: {X_aligned.shape} → {X_processed.shape}")
                else:
                    logger.warning(f"Feature processing failed for {modality_name}")
                
            except Exception as e:
                logger.error(f"Error processing {modality_name} with {algorithm_name}: {str(e)}")
                continue
        
        return processed_modalities
    
    def _apply_fusion_to_processed_features(self,
                                          processed_modalities: Dict[str, np.ndarray],
                                          y: np.ndarray,
                                          dataset_name: str,
                                          fold_idx: int) -> Optional[np.ndarray]:
        """
        Apply fusion method to processed features from each modality.
        
        Parameters
        ----------
        processed_modalities : Dict[str, np.ndarray]
            Processed features from each modality
        y : np.ndarray
            Target values
        dataset_name : str
            Dataset name
        fold_idx : int
            Fold index
            
        Returns
        -------
        Optional[np.ndarray]
            Fused feature matrix
        """
        
        if len(processed_modalities) == 0:
            logger.warning("No processed modalities to fuse")
            return None
        
        if len(processed_modalities) == 1:
            # Single modality - no fusion needed
            modality_name = list(processed_modalities.keys())[0]
            logger.info(f"Single modality {modality_name}, no fusion needed")
            return list(processed_modalities.values())[0]
        
        try:
            logger.info(f"Applying {self.fusion_method} fusion to {len(processed_modalities)} processed modalities")
            
            # Convert to list of arrays for fusion
            modality_arrays = list(processed_modalities.values())
            
            # Log shapes before fusion
            for i, (name, arr) in enumerate(processed_modalities.items()):
                logger.debug(f"  {name}: {arr.shape}")
            
            # Apply fusion strategy
            if self.fusion_method in self.available_fusion_methods:
                # Create imputer for missing data handling
                from fusion import ModalityImputer
                imputer = ModalityImputer()
                
                # Calculate appropriate n_components for early_fusion_pca
                n_components_to_use = None
                if self.fusion_method == "early_fusion_pca":
                    # For early fusion PCA, reduce to 1/3 of input features
                    total_features = sum(arr.shape[1] for arr in modality_arrays)
                    n_components_to_use = max(1, total_features // 3)  # At least 1 component
                    logger.info(f"EarlyFusionPCA: Reducing {total_features} features to {n_components_to_use} components (1/3 reduction)")
                
                # Apply fusion
                fused_result = merge_modalities(
                    *modality_arrays,
                    imputer=imputer,
                    is_train=True,
                    strategy=self.fusion_method,
                    n_components=n_components_to_use,
                    y=y,
                    is_regression=self.is_regression
                )
                
                # Handle tuple return values
                if isinstance(fused_result, tuple):
                    fused_features, fitted_fusion = fused_result
                else:
                    fused_features = fused_result
                
                logger.info(f"Fusion completed: {fused_features.shape}")
                return fused_features
                
            else:
                logger.warning(f"Fusion method {self.fusion_method} not available for {self.missing_percentage:.1%} missing data")
                # Fallback to concatenation
                fused_features = np.column_stack(modality_arrays)
                logger.info(f"Fallback concatenation: {fused_features.shape}")
                return fused_features
                
        except Exception as e:
            logger.error(f"Fusion failed: {str(e)}, using concatenation fallback")
            # Ultimate fallback to simple concatenation
            try:
                modality_arrays = list(processed_modalities.values())
                fused_features = np.column_stack(modality_arrays)
                logger.info(f"Concatenation fallback: {fused_features.shape}")
                return fused_features
            except Exception as e2:
                logger.error(f"Concatenation fallback also failed: {str(e2)}")
                return None
    
    def _train_model_on_fused_features(self,
                                     fused_features: np.ndarray,
                                     y: np.ndarray,
                                     model_name: str,
                                     model_obj: Any,
                                     cv_strategy: Any,
                                     cv_groups: Any,
                                     dataset_name: str,
                                     algorithm_name: str,
                                     fold_idx: int) -> Dict[str, Any]:
        """
        Train model on fused processed features.
        
        Parameters
        ----------
        fused_features : np.ndarray
            Fused feature matrix
        y : np.ndarray
            Target values
        model_name : str
            Model name
        model_obj : Any
            Model object
        cv_strategy : Any
            Cross-validation strategy
        cv_groups : Any
            Cross-validation groups
        dataset_name : str
            Dataset name
        algorithm_name : str
            Feature processing algorithm name
        fold_idx : int
            Fold index
            
        Returns
        -------
        Dict[str, Any]
            Training results
        """
        
        try:
            logger.info(f"Training {model_name} on fused features: {fused_features.shape}")
            logger.debug(f"Target array shape: {y.shape}")
            
            # CRITICAL FIX: Use fused_features dimensions, don't truncate based on y
            # The fused_features should already be aligned with y from previous steps
            X_final = fused_features
            y_final = y
            
            # Validate dimensions match
            if X_final.shape[0] != len(y_final):
                logger.error(f"DIMENSION MISMATCH: X_final={X_final.shape[0]} samples, y_final={len(y_final)} samples")
                # Emergency alignment - use the minimum but log the issue
                min_samples = min(X_final.shape[0], len(y_final))
                logger.warning(f"Emergency alignment: using {min_samples} samples")
                X_final = X_final[:min_samples]
                y_final = y_final[:min_samples]
            
            # Basic validation
            if X_final.size == 0 or y_final.size == 0:
                logger.error(f"EMPTY DATA: X_final.size={X_final.size}, y_final.size={y_final.size}")
                return {"error": f"Empty data for model training: X_final.size={X_final.size}, y_final.size={y_final.size}"}
            
            if np.any(np.isnan(X_final)) or np.any(np.isinf(X_final)):
                logger.warning("NaN/inf values detected in fused features")
                # Replace NaN/inf with zeros
                X_final = np.nan_to_num(X_final, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Perform cross-validation
            from sklearn.model_selection import cross_validate
            from sklearn.metrics import make_scorer
            
            # Define scoring metrics
            if self.is_regression:
                from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
                scoring = {
                    'mse': make_scorer(mean_squared_error, greater_is_better=False),
                    'r2': make_scorer(r2_score),
                    'mae': make_scorer(mean_absolute_error, greater_is_better=False)
                }
            else:
                from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef
                
                # Safe MCC scorer that handles multiclass scenarios robustly
                def safe_mcc_score(y_true, y_pred):
                    """
                    Safe MCC calculation that handles edge cases for imbalanced datasets.
                    
                    MCC is particularly important for genomic/biomedical data because:
                    - Handles severe class imbalance better than accuracy or F1
                    - Takes into account all confusion matrix quadrants (TP, TN, FP, FN)
                    - Ranges from -1 (worst) to +1 (perfect), with 0 = random
                    - More informative than accuracy for imbalanced datasets
                    """
                    import numpy as np
                    try:
                        # Calculate MCC using sklearn's implementation
                        mcc = matthews_corrcoef(y_true, y_pred)
                        
                        # Handle NaN cases (can occur with extreme class imbalance)
                        if np.isnan(mcc):
                            logger.debug("MCC calculation returned NaN, likely due to extreme class imbalance")
                            return 0.0  # No correlation (random performance)
                        
                        return mcc
                        
                    except Exception as e:
                        # Fallback for any MCC calculation issues
                        logger.debug(f"MCC calculation failed: {e}, returning 0.0")
                        return 0.0  # No correlation as fallback
                
                # Safe AUC scorer that handles class imbalance and missing classes in CV folds
                def safe_auc_score(y_true, y_pred_proba):
                    import numpy as np
                    try:
                        # Get unique classes in y_true for this fold
                        unique_classes = np.unique(y_true)
                        
                        # If only one class present, return 0.5 (no discriminative power)
                        if len(unique_classes) < 2:
                            return 0.5
                        
                        # Handle binary vs multiclass cases
                        if len(unique_classes) == 2:
                            # Binary classification - use column 1 for positive class
                            if y_pred_proba.shape[1] == 2:
                                return roc_auc_score(y_true, y_pred_proba[:, 1])
                            else:
                                return roc_auc_score(y_true, y_pred_proba)
                        else:
                            # Multiclass case - use robust approach that handles missing classes
                            try:
                                # Try standard multiclass AUC first
                                return roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
                            except ValueError as ve:
                                if "Number of classes in y_true not equal to the number of columns in 'y_score'" in str(ve):
                                    # This error occurs when validation set has classes not seen during training
                                    # Fall back to a more basic AUC calculation approach
                                    logger.debug(f"AUC: Handling missing classes issue")
                                    
                                    # Try with average='macro' which is more forgiving
                                    try:
                                        return roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
                                    except:
                                        # Ultimate fallback: binary-style AUC for dominant class
                                        unique_classes = np.unique(y_true)
                                        if len(unique_classes) == 2:
                                            # Binary case: use positive class probability
                                            pos_class_idx = 1 if y_pred_proba.shape[1] > 1 else 0
                                            return roc_auc_score(y_true, y_pred_proba[:, pos_class_idx])
                                        else:
                                            # Multiclass: return weighted average of binary AUCs
                                            aucs = []
                                            for i, class_label in enumerate(unique_classes):
                                                if i < y_pred_proba.shape[1]:
                                                    y_binary = (y_true == class_label).astype(int)
                                                    if len(np.unique(y_binary)) > 1:  # Only if both classes present
                                                        auc_i = roc_auc_score(y_binary, y_pred_proba[:, i])
                                                        aucs.append(auc_i)
                                            return np.mean(aucs) if aucs else 0.5
                                else:
                                    # Re-raise other ValueError types
                                    raise ve
                    except Exception as e:
                        # Fallback for any AUC calculation issues
                        logger.debug(f"AUC calculation failed: {e}, returning 0.5")
                        logger.debug(f"  y_true shape: {y_true.shape}, unique classes: {np.unique(y_true)}")
                        logger.debug(f"  y_pred_proba shape: {y_pred_proba.shape}")
                        return 0.5
                
                scoring = {
                    'accuracy': make_scorer(accuracy_score),
                    'f1': make_scorer(f1_score, average='weighted'),
                    'mcc': make_scorer(safe_mcc_score),
                    'auc': make_scorer(safe_auc_score, response_method='predict_proba')
                }
            
            # Debug cross-validation setup
            logger.debug(f"About to run cross-validation with:")
            logger.debug(f"  X_final shape: {X_final.shape}")
            logger.debug(f"  y_final shape: {y_final.shape}")
            logger.debug(f"  CV strategy: {cv_strategy}")
            logger.debug(f"  CV type: {type(cv_strategy)}")
            
            # Check CV splits before running
            try:
                n_splits = cv_strategy.get_n_splits(X_final, y_final)
                logger.debug(f"  Number of CV splits: {n_splits}")
                
                # Test the first split to see what's happening
                if cv_groups is not None:
                    split_gen = cv_strategy.split(X_final, y_final, cv_groups)
                else:
                    split_gen = cv_strategy.split(X_final, y_final)
                
                train_idx, val_idx = next(split_gen)
                logger.debug(f"  First split - train: {len(train_idx)} samples, val: {len(val_idx)} samples")
                logger.debug(f"  Train indices range: {train_idx[:5]}...{train_idx[-5:] if len(train_idx) > 5 else train_idx}")
                logger.debug(f"  Val indices range: {val_idx[:5]}...{val_idx[-5:] if len(val_idx) > 5 else val_idx}")
                
            except Exception as e:
                logger.error(f"Error analyzing CV splits: {e}")
            
            # Run cross-validation with proper groups handling
            if cv_groups is not None:
                # For grouped CV strategies like StratifiedGroupKFold, GroupKFold
                logger.debug(f"Using grouped CV with {len(cv_groups)} group labels")
                cv_results = cross_validate(
                    model_obj, X_final, y_final, 
                    cv=cv_strategy, 
                    scoring=scoring, 
                    return_train_score=True,
                    error_score='raise',
                    groups=cv_groups
                )
            else:
                # For non-grouped CV strategies
                logger.debug(f"Using non-grouped CV")
                cv_results = cross_validate(
                    model_obj, X_final, y_final, 
                    cv=cv_strategy, 
                    scoring=scoring, 
                    return_train_score=True,
                    error_score='raise'
                )
            
            # Compile results
            results = {
                'n_samples': X_final.shape[0],
                'n_features': X_final.shape[1],
                'cv_scores': cv_results,
                'mean_scores': {metric: np.mean(scores) for metric, scores in cv_results.items()},
                'std_scores': {metric: np.std(scores) for metric, scores in cv_results.items()}
            }
            
            logger.info(f"Cross-validation completed for {model_name}")
            return results
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            return {"error": str(e)}


def run_feature_first_pipeline(ds_name: str,
                              data_modalities: Dict[str, pd.DataFrame],
                              common_ids: List[str],
                              y: np.ndarray,
                              base_out: str,
                              algorithms: Dict[str, Any],
                              n_values: Union[List[int], Dict[str, List[int]]],
                              models: List[str],
                              is_regression: bool = True,
                              missing_percentage: float = 0.0) -> None:
    """
    Run the complete feature-first pipeline for a dataset.
    
    NEW EXPERIMENTAL LOOP:
    for algorithm in algorithms:
        for n_features_components in n_values:
            for fusion_method in fusion_methods:
                for model in models:
                    # 1. Apply algorithm to each modality separately
                    # 2. Apply fusion to processed features  
                    # 3. Train model on fused processed features
    
    Parameters
    ----------
    ds_name : str
        Dataset name
    data_modalities : Dict[str, pd.DataFrame]
        Raw modality data
    common_ids : List[str]
        Common sample IDs
    y : np.ndarray
        Target values
    base_out : str
        Output directory
    algorithms : Dict[str, Any]
        Feature processing algorithms
    n_values : Union[List[int], Dict[str, List[int]]]
        Number of features/components values
    models : List[str]
        Model names
    is_regression : bool
        Whether this is a regression task
    missing_percentage : float
        Percentage of missing data
    """
    
    suppress_sklearn_warnings()
    logger.info(f"Starting feature-first pipeline for {ds_name}")
    
    # Get available fusion methods based on missing data
    task_type = "regression" if is_regression else "classification"
    
    if missing_percentage == 0.0:
        fusion_methods = ['attention_weighted', 'learnable_weighted', 'mkl', 'average', 'sum', 'early_fusion_pca']
    else:
        fusion_methods = ['mkl', 'average', 'sum', 'early_fusion_pca']
    
    logger.info(f"Available fusion methods for {missing_percentage:.1%} missing: {fusion_methods}")
    
    # Import model objects
    from models import build_model
    if is_regression:
        model_names = ["LinearRegression", "ElasticNet", "RandomForestRegressor"]
        model_objects = {name: build_model(name, "reg") for name in model_names}
    else:
        model_names = ["LogisticRegression", "RandomForestClassifier", "SVC"]
        model_objects = {name: build_model(name, "clf") for name in model_names}
    
    # CRITICAL FIX: Set up cross-validation with ACTUAL data dimensions
    # Don't use common_ids length which might be stale - use actual y array
    from cv import get_cv_strategy
    logger.info(f"Setting up CV strategy: y.shape={y.shape}, common_ids.length={len(common_ids)}")
    
    # Use actual y dimensions for CV setup
    if len(y) != len(common_ids):
        logger.warning(f"MISMATCH: y has {len(y)} samples but common_ids has {len(common_ids)} - using y.shape")
        # Create sample IDs that match the actual data
        actual_sample_ids = [f"sample_{i}" for i in range(len(y))]
    else:
        actual_sample_ids = common_ids
    
    cv_strategy, cv_groups = get_cv_strategy(len(y), y, is_regression, sample_ids=actual_sample_ids)
    
    all_results = []
    
    # NEW EXPERIMENTAL LOOP: Algorithm → Features → Fusion → Model
    for algorithm_name, algorithm_obj in algorithms.items():
        logger.info(f"Processing algorithm: {algorithm_name}")
        
        # OPTIMIZATION: Extractors use hyperparameter tuning so only need one n_value
        # Selectors should still test multiple n_values for different feature counts
        extractor_algorithms = ['PCA', 'KPCA', 'PLS', 'KPLS', 'SparsePLS', 'LDA', 'FA', 'PLS-DA']
        is_extractor = any(ext in algorithm_name for ext in extractor_algorithms)
        
        if is_extractor:
            # Extractors: Use only one n_value since components are hyperparameter-tuned
            algorithm_n_values = [32]  # Default value, will be overridden by hyperparameters
            logger.info(f"  Extractor detected: using single n_value (hyperparameter-tuned components)")
        else:
            # Selectors: Test multiple n_values for different feature counts
            if isinstance(n_values, dict):
                algorithm_n_values = n_values.get(algorithm_name, [8, 16, 32])
            else:
                algorithm_n_values = n_values
            logger.info(f"  Selector detected: testing multiple n_values {algorithm_n_values}")
        
        for n_val in algorithm_n_values:
            logger.info(f"  Processing n={n_val}")
            
            for fusion_method in fusion_methods:
                logger.info(f"    Processing fusion: {fusion_method}")
                
                # Create feature-first pipeline
                pipeline = FeatureFirstPipeline(
                    fusion_method=fusion_method,
                    task_type=task_type,
                    missing_percentage=missing_percentage
                )
                
                for model_name in models:
                    if model_name not in model_objects:
                        logger.warning(f"Model {model_name} not available")
                        continue
                    
                    logger.info(f"      Training model: {model_name}")
                    
                    # CRITICAL FIX: Create fresh algorithm object for each model
                    # This ensures each model gets its own extractor with model-specific hyperparameters
                    from models import build_model
                    fresh_algorithm_obj = algorithms[algorithm_name]
                    
                    # Handle both extractor objects and selector strings
                    if hasattr(fresh_algorithm_obj, 'get_params'):
                        # Extractor object - create a fresh instance for this specific model
                        fresh_algorithm_obj = fresh_algorithm_obj.__class__(**fresh_algorithm_obj.get_params())
                    # If it's a string (selector code), just use it as-is
                    
                    # Run feature-first experiment with fresh algorithm
                    results = pipeline.run_feature_first_experiment(
                        data_modalities=data_modalities,
                        y=y,
                        common_ids=common_ids,
                        algorithm_name=algorithm_name,
                        algorithm_obj=fresh_algorithm_obj,
                        n_features_or_components=n_val,
                        model_name=model_name,
                        model_obj=model_objects[model_name],
                        cv_strategy=cv_strategy,
                        cv_groups=cv_groups,
                        dataset_name=ds_name,
                        fold_idx=0
                    )
                    
                    # Add experiment metadata
                    results.update({
                        'dataset': ds_name,
                        'algorithm': algorithm_name,
                        'n_value': n_val,
                        'fusion_method': fusion_method,
                        'model': model_name,
                        'task_type': task_type,
                        'missing_percentage': missing_percentage
                    })
                    
                    all_results.append(results)
                    
                    # Log results with standard deviations
                    if 'error' not in results and 'mean_scores' in results and 'std_scores' in results:
                        if is_regression:
                            # Regression metrics with ± standard deviation
                            r2_mean = results['mean_scores'].get('test_r2', 0.0)
                            r2_std = results['std_scores'].get('test_r2', 0.0)
                            mse_mean = -results['mean_scores'].get('test_mse', 0.0)
                            mse_std = results['std_scores'].get('test_mse', 0.0)
                            mae_mean = -results['mean_scores'].get('test_mae', 0.0)
                            mae_std = results['std_scores'].get('test_mae', 0.0)
                            
                            logger.info(f"        Results:")
                            logger.info(f"          R² Score: {r2_mean:.4f} ± {r2_std:.4f}")
                            logger.info(f"          MSE: {mse_mean:.4f} ± {mse_std:.4f}")
                            logger.info(f"          MAE: {mae_mean:.4f} ± {mae_std:.4f}")
                        else:
                            # Classification metrics with ± standard deviation
                            acc_mean = results['mean_scores'].get('test_accuracy', 0.0)
                            acc_std = results['std_scores'].get('test_accuracy', 0.0)
                            f1_mean = results['mean_scores'].get('test_f1', 0.0)
                            f1_std = results['std_scores'].get('test_f1', 0.0)
                            mcc_mean = results['mean_scores'].get('test_mcc', 0.0)
                            mcc_std = results['std_scores'].get('test_mcc', 0.0)
                            auc_mean = results['mean_scores'].get('test_auc', 0.0)
                            auc_std = results['std_scores'].get('test_auc', 0.0)
                            
                            logger.info(f"        Results:")
                            logger.info(f"          Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
                            logger.info(f"          F1 Score: {f1_mean:.4f} ± {f1_std:.4f}")
                            logger.info(f"          MCC Score: {mcc_mean:.4f} ± {mcc_std:.4f}")
                            logger.info(f"          AUC Score: {auc_mean:.4f} ± {auc_std:.4f}")
                    else:
                        logger.warning(f"        Experiment failed: {results.get('error', 'Unknown error')}")
    
    # Save results
    import os
    os.makedirs(base_out, exist_ok=True)
    
    results_file = os.path.join(base_out, f"{ds_name}_feature_first_results.json")
    import json
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info(f"Feature-first pipeline completed for {ds_name}, results saved to {results_file}") 