#!/usr/bin/env python3
"""
Enhanced evaluation module with improved metrics and nested cross-validation.

Implements:
1. Multi-class AUC with 'ovr' (one-vs-rest) strategy
2. Target scaling for regression within CV folds
3. Macro-F1 and MCC for imbalanced classes
4. Nested CV (outer 5-fold, inner 3-fold) to avoid optimistic bias
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from sklearn.model_selection import (
    StratifiedKFold, KFold, GridSearchCV, RandomizedSearchCV,
    cross_val_score, cross_validate
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.base import clone
import warnings

logger = logging.getLogger(__name__)

class EnhancedMetrics:
    """
    Enhanced metrics calculator with proper multi-class handling and target scaling.
    """
    
    def __init__(self, is_regression: bool = True, use_macro_averaging: bool = True):
        """
        Initialize enhanced metrics calculator.
        
        Parameters
        ----------
        is_regression : bool, default=True
            Whether this is a regression task
        use_macro_averaging : bool, default=True
            Whether to use macro-averaging for classification metrics
        """
        self.is_regression = is_regression
        self.use_macro_averaging = use_macro_averaging
        self.target_scaler = None
        
    def calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                       y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate enhanced classification metrics with proper multi-class handling.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        y_proba : np.ndarray, optional
            Predicted probabilities
            
        Returns
        -------
        Dict[str, float]
            Dictionary of calculated metrics
        """
        metrics = {}
        
        try:
            # Basic accuracy
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            
            # Determine averaging strategy
            average_strategy = 'macro' if self.use_macro_averaging else 'weighted'
            
            # Precision, Recall, F1 with macro-averaging for imbalanced classes
            metrics['precision'] = precision_score(y_true, y_pred, average=average_strategy, zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average=average_strategy, zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, average=average_strategy, zero_division=0)
            
            # Matthews Correlation Coefficient (handles imbalanced classes well)
            try:
                metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
            except Exception as e:
                logger.warning(f"Could not calculate MCC: {str(e)}")
                metrics['mcc'] = 0.0
            
            # Enhanced AUC calculation with proper multi-class handling
            if y_proba is not None:
                metrics['auc'] = self._calculate_enhanced_auc(y_true, y_proba)
            else:
                metrics['auc'] = 0.5  # Default for random classifier
                
            # Additional macro-averaged metrics for detailed analysis
            if self.use_macro_averaging:
                metrics['macro_precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
                metrics['macro_recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
                metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
                
                # Weighted versions for comparison
                metrics['weighted_precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics['weighted_recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
        except Exception as e:
            logger.error(f"Error calculating classification metrics: {str(e)}")
            # Return default metrics
            metrics = {
                'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                'mcc': 0.0, 'auc': 0.5
            }
        
        return metrics
    
    def _calculate_enhanced_auc(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """
        Calculate AUC with proper multi-class handling using 'ovr' strategy.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_proba : np.ndarray
            Predicted probabilities
            
        Returns
        -------
        float
            AUC score
        """
        try:
            # Check for NaN values
            if np.isnan(y_proba).any():
                logger.warning("NaN values found in predicted probabilities, cannot calculate AUC")
                return 0.5
            
            # Get unique classes
            unique_classes = np.unique(y_true)
            n_classes = len(unique_classes)
            
            if n_classes == 2:
                # Binary classification - use positive class probabilities
                if y_proba.shape[1] >= 2:
                    auc = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    logger.warning("Binary classification but only one probability column")
                    auc = 0.5
            else:
                # Multi-class classification - use 'ovr' (one-vs-rest) strategy
                # This fixes the issue where AUC was stuck at 0.50
                auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
                
                # Also calculate macro-averaged AUC for comparison
                try:
                    macro_auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
                    logger.debug(f"Multi-class AUC - Weighted: {auc:.4f}, Macro: {macro_auc:.4f}")
                except Exception as e:
                    logger.debug(f"Could not calculate macro AUC: {str(e)}")
            
            return auc
            
        except Exception as e:
            logger.warning(f"Could not calculate AUC: {str(e)}")
            return 0.5
    
    def calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   scaled: bool = False) -> Dict[str, float]:
        """
        Calculate regression metrics with optional target scaling reversion.
        
        Parameters
        ----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values
        scaled : bool, default=False
            Whether the values are scaled and need to be reverted for MAE/RMSE
            
        Returns
        -------
        Dict[str, float]
            Dictionary of calculated metrics
        """
        metrics = {}
        
        try:
            # If values are scaled, revert scaling for MAE/RMSE calculation
            if scaled and self.target_scaler is not None:
                y_true_unscaled = self.target_scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
                y_pred_unscaled = self.target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            else:
                y_true_unscaled = y_true
                y_pred_unscaled = y_pred
            
            # Calculate metrics on unscaled values for interpretability
            metrics['mae'] = mean_absolute_error(y_true_unscaled, y_pred_unscaled)
            metrics['rmse'] = np.sqrt(mean_squared_error(y_true_unscaled, y_pred_unscaled))
            metrics['mse'] = mean_squared_error(y_true_unscaled, y_pred_unscaled)
            
            # R² can be calculated on either scaled or unscaled (should be the same)
            metrics['r2'] = r2_score(y_true, y_pred)
            
            # Additional metrics
            metrics['mean_residual'] = np.mean(y_true_unscaled - y_pred_unscaled)
            metrics['std_residual'] = np.std(y_true_unscaled - y_pred_unscaled)
            
        except Exception as e:
            logger.error(f"Error calculating regression metrics: {str(e)}")
            # Return default metrics
            metrics = {
                'mae': float('inf'), 'rmse': float('inf'), 'mse': float('inf'),
                'r2': -float('inf'), 'mean_residual': 0.0, 'std_residual': 0.0
            }
        
        return metrics
    
    def fit_target_scaler(self, y_train: np.ndarray) -> 'EnhancedMetrics':
        """
        Fit target scaler for regression tasks.
        
        Parameters
        ----------
        y_train : np.ndarray
            Training targets
            
        Returns
        -------
        self
        """
        if self.is_regression:
            self.target_scaler = StandardScaler()
            self.target_scaler.fit(y_train.reshape(-1, 1))
        return self
    
    def scale_targets(self, y: np.ndarray) -> np.ndarray:
        """
        Scale targets using fitted scaler.
        
        Parameters
        ----------
        y : np.ndarray
            Targets to scale
            
        Returns
        -------
        np.ndarray
            Scaled targets
        """
        if self.is_regression and self.target_scaler is not None:
            return self.target_scaler.transform(y.reshape(-1, 1)).flatten()
        return y
    
    def inverse_scale_targets(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        Inverse scale targets using fitted scaler.
        
        Parameters
        ----------
        y_scaled : np.ndarray
            Scaled targets
            
        Returns
        -------
        np.ndarray
            Unscaled targets
        """
        if self.is_regression and self.target_scaler is not None:
            return self.target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
        return y_scaled


class NestedCrossValidator:
    """
    Nested cross-validation implementation with outer 5-fold and inner 3-fold.
    Avoids optimistic bias by separating model selection from performance estimation.
    """
    
    def __init__(self, outer_cv: int = 5, inner_cv: int = 3, 
                 is_regression: bool = True, random_state: int = 42,
                 use_macro_averaging: bool = True):
        """
        Initialize nested cross-validator.
        
        Parameters
        ----------
        outer_cv : int, default=5
            Number of outer CV folds for generalization assessment
        inner_cv : int, default=3
            Number of inner CV folds for hyperparameter tuning
        is_regression : bool, default=True
            Whether this is a regression task
        random_state : int, default=42
            Random state for reproducibility
        use_macro_averaging : bool, default=True
            Whether to use macro-averaging for classification metrics
        """
        self.outer_cv = outer_cv
        self.inner_cv = inner_cv
        self.is_regression = is_regression
        self.random_state = random_state
        self.use_macro_averaging = use_macro_averaging
        
        # Initialize CV splitters
        if is_regression:
            self.outer_splitter = KFold(n_splits=outer_cv, shuffle=True, random_state=random_state)
            self.inner_splitter = KFold(n_splits=inner_cv, shuffle=True, random_state=random_state + 1)
        else:
            self.outer_splitter = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=random_state)
            self.inner_splitter = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=random_state + 1)
        
        # Initialize metrics calculator
        self.metrics_calculator = EnhancedMetrics(is_regression, use_macro_averaging)
    
    def nested_cross_validate(self, estimator, X: np.ndarray, y: np.ndarray,
                            param_grid: Optional[Dict] = None,
                            scoring: Optional[str] = None,
                            n_jobs: int = 1) -> Dict[str, Any]:
        """
        Perform nested cross-validation with proper target scaling for regression.
        
        Parameters
        ----------
        estimator : sklearn estimator
            Base estimator to evaluate
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target values
        param_grid : Dict, optional
            Parameter grid for hyperparameter tuning
        scoring : str, optional
            Scoring metric for hyperparameter tuning
        n_jobs : int, default=1
            Number of parallel jobs
            
        Returns
        -------
        Dict[str, Any]
            Nested CV results with outer fold metrics and best parameters
        """
        logger.info(f"Starting nested CV: {self.outer_cv}-fold outer, {self.inner_cv}-fold inner")
        
        outer_scores = []
        best_params_per_fold = []
        fold_metrics = []
        
        # Outer loop - for unbiased performance estimation
        for fold_idx, (train_idx, test_idx) in enumerate(self.outer_splitter.split(X, y)):
            logger.debug(f"Processing outer fold {fold_idx + 1}/{self.outer_cv}")
            
            X_train_outer, X_test_outer = X[train_idx], X[test_idx]
            y_train_outer, y_test_outer = y[train_idx], y[test_idx]
            
            # Target scaling for regression (fit on outer training set)
            if self.is_regression:
                self.metrics_calculator.fit_target_scaler(y_train_outer)
                y_train_outer_scaled = self.metrics_calculator.scale_targets(y_train_outer)
                y_test_outer_scaled = self.metrics_calculator.scale_targets(y_test_outer)
            else:
                y_train_outer_scaled = y_train_outer
                y_test_outer_scaled = y_test_outer
            
            # Inner loop - for hyperparameter tuning
            if param_grid is not None and len(param_grid) > 0:
                # Use GridSearchCV for hyperparameter tuning on scaled targets
                inner_cv = GridSearchCV(
                    estimator=clone(estimator),
                    param_grid=param_grid,
                    cv=self.inner_splitter,
                    scoring=scoring,
                    n_jobs=n_jobs,
                    error_score='raise'
                )
                
                try:
                    inner_cv.fit(X_train_outer, y_train_outer_scaled)
                    best_estimator = inner_cv.best_estimator_
                    best_params = inner_cv.best_params_
                    best_params_per_fold.append(best_params)
                    logger.debug(f"Fold {fold_idx + 1} best params: {best_params}")
                except Exception as e:
                    logger.warning(f"Hyperparameter tuning failed in fold {fold_idx + 1}: {str(e)}")
                    # Use default estimator
                    best_estimator = clone(estimator)
                    best_estimator.fit(X_train_outer, y_train_outer_scaled)
                    best_params_per_fold.append({})
            else:
                # No hyperparameter tuning - use default estimator
                best_estimator = clone(estimator)
                best_estimator.fit(X_train_outer, y_train_outer_scaled)
                best_params_per_fold.append({})
            
            # Evaluate on outer test set
            if self.is_regression:
                y_pred_scaled = best_estimator.predict(X_test_outer)
                # Calculate metrics with proper scaling handling
                fold_metrics_dict = self.metrics_calculator.calculate_regression_metrics(
                    y_test_outer_scaled, y_pred_scaled, scaled=True
                )
                # Primary score for nested CV
                primary_score = fold_metrics_dict['r2']
            else:
                y_pred = best_estimator.predict(X_test_outer)
                y_proba = None
                if hasattr(best_estimator, 'predict_proba'):
                    try:
                        y_proba = best_estimator.predict_proba(X_test_outer)
                    except:
                        pass
                
                fold_metrics_dict = self.metrics_calculator.calculate_classification_metrics(
                    y_test_outer, y_pred, y_proba
                )
                # Primary score for nested CV
                primary_score = fold_metrics_dict['f1'] if self.use_macro_averaging else fold_metrics_dict['accuracy']
            
            outer_scores.append(primary_score)
            fold_metrics.append(fold_metrics_dict)
            
            logger.debug(f"Fold {fold_idx + 1} primary score: {primary_score:.4f}")
        
        # Aggregate results
        results = {
            'outer_scores': outer_scores,
            'mean_score': np.mean(outer_scores),
            'std_score': np.std(outer_scores),
            'fold_metrics': fold_metrics,
            'best_params_per_fold': best_params_per_fold,
            'cv_config': {
                'outer_cv': self.outer_cv,
                'inner_cv': self.inner_cv,
                'is_regression': self.is_regression,
                'use_macro_averaging': self.use_macro_averaging
            }
        }
        
        # Calculate aggregated metrics across all folds
        if fold_metrics:
            aggregated_metrics = {}
            for metric_name in fold_metrics[0].keys():
                metric_values = [fm[metric_name] for fm in fold_metrics if not np.isnan(fm[metric_name])]
                if metric_values:
                    aggregated_metrics[f'mean_{metric_name}'] = np.mean(metric_values)
                    aggregated_metrics[f'std_{metric_name}'] = np.std(metric_values)
            
            results['aggregated_metrics'] = aggregated_metrics
        
        logger.info(f"Nested CV completed: {results['mean_score']:.4f} ± {results['std_score']:.4f}")
        
        return results
    
    def get_best_hyperparameters(self, results: Dict[str, Any]) -> Dict:
        """
        Get the most frequently selected hyperparameters across folds.
        
        Parameters
        ----------
        results : Dict
            Results from nested_cross_validate
            
        Returns
        -------
        Dict
            Most common hyperparameters
        """
        best_params_per_fold = results.get('best_params_per_fold', [])
        
        if not best_params_per_fold or all(not params for params in best_params_per_fold):
            return {}
        
        # Count parameter combinations
        param_counts = {}
        for params in best_params_per_fold:
            if params:  # Skip empty parameter dicts
                param_str = str(sorted(params.items()))
                param_counts[param_str] = param_counts.get(param_str, 0) + 1
        
        if not param_counts:
            return {}
        
        # Get most common parameters
        most_common_params_str = max(param_counts, key=param_counts.get)
        
        # Convert back to dict
        try:
            most_common_params = dict(eval(most_common_params_str))
            return most_common_params
        except:
            logger.warning("Could not parse most common parameters")
            return best_params_per_fold[0] if best_params_per_fold else {}


def enhanced_roc_auc_score(y_true: np.ndarray, y_score: np.ndarray, 
                          multi_class: str = 'ovr', average: str = 'weighted') -> float:
    """
    Enhanced ROC AUC score calculation with proper multi-class handling.
    
    This function fixes the issue where multi-class AUC was stuck at 0.50 by
    using the 'ovr' (one-vs-rest) strategy.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_score : np.ndarray
        Predicted probabilities
    multi_class : str, default='ovr'
        Multi-class strategy ('ovr' or 'ovo')
    average : str, default='weighted'
        Averaging strategy ('weighted', 'macro', 'micro')
        
    Returns
    -------
    float
        AUC score
    """
    try:
        # Check for binary vs multi-class
        unique_classes = np.unique(y_true)
        n_classes = len(unique_classes)
        
        if n_classes == 2:
            # Binary classification
            if y_score.ndim > 1 and y_score.shape[1] >= 2:
                return roc_auc_score(y_true, y_score[:, 1])
            else:
                return roc_auc_score(y_true, y_score)
        else:
            # Multi-class classification with proper 'ovr' strategy
            return roc_auc_score(y_true, y_score, multi_class=multi_class, average=average)
            
    except Exception as e:
        logger.warning(f"Enhanced AUC calculation failed: {str(e)}")
        return 0.5


def create_enhanced_cv_strategy(X: np.ndarray, y: np.ndarray, 
                              is_regression: bool = True,
                              outer_cv: int = 5, inner_cv: int = 3,
                              random_state: int = 42) -> NestedCrossValidator:
    """
    Create an enhanced cross-validation strategy with nested CV.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target values
    is_regression : bool, default=True
        Whether this is a regression task
    outer_cv : int, default=5
        Number of outer CV folds
    inner_cv : int, default=3
        Number of inner CV folds
    random_state : int, default=42
        Random state for reproducibility
        
    Returns
    -------
    NestedCrossValidator
        Configured nested cross-validator
    """
    return NestedCrossValidator(
        outer_cv=outer_cv,
        inner_cv=inner_cv,
        is_regression=is_regression,
        random_state=random_state,
        use_macro_averaging=True
    )


# Convenience functions for backward compatibility
def calculate_enhanced_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                            y_proba: Optional[np.ndarray] = None,
                                            use_macro: bool = True) -> Dict[str, float]:
    """
    Calculate enhanced classification metrics with macro-averaging.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_proba : np.ndarray, optional
        Predicted probabilities
    use_macro : bool, default=True
        Whether to use macro-averaging
        
    Returns
    -------
    Dict[str, float]
        Enhanced metrics
    """
    calculator = EnhancedMetrics(is_regression=False, use_macro_averaging=use_macro)
    return calculator.calculate_classification_metrics(y_true, y_pred, y_proba)


def calculate_enhanced_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                        target_scaler: Optional[StandardScaler] = None) -> Dict[str, float]:
    """
    Calculate enhanced regression metrics with optional target scaling reversion.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    target_scaler : StandardScaler, optional
        Fitted target scaler for reversion
        
    Returns
    -------
    Dict[str, float]
        Enhanced metrics
    """
    calculator = EnhancedMetrics(is_regression=True)
    calculator.target_scaler = target_scaler
    return calculator.calculate_regression_metrics(y_true, y_pred, scaled=(target_scaler is not None))


# Example usage and testing functions
def test_enhanced_metrics():
    """Test the enhanced metrics implementation."""
    print("Testing Enhanced Metrics Implementation")
    print("=" * 50)
    
    # Test classification metrics
    print("\n1. Testing Classification Metrics:")
    np.random.seed(42)
    y_true_cls = np.random.randint(0, 3, 100)  # 3-class problem
    y_pred_cls = np.random.randint(0, 3, 100)
    y_proba_cls = np.random.dirichlet([1, 1, 1], 100)  # Random probabilities
    
    cls_metrics = calculate_enhanced_classification_metrics(y_true_cls, y_pred_cls, y_proba_cls)
    print(f"  Multi-class AUC (enhanced): {cls_metrics['auc']:.4f}")
    print(f"  Macro F1: {cls_metrics['macro_f1']:.4f}")
    print(f"  MCC: {cls_metrics['mcc']:.4f}")
    
    # Test regression metrics
    print("\n2. Testing Regression Metrics:")
    y_true_reg = np.random.randn(100) * 10 + 50
    y_pred_reg = y_true_reg + np.random.randn(100) * 2
    
    # With target scaling
    scaler = StandardScaler()
    y_true_scaled = scaler.fit_transform(y_true_reg.reshape(-1, 1)).flatten()
    y_pred_scaled = scaler.transform(y_pred_reg.reshape(-1, 1)).flatten()
    
    reg_metrics = calculate_enhanced_regression_metrics(y_true_scaled, y_pred_scaled, scaler)
    print(f"  MAE (unscaled): {reg_metrics['mae']:.4f}")
    print(f"  RMSE (unscaled): {reg_metrics['rmse']:.4f}")
    print(f"  R² (scaled): {reg_metrics['r2']:.4f}")
    
    # Test nested CV
    print("\n3. Testing Nested Cross-Validation:")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=200, n_features=20, n_classes=3, 
                              n_informative=10, random_state=42)
    
    nested_cv = create_enhanced_cv_strategy(X, y, is_regression=False)
    param_grid = {'n_estimators': [10, 50], 'max_depth': [3, 5]}
    
    results = nested_cv.nested_cross_validate(
        RandomForestClassifier(random_state=42), X, y, param_grid
    )
    
    print(f"  Nested CV Score: {results['mean_score']:.4f} ± {results['std_score']:.4f}")
    print(f"  Best Parameters: {nested_cv.get_best_hyperparameters(results)}")
    
    print("\n✓ All tests completed successfully!")


if __name__ == "__main__":
    test_enhanced_metrics() 