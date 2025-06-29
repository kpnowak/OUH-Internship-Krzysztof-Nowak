#!/usr/bin/env python3
"""
Safe samplers module for handling class imbalance with robust edge case handling.
"""

import numpy as np
import logging
from typing import Union, Optional
from imblearn.over_sampling import SMOTE, RandomOverSampler

logger = logging.getLogger(__name__)

def safe_sampler(y, k_default=5, random_state=42):
    """
    Return the best oversampler given minority class size.
    
    This function automatically selects the most appropriate oversampling strategy
    based on the class distribution in the target variable.
    
    Parameters
    ----------
    y : array-like
        Target variable
    k_default : int, default=5
        Default number of neighbors for SMOTE
    random_state : int, default=42
        Random state for reproducibility
        
    Returns
    -------
    sampler : object
        Appropriate oversampler (SMOTE, SMOTEN, RandomOverSampler, or None)
    """
    try:
        from imblearn.over_sampling import SMOTE, RandomOverSampler
        
        # Handle different input types
        if hasattr(y, 'dtype') and y.dtype.kind in 'iu':
            # Integer array - use bincount
            class_counts = np.bincount(y)
            # Remove zero counts (unused class labels)
            class_counts = class_counts[class_counts > 0]
        else:
            # General case - use unique
            unique_classes, class_counts = np.unique(y, return_counts=True)
        
        min_c = class_counts.min()
        n_classes = len(class_counts)
        
        logger.debug(f"Class distribution: {class_counts}, min_class_size: {min_c}, n_classes: {n_classes}")
        
        # Case 1: Extremely small minority class (< 6 samples)
        # This is more conservative to account for CV splits
        if min_c < 6:
            logger.info(f"Minority class has {min_c} samples - using RandomOverSampler for safety")
            return RandomOverSampler(random_state=random_state, sampling_strategy='not majority')
        
        # Case 2: Small minority class (6 <= samples <= k_default + 2)
        # Added buffer of +2 to be more conservative
        elif min_c <= k_default + 2:
            # Use even more conservative k_neighbors calculation
            safe_k = max(1, min_c - 2)  # Leave more buffer
            logger.info(f"Minority class has {min_c} samples - using SMOTE with k_neighbors={safe_k}")
            return SMOTE(k_neighbors=safe_k, random_state=random_state)
        
        # Case 3: Sufficient samples for default SMOTE
        else:
            logger.debug(f"Using standard SMOTE with k_neighbors={k_default}")
            return SMOTE(k_neighbors=k_default, random_state=random_state)
            
    except ImportError as e:
        logger.warning(f"imbalanced-learn not available: {e}")
        return None
    except Exception as e:
        logger.warning(f"Error creating safe sampler: {e}, falling back to RandomOverSampler")
        try:
            from imblearn.over_sampling import RandomOverSampler
            return RandomOverSampler(random_state=random_state)
        except ImportError:
            logger.warning("RandomOverSampler also not available, returning None")
            return None

def safe_smoten_sampler(y, k_default=5, random_state=42):
    """
    Return SMOTEN (for mixed data types) or fallback sampler.
    
    SMOTEN is designed for datasets with mixed categorical and continuous features.
    
    Parameters
    ----------
    y : array-like
        Target variable
    k_default : int, default=5
        Default number of neighbors for SMOTEN
    random_state : int, default=42
        Random state for reproducibility
        
    Returns
    -------
    sampler : object
        Appropriate oversampler (SMOTEN, RandomOverSampler, or None)
    """
    try:
        from imblearn.over_sampling import SMOTEN, RandomOverSampler
        
        # Get class distribution
        if hasattr(y, 'dtype') and y.dtype.kind in 'iu':
            class_counts = np.bincount(y)
            class_counts = class_counts[class_counts > 0]
        else:
            unique_classes, class_counts = np.unique(y, return_counts=True)
        
        min_c = class_counts.min()
        
        logger.debug(f"SMOTEN sampler - Class distribution: {class_counts}, min_class_size: {min_c}")
        
        # Case 1: Too few samples for SMOTEN
        if min_c < 3:
            logger.info(f"Minority class has {min_c} samples - using RandomOverSampler instead of SMOTEN")
            return RandomOverSampler(random_state=random_state, sampling_strategy='not majority')
        
        # Case 2: Adjust k_neighbors for small classes
        elif min_c <= k_default:
            safe_k = min_c - 1
            logger.info(f"Using SMOTEN with adjusted k_neighbors={safe_k}")
            return SMOTEN(k_neighbors=safe_k, random_state=random_state)
        
        # Case 3: Use default SMOTEN
        else:
            logger.debug(f"Using standard SMOTEN with k_neighbors={k_default}")
            return SMOTEN(k_neighbors=k_default, random_state=random_state)
            
    except ImportError as e:
        logger.warning(f"SMOTEN not available: {e}, falling back to safe_sampler")
        return safe_sampler(y, k_default, random_state)
    except Exception as e:
        logger.warning(f"Error creating SMOTEN sampler: {e}, falling back to safe_sampler")
        return safe_sampler(y, k_default, random_state)

def create_adaptive_pipeline(base_model, y_train, use_oversampling=True, use_undersampling=True, 
                           sampler_type='auto', k_neighbors=5, random_state=42):
    """
    Create an adaptive pipeline that automatically selects the best sampling strategy.
    
    Parameters
    ----------
    base_model : sklearn estimator
        Base model to wrap in the pipeline
    y_train : array-like
        Training target variable for determining sampling strategy
    use_oversampling : bool, default=True
        Whether to use oversampling
    use_undersampling : bool, default=True
        Whether to use undersampling
    sampler_type : str, default='auto'
        Type of sampler ('auto', 'smote', 'smoten', 'random')
    k_neighbors : int, default=5
        Number of neighbors for SMOTE/SMOTEN
    random_state : int, default=42
        Random state for reproducibility
        
    Returns
    -------
    pipeline : imblearn.pipeline.Pipeline or sklearn estimator
        Adaptive pipeline or original model if sampling not needed/available
    """
    try:
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.under_sampling import RandomUnderSampler
        
        steps = []
        
        # Add oversampling step if requested
        if use_oversampling:
            if sampler_type == 'auto':
                oversampler = safe_sampler(y_train, k_neighbors, random_state)
            elif sampler_type == 'smote':
                oversampler = safe_sampler(y_train, k_neighbors, random_state)
            elif sampler_type == 'smoten':
                oversampler = safe_smoten_sampler(y_train, k_neighbors, random_state)
            elif sampler_type == 'random':
                from imblearn.over_sampling import RandomOverSampler
                oversampler = RandomOverSampler(random_state=random_state)
            else:
                logger.warning(f"Unknown sampler_type: {sampler_type}, using auto")
                oversampler = safe_sampler(y_train, k_neighbors, random_state)
            
            if oversampler is not None:
                steps.append(('over', oversampler))
                logger.debug(f"Added oversampler: {type(oversampler).__name__}")
        
        # Add undersampling step if requested
        if use_undersampling and len(steps) > 0:  # Only undersample if we oversampled
            undersampler = RandomUnderSampler(random_state=random_state)
            steps.append(('under', undersampler))
            logger.debug("Added RandomUnderSampler")
        
        # Add the model
        steps.append(('model', base_model))
        
        # Create pipeline if we have sampling steps
        if len(steps) > 1:
            pipeline = ImbPipeline(steps=steps)
            logger.info(f"Created adaptive pipeline with {len(steps)-1} sampling step(s)")
            return pipeline
        else:
            logger.info("No sampling applied, returning original model")
            return base_model
            
    except ImportError:
        logger.warning("imbalanced-learn not available, using original model without sampling")
        return base_model
    except Exception as e:
        logger.warning(f"Error creating adaptive pipeline: {str(e)}, using original model")
        return base_model

def dynamic_cv(y, max_splits=5, is_regression=False):
    """
    Create dynamic CV splitter that adapts to data characteristics.
    
    Parameters
    ----------
    y : array-like
        Target variable
    max_splits : int, default=5
        Maximum number of CV splits
    is_regression : bool, default=False
        Whether this is a regression task
        
    Returns
    -------
    cv_splitter : object
        Appropriate CV splitter
    """
    from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut
    
    n_samples = len(y)
    
    if is_regression:
        # For regression, base splits on sample size
        if n_samples < 6:
            return LeaveOneOut()
        elif n_samples < 15:
            n_splits = 2
        elif n_samples < 30:
            n_splits = 3
        else:
            n_splits = min(max_splits, n_samples // 10)
        
        return KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    else:
        # For classification, consider class distribution
        unique_classes, class_counts = np.unique(y, return_counts=True)
        min_class_count = np.min(class_counts)
        
        # Calculate maximum possible splits while maintaining minimum samples per class per fold
        max_splits_by_class = min_class_count // 2  # At least 2 samples per class per fold
        max_splits_by_total = n_samples // 5  # At least 5 total samples per fold
        
        # Take the minimum of both constraints
        n_splits = min(max_splits, int(max_splits_by_class), int(max_splits_by_total))
        
        # Handle edge cases
        if n_splits < 2:
            if n_samples < 6:
                return LeaveOneOut()
            else:
                # Use KFold as fallback
                n_splits = min(2, n_samples // 3)
                return KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        logger.debug(f"Dynamic CV: n_samples={n_samples}, min_class_count={min_class_count}, n_splits={n_splits}")
        
        try:
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        except Exception as e:
            logger.warning(f"StratifiedKFold failed: {e}, falling back to KFold")
            return KFold(n_splits=n_splits, shuffle=True, random_state=42)

def get_class_distribution_info(y):
    """
    Get detailed information about class distribution.
    
    Parameters
    ----------
    y : array-like
        Target variable
        
    Returns
    -------
    dict
        Dictionary with class distribution information
    """
    if hasattr(y, 'dtype') and y.dtype.kind in 'iu':
        class_counts = np.bincount(y)
        # Get non-zero counts and their indices (class labels)
        non_zero_mask = class_counts > 0
        class_labels = np.where(non_zero_mask)[0]
        class_counts = class_counts[non_zero_mask]
    else:
        class_labels, class_counts = np.unique(y, return_counts=True)
    
    total_samples = len(y)
    n_classes = len(class_counts)
    min_class_size = class_counts.min()
    max_class_size = class_counts.max()
    
    # Calculate imbalance ratio
    imbalance_ratio = max_class_size / min_class_size if min_class_size > 0 else float('inf')
    
    return {
        'total_samples': total_samples,
        'n_classes': n_classes,
        'class_labels': class_labels,
        'class_counts': class_counts,
        'min_class_size': min_class_size,
        'max_class_size': max_class_size,
        'imbalance_ratio': imbalance_ratio,
        'class_distribution': dict(zip(class_labels, class_counts))
    }

def recommend_sampling_strategy(y, verbose=True):
    """
    Recommend the best sampling strategy for the given target distribution.
    
    Parameters
    ----------
    y : array-like
        Target variable
    verbose : bool, default=True
        Whether to print recommendations
        
    Returns
    -------
    dict
        Dictionary with recommendations
    """
    info = get_class_distribution_info(y)
    
    recommendations = {
        'use_oversampling': False,
        'use_undersampling': False,
        'sampler_type': 'none',
        'k_neighbors': 5,
        'reasoning': []
    }
    
    # Decision logic
    if info['n_classes'] < 2:
        recommendations['reasoning'].append("Single class detected - no sampling needed")
    elif info['min_class_size'] < 2:
        recommendations['reasoning'].append("Class with <2 samples - sampling not recommended")
    elif info['imbalance_ratio'] < 2:
        recommendations['reasoning'].append("Classes are well balanced - no sampling needed")
    elif info['min_class_size'] < 3:
        recommendations['use_oversampling'] = True
        recommendations['sampler_type'] = 'random'
        recommendations['reasoning'].append("Very small minority class - use RandomOverSampler")
    elif info['min_class_size'] <= 5:
        recommendations['use_oversampling'] = True
        recommendations['sampler_type'] = 'smote'
        recommendations['k_neighbors'] = info['min_class_size'] - 1
        recommendations['reasoning'].append(f"Small minority class - use SMOTE with k={info['min_class_size']-1}")
    elif info['imbalance_ratio'] > 3:
        recommendations['use_oversampling'] = True
        recommendations['sampler_type'] = 'smote'
        recommendations['reasoning'].append("Moderately imbalanced - use SMOTE")
    
    if verbose:
        print("Class Distribution Analysis:")
        print(f"  Total samples: {info['total_samples']}")
        print(f"  Number of classes: {info['n_classes']}")
        print(f"  Class distribution: {info['class_distribution']}")
        print(f"  Imbalance ratio: {info['imbalance_ratio']:.2f}")
        print("\nRecommendations:")
        for reason in recommendations['reasoning']:
            print(f"  - {reason}")
        print(f"  Suggested strategy: {recommendations['sampler_type']}")
        if recommendations['use_oversampling']:
            print(f"  Use oversampling: Yes")
            if recommendations['sampler_type'] == 'smote':
                print(f"  SMOTE k_neighbors: {recommendations['k_neighbors']}")
        if recommendations['use_undersampling']:
            print(f"  Use undersampling: Yes")
    
    return recommendations