#!/usr/bin/env python3
"""
Plotting module for visualizing results.
"""

import os
import numpy as np
import pandas as pd
import logging

# Configure matplotlib backend before any matplotlib imports to prevent tkinter errors
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for parallel processing
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.metrics import RocCurveDisplay
from typing import List, Optional, Union, Any

# Set up logger for this module
logger = logging.getLogger(__name__)

def verify_plot_exists(plot_path: str) -> bool:
    """
    Verify that a plot file exists and is not empty.
    
    Parameters
    ----------
    plot_path : str
        Path to the plot file
        
    Returns
    -------
    bool
        True if plot exists and is not empty, False otherwise
    """
    try:
        return os.path.exists(plot_path) and os.path.getsize(plot_path) > 0
    except Exception as e:
        logger.debug(f"Error checking plot file {plot_path}: {str(e)}")
        return False

def plot_regression_scatter(y_test: np.ndarray, y_pred: np.ndarray, title: str, out_path: str) -> bool:
    """
    Plot regression scatter plot and return success status.
    
    Parameters
    ----------
    y_test : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    title : str
        Plot title
    out_path : str
        Output path for the plot
        
    Returns
    -------
    bool
        True if plot was successfully created, False otherwise
    """
    # Check if all values are NaN
    if np.isnan(y_test).all() or np.isnan(y_pred).all():
        logger.warning(f"All values are NaN for {title}, skipping plot")
        return False
        
    try:
        # Create a new figure for each plot
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
        
        # Create scatter plot
        scatter = ax.scatter(y_test, y_pred, alpha=0.5)
        
        # Add diagonal line
        mn = min(min(y_test), min(y_pred))
        mx = max(max(y_test), max(y_pred))
        ax.plot([mn, mx], [mn, mx], 'r--', alpha=0.7)
        
        # Set labels and title
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(title + ": Actual vs. Predicted")
        
        # Save and close
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)  # Explicitly close the figure
        
        return verify_plot_exists(out_path)
    except Exception as e:
        logger.error(f"Error creating scatter plot for {title}: {str(e)}")
        logger.debug(f"Scatter plot error details: {repr(e)}")
        plt.close('all')  # Ensure all figures are closed on error
        return False

def plot_regression_residuals(y_test: np.ndarray, y_pred: np.ndarray, title: str, out_path: str) -> bool:
    """
    Plot regression residuals and return success status.
    
    Parameters
    ----------
    y_test : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    title : str
        Plot title
    out_path : str
        Output path for the plot
        
    Returns
    -------
    bool
        True if plot was successfully created, False otherwise
    """
    # Check if all values are NaN
    if np.isnan(y_test).all() or np.isnan(y_pred).all():
        logger.warning(f"All values are NaN for {title}, skipping plot")
        return False
        
    try:
        # Create a new figure for each plot
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
        
        # Calculate and plot residuals
        residuals = y_test - y_pred
        scatter = ax.scatter(y_pred, residuals, alpha=0.5)
        
        # Add horizontal line at y=0
        ax.axhline(0, color='r', linestyle='--')
        
        # Set labels and title
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residual")
        ax.set_title(title + ": Residual Plot")
        
        # Save and close
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)  # Explicitly close the figure
        
        return verify_plot_exists(out_path)
    except Exception as e:
        logger.error(f"Error creating residuals plot for {title}: {str(e)}")
        logger.debug(f"Residuals plot error details: {repr(e)}")
        plt.close('all')  # Ensure all figures are closed on error
        return False

def plot_confusion_matrix(cm: np.ndarray, class_labels: List, title: str, out_path: str) -> bool:
    """
    Plot confusion matrix and return success status.
    
    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix
    class_labels : List
        Class labels
    title : str
        Plot title
    out_path : str
        Output path for the plot
        
    Returns
    -------
    bool
        True if plot was successfully created, False otherwise
    """
    try:
        # Validate inputs
        if cm is None or cm.size == 0:
            logger.error(f"Empty or None confusion matrix for {title}")
            return False
            
        if class_labels is None or len(class_labels) == 0:
            logger.error(f"Empty or None class labels for {title}")
            return False
            
        # Ensure output directory exists
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        # Create a new figure for each plot
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d',
                    cmap='Blues', xticklabels=class_labels, yticklabels=class_labels,
                    ax=ax)
        
        # Set labels and title
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        
        # Save and close
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)  # Explicitly close the figure
        
        success = verify_plot_exists(out_path)
        if not success:
            logger.error(f"Confusion matrix plot file was not created or is empty: {out_path}")
        return success
    except Exception as e:
        logger.error(f"Error creating confusion matrix for {title}: {str(e)}")
        logger.debug(f"Confusion matrix error details: {repr(e)}")
        import traceback
        logger.debug(f"Confusion matrix traceback:\n{traceback.format_exc()}")
        plt.close('all')  # Ensure all figures are closed on error
        return False

def validate_model_predictions(model: Any, X_test: np.ndarray, y_test: np.ndarray, title: str) -> bool:
    """
    Validate that a model can produce valid predictions and probabilities.
    
    Parameters
    ----------
    model : Any
        Trained model
    X_test : np.ndarray
        Test feature matrix
    y_test : np.ndarray
        Test target values
    title : str
        Title for logging purposes
        
    Returns
    -------
    bool
        True if model predictions are valid, False otherwise
    """
    try:
        # Check basic predictions
        y_pred = model.predict(X_test)
        if np.isnan(y_pred).any():
            logger.warning(f"Model produces NaN predictions for {title}")
            return False
        
        # Check probability predictions if available
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
            if np.isnan(y_proba).any():
                logger.warning(f"Model produces NaN probabilities for {title}")
                return False
            if not np.isfinite(y_proba).all():
                logger.warning(f"Model produces infinite probabilities for {title}")
                return False
            # Check if probabilities sum to 1 (approximately)
            prob_sums = np.sum(y_proba, axis=1)
            if not np.allclose(prob_sums, 1.0, atol=1e-3):
                logger.warning(f"Model probabilities don't sum to 1 for {title}: min={prob_sums.min():.3f}, max={prob_sums.max():.3f}")
                return False
        
        return True
    except Exception as e:
        logger.warning(f"Error validating model predictions for {title}: {str(e)}")
        return False

def plot_roc_curve_binary(model: Any, X_test: np.ndarray, y_test: np.ndarray, class_labels: List, title: str, out_path: str) -> bool:
    """
    Plot ROC curve for binary classification and return success status.
    
    Parameters
    ----------
    model : Any
        Trained model
    X_test : np.ndarray
        Test feature matrix
    y_test : np.ndarray
        Test target values
    class_labels : List
        Class labels
    title : str
        Plot title
    out_path : str
        Output path for the plot
        
    Returns
    -------
    bool
        True if plot was successfully created, False otherwise
    """
    try:
        # Validate inputs
        if model is None:
            logger.error(f"Model is None for {title}")
            return False
            
        if X_test is None or X_test.size == 0:
            logger.error(f"Empty or None X_test for {title}")
            return False
            
        if y_test is None or len(y_test) == 0:
            logger.error(f"Empty or None y_test for {title}")
            return False
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        # Convert y_test to numeric if it's string
        if isinstance(y_test[0], str):
            y_test = np.array([int(x) for x in y_test])
        
        # Validate that we have both classes in the test set
        unique_classes = np.unique(y_test)
        if len(unique_classes) < 2:
            logger.warning(f"Test set only contains one class for {title}: {unique_classes}, cannot create ROC curve")
            return False
        
        # Validate model predictions before proceeding
        if not validate_model_predictions(model, X_test, y_test, title):
            logger.error(f"Model validation failed for {title}, cannot create ROC curve")
            return False
            
        # Get probability predictions - handle case where only one class is predicted
        try:
            y_proba_full = model.predict_proba(X_test)
        except Exception as e:
            logger.error(f"Failed to get probability predictions for {title}: {str(e)}")
            return False
        
        # Check for NaN values in predicted probabilities
        if np.isnan(y_proba_full).any():
            logger.error(f"Error creating binary ROC curve for {title}: Input contains NaN.")
            logger.debug(f"Binary ROC curve error details: ValueError('Input contains NaN.')")
            logger.debug(f"NaN values found in predicted probabilities. Shape: {y_proba_full.shape}, NaN count: {np.isnan(y_proba_full).sum()}")
            return False
        
        # Check for infinite values in predicted probabilities
        if not np.isfinite(y_proba_full).all():
            logger.error(f"Error creating binary ROC curve for {title}: Input contains infinite values.")
            logger.debug(f"Infinite values found in predicted probabilities. Shape: {y_proba_full.shape}")
            return False
        
        # Check if we have probabilities for both classes
        if y_proba_full.shape[1] < 2:
            logger.warning(f"Model only predicts one class for {title}, cannot create ROC curve")
            logger.debug(f"predict_proba shape: {y_proba_full.shape}, unique classes in y_test: {np.unique(y_test)}")
            return False
        
        # Get probability predictions for positive class
        y_proba = y_proba_full[:, 1]
        
        # Additional check for NaN values in the positive class probabilities
        if np.isnan(y_proba).any():
            logger.error(f"NaN values found in positive class probabilities for {title}")
            return False
        
        # Create a new figure for each plot
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
        
        # Create ROC curve
        disp = RocCurveDisplay.from_predictions(
            y_test, 
            y_proba,
            name='Binary ROC',
            pos_label=1,  # Explicitly set positive label
            ax=ax
        )
        
        # Set title
        ax.set_title(title + " - ROC Curve")
        
        # Save and close
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)  # Explicitly close the figure
        
        success = verify_plot_exists(out_path)
        if not success:
            logger.error(f"Binary ROC curve plot file was not created or is empty: {out_path}")
        return success
    except Exception as e:
        logger.error(f"Error creating ROC curve for {title}: {str(e)}")
        logger.debug(f"Binary ROC curve error details: {repr(e)}")
        import traceback
        logger.debug(f"Binary ROC curve traceback:\n{traceback.format_exc()}")
        plt.close('all')  # Ensure all figures are closed on error
        return False

def plot_roc_curve_multiclass(model: Any, X_test: np.ndarray, y_test: np.ndarray, class_labels: List, title: str, out_path: str) -> bool:
    """
    Plot ROC curve for multi-class classification and return success status.
    
    Parameters
    ----------
    model : Any
        Trained model
    X_test : np.ndarray
        Test feature matrix
    y_test : np.ndarray
        Test target values
    class_labels : List
        Class labels
    title : str
        Plot title
    out_path : str
        Output path for the plot
        
    Returns
    -------
    bool
        True if plot was successfully created, False otherwise
    """
    try:
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
        
        # Validate inputs
        if model is None:
            logger.error(f"Model is None for {title}")
            return False
            
        if X_test is None or X_test.size == 0:
            logger.error(f"Empty or None X_test for {title}")
            return False
            
        if y_test is None or len(y_test) == 0:
            logger.error(f"Empty or None y_test for {title}")
            return False
            
        if class_labels is None or len(class_labels) == 0:
            logger.error(f"Empty or None class_labels for {title}")
            return False
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        # Convert y_test to numeric if it's string
        if isinstance(y_test[0], str):
            y_test = np.array([int(x) for x in y_test])
        
        # Get the actual number of classes in the test set and model predictions
        unique_test_classes = np.unique(y_test)
        n_test_classes = len(unique_test_classes)
        n_expected_classes = len(class_labels)
        
        # Check if we have enough classes for ROC analysis
        if n_test_classes < 2:
            logger.warning(f"Test set only contains {n_test_classes} unique class(es) for {title}, cannot create ROC curve")
            logger.debug(f"Unique classes in test set: {unique_test_classes}")
            return False
        
        # Validate model predictions before proceeding
        if not validate_model_predictions(model, X_test, y_test, title):
            logger.error(f"Model validation failed for {title}, cannot create ROC curve")
            return False
            
        # Get probability predictions for all classes
        try:
            y_proba = model.predict_proba(X_test)
        except Exception as e:
            logger.error(f"Failed to get probability predictions for {title}: {str(e)}")
            return False
        
        # Check for NaN values in predicted probabilities
        if np.isnan(y_proba).any():
            logger.error(f"Error creating multi-class ROC curve for {title}: Input contains NaN.")
            logger.debug(f"Multi-class ROC curve error details: ValueError('Input contains NaN.')")
            logger.debug(f"NaN values found in predicted probabilities. Shape: {y_proba.shape}, NaN count: {np.isnan(y_proba).sum()}")
            return False
        
        # Check for infinite values in predicted probabilities
        if not np.isfinite(y_proba).all():
            logger.error(f"Error creating multi-class ROC curve for {title}: Input contains infinite values.")
            logger.debug(f"Infinite values found in predicted probabilities. Shape: {y_proba.shape}")
            return False
        
        n_model_classes = y_proba.shape[1]
        
        # Handle shape mismatches more gracefully
        if n_model_classes != n_expected_classes:
            logger.error(f"Mismatch between predicted probabilities shape ({y_proba.shape[0]}, {n_model_classes}) and class labels {n_expected_classes} for {title}")
            logger.debug(f"Model predicts {n_model_classes} classes, but {n_expected_classes} class labels provided")
            logger.debug(f"Test set contains {n_test_classes} unique classes: {unique_test_classes}")
            logger.debug(f"Expected class labels: {class_labels}")
            return False
        
        # Create a new figure for each plot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        
        if n_expected_classes == 2:
            # For binary classification, use the standard binary ROC
            # Additional check to ensure we have probabilities for both classes
            if y_proba.shape[1] < 2:
                logger.warning(f"Binary classification in multiclass function only has {y_proba.shape[1]} probability columns for {title}")
                return False
            
            # Check for NaN values specifically in the positive class probabilities
            positive_class_proba = y_proba[:, 1]
            if np.isnan(positive_class_proba).any():
                logger.error(f"NaN values found in positive class probabilities for {title}")
                return False
            
            fpr, tpr, _ = roc_curve(y_test, positive_class_proba)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color='darkorange', lw=2, 
                   label=f'ROC curve (AUC = {roc_auc:.2f})')
        else:
            # For multi-class, binarize the output and compute ROC for each class
            # Use the actual range of classes present in the test set
            min_class = int(min(unique_test_classes))
            max_class = int(max(unique_test_classes))
            class_range = list(range(min_class, max_class + 1))
            
            # Ensure we don't exceed the number of classes the model can predict
            class_range = class_range[:n_model_classes]
            
            try:
                y_test_bin = label_binarize(y_test, classes=class_range)
            except Exception as e:
                logger.error(f"Failed to binarize labels for {title}: {str(e)}")
                logger.debug(f"y_test unique values: {unique_test_classes}")
                logger.debug(f"class_range: {class_range}")
                return False
            
            # Handle case where y_test_bin might be 1D (only one class present)
            if y_test_bin.ndim == 1:
                y_test_bin = y_test_bin.reshape(-1, 1)
            
            # Ensure we don't try to access more classes than available
            n_classes_to_plot = min(y_test_bin.shape[1], n_model_classes, len(class_range))
            
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i in range(n_classes_to_plot):
                try:
                    # Check for NaN values in this class's probabilities
                    class_proba = y_proba[:, i]
                    if np.isnan(class_proba).any():
                        logger.warning(f"NaN values found in class {i} probabilities for {title}, skipping this class")
                        continue
                    
                    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], class_proba)
                    roc_auc[i] = auc(fpr[i], tpr[i])
                    
                    # Use the actual class label if available, otherwise use index
                    class_label = class_labels[i] if i < len(class_labels) else f"Class {i}"
                    
                    # Plot ROC curve for each class
                    ax.plot(fpr[i], tpr[i], lw=2, 
                           label=f'{class_label} (AUC = {roc_auc[i]:.2f})')
                except Exception as e:
                    logger.warning(f"Failed to compute ROC for class {i} in {title}: {str(e)}")
                    continue
            
            # Compute micro-average ROC curve and ROC area if we have multiple classes
            if y_test_bin.shape[1] > 1 and len(roc_auc) > 1:
                try:
                    # Only use the classes we successfully computed ROC for
                    y_test_bin_subset = y_test_bin[:, :n_classes_to_plot]
                    y_proba_subset = y_proba[:, :n_classes_to_plot]
                    
                    # Check for NaN values in the flattened arrays
                    if not (np.isnan(y_test_bin_subset.ravel()).any() or np.isnan(y_proba_subset.ravel()).any()):
                        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin_subset.ravel(), y_proba_subset.ravel())
                        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                        
                        # Plot micro-average ROC curve
                        ax.plot(fpr["micro"], tpr["micro"], 
                               color='deeppink', linestyle=':', linewidth=4,
                               label=f'Micro-avg (AUC = {roc_auc["micro"]:.2f})')
                except Exception as e:
                    logger.warning(f"Failed to compute micro-average ROC for {title}: {str(e)}")
        
        # Check if we have any curves to plot
        if len(ax.lines) == 0:
            logger.error(f"No ROC curves could be generated for {title}")
            plt.close(fig)
            return False
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.8)
        
        # Set labels and title
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc="lower right")
        
        # Save and close
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)  # Explicitly close the figure
        
        success = verify_plot_exists(out_path)
        if not success:
            logger.error(f"Multi-class ROC curve plot file was not created or is empty: {out_path}")
        return success
    except Exception as e:
        logger.error(f"Error creating multi-class ROC curve for {title}: {str(e)}")
        logger.debug(f"Multi-class ROC curve error details: {repr(e)}")
        import traceback
        logger.debug(f"Multi-class ROC curve traceback:\n{traceback.format_exc()}")
        plt.close('all')  # Ensure all figures are closed on error
        return False

def plot_feature_importance(model, feature_names, title, out_path, top_n=20):
    """
    Plot feature importances for a trained model.
    Supports tree-based models (feature_importances_) and linear models (coef_).
    """
    try:
        # Validate inputs
        if model is None:
            logger.error(f"Model is None for {title}")
            return False
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        # Try to get feature importances
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
            if importances.ndim > 1:
                importances = np.sum(importances, axis=0)
        else:
            logger.warning(f"Model does not have feature importances or coefficients: {type(model)}")
            return False
        
        # If feature_names is None or too short, use generic names
        if feature_names is None or len(feature_names) != len(importances):
            feature_names = [f"Feature {i}" for i in range(len(importances))]
        
        # Get top N features
        top_n = min(top_n, len(importances))
        indices = np.argsort(importances)[-top_n:][::-1]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]
        
        # Create a new figure for each plot
        fig = plt.figure(figsize=(7, max(4, int(top_n/2))))
        ax = fig.add_subplot(111)
        
        # Plot
        sns.barplot(x=top_importances, y=top_features, ax=ax, orient='h')
        ax.set_title(title)
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        
        # Save and close
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)  # Explicitly close the figure
        
        success = verify_plot_exists(out_path)
        if not success:
            logger.error(f"Feature importance plot file was not created or is empty: {out_path}")
        return success
    except Exception as e:
        logger.error(f"Error creating feature importance plot for {title}: {str(e)}")
        logger.debug(f"Feature importance error details: {repr(e)}")
        import traceback
        logger.debug(f"Feature importance traceback:\n{traceback.format_exc()}")
        plt.close('all')  # Ensure all figures are closed on error
        return False 