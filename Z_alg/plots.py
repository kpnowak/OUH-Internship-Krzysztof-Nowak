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
            
        # Get probability predictions - handle case where only one class is predicted
        y_proba_full = model.predict_proba(X_test)
        
        # Check if we have probabilities for both classes
        if y_proba_full.shape[1] < 2:
            logger.warning(f"Model only predicts one class for {title}, cannot create ROC curve")
            logger.debug(f"predict_proba shape: {y_proba_full.shape}, unique classes in y_test: {np.unique(y_test)}")
            return False
        
        # Get probability predictions for positive class
        y_proba = y_proba_full[:, 1]
        
        # Validate that we have both classes in the test set
        unique_classes = np.unique(y_test)
        if len(unique_classes) < 2:
            logger.warning(f"Test set only contains one class for {title}: {unique_classes}, cannot create ROC curve")
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
            
        # Get probability predictions for all classes
        y_proba = model.predict_proba(X_test)
        n_classes = len(class_labels)
        
        # Validate that we have the right number of classes
        if y_proba.shape[1] != n_classes:
            logger.error(f"Mismatch between predicted probabilities shape {y_proba.shape} and class labels {n_classes} for {title}")
            return False
        
        # Create a new figure for each plot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        
        if n_classes == 2:
            # For binary classification, use the standard binary ROC
            # Additional check to ensure we have probabilities for both classes
            if y_proba.shape[1] < 2:
                logger.warning(f"Binary classification in multiclass function only has {y_proba.shape[1]} probability columns for {title}")
                return False
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color='darkorange', lw=2, 
                   label=f'ROC curve (AUC = {roc_auc:.2f})')
        else:
            # For multi-class, binarize the output and compute ROC for each class
            y_test_bin = label_binarize(y_test, classes=range(n_classes))
            
            # Handle case where y_test_bin might be 1D (only one class present)
            if y_test_bin.ndim == 1:
                y_test_bin = y_test_bin.reshape(-1, 1)
            
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i in range(n_classes):
                if i < y_test_bin.shape[1]:
                    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                    
                    # Plot ROC curve for each class
                    ax.plot(fpr[i], tpr[i], lw=2, 
                           label=f'Class {class_labels[i]} (AUC = {roc_auc[i]:.2f})')
            
            # Compute micro-average ROC curve and ROC area if we have multiple classes
            if y_test_bin.shape[1] > 1:
                fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                
                # Plot micro-average ROC curve
                ax.plot(fpr["micro"], tpr["micro"], 
                       color='deeppink', linestyle=':', linewidth=4,
                       label=f'Micro-avg (AUC = {roc_auc["micro"]:.2f})')
        
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