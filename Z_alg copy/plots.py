#!/usr/bin/env python3
"""
Plotting module for visualizing results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import RocCurveDisplay
from typing import List, Optional, Union, Any
import matplotlib
matplotlib.use('Agg')

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
    except Exception:
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
        print(f"Warning: All values are NaN for {title}, skipping plot")
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
        print(f"Error creating scatter plot for {title}: {str(e)}")
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
        print(f"Warning: All values are NaN for {title}, skipping plot")
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
        print(f"Error creating residuals plot for {title}: {str(e)}")
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
        plt.savefig(out_path)
        plt.close(fig)  # Explicitly close the figure
        
        return verify_plot_exists(out_path)
    except Exception as e:
        print(f"Error creating confusion matrix for {title}: {str(e)}")
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
        # Convert y_test to numeric if it's string
        if isinstance(y_test[0], str):
            y_test = np.array([int(x) for x in y_test])
            
        # Get probability predictions for positive class
        y_proba = model.predict_proba(X_test)[:, 1]
        
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
        plt.savefig(out_path)
        plt.close(fig)  # Explicitly close the figure
        
        return verify_plot_exists(out_path)
    except Exception as e:
        print(f"Error creating ROC curve for {title}: {str(e)}")
        plt.close('all')  # Ensure all figures are closed on error
        return False 