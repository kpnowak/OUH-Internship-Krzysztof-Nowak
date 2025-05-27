#!/usr/bin/env python3
"""
Enhanced utilities module with comprehensive logging and monitoring capabilities.
"""

import os
import time
import psutil
import threading
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from functools import wraps
from contextlib import contextmanager
import traceback
import gc

# Configure enhanced logging
logger = logging.getLogger(__name__)

class ComprehensiveLogger:
    """
    Comprehensive logging system with memory, performance, and operation tracking.
    """
    
    def __init__(self, name: str = "Z_alg", level: str = "WARNING"):
        """
        Initialize comprehensive logger.
        
        Parameters
        ----------
        name : str
            Logger name
        level : str
            Logging level (default: WARNING for less verbose output)
        """
        # Check environment variables for verbosity control
        if os.getenv("Z_ALG_VERBOSE", "").lower() in ("1", "true", "yes"):
            level = "INFO"
        elif os.getenv("Z_ALG_DEBUG", "").lower() in ("1", "true", "yes"):
            level = "DEBUG"
        elif os.getenv("Z_ALG_QUIET", "").lower() in ("1", "true", "yes"):
            level = "ERROR"
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Performance tracking
        self.operation_times = {}
        self.memory_snapshots = {}
        self.error_counts = {}
        
        # Memory monitoring
        self.process = psutil.Process()
        self.memory_threshold_mb = 2000  # Log when memory usage exceeds this (increased)
        self.last_memory_log = 0
        self.memory_log_interval = 500  # Only log if memory changes by 500MB
        
        # Setup enhanced formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
        )
        
        # Console handler with enhanced formatting
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def log_memory_usage(self, operation: str = "general", force: bool = False):
        """
        Log current memory usage.
        
        Parameters
        ----------
        operation : str
            Operation being performed
        force : bool
            Force logging even if threshold not met
        """
        try:
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            # Log if significant change or forced
            if force or memory_mb > self.memory_threshold_mb or \
               abs(memory_mb - self.last_memory_log) > self.memory_log_interval:
                
                self.logger.debug(f"[MEMORY] {operation}: {memory_mb:.1f} MB RSS, "
                               f"{memory_info.vms / 1024 / 1024:.1f} MB VMS")
                self.last_memory_log = memory_mb
                
                # Store snapshot
                self.memory_snapshots[operation] = {
                    'timestamp': time.time(),
                    'rss_mb': memory_mb,
                    'vms_mb': memory_info.vms / 1024 / 1024
                }
                
        except Exception as e:
            self.logger.warning(f"Failed to log memory usage: {str(e)}")
    
    def log_performance(self, operation: str, duration: float, **kwargs):
        """
        Log performance metrics.
        
        Parameters
        ----------
        operation : str
            Operation name
        duration : float
            Duration in seconds
        **kwargs
            Additional metrics
        """
        # Store timing
        if operation not in self.operation_times:
            self.operation_times[operation] = []
        self.operation_times[operation].append(duration)
        
        # Log performance
        metrics_str = f"Duration: {duration:.3f}s"
        for key, value in kwargs.items():
            metrics_str += f", {key}: {value}"
        
        self.logger.debug(f"[PERFORMANCE] {operation}: {metrics_str}")
    
    def log_shape_mismatch(self, operation: str, X_shape: tuple, y_shape: tuple, 
                          action: str, result_shape: tuple = None):
        """
        Log shape mismatch fixes.
        
        Parameters
        ----------
        operation : str
            Operation where mismatch occurred
        X_shape : tuple
            Original X shape
        y_shape : tuple
            Original y shape
        action : str
            Action taken to fix
        result_shape : tuple, optional
            Resulting shape after fix
        """
        msg = f"[SHAPE_FIX] {operation}: X{X_shape} vs y{y_shape} -> {action}"
        if result_shape:
            msg += f" -> {result_shape}"
        self.logger.info(msg)
    
    def log_cache_operation(self, cache_type: str, operation: str, key: str, 
                           hit: bool = None, size_mb: float = None):
        """
        Log cache operations.
        
        Parameters
        ----------
        cache_type : str
            Type of cache
        operation : str
            Operation (get, put, evict)
        key : str
            Cache key
        hit : bool, optional
            Whether it was a cache hit
        size_mb : float, optional
            Size in MB
        """
        msg = f"[CACHE] {cache_type}.{operation}: {key[:50]}..."
        if hit is not None:
            msg += f" {'HIT' if hit else 'MISS'}"
        if size_mb is not None:
            msg += f" ({size_mb:.2f} MB)"
        self.logger.debug(msg)
    
    def log_feature_selection(self, modality: str, method: str, 
                             original_features: int, selected_features: int,
                             fold_idx: int = None):
        """
        Log feature selection details.
        
        Parameters
        ----------
        modality : str
            Modality name
        method : str
            Selection method
        original_features : int
            Original number of features
        selected_features : int
            Selected number of features
        fold_idx : int, optional
            Fold index
        """
        fold_str = f" (fold {fold_idx})" if fold_idx is not None else ""
        reduction_pct = 100 * (1 - selected_features / original_features) if original_features > 0 else 0
        
        self.logger.debug(f"[FEATURE_SELECTION] {modality}{fold_str}: {method} "
                        f"{original_features} -> {selected_features} features "
                        f"({reduction_pct:.1f}% reduction)")
    
    def log_extractor_operation(self, modality: str, extractor_type: str,
                               input_shape: tuple, output_shape: tuple,
                               fold_idx: int = None):
        """
        Log extractor operations.
        
        Parameters
        ----------
        modality : str
            Modality name
        extractor_type : str
            Type of extractor
        input_shape : tuple
            Input data shape
        output_shape : tuple
            Output data shape
        fold_idx : int, optional
            Fold index
        """
        fold_str = f" (fold {fold_idx})" if fold_idx is not None else ""
        
        self.logger.debug(f"[EXTRACTOR] {modality}{fold_str}: {extractor_type} "
                        f"{input_shape} -> {output_shape}")
    
    def log_model_training(self, model_name: str, dataset: str, 
                          train_samples: int, val_samples: int,
                          early_stopping_epoch: int = None,
                          best_score: float = None, fold_idx: int = None):
        """
        Log model training details.
        
        Parameters
        ----------
        model_name : str
            Model name
        dataset : str
            Dataset name
        train_samples : int
            Number of training samples
        val_samples : int
            Number of validation samples
        early_stopping_epoch : int, optional
            Epoch where early stopping occurred
        best_score : float, optional
            Best validation score
        fold_idx : int, optional
            Fold index
        """
        fold_str = f" (fold {fold_idx})" if fold_idx is not None else ""
        msg = f"[MODEL_TRAINING] {dataset}{fold_str}: {model_name} "
        msg += f"train={train_samples}, val={val_samples}"
        
        if early_stopping_epoch is not None:
            msg += f", early_stop@{early_stopping_epoch}"
        if best_score is not None:
            msg += f", best_score={best_score:.4f}"
            
        self.logger.info(msg)
    
    def log_error(self, operation: str, error: Exception, **context):
        """
        Log errors with context.
        
        Parameters
        ----------
        operation : str
            Operation where error occurred
        error : Exception
            The exception
        **context
            Additional context
        """
        # Count errors
        if operation not in self.error_counts:
            self.error_counts[operation] = 0
        self.error_counts[operation] += 1
        
        # Log error with context
        context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
        self.logger.error(f"[ERROR] {operation}: {str(error)} | Context: {context_str}")
        self.logger.debug(f"[ERROR] {operation} traceback:\n{traceback.format_exc()}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary.
        
        Returns
        -------
        Dict[str, Any]
            Performance summary
        """
        summary = {
            'operation_times': {},
            'memory_snapshots': self.memory_snapshots,
            'error_counts': self.error_counts,
            'total_errors': sum(self.error_counts.values())
        }
        
        # Calculate timing statistics
        for operation, times in self.operation_times.items():
            if times:
                summary['operation_times'][operation] = {
                    'count': len(times),
                    'total': sum(times),
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': min(times),
                    'max': max(times)
                }
        
        return summary

# Global logger instance
comprehensive_logger = ComprehensiveLogger()

def performance_monitor(operation_name: str = None):
    """
    Decorator to monitor performance of functions.
    
    Parameters
    ----------
    operation_name : str, optional
        Custom operation name
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            # Log memory before
            comprehensive_logger.log_memory_usage(f"{op_name}_start")
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Log successful completion
                comprehensive_logger.log_performance(op_name, duration)
                comprehensive_logger.log_memory_usage(f"{op_name}_end")
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                comprehensive_logger.log_error(op_name, e, duration=duration)
                raise
                
        return wrapper
    return decorator

@contextmanager
def memory_monitor(operation: str, log_threshold_mb: float = 100):
    """
    Context manager to monitor memory usage during operations.
    
    Parameters
    ----------
    operation : str
        Operation name
    log_threshold_mb : float
        Log if memory increase exceeds this threshold
    """
    # Get initial memory
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024
    
    comprehensive_logger.log_memory_usage(f"{operation}_start", force=True)
    
    try:
        yield
    finally:
        # Get final memory
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        if abs(memory_increase) > log_threshold_mb:
            comprehensive_logger.logger.warning(
                f"[MEMORY] {operation}: Memory change {memory_increase:+.1f} MB "
                f"({initial_memory:.1f} -> {final_memory:.1f} MB)"
            )
        
        comprehensive_logger.log_memory_usage(f"{operation}_end", force=True)

def monitor_memory_usage(interval_seconds: int = 60, log_threshold_percent: float = 5):
    """
    Start background memory monitoring.
    
    Parameters
    ----------
    interval_seconds : int
        Monitoring interval in seconds
    log_threshold_percent : float
        Log when memory usage changes by this percentage
    
    Returns
    -------
    threading.Thread
        The monitoring thread
    """
    def monitor():
        last_memory = 0
        while True:
            try:
                process = psutil.Process()
                current_memory = process.memory_info().rss / 1024 / 1024
                
                if last_memory > 0:
                    change_percent = abs(current_memory - last_memory) / last_memory * 100
                    if change_percent > log_threshold_percent:
                        comprehensive_logger.logger.info(
                            f"[MEMORY_MONITOR] Memory: {current_memory:.1f} MB "
                            f"({change_percent:+.1f}% change)"
                        )
                
                last_memory = current_memory
                time.sleep(interval_seconds)
                
            except Exception as e:
                comprehensive_logger.logger.warning(f"Memory monitoring error: {str(e)}")
                time.sleep(interval_seconds)
    
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    return thread

def log_dataset_info(dataset_name: str, modalities: Dict[str, Any], 
                    outcome_shape: tuple, common_samples: int):
    """
    Log dataset information.
    
    Parameters
    ----------
    dataset_name : str
        Dataset name
    modalities : Dict[str, Any]
        Modality information
    outcome_shape : tuple
        Outcome data shape
    common_samples : int
        Number of common samples
    """
    comprehensive_logger.logger.info(f"=== DATASET INFO: {dataset_name} ===")
    comprehensive_logger.logger.info(f"Common samples: {common_samples}")
    comprehensive_logger.logger.info(f"Outcome shape: {outcome_shape}")
    
    for mod_name, mod_data in modalities.items():
        if hasattr(mod_data, 'shape'):
            comprehensive_logger.logger.info(f"{mod_name}: {mod_data.shape}")
        else:
            comprehensive_logger.logger.info(f"{mod_name}: {type(mod_data)}")

def log_cv_fold_info(fold_idx: int, train_size: int, val_size: int, test_size: int = None):
    """
    Log cross-validation fold information.
    
    Parameters
    ----------
    fold_idx : int
        Fold index
    train_size : int
        Training set size
    val_size : int
        Validation set size
    test_size : int, optional
        Test set size
    """
    msg = f"[CV_FOLD] Fold {fold_idx}: train={train_size}, val={val_size}"
    if test_size is not None:
        msg += f", test={test_size}"
    comprehensive_logger.logger.info(msg)

def force_garbage_collection():
    """Force garbage collection and log memory freed."""
    initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    # Force garbage collection
    collected = gc.collect()
    
    final_memory = psutil.Process().memory_info().rss / 1024 / 1024
    memory_freed = initial_memory - final_memory
    
    if memory_freed > 1:  # Only log if significant memory was freed
        comprehensive_logger.logger.info(
            f"[GC] Collected {collected} objects, freed {memory_freed:.1f} MB"
        )



# Export the comprehensive logger for use in other modules
__all__ = [
    'comprehensive_logger', 'performance_monitor', 'memory_monitor',
    'monitor_memory_usage', 'log_dataset_info', 'log_cv_fold_info',
    'force_garbage_collection'
] 