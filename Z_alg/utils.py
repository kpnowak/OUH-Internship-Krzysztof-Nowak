#!/usr/bin/env python3
"""
Utilities module for threading, resource management, and common helpers.
"""

import os
import time
import psutil
import threading
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from contextlib import contextmanager
from typing import Dict, List, Optional, Union, Any
import logging

# Local imports
from Z_alg.config import OMP_BLAS_THREADS, N_JOBS, JOBLIB_PARALLEL_CONFIG

logger = logging.getLogger(__name__)

class ProgressCounter:
    """A thread-safe progress counter for parallel processing."""
    def __init__(self, total: int = 100):
        self.total = total
        self.count = 0
        self.lock = threading.Lock()
        
    def increment(self):
        """Increment the counter by 1 and return the new value."""
        with self.lock:
            self.count += 1
            return self.count
            
    def reset(self):
        """Reset the counter to 0."""
        with self.lock:
            self.count = 0
            
    def get_count(self) -> int:
        """Get the current count."""
        with self.lock:
            return self.count

def TParallel(*args, **kwargs):
    """
    Thread-backend Parallel that never spawns new processes.
    Designed to be safe in nested parallelism situations.
    """
    # Check if we're already in a parallel context
    try:
        from joblib.parallel import get_active_backend
        active_backend = get_active_backend()[0].backend
        if active_backend == "loky":
            # If we're in a process, use single thread
            kwargs["n_jobs"] = 1
    except Exception:
        pass
        
    # Update with default config
    all_kwargs = JOBLIB_PARALLEL_CONFIG.copy()
    all_kwargs.update(kwargs)
    
    # Always use threading backend
    return Parallel(*args, backend="threading", **all_kwargs)

@contextmanager
def heavy_cpu_section(num_threads: int = OMP_BLAS_THREADS):
    """
    Context manager to limit thread count in heavy CPU sections.
    
    Parameters
    ----------
    num_threads : int
        Number of threads to allow for BLAS operations
        
    Yields
    ------
    None
    """
    with threadpool_limits(limits=num_threads):
        yield

def log_resource_usage(stage: str):
    """
    Log current resource usage for diagnostic purposes.
    
    Parameters
    ----------
    stage : str
        Description of the current stage for the log message
    """
    if os.environ.get("DEBUG_RESOURCES", "0") == "1":
        mem_percent = psutil.virtual_memory().percent
        cpu_percent = psutil.cpu_percent(interval=1)
        logger.info(f"[INFO] {stage} - RAM used: {mem_percent}%, CPU: {cpu_percent}%")

def monitor_memory_usage(interval_seconds=60, log_threshold_percent=5):
    """
    Start a background thread to monitor memory usage.
    Logs warnings when memory usage increases significantly.
    
    Parameters
    ----------
    interval_seconds : int
        Interval between memory checks in seconds
    log_threshold_percent : int
        Minimum percent increase to trigger a log message
        
    Returns
    -------
    threading.Thread
        The monitoring thread object
    """
    import threading
    import time
    import psutil
    import gc
    
    def memory_monitor():
        last_mem_percent = psutil.virtual_memory().percent
        logger.info(f"[MEMORY] Starting memory monitor: current usage {last_mem_percent}%")
        
        while True:
            time.sleep(interval_seconds)
            
            # Get current memory usage
            current_mem = psutil.virtual_memory()
            current_mem_percent = current_mem.percent
            current_mem_used = current_mem.used / (1024 * 1024 * 1024)  # GB
            
            # Calculate change
            change = current_mem_percent - last_mem_percent
            
            if change > log_threshold_percent:
                # Significant increase, log a warning
                logger.warning(f"[MEMORY] Warning: Memory usage increased by {change:.1f}% "
                              f"to {current_mem_percent:.1f}% ({current_mem_used:.2f} GB)")
                
                # Suggest garbage collection
                collected = gc.collect()
                logger.info(f"[MEMORY] Forced garbage collection: {collected} objects collected")
                
                # Update after collection
                new_mem_percent = psutil.virtual_memory().percent
                logger.info(f"[MEMORY] After collection: {new_mem_percent:.1f}% "
                              f"(reduced by {current_mem_percent - new_mem_percent:.1f}%)")
            
            elif current_mem_percent > 85:  # Always warn if usage is very high
                logger.warning(f"[MEMORY] High memory usage: {current_mem_percent:.1f}% ({current_mem_used:.2f} GB)")
                
            # Update for next iteration
            last_mem_percent = new_mem_percent if 'new_mem_percent' in locals() else current_mem_percent
    
    # Create and start thread
    monitor_thread = threading.Thread(
        target=memory_monitor,
        daemon=True,  # Allow program to exit even if thread is running
        name="MemoryMonitor"
    )
    monitor_thread.start()
    return monitor_thread 