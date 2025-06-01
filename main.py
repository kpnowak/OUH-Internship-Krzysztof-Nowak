#!/usr/bin/env python3
"""
Main entry point for the multi-modal machine learning pipeline.
"""

# Configure matplotlib backend before any other imports to prevent tkinter errors in parallel processing
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cli import main
from utils import monitor_memory_usage, comprehensive_logger, memory_monitor, force_garbage_collection
from models import clear_all_caches, get_cache_stats
from config import CACHE_CONFIG, MEMORY_OPTIMIZATION
import threading
import time
import logging

logger = logging.getLogger(__name__)

def enhanced_cache_monitor(interval_seconds=600):  # Less frequent monitoring for high-memory server
    """
    Enhanced cache monitoring with automatic cleanup and comprehensive logging.
    
    Parameters
    ----------
    interval_seconds : int
        How often to check cache size (in seconds)
    """
    while True:
        try:
            # Sleep first to allow initial processing
            time.sleep(interval_seconds)
            
            # Check cache stats
            stats = get_cache_stats()
            total_mb = stats["total_memory_mb"]
            
            # Enhanced logging with detailed breakdown
            comprehensive_logger.log_cache_operation(
                "system", "monitor", "total_usage", 
                size_mb=total_mb
            )
            
            # Log individual cache usage
            for cache_type, cache_stats in stats.items():
                if isinstance(cache_stats, dict) and "memory_usage_mb" in cache_stats:
                    comprehensive_logger.logger.debug(
                        f"[CACHE] {cache_type}: {cache_stats['memory_usage_mb']:.2f} MB, "
                        f"hit_ratio: {cache_stats.get('hit_ratio', 0):.2f}"
                    )
            
            # Check against configured limits
            total_limit_mb = CACHE_CONFIG.get("total_limit_mb", 8000)
            auto_clear_threshold = MEMORY_OPTIMIZATION.get("auto_clear_threshold", 0.9)
            ## Check against configured limits for high-memory server
            #total_limit_mb = CACHE_CONFIG.get("total_limit_mb", 32000)
            #auto_clear_threshold = MEMORY_OPTIMIZATION.get("auto_clear_threshold", 0.85)
            
            if total_mb > (total_limit_mb * auto_clear_threshold):
                comprehensive_logger.logger.warning(
                    f"Cache memory usage high ({total_mb:.2f} MB / {total_limit_mb} MB limit), "
                    f"clearing caches"
                )
                
                # Clear caches and force garbage collection
                stats_before = clear_all_caches()
                force_garbage_collection()
                
                comprehensive_logger.logger.info(
                    f"Cleared {stats_before['total_memory_mb']:.2f} MB from caches"
                )
                
        except Exception as e:
            comprehensive_logger.log_error("cache_monitor", e)

def performance_summary_reporter(interval_seconds=1800):  # 30 minutes
    """
    Periodically report performance summary.
    
    Parameters
    ----------
    interval_seconds : int
        How often to report performance summary (in seconds)
    """
    while True:
        try:
            time.sleep(interval_seconds)
            
            # Get performance summary
            summary = comprehensive_logger.get_performance_summary()
            
            # Log summary (only if there are errors or significant performance issues)
            if summary['total_errors'] > 0:
                logger.warning("=== PERFORMANCE SUMMARY ===")
                logger.warning(f"Total errors: {summary['total_errors']}")
            
            # Only log slowest operations if they're significantly slow
            if summary['operation_times']:
                sorted_ops = sorted(
                    summary['operation_times'].items(),
                    key=lambda x: x[1]['mean'],
                    reverse=True
                )[:3]  # Reduced to top 3
                
                slow_ops = [op for op in sorted_ops if op[1]['mean'] > 10.0]  # Only if > 10 seconds
                if slow_ops:
                    logger.warning("Slow operations detected:")
                    for op_name, stats in slow_ops:
                        logger.warning(
                            f"  {op_name}: {stats['mean']:.3f}s avg "
                            f"({stats['count']} calls)"
                        )
                    
        except Exception as e:
            comprehensive_logger.log_error("performance_reporter", e)

if __name__ == "__main__":
    # Enhanced startup logging
    print("=== Multi-Omics Data Fusion Optimization Pipeline STARTING ===")
    comprehensive_logger.log_memory_usage("startup", force=True)
    
    # Start enhanced memory monitoring
    mem_monitor = monitor_memory_usage(
        interval_seconds=MEMORY_OPTIMIZATION.get("memory_monitor_interval", 60),
        log_threshold_percent=5
    )
    
    # Start enhanced cache monitoring thread
    cache_thread = threading.Thread(target=enhanced_cache_monitor, daemon=True)
    cache_thread.start()
    
    # Start performance summary reporter
    perf_thread = threading.Thread(target=performance_summary_reporter, daemon=True)
    perf_thread.start()
    
    # Set environment variable to enable resource logging
    os.environ["DEBUG_RESOURCES"] = "1"
    
    try:
        # Run the main pipeline with memory monitoring
        with memory_monitor("main_pipeline"):
            main()
            
        # Final performance summary
        print("=== PIPELINE COMPLETED ===")
        summary = comprehensive_logger.get_performance_summary()
        print(f"Total operations: {len(summary['operation_times'])}")
        print(f"Total errors: {summary['total_errors']}")
        
    except Exception as e:
        comprehensive_logger.log_error("main_pipeline", e)
        raise
    finally:
        # Final memory cleanup
        comprehensive_logger.log_memory_usage("shutdown", force=True)
        force_garbage_collection() 