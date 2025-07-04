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

# Import validation and error handling
import logging
import threading
import time

# Configure basic logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for imported modules
cli_main = None
comprehensive_logger = None
memory_monitor = None
suppress_sklearn_warnings = None
force_garbage_collection = None
monitor_memory_usage = None
MEMORY_OPTIMIZATION = None

def validate_imports():
    """Validate that all required imports are available."""
    global cli_main, comprehensive_logger, memory_monitor
    global suppress_sklearn_warnings, force_garbage_collection, monitor_memory_usage, MEMORY_OPTIMIZATION
    
    try:
        # Test critical imports
        from cli import main as cli_main_import
        from utils import (
            monitor_memory_usage as monitor_memory_usage_import, 
            comprehensive_logger as comprehensive_logger_import, 
            memory_monitor as memory_monitor_import, 
            force_garbage_collection as force_garbage_collection_import, 
            suppress_sklearn_warnings as suppress_sklearn_warnings_import
        )
        from config import CACHE_CONFIG, MEMORY_OPTIMIZATION as MEMORY_OPTIMIZATION_import
        
        # Assign to global variables
        cli_main = cli_main_import
        comprehensive_logger = comprehensive_logger_import
        memory_monitor = memory_monitor_import
        suppress_sklearn_warnings = suppress_sklearn_warnings_import
        force_garbage_collection = force_garbage_collection_import
        monitor_memory_usage = monitor_memory_usage_import
        MEMORY_OPTIMIZATION = MEMORY_OPTIMIZATION_import
        
        logger.info("All critical imports validated successfully")
        return True
        
    except ImportError as e:
        logger.error(f"Critical import failed: {e}")
        logger.error("Pipeline cannot start without required modules")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during import validation: {e}")
        return False

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
            
            # Cache monitoring temporarily disabled
            # TODO: Re-enable when cache functions are available
            # stats = get_cache_stats()
            # total_mb = stats["total_memory_mb"]
            pass
                
        except Exception as e:
            try:
                if comprehensive_logger:
                    comprehensive_logger.log_error("cache_monitor", e)
                else:
                    logger.error(f"Cache monitor error: {e}")
            except:
                logger.error(f"Cache monitor error: {e}")

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
            try:
                if comprehensive_logger:
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
            except Exception as summary_error:
                logger.debug(f"Performance summary generation failed: {summary_error}")
                    
        except Exception as e:
            try:
                if comprehensive_logger:
                    comprehensive_logger.log_error("performance_reporter", e)
                else:
                    logger.error(f"Performance reporter error: {e}")
            except:
                logger.error(f"Performance reporter error: {e}")

def setup_environment():
    """Setup environment variables and configurations."""
    try:
        # Set environment variable to enable resource logging
        os.environ["DEBUG_RESOURCES"] = "1"
        
        # Set Octave executable path for oct2py
        # laptop 1 - my
        octave_path = os.environ["OCTAVE_EXECUTABLE"] = r"C:\Users\krzys\AppData\Local\Programs\GNU Octave\Octave-10.2.0\mingw64\bin\octave-cli.exe"
        # Laptop 2 - Tata
        #octave_path = r"C:\Program Files\GNU Octave\Octave-10.2.0\mingw64\bin\octave-cli.exe"
        if os.path.exists(octave_path):
            os.environ["OCTAVE_EXECUTABLE"] = octave_path
            logger.info(f"Octave executable configured: {octave_path}")
        else:
            logger.warning(f"Octave executable not found at {octave_path}")
            logger.warning("Some fusion methods may not work properly")
        
        return True
    except Exception as e:
        logger.error(f"Environment setup failed: {e}")
        return False

def start_monitoring_threads():
    """Start monitoring threads with error handling."""
    threads_started = []
    
    try:
        # Start enhanced memory monitoring
        try:
            if monitor_memory_usage and MEMORY_OPTIMIZATION:
                mem_monitor = monitor_memory_usage(
                    interval_seconds=MEMORY_OPTIMIZATION.get("memory_monitor_interval", 60),
                    log_threshold_percent=5
                )
                logger.info("Memory monitoring started")
            else:
                logger.warning("Memory monitoring not available")
        except Exception as e:
            logger.warning(f"Memory monitoring failed to start: {e}")
        
        # Start enhanced cache monitoring thread
        try:
            cache_thread = threading.Thread(target=enhanced_cache_monitor, daemon=True)
            cache_thread.start()
            threads_started.append("cache_monitor")
            logger.info("Cache monitoring thread started")
        except Exception as e:
            logger.warning(f"Cache monitoring thread failed to start: {e}")
        
        # Start performance summary reporter
        try:
            perf_thread = threading.Thread(target=performance_summary_reporter, daemon=True)
            perf_thread.start()
            threads_started.append("performance_reporter")
            logger.info("Performance monitoring thread started")
        except Exception as e:
            logger.warning(f"Performance monitoring thread failed to start: {e}")
            
        return threads_started
        
    except Exception as e:
        logger.error(f"Failed to start monitoring threads: {e}")
        return threads_started

if __name__ == "__main__":
    logger.info("Starting Multi-Omics Data Fusion Optimization Pipeline...")
    
    # Step 1: Validate imports
    imports_valid = validate_imports()
    if not imports_valid:
        logger.error("Pipeline startup failed due to import errors")
        sys.exit(1)
    
    # Step 2: Setup environment
    if not setup_environment():
        logger.error("Pipeline startup failed due to environment setup errors")
        sys.exit(1)
    
    # Step 3: Suppress sklearn warnings early in pipeline startup
    try:
        if suppress_sklearn_warnings:
            suppress_sklearn_warnings()
            logger.info("Sklearn warnings suppressed")
        else:
            logger.warning("Sklearn warning suppression not available")
    except Exception as e:
        logger.warning(f"Failed to suppress sklearn warnings: {e}")
    
    # Step 4: Enhanced startup logging
    print("=== Multi-Omics Data Fusion Optimization Pipeline STARTING ===")
    try:
        if comprehensive_logger:
            comprehensive_logger.log_memory_usage("startup", force=True)
        else:
            logger.info("Initial memory logging not available")
    except Exception as e:
        logger.warning(f"Initial memory logging failed: {e}")
    
    # Step 5: Start monitoring threads
    active_threads = start_monitoring_threads()
    logger.info(f"Started {len(active_threads)} monitoring threads: {', '.join(active_threads)}")
    
    # Step 6: Run the main pipeline
    exit_code = 0
    try:
        logger.info("Starting main pipeline execution...")
        
        if not cli_main:
            raise RuntimeError("CLI main function not available")
        
        # Run the main pipeline with memory monitoring
        try:
            if memory_monitor:
                with memory_monitor("main_pipeline"):  # type: ignore
                    cli_main()  # Call the validated CLI main function
            else:
                logger.info("Running without memory monitoring")
                cli_main()  # Run without memory monitoring
        except Exception as memory_monitor_error:
            logger.warning(f"Memory monitoring failed, running without it: {memory_monitor_error}")
            cli_main()  # Run without memory monitoring as fallback
            
        # Final performance summary
        print("=== PIPELINE COMPLETED SUCCESSFULLY ===")
        try:
            if comprehensive_logger:
                summary = comprehensive_logger.get_performance_summary()
                print(f"Total operations: {len(summary['operation_times'])}")
                print(f"Total errors: {summary['total_errors']}")
                logger.info("Pipeline completed successfully")
            else:
                print("Pipeline completed (summary unavailable)")
        except Exception as summary_error:
            logger.warning(f"Could not generate final summary: {summary_error}")
            print("Pipeline completed (summary unavailable)")
        
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user (Ctrl+C)")
        print("Pipeline interrupted by user")
        exit_code = 130  # Standard exit code for Ctrl+C
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        try:
            if comprehensive_logger:
                comprehensive_logger.log_error("main_pipeline", e)
            else:
                logger.error("Could not log error to comprehensive logger")
        except:
            logger.error("Could not log error to comprehensive logger")
        
        # Print error details for user
        print(f"PIPELINE FAILED: {e}")
        print("Check the logs for detailed error information")
        exit_code = 1
        
    finally:
        # Step 7: Cleanup
        try:
            logger.info("Performing final cleanup...")
            if comprehensive_logger:
                comprehensive_logger.log_memory_usage("shutdown", force=True)
            if force_garbage_collection:
                force_garbage_collection()
            logger.info("Cleanup completed")
        except Exception as cleanup_error:
            logger.warning(f"Cleanup failed: {cleanup_error}")
        
        # Final status message
        if exit_code == 0:
            print("Pipeline execution completed successfully!")
        else:
            print(f"Pipeline execution failed with exit code {exit_code}")
        
        sys.exit(exit_code) 