#!/usr/bin/env python3
"""
Main entry point for the multi-modal machine learning pipeline.
"""

from Z_alg.cli import main
from Z_alg.utils import monitor_memory_usage

if __name__ == "__main__":
    # Start memory monitoring in background
    mem_monitor = monitor_memory_usage(interval_seconds=60, log_threshold_percent=5)
    
    # Set environment variable to enable resource logging
    import os
    os.environ["DEBUG_RESOURCES"] = "1"
    
    # Run the main pipeline
    main() 