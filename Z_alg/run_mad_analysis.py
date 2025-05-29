#!/usr/bin/env python3
"""
Simple script to run MAD analysis only.
"""

import logging
import time
from Z_alg.mad_analysis import run_mad_analysis

# Set up logging to file and console
log_file = "debug.log"

# Create file handler
file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s')
file_handler.setFormatter(file_formatter)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Set up root logger
logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, console_handler])

logger = logging.getLogger(__name__)

def log_timing_summary(start_time: float, operation_name: str = "MAD Analysis"):
    """Log timing summary for MAD analysis."""
    # Calculate total runtime
    total_elapsed_time = time.time() - start_time
    hours, remainder = divmod(total_elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Format times
    start_time_formatted = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    end_time_formatted = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    # Create timing summary
    timing_summary = (
        f"{operation_name} started: {start_time_formatted}\n"
        f"{operation_name} ended: {end_time_formatted}\n"
        f"Total runtime: {int(hours)}h {int(minutes)}m {int(seconds)}s"
    )
    
    # Log to debug.log
    logger.info("\n" + "=" * 70)
    logger.info(f"{operation_name.upper()} TIMING SUMMARY")
    logger.info("=" * 70)
    logger.info(timing_summary)
    logger.info("=" * 70)
    
    # Print to console
    print("\n" + "=" * 70)
    print(f"{operation_name.upper()} TIMING SUMMARY")
    print("=" * 70)
    print(timing_summary)
    print("=" * 70)

if __name__ == "__main__":
    # Record start time
    start_time = time.time()
    start_time_formatted = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    
    print("=" * 70)
    print("Running MAD Analysis...")
    print(f"Started at: {start_time_formatted}")
    print("=" * 70)
    
    logger.info("=" * 70)
    logger.info("MAD Analysis standalone execution started")
    logger.info(f"Started at: {start_time_formatted}")
    logger.info("=" * 70)
    
    try:
        run_mad_analysis(output_dir="output")
        logger.info("MAD Analysis completed successfully!")
        print("MAD Analysis completed successfully!")
        
        # Log timing summary
        log_timing_summary(start_time, "MAD Analysis")
        
    except Exception as e:
        logger.error(f"Error in MAD analysis: {str(e)}")
        print(f"Error in MAD analysis: {str(e)}")
        import traceback
        logger.debug(f"MAD analysis traceback:\n{traceback.format_exc()}")
        traceback.print_exc()
        
        # Still log timing even if there was an error
        log_timing_summary(start_time, "MAD Analysis (with errors)") 