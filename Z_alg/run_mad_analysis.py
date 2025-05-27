#!/usr/bin/env python3
"""
Simple script to run MAD analysis only.
"""

import logging
from Z_alg.mad_analysis import run_mad_analysis

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    print("Running MAD Analysis...")
    try:
        run_mad_analysis(output_dir="output")
        print("MAD Analysis completed successfully!")
    except Exception as e:
        print(f"Error in MAD analysis: {str(e)}")
        import traceback
        traceback.print_exc() 