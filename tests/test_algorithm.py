#!/usr/bin/env python3
"""
Test script to run the algorithm with small test data.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Configure matplotlib backend before any other imports
import matplotlib
matplotlib.use('Agg')

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the main CLI function
from cli import main

def test_with_small_dataset():
    """Test the algorithm with a small dataset."""
    print("=== TESTING ALGORITHM WITH SMALL DATASET ===")
    
    # Set up minimal test arguments
    original_argv = sys.argv.copy()
    
    try:
        # Test with regression dataset first
        print("\n1. Testing REGRESSION pipeline...")
        sys.argv = [
            'test_algorithm.py',
            '--dataset', 'test_data',  # Use test data directory
            '--task', 'regression',
            '--n_splits', '2',  # Minimal splits for speed
            '--test_size', '0.3',
            '--verbose', '1',
            '--output_dir', 'test_output_regression'
        ]
        
        start_time = time.time()
        try:
            main()
            regression_time = time.time() - start_time
            print(f"✓ Regression test completed in {regression_time:.2f} seconds")
        except Exception as e:
            print(f"✗ Regression test failed: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Test with classification dataset
        print("\n2. Testing CLASSIFICATION pipeline...")
        sys.argv = [
            'test_algorithm.py',
            '--dataset', 'test_data',  # Use test data directory
            '--task', 'classification',
            '--n_splits', '2',  # Minimal splits for speed
            '--test_size', '0.3',
            '--verbose', '1',
            '--output_dir', 'test_output_classification'
        ]
        
        start_time = time.time()
        try:
            main()
            classification_time = time.time() - start_time
            print(f"✓ Classification test completed in {classification_time:.2f} seconds")
        except Exception as e:
            print(f"✗ Classification test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            
    finally:
        # Restore original argv
        sys.argv = original_argv

if __name__ == "__main__":
    # Set up basic logging
    logging.basicConfig(level=logging.INFO)
    
    # Run the test
    test_with_small_dataset()
    
    print("\n=== TEST COMPLETED ===") 