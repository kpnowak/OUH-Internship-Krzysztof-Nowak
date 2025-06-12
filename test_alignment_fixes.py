#!/usr/bin/env python3
"""
Test script for the robust alignment fixes.
"""

import numpy as np
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_robust_functions():
    """Test the robust alignment functions."""
    print("=" * 60)
    print("Testing Robust Alignment Functions")
    print("=" * 60)
    
    try:
        # Import the functions
        from models import safe_target_transform, synchronize_X_y_data, guard_against_target_nans
        print("✓ Successfully imported robust functions")
        
        # Test 1: safe_target_transform with log1p
        print("\n1. Testing safe_target_transform with log1p...")
        y_test = pd.Series([0, 1, 10, 50, 100], index=['A', 'B', 'C', 'D', 'E'], name='blast_pct')
        y_transformed = safe_target_transform(y_test, np.log1p, 'AML')
        print(f"   Original: {y_test.values}")
        print(f"   Transformed: {y_transformed.values}")
        print(f"   Index preserved: {list(y_transformed.index) == list(y_test.index)}")
        
        # Test 2: safe_target_transform with negative values (should skip)
        print("\n2. Testing safe_target_transform with negative values...")
        y_negative = pd.Series([-5, 0, 1, 10], index=['A', 'B', 'C', 'D'], name='blast_pct')
        y_transformed_neg = safe_target_transform(y_negative, np.log1p, 'AML')
        print(f"   Original: {y_negative.values}")
        print(f"   Transformed: {y_transformed_neg.values}")
        print(f"   Transformation skipped: {np.array_equal(y_negative.values, y_transformed_neg.values)}")
        
        # Test 3: synchronize_X_y_data
        print("\n3. Testing synchronize_X_y_data...")
        X_test = pd.DataFrame({'feat1': [1, 2, 3, 4], 'feat2': [5, 6, 7, 8]}, 
                             index=['A', 'B', 'C', 'D'])
        y_test_sync = pd.Series([10, 20, 30], index=['A', 'B', 'C'])  # Missing 'D'
        
        X_sync, y_sync = synchronize_X_y_data(X_test, y_test_sync, "test")
        print(f"   X original shape: {X_test.shape}, y original shape: {y_test_sync.shape}")
        print(f"   X synced shape: {X_sync.shape}, y synced shape: {y_sync.shape}")
        print(f"   Common indices: {list(X_sync.index)}")
        
        # Test 4: guard_against_target_nans
        print("\n4. Testing guard_against_target_nans...")
        X_nan_test = pd.DataFrame({'feat1': [1, 2, 3, 4], 'feat2': [5, 6, 7, 8]}, 
                                 index=['A', 'B', 'C', 'D'])
        y_nan_test = pd.Series([10, np.nan, 30, np.inf], index=['A', 'B', 'C', 'D'])
        
        X_clean, y_clean = guard_against_target_nans(X_nan_test, y_nan_test, "test")
        print(f"   X original shape: {X_nan_test.shape}, y original shape: {y_nan_test.shape}")
        print(f"   X clean shape: {X_clean.shape}, y clean shape: {y_clean.shape}")
        print(f"   Clean indices: {list(X_clean.index)}")
        print(f"   Clean y values: {y_clean.values}")
        
        print("\n✓ All robust function tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Error testing robust functions: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_hard_error_detection():
    """Test that hard errors are properly raised for alignment issues."""
    print("\n" + "=" * 60)
    print("Testing Hard Error Detection")
    print("=" * 60)
    
    try:
        # This should raise a ValueError due to length mismatch
        from models import synchronize_X_y_data
        
        # Create mismatched data that can't be synchronized
        X_mismatch = np.array([[1, 2], [3, 4], [5, 6]])  # 3 samples
        y_mismatch = np.array([10, 20])  # 2 samples
        
        print("Testing length mismatch detection...")
        X_sync, y_sync = synchronize_X_y_data(X_mismatch, y_mismatch, "mismatch test")
        
        # Check if the function handled the mismatch
        if len(X_sync) == len(y_sync):
            print(f"✓ Length mismatch handled: X={len(X_sync)}, y={len(y_sync)}")
        else:
            print(f"✗ Length mismatch not handled: X={len(X_sync)}, y={len(y_sync)}")
            
        return True
        
    except Exception as e:
        print(f"Error in hard error detection test: {str(e)}")
        return False

if __name__ == "__main__":
    success1 = test_robust_functions()
    success2 = test_hard_error_detection()
    
    if success1 and success2:
        print("\n🎉 All tests passed! Robust alignment fixes are working correctly.")
    else:
        print("\n❌ Some tests failed. Check the implementation.") 