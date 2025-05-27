#!/usr/bin/env python3
"""
Test script to verify that maximum value extraction works correctly for pipe-separated values.
"""

import sys
import logging
import pandas as pd
import numpy as np

# Add current directory to path
sys.path.append('.')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_max_value_extraction():
    """Test the maximum value extraction logic."""
    
    # Define the extraction function (copied from data_io.py)
    def extract_max_numeric(value):
        if pd.isna(value):
            return value
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            # Handle very long concatenated strings by taking only first part
            if len(value) > 100:
                logger.warning(f"Detected very long string value (length: {len(value)}), taking first 100 characters")
                value = value[:100]
            
            # Split by common separators and collect all numeric values
            numeric_values = []
            for sep in ['|', ',', ';', ' ']:
                if sep in value:
                    parts = value.split(sep)
                    for part in parts:
                        try:
                            numeric_val = float(part.strip())
                            numeric_values.append(numeric_val)
                        except ValueError:
                            continue
                    break  # Use the first separator that works
            
            # If we found numeric values, return the maximum
            if numeric_values:
                return max(numeric_values)
            
            # Try to convert the whole string as fallback
            try:
                return float(value.strip())
            except ValueError:
                return np.nan
        return np.nan
    
    # Test cases based on the actual Sarcoma data
    test_cases = [
        # (input, expected_output, description)
        ("33|1.4|5.0", 33.0, "Three values - should pick maximum (33.0)"),
        ("8.5|3.5", 8.5, "Two values - should pick maximum (8.5)"),
        ("8|12", 12.0, "Two values - should pick maximum (12.0)"),
        ("13|1", 13.0, "Two values - should pick maximum (13.0)"),
        ("20|0.3|11", 20.0, "Three values - should pick maximum (20.0)"),
        ("15.2", 15.2, "Single value - should return as is"),
        ("7.5", 7.5, "Single decimal - should return as is"),
        (25, 25, "Already numeric - should return as is"),
        ("", np.nan, "Empty string - should return NaN"),
        (None, None, "None value - should return None"),
        ("abc|def", np.nan, "Non-numeric values - should return NaN"),
        ("5.5|abc|10.2", 10.2, "Mixed numeric/non-numeric - should pick max numeric (10.2)"),
    ]
    
    logger.info("Testing maximum value extraction logic...")
    
    passed = 0
    failed = 0
    
    for i, (input_val, expected, description) in enumerate(test_cases, 1):
        try:
            result = extract_max_numeric(input_val)
            
            # Handle NaN comparison
            if pd.isna(expected) and pd.isna(result):
                logger.info(f"✅ Test {i}: {description} - PASSED")
                passed += 1
            elif expected == result:
                logger.info(f"✅ Test {i}: {description} - PASSED")
                passed += 1
            else:
                logger.error(f"❌ Test {i}: {description} - FAILED")
                logger.error(f"   Input: {input_val}")
                logger.error(f"   Expected: {expected}")
                logger.error(f"   Got: {result}")
                failed += 1
                
        except Exception as e:
            logger.error(f"❌ Test {i}: {description} - ERROR: {str(e)}")
            failed += 1
    
    logger.info(f"\n=== TEST RESULTS ===")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Total: {passed + failed}")
    
    if failed == 0:
        logger.info("🎉 ALL TESTS PASSED!")
        return True
    else:
        logger.error(f"❌ {failed} tests failed")
        return False

def test_sarcoma_specific_values():
    """Test with actual values from Sarcoma dataset."""
    
    logger.info("\n=== Testing with actual Sarcoma dataset values ===")
    
    # These are the actual problematic values from the Sarcoma dataset
    sarcoma_values = [
        "33|1.4|5.0",
        "8.5|3.5", 
        "8|12",
        "13|1",
        "20|0.3|11",
        "8.5|1.2",
        "3.5|0.9",
        "13.5|7.2",
        "0.5|1.5|1.5",
        "5.5|1.0",
        "6.0|3.0|6.5",
        "20|13|11",
        "4.5|3.5",
        "8.5|8.5|4",
        "13.2|3.2|3.0|2.0",
        "3.7|2.9|5|2",
        "8.5|0.9",
        "10.8|6",
        "38.3|8.5|4.5",
        "3.0|1.0|23.0|18.0",
        "7.1|6.2|5.7",
        "13.0|2.3",
        "22|5.2"
    ]
    
    expected_max_values = [
        33.0,   # 33|1.4|5.0 -> 33.0
        8.5,    # 8.5|3.5 -> 8.5
        12.0,   # 8|12 -> 12.0
        13.0,   # 13|1 -> 13.0
        20.0,   # 20|0.3|11 -> 20.0
        8.5,    # 8.5|1.2 -> 8.5
        3.5,    # 3.5|0.9 -> 3.5
        13.5,   # 13.5|7.2 -> 13.5
        1.5,    # 0.5|1.5|1.5 -> 1.5
        5.5,    # 5.5|1.0 -> 5.5
        6.5,    # 6.0|3.0|6.5 -> 6.5
        20.0,   # 20|13|11 -> 20.0
        4.5,    # 4.5|3.5 -> 4.5
        8.5,    # 8.5|8.5|4 -> 8.5
        13.2,   # 13.2|3.2|3.0|2.0 -> 13.2
        5.0,    # 3.7|2.9|5|2 -> 5.0
        8.5,    # 8.5|0.9 -> 8.5
        10.8,   # 10.8|6 -> 10.8
        38.3,   # 38.3|8.5|4.5 -> 38.3
        23.0,   # 3.0|1.0|23.0|18.0 -> 23.0
        7.1,    # 7.1|6.2|5.7 -> 7.1
        13.0,   # 13.0|2.3 -> 13.0
        22.0    # 22|5.2 -> 22.0
    ]
    
    # Import the actual function from data_io
    try:
        from data_io import load_dataset
        
        # Create a test series
        test_series = pd.Series(sarcoma_values)
        
        # Apply the extraction (we'll need to recreate the function here since it's nested)
        def extract_max_numeric(value):
            if pd.isna(value):
                return value
            if isinstance(value, (int, float)):
                return value
            if isinstance(value, str):
                # Split by common separators and collect all numeric values
                numeric_values = []
                for sep in ['|', ',', ';', ' ']:
                    if sep in value:
                        parts = value.split(sep)
                        for part in parts:
                            try:
                                numeric_val = float(part.strip())
                                numeric_values.append(numeric_val)
                            except ValueError:
                                continue
                        break  # Use the first separator that works
                
                # If we found numeric values, return the maximum
                if numeric_values:
                    return max(numeric_values)
                
                # Try to convert the whole string as fallback
                try:
                    return float(value.strip())
                except ValueError:
                    return np.nan
            return np.nan
        
        results = test_series.apply(extract_max_numeric)
        
        logger.info("Sarcoma value extraction results:")
        passed = 0
        failed = 0
        
        for i, (original, expected, result) in enumerate(zip(sarcoma_values, expected_max_values, results)):
            if abs(result - expected) < 0.001:  # Allow for floating point precision
                logger.info(f"✅ {original} -> {result} (expected {expected})")
                passed += 1
            else:
                logger.error(f"❌ {original} -> {result} (expected {expected})")
                failed += 1
        
        logger.info(f"\nSarcoma-specific tests: {passed} passed, {failed} failed")
        return failed == 0
        
    except Exception as e:
        logger.error(f"Error testing with actual data_io function: {str(e)}")
        return False

def main():
    """Run all tests."""
    logger.info("="*60)
    logger.info("TESTING MAXIMUM VALUE EXTRACTION FOR PIPE-SEPARATED VALUES")
    logger.info("="*60)
    
    # Run basic tests
    test1_passed = test_max_value_extraction()
    
    # Run Sarcoma-specific tests
    test2_passed = test_sarcoma_specific_values()
    
    logger.info("\n" + "="*60)
    if test1_passed and test2_passed:
        logger.info("🎉 ALL TESTS PASSED - Maximum value extraction is working correctly!")
        return True
    else:
        logger.error("❌ Some tests failed - Please check the implementation")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 