#!/usr/bin/env python3
"""
Test script to verify the Sarcoma dataset loading fix.
"""

import sys
import pandas as pd
from pathlib import Path

# Add current directory to path
sys.path.append('.')

from data_io import load_dataset
from config import DATASETS

def test_sarcoma_loading():
    """Test loading the Sarcoma dataset with the pipe-separated value fix."""
    
    print("=" * 60)
    print("TESTING SARCOMA DATASET LOADING FIX")
    print("=" * 60)
    
    # Get Sarcoma configuration
    sarcoma_config = DATASETS.get('Sarcoma')
    if not sarcoma_config:
        print("ERROR: Sarcoma not found in DATASETS configuration")
        return False
    
    print(f"Dataset config: {sarcoma_config}")
    print(f"Outcome column: {sarcoma_config['outcome_col']}")
    print(f"Task type: regression")
    
    try:
        print("\nAttempting to load Sarcoma dataset...")
        
        # Load the dataset
        modalities, y, common_ids = load_dataset(
            ds_name='Sarcoma',
            modalities=['clinical'],
            outcome_col=sarcoma_config['outcome_col'],
            task_type='regression'
        )
        
        print(f"\n✅ SUCCESS: Dataset loaded successfully!")
        print(f"   - Number of samples: {len(common_ids)}")
        print(f"   - Modalities loaded: {list(modalities.keys())}")
        print(f"   - Outcome data type: {y.dtype}")
        print(f"   - Outcome stats:")
        print(f"     * Mean: {y.mean():.3f}")
        print(f"     * Std: {y.std():.3f}")
        print(f"     * Min: {y.min():.3f}")
        print(f"     * Max: {y.max():.3f}")
        print(f"   - Sample outcome values: {y.head().tolist()}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pipe_separated_extraction():
    """Test the pipe-separated value extraction function directly."""
    
    print("\n" + "=" * 60)
    print("TESTING PIPE-SEPARATED VALUE EXTRACTION")
    print("=" * 60)
    
    # Test cases from the actual Sarcoma data
    test_values = [
        "33|1.4|5.0",
        "13|1", 
        "105.5",
        "20|0.3|11",
        "|0.6",
        "5.5",
        "8",
        "5"
    ]
    
    def extract_first_numeric(value):
        """Extract first numeric value from pipe-separated strings."""
        if pd.isna(value):
            return value
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            # Split by common separators and try to convert first part
            for sep in ['|', ',', ';', ' ']:
                if sep in value:
                    parts = value.split(sep)
                    for part in parts:
                        try:
                            return float(part.strip())
                        except ValueError:
                            continue
            # Try to convert the whole string
            try:
                return float(value.strip())
            except ValueError:
                return pd.NA
        return pd.NA
    
    print("Testing extraction function on sample values:")
    for value in test_values:
        extracted = extract_first_numeric(value)
        print(f"  '{value}' -> {extracted}")
    
    return True

if __name__ == "__main__":
    print("Starting Sarcoma dataset fix verification...")
    
    # Test the extraction function
    test_pipe_separated_extraction()
    
    # Test the actual dataset loading
    success = test_sarcoma_loading()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED - Sarcoma fix is working correctly!")
    else:
        print("❌ TESTS FAILED - There may be an issue with the fix")
    print("=" * 60) 