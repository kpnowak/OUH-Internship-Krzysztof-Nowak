#!/usr/bin/env python3
"""
Tests for preprocessing functions.
"""

import pytest
import numpy as np
import pandas as pd
from Z_alg.preprocessing import (
    _keep_top_variable_rows,
    fix_tcga_id_slicing,
    custom_parse_outcome,
    safe_convert_to_numeric, 
    process_with_missing_modalities
)

def test_keep_top_variable_rows():
    """Test the function to keep top variable rows."""
    # Create test data with some high and low variance rows
    np.random.seed(42)
    df = pd.DataFrame({
        'A': np.random.normal(0, 10, 100),  # High variance
        'B': np.random.normal(0, 5, 100),   # Medium variance
        'C': np.random.normal(0, 1, 100),   # Low variance
        'D': np.random.normal(0, 20, 100),  # Highest variance
        'E': np.random.normal(0, 0.1, 100)  # Lowest variance
    }).T  # Transpose to get rows as features
    
    # Test keeping top 3 rows
    df_filtered = _keep_top_variable_rows(df, k=3)
    
    # Check that we have the right number of rows
    assert df_filtered.shape[0] == 3
    
    # Check that highest variance rows are kept
    # D should be kept (highest variance)
    assert 'D' in df_filtered.index
    # A should be kept (second highest variance)
    assert 'A' in df_filtered.index
    # E should not be kept (lowest variance)
    assert 'E' not in df_filtered.index

def test_fix_tcga_id_slicing():
    """Test TCGA ID standardization."""
    # Create a list of TCGA IDs with different formats
    test_ids = [
        "TCGA-AB-1234-01A-11R-1234-13",
        "TCGA-CD-5678",
        "TCGA-EF-9012-02",
        "non-tcga-id",
        "TCGA-GH-3456-01A-11"
    ]
    
    fixed_ids = fix_tcga_id_slicing(test_ids)
    
    # Check that we have the same number of IDs
    assert len(fixed_ids) == len(test_ids)
    
    # Check specific cases
    assert fixed_ids[0] == "TCGA-AB-1234"  # Should keep first 3 parts
    assert fixed_ids[1] == "TCGA-CD-5678"  # Already good format
    assert fixed_ids[2] == "TCGA-EF-9012"  # Should keep first 3 parts
    assert fixed_ids[3] == "non-tcga-id"   # Non-TCGA ID should be kept as is
    assert fixed_ids[4] == "TCGA-GH-3456"  # Should keep first 3 parts

def test_custom_parse_outcome():
    """Test parsing different outcome types."""
    # Test survival time parsing
    survival_series = pd.Series(['10.5', '20', '30.2', np.nan])
    result = custom_parse_outcome(survival_series, 'os')
    assert result.dtype == float
    assert result.iloc[0] == 10.5
    
    # Test response categorical parsing
    response_series = pd.Series(['Responder', 'Non-responder', 'Responder', 'Unknown'])
    result = custom_parse_outcome(response_series, 'response')
    assert result.dtype == np.int64
    # Check that unique values are encoded properly
    unique_values = set(result.dropna().tolist())
    assert len(unique_values) == 3
    
    # Test numeric response parsing
    numeric_series = pd.Series([1, 0, 1, 1, 0])
    result = custom_parse_outcome(numeric_series, 'response')
    assert result.dtype == 'Int64'
    assert result.iloc[0] == 1
    assert result.iloc[1] == 0

def test_safe_convert_to_numeric():
    """Test conversion to numeric arrays with error handling."""
    # Test DataFrame conversion
    df = pd.DataFrame({
        'A': [1, 2, 3, np.nan],
        'B': [4, 5, np.nan, 7]
    })
    result = safe_convert_to_numeric(df)
    assert isinstance(result, np.ndarray)
    assert result.shape == (4, 2)
    assert not np.isnan(result).any()  # NaNs should be replaced with 0
    
    # Test list conversion
    test_list = [1, 2, '3', '4.5', None]
    result = safe_convert_to_numeric(test_list)
    assert isinstance(result, np.ndarray)
    assert result.shape == (5,)
    assert not np.isnan(result).any()  # None should be converted to 0
    
    # Test numpy array conversion
    test_array = np.array([1.1, 2.2, np.nan, 4.4])
    result = safe_convert_to_numeric(test_array)
    assert isinstance(result, np.ndarray)
    assert result.shape == (4,)
    assert not np.isnan(result).any()  # NaN should be replaced with 0
    
    # Test error case
    class NonConvertible:
        pass
    obj = NonConvertible()
    result = safe_convert_to_numeric(obj)
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 1)  # Should return fallback array

def test_process_with_missing_modalities():
    """Test simulating missing data in modalities."""
    # Create test data with two modalities
    mod1 = pd.DataFrame({
        'sample1': [1, 2, 3],
        'sample2': [4, 5, 6],
        'sample3': [7, 8, 9],
        'sample4': [10, 11, 12]
    })
    mod2 = pd.DataFrame({
        'sample1': [13, 14, 15],
        'sample2': [16, 17, 18],
        'sample3': [19, 20, 21],
        'sample4': [22, 23, 24]
    })
    
    modalities = {'mod1': mod1, 'mod2': mod2}
    all_ids = ['sample1', 'sample2', 'sample3', 'sample4']
    
    # Test with 0% missing (should return copies of originals)
    result = process_with_missing_modalities(modalities, all_ids, 0.0, random_state=42)
    assert len(result) == len(modalities)
    assert result['mod1'].shape == mod1.shape
    assert result['mod2'].shape == mod2.shape
    
    # Test with 50% missing (should remove some samples from each modality)
    result = process_with_missing_modalities(modalities, all_ids, 0.5, random_state=42)
    assert len(result) == len(modalities)
    # Should have fewer columns than original
    assert result['mod1'].shape[1] < mod1.shape[1]
    assert result['mod2'].shape[1] < mod2.shape[1]
    # Should have removed 2 samples (50% of 4)
    assert result['mod1'].shape[1] == 2
    assert result['mod2'].shape[1] == 2 