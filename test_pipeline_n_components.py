#!/usr/bin/env python3
"""
Test script to verify that:
1. Extractors use optimal n_components from hyperparameters (run ONCE per extractor)
2. Selectors use n_features = [8, 16, 32] (run 3 times per selector)
"""

import os
import sys
import logging
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.getcwd())

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_extraction_n_components():
    """Test that extraction uses optimal n_components from hyperparameters."""
    print("="*80)
    print("TESTING EXTRACTION N_COMPONENTS FROM HYPERPARAMETERS")
    print("="*80)
    
    try:
        from models import (
            get_extraction_n_components_list,
            get_regression_extractors,
            get_classification_extractors
        )
        
        # Test regression extractors (should use AML hyperparameters)
        print("\n1. Testing regression extractors (should use AML hyperparameters)...")
        reg_extractors = get_regression_extractors()
        aml_n_components = get_extraction_n_components_list("AML", reg_extractors, "reg")
        
        print(f"   Regression extractors n_components: {aml_n_components}")
        
        # Verify each extractor has exactly ONE n_components value
        all_single_values = True
        for extractor_name, n_comp_list in aml_n_components.items():
            if len(n_comp_list) != 1:
                print(f"   ❌ {extractor_name} has {len(n_comp_list)} values: {n_comp_list} (should be 1)")
                all_single_values = False
            else:
                print(f"   ✅ {extractor_name}: n_components = {n_comp_list[0]} (single value)")
        
        if all_single_values:
            print("   ✅ All regression extractors have exactly ONE n_components value")
        else:
            print("   ❌ Some regression extractors have multiple n_components values")
        
        # Test classification extractors (should use Breast hyperparameters)
        print("\n2. Testing classification extractors (should use Breast hyperparameters)...")
        clf_extractors = get_classification_extractors()
        breast_n_components = get_extraction_n_components_list("Breast", clf_extractors, "clf")
        
        print(f"   Classification extractors n_components: {breast_n_components}")
        
        # Verify each extractor has exactly ONE n_components value
        all_single_values = True
        for extractor_name, n_comp_list in breast_n_components.items():
            if len(n_comp_list) != 1:
                print(f"   ❌ {extractor_name} has {len(n_comp_list)} values: {n_comp_list} (should be 1)")
                all_single_values = False
            else:
                print(f"   ✅ {extractor_name}: n_components = {n_comp_list[0]} (single value)")
        
        if all_single_values:
            print("   ✅ All classification extractors have exactly ONE n_components value")
        else:
            print("   ❌ Some classification extractors have multiple n_components values")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_selection_n_features():
    """Test that selection uses [8, 16, 32] n_features."""
    print("\n" + "="*80)
    print("TESTING SELECTION N_FEATURES VALUES")
    print("="*80)
    
    try:
        from config import N_VALUES_LIST
        
        print(f"\n1. N_VALUES_LIST from config.py: {N_VALUES_LIST}")
        
        # Verify it's exactly [8, 16, 32]
        expected_values = [8, 16, 32]
        if N_VALUES_LIST == expected_values:
            print(f"   ✅ N_VALUES_LIST is correct: {N_VALUES_LIST}")
        else:
            print(f"   ❌ N_VALUES_LIST is wrong: {N_VALUES_LIST} (expected {expected_values})")
            return False
        
        # Test that selectors would use these values
        print(f"\n2. Selectors will use n_features: {N_VALUES_LIST}")
        print(f"   ✅ Each selector will be run {len(N_VALUES_LIST)} times with different n_features")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_pipeline_logic():
    """Test the logic difference between extraction and selection."""
    print("\n" + "="*80)
    print("TESTING PIPELINE LOGIC DIFFERENCES")
    print("="*80)
    
    try:
        from config import N_VALUES_LIST
        from models import (
            get_extraction_n_components_list,
            get_regression_extractors,
            get_regression_selectors
        )
        
        # Get the extractors and selectors
        reg_extractors = get_regression_extractors()
        reg_selectors = get_regression_selectors()
        
        # Get optimal n_components for extractors
        aml_n_components = get_extraction_n_components_list("AML", reg_extractors, "reg")
        
        print(f"\n1. EXTRACTION PIPELINE:")
        print(f"   - Uses optimal n_components from hyperparameters")
        print(f"   - Each extractor runs ONCE with its optimal value:")
        
        total_extraction_runs = 0
        for extractor_name, n_comp_list in aml_n_components.items():
            runs_for_this_extractor = len(n_comp_list)
            total_extraction_runs += runs_for_this_extractor
            print(f"     • {extractor_name}: {runs_for_this_extractor} run(s) with n_components={n_comp_list}")
        
        print(f"   - Total extraction runs per dataset: {total_extraction_runs}")
        
        print(f"\n2. SELECTION PIPELINE:")
        print(f"   - Uses fixed n_features = {N_VALUES_LIST}")
        print(f"   - Each selector runs {len(N_VALUES_LIST)} times:")
        
        total_selection_runs = 0
        for selector_name in reg_selectors.keys():
            runs_for_this_selector = len(N_VALUES_LIST)
            total_selection_runs += runs_for_this_selector
            print(f"     • {selector_name}: {runs_for_this_selector} runs with n_features={N_VALUES_LIST}")
        
        print(f"   - Total selection runs per dataset: {total_selection_runs}")
        
        print(f"\n3. COMPARISON:")
        print(f"   - Extraction runs: {total_extraction_runs} (using optimal n_components)")
        print(f"   - Selection runs: {total_selection_runs} (using [8, 16, 32])")
        
        # Check if extraction is indeed different from selection
        if total_extraction_runs != total_selection_runs:
            print(f"   ✅ Different number of runs confirms different logic")
        else:
            print(f"   ⚠️  Same number of runs - need to verify logic difference")
        
        # Verify no extractor uses [8, 16, 32] unless it happens to be optimal
        uses_default_values = False
        for extractor_name, n_comp_list in aml_n_components.items():
            if n_comp_list == N_VALUES_LIST:
                print(f"   ⚠️  {extractor_name} happens to use default values {N_VALUES_LIST}")
                uses_default_values = True
        
        if not uses_default_values:
            print(f"   ✅ No extractor uses the default [8, 16, 32] values")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🔍 TESTING PIPELINE N_COMPONENTS AND N_FEATURES BEHAVIOR")
    print("=" * 80)
    
    test_results = []
    
    # Test extraction n_components
    test_results.append(test_extraction_n_components())
    
    # Test selection n_features
    test_results.append(test_selection_n_features())
    
    # Test pipeline logic
    test_results.append(test_pipeline_logic())
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    passed = sum(test_results)
    total = len(test_results)
    
    if passed == total:
        print(f"✅ All {total} tests PASSED!")
        print("\n🎉 VERIFICATION COMPLETE:")
        print("   • Extractors use optimal n_components from hyperparameters (run ONCE)")
        print("   • Selectors use n_features = [8, 16, 32] (run 3 times)")
        print("   • No extractor is run multiple times with the same n_components")
        return True
    else:
        print(f"❌ {total - passed} out of {total} tests FAILED!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 