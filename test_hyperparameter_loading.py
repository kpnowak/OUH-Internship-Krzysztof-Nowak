#!/usr/bin/env python3
"""
Test script to verify that hyperparameters are loaded correctly from hp_best folder.
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

def test_hyperparameter_loading():
    """Test hyperparameter loading functionality."""
    print("=" * 80)
    print("TESTING HYPERPARAMETER LOADING FUNCTIONALITY")
    print("=" * 80)
    
    # Import the hyperparameter loading function
    try:
        from models import load_best_hyperparameters
        print("✅ Successfully imported load_best_hyperparameters from models.py")
    except ImportError as e:
        print(f"❌ Failed to import load_best_hyperparameters: {e}")
        return False
    
    # Check if hp_best folder exists
    hp_dir = Path("hp_best")
    if not hp_dir.exists():
        print(f"❌ hp_best directory not found at {hp_dir.absolute()}")
        return False
    
    # List available hyperparameter files
    hp_files = list(hp_dir.glob("*.json"))
    print(f"📁 Found {len(hp_files)} hyperparameter files in hp_best/")
    
    if len(hp_files) == 0:
        print("❌ No hyperparameter files found in hp_best/")
        return False
    
    # Test specific combinations
    test_cases = [
        # Regression tests (AML)
        ("AML", "PCA", "LinearRegression", "reg"),
        ("AML", "KPCA", "ElasticNet", "reg"),
        ("AML", "FA", "RandomForestRegressor", "reg"),
        
        # Classification tests (Breast)
        ("Breast", "PCA", "LogisticRegression", "clf"),
        ("Breast", "LDA", "SVC", "clf"),
        ("Breast", "KPCA", "RandomForestClassifier", "clf"),
        
        # Fallback tests (using AML hyperparams for other regression datasets)
        ("Sarcoma", "PCA", "LinearRegression", "reg"),
        
        # Fallback tests (using Breast hyperparams for other classification datasets) 
        ("Colon", "LDA", "LogisticRegression", "clf"),
    ]
    
    print("\n" + "=" * 60)
    print("TESTING HYPERPARAMETER LOADING FOR DIFFERENT COMBINATIONS")
    print("=" * 60)
    
    success_count = 0
    for i, (dataset, extractor, model, task) in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {dataset} + {extractor} + {model} ({task})")
        print("-" * 50)
        
        try:
            hyperparams = load_best_hyperparameters(dataset, extractor, model, task)
            
            extractor_params = hyperparams['extractor_params']
            model_params = hyperparams['model_params']
            source = hyperparams['source']
            
            print(f"   📊 Source: {source}")
            print(f"   🔧 Extractor parameters: {len(extractor_params)} found")
            if extractor_params:
                for key, value in extractor_params.items():
                    print(f"      • {key}: {value}")
            else:
                print("      • No extractor parameters found")
                
            print(f"   🤖 Model parameters: {len(model_params)} found")
            if model_params:
                for key, value in model_params.items():
                    print(f"      • {key}: {value}")
            else:
                print("      • No model parameters found")
            
            if extractor_params or model_params:
                print("   ✅ LOADED SUCCESSFULLY")
                success_count += 1
            else:
                print("   ⚠️  NO PARAMETERS LOADED")
                
        except Exception as e:
            print(f"   ❌ ERROR: {str(e)}")
    
    print("\n" + "=" * 60)
    print("TESTING EXTRACTOR HYPERPARAMETER APPLICATION")
    print("=" * 60)
    
    # Test that extractors can actually apply the hyperparameters
    try:
        from models import get_regression_extractors, get_classification_extractors
        
        print("\n1. Testing regression extractors with hyperparameters...")
        reg_extractors = get_regression_extractors()
        
        for extractor_name, extractor in reg_extractors.items():
            print(f"\n   Testing {extractor_name}:")
            
            # Load hyperparameters for this extractor
            hyperparams = load_best_hyperparameters("AML", extractor_name, "LinearRegression", "reg")
            extractor_params = hyperparams['extractor_params']
            
            if extractor_params:
                try:
                    # Try to apply the parameters
                    extractor.set_params(**extractor_params)
                    print(f"      ✅ Applied {len(extractor_params)} parameters successfully")
                    
                    # Show the applied parameters
                    for key, value in extractor_params.items():
                        if hasattr(extractor, key):
                            actual_value = getattr(extractor, key)
                            print(f"         • {key}: {value} → {actual_value}")
                        else:
                            print(f"         • {key}: {value} (parameter not found)")
                except Exception as e:
                    print(f"      ❌ Failed to apply parameters: {str(e)}")
            else:
                print(f"      ⚠️  No parameters found for {extractor_name}")
        
        print("\n2. Testing classification extractors with hyperparameters...")
        clf_extractors = get_classification_extractors()
        
        for extractor_name, extractor in clf_extractors.items():
            print(f"\n   Testing {extractor_name}:")
            
            # Load hyperparameters for this extractor
            hyperparams = load_best_hyperparameters("Breast", extractor_name, "LogisticRegression", "clf")
            extractor_params = hyperparams['extractor_params']
            
            if extractor_params:
                try:
                    # Try to apply the parameters
                    extractor.set_params(**extractor_params)
                    print(f"      ✅ Applied {len(extractor_params)} parameters successfully")
                    
                    # Show the applied parameters
                    for key, value in extractor_params.items():
                        if hasattr(extractor, key):
                            actual_value = getattr(extractor, key)
                            print(f"         • {key}: {value} → {actual_value}")
                        else:
                            print(f"         • {key}: {value} (parameter not found)")
                except Exception as e:
                    print(f"      ❌ Failed to apply parameters: {str(e)}")
            else:
                print(f"      ⚠️  No parameters found for {extractor_name}")
        
    except Exception as e:
        print(f"❌ ERROR in extractor testing: {str(e)}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ Successfully loaded hyperparameters for {success_count}/{len(test_cases)} test cases")
    print(f"📁 Total hyperparameter files available: {len(hp_files)}")
    
    if success_count == len(test_cases):
        print("🎉 ALL TESTS PASSED! Hyperparameter loading is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the hyperparameter files and loading logic.")
        return False

def test_optimal_n_components_extraction():
    """Test the new optimal n_components extraction functionality."""
    print("\n" + "="*80)
    print("TESTING OPTIMAL N_COMPONENTS EXTRACTION FROM HYPERPARAMETERS")
    print("="*80)
    
    try:
        from models import (
            get_optimal_n_components_from_hyperparams, 
            get_extraction_n_components_list,
            get_regression_extractors,
            get_classification_extractors
        )
        
        print("✅ Successfully imported n_components extraction functions")
        
        # Test regression extractors
        print("\n1. Testing optimal n_components for regression extractors (AML)...")
        reg_extractors = get_regression_extractors()
        
        for extractor_name in reg_extractors.keys():
            optimal_n = get_optimal_n_components_from_hyperparams("AML", extractor_name, "reg")
            if optimal_n is not None:
                print(f"   ✅ {extractor_name}: n_components = {optimal_n}")
            else:
                print(f"   ⚠️  {extractor_name}: No optimal n_components found")
        
        # Test classification extractors
        print("\n2. Testing optimal n_components for classification extractors (Breast)...")
        clf_extractors = get_classification_extractors()
        
        for extractor_name in clf_extractors.keys():
            optimal_n = get_optimal_n_components_from_hyperparams("Breast", extractor_name, "clf")
            if optimal_n is not None:
                print(f"   ✅ {extractor_name}: n_components = {optimal_n}")
            else:
                print(f"   ⚠️  {extractor_name}: No optimal n_components found")
        
        # Test the complete extraction function
        print("\n3. Testing complete extraction n_components list...")
        
        # Test for AML (regression)
        aml_n_components = get_extraction_n_components_list("AML", reg_extractors, "reg")
        print(f"   AML extraction n_components: {aml_n_components}")
        
        # Test for Breast (classification)
        breast_n_components = get_extraction_n_components_list("Breast", clf_extractors, "clf")
        print(f"   Breast extraction n_components: {breast_n_components}")
        
        # Verify the format is correct
        if isinstance(aml_n_components, dict) and isinstance(breast_n_components, dict):
            print("   ✅ Returned correct dictionary format")
            
            # Check if all extractors have values
            all_have_values = True
            for extractor_name in reg_extractors.keys():
                if extractor_name not in aml_n_components or not aml_n_components[extractor_name]:
                    print(f"   ⚠️  Missing or empty values for AML {extractor_name}")
                    all_have_values = False
                    
            for extractor_name in clf_extractors.keys():
                if extractor_name not in breast_n_components or not breast_n_components[extractor_name]:
                    print(f"   ⚠️  Missing or empty values for Breast {extractor_name}")
                    all_have_values = False
            
            if all_have_values:
                print("   ✅ All extractors have n_components values")
            else:
                print("   ⚠️  Some extractors missing n_components values")
        else:
            print("   ❌ Returned incorrect format (should be dict)")
        
        print("\n4. Testing fallback behavior...")
        # Test with a non-existent dataset
        fake_n_components = get_extraction_n_components_list("FakeDataset", reg_extractors, "reg")
        print(f"   FakeDataset extraction n_components (should fallback to AML): {fake_n_components}")
        
        print("\n✅ All n_components extraction tests completed successfully!")
        
    except Exception as e:
        print(f"❌ n_components extraction test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_hyperparameter_loading()
    test_optimal_n_components_extraction() 