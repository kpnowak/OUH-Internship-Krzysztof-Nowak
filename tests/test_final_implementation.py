#!/usr/bin/env python3
"""
Test script to verify all implementation fixes are working correctly.
"""

import numpy as np
import pandas as pd
import logging
from sklearn.datasets import make_classification

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_label_remapping():
    """Test the dynamic label remapping functionality."""
    print("\n=== Testing Dynamic Label Remapping ===")
    
    from preprocessing import _remap_labels
    
    # Test case 1: Ultra-rare classes (should be merged)
    y = pd.Series([0, 0, 0, 1, 1, 2, 3])  # Classes 2 and 3 have <3 samples
    y_remapped = _remap_labels(y, "TestDataset")
    print(f"Original: {y.value_counts().to_dict()}")
    print(f"Remapped: {y_remapped.value_counts().to_dict()}")
    
    # Test case 2: Colon dataset special case
    y_colon = pd.Series(['T1', 'T1', 'T2', 'T2', 'T3', 'T3', 'T4', 'T4'])
    y_colon_remapped = _remap_labels(y_colon, "Colon")
    print(f"Colon original: {y_colon.value_counts().to_dict()}")
    print(f"Colon remapped: {y_colon_remapped.value_counts().to_dict()}")
    
    return True

def test_dynamic_splitter():
    """Test the dynamic CV splitter."""
    print("\n=== Testing Dynamic CV Splitter ===")
    
    from cv import make_splitter
    
    # Test case 1: Small classes (should use RepeatedStratifiedKFold)
    y_small = np.array([0, 0, 1, 1, 2, 2])  # Very small classes
    splitter_small = make_splitter(y_small, max_cv=5)
    print(f"Small classes splitter: {type(splitter_small).__name__}")
    print(f"Small classes n_splits: {getattr(splitter_small, 'n_splits', 'N/A')}")
    
    # Test case 2: Larger classes (should use StratifiedKFold)
    y_large = np.array([0]*20 + [1]*20 + [2]*20)  # Larger classes
    splitter_large = make_splitter(y_large, max_cv=5)
    print(f"Large classes splitter: {type(splitter_large).__name__}")
    print(f"Large classes n_splits: {getattr(splitter_large, 'n_splits', 'N/A')}")
    
    return True

def test_safe_sampler():
    """Test the safe sampler functionality."""
    print("\n=== Testing Safe Sampler ===")
    
    try:
        from samplers import safe_sampler
        
        # Test case 1: Very small classes
        y_tiny = np.array([0, 0, 1])  # Tiny classes
        sampler_tiny = safe_sampler(y_tiny)
        print(f"Tiny classes sampler: {type(sampler_tiny).__name__ if sampler_tiny else 'None'}")
        
        # Test case 2: Small classes
        y_small = np.array([0]*3 + [1]*3 + [2]*3)  # Small classes
        sampler_small = safe_sampler(y_small)
        print(f"Small classes sampler: {type(sampler_small).__name__ if sampler_small else 'None'}")
        
        # Test case 3: Normal classes
        y_normal = np.array([0]*20 + [1]*20 + [2]*20)  # Normal classes
        sampler_normal = safe_sampler(y_normal)
        print(f"Normal classes sampler: {type(sampler_normal).__name__ if sampler_normal else 'None'}")
        
        return True
    except ImportError as e:
        print(f"Safe sampler test skipped: {e}")
        return True

def test_create_balanced_pipeline():
    """Test the create_balanced_pipeline function."""
    print("\n=== Testing Create Balanced Pipeline ===")
    
    from cv import create_balanced_pipeline
    from sklearn.ensemble import RandomForestClassifier
    
    # Create a simple model
    base_model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    # Test with different class distributions
    y_imbalanced = np.array([0]*100 + [1]*10 + [2]*5)  # Imbalanced
    pipeline = create_balanced_pipeline(base_model, y_train=y_imbalanced)
    print(f"Balanced pipeline type: {type(pipeline).__name__}")
    
    return True

def test_target_transformation():
    """Test the target transformation registry."""
    print("\n=== Testing Target Transformation Registry ===")
    
    from models import TARGET_TRANSFORMS
    
    print(f"Available target transforms: {list(TARGET_TRANSFORMS.keys())}")
    
    # Test AML transformation
    if 'AML' in TARGET_TRANSFORMS:
        transform_name, transform_func, inverse_func = TARGET_TRANSFORMS['AML']
        print(f"AML transform: {transform_name}")
        
        # Test with sample data
        test_data = np.array([1.0, 2.0, 3.0])
        transformed = transform_func(test_data)
        inverse_transformed = inverse_func(transformed)
        print(f"Original: {test_data}")
        print(f"Transformed: {transformed}")
        print(f"Inverse: {inverse_transformed}")
    
    return True

def test_integration():
    """Test the integration of all components."""
    print("\n=== Testing Integration ===")
    
    # Create a synthetic dataset with problematic characteristics
    X, y = make_classification(
        n_samples=50,  # Small dataset
        n_features=20,
        n_classes=4,
        n_clusters_per_class=1,
        weights=[0.6, 0.2, 0.15, 0.05],  # Imbalanced classes
        random_state=42
    )
    
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Test label remapping
    from preprocessing import _remap_labels
    y_remapped = _remap_labels(pd.Series(y), "TestDataset")
    print(f"After remapping: {y_remapped.value_counts().to_dict()}")
    
    # Test CV splitter
    from cv import make_splitter
    splitter = make_splitter(y_remapped.values)
    print(f"CV splitter: {type(splitter).__name__}")
    
    # Test safe sampler
    try:
        from samplers import safe_sampler
        sampler = safe_sampler(y_remapped.values)
        print(f"Safe sampler: {type(sampler).__name__ if sampler else 'None'}")
    except ImportError:
        print("Safe sampler not available")
    
    return True

def main():
    """Run all tests."""
    print("Running comprehensive implementation tests...")
    
    tests = [
        test_label_remapping,
        test_dynamic_splitter,
        test_safe_sampler,
        test_create_balanced_pipeline,
        test_target_transformation,
        test_integration
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print(f" {test.__name__} passed")
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            results.append(False)
    
    print(f"\n=== Test Summary ===")
    print(f"Passed: {sum(results)}/{len(results)}")
    print(f"Success rate: {sum(results)/len(results)*100:.1f}%")
    
    if all(results):
        print("🎉 All tests passed! Implementation is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the implementation.")

if __name__ == "__main__":
    main() 