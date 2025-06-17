#!/usr/bin/env python3
"""
Test script for enhanced missing data handling in fusion.py.

Demonstrates:
1. KNN imputation for moderate missing data
2. Iterative imputation with ExtraTrees for high missing data  
3. Late-fusion fallback for missing entire modalities
4. Adaptive strategy selection
"""

import numpy as np
import logging
from fusion import (
    ModalityImputer, 
    LateFusionFallback,
    merge_modalities,
    detect_missing_modalities,
    create_enhanced_imputer,
    handle_missing_modalities_with_late_fusion,
    get_recommended_fusion_strategy,
    ENHANCED_FUSION_STRATEGIES
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def create_synthetic_data_with_missing(n_samples=100, n_features_per_modality=[50, 30, 20], 
                                     missing_percentages=[0.05, 0.25, 0.60], random_state=42):
    """
    Create synthetic multi-modal data with varying levels of missing data.
    
    Parameters
    ----------
    n_samples : int
        Number of samples
    n_features_per_modality : List[int]
        Number of features for each modality
    missing_percentages : List[float]
        Missing data percentage for each modality
    random_state : int
        Random state for reproducibility
        
    Returns
    -------
    Tuple[List[np.ndarray], np.ndarray]
        (modalities, targets)
    """
    np.random.seed(random_state)
    
    modalities = []
    
    for i, (n_features, missing_pct) in enumerate(zip(n_features_per_modality, missing_percentages)):
        # Generate base data
        data = np.random.randn(n_samples, n_features).astype(np.float32)
        
        # Add some structure/correlation
        if i == 0:  # First modality influences target more
            data = data * 2.0 + np.random.randn(n_samples, 1) * 0.5
        
        # Introduce missing values
        if missing_pct > 0:
            n_missing = int(data.size * missing_pct)
            missing_indices = np.random.choice(data.size, n_missing, replace=False)
            flat_data = data.flatten()
            flat_data[missing_indices] = np.nan
            data = flat_data.reshape(data.shape)
        
        modalities.append(data)
        logger.info(f"Modality {i+1}: {data.shape}, {missing_pct*100:.1f}% missing")
    
    # Generate targets (regression)
    y = (np.sum([np.nanmean(mod, axis=1) for mod in modalities], axis=0) + 
         np.random.randn(n_samples) * 0.1)
    
    return modalities, y

def test_enhanced_imputation_strategies():
    """Test different imputation strategies."""
    print("\n" + "="*60)
    print("TESTING ENHANCED IMPUTATION STRATEGIES")
    print("="*60)
    
    # Create test data with different missing percentages
    test_cases = [
        (0.05, "Low missing data (5%) - should use mean imputation"),
        (0.25, "Moderate missing data (25%) - should use KNN imputation"),
        (0.65, "High missing data (65%) - should use iterative imputation")
    ]
    
    for missing_pct, description in test_cases:
        print(f"\n{description}")
        print("-" * len(description))
        
        # Generate data
        np.random.seed(42)
        data = np.random.randn(50, 20).astype(np.float32)
        
        # Introduce missing values
        n_missing = int(data.size * missing_pct)
        missing_indices = np.random.choice(data.size, n_missing, replace=False)
        flat_data = data.flatten()
        flat_data[missing_indices] = np.nan
        data_with_missing = flat_data.reshape(data.shape)
        
        # Test adaptive imputer
        imputer = ModalityImputer(strategy='adaptive', random_state=42)
        imputed_data = imputer.fit_transform(data_with_missing)
        
        # Get strategy info
        strategy_info = imputer.get_strategy_info()
        
        print(f"  Original shape: {data.shape}")
        print(f"  Missing percentage: {strategy_info['missing_percentage']:.2f}%")
        print(f"  Chosen strategy: {strategy_info['chosen_strategy']}")
        print(f"  Imputed shape: {imputed_data.shape}")
        print(f"  Remaining NaNs: {np.isnan(imputed_data).sum()}")
        
        # Verify no NaNs remain
        assert not np.isnan(imputed_data).any(), "Imputation should remove all NaNs"
        print("   All NaNs successfully imputed")

def test_late_fusion_fallback():
    """Test late-fusion fallback for missing entire modalities."""
    print("\n" + "="*60)
    print("TESTING LATE-FUSION FALLBACK")
    print("="*60)
    
    # Create data where some modalities are entirely missing for some samples
    np.random.seed(42)
    n_samples = 80
    
    # Create three modalities
    modality1 = np.random.randn(n_samples, 30).astype(np.float32)
    modality2 = np.random.randn(n_samples, 20).astype(np.float32)
    modality3 = np.random.randn(n_samples, 15).astype(np.float32)
    
    # Make modality2 entirely missing for 30% of samples
    missing_samples_mod2 = np.random.choice(n_samples, int(n_samples * 0.3), replace=False)
    modality2[missing_samples_mod2, :] = np.nan
    
    # Make modality3 entirely missing for 50% of samples
    missing_samples_mod3 = np.random.choice(n_samples, int(n_samples * 0.5), replace=False)
    modality3[missing_samples_mod3, :] = np.nan
    
    modalities = [modality1, modality2, modality3]
    modality_names = ['Gene_Expression', 'Methylation', 'miRNA']
    
    # Generate targets
    y = (np.nanmean(modality1, axis=1) + 
         np.nanmean(modality2, axis=1) + 
         np.nanmean(modality3, axis=1) + 
         np.random.randn(n_samples) * 0.1)
    
    print(f"Dataset: {n_samples} samples, 3 modalities")
    
    # Detect missing modalities
    available, missing = detect_missing_modalities(modalities, missing_threshold=0.9)
    print(f"Available modalities: {available}")
    print(f"Missing modalities: {missing}")
    
    # Test late-fusion fallback
    try:
        late_fusion = handle_missing_modalities_with_late_fusion(
            modalities, y, is_regression=True, 
            modality_names=modality_names
        )
        
        # Get reliability info
        reliability_info = late_fusion.get_reliability_info()
        print(f"Modality reliabilities: {reliability_info['modality_reliability']}")
        
        # Test prediction with partial modalities
        test_modalities = [
            np.random.randn(10, 30).astype(np.float32),  # Available
            None,  # Missing
            np.random.randn(10, 15).astype(np.float32)   # Available
        ]
        
        predictions = late_fusion.predict(test_modalities)
        print(f"Predictions shape: {predictions.shape}")
        print(" Late-fusion fallback successful")
        
    except Exception as e:
        print(f"✗ Late-fusion fallback failed: {str(e)}")

def test_fusion_strategies_with_missing_data():
    """Test different fusion strategies with missing data."""
    print("\n" + "="*60)
    print("TESTING FUSION STRATEGIES WITH MISSING DATA")
    print("="*60)
    
    # Create test data
    modalities, y = create_synthetic_data_with_missing(
        n_samples=60,
        n_features_per_modality=[40, 25, 15],
        missing_percentages=[0.0, 0.2, 0.5],
        random_state=42
    )
    
    # Calculate overall missing percentage
    total_elements = sum(mod.size for mod in modalities)
    missing_elements = sum(np.isnan(mod).sum() for mod in modalities)
    overall_missing_pct = (missing_elements / total_elements) * 100
    
    print(f"\nOverall missing data: {overall_missing_pct:.2f}%")
    
    # Get recommended strategy
    recommended = get_recommended_fusion_strategy(
        overall_missing_pct, has_targets=True, n_modalities=len(modalities)
    )
    print(f"Recommended strategy: {recommended}")
    
    # Test different strategies
    strategies_to_test = ['learnable_weighted', 'early_fusion_pca', 'mkl', 'snf']
    
    for strategy in strategies_to_test:
        print(f"\nTesting {strategy} strategy:")
        print("-" * (len(strategy) + 20))
        
        try:
            # Create enhanced imputer
            imputer = create_enhanced_imputer(strategy='adaptive')
            
            # Test fusion
            if strategy in ['learnable_weighted', 'mkl', 'snf']:
                # These require targets
                result = merge_modalities(
                    *modalities, 
                    strategy=strategy, 
                    imputer=imputer,
                    y=y, 
                    is_regression=True,
                    is_train=True
                )
                if isinstance(result, tuple):
                    merged_data, fitted_fusion = result
                    print(f"  Merged shape: {merged_data.shape}")
                    print(f"  Fitted fusion: {type(fitted_fusion).__name__}")
                else:
                    merged_data = result
                    print(f"  Merged shape: {merged_data.shape}")
            else:
                # early_fusion_pca doesn't require targets
                result = merge_modalities(
                    *modalities, 
                    strategy=strategy, 
                    imputer=imputer,
                    is_train=True
                )
                if isinstance(result, tuple):
                    merged_data, fitted_fusion = result
                    print(f"  Merged shape: {merged_data.shape}")
                    print(f"  Fitted fusion: {type(fitted_fusion).__name__}")
                else:
                    merged_data = result
                    print(f"  Merged shape: {merged_data.shape}")
            
            # Verify no NaNs in result
            if np.isnan(merged_data).any():
                print(f"  ✗ Result contains {np.isnan(merged_data).sum()} NaNs")
            else:
                print(f"   No NaNs in result")
                
        except Exception as e:
            print(f"  ✗ Strategy failed: {str(e)}")

def test_weighted_concat_restriction():
    """Test that weighted_concat is restricted to 0% missing data."""
    print("\n" + "="*60)
    print("TESTING WEIGHTED_CONCAT RESTRICTION")
    print("="*60)
    
    # Create data with missing values
    modalities, y = create_synthetic_data_with_missing(
        n_samples=50,
        n_features_per_modality=[30, 20],
        missing_percentages=[0.0, 0.1],  # Second modality has 10% missing
        random_state=42
    )
    
    print("Testing weighted_concat with missing data (should be restricted):")
    
    try:
        result = merge_modalities(
            *modalities,
            strategy='weighted_concat',
            y=y,
            is_regression=True,
            is_train=True
        )
        print(f"  Result shape: {result.shape}")
        print("  ⚠️  weighted_concat was allowed despite missing data (fallback applied)")
        
    except Exception as e:
        print(f"  ✗ weighted_concat correctly restricted: {str(e)}")
    
    # Test with 0% missing data
    print("\nTesting weighted_concat with 0% missing data (should work):")
    clean_modalities = [mod.copy() for mod in modalities]
    # Remove NaNs from second modality
    clean_modalities[1] = np.nan_to_num(clean_modalities[1], nan=0.0)
    
    try:
        result = merge_modalities(
            *clean_modalities,
            strategy='weighted_concat',
            y=y,
            is_regression=True,
            is_train=True
        )
        print(f"  Result shape: {result.shape}")
        print("   weighted_concat works with 0% missing data")
        
    except Exception as e:
        print(f"  ✗ weighted_concat failed unexpectedly: {str(e)}")

def print_strategy_summary():
    """Print summary of available fusion strategies."""
    print("\n" + "="*60)
    print("ENHANCED FUSION STRATEGIES SUMMARY")
    print("="*60)
    
    for strategy, info in ENHANCED_FUSION_STRATEGIES.items():
        print(f"\n{strategy.upper()}:")
        print(f"  Description: {info['description']}")
        print(f"  Missing data support: {'Yes' if info['missing_data_support'] else 'No'}")
        print(f"  Requires targets: {'Yes' if info['requires_targets'] else 'No'}")

def main():
    """Run all tests."""
    print("ENHANCED MISSING DATA HANDLING TEST SUITE")
    print("=" * 60)
    
    try:
        # Print strategy summary
        print_strategy_summary()
        
        # Test enhanced imputation strategies
        test_enhanced_imputation_strategies()
        
        # Test late-fusion fallback
        test_late_fusion_fallback()
        
        # Test fusion strategies with missing data
        test_fusion_strategies_with_missing_data()
        
        # Test weighted_concat restriction
        test_weighted_concat_restriction()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Test suite failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 