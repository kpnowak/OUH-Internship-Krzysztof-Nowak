#!/usr/bin/env python3
"""
Test script for the enhanced fusion methods:
1. Learnable weighted fusion
2. Multiple-Kernel Learning (MKL)
3. Similarity Network Fusion (SNF)
"""

import numpy as np
import logging
from fusion import (
    merge_modalities, 
    LearnableWeightedFusion, 
    MultipleKernelLearning, 
    SimilarityNetworkFusion
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_synthetic_genomic_data(n_samples=100, n_genes=1000, n_mirna=200, n_clinical=20, random_state=42):
    """
    Generate synthetic multi-modal genomic data for testing.
    
    Parameters
    ----------
    n_samples : int
        Number of samples
    n_genes : int
        Number of gene expression features
    n_mirna : int
        Number of miRNA features
    n_clinical : int
        Number of clinical features
    random_state : int
        Random state for reproducibility
        
    Returns
    -------
    tuple
        (gene_expression, mirna_expression, clinical_data, target_regression, target_classification)
    """
    np.random.seed(random_state)
    
    # Generate gene expression data (log-normal distribution)
    gene_expression = np.random.lognormal(mean=0, sigma=1, size=(n_samples, n_genes))
    
    # Generate miRNA expression data (normal distribution)
    mirna_expression = np.random.normal(loc=0, scale=1, size=(n_samples, n_mirna))
    
    # Generate clinical data (mixed types)
    clinical_data = np.random.normal(loc=50, scale=15, size=(n_samples, n_clinical))
    
    # Generate synthetic targets
    # Regression target: combination of some features with noise
    important_genes = gene_expression[:, :10].mean(axis=1)
    important_mirna = mirna_expression[:, :5].mean(axis=1)
    important_clinical = clinical_data[:, :3].mean(axis=1)
    
    target_regression = (
        0.5 * important_genes + 
        0.3 * important_mirna + 
        0.2 * important_clinical + 
        np.random.normal(0, 0.1, n_samples)
    )
    
    # Classification target: binary based on regression target
    target_classification = (target_regression > np.median(target_regression)).astype(int)
    
    return gene_expression, mirna_expression, clinical_data, target_regression, target_classification

def test_learnable_weighted_fusion():
    """Test learnable weighted fusion."""
    logger.info("Testing Learnable Weighted Fusion...")
    
    # Generate synthetic data
    gene_expr, mirna_expr, clinical, y_reg, y_clf = generate_synthetic_genomic_data()
    
    # Test regression
    logger.info("Testing learnable weights for regression...")
    learnable_fusion_reg = LearnableWeightedFusion(is_regression=True, cv_folds=3)
    
    # Fit and transform
    fused_data_reg = learnable_fusion_reg.fit_transform([gene_expr, mirna_expr, clinical], y_reg)
    logger.info(f"Regression fusion result shape: {fused_data_reg.shape}")
    logger.info(f"Modality weights: {learnable_fusion_reg.weights_}")
    logger.info(f"Modality performances: {learnable_fusion_reg.modality_performances_}")
    
    # Test classification
    logger.info("Testing learnable weights for classification...")
    learnable_fusion_clf = LearnableWeightedFusion(is_regression=False, cv_folds=3)
    
    # Fit and transform
    fused_data_clf = learnable_fusion_clf.fit_transform([gene_expr, mirna_expr, clinical], y_clf)
    logger.info(f"Classification fusion result shape: {fused_data_clf.shape}")
    logger.info(f"Modality weights: {learnable_fusion_clf.weights_}")
    logger.info(f"Modality performances: {learnable_fusion_clf.modality_performances_}")
    
    return fused_data_reg, fused_data_clf

def test_mkl_fusion():
    """Test Multiple-Kernel Learning fusion."""
    logger.info("Testing Multiple-Kernel Learning (MKL) Fusion...")
    
    # Generate smaller synthetic data for MKL (computationally intensive)
    gene_expr, mirna_expr, clinical, y_reg, y_clf = generate_synthetic_genomic_data(
        n_samples=50, n_genes=100, n_mirna=50, n_clinical=10
    )
    
    try:
        # Test regression
        logger.info("Testing MKL for regression...")
        mkl_fusion_reg = MultipleKernelLearning(is_regression=True, n_components=5)
        
        # Fit and transform
        fused_data_reg = mkl_fusion_reg.fit_transform([gene_expr, mirna_expr, clinical], y_reg)
        logger.info(f"MKL regression fusion result shape: {fused_data_reg.shape}")
        
        # Test classification
        logger.info("Testing MKL for classification...")
        mkl_fusion_clf = MultipleKernelLearning(is_regression=False, n_components=5)
        
        # Fit and transform
        fused_data_clf = mkl_fusion_clf.fit_transform([gene_expr, mirna_expr, clinical], y_clf)
        logger.info(f"MKL classification fusion result shape: {fused_data_clf.shape}")
        
        return fused_data_reg, fused_data_clf
        
    except ImportError:
        logger.warning("MKL library (mklaren) not available, skipping MKL test")
        return None, None
    except Exception as e:
        logger.error(f"MKL test failed: {str(e)}")
        return None, None

def test_snf_fusion():
    """Test Similarity Network Fusion."""
    logger.info("Testing Similarity Network Fusion (SNF)...")
    
    # Generate smaller synthetic data for SNF (computationally intensive)
    gene_expr, mirna_expr, clinical, y_reg, y_clf = generate_synthetic_genomic_data(
        n_samples=50, n_genes=100, n_mirna=50, n_clinical=10
    )
    
    try:
        # Test regression
        logger.info("Testing SNF for regression...")
        snf_fusion_reg = SimilarityNetworkFusion(
            K=10, alpha=0.5, T=10, 
            use_spectral_clustering=True, 
            is_regression=True
        )
        
        # Fit and transform
        fused_data_reg = snf_fusion_reg.fit_transform([gene_expr, mirna_expr, clinical], y_reg)
        logger.info(f"SNF regression fusion result shape: {fused_data_reg.shape}")
        
        # Test classification
        logger.info("Testing SNF for classification...")
        snf_fusion_clf = SimilarityNetworkFusion(
            K=10, alpha=0.5, T=10, 
            use_spectral_clustering=True, 
            is_regression=False
        )
        
        # Fit and transform
        fused_data_clf = snf_fusion_clf.fit_transform([gene_expr, mirna_expr, clinical], y_clf)
        logger.info(f"SNF classification fusion result shape: {fused_data_clf.shape}")
        
        return fused_data_reg, fused_data_clf
        
    except ImportError:
        logger.warning("SNF library (snfpy) not available, skipping SNF test")
        return None, None
    except Exception as e:
        logger.error(f"SNF test failed: {str(e)}")
        return None, None

def test_merge_modalities_with_new_strategies():
    """Test the updated merge_modalities function with new strategies."""
    logger.info("Testing merge_modalities with new fusion strategies...")
    
    # Generate synthetic data
    gene_expr, mirna_expr, clinical, y_reg, y_clf = generate_synthetic_genomic_data(
        n_samples=50, n_genes=100, n_mirna=50, n_clinical=10
    )
    
    # Test learnable_weighted strategy
    logger.info("Testing merge_modalities with 'learnable_weighted' strategy...")
    try:
        fused_learnable, fitted_fusion = merge_modalities(
            gene_expr, mirna_expr, clinical,
            strategy="learnable_weighted",
            y=y_reg,
            is_regression=True,
            is_train=True,
            fusion_params={'cv_folds': 3, 'random_state': 42}
        )
        logger.info(f"Learnable weighted merge result shape: {fused_learnable.shape}")
        
        # Test transform on validation data
        fused_val = merge_modalities(
            gene_expr, mirna_expr, clinical,
            strategy="learnable_weighted",
            fitted_fusion=fitted_fusion,
            is_train=False
        )
        logger.info(f"Learnable weighted validation transform shape: {fused_val.shape}")
        
    except Exception as e:
        logger.error(f"Learnable weighted merge test failed: {str(e)}")
    
    # Test MKL strategy
    logger.info("Testing merge_modalities with 'mkl' strategy...")
    try:
        fused_mkl, fitted_mkl = merge_modalities(
            gene_expr, mirna_expr, clinical,
            strategy="mkl",
            y=y_reg,
            is_regression=True,
            is_train=True,
            fusion_params={'n_components': 5, 'gamma': 1.0}
        )
        logger.info(f"MKL merge result shape: {fused_mkl.shape}")
        
    except Exception as e:
        logger.error(f"MKL merge test failed: {str(e)}")
    
    # Test SNF strategy
    logger.info("Testing merge_modalities with 'snf' strategy...")
    try:
        fused_snf, fitted_snf = merge_modalities(
            gene_expr, mirna_expr, clinical,
            strategy="snf",
            y=y_reg,
            is_regression=True,
            is_train=True,
            fusion_params={'K': 10, 'alpha': 0.5, 'T': 10}
        )
        logger.info(f"SNF merge result shape: {fused_snf.shape}")
        
    except Exception as e:
        logger.error(f"SNF merge test failed: {str(e)}")
    
    # Test enhanced weighted_concat with learnable weights
    logger.info("Testing enhanced 'weighted_concat' with learnable weights...")
    try:
        fused_enhanced = merge_modalities(
            gene_expr, mirna_expr, clinical,
            strategy="weighted_concat",
            y=y_reg,
            is_regression=True,
            is_train=True,
            fusion_params={'cv_folds': 3}
        )
        logger.info(f"Enhanced weighted_concat result shape: {fused_enhanced.shape}")
        
    except Exception as e:
        logger.error(f"Enhanced weighted_concat test failed: {str(e)}")

def main():
    """Run all fusion enhancement tests."""
    logger.info("Starting fusion enhancement tests...")
    
    # Test individual fusion methods
    test_learnable_weighted_fusion()
    test_mkl_fusion()
    test_snf_fusion()
    
    # Test integration with merge_modalities
    test_merge_modalities_with_new_strategies()
    
    logger.info("Fusion enhancement tests completed!")

if __name__ == "__main__":
    main() 