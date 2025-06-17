#!/usr/bin/env python3
"""
Data Quality Analyzer for Pre-Model Training Data

This script analyzes the quality of data right before it gets fed into the models.
It runs independently from the main pipeline to allow for hyperparameter adjustments
without affecting the main training process.

The script analyzes data for every combination of:

REGRESSION ALGORITHMS:
- Extraction techniques: PCA, KPLS, FA, PLS, SparsePLS + KernelPCA-RBF (feature engineering)
- Selection techniques: ElasticNetFS, RFImportance, VarianceFTest, LASSO, f_regressionFS

CLASSIFICATION ALGORITHMS:
- Extraction techniques: PCA, FA, LDA, PLS-DA, SparsePLS + SparsePLS-DA (feature engineering)
- Selection techniques: ElasticNetFS, RFImportance, VarianceFTest, LogisticL1, XGBoostFS

FUSION TECHNIQUES:
- Basic: weighted_concat
- Advanced: learnable_weighted, attention_weighted, mkl, snf, early_fusion_pca

DATASETS:
- Regression (2): AML, Sarcoma
- Classification (7): Colon, Breast, Kidney, Liver, Lung, Melanoma, Ovarian

Metrics analyzed:
- Zero appearance percentage
- Mean and standard deviation
- Min/max values
- Variance
- Skewness and kurtosis
- Number of features and samples
- Missing value percentage
- Outlier percentage
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import logging
from scipy import stats
from sklearn.preprocessing import RobustScaler
import json
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_quality_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import project modules
try:
    from config import (
        N_VALUES_LIST, MISSING_MODALITIES_CONFIG, FUSION_UPGRADES_CONFIG,
        FEATURE_ENGINEERING_CONFIG
    )
    from data_io import load_dataset
    from models import (
        get_regression_extractors, get_classification_extractors,
        get_regression_selectors, get_classification_selectors,
        cached_fit_transform_extractor_regression, cached_fit_transform_extractor_classification,
        cached_fit_transform_selector_regression, cached_fit_transform_selector_classification,
        transform_extractor_regression, transform_extractor_classification,
        transform_selector_regression, transform_selector_classification
    )
    from fusion import merge_modalities, ModalityImputer
    from preprocessing import (
        process_with_missing_modalities, safe_statistical_computation, 
        check_numerical_stability
    )
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

# Define datasets directly to ensure all 9 are analyzed
REGRESSION_DATASETS_FOR_ANALYSIS = [
    #{
    #    "name": "AML",
    #    "base_path": "data/aml",
    #    "modalities": {
    #        "Gene Expression": "exp.csv",
    #        "miRNA": "mirna.csv",
    #        "Methylation": "methy.csv"
    #    },
    #    "outcome_file": "data/clinical/aml.csv",
    #    "outcome_col": "lab_procedure_bone_marrow_blast_cell_outcome_percent_value",
    #    "id_col": "sampleID",
    #    "outcome_type": "continuous",
    #    "fix_tcga_ids": True
    #},
    #{
    #    "name": "Sarcoma",
    #    "base_path": "data/sarcoma",
    #    "modalities": {
    #        "Gene Expression": "exp.csv",
    #        "miRNA": "mirna.csv",
    #        "Methylation": "methy.csv"
    #    },
    #    "outcome_file": "data/clinical/sarcoma.csv",
    #    "outcome_col": "pathologic_tumor_length",
    #    "id_col": "metsampleID",
    #    "outcome_type": "continuous",
    #    "fix_tcga_ids": True
    #}
]

CLASSIFICATION_DATASETS_FOR_ANALYSIS = [
    {
        "name": "Colon",
        "base_path": "data/colon",
        "modalities": {
            "Gene Expression": "exp.csv",
            "miRNA": "mirna.csv",
            "Methylation": "methy.csv"
        },
        "outcome_file": "data/clinical/colon.csv",
        "outcome_col": "pathologic_T",
        "id_col": "sampleID",
        "outcome_type": "class",
        "fix_tcga_ids": True
    },
    #{
    #    "name": "Breast",
    #    "base_path": "data/breast",
    #    "modalities": {
    #        "Gene Expression": "exp.csv",
    #        "miRNA": "mirna.csv",
    #        "Methylation": "methy.csv"
    #    },
    #    "outcome_file": "data/clinical/breast.csv",
    #    "outcome_col": "pathologic_T",
    #    "id_col": "sampleID",
    #    "outcome_type": "class",
    #    "fix_tcga_ids": True
    #},
    #{
    #    "name": "Kidney",
    #    "base_path": "data/kidney",
    #    "modalities": {
    #        "Gene Expression": "exp.csv",
    #        "miRNA": "mirna.csv",
    #        "Methylation": "methy.csv"
    #    },
    #    "outcome_file": "data/clinical/kidney.csv",
    #    "outcome_col": "pathologic_T",
    #    "id_col": "submitter_id.samples",
    #    "outcome_type": "class",
    #    "fix_tcga_ids": True
    #},
    #{
    #    "name": "Liver",
    #    "base_path": "data/liver",
    #    "modalities": {
    #        "Gene Expression": "exp.csv",
    #        "miRNA": "mirna.csv",
    #        "Methylation": "methy.csv"
    #    },
    #    "outcome_file": "data/clinical/liver.csv",
    #    "outcome_col": "pathologic_T",
    #    "id_col": "sampleID",
    #    "outcome_type": "class",
    #    "fix_tcga_ids": True
    #},
    #{
    #    "name": "Lung",
    #    "base_path": "data/lung",
    #    "modalities": {
    #        "Gene Expression": "exp.csv",
    #        "miRNA": "mirna.csv",
    #        "Methylation": "methy.csv"
    #    },
    #    "outcome_file": "data/clinical/lung.csv",
    #    "outcome_col": "pathologic_T",
    #    "id_col": "sampleID",
    #    "outcome_type": "class",
    #    "fix_tcga_ids": True
    #},
    #{
    #    "name": "Melanoma",
    #    "base_path": "data/melanoma",
    #    "modalities": {
    #        "Gene Expression": "exp.csv",
    #        "miRNA": "mirna.csv",
    #        "Methylation": "methy.csv"
    #    },
    #    "outcome_file": "data/clinical/melanoma.csv",
    #    "outcome_col": "pathologic_T",
    #    "id_col": "sampleID",
    #    "outcome_type": "class",
    #    "fix_tcga_ids": True
    #},
    #{
    #    "name": "Ovarian",
    #    "base_path": "data/ovarian",
    #    "modalities": {
    #        "Gene Expression": "exp.csv",
    #        "miRNA": "mirna.csv",
    #        "Methylation": "methy.csv"
    #    },
    #    "outcome_file": "data/clinical/ovarian.csv",
    #    "outcome_col": "clinical_stage",
    #    "id_col": "sampleID",
    #    "outcome_type": "class",
    #    "fix_tcga_ids": True
    #}
]


class DataQualityAnalyzer:
    """Analyzes data quality at various stages of the preprocessing pipeline."""
    
    def __init__(self, output_dir: str = "data_quality_analysis"):
        """
        Initialize the data quality analyzer.
        
        Parameters
        ----------
        output_dir : str
            Directory to save analysis results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "regression").mkdir(exist_ok=True)
        (self.output_dir / "classification").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "summary").mkdir(exist_ok=True)
        
        self.results = {}
        
        # Enable feature engineering for comprehensive analysis
        FEATURE_ENGINEERING_CONFIG["enabled"] = True
        logger.info("Feature engineering enabled for comprehensive analysis")
        
    def calculate_data_metrics(self, X: np.ndarray, name: str = "unknown") -> Dict[str, Any]:
        """
        Calculate comprehensive data quality metrics using safe statistical computation.
        
        Parameters
        ----------
        X : np.ndarray
            Input data matrix
        name : str
            Name/identifier for the data
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing all calculated metrics with numerical stability
        """
        try:
            # Use safe statistical computation to avoid NaN issues
            safe_stats = safe_statistical_computation(X)
            
            # Extract global statistics for backward compatibility
            global_stats = safe_stats.get('global_stats', {})
            
            # Basic shape information
            metrics = {
                'n_samples': safe_stats['n_samples'],
                'n_features': safe_stats['n_features'],
                'total_elements': global_stats.get('total_elements', X.size)
            }
            
            # Handle empty arrays
            if X.size == 0:
                logger.warning(f"Empty array for {name}")
                metrics.update({
                    'zero_percentage': np.nan,
                    'missing_percentage': np.nan,
                    'mean': np.nan,
                    'std': np.nan,
                    'min': np.nan,
                    'max': np.nan,
                    'variance': np.nan,
                    'skewness': np.nan,
                    'kurtosis': np.nan,
                    'outlier_percentage': np.nan,
                    'range': np.nan,
                    'median': np.nan,
                    'q25': np.nan,
                    'q75': np.nan,
                    'iqr': np.nan
                })
                return metrics
            
            # Extract safe global statistics
            metrics.update({
                'missing_percentage': global_stats.get('missing_percentage', 0.0),
                'zero_percentage': global_stats.get('zero_percentage', 0.0),
                'mean': global_stats.get('mean', np.nan),
                'std': global_stats.get('std', np.nan),
                'min': global_stats.get('min', np.nan),
                'max': global_stats.get('max', np.nan),
                'median': global_stats.get('median', np.nan)
            })
            
            # Calculate derived metrics safely
            if not np.isnan(metrics['std']):
                metrics['variance'] = metrics['std'] ** 2
            else:
                metrics['variance'] = np.nan
                
            if not (np.isnan(metrics['max']) or np.isnan(metrics['min'])):
                metrics['range'] = metrics['max'] - metrics['min']
            else:
                metrics['range'] = np.nan
            
            # Calculate percentiles and IQR safely
            try:
                X_flat = X.astype(np.float64).flatten()
                finite_values = X_flat[~(np.isnan(X_flat) | np.isinf(X_flat))]
                
                if len(finite_values) > 0:
                    metrics['q25'] = float(np.percentile(finite_values, 25))
                    metrics['q75'] = float(np.percentile(finite_values, 75))
                    metrics['iqr'] = metrics['q75'] - metrics['q25']
                    
                    # Safe skewness and kurtosis calculation
                    if len(finite_values) > 3 and metrics['std'] > 1e-10:
                        try:
                            metrics['skewness'] = float(stats.skew(finite_values))
                            if np.isnan(metrics['skewness']) or np.isinf(metrics['skewness']):
                                metrics['skewness'] = np.nan
                        except:
                            metrics['skewness'] = np.nan
                            
                        try:
                            metrics['kurtosis'] = float(stats.kurtosis(finite_values))
                            if np.isnan(metrics['kurtosis']) or np.isinf(metrics['kurtosis']):
                                metrics['kurtosis'] = np.nan
                        except:
                            metrics['kurtosis'] = np.nan
                    else:
                        metrics['skewness'] = np.nan
                        metrics['kurtosis'] = np.nan
                    
                    # Safe outlier detection using IQR method
                    if metrics['iqr'] > 1e-10:
                        q1, q3 = metrics['q25'], metrics['q75']
                        iqr = metrics['iqr']
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        outliers = np.sum((finite_values < lower_bound) | (finite_values > upper_bound))
                        metrics['outlier_percentage'] = (outliers / len(finite_values)) * 100
                    else:
                        metrics['outlier_percentage'] = 0.0
                else:
                    # No finite values
                    metrics.update({
                        'q25': np.nan, 'q75': np.nan, 'iqr': np.nan,
                        'skewness': np.nan, 'kurtosis': np.nan,
                        'outlier_percentage': np.nan
                    })
                    
            except Exception as e:
                logger.warning(f"Error in percentile/distribution calculations for {name}: {e}")
                metrics.update({
                    'q25': np.nan, 'q75': np.nan, 'iqr': np.nan,
                    'skewness': np.nan, 'kurtosis': np.nan,
                    'outlier_percentage': np.nan
                })
            
                                        # Add numerical stability information
            try:
                stability_report = check_numerical_stability(X, min_variance=1e-10)
                metrics['numerical_stability'] = {
                    'problematic_features_count': len(stability_report['problematic_features']),
                    'zero_variance_features_count': len(stability_report['zero_variance_features']),
                    'constant_features_count': len(stability_report['constant_features']),
                    'nan_producing_features_count': len(stability_report['nan_producing_features']),
                    'recommendations': stability_report['recommendations']
                }
                
                if len(stability_report['problematic_features']) > 0:
                    logger.warning(f"Found {len(stability_report['problematic_features'])} numerically unstable features in {name}")
                    
            except Exception as e:
                logger.warning(f"Numerical stability check failed for {name}: {e}")
                metrics['numerical_stability'] = {'error': str(e)}
            
            # Add robust scaling effectiveness metrics (NEW)
            try:
                # Test robust scaling effectiveness on this data
                scaling_effectiveness = self.analyze_scaling_effectiveness(X, name)
                metrics['scaling_effectiveness'] = scaling_effectiveness
                
            except Exception as e:
                logger.warning(f"Scaling effectiveness analysis failed for {name}: {e}")
                metrics['scaling_effectiveness'] = {'error': str(e)}
            
        except Exception as e:
            logger.error(f"Error calculating safe metrics for {name}: {e}")
            # Return NaN metrics in case of error
            metrics = {
                'n_samples': 0, 'n_features': 0, 'total_elements': 0,
                'zero_percentage': np.nan, 'missing_percentage': np.nan,
                'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan,
                'variance': np.nan, 'skewness': np.nan, 'kurtosis': np.nan,
                'outlier_percentage': np.nan, 'range': np.nan, 'median': np.nan,
                'q25': np.nan, 'q75': np.nan, 'iqr': np.nan,
                'numerical_stability': {'error': str(e)}
            }
        
        return metrics
    
    def analyze_scaling_effectiveness(self, X: np.ndarray, name: str = "unknown") -> Dict[str, Any]:
        """
        Analyze the effectiveness of robust scaling vs standard scaling.
        
        Parameters
        ----------
        X : np.ndarray
            Input data matrix
        name : str
            Name/identifier for the data
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing scaling effectiveness metrics
        """
        try:
            from sklearn.preprocessing import StandardScaler, RobustScaler
            from sklearn.decomposition import PCA
            
            # Skip analysis if data is too small or empty
            if X.size == 0 or X.shape[0] < 10 or X.shape[1] < 5:
                return {
                    'analysis_skipped': True,
                    'reason': 'insufficient_data',
                    'data_shape': X.shape
                }
            
            # Ensure we have finite values
            X_clean = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
            
            scaling_analysis = {
                'data_characteristics': {
                    'shape': X.shape,
                    'outlier_potential': 'unknown',
                    'sparsity': float(np.sum(X_clean == 0) / X_clean.size),
                    'dynamic_range': float(np.max(X_clean) - np.min(X_clean)) if X_clean.size > 0 else 0.0
                }
            }
            
            # Analyze outlier potential
            try:
                feature_vars = np.var(X_clean, axis=0)
                if len(feature_vars) > 0:
                    var_ratio = np.max(feature_vars) / (np.min(feature_vars) + 1e-10)
                    scaling_analysis['data_characteristics']['variance_ratio'] = float(var_ratio)
                    scaling_analysis['data_characteristics']['outlier_potential'] = 'high' if var_ratio > 100 else 'moderate' if var_ratio > 10 else 'low'
            except:
                scaling_analysis['data_characteristics']['variance_ratio'] = np.nan
            
            # Test StandardScaler
            try:
                standard_scaler = StandardScaler()
                X_standard = standard_scaler.fit_transform(X_clean)
                
                # Test PCA on standard scaled data
                pca_standard = PCA(n_components=min(10, X.shape[0]-1, X.shape[1]))
                pca_standard.fit(X_standard)
                
                scaling_analysis['standard_scaler'] = {
                    'scaling_method': 'standard',
                    'pca_explained_variance': pca_standard.explained_variance_.tolist(),
                    'pca_variance_std': float(np.std(pca_standard.explained_variance_)),
                    'pca_variance_range': float(np.max(pca_standard.explained_variance_) - np.min(pca_standard.explained_variance_)),
                    'pca_variance_ratio': float(np.max(pca_standard.explained_variance_) / (np.min(pca_standard.explained_variance_) + 1e-10)),
                    'scaled_data_stats': {
                        'mean': float(np.mean(X_standard)),
                        'std': float(np.std(X_standard)),
                        'min': float(np.min(X_standard)),
                        'max': float(np.max(X_standard))
                    }
                }
                
            except Exception as e:
                scaling_analysis['standard_scaler'] = {'error': str(e)}
            
            # Test RobustScaler
            try:
                robust_scaler = RobustScaler()
                X_robust = robust_scaler.fit_transform(X_clean)
                
                # Test PCA on robust scaled data
                pca_robust = PCA(n_components=min(10, X.shape[0]-1, X.shape[1]))
                pca_robust.fit(X_robust)
                
                scaling_analysis['robust_scaler'] = {
                    'scaling_method': 'robust',
                    'pca_explained_variance': pca_robust.explained_variance_.tolist(),
                    'pca_variance_std': float(np.std(pca_robust.explained_variance_)),
                    'pca_variance_range': float(np.max(pca_robust.explained_variance_) - np.min(pca_robust.explained_variance_)),
                    'pca_variance_ratio': float(np.max(pca_robust.explained_variance_) / (np.min(pca_robust.explained_variance_) + 1e-10)),
                    'scaled_data_stats': {
                        'mean': float(np.mean(X_robust)),
                        'std': float(np.std(X_robust)),
                        'min': float(np.min(X_robust)),
                        'max': float(np.max(X_robust))
                    },
                    'scaler_params': {
                        'center': robust_scaler.center_.tolist() if hasattr(robust_scaler, 'center_') else None,
                        'scale': robust_scaler.scale_.tolist() if hasattr(robust_scaler, 'scale_') else None
                    }
                }
                
            except Exception as e:
                scaling_analysis['robust_scaler'] = {'error': str(e)}
            
            # Compare effectiveness
            try:
                if 'standard_scaler' in scaling_analysis and 'robust_scaler' in scaling_analysis:
                    std_results = scaling_analysis['standard_scaler']
                    rob_results = scaling_analysis['robust_scaler']
                    
                    if 'error' not in std_results and 'error' not in rob_results:
                        # Calculate improvement metrics
                        std_variance_std = std_results['pca_variance_std']
                        rob_variance_std = rob_results['pca_variance_std']
                        
                        std_variance_ratio = std_results['pca_variance_ratio']
                        rob_variance_ratio = rob_results['pca_variance_ratio']
                        
                        scaling_analysis['comparison'] = {
                            'pca_variance_std_improvement': float(std_variance_std / (rob_variance_std + 1e-10)),
                            'pca_variance_ratio_improvement': float(std_variance_ratio / (rob_variance_ratio + 1e-10)),
                            'robust_scaler_better_variance_std': bool(rob_variance_std < std_variance_std),
                            'robust_scaler_better_variance_ratio': bool(rob_variance_ratio < std_variance_ratio),
                            'recommendation': 'robust' if (rob_variance_std < std_variance_std and rob_variance_ratio < std_variance_ratio) else 'standard',
                            'improvement_significant': bool(std_variance_std / (rob_variance_std + 1e-10) > 1.2 or std_variance_ratio / (rob_variance_ratio + 1e-10) > 1.2)
                        }
                        
                        # Log significant improvements
                        if scaling_analysis['comparison']['improvement_significant']:
                            logger.info(f"Significant scaling improvement detected for {name}: "
                                       f"variance std {std_variance_std:.3f} -> {rob_variance_std:.3f}, "
                                       f"variance ratio {std_variance_ratio:.2f} -> {rob_variance_ratio:.2f}")
                        
            except Exception as e:
                scaling_analysis['comparison'] = {'error': str(e)}
            
            # Test our robust_data_scaling function if available
            try:
                from preprocessing import robust_data_scaling
                from config import ENHANCED_PREPROCESSING_CONFIGS
                
                # Test with different modality configurations
                for modality_type in ['miRNA', 'Gene Expression', 'Methylation']:
                    config = ENHANCED_PREPROCESSING_CONFIGS.get(modality_type, {})
                    
                    # Apply our robust scaling function
                    X_scaled, _, scaler, scaling_report = robust_data_scaling(
                        X_clean, None, config, modality_type
                    )
                    
                    if scaling_report.get('scaling_applied', False):
                        scaling_analysis[f'robust_data_scaling_{modality_type.lower().replace(" ", "_")}'] = {
                            'scaling_report': scaling_report,
                            'variance_reduction_ratio': scaling_report.get('variance_reduction_ratio', np.nan),
                            'scaling_method': scaling_report.get('scaling_method', 'unknown'),
                            'outlier_clipping_applied': scaling_report.get('outlier_clipping_applied', False),
                            'quantile_range': scaling_report.get('quantile_range', None)
                        }
                        
            except Exception as e:
                scaling_analysis['robust_data_scaling_test'] = {'error': str(e)}
            
            return scaling_analysis
            
        except Exception as e:
            logger.error(f"Scaling effectiveness analysis failed for {name}: {e}")
            return {'error': str(e)}
    
    def analyze_dataset_quality(self, dataset_config: Dict[str, Any], is_regression: bool = True) -> Dict[str, Any]:
        """
        Analyze data quality for a complete dataset through all processing stages.
        
        Parameters
        ----------
        dataset_config : Dict[str, Any]
            Dataset configuration
        is_regression : bool
            Whether this is a regression or classification task
            
        Returns
        -------
        Dict[str, Any]
            Comprehensive analysis results
        """
        dataset_name = dataset_config['name']
        logger.info(f"Analyzing data quality for {dataset_name} ({'regression' if is_regression else 'classification'})")
        
        analysis_results = {
            'dataset_name': dataset_name,
            'task_type': 'regression' if is_regression else 'classification',
            'timestamp': datetime.now().isoformat(),
            'stages': {}
        }
        
        try:
            # Load data using the existing load_dataset function
            logger.info(f"Loading data for {dataset_name}")
            task_type = "regression" if is_regression else "classification"
            
            # Use the custom loading that handles modality filename mapping correctly
            modalities_config = dataset_config.get('modalities', {})
            modalities = {}
            
            # Load each modality using the correct filename
            for modality_name, filename in modalities_config.items():
                try:
                    from data_io import load_modality
                    base_path = dataset_config.get('base_path', f'data/{dataset_name.lower()}')
                    
                    # Load modality with correct path
                    mod_df = load_modality(base_path, filename, modality_name)
                    if mod_df is not None and not mod_df.empty:
                        modalities[modality_name] = mod_df
                        logger.info(f"[OK] Loaded {modality_name}: {mod_df.shape}")
                    else:
                        logger.warning(f"[FAIL] Failed to load {modality_name} from {filename}")
                except Exception as e:
                    logger.error(f"Error loading {modality_name}: {str(e)}")
            
            if not modalities:
                logger.error(f"No modalities loaded successfully for {dataset_name}")
                return analysis_results
            
            # Load outcome data
            from data_io import load_outcome
            outcome_file = dataset_config.get('outcome_file')
            outcome_col = dataset_config.get('outcome_col')
            id_col = dataset_config.get('id_col', 'sampleID')
            dataset_name = dataset_config.get('name')
            
            # Determine outcome type based on task type
            outcome_type = dataset_config.get('outcome_type', 'class' if not is_regression else 'os')
            
            y, clinical_df = load_outcome(".", outcome_file, outcome_col, id_col, outcome_type, dataset_name)
            if y is None:
                logger.error(f"Failed to load outcome data for {dataset_name}")
                return analysis_results
            
            # Find common sample IDs
            from data_io import optimize_sample_intersection
            common_ids, modalities = optimize_sample_intersection(modalities, y, dataset_name)
            
            if modalities is None or y is None or len(common_ids) == 0:
                logger.error(f"Failed to load data for {dataset_name}")
                return analysis_results
            
            logger.info(f"Loaded {len(common_ids)} samples with {len(modalities)} modalities")
            
            # Stage 1: Raw modality data
            analysis_results['stages']['1_raw_modalities'] = {}
            for mod_name, mod_data in modalities.items():
                metrics = self.calculate_data_metrics(mod_data.values, f"{dataset_name}_{mod_name}_raw")
                analysis_results['stages']['1_raw_modalities'][mod_name] = metrics
                logger.info(f"Raw {mod_name}: {metrics['n_samples']}x{metrics['n_features']}, "
                           f"zeros: {metrics['zero_percentage']:.2f}%, missing: {metrics['missing_percentage']:.2f}%")
            
            # Stage 2: Enhanced preprocessing comparison (NEW)
            logger.info("Analyzing enhanced preprocessing improvements...")
            analysis_results['stages']['2_enhanced_preprocessing'] = {}
            
            for mod_name, mod_data in modalities.items():
                try:
                    from preprocessing import robust_biomedical_preprocessing_pipeline
                    
                    # Convert to sklearn format (samples × features)
                    X_original = mod_data.T.values
                    
                    # Apply robust preprocessing
                    modality_type = mod_name.lower().replace(' ', '_')
                    X_enhanced, transformers, preprocessing_report = robust_biomedical_preprocessing_pipeline(
                        X_original, modality_type=modality_type
                    )
                    
                    # Calculate metrics for enhanced data
                    enhanced_metrics = self.calculate_data_metrics(X_enhanced, f"{dataset_name}_{mod_name}_enhanced")
                    
                    # Store results with preprocessing report
                    analysis_results['stages']['2_enhanced_preprocessing'][mod_name] = {
                        'enhanced_metrics': enhanced_metrics,
                        'preprocessing_report': preprocessing_report,
                        'improvement_summary': {
                            'sparsity_reduction': (
                                analysis_results['stages']['1_raw_modalities'][mod_name]['zero_percentage'] - 
                                enhanced_metrics['zero_percentage']
                            ),
                            'skewness_improvement': abs(enhanced_metrics.get('skewness', 0)) < abs(analysis_results['stages']['1_raw_modalities'][mod_name].get('skewness', 0))
                        }
                    }
                    
                    # Add robust scaling effectiveness analysis (NEW)
                    if 'robust_scaling' in preprocessing_report:
                        robust_scaling_report = preprocessing_report['robust_scaling']
                        analysis_results['stages']['2_enhanced_preprocessing'][mod_name]['robust_scaling_analysis'] = {
                            'scaling_applied': robust_scaling_report.get('scaling_applied', False),
                            'scaling_method': robust_scaling_report.get('scaling_method', 'unknown'),
                            'variance_reduction_ratio': robust_scaling_report.get('variance_reduction_ratio', np.nan),
                            'outlier_clipping_applied': robust_scaling_report.get('outlier_clipping_applied', False),
                            'clip_range': robust_scaling_report.get('clip_range', None),
                            'quantile_range': robust_scaling_report.get('quantile_range', None),
                            'original_variance_stats': robust_scaling_report.get('original_variance_stats', {}),
                            'scaled_variance_stats': robust_scaling_report.get('scaled_variance_stats', {}),
                            'center_stats': robust_scaling_report.get('center_stats', {}),
                            'scale_stats': robust_scaling_report.get('scale_stats', {})
                        }
                        
                        # Log robust scaling effectiveness
                        if robust_scaling_report.get('scaling_applied', False):
                            variance_reduction = robust_scaling_report.get('variance_reduction_ratio', 1.0)
                            logger.info(f"  Robust scaling: variance reduction ratio = {variance_reduction:.3f}")
                            if variance_reduction < 0.5:
                                logger.info(f"  ✅ Significant variance reduction achieved for {mod_name}")
                            elif variance_reduction > 2.0:
                                logger.warning(f"  ⚠️  Variance increased for {mod_name} - may indicate scaling issues")
                    
                    # Log improvements
                    logger.info(f"Enhanced {mod_name}: {enhanced_metrics['n_samples']}x{enhanced_metrics['n_features']}, "
                               f"zeros: {enhanced_metrics['zero_percentage']:.2f}% "
                               f"(reduced by {analysis_results['stages']['2_enhanced_preprocessing'][mod_name]['improvement_summary']['sparsity_reduction']:.2f}%)")
                    
                    if 'sparsity_handling' in preprocessing_report:
                        sparsity_info = preprocessing_report['sparsity_handling']
                        logger.info(f"  Sparsity: {sparsity_info.get('initial_sparsity', 0):.2%} -> {sparsity_info.get('final_sparsity', 0):.2%}")
                    
                    if 'skewness_correction' in preprocessing_report:
                        skewness_info = preprocessing_report['skewness_correction']
                        logger.info(f"  Skewness: {skewness_info.get('initial_skewness', 0):.3f} -> {skewness_info.get('final_skewness', 0):.3f} ({skewness_info.get('transformation_applied', 'none')})")
                        
                except Exception as e:
                    logger.warning(f"Robust preprocessing analysis failed for {mod_name}: {e}")
                    analysis_results['stages']['2_enhanced_preprocessing'][mod_name] = {
                        'error': str(e)
                    }
            
            # Get extractors and selectors
            if is_regression:
                extractors = get_regression_extractors()
                selectors = get_regression_selectors()
            else:
                extractors = get_classification_extractors()
                selectors = get_classification_selectors()
            
            # Test ALL 7 fusion techniques for comprehensive analysis
            fusion_techniques = [
                "weighted_concat",           # Default baseline
                "learnable_weighted",        # Learnable weights
                "attention_weighted",        # Attention mechanism
                #"late_fusion_stacking",      # Late fusion with stacking
                "mkl",                       # Multiple Kernel Learning
                "snf",                       # Similarity Network Fusion  
                "early_fusion_pca"           # Early fusion with PCA
            ]
            
            # Analyze different n_components values
            for n_components in N_VALUES_LIST:
                stage_key = f"n_components_{n_components}"
                analysis_results['stages'][stage_key] = {}
                
                logger.info(f"Analyzing with n_components={n_components}")
                
                # Create train/test split for consistent analysis
                np.random.seed(42)
                n_samples = len(common_ids)
                train_idx = np.random.choice(n_samples, size=int(0.8 * n_samples), replace=False)
                test_idx = np.setdiff1d(np.arange(n_samples), train_idx)
                
                # Analyze different missing data scenarios
                missing_percentages = MISSING_MODALITIES_CONFIG.get("missing_percentages", [0.0])
                
                for missing_pct in missing_percentages:
                    missing_key = f"missing_{int(missing_pct*100)}pct"
                    analysis_results['stages'][stage_key][missing_key] = {}
                    
                    # Create missing modality scenario
                    scenario_modalities = process_with_missing_modalities(
                        modalities, common_ids, missing_pct, random_state=42
                    )
                    
                    # Analyze fusion techniques
                    for fusion_technique in fusion_techniques:
                        fusion_key = f"fusion_{fusion_technique}"
                        analysis_results['stages'][stage_key][missing_key][fusion_key] = {}
                        
                        logger.info(f"  Fusion: {fusion_technique}, Missing: {missing_pct*100:.0f}%")
                        
                        try:
                            # Process modalities with enhanced preprocessing and imputation
                            imputer = ModalityImputer(strategy='median')
                            processed_modalities = []
                            preprocessing_reports = {}
                            
                            for mod_name, mod_data in scenario_modalities.items():
                                if mod_data is not None and not mod_data.empty:
                                    # Get actual sample IDs for train/test split
                                    available_ids = [id for id in common_ids if id in mod_data.columns]
                                    if len(available_ids) < len(train_idx) + len(test_idx):
                                        # Adjust indices to available samples
                                        n_available = len(available_ids)
                                        train_size = int(0.8 * n_available)
                                        train_sample_ids = available_ids[:train_size]
                                        test_sample_ids = available_ids[train_size:]
                                    else:
                                        # Use original indices mapped to sample IDs
                                        train_sample_ids = [common_ids[i] for i in train_idx if common_ids[i] in mod_data.columns]
                                        test_sample_ids = [common_ids[i] for i in test_idx if common_ids[i] in mod_data.columns]
                                    
                                    # Split data using sample IDs (mod_data is features × samples format)
                                    train_data = mod_data[train_sample_ids]  # Features × samples format
                                    test_data = mod_data[test_sample_ids]
                                    
                                    # Apply ROBUST preprocessing (FIXED)
                                    try:
                                        from preprocessing import robust_biomedical_preprocessing_pipeline
                                        
                                        # Convert to sklearn format (samples × features)
                                        X_train = train_data.T.values
                                        X_test = test_data.T.values
                                        
                                        # Apply robust preprocessing with guaranteed consistency
                                        modality_type = mod_name.lower().replace(' ', '_')
                                        X_train_enhanced, X_test_enhanced, transformers, preprocessing_report = robust_biomedical_preprocessing_pipeline(
                                            X_train, X_test, modality_type=modality_type
                                        )
                                        
                                        # Store preprocessing report for analysis
                                        preprocessing_reports[mod_name] = preprocessing_report
                                        
                                        # Log preprocessing improvements
                                        if 'sparsity_handling' in preprocessing_report:
                                            sparsity_info = preprocessing_report['sparsity_handling']
                                            logger.info(f"    {mod_name} sparsity: {sparsity_info.get('initial_sparsity', 0):.2%} -> {sparsity_info.get('final_sparsity', 0):.2%}")
                                        
                                        if 'skewness_correction' in preprocessing_report:
                                            skewness_info = preprocessing_report['skewness_correction']
                                            logger.info(f"    {mod_name} skewness: {skewness_info.get('initial_skewness', 0):.3f} -> {skewness_info.get('final_skewness', 0):.3f} ({skewness_info.get('transformation_applied', 'none')})")
                                        
                                        # CRITICAL FIX: Ensure consistent sample alignment after preprocessing
                                        # Enhanced preprocessing might change sample counts, so we need to realign
                                        min_train_samples = min(len(train_sample_ids), X_train_enhanced.shape[0])
                                        min_test_samples = min(len(test_sample_ids), X_test_enhanced.shape[0])
                                        
                                        # Truncate to consistent sample counts
                                        X_train_enhanced = X_train_enhanced[:min_train_samples]
                                        X_test_enhanced = X_test_enhanced[:min_test_samples]
                                        
                                        # Impute any remaining missing values
                                        train_imputed = imputer.fit_transform(X_train_enhanced)
                                        test_imputed = imputer.transform(X_test_enhanced)
                                        processed_modalities.append((train_imputed, test_imputed))
                                        
                                    except Exception as e:
                                        logger.warning(f"Robust preprocessing failed for {mod_name}, falling back to basic processing: {e}")
                                        # Fallback to original processing
                                        try:
                                            train_imputed = imputer.fit_transform(train_data.T.values)
                                            test_imputed = imputer.transform(test_data.T.values)
                                            processed_modalities.append((train_imputed, test_imputed))
                                        except Exception as e2:
                                            logger.warning(f"Imputation failed for {mod_name}: {e2}")
                                            continue
                            
                            if not processed_modalities:
                                logger.warning(f"No valid modalities for {fusion_technique}")
                                continue
                            
                            # Apply fusion with consistent alignment
                            train_arrays = [mod[0] for mod in processed_modalities]
                            test_arrays = [mod[1] for mod in processed_modalities]
                            
                            # CRITICAL FIX: Ensure all arrays have consistent sample counts
                            if train_arrays:
                                min_train_samples = min(arr.shape[0] for arr in train_arrays)
                                train_arrays = [arr[:min_train_samples] for arr in train_arrays]
                                
                            if test_arrays:
                                min_test_samples = min(arr.shape[0] for arr in test_arrays)
                                test_arrays = [arr[:min_test_samples] for arr in test_arrays]
                            
                            # CRITICAL FIX: Ensure consistent feature alignment between train/test
                            # This prevents StandardScaler feature mismatch errors
                            if train_arrays and test_arrays:
                                for i in range(len(train_arrays)):
                                    if i < len(test_arrays):
                                        train_features = train_arrays[i].shape[1]
                                        test_features = test_arrays[i].shape[1]
                                        
                                        if train_features != test_features:
                                            # Align to minimum feature count
                                            min_features = min(train_features, test_features)
                                            logger.debug(f"Aligning modality {i}: train {train_features} -> {min_features}, test {test_features} -> {min_features}")
                                            train_arrays[i] = train_arrays[i][:, :min_features]
                                            test_arrays[i] = test_arrays[i][:, :min_features]
                            
                            # Adjust y arrays to match the aligned sample counts
                            y_train_full = y[train_idx] if hasattr(y, '__getitem__') else y.iloc[train_idx]
                            y_test_full = y[test_idx] if hasattr(y, '__getitem__') else y.iloc[test_idx]
                            
                            if train_arrays:
                                y_train = y_train_full[:min_train_samples] if hasattr(y_train_full, '__getitem__') else y_train_full.iloc[:min_train_samples]
                            else:
                                y_train = y_train_full
                                
                            if test_arrays:
                                y_test = y_test_full[:min_test_samples] if hasattr(y_test_full, '__getitem__') else y_test_full.iloc[:min_test_samples]
                            else:
                                y_test = y_test_full
                            
                            # Merge modalities with enhanced error handling and optimized SNF parameters
                            try:
                                # Configure optimized fusion parameters for SNF
                                fusion_params = {}
                                if fusion_technique == "snf":
                                    fusion_params = {
                                        'K': 30,  # Increased neighbors to reduce sparsity
                                        'alpha': 0.8,  # Higher alpha for stronger fusion
                                        'T': 30,  # More iterations for convergence
                                        'mu': 0.8,  # Higher variance parameter
                                        'sigma': None,  # Auto-computed based on data
                                        'distance_metrics': ['euclidean', 'cosine', 'correlation'],
                                        'adaptive_neighbors': True,  # Adapt K based on data characteristics
                                        'random_state': 42
                                    }
                                
                                if fusion_technique in ["learnable_weighted", "attention_weighted", "late_fusion_stacking", "mkl", "snf", "early_fusion_pca"]:
                                    train_result = merge_modalities(
                                        *train_arrays, strategy=fusion_technique, 
                                        is_train=True, n_components=n_components, 
                                        y=y_train, is_regression=is_regression,
                                        fusion_params=fusion_params
                                    )
                                    if isinstance(train_result, tuple):
                                        X_train_fused, fitted_fusion = train_result
                                    else:
                                        X_train_fused = train_result
                                        fitted_fusion = None
                                    
                                    X_test_fused = merge_modalities(
                                        *test_arrays, strategy=fusion_technique,
                                        is_train=False, fitted_fusion=fitted_fusion,
                                        y=y_test, is_regression=is_regression,
                                        fusion_params=fusion_params
                                    )
                                    if isinstance(X_test_fused, tuple):
                                        X_test_fused = X_test_fused[0]
                                else:
                                    X_train_fused = merge_modalities(
                                        *train_arrays, strategy=fusion_technique, is_train=True
                                    )
                                    X_test_fused = merge_modalities(
                                        *test_arrays, strategy=fusion_technique, is_train=False
                                    )
                                
                                # Final alignment check after fusion
                                if X_train_fused.shape[0] != y_train.shape[0]:
                                    min_samples = min(X_train_fused.shape[0], y_train.shape[0])
                                    logger.debug(f"Final train alignment: {X_train_fused.shape[0]} -> {min_samples}")
                                    X_train_fused = X_train_fused[:min_samples]
                                    y_train = y_train[:min_samples] if hasattr(y_train, '__getitem__') else y_train.iloc[:min_samples]
                                
                                if X_test_fused.shape[0] != y_test.shape[0]:
                                    min_samples = min(X_test_fused.shape[0], y_test.shape[0])
                                    logger.debug(f"Final test alignment: {X_test_fused.shape[0]} -> {min_samples}")
                                    X_test_fused = X_test_fused[:min_samples]
                                    y_test = y_test[:min_samples] if hasattr(y_test, '__getitem__') else y_test.iloc[:min_samples]
                                    
                            except Exception as fusion_error:
                                logger.warning(f"Fusion failed for {fusion_technique}: {fusion_error}")
                                # Fallback to simple concatenation
                                try:
                                    X_train_fused = np.column_stack(train_arrays) if train_arrays else np.array([]).reshape(0, 0)
                                    X_test_fused = np.column_stack(test_arrays) if test_arrays else np.array([]).reshape(0, 0)
                                    logger.info(f"Using concatenation fallback for {fusion_technique}")
                                except Exception as fallback_error:
                                    logger.error(f"Even concatenation fallback failed: {fallback_error}")
                                    continue
                            
                            # Analyze fused data
                            fused_metrics_train = self.calculate_data_metrics(
                                X_train_fused, f"{dataset_name}_{fusion_technique}_train"
                            )
                            fused_metrics_test = self.calculate_data_metrics(
                                X_test_fused, f"{dataset_name}_{fusion_technique}_test"
                            )
                            
                            analysis_results['stages'][stage_key][missing_key][fusion_key]['fused'] = {
                                'train': fused_metrics_train,
                                'test': fused_metrics_test
                            }
                            
                            # Add preprocessing reports to results (NEW)
                            if preprocessing_reports:
                                analysis_results['stages'][stage_key][missing_key][fusion_key]['preprocessing_reports'] = preprocessing_reports
                            
                            logger.info(f"    Fused: {fused_metrics_train['n_samples']}x{fused_metrics_train['n_features']}")
                            
                            # Analyze extraction techniques
                            extraction_results = {}
                            for ext_name, extractor in extractors.items():
                                try:
                                    if is_regression:
                                        fitted_ext, X_train_ext = cached_fit_transform_extractor_regression(
                                            X_train_fused, y_train, extractor, n_components,
                                            ds_name=dataset_name, fold_idx=0
                                        )
                                        X_test_ext = transform_extractor_regression(X_test_fused, fitted_ext)
                                    else:
                                        fitted_ext, X_train_ext = cached_fit_transform_extractor_classification(
                                            X_train_fused, y_train, extractor, n_components,
                                            ds_name=dataset_name, fold_idx=0
                                        )
                                        X_test_ext = transform_extractor_classification(X_test_fused, fitted_ext)
                                    
                                    if X_train_ext is not None and X_test_ext is not None:
                                        ext_metrics_train = self.calculate_data_metrics(
                                            X_train_ext, f"{dataset_name}_{ext_name}_train"
                                        )
                                        ext_metrics_test = self.calculate_data_metrics(
                                            X_test_ext, f"{dataset_name}_{ext_name}_test"
                                        )
                                        
                                        extraction_results[ext_name] = {
                                            'train': ext_metrics_train,
                                            'test': ext_metrics_test
                                        }
                                        
                                        logger.info(f"      {ext_name}: {ext_metrics_train['n_features']} features")
                                        
                                except Exception as e:
                                    logger.warning(f"Extraction failed for {ext_name}: {e}")
                                    continue
                            
                            analysis_results['stages'][stage_key][missing_key][fusion_key]['extraction'] = extraction_results
                            
                            # Analyze selection techniques
                            selection_results = {}
                            for sel_name, selector_code in selectors.items():
                                try:
                                    if is_regression:
                                        selected_features, X_train_sel = cached_fit_transform_selector_regression(
                                            selector_code, X_train_fused, y_train, n_components,
                                            fold_idx=0, ds_name=dataset_name
                                        )
                                        X_test_sel = transform_selector_regression(X_test_fused, selected_features)
                                    else:
                                        selected_features, X_train_sel = cached_fit_transform_selector_classification(
                                            X_train_fused, y_train, selector_code, n_components,
                                            ds_name=dataset_name, modality_name=None, fold_idx=0
                                        )
                                        X_test_sel = transform_selector_classification(X_test_fused, selected_features)
                                    
                                    if X_train_sel is not None and X_test_sel is not None:
                                        sel_metrics_train = self.calculate_data_metrics(
                                            X_train_sel, f"{dataset_name}_{sel_name}_train"
                                        )
                                        sel_metrics_test = self.calculate_data_metrics(
                                            X_test_sel, f"{dataset_name}_{sel_name}_test"
                                        )
                                        
                                        selection_results[sel_name] = {
                                            'train': sel_metrics_train,
                                            'test': sel_metrics_test
                                        }
                                        
                                        logger.info(f"      {sel_name}: {sel_metrics_train['n_features']} features")
                                        
                                except Exception as e:
                                    logger.warning(f"Selection failed for {sel_name}: {e}")
                                    continue
                            
                            analysis_results['stages'][stage_key][missing_key][fusion_key]['selection'] = selection_results
                            
                        except Exception as e:
                            logger.error(f"Error analyzing {fusion_technique}: {e}")
                            continue
                            
        except Exception as e:
            logger.error(f"Error analyzing {dataset_name}: {e}")
            
        return analysis_results
    
    def save_results(self, results: Dict[str, Any], is_regression: bool = True) -> None:
        """
        Save analysis results to files.
        
        Parameters
        ----------
        results : Dict[str, Any]
            Analysis results
        is_regression : bool
            Whether this is regression or classification
        """
        task_type = "regression" if is_regression else "classification"
        dataset_name = results['dataset_name']
        
        # Save detailed JSON results
        json_path = self.output_dir / task_type / f"{dataset_name}_detailed_analysis.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create summary CSV
        summary_data = []
        
        for stage_name, stage_data in results.get('stages', {}).items():
            if isinstance(stage_data, dict):
                for scenario_name, scenario_data in stage_data.items():
                    if isinstance(scenario_data, dict):
                        for technique_name, technique_data in scenario_data.items():
                            if isinstance(technique_data, dict):
                                for process_type, process_data in technique_data.items():
                                    if isinstance(process_data, dict):
                                        for method_name, method_data in process_data.items():
                                            if isinstance(method_data, dict) and 'train' in method_data:
                                                train_metrics = method_data['train']
                                                test_metrics = method_data['test']
                                                
                                                row = {
                                                    'dataset': dataset_name,
                                                    'stage': stage_name,
                                                    'scenario': scenario_name,
                                                    'technique': technique_name,
                                                    'process_type': process_type,
                                                    'method': method_name,
                                                    'split': 'train',
                                                    **train_metrics
                                                }
                                                summary_data.append(row)
                                                
                                                row = {
                                                    'dataset': dataset_name,
                                                    'stage': stage_name,
                                                    'scenario': scenario_name,
                                                    'technique': technique_name,
                                                    'process_type': process_type,
                                                    'method': method_name,
                                                    'split': 'test',
                                                    **test_metrics
                                                }
                                                summary_data.append(row)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            csv_path = self.output_dir / task_type / f"{dataset_name}_summary.csv"
            summary_df.to_csv(csv_path, index=False)
        
        logger.info(f"Saved results for {dataset_name} to {json_path} and CSV summary")
    
    def generate_overall_summary(self) -> None:
        """Generate an overall summary across all datasets."""
        logger.info("Generating overall summary")
        
        summary_stats = {
            'regression': [],
            'classification': []
        }
        
        # Collect summaries from both task types
        for task_type in ['regression', 'classification']:
            task_dir = self.output_dir / task_type
            if task_dir.exists():
                for csv_file in task_dir.glob("*_summary.csv"):
                    try:
                        df = pd.read_csv(csv_file)
                        df['task_type'] = task_type
                        summary_stats[task_type].append(df)
                    except Exception as e:
                        logger.warning(f"Error reading {csv_file}: {e}")
        
        # Combine and save overall summary
        all_data = []
        for task_type, dfs in summary_stats.items():
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                all_data.append(combined_df)
        
        if all_data:
            overall_df = pd.concat(all_data, ignore_index=True)
            overall_path = self.output_dir / "summary" / "overall_data_quality_summary.csv"
            overall_df.to_csv(overall_path, index=False)
            
            # Generate summary statistics (including robust scaling metrics)
            numeric_cols = ['zero_percentage', 'missing_percentage', 'mean', 'std', 
                           'n_features', 'n_samples', 'outlier_percentage']
            
            # Add robust scaling metrics if available
            scaling_cols = ['variance_reduction_ratio', 'pca_variance_std', 'pca_variance_ratio', 
                           'pca_variance_std_improvement', 'pca_variance_ratio_improvement']
            
            # Check which columns actually exist in the data
            available_cols = [col for col in numeric_cols + scaling_cols if col in overall_df.columns]
            
            stats_summary = overall_df.groupby(['task_type', 'technique', 'process_type', 'method'])[available_cols].agg([
                'mean', 'std', 'min', 'max', 'median'
            ]).round(4)
            
            stats_path = self.output_dir / "summary" / "summary_statistics.csv"
            stats_summary.to_csv(stats_path)
            
            logger.info(f"Generated overall summary with {len(overall_df)} records")
            
            # Generate robust scaling effectiveness report (NEW)
            self.generate_robust_scaling_report(overall_df)
            
        else:
            logger.warning("No data found for summary generation")
    
    def generate_robust_scaling_report(self, overall_df: pd.DataFrame) -> None:
        """
        Generate a specific report on robust scaling effectiveness.
        
        Parameters
        ----------
        overall_df : pd.DataFrame
            Overall data quality summary
        """
        try:
            logger.info("Generating robust scaling effectiveness report")
            
            # Filter for scaling effectiveness data
            scaling_data = []
            
            # Look for scaling effectiveness metrics in the data
            scaling_metrics = overall_df[overall_df.columns[overall_df.columns.str.contains('scaling|variance_reduction|pca_variance', case=False, na=False)]]
            
            if not scaling_metrics.empty:
                scaling_summary = {
                    'total_datasets_analyzed': len(overall_df['dataset'].unique()) if 'dataset' in overall_df.columns else 0,
                    'datasets_with_scaling_data': len(scaling_metrics['dataset'].unique()) if 'dataset' in scaling_metrics.columns else 0,
                    'scaling_effectiveness_summary': {}
                }
                
                # Analyze by modality type if available
                if 'technique' in overall_df.columns:
                    for technique in overall_df['technique'].unique():
                        technique_data = overall_df[overall_df['technique'] == technique]
                        
                        # Look for variance reduction metrics
                        variance_cols = [col for col in technique_data.columns if 'variance' in col.lower()]
                        if variance_cols:
                            scaling_summary['scaling_effectiveness_summary'][technique] = {
                                'samples_analyzed': len(technique_data),
                                'variance_metrics_available': variance_cols
                            }
                            
                            # Calculate average improvements if data is available
                            for col in variance_cols:
                                if technique_data[col].notna().any():
                                    scaling_summary['scaling_effectiveness_summary'][technique][f'{col}_stats'] = {
                                        'mean': float(technique_data[col].mean()),
                                        'median': float(technique_data[col].median()),
                                        'std': float(technique_data[col].std()),
                                        'min': float(technique_data[col].min()),
                                        'max': float(technique_data[col].max())
                                    }
                
                # Save robust scaling report
                scaling_report_path = self.output_dir / "summary" / "robust_scaling_effectiveness_report.json"
                with open(scaling_report_path, 'w') as f:
                    json.dump(scaling_summary, f, indent=2, default=str)
                
                logger.info(f"Robust scaling effectiveness report saved to {scaling_report_path}")
                
                # Log key findings
                if scaling_summary['datasets_with_scaling_data'] > 0:
                    logger.info(f"📊 Robust Scaling Analysis Results:")
                    logger.info(f"   • Datasets analyzed: {scaling_summary['total_datasets_analyzed']}")
                    logger.info(f"   • Datasets with scaling data: {scaling_summary['datasets_with_scaling_data']}")
                    logger.info(f"   • Techniques analyzed: {len(scaling_summary['scaling_effectiveness_summary'])}")
                else:
                    logger.warning("No robust scaling effectiveness data found in results")
                    
            else:
                logger.warning("No scaling metrics found in overall data")
                
        except Exception as e:
            logger.error(f"Error generating robust scaling report: {e}")

    def analyze_kpls_stability(self, X: np.ndarray, y: np.ndarray, name: str = "unknown") -> Dict[str, Any]:
        """
        Analyze KPLS numerical stability and validate improvements.
        
        This method tests the KPLS implementation with different configurations
        to validate that the fixes for extreme values and numerical instability work.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        name : str
            Name for logging purposes
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing KPLS stability analysis results
        """
        logger.info(f"🔍 Analyzing KPLS stability for {name}")
        
        results = {
            'dataset_name': name,
            'analysis_timestamp': datetime.now().isoformat(),
            'input_shape': X.shape,
            'target_shape': y.shape,
            'configurations_tested': {},
            'stability_metrics': {},
            'recommendations': []
        }
        
        try:
            from models import KernelPLSRegression
            
            # Test configurations: old vs new
            configs = {
                'old_config': {
                    'n_components': 8,
                    'regularization': 0.0,
                    'use_cv_components': False,
                    'gamma_bounds': (1e-10, 1e10),
                    'tol': 1e-6
                },
                'new_improved': {
                    'n_components': 5,
                    'regularization': 1e-6,
                    'use_cv_components': True,
                    'cv_folds': 3,
                    'gamma_bounds': (1e-6, 1e3),
                    'tol': 1e-4
                },
                'conservative': {
                    'n_components': 3,
                    'regularization': 1e-5,
                    'use_cv_components': True,
                    'cv_folds': 3,
                    'gamma_bounds': (1e-5, 1e2),
                    'tol': 1e-3
                }
            }
            
            for config_name, config_params in configs.items():
                logger.info(f"  Testing {config_name} configuration...")
                
                config_results = {
                    'parameters': config_params,
                    'success': False,
                    'fit_time': None,
                    'transform_values': {},
                    'prediction_values': {},
                    'errors': [],
                    'warnings': []
                }
                
                try:
                    import time
                    start_time = time.time()
                    
                    # Create KPLS with current configuration
                    kpls = KernelPLSRegression(**config_params, random_state=42)
                    
                    # Fit the model
                    kpls.fit(X, y)
                    fit_time = time.time() - start_time
                    config_results['fit_time'] = fit_time
                    
                    # Transform data
                    X_transformed = kpls.transform(X)
                    
                    # Analyze transformed values
                    config_results['transform_values'] = {
                        'shape': X_transformed.shape,
                        'mean': float(np.mean(X_transformed)),
                        'std': float(np.std(X_transformed)),
                        'min': float(np.min(X_transformed)),
                        'max': float(np.max(X_transformed)),
                        'has_extreme_values': bool(np.any(np.abs(X_transformed) > 50)),
                        'has_nan': bool(np.any(np.isnan(X_transformed))),
                        'has_inf': bool(np.any(np.isinf(X_transformed)))
                    }
                    
                    # Test predictions
                    y_pred = kpls.predict(X)
                    config_results['prediction_values'] = {
                        'shape': y_pred.shape,
                        'mean': float(np.mean(y_pred)),
                        'std': float(np.std(y_pred)),
                        'min': float(np.min(y_pred)),
                        'max': float(np.max(y_pred)),
                        'has_extreme_values': bool(np.any(np.abs(y_pred) > 1000)),
                        'has_nan': bool(np.any(np.isnan(y_pred))),
                        'has_inf': bool(np.any(np.isinf(y_pred)))
                    }
                    
                    # Check for numerical stability issues
                    if hasattr(kpls, 'optimal_components_'):
                        config_results['optimal_components_selected'] = kpls.optimal_components_
                    
                    if hasattr(kpls, 'gamma_'):
                        config_results['gamma_computed'] = float(kpls.gamma_)
                    
                    config_results['success'] = True
                    logger.info(f"    ✅ {config_name}: Success (fit_time={fit_time:.3f}s)")
                    
                except Exception as e:
                    error_msg = str(e)
                    config_results['errors'].append(error_msg)
                    logger.warning(f"    ❌ {config_name}: Failed - {error_msg}")
                
                results['configurations_tested'][config_name] = config_results
            
            # Analyze stability across configurations
            successful_configs = [name for name, result in results['configurations_tested'].items() 
                                if result['success']]
            
            results['stability_metrics'] = {
                'successful_configurations': len(successful_configs),
                'total_configurations': len(configs),
                'success_rate': len(successful_configs) / len(configs),
                'fastest_config': None,
                'most_stable_config': None,
                'extreme_value_issues': []
            }
            
            # Find fastest and most stable configurations
            if successful_configs:
                fit_times = {name: results['configurations_tested'][name]['fit_time'] 
                           for name in successful_configs}
                results['stability_metrics']['fastest_config'] = min(fit_times, key=fit_times.get)
                
                # Check for extreme values
                for config_name in successful_configs:
                    config_result = results['configurations_tested'][config_name]
                    if config_result['transform_values']['has_extreme_values']:
                        results['stability_metrics']['extreme_value_issues'].append(
                            f"{config_name}: transform values > 50"
                        )
                    if config_result['prediction_values']['has_extreme_values']:
                        results['stability_metrics']['extreme_value_issues'].append(
                            f"{config_name}: prediction values > 1000"
                        )
                
                # Determine most stable (fewest extreme values, no NaN/Inf)
                stability_scores = {}
                for config_name in successful_configs:
                    config_result = results['configurations_tested'][config_name]
                    score = 0
                    
                    # Penalize extreme values
                    if config_result['transform_values']['has_extreme_values']:
                        score -= 10
                    if config_result['prediction_values']['has_extreme_values']:
                        score -= 10
                    
                    # Penalize NaN/Inf
                    if config_result['transform_values']['has_nan'] or config_result['transform_values']['has_inf']:
                        score -= 20
                    if config_result['prediction_values']['has_nan'] or config_result['prediction_values']['has_inf']:
                        score -= 20
                    
                    # Reward reasonable value ranges
                    transform_range = abs(config_result['transform_values']['max'] - config_result['transform_values']['min'])
                    if 0.1 <= transform_range <= 20:  # Reasonable range
                        score += 5
                    
                    stability_scores[config_name] = score
                
                results['stability_metrics']['most_stable_config'] = max(stability_scores, key=stability_scores.get)
            
            # Generate recommendations
            if results['stability_metrics']['success_rate'] == 1.0:
                results['recommendations'].append("✅ All KPLS configurations working successfully")
            elif results['stability_metrics']['success_rate'] >= 0.5:
                results['recommendations'].append("⚠️ Some KPLS configurations failing, use successful ones")
            else:
                results['recommendations'].append("❌ Most KPLS configurations failing, investigate data issues")
            
            if results['stability_metrics']['extreme_value_issues']:
                results['recommendations'].append(
                    f"⚠️ Extreme value issues detected: {', '.join(results['stability_metrics']['extreme_value_issues'])}"
                )
            else:
                results['recommendations'].append("✅ No extreme value issues detected")
            
            if 'new_improved' in successful_configs:
                results['recommendations'].append("✅ New improved KPLS configuration working correctly")
            else:
                results['recommendations'].append("❌ New improved KPLS configuration failed")
            
        except Exception as e:
            logger.error(f"KPLS stability analysis failed for {name}: {str(e)}")
            results['error'] = str(e)
            results['recommendations'].append(f"❌ KPLS analysis failed: {str(e)}")
        
        return results

    def analyze_sparse_pls_variance(self, X: np.ndarray, y: np.ndarray, name: str = "unknown") -> Dict[str, Any]:
        """
        Analyze SparsePLS variance and overfitting issues.
        
        This method tests different SparsePLS configurations to validate that
        the optimizations for high variance and overfitting are working.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        name : str
            Name for logging purposes
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing SparsePLS variance analysis results
        """
        logger.info(f"🔍 Analyzing SparsePLS variance for {name}")
        
        results = {
            'dataset_name': name,
            'analysis_timestamp': datetime.now().isoformat(),
            'input_shape': X.shape,
            'target_shape': y.shape,
            'configurations_tested': {},
            'variance_metrics': {},
            'recommendations': []
        }
        
        try:
            from models import SparsePLS
            
            # Test configurations: old vs new optimized
            configs = {
                'old_high_variance': {
                    'n_components': 8,
                    'alpha': 0.1,
                    'max_iter': 1000,
                    'description': 'Old configuration prone to overfitting'
                },
                'current_default': {
                    'n_components': 5,
                    'alpha': 0.1,
                    'max_iter': 1000,
                    'description': 'Previous default configuration'
                },
                'new_optimized': {
                    'n_components': 3,
                    'alpha': 0.3,
                    'max_iter': 500,
                    'description': 'New optimized configuration for variance control'
                },
                'conservative': {
                    'n_components': 2,
                    'alpha': 0.5,
                    'max_iter': 500,
                    'description': 'Conservative configuration for high-variance data'
                }
            }
            
            for config_name, config_params in configs.items():
                logger.info(f"  Testing {config_name} configuration...")
                
                config_results = {
                    'parameters': config_params,
                    'success': False,
                    'fit_time': None,
                    'component_variances': [],
                    'transform_variance': None,
                    'sparsity_achieved': None,
                    'n_components_fitted': None,
                    'overfitting_indicators': {},
                    'errors': []
                }
                
                try:
                    import time
                    start_time = time.time()
                    
                    # Create SparsePLS with current configuration
                    sparse_pls = SparsePLS(
                        n_components=config_params['n_components'],
                        alpha=config_params['alpha'],
                        max_iter=config_params['max_iter'],
                        scale=True
                    )
                    
                    # Fit the model
                    sparse_pls.fit(X, y)
                    fit_time = time.time() - start_time
                    config_results['fit_time'] = fit_time
                    
                    # Transform data
                    X_transformed = sparse_pls.transform(X)
                    
                    # Analyze component variances
                    if hasattr(sparse_pls, 'component_variances_'):
                        config_results['component_variances'] = sparse_pls.component_variances_
                        
                        # Check for high variance (overfitting indicator)
                        max_variance = max(sparse_pls.component_variances_) if sparse_pls.component_variances_ else 0
                        mean_variance = np.mean(sparse_pls.component_variances_) if sparse_pls.component_variances_ else 0
                        
                        config_results['overfitting_indicators'] = {
                            'max_component_variance': max_variance,
                            'mean_component_variance': mean_variance,
                            'high_variance_components': sum(1 for v in sparse_pls.component_variances_ if v > 50),
                            'variance_std': np.std(sparse_pls.component_variances_) if len(sparse_pls.component_variances_) > 1 else 0
                        }
                    
                    # Analyze transform variance
                    transform_variances = np.var(X_transformed, axis=0)
                    config_results['transform_variance'] = {
                        'max': float(np.max(transform_variances)),
                        'mean': float(np.mean(transform_variances)),
                        'std': float(np.std(transform_variances))
                    }
                    
                    # Analyze sparsity achieved
                    if hasattr(sparse_pls, 'x_weights_'):
                        weights = sparse_pls.x_weights_
                        sparsity_ratio = np.mean(np.abs(weights) < 1e-6)
                        config_results['sparsity_achieved'] = float(sparsity_ratio)
                    
                    # Number of components actually fitted
                    if hasattr(sparse_pls, 'n_components_fitted_'):
                        config_results['n_components_fitted'] = sparse_pls.n_components_fitted_
                    else:
                        config_results['n_components_fitted'] = sparse_pls.x_weights_.shape[1] if hasattr(sparse_pls, 'x_weights_') else 0
                    
                    config_results['success'] = True
                    logger.info(f"    ✅ {config_name}: Success (fit_time={fit_time:.3f}s)")
                    
                except Exception as e:
                    error_msg = str(e)
                    config_results['errors'].append(error_msg)
                    logger.warning(f"    ❌ {config_name}: Failed - {error_msg}")
                
                results['configurations_tested'][config_name] = config_results
            
            # Analyze variance improvements across configurations
            successful_configs = [name for name, result in results['configurations_tested'].items() 
                                if result['success']]
            
            results['variance_metrics'] = {
                'successful_configurations': len(successful_configs),
                'total_configurations': len(configs),
                'success_rate': len(successful_configs) / len(configs),
                'variance_improvements': {},
                'sparsity_improvements': {},
                'overfitting_reduction': {}
            }
            
            # Compare variance metrics between configurations
            if len(successful_configs) >= 2:
                baseline_config = 'old_high_variance' if 'old_high_variance' in successful_configs else successful_configs[0]
                optimized_config = 'new_optimized' if 'new_optimized' in successful_configs else successful_configs[-1]
                
                if baseline_config != optimized_config:
                    baseline_result = results['configurations_tested'][baseline_config]
                    optimized_result = results['configurations_tested'][optimized_config]
                    
                    # Compare transform variances
                    if baseline_result.get('transform_variance') and optimized_result.get('transform_variance'):
                        baseline_var = baseline_result['transform_variance']['max']
                        optimized_var = optimized_result['transform_variance']['max']
                        
                        variance_reduction = (baseline_var - optimized_var) / baseline_var if baseline_var > 0 else 0
                        results['variance_metrics']['variance_improvements'] = {
                            'baseline_max_variance': baseline_var,
                            'optimized_max_variance': optimized_var,
                            'variance_reduction_ratio': variance_reduction
                        }
                    
                    # Compare sparsity
                    baseline_sparsity = baseline_result.get('sparsity_achieved', 0)
                    optimized_sparsity = optimized_result.get('sparsity_achieved', 0)
                    
                    results['variance_metrics']['sparsity_improvements'] = {
                        'baseline_sparsity': baseline_sparsity,
                        'optimized_sparsity': optimized_sparsity,
                        'sparsity_increase': optimized_sparsity - baseline_sparsity
                    }
                    
                    # Compare overfitting indicators
                    baseline_overfitting = baseline_result.get('overfitting_indicators', {})
                    optimized_overfitting = optimized_result.get('overfitting_indicators', {})
                    
                    results['variance_metrics']['overfitting_reduction'] = {
                        'baseline_max_component_var': baseline_overfitting.get('max_component_variance', 0),
                        'optimized_max_component_var': optimized_overfitting.get('max_component_variance', 0),
                        'baseline_high_var_components': baseline_overfitting.get('high_variance_components', 0),
                        'optimized_high_var_components': optimized_overfitting.get('high_variance_components', 0)
                    }
            
            # Generate recommendations
            variance_improvements = results['variance_metrics'].get('variance_improvements', {})
            sparsity_improvements = results['variance_metrics'].get('sparsity_improvements', {})
            overfitting_reduction = results['variance_metrics'].get('overfitting_reduction', {})
            
            if variance_improvements.get('variance_reduction_ratio', 0) > 0.1:
                results['recommendations'].append("✅ Significant variance reduction achieved with optimized configuration")
            elif variance_improvements.get('variance_reduction_ratio', 0) > 0:
                results['recommendations'].append("⚠️ Modest variance reduction achieved, consider further optimization")
            else:
                results['recommendations'].append("❌ No variance reduction detected, investigate data characteristics")
            
            if sparsity_improvements.get('sparsity_increase', 0) > 0.1:
                results['recommendations'].append("✅ Increased sparsity achieved, reducing overfitting risk")
            
            if overfitting_reduction.get('optimized_high_var_components', 0) < overfitting_reduction.get('baseline_high_var_components', 0):
                results['recommendations'].append("✅ Reduced high-variance components, overfitting risk decreased")
            
            if 'new_optimized' in successful_configs:
                results['recommendations'].append("✅ New optimized SparsePLS configuration working correctly")
            else:
                results['recommendations'].append("❌ New optimized SparsePLS configuration failed")
            
            # Check for remaining high variance issues
            for config_name in successful_configs:
                config_result = results['configurations_tested'][config_name]
                max_transform_var = config_result.get('transform_variance', {}).get('max', 0)
                
                if max_transform_var > 80:  # High variance threshold
                    results['recommendations'].append(f"⚠️ {config_name} still shows high variance ({max_transform_var:.1f}), consider more aggressive sparsity")
                elif max_transform_var > 19:  # Medium variance threshold
                    results['recommendations'].append(f"⚠️ {config_name} shows moderate variance ({max_transform_var:.1f}), monitor for overfitting")
                else:
                    results['recommendations'].append(f"✅ {config_name} shows controlled variance ({max_transform_var:.1f})")
            
        except Exception as e:
            logger.error(f"SparsePLS variance analysis failed for {name}: {str(e)}")
            results['error'] = str(e)
            results['recommendations'].append(f"❌ SparsePLS analysis failed: {str(e)}")
        
        return results


def main():
    """Main function to run the data quality analysis."""
    logger.info("Starting Comprehensive Data Quality Analysis")
    logger.info("This will analyze all 9 datasets with all algorithm combinations")
    
    # Initialize analyzer
    analyzer = DataQualityAnalyzer()
    
    # Log algorithm information
    logger.info("=" * 70)
    logger.info("ALGORITHM CONFIGURATION")
    logger.info("=" * 70)
    
    # Get algorithm lists for logging
    reg_extractors = get_regression_extractors()
    reg_selectors = get_regression_selectors()
    clf_extractors = get_classification_extractors()
    clf_selectors = get_classification_selectors()
    
    logger.info(f"REGRESSION EXTRACTORS ({len(reg_extractors)}): {list(reg_extractors.keys())}")
    logger.info(f"REGRESSION SELECTORS ({len(reg_selectors)}): {list(reg_selectors.keys())}")
    logger.info(f"CLASSIFICATION EXTRACTORS ({len(clf_extractors)}): {list(clf_extractors.keys())}")
    logger.info(f"CLASSIFICATION SELECTORS ({len(clf_selectors)}): {list(clf_selectors.keys())}")
    
    # Test ALL 7 fusion techniques for comprehensive analysis
    fusion_techniques = [
        "weighted_concat",           # Default baseline
        "learnable_weighted",        # Learnable weights
        "attention_weighted",        # Attention mechanism
        #"late_fusion_stacking",      # Late fusion with stacking
        "mkl",                       # Multiple Kernel Learning
        "snf",                       # Similarity Network Fusion  
        "early_fusion_pca"           # Early fusion with PCA
    ]
    logger.info(f"FUSION TECHNIQUES ({len(fusion_techniques)}): {fusion_techniques}")
    logger.info(f"N_COMPONENTS VALUES: {N_VALUES_LIST}")
    
    total_combinations_reg = len(reg_extractors) * len(reg_selectors) * len(fusion_techniques) * len(N_VALUES_LIST)
    total_combinations_clf = len(clf_extractors) * len(clf_selectors) * len(fusion_techniques) * len(N_VALUES_LIST)
    
    logger.info(f"TOTAL COMBINATIONS PER REGRESSION DATASET: {total_combinations_reg}")
    logger.info(f"TOTAL COMBINATIONS PER CLASSIFICATION DATASET: {total_combinations_clf}")
    logger.info(f"TOTAL DATASETS: {len(REGRESSION_DATASETS_FOR_ANALYSIS)} regression + {len(CLASSIFICATION_DATASETS_FOR_ANALYSIS)} classification = {len(REGRESSION_DATASETS_FOR_ANALYSIS) + len(CLASSIFICATION_DATASETS_FOR_ANALYSIS)}")
    
    # Analyze regression datasets
    logger.info("=" * 70)
    logger.info("ANALYZING REGRESSION DATASETS")
    logger.info("=" * 70)
    
    for i, dataset_config in enumerate(REGRESSION_DATASETS_FOR_ANALYSIS, 1):
        logger.info(f"Processing regression dataset {i}/{len(REGRESSION_DATASETS_FOR_ANALYSIS)}: {dataset_config['name']}")
        try:
            results = analyzer.analyze_dataset_quality(dataset_config, is_regression=True)
            analyzer.save_results(results, is_regression=True)
        except Exception as e:
            logger.error(f"Failed to analyze regression dataset {dataset_config.get('name', 'unknown')}: {e}")
    
    # Analyze classification datasets
    logger.info("=" * 70)
    logger.info("ANALYZING CLASSIFICATION DATASETS")
    logger.info("=" * 70)
    
    for i, dataset_config in enumerate(CLASSIFICATION_DATASETS_FOR_ANALYSIS, 1):
        logger.info(f"Processing classification dataset {i}/{len(CLASSIFICATION_DATASETS_FOR_ANALYSIS)}: {dataset_config['name']}")
        try:
            results = analyzer.analyze_dataset_quality(dataset_config, is_regression=False)
            analyzer.save_results(results, is_regression=False)
        except Exception as e:
            logger.error(f"Failed to analyze classification dataset {dataset_config.get('name', 'unknown')}: {e}")
    
    # Generate overall summary
    logger.info("=" * 70)
    logger.info("GENERATING OVERALL SUMMARY")
    logger.info("=" * 70)
    analyzer.generate_overall_summary()
    
    logger.info("=" * 70)
    logger.info("DATA QUALITY ANALYSIS COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"Results saved to: {analyzer.output_dir}")
    logger.info("Summary files:")
    logger.info(f"  - Overall: {analyzer.output_dir}/summary/overall_data_quality_summary.csv")
    logger.info(f"  - Statistics: {analyzer.output_dir}/summary/summary_statistics.csv")
    logger.info(f"  - Robust Scaling Report: {analyzer.output_dir}/summary/robust_scaling_effectiveness_report.json")
    logger.info(f"  - Regression: {analyzer.output_dir}/regression/")
    logger.info(f"  - Classification: {analyzer.output_dir}/classification/")
    logger.info("")
    logger.info(" ROBUST SCALING ANALYSIS INCLUDED:")
    logger.info("   • PCA variance comparison (StandardScaler vs RobustScaler)")
    logger.info("   • Variance reduction ratio tracking")
    logger.info("   • Outlier clipping effectiveness")
    logger.info("   • Modality-specific scaling parameter analysis")
    logger.info("   • Scaling effectiveness recommendations")


if __name__ == "__main__":
    main() 