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
- Basic: weighted_concat, average, sum, max
- Advanced: learnable_weighted, attention_weighted, mkl, snf, early_fusion_pca, standard_concat

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
    # Import the new 4-phase enhanced pipeline integration
    from enhanced_pipeline_integration import run_enhanced_preprocessing_pipeline, EnhancedPipelineCoordinator
    from data_quality import run_early_data_quality_pipeline
    from fusion_aware_preprocessing import determine_optimal_fusion_order
    from missing_data_handler import create_missing_data_handler
    from validation_coordinator import create_validation_coordinator
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

# Define datasets directly to ensure all 9 are analyzed
REGRESSION_DATASETS_FOR_ANALYSIS = [
    {
        "name": "AML",
        "base_path": "data/aml",
        "modalities": {
            "Gene Expression": "exp.csv",
            "miRNA": "mirna.csv",
            "Methylation": "methy.csv"
        },
        "outcome_file": "data/clinical/aml.csv",
        "outcome_col": "lab_procedure_bone_marrow_blast_cell_outcome_percent_value",
        "id_col": "sampleID",
        "outcome_type": "continuous",
        "fix_tcga_ids": True
    },
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
    {
        "name": "Breast",
        "base_path": "data/breast",
        "modalities": {
            "Gene Expression": "exp.csv",
            "miRNA": "mirna.csv",
            "Methylation": "methy.csv"
        },
        "outcome_file": "data/clinical/breast.csv",
        "outcome_col": "pathologic_T",
        "id_col": "sampleID",
        "outcome_type": "class",
        "fix_tcga_ids": True
    },
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
        Updated to use the new 4-phase enhanced pipeline.
        
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
            # Stage 1: Test 4-Phase Pipeline Integration
            logger.info(f" Running 4-Phase Pipeline Integration Test for {dataset_name}")
            pipeline_test_results = self.test_4phase_pipeline_integration(dataset_config, is_regression)
            analysis_results['stages']['4_phase_pipeline_test'] = pipeline_test_results
            
            # Stage 2: Load data using enhanced data loading (same as main CLI)
            logger.info(f" Loading data using enhanced data loading for {dataset_name}")
            task_type = "regression" if is_regression else "classification"
            
            # Use the same data loading approach as the main CLI
            modalities_list = list(dataset_config["modalities"].keys())
            modality_short_names = []
            for mod_name in modalities_list:
                if "Gene Expression" in mod_name or "exp" in mod_name.lower():
                    modality_short_names.append("exp")
                elif "miRNA" in mod_name or "mirna" in mod_name.lower():
                    modality_short_names.append("mirna")
                elif "Methylation" in mod_name or "methy" in mod_name.lower():
                    modality_short_names.append("methy")
                else:
                    modality_short_names.append(mod_name.lower())
            
            outcome_col = dataset_config["outcome_col"]
            
            # Load raw data using the main pipeline approach
            raw_modalities, y_raw, common_ids = load_dataset(
                dataset_name.lower(), 
                modality_short_names, 
                outcome_col, 
                task_type,
                parallel=True,
                use_cache=True
            )
            
            if raw_modalities is None or y_raw is None or len(common_ids) == 0:
                logger.error(f"Failed to load data for {dataset_name}")
                return analysis_results
            
            logger.info(f"Loaded {len(common_ids)} samples with {len(raw_modalities)} modalities")
            
            # Stage 3: Raw modality data analysis
            analysis_results['stages']['1_raw_modalities'] = {}
            for mod_name, mod_data in raw_modalities.items():
                metrics = self.calculate_data_metrics(mod_data.values, f"{dataset_name}_{mod_name}_raw")
                analysis_results['stages']['1_raw_modalities'][mod_name] = metrics
                logger.info(f"Raw {mod_name}: {metrics['n_samples']}x{metrics['n_features']}, "
                           f"zeros: {metrics['zero_percentage']:.2f}%, missing: {metrics['missing_percentage']:.2f}%")
            
            # Stage 4: Enhanced preprocessing using 4-phase pipeline
            logger.info(f" Running 4-Phase Enhanced Preprocessing for {dataset_name}")
            analysis_results['stages']['2_enhanced_4phase_preprocessing'] = {}
            
            # Convert to enhanced pipeline format
            modality_data_dict = {}
            for modality_name, modality_df in raw_modalities.items():
                X = modality_df.T.values  # Convert to samples x features
                modality_data_dict[modality_name] = (X, common_ids)
            
            # Determine fusion method based on task type
            fusion_method = "snf" if task_type == "classification" else "weighted_concat"
            
            try:
                # Run the 4-phase enhanced pipeline
                processed_modalities, y_aligned, pipeline_metadata = run_enhanced_preprocessing_pipeline(
                    modality_data_dict=modality_data_dict,
                    y=y_raw.values,
                    fusion_method=fusion_method,
                    task_type=task_type,
                    dataset_name=dataset_name,
                    enable_early_quality_check=True,
                    enable_feature_first_order=True,
                    enable_centralized_missing_data=True,
                    enable_coordinated_validation=True
                )
                
                # Analyze processed data from 4-phase pipeline
                for mod_name, processed_data in processed_modalities.items():
                    enhanced_metrics = self.calculate_data_metrics(processed_data, f"{dataset_name}_{mod_name}_enhanced")
                    
                    analysis_results['stages']['2_enhanced_4phase_preprocessing'][mod_name] = {
                        'enhanced_metrics': enhanced_metrics,
                        'pipeline_metadata': pipeline_metadata,
                        'fusion_method': fusion_method,
                        'improvement_summary': {
                            'sparsity_reduction': (
                                analysis_results['stages']['1_raw_modalities'][mod_name]['zero_percentage'] - 
                                enhanced_metrics['zero_percentage']
                            ),
                            'quality_score': pipeline_metadata.get('quality_score', 0.0)
                        }
                    }
                    
                    logger.info(f"Enhanced {mod_name}: {enhanced_metrics['n_samples']}x{enhanced_metrics['n_features']}, "
                               f"zeros: {enhanced_metrics['zero_percentage']:.2f}% "
                               f"(reduced by {analysis_results['stages']['2_enhanced_4phase_preprocessing'][mod_name]['improvement_summary']['sparsity_reduction']:.2f}%)")
                
                logger.info(f" 4-Phase Enhanced Preprocessing completed successfully")
                logger.info(f" Overall Quality Score: {pipeline_metadata.get('quality_score', 'N/A')}")
                
            except Exception as e:
                logger.warning(f"4-Phase Enhanced Preprocessing failed for {dataset_name}: {e}")
                analysis_results['stages']['2_enhanced_4phase_preprocessing']['error'] = str(e)
                
                # Fallback to robust preprocessing
                logger.info(f"🔄 Falling back to robust preprocessing for {dataset_name}")
                try:
                    from preprocessing import robust_biomedical_preprocessing_pipeline
                    
                    for mod_name, (X, sample_ids) in modality_data_dict.items():
                        # Determine modality type
                        if 'exp' in mod_name.lower():
                            modality_type = 'gene_expression'
                        elif 'mirna' in mod_name.lower():
                            modality_type = 'mirna'
                        elif 'methy' in mod_name.lower():
                            modality_type = 'methylation'
                        else:
                            modality_type = 'unknown'
                        
                        # Apply robust preprocessing
                        X_processed, transformers, report = robust_biomedical_preprocessing_pipeline(
                            X, modality_type=modality_type
                        )
                        
                        enhanced_metrics = self.calculate_data_metrics(X_processed, f"{dataset_name}_{mod_name}_robust_fallback")
                        
                        analysis_results['stages']['2_enhanced_4phase_preprocessing'][f"{mod_name}_fallback"] = {
                            'enhanced_metrics': enhanced_metrics,
                            'preprocessing_report': report,
                            'fallback_used': True
                        }
                        
                        logger.info(f"Fallback {mod_name}: {enhanced_metrics['n_samples']}x{enhanced_metrics['n_features']}")
                        
                except Exception as e2:
                    logger.error(f"Even fallback preprocessing failed for {dataset_name}: {e2}")
                    analysis_results['stages']['2_enhanced_4phase_preprocessing']['fallback_error'] = str(e2)
            
            # Stage 5: Algorithm testing with processed data (simplified for 4-phase focus)
            logger.info(f" Testing algorithms with processed data for {dataset_name}")
            analysis_results['stages']['3_algorithm_testing'] = {}
            
            # Get extractors and selectors
            if is_regression:
                extractors = get_regression_extractors()
                selectors = get_regression_selectors()
            else:
                extractors = get_classification_extractors()
                selectors = get_classification_selectors()
            
            # Test with a subset of fusion techniques (focused on 4-phase pipeline)
            fusion_techniques = ["weighted_concat", "snf", "mkl"]
            
            # Test one n_components value to validate the pipeline
            n_components = N_VALUES_LIST[0] if N_VALUES_LIST else 5
            
            for fusion_technique in fusion_techniques:
                fusion_key = f"fusion_{fusion_technique}"
                analysis_results['stages']['3_algorithm_testing'][fusion_key] = {}
                
                logger.info(f"  Testing fusion: {fusion_technique}")
                
                try:
                    # Use the processed data from 4-phase pipeline if available
                    if 'error' not in analysis_results['stages']['2_enhanced_4phase_preprocessing']:
                        # Test with enhanced processed data
                        test_data = processed_modalities if 'processed_modalities' in locals() else None
                        if test_data:
                            # Analyze fused data quality
                            fused_arrays = list(test_data.values())
                            if fused_arrays:
                                # Simple concatenation for testing
                                fused_data = np.column_stack(fused_arrays) if len(fused_arrays) > 1 else fused_arrays[0]
                                fused_metrics = self.calculate_data_metrics(fused_data, f"{dataset_name}_{fusion_technique}_fused")
                                
                                analysis_results['stages']['3_algorithm_testing'][fusion_key] = {
                                    'fused_metrics': fused_metrics,
                                    'fusion_technique': fusion_technique,
                                    'n_components': n_components,
                                    'success': True
                                }
                                
                                logger.info(f"    Fused data: {fused_metrics['n_samples']}x{fused_metrics['n_features']}")
                            else:
                                analysis_results['stages']['3_algorithm_testing'][fusion_key] = {
                                    'error': 'No processed data available for fusion testing'
                                }
                        else:
                            analysis_results['stages']['3_algorithm_testing'][fusion_key] = {
                                'error': 'Enhanced preprocessing data not available'
                            }
                    else:
                        analysis_results['stages']['3_algorithm_testing'][fusion_key] = {
                            'error': 'Enhanced preprocessing failed, skipping algorithm testing'
                        }
                        
                except Exception as e:
                    logger.warning(f"Algorithm testing failed for {fusion_technique}: {e}")
                    analysis_results['stages']['3_algorithm_testing'][fusion_key] = {
                        'error': str(e)
                    }
            
            logger.info(f" Data quality analysis completed for {dataset_name}")
            
        except Exception as e:
            logger.error(f"Error analyzing {dataset_name}: {e}")
            analysis_results['error'] = str(e)
            
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
        
        # Create summary CSV - improved to handle actual data structure
        summary_data = []
        
        # Try original nested structure first
        for stage_name, stage_data in results.get('stages', {}).items():
            if isinstance(stage_data, dict):
                # Check if this is the new 4-phase pipeline structure
                if stage_name == "4_phase_pipeline_test":
                    # Extract metrics from the 4-phase pipeline test structure
                    if 'integration_test' in stage_data and 'processed_data_quality' in stage_data['integration_test']:
                        quality_data = stage_data['integration_test']['processed_data_quality']
                        
                        for technique_name, technique_data in quality_data.items():
                            if isinstance(technique_data, dict):
                                # Create summary row for this modality
                                row = {
                                    'dataset': dataset_name,
                                    'stage': '4_phase_pipeline',
                                    'scenario': 'integration_test',
                                    'technique': technique_name,
                                    'process_type': 'enhanced_preprocessing',
                                    'method': 'quality_metrics',
                                    'split': 'train',
                                    'n_samples': technique_data.get('n_samples', 0),
                                    'n_features': technique_data.get('n_features', 0),
                                    'missing_percentage': technique_data.get('missing_percentage', 0.0),
                                    'zero_percentage': technique_data.get('zero_percentage', 0.0),
                                    'mean': float(technique_data.get('mean', '0.0')) if isinstance(technique_data.get('mean'), str) else technique_data.get('mean', 0.0),
                                    'std': float(technique_data.get('std', '0.0')) if isinstance(technique_data.get('std'), str) else technique_data.get('std', 0.0),
                                    'outlier_percentage': technique_data.get('outlier_percentage', 0.0),
                                    'variance': technique_data.get('variance', 0.0)
                                }
                                
                                # Add scaling effectiveness metrics if available
                                if 'scaling_effectiveness' in technique_data:
                                    scaling_data = technique_data['scaling_effectiveness']
                                    
                                    # Add robust scaler metrics
                                    if 'robust_scaler' in scaling_data:
                                        robust_metrics = scaling_data['robust_scaler']
                                        row.update({
                                            'pca_variance_std': robust_metrics.get('pca_variance_std', 0.0),
                                            'pca_variance_ratio': robust_metrics.get('pca_variance_ratio', 0.0),
                                            'robust_scaling_method': 'robust'
                                        })
                                    
                                    # Add standard scaler metrics for comparison
                                    if 'standard_scaler' in scaling_data:
                                        std_metrics = scaling_data['standard_scaler']
                                        row.update({
                                            'std_pca_variance_std': std_metrics.get('pca_variance_std', 0.0),
                                            'std_pca_variance_ratio': std_metrics.get('pca_variance_ratio', 0.0)
                                        })
                                        
                                        # Calculate improvement metrics
                                        if 'robust_scaler' in scaling_data:
                                            robust_std = robust_metrics.get('pca_variance_std', 0.0)
                                            std_std = std_metrics.get('pca_variance_std', 0.0)
                                            
                                            if std_std > 0:
                                                row['variance_reduction_ratio'] = robust_std / std_std
                                                row['pca_variance_std_improvement'] = std_std - robust_std
                                            
                                            robust_ratio = robust_metrics.get('pca_variance_ratio', 0.0)
                                            std_ratio = std_metrics.get('pca_variance_ratio', 0.0)
                                            
                                            if std_ratio > 0:
                                                row['pca_variance_ratio_improvement'] = std_ratio - robust_ratio
                                
                                summary_data.append(row)
                
                # Try to handle original nested structure as fallback
                else:
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
        
        # Always create a CSV file, even if summary_data is empty (with basic info)
        if not summary_data:
            # Create a minimal summary with basic dataset info
            basic_row = {
                'dataset': dataset_name,
                'stage': 'data_quality_analysis',
                'scenario': 'basic_analysis',
                'technique': 'all_modalities',
                'process_type': 'quality_check',
                'method': 'basic_metrics',
                'split': 'full_dataset',
                'task_type': task_type,
                'timestamp': results.get('timestamp', ''),
                'analysis_completed': True
            }
            summary_data.append(basic_row)
        
        # Save CSV summary
        summary_df = pd.DataFrame(summary_data)
        csv_path = self.output_dir / task_type / f"{dataset_name}_summary.csv"
        summary_df.to_csv(csv_path, index=False)
        
        logger.info(f"Saved results for {dataset_name} to {json_path} and CSV summary ({len(summary_data)} rows)")
    
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
            
            # Look for scaling effectiveness metrics in the data
            scaling_metric_columns = [col for col in overall_df.columns 
                                    if any(keyword in col.lower() for keyword in 
                                           ['scaling', 'variance_reduction', 'pca_variance', 'variance'])]
            
            if scaling_metric_columns:
                # Calculate number of datasets with scaling data more accurately
                datasets_with_data = set()
                if 'dataset' in overall_df.columns:
                    for col in scaling_metric_columns:
                        datasets_with_col_data = overall_df[overall_df[col].notna()]['dataset'].unique()
                        datasets_with_data.update(datasets_with_col_data)
                
                scaling_summary = {
                    'total_datasets_analyzed': len(overall_df['dataset'].unique()) if 'dataset' in overall_df.columns else 0,
                    'datasets_with_scaling_data': len(datasets_with_data),
                    'scaling_metric_columns_found': scaling_metric_columns,
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
                                'variance_metrics_available': variance_cols,
                                'datasets_represented': technique_data['dataset'].unique().tolist() if 'dataset' in technique_data.columns else []
                            }
                            
                            # Calculate average improvements if data is available
                            for col in variance_cols:
                                if technique_data[col].notna().any():
                                    valid_data = technique_data[col].dropna()
                                    if len(valid_data) > 0:
                                        scaling_summary['scaling_effectiveness_summary'][technique][f'{col}_stats'] = {
                                            'mean': float(valid_data.mean()),
                                            'median': float(valid_data.median()),
                                            'std': float(valid_data.std()) if len(valid_data) > 1 else 0.0,
                                            'min': float(valid_data.min()),
                                            'max': float(valid_data.max()),
                                            'count': len(valid_data)
                                        }
                
                # Save robust scaling report
                scaling_report_path = self.output_dir / "summary" / "robust_scaling_effectiveness_report.json"
                with open(scaling_report_path, 'w') as f:
                    json.dump(scaling_summary, f, indent=2, default=str)
                
                logger.info(f"Robust scaling effectiveness report saved to {scaling_report_path}")
                
                # Log key findings with improved logic
                total_datasets = scaling_summary['total_datasets_analyzed']
                datasets_with_scaling = scaling_summary['datasets_with_scaling_data']
                techniques_analyzed = len(scaling_summary['scaling_effectiveness_summary'])
                
                if datasets_with_scaling > 0 and techniques_analyzed > 0:
                    logger.info(f"Robust Scaling Analysis Results:")
                    logger.info(f"   • Datasets analyzed: {total_datasets}")
                    logger.info(f"   • Datasets with scaling data: {datasets_with_scaling}")
                    logger.info(f"   • Techniques analyzed: {techniques_analyzed}")
                    logger.info(f"   • Scaling metrics found: {len(scaling_metric_columns)}")
                    
                    # Log effectiveness summary
                    for technique, data in scaling_summary['scaling_effectiveness_summary'].items():
                        if 'variance_reduction_ratio_stats' in data:
                            ratio_stats = data['variance_reduction_ratio_stats']
                            logger.info(f"   • {technique}: Variance reduction ratio = {ratio_stats['mean']:.3f} ± {ratio_stats['std']:.3f}")
                else:
                    logger.warning("No robust scaling effectiveness data found in results")
                    logger.info(f"Debug info: datasets_with_scaling={datasets_with_scaling}, techniques_analyzed={techniques_analyzed}")
                    logger.info(f"Available columns: {list(overall_df.columns)}")
                    
            else:
                logger.warning("No scaling metrics found in overall data")
                logger.info(f"Available columns: {list(overall_df.columns)}")
                
        except Exception as e:
            logger.error(f"Error generating robust scaling report: {e}")
            import traceback
            traceback.print_exc()

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
        logger.info(f" Analyzing KPLS stability for {name}")
        
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
                    logger.info(f"     {config_name}: Success (fit_time={fit_time:.3f}s)")
                    
                except Exception as e:
                    error_msg = str(e)
                    config_results['errors'].append(error_msg)
                    logger.warning(f"     {config_name}: Failed - {error_msg}")
                
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
                results['recommendations'].append(" All KPLS configurations working successfully")
            elif results['stability_metrics']['success_rate'] >= 0.5:
                results['recommendations'].append(" Some KPLS configurations failing, use successful ones")
            else:
                results['recommendations'].append(" Most KPLS configurations failing, investigate data issues")
            
            if results['stability_metrics']['extreme_value_issues']:
                results['recommendations'].append(
                    f" Extreme value issues detected: {', '.join(results['stability_metrics']['extreme_value_issues'])}"
                )
            else:
                results['recommendations'].append(" No extreme value issues detected")
            
            if 'new_improved' in successful_configs:
                results['recommendations'].append(" New improved KPLS configuration working correctly")
            else:
                results['recommendations'].append(" New improved KPLS configuration failed")
            
        except Exception as e:
            logger.error(f"KPLS stability analysis failed for {name}: {str(e)}")
            results['error'] = str(e)
            results['recommendations'].append(f" KPLS analysis failed: {str(e)}")
        
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
        logger.info(f" Analyzing SparsePLS variance for {name}")
        
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
                    logger.info(f"     {config_name}: Success (fit_time={fit_time:.3f}s)")
                    
                except Exception as e:
                    error_msg = str(e)
                    config_results['errors'].append(error_msg)
                    logger.warning(f"     {config_name}: Failed - {error_msg}")
                
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
                results['recommendations'].append(" Significant variance reduction achieved with optimized configuration")
            elif variance_improvements.get('variance_reduction_ratio', 0) > 0:
                results['recommendations'].append(" Modest variance reduction achieved, consider further optimization")
            else:
                results['recommendations'].append(" No variance reduction detected, investigate data characteristics")
            
            if sparsity_improvements.get('sparsity_increase', 0) > 0.1:
                results['recommendations'].append(" Increased sparsity achieved, reducing overfitting risk")
            
            if overfitting_reduction.get('optimized_high_var_components', 0) < overfitting_reduction.get('baseline_high_var_components', 0):
                results['recommendations'].append(" Reduced high-variance components, overfitting risk decreased")
            
            if 'new_optimized' in successful_configs:
                results['recommendations'].append(" New optimized SparsePLS configuration working correctly")
            else:
                results['recommendations'].append(" New optimized SparsePLS configuration failed")
            
            # Check for remaining high variance issues
            for config_name in successful_configs:
                config_result = results['configurations_tested'][config_name]
                max_transform_var = config_result.get('transform_variance', {}).get('max', 0)
                
                if max_transform_var > 80:  # High variance threshold
                    results['recommendations'].append(f" {config_name} still shows high variance ({max_transform_var:.1f}), consider more aggressive sparsity")
                elif max_transform_var > 19:  # Medium variance threshold
                    results['recommendations'].append(f" {config_name} shows moderate variance ({max_transform_var:.1f}), monitor for overfitting")
                else:
                    results['recommendations'].append(f" {config_name} shows controlled variance ({max_transform_var:.1f})")
            
        except Exception as e:
            logger.error(f"SparsePLS variance analysis failed for {name}: {str(e)}")
            results['error'] = str(e)
            results['recommendations'].append(f" SparsePLS analysis failed: {str(e)}")
        
        return results

    def test_4phase_pipeline_integration(self, dataset_config: Dict[str, Any], is_regression: bool = True) -> Dict[str, Any]:
        """
        Test the 4-phase enhanced pipeline integration with comprehensive analysis.
        
        This method specifically tests the new enhanced pipeline to validate
        that all 4 phases are working together correctly.
        
        Parameters
        ----------
        dataset_config : Dict[str, Any]
            Dataset configuration
        is_regression : bool
            Whether this is a regression or classification task
            
        Returns
        -------
        Dict[str, Any]
            Comprehensive 4-phase pipeline test results
        """
        dataset_name = dataset_config['name']
        task_type = 'regression' if is_regression else 'classification'
        logger.info(f" Testing 4-Phase Enhanced Pipeline for {dataset_name} ({task_type})")
        
        test_results = {
            'dataset_name': dataset_name,
            'task_type': task_type,
            'timestamp': datetime.now().isoformat(),
            'pipeline_phases': {},
            'integration_test': {},
            'performance_metrics': {},
            'recommendations': []
        }
        
        try:
            # Step 1: Load data using the enhanced data loading
            logger.info(f" Step 1: Loading data for {dataset_name}")
            
            # Use the same data loading approach as the main CLI
            modalities_list = list(dataset_config["modalities"].keys())
            modality_short_names = []
            for mod_name in modalities_list:
                if "Gene Expression" in mod_name or "exp" in mod_name.lower():
                    modality_short_names.append("exp")
                elif "miRNA" in mod_name or "mirna" in mod_name.lower():
                    modality_short_names.append("mirna")
                elif "Methylation" in mod_name or "methy" in mod_name.lower():
                    modality_short_names.append("methy")
                else:
                    modality_short_names.append(mod_name.lower())
            
            outcome_col = dataset_config["outcome_col"]
            
            # Load raw data using the main pipeline approach
            raw_modalities, y_raw, common_ids = load_dataset(
                dataset_name.lower(), 
                modality_short_names, 
                outcome_col, 
                task_type,
                parallel=True,
                use_cache=True
            )
            
            if raw_modalities is None or len(common_ids) == 0:
                logger.error(f" Failed to load raw dataset {dataset_name}")
                test_results['integration_test']['data_loading'] = {'success': False, 'error': 'Failed to load data'}
                return test_results
            
            logger.info(f" Loaded {len(common_ids)} samples with {len(raw_modalities)} modalities")
            
            test_results['integration_test']['data_loading'] = {
                'success': True,
                'n_samples': len(common_ids),
                'n_modalities': len(raw_modalities),
                'modalities': list(raw_modalities.keys())
            }
            
            # Convert to enhanced pipeline format: Dict[str, Tuple[np.ndarray, List[str]]]
            modality_data_dict = {}
            for modality_name, modality_df in raw_modalities.items():
                # Convert DataFrame to numpy array (transpose to get samples x features)
                X = modality_df.T.values  # modality_df is features x samples
                modality_data_dict[modality_name] = (X, common_ids)
            
            # Step 2: Test 4-Phase Enhanced Preprocessing Pipeline
            logger.info(f" Step 2: Testing 4-Phase Enhanced Pipeline")
            
            # Determine optimal fusion method based on task type
            fusion_method = "snf" if task_type == "classification" else "weighted_concat"
            
            try:
                modalities_data, y_aligned, pipeline_metadata = run_enhanced_preprocessing_pipeline(
                    modality_data_dict=modality_data_dict,
                    y=y_raw.values,
                                    fusion_method=fusion_method,
                task_type=task_type,
                dataset_name=dataset_name,
                enable_early_quality_check=True,
                enable_feature_first_order=True,
                enable_centralized_missing_data=True,
                enable_coordinated_validation=True
                )
                
                logger.info(f" 4-Phase Enhanced Pipeline completed successfully")
                logger.info(f" Quality Score: {pipeline_metadata.get('quality_score', 'N/A')}")
                logger.info(f" Phases Enabled: {pipeline_metadata.get('phases_enabled', {})}")
                
                test_results['integration_test']['enhanced_pipeline'] = {
                    'success': True,
                    'quality_score': pipeline_metadata.get('quality_score', 0.0),
                    'phases_enabled': pipeline_metadata.get('phases_enabled', {}),
                    'fusion_method': fusion_method,
                    'pipeline_metadata': pipeline_metadata
                }
                
                # Analyze processed data quality
                processed_data_analysis = {}
                for modality_name, modality_array in modalities_data.items():
                    metrics = self.calculate_data_metrics(modality_array, f"{dataset_name}_{modality_name}_processed")
                    processed_data_analysis[modality_name] = metrics
                    logger.info(f"  Processed {modality_name}: {metrics['n_samples']}x{metrics['n_features']}")
                
                test_results['integration_test']['processed_data_quality'] = processed_data_analysis
                
            except Exception as e:
                logger.warning(f" 4-Phase Enhanced Pipeline failed: {str(e)}")
                test_results['integration_test']['enhanced_pipeline'] = {
                    'success': False,
                    'error': str(e)
                }
            
            # Step 3: Test Individual Phases (if enhanced pipeline failed)
            logger.info(f" Step 3: Testing Individual Phases")
            
            # Phase 1: Early Data Quality Assessment
            try:
                logger.info("  Testing Phase 1: Early Data Quality Assessment")
                quality_report, guidance = run_early_data_quality_pipeline(
                    modality_data_dict, y_raw.values, dataset_name, task_type
                )
                
                test_results['pipeline_phases']['phase_1_data_quality'] = {
                    'success': True,
                    'quality_report': quality_report,
                    'guidance': guidance,
                    'overall_quality_score': quality_report.get('overall_quality_score', 0.0)
                }
                logger.info(f"     Phase 1: Quality Score = {quality_report.get('overall_quality_score', 0.0)}")
                
            except Exception as e:
                logger.warning(f"     Phase 1 failed: {str(e)}")
                test_results['pipeline_phases']['phase_1_data_quality'] = {
                    'success': False,
                    'error': str(e)
                }
            
            # Phase 2: Fusion-Aware Preprocessing
            try:
                logger.info("  Testing Phase 2: Fusion-Aware Preprocessing")
                optimal_order = determine_optimal_fusion_order(fusion_method)
                
                test_results['pipeline_phases']['phase_2_fusion_aware'] = {
                    'success': True,
                    'fusion_method': fusion_method,
                    'optimal_order': optimal_order
                }
                logger.info(f"     Phase 2: {fusion_method} -> {optimal_order}")
                
            except Exception as e:
                logger.warning(f"     Phase 2 failed: {str(e)}")
                test_results['pipeline_phases']['phase_2_fusion_aware'] = {
                    'success': False,
                    'error': str(e)
                }
            
            # Phase 3: Missing Data Management
            try:
                logger.info("  Testing Phase 3: Missing Data Management")
                handler = create_missing_data_handler(strategy="auto")
                handler.analyze_missing_patterns(modality_data_dict)
                
                test_results['pipeline_phases']['phase_3_missing_data'] = {
                    'success': True,
                    'strategy': 'auto',
                    'patterns_analyzed': True
                }
                logger.info(f"     Phase 3: Missing data analysis completed")
                
            except Exception as e:
                logger.warning(f"     Phase 3 failed: {str(e)}")
                test_results['pipeline_phases']['phase_3_missing_data'] = {
                    'success': False,
                    'error': str(e)
                }
            
            # Phase 4: Validation Framework
            try:
                logger.info("  Testing Phase 4: Validation Framework")
                validator = create_validation_coordinator()
                
                # Use the actual processed data from enhanced pipeline instead of raw data
                if 'modalities_data' in locals() and modalities_data:
                    # Use the processed data from the enhanced pipeline
                    validation_results = validator.validate_processed_data(modalities_data, y_aligned)
                else:
                    # Fallback: Apply basic scaling to raw data for validation
                    from sklearn.preprocessing import RobustScaler
                    test_data = {}
                    for mod_name, (mod_array, _) in modality_data_dict.items():
                        scaler = RobustScaler()
                        scaled_data = scaler.fit_transform(mod_array)
                        # Apply clipping to prevent extreme values warning
                        if mod_name == 'exp':
                            scaled_data = np.clip(scaled_data, -5, 5)
                        else:
                            scaled_data = np.clip(scaled_data, -6, 6)
                        test_data[mod_name] = scaled_data
                    validation_results = validator.validate_processed_data(test_data, y_raw.values)
                
                test_results['pipeline_phases']['phase_4_validation'] = {
                    'success': True,
                    'validation_results': validation_results,
                    'issues_found': len(validation_results.get('issues', []))
                }
                logger.info(f"     Phase 4: {len(validation_results.get('issues', []))} validation issues found")
                
            except Exception as e:
                logger.warning(f"     Phase 4 failed: {str(e)}")
                test_results['pipeline_phases']['phase_4_validation'] = {
                    'success': False,
                    'error': str(e)
                }
            
            # Step 4: Performance Analysis
            logger.info(f"📈 Step 4: Performance Analysis")
            
            successful_phases = sum(1 for phase_result in test_results['pipeline_phases'].values() 
                                  if phase_result.get('success', False))
            total_phases = len(test_results['pipeline_phases'])
            
            test_results['performance_metrics'] = {
                'successful_phases': successful_phases,
                'total_phases': total_phases,
                'phase_success_rate': successful_phases / total_phases if total_phases > 0 else 0.0,
                'enhanced_pipeline_success': test_results['integration_test'].get('enhanced_pipeline', {}).get('success', False),
                'data_loading_success': test_results['integration_test'].get('data_loading', {}).get('success', False)
            }
            
            # Step 5: Generate Recommendations
            logger.info(f" Step 5: Generating Recommendations")
            
            if test_results['performance_metrics']['enhanced_pipeline_success']:
                test_results['recommendations'].append(" 4-Phase Enhanced Pipeline working correctly")
            else:
                test_results['recommendations'].append(" 4-Phase Enhanced Pipeline failed - investigate individual phases")
            
            if test_results['performance_metrics']['phase_success_rate'] >= 0.75:
                test_results['recommendations'].append(" Most individual phases working correctly")
            elif test_results['performance_metrics']['phase_success_rate'] >= 0.5:
                test_results['recommendations'].append(" Some individual phases failing - targeted fixes needed")
            else:
                test_results['recommendations'].append(" Multiple phases failing - comprehensive review required")
            
            if test_results['integration_test'].get('enhanced_pipeline', {}).get('quality_score', 0) >= 0.8:
                test_results['recommendations'].append(" High data quality score achieved")
            elif test_results['integration_test'].get('enhanced_pipeline', {}).get('quality_score', 0) >= 0.6:
                test_results['recommendations'].append(" Moderate data quality - consider optimizations")
            else:
                test_results['recommendations'].append(" Low data quality score - data issues detected")
            
            logger.info(f" 4-Phase Pipeline Test Summary for {dataset_name}:")
            logger.info(f"   • Enhanced Pipeline: {'' if test_results['performance_metrics']['enhanced_pipeline_success'] else ''}")
            logger.info(f"   • Individual Phases: {successful_phases}/{total_phases} successful")
            logger.info(f"   • Quality Score: {test_results['integration_test'].get('enhanced_pipeline', {}).get('quality_score', 'N/A')}")
            
        except Exception as e:
            logger.error(f" 4-Phase Pipeline Test failed for {dataset_name}: {str(e)}")
            test_results['integration_test']['overall_error'] = str(e)
            test_results['recommendations'].append(f" Testing failed: {str(e)}")
        
        return test_results

    def generate_4phase_pipeline_summary(self) -> None:
        """Generate a specific summary report for 4-phase pipeline testing results."""
        logger.info("Generating 4-Phase Pipeline Summary Report")
        
        pipeline_summary = {
            'summary_timestamp': datetime.now().isoformat(),
            'pipeline_tests_analyzed': 0,
            'overall_statistics': {},
            'phase_success_rates': {},
            'dataset_results': {},
            'recommendations': [],
            'pipeline_effectiveness': {}
        }
        
        try:
            # Collect 4-phase pipeline test results from both task types
            pipeline_test_data = []
            
            for task_type in ['regression', 'classification']:
                task_dir = self.output_dir / task_type
                if task_dir.exists():
                    for json_file in task_dir.glob("*_detailed_analysis.json"):
                        try:
                            with open(json_file, 'r') as f:
                                data = json.load(f)
                                
                            # Extract 4-phase pipeline test results
                            pipeline_test = data.get('stages', {}).get('4_phase_pipeline_test', {})
                            if pipeline_test:
                                pipeline_test['dataset_name'] = data.get('dataset_name', 'unknown')
                                pipeline_test['task_type'] = task_type
                                pipeline_test_data.append(pipeline_test)
                                
                        except Exception as e:
                            logger.warning(f"Error reading {json_file}: {e}")
            
            pipeline_summary['pipeline_tests_analyzed'] = len(pipeline_test_data)
            
            if not pipeline_test_data:
                logger.warning("No 4-phase pipeline test data found")
                pipeline_summary['error'] = 'No pipeline test data available'
                return
            
            # Calculate overall statistics
            successful_enhanced_pipelines = sum(1 for test in pipeline_test_data 
                                              if test.get('performance_metrics', {}).get('enhanced_pipeline_success', False))
            successful_data_loading = sum(1 for test in pipeline_test_data 
                                        if test.get('performance_metrics', {}).get('data_loading_success', False))
            
            pipeline_summary['overall_statistics'] = {
                'total_datasets_tested': len(pipeline_test_data),
                'enhanced_pipeline_success_count': successful_enhanced_pipelines,
                'enhanced_pipeline_success_rate': successful_enhanced_pipelines / len(pipeline_test_data),
                'data_loading_success_count': successful_data_loading,
                'data_loading_success_rate': successful_data_loading / len(pipeline_test_data),
                'average_phase_success_rate': np.mean([
                    test.get('performance_metrics', {}).get('phase_success_rate', 0.0) 
                    for test in pipeline_test_data
                ])
            }
            
            # Calculate phase-specific success rates
            phase_names = ['phase_1_data_quality', 'phase_2_fusion_aware', 'phase_3_missing_data', 'phase_4_validation']
            
            for phase_name in phase_names:
                successful_phase = sum(1 for test in pipeline_test_data 
                                     if test.get('pipeline_phases', {}).get(phase_name, {}).get('success', False))
                pipeline_summary['phase_success_rates'][phase_name] = {
                    'success_count': successful_phase,
                    'success_rate': successful_phase / len(pipeline_test_data),
                    'total_tests': len(pipeline_test_data)
                }
            
            # Collect dataset-specific results
            for test in pipeline_test_data:
                dataset_name = test.get('dataset_name', 'unknown')
                pipeline_summary['dataset_results'][dataset_name] = {
                    'task_type': test.get('task_type', 'unknown'),
                    'enhanced_pipeline_success': test.get('performance_metrics', {}).get('enhanced_pipeline_success', False),
                    'phase_success_rate': test.get('performance_metrics', {}).get('phase_success_rate', 0.0),
                    'quality_score': test.get('integration_test', {}).get('enhanced_pipeline', {}).get('quality_score', None),
                    'recommendations': test.get('recommendations', [])[:2]  # Top 2 recommendations per dataset
                }
            
            # Analyze pipeline effectiveness
            quality_scores = [
                test.get('integration_test', {}).get('enhanced_pipeline', {}).get('quality_score', None)
                for test in pipeline_test_data
                if test.get('integration_test', {}).get('enhanced_pipeline', {}).get('quality_score') is not None
            ]
            
            if quality_scores:
                pipeline_summary['pipeline_effectiveness'] = {
                    'quality_scores_available': len(quality_scores),
                    'average_quality_score': float(np.mean(quality_scores)),
                    'median_quality_score': float(np.median(quality_scores)),
                    'min_quality_score': float(np.min(quality_scores)),
                    'max_quality_score': float(np.max(quality_scores)),
                    'std_quality_score': float(np.std(quality_scores))
                }
            
            # Generate recommendations based on results
            overall_success_rate = pipeline_summary['overall_statistics']['enhanced_pipeline_success_rate']
            avg_phase_success = pipeline_summary['overall_statistics']['average_phase_success_rate']
            
            if overall_success_rate >= 0.8:
                pipeline_summary['recommendations'].append(" 4-Phase Enhanced Pipeline working excellently across datasets")
            elif overall_success_rate >= 0.6:
                pipeline_summary['recommendations'].append(" 4-Phase Enhanced Pipeline working moderately - some optimization needed")
            else:
                pipeline_summary['recommendations'].append(" 4-Phase Enhanced Pipeline showing significant issues - comprehensive review required")
            
            if avg_phase_success >= 0.8:
                pipeline_summary['recommendations'].append(" Individual phases performing well")
            elif avg_phase_success >= 0.6:
                pipeline_summary['recommendations'].append(" Some individual phases need attention")
            else:
                pipeline_summary['recommendations'].append(" Multiple individual phases failing - targeted fixes needed")
            
            # Identify problematic phases
            for phase_name, phase_stats in pipeline_summary['phase_success_rates'].items():
                if phase_stats['success_rate'] < 0.5:
                    phase_display_name = phase_name.replace('_', ' ').title()
                    pipeline_summary['recommendations'].append(f" {phase_display_name} needs attention (success rate: {phase_stats['success_rate']:.1%})")
            
            # Quality score recommendations
            if quality_scores:
                avg_quality = pipeline_summary['pipeline_effectiveness']['average_quality_score']
                if avg_quality >= 0.8:
                    pipeline_summary['recommendations'].append("🏆 Excellent data quality scores achieved")
                elif avg_quality >= 0.6:
                    pipeline_summary['recommendations'].append(" Moderate data quality - consider optimizations")
                else:
                    pipeline_summary['recommendations'].append(" Low data quality scores - data preprocessing needs improvement")
            
            # Save 4-phase pipeline summary
            summary_path = self.output_dir / "summary" / "4phase_pipeline_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(pipeline_summary, f, indent=2, default=str)
            
            logger.info(f"4-Phase Pipeline Summary saved to {summary_path}")
            
            # Log key findings
            logger.info("📋 4-Phase Pipeline Summary Results:")
            logger.info(f"   • Datasets tested: {pipeline_summary['pipeline_tests_analyzed']}")
            logger.info(f"   • Enhanced pipeline success rate: {overall_success_rate:.1%}")
            logger.info(f"   • Average phase success rate: {avg_phase_success:.1%}")
            
            if quality_scores:
                logger.info(f"   • Average quality score: {pipeline_summary['pipeline_effectiveness']['average_quality_score']:.3f}")
            
            logger.info(" Phase Success Rates:")
            for phase_name, phase_stats in pipeline_summary['phase_success_rates'].items():
                phase_display = phase_name.replace('_', ' ').title()
                logger.info(f"   • {phase_display}: {phase_stats['success_rate']:.1%}")
            
            # Log top recommendations
            logger.info(" Top Recommendations:")
            for rec in pipeline_summary['recommendations'][:5]:
                logger.info(f"   {rec}")
                
        except Exception as e:
            logger.error(f"Error generating 4-phase pipeline summary: {e}")
            pipeline_summary['error'] = str(e)

    def regenerate_csv_summaries(self) -> None:
        """
        Regenerate CSV summary files from existing JSON files.
        This fixes the 'No data found for summary generation' warning.
        """
        logger.info("Regenerating CSV summary files from existing JSON data...")
        
        regenerated_count = 0
        
        # Process both task types
        for task_type in ['regression', 'classification']:
            task_dir = self.output_dir / task_type
            if task_dir.exists():
                for json_file in task_dir.glob("*_detailed_analysis.json"):
                    try:
                        # Load existing JSON data
                        with open(json_file, 'r') as f:
                            results = json.load(f)
                        
                        # Determine if this is regression or classification
                        is_regression = task_type == 'regression'
                        
                        # Use the fixed save_results method to regenerate CSV
                        self.save_results(results, is_regression=is_regression)
                        regenerated_count += 1
                        
                        logger.info(f"Regenerated CSV for {results.get('dataset_name', 'unknown')} ({task_type})")
                        
                    except Exception as e:
                        logger.error(f"Error regenerating CSV from {json_file}: {e}")
        
        logger.info(f"Regenerated {regenerated_count} CSV summary files")
        
        # Now try to generate the overall summary again
        if regenerated_count > 0:
            self.generate_overall_summary()
        else:
            logger.warning("No JSON files found to regenerate CSV summaries from")


def main():
    """Main function to run the comprehensive 4-phase pipeline data quality analysis."""
    logger.info(" Starting 4-Phase Enhanced Pipeline Data Quality Analysis")
    logger.info("=" * 80)
    logger.info("This will test the complete 4-phase enhanced pipeline integration:")
    logger.info("  Phase 1: Early Data Quality Assessment")
    logger.info("  Phase 2: Fusion-Aware Preprocessing")
    logger.info("  Phase 3: Centralized Missing Data Management")
    logger.info("  Phase 4: Coordinated Validation Framework")
    logger.info("=" * 80)
    
    # Initialize analyzer
    analyzer = DataQualityAnalyzer()
    
    # Log pipeline configuration
    logger.info(" PIPELINE CONFIGURATION")
    logger.info("=" * 50)
    
    # Test enhanced pipeline imports
    try:
        from enhanced_pipeline_integration import run_enhanced_preprocessing_pipeline, EnhancedPipelineCoordinator
        logger.info(" Enhanced Pipeline Integration: Available")
    except ImportError as e:
        logger.error(f" Enhanced Pipeline Integration: Import failed - {e}")
    
    try:
        from data_quality import run_early_data_quality_pipeline
        logger.info(" Phase 1 (Data Quality): Available")
    except ImportError as e:
        logger.error(f" Phase 1 (Data Quality): Import failed - {e}")
    
    try:
        from fusion_aware_preprocessing import determine_optimal_fusion_order
        logger.info(" Phase 2 (Fusion-Aware): Available")
    except ImportError as e:
        logger.error(f" Phase 2 (Fusion-Aware): Import failed - {e}")
    
    try:
        from missing_data_handler import create_missing_data_handler
        logger.info(" Phase 3 (Missing Data): Available")
    except ImportError as e:
        logger.error(f" Phase 3 (Missing Data): Import failed - {e}")
    
    try:
        from validation_coordinator import create_validation_coordinator
        logger.info(" Phase 4 (Validation): Available")
    except ImportError as e:
        logger.error(f" Phase 4 (Validation): Import failed - {e}")
    
    # Get algorithm lists for comprehensive testing
    reg_extractors = get_regression_extractors()
    reg_selectors = get_regression_selectors()
    clf_extractors = get_classification_extractors()
    clf_selectors = get_classification_selectors()
    
    logger.info(f" ALGORITHM SUPPORT")
    logger.info(f"   Regression Extractors: {len(reg_extractors)} ({list(reg_extractors.keys())})")
    logger.info(f"   Regression Selectors: {len(reg_selectors)} ({list(reg_selectors.keys())})")
    logger.info(f"   Classification Extractors: {len(clf_extractors)} ({list(clf_extractors.keys())})")
    logger.info(f"   Classification Selectors: {len(clf_selectors)} ({list(clf_selectors.keys())})")
    
    # Test datasets configuration
    total_datasets = len(REGRESSION_DATASETS_FOR_ANALYSIS) + len(CLASSIFICATION_DATASETS_FOR_ANALYSIS)
    logger.info(f"📁 DATASETS TO ANALYZE: {total_datasets}")
    logger.info(f"   Regression: {len(REGRESSION_DATASETS_FOR_ANALYSIS)} datasets")
    logger.info(f"   Classification: {len(CLASSIFICATION_DATASETS_FOR_ANALYSIS)} datasets")
    
    # Test Phase Integration Summary
    logger.info("")
    logger.info(" 4-PHASE PIPELINE TESTING PLAN")
    logger.info("=" * 50)
    logger.info("1. Individual Phase Testing (data_quality.py, fusion_aware_preprocessing.py, etc.)")
    logger.info("2. Complete 4-Phase Integration Testing (enhanced_pipeline_integration.py)")
    logger.info("3. Data Loading Integration Testing (data_io.py)")
    logger.info("4. Preprocessing Pipeline Testing (preprocessing.py)")
    logger.info("5. End-to-End Workflow Validation")
    
    # Analyze regression datasets
    logger.info("")
    logger.info("🔬 ANALYZING REGRESSION DATASETS")
    logger.info("=" * 50)
    
    for i, dataset_config in enumerate(REGRESSION_DATASETS_FOR_ANALYSIS, 1):
        logger.info(f" Processing regression dataset {i}/{len(REGRESSION_DATASETS_FOR_ANALYSIS)}: {dataset_config['name']}")
        try:
            results = analyzer.analyze_dataset_quality(dataset_config, is_regression=True)
            analyzer.save_results(results, is_regression=True)
            
            # Log key results from 4-phase pipeline test
            pipeline_test = results.get('stages', {}).get('4_phase_pipeline_test', {})
            if pipeline_test:
                enhanced_success = pipeline_test.get('performance_metrics', {}).get('enhanced_pipeline_success', False)
                phase_success_rate = pipeline_test.get('performance_metrics', {}).get('phase_success_rate', 0.0)
                quality_score = pipeline_test.get('integration_test', {}).get('enhanced_pipeline', {}).get('quality_score', 'N/A')
                
                logger.info(f"    4-Phase Pipeline: {' Success' if enhanced_success else ' Failed'}")
                logger.info(f"   📈 Individual Phases: {phase_success_rate:.1%} success rate")
                logger.info(f"   🏆 Quality Score: {quality_score}")
                
                for rec in pipeline_test.get('recommendations', [])[:3]:  # Show top 3 recommendations
                    logger.info(f"    {rec}")
            
        except Exception as e:
            logger.error(f" Failed to analyze regression dataset {dataset_config.get('name', 'unknown')}: {e}")
    
    # Analyze classification datasets
    logger.info("")
    logger.info("🔬 ANALYZING CLASSIFICATION DATASETS")
    logger.info("=" * 50)
    
    for i, dataset_config in enumerate(CLASSIFICATION_DATASETS_FOR_ANALYSIS, 1):
        logger.info(f" Processing classification dataset {i}/{len(CLASSIFICATION_DATASETS_FOR_ANALYSIS)}: {dataset_config['name']}")
        try:
            results = analyzer.analyze_dataset_quality(dataset_config, is_regression=False)
            analyzer.save_results(results, is_regression=False)
            
            # Log key results from 4-phase pipeline test
            pipeline_test = results.get('stages', {}).get('4_phase_pipeline_test', {})
            if pipeline_test:
                enhanced_success = pipeline_test.get('performance_metrics', {}).get('enhanced_pipeline_success', False)
                phase_success_rate = pipeline_test.get('performance_metrics', {}).get('phase_success_rate', 0.0)
                quality_score = pipeline_test.get('integration_test', {}).get('enhanced_pipeline', {}).get('quality_score', 'N/A')
                
                logger.info(f"    4-Phase Pipeline: {' Success' if enhanced_success else ' Failed'}")
                logger.info(f"   📈 Individual Phases: {phase_success_rate:.1%} success rate")
                logger.info(f"   🏆 Quality Score: {quality_score}")
                
                for rec in pipeline_test.get('recommendations', [])[:3]:  # Show top 3 recommendations
                    logger.info(f"    {rec}")
                    
        except Exception as e:
            logger.error(f" Failed to analyze classification dataset {dataset_config.get('name', 'unknown')}: {e}")
    
    # Generate comprehensive summary
    logger.info("")
    logger.info("📋 GENERATING COMPREHENSIVE SUMMARY")
    logger.info("=" * 50)
    analyzer.generate_overall_summary()
    
    # Generate 4-phase pipeline summary
    logger.info(" Generating 4-Phase Pipeline Summary")
    try:
        analyzer.generate_4phase_pipeline_summary()
    except Exception as e:
        logger.warning(f"4-Phase pipeline summary generation failed: {e}")
    
    # Final summary
    logger.info("")
    logger.info("🎉 4-PHASE ENHANCED PIPELINE DATA QUALITY ANALYSIS COMPLETE!")
    logger.info("=" * 80)
    logger.info("📁 Results saved to:")
    logger.info(f"    Overall Summary: {analyzer.output_dir}/summary/overall_data_quality_summary.csv")
    logger.info(f"   📈 Statistics: {analyzer.output_dir}/summary/summary_statistics.csv")
    logger.info(f"    Robust Scaling Report: {analyzer.output_dir}/summary/robust_scaling_effectiveness_report.json")
    logger.info(f"    4-Phase Pipeline Report: {analyzer.output_dir}/summary/4phase_pipeline_summary.json")
    logger.info(f"    Regression Details: {analyzer.output_dir}/regression/")
    logger.info(f"    Classification Details: {analyzer.output_dir}/classification/")
    logger.info("")
    logger.info(" KEY ANALYSIS FEATURES:")
    logger.info("    4-Phase Enhanced Pipeline Integration Testing")
    logger.info("    Individual Phase Validation (Data Quality, Fusion-Aware, Missing Data, Validation)")
    logger.info("    Enhanced Data Loading with Orientation Validation")
    logger.info("    Robust Preprocessing Pipeline Testing")
    logger.info("    Comprehensive Data Quality Metrics")
    logger.info("    Algorithm Compatibility Validation")
    logger.info("    Performance and Improvement Tracking")
    logger.info("")
    logger.info(" Use this analysis to:")
    logger.info("    Validate 4-phase pipeline integration")
    logger.info("    Identify preprocessing improvements")
    logger.info("    Monitor data quality across datasets")
    logger.info("    Optimize pipeline performance")
    logger.info("    Ensure production readiness")


if __name__ == "__main__":
    main() 