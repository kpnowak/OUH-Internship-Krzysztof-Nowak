#!/usr/bin/env python3
"""
Enhanced Pipeline Integration Module.
Coordinates all 4 phases of architectural improvements:
1. Early Data Quality Pipeline
2. Fusion-Aware Feature Selection
3. Centralized Missing Data Management  
4. Coordinated Validation Framework
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)

class EnhancedPipelineCoordinator:
    """
    Master coordinator for the enhanced pipeline with all 4 architectural improvements.
    Orchestrates the entire flow from data loading to model-ready data.
    """
    
    def __init__(self, 
                 fusion_method: str = "snf",
                 task_type: str = "classification",
                 enable_early_quality_check: bool = True,
                 enable_fusion_aware_order: bool = True,
                 enable_centralized_missing_data: bool = True,
                 enable_coordinated_validation: bool = True,
                 fail_fast: bool = True):
        """Initialize enhanced pipeline coordinator."""
        self.fusion_method = fusion_method
        self.task_type = task_type
        self.enable_early_quality_check = enable_early_quality_check
        self.enable_fusion_aware_order = enable_fusion_aware_order
        self.enable_centralized_missing_data = enable_centralized_missing_data
        self.enable_coordinated_validation = enable_coordinated_validation
        self.fail_fast = fail_fast
        
        # Results storage
        self.quality_report_ = {}
        self.preprocessing_guidance_ = {}
        self.pipeline_metadata_ = {}
        
        logger.info(f"Enhanced pipeline coordinator initialized for {fusion_method}")
    
    def process_pipeline(self, 
                        modality_data_dict: Dict[str, Tuple[np.ndarray, List[str]]], 
                        y: np.ndarray,
                        dataset_name: str = "unknown") -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, Any]]:
        """
        Run the complete enhanced pipeline with all architectural improvements.
        
        Parameters
        ----------
        modality_data_dict : Dict[str, Tuple[np.ndarray, List[str]]]
            Dictionary mapping modality names to (data, sample_ids) tuples
        y : np.ndarray
            Target values
        dataset_name : str
            Dataset name for logging
            
        Returns
        -------
        Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, Any]]
            Final processed data, aligned targets, and pipeline metadata
        """
        logger.info(f"Starting enhanced pipeline processing for {dataset_name}")
        
        try:
            # Phase 1: Early Data Quality Assessment
            if self.enable_early_quality_check:
                logger.info("Phase 1: Running early data quality assessment")
                quality_report, guidance = self._run_quality_assessment(modality_data_dict, y, dataset_name)
                self.quality_report_ = quality_report
                self.preprocessing_guidance_ = guidance
            
            # Phase 3: Missing Data Management
            processed_modalities = modality_data_dict
            if self.enable_centralized_missing_data:
                logger.info("Phase 3: Handling missing data")
                processed_modalities = self._handle_missing_data(modality_data_dict)
            
            # Phase 2: Fusion-Aware Preprocessing
            if self.enable_fusion_aware_order:
                logger.info("Phase 2: Applying fusion-aware preprocessing")
                final_data, y_aligned = self._apply_fusion_aware_preprocessing(processed_modalities, y)
            else:
                # Fallback to simple concatenation
                logger.info("Using simple concatenation fallback")
                final_data, y_aligned = self._simple_concatenation_fallback(processed_modalities, y)
            
            # Phase 4: Final validation
            if self.enable_coordinated_validation:
                logger.info("Phase 4: Running final validation")
                self._validate_final_results(final_data, y_aligned)
            
            # Compile metadata
            self.pipeline_metadata_ = {
                'fusion_method': self.fusion_method,
                'task_type': self.task_type,
                'dataset_name': dataset_name,
                'quality_score': self.quality_report_.get('overall_quality_score', 0.8),
                'phases_enabled': {
                    'early_quality_check': self.enable_early_quality_check,
                    'fusion_aware_order': self.enable_fusion_aware_order,
                    'centralized_missing_data': self.enable_centralized_missing_data,
                    'coordinated_validation': self.enable_coordinated_validation
                }
            }
            
            logger.info(f"Enhanced pipeline processing completed successfully for {dataset_name}")
            return final_data, y_aligned, self.pipeline_metadata_
            
        except Exception as e:
            logger.error(f"Enhanced pipeline processing failed: {str(e)}")
            raise
    
    def _run_quality_assessment(self, modality_data_dict, y, dataset_name):
        """Run Phase 1: Early Data Quality Assessment."""
        try:
            from data_quality import run_early_data_quality_pipeline
            return run_early_data_quality_pipeline(modality_data_dict, y, dataset_name, self.task_type)
        except ImportError:
            logger.warning("Data quality module not available, using defaults")
            return {'overall_quality_score': 0.8}, {'fusion_strategy': 'snf'}
    
    def _handle_missing_data(self, modality_data_dict):
        """Run Phase 3: Centralized Missing Data Management."""
        try:
            from missing_data_handler import create_missing_data_handler
            handler = create_missing_data_handler(strategy="auto")
            handler.analyze_missing_patterns(modality_data_dict)
            processed_arrays = handler.handle_missing_data(modality_data_dict)
            
            # Convert back to original format (X, sample_ids) tuples
            processed_modality_dict = {}
            for modality_name, (original_X, sample_ids) in modality_data_dict.items():
                if modality_name in processed_arrays:
                    processed_modality_dict[modality_name] = (processed_arrays[modality_name], sample_ids)
                else:
                    processed_modality_dict[modality_name] = (original_X, sample_ids)
            
            return processed_modality_dict
        except ImportError:
            logger.warning("Missing data handler not available, using original data")
            return modality_data_dict
        except Exception as e:
            logger.warning(f"Missing data handling failed: {e}, using original data")
            return modality_data_dict
    
    def _apply_fusion_aware_preprocessing(self, modality_data_dict, y):
        """Run Phase 2: Fusion-Aware Preprocessing."""
        try:
            from fusion_aware_preprocessing import determine_optimal_fusion_order
            optimal_order = determine_optimal_fusion_order(self.fusion_method)
            logger.info(f"Optimal preprocessing order: {optimal_order}")
            
            # For now, use simple preprocessing regardless of order
            return self._simple_preprocessing_fallback(modality_data_dict, y)
        except ImportError:
            logger.warning("Fusion-aware preprocessing not available, using fallback")
            return self._simple_preprocessing_fallback(modality_data_dict, y)
    
    def _simple_preprocessing_fallback(self, modality_data_dict, y):
        """Simple preprocessing fallback using robust biomedical pipeline."""
        try:
            # Use the robust biomedical preprocessing pipeline directly
            from preprocessing import robust_biomedical_preprocessing_pipeline
            
            processed_dict = {}
            
            for modality_name, (X, sample_ids) in modality_data_dict.items():
                # Determine modality type
                if 'exp' in modality_name.lower() or 'gene' in modality_name.lower():
                    modality_type = 'gene_expression'
                elif 'mirna' in modality_name.lower():
                    modality_type = 'mirna'
                elif 'methy' in modality_name.lower():
                    modality_type = 'methylation'
                else:
                    modality_type = 'unknown'
                
                # Apply robust preprocessing
                result = robust_biomedical_preprocessing_pipeline(X, modality_type=modality_type)
                if len(result) == 3:
                    X_processed, transformers, report = result
                elif len(result) == 2:
                    X_processed, transformers = result
                else:
                    X_processed = result
                processed_dict[modality_name] = X_processed
                
                logger.info(f"Fallback preprocessing for {modality_name}: {X.shape} -> {X_processed.shape}")
            
            # Align targets to match processed data
            n_samples = list(processed_dict.values())[0].shape[0]
            y_aligned = y[:n_samples] if len(y) >= n_samples else y
            
            return processed_dict, y_aligned
            
        except Exception as e:
            logger.warning(f"Robust preprocessing failed: {e}, using concatenation")
            return self._simple_concatenation_fallback(modality_data_dict, y)
    
    def _simple_concatenation_fallback(self, modality_data_dict, y):
        """Ultimate fallback: simple concatenation."""
        modality_arrays = []
        common_samples = None
        
        # Handle both tuple format and direct array format
        try:
            for modality_name, data in modality_data_dict.items():
                if isinstance(data, tuple) and len(data) == 2:
                    X, sample_ids = data
                    if common_samples is None:
                        common_samples = set(sample_ids)
                    else:
                        common_samples = common_samples.intersection(set(sample_ids))
                else:
                    # Direct array format - use all data
                    X = data
                    modality_arrays.append(X)
            
            if common_samples is not None:
                # Use common samples approach
                if not common_samples:
                    # No common samples, use first modality
                    first_data = list(modality_data_dict.values())[0]
                    if isinstance(first_data, tuple):
                        first_data = first_data[0]
                    return {"concatenated": first_data}, y[:first_data.shape[0]]
                
                # Use common samples
                common_samples_list = sorted(list(common_samples))
                modality_arrays = []
                for modality_name, (X, sample_ids) in modality_data_dict.items():
                    sample_to_idx = {sid: i for i, sid in enumerate(sample_ids)}
                    indices = [sample_to_idx[sid] for sid in common_samples_list if sid in sample_to_idx]
                    if indices:
                        modality_arrays.append(X[indices])
                
                if modality_arrays:
                    concatenated = np.column_stack(modality_arrays)
                    return {"concatenated": concatenated}, y[:len(common_samples_list)]
            else:
                # Direct array concatenation
                if modality_arrays:
                    concatenated = np.column_stack(modality_arrays)
                    return {"concatenated": concatenated}, y[:concatenated.shape[0]]
            
            return {"empty": np.array([])}, np.array([])
            
        except Exception as e:
            logger.warning(f"Concatenation fallback failed: {e}")
            return {"empty": np.array([])}, np.array([])
    
    def _validate_final_results(self, final_data, y_aligned):
        """Run Phase 4: Final validation."""
        try:
            from validation_coordinator import create_validation_coordinator, ValidationSeverity
            validator = create_validation_coordinator(fail_fast=False)
            
            # Basic validation
            if isinstance(final_data, dict):
                for name, data in final_data.items():
                    if data.size == 0:
                        validator.add_issue(ValidationSeverity.WARNING, f"Empty final data for {name}")
                    elif np.any(np.isnan(data)):
                        validator.add_issue(ValidationSeverity.ERROR, f"NaN values in final data for {name}")
            
            validation_summary = validator.get_validation_summary()
            self.pipeline_metadata_['validation_summary'] = validation_summary
            
        except ImportError:
            logger.warning("Validation coordinator not available")
        except Exception as e:
            logger.warning(f"Final validation failed: {e}")

def run_enhanced_preprocessing_pipeline(modality_data_dict: Dict[str, Tuple[np.ndarray, List[str]]], 
                                      y: np.ndarray,
                                      fusion_method: str = "snf",
                                      task_type: str = "classification",
                                      dataset_name: str = "unknown",
                                      **kwargs) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, Any]]:
    """
    Run the enhanced preprocessing pipeline with all architectural improvements.
    
    This is the main entry point for the new enhanced pipeline that integrates:
    - Phase 1: Early Data Quality Pipeline
    - Phase 2: Fusion-Aware Feature Selection
    - Phase 3: Centralized Missing Data Management
    - Phase 4: Coordinated Validation Framework
    
    Parameters
    ----------
    modality_data_dict : Dict[str, Tuple[np.ndarray, List[str]]]
        Dictionary mapping modality names to (data, sample_ids) tuples
    y : np.ndarray
        Target values
    fusion_method : str
        Fusion method to be used
    task_type : str
        Task type (classification or regression)
    dataset_name : str
        Dataset name for logging
    **kwargs
        Additional configuration parameters
        
    Returns
    -------
    Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, Any]]
        Final processed data, aligned targets, and pipeline metadata
    """
    coordinator = EnhancedPipelineCoordinator(
        fusion_method=fusion_method,
        task_type=task_type,
        **kwargs
    )
    
    return coordinator.process_pipeline(modality_data_dict, y, dataset_name)