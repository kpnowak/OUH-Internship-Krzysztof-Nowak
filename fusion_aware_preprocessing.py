#!/usr/bin/env python3
"""
Fusion-Aware Preprocessing Module.
Implements Phase 2: Fusion-Aware Feature Selection (Option A: Move feature selection after fusion).
This module reorders preprocessing steps based on the fusion method being used.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler

# Local imports (will be updated when integrated)
try:
    from preprocessing import ModalityAwareScaler, AdaptiveFeatureSelector
    from fusion import merge_modalities
except ImportError:
    # Fallback for development
    ModalityAwareScaler = None
    AdaptiveFeatureSelector = None

logger = logging.getLogger(__name__)

class FusionAwarePreprocessor:
    """
    Fusion-aware preprocessing pipeline that optimizes the order of operations
    based on the fusion method being used.
    
    For SNF, MKL, and Attention-based methods: Scale  Fuse  Select Features
    For simple methods: Select Features  Scale  Fuse (traditional order)
    """
    
    # Fusion methods that benefit from having more features during fusion
    FEATURE_RICH_FUSION_METHODS = {
        'snf', 'mkl', 'attention_weighted', 'multiple_kernel_learning',
        'similarity_network_fusion'
    }
    
    # Fusion methods that are order-independent
    ORDER_INDEPENDENT_METHODS = {
        'weighted_concat', 'early_fusion_pca', 'average', 'sum'
    }
    
    def __init__(self, fusion_method: str = "snf", task_type: str = "classification"):
        """
        Initialize fusion-aware preprocessor.
        
        Parameters
        ----------
        fusion_method : str
            The fusion method to be used
        task_type : str
            Task type (classification or regression)
        """
        self.fusion_method = fusion_method.lower()
        self.task_type = task_type
        self.preprocessing_order = self._determine_optimal_order()
        
        # Store fitted components
        self.scalers_ = {}
        self.feature_selectors_ = {}
        self.fusion_object_ = None
        
        logger.info(f"Fusion-aware preprocessor initialized for {fusion_method} with order: {self.preprocessing_order}")
    
    def _determine_optimal_order(self) -> str:
        """Determine optimal preprocessing order based on fusion method."""
        if self.fusion_method in self.FEATURE_RICH_FUSION_METHODS:
            return "scale_fuse_select"  # Feature-rich methods need features during fusion
        elif self.fusion_method in self.ORDER_INDEPENDENT_METHODS:
            return "select_scale_fuse"  # Traditional order for efficiency
        else:
            # For unknown methods, use conservative approach
            logger.warning(f"Unknown fusion method {self.fusion_method}, using conservative order")
            return "scale_fuse_select"
    
    def fit_transform(self, 
                     modality_data_dict: Dict[str, Tuple[np.ndarray, List[str]]], 
                     y: np.ndarray,
                     fusion_params: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply fusion-aware preprocessing with optimal order.
        
        Parameters
        ----------
        modality_data_dict : Dict[str, Tuple[np.ndarray, List[str]]]
            Dictionary mapping modality names to (data, sample_ids) tuples
        y : np.ndarray
            Target values
        fusion_params : Optional[Dict]
            Parameters for fusion method
            
        Returns
        -------
        Tuple[np.ndarray, Dict[str, Any]]
            Processed data and metadata
        """
        if fusion_params is None:
            fusion_params = {}
        
        logger.info(f"Starting fusion-aware preprocessing with order: {self.preprocessing_order}")
        
        if self.preprocessing_order == "scale_fuse_select":
            return self._scale_fuse_select(modality_data_dict, y, fusion_params)
        else:
            return self._select_scale_fuse(modality_data_dict, y, fusion_params)
    
    def _scale_fuse_select(self, 
                          modality_data_dict: Dict[str, Tuple[np.ndarray, List[str]]], 
                          y: np.ndarray,
                          fusion_params: Dict) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Optimal order for feature-rich fusion methods: Scale  Fuse  Select Features.
        """
        logger.info("Applying Scale  Fuse  Select order for feature-rich fusion")
        
        # Step 1: Scale each modality individually
        scaled_modalities = {}
        for modality_name, (X, sample_ids) in modality_data_dict.items():
            logger.debug(f"Scaling {modality_name} modality: {X.shape}")
            
            if ModalityAwareScaler:
                X_scaled, scaler = ModalityAwareScaler.scale_modality_data(X, modality_name)
            else:
                # Fallback scaling
                scaler = RobustScaler()
                X_scaled = scaler.fit_transform(X)
            
            scaled_modalities[modality_name] = X_scaled
            self.scalers_[modality_name] = scaler
        
        # Step 2: Apply fusion with full feature sets
        logger.info(f"Applying {self.fusion_method} fusion with full feature sets")
        
        modality_arrays = list(scaled_modalities.values())
        
        # Apply fusion
        if len(modality_arrays) > 1:
            try:
                fused_result = merge_modalities(
                    *modality_arrays,
                    strategy=self.fusion_method,
                    y=y,
                    is_regression=(self.task_type == "regression"),
                    fusion_params=fusion_params,
                    is_train=True
                )
                
                if isinstance(fused_result, tuple):
                    fused_data, self.fusion_object_ = fused_result
                else:
                    fused_data = fused_result
                    
            except Exception as e:
                logger.warning(f"Fusion failed: {str(e)}, using concatenation fallback")
                fused_data = np.column_stack(modality_arrays)
        else:
            fused_data = modality_arrays[0]
        
        logger.info(f"Fusion completed: {fused_data.shape}")
        
        # Step 3: Apply feature selection on fused data
        logger.info("Applying feature selection on fused data")
        
        if AdaptiveFeatureSelector and fused_data.shape[1] > 50:  # Only if many features
            try:
                final_data, feature_selector = AdaptiveFeatureSelector.select_features_adaptive(
                    fused_data, y, "fused_data", self.task_type
                )
                self.feature_selectors_["fused"] = feature_selector
                logger.info(f"Feature selection: {fused_data.shape[1]}  {final_data.shape[1]} features")
            except Exception as e:
                logger.warning(f"Feature selection failed: {str(e)}, using original data")
                final_data = fused_data
        else:
            final_data = fused_data
        
        metadata = {
            'preprocessing_order': self.preprocessing_order,
            'fusion_method': self.fusion_method,
            'original_shapes': {name: data[0].shape for name, data in modality_data_dict.items()},
            'final_shape': final_data.shape,
            'feature_reduction_ratio': final_data.shape[1] / sum(data[0].shape[1] for data in modality_data_dict.values())
        }
        
        return final_data, metadata
    
    def _select_scale_fuse(self, 
                          modality_data_dict: Dict[str, Tuple[np.ndarray, List[str]]], 
                          y: np.ndarray,
                          fusion_params: Dict) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Traditional order for simple fusion methods: Select Features  Scale  Fuse.
        """
        logger.info("Applying Select Features  Scale  Fuse order for simple fusion")
        
        # Step 1: Apply feature selection to each modality
        selected_modalities = {}
        for modality_name, (X, sample_ids) in modality_data_dict.items():
            logger.debug(f"Selecting features for {modality_name} modality: {X.shape}")
            
            if AdaptiveFeatureSelector and X.shape[1] > 100:  # Only if many features
                try:
                    X_selected, feature_selector = AdaptiveFeatureSelector.select_features_adaptive(
                        X, y, modality_name, self.task_type
                    )
                    self.feature_selectors_[modality_name] = feature_selector
                    logger.debug(f"Feature selection for {modality_name}: {X.shape[1]}  {X_selected.shape[1]} features")
                except Exception as e:
                    logger.warning(f"Feature selection failed for {modality_name}: {str(e)}")
                    X_selected = X
            else:
                X_selected = X
            
            selected_modalities[modality_name] = X_selected
        
        # Step 2: Scale selected features
        scaled_modalities = {}
        for modality_name, X_selected in selected_modalities.items():
            if ModalityAwareScaler:
                X_scaled, scaler = ModalityAwareScaler.scale_modality_data(X_selected, modality_name)
            else:
                # Fallback scaling
                scaler = RobustScaler()
                X_scaled = scaler.fit_transform(X_selected)
            
            scaled_modalities[modality_name] = X_scaled
            self.scalers_[modality_name] = scaler
        
        # Step 3: Apply fusion
        logger.info(f"Applying {self.fusion_method} fusion")
        
        modality_arrays = list(scaled_modalities.values())
        
        if len(modality_arrays) > 1:
            try:
                fused_result = merge_modalities(
                    *modality_arrays,
                    strategy=self.fusion_method,
                    y=y,
                    is_regression=(self.task_type == "regression"),
                    fusion_params=fusion_params,
                    is_train=True
                )
                
                if isinstance(fused_result, tuple):
                    final_data, self.fusion_object_ = fused_result
                else:
                    final_data = fused_result
                    
            except Exception as e:
                logger.warning(f"Fusion failed: {str(e)}, using concatenation fallback")
                final_data = np.column_stack(modality_arrays)
        else:
            final_data = modality_arrays[0]
        
        metadata = {
            'preprocessing_order': self.preprocessing_order,
            'fusion_method': self.fusion_method,
            'original_shapes': {name: data[0].shape for name, data in modality_data_dict.items()},
            'final_shape': final_data.shape,
            'feature_reduction_ratio': final_data.shape[1] / sum(data[0].shape[1] for data in modality_data_dict.values())
        }
        
        return final_data, metadata
    
    def transform(self, 
                 modality_data_dict: Dict[str, Tuple[np.ndarray, List[str]]]) -> np.ndarray:
        """
        Transform validation/test data using fitted preprocessing pipeline.
        
        Parameters
        ----------
        modality_data_dict : Dict[str, Tuple[np.ndarray, List[str]]]
            Dictionary mapping modality names to (data, sample_ids) tuples
            
        Returns
        -------
        np.ndarray
            Transformed data
        """
        logger.debug("Transforming validation/test data with fitted pipeline")
        
        if self.preprocessing_order == "scale_fuse_select":
            return self._transform_scale_fuse_select(modality_data_dict)
        else:
            return self._transform_select_scale_fuse(modality_data_dict)
    
    def _transform_scale_fuse_select(self, modality_data_dict: Dict[str, Tuple[np.ndarray, List[str]]]) -> np.ndarray:
        """Transform using scale  fuse  select order."""
        # Step 1: Scale using fitted scalers
        scaled_modalities = {}
        for modality_name, (X, sample_ids) in modality_data_dict.items():
            if modality_name in self.scalers_:
                X_scaled = self.scalers_[modality_name].transform(X)
                scaled_modalities[modality_name] = X_scaled
            else:
                logger.warning(f"No fitted scaler for {modality_name}, using original data")
                scaled_modalities[modality_name] = X
        
        # Step 2: Apply fusion using fitted fusion object
        modality_arrays = list(scaled_modalities.values())
        
        if len(modality_arrays) > 1 and self.fusion_object_ is not None:
            try:
                fused_data = self.fusion_object_.transform(modality_arrays)  # type: ignore
            except Exception as e:
                logger.warning(f"Fusion transform failed: {str(e)}, using concatenation")
                fused_data = np.column_stack(modality_arrays)
        else:
            fused_data = modality_arrays[0] if modality_arrays else np.array([])
        
        # Step 3: Apply feature selection using fitted selector
        if "fused" in self.feature_selectors_:
            try:
                final_data = self.feature_selectors_["fused"].transform(fused_data)
            except Exception as e:
                logger.warning(f"Feature selection transform failed: {str(e)}")
                final_data = fused_data
        else:
            final_data = fused_data
        
        return final_data
    
    def _transform_select_scale_fuse(self, modality_data_dict: Dict[str, Tuple[np.ndarray, List[str]]]) -> np.ndarray:
        """Transform using select  scale  fuse order."""
        # Step 1: Apply feature selection using fitted selectors
        selected_modalities = {}
        for modality_name, (X, sample_ids) in modality_data_dict.items():
            if modality_name in self.feature_selectors_:
                try:
                    X_selected = self.feature_selectors_[modality_name].transform(X)
                    selected_modalities[modality_name] = X_selected
                except Exception as e:
                    logger.warning(f"Feature selection transform failed for {modality_name}: {str(e)}")
                    selected_modalities[modality_name] = X
            else:
                selected_modalities[modality_name] = X
        
        # Step 2: Scale using fitted scalers
        scaled_modalities = {}
        for modality_name, X_selected in selected_modalities.items():
            if modality_name in self.scalers_:
                X_scaled = self.scalers_[modality_name].transform(X_selected)
                scaled_modalities[modality_name] = X_scaled
            else:
                logger.warning(f"No fitted scaler for {modality_name}")
                scaled_modalities[modality_name] = X_selected
        
        # Step 3: Apply fusion
        modality_arrays = list(scaled_modalities.values())
        
        if len(modality_arrays) > 1 and self.fusion_object_ is not None:
            try:
                final_data = self.fusion_object_.transform(modality_arrays)  # type: ignore
            except Exception as e:
                logger.warning(f"Fusion transform failed: {str(e)}, using concatenation")
                final_data = np.column_stack(modality_arrays)
        else:
            final_data = modality_arrays[0] if modality_arrays else np.array([])
        
        return final_data

def create_fusion_aware_preprocessor(fusion_method: str, task_type: str = "classification") -> FusionAwarePreprocessor:
    """
    Factory function to create fusion-aware preprocessor.
    
    Parameters
    ----------
    fusion_method : str
        Fusion method to be used
    task_type : str
        Task type (classification or regression)
        
    Returns
    -------
    FusionAwarePreprocessor
        Configured preprocessor instance
    """
    return FusionAwarePreprocessor(fusion_method=fusion_method, task_type=task_type)

def determine_optimal_fusion_order(fusion_method: str) -> str:
    """
    Determine optimal preprocessing order for a given fusion method.
    
    Parameters
    ----------
    fusion_method : str
        Name of fusion method
        
    Returns
    -------
    str
        Optimal order: "scale_fuse_select" or "select_scale_fuse"
    """
    preprocessor = FusionAwarePreprocessor(fusion_method=fusion_method)
    return preprocessor.preprocessing_order

def get_fusion_method_category(fusion_method: str) -> str:
    """
    Categorize fusion method for preprocessing decisions.
    
    Parameters
    ----------
    fusion_method : str
        Name of fusion method
        
    Returns
    -------
    str
        Category: "feature_rich", "order_independent", or "unknown"
    """
    method = fusion_method.lower()
    
    feature_rich_methods = {
        'snf', 'mkl', 'attention_weighted', 'multiple_kernel_learning',
        'similarity_network_fusion'
    }
    
    order_independent_methods = {
        'weighted_concat', 'early_fusion_pca', 'average', 'sum'
    }
    
    if method in feature_rich_methods:
        return "feature_rich"
    elif method in order_independent_methods:
        return "order_independent"
    else:
        return "unknown" 