#!/usr/bin/env python3
"""
Centralized Missing Data Management Module.
Implements Phase 3: Consolidates all missing data logic from multiple modules.
Intelligent strategy selection based on missingness patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from sklearn.impute import KNNImputer, SimpleImputer

logger = logging.getLogger(__name__)

class MissingDataStrategy:
    """Enumeration of missing data strategies."""
    SIMPLE = "simple"
    KNN = "knn"
    ADVANCED = "advanced"
    LATE_FUSION = "late_fusion"
    DROP_SAMPLES = "drop_samples"

class CentralizedMissingDataHandler:
    """
    Centralized handler for all missing data management.
    Consolidates logic from ModalityImputer, MissingModalityImputer, and other modules.
    """
    
    def __init__(self, 
                 strategy: str = "auto",
                 missing_threshold: float = 0.5,
                 modality_threshold: float = 0.9):
        """Initialize centralized missing data handler."""
        self.strategy = strategy
        self.missing_threshold = missing_threshold
        self.modality_threshold = modality_threshold
        
        # Store fitted imputers
        self.imputers_ = {}
        self.missing_patterns_ = {}
        self.chosen_strategy_ = None
        
    def analyze_missing_patterns(self, 
                                modality_data_dict: Dict[str, Tuple[np.ndarray, List[str]]]) -> Dict[str, Any]:
        """Comprehensive analysis of missing data patterns across all modalities."""
        logger.info("Analyzing missing data patterns across modalities")
        
        analysis = {
            'overall_missing_percentage': 0.0,
            'modality_missing_percentages': {},
            'recommended_strategy': MissingDataStrategy.SIMPLE,
            'critical_issues': []
        }
        
        try:
            total_elements = 0
            total_missing = 0
            
            for modality_name, (X, sample_ids) in modality_data_dict.items():
                # Calculate missing percentage for this modality
                if np.issubdtype(X.dtype, np.number):
                    missing_count = np.sum(np.isnan(X))
                else:
                    missing_count = np.sum(pd.isna(X))
                
                total_elements += X.size
                total_missing += missing_count
                
                missing_pct = missing_count / X.size if X.size > 0 else 0
                analysis['modality_missing_percentages'][modality_name] = missing_pct
                
                if missing_pct > self.modality_threshold:
                    analysis['critical_issues'].append(f"Modality {modality_name} has {missing_pct:.1%} missing data")
            
            # Calculate overall missing percentage
            analysis['overall_missing_percentage'] = total_missing / total_elements if total_elements > 0 else 0
            
            # Determine recommended strategy
            analysis['recommended_strategy'] = self._determine_strategy(analysis)
            
            # Store analysis results
            self.missing_patterns_ = analysis
            
            logger.info(f"Missing data analysis complete. Overall missing: {analysis['overall_missing_percentage']:.1%}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in missing data analysis: {str(e)}")
            analysis['critical_issues'].append(f"Analysis failed: {str(e)}")
            return analysis
    
    def _determine_strategy(self, analysis: Dict[str, Any]) -> str:
        """Determine optimal missing data strategy based on analysis."""
        overall_missing = analysis['overall_missing_percentage']
        
        if self.strategy != "auto":
            return self.strategy
        
        # Auto-determine strategy
        if overall_missing < 0.05:
            return MissingDataStrategy.SIMPLE
        elif overall_missing < 0.2:
            return MissingDataStrategy.KNN
        else:
            return MissingDataStrategy.ADVANCED
    
    def handle_missing_data(self, 
                           modality_data_dict: Dict[str, Tuple[np.ndarray, List[str]]]) -> Dict[str, np.ndarray]:
        """
        Handle missing data across all modalities using the determined strategy.
        
        Parameters
        ----------
        modality_data_dict : Dict[str, Tuple[np.ndarray, List[str]]]
            Dictionary mapping modality names to (data, sample_ids) tuples
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of processed modality data with missing values handled
        """
        logger.info("Handling missing data across modalities")
        
        # First analyze missing patterns if not done already
        if not hasattr(self, 'missing_patterns_') or not self.missing_patterns_:
            self.analyze_missing_patterns(modality_data_dict)
        
        processed_data = {}
        
        try:
            strategy = self._determine_strategy(self.missing_patterns_)
            self.chosen_strategy_ = strategy
            
            logger.info(f"Using missing data strategy: {strategy}")
            
            for modality_name, (X, sample_ids) in modality_data_dict.items():
                if strategy == MissingDataStrategy.SIMPLE:
                    processed_data[modality_name] = self._simple_imputation(X, modality_name)
                elif strategy == MissingDataStrategy.KNN:
                    processed_data[modality_name] = self._knn_imputation(X, modality_name)
                else:
                    # Fallback to simple imputation
                    processed_data[modality_name] = self._simple_imputation(X, modality_name)
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Missing data handling failed: {str(e)}")
            # Return original data as fallback
            return {name: X for name, (X, _) in modality_data_dict.items()}
    
    def _simple_imputation(self, X: np.ndarray, modality_name: str) -> np.ndarray:
        """Apply simple imputation (mean/median)."""
        if not np.any(np.isnan(X)):
            return X
        
        try:
            imputer = SimpleImputer(strategy='mean')
            X_imputed = imputer.fit_transform(X)
            self.imputers_[f"{modality_name}_simple"] = imputer
            
            missing_count = np.sum(np.isnan(X))
            logger.info(f"Simple imputation for {modality_name}: {missing_count} values imputed")
            
            return X_imputed
        except Exception as e:
            logger.warning(f"Simple imputation failed for {modality_name}: {str(e)}")
            return np.nan_to_num(X, nan=0.0)
    
    def _knn_imputation(self, X: np.ndarray, modality_name: str) -> np.ndarray:
        """Apply KNN imputation."""
        if not np.any(np.isnan(X)):
            return X
        
        try:
            # Use fewer neighbors for small datasets
            n_neighbors = min(5, max(1, X.shape[0] // 3))
            imputer = KNNImputer(n_neighbors=n_neighbors)
            X_imputed = imputer.fit_transform(X)
            self.imputers_[f"{modality_name}_knn"] = imputer
            
            missing_count = np.sum(np.isnan(X))
            logger.info(f"KNN imputation for {modality_name}: {missing_count} values imputed")
            
            return X_imputed
        except Exception as e:
            logger.warning(f"KNN imputation failed for {modality_name}: {str(e)}")
            return self._simple_imputation(X, modality_name)

def create_missing_data_handler(strategy: str = "auto") -> CentralizedMissingDataHandler:
    """Factory function to create centralized missing data handler."""
    return CentralizedMissingDataHandler(strategy=strategy)
