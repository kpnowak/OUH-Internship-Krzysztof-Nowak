#!/usr/bin/env python3
"""
Coordinated Validation Framework Module.
Implements Phase 4: Centralizes all validation logic from multiple modules.
Hierarchical validation (data  preprocessing  fusion  CV) with fail-fast error reporting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class ValidationStage(Enum):
    """Enumeration of validation stages."""
    DATA_LOADING = "data_loading"
    DATA_QUALITY = "data_quality"
    PREPROCESSING = "preprocessing"
    FUSION = "fusion"
    CROSS_VALIDATION = "cross_validation"
    MODEL_TRAINING = "model_training"

class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ValidationIssue:
    """Represents a validation issue."""
    
    def __init__(self, 
                 stage: ValidationStage, 
                 severity: ValidationSeverity, 
                 message: str, 
                 details: Optional[Dict] = None):
        self.stage = stage
        self.severity = severity
        self.message = message
        self.details = details or {}
        self.timestamp = pd.Timestamp.now()

class CoordinatedValidationFramework:
    """
    Centralized validation framework that coordinates all validation across the pipeline.
    Implements hierarchical validation with fail-fast error reporting.
    """
    
    def __init__(self, 
                 fail_fast: bool = True,
                 error_threshold: int = 5,
                 critical_threshold: int = 1):
        """
        Initialize coordinated validation framework.
        
        Parameters
        ----------
        fail_fast : bool
            Whether to fail immediately on critical issues
        error_threshold : int
            Maximum number of errors before failing
        critical_threshold : int
            Maximum number of critical issues before failing
        """
        self.fail_fast = fail_fast
        self.error_threshold = error_threshold
        self.critical_threshold = critical_threshold
        
        # Store validation results
        self.validation_issues: List[ValidationIssue] = []
        self.validation_summary = {}
        self.current_stage = None
        
    def start_stage(self, stage: ValidationStage) -> None:
        """Start validation for a new stage."""
        self.current_stage = stage
        logger.info(f"Starting validation stage: {stage.value}")
    
    def add_issue(self, 
                  severity: ValidationSeverity, 
                  message: str, 
                  details: Optional[Dict] = None,
                  stage: Optional[ValidationStage] = None) -> None:
        """
        Add a validation issue.
        
        Parameters
        ----------
        severity : ValidationSeverity
            Severity of the issue
        message : str
            Issue description
        details : Optional[Dict]
            Additional issue details
        stage : Optional[ValidationStage]
            Stage where issue occurred (uses current stage if None)
        """
        issue_stage = stage or self.current_stage or ValidationStage.DATA_LOADING
        issue = ValidationIssue(issue_stage, severity, message, details)
        self.validation_issues.append(issue)
        
        # Log the issue
        log_level = {
            ValidationSeverity.INFO: logging.INFO,
            ValidationSeverity.WARNING: logging.WARNING,
            ValidationSeverity.ERROR: logging.ERROR,
            ValidationSeverity.CRITICAL: logging.CRITICAL
        }[severity]
        
        logger.log(log_level, f"[{issue_stage.value}] {severity.value.upper()}: {message}")
        
        # Check fail-fast conditions
        if self.fail_fast:
            self._check_fail_conditions()
    
    def _check_fail_conditions(self) -> None:
        """Check if fail conditions are met and raise exception if needed."""
        critical_count = sum(1 for issue in self.validation_issues 
                           if issue.severity == ValidationSeverity.CRITICAL)
        error_count = sum(1 for issue in self.validation_issues 
                         if issue.severity == ValidationSeverity.ERROR)
        
        if critical_count >= self.critical_threshold:
            raise ValidationError(f"Critical validation threshold exceeded: {critical_count} critical issues")
        
        if error_count >= self.error_threshold:
            raise ValidationError(f"Error validation threshold exceeded: {error_count} errors")
    
    def validate_data_loading(self, 
                             modality_data_dict: Dict[str, Tuple[np.ndarray, List[str]]], 
                             y: np.ndarray,
                             dataset_name: str) -> bool:
        """
        Validate data loading stage.
        
        Parameters
        ----------
        modality_data_dict : Dict[str, Tuple[np.ndarray, List[str]]]
            Loaded modality data
        y : np.ndarray
            Target values
        dataset_name : str
            Dataset name
            
        Returns
        -------
        bool
            True if validation passes
        """
        self.start_stage(ValidationStage.DATA_LOADING)
        
        try:
            # Check if data was loaded successfully
            if not modality_data_dict:
                self.add_issue(ValidationSeverity.CRITICAL, "No modality data loaded")
                return False
            
            if len(y) == 0:
                self.add_issue(ValidationSeverity.CRITICAL, "No target data loaded")
                return False
            
            # Check data consistency
            for modality_name, (X, sample_ids) in modality_data_dict.items():
                # Check array dimensions
                if X.size == 0:
                    self.add_issue(ValidationSeverity.ERROR, f"Empty data array for {modality_name}")
                    continue
                
                if len(X.shape) != 2:
                    self.add_issue(ValidationSeverity.ERROR, f"Invalid data shape for {modality_name}: {X.shape}")
                    continue
                
                # Check sample ID consistency
                if len(sample_ids) != X.shape[0]:
                    self.add_issue(ValidationSeverity.ERROR, 
                                 f"Sample ID count mismatch for {modality_name}: {len(sample_ids)} vs {X.shape[0]}")
                
                # Check for obvious data issues
                if np.issubdtype(X.dtype, np.number):
                    inf_count = np.sum(np.isinf(X))
                    if inf_count > 0:
                        self.add_issue(ValidationSeverity.WARNING, 
                                     f"Found {inf_count} infinite values in {modality_name}")
            
            # Check target-data alignment
            first_modality_samples = len(list(modality_data_dict.values())[0][1])
            if len(y) != first_modality_samples:
                self.add_issue(ValidationSeverity.WARNING, 
                             f"Target count ({len(y)}) != sample count ({first_modality_samples})")
            
            logger.info(f"Data loading validation completed for {dataset_name}")
            return True
            
        except Exception as e:
            self.add_issue(ValidationSeverity.CRITICAL, f"Data loading validation failed: {str(e)}")
            return False
    
    def validate_preprocessing(self, 
                              processed_data: Dict[str, np.ndarray], 
                              original_shapes: Dict[str, Tuple],
                              preprocessing_config: Dict) -> bool:
        """
        Validate preprocessing stage.
        
        Parameters
        ----------
        processed_data : Dict[str, np.ndarray]
            Processed modality data
        original_shapes : Dict[str, Tuple]
            Original data shapes for comparison
        preprocessing_config : Dict
            Preprocessing configuration used
            
        Returns
        -------
        bool
            True if validation passes
        """
        self.start_stage(ValidationStage.PREPROCESSING)
        
        try:
            if not processed_data:
                self.add_issue(ValidationSeverity.CRITICAL, "No processed data available")
                return False
            
            for modality_name, X_processed in processed_data.items():
                # Check for data corruption
                if X_processed.size == 0:
                    self.add_issue(ValidationSeverity.ERROR, f"Empty processed data for {modality_name}")
                    continue
                
                # Check for numerical issues
                if np.issubdtype(X_processed.dtype, np.number):
                    nan_count = np.sum(np.isnan(X_processed))
                    inf_count = np.sum(np.isinf(X_processed))
                    
                    if nan_count > 0:
                        self.add_issue(ValidationSeverity.ERROR, 
                                     f"Found {nan_count} NaN values after preprocessing in {modality_name}")
                    
                    if inf_count > 0:
                        self.add_issue(ValidationSeverity.WARNING, 
                                     f"Found {inf_count} infinite values after preprocessing in {modality_name}")
                    
                    # Check for extreme values that might indicate scaling issues
                    if np.any(np.abs(X_processed) > 1000):
                        self.add_issue(ValidationSeverity.WARNING, 
                                     f"Extreme values detected in {modality_name} - possible scaling issue")
                
                # Check sample preservation
                if modality_name in original_shapes:
                    original_samples = original_shapes[modality_name][0]
                    processed_samples = X_processed.shape[0]
                    
                    if processed_samples != original_samples:
                        self.add_issue(ValidationSeverity.WARNING, 
                                     f"Sample count changed for {modality_name}: {original_samples}  {processed_samples}")
            
            logger.info("Preprocessing validation completed")
            return True
            
        except Exception as e:
            self.add_issue(ValidationSeverity.CRITICAL, f"Preprocessing validation failed: {str(e)}")
            return False
    
    def validate_fusion(self, 
                       fused_data: np.ndarray, 
                       input_modalities: Dict[str, np.ndarray],
                       fusion_method: str) -> bool:
        """
        Validate fusion stage.
        
        Parameters
        ----------
        fused_data : np.ndarray
            Fused data result
        input_modalities : Dict[str, np.ndarray]
            Input modality data for fusion
        fusion_method : str
            Fusion method used
            
        Returns
        -------
        bool
            True if validation passes
        """
        self.start_stage(ValidationStage.FUSION)
        
        try:
            if fused_data.size == 0:
                self.add_issue(ValidationSeverity.CRITICAL, "Fusion produced empty result")
                return False
            
            # Check sample preservation
            if input_modalities:
                expected_samples = list(input_modalities.values())[0].shape[0]
                if fused_data.shape[0] != expected_samples:
                    self.add_issue(ValidationSeverity.ERROR, 
                                 f"Sample count mismatch after fusion: expected {expected_samples}, got {fused_data.shape[0]}")
            
            # Check for numerical issues
            if np.issubdtype(fused_data.dtype, np.number):
                nan_count = np.sum(np.isnan(fused_data))
                inf_count = np.sum(np.isinf(fused_data))
                
                if nan_count > 0:
                    self.add_issue(ValidationSeverity.ERROR, f"Fusion produced {nan_count} NaN values")
                
                if inf_count > 0:
                    self.add_issue(ValidationSeverity.WARNING, f"Fusion produced {inf_count} infinite values")
                
                # Check for degenerate fusion (all zeros or same values)
                if np.all(fused_data == 0):
                    self.add_issue(ValidationSeverity.WARNING, "Fusion produced all-zero result")
                elif np.all(fused_data == fused_data.flat[0]):
                    self.add_issue(ValidationSeverity.WARNING, "Fusion produced constant result")
            
            # Check feature dimensionality
            total_input_features = sum(X.shape[1] for X in input_modalities.values())
            fusion_features = fused_data.shape[1]
            
            if fusion_features > total_input_features:
                self.add_issue(ValidationSeverity.WARNING, 
                             f"Fusion increased feature count: {total_input_features}  {fusion_features}")
            
            logger.info(f"Fusion validation completed for {fusion_method}")
            return True
            
        except Exception as e:
            self.add_issue(ValidationSeverity.CRITICAL, f"Fusion validation failed: {str(e)}")
            return False
    
    def validate_processed_data(self, 
                               processed_data: Dict[str, np.ndarray], 
                               y: np.ndarray) -> Dict[str, Any]:
        """
        Validate processed data quality and consistency.
        
        This method provides a comprehensive validation of processed data
        that can be used by the data quality analyzer.
        
        Parameters
        ----------
        processed_data : Dict[str, np.ndarray]
            Dictionary of processed modality data
        y : np.ndarray
            Target values
            
        Returns
        -------
        Dict[str, Any]
            Validation results with issues and recommendations
        """
        self.start_stage(ValidationStage.DATA_QUALITY)
        
        validation_results = {
            'stage': 'processed_data_validation',
            'timestamp': pd.Timestamp.now().isoformat(),
            'modalities_validated': list(processed_data.keys()),
            'issues': [],
            'recommendations': [],
            'overall_status': 'passed'
        }
        
        try:
            if not processed_data:
                self.add_issue(ValidationSeverity.CRITICAL, "No processed data provided")
                validation_results['overall_status'] = 'failed'
                return validation_results
            
            if len(y) == 0:
                self.add_issue(ValidationSeverity.CRITICAL, "No target data provided")
                validation_results['overall_status'] = 'failed'
                return validation_results
            
            # Validate each modality
            for modality_name, X in processed_data.items():
                modality_issues = []
                
                # Basic shape validation
                if X.size == 0:
                    issue = f"Empty data array for {modality_name}"
                    self.add_issue(ValidationSeverity.ERROR, issue)
                    modality_issues.append(issue)
                    continue
                
                if len(X.shape) != 2:
                    issue = f"Invalid data shape for {modality_name}: {X.shape} (expected 2D)"
                    self.add_issue(ValidationSeverity.ERROR, issue)
                    modality_issues.append(issue)
                    continue
                
                # Sample count validation
                if X.shape[0] != len(y):
                    issue = f"Sample count mismatch for {modality_name}: {X.shape[0]} vs {len(y)} targets"
                    self.add_issue(ValidationSeverity.WARNING, issue)
                    modality_issues.append(issue)
                
                # Numerical validation
                if np.issubdtype(X.dtype, np.number):
                    # Check for NaN values
                    nan_count = np.sum(np.isnan(X))
                    if nan_count > 0:
                        issue = f"Found {nan_count} NaN values in {modality_name}"
                        self.add_issue(ValidationSeverity.ERROR, issue)
                        modality_issues.append(issue)
                    
                    # Check for infinite values
                    inf_count = np.sum(np.isinf(X))
                    if inf_count > 0:
                        issue = f"Found {inf_count} infinite values in {modality_name}"
                        self.add_issue(ValidationSeverity.WARNING, issue)
                        modality_issues.append(issue)
                    
                    # Check for extreme values (potential scaling issues)
                    extreme_values = np.sum(np.abs(X) > 100)
                    if extreme_values > 0:
                        max_value = np.max(np.abs(X))
                        issue = f"Found {extreme_values} extreme values (>100) in {modality_name} (max abs value: {max_value:.2f})"
                        self.add_issue(ValidationSeverity.WARNING, issue)
                        modality_issues.append(issue)
                    
                    # Check data variance (constant features)
                    feature_variances = np.var(X, axis=0)
                    zero_var_features = np.sum(feature_variances < 1e-10)
                    if zero_var_features > 0:
                        issue = f"Found {zero_var_features} zero-variance features in {modality_name}"
                        self.add_issue(ValidationSeverity.INFO, issue)
                        modality_issues.append(issue)
                
                validation_results['issues'].extend(modality_issues)
            
            # Cross-modality validation
            if len(processed_data) > 1:
                # Check sample consistency across modalities
                sample_counts = [X.shape[0] for X in processed_data.values()]
                if len(set(sample_counts)) > 1:
                    issue = f"Inconsistent sample counts across modalities: {dict(zip(processed_data.keys(), sample_counts))}"
                    self.add_issue(ValidationSeverity.ERROR, issue)
                    validation_results['issues'].append(issue)
            
            # Generate recommendations
            if not validation_results['issues']:
                validation_results['recommendations'].append(" All processed data validation checks passed")
            else:
                error_count = sum(1 for issue in self.validation_issues 
                                if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL])
                if error_count > 0:
                    validation_results['overall_status'] = 'failed'
                    validation_results['recommendations'].append(f" {error_count} critical/error issues found - data quality needs attention")
                else:
                    validation_results['recommendations'].append(" Some warnings detected - review data quality")
                
                # Specific recommendations based on issues
                if any('NaN' in issue for issue in validation_results['issues']):
                    validation_results['recommendations'].append(" Consider improving missing data handling")
                
                if any('extreme values' in issue for issue in validation_results['issues']):
                    validation_results['recommendations'].append(" Consider reviewing data scaling/normalization")
                
                if any('zero-variance' in issue for issue in validation_results['issues']):
                    validation_results['recommendations'].append(" Consider feature selection to remove constant features")
            
            logger.info(f"Processed data validation completed: {validation_results['overall_status']}")
            return validation_results
            
        except Exception as e:
            error_msg = f"Processed data validation failed: {str(e)}"
            self.add_issue(ValidationSeverity.CRITICAL, error_msg)
            validation_results['issues'].append(error_msg)
            validation_results['overall_status'] = 'failed'
            validation_results['recommendations'].append(" Validation process failed - investigate data processing pipeline")
            return validation_results
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive validation summary.
        
        Returns
        -------
        Dict[str, Any]
            Validation summary with counts by stage and severity
        """
        summary = {
            'total_issues': len(self.validation_issues),
            'by_severity': {},
            'by_stage': {},
            'critical_issues': [],
            'error_issues': [],
            'has_failures': False
        }
        
        # Count by severity
        for severity in ValidationSeverity:
            count = sum(1 for issue in self.validation_issues if issue.severity == severity)
            summary['by_severity'][severity.value] = count
        
        # Count by stage
        for stage in ValidationStage:
            count = sum(1 for issue in self.validation_issues if issue.stage == stage)
            summary['by_stage'][stage.value] = count
        
        # Collect critical and error issues
        for issue in self.validation_issues:
            if issue.severity == ValidationSeverity.CRITICAL:
                summary['critical_issues'].append(issue.message)
            elif issue.severity == ValidationSeverity.ERROR:
                summary['error_issues'].append(issue.message)
        
        # Determine if there are failures
        summary['has_failures'] = (summary['by_severity'].get('critical', 0) > 0 or 
                                  summary['by_severity'].get('error', 0) > 0)
        
        return summary
    
    def clear_issues(self) -> None:
        """Clear all validation issues."""
        self.validation_issues.clear()
        logger.info("Validation issues cleared")

class ValidationError(Exception):
    """Exception raised when validation fails."""
    pass

def create_validation_coordinator(fail_fast: bool = True) -> CoordinatedValidationFramework:
    """
    Factory function to create coordinated validation framework.
    
    Parameters
    ----------
    fail_fast : bool
        Whether to fail immediately on critical issues
        
    Returns
    -------
    CoordinatedValidationFramework
        Configured validation framework instance
    """
    return CoordinatedValidationFramework(fail_fast=fail_fast)

# Alias for backward compatibility
ValidationCoordinator = CoordinatedValidationFramework
