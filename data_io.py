#!/usr/bin/env python3
"""
Input/Output module for reading and writing data files.
Enhanced with parallel processing and caching for optimal performance.
"""

import numpy as np
import pandas as pd
import warnings
from pathlib import Path

# Suppress sklearn deprecation warning about force_all_finite -> ensure_all_finite
warnings.filterwarnings("ignore", message=".*force_all_finite.*was renamed to.*ensure_all_finite.*", category=FutureWarning)
from typing import Optional, List, Dict, Union, Tuple, Any
import logging
import re
import hashlib
import pickle
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import lru_cache

# Local imports
from config import DatasetConfig, MAX_VARIABLE_FEATURES, SAMPLE_RETENTION_CONFIG
from preprocessing import (
    _keep_top_variable_rows, 
    fix_tcga_id_slicing, 
    custom_parse_outcome, 
    normalize_sample_ids, 
    advanced_feature_filtering,
    robust_biomedical_preprocessing_pipeline,
    log_transform_data,
    quantile_normalize_data,
    handle_sparse_features,
    # Add new enhanced preprocessing imports (DataOrientationValidator moved to this file)
    ModalityAwareScaler,
    AdaptiveFeatureSelector,
    SampleIntersectionManager,
    PreprocessingValidator,
    FusionMethodStandardizer,
    enhanced_comprehensive_preprocessing_pipeline
)

logger = logging.getLogger(__name__)

# Global cache for loaded modalities
_modality_cache = {}
_cache_lock = threading.Lock()

class DataOrientationValidationError(Exception):
    """Custom exception for data orientation validation errors"""
    pass

class DataOrientationValidator:
    """
    Priority 1: Fix Data Orientation Issues (IMMEDIATE)
    Validates and fixes data matrix orientation to prevent transposition issues.
    
    This class has been moved from preprocessing.py to data_io.py to catch
    orientation issues as early as possible in the data loading process.
    """
    
    @staticmethod
    def validate_dataframe_orientation(df: pd.DataFrame, modality_name: str = "unknown") -> pd.DataFrame:
        """
        Validate and fix DataFrame orientation issues that cause preprocessing inconsistencies.
        This version works with pandas DataFrames and provides rich context about the data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with features as rows and samples as columns (expected)
        modality_name : str
            Name of the modality for logging
            
        Returns
        -------
        pd.DataFrame
            Properly oriented DataFrame
        """
        if df.empty:
            logger.warning(f"Empty DataFrame provided for {modality_name}")
            return df
            
        n_features, n_samples = df.shape  # DataFrame: rows=features, columns=samples
        
        # Log original dimensions
        logger.info(f"{modality_name} original shape: {df.shape} (features={n_features}, samples={n_samples})")
        
        # Check if sample IDs are in the index instead of columns (common mistake)
        sample_pattern = ['TCGA', 'sample', 'SAMPLE_', 'patient', 'PATIENT']
        
        # Count potential sample IDs in index vs columns
        index_sample_count = sum(1 for idx in df.index if isinstance(idx, str) 
                                and any(pattern in idx for pattern in sample_pattern))
        column_sample_count = sum(1 for col in df.columns if isinstance(col, str) 
                                 and any(pattern in col for pattern in sample_pattern))
        
        logger.debug(f"{modality_name}: sample IDs in index={index_sample_count}, in columns={column_sample_count}")
        
        # Decision logic for transposition
        needs_transpose = False
        transpose_reason = ""
        
        # Rule 1: If samples are clearly in the index, transpose
        if index_sample_count >= 5 and column_sample_count < 5:
            needs_transpose = True
            transpose_reason = f"samples detected in index ({index_sample_count}) rather than columns ({column_sample_count})"
        
        # Rule 2: For gene expression data, be more aggressive about detecting wrong orientation
        elif modality_name.lower() in ["gene_expression", "gene", "expression", "exp"]:
            # Gene expression should have many more features than samples
            if n_features < n_samples and n_samples > 100:
                needs_transpose = True
                transpose_reason = f"gene expression with {n_features} features < {n_samples} samples (suspicious)"
            # Also check if we have way too many "samples" (likely features)
            elif n_samples > 10000:
                needs_transpose = True
                transpose_reason = f"gene expression with {n_samples} samples > 10000 (likely features)"
        
        # Rule 3: General biological data validation
        elif n_features < 50 and n_samples > 1000:
            # Very few features but many samples - likely wrong orientation
            needs_transpose = True
            transpose_reason = f"suspicious ratio: {n_features} features vs {n_samples} samples"
        
        # Apply transposition if needed
        if needs_transpose:
            logger.warning(f"Transposing {modality_name}: {transpose_reason}")
            logger.info(f"Before transpose: {df.shape} -> After transpose: {(df.T.shape)}")
            df = df.T
            n_features, n_samples = df.shape
        
        # Final validation checks
        if n_samples < 2:
            raise DataOrientationValidationError(f"Insufficient samples for {modality_name}: {n_samples}")
        elif n_samples < 5:
            logger.warning(f"Very few samples for {modality_name}: {n_samples} (minimum for robust analysis)")
        
        if n_features < 5:
            logger.warning(f"Very few features for {modality_name}: {n_features}")
        
        # Log final validated dimensions
        logger.info(f"{modality_name} validated shape: {df.shape} (features={n_features}, samples={n_samples})")
        
        return df
    
    @staticmethod
    def validate_data_orientation(X: np.ndarray, modality_name: str = "unknown") -> np.ndarray:
        """
        Validate and fix data matrix orientation issues (numpy array version).
        This is kept for backward compatibility with preprocessing.py functions.
        
        Parameters
        ----------
        X : np.ndarray
            Input data matrix
        modality_name : str
            Name of the modality for logging
            
        Returns
        -------
        np.ndarray
            Properly oriented data matrix
        """
        if X.size == 0:
            logger.warning(f"Empty array provided for {modality_name}")
            return X
            
        n_samples, n_features = X.shape
        
        # Log original dimensions
        logger.info(f"{modality_name} original shape: {X.shape}")
        
        # Critical validation for gene expression data
        if modality_name.lower() in ["gene_expression", "gene", "expression", "exp"]:
            if n_samples > n_features and n_samples > 1000:
                logger.warning(f"Suspicious orientation for {modality_name}: {n_samples} samples > {n_features} features")
                logger.info(f"Auto-transposing {modality_name}: {X.shape} -> {X.T.shape}")
                X = X.T
                n_samples, n_features = X.shape
        
        # General validation - biological data should have samples <= features in most cases
        if n_samples > n_features * 10:
            logger.error(f"Extreme sample/feature ratio for {modality_name}: {n_samples}/{n_features}")
            raise DataOrientationValidationError(f"Suspicious data orientation for {modality_name}")
        
        # Validate minimum requirements
        if n_samples < 10:
            raise DataOrientationValidationError(f"Insufficient samples for {modality_name}: {n_samples}")
        
        logger.info(f"{modality_name} validated shape: {X.shape}")
        return X
    
    @staticmethod
    def validate_modality_consistency(modality_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Ensure all modalities have consistent sample counts after orientation fixes.
        
        Parameters
        ----------
        modality_dict : Dict[str, np.ndarray]
            Dictionary of modality data
            
        Returns
        -------
        Dict[str, np.ndarray]
            Validated modality dictionary
        """
        validated_dict = {}
        sample_counts = {}
        
        # Validate each modality
        for modality, X in modality_dict.items():
            validated_X = DataOrientationValidator.validate_data_orientation(X, modality)
            validated_dict[modality] = validated_X
            sample_counts[modality] = validated_X.shape[0]
        
        # Check for sample count consistency
        unique_counts = set(sample_counts.values())
        if len(unique_counts) > 1:
            logger.warning(f"Inconsistent sample counts across modalities: {sample_counts}")
        
        return validated_dict

def get_file_hash(file_path: Path) -> str:
    """Generate a hash for file caching based on path and modification time."""
    stat = file_path.stat()
    content = f"{file_path}_{stat.st_size}_{stat.st_mtime}"
    return hashlib.md5(content.encode()).hexdigest()

def cache_modality(cache_key: str, data: pd.DataFrame) -> None:
    """Cache modality data with thread safety."""
    with _cache_lock:
        _modality_cache[cache_key] = data.copy()

def get_cached_modality(cache_key: str) -> Optional[pd.DataFrame]:
    """Retrieve cached modality data with thread safety."""
    with _cache_lock:
        return _modality_cache.get(cache_key)

def clear_modality_cache() -> None:
    """Clear the modality cache to free memory."""
    with _cache_lock:
        _modality_cache.clear()
        logger.info("Cleared modality cache")

def parse_malformed_header(header_string: str) -> List[str]:
    """
    Parse a malformed header string that contains multiple sample IDs.
    Optimized for the specific format seen in TCGA data files.
    
    Parameters
    ----------
    header_string : str
        Header string containing sample IDs (possibly quoted and concatenated)
        
    Returns
    -------
    List[str]
        List of extracted sample IDs
    """
    # Remove quotes and clean the string
    cleaned = header_string.replace('"', '').replace("'", '').strip()
    
    # Strategy 1: Split by spaces (most common for malformed headers like the AML data)
    if ' ' in cleaned:
        potential_ids = [id_str.strip() for id_str in cleaned.split() if id_str.strip()]
        # Validate that these look like TCGA IDs
        tcga_ids = [id_str for id_str in potential_ids if re.match(r'TCGA[.\-_][A-Z0-9]+[.\-_][A-Z0-9]+[.\-_][0-9]+', id_str)]
        if len(tcga_ids) > 5:  # If we found many TCGA IDs, use them
            logger.debug(f"Extracted {len(tcga_ids)} TCGA sample IDs from space-separated header")
            return tcga_ids
        elif len(potential_ids) > 5:  # Otherwise use all potential IDs
            logger.debug(f"Extracted {len(potential_ids)} sample IDs from space-separated header")
            return potential_ids
    
    # Strategy 2: Use regex to find TCGA-like patterns
    tcga_pattern = r'TCGA[.\-_][A-Z0-9]+[.\-_][A-Z0-9]+[.\-_][0-9]+'
    regex_matches = re.findall(tcga_pattern, cleaned)
    if len(regex_matches) > 5:
        logger.debug(f"Extracted {len(regex_matches)} TCGA sample IDs using regex")
        return regex_matches
    
    # Strategy 3: Split by common separators if no spaces
    for sep in ['\t', ',', ';', '|']:
        if sep in cleaned:
            potential_ids = [id_str.strip() for id_str in cleaned.split(sep) if id_str.strip()]
            if len(potential_ids) > 5:
                logger.debug(f"Extracted {len(potential_ids)} sample IDs using separator '{sep}'")
                return potential_ids
    
    logger.debug("Could not extract sample IDs from malformed header")
    return []

def fix_malformed_data_file(file_path: Path, modality_name: str) -> Optional[pd.DataFrame]:
    """
    Attempt to fix malformed data files where sample IDs are in a single header string.
    Optimized for the specific format seen in TCGA data files.
    
    Parameters
    ----------
    file_path : Path
        Path to the malformed data file
    modality_name : str
        Name of the modality for logging
        
    Returns
    -------
    pd.DataFrame or None
        Fixed DataFrame with proper sample columns, or None if unfixable
    """
    try:
        logger.info(f"Attempting to repair malformed {modality_name} file: {file_path}")
        
        # Read the file as text to examine structure
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if len(lines) < 2:
            logger.warning(f"File {file_path} has insufficient lines")
            return None
        
        # Extract sample IDs from the malformed header
        first_line = lines[0].strip()
        sample_ids = parse_malformed_header(first_line)
        
        if len(sample_ids) < 5:  # Not enough samples to be worth fixing
            logger.warning(f"Not enough sample IDs found in header for {modality_name}: {len(sample_ids)}")
            return None
        
        logger.info(f"Found {len(sample_ids)} sample IDs in malformed header")
        
        # Try different strategies to read the data part
        df_data = None
        
        # Strategy 1: Try reading without header, then manually assign columns
        try:
            df_data = pd.read_csv(file_path, sep=None, engine='python', header=None, 
                                skiprows=1, index_col=0, low_memory=False)
            logger.debug(f"Successfully read data part: {df_data.shape}")
        except Exception as e:
            logger.debug(f"Strategy 1 failed: {str(e)}")
        
        # Strategy 2: If that fails, try with specific delimiter
        if df_data is None:
            for delimiter in [',', '\t', ' ']:
                try:
                    df_data = pd.read_csv(file_path, sep=delimiter, header=None, 
                                        skiprows=1, index_col=0, low_memory=False)
                    if not df_data.empty:
                        logger.debug(f"Successfully read data with delimiter '{delimiter}': {df_data.shape}")
                        break
                except Exception as e:
                    logger.debug(f"Failed with delimiter '{delimiter}': {str(e)}")
                    continue
        
        if df_data is None or df_data.empty:
            logger.warning(f"Could not read data part of malformed file")
            return None
        
        # Adjust dimensions if needed
        if df_data.shape[1] != len(sample_ids):
            logger.warning(f"Dimension mismatch: data has {df_data.shape[1]} columns, header has {len(sample_ids)} sample IDs")
            
            if df_data.shape[1] < len(sample_ids):
                # Truncate sample IDs to match data
                sample_ids = sample_ids[:df_data.shape[1]]
                logger.info(f"Truncated sample IDs to {len(sample_ids)} to match data columns")
            else:
                # Pad with generic names or truncate data
                if df_data.shape[1] - len(sample_ids) < 10:  # Small difference, pad sample IDs
                    while len(sample_ids) < df_data.shape[1]:
                        sample_ids.append(f"Sample_{len(sample_ids)+1}")
                    logger.info(f"Padded sample IDs to {len(sample_ids)}")
                else:
                    # Large difference, truncate data
                    df_data = df_data.iloc[:, :len(sample_ids)]
                    logger.info(f"Truncated data to {df_data.shape[1]} columns to match sample IDs")
        
        # Assign the sample IDs as column names
        df_data.columns = sample_ids
        
        # Validate the result
        if df_data.empty or df_data.shape[1] == 0:
            logger.warning(f"Repaired data is empty")
            return None
        
        # Check for reasonable data values
        if modality_name.lower() in ['gene expression', 'mirna']:
            # Expression data should be mostly positive
            negative_ratio = (df_data < 0).sum().sum() / (df_data.shape[0] * df_data.shape[1])
            if negative_ratio > 0.5:
                logger.warning(f"High proportion of negative values ({negative_ratio:.1%}) in {modality_name} - data may be corrupted")
        
        logger.info(f"Successfully repaired malformed {modality_name} file: {df_data.shape}")
        return df_data
        
    except Exception as e:
        logger.error(f"Failed to repair malformed file {file_path}: {str(e)}")
        return None

def standardize_sample_ids(sample_ids: List[str], target_format: str = 'hyphen') -> Dict[str, str]:
    """
    Standardize sample IDs to a consistent format.
    
    Parameters
    ----------
    sample_ids : List[str]
        List of sample IDs to standardize
    target_format : str
        Target format: 'hyphen' for TCGA-XX-XXXX, 'dot' for TCGA.XX.XXXX
        
    Returns
    -------
    Dict[str, str]
        Mapping from original ID to standardized ID
    """
    target_sep = '-' if target_format == 'hyphen' else '.'
    mapping = {}
    
    for original_id in sample_ids:
        if not isinstance(original_id, str):
            continue
            
        # Convert separators to target format
        standardized = original_id
        for sep in ['.', '-', '_']:
            if sep != target_sep:
                standardized = standardized.replace(sep, target_sep)
        
        mapping[original_id] = standardized
    
    return mapping

def validate_data_quality(df: pd.DataFrame, modality_name: str) -> Tuple[bool, str]:
    """
    Validate data quality and return issues found.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    modality_name : str
        Name of the modality for logging
        
    Returns
    -------
    Tuple[bool, str]
        (is_valid, issues_description)
    """
    issues = []
    
    # Check basic structure
    if df.empty:
        issues.append("DataFrame is empty")
    
    if df.shape[0] == 0:
        issues.append("No features/rows")
    
    if df.shape[1] == 0:
        issues.append("No samples/columns")
    
    # Check for excessive missing data
    missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
    if missing_ratio > 0.8:
        issues.append(f"Excessive missing data: {missing_ratio:.1%}")
    
    # Check sample ID format
    sample_pattern = r'TCGA[.\-_][A-Z0-9]+[.\-_][A-Z0-9]+[.\-_][0-9]+'
    valid_samples = sum(1 for col in df.columns if re.match(sample_pattern, str(col)))
    if valid_samples < df.shape[1] * 0.5:
        issues.append(f"Many invalid sample IDs: only {valid_samples}/{df.shape[1]} match TCGA pattern")
    
    # Check for duplicate columns
    if df.columns.duplicated().any():
        issues.append("Duplicate sample IDs found")
    
    is_valid = len(issues) == 0
    issues_str = "; ".join(issues) if issues else "No issues found"
    
    logger.info(f"Data quality check for {modality_name}: {'PASS' if is_valid else 'ISSUES'} - {issues_str}")
    return is_valid, issues_str

def find_fuzzy_id_matches(outcome_ids: List[str], modality_ids: List[str], 
                         similarity_threshold: float = 0.8) -> Dict[str, str]:
    """
    Find fuzzy matches between outcome and modality sample IDs.
    
    Parameters
    ----------
    outcome_ids : List[str]
        List of outcome sample IDs
    modality_ids : List[str]
        List of modality sample IDs
    similarity_threshold : float
        Minimum similarity score for a match
        
    Returns
    -------
    Dict[str, str]
        Mapping from outcome ID to best matching modality ID
    """
    from difflib import SequenceMatcher
    
    matches = {}
    
    for outcome_id in outcome_ids:
        best_match = None
        best_score = 0
        
        for modality_id in modality_ids:
            # Calculate similarity
            similarity = SequenceMatcher(None, outcome_id.lower(), modality_id.lower()).ratio()
            
            if similarity > best_score and similarity >= similarity_threshold:
                best_score = similarity
                best_match = modality_id
        
        if best_match:
            matches[outcome_id] = best_match
    
    return matches

def find_pattern_matches(outcome_ids: List[str], modality_ids: List[str]) -> Dict[str, str]:
    """
    Find pattern-based matches for TCGA-style IDs.
    
    Parameters
    ----------
    outcome_ids : List[str]
        List of outcome sample IDs
    modality_ids : List[str]
        List of modality sample IDs
        
    Returns
    -------
    Dict[str, str]
        Mapping from outcome ID to best matching modality ID
    """
    matches = {}
    
    # Extract core patterns (first 3 parts of TCGA IDs)
    def extract_core_pattern(id_str):
        parts = re.split(r'[.\-_]', id_str)
        if len(parts) >= 3:
            return '-'.join(parts[:3])
        return id_str
    
    # Create pattern mappings
    outcome_patterns = {extract_core_pattern(id_): id_ for id_ in outcome_ids}
    modality_patterns = {extract_core_pattern(id_): id_ for id_ in modality_ids}
    
    # Find matches
    for pattern, outcome_id in outcome_patterns.items():
        if pattern in modality_patterns:
            matches[outcome_id] = modality_patterns[pattern]
    
    return matches

def find_relaxed_intersection(outcome_ids: List[str], modalities: Dict[str, pd.DataFrame], 
                             min_modalities: int = 2) -> List[str]:
    """
    Find samples present in at least min_modalities modalities.
    
    Parameters
    ----------
    outcome_ids : List[str]
        List of outcome sample IDs
    modalities : Dict[str, pd.DataFrame]
        Dictionary of modality DataFrames
    min_modalities : int
        Minimum number of modalities a sample must be present in
        
    Returns
    -------
    List[str]
        List of sample IDs meeting the criteria
    """
    sample_counts = {}
    
    # Count how many modalities each sample appears in
    for outcome_id in outcome_ids:
        count = 0
        for mod_name, df in modalities.items():
            if outcome_id in df.columns:
                count += 1
        sample_counts[outcome_id] = count
    
    # Return samples present in at least min_modalities
    relaxed_samples = [id_ for id_, count in sample_counts.items() if count >= min_modalities]
    
    logger.info(f"Relaxed intersection: {len(relaxed_samples)} samples present in >= {min_modalities} modalities")
    return relaxed_samples

def try_read_file(path: Union[str, Path], 
                 clinical_cols: Optional[List[str]] = None, 
                 id_col: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Try to read a file, return None if it fails.
    Uses multiple strategies to handle different file formats.

    Parameters
    ----------
    path            Path to the file
    clinical_cols   List of clinical column names to keep
    id_col          ID column name that contains patient identifiers
    
    Returns
    -------
    pd.DataFrame or None
    """
    # Convert to Path object
    path = Path(path)
    
    # Skip if file doesn't exist
    if not path.exists():
        # Try with forward slashes for Windows compatibility
        alt_path = Path(str(path).replace('\\', '/'))
        if alt_path.exists():
            path = alt_path
        else:
            logger.warning(f"Warning: File {path} does not exist")
            return None
    
    # Try reading with different delimiters
    for delimiter in [',', '\t']:
        try:
            df = pd.read_csv(path, sep=delimiter, index_col=0 if id_col is None else None)
            
            # If specified ID column, set it as index
            if id_col is not None and id_col in df.columns:
                df = df.set_index(id_col)
            
            # If clinical columns specified, keep only those
            if clinical_cols is not None:
                cols_to_keep = [c for c in clinical_cols if c in df.columns]
                if cols_to_keep:
                    df = df[cols_to_keep]
                else:
                    logger.warning(f"Warning: None of the specified clinical columns found in {path}")
            
            return df
        except Exception:
            continue
    
    logger.error(f"Error reading {path}: Could not parse with any delimiter")
    return None

def optimize_data_types(df: pd.DataFrame, modality_name: str) -> pd.DataFrame:
    """
    Optimize data types for memory efficiency and performance.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to optimize
    modality_name : str
        Name of the modality for logging
        
    Returns
    -------
    pd.DataFrame
        DataFrame with optimized data types
    """
    original_memory = df.memory_usage(deep=True).sum() / 1024**2  # MB
    
    # Convert numeric columns to more efficient types
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to convert to numeric
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                pass
        
        # Optimize numeric types
        if df[col].dtype in ['int64', 'int32']:
            # Check if we can use smaller integer types
            col_min, col_max = df[col].min(), df[col].max()
            if col_min >= -128 and col_max <= 127:
                df[col] = df[col].astype('int8')
            elif col_min >= -32768 and col_max <= 32767:
                df[col] = df[col].astype('int16')
            elif col_min >= -2147483648 and col_max <= 2147483647:
                df[col] = df[col].astype('int32')
        
        elif df[col].dtype in ['float64', 'float32']:
            # Use float32 for most genomic data (sufficient precision)
            if df[col].dtype == 'float64':
                # Check if we can safely convert to float32
                try:
                    df_test = df[col].astype('float32')
                    if np.allclose(df[col].values, df_test.values, equal_nan=True):
                        df[col] = df_test
                except:
                    pass  # Keep as float64 if conversion fails
    
    new_memory = df.memory_usage(deep=True).sum() / 1024**2  # MB
    memory_saved = original_memory - new_memory
    
    if memory_saved > 0.1:  # Only log if significant savings
        logger.info(f"Memory optimization for {modality_name}: {original_memory:.1f}MB -> {new_memory:.1f}MB (saved {memory_saved:.1f}MB)")
    
    return df

def load_modality_chunked(file_path: Path, modality_name: str, chunk_size: int = 10000) -> Optional[pd.DataFrame]:
    """
    Load large modality files in chunks for memory efficiency.
    
    Parameters
    ----------
    file_path : Path
        Path to the data file
    modality_name : str
        Name of the modality
    chunk_size : int
        Number of rows to read at a time
        
    Returns
    -------
    pd.DataFrame or None
        Loaded DataFrame or None if failed
    """
    try:
        # First, try to determine file size and structure
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        # If file is small enough, load normally
        if file_size_mb < 100:  # Less than 100MB
            return None  # Signal to use normal loading
        
        logger.info(f"Large file detected ({file_size_mb:.1f}MB), using chunked loading for {modality_name}")
        
        # Check for malformed header first by reading the first line
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
        
        # If the first line has many TCGA IDs, it's malformed
        if first_line.count('TCGA') > 10:
            logger.info(f"Malformed large file detected for {modality_name} ({first_line.count('TCGA')} TCGA IDs in header), falling back to repair method")
            return None  # Let the repair method handle it
        
        # Read first chunk to determine structure for normal files
        first_chunk = pd.read_csv(file_path, nrows=100, index_col=0)
        
        # Additional check for other malformed patterns
        if first_chunk.shape[1] == 0:
            logger.info(f"Empty columns detected for {modality_name}, falling back to repair method")
            return None
        
        # Read the file in chunks
        chunks = []
        total_rows = 0
        
        for chunk in pd.read_csv(file_path, chunksize=chunk_size, index_col=0):
            chunks.append(chunk)
            total_rows += len(chunk)
            
            # Progress logging for very large files
            if len(chunks) % 10 == 0:
                logger.debug(f"Loaded {total_rows} rows for {modality_name}")
        
        # Concatenate all chunks
        df = pd.concat(chunks, axis=0)
        logger.info(f"Successfully loaded {modality_name} in {len(chunks)} chunks: {df.shape}")
        
        return df
        
    except Exception as e:
        logger.warning(f"Chunked loading failed for {modality_name}: {str(e)}")
        return None

def preprocess_genomic_data(df: pd.DataFrame, modality_name: str) -> pd.DataFrame:
    """
    Apply genomic data-specific preprocessing optimizations.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw genomic data
    modality_name : str
        Type of genomic data (Gene Expression, miRNA, Methylation)
        
    Returns
    -------
    pd.DataFrame
        Preprocessed data
    """
    logger.info(f"Applying genomic preprocessing for {modality_name}")
    
    # Remove features with too many missing values
    missing_threshold = 0.5  # Remove features missing in >50% of samples
    missing_ratio = df.isnull().sum(axis=1) / df.shape[1]
    features_to_keep = missing_ratio <= missing_threshold
    
    if not features_to_keep.all():
        removed_count = (~features_to_keep).sum()
        logger.info(f"Removing {removed_count} features with >{missing_threshold*100}% missing values")
        df = df[features_to_keep]
    
    # Handle remaining missing values
    if df.isnull().any().any():
        if modality_name.lower() in ['gene expression', 'mirna']:
            # For expression data, use 0 (no expression)
            df = df.fillna(0)
            logger.debug(f"Filled missing values with 0 for {modality_name}")
        elif modality_name.lower() == 'methylation':
            # For methylation, use median (more appropriate than 0)
            df = df.fillna(df.median(axis=1), axis=0)
            logger.debug(f"Filled missing values with median for {modality_name}")
        else:
            # For other data types, use median
            df = df.fillna(df.median(axis=1), axis=0)
            logger.debug(f"Filled missing values with median for {modality_name}")
    
    # Remove constant features (no MAD - more robust than variance)
    if modality_name.lower() != 'methylation':  # Methylation can have legitimate constant values
        # Calculate MAD for each feature (row-wise since features are in rows)
        feature_mad = df.apply(lambda row: np.median(np.abs(row - np.median(row))) * 1.4826, axis=1)
        non_constant = feature_mad > 1e-8  # Very small threshold for numerical stability
        
        if not non_constant.all():
            removed_count = (~non_constant).sum()
            logger.info(f"Removing {removed_count} constant features from {modality_name} using MAD")
            df = df[non_constant]
    
    # Apply log transformation for expression data
    if modality_name.lower() in ['gene expression', 'mirna']:
        # Check if data appears to be already log-transformed
        max_val = df.max().max()
        min_val = df.min().min()
        
        # Only apply log transformation if data appears to be raw counts
        if max_val > 50 and min_val >= 0:  # Likely raw counts (non-negative, large values)
            logger.info(f"Applying log2(x+1) transformation to {modality_name}")
            df = np.log2(df + 1)
        elif min_val < 0:
            # Data contains negative values - likely already processed/normalized
            logger.info(f"Data contains negative values, skipping log transformation for {modality_name}")
        elif max_val <= 50:
            # Data appears already log-transformed or normalized
            logger.info(f"Data appears already transformed (max={max_val:.2f}), skipping log transformation for {modality_name}")
        else:
            # Edge case: positive data with moderate values - apply safe log transformation
            logger.info(f"Applying safe log transformation to {modality_name}")
            # Ensure all values are positive before log transformation
            df_shifted = df - df.min().min() + 1e-6  # Shift to make all values positive
            df = np.log2(df_shifted + 1)
    
    return df

def load_modality(base_path: Union[str, Path], 
                 modality_path: Union[str, Path], 
                 modality_name: str,
                 k_features: int = MAX_VARIABLE_FEATURES,
                 chunk_size: int = 10000,
                 use_cache: bool = True) -> Optional[pd.DataFrame]:
    """
    Load a single modality file with comprehensive optimizations.
    OPTIMIZED VERSION - Reduced unnecessary file repair attempts for better performance.
    
    Parameters
    ----------
    base_path       Base path for the dataset
    modality_path   Path to modality file (relative to base_path)
    modality_name   Name of the modality
    k_features      Maximum number of features to keep
    chunk_size      Number of rows to read at a time when loading large files
    use_cache       Whether to use caching for faster repeated loads
    
    Returns
    -------
    pd.DataFrame or None - formatted with samples as columns and features as rows
    """
    # Try different path combinations to handle platform differences
    paths_to_try = [
        Path(base_path) / modality_path,
        Path(f"{str(base_path).replace('\\', '/')}/{str(modality_path).replace('\\', '/').lstrip('/')}"),
        Path(modality_path)  # Direct path as last resort
    ]
    
    # Find the first valid path
    valid_path = None
    for path in paths_to_try:
        if path.exists():
            valid_path = path
            logger.debug(f"Found valid path for {modality_name}: {path}")
            break
    
    if valid_path is None:
        logger.warning(f"Warning: Could not find valid path for modality {modality_name}")
        return None
    
    # Check cache first if enabled (before any processing)
    cache_key = None
    if use_cache:
        cache_key = f"{get_file_hash(valid_path)}_{modality_name}_{k_features}"
        cached_data = get_cached_modality(cache_key)
        if cached_data is not None:
            logger.info(f"Using cached data for {modality_name}: {cached_data.shape}")
            return cached_data.copy()
    
    # OPTIMIZED: Try standard loading first with common delimiters
    df = None
    encodings_to_try = ['utf-8', 'latin1']  # Reduced encoding attempts
    delimiters_to_try = [',', '\t']  # Most common delimiters first
    
    for delimiter in delimiters_to_try:
        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(valid_path, sep=delimiter, index_col=0, encoding=encoding, low_memory=False)
                if not df.empty and df.shape[1] > 5:  # Reasonable number of samples
                    logger.debug(f"Successfully loaded {modality_name} with delimiter='{delimiter}', encoding='{encoding}'")
                    break
            except Exception as e:
                logger.debug(f"Failed with delimiter='{delimiter}', encoding='{encoding}': {str(e)}")
                continue
        
        if df is not None and not df.empty:
            break
    
    # OPTIMIZED: Only attempt repair if standard loading completely failed
    if df is None or df.empty:
        logger.info(f"Standard loading failed for {modality_name}, attempting repair")
        df = fix_malformed_data_file(valid_path, modality_name)
    
    # OPTIMIZED: Quick malformed structure check only if needed
    elif df.shape[1] == 1:
        # Check if the single column contains multiple sample IDs (malformed)
        if len(parse_malformed_header(df.columns[0])) > 5:
            logger.info(f"Detected malformed structure in {modality_name}, attempting repair")
            df = fix_malformed_data_file(valid_path, modality_name)
    
    if df is None or df.empty:
        logger.warning(f"Warning: Could not load modality {modality_name}")
        return None
    
    # OPTIMIZED: Simplified data quality validation
    if df.isnull().all().all():
        logger.warning(f"Data quality issue in {modality_name}: All values are null")
        return None
    
    logger.info(f"Loaded {modality_name} data: shape={df.shape}")
    
    # ENHANCED: Use sophisticated orientation validation (moved from preprocessing.py)
    try:
        df = DataOrientationValidator.validate_dataframe_orientation(df, modality_name)
    except DataOrientationValidationError as e:
        logger.error(f"Critical orientation validation error for {modality_name}: {str(e)}")
        return None
    
    # Ensure index/column names are strings
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    
    # Standardize sample IDs to use hyphens (clinical data format)
    logger.info(f"Standardizing sample IDs for {modality_name}")
    id_mapping = standardize_sample_ids(df.columns.tolist(), target_format='hyphen')
    if id_mapping:
        df = df.rename(columns=id_mapping)
        logger.info(f"Standardized {len(id_mapping)} sample IDs in {modality_name}")
    
    # OPTIMIZED: Handle duplicates only if they exist
    if df.index.duplicated().any():
        logger.warning(f"Found duplicate feature names in {modality_name}, making them unique")
        df.index = pd.Index([f"{idx}_{i}" if i > 0 else idx 
                            for i, idx in enumerate(df.groupby(df.index).cumcount().add(df.index))])
    
    if df.columns.duplicated().any():
        logger.warning(f"Found duplicate sample IDs in {modality_name}, making them unique")
        df.columns = pd.Index([f"{col}_{i}" if i > 0 else col 
                              for i, col in enumerate(df.groupby(df.columns).cumcount().add(df.columns))])
    
    # Apply genomic data preprocessing
    df = preprocess_genomic_data(df, modality_name)
    
    # Apply advanced filtering BEFORE MAD filtering for better feature quality
    logger.info(f"Applying advanced feature filtering to {modality_name}")
    df = advanced_feature_filtering(df)
    
    # Apply MAD filtering AFTER advanced filtering for final feature count
    if df.shape[0] > k_features:
        logger.info(f"Applying MAD filtering to {modality_name}, keeping top {k_features} most variable features")
        df = _keep_top_variable_rows(df, k=k_features)
    
    # Optimize data types for memory efficiency AFTER filtering
    df = optimize_data_types(df, modality_name)
    
    # Final validation and logging
    logger.info(f"Final {modality_name} data: {df.shape[0]} features x {df.shape[1]} samples")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f}MB")
    
    # Cache the processed data if caching is enabled
    if use_cache:
        cache_modality(cache_key, df)
        logger.debug(f"Cached processed data for {modality_name}")
    
    return df

def load_outcome(base_path: Union[str, Path], 
                outcome_file: str, 
                outcome_col: str, 
                id_col: str,
                outcome_type: str = "os",
                dataset: str = None) -> Tuple[Optional[pd.Series], Optional[pd.DataFrame]]:
    """
    Load outcome data from a CSV file.
    
    Parameters
    ----------
    base_path       Base path for the dataset
    outcome_file    Path to outcome file (relative to base_path)
    outcome_col     Column name containing outcome data
    id_col          Column name containing sample IDs
    outcome_type    Type of outcome data
    
    Returns
    -------
    Tuple of (outcome Series, full DataFrame) or (None, None)
    """
    try:
        # Construct full file path - handle absolute vs relative paths
        outcome_file_str = str(outcome_file).replace('\\', '/')
        
        if outcome_file_str.startswith('data/') or Path(outcome_file).is_absolute():
            # If outcome_file is already a full path from project root or absolute, use as-is
            outcome_path = Path(outcome_file)
        else:
            # Otherwise, join with base_path
            outcome_path = Path(base_path) / outcome_file
        
        logger.info(f"Loading outcome data from: {outcome_path}")
        
        # Try different delimiters and encodings
        delimiters = ['\t', ',', ';', ' ']
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        
        outcome_df = None
        for delimiter in delimiters:
            for encoding in encodings:
                try:
                    outcome_df = pd.read_csv(outcome_path, sep=delimiter, encoding=encoding, index_col=False)
                    if len(outcome_df.columns) > 1 and not outcome_df.empty:
                        logger.info(f"Successfully loaded outcome file with delimiter='{delimiter}', encoding='{encoding}'")
                        break
                except Exception:
                    continue
            if outcome_df is not None and len(outcome_df.columns) > 1:
                break
        
        if outcome_df is None or outcome_df.empty:
            logger.error(f"Failed to load outcome file: {outcome_path}")
            return None, None
        
        logger.info(f"Outcome data shape: {outcome_df.shape}")
        logger.info(f"Available columns: {list(outcome_df.columns)}")
        logger.info(f"Looking for ID column: '{id_col}' and outcome column: '{outcome_col}'")
        
        # Check if the required columns exist
        if id_col not in outcome_df.columns:
            logger.error(f"ID column '{id_col}' not found in outcome data")
            return None, None
        
        if outcome_col not in outcome_df.columns:
            logger.error(f"Outcome column '{outcome_col}' not found in outcome data")
            return None, None
        
        # Extract the outcome column safely
        try:
            logger.info(f"Extracting outcome column: {outcome_col}")
            outcome_series = outcome_df[outcome_col]
            logger.info(f"Outcome series type: {type(outcome_series)}")
            logger.info(f"Outcome series dtype: {outcome_series.dtype}")
            logger.info(f"First few outcome values: {outcome_series.head()}")
            
            # Add safety check before calling custom_parse_outcome
            if isinstance(outcome_series, np.ndarray):
                logger.info("Converting numpy.ndarray to pandas Series before parsing")
                outcome_series = pd.Series(outcome_series, name=outcome_col)
            
            # Parse the outcome based on its type
            logger.info(f"Parsing outcome with type: {outcome_type}")
            parsed_outcome = custom_parse_outcome(outcome_series, outcome_type, dataset)
            logger.info(f"Parsed outcome type: {type(parsed_outcome)}")
            
        except Exception as e:
            logger.error(f"Error extracting/parsing outcome column: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None, None
        
        # Extract sample IDs
        try:
            logger.info(f"Extracting ID column: {id_col}")
            sample_ids = outcome_df[id_col]
            logger.info(f"Sample IDs type: {type(sample_ids)}")
            logger.info(f"Sample IDs dtype: {sample_ids.dtype}")
            logger.info(f"First few sample IDs: {sample_ids.head()}")
            logger.info(f"Sample IDs length: {len(sample_ids)}")
            
            # Add safety check for sample IDs as well
            if isinstance(sample_ids, np.ndarray):
                logger.info("Converting numpy.ndarray sample IDs to pandas Series")
                sample_ids = pd.Series(sample_ids, name=id_col)
            
            # Get the values safely - handle both pandas Series and numpy arrays
            if hasattr(parsed_outcome, 'values'):
                outcome_values = parsed_outcome.values
            else:
                outcome_values = np.array(parsed_outcome)
                
            if hasattr(sample_ids, 'values'):
                id_values = sample_ids.values
            else:
                id_values = np.array(sample_ids)
            
            logger.info(f"ID values type: {type(id_values)}")
            logger.info(f"First few ID values: {id_values[:5]}")
            logger.info(f"Outcome values type: {type(outcome_values)}")
            logger.info(f"First few outcome values: {outcome_values[:5]}")
            
            # Create indexed outcome series
            outcome_indexed = pd.Series(outcome_values, index=id_values, name=outcome_col)
            logger.info(f"Outcome indexed shape: {outcome_indexed.shape}")
            logger.info(f"First few outcome indexed index: {outcome_indexed.index[:5].tolist()}")
            
        except Exception as e:
            logger.error(f"Error extracting sample IDs: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None, None
        
        # Remove samples with missing outcome data
        outcome_clean = outcome_indexed.dropna()
        logger.info(f"Loaded {len(outcome_clean)} samples with valid outcome data (removed {len(outcome_indexed) - len(outcome_clean)} missing values)")
        
        return outcome_clean, outcome_df
        
    except Exception as e:
        logger.error(f"Error loading outcome data: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None, None

def normalize_sample_ids(sample_ids: List[str], target_separator: str = '-') -> Dict[str, str]:
    """
    Normalize sample IDs to use a consistent separator format.
    
    Parameters
    ----------
    sample_ids : List[str]
        List of sample IDs to normalize
    target_separator : str
        Target separator to use (default: '-')
        
    Returns
    -------
    Dict[str, str]
        Mapping from original ID to normalized ID
    """
    id_mapping = {}
    
    for sample_id in sample_ids:
        if not isinstance(sample_id, str):
            continue
            
        # Replace common separators with target separator
        normalized_id = sample_id
        for sep in ['.', '_', ' ', '+']:
            if sep != target_separator:
                normalized_id = normalized_id.replace(sep, target_separator)
        
        # Only add to mapping if there was a change
        if normalized_id != sample_id:
            id_mapping[sample_id] = normalized_id
    
    return id_mapping

def enhanced_sample_recovery(modalities: Dict[str, pd.DataFrame], 
                           y_series: pd.Series, 
                           ds_name: str,
                           current_common_ids: List[str]) -> Tuple[List[str], Dict[str, pd.DataFrame]]:
    """
    Enhanced sample recovery using aggressive ID matching strategies.
    
    Parameters
    ----------
    modalities : Dict[str, pd.DataFrame]
        Dictionary of modality DataFrames
    y_series : pd.Series
        Outcome data series
    ds_name : str
        Dataset name for logging
    current_common_ids : List[str]
        Currently identified common sample IDs
        
    Returns
    -------
    Tuple[List[str], Dict[str, pd.DataFrame]]
        (enhanced_common_sample_ids, enhanced_filtered_modalities)
    """
    logger.info("=== ENHANCED SAMPLE RECOVERY ===")
    
    # Start with outcome samples not yet recovered
    outcome_samples = set(y_series.index)
    missing_samples = outcome_samples - set(current_common_ids)
    logger.info(f"Attempting to recover {len(missing_samples)} missing samples")
    
    if not missing_samples:
        return current_common_ids, modalities
    
    # Enhanced ID standardization strategies
    recovered_samples = set(current_common_ids)
    enhanced_modalities = {}
    
    for mod_name, df in modalities.items():
        mod_samples = set(df.columns)
        enhanced_df = df.copy()
        
        # Strategy 1: More aggressive separator normalization
        id_mapping = {}
        for missing_id in missing_samples:
            # Try different separator combinations
            for sep_combo in [('.', '-'), ('-', '.'), ('_', '-'), ('-', '_')]:
                test_id = missing_id.replace(sep_combo[0], sep_combo[1])
                if test_id in mod_samples:
                    id_mapping[test_id] = missing_id
                    logger.debug(f"Separator recovery: {test_id} -> {missing_id}")
                    break
        
        # Strategy 2: Partial ID matching (first 12 characters for TCGA IDs)
        for missing_id in missing_samples:
            if missing_id in id_mapping.values():
                continue  # Already mapped
            missing_prefix = missing_id[:12]  # TCGA-XX-XXXX
            for mod_id in mod_samples:
                if mod_id in id_mapping:
                    continue  # Already mapped
                mod_prefix = mod_id[:12].replace('.', '-').replace('_', '-')
                if missing_prefix == mod_prefix:
                    id_mapping[mod_id] = missing_id
                    logger.debug(f"Prefix recovery: {mod_id} -> {missing_id}")
                    break
        
        # Strategy 3: Case-insensitive matching
        if not id_mapping:
            missing_lower = {mid.lower(): mid for mid in missing_samples}
            for mod_id in mod_samples:
                mod_lower = mod_id.lower().replace('.', '-').replace('_', '-')
                if mod_lower in missing_lower:
                    id_mapping[mod_id] = missing_lower[mod_lower]
                    logger.debug(f"Case recovery: {mod_id} -> {missing_lower[mod_lower]}")
        
        # Apply ID mapping
        if id_mapping:
            enhanced_df = enhanced_df.rename(columns=id_mapping)
            logger.info(f"Enhanced recovery for {mod_name}: +{len(id_mapping)} samples")
        
        enhanced_modalities[mod_name] = enhanced_df
    
    # Recalculate intersection with enhanced modalities
    enhanced_common = set(y_series.index)
    for mod_name, df in enhanced_modalities.items():
        enhanced_common = enhanced_common.intersection(set(df.columns))
    
    # Filter modalities to enhanced common samples
    final_enhanced_modalities = {}
    for mod_name, df in enhanced_modalities.items():
        available_samples = [col for col in enhanced_common if col in df.columns]
        if available_samples:
            final_enhanced_modalities[mod_name] = df[available_samples]
    
    enhanced_common_list = sorted(list(enhanced_common))
    
    logger.info(f"Enhanced recovery result: {len(current_common_ids)} -> {len(enhanced_common_list)} samples")
    
    return enhanced_common_list, final_enhanced_modalities

def optimize_sample_intersection(modalities: Dict[str, pd.DataFrame], 
                                y_series: pd.Series, 
                                ds_name: str) -> Tuple[List[str], Dict[str, pd.DataFrame]]:
    """
    Optimize sample intersection with enhanced recovery strategies.
    
    Parameters
    ----------
    modalities : Dict[str, pd.DataFrame]
        Dictionary of modality DataFrames
    y_series : pd.Series
        Outcome data series
    ds_name : str
        Dataset name for logging
        
    Returns
    -------
    Tuple[List[str], Dict[str, pd.DataFrame]]
        (common_sample_ids, filtered_modalities)
    """
    logger.info("=== OPTIMIZED SAMPLE INTERSECTION ANALYSIS ===")
    
    # Start with outcome samples
    common_ids = set(y_series.index)
    original_outcome_samples = len(common_ids)
    logger.info(f"Starting with {original_outcome_samples} samples from outcome data")
    
    # Track intersection statistics
    intersection_stats = {
        'outcome_samples': original_outcome_samples,
        'modality_samples': {},
        'intersection_steps': [],
        'recovery_attempts': []
    }
    
    # Analyze each modality intersection with recovery strategies
    for mod_name, df in modalities.items():
        mod_samples = set(df.columns)
        intersection_stats['modality_samples'][mod_name] = len(mod_samples)
        
        logger.info(f"Processing {mod_name}: {len(mod_samples)} samples")
        if len(df.columns) > 0:
            logger.debug(f"First 5 {mod_name} samples: {list(df.columns)[:5]}")
        
        common_before = len(common_ids)
        common_ids_new = common_ids.intersection(mod_samples)
        lost_samples = common_ids - common_ids_new
        
        intersection_stats['intersection_steps'].append({
            'modality': mod_name,
            'before': common_before,
            'after': len(common_ids_new),
            'lost': len(lost_samples)
        })
        
        logger.info(f"After intersecting with {mod_name}: {len(common_ids_new)} common samples (lost {len(lost_samples)})")
        
        # If we lost significant samples, try recovery strategies
        if len(lost_samples) > 0 and len(lost_samples) / common_before > 0.1:  # Lost >10% of samples
            logger.info(f"Attempting sample recovery for {mod_name} (lost {len(lost_samples)} samples)")
            
            # Strategy 1: Fuzzy matching
            fuzzy_matches = find_fuzzy_id_matches(list(lost_samples), list(mod_samples), similarity_threshold=0.85)
            if fuzzy_matches:
                logger.info(f"Fuzzy matching found {len(fuzzy_matches)} recoverable samples")
                # Update the modality DataFrame
                reverse_mapping = {v: k for k, v in fuzzy_matches.items()}
                df_renamed = df.rename(columns=reverse_mapping)
                modalities[mod_name] = df_renamed
                mod_samples = set(df_renamed.columns)
                common_ids_new = common_ids.intersection(mod_samples)
                intersection_stats['recovery_attempts'].append({
                    'modality': mod_name,
                    'method': 'fuzzy_matching',
                    'recovered': len(fuzzy_matches)
                })
            
            # Strategy 2: Pattern-based matching for remaining samples
            remaining_lost = common_ids - common_ids_new
            if remaining_lost:
                pattern_matches = find_pattern_matches(list(remaining_lost), list(mod_samples))
                if pattern_matches:
                    logger.info(f"Pattern matching found {len(pattern_matches)} additional recoverable samples")
                    reverse_mapping = {v: k for k, v in pattern_matches.items()}
                    df_renamed = modalities[mod_name].rename(columns=reverse_mapping)
                    modalities[mod_name] = df_renamed
                    mod_samples = set(df_renamed.columns)
                    common_ids_new = common_ids.intersection(mod_samples)
                    intersection_stats['recovery_attempts'].append({
                        'modality': mod_name,
                        'method': 'pattern_matching',
                        'recovered': len(pattern_matches)
                    })
        
        common_ids = common_ids_new
        
        # Early termination if no samples left
        if len(common_ids) == 0:
            logger.error(f"No common samples remaining after {mod_name}")
            break
    
    # Log recovery statistics
    total_recovered = sum(attempt['recovered'] for attempt in intersection_stats['recovery_attempts'])
    if total_recovered > 0:
        logger.info(f"Sample recovery successful: {total_recovered} samples recovered across all modalities")
    
    # Final sample retention analysis
    final_samples = len(common_ids)
    retention_rate = (final_samples / original_outcome_samples) * 100 if original_outcome_samples > 0 else 0
    
    logger.info("=== FINAL INTERSECTION SUMMARY ===")
    logger.info(f"Original outcome samples: {original_outcome_samples}")
    logger.info(f"Final common samples: {final_samples}")
    logger.info(f"Sample retention rate: {retention_rate:.1f}%")
    
    # Filter modalities to only include common samples
    filtered_modalities = {}
    for mod_name, df in modalities.items():
        available_samples = [col for col in common_ids if col in df.columns]
        if available_samples:
            filtered_modalities[mod_name] = df[available_samples]
            logger.info(f"Filtered {mod_name}: {df.shape[0]} features x {len(available_samples)} samples")
        else:
            logger.warning(f"No common samples found in {mod_name} - excluding from analysis")
    
    # Provide recommendations based on retention rate with dataset-specific handling
    low_threshold = SAMPLE_RETENTION_CONFIG["low_retention_threshold"]
    moderate_threshold = SAMPLE_RETENTION_CONFIG["moderate_retention_threshold"]
    suppress_datasets = SAMPLE_RETENTION_CONFIG["suppress_warnings_for_datasets"]
    
    # Check if this dataset is expected to have low retention
    is_expected_low_retention = any(dataset in ds_name.lower() for dataset in suppress_datasets)
    
    if retention_rate < low_threshold:
        if is_expected_low_retention:
            # Suppress warnings for known problematic datasets
            logger.info(f"Sample retention: {retention_rate:.1f}% - {SAMPLE_RETENTION_CONFIG['expected_low_retention_message']}")
            if SAMPLE_RETENTION_CONFIG["log_retention_details"]:
                logger.debug("RETENTION DETAILS:")
                logger.debug("1. ID format differences between clinical and expression data")
                logger.debug("2. Malformed data files requiring repair")
                logger.debug("3. Quality filtering and class optimization")
        else:
            # Show warnings for unexpected low retention
            logger.warning("LOW SAMPLE RETENTION detected!")
            logger.warning("RECOMMENDATIONS:")
            logger.warning("1. Check for ID format mismatches between clinical and modality data")
            logger.warning("2. Consider using more aggressive fuzzy matching")
            logger.warning("3. Verify data file integrity")
    elif retention_rate < moderate_threshold:
        if is_expected_low_retention:
            logger.info(f"Sample retention: {retention_rate:.1f}% - within expected range for this dataset type")
        else:
            logger.warning("MODERATE SAMPLE RETENTION detected!")
            logger.info("MODERATE sample retention detected (this is normal for TCGA data)")
            logger.info("EXPLANATION: Sample loss occurs due to:")
            logger.info("1. ID format differences between clinical and expression data")
            logger.info("2. Malformed data files requiring repair")
            logger.info("3. Quality filtering and class optimization")
            logger.info("The system has already applied advanced recovery strategies")
    elif retention_rate >= 80:
        logger.info("EXCELLENT sample retention achieved!")
    
    return sorted(list(common_ids)), filtered_modalities

def optimize_class_distribution(y_series: pd.Series, 
                               common_ids: List[str], 
                               task_type: str = 'classification',
                               min_class_size: int = 5) -> Tuple[pd.Series, List[str]]:
    """
    Optimize class distribution for better CV performance.
    Generalized: merges rare classes for all datasets, not just those with 'T' in the label.
    For numeric classes, merges smallest classes with nearest neighbor.
    For string classes, merges rare classes into the most common class.
    
    Parameters
    ----------
    y_series : pd.Series
        Outcome data series
    common_ids : List[str]
        List of common sample IDs
    task_type : str
        Type of task ('classification' or 'regression')
    min_class_size : int
        Minimum samples per class for classification
        
    Returns
    -------
    Tuple[pd.Series, List[str]]
        (optimized_outcome_series, optimized_sample_ids)
    """
    import numpy as np
    if task_type != 'classification':
        return y_series, common_ids
    
    # Filter to common samples
    y_filtered = y_series.loc[common_ids]
    class_counts = y_filtered.value_counts()
    logger.info(f"Original class distribution: {class_counts.to_dict()}")
    
    # Identify classes with insufficient samples
    small_classes = class_counts[class_counts < min_class_size].index.tolist()
    if not small_classes:
        logger.info("All classes have sufficient samples for CV")
        return y_filtered, common_ids
    logger.info(f"Classes with <{min_class_size} samples: {small_classes}")
    
    y_optimized = y_filtered.copy()
    samples_to_keep = set(common_ids)
    
    # --- Generalized merging logic ---
    # Numeric classes: merge with nearest neighbor
    # String classes: merge into most common class
    try:
        # Try to treat as numeric
        y_numeric = pd.to_numeric(y_optimized, errors='coerce')
        is_numeric = not y_numeric.isna().all()
    except Exception:
        is_numeric = False
    
    if is_numeric:
        # For each small class, merge with nearest neighbor (by value)
        unique_classes = sorted(class_counts.index)
        for small_class in small_classes:
            # Find nearest neighbor class (by value, not in small_classes)
            try:
                small_val = float(small_class)
                other_classes = [float(c) for c in unique_classes if c not in small_classes]
                if not other_classes:
                    continue
                nearest = min(other_classes, key=lambda x: abs(x - small_val))
                logger.info(f"Merging rare class {small_class} into nearest neighbor {nearest}")
                y_optimized = y_optimized.replace(small_class, nearest)
            except Exception as e:
                logger.warning(f"Failed to merge numeric class {small_class}: {e}")
    else:
        # For string/categorical classes, merge into most common class
        most_common = class_counts.idxmax()
        for small_class in small_classes:
            logger.info(f"Merging rare class {small_class} into most common class {most_common}")
            y_optimized = y_optimized.replace(small_class, most_common)
    
    # Remove classes that are still too small after merging
    updated_counts = y_optimized.value_counts()
    still_small = updated_counts[updated_counts < min_class_size].index.tolist()
    if still_small:
        logger.info(f"Removing classes with <{min_class_size} samples: {still_small}")
        samples_to_remove = y_optimized[y_optimized.isin(still_small)].index
        samples_to_keep = samples_to_keep - set(samples_to_remove)
        y_optimized = y_optimized.drop(samples_to_remove)
    final_counts = y_optimized.value_counts()
    logger.info(f"Optimized class distribution: {final_counts.to_dict()}")
    logger.info(f"Samples retained: {len(y_optimized)}/{len(y_filtered)} ({len(y_optimized)/len(y_filtered)*100:.1f}%)")
    return y_optimized, sorted(list(samples_to_keep))

def load_dataset(ds_name: str, modalities: List[str], outcome_col: Optional[str] = None, 
                 task_type: str = 'classification', 
                 parallel: bool = True, 
                 use_cache: bool = True, 
                 min_class_size: int = 5) -> Tuple[Dict[str, pd.DataFrame], pd.Series, List[str]]:
    """
    Load dataset with comprehensive optimizations for all cancer types.
    Enhanced with parallel processing and caching for maximum performance.
    Now supports min_class_size as a parameter.
    """
    logger.info(f"=== LOADING DATASET: {ds_name.upper()} ===")
    logger.info(f"Modalities: {modalities}")
    
    # If outcome_col is None, get it from configuration
    if outcome_col is None:
        from config import DatasetConfig
        config = DatasetConfig.get_config(ds_name)
        if config:
            outcome_col = config['outcome_col']
            logger.info(f"Using outcome column from config: {outcome_col}")
        else:
            raise ValueError(f"No configuration found for dataset {ds_name} and no outcome_col specified")
    
    logger.info(f"Outcome column: {outcome_col}")
    logger.info(f"Task type: {task_type}")
    
    # Load clinical data first
    clinical_path = Path(f"data/clinical/{ds_name}.csv")
    if not clinical_path.exists():
        raise FileNotFoundError(f"Clinical data not found: {clinical_path}")
    
    logger.info(f"Loading clinical data from: {clinical_path}")
    
    # Try different parsing strategies for complex clinical files
    clinical_df = None
    parsing_strategies = [
        # Strategy 1: Standard parsing with error handling
        {'sep': '\t', 'index_col': 0, 'low_memory': False, 'on_bad_lines': 'skip'},
        # Strategy 2: Comma-separated with error handling
        {'sep': ',', 'index_col': 0, 'low_memory': False, 'on_bad_lines': 'skip'},
        # Strategy 3: Tab-separated with quoting and error handling
        {'sep': '\t', 'index_col': 0, 'low_memory': False, 'quoting': 1, 'on_bad_lines': 'skip'},
        # Strategy 4: Auto-detect separator with error handling
        {'sep': None, 'engine': 'python', 'index_col': 0, 'low_memory': False, 'on_bad_lines': 'skip'},
        # Strategy 5: Tab-separated without index column initially
        {'sep': '\t', 'low_memory': False, 'on_bad_lines': 'skip'},
        # Strategy 6: Force tab separation with minimal validation
        {'sep': '\t', 'low_memory': False, 'on_bad_lines': 'warn', 'error_bad_lines': False},
    ]
    
    for i, strategy in enumerate(parsing_strategies):
        try:
            logger.debug(f"Trying parsing strategy {i+1}...")
            clinical_df = pd.read_csv(clinical_path, **strategy)
            
            # If no index_col was set, use first column as index
            if 'index_col' not in strategy:
                clinical_df = clinical_df.set_index(clinical_df.columns[0])
            
            # Validate that we have reasonable data
            if clinical_df.shape[0] > 0 and clinical_df.shape[1] > 0:
                logger.info(f"Successfully parsed clinical data with strategy {i+1}")
                logger.info(f"Clinical data shape: {clinical_df.shape}")
                break
            else:
                logger.debug(f"Strategy {i+1} produced empty data, trying next strategy")
                clinical_df = None
                continue
            
        except Exception as e:
            logger.debug(f"Strategy {i+1} failed: {str(e)}")
            continue
    
    # Final fallback for severely malformed files
    if clinical_df is None:
        logger.warning("All standard parsing strategies failed, attempting manual repair...")
        try:
            # Try to manually fix the file by reading as text and reconstructing
            with open(clinical_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if len(lines) < 2:
                raise ValueError(f"Clinical file has insufficient data: {clinical_path}")
            
            # Get header line and try to parse it
            header_line = lines[0].strip()
            # Split by tab first, then by other separators if needed
            if '\t' in header_line:
                headers = header_line.split('\t')
            elif ',' in header_line:
                headers = header_line.split(',')
            else:
                headers = header_line.split()
            
            logger.info(f"Detected {len(headers)} columns in clinical data header")
            
            # Try to parse data lines with flexible approach
            data_rows = []
            for line_num, line in enumerate(lines[1:], 2):
                line = line.strip()
                if not line:
                    continue
                
                # Try different separators
                for sep in ['\t', ',', ' ']:
                    parts = line.split(sep)
                    if len(parts) >= len(headers) * 0.5:  # At least 50% of expected columns
                        # Pad or truncate to match header length
                        if len(parts) < len(headers):
                            parts.extend([''] * (len(headers) - len(parts)))
                        elif len(parts) > len(headers):
                            parts = parts[:len(headers)]
                        data_rows.append(parts)
                        break
                else:
                    # If no separator works well, skip this line
                    logger.debug(f"Skipping malformed line {line_num}")
                    continue
            
            if data_rows:
                # Create DataFrame from parsed data
                clinical_df = pd.DataFrame(data_rows, columns=headers)
                # Set first column as index if it looks like sample IDs
                if headers[0].lower() in ['sampleid', 'sample_id', 'id', 'patient_id']:
                    clinical_df = clinical_df.set_index(headers[0])
                
                logger.info(f"Manual repair successful: {clinical_df.shape}")
            else:
                raise ValueError("No valid data rows found after manual repair")
                
        except Exception as e:
            logger.error(f"Manual repair also failed: {str(e)}")
            raise ValueError(f"Failed to parse clinical data file: {clinical_path}")
    
    if clinical_df is None or clinical_df.empty:
        raise ValueError(f"Failed to parse clinical data file: {clinical_path}")
    
    logger.info(f"Clinical data columns: {list(clinical_df.columns)[:10]}...")  # Show first 10
    
    # Extract outcome data
    if outcome_col not in clinical_df.columns:
        available_cols = list(clinical_df.columns)
        logger.error(f"Outcome column '{outcome_col}' not found in clinical data")
        logger.error(f"Available columns: {available_cols[:10]}...")  # Show first 10
        raise KeyError(f"Outcome column '{outcome_col}' not found")
    
    y_series = clinical_df[outcome_col].copy()
    logger.info(f"Outcome data shape: {y_series.shape}")
    logger.info(f"Outcome value counts: {y_series.value_counts().to_dict()}")
    
    # FIX A: Drop rows with missing targets BEFORE any split
    # This ensures every sample in X has a target, preventing alignment issues
    if outcome_col is not None:
        # First check if data is numeric or can be converted to numeric
        outcome_data = clinical_df[outcome_col].copy()
        
        # For classification tasks, the outcome might be strings (classes)
        if task_type == 'classification':
            # For classification, just remove NaN/missing values
            mask = ~outcome_data.isna() & (outcome_data != '') & (outcome_data.notna())
        else:
            # For regression, ensure data is numeric
            try:
                # Try to convert to numeric, handling errors gracefully
                numeric_outcome = pd.to_numeric(outcome_data, errors='coerce')
                # Create mask for valid numeric values (not NaN and finite)
                mask = ~numeric_outcome.isna() & np.isfinite(numeric_outcome)
            except Exception as e:
                logger.warning(f"Could not apply numeric filtering to outcome data: {str(e)}")
                # Fallback to just removing missing values
                mask = ~outcome_data.isna() & (outcome_data != '') & (outcome_data.notna())
        
        # Apply mask to both outcome and clinical data to maintain alignment
        y_series = clinical_df.loc[mask, outcome_col]
        clinical_df = clinical_df.loc[mask]
        logger.info(f"Applied missing target filter: {len(y_series)} samples remain with valid targets")
    
    clinical_samples = set(y_series.index)
    logger.info(f"Samples with valid outcomes: {len(clinical_samples)}")
    
    # Load modality data with enhanced error handling and optional parallel processing
    modality_data = {}
    intersection_stats = {
        'initial_clinical_samples': len(clinical_samples),
        'modality_samples': {},
        'lost_samples': {},
        'recovery_attempts': {}
    }
    
    def load_single_modality(mod_name: str) -> Tuple[str, Optional[pd.DataFrame]]:
        """Helper function for parallel modality loading."""
        try:
            logger.info(f"Loading {mod_name} modality...")
            base_path = Path("data")
            modality_path = f"{ds_name}/{mod_name}.csv"
            mod_df = load_modality(base_path, modality_path, mod_name, use_cache=use_cache)
            
            if mod_df is None or mod_df.empty:
                logger.warning(f"Failed to load {mod_name} modality")
                return mod_name, None
                
            logger.info(f"{mod_name} modality loaded successfully: {mod_df.shape}")
            return mod_name, mod_df
            
        except Exception as e:
            logger.error(f"Error loading {mod_name} modality: {str(e)}")
            return mod_name, None
    
    # Load modalities either in parallel or sequentially
    if parallel and len(modalities) > 1:
        logger.info(f"Loading {len(modalities)} modalities in parallel...")
        with ThreadPoolExecutor(max_workers=min(len(modalities), 4)) as executor:
            # Submit all modality loading tasks
            future_to_modality = {
                executor.submit(load_single_modality, mod_name): mod_name 
                for mod_name in modalities
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_modality):
                mod_name, mod_df = future.result()
                if mod_df is not None:
                    modality_data[mod_name] = mod_df
                    mod_samples = set(mod_df.columns)
                    intersection_stats['modality_samples'][mod_name] = len(mod_samples)
                    logger.info(f"[OK] {mod_name}: {mod_df.shape[0]} features x {len(mod_samples)} samples")
    else:
        # Sequential loading for single modality or when parallel is disabled
        for mod_name in modalities:
            logger.info(f"\\n--- Loading {mod_name} modality ---")
            mod_name_result, mod_df = load_single_modality(mod_name)
            if mod_df is not None:
                modality_data[mod_name] = mod_df
                mod_samples = set(mod_df.columns)
                intersection_stats['modality_samples'][mod_name] = len(mod_samples)
    
    if not modality_data:
        raise ValueError("No modalities were successfully loaded")
    
    # Optimize sample intersection with enhanced recovery
    common_ids, filtered_modalities = optimize_sample_intersection(
        modality_data, y_series, ds_name
    )
    
    # Final validation and statistics
    logger.info(f"\\n=== FINAL DATASET STATISTICS ===")
    logger.info(f"Common samples found: {len(common_ids)}")
    logger.info(f"Modalities loaded: {list(filtered_modalities.keys())}")
    
    # Calculate retention rate
    initial_samples = intersection_stats['initial_clinical_samples']
    retention_rate = (len(common_ids) / initial_samples) * 100 if initial_samples > 0 else 0
    logger.info(f"Sample retention rate: {retention_rate:.1f}%")
    
    # Optimize class distribution for better CV performance (classification only)
    if task_type == 'classification':
        logger.info(f"\n=== CLASS DISTRIBUTION OPTIMIZATION ===")
        from preprocessing import _remap_labels
        y_series = _remap_labels(y_series, ds_name)
        y_optimized, optimized_ids = optimize_class_distribution(
            y_series, common_ids, task_type, min_class_size=min_class_size
        )
    else:
        # For regression, just filter to common_ids
        logger.info(f"\\n=== REGRESSION DATA FILTERING ===")
        logger.info(f"y_series type before filtering: {type(y_series)}")
        logger.info(f"y_series dtype before filtering: {y_series.dtype}")
        
        # CRITICAL DEBUG: Check if y_series is already corrupted
        if isinstance(y_series, str):
            logger.error(f"CRITICAL ERROR: y_series is already a string before filtering!")
            logger.error(f"String length: {len(y_series)}")
            logger.error(f"First 100 chars: {y_series[:100]}")
            raise ValueError("Outcome data was corrupted before filtering - already a string")
        
        y_optimized = y_series.loc[common_ids]
        optimized_ids = common_ids
        
        logger.info(f"y_optimized type after filtering: {type(y_optimized)}")
        logger.info(f"y_optimized dtype after filtering: {y_optimized.dtype}")
    
        # Update modalities to match optimized sample set
    if len(optimized_ids) != len(common_ids):
        logger.info(f"Updating modalities to match optimized sample set: {len(optimized_ids)} samples")
        for mod_name in filtered_modalities:
            available_samples = [col for col in optimized_ids if col in filtered_modalities[mod_name].columns]
            filtered_modalities[mod_name] = filtered_modalities[mod_name][available_samples]
        common_ids = optimized_ids

    # CRITICAL FIX: Filter outcome data to match common_ids exactly
    y_filtered = y_optimized.loc[common_ids]
    logger.info(f"Final outcome data filtered to {len(y_filtered)} samples matching common_ids")
    
    # CRITICAL DEBUG: Check if y_filtered is being corrupted
    if isinstance(y_filtered, str):
        logger.error(f"CRITICAL ERROR: y_filtered was converted to string during processing!")
        logger.error(f"String length: {len(y_filtered)}")
        logger.error(f"First 100 chars: {y_filtered[:100]}")
        raise ValueError("Outcome data was corrupted - converted to string instead of Series")
    
    if not isinstance(y_filtered, pd.Series):
        logger.warning(f"y_filtered is not a pandas Series, it's: {type(y_filtered)}")
        try:
            y_filtered = pd.Series(y_filtered, index=common_ids)
            logger.info(f"Successfully converted to pandas Series")
        except Exception as e:
            logger.error(f"Failed to convert to pandas Series: {str(e)}")
            raise ValueError("Could not convert outcome data to proper Series format")
    
    logger.info(f"Final outcome distribution: {y_filtered.value_counts().to_dict()}")
    
    # CRITICAL FIX: Convert string labels to numeric for classification
    if task_type == 'classification' and y_filtered.dtype == 'object':
        logger.info("Converting string class labels to numeric labels for classification")
        unique_classes = sorted(y_filtered.unique())
        label_mapping = {class_label: idx for idx, class_label in enumerate(unique_classes)}
        y_filtered = y_filtered.map(label_mapping)
        logger.info(f"Label mapping: {label_mapping}")
        logger.info(f"Converted to numeric labels: {y_filtered.value_counts().to_dict()}")
    
    # Validate data consistency
    if len(y_filtered) != len(common_ids):
        logger.error(f"CRITICAL ERROR: Outcome data length ({len(y_filtered)}) != common_ids length ({len(common_ids)})")
        raise ValueError(f"Data consistency error: outcome and sample ID lengths don't match")
    
    # Validate for task type
    if task_type == 'classification':
        unique_classes = y_filtered.nunique()
        min_class_size = y_filtered.value_counts().min()
        logger.info(f"Classification validation: {unique_classes} classes, min class size: {min_class_size}")
        
        if unique_classes < 2:
            logger.warning(f"Only {unique_classes} unique class(es) found - may cause issues")
        elif min_class_size >= 5:
            logger.info(f"All classes have >=5 samples - excellent for CV!")
        elif min_class_size >= 2:
            logger.info(f"All classes have >=2 samples - good for CV")
        else:
            logger.warning(f"Minimum class size is {min_class_size} - may cause CV issues")
    
    elif task_type == 'regression':
        # CRITICAL FIX: Ensure y_filtered is a proper pandas Series, not concatenated string
        if isinstance(y_filtered, str):
            logger.error(f"Outcome data was incorrectly concatenated into a string. This indicates a data processing error.")
            raise ValueError("Outcome data was corrupted during processing - received concatenated string instead of Series")
        
        # Ensure we have a proper pandas Series
        if not isinstance(y_filtered, pd.Series):
            try:
                y_filtered = pd.Series(y_filtered, index=common_ids)
            except Exception as e:
                logger.error(f"Could not convert outcome data to pandas Series: {str(e)}")
                raise ValueError("Invalid outcome data format for regression")
        
        # Convert outcome data to numeric, handling non-numeric values
        y_numeric = pd.to_numeric(y_filtered, errors='coerce')
        
        # Check for non-numeric values that were converted to NaN
        non_numeric_count = y_numeric.isna().sum() - y_filtered.isna().sum()
        if non_numeric_count > 0:
            logger.info(f"Found {non_numeric_count} pipe-separated values in regression outcome column '{outcome_col}'")
            
            # Get sample of non-numeric values for logging (safely)
            non_numeric_mask = y_numeric.isna() & y_filtered.notna()
            if non_numeric_mask.any():
                sample_non_numeric = y_filtered[non_numeric_mask].head().tolist()
                # Truncate very long strings for logging
                sample_non_numeric = [str(val)[:50] + "..." if len(str(val)) > 50 else str(val) for val in sample_non_numeric]
                logger.info(f"Sample pipe-separated values: {sample_non_numeric}")
            
            # Enhanced extraction function with better error handling
            def extract_max_numeric(value):
                if pd.isna(value):
                    return np.nan
                if isinstance(value, (int, float)):
                    # Check if the numeric value is valid (not NaN or infinite)
                    if np.isnan(value) or np.isinf(value):
                        return np.nan
                    return float(value)
                if isinstance(value, str):
                    # Handle very long concatenated strings by taking only first part
                    if len(value) > 100:
                        logger.warning(f"Detected very long string value (length: {len(value)}), taking first 100 characters")
                        value = value[:100]
                    
                    # Clean the string first
                    value = value.strip()
                    if not value:  # Empty string after stripping
                        return np.nan
                    
                    # Split by common separators and collect all numeric values
                    numeric_values = []
                    for sep in ['|', ',', ';', ' ', '\t']:
                        if sep in value:
                            parts = value.split(sep)
                            for part in parts:
                                part = part.strip()
                                if not part:  # Skip empty parts
                                    continue
                                try:
                                    numeric_val = float(part)
                                    # Validate the numeric value
                                    if not np.isnan(numeric_val) and not np.isinf(numeric_val):
                                        numeric_values.append(numeric_val)
                                except (ValueError, TypeError):
                                    continue
                            break  # Use the first separator that works
                    
                    # If we found numeric values, return the maximum
                    if numeric_values:
                        max_val = max(numeric_values)
                        logger.debug(f"Extracted max value {max_val} from '{value[:30]}...'")
                        return max_val
                    
                    # Try to convert the whole string as fallback
                    try:
                        numeric_val = float(value)
                        if not np.isnan(numeric_val) and not np.isinf(numeric_val):
                            return numeric_val
                        else:
                            return np.nan
                    except (ValueError, TypeError):
                        logger.debug(f"Could not extract numeric value from '{value[:30]}...'")
                        return np.nan
                
                # For any other type, try to convert to float
                try:
                    numeric_val = float(value)
                    if not np.isnan(numeric_val) and not np.isinf(numeric_val):
                        return numeric_val
                    else:
                        return np.nan
                except (ValueError, TypeError):
                    return np.nan
            
            logger.info("Attempting to extract maximum numeric values from pipe-separated entries...")
            y_extracted = y_filtered.apply(extract_max_numeric)
            
            # Count successful extractions
            extracted_count = (~y_extracted.isna()).sum() - (~y_numeric.isna()).sum()
            
            if extracted_count > 0:
                logger.info(f"Successfully extracted {extracted_count} additional numeric values")
                y_numeric = y_extracted
            else:
                logger.warning("No additional numeric values could be extracted from pipe-separated entries")
            
            # Additional validation: ensure no NaN values remain in the extracted data
            remaining_nan_count = y_numeric.isna().sum()
            if remaining_nan_count > 0:
                logger.warning(f"Still have {remaining_nan_count} NaN values after extraction")
                
                # For AML dataset specifically, try more aggressive extraction
                if ds_name.lower() == 'aml':
                    logger.info("Applying AML-specific aggressive numeric extraction...")
                    
                    def aml_aggressive_extract(value):
                        if pd.isna(value):
                            return np.nan
                        if isinstance(value, (int, float)):
                            return float(value) if not (np.isnan(value) or np.isinf(value)) else np.nan
                        
                        # Convert to string and clean
                        str_val = str(value).strip()
                        if not str_val or str_val.lower() in ['nan', 'null', 'none', '']:
                            return np.nan
                        
                        # Extract all numbers from the string using regex
                        import re
                        numbers = re.findall(r'-?\d+\.?\d*', str_val)
                        valid_numbers = []
                        
                        for num_str in numbers:
                            try:
                                num = float(num_str)
                                if not (np.isnan(num) or np.isinf(num)):
                                    valid_numbers.append(num)
                            except (ValueError, TypeError):
                                continue
                        
                        if valid_numbers:
                            # For AML blast cell percentage, return the maximum value
                            return max(valid_numbers)
                        else:
                            return np.nan
                    
                    y_aml_extracted = y_filtered.apply(aml_aggressive_extract)
                    aml_extracted_count = (~y_aml_extracted.isna()).sum() - (~y_numeric.isna()).sum()
                    
                    if aml_extracted_count > 0:
                        logger.info(f"AML aggressive extraction recovered {aml_extracted_count} additional values")
                        y_numeric = y_aml_extracted
            
            # Remove samples with non-numeric outcomes
            valid_numeric_mask = y_numeric.notna()
            if not valid_numeric_mask.all():
                removed_count = (~valid_numeric_mask).sum()
                logger.warning(f"Removing {removed_count} samples with non-numeric outcomes")
                
                # Update all data structures to match valid numeric samples
                valid_ids = y_numeric[valid_numeric_mask].index.tolist()
                y_filtered = y_numeric[valid_numeric_mask]
                
                # Update modalities to match valid samples
                for mod_name in filtered_modalities:
                    available_samples = [col for col in valid_ids if col in filtered_modalities[mod_name].columns]
                    filtered_modalities[mod_name] = filtered_modalities[mod_name][available_samples]
                
                # Update common_ids
                common_ids = valid_ids
                logger.info(f"Updated dataset to {len(common_ids)} samples with valid numeric outcomes")
            else:
                # Even if no samples were removed, we still need to update y_filtered with the numeric version
                y_filtered = y_numeric
                logger.info(f"Updated y_filtered to numeric version (no samples removed)")
        else:
            # No non-numeric values found, but still ensure y_filtered is numeric
            y_filtered = y_numeric
            logger.info(f"All values were already numeric, updated y_filtered to ensure proper dtype")
        
        # CRITICAL: Final validation to ensure no NaN values remain
        final_nan_count = y_filtered.isna().sum()
        if final_nan_count > 0:
            logger.error(f"CRITICAL: {final_nan_count} NaN values still remain in outcome data after all processing!")
            logger.error("This will cause 'Input contains NaN' errors in model training")
            
            # Emergency fallback: replace remaining NaN values with median
            if len(y_filtered.dropna()) > 0:
                median_value = y_filtered.median()
                logger.warning(f"Emergency fallback: replacing {final_nan_count} NaN values with median ({median_value})")
                y_filtered = y_filtered.fillna(median_value)
            else:
                logger.error("Cannot compute median - all values are NaN!")
                raise ValueError("All outcome values are NaN - cannot proceed with regression task")
        
        # Final validation: ensure all values are finite (only for numeric data)
        try:
            # Only check for infinite values if data is numeric
            if pd.api.types.is_numeric_dtype(y_filtered):
                infinite_count = np.isinf(y_filtered).sum()
                if infinite_count > 0:
                    logger.warning(f"Found {infinite_count} infinite values in outcome data, replacing with median")
                    finite_values = y_filtered[np.isfinite(y_filtered)]
                    if len(finite_values) > 0:
                        median_value = finite_values.median()
                        y_filtered = y_filtered.replace([np.inf, -np.inf], median_value)
                    else:
                        logger.warning("All values are infinite, cannot replace with median")
        except Exception as e:
            logger.debug(f"Could not check for infinite values (data may be non-numeric): {str(e)}")
        
        # Dataset-specific validation for regression targets
        if task_type == 'regression':
            if ds_name.lower() == 'aml' and 'blast' in outcome_col.lower():
                # AML blast percentage should be 0-100%
                min_val = y_filtered.min()
                max_val = y_filtered.max()
                if min_val < 0:
                    logger.error(f"CRITICAL: AML blast % contains negative values (min={min_val:.3f})")
                    logger.error("This indicates you've selected the wrong column (possibly z-score normalized version)")
                    logger.error(f"Expected column: 'lab_procedure_bone_marrow_blast_cell_outcome_percent_value'")
                    logger.error(f"Current column: '{outcome_col}'")
                    raise ValueError(f"AML blast % should be non-negative, got min={min_val:.3f}")
                if max_val > 100:
                    logger.warning(f"AML blast % exceeds 100% (max={max_val:.3f}), this may indicate data issues")
                logger.info(f"AML blast % validation passed: range [{min_val:.1f}, {max_val:.1f}]%")
            
            elif ds_name.lower() == 'sarcoma' and 'length' in outcome_col.lower():
                # Sarcoma tumor length should be positive
                min_val = y_filtered.min()
                if min_val < 0:
                    logger.error(f"CRITICAL: Sarcoma tumor length contains negative values (min={min_val:.3f})")
                    logger.error("This indicates you've selected the wrong column (possibly z-score normalized version)")
                    raise ValueError(f"Sarcoma tumor length should be non-negative, got min={min_val:.3f}")
                logger.info(f"Sarcoma tumor length validation passed: range [{min_val:.1f}, {y_filtered.max():.1f}] cm")
        
        logger.info(f"Final outcome data validation: {len(y_filtered)} samples, all numeric and finite")
        
        # Now calculate statistics on the clean numeric data
        if len(y_filtered) > 0:
            try:
                y_stats = y_filtered.describe()
                # Check if we have valid statistics
                if 'mean' in y_stats and 'std' in y_stats and not pd.isna(y_stats['mean']):
                    logger.info(f"Regression validation - Target stats: mean={y_stats['mean']:.3f}, std={y_stats['std']:.3f}")
                    logger.info(f"Target range: [{y_stats['min']:.3f}, {y_stats['max']:.3f}]")
                else:
                    # Fallback to basic statistics
                    mean_val = y_filtered.mean() if not y_filtered.empty else 0
                    std_val = y_filtered.std() if not y_filtered.empty else 0
                    min_val = y_filtered.min() if not y_filtered.empty else 0
                    max_val = y_filtered.max() if not y_filtered.empty else 0
                    logger.info(f"Regression validation - Target stats: mean={mean_val:.3f}, std={std_val:.3f}")
                    logger.info(f"Target range: [{min_val:.3f}, {max_val:.3f}]")
            except Exception as e:
                logger.warning(f"Could not calculate regression statistics: {str(e)}")
                logger.info(f"Regression validation - {len(y_filtered)} samples loaded")
        else:
            logger.error("No valid numeric samples remaining for regression task")
            raise ValueError("No valid numeric samples found for regression outcome")
    
    logger.info(f"=== DATASET {ds_name.upper()} LOADED SUCCESSFULLY ===")
    
    return filtered_modalities, y_filtered, common_ids

def save_results(results_df: pd.DataFrame, output_dir: Union[str, Path], filename: str) -> None:
    """
    Save results DataFrame to file.
    
    Parameters
    ----------
    results_df      DataFrame with results
    output_dir      Output directory path
    filename        Output filename
    
    Returns
    -------
    None
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Full output path
    output_path = Path(output_dir) / filename
    
    # Check if file exists
    file_exists = output_path.exists()
    
    # Save to CSV
    results_df.to_csv(
        output_path,
        mode='a' if file_exists else 'w',
        header=not file_exists,
        index=False
    ) 

def load_multimodal_data(dataset_name: str, task_type: str) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[np.ndarray]]:
    """
    Load multimodal data for a given dataset and task type.
    
    Args:
        dataset_name: Name of the dataset
        task_type: 'classification' or 'regression'
    
    Returns:
        Tuple of (data_dict, target_array) or (None, None) if loading fails
    """
    try:
        # Get dataset configuration
        config = DatasetConfig.get_config(dataset_name)
        if not config:
            logger.error(f"No configuration found for dataset: {dataset_name}")
            return None, None
        
        # Load dataset using existing load_dataset function
        modality_data, outcome_series, common_ids = load_dataset(
            ds_name=dataset_name,
            modalities=config['modalities'],
            outcome_col=config['outcome_col'],
            task_type=task_type
        )
        
        # Convert to numpy arrays
        data_dict = {}
        for modality, df in modality_data.items():
            data_dict[modality] = df.values.T  # Transpose to samples x features
        
        target = outcome_series.values
        
        return data_dict, target
        
    except Exception as e:
        logger.error(f"Failed to load multimodal data for {dataset_name}: {str(e)}")
        return None, None

def load_and_preprocess_data(dataset_name: str, task_type: str, 
                           apply_advanced_filtering: bool = True,
                           apply_biomedical_preprocessing: bool = True) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    DEPRECATED: Use load_and_preprocess_data_enhanced instead.
    
    This function is kept for backward compatibility only.
    For new code, use load_and_preprocess_data_enhanced which provides
    the 4-phase enhanced preprocessing pipeline.
    
    Args:
        dataset_name: Name of the dataset
        task_type: 'classification' or 'regression'
        apply_advanced_filtering: Whether to apply advanced feature filtering
        apply_biomedical_preprocessing: Whether to apply biomedical preprocessing
    
    Returns:
        Tuple of (data_dict, target_array)
    """
    import warnings
    warnings.warn(
        "load_and_preprocess_data is deprecated. Use load_and_preprocess_data_enhanced instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Redirect to enhanced version
    processed_modalities, y_aligned, sample_ids, report = load_and_preprocess_data_enhanced(
        dataset_name, task_type, enable_all_improvements=apply_biomedical_preprocessing
    )
    
    return processed_modalities, y_aligned

def load_and_preprocess_data_enhanced(
    dataset_name, 
    task_type='regression',
    enable_all_improvements=True,
    apply_priority_fixes=True,
    enable_missing_imputation=True,
    enable_target_analysis=True,
    enable_mad_recalibration=True,
    enable_target_aware_selection=True
):
    """
    Enhanced data loading and preprocessing with all improvements enabled.
    
    Args:
        dataset_name: Name of the dataset to load
        task_type: 'regression' or 'classification'
        enable_all_improvements: Enable all preprocessing improvements
        apply_priority_fixes: Apply priority data quality fixes
        enable_missing_imputation: Enable missing modality imputation
        enable_target_analysis: Enable target distribution analysis
        enable_mad_recalibration: Enable MAD threshold recalibration
        enable_target_aware_selection: Enable target-aware feature selection
    
    Returns:
        tuple: (processed_modalities, y_aligned, sample_ids, report)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading and preprocessing {dataset_name} dataset for {task_type}")
    
    # Load the dataset with sample IDs using configuration-based outcome columns
    modalities, y_series, common_ids = load_dataset(
        dataset_name.lower(), 
        modalities=['exp', 'methy', 'mirna'], 
        outcome_col=None,  # Let load_dataset determine from config
        task_type=task_type
    )
    
    logger.info(f"Successfully loaded {dataset_name} with {len(common_ids)} samples")
    
    # Convert modalities to the expected format for the preprocessing pipeline
    # The pipeline expects Dict[str, Tuple[np.ndarray, List[str]]]
    modality_data_dict = {}
    for modality_name, modality_df in modalities.items():
        # Convert DataFrame to numpy array (transpose to get samples x features)
        X = modality_df.T.values  # modality_df is features x samples
        modality_data_dict[modality_name] = (X, common_ids)
    
    # Apply 4-phase enhanced preprocessing pipeline
    if enable_all_improvements:
        try:
            from enhanced_pipeline_integration import run_enhanced_preprocessing_pipeline
            
            # Determine optimal fusion method based on task type
            fusion_method = "snf" if task_type == "classification" else "weighted_concat"
            
            processed_modalities, y_aligned, pipeline_metadata = run_enhanced_preprocessing_pipeline(
                modality_data_dict=modality_data_dict,
                y=y_series.values,
                fusion_method=fusion_method,
                task_type=task_type,
                dataset_name=dataset_name,
                enable_early_quality_check=True,
                enable_fusion_aware_order=True,
                enable_centralized_missing_data=True,
                enable_coordinated_validation=True
            )
            
            logger.info(f"4-Phase pipeline completed for {dataset_name}")
            logger.info(f"Quality Score: {pipeline_metadata.get('quality_score', 'N/A')}")
            
        except Exception as e:
            logger.warning(f"4-Phase pipeline failed for {dataset_name}: {e}")
            logger.info("Falling back to robust biomedical preprocessing")
            
            # Fallback to robust biomedical preprocessing
            from preprocessing import robust_biomedical_preprocessing_pipeline
            
            processed_modalities = {}
            for modality_name, (X, sample_ids) in modality_data_dict.items():
                # Determine modality type
                if 'exp' in modality_name.lower():
                    modality_type = 'gene_expression'
                elif 'mirna' in modality_name.lower():
                    modality_type = 'mirna'
                elif 'methy' in modality_name.lower():
                    modality_type = 'methylation'
                else:
                    modality_type = 'unknown'
                
                # Apply robust preprocessing
                X_processed, transformers, report = robust_biomedical_preprocessing_pipeline(
                    X, modality_type=modality_type
                )
                processed_modalities[modality_name] = X_processed
            
            # Align targets
            n_samples = list(processed_modalities.values())[0].shape[0]
            y_aligned = y_series.values[:n_samples] if len(y_series.values) >= n_samples else y_series.values
        
        # Create report
        report = {
            'preprocessing_method': 'enhanced_comprehensive',
            'n_samples': len(y_aligned),
            'n_modalities': len(processed_modalities),
            'total_features': sum(X.shape[1] for X in processed_modalities.values())
        }
        
        # Return sample_ids as common_ids
        sample_ids = common_ids[:len(y_aligned)]
    else:
        # Basic preprocessing without improvements
        processed_modalities, y_aligned, sample_ids, report = basic_preprocessing_pipeline(
            modalities, y_series, common_ids
        )
    
    logger.info(f"Preprocessing completed successfully for {dataset_name}")
    return processed_modalities, y_aligned, sample_ids, report







# ============================================================================
# TUNER_HALVING.PY COMPATIBILITY WRAPPER
# ============================================================================

def load_dataset_for_tuner(dataset_name, task=None):
    """
    Load dataset with FULL preprocessing pipeline for tuner_halving.py compatibility.
    
    This ensures hyperparameters are optimized on the same preprocessed data
    that the main pipeline uses, providing consistent and meaningful optimization.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g., 'AML', 'Breast', etc.)
    task : str
        Task type ('reg' or 'clf')
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Fully preprocessed features (X) and targets (y) as numpy arrays
    """
    from config import DatasetConfig
    import logging
    
    logger = logging.getLogger(__name__)
    logger.info(f"Loading dataset {dataset_name} for tuning with FULL preprocessing pipeline (task: {task})")
    
    # Determine task type
    if task is None:
        # Get configuration for the dataset
        config = DatasetConfig.get_config(dataset_name.lower())
        if not config:
            raise ValueError(f"No configuration found for dataset: {dataset_name}")
        
        # Infer from outcome column name
        outcome_col = config.get('outcome_col', '')
        if 'blast' in outcome_col.lower() or 'length' in outcome_col.lower():
            task_type = 'regression'
        else:
            task_type = 'classification'
    else:
        task_type = 'regression' if task == 'reg' else 'classification'
    
    # Use the SAME data loading as main pipeline, but simplified for single modality concatenation
    logger.info(f"Loading data using main pipeline approach for {dataset_name}")
    
    # Get configuration for the dataset
    config = DatasetConfig.get_config(dataset_name.lower())
    if not config:
        raise ValueError(f"No configuration found for dataset: {dataset_name}")
    
    # Map modality names to short names (same as main pipeline)
    modality_mapping = {
        "Gene Expression": "exp",
        "miRNA": "mirna", 
        "Methylation": "methy"
    }
    
    modality_short_names = []
    for full_name in config['modalities'].keys():
        short_name = modality_mapping.get(full_name, full_name.lower())
        modality_short_names.append(short_name)
    
    # Load dataset using the main function (includes preprocessing)
    modalities_data, y_series, common_ids = load_dataset(
        ds_name=dataset_name.lower(),
        modalities=modality_short_names,
        outcome_col=config['outcome_col'],
        task_type=task_type,
        parallel=True,
        use_cache=True
    )
    
    if not modalities_data or y_series is None:
        raise ValueError(f"Failed to load data for {dataset_name}")
    
    # Ensure all modalities have the same samples (aligned by common_ids)
    logger.info(f"Loaded {len(modalities_data)} modalities with {len(common_ids)} common samples")
    
    # Concatenate all modalities into single feature matrix
    X_parts = []
    modality_info = []
    
    for modality_name, modality_df in modalities_data.items():
        # Transpose to get samples x features (modality_df is features x samples)
        X_modality = modality_df.T.values
        X_parts.append(X_modality)
        modality_info.append(f"{modality_name}: {X_modality.shape[1]} features")
        logger.info(f"  {modality_name} shape: {X_modality.shape}")
    
    # Concatenate horizontally (samples x all_features)
    X = np.concatenate(X_parts, axis=1)
    y = y_series.values
    
    logger.info(f"Dataset {dataset_name} loaded with FULL preprocessing:")
    logger.info(f"  Final X shape: {X.shape}")
    logger.info(f"  Final y shape: {y.shape}")
    logger.info(f"  Modalities: {', '.join(modality_info)}")
    logger.info(f"  Total features after preprocessing: {X.shape[1]}")
    
    # Final validation
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        logger.warning("Found NaN/Inf values in preprocessed data, cleaning...")
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        logger.warning("Found NaN/Inf values in target, cleaning...")
        y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return X, y

