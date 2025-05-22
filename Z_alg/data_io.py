#!/usr/bin/env python3
"""
Input/Output module for reading and writing data files.
"""

import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Union, Tuple, Any
import logging

# Local imports
from Z_alg.config import DatasetConfig, MAX_VARIABLE_FEATURES
from Z_alg.preprocessing import _keep_top_variable_rows, fix_tcga_id_slicing, custom_parse_outcome

logger = logging.getLogger(__name__)

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

def load_modality(base_path: Union[str, Path], 
                 modality_path: Union[str, Path], 
                 modality_name: str,
                 k_features: int = MAX_VARIABLE_FEATURES) -> Optional[pd.DataFrame]:
    """
    Load a single modality file, apply preprocessing, and return DataFrame.
    
    Parameters
    ----------
    base_path       Base path for the dataset
    modality_path   Path to modality file (relative to base_path)
    modality_name   Name of the modality
    k_features      Maximum number of features to keep
    
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
    
    df = None
    for path in paths_to_try:
        if path.exists():
            df = try_read_file(path)
            if df is not None:
                break
    
    if df is None or df.empty:
        logger.warning(f"Warning: Could not load modality {modality_name}")
        return None
    
    # Brief debug info
    logger.debug(f"Loaded {modality_name} data: shape={df.shape}")
    
    # Check if index/row names look like sample IDs (TCGA-XX-XXXX format)
    # and columns look like features (gene symbols, miRNA names, cpg sites)
    # We want features in rows, samples in columns
    
    # If first column name contains TCGA or sample, transpose (samples in columns)
    sample_pattern = ['TCGA', 'sample', 'SAMPLE_']
    
    # Check if samples are in rows (need to transpose)
    if any(x in str(df.index[0]) for x in sample_pattern):
        logger.debug(f"Transposing {modality_name}: samples detected in rows")
        df = df.T
    
    # Check if samples are in columns (already correct orientation)
    elif any(x in str(df.columns[0]) for x in sample_pattern):
        logger.debug(f"Correct orientation for {modality_name}: samples in columns")
    
    # More thorough orientation check based on sample ID patterns in columns
    tcga_pattern = sum(1 for col in df.columns if isinstance(col, str) 
                       and any(pattern in col for pattern in sample_pattern))
    
    # If no TCGA pattern in columns but found in rows, transpose
    if tcga_pattern < 5:
        tcga_rows = sum(1 for idx in df.index if isinstance(idx, str) 
                        and any(pattern in idx for pattern in sample_pattern))
        if tcga_rows > 5:
            logger.debug(f"Transposing {modality_name}: {tcga_rows} sample IDs detected in rows")
            df = df.T
    
    # Ensure index/column names are strings
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    
    # Variance filtering if needed
    if df.shape[0] > k_features:
        logger.info(f"Applying variance filtering to {modality_name}, keeping top {k_features} features")
        df = _keep_top_variable_rows(df, k=k_features)
    
    return df

def load_outcome(base_path: Union[str, Path], 
                outcome_file: Union[str, Path], 
                outcome_col: str,
                id_col: str,
                outcome_type: str = 'os') -> Tuple[Optional[pd.Series], Optional[pd.DataFrame]]:
    """
    Load outcome data from file.
    
    Parameters
    ----------
    base_path       Base path for the dataset
    outcome_file    Path to outcome file (relative to base_path)
    outcome_col     Column name for the outcome variable
    id_col          Column name for sample IDs
    outcome_type    Type of outcome (os, pfs, response, etc.)
    
    Returns
    -------
    Tuple of (outcome Series, full outcomes DataFrame) or (None, None)
    """
    # Combine base path and outcome file path
    try:
        # Try standard path joining
        full_path = Path(base_path) / outcome_file
        
        # For Windows, also try explicit string path with forward slashes
        if not full_path.exists():
            base_str = str(base_path).replace('\\', '/')
            outcome_str = str(outcome_file).replace('\\', '/')
            if outcome_str.startswith('/'):
                outcome_str = outcome_str[1:]
            alt_path = Path(f"{base_str}/{outcome_str}")
            if alt_path.exists():
                logger.info(f"Using alternate path for outcome: {alt_path}")
                full_path = alt_path
    except Exception as e:
        logger.info(f"Error constructing outcome path: {str(e)}, trying direct path")
        full_path = Path(outcome_file)  # Last resort - direct path
    
    # Try to read the file
    df = try_read_file(full_path)
    if df is None:
        return None, None
    
    # Skip if empty
    if df.empty:
        logger.warning(f"Warning: Empty outcome DataFrame")
        return None, None
    
    # Check if outcome column exists
    if outcome_col not in df.columns:
        logger.warning(f"Warning: Outcome column '{outcome_col}' not found in {outcome_file}")
        return None, None
        
    # Check if ID column exists
    if id_col not in df.columns and df.index.name != id_col:
        logger.warning(f"Warning: ID column '{id_col}' not found in {outcome_file}")
        return None, None
    
    # If ID column is not the index, set it as index
    if id_col in df.columns and df.index.name != id_col:
        df = df.set_index(id_col)
    
    # Extract outcome series and apply custom parsing
    outcome_series = df[outcome_col]
    parsed_outcome = custom_parse_outcome(outcome_series, outcome_type)
    
    # Handle NaN values
    if parsed_outcome.isna().all():
        logger.warning(f"Warning: All outcome values are NaN in {outcome_file}")
        return None, None
    
    # Ensure that index (sample IDs) are strings
    df.index = df.index.astype(str)
    parsed_outcome.index = parsed_outcome.index.astype(str)
    
    # Debug information
    logger.info(f"Loaded outcomes for {len(parsed_outcome)} samples, sample IDs: {parsed_outcome.index[:5]}...")
    
    # Return both the parsed outcome series and the full DataFrame
    return parsed_outcome, df

def load_dataset(ds_conf: Dict[str, Any]) -> Optional[Tuple[Dict[str, pd.DataFrame], List[str], np.ndarray]]:
    """
    Load a complete dataset with all modalities and outcomes.
    
    Parameters
    ----------
    ds_conf         Dataset configuration dictionary
    
    Returns
    -------
    Tuple of (modalities dict, common_ids list, y array) or None
    """
    # Extract configuration values
    ds_name = ds_conf["name"]
    base_path = ds_conf["base_path"]
    modality_paths = ds_conf["modalities"]
    outcome_file = ds_conf["outcome_file"]
    outcome_col = ds_conf["outcome_col"]
    id_col = ds_conf["id_col"]
    outcome_type = ds_conf.get("outcome_type", "os")
    
    # Load outcome data
    y_series, outcome_df = load_outcome(base_path, outcome_file, outcome_col, id_col, outcome_type)
    if y_series is None:
        logger.error(f"Error: Failed to load outcome data for {ds_name}")
        return None
    
    # Load all modalities
    modalities = {}
    for mod_name, mod_path in modality_paths.items():
        df = load_modality(base_path, mod_path, mod_name)
        if df is not None and not df.empty:
            # Standardize sample IDs if needed (e.g., TCGA IDs)
            if ds_conf.get("fix_tcga_ids", False):
                # Get original column names
                orig_cols = df.columns.tolist()
                # Fix IDs
                fixed_cols = fix_tcga_id_slicing(orig_cols)
                # Create mapping from original to fixed
                col_map = {orig: fixed for orig, fixed in zip(orig_cols, fixed_cols)}
                # Rename columns
                df = df.rename(columns=col_map)
            
            modalities[mod_name] = df
        else:
            logger.warning(f"Warning: Failed to load modality {mod_name} for {ds_name}")
    
    # Check if we have at least one modality
    if not modalities:
        logger.error(f"Error: No valid modalities found for {ds_name}")
        return None
    
    # For each modality, normalize column names to strings
    for mod_name, df in modalities.items():
        modalities[mod_name].columns = modalities[mod_name].columns.astype(str)
    
    # Ensure outcome index is string type
    outcome_df.index = outcome_df.index.astype(str)
    y_series.index = y_series.index.astype(str)
    
    # Find common sample IDs across all modalities and the outcome data
    common_ids = set(outcome_df.index)
    logger.info(f"Starting with {len(common_ids)} samples from outcome data")
    
    for mod_name, df in modalities.items():
        mod_samples = set(df.columns)
        common_before = len(common_ids)
        common_ids = common_ids.intersection(mod_samples)
        logger.info(f"After intersecting with {mod_name} ({len(mod_samples)} samples): {len(common_ids)} common samples")
        
        if common_before > 0 and len(common_ids) == 0:
            # Print more details to diagnose the issue
            logger.info(f"  Sample IDs in outcome data: {list(outcome_df.index)[:5]}")
            logger.info(f"  Sample IDs in {mod_name}: {list(df.columns)[:5]}")
    
    # Check if we have enough common samples
    if len(common_ids) < 10:
        logger.warning(f"Warning: Only {len(common_ids)} common samples found for {ds_name}, may be insufficient")
    
    # Convert to sorted list
    common_ids = sorted(list(common_ids))
    
    # Extract aligned outcome values
    try:
        y_aligned = y_series.loc[common_ids].values
        
        # Print diagnostic information
        logger.info(f"Successfully aligned {len(y_aligned)} samples for {ds_name}")
        if len(y_aligned) > 0:
            logger.info(f"First few outcomes: {y_aligned[:5]}")
        
        # Return the loaded dataset
        return modalities, common_ids, y_aligned
    except Exception as e:
        logger.error(f"Error aligning outcome values: {str(e)}")
        return None

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