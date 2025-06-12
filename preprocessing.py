#!/usr/bin/env python3
"""
Preprocessing module for data preparation and cleaning functions.
"""

import numpy as np
import pandas as pd
import random
import warnings
from typing import Dict, List, Optional, Union, Any, Tuple
import logging

# Suppress sklearn deprecation warning about force_all_finite -> ensure_all_finite
warnings.filterwarnings("ignore", message=".*force_all_finite.*was renamed to.*ensure_all_finite.*", category=FutureWarning)

from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from scipy import stats

# Local imports
from config import MAX_VARIABLE_FEATURES, PREPROCESSING_CONFIG

logger = logging.getLogger(__name__)

def _keep_top_variable_rows(df: pd.DataFrame,
                          k: int = MAX_VARIABLE_FEATURES) -> pd.DataFrame:
    """
    Keep at most *k* rows with the highest Median Absolute Deviation (MAD) across samples.

    The omics matrices in this project are all shaped (features × samples),
    so we compute row-wise MAD. MAD is more robust to outliers than variance.
    Sparse frames are handled efficiently with toarray() fallback if needed.

    Parameters
    ----------
    df : pd.DataFrame (features × samples)
    k  : int – number of rows to keep (default = 10,000, increased from 5,000)

    Returns
    -------
    pd.DataFrame containing ≤ k rows with highest MAD
    """
    # Skip if the data frame is already small enough
    if df.shape[0] <= k:
        return df
    
    # Compute row-wise Median Absolute Deviation (MAD) with improved handling
    try:
        if hasattr(df, 'sparse') and df.sparse.density < 0.3:
            # For sparse DataFrames, convert to dense for MAD computation
            data_array = df.sparse.to_dense().values
        else:
            data_array = df.values
        
        # Calculate MAD for each feature (row) with better NaN handling
        mad_values = []
        for i in range(data_array.shape[0]):
            row_data = data_array[i, :]
            # Remove NaN values for MAD calculation
            row_data_clean = row_data[~np.isnan(row_data)]
            if len(row_data_clean) >= 3:  # Require at least 3 valid values
                # Calculate median
                median_val = np.median(row_data_clean)
                # Calculate absolute deviations from median
                abs_deviations = np.abs(row_data_clean - median_val)
                # Calculate MAD (median of absolute deviations)
                mad_val = np.median(abs_deviations)
                # Add small bonus for features with more valid values
                completeness_bonus = len(row_data_clean) / len(row_data) * 0.1
                mad_val += completeness_bonus
            elif len(row_data_clean) > 0:
                # For features with few valid values, use standard deviation as fallback
                mad_val = np.std(row_data_clean) * 0.5  # Reduced weight for incomplete features
            else:
                mad_val = 0.0
            mad_values.append(mad_val)
        
        # Convert to pandas Series with original index
        mad_series = pd.Series(mad_values, index=df.index)
        
    except Exception as e:
        # Fallback to variance if MAD computation fails
        logger.warning(f"Warning: MAD computation failed ({str(e)}), falling back to variance")
        try:
            if hasattr(df, 'sparse') and df.sparse.density < 0.3:
                mad_series = df.sparse.to_dense().var(axis=1, skipna=True)
            else:
                mad_series = df.var(axis=1, skipna=True)
        except Exception as e2:
            logger.warning(f"Warning: Using numpy fallback due to: {str(e2)}")
            mad_series = pd.Series(np.nanvar(df.values, axis=1), index=df.index)
    
    # Get indices of top-k variable rows by MAD
    if len(mad_series) <= k:
        # If we have fewer rows than k, keep them all
        return df
    else:
        # Get indices of top-k rows by MAD (highest MAD = most variable)
        # Filter out features with zero variance first
        non_zero_mad = mad_series[mad_series > 0]
        if len(non_zero_mad) >= k:
            top_indices = non_zero_mad.nlargest(k).index
        else:
            # If we don't have enough non-zero variance features, include some zero-variance ones
            top_indices = mad_series.nlargest(k).index
        return df.loc[top_indices]

def fix_tcga_id_slicing(id_list: List[str]) -> List[str]:
    """
    Standardize TCGA patient IDs by slicing to maintain only the core part.
    
    Parameters
    ----------
    id_list : List[str]
        List of patient IDs
        
    Returns
    -------
    List[str]
        List of standardized patient IDs
    """
    fixed_ids = []
    for id_str in id_list:
        # Check if it's a TCGA ID (typically starts with TCGA)
        if isinstance(id_str, str) and id_str.startswith("TCGA"):
            # Keep only the first 12 characters, which identify the patient uniquely
            # Format typically: TCGA-XX-XXXX
            parts = id_str.split("-")
            if len(parts) >= 3:
                # Ensure we get the core patient ID (first 3 parts)
                fixed_id = "-".join(parts[:3])
                fixed_ids.append(fixed_id)
            else:
                # If the ID doesn't have enough parts, keep it as is
                fixed_ids.append(id_str)
        else:
            # Non-TCGA ID, keep it as is
            fixed_ids.append(id_str)
            
    return fixed_ids

def _remap_labels(y: pd.Series, dataset: str) -> pd.Series:
    """
    Dynamic label re-mapping helper for classification datasets.
    
    1. Merges ultra-rare classes (<3 samples) into the first rare label
    2. Applies dataset-specific binary conversions
    3. Ensures output is always numeric for downstream compatibility
    
    Parameters
    ----------
    y : pd.Series
        Target labels
    dataset : str
        Dataset name for specific conversions
        
    Returns
    -------
    pd.Series
        Re-mapped labels (always numeric)
    """
    # Step 1: Merge ultra-rare classes (<3 samples)
    vc = y.value_counts()
    rare = vc[vc < 3].index
    if len(rare) > 0:
        logger.info(f"Dataset {dataset}: Merging {len(rare)} ultra-rare classes with <3 samples: {list(rare)}")
        # Merge all rare classes into the first rare label
        y = y.replace(dict.fromkeys(rare, rare[0]))
        logger.info(f"Dataset {dataset}: After merging, class distribution: {y.value_counts().to_dict()}")
    
    # Step 2: Dataset-specific binary conversions
    if dataset == 'Colon':
        # Convert T-stage to early/late binary classification
        original_unique = y.unique()
        logger.info(f"Dataset {dataset}: Original classes: {list(original_unique)}")
        y = y.map(lambda s: 'early' if str(s) in {'T1', 'T2'} else 'late')
        logger.info(f"Dataset {dataset}: After binary conversion: {y.value_counts().to_dict()}")
    
    # Step 3: Ensure output is always numeric
    if y.dtype == 'object' or not pd.api.types.is_numeric_dtype(y):
        # Convert string labels to numeric
        unique_labels = y.unique()
        label_mapping = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        y = y.map(label_mapping)
        logger.info(f"Dataset {dataset}: Converted to numeric labels: {label_mapping}")
        logger.info(f"Dataset {dataset}: Final numeric distribution: {y.value_counts().to_dict()}")
    
    return y

def custom_parse_outcome(series: pd.Series, outcome_type: str, dataset: str = None) -> pd.Series:
    """
    Parse outcome data based on the specified type.
    
    Parameters
    ----------
    series : pd.Series
        Series containing outcome data
    outcome_type : str
        Type of outcome data ('os', 'pfs', 'response', etc.)
    dataset : str, optional
        Dataset name for specific label re-mapping
        
    Returns
    -------
    pd.Series
        Parsed outcome data
    """
    # If series is a numpy.ndarray, convert to pandas Series first
    if isinstance(series, np.ndarray):
        series = pd.Series(series)
    
    # Handle the case when we cannot check for NaN values with isna()
    if not hasattr(series, 'isna'):
        # Convert to pandas Series to ensure we have the isna method
        series = pd.Series(series)
    
    if outcome_type in ['os', 'pfs', 'survival', 'continuous']:
        # For continuous outcomes like survival time, convert to float
        return pd.to_numeric(series, errors='coerce')
    elif outcome_type in ['response', 'class', 'status']:
        # For categorical outcomes, handle various formats
        if all(isinstance(x, (int, float, np.number)) or 
               (isinstance(x, str) and x.isdigit()) for x in series if pd.notna(x)):
            # If all values are numeric or numeric strings, convert to integers
            parsed_series = pd.to_numeric(series, errors='coerce').astype('Int64')
        else:
            # For text categories like "Responder"/"Non-responder", encode as categorical
            try:
                parsed_series = pd.Categorical(series).codes
                parsed_series = pd.Series(parsed_series, index=series.index)
            except:
                # If categorical encoding fails, try to convert to string first
                parsed_series = pd.Categorical(series.astype(str)).codes
                parsed_series = pd.Series(parsed_series, index=series.index)
        
        # Apply dynamic label re-mapping for classification datasets
        if dataset is not None:
            parsed_series = _remap_labels(parsed_series, dataset)
        
        return parsed_series
    else:
        # Default handling for unknown types - try numeric conversion with fallback
        try:
            return pd.to_numeric(series, errors='coerce')
        except Exception as e:
            logger.warning(f"Failed to convert outcome to numeric: {str(e)}")
            # Last resort - convert to string and then categorical
            try:
                return pd.Categorical(series.astype(str)).codes
            except Exception as e2:
                logger.warning(f"Failed to convert outcome to categorical: {str(e2)}")
                return series

def safe_convert_to_numeric(X: Any) -> np.ndarray:
    """
    Safely convert data to numeric numpy array, handling various input types.
    
    Parameters
    ----------
    X : Any
        Input data to convert
        
    Returns
    -------
    np.ndarray
        Numeric numpy array (float64 for sklearn compatibility)
    """
    try:
        if isinstance(X, pd.DataFrame):
            # Convert DataFrame to numpy array, filling NaNs with 0
            # Use float64 for sklearn compatibility
            return X.fillna(0).astype(np.float64).values
        elif isinstance(X, pd.Series):
            # Convert Series to numpy array, filling NaNs with 0
            return X.fillna(0).astype(np.float64).values
        elif isinstance(X, list):
            # Convert list to numpy array - use float64 for sklearn compatibility
            X_np = np.array(X, dtype=np.float64)
            # Replace any NaNs with 0
            X_np = np.nan_to_num(X_np, nan=0.0)
            return X_np
        elif isinstance(X, np.ndarray):
            # Already a numpy array, ensure float64 and handle NaNs
            X_float = X.astype(np.float64)
            # Replace any NaNs with 0
            X_float = np.nan_to_num(X_float, nan=0.0)
            return X_float
        else:
            # Unsupported type, try to convert to numpy array with float64
            X_np = np.array(X, dtype=np.float64)
            # Replace any NaNs with 0
            X_np = np.nan_to_num(X_np, nan=0.0)
            return X_np
    except Exception as e:
        logger.error(f"Error in safe_convert_to_numeric: {str(e)}")
        # Last resort: empty array with appropriate shape - use float64
        return np.zeros((1, 1), dtype=np.float64)

def process_with_missing_modalities(data_modalities: Dict[str, pd.DataFrame], 
                                   all_ids: List[str],
                                   missing_percentage: float,
                                   random_state: Optional[int] = None,
                                   min_overlap_ratio: float = 0.3) -> Dict[str, pd.DataFrame]:
    """
    Process modalities by randomly marking some samples as missing.
    This simulates real-world scenarios where some samples might not have data for all modalities.
    
    Parameters
    ----------
    data_modalities : Dict[str, pd.DataFrame]
        Dictionary of modality DataFrames
    all_ids : List[str]
        List of all sample IDs
    missing_percentage : float
        Percentage of data to mark as missing (0.0 to 1.0)
    random_state : Optional[int]
        Random seed for reproducibility
    min_overlap_ratio : float
        Minimum ratio of samples that must be present in all modalities
        
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary of processed modality DataFrames
    """
    # If missing_percentage is 0, return original data
    if missing_percentage == 0.0:
        return data_modalities
    
    # Set random seed for reproducibility if provided
    if random_state is not None:
        np.random.seed(random_state)
    
    # Initialize result dictionary - we'll modify DataFrames in-place when possible
    modified_modalities = {}
    
    # Keep track of sample availability for efficient overlap calculation
    sample_availability = {id_: set() for id_ in all_ids}
    
    # Ensure we have IDs as a set for O(1) lookup
    all_ids_set = set(all_ids)
    
    # Process each modality
    for mod_name, mod_df in data_modalities.items():
        if mod_df is None or mod_df.empty:
            modified_modalities[mod_name] = mod_df
            continue
        
        # Get available samples in this modality
        avail_samples = set(mod_df.columns).intersection(all_ids_set)
        
        # Skip modalities with very few samples
        if len(avail_samples) < 5:
            modified_modalities[mod_name] = mod_df
            for id_ in avail_samples:
                sample_availability[id_].add(mod_name)
            continue
        
        # Decide how many samples to keep (non-missing)
        samples_to_keep = max(
            int(len(avail_samples) * (1.0 - missing_percentage)),
            5  # Ensure at least 5 samples remain
        )
        
        # Randomly select samples to keep
        if samples_to_keep < len(avail_samples):
            samples_list = list(avail_samples)
            np.random.shuffle(samples_list)  # In-place shuffle
            keep_samples = set(samples_list[:samples_to_keep])
            
            # Filter the modality to keep only selected samples
            # Use view when possible to avoid copying
            keep_cols = [col for col in mod_df.columns if col in keep_samples]
            modified_modalities[mod_name] = mod_df[keep_cols]
            
            # Update sample availability
            for id_ in keep_samples:
                sample_availability[id_].add(mod_name)
        else:
            # Keep all samples if we would keep more than available
            modified_modalities[mod_name] = mod_df
            for id_ in avail_samples:
                sample_availability[id_].add(mod_name)
    
    # Calculate the number of modalities each sample appears in
    sample_mod_counts = {id_: len(mods) for id_, mods in sample_availability.items()}
    
    # Find samples present in all modalities
    all_mod_count = len(data_modalities)
    samples_in_all = [id_ for id_, count in sample_mod_counts.items() if count == all_mod_count]
    
    # Check if we have enough overlap
    if len(samples_in_all) < max(5, min_overlap_ratio * len(all_ids)):
        # We need to adjust to ensure sufficient overlap
        modality_names = list(data_modalities.keys())
        
        # Find samples with high presence but not in all modalities
        near_complete_samples = [
            id_ for id_, count in sample_mod_counts.items() 
            if count >= all_mod_count - 1 and count < all_mod_count
        ]
        
        # For some of these samples, add them to all modalities they're missing from
        np.random.shuffle(near_complete_samples)
        samples_to_add = near_complete_samples[:min(
            len(near_complete_samples),
            max(5, int(min_overlap_ratio * len(all_ids))) - len(samples_in_all)
        )]
        
        # Add these samples to modalities they're missing from
        for id_ in samples_to_add:
            missing_mods = [mod for mod in modality_names if mod not in sample_availability[id_]]
            for mod_name in missing_mods:
                if mod_name in modified_modalities and modified_modalities[mod_name] is not None:
                    # Check if sample is available in original data
                    if id_ in data_modalities[mod_name].columns:
                        # Get the column data
                        col_data = data_modalities[mod_name][id_]
                        
                        # Ensure we're working with a copy to avoid SettingWithCopyWarning
                        if isinstance(modified_modalities[mod_name], pd.DataFrame):
                            # Create a proper copy and add the column using pd.concat to avoid warnings
                            current_df = modified_modalities[mod_name].copy()
                            
                            # Add the column using pd.concat instead of direct assignment
                            new_col_df = pd.DataFrame({id_: col_data})
                            modified_modalities[mod_name] = pd.concat([current_df, new_col_df], axis=1)
    
    return modified_modalities

def align_modality_data(modalities: Dict[str, pd.DataFrame], common_ids: List[str], target_col: str) -> List[str]:
    """
    Align modality data with common IDs, prioritizing samples that appear across all modalities.
    
    Parameters
    ----------
    modalities : Dict[str, pd.DataFrame]
        Dictionary of modality DataFrames
    common_ids : List[str]
        List of common IDs across modalities
    target_col : str
        Target column name in labels
        
    Returns
    -------
    List[str]
        List of aligned sample IDs
    """
    # If no modalities, return empty list
    if not modalities:
        return []
    
    # Track which samples appear in how many modalities
    overlap_counts = {}
    
    # Count how many modalities each sample appears in
    for mod_name, mod_df in modalities.items():
        if mod_df.empty:
            continue
        
        for sample_id in mod_df.columns:
            if sample_id in common_ids:
                overlap_counts[sample_id] = overlap_counts.get(sample_id, 0) + 1
    
    # Prioritize samples that appear in all modalities
    all_modalities_count = len([m for m in modalities.values() if not m.empty])
    
    # Get samples present in all modalities first, then samples with partial presence
    complete_samples = [id for id in common_ids if overlap_counts.get(id, 0) == all_modalities_count]
    
    # If we have enough complete samples, use only those
    if len(complete_samples) >= 5:
        logger.info(f"Found {len(complete_samples)} samples present in all {all_modalities_count} modalities")
        return sorted(complete_samples)
    
    # Otherwise, prioritize samples by how many modalities they appear in
    partial_samples = sorted(
        [(id, count) for id, count in overlap_counts.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    # Take samples with highest overlap first
    prioritized_samples = [id for id, _ in partial_samples if id in common_ids]
    
    logger.info(f"Using {len(prioritized_samples)} prioritized samples - complete samples: {len(complete_samples)}")
    return prioritized_samples

def normalize_sample_ids(ids: List[str], target_separator: str = '-') -> Dict[str, str]:
    """
    Normalize sample IDs by replacing different separators with a target separator.
    Useful for standardizing IDs across different data sources.
    
    Parameters
    ----------
    ids : List[str]
        List of sample IDs to normalize
    target_separator : str, default='-'
        The separator to use in the normalized IDs
        
    Returns
    -------
    Dict[str, str]
        Dictionary mapping original IDs to normalized IDs
    """
    separators = ['-', '.', '_', ' ', '+']
    id_map = {}
    
    for id_str in ids:
        if not isinstance(id_str, str):
            continue
            
        normalized = id_str
        # Replace all separators (except target) with the target separator
        for sep in separators:
            if sep != target_separator and sep in normalized:
                normalized = normalized.replace(sep, target_separator)
        
        # Only add to map if it actually changed
        if normalized != id_str:
            id_map[id_str] = normalized
    
    return id_map

def filter_rare_categories(series: pd.Series, min_count: int = 3) -> pd.Series:
    """
    Filter out rare categories from a Series.
    
    Parameters
    ----------
    series : pd.Series
        Series containing categorical data
    min_count : int
        Minimum count for a category to be retained
        
    Returns
    -------
    pd.Series
        Series with rare categories filtered out
    """
    if pd.api.types.is_numeric_dtype(series) or len(series) < 5:
        # Not categorical or too few samples
        return series
        
    # Count values and identify rare categories
    value_counts = series.value_counts()
    rare_categories = value_counts[value_counts < min_count].index.tolist()
    
    # If most categories are rare, don't filter
    if len(rare_categories) > len(value_counts) / 2:
        return series
        
    # Replace rare categories with NaN
    filtered = series.copy()
    filtered[filtered.isin(rare_categories)] = np.nan
    
    return filtered

def advanced_feature_filtering(df: pd.DataFrame, 
                             config: Dict = None) -> pd.DataFrame:
    """
    Apply advanced feature filtering for high-dimensional, small-sample data.
    OPTIMIZED VERSION - Removed expensive correlation filtering for speed.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (features × samples)
    config : Dict
        Configuration dictionary with filtering parameters
        
    Returns
    -------
    pd.DataFrame
        Filtered dataframe
    """
    if config is None:
        config = PREPROCESSING_CONFIG
    
    # 1. Remove features with too many missing values
    missing_threshold = config.get("missing_threshold", 0.5)
    if missing_threshold < 1.0:
        missing_ratio = df.isnull().sum(axis=1) / df.shape[1]
        df = df[missing_ratio <= missing_threshold]
    
    # 2. Remove low-variance features (more aggressive for speed)
    variance_threshold = config.get("variance_threshold", 0.01)
    if variance_threshold > 0:
        variances = df.var(axis=1, skipna=True)
        df = df[variances > variance_threshold]
    
    # 3. Remove highly correlated features (RE-ENABLED with optimizations)
    if config.get("remove_highly_correlated", False):
        correlation_threshold = config.get("correlation_threshold", 0.95)
        if df.shape[0] > 1:  # Only if we have more than 1 feature
            try:
                # OPTIMIZATION: Use sample of features if too many (>5000) for initial screening
                if df.shape[0] > 5000:
                    # Sample features for correlation analysis to speed up
                    sample_size = min(2000, df.shape[0])
                    sample_indices = np.random.choice(df.index, sample_size, replace=False)
                    df_sample = df.loc[sample_indices]
                    
                    # Calculate correlation matrix on sample
                    corr_matrix = df_sample.T.corr().abs()
                    
                    # Find highly correlated pairs in sample
                    upper_tri = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                    high_corr_pairs = np.where((corr_matrix > correlation_threshold) & upper_tri)
                    
                    # Get feature names to drop from sample
                    to_drop_sample = set()
                    for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]):
                        # Drop the feature with lower variance
                        var_i = df_sample.iloc[i].var()
                        var_j = df_sample.iloc[j].var()
                        if var_i < var_j:
                            to_drop_sample.add(df_sample.index[i])
                        else:
                            to_drop_sample.add(df_sample.index[j])
                    
                    # Apply filtering to full dataset
                    df = df.drop(index=list(to_drop_sample), errors='ignore')
                    
                else:
                    # Full correlation analysis for smaller datasets
                    corr_matrix = df.T.corr().abs()
                    upper_tri = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                    high_corr_pairs = np.where((corr_matrix > correlation_threshold) & upper_tri)
                    
                    to_drop = set()
                    for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]):
                        var_i = df.iloc[i].var()
                        var_j = df.iloc[j].var()
                        if var_i < var_j:
                            to_drop.add(df.index[i])
                        else:
                            to_drop.add(df.index[j])
                    
                    df = df.drop(index=list(to_drop))
                    
            except Exception as e:
                logger.warning(f"Correlation filtering failed: {str(e)}")
    
    # 4. Remove outlier features (simplified for speed)
    outlier_threshold = config.get("outlier_threshold", 3.0)
    if outlier_threshold > 0:
        try:
            # Simplified outlier detection using IQR method (faster than z-score)
            Q1 = df.quantile(0.25, axis=1)
            Q3 = df.quantile(0.75, axis=1)
            IQR = Q3 - Q1
            
            # Features with extreme IQR are likely outliers
            outlier_mask = (IQR > IQR.quantile(0.95)) | (IQR < IQR.quantile(0.05))
            df = df[~outlier_mask]
            
        except Exception as e:
            logger.warning(f"Outlier filtering failed: {str(e)}")
    
    return df

def log_transform_data(X, offset=1e-6):
    """
    Apply log transformation to handle skewed gene expression data.
    
    Args:
        X: Input data
        offset: Small value to add before log to handle zeros
    
    Returns:
        Log-transformed data
    """
    try:
        # Check for negative values
        min_val = np.min(X)
        max_val = np.max(X)
        
        if min_val < 0:
            # Data contains negative values - likely already processed/normalized
            logging.info(f"Data contains negative values (min={min_val:.3f}), skipping log transformation")
            return X
        elif max_val <= 50:
            # Data appears already log-transformed or normalized
            logging.info(f"Data appears already transformed (max={max_val:.2f}), skipping log transformation")
            return X
        else:
            # Apply log transformation to raw counts
            X_log = np.log1p(X + offset)
            logging.info(f"Applied log transformation with offset {offset}")
            return X_log
    except Exception as e:
        logging.warning(f"Log transformation failed: {e}")
        return X

def quantile_normalize_data(X, n_quantiles=1000):
    """
    Apply quantile normalization for robust scaling of biomedical data.
    
    Args:
        X: Input data
        n_quantiles: Number of quantiles for transformation
    
    Returns:
        Quantile-normalized data
    """
    try:
        # Use fewer quantiles for small datasets
        n_samples = X.shape[0]
        n_quantiles = min(n_quantiles, n_samples)
        
        transformer = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution='normal',
            random_state=42
        )
        X_quantile = transformer.fit_transform(X)
        logging.info(f"Applied quantile normalization with {n_quantiles} quantiles")
        return X_quantile, transformer
    except Exception as e:
        logging.warning(f"Quantile normalization failed: {e}")
        return X, None

def handle_sparse_features(X, variance_threshold=0.001):
    """
    Handle sparse features common in biomedical data.
    
    Args:
        X: Input data
        variance_threshold: Minimum variance threshold
    
    Returns:
        Data with low-variance features removed
    """
    try:
        # Calculate sparsity
        sparsity = np.mean(X == 0)
        logging.info(f"Data sparsity: {sparsity:.2%}")
        
        # Remove low-variance features
        selector = VarianceThreshold(threshold=variance_threshold)
        X_filtered = selector.fit_transform(X)
        
        n_removed = X.shape[1] - X_filtered.shape[1]
        logging.info(f"Removed {n_removed} low-variance features (threshold: {variance_threshold})")
        
        return X_filtered, selector
    except Exception as e:
        logging.warning(f"Sparse feature handling failed: {e}")
        return X, None

def robust_outlier_detection(X, threshold=4.0):
    """
    Robust outlier detection for biomedical data.
    
    Args:
        X: Input data
        threshold: Z-score threshold for outlier detection
    
    Returns:
        Data with outliers handled
    """
    try:
        # Use robust statistics (median, MAD)
        median = np.median(X, axis=0)
        mad = np.median(np.abs(X - median), axis=0)
        
        # Avoid division by zero
        mad = np.where(mad == 0, 1, mad)
        
        # Calculate modified z-scores
        modified_z_scores = 0.6745 * (X - median) / mad
        
        # Identify outliers
        outliers = np.abs(modified_z_scores) > threshold
        
        # Cap outliers at threshold
        X_capped = X.copy()
        X_capped[outliers] = np.sign(X[outliers]) * threshold * mad + median
        
        n_outliers = np.sum(outliers)
        logging.info(f"Capped {n_outliers} outliers (threshold: {threshold})")
        
        return X_capped
    except Exception as e:
        logging.warning(f"Outlier detection failed: {e}")
        return X

def impute_missing_values(X, strategy='median'):
    """
    Impute missing values with robust strategy.
    
    Args:
        X: Input data
        strategy: Imputation strategy
    
    Returns:
        Data with missing values imputed
    """
    try:
        imputer = SimpleImputer(strategy=strategy)
        X_imputed = imputer.fit_transform(X)
        
        n_missing = np.sum(np.isnan(X))
        logging.info(f"Imputed {n_missing} missing values using {strategy} strategy")
        
        return X_imputed, imputer
    except Exception as e:
        logging.warning(f"Missing value imputation failed: {e}")
        return X, None

def biomedical_preprocessing_pipeline(X, y=None, config=None):
    """
    Comprehensive preprocessing pipeline for biomedical data.
    
    Args:
        X: Feature matrix
        y: Target vector (optional)
        config: Preprocessing configuration
    
    Returns:
        Preprocessed data and transformers
    """
    if config is None:
        config = PREPROCESSING_CONFIG
    
    transformers = {}
    
    logging.info("Starting biomedical preprocessing pipeline")
    logging.info(f"Initial data shape: {X.shape}")
    
    # Step 1: Handle missing values
    if config.get('impute_missing', True):
        X, imputer = impute_missing_values(X)
        transformers['imputer'] = imputer
    
    # Step 2: Log transformation for gene expression data
    if config.get('log_transform', True):
        X = log_transform_data(X)
    
    # Step 3: Handle sparse features
    if config.get('remove_low_variance', True):
        X, variance_selector = handle_sparse_features(
            X, config.get('variance_threshold', 0.001)
        )
        transformers['variance_selector'] = variance_selector
    
    # Step 4: Outlier handling
    if config.get('handle_outliers', True):
        X = robust_outlier_detection(X, config.get('outlier_threshold', 4.0))
    
    # Step 5: Quantile normalization
    if config.get('quantile_transform', True):
        X, quantile_transformer = quantile_normalize_data(X)
        transformers['quantile_transformer'] = quantile_transformer
    
    # Step 6: Remove highly correlated features
    if config.get('remove_highly_correlated', True):
        X, correlation_selector = remove_highly_correlated_features(
            X, config.get('correlation_threshold', 0.98)
        )
        transformers['correlation_selector'] = correlation_selector
    
    logging.info(f"Final preprocessed data shape: {X.shape}")
    
    return X, transformers

def remove_highly_correlated_features(X, threshold=0.98):
    """
    Remove highly correlated features.
    
    Args:
        X: Input data
        threshold: Correlation threshold
    
    Returns:
        Data with highly correlated features removed
    """
    try:
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(X.T)
        
        # Find highly correlated pairs
        high_corr_pairs = np.where(
            (np.abs(corr_matrix) > threshold) & 
            (np.abs(corr_matrix) < 1.0)
        )
        
        # Select features to remove
        features_to_remove = set()
        for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]):
            if i < j:  # Avoid duplicates
                # Remove the feature with lower variance
                var_i = np.var(X[:, i])
                var_j = np.var(X[:, j])
                if var_i < var_j:
                    features_to_remove.add(i)
                else:
                    features_to_remove.add(j)
        
        # Create selector
        features_to_keep = [i for i in range(X.shape[1]) if i not in features_to_remove]
        X_filtered = X[:, features_to_keep]
        
        logging.info(f"Removed {len(features_to_remove)} highly correlated features")
        
        return X_filtered, features_to_keep
    except Exception as e:
        logging.warning(f"Correlation filtering failed: {e}")
        return X, None 