#!/usr/bin/env python3
"""
Median Absolute Deviation (MAD) analysis module for multi‑modal datasets.
This module calculates MAD metrics for each dataset and modality, prints
summary statistics, and saves both the detailed and summary statistics to a
single CSV file for downstream inspection.
"""

import os
import pandas as pd
import numpy as np

# Configure matplotlib backend before any matplotlib imports to prevent tkinter errors
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for parallel processing
import matplotlib.pyplot as plt

import scikit_posthocs as sp
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
import traceback
from scipy import stats

# Local imports
from config import REGRESSION_DATASETS, CLASSIFICATION_DATASETS
from data_io import load_dataset
from logging_utils import log_mad_analysis_info, log_data_save_info, log_plot_save_info

logger = logging.getLogger(__name__)


def calculate_mad(data: np.ndarray, axis: int = 1) -> float:
    """
    Calculate Median Absolute Deviation (MAD) for a dataset.

    Parameters
    ----------
    data : np.ndarray
        Input data array (features x samples)
    axis : int
        Axis along which to calculate MAD (1 for features, 0 for samples)

    Returns
    -------
    float
        MAD value
    """
    median_vals = np.median(data, axis=axis)

    if axis == 1:
        abs_deviations = np.abs(data - median_vals[:, np.newaxis])
        mad_per_feature = np.median(abs_deviations, axis=axis)
        return np.median(mad_per_feature)
    else:
        abs_deviations = np.abs(data - median_vals[np.newaxis, :])
        mad_per_sample = np.median(abs_deviations, axis=axis)
        return np.median(mad_per_sample)


def calculate_dataset_mad_metrics(ds_conf: Dict[str, Any], is_regression: bool = True) -> Optional[Dict[str, float]]:
    """
    Calculate MAD metrics for all modalities in a dataset.
    """
    try:
        ds_name = ds_conf["name"]
        logger.info(f"Calculating MAD metrics for dataset: {ds_name}")

        modalities_list = list(ds_conf["modalities"].keys())
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

        outcome_col = ds_conf["outcome_col"]
        task_type = 'regression' if is_regression else 'classification'

        modalities_data, y_aligned, common_ids = load_dataset(
            ds_name.lower(),
            modality_short_names,
            outcome_col,
            task_type,
            parallel=True,
            use_cache=True
        )

        if modalities_data is None or len(common_ids) == 0:
            logger.warning(f"Failed to load dataset {ds_name}")
            return None

        mad_results = {}
        for mod_name, mod_df in modalities_data.items():
            if mod_df.empty:
                logger.warning(f"Empty modality {mod_name} in {ds_name}")
                continue

            data_array = mod_df.values
            mad_value = calculate_mad(data_array, axis=1)
            mad_results[f"{ds_name}_{mod_name}"] = mad_value
            logger.info(f"MAD for {ds_name} {mod_name}: {mad_value:.6f}")

        return mad_results

    except Exception as e:
        logger.error(f"Error calculating MAD for dataset {ds_conf['name']}: {str(e)}")
        logger.debug(f"Full traceback for MAD error in {ds_conf['name']}: {traceback.format_exc()}")
        return None


def calculate_all_mad_metrics() -> pd.DataFrame:
    """Calculate MAD metrics for all datasets and modalities and return a DataFrame."""
    all_mad_results = {}

    logger.info("Processing regression datasets for MAD calculation…")
    for ds_conf in REGRESSION_DATASETS:
        try:
            mad_results = calculate_dataset_mad_metrics(ds_conf, is_regression=True)
            if mad_results:
                all_mad_results.update(mad_results)
        except Exception as e:
            logger.warning(f"Failed to process regression dataset {ds_conf.get('name', 'unknown')}: {str(e)}")

    logger.info("Processing classification datasets for MAD calculation…")
    for ds_conf in CLASSIFICATION_DATASETS:
        try:
            mad_results = calculate_dataset_mad_metrics(ds_conf, is_regression=False)
            if mad_results:
                all_mad_results.update(mad_results)
        except Exception as e:
            logger.warning(f"Failed to process classification dataset {ds_conf.get('name', 'unknown')}: {str(e)}")

    if not all_mad_results:
        logger.warning("No MAD results calculated – returning empty DataFrame")
        return pd.DataFrame(columns=['dataset_modality', 'mad_value', 'dataset', 'modality'])

    mad_df = pd.DataFrame([
        {'dataset_modality': key, 'mad_value': value}
        for key, value in all_mad_results.items()
    ])
    mad_df[['dataset', 'modality']] = mad_df['dataset_modality'].str.rsplit('_', n=1, expand=True)
    return mad_df


def create_modality_specific_critical_difference_diagram(mad_df: pd.DataFrame, modality: str, output_dir: str = "output") -> None:
    """Create a critical‑difference diagram for a single modality."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    modality_df = mad_df[mad_df['modality'] == modality].copy()

    if modality_df.empty:
        logger.warning(f"No data found for modality: {modality}")
        return

    mad_values = dict(zip(modality_df['dataset_modality'], modality_df['mad_value']))

    combinations = list(mad_values.keys())
    n = len(combinations)
    sig_matrix = pd.DataFrame(np.ones((n, n)), index=combinations, columns=combinations)

    plt.figure(figsize=(12, 6), dpi=100)
    plt.title(
        f'Critical Difference Diagram – {modality.upper()} Modality MAD Values\n(Lower values = More stable data)',
        fontsize=14, fontweight='bold', pad=20
    )

    try:
        sp.critical_difference_diagram(
            ranks=mad_values,
            sig_matrix=sig_matrix,
            label_fmt_left='{label} ({rank:.3f})  ',
            label_fmt_right='  ({rank:.3f}) {label}',
            text_h_margin=0.3,
            label_props={'fontweight': 'bold', 'fontsize': 10},
            crossbar_props={'linewidth': 2, 'color': 'red'},
            marker_props={'marker': 'o', 's': 80, 'edgecolor': 'black'},
            elbow_props={'linewidth': 1},
        )

        plt.figtext(
            0.5,
            0.02,
            f'{modality.upper()} modality MAD values shown. All combinations connected (no statistical testing applied)',
            ha='center', fontsize=10, style='italic'
        )

        output_path = Path(output_dir) / f"mad_{modality}_critical_difference_diagram.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"{modality.upper()} critical difference diagram saved to: {output_path}")
        plt.close()

    except Exception as e:
        logger.error(f"Error creating {modality} critical difference diagram: {str(e)}")


def create_detailed_statistics_table(mad_df: pd.DataFrame, output_dir: str = "output") -> None:
    """
    Save detailed MAD values *and* summary statistics by modality to a single CSV
    (``mad_detailed_statistics.csv``) and print a readable version to the logger.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    mad_df_stats = mad_df.copy()

    # Normalised MAD values for potential future use
    min_mad = mad_df_stats['mad_value'].min()
    max_mad = mad_df_stats['mad_value'].max()
    mad_df_stats['mad_normalized'] = (mad_df_stats['mad_value'] - min_mad) / (max_mad - min_mad)

    # Percentile ranks and simple ordering
    mad_df_stats['percentile'] = mad_df_stats['mad_value'].rank(pct=True) * 100
    mad_df_stats = mad_df_stats.sort_values('mad_value').reset_index(drop=True)
    mad_df_stats['rank'] = range(1, len(mad_df_stats) + 1)

    # --- Section 1: detailed per‑dataset values --------------------------------
    detailed_stats = mad_df_stats[['dataset', 'modality', 'mad_value']].copy()

    # --- Section 2: summary statistics by modality -----------------------------
    summary_stats = mad_df_stats.groupby('modality')['mad_value'].agg(
        ['count', 'mean', 'std', 'min', 'max']
    )

    output_path = Path(output_dir) / "mad_detailed_statistics.csv"

    # Write detailed stats first, then append summary section
    detailed_stats.to_csv(output_path, index=False)

    with open(output_path, 'a', newline='') as fp:
        fp.write('\n# Summary statistics by modality\n')
        summary_stats.to_csv(fp)

    # Log human‑readable version -------------------------------------------------
    logger.info("\nMAD Statistics (detailed per dataset)")
    logger.info("=" * 60)
    logger.info(f"{'Dataset':<15}{'Modality':<10}{'MAD Value':<12}")
    logger.info("-" * 60)
    for _, row in detailed_stats.iterrows():
        logger.info(f"{row['dataset']:<15}{row['modality']:<10}{row['mad_value']:<12.6f}")
    logger.info("=" * 60)

    logger.info("\nSummary statistics by modality:")
    logger.info(summary_stats)
    logger.info("=" * 60)


def run_mad_analysis(output_dir: str = "output") -> None:
    """Run the complete MAD analysis pipeline."""
    log_mad_analysis_info("Starting MAD analysis pipeline")
    logger.info("Starting MAD analysis pipeline…")

    log_mad_analysis_info("Step 1: Calculating MAD metrics for all datasets")
    logger.info("Step 1: Calculating MAD metrics for all datasets…")
    
    try:
        mad_df = calculate_all_mad_metrics()

        if mad_df.empty:
            log_mad_analysis_info("No MAD metrics calculated - analysis failed", level="error")
            logger.error("No MAD metrics calculated. Exiting.")
            return

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        log_mad_analysis_info(f"Created output directory: {output_dir}")

        # Save detailed + summary statistics table
        log_mad_analysis_info("Saving detailed statistics table")
        try:
            create_detailed_statistics_table(mad_df, output_dir)
            stats_path = Path(output_dir) / "mad_detailed_statistics.csv"
            log_data_save_info("MAD_Analysis", "detailed_statistics", str(stats_path), success=True)
            log_mad_analysis_info(f"Detailed statistics saved to: {stats_path}")
        except Exception as e:
            log_data_save_info("MAD_Analysis", "detailed_statistics", str(Path(output_dir) / "mad_detailed_statistics.csv"), success=False, error_msg=str(e))
            log_mad_analysis_info(f"Failed to save detailed statistics: {str(e)}", level="error")

        # Print high‑level summary to logger
        log_mad_analysis_info("Generating high-level MAD summary")
        logger.info("\nHigh‑level MAD Summary:")
        logger.info(f"Total dataset‑modality combinations: {len(mad_df)}")
        logger.info(f"Unique datasets: {mad_df['dataset'].nunique()}")
        logger.info(f"Unique modalities: {mad_df['modality'].nunique()}")
        
        log_mad_analysis_info(f"Processed {len(mad_df)} dataset-modality combinations from {mad_df['dataset'].nunique()} datasets")

        modality_stats = mad_df.groupby('modality')['mad_value'].agg(['count', 'mean', 'std', 'min', 'max'])
        logger.info("\nMAD statistics by modality:")
        logger.info(modality_stats)
        
        # Log modality statistics
        for modality in modality_stats.index:
            stats = modality_stats.loc[modality]
            log_mad_analysis_info(f"{modality.upper()} modality: {stats['count']} datasets, mean MAD={stats['mean']:.6f}")

        # Step 2: Critical difference diagrams
        log_mad_analysis_info("Step 2: Creating modality-specific critical difference diagrams")
        logger.info("\nStep 2: Creating modality‑specific critical difference diagrams…")
        
        plot_success_count = 0
        plot_total_count = 0
        
        for mod in ['exp', 'mirna', 'methy']:
            plot_total_count += 1
            log_mad_analysis_info(f"Creating {mod.upper()} modality diagram")
            logger.info(f"Creating {mod.upper()} modality diagram…")
            
            try:
                create_modality_specific_critical_difference_diagram(mad_df, mod, output_dir)
                plot_path = Path(output_dir) / f"mad_{mod}_critical_difference_diagram.png"
                
                # Check if plot was actually created
                if plot_path.exists() and plot_path.stat().st_size > 0:
                    plot_success_count += 1
                    log_plot_save_info("MAD_Analysis", f"{mod}_critical_difference", str(plot_path), success=True)
                    log_mad_analysis_info(f"{mod.upper()} diagram saved successfully")
                else:
                    log_plot_save_info("MAD_Analysis", f"{mod}_critical_difference", str(plot_path), success=False, error_msg="Plot file not created or empty")
                    log_mad_analysis_info(f"{mod.upper()} diagram creation failed - file not found", level="warning")
                    
            except Exception as e:
                plot_path = Path(output_dir) / f"mad_{mod}_critical_difference_diagram.png"
                log_plot_save_info("MAD_Analysis", f"{mod}_critical_difference", str(plot_path), success=False, error_msg=str(e))
                log_mad_analysis_info(f"Failed to create {mod.upper()} diagram: {str(e)}", level="error")
                logger.error(f"Error creating {mod} critical difference diagram: {str(e)}")

        # Log plot creation summary
        log_mad_analysis_info(f"Created {plot_success_count}/{plot_total_count} critical difference diagrams successfully")
        
        log_mad_analysis_info("MAD analysis pipeline completed successfully")
        logger.info("MAD analysis pipeline completed successfully!")
        
    except Exception as e:
        log_mad_analysis_info(f"MAD analysis pipeline failed: {str(e)}", level="error")
        logger.error(f"Error in MAD analysis pipeline: {str(e)}")
        import traceback
        logger.debug(f"MAD analysis traceback:\n{traceback.format_exc()}")
        raise