#!/usr/bin/env python3
"""
Simple script to create only the critical difference diagrams from existing MAD data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scikit_posthocs as sp
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_modality_diagram(mad_df, modality, output_dir="output"):
    """Create critical difference diagram for a specific modality."""
    # Filter for specific modality
    modality_df = mad_df[mad_df['modality'] == modality].copy()
    
    if modality_df.empty:
        logger.warning(f"No data found for modality: {modality}")
        return
    
    # Use original MAD values
    mad_values = dict(zip(modality_df['dataset_modality'], modality_df['mad_value']))
    
    # Create dummy significance matrix (all non-significant for visualization)
    combinations = list(mad_values.keys())
    n_combinations = len(combinations)
    sig_matrix = pd.DataFrame(
        np.ones((n_combinations, n_combinations)),
        index=combinations,
        columns=combinations
    )
    
    # Create the plot
    plt.figure(figsize=(12, 6), dpi=100)
    plt.title(f'Critical Difference Diagram - {modality.upper()} Modality MAD Values\n(Lower values = More stable data)', 
              fontsize=14, fontweight='bold', pad=20)
    
    try:
        # Create critical difference diagram with MAD values
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
        
        # Add explanation text
        plt.figtext(0.5, 0.02, 
                   f'{modality.upper()} modality MAD values shown. All combinations connected (no statistical testing applied)',
                   ha='center', fontsize=10, style='italic')
        
        # Save the plot
        output_path = Path(output_dir) / f"mad_{modality}_critical_difference_diagram.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        logger.info(f"{modality.upper()} critical difference diagram saved to: {output_path}")
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating {modality} critical difference diagram: {str(e)}")

def main():
    """Main function to create diagrams from existing CSV data."""
    logger.info("Creating critical difference diagrams from existing MAD data...")
    
    # Load existing MAD data
    csv_path = Path("output/mad_detailed_statistics.csv")
    if not csv_path.exists():
        logger.error(f"MAD data file not found: {csv_path}")
        return
    
    # Read the CSV data
    mad_df = pd.read_csv(csv_path)
    
    # Add dataset_modality column
    mad_df['dataset_modality'] = mad_df['dataset'] + '_' + mad_df['modality']
    
    logger.info(f"Loaded MAD data: {len(mad_df)} dataset-modality combinations")
    
    # Create output directory
    Path("output").mkdir(parents=True, exist_ok=True)
    
    # Create individual modality diagrams
    logger.info("Creating EXP modality diagram...")
    create_modality_diagram(mad_df, 'exp')
    
    logger.info("Creating miRNA modality diagram...")
    create_modality_diagram(mad_df, 'mirna')
    
    logger.info("Creating Methylation modality diagram...")
    create_modality_diagram(mad_df, 'methy')
    
    logger.info("All critical difference diagrams created successfully!")

if __name__ == "__main__":
    main() 