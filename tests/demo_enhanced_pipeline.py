#!/usr/bin/env python3
"""
Demo: Enhanced Pipeline with All 4 Architectural Improvements

This script demonstrates how to use the enhanced preprocessing pipeline 
that integrates all 4 architectural improvements:

1. Early Data Quality Pipeline
2. Fusion-Aware Feature Selection  
3. Centralized Missing Data Management
4. Coordinated Validation Framework
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple

# Configure logging to see the pipeline in action
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_demo_data() -> Tuple[Dict[str, Tuple[np.ndarray, List[str]]], np.ndarray]:
    """Generate realistic demo data for pipeline demonstration."""
    
    np.random.seed(42)
    n_samples = 120
    
    # Generate sample IDs
    sample_ids = [f"Patient_{i:03d}" for i in range(n_samples)]
    
    # Generate multi-modal data
    modality_data_dict = {}
    
    # Gene expression: high-dimensional, some missing values
    gene_expression = np.random.randn(n_samples, 1000) * 2 + 5
    missing_mask_exp = np.random.random((n_samples, 1000)) < 0.08  # 8% missing
    gene_expression[missing_mask_exp] = np.nan
    modality_data_dict['exp'] = (gene_expression, sample_ids.copy())
    
    # DNA methylation: medium-dimensional, bounded values
    methylation = np.random.beta(2, 5, (n_samples, 400))  # Beta distribution for methylation
    missing_mask_methy = np.random.random((n_samples, 400)) < 0.05  # 5% missing
    methylation[missing_mask_methy] = np.nan
    modality_data_dict['methy'] = (methylation, sample_ids.copy())
    
    # miRNA: lower-dimensional, some missing samples
    mirna = np.random.lognormal(0, 1.5, (n_samples, 150))
    missing_mask_mirna = np.random.random((n_samples, 150)) < 0.12  # 12% missing
    mirna[missing_mask_mirna] = np.nan
    modality_data_dict['mirna'] = (mirna, sample_ids.copy())
    
    # Generate realistic target: disease status (0=healthy, 1=disease)
    # Create some correlation with the data to make it realistic
    feature_signal = np.mean(gene_expression[:, :50], axis=1, where=~np.isnan(gene_expression[:, :50]))
    feature_signal = np.nan_to_num(feature_signal, nan=np.nanmean(feature_signal))
    probabilities = 1 / (1 + np.exp(-(feature_signal - np.median(feature_signal))))
    y = np.random.binomial(1, probabilities)
    
    logger.info("Generated demo dataset:")
    logger.info(f"   Samples: {n_samples}")
    logger.info(f"  🧬 Gene expression: {gene_expression.shape} ({np.sum(missing_mask_exp)} missing values)")
    logger.info(f"  🔬 Methylation: {methylation.shape} ({np.sum(missing_mask_methy)} missing values)")
    logger.info(f"  🧪 miRNA: {mirna.shape} ({np.sum(missing_mask_mirna)} missing values)")
    logger.info(f"   Target distribution: {np.sum(y==0)} healthy, {np.sum(y==1)} disease")
    
    return modality_data_dict, y

def demo_enhanced_pipeline():
    """Demonstrate the enhanced pipeline with all architectural improvements."""
    
    print(" Enhanced Pipeline Demo: All 4 Architectural Improvements")
    print("=" * 70)
    
    # Generate demo data
    print("\n1. Generating Demo Data...")
    modality_data_dict, y = generate_demo_data()
    
    # Import the enhanced pipeline
    print("\n2. Loading Enhanced Pipeline...")
    try:
        from enhanced_pipeline_integration import run_enhanced_preprocessing_pipeline
        logger.info(" Enhanced pipeline integration loaded successfully")
    except ImportError as e:
        logger.error(f" Failed to load enhanced pipeline: {e}")
        return
    
    # Run the enhanced pipeline with different configurations
    configurations = [
        {
            "name": "🔥 Full Enhanced Pipeline (All Phases)",
            "config": {
                "enable_early_quality_check": True,
                "enable_fusion_aware_order": True,
                "enable_centralized_missing_data": True,
                "enable_coordinated_validation": True,
                "fail_fast": False  # Don't fail on warnings for demo
            },
            "fusion_method": "snf"
        },
        {
            "name": "⚡ Fusion-Aware Only",
            "config": {
                "enable_early_quality_check": False,
                "enable_fusion_aware_order": True,
                "enable_centralized_missing_data": False,
                "enable_coordinated_validation": False
            },
            "fusion_method": "mkl"
        },
        {
            "name": "🧪 Quality + Missing Data Focus",
            "config": {
                "enable_early_quality_check": True,
                "enable_fusion_aware_order": False,
                "enable_centralized_missing_data": True,
                "enable_coordinated_validation": True
            },
            "fusion_method": "weighted_concat"
        }
    ]
    
    results = []
    
    for i, test_config in enumerate(configurations):
        print(f"\n{3+i}. Running: {test_config['name']}")
        print("-" * 50)
        
        try:
            # Run the enhanced pipeline
            final_data, y_aligned, metadata = run_enhanced_preprocessing_pipeline(
                modality_data_dict, 
                y,
                fusion_method=test_config["fusion_method"],
                task_type="classification",
                dataset_name=f"Demo_{test_config['fusion_method']}",
                **test_config["config"]
            )
            
            # Collect results
            result_info = {
                "config_name": test_config["name"],
                "fusion_method": test_config["fusion_method"],
                "final_shape": {k: v.shape for k, v in final_data.items()},
                "samples_processed": len(y_aligned),
                "quality_score": metadata.get("quality_score", "N/A"),
                "phases_enabled": metadata.get("phases_enabled", {}),
                "success": True
            }
            results.append(result_info)
            
            logger.info(f" Success! Processed {len(y_aligned)} samples")
            logger.info(f"   Final data shapes: {result_info['final_shape']}")
            logger.info(f"   Quality score: {result_info['quality_score']}")
            
        except Exception as e:
            logger.error(f" Configuration failed: {str(e)}")
            results.append({
                "config_name": test_config["name"],
                "error": str(e),
                "success": False
            })
    
    # Print comprehensive results summary
    print("\n" + "=" * 70)
    print(" ENHANCED PIPELINE DEMO RESULTS")
    print("=" * 70)
    
    for i, result in enumerate(results):
        print(f"\n{i+1}. {result['config_name']}")
        if result["success"]:
            print(f"    Status: SUCCESS")
            print(f"    Fusion Method: {result['fusion_method']}")
            print(f"   📏 Final Shapes: {result['final_shape']}")
            print(f"   👥 Samples: {result['samples_processed']}")
            print(f"   📈 Quality Score: {result['quality_score']}")
            print(f"   🎛️  Phases Enabled: {result['phases_enabled']}")
        else:
            print(f"    Status: FAILED")
            print(f"   🚨 Error: {result['error']}")
    
    # Architecture comparison
    print(f"\n🏗️  ARCHITECTURAL IMPROVEMENTS SUMMARY")
    print("-" * 40)
    print(" Phase 1: Early Data Quality Pipeline - Moves target analysis early")
    print(" Phase 2: Fusion-Aware Feature Selection - Optimizes order for fusion methods")  
    print(" Phase 3: Centralized Missing Data Management - Intelligent strategy selection")
    print(" Phase 4: Coordinated Validation Framework - Hierarchical fail-fast validation")
    
    successful_runs = sum(1 for r in results if r["success"])
    print(f"\n Demo Results: {successful_runs}/{len(results)} configurations successful")
    
    if successful_runs == len(results):
        print("🎉 All enhanced pipeline configurations working perfectly!")
    else:
        print("  Some configurations had issues - check logs above")
    
    return results

def demo_individual_phases():
    """Demonstrate individual phases separately."""
    
    print("\n" + "=" * 70)
    print(" INDIVIDUAL PHASE DEMONSTRATIONS")
    print("=" * 70)
    
    # Generate small test data
    modality_data_dict, y = generate_demo_data()
    
    # Phase 1: Early Data Quality
    print("\n Phase 1: Early Data Quality Pipeline")
    print("-" * 40)
    try:
        from data_quality import run_early_data_quality_pipeline
        quality_report, guidance = run_early_data_quality_pipeline(
            modality_data_dict, y, "DemoPhase1", "classification"
        )
        print(f" Quality Assessment Complete")
        print(f"   📈 Overall Quality Score: {quality_report['overall_quality_score']:.3f}")
        print(f"   🚨 Critical Issues: {len(quality_report['critical_issues'])}")
        print(f"     Warnings: {len(quality_report['warnings'])}")
    except Exception as e:
        print(f" Phase 1 Demo Failed: {e}")
    
    # Phase 2: Fusion-Aware Preprocessing
    print("\n⚡ Phase 2: Fusion-Aware Feature Selection")
    print("-" * 40)
    try:
        from fusion_aware_preprocessing import determine_optimal_fusion_order, get_fusion_method_category
        fusion_methods = ["snf", "mkl", "weighted_concat", "early_fusion_pca"]
        print("Optimal preprocessing orders:")
        for method in fusion_methods:
            order = determine_optimal_fusion_order(method)
            category = get_fusion_method_category(method)
            print(f"   {method:.<20} {order} ({category})")
    except Exception as e:
        print(f" Phase 2 Demo Failed: {e}")
    
    # Phase 3: Missing Data Management
    print("\n Phase 3: Centralized Missing Data Management")
    print("-" * 40)
    try:
        from missing_data_handler import create_missing_data_handler
        handler = create_missing_data_handler(strategy="auto")
        analysis = handler.analyze_missing_patterns(modality_data_dict)
        print(f" Missing Data Analysis Complete")
        print(f"    Overall Missing: {analysis['overall_missing_percentage']:.1%}")
        print(f"    Recommended Strategy: {analysis['recommended_strategy']}")
        print(f"   🚨 Critical Issues: {len(analysis['critical_issues'])}")
    except Exception as e:
        print(f" Phase 3 Demo Failed: {e}")
    
    # Phase 4: Coordinated Validation
    print("\n🛡️  Phase 4: Coordinated Validation Framework")
    print("-" * 40)
    try:
        from validation_coordinator import create_validation_coordinator, ValidationSeverity
        validator = create_validation_coordinator(fail_fast=False)
        
        # Test data loading validation
        result = validator.validate_data_loading(modality_data_dict, y, "DemoPhase4")
        summary = validator.get_validation_summary()
        
        print(f" Validation Framework Active")
        print(f"    Total Issues Tracked: {summary['total_issues']}")
        print(f"   🚨 Critical: {summary['by_severity'].get('critical', 0)}")
        print(f"     Warnings: {summary['by_severity'].get('warning', 0)}")
        print(f"   ℹ️  Info: {summary['by_severity'].get('info', 0)}")
    except Exception as e:
        print(f" Phase 4 Demo Failed: {e}")

if __name__ == "__main__":
    print("🎭 ENHANCED PIPELINE ARCHITECTURE DEMO")
    print("Demonstrating all 4 architectural improvements working together")
    
    # Run main demo
    demo_results = demo_enhanced_pipeline()
    
    # Show individual phases
    demo_individual_phases()
    
    print("\n" + "=" * 70)
    print("✨ DEMO COMPLETE!")
    print("The enhanced pipeline with all 4 architectural improvements is ready for use!")
    print("=" * 70) 