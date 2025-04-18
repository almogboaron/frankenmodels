#!/usr/bin/env python3
"""
Visualization script for Frankenmodels.

This script generates publication-quality visualizations of Frankenmodel architectures
and performance metrics from experimental results.
"""

import argparse
import os
import pandas as pd
import sys
from typing import List, Dict, Optional

from visualizations.model_architecture import (
    visualize_model_architecture,
    visualize_layer_impact,
    visualize_multi_metric_comparison
)


def create_output_directory(output_dir: str) -> None:
    """Create output directory if it doesn't exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")


def load_results_data(input_file: str) -> Optional[pd.DataFrame]:
    """Load experimental results from CSV file."""
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        print("Run layer_check.py first to generate the necessary data.")
        return None
    
    try:
        df = pd.read_csv(input_file)
        print(f"Successfully loaded data from {input_file}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def extract_metrics(df: pd.DataFrame) -> tuple:
    """Extract metrics and base metrics from the dataframe."""
    # Extract layer metrics
    layer_metrics = {
        row["layer_idx"]: {
            "accuracy": row["accuracy"],
            "f1": row["f1"],
            "precision": row["precision"],
            "recall": row["recall"]
        }
        for _, row in df.iterrows()
    }
    
    # Calculate base metrics from differences in the first row
    base_metrics = {
        "accuracy": df.iloc[0]["accuracy"] - df.iloc[0]["acc_diff"],
        "f1": df.iloc[0]["f1"] - df.iloc[0]["f1_diff"],
        "precision": df.iloc[0]["precision"] - df.iloc[0]["precision_diff"],
        "recall": df.iloc[0]["recall"] - df.iloc[0]["recall_diff"]
    }
    
    print(f"Base model metrics: {base_metrics}")
    print(f"Extracted metrics for {len(layer_metrics)} layers")
    
    return layer_metrics, base_metrics


def generate_visualizations(
    layer_metrics: Dict[int, Dict[str, float]],
    base_metrics: Dict[str, float],
    output_dir: str,
    duplicate_layers: Optional[List[int]] = None,
    duplication_counts: Optional[List[int]] = None
) -> None:
    """Generate all visualizations and save them to the output directory."""
    # 1. Model architecture visualization
    print("Generating model architecture visualization...")
    arch_path = os.path.join(output_dir, "frankenmodel_architecture.png")
    visualize_model_architecture(
        duplicate_layers=duplicate_layers,
        duplication_counts=duplication_counts,
        base_layers=12,  # BERT-base has 12 layers
        model_name="BERT",
        save_path=arch_path,
        style="academic"  # Use the academic style for publication
    )
    
    # 2. Per-metric visualizations
    for metric in ["accuracy", "f1", "precision", "recall"]:
        print(f"Generating visualization for {metric}...")
        metric_path = os.path.join(output_dir, f"layer_impact_{metric}.png")
        visualize_layer_impact(
            layer_metrics=layer_metrics,
            base_metrics=base_metrics,
            metric_name=metric,
            save_path=metric_path
        )
    
    # 3. Multi-metric comparison
    print("Generating multi-metric comparison...")
    multi_path = os.path.join(output_dir, "multi_metric_comparison.png")
    visualize_multi_metric_comparison(
        layer_metrics=layer_metrics,
        base_metrics=base_metrics,
        save_path=multi_path
    )
    
    print(f"All visualizations have been saved to {output_dir}")


def parse_layer_spec(layer_spec: Optional[str]) -> tuple:
    """Parse the layer specification string into duplicate_layers and duplication_counts."""
    if not layer_spec:
        return None, None
    
    try:
        # Format: "0:1,3:2,8:1" meaning layer 0 duplicated once, layer 3 duplicated twice, etc.
        parts = layer_spec.split(',')
        duplicate_layers = []
        duplication_counts = []
        
        for part in parts:
            layer, count = part.split(':')
            duplicate_layers.append(int(layer))
            duplication_counts.append(int(count))
        
        return duplicate_layers, duplication_counts
    except Exception as e:
        print(f"Error parsing layer specification: {e}")
        print("Format should be like '0:1,3:2,8:1' (layer:count,layer:count,...)")
        return None, None


def main() -> None:
    """Main function to parse arguments and generate visualizations."""
    parser = argparse.ArgumentParser(
        description="Generate visualizations of Frankenmodel architecture and performance metrics.")
    parser.add_argument("--input", type=str, default="layer_replication_results.csv",
                      help="Path to the CSV file with layer replication results")
    parser.add_argument("--output_dir", type=str, default="visualizations",
                      help="Directory to save visualizations")
    parser.add_argument("--layers", type=str, default=None,
                      help="Optional specification of which layers to visualize in the architecture, " 
                           "format: '0:1,3:2,8:1' meaning layer 0 duplicated once, layer 3 duplicated twice, etc.")

    args = parser.parse_args()
    
    # Create output directory
    create_output_directory(args.output_dir)
    
    # Load results data
    df = load_results_data(args.input)
    if df is None:
        sys.exit(1)
    
    # Extract metrics
    layer_metrics, base_metrics = extract_metrics(df)
    
    # Parse layer specification
    duplicate_layers, duplication_counts = parse_layer_spec(args.layers)
    
    # Generate visualizations
    generate_visualizations(
        layer_metrics=layer_metrics,
        base_metrics=base_metrics,
        output_dir=args.output_dir,
        duplicate_layers=duplicate_layers,
        duplication_counts=duplication_counts
    )


if __name__ == "__main__":
    main() 