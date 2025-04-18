#!/usr/bin/env python3
"""
Simple script to visualize Frankenmodel architectures.

This script creates visualizations of Frankenmodel architectures by specifying
which layers to duplicate and how many times to duplicate them.
"""

import argparse
import os
from typing import List, Optional

from visualizations.model_architecture import visualize_model_architecture


def parse_layer_spec(layer_spec: str) -> tuple:
    """Parse the layer specification string into duplicate_layers and duplication_counts."""
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
    """Main function to parse arguments and generate architecture visualization."""
    parser = argparse.ArgumentParser(
        description="Generate visualization of Frankenmodel architecture.")
    parser.add_argument("--layers", type=str, required=True,
                      help="Specification of which layers to duplicate, "
                           "format: '0:1,3:2,8:1' meaning layer 0 duplicated once, layer 3 duplicated twice, etc.")
    parser.add_argument("--output", type=str, default="visualization_results/frankenmodel_architecture.png",
                      help="Path to save the visualization")
    parser.add_argument("--model", type=str, default="BERT",
                      help="Name of the base model (e.g., 'BERT')")
    parser.add_argument("--base_layers", type=int, default=12,
                      help="Number of layers in the base model")
    parser.add_argument("--style", type=str, default="modern",
                      help="Visualization style ('modern' or 'academic')")

    args = parser.parse_args()
    
    # Parse layer specification
    duplicate_layers, duplication_counts = parse_layer_spec(args.layers)
    if duplicate_layers is None or duplication_counts is None:
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir:  # Only create directory if path contains a directory
        os.makedirs(output_dir, exist_ok=True)
    
    # Generate visualization
    visualize_model_architecture(
        duplicate_layers=duplicate_layers,
        duplication_counts=duplication_counts,
        base_layers=args.base_layers,
        model_name=args.model,
        save_path=args.output,
        style=args.style
    )
    
    print(f"Architecture visualization saved to {args.output}")


if __name__ == "__main__":
    main() 