#!/usr/bin/env python3
"""
Script to visualize standard model architectures.

This script creates visualizations of standard model architectures
using the same style as the Frankenmodel visualizations.
"""

import argparse
import os
import json
from typing import List, Dict, Optional

from visualizations.model_architecture import visualize_model_architecture


def parse_layer_info(layer_info_str: str) -> Optional[List[str]]:
    """Parse the layer information string into a list of layer descriptions."""
    try:
        if layer_info_str.endswith('.json'):
            # If it's a JSON file, load it
            with open(layer_info_str, 'r') as f:
                info = json.load(f)
                # Check if it's a list or a dict with a 'layers' key
                if isinstance(info, list):
                    return info
                elif isinstance(info, dict) and 'layers' in info:
                    return info['layers']
                else:
                    print("JSON file must contain either a list of layer descriptions or a dict with a 'layers' key")
                    return None
        else:
            # Treat as comma-separated values
            return layer_info_str.split(',')
    except Exception as e:
        print(f"Error parsing layer information: {e}")
        print("Format should be either a comma-separated list or a path to a JSON file")
        return None


def main() -> None:
    """Main function to parse arguments and generate standard model architecture visualization."""
    parser = argparse.ArgumentParser(
        description="Generate visualization of standard model architecture.")
    parser.add_argument("--output", type=str, default="visualization_results/standard_model_architecture.png",
                      help="Path to save the visualization")
    parser.add_argument("--model", type=str, default="BERT",
                      help="Name of the model (e.g., 'BERT', 'GPT', 'T5')")
    parser.add_argument("--layers", type=int, default=12,
                      help="Number of layers in the model")
    parser.add_argument("--layer_info", type=str, 
                      help="Layer information as comma-separated descriptions or path to JSON file")
    parser.add_argument("--style", type=str, default="modern",
                      choices=["modern", "academic"],
                      help="Visualization style ('modern' or 'academic')")
    parser.add_argument("--figsize", type=str, default="12,8",
                      help="Figure size in inches as width,height (e.g., '12,8')")

    args = parser.parse_args()
    
    # Parse figure size
    try:
        width, height = map(int, args.figsize.split(','))
        figsize = (width, height)
    except ValueError:
        print(f"Invalid figsize format: {args.figsize}. Using default (12,8).")
        figsize = (12, 8)
    
    # Parse layer information if provided
    layer_descriptions = None
    if args.layer_info:
        layer_descriptions = parse_layer_info(args.layer_info)
        if layer_descriptions and len(layer_descriptions) != args.layers:
            print(f"Warning: Number of layer descriptions ({len(layer_descriptions)}) " 
                  f"doesn't match specified number of layers ({args.layers})")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir:  # Only create directory if path contains a directory
        os.makedirs(output_dir, exist_ok=True)
    
    # Generate visualization
    # We pass None for duplicate_layers and duplication_counts to visualize a standard model
    visualize_model_architecture(
        duplicate_layers=None,
        duplication_counts=None,
        base_layers=args.layers,
        model_name=args.model,
        save_path=args.output,
        figsize=figsize,
        style=args.style,
        layer_descriptions=layer_descriptions
    )
    
    print(f"Standard model architecture visualization saved to {args.output}")


if __name__ == "__main__":
    main() 