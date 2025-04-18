#!/usr/bin/env python3
"""
Example script to visualize a BERT model with detailed layer information.

This shows how to use the layer_info parameter to add descriptive information
about each layer in the visualization.
"""

import os
import sys
import json

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualizations.model_architecture import visualize_model_architecture

# BERT layer information
BERT_LAYERS = [
    "Self-Attention + Layer Norm",
    "Feed Forward + Layer Norm",
    "Self-Attention + Layer Norm",
    "Feed Forward + Layer Norm",
    "Self-Attention + Layer Norm",
    "Feed Forward + Layer Norm",
    "Self-Attention + Layer Norm",
    "Feed Forward + Layer Norm",
    "Self-Attention + Layer Norm",
    "Feed Forward + Layer Norm",
    "Self-Attention + Layer Norm",
    "Feed Forward + Layer Norm"
]

# Alternative - more detailed layer info showing exact components
BERT_DETAILED_LAYERS = [
    "Multi-Head Attention (12 heads, dk=64)",
    "Layer Norm + FFN (3072 units)",
    "Multi-Head Attention (12 heads, dk=64)",
    "Layer Norm + FFN (3072 units)",
    "Multi-Head Attention (12 heads, dk=64)",
    "Layer Norm + FFN (3072 units)",
    "Multi-Head Attention (12 heads, dk=64)",
    "Layer Norm + FFN (3072 units)",
    "Multi-Head Attention (12 heads, dk=64)",
    "Layer Norm + FFN (3072 units)",
    "Multi-Head Attention (12 heads, dk=64)",
    "Layer Norm + FFN (3072 units)"
]

def main():
    # Create output directory if it doesn't exist
    os.makedirs("visualization_results", exist_ok=True)
    
    # Save layer info to JSON for potential reuse
    with open("visualization_results/bert_layer_info.json", "w") as f:
        json.dump({"layers": BERT_DETAILED_LAYERS}, f, indent=2)
    
    # Visualize BERT with layer information
    visualize_model_architecture(
        duplicate_layers=None,
        duplication_counts=None,
        base_layers=12,
        model_name="BERT-base",
        save_path="visualization_results/bert_with_layer_info.png",
        figsize=(12, 10),
        style="modern",
        layer_descriptions=BERT_DETAILED_LAYERS
    )
    
    print("BERT model visualization with layer information saved to visualization_results/bert_with_layer_info.png")
    print("Layer information JSON saved to visualization_results/bert_layer_info.json")

if __name__ == "__main__":
    main() 