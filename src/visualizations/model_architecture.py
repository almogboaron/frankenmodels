"""
Visualization module for Frankenmodel architecture.

This module provides functions to visualize the architecture of Frankenmodels,
highlighting layer duplications and model structure in publication-quality graphics.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from typing import List, Dict, Optional, Tuple, Union
import os


def configure_plot_style():
    """Set up matplotlib style for publication-quality figures."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'Times New Roman'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })


def visualize_model_architecture(
    duplicate_layers: Optional[List[int]] = None,
    duplication_counts: Optional[List[int]] = None,
    base_layers: int = 12,
    model_name: str = "BERT",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    highlight_color: str = "#ff7f7f",
    style: str = "modern",
    layer_descriptions: Optional[List[str]] = None
) -> plt.Figure:
    """
    Visualize the architecture of a Frankenmodel, highlighting duplicated layers.
    
    Args:
        duplicate_layers: List of encoder indices (0-indexed) that were duplicated
        duplication_counts: List of counts of how many times each layer was duplicated
        base_layers: Number of layers in the base model
        model_name: Name of the base model (e.g., "BERT")
        save_path: Optional path to save the figure
        figsize: Figure size as (width, height) in inches
        highlight_color: Color to highlight duplicated layers
        style: Visualization style ('modern' or 'academic')
        layer_descriptions: Optional list of descriptions for each layer
        
    Returns:
        The matplotlib Figure object
    """
    configure_plot_style()
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate layer information
    if duplicate_layers is None or duplication_counts is None:
        # Just visualize the base model
        layer_info = [(i, 0) for i in range(base_layers)]
        title = f"Base {model_name} Model Architecture"
    else:
        layer_info = []
        layer_idx = 0
        for i in range(base_layers):
            # Add original layer
            layer_info.append((i, 0))
            layer_idx += 1
            
            # Add duplicated layers if this layer was duplicated
            if i in duplicate_layers:
                dup_idx = duplicate_layers.index(i)
                count = duplication_counts[dup_idx]
                for j in range(count):
                    layer_info.append((i, j+1))
                    layer_idx += 1
        
        total_dups = sum(duplication_counts)
        title = f"Franken{model_name} Architecture"
    
    # Create color palette
    if style == "modern":
        # Modern color palette - more distinct colors
        colors = sns.color_palette("viridis", base_layers)
        bg_color = "#f8f9fa"  # Light background
        arrow_color = "#2c3e50"  # Dark blue for arrows
        text_color = "#343a40"  # Dark gray for text
        frame_color = "#212529"  # Almost black for frames
    else:
        # Academic/publication color palette (colorblind-friendly)
        colors = sns.color_palette("colorblind", base_layers)
        bg_color = "#ffffff"  # White background
        arrow_color = "#000000"  # Black for arrows
        text_color = "#000000"  # Black for text
        frame_color = "#000000"  # Black for frames
    
    # Set figure background
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    # Draw the model architecture
    box_height = 0.7
    box_width = 1.2
    layer_spacing = 1.3
    
    # Set up layer groups - track which original layers have duplicates
    duplicated_original_indices = set()
    if duplicate_layers is not None:
        duplicated_original_indices = set(duplicate_layers)
    
    # Draw embedding layer
    y_position = 0
    # Add special styling for embedding and output layers
    embed_color = "#adb5bd" if style == "modern" else "lightgrey"
    ax.add_patch(
        patches.FancyBboxPatch(
            (-0.5, y_position), box_width * 3, box_height,
            edgecolor=frame_color, facecolor=embed_color, alpha=0.8,
            linewidth=1.5, zorder=1, 
            boxstyle="round,pad=0.3"
        )
    )
    ax.text(
        box_width, y_position + box_height/2,
        f"{model_name} Embedding Layer", 
        ha='center', va='center', fontsize=11, fontweight='bold',
        color=text_color
    )
    
    # Group for indentation - how many layers are in each group
    layer_groups = {}
    if duplicate_layers is not None and duplication_counts is not None:
        for layer, count in zip(duplicate_layers, duplication_counts):
            layer_groups[layer] = count + 1  # original + duplicates
    
    # Draw transformer layers
    current_group = None
    group_start_idx = 0
    last_original_idx = -1
    
    for idx, (original_idx, dup_idx) in enumerate(layer_info):
        y_position = -((idx + 1) * layer_spacing)
        
        # Check if we're starting a new group
        if original_idx != last_original_idx:
            # If we were in a group, draw the group container
            if current_group is not None and original_idx in duplicated_original_indices:
                # Draw group background
                group_height = (idx - group_start_idx) * layer_spacing
                group_y = -((group_start_idx + 1) * layer_spacing)
                ax.add_patch(
                    patches.FancyBboxPatch(
                        (0.4, group_y - 0.25), box_width + 0.2, group_height + 0.5,
                        edgecolor=to_rgba(colors[current_group], 0.7),
                        facecolor=to_rgba(colors[current_group], 0.15),
                        linewidth=1, linestyle='solid', zorder=1,
                        boxstyle="round,pad=0.2"
                    )
                )
            
            if original_idx in duplicated_original_indices:
                current_group = original_idx
                group_start_idx = idx
            else:
                current_group = None
        
        last_original_idx = original_idx
        
        # Calculate x offset for indentation - duplicates are indented
        x_offset = 0
        if dup_idx > 0:
            x_offset = 0.3
        
        # Determine color and label based on whether this is a duplicated layer
        if dup_idx > 0:
            # Duplicate layer
            facecolor = highlight_color
            edge_color = to_rgba(frame_color, 0.8)
            alpha = 0.7 + (0.3 * (dup_idx / (max(duplication_counts) if duplication_counts else 1)))
            
            # Create label with layer description if available
            if layer_descriptions and original_idx < len(layer_descriptions):
                desc = layer_descriptions[original_idx]
                label = f"Layer {original_idx}: {desc} (Copy {dup_idx})"
            else:
                label = f"Layer {original_idx} (Copy {dup_idx})"
                
            linestyle = 'dashed'
            linewidth = 1.5
            boxstyle = "round,pad=0.3"
        else:
            # Original layer
            facecolor = colors[original_idx]
            edge_color = frame_color
            alpha = 0.8
            
            # Create label with layer description if available
            if layer_descriptions and original_idx < len(layer_descriptions):
                desc = layer_descriptions[original_idx]
                label = f"Layer {original_idx}: {desc}"
            else:
                label = f"Layer {original_idx}"
                
            linestyle = 'solid'
            linewidth = 1.5
            boxstyle = "round,pad=0.3"
        
        # Draw the layer box
        rect = patches.FancyBboxPatch(
            (1 - box_width/2 + x_offset, y_position), box_width, box_height,
            edgecolor=edge_color, facecolor=facecolor, alpha=alpha,
            linewidth=linewidth, linestyle=linestyle, zorder=3,
            boxstyle=boxstyle
        )
        ax.add_patch(rect)
        
        # Add layer text
        ax.text(
            1 + x_offset, y_position + box_height/2, label,
            ha='center', va='center', fontsize=10,
            fontweight='bold' if dup_idx == 0 else 'normal',
            color=text_color
        )
    
    # Draw final group if needed
    if current_group is not None:
        # Draw group background
        group_height = (len(layer_info) - group_start_idx) * layer_spacing
        group_y = -((group_start_idx + 1) * layer_spacing)
        ax.add_patch(
            patches.FancyBboxPatch(
                (0.4, group_y - 0.25), box_width + 0.2, group_height + 0.5,
                edgecolor=to_rgba(colors[current_group], 0.7),
                facecolor=to_rgba(colors[current_group], 0.15),
                linewidth=1, linestyle='solid', zorder=1,
                boxstyle="round,pad=0.2"
            )
        )
    
    # Draw output layer
    y_position = -((len(layer_info) + 1) * layer_spacing)
    ax.add_patch(
        patches.FancyBboxPatch(
            (-0.5, y_position), box_width * 3, box_height,
            edgecolor=frame_color, facecolor=embed_color, alpha=0.8,
            linewidth=1.5, zorder=1,
            boxstyle="round,pad=0.3"
        )
    )
    ax.text(
        box_width, y_position + box_height/2,
        f"{model_name} Output Layer", 
        ha='center', va='center', fontsize=11, fontweight='bold',
        color=text_color
    )
    
    # Add arrows connecting layers
    for i in range(len(layer_info) + 1):
        y_start = -i * layer_spacing + box_height/2
        y_end = -(i+1) * layer_spacing + box_height/2
        
        # Create arrow
        ax.annotate(
            "", xy=(1, y_end), xytext=(1, y_start),
            arrowprops=dict(arrowstyle="-|>", linewidth=1.5, 
                           color=arrow_color, alpha=0.8,
                           connectionstyle="arc3,rad=0")
        )
    
    # Create a more informative legend
    legend_elements = []
    
    # Add original layer legend item
    legend_elements.append(
        patches.Patch(facecolor=colors[0], alpha=0.8, edgecolor=frame_color,
                     label="Original Layers")
    )
    
    # Add duplicated layer legend item if there are duplicates
    if duplicate_layers is not None and len(duplicate_layers) > 0:
        legend_elements.append(
            patches.Patch(facecolor=highlight_color, alpha=0.8, edgecolor=frame_color,
                         label="Duplicated Layers")
        )
    
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, 
              fancybox=True, shadow=True)
    
    # Set up the axis
    ax.set_xlim(-1, 3)
    ax.set_ylim(y_position - 1, 1)
    ax.axis('off')
    
    # Add title and subtitle
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    if duplicate_layers is not None and duplication_counts is not None:
        # Create more descriptive subtitle
        duplications_list = []
        for l, c in zip(duplicate_layers, duplication_counts):
            duplications_list.append(f"Layer {l}: {c}× duplicated")
        
        if duplications_list:
            subtitle = "Duplication pattern: " + ", ".join(duplications_list)
            plt.figtext(0.5, 0.94, subtitle, ha='center', fontsize=12, 
                       style='italic', color='#555555')
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])  # Adjust for title space
    
    # Save if path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def visualize_layer_impact(
    layer_metrics: Dict[int, Dict[str, float]],
    base_metrics: Dict[str, float],
    metric_name: str = "accuracy",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Visualize the impact of duplicating each layer on model performance.
    
    Args:
        layer_metrics: Dictionary mapping layer indices to their performance metrics
        base_metrics: Performance metrics of the base model
        metric_name: Name of the metric to visualize
        save_path: Optional path to save the figure
        figsize: Figure size as (width, height) in inches
        
    Returns:
        The matplotlib Figure object
    """
    configure_plot_style()
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract layer indices and differences
    layers = sorted(layer_metrics.keys())
    base_value = base_metrics[metric_name]
    values = [layer_metrics[layer][metric_name] for layer in layers]
    differences = [value - base_value for value in values]
    
    # Create bar chart
    bars = ax.bar(layers, differences, color='skyblue', alpha=0.7, edgecolor='black')
    
    # Add a horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Color the bars based on positive or negative impact
    for i, diff in enumerate(differences):
        if diff > 0:
            bars[i].set_color('green')
            bars[i].set_alpha(0.6)
        elif diff < 0:
            bars[i].set_color('red')
            bars[i].set_alpha(0.6)
    
    # Annotate bars with values
    for i, diff in enumerate(differences):
        text_color = 'black'
        text_position = diff + 0.001 if diff >= 0 else diff - 0.003
        annotation = f"{diff:.3f}"
        ax.annotate(annotation, xy=(layers[i], text_position),
                   xytext=(0, 5 if diff >= 0 else -12),
                   textcoords="offset points", ha='center', va='bottom',
                   color=text_color, fontsize=8)
    
    # Set labels and title
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel(f'Change in {metric_name.capitalize()}', fontsize=12)
    ax.set_title(f'Impact of Layer Duplication on {metric_name.capitalize()}', fontsize=14)
    
    # Set x-ticks to be integer layer indices
    ax.set_xticks(layers)
    ax.set_xticklabels([str(layer) for layer in layers])
    
    # Add a note about base model performance
    ax.text(0.02, 0.98, f'Base Model {metric_name.capitalize()}: {base_value:.4f}',
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # Add grid for readability
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def visualize_multi_metric_comparison(
    layer_metrics: Dict[int, Dict[str, float]],
    base_metrics: Dict[str, float],
    metrics: List[str] = ["accuracy", "f1", "precision", "recall"],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Visualize multiple metrics across layers in a single figure.
    
    Args:
        layer_metrics: Dictionary mapping layer indices to their performance metrics
        base_metrics: Performance metrics of the base model
        metrics: List of metric names to visualize
        save_path: Optional path to save the figure
        figsize: Figure size as (width, height) in inches
        
    Returns:
        The matplotlib Figure object
    """
    configure_plot_style()
    
    # Set up the figure
    fig, axes = plt.subplots(len(metrics), 1, figsize=figsize, sharex=True)
    if len(metrics) == 1:
        axes = [axes]
        
    # Extract layer indices
    layers = sorted(layer_metrics.keys())
    
    # Color palette
    colors = sns.color_palette("bright", len(metrics))
    
    # Create plots for each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Extract values and differences
        base_value = base_metrics[metric]
        values = [layer_metrics[layer][metric] for layer in layers]
        differences = [value - base_value for value in values]
        
        # Create bar chart
        bars = ax.bar(layers, differences, color=colors[i], alpha=0.7, edgecolor='black')
        
        # Add a horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Color the bars based on positive or negative impact
        for j, diff in enumerate(differences):
            if diff > 0:
                bars[j].set_color(colors[i])
                bars[j].set_alpha(0.8)
            elif diff < 0:
                bars[j].set_color('grey')
                bars[j].set_alpha(0.6)
        
        # Set labels
        ax.set_ylabel(f'Δ {metric}', fontsize=10)
        ax.text(0.02, 0.85, f'Base: {base_value:.4f}',
               transform=ax.transAxes, fontsize=8,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # Add grid for readability
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add metric name as title
        ax.set_title(f'{metric.capitalize()}', fontsize=10, loc='right')
    
    # Set x-label on the bottom axis only
    axes[-1].set_xlabel('Layer Index', fontsize=12)
    axes[-1].set_xticks(layers)
    
    # Add a main title
    plt.suptitle('Impact of Layer Duplication on Model Performance', fontsize=14, y=0.98)
    
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.3)
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def visualize_heatmap_layer_combinations(
    combinations_metrics: Dict[Tuple[int, int], Dict[str, float]],
    base_metrics: Dict[str, float],
    metric_name: str = "accuracy",
    max_layer: int = 11,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Visualize the impact of duplicating combinations of layers as a heatmap.
    
    Args:
        combinations_metrics: Dictionary mapping (layer1, layer2) to their performance metrics
        base_metrics: Performance metrics of the base model
        metric_name: Name of the metric to visualize
        max_layer: Maximum layer index to include in the heatmap
        save_path: Optional path to save the figure
        figsize: Figure size as (width, height) in inches
        
    Returns:
        The matplotlib Figure object
    """
    configure_plot_style()
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create a matrix for the heatmap
    matrix = np.zeros((max_layer + 1, max_layer + 1))
    
    # Fill the matrix with differences from base metric
    base_value = base_metrics[metric_name]
    for (layer1, layer2), metrics in combinations_metrics.items():
        if layer1 <= max_layer and layer2 <= max_layer:
            matrix[layer1, layer2] = metrics[metric_name] - base_value
    
    # Create a diverging colormap centered at 0
    cmap = LinearSegmentedColormap.from_list("RdBu_alpha", 
                                            [(0.0, "red"), 
                                            (0.5, "white"), 
                                            (1.0, "blue")])
    
    # Determine the maximum absolute difference for symmetric color scale
    max_diff = max(abs(np.max(matrix)), abs(np.min(matrix)))
    
    # Create the heatmap
    sns.heatmap(matrix, cmap=cmap, center=0, vmin=-max_diff, vmax=max_diff,
               annot=True, fmt=".3f", linewidths=0.5, ax=ax,
               cbar_kws={"label": f"Change in {metric_name.capitalize()}"})
    
    # Set labels and title
    ax.set_xlabel('First Layer Duplicated', fontsize=12)
    ax.set_ylabel('Second Layer Duplicated', fontsize=12)
    ax.set_title(f'Impact of Duplicating Layer Combinations on {metric_name.capitalize()}', 
                fontsize=14)
    
    # Add a note about base model performance
    ax.text(0.02, 1.02, f'Base Model {metric_name.capitalize()}: {base_value:.4f}',
           transform=ax.transAxes, fontsize=10,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Load data from CSV (assuming it exists from layer_check.py output)
    try:
        df = pd.read_csv("layer_replication_results.csv")
        
        # Extract metrics for each layer
        layer_metrics = {
            row["layer_idx"]: {
                "accuracy": row["accuracy"],
                "f1": row["f1"],
                "precision": row["precision"],
                "recall": row["recall"]
            }
            for _, row in df.iterrows()
        }
        
        # Get the first row data as base model metrics (this assumes base model results were also saved)
        base_metrics = {
            "accuracy": df.iloc[0]["accuracy"] - df.iloc[0]["acc_diff"],
            "f1": df.iloc[0]["f1"] - df.iloc[0]["f1_diff"],
            "precision": df.iloc[0]["precision"] - df.iloc[0]["precision_diff"],
            "recall": df.iloc[0]["recall"] - df.iloc[0]["recall_diff"]
        }
        
        # Visualize model architecture - example with layers 3 and 8 duplicated
        visualize_model_architecture(
            duplicate_layers=[3, 8],
            duplication_counts=[1, 2],
            save_path="visualizations/frankenmodel_architecture.png"
        )
        
        # Visualize impact on accuracy
        visualize_layer_impact(
            layer_metrics=layer_metrics,
            base_metrics=base_metrics,
            metric_name="accuracy",
            save_path="visualizations/layer_impact_accuracy.png"
        )
        
        # Visualize multi-metric comparison
        visualize_multi_metric_comparison(
            layer_metrics=layer_metrics,
            base_metrics=base_metrics,
            save_path="visualizations/multi_metric_comparison.png"
        )
        
        print("Example visualizations generated!")
    except Exception as e:
        print(f"Example generation failed: {e}")
        print("Run layer_check.py first to generate the necessary data.") 