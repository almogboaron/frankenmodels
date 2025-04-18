#!/usr/bin/env python3
"""
Visualization script for layer performance in Frankenmodels.

This script generates publication-quality visualizations of layer performance metrics
from experimental results in the layer_replication_results.csv file.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Optional, Tuple, List

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


def load_results_data(input_file: str) -> Optional[pd.DataFrame]:
    """Load experimental results from CSV file."""
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return None
    
    try:
        df = pd.read_csv(input_file)
        print(f"Successfully loaded data from {input_file}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def create_output_directory(output_dir: str) -> None:
    """Create output directory if it doesn't exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")


def visualize_layer_performance(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Create a visualization of performance metrics for each layer.
    
    Args:
        df: DataFrame containing layer performance data
        save_path: Optional path to save the figure
        figsize: Figure size as (width, height) in inches
        
    Returns:
        The matplotlib Figure object
    """
    configure_plot_style()
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up x-axis with layer indices
    x = df['layer_idx'].values
    
    # Plot the metrics
    metrics = ['accuracy', 'f1', 'precision', 'recall']
    colors = sns.color_palette("colorblind", len(metrics))
    
    for i, metric in enumerate(metrics):
        ax.plot(x, df[metric], marker='o', linestyle='-', color=colors[i], 
                linewidth=2, markersize=8, label=metric.capitalize())
    
    # Set up the plot styling
    ax.set_xticks(x)
    ax.set_xlabel('Layer Index', fontweight='bold')
    ax.set_ylabel('Performance Score', fontweight='bold')
    ax.set_title('Performance Metrics by Layer in Frankenmodel', fontsize=16, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='best', frameon=True, fancybox=True, framealpha=0.8)
    
    # Add value labels on the plot
    for i, metric in enumerate(metrics):
        for j, value in enumerate(df[metric]):
            ax.annotate(f'{value:.3f}', 
                        (x[j], df[metric][j]),
                        textcoords="offset points", 
                        xytext=(0, 10 if j % 2 == 0 else -15), 
                        ha='center',
                        fontsize=8,
                        alpha=0.8)
    
    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved layer performance visualization to {save_path}")
    
    return fig


def visualize_performance_improvement(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Create a visualization of performance improvement for each layer.
    
    Args:
        df: DataFrame containing layer performance data
        save_path: Optional path to save the figure
        figsize: Figure size as (width, height) in inches
        
    Returns:
        The matplotlib Figure object
    """
    configure_plot_style()
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up x-axis with layer indices
    x = df['layer_idx'].values
    
    # Plot the metrics improvements
    diff_columns = ['acc_diff', 'f1_diff', 'precision_diff', 'recall_diff']
    metric_names = ['Accuracy', 'F1', 'Precision', 'Recall']
    colors = sns.color_palette("colorblind", len(diff_columns))
    
    # Find the min and max values for better y-axis limits
    all_diffs = np.concatenate([df[col].values for col in diff_columns])
    min_diff, max_diff = np.min(all_diffs), np.max(all_diffs)
    # Add some padding
    y_min = min_diff - 0.002
    y_max = max_diff + 0.002
    
    # Plot a horizontal line at y=0
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    for i, (col, name) in enumerate(zip(diff_columns, metric_names)):
        ax.plot(x, df[col], marker='o', linestyle='-', color=colors[i], 
                linewidth=2, markersize=8, label=name)
        
        # Highlight positive and negative regions with light shading
        ax.fill_between(x, df[col], 0, where=(df[col] > 0), 
                       color=colors[i], alpha=0.2)
        ax.fill_between(x, df[col], 0, where=(df[col] < 0), 
                       color=colors[i], alpha=0.1)
    
    # Set up the plot styling
    ax.set_xticks(x)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Layer Index', fontweight='bold')
    ax.set_ylabel('Performance Difference from Base Model', fontweight='bold')
    ax.set_title('Performance Improvement by Layer in Frankenmodel', fontsize=16, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='best', frameon=True, fancybox=True, framealpha=0.8)
    
    # Add value labels on the plot
    for i, col in enumerate(diff_columns):
        for j, value in enumerate(df[col]):
            ax.annotate(f'{value:.4f}', 
                        (x[j], df[col][j]),
                        textcoords="offset points", 
                        xytext=(0, 10 if j % 2 == 0 else -15), 
                        ha='center',
                        fontsize=8,
                        alpha=0.8)
    
    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved performance improvement visualization to {save_path}")
    
    return fig


def visualize_best_layers(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    top_n: int = 3
) -> plt.Figure:
    """
    Create a bar chart showing the top performing layers.
    
    Args:
        df: DataFrame containing layer performance data
        save_path: Optional path to save the figure
        figsize: Figure size as (width, height) in inches
        top_n: Number of top layers to display
        
    Returns:
        The matplotlib Figure object
    """
    configure_plot_style()
    
    # Create a figure with 2x2 subplots for each metric
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    axs = axs.flatten()
    
    metrics = ['acc_diff', 'f1_diff', 'precision_diff', 'recall_diff']
    metric_names = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    colors = sns.color_palette("colorblind", 4)
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        # Sort the dataframe by the metric
        sorted_df = df.sort_values(by=metric, ascending=False)
        
        # Get the top N layers
        top_layers = sorted_df.head(top_n)
        
        # Create the bar chart
        bars = axs[i].bar(top_layers['layer_idx'], top_layers[metric], 
                          color=colors[i], alpha=0.8)
        
        # Add value labels on the bars
        for bar in bars:
            height = bar.get_height()
            axs[i].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{height:.4f}',
                        ha='center', va='bottom', fontsize=9)
        
        # Set up the subplot styling
        axs[i].set_title(f'Top {top_n} Layers by {name}', fontsize=12, pad=20)
        axs[i].set_xlabel('Layer Index')
        axs[i].set_ylabel(f'Improvement in {name}')
        axs[i].grid(True, linestyle='--', alpha=0.3)
    
    # Increase the vertical space between subplots and adjust the layout
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # Increase the top margin and adjust the suptitle position
    plt.suptitle('Best Performing Layers in Frankenmodel', fontsize=16, fontweight='bold', y=1.05)
    
    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved best layers visualization to {save_path}")
    
    return fig


def main():
    """Main function to generate visualizations."""
    # Define file paths
    input_file = "layer_replication_results.csv"
    output_dir = "visualization_results"
    
    # Create output directory
    create_output_directory(output_dir)
    
    # Load data
    df = load_results_data(input_file)
    if df is None:
        return
    
    # Generate visualizations
    visualize_layer_performance(
        df=df,
        save_path=os.path.join(output_dir, "layer_performance_metrics.png")
    )
    
    visualize_performance_improvement(
        df=df,
        save_path=os.path.join(output_dir, "layer_performance_improvement.png")
    )
    
    visualize_best_layers(
        df=df,
        save_path=os.path.join(output_dir, "best_performing_layers.png")
    )
    
    plt.close('all')
    print("All layer performance visualizations completed!")


if __name__ == "__main__":
    main() 