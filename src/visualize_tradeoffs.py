#!/usr/bin/env python3
"""
Visualization script for Frankenmodels tradeoffs.

This script generates publication-quality visualizations of the tradeoffs
between base models and Frankenmodels in terms of parameters, inference time,
and memory requirements.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional, Tuple

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


def visualize_inference_time_comparison(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Create a bar chart comparing inference time between model types.
    
    Args:
        df: DataFrame containing tradeoffs data
        save_path: Optional path to save the figure
        figsize: Figure size as (width, height) in inches
        
    Returns:
        The matplotlib Figure object
    """
    configure_plot_style()
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract relevant data
    models = df['model_type'].tolist()
    inference_times = df['avg_inference_time_sec'].tolist()
    
    # Set up bar colors
    colors = sns.color_palette("colorblind", 2)
    
    # Create the bar chart
    bars = ax.bar(models, inference_times, color=colors, width=0.6)
    
    # Calculate percentage increase
    base_time = inference_times[0]
    franken_time = inference_times[1]
    pct_increase = ((franken_time - base_time) / base_time) * 100
    
    # Add value labels on the bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height * 1.01,
                f'{height:.4f}s',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add comparison annotation
    ax.annotate(f'+{pct_increase:.1f}%', 
                xy=(1, franken_time), 
                xytext=(1.3, (franken_time + base_time) / 2),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Set up the plot styling
    ax.set_ylabel('Average Inference Time (seconds)', fontweight='bold')
    ax.set_title('Inference Time Comparison', fontsize=16, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Adjust y-axis to start slightly below zero
    y_min = 0
    y_max = max(inference_times) * 1.2
    ax.set_ylim(y_min, y_max)
    
    # Add a light background color to differentiate the plot area
    ax.set_facecolor('#f8f9fa')
    
    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved inference time comparison to {save_path}")
    
    return fig


def visualize_memory_comparison(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Create a bar chart comparing memory usage between model types.
    
    Args:
        df: DataFrame containing tradeoffs data
        save_path: Optional path to save the figure
        figsize: Figure size as (width, height) in inches
        
    Returns:
        The matplotlib Figure object
    """
    configure_plot_style()
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract relevant data
    models = df['model_type'].tolist()
    memory_usage = df['max_memory_MB'].tolist()
    
    # Set up bar colors
    colors = sns.color_palette("colorblind", 2)
    
    # Create the bar chart
    bars = ax.bar(models, memory_usage, color=colors, width=0.6)
    
    # Calculate percentage increase (if any)
    base_memory = memory_usage[0]
    franken_memory = memory_usage[1]
    pct_change = ((franken_memory - base_memory) / base_memory) * 100
    
    # Add value labels on the bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height * 1.01,
                f'{height:.2f} MB',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add comparison annotation if there's a difference
    if abs(pct_change) > 0.01:
        change_symbol = '+' if pct_change > 0 else ''
        ax.annotate(f'{change_symbol}{pct_change:.1f}%', 
                    xy=(1, franken_memory), 
                    xytext=(1.3, (franken_memory + base_memory) / 2),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    ha='center', va='center', fontsize=12, fontweight='bold')
    else:
        # If no change, indicate equality
        ax.annotate('No change', 
                    xy=(0.5, max(memory_usage) * 1.1),
                    ha='center', fontsize=12, fontweight='bold')
    
    # Set up the plot styling
    ax.set_ylabel('Maximum Memory Usage (MB)', fontweight='bold')
    ax.set_title('Memory Usage Comparison', fontsize=16, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Adjust y-axis to start slightly below zero for better visual
    y_min = min(memory_usage) * 0.9
    y_max = max(memory_usage) * 1.2
    ax.set_ylim(y_min, y_max)
    
    # Add a light background color to differentiate the plot area
    ax.set_facecolor('#f8f9fa')
    
    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved memory usage comparison to {save_path}")
    
    return fig


def visualize_combined_tradeoffs(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Create a combined visualization of all tradeoffs.
    
    Args:
        df: DataFrame containing tradeoffs data
        save_path: Optional path to save the figure
        figsize: Figure size as (width, height) in inches
        
    Returns:
        The matplotlib Figure object
    """
    configure_plot_style()
    
    # Create the figure with subplots
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    
    # Extract data
    models = df['model_type'].tolist()
    params = df['num_parameters'].tolist()
    inference_times = df['avg_inference_time_sec'].tolist()
    memory_usage = df['max_memory_MB'].tolist()
    
    # Set up colors
    colors = sns.color_palette("colorblind", 2)
    
    # 1. Parameters subplot
    axs[0].bar(models, params, color=colors, width=0.6)
    axs[0].set_ylabel('Number of Parameters', fontweight='bold')
    axs[0].set_title('Model Size', fontsize=12)
    axs[0].grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Format y-axis for parameters (in millions)
    params_in_millions = [p / 1_000_000 for p in params]
    for i, model in enumerate(models):
        axs[0].text(i, params[i] * 0.5, f'{params_in_millions[i]:.1f}M',
                   ha='center', color='white', fontsize=10, fontweight='bold')
    
    # 2. Inference time subplot
    bars_time = axs[1].bar(models, inference_times, color=colors, width=0.6)
    axs[1].set_ylabel('Avg Inference Time (sec)', fontweight='bold')
    axs[1].set_title('Inference Speed', fontsize=12)
    axs[1].grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Add value labels for inference time
    for i, bar in enumerate(bars_time):
        height = bar.get_height()
        axs[1].text(bar.get_x() + bar.get_width()/2., height * 0.5,
                    f'{height:.3f}s',
                    ha='center', va='center', color='white', fontsize=10, fontweight='bold')
    
    # Calculate percentage increase for inference time
    base_time = inference_times[0]
    franken_time = inference_times[1]
    pct_increase = ((franken_time - base_time) / base_time) * 100
    axs[1].text(0.5, max(inference_times) * 1.1, f'+{pct_increase:.1f}%',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 3. Memory usage subplot
    bars_mem = axs[2].bar(models, memory_usage, color=colors, width=0.6)
    axs[2].set_ylabel('Max Memory Usage (MB)', fontweight='bold')
    axs[2].set_title('Memory Efficiency', fontsize=12)
    axs[2].grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Add value labels for memory
    for i, bar in enumerate(bars_mem):
        height = bar.get_height()
        axs[2].text(bar.get_x() + bar.get_width()/2., height * 0.5,
                    f'{height:.0f} MB',
                    ha='center', va='center', color='white', fontsize=10, fontweight='bold')
    
    # Apply consistent styling to all subplots
    for ax in axs:
        ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    plt.suptitle('Frankenmodel Performance Tradeoffs', fontsize=16, fontweight='bold', y=1.05)
    
    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved combined tradeoffs visualization to {save_path}")
    
    return fig


def main():
    """Main function to generate visualizations."""
    # Define file paths
    input_file = "tradeoffs_results.csv"
    output_dir = "visualization_results"
    
    # Create output directory
    create_output_directory(output_dir)
    
    # Load data
    df = load_results_data(input_file)
    if df is None:
        return
    
    # Generate visualizations
    visualize_inference_time_comparison(
        df=df,
        save_path=os.path.join(output_dir, "inference_time_comparison.png")
    )
    
    visualize_memory_comparison(
        df=df,
        save_path=os.path.join(output_dir, "memory_usage_comparison.png")
    )
    
    visualize_combined_tradeoffs(
        df=df,
        save_path=os.path.join(output_dir, "frankenmodel_tradeoffs.png")
    )
    
    plt.close('all')
    print("All tradeoff visualizations completed!")


if __name__ == "__main__":
    main() 