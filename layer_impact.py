#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Configure plot style for publication-quality
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

# Sample data based on the project results
# This could be replaced with actual data from layer_replication_results.csv
layer_indices = list(range(12))  # 0-11 for BERT base
accuracy_diff = [
    -0.0011,  # Layer 0
    -0.0005,  # Layer 1
    0.0003,   # Layer 2
    0.0010,   # Layer 3
    0.0021,   # Layer 4
    0.0035,   # Layer 5
    0.0056,   # Layer 6 (highest impact)
    0.0034,   # Layer 7
    0.0020,   # Layer 8
    0.0012,   # Layer 9
    -0.0002,  # Layer 10
    -0.0010   # Layer 11
]

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the bars
bars = ax.bar(layer_indices, accuracy_diff, color='lightblue', edgecolor='black', width=0.6)

# Color the bars based on positive/negative impact
for i, bar in enumerate(bars):
    if accuracy_diff[i] > 0:
        bar.set_color('#4CAF50')  # Green for positive impact
    else:
        bar.set_color('#F44336')  # Red for negative impact
    
    # Add value labels on top of bars
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., 
            height + 0.0008 if height > 0 else height - 0.0015,
            f'{height:.4f}',
            ha='center', va='bottom' if height > 0 else 'top',
            fontsize=9)

# Highlight the best layer (Layer 6)
bars[6].set_color('#2196F3')  # Blue highlight
bars[6].set_edgecolor('black')
bars[6].set_linewidth(1.5)

# Add a horizontal line at y=0
ax.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=0.8)

# Add grid
ax.grid(True, axis='y', linestyle='--', alpha=0.3)

# Set labels and title
ax.set_xlabel('Layer Index')
ax.set_ylabel('Change in Accuracy')
ax.set_title('Impact of Layer Duplication on Model Performance')

# Set x-axis ticks
ax.set_xticks(layer_indices)
ax.set_xticklabels([str(i) for i in layer_indices])

# Set y-axis limits with some padding
y_max = max(accuracy_diff) * 1.2
y_min = min(accuracy_diff) * 1.2
ax.set_ylim(y_min, y_max)

# Add a text annotation for the best layer
plt.annotate('Best Layer\nfor Duplication',
             xy=(6, accuracy_diff[6]),
             xytext=(6, accuracy_diff[6] + 0.002),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
             ha='center', va='bottom',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))

# Improve layout
plt.tight_layout()

# Save the figure
plt.savefig('layer_impact.png', dpi=300, bbox_inches='tight')
print("Figure saved as 'layer_impact.png'")

# Show the plot
plt.show() 