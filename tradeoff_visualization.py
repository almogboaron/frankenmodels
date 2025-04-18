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

# Sample data based on the results from tradeoffs.py
# Values from the paper tables
tradeoff_data = {
    'Model': ['Base Model', 'FrankenModel'],
    'Parameters (M)': [109.5, 117.1],
    'Memory (MB)': [435.2, 452.7],
    'Inference (ms/batch)': [18.3, 20.1],
    'Accuracy': [0.9231, 0.9287],
    'F1': [0.9231, 0.9287],
    'Precision': [0.9230, 0.9290],
    'Recall': [0.9231, 0.9287]
}

# Convert to DataFrame
df = pd.DataFrame(tradeoff_data)

# Calculate percentage increases
increases = {
    'Parameters': (df['Parameters (M)'][1] - df['Parameters (M)'][0]) / df['Parameters (M)'][0] * 100,
    'Memory': (df['Memory (MB)'][1] - df['Memory (MB)'][0]) / df['Memory (MB)'][0] * 100,
    'Inference': (df['Inference (ms/batch)'][1] - df['Inference (ms/batch)'][0]) / df['Inference (ms/batch)'][0] * 100,
    'Accuracy': (df['Accuracy'][1] - df['Accuracy'][0]) / df['Accuracy'][0] * 100,
}

# Create a figure with two subplots
fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121)  # Left subplot for bar chart
ax2 = fig.add_subplot(122, polar=True)  # Right subplot for radar chart

# Colors
colors = ['#3498db', '#e74c3c']  # Blue for base, Red for franken

# First subplot: Bar plot comparing absolute values
metrics = ['Parameters (M)', 'Memory (MB)', 'Inference (ms/batch)']
x = np.arange(len(metrics))
width = 0.35

# Plot the bars for base model
base_bars = ax1.bar(x - width/2, df[metrics].iloc[0], width, label='Base Model', color=colors[0])
# Plot the bars for frankenmodel
franken_bars = ax1.bar(x + width/2, df[metrics].iloc[1], width, label='FrankenModel', color=colors[1])

# Add labels and formatting
ax1.set_title('Computational Resource Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics)
ax1.legend()

# Add value labels on the bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom')

add_value_labels(base_bars)
add_value_labels(franken_bars)

# Second subplot: Performance vs. Efficiency tradeoff
# Create a radar chart (polar plot) to show the tradeoffs
metrics_radar = ['Accuracy\nImprovement', 'Parameter\nIncrease', 'Memory\nIncrease', 'Inference Time\nIncrease']
values_radar = [
    increases['Accuracy'], 
    increases['Parameters'], 
    increases['Memory'], 
    increases['Inference']
]

# Number of variables
N = len(metrics_radar)

# Normalize to make the scales more comparable
max_val = max(values_radar[1:])  # Exclude accuracy as it's typically a smaller percentage
accuracy_scaled = values_radar[0] * (max_val / values_radar[0] * 0.5)  # Scale accuracy for visibility
values_radar_norm = [accuracy_scaled] + values_radar[1:]

# Compute angle for each axis
angles = np.linspace(0, 2*np.pi, len(metrics_radar), endpoint=False).tolist()

# Close the loop for the plot
values_radar_norm = values_radar_norm + [values_radar_norm[0]]
angles = angles + [angles[0]]

# Plot the radar chart
ax2.plot(angles, values_radar_norm, linewidth=1, linestyle='solid', color='r')
ax2.fill(angles, values_radar_norm, color='r', alpha=0.25)

# Set the angle of the first axis
ax2.set_theta_offset(np.pi / 2)
ax2.set_theta_direction(-1)  # Go clockwise

# Set labels for each axis
ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(metrics_radar)

# Add actual percentage values as text annotations
for i, (angle, value, norm_value, metric) in enumerate(zip(angles[:-1], values_radar, values_radar_norm[:-1], metrics_radar)):
    if metric == 'Accuracy\nImprovement':
        color = 'green'
        offset = 1.0
    else:
        color = '#e74c3c'
        offset = max_val * 0.1
    
    # Convert from polar to cartesian coordinates for text placement
    x = norm_value * np.cos(angle)
    y = norm_value * np.sin(angle)
    # Add a bit of offset
    x_text = (norm_value + offset) * np.cos(angle)
    y_text = (norm_value + offset) * np.sin(angle)
    
    ax2.text(angle, norm_value + offset, f'+{value:.2f}%', 
             ha='center', va='center', color=color, fontweight='bold')

# Add title to the radar plot
ax2.set_title('Performance-Efficiency Tradeoff')

# Add a helpful note
note_text = "Note: Accuracy improvement is scaled for visibility\nin comparison to efficiency metrics"
fig.text(0.75, 0.02, note_text, ha='center', va='bottom', fontsize=9, style='italic', 
         bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.3'))

# Add an overall title
fig.suptitle('FrankenModel Tradeoffs: Performance Gains vs. Resource Costs', fontsize=16)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.88)

# Save the figure
plt.savefig('tradeoff_visualization.png', dpi=300, bbox_inches='tight')
print("Figure saved as 'tradeoff_visualization.png'")

# Show the plot
plt.show() 