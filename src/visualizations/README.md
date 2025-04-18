# Frankenmodel Visualizations

This module provides tools for creating publication-quality visualizations of Frankenmodel architectures and performance metrics.

## Features

- **Model Architecture Visualization**: Clearly illustrates the structure of Frankenmodels, showing which layers have been duplicated
- **Performance Impact Visualization**: Shows how duplicating specific layers affects model performance metrics
- **Multi-metric Comparison**: Compares the impact of layer duplication across multiple performance metrics
- **Publication-ready Graphics**: All visualizations are generated in high resolution and follow academic publication standards

## Usage

### Basic Usage

After running experiments with `layer_check.py`, you can generate visualizations with:

```
python src/visualize_results.py --input layer_replication_results.csv --output_dir visualizations
```

### Advanced Usage

To visualize a specific Frankenmodel architecture (e.g., with layers 3 and 8 duplicated):

```
python src/visualize_results.py --layers "3:1,8:2"
```

This will produce visualizations where layer 3 is duplicated once and layer 8 is duplicated twice.

## Visualization Types

### 1. Model Architecture

A graphical representation of the model's layers, showing:
- Original layers from the base model
- Duplicated layers with clear highlighting
- Layer connections and information flow

### 2. Layer Impact

For each metric (accuracy, F1, precision, recall), these visualizations show:
- The impact of duplicating each layer individually
- Positive and negative changes relative to the base model
- Exact numerical differences for precise interpretation

### 3. Multi-metric Comparison

A consolidated view that shows how layer duplication affects multiple metrics:
- Aligned bar charts for each metric
- Consistent scales for easy comparison
- Color coding for positive and negative impacts

## Customization

The visualization module provides extensive customization options:

```python
from visualizations.model_architecture import visualize_model_architecture

# Customize visualization appearance
visualize_model_architecture(
    duplicate_layers=[3, 8],
    duplication_counts=[1, 2],
    base_layers=12,
    model_name="BERT",
    figsize=(12, 8),
    highlight_color="#ffcccc",
    style="academic",  # or "modern"
    save_path="custom_visualization.png"
)
```

## Integration with Experiments

These visualizations are designed to work seamlessly with the experimental pipeline:

1. Run experiments with `layer_check.py` to generate CSV results
2. Use `visualize_results.py` to create visualizations from those results
3. Include the high-quality figures in papers or presentations

## Requirements

- matplotlib
- seaborn
- numpy
- pandas

These are standard data science libraries that should already be in your environment if you're running the Frankenmodel experiments. 