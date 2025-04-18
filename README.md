# FrankenModels: Boosting Transformer Performance Through Targeted Layer Replication

This repository contains the code and LaTeX source for our academic paper on FrankenModels. FrankenModels are a novel approach to enhancing transformer model performance through targeted layer duplication, achieving statistically significant improvements with minimal computational overhead.

## Project Overview

The repository includes:
- Python code for creating and evaluating FrankenModels
- Visualization scripts for analyzing performance
- LaTeX source for the academic paper
- Supporting figures and data

## Paper Abstract

Transformer-based models have achieved remarkable success across various natural language processing tasks but often require significant computational resources to scale up performance through increased model size. In this work, we introduce FrankenModels, a novel approach that selectively replicates specific transformer layers to enhance model performance with minimal additional parameters. We conduct an empirical study on BERT models fine-tuned for sentiment analysis, demonstrating that strategic layer duplication can lead to statistically significant improvements in accuracy, F1 score, precision, and recall. Our approach achieves these gains while maintaining a lower computational footprint compared to training larger models from scratch. We analyze the performance impacts of duplicating different layers and present comprehensive evaluations of the memory, inference time, and parameter count trade-offs. Our findings suggest that FrankenModels can serve as an efficient alternative to full model scaling when computational resources are limited.

## Visualizations

The repository includes several visualization scripts that generate publication-quality figures:

1. **Layer Impact Visualization** (`layer_impact.py`): Shows the performance impact of duplicating each individual layer in BERT, demonstrating that middle layers (particularly layer 6) yield the most significant improvements.

2. **Metric Comparison Visualization** (`metric_comparison.py`): Compares the performance of base models and FrankenModels across multiple evaluation seeds, showing consistent improvements in accuracy, F1 score, precision, and recall.

3. **Tradeoff Visualization** (`tradeoff_visualization.py`): Visualizes the performance-efficiency tradeoffs, comparing computational costs (parameters, memory, inference time) against performance gains.

## Compiling the LaTeX Paper

### Prerequisites

- A LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
- The IEEE conference class files (for `frankenmodels_article.tex`)
- Basic LaTeX packages (for `frankenmodels_article_simple.tex`)

### Compilation Steps

#### IEEE Version

1. Make sure all visualization PNG files are in the same directory as the `.tex` file.

2. Compile the document using pdfLaTeX:
   ```bash
   pdflatex frankenmodels_article.tex
   ```

3. Generate the bibliography (if BibTeX is used):
   ```bash
   bibtex frankenmodels_article
   ```

4. Run pdfLaTeX twice more to incorporate the bibliography and references:
   ```bash
   pdflatex frankenmodels_article.tex
   pdflatex frankenmodels_article.tex
   ```

#### Simple Version (No IEEE Style Required)

For a version that doesn't require IEEE style files:

```bash
pdflatex frankenmodels_article_simple.tex
```

The final PDF will be named `frankenmodels_article_simple.pdf`.

## Project Structure

- `src/`: Source code for FrankenModel implementation and evaluation
  - `utils.py`: Utility functions for model loading and evaluation
  - `layer_check.py`: Analysis of individual layer impact on performance
  - `tradeoffs.py`: Performance and efficiency tradeoff evaluation
  - `Evaluate_metrics_and_significance.py`: Statistical significance testing
  - `visualize_*.py`: Built-in visualization scripts for experimental results
- Visualization scripts:
  - `layer_impact.py`: Generates the layer impact visualization
  - `metric_comparison.py`: Generates the metric comparison visualization
  - `tradeoff_visualization.py`: Generates the tradeoff visualization
- LaTeX papers:
  - `frankenmodels_article.tex`: IEEE conference format paper
  - `frankenmodels_article_simple.tex`: Simple format paper (no IEEE style required)

## Key Findings

- Duplicating certain layers (particularly middle layers) in BERT models leads to statistically significant performance improvements
- Layer 6 showed the highest performance improvement when duplicated
- FrankenModels provide ~0.5-0.6% accuracy gains with only ~7% increase in parameters
- The approach represents a middle ground between full model scaling and parameter-efficient techniques

## Contact

For questions or further information, please contact the repository authors. 