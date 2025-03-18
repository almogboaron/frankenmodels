import argparse
import torch
import copy
import json
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import BertForSequenceClassification, BertTokenizer, default_data_collator
from  utils import tokenize_function, evaluate_model, get_dataloader

def load_tokenized_dataset(tokenizer, split="validation", max_length=128):
    dataset = load_dataset("glue", "sst2")
    print("Original columns:", dataset[split].column_names)
    tokenized_dataset = dataset[split].map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=[col for col in dataset[split].column_names if col not in ["label"]]
    )
    print("Columns after mapping:", tokenized_dataset.column_names)
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    sample = tokenized_dataset[0]
    print("Sample keys after set_format:", list(sample.keys()))
    return tokenized_dataset

def load_base_model(device, model_dir="./results/bert_sst2"):
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    return model

def load_frankenmodel(device, model_dir="./results/bert_sst2", duplicate_layers=None, duplication_counts=None):
    """
    Loads the base model and creates a Frankenmodel by duplicating specified encoder layers.
    
    Args:
        device: the torch device.
        model_dir: directory to load the fine-tuned base model.
        duplicate_layers: a list of integers specifying which layer indices (0-indexed) to duplicate.
                          If None, duplicates all layers.
        duplication_counts: a list of integers specifying how many times to duplicate each corresponding layer.
                            Must be the same length as duplicate_layers. If None, duplicates once for each.
                            
    Returns:
        The modified model with duplicated layers and an updated configuration.
    """
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    # Freeze parameters to simulate no extra training cost.
    for param in model.parameters():
        param.requires_grad = False
    
    original_layers = model.bert.encoder.layer
    num_original_layers = len(original_layers)
    
    # If no specific layers provided, duplicate all layers once.
    if duplicate_layers is None:
        duplicate_layers = list(range(num_original_layers))
    if duplication_counts is None:
        duplication_counts = [1] * len(duplicate_layers)
    
    # Ensure the lists have the same length.
    assert len(duplicate_layers) == len(duplication_counts), "duplicate_layers and duplication_counts must be the same length"
    
    new_layers = torch.nn.ModuleList()
    for i, layer in enumerate(original_layers):
        # Always add the original layer.
        new_layers.append(layer)
        # If this layer index is in duplicate_layers, add the specified duplicates.
        if i in duplicate_layers:
            idx = duplicate_layers.index(i)
            count = duplication_counts[idx]
            for _ in range(count):
                duplicate_layer = copy.deepcopy(layer)
                new_layers.append(duplicate_layer)
                
    model.bert.encoder.layer = new_layers
    # Update configuration to reflect the new number of layers.
    model.config.num_hidden_layers = len(new_layers)
    return model

def plot_all_metrics(results, save_path="all_metrics_comparison.png"):
    """
    Plots all evaluation metrics (accuracy, f1, precision, recall) for each model in a grouped bar chart,
    and saves the plot to the specified file.
    """
    metrics = ["accuracy", "f1", "precision", "recall"]
    model_names = list(results.keys())  # e.g., ["Base Model", "Frankenmodel"]

    # Build data: create a list of metric values for each model in the order of the metrics.
    data = {metric: [] for metric in metrics}
    for model in model_names:
        for metric in metrics:
            data[metric].append(results[model][metric])
    
    # Setup positions for groups and bars.
    x = range(len(metrics))
    num_models = len(model_names)
    total_bar_width = 0.8
    bar_width = total_bar_width / num_models

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, model in enumerate(model_names):
        # Offset each model's bar group.
        offsets = [p + (i - num_models/2) * bar_width + bar_width/2 for p in x]
        values = [results[model][m] for m in metrics]
        ax.bar(offsets, values, bar_width, label=model)
    
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Evaluation Metrics Comparison on SST-2")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)  # Save the plot to file.
    plt.show()
    print(f"All-metrics plot saved to {save_path}")

def save_results(results, file_path="evaluation_results.json"):
    with open(file_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {file_path}")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    tokenizer = BertTokenizer.from_pretrained("./results/bert_sst2")

    tokenized_val = load_tokenized_dataset(tokenizer, split="validation")
    dataloader = get_dataloader(tokenized_val, batch_size=args.batch_size)

    # Evaluate Base Model.
    print("Evaluating Base Model...")
    base_model = load_base_model(device)
    base_metrics = evaluate_model(base_model, dataloader, device)
    print("Base Model Metrics:")
    print(base_metrics)

    # Parse the duplicate layers and counts from command-line arguments.
    duplicate_layers = None
    duplication_counts = None
    if args.duplicate_layers:
        duplicate_layers = [int(x) for x in args.duplicate_layers.split(",")]
    if args.duplication_counts:
        duplication_counts = [int(x) for x in args.duplication_counts.split(",")]
    
    # Evaluate Frankenmodel with user-specified duplications.
    print("Evaluating Frankenmodel...")
    franken_model = load_frankenmodel(device, duplicate_layers=duplicate_layers, duplication_counts=duplication_counts)
    franken_metrics = evaluate_model(franken_model, dataloader, device)
    print("Frankenmodel Metrics:")
    print(franken_metrics)

    # Save results for each model.
    results = {
        "Base Model": base_metrics,
        "Frankenmodel": franken_metrics
    }
    save_results(results, file_path=args.results_path)

    # Plot all metrics in a grouped bar chart and save the plot.
    plot_all_metrics(results, save_path=args.all_metrics_plot_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and compare models on SST-2 with multiple metrics and versatile Frankenmodel duplication")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--results_path", type=str, default="evaluation_results.json", help="Path to save the results JSON")
    parser.add_argument("--all_metrics_plot_path", type=str, default="all_metrics_comparison.png", help="Path to save the grouped metrics plot")
    parser.add_argument("--duplicate_layers", type=str, default="", help="Comma-separated list of layer indices to duplicate (e.g., '3,5')")
    parser.add_argument("--duplication_counts", type=str, default="", help="Comma-separated list of duplication counts corresponding to duplicate_layers (e.g., '2,1')")
    args = parser.parse_args()
    main(args)

# Run the evaluation script with the following command:
#python src/evaluation.py --batch_size 16 --results_path results.json --all_metrics_plot_path all_metrics.png --duplicate_layers 3,5 --duplication_counts 2,1
