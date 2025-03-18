import argparse
import csv
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from datasets import load_dataset
from utils import tokenize_function, evaluate_model, get_dataloader

def load_tokenized_dataset(tokenizer, split="validation", max_length=128):
    dataset = load_dataset("glue", "sst2")
    tokenized_dataset = dataset[split].map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=[col for col in dataset[split].column_names if col not in ["label"]]
    )
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return tokenized_dataset

def load_base_model(device, model_dir="./results/bert_sst2"):
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    return model

def load_frankenmodel(device, model_dir="./results/bert_sst2", duplicate_layers=None, duplication_counts=None):
    """
    Loads the base model and creates a Frankenmodel by duplicating specified encoder layers.
    Args:
        duplicate_layers: list of encoder indices (0-indexed) to duplicate.
        duplication_counts: list of counts (how many times to duplicate) corresponding to duplicate_layers.
    """
    import copy
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False

    original_layers = model.bert.encoder.layer
    new_layers = torch.nn.ModuleList()
    num_layers = len(original_layers)
    
    # If no layers specified, return original model (no duplication)
    if duplicate_layers is None or duplication_counts is None:
        new_layers = original_layers
    else:
        for i, layer in enumerate(original_layers):
            new_layers.append(layer)
            if i in duplicate_layers:
                # Get number of duplicates for this layer
                idx = duplicate_layers.index(i)
                count = duplication_counts[idx]
                for _ in range(count):
                    duplicate_layer = copy.deepcopy(layer)
                    new_layers.append(duplicate_layer)
                    
    model.bert.encoder.layer = new_layers
    model.config.num_hidden_layers = len(new_layers)
    return model

def evaluate_on_layer(device, tokenizer, dataloader, layer_idx, model_dir="./results/bert_sst2"):
    """
    Creates a Frankenmodel that duplicates the specified layer (layer_idx) once,
    evaluates it, and returns the evaluation metrics.
    """
    duplicate_layers = [layer_idx]
    duplication_counts = [1]  # duplicate once
    model = load_frankenmodel(device, model_dir=model_dir,
                              duplicate_layers=duplicate_layers,
                              duplication_counts=duplication_counts)
    metrics = evaluate_model(model, dataloader, device)
    return metrics

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load tokenizer from the saved fine-tuned base model directory.
    tokenizer = BertTokenizer.from_pretrained("./results/bert_sst2")
    tokenized_val = load_tokenized_dataset(tokenizer, split="validation")
    dataloader = get_dataloader(tokenized_val, batch_size=args.batch_size)

    # Evaluate base model once.
    print("Evaluating Base Model...")
    base_model = load_base_model(device, model_dir="./results/bert_sst2")
    base_metrics = evaluate_model(base_model, dataloader, device)
    print("Base Model Metrics:", base_metrics)

    # Prepare CSV file
    csv_file = args.csv_path
    fieldnames = ["layer_idx", "accuracy", "f1", "precision", "recall",
                  "acc_diff", "f1_diff", "precision_diff", "recall_diff"]
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Iterate over each layer (for bert-base: 0 to 11)
        for layer_idx in range(12):
            print(f"Evaluating Frankenmodel with duplication on layer {layer_idx}...")
            metrics = evaluate_on_layer(device, tokenizer, dataloader, layer_idx, model_dir="./results/bert_sst2")
            print(f"Layer {layer_idx} Metrics: {metrics}")
            
            # Compute differences from base model
            row = {
                "layer_idx": layer_idx,
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "acc_diff": metrics["accuracy"] - base_metrics["accuracy"],
                "f1_diff": metrics["f1"] - base_metrics["f1"],
                "precision_diff": metrics["precision"] - base_metrics["precision"],
                "recall_diff": metrics["recall"] - base_metrics["recall"]
            }
            writer.writerow(row)
    print(f"Results saved to {csv_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate individual layer replication impact on Frankenmodel performance.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--csv_path", type=str, default="layer_replication_results.csv", help="Path to save the CSV results")
    args = parser.parse_args()
    main(args)
