import argparse
import csv
import torch
from transformers import AutoTokenizer
from utils import load_trained_model,evaluate_model, get_dataloader,load_frankenmodel,load_tokenized_dataset

def evaluate_on_layer(device, dataloader, layer_idx, base_model_path):
    duplicate_layers = [layer_idx]
    duplication_counts = [1]
    model = load_frankenmodel(device, base_model_path, duplicate_layers, duplication_counts)
    metrics = evaluate_model(model, dataloader, device)
    return metrics


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path if args.tokenizer_path else args.model_type)
    tokenized_val = load_tokenized_dataset(tokenizer, args.dataset_name, args.subset)
    dataloader = get_dataloader(tokenized_val, batch_size=args.batch_size)

    print("Evaluating Base Model...")
    base_model = load_trained_model(device, args.base_model_path)
    base_metrics = evaluate_model(base_model, dataloader, device)
    print("Base Model Metrics:", base_metrics)

    fieldnames = ["layer_idx", "accuracy", "f1", "precision", "recall",
                  "acc_diff", "f1_diff", "precision_diff", "recall_diff"]

    with open(args.csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        num_layers = base_model.config.num_hidden_layers

        for layer_idx in range(num_layers):
            print(f"Evaluating Frankenmodel duplicating layer {layer_idx}...")
            metrics = evaluate_on_layer(device, dataloader, layer_idx, args.base_model_path)
            print(f"Layer {layer_idx} Metrics: {metrics}")

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

    print(f"Results saved to {args.csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generalized evaluation of layer duplication impact on Frankenmodels.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--csv_path", type=str, default="layer_replication_results.csv", help="Path for saving results CSV")
    parser.add_argument("--model_type", type=str, required=True, help="HuggingFace model type")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the fine-tuned base model")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer if different from model")
    parser.add_argument("--dataset_name", type=str, default="glue", help="Dataset name")
    parser.add_argument("--subset", type=str, default="sst2", help="Dataset subset name")

    args = parser.parse_args()
    main(args)
