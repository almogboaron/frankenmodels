import argparse
import csv
import os
import torch
import random
from collections import Counter, defaultdict
from transformers import AutoTokenizer
from utils import (
    load_trained_model,
    evaluate_model,
    get_dataloader,
    load_frankenmodel,
    load_tokenized_dataset,
)

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

    fieldnames = ["trial", "duplicate_layers", "duplication_counts", "accuracy", "f1", "precision", "recall",
                  "acc_diff", "f1_diff", "precision_diff", "recall_diff"]

    write_header = not os.path.exists(args.csv_path)

    with open(args.csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        print(f"Evaluating Frankenmodel duplicating layers {args.duplicate_layers} counts {args.duplication_counts}...")
        model = load_frankenmodel(device, args.base_model_path, args.duplicate_layers, args.duplication_counts)
        metrics = evaluate_model(model, dataloader, device)
        print(f"Layers {args.duplicate_layers} counts {args.duplication_counts}... Metrics: {metrics}")

        row = {
            "trial": 1,
            "duplicate_layers": args.duplicate_layers,
            "duplication_counts": args.duplication_counts,
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

def random_layers_by_counts_max(args, total_duplications=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path if args.tokenizer_path else args.model_type)
    tokenized_val = load_tokenized_dataset(tokenizer, args.dataset_name, args.subset)
    dataloader = get_dataloader(tokenized_val, batch_size=args.batch_size)

    print("Loading base model...")
    base_model = load_trained_model(device, args.base_model_path)
    base_metrics = evaluate_model(base_model, dataloader, device)
    print("Base Model Metrics:", base_metrics)

    total_layers = base_model.config.num_hidden_layers
    layer_range = list(range(args.min_layer, min(args.max_layer + 1, total_layers)))

    if not layer_range:
        raise ValueError(f"Invalid layer range: {args.min_layer} to {args.max_layer}")

    aggregated = defaultdict(float)
    all_trials_layers = []
    all_trials_counts = []

    fieldnames = ["trial", "duplicate_layers", "duplication_counts", "accuracy", "f1", "precision", "recall",
                  "acc_diff", "f1_diff", "precision_diff", "recall_diff"]

    write_header = not os.path.exists(args.csv_path)

    with open(args.csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for trial in range(args.trials):
            sampled_layers = [random.choice(layer_range) for _ in range(total_duplications)]
            counts = Counter(sampled_layers)
            selected_layers = sorted(counts.keys())
            duplication_counts = [counts[layer] for layer in selected_layers]

            all_trials_layers.append(selected_layers)
            all_trials_counts.append(duplication_counts)

            print(f"Trial {trial+1}: Layers {selected_layers} with counts {duplication_counts}")

            model = load_frankenmodel(device, args.base_model_path, selected_layers, duplication_counts)
            metrics = evaluate_model(model, dataloader, device)

            row = {
                "trial": trial + 1,
                "duplicate_layers": selected_layers,
                "duplication_counts": duplication_counts,
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

            for key in ["accuracy", "f1", "precision", "recall", "acc_diff", "f1_diff", "precision_diff", "recall_diff"]:
                aggregated[key] += row[key]

        # Average row after all trials
        avg_row = {
            "trial": "average",
            "duplicate_layers": "-",
            "duplication_counts": "-",
        }
        for key in ["accuracy", "f1", "precision", "recall", "acc_diff", "f1_diff", "precision_diff", "recall_diff"]:
            avg_row[key] = aggregated[key] / args.trials
        writer.writerow(avg_row)

    print("\nAverage metrics over", args.trials, "trials:")
    for key in ["accuracy", "f1", "precision", "recall", "acc_diff", "f1_diff", "precision_diff", "recall_diff"]:
        avg = aggregated[key] / args.trials
        print(f"{key}: {avg:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Frankenmodel with duplicated layers.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--csv_path", type=str, default="layer_replication_results.csv")
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="glue")
    parser.add_argument("--subset", type=str, default="sst2")
    parser.add_argument("--duplicate_layers", nargs="+", type=int, default=[])
    parser.add_argument("--duplication_counts", nargs="+", type=int, default=[])
    parser.add_argument("--total_duplications", type=int, default=6, help="Total number of duplications to sample with replacement")
    parser.add_argument("--trials", type=int, default=5, help="Number of randomized trials")
    parser.add_argument("--min_layer", type=int, default=0, help="Minimum encoder layer index to consider for random duplication")
    parser.add_argument("--max_layer", type=int, default=11, help="Maximum encoder layer index to consider for random duplication")
    args = parser.parse_args()

    if not args.duplicate_layers and not args.duplication_counts:
        random_layers_by_counts_max(args, total_duplications=args.total_duplications)
    else:
        main(args)

# ================================
# Example 1: Fixed layers [5, 6, 7] with 3 duplications each:
# python your_script.py \
#   --model_type bert-base-uncased \
#   --base_model_path ./results/bert_sst2 \
#   --csv_path results_fixed.csv \
#   --batch_size 32 \
#   --duplicate_layers 5 6 7 \
#   --duplication_counts 3 3 3

# Example 2: Randomly sample 6 total duplications from layers [4–10], across 5 trials:
# python src/evaluate_duplication.py \
#   --model_type bert-base-uncased \
#   --base_model_path ./results/bert_sst2 \
#   --tokenizer_path ./results/bert_sst2 \
#   --csv_path results_random.csv \
#   --batch_size 32 \
#   --total_duplications 3 \
#   --min_layer 5 \
#   --max_layer 8 \
#   --trials 15