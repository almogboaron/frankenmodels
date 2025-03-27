import argparse
import csv
import random
import numpy as np
import torch
import copy
import os
import json
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, default_data_collator)
from utils import tokenize_function, evaluate_model, get_dataloader ,set_seed,load_and_tokenize_datasets,train_model,load_trained_model,load_frankenmodel
from scipy.stats import ttest_1samp

def evaluate_models(device, tokenizer, val_dataset, batch_size, model_dir, duplicate_layers, duplication_counts):
    dataloader = get_dataloader(val_dataset, batch_size=batch_size)
    
    # Evaluate base model.
    base_model = load_trained_model(device, model_dir)
    base_metrics = evaluate_model(base_model, dataloader, device)
    
    # Evaluate Frankenmodel.
    franken_model = load_frankenmodel(device, model_dir, duplicate_layers, duplication_counts)
    franken_metrics = evaluate_model(franken_model, dataloader, device)
    
    diff = {k: franken_metrics[k] - base_metrics[k] for k in base_metrics}
    return base_metrics, franken_metrics, diff

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in args.seeds.split(",")]
    all_results = []  # Will store results for each seed.
    
    # For statistical tests: store differences across seeds.
    improvements_list = []  # list of dicts, each dict: {metric: diff}
    
    for seed in seeds:
        print(f"\n--- Running training and evaluation for seed {seed} ---")
        # Use a unique directory per seed.
        model_dir = os.path.join(args.model_output_dir, f"seed_{seed}")
        if not os.path.exists(model_dir):
            print("Model with seed does not exist , creating it.")
            os.makedirs(model_dir, exist_ok=True)
            
            # Train the model (fine-tuning).
            print("Training the base model...")
            train_model(seed, model_dir, args.num_train_epochs, args.batch_size, args.learning_rate)
        
        print("Model exists and trained")

        # Load tokenizer from the trained model.
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        # Get validation dataset.
        _, val_dataset = load_and_tokenize_datasets(tokenizer)
        
        # Parse duplication parameters.
        duplicate_layers = [int(x) for x in args.duplicate_layers.split(",")] if args.duplicate_layers else None
        duplication_counts = [int(x) for x in args.duplication_counts.split(",")] if args.duplication_counts else None
        
        base_metrics, franken_metrics, diff = evaluate_models(device, tokenizer, val_dataset,
                                                              args.batch_size, model_dir,
                                                              duplicate_layers, duplication_counts)
        print(f"Seed {seed} Base Metrics: {base_metrics}")
        print(f"Seed {seed} Franken Metrics: {franken_metrics}")
        print(f"Seed {seed} Improvement: {diff}")
        
        row = {
            "seed": seed,
            "base_accuracy": base_metrics["accuracy"],
            "franken_accuracy": franken_metrics["accuracy"],
            "accuracy_diff": diff["accuracy"],
            "base_f1": base_metrics["f1"],
            "franken_f1": franken_metrics["f1"],
            "f1_diff": diff["f1"],
            "base_precision": base_metrics["precision"],
            "franken_precision": franken_metrics["precision"],
            "precision_diff": diff["precision"],
            "base_recall": base_metrics["recall"],
            "franken_recall": franken_metrics["recall"],
            "recall_diff": diff["recall"]
        }
        all_results.append(row)
        improvements_list.append(diff)
    
    # Aggregate improvements: mean and std for each metric.
    agg_improvements = {}
    for key in improvements_list[0]:
        values = [r[key] for r in improvements_list]
        agg_improvements[key] = {"mean": np.mean(values), "std": np.std(values)}
    
    # Perform one-sample one-sided t-test (H0: mean improvement <= 0; H1: mean improvement > 0).
    p_values = {}
    for metric in improvements_list[0]:
        values = [r[metric] for r in improvements_list]
        # Using alternative='greater' tests if mean(values) > 0.
        t_stat, p_val = ttest_1samp(values, popmean=0, alternative='greater')
        p_values[metric] = p_val
        print(f"Metric: {metric}, t-statistic: {t_stat:.4f}, one-sided p-value: {p_val:.4f}")
    
    # Write results to CSV.
    csv_file = args.csv_path
    fieldnames = ["seed", "base_accuracy", "franken_accuracy", "accuracy_diff",
                  "base_f1", "franken_f1", "f1_diff",
                  "base_precision", "franken_precision", "precision_diff",
                  "base_recall", "franken_recall", "recall_diff"]
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)
        # Write aggregated statistics.
        writer.writerow({})
        writer.writerow({"seed": "Aggregated Mean",
                         "accuracy_diff": agg_improvements["accuracy"]["mean"],
                         "f1_diff": agg_improvements["f1"]["mean"],
                         "precision_diff": agg_improvements["precision"]["mean"],
                         "recall_diff": agg_improvements["recall"]["mean"]})
        writer.writerow({"seed": "Aggregated Std",
                         "accuracy_diff": agg_improvements["accuracy"]["std"],
                         "f1_diff": agg_improvements["f1"]["std"],
                         "precision_diff": agg_improvements["precision"]["std"],
                         "recall_diff": agg_improvements["recall"]["std"]})
        writer.writerow({})
        writer.writerow({"seed": "p-values (one-sided)",
                         "accuracy_diff": p_values["accuracy"],
                         "f1_diff": p_values["f1"],
                         "precision_diff": p_values["precision"],
                         "recall_diff": p_values["recall"]})
    
    print(f"\nAll training and evaluation results saved to {csv_file}")
    print("Aggregated improvements:", agg_improvements)
    print("P-values for one-sided t-tests:", p_values)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train, evaluate, and statistically validate improvements from base to Frankenmodel over multiple seeds."
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation")
    parser.add_argument("--seeds", type=str, default="42,43,44", help="Comma-separated list of random seeds")
    parser.add_argument("--csv_path", type=str, default="train_eval_validation_results.csv", help="CSV file to save the results")
    parser.add_argument("--model_output_dir", type=str, default="./trained_models", help="Directory to save fine-tuned models")
    parser.add_argument("--num_train_epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for training")
    parser.add_argument("--duplicate_layers", type=str, default="6", help="Comma-separated list of layer indices to duplicate for Frankenmodel")
    parser.add_argument("--duplication_counts", type=str, default="1", help="Comma-separated duplication counts corresponding to duplicate_layers")
    args = parser.parse_args()
    main(args)


# Example
# python src/Evaluate_metrics_and_significance.py --batch_size 16 --seeds 42,43,44,45,46,47,48,49,50 --num_train_epochs 2 --duplicate_layers 6 --duplication_counts 1 --csv_path results.csv
