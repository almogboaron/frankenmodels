import argparse
import csv
import random
import numpy as np
import torch
import copy
import os
import json
from datasets import load_dataset
from transformers import (BertForSequenceClassification, BertTokenizer,
                          TrainingArguments, Trainer, default_data_collator)
from utils import tokenize_function, evaluate_model, get_dataloader
from scipy.stats import ttest_1samp

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_and_tokenize_datasets(tokenizer, max_length=128):
    dataset = load_dataset("glue", "sst2")
    train_dataset = dataset["train"].map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=[col for col in dataset["train"].column_names if col not in ["label"]]
    )
    val_dataset = dataset["validation"].map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=[col for col in dataset["validation"].column_names if col not in ["label"]]
    )
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return train_dataset, val_dataset

def train_model(seed, model_dir, num_train_epochs, batch_size, learning_rate):
    set_seed(seed)
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    train_dataset, val_dataset = load_and_tokenize_datasets(tokenizer)
    
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    training_args = TrainingArguments(
        output_dir=model_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        logging_steps=50,
        seed=seed,
        report_to=[]  # disable external logging
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )
    
    trainer.train()
    
    # Save the fine-tuned model and tokenizer.
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    
    return model, tokenizer, val_dataset

def load_trained_model(device, model_dir):
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    return model

def load_frankenmodel(device, model_dir, duplicate_layers=None, duplication_counts=None):
    """
    Loads the fine-tuned model and creates a Frankenmodel by duplicating specified encoder layers.
    Args:
        duplicate_layers: list of 0-indexed layer indices to duplicate.
        duplication_counts: list of counts corresponding to duplicate_layers.
    """
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False

    original_layers = model.bert.encoder.layer
    new_layers = torch.nn.ModuleList()
    if duplicate_layers is None or duplication_counts is None:
        new_layers = original_layers
    else:
        for i, layer in enumerate(original_layers):
            new_layers.append(layer)
            if i in duplicate_layers:
                idx = duplicate_layers.index(i)
                count = duplication_counts[idx]
                for _ in range(count):
                    duplicate_layer = copy.deepcopy(layer)
                    new_layers.append(duplicate_layer)
    model.bert.encoder.layer = new_layers
    model.config.num_hidden_layers = len(new_layers)
    return model

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
        os.makedirs(model_dir, exist_ok=True)
        
        # Train the model (fine-tuning).
        print("Training the base model...")
        train_model(seed, model_dir, args.num_train_epochs, args.batch_size, args.learning_rate)
        
        # Load tokenizer from the trained model.
        tokenizer = BertTokenizer.from_pretrained(model_dir)
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
# python src/train_evaluate_validate.py --batch_size 16 --seeds 42,43,44,45,46,47,48,49,50 --num_train_epochs 2 --duplicate_layers 6 --duplication_counts 1 --csv_path results.csv
