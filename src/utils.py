import torch
import matplotlib.pyplot as plt
import evaluate
import argparse
import csv
import random
import numpy as np
import copy
import os
import json
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, default_data_collator)
from scipy.stats import ttest_1samp

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def tokenize_function(examples, tokenizer, max_length=128):
    tokenized = tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=max_length)
    tokenized["label"] = examples["label"]
    return tokenized

def load_tokenized_dataset(tokenizer, dataset_name="glue", subset="sst2", split="validation", max_length=128):
    dataset = load_dataset(dataset_name, subset)
    tokenized_dataset = dataset[split].map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=[col for col in dataset[split].column_names if col not in ["label"]]
    )
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return tokenized_dataset

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

def train_model(seed,model_name,model_dir, num_train_epochs, batch_size, learning_rate):
    set_seed(seed)
    #model_name ="bert-base-uncased",

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset, val_dataset = load_and_tokenize_datasets(tokenizer)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
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
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    return model

def model_real_parameters(model):
    unique_params = {id(p): p for p in model.parameters()}
    num_unique = sum(p.numel() for p in unique_params.values())
    return num_unique

def load_frankenmodel(device, model_dir, duplicate_layers=None, duplication_counts=None):
    """
    Loads the fine-tuned model and creates a Frankenmodel by duplicating specified encoder layers.
    Args:
        duplicate_layers: list of 0-indexed layer indices to duplicate.
        duplication_counts: list of counts corresponding to duplicate_layers.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False

    original_layers = model.base_model.encoder.layer
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
                    duplicate_layer = copy.deepcopy(layer)  #Done:Check for soft copy to not spend more space on bigger model :DONE
                    new_layers.append(duplicate_layer)
                    #new_layers.append(layer)                 # Changed from Above to soft copy.
    model.base_model.encoder.layer = new_layers
    model.config.num_hidden_layers = len(new_layers)
    return model

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in dataloader:
        # Use "label" if available, else fallback to "labels"
        label_key = "label" if "label" in batch else "labels"
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch[label_key].cpu().numpy())
    
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    
    accuracy = accuracy_metric.compute(predictions=all_preds, references=all_labels)
    f1 = f1_metric.compute(predictions=all_preds, references=all_labels, average="macro")
    precision = precision_metric.compute(predictions=all_preds, references=all_labels, average="macro")
    recall = recall_metric.compute(predictions=all_preds, references=all_labels, average="macro")
    
    results = {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
        "precision": precision["precision"],
        "recall": recall["recall"]
    }
    return results

def get_dataloader(tokenized_dataset, batch_size=16):
    from torch.utils.data import DataLoader
    from transformers import default_data_collator
    return DataLoader(tokenized_dataset, batch_size=batch_size, collate_fn=default_data_collator)

def plot_accuracies(model_names, accuracies, save_path="evaluation_comparison.png"):
    """
    Plots and saves a bar chart for accuracy comparison.
    """
    plt.figure(figsize=(6, 4))
    bars = plt.bar(model_names, accuracies, color=['skyblue', 'salmon'])
    plt.xlabel("Model Type")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison on SST-2")
    plt.ylim(0, 1)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.4f}", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Plot saved to {save_path}")
