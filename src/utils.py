import torch
import matplotlib.pyplot as plt
import evaluate

def tokenize_function(examples, tokenizer, max_length=128):
    tokenized = tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=max_length)
    tokenized["label"] = examples["label"]
    return tokenized

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
