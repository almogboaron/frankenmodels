import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from utils import tokenize_function

def main():
    # Set device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # 1. Load SST-2 dataset (from GLUE)
    dataset = load_dataset("glue", "sst2")
    
    # 2. Load tokenizer and model (BERT base)
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)
    
    # 3. Tokenize the training and validation sets
    tokenized_train = dataset["train"].map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_val = dataset["validation"].map(lambda x: tokenize_function(x, tokenizer), batched=True)
    
    # Set format for PyTorch tensors
    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_val.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    # 4. Set up training arguments
    training_args = TrainingArguments(
        output_dir="./results/bert_sst2",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        seed=42,
        disable_tqdm=False,
    )
    
    # 5. Define a simple compute_metrics function
    def compute_metrics(eval_pred):
        import numpy as np
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        import evaluate
        metric = evaluate.load("accuracy")
        return metric.compute(predictions=preds, references=labels)
    
    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
    )
    
    # 7. Train the model
    trainer.train()
    
    # 8. Save the fine-tuned model and tokenizer for later evaluation
    model.save_pretrained("./results/bert_sst2")
    tokenizer.save_pretrained("./results/bert_sst2")
    print("Training complete and model saved in ./results/bert_sst2")

if __name__ == "__main__":
    main()
