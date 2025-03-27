import argparse
import time
import torch
import csv
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import  AutoTokenizer
from utils import get_dataloader, load_tokenized_dataset, load_trained_model, load_frankenmodel,model_real_parameters

def measure_inference_speed_profiler(model, dataloader, device, num_batches=20):
    model.eval()
    total_time = 0.0
    batches = 0

    # Warm-up runs to stabilize performance.
    with torch.no_grad():
        for _ in range(5):
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                _ = model(**batch)
                break

    # Use PyTorch's profiler to capture detailed timings.
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True, profile_memory=True) as prof:
        with record_function("inference_loop"):
            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    if i >= num_batches:
                        break
                    batch = {k: v.to(device) for k, v in batch.items()}
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    start_time = time.time()
                    _ = model(**batch)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    elapsed = time.time() - start_time
                    total_time += elapsed
                    batches += 1
    avg_time = total_time / batches if batches > 0 else None

    # Print a detailed profiler table.
    # print(prof.key_averages().table(sort_by="self_cuda_time_total"))
    return avg_time

def measure_memory_usage(model, dataloader, device):
    # Reset memory stats if using CUDA.
    if device.type == "cuda":
        torch.cuda.reset_max_memory_allocated(device)
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            _ = model(**batch)
            break  # Only one batch needed
    max_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return max_memory

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load tokenizer and prepare validation dataloader.
    tokenizer = AutoTokenizer.from_pretrained("./results/bert_sst2")
    tokenized_val = load_tokenized_dataset(tokenizer, split="validation")
    dataloader = get_dataloader(tokenized_val, batch_size=args.batch_size)

    # Load the Base and Franken Models.
    base_model = load_trained_model(device, model_dir=args.model_dir)
    franken_model = load_frankenmodel(device, model_dir=args.model_dir,
                                      duplicate_layers=[int(x) for x in args.duplicate_layers.split(",")] if args.duplicate_layers else None,
                                      duplication_counts=[int(x) for x in args.duplication_counts.split(",")] if args.duplication_counts else None)
    
    # Report model size.
    base_params = model_real_parameters(base_model)
    franken_params = model_real_parameters(franken_model)
    print(f"Base Model Parameters: {base_params}")
    print(f"Frankenmodel Parameters: {franken_params}")

    # Measure inference speed using our profiler-enhanced function.
    base_time = measure_inference_speed_profiler(base_model, dataloader, device, num_batches=args.num_batches)
    franken_time = measure_inference_speed_profiler(franken_model, dataloader, device, num_batches=args.num_batches)
    print(f"Average inference time per batch (Base): {base_time:.4f} sec")
    print(f"Average inference time per batch (Franken): {franken_time:.4f} sec")

    # Measure memory usage (works only on CUDA).
    base_memory = measure_memory_usage(base_model, dataloader, device)
    franken_memory = measure_memory_usage(franken_model, dataloader, device)
    if device.type == "cuda":
        base_memory_mb = base_memory / (1024 ** 2)
        franken_memory_mb = franken_memory / (1024 ** 2)
        print(f"Max GPU memory allocated (Base): {base_memory_mb:.2f} MB")
        print(f"Max GPU memory allocated (Franken): {franken_memory_mb:.2f} MB")
    else:
        base_memory_mb = "N/A"
        franken_memory_mb = "N/A"
        print("Memory usage measurement is only available on CUDA devices.")

    # Save the trade-off results to CSV.
    csv_file = args.csv_path
    fieldnames = ["model_type", "num_parameters", "avg_inference_time_sec", "max_memory_MB"]
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "model_type": "Base Model",
            "num_parameters": base_params,
            "avg_inference_time_sec": base_time,
            "max_memory_MB": base_memory_mb
        })
        writer.writerow({
            "model_type": "Frankenmodel",
            "num_parameters": franken_params,
            "avg_inference_time_sec": franken_time,
            "max_memory_MB": franken_memory_mb
        })
    print(f"Trade-off results saved to {csv_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure trade-offs between base and Frankenmodels with profiler integration.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--num_batches", type=int, default=20, help="Number of batches to use for timing measurements")
    parser.add_argument("--model_dir", type=str, default="./results/bert_sst2", help="Directory for the fine-tuned base model")
    parser.add_argument("--duplicate_layers", type=str, default="6", help="Comma-separated list of layer indices to duplicate (e.g., '6')")
    parser.add_argument("--duplication_counts", type=str, default="1", help="Comma-separated duplication counts corresponding to duplicate_layers (e.g., '1')")
    parser.add_argument("--csv_path", type=str, default="tradeoffs_results.csv", help="CSV file to save trade-off results")
    args = parser.parse_args()
    main(args)
