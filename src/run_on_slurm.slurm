#! /bin/bash

#SBATCH --job-name=model_creation
#SBATCH --output=/home/iscb/wolfson/almogboaron/test/test.out 
#SBATCH --error=/home/iscb/wolfson/almogboaron/test/test.err
#SBATCH --partition=gpu-mol
#SBATCH --nodelist=n-801
#SBATCH --time=1440
#SBATCH --signal=USR1@120 
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4 
#SBATCH --gpus=1 # GPUs in total

python src/evaluate_duplication.py \
  --model_type bert-base-uncased \
  --base_model_path ./results/bert_sst2 \
  --tokenizer_path ./results/bert_sst2 \
  --csv_path results_random.csv \
  --batch_size 32 \
  --total_duplications 3 \
  --min_layer 5 \
  --max_layer 8 \
  --trials 15
