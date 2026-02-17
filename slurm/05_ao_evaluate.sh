#!/bin/bash
#SBATCH --job-name=ao-eval
#SBATCH --partition=batch
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=slurm/logs/05_ao_evaluate_%j.log

cd /home/aryaman/axbench
source .venv/bin/activate
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-dummy}"

python axbench/scripts/evaluate.py \
    --config axbench/sweep/aryaman/activation_oracle/reading_llama3_1_8b.yaml \
    --mode latent --dump_dir results/ao_detection
