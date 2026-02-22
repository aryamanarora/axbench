#!/bin/bash
#SBATCH --job-name=lqa-evaluate
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=slurm/logs/03_evaluate_%j.log

set -e

cd /home/aryaman/axbench
source .venv/bin/activate


DUMP_DIR=results/latentqa_detection
CONFIG=axbench/sweep/aryaman/latentqa/reading_llama3_8b.yaml

echo "Running LatentQA detection evaluation..."
python axbench/scripts/evaluate.py \
    --config "$CONFIG" \
    --mode latent \
    --dump_dir "$DUMP_DIR"

echo "Evaluation complete."
