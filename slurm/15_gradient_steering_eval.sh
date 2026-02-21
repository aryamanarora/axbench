#!/bin/bash
#SBATCH --job-name=lqa-grad-eval
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=slurm/logs/15_gradient_steering_eval_%j.log

set -e

cd /home/aryaman/axbench
source .venv/bin/activate
set -a && source /scratch/aryaman/clarity/.env && set +a

DUMP_DIR=results/latentqa_gradient_steering
CONFIG=axbench/sweep/aryaman/latentqa/gradient_steering_llama3_8b.yaml

echo "Running gradient steering evaluation..."
python axbench/scripts/evaluate.py \
    --config "$CONFIG" \
    --mode steering \
    --dump_dir "$DUMP_DIR"

echo "Evaluation complete."
