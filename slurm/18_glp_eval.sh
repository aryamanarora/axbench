#!/bin/bash
#SBATCH --job-name=glp-eval
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=slurm/logs/18_glp_eval_%j.log

set -e

cd /home/aryaman/axbench
source .venv/bin/activate
set -a && source /scratch/aryaman/clarity/.env && set +a

DUMP_DIR=results/glp_diffmean
CONFIG=axbench/sweep/aryaman/glp/glp_diffmean_llama3_8b.yaml

echo "Running GLP DiffMean steering evaluation..."
python axbench/scripts/evaluate.py \
    --config "$CONFIG" \
    --mode steering \
    --dump_dir "$DUMP_DIR"

echo "Evaluation complete."
