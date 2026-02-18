#!/bin/bash
#SBATCH --job-name=glp-diffmean
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/14_glp_steering_%j.log

set -e

cd /home/aryaman/axbench
source .venv/bin/activate
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-dummy}"

DUMP_DIR=results/glp_diffmean
CONFIG=axbench/sweep/aryaman/glp/glp_diffmean_llama3_8b.yaml

# Symlink generated data from existing run
if [ ! -d "$DUMP_DIR/generate" ]; then
    mkdir -p "$DUMP_DIR"
    ln -s /home/aryaman/axbench/results/latentqa_detection/generate "$DUMP_DIR/generate"
fi

echo "Training GLPDiffMean (diff-in-means)..."
torchrun --nproc_per_node=1 --master_port=29502 axbench/scripts/train.py \
    --config "$CONFIG" \
    --dump_dir "$DUMP_DIR"

echo "Training complete. Running steering inference with GLP post-processing..."
torchrun --nproc_per_node=1 --master_port=29502 axbench/scripts/inference.py \
    --config "$CONFIG" \
    --mode steering \
    --dump_dir "$DUMP_DIR" \
    --overwrite_inference_data_dir "$DUMP_DIR/inference"

echo "Inference complete."
