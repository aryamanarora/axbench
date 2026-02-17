#!/bin/bash
#SBATCH --job-name=lqa-inference
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/02_inference_%j.log

set -e

cd /home/aryaman/axbench
source .venv/bin/activate
export PYTHONPATH="/home/aryaman/axbench/latentqa:$PYTHONPATH"
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-dummy}"
# export CUDA_LAUNCH_BLOCKING=1

DUMP_DIR=results/latentqa_detection
CONFIG=axbench/sweep/aryaman/latentqa/reading_llama3_8b.yaml

echo "Running LatentQA detection inference..."
torchrun --nproc_per_node=1 --master_port=29500 axbench/scripts/inference.py \
    --config "$CONFIG" \
    --mode latent \
    --dump_dir "$DUMP_DIR" \
    --overwrite_inference_data_dir "$DUMP_DIR/inference"

echo "Inference complete."
