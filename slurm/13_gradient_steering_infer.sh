#!/bin/bash
#SBATCH --job-name=lqa-grad-steer-infer
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/13_gradient_steering_infer_%j.log

set -e

cd /home/aryaman/axbench
source .venv/bin/activate

export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-dummy}"

DUMP_DIR=results/latentqa_gradient_steering
CONFIG=axbench/sweep/aryaman/latentqa/gradient_steering_llama3_8b.yaml

echo "Running steering inference..."
torchrun --nproc_per_node=1 --master_port=29501 axbench/scripts/inference.py \
    --config "$CONFIG" \
    --mode steering \
    --dump_dir "$DUMP_DIR" \
    --overwrite_inference_data_dir "$DUMP_DIR/inference"

echo "Inference complete."
