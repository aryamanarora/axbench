#!/bin/bash
#SBATCH --job-name=ao-grad-steer-train
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/18_ao_gradient_steer_%j.log

set -e

cd /home/aryaman/axbench
source .venv/bin/activate
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-dummy}"
export CUDA_LAUNCH_BLOCKING=1

DUMP_DIR=results/ao_gradient_steering
CONFIG=axbench/sweep/aryaman/activation_oracle/gradient_steering_llama3_8b.yaml

# Symlink generated data from AO reading run if not already present
if [ ! -d "$DUMP_DIR/generate" ]; then
    mkdir -p "$DUMP_DIR"
    ln -s /home/aryaman/axbench/results/ao_reading/generate "$DUMP_DIR/generate"
fi

echo "Training Activation Oracle gradient steering vectors..."
torchrun --nproc_per_node=1 --master_port=29503 axbench/scripts/train.py \
    --config "$CONFIG" \
    --dump_dir "$DUMP_DIR"

echo "Training complete. Running steering inference..."
torchrun --nproc_per_node=1 --master_port=29503 axbench/scripts/inference.py \
    --config "$CONFIG" \
    --mode steering \
    --dump_dir "$DUMP_DIR" \
    --overwrite_inference_data_dir "$DUMP_DIR/inference"

echo "Inference complete."
