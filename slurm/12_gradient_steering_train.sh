#!/bin/bash
#SBATCH --job-name=lqa-grad-steer-train
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/12_gradient_steering_train_%j.log

set -e

cd /home/aryaman/axbench
source .venv/bin/activate
export PYTHONPATH="/home/aryaman/axbench/latentqa:$PYTHONPATH"
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-dummy}"

DUMP_DIR=results/latentqa_gradient_steering
CONFIG=axbench/sweep/aryaman/latentqa/gradient_steering_llama3_8b.yaml

# Symlink generated data from detection run if not already present
if [ ! -d "$DUMP_DIR/generate" ]; then
    mkdir -p "$DUMP_DIR"
    ln -s /home/aryaman/axbench/results/latentqa_detection/generate "$DUMP_DIR/generate"
fi

echo "Training LatentQA gradient steering vectors..."
torchrun --nproc_per_node=1 --master_port=29501 axbench/scripts/train.py \
    --config "$CONFIG" \
    --dump_dir "$DUMP_DIR"

echo "Training complete. Running steering inference..."
torchrun --nproc_per_node=1 --master_port=29501 axbench/scripts/inference.py \
    --config "$CONFIG" \
    --mode steering \
    --dump_dir "$DUMP_DIR" \
    --overwrite_inference_data_dir "$DUMP_DIR/inference"

echo "Inference complete."
