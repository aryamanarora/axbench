#!/bin/bash
#SBATCH --job-name=lqa-steer-train
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/16_lqa_steering_train_%j.log

set -e

cd /home/aryaman/axbench
source .venv/bin/activate
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-dummy}"

DUMP_DIR=results/latentqa_steering
CONFIG=axbench/sweep/aryaman/latentqa/steering_llama3_8b.yaml

# Symlink generated data from detection run if not already present
if [ ! -d "$DUMP_DIR/generate" ]; then
    mkdir -p "$DUMP_DIR"
    ln -s /home/aryaman/axbench/results/latentqa_detection/generate "$DUMP_DIR/generate"
fi

echo "Training LatentQA steering LoRA adapters..."
torchrun --nproc_per_node=1 --master_port=29501 axbench/scripts/train.py \
    --config "$CONFIG" \
    --dump_dir "$DUMP_DIR" \
    --max_concepts 1

echo "Training complete."
