#!/bin/bash
#SBATCH --job-name=lqa-act-steer
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/17_lqa_activation_steer_%j.log

set -e

cd /home/aryaman/axbench
source .venv/bin/activate
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-dummy}"

DUMP_DIR=results/latentqa_activation_steering
CONFIG=axbench/sweep/aryaman/latentqa/activation_steering_llama3_8b.yaml

# Symlink generated data from detection run if not already present
if [ ! -d "$DUMP_DIR/generate" ]; then
    mkdir -p "$DUMP_DIR"
    ln -s /home/aryaman/axbench/results/latentqa_detection/generate "$DUMP_DIR/generate"
fi

# No training needed — skip straight to inference
echo "Running LatentQA activation steering inference..."
torchrun --nproc_per_node=1 --master_port=29502 axbench/scripts/inference.py \
    --config "$CONFIG" \
    --dump_dir "$DUMP_DIR" \
    --mode steering \
    --max_concepts 500

echo "Inference complete."
