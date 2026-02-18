#!/bin/bash
#SBATCH --job-name=glp-steer-infer
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/14_glp_steering_infer_%j.log

set -e

cd /home/aryaman/axbench
source .venv/bin/activate
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-dummy}"

DUMP_DIR=results/glp_diffmean
CONFIG=axbench/sweep/aryaman/glp/glp_diffmean_llama3_8b.yaml

# Optional: symlink DiffMean weights if reusing existing training
# cd "$DUMP_DIR" && ln -sf DiffMean_weight.pt GLPDiffMean_weight.pt \
#                && ln -sf DiffMean_bias.pt GLPDiffMean_bias.pt && cd -

echo "Running GLP steering inference..."
torchrun --nproc_per_node=1 --master_port=29502 axbench/scripts/inference.py \
    --config "$CONFIG" \
    --mode steering \
    --dump_dir "$DUMP_DIR" \
    --overwrite_inference_data_dir "$DUMP_DIR/inference"

echo "Inference complete."
