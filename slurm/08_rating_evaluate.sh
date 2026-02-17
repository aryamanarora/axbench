#!/bin/bash
#SBATCH --job-name=rating-eval
#SBATCH --partition=batch
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=slurm/logs/08_rating_eval_%j.log

cd /home/aryaman/axbench
source .venv/bin/activate
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-dummy}"

echo "=== Evaluating AO Rating ==="
python axbench/scripts/evaluate.py \
    --config axbench/sweep/aryaman/rating_comparison/ao_rating_llama3_1_8b.yaml \
    --mode latent --dump_dir results/ao_rating_detection

echo "=== Evaluating LQA Rating ==="
python axbench/scripts/evaluate.py \
    --config axbench/sweep/aryaman/rating_comparison/lqa_rating_llama3_8b.yaml \
    --mode latent --dump_dir results/lqa_rating_detection
