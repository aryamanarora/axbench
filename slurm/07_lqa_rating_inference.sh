#!/bin/bash
#SBATCH --job-name=lqa-rating
#SBATCH --partition=batch
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/07_lqa_rating_%j.log

cd /home/aryaman/axbench
source .venv/bin/activate
export PYTHONPATH="/home/aryaman/axbench/latentqa:$PYTHONPATH"
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-dummy}"

DUMP_DIR="results/lqa_rating_detection"
mkdir -p "${DUMP_DIR}/inference" "${DUMP_DIR}/generate" "${DUMP_DIR}/train"

# Copy pre-generated data
cp -n results/latentqa_detection/inference/latent_eval_data.parquet "${DUMP_DIR}/inference/" 2>/dev/null
cp -n results/ao_detection/generate/metadata.jsonl "${DUMP_DIR}/generate/" 2>/dev/null
python -c "import json; json.dump({'layer': 15, 'component': 'res'}, open('${DUMP_DIR}/train/config.json', 'w'))"

torchrun --nproc_per_node=1 axbench/scripts/inference.py \
    --config axbench/sweep/aryaman/rating_comparison/lqa_rating_llama3_8b.yaml \
    --mode latent --dump_dir "${DUMP_DIR}" \
    --overwrite_inference_data_dir "${DUMP_DIR}/inference"
