#!/bin/bash
#SBATCH --job-name=ao-inference
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/04_ao_inference_%j.log

cd /home/aryaman/axbench
source .venv/bin/activate
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-dummy}"

# Setup data dir (reuse concept500 data from LatentQA run)
DUMP_DIR="results/ao_detection"
mkdir -p "${DUMP_DIR}/inference"

# Copy pre-generated data if not already present
if [ ! -f "${DUMP_DIR}/inference/latent_eval_data.parquet" ]; then
    cp results/latentqa_detection/inference/latent_eval_data.parquet "${DUMP_DIR}/inference/"
fi
if [ ! -f "${DUMP_DIR}/inference/metadata.jsonl" ]; then
    cp results/latentqa_detection/inference/metadata.jsonl "${DUMP_DIR}/inference/"
fi
if [ ! -f "${DUMP_DIR}/inference/config.json" ]; then
    # Create config with correct layer for Llama-3.1-8B
    python -c "
import json
config = {'layer': 16, 'component': 'res'}
json.dump(config, open('${DUMP_DIR}/inference/config.json', 'w'))
"
fi

torchrun --nproc_per_node=1 axbench/scripts/inference.py \
    --config axbench/sweep/aryaman/activation_oracle/reading_llama3_1_8b.yaml \
    --mode latent --dump_dir "${DUMP_DIR}" \
    --overwrite_inference_data_dir "${DUMP_DIR}/inference"
