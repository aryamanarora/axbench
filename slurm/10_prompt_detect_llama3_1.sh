#!/bin/bash
#SBATCH --job-name=pd-llama31
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/10_prompt_detect_llama3_1_%j.log

cd /home/aryaman/axbench
source .venv/bin/activate
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-dummy}"

DUMP_DIR="results/prompt_detect_llama3_1"
mkdir -p "${DUMP_DIR}/inference" "${DUMP_DIR}/generate" "${DUMP_DIR}/train"

cp -n results/ao_detection/inference/latent_eval_data.parquet "${DUMP_DIR}/inference/" 2>/dev/null
cp -n results/ao_detection/generate/metadata.jsonl "${DUMP_DIR}/generate/" 2>/dev/null
python -c "import json; json.dump({'layer': 16, 'component': 'res'}, open('${DUMP_DIR}/train/config.json', 'w'))"

torchrun --nproc_per_node=1 --master_port=29503 axbench/scripts/inference.py \
    --config axbench/sweep/aryaman/prompt_detection/llama3_1_8b.yaml \
    --mode latent --dump_dir "${DUMP_DIR}" \
    --overwrite_inference_data_dir "${DUMP_DIR}/inference"
