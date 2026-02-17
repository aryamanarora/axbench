#!/bin/bash
#SBATCH --job-name=lqa-setup
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=slurm/logs/01_setup_%j.log

set -e

cd /home/aryaman/axbench
source .venv/bin/activate

DUMP_DIR=results/latentqa_detection
CONCEPT_JSON=axbench/data/gemma-2-2b_20-gemmascope-res-16k.json

# Download concept JSON if missing
if [ ! -f "$CONCEPT_JSON" ]; then
    echo "Downloading concept JSON..."
    wget -P axbench/data https://neuronpedia-exports.s3.amazonaws.com/explanations-only/gemma-2-2b_20-gemmascope-res-16k.json
fi

# Setup LatentQA
if [ ! -d "./latentqa" ]; then
    echo "Cloning LatentQA..."
    git clone https://github.com/aypan17/latentqa.git ./latentqa
fi
export PYTHONPATH="/home/aryaman/axbench/latentqa:$PYTHONPATH"
python -c "from lit.utils.activation_utils import latent_qa; print('LatentQA import OK')"

# Download seed sentences/instructions if missing
if [ ! -d "axbench/data/seed_sentences" ] || [ ! -d "axbench/data/seed_instructions" ]; then
    echo "Downloading seed sentences and instructions..."
    pushd axbench/data && python download-seed-sentences.py && popd
fi

# Download HF dataset and prepare directory structure
echo "Preparing data from pyvene/axbench-concept500..."
python slurm/prepare_data.py \
    --dump_dir "$DUMP_DIR" \
    --concept_path "$CONCEPT_JSON" \
    --layer 15 \
    --max_concepts 500

echo "Setup complete."
