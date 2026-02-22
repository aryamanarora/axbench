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

# Setup LatentQA (cloned into axbench/models/_latentqa, auto-discovered via sys.path)
if [ ! -d "axbench/models/_latentqa" ]; then
    echo "Cloning LatentQA..."
    git clone https://github.com/aypan17/latentqa.git axbench/models/_latentqa
fi
python -c "from axbench.models.latentqa import LatentQAReading; print('LatentQA import OK')"

# Setup Activation Oracles (cloned into axbench/models/_activation_oracles, auto-discovered via sys.path)
if [ ! -d "axbench/models/_activation_oracles" ]; then
    echo "Cloning Activation Oracles..."
    git clone https://github.com/adamkarvonen/activation_oracles.git axbench/models/_activation_oracles
fi
python -c "from axbench.models.activation_oracle import ActivationOracleReading; print('Activation Oracle import OK')"

# Download seed sentences/instructions if missing
if [ ! -d "axbench/data/seed_sentences" ] || [ ! -d "axbench/data/seed_instructions" ]; then
    echo "Downloading seed sentences and instructions..."
    pushd axbench/data && python download-seed-sentences.py && popd
fi

# Download HF dataset and prepare directory structure
echo "Preparing data from pyvene/axbench-concept500..."
python axbench/data/prepare_data.py \
    --dump_dir "$DUMP_DIR" \
    --hf_subdir 2b/l20 \
    --layer 15

echo "Setup complete."
