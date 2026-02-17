# Using HuggingFace data for concept detection evaluation

The `pyvene/axbench-concept500` HuggingFace dataset contains pre-generated evaluation data for 500 concepts. You can use it directly instead of running the full `generate.py` pipeline.

## Data preparation

```bash
python slurm/prepare_data.py \
    --dump_dir results/my_experiment \
    --hf_subdir 2b/l20 \
    --layer 20 \
    --component res
```

Available `--hf_subdir` options correspond to model/layer combos in the HF repo (e.g. `2b/l10`, `2b/l20`, `9b/l20`, `9b/l31`).

This creates the directory structure expected by `inference.py`:
```
results/my_experiment/
├── generate/
│   ├── metadata.jsonl          # concept descriptions (from parquet)
│   └── train_data.parquet      # training split
├── train/
│   └── config.json             # layer/component config
└── inference/
    └── latent_eval_data.parquet  # test split for evaluation
```

## Running inference

```bash
torchrun --nproc_per_node=1 --master_port=29500 axbench/scripts/inference.py \
    --config axbench/sweep/your_config.yaml \
    --mode latent \
    --dump_dir results/my_experiment \
    --overwrite_inference_data_dir results/my_experiment/inference
```

## Important: metadata alignment

The `metadata.jsonl` **must** be built from the parquet's `output_concept` column, not from the neuronpedia JSON files. The HF dataset uses a specific curated set of 500 concepts whose `concept_id` values do not correspond to the first 500 entries of the neuronpedia JSON. Using the wrong metadata will silently cause every concept label to be mismatched with the evaluation examples.

`prepare_data.py` handles this correctly by extracting concept descriptions directly from the parquet data.
