"""Download pyvene/axbench-concept500 from HuggingFace and prepare it
for LatentQA detection inference on Llama-3-8B.

Creates the directory structure expected by inference.py:
  {dump_dir}/generate/metadata.jsonl
  {dump_dir}/train/config.json
  {dump_dir}/inference/latent_eval_data.parquet   (for overwrite_inference_data_dir)
"""
import os, json, argparse
import pandas as pd
from huggingface_hub import hf_hub_download

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_dir", type=str, required=True)
    parser.add_argument("--concept_path", type=str, required=True,
                        help="Path to neuronpedia JSON (e.g. gemma-2-2b_20-gemmascope-res-16k.json)")
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--max_concepts", type=int, default=500)
    args = parser.parse_args()

    dump_dir = args.dump_dir
    os.makedirs(f"{dump_dir}/generate", exist_ok=True)
    os.makedirs(f"{dump_dir}/train", exist_ok=True)
    os.makedirs(f"{dump_dir}/inference", exist_ok=True)

    # 1. Load concepts from neuronpedia JSON to build metadata
    print("Loading concepts from", args.concept_path)
    with open(args.concept_path) as f:
        json_concepts = json.load(f)

    seen_index = set()
    concepts = []
    refs = []
    for c in json_concepts:
        subspace_id = c["index"]
        if subspace_id in seen_index:
            continue
        seen_index.add(subspace_id)
        concepts.append(c["description"].strip())
        refs.append(f"https://www.neuronpedia.org/{c['modelId']}/{c['layer']}/{subspace_id}")
        if len(concepts) >= args.max_concepts:
            break

    # 2. Download parquet files directly from HF
    # Structure: {model}/{layer}/{split}/data.parquet
    # Use 2b/l20 to match gemma-2-2b_20-gemmascope-res-16k.json
    hf_subdir = "2b/l20"
    print(f"Downloading parquet files from pyvene/axbench-concept500/{hf_subdir}...")
    test_path = hf_hub_download(
        repo_id="pyvene/axbench-concept500",
        filename=f"{hf_subdir}/test/data.parquet",
        repo_type="dataset",
    )
    train_path = hf_hub_download(
        repo_id="pyvene/axbench-concept500",
        filename=f"{hf_subdir}/train/data.parquet",
        repo_type="dataset",
    )
    test_df = pd.read_parquet(test_path)
    train_df = pd.read_parquet(train_path)
    print(f"Loaded test: {len(test_df)} rows, train: {len(train_df)} rows")
    print(f"Test columns: {list(test_df.columns)}")
    print(f"Train columns: {list(train_df.columns)}")

    # 3. Build metadata.jsonl
    print("Building metadata.jsonl...")
    metadata_path = f"{dump_dir}/generate/metadata.jsonl"
    with open(metadata_path, "w") as f:
        for concept_id in range(len(concepts)):
            concept = concepts[concept_id]
            ref = refs[concept_id]
            concept_rows = test_df[test_df["concept_id"] == concept_id]
            if len(concept_rows) > 0:
                genres = concept_rows["concept_genre"].unique().tolist()
            else:
                genres = ["text"]
            entry = {
                "concept_id": concept_id,
                "concept": concept,
                "ref": ref,
                "concept_genres_map": {concept: genres},
            }
            f.write(json.dumps(entry) + "\n")
    print(f"Wrote {len(concepts)} concepts to {metadata_path}")

    # 4. Create config.json
    config = {"layer": args.layer}
    config_path = f"{dump_dir}/train/config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
    print(f"Wrote config to {config_path}")

    # 5. Save test split as latent_eval_data.parquet
    eval_df = test_df.copy()
    if "sae_link" not in eval_df.columns:
        eval_df["sae_link"] = eval_df["concept_id"].apply(
            lambda cid: refs[cid] if 0 <= cid < len(refs) else "")
    if "sae_id" not in eval_df.columns:
        eval_df["sae_id"] = eval_df["concept_id"]
    eval_path = f"{dump_dir}/inference/latent_eval_data.parquet"
    eval_df.to_parquet(eval_path, index=False)
    print(f"Wrote {len(eval_df)} rows to {eval_path}")

    # 6. Save train data
    if "sae_link" not in train_df.columns:
        train_df["sae_link"] = train_df["concept_id"].apply(
            lambda cid: refs[cid] if 0 <= cid < len(refs) else "")
    if "sae_id" not in train_df.columns:
        train_df["sae_id"] = train_df["concept_id"]
    train_out = f"{dump_dir}/generate/train_data.parquet"
    train_df.to_parquet(train_out, index=False)
    print(f"Wrote {len(train_df)} rows to {train_out}")

    print("Data preparation complete!")


if __name__ == "__main__":
    main()
