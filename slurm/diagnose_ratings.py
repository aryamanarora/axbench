"""Quick diagnostic: show raw completions from both rating models on a few examples.

Run on a node with 2 GPUs:
  python slurm/diagnose_ratings.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "latentqa"))

import re
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig

# ── Shared ──
RATING_TEMPLATE = (
    "Is the following concept present in the text? Concept: '{concept}'. "
    "Answer with a rating from 0 to 2, where 0 means the concept is not present at all, "
    "1 means the concept is somewhat present, and 2 means the concept is strongly present. "
    "Provide your rating using this exact format: Rating: [[score]]."
)
DATA_PATH = "results/ao_detection/inference/latent_eval_data.parquet"


def parse_rating(completion):
    try:
        if "Rating:" in completion:
            rt = completion.split("Rating:")[-1].strip().split("\n")[0].strip()
            rt = rt.replace("[", "").replace("]", "").strip('"').strip("'").strip("*").strip()
            r = float(rt)
            if 0 <= r <= 2:
                return r
        nums = re.findall(r"\b([012](?:\.\d+)?)\b", completion)
        if nums:
            return float(nums[-1])
        return -1
    except:
        return -1


def test_activation_oracle():
    """Test AO Rating on a few examples."""
    from axbench.models.activation_oracle import (
        _collect_activations, _get_model_layers, _make_steering_hook,
        _get_introspection_prefix, SPECIAL_TOKEN,
    )

    print("\n" + "="*70)
    print("ACTIVATION ORACLE RATING")
    print("="*70)

    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    oracle_lora = "adamkarvonen/checkpoints_latentqa_cls_past_lens_Llama-3_1-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()

    # Wrap with dummy + oracle adapter
    dummy_config = LoraConfig(r=1, lora_alpha=1, target_modules=["q_proj"], bias="none")
    model = PeftModel(model, dummy_config, adapter_name="base_default")
    model.load_adapter(oracle_lora, adapter_name="activation_oracle")

    extraction_layer = 16  # 50% of 32
    injection_layer = 1

    df = pd.read_parquet(DATA_PATH)

    # Pick 3 positive and 3 negative from concept 0
    concept_df = df[df.concept_id == 0]
    concept_name = concept_df.output_concept.iloc[0]
    question = RATING_TEMPLATE.format(concept=concept_name)

    for cat in ["positive", "negative"]:
        samples = concept_df[concept_df.category == cat].head(3)
        for _, row in samples.iterrows():
            text = row.get("output", row.get("input", ""))

            # Extract activations (base model)
            model.disable_adapter_layers()
            inputs = tokenizer(text, return_tensors="pt", truncation=True).to("cuda:0")
            activations = _collect_activations(model, extraction_layer, inputs.input_ids)
            act_vecs = activations[0]  # [seq_len, D]

            # Build oracle prompt
            model.enable_adapter_layers()
            model.set_adapter("activation_oracle")

            prefix = _get_introspection_prefix(extraction_layer, act_vecs.shape[0])
            user_msg = prefix + question
            prompt_str = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                tokenize=False, add_generation_prompt=True,
            )
            oracle_inputs = tokenizer(prompt_str, return_tensors="pt").to("cuda:0")

            # Find ? positions
            sp_id = tokenizer.encode(SPECIAL_TOKEN, add_special_tokens=False)
            ids_list = oracle_inputs.input_ids[0].tolist()
            if len(sp_id) == 1:
                positions = [i for i, t in enumerate(ids_list) if t == sp_id[0]]
            else:
                positions = []
                for i in range(len(ids_list) - len(sp_id) + 1):
                    if ids_list[i:i+len(sp_id)] == sp_id:
                        positions.append(i)

            n = min(len(positions), act_vecs.shape[0])
            hook_fn = _make_steering_hook(
                [act_vecs[:n]], [positions[:n]], 1.0)
            layers = _get_model_layers(model)
            handle = layers[injection_layer].register_forward_hook(hook_fn)

            try:
                out = model.generate(**oracle_inputs, max_new_tokens=50, do_sample=False)
            finally:
                handle.remove()

            model.set_adapter("base_default")

            completion = tokenizer.decode(out[0, oracle_inputs.input_ids.shape[1]:], skip_special_tokens=True)
            rating = parse_rating(completion)

            print(f"\n[{cat.upper()}] text: {text[:100]}...")
            print(f"  completion: {completion!r}")
            print(f"  rating: {rating}")

    del model
    torch.cuda.empty_cache()


def test_latentqa():
    """Test LQA Rating on a few examples."""
    from lit.utils.activation_utils import latent_qa
    from lit.utils.dataset_utils import lqa_tokenize, BASE_DIALOG, ENCODER_CHAT_TEMPLATES
    from lit.utils.infra_utils import get_model as lqa_get_model, get_tokenizer as lqa_get_tokenizer

    print("\n" + "="*70)
    print("LATENTQA RATING")
    print("="*70)

    target_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    decoder_name = "aypan17/latentqa_llama-3-8b-instruct"

    tokenizer = AutoTokenizer.from_pretrained(target_name, model_max_length=1024)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    target = AutoModelForCausalLM.from_pretrained(
        target_name, torch_dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()
    target.resize_token_embeddings(len(tokenizer))

    lqa_tok = lqa_get_tokenizer(target_name)
    decoder = lqa_get_model(
        model_name=target_name, tokenizer=lqa_tok,
        load_peft_checkpoint=decoder_name, device="cuda:1",
    ).eval()
    if target.get_input_embeddings().weight.shape[0] != decoder.get_input_embeddings().weight.shape[0]:
        decoder.resize_token_embeddings(target.get_input_embeddings().weight.shape[0])

    # Hooks
    def get_layers(m):
        for p in ["model.layers", "model.model.layers"]:
            obj = m
            try:
                for a in p.split("."):
                    obj = getattr(obj, a)
                return obj
            except AttributeError:
                continue

    target_layers = get_layers(target)
    decoder_layers = get_layers(decoder)
    module_read = [target_layers[15]]
    module_write = [decoder_layers[0]]

    chat_template = ENCODER_CHAT_TEMPLATES.get(tokenizer.name_or_path, None)

    df = pd.read_parquet(DATA_PATH)
    concept_df = df[df.concept_id == 0]
    concept_name = concept_df.output_concept.iloc[0]
    question = RATING_TEMPLATE.format(concept=concept_name)

    for cat in ["positive", "negative"]:
        samples = concept_df[concept_df.category == cat].head(3)
        for _, row in samples.iterrows():
            user_text = row.get("input", "")
            assistant_text = row.get("output", "")

            read_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text},
                 {"role": "assistant", "content": assistant_text}],
                tokenize=False, add_generation_prompt=False,
                chat_template=chat_template,
            )
            dialog = BASE_DIALOG + [{"role": "user", "content": question}]
            probe_data = [{"read_prompt": read_prompt, "dialog": dialog}]

            batch = lqa_tokenize(
                probe_data, tokenizer, name=target_name,
                generate=True, mask_type=None, mask_all_but_last=True,
                modify_chat_template=True,
            )
            with torch.no_grad():
                out = latent_qa(
                    batch, target, decoder, module_read, module_write,
                    tokenizer, shift_position_ids=False, generate=True,
                    max_new_tokens=50, no_grad=True,
                )
            num_tokens = batch["tokenized_write"]["input_ids"][0].shape[0]
            completion = tokenizer.decode(out[0][num_tokens:], skip_special_tokens=True)
            rating = parse_rating(completion)

            print(f"\n[{cat.upper()}] text: {assistant_text[:100]}...")
            print(f"  completion: {completion!r}")
            print(f"  rating: {rating}")


if __name__ == "__main__":
    test_activation_oracle()
    test_latentqa()
