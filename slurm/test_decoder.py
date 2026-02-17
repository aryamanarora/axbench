"""Interactive test script for LatentQA decoder with concept500 data.

Run on a node with 2 GPUs:
  python slurm/test_decoder.py

Loads a random row from concept500, feeds input/output through the target model,
then lets you ask arbitrary questions to the decoder about the activations.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "latentqa"))

import random
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lit.utils.activation_utils import latent_qa
from lit.utils.dataset_utils import lqa_tokenize, BASE_DIALOG, ENCODER_CHAT_TEMPLATES
from lit.utils.infra_utils import get_model as lqa_get_model, get_tokenizer as lqa_get_tokenizer

TARGET_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DECODER_MODEL = "aypan17/latentqa_llama-3-8b-instruct"
LAYER = 15
DATA_PATH = "results/latentqa_detection/inference/latent_eval_data.parquet"


def get_model_layers_str(model):
    for path in ["model.layers", "model.model.layers"]:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return path
        except AttributeError:
            continue
    raise RuntimeError("Cannot find model layers")


def get_layer(model, path, idx):
    obj = model
    for attr in path.split("."):
        obj = getattr(obj, attr)
    return obj[idx]


def main():
    # Load concept500 data
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded {len(df)} rows, {df.concept_id.nunique()} concepts")

    print("Loading target model...")
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL, model_max_length=1024)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    target = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()
    target.resize_token_embeddings(len(tokenizer))

    print("Loading decoder model...")
    lqa_tok = lqa_get_tokenizer(TARGET_MODEL)
    decoder = lqa_get_model(
        model_name=TARGET_MODEL,
        tokenizer=lqa_tok,
        load_peft_checkpoint=DECODER_MODEL,
        device="cuda:1",
    ).eval()
    if target.get_input_embeddings().weight.shape[0] != decoder.get_input_embeddings().weight.shape[0]:
        decoder.resize_token_embeddings(target.get_input_embeddings().weight.shape[0])

    # Setup hooks
    target_path = get_model_layers_str(target)
    decoder_path = get_model_layers_str(decoder)
    module_read = [get_layer(target, target_path, LAYER)]
    module_write = [get_layer(decoder, decoder_path, 0)]

    chat_template = ENCODER_CHAT_TEMPLATES.get(tokenizer.name_or_path, None)

    print("\n=== LatentQA Decoder Test (concept500) ===")
    print("Commands:")
    print("  'next' or Enter  — pick a new random row")
    print("  'pos'            — pick a random positive row")
    print("  'neg'            — pick a random negative row")
    print("  'concept N'      — pick from concept N")
    print("  'quit'           — exit")
    print("  anything else    — ask that question to the decoder\n")

    current_row = None
    current_read_prompt = None

    def pick_row(subset=None):
        nonlocal current_row, current_read_prompt
        if subset is None:
            subset = df
        current_row = subset.sample(1).iloc[0]
        user_text = current_row.get("input", "")
        assistant_text = current_row.get("output", "")
        # Format as user+assistant conversation
        current_read_prompt = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ],
            tokenize=False,
            add_generation_prompt=False,
            chat_template=chat_template,
        )
        print(f"\n{'='*60}")
        print(f"Concept {current_row['concept_id']}: {current_row.get('output_concept', 'N/A')}")
        print(f"Category: {current_row.get('category', 'N/A')}")
        print(f"Input:  {user_text[:120]}...")
        print(f"Output: {assistant_text[:120]}...")
        print(f"{'='*60}\n")

    def ask_question(question):
        dialog = BASE_DIALOG + [{"role": "user", "content": question}]
        probe_data = [{"read_prompt": current_read_prompt, "dialog": dialog}]

        # Open-ended generation
        print("\n--- Generation ---")
        batch_gen = lqa_tokenize(
            probe_data, tokenizer, name=TARGET_MODEL,
            generate=True, mask_type=None, mask_all_but_last=True,
            modify_chat_template=True,
        )
        with torch.no_grad():
            out = latent_qa(
                batch_gen, target, decoder, module_read, module_write,
                tokenizer, shift_position_ids=False, generate=True,
                max_new_tokens=100, no_grad=True,
            )
        num_tokens = batch_gen["tokenized_write"]["input_ids"][0].shape[0]
        completion = tokenizer.decode(out[0][num_tokens:], skip_special_tokens=True)
        print(f"Decoder: {completion}")

        # Logits
        print("\n--- Top 20 logits ---")
        batch_fwd = lqa_tokenize(
            probe_data, tokenizer, name=TARGET_MODEL,
            generate=True, mask_type=None, mask_all_but_last=True,
            modify_chat_template=True,
        )
        batch_fwd["tokenized_write"]["labels"] = batch_fwd["tokenized_write"]["input_ids"].clone()
        with torch.no_grad():
            out_fwd = latent_qa(
                batch_fwd, target, decoder, module_read, module_write,
                tokenizer, shift_position_ids=False, generate=False,
                no_grad=True,
            )
        logits = out_fwd.logits[0]
        attn_mask = batch_fwd["tokenized_write"]["attention_mask"][0].to(logits.device)
        last_pos = attn_mask.sum() - 1
        last_logits = logits[last_pos]

        top_vals, top_ids = torch.topk(last_logits, 20)
        print(f"{'Token':<20} {'ID':>8} {'Logit':>10} {'Prob':>10}")
        probs = torch.softmax(top_vals, dim=0)
        for val, tid, prob in zip(top_vals, top_ids, probs):
            tok_str = tokenizer.decode([tid.item()])
            print(f"{repr(tok_str):<20} {tid.item():>8} {val.item():>10.3f} {prob.item():>10.4f}")

        # Yes/No
        print("\n--- Yes/No logits ---")
        for label in ["Yes", "No", " Yes", " No"]:
            tid = tokenizer.encode(label, add_special_tokens=False)[0]
            logit_val = last_logits[tid].item()
            print(f"{repr(label):<8} (id={tid:>6}): logit={logit_val:.3f}")
        print()

    # Start with a random row
    pick_row()

    while True:
        cmd = input("Question (or command): ").strip()
        if cmd.lower() == "quit":
            break
        elif cmd.lower() in ("next", ""):
            pick_row()
        elif cmd.lower() == "pos":
            pick_row(df[df.category == "positive"])
        elif cmd.lower() == "neg":
            pick_row(df[df.category == "negative"])
        elif cmd.lower().startswith("concept "):
            try:
                cid = int(cmd.split()[1])
                pick_row(df[df.concept_id == cid])
            except (ValueError, IndexError):
                print("Usage: concept <number>")
        else:
            ask_question(cmd)


if __name__ == "__main__":
    main()
