"""
Activation Oracle integration for AxBench.

Implements concept detection using activation oracles — LoRA-finetuned LLMs
that interpret activations from a target model by receiving them as additive
steering vectors injected at an early layer.

Based on: https://github.com/adamkarvonen/activation_oracles
Paper: https://arxiv.org/abs/2512.15674

The oracle is the *same* base LLM architecture finetuned with LoRA.  During
inference the target model's hidden-state activations (extracted at some
percentage of depth) are injected into the oracle at layer 1 via additive
steering hooks, and the oracle generates a natural-language answer about
what it "sees" in those activations.

No dependency on the activation_oracles repo is required — the core
hook/steering logic is implemented here directly.

Install requirements: pip install peft bitsandbytes
"""
import os
import re

import numpy as np
import torch
from tqdm.auto import tqdm

from .model import BaseModel

import logging
logging.basicConfig(
    format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN,
)
logger = logging.getLogger(__name__)


# ── Prompt templates ─────────────────────────────────────────────────────────
CONCEPT_DETECTION_QUESTION_TEMPLATE_RATING = (
    "Is the following concept present in the text? Concept: '{concept}'. "
    "Answer with a rating from 0 to 2, where 0 means the concept is not present at all, "
    "1 means the concept is somewhat present, and 2 means the concept is strongly present. "
    "Provide your rating using this exact format: Rating: [[score]]."
)

CONCEPT_DETECTION_QUESTION_TEMPLATE = (
    "Is the assistant thinking about the concept '{concept}'?"
)

# The activation_oracles library marks injection positions with " ?" tokens.
SPECIAL_TOKEN = " ?"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_model_layers(model):
    """Return the ``nn.ModuleList`` of transformer layers, handling common HF
    architectures (Gemma, Llama, Qwen, Mistral) and PEFT wrappers."""
    for path_parts in [
        ["model", "layers"],
        ["model", "model", "layers"],
        ["base_model", "model", "model", "layers"],
        ["language_model", "model", "layers"],
        ["module", "model", "model", "layers"],
    ]:
        obj = model
        try:
            for attr in path_parts:
                obj = getattr(obj, attr)
            # Quick sanity check: it should be subscriptable
            _ = obj[0]
            return obj
        except (AttributeError, IndexError, TypeError):
            continue
    raise RuntimeError(
        "Cannot locate model transformer layers.  "
        "Supported architectures: Llama, Gemma-2/3, Qwen, Mistral."
    )


def _layer_percent_to_index(model, percent):
    """Convert a layer-depth percentage (e.g. 50) to an absolute index."""
    if hasattr(model, "config"):
        num_layers = model.config.num_hidden_layers
    else:
        num_layers = len(_get_model_layers(model))
    return int(num_layers * percent / 100)


@torch.no_grad()
def _collect_activations(model, layer_idx, input_ids, attention_mask=None):
    """Extract residual-stream activations at *layer_idx* via a forward hook.

    Returns a tensor of shape ``[batch, seq_len, hidden_dim]``.
    """
    layers = _get_model_layers(model)
    target_layer = layers[layer_idx]

    cache = []

    def _hook(module, inp, out):
        # Most HF models return (hidden_states, …)
        hidden = out[0] if isinstance(out, tuple) else out
        cache.append(hidden.detach())

    handle = target_layer.register_forward_hook(_hook)
    try:
        kwargs = {"input_ids": input_ids}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        model(**kwargs)
    finally:
        handle.remove()

    if not cache:
        raise RuntimeError(f"No activations captured at layer {layer_idx}")
    return cache[0]


def _make_steering_hook(vectors_list, positions_list, coefficient=1.0):
    """Return a forward-hook that injects activation vectors via the
    normalised-additive formula used by activation_oracles:

        steered = original + normalise(vec) * ‖original‖ * coefficient

    Args:
        vectors_list: length-B list of tensors, each ``[K_b, D]``.
        positions_list: length-B list of int lists, each length ``K_b``.
        coefficient: multiplicative strength of injection.
    """
    def hook_fn(module, inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        seq_len = hidden.shape[1]

        for b in range(min(len(vectors_list), hidden.shape[0])):
            vecs = vectors_list[b]          # [K, D]
            positions = positions_list[b]    # list[int]
            for k, p in enumerate(positions):
                # During generation with KV-cache, seq_len == 1 after prefill
                # so prompt positions are out of bounds — skip them.
                if p >= seq_len:
                    continue
                vec = vecs[k].to(hidden.device, hidden.dtype)
                vec_norm = vec.norm()
                if vec_norm > 0:
                    orig_norm = hidden[b, p].norm()
                    hidden[b, p] = (
                        hidden[b, p]
                        + (vec / vec_norm) * orig_norm * coefficient
                    )

        if isinstance(out, tuple):
            return (hidden,) + out[1:]
        return hidden

    return hook_fn


def _get_introspection_prefix(layer, num_positions):
    """Build the activation_oracles introspection prefix.

    Format::

        Layer: <layer>
         ? ? ? ? ?
    """
    prefix = f"Layer: {layer}\n"
    prefix += SPECIAL_TOKEN * num_positions
    prefix += " \n"
    return prefix


# ── Model class ──────────────────────────────────────────────────────────────

class ActivationOracleReading(BaseModel):
    """Activation Oracle concept-detection for AxBench.

    **No training required.**  Uses a pre-trained oracle LoRA adapter
    (e.g. ``adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B``)
    loaded onto the same base model as the target.

    Flow for each example:

    1. Run the *target model* (base weights, oracle LoRA disabled) on the
       input text and extract residual-stream activations at
       ``layer_percent`` depth.
    2. Build an introspection prompt whose ``" ?"`` placeholder tokens mark
       where activations will be injected.
    3. Switch to the *oracle adapter*, register a steering hook at
       ``injection_layer`` (default 1), and generate a response.
    4. Parse the response for a 0-2 concept-presence rating.

    Constructor kwargs (beyond the standard ``model, tokenizer, layer``):

    ==================== ====================================================
    ``oracle_lora_path``  HuggingFace repo or local path to the oracle LoRA.
    ``target_model_name`` Model name (for ``apply_chat_template``).
    ``layer_percent``     Activation extraction depth as % (default 50).
    ``injection_layer``   Layer in the oracle to inject at (default 1).
    ``steering_coefficient``  Injection strength (default 1.0).
    ``max_new_tokens``    Max tokens to generate per question (default 50).
    ==================== ====================================================
    """

    def __init__(self, model, tokenizer, layer=15, training_args=None, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        self.training_args = training_args
        self.device = kwargs.get("device", "cuda:0")
        self.seed = kwargs.get("seed", 42)

        # Oracle config — auto-detect LoRA path from model name if not provided
        _ORACLE_LORA_MAP = {
            "meta-llama/Llama-3.1-8B-Instruct": "adamkarvonen/checkpoints_latentqa_cls_past_lens_Llama-3_1-8B-Instruct",
            "meta-llama/Meta-Llama-3-8B-Instruct": "adamkarvonen/checkpoints_latentqa_cls_past_lens_Llama-3_1-8B-Instruct",
            "Qwen/Qwen3-8B": "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B",
        }
        model_name_str = getattr(tokenizer, "name_or_path", "")
        default_lora = _ORACLE_LORA_MAP.get(
            model_name_str,
            "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B",
        )
        self.oracle_lora_path = kwargs.get("oracle_lora_path", default_lora)
        self.target_model_name = kwargs.get(
            "target_model_name", model_name_str or "Qwen/Qwen3-8B")
        self.layer_percent = kwargs.get("layer_percent", 50)
        self.injection_layer = kwargs.get("injection_layer", 1)
        self.steering_coefficient = kwargs.get("steering_coefficient", 1.0)
        self.max_new_tokens = kwargs.get("max_new_tokens", 50)

        # Internal state (populated by load())
        self._extraction_layer = None
        self._oracle_adapter_name = None
        self._original_model = model  # keep reference to unwrapped model
        self._special_token_id = None

    # ── BaseModel interface ──────────────────────────────────────────────

    def __str__(self):
        return "ActivationOracleReading"

    def make_model(self, **kwargs):
        pass

    def save(self, dump_dir, **kwargs):
        pass

    def train(self, examples, **kwargs):
        pass

    def load(self, dump_dir=None, **kwargs):
        """Load the oracle LoRA adapter onto the target model.

        Converts ``self.model`` to a ``PeftModel`` (if needed) and loads the
        oracle adapter so that we can switch between base-model inference
        (for activation extraction) and oracle inference (for generation).
        """
        if self._oracle_adapter_name is not None:
            return  # already loaded

        from peft import PeftModel, LoraConfig

        # Compute absolute extraction layer
        self._extraction_layer = _layer_percent_to_index(
            self.model, self.layer_percent)
        logger.warning(
            f"Extraction layer: {self._extraction_layer} "
            f"({self.layer_percent}% of model depth)")

        # Make model a PeftModel if it isn't already
        if not isinstance(self.model, PeftModel):
            dummy_config = LoraConfig(
                r=1,
                lora_alpha=1,
                target_modules=["q_proj"],
                bias="none",
            )
            self.model = PeftModel(
                self.model, dummy_config, adapter_name="base_default")

        # Load the oracle LoRA adapter
        adapter_name = "activation_oracle"
        self.model.load_adapter(self.oracle_lora_path, adapter_name=adapter_name)
        self._oracle_adapter_name = adapter_name
        logger.warning(
            f"Loaded oracle adapter '{adapter_name}' from "
            f"{self.oracle_lora_path}")

        # Tokenizer setup for batched generation
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Pre-compute the token id(s) for the special " ?" token
        self._special_token_id = self.tokenizer.encode(
            SPECIAL_TOKEN, add_special_tokens=False)

        # Pre-compute Yes/No token ids
        self._yes_token_id = self.tokenizer.encode(
            "Yes", add_special_tokens=False)[0]
        self._no_token_id = self.tokenizer.encode(
            "No", add_special_tokens=False)[0]
        logger.warning(
            f"Yes token: {self._yes_token_id}, No token: {self._no_token_id}")

    # ── Private helpers ──────────────────────────────────────────────────

    def _find_special_token_positions(self, input_ids_1D):
        """Find positions of the ``" ?"`` placeholder tokens in a 1-D id tensor."""
        ids_list = input_ids_1D.tolist()
        sp = self._special_token_id

        if len(sp) == 1:
            target = sp[0]
            return [i for i, tok in enumerate(ids_list) if tok == target]

        # Multi-token case: scan for contiguous matches
        positions = []
        for i in range(len(ids_list) - len(sp) + 1):
            if ids_list[i : i + len(sp)] == sp:
                positions.append(i)
        return positions

    @staticmethod
    def _get_rating_from_completion(completion):
        """Parse a 0-2 rating from the oracle's free-text completion."""
        try:
            if "Rating:" in completion:
                rating_text = completion.split("Rating:")[-1].strip()
                rating_text = rating_text.split("\n")[0].strip()
                rating_text = (
                    rating_text.replace("[", "")
                    .replace("]", "")
                    .strip('"')
                    .strip("'")
                    .strip("*")
                    .strip()
                )
                rating = float(rating_text)
                if 0 <= rating <= 2:
                    return rating
            # Fallback: look for any isolated 0/1/2
            numbers = re.findall(r"\b([012](?:\.\d+)?)\b", completion)
            if numbers:
                return float(numbers[-1])
            logger.warning(
                f"Cannot find rating in completion: {completion[:200]}")
            return -1
        except (ValueError, IndexError) as e:
            logger.error(
                f"Error parsing rating: {completion[:200]}. Error: {e}")
            return -1

    # ── Core inference ───────────────────────────────────────────────────

    @torch.no_grad()
    def predict_latent(self, examples, **kwargs):
        """Run activation-oracle concept detection.

        Returns ``{"max_act": [rating_per_example]}``.
        """
        self.model.eval()
        concept = kwargs.get("concept", "")
        batch_size = kwargs.get("batch_size", 4)

        question_text = CONCEPT_DETECTION_QUESTION_TEMPLATE.format(
            concept=concept)

        all_max_act = []

        for i in tqdm(
            range(0, len(examples), batch_size),
            desc="ActivationOracle Reading",
        ):
            batch_examples = examples.iloc[i : i + batch_size]

            # ── 1. Extract activations from target (base) model ──────
            texts = []
            for _, row in batch_examples.iterrows():
                texts.append(row.get("output", row.get("input", "")))

            # Disable oracle adapter so we run the vanilla base model
            self.model.disable_adapter_layers()

            target_inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

            activations = _collect_activations(
                self.model,
                self._extraction_layer,
                target_inputs.input_ids,
                target_inputs.attention_mask,
            )
            # activations: [batch, seq_len, hidden_dim]

            # Re-enable adapters and select the oracle
            self.model.enable_adapter_layers()
            self.model.set_adapter(self._oracle_adapter_name)

            # ── 2. Build oracle prompts with introspection prefix ────
            oracle_prompts = []
            per_example_vecs = []

            for b in range(len(texts)):
                # Number of non-padding tokens for this example
                if target_inputs.attention_mask is not None:
                    num_valid = int(
                        target_inputs.attention_mask[b].sum().item())
                else:
                    num_valid = target_inputs.input_ids.shape[1]

                # Activation vectors for valid (non-padding) positions
                # Left-padded → valid tokens are the *last* num_valid
                act_vecs = activations[b, -num_valid:]  # [num_valid, D]
                per_example_vecs.append(act_vecs)

                # Build the full user message
                prefix = _get_introspection_prefix(
                    self._extraction_layer, num_valid)
                user_msg = prefix + question_text

                prompt_str = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_msg}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                oracle_prompts.append(prompt_str)

            # Tokenize oracle prompts
            oracle_inputs = self.tokenizer(
                oracle_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

            # ── 3. Locate " ?" positions & align with act vectors ────
            all_positions = []
            all_vectors = []

            for b in range(oracle_inputs.input_ids.shape[0]):
                positions = self._find_special_token_positions(
                    oracle_inputs.input_ids[b])
                vecs = per_example_vecs[b]

                # Align lengths (positions may differ from vecs due to
                # tokenization details)
                n = min(len(positions), vecs.shape[0])
                all_positions.append(positions[:n])
                all_vectors.append(vecs[:n])

            # ── 4. Forward pass with steering hook at injection layer ──
            layers = _get_model_layers(self.model)
            injection_module = layers[self.injection_layer]

            hook_fn = _make_steering_hook(
                all_vectors, all_positions, self.steering_coefficient)
            hook_handle = injection_module.register_forward_hook(hook_fn)

            try:
                outputs = self.model(**oracle_inputs)
            finally:
                hook_handle.remove()

            # Switch back to base default (for next iteration's act extraction)
            self.model.set_adapter("base_default")

            # ── 5. Extract Yes/No logits at last position ─────────
            logits = outputs.logits  # [batch, seq_len, vocab]
            attn_mask = oracle_inputs.attention_mask
            last_pos = attn_mask.sum(dim=1) - 1  # [batch]

            for b in range(logits.shape[0]):
                last_logits = logits[b, last_pos[b]]
                yes_logit = last_logits[self._yes_token_id].item()
                no_logit = last_logits[self._no_token_id].item()
                score = torch.softmax(
                    torch.tensor([yes_logit, no_logit]), dim=0)[0].item()
                all_max_act.append(score)

            torch.cuda.empty_cache()

        return {"max_act": all_max_act}

    def predict_latents(self, examples, **kwargs):
        return self.predict_latent(examples, **kwargs)

    def pre_compute_mean_activations(self, dump_dir, **kwargs):
        return {}

    def to(self, device):
        self.device = device
        return self


class ActivationOracleReadingRating(ActivationOracleReading):
    """Activation Oracle using 0-2 rating generation instead of Yes/No logits."""

    def __str__(self):
        return "ActivationOracleReadingRating"

    @torch.no_grad()
    def predict_latent(self, examples, **kwargs):
        self.model.eval()
        concept = kwargs.get("concept", "")
        batch_size = kwargs.get("batch_size", 4)

        question_text = CONCEPT_DETECTION_QUESTION_TEMPLATE_RATING.format(
            concept=concept)

        all_max_act = []

        for i in tqdm(
            range(0, len(examples), batch_size),
            desc="ActivationOracle Rating",
        ):
            batch_examples = examples.iloc[i : i + batch_size]

            texts = []
            for _, row in batch_examples.iterrows():
                texts.append(row.get("output", row.get("input", "")))

            self.model.disable_adapter_layers()

            target_inputs = self.tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True,
            ).to(self.device)

            activations = _collect_activations(
                self.model, self._extraction_layer,
                target_inputs.input_ids, target_inputs.attention_mask,
            )

            self.model.enable_adapter_layers()
            self.model.set_adapter(self._oracle_adapter_name)

            oracle_prompts = []
            per_example_vecs = []

            for b in range(len(texts)):
                if target_inputs.attention_mask is not None:
                    num_valid = int(target_inputs.attention_mask[b].sum().item())
                else:
                    num_valid = target_inputs.input_ids.shape[1]

                act_vecs = activations[b, -num_valid:]
                per_example_vecs.append(act_vecs)

                prefix = _get_introspection_prefix(
                    self._extraction_layer, num_valid)
                user_msg = prefix + question_text

                prompt_str = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_msg}],
                    tokenize=False, add_generation_prompt=True,
                )
                oracle_prompts.append(prompt_str)

            oracle_inputs = self.tokenizer(
                oracle_prompts, return_tensors="pt", padding=True, truncation=True,
            ).to(self.device)

            all_positions = []
            all_vectors = []

            for b in range(oracle_inputs.input_ids.shape[0]):
                positions = self._find_special_token_positions(
                    oracle_inputs.input_ids[b])
                vecs = per_example_vecs[b]
                n = min(len(positions), vecs.shape[0])
                all_positions.append(positions[:n])
                all_vectors.append(vecs[:n])

            layers = _get_model_layers(self.model)
            injection_module = layers[self.injection_layer]

            hook_fn = _make_steering_hook(
                all_vectors, all_positions, self.steering_coefficient)
            hook_handle = injection_module.register_forward_hook(hook_fn)

            try:
                outputs = self.model.generate(
                    **oracle_inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            finally:
                hook_handle.remove()

            self.model.set_adapter("base_default")

            prompt_len = oracle_inputs.input_ids.shape[1]
            for b in range(outputs.shape[0]):
                completion_ids = outputs[b, prompt_len:]
                completion = self.tokenizer.decode(
                    completion_ids, skip_special_tokens=True)
                rating = self._get_rating_from_completion(completion)
                all_max_act.append(rating)

            torch.cuda.empty_cache()

        return {"max_act": all_max_act}
