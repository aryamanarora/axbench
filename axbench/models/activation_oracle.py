"""
Activation Oracle integration for AxBench.

Implements concept detection using activation oracles — LoRA-finetuned LLMs
that interpret activations from a target model by receiving them as additive
steering vectors injected at an early layer.

Based on: https://github.com/adamkarvonen/activation_oracles
Paper: https://arxiv.org/abs/2512.15674

Requires the activation_oracles repo:
  pip install activation_oracles
  Or: pip install 'axbench[activation_oracles]'
"""
import os
import re
import sys

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


# ── Optional dependency: activation_oracles (nl_probes) ──────────────────────
# Add local clone to sys.path if present: axbench/models/_activation_oracles/
_ao_local = os.path.join(os.path.dirname(__file__), "_activation_oracles")
if os.path.isdir(_ao_local) and _ao_local not in sys.path:
    sys.path.insert(0, _ao_local)

try:
    from nl_probes.utils.activation_utils import collect_activations, get_hf_submodule
    from nl_probes.utils.steering_hooks import get_hf_activation_steering_hook, add_hook
    from nl_probes.utils.dataset_utils import (
        SPECIAL_TOKEN, get_introspection_prefix, find_pattern_in_tokens,
    )
    from nl_probes.utils.common import layer_percent_to_layer
    _HAS_AO = True
except ImportError:
    _HAS_AO = False


def _require_ao():
    if not _HAS_AO:
        raise ImportError(
            "activation_oracles is not installed. Clone into axbench/models/:\n"
            "  cd axbench/models && git clone https://github.com/adamkarvonen/activation_oracles.git _activation_oracles\n"
            "Or add the repo to PYTHONPATH."
        )


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
        _require_ao()
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        self.training_args = training_args
        self.device = kwargs.get("device", "cuda:0")
        self.seed = kwargs.get("seed", 42)

        # Oracle config — auto-detect LoRA path from model name if not provided
        _ORACLE_LORA_MAP = {
            "meta-llama/Llama-3.1-8B-Instruct": "adamkarvonen/checkpoints_latentqa_cls_past_lens_Llama-3_1-8B-Instruct",
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
        self._original_model = model

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
        """Load the oracle LoRA adapter onto the target model."""
        if self._oracle_adapter_name is not None:
            return  # already loaded

        from peft import PeftModel, LoraConfig

        # Compute absolute extraction layer
        self._extraction_layer = layer_percent_to_layer(
            self.target_model_name, self.layer_percent)
        logger.warning(
            f"Extraction layer: {self._extraction_layer} "
            f"({self.layer_percent}% of model depth)")

        # Make model a PeftModel if it isn't already
        if not isinstance(self.model, PeftModel):
            dummy_config = LoraConfig(
                r=1, lora_alpha=1, target_modules=["q_proj"], bias="none",
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

        # Pre-compute Yes/No token ids
        self._yes_token_id = self.tokenizer.encode(
            "Yes", add_special_tokens=False)[0]
        self._no_token_id = self.tokenizer.encode(
            "No", add_special_tokens=False)[0]
        logger.warning(
            f"Yes token: {self._yes_token_id}, No token: {self._no_token_id}")

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
        use_lora = isinstance(self.model, __import__('peft').PeftModel)

        for i in tqdm(
            range(0, len(examples), batch_size),
            desc="ActivationOracle Reading",
        ):
            batch_examples = examples.iloc[i : i + batch_size]

            # ── 1. Extract activations from target (base) model ──────
            texts = []
            for _, row in batch_examples.iterrows():
                texts.append(row.get("output", row.get("input", "")))

            self.model.disable_adapter_layers()

            target_inputs = self.tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True,
            ).to(self.device)

            extraction_submodule = get_hf_submodule(
                self.model, self._extraction_layer, use_lora=use_lora)
            activations = collect_activations(
                self.model, extraction_submodule, target_inputs)

            self.model.enable_adapter_layers()
            self.model.set_adapter(self._oracle_adapter_name)

            # ── 2. Build oracle prompts with introspection prefix ────
            oracle_prompts = []
            per_example_vecs = []

            for b in range(len(texts)):
                if target_inputs.attention_mask is not None:
                    num_valid = int(
                        target_inputs.attention_mask[b].sum().item())
                else:
                    num_valid = target_inputs.input_ids.shape[1]

                act_vecs = activations[b, -num_valid:]
                per_example_vecs.append(act_vecs)

                prefix = get_introspection_prefix(
                    self._extraction_layer, num_valid)
                user_msg = prefix + question_text

                prompt_str = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_msg}],
                    tokenize=False, add_generation_prompt=True,
                )
                oracle_prompts.append(prompt_str)

            oracle_inputs = self.tokenizer(
                oracle_prompts, return_tensors="pt",
                padding=True, truncation=True,
            ).to(self.device)

            # ── 3. Locate " ?" positions & build steering vectors ────
            all_positions = []
            all_vectors = []

            for b in range(oracle_inputs.input_ids.shape[0]):
                num_valid = per_example_vecs[b].shape[0]
                positions = find_pattern_in_tokens(
                    oracle_inputs.input_ids[b].tolist(),
                    SPECIAL_TOKEN, num_valid, self.tokenizer)
                vecs = per_example_vecs[b]
                n = min(len(positions), vecs.shape[0])
                all_positions.append(positions[:n])
                all_vectors.append(vecs[:n])

            # ── 4. Forward pass with steering hook at injection layer ──
            injection_submodule = get_hf_submodule(
                self.model, self.injection_layer, use_lora=use_lora)
            hook_fn = get_hf_activation_steering_hook(
                all_vectors, all_positions,
                self.steering_coefficient,
                device=self.device,
                dtype=activations.dtype,
            )

            with add_hook(injection_submodule, hook_fn):
                outputs = self.model(**oracle_inputs)

            self.model.set_adapter("base_default")

            # ── 5. Extract Yes/No logits at last position ─────────
            logits = outputs.logits
            attn_mask = oracle_inputs.attention_mask
            last_pos = attn_mask.sum(dim=1) - 1

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

    @torch.no_grad()
    def predict_latent(self, examples, **kwargs):
        self.model.eval()
        concept = kwargs.get("concept", "")
        batch_size = kwargs.get("batch_size", 4)

        question_text = CONCEPT_DETECTION_QUESTION_TEMPLATE_RATING.format(
            concept=concept)

        all_max_act = []
        use_lora = isinstance(self.model, __import__('peft').PeftModel)

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

            extraction_submodule = get_hf_submodule(
                self.model, self._extraction_layer, use_lora=use_lora)
            activations = collect_activations(
                self.model, extraction_submodule, target_inputs)

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

                prefix = get_introspection_prefix(
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
                num_valid = per_example_vecs[b].shape[0]
                positions = find_pattern_in_tokens(
                    oracle_inputs.input_ids[b].tolist(),
                    SPECIAL_TOKEN, num_valid, self.tokenizer)
                vecs = per_example_vecs[b]
                n = min(len(positions), vecs.shape[0])
                all_positions.append(positions[:n])
                all_vectors.append(vecs[:n])

            injection_submodule = get_hf_submodule(
                self.model, self.injection_layer, use_lora=use_lora)
            hook_fn = get_hf_activation_steering_hook(
                all_vectors, all_positions,
                self.steering_coefficient,
                device=self.device,
                dtype=activations.dtype,
            )

            with add_hook(injection_submodule, hook_fn):
                outputs = self.model.generate(
                    **oracle_inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )

            self.model.set_adapter("base_default")

            prompt_len = oracle_inputs.input_ids.shape[1]
            for b in range(outputs.shape[0]):
                completion_ids = outputs[b, prompt_len:]
                completion = self.tokenizer.decode(
                    completion_ids, skip_special_tokens=True)
                rating = self._get_rating_from_completion(completion)
                if len(all_max_act) < 20:
                    logger.warning(
                        f"[AO Rating] text={texts[b][:80]}... "
                        f"completion={completion!r} rating={rating}")
                all_max_act.append(rating)

            torch.cuda.empty_cache()

        return {"max_act": all_max_act}


def _make_rating_labels(input_ids, tokenizer):
    """Build labels that only supervise the rating token after '[['.

    Given "Rating: [[2]]", we want loss only on predicting "2" (the token
    right after "[["). Everything else is masked with -100.
    """
    labels = torch.full_like(input_ids, -100)
    bracket_ids = tokenizer.encode("[[", add_special_tokens=False)
    for b in range(input_ids.shape[0]):
        ids = input_ids[b].tolist()
        for pos in range(len(ids) - len(bracket_ids)):
            if ids[pos:pos+len(bracket_ids)] == bracket_ids:
                target_pos = pos + len(bracket_ids) - 1  # last token of "[["
                if target_pos + 1 < len(ids):
                    labels[b, target_pos] = ids[target_pos + 1]  # predict "2"
                break
    return labels


# ── Gradient-preserving steering hook ────────────────────────────────────────
def _get_gradient_preserving_steering_hook(
    all_vectors, all_positions, steering_coefficient, device, dtype,
):
    """Like get_hf_activation_steering_hook but WITHOUT .detach() so gradients flow.

    Builds an additive delta tensor (non-in-place) so autograd can track
    through from the injected vectors back to their source.
    """
    def hook_fn(module, input, output):
        out_tensor = output[0] if isinstance(output, tuple) else output
        batch_size, seq_len, hidden_dim = out_tensor.shape
        # Build a delta tensor that's zero everywhere except at injection positions
        # This keeps the computation graph intact for gradient flow
        delta = torch.zeros_like(out_tensor)
        for b in range(batch_size):
            if b >= len(all_vectors) or b >= len(all_positions):
                continue
            vecs = all_vectors[b].to(device=device, dtype=dtype)
            positions = all_positions[b]
            n = min(len(positions), vecs.shape[0])
            for j in range(n):
                pos = positions[j]
                if pos < seq_len:
                    # Non-in-place: delta is zeros, so this scatter is safe
                    # But we need vecs[j] in the graph, so use addition
                    delta[b, pos] = delta[b, pos] + steering_coefficient * vecs[j]
        new_out = out_tensor + delta
        if isinstance(output, tuple):
            return (new_out,) + output[1:]
        return new_out
    return hook_fn


class ActivationOracleGradientSteering(BaseModel):
    """Gradient-based steering using activation oracle loss.

    Computes d(oracle_loss) / d(target_activations) to get a steering
    direction, then stores it in self.ax.proj.weight for compatibility
    with the standard axbench save/load/predict_steer pipeline.

    Flow:
    1. train(): For each concept, extract activations at extraction_layer (with grad),
       build oracle prompt with introspection prefix + "Rating: [[2]]" target,
       inject via gradient-preserving hook, forward through oracle, backprop,
       collect gradient on activations, average → steering vector.
    2. predict_steer(): Uses pyvene IntervenableModel + AdditionIntervention
       at extraction_layer.
    """

    def __init__(self, model, tokenizer, layer=15, training_args=None, **kwargs):
        _require_ao()
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        self.training_args = training_args
        self.max_activations = {}
        self.device = kwargs.get("device", "cuda:0")
        self.seed = kwargs.get("seed", 42)
        self.steering_layers = kwargs.get("steering_layers", None)
        self.num_of_layers = len(self.steering_layers) if self.steering_layers else 1
        self.dump_dir = kwargs.get("dump_dir", None)
        self.use_wandb = kwargs.get("use_wandb", False)

        # Oracle config
        _ORACLE_LORA_MAP = {
            "meta-llama/Llama-3.1-8B-Instruct": "adamkarvonen/checkpoints_latentqa_cls_past_lens_Llama-3_1-8B-Instruct",
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
        self.gradient_batch_size = kwargs.get("gradient_batch_size", 4)

        # Internal state
        self._extraction_layer = None
        self._oracle_adapter_name = None

    def __str__(self):
        return "ActivationOracleGradientSteering"

    def make_model(self, **kwargs):
        from .mean import LogisticRegressionModel
        from .interventions import AdditionIntervention
        from pyvene import IntervenableConfig, IntervenableModel

        mode = kwargs.get("mode", "train")
        if mode == "steering":
            ax = AdditionIntervention(
                embed_dim=self.model.config.hidden_size,
                low_rank_dimension=kwargs.get("low_rank_dimension", 1),
            )
            self.ax = ax
            self.ax.train()
            ax_config = IntervenableConfig(representations=[{
                "layer": l,
                "component": f"model.layers[{l}].output",
                "low_rank_dimension": kwargs.get("low_rank_dimension", 1),
                "intervention": self.ax} for l in [self.layer]])
            ax_model = IntervenableModel(ax_config, self.model)
            ax_model.set_device(self.device)
            self.ax_model = ax_model
        else:
            ax = LogisticRegressionModel(
                self.model.config.hidden_size, kwargs.get("low_rank_dimension", 1))
            ax.to(self.device)
            self.ax = ax

    def _load_oracle(self):
        """Load oracle LoRA adapter onto the target model."""
        if self._oracle_adapter_name is not None:
            return

        from peft import PeftModel, LoraConfig

        self._extraction_layer = layer_percent_to_layer(
            self.target_model_name, self.layer_percent)
        logger.warning(
            f"Extraction layer: {self._extraction_layer} "
            f"({self.layer_percent}% of model depth)")

        if not isinstance(self.model, __import__('peft').PeftModel):
            dummy_config = LoraConfig(
                r=1, lora_alpha=1, target_modules=["q_proj"], bias="none",
            )
            self.model = __import__('peft').PeftModel(
                self.model, dummy_config, adapter_name="base_default")

        adapter_name = "activation_oracle"
        self.model.load_adapter(self.oracle_lora_path, adapter_name=adapter_name)
        self._oracle_adapter_name = adapter_name
        logger.warning(
            f"Loaded oracle adapter '{adapter_name}' from {self.oracle_lora_path}")

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def save(self, dump_dir, **kwargs):
        from pathlib import Path
        dump_dir = Path(dump_dir)
        model_name = kwargs.get("model_name", self.__str__())
        weight_file = dump_dir / f"{model_name}_weight.pt"
        weight = self.ax.proj.weight.data.cpu()
        if weight_file.exists():
            weight = torch.cat([torch.load(weight_file), weight], dim=0)
        torch.save(weight, weight_file)

        bias_file = dump_dir / f"{model_name}_bias.pt"
        bias = self.ax.proj.bias.data.cpu()
        if bias_file.exists():
            bias = torch.cat([torch.load(bias_file), bias], dim=0)
        torch.save(bias, bias_file)

    def load(self, dump_dir=None, **kwargs):
        if dump_dir is None:
            return
        model_name = kwargs.get("model_name", self.__str__())
        print(f"Loading {model_name} from {dump_dir}.")
        weight = torch.load(
            f"{dump_dir}/{model_name}_weight.pt",
            map_location=torch.device("cpu"),
        )
        bias = torch.load(
            f"{dump_dir}/{model_name}_bias.pt",
            map_location=torch.device("cpu"),
        )
        kwargs["low_rank_dimension"] = weight.shape[0]
        self.make_model(**kwargs)
        self.ax.proj.weight.data = weight.to(self.device)
        self.ax.proj.bias.data = bias.to(self.device)

    def _compute_steering_vector(self, examples, concept):
        """Compute steering vector via oracle-loss gradients on target activations."""
        self._load_oracle()
        use_lora = isinstance(self.model, __import__('peft').PeftModel)

        question_text = CONCEPT_DETECTION_QUESTION_TEMPLATE_RATING.format(concept=concept)
        target_answer = "Rating: [[2]]"

        batch_size = self.gradient_batch_size
        all_grads = []

        texts = []
        for _, row in examples.iterrows():
            texts.append(row.get("output", row.get("input", "")))

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # 1. Extract activations from target (base) model WITH gradients
            self.model.disable_adapter_layers()

            target_inputs = self.tokenizer(
                batch_texts, return_tensors="pt", padding=True, truncation=True,
            ).to(self.device)

            extraction_submodule = get_hf_submodule(
                self.model, self._extraction_layer, use_lora=use_lora)
            with torch.no_grad():
                activations = collect_activations(
                    self.model, extraction_submodule, target_inputs)
            # Detach and make a grad-tracking leaf so the hook's addition
            # creates a gradient path: loss → hook output → activations.grad
            activations = activations.detach().requires_grad_(True)

            self.model.enable_adapter_layers()
            self.model.set_adapter(self._oracle_adapter_name)

            # 2. Build oracle prompts
            oracle_prompts = []
            per_example_vecs = []

            for b in range(len(batch_texts)):
                if target_inputs.attention_mask is not None:
                    num_valid = int(target_inputs.attention_mask[b].sum().item())
                else:
                    num_valid = target_inputs.input_ids.shape[1]

                act_vecs = activations[b, -num_valid:]
                per_example_vecs.append(act_vecs)

                prefix = get_introspection_prefix(self._extraction_layer, num_valid)
                user_msg = prefix + question_text

                prompt_str = self.tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": target_answer},
                    ],
                    tokenize=False, add_generation_prompt=False,
                )
                oracle_prompts.append(prompt_str)

            oracle_inputs = self.tokenizer(
                oracle_prompts, return_tensors="pt", padding=True, truncation=True,
            ).to(self.device)

            # 3. Find injection positions and build steering vectors
            all_positions = []
            all_vectors = []

            for b in range(oracle_inputs.input_ids.shape[0]):
                num_valid = per_example_vecs[b].shape[0]
                positions = find_pattern_in_tokens(
                    oracle_inputs.input_ids[b].tolist(),
                    SPECIAL_TOKEN, num_valid, self.tokenizer)
                vecs = per_example_vecs[b]
                n = min(len(positions), vecs.shape[0])
                all_positions.append(positions[:n])
                all_vectors.append(vecs[:n])

            # 4. Forward with gradient-preserving hook
            injection_submodule = get_hf_submodule(
                self.model, self.injection_layer, use_lora=use_lora)
            hook_fn = _get_gradient_preserving_steering_hook(
                all_vectors, all_positions,
                self.steering_coefficient,
                device=self.device,
                dtype=activations.dtype,
            )

            hook_handle = injection_submodule.register_forward_hook(hook_fn)

            labels = _make_rating_labels(oracle_inputs.input_ids, self.tokenizer)
            outputs = self.model(**oracle_inputs, labels=labels)

            hook_handle.remove()

            loss = outputs.loss
            loss.backward()

            # 5. Collect gradient from activations
            if activations.grad is not None:
                avg_grad = -activations.grad.mean(dim=(0, 1)).float().cpu()
                all_grads.append(avg_grad)
                logger.warning(
                    f"  batch {i//batch_size}: loss={loss.item():.4f} "
                    f"grad_norm={avg_grad.norm():.6f}")
            else:
                logger.warning(f"  batch {i//batch_size}: no gradient on activations")

            self.model.set_adapter("base_default")
            self.model.zero_grad()
            torch.cuda.empty_cache()

        if not all_grads:
            logger.error(f"No gradients collected for concept '{concept}'")
            return torch.zeros(self.model.config.hidden_size)

        steering_vector = torch.stack(all_grads).mean(dim=0)
        logger.warning(f"Steering vector norm: {steering_vector.norm():.4f}")
        return steering_vector

    def train(self, examples, **kwargs):
        concept = kwargs.get("concept", "")

        if not hasattr(self, 'ax'):
            from .mean import LogisticRegressionModel
            self.ax = LogisticRegressionModel(
                self.model.config.hidden_size, 1)
            self.ax.to(self.device)

        logger.warning(f"Computing oracle gradient steering vector for concept: {concept}")

        import numpy as np
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        steering_vector = self._compute_steering_vector(examples, concept)

        self.ax.proj.weight.data = steering_vector.unsqueeze(0).to(self.device)
        self.ax.proj.bias.data = torch.zeros(1).to(self.device)

    @torch.no_grad()
    def predict_steer(self, examples, **kwargs):
        self.ax.eval()
        self.tokenizer.padding_side = "left"

        batch_size = kwargs.get("batch_size", 64)
        eval_output_length = kwargs.get("eval_output_length", 128)
        temperature = kwargs.get("temperature", 1.0)
        all_generations = []
        all_strengths = []
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        progress_bar = tqdm(range(0, len(examples), batch_size), position=rank, leave=True)
        for i in range(0, len(examples), batch_size):
            batch_examples = examples.iloc[i:i+batch_size]
            input_strings = batch_examples['input'].tolist()
            mag = torch.tensor(batch_examples['factor'].tolist()).to(self.device)
            idx = torch.tensor(batch_examples["concept_id"].tolist()).to(self.device)
            max_acts = torch.tensor([
                self.max_activations.get(id, 1.0)
                for id in batch_examples["concept_id"].tolist()]).to(self.device)
            inputs = self.tokenizer(
                input_strings, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            _, generations = self.ax_model.generate(
                inputs,
                unit_locations=None, intervene_on_prompt=True,
                subspaces=[{"idx": idx, "mag": mag, "max_act": max_acts,
                            "prefix_length": kwargs["prefix_length"]}]*self.num_of_layers,
                max_new_tokens=eval_output_length, do_sample=True,
                temperature=temperature,
            )
            input_lengths = [len(input_ids) for input_ids in inputs.input_ids]
            generated_texts = [
                self.tokenizer.decode(generation[input_length:], skip_special_tokens=True)
                for generation, input_length in zip(generations, input_lengths)
            ]
            all_generations += generated_texts
            all_strengths.extend((mag*max_acts).tolist())
            progress_bar.update(1)

        return {
            "steered_generation": all_generations,
            "strength": all_strengths,
        }

    def pre_compute_mean_activations(self, dump_dir, **kwargs):
        import os, pandas as pd
        max_activations = {}
        for file in os.listdir(dump_dir):
            if file.startswith("latent_") and file.endswith(".parquet"):
                latent_path = os.path.join(dump_dir, file)
                latent = pd.read_parquet(latent_path)
                for concept_id in sorted(latent["concept_id"].unique()):
                    concept_latent = latent[latent["concept_id"] == concept_id]
                    max_act = concept_latent[f"{self.__str__()}_max_act"].max()
                    max_activations[concept_id] = max_act if max_act > 0 else 50
        self.max_activations = max_activations
        return max_activations

    def to(self, device):
        self.device = device
        if hasattr(self, 'ax'):
            self.ax = self.ax.to(device)
            if hasattr(self, 'ax_model'):
                from pyvene import IntervenableModel
                if isinstance(self.ax_model, IntervenableModel):
                    self.ax_model.set_device(device)
                else:
                    self.ax_model = self.ax_model.to(device)
        return self


class ActivationOracleActivationSteering(BaseModel):
    """Per-instance activation optimization using activation oracle loss.

    Instead of computing a reusable steering vector, this optimizes activations
    per-instance via gradient descent on the oracle loss. Produces stronger
    steering than gradient vectors but is not reusable across examples.

    Flow (per example in predict_steer):
    1. Run target model on input → capture activations at extraction_layer
    2. Clone activations as optimizable parameter
    3. For num_steps iterations: inject via gradient-preserving hook → oracle loss → update
    4. Inject final delta at extraction_layer during generation (prefill only)
    """

    def __init__(self, model, tokenizer, layer=15, training_args=None, **kwargs):
        _require_ao()
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        self.training_args = training_args
        self.max_activations = {}
        self.device = kwargs.get("device", "cuda:0")
        self.seed = kwargs.get("seed", 42)

        # Oracle config
        _ORACLE_LORA_MAP = {
            "meta-llama/Llama-3.1-8B-Instruct": "adamkarvonen/checkpoints_latentqa_cls_past_lens_Llama-3_1-8B-Instruct",
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

        # Optimization config
        self.num_steps = kwargs.get("num_steps", 20)
        self.step_size = kwargs.get("step_size", 0.5)

        # Metadata for concept lookup
        self.metadata = kwargs.get("metadata", None)

        # Internal state
        self._extraction_layer = None
        self._oracle_adapter_name = None

    def __str__(self):
        return "ActivationOracleActivationSteering"

    def make_model(self, **kwargs):
        pass

    def _load_oracle(self):
        """Load oracle LoRA adapter onto the target model."""
        if self._oracle_adapter_name is not None:
            return

        from peft import PeftModel, LoraConfig

        self._extraction_layer = layer_percent_to_layer(
            self.target_model_name, self.layer_percent)
        logger.warning(
            f"Extraction layer: {self._extraction_layer} "
            f"({self.layer_percent}% of model depth)")

        if not isinstance(self.model, __import__('peft').PeftModel):
            dummy_config = LoraConfig(
                r=1, lora_alpha=1, target_modules=["q_proj"], bias="none",
            )
            self.model = __import__('peft').PeftModel(
                self.model, dummy_config, adapter_name="base_default")

        adapter_name = "activation_oracle"
        self.model.load_adapter(self.oracle_lora_path, adapter_name=adapter_name)
        self._oracle_adapter_name = adapter_name
        logger.warning(
            f"Loaded oracle adapter '{adapter_name}' from {self.oracle_lora_path}")

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def save(self, dump_dir, **kwargs):
        pass  # nothing to persist

    def train(self, examples, **kwargs):
        pass  # all work happens at inference time

    def load(self, dump_dir=None, **kwargs):
        """Load the oracle adapter (no steering vectors to load)."""
        self._load_oracle()

    def _optimize_activations(self, input_text, concept, factor):
        """Optimize activations for a single example via oracle loss gradient descent."""
        use_lora = isinstance(self.model, __import__('peft').PeftModel)

        question_text = CONCEPT_DETECTION_QUESTION_TEMPLATE_RATING.format(concept=concept)
        target_answer = "Rating: [[2]]"

        # 1. Extract original activations (base weights, no grad)
        self.model.disable_adapter_layers()

        target_inputs = self.tokenizer(
            input_text, return_tensors="pt", padding=True, truncation=True,
        ).to(self.device)

        extraction_submodule = get_hf_submodule(
            self.model, self._extraction_layer, use_lora=use_lora)

        with torch.no_grad():
            activations = collect_activations(
                self.model, extraction_submodule, target_inputs)

        self.model.enable_adapter_layers()
        self.model.set_adapter(self._oracle_adapter_name)

        # Get valid token count
        if target_inputs.attention_mask is not None:
            num_valid = int(target_inputs.attention_mask[0].sum().item())
        else:
            num_valid = target_inputs.input_ids.shape[1]

        original_vecs = activations[0, -num_valid:]  # (num_valid, hidden_dim)

        # Build oracle prompt (with answer for loss)
        prefix = get_introspection_prefix(self._extraction_layer, num_valid)
        user_msg = prefix + question_text
        prompt_str = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": target_answer},
            ],
            tokenize=False, add_generation_prompt=False,
        )
        oracle_inputs = self.tokenizer(
            prompt_str, return_tensors="pt", padding=True, truncation=True,
        ).to(self.device)

        # Find injection positions
        positions = find_pattern_in_tokens(
            oracle_inputs.input_ids[0].tolist(),
            SPECIAL_TOKEN, num_valid, self.tokenizer)
        n = min(len(positions), original_vecs.shape[0])
        positions = positions[:n]

        # 2. Clone as optimizable
        opt_vecs = original_vecs[:n].clone().detach().requires_grad_(True)

        injection_submodule = get_hf_submodule(
            self.model, self.injection_layer, use_lora=use_lora)

        # 3. Iterative optimization
        for step in range(self.num_steps):
            hook_fn = _get_gradient_preserving_steering_hook(
                [opt_vecs], [positions],
                self.steering_coefficient,
                device=self.device,
                dtype=opt_vecs.dtype,
            )

            hook_handle = injection_submodule.register_forward_hook(hook_fn)

            labels = _make_rating_labels(oracle_inputs.input_ids, self.tokenizer)
            outputs = self.model(**oracle_inputs, labels=labels)

            hook_handle.remove()

            loss = outputs.loss
            loss.backward()

            if opt_vecs.grad is not None:
                with torch.no_grad():
                    opt_vecs = (opt_vecs - self.step_size * opt_vecs.grad).detach().requires_grad_(True)

                if step < 3 or step == self.num_steps - 1:
                    logger.warning(
                        f"  step {step}: loss={loss.item():.4f} "
                        f"delta_norm={(opt_vecs - original_vecs[:n]).norm():.4f}")
            else:
                logger.warning(f"  step {step}: no gradient on opt_vecs")

            self.model.zero_grad()

        self.model.set_adapter("base_default")

        # Compute delta scaled by factor
        delta = (opt_vecs - original_vecs[:n]).detach()
        return delta * factor, num_valid

    def predict_steer(self, examples, **kwargs):
        """Generate steered text by optimizing activations per-example."""
        self._load_oracle()
        self.model.eval()
        self.tokenizer.padding_side = "left"
        use_lora = isinstance(self.model, __import__('peft').PeftModel)

        eval_output_length = kwargs.get("eval_output_length", 128)
        temperature = kwargs.get("temperature", 1.0)

        all_generations = []
        all_strengths = []

        # Get concept name from metadata
        concept_ids = examples["concept_id"].unique()
        concept_id = concept_ids[0] if len(concept_ids) == 1 else kwargs.get("concept_id", 0)
        if self.metadata is not None and concept_id < len(self.metadata):
            concept = self.metadata[concept_id]["concept"]
        else:
            concept = ""
            logger.warning("No metadata available, using empty concept name")

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        progress_bar = tqdm(range(len(examples)), position=rank, leave=True,
                           desc="AO Activation Steering")

        for idx in range(len(examples)):
            row = examples.iloc[idx]
            input_text = row["input"]
            factor_val = row["factor"]

            logger.warning(
                f"Optimizing activations for example {idx+1}/{len(examples)} "
                f"(concept={concept!r}, factor={factor_val})")

            # Optimize activations
            result = self._optimize_activations(input_text, concept, factor_val)
            if result is None:
                all_generations.append("")
                all_strengths.append(factor_val)
                progress_bar.update(1)
                continue

            delta, num_valid = result

            # Generate with delta injected at extraction_layer (prefill only)
            self.model.disable_adapter_layers()
            extraction_submodule = get_hf_submodule(
                self.model, self._extraction_layer, use_lora=use_lora)

            prefill_done = [False]

            def generation_hook(module, input, output, _delta=delta, _num_valid=num_valid):
                if prefill_done[0]:
                    return output
                out_tensor = output[0] if isinstance(output, tuple) else output
                if out_tensor.shape[1] > 1:
                    prefill_done[0] = True
                    out_tensor = out_tensor.clone()
                    # Add delta to last num_valid positions
                    seq_len = out_tensor.shape[1]
                    n = min(_delta.shape[0], _num_valid, seq_len)
                    out_tensor[0, -n:, :] += _delta[:n].to(out_tensor.device, out_tensor.dtype)
                    if isinstance(output, tuple):
                        return (out_tensor,) + output[1:]
                    return out_tensor
                return output

            hook_handle = extraction_submodule.register_forward_hook(generation_hook)

            inputs = self.tokenizer(
                input_text, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)

            with torch.no_grad():
                generations = self.model.generate(
                    **inputs,
                    max_new_tokens=eval_output_length,
                    do_sample=True,
                    temperature=temperature,
                )

            hook_handle.remove()
            self.model.enable_adapter_layers()
            self.model.set_adapter("base_default")

            input_length = len(inputs.input_ids[0])
            generated_text = self.tokenizer.decode(
                generations[0][input_length:], skip_special_tokens=True)
            all_generations.append(generated_text)
            all_strengths.append(factor_val)

            progress_bar.update(1)
            torch.cuda.empty_cache()

        return {
            "steered_generation": all_generations,
            "strength": all_strengths,
        }

    def pre_compute_mean_activations(self, dump_dir, **kwargs):
        return {}

    def to(self, device):
        self.device = device
        return self
