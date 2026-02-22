"""
LatentQA integration for AxBench.

Implements three model classes:
- LatentQAReading: Uses LatentQA's reading mode for concept detection (latent inference).
- LatentQASteering: Uses LatentQA's control mode (LoRA) for steering (steering inference).
- LatentQAGradientSteering: Uses decoder-loss gradients on target activations as
  steering vectors. Cheaper than LoRA — no per-concept training, just one forward+backward.

Requires the LatentQA repo (https://github.com/aypan17/latentqa) to be installed.
"""
import json
import os
import sys
import re

import numpy as np
import torch
from tqdm.auto import tqdm

from .model import Model, BaseModel

import logging
logging.basicConfig(
    format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN,
)
logger = logging.getLogger(__name__)


# Yes/No question template for concept detection.
# We measure logits for "Yes" vs "No" tokens from the decoder.
CONCEPT_DETECTION_QUESTION_TEMPLATE = (
    "Is the assistant thinking about the concept '{concept}'?"
)

CONCEPT_DETECTION_QUESTION_TEMPLATE_RATING = (
    "Is the following concept present in the text? Concept: '{concept}'. "
    "Answer with a rating from 0 to 2, where 0 means the concept is not present at all, "
    "1 means the concept is somewhat present, and 2 means the concept is strongly present. "
    "Provide your rating using this exact format: Rating: [[score]]."
)


try:
    from lit.utils.activation_utils import latent_qa as _latent_qa
    from lit.utils.dataset_utils import BASE_DIALOG as _BASE_DIALOG, ENCODER_CHAT_TEMPLATES as _ENCODER_CHAT_TEMPLATES
    from lit.utils.infra_utils import get_model as _lqa_get_model, get_tokenizer as _lqa_get_tokenizer
    try:
        from lit.utils.dataset_utils import lqa_tokenize as _lqa_tokenize
    except ImportError:
        from lit.utils.dataset_utils import tokenize as _lqa_tokenize
    _HAS_LATENTQA = True
except ImportError:
    _HAS_LATENTQA = False


def _require_latentqa():
    if not _HAS_LATENTQA:
        raise ImportError(
            "LatentQA is not installed. Install with:\n"
            "  pip install latentqa\n"
            "Or: pip install 'axbench[latentqa]'"
        )


def _get_model_layers_str(model):
    """Determine the correct attribute path to model layers."""
    for path in [
        "model.layers",
        "model.model.layers",
        "module.model.model.layers",
        "language_model.model.layers",
        "module.language_model.model.layers",
    ]:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return path
        except AttributeError:
            continue
    raise RuntimeError("Cannot find model layers. Unsupported model architecture.")


def _load_decoder_model(target_model_name, decoder_model_name, decoder_device):
    """Load the LatentQA decoder model (shared across model classes)."""
    _require_latentqa()

    logger.warning(f"Loading LatentQA decoder from {decoder_model_name} to {decoder_device}")
    lqa_tokenizer = _lqa_get_tokenizer(target_model_name)
    decoder_model = _lqa_get_model(
        model_name=target_model_name,
        tokenizer=lqa_tokenizer,
        load_peft_checkpoint=decoder_model_name,
        device=decoder_device,
    )
    decoder_model.eval()
    return decoder_model


def _get_modules(target_model, decoder_model, min_layer=15, max_layer=16,
                 layer_to_write=0, num_layers_to_read=1):
    """Get read/write module hooks for LatentQA.

    Returns List[List[Module]] for read and write modules.
    """
    target_path = _get_model_layers_str(target_model)
    decoder_path = _get_model_layers_str(decoder_model)

    def get_layer(model, path, idx):
        obj = model
        for attr in path.split("."):
            obj = getattr(obj, attr)
        return obj[idx]

    module_read, module_write = [], []
    for i in range(min_layer, max_layer):
        module_read_i = [get_layer(target_model, target_path, j)
                         for j in range(i, i + num_layers_to_read)]
        module_write_i = [get_layer(decoder_model, decoder_path, j)
                          for j in range(layer_to_write, layer_to_write + num_layers_to_read)]
        module_read.append(module_read_i)
        module_write.append(module_write_i)
    return module_read, module_write


class LatentQAReading(BaseModel):
    """LatentQA Reading mode for concept detection.

    Uses LatentQA's decoder to read and interpret activations from the
    target model, then determines if a concept is present.

    This is similar to PromptDetection but reads from internal activations
    rather than prompting the model directly about its output.
    """

    def __init__(self, model, tokenizer, layer=15, training_args=None, **kwargs):
        self.model = model  # target model
        self.tokenizer = tokenizer
        self.layer = layer
        self.training_args = training_args
        self.device = kwargs.get("device", "cuda:0")
        self.decoder_device = kwargs.get("decoder_device", "cuda:1")
        self.seed = kwargs.get("seed", 42)

        # LatentQA-specific config
        self.decoder_model_name = kwargs.get(
            "decoder_model_name", "aypan17/latentqa_llama-3-8b-instruct")
        self.target_model_name = kwargs.get(
            "target_model_name", "meta-llama/Meta-Llama-3-8B-Instruct")
        self.min_layer_to_read = kwargs.get("min_layer_to_read", 15)
        self.max_layer_to_read = kwargs.get("max_layer_to_read", 16)
        self.num_layers_to_read = kwargs.get("num_layers_to_read", 1)
        self.layer_to_write = kwargs.get("layer_to_write", 0)
        self.modify_chat_template = kwargs.get("modify_chat_template", True)
        self.max_new_tokens = kwargs.get("max_new_tokens", 100)

        self.decoder_model = None
        self.module_read = None
        self.module_write = None

    def __str__(self):
        return 'LatentQAReading'

    def make_model(self, **kwargs):
        pass

    def save(self, dump_dir, **kwargs):
        pass  # no training needed

    def train(self, examples, **kwargs):
        pass  # no training needed

    def load(self, dump_dir=None, **kwargs):
        """Load the LatentQA decoder model."""
        if self.decoder_model is not None:
            return  # already loaded

        self.decoder_model = _load_decoder_model(
            self.target_model_name, self.decoder_model_name, self.decoder_device)

        # Ensure decoder vocab matches target model (e.g. if PAD token was added)
        target_vocab_size = self.model.get_input_embeddings().weight.shape[0]
        decoder_vocab_size = self.decoder_model.get_input_embeddings().weight.shape[0]
        if target_vocab_size != decoder_vocab_size:
            logger.warning(
                f"Resizing decoder embeddings from {decoder_vocab_size} to {target_vocab_size}")
            self.decoder_model.resize_token_embeddings(target_vocab_size)

        # Set up read/write module hooks
        self.module_read, self.module_write = _get_modules(
            self.model, self.decoder_model,
            min_layer=self.min_layer_to_read,
            max_layer=self.max_layer_to_read,
            layer_to_write=self.layer_to_write,
            num_layers_to_read=self.num_layers_to_read,
        )

    @torch.no_grad()
    def predict_latent(self, examples, **kwargs):
        """Use LatentQA reading mode for concept detection.

        For each example, extracts activations from the target model,
        feeds them to the LatentQA decoder with a yes/no concept question,
        and uses P(Yes) - P(No) logit difference as the detection score.
        """
        _require_latentqa()

        self.model.eval()
        self.decoder_model.eval()

        concept = kwargs.get("concept", "")
        batch_size = kwargs.get("batch_size", 4)

        # Get token IDs for "Yes" and "No"
        yes_token_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
        no_token_id = self.tokenizer.encode("No", add_special_tokens=False)[0]

        question_text = CONCEPT_DETECTION_QUESTION_TEMPLATE.format(concept=concept)
        chat_template = _ENCODER_CHAT_TEMPLATES.get(self.tokenizer.name_or_path, None)

        all_max_act = []

        # LatentQA requires left padding
        orig_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        for i in tqdm(range(0, len(examples), batch_size), desc="LatentQA Reading"):
            batch_examples = examples.iloc[i:i + batch_size]

            probe_data = []
            for _, row in batch_examples.iterrows():
                user_text = row.get("input", "")
                assistant_text = row.get("output", "")
                read_prompt = self.tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": assistant_text},
                    ],
                    tokenize=False,
                    add_generation_prompt=False,
                    chat_template=chat_template,
                )
                dialog = _BASE_DIALOG + [
                    {"role": "user", "content": question_text},
                ]
                probe_data.append({
                    "read_prompt": read_prompt,
                    "dialog": dialog,
                })

            # Tokenize with generate=True to include the assistant header,
            # then do a forward pass to get logits (not model.generate).
            batch_tokenized = _lqa_tokenize(
                probe_data,
                self.tokenizer,
                name=self.target_model_name,
                generate=True,
                mask_type=None,
                mask_all_but_last=True,
                modify_chat_template=self.modify_chat_template,
            )

            # Add dummy labels so latent_qa accepts generate=False
            input_ids = batch_tokenized["tokenized_write"]["input_ids"]
            batch_tokenized["tokenized_write"]["labels"] = input_ids.clone()

            # Forward pass to get logits
            out = _latent_qa(
                batch_tokenized,
                self.model,
                self.decoder_model,
                self.module_read[0],
                self.module_write[0],
                self.tokenizer,
                shift_position_ids=False,
                generate=False,
                no_grad=True,
            )

            # Extract logits at the last non-padding position for each example
            logits = out.logits  # (batch, seq_len, vocab_size)
            attention_mask = batch_tokenized["tokenized_write"]["attention_mask"].to(logits.device)
            # Last real token position per example
            last_pos = attention_mask.sum(dim=1) - 1  # (batch,)

            for j in range(logits.shape[0]):
                last_logits = logits[j, last_pos[j]]  # (vocab_size,)
                yes_logit = last_logits[yes_token_id].item()
                no_logit = last_logits[no_token_id].item()
                # Score: P(Yes) from softmax over Yes/No
                score = torch.softmax(
                    torch.tensor([yes_logit, no_logit]), dim=0
                )[0].item()
                all_max_act.append(score)

            torch.cuda.empty_cache()

        self.tokenizer.padding_side = orig_padding_side
        return {"max_act": all_max_act}

    def predict_latents(self, examples, **kwargs):
        return self.predict_latent(examples, **kwargs)

    def pre_compute_mean_activations(self, dump_dir, **kwargs):
        return {}

    def to(self, device):
        self.device = device
        # Note: target model device is managed by AxBench infrastructure
        # decoder stays on its own device
        return self


class LatentQAReadingRating(LatentQAReading):
    """LatentQA Reading using 0-2 rating generation instead of Yes/No logits."""

    def __str__(self):
        return "LatentQAReadingRating"

    @staticmethod
    def _get_rating_from_completion(completion):
        """Parse a 0-2 rating from the decoder's free-text completion."""
        import re
        try:
            if "Rating:" in completion:
                rating_text = completion.split("Rating:")[-1].strip()
                rating_text = rating_text.split("\n")[0].strip()
                rating_text = (
                    rating_text.replace("[", "").replace("]", "")
                    .strip('"').strip("'").strip("*").strip()
                )
                rating = float(rating_text)
                if 0 <= rating <= 2:
                    return rating
            numbers = re.findall(r"\b([012](?:\.\d+)?)\b", completion)
            if numbers:
                return float(numbers[-1])
            return -1
        except (ValueError, IndexError):
            return -1

    def predict_latent(self, examples, **kwargs):
        _require_latentqa()

        self.model.eval()
        self.decoder_model.eval()

        concept = kwargs.get("concept", "")
        batch_size = kwargs.get("batch_size", 4)

        question_text = CONCEPT_DETECTION_QUESTION_TEMPLATE_RATING.format(concept=concept)
        chat_template = _ENCODER_CHAT_TEMPLATES.get(self.tokenizer.name_or_path, None)

        all_max_act = []

        orig_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        for i in tqdm(range(0, len(examples), batch_size), desc="LatentQA Rating"):
            batch_examples = examples.iloc[i:i + batch_size]

            probe_data = []
            for _, row in batch_examples.iterrows():
                user_text = row.get("input", "")
                assistant_text = row.get("output", "")
                read_prompt = self.tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": assistant_text},
                    ],
                    tokenize=False,
                    add_generation_prompt=False,
                    chat_template=chat_template,
                )
                dialog = _BASE_DIALOG + [
                    {"role": "user", "content": question_text},
                ]
                probe_data.append({
                    "read_prompt": read_prompt,
                    "dialog": dialog,
                })

            batch_tokenized = _lqa_tokenize(
                probe_data, self.tokenizer, name=self.target_model_name,
                generate=True, mask_type=None, mask_all_but_last=True,
                modify_chat_template=self.modify_chat_template,
            )

            with torch.no_grad():
                out = _latent_qa(
                    batch_tokenized, self.model, self.decoder_model,
                    self.module_read[0], self.module_write[0], self.tokenizer,
                    shift_position_ids=False, generate=True,
                    max_new_tokens=50, no_grad=True,
                )

            num_tokens = batch_tokenized["tokenized_write"]["input_ids"][0].shape[0]
            for j in range(len(batch_examples)):
                completion = self.tokenizer.decode(
                    out[j][num_tokens:], skip_special_tokens=True)
                rating = self._get_rating_from_completion(completion)
                if len(all_max_act) < 20:
                    text = batch_examples.iloc[j].get("output", "")[:80]
                    logger.warning(
                        f"[LQA Rating] text={text}... "
                        f"completion={completion!r} rating={rating}")
                all_max_act.append(rating)

            torch.cuda.empty_cache()

        self.tokenizer.padding_side = orig_padding_side
        return {"max_act": all_max_act}


class LatentQASteering(BaseModel):
    """LatentQA Control mode for steering.

    Uses LatentQA's control mechanism to steer model behavior:
    1. During train: Generate QA pairs for each concept via reading mode,
       then optimize a LoRA adapter on the target model.
    2. During predict_steer: Load LoRA-adapted model and generate steered outputs.
       The steering factor scales the LoRA delta weights.
    """

    def __init__(self, model, tokenizer, layer=15, training_args=None, **kwargs):
        self.model = model  # target model (base, not LoRA)
        self.tokenizer = tokenizer
        self.layer = layer
        self.training_args = training_args
        self.device = kwargs.get("device", "cuda:0")
        self.decoder_device = kwargs.get("decoder_device", "cuda:1")
        self.seed = kwargs.get("seed", 42)

        # LatentQA-specific config
        self.decoder_model_name = kwargs.get(
            "decoder_model_name", "aypan17/latentqa_llama-3-8b-instruct")
        self.target_model_name = kwargs.get(
            "target_model_name", "meta-llama/Meta-Llama-3-8B-Instruct")
        self.min_layer_to_read = kwargs.get("min_layer_to_read", 15)
        self.max_layer_to_read = kwargs.get("max_layer_to_read", 16)
        self.num_layers_to_read = kwargs.get("num_layers_to_read", 1)
        self.layer_to_write = kwargs.get("layer_to_write", 0)
        self.modify_chat_template = kwargs.get("modify_chat_template", True)

        # Steering LoRA config
        self.lora_r = kwargs.get("lora_r", 16)
        self.lora_alpha = kwargs.get("lora_alpha", 32)
        self.steering_lr = kwargs.get("steering_lr", 1e-4)
        self.steering_samples = kwargs.get("steering_samples", 50)
        self.steering_batch_size_train = kwargs.get("steering_batch_size_train", 1)
        self.layers_to_optimize = kwargs.get(
            "layers_to_optimize", tuple(range(16)))

        self.decoder_model = None
        self.steered_model = None
        self.max_activations = {}

    def __str__(self):
        return 'LatentQASteering'

    def make_model(self, **kwargs):
        pass

    def _load_decoder(self):
        """Load the LatentQA decoder model if not already loaded."""
        if self.decoder_model is not None:
            return
        self.decoder_model = _load_decoder_model(
            self.target_model_name, self.decoder_model_name, self.decoder_device)
        # Ensure decoder vocab matches target model (e.g. if PAD token was added)
        target_vocab_size = self.model.get_input_embeddings().weight.shape[0]
        decoder_vocab_size = self.decoder_model.get_input_embeddings().weight.shape[0]
        if target_vocab_size != decoder_vocab_size:
            logger.warning(
                f"Resizing decoder embeddings from {decoder_vocab_size} to {target_vocab_size}")
            self.decoder_model.resize_token_embeddings(target_vocab_size)

    def _generate_qa_pairs(self, concept, num_questions=15):
        """Generate QA pairs for a concept using LatentQA reading mode.

        Creates a prompt embodying the concept, runs the target model on it,
        and uses the decoder to generate descriptions of the activations.
        """
        _require_latentqa()
        chat_template = _ENCODER_CHAT_TEMPLATES.get(self.tokenizer.name_or_path, None)

        # Questions to probe the concept
        questions = [
            [f"What concept or theme is the assistant expressing?"],
            [f"How does the assistant incorporate the concept of '{concept}'?"],
            [f"Describe the assistant's behavior in terms of '{concept}'."],
            [f"What characteristics related to '{concept}' are present?"],
            [f"How strongly does '{concept}' influence the response?"],
        ]

        # Create a prompt that embodies the concept
        concept_prompt = f"You must {concept} in all your responses."
        read_prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": concept_prompt}],
            tokenize=False,
            add_generation_prompt=True,
            chat_template=chat_template,
        )

        probe_data = []
        for q in questions:
            dialog = _BASE_DIALOG + [{"role": "user", "content": q[0]}]
            probe_data.append({
                "read_prompt": read_prompt,
                "dialog": dialog,
            })

        batch = _lqa_tokenize(
            probe_data,
            self.tokenizer,
            name=self.target_model_name,
            generate=True,
            mask_type=None,
            mask_all_but_last=True,
            modify_chat_template=self.modify_chat_template,
        )

        module_read, module_write = _get_modules(
            self.model, self.decoder_model,
            min_layer=self.min_layer_to_read,
            max_layer=self.max_layer_to_read,
            layer_to_write=self.layer_to_write,
            num_layers_to_read=self.num_layers_to_read,
        )

        out = _latent_qa(
            batch,
            self.model,
            self.decoder_model,
            module_read[0],
            module_write[0],
            self.tokenizer,
            shift_position_ids=False,
            generate=True,
            max_new_tokens=100,
            no_grad=True,
        )

        qa_pairs = []
        for j in range(len(out)):
            prompt = questions[j % len(questions)][0]
            num_tokens = batch["tokenized_write"][j].shape[0]
            completion = self.tokenizer.decode(out[j][num_tokens:], skip_special_tokens=True)
            qa_pairs.append((prompt, completion))

        return qa_pairs

    def train(self, examples, **kwargs):
        """Generate QA pairs for the concept and optimize LoRA.

        For each concept, generates QA descriptions via reading mode,
        then trains a LoRA adapter to match those descriptions.
        """
        _require_latentqa()
        from peft import LoraConfig, get_peft_model

        self._load_decoder()
        concept = kwargs.get("concept", "")
        concept_id = kwargs.get("concept_id", 0)
        dump_dir = kwargs.get("dump_dir", ".")

        logger.warning(f"Training LatentQASteering for concept: {concept}")

        # Step 1: Generate QA pairs
        qa_pairs = self._generate_qa_pairs(concept)

        # Save QA pairs
        qa_path = os.path.join(dump_dir, f"latentqa_qa_concept_{concept_id}.json")
        with open(qa_path, "w") as f:
            json.dump(qa_pairs, f, indent=2)

        # Step 2: Set up LoRA on target model
        layers_to_optimize = list(self.layers_to_optimize)
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
            lora_dropout=0.1,
            inference_mode=False,
            layers_to_transform=layers_to_optimize,
        )

        steered_model = get_peft_model(self.model, lora_config)
        steered_model.to(self.device)

        module_read, module_write = _get_modules(
            steered_model, self.decoder_model,
            min_layer=self.min_layer_to_read,
            max_layer=self.max_layer_to_read,
            layer_to_write=self.layer_to_write,
            num_layers_to_read=self.num_layers_to_read,
        )

        # Step 3: Optimize LoRA to match QA pairs
        optimizer = torch.optim.Adam(steered_model.parameters(), lr=self.steering_lr)

        # Build training data from QA pairs + random prompts
        from datasets import load_dataset
        raw_data = load_dataset("databricks/databricks-dolly-15k")["train"]
        prompts = []
        for item in raw_data:
            if len(item["instruction"].split()) > 100:
                continue
            if item["context"] == "":
                prompts.append(item["instruction"])
            elif len(item["context"].split()) < 200:
                prompts.append(item["instruction"] + "\n\n" + item["context"])
            if len(prompts) >= self.steering_samples:
                break

        chat_template = _ENCODER_CHAT_TEMPLATES.get(self.tokenizer.name_or_path, None)

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        for step_i, prompt_text in enumerate(tqdm(prompts, desc="LatentQA Steering")):
            qa_idx = step_i % len(qa_pairs)
            q, a = qa_pairs[qa_idx]

            read_prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                tokenize=False,
                add_generation_prompt=True,
                chat_template=chat_template,
            )

            formatted_data = [{
                "read_prompt": read_prompt,
                "dialog": _BASE_DIALOG + [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ],
            }]

            batch = _lqa_tokenize(
                formatted_data,
                self.tokenizer,
                name=self.target_model_name,
                generate=False,
                mask_all_but_last=True,
                modify_chat_template=self.modify_chat_template,
            )

            out = _latent_qa(
                batch,
                steered_model,
                self.decoder_model,
                module_read[0],
                module_write[0],
                self.tokenizer,
                shift_position_ids=True,
                generate=False,
                cache_target_model_grad=True,
            )

            loss = out["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Save LoRA weights
        lora_path = os.path.join(dump_dir, f"latentqa_lora_concept_{concept_id}")
        steered_model.save_pretrained(lora_path)
        logger.warning(f"Saved LoRA weights to {lora_path}")

        # Clean up
        del steered_model, optimizer
        torch.cuda.empty_cache()

    def save(self, dump_dir, **kwargs):
        pass  # LoRA weights are saved during train()

    def load(self, dump_dir=None, **kwargs):
        """Load LoRA weights for a specific concept."""
        from peft import PeftModel

        concept_id = kwargs.get("concept_id", 0)
        mode = kwargs.get("mode", "steering")

        if mode == "steering" and dump_dir is not None:
            lora_path = os.path.join(str(dump_dir), f"latentqa_lora_concept_{concept_id}")
            if os.path.exists(lora_path):
                logger.warning(f"Loading LoRA from {lora_path}")
                self.steered_model = PeftModel.from_pretrained(
                    self.model, lora_path)
                self.steered_model.to(self.device)
                self.steered_model.eval()
            else:
                logger.warning(f"No LoRA weights at {lora_path}, using base model")
                self.steered_model = self.model

    @torch.no_grad()
    def predict_steer(self, examples, **kwargs):
        """Generate steered text using the LoRA-adapted model.

        The steering factor scales the LoRA delta weights.
        """
        model = self.steered_model if self.steered_model is not None else self.model
        model.eval()
        self.tokenizer.padding_side = "left"

        batch_size = kwargs.get("batch_size", 8)
        eval_output_length = kwargs.get("eval_output_length", 128)
        temperature = kwargs.get("temperature", 1.0)

        all_generations = []
        all_strengths = []

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        progress_bar = tqdm(range(0, len(examples), batch_size), position=rank, leave=True)

        for i in range(0, len(examples), batch_size):
            batch_examples = examples.iloc[i:i + batch_size]
            input_strings = batch_examples['input'].tolist()
            factors = batch_examples['factor'].tolist()

            # Scale LoRA weights by factor if we have a PeftModel
            # For factor != 1.0, we temporarily scale the LoRA
            # This is done via the model's scaling parameter
            if hasattr(model, 'peft_config'):
                for adapter_name in model.peft_config:
                    original_alpha = model.peft_config[adapter_name].lora_alpha
                    # Scale by average factor in this batch
                    avg_factor = np.mean(factors)
                    model.peft_config[adapter_name].lora_alpha = original_alpha * avg_factor

            inputs = self.tokenizer(
                input_strings, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)

            generations = model.generate(
                **inputs,
                max_new_tokens=eval_output_length,
                do_sample=True,
                temperature=temperature,
            )

            # Restore original alpha
            if hasattr(model, 'peft_config'):
                for adapter_name in model.peft_config:
                    model.peft_config[adapter_name].lora_alpha = original_alpha

            input_lengths = [len(input_ids) for input_ids in inputs.input_ids]
            generated_texts = [
                self.tokenizer.decode(generation[input_length:], skip_special_tokens=True)
                for generation, input_length in zip(generations, input_lengths)
            ]
            all_generations += generated_texts
            all_strengths.extend(factors)

            progress_bar.update(1)

        return {
            "steered_generation": all_generations,
            "strength": all_strengths,
        }

    def pre_compute_mean_activations(self, dump_dir, **kwargs):
        return {}

    def to(self, device):
        self.device = device
        if self.steered_model is not None and self.steered_model is not self.model:
            self.steered_model.to(device)
        return self


class LatentQAGradientSteering(BaseModel):
    """Gradient-based steering using LatentQA decoder loss.

    Computes d(decoder_loss) / d(target_activations) to get a steering
    direction, then stores it in self.ax.proj.weight for compatibility
    with the standard axbench save/load/predict_steer pipeline.

    Much cheaper than LatentQASteering (LoRA):
    - No per-concept LoRA training
    - Just one forward+backward through target+decoder per concept
    - Produces a steering vector that naturally supports factor scaling

    The flow:
    1. train(): For each concept, run target model on dataset examples,
       feed activations to decoder with rating question (target: "Rating: [[2]]"),
       backprop to get gradient on activations, average → steering vector.
       Store in self.ax.proj.weight for standard save/load.
    2. predict_steer(): Inherited from Model — uses IntervenableModel
       with AdditionIntervention for activation addition during generation.
    """

    def __init__(self, model, tokenizer, layer=15, training_args=None, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        self.training_args = training_args
        self.max_activations = {}
        self.device = kwargs.get("device", "cuda:0")
        self.decoder_device = kwargs.get("decoder_device", "cuda:1")
        self.seed = kwargs.get("seed", 42)
        self.steering_layers = kwargs.get("steering_layers", None)
        self.num_of_layers = len(self.steering_layers) if self.steering_layers else 1
        self.dump_dir = kwargs.get("dump_dir", None)
        self.use_wandb = kwargs.get("use_wandb", False)

        # LatentQA-specific config
        self.decoder_model_name = kwargs.get(
            "decoder_model_name", "aypan17/latentqa_llama-3-8b-instruct")
        self.target_model_name = kwargs.get(
            "target_model_name", "meta-llama/Meta-Llama-3-8B-Instruct")
        self.min_layer_to_read = kwargs.get("min_layer_to_read", 15)
        self.max_layer_to_read = kwargs.get("max_layer_to_read", 16)
        self.num_layers_to_read = kwargs.get("num_layers_to_read", 1)
        self.layer_to_write = kwargs.get("layer_to_write", 0)
        self.modify_chat_template = kwargs.get("modify_chat_template", True)
        self.gradient_batch_size = kwargs.get("gradient_batch_size", 4)

        self.decoder_model = None

    def __str__(self):
        return 'LatentQAGradientSteering'

    def make_model(self, **kwargs):
        """Set up self.ax (and self.ax_model for steering), same as MeanEmbedding."""
        from .mean import LogisticRegressionModel
        from .interventions import AdditionIntervention
        from pyvene import IntervenableConfig, IntervenableModel

        mode = kwargs.get("mode", "train")
        intervention_type = kwargs.get("intervention_type", "addition")
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

    def save(self, dump_dir, **kwargs):
        """Save steering vector via self.ax.proj.weight/bias (standard format)."""
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
        """Load steering vectors (standard _weight.pt/_bias.pt format)."""
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

    @torch.no_grad()
    def predict_steer(self, examples, **kwargs):
        """Inherited steering logic from Model using IntervenableModel."""
        self.ax.eval()
        self.tokenizer.padding_side = "left"
        concept_id_col = "concept_id"

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
                for id in batch_examples[concept_id_col].tolist()]).to(self.device)
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
        max_activations = {}
        for file in os.listdir(dump_dir):
            if file.startswith("latent_") and file.endswith(".parquet"):
                import pandas as pd
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

    def _load_decoder(self):
        """Load the LatentQA decoder model if not already loaded."""
        if self.decoder_model is not None:
            return
        self.decoder_model = _load_decoder_model(
            self.target_model_name, self.decoder_model_name, self.decoder_device)
        # Ensure decoder vocab matches target model (e.g. if PAD token was added)
        target_vocab_size = self.model.get_input_embeddings().weight.shape[0]
        decoder_vocab_size = self.decoder_model.get_input_embeddings().weight.shape[0]
        if target_vocab_size != decoder_vocab_size:
            logger.warning(
                f"Resizing decoder embeddings from {decoder_vocab_size} to {target_vocab_size}")
            self.decoder_model.resize_token_embeddings(target_vocab_size)

    def _compute_steering_vector(self, examples, concept):
        """Compute a steering vector for a concept via decoder-loss gradients.

        Uses actual dataset examples as read prompts. For each mini-batch:
        1. Run target model on examples → get activations at read layer (with grad)
        2. Feed activations to decoder with rating question, target answer = "Rating: [[2]]"
        3. Compute cross-entropy loss on the answer
        4. Backprop → gradient on target activations
        5. Average gradient across examples and sequence positions

        Returns: steering vector of shape (hidden_dim,)
        """
        _require_latentqa()
        chat_template = _ENCODER_CHAT_TEMPLATES.get(self.tokenizer.name_or_path, None)

        module_read, module_write = _get_modules(
            self.model, self.decoder_model,
            min_layer=self.min_layer_to_read,
            max_layer=self.max_layer_to_read,
            layer_to_write=self.layer_to_write,
            num_layers_to_read=self.num_layers_to_read,
        )

        # QA pair: rating question with target answer of 2 (strongly present)
        question_text = CONCEPT_DETECTION_QUESTION_TEMPLATE_RATING.format(concept=concept)
        answer_text = "Rating: [[2]]"

        # Build read prompts from actual dataset examples
        all_probe_data = []
        for _, row in examples.iterrows():
            user_text = row.get("input", "")
            assistant_text = row.get("output", "")
            read_prompt = self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text},
                ],
                tokenize=False,
                add_generation_prompt=False,
                chat_template=chat_template,
            )
            all_probe_data.append({
                "read_prompt": read_prompt,
                "dialog": _BASE_DIALOG + [
                    {"role": "user", "content": question_text},
                    {"role": "assistant", "content": answer_text},
                ],
            })

        # LatentQA requires left padding
        orig_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        # Process in mini-batches to avoid OOM (gradient computation is memory-heavy)
        batch_size = self.gradient_batch_size
        all_grads = []

        for i in range(0, len(all_probe_data), batch_size):
            probe_data = all_probe_data[i:i + batch_size]

            batch = _lqa_tokenize(
                probe_data,
                self.tokenizer,
                name=self.target_model_name,
                generate=False,
                mask_type=None,
                mask_all_but_last=True,
                modify_chat_template=self.modify_chat_template,
            )

            # Use forward hook to capture the output tensor (which retains grad
            # when cache_target_model_grad=True), then read .grad after backward.
            activation_cache = []

            def fwd_hook(module, input, output):
                out_tensor = output[0] if isinstance(output, tuple) else output
                out_tensor.retain_grad()
                activation_cache.append(out_tensor)

            hook_handles = [
                mod.register_forward_hook(fwd_hook)
                for mod in module_read[0]
            ]

            out = _latent_qa(
                batch,
                self.model,
                self.decoder_model,
                module_read[0],
                module_write[0],
                self.tokenizer,
                shift_position_ids=True,
                generate=False,
                cache_target_model_grad=True,
                no_grad=False,
            )

            loss = out.loss
            loss.backward()

            for h in hook_handles:
                h.remove()

            if activation_cache and activation_cache[0].grad is not None:
                # Average over batch and sequence dims, negate for gradient descent
                avg_grad = -activation_cache[0].grad.mean(dim=(0, 1)).float().cpu()
                all_grads.append(avg_grad)

            self.model.zero_grad()
            self.decoder_model.zero_grad()
            torch.cuda.empty_cache()

        self.tokenizer.padding_side = orig_padding_side

        if not all_grads:
            logger.error(f"No gradients collected for concept '{concept}'")
            hidden_dim = self.model.config.hidden_size
            return torch.zeros(hidden_dim)

        # Average across mini-batches
        steering_vector = torch.stack(all_grads).mean(dim=0)

        # Normalize to unit length for consistent scaling with factors
        norm = steering_vector.norm()
        if norm > 0:
            steering_vector = steering_vector / norm
        return steering_vector

    def train(self, examples, **kwargs):
        """Compute gradient steering vector and store in self.ax.proj.weight."""
        self._load_decoder()
        concept = kwargs.get("concept", "")

        # Initialize self.ax if not already done
        if not hasattr(self, 'ax'):
            from .mean import LogisticRegressionModel
            self.ax = LogisticRegressionModel(
                self.model.config.hidden_size, 1)
            self.ax.to(self.device)

        logger.warning(f"Computing gradient steering vector for concept: {concept}")

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        steering_vector = self._compute_steering_vector(examples, concept)

        # Store in self.ax.proj.weight for standard save/load
        self.ax.proj.weight.data = steering_vector.unsqueeze(0).to(self.device)
        self.ax.proj.bias.data = torch.zeros(1).to(self.device)
