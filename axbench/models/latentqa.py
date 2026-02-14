"""
LatentQA integration for AxBench.

Implements two model classes:
- LatentQAReading: Uses LatentQA's reading mode for concept detection (latent inference).
- LatentQASteering: Uses LatentQA's control mode for steering (steering inference).

Requires the LatentQA repo (https://github.com/aypan17/latentqa) to be installed.
Install with: pip install -e /path/to/latentqa
"""
import json
import os
import sys
import re

import numpy as np
import torch
from tqdm.auto import tqdm
from dataclasses import dataclass

from .model import Model, BaseModel

import logging
logging.basicConfig(
    format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN,
)
logger = logging.getLogger(__name__)


# Default questions for reading mode (concept detection).
# These are asked to the LatentQA decoder to probe what's in the activations.
CONCEPT_DETECTION_QUESTION_TEMPLATE = (
    "Is the following concept present in the text? Concept: '{concept}'. "
    "Answer with a rating from 0 to 2, where 0 means the concept is not present at all, "
    "1 means the concept is somewhat present, and 2 means the concept is strongly present. "
    "Provide your rating using this exact format: Rating: [[score]]."
)


def _ensure_latentqa_imported():
    """Ensure the LatentQA library is importable."""
    try:
        from lit.utils.activation_utils import latent_qa
        return True
    except ImportError:
        raise ImportError(
            "LatentQA is not installed. Please install it:\n"
            "  git clone https://github.com/aypan17/latentqa.git\n"
            "  pip install -e latentqa/\n"
            "Or add its path to PYTHONPATH."
        )


def _get_tokenize_fn():
    """Get the tokenize function from LatentQA, handling naming differences."""
    try:
        from lit.utils.dataset_utils import lqa_tokenize
        return lqa_tokenize
    except ImportError:
        from lit.utils.dataset_utils import tokenize
        return tokenize


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

        _ensure_latentqa_imported()
        from lit.utils.infra_utils import get_model as lqa_get_model, get_tokenizer as lqa_get_tokenizer

        logger.warning(f"Loading LatentQA decoder from {self.decoder_model_name} to {self.decoder_device}")

        # Load the decoder (same architecture as target + LoRA adapter)
        lqa_tokenizer = lqa_get_tokenizer(self.target_model_name)
        self.decoder_model = lqa_get_model(
            model_name=self.target_model_name,
            tokenizer=lqa_tokenizer,
            load_peft_checkpoint=self.decoder_model_name,
            device=self.decoder_device,
        )
        self.decoder_model.eval()

        # Set up read/write module hooks
        self.module_read, self.module_write = _get_modules(
            self.model, self.decoder_model,
            min_layer=self.min_layer_to_read,
            max_layer=self.max_layer_to_read,
            layer_to_write=self.layer_to_write,
            num_layers_to_read=self.num_layers_to_read,
        )

    def _get_rating_from_completion(self, completion):
        """Parse a 0-2 rating from the decoder's completion."""
        try:
            if "Rating:" in completion:
                rating_text = completion.split("Rating:")[-1].strip()
                rating_text = rating_text.split('\n')[0].strip()
                rating_text = rating_text.replace('[', '').replace(']', '').strip('"').strip("'").strip("*").strip()
                rating = float(rating_text)
                if 0 <= rating <= 2:
                    return rating
            # Try to find any number 0-2 in the response
            numbers = re.findall(r'\b([012](?:\.\d+)?)\b', completion)
            if numbers:
                return float(numbers[-1])
            logger.warning(f"Cannot find rating in completion: {completion[:200]}")
            return -1
        except (ValueError, IndexError) as e:
            logger.error(f"Error parsing rating: {completion[:200]}. Error: {e}")
            return -1

    @torch.no_grad()
    def predict_latent(self, examples, **kwargs):
        """Use LatentQA reading mode for concept detection.

        For each example, extracts activations from the target model,
        feeds them to the LatentQA decoder with a concept-specific question,
        and parses the response for a relevance rating.
        """
        _ensure_latentqa_imported()
        from lit.utils.activation_utils import latent_qa
        from lit.utils.dataset_utils import BASE_DIALOG, ENCODER_CHAT_TEMPLATES

        tokenize_fn = _get_tokenize_fn()

        self.model.eval()
        self.decoder_model.eval()

        concept = kwargs.get("concept", "")
        batch_size = kwargs.get("batch_size", 4)  # smaller default due to 2 models in memory

        # Build the question for this concept
        question_text = CONCEPT_DETECTION_QUESTION_TEMPLATE.format(concept=concept)

        chat_template = ENCODER_CHAT_TEMPLATES.get(self.tokenizer.name_or_path, None)

        all_max_act = []

        for i in tqdm(range(0, len(examples), batch_size), desc="LatentQA Reading"):
            batch_examples = examples.iloc[i:i + batch_size]

            # For each example in the batch, construct LatentQA inputs
            probe_data = []
            for _, row in batch_examples.iterrows():
                # Use the output text as what we want to read from
                text = row.get("output", row.get("input", ""))

                # Format as dialog for the target model
                read_prompt = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=False,
                    add_generation_prompt=True,
                    chat_template=chat_template,
                )

                # The decoder gets a QA dialog asking about the concept
                dialog = BASE_DIALOG + [
                    {"role": "user", "content": question_text},
                ]

                probe_data.append({
                    "read_prompt": read_prompt,
                    "dialog": dialog,
                })

            # Tokenize for LatentQA
            batch_tokenized = tokenize_fn(
                probe_data,
                self.tokenizer,
                name=self.target_model_name,
                generate=True,
                mask_type=None,
                mask_all_but_last=True,
                modify_chat_template=self.modify_chat_template,
            )

            # Run LatentQA: extract activations from target, decode with decoder
            out = latent_qa(
                batch_tokenized,
                self.model,
                self.decoder_model,
                self.module_read[0],
                self.module_write[0],
                self.tokenizer,
                shift_position_ids=False,
                generate=True,
                max_new_tokens=self.max_new_tokens,
                no_grad=True,
            )

            # Parse completions
            for j in range(len(out)):
                num_tokens = batch_tokenized["tokenized_write"][j].shape[0]
                completion = self.tokenizer.decode(out[j][num_tokens:], skip_special_tokens=True)
                rating = self._get_rating_from_completion(completion)
                all_max_act.append(rating)

            torch.cuda.empty_cache()

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
        _ensure_latentqa_imported()
        from lit.utils.infra_utils import get_model as lqa_get_model, get_tokenizer as lqa_get_tokenizer

        logger.warning(f"Loading LatentQA decoder to {self.decoder_device}")
        lqa_tokenizer = lqa_get_tokenizer(self.target_model_name)
        self.decoder_model = lqa_get_model(
            model_name=self.target_model_name,
            tokenizer=lqa_tokenizer,
            load_peft_checkpoint=self.decoder_model_name,
            device=self.decoder_device,
        )
        self.decoder_model.eval()

    def _generate_qa_pairs(self, concept, num_questions=15):
        """Generate QA pairs for a concept using LatentQA reading mode.

        Creates a prompt embodying the concept, runs the target model on it,
        and uses the decoder to generate descriptions of the activations.
        """
        _ensure_latentqa_imported()
        from lit.utils.activation_utils import latent_qa
        from lit.utils.dataset_utils import BASE_DIALOG, ENCODER_CHAT_TEMPLATES

        tokenize_fn = _get_tokenize_fn()
        chat_template = ENCODER_CHAT_TEMPLATES.get(self.tokenizer.name_or_path, None)

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
            dialog = BASE_DIALOG + [{"role": "user", "content": q[0]}]
            probe_data.append({
                "read_prompt": read_prompt,
                "dialog": dialog,
            })

        batch = tokenize_fn(
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

        out = latent_qa(
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
        from lit.utils.activation_utils import latent_qa
        from lit.utils.dataset_utils import BASE_DIALOG
        from peft import LoraConfig, get_peft_model
        from dataclasses import fields

        tokenize_fn = _get_tokenize_fn()

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

        from lit.utils.dataset_utils import ENCODER_CHAT_TEMPLATES
        chat_template = ENCODER_CHAT_TEMPLATES.get(self.tokenizer.name_or_path, None)

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
                "dialog": BASE_DIALOG + [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ],
            }]

            batch = tokenize_fn(
                formatted_data,
                self.tokenizer,
                name=self.target_model_name,
                generate=False,
                mask_all_but_last=True,
                modify_chat_template=self.modify_chat_template,
            )

            out = latent_qa(
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
