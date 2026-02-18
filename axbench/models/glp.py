"""
GLP (Generative Latent Prior) wrapper classes for AxBench.

GLP is a diffusion model trained on natural LLM activations that post-processes
steering interventions to snap them back onto the activation manifold. It composes
with any base steering method: after h' = h + α·w, GLP denoises h' via partial
SDEdit (noise at u=0.5, denoise back).

Pre-trained model: generative-latent-prior/glp-llama8b-d6 (layer 15, meta-llama/Llama-3.1-8B).

GLP needs no per-concept training — it's a universal post-processor. These wrapper
classes delegate training to a base method and swap in a GLP-aware intervention at
inference.

To reuse pre-existing base model weights without retraining, symlink:
    GLPDiffMean_weight.pt → DiffMean_weight.pt
    GLPDiffMean_bias.pt → DiffMean_bias.pt
"""

import os
import torch
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

from .model import Model
from .mean import LogisticRegressionModel, DiffMean
from .interventions import GLPAdditionIntervention
from pyvene import IntervenableConfig, IntervenableModel
from ..utils.model_utils import gather_residual_activations, set_decoder_norm_to_unit_norm
from .probe import DataCollator, make_data_module
from torch.utils.data import DataLoader

import logging
logging.basicConfig(
    format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN,
)
logger = logging.getLogger(__name__)

GLP_HF_REPO = "generative-latent-prior/glp-llama8b-d6"


def _load_glp_model(repo_id=GLP_HF_REPO, device="cuda:0"):
    """Load a pre-trained GLP denoiser from HuggingFace."""
    from glp.denoiser import load_glp
    glp_model = load_glp(repo_id, device=device)
    glp_model.eval()
    return glp_model


class GLPDiffMean(Model):
    """DiffMean steering with GLP post-processing.

    Training: delegates to DiffMean (diff-in-means on activations).
    Inference: uses GLPAdditionIntervention which applies GLP denoising
    after the standard activation addition.
    """

    def __str__(self):
        return 'GLPDiffMean'

    def make_model(self, **kwargs):
        mode = kwargs.get("mode", "train")
        if mode == "steering":
            ax = GLPAdditionIntervention(
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

    def make_dataloader(self, examples, **kwargs):
        data_module = make_data_module(self.tokenizer, self.model, examples)
        train_dataloader = DataLoader(
            data_module["train_dataset"], shuffle=True,
            batch_size=self.training_args.batch_size,
            collate_fn=data_module["data_collator"])
        return train_dataloader

    @torch.no_grad()
    def train(self, examples, **kwargs):
        """Diff-in-means training (same as DiffMean)."""
        train_dataloader = self.make_dataloader(examples)
        torch.cuda.empty_cache()
        self.ax.eval()
        self.ax.to(self.device)

        positive_activations = []
        negative_activations = []
        for _ in range(self.training_args.n_epochs):
            for batch in train_dataloader:
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                activations = gather_residual_activations(
                    self.model, self.layer,
                    {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
                ).detach()
                nonbos_mask = inputs["attention_mask"][:, kwargs["prefix_length"]:]
                activations = activations[:, kwargs["prefix_length"]:][nonbos_mask.bool()]
                labels = inputs["labels"].unsqueeze(1).repeat(
                    1, inputs["input_ids"].shape[1] - kwargs["prefix_length"])
                positive_activations.append(activations[labels[nonbos_mask.bool()] == 1])
                negative_activations.append(activations[labels[nonbos_mask.bool()] != 1])

        mean_positive_activation = torch.cat(positive_activations, dim=0).mean(dim=0)
        mean_negative_activation = torch.cat(negative_activations, dim=0).mean(dim=0)
        self.ax.proj.weight.data = mean_positive_activation.unsqueeze(0) - mean_negative_activation.unsqueeze(0)
        set_decoder_norm_to_unit_norm(self.ax)
        logger.warning("Training finished.")

    def _load_glp(self):
        """Load GLP denoiser from HF and attach to the intervention."""
        logger.warning(f"Loading GLP denoiser from {GLP_HF_REPO}")
        glp_model = _load_glp_model(repo_id=GLP_HF_REPO, device=self.device)
        self.ax.set_glp_model(glp_model)
        logger.warning("GLP denoiser attached to intervention.")

    def load(self, dump_dir=None, **kwargs):
        """Load base steering weights, then attach GLP denoiser."""
        super().load(dump_dir, **kwargs)
        if kwargs.get("mode", "steering") == "steering":
            self._load_glp()


class GLPLatentQAGradientSteering(Model):
    """LatentQAGradientSteering with GLP post-processing.

    Training: delegates to LatentQAGradientSteering (gradient-based steering vectors).
    Inference: uses GLPAdditionIntervention which applies GLP denoising
    after the standard activation addition.
    """

    def __init__(self, model, tokenizer, layer, training_args=None, **kwargs):
        super().__init__(model, tokenizer, layer, training_args, **kwargs)
        self.decoder_device = kwargs.get("decoder_device", "cuda:1")
        self.decoder_model_name = kwargs.get(
            "decoder_model_name", "aypan17/latentqa_llama-3-8b-instruct")
        self.target_model_name = kwargs.get(
            "target_model_name", "meta-llama/Llama-3.1-8B-Instruct")
        self.min_layer_to_read = kwargs.get("min_layer_to_read", 15)
        self.max_layer_to_read = kwargs.get("max_layer_to_read", 16)
        self.num_layers_to_read = kwargs.get("num_layers_to_read", 1)
        self.layer_to_write = kwargs.get("layer_to_write", 0)
        self.modify_chat_template = kwargs.get("modify_chat_template", True)
        self.gradient_batch_size = kwargs.get("gradient_batch_size", 4)
        self.decoder_model = None

    def __str__(self):
        return 'GLPLatentQAGradientSteering'

    def make_model(self, **kwargs):
        mode = kwargs.get("mode", "train")
        if mode == "steering":
            ax = GLPAdditionIntervention(
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

    def _load_decoder(self):
        if self.decoder_model is not None:
            return
        from .latentqa import _load_decoder_model
        self.decoder_model = _load_decoder_model(
            self.target_model_name, self.decoder_model_name, self.decoder_device)
        target_vocab_size = self.model.get_input_embeddings().weight.shape[0]
        decoder_vocab_size = self.decoder_model.get_input_embeddings().weight.shape[0]
        if target_vocab_size != decoder_vocab_size:
            self.decoder_model.resize_token_embeddings(target_vocab_size)

    def train(self, examples, **kwargs):
        """Compute gradient steering vector (delegates to LatentQAGradientSteering logic)."""
        from .latentqa import LatentQAGradientSteering

        # Create a temporary LatentQAGradientSteering instance to reuse its training logic
        lqa_gs = LatentQAGradientSteering(
            self.model, self.tokenizer, self.layer, self.training_args,
            device=self.device, decoder_device=self.decoder_device,
            decoder_model_name=self.decoder_model_name,
            target_model_name=self.target_model_name,
            min_layer_to_read=self.min_layer_to_read,
            max_layer_to_read=self.max_layer_to_read,
            num_layers_to_read=self.num_layers_to_read,
            layer_to_write=self.layer_to_write,
            modify_chat_template=self.modify_chat_template,
            gradient_batch_size=self.gradient_batch_size,
            seed=self.seed,
        )

        if not hasattr(self, 'ax'):
            self.ax = LogisticRegressionModel(
                self.model.config.hidden_size, 1)
            self.ax.to(self.device)

        # Share the ax so the gradient vector ends up in our weights
        lqa_gs.ax = self.ax
        lqa_gs.train(examples, **kwargs)

    def save(self, dump_dir, **kwargs):
        """Save steering vector (same format as LatentQAGradientSteering)."""
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

    def _load_glp(self):
        logger.warning(f"Loading GLP denoiser from {GLP_HF_REPO}")
        glp_model = _load_glp_model(repo_id=GLP_HF_REPO, device=self.device)
        self.ax.set_glp_model(glp_model)
        logger.warning("GLP denoiser attached to intervention.")

    def load(self, dump_dir=None, **kwargs):
        """Load base steering weights, then attach GLP denoiser."""
        super().load(dump_dir, **kwargs)
        if kwargs.get("mode", "steering") == "steering":
            self._load_glp()
