import torch
import torch.nn as nn
import numpy as np
import yaml
import math
from .VQGAN import VQGAN
from utils import mask_image
from .Transformer import BidirectionalTransformer


class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs["VQ_Configs"])

        self.num_image_tokens = configs["num_image_tokens"]
        self.mask_token_id = configs["num_codebook_vectors"]
        self.choice_temperature = configs["choice_temperature"]
        self.gamma = self.gamma_func(configs["gamma_type"])
        self.transformer = BidirectionalTransformer(configs["Transformer_param"])

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs["VQ_config_path"], "r"))
        model = VQGAN(cfg["model_param"])
        model.load_state_dict(
            torch.load(configs["VQ_CKPT_path"], weights_only=True), strict=True
        )
        model = model.eval()
        return model

    @torch.no_grad()
    def encode_to_z(self, x):
        codebook_mapping, codebook_indices, _ = self.vqgan.encode(x)
        batch_size = codebook_mapping.shape[0]
        codebook_indices = codebook_indices.view(batch_size, -1)
        return codebook_mapping, codebook_indices

    def gamma_func(self, mode="cosine"):
        def linear_func(ratio):
            return 1.0 - ratio

        def cosine_func(ratio):
            return 0.5 * (1.0 + math.cos(math.pi * ratio))

        def square_func(ratio):
            return 1.0 - (ratio**2)

        if mode == "linear":
            return linear_func
        elif mode == "cosine":
            return cosine_func
        elif mode == "square":
            return square_func
        else:
            raise NotImplementedError(f"Unknown gamma mode: {mode}")

    def forward(self, x):
        _, z_indices = self.encode_to_z(x)

        mask_ratio = np.random.uniform(0, 1)
        mask_rate = self.gamma(mask_ratio)
        masked_z_indices = mask_image(z_indices, self.mask_token_id, mask_rate)

        logits = self.transformer(masked_z_indices)

        return logits, z_indices


__MODEL_TYPE__ = {"MaskGit": MaskGit}
