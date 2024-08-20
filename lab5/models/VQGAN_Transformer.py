import torch
import torch.nn as nn
import numpy as np
import yaml
import math
from .VQGAN import VQGAN
from utils import mask_latent
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

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path, weights_only=True))

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
        masked_z_indices = mask_latent(z_indices, self.mask_token_id, mask_rate)

        logits = self.transformer(masked_z_indices)

        return logits, z_indices

    @torch.no_grad()
    def inpainting(self, image, mask_b, ratio, mask_num):
        _, z_indices = self.encode_to_z(image)
        logits = self.transformer(z_indices)
        z_indices_predict_prob = torch.nn.functional.softmax(logits, dim=-1)

        # Add temperature annealing gumbel noise to the confidence
        g = -torch.log(-torch.log(torch.rand_like(z_indices_predict_prob)))
        temperature = self.choice_temperature * (1 - ratio)
        confidence = z_indices_predict_prob + temperature * g

        # Predict the tokens and replace the masked tokens
        z_indices_predict = torch.argmax(confidence, dim=-1)
        z_indices_predict[~mask_b] = z_indices[~mask_b]

        # If the prediction is special token, replace it with 0 and set confidence to -inf
        z_indices_predict[z_indices_predict == self.mask_token_id] = 0

        # Get next mask
        confidence, _ = torch.max(confidence, dim=-1)
        # Replace the mask token with -inf if the prediction is mask token
        confidence[z_indices_predict == self.mask_token_id] = -float("inf")
        confidence[~mask_b] = -float("inf")
        tokens_to_unmask = mask_b.sum() - math.floor(self.gamma(ratio) * mask_num)
        _, indices = torch.topk(confidence, tokens_to_unmask, dim=-1)
        if indices.nelement() > 0:
            mask_b[0, indices[0]] = False

        return z_indices_predict, mask_b


__MODEL_TYPE__ = {"MaskGit": MaskGit}
