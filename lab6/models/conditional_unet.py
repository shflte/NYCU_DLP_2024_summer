import torch
import torch.nn as nn
from diffusers import UNet2DModel


class ClassConditionedUnet(nn.Module):
    def __init__(self, num_classes=24):
        super().__init__()

        self.model = UNet2DModel(
            sample_size=64,
            in_channels=3 + num_classes,
            out_channels=3,
            time_embedding_type="positional",
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

    def forward(self, x, t, class_labels):
        bs, ch, w, h = x.shape

        class_cond = class_labels.view(bs, class_labels.shape[1], 1, 1).expand(
            bs, class_labels.shape[1], w, h
        )

        net_input = torch.cat((x, class_cond), 1)

        return self.model(net_input, t).sample
