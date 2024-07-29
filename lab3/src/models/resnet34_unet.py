import torch
import torch.nn as nn
from unet import DoubleConvBlock, DecodingBlock


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1):
        super().__init__()
        self.double_conv = DoubleConvBlock(in_channels, out_channels, stride=stride, padding=padding)
        self.skip_connect = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.double_conv(x) + self.skip_connect(x)


class ResNet34_UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = self._make_residual_block(64, 64, 3)
        self.conv3 = self._make_residual_block(64, 128, 4, stride=2)
        self.conv4 = self._make_residual_block(128, 256, 6, stride=2)
        self.conv5 = self._make_residual_block(256, 512, 3, stride=2)

    def _make_residual_block(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualConvBlock(in_channels, out_channels, stride=stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualConvBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x
