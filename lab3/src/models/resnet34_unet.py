import torch
import torch.nn as nn
from .unet import DoubleConvBlock, EncodingBlock, DecodingBlock


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

        # encoding
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

        # bottleneck
        self.bottleneck = DoubleConvBlock(512, 512)

        # decoding
        self.decode1 = DecodingBlock(256, 512, 256)
        self.decode2 = DecodingBlock(128, 256, 128)
        self.decode3 = DecodingBlock(64, 128, 64)
        self.upsample4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(64, 32, kernel_size=1),
        )
        self.decode4 = DoubleConvBlock(96, 64)

        self.upsample5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(64, 32, kernel_size=1),
        )

        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def _make_residual_block(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualConvBlock(in_channels, out_channels, stride=stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualConvBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        # encoding
        conv1 = self.conv1(input)
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        # bottleneck
        bottleneck = self.bottleneck(conv5)

        # decoding
        decode1 = self.decode1(conv4, bottleneck)
        decode2 = self.decode2(conv3, decode1)
        decode3 = self.decode3(conv2, decode2)
        upsample4 = self.upsample4(decode3)
        decode4 = self.decode4(torch.cat([conv1, upsample4], dim=1))
        upsample5 = self.upsample5(decode4)
        final = self.final(upsample5)
        output = torch.sigmoid(final)

        return output
