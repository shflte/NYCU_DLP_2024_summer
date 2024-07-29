from torch import nn
import torch


class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class EncodingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = DoubleConvBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.double_conv(x)

        return x


class DecodingBlock(nn.Module):
    def __init__(self, skipped_in_channels, upper_in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(mode="bilinear", scale_factor=scale_factor),
            nn.Conv2d(upper_in_channels, out_channels, kernel_size=1),
        )

        self.conv = DoubleConvBlock(skipped_in_channels + upper_in_channels // 2, out_channels)

    def forward(self, skipped_input, upper_input):
        upper_output = self.upsample(upper_input)
        concat = torch.cat([skipped_input, upper_output], 1)
        return self.conv(concat)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # encoding
        self.conv1 = EncodingBlock(3, 64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = EncodingBlock(64, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = EncodingBlock(128, 256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = EncodingBlock(256, 512)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        # bottleneck
        self.bottleneck = EncodingBlock(512, 1024)

        # decoding
        self.decode1 = DecodingBlock(512, 1024, 512)
        self.decode2 = DecodingBlock(256, 512, 256)
        self.decode3 = DecodingBlock(128, 256, 128)
        self.decode4 = DecodingBlock(64, 128, 64)

        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, input):
        # encoding
        conv1 = self.conv1(input)
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # bottleneck
        bottleneck = self.bottleneck(maxpool4)

        # decoding
        decode1 = self.decode1(conv4, bottleneck)
        decode2 = self.decode2(conv3, decode1)
        decode3 = self.decode3(conv2, decode2)
        decode4 = self.decode4(conv1, decode3)

        final = self.final(decode4)
        output = torch.sigmoid(final)

        return output
