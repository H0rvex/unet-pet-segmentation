import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1    = nn.BatchNorm2d(out_channels)
        self.layer2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2    = nn.BatchNorm2d(out_channels)
        self.pool   = nn.MaxPool2d(2)

    def forward(self, x):
        x = torch.relu(self.bn1(self.layer1(x)))
        x = torch.relu(self.bn2(self.layer2(x)))
        pooled = self.pool(x)
        return x, pooled  # x: skip connection, pooled: next encoder level


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.layer1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.bn1    = nn.BatchNorm2d(out_channels)
        self.layer2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2    = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)
        x = torch.relu(self.bn1(self.layer1(x)))
        x = torch.relu(self.bn2(self.layer2(x)))
        return x


class UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.enc1  = EncoderBlock(3, 64)
        self.enc2  = EncoderBlock(64, 128)
        self.enc3  = EncoderBlock(128, 256)
        self.conv1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(512)
        self.dec1  = DecoderBlock(512, 256)
        self.dec2  = DecoderBlock(256, 128)
        self.dec3  = DecoderBlock(128, 64)
        self.conv3 = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # encoding
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        # bottleneck
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        # decoding
        x = self.dec1(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec3(x, skip1)
        # final layer
        x = self.conv3(x)
        return x
