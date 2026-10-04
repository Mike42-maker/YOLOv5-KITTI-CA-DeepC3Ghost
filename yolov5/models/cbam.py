# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convolutional Block Attention Module (CBAM) for YOLOv5."""

import torch
from torch import nn


class ChannelAttention(nn.Module):
    """CBAM channel attention using average- and max-pooled descriptors."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.mlp(torch.mean(x, dim=(2, 3), keepdim=True))
        max_pool = self.mlp(torch.amax(x, dim=(2, 3), keepdim=True))
        return self.sigmoid(avg + max_pool)


class SpatialAttention(nn.Module):
    """CBAM spatial attention from channel-wise average and maximum maps."""

    def __init__(self, kernel_size=7):
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError(f"kernel_size must be 3 or 7, got {kernel_size}")
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_pool = torch.amax(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, max_pool], dim=1)))


class CBAM(nn.Module):
    """Sequential channel and spatial attention that preserves tensor shape."""

    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        return x * self.spatial_attention(x)
