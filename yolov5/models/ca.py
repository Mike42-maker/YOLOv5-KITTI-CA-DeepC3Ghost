# Ultralytics YOLOv5-compatible Coordinate Attention module.
"""Coordinate Attention based on Hou et al., CVPR 2021.

Reference paper: https://arxiv.org/abs/2103.02907
Reference implementation: https://github.com/Andrew-Qibin/CoordAttention
"""

import torch
from torch import nn


class HSigmoid(nn.Module):
    """Hard sigmoid used by the original Coordinate Attention design."""

    def forward(self, x):
        return nn.functional.relu6(x + 3.0, inplace=True) / 6.0


class HSwish(nn.Module):
    """Hard swish used by the original Coordinate Attention design."""

    def forward(self, x):
        return x * (nn.functional.relu6(x + 3.0, inplace=True) / 6.0)


class CoordinateAttention(nn.Module):
    """Encode height- and width-aware attention without changing tensor shape."""

    def __init__(self, channels, reduction=32):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        mip = max(8, channels // reduction)
        self.channels = channels
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = HSwish()
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = HSigmoid()

    def forward(self, x):
        identity = x
        n, _, h, w = x.shape
        # Explicit reductions are equivalent to the coordinate pools and have
        # a deterministic CUDA backward implementation.
        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.sigmoid(self.conv_h(x_h))
        a_w = self.sigmoid(self.conv_w(x_w))
        return identity * a_h * a_w
