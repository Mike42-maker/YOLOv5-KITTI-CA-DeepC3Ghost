# Ultralytics YOLOv5-compatible model definition with Coordinate Attention.
"""Independent CA model entry point; the baseline model files stay untouched."""

import argparse
import contextlib
import math
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.ca import CoordinateAttention
from models.common import (
    C3,
    C3SPP,
    C3TR,
    SPP,
    SPPF,
    Bottleneck,
    BottleneckCSP,
    C3Ghost,
    C3x,
    Classify,
    Concat,
    Contract,
    Conv,
    CrossConv,
    DWConv,
    DWConvTranspose2d,
    Expand,
    Focus,
    GhostBottleneck,
    GhostConv,
    Proto,
)
from models.experimental import MixConv2d
from models.yolo import BaseModel, Detect, Segment
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, make_divisible
from utils.torch_utils import initialize_weights


def parse_model_ca(d, ch):
    """Parse the official YOLOv5 graph plus channel-preserving CA layers."""
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw, act, ch_mul = (
        d["anchors"],
        d["nc"],
        d["depth_multiple"],
        d["width_multiple"],
        d.get("activation"),
        d.get("channel_multiple"),
    )
    if act:
        Conv.default_act = eval(act)
        LOGGER.info(f"activation: {act}")
    ch_mul = ch_mul or 8
    na = len(anchors[0]) // 2 if isinstance(anchors, list) else anchors
    no = na * (nc + 5)

    layers, save, c2 = [], [], ch[-1]
    module_set = {
        Conv,
        GhostConv,
        Bottleneck,
        GhostBottleneck,
        SPP,
        SPPF,
        DWConv,
        MixConv2d,
        Focus,
        CrossConv,
        BottleneckCSP,
        C3,
        C3TR,
        C3SPP,
        C3Ghost,
        nn.ConvTranspose2d,
        DWConvTranspose2d,
        C3x,
    }
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
        m = eval(m) if isinstance(m, str) else m
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a

        n = max(round(n * gd), 1) if n > 1 else n
        if m in module_set:
            c1, c2 = ch[f], args[0]
            if c2 != no:
                c2 = make_divisible(c2 * gw, ch_mul)
            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x}:
                args.insert(2, n)
                n = 1
        elif m is CoordinateAttention:
            c2 = ch[f]
            args = [c2, *args]
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in {Detect, Segment}:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m is Segment:
                args[3] = make_divisible(args[3] * gw, ch_mul)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace("__main__.", "")
        np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type, m_.np = i, f, t, np
        LOGGER.info(f"{i:>3}{f!s:>18}{n:>3}{np:10.0f}  {t:<40}{args!s:<30}")
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


class DetectionModelCA(BaseModel):
    """DetectionModel equivalent that uses only the independent CA parser."""

    def __init__(self, cfg="yolov5s_ca.yaml", ch=3, nc=None, anchors=None):
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            import yaml

            self.yaml_file = Path(cfg).name
            with open(cfg, encoding="ascii", errors="ignore") as f:
                self.yaml = yaml.safe_load(f)

        ch = self.yaml["ch"] = self.yaml.get("ch", ch)
        if nc is not None and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc
        if anchors:
            self.yaml["anchors"] = round(anchors)
        self.model, self.save = parse_model_ca(deepcopy(self.yaml), ch=[ch])
        self.names = [str(i) for i in range(self.yaml["nc"])]
        self.inplace = self.yaml.get("inplace", True)

        m = self.model[-1]
        if isinstance(m, (Detect, Segment)):
            def _forward(x):
                return self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)

            s = 256
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()

        initialize_weights(self)
        self.info()
        LOGGER.info("")

    def _initialize_biases(self, cf=None):
        m = self.model[-1]
        for mi, s in zip(m.m, m.stride):
            b = mi.bias.view(m.na, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5 : 5 + m.nc] += (
                math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())
            )
            mi.bias = nn.Parameter(b.view(-1), requires_grad=True)


ModelCA = DetectionModelCA


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="models/yolov5s_ca.yaml")
    parser.add_argument("--nc", type=int, default=4)
    opt = parser.parse_args()
    model = ModelCA(opt.cfg, nc=opt.nc)
    model(torch.rand(1, 3, 640, 640))
