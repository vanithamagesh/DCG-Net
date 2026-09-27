"""DCG-Net: deep context-gated encoder-decoder network for retinal vessel segmentation.

Components
  * five-level residual encoder (32, 64, 128, 256, 256 channels, 1/16 resolution at the bottleneck)
  * multi-dilation context (MDC) module at the bottleneck (dilations 1, 2, 4, 8)
  * spatially gated skip connections, gate computed from [upsampled decoder feature, encoder feature]
  * residual decoder with deep supervision heads at 1/2, 1/4 and 1/8 resolution (training only)
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def bn(c):
    return nn.BatchNorm2d(c)


class ResBlock(nn.Module):
    """Two 3x3 convolutions with batch normalisation and a (projected) identity shortcut."""

    def __init__(self, ci, co, d=1):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(ci, co, 3, padding=d, dilation=d, bias=False), bn(co), nn.ReLU(inplace=True),
                                  nn.Conv2d(co, co, 3, padding=1, bias=False), bn(co))
        self.short = nn.Sequential(nn.Conv2d(ci, co, 1, bias=False), bn(co)) if ci != co else nn.Identity()

    def forward(self, x):
        return F.relu(self.body(x) + self.short(x))


class Context(nn.Module):
    """Multi-dilation context (MDC) module: parallel dilated branches, 1x1 projection, residual."""

    def __init__(self, c):
        super().__init__()
        self.b = nn.ModuleList([nn.Sequential(nn.Conv2d(c, c // 4, 3, padding=d, dilation=d, bias=False), bn(c // 4), nn.ReLU(inplace=True))
                                for d in (1, 2, 4, 8)])
        self.proj = nn.Sequential(nn.Conv2d(c, c, 1, bias=False), bn(c))

    def forward(self, x):
        return F.relu(x + self.proj(torch.cat([b(x) for b in self.b], 1)))


class SkipGate(nn.Module):
    """Spatial gate alpha = sigmoid(W_b * ReLU(W_a * [D_up, S])); returns alpha * S."""

    def __init__(self, cd, cs):
        super().__init__()
        self.g = nn.Sequential(nn.Conv2d(cd + cs, cs // 2, 1), nn.ReLU(inplace=True), nn.Conv2d(cs // 2, 1, 1), nn.Sigmoid())

    def forward(self, d, s):
        return s * self.g(torch.cat([d, s], 1))


class DCGNet(nn.Module):
    """Deep context-gated network (5.90 M parameters at width 32)."""

    def __init__(self, width=32, in_channels=5):
        super().__init__()
        c = [width, 2 * width, 4 * width, 8 * width, 8 * width]
        self.enc = nn.ModuleList([ResBlock(in_channels, c[0])] + [ResBlock(c[i - 1], c[i]) for i in range(1, 5)])
        self.ctx = Context(c[4])
        self.gates = nn.ModuleList([SkipGate(c[i + 1], c[i]) for i in range(4)])
        self.dec = nn.ModuleList([ResBlock(c[i + 1] + c[i], c[i]) for i in range(4)])
        self.head = nn.Conv2d(c[0], 1, 1)
        self.aux = nn.ModuleList([nn.Conv2d(c[i], 1, 1) for i in (1, 2, 3)])

    def forward(self, x):
        H, W = x.shape[-2:]; ph, pw = (-H) % 16, (-W) % 16
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        s = [self.enc[0](x)]
        for i in range(1, 5):
            s.append(self.enc[i](F.max_pool2d(s[-1], 2)))
        d = self.ctx(s[4]); outs = {}
        for i in (3, 2, 1, 0):
            d = F.interpolate(d, size=s[i].shape[-2:], mode="bilinear", align_corners=False)
            d = self.dec[i](torch.cat([d, self.gates[i](d, s[i])], 1))
            if i in (1, 2, 3) and self.training:
                outs[f"aux{i}"] = F.interpolate(self.aux[i - 1](d), size=x.shape[-2:], mode="bilinear", align_corners=False)
        outs["logits"] = self.head(d)
        return {k: v[..., :H, :W] for k, v in outs.items()}


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def load_checkpoint(path, device="cpu"):
    """Load a DCG-Net checkpoint; returns (model in eval mode, validation-selected threshold)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = DCGNet(ck.get("cfg", {}).get("width", 32)); m.load_state_dict(ck["model"])
    return m.to(device).eval(), float(ck.get("threshold", 0.5))
