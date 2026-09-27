"""Training objective of DCG-Net.

L = L_reg(P, Y) + lambda_cl(t) * L_clDice + sum_k lambda_k * L_reg(P_k, Y)
  L_reg   = BCE + 0.5 * soft Dice, both restricted to FOV pixels
  L_clDice= 1 - 2 Tprec Tsens / (Tprec + Tsens)  (soft skeleton, 6 iterations)
  lambda_cl(t) = 0.2 * min(1, t / 0.2T);  lambda_1..3 = 0.3, 0.2, 0.1 (auxiliary heads at 1/2, 1/4, 1/8)
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

AUX_WEIGHTS = ((1, 0.3), (2, 0.2), (3, 0.1))


def masked_mean(x, m):
    return (x * m).sum() / m.sum().clamp_min(1.0)


def region_loss(logits, y, m):
    bce = masked_mean(F.binary_cross_entropy_with_logits(logits, y, reduction="none"), m)
    p = torch.sigmoid(logits); d = (1, 2, 3)
    dice = 1 - ((2 * (p * y * m).sum(d) + 1) / ((p * m).sum(d) + (y * m).sum(d) + 1)).mean()
    return bce + 0.5 * dice


def soft_erode(x):
    return torch.minimum(-F.max_pool2d(-x, (3, 1), 1, (1, 0)), -F.max_pool2d(-x, (1, 3), 1, (0, 1)))


def soft_skeleton(x, iterations=6):
    x1 = F.max_pool2d(soft_erode(x), 3, 1, 1); sk = F.relu(x - x1)
    for _ in range(iterations):
        x = soft_erode(x); x1 = F.max_pool2d(soft_erode(x), 3, 1, 1); delta = F.relu(x - x1)
        sk = sk + F.relu(delta - sk * delta)
    return sk


def cldice_loss(p, y, y_skeleton, m):
    ps = soft_skeleton(p) * m; d = (1, 2, 3)
    tprec = ((ps * y).sum(d) + 1) / (ps.sum(d) + 1)
    tsens = ((y_skeleton * m * p).sum(d) + 1) / ((y_skeleton * m).sum(d) + 1)
    return 1 - (2 * tprec * tsens / (tprec + tsens)).mean()


def dcgnet_loss(out, y, fov, y_skeleton, progress, cldice_weight=0.2, ramp=0.2):
    """progress = t / T in [0, 1]."""
    loss = region_loss(out["logits"], y, fov)
    loss = loss + cldice_weight * min(1.0, progress / ramp) * cldice_loss(torch.sigmoid(out["logits"]), y, y_skeleton, fov)
    for i, w in AUX_WEIGHTS:
        if f"aux{i}" in out:
            loss = loss + w * region_loss(out[f"aux{i}"], y, fov)
    return loss
