"""Training, validation, test-time augmentation and pooled FOV metrics."""
from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import Patches
from .losses import dcgnet_loss
from .model import DCGNet, count_parameters

DEFAULT_CFG = dict(width=32, patch=384, batch=4, steps=6000, lr=1e-3, val_every=1000, workers=2, seed=1)


# ---------------------------------------------------------------- inference
@torch.inference_mode()
def predict(model, case, device, tta=True):
    """Vessel probability map; four flip views averaged when tta=True."""
    x = torch.from_numpy(case.x.transpose(2, 0, 1)).float()[None].to(device)
    outs = []
    for dims in ([(), (-1,), (-2,), (-2, -1)] if tta else [()]):
        xv = torch.flip(x, dims) if dims else x
        with torch.autocast("cuda", dtype=torch.float16, enabled=str(device).startswith("cuda")):
            p = torch.sigmoid(model(xv)["logits"].float())
        outs.append(torch.flip(p, dims) if dims else p)
    return torch.stack(outs).mean(0)[0, 0].cpu().numpy()


def predict_ensemble(models, case, device):
    return np.mean([predict(m, case, device) for m in models], 0)


# ---------------------------------------------------------------- metrics (pooled over FOV pixels)
def counts(p, c, t):
    v = c.fov > 0; y = c.target > 0; b = (p >= t) & v
    return {"tp": int((b & y).sum()), "fp": int((b & ~y & v).sum()), "fn": int((~b & y & v).sum()), "tn": int((~b & ~y & v).sum()),
            "thin_tp": int((b & (c.thin > 0)).sum()), "thin_n": int((c.thin > 0).sum()),
            "center_tp": int((b & (c.center > 0)).sum()), "center_n": int((c.center > 0).sum()),
            "thick_tp": int((b & (c.thick > 0)).sum()), "thick_n": int((c.thick > 0).sum())}


def summarize(cnts, probs=None, targets=None):
    s = {k: sum(c[k] for c in cnts) for k in cnts[0]}; tp, fp, fn, tn = s["tp"], s["fp"], s["fn"], s["tn"]; e = 1e-12
    out = dict(s, sensitivity=tp / (tp + fn + e), specificity=tn / (tn + fp + e), accuracy=(tp + tn) / (tp + tn + fp + fn + e),
               precision=tp / (tp + fp + e), f1=2 * tp / (2 * tp + fp + fn + e), iou=tp / (tp + fp + fn + e),
               thin_recall=s["thin_tp"] / max(s["thin_n"], 1), thin_centerline_recall=s["center_tp"] / max(s["center_n"], 1),
               thick_recall=s["thick_tp"] / max(s["thick_n"], 1))
    if probs is not None:
        from sklearn.metrics import average_precision_score, roc_auc_score
        p = np.concatenate(probs); y = np.concatenate(targets).astype(bool)
        out["auc"] = float(roc_auc_score(y, p)); out["pr_auc"] = float(average_precision_score(y, p))
    return out


def evaluate(probs, cases, thresholds):
    """probs: id -> probability map; thresholds: float or id -> float. Returns (pooled summary, per-image rows)."""
    thr = (lambda i: thresholds[i]) if isinstance(thresholds, dict) else (lambda i: thresholds)
    cn, ps, ys, rows = [], [], [], []
    for c in cases:
        p = probs[c.id].astype(np.float32); k = counts(p, c, thr(c.id)); v = c.fov > 0
        cn.append(k); ps.append(p[v]); ys.append(c.target[v]); rows.append({"case": c.id, **summarize([k], [p[v]], [c.target[v]])})
    return summarize(cn, ps, ys), rows


def best_threshold(probs, cases, lo=0.30, hi=0.70):
    """Pooled validation F1 over a threshold grid (step 0.02); returns (f1, threshold)."""
    p = np.concatenate([probs[c.id][c.fov > 0] for c in cases]); y = np.concatenate([c.target[c.fov > 0] for c in cases]) > 0
    return max((float(2 * ((p >= t) & y).sum() / ((p >= t).sum() + y.sum() + 1e-12)), round(float(t), 2)) for t in np.arange(lo, hi + 1e-3, 0.02))


# ---------------------------------------------------------------- training
def train(cfg, train_cases, val_cases, out_dir, device, log=print):
    """Train one DCG-Net; the EMA weights with the best validation F1 are saved with their threshold."""
    cfg = dict(DEFAULT_CFG, **cfg); out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"]); random.seed(cfg["seed"])
    model = DCGNet(cfg["width"]).to(device); ema = copy.deepcopy(model).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=cfg["steps"], pct_start=0.05, div_factor=10, final_div_factor=50)
    cuda = str(device).startswith("cuda")
    dl = DataLoader(Patches(train_cases, cfg["steps"] * cfg["batch"], cfg["patch"], cfg["seed"]), batch_size=cfg["batch"],
                    num_workers=cfg["workers"], pin_memory=cuda, persistent_workers=cfg["workers"] > 0, drop_last=True)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    hist = {"loss": [], "parameters": count_parameters(model), "config": cfg}; best = {"f1": -1.0}; bad = 0; t0 = time.time()
    for step, (x, y, v, thin, thick, sk) in enumerate(dl, 1):
        x, y, v, sk = [a.to(device, non_blocking=True) for a in (x, y, v, sk)]
        model.train(); opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
            out = model(x)
        out = {k: o.float() for k, o in out.items()}          # all loss terms in fp32
        loss = dcgnet_loss(out, y, v, sk, step / cfg["steps"])
        if not torch.isfinite(loss):
            bad += 1
            if bad > 20: raise RuntimeError(f"non-finite loss for 20 consecutive steps (step {step})")
            continue
        bad = 0
        scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        scaler.step(opt); scaler.update(); sch.step()
        with torch.no_grad():                                   # EMA of weights and BN statistics
            dec = min(0.999, (1 + step) / (10 + step))
            for e, p in zip(ema.parameters(), model.parameters()): e.lerp_(p, 1 - dec)
            for e, p in zip(ema.buffers(), model.buffers()):
                if e.dtype.is_floating_point: e.lerp_(p, 1 - dec)
                else: e.copy_(p)
        hist["loss"].append(float(loss.detach()))
        if step == 1 or step % 500 == 0:
            log(json.dumps({"step": step, "of": cfg["steps"], "loss": round(float(np.mean(hist["loss"][-500:])), 4), "minutes": round((time.time() - t0) / 60, 1)}))
        if step % cfg["val_every"] == 0 or step == cfg["steps"]:
            probs = {c.id: predict(ema, c, device, tta=False) for c in val_cases}
            f1, t = best_threshold(probs, val_cases); log(f"validation step {step}: F1 {100 * f1:.2f} at threshold {t:.2f}")
            hist.setdefault("validation", []).append({"step": step, "f1": f1, "threshold": t})
            if f1 > best["f1"]: best = {"f1": f1, "threshold": t, "step": step, "state": copy.deepcopy(ema.state_dict())}
    ema.load_state_dict(best.pop("state")); hist["selected"] = best
    log(f"selected step {best['step']} (validation F1 {100 * best['f1']:.2f}, threshold {best['threshold']:.2f})")
    torch.save({"model": ema.state_dict(), "cfg": cfg, "threshold": best["threshold"]}, out_dir / "model.pt")
    (out_dir / "history.json").write_text(json.dumps(hist))
    return ema, best["threshold"]
