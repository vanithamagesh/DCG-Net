"""Segment vessels in your own fundus images with released DCG-Net weights.

    python scripts/predict.py --weights weights/dcgnet_drive_seed1.pt weights/dcgnet_drive_seed2.pt weights/dcgnet_drive_seed3.pt \
                              --images my_images/ --out predictions/

Writes <name>_prob.png (probability x 255) and <name>_mask.png (binary vessel map) for each image.
With several weight files the probability maps are averaged and the mean of their thresholds is used,
unless --threshold is given. Use --fov-mask to supply masks (same file name, in another folder);
otherwise the FOV is estimated from the image.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dcgnet as D  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, nargs="+", required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("predictions"))
    ap.add_argument("--fov-mask", type=Path, default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    loaded = [D.load_checkpoint(w, a.device) for w in a.weights]; models = [m for m, _ in loaded]
    thr = a.threshold if a.threshold is not None else float(np.mean([t for _, t in loaded]))
    files = sorted(p for p in a.images.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".ppm", ".bmp", ".gif"))
    for p in files:
        im = np.asarray(Image.open(p).convert("RGB"))
        if a.fov_mask:
            fov = np.asarray(Image.open(next(a.fov_mask.glob(p.stem + ".*"))).convert("L")) > 0
        else:
            fov = D.fov_from_image(im)
        c = D.Case(p.stem, im, np.zeros(im.shape[:2], bool), fov)
        prob = D.predict_ensemble(models, c, a.device) * fov
        Image.fromarray((prob * 255).astype(np.uint8)).save(a.out / f"{p.stem}_prob.png")
        Image.fromarray(((prob >= thr) * 255).astype(np.uint8)).save(a.out / f"{p.stem}_mask.png")
        print(f"{p.name}: done (threshold {thr:.2f})")


if __name__ == "__main__":
    main()
