"""Datasets, input representation and patch sampling for DCG-Net."""
from __future__ import annotations

import gzip
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from torch.utils.data import Dataset

# images of each official training set held out for checkpoint and threshold selection
VAL_IDS = {"DRIVE": {"22", "27", "34", "39"}, "CHASE": {"09L", "09R", "10L", "10R"}}


def features(image):
    """Five input channels: RGB, CLAHE green, gamma-corrected CLAHE grey, each scaled to [-1, 1]."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    green = clahe.apply(image[..., 1])
    grey = clahe.apply(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)).astype(np.float32) / 255.0
    grey = np.power(grey, 1.0 / 1.2)
    x = np.concatenate([image.astype(np.float32) / 255.0, green[..., None].astype(np.float32) / 255.0, grey[..., None]], 2)
    return x * 2.0 - 1.0


def fov_from_image(image):
    """FOV mask for datasets without one (CHASE_DB1, STARE): red channel > 15, 5x5 opening,
    hole filling, largest component, two 3x3 erosions."""
    m = image[..., 0] > 15
    m = ndi.binary_fill_holes(ndi.binary_opening(m, np.ones((5, 5))))
    lab, n = ndi.label(m)
    if n > 1:
        m = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    return ndi.binary_erosion(m, np.ones((3, 3)), iterations=2)


class Case:
    """One image with target, FOV, and the thin/thick/centreline masks used for sampling and evaluation.
    A vessel pixel is thin when the radius at its nearest centreline point is at most 2 pixels."""

    def __init__(self, cid, image, target, fov):
        self.id, self.image = cid, image
        self.target, self.fov = target.astype(np.float32), fov.astype(np.float32)
        self.x = features(image)
        y = self.target > 0; v = self.fov > 0
        sk = skeletonize(y & v)
        radius = ndi.distance_transform_edt(y)
        _, idx = ndi.distance_transform_edt(~sk, return_indices=True)
        local = radius[tuple(idx)]
        self.thin = (y & v & (local <= 2.0)).astype(np.float32)
        self.thick = (y & v & (local > 2.0)).astype(np.float32)
        self.center = (sk & (radius <= 2.0) & v).astype(np.float32)
        self.skeleton = sk.astype(np.float32)


def load_dataset(root, name, split):
    """DRIVE / CHASE in the folder layout of the SGL repository (HR/, VE/, mask/); split = 'train' or 'test'."""
    root = Path(root); cases = []
    if name == "DRIVE":
        folder = root / ("DRIVE" if split == "train" else "DRIVET")
        for p in sorted((folder / "HR").glob("*.tif")):
            cid = p.name.split("_")[0]
            im = np.asarray(Image.open(p).convert("RGB"))
            y = np.asarray(Image.open(folder / "VE" / f"{cid}_manual1.gif").convert("L")) > 0
            v = np.asarray(Image.open(folder / "mask" / f"{cid}_mask.gif").convert("L")) > 0
            cases.append(Case(cid, im, y, v))
        assert len(cases) == 20, (name, split, len(cases))
    elif name == "CHASE":
        folder = root / ("CHASE" if split == "train" else "CHASET")
        for p in sorted((folder / "HR").glob("*.jpg")):
            cid = p.stem.replace("Image_", "")
            im = np.asarray(Image.open(p).convert("RGB"))
            y = np.asarray(Image.open(folder / "VE" / f"{p.stem}_1stHO.png").convert("L")) > 0
            cases.append(Case(cid, im, y, fov_from_image(im)))
        assert len(cases) == (20 if split == "train" else 8), (name, split, len(cases))
    else:
        raise ValueError(name)
    return cases


def load_stare(root):
    """All 20 STARE images with the first observer's (A. Hoover) labels; .gz files are unpacked in place."""
    root = Path(root)
    for g in root.rglob("*.gz"):
        out = g.with_suffix("")
        if not out.exists():
            with gzip.open(g, "rb") as src, open(out, "wb") as dst: shutil.copyfileobj(src, dst)
    cases = []
    for p in sorted(q for q in root.rglob("im*.ppm") if ".ah." not in q.name and ".vk." not in q.name):
        cid = p.name.split(".")[0].replace("im", "")
        im = np.asarray(Image.open(p).convert("RGB"))
        y = np.asarray(Image.open(next(root.rglob(f"im{cid}.ah.ppm"))).convert("L")) > 127
        cases.append(Case(cid, im, y, fov_from_image(im)))
    assert len(cases) == 20, f"expected 20 STARE images, found {len(cases)}"
    return cases


class Patches(Dataset):
    """Random square patches. Centre drawn from thin-vessel pixels (p = 0.35), any vessel pixel (0.30)
    or any FOV pixel (0.35). Label-preserving augmentation: 90-degree rotations, flips, intensity jitter."""

    def __init__(self, cases, n, patch, seed):
        self.cases, self.n, self.patch, self.seed = cases, n, patch, seed
        self.coords = [[np.argwhere(c.thin > 0), np.argwhere(c.target > 0), np.argwhere(c.fov > 0)] for c in cases]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = random.Random(self.seed * 1_000_003 + i)
        k = rng.randrange(len(self.cases)); c = self.cases[k]; co = self.coords[k]
        u = rng.random(); pts = co[0] if u < 0.35 else co[1] if u < 0.65 else co[2]
        cy, cx = pts[rng.randrange(len(pts))]
        H, W = c.target.shape; P = self.patch
        top = min(max(int(cy) - rng.randrange(P), 0), H - P); left = min(max(int(cx) - rng.randrange(P), 0), W - P)
        arrs = [a[top:top + P, left:left + P] for a in (c.x, c.target, c.fov, c.thin, c.thick, c.skeleton)]
        r = rng.randrange(4); arrs = [np.rot90(a, r) for a in arrs]
        if rng.random() < 0.5: arrs = [a[:, ::-1] for a in arrs]
        if rng.random() < 0.5: arrs = [a[::-1] for a in arrs]
        arrs[0] = np.clip(arrs[0] * rng.uniform(0.85, 1.15) + rng.uniform(-0.08, 0.08), -1, 1)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float()
        return (t(arrs[0].transpose(2, 0, 1)),) + tuple(t(a)[None] for a in arrs[1:])
