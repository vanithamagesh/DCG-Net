"""DCG-Net on STARE with four-fold cross-validation (GPU).

Protocol (fixed before any test image is seen):
  * the 20 STARE images are sorted by name and split into four folds of five images;
  * for each fold, the other 15 images are used for training, of which 3 are held out for
    checkpoint and threshold selection (validation); the 5 fold images are the test images;
  * three models (seeds 1-3) are trained per fold with the same settings as for DRIVE and CHASE_DB1,
    their four-view predictions are averaged, and the ensemble threshold is selected on the fold's
    validation images;
  * metrics are pooled over the FOV pixels of all 20 test predictions (each image tested exactly once).
Ground truth: first observer (A. Hoover). FOV masks are generated from the image.
Finished models are skipped on restart.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dcgnet as D  # noqa: E402

CFG = dict(steps=6000)
SEEDS = (1, 2, 3)
VAL_POS = (1, 7, 13)  # positions within the 15 training images of a fold that are held out for validation


def logger(path):
    def log(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True)
        with open(path, "a") as h: h.write(line + "\n")
    return log


def write_rows(path, rows):
    with open(path, "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def best_threshold(probs, cases, lo=0.30, hi=0.90):
    p = np.concatenate([probs[c.id][c.fov > 0] for c in cases]); y = np.concatenate([c.target[c.fov > 0] for c in cases]) > 0
    return max((float(2 * ((p >= t) & y).sum() / ((p >= t).sum() + y.sum())), round(float(t), 2)) for t in np.arange(lo, hi + 1e-3, 0.02))


def pooled(cases, probs, thr):
    """Pool counts and probabilities over all test images; thr maps image id -> threshold of its fold."""
    cn, ps, ys, rows = [], [], [], []
    for c in cases:
        p = probs[c.id].astype(np.float32); k = D.engine.counts(p, c, thr[c.id]); v = c.fov > 0
        cn.append(k); ps.append(p[v]); ys.append(c.target[v]); rows.append({"case": c.id, **D.summarize([k], [p[v]], [c.target[v]])})
    return D.summarize(cn, ps, ys), rows


def fmt(s):
    return (f"F1 {100 * s['f1']:.2f} Se {100 * s['sensitivity']:.2f} Sp {100 * s['specificity']:.2f} Acc {100 * s['accuracy']:.2f} "
            f"AUC {100 * s['auc']:.2f} thin Se {100 * s['thin_recall']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stare", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True); log = logger(a.out / "run_log_stare.txt")
    if a.device == "cpu" and not a.smoke:
        raise SystemExit("No GPU. Colab: Runtime > Change runtime type > T4 GPU, then Run all.")
    if a.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True; log(f"GPU: {torch.cuda.get_device_name(0)}")
    t0 = time.time(); cases = D.load_stare(a.stare); ids = [c.id for c in cases]; byid = {c.id: c for c in cases}
    log("STARE images: " + " ".join(ids))
    folds = [ids[5 * k:5 * k + 5] for k in range(4)]
    ens_probs, ens_thr, seed_probs, seed_thr, fold_info = {}, {}, {s: {} for s in SEEDS}, {s: {} for s in SEEDS}, []
    for k, test_ids in enumerate(folds):
        train_ids = [i for i in ids if i not in test_ids]; val_ids = [train_ids[j] for j in VAL_POS]
        tr = [byid[i] for i in train_ids if i not in val_ids]; va = [byid[i] for i in val_ids]; te = [byid[i] for i in test_ids]
        log(f"=== fold {k + 1}/4: test {test_ids}, validation {val_ids}, training on {len(tr)} images")
        models = []
        for s in SEEDS:
            rd = a.out / "STARE" / f"fold{k + 1}" / f"ctx_s{s}"; ck = rd / "model.pt"
            if ck.exists():
                log(f"skip fold {k + 1} seed {s} (finished)")
            else:
                cfg = dict(CFG, seed=s, workers=a.workers)
                if a.smoke: cfg.update(steps=4, batch=2, val_every=2, patch=128)
                log(f"--- fold {k + 1} seed {s}: training {cfg['steps']} steps")
                m, _ = D.train(cfg, tr, va, rd, a.device, log); del m
                if a.device.startswith("cuda"): torch.cuda.empty_cache()
            m, t = D.load_checkpoint(ck, a.device); m.threshold = t; models.append(m)
        with torch.inference_mode():
            per = {s: {cs.id: D.predict(m, cs, a.device) for cs in va + te} for s, m in zip(SEEDS, models)}
        for s, m in zip(SEEDS, models):
            for i in test_ids: seed_probs[s][i] = per[s][i]; seed_thr[s][i] = m.threshold
        avg = {i: np.mean([per[s][i] for s in SEEDS], 0) for i in val_ids + test_ids}
        vf1, thr = best_threshold(avg, va); log(f"fold {k + 1} ensemble: validation F1 {100 * vf1:.2f} at threshold {thr:.2f}")
        for i in test_ids: ens_probs[i] = avg[i]; ens_thr[i] = thr
        fold_info.append(dict(fold=k + 1, test=test_ids, validation=val_ids, ensemble_threshold=thr, ensemble_validation_f1=vf1,
                              seed_thresholds={s: models[j].threshold for j, s in enumerate(SEEDS)}))
        del models, per
        if a.device.startswith("cuda"): torch.cuda.empty_cache()
    res = {"folds": fold_info, "runs": {}}
    pd = a.out / "STARE" / "ctx_ens" / "test" / "predictions"; pd.mkdir(parents=True, exist_ok=True)
    for i, p in ens_probs.items(): np.save(pd / f"{i}_probability.npy", p.astype(np.float16))
    s, rows = pooled(cases, ens_probs, ens_thr); s.update(label="DCG-Net, 3-model ensemble", dataset="STARE")
    write_rows(a.out / "STARE" / "ctx_ens" / "test" / "per_image.csv", rows); res["runs"]["STARE/ctx_ens"] = s
    log("TEST STARE/ctx_ens (4-fold, pooled over 20 images): " + fmt(s))
    for sd in SEEDS:
        ss, rows = pooled(cases, seed_probs[sd], seed_thr[sd]); ss.update(label=f"DCG-Net (seed {sd})", dataset="STARE")
        d = a.out / "STARE" / f"ctx_s{sd}" / "test"; d.mkdir(parents=True, exist_ok=True); write_rows(d / "per_image.csv", rows)
        (d / "predictions").mkdir(exist_ok=True)
        for i, p in seed_probs[sd].items(): np.save(d / "predictions" / f"{i}_probability.npy", p.astype(np.float16))
        res["runs"][f"STARE/ctx_s{sd}"] = ss; log(f"TEST STARE/ctx_s{sd}: " + fmt(ss))
    (a.out / "stare_results.json").write_text(json.dumps(res, indent=2, default=float))
    (a.out / "ALL_DONE_STARE.txt").write_text("finished\n"); log(f"ALL DONE in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
