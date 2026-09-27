"""Train three DCG-Net models (seeds 1-3) on DRIVE or CHASE_DB1, build the ensemble and test once.

    python scripts/train_eval.py --data DATA_ROOT --dataset DRIVE --out runs

Four images of the official training set are held out for checkpoint and threshold selection
(DRIVE 22, 27, 34, 39; CHASE_DB1 09L, 09R, 10L, 10R). Test images are used only for the final evaluation.
Finished models are skipped on restart.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dcgnet as D  # noqa: E402


def write_rows(path, rows):
    with open(path, "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def fmt(s):
    return " ".join(f"{k} {100 * s[k]:.2f}" for k in ("f1", "sensitivity", "specificity", "accuracy", "auc", "thin_recall"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="folder with DRIVE/, DRIVET/, CHASE/, CHASET/")
    ap.add_argument("--dataset", choices=("DRIVE", "CHASE"), required=True)
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(); out = a.out / a.dataset; out.mkdir(parents=True, exist_ok=True)
    log = lambda m: (print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True), open(out / "log.txt", "a").write(m + "\n"))
    train_cases = D.load_dataset(a.data, a.dataset, "train"); test = D.load_dataset(a.data, a.dataset, "test")
    tr = [c for c in train_cases if c.id not in D.VAL_IDS[a.dataset]]; va = [c for c in train_cases if c.id in D.VAL_IDS[a.dataset]]
    models, results = [], {}
    for s in a.seeds:
        ck = out / f"seed{s}" / "model.pt"
        if not ck.exists():
            log(f"training seed {s} on {len(tr)} images ({len(va)} held out)")
            D.train(dict(seed=s, steps=a.steps, workers=a.workers), tr, va, out / f"seed{s}", a.device, log)
        m, t = D.load_checkpoint(ck, a.device); models.append(m)
        probs = {c.id: D.predict(m, c, a.device) for c in test}
        s_, rows = D.evaluate(probs, test, t); results[f"seed{s}"] = dict(s_, threshold=t)
        write_rows(out / f"seed{s}" / "per_image.csv", rows); log(f"TEST seed {s}: " + fmt(s_))
    vp = {c.id: D.predict_ensemble(models, c, a.device) for c in va}
    vf1, t = D.best_threshold(vp, va, 0.30, 0.90); log(f"ensemble threshold {t:.2f} (validation F1 {100 * vf1:.2f})")
    probs = {c.id: D.predict_ensemble(models, c, a.device) for c in test}
    s_, rows = D.evaluate(probs, test, t); results["ensemble"] = dict(s_, threshold=t)
    write_rows(out / "ensemble_per_image.csv", rows); log("TEST ensemble: " + fmt(s_))
    (out / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
