"""Evaluate released DCG-Net weights on the DRIVE or CHASE_DB1 test set (single models and ensemble).

    python scripts/evaluate.py --data DATA_ROOT --dataset DRIVE --weights weights/dcgnet_drive_seed*.pt --ensemble-threshold 0.52
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dcgnet as D  # noqa: E402

ENSEMBLE_THRESHOLD = {"DRIVE": 0.52, "CHASE": 0.72}  # selected on the held-out validation images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True); ap.add_argument("--dataset", choices=("DRIVE", "CHASE"), required=True)
    ap.add_argument("--weights", type=Path, nargs="+", required=True)
    ap.add_argument("--ensemble-threshold", type=float, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(); test = D.load_dataset(a.data, a.dataset, "test")
    loaded = [D.load_checkpoint(w, a.device) for w in a.weights]
    per = []
    for w, (m, t) in zip(a.weights, loaded):
        probs = {c.id: D.predict(m, c, a.device) for c in test}; per.append(probs)
        s, _ = D.evaluate(probs, test, t)
        print(f"{w.name}: F1 {100 * s['f1']:.2f}  Se {100 * s['sensitivity']:.2f}  Sp {100 * s['specificity']:.2f}  AUC {100 * s['auc']:.2f}")
    if len(per) > 1:
        t = a.ensemble_threshold or ENSEMBLE_THRESHOLD[a.dataset]
        probs = {c.id: sum(p[c.id] for p in per) / len(per) for c in test}
        s, _ = D.evaluate(probs, test, t)
        print(f"ensemble (threshold {t:.2f}): F1 {100 * s['f1']:.2f}  Se {100 * s['sensitivity']:.2f}  Sp {100 * s['specificity']:.2f}  "
              f"Acc {100 * s['accuracy']:.2f}  AUC {100 * s['auc']:.2f}  thin Se {100 * s['thin_recall']:.2f}")


if __name__ == "__main__":
    main()
