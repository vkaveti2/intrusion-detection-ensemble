"""
train_v5_rotation.py — CNN band rotation, a bounded diagnostic (docs/10_band_rotation.md).

Question: does the gap the cov1 CNN leaves at small payloads change with the band it was assigned?
In cov1 the CNN is hardened on band C (363-1460 B) while XGBoost sees A (1-90 B) and RF sees B
(91-362 B). Here only the CNN changes: it is hardened on band A or band B instead. XGBoost and RF
are the matching cov1 model-seed models, symlinked, so within a model seed the CNN's band is the
only difference from the cov1 baseline (cov1_ms1 / cov1_ms2, CNN on C).

Imports the frozen v5 training code unchanged (train_v5.py, train_adversarial.py); nothing frozen is edited.
Output: models_v5/cov1_cnn<band>_ms<seed>/   (a complete detector set; trees are symlinks)

    python train_v5_rotation.py --band A --model-seed 1
"""

import argparse
import json
import os

import numpy as np

import train_v5
from train_adversarial import AUG_SEED, BANDS, sample_payloads

CNN_INDEX = 2   # position of "cnn" in DETECTORS; cov1 draws each detector's payloads from rng([AUG_SEED, index])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", choices=["A", "B"], required=True, help="band the CNN is hardened on (C is the cov1 baseline)")
    ap.add_argument("--model-seed", type=int, required=True, choices=[1, 2],
                    help="must match an existing cov1_ms<seed> so the trees pair with the baseline")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    tag = f"cov1_cnn{args.band}_ms{args.model_seed}"
    base = f"cov1_ms{args.model_seed}"
    out_dir = os.path.join(train_v5.OUT_ROOT, tag)

    data = train_v5.V5Data("cov1")                      # same train rows, scaler and padded-copy sources as cov1
    n = len(data.base)
    lo, hi = BANDS[args.band]
    data.plan["cnn"] = sample_payloads(np.random.default_rng([AUG_SEED, CNN_INDEX]), n, [args.band])
    data.meta.update({
        "design": tag, "rotation": True, "cnn_band": args.band, "cnn_band_bytes": [lo, hi],
        "trees_from": base, "baseline_cnn_band": "C",
        "detector_bands": {"xgboost": ["A"], "rf": ["B"], "cnn": [args.band]},
        "payload_bytes_per_detector": {d: {"min": int(p.min()), "max": int(p.max()), "mean": round(float(p.mean()), 1)}
                                       for d, p in data.plan.items()},
    })

    train_v5.prepare_dir(out_dir, data)
    for name in ("xgboost_model.pkl", "rf_model.pkl"):
        link = os.path.join(out_dir, name)
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(os.path.join("..", base, name), link)
    with open(os.path.join(out_dir, "augmentation.json"), "w") as f:
        json.dump({**data.meta, "model_seed": args.model_seed}, f, indent=2)
    print(json.dumps(data.meta, indent=2), flush=True)
    train_v5.train_cnn_stage(data, out_dir, tag, args.model_seed, args.epochs)


if __name__ == "__main__":
    main()
