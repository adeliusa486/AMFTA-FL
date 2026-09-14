"""
Build a stratified subsample of the processed NF-TON-IoT tensors.

The full split is 9.2M training flows, and a single 20-client run over it at the
paper's settings takes over half an hour. For the compact controlled study of
the redesign, what matters is that the class balance and feature distribution
are preserved, not that every flow is present.

    python scripts/make_subsample.py --n 500000 --out data/processed_500k

Class proportions are preserved exactly; the sample is drawn with a fixed seed
so the subset is reproducible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def stratified(X: np.ndarray, y: np.ndarray, n: int, rng) -> tuple:
    if n >= len(y):
        return X, y
    idx = []
    for cls in np.unique(y):
        pool = np.flatnonzero(y == cls)
        take = int(round(n * len(pool) / len(y)))
        take = min(max(take, 1), len(pool))
        idx.append(rng.choice(pool, size=take, replace=False))
    sel = np.concatenate(idx)
    rng.shuffle(sel)
    return X[sel], y[sel]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/processed")
    ap.add_argument("--out", default="data/processed_500k")
    ap.add_argument("--n", type=int, default=500_000, help="training rows to keep")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    # test and validation shrink in proportion so the ratios stay put
    for split in ("train", "val", "test"):
        f = src / f"{split}.npz"
        if not f.exists():
            print(f"  skip {split}: not found")
            continue
        z = np.load(f)
        X, y = z["X"], z["y"]
        share = a.n if split == "train" else max(1, int(a.n * len(y) / 9_195_116))
        Xs, ys = stratified(X, y, share, rng)
        np.savez_compressed(out / f"{split}.npz", X=Xs, y=ys)
        print(f"  {split}: {len(y):>9,} -> {len(ys):>8,}   "
              f"positive fraction {y.mean():.4f} -> {ys.mean():.4f}")

    for extra in ("feature_names.txt", "scaler.npz"):
        if (src / extra).exists():
            (out / extra).write_bytes((src / extra).read_bytes())

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
