"""Leakage-safe train/validation split carved out of the Kaggle Training folder.

WHY THIS EXISTS
---------------
The published protocol has no validation set: every notebook (and the parity
reproduction in train.py) picks its checkpoint by accuracy on the *test* set,
so test metrics are optimistically biased. Any further model improvement must
be selected on held-out data the test set never influences.

A plain random split would not be held out either. Training contains clusters
of near-identical MRI slices from the same scan (the same problem leakage.py
measures across train/test), so a random split puts siblings on both sides and
validation accuracy becomes optimistic. This module instead groups Training
images into near-duplicate clusters (perceptual-hash Hamming distance <=
PHASH_NEAR_DUP_THRESHOLD, plus exact pixel duplicates) and assigns whole
clusters to one side, stratified by class.

Output (small, tracked in git so the split is reproducible):
  results/splits/train_val_split.json

Usage:
    python -m btc.splits [--val-fraction 0.15]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import config

SPLIT_PATH = config.RESULTS_DIR / "splits" / "train_val_split.json"


def _cluster(bits: np.ndarray, chashes: list[str], threshold: int) -> np.ndarray:
    """Union-find over near-duplicate pairs; returns a cluster id per image."""
    n = len(bits)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Pairwise Hamming distances via the matmul identity used in leakage.py,
    # computed in row blocks to bound memory.
    sums = bits.sum(axis=1)
    for start in range(0, n, 1024):
        blk = bits[start:start + 1024]
        d = sums[start:start + 1024, None] + sums[None, :] - 2.0 * (blk @ bits.T)
        rows, cols = np.nonzero(np.rint(d) <= threshold)
        for r, c in zip(rows + start, cols):
            if r < c:
                union(r, c)
    # Exact pixel duplicates (content hash) always share a cluster.
    first = {}
    for i, h in enumerate(chashes):
        if h in first:
            union(first[h], i)
        else:
            first[h] = i
    return np.array([find(i) for i in range(n)])


def build_split(val_fraction: float = 0.15, seed: int = config.SEED) -> dict:
    from .data import build_dataframes
    from .leakage import PHASH_NEAR_DUP_THRESHOLD, _content_hash, _phash

    train_df, _ = build_dataframes()
    paths, labels = list(train_df["filepaths"]), list(train_df["label"])
    bits = np.stack([np.asarray(_phash(p).hash, dtype=np.float32).ravel() for p in paths])
    chashes = [_content_hash(p) for p in paths]
    cluster = _cluster(bits, chashes, PHASH_NEAR_DUP_THRESHOLD)

    # Each cluster takes its majority label; clusters are shuffled and assigned
    # to validation per class until that class reaches its validation quota.
    members = defaultdict(list)
    for i, c in enumerate(cluster):
        members[c].append(i)
    rng = np.random.default_rng(seed)
    by_class = defaultdict(list)
    for c, idx in members.items():
        by_class[Counter(labels[i] for i in idx).most_common(1)[0][0]].append(c)

    val_idx = set()
    for cls in config.CLASS_NAMES:
        quota = round(val_fraction * sum(1 for l in labels if l == cls))
        taken = 0
        for c in rng.permutation(by_class[cls]):
            if taken >= quota:
                break
            val_idx.update(members[c])
            taken += len(members[c])

    def rel(p: str) -> str:
        return str(Path(p).relative_to(config.DATA_DIR))

    train = [rel(paths[i]) for i in range(len(paths)) if i not in val_idx]
    val = [rel(paths[i]) for i in sorted(val_idx)]
    sizes = Counter(cluster.tolist()).values()
    return {
        "description": "Leakage-safe split of Kaggle Training/ into train/val; "
                       "near-duplicate clusters never straddle the two sides.",
        "seed": seed,
        "val_fraction": val_fraction,
        "phash_threshold": PHASH_NEAR_DUP_THRESHOLD,
        "n_train": len(train),
        "n_val": len(val),
        "n_clusters": len(members),
        "n_multi_image_clusters": sum(1 for s in sizes if s > 1),
        "largest_cluster": max(sizes),
        "images_in_multi_image_clusters": sum(s for s in sizes if s > 1),
        "val_per_class": dict(Counter(labels[i] for i in val_idx)),
        "train": train,
        "val": val,
    }


def load_split() -> dict:
    """Return the saved split with absolute paths and integer labels."""
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(f"{SPLIT_PATH} missing — run `python -m btc.splits`.")
    s = json.loads(SPLIT_PATH.read_text())
    idx = {n: i for i, n in enumerate(config.CLASS_NAMES)}
    out = {}
    for part in ("train", "val"):
        paths = [str(config.DATA_DIR / p) for p in s[part]]
        out[part] = (paths, np.array([idx[p.split("/")[-2]] for p in s[part]]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-safe train/val split")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    args = parser.parse_args()
    split = build_split(args.val_fraction)
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_PATH.write_text(json.dumps(split, indent=1))
    print(json.dumps({k: v for k, v in split.items() if k not in ("train", "val")}, indent=2))
    print(f"Saved -> {SPLIT_PATH}")


if __name__ == "__main__":
    main()
