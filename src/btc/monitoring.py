"""Module 7 — Input-drift monitoring stub.

Detects distribution shift in incoming images by comparing their embeddings
against a reference distribution captured from the training data. The embedding
is EfficientNetB3's penultimate 1536-d global-max-pool feature. We reduce it
with a fitted PCA and quantify drift with the Population Stability Index (PSI)
per component plus an aggregate score, and a mean Mahalanobis distance.

Two entry points:
  build-reference : fit PCA + reference histograms/stats from the training set
                    (run once, persisted to models/drift_reference.npz)
  detect          : score a batch of images and append a metric to the drift log

A PSI > 0.2 on the aggregate is the conventional "significant shift" alert
threshold; we log the value rather than hard-failing (this is a stub).

Outputs:
  models/drift_reference.npz
  results/metrics/drift_log.jsonl   (one JSON record per detect call)

Usage:
    python -m btc.monitoring build-reference [--limit N]
    python -m btc.monitoring detect --dir PATH/TO/IMAGES
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import config

N_PCA = 8
PSI_ALERT = 0.2
DRIFT_LOG = config.METRICS_DIR / "drift_log.jsonl"


def _embedder(model=None):
    """Return (embedder, model) where embedder maps input -> 1536-d embedding.

    The nested ``efficientnetb3`` submodel is itself a Model from the 256x256x3
    input to the (None, 1536) global-max-pooled embedding, so we use it directly.
    Rewrapping it as Model(model.input, base.output) triggers a Keras
    "Graph disconnected" error on this nested architecture.
    """
    from .model import load_trained_model

    model = model or load_trained_model()
    base = model.get_layer("efficientnetb3")  # outputs (None, 1536) with pooling='max'
    return base, model


def _embed_paths(paths, embedder, batch=32):
    from tensorflow.keras.preprocessing.image import load_img, img_to_array

    embs = []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        xs = np.asarray(
            [img_to_array(load_img(str(p), target_size=config.IMG_SIZE, color_mode="rgb"))
             for p in chunk], dtype="float32")
        embs.append(embedder.predict(xs, verbose=0))
    return np.concatenate(embs, axis=0)


def _list_images(root: Path):
    root = Path(root)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts and p.is_file()]


def _psi(ref_counts, new_vals, edges):
    """Population Stability Index between reference bin proportions and new sample."""
    new_counts, _ = np.histogram(new_vals, bins=edges)
    ref_p = ref_counts / max(ref_counts.sum(), 1)
    new_p = new_counts / max(new_counts.sum(), 1)
    eps = 1e-6
    ref_p = np.clip(ref_p, eps, None)
    new_p = np.clip(new_p, eps, None)
    return float(np.sum((new_p - ref_p) * np.log(new_p / ref_p)))


def build_reference(limit: int | None = None) -> None:
    from sklearn.decomposition import PCA
    from .data import build_dataframes

    config.ensure_dirs()
    train_df, _ = build_dataframes()
    paths = list(train_df["filepaths"])
    if limit:
        rng = np.random.default_rng(config.SEED)
        paths = list(rng.permutation(paths)[:limit])

    embedder, _ = _embedder()
    emb = _embed_paths(paths, embedder)

    pca = PCA(n_components=N_PCA, random_state=config.SEED).fit(emb)
    proj = pca.transform(emb)

    # Per-component histograms (10 bins between observed min/max) for PSI.
    edges, counts = [], []
    for k in range(N_PCA):
        e = np.linspace(proj[:, k].min(), proj[:, k].max(), 11)
        c, _ = np.histogram(proj[:, k], bins=e)
        edges.append(e)
        counts.append(c)

    np.savez(
        config.DRIFT_REFERENCE_PATH,
        pca_components=pca.components_,
        pca_mean=pca.mean_,
        ref_mean=proj.mean(axis=0),
        ref_cov_inv=np.linalg.pinv(np.cov(proj, rowvar=False)),
        edges=np.array(edges),
        counts=np.array(counts),
        n_ref=len(paths),
    )
    print(f"Saved drift reference ({len(paths)} train images) -> "
          f"{config.DRIFT_REFERENCE_PATH}")


def detect(image_dir: str, tag: str = "") -> dict:
    if not config.DRIFT_REFERENCE_PATH.exists():
        raise FileNotFoundError(
            "Drift reference missing. Run `python -m btc.monitoring build-reference` first."
        )
    ref = np.load(config.DRIFT_REFERENCE_PATH)
    paths = _list_images(Path(image_dir))
    if not paths:
        raise RuntimeError(f"No images found under {image_dir}")

    embedder, _ = _embedder()
    emb = _embed_paths(paths, embedder)
    proj = (emb - ref["pca_mean"]) @ ref["pca_components"].T

    psi_per = [
        _psi(ref["counts"][k], proj[:, k], ref["edges"][k]) for k in range(N_PCA)
    ]
    psi_agg = float(np.mean(psi_per))

    diff = proj - ref["ref_mean"]
    mahalanobis = float(np.sqrt(np.einsum("ij,jk,ik->i", diff, ref["ref_cov_inv"], diff)).mean())

    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tag": tag,
        "image_dir": str(image_dir),
        "n_images": len(paths),
        "psi_aggregate": psi_agg,
        "psi_per_component": [float(v) for v in psi_per],
        "mean_mahalanobis": mahalanobis,
        "psi_alert_threshold": PSI_ALERT,
        "drift_detected": bool(psi_agg > PSI_ALERT),
    }
    config.ensure_dirs()
    with open(DRIFT_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Input-drift monitoring")
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-reference")
    b.add_argument("--limit", type=int, default=None)
    d = sub.add_parser("detect")
    d.add_argument("--dir", required=True)
    d.add_argument("--tag", default="")
    args = parser.parse_args()

    if args.cmd == "build-reference":
        build_reference(limit=args.limit)
    else:
        rec = detect(args.dir, tag=args.tag)
        print(json.dumps(rec, indent=2))
        print(f"\nAppended -> {DRIFT_LOG}")


if __name__ == "__main__":
    main()
