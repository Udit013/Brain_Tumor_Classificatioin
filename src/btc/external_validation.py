"""Module 3 — External (out-of-distribution) validation.

Evaluates the trained EfficientNetB3 on a SEPARATE public brain-tumor MRI
dataset that is NOT part of the Kaggle compilation it was trained on. This is
the single most honest signal of real-world generalisation, and the expected
result is a meaningful drop from the in-distribution 99.844%.

DEFAULT EXTERNAL SOURCE
-----------------------
Navoneel Chakrabarty, "Brain MRI Images for Brain Tumor Detection"
(Kaggle: navoneel/brain-mri-images-for-brain-tumor-detection). It is a binary
set (``yes`` = tumor present, ``no`` = no tumor) of ~253 images from a different
acquisition source, with no overlap with the training compilation. See
scripts/download_data.sh.

TAXONOMY HANDLING
-----------------
Our model predicts 4 classes {glioma, meningioma, notumor, pituitary}. When the
external set's taxonomy differs we evaluate two ways:
  * binary collapse: {glioma,meningioma,pituitary} -> "tumor", {notumor} -> "no_tumor",
    matched against the external yes/no labels. (Always available.)
  * overlapping classes: if the external set carries our class names in its
    subfolder layout, we also report 4-class metrics on the overlap.

Directory layout expected at BTC_EXTERNAL_DIR (override via env):
  external/
    yes/  *.jpg        # or "tumor"
    no/   *.jpg        # or "notumor" / "healthy"
  (or class-named subfolders: glioma/ meningioma/ notumor/ pituitary/)

Usage:
    python -m btc.external_validation
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config

# Map common external folder names -> our binary label.
_POSITIVE_NAMES = {"yes", "tumor", "tumour", "glioma", "meningioma", "pituitary"}
_NEGATIVE_NAMES = {"no", "notumor", "no_tumor", "healthy", "normal"}

TUMOR_CLASS_IDX = [
    config.CLASS_NAMES.index(c) for c in ("glioma", "meningioma", "pituitary")
]
NOTUMOR_IDX = config.CLASS_NAMES.index("notumor")


def _scan_external(root: Path):
    """Return list of (path, folder_name)."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"External dataset not found at {root}.\n"
            "Download a held-out source (see scripts/download_data.sh) or set "
            "BTC_EXTERNAL_DIR. Expected class- or yes/no-named subfolders."
        )
    items = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        for f in sub.iterdir():
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                items.append((f, sub.name.lower()))
    if not items:
        raise RuntimeError(f"No images found under {root}")
    return items


def _load_batch(paths):
    from tensorflow.keras.preprocessing.image import load_img, img_to_array

    xs = []
    for p in paths:
        img = load_img(str(p), target_size=config.IMG_SIZE, color_mode="rgb")
        xs.append(img_to_array(img))  # float32 [0,255], parity preprocessing
    return np.asarray(xs, dtype="float32")


def evaluate_external() -> dict:
    from .model import load_trained_model
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    items = _scan_external(config.EXTERNAL_DIR)
    paths = [p for p, _ in items]
    folders = [name for _, name in items]

    model = load_trained_model()
    x = _load_batch(paths)
    probs = model.predict(x, batch_size=config.BATCH_SIZE, verbose=0)
    pred_idx = np.argmax(probs, axis=1)

    # ---- binary collapse ------------------------------------------------ #
    pred_binary = np.where(np.isin(pred_idx, TUMOR_CLASS_IDX), 1, 0)  # 1=tumor
    true_binary, keep = [], []
    for name in folders:
        if name in _POSITIVE_NAMES:
            true_binary.append(1)
            keep.append(True)
        elif name in _NEGATIVE_NAMES:
            true_binary.append(0)
            keep.append(True)
        else:
            true_binary.append(-1)
            keep.append(False)
    true_binary = np.array(true_binary)
    keep = np.array(keep)

    result = {
        "external_source": str(config.EXTERNAL_DIR),
        "n_images": len(paths),
        "folders_seen": sorted(set(folders)),
        "note": (
            "Out-of-distribution evaluation; a large drop from the in-distribution "
            "99.844% is expected and is the honest generalisation signal."
        ),
    }

    if keep.any():
        tb, pb = true_binary[keep], pred_binary[keep]
        result["binary_collapse"] = {
            "n": int(keep.sum()),
            "accuracy": float(accuracy_score(tb, pb)),
            "report": classification_report(
                tb, pb, target_names=["no_tumor", "tumor"],
                output_dict=True, zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(tb, pb, labels=[0, 1]).tolist(),
        }

    # ---- overlapping 4-class (only if external uses our class names) ---- #
    overlap_mask = np.array([name in config.CLASS_NAMES for name in folders])
    if overlap_mask.any():
        true_multi = np.array(
            [config.CLASS_NAMES.index(name) if name in config.CLASS_NAMES else -1
             for name in folders]
        )
        tm, pm = true_multi[overlap_mask], pred_idx[overlap_mask]
        present = sorted(set(tm.tolist()))
        result["overlapping_classes"] = {
            "n": int(overlap_mask.sum()),
            "classes_present": [config.CLASS_NAMES[i] for i in present],
            "accuracy": float(accuracy_score(tm, pm)),
            "confusion_matrix": confusion_matrix(
                tm, pm, labels=list(range(config.NUM_CLASSES))
            ).tolist(),
        }

    return result


def main() -> None:
    config.ensure_dirs()
    result = evaluate_external()
    out = config.METRICS_DIR / "external_validation.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
