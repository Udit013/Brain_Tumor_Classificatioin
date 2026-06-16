"""Module 2 — Train/test leakage analysis (the most important extension).

WHY THIS MATTERS
----------------
The published Kaggle "Brain Tumor MRI Dataset" is a *compilation* of three
sources (figshare CE-MRI, SARTAJ, Br35H) with an **image-level** train/test
split. MRI volumes are 3-D; a single patient contributes many adjacent 2-D
slices that look nearly identical. When the split is performed per-image rather
than per-patient, near-identical slices from the same scan can land on BOTH
sides of the split. A model can then "recognise" a test slice because it
memorised a sibling slice during training — inflating reported accuracy in a
way that will NOT transfer to new patients.

WHAT WE CAN AND CANNOT MEASURE
------------------------------
* Patient/scan IDs are **not recoverable** from the compiled Kaggle release:
  the figshare source carries patient IDs (PID in the original .mat files) but
  the Kaggle repackaging stores only flat per-class JPGs with re-indexed
  filenames, and the SARTAJ / Br35H sources never carried patient IDs through.
  Therefore a rigorous *patient-level* leakage audit is impossible on this
  artifact. We state this limitation explicitly rather than fabricate IDs.
* What we CAN measure is a lower bound on leakage via image similarity:
  exact-duplicate detection (content hash) and near-duplicate detection
  (perceptual hash, Hamming distance). Cross-split near-duplicates are direct
  evidence of the slice-sibling problem above.

WHAT THIS MODULE PRODUCES
-------------------------
  results/metrics/leakage.json   — duplicate counts + leak-adjusted accuracy
  results/figures/leakage_nn_hist.png — histogram of each test image's nearest
                                        train perceptual distance

Usage:
    python -m btc.leakage            # hashing analysis (no TF needed)
    python -m btc.leakage --eval     # also re-score accuracy on leak-free subset
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict

import numpy as np

from . import config

# Hamming distance (over a 64-bit perceptual hash) at/below which two images are
# treated as near-duplicates. 0 = identical hash; <=5 is the conventional
# "visually near-identical" band for phash.
PHASH_NEAR_DUP_THRESHOLD = 5


def _content_hash(path: str) -> str:
    """MD5 of the decoded, resized grayscale pixels — robust to re-encoding."""
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("L").resize((64, 64))
        return hashlib.md5(im.tobytes()).hexdigest()


def _phash(path: str):
    import imagehash
    from PIL import Image

    with Image.open(path) as im:
        return imagehash.phash(im)  # 64-bit perceptual hash


def _collect(df):
    paths = list(df["filepaths"])
    labels = list(df["label"])
    return paths, labels


def analyse(eval_clean_subset: bool = False) -> dict:
    from .data import build_dataframes

    train_df, test_df = build_dataframes()
    tr_paths, tr_labels = _collect(train_df)
    te_paths, te_labels = _collect(test_df)

    # ---- exact duplicates (content hash) -------------------------------- #
    tr_chash = {}
    for p in tr_paths:
        tr_chash.setdefault(_content_hash(p), []).append(p)
    exact_dupe_test = []
    for p in te_paths:
        h = _content_hash(p)
        if h in tr_chash:
            exact_dupe_test.append(p)

    # ---- near duplicates (perceptual hash) ----------------------------- #
    tr_phash = [(_phash(p), lab) for p, lab in zip(tr_paths, tr_labels)]
    nn_distances = []          # nearest train phash distance for each test img
    near_dupe_flags = []       # bool per test image
    cross_label_near = 0       # near-dup whose train match has a DIFFERENT label
    for p, lab in zip(te_paths, te_labels):
        h = _phash(p)
        best, best_lab = min(((h - th, tl) for th, tl in tr_phash),
                             key=lambda t: t[0])
        nn_distances.append(int(best))
        is_near = best <= PHASH_NEAR_DUP_THRESHOLD
        near_dupe_flags.append(is_near)
        if is_near and best_lab != lab:
            cross_label_near += 1

    near_dupe_flags = np.array(near_dupe_flags)
    nn_distances = np.array(nn_distances)
    n_test = len(te_paths)

    result = {
        "n_train": len(tr_paths),
        "n_test": n_test,
        "exact_duplicates_in_test": len(exact_dupe_test),
        "exact_duplicate_rate": len(exact_dupe_test) / n_test,
        "phash_near_dup_threshold": PHASH_NEAR_DUP_THRESHOLD,
        "near_duplicates_in_test": int(near_dupe_flags.sum()),
        "near_duplicate_rate": float(near_dupe_flags.mean()),
        "near_dup_cross_label": int(cross_label_near),
        "nn_distance_min": int(nn_distances.min()),
        "nn_distance_median": float(np.median(nn_distances)),
        "nn_distance_mean": float(nn_distances.mean()),
        "patient_level_audit_possible": False,
        "patient_id_limitation": (
            "Patient/scan IDs are not recoverable from the compiled Kaggle "
            "release (flat per-class JPGs, re-indexed filenames; SARTAJ/Br35H "
            "never carried IDs). Image-similarity duplicates below are a LOWER "
            "BOUND on true patient-level leakage."
        ),
    }

    # ---- leak-adjusted accuracy on the leak-free test subset ----------- #
    if eval_clean_subset:
        from .evaluate import predict_test
        from sklearn.metrics import accuracy_score

        y_true, y_prob = predict_test()
        y_pred = np.argmax(y_prob, axis=1)
        # predict_test() iterates test images in the same build_dataframes()
        # order, so near_dupe_flags aligns index-for-index.
        clean_mask = ~near_dupe_flags
        result["accuracy_full_test"] = float(accuracy_score(y_true, y_pred))
        result["accuracy_leak_free_subset"] = (
            float(accuracy_score(y_true[clean_mask], y_pred[clean_mask]))
            if clean_mask.any() else None
        )
        result["accuracy_near_dup_subset"] = (
            float(accuracy_score(y_true[near_dupe_flags], y_pred[near_dupe_flags]))
            if near_dupe_flags.any() else None
        )
        result["n_leak_free"] = int(clean_mask.sum())

    return result, nn_distances


def _plot_hist(nn_distances, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(nn_distances, bins=range(0, max(nn_distances) + 2), edgecolor="black")
    ax.axvline(PHASH_NEAR_DUP_THRESHOLD + 0.5, color="red", linestyle="--",
               label=f"near-dup threshold ({PHASH_NEAR_DUP_THRESHOLD})")
    ax.set(xlabel="Nearest train perceptual-hash distance",
           ylabel="# test images",
           title="Test-image proximity to training set (lower = more leakage)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/test leakage analysis")
    parser.add_argument("--eval", action="store_true",
                        help="also re-score accuracy on the leak-free subset "
                             "(requires trained weights + tensorflow)")
    args = parser.parse_args()

    config.ensure_dirs()
    result, nn_distances = analyse(eval_clean_subset=args.eval)
    out = config.METRICS_DIR / "leakage.json"
    out.write_text(json.dumps(result, indent=2))
    _plot_hist(nn_distances, config.FIGURES_DIR / "leakage_nn_hist.png")

    print(json.dumps(result, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
