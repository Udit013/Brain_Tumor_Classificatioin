"""Uncertainty estimation via Test-Time Augmentation (TTA).

Lightweight and CPU-friendly (a handful of forward passes, no gradient work),
so it runs comfortably on free-tier Spaces. For a single image we run the model
over a small set of label-preserving augmentations (horizontal flip, small
rotations, mild brightness shifts), average the softmax probabilities, and
summarise dispersion as:

  * ``predictive_entropy`` — entropy of the mean probability vector (total
    uncertainty; higher = less certain).
  * ``mean_top_prob`` / ``std_top_prob`` — mean and spread of the predicted
    class probability across augmentations (epistemic-ish stability signal).

The same routine, applied over a labelled set, yields a mean-uncertainty metric
we log during evaluation.
"""

from __future__ import annotations

import numpy as np

from . import config


def _augmentations(x: np.ndarray) -> list[np.ndarray]:
    """Return label-preserving views of a single (H,W,3) [0,255] image."""
    import cv2

    h, w = x.shape[:2]
    views = [x, x[:, ::-1, :]]  # identity + horizontal flip
    for angle in (-8, 8):
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        views.append(cv2.warpAffine(x, m, (w, h), borderMode=cv2.BORDER_REFLECT))
    for factor in (0.9, 1.1):  # mild brightness
        views.append(np.clip(x * factor, 0, 255).astype("float32"))
    return [v.astype("float32") for v in views]


def _predict_fn_from_model(model):
    def _fn(batch: np.ndarray) -> np.ndarray:
        return model.predict(batch, verbose=0)
    return _fn


def tta_predict(x_single, predict_fn, temperature: float = 1.0) -> dict:
    """TTA uncertainty for one image.

    Parameters
    ----------
    x_single : np.ndarray
        (H, W, 3) float image in [0, 255] (parity preprocessing).
    predict_fn : callable
        Maps a (N, H, W, 3) batch to (N, num_classes) softmax probabilities.
        Accepts a TF model's ``predict`` or an ONNX-backed closure.
    temperature : float
        Optional temperature scaling applied to the averaged probabilities.
    """
    views = np.stack(_augmentations(np.asarray(x_single, dtype="float32")))
    probs = np.asarray(predict_fn(views))  # (n_aug, C)
    if temperature != 1.0:
        logits = np.log(np.clip(probs, 1e-12, 1.0)) / temperature
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)

    mean_p = probs.mean(axis=0)
    top = int(mean_p.argmax())
    entropy = float(-(mean_p * np.log(np.clip(mean_p, 1e-12, 1.0))).sum())
    return {
        "class_index": top,
        "class_name": config.CLASS_NAMES[top],
        "mean_probabilities": mean_p.tolist(),
        "confidence": float(mean_p[top]),
        "predictive_entropy": entropy,
        "mean_top_prob": float(probs[:, top].mean()),
        "std_top_prob": float(probs[:, top].std()),
        "n_augmentations": int(len(views)),
    }


def mean_dataset_uncertainty(model=None, limit: int = 200) -> dict:
    """Average TTA uncertainty over a balanced slice of the test set (logged)."""
    from .model import load_trained_model
    from .data import load_test_arrays

    model = model or load_trained_model()
    fn = _predict_fn_from_model(model)
    x, y, _ = load_test_arrays()
    per = max(1, limit // config.NUM_CLASSES)
    idx = np.concatenate([np.where(y == c)[0][:per] for c in range(config.NUM_CLASSES)])

    entropies, correct_ent, wrong_ent = [], [], []
    for i in idx:
        r = tta_predict(x[i], fn)
        entropies.append(r["predictive_entropy"])
        (correct_ent if r["class_index"] == y[i] else wrong_ent).append(
            r["predictive_entropy"])
    return {
        "n": int(len(idx)),
        "mean_predictive_entropy": float(np.mean(entropies)),
        "mean_entropy_correct": float(np.mean(correct_ent)) if correct_ent else None,
        "mean_entropy_wrong": float(np.mean(wrong_ent)) if wrong_ent else None,
        "note": "Higher entropy on wrong predictions indicates useful uncertainty.",
    }


def main() -> None:
    import json

    config.ensure_dirs()
    res = mean_dataset_uncertainty()
    (config.METRICS_DIR / "uncertainty.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
