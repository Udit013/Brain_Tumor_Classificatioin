"""Module — Robustness testing under common image corruptions.

Applies six corruption families at increasing severities to the test set and
measures accuracy degradation, quantifying how gracefully the model fails under
realistic acquisition/preprocessing perturbations:

  * gaussian_noise, gaussian_blur, brightness, contrast, jpeg_compression, rotation

Outputs:
  results/metrics/robustness.json         (accuracy per corruption x severity)
  results/figures/robustness_curves.png   (degradation curves)

Usage:
    python -m btc.robustness [--limit N]
"""

from __future__ import annotations

import argparse
import io
import json

import numpy as np

from . import config

SEVERITIES = [1, 2, 3, 4, 5]


def _corrupt(x: np.ndarray, kind: str, sev: int) -> np.ndarray:
    """Corrupt a single (H,W,3) [0,255] image at severity 1..5."""
    import cv2
    from PIL import Image

    x = np.asarray(x, dtype="float32")
    if kind == "gaussian_noise":
        std = [8, 16, 24, 34, 46][sev - 1]
        return np.clip(x + np.random.normal(0, std, x.shape), 0, 255)
    if kind == "gaussian_blur":
        k = [3, 5, 7, 9, 11][sev - 1]
        return cv2.GaussianBlur(x, (k, k), 0)
    if kind == "brightness":
        delta = [15, 30, 45, 60, 80][sev - 1]
        return np.clip(x + delta, 0, 255)
    if kind == "contrast":
        f = [0.85, 0.7, 0.55, 0.4, 0.3][sev - 1]
        mean = x.mean(axis=(0, 1), keepdims=True)
        return np.clip((x - mean) * f + mean, 0, 255)
    if kind == "jpeg_compression":
        q = [50, 35, 25, 15, 8][sev - 1]
        buf = io.BytesIO()
        Image.fromarray(x.astype("uint8")).save(buf, format="JPEG", quality=q)
        return np.asarray(Image.open(buf).convert("RGB"), dtype="float32")
    if kind == "rotation":
        angle = [5, 10, 15, 22, 30][sev - 1]
        h, w = x.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(x, m, (w, h), borderMode=cv2.BORDER_REFLECT)
    raise ValueError(kind)


CORRUPTIONS = ["gaussian_noise", "gaussian_blur", "brightness",
               "contrast", "jpeg_compression", "rotation"]


def evaluate_robustness(limit: int = 400) -> dict:
    from .model import load_trained_model
    from .data import load_test_arrays

    np.random.seed(config.SEED)
    model = load_trained_model()
    x, y, _ = load_test_arrays()

    # balanced subset for speed
    per = max(1, limit // config.NUM_CLASSES)
    idx = np.concatenate([np.where(y == c)[0][:per] for c in range(config.NUM_CLASSES)])
    xs, ys = x[idx], y[idx]

    clean_acc = float((model.predict(xs, batch_size=16, verbose=0).argmax(1) == ys).mean())
    results = {"n": int(len(ys)), "clean_accuracy": clean_acc, "severities": SEVERITIES,
               "corruptions": {}}
    for kind in CORRUPTIONS:
        accs = []
        for sev in SEVERITIES:
            xc = np.stack([_corrupt(im, kind, sev) for im in xs]).astype("float32")
            acc = float((model.predict(xc, batch_size=16, verbose=0).argmax(1) == ys).mean())
            accs.append(acc)
        results["corruptions"][kind] = accs
        print(f"{kind:18} {['%.3f'%a for a in accs]}")

    # mean corruption accuracy across all corruptions/severities
    allv = [a for accs in results["corruptions"].values() for a in accs]
    results["mean_corruption_accuracy"] = float(np.mean(allv))
    results["mean_relative_drop"] = float(clean_acc - np.mean(allv))
    return results


def _plot(results, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for kind, accs in results["corruptions"].items():
        ax.plot(results["severities"], accs, marker="o", label=kind)
    ax.axhline(results["clean_accuracy"], color="k", ls="--", alpha=0.5,
               label=f"clean ({results['clean_accuracy']:.3f})")
    ax.set(xlabel="Corruption severity", ylabel="Accuracy",
           title="Robustness to common image corruptions", ylim=(0, 1))
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Robustness / corruption testing")
    parser.add_argument("--limit", type=int, default=400)
    args = parser.parse_args()

    config.ensure_dirs()
    results = evaluate_robustness(limit=args.limit)
    (config.METRICS_DIR / "robustness.json").write_text(json.dumps(results, indent=2))
    _plot(results, config.FIGURES_DIR / "robustness_curves.png")
    print(f"\nclean={results['clean_accuracy']:.3f} "
          f"mean_corrupted={results['mean_corruption_accuracy']:.3f} "
          f"drop={results['mean_relative_drop']:.3f}")
    print(f"Saved -> {config.METRICS_DIR / 'robustness.json'}")


if __name__ == "__main__":
    main()
