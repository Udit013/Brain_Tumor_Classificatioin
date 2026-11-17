"""Measure the reliability rules and the input-shift heuristic.

For the deployed serving bundle (platform environment, no TensorFlow):

  * accuracy and coverage per reliability level (standard / caution /
    unreliable) on the validation split and the test set;
  * AUROC of the Mahalanobis input-shift distance for
      - near-OOD: test set (in-distribution) vs the external MRI dataset;
      - far-OOD: test set vs synthetic non-MRI images (noise, gradients,
        solid colours, colour-tinted MRIs, checkerboards);
    and the fraction of each group flagged at the deployed threshold.

The rules and threshold were fixed before this script ran; nothing is tuned
here. Writes results/metrics/reliability_eval.json.

Usage (platform environment):
    PYTHONPATH=src python scripts/eval_reliability.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from btc import config  # noqa: E402
from neuroclass.reliability import RULES_VERSION, assess  # noqa: E402
from neuroclass.vision.service import VisionService  # noqa: E402
from neuroclass.vision.validation import assess_quality  # noqa: E402

CLASSES = config.CLASS_NAMES


def auroc(neg: np.ndarray, pos: np.ndarray) -> float:
    """P(score_pos > score_neg), ties counted half (Mann-Whitney U)."""
    scores = np.concatenate([neg, pos])
    ranks = scores.argsort().argsort().astype("float64") + 1
    # average ranks for ties
    order = np.argsort(scores)
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    r_pos = ranks[len(neg):].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def synthetic_images(rng: np.random.Generator, mri_paths: list[str], n_each: int = 20):
    """Clearly non-MRI inputs, generated deterministically."""
    out = []
    for _ in range(n_each):
        out.append(("noise", Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8))))
        c = rng.integers(0, 256, 3)
        out.append(("solid_color", Image.fromarray(np.full((256, 256, 3), c, dtype=np.uint8))))
        g = np.linspace(0, 255, 256, dtype=np.float32)
        grad = np.stack([np.add.outer(g, g) / 2 * rng.uniform(0.3, 1)] * 3, axis=-1)
        out.append(("gradient", Image.fromarray(grad.astype(np.uint8))))
        k = int(rng.integers(8, 64))
        cb = ((np.indices((256, 256)) // k).sum(axis=0) % 2 * 255).astype(np.uint8)
        out.append(("checkerboard", Image.fromarray(np.stack([cb] * 3, axis=-1))))
    for p in rng.choice(mri_paths, n_each, replace=False):
        arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)
        tint = rng.uniform(0.3, 1.0, 3)
        out.append(("tinted_mri", Image.fromarray(np.clip(arr * tint, 0, 255).astype(np.uint8))))
    return out


def run(svc: VisionService, images) -> list[dict]:
    rows = []
    for label, img in images:
        # Round-trip through JPEG bytes like an upload, so quality checks see a real file.
        if not isinstance(img, Image.Image):
            img = Image.open(img)
        q = assess_quality(img)
        r = svc.predict(img, q, with_gradcam=False)
        a = assess(r)
        rows.append({"label": label, "pred": r.predicted_class, "conf": r.calibrated_confidence,
                     "ood": r.ood_distance, "ood_flag": r.ood_flag, "level": a.level,
                     "tta": r.tta_agreement})
    return rows


def level_table(rows: list[dict]) -> dict:
    out = {}
    for level in ("standard", "caution", "unreliable"):
        sel = [r for r in rows if r["level"] == level]
        out[level] = {"n": len(sel), "coverage": len(sel) / max(1, len(rows)),
                      "accuracy": (sum(r["pred"] == r["label"] for r in sel) / len(sel)) if sel else None}
    out["all"] = {"n": len(rows), "accuracy": sum(r["pred"] == r["label"] for r in rows) / len(rows)}
    return out


def main() -> None:
    from btc.splits import load_split

    t0 = time.time()
    svc = VisionService(ROOT / "models" / "serving", timeout_s=60)
    split = load_split()
    val = [(CLASSES[y], p) for p, y in zip(*split["val"])]
    test = [(lab, str(p)) for lab in CLASSES
            for p in sorted((config.TEST_DIR / lab).iterdir()) if p.is_file()]
    ext_root = config.EXTERNAL_DIR
    external = [("external", str(p)) for sub in sorted(ext_root.iterdir()) if sub.is_dir()
                for p in sorted(sub.iterdir()) if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    rng = np.random.default_rng(config.SEED)
    synth = synthetic_images(rng, [p for _, p in test])

    rows = {name: run(svc, items) for name, items in
            (("val", val), ("test", test), ("external", external), ("synthetic", synth))}
    d = {k: np.array([r["ood"] for r in v]) for k, v in rows.items()}
    kinds = sorted({r["label"] for r in rows["synthetic"]})
    result = {
        "rules_version": RULES_VERSION,
        "model_version": svc.model_info().model_version,
        "reliability_levels": {"val": level_table(rows["val"]), "test": level_table(rows["test"])},
        "input_shift": {
            "threshold": svc.model_info().ood_threshold,
            "auroc_test_vs_external_mri": auroc(d["test"], d["external"]),
            "auroc_test_vs_synthetic_non_mri": auroc(d["test"], d["synthetic"]),
            "flag_rate": {k: float(np.mean([r["ood_flag"] for r in v])) for k, v in rows.items()},
            "flag_rate_by_synthetic_kind": {
                k: float(np.mean([r["ood_flag"] for r in rows["synthetic"] if r["label"] == k]))
                for k in kinds},
            "median_distance": {k: float(np.median(v)) for k, v in d.items()},
        },
        "level_counts_external": {lv: sum(r["level"] == lv for r in rows["external"])
                                  for lv in ("standard", "caution", "unreliable")},
        "n": {k: len(v) for k, v in rows.items()},
        "seconds": round(time.time() - t0),
    }
    out = ROOT / "results" / "metrics" / "reliability_eval.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
