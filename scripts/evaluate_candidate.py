"""Evaluate a candidate model on CPU, on one split, optionally with TTA.

Protocol (enforced by how this script is meant to be used, see README):
  * --split val       model selection, TTA decision, temperature fitting
  * --split test      once, for the final candidate only (reports the novel /
                      near-duplicate subsets alongside the full test set)
  * --split external  out-of-distribution check (binary tumor / no tumor)

All inference is on CPU through btc.model.load_trained_model, with the serving
preprocessing (btc.preprocess). TTA uses exactly the Space's 6 views
(space/logic.tta_views) and averages their probabilities.

Usage:
    python scripts/evaluate_candidate.py --weights runs/<run>/best.weights.h5 \
        --split val [--tta] [--temperature T] [--out runs/<run>/eval_val.json]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from btc import config  # noqa: E402

_spec = importlib.util.spec_from_file_location("space_logic", ROOT / "space" / "logic.py")
space_logic = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(space_logic)


def _arrays(split: str):
    """-> (x, y, extra) for the requested split."""
    from btc.preprocess import load_images

    if split == "val":
        from btc.splits import load_split
        paths, labels = load_split()["val"]
        return load_images(paths), np.asarray(labels), {}
    if split == "test":
        from btc.data import load_test_arrays
        from btc.leakage import PHASH_NEAR_DUP_THRESHOLD, analyse
        x, y, _ = load_test_arrays()
        _, nn = analyse(eval_clean_subset=False)
        return x, y, {"near_dup": nn <= PHASH_NEAR_DUP_THRESHOLD}
    from btc.external_validation import _NEGATIVE_NAMES, _POSITIVE_NAMES, _scan_external
    items = [(p, n) for p, n in _scan_external(config.EXTERNAL_DIR)
             if n in _POSITIVE_NAMES | _NEGATIVE_NAMES]
    y = np.asarray([int(n in _POSITIVE_NAMES) for _, n in items])   # 1 = tumor
    return load_images([p for p, _ in items]), y, {}


def _predict(model, x: np.ndarray, tta: bool) -> np.ndarray:
    if not tta:
        return model.predict(x, batch_size=32, verbose=0)
    out = np.empty((len(x), config.NUM_CLASSES), dtype="float32")
    for i in range(0, len(x), 8):     # 8 images x 6 views = 48 per batch
        chunk = x[i:i + 8]
        views = np.concatenate([space_logic.tta_views(img) for img in chunk])
        p = model.predict(views, batch_size=48, verbose=0)
        out[i:i + len(chunk)] = p.reshape(len(chunk), space_logic.N_TTA_VIEWS, -1).mean(1)
    return out


def _multiclass(y: np.ndarray, p: np.ndarray) -> dict:
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    from btc.calibration import expected_calibration_error

    pred = p.argmax(1)
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, labels=range(4),
                                                       zero_division=0)
    cm = confusion_matrix(y, pred, labels=range(4))
    g, nt = config.CLASS_NAMES.index("glioma"), config.CLASS_NAMES.index("notumor")
    return {
        "n": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "cross_entropy": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()),
        "macro_f1": float(f1.mean()),
        "per_class": {c: {"precision": round(float(prec[i]), 4),
                          "recall": round(float(rec[i]), 4), "f1": round(float(f1[i]), 4)}
                      for i, c in enumerate(config.CLASS_NAMES)},
        "confusion_matrix": cm.tolist(),
        "glioma_recall": float(rec[g]),
        "glioma_to_notumor": int(cm[g, nt]),
        "ece": expected_calibration_error(p, y)[0],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=config.WEIGHTS_PATH)
    ap.add_argument("--split", choices=["val", "test", "external"], required=True)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--temperature", type=float, default=None,
                    help="apply a temperature fitted elsewhere (e.g. on val)")
    ap.add_argument("--fit-temperature", action="store_true",
                    help="fit T on this split (use on val only)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from btc.calibration import _fit_temperature, apply_temperature
    from btc.model import load_trained_model   # pins inference to CPU

    t0 = time.time()
    x, y, extra = _arrays(args.split)
    t_load = time.time() - t0
    model = load_trained_model(args.weights)
    t1 = time.time()
    p = _predict(model, x, args.tta)
    t_pred = time.time() - t1

    res = {"weights": str(args.weights), "split": args.split, "tta": args.tta,
           "device": "cpu", "load_seconds": round(t_load), "predict_seconds": round(t_pred)}
    if args.split == "external":
        # same collapse as btc.external_validation: any tumor class -> tumor
        pred = (p.argmax(1) != config.CLASS_NAMES.index("notumor")).astype(int)
        res.update({"n": int(len(y)), "accuracy_binary": float((pred == y).mean()),
                    "tumor_recall": float(pred[y == 1].mean()),
                    "notumor_specificity": float(1 - pred[y == 0].mean())})
    else:
        res["uncalibrated"] = _multiclass(y, p)
        if args.fit_temperature:
            res["temperature_fitted"] = float(_fit_temperature(p, y))
        T = res.get("temperature_fitted", args.temperature)
        if T is not None:
            res["temperature"] = T
            res["ece_calibrated"] = _multiclass(y, apply_temperature(p, T))["ece"]
        if "near_dup" in extra:
            m = extra["near_dup"]
            pred = p.argmax(1)
            res["near_duplicate_subset"] = {"n": int(m.sum()),
                                            "accuracy": float((pred[m] == y[m]).mean())}
            res["novel_subset"] = {"n": int((~m).sum()),
                                   "accuracy": float((pred[~m] == y[~m]).mean()),
                                   **{k: v for k, v in _multiclass(y[~m], p[~m]).items()
                                      if k in ("glioma_recall", "glioma_to_notumor",
                                               "macro_f1")}}
    print(json.dumps(res, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=2))
        np.save(args.out.with_suffix(".probs.npy"), p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
