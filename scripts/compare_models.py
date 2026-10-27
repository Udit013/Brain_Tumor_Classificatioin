"""Summarise the upgrade comparison (previous parity model vs current model).

Reads the one-time evaluation outputs archived under models/archive/ (written by
scripts/evaluate_candidate.py and the robustness/calibration runs) and writes
results/metrics/model_comparison.json, the source of the comparison tables in
README.md and MODEL_CARD.md. Significance: exact McNemar test on paired test
predictions.

Usage:
    python scripts/compare_models.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from btc import config  # noqa: E402

RUN = ROOT / "models" / "archive" / "retrain-2026-10"
FINAL = RUN / "test_evaluation"
BASE = ROOT / "models" / "archive" / "parity-2024" / "evaluation"
PARITY_METRICS = ROOT / "results" / "archive" / "parity-2024" / "metrics"


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def _mcnemar(p_ok: np.ndarray, v_ok: np.ndarray) -> dict:
    from scipy.stats import binomtest

    a, b = int((p_ok & ~v_ok).sum()), int((~p_ok & v_ok).sum())
    return {"previous_only_correct": a, "current_only_correct": b,
            "exact_mcnemar_p": float(binomtest(b, a + b, 0.5).pvalue) if a + b else 1.0}


def main() -> None:
    from btc.data import build_dataframes
    from btc.leakage import PHASH_NEAR_DUP_THRESHOLD, analyse

    p, v = _load(FINAL / "parity_test.json"), _load(FINAL / "current_test.json")
    _, df = build_dataframes()
    y = np.array([config.CLASS_NAMES.index(lab) for lab in df["label"]])
    pp = np.load(FINAL / "parity_test.probs.npy").argmax(1)
    vp = np.load(FINAL / "current_test.probs.npy").argmax(1)
    _, nn = analyse(eval_clean_subset=False)
    novel = nn > PHASH_NEAR_DUP_THRESHOLD

    def side(r: dict, ext: dict, rob: dict, ece_cal: float, protocol: str) -> dict:
        u = r["uncalibrated"]
        return {
            "test_accuracy": u["accuracy"], "macro_f1": u["macro_f1"],
            "novel_subset_accuracy": r["novel_subset"]["accuracy"],
            "near_duplicate_subset_accuracy": r["near_duplicate_subset"]["accuracy"],
            "glioma_recall": u["glioma_recall"],
            "glioma_recall_novel_subset": r["novel_subset"]["glioma_recall"],
            "glioma_to_notumor": u["glioma_to_notumor"],
            "per_class": u["per_class"], "confusion_matrix": u["confusion_matrix"],
            "cross_entropy": u["cross_entropy"], "ece_raw": u["ece"],
            "ece_calibrated": ece_cal, "calibration_protocol": protocol,
            "mean_corrupted_accuracy": rob["mean_corruption_accuracy"],
            "robustness_by_corruption": rob["corruptions"],
            "external_accuracy": ext["accuracy_binary"],
            "external_tumor_recall": ext["tumor_recall"],
            "external_notumor_specificity": ext["notumor_specificity"],
        }

    meta = _load(RUN / "metadata.json")
    out = {
        "note": "One-time test-set comparison made before the upgrade; the current "
                "model was selected on the validation split only. CPU, serving "
                "preprocessing (btc.preprocess). previous = parity model deployed "
                "until 2026-10 (Hub tag parity-2024).",
        "n_test": int(len(y)), "n_novel": int(novel.sum()),
        "previous": side(p, _load(BASE / "external_plain.json"),
                         _load(PARITY_METRICS / "robustness.json"),
                       _load(BASE / "calibration_holdout50.json")["ece_after"],
                       "T fit on half the test set, ECE on the other half"),
        "current": side(v, _load(FINAL / "current_external.json"),
                        _load(FINAL / "current_test_robustness.json"), v["ece_calibrated"],
                   "T fit on the validation split, ECE on the full test set"),
        "current_validation_candidates": {k: _load(RUN / f"val_{k}.json")["uncalibrated"]["accuracy"]
                          for k in ("head", "head_tta", "best", "best_tta")},
        "current_validation_mean_corrupted": {
            k: _load(RUN / f"val_robustness_{k}.json")["mean_corruption_accuracy"]
            for k in ("head", "best")},
        "current_temperature": v["temperature"],
        "current_training": {k: meta.get(k) for k in (
            "recipe", "git_rev", "seed", "split_sha256", "device", "machine",
            "unfreeze_from", "finetune_lr", "head_lr", "feature_seconds", "stage1",
            "stage2_seconds_this_session")},
        "significance": {
            "full_test": _mcnemar(pp == y, vp == y),
            "novel_subset": _mcnemar((pp == y)[novel], (vp == y)[novel]),
            "glioma": _mcnemar((pp == y)[y == 0], (vp == y)[y == 0]),
        },
    }
    dest = config.METRICS_DIR / "model_comparison.json"
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["significance"], indent=2))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
