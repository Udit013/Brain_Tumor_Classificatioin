"""Deterministic reliability assessment of one prediction.

No LLM is involved: every finding comes from a fixed rule over the inference
result and the model's *measured* evaluation (results/metrics), loaded at
runtime so the wording cannot drift from the numbers. Thresholds were fixed
before evaluation; scripts/eval_reliability.py reports how accuracy differs
across the resulting levels.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .schemas import InferenceResult, ReliabilityAssessment, ReliabilityFinding, Severity
from .settings import REPO_ROOT

RULES_VERSION = "2026.11-r1"
LOW_CONFIDENCE = 0.60
VERY_LOW_CONFIDENCE = 0.40
METRICS_DIR = REPO_ROOT / "results" / "metrics"


@lru_cache
def measured_facts(metrics_dir: Path = METRICS_DIR) -> dict:
    """Per-class test precision/recall and external results of the deployed model."""
    try:
        ev = json.loads((metrics_dir / "evaluation.json").read_text())
        ext = json.loads((metrics_dir / "external_validation.json").read_text())["binary_collapse"]
    except (OSError, KeyError, ValueError):
        return {}
    rep = ev["classification_report"]
    cm = ev["confusion_matrix"]
    names = ["glioma", "meningioma", "notumor", "pituitary"]
    tumor_rows = [cm[i] for i, n in enumerate(names) if n != "notumor"]
    missed = sum(row[names.index("notumor")] for row in tumor_rows)
    pred_notumor = sum(row[names.index("notumor")] for row in cm)
    ext_cm = ext["confusion_matrix"]          # [[tn, fp], [fn, tp]], 1 = tumor
    return {
        "test_accuracy": ev["accuracy"],
        "per_class": {n: {"precision": rep[n]["precision"], "recall": rep[n]["recall"]}
                      for n in names},
        "notumor_predictions_that_were_tumors": missed,
        "notumor_predictions": pred_notumor,
        "external_accuracy": ext["accuracy"],
        "external_tumor_recall": ext_cm[1][1] / max(1, sum(ext_cm[1])),
        "external_n": ext["n"],
    }


def assess(result: InferenceResult, facts: dict | None = None) -> ReliabilityAssessment:
    facts = measured_facts() if facts is None else facts
    findings: list[ReliabilityFinding] = []

    def add(code: str, severity: Severity, message: str) -> None:
        findings.append(ReliabilityFinding(code=code, severity=severity, message=message))

    q = result.input_quality
    if not q.grayscale_like:
        add("color_image", Severity.caution, "The image contains colour; the model was trained "
            "only on grayscale MRI slices, so the prediction may be meaningless.")
    for f in q.findings:
        code = f.split(":", 1)[0]
        if code in ("low_contrast", "unusual_aspect_ratio"):
            add(code, Severity.caution, f.split(":", 1)[1].strip().capitalize() + ".")
        elif code == "low_resolution":
            add(code, Severity.info, f.split(":", 1)[1].strip().capitalize() + ".")

    if result.ood_flag:
        add("input_shift", Severity.caution,
            f"The image's features are unusually far from the training data "
            f"(distance {result.ood_distance:.0f} > threshold {result.ood_threshold:.0f}). "
            "This is a heuristic warning, not a guarantee in either direction.")

    conf = result.calibrated_confidence
    if conf < VERY_LOW_CONFIDENCE:
        add("very_low_confidence", Severity.caution,
            f"Calibrated confidence is {conf:.0%}: the model is close to guessing between classes.")
    elif conf < LOW_CONFIDENCE:
        add("low_confidence", Severity.caution, f"Calibrated confidence is only {conf:.0%}.")

    half = result.tta_views // 2
    if result.tta_agreement <= half:
        add("unstable_prediction", Severity.caution,
            f"Only {result.tta_agreement} of {result.tta_views} slightly altered versions of the "
            "image give the same class.")
    elif result.tta_agreement < result.tta_views:
        add("partly_unstable", Severity.info,
            f"{result.tta_agreement} of {result.tta_views} slightly altered versions agree.")

    per_class = facts.get("per_class", {})
    cls = result.predicted_class
    if cls in per_class:
        p, r = per_class[cls]["precision"], per_class[cls]["recall"]
        if cls == "notumor" and facts.get("notumor_predictions"):
            n_wrong, n_pred = facts["notumor_predictions_that_were_tumors"], facts["notumor_predictions"]
            add("missed_tumor_risk", Severity.info,
                f"On the test set, {n_wrong} of {n_pred} 'no tumor' predictions were tumor images "
                f"(precision {p:.1%}).")
        elif cls == "glioma":
            add("class_limitation", Severity.info,
                f"Glioma is the model's weakest class: test recall {r:.1%}, precision {p:.1%}.")
        elif p < 0.9:
            add("class_limitation", Severity.info,
                f"Test precision for {cls} is {p:.1%}: other tumor types are often predicted as {cls}.")
    if "external_accuracy" in facts:
        add("external_generalization", Severity.info,
            f"On a separate public MRI dataset (n={facts['external_n']}) the model was right "
            f"{facts['external_accuracy']:.1%} of the time (tumor vs no tumor) and found "
            f"{facts['external_tumor_recall']:.1%} of tumors.")

    severities = {f.code: f.severity for f in findings}
    if (not q.grayscale_like or (result.ood_flag and conf < LOW_CONFIDENCE)
            or "very_low_confidence" in severities):
        level = "unreliable"
    elif any(s == Severity.caution for s in severities.values()):
        level = "caution"
    else:
        level = "standard"
    return ReliabilityAssessment(level=level, findings=findings,
                                 requires_review=level != "standard",
                                 rules_version=RULES_VERSION)
