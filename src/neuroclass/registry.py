"""Model registry and promotion gate.

    python -m neuroclass.registry show
    python -m neuroclass.registry gate --comparison results/metrics/model_comparison.json

The registry (models/registry.json, tracked) lists every model version that
has been deployed: Hub revision, checksums, headline metrics, status. The gate
compares a candidate with the deployed model on the same one-time test
evaluation (scripts/evaluate_candidate.py -> scripts/compare_models.py) and
refuses promotion on any regression beyond the tolerances below, or when the
paired test shows the candidate significantly worse. It exits non-zero on
failure, so it can block a deploy step.

Provenance, stated plainly: the gate was written in November 2026, after the
October promotion, and applied to it retroactively. It is a contract for future
candidates, not evidence that the October decision was pre-registered. It does
not gate external no-tumour specificity, which that promotion lowered by one
image (98/98 -> 97/98): one image of 98 negatives is below what the external
set can resolve. Tolerances are deliberately asymmetric: the missed-tumour count
(glioma -> no tumour) may not rise at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from .settings import REPO_ROOT

REGISTRY_PATH = REPO_ROOT / "models" / "registry.json"


@dataclass(frozen=True)
class Rule:
    metric: str
    direction: str              # "higher" or "lower" is better
    tolerance: float            # allowed regression, in the metric's units
    why: str


RULES = (
    Rule("test_accuracy", "higher", 0.005, "overall accuracy may not drop by more than 0.5 pt"),
    Rule("novel_subset_accuracy", "higher", 0.005, "accuracy on images without a near-duplicate in training"),
    Rule("glioma_recall", "higher", 0.01, "the weakest class may not get weaker by more than 1 pt"),
    Rule("glioma_to_notumor", "lower", 0, "missed tumours (glioma called no tumour) may not increase"),
    Rule("macro_f1", "higher", 0.005, "class-balanced quality"),
    Rule("ece_raw", "lower", 0.01, "calibration error may not rise by more than 0.01"),
    Rule("mean_corrupted_accuracy", "higher", 0.01, "robustness to the corruption benchmark"),
    Rule("external_accuracy", "higher", 0.01, "accuracy on the external MRI dataset"),
    Rule("external_tumor_recall", "higher", 0.01, "tumours found in the external dataset"),
)
SIGNIFICANCE_ALPHA = 0.05


def gate(comparison: dict) -> dict:
    """Evaluate the rules on a model_comparison.json payload (previous = deployed)."""
    base, cand = comparison["previous"], comparison["current"]
    checks = []
    for r in RULES:
        b, c = base[r.metric], cand[r.metric]
        delta = c - b
        regression = -delta if r.direction == "higher" else delta
        checks.append({"metric": r.metric, "deployed": b, "candidate": c, "delta": delta,
                       "allowed_regression": r.tolerance, "passed": regression <= r.tolerance + 1e-12,
                       "rule": r.why})
    sig = comparison.get("significance", {}).get("full_test")
    if sig:
        worse = sig["previous_only_correct"] > sig["current_only_correct"]
        checks.append({"metric": "paired_test_full", "deployed_only_correct": sig["previous_only_correct"],
                       "candidate_only_correct": sig["current_only_correct"],
                       "p_value": sig["exact_mcnemar_p"],
                       "passed": not (worse and sig["exact_mcnemar_p"] < SIGNIFICANCE_ALPHA),
                       "rule": f"candidate may not be significantly worse (exact McNemar, alpha {SIGNIFICANCE_ALPHA})"})
    return {"passed": all(c["passed"] for c in checks), "checks": checks}


def load_registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NeuroClass model registry")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("show")
    g = sub.add_parser("gate")
    g.add_argument("--comparison", type=Path, default=REPO_ROOT / "results/metrics/model_comparison.json")
    g.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.command == "show":
        reg = load_registry()
        for v in reg["versions"]:
            print(f"{v['version']:20s} {v['status']:9s} hub {v['hub_revision']:12s} "
                  f"test {v['metrics']['test_accuracy']:.4f}  novel {v['metrics']['novel_subset_accuracy']:.4f}")
        return 0
    result = gate(json.loads(args.comparison.read_text()))
    for c in result["checks"]:
        mark = "PASS" if c["passed"] else "FAIL"
        if "delta" in c:
            print(f"[{mark}] {c['metric']:26s} deployed {c['deployed']:.4f} -> candidate {c['candidate']:.4f}"
                  f" (allowed regression {c['allowed_regression']})")
        else:
            print(f"[{mark}] {c['metric']:26s} p={c['p_value']:.2e}")
    print("PROMOTION ALLOWED" if result["passed"] else "PROMOTION REFUSED")
    if args.out:
        args.out.write_text(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
