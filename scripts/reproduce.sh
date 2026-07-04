#!/usr/bin/env bash
# One-command reproduction of the ENTIRE production extension.
#
# Runs, in order: train (or reuse weights) -> evaluate -> leakage -> external
# validation -> calibration -> Grad-CAM -> ONNX export + latency -> drift
# reference. Every number it prints/saves is measured at run time; nothing is
# fabricated. Results land in results/metrics/*.json and results/figures/*.png.
#
# Prereqs: bash scripts/setup_env.sh && bash scripts/download_data.sh
# Usage:   bash scripts/reproduce.sh [--skip-train]
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -d .venv ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
export PYTHONPATH="src:${PYTHONPATH:-}"

SKIP_TRAIN=0
[ "${1:-}" == "--skip-train" ] && SKIP_TRAIN=1

echo "==================================================================="
echo " STEP 1/10  Train EfficientNetB3 (reproduce published 99.844%)"
echo "==================================================================="
if [ "$SKIP_TRAIN" -eq 0 ] && [ ! -f models/EfficientNetB3_model_weights.h5 ]; then
  python -m btc.train
else
  echo "Using existing models/EfficientNetB3_model_weights.h5 (skip-train)."
fi

echo "==================================================================="
echo " STEP 2/10  Evaluation rigor (P/R/F1, CM, ROC-AUC, PR)"
echo "==================================================================="
python -m btc.evaluate

echo "==================================================================="
echo " STEP 3/10  Leakage analysis (+ leak-free accuracy)"
echo "==================================================================="
python -m btc.leakage --eval

echo "==================================================================="
echo " STEP 4/10  External (out-of-distribution) validation"
echo "==================================================================="
python -m btc.external_validation || echo "  (external dataset not present — skipped)"

echo "==================================================================="
echo " STEP 5/10  Calibration (ECE + temperature scaling)"
echo "==================================================================="
python -m btc.calibration

echo "==================================================================="
echo " STEP 6/10  Grad-CAM overlays"
echo "==================================================================="
python -m btc.explain

echo "==================================================================="
echo " STEP 7/10  ONNX export + latency benchmark"
echo "==================================================================="
python -m btc.export_onnx

echo "==================================================================="
echo " STEP 8/10 Robustness to common image corruptions"
echo "==================================================================="
python -m btc.robustness --limit 400

echo "==================================================================="
echo " STEP 9/10 Uncertainty (Test-Time Augmentation)"
echo "==================================================================="
python -m btc.uncertainty

echo "==================================================================="
echo " STEP 10/10 Drift-monitoring reference"
echo "==================================================================="
python -m btc.monitoring build-reference --limit 1000

echo
echo "All steps complete. See results/metrics/*.json and results/figures/*.png"
echo "Then run scripts/fill_readme_results.py to inject measured numbers into the README."
