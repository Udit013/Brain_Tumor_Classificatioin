#!/usr/bin/env bash
# Download the published training dataset and a SEPARATE external dataset for
# out-of-distribution validation. Requires Kaggle API credentials:
#   - place kaggle.json at ~/.kaggle/kaggle.json (chmod 600), OR
#   - export KAGGLE_USERNAME and KAGGLE_KEY
#
# Usage:  bash scripts/download_data.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR="${BTC_DATA_DIR:-$(pwd)/data}"
mkdir -p "$DATA_DIR" "$DATA_DIR/external"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "kaggle CLI not found. Activate the venv (source .venv/bin/activate) first."
  exit 1
fi

# ---- 1. Published compilation (training + in-distribution test) ----------
if [ ! -d "$DATA_DIR/brain-tumor-mri-dataset/Training" ]; then
  echo "Downloading published Kaggle dataset (masoudnickparvar/brain-tumor-mri-dataset)..."
  kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p "$DATA_DIR"
  unzip -q -o "$DATA_DIR/brain-tumor-mri-dataset.zip" -d "$DATA_DIR/brain-tumor-mri-dataset"
  rm -f "$DATA_DIR/brain-tumor-mri-dataset.zip"
else
  echo "Published dataset already present, skipping."
fi

# ---- 2. External OOD dataset (NOT part of the compilation) ----------------
# Navoneel Chakrabarty: "Brain MRI Images for Brain Tumor Detection" (binary).
if [ ! -d "$DATA_DIR/external/yes" ] && [ ! -d "$DATA_DIR/external/brain_tumor_dataset" ]; then
  echo "Downloading external OOD dataset (navoneel/brain-mri-images-for-brain-tumor-detection)..."
  kaggle datasets download -d navoneel/brain-mri-images-for-brain-tumor-detection -p "$DATA_DIR/external"
  unzip -q -o "$DATA_DIR/external/brain-mri-images-for-brain-tumor-detection.zip" -d "$DATA_DIR/external"
  rm -f "$DATA_DIR/external/brain-mri-images-for-brain-tumor-detection.zip"
  # Normalise layout: this dataset unzips to yes/ and no/ (sometimes nested).
  if [ -d "$DATA_DIR/external/brain_tumor_dataset" ]; then
    cp -r "$DATA_DIR/external/brain_tumor_dataset/"* "$DATA_DIR/external/" 2>/dev/null || true
  fi
else
  echo "External dataset already present, skipping."
fi

echo "Data ready under: $DATA_DIR"
