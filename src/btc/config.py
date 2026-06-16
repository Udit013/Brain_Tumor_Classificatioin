"""Central configuration: paths, dataset constants, and parity hyper-parameters.

Every constant that the published EfficientNetB3 notebook hard-codes is mirrored
here so the reproduction stays faithful and there is a single source of truth.
Values flagged ``# PARITY`` must not be changed without breaking reproduction of
the published 99.844% result.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Filesystem layout
# --------------------------------------------------------------------------- #
# Repo root = two levels up from this file (src/btc/config.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]

# Data root can be overridden with BTC_DATA_DIR (used by Docker / CI).
DATA_DIR = Path(os.environ.get("BTC_DATA_DIR", REPO_ROOT / "data"))

# The Kaggle archive unzips to <DATA_DIR>/brain-tumor-mri-dataset/{Training,Testing}.
KAGGLE_DATASET = "masoudnickparvar/brain-tumor-mri-dataset"
DATASET_DIR = DATA_DIR / "brain-tumor-mri-dataset"
TRAIN_DIR = DATASET_DIR / "Training"
TEST_DIR = DATASET_DIR / "Testing"

# External (out-of-distribution) validation dataset — a SEPARATE public source
# that is NOT part of the Kaggle compilation. See src/btc/external_validation.py
# and scripts/download_data.sh for provenance.
EXTERNAL_DIR = Path(os.environ.get("BTC_EXTERNAL_DIR", DATA_DIR / "external"))

# Artifact locations.
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

# Trained weights produced by btc.train (mirrors the notebook's filename).
WEIGHTS_PATH = MODELS_DIR / "EfficientNetB3_model_weights.h5"
ONNX_PATH = MODELS_DIR / "efficientnetb3.onnx"
# Fitted temperature for calibration (single scalar saved as JSON).
TEMPERATURE_PATH = MODELS_DIR / "temperature.json"
# Reference embedding statistics for drift monitoring.
DRIFT_REFERENCE_PATH = MODELS_DIR / "drift_reference.npz"

# --------------------------------------------------------------------------- #
# Dataset constants  (PARITY)
# --------------------------------------------------------------------------- #
# flow_from_dataframe sorts labels alphabetically, which yields this exact order
# in the published notebook: {'glioma':0,'meningioma':1,'notumor':2,'pituitary':3}.
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]  # PARITY
NUM_CLASSES = len(CLASS_NAMES)

IMG_SIZE = (256, 256)          # PARITY  target_size
IMG_SHAPE = (256, 256, 3)      # PARITY  RGB, no grayscale for EfficientNetB3
BATCH_SIZE = 16                # PARITY  EfficientNetB3 notebook batch_size

# --- PUBLISHED split sizes (the paper / original notebooks) --------------- #
# IMPORTANT: the live Kaggle dataset has been UPDATED since publication. As
# downloaded on 2026-06-16 (Kaggle lastUpdated 2026-02-13) it contains 5600
# train (1400/class) and 1600 test (400/class) = 7200 images, whereas the paper
# used 5712 / 1311 = 7023. We keep the published numbers below for the parity
# *training schedule* (steps_per_epoch was derived from them) but evaluation
# always runs over the FULL observed test set, and data.py emits a warning if
# the observed counts differ. This drift is documented in the README's
# Evaluation Limitations — it is a real reproducibility caveat, not hidden.
N_TRAIN_PUBLISHED = 5712       # paper's train split size
N_TEST_PUBLISHED = 1311        # paper's test split size
N_TRAIN = N_TRAIN_PUBLISHED    # PARITY  (used only for the training schedule)
N_TEST = N_TEST_PUBLISHED      # PARITY

# --------------------------------------------------------------------------- #
# Training schedule  (PARITY with EfficientNetB3_Brain_Tumor.ipynb)
# --------------------------------------------------------------------------- #
EPOCHS = 20                    # PARITY
LEARNING_RATE = 0.001          # PARITY  Adamax(learning_rate=0.001)
# NOTE: the published notebook uses steps_per_epoch = 5712 // 32 = 178 while the
# batch size is 16. This means each epoch iterates 178 *batches of 16* (2848
# images) rather than the full training set. We preserve this EXACTLY to
# reproduce the published curve; it is documented as a faithful-reproduction
# quirk in the README, not "corrected".
STEPS_PER_EPOCH = N_TRAIN // 32        # PARITY  -> 178
VALIDATION_STEPS = N_TEST // 32        # PARITY  -> 40
EVAL_STEPS = N_TEST // 32              # PARITY  notebook evaluate(steps=...)

DROPOUT_SEED = 75              # PARITY  Dropout(seed=75)

# Regularisation constants for the dense head (PARITY).
L2_KERNEL = 0.016
L1_ACTIVITY = 0.006
L1_BIAS = 0.006

# --------------------------------------------------------------------------- #
# Published baseline (IEEE 2024) — for README/model-card attribution ONLY.
# These are NOT used as computed results anywhere; they are the numbers to
# *reproduce*, never to carry over into the production results table.
# --------------------------------------------------------------------------- #
PUBLISHED_ACCURACY = {
    "CNN": 0.98359,
    "VGG16": 0.99297,
    "InceptionV3": 0.97734,
    "EfficientNetB3": 0.99844,  # best
}
PRIMARY_MODEL = "EfficientNetB3"

# Random seed for the extension's own stochastic steps (NOT the published run).
SEED = 75


def ensure_dirs() -> None:
    """Create artifact directories if missing (safe to call repeatedly)."""
    for d in (MODELS_DIR, RESULTS_DIR, FIGURES_DIR, METRICS_DIR):
        d.mkdir(parents=True, exist_ok=True)
