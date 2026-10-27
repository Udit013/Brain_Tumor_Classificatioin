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

# The deployed model: produced by btc.finetune (selected on a leakage-safe
# validation split) and exported for the serving architecture. Every evaluation
# and serving path loads it; BTC_WEIGHTS overrides it.
DEPLOYED_WEIGHTS_PATH = MODELS_DIR / "EfficientNetB3_model_weights.h5"
WEIGHTS_PATH = Path(os.environ.get("BTC_WEIGHTS", DEPLOYED_WEIGHTS_PATH))
# Output of btc.train, the exact reproduction of the published recipe (the
# model deployed until 2026-10). Kept separate so reproducing the paper can
# never overwrite the deployed model; evaluate it with BTC_WEIGHTS=<this path>.
PARITY_WEIGHTS_PATH = MODELS_DIR / "EfficientNetB3_parity_weights.h5"
# Read-only backups of every deployed model generation (with SHA256SUMS).
ARCHIVE_DIR = MODELS_DIR / "archive"
ONNX_PATH = MODELS_DIR / "efficientnetb3.onnx"
# Fitted temperature for calibration (single scalar saved as JSON).
TEMPERATURE_PATH = Path(os.environ.get("BTC_TEMPERATURE", MODELS_DIR / "temperature.json"))
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


def disable_metal_graph_optimizer() -> None:
    """Work around a tensorflow-metal graph-mode defect (TF 2.15, plugin 1.1.0).

    On Apple Silicon, Metal *eager* execution matches CPU to ~4e-6, but Metal
    *graph* execution (tf.function, so also predict and fit) computes a
    different function: the parity model scores 93.81% on the test set through
    it versus 90.25% on CPU, with 3.6% of predictions flipped. Turning off
    grappler's meta-optimizer removes the forward-pass difference (max 1.2e-5,
    no flips); disabling individual built-in passes does not, which points at
    the plugin's own graph optimizer. Gradients still differ from CPU (median
    4.7% on per-variable gradient norms), so Metal training is not verified
    equivalent even with this applied. A no-op on CPU-only machines.
    """
    import tensorflow as tf

    if tf.config.list_physical_devices("GPU"):
        tf.config.optimizer.set_experimental_options({"disable_meta_optimizer": True})


def configure_inference_device() -> None:
    """Pin TensorFlow inference to CPU unless BTC_INFERENCE_DEVICE=gpu.

    WHY: CPU TensorFlow and ONNX Runtime agree with each other to 2.4e-5, and
    CPU is what the deployed Space (ONNX Runtime) and any Linux/Docker
    environment run, so every reported metric is measured on CPU. On Apple
    Silicon the Metal plugin's graph mode computes a different function (see
    disable_metal_graph_optimizer); with BTC_INFERENCE_DEVICE=gpu that
    workaround is applied instead of hiding the GPU.

    Must run before TensorFlow initialises its devices (i.e. before the first
    op); if called too late it warns rather than silently measuring on GPU.
    """
    if os.environ.get("BTC_INFERENCE_DEVICE", "cpu").lower() == "gpu":
        disable_metal_graph_optimizer()
        return
    import warnings

    import tensorflow as tf

    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError as exc:  # devices already initialised in this process
        warnings.warn(
            f"Could not pin inference to CPU ({exc}); results may come from a "
            "GPU backend and differ from the deployed CPU model.",
            stacklevel=2,
        )


def ensure_dirs() -> None:
    """Create artifact directories if missing (safe to call repeatedly)."""
    for d in (MODELS_DIR, RESULTS_DIR, FIGURES_DIR, METRICS_DIR):
        d.mkdir(parents=True, exist_ok=True)
