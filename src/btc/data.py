"""Data loading with exact parity to the published EfficientNetB3 pipeline.

The published notebook builds a dataframe of (filepath, label) from the
predefined Kaggle Training/ Testing folders and feeds it through a *bare*
``ImageDataGenerator()`` (no rescaling — EfficientNet performs its own input
normalisation internally) via ``flow_from_dataframe`` at 256x256 RGB.

This module reproduces that exactly and adds small, clearly-separated helpers
the extension needs (e.g. materialising the full test set as arrays for
calibration / ROC / Grad-CAM) without altering the parity path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def _scan_split(split_dir: Path) -> pd.DataFrame:
    """Build a (filepaths, label) dataframe from a class-subfolder directory.

    Mirrors the notebook's os.listdir loop. Labels are the folder names.
    """
    split_dir = Path(split_dir)
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Dataset split not found: {split_dir}\n"
            "Run scripts/download_data.sh (needs Kaggle credentials) or set "
            "BTC_DATA_DIR to the directory containing brain-tumor-mri-dataset/."
        )
    filepaths, labels = [], []
    for fold in sorted(split_dir.iterdir()):
        if not fold.is_dir():
            continue
        for file in fold.iterdir():
            if file.is_file():
                filepaths.append(str(file))
                labels.append(fold.name)
    df = pd.concat(
        [pd.Series(filepaths, name="filepaths"), pd.Series(labels, name="label")],
        axis=1,
    )
    return df


def build_dataframes() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df) over the predefined Kaggle split."""
    return _scan_split(config.TRAIN_DIR), _scan_split(config.TEST_DIR)


def make_generators(seed: int = config.SEED):
    """Recreate the published train/test generators (PARITY).

    Returns ``(train_generator, test_generator)``. Importing tensorflow is done
    lazily so that non-TF modules (e.g. leakage hashing) work without TF.
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_df, test_df = build_dataframes()

    # Honest reproducibility check: warn if the live dataset has drifted from
    # the published split sizes (it has, as of 2026 — see config.py).
    if len(train_df) != config.N_TRAIN_PUBLISHED or len(test_df) != config.N_TEST_PUBLISHED:
        import warnings

        warnings.warn(
            f"Dataset size differs from the published split: observed "
            f"{len(train_df)} train / {len(test_df)} test vs published "
            f"{config.N_TRAIN_PUBLISHED} / {config.N_TEST_PUBLISHED}. The live "
            f"Kaggle dataset was updated after publication; reproduced numbers "
            f"are therefore on a (slightly) different test set. Documented in "
            f"the README Evaluation Limitations.",
            stacklevel=2,
        )

    tr_gen = ImageDataGenerator()  # PARITY: no rescale
    ts_gen = ImageDataGenerator()  # PARITY

    train_generator = tr_gen.flow_from_dataframe(
        train_df,
        x_col="filepaths",
        y_col="label",
        target_size=config.IMG_SIZE,
        class_mode="categorical",
        color_mode="rgb",
        shuffle=True,
        batch_size=config.BATCH_SIZE,
        seed=seed,
    )
    test_generator = ts_gen.flow_from_dataframe(
        test_df,
        x_col="filepaths",
        y_col="label",
        target_size=config.IMG_SIZE,
        class_mode="categorical",
        color_mode="rgb",
        shuffle=False,  # PARITY: ordered so labels align with predictions
        batch_size=config.BATCH_SIZE,
    )

    # Defensive parity check: the published label->index map must hold.
    expected = {name: i for i, name in enumerate(config.CLASS_NAMES)}
    if train_generator.class_indices != expected:
        raise RuntimeError(
            "class_indices mismatch — parity broken.\n"
            f"  expected: {expected}\n"
            f"  got:      {train_generator.class_indices}"
        )
    return train_generator, test_generator


def load_test_arrays() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Materialise the full ordered test set as (X, y_int, filepaths).

    Used by calibration, ROC/PR and Grad-CAM, which need raw arrays rather than
    an exhausting generator. Preprocessing matches the parity path: 256x256 RGB,
    float32, NO rescaling (EfficientNet handles it internally).
    """
    from tensorflow.keras.preprocessing.image import load_img, img_to_array

    _, test_df = build_dataframes()
    xs, ys, paths = [], [], []
    label_to_idx = {name: i for i, name in enumerate(config.CLASS_NAMES)}
    for fp, label in zip(test_df["filepaths"], test_df["label"]):
        img = load_img(fp, target_size=config.IMG_SIZE, color_mode="rgb")
        xs.append(img_to_array(img))  # float32, range [0,255]
        ys.append(label_to_idx[label])
        paths.append(fp)
    return np.asarray(xs, dtype="float32"), np.asarray(ys, dtype="int64"), paths
