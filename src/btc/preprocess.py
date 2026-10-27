"""Image preprocessing — the single definition used for serving AND evaluation.

The FastAPI service, the Hugging Face Space (space/logic.py, asserted
identical in tests/test_space.py), every evaluation loader (test set, external
set, drift reference, Grad-CAM) and btc.finetune all turn an image into model
input with these exact PIL calls, so every reported metric describes the
computation that is deployed.

History: evaluation used to resize with Keras ``load_img`` (nearest-neighbour,
as the published training generators do) while serving resized with PIL's
default (bicubic for RGB). Measured on the serving path, the parity model's
test accuracy was 90.25%, not the 91.25% then reported; evaluation now uses
this module. The parity *training* pipeline (btc.data.make_generators) still
uses the published Keras path, which is part of what it reproduces.

Output contract: float32, shape (256, 256, 3), RGB, raw [0, 255] — NOT divided
by 255, because EfficientNet rescales internally.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from . import config


def to_model_input(img: Image.Image) -> np.ndarray:
    """PIL image of any mode/size -> model input array."""
    return np.asarray(img.convert("RGB").resize(config.IMG_SIZE), dtype="float32")


def load_image(path) -> np.ndarray:
    """Read an image file and return the model input array."""
    with Image.open(path) as img:
        return to_model_input(img)


def load_images(paths) -> np.ndarray:
    """Stack many images into one (N, 256, 256, 3) float32 array."""
    return np.stack([load_image(p) for p in paths]) if len(paths) else \
        np.zeros((0, *config.IMG_SHAPE), dtype="float32")
