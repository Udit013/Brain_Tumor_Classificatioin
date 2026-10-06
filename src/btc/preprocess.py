"""Image preprocessing shared by serving and the train/val tooling.

The FastAPI service (and, inline, the Hugging Face Space) turn uploads into
model input with these exact PIL calls; btc.train_v2 uses them too, so a
model trained through train_v2 sees exactly what it is served.

KNOWN SKEW (measured, documented, not yet fixed in evaluation): the parity
pipeline — the published training generators and the evaluation loaders in
data.py — resizes with Keras ``load_img`` (nearest-neighbour), while serving
resizes with PIL's default (bicubic for RGB). Under the serving preprocessing
the deployed parity model scores 90.25% on the test set, versus the 91.25%
reported from the evaluation preprocessing. The reported metrics are kept on
the committed evaluation path so they stay reproducible; moving evaluation
onto this function is on the roadmap.

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
