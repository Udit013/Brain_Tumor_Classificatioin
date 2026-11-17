"""Upload validation: bounded decoding and input-quality findings.

Everything here runs before the model. Rejections raise `InputRejected` with a
message safe to show users; quality findings are returned, never hidden.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, UnidentifiedImageError

from ..schemas import InputQuality

ALLOWED_FORMATS = {"JPEG", "PNG"}
MIN_SIDE = 32
SMALL_SIDE = 128


class InputRejected(ValueError):
    """The upload cannot be scored; `str(exc)` is a user-facing reason."""

    def __init__(self, message: str, code: str):
        super().__init__(message)
        self.code = code


def decode_image(raw: bytes, max_bytes: int, max_pixels: int) -> Image.Image:
    """Decode an upload with size, format and decompression-bomb limits."""
    if not raw:
        raise InputRejected("The file is empty.", "empty_file")
    if len(raw) > max_bytes:
        raise InputRejected(f"The file exceeds the {max_bytes // (1024 * 1024)} MB limit.",
                            "file_too_large")
    try:
        with Image.open(io.BytesIO(raw)) as probe:
            fmt = probe.format
            width, height = probe.size
            if fmt not in ALLOWED_FORMATS:
                raise InputRejected(f"Unsupported format {fmt or 'unknown'}; upload a JPEG or PNG.",
                                    "unsupported_format")
            if width * height > max_pixels:
                raise InputRejected("The image has too many pixels to process safely.",
                                    "too_many_pixels")
            probe.verify()                       # structural check, no full decode
        img = Image.open(io.BytesIO(raw))
        img.load()                               # full decode, catches truncation
    except InputRejected:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise InputRejected("The file could not be read as an image.", "corrupt_image") from exc
    if min(img.size) < MIN_SIDE:
        raise InputRejected(f"The image is too small ({img.size[0]}x{img.size[1]} px); "
                            f"the minimum is {MIN_SIDE}x{MIN_SIDE}.", "too_small")
    return img


def assess_quality(img: Image.Image) -> InputQuality:
    """Cheap, explainable checks on the decoded image (before resizing)."""
    rgb = np.asarray(img.convert("RGB"), dtype="float32")
    channel_spread = float(np.abs(rgb - rgb.mean(axis=2, keepdims=True)).mean())
    gray = rgb.mean(axis=2)
    findings = []
    grayscale_like = channel_spread < 3.0
    if not grayscale_like:
        findings.append("color_image: MRI slices are grayscale; color content suggests a "
                        "photo, screenshot or annotated image the model was not trained on")
    if gray.std() < 8.0:
        findings.append("low_contrast: the image is nearly uniform")
    w, h = img.size
    if min(w, h) < SMALL_SIDE:
        findings.append(f"low_resolution: shorter side {min(w, h)} px; the model "
                        "upsamples to 256 px")
    if max(w, h) / min(w, h) > 2.0:
        findings.append("unusual_aspect_ratio: the image will be distorted to 256x256")
    return InputQuality(width=w, height=h, mode=img.mode, format=img.format,
                        grayscale_like=grayscale_like,
                        mean_intensity=round(float(gray.mean()), 2),
                        std_intensity=round(float(gray.std()), 2), findings=findings)
