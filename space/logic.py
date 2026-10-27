"""Pure logic for the NeuroClass Space — no TensorFlow, ONNX Runtime or Gradio.

Kept separate from app.py so it can be unit-tested in CI (tests/test_space.py)
and so the preprocessing provably matches btc.preprocess, the function the
FastAPI service and the evaluation pipeline use.
"""

from __future__ import annotations

import numpy as np

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No tumor",
    "pituitary": "Pituitary tumor",
}
# One neutral sentence per class: what the label means, never advice.
CLASS_BLURB = {
    "glioma": "a tumor arising from glial cells within the brain tissue.",
    "meningioma": "a tumor arising from the meninges, the membranes around the brain.",
    "notumor": "no tumor of the three trained types was detected in this slice.",
    "pituitary": "a tumor of the pituitary gland at the base of the brain.",
}
IMG_SIZE = (256, 256)
MIN_SIDE = 32           # smaller uploads carry too little information to resize up
N_TTA_VIEWS = 6


class InputError(ValueError):
    """An upload the model cannot score; the message is shown to the user."""


def preprocess(image: np.ndarray) -> np.ndarray:
    """Gradio numpy image -> (256, 256, 3) float32 in [0, 255].

    Same operations as btc.preprocess.to_model_input (PIL RGB conversion, PIL
    default resize); tests/test_space.py asserts the two stay identical.
    """
    from PIL import Image

    if image is None:
        raise InputError("Please upload an MRI image first.")
    arr = np.asarray(image)
    if arr.ndim not in (2, 3) or (arr.ndim == 3 and arr.shape[2] not in (1, 3, 4)):
        raise InputError(f"Unsupported image layout {arr.shape}; upload a JPG or PNG.")
    if min(arr.shape[:2]) < MIN_SIDE:
        raise InputError(f"Image is too small ({arr.shape[1]}×{arr.shape[0]} px); "
                         f"upload one at least {MIN_SIDE}×{MIN_SIDE} px.")
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr).convert("RGB").resize(IMG_SIZE)
    return np.asarray(img, dtype="float32")


def apply_temperature(p: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature-scale a probability vector (softmax of log p / T)."""
    if temperature == 1.0:
        return p
    logits = np.log(np.clip(p, 1e-12, 1.0)) / temperature
    logits -= logits.max()
    e = np.exp(logits)
    return e / e.sum()


def tta_views(x: np.ndarray) -> np.ndarray:
    """The 6 test-time-augmentation views: identity, flip, ±8° rotation, ×0.9/×1.1."""
    import cv2

    h, w = x.shape[:2]
    views = [x, x[:, ::-1, :]]
    for angle in (-8, 8):
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        views.append(cv2.warpAffine(x, m, (w, h), borderMode=cv2.BORDER_REFLECT))
    for f in (0.9, 1.1):
        views.append(np.clip(x * f, 0, 255).astype("float32"))
    return np.stack(views).astype("float32")


def entropy(p: np.ndarray) -> float:
    return float(-(p * np.log(np.clip(p, 1e-12, 1.0))).sum())


def stability(tta_probs: np.ndarray, cls: int, confidence: float) -> tuple[str, str]:
    """Plain-language stability level of a prediction.

    A fixed heuristic, chosen before looking at any evaluation data: how many of
    the 6 slightly altered views keep the same answer, and how confident the
    calibrated model is. It flags fragile predictions; a "high" level is not
    evidence that the prediction is correct.
    """
    agree = int((tta_probs.argmax(axis=1) == cls).sum())
    n = len(tta_probs)
    if agree == n and confidence >= 0.90:
        level = "High"
    elif agree <= n // 2 or confidence < 0.60:
        level = "Low"
    else:
        level = "Moderate"
    return level, f"{agree} of {n} slightly altered versions of the image gave the same answer"


def summary_markdown(cls: int, confidence: float, level: str, reason: str) -> str:
    name = CLASS_NAMES[cls]
    caution = ("" if level == "High" else
               "\n\n> The answer changes or weakens when the image is slightly "
               "altered — treat it as unreliable.")
    return (
        f"## {DISPLAY_NAMES[name]}\n"
        f"The model's best match is **{DISPLAY_NAMES[name].lower()}** — "
        f"{CLASS_BLURB[name]}\n\n"
        f"- **Calibrated confidence:** {confidence * 100:.0f}%\n"
        f"- **Stability:** {level} — {reason}"
        f"{caution}"
    )


def technical_markdown(raw: np.ndarray, cal: np.ndarray, temperature: float,
                       tta_probs: np.ndarray, latency_ms: float,
                       cam_ok: bool) -> str:
    rows = "\n".join(
        f"| {DISPLAY_NAMES[c]} | {raw[i] * 100:.1f}% | {cal[i] * 100:.1f}% | "
        f"{tta_probs[:, i].mean() * 100:.1f}% |"
        for i, c in enumerate(CLASS_NAMES))
    return (
        "| Class | Raw softmax | Calibrated | Mean over 6 TTA views |\n"
        "|---|---|---|---|\n" + rows + "\n\n"
        f"- Temperature T = {temperature:.3f} (T < 1 sharpens the raw probabilities, T > 1 softens them)\n"
        f"- Predictive entropy over TTA views: {entropy(tta_probs.mean(0)):.3f} "
        f"(0 = certain, {np.log(4):.2f} = uniform)\n"
        f"- ONNX Runtime CPU inference: {latency_ms:.0f} ms"
        + ("" if cam_ok else "\n- Grad-CAM could not be computed for this image.")
    )
