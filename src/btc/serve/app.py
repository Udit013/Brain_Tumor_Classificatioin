"""Module 6 (part B) — FastAPI inference service.

POST /predict  (multipart image file) -> JSON:
    {
      "class": "glioma",
      "class_index": 0,
      "calibrated_confidence": 0.97,        # temperature-scaled top prob
      "raw_confidence": 0.99,               # uncalibrated top prob
      "probabilities": {...},               # calibrated per-class
      "gradcam_png_base64": "...",          # overlay for the predicted class
      "latency_ms": 42.1
    }

The model and (optional) fitted temperature are loaded once at startup. If the
temperature file is absent, calibrated == raw and a flag says so, rather than
silently pretending the output is calibrated.

Run:
    uvicorn btc.serve.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import base64
import io
import json
import time

import numpy as np

# NOTE: these MUST be imported at module scope, not inside _build_app().
# This module uses `from __future__ import annotations`, so the route handler's
# `file: UploadFile = File(...)` annotation is stored as the *string*
# "UploadFile". FastAPI/Pydantic resolves that string against this module's
# global namespace when building the request validator — if UploadFile is only
# a local inside _build_app(), resolution fails at request time with
# `PydanticUserError: ... is not fully defined`, turning every /predict call
# into a 500. Keeping the import here is what makes the annotation resolvable.
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from .. import config

# Hard cap on upload size: an unauthenticated public endpoint that reads an
# unbounded request body into memory (the previous behaviour) is a trivial
# memory-exhaustion DoS vector. 10 MB comfortably covers any real MRI JPEG/PNG.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

app = None  # set below by _build_app()

# Lazily-populated singletons.
_state: dict = {"model": None, "grad_model": None, "temperature": 1.0,
                "calibrated": False}


def _load():
    from ..model import load_trained_model, find_last_conv_layer_name
    from ..explain import _grad_model

    model = load_trained_model()
    _state["model"] = model
    _state["grad_model"] = _grad_model(model, find_last_conv_layer_name(model))
    if config.TEMPERATURE_PATH.exists():
        _state["temperature"] = float(json.loads(config.TEMPERATURE_PATH.read_text())["temperature"])
        _state["calibrated"] = True


def _apply_temperature(probs: np.ndarray) -> np.ndarray:
    T = _state["temperature"]
    if T == 1.0:
        return probs
    logits = np.log(np.clip(probs, 1e-12, 1.0)) / T
    logits -= logits.max()
    p = np.exp(logits)
    return p / p.sum()


def _build_app():
    api = FastAPI(title="Brain Tumor Classifier (EfficientNetB3)", version="0.1.0")

    @api.on_event("startup")
    def _startup():
        _load()

    @api.get("/health")
    def health():
        return {"status": "ok", "model_loaded": _state["model"] is not None,
                "calibrated": _state["calibrated"]}

    @api.post("/predict")
    async def predict(file: UploadFile = File(...)):
        from PIL import Image
        from tensorflow.keras.preprocessing.image import img_to_array
        from ..explain import gradcam_heatmap, overlay
        import imageio.v2 as imageio

        if _state["model"] is None:
            _load()
        try:
            # Read up to MAX_UPLOAD_BYTES+1: if that many bytes come back, the
            # upload exceeds the cap regardless of what Content-Length claimed
            # (a client can lie about or omit that header).
            raw = await file.read(MAX_UPLOAD_BYTES + 1)
            if len(raw) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Image exceeds the {MAX_UPLOAD_BYTES // (1024*1024)}MB upload limit.",
                )
            img = Image.open(io.BytesIO(raw)).convert("RGB").resize(config.IMG_SIZE)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

        x = img_to_array(img).astype("float32")  # [0,255], parity preprocessing
        t_start = time.perf_counter()
        raw_probs = _state["model"].predict(x[None, ...], verbose=0)[0]
        cal_probs = _apply_temperature(raw_probs)
        cls = int(np.argmax(cal_probs))
        inference_ms = (time.perf_counter() - t_start) * 1000.0

        # Grad-CAM overlay for the predicted class. This costs roughly an order
        # of magnitude more than the forward pass (an extra forward+backward
        # pass plus PNG encoding), so it is timed and reported separately —
        # reporting only the inference time as "latency" would understate the
        # real cost of a /predict call by ~10x.
        t_cam = time.perf_counter()
        hm, _ = gradcam_heatmap(x, _state["model"], _state["grad_model"], class_idx=cls)
        ov = overlay(x, hm)
        buf = io.BytesIO()
        imageio.imwrite(buf, ov, format="png")
        gradcam_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        gradcam_ms = (time.perf_counter() - t_cam) * 1000.0

        return JSONResponse({
            "class": config.CLASS_NAMES[cls],
            "class_index": cls,
            "calibrated_confidence": float(cal_probs[cls]),
            "raw_confidence": float(raw_probs[cls]),
            "is_calibrated": _state["calibrated"],
            "probabilities": {n: float(cal_probs[i]) for i, n in enumerate(config.CLASS_NAMES)},
            "gradcam_png_base64": gradcam_b64,
            # Server-side timing breakdown (excludes network transfer).
            "inference_ms": inference_ms,
            "gradcam_ms": gradcam_ms,
            "latency_ms": inference_ms + gradcam_ms,
        })

    return api


app = _build_app()
