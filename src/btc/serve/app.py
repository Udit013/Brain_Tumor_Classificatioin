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

from .. import config

app = None  # set below after FastAPI import

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
    from fastapi import FastAPI, File, UploadFile, HTTPException
    from fastapi.responses import JSONResponse

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
            raw = await file.read()
            img = Image.open(io.BytesIO(raw)).convert("RGB").resize(config.IMG_SIZE)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

        x = img_to_array(img).astype("float32")  # [0,255], parity preprocessing
        t0 = time.perf_counter()
        raw_probs = _state["model"].predict(x[None, ...], verbose=0)[0]
        cal_probs = _apply_temperature(raw_probs)
        cls = int(np.argmax(cal_probs))
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Grad-CAM overlay for the predicted class.
        hm, _ = gradcam_heatmap(x, _state["model"], _state["grad_model"], class_idx=cls)
        ov = overlay(x, hm)
        buf = io.BytesIO()
        imageio.imwrite(buf, ov, format="png")
        gradcam_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        return JSONResponse({
            "class": config.CLASS_NAMES[cls],
            "class_index": cls,
            "calibrated_confidence": float(cal_probs[cls]),
            "raw_confidence": float(raw_probs[cls]),
            "is_calibrated": _state["calibrated"],
            "probabilities": {n: float(cal_probs[i]) for i, n in enumerate(config.CLASS_NAMES)},
            "gradcam_png_base64": gradcam_b64,
            "latency_ms": latency_ms,
        })

    return api


app = _build_app()
